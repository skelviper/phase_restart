import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from pr import allele_models, contact_model, joint_fit, reconstruction_init, reconstruct
from pr import multires_variant_runner as runner

class MultiresVariantControllerTests(unittest.TestCase):
    def test_plan_is_exactly_24_remaining_attempts_with_frozen_budgets(self):
        rows = runner._planned_rows()
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({row["attempt_id"] for row in rows}), 24)
        self.assertEqual({row["variant"] for row in rows}, set(runner.VARIANTS))
        self.assertEqual({row["stage"] for row in rows}, {"5m", "2m", "1m"})
        expected = {"5m": (300, 930), "2m": (200, 630), "1m": (240, 750)}
        for row in rows:
            self.assertEqual((row["maxiter"], row["maxfun"]), expected[row["stage"]])

    def test_transfer_matches_frozen_warm_start_and_free_has_no_clip(self):
        result = runner._transfer_probe()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["bounded_max_abs_difference_vs_original"], 0.0)
        self.assertTrue(result["bounded_positions_equal"])
        self.assertEqual(result["free_outside_radius_preserved"], 1.2)
        self.assertFalse(result["free_clip_applied"])
        self.assertFalse(result["additional_center_rescale"])

    def test_c0_path_parity_covers_solver_hooks_writer_and_selection(self):
        result = runner._parity_probe()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["max_abs_difference"], 0.0)
        self.assertTrue(result["solver_kwargs_equal"])
        self.assertTrue(result["callback_and_checkpoint_hooks_exercised"])
        self.assertTrue(result["writer_bytes_equal"])
        self.assertTrue(result["selection_rule_equal"])

    def test_domain_probe_uses_correct_objective_and_free_writer(self):
        with tempfile.TemporaryDirectory(prefix="multires-domain-test-") as directory:
            result = runner._domain_probe(Path(directory))
        self.assertEqual(result["status"], "passed")
        self.assertLessEqual(result["c2_map_gradient"]["abs_difference"], runner.FD_ATOL)
        self.assertEqual(result["c2_free"]["evaluate_radius"], 1.2)
        self.assertTrue(result["c2_free"]["ever_physical_radius_gt1"])
        self.assertFalse(result["c2_free"]["serialization_clip"]["applied"])
        self.assertTrue(result["c3"]["non_exposure_arrays_equal"])
        self.assertEqual(result["c1"]["abs_difference"], 0.0)

    def test_c2_free_three_layer_chain_carries_raw_q_and_never_clips(self):
        data_by_bin = {
            10: runner._synthetic_data(10),
            5: runner._synthetic_data(5),
            2: runner._synthetic_data(2),
        }
        headers = (("chrA", 30), ("chrB", 30))
        context = reconstruct.RunContext(
            headers=headers, data_path="synthetic", data_sha256="0" * 64,
            cohort={"sample_id": "synthetic", "biological_samples": 8,
                    "raw_contacts": 8, "intra_contacts": 6, "inter_contacts": 2,
                    "snpfree_sha256": "0" * 64}, source_assets={}, baseline_assets=(),
            stages=(reconstruct.StageSpec("5m", 10, 1),
                    reconstruct.StageSpec("2m", 5, 1),
                    reconstruct.StageSpec("1m", 2, 1)), strict_p9016=False)
        candidate = runner.CANDIDATES[1]
        config = {"models": {"C2-free": runner._model_contract("C2-free", data_by_bin[2].n_loci)}}
        fit_calls = []

        def initialize(_candidate, names, lengths, bin_size):
            layout = reconstruction_init.full_grid_layout(names, lengths, bin_size)
            coords = np.zeros((2, layout["n_loci"], 3), dtype=np.float64)
            coords[0, 0] = [1.2, 0.0, 0.0]
            return {"coords": coords, "positions": layout["positions"],
                    "chromosome_index": layout["chromosome_index"], "names": tuple(names),
                    "header_lengths": tuple(lengths), "bin_size": bin_size,
                    "metadata": {"synthetic": True}}

        def fake_fit(objective, initial_y, **kwargs):
            fit_calls.append(kwargs.get("q_init"))
            theta = objective.pack(initial_y, p=kwargs["p_init"])
            q = 1.0986122886681098 if kwargs.get("q_init") is None else float(kwargs["q_init"])
            theta[-1] = q
            value, _gradient, components = objective.evaluate(theta, need_gradient=False)
            coords, p = objective.coordinates_and_p(theta)
            return joint_fit.JointFitResult(
                objective=objective, theta=theta, y=np.asarray(initial_y).copy(),
                coordinates=coords.copy(), p=p, fun=value, components=components,
                success=True, status=0, message="synthetic chain", nit=1, nfev=1,
                actual_nfev=1, scipy_nfev=1, njev=1, elapsed_seconds=0.0, history=[])

        with tempfile.TemporaryDirectory(prefix="multires-chain-test-") as directory:
            root = Path(directory)
            for subdir in ("coords", "checkpoints", "logs", "stages", "selected", "run_status/candidates"):
                (root / subdir).mkdir(parents=True, exist_ok=True)
            result = runner._run_chain(
                root, context, config, "C2-free", candidate,
                load_data=lambda bin_size: data_by_bin[bin_size],
                initialize=initialize, fit_callable=fake_fit)
            self.assertEqual(result["optimization"]["terminal_status"], "converged")
            self.assertEqual(fit_calls, [None, 1.0986122886681098, 1.0986122886681098])
            records = [json.loads((root / row["stage_record"]).read_text()) for row in result["attempts"]]
            self.assertEqual([record["fit"]["q_in"] for record in records], [None, 1.0986122886681098, 1.0986122886681098])
            self.assertEqual([record["fit"]["q_out"] for record in records],
                             [1.0986122886681098, 1.0986122886681098, 1.0986122886681098])
            final = np.asarray(result["final_coordinates_sha256"])
            self.assertEqual(len(str(final)), 64)
            data = data_by_bin[2]
            final_coordinates = runner._read_model_coordinates(
                root / result["final_coordinates_path"], data, "C2-free", result["final_coordinates_sha256"])
            self.assertEqual(float(np.linalg.norm(final_coordinates[0, 0])), 1.2)

    def test_real_scipy_c2_free_helper_matches_original_when_endpoint_stays_bounded(self):
        data = runner._synthetic_data(10)
        initial = np.random.default_rng(7).normal(
            0.0, 0.01, size=(2, data.n_loci, 3)).astype(np.float64)
        old_objective = allele_models.objective_for_model(
            data, "C2-free", block_size=3, repulsion_block_size=3)
        new_objective = allele_models.objective_for_model(
            data, "C2-free", block_size=3, repulsion_block_size=3)
        old_callbacks, new_callbacks = [], []
        old_checkpoints, new_checkpoints = [], []
        kwargs = {"p_init": 0.75, "q_init": 0.21, "maxiter": 3, "maxfun": 20,
                  "maxls": 20, "ftol": 1e-10, "gtol": 1e-6,
                  "checkpoint_every": 1}
        old = joint_fit.fit_joint(
            old_objective, initial, callback=old_callbacks.append,
            checkpoint_hook=old_checkpoints.append, **kwargs)
        new = runner._fit_free_domain_aware(
            new_objective, initial, callback=new_callbacks.append,
            checkpoint_hook=new_checkpoints.append, **kwargs)
        self.assertLess(float(np.linalg.norm(old.coordinates, axis=2).max()), 1.0)
        self.assertLess(float(np.linalg.norm(new.coordinates, axis=2).max()), 1.0)
        np.testing.assert_array_equal(old.theta, new.theta)
        np.testing.assert_array_equal(old.y, new.y)
        np.testing.assert_array_equal(old.coordinates, new.coordinates)
        self.assertEqual(old.components, new.components)
        self.assertEqual((old.success, old.status, old.message, old.nit, old.nfev,
                          old.actual_nfev, old.scipy_nfev, old.njev),
                         (new.success, new.status, new.message, new.nit, new.nfev,
                          new.actual_nfev, new.scipy_nfev, new.njev))

        def without_time(entry):
            return {key: value for key, value in entry.items() if key != "elapsed_seconds"}

        self.assertEqual([without_time(item) for item in old.history],
                         [without_time(item) for item in new.history])
        self.assertEqual([without_time(item) for item in old_callbacks],
                         [without_time(item) for item in new_callbacks])
        self.assertEqual(len(old_checkpoints), len(new_checkpoints))
        for old_checkpoint, new_checkpoint in zip(old_checkpoints, new_checkpoints):
            self.assertEqual((old_checkpoint.iteration, old_checkpoint.nfev,
                              old_checkpoint.fun, old_checkpoint.p, old_checkpoint.gradient_norm),
                             (new_checkpoint.iteration, new_checkpoint.nfev,
                              new_checkpoint.fun, new_checkpoint.p, new_checkpoint.gradient_norm))
            np.testing.assert_array_equal(old_checkpoint.theta, new_checkpoint.theta)
            np.testing.assert_array_equal(old_checkpoint.y, new_checkpoint.y)
            np.testing.assert_array_equal(old_checkpoint.coordinates, new_checkpoint.coordinates)
            self.assertEqual(old_checkpoint.components, new_checkpoint.components)

    def test_real_scipy_c2_free_helper_preserves_a_translated_outside_domain(self):
        data = runner._synthetic_data(10)
        base = np.random.default_rng(7).normal(
            0.0, 0.01, size=(2, data.n_loci, 3)).astype(np.float64)
        translated = base + np.asarray([1.5, 0.0, 0.0], dtype=np.float64)
        objective = allele_models.objective_for_model(
            data, "C2-free", block_size=3, repulsion_block_size=3)
        checkpoints = []
        result = runner._fit_free_domain_aware(
            objective, translated, p_init=0.75, q_init=0.21,
            maxiter=1, maxfun=2, maxls=20, ftol=1e-10, gtol=1e-6,
            checkpoint_every=1, checkpoint_hook=checkpoints.append)
        self.assertIsNotNone(result)
        self.assertGreater(float(np.linalg.norm(result.coordinates, axis=2).max()), 1.0)
        np.testing.assert_array_equal(result.coordinates, result.y)
        self.assertGreaterEqual(len(checkpoints), 1)
        self.assertTrue(all(float(np.linalg.norm(item.coordinates, axis=2).max()) > 1.0
                            for item in checkpoints))

        rows = [
            {"id": "consensus_joint", "count_nll_per_record": 2.0,
             "optimization": {"terminal_status": "not_converged"}},
            {"id": "random_joint", "count_nll_per_record": 2.0 + 0.5e-9,
             "optimization": {"terminal_status": "not_converged"}},
        ]
        self.assertEqual(runner.select_variant_candidate(rows), "consensus_joint")
        rows[1]["count_nll_per_record"] = 1.0
        self.assertEqual(runner.select_variant_candidate(rows), "random_joint")

if __name__ == "__main__":
    unittest.main()
