import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.optimize import OptimizeResult

from pr.allele_models import MODEL_IDS, objective_for_model
from pr.contact_model import aggregate_from_arrays, sphere_forward
from pr.paired_run import (
    FitConfig,
    PairedRunError,
    PairedStart,
    load_paired_start,
    plan_paired_runs,
    run_one_fit,
    save_paired_start,
    select_best_start,
    write_coordinates,
)


TEST_SEED = 20260914


def _one_mb_data():
    records = [
        (0, 0, 0, 1_000_000),
        (0, 1_000_000, 0, 2_000_000),
        (1, 0, 1, 1_000_000),
        (0, 0, 1, 0),
        (0, 1_000_000, 1, 1_000_000),
        (0, 2_000_000, 1, 2_000_000),
        (0, 0, 0, 0),
    ]
    ci, p1, cj, p2 = (np.asarray(part, dtype=np.int64) for part in zip(*records))
    return aggregate_from_arrays(
        ("chrA", "chrB"), (3_000_000, 3_000_000),
        ci, p1, cj, p2, 1_000_000,
    )


def _starts(data):
    rng = np.random.default_rng(TEST_SEED)
    return tuple(
        PairedStart(
            "seed-%d" % index,
            sphere_forward(rng.normal(0.0, 0.08, size=(2, data.n_loci, 3))),
            metadata={"assignment_seed": 100 + index, "native_seed": 200 + index},
        )
        for index in range(3)
    )


class PairedDesignTests(unittest.TestCase):
    def test_fit_config_requires_explicit_positive_budget(self):
        self.assertEqual(FitConfig(7, 23).as_dict()["maxiter"], 7)
        self.assertEqual(FitConfig(7, 23).as_dict()["maxfun"], 23)
        with self.assertRaises(ValueError):
            FitConfig(0, 23)
        with self.assertRaises(ValueError):
            FitConfig(7, 0)
        with self.assertRaises(ValueError):
            FitConfig(7, 23, gtol=-1.0)

    def test_plan_is_five_variants_by_three_shared_physical_starts(self):
        data = _one_mb_data()
        starts = _starts(data)
        plan = plan_paired_runs(data, starts)
        self.assertEqual(len(plan), 15)
        self.assertEqual({row["model_id"] for row in plan}, set(MODEL_IDS))
        self.assertEqual({row["start_id"] for row in plan}, {start.start_id for start in starts})
        self.assertTrue(all(row["fit_not_run"] for row in plan))
        hashes_by_model = {
            model_id: {row["initial_coordinate_sha256"] for row in plan
                       if row["model_id"] == model_id}
            for model_id in MODEL_IDS
        }
        self.assertEqual(len({tuple(sorted(values)) for values in hashes_by_model.values()}), 1)
        self.assertEqual({row["p_init"] for row in plan}, {0.75})

    def test_plan_rejects_non_one_mb_data_and_outside_shared_start(self):
        non_final = aggregate_from_arrays(
            ("chrA",), (30,), np.asarray([0]), np.asarray([0]),
            np.asarray([0]), np.asarray([10]), 10,
        )
        start = PairedStart("s", sphere_forward(np.zeros((2, non_final.n_loci, 3))))
        with self.assertRaisesRegex(PairedRunError, "1 Mb"):
            plan_paired_runs(non_final, (start,))

        data = _one_mb_data()
        outside = np.zeros((2, data.n_loci, 3), dtype=np.float64)
        outside[0, 0] = [1.01, 0.0, 0.0]
        with self.assertRaises(AssertionError):
            plan_paired_runs(data, (PairedStart("outside", outside),))

    def test_paired_start_round_trip_preserves_bytes_and_seed_metadata(self):
        data = _one_mb_data()
        start = _starts(data)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.npz"
            save_paired_start(path, start)
            loaded = load_paired_start(path)
        self.assertEqual(loaded.start_id, start.start_id)
        self.assertEqual(loaded.coordinate_sha256, start.coordinate_sha256)
        self.assertEqual(loaded.p_init, start.p_init)
        np.testing.assert_array_equal(loaded.coordinates, start.coordinates)
        self.assertEqual(dict(loaded.metadata), dict(start.metadata))

    def test_free_writer_keeps_outside_coordinate_and_bounded_writer_rejects(self):
        data = _one_mb_data()
        values = np.zeros((2, data.n_loci, 3), dtype=np.float64)
        values[0, 0] = [1.2, 0.0, 0.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "free.3dg"
            metadata = write_coordinates(path, data, "C2-free", values)
            text = path.read_text()
        self.assertEqual(metadata["serialization_clip"]["clipped_coordinates"], 0)
        self.assertAlmostEqual(metadata["max_radius"], 1.2, places=14)
        self.assertIn("1.2", text)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):
                write_coordinates(Path(directory) / "bounded.3dg", data, "C0", values)

    def test_mocked_map_runner_counts_initial_and_trial_evaluations(self):
        data = _one_mb_data()
        coordinates = np.zeros((2, data.n_loci, 3), dtype=np.float64)
        coordinates[0, 0] = [0.95, 0.0, 0.0]
        start = PairedStart("near-boundary", coordinates)

        def fake_minimize(fun, x0, method, jac, callback, options):
            self.assertEqual(method, "L-BFGS-B")
            self.assertTrue(jac)
            self.assertEqual(options["maxiter"], 1)
            value, _gradient = fun(x0)
            trial = np.asarray(x0, dtype=np.float64).copy()
            trial[1] += 1e-3
            trial_value, _trial_gradient = fun(trial)
            callback(trial)
            return OptimizeResult(
                x=trial, success=True, status=0, message="mocked optimizer",
                nit=1, nfev=2, njev=2, fun=trial_value,
            )

        with mock.patch("pr.paired_run.minimize", side_effect=fake_minimize):
            result = run_one_fit(
                data, start, "C2-map", FitConfig(maxiter=1, maxfun=10),
                block_size=2, repulsion_block_size=2,
            )
        self.assertEqual(result.nfev, 2)
        self.assertEqual(result.map_diagnostics["objective_eval_count"], 3)
        self.assertEqual(result.map_diagnostics["nonidentity_map_eval_count"], 3)
        self.assertTrue(result.map_diagnostics["line_search_probes_included"])

    def test_mocked_runner_keeps_direct_x_outside_ball_without_optimizer_call(self):
        data = _one_mb_data()
        start = _starts(data)[0]
        objective = objective_for_model(data, "C2-free", block_size=2,
                                        repulsion_block_size=2)
        final_theta = objective.pack(start.coordinates, p=start.p_init)
        final_theta[0] = 1.2
        fake = OptimizeResult(
            x=final_theta,
            success=True,
            status=0,
            message="mocked optimizer",
            nit=0,
            nfev=0,
            njev=0,
        )
        with mock.patch("pr.paired_run.minimize", return_value=fake) as minimize_call:
            result = run_one_fit(
                data, start, "C2-free", FitConfig(maxiter=1, maxfun=1),
                block_size=2, repulsion_block_size=2,
            )
        minimize_call.assert_called_once()
        self.assertEqual(minimize_call.call_args.kwargs["method"], "L-BFGS-B")
        self.assertEqual(minimize_call.call_args.kwargs["options"]["maxiter"], 1)
        self.assertEqual(minimize_call.call_args.kwargs["options"]["maxfun"], 1)
        self.assertAlmostEqual(result.coordinates[0, 0, 0], 1.2, places=14)
        self.assertEqual(result.map_diagnostics["physical_domain"], "finite_unbounded")

    def test_selection_uses_tie_tolerance_and_rejects_nonfinite(self):
        results = [
            SimpleNamespace(model_id="C0", start_id="z-start", count_nll_normalized=1.0),
            SimpleNamespace(model_id="C0", start_id="a-start", count_nll_normalized=1.0 + 0.5e-12),
            SimpleNamespace(model_id="C0", start_id="outside", count_nll_normalized=1.0 + 2e-12),
        ]
        self.assertEqual(select_best_start(results).start_id, "a-start")
        with self.assertRaisesRegex(PairedRunError, "nonfinite"):
            select_best_start(results + [
                SimpleNamespace(model_id="C0", start_id="nan", count_nll_normalized=np.nan),
            ])

        data = _one_mb_data()
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                objective = objective_for_model(data, model_id)
                self.assertEqual(objective.data.n_pairs,
                                 data.n_loci * (data.n_loci - 1) // 2)
                self.assertEqual(objective.data.diag_counts.shape, data.diag_counts.shape)
                self.assertEqual(objective.data.budget()["n_diag_bins"], data.n_loci)


if __name__ == "__main__":
    unittest.main()
