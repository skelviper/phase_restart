import json
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pr import contact_model, joint_fit, reconstruction_report
from pr import reconstruct
from pr.gate import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch"


class FakeOptimizer:
    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.q_inputs = []
        self.fail_on_call = fail_on_call

    def __call__(self, objective, initial_y, **kwargs):
        self.calls += 1
        self.q_inputs.append(kwargs["q_init"])
        if self.calls == self.fail_on_call:
            raise RuntimeError("synthetic optimizer failure")
        theta = objective.pack(initial_y, p=kwargs["p_init"])
        if kwargs["q_init"] is not None:
            theta[-1] = kwargs["q_init"]
        else:
            theta[-1] = 0.10 + 0.01 * self.calls
        coordinates, p = objective.coordinates_and_p(theta)
        total, _gradient, components = objective.evaluate(theta, need_gradient=False)
        entry = {
            "iteration": 10,
            "nfev": 10,
            "elapsed_seconds": 0.01,
            "fun": total,
            "p": p,
            "components": components,
        }
        checkpoint = joint_fit.JointCheckpoint(
            iteration=10,
            nfev=10,
            elapsed_seconds=0.01,
            fun=total,
            p=p,
            theta=theta.copy(),
            y=initial_y.copy(),
            coordinates=coordinates.copy(),
            components=dict(components),
        )
        kwargs["checkpoint_hook"](checkpoint)
        kwargs["callback"](entry)
        return SimpleNamespace(
            objective=objective,
            theta=theta.copy(),
            y=initial_y.copy(),
            coordinates=coordinates.copy(),
            p=p,
            fun=total,
            components=dict(components),
            success=True,
            status=0,
            message="synthetic convergence",
            nit=10,
            nfev=10,
            actual_nfev=10,
            scipy_nfev=10,
            njev=10,
            elapsed_seconds=0.01,
            history=[entry],
        )


class ReconstructionRunnerTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="reconstruct-runner-", dir=SCRATCH))
        self.headers = (("chrZ", 12_000_000), ("chrA", 12_000_000))
        self.data_path = self.root / "synthetic.snpfree.bytes"
        self.data_path.write_bytes(b"synthetic phase-free identity only\n")
        self.data_sha = sha256_file(self.data_path)
        self.records = (
            np.asarray([0, 0, 0, 1, 1, 0], dtype=np.int64),
            np.asarray([0, 1_000_000, 2_000_000, 1_000_000, 0, 3_000_000], dtype=np.int64),
            np.asarray([0, 0, 1, 1, 0, 0], dtype=np.int64),
            np.asarray([0, 6_000_000, 3_000_000, 7_000_000, 1_000_000, 3_000_000], dtype=np.int64),
        )
        self.template = contact_model.aggregate_from_arrays(
            tuple(name for name, _ in self.headers), tuple(length for _, length in self.headers),
            *self.records, 1_000_000,
        )
        audit = self.template.budget()
        self.cohort = {
            "sample_id": "synthetic-P9016-contract",
            "biological_samples": 1,
            "raw_contacts": int(audit["raw_records"]),
            "intra_contacts": int(audit["raw_same_bin"] + audit["raw_cis_offdiag"]),
            "inter_contacts": int(audit["raw_inter"]),
            "snpfree_sha256": self.data_sha,
        }
        self.context = self._context()

    def tearDown(self):
        shutil.rmtree(self.root)

    def _context(self):
        source_dir = self.root / "sources"
        source_dir.mkdir(exist_ok=True)
        source_assets = {}
        gate_rows = []
        for candidate, seed in (("consensus", 1103), ("random", 2207)):
            path = source_dir / (candidate + ".3dg")
            path.write_text("%s blind fixture\n" % candidate)
            digest = sha256_file(path)
            gate_rows.append({"stage": "blind", "tag": candidate, "path": str(path), "sha256": digest})
            source_assets[candidate] = {
                "tag": candidate,
                "stage": "blind",
                "path": str(path),
                "sha256": digest,
                "gate_path": str(source_dir / "gate.json"),
                "base_seed": seed,
            }
        gate_rows.append({"stage": "eval", "tag": "oracle", "path": "not-opened-oracle.3dg", "sha256": "c" * 64})
        gate = source_dir / "gate.json"
        gate.write_text(json.dumps(gate_rows) + "\n")
        gate_sha = sha256_file(gate)
        for source in source_assets.values():
            source["gate_sha256"] = gate_sha
        baselines = (
            {"id": "baseline_014_consensus", "tag": "consensus", "role": "blind_baseline"},
            {"id": "baseline_014_random", "tag": "random", "role": "blind_baseline"},
            {"id": "baseline_014_oracle", "tag": "oracle", "role": "evaluation_ceiling",
             "source_gate": {"path": str(gate), "sha256": gate_sha},
             "source_coordinate": {"path": "not-opened-oracle.3dg", "sha256": "c" * 64, "stage": "eval"}},
        )
        return reconstruct.RunContext(
            headers=self.headers,
            data_path=str(self.data_path),
            data_sha256=self.data_sha,
            cohort=self.cohort,
            source_assets=source_assets,
            baseline_assets=baselines,
            strict_p9016=False,
        )

    def _data(self, bin_size):
        return contact_model.aggregate_from_arrays(
            tuple(name for name, _ in self.headers), tuple(length for _, length in self.headers),
            *self.records, bin_size,
        )

    def _state(self, names, lengths, bin_size, candidate):
        layout = reconstruct.reconstruction_init.full_grid_layout(names, lengths, bin_size)
        coords = np.empty((2, layout["n_loci"], 3), dtype=float)
        for copy in (0, 1):
            for locus in range(layout["n_loci"]):
                coords[copy, locus] = (0.02 * (locus + 1), 0.025 * (copy + 1),
                                       0.01 * ((locus % 3) + 1))
        return {
            "coords": coords,
            "positions": layout["positions"],
            "chromosome_index": layout["chromosome_index"],
            "names": layout["names"],
            "header_lengths": layout["header_lengths"],
            "bin_size": layout["bin_size"],
            "metadata": {"synthetic_candidate": candidate, "full_grid_from_zero": True},
        }

    def _hooks(self, optimizer):
        def initialize(candidate, names, lengths, bin_size):
            return self._state(names, lengths, bin_size, candidate)

        def warm_start(_coords, _positions, _chromosome_index, names, lengths, bin_size, base_seed):
            return self._state(names, lengths, bin_size, "warm-%d" % base_seed)

        return reconstruct.RunnerHooks(self._data, initialize, warm_start, optimizer)

    def _run(self, name="formal", optimizer=None):
        optimizer = optimizer or FakeOptimizer()
        result = reconstruct.run_training(
            self.root / name, self.context, hooks=self._hooks(optimizer), workers=1, threads=1
        )
        return result, optimizer

    def test_synthetic_full_run_writes_frozen_layout_checkpoints_and_selection(self):
        result, optimizer = self._run()
        out = Path(result["outdir"])
        self.assertTrue((out / "config.json").is_file())
        self.assertTrue((out / "track_map.json").is_file())
        self.assertTrue((out / "selected.3dg").is_file())
        self.assertTrue((out / "provenance" / "training-code-manifest.json").is_file())
        self.assertTrue((out / "provenance" / "protocol-manifest.json").is_file())
        self.assertEqual(optimizer.q_inputs[0], None)
        self.assertAlmostEqual(optimizer.q_inputs[1], 0.11)
        self.assertAlmostEqual(optimizer.q_inputs[2], 0.11)
        self.assertEqual(optimizer.q_inputs[3], None)
        self.assertAlmostEqual(optimizer.q_inputs[4], 0.14)
        self.assertAlmostEqual(optimizer.q_inputs[5], 0.14)

        selection = json.loads((out / "selection.json").read_text())
        self.assertEqual(selection["schema_version"], "reconstruction-selection-v1")
        self.assertEqual(selection["status"], "training_complete")
        self.assertEqual(selection["objective_contract"]["criterion"], "count_nll_per_record")
        self.assertEqual(selection["coordinate_grid"]["origin_bp"], 0)
        self.assertTrue(selection["coordinate_grid"]["full_grid"])
        self.assertEqual(selection["coordinate_grid"]["expected_physical_beads"], 48)
        self.assertEqual(selection["frozen_provenance"]["code"]["hash_source"], "snapshot_manifest")
        self.assertEqual(selection["frozen_provenance"]["protocol"]["hash_source"], "file_bytes")
        self.assertEqual(selection["frozen_provenance"]["protocol"]["manifest_schema"],
                         reconstruct.PROTOCOL_MANIFEST_SCHEMA_VERSION)
        protocol_manifest = json.loads(
            (out / selection["frozen_provenance"]["protocol"]["path"]).read_text())
        protocol_member = next(row for row in protocol_manifest["files"]
                               if row["source_path"] == "docs/RECONSTRUCTION_V1_PROTOCOL.md")
        self.assertEqual(protocol_member["sha256"], reconstruct.FROZEN_PROTOCOL_SHA256)
        self.assertEqual(set(selection["frozen_provenance"]), {"protocol", "config", "code", "data"})
        self.assertEqual(len(selection["attempts"]), 6)
        self.assertEqual({row["attempt_id"] for row in selection["attempts"]}, {
            "consensus_joint-5m", "consensus_joint-2m", "consensus_joint-1m",
            "random_joint-5m", "random_joint-2m", "random_joint-1m",
        })
        for candidate in selection["candidates"]:
            terminal = [row for row in selection["attempts"]
                        if row["candidate_id"] == candidate["id"] and row["is_terminal"]]
            self.assertEqual(len(terminal), 1)
            self.assertEqual(terminal[0]["stage"], "1m")
            self.assertEqual(set(candidate["count_model"]), set(reconstruction_report.COUNT_MODEL_FIELDS))
            self.assertAlmostEqual(candidate["count_nll_per_record"],
                                   candidate["count_model"]["count_nll_normalized"], places=15)
            for layer in candidate["layers"]:
                stage = json.loads((out / layer["path"]).read_text())
                self.assertTrue(stage["data_budget"]["raw_conserved"])
                self.assertTrue(stage["data_budget"]["aggregate_conserved"])
                self.assertTrue(stage["data_budget"]["endpoint_conserved"])
                self.assertLessEqual(stage["fit"]["final_total"], stage["fit"]["initial_total"] + 1e-9)
                self.assertTrue((out / stage["initial_coordinates"]["path"]).is_file())
                self.assertTrue((out / stage["final_coordinates"]["path"]).is_file())
                checkpoint_path = out / stage["fit"]["checkpoint_paths"][0]
                self.assertTrue(checkpoint_path.is_file())
                with np.load(checkpoint_path) as checkpoint:
                    self.assertIn("fullhistory_json", checkpoint.files)
                    self.assertIn("coordinates", checkpoint.files)
                    self.assertIn("theta", checkpoint.files)
        reconstructed = reconstruct.verify_completed_training(out, strict_p9016=False)
        self.assertEqual(reconstructed.selected_id, selection["selection"]["selected_id"])

    def test_existing_output_and_frozen_input_model_and_source_conflicts_are_rejected(self):
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaisesRegex(FileExistsError, "refusing to reuse"):
            reconstruct.run_training(existing, self.context, hooks=self._hooks(FakeOptimizer()))
        with self.assertRaisesRegex(reconstruct.ReconstructionError, "input SHA256"):
            reconstruct.run_training(self.root / "wrong-input", replace(self.context, data_sha256="0" * 64),
                                      hooks=self._hooks(FakeOptimizer()))
        wrong_weights = dict(self.context.weights)
        wrong_weights["bend"] = 0.02
        with self.assertRaisesRegex(reconstruct.ReconstructionError, "weights are frozen"):
            reconstruct.run_training(self.root / "wrong-model", replace(self.context, weights=wrong_weights),
                                      hooks=self._hooks(FakeOptimizer()))
    def test_solver_status_distinguishes_iteration_function_limits_and_success(self):
        stage = reconstruct.StageSpec("unit", 1_000_000, 300)
        iteration_limit = SimpleNamespace(
            nit=300, nfev=303, actual_nfev=303, success=False, status=1,
            message="STOP: TOTAL NO. of ITERATIONS REACHED LIMIT",
        )
        self.assertEqual(reconstruct._solver_status(iteration_limit, stage),
                         ("not_converged", True, "budget_exhausted"))

        function_limit = SimpleNamespace(
            nit=42, nfev=1, actual_nfev=930, success=False, status=1,
            message="STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT",
        )
        self.assertEqual(reconstruct._solver_status(function_limit, stage),
                         ("not_converged", True, "budget_exhausted"))

        converged = SimpleNamespace(
            nit=42, nfev=43, actual_nfev=43, success=True, status=0,
            message="CONVERGENCE: REL_REDUCTION_OF_F_ <= FACTR*EPSMCH",
        )
        self.assertEqual(reconstruct._solver_status(converged, stage),
                         ("converged", False, "solver_reported_success"))

    def test_parallel_workers_request_spawn_context_before_submitting_candidates(self):
        calls = {}

        class FakeFuture:
            def __init__(self, value):
                self.value = value

            def result(self):
                return self.value

        class FakePool:
            def __init__(self, *args, **kwargs):
                calls["args"] = args
                calls["kwargs"] = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return False

            def submit(self, _function, _out_text, _context, candidate, _config):
                return FakeFuture({"id": candidate.candidate_id})

        config = {"model_contract": {"model_signature": "a" * 64,
                                     "prior_config_sha256": "b" * 64,
                                     "parameter_dimension": 1}}
        with mock.patch.object(reconstruct.multiprocessing, "get_context",
                               return_value=mock.sentinel.spawn) as get_context, \
             mock.patch.object(reconstruct, "ProcessPoolExecutor", FakePool), \
             mock.patch.object(reconstruct, "as_completed",
                                side_effect=lambda futures: futures):
            results = reconstruct._run_candidates(
                self.root / "spawn-check", self.context, config,
                reconstruct.DEFAULT_HOOKS, workers=2, use_processes=True)
        get_context.assert_called_once_with("spawn")
        self.assertEqual(calls["kwargs"], {"max_workers": 2, "mp_context": mock.sentinel.spawn})
        self.assertEqual([row["id"] for row in results], ["consensus_joint", "random_joint"])

    def test_dangling_existing_output_path_is_rejected(self):
        dangling = self.root / "dangling-output"
        dangling.symlink_to(self.root / "missing-output")
        with self.assertRaisesRegex(FileExistsError, "refusing to reuse"):
            reconstruct.run_training(dangling, self.context, hooks=self._hooks(FakeOptimizer()))

    def test_failure_is_journaled_without_losing_successful_candidate(self):
        result, _optimizer = self._run("partial-failure", FakeOptimizer(fail_on_call=5))
        out = Path(result["outdir"])
        selection = json.loads((out / "selection.json").read_text())
        random = next(row for row in selection["candidates"] if row["id"] == "random_joint")
        self.assertEqual(random["optimization"]["terminal_status"], "failed")
        self.assertIsNone(random["coordinates"])
        self.assertIsNone(random["count_nll_per_record"])
        terminal = [row for row in selection["attempts"]
                    if row["candidate_id"] == "random_joint" and row["is_terminal"]]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["stage"], "2m")
        self.assertEqual(terminal[0]["status"], "failed")
        journal = [json.loads(line) for line in (out / "run_status" / "attempts.jsonl").read_text().splitlines()]
        self.assertEqual(len(journal), len(selection["attempts"]))
        self.assertEqual([row["attempt_id"] for row in journal],
                         [row["attempt_id"] for row in selection["attempts"]])
        self.assertEqual(sum(row["is_terminal"] for row in journal), 2)
        self.assertTrue(any(row["candidate_id"] == "consensus_joint" for row in journal))

    def test_count_selection_ignores_total_and_uses_preregistered_tie_order(self):
        candidates = [
            {"id": "consensus_joint", "count_nll_per_record": 1.0,
             "optimization": {"terminal_status": "converged"}, "total": 1000.0},
            {"id": "random_joint", "count_nll_per_record": 1.0 + 5e-10,
             "optimization": {"terminal_status": "converged"}, "total": -1000.0},
        ]
        self.assertEqual(reconstruct.select_candidate(candidates, ["consensus_joint", "random_joint"]),
                         "consensus_joint")
        candidates[1]["count_nll_per_record"] = 0.9
        self.assertEqual(reconstruct.select_candidate(candidates, ["consensus_joint", "random_joint"]),
                         "random_joint")

    def test_pure_guard_never_calls_evaluation_loaders_and_checks_snapshot_members(self):
        with mock.patch.object(reconstruction_report, "guarded_evaluation_inputs",
                               side_effect=AssertionError("evaluation loader must not run")), \
             mock.patch.object(reconstruction_report, "_refeval",
                               side_effect=AssertionError("reference helper must not run")):
            result, _optimizer = self._run("pure-guard")
        out = Path(result["outdir"])
        snapshot = out / "provenance" / "training-code" / "pr" / "joint_fit.py"
        snapshot.write_text(snapshot.read_text() + "# modified after formal freeze\n")
        with self.assertRaises(reconstruction_report.SelectionValidationError):
            reconstruction_report.verify_training_complete(out / "selection.json", strict_p9016=False)
        protocol_snapshot = out / "provenance" / "protocol" / "docs" / "RECONSTRUCTION_V1_RUN.md"
        protocol_snapshot.write_text(protocol_snapshot.read_text() + "\nchanged\n")
        with self.assertRaisesRegex(reconstruct.ReconstructionError, "snapshot member hash mismatch"):
            reconstruct.verify_completed_training(out, strict_p9016=False)


if __name__ == "__main__":
    unittest.main()
