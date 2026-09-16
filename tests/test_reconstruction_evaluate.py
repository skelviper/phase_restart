import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from pr import reconstruction_evaluate as evaluate
from pr import reconstruction_report as report
from pr.gate import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch"


class ReconstructionEvaluationFixture(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="reconstruction-evaluate-", dir=SCRATCH))
        self.grid = report.FullGrid((
            report.Chromosome("chr1", 25_000_000),
            report.Chromosome("chr2", 25_000_000),
        ))
        self.structures = self._make_structures()
        self._write_fixture_files()
        self.selection_path = self._write_selection()
        self.baseline_specs = self._write_baselines()
        self.input_spec = self._write_evaluator_inputs()
        self.contacts = self._make_contacts()
        self.events = []

    def tearDown(self):
        shutil.rmtree(self.root)

    def _make_structures(self):
        positions = {
            chromosome.name: tuple(self.grid.positions(chromosome))
            for chromosome in self.grid.chromosomes
        }
        candidate = {}
        reference = {}
        for ci, chromosome in enumerate(self.grid.chromosomes):
            t = np.arange(len(positions[chromosome.name]), dtype=float)
            # 两条确定性的非共线轨迹都保持在 R=1 内部。
            mat = np.column_stack((
                0.22 * np.cos(t * 0.43) + 0.015 * ci,
                0.22 * np.sin(t * 0.43) + 0.01 * ci,
                0.006 * t + 0.005 * ci,
            ))
            pat = np.column_stack((
                0.20 * np.sin(t * 0.31 + 0.6) - 0.012 * ci,
                0.20 * np.cos(t * 0.31 + 0.6) + 0.012 * ci,
                0.006 * t + 0.01 + 0.004 * ci,
            ))
            # selected candidate 遵循固定 reference；random candidate 是其 gauge complement，
            # 用于触发 sign aliases。
            def track(points):
                return {
                    int(position): np.asarray(point, dtype=float)
                    for position, point in zip(positions[chromosome.name], points)
                }
            a = "c%02da" % (ci + 1)
            b = "c%02db" % (ci + 1)
            candidate[a] = track(mat)
            candidate[b] = track(pat)
            reference["%s(mat)" % chromosome.name] = track(mat)
            reference["%s(pat)" % chromosome.name] = track(pat)
        random = {track: dict(points) for track, points in candidate.items()}
        for ci in range(len(self.grid.chromosomes)):
            a = "c%02da" % (ci + 1)
            b = "c%02db" % (ci + 1)
            random[a], random[b] = candidate[b], candidate[a]
        consensus = {
            "c%02da" % (ci + 1): dict(candidate["c%02da" % (ci + 1)])
            for ci in range(len(self.grid.chromosomes))
        }
        return {"selected": candidate, "random": random, "consensus": consensus,
                "oracle": candidate, "reference": reference}

    def _write_coordinate_file(self, path, structures):
        with Path(path).open("w", encoding="utf-8") as handle:
            for track, points in structures.items():
                for position, xyz in sorted(points.items()):
                    handle.write("%s\t%d\t%.17g\t%.17g\t%.17g\n" %
                                 (track, position, xyz[0], xyz[1], xyz[2]))

    def _write_fixture_files(self):
        self.coordinate_paths = {}
        for candidate_id, key in (("consensus_joint", "selected"), ("random_joint", "random")):
            path = self.root / (candidate_id + ".3dg")
            self._write_coordinate_file(path, self.structures[key])
            self.coordinate_paths[candidate_id] = path
        for role in ("protocol", "config", "code", "data"):
            path = self.root / (role + ".freeze")
            path.write_text("synthetic immutable %s\n" % role, encoding="utf-8")
            setattr(self, "%s_freeze" % role, path)

    def _model_contract(self):
        model_signature = "a" * 64
        prior_sha = "b" * 64
        contract = {
            "criterion": "count_nll_per_record",
            "direction": "minimize",
            "conditional_count_objective": True,
            "includes_candidate_invariant_diagonal_layer": True,
            "includes_priors": False,
            "uses_all_contacts": True,
            "selection_is_label_free": True,
            "reference_used_for_selection": False,
            "phase_used_for_selection": False,
            "bin_size_bp": 1_000_000,
            "tie_tolerance_per_record": 1e-9,
            "tie_break": "preregistered_order",
            "count_model_fields": list(report.COUNT_MODEL_FIELDS),
            "prior_component_names": list(report.PRIOR_REPORT_FIELDS),
            "model_contract": {
                "model_signature": model_signature,
                "prior_config_sha256": prior_sha,
                "parameter_dimension": 601,
                "bin_size_bp": 1_000_000,
                "prior_component_names": list(report.PRIOR_REPORT_FIELDS),
            },
        }
        return model_signature, prior_sha, contract

    def _write_selection(self, random_failed=False):
        track_map = report.write_track_map(self.root / "track_map.json", self.grid)
        model_signature, prior_sha, contract = self._model_contract()

        def candidate_record(candidate_id, score):
            return {
                "id": candidate_id,
                "model_signature": model_signature,
                "prior_config_sha256": prior_sha,
                "parameter_dimension": 601,
                "coordinates": {
                    "path": self.coordinate_paths[candidate_id].name,
                    "sha256": sha256_file(self.coordinate_paths[candidate_id]),
                },
                "count_nll_per_record": score,
                "count_model": {
                    "count_nll_normalized": score,
                    "conditional_nll_raw": score * 10.0,
                    "diag_profiled_nll_raw": 0.1,
                    "count_nll_raw": score * 10.0,
                    "p": 0.63,
                    "bond": 1.2,
                    "repulsion": 0.8,
                    "bend": 0.4,
                    "p_prior": 0.02,
                    "total": score * 10.0 + 2.42,
                },
                "attempt_ids": [candidate_id + "-fit"],
                "optimization": {"terminal_status": "converged",
                                 "terminal_attempt_id": candidate_id + "-fit"},
            }

        selected = candidate_record("consensus_joint", 0.50)
        if random_failed:
            random = {
                "id": "random_joint",
                "model_signature": model_signature,
                "prior_config_sha256": prior_sha,
                "parameter_dimension": 601,
                "coordinates": None,
                "count_nll_per_record": None,
                "failure_reason": "synthetic optimizer failure",
                "attempt_ids": ["random_joint-fit"],
                "optimization": {"terminal_status": "failed",
                                 "terminal_attempt_id": "random_joint-fit"},
            }
        else:
            random = candidate_record("random_joint", 0.70)
        selection = {
            "schema_version": report.SELECTION_SCHEMA_VERSION,
            "status": "training_complete",
            "cohort": {
                "sample_id": "synthetic-P9016-evaluation",
                "biological_samples": 1,
                "raw_contacts": 12,
                "intra_contacts": 10,
                "inter_contacts": 2,
                "snpfree_sha256": sha256_file(self.data_freeze),
            },
            "coordinate_grid": {
                "bin_size_bp": 1_000_000,
                "origin_bp": 0,
                "full_grid": True,
                "nuclear_radius": 1.0,
                "coordinate_units": "dimensionless_R1",
                "chromosomes": [{"name": chromosome.name, "length_bp": chromosome.length_bp}
                                for chromosome in self.grid.chromosomes],
                "expected_loci": self.grid.n_loci,
                "expected_physical_beads": self.grid.n_physical_beads,
            },
            "track_map": {"path": track_map["path"].split("/")[-1], "sha256": track_map["sha256"]},
            "frozen_provenance": {
                role: {"path": getattr(self, "%s_freeze" % role).name,
                       "sha256": sha256_file(getattr(self, "%s_freeze" % role)),
                       "hash_source": "file_bytes"}
                for role in ("protocol", "config", "code", "data")
            },
            "objective_contract": contract,
            "preregistered_candidate_ids": ["consensus_joint", "random_joint"],
            "candidates": [selected, random],
            "attempts": [
                {"attempt_id": "consensus_joint-fit", "candidate_id": "consensus_joint",
                 "stage": "1m", "status": "converged", "is_terminal": True,
                 "budget_exhausted": False},
                {"attempt_id": "random_joint-fit", "candidate_id": "random_joint",
                 "stage": "1m", "status": "failed" if random_failed else "converged",
                 "is_terminal": True, "budget_exhausted": random_failed},
            ],
            "baseline_assets": [
                {"id": "baseline_014_consensus", "tag": "consensus", "role": "blind_baseline"},
                {"id": "baseline_014_random", "tag": "random", "role": "blind_baseline"},
                {"id": "baseline_014_oracle", "tag": "oracle", "role": "evaluation_ceiling"},
            ],
            "selection": {
                "selected_id": "consensus_joint",
                "rule": "minimize_count_nll_per_record",
                "reference_used": False,
                "phase_used": False,
                "all_attempts_accounted_for": True,
                "tie_break": "preregistered_order",
            },
        }
        path = self.root / "selection.json"
        path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        return path

    def _write_baselines(self):
        specs = {}
        for tag, role in (("consensus", "blind_baseline"),
                          ("random", "blind_baseline"),
                          ("oracle", "evaluation_ceiling")):
            path = self.root / ("baseline-%s.3dg" % tag)
            self._write_coordinate_file(path, self.structures[tag])
            gate = self.root / ("baseline-%s-gate.json" % tag)
            gate.write_text(json.dumps([{
                "stage": "source", "tag": tag, "path": path.name,
                "sha256": sha256_file(path),
            }]) + "\n", encoding="utf-8")
            specs[tag] = report.BaselineSpec(
                tag=tag, role=role,
                source_gate_path=str(gate), source_gate_sha256=sha256_file(gate),
                coordinates_path=str(path), coordinates_sha256=sha256_file(path),
            )
        return specs

    def _write_evaluator_inputs(self):
        raw = self.root / "raw-phase-bearing.pairs.gz"
        raw.write_bytes(b"synthetic raw evaluator fixture\n")
        reference = self.root / "reference.3dg.gz"
        reference.write_bytes(b"synthetic reference evaluator fixture\n")
        provenance = self.root / "017-provenance.json"
        provenance.write_text(json.dumps({
            "reference_3dg_sha256": sha256_file(reference), "source": "synthetic-017"
        }) + "\n", encoding="utf-8")
        return report.EvaluatorInputAuditSpec(
            raw_pairs_path=str(raw), reference_3dg_path=str(reference),
            reference_017_provenance_path=str(provenance),
            reference_017_provenance_sha256=sha256_file(provenance),
            expected_raw_pairs_sha256=sha256_file(raw), expected_record_count=12,
        )

    @staticmethod
    def _make_contacts():
        million = 1_000_000
        return {
            "ci": np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=np.int32),
            "p1": np.asarray([3, 4, 5, 6, 3, 3, 4, 5, 6, 0, 3, 4], dtype=np.int64) * million,
            "cj": np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
            "p2": np.asarray([4, 5, 6, 7, 3, 4, 5, 6, 7, 1, 3, 4], dtype=np.int64) * million,
            "names": ("chr1", "chr2"),
            "cis": np.asarray([True] * 10 + [False, False], dtype=bool),
        }

    def _callbacks(self):
        def candidate_loader(path):
            self.events.append("candidate:%s" % Path(path).name)
            return report.read_full_grid_coordinates(path, self.grid)

        def baseline_loader(tag, path):
            self.events.append("baseline:%s" % tag)
            return report._read_coordinate_rows(Path(path))

        def contacts_loader():
            self.events.append("contacts")
            return self.contacts

        def alignment(contacts, raw_path):
            self.events.append("alignment")
            self.assertEqual(len(contacts["ci"]), 12)
            self.assertEqual(raw_path, self.input_spec.raw_pairs_path)
            return {"aligned": True, "raw_records": 12, "contacts_records": 12,
                    "canonicalized_cis_endpoint_swaps": 1,
                    "fixture_numeric_comparison": True}

        def labels_loader(contacts, raw_path):
            self.events.append("labels")
            self.assertEqual(raw_path, self.input_spec.raw_pairs_path)
            return (
                np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8),
                np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int8),
            )

        def reference_loader(path):
            self.events.append("reference")
            self.assertEqual(path, self.input_spec.reference_3dg_path)
            return self.structures["reference"]

        return {
            "candidate_loader": candidate_loader,
            "baseline_loader": baseline_loader,
            "contacts_loader": contacts_loader,
            "alignment_verifier": alignment,
            "labels_loader": labels_loader,
            "reference_loader": reference_loader,
        }

    def _run(self, selection=None, output_name="evaluation"):
        return evaluate.run_final_evaluation(
            selection or self.selection_path,
            evaluator_input_spec=self.input_spec,
            baseline_specs=self.baseline_specs,
            output_dir=self.root / output_name,
            strict_p9016=False,
            synthetic_fixture=True,
            **self._callbacks(),
        )

    def test_iteration_budget_semantics_do_not_trust_legacy_false(self):
        flags = evaluate._termination_flags({
            "status": "not_converged",
            "budget_exhausted": False,
            "solver": {"nit": 300, "maxiter": 300, "nfev": 420, "maxfun": 930,
                        "message": "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"},
        })
        self.assertTrue(flags["iteration_limit_reached"])
        self.assertFalse(flags["function_limit_reached"])
        self.assertTrue(flags["budget_reached"])
        self.assertFalse(flags["legacy_budget_exhausted"])

    def test_function_budget_semantics_use_actual_nfev_and_real_stop_message(self):
        flags = evaluate._termination_flags({
            "status": "not_converged",
            "budget_exhausted": False,
            "solver": {
                "nit": 42, "maxiter": 300, "nfev": 1, "actual_nfev": 930, "maxfun": 930,
                "message": "STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT",
            },
        })
        self.assertFalse(flags["iteration_limit_reached"])
        self.assertTrue(flags["function_limit_reached"])
        self.assertTrue(flags["budget_reached"])
        self.assertFalse(flags["legacy_budget_exhausted"])

        raw = self.root / "alignment.pairs"
        raw.write_text(
            "#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\tphase0\tphase1\n"
            ".\tchr1\t5000000\tchr1\t3000000\t+\t+\t0\t1\n", encoding="utf-8"
        )
        contacts = {
            "ci": np.asarray([0], dtype=np.int32), "p1": np.asarray([3_000_000], dtype=np.int64),
            "cj": np.asarray([0], dtype=np.int32), "p2": np.asarray([5_000_000], dtype=np.int64),
            "names": ("chr1",),
        }
        result = evaluate.alignment_verifier(contacts, str(raw))
        self.assertTrue(result["aligned"])
        self.assertEqual(result["canonicalized_cis_endpoint_swaps"], 1)
        self.assertFalse(result["comparison"]["phase_fields_inspected"])

    def test_end_to_end_guard_metrics_signs_and_delivery(self):
        result = self._run()
        output = Path(result["outdir"])
        metrics = result["metrics"]
        self.assertEqual(self.events, [
            "candidate:consensus_joint.3dg", "candidate:random_joint.3dg",
            "baseline:consensus", "baseline:random", "baseline:oracle",
            "contacts", "alignment", "labels", "reference",
        ])
        self.assertEqual(metrics["selection"]["selected_id_frozen"], "consensus_joint")
        self.assertFalse(metrics["selection"]["reference_reselection_performed"])
        self.assertEqual(metrics["record_alignment"]["evidence"]["fixture_numeric_comparison"], True)
        self.assertEqual(metrics["record_alignment"]["evidence"]["canonicalized_cis_endpoint_swaps"], 1)
        self.assertEqual(len(metrics["evaluation_gate"]["entries"]), 5)
        self.assertTrue(metrics["evaluation_gate"]["all_coordinate_paths_registered_before_arm"])
        self.assertEqual(set(metrics["candidate_summary"]), {"consensus_joint", "random_joint"})
        self.assertEqual(set(metrics["candidate_vs_random"]), {"consensus_joint", "random_joint"})
        self.assertGreater(metrics["candidate_summary"]["consensus_joint"]["R1_chromosomes"], 0)
        self.assertGreater(metrics["candidate_summary"]["random_joint"]["R1_chromosomes"], 0)
        self.assertIsNotNone(metrics["candidate_summary"]["consensus_joint"]["R1_macro_mean"])
        self.assertIsNotNone(metrics["candidate_summary"]["random_joint"]["R1_macro_mean"])
        self.assertGreater(metrics["candidate_summary"]["consensus_joint"]["R3_status_counts"]["labelled"], 0)
        self.assertEqual(metrics["candidate_summary"]["consensus_joint"]["R3_fragments_tied"], 0)

        for chromosome, chromosome_row in metrics["per_chromosome"].items():
            self.assertEqual(chromosome_row["n_bins_metric_grid"], 22)
            for candidate_id, candidate_row in chromosome_row["candidates"].items():
                r1 = candidate_row["R1"]
                denominator = r1["paired_denominator"]["n_common"]
                self.assertGreater(denominator, 0)
                self.assertEqual(r1["selected"]["n_denominator"], denominator)
                self.assertEqual(r1["random"]["n_denominator"], denominator)
                self.assertEqual(r1["oracle"]["n_denominator"], denominator)
                self.assertEqual(r1["reference_ceiling"]["n_denominator"], denominator)
                self.assertEqual(r1["delta_candidate_minus_random"],
                                 r1["delta_selected_minus_random"])
                r2 = candidate_row["R2"]["selected_vs_random"]
                self.assertEqual(r2["delta_candidate_minus_random"],
                                 r2["delta_selected_minus_random"])
                r3 = candidate_row["R3"]["selected_vs_random"]
                self.assertEqual(r3["delta_candidate_minus_random"],
                                 r3["delta_selected_minus_random"])
                self.assertIsNotNone(candidate_row["R2"]["by_candidate"]["selected"]["contrast"])
                self.assertIsNotNone(candidate_row["R3"]["by_candidate"]["selected"]["frac_consistent"])

        expected_files = [
            output / "metrics" / "metrics.json", output / "metrics" / "summary.json",
            output / "metrics" / "per_chromosome.tsv", output / "track_map.json",
            output / "coordinates" / "selected.3dg",
            output / "coordinates" / "candidates" / "consensus_joint.3dg",
            output / "coordinates" / "candidates" / "random_joint.3dg",
            output / "plots" / "selected_structure_3d.png",
            output / "plots" / "selected_structure_3d.html",
            output / "plots" / "track_coverage.tsv",
            output / "plots" / "fragment_consistency_stripes.png",
            output / "plots" / "fragment_consistency_stripes.tsv",
            output / "plots" / "candidate_gallery.png",
            output / "plots" / "selected_distance_maps.png",
            output / "README.md", output / "config.json",
        ]
        for path in expected_files:
            self.assertTrue(path.is_file(), str(path))
        self.assertEqual(sha256_file(output / "coordinates" / "selected.3dg"),
                         sha256_file(self.coordinate_paths["consensus_joint"]))
        html = (output / "plots" / "selected_structure_3d.html").read_text(encoding="utf-8")
        self.assertIn("const STRUCTURE_DATA=", html)
        self.assertIn("rotatePoint", html)
        self.assertIn("export-raw", html)
        self.assertNotIn("http://", html.lower())
        self.assertNotIn("https://", html.lower())
        self.assertIn("coolwarm_r", json.loads((output / "metrics" / "metrics.json").read_text())["delivery"]["distance_maps"]["colormap"])
        self.assertGreater(metrics["delivery"]["fragment_stripes"]["n_fragments_max"], 1)
        self.assertIn('"labelled"', (output / "metrics" / "per_chromosome.tsv").read_text(encoding="utf-8"))
        self.assertIn("nuisance coefficient", (output / "README.md").read_text(encoding="utf-8"))
        self.assertIn("L2", (output / "README.md").read_text(encoding="utf-8"))
        snapshot = metrics["evaluation_code_snapshot"]
        self.assertEqual(snapshot["schema_version"], evaluate.EVALUATION_CODE_SNAPSHOT_SCHEMA_VERSION)
        self.assertTrue(any(row["source_path"] == "pr/reconstruction_evaluate.py"
                            for row in snapshot["files"]))
        self.assertTrue(any(row["source_path"] == "pr/reconstruction_report.py"
                            for row in snapshot["files"]))

    def test_failed_candidate_is_retained_without_coordinate_or_comparison(self):
        failed_selection = self._write_selection(random_failed=True)
        result = self._run(selection=failed_selection, output_name="evaluation-failed")
        metrics = result["metrics"]
        failed = metrics["candidate_metadata"]["random_joint"]
        self.assertEqual(failed["terminal_status"], "failed")
        self.assertEqual(failed["failure_reason"], "synthetic optimizer failure")
        self.assertIsNone(failed["coordinates"])
        self.assertIsNone(metrics["candidate_delivery"]["random_joint"]["coordinates"])
        self.assertEqual(set(metrics["candidate_vs_random"]), {"consensus_joint"})
        self.assertIsNone(metrics["candidate_summary"]["random_joint"]["comparison_vs_random"])
        tsv = Path(result["outdir"]) / "metrics" / "per_chromosome.tsv"
        self.assertIn("\tfailed\tsynthetic optimizer failure", tsv.read_text(encoding="utf-8"))

    def test_input_audit_failure_happens_before_any_evaluator_loader(self):
        bad_spec = report.EvaluatorInputAuditSpec(
            raw_pairs_path=self.input_spec.raw_pairs_path,
            reference_3dg_path=self.input_spec.reference_3dg_path,
            reference_017_provenance_path=self.input_spec.reference_017_provenance_path,
            reference_017_provenance_sha256=self.input_spec.reference_017_provenance_sha256,
            expected_raw_pairs_sha256="0" * 64,
            expected_record_count=12,
        )
        output = self.root / "evaluation-audit-failure"
        with self.assertRaises(report.SelectionValidationError):
            evaluate.run_final_evaluation(
                self.selection_path, evaluator_input_spec=bad_spec,
                baseline_specs=self.baseline_specs, output_dir=output,
                strict_p9016=False, synthetic_fixture=True, **self._callbacks(),
            )
        self.assertEqual(self.events, [])
        failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["status"], "evaluation_input_audit_failed")


if __name__ == "__main__":
    unittest.main()
