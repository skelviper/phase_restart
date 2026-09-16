import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from pr import reconstruction_report as report
from pr.gate import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch"


class ReconstructionReportFixture(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="reconstruction-report-", dir=SCRATCH))
        self.old_mplconfig = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = str(self.root / "mplcache")

    def tearDown(self):
        if self.old_mplconfig is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = self.old_mplconfig
        shutil.rmtree(self.root)

    def _grid20(self):
        chromosomes = tuple(report.Chromosome("chr%d" % (index + 1), 2_000_000)
                            for index in range(20))
        return report.FullGrid(chromosomes)

    def _write_coordinates(self, path, grid, shift=0.0, tracks=None):
        tracks = tracks if tracks is not None else report.track_map_for_grid(grid)
        with Path(path).open("w") as handle:
            for entry in tracks:
                chromosome_index = entry["chromosome_index"]
                copy_offset = 0.025 if entry["copy"] == "b" else 0.0
                for bin_index, position in enumerate(grid.positions(grid.chromosomes[chromosome_index])):
                    x = 0.02 * (chromosome_index + 1) + copy_offset + 0.005 * bin_index + shift
                    y = 0.015 * (chromosome_index + 1) - copy_offset + 0.01 * bin_index
                    z = 0.005 * (chromosome_index + 1) + 0.01 * bin_index
                    handle.write("%s\t%d\t%.8f\t%.8f\t%.8f\n" %
                                 (entry["track"], position, x, y, z))

    def _write_file(self, name, contents):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path

    def _selection_fixture(self, random_failed=False, code_manifest=False):
        grid = self._grid20()
        selected_path = self.root / "coords-selected.3dg"
        random_path = self.root / "coords-random.3dg"
        self._write_coordinates(selected_path, grid, shift=0.0)
        if not random_failed:
            self._write_coordinates(random_path, grid, shift=0.17)
        track_map = report.write_track_map(self.root / "track_map.json", grid)
        frozen = {}
        for role in ("protocol", "config", "code", "data"):
            path = self._write_file("%s.freeze" % role, "%s immutable fixture\n" % role)
            frozen[role] = {"path": path.name, "sha256": sha256_file(path), "hash_source": "file_bytes"}
        if code_manifest:
            snapshot = self._write_file("snapshots/training/joint_fit.py", "synthetic immutable training code\n")
            manifest = self.root / "training-code-manifest.json"
            manifest.write_text(json.dumps({
                "schema_version": report.TRAINING_CODE_MANIFEST_SCHEMA_VERSION,
                "files": [{"source_path": "pr/joint_fit.py", "snapshot_path": str(snapshot.relative_to(self.root)),
                           "sha256": sha256_file(snapshot)}],
            }, indent=2) + "\n")
            frozen["code"] = {"path": manifest.name, "sha256": sha256_file(manifest),
                              "hash_source": "snapshot_manifest"}
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
                "parameter_dimension": 240,
                "bin_size_bp": 1_000_000,
                "prior_component_names": list(report.PRIOR_REPORT_FIELDS),
            },
        }

        def complete_candidate(candidate_id, path, score, terminal_status="converged"):
            return {
                "id": candidate_id,
                "model_signature": model_signature,
                "prior_config_sha256": prior_sha,
                "parameter_dimension": 240,
                "coordinates": {"path": path.name, "sha256": sha256_file(path)},
                "count_nll_per_record": score,
                "count_model": {
                    "count_nll_normalized": score,
                    "conditional_nll_raw": score * 10.0 - 0.1,
                    "diag_profiled_nll_raw": 0.1,
                    "count_nll_raw": score * 10.0,
                    "p": 0.5,
                    "bond": 1.2,
                    "repulsion": 0.8,
                    "bend": 0.4,
                    "p_prior": 0.02,
                    "total": score * 10.0 + 2.42,
                },
                "attempt_ids": [candidate_id + "-fit"],
                "optimization": {"terminal_status": terminal_status,
                                 "terminal_attempt_id": candidate_id + "-fit"},
            }

        selected = complete_candidate("consensus_joint", selected_path, 0.50)
        if random_failed:
            random = {
                "id": "random_joint",
                "model_signature": model_signature,
                "prior_config_sha256": prior_sha,
                "parameter_dimension": 240,
                "coordinates": None,
                "count_nll_per_record": None,
                "failure_reason": "synthetic terminal failure",
                "attempt_ids": ["random_joint-fit"],
                "optimization": {"terminal_status": "failed", "terminal_attempt_id": "random_joint-fit"},
            }
        else:
            random = complete_candidate("random_joint", random_path, 0.50 + 5e-10,
                                        terminal_status="not_converged")
        candidates = [selected, random]
        attempts = [
            {"attempt_id": "consensus_joint-fit", "candidate_id": "consensus_joint",
             "stage": "joint_fit", "status": "converged", "is_terminal": True},
            {"attempt_id": "random_joint-fit", "candidate_id": "random_joint",
             "stage": "joint_fit", "status": random["optimization"]["terminal_status"], "is_terminal": True},
        ]
        selection = {
            "schema_version": report.SELECTION_SCHEMA_VERSION,
            "status": "training_complete",
            "cohort": {
                "sample_id": "synthetic-P9016-contract",
                "biological_samples": 1,
                "raw_contacts": 10,
                "intra_contacts": 6,
                "inter_contacts": 4,
                "snpfree_sha256": frozen["data"]["sha256"],
            },
            "coordinate_grid": {
                "bin_size_bp": 1_000_000,
                "origin_bp": 0,
                "full_grid": True,
                "nuclear_radius": 1.0,
                "coordinate_units": "dimensionless_R1",
                "chromosomes": [{"name": chromosome.name, "length_bp": chromosome.length_bp}
                                for chromosome in grid.chromosomes],
                "expected_loci": grid.n_loci,
                "expected_physical_beads": grid.n_physical_beads,
            },
            "track_map": {"path": Path(track_map["path"]).name, "sha256": track_map["sha256"]},
            "frozen_provenance": frozen,
            "objective_contract": contract,
            "preregistered_candidate_ids": ["consensus_joint", "random_joint"],
            "candidates": candidates,
            "attempts": attempts,
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
        selection_path = self.root / "selection.json"
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
        return grid, selection_path, selected_path, random_path

    def _baseline_specs(self, grid):
        specs = {}
        for tag, role in (("consensus", "blind_baseline"), ("random", "blind_baseline"),
                          ("oracle", "evaluation_ceiling")):
            if tag == "consensus":
                tracks = [entry for entry in report.track_map_for_grid(grid) if entry["copy"] == "a"]
            else:
                tracks = report.track_map_for_grid(grid)
            coordinates = self.root / ("baseline-%s.3dg" % tag)
            self._write_coordinates(coordinates, grid, shift={"consensus": 0.03, "random": 0.08,
                                                               "oracle": 0.12}[tag], tracks=tracks)
            gate = self.root / ("baseline-%s-gate.json" % tag)
            gate.write_text(json.dumps([{"stage": "source", "tag": tag,
                                         "path": coordinates.name,
                                         "sha256": sha256_file(coordinates)}]) + "\n")
            specs[tag] = report.BaselineSpec(
                tag=tag,
                role=role,
                source_gate_path=str(gate),
                source_gate_sha256=sha256_file(gate),
                coordinates_path=str(coordinates),
                coordinates_sha256=sha256_file(coordinates),
            )
        return specs

    def _evaluator_input_spec(self):
        raw = self._write_file("synthetic-raw.pairs", "".join("raw pair %d\n" % index for index in range(10)))
        reference = self._write_file("synthetic-reference.3dg", "reference coordinate fixture\n")
        provenance = self._write_file(
            "017-reference-provenance.json",
            json.dumps({"reference_3dg_sha256": sha256_file(reference), "source": "synthetic-017"}) + "\n")
        return report.EvaluatorInputAuditSpec(
            raw_pairs_path=str(raw),
            reference_3dg_path=str(reference),
            reference_017_provenance_path=str(provenance),
            reference_017_provenance_sha256=sha256_file(provenance),
            expected_raw_pairs_sha256=sha256_file(raw),
            expected_record_count=10,
        ), raw

    def test_selection_guard_checks_full_40_track_inventory_and_frozen_contract(self):
        grid, selection_path, _selected_path, _random_path = self._selection_fixture()
        verified = report.verify_training_complete(selection_path, strict_p9016=False)
        self.assertEqual(verified.selected_id, "consensus_joint")
        self.assertEqual(verified.grid.n_tracks, 40)
        self.assertEqual(verified.selected.inventory.n_tracks, 40)
        self.assertEqual(verified.selected.inventory.n_beads, grid.n_physical_beads)
        self.assertLessEqual(verified.selected.inventory.max_radius,
                             report.NUCLEAR_RADIUS + report.NUCLEAR_RADIUS_TOL)
        self.assertEqual(verified.candidates["random_joint"].terminal_status, "not_converged")
        self.assertEqual(report.track_map_for_grid(grid)[0]["track"], "c01a")
        self.assertEqual(report.track_map_for_grid(grid)[-1]["track"], "c20b")
        self.assertEqual(report.FROZEN_P9016["full_grid_loci"], 2645)
        self.assertEqual(report.FROZEN_P9016["full_grid_physical_beads"], 5290)

    def test_code_snapshot_manifest_checks_each_member(self):
        _grid, selection_path, _selected_path, _random_path = self._selection_fixture(code_manifest=True)
        verified = report.verify_training_complete(selection_path, strict_p9016=False)
        self.assertEqual(verified.frozen_provenance["code"]["member_count"], 1)
        snapshot = self.root / "snapshots" / "training" / "joint_fit.py"
        snapshot.write_text(snapshot.read_text() + "# modified after freeze\n")
        with self.assertRaises(report.SelectionValidationError):
            report.verify_training_complete(selection_path, strict_p9016=False)

    def test_provenance_data_protocol_and_config_require_file_byte_hashes(self):
        _grid, selection_path, _selected_path, _random_path = self._selection_fixture()
        document = json.loads(selection_path.read_text())
        for role in ("protocol", "config", "data"):
            document["frozen_provenance"][role]["hash_source"] = "snapshot_manifest"
            selection_path.write_text(json.dumps(document, indent=2) + "\n")
            with self.assertRaisesRegex(report.SelectionValidationError, "hash_source must be file_bytes"):
                report.verify_training_complete(selection_path, strict_p9016=False)
            document = json.loads(selection_path.read_text())
            document["frozen_provenance"][role]["hash_source"] = "file_bytes"

        _grid, selection_path, _selected_path, _random_path = self._selection_fixture()
        document = json.loads(selection_path.read_text())
        document["candidates"][1]["count_model"]["total"] = -1000.0
        selection_path.write_text(json.dumps(document, indent=2) + "\n")
        verified = report.verify_training_complete(selection_path, strict_p9016=False)
        self.assertEqual(verified.selected_id, "consensus_joint")

    def test_evaluator_input_versions_reject_modified_raw_before_loading(self):
        spec, raw = self._evaluator_input_spec()
        audit = report.verify_evaluator_input_versions(spec, strict_p9016=False)
        self.assertTrue(audit["reference_3dg"]["declared_in_017_provenance"])
        self.assertEqual(report.RAW_P9016_PAIRS_SHA256,
                         "071a6cc76bfad543ea1ace6ee1ce3022b30ac1b1e50a9f0c3a3a1967b9649505")
        self.assertEqual(audit["raw_pairs"]["expected_records"], 10)
        raw.write_text(raw.read_text() + "modified\n")
        with self.assertRaises(report.SelectionValidationError):
            report.verify_evaluator_input_versions(spec, strict_p9016=False)

    def test_final_guard_audits_versions_and_full_record_alignment(self):
        grid, selection_path, _selected_path, _random_path = self._selection_fixture()
        specs = self._baseline_specs(grid)
        input_spec, _raw = self._evaluator_input_spec()
        calls = []
        contacts = {"ci": np.arange(10, dtype=np.int32)}
        result = report.guarded_final_evaluation_inputs(
            selection_path,
            baseline_specs=specs,
            evaluator_input_spec=input_spec,
            candidate_loader=lambda path: {"path": path},
            baseline_loader=lambda tag, path: {"tag": tag, "path": path},
            contacts_loader=lambda: contacts,
            alignment_verifier=lambda loaded, raw_path: (
                calls.append(("alignment", raw_path, len(loaded["ci"]))) or
                {"aligned": True, "raw_records": 10, "contacts_records": 10}),
            labels_loader=lambda loaded, raw_path: (
                calls.append(("labels", raw_path, len(loaded["ci"]))) or
                (np.zeros(10, dtype=np.int8), np.zeros(10, dtype=np.int8))),
            reference_loader=lambda path: calls.append(("reference", path)) or {"path": path},
            strict_p9016=False,
        )
        self.assertEqual(result["record_alignment"],
                         {"aligned": True, "raw_records": 10, "contacts_records": 10, "label_records": 10})
        self.assertEqual(result["evaluator_input_audit"]["raw_pairs"]["actual_sha256"],
                         input_spec.expected_raw_pairs_sha256)
        self.assertEqual([call[0] for call in calls], ["alignment", "labels", "reference"])

    def test_failed_candidate_is_explicit_not_silently_deleted(self):
        _grid, selection_path, _selected_path, _random_path = self._selection_fixture(random_failed=True)
        verified = report.verify_training_complete(selection_path, strict_p9016=False)
        failed = verified.candidates["random_joint"]
        self.assertEqual(failed.terminal_status, "failed")
        self.assertIsNone(failed.coordinates_path)
        self.assertEqual(verified.selected_id, "consensus_joint")

    def test_guarded_input_loads_all_nonfailed_candidates_before_evaluation(self):
        grid, selection_path, _selected_path, _random_path = self._selection_fixture()
        specs = self._baseline_specs(grid)
        loaded_candidates = []
        result = report.guarded_evaluation_inputs(
            selection_path,
            baseline_specs=specs,
            candidate_loader=lambda path: loaded_candidates.append(Path(path).name) or {"path": path},
            baseline_loader=lambda tag, path: {"tag": tag, "path": path},
            labels_loader=lambda: "synthetic-labels",
            reference_loader=lambda: "synthetic-reference",
            strict_p9016=False,
        )
        self.assertEqual(set(result["candidate_coordinates"]), {"consensus_joint", "random_joint"})
        self.assertEqual(len(loaded_candidates), 2)
        self.assertEqual(result["selected_coordinates"], result["candidate_coordinates"]["consensus_joint"])
        self.assertEqual(result["labels"], "synthetic-labels")
        self.assertEqual(result["reference"], "synthetic-reference")

    def test_coordinate_hash_failure_stops_all_evaluator_loaders(self):
        _grid, selection_path, selected_path, _random_path = self._selection_fixture()
        with selected_path.open("a") as handle:
            handle.write("# changed after selection hash\n")
        calls = []
        with self.assertRaises(report.SelectionValidationError):
            report.guarded_evaluation_inputs(
                selection_path,
                baseline_specs={},
                candidate_loader=lambda _path: calls.append("candidate"),
                baseline_loader=lambda _tag, _path: calls.append("baseline"),
                labels_loader=lambda: calls.append("labels"),
                reference_loader=lambda: calls.append("reference"),
                strict_p9016=False,
            )
        self.assertEqual(calls, [])

    def test_source_gate_hash_failure_stops_phase_and_reference_loaders(self):
        grid, selection_path, _selected_path, _random_path = self._selection_fixture()
        specs = self._baseline_specs(grid)
        gate = Path(specs["oracle"].source_gate_path)
        gate.write_text(gate.read_text() + "\n")
        calls = []
        with self.assertRaises(report.SelectionValidationError):
            report.guarded_evaluation_inputs(
                selection_path,
                baseline_specs=specs,
                candidate_loader=lambda _path: calls.append("candidate"),
                baseline_loader=lambda _tag, _path: calls.append("baseline"),
                labels_loader=lambda: calls.append("labels"),
                reference_loader=lambda: calls.append("reference"),
                strict_p9016=False,
            )
        self.assertEqual(calls, [])

    def _metric_structures(self, n_bins=60):
        positions = report.METRIC_GRID_OFFSET_BP + np.arange(n_bins, dtype=np.int64) * report.FINAL_BIN_BP
        t = np.arange(n_bins, dtype=float)
        mat = np.column_stack((t, 0.09 * t * t + np.sin(t * 0.31), np.cos(t * 0.17) * 4.0))
        pat = np.column_stack((np.sin(t * 0.37) * 7.0, 0.71 * t + np.cos(t * 0.11),
                               np.cos(t * 0.29) * 5.0))

        def track(points):
            return {int(position): np.asarray(point, dtype=float) for position, point in zip(positions, points)}

        reference = {"chr1(mat)": track(mat), "chr1(pat)": track(pat)}
        ours = {"c01a": track(mat), "c01b": track(pat)}
        return ours, reference

    @staticmethod
    def _swapped(structs):
        return {"c01a": structs["c01b"], "c01b": structs["c01a"]}

    def test_gauge_swap_and_missing_common_masks_are_preserved(self):
        selected, reference = self._metric_structures()
        random = {track: dict(points) for track, points in selected.items()}
        oracle = {track: dict(points) for track, points in selected.items()}
        b1, b2 = np.triu_indices(60, k=1)
        labels = np.where((b1 + 2 * b2) % 2, 0, 1).astype(np.int8)
        before = report.paired_r1(selected, random, oracle, reference, chromosome_index=0,
                                  chromosome_name="chr1", labels=labels, b1=b1, b2=b2, n_bins=60)
        after = report.paired_r1(self._swapped(selected), random, oracle, reference, chromosome_index=0,
                                 chromosome_name="chr1", labels=labels, b1=b1, b2=b2, n_bins=60)
        self.assertAlmostEqual(before["selected"]["accuracy"], after["selected"]["accuracy"], places=12)
        r2_before = report.paired_r2(selected, random, reference, chromosome_index=0,
                                     chromosome_name="chr1", n_bins=60)
        r2_after = report.paired_r2(self._swapped(selected), random, reference, chromosome_index=0,
                                    chromosome_name="chr1", n_bins=60)
        self.assertAlmostEqual(r2_before["left"], r2_after["left"], places=12)

        missing_random = {track: dict(points) for track, points in random.items()}
        del missing_random["c01a"][report.METRIC_GRID_OFFSET_BP + 7 * report.FINAL_BIN_BP]
        masked = report.paired_r1(selected, missing_random, oracle, reference, chromosome_index=0,
                                  chromosome_name="chr1", labels=labels, b1=b1, b2=b2, n_bins=60)
        denominator = masked["paired_denominator"]
        self.assertLess(denominator["n_common"], denominator["n_eligible_before_finite"])
        self.assertGreater(denominator["n_random_missing_excluded"], 0)
        for tag in ("selected", "random", "oracle"):
            self.assertEqual(masked[tag]["n_denominator"], denominator["n_common"])

    def test_preregistered_metrics_keep_frozen_selection_and_positive_aliases(self):
        selected, reference = self._metric_structures()
        random_baseline = {"c01a": dict(selected["c01a"]), "c01b": dict(selected["c01a"])}
        oracle = {track: dict(points) for track, points in selected.items()}
        consensus = {"c01a": dict(selected["c01a"])}
        b1, b2 = np.triu_indices(60, k=1)
        labels = np.where((b1 + 2 * b2) % 2, 0, 1).astype(np.int8)
        evaluated = report.evaluate_preregistered_candidates(
            candidates={"consensus_joint": selected, "random_joint": self._swapped(selected)},
            selected_id="consensus_joint", random=random_baseline, oracle=oracle, consensus=consensus,
            reference=reference, chromosome_index=0, chromosome_name="chr1", labels=labels,
            b1=b1, b2=b2, n_metric_bins=60)
        self.assertEqual(evaluated["selected_id_frozen"], "consensus_joint")
        self.assertFalse(evaluated["reference_reselection_performed"])
        self.assertEqual(set(evaluated["candidates"]), {"consensus_joint", "random_joint"})
        self.assertTrue(evaluated["candidates"]["consensus_joint"]["selected_by_training_count_nll"])
        for row in evaluated["candidates"].values():
            self.assertGreater(row["R1"]["paired_denominator"]["n_common"], 0)
        paired_r2 = evaluated["candidates"]["consensus_joint"]["R2"]["selected_vs_random"]
        self.assertTrue(paired_r2["applicable"])
        self.assertAlmostEqual(paired_r2["delta_selected_minus_random"],
                               paired_r2["delta_left_minus_right"], places=12)
        self.assertAlmostEqual(paired_r2["delta_selected_minus_random"],
                               -paired_r2["delta_right_minus_left"], places=12)
        self.assertGreater(paired_r2["delta_selected_minus_random"], 0.0)
        r3_left = {"applicable": True, "global_label": 0,
                   "detail": [{"grid_start_bin": 0, "label": 0}, {"grid_start_bin": 20, "label": 0}]}
        r3_right = {"applicable": True, "global_label": 0,
                    "detail": [{"grid_start_bin": 0, "label": 1}, {"grid_start_bin": 20, "label": 1}]}
        paired_r3 = report.paired_r3(r3_left, r3_right, left_name="selected", right_name="random")
        self.assertGreater(paired_r3["delta_selected_minus_random"], 0.0)
        self.assertAlmostEqual(paired_r3["delta_selected_minus_random"],
                               -paired_r3["delta_right_minus_left"], places=12)

    def test_single_consensus_is_explicit_n_a_and_keeps_structural_baseline(self):
        selected, reference = self._metric_structures()
        random = {track: dict(points) for track, points in selected.items()}
        oracle = {track: dict(points) for track, points in selected.items()}
        consensus = {"c01a": dict(selected["c01a"])}
        b1, b2 = np.triu_indices(60, k=1)
        labels = np.where((b1 + b2) % 2, 0, 1).astype(np.int8)
        result = report.evaluate_chromosome(selected=selected, random=random, oracle=oracle,
                                            consensus=consensus, reference=reference, chromosome_index=0,
                                            chromosome_name="chr1", labels=labels, b1=b1, b2=b2,
                                            n_metric_bins=60)
        self.assertFalse(result["consensus"]["R1"]["applicable"])
        self.assertIsNone(result["consensus"]["R1"]["accuracy"])
        self.assertFalse(result["consensus"]["R2"]["applicable"])
        self.assertIn("a0_mat", result["consensus"]["R2"]["single_track_baseline"])
        self.assertFalse(result["consensus"]["R3"]["applicable"])
        self.assertIsNone(result["consensus"]["R3"]["frac_consistent"])

    def test_offline_canvas_embeds_all_tracks_and_preserves_raw_export(self):
        grid = self._grid20()
        coordinates_path = self.root / "selected.3dg"
        self._write_coordinates(coordinates_path, grid)
        structs = report.read_full_grid_coordinates(coordinates_path, grid)
        delivery = report.write_selected_structure_delivery(self.root / "delivery", structs, grid)
        html_path = Path(delivery["interactive_3d"]["html"])
        delivery_contents = html_path.read_text()
        self.assertTrue(Path(delivery["selected_3d"]["figure"]).is_file())
        self.assertEqual(delivery["track_coverage"]["n_tracks"], 40)
        self.assertEqual(delivery["renderer_provenance"]["role"], "evaluation_renderer")
        self.assertFalse(delivery["interactive_3d"]["external_cdn"])
        if delivery["interactive_3d"]["mode"] == "vanilla_canvas_embedded":
            self.assertNotIn("cdn.", delivery_contents.lower())
            self.assertNotIn("http://", delivery_contents.lower())
            self.assertNotIn("https://", delivery_contents.lower())
        else:
            self.assertEqual(delivery["interactive_3d"]["mode"], "plotly_embedded")
            self.assertNotIn("cdn.plot.ly", delivery_contents.lower())
        canvas_path = self.root / "canvas-viewer.html"
        canvas_path.write_text(report._canvas_html(structs, grid, "synthetic selected structure"))
        contents = canvas_path.read_text()
        self.assertNotIn("cdn.", contents.lower())
        self.assertNotIn("http://", contents.lower())
        self.assertNotIn("https://", contents.lower())
        self.assertIn("const STRUCTURE_DATA=", contents)
        self.assertIn("data-copy=", contents)
        self.assertIn("data-chromosome=", contents)
        self.assertIn("export-raw", contents)
        self.assertIn("rotatePoint", contents)
        self.assertIn(" | bin ", contents)
        self.assertIn("start", contents)
        match = re.search(r"<script>(.*?)</script>", contents, flags=re.DOTALL)
        self.assertIsNotNone(match)
        script_path = self.root / "canvas-viewer.js"
        script_path.write_text(match.group(1))
        syntax = subprocess.run(["node", "--check", str(script_path)], text=True,
                                capture_output=True, check=False)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        program = """
require(process.argv[1]);
const data = globalThis.__RECONSTRUCTION_CANVAS_DATA;
const math = globalThis.__RECONSTRUCTION_CANVAS_MATH;
const viewer = globalThis.ReconstructionCanvasViewer;
if (!data || !math || !viewer) throw new Error('canvas globals unavailable');
if (data.tracks.length !== 40) throw new Error('wrong embedded track count');
const pointCount = data.tracks.reduce((n, track) => n + track.points.length, 0);
const source = [1, 2, 3];
const before = JSON.stringify(source);
const rotated = math.rotatePoint(source, Math.PI / 2, 0);
if (JSON.stringify(source) !== before) throw new Error('rotation mutated source');
if (Math.abs(rotated[0] - 3) > 1e-12 || Math.abs(rotated[2] + 1) > 1e-12) throw new Error('yaw rotation is incorrect');
if (Math.abs(Math.hypot(...rotated) - Math.hypot(...source)) > 1e-12) throw new Error('rotation changed norm');
if (viewer.rawCoordinateJSON() !== JSON.stringify(data)) throw new Error('raw export was transformed');
console.log(JSON.stringify({tracks: data.tracks.length, points: pointCount}));
"""
        execution = subprocess.run(["node", "-e", program, str(script_path)], text=True,
                                   capture_output=True, check=False)
        self.assertEqual(execution.returncode, 0, execution.stderr)
        embedded = json.loads(execution.stdout)
        self.assertEqual(embedded["tracks"], 40)
        self.assertEqual(embedded["points"], grid.n_physical_beads)


if __name__ == "__main__":
    unittest.main()
