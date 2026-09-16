import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pr import allele_r2 as r2
from pr import r2comparison


N_BINS = 24
N_CHROMOSOMES = 20
TOY_CHROMOSOMES = [
    {"name": ("chr%d" % index) if index < 20 else "chrX", "length_bp": r2.GRID_OFFSET_BP + N_BINS * r2.BIN_SIZE_BP}
    for index in range(1, 21)
]
# 上面的列表推导有意将第 20 个元素映射到 chrX。
TOY_CHROMOSOMES[-1]["name"] = "chrX"


def _track(points):
    return {
        r2.GRID_OFFSET_BP + index * r2.BIN_SIZE_BP: np.asarray(point, dtype=float)
        for index, point in enumerate(points)
    }


def _points(chromosome_index: int, copy_name: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(N_BINS, dtype=float)
    phase = 0.07 * chromosome_index
    mat = np.column_stack((
        t + phase,
        np.sin(0.31 * t + phase) + 0.03 * chromosome_index,
        1.7 * np.cos(0.17 * t + phase),
    ))
    pat = np.column_stack((
        1.5 * np.sin(0.37 * t + 0.2 + phase),
        0.71 * t + np.cos(0.11 * t + phase),
        2.1 * np.cos(0.29 * t + 0.3 + phase),
    ))
    return mat, pat


def _structures(*, swap: bool = False, single: bool = False, constant_a: bool = False):
    structures = {}
    for chromosome_index in range(N_CHROMOSOMES):
        mat, pat = _points(chromosome_index, "mat")
        if constant_a:
            first = np.zeros_like(mat)
        elif swap:
            first = pat
        else:
            first = mat
        second = mat if swap else pat
        if single:
            structures["c%02da" % (chromosome_index + 1)] = _track(first)
        else:
            structures["c%02da" % (chromosome_index + 1)] = _track(first)
            structures["c%02db" % (chromosome_index + 1)] = _track(second)
    return structures


def _reference(*, swapped: bool = False):
    result = {}
    for chromosome_index in range(N_CHROMOSOMES):
        chromosome = ("chr%d" % (chromosome_index + 1)) if chromosome_index < 19 else "chrX"
        mat, pat = _points(chromosome_index, "mat")
        if swapped:
            mat, pat = pat, mat
        result["%s(mat)" % chromosome] = _track(mat)
        result["%s(pat)" % chromosome] = _track(pat)
    return result


def _one_chromosome_conditions():
    mat, pat = _points(0, "mat")
    return [
        {"name": "chr1", "length_bp": r2.GRID_OFFSET_BP + 8 * r2.BIN_SIZE_BP}
    ], {
        "chr1(mat)": _track(mat[:8]),
        "chr1(pat)": _track(pat[:8]),
    }, {
        "good": r2comparison.R2Condition("good", "Good", {"c01a": _track(mat[:8]), "c01b": _track(pat[:8])}),
        "swapped": r2comparison.R2Condition("swapped", "Swapped", {"c01a": _track(pat[:8]), "c01b": _track(mat[:8])}),
        "identical": r2comparison.R2Condition("identical", "Identical", {"c01a": _track(mat[:8]), "c01b": _track(mat[:8])}),
        "constant": r2comparison.R2Condition("constant", "Constant", {"c01a": _track(np.zeros_like(mat[:8])), "c01b": _track(pat[:8])}),
    }


def _toy_conditions():
    conditions = {}
    for condition_id in r2.NEW_CONDITION_IDS:
        variant_id, bundle_id = condition_id.rsplit("-", 1)
        conditions[condition_id] = r2comparison.R2Condition(
            condition_id, "%s %s" % (variant_id, bundle_id), _structures(), n_copies=2,
            role="new_variant", endpoint_status="solver_converged", accepted_as=condition_id,
        )
    for item in r2.HISTORICAL_CONTROLS:
        conditions[item["id"]] = r2comparison.R2Condition(
            item["id"], item["display_name"], _structures(single=item["n_copies"] == 1),
            n_copies=item["n_copies"], role="historical_control", endpoint_status="accepted",
            accepted_as=item["accepted_as"],
        )
    return conditions


def _toy_meta(failed_id):
    meta = {}
    for condition_id in r2.NEW_CONDITION_IDS:
        variant_id, bundle_id = condition_id.rsplit("-", 1)
        meta[condition_id] = {
            "id": condition_id, "variant_id": variant_id, "bundle_id": bundle_id,
            "display_name": condition_id, "role": "new_variant",
            "coordinate_status": "failed" if condition_id == failed_id else "available",
            "endpoint_status": "failed_input" if condition_id == failed_id else "solver_converged",
        }
    for item in r2.HISTORICAL_CONTROLS:
        meta[item["id"]] = {**item, "coordinate_status": "available", "variant_id": None, "bundle_id": None}
    return meta


def _evaluate_toy(failed_id="C3-bundle3"):
    conditions = _toy_conditions()
    failed = conditions.pop(failed_id)
    failed_meta = {
        "id": failed_id, "variant_id": "C3", "bundle_id": "bundle3",
        "display_name": failed_id, "role": "new_variant", "n_copies": 2,
        "endpoint_status": "failed_input", "coordinate_status": "failed",
        "failure_reason": "synthetic_fit_missing",
    }
    evaluated = r2.evaluate_in_memory(TOY_CHROMOSOMES, conditions, _reference(),
                                       failed_conditions={failed_id: failed_meta})
    return evaluated, _toy_meta(failed_id), conditions


class OrientationAndAvailabilityTests(unittest.TestCase):
    def test_candidate_row_swap_preserves_named_metrics(self):
        chromosomes, reference, conditions = _one_chromosome_conditions()
        evaluated = r2.evaluate_in_memory(chromosomes, conditions, reference)
        good = evaluated[0]["conditions"]["good"]
        swapped = evaluated[0]["conditions"]["swapped"]
        for key in ("matched", "cross", "contrast", "margin_mat", "margin_pat", "minmargin"):
            self.assertAlmostEqual(good[key], swapped[key], places=12, msg=key)
        self.assertEqual(swapped["orientation"], "swapped")
        self.assertEqual(swapped["matched_rho_candidate_A"], good["matched_rho_candidate_A"])
        self.assertEqual(swapped["matched_rho_candidate_B"], good["matched_rho_candidate_B"])

    def test_reference_swap_exchanges_named_margins(self):
        chromosomes, reference, conditions = _one_chromosome_conditions()
        original = r2.evaluate_in_memory(chromosomes, {"good": conditions["good"]}, reference)[0]["conditions"]["good"]
        swapped_reference = {"chr1(mat)": reference["chr1(pat)"], "chr1(pat)": reference["chr1(mat)"]}
        swapped = r2.evaluate_in_memory(chromosomes, {"good": conditions["good"]}, swapped_reference)[0]["conditions"]["good"]
        self.assertAlmostEqual(swapped["margin_mat"], original["margin_pat"], places=12)
        self.assertAlmostEqual(swapped["margin_pat"], original["margin_mat"], places=12)
        self.assertAlmostEqual(swapped["contrast"], original["contrast"], places=12)

    def test_orientation_is_selected_independently_per_chromosome(self):
        mat1, pat1 = _points(0, "mat")
        mat2, pat2 = _points(1, "mat")
        structures = {
            "c01a": _track(mat1), "c01b": _track(pat1),
            "c02a": _track(pat2), "c02b": _track(mat2),
        }
        reference = {
            "chr1(mat)": _track(mat1), "chr1(pat)": _track(pat1),
            "chr2(mat)": _track(mat2), "chr2(pat)": _track(pat2),
        }
        condition = r2comparison.R2Condition("candidate", "Candidate", structures)
        chromosomes = [
            {"name": "chr1", "length_bp": r2.GRID_OFFSET_BP + 8 * r2.BIN_SIZE_BP},
            {"name": "chr2", "length_bp": r2.GRID_OFFSET_BP + 8 * r2.BIN_SIZE_BP},
        ]
        evaluated = r2.evaluate_in_memory(chromosomes, {"candidate": condition}, reference)
        self.assertEqual(evaluated[0]["conditions"]["candidate"]["orientation"], "direct")
        self.assertEqual(evaluated[1]["conditions"]["candidate"]["orientation"], "swapped")

    def test_bad_copy_is_not_averaged_with_good_copy(self):
        chromosomes, reference, conditions = _one_chromosome_conditions()
        evaluated = r2.evaluate_in_memory(chromosomes, {"constant": conditions["constant"]}, reference)
        row = evaluated[0]["conditions"]["constant"]
        self.assertEqual(row["metric_status"], "unavailable")
        self.assertIsNone(row["similarity"])
        self.assertIsNone(row["rho_A_mat"])
        self.assertIsNotNone(row["rho_B_pat"])
        self.assertIn("constant_candidate_distance", row["reason"])

    def test_geometry_tie_is_unresolved_and_not_both_positive(self):
        chromosomes, reference, conditions = _one_chromosome_conditions()
        evaluated = r2.evaluate_in_memory(chromosomes, {"identical": conditions["identical"]}, reference)
        row = evaluated[0]["conditions"]["identical"]
        self.assertEqual(row["orientation"], "unresolved_tie")
        self.assertEqual(row["pairing"], "tie_average")
        self.assertEqual(row["contrast"], 0.0)
        self.assertIsNone(row["margin_mat"])
        self.assertIsNone(row["margin_pat"])
        self.assertFalse(row["both_positive"])
        self.assertTrue(row["geometry_tie"])

    def test_cross_orientation_keeps_original_candidate_labels_and_fixed_refs(self):
        row = {
            "rho_A_mat": 0.1, "rho_A_pat": 0.8,
            "rho_B_mat": 0.7, "rho_B_pat": 0.2,
            "rho_reasons": {},
        }
        result = r2._derive_two_copy(row)
        self.assertEqual(result["orientation"], "swapped")
        self.assertAlmostEqual(result["matched_rho_candidate_A"], 0.8, places=12)
        self.assertAlmostEqual(result["matched_rho_candidate_B"], 0.7, places=12)
        self.assertAlmostEqual(result["matched_ref1"], 0.7, places=12)
        self.assertAlmostEqual(result["matched_ref2"], 0.8, places=12)
        self.assertAlmostEqual(result["margin_mat"], 0.5, places=12)
        self.assertAlmostEqual(result["margin_pat"], 0.7, places=12)

    def test_solver_failed_finite_rows_are_descriptive_only(self):
        evaluated, _meta, _conditions = _evaluate_toy()
        seeds = {bundle: copy.deepcopy(evaluated) for bundle in r2.BUNDLE_IDS}
        row = seeds["bundle2"][0]["conditions"]["C1-bundle2"]
        row["endpoint_status"] = "solver_failed"
        row["similarity"] = 0.42
        matrix = r2.make_bootstrap_index_matrix()
        summary = r2.summarise_paired_seed_effects(
            seeds,
            {bundle: "C1-%s" % bundle for bundle in r2.BUNDLE_IDS},
            {bundle: "C0-%s" % bundle for bundle in r2.BUNDLE_IDS},
            bootstrap_indices=matrix,
        )
        self.assertEqual(summary["primary"]["status"], "n/a_planned_seed_failure")
        self.assertEqual(summary["primary"]["valid_seed_count"], 2)
        self.assertEqual(summary["per_seed"][1]["reason"], "solver_failed")
        self.assertTrue(summary["per_seed"][1]["descriptive_only"])
        self.assertIsNotNone(summary["per_seed"][1]["descriptive_per_chromosome_delta"]["chr1"])
        self.assertIsNone(summary["per_seed"][1]["per_chromosome_delta"]["chr1"])


class ToyPipelineTests(unittest.TestCase):
    def test_full_21_condition_20_chromosome_toy_writes_and_renders(self):
        evaluated, condition_meta, _conditions = _evaluate_toy()
        self.assertEqual(len(evaluated), 20)
        self.assertEqual(sum(len(item["conditions"]) for item in evaluated), 420)
        self.assertEqual(len(r2.flatten_rows(evaluated, r2.ALL_CONDITION_IDS, condition_meta)), 420)
        self.assertEqual(evaluated[0]["mask"]["excluded_failed_condition_ids"], ["C3-bundle3"])
        self.assertIn("C3-bundle3", evaluated[0]["mask"]["excluded_conditions"])
        self.assertEqual(evaluated[0]["mask"]["included_condition_ids"],
                         [item for item in r2.NEW_CONDITION_IDS if item != "C3-bundle3"] + list(r2.HISTORICAL_IDS))
        failed = evaluated[0]["conditions"]["C3-bundle3"]
        self.assertIsNone(failed["similarity"])
        self.assertEqual(failed["reason"], "synthetic_fit_missing")

        matrix = r2.make_bootstrap_index_matrix()
        seed_evaluated = {bundle: evaluated for bundle in r2.BUNDLE_IDS}
        pair = r2.summarise_paired_seed_effects(
            seed_evaluated,
            {bundle: "C1-%s" % bundle for bundle in r2.BUNDLE_IDS},
            {bundle: "C0-%s" % bundle for bundle in r2.BUNDLE_IDS},
            bootstrap_indices=matrix, label="C1-C0",
        )
        self.assertEqual(pair["primary"]["status"], "ok")
        self.assertEqual(sum(item["mean_delta"] is not None for item in pair["per_seed"]), 3)
        self.assertEqual(len(pair["primary"]["per_chromosome"]), 20)
        self.assertEqual(pair["primary"]["bootstrap"]["matrix_array_sha256"], r2.bootstrap_index_sha256(matrix))
        self.assertTrue(all(item["bootstrap"]["matrix_array_sha256"] == r2.bootstrap_index_sha256(matrix)
                            for item in pair["per_seed"]))

        all_pairs = r2._build_primary_pairs(seed_evaluated, matrix)
        self.assertEqual(len(all_pairs), 40)
        metric_counts = {}
        for item in all_pairs:
            metric_counts[item["metric"]] = metric_counts.get(item["metric"], 0) + 1
        self.assertEqual(metric_counts, {metric: 5 for metric in r2.PRIMARY_PAIR_METRICS})
        self.assertTrue(all(item["primary"]["bootstrap"]["matrix_array_sha256"] == r2.bootstrap_index_sha256(matrix)
                            for item in all_pairs))
        paired_rows = r2.flatten_paired_rows(all_pairs)
        self.assertEqual({row["metric"] for row in paired_rows}, set(r2.PRIMARY_PAIR_METRICS))
        self.assertEqual(len([row for row in paired_rows if row["level"] == "primary"]), 40)
        self.assertEqual(len([row for row in paired_rows if row["level"] == "seed"]), 120)

        summaries = {
            "condition_summary": r2.condition_summary(evaluated, r2.ALL_CONDITION_IDS),
            "representatives": {variant: {"condition_id": "%s-bundle1" % variant}
                                for variant in r2.VARIANT_IDS},
            "paired_comparisons": all_pairs,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "toy-r2"
            output.mkdir()
            rows = r2.flatten_rows(evaluated, r2.ALL_CONDITION_IDS, condition_meta)
            r2.write_table(output / "toy.tsv", rows, delimiter="\t")
            r2.write_table(output / "toy.csv", rows, delimiter=",")
            np.save(output / "bootstrap_index_matrix_seed9301.npy", matrix)
            file_sha = r2.bootstrap_file_sha256(output / "bootstrap_index_matrix_seed9301.npy")
            array_sha = r2.bootstrap_index_sha256(matrix)
            self.assertNotEqual(file_sha, array_sha)
            self.assertEqual(file_sha, hashlib.sha256((output / "bootstrap_index_matrix_seed9301.npy").read_bytes()).hexdigest())
            self.assertTrue(np.array_equal(np.load(output / "bootstrap_index_matrix_seed9301.npy", allow_pickle=False), matrix))
            r2._write_json(output / "summary.json", {
                "condition_summary": summaries["condition_summary"],
                "paired": summaries["paired_comparisons"],
                "bootstrap": {"matrix_array_sha256": array_sha, "file_sha256": file_sha},
            })
            plots = r2._render_outputs(output, evaluated, r2.ALL_CONDITION_IDS,
                                       condition_meta, summaries, all_pairs)
            self.assertTrue(plots)
            contrast_plot = plots["paired_effects"]["contrast"]
            self.assertEqual(contrast_plot["n_comparisons"], 5)
            self.assertEqual(contrast_plot["n_na"], 1)
            self.assertIn("C3-C0", contrast_plot["na_reasons"])
            contrast_seed = plots["seed_stability"]["contrast"]
            self.assertEqual(contrast_seed["n_comparisons"], 5)
            self.assertEqual(contrast_seed["valid_seed_counts"]["C3-C0"], {"valid": 2, "planned": 3})
            rendered_files = list((output / "plots").rglob("*"))
            self.assertGreaterEqual(len(rendered_files), 24)
            self.assertTrue(all(path.stat().st_size > 0 for path in rendered_files))

    def test_incomplete_seed_is_primary_na_and_available_only_is_explicit(self):
        evaluated, _meta, _conditions = _evaluate_toy()
        seeds = {bundle: copy.deepcopy(evaluated) for bundle in r2.BUNDLE_IDS}
        seeds["bundle2"][0]["conditions"]["C1-bundle2"]["similarity"] = None
        matrix = r2.make_bootstrap_index_matrix()
        result = r2.summarise_paired_seed_effects(
            seeds,
            {bundle: "C1-%s" % bundle for bundle in r2.BUNDLE_IDS},
            {bundle: "C0-%s" % bundle for bundle in r2.BUNDLE_IDS},
            bootstrap_indices=matrix, label="incomplete",
        )
        self.assertEqual(result["primary"]["status"], "n/a_planned_seed_failure")
        self.assertEqual(result["primary"]["planned_seed_count"], 3)
        self.assertEqual(result["primary"]["valid_seed_count"], 2)
        self.assertIsNone(result["primary"]["mean_delta"])
        self.assertEqual(result["per_seed"][1]["reason"], "metric_missing")
        self.assertEqual(result["available_only"]["status"], "available_only")
        self.assertEqual(result["available_only"]["seed_ids"], ["bundle1", "bundle3"])
        self.assertEqual(len(result["per_seed"]), 3)
        self.assertEqual(len(result["per_seed"][0]["per_chromosome_delta"]), 20)


class LockAdapterTests(unittest.TestCase):
    def test_fresh_controller_terminal_records_adapt_to_release_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fresh-real-run"
            root.mkdir()
            prepared = Path(directory) / "preparation"
            r2.prepare_only(prepared)
            jobs = []
            terminal_jobs = []
            selection_variants = []
            zero_sha = "0" * 64
            for variant_id in r2.VARIANT_IDS:
                candidates = []
                for bundle_id in r2.BUNDLE_IDS:
                    condition_id = "%s-%s" % (variant_id, bundle_id)
                    job_root = root / "jobs" / condition_id
                    job_root.mkdir(parents=True)
                    job_config_path = job_root / "job.json"
                    job_payload = {
                        "job_id": condition_id,
                        "model_id": variant_id,
                        "bundle_id": bundle_id,
                        "source_snpfree_sha256": zero_sha,
                        "source_snapshot_sha256": zero_sha,
                        "synthetic_gate_sha256": zero_sha,
                        "x0_file_sha256": zero_sha,
                        "x0_coordinate_sha256": zero_sha,
                        "protocol_release_sha256": r2.AUTHORITATIVE_PROTOCOL_SHA256,
                    }
                    job_config_path.write_text(json.dumps(job_payload, sort_keys=True) + "\n", encoding="utf-8")
                    job_config_sha = hashlib.sha256(job_config_path.read_bytes()).hexdigest()
                    jobs.append({"job_id": condition_id, "model_id": variant_id, "bundle_id": bundle_id,
                                 "job_config": "jobs/%s/job.json" % condition_id,
                                 "job_config_sha256": job_config_sha})
                    coordinate_path = job_root / "final_coordinates.3dg"
                    coordinate_path.write_text("c01a\\t3000000\\t0\\t0\\t0\\n", encoding="utf-8")
                    coordinate_sha = hashlib.sha256(coordinate_path.read_bytes()).hexdigest()
                    final_path = job_root / "final.json"
                    final_payload = {
                        "status": "solver_converged", "fit_called": True,
                        "source": {"sha256": zero_sha, "snapshot_sha256": zero_sha},
                        "synthetic_gate": {"sha256": zero_sha},
                        "x0": {"file_sha256": zero_sha, "coordinate_sha256": zero_sha},
                        "final": {"coordinates_file": {"path": str(coordinate_path), "sha256": coordinate_sha}},
                    }
                    final_path.write_text(json.dumps(final_payload, sort_keys=True) + "\n", encoding="utf-8")
                    (job_root / "status.json").write_text(json.dumps({
                        "job_id": condition_id, "status": "solver_converged", "fit_called": True,
                        "final_path": str(final_path),
                    }, sort_keys=True) + "\n", encoding="utf-8")
                    candidates.append({"bundle_id": bundle_id, "selection_eligible": True,
                                       "count_nll_normalized": 1.0, "excluded_reason": None})
                    terminal_jobs.append({"job_id": condition_id, "status": "solver_converged", "fit_called": True})
                selection_variants.append({"model_id": variant_id, "status": "complete",
                                          "selected_bundle_id": "bundle1",
                                          "criterion": "count_nll_normalized", "candidates": candidates})
            (root / "manifest.json").write_text(json.dumps({"status": "ready_for_worker", "fit_called": False,
                "jobs": jobs}, sort_keys=True) + "\n", encoding="utf-8")
            (root / "selection.json").write_text(json.dumps({"status": "complete", "variants": selection_variants}, sort_keys=True) + "\n", encoding="utf-8")
            (root / "termination_audit.json").write_text(json.dumps({"status": "complete", "job_count": 15,
                "terminal_job_count": 15, "active_jobs": [], "jobs": terminal_jobs}, sort_keys=True) + "\n", encoding="utf-8")
            release_path = root / "release_manifest.json"
            release = r2.build_release_manifest_from_experiment(root, prepared / "config.json",
                                                                 output_path=release_path)
            self.assertEqual(release["status"], "locked_for_evaluation")
            self.assertEqual([item["id"] for item in release["conditions"]], list(r2.ALL_CONDITION_IDS))
            self.assertEqual(sum(item["coordinate_status"] == "available" for item in release["conditions"]), 21)
            self.assertEqual(release["within_variant_representatives"]["C2-free"], "C2-free-bundle1")
            validated = r2._validate_release_manifest(
                json.loads((prepared / "config.json").read_text(encoding="utf-8")),
                prepared / "config.json", release_path,
            )
            self.assertEqual(len(validated["conditions"]), 21)

    def test_prepare_only_never_activates_027_predecessor(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / "preparation"
            manifest = r2.prepare_only(prepared)
            with self.assertRaisesRegex(r2.AlleleR2Error, "no-fit predecessor"):
                r2.build_release_manifest_from_experiment(
                    "/mnt/ssd/zliu/phase_restart/test_res/027-20260913_150114-post020-allele-ablation-real",
                    prepared / "config.json",
                )
            self.assertEqual(manifest["coordinate_files_opened"], 0)
            self.assertFalse(manifest["reference_opened"])
            self.assertEqual(manifest["expected_condition_count"], 21)


if __name__ == "__main__":
    unittest.main()
