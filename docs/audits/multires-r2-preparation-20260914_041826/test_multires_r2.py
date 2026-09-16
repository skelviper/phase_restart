"""已准备多分辨率 R2 模块的轻量 contract 与 metric tests。"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import multires_r2 as r2


class FrozenContractTests(unittest.TestCase):
    def test_preparation_validation_is_read_boundary_only(self):
        result = r2.validate_preparation()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["prepared_only"])
        self.assertTrue(result["evaluation_not_run"])
        self.assertEqual(result["candidate_coordinates_opened"], 0)
        self.assertFalse(result["reference_opened"])
        self.assertEqual(result["pending_endpoint_paths"], 10)
        self.assertEqual(result["selection_tie_tolerance_per_record"], 1e-9)
        self.assertEqual(result["orientation_tie_tolerance"], 1e-12)

    def test_pending_release_is_rejected_before_payload_access(self):
        with self.assertRaises(r2.PreparationError):
            r2.validate_release(r2.CONFIG_PATH, r2.PREP_DIR / "release_contract.json")

    def test_endpoint_ids_are_all_two_copy_contract(self):
        config = r2.read_json(r2.CONFIG_PATH)
        self.assertEqual(len(config["endpoints"]), 10)
        self.assertTrue(all(item["n_copies"] == 2 for item in config["endpoints"]))
        self.assertTrue(config["training"]["consensus_joint_is_initialization_not_single_copy_baseline"])
        self.assertEqual(config["training"]["main_backend"], "GPU")
        self.assertEqual(config["training"]["gpu_stage_count"], 30)

    def test_post_release_contract_is_frozen(self):
        config = r2.read_json(r2.CONFIG_PATH)
        post = config["post_release_acceptance"]
        self.assertEqual(post["mask_compare_fields"], ["positions", "pair_i", "pair_j", "common"])
        self.assertEqual(post["published_condition_id"], r2.HISTORICAL_020_ID)
        self.assertEqual(post["published_comparison_atol"], 1e-12)
        self.assertFalse(post["use_020_raw_evaluation"])
        self.assertTrue(post["c0_selected_cpu_gpu_diagnostic"])
        self.assertTrue(post["c0_cpu_gpu_diagnostic_nonblocking"])
        self.assertFalse(post["c0_cpu_gpu_diagnostic_reject_on_difference"])
        self.assertEqual(post["historical_cpu_anchor_label"], "020 / C0 CPU")
        self.assertTrue(post["historical_cpu_anchor_is_not_gpu_endpoint"])

    def test_exact_mask_check_rejects_same_count_different_bits(self):
        base = {
            "positions": np.array([0, 1, 2], dtype=np.int64),
            "pair_i": np.array([0, 0, 1], dtype=np.int64),
            "pair_j": np.array([1, 2, 2], dtype=np.int64),
            "common": np.array([True, False, True], dtype=bool),
        }
        new_masks = {chromosome: {key: value.copy() for key, value in base.items()}
                     for chromosome, _length in r2.CHROMOSOMES}
        legacy_masks = {chromosome: {key: value.copy() for key, value in base.items()}
                        for chromosome, _length in r2.CHROMOSOMES}
        legacy_masks["chr1"]["common"] = np.array([False, True, True], dtype=bool)
        result = r2.compare_mask_exact(new_masks, legacy_masks)
        self.assertEqual(result["status"], "FAIL")
        common_check = next(item for item in result["by_chromosome"][0]["checks"] if item["field"] == "common")
        self.assertEqual(common_check["mismatch_count"], 2)


class MetricTests(unittest.TestCase):
    def test_direct_and_swapped_are_gauge_invariant(self):
        direct = r2.derive_four_rho({"rho_A_mat": 0.8, "rho_A_pat": 0.2, "rho_B_mat": 0.1, "rho_B_pat": 0.7})
        swapped = r2.derive_four_rho({"rho_A_mat": 0.1, "rho_A_pat": 0.7, "rho_B_mat": 0.8, "rho_B_pat": 0.2})
        self.assertEqual(direct["orientation"], "direct")
        for metric in ("matched", "cross", "contrast", "minmargin"):
            self.assertAlmostEqual(direct[metric], swapped[metric], places=12)

    def test_orientation_tie_has_zero_contrast_and_no_margins(self):
        row = r2.derive_four_rho({"rho_A_mat": 0.5, "rho_A_pat": 0.5, "rho_B_mat": 0.5, "rho_B_pat": 0.5})
        self.assertEqual(row["orientation"], "unresolved_tie")
        self.assertEqual(row["contrast"], 0.0)
        self.assertIsNone(row["margin_mat"])
        self.assertIsNone(row["margin_pat"])
        self.assertFalse(row["both_positive"])

    def test_published_parity_uses_atol_and_all_geometry_fields(self):
        new_rows = []
        published_rows = []
        for chromosome, _length in r2.CHROMOSOMES:
            new = {"chromosome": chromosome, "n_common_pairs_frozen": 123, "orientation": "direct"}
            old = {"chromosome": chromosome, "n_common_pairs": "123", "orientation": "direct"}
            for index, (new_field, old_field) in enumerate(r2.POST_ACCEPTANCE_NUMERIC_FIELDS, start=1):
                value = index / 100.0
                new[new_field] = value
                old[old_field] = str(value)
            new_rows.append(new)
            published_rows.append(old)
        result = r2.compare_published_r2_rows(new_rows, published_rows, endpoint_label="synthetic")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["compared_numeric_values"], 20 * len(r2.POST_ACCEPTANCE_NUMERIC_FIELDS))
        altered = [dict(row) for row in new_rows]
        altered[0]["contrast"] += 2e-12
        failed = r2.compare_published_r2_rows(altered, published_rows, endpoint_label="synthetic")
        self.assertEqual(failed["status"], "FAIL")

    def test_missing_copy_does_not_average(self):
        row = r2.derive_four_rho({"rho_A_mat": 0.5, "rho_A_pat": None, "rho_B_mat": 0.4, "rho_B_pat": 0.6,
                                  "rho_A_pat_reason": "nonfinite_distance_vector"})
        self.assertEqual(row["metric_status"], "unavailable")
        self.assertIsNone(row["matched"])
        self.assertIn("rho_A_pat", row["reason"])

    def test_plot_arrays_keep_nan_for_na(self):
        rows = []
        for endpoint_id in r2.ENDPOINT_IDS:
            variant = next(item for item in r2.VARIANT_IDS if endpoint_id.startswith(item + "-"))
            source = endpoint_id[len(variant) + 1:]
            for chromosome, _length in r2.CHROMOSOMES:
                rows.append({"endpoint_id": endpoint_id, "variant_id": variant, "source_id": source,
                             "chromosome": chromosome, "matched": 0.5, "cross": 0.2,
                             "contrast": 0.3, "minmargin": 0.1})
        rows[0]["contrast"] = None
        reps = [{"variant_id": variant, "representative_endpoint_id": "%s-%s" % (variant, r2.SOURCE_IDS[0])}
                for variant in r2.VARIANT_IDS]
        anchor_rows = [{"endpoint_id": r2.HISTORICAL_CPU_ANCHOR_ID, "variant_id": "historical_cpu_anchor",
                        "source_id": "historical_cpu_anchor", "chromosome": chromosome,
                        "matched": 0.4, "cross": 0.1, "contrast": 0.3, "minmargin": 0.2}
                       for chromosome, _length in r2.CHROMOSOMES]
        arrays, metadata = r2.build_plot_arrays(rows, reps, anchor_rows)
        self.assertEqual(arrays["main_contrast"].shape, (6, 20))
        self.assertEqual(arrays["source_consensus_joint_base1103_contrast"].shape, (5, 20))
        self.assertEqual(metadata["main_condition_order"][-1], "020 / C0 CPU")
        self.assertTrue(np.isnan(arrays["main_contrast"][0, 0]))
        self.assertEqual(metadata["arrays"]["main_contrast"]["n_a"], 1)


if __name__ == "__main__":
    unittest.main()
