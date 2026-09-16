#!/usr/bin/env python
"""020 与 029 C0 报告的独立只读检查。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
CHROMS = [*(f"chr{i}" for i in range(1, 20)), "chrX"]
METHODS = ["020 multires", "029 C0 direct 1Mb"]
TOL = 1e-10


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def isclose(a: Any, b: Any, tol: float = TOL) -> bool:
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def main() -> int:
    checks: dict[str, Any] = {}
    failures: list[str] = []
    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks[name] = {"passed": bool(condition)}
        if detail is not None:
            checks[name]["detail"] = detail
        if not condition:
            failures.append(name)

    with (OUT / "per_chromosome.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    check("per_chromosome_40_rows", len(rows) == 40, {"rows": len(rows)})
    check("per_chromosome_two_methods", sorted({row["method"] for row in rows}) == sorted(METHODS))
    check("per_chromosome_all_20", all(sum(row["method"] == method for row in rows) == 20 for method in METHODS))
    check("per_chromosome_all_chromosomes", sorted({row["chromosome"] for row in rows}) == sorted(CHROMS))

    formula_failures = []
    orientation_failures = []
    for row in rows:
        values = {key: float(row[key]) for key in (
            "rho_a_mat", "rho_a_pat", "rho_b_mat", "rho_b_pat",
            "direct_original", "cross_original", "matched", "cross_matched", "contrast",
            "margin_mat", "margin_pat", "minmargin")}
        direct_swap = (values["rho_b_mat"] + values["rho_a_pat"]) / 2.0
        cross_swap = (values["rho_b_pat"] + values["rho_a_mat"]) / 2.0
        if not (isclose(direct_swap, values["cross_original"]) and isclose(cross_swap, values["direct_original"])):
            orientation_failures.append((row["chromosome"], row["method"], "candidate-row swap invariance"))
        if not (isclose(values["matched"], max(values["direct_original"], values["cross_original"])) and
                isclose(values["cross_matched"], min(values["direct_original"], values["cross_original"])) and
                isclose(values["contrast"], values["matched"] - values["cross_matched"])):
            formula_failures.append((row["chromosome"], row["method"], "matched/cross/contrast"))
        if row["orientation"] == "direct":
            expected = (values["rho_a_mat"] - values["rho_a_pat"], values["rho_b_pat"] - values["rho_b_mat"])
            if row["swap"] != "False" or not (values["direct_original"] > values["cross_original"]):
                orientation_failures.append((row["chromosome"], row["method"], "direct orientation"))
        elif row["orientation"] == "swapped":
            expected = (values["rho_b_mat"] - values["rho_b_pat"], values["rho_a_pat"] - values["rho_a_mat"])
            if row["swap"] != "True" or not (values["cross_original"] > values["direct_original"]):
                orientation_failures.append((row["chromosome"], row["method"], "swapped orientation"))
        else:
            expected = (float("nan"), float("nan"))
        if row["orientation"] in ("direct", "swapped"):
            if not (isclose(values["margin_mat"], expected[0]) and isclose(values["margin_pat"], expected[1]) and
                    isclose(values["minmargin"], min(expected))):
                formula_failures.append((row["chromosome"], row["method"], "margins"))
    check("strict_swap_invariant", not orientation_failures, orientation_failures[:10])
    check("matched_contrast_formulas", not formula_failures, formula_failures[:10])

    with (OUT / "per_copy.tsv").open(encoding="utf-8", newline="") as handle:
        copy_rows = list(csv.DictReader(handle, delimiter="\t"))
    check("per_copy_80_rows", len(copy_rows) == 80, {"rows": len(copy_rows)})
    copy_failures = []
    primary_by_key = {(row["chromosome"], row["method"]): row for row in rows}
    for row in copy_rows:
        primary = primary_by_key.get((row["chromosome"], row["method"]))
        if primary is None:
            copy_failures.append((row["chromosome"], row["method"], "missing primary row"))
            continue
        raw_key = ("rho_a_pat" if row["copy"] == "a" and row["reference_copy"] == "pat" else
                   "rho_a_mat" if row["copy"] == "a" else
                   "rho_b_pat" if row["reference_copy"] == "pat" else "rho_b_mat")
        if not isclose(row["spearman"], primary[raw_key]):
            copy_failures.append((row["chromosome"], row["method"], "per-copy rho"))
        if row["n_pairs"] != primary["n_pairs"] or row["n_bins"] != primary["n_bins"]:
            copy_failures.append((row["chromosome"], row["method"], "per-copy denominator"))
    check("per_copy_matches_primary", not copy_failures, copy_failures[:10])

    masks = read_json(OUT / "masks/common_positions.json")
    mask_chroms = masks.get("chromosomes", {})
    check("common_mask_20_chromosomes", sorted(mask_chroms) == sorted(CHROMS), sorted(mask_chroms))
    mask_failures = []
    for chrom in CHROMS:
        record = mask_chroms.get(chrom, {})
        positions = record.get("common_positions_bp", [])
        n = len(positions)
        if record.get("main_common_bins") != n or record.get("main_common_non_diagonal_pairs") != n * (n - 1) // 2:
            mask_failures.append((chrom, "pair denominator"))
        if record.get("main_common_bins", 0) > record.get("full_grid_bins", 0):
            mask_failures.append((chrom, "common larger than full"))
        if any(position < 3_000_000 or position % 1_000_000 != 0 for position in positions):
            mask_failures.append((chrom, "non-numeric/invalid position"))
    check("mask_pair_denominators", not mask_failures, mask_failures[:10])
    check("main_same_denominator_by_method",
          all(row["n_bins"] == str(mask_chroms[row["chromosome"]]["main_common_bins"])
              and row["n_pairs"] == str(mask_chroms[row["chromosome"]]["main_common_non_diagonal_pairs"])
              for row in rows))

    with (OUT / "masks/mask_denominators.tsv").open(encoding="utf-8", newline="") as handle:
        denominator_rows = list(csv.DictReader(handle, delimiter="\t"))
    check("mask_denominator_120_rows", len(denominator_rows) == 120, {"rows": len(denominator_rows)})
    check("mask_denominator_no_placeholder", all("see_common" not in str(value) for row in denominator_rows for value in row.values()))
    check("mask_denominator_diagonal_flag", all(row["diagonal_excluded"] == "True" for row in denominator_rows))

    matrix_meta = read_json(OUT / "matrices/chr1_scaling_factors.json")
    with np.load(OUT / "matrices/chr1_distance_matrices.npz", allow_pickle=False) as arrays:
        array_keys = sorted(key for key in arrays.files if key.endswith("_normalized"))
        shape_ok = all(arrays[key].shape == (193, 193) for key in arrays.files if key.startswith("map_"))
        diagonal_ok = all(np.isnan(arrays[key].diagonal()).all() for key in arrays.files if key.startswith("map_"))
        symmetry_ok = True
        support_ok = True
        full_positions = arrays["full_positions_bp"]
        common_positions = arrays["common_positions_bp"]
        common_index = {int(position): index for index, position in enumerate(full_positions)}
        common_indices = {common_index[int(position)] for position in common_positions}
        for key in arrays.files:
            if not key.startswith("map_"):
                continue
            matrix = arrays[key]
            finite = np.isfinite(matrix)
            symmetry_ok = symmetry_ok and np.array_equal(finite, finite.T) and np.allclose(
                np.nan_to_num(matrix, nan=0.0), np.nan_to_num(matrix.T, nan=0.0), atol=1e-12)
            for index in range(matrix.shape[0]):
                if index not in common_indices and finite[index].any():
                    support_ok = False
                if index not in common_indices and finite[:, index].any():
                    support_ok = False
    check("chr1_matrix_six_maps", array_keys == [f"map_{i}_normalized" for i in range(1, 7)], array_keys)
    check("chr1_matrix_full_regular_shape", shape_ok, {"shape": [193, 193]})
    check("chr1_matrix_diagonal_masked", diagonal_ok)
    check("chr1_matrix_symmetric", symmetry_ok)
    check("chr1_matrix_missing_support_masked", support_ok)
    check("chr1_matrix_common_bins_188", len(common_positions) == 188)
    check("chr1_matrix_metadata_denominator", matrix_meta.get("n_common_non_diagonal_pairs") == 17578)
    check("chr1_matrix_common_color_limits", finite_number(matrix_meta.get("vmin")) and finite_number(matrix_meta.get("vmax")) and matrix_meta["vmax"] > matrix_meta["vmin"])
    check("chr1_scaling_three_methods", sorted(matrix_meta.get("method_scales", {})) == ["020 multires", "029 C0 direct 1Mb", "Reference"])
    check("chr1_pooled_scaling_each_two_copies", all(item.get("n_copy_matrices") == 2 and item.get("n_pooled_offdiag_values") == 35156
                                                       for item in matrix_meta.get("method_scales", {}).values()))

    bootstrap = np.load(OUT / "bootstrap_index_matrix_seed9301.npy", allow_pickle=False)
    bootstrap_hash = hashlib.sha256(np.ascontiguousarray(bootstrap).tobytes()).hexdigest()
    bootstrap_summary = read_json(OUT / "bootstrap_summary.json")
    check("bootstrap_shape_seed9301", bootstrap.shape == (10000, 20), list(bootstrap.shape))
    check("bootstrap_index_range", bool(bootstrap.min() >= 0 and bootstrap.max() < 20))
    check("bootstrap_hash_matches_summary", bootstrap_hash == bootstrap_summary["bootstrap_index_matrix_sha256"])
    check("bootstrap_primary_20_values", bootstrap_summary["matched_delta_029_minus_020"]["n_chromosomes"] == 20)

    terminal = read_json(OUT / "terminal_verification.json")
    endpoint_failures = []
    for key in ("020_multires_selected", "029_C0_bundle1", "029_C0_bundle2", "029_C0_bundle3"):
        audit = terminal.get("endpoint_audits", {}).get(key, {})
        if audit.get("rows") != 5290 or audit.get("n_tracks") != 40 or not audit.get("all_finite"):
            endpoint_failures.append(key)
    check("terminal_40_track_endpoints", not endpoint_failures, endpoint_failures)
    check("terminal_20_status_complete", terminal.get("020", {}).get("attempt_count") == 6 and
          terminal.get("029_C0", {}).get("all_15_jobs_terminal") is True)
    check("terminal_primary_selection_frozen", terminal.get("020", {}).get("selected_id") == "random_joint" and
          terminal.get("029_C0", {}).get("selected_bundle_id") == "bundle2")
    check("terminal_post_hash_unchanged", all(terminal.get("post_run_hash_unchanged", {}).values()), terminal.get("post_run_hash_unchanged"))

    manifest = read_json(OUT / "input_hash_manifest.json")
    hash_failures = []
    for key, item in manifest.get("hashes", {}).items():
        path = Path(item["path"])
        actual = sha256_file(path)
        if actual != item.get("sha256"):
            hash_failures.append({"key": key, "path": str(path), "expected_manifest": item.get("sha256"), "actual": actual})
        if "expected" in item and actual != item["expected"]:
            hash_failures.append({"key": key, "path": str(path), "expected_input": item["expected"], "actual": actual})
    check("input_manifest_rehash", not hash_failures, hash_failures[:10])
    check("reference_hash_order_recorded", manifest.get("candidate_hashes_completed_before_reference") is True and
          manifest.get("reference_opened_after_candidate_hashes") is True)

    result = {
        "schema_version": "p9016-report-independent-validation-v1",
        "status": "passed" if not failures else "failed",
        "output_dir": str(OUT),
        "checks": checks,
        "failed_checks": failures,
        "summary": {
            "n_primary_chromosome_rows": len(rows),
            "n_per_copy_rows": len(copy_rows),
            "n_chromosomes": len(CHROMS),
            "n_bootstrap_draws": int(bootstrap.shape[0]),
            "chr1_full_grid_bins": int(matrix_meta.get("n_full_grid_bins", -1)),
            "chr1_common_bins": int(matrix_meta.get("n_common_bins", -1)),
            "chr1_common_pairs": int(matrix_meta.get("n_common_non_diagonal_pairs", -1)),
        },
    }
    (OUT / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
