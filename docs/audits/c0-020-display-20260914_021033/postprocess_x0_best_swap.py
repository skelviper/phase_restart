#!/usr/bin/env python3
"""根据已计算的四 rho 行，修正 posthoc x0 诊断。

本后处理脚本不会打开坐标或 reference。它读取已有的独立四 rho 表，推导 x0 自身逐染色体的最佳交换，同时将此前固定 endpoint 方向的读出保留为明确命名的次要结果。
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TOL = 1e-12
BUNDLES = ("bundle1", "bundle2", "bundle3")
METRICS = ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin")
RHO_KEYS = ("rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat")


def finite(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(value)
    return result


def derive(rhos: dict[str, float]) -> dict[str, Any]:
    a_mat, a_pat, b_mat, b_pat = (rhos[key] for key in RHO_KEYS)
    direct = (a_mat + b_pat) / 2.0
    cross = (a_pat + b_mat) / 2.0
    if abs(direct - cross) <= TOL:
        return {
            "orientation": "unresolved_tie",
            "pairing": "tie_average",
            "direct": direct,
            "raw_cross": cross,
            "matched": (direct + cross) / 2.0,
            "cross": (direct + cross) / 2.0,
            "contrast": 0.0,
            "matched_ref1": None,
            "matched_ref2": None,
            "margin_mat": None,
            "margin_pat": None,
            "minmargin": None,
            "geometry_tie": True,
        }
    if direct > cross:
        abcd = (a_mat, a_pat, b_mat, b_pat)
        orientation = "direct"
        matched, other = direct, cross
    else:
        abcd = (b_mat, b_pat, a_mat, a_pat)
        orientation = "swapped"
        matched, other = cross, direct
    a, b, c, d = abcd
    return {
        "orientation": orientation,
        "pairing": "direct" if orientation == "direct" else "cross",
        "direct": direct,
        "raw_cross": cross,
        "matched": matched,
        "cross": other,
        "contrast": matched - other,
        "matched_ref1": a,
        "matched_ref2": d,
        "margin_mat": a - b,
        "margin_pat": d - c,
        "minmargin": min(a - b, d - c),
        "geometry_tie": False,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    source = HERE / "c0_seed_and_x0_diagnostic.tsv"
    fixed_secondary = HERE / "c0_seed_and_x0_fixed_endpoint_direction.tsv"
    input_source = source
    with input_source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if rows and "orientation_locked_to_endpoint" not in rows[0]:
        input_source = fixed_secondary
        with input_source.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 60:
        raise RuntimeError(f"expected 60 existing x0 rows, got {len(rows)}")
    # 以明确名称保留原始的固定 endpoint 方向表。
    fields = list(rows[0])
    write_tsv(fixed_secondary, rows, fields)

    corrected: list[dict[str, Any]] = []
    direction_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"same": 0, "different": 0, "tie": 0})
    for row in rows:
        bundle = row["bundle"]
        endpoint_orientation = row["orientation_locked_to_endpoint"]
        x0_rhos = {key: finite(row[f"x0_{key}"]) for key in RHO_KEYS}
        standard = derive(x0_rhos)
        comparison = "same" if standard["orientation"] == endpoint_orientation else "different"
        if standard["orientation"] == "unresolved_tie":
            comparison = "tie"
        direction_counts[bundle][comparison] += 1
        out: dict[str, Any] = {
            "bundle": bundle,
            "chromosome": row["chromosome"],
            "endpoint_orientation": endpoint_orientation,
            "x0_standard_orientation": standard["orientation"],
            "x0_standard_pairing": standard["pairing"],
            "orientation_relation": comparison,
            "x0_standard_geometry_tie": standard["geometry_tie"],
        }
        for key in RHO_KEYS:
            out[f"endpoint_{key}"] = finite(row[f"endpoint_{key}"])
            out[f"x0_{key}"] = x0_rhos[key]
        for key in METRICS:
            endpoint_value = finite(row[f"endpoint_{key}"])
            x0_value = standard[key]
            out[f"endpoint_{key}"] = endpoint_value
            out[f"x0_standard_{key}"] = x0_value
            out[f"delta_endpoint_minus_x0_standard_{key}"] = None if x0_value is None else endpoint_value - x0_value
            out[f"x0_fixed_endpoint_{key}"] = float(row[f"x0_{key}"]) if row[f"x0_{key}"] not in ("", "None") else None
            out[f"delta_endpoint_minus_x0_fixed_endpoint_{key}"] = float(row[f"delta_endpoint_minus_x0_{key}"]) if row[f"delta_endpoint_minus_x0_{key}"] not in ("", "None") else None
        out["x0_fixed_endpoint_orientation"] = endpoint_orientation
        out["wins_standard_matched"] = int(out["delta_endpoint_minus_x0_standard_matched"] is not None and out["delta_endpoint_minus_x0_standard_matched"] > 0)
        out["wins_standard_contrast"] = int(out["delta_endpoint_minus_x0_standard_contrast"] is not None and out["delta_endpoint_minus_x0_standard_contrast"] > 0)
        corrected.append(out)

    output_fields = ["bundle", "chromosome", "endpoint_orientation", "x0_standard_orientation", "x0_standard_pairing", "orientation_relation", "x0_standard_geometry_tie"]
    output_fields += [f"x0_{key}" for key in RHO_KEYS]
    for key in METRICS:
        output_fields += [f"endpoint_{key}", f"x0_standard_{key}", f"delta_endpoint_minus_x0_standard_{key}", f"x0_fixed_endpoint_{key}", f"delta_endpoint_minus_x0_fixed_endpoint_{key}"]
    output_fields += ["x0_fixed_endpoint_orientation", "wins_standard_matched", "wins_standard_contrast"]
    write_tsv(source, corrected, output_fields)

    macro: dict[str, Any] = {}
    for bundle in BUNDLES:
        bundle_rows = [row for row in corrected if row["bundle"] == bundle]
        macro[bundle] = {
            "n_chromosomes": len(bundle_rows),
            "orientation_different_from_endpoint": int(sum(row["orientation_relation"] == "different" for row in bundle_rows)),
            "orientation_same_as_endpoint": int(sum(row["orientation_relation"] == "same" for row in bundle_rows)),
            "orientation_ties": int(sum(row["orientation_relation"] == "tie" for row in bundle_rows)),
            "standard_endpoint_minus_x0": {key: float(sum(row[f"delta_endpoint_minus_x0_standard_{key}"] for row in bundle_rows) / len(bundle_rows)) for key in METRICS if all(row[f"delta_endpoint_minus_x0_standard_{key}"] is not None for row in bundle_rows)},
            "standard_wins": {
                "matched": int(sum(row["wins_standard_matched"] for row in bundle_rows)),
                "contrast": int(sum(row["wins_standard_contrast"] for row in bundle_rows)),
            },
            "fixed_endpoint_direction_endpoint_minus_x0": {key: float(sum(row[f"delta_endpoint_minus_x0_fixed_endpoint_{key}"] for row in bundle_rows) / len(bundle_rows)) for key in METRICS if all(row[f"delta_endpoint_minus_x0_fixed_endpoint_{key}"] is not None for row in bundle_rows)},
        }
    result = {
        "schema": "p9016-posthoc-x0-independent-best-swap-v1",
        "status": "PASS",
        "source_table": str(source),
        "fixed_endpoint_secondary_table": str(fixed_secondary),
        "rows": len(corrected),
        "bundles": macro,
        "tie_tolerance": TOL,
        "rho_recomputed": False,
        "coordinates_reopened": False,
        "common_mask_changed": False,
        "interpretation": "standard x0 metrics use x0's own per-chromosome best swap with fixed reference columns; fixed-endpoint-direction values are secondary trajectory-aligned diagnostics",
    }
    json_dump(HERE / "x0_best_swap_validation.json", result)

    summary_path = HERE / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["x0_standard_best_swap"] = result
    json_dump(summary_path, summary)
    validation_path = HERE / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["posthoc_x0"] = {
        "rows": 60,
        "seed_count": 3,
        "chromosomes_per_seed": 20,
        "uses_existing_21_condition_mask": True,
        "new_inference_registered": False,
        "no_ci_or_pvalue": True,
        "standard_metric": "x0 own per-chromosome best-swap from already-computed four rho",
        "standard_validation_path": str(HERE / "x0_best_swap_validation.json"),
        "fixed_endpoint_direction_secondary_path": str(fixed_secondary),
        "direction_counts": direction_counts,
        "rho_recomputed": False,
        "coordinates_reopened": False,
    }
    validation["posthoc_x0_correction"] = {"status": "corrected_from_existing_four_rho", "script": str(Path(__file__).resolve())}
    json_dump(validation_path, validation)


if __name__ == "__main__":
    main()
