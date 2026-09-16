#!/usr/bin/env python3
"""推导已锁定的 R2 allele-signal diagnostics，不重新拟合或重新评估。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RAW_FIELDS = ("rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat")
DOUBLE_METRICS = (
    "matched",
    "cross",
    "contrast",
    "matched_ref1",
    "matched_ref2",
    "margin_ref1",
    "margin_ref2",
    "minmargin",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"null", "none", "na", "n/a"}:
            return None
        return float(stripped)
    return float(value)


def is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def json_number(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def ordered_chromosomes(config: dict[str, Any]) -> list[str]:
    return list(config["chromosomes"])


def ordered_conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(config["conditions"])


def condition_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["condition_id"]: item for item in ordered_conditions(config)}


def load_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def load_json_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("chromosomes"), list):
        raise ValueError("r2_results.json lacks chromosomes list")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for chromosome in payload["chromosomes"]:
        chrom = chromosome.get("chromosome")
        for condition_id, record in chromosome.get("conditions", {}).items():
            key = (chrom, condition_id)
            if key in rows:
                raise ValueError(f"Duplicate JSON row: {key}")
            rows[key] = record
    return rows


def load_summary_scope(path: Path, expected_conditions: set[str]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("metric_scope") != "R2 only":
        raise ValueError("r2_summary.json is not marked R2 only")
    observed = set(payload.get("condition_summary", {}))
    if observed != expected_conditions:
        raise ValueError(f"r2_summary condition set mismatch: {sorted(observed)}")
    return payload


def validate_sources(
    config: dict[str, Any],
    tsv_rows: list[dict[str, str]],
    tsv_fields: list[str],
    json_rows: dict[tuple[str, str], dict[str, Any]],
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    del summary_payload
    chroms = ordered_chromosomes(config)
    conditions = ordered_conditions(config)
    condition_ids = {item["condition_id"] for item in conditions}
    expected_keys = {(chrom, condition_id) for chrom in chroms for condition_id in condition_ids}
    required_fields = {
        "chromosome", "condition_id", "display_name", "role", "endpoint_status", "accepted_as",
        "n_copies", "two_copy", "n_bins", "n_total_non_diagonal_pairs", "n_common_pairs",
        "mask_status", "metric_status", "reason", "track_names", "coverage_json", *RAW_FIELDS,
        "direct", "cross", "matched", "other", "contrast", "similarity", "pairing",
        "orientation", "geometry_status", "rho_reasons_json",
    }
    if set(tsv_fields) != required_fields:
        missing = sorted(required_fields - set(tsv_fields))
        extra = sorted(set(tsv_fields) - required_fields)
        raise ValueError(f"TSV fields differ; missing={missing}, extra={extra}")
    if len(tsv_rows) != len(expected_keys):
        raise ValueError(f"Expected {len(expected_keys)} TSV rows, found {len(tsv_rows)}")
    tsv_map: dict[tuple[str, str], dict[str, str]] = {}
    for row in tsv_rows:
        key = (row["chromosome"], row["condition_id"])
        if key in tsv_map:
            raise ValueError(f"Duplicate TSV row: {key}")
        tsv_map[key] = row
    if set(tsv_map) != expected_keys:
        raise ValueError("TSV chromosome/condition keys do not match frozen 20 x 6 scope")
    if set(json_rows) != expected_keys:
        raise ValueError("r2_results.json chromosome/condition keys do not match frozen 20 x 6 scope")

    raw_max_abs_diff = 0.0
    source_derived_max_abs_diff = {name: 0.0 for name in ("direct", "cross", "matched", "contrast")}
    raw_mismatch_count = 0
    source_derived_mismatch_count = 0
    source_orientation_mismatch_count = 0
    source_pairing_mismatch_count = 0
    nonfinite_raw_fields = 0
    display_names_with_newline = 0
    for key in sorted(expected_keys):
        tsv = tsv_map[key]
        record = json_rows[key]
        if "\n" in tsv["display_name"]:
            display_names_with_newline += 1
        info = condition_map(config)[key[1]]
        expected_double = bool(info["double_copy"])
        if (tsv["two_copy"].strip().lower() == "true") != expected_double:
            raise ValueError(f"two_copy mismatch at {key}")
        for field in RAW_FIELDS:
            tsv_value = parse_number(tsv[field])
            json_value = parse_number(record.get(field))
            if tsv_value is None or json_value is None:
                if tsv_value is not None or json_value is not None:
                    raw_mismatch_count += 1
                continue
            if not math.isfinite(tsv_value):
                nonfinite_raw_fields += 1
            difference = abs(tsv_value - json_value)
            raw_max_abs_diff = max(raw_max_abs_diff, difference)
            if difference > 1e-12:
                raw_mismatch_count += 1
        if expected_double and all(is_finite(parse_number(tsv[field])) for field in RAW_FIELDS):
            a0 = parse_number(tsv["rho_A_mat"])
            b0 = parse_number(tsv["rho_A_pat"])
            c0 = parse_number(tsv["rho_B_mat"])
            d0 = parse_number(tsv["rho_B_pat"])
            assert a0 is not None and b0 is not None and c0 is not None and d0 is not None
            direct = (a0 + d0) / 2.0
            cross = (b0 + c0) / 2.0
            expected_fields = {
                "direct": direct,
                "cross": cross,
                "matched": max(direct, cross),
                "contrast": abs(direct - cross),
            }
            for name, expected in expected_fields.items():
                observed = parse_number(record.get(name))
                if observed is None:
                    source_derived_mismatch_count += 1
                    continue
                difference = abs(expected - observed)
                source_derived_max_abs_diff[name] = max(source_derived_max_abs_diff[name], difference)
                if difference > 1e-12:
                    source_derived_mismatch_count += 1
            orientation_tol = float(config["orientation_policy"]["tie_tolerance"])
            if direct - cross > orientation_tol:
                expected_orientation = "direct"
            elif cross - direct > orientation_tol:
                expected_orientation = "swapped"
            else:
                expected_orientation = None
            if expected_orientation is not None:
                if record.get("orientation") != expected_orientation:
                    source_orientation_mismatch_count += 1
                expected_pairing = "direct" if expected_orientation == "direct" else "cross"
                if record.get("pairing") != expected_pairing:
                    source_pairing_mismatch_count += 1

    denominator_by_chromosome: dict[str, dict[str, int]] = {}
    denominator_consistent = True
    for chrom in chroms:
        rows_for_chrom = [tsv_map[(chrom, item["condition_id"])] for item in conditions]
        totals = {int(row["n_total_non_diagonal_pairs"]) for row in rows_for_chrom}
        commons = {int(row["n_common_pairs"]) for row in rows_for_chrom}
        denominator_consistent &= len(totals) == 1 and len(commons) == 1
        denominator_by_chromosome[chrom] = {
            "n_total_non_diagonal_pairs": min(totals),
            "n_common_pairs": min(commons),
        }
    total_pairs = sum(item["n_total_non_diagonal_pairs"] for item in denominator_by_chromosome.values())
    common_pairs = sum(item["n_common_pairs"] for item in denominator_by_chromosome.values())
    if total_pairs != int(config["expected_total_non_diagonal_pairs"]):
        raise ValueError(f"Frozen full denominator total mismatch: {total_pairs}")
    if common_pairs != int(config["expected_total_common_pairs"]):
        raise ValueError(f"Frozen common-mask total mismatch: {common_pairs}")
    if not denominator_consistent:
        raise ValueError("Pair denominators are not common across the six conditions")

    return {
        "tsv_rows": len(tsv_rows),
        "json_rows": len(json_rows),
        "expected_rows": len(expected_keys),
        "chromosomes": len(chroms),
        "conditions": len(condition_ids),
        "display_names_with_newline": display_names_with_newline,
        "raw_max_abs_diff_tsv_vs_json": raw_max_abs_diff,
        "raw_mismatch_count": raw_mismatch_count,
        "source_derived_max_abs_diff": source_derived_max_abs_diff,
        "source_derived_mismatch_count": source_derived_mismatch_count,
        "source_orientation_mismatch_count": source_orientation_mismatch_count,
        "source_pairing_mismatch_count": source_pairing_mismatch_count,
        "nonfinite_raw_fields": nonfinite_raw_fields,
        "common_denominator_across_conditions": denominator_consistent,
        "n_total_non_diagonal_pairs_total": total_pairs,
        "n_common_pairs_total": common_pairs,
        "denominator_by_chromosome": denominator_by_chromosome,
        "r2_summary_metrics_reused": False,
    }


def derive_four_rho(raw: tuple[float | None, float | None, float | None, float | None], tie_tol: float) -> dict[str, Any]:
    a0, b0, c0, d0 = raw
    if not all(is_finite(value) for value in raw):
        return {
            "derived_status": "nonfinite_input",
            "fixed_orientation": "unresolved_nonfinite",
            "geometry_tie": False,
            "direct_original": None,
            "cross_original": None,
            "a": None,
            "b": None,
            "c": None,
            "d": None,
            "matched_ref1": None,
            "matched_ref2": None,
            "matched": None,
            "cross": None,
            "contrast": None,
            "margin_ref1": None,
            "margin_ref2": None,
            "minmargin": None,
            "margin_pattern": "nonfinite",
            "copy_success_eligible": False,
        }
    assert a0 is not None and b0 is not None and c0 is not None and d0 is not None
    direct_original = (a0 + d0) / 2.0
    cross_original = (b0 + c0) / 2.0
    difference = direct_original - cross_original
    if difference > tie_tol:
        orientation = "direct"
        a, b, c, d = a0, b0, c0, d0
        matched = direct_original
        cross = cross_original
        geometry_tie = False
    elif difference < -tie_tol:
        orientation = "swapped"
        # 保持 reference columns 固定；只交换 candidate rows A/B。
        a, b, c, d = c0, d0, a0, b0
        matched = cross_original
        cross = direct_original
        geometry_tie = False
    else:
        orientation = "unresolved_tie"
        # 保留原始四个 rho 的顺序；不任意选择 orientation。
        a, b, c, d = a0, b0, c0, d0
        matched = (direct_original + cross_original) / 2.0
        cross = matched
        geometry_tie = True
    if geometry_tie:
        matched_ref1 = None
        matched_ref2 = None
        margin_ref1 = None
        margin_ref2 = None
        minmargin = None
        contrast = 0.0
        margin_pattern = "geometry_tie_excluded"
    else:
        matched_ref1 = a
        matched_ref2 = d
        margin_ref1 = a - b
        margin_ref2 = d - c
        minmargin = min(margin_ref1, margin_ref2)
        contrast = matched - cross
        if abs(margin_ref1) <= tie_tol or abs(margin_ref2) <= tie_tol:
            margin_pattern = "margin_tie"
        elif margin_ref1 > tie_tol and margin_ref2 > tie_tol:
            margin_pattern = "both_positive"
        elif margin_ref1 < -tie_tol and margin_ref2 < -tie_tol:
            margin_pattern = "both_negative"
        else:
            margin_pattern = "one_negative"
    return {
        "derived_status": "ok",
        "fixed_orientation": orientation,
        "geometry_tie": geometry_tie,
        "direct_original": direct_original,
        "cross_original": cross_original,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "matched_ref1": matched_ref1,
        "matched_ref2": matched_ref2,
        "matched": matched,
        "cross": cross,
        "contrast": contrast,
        "margin_ref1": margin_ref1,
        "margin_ref2": margin_ref2,
        "minmargin": minmargin,
        "margin_pattern": margin_pattern,
        "copy_success_eligible": not geometry_tie,
    }


def build_rows(
    config: dict[str, Any],
    tsv_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    chroms = ordered_chromosomes(config)
    conditions = ordered_conditions(config)
    info_by_condition = condition_map(config)
    source_map = {(row["chromosome"], row["condition_id"]): row for row in tsv_rows}
    output: list[dict[str, Any]] = []
    tie_tol = float(config["orientation_policy"]["tie_tolerance"])
    for chrom in chroms:
        for condition in conditions:
            condition_id = condition["condition_id"]
            source = source_map[(chrom, condition_id)]
            raw = tuple(parse_number(source[field]) for field in RAW_FIELDS)
            expected_double = bool(info_by_condition[condition_id]["double_copy"])
            row: dict[str, Any] = {
                "chromosome": chrom,
                "condition_id": condition_id,
                "display_name": source["display_name"],
                "short_label": condition["short_label"],
                "role": source["role"],
                "endpoint_status": source["endpoint_status"],
                "accepted_as": source["accepted_as"],
                "n_copies": int(source["n_copies"]),
                "two_copy": source["two_copy"].strip().lower() == "true",
                "n_bins": int(source["n_bins"]),
                "n_total_non_diagonal_pairs": int(source["n_total_non_diagonal_pairs"]),
                "n_common_pairs": int(source["n_common_pairs"]),
                "common_pair_fraction": int(source["n_common_pairs"]) / int(source["n_total_non_diagonal_pairs"]),
                "mask_status": source["mask_status"],
                "source_metric_status": source["metric_status"],
                "source_reason": source["reason"] or None,
                "track_names": source["track_names"],
                "rho_A_mat": json_number(raw[0]),
                "rho_A_pat": json_number(raw[1]),
                "rho_B_mat": json_number(raw[2]),
                "rho_B_pat": json_number(raw[3]),
                "source_pairing": source["pairing"] or None,
                "source_orientation": source["orientation"] or None,
                "source_geometry_status": source["geometry_status"] or None,
            }
            if expected_double:
                derived = derive_four_rho(raw, tie_tol)
                row.update(derived)
                row["similarity"] = derived["matched"]
                row["pairing"] = {"direct": "direct", "swapped": "cross", "unresolved_tie": "unresolved_tie"}[derived["fixed_orientation"]]
            else:
                finite_consensus = is_finite(raw[0]) and is_finite(raw[1])
                similarity = (raw[0] + raw[1]) / 2.0 if finite_consensus else None
                row.update({
                    "derived_status": "ok" if finite_consensus else "nonfinite_input",
                    "fixed_orientation": "reference_mean",
                    "geometry_tie": False,
                    "direct_original": None,
                    "cross_original": None,
                    "a": None,
                    "b": None,
                    "c": None,
                    "d": None,
                    "matched_ref1": None,
                    "matched_ref2": None,
                    "matched": None,
                    "cross": None,
                    "contrast": None,
                    "margin_ref1": None,
                    "margin_ref2": None,
                    "minmargin": None,
                    "margin_pattern": "not_applicable_single_consensus",
                    "copy_success_eligible": False,
                    "similarity": similarity,
                    "pairing": "reference_mean",
                })
            output.append(row)
    return output


def summary_stats(values: Iterable[Any]) -> dict[str, Any]:
    finite = [float(value) for value in values if is_finite(value)]
    if not finite:
        return {"n_finite": 0, "mean": None, "median": None, "sd": None, "min": None, "q025": None, "q25": None, "q75": None, "q975": None, "max": None}
    array = np.asarray(finite, dtype=float)
    quantiles = np.quantile(array, [0.025, 0.25, 0.75, 0.975])
    return {
        "n_finite": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "q025": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q75": float(quantiles[2]),
        "q975": float(quantiles[3]),
        "max": float(np.max(array)),
    }


def condition_summaries(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for condition in ordered_conditions(config):
        condition_id = condition["condition_id"]
        subset = [row for row in rows if row["condition_id"] == condition_id]
        two_copy = bool(condition["double_copy"])
        common = {
            "condition_id": condition_id,
            "short_label": condition["short_label"],
            "role": condition["role"],
            "n_chromosomes_expected": len(config["chromosomes"]),
            "n_rows": len(subset),
            "double_copy_applicable": two_copy,
            "similarity": summary_stats(row["similarity"] for row in subset),
            "endpoint_status_counts": dict(sorted(Counter(row["endpoint_status"] for row in subset).items())),
        }
        if not two_copy:
            common.update({
                "reference_mean_similarity": common["similarity"],
                "double_copy_metrics": None,
                "fixed_orientation_counts": None,
                "margin_pattern_counts": None,
                "one_copy_dominance": None,
            })
        else:
            common["double_copy_metrics"] = {metric: summary_stats(row[metric] for row in subset) for metric in DOUBLE_METRICS}
            common["fixed_orientation_counts"] = dict(sorted(Counter(row["fixed_orientation"] for row in subset).items()))
            common["per_reference_summary_excludes_geometry_tie"] = True
            patterns = Counter(row["margin_pattern"] for row in subset)
            finite_rows = [row for row in subset if row["derived_status"] == "ok" and not row["geometry_tie"]]
            margin1 = np.asarray([row["margin_ref1"] for row in finite_rows], dtype=float)
            margin2 = np.asarray([row["margin_ref2"] for row in finite_rows], dtype=float)
            abs1 = np.abs(margin1)
            abs2 = np.abs(margin2)
            dominance = {
                "finite_rows": len(finite_rows),
                "both_positive": int(patterns.get("both_positive", 0)),
                "one_negative": int(patterns.get("one_negative", 0)),
                "both_negative": int(patterns.get("both_negative", 0)),
                "margin_tie": int(patterns.get("margin_tie", 0)),
                "geometry_tie": int(patterns.get("geometry_tie_excluded", 0)),
                "abs_dominant_ref1": int(np.sum(abs1 > abs2 + float(config["orientation_policy"]["tie_tolerance"]))),
                "abs_dominant_ref2": int(np.sum(abs2 > abs1 + float(config["orientation_policy"]["tie_tolerance"]))),
                "abs_dominance_tie": int(np.sum(np.abs(abs1 - abs2) <= float(config["orientation_policy"]["tie_tolerance"]))),
                "mean_signed_margin_ref1": float(np.mean(margin1)) if len(margin1) else None,
                "mean_signed_margin_ref2": float(np.mean(margin2)) if len(margin2) else None,
                "mean_abs_margin_ref1": float(np.mean(abs1)) if len(abs1) else None,
                "mean_abs_margin_ref2": float(np.mean(abs2)) if len(abs2) else None,
                "sum_abs_margin_ref1": float(np.sum(abs1)) if len(abs1) else None,
                "sum_abs_margin_ref2": float(np.sum(abs2)) if len(abs2) else None,
                "absolute_contribution_share_ref1": float(np.sum(abs1) / (np.sum(abs1) + np.sum(abs2))) if (np.sum(abs1) + np.sum(abs2)) else None,
                "absolute_contribution_share_ref2": float(np.sum(abs2) / (np.sum(abs1) + np.sum(abs2))) if (np.sum(abs1) + np.sum(abs2)) else None,
            }
            common["margin_pattern_counts"] = {
                "finite": int(sum(patterns[name] for name in ("both_positive", "one_negative", "both_negative", "margin_tie", "geometry_tie_excluded"))),
                "both_positive": int(patterns.get("both_positive", 0)),
                "one_negative": int(patterns.get("one_negative", 0)),
                "both_negative": int(patterns.get("both_negative", 0)),
                "margin_tie": int(patterns.get("margin_tie", 0)),
                "geometry_tie": int(patterns.get("geometry_tie_excluded", 0)),
                "nonfinite": int(patterns.get("nonfinite", 0)),
            }
            common["one_copy_dominance"] = dominance
        result[condition_id] = common
    return result


def compare_metric(
    label: str,
    left: str,
    right: str,
    metric: str,
    rows: list[dict[str, Any]],
    chromosomes: list[str],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    tie_tol = float(config["bootstrap"]["delta_tie_tolerance"])
    n_boot = int(config["bootstrap"]["n_boot"])
    left_map = {(row["chromosome"], row["condition_id"]): row for row in rows}
    per_chromosome: list[dict[str, Any]] = []
    deltas: list[float] = []
    missing: list[str] = []
    for chrom in chromosomes:
        left_value = left_map[(chrom, left)].get(metric)
        right_value = left_map[(chrom, right)].get(metric)
        if is_finite(left_value) and is_finite(right_value):
            delta = float(left_value) - float(right_value)
            deltas.append(delta)
            status = "ok"
        else:
            delta = None
            missing.append(chrom)
            status = "nonfinite_or_not_applicable"
        per_chromosome.append({
            "chromosome": chrom,
            "left_value": json_number(left_value),
            "right_value": json_number(right_value),
            "delta_left_minus_right": json_number(delta),
            "status": status,
        })
    finite = np.asarray(deltas, dtype=float)
    if finite.size:
        wins = int(np.sum(finite > tie_tol))
        ties = int(np.sum(np.abs(finite) <= tie_tol))
        losses = int(np.sum(finite < -tie_tol))
        indices = rng.integers(0, finite.size, size=(n_boot, finite.size))
        bootstrap_means = np.mean(finite[indices], axis=1)
        ci95 = [float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])]
        mean_delta = float(np.mean(finite))
        median_delta = float(np.median(finite))
    else:
        wins = ties = losses = 0
        ci95 = [None, None]
        mean_delta = median_delta = None
    return {
        "label": label,
        "left_condition": left,
        "right_condition": right,
        "direction": "left_minus_right",
        "metric": metric,
        "n_chromosomes_expected": len(chromosomes),
        "n_chromosomes_used": int(finite.size),
        "missing_chromosomes": missing,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "ci95": ci95,
        "bootstrap": {
            "unit": "chromosome",
            "n_boot": n_boot,
            "seed": int(config["bootstrap"]["seed"]),
            "ci95": ci95,
            "mean": mean_delta,
            "median": median_delta,
            "interpretation": config["bootstrap"]["interpretation"],
        },
        "per_chromosome": per_chromosome,
    }


def paired_comparisons(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    chromosomes = ordered_chromosomes(config)
    result: list[dict[str, Any]] = []
    for spec in config["comparisons"]:
        for metric in DOUBLE_METRICS:
            result.append(compare_metric(spec["label"], spec["left"], spec["right"], metric, rows, chromosomes, config, rng))
    return result


def algebra_checks(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    tie_tol = float(config["orientation_policy"]["tie_tolerance"])
    finite_double = [row for row in rows if row["two_copy"] and row["derived_status"] == "ok"]
    non_tie_double = [row for row in finite_double if not row["geometry_tie"]]
    algebra_identity = 0
    copy_swap_invariant = 0
    reference_swap_invariant = 0
    for row in non_tie_double:
        raw = (row["rho_A_mat"], row["rho_A_pat"], row["rho_B_mat"], row["rho_B_pat"])
        derived = derive_four_rho(raw, tie_tol)
        if abs(derived["contrast"] - (derived["margin_ref1"] + derived["margin_ref2"]) / 2.0) > 1e-12:
            raise AssertionError(f"contrast identity failed at {row['chromosome']} {row['condition_id']}")
        if abs(derived["matched"] - (derived["a"] + derived["d"]) / 2.0) > 1e-12:
            raise AssertionError(f"matched identity failed at {row['chromosome']} {row['condition_id']}")
        if abs(derived["cross"] - (derived["b"] + derived["c"]) / 2.0) > 1e-12:
            raise AssertionError(f"cross identity failed at {row['chromosome']} {row['condition_id']}")
        algebra_identity += 1
        copy_swapped = derive_four_rho((raw[2], raw[3], raw[0], raw[1]), tie_tol)
        for metric in ("matched", "cross", "contrast", "minmargin", "matched_ref1", "matched_ref2", "margin_ref1", "margin_ref2"):
            if abs(float(derived[metric]) - float(copy_swapped[metric])) > 1e-12:
                raise AssertionError(f"copy swap named-metric invariance failed at {row['chromosome']} {row['condition_id']} {metric}")
        copy_swap_invariant += 1
        ref_swapped = derive_four_rho((raw[1], raw[0], raw[3], raw[2]), tie_tol)
        for metric in ("matched", "cross", "contrast", "minmargin"):
            if abs(float(derived[metric]) - float(ref_swapped[metric])) > 1e-12:
                raise AssertionError(f"reference swap scalar invariance failed at {row['chromosome']} {row['condition_id']} {metric}")
        if abs(float(derived["matched_ref1"]) - float(ref_swapped["matched_ref2"])) > 1e-12:
            raise AssertionError(f"reference swap matched-ref exchange failed at {row['chromosome']} {row['condition_id']}")
        if abs(float(derived["matched_ref2"]) - float(ref_swapped["matched_ref1"])) > 1e-12:
            raise AssertionError(f"reference swap matched-ref exchange failed at {row['chromosome']} {row['condition_id']}")
        if abs(float(derived["margin_ref1"]) - float(ref_swapped["margin_ref2"])) > 1e-12:
            raise AssertionError(f"reference swap margin exchange failed at {row['chromosome']} {row['condition_id']}")
        if abs(float(derived["margin_ref2"]) - float(ref_swapped["margin_ref1"])) > 1e-12:
            raise AssertionError(f"reference swap margin exchange failed at {row['chromosome']} {row['condition_id']}")
        reference_swap_invariant += 1

    anchor_raw = (0.1, 0.8, 0.7, 0.2)
    anchor = derive_four_rho(anchor_raw, tie_tol)
    anchor_expected = (0.7, 0.2, 0.1, 0.8)
    anchor_passed = (
        anchor["fixed_orientation"] == "swapped"
        and all(abs(float(anchor[key]) - expected) <= 1e-12 for key, expected in zip(("a", "b", "c", "d"), anchor_expected))
        and abs(float(anchor["margin_ref1"]) - 0.5) <= 1e-12
        and abs(float(anchor["margin_ref2"]) - 0.7) <= 1e-12
    )
    if not anchor_passed:
        raise AssertionError("fixed-reference candidate-row swap anchor failed")
    anchor_candidate_swap = derive_four_rho((anchor_raw[2], anchor_raw[3], anchor_raw[0], anchor_raw[1]), tie_tol)
    if any(abs(float(anchor[key]) - float(anchor_candidate_swap[key])) > 1e-12 for key in ("matched_ref1", "matched_ref2", "margin_ref1", "margin_ref2")):
        raise AssertionError("candidate-label swap changed fixed-reference named margins")
    anchor_reference_swap = derive_four_rho((anchor_raw[1], anchor_raw[0], anchor_raw[3], anchor_raw[2]), tie_tol)
    if abs(float(anchor["margin_ref1"]) - float(anchor_reference_swap["margin_ref2"])) > 1e-12 or abs(float(anchor["margin_ref2"]) - float(anchor_reference_swap["margin_ref1"])) > 1e-12:
        raise AssertionError("reference-label swap did not exchange named margins")

    tie_example = derive_four_rho((0.5, 0.3, 0.7, 0.5), tie_tol)
    tie_passed = (
        tie_example["fixed_orientation"] == "unresolved_tie"
        and tie_example["geometry_tie"]
        and tie_example["margin_pattern"] == "geometry_tie_excluded"
        and not tie_example["copy_success_eligible"]
        and tie_example["matched_ref1"] is None
        and tie_example["matched_ref2"] is None
        and tie_example["margin_ref1"] is None
        and tie_example["margin_ref2"] is None
        and tie_example["minmargin"] is None
        and abs(tie_example["contrast"]) <= tie_tol
    )
    if not tie_passed:
        raise AssertionError("synthetic geometry tie policy failed")
    return {
        "finite_double_rows_checked": len(finite_double),
        "non_tie_double_rows_checked": len(non_tie_double),
        "contrast_equals_half_margin_sum": algebra_identity,
        "copy_label_swap_named_metric_invariant": copy_swap_invariant,
        "reference_label_swap_named_metric_exchange": reference_swap_invariant,
        "fixed_reference_candidate_row_anchor": {
            "passed": True,
            "input": list(anchor_raw),
            "expected_fixed_ref_abcd": list(anchor_expected),
            "margin_ref1": anchor["margin_ref1"],
            "margin_ref2": anchor["margin_ref2"],
            "candidate_label_swap_preserves_named_margins": True,
            "reference_label_swap_exchanges_named_margins": True,
        },
        "synthetic_geometry_tie_policy": {
            "passed": True,
            "input": [0.5, 0.3, 0.7, 0.5],
            "orientation": tie_example["fixed_orientation"],
            "contrast": tie_example["contrast"],
            "margin_pattern": tie_example["margin_pattern"],
            "fixed_reference_values_are_na": True,
            "copy_success_eligible": tie_example["copy_success_eligible"],
        },
        "tie_tolerance": tie_tol,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "chromosome", "condition_id", "short_label", "display_name", "role", "endpoint_status", "accepted_as",
        "n_copies", "two_copy", "n_bins", "n_total_non_diagonal_pairs", "n_common_pairs", "common_pair_fraction",
        "mask_status", "source_metric_status", "source_reason", "track_names",
        "rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat",
        "source_pairing", "source_orientation", "source_geometry_status",
        "derived_status", "fixed_orientation", "pairing", "geometry_tie",
        "a", "b", "c", "d", "matched_ref1", "matched_ref2", "matched", "cross", "contrast",
        "margin_ref1", "margin_ref2", "minmargin", "margin_pattern", "copy_success_eligible", "similarity",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_comparisons_tsv(path: Path, comparisons: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "comparison_label", "left_condition", "right_condition", "metric",
        "n_chromosomes_expected", "n_chromosomes_used", "mean_delta", "median_delta",
        "ci95_low", "ci95_high", "wins", "ties", "losses", "missing_chromosomes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow({
                "comparison_label": comparison["label"],
                "left_condition": comparison["left_condition"],
                "right_condition": comparison["right_condition"],
                "metric": comparison["metric"],
                "n_chromosomes_expected": comparison["n_chromosomes_expected"],
                "n_chromosomes_used": comparison["n_chromosomes_used"],
                "mean_delta": comparison["mean_delta"],
                "median_delta": comparison["median_delta"],
                "ci95_low": comparison["ci95"][0],
                "ci95_high": comparison["ci95"][1],
                "wins": comparison["wins"],
                "ties": comparison["ties"],
                "losses": comparison["losses"],
                "missing_chromosomes": ",".join(comparison["missing_chromosomes"]),
            })


def save_figure(fig: Any, png_path: Path, pdf_path: Path, dpi: int) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_figures(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    plot_cfg = config["plot"]
    dpi = int(plot_cfg["dpi"])
    font_size = float(plot_cfg["base_fontsize_pt"])
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
    })
    double_conditions = [item for item in ordered_conditions(config) if item["double_copy"]]
    condition_ids = [item["condition_id"] for item in double_conditions]
    labels = [item["short_label"] for item in double_conditions]
    by_key = {(row["chromosome"], row["condition_id"]): row for row in rows}
    chroms = ordered_chromosomes(config)
    colors = ["#176b87", "#d05a3a", "#4b8b3b", "#8b5a9e", "#cc8b25"]

    fig, ax = plt.subplots(figsize=(float(plot_cfg["width_inches"]), 4.0))
    x = np.arange(len(condition_ids), dtype=float)
    for index, condition_id in enumerate(condition_ids):
        x_pair = (x[index] - 0.12, x[index] + 0.12)
        for chrom in chroms:
            row = by_key[(chrom, condition_id)]
            if is_finite(row["matched"]) and is_finite(row["cross"]):
                ax.plot(x_pair, [row["matched"], row["cross"]], color="0.55", linewidth=0.45, alpha=0.35, zorder=1)
                ax.scatter([x_pair[0]], [row["matched"]], color="#176b87", s=8, alpha=0.55, linewidths=0, zorder=2)
                ax.scatter([x_pair[1]], [row["cross"]], color="#d05a3a", s=8, alpha=0.55, linewidths=0, zorder=2)
        values_m = [by_key[(chrom, condition_id)]["matched"] for chrom in chroms]
        values_c = [by_key[(chrom, condition_id)]["cross"] for chrom in chroms]
        ax.scatter([x_pair[0], x_pair[1]], [np.nanmean(values_m), np.nanmean(values_c)], color="black", marker="D", s=20, zorder=4, linewidths=0.4, edgecolors="white")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Spearman rho")
    ax.set_title("R2 matched versus cross (20 chromosomes)")
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#176b87", markersize=4, label="matched"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d05a3a", markersize=4, label="cross"),
        Line2D([0], [0], marker="D", color="black", markerfacecolor="black", markersize=4, label="mean"),
    ], frameon=False, ncol=3, loc="upper right")
    save_figure(fig, Path(config["outputs"]["matched_vs_cross_png"]), Path(config["outputs"]["matched_vs_cross_pdf"]), dpi)

    fig, ax = plt.subplots(figsize=(float(plot_cfg["width_inches"]), 4.0))
    for chrom in chroms:
        values = [by_key[(chrom, condition_id)]["contrast"] for condition_id in condition_ids]
        if all(is_finite(value) for value in values):
            ax.plot(x, values, color="0.55", linewidth=0.5, alpha=0.4, zorder=1)
            ax.scatter(x, values, color=colors, s=9, alpha=0.65, linewidths=0, zorder=2)
    means = [np.nanmean([by_key[(chrom, condition_id)]["contrast"] for chrom in chroms]) for condition_id in condition_ids]
    ax.plot(x, means, color="black", marker="D", markersize=4, linewidth=0.9, label="mean", zorder=4)
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xticks(x, labels)
    ax.set_ylabel("matched - cross")
    ax.set_title("R2 contrast by chromosome")
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, Path(config["outputs"]["contrast_pairs_png"]), Path(config["outputs"]["contrast_pairs_pdf"]), dpi)

    fig, ax = plt.subplots(figsize=(float(plot_cfg["width_inches"]), 5.0))
    marker_cycle = ["o", "s", "^", "D", "P"]
    finite_margin_values: list[float] = []
    for condition_id in condition_ids:
        for chrom in chroms:
            row = by_key[(chrom, condition_id)]
            if is_finite(row["margin_ref1"]):
                finite_margin_values.extend([float(row["margin_ref1"]), float(row["margin_ref2"])])
    max_abs = max([abs(value) for value in finite_margin_values] + [0.1]) * 1.08
    for index, condition_id in enumerate(condition_ids):
        x_values = [by_key[(chrom, condition_id)]["margin_ref1"] for chrom in chroms]
        y_values = [by_key[(chrom, condition_id)]["margin_ref2"] for chrom in chroms]
        ax.scatter(x_values, y_values, color=colors[index], marker=marker_cycle[index], s=22, alpha=0.78, edgecolors="white", linewidths=0.25, label=labels[index])
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
    ax.axvline(0.0, color="black", linewidth=0.6, linestyle="--")
    ax.text(0.97, 0.96, "both +", transform=ax.transAxes, ha="right", va="top", fontsize=font_size)
    ax.text(0.03, 0.04, "both - / one -", transform=ax.transAxes, ha="left", va="bottom", fontsize=font_size)
    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("margin ref1 (a-b)")
    ax.set_ylabel("margin ref2 (d-c)")
    ax.set_title("R2 double-copy margins (20 chromosomes per method)")
    ax.grid(color="0.92", linewidth=0.5)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    save_figure(fig, Path(config["outputs"]["margin_scatter_png"]), Path(config["outputs"]["margin_scatter_pdf"]), dpi)


def build_payloads(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    validation: dict[str, Any],
    checks: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
    code_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_manifest: dict[str, Any] = {}
    for name, item in config["source_files"].items():
        path = Path(item["path"])
        source_manifest[name] = {
            "path": str(path),
            "expected_sha256": item["sha256"],
            "actual_sha256": sha256_file(path),
        }
    input_manifest = {
        "schema_version": "p9016-r2-allele-signal-input-manifest-v1",
        "scope": "R2 only",
        "new_fit": False,
        "reads_reference": False,
        "reads_phase": False,
        "reads_contacts": False,
        "numeric_sources": source_manifest,
        "code": {
            "script": str(Path(__file__).resolve()),
            "sha256": code_sha256,
        },
        "parser": "csv.DictReader(delimiter=\\t, newline='')",
        "source_summary_json_metric_values_reused": False,
        "matrix_mask_recomputed": False,
        "validation": validation,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_payload = {
        "schema_version": "p9016-r2-allele-signal-summary-v1",
        "status": "complete",
        "scope": "R2 only",
        "condition_summary": summaries,
        "paired_comparisons": comparisons,
        "bootstrap_policy": config["bootstrap"],
        "orientation_policy": config["orientation_policy"],
        "denominator": {
            "chromosome_order": ordered_chromosomes(config),
            "expected_total_non_diagonal_pairs": config["expected_total_non_diagonal_pairs"],
            "expected_total_common_pairs": config["expected_total_common_pairs"],
            "n_total_non_diagonal_pairs_total": validation["n_total_non_diagonal_pairs_total"],
            "n_common_pairs_total": validation["n_common_pairs_total"],
            "by_chromosome": validation["denominator_by_chromosome"],
        },
        "algebra_and_exchange_checks": checks,
        "future_3seed_single_ablation_reporting": config["future_3seed_single_ablation_reporting"],
        "input_manifest_path": config["outputs"]["input_manifest_json"],
        "code_sha256": code_sha256,
    }
    results_payload = {
        "schema_version": "p9016-r2-allele-signal-results-v1",
        "status": "complete",
        "scope": "R2 only",
        "experiment_id": config["experiment_id"],
        "input_manifest": input_manifest,
        "protocol": {
            "conditions": config["conditions"],
            "chromosomes": ordered_chromosomes(config),
            "expected_total_non_diagonal_pairs": config["expected_total_non_diagonal_pairs"],
            "expected_total_common_pairs": config["expected_total_common_pairs"],
            "raw_fields": list(RAW_FIELDS),
            "no_local_copy_repair": True,
            "fdg_accepted_equals_continuation_not_a_new_condition": True,
        },
        "validation": validation,
        "algebra_and_exchange_checks": checks,
        "condition_summary": summaries,
        "paired_comparisons": comparisons,
        "rows": rows,
        "future_3seed_single_ablation_reporting": config["future_3seed_single_ablation_reporting"],
    }
    return input_manifest, summary_payload, results_payload


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def print_terminal_summary(config: dict[str, Any], summaries: dict[str, dict[str, Any]], comparisons: list[dict[str, Any]], checks: dict[str, Any]) -> None:
    print("R2 allele-signal derived analysis")
    print(f"experiment={config['experiment_id']}")
    print("scope=R2 only; new_fit=false; reads_reference=false; reads_phase=false; reads_contacts=false")
    print("conditions=Softall124101,020original,022continuation,FDGfullrejected,random014,consensus014")
    for condition in ordered_conditions(config):
        summary = summaries[condition["condition_id"]]
        similarity = summary["similarity"]["mean"]
        if condition["double_copy"]:
            metrics = summary["double_copy_metrics"]
            patterns = summary["margin_pattern_counts"]
            dominance = summary["one_copy_dominance"]
            print(
                f"condition={condition['condition_id']} n=20 "
                f"matched_mean={metrics['matched']['mean']:.9f} cross_mean={metrics['cross']['mean']:.9f} "
                f"contrast_mean={metrics['contrast']['mean']:.9f} minmargin_mean={metrics['minmargin']['mean']:.9f} "
                f"matched_ref1_mean={metrics['matched_ref1']['mean']:.9f} matched_ref2_mean={metrics['matched_ref2']['mean']:.9f} "
                f"both_positive={patterns['both_positive']} one_negative={patterns['one_negative']} "
                f"both_negative={patterns['both_negative']} margin_tie={patterns['margin_tie']} geometry_tie={patterns['geometry_tie']} "
                f"abs_dominant_ref1={dominance['abs_dominant_ref1']} abs_dominant_ref2={dominance['abs_dominant_ref2']} "
                f"abs_share_ref1={dominance['absolute_contribution_share_ref1']:.6f}"
            )
        else:
            print(f"condition={condition['condition_id']} n=20 reference_mean_similarity_mean={similarity:.9f}")
    for comparison in comparisons:
        ci = comparison["ci95"]
        print(
            f"comparison={comparison['label']} metric={comparison['metric']} "
            f"mean_delta={comparison['mean_delta']:.9f} median_delta={comparison['median_delta']:.9f} "
            f"wins_ties_losses={comparison['wins']}/{comparison['ties']}/{comparison['losses']} "
            f"ci95=[{ci[0]:.9f},{ci[1]:.9f}]"
        )
    print(
        "checks="
        f"algebra:{checks['contrast_equals_half_margin_sum']} "
        f"copy_swap:{checks['copy_label_swap_named_metric_invariant']} "
        f"reference_swap:{checks['reference_label_swap_named_metric_exchange']} "
        f"tie_policy:{checks['synthetic_geometry_tie_policy']['passed']}"
    )
    print("analysis completed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    code_sha256 = sha256_file(Path(__file__).resolve())
    source_paths = {name: Path(item["path"]) for name, item in config["source_files"].items()}
    actual_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    for name, item in config["source_files"].items():
        if actual_hashes[name] != item["sha256"]:
            raise ValueError(f"Input SHA256 mismatch for {name}: {actual_hashes[name]}")
    tsv_rows, tsv_fields = load_tsv(source_paths["r2_per_chromosome_tsv"])
    json_rows = load_json_rows(source_paths["r2_results_json"])
    summary_payload = load_summary_scope(source_paths["r2_summary_json"], {item["condition_id"] for item in config["conditions"]})
    validation = validate_sources(config, tsv_rows, tsv_fields, json_rows, summary_payload)
    rows = build_rows(config, tsv_rows)
    checks = algebra_checks(config, rows)
    summaries = condition_summaries(config, rows)
    comparisons = paired_comparisons(config, rows)
    input_manifest, summary_out, results_out = build_payloads(config, rows, validation, checks, summaries, comparisons, code_sha256)

    output_cfg = config["outputs"]
    write_tsv(Path(output_cfg["per_chromosome_tsv"]), rows)
    write_comparisons_tsv(Path(output_cfg["paired_comparisons_tsv"]), comparisons)
    dump_json(Path(output_cfg["input_manifest_json"]), input_manifest)
    dump_json(Path(output_cfg["summary_json"]), summary_out)
    dump_json(Path(output_cfg["results_json"]), results_out)
    plot_figures(config, rows)
    print_terminal_summary(config, summaries, comparisons, checks)


if __name__ == "__main__":
    main()
