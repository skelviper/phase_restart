#!/usr/bin/env python3
"""从 run 058 的冻结 JSON 摘要导出配对 TSV，不重算科学指标。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("cis_1_5Mb", "cis_5_20Mb", "cis_20Mb_inf", "inter")
STRUCT_METRICS = ("same", "cross", "contrast", "margin_A", "margin_B", "min_margin")
RAW_REGULARIZERS = ("bend", "bond", "p_prior", "repulsion")
WEIGHTED_REGULARIZERS = ("weighted_bend", "weighted_bond", "weighted_p_prior", "weighted_repulsion")


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def scalar(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value: {value}")
        return repr(value)
    return value


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent columns in {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: scalar(value) for key, value in row.items()} for row in rows)


def structural_values(structural: dict[str, Any], endpoint: str, chromosome: str | None = None) -> dict[str, float]:
    result = structural[endpoint]
    if chromosome is None:
        return {metric: result["macro"][metric] for metric in STRUCT_METRICS}
    per_chr = {entry["chromosome"]: entry for entry in result["per_chromosome"]}
    return {metric: per_chr[chromosome][metric] for metric in STRUCT_METRICS}


def add_structural_pair(
    row: dict[str, Any],
    prefix: str,
    left_label: str,
    right_label: str,
    left: dict[str, float],
    right: dict[str, float],
) -> None:
    for metric in STRUCT_METRICS:
        row[f"{prefix}_{left_label}_{metric}"] = left[metric]
        row[f"{prefix}_{right_label}_{metric}"] = right[metric]
        row[f"{prefix}_delta_{metric}"] = right[metric] - left[metric]


def export_paired_a(final: dict[str, Any], structural: dict[str, Any]) -> list[dict[str, Any]]:
    arm_summaries = {entry["arm_id"]: entry for entry in final["A"]["arms"]}
    selections = {entry["seed"]: entry for entry in final["A"]["selection"]}
    rows: list[dict[str, Any]] = []
    for comparison in final["A"]["comparisons"]:
        control_id = comparison["control"]
        swap_id = comparison["candidate_endpoint"]
        control = arm_summaries[control_id]
        swap = arm_summaries[swap_id]
        candidate = comparison["candidate"]
        selection = selections[comparison["seed"]]
        selected_candidate = next(
            item for item in selection["candidates"] if item["candidate_id"] == candidate["candidate_id"]
        )
        control_arm = load(ROOT / "results" / "A" / "arms" / f"{control_id}.json")
        swap_arm = load(ROOT / "results" / "A" / "arms" / f"{swap_id}.json")
        control_components = control_arm["endpoint"]["components"]
        swap_components = swap_arm["endpoint"]["components"]

        row: dict[str, Any] = {
            "seed": comparison["seed"],
            "candidate_slot": candidate["slot"],
            "candidate_id": candidate["candidate_id"],
            "chromosome": candidate["chromosome"],
            "start_bp": candidate["start_bp"],
            "end_bp": candidate["end_bp"],
            "control_id": control_id,
            "swap_id": swap_id,
            "control_train_count": control["train_count"],
            "swap_train_count": swap["train_count"],
            "train_count_gain_control_minus_swap": control["train_count"] - swap["train_count"],
            "control_train_J": control["train_J"],
            "swap_train_J": swap["train_J"],
            "train_J_gain_control_minus_swap": control["train_J"] - swap["train_J"],
        }
        for regularizer in RAW_REGULARIZERS + WEIGHTED_REGULARIZERS:
            row[f"control_{regularizer}"] = control_components[regularizer]
            row[f"swap_{regularizer}"] = swap_components[regularizer]
        row.update(
            {
                "control_dev_nll": control["dev_nll"],
                "swap_dev_nll": swap["dev_nll"],
                "dev_gain_control_minus_swap": control["dev_nll"] - swap["dev_nll"],
            }
        )
        add_structural_pair(
            row,
            "macro20chr",
            "control",
            "swap",
            structural_values(structural, control_id),
            structural_values(structural, swap_id),
        )
        row["affected_chromosome"] = candidate["chromosome"]
        add_structural_pair(
            row,
            "affected_chr",
            "control",
            "swap",
            structural_values(structural, control_id, candidate["chromosome"]),
            structural_values(structural, swap_id, candidate["chromosome"]),
        )
        row.update(
            {
                "control_active_maxgrad": control["active_maxgrad"],
                "swap_active_maxgrad": swap["active_maxgrad"],
                "control_full_raw_y_maxgrad": control["full_raw_y_maxgrad"],
                "swap_full_raw_y_maxgrad": swap["full_raw_y_maxgrad"],
                "control_FG": control["nfev"],
                "swap_FG": swap["nfev"],
                "control_terminal": control["terminal"],
                "swap_terminal": swap["terminal"],
                "near_control": comparison["return_to_control"]["near_control"],
                "training_gate_passed": selected_candidate["training_gate_passed"],
                "training_selected": selection["selected"],
                "selection_status": selection["status"],
                "identity_status": comparison["identity_status"],
                "R1_direct": comparison["R1_direct"],
            }
        )
        rows.append(row)
    return rows


def export_profile_b(final: dict[str, Any], structural: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = {entry["seed"]: entry for entry in final["B"]["comparisons"]}
    states = {entry["endpoint_id"]: entry for entry in final["B"]["states"]}
    rows: list[dict[str, Any]] = []
    for diagnostic in final["B"]["diagnostics"]:
        seed = diagnostic["seed"]
        profile = load(ROOT / "results" / "B" / f"profile_seed{seed}.json")
        comparison = comparisons[seed]
        original_id = comparison["control"]
        profile_id = comparison["candidate_endpoint"]
        original = states[original_id]
        profiled = states[profile_id]
        row: dict[str, Any] = {
            "seed": seed,
            "original_id": original_id,
            "profile_id": profile_id,
            "p_before": diagnostic["p_before"],
            "p_after": diagnostic["p_after"],
            "original_train_J": original["train_J"],
            "profile_train_J": profiled["train_J"],
            "train_J_gain_original_minus_profile": diagnostic["train_J_gain"],
            "original_train_count": profile["original"]["components"]["count"],
            "profile_train_count": profile["profiled"]["components"]["count"],
            "train_count_gain_original_minus_profile": diagnostic["train_count_gain"],
            "dJ_dp_before": diagnostic["dJ_dp_before"],
            "dJ_dp_after": diagnostic["dJ_dp_after"],
            "dJ_dq_before": diagnostic["dJ_dq_before"],
            "dJ_dq_after": diagnostic["dJ_dq_after"],
            "scalar_unique_calls": diagnostic["scalar_unique_calls"],
            "original_dev_nll": original["dev_nll"],
            "profile_dev_nll": profiled["dev_nll"],
            "dev_gain_original_minus_profile": comparison["dev_gain"],
            "coordinate_triggered": final["B"]["coordinate_triggered"],
            "coordinate_FG": final["B"]["coordinate_training_fg"],
            "diagnostic_only": True,
            "strategy_comparison_3x50": False,
        }
        add_structural_pair(
            row,
            "macro20chr",
            "original",
            "profile",
            structural_values(structural, original_id),
            structural_values(structural, profile_id),
        )
        row.update({"identity_status": comparison["identity_status"], "R1_direct": comparison["R1_direct"]})
        rows.append(row)
    return rows


def export_dev_strata(comparisons: list[dict[str, Any]], dev: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        control_id = comparison["control"]
        candidate_id = comparison["candidate_endpoint"]
        control = dev[control_id]
        candidate = dev[candidate_id]
        candidate_meta = comparison["candidate"] or {}
        for group in GROUPS:
            left = control["groups"][group]
            right = candidate["groups"][group]
            rows.append(
                {
                    "branch": comparison["branch"],
                    "seed": comparison["seed"],
                    "candidate_id": candidate_meta.get("candidate_id"),
                    "control_id": control_id,
                    "candidate_endpoint_id": candidate_id,
                    "group": group,
                    "Nrecords": left["records"],
                    "eligible_pairs": left["eligible_pairs"],
                    "control_group_conditional_nll": left["within_group_conditional_nll"],
                    "candidate_group_conditional_nll": right["within_group_conditional_nll"],
                    "conditional_gain_control_minus_candidate": left["within_group_conditional_nll"]
                    - right["within_group_conditional_nll"],
                    "control_global_contribution": left["global_contribution"],
                    "candidate_global_contribution": right["global_contribution"],
                    "global_contribution_gain_control_minus_candidate": left["global_contribution"]
                    - right["global_contribution"],
                    "comparison_dev_gain_control_minus_candidate": comparison["dev_gain"],
                }
            )
    return rows


def main() -> None:
    final = load(ROOT / "FINAL_RESULTS.json")
    comparisons_doc = load(ROOT / "evaluation" / "comparisons.json")
    dev_doc = load(ROOT / "evaluation" / "dev_results.json")
    structural_doc = load(ROOT / "evaluation" / "structural_results.json")

    a_rows = export_paired_a(final, structural_doc["results"])
    b_rows = export_profile_b(final, structural_doc["results"])
    strata_rows = export_dev_strata(comparisons_doc["comparisons"], dev_doc["results"])
    if (len(a_rows), len(b_rows), len(strata_rows)) != (4, 2, 24):
        raise ValueError(f"unexpected row counts: {(len(a_rows), len(b_rows), len(strata_rows))}")

    write_tsv(ROOT / "paired_A.tsv", a_rows)
    write_tsv(ROOT / "p_profile_B.tsv", b_rows)
    write_tsv(ROOT / "dev_strata_pairs.tsv", strata_rows)


if __name__ == "__main__":
    main()
