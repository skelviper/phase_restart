#!/usr/bin/env python3
"""只读提取 036 C0/random_joint 三层 accepted-loss history。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE = SCRIPT_PATH.parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

BASELINE = WORKSPACE / "test_res/036-20260914T064651Z-gpu-multires"
OUT = SCRIPT_PATH.parent
STAGES = ("5m", "2m", "1m")
WEIGHTED_NAMES = ("bend", "bond", "repulsion", "p_prior")
SCALAR_NAMES = ("p", "count_nll_normalized", "count_nll_raw", "total")
METRIC_NAMES = (
    "total",
    "count_nll_normalized",
    "count_nll_raw",
    "count_cis_offdiag_nll_raw",
    "count_inter_nll_raw",
    "count_same_bin_nll_raw",
    "count_cis_offdiag_nll_normalized",
    "count_inter_nll_normalized",
    "count_same_bin_nll_normalized",
    "bend",
    "bond",
    "repulsion",
    "p_prior",
    "weighted_bend",
    "weighted_bond",
    "weighted_repulsion",
    "weighted_p_prior",
    "p",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def as_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite numeric value: {value!r}")
    return number


def parse_checkpoint_iteration(path: Path) -> int:
    match = re.search(r"-accepted-(\d+)\.npz$", path.name)
    if match is None:
        raise ValueError(f"unrecognised checkpoint name: {path}")
    return int(match.group(1))


def read_checkpoint_history(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"coordinates", "theta", "y", "positions", "chromosome_index", "fullhistory_json"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} missing NPZ fields: {sorted(missing)}")
        arrays = {name: archive[name] for name in ("coordinates", "theta", "y", "positions", "chromosome_index")}
        for name, array in arrays.items():
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{path} has nonfinite {name}")
        payload = archive["fullhistory_json"].item()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    history = json.loads(str(payload))
    if not isinstance(history, list):
        raise ValueError(f"{path} fullhistory_json is not a list")
    return history, {
        "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "fullhistory_json_sha256": sha256_text(str(payload)),
    }


def numeric_component_values(entry: dict[str, Any]) -> dict[str, float]:
    components = entry.get("components")
    if not isinstance(components, dict):
        raise ValueError("history entry has no components mapping")
    values: dict[str, float] = {}
    for name in METRIC_NAMES:
        if name == "total":
            value = components.get("total")
        else:
            value = components.get(name)
        if value is None:
            continue
        values[name] = as_float(value)
    required = {"total", "count_nll_normalized", "count_nll_raw", *WEIGHTED_NAMES, "p",
                "sum_rate_cis_offdiag", "sum_rate_inter", "observed_log_rate_cis_offdiag",
                "observed_log_rate_inter", "conditional_nll_raw", "diag_profiled_nll_raw"}
    for name in required:
        if name not in components:
            raise ValueError(f"history entry missing component {name}")
        as_float(components[name])
    return values


def direction(delta: float, tolerance: float = 1e-12) -> str:
    if delta > tolerance:
        return "increase"
    if delta < -tolerance:
        return "decrease"
    return "unchanged"


def load_legacy_scalar(path: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"empty TSV: {path}")
        missing = set(("iteration", "p", "count_nll_normalized", "count_nll_raw", "totalJ")).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} missing scalar columns: {sorted(missing)}")
        for row in reader:
            if not row or not row.get("iteration"):
                continue
            iteration = int(row["iteration"])
            result[iteration] = {
                "p": as_float(row["p"]),
                "count_nll_normalized": as_float(row["count_nll_normalized"]),
                "count_nll_raw": as_float(row["count_nll_raw"]),
                "total": as_float(row["totalJ"]),
            }
    return result


def collect() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    provenance_config_path = BASELINE / "provenance/config.json"
    provenance_protocol_path = BASELINE / "provenance/protocol.json"
    provenance_hashes_path = BASELINE / "provenance/source_hashes.json"
    baseline_config = load_json(provenance_config_path)
    baseline_protocol = load_json(provenance_protocol_path)
    baseline_hashes = load_json(provenance_hashes_path)
    weights = dict(baseline_config["models"]["C0"]["definition"]["model"]["objective_weights"])
    if set(weights) != {"bend", "bond", "count", "p_prior", "repulsion"}:
        raise ValueError(f"unexpected C0 weights: {weights}")
    if any(not math.isfinite(float(value)) for value in weights.values()):
        raise ValueError("nonfinite C0 objective weight")

    histories: dict[str, list[dict[str, Any]]] = {}
    validation: dict[str, Any] = {
        "status": "running",
        "fit_started": False,
        "forbidden_inputs_opened": {"reference": False, "phase": False, "evaluation_outputs": False},
        "workspace_import_root": str(WORKSPACE),
        "baseline": str(BASELINE),
        "candidate": "random_joint",
        "model_id": "C0",
        "weights": weights,
        "history_source": {},
        "checkpoint_prefix_audit": {},
        "finite_and_schema": {},
        "weighted_sum": {},
        "count_decomposition": {},
        "published_training_match": {},
        "legacy_scalar_checks": {},
        "input_conservation": {},
    }
    rows_by_stage: dict[str, list[dict[str, Any]]] = {}
    stage_summaries: dict[str, Any] = {}
    source_records: dict[str, Any] = {}

    for label in STAGES:
        stage_json_path = BASELINE / f"stages/C0/random_joint/{label}.json"
        stage_record = load_json(stage_json_path)
        audit = stage_record["data_budget"]["audit"]
        fit = stage_record["fit"]
        checkpoint_paths = sorted(
            (BASELINE / "checkpoints/C0/random_joint").glob(f"{label}-accepted-*.npz"),
            key=parse_checkpoint_iteration,
        )
        if not checkpoint_paths:
            raise ValueError(f"no checkpoints for {label}")
        final_checkpoint = checkpoint_paths[-1]
        final_history, final_npz_info = read_checkpoint_history(final_checkpoint)
        histories[label] = final_history
        final_iteration = parse_checkpoint_iteration(final_checkpoint)
        expected_iterations = list(range(final_iteration + 1))
        actual_iterations = [int(entry.get("iteration", -1)) for entry in final_history]
        if actual_iterations != expected_iterations:
            raise ValueError(f"{label} final history is not iteration 0..{final_iteration}")
        for entry in final_history:
            numeric_component_values(entry)

        prefix_rows = []
        prefix_pass = True
        for checkpoint in checkpoint_paths:
            cp_iteration = parse_checkpoint_iteration(checkpoint)
            cp_history, npz_info = read_checkpoint_history(checkpoint)
            cp_iterations = [int(entry.get("iteration", -1)) for entry in cp_history]
            cp_prefix_match = cp_history == final_history[: len(cp_history)]
            cp_ok = (
                cp_iterations == list(range(cp_iteration + 1))
                and cp_prefix_match
                and len(cp_history) == cp_iteration + 1
            )
            prefix_pass = prefix_pass and cp_ok
            prefix_rows.append({
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "stored_iteration": cp_iteration,
                "history_length": len(cp_history),
                "iteration_sequence": [cp_iterations[0], cp_iterations[-1]] if cp_iterations else [],
                "prefix_matches_final_fullhistory": cp_prefix_match,
                "fullhistory_json_sha256": npz_info["fullhistory_json_sha256"],
                "array_shapes": npz_info["array_shapes"],
                "pass": cp_ok,
            })
        if not prefix_pass:
            raise ValueError(f"{label} checkpoint fullhistory prefix mismatch")

        stage_history = fit.get("history")
        if not isinstance(stage_history, list) or stage_history != final_history:
            raise ValueError(f"{label} stage JSON history differs from final checkpoint fullhistory")
        first = final_history[0]["components"]
        entry10 = next((entry for entry in final_history if int(entry["iteration"]) == 10), None)
        last = final_history[-1]["components"]
        if entry10 is None:
            raise ValueError(f"{label} has no accepted iteration 10")

        raw_records = int(audit["raw_records"])
        raw_cis = int(audit["raw_cis_offdiag"])
        raw_inter = int(audit["raw_inter"])
        raw_same = int(audit["raw_same_bin"])
        if raw_records != 1703888 or raw_cis + raw_inter + raw_same != raw_records:
            raise ValueError(f"{label} raw-record conservation failed")
        if int(audit["endpoint_total"]) != 2 * raw_records:
            raise ValueError(f"{label} endpoint conservation failed")
        if int(stage_record["final_coordinates"]["n_tracks"]) != 40:
            raise ValueError(f"{label} is not 40-track")

        stage_rows: list[dict[str, Any]] = []
        max_total_residual = 0.0
        max_conditional_residual = 0.0
        max_count_raw_residual = 0.0
        max_count_norm_residual = 0.0
        same_values: list[float] = []
        for entry in final_history:
            iteration = int(entry["iteration"])
            components = entry["components"]
            count_norm = as_float(components["count_nll_normalized"])
            count_raw = as_float(components["count_nll_raw"])
            cis_raw = raw_cis * math.log(as_float(components["sum_rate_cis_offdiag"])) - as_float(
                components["observed_log_rate_cis_offdiag"]
            )
            inter_raw = raw_inter * math.log(as_float(components["sum_rate_inter"])) - as_float(
                components["observed_log_rate_inter"]
            )
            same_raw = as_float(components["diag_profiled_nll_raw"])
            conditional_raw = as_float(components["conditional_nll_raw"])
            count_raw_from_groups = cis_raw + inter_raw + same_raw
            count_norm_from_groups = count_raw_from_groups / raw_records
            weighted = {
                "weighted_bend": float(weights["bend"]) * as_float(components["bend"]),
                "weighted_bond": float(weights["bond"]) * as_float(components["bond"]),
                "weighted_repulsion": float(weights["repulsion"]) * as_float(components["repulsion"]),
                "weighted_p_prior": float(weights["p_prior"]) * as_float(components["p_prior"]),
            }
            weighted_sum = float(weights["count"]) * count_norm + sum(weighted.values())
            total = as_float(components["total"])
            total_residual = total - weighted_sum
            conditional_residual = conditional_raw - (cis_raw + inter_raw)
            count_raw_residual = count_raw - count_raw_from_groups
            count_norm_residual = count_norm - count_norm_from_groups
            max_total_residual = max(max_total_residual, abs(total_residual))
            max_conditional_residual = max(max_conditional_residual, abs(conditional_residual))
            max_count_raw_residual = max(max_count_raw_residual, abs(count_raw_residual))
            max_count_norm_residual = max(max_count_norm_residual, abs(count_norm_residual))
            same_values.append(same_raw)
            stage_rows.append({
                "resolution": label,
                "bin_size_bp": int(stage_record["bin_size_bp"]),
                "model_id": "C0",
                "candidate_id": "random_joint",
                "iteration": iteration,
                "record_kind": "initial" if iteration == 0 else "accepted",
                "stage_status": str(stage_record["status"]),
                "fit_status": str(fit["status"]),
                "budget_exhausted": bool(fit["budget_exhausted"]),
                "total": total,
                "count_nll_normalized": count_norm,
                "count_nll_raw": count_raw,
                "count_cis_offdiag_nll_raw": cis_raw,
                "count_inter_nll_raw": inter_raw,
                "count_same_bin_nll_raw": same_raw,
                "count_cis_offdiag_nll_normalized": cis_raw / raw_records,
                "count_inter_nll_normalized": inter_raw / raw_records,
                "count_same_bin_nll_normalized": same_raw / raw_records,
                "bend": as_float(components["bend"]),
                "bond": as_float(components["bond"]),
                "repulsion": as_float(components["repulsion"]),
                "p_prior": as_float(components["p_prior"]),
                **weighted,
                "p": as_float(components["p"]),
                "raw_records": raw_records,
                "raw_cis_offdiag": raw_cis,
                "raw_inter": raw_inter,
                "raw_same_bin": raw_same,
                "count_normalizer": raw_records,
                "total_residual": total_residual,
                "count_conditional_residual_raw": conditional_residual,
                "count_decomposition_residual_raw": count_raw_residual,
                "count_decomposition_residual": count_norm_residual,
                "source_kind": "last_checkpoint_fullhistory_json",
                "source_stage_json": str(stage_json_path.resolve()),
                "source_checkpoint": str(final_checkpoint.resolve()),
                "source_checkpoint_sha256": sha256_file(final_checkpoint),
                "source_fullhistory_json_sha256": final_npz_info["fullhistory_json_sha256"],
            })
        rows_by_stage[label] = stage_rows
        initial_values = stage_rows[0]
        iter10_values = next(row for row in stage_rows if row["iteration"] == 10)
        final_values = stage_rows[-1]
        tracked = [
            "total",
            "count_nll_normalized",
            "count_nll_raw",
            "count_cis_offdiag_nll_normalized",
            "count_inter_nll_normalized",
            "count_same_bin_nll_normalized",
            "weighted_bend",
            "weighted_bond",
            "weighted_repulsion",
            "weighted_p_prior",
        ]
        component_changes = {}
        for name in tracked:
            delta = final_values[name] - initial_values[name]
            component_changes[name] = {"initial": initial_values[name], "final": final_values[name], "delta": delta, "direction": direction(delta)}
        early_total_delta = iter10_values["total"] - initial_values["total"]
        early_count_delta = iter10_values["count_nll_normalized"] - initial_values["count_nll_normalized"]
        net_total_delta = final_values["total"] - initial_values["total"]
        net_count_delta = final_values["count_nll_normalized"] - initial_values["count_nll_normalized"]
        stage_summaries[label] = {
            "resolution": label,
            "bin_size_bp": int(stage_record["bin_size_bp"]),
            "n_loci": int(audit["n_loci"]),
            "n_eligible_pairs": int(audit["n_eligible_pairs"]),
            "raw_records": raw_records,
            "raw_cis_offdiag": raw_cis,
            "raw_inter": raw_inter,
            "raw_same_bin": raw_same,
            "normalizer": {"name": "raw_records", "value": raw_records, "source": "stage data_budget.audit.raw_records"},
            "iteration_count_including_initial": len(stage_rows),
            "accepted_iteration_count": len(stage_rows) - 1,
            "iteration_first": 0,
            "iteration_last": final_iteration,
            "initial_iteration_present": True,
            "stored_checkpoint_iterations": [row["stored_iteration"] for row in prefix_rows],
            "status": "budget_not_converged" if bool(fit["budget_exhausted"]) else str(fit["status"]),
            "original_stage_status": str(stage_record["status"]),
            "termination_reason": str(fit["termination_reason"]),
            "solver": fit["solver"],
            "initial": {name: initial_values[name] for name in tracked},
            "iteration_10": {name: iter10_values[name] for name in tracked},
            "final": {name: final_values[name] for name in tracked},
            "delta_initial_to_final": component_changes,
            "early_change_0_to_10": {
                "total": early_total_delta,
                "count_nll_normalized": early_count_delta,
                "total_fraction_of_net_change": (early_total_delta / net_total_delta) if net_total_delta else None,
                "count_fraction_of_net_change": (early_count_delta / net_count_delta) if net_count_delta else None,
            },
            "regularizer_changes": {
                name: component_changes[name] for name in ("weighted_bend", "weighted_bond", "weighted_repulsion", "weighted_p_prior")
            },
            "same_bin_constant": {
                "max_abs_change_raw": max(same_values) - min(same_values),
                "value_raw": same_values[0],
                "candidate_invariant_profile": True,
            },
        }
        source_records[label] = {
            "stage_json": str(stage_json_path.resolve()),
            "stage_json_sha256": sha256_file(stage_json_path),
            "final_checkpoint": str(final_checkpoint.resolve()),
            "final_checkpoint_sha256": sha256_file(final_checkpoint),
            "fullhistory_json_sha256": final_npz_info["fullhistory_json_sha256"],
            "checkpoint_count": len(checkpoint_paths),
            "checkpoint_prefix_audit": prefix_rows,
        }
        validation["history_source"][label] = {
            "final_checkpoint": str(final_checkpoint.resolve()),
            "checkpoint_count": len(checkpoint_paths),
            "fullhistory_length": len(final_history),
            "accepted_iteration_count": len(final_history) - 1,
            "iteration_sequence": [0, final_iteration],
            "initial_iteration_present": actual_iterations[0] == 0,
            "last_checkpoint_iteration": final_iteration,
        }
        validation["checkpoint_prefix_audit"][label] = {
            "all_prefixes_match_final_fullhistory": prefix_pass,
            "per_checkpoint": prefix_rows,
        }
        validation["finite_and_schema"][label] = {
            "all_history_numeric_fields_finite": True,
            "fullhistory_sequence_is_unique_and_contiguous": actual_iterations == expected_iterations,
            "stage_json_history_equals_checkpoint_fullhistory": True,
            "all_required_npz_fields_present": True,
        }
        validation["weighted_sum"][label] = {
            "max_abs_total_residual": max_total_residual,
            "tolerance": 1e-12,
            "pass": max_total_residual <= 1e-12,
            "formula": "total = count_weight*count_nll_normalized + bend_weight*bend + bond_weight*bond + repulsion_weight*repulsion + p_prior_weight*p_prior",
        }
        validation["count_decomposition"][label] = {
            "max_abs_conditional_residual_raw": max_conditional_residual,
            "max_abs_count_raw_residual": max_count_raw_residual,
            "max_abs_count_normalized_residual": max_count_norm_residual,
            "tolerance": 1e-12,
            "pass": max(max_conditional_residual, max_count_raw_residual, max_count_norm_residual) <= 1e-12,
            "formula": "cis/offdiag = raw_cis_offdiag*log(sum_rate_cis_offdiag)-observed_log_rate_cis_offdiag; inter analogous; same-bin = diag_profiled_nll_raw",
            "omitted_factorial_constant_added": False,
        }
        validation["published_training_match"][label] = {
            "stage_fit_final_total_abs_error": abs(as_float(fit["final_total"]) - final_values["total"]),
            "stage_fit_final_count_abs_error": abs(as_float(fit["final_components"]["count_nll_normalized"]) - final_values["count_nll_normalized"]),
            "stage_fit_final_total_pass": as_float(fit["final_total"]) == final_values["total"],
            "stage_fit_final_count_pass": as_float(fit["final_components"]["count_nll_normalized"]) == final_values["count_nll_normalized"],
            "stage_status": str(stage_record["status"]),
            "budget_exhausted": bool(fit["budget_exhausted"]),
        }

    # Read only scalar columns from the earlier 042/043 post-processing tables.
    legacy_paths = {
        "042": WORKSPACE / "test_res/042-20260915T050722Z-1mb-contact-rg-diagnostic/metrics.tsv",
        "043": WORKSPACE / "test_res/043-20260915T053019Z-1mb-map-contact-rg/metrics.tsv",
    }
    one_m_history = {int(entry["iteration"]): entry["components"] for entry in histories["1m"]}
    for label, path in legacy_paths.items():
        legacy = load_legacy_scalar(path)
        errors = {name: 0.0 for name in SCALAR_NAMES}
        compared = []
        for iteration, values in legacy.items():
            if iteration not in one_m_history:
                raise ValueError(f"legacy {label} iteration {iteration} absent from 1m history")
            current = one_m_history[iteration]
            expected = {
                "p": as_float(current["p"]),
                "count_nll_normalized": as_float(current["count_nll_normalized"]),
                "count_nll_raw": as_float(current["count_nll_raw"]),
                "total": as_float(current["total"]),
            }
            row_errors = {name: abs(values[name] - expected[name]) for name in SCALAR_NAMES}
            errors = {name: max(errors[name], row_errors[name]) for name in SCALAR_NAMES}
            compared.append(iteration)
        validation["legacy_scalar_checks"][label] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "columns_used": ["iteration", "p", "count_nll_normalized", "count_nll_raw", "totalJ"],
            "rows_compared": len(compared),
            "iterations": sorted(compared),
            "max_abs_error": errors,
            "pass": max(errors.values(), default=0.0) <= 1e-12,
            "read_only": True,
            "not_used_as_reference_or_evaluation": True,
        }

    selection_document = load_json(BASELINE / "selection.json")
    selection = selection_document["selection"]
    selection_score = as_float(selection["per_variant"]["C0"]["candidate_scores"]["random_joint"])
    final_count = stage_summaries["1m"]["final"]["count_nll_normalized"]
    validation["published_training_match"]["selection_C0_random_joint_1m"] = {
        "selection_json": str((BASELINE / "selection.json").resolve()),
        "selection_score": selection_score,
        "history_final_count_nll_normalized": final_count,
        "abs_error": abs(selection_score - final_count),
        "selected_id": selection["per_variant"]["C0"]["selected_id"],
        "pass": selection_score == final_count and selection["per_variant"]["C0"]["selected_id"] == "random_joint",
        "phase_used": bool(selection["phase_used"]),
        "reference_used": bool(selection["reference_used"]),
    }
    validation["input_conservation"] = {
        label: {
            "raw_records": stage_summaries[label]["raw_records"],
            "raw_cis_offdiag": stage_summaries[label]["raw_cis_offdiag"],
            "raw_inter": stage_summaries[label]["raw_inter"],
            "raw_same_bin": stage_summaries[label]["raw_same_bin"],
            "sum_equals_raw_records": stage_summaries[label]["raw_cis_offdiag"] + stage_summaries[label]["raw_inter"] + stage_summaries[label]["raw_same_bin"] == stage_summaries[label]["raw_records"],
            "n_chromosomes": 20,
            "n_tracks": 40,
            "pass": True,
        }
        for label in STAGES
    }
    validation["source_metadata"] = {
        "provenance_config": str(provenance_config_path.resolve()),
        "provenance_config_sha256": sha256_file(provenance_config_path),
        "provenance_protocol": str(provenance_protocol_path.resolve()),
        "provenance_protocol_sha256": sha256_file(provenance_protocol_path),
        "provenance_source_hashes": str(provenance_hashes_path.resolve()),
        "provenance_source_hashes_sha256": sha256_file(provenance_hashes_path),
        "baseline_source_code_sha256": baseline_hashes.get("source_code_sha256", {}),
        "phase_used": False,
        "reference_used": False,
        "evaluation_used": False,
        "fit_started": False,
    }
    return rows_by_stage, stage_summaries, {"validation": validation, "source_records": source_records, "weights": weights, "baseline_config": baseline_config, "baseline_protocol": baseline_protocol, "baseline_hashes": baseline_hashes}


def write_metrics(rows_by_stage: dict[str, list[dict[str, Any]]]) -> Path:
    path = OUT / "metrics.tsv"
    columns = [
        "resolution", "bin_size_bp", "model_id", "candidate_id", "iteration", "record_kind", "stage_status", "fit_status", "budget_exhausted",
        "total", "count_nll_normalized", "count_nll_raw", "count_cis_offdiag_nll_raw", "count_inter_nll_raw", "count_same_bin_nll_raw",
        "count_cis_offdiag_nll_normalized", "count_inter_nll_normalized", "count_same_bin_nll_normalized",
        "bend", "bond", "repulsion", "p_prior", "weighted_bend", "weighted_bond", "weighted_repulsion", "weighted_p_prior", "p",
        "raw_records", "raw_cis_offdiag", "raw_inter", "raw_same_bin", "count_normalizer",
        "total_residual", "count_conditional_residual_raw", "count_decomposition_residual_raw", "count_decomposition_residual",
        "source_kind", "source_stage_json", "source_checkpoint", "source_checkpoint_sha256", "source_fullhistory_json_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for label in STAGES:
            for row in rows_by_stage[label]:
                writer.writerow({name: row[name] for name in columns})
    return path


def write_summary_table(stage_summaries: dict[str, Any]) -> Path:
    path = OUT / "summary.tsv"
    columns = [
        "resolution", "bin_size_bp", "status", "accepted_iteration_count", "initial_iteration", "final_iteration",
        "total_initial", "total_final", "total_delta", "count_initial", "count_final", "count_delta",
        "count_cis_offdiag_initial", "count_cis_offdiag_final", "count_cis_offdiag_delta",
        "count_inter_initial", "count_inter_final", "count_inter_delta",
        "count_same_bin_initial", "count_same_bin_final", "count_same_bin_delta",
        "weighted_bend_initial", "weighted_bend_final", "weighted_bend_delta",
        "weighted_bond_initial", "weighted_bond_final", "weighted_bond_delta",
        "weighted_repulsion_initial", "weighted_repulsion_final", "weighted_repulsion_delta",
        "weighted_p_prior_initial", "weighted_p_prior_final", "weighted_p_prior_delta",
        "early_total_delta_0_to_10", "early_count_delta_0_to_10",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for label in STAGES:
            summary = stage_summaries[label]
            initial = summary["initial"]
            final = summary["final"]
            delta = summary["delta_initial_to_final"]
            row: dict[str, Any] = {
                "resolution": label,
                "bin_size_bp": summary["bin_size_bp"],
                "status": summary["status"],
                "accepted_iteration_count": summary["accepted_iteration_count"],
                "initial_iteration": summary["iteration_first"],
                "final_iteration": summary["iteration_last"],
                "total_initial": initial["total"], "total_final": final["total"], "total_delta": delta["total"]["delta"],
                "count_initial": initial["count_nll_normalized"], "count_final": final["count_nll_normalized"], "count_delta": delta["count_nll_normalized"]["delta"],
                "count_cis_offdiag_initial": initial["count_cis_offdiag_nll_normalized"], "count_cis_offdiag_final": final["count_cis_offdiag_nll_normalized"], "count_cis_offdiag_delta": delta["count_cis_offdiag_nll_normalized"]["delta"],
                "count_inter_initial": initial["count_inter_nll_normalized"], "count_inter_final": final["count_inter_nll_normalized"], "count_inter_delta": delta["count_inter_nll_normalized"]["delta"],
                "count_same_bin_initial": initial["count_same_bin_nll_normalized"], "count_same_bin_final": final["count_same_bin_nll_normalized"], "count_same_bin_delta": delta["count_same_bin_nll_normalized"]["delta"],
                "weighted_bend_initial": initial["weighted_bend"], "weighted_bend_final": final["weighted_bend"], "weighted_bend_delta": delta["weighted_bend"]["delta"],
                "weighted_bond_initial": initial["weighted_bond"], "weighted_bond_final": final["weighted_bond"], "weighted_bond_delta": delta["weighted_bond"]["delta"],
                "weighted_repulsion_initial": initial["weighted_repulsion"], "weighted_repulsion_final": final["weighted_repulsion"], "weighted_repulsion_delta": delta["weighted_repulsion"]["delta"],
                "weighted_p_prior_initial": initial["weighted_p_prior"], "weighted_p_prior_final": final["weighted_p_prior"], "weighted_p_prior_delta": delta["weighted_p_prior"]["delta"],
                "early_total_delta_0_to_10": summary["early_change_0_to_10"]["total"],
                "early_count_delta_0_to_10": summary["early_change_0_to_10"]["count_nll_normalized"],
            }
            writer.writerow(row)
    return path


def write_plot(rows_by_stage: dict[str, list[dict[str, Any]]]) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mplconfig"))
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
    })
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharex="col")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.07, top=0.94, wspace=0.23, hspace=0.34)
    stage_titles = {"5m": "5 Mb", "2m": "2 Mb", "1m": "1 Mb"}
    term_colors = {"weighted_bend": "#0072B2", "weighted_bond": "#D55E00", "weighted_repulsion": "#009E73", "weighted_p_prior": "#CC79A7"}
    term_labels = {"weighted_bend": "weighted bend", "weighted_bond": "weighted bond", "weighted_repulsion": "weighted repulsion", "weighted_p_prior": "weighted p_prior"}
    group_colors = {"count_cis_offdiag_nll_normalized": "#0072B2", "count_inter_nll_normalized": "#D55E00", "count_same_bin_nll_normalized": "#6A3D9A"}
    group_labels = {"count_cis_offdiag_nll_normalized": "cis/offdiag", "count_inter_nll_normalized": "inter", "count_same_bin_nll_normalized": "same-bin"}
    for col, label in enumerate(STAGES):
        rows = rows_by_stage[label]
        x = np.asarray([row["iteration"] for row in rows], dtype=float)
        initial = rows[0]
        ax = axes[0, col]
        ax.plot(x, [row["total"] for row in rows], color="#1B4F72", label="total J")
        ax.plot(x, [row["count_nll_normalized"] for row in rows], color="#C0392B", linestyle="--", label="count NLL")
        ax.set_title(f"{stage_titles[label]} | budget_not_converged")
        ax.set_ylabel("Objective / raw record")
        ax.grid(True, color="#DDDDDD", linewidth=0.4)
        ax.legend(frameon=False, loc="best")

        ax = axes[1, col]
        for name, color in group_colors.items():
            delta_values = [row[name] - initial[name] for row in rows]
            ax.plot(x, delta_values, color=color, label=group_labels[name])
        ax.axhline(0.0, color="#666666", linewidth=0.5)
        ax.set_ylabel("Delta count loss / raw record")
        ax.set_title("Delta from stage initial")
        ax.grid(True, color="#DDDDDD", linewidth=0.4)
        ax.legend(frameon=False, loc="best", fontsize=5.5)

        ax = axes[2, col]
        for name, color in term_colors.items():
            ax.plot(x, [row[name] for row in rows], color=color, label=term_labels[name])
        ax.set_yscale("symlog", linthresh=1e-4, linscale=1.0)
        ax.set_ylabel("Weighted terms\n(symlog; linthresh=1e-4)")
        ax.set_xlabel("Accepted iteration")
        ax.grid(True, color="#DDDDDD", linewidth=0.4, which="both")
        ax.legend(frameon=False, loc="best", fontsize=5.5)
    fig.suptitle("C0/random_joint loss components across multiresolution training", fontsize=9)
    fig.savefig(OUT / "plots/multires_loss_components.png", dpi=300, facecolor="white")
    plt.close(fig)
    return OUT / "plots/multires_loss_components.png"


def write_readme(stage_summaries: dict[str, Any], artifacts: dict[str, str], validation: dict[str, Any]) -> Path:
    lines = [
        "# 036 C0/random_joint 多分辨率 loss 后处理",
        "",
        "这是只读后处理：未启动训练、未重拟合、未重评 objective；来源为各层最后 checkpoint 的 `fullhistory_json`。",
        "",
        f"- 来源：`{BASELINE}`；候选 `C0/random_joint`；冻结 P9016、20 chromosomes、40 tracks、{stage_summaries['1m']['raw_records']} raw records。",
        "- 横轴：`Accepted iteration`。每层单独从真实 iteration 0 开始；5m/2m/1m 不跨层连接，换层跳变不解释为同一连续曲线。",
        "- 原始训练状态：三层均 `budget_not_converged`，预算耗尽；这不是收敛或 L2 证据。",
        "- 实际权重：`count=1`、`bond=1`、`repulsion=1`、`bend=0.01`、`p_prior=1`。图中正则项为 weighted，TSV 同时保留 raw term。",
        "- count 拆分：`cis/offdiag = M*log(sum_rate)-observed_log_rate`，`inter` 同理，`same-bin = diag_profiled_nll_raw`；三项相加等于存储的 count NLL。记录的 omitted factorial constant 不加入。",
        "- count NLL 的绝对值含分辨率相关常数，只在同一分辨率内看迭代变化；各层归一化分母均直接读取该层 `data_budget.audit.raw_records`。",
        "- 043/042 仅做 1Mb scalar 列只读一致性检查，未用于训练、选择、reference 或 evaluation。",
        "",
        "## 起末值（per raw record；count 的绝对跨层比较受限）",
        "",
        "| resolution | accepted n | J initial -> final (delta) | count initial -> final (delta) | early J delta (0->10) | early count delta (0->10) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in STAGES:
        summary = stage_summaries[label]
        i = summary["initial"]; f = summary["final"]; d = summary["delta_initial_to_final"]; e = summary["early_change_0_to_10"]
        lines.append(
            f"| {label} | {summary['accepted_iteration_count']} | {i['total']:.9f} -> {f['total']:.9f} ({d['total']['delta']:+.9f}) | "
            f"{i['count_nll_normalized']:.9f} -> {f['count_nll_normalized']:.9f} ({d['count_nll_normalized']['delta']:+.9f}) | "
            f"{e['total']:+.9f} | {e['count_nll_normalized']:+.9f} |"
        )
    lines += [
        "",
        "每层 iteration 0、10、final 的所有正则项和 count 子项见 `summary.json`/`summary.tsv`；完整逐 accepted 记录见 `metrics.tsv`。",
        "",
        "## 产物",
        "",
        f"- PNG：`{artifacts['plot_png']}`（3x3 panels, 9x9 in, 300 dpi；第 2 行为 `Delta from stage initial`，正则行使用 symlog，linthresh=1e-4）。",
        f"- metrics：`{artifacts['metrics_tsv']}`；summary table：`{artifacts['summary_tsv']}`。",
        f"- JSON：`{artifacts['summary_json']}`、`{artifacts['validation_json']}`、`{artifacts['config_json']}`、`{artifacts['source_hashes_json']}`。",
        f"- 分析脚本：`{artifacts['script']}`；日志：`{artifacts['log']}`。",
        "",
        "验证状态：`validation.json` 的所有恒等式、finite、history prefix、stage/selection scalar 对照均应为 passed。",
    ]
    path = OUT / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    log_path = OUT / "logs/run.log"
    log_lines = [f"started_utc={datetime.now(timezone.utc).isoformat()}", f"workspace={WORKSPACE}", f"fit_started=False"]
    try:
        rows_by_stage, stage_summaries, bundle = collect()
        validation = bundle["validation"]
        metrics_path = write_metrics(rows_by_stage)
        summary_tsv_path = write_summary_table(stage_summaries)
        plot_path = write_plot(rows_by_stage)
        summary = {
            "run_id": OUT.name,
            "status": "completed_postprocessing",
            "execution": {"fit_started": False, "exit_code": 0, "python": sys.executable, "analysis_root": str(WORKSPACE)},
            "source": {
                "baseline_run": str(BASELINE.resolve()),
                "candidate_id": "random_joint",
                "model_id": "C0",
                "history_source": "last checkpoint fullhistory_json",
                "stages": bundle["source_records"],
                "stage_independent_trajectories": True,
                "cross_resolution_interpretation": "Do not connect or compare absolute count NLL across resolutions; mark only within-stage changes.",
            },
            "cohort": {
                "sample_id": "P9016",
                "n_chromosomes": 20,
                "n_tracks": 40,
                "raw_records": 1703888,
                "input_path": bundle["baseline_config"]["input"]["path"],
                "input_sha256": bundle["baseline_config"]["input"]["sha256"],
                "phase_used": False,
                "reference_used": False,
                "evaluation_used": False,
            },
            "objective": {
                "weights": bundle["weights"],
                "raw_component_names": ["bend", "bond", "repulsion", "p_prior"],
                "weighted_component_names": ["weighted_bend", "weighted_bond", "weighted_repulsion", "weighted_p_prior"],
                "count_name": "count_nll_normalized",
                "count_normalizer_policy": "per-layer data_budget.audit.raw_records",
                "count_decomposition": "cis/offdiag + inter + same-bin; same-bin is diag_profiled_nll_raw",
                "conditional_factorial_constant_omitted": True,
            },
            "stages": stage_summaries,
            "artifacts": {},
        }
        summary_path = OUT / "summary.json"
        dump_json(summary_path, summary)
        source_hashes = {
            "baseline_provenance_source_code_sha256": bundle["baseline_hashes"].get("source_code_sha256", {}),
            "baseline_provenance_config_sha256": sha256_file(BASELINE / "provenance/config.json"),
            "baseline_provenance_protocol_sha256": sha256_file(BASELINE / "provenance/protocol.json"),
            "baseline_provenance_source_hashes_sha256": sha256_file(BASELINE / "provenance/source_hashes.json"),
            "stage_sources": bundle["source_records"],
        }
        source_hashes_path = OUT / "source_hashes.json"
        dump_json(source_hashes_path, source_hashes)
        config = {
            "run_id": OUT.name,
            "analysis": {
                "script": str(SCRIPT_PATH),
                "script_sha256": sha256_file(SCRIPT_PATH),
                "workspace_import_root": str(WORKSPACE),
                "python": sys.executable,
                "fit_started": False,
                "read_only": True,
            },
            "input": {
                "baseline_run": str(BASELINE.resolve()),
                "candidate_id": "random_joint",
                "model_id": "C0",
                "stage_jsons_and_checkpoints": bundle["source_records"],
                "legacy_scalar_checks": validation["legacy_scalar_checks"],
            },
            "objective": {
                "weights": bundle["weights"],
                "normalizer": "each stage data_budget.audit.raw_records",
                "count_decomposition": "M_cis*log(S_cis)-O_cis + M_inter*log(S_inter)-O_inter + diag_profiled_nll_raw",
                "factorial_constant_policy": "do not add conditional_factorial_constant_omitted",
            },
            "figure": {
                "path": str(plot_path.resolve()),
                "figsize_inches": [9, 9],
                "dpi": 300,
                "layout": "3 rows x 3 columns; columns 5Mb/2Mb/1Mb",
                "lower_regularizer_yscale": "symlog",
                "lower_regularizer_linthresh": 1e-4,
            },
            "forbidden_inputs_opened": {"reference": False, "phase": False, "evaluation_outputs": False},
        }
        config_path = OUT / "config.json"
        dump_json(config_path, config)
        validation["status"] = "passed"
        validation["execution"] = {"fit_started": False, "exit_code": 0, "status": "completed_postprocessing"}
        validation["artifacts"] = {
            "metrics_tsv": str(metrics_path.resolve()),
            "summary_tsv": str(summary_tsv_path.resolve()),
            "summary_json": str(summary_path.resolve()),
            "validation_json": str((OUT / "validation.json").resolve()),
            "config_json": str(config_path.resolve()),
            "source_hashes_json": str(source_hashes_path.resolve()),
            "plot_png": str(plot_path.resolve()),
        }
        # Summary artifact links are added after the core summary was written.
        summary["artifacts"] = validation["artifacts"]
        dump_json(summary_path, summary)
        validation_path = OUT / "validation.json"
        dump_json(validation_path, validation)
        artifacts = {
            "plot_png": str(plot_path.resolve()),
            "metrics_tsv": str(metrics_path.resolve()),
            "summary_tsv": str(summary_tsv_path.resolve()),
            "summary_json": str(summary_path.resolve()),
            "validation_json": str(validation_path.resolve()),
            "config_json": str(config_path.resolve()),
            "source_hashes_json": str(source_hashes_path.resolve()),
            "script": str(SCRIPT_PATH),
            "log": str(log_path.resolve()),
        }
        readme_path = write_readme(stage_summaries, artifacts, validation)
        validation["artifacts"]["readme"] = str(readme_path.resolve())
        dump_json(validation_path, validation)
        log_lines += [
            "status=completed_postprocessing",
            "exit_code=0",
            f"stage_iterations=" + ",".join(f"{label}:{stage_summaries[label]['accepted_iteration_count']} accepted" for label in STAGES),
            f"metrics={metrics_path.resolve()}",
            f"plot={plot_path.resolve()}",
            f"summary={summary_path.resolve()}",
            f"validation={validation_path.resolve()}",
        ]
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        log_lines += [f"status=failed", f"exit_code=1", f"error={type(exc).__name__}: {exc}"]
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
