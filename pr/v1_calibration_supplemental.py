"""冻结 P3 calibration recovery 的 evaluation-only paired controls。

本模块从不运行优化，也不写入 metrics.json。它在检查已记录的 pre-truth gate 后，才读取已获授权的 synthetic truth；随后使用匹配的 chromosome/fragment masks，将共同的 iter50 candidates 与 deterministic fixed random-field 和 collapse controls 比较。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .v1_calibration import (
    FINAL_BIN,
    FRAGMENT_BINS,
    TIE_TOL,
    _collapse_coordinates,
    _load_truth as _load_truth_artifact,
    _random_field,
    load_layer,
    synthetic_r2,
    synthetic_r3,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "019-20260913_064842-v1-calibration-recovery"
RUN_DIR = ROOT / "test_res" / RUN_NAME
ORIGINAL_RUN = ROOT / "test_res" / "018-20260913_121446-v1-synthetic-calibration"
CONDITIONS = ("A", "B", "C", "D")
BOOTSTRAP_SEED = 9301
BOOTSTRAP_RESAMPLES = 10_000
R2_TIE_TOL = TIE_TOL


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with open(path, "wt") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _append_status(path: Path, value: Mapping[str, Any]) -> None:
    with open(path, "at") as handle:
        handle.write(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False) + "\n")


def _load_candidate(condition: str) -> tuple[np.ndarray, str]:
    training_path = RUN_DIR / "work" / (condition + "_training.json")
    training = json.loads(training_path.read_text())
    candidate_path = RUN_DIR / training["candidate_coordinate_npz"]
    actual_hash = _sha256(candidate_path)
    if actual_hash != training["candidate_coordinate_npz_sha256"]:
        raise RuntimeError("candidate hash changed for %s" % condition)
    with np.load(candidate_path, allow_pickle=False) as payload:
        coordinates = payload["coordinates"].copy()
    return coordinates, actual_hash


def _load_truth(condition: str) -> tuple[np.ndarray, str]:
    path = ORIGINAL_RUN / "eval_truth" / (condition + "_truth_1m.npz")
    truth, _exposure, _metadata = _load_truth_artifact(path)
    return truth, _sha256(path)


def _raw_r2_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    best_scores = []
    contrasts = []
    for row in result["per_chromosome"]:
        direct = _finite(row.get("direct_mean"))
        swapped = _finite(row.get("swapped_mean"))
        if direct is None or swapped is None:
            best = contrast = None
        else:
            best = max(direct, swapped)
            contrast = abs(direct - swapped)
            best_scores.append(best)
            contrasts.append(contrast)
        rows.append({
            "chromosome_index": int(row["chromosome_index"]),
            "chromosome": row["chromosome"],
            "direct_mean": direct,
            "swapped_mean": swapped,
            "best_swap_score": best,
            "best_swap_contrast": contrast,
            "orientation": row.get("orientation"),
            "n_common_finite_pairs": int(row["n_common_finite_pairs"]),
        })
    return {
        "n_chromosomes": len(rows),
        "n_finite": len(best_scores),
        "n_tied_or_unavailable": int(result["n_tied_or_unavailable"]),
        "mean_best_swap_score": float(np.mean(best_scores)) if best_scores else None,
        "mean_best_swap_contrast": float(np.mean(contrasts)) if contrasts else None,
        "per_chromosome": rows,
    }


def _raw_r3_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for row in result["per_chromosome"]:
        details = []
        for detail in row["detail"]:
            details.append({
                "grid_start_bin": int(detail["grid_start_bin"]),
                "grid_start_bp": int(detail["grid_start_bp"]),
                "label": detail.get("label"),
                "direct_mean": _finite(detail.get("direct_mean")),
                "swapped_mean": _finite(detail.get("swapped_mean")),
                "n_pairs": int(detail["n_pairs"]),
            })
        rows.append({
            "chromosome_index": int(row["chromosome_index"]),
            "chromosome": row["chromosome"],
            "n_complete_20mb_fragments": int(row["n_complete_20mb_fragments"]),
            "n_resolved": int(row["n_resolved"]),
            "n_tied_or_unavailable": int(row["n_tied_or_unavailable"]),
            "applicable": bool(row["applicable"]),
            "global_label": row.get("global_label"),
            "frac_consistent": _finite(row.get("frac_consistent")),
            "longest_run": row.get("longest_run"),
            "n_walls": row.get("n_walls"),
            "detail": details,
        })
    return {
        "n_chromosomes": len(rows),
        "n_applicable": int(result["n_applicable"]),
        "n_tied_or_unavailable": int(result["n_tied_or_unavailable"]),
        "mean_frac_consistent": _finite(result.get("mean_frac_consistent")),
        "mean_n_walls": _finite(result.get("mean_n_walls")),
        "per_chromosome": rows,
    }


def _winner(delta: float | None, tolerance: float = R2_TIE_TOL) -> str | None:
    if delta is None:
        return None
    if delta > tolerance:
        return "left"
    if delta < -tolerance:
        return "right"
    return "tie"


def _bootstrap(values: list[float], seed: int) -> dict[str, Any]:
    if not values:
        return {
            "mean_delta": None,
            "ci_low": None,
            "ci_high": None,
            "n_units": 0,
            "unit": "chromosome",
            "seed": seed,
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "technical_only": True,
        }
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean_delta": float(array.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n_units": int(len(array)),
        "unit": "chromosome",
        "seed": seed,
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "technical_only": True,
    }


def _paired_r2(left: Mapping[str, Any], right: Mapping[str, Any], left_name: str,
               right_name: str, seed: int) -> dict[str, Any]:
    rows = []
    contrast_deltas = []
    score_deltas = []
    for left_row, right_row in zip(left["per_chromosome"], right["per_chromosome"], strict=True):
        left_score = _finite(left_row.get("best_swap_score"))
        right_score = _finite(right_row.get("best_swap_score"))
        left_contrast = _finite(left_row.get("best_swap_contrast"))
        right_contrast = _finite(right_row.get("best_swap_contrast"))
        score_delta = (None if left_score is None or right_score is None
                       else left_score - right_score)
        contrast_delta = (None if left_contrast is None or right_contrast is None
                          else left_contrast - right_contrast)
        if score_delta is not None:
            score_deltas.append(score_delta)
        if contrast_delta is not None:
            contrast_deltas.append(contrast_delta)
        rows.append({
            "chromosome_index": int(left_row["chromosome_index"]),
            "chromosome": left_row["chromosome"],
            "left_best_swap_score": left_score,
            "right_best_swap_score": right_score,
            "delta_best_swap_score": score_delta,
            "winner_best_swap_score": _winner(score_delta),
            "left_contrast": left_contrast,
            "right_contrast": right_contrast,
            "delta_left_minus_right": contrast_delta,
            "winner": _winner(contrast_delta),
        })
    wins = sum(row["winner"] == "left" for row in rows)
    losses = sum(row["winner"] == "right" for row in rows)
    ties = sum(row["winner"] == "tie" for row in rows)
    result = _bootstrap(contrast_deltas, seed)
    secondary = _bootstrap(score_deltas, seed)
    result.update({
        "metric": "R2 geometry best-swap contrast",
        "left": left_name,
        "right": right_name,
        "interpretation": "paired matched-minus-swapped contrast; contrast itself is non-negative by construction and raw positivity is not a control win",
        "n_common_chromosomes": len(contrast_deltas),
        "wins_left": int(wins),
        "wins_right": int(losses),
        "ties": int(ties),
        "per_chromosome": rows,
        "secondary_best_swap_score": {
            **secondary,
            "metric": "R2 best-swap score",
            "interpretation": "secondary geometry-agreement sanity check, not the frozen R2 contrast",
        },
    })
    return result


def _r3_fragment_is_resolved(detail: Mapping[str, Any]) -> bool:
    return (
        detail.get("label") is not None
        and _finite(detail.get("direct_mean")) is not None
        and _finite(detail.get("swapped_mean")) is not None
    )


def _common_fragment_r3(left: Mapping[str, Any], right: Mapping[str, Any], left_name: str,
                        right_name: str, seed: int) -> dict[str, Any]:
    rows = []
    deltas = []
    total_common_fragments = 0
    total_common_fragments_applicable = 0
    total_left_excluded = 0
    total_right_excluded = 0
    for left_row, right_row in zip(left["per_chromosome"], right["per_chromosome"], strict=True):
        left_details = left_row["detail"]
        right_details = right_row["detail"]
        if len(left_details) != len(right_details):
            raise RuntimeError("fragment grid mismatch for %s" % left_row["chromosome"])
        common = []
        left_resolved = 0
        right_resolved = 0
        for left_detail, right_detail in zip(left_details, right_details, strict=True):
            left_ok = _r3_fragment_is_resolved(left_detail)
            right_ok = _r3_fragment_is_resolved(right_detail)
            left_resolved += int(left_ok)
            right_resolved += int(right_ok)
            if left_ok and right_ok:
                common.append({
                    "grid_start_bin": int(left_detail["grid_start_bin"]),
                    "left_label": int(left_detail["label"]),
                    "right_label": int(right_detail["label"]),
                })
        n_complete = len(left_details)
        total_common_fragments += len(common)
        total_left_excluded += n_complete - left_resolved
        total_right_excluded += n_complete - right_resolved
        left_global = right_global = None
        left_frac = right_frac = delta = None
        left_walls = right_walls = None
        common_global_tie = False
        applicable = False
        if common:
            left_labels = [item["left_label"] for item in common]
            right_labels = [item["right_label"] for item in common]
            left_counts = {label: left_labels.count(label) for label in (0, 1)}
            right_counts = {label: right_labels.count(label) for label in (0, 1)}
            if left_counts[0] == left_counts[1] or right_counts[0] == right_counts[1]:
                common_global_tie = True
            else:
                left_global = 0 if left_counts[0] > left_counts[1] else 1
                right_global = 0 if right_counts[0] > right_counts[1] else 1
                left_frac = left_counts[left_global] / len(left_labels)
                right_frac = right_counts[right_global] / len(right_labels)
                delta = left_frac - right_frac
                left_walls = sum(
                    first["left_label"] != second["left_label"]
                    for first, second in zip(common, common[1:])
                    if second["grid_start_bin"] - first["grid_start_bin"] == FRAGMENT_BINS
                )
                right_walls = sum(
                    first["right_label"] != second["right_label"]
                    for first, second in zip(common, common[1:])
                    if second["grid_start_bin"] - first["grid_start_bin"] == FRAGMENT_BINS
                )
                applicable = True
                total_common_fragments_applicable += len(common)
                deltas.append(delta)
        rows.append({
            "chromosome_index": int(left_row["chromosome_index"]),
            "chromosome": left_row["chromosome"],
            "n_complete_20mb_fragments": n_complete,
            "n_left_resolved": left_resolved,
            "n_right_resolved": right_resolved,
            "n_common_fragments": len(common),
            "n_common_fragments_after_global_tie_exclusion": len(common) if applicable else 0,
            "n_left_excluded_tie_or_unavailable": n_complete - left_resolved,
            "n_right_excluded_tie_or_unavailable": n_complete - right_resolved,
            "common_fragment_grid_start_bins": [item["grid_start_bin"] for item in common],
            "left_global_label_on_common_mask": left_global,
            "right_global_label_on_common_mask": right_global,
            "left_frac_consistent_on_common_mask": left_frac,
            "right_frac_consistent_on_common_mask": right_frac,
            "delta_left_minus_right": delta,
            "left_n_walls_on_common_adjacent_fragments": left_walls,
            "right_n_walls_on_common_adjacent_fragments": right_walls,
            "common_global_label_tie": common_global_tie,
            "applicable": applicable,
            "winner": _winner(delta, tolerance=1e-12),
        })
    wins = sum(row["winner"] == "left" for row in rows)
    losses = sum(row["winner"] == "right" for row in rows)
    ties = sum(row["winner"] == "tie" for row in rows)
    result = _bootstrap(deltas, seed)
    result.update({
        "metric": "R3 common-mask fragment consistency",
        "left": left_name,
        "right": right_name,
        "interpretation": "paired fractions recomputed on the same fragments resolved by both geometries; raw overall means are not subtracted",
        "n_common_chromosomes": len(deltas),
        "n_common_fragments_total": total_common_fragments,
        "n_common_fragments_after_global_tie_exclusion": total_common_fragments_applicable,
        "n_left_excluded_tie_or_unavailable_total": total_left_excluded,
        "n_right_excluded_tie_or_unavailable_total": total_right_excluded,
        "wins_left": int(wins),
        "wins_right": int(losses),
        "ties_or_nonapplicable_chromosomes": int(ties + len(rows) - len(deltas)),
        "per_chromosome": rows,
    })
    return result


def _raw_control(name: str, coordinates: np.ndarray, truth: np.ndarray, data) -> dict[str, Any]:
    return {
        "control_name": name,
        "r2": _raw_r2_summary(synthetic_r2(coordinates, truth, data)),
        "r3": _raw_r3_summary(synthetic_r3(coordinates, truth, data)),
    }


def _tie_count(metric: Mapping[str, Any]) -> int:
    if "ties_or_nonapplicable_chromosomes" in metric:
        return int(metric["ties_or_nonapplicable_chromosomes"])
    return int(metric["ties"])


def _write_tsvs(run_dir: Path, report: Mapping[str, Any]) -> tuple[Path, Path]:
    summary_path = run_dir / "plots" / "supplemental_controls_summary.tsv"
    per_chromosome_path = run_dir / "plots" / "supplemental_controls_perchromosome.tsv"
    with open(summary_path, "wt") as handle:
        handle.write(
            "comparison\tcondition\tmetric\tn_common_chromosomes\tn_common_fragments_after_global_tie_exclusion\t"
            "mean_delta\tci_low\tci_high\twins_left\twins_right\tties_or_nonapplicable\n")
        for condition, result in report["conditions"].items():
            for comparison, pair in result["paired"].items():
                for metric_key in ("r2", "r3"):
                    metric = pair[metric_key]
                    handle.write("%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%d\t%d\t%d\n" % (
                        comparison, condition, metric["metric"], metric["n_common_chromosomes"],
                        metric.get("n_common_fragments_after_global_tie_exclusion", metric.get("n_common_fragments_total", "")),
                        "" if metric["mean_delta"] is None else "%.17g" % metric["mean_delta"],
                        "" if metric["ci_low"] is None else "%.17g" % metric["ci_low"],
                        "" if metric["ci_high"] is None else "%.17g" % metric["ci_high"],
                        metric["wins_left"], metric["wins_right"],
                        _tie_count(metric),
                    ))
        for comparison, pair in report["cross_condition_B50_C50"].items():
            for metric_key in ("r2", "r3"):
                metric = pair[metric_key]
                handle.write("%s\tB50_vs_C50\t%s\t%d\t%s\t%s\t%s\t%s\t%d\t%d\t%d\n" % (
                    comparison, metric["metric"], metric["n_common_chromosomes"],
                    metric.get("n_common_fragments_after_global_tie_exclusion", metric.get("n_common_fragments_total", "")),
                    "" if metric["mean_delta"] is None else "%.17g" % metric["mean_delta"],
                    "" if metric["ci_low"] is None else "%.17g" % metric["ci_low"],
                    "" if metric["ci_high"] is None else "%.17g" % metric["ci_high"],
                    metric["wins_left"], metric["wins_right"],
                    _tie_count(metric),
                ))
    with open(per_chromosome_path, "wt") as handle:
        handle.write(
            "comparison\tcondition\tmetric\tchromosome\tn_common_fragments\t"
            "left_score_or_fraction\tright_score_or_fraction\tdelta_left_minus_right\t"
            "winner\tapplicable\n")
        for condition, result in report["conditions"].items():
            for comparison, pair in result["paired"].items():
                for metric_key in ("r2", "r3"):
                    metric = pair[metric_key]
                    for row in metric["per_chromosome"]:
                        if metric_key == "r2":
                            left_value = row["left_contrast"]
                            right_value = row["right_contrast"]
                            n_fragments = ""
                        else:
                            left_value = row["left_frac_consistent_on_common_mask"]
                            right_value = row["right_frac_consistent_on_common_mask"]
                            n_fragments = row["n_common_fragments"]
                        handle.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                            comparison, condition, metric["metric"], row["chromosome"], n_fragments,
                            "" if left_value is None else "%.17g" % left_value,
                            "" if right_value is None else "%.17g" % right_value,
                            "" if row["delta_left_minus_right"] is None else "%.17g" % row["delta_left_minus_right"],
                            row["winner"], row.get("applicable", True),
                        ))
        for comparison, pair in report["cross_condition_B50_C50"].items():
            for metric_key in ("r2", "r3"):
                metric = pair[metric_key]
                for row in metric["per_chromosome"]:
                    if metric_key == "r2":
                        left_value = row["left_contrast"]
                        right_value = row["right_contrast"]
                        n_fragments = ""
                    else:
                        left_value = row["left_frac_consistent_on_common_mask"]
                        right_value = row["right_frac_consistent_on_common_mask"]
                        n_fragments = row["n_common_fragments"]
                    handle.write("%s\tB50_vs_C50\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                        comparison, metric["metric"], row["chromosome"], n_fragments,
                        "" if left_value is None else "%.17g" % left_value,
                        "" if right_value is None else "%.17g" % right_value,
                        "" if row["delta_left_minus_right"] is None else "%.17g" % row["delta_left_minus_right"],
                        row["winner"], row.get("applicable", True),
                    ))
    return summary_path, per_chromosome_path


def build_report(run_dir: str | Path = RUN_DIR) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    gate = json.loads((run_dir / "work" / "pretruth_gate.json").read_text())
    if not gate.get("all_pretruth_gates_pass") or not gate.get("truth_read_authorized"):
        raise RuntimeError("supplemental evaluation lacks an authorized pre-truth gate")
    metrics_path = run_dir / "metrics.json"
    metrics_hash = _sha256(metrics_path)
    candidates = {}
    truths = {}
    data = {}
    for condition in CONDITIONS:
        candidates[condition], candidate_hash = _load_candidate(condition)
        truths[condition], truth_hash = _load_truth(condition)
        data[condition] = load_layer(run_dir / "inputs_snapshot" / (condition + "_1mb_layer.npz"))
        candidates[condition] = {
            "coordinates": candidates[condition],
            "sha256": candidate_hash,
        }
        truths[condition] = {
            "coordinates": truths[condition],
            "sha256": truth_hash,
        }

    controls = {}
    paired = {}
    for condition in CONDITIONS:
        truth = truths[condition]["coordinates"]
        condition_data = data[condition]
        geometries = {
            "candidate": candidates[condition]["coordinates"],
            "random_field": _random_field(condition_data, seed=7101),
            "collapse": _collapse_coordinates(truth),
        }
        controls[condition] = {
            name: _raw_control(name, coords, truth, condition_data)
            for name, coords in geometries.items()
        }
        paired[condition] = {
            "candidate_vs_random_field": {
                "r2": _paired_r2(controls[condition]["candidate"]["r2"],
                                 controls[condition]["random_field"]["r2"],
                                 "candidate", "random_field", BOOTSTRAP_SEED),
                "r3": _common_fragment_r3(controls[condition]["candidate"]["r3"],
                                            controls[condition]["random_field"]["r3"],
                                            "candidate", "random_field", BOOTSTRAP_SEED),
            },
            "candidate_vs_collapse": {
                "r2": _paired_r2(controls[condition]["candidate"]["r2"],
                                 controls[condition]["collapse"]["r2"],
                                 "candidate", "collapse", BOOTSTRAP_SEED),
                "r3": _common_fragment_r3(controls[condition]["candidate"]["r3"],
                                            controls[condition]["collapse"]["r3"],
                                            "candidate", "collapse", BOOTSTRAP_SEED),
            },
        }

    # B 和 C 使用相同的 iter50 checkpoint policy，只在共同的
    # chromosome/fragment masks 上比较，而不是相减不匹配的均值。
    cross = {
        "C50_minus_B50": {
            "r2": _paired_r2(controls["C"]["candidate"]["r2"],
                              controls["B"]["candidate"]["r2"],
                              "C50_candidate", "B50_candidate", BOOTSTRAP_SEED),
            "r3": _common_fragment_r3(controls["C"]["candidate"]["r3"],
                                       controls["B"]["candidate"]["r3"],
                                       "C50_candidate", "B50_candidate", BOOTSTRAP_SEED),
        }
    }
    report = {
        "schema": "v1-p3-supplemental-controls-v1",
        "run_id": run_dir.name,
        "evaluation_only": True,
        "pretruth_gate_path": "work/pretruth_gate.json",
        "pretruth_gate_sha256": _sha256(run_dir / "work" / "pretruth_gate.json"),
        "truth_read_after_authorized_gate": True,
        "metrics_sha256_before": metrics_hash,
        "metrics_sha256_after": _sha256(metrics_path),
        "metrics_unchanged": metrics_hash == _sha256(metrics_path),
        "deterministic_controls": {
            "random_field": {
                "seed": 7101,
                "rule": "sphere_forward(default_rng(7101).normal(0, 0.08, size=(2, n_loci, 3)))",
                "same_for_all_conditions_with_same_full_grid": True,
            },
            "collapse": {
                "rule": "midpoint=truth.mean(axis=0); coordinates=stack((midpoint, midpoint), axis=0)",
                "depends_on_synthetic_truth": True,
            },
            "independent_initialization": {
                "included": False,
                "reason": "required fixed random-field and collapse controls already provide the requested evaluation-only controls; no extra warm-start construction was needed",
            },
        },
        "r2_policy": {
            "raw_metric": "per-chromosome direct/swap contrast=max(direct, swapped)-min(direct, swapped)",
            "raw_contrast_nonnegative_by_construction": True,
            "paired_metric": "per-chromosome best_swap_contrast=abs(direct-swapped)",
            "paired_delta": "left best_swap_contrast - right best_swap_contrast",
            "secondary_metric": "per-chromosome best_swap_score=max(direct, swapped)",
            "wins_tolerance": R2_TIE_TOL,
        },
        "r3_policy": {
            "raw_metric": "existing synthetic_r3 global-label fragment consistency",
            "paired_mask": "same complete 20Mb fragment indices resolved by both geometries",
            "paired_denominator": "common resolved fragments per chromosome",
            "ties_excluded": True,
        },
        "bootstrap_policy": {
            "unit": "chromosome",
            "seed": BOOTSTRAP_SEED,
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "interval": "percentile 2.5/97.5",
            "interpretation": "technical/structural only; 20 chromosomes are one cell, not biological replicates",
        },
        "candidate_sha256": {condition: candidates[condition]["sha256"] for condition in CONDITIONS},
        "truth_sha256": {condition: truths[condition]["sha256"] for condition in CONDITIONS},
        "controls": controls,
        "conditions": {
            condition: {"paired": paired[condition]} for condition in CONDITIONS
        },
        "cross_condition_B50_C50": cross,
        "interpretation_boundary": "supplemental fixed-control diagnostics; not preregistered selection criteria, not an optimization trigger, and no L2 claim",
    }
    output = run_dir / "work" / "supplemental_controls.json"
    _write_json(output, report)
    summary_path, per_chromosome_path = _write_tsvs(run_dir, report)
    report["summary_tsv"] = str(summary_path.relative_to(run_dir))
    report["per_chromosome_tsv"] = str(per_chromosome_path.relative_to(run_dir))
    _write_json(output, report)
    _append_status(run_dir / "logs" / "status.jsonl", {
        "event": "supplemental_fixed_control_evaluation_completed",
        "artifact": str(output.relative_to(run_dir)),
        "metrics_unchanged": report["metrics_unchanged"],
        "controls": ["random_field", "collapse"],
        "independent_initialization_included": False,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 supplemental fixed-control evaluation")
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    args = parser.parse_args()
    print(json.dumps(_jsonable(build_report(args.run_dir)), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
