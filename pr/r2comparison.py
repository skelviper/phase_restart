"""专用 R2 比较、汇总和图形辅助函数。

本模块只用于评估，并有意接收已经加载的 coordinate maps。它不打开带 phase 的 pairs、reference 或 training metadata。正式 runner 负责在读取 reference 前哈希并 arm 这些输入。单条染色体的所有坐标都在一份有限的 upper-triangle mask 上评估；所有提供的 condition 和两条 reference tracks 共享这份 mask。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .score import spearman


BIN_SIZE_BP = 1_000_000
GRID_OFFSET_BP = 3_000_000
MIN_RHO_PAIRS = 20
GEOMETRY_TIE_TOL = 1e-12
BOOTSTRAP_SEED = 9301
BOOTSTRAP_DRAWS = 10_000
SCHEMA_VERSION = "p9016-r2-comparison-v1"


@dataclass(frozen=True)
class R2Condition:
    """提供给纯 R2 evaluator 的一项 coordinate condition。

    ``n_copies`` 在 014 consensus control 中为 1，在每个 main condition 和 random control 中为 2。``endpoint_status`` 和 ``accepted_as`` 只是 provenance labels；任何 R2 值都不会改变 acceptance。
    """

    condition_id: str
    display_name: str
    structures: Mapping[str, Mapping[int, np.ndarray]]
    n_copies: int = 2
    role: str = "main"
    endpoint_status: str = "accepted"
    accepted_as: str | None = None

    def __post_init__(self) -> None:
        if self.n_copies not in (1, 2):
            raise ValueError("R2Condition.n_copies must be 1 or 2")
        if not self.condition_id:
            raise ValueError("R2Condition.condition_id must be non-empty")


@dataclass(frozen=True)
class CommonMask:
    """有限的无序非对角 pair 集合及其 coverage 证据。"""

    chromosome: str
    positions: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    common: np.ndarray
    distances: Mapping[str, np.ndarray]
    track_coverage: Mapping[str, Mapping[str, Any]]
    status: str

    @property
    def n_bins(self) -> int:
        return int(len(self.positions))

    @property
    def n_total_pairs(self) -> int:
        return int(len(self.pair_i))

    @property
    def n_common_pairs(self) -> int:
        return int(self.common.sum())

    def metadata(self) -> dict[str, Any]:
        return {
            "chromosome": self.chromosome,
            "bin_size_bp": BIN_SIZE_BP,
            "offset_bp": GRID_OFFSET_BP,
            "n_bins": self.n_bins,
            "n_total_non_diagonal_pairs": self.n_total_pairs,
            "n_common_pairs": self.n_common_pairs,
            "status": self.status,
            "track_coverage": _jsonable(self.track_coverage),
        }


def _jsonable(value: Any) -> Any:
    """将 arrays 和非有限数转换为严格 JSON 输出。"""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def grid_positions(length_bp: int, *, bin_size_bp: int = BIN_SIZE_BP,
                   offset_bp: int = GRID_OFFSET_BP) -> np.ndarray:
    """返回一条染色体的完整 legacy metric grid。"""
    if int(length_bp) <= int(offset_bp):
        return np.empty(0, dtype=np.int64)
    return np.arange(int(offset_bp), int(length_bp), int(bin_size_bp), dtype=np.int64)


def default_track_names(chromosome_index: int, n_copies: int) -> tuple[str, ...]:
    """返回一条染色体的 canonical ``c01a``/``c01b`` 名称。"""
    if n_copies not in (1, 2):
        raise ValueError("n_copies must be 1 or 2")
    prefix = "c%02d" % (int(chromosome_index) + 1)
    return tuple(prefix + suffix for suffix in ("a", "b")[:n_copies])


def _dense_points(structures: Mapping[str, Mapping[int, np.ndarray]], track: str,
                  positions: np.ndarray) -> np.ndarray:
    points = np.full((len(positions), 3), np.nan, dtype=float)
    rows = structures.get(track)
    if not isinstance(rows, Mapping):
        return points
    for index, position in enumerate(positions):
        value = rows.get(int(position))
        if value is None:
            value = rows.get(str(int(position)))
        if value is None:
            continue
        try:
            point = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if point.shape == (3,) and np.isfinite(point).all():
            points[index] = point
    return points


def _distance_vector(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    out = np.full(len(pair_i), np.nan, dtype=float)
    if not len(pair_i):
        return out
    finite = np.isfinite(points[pair_i]).all(axis=1) & np.isfinite(points[pair_j]).all(axis=1)
    if finite.any():
        delta = points[pair_i[finite]] - points[pair_j[finite]]
        out[finite] = np.sqrt((delta * delta).sum(axis=1))
    return out


def _condition_value(condition: R2Condition | Mapping[str, Any]) -> R2Condition:
    if isinstance(condition, R2Condition):
        return condition
    if not isinstance(condition, Mapping):
        raise TypeError("conditions must contain R2Condition or mapping values")
    fields = dict(condition)
    if "condition_id" not in fields and "id" in fields:
        fields["condition_id"] = fields.pop("id")
    if "display_name" not in fields:
        fields["display_name"] = fields["condition_id"]
    return R2Condition(**fields)


def _normalise_conditions(
    conditions: Mapping[str, R2Condition | Mapping[str, Any]] | Sequence[R2Condition],
) -> dict[str, R2Condition]:
    if isinstance(conditions, Mapping):
        result = {str(key): _condition_value(value) for key, value in conditions.items()}
    else:
        result = {item.condition_id: item for item in conditions}
    if not result:
        raise ValueError("at least one R2 condition is required")
    for key, value in result.items():
        if key != value.condition_id:
            raise ValueError("condition mapping key disagrees with condition_id: %s" % key)
    return result


def _track_key(condition_id: str, copy_name: str) -> str:
    return "condition:%s:%s" % (condition_id, copy_name)


def build_common_mask(
    chromosome: str,
    length_bp: int,
    conditions: Mapping[str, R2Condition | Mapping[str, Any]] | Sequence[R2Condition],
    reference: Mapping[str, Mapping[int, np.ndarray]],
    *,
    chromosome_index: int = 0,
    condition_tracks: Mapping[str, Sequence[str]] | None = None,
    reference_tracks: Sequence[str] | None = None,
) -> CommonMask:
    """构建所有 condition 共同的有限非对角 pair mask。

    每个 condition 的轨迹和两条 reference 轨迹都参与 mask，包括单轨 controls 和被拒绝的 proposal。因此缺失轨迹或 bead 会减少共同 pair count，但绝不会从结果中删除某条染色体。
    """
    normalised = _normalise_conditions(conditions)
    positions = grid_positions(length_bp)
    pair_i, pair_j = np.triu_indices(len(positions), k=1)
    if reference_tracks is None:
        reference_tracks = ("%s(mat)" % chromosome, "%s(pat)" % chromosome)
    if len(reference_tracks) != 2:
        raise ValueError("reference_tracks must contain maternal and paternal tracks")
    tracks_by_condition: dict[str, tuple[str, ...]] = {}
    if condition_tracks is None:
        condition_tracks = {}
    for condition_id, condition in normalised.items():
        raw = condition_tracks.get(condition_id)
        tracks = tuple(raw) if raw is not None else default_track_names(chromosome_index, condition.n_copies)
        if len(tracks) != condition.n_copies:
            raise ValueError("condition %s has %d tracks but n_copies=%d" % (
                condition_id, len(tracks), condition.n_copies))
        if any(not isinstance(track, str) or not track for track in tracks):
            raise ValueError("condition %s has an invalid track name" % condition_id)
        tracks_by_condition[condition_id] = tracks

    points_by_key: dict[str, np.ndarray] = {}
    coverage: dict[str, dict[str, Any]] = {}
    for condition_id, condition in normalised.items():
        for copy_index, track in enumerate(tracks_by_condition[condition_id]):
            key = _track_key(condition_id, "copy%s" % ("AB"[copy_index]))
            points = _dense_points(condition.structures, track, positions)
            points_by_key[key] = points
            n_finite = int(np.isfinite(points).all(axis=1).sum())
            coverage[key] = {
                "condition_id": condition_id,
                "copy": "AB"[copy_index],
                "track": track,
                "n_expected_bins": int(len(positions)),
                "n_finite_bins": n_finite,
                "n_missing_bins": int(len(positions) - n_finite),
                "status": "ok" if n_finite == len(positions) else (
                    "missing_track" if n_finite == 0 else "partial_coordinate_coverage"
                ),
            }
    for suffix, track in zip(("mat", "pat"), reference_tracks):
        key = "reference:%s" % suffix
        points = _dense_points(reference, track, positions)
        points_by_key[key] = points
        n_finite = int(np.isfinite(points).all(axis=1).sum())
        coverage[key] = {
            "condition_id": "reference",
            "copy": suffix,
            "track": track,
            "n_expected_bins": int(len(positions)),
            "n_finite_bins": n_finite,
            "n_missing_bins": int(len(positions) - n_finite),
            "status": "ok" if n_finite == len(positions) else (
                "missing_track" if n_finite == 0 else "partial_coordinate_coverage"
            ),
        }

    distances = {
        key: _distance_vector(points, pair_i, pair_j)
        for key, points in points_by_key.items()
    }
    common = np.ones(len(pair_i), dtype=bool)
    for value in distances.values():
        common &= np.isfinite(value)
    for key, value in distances.items():
        coverage[key]["n_finite_pairs"] = int(np.isfinite(value).sum())
        coverage[key]["n_missing_pairs"] = int(len(value) - np.isfinite(value).sum())
    if len(pair_i) < MIN_RHO_PAIRS or int(common.sum()) < MIN_RHO_PAIRS:
        status = "insufficient_common_pairs"
    elif any(item["status"] != "ok" for item in coverage.values()):
        status = "partial_coordinate_coverage"
    else:
        status = "ok"
    return CommonMask(chromosome, positions, pair_i, pair_j, common, distances, coverage, status)


def _rho_with_reason(x: np.ndarray, y: np.ndarray) -> tuple[float | None, str | None]:
    if len(x) < MIN_RHO_PAIRS:
        return None, "insufficient_common_pairs"
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None, "nonfinite_distance_vector"
    if np.ptp(x) == 0:
        return None, "constant_candidate_distance"
    if np.ptp(y) == 0:
        return None, "constant_reference_distance"
    value = spearman(x, y)
    if not np.isfinite(value):
        return None, "nonfinite_spearman"
    return float(value), None


def _row_base(condition: R2Condition, mask: CommonMask, tracks: Sequence[str]) -> dict[str, Any]:
    return {
        "chromosome": mask.chromosome,
        "condition_id": condition.condition_id,
        "display_name": condition.display_name,
        "role": condition.role,
        "endpoint_status": condition.endpoint_status,
        "accepted_as": condition.accepted_as,
        "n_copies": int(condition.n_copies),
        "two_copy": bool(condition.n_copies == 2),
        "track_names": list(tracks),
        "n_bins": mask.n_bins,
        "n_total_non_diagonal_pairs": mask.n_total_pairs,
        "n_common_pairs": mask.n_common_pairs,
        "mask_status": mask.status,
        "coverage": _jsonable(mask.track_coverage),
        "rho_A_mat": None,
        "rho_A_pat": None,
        "rho_B_mat": None,
        "rho_B_pat": None,
        "direct": None,
        "cross": None,
        "matched": None,
        "other": None,
        "contrast": None,
        "similarity": None,
        "pairing": None,
        "orientation": None,
        "geometry_status": "not_applicable",
        "metric_status": "unavailable",
        "reason": None,
        "rho_reasons": {},
    }


def _evaluate_two_copy(row: dict[str, Any], mask: CommonMask, condition_id: str) -> dict[str, Any]:
    common = mask.common
    values = {
        "rho_A_mat": (mask.distances[_track_key(condition_id, "copyA")][common],
                      mask.distances["reference:mat"][common]),
        "rho_A_pat": (mask.distances[_track_key(condition_id, "copyA")][common],
                      mask.distances["reference:pat"][common]),
        "rho_B_mat": (mask.distances[_track_key(condition_id, "copyB")][common],
                      mask.distances["reference:mat"][common]),
        "rho_B_pat": (mask.distances[_track_key(condition_id, "copyB")][common],
                      mask.distances["reference:pat"][common]),
    }
    for key, (candidate_dist, reference_dist) in values.items():
        rho, reason = _rho_with_reason(candidate_dist, reference_dist)
        row[key] = rho
        if reason is not None:
            row["rho_reasons"][key] = reason
    rho = {key: row[key] for key in values}
    if not all(value is not None for value in rho.values()):
        row["metric_status"] = "unavailable"
        row["geometry_status"] = "unavailable"
        row["reason"] = "; ".join(
            "%s=%s" % (key, row["rho_reasons"].get(key, "nonfinite_rho"))
            for key, value in rho.items() if value is None
        ) or "nonfinite_rho"
        return row
    direct = (rho["rho_A_mat"] + rho["rho_B_pat"]) / 2.0
    cross = (rho["rho_A_pat"] + rho["rho_B_mat"]) / 2.0
    row["direct"] = float(direct)
    row["cross"] = float(cross)
    if abs(direct - cross) <= GEOMETRY_TIE_TOL:
        matched = other = float((direct + cross) / 2.0)
        row["pairing"] = "tie_average"
        row["orientation"] = None
        row["geometry_status"] = "tie_average_within_1e-12"
        row["reason"] = "direct_cross_tie_within_1e-12"
    elif direct > cross:
        matched, other = float(direct), float(cross)
        row["pairing"] = "direct"
        row["orientation"] = "direct"
        row["geometry_status"] = "direct_maximum"
    else:
        matched, other = float(cross), float(direct)
        row["pairing"] = "cross"
        row["orientation"] = "swapped"
        row["geometry_status"] = "cross_maximum"
    row.update({
        "matched": matched,
        "other": other,
        "contrast": float(matched - other),
        "similarity": matched,
        "metric_status": "ok",
    })
    return row


def _evaluate_single(row: dict[str, Any], mask: CommonMask, condition_id: str) -> dict[str, Any]:
    common = mask.common
    candidate = mask.distances[_track_key(condition_id, "copyA")][common]
    ref_mat = mask.distances["reference:mat"][common]
    ref_pat = mask.distances["reference:pat"][common]
    for key, reference_dist in (("rho_A_mat", ref_mat), ("rho_A_pat", ref_pat)):
        rho, reason = _rho_with_reason(candidate, reference_dist)
        row[key] = rho
        if reason is not None:
            row["rho_reasons"][key] = reason
    if row["rho_A_mat"] is None or row["rho_A_pat"] is None:
        row["metric_status"] = "unavailable"
        row["reason"] = "single_track_requires_both_reference_rhos; " + "; ".join(
            "%s=%s" % (key, row["rho_reasons"].get(key, "nonfinite_rho"))
            for key in ("rho_A_mat", "rho_A_pat") if row[key] is None
        )
        return row
    row["similarity"] = float((row["rho_A_mat"] + row["rho_A_pat"]) / 2.0)
    row["pairing"] = "reference_mean"
    row["geometry_status"] = "reference_mean"
    row["metric_status"] = "ok"
    return row


def evaluate_chromosome(
    chromosome: str,
    length_bp: int,
    conditions: Mapping[str, R2Condition | Mapping[str, Any]] | Sequence[R2Condition],
    reference: Mapping[str, Mapping[int, np.ndarray]],
    *,
    chromosome_index: int = 0,
    condition_tracks: Mapping[str, Sequence[str]] | None = None,
    reference_tracks: Sequence[str] | None = None,
) -> dict[str, Any]:
    """在一份共享的染色体 mask 上评估所有提供的 conditions。"""
    normalised = _normalise_conditions(conditions)
    if condition_tracks is None:
        condition_tracks = {
            condition_id: default_track_names(chromosome_index, condition.n_copies)
            for condition_id, condition in normalised.items()
        }
    mask = build_common_mask(
        chromosome, length_bp, normalised, reference,
        chromosome_index=chromosome_index,
        condition_tracks=condition_tracks,
        reference_tracks=reference_tracks,
    )
    rows: dict[str, dict[str, Any]] = {}
    for condition_id, condition in normalised.items():
        tracks = tuple(condition_tracks[condition_id])
        row = _row_base(condition, mask, tracks)
        if condition.n_copies == 2:
            row = _evaluate_two_copy(row, mask, condition_id)
        else:
            row = _evaluate_single(row, mask, condition_id)
        rows[condition_id] = row
    return {
        "chromosome": chromosome,
        "chromosome_index": int(chromosome_index),
        "mask": mask.metadata(),
        "conditions": rows,
    }


def evaluate_genome(
    chromosomes: Sequence[Mapping[str, Any] | Sequence[Any]],
    conditions: Mapping[str, R2Condition | Mapping[str, Any]] | Sequence[R2Condition],
    reference: Mapping[str, Mapping[int, np.ndarray]],
    *,
    condition_tracks_by_chromosome: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> list[dict[str, Any]]:
    """评估所有染色体 rows，不丢弃不可用染色体。"""
    normalised = _normalise_conditions(conditions)
    results = []
    for chromosome_index, item in enumerate(chromosomes):
        if isinstance(item, Mapping):
            name, length_bp = item["name"], item["length_bp"]
        else:
            name, length_bp = item
        tracks = None
        if condition_tracks_by_chromosome is not None:
            tracks = condition_tracks_by_chromosome.get(str(name))
        results.append(evaluate_chromosome(
            str(name), int(length_bp), normalised, reference,
            chromosome_index=chromosome_index,
            condition_tracks=tracks,
        ))
    return results


def _finite_values(values: Sequence[Any]) -> list[float]:
    result = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _chromosome_bootstrap(deltas: Sequence[float], *, seed: int = BOOTSTRAP_SEED,
                          n_boot: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=float)
    if not len(values):
        return {
            "n_chromosomes": 0,
            "mean": None,
            "median": None,
            "ci95": None,
            "seed": int(seed),
            "n_boot": int(n_boot),
        }
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(int(n_boot), len(values)))].mean(axis=1)
    return {
        "n_chromosomes": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [float(item) for item in np.percentile(draws, [2.5, 97.5])],
        "seed": int(seed),
        "n_boot": int(n_boot),
        "unit": "chromosome",
        "interpretation": "technical/structural chromosome bootstrap within one cell; not biological replication or a p-value",
    }


def _row_map(evaluated: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Mapping[str, Any]]]:
    result = {}
    for chromosome in evaluated:
        name = str(chromosome["chromosome"])
        result[name] = chromosome.get("conditions", {})
    return result


def paired_comparison(
    evaluated: Sequence[Mapping[str, Any]],
    left_condition: str,
    right_condition: str,
    *,
    metric: str = "similarity",
    label: str | None = None,
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """在共享 mask 上逐染色体比较两项 conditions。"""
    rows = _row_map(evaluated)
    per_chromosome = []
    finite_deltas = []
    missing = []
    for chromosome, conditions in rows.items():
        left = conditions.get(left_condition, {})
        right = conditions.get(right_condition, {})
        left_value = left.get(metric)
        right_value = right.get(metric)
        if left_value is None or right_value is None:
            delta = None
            missing.append({
                "chromosome": chromosome,
                "left_status": left.get("metric_status"),
                "right_status": right.get("metric_status"),
                "left_value": left_value,
                "right_value": right_value,
            })
        else:
            delta = float(left_value) - float(right_value)
            if not math.isfinite(delta):
                delta = None
                missing.append({
                    "chromosome": chromosome,
                    "left_status": left.get("metric_status"),
                    "right_status": right.get("metric_status"),
                    "left_value": left_value,
                    "right_value": right_value,
                })
            else:
                finite_deltas.append(delta)
        per_chromosome.append({
            "chromosome": chromosome,
            "left_value": left_value,
            "right_value": right_value,
            "delta_left_minus_right": delta,
            "left_status": left.get("metric_status"),
            "right_status": right.get("metric_status"),
        })
    values = np.asarray(finite_deltas, dtype=float)
    ties = int(np.isclose(values, 0.0, rtol=0.0, atol=GEOMETRY_TIE_TOL).sum()) if len(values) else 0
    wins = int((values > GEOMETRY_TIE_TOL).sum()) if len(values) else 0
    losses = int((values < -GEOMETRY_TIE_TOL).sum()) if len(values) else 0
    bootstrap = _chromosome_bootstrap(finite_deltas, seed=seed, n_boot=n_boot)
    return {
        "label": label or "%s_minus_%s" % (left_condition, right_condition),
        "left_condition": left_condition,
        "right_condition": right_condition,
        "metric": metric,
        "direction": "left_minus_right",
        "n_chromosomes_expected": int(len(rows)),
        "n_chromosomes_used": int(len(finite_deltas)),
        "missing_chromosomes": missing,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "per_chromosome": per_chromosome,
        "bootstrap": bootstrap,
        "mean_delta": bootstrap["mean"],
        "median_delta": bootstrap["median"],
        "ci95": bootstrap["ci95"],
        "note": "A paired delta may be negative; it is not clipped. Chromosome bootstrap is technical/structural only.",
    }


def condition_summary(
    evaluated: Sequence[Mapping[str, Any]],
    condition_ids: Sequence[str],
) -> dict[str, Any]:
    """返回仅 R2 的 macro/median 汇总，并显式列出缺失染色体。"""
    expected = [str(item["chromosome"]) for item in evaluated]
    output = {}
    for condition_id in condition_ids:
        rows = [item.get("conditions", {}).get(condition_id, {}) for item in evaluated]
        similarity = _finite_values([row.get("similarity") for row in rows])
        contrast = _finite_values([row.get("contrast") for row in rows])
        similarity_missing = [chromosome for chromosome, row in zip(expected, rows)
                              if row.get("similarity") is None]
        contrast_missing = [chromosome for chromosome, row in zip(expected, rows)
                            if row.get("contrast") is None]
        output[condition_id] = {
            "condition_id": condition_id,
            "n_chromosomes_expected": len(expected),
            "similarity_macro_mean": float(np.mean(similarity)) if similarity else None,
            "similarity_median": float(np.median(similarity)) if similarity else None,
            "similarity_n_finite": len(similarity),
            "similarity_missing_chromosomes": similarity_missing,
            "contrast_macro_mean": float(np.mean(contrast)) if contrast else None,
            "contrast_median": float(np.median(contrast)) if contrast else None,
            "contrast_n_finite": len(contrast),
            "contrast_missing_chromosomes": contrast_missing,
            "contrast_is_applicable": len(contrast) > 0,
        }
    return output


def summarise(
    evaluated: Sequence[Mapping[str, Any]],
    condition_ids: Sequence[str],
    comparisons: Sequence[Mapping[str, Any] | Sequence[str]] = (),
    *,
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """构建仅 R2 的 JSON 汇总和请求的 paired comparisons。"""
    paired = []
    for item in comparisons:
        if isinstance(item, Mapping):
            left = str(item["left"])
            right = str(item["right"])
            metric = str(item.get("metric", "similarity"))
            label = item.get("label")
        else:
            if len(item) not in (2, 3):
                raise ValueError("comparison tuple must be (left, right[, metric])")
            left, right = str(item[0]), str(item[1])
            metric = str(item[2]) if len(item) == 3 else "similarity"
            label = None
        paired.append(paired_comparison(
            evaluated, left, right, metric=metric, label=label,
            seed=seed, n_boot=n_boot,
        ))
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_scope": "R2 only",
        "condition_summary": condition_summary(evaluated, condition_ids),
        "paired_comparisons": paired,
        "bootstrap_policy": {
            "seed": int(seed),
            "n_boot": int(n_boot),
            "unit": "chromosome",
            "interpretation": "technical/structural variation within one P9016 cell; not biological replicate uncertainty and no p-values",
        },
    }


# 为偏好 ``summarize`` 的调用者保留 American-spelling alias。
summarize = summarise


TSV_COLUMNS = (
    "chromosome", "condition_id", "display_name", "role", "endpoint_status", "accepted_as",
    "n_copies", "two_copy", "n_bins", "n_total_non_diagonal_pairs", "n_common_pairs",
    "mask_status", "metric_status", "reason", "track_names", "coverage_json",
    "rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat", "direct", "cross", "matched",
    "other", "contrast", "similarity", "pairing", "orientation", "geometry_status", "rho_reasons_json",
)


def flatten_rows(evaluated: Sequence[Mapping[str, Any]], condition_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for chromosome in evaluated:
        conditions = chromosome.get("conditions", {})
        for condition_id in condition_ids:
            raw = dict(conditions.get(condition_id, {}))
            row = {column: raw.get(column) for column in TSV_COLUMNS}
            row["track_names"] = json.dumps(_jsonable(raw.get("track_names", [])), separators=(",", ":"))
            row["coverage_json"] = json.dumps(_jsonable(raw.get("coverage", {})), sort_keys=True,
                                               separators=(",", ":"))
            row["rho_reasons_json"] = json.dumps(_jsonable(raw.get("rho_reasons", {})), sort_keys=True,
                                                  separators=(",", ":"))
            rows.append(row)
    return rows


def write_table(path: str | Path, rows: Sequence[Mapping[str, Any]], *, delimiter: str = "\t") -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TSV_COLUMNS), delimiter=delimiter,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row.get(column) is None else _jsonable(row.get(column))
                             for column in TSV_COLUMNS})
    return {"path": str(target), "rows": int(len(rows)), "columns": list(TSV_COLUMNS),
            "delimiter": delimiter}


def write_json(path: str | Path, value: Any) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True,
                                 ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {"path": str(target)}


def _stable_jitter(n_chromosomes: int, n_conditions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.035, 0.035, size=(n_chromosomes, n_conditions))


def render_boxplot_figure(
    path: str | Path,
    evaluated: Sequence[Mapping[str, Any]],
    condition_ids: Sequence[str],
    *,
    metric: str,
    title: str,
    y_label: str,
    seed: int = BOOTSTRAP_SEED,
    colors: Sequence[str] = ("#2878b5", "#d77a1f", "#3a9d5d", "#ba4a62"),
) -> dict[str, Any]:
    """渲染请求的紧凑 boxplot 与 paired chromosome points。"""
    if not condition_ids:
        raise ValueError("at least one condition is required for a figure")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows_by_chromosome = _row_map(evaluated)
    chromosome_names = list(rows_by_chromosome)
    data = []
    for condition_id in condition_ids:
        data.append(np.asarray(_finite_values([
            rows_by_chromosome[chromosome].get(condition_id, {}).get(metric)
            for chromosome in chromosome_names
        ]), dtype=float))
    all_values = np.concatenate([value for value in data if len(value)], dtype=float) \
        if any(len(value) for value in data) else np.empty(0, dtype=float)
    if len(all_values):
        span = max(float(np.ptp(all_values)), 0.05)
        lower = min(0.0, float(all_values.min()) - 0.08 * span)
        upper = max(0.05, float(all_values.max()) + 0.12 * span)
        if upper <= lower:
            upper = lower + 0.1
    else:
        lower, upper = 0.0, 1.0

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.0))
    positions = np.arange(1, len(condition_ids) + 1, dtype=float)
    present = [(position, value) for position, value in zip(positions, data) if len(value)]
    if present:
        box = ax.boxplot([value for _position, value in present],
                         positions=[position for position, _value in present],
                         widths=0.48, patch_artist=True, showfliers=False,
                         medianprops={"color": "black", "linewidth": 0.9},
                         whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7},
                         boxprops={"linewidth": 0.7})
        present_index = 0
        for index, value in enumerate(data):
            if not len(value):
                continue
            color = colors[index % len(colors)]
            box["boxes"][present_index].set_facecolor(color)
            box["boxes"][present_index].set_alpha(0.55)
            present_index += 1

    jitter = _stable_jitter(len(chromosome_names), len(condition_ids), seed)
    for chrom_index, chromosome in enumerate(chromosome_names):
        x_values, y_values = [], []
        for condition_index, condition_id in enumerate(condition_ids):
            value = rows_by_chromosome[chromosome].get(condition_id, {}).get(metric)
            if value is None or not math.isfinite(float(value)):
                continue
            x_values.append(float(positions[condition_index] + jitter[chrom_index, condition_index]))
            y_values.append(float(value))
        if len(x_values) >= 2:
            ax.plot(x_values, y_values, color="#9a9a9a", linewidth=0.55, alpha=0.62, zorder=1)
        for condition_index, condition_id in enumerate(condition_ids):
            value = rows_by_chromosome[chromosome].get(condition_id, {}).get(metric)
            if value is None or not math.isfinite(float(value)):
                continue
            ax.scatter(
                [positions[condition_index] + jitter[chrom_index, condition_index]], [float(value)],
                s=11, color=colors[condition_index % len(colors)], edgecolors="white",
                linewidths=0.25, zorder=3,
            )
    labels = [str(rows_by_chromosome[chromosome_names[0]].get(condition_id, {}).get("display_name", condition_id))
              if chromosome_names else condition_id for condition_id in condition_ids]
    ax.set_xticks(positions, labels)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(lower, upper)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y", color="#dddddd", linewidth=0.4, alpha=0.6)
    handles = [Patch(facecolor=colors[index % len(colors)], edgecolor="none", alpha=0.55,
                     label=label.replace("\n", " ")) for index, label in enumerate(labels)]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7,
              handlelength=1.0, handletextpad=0.35, borderpad=0.15, labelspacing=0.2)
    fig.suptitle("P9016 | 1 Mb bins, OFF=3 Mb | 20 chromosomes", fontsize=7, y=0.995)
    fig.text(0.5, 0.012, "Dots = chromosome; gray lines pair the same chromosome; one cell",
             ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.19, top=0.87)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return {
        "figure": str(target),
        "metric": metric,
        "condition_ids": list(condition_ids),
        "chromosomes": chromosome_names,
        "seed_jitter": int(seed),
        "dpi": 300,
        "figsize_inches": [6.0, 3.0],
        "axis_limits": [float(lower), float(upper)],
    }


def render_figure_pair(
    output_stem: str | Path,
    evaluated: Sequence[Mapping[str, Any]],
    condition_ids: Sequence[str],
    *,
    title: str,
    similarity_title: str = "Similarity to reference",
    contrast_title: str = "Copy-specific R2 contrast",
) -> dict[str, Any]:
    """写出请求的两个 R2 panel 的 PNG 和 PDF 版本。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    rows_by_chromosome = _row_map(evaluated)
    chromosome_names = list(rows_by_chromosome)
    colors = ("#2878b5", "#d77a1f", "#3a9d5d", "#ba4a62")
    jitter = _stable_jitter(len(chromosome_names), len(condition_ids), BOOTSTRAP_SEED)
    all_metric_values: dict[str, list[np.ndarray]] = {"similarity": [], "contrast": []}
    for metric in all_metric_values:
        for condition_id in condition_ids:
            all_metric_values[metric].append(np.asarray(_finite_values([
                rows_by_chromosome[chromosome].get(condition_id, {}).get(metric)
                for chromosome in chromosome_names
            ]), dtype=float))

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), squeeze=False)
    axes = axes[0]
    for ax, metric, panel_title, y_label in (
        (axes[0], "similarity", similarity_title, "Mean matched Spearman rho"),
        (axes[1], "contrast", contrast_title, "Matched minus other Spearman rho"),
    ):
        data = all_metric_values[metric]
        positions = np.arange(1, len(condition_ids) + 1, dtype=float)
        present = [(position, value) for position, value in zip(positions, data) if len(value)]
        if present:
            box = ax.boxplot([value for _position, value in present],
                             positions=[position for position, _value in present], widths=0.48,
                             patch_artist=True, showfliers=False,
                             medianprops={"color": "black", "linewidth": 0.9},
                             whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7},
                             boxprops={"linewidth": 0.7})
            present_index = 0
            for index, value in enumerate(data):
                if not len(value):
                    continue
                box["boxes"][present_index].set_facecolor(colors[index % len(colors)])
                box["boxes"][present_index].set_alpha(0.55)
                present_index += 1
        for chromosome_index, chromosome in enumerate(chromosome_names):
            x_values, y_values = [], []
            for condition_index, condition_id in enumerate(condition_ids):
                value = rows_by_chromosome[chromosome].get(condition_id, {}).get(metric)
                if value is None or not math.isfinite(float(value)):
                    continue
                x_values.append(float(positions[condition_index] + jitter[chromosome_index, condition_index]))
                y_values.append(float(value))
            if len(x_values) >= 2:
                ax.plot(x_values, y_values, color="#9a9a9a", linewidth=0.55, alpha=0.62, zorder=1)
            for condition_index, condition_id in enumerate(condition_ids):
                value = rows_by_chromosome[chromosome].get(condition_id, {}).get(metric)
                if value is None or not math.isfinite(float(value)):
                    continue
                ax.scatter([positions[condition_index] + jitter[chromosome_index, condition_index]], [float(value)],
                           s=11, color=colors[condition_index % len(colors)], edgecolors="white",
                           linewidths=0.25, zorder=3)
        all_values = np.concatenate([value for value in data if len(value)], dtype=float) \
            if any(len(value) for value in data) else np.empty(0, dtype=float)
        if len(all_values):
            span = max(float(np.ptp(all_values)), 0.05)
            lower = min(0.0, float(all_values.min()) - 0.08 * span)
            upper = max(0.05, float(all_values.max()) + 0.12 * span)
        else:
            lower, upper = 0.0, 1.0
        if upper <= lower:
            upper = lower + 0.1
        ax.set_ylim(lower, upper)
        ax.set_xticks(positions, [
            str(rows_by_chromosome[chromosome_names[0]].get(condition_id, {}).get("display_name", condition_id))
            if chromosome_names else condition_id for condition_id in condition_ids
        ])
        ax.set_ylabel(y_label)
        ax.set_title(panel_title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(axis="y", color="#dddddd", linewidth=0.4, alpha=0.6)
    handles = [Patch(facecolor=colors[index % len(colors)], edgecolor="none", alpha=0.55,
                     label=str(rows_by_chromosome[chromosome_names[0]].get(condition_id, {}).get("display_name", condition_id))
                     .replace("\n", " "))
               for index, condition_id in enumerate(condition_ids)] if chromosome_names else []
    if handles:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=min(4, len(handles)),
                   frameon=False, fontsize=7, handlelength=1.0, handletextpad=0.35,
                   columnspacing=0.7, borderpad=0.1)
    fig.suptitle(title, fontsize=7, y=0.995)
    fig.text(0.5, 0.012, "Dots = chromosome; gray lines pair the same chromosome; one cell",
             ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.76, wspace=0.30)
    paths = {}
    for extension in ("png", "pdf"):
        target = output_stem.with_suffix("." + extension)
        fig.savefig(target, dpi=300)
        paths[extension] = str(target)
    plt.close(fig)
    return {
        "png": paths["png"],
        "pdf": paths["pdf"],
        "condition_ids": list(condition_ids),
        "metric_panels": ["similarity", "contrast"],
        "figure_size_inches": [6.0, 3.0],
        "dpi": 300,
        "jitter_seed": BOOTSTRAP_SEED,
        "chromosomes": chromosome_names,
    }


def write_outputs(
    output_dir: str | Path,
    evaluated: Sequence[Mapping[str, Any]],
    condition_ids: Sequence[str],
    *,
    main_condition_ids: Sequence[str],
    control_condition_ids: Sequence[str] = (),
    supplement_condition_ids: Sequence[str] = (),
    comparisons: Sequence[Mapping[str, Any] | Sequence[str]] = (),
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = BOOTSTRAP_DRAWS,
    title: str = "P9016 R2 comparison",
) -> dict[str, Any]:
    """写出完整 R2 表、JSON 汇总以及主/控制图。"""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = flatten_rows(evaluated, condition_ids)
    summary = summarise(evaluated, condition_ids, comparisons, seed=seed, n_boot=n_boot)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": {
            "metric": "Spearman of Euclidean distances on unordered non-diagonal genomic bin pairs",
            "grid": "complete 1 Mb starts in range(OFF=3 Mb, chromosome_length)",
            "mask": "one finite pair mask shared by all condition tracks and both reference tracks per chromosome",
            "rho_min_pairs": MIN_RHO_PAIRS,
            "gauge": "direct/cross means; geometry chooses the larger, ties within 1e-12 average and contrast=0",
            "scale": "raw coordinate distance ranks only; no physical scale or Procrustes calibration",
        },
        "chromosomes": _jsonable(evaluated),
        "summary": summary,
        "main_condition_ids": list(main_condition_ids),
        "control_condition_ids": list(control_condition_ids),
        "supplement_condition_ids": list(supplement_condition_ids),
    }
    paths = {
        "r2_json": write_json(root / "r2_results.json", payload),
        "r2_summary_json": write_json(root / "r2_summary.json", summary),
        "r2_tsv": write_table(root / "r2_per_chromosome.tsv", rows, delimiter="\t"),
        "r2_csv": write_table(root / "r2_per_chromosome.csv", rows, delimiter=",") ,
    }
    if main_condition_ids:
        paths["main_figure"] = render_figure_pair(root / "plots" / "r2_main", evaluated,
                                                    main_condition_ids, title=title)
    if control_condition_ids:
        paths["control_figure"] = render_figure_pair(root / "plots" / "r2_controls", evaluated,
                                                       control_condition_ids,
                                                       title=title + " | controls")
    if supplement_condition_ids:
        paths["supplement_figure"] = render_figure_pair(root / "plots" / "r2_supplement", evaluated,
                                                          supplement_condition_ids,
                                                          title=title + " | accepted trial")
    return {"paths": paths, "summary": summary, "n_rows": len(rows)}


def _synthetic_track(points: np.ndarray) -> dict[int, np.ndarray]:
    return {GRID_OFFSET_BP + index * BIN_SIZE_BP: np.asarray(point, dtype=float)
            for index, point in enumerate(points)}


def synthetic_check() -> dict[str, Any]:
    """不读取真实输入，运行小型冻结 invariant/availability fixture。"""
    n_bins = 8
    t = np.arange(n_bins, dtype=float)
    mat = np.column_stack((t, np.sin(t * 0.31), np.cos(t * 0.17) * 2.0))
    pat = np.column_stack((np.sin(t * 0.37) * 3.0, 0.71 * t + np.cos(t * 0.11),
                           np.cos(t * 0.29) * 2.5))
    reference = {"chr1(mat)": _synthetic_track(mat), "chr1(pat)": _synthetic_track(pat)}

    good = R2Condition("good", "Good", {"c01a": _synthetic_track(mat), "c01b": _synthetic_track(pat)})
    swapped = R2Condition("swapped", "Swapped", {"c01a": _synthetic_track(pat), "c01b": _synthetic_track(mat)})
    identical = R2Condition("identical", "Identical", {
        "c01a": _synthetic_track(mat), "c01b": _synthetic_track(mat),
    })
    single = R2Condition("consensus", "Consensus", {"c01a": _synthetic_track(mat)}, n_copies=1,
                         role="control")
    missing_points = dict(_synthetic_track(pat))
    missing_points.pop(GRID_OFFSET_BP + 3 * BIN_SIZE_BP)
    missing = R2Condition("missing", "Missing", {
        "c01a": _synthetic_track(mat), "c01b": missing_points,
    })
    constant = R2Condition("constant", "Constant", {
        "c01a": _synthetic_track(np.zeros_like(mat)), "c01b": _synthetic_track(pat),
    })
    conditions = {item.condition_id: item for item in (good, swapped, identical, single, missing, constant)}
    tracks = {condition_id: default_track_names(0, condition.n_copies)
              for condition_id, condition in conditions.items()}
    evaluated = evaluate_chromosome("chr1", GRID_OFFSET_BP + n_bins * BIN_SIZE_BP,
                                    conditions, reference, chromosome_index=0,
                                    condition_tracks=tracks)
    rows = evaluated["conditions"]
    single_mean_row = rows["consensus"]
    constant_reference = {
        "chr1(mat)": _synthetic_track(np.zeros_like(mat)),
        "chr1(pat)": _synthetic_track(pat),
    }
    constant_reference_row = evaluate_chromosome(
        "chr1", GRID_OFFSET_BP + n_bins * BIN_SIZE_BP, {"consensus": single},
        constant_reference, chromosome_index=0, condition_tracks={"consensus": ("c01a",)},
    )["conditions"]["consensus"]
    missing_reference = {"chr1(mat)": {}, "chr1(pat)": _synthetic_track(pat)}
    missing_reference_row = evaluate_chromosome(
        "chr1", GRID_OFFSET_BP + n_bins * BIN_SIZE_BP, {"consensus": single},
        missing_reference, chromosome_index=0, condition_tracks={"consensus": ("c01a",)},
    )["conditions"]["consensus"]
    checks = {
        "common_mask_shared": len({row["n_common_pairs"] for row in rows.values()}) == 1,
        "common_mask_expected": rows["good"]["n_common_pairs"] == n_bins * (n_bins - 1) // 2 - (n_bins - 1),
        "gauge_exchange_invariant_similarity": np.isclose(rows["good"]["similarity"], rows["swapped"]["similarity"], atol=1e-12),
        "gauge_exchange_invariant_contrast": np.isclose(rows["good"]["contrast"], rows["swapped"]["contrast"], atol=1e-12),
        "identical_copies_contrast_zero": np.isclose(rows["identical"]["contrast"], 0.0, atol=1e-12),
        "zero_contrast_still_applicable": condition_summary([evaluated], ["identical"])["identical"]["contrast_is_applicable"],
        "single_similarity_reference_mean": np.isclose(
            single_mean_row["similarity"],
            (single_mean_row["rho_A_mat"] + single_mean_row["rho_A_pat"]) / 2.0,
            atol=1e-12,
        ),
        "single_reference_mean_fields": (
            single_mean_row["pairing"] == "reference_mean"
            and single_mean_row["geometry_status"] == "reference_mean"
        ),
        "single_reference_constant_is_na": (
            constant_reference_row["similarity"] is None
            and constant_reference_row["metric_status"] == "unavailable"
            and "constant_reference_distance" in constant_reference_row["reason"]
        ),
        "single_reference_missing_is_na": (
            missing_reference_row["similarity"] is None
            and missing_reference_row["metric_status"] == "unavailable"
            and "insufficient_common_pairs" in missing_reference_row["reason"]
        ),
        "single_consensus_contrast_na": rows["consensus"]["contrast"] is None,
        "missing_coverage_recorded": rows["missing"]["coverage"]["condition:missing:copyB"]["n_missing_bins"] == 1,
        "constant_rho_na_with_reason": rows["constant"]["rho_A_mat"] is None and "constant_candidate_distance" in rows["constant"]["reason"],
    }
    fake = []
    for chromosome, left, right in (("chr1", 0.2, 0.4), ("chr2", 0.7, 0.5), ("chr3", 0.4, 0.4)):
        fake.append({"chromosome": chromosome, "conditions": {
            "left": {"similarity": left, "contrast": 0.2, "metric_status": "ok"},
            "right": {"similarity": right, "contrast": 0.1, "metric_status": "ok"},
        }})
    negative = paired_comparison(fake, "left", "right", metric="similarity")
    checks["negative_paired_delta_preserved"] = (
        negative["per_chromosome"][0]["delta_left_minus_right"] < 0
        and negative["mean_delta"] < 0
        and negative["wins"] == 1
        and negative["losses"] == 1
    )
    failed = [name for name, value in checks.items() if not value]
    result = {"schema_version": SCHEMA_VERSION, "checks": checks, "passed": not failed,
              "failed": failed, "synthetic_common_pairs": rows["good"]["n_common_pairs"]}
    if failed:
        raise AssertionError("synthetic R2 checks failed: %s" % ", ".join(failed))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dedicated P9016 R2 comparison helpers")
    parser.add_argument("--synthetic-check", action="store_true", help="run the lightweight pure synthetic checks")
    args = parser.parse_args(argv)
    if not args.synthetic_check:
        parser.error("only --synthetic-check is available in the code-preparation phase")
    print(json.dumps(_jsonable(synthetic_check()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
