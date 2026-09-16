#!/usr/bin/env python
"""036 C0 random 1 Mb checkpoint 的 MAP contact/Rg 无重拟合后处理。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pr import contact_model

OUT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = ROOT / "test_res/036-20260914T064651Z-gpu-multires/checkpoints/C0/random_joint"
PUBLISHED_COORDS = ROOT / "test_res/036-20260914T064651Z-gpu-multires/coords/C0/random_joint/final-1m.3dg"
SNPFREE = ROOT / "inputs/P9016.snpfree.pairs.gz"
LEGACY_042 = ROOT / "test_res/042-20260915T050722Z-1mb-contact-rg-diagnostic/metrics.tsv"
BIN_SIZE = 1_000_000
EXPECTED_ITERS = list(range(10, 241, 10))
EXPECTED_SHAPE = (2, 2645, 3)
EXPECTED_THETA = 6 * 2645 + 1
EPSILON = 1e-6
BLOCK_SIZE = 65_536
SERIALIZATION_TOL = 5e-15
INTER_MIN_DISTANCE_TOL = 1e-12
CHOICE_LABELS = ("AA", "AB", "BA", "BB")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint_iteration(path: Path) -> int:
    stem = path.stem
    return int(stem.rsplit("-", 1)[1])


def load_legacy_metrics(path: Path) -> dict[int, dict[str, str]]:
    """读取 042 已完成表中的 scalar，不重新评估训练目标。"""
    with path.open("rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != len(EXPECTED_ITERS):
        raise ValueError(f"042 metrics has {len(rows)} rows, expected {len(EXPECTED_ITERS)}")
    required = {"iteration", "p", "count_nll_normalized", "count_nll_raw", "totalJ", "Rg"}
    if not required.issubset(rows[0]):
        raise ValueError(f"042 metrics missing scalar fields: {sorted(required.difference(rows[0]))}")
    indexed: dict[int, dict[str, str]] = {}
    for row in rows:
        iteration = int(row["iteration"])
        if iteration in indexed:
            raise ValueError(f"duplicate iteration {iteration} in 042 metrics")
        indexed[iteration] = row
        for field in ("p", "count_nll_normalized", "count_nll_raw", "totalJ", "Rg"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"non-finite 042 scalar {field} at iteration {iteration}")
    if sorted(indexed) != EXPECTED_ITERS:
        raise ValueError("042 metrics iterations are not exactly accepted iterations 10..240")
    return indexed


def load_checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], int]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"coordinates", "theta", "y", "positions", "chromosome_index", "fullhistory_json"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{path} missing checkpoint fields: {sorted(missing)}")
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        theta = np.asarray(payload["theta"], dtype=np.float64).copy()
        raw_y = np.asarray(payload["y"], dtype=np.float64).copy()
        positions = np.asarray(payload["positions"], dtype=np.int64).copy()
        chromosome_index = np.asarray(payload["chromosome_index"], dtype=np.int64).copy()
        history = json.loads(str(payload["fullhistory_json"].item()))
    if coordinates.shape != EXPECTED_SHAPE or raw_y.shape != EXPECTED_SHAPE:
        raise ValueError(f"{path} coordinate/y shape mismatch: {coordinates.shape} / {raw_y.shape}")
    if theta.shape != (EXPECTED_THETA,):
        raise ValueError(f"{path} theta shape mismatch: {theta.shape}")
    if positions.shape != (2645,) or chromosome_index.shape != (2645,):
        raise ValueError(f"{path} grid metadata shape mismatch")
    if not all(np.all(np.isfinite(a)) for a in (coordinates, theta, raw_y)):
        raise ValueError(f"{path} contains non-finite coordinate state")
    iteration = checkpoint_iteration(path)
    entries = [entry for entry in history if int(entry.get("iteration", -1)) == iteration]
    if len(entries) != 1:
        raise ValueError(f"{path} history has {len(entries)} entries for iteration {iteration}")
    entry = entries[0]
    components = entry.get("components")
    if not isinstance(components, dict):
        raise ValueError(f"{path} history entry lacks components")
    return coordinates, theta, raw_y, positions, chromosome_index, entry, components, len(history)


def parse_published_coordinates(data: contact_model.AggregatedContacts) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.empty((2, data.n_loci, 3), dtype=np.float64)
    result.fill(np.nan)
    expected_rows = 0
    seen: set[tuple[str, int]] = set()
    track_to_copy_locus: dict[str, tuple[int, slice]] = {}
    for spec in data.track_specs:
        track_to_copy_locus[spec.name] = (spec.copy_index, data.chromosome_slice(spec.chromosome_index))
    with PUBLISHED_COORDS.open("rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"published coordinate row {line_no} has {len(fields)} fields")
            track, position_text = fields[0], fields[1]
            if track not in track_to_copy_locus:
                raise ValueError(f"published coordinate row {line_no} has unknown track {track}")
            try:
                position = int(position_text)
                xyz = np.asarray([float(v) for v in fields[2:]], dtype=np.float64)
            except ValueError as exc:
                raise ValueError(f"published coordinate row {line_no} is not numeric") from exc
            copy_index, slc = track_to_copy_locus[track]
            local_bin = position // data.bin_size
            if position != local_bin * data.bin_size or local_bin < 0 or local_bin >= slc.stop - slc.start:
                raise ValueError(f"published coordinate row {line_no} has invalid position {position}")
            key = (track, position)
            if key in seen:
                raise ValueError(f"duplicate published coordinate row {line_no}: {key}")
            seen.add(key)
            result[copy_index, slc.start + local_bin] = xyz
            expected_rows += 1
    if expected_rows != 5290 or len(seen) != 5290 or not np.all(np.isfinite(result)):
        raise ValueError("published coordinate file is not a complete finite 40-track grid")
    return result, {"rows": expected_rows, "tracks": len(track_to_copy_locus), "unique_rows": len(seen)}


def map_distances_for_block(x: np.ndarray, p: float, i: np.ndarray, j: np.ndarray,
                            cis: np.ndarray, r0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回四个距离、四个 score、固定顺序的 MAP choice 和 exact tie mask。"""
    deltas = np.stack((
        x[0, i] - x[0, j],
        x[0, i] - x[1, j],
        x[1, i] - x[0, j],
        x[1, i] - x[1, j],
    ), axis=0)
    squared = np.sum(deltas * deltas, axis=2)
    distances = np.sqrt(squared)
    scale = r0 * r0
    kernels = EPSILON + (1.0 - EPSILON) * (1.0 + squared / scale) ** -2
    weights = np.stack((
        np.where(cis, 0.5 * p, 0.25),
        np.where(cis, 0.5 * (1.0 - p), 0.25),
        np.where(cis, 0.5 * (1.0 - p), 0.25),
        np.where(cis, 0.5 * p, 0.25),
    ), axis=0)
    scores = weights * kernels
    if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(scores)):
        raise FloatingPointError("non-finite MAP distance or score")
    # np.argmax 的第一项规则正好实现 AA, AB, BA, BB 的 exact-tie policy。
    choice = np.argmax(scores, axis=0).astype(np.uint8, copy=False)
    column = np.arange(scores.shape[1])
    selected_scores = scores[choice, column]
    if not np.all(selected_scores >= scores):
        raise AssertionError("MAP choice is not a score argmax")
    tie = np.sum(scores == selected_scores[None, :], axis=0) > 1
    return (distances[0], distances[1], distances[2], distances[3],
            scores, choice, tie)


def contact_distances(data: contact_model.AggregatedContacts, x: np.ndarray, p: float,
                      fixed_choices: np.ndarray | None) -> tuple[dict[str, float], np.ndarray, dict[str, int]]:
    """按 bin-pair MAP 距离做 counts 加权，并返回本步 choices 与 tie audit。"""
    n_intra = 0
    n_inter = 0
    sums: dict[str, float] = {}
    for metric in ("map", "fixed_map"):
        sums[f"intra_{metric}"] = 0.0
        sums[f"inter_{metric}"] = 0.0
    choices = np.empty(data.n_pairs, dtype=np.uint8)
    tie_stats = {
        "map_tie_pair_rows": 0,
        "map_tie_pair_rows_intra": 0,
        "map_tie_pair_rows_inter": 0,
        "map_tie_records": 0,
        "map_tie_records_intra": 0,
        "map_tie_records_inter": 0,
        "argmax_verified_pair_rows": 0,
        "inter_min_distance_max_abs_error": 0.0,
        "intra_map_nonmin_pair_rows": 0,
    }
    for start in range(0, data.n_pairs, BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, data.n_pairs)
        i = data.pair_i[start:stop]
        j = data.pair_j[start:stop]
        cis = data.cis_pair[start:stop]
        count = data.counts[start:stop].astype(np.float64, copy=False)
        daa, dab, dba, dbb, scores, choice, tie = map_distances_for_block(
            x, p, i, j, cis, data.r0)
        distances = np.stack((daa, dab, dba, dbb), axis=0)
        column = np.arange(stop - start)
        map_distance = distances[choice, column]
        if fixed_choices is None:
            fixed_distance = map_distance
        else:
            fixed_distance = distances[fixed_choices[start:stop], column]
        choices[start:stop] = choice
        if not all(np.all(np.isfinite(v)) for v in (distances, scores, map_distance, fixed_distance)):
            raise FloatingPointError("non-finite contact distance")
        cis_count = count[cis]
        inter_count = count[~cis]
        n_intra += int(np.sum(cis_count, dtype=np.int64))
        n_inter += int(np.sum(inter_count, dtype=np.int64))
        for metric, values in (("map", map_distance), ("fixed_map", fixed_distance)):
            sums[f"intra_{metric}"] += float(np.dot(cis_count, values[cis]))
            sums[f"inter_{metric}"] += float(np.dot(inter_count, values[~cis]))
        tie_intra = tie & cis
        tie_inter = tie & ~cis
        tie_stats["map_tie_pair_rows"] += int(np.count_nonzero(tie))
        tie_stats["map_tie_pair_rows_intra"] += int(np.count_nonzero(tie_intra))
        tie_stats["map_tie_pair_rows_inter"] += int(np.count_nonzero(tie_inter))
        tie_stats["map_tie_records"] += int(np.sum(data.counts[start:stop][tie], dtype=np.int64))
        tie_stats["map_tie_records_intra"] += int(np.sum(data.counts[start:stop][tie_intra], dtype=np.int64))
        tie_stats["map_tie_records_inter"] += int(np.sum(data.counts[start:stop][tie_inter], dtype=np.int64))
        tie_stats["argmax_verified_pair_rows"] += stop - start
        if np.any(~cis):
            min_inter = np.min(distances[:, ~cis], axis=0)
            inter_error = float(np.max(np.abs(map_distance[~cis] - min_inter)))
            tie_stats["inter_min_distance_max_abs_error"] = max(
                tie_stats["inter_min_distance_max_abs_error"], inter_error)
        if np.any(cis):
            min_intra = np.min(distances[:, cis], axis=0)
            tie_stats["intra_map_nonmin_pair_rows"] += int(
                np.count_nonzero(map_distance[cis] > min_intra + INTER_MIN_DISTANCE_TOL))
    if n_intra <= 0 or n_inter <= 0:
        raise ValueError("contact denominators are empty")
    result: dict[str, float] = {}
    for metric in ("map", "fixed_map"):
        result[f"distance_intra_{metric}"] = sums[f"intra_{metric}"] / n_intra
        result[f"distance_inter_{metric}"] = sums[f"inter_{metric}"] / n_inter
        result[f"distance_all_{metric}"] = (
            sums[f"intra_{metric}"] + sums[f"inter_{metric}"]) / (n_intra + n_inter)
    result["count_intra"] = int(n_intra)
    result["count_inter"] = int(n_inter)
    return result, choices, tie_stats


def rg_and_center(x: np.ndarray) -> tuple[float, np.ndarray, float]:
    beads = x.reshape(-1, 3)
    center = beads.mean(axis=0)
    radius = np.sqrt(np.sum((beads - center) ** 2, axis=1))
    rg = float(np.sqrt(np.mean(radius ** 2)))
    return rg, center, float(np.max(radius))


def relative_percent(start: float, end: float) -> float:
    return float(100.0 * (end - start) / start) if start != 0.0 else float("nan")


def monotonic_flags(values: list[float], tol: float = 1e-12) -> dict[str, bool]:
    diffs = np.diff(np.asarray(values, dtype=np.float64))
    return {
        "nonincreasing": bool(np.all(diffs <= tol)),
        "nondecreasing": bool(np.all(diffs >= -tol)),
        "strictly_nonincreasing": bool(np.all(diffs < -tol)),
        "strictly_nondecreasing": bool(np.all(diffs > tol)),
    }


def make_plot(rows: list[dict[str, Any]]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "lines.linewidth": 1.0})
    iterations = [row["iteration"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=300)
    ax = axes[0]
    colors = {"intra": "#1f77b4", "inter": "#d62728"}
    for group in ("intra", "inter"):
        ax.plot(iterations, [row[f"distance_{group}_map"] for row in rows],
                color=colors[group], linestyle="-", label=f"{group} MAP")
        ax.plot(iterations, [row[f"distance_{group}_fixed_map"] for row in rows],
                color=colors[group], linestyle="--", label=f"{group} fixed iter10 MAP")
    ax.set_xlabel("Accepted iteration")
    ax.set_ylabel("Contact distance (model units)")
    ax.set_title("MAP contact distance")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, handlelength=2.2, loc="best")
    ax = axes[1]
    ax.plot(iterations, [row["Rg"] for row in rows], color="#2ca02c", linestyle="-", label="whole-cell Rg")
    ax.set_xlabel("Accepted iteration")
    ax.set_ylabel("Rg (model units)")
    ax.set_title("Whole-cell Rg")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, handlelength=2.2, loc="best")
    fig.tight_layout(pad=0.7, w_pad=1.2)
    path = OUT / "plots/map_contact_distance_rg.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> int:
    OUT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("plots").mkdir(parents=True, exist_ok=True)
    log_lines = []
    def log(message: str) -> None:
        log_lines.append(message)
        print(message, flush=True)

    checkpoint_paths = sorted(CHECKPOINT_ROOT.glob("1m-accepted-*.npz"), key=checkpoint_iteration)
    if [checkpoint_iteration(path) for path in checkpoint_paths] != EXPECTED_ITERS:
        raise ValueError("checkpoint set is not exactly accepted iterations 10..240")
    legacy_metrics = load_legacy_metrics(LEGACY_042)
    log(f"loaded 042 scalar metrics read-only: {LEGACY_042}")
    log(f"loading frozen SNP-free aggregation at {BIN_SIZE} bp")
    data = contact_model.load_frozen_p9016_aggregate(BIN_SIZE)
    audit = data.budget()
    expected_audit = {
        "raw_records": 1_703_888,
        "raw_same_bin": 438_774,
        "raw_cis_offdiag": 696_680,
        "raw_inter": 568_434,
        "aggregate_same_bin": 438_774,
        "aggregate_cis_offdiag": 696_680,
        "aggregate_inter": 568_434,
        "endpoint_total": 3_407_776,
        "n_loci": 2645,
        "n_chromosomes": 20,
    }
    for key, expected in expected_audit.items():
        if int(audit[key]) != expected:
            raise ValueError(f"budget mismatch for {key}: {audit[key]} != {expected}")
    log(f"raw budget verified: same-bin={audit['raw_same_bin']} intra-offdiag={audit['raw_cis_offdiag']} inter={audit['raw_inter']}")

    published, published_meta = parse_published_coordinates(data)
    rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    fixed_choices: np.ndarray | None = None
    fixed_choice_consistency_error = 0
    legacy_scalar_max_abs_error = 0.0
    p_theta_max_abs_error = 0.0
    rg_042_max_abs_error = 0.0
    expected_positions = data.locus_bin.astype(np.int64) * BIN_SIZE
    for index, checkpoint_path in enumerate(checkpoint_paths):
        iteration = checkpoint_iteration(checkpoint_path)
        x, theta, raw_y, positions, chromosome_index, history_entry, components, history_length = load_checkpoint(checkpoint_path)
        if not np.array_equal(positions, expected_positions):
            raise ValueError(f"{checkpoint_path} positions are not the origin-0 1Mb grid")
        if not np.array_equal(chromosome_index, data.locus_chromosome.astype(np.int64)):
            raise ValueError(f"{checkpoint_path} chromosome_index does not match the full grid")
        transformed = contact_model.sphere_forward(raw_y)
        transform_error = float(np.max(np.abs(transformed - x)))
        if transform_error > SERIALIZATION_TOL:
            raise ValueError(f"{checkpoint_path} coordinate transform error {transform_error}")
        radius = np.linalg.norm(x, axis=-1)
        if not np.all(radius < 1.0) or not np.all(np.isfinite(radius)):
            raise ValueError(f"{checkpoint_path} physical coordinates leave strict unit ball")
        p_theta, _ = contact_model.p_from_q(float(theta[-1]))
        legacy = legacy_metrics[iteration]
        p = float(legacy["p"])
        p_error = abs(p - p_theta)
        p_theta_max_abs_error = max(p_theta_max_abs_error, p_error)
        if p_error > SERIALIZATION_TOL:
            raise ValueError(f"{checkpoint_path} p differs from 042 table/theta by {p_error}")
        for field, component_field in (
                ("count_nll_normalized", "count_nll_normalized"),
                ("count_nll_raw", "count_nll_raw"),
                ("totalJ", "total")):
            scalar_error = abs(float(legacy[field]) - float(components[component_field]))
            legacy_scalar_max_abs_error = max(legacy_scalar_max_abs_error, scalar_error)
            if scalar_error > SERIALIZATION_TOL:
                raise ValueError(
                    f"{checkpoint_path} {field} differs from 042/checkpoint history by {scalar_error}")
        distances, current_choices, tie_stats = contact_distances(data, x, p, fixed_choices)
        if fixed_choices is None:
            fixed_choices = current_choices.copy()
        if fixed_choices.shape != current_choices.shape or not np.all((fixed_choices >= 0) & (fixed_choices < 4)):
            raise ValueError("fixed iter10 MAP choices are malformed")
        choice_changed = current_choices != fixed_choices
        changed_pair_rows = int(np.count_nonzero(choice_changed))
        changed_records = int(np.sum(data.counts[choice_changed], dtype=np.int64))
        if index == 0:
            fixed_choice_consistency_error = changed_pair_rows
        rg, center, max_bead_radius = rg_and_center(x)
        rg_042_error = abs(rg - float(legacy["Rg"]))
        rg_042_max_abs_error = max(rg_042_max_abs_error, rg_042_error)
        if rg_042_error > SERIALIZATION_TOL:
            raise ValueError(f"{checkpoint_path} Rg differs from 042 by {rg_042_error}")
        row: dict[str, Any] = {
            "iteration": iteration,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "p": float(p),
            "count_nll_normalized": float(legacy["count_nll_normalized"]),
            "count_nll_raw": float(legacy["count_nll_raw"]),
            "totalJ": float(legacy["totalJ"]),
            "Rg": rg,
            "Rg_vs_042_abs_error": rg_042_error,
            "center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2]),
            "max_bead_radius": max_bead_radius,
            "fixed_choice_changed_pair_rows": changed_pair_rows,
            "fixed_choice_changed_records": changed_records,
            **distances,
            **tie_stats,
        }
        for metric in ("map", "fixed_map"):
            for group in ("intra", "inter", "all"):
                row[f"distance_{group}_{metric}_over_Rg"] = row[f"distance_{group}_{metric}"] / rg
        if not all(math.isfinite(float(value)) for key, value in row.items() if key not in {"checkpoint_path", "checkpoint_sha256"}):
            raise ValueError(f"non-finite metric row at iteration {iteration}")
        rows.append(row)
        validation_rows.append({
            "iteration": iteration,
            "transform_max_abs": transform_error,
            "physical_max_radius": float(np.max(radius)),
            "theta_finite": bool(np.all(np.isfinite(theta))),
            "raw_y_finite": bool(np.all(np.isfinite(raw_y))),
            "coordinates_finite": bool(np.all(np.isfinite(x))),
            "history_length": history_length,
            "Rg_vs_042_abs_error": rg_042_error,
            "p_from_theta": p_theta,
            "p_from_042_max_abs_error": p_error,
            "legacy_scalar_max_abs_error": legacy_scalar_max_abs_error,
            "argmax_verified_pair_rows": tie_stats["argmax_verified_pair_rows"],
            "map_tie_pair_rows": tie_stats["map_tie_pair_rows"],
            "map_tie_records": tie_stats["map_tie_records"],
            "inter_min_distance_max_abs_error": tie_stats["inter_min_distance_max_abs_error"],
            "intra_map_nonmin_pair_rows": tie_stats["intra_map_nonmin_pair_rows"],
            "fixed_choice_changed_pair_rows": changed_pair_rows,
            "fixed_choice_changed_records": changed_records,
        })
        log(f"iteration {iteration}: MAP intra={row['distance_intra_map']:.8f} inter={row['distance_inter_map']:.8f} Rg={rg:.8f} ties={row['map_tie_records']}")

    if fixed_choices is None:
        raise ValueError("no iter10 MAP choice was established")
    published_error = float(np.max(np.abs(x - published)))
    published_rms = float(np.sqrt(np.mean((x - published) ** 2)))
    published_match = bool(published_error <= SERIALIZATION_TOL)
    if not published_match:
        raise ValueError(f"last checkpoint differs from published coordinates by {published_error}")
    max_inter_min_error = max(float(row["inter_min_distance_max_abs_error"]) for row in validation_rows)
    if max_inter_min_error > INTER_MIN_DISTANCE_TOL:
        raise ValueError(
            f"inter MAP distance is not the minimum four-way distance within tolerance: {max_inter_min_error}")
    if fixed_choice_consistency_error != 0:
        raise AssertionError("iter10 fixed MAP choices do not match iter10 main MAP choices")
    fieldnames = [
        "iteration", "checkpoint_path", "checkpoint_sha256", "p", "count_nll_normalized", "count_nll_raw", "totalJ",
        "count_intra", "count_inter",
        "distance_intra_map", "distance_inter_map", "distance_all_map",
        "distance_intra_fixed_map", "distance_inter_fixed_map", "distance_all_fixed_map",
        "distance_intra_map_over_Rg", "distance_inter_map_over_Rg", "distance_all_map_over_Rg",
        "distance_intra_fixed_map_over_Rg", "distance_inter_fixed_map_over_Rg", "distance_all_fixed_map_over_Rg",
        "Rg", "Rg_vs_042_abs_error", "center_x", "center_y", "center_z", "max_bead_radius",
        "map_tie_pair_rows", "map_tie_pair_rows_intra", "map_tie_pair_rows_inter",
        "map_tie_records", "map_tie_records_intra", "map_tie_records_inter",
        "argmax_verified_pair_rows", "inter_min_distance_max_abs_error", "intra_map_nonmin_pair_rows",
        "fixed_choice_changed_pair_rows", "fixed_choice_changed_records",
    ]
    with (OUT / "metrics.tsv").open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plot_path = make_plot(rows)

    input_hashes = {
        "snpfree_pairs": {"path": str(SNPFREE), "sha256": sha256_file(SNPFREE)},
        "legacy_042_metrics": {"path": str(LEGACY_042), "sha256": sha256_file(LEGACY_042)},
        "checkpoint_files": [{"path": str(path), "sha256": sha256_file(path)} for path in checkpoint_paths],
        "published_final_coordinates": {"path": str(PUBLISHED_COORDS), "sha256": sha256_file(PUBLISHED_COORDS)},
        "loader_sources": {
            "pr/contact_model.py": sha256_file(ROOT / "pr/contact_model.py"),
            "pr/genome.py": sha256_file(ROOT / "pr/genome.py"),
            "pr/pairs7.py": sha256_file(ROOT / "pr/pairs7.py"),
        },
    }
    primary_metrics = {}
    for metric in ("map", "fixed_map"):
        primary_metrics[metric] = {
            group: {
                "start": float(rows[0][f"distance_{group}_{metric}"]),
                "end": float(rows[-1][f"distance_{group}_{metric}"]),
                "change_percent": relative_percent(rows[0][f"distance_{group}_{metric}"], rows[-1][f"distance_{group}_{metric}"]),
                "monotonicity": monotonic_flags([row[f"distance_{group}_{metric}"] for row in rows]),
            }
            for group in ("intra", "inter", "all")
        }
    summary = {
        "run_id": OUT.name,
        "status": "completed_postprocessing",
        "execution": {"exit_code": 0, "status": "completed_postprocessing"},
        "fit_started": False,
        "model": "036 C0 random_joint 1Mb",
        "coordinate_units": "model dimensionless physical sphere coordinates",
        "dimensions": {"n_chromosomes": 20, "n_tracks": 40, "n_loci": 2645, "n_physical_beads": 5290},
        "source": {
            "checkpoint_root": str(CHECKPOINT_ROOT),
            "checkpoint_count": len(rows),
            "iterations": [int(row["iteration"]) for row in rows],
            "first_point_is": "accepted iteration 10; no fabricated iteration 0",
            "published_coordinate_manifest": str(PUBLISHED_COORDS),
            "fit_status": "budget_not_converged",
        },
        "contact_definition": {
            "kernel": "K(d)=1e-6+(1-1e-6)*(1+d^2/r0^2)^-2",
            "r0": float(data.r0),
            "score": "score_ab=w_ab*K(d_ab)",
            "weights": "intra AA/BB=p/2, AB/BA=(1-p)/2; inter all four=1/4",
            "p_source": "checkpoint theta[-1] through contact_model.p_from_q; scalar output reused from 042 metrics.tsv",
            "choice_order": list(CHOICE_LABELS),
            "tie_policy": "exact score equality only; np.argmax first choice in AA, AB, BA, BB order",
            "primary": "counts-weighted distance of one score-MAP copy combination per bin-pair",
            "fixed_control": "iter10 MAP copy combination held fixed across later geometry",
            "not_posterior_expectation": True,
            "not_true_allele_label": True,
            "counts_weighting": "observed raw counts weight bin-pairs; no average over unique pair rows",
            "same_bin_excluded": True,
            "known_phase_oracle": False,
        },
        "raw_budget": {
            "all_records": 1_703_888,
            "same_bin_excluded": 438_774,
            "intra_offdiag": 696_680,
            "inter": 568_434,
            "distance_denominator": 1_265_114,
            "all_counts_weighted": True,
        },
        "Rg_definition": "sqrt(mean(||x_bead-mean(x_all_5290)||^2)); all 5290 physical beads equal weight; no per-step scaling",
        "primary_metrics": primary_metrics,
        "distance_over_Rg_metrics": {
            metric: {
                group: {
                    "start": float(rows[0][f"distance_{group}_{metric}_over_Rg"]),
                    "end": float(rows[-1][f"distance_{group}_{metric}_over_Rg"]),
                    "change_percent": relative_percent(
                        rows[0][f"distance_{group}_{metric}_over_Rg"],
                        rows[-1][f"distance_{group}_{metric}_over_Rg"]),
                }
                for group in ("intra", "inter", "all")
            }
            for metric in ("map", "fixed_map")
        },
        "Rg_metrics": {
            "start": float(rows[0]["Rg"]),
            "end": float(rows[-1]["Rg"]),
            "change_percent": relative_percent(rows[0]["Rg"], rows[-1]["Rg"]),
            "monotonicity": monotonic_flags([row["Rg"] for row in rows]),
            "minimum": float(min(row["Rg"] for row in rows)),
            "maximum": float(max(row["Rg"] for row in rows)),
            "minimum_iteration": int(min(rows, key=lambda row: row["Rg"])["iteration"]),
            "maximum_iteration": int(max(rows, key=lambda row: row["Rg"])["iteration"]),
        },
        "tie_audit": {
            "records_meaning": "raw observed record mass, i.e. sum of counts on tied bin-pairs",
            "per_iteration": [
                {key: row[key] for key in (
                    "iteration", "map_tie_pair_rows", "map_tie_pair_rows_intra",
                    "map_tie_pair_rows_inter", "map_tie_records", "map_tie_records_intra",
                    "map_tie_records_inter", "intra_map_nonmin_pair_rows")}
                for row in rows
            ],
        },
        "scalar_reuse_audit": {
            "source": str(LEGACY_042),
            "max_abs_error_vs_checkpoint_history": legacy_scalar_max_abs_error,
            "p_theta_max_abs_error": p_theta_max_abs_error,
            "Rg_042_max_abs_error": rg_042_max_abs_error,
            "objective_recomputed": False,
        },
        "notes": [
            "This run changes only the reported contact-distance statistic; it does not change the four-pair marginalized likelihood used for training.",
            "A MAP copy combination is an inferred assignment for this statistic, not a true allele label.",
            "The original 036 trajectory stopped at budget_not_converged; this post-processing does not claim convergence.",
        ],
        "input_hashes": input_hashes,
        "artifacts": {
            "metrics_tsv": str(OUT / "metrics.tsv"),
            "plot_png": str(plot_path),
            "validation_json": str(OUT / "validation.json"),
        },
    }
    validation = {
        "status": "passed",
        "execution": {"exit_code": 0, "status": "completed_postprocessing"},
        "input_hashes": input_hashes,
        "checkpoint_schema": {
            "all_required_fields_present": True,
            "coordinates_shape": list(EXPECTED_SHAPE),
            "raw_y_shape": list(EXPECTED_SHAPE),
            "theta_length": EXPECTED_THETA,
            "stored_granularity": "accepted iterations every 10, 10..240",
            "checkpoint_count": len(checkpoint_paths),
        },
        "raw_budget": {
            "expected": expected_audit,
            "observed": {key: int(audit[key]) for key in expected_audit},
            "mask_and_rawcount_conserved": True,
            "distance_denominator": 1_265_114,
        },
        "finite_and_transform": {
            "all_rows_finite": True,
            "serialization_tolerance": SERIALIZATION_TOL,
            "rows": validation_rows,
        },
        "map_argmax": {
            "score_formula": "w_ab*K(d_ab)",
            "choice_order": list(CHOICE_LABELS),
            "exact_tie_only": True,
            "argmax_verified_pair_rows_per_checkpoint": data.n_pairs,
            "all_argmax_checks_passed": all(
                int(row["argmax_verified_pair_rows"]) == data.n_pairs for row in validation_rows),
            "inter_map_min_distance_max_abs_error": max_inter_min_error,
            "inter_map_min_distance_tolerance": INTER_MIN_DISTANCE_TOL,
            "inter_map_equals_min_distance": max_inter_min_error <= INTER_MIN_DISTANCE_TOL,
            "intra_min_distance_identity_forced": False,
            "intra_nonmin_pair_rows": [
                {"iteration": int(row["iteration"]), "count": int(row["intra_map_nonmin_pair_rows"])}
                for row in validation_rows
            ],
            "fixed_iter10_choice_consistent": fixed_choice_consistency_error == 0,
        },
        "tie_audit": {
            "exact_score_ties_only": True,
            "tie_order": list(CHOICE_LABELS),
            "observed_tie_records_are_counts_weighted": True,
            "per_iteration": [
                {key: row[key] for key in (
                    "iteration", "map_tie_pair_rows", "map_tie_pair_rows_intra",
                    "map_tie_pair_rows_inter", "map_tie_records", "map_tie_records_intra",
                    "map_tie_records_inter")}
                for row in rows
            ],
        },
        "scalar_reuse": {
            "legacy_042_metrics": str(LEGACY_042),
            "max_abs_error_vs_checkpoint_history": legacy_scalar_max_abs_error,
            "p_theta_max_abs_error": p_theta_max_abs_error,
            "Rg_042_max_abs_error": rg_042_max_abs_error,
            "Rg_pointwise_match": rg_042_max_abs_error <= SERIALIZATION_TOL,
            "objective_recomputed": False,
        },
        "last_checkpoint_vs_published": {
            "checkpoint": str(checkpoint_paths[-1]),
            "published": str(PUBLISHED_COORDS),
            "max_abs_difference": published_error,
            "rms_difference": published_rms,
            "published_rows": published_meta,
            "within_serialization_tolerance": published_match,
        },
        "no_forbidden_inputs_opened": ["reference", "phase", "evaluation outputs"],
        "allowed_read_only_diagnostic_input": str(LEGACY_042),
    }
    json_dump(OUT / "summary.json", summary)
    json_dump(OUT / "validation.json", validation)
    config = {
        "run_id": OUT.name,
        "command": "conda run -n analysis python analyze_contact_map_rg.py",
        "purpose": "post-processing only; score-MAP contact distance and whole-cell Rg; no refit and no objective reevaluation",
        "status": "completed_postprocessing",
        "execution": {"exit_code": 0, "status": "completed_postprocessing"},
        "data_boundary": {"training_input_only": True, "reference_opened": False, "phase_opened": False, "evaluation_opened": False, "allowed_read_only_diagnostic_input": str(LEGACY_042)},
        "parameters": {"bin_size_bp": BIN_SIZE, "block_size": BLOCK_SIZE, "epsilon": EPSILON, "r0": float(data.r0), "Rg_beads": 5290, "serialization_tolerance": SERIALIZATION_TOL, "inter_min_distance_tolerance": INTER_MIN_DISTANCE_TOL, "choice_order": list(CHOICE_LABELS)},
        "input_hashes": input_hashes,
        "artifacts": {"metrics": "metrics.tsv", "summary": "summary.json", "validation": "validation.json", "plot": "plots/map_contact_distance_rg.png", "log": "logs/run.log", "script": "analyze_contact_map_rg.py", "readme": "README.md"},
    }
    json_dump(OUT / "config.json", config)
    log(f"published-coordinate max abs difference: {published_error:.3e}")
    log(f"wrote {OUT / 'metrics.tsv'}")
    log(f"wrote {plot_path}")
    log("STATUS: completed_postprocessing")
    log("EXIT_CODE: 0")
    (OUT / "logs/run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
