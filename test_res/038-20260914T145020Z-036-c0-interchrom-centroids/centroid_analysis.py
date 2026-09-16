#!/usr/bin/env python
"""036 C0 endpoint 的独立染色体间 centroid 评估。

本评价器有意独立于旧版 R2 pair-mask 代码。它只使用 candidate A/B 与 reference mat/pat 中有限的精确整数 1 Mb 位置，然后计算染色体-拷贝中心和等权染色体中心。只有调用者写出 endpoint 哈希/选择锁后，本评价器才会打开 reference。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
CANDIDATE = ROOT / "test_res/036-20260914T064651Z-gpu-multires/coords/C0/random_joint/final-1m.3dg"
REFERENCE = ROOT / "data/P9016.1m.3dg.gz"
EXPECTED_CHROMS = ["chr%d" % i for i in range(1, 20)] + ["chrX"]
BIN = 1_000_000
NCHR = 20
N_PERM = 9999
PERM_SEED = 20260914


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def log(message: str) -> None:
    text = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message)
    print(text, flush=True)
    log_path = RUN / "logs" / "analysis.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def parse_3dg(path: Path) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    """解析 native 3DG 行，保留整数位置和轨迹身份。"""
    result: dict[str, dict[int, np.ndarray]] = {}
    bad_rows = 0
    duplicate_rows = 0
    nonfinite_rows = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                bad_rows += 1
                continue
            try:
                track = str(fields[0])
                position = int(fields[1])
                point = np.asarray([float(item) for item in fields[2:5]], dtype=np.float64)
            except (TypeError, ValueError):
                bad_rows += 1
                continue
            target = result.setdefault(track, {})
            if position in target:
                duplicate_rows += 1
            if not np.isfinite(point).all():
                nonfinite_rows += 1
                point = np.full(3, np.nan, dtype=np.float64)
            target[position] = point
    if bad_rows or duplicate_rows:
        raise ValueError("invalid 3DG rows in %s: bad=%d duplicate=%d" % (path, bad_rows, duplicate_rows))
    return result, {
        "path": str(path),
        "sha256": sha256(path),
        "n_tracks": len(result),
        "n_rows": int(sum(len(v) for v in result.values())),
        "bad_rows": bad_rows,
        "duplicate_rows": duplicate_rows,
        "nonfinite_rows": nonfinite_rows,
    }


def chr_index(chromosome: str) -> int:
    if chromosome == "chrX":
        return 19
    if chromosome.startswith("chr") and chromosome[3:].isdigit():
        value = int(chromosome[3:])
        if 1 <= value <= 19:
            return value - 1
    raise ValueError("unexpected chromosome label: %s" % chromosome)


def track_maps(structure: dict[str, dict[int, np.ndarray]], kind: str) -> dict[str, list[str]]:
    if kind == "candidate":
        pattern = re.compile(r"^c(\d{2})([ab])$")
        observed: dict[str, dict[int, str]] = {}
        for track in structure:
            match = pattern.match(track)
            if match is None:
                raise ValueError("unexpected candidate track: %s" % track)
            index = int(match.group(1))
            copy = match.group(2)
            if not 1 <= index <= 20:
                raise ValueError("candidate track index out of range: %s" % track)
            observed.setdefault("chrX" if index == 20 else "chr%d" % index, {})[0 if copy == "a" else 1] = track
        output = {}
        for chromosome in EXPECTED_CHROMS:
            if chromosome not in observed or set(observed[chromosome]) != {0, 1}:
                raise ValueError("candidate track map incomplete for %s" % chromosome)
            output[chromosome] = [observed[chromosome][0], observed[chromosome][1]]
        if set(structure) != {track for tracks in output.values() for track in tracks}:
            raise ValueError("candidate has unexpected or missing tracks")
        return output
    pattern = re.compile(r"^(chr(?:[1-9]|1[0-9])|chrX)\((mat|pat)\)$")
    observed: dict[str, dict[str, str]] = {}
    for track in structure:
        match = pattern.match(track)
        if match is None:
            raise ValueError("unexpected reference track: %s" % track)
        observed.setdefault(match.group(1), {})[match.group(2)] = track
    output = {}
    for chromosome in EXPECTED_CHROMS:
        if chromosome not in observed or set(observed[chromosome]) != {"mat", "pat"}:
            raise ValueError("reference track map incomplete for %s" % chromosome)
        output[chromosome] = [observed[chromosome]["mat"], observed[chromosome]["pat"]]
    if set(structure) != {track for tracks in output.values() for track in tracks}:
        raise ValueError("reference has unexpected or missing tracks")
    return output


def track_diagnostics(structure: dict[str, dict[int, np.ndarray]], track_map: dict[str, list[str]], kind: str) -> list[dict[str, Any]]:
    rows = []
    for chromosome in EXPECTED_CHROMS:
        for copy_index, track in enumerate(track_map[chromosome]):
            positions = np.asarray(sorted(structure[track]), dtype=np.int64)
            steps = np.diff(positions) if len(positions) > 1 else np.asarray([], dtype=np.int64)
            step_ok = bool(len(positions) > 0 and np.all(positions % BIN == 0))
            expected = np.arange(int(positions.min()), int(positions.max()) + BIN, BIN, dtype=np.int64) if len(positions) else positions
            regular_range = bool(np.array_equal(positions, expected))
            rows.append({
                "structure": kind,
                "chromosome": chromosome,
                "copy": ["A", "B"][copy_index] if kind == "candidate" else ["mat", "pat"][copy_index],
                "track": track,
                "n_rows": int(len(positions)),
                "min_position_bp": int(positions.min()) if len(positions) else None,
                "max_position_bp": int(positions.max()) if len(positions) else None,
                "regular_1Mb_within_observed_range": regular_range,
                "positions_are_1Mb_grid": step_ok,
                "finite_rows": int(sum(np.isfinite(structure[track][int(p)]).all() for p in positions)),
                "nonfinite_rows": int(sum(not np.isfinite(structure[track][int(p)]).all() for p in positions)),
            })
    return rows


def correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 2 or len(y) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    if method == "spearman":
        value = spearmanr(x, y).statistic
    else:
        value = pearsonr(x, y).statistic
    return float(value) if np.isfinite(value) else float("nan")


def fit_similarity(source: np.ndarray, target: np.ndarray, proper_rotation: bool = False) -> dict[str, Any]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    x = source - source_mean
    y = target - target_mean
    u, singular, vt = np.linalg.svd(x.T @ y)
    raw_r = vt.T @ u.T
    d = float(np.linalg.det(raw_r))
    correction = np.eye(3)
    if proper_rotation and d < 0:
        correction[-1, -1] = -1.0
    r = vt.T @ correction @ u.T
    scale_numerator = float(np.sum(singular * np.diag(correction)))
    scale_denominator = float(np.sum(x * x))
    scale = scale_numerator / scale_denominator
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid similarity scale: %r" % scale)
    translation = target_mean - scale * (r @ source_mean)
    transformed = scale * (source @ r.T) + translation
    residual = transformed - target
    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "rotation_matrix": r,
        "singular_values": singular,
        "scale": float(scale),
        "translation": translation,
        "determinant": float(np.linalg.det(r)),
        "raw_unconstrained_determinant": d,
        "transformed_centers": transformed,
        "residuals": residual,
        "rmsd": float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }


def apply_similarity(points: np.ndarray, alignment: dict[str, Any]) -> np.ndarray:
    return alignment["scale"] * (points @ alignment["rotation_matrix"].T) + alignment["translation"]


def distance_matrix(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(delta, axis=2)


def upper_vector(matrix: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(matrix.shape[0], 1)
    return matrix[i, j]


def scale_and_stress(pred: np.ndarray, ref: np.ndarray, scale: float | None = None) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if scale is None:
        scale = float(np.dot(pred, ref) / np.dot(pred, pred))
    residual = float(np.sqrt(np.sum((scale * pred - ref) ** 2) / np.sum(ref ** 2)))
    return {"scale": float(scale), "stress": residual}


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "%.12g" % float(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"))
    return str(value)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def write_3dg(path: Path, structure: dict[str, dict[int, np.ndarray]], transform: dict[str, Any] | None = None) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# generated by centroid_analysis.py; numeric positions preserved\n")
        for track in sorted(structure):
            for position in sorted(structure[track]):
                point = structure[track][position]
                if transform is not None:
                    point = apply_similarity(point[None, :], transform)[0]
                handle.write("%s\t%d\t%.12g\t%.12g\t%.12g\n" % (track, position, point[0], point[1], point[2]))


def core_stage() -> None:
    log("core stage: checking locked hashes before parsing reference")
    hashes = json.loads((RUN / "input_hashes.json").read_text(encoding="utf-8"))
    candidate_hash = sha256(CANDIDATE)
    reference_hash = sha256(REFERENCE)
    if candidate_hash != hashes["files"]["candidate_endpoint"]["sha256"]:
        raise ValueError("candidate endpoint hash changed")
    if reference_hash != hashes["files"]["reference"]["sha256"]:
        raise ValueError("reference hash changed")
    selection = json.loads((ROOT / "test_res/036-20260914T064651Z-gpu-multires/selection.json").read_text(encoding="utf-8"))
    if selection["selection"]["per_variant"]["C0"]["selected_id"] != "random_joint":
        raise ValueError("C0 selection is not random_joint")
    if selection["selection"]["per_variant"]["C0"]["criterion"] != "count_nll_per_record":
        raise ValueError("unexpected C0 selection criterion")
    if selection["selection"]["phase_used"] or selection["selection"]["reference_used"]:
        raise ValueError("selection unexpectedly used phase/reference")
    stage = json.loads((ROOT / "test_res/036-20260914T064651Z-gpu-multires/stages/C0/random_joint/1m.json").read_text(encoding="utf-8"))
    if stage["status"] != "not_converged" or stage["fit"]["termination_reason"] != "budget_exhausted":
        raise ValueError("unexpected 1 Mb termination evidence")
    log("core stage: endpoint and selection locks pass; now opening reference")

    candidate, candidate_meta = parse_3dg(CANDIDATE)
    reference, reference_meta = parse_3dg(REFERENCE)
    candidate_map = track_maps(candidate, "candidate")
    reference_map = track_maps(reference, "reference")
    candidate_diag = track_diagnostics(candidate, candidate_map, "candidate")
    reference_diag = track_diagnostics(reference, reference_map, "reference")
    log("parsed candidate tracks=%d rows=%d; reference tracks=%d rows=%d" % (candidate_meta["n_tracks"], candidate_meta["n_rows"], reference_meta["n_tracks"], reference_meta["n_rows"]))

    positions_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    candidate_copy_centers = np.full((NCHR, 2, 3), np.nan, dtype=np.float64)
    reference_copy_centers = np.full((NCHR, 2, 3), np.nan, dtype=np.float64)
    candidate_chr_centers = np.full((NCHR, 3), np.nan, dtype=np.float64)
    reference_chr_centers = np.full((NCHR, 3), np.nan, dtype=np.float64)
    shared_positions: dict[str, list[int]] = {}
    mask_summary: list[dict[str, Any]] = []
    labels = [("candidate_A", 0), ("candidate_B", 1), ("reference_mat", 0), ("reference_pat", 1)]
    for ci, chromosome in enumerate(EXPECTED_CHROMS):
        c_tracks = candidate_map[chromosome]
        r_tracks = reference_map[chromosome]
        union = sorted(set(candidate[c_tracks[0]]) | set(candidate[c_tracks[1]]) | set(reference[r_tracks[0]]) | set(reference[r_tracks[1]]))
        shared: list[int] = []
        excluded_reasons: dict[str, int] = {}
        for position in union:
            values = [candidate[c_tracks[0]].get(position), candidate[c_tracks[1]].get(position), reference[r_tracks[0]].get(position), reference[r_tracks[1]].get(position)]
            present = [value is not None and np.isfinite(value).all() for value in values]
            missing = [labels[index][0] for index, ok in enumerate(present) if not ok]
            is_shared = not missing
            reason = "" if is_shared else "missing:" + ",".join(missing)
            if is_shared:
                shared.append(position)
            else:
                excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
            positions_rows.append({
                "chromosome": chromosome,
                "position_bp": position,
                "candidate_A_present": int(present[0]),
                "candidate_B_present": int(present[1]),
                "reference_mat_present": int(present[2]),
                "reference_pat_present": int(present[3]),
                "shared": int(is_shared),
                "exclude_reason": reason,
            })
        if not shared:
            raise ValueError("no shared finite positions for %s" % chromosome)
        shared_positions[chromosome] = shared
        for copy_index, track in enumerate(c_tracks):
            candidate_copy_centers[ci, copy_index] = np.vstack([candidate[track][p] for p in shared]).mean(axis=0)
        for copy_index, track in enumerate(r_tracks):
            reference_copy_centers[ci, copy_index] = np.vstack([reference[track][p] for p in shared]).mean(axis=0)
        candidate_chr_centers[ci] = candidate_copy_centers[ci].mean(axis=0)
        reference_chr_centers[ci] = reference_copy_centers[ci].mean(axis=0)
        raw_counts = [len(candidate[t]) for t in c_tracks] + [len(reference[t]) for t in r_tracks]
        mask_summary.append({
            "chromosome": chromosome,
            "candidate_A_track": c_tracks[0],
            "candidate_B_track": c_tracks[1],
            "reference_mat_track": r_tracks[0],
            "reference_pat_track": r_tracks[1],
            "candidate_A_raw": raw_counts[0],
            "candidate_B_raw": raw_counts[1],
            "reference_mat_raw": raw_counts[2],
            "reference_pat_raw": raw_counts[3],
            "union_positions": len(union),
            "shared_positions": len(shared),
            "excluded_positions": len(union) - len(shared),
            "excluded_reason_counts": excluded_reasons,
            "shared_min_bp": min(shared),
            "shared_max_bp": max(shared),
        })

    write_tsv(RUN / "coordinates" / "positions.tsv", ["chromosome", "position_bp", "candidate_A_present", "candidate_B_present", "reference_mat_present", "reference_pat_present", "shared", "exclude_reason"], positions_rows)
    log("mask complete: shared positions=%d/%d union positions" % (sum(x["shared_positions"] for x in mask_summary), sum(x["union_positions"] for x in mask_summary)))

    primary_pred_matrix = distance_matrix(candidate_chr_centers)
    primary_ref_matrix = distance_matrix(reference_chr_centers)
    pred190 = upper_vector(primary_pred_matrix)
    ref190 = upper_vector(primary_ref_matrix)
    distance_scale = scale_and_stress(pred190, ref190)
    primary_spearman = correlation(pred190, ref190, "spearman")
    primary_pearson = correlation(pred190, ref190, "pearson")
    upper_i, upper_j = np.triu_indices(NCHR, 1)
    pair_rows: list[dict[str, Any]] = []
    for pair_index, (i, j) in enumerate(zip(upper_i, upper_j)):
        pair_rows.append({
            "pair_index": pair_index,
            "chromosome_i": EXPECTED_CHROMS[i],
            "chromosome_j": EXPECTED_CHROMS[j],
            "candidate_distance_raw": pred190[pair_index],
            "reference_distance": ref190[pair_index],
            "candidate_distance_scaled_by_primary_a": distance_scale["scale"] * pred190[pair_index],
            "scaled_residual_candidate_minus_reference": distance_scale["scale"] * pred190[pair_index] - ref190[pair_index],
            "chr1_involved": int(i == 0 or j == 0),
        })
    write_tsv(RUN / "pairs190.tsv", ["pair_index", "chromosome_i", "chromosome_j", "candidate_distance_raw", "reference_distance", "candidate_distance_scaled_by_primary_a", "scaled_residual_candidate_minus_reference", "chr1_involved"], pair_rows)

    rng = np.random.default_rng(PERM_SEED)
    null_spearman = np.empty(N_PERM, dtype=np.float64)
    for draw in range(N_PERM):
        permutation = rng.permutation(NCHR)
        permuted = primary_pred_matrix[np.ix_(permutation, permutation)]
        null_spearman[draw] = correlation(upper_vector(permuted), ref190, "spearman")
    permutation_quantiles = {"q025": float(np.quantile(null_spearman, 0.025)), "q50": float(np.quantile(null_spearman, 0.5)), "q975": float(np.quantile(null_spearman, 0.975))}
    descriptive_p = float((1 + np.count_nonzero(null_spearman >= primary_spearman)) / (N_PERM + 1))
    with (RUN / "permutation_null_spearman.tsv").open("w", encoding="utf-8") as handle:
        handle.write("draw\tspearman\n")
        for index, value in enumerate(null_spearman):
            handle.write("%d\t%.12g\n" % (index, value))

    free_alignment = fit_similarity(candidate_chr_centers, reference_chr_centers, proper_rotation=False)
    proper_alignment = fit_similarity(candidate_chr_centers, reference_chr_centers, proper_rotation=True)
    reference_center_mean = reference_chr_centers.mean(axis=0)
    reference_rg = float(np.sqrt(np.mean(np.sum((reference_chr_centers - reference_center_mean) ** 2, axis=1))))
    free_alignment["rmsd_over_reference_center_rg"] = free_alignment["rmsd"] / reference_rg
    proper_alignment["rmsd_over_reference_center_rg"] = proper_alignment["rmsd"] / reference_rg
    candidate_copy_aligned = apply_similarity(candidate_copy_centers.reshape(-1, 3), free_alignment).reshape(NCHR, 2, 3)
    candidate_chr_aligned = apply_similarity(candidate_chr_centers, free_alignment)

    swaps: list[dict[str, Any]] = []
    swap_indices = np.zeros(NCHR, dtype=np.int8)
    for ci, chromosome in enumerate(EXPECTED_CHROMS):
        direct = float(np.sum((candidate_copy_aligned[ci, 0] - reference_copy_centers[ci, 0]) ** 2) + np.sum((candidate_copy_aligned[ci, 1] - reference_copy_centers[ci, 1]) ** 2))
        swapped = float(np.sum((candidate_copy_aligned[ci, 0] - reference_copy_centers[ci, 1]) ** 2) + np.sum((candidate_copy_aligned[ci, 1] - reference_copy_centers[ci, 0]) ** 2))
        choice = int(swapped < direct)
        swap_indices[ci] = choice
        swaps.append({
            "chromosome": chromosome,
            "swap": choice,
            "candidate_A_maps_to": "pat" if choice else "mat",
            "candidate_B_maps_to": "mat" if choice else "pat",
            "direct_sse": direct,
            "swapped_sse": swapped,
            "chosen_sse": min(direct, swapped),
            "tie": int(np.isclose(direct, swapped, rtol=0.0, atol=1e-15)),
        })

    def mapped_reference_copy(ci: int, candidate_copy: int) -> int:
        return candidate_copy if swap_indices[ci] == 0 else 1 - candidate_copy

    copy_rows: list[dict[str, Any]] = []
    gauge_free_rows: list[dict[str, Any]] = []
    gauge_free_pred_blocks: list[np.ndarray] = []
    gauge_free_ref_blocks: list[np.ndarray] = []
    copy_pred_raw: list[float] = []
    copy_ref: list[float] = []
    copy_pred_aligned: list[float] = []
    for i, j in zip(upper_i, upper_j):
        pred_four_raw = []
        ref_four = []
        for copy_i in (0, 1):
            for copy_j in (0, 1):
                pred_raw = float(np.linalg.norm(candidate_copy_centers[i, copy_i] - candidate_copy_centers[j, copy_j]))
                pred_aligned = float(np.linalg.norm(candidate_copy_aligned[i, copy_i] - candidate_copy_aligned[j, copy_j]))
                ref_i = mapped_reference_copy(i, copy_i)
                ref_j = mapped_reference_copy(j, copy_j)
                ref_distance = float(np.linalg.norm(reference_copy_centers[i, ref_i] - reference_copy_centers[j, ref_j]))
                copy_pred_raw.append(pred_raw)
                copy_pred_aligned.append(pred_aligned)
                copy_ref.append(ref_distance)
                copy_rows.append({
                    "chromosome_i": EXPECTED_CHROMS[i],
                    "chromosome_j": EXPECTED_CHROMS[j],
                    "copy_i": "A" if copy_i == 0 else "B",
                    "copy_j": "A" if copy_j == 0 else "B",
                    "reference_copy_i": "mat" if ref_i == 0 else "pat",
                    "reference_copy_j": "mat" if ref_j == 0 else "pat",
                    "candidate_distance_raw": pred_raw,
                    "candidate_distance_after_global_similarity": pred_aligned,
                    "reference_distance": ref_distance,
                    "candidate_distance_raw_scaled_by_primary_a20": distance_scale["scale"] * pred_raw,
                    "primary_a20_scaled_residual": distance_scale["scale"] * pred_raw - ref_distance,
                    "aligned_unit_residual": pred_aligned - ref_distance,
                })
                pred_four_raw.append(pred_raw)
                ref_four.append(ref_distance)
        sorted_pred = np.sort(np.asarray(pred_four_raw))
        sorted_ref = np.sort(np.asarray(ref_four))
        gauge_free_pred_blocks.append(sorted_pred)
        gauge_free_ref_blocks.append(sorted_ref)
        gauge_free_rows.append({
            "chromosome_i": EXPECTED_CHROMS[i],
            "chromosome_j": EXPECTED_CHROMS[j],
            "sorted_candidate_distances_raw": sorted_pred.tolist(),
            "sorted_reference_distances": sorted_ref.tolist(),
            "block_n": 4,
        })
    write_tsv(RUN / "copy_pairs760.tsv", ["chromosome_i", "chromosome_j", "copy_i", "copy_j", "reference_copy_i", "reference_copy_j", "candidate_distance_raw", "candidate_distance_after_global_similarity", "reference_distance", "candidate_distance_raw_scaled_by_primary_a20", "primary_a20_scaled_residual", "aligned_unit_residual"], copy_rows)
    write_tsv(RUN / "copy_gauge_free190.tsv", ["chromosome_i", "chromosome_j", "sorted_candidate_distances_raw", "sorted_reference_distances", "block_n"], gauge_free_rows)
    gauge_free_pred_concat = np.concatenate(gauge_free_pred_blocks)
    gauge_free_ref_concat = np.concatenate(gauge_free_ref_blocks)
    gauge_free_pooled = {
        "n_values": int(len(gauge_free_pred_concat)),
        "n_sorted_blocks": int(len(gauge_free_pred_blocks)),
        "block_size": 4,
        "spearman": correlation(gauge_free_pred_concat, gauge_free_ref_concat, "spearman"),
        "pearson": correlation(gauge_free_pred_concat, gauge_free_ref_concat, "pearson"),
        "primary_a20_stress": scale_and_stress(gauge_free_pred_concat, gauge_free_ref_concat, distance_scale["scale"])["stress"],
        "primary_a20": distance_scale["scale"],
        "interpretation": "190 chromosome-pair blocks independently sorted within each four-distance block, then concatenated to 760 values; sorting induces structural rank correlation and this is gauge-free supplemental evidence only, not proof of consistent copy matching",
    }
    copy_pred_raw_array = np.asarray(copy_pred_raw)
    copy_pred_aligned_array = np.asarray(copy_pred_aligned)
    copy_ref_array = np.asarray(copy_ref)
    copy_primary_stress = scale_and_stress(copy_pred_raw_array, copy_ref_array, distance_scale["scale"])
    copy_aligned_stress = scale_and_stress(copy_pred_aligned_array, copy_ref_array, 1.0)
    copy_best_scale = scale_and_stress(copy_pred_raw_array, copy_ref_array)
    homolog_rows = []
    for ci, chromosome in enumerate(EXPECTED_CHROMS):
        homolog_rows.append({
            "chromosome": chromosome,
            "candidate_homolog_separation_raw": float(np.linalg.norm(candidate_copy_centers[ci, 0] - candidate_copy_centers[ci, 1])),
            "candidate_homolog_separation_after_global_similarity": float(np.linalg.norm(candidate_copy_aligned[ci, 0] - candidate_copy_aligned[ci, 1])),
            "reference_homolog_separation": float(np.linalg.norm(reference_copy_centers[ci, 0] - reference_copy_centers[ci, 1])),
        })

    per_chromosome_rows: list[dict[str, Any]] = []
    for ci, chromosome in enumerate(EXPECTED_CHROMS):
        other = [index for index in range(NCHR) if index != ci]
        candidate_dist = primary_pred_matrix[ci, other]
        reference_dist = primary_ref_matrix[ci, other]
        candidate_order = [other[index] for index in np.argsort(candidate_dist, kind="stable")[:3]]
        reference_order = [other[index] for index in np.argsort(reference_dist, kind="stable")[:3]]
        per_chromosome_rows.append({
            **mask_summary[ci],
            "candidate_center_x": candidate_chr_centers[ci, 0],
            "candidate_center_y": candidate_chr_centers[ci, 1],
            "candidate_center_z": candidate_chr_centers[ci, 2],
            "reference_center_x": reference_chr_centers[ci, 0],
            "reference_center_y": reference_chr_centers[ci, 1],
            "reference_center_z": reference_chr_centers[ci, 2],
            "distance_n": len(other),
            "distance_spearman_19": correlation(candidate_dist, reference_dist, "spearman"),
            "distance_pearson_19": correlation(candidate_dist, reference_dist, "pearson"),
            "distance_stress_19_using_primary_a20": scale_and_stress(candidate_dist, reference_dist, distance_scale["scale"])["stress"],
            "top3_candidate_nearest": [EXPECTED_CHROMS[index] for index in candidate_order],
            "top3_reference_nearest": [EXPECTED_CHROMS[index] for index in reference_order],
            "top3_overlap": len(set(candidate_order) & set(reference_order)),
            "is_chr1": int(ci == 0),
            "candidate_homolog_separation_raw": homolog_rows[ci]["candidate_homolog_separation_raw"],
            "reference_homolog_separation": homolog_rows[ci]["reference_homolog_separation"],
        })
    write_tsv(RUN / "per_chromosome.tsv", ["chromosome", "candidate_A_track", "candidate_B_track", "reference_mat_track", "reference_pat_track", "candidate_A_raw", "candidate_B_raw", "reference_mat_raw", "reference_pat_raw", "union_positions", "shared_positions", "excluded_positions", "excluded_reason_counts", "shared_min_bp", "shared_max_bp", "candidate_center_x", "candidate_center_y", "candidate_center_z", "reference_center_x", "reference_center_y", "reference_center_z", "distance_n", "distance_spearman_19", "distance_pearson_19", "distance_stress_19_using_primary_a20", "top3_candidate_nearest", "top3_reference_nearest", "top3_overlap", "is_chr1", "candidate_homolog_separation_raw", "reference_homolog_separation"], per_chromosome_rows)

    aligned_rows: list[dict[str, Any]] = []
    for ci, chromosome in enumerate(EXPECTED_CHROMS):
        for copy_index, track in enumerate(candidate_map[chromosome]):
            for position in sorted(candidate[track]):
                raw = candidate[track][position]
                aligned = apply_similarity(raw[None, :], free_alignment)[0]
                aligned_rows.append({
                    "source": "candidate",
                    "track": track,
                    "reference_track": "",
                    "chromosome": chromosome,
                    "copy": "A" if copy_index == 0 else "B",
                    "position_bp": position,
                    "original_x": raw[0], "original_y": raw[1], "original_z": raw[2],
                    "aligned_x": aligned[0], "aligned_y": aligned[1], "aligned_z": aligned[2],
                    "shared_mask_position": int(position in shared_positions[chromosome]),
                })
        for copy_index, track in enumerate(reference_map[chromosome]):
            for position in sorted(reference[track]):
                raw = reference[track][position]
                aligned_rows.append({
                    "source": "reference",
                    "track": track,
                    "reference_track": track,
                    "chromosome": chromosome,
                    "copy": "mat" if copy_index == 0 else "pat",
                    "position_bp": position,
                    "original_x": raw[0], "original_y": raw[1], "original_z": raw[2],
                    "aligned_x": raw[0], "aligned_y": raw[1], "aligned_z": raw[2],
                    "shared_mask_position": int(position in shared_positions[chromosome]),
                })
    write_tsv(RUN / "coordinates" / "aligned_coordinates.tsv", ["source", "track", "reference_track", "chromosome", "copy", "position_bp", "original_x", "original_y", "original_z", "aligned_x", "aligned_y", "aligned_z", "shared_mask_position"], aligned_rows)
    write_3dg(RUN / "coordinates" / "candidate_aligned.3dg", candidate, free_alignment)
    write_3dg(RUN / "coordinates" / "reference.3dg", reference)

    center_rows = []
    for ci, chromosome in enumerate(EXPECTED_CHROMS):
        n_shared = len(shared_positions[chromosome])
        for copy_index in (0, 1):
            c = candidate_copy_centers[ci, copy_index]
            ca = candidate_copy_aligned[ci, copy_index]
            r = reference_copy_centers[ci, copy_index]
            center_rows.append({"structure": "candidate", "chromosome": chromosome, "copy": "A" if copy_index == 0 else "B", "center_role": "copy_centroid", "n_shared_positions": n_shared, "raw_x": c[0], "raw_y": c[1], "raw_z": c[2], "aligned_x": ca[0], "aligned_y": ca[1], "aligned_z": ca[2]})
            center_rows.append({"structure": "reference", "chromosome": chromosome, "copy": "mat" if copy_index == 0 else "pat", "center_role": "copy_centroid", "n_shared_positions": n_shared, "raw_x": r[0], "raw_y": r[1], "raw_z": r[2], "aligned_x": r[0], "aligned_y": r[1], "aligned_z": r[2]})
        c = candidate_chr_centers[ci]
        ca = candidate_chr_aligned[ci]
        r = reference_chr_centers[ci]
        center_rows.append({"structure": "candidate", "chromosome": chromosome, "copy": "A+B_equal_weight", "center_role": "chromosome_centroid", "n_shared_positions": n_shared, "raw_x": c[0], "raw_y": c[1], "raw_z": c[2], "aligned_x": ca[0], "aligned_y": ca[1], "aligned_z": ca[2]})
        center_rows.append({"structure": "reference", "chromosome": chromosome, "copy": "mat+pat_equal_weight", "center_role": "chromosome_centroid", "n_shared_positions": n_shared, "raw_x": r[0], "raw_y": r[1], "raw_z": r[2], "aligned_x": r[0], "aligned_y": r[1], "aligned_z": r[2]})
    write_tsv(RUN / "centers.tsv", ["structure", "chromosome", "copy", "center_role", "n_shared_positions", "raw_x", "raw_y", "raw_z", "aligned_x", "aligned_y", "aligned_z"], center_rows)

    def alignment_json(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "scale": record["scale"],
            "translation": record["translation"],
            "rotation_matrix_column_vector": record["rotation_matrix"],
            "determinant": record["determinant"],
            "raw_unconstrained_determinant": record["raw_unconstrained_determinant"],
            "singular_values": record["singular_values"],
            "source_mean": record["source_mean"],
            "target_mean": record["target_mean"],
            "rmsd": record["rmsd"],
            "max_abs_residual": record["max_abs_residual"],
            "rmsd_over_reference_center_rg": record["rmsd_over_reference_center_rg"],
            "formula": "y = scale * (R @ x) + translation for column vectors",
            "fit_unit": "20 equal-weight chromosome centers",
        }
    alignment_obj = {
        "schema": "p9016-c0-centroid-similarity-alignment-v1",
        "fit": alignment_json(free_alignment),
        "proper_rotation_only_sensitivity": alignment_json(proper_alignment),
        "application": "same candidate-to-reference transform applied to every candidate bead and all 40 candidate copy centroids",
        "reflection_allowed": True,
        "per_chromosome_fit": False,
        "distance_scale_separate": {"a20": distance_scale["scale"], "definition": "a20=sum(pred_chromosome_distance*ref_distance)/sum(pred_chromosome_distance^2)"},
        "copy_swap_gauge": swaps,
        "copy_swap_total_chosen_sse": float(sum(row["chosen_sse"] for row in swaps)),
    }
    write_json(RUN / "alignment.json", alignment_obj)

    # 核心 arrays 使绘图和独立验证无需重新读取 prose，且保持确定性。
    np.savez_compressed(RUN / "core_data.npz", candidate_centers=candidate_chr_centers, reference_centers=reference_chr_centers, candidate_copy_centers=candidate_copy_centers, reference_copy_centers=reference_copy_centers, candidate_copy_aligned=candidate_copy_aligned, reference_distance_matrix=primary_ref_matrix, candidate_distance_matrix=primary_pred_matrix, candidate_distance_matrix_scaled=distance_scale["scale"] * primary_pred_matrix, candidate_chr_aligned=candidate_chr_aligned, null_spearman=null_spearman, pair_i=upper_i, pair_j=upper_j)

    metrics = {
        "schema": "p9016-c0-interchrom-centroids-metrics-v1",
        "identity": {"sample_id": "P9016", "biological_replicates": 1, "candidate_model": "C0", "candidate_id": "random_joint", "candidate_base_seed": 2207, "selection_criterion": "count_nll_per_record", "candidate_path": str(CANDIDATE), "reference_path": str(REFERENCE)},
        "input_meta": {"candidate": candidate_meta, "reference": reference_meta, "candidate_track_map": candidate_map, "reference_track_map": reference_map, "candidate_track_diagnostics": candidate_diag, "reference_track_diagnostics": reference_diag},
        "mask": {"matching": "exact integer chromosome + position_bp; finite in candidate A/B/reference mat/pat", "imputation": False, "extrapolation": False, "old_intra_pair_mask": False, "chromosomes_total": NCHR, "total_union_positions": int(sum(row["union_positions"] for row in mask_summary)), "total_shared_positions": int(sum(row["shared_positions"] for row in mask_summary)), "total_excluded_positions": int(sum(row["excluded_positions"] for row in mask_summary)), "per_chromosome": mask_summary},
        "primary_20_chromosome_centers": {"n_centers": NCHR, "n_unordered_pairs": len(pred190), "centroid_definition": "each copy arithmetic mean over shared beads; chromosome center is equal-weight mean of two copies", "distance_units": "raw candidate input 3DG coordinate units; scaled distances are reference 3DG coordinate units; stress is dimensionless", "spearman": primary_spearman, "pearson": primary_pearson, "distance_scale_a20": distance_scale["scale"], "normalized_stress_a20": distance_scale["stress"], "formula_scale": "a20=sum(pred*ref)/sum(pred^2)", "formula_stress": "sqrt(sum((a20*pred-ref)^2)/sum(ref^2))", "pair_pvalues": False, "permutation_null": {"draws": N_PERM, "seed": PERM_SEED, "descriptive_p_ge_observed": descriptive_p, "quantiles": permutation_quantiles, "interpretation": "complete chromosome-label permutation reference; dependent distances retained; descriptive only, not biological inference"}},
        "per_chromosome_19_distance_readout": per_chromosome_rows,
        "chr1": {"n_distances": 19, "spearman_19": per_chromosome_rows[0]["distance_spearman_19"], "pearson_19": per_chromosome_rows[0]["distance_pearson_19"], "stress_19_using_primary_a20": per_chromosome_rows[0]["distance_stress_19_using_primary_a20"], "top3_candidate_nearest": per_chromosome_rows[0]["top3_candidate_nearest"], "top3_reference_nearest": per_chromosome_rows[0]["top3_reference_nearest"], "top3_overlap": per_chromosome_rows[0]["top3_overlap"]},
        "global_alignment": {"fit": "one 20-center equal-weight similarity Procrustes", "translation": True, "orthogonal_matrix": True, "positive_uniform_scale": True, "reflection_allowed": True, "determinant": free_alignment["determinant"], "fit_scale": free_alignment["scale"], "translation": free_alignment["translation"], "rotation_matrix": free_alignment["rotation_matrix"], "center_rmsd": free_alignment["rmsd"], "center_max_abs_residual": free_alignment["max_abs_residual"], "reference_center_rg": reference_rg, "center_rmsd_over_reference_rg": free_alignment["rmsd_over_reference_center_rg"], "proper_rotation_only": {"determinant": proper_alignment["determinant"], "fit_scale": proper_alignment["scale"], "center_rmsd": proper_alignment["rmsd"], "center_rmsd_over_reference_center_rg": proper_alignment["rmsd_over_reference_center_rg"]}, "same_transform_application": True, "per_chromosome_fit": False},
        "copy_supplement": {"n_copy_centers": 40, "n_cross_chromosome_pairs": len(copy_rows), "same_chromosome_copy_pairs_excluded": NCHR, "swap_policy": "whole-chromosome A/B swap selected by minimum aligned copy-center SSE to reference mat/pat", "swap_mapping": swaps, "copy_rho_after_global_alignment": correlation(copy_pred_aligned_array, copy_ref_array, "spearman"), "copy_pearson_after_global_alignment": correlation(copy_pred_aligned_array, copy_ref_array, "pearson"), "copy_primary_a20_stress_raw_distances": copy_primary_stress["stress"], "copy_primary_a20_scale_reused": distance_scale["scale"], "copy_aligned_unit_stress": copy_aligned_stress["stress"], "copy_best_scale_raw_distances": copy_best_scale["scale"], "copy_best_scale_stress_raw_distances": copy_best_scale["stress"], "copy_best_scale_is_not_primary": True, "gauge_free_pooled_sorted_blocks": gauge_free_pooled, "homolog_separation": homolog_rows},
        "status": "core_complete",
    }
    write_json(RUN / "metrics.json", metrics)
    write_json(RUN / "validation_core.json", {"core_complete": True, "candidate_hash_at_parse": candidate_hash, "reference_hash_at_parse": reference_hash, "candidate_tracks": candidate_meta["n_tracks"], "reference_tracks": reference_meta["n_tracks"], "candidate_rows": candidate_meta["n_rows"], "reference_rows": reference_meta["n_rows"], "shared_positions_total": int(sum(row["shared_positions"] for row in mask_summary)), "pairs190": len(pair_rows), "copy_pairs760": len(copy_rows), "per_chromosome_n": [row["distance_n"] for row in per_chromosome_rows], "chr1_n": per_chromosome_rows[0]["distance_n"]})
    log("core metrics: rho=%.6f pearson=%.6f a20=%.6f stress=%.6f; copy rho=%.6f" % (primary_spearman, primary_pearson, distance_scale["scale"], distance_scale["stress"], correlation(copy_pred_aligned_array, copy_ref_array, "spearman")))
    print("CORE_SUMMARY", json.dumps({"spearman": primary_spearman, "pearson": primary_pearson, "a20": distance_scale["scale"], "stress": distance_scale["stress"], "shared": int(sum(row["shared_positions"] for row in mask_summary)), "union": int(sum(row["union_positions"] for row in mask_summary)), "copy_rho": correlation(copy_pred_aligned_array, copy_ref_array, "spearman")}, sort_keys=True))


def load_plot_state() -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[int, np.ndarray]], dict[str, Any], dict[str, Any]]:
    candidate, _ = parse_3dg(CANDIDATE)
    reference, _ = parse_3dg(REFERENCE)
    state = np.load(RUN / "core_data.npz")
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    alignment = json.loads((RUN / "alignment.json").read_text(encoding="utf-8"))
    return candidate, reference, {key: state[key] for key in state.files}, metrics, alignment


def write_custom_html(path: Path, candidate_aligned_tracks: dict[str, np.ndarray], reference_tracks: dict[str, np.ndarray], candidate_map: dict[str, list[str]], reference_map: dict[str, list[str]], shared_positions: dict[str, list[int]], colors: list[Any], display_center: np.ndarray, display_extent: float, det: float, fit_scale: float, distance_scale_a20: float) -> dict[str, Any]:
    """在同一共享 reference frame 中写出无依赖的 canvas viewer。"""
    def color_hex(rgba: Any) -> str:
        return "#%02x%02x%02x" % tuple(int(round(float(value) * 255)) for value in rgba[:3])

    tracks = []
    for ci, chromosome in enumerate(EXPECTED_CHROMS):
        alpha = 1.0 if ci == 0 else 0.14
        for copy_index, track in enumerate(reference_map[chromosome]):
            points = [[position, float(xyz[0]), float(xyz[1]), float(xyz[2])] for position, xyz in zip(shared_positions[chromosome], reference_tracks[track])]
            tracks.append({"id": track, "source": "reference", "chromosome": chromosome, "copy": "mat" if copy_index == 0 else "pat", "color": color_hex(colors[ci]), "alpha": alpha, "points": points})
        for copy_index, track in enumerate(candidate_map[chromosome]):
            points = [[position, float(xyz[0]), float(xyz[1]), float(xyz[2])] for position, xyz in zip(shared_positions[chromosome], candidate_aligned_tracks[track])]
            tracks.append({"id": track, "source": "candidate", "chromosome": chromosome, "copy": "A" if copy_index == 0 else "B", "color": color_hex(colors[ci]), "alpha": alpha, "points": points})
    payload = {"schema_version": "p9016-c0-interchrom-centroids-canvas-v1", "coordinate_space": "reference_frame_after_one_global_similarity", "bin_size_bp": BIN, "shared_position_count": int(sum(len(v) for v in shared_positions.values())), "point_count_per_source": int(sum(len(v) for v in shared_positions.values()) * 2), "display_center": display_center.tolist(), "display_extent": float(display_extent), "determinant": float(det), "fit_scale": float(fit_scale), "distance_scale_a20": float(distance_scale_a20), "chromosomes": EXPECTED_CHROMS, "tracks": tracks}
    data_text = json.dumps(jsonable(payload), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html_text = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>P9016 C0 interchromosome centroids</title>
<style>
html,body{height:100%;margin:0;background:#f7f8f8;color:#152127;font:13px/1.35 Arial,sans-serif}.viewer{min-height:100%;display:grid;grid-template-columns:minmax(0,1fr) 210px}.stage{position:relative;min-height:560px;background:#fff;border-right:1px solid #cbd3d5}canvas{display:block;width:100%;height:100%;min-height:560px;touch-action:none;cursor:grab}canvas:active{cursor:grabbing}.title{position:absolute;left:14px;top:12px;font-weight:600;pointer-events:none}.meta{position:absolute;right:14px;top:12px;color:#526166;pointer-events:none;text-align:right;font-size:12px}.hover{position:absolute;max-width:280px;padding:5px 7px;border:1px solid #9aa8ac;background:#fff;box-shadow:0 1px 4px #0002;pointer-events:none;font-size:12px}.controls{padding:14px 12px;overflow:auto;background:#f7f8f8}.controls h2{font-size:12px;margin:0 0 7px;font-weight:600}.group{margin:0 0 16px;display:grid;gap:5px}.toggle{display:flex;align-items:center;gap:6px;white-space:nowrap}.toggle input{margin:0}.swatch{width:10px;height:10px;display:inline-block;border:1px solid #69777b}.button{width:100%;margin:2px 0 6px;padding:5px 7px;border:1px solid #819095;border-radius:2px;background:#fff;color:#152127;font:inherit;cursor:pointer;text-align:left}.button:hover{background:#eaf0f0}.hint{font-size:12px;color:#526166;margin-top:12px}@media(max-width:720px){.viewer{grid-template-columns:1fr}.stage{border-right:0;border-bottom:1px solid #cbd3d5;min-height:460px}canvas{min-height:460px}.controls{display:grid;grid-template-columns:1fr 1fr;gap:14px}.group{margin:0}}
</style></head><body><main class="viewer"><section class="stage"><canvas id="reconstruction-canvas" aria-label="P9016 C0 interchromosome centroid viewer"></canvas><div class="title">P9016 C0 interchromosome centroids</div><div class="meta">one global centroid similarity<br>reflection allowed; shared mask only<br>drag rotate / wheel zoom</div><div id="hover" class="hover" hidden></div></section><aside class="controls"><div class="group"><h2>Source</h2><label class="toggle"><input type="checkbox" data-source="reference" checked><span>Reference</span></label><label class="toggle"><input type="checkbox" data-source="candidate" checked><span>Globally aligned C0</span></label></div><div class="group"><h2>Copy markers</h2><label class="toggle"><input type="checkbox" data-copy="mat" checked><span>reference mat (circle)</span></label><label class="toggle"><input type="checkbox" data-copy="pat" checked><span>reference pat (triangle)</span></label><label class="toggle"><input type="checkbox" data-copy="A" checked><span>candidate A (square)</span></label><label class="toggle"><input type="checkbox" data-copy="B" checked><span>candidate B (diamond)</span></label></div><div class="group"><h2>Chromosomes</h2><button class="button" id="show-all" type="button">Show all chromosomes</button><button class="button" id="show-chr1" type="button">Show chr1 only</button><button class="button" id="reset-view" type="button">Reset view</button></div><button class="button" id="export-json" type="button">Export shared coordinates JSON</button><div class="hint">chr1 is saturated; other chromosomes use alpha 0.14. Fit scale and distance a20 are separate.</div></aside></main>
<script>
const STRUCTURE_DATA=""" + data_text + """;
(function(){
const data=STRUCTURE_DATA,canvas=document.getElementById("reconstruction-canvas"),ctx=canvas.getContext("2d"),hover=document.getElementById("hover");
const state={yaw:-0.72,pitch:0.34,zoom:1,source:{reference:true,candidate:true},copy:{mat:true,pat:true,A:true,B:true},chromosomes:{}};data.chromosomes.forEach(function(chromosome){state.chromosomes[chromosome]=true;});
const view={width:1,height:1,dpr:1};let projected=[],drag=null;
function visible(track){return state.source[track.source]&&state.copy[track.copy]&&state.chromosomes[track.chromosome];}
function normal(point){return [(point[1]-data.display_center[0])/data.display_extent,(point[2]-data.display_center[1])/data.display_extent,(point[3]-data.display_center[2])/data.display_extent];}
function rotate(point){const cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch),x=cy*point[0]+sy*point[2],z=-sy*point[0]+cy*point[2];return [x,cp*point[1]-sp*z,sp*point[1]+cp*z];}
function project(point){const q=rotate(point),perspective=4/(4+q[2]),scale=Math.min(view.width,view.height)*0.42*state.zoom*perspective;return {x:view.width/2+q[0]*scale,y:view.height/2-q[1]*scale,z:q[2]};}
function curve(points,color,alpha){if(points.length<2)return;ctx.save();ctx.strokeStyle=color;ctx.globalAlpha=Math.min(1,alpha*1.15);ctx.lineWidth=1;ctx.beginPath();points.forEach(function(raw,index){const p=project(raw);if(index===0)ctx.moveTo(p.x,p.y);else ctx.lineTo(p.x,p.y);});ctx.stroke();ctx.restore();}
function marker(item){ctx.save();ctx.fillStyle=item.track.color;ctx.globalAlpha=Math.min(1,item.track.alpha*1.05);ctx.beginPath();const r=item.track.chromosome==="chr1"?2.25:1.7;if(item.track.copy==="pat"){ctx.moveTo(item.x,item.y-r);ctx.lineTo(item.x+r,item.y+r);ctx.lineTo(item.x-r,item.y+r);}else if(item.track.copy==="A"){ctx.rect(item.x-r,item.y-r,2*r,2*r);}else if(item.track.copy==="B"){ctx.moveTo(item.x,item.y-r);ctx.lineTo(item.x+r,item.y);ctx.lineTo(item.x,item.y+r);ctx.lineTo(item.x-r,item.y);}else{ctx.arc(item.x,item.y,r,0,Math.PI*2);}ctx.closePath();ctx.fill();ctx.restore();}
function render(){ctx.setTransform(view.dpr,0,0,view.dpr,0,0);ctx.clearRect(0,0,view.width,view.height);projected=[];data.tracks.forEach(function(track){if(!visible(track))return;const raw=track.points.map(normal);track.points.forEach(function(point,index){const p=project(raw[index]);projected.push({x:p.x,y:p.y,z:p.z,track:track,raw:point});});});projected.sort(function(a,b){return b.z-a.z;});projected.forEach(marker);}
function resize(){const box=canvas.getBoundingClientRect();view.width=Math.max(1,box.width);view.height=Math.max(1,box.height);view.dpr=Math.max(1,window.devicePixelRatio||1);canvas.width=Math.round(view.width*view.dpr);canvas.height=Math.round(view.height*view.dpr);render();}
function hideHover(){hover.hidden=true;}
function pick(event){const box=canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top;let best=null,bestDistance=100;projected.forEach(function(item){const dx=item.x-x,dy=item.y-y,distance=dx*dx+dy*dy;if(distance<bestDistance){best=item;bestDistance=distance;}});if(!best){hideHover();return;}hover.textContent=best.track.source+" | "+best.track.chromosome+" | copy "+best.track.copy+" | start "+best.raw[0].toLocaleString();hover.style.left=Math.min(view.width-290,Math.max(8,x+12))+"px";hover.style.top=Math.min(view.height-36,Math.max(8,y+12))+"px";hover.hidden=false;}
canvas.addEventListener("pointerdown",function(event){drag={x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);hideHover();});canvas.addEventListener("pointermove",function(event){if(drag){state.yaw+=(event.clientX-drag.x)*0.01;state.pitch=Math.max(-1.45,Math.min(1.45,state.pitch+(event.clientY-drag.y)*0.01));drag={x:event.clientX,y:event.clientY};render();}else pick(event);});canvas.addEventListener("pointerup",function(event){drag=null;if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);});canvas.addEventListener("pointercancel",function(){drag=null;});canvas.addEventListener("pointerleave",function(){if(!drag)hideHover();});canvas.addEventListener("wheel",function(event){event.preventDefault();state.zoom=Math.max(0.35,Math.min(3,state.zoom*Math.exp(-event.deltaY*0.001)));render();},{passive:false});
document.querySelectorAll("input[data-source]").forEach(function(input){input.addEventListener("change",function(){state.source[input.dataset.source]=input.checked;hideHover();render();});});document.querySelectorAll("input[data-copy]").forEach(function(input){input.addEventListener("change",function(){state.copy[input.dataset.copy]=input.checked;hideHover();render();});});
document.getElementById("show-all").addEventListener("click",function(){data.chromosomes.forEach(function(c){state.chromosomes[c]=true;});render();});document.getElementById("show-chr1").addEventListener("click",function(){data.chromosomes.forEach(function(c){state.chromosomes[c]=c==="chr1";});render();});document.getElementById("reset-view").addEventListener("click",function(){state.yaw=-0.72;state.pitch=0.34;state.zoom=1;render();});document.getElementById("export-json").addEventListener("click",function(){const blob=new Blob([JSON.stringify(data)],{type:"application/json"}),url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download="p9016_c0_shared_aligned_coordinates.json";document.body.appendChild(link);link.click();link.remove();URL.revokeObjectURL(url);});window.addEventListener("resize",resize);resize();
})();
</script></body></html>
"""
    path.write_text(html_text, encoding="utf-8")
    return {"available": True, "path": str(path), "self_contained": True, "engine": "dependency-free-canvas-3d", "external_script_or_link_resources": [], "inline_viewer_code": True, "shared_mask_only": True}


def plot_stage() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.colors import to_rgba
    except Exception as exc:
        raise RuntimeError("matplotlib is required for static figures") from exc
    candidate, reference, state, metrics, alignment_obj = load_plot_state()
    candidate_map = metrics["input_meta"]["candidate_track_map"]
    reference_map = metrics["input_meta"]["reference_track_map"]
    free_alignment = {
        "scale": alignment_obj["fit"]["scale"],
        "translation": np.asarray(alignment_obj["fit"]["translation"], dtype=np.float64),
        "rotation_matrix": np.asarray(alignment_obj["fit"]["rotation_matrix_column_vector"], dtype=np.float64),
    }
    shared_positions: dict[str, list[int]] = {chromosome: [] for chromosome in EXPECTED_CHROMS}
    with (RUN / "coordinates" / "positions.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["shared"]):
                shared_positions[row["chromosome"]].append(int(row["position_bp"]))
    if any(len(shared_positions[chromosome]) == 0 for chromosome in EXPECTED_CHROMS):
        raise ValueError("plotting shared mask lost a chromosome")
    colors = [plt.get_cmap("tab20")(index) for index in range(NCHR)]
    alpha_by_chr = [1.0] + [0.14] * (NCHR - 1)
    candidate_aligned_tracks: dict[str, np.ndarray] = {}
    reference_tracks: dict[str, np.ndarray] = {}
    all_points: list[np.ndarray] = []
    for ci, chromosome in enumerate(EXPECTED_CHROMS):
        for track in candidate_map[chromosome]:
            raw = np.vstack([candidate[track][p] for p in shared_positions[chromosome]])
            candidate_aligned_tracks[track] = apply_similarity(raw, free_alignment)
            all_points.append(candidate_aligned_tracks[track])
        for track in reference_map[chromosome]:
            raw = np.vstack([reference[track][p] for p in shared_positions[chromosome]])
            reference_tracks[track] = raw
            all_points.append(raw)
    plotted_candidate_points = int(sum(len(points) for points in candidate_aligned_tracks.values()))
    plotted_reference_points = int(sum(len(points) for points in reference_tracks.values()))
    all_points_arr = np.vstack(all_points)
    display_center = all_points_arr.mean(axis=0)
    max_span = float(np.max(np.abs(all_points_arr - display_center))) * 1.08
    max_span = max(max_span, 1e-6)
    limits = [(float(display_center[k] - max_span), float(display_center[k] + max_span)) for k in range(3)]
    det = metrics["global_alignment"]["determinant"]
    fit_scale = metrics["global_alignment"]["fit_scale"]
    distance_scale_a20 = metrics["primary_20_chromosome_centers"]["distance_scale_a20"]
    title_suffix = "centroid-based global similarity; reflection allowed (det=%.3f; fit scale=%.4g; distance a20=%.4g; shared-mask plot=%d/%d points)" % (det, fit_scale, distance_scale_a20, plotted_candidate_points, plotted_reference_points)

    def style_3d(ax):
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-55)
        ax.tick_params(labelsize=7, pad=-2)
        ax.set_xlabel("x", fontsize=7, labelpad=-5); ax.set_ylabel("y", fontsize=7, labelpad=-5); ax.set_zlabel("z", fontsize=7, labelpad=-5)

    fig = plt.figure(figsize=(9.0, 3.9))
    axes = [fig.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    for ci, chromosome in enumerate(EXPECTED_CHROMS):
        alpha = alpha_by_chr[ci]
        for copy_index, track in enumerate(reference_map[chromosome]):
            points = reference_tracks[track]
            axes[0].scatter(points[:, 0], points[:, 1], points[:, 2], s=2.5, marker="o" if copy_index == 0 else "^", color=colors[ci], alpha=alpha, linewidths=0)
            axes[2].scatter(points[:, 0], points[:, 1], points[:, 2], s=2.5, marker="o" if copy_index == 0 else "^", color=colors[ci], alpha=alpha, linewidths=0)
        for copy_index, track in enumerate(candidate_map[chromosome]):
            points = candidate_aligned_tracks[track]
            axes[1].scatter(points[:, 0], points[:, 1], points[:, 2], s=2.5, marker="D" if copy_index == 0 else "P", color=colors[ci], alpha=alpha, linewidths=0)
            axes[2].scatter(points[:, 0], points[:, 1], points[:, 2], s=2.5, marker="D" if copy_index == 0 else "P", color=colors[ci], alpha=alpha, linewidths=0)
    for ax, label in zip(axes, ["Reference", "Globally aligned C0", "Overlay"]):
        style_3d(ax); ax.set_title(label, fontsize=7, pad=1)
    chromosome_handles = [Line2D([0], [0], marker="o", linestyle="None", color=colors[i], markersize=4, label=chromosome) for i, chromosome in enumerate(EXPECTED_CHROMS)]
    marker_handles = [Line2D([0], [0], marker="o", linestyle="None", color="0.2", markersize=4, label="reference mat"), Line2D([0], [0], marker="^", linestyle="None", color="0.2", markersize=4, label="reference pat"), Line2D([0], [0], marker="D", linestyle="None", color="0.2", markersize=4, label="candidate A"), Line2D([0], [0], marker="P", linestyle="None", color="0.2", markersize=4, label="candidate B")]
    fig.legend(handles=chromosome_handles, title="Chromosome", title_fontsize=7, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.01), frameon=False, handletextpad=0.25, columnspacing=0.55)
    fig.legend(handles=marker_handles, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.965), frameon=False, handletextpad=0.25, columnspacing=0.8)
    fig.text(0.5, 0.995, title_suffix, ha="center", va="top", fontsize=7)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.23, top=0.88, wspace=0.02)
    fig.savefig(RUN / "plots" / "main_3panel.png", dpi=300)
    fig.savefig(RUN / "plots" / "main_3panel.pdf")
    plt.close(fig)

    # 正交投影有意独立于 3D figure，以便检查遮挡。
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    projections = [(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]
    for ax, (xk, yk, label) in zip(axes, projections):
        for ci, chromosome in enumerate(EXPECTED_CHROMS):
            alpha = alpha_by_chr[ci]
            for copy_index, track in enumerate(reference_map[chromosome]):
                points = reference_tracks[track]
                ax.scatter(points[:, xk], points[:, yk], s=2, marker="o" if copy_index == 0 else "^", color=colors[ci], alpha=alpha, linewidths=0)
            for copy_index, track in enumerate(candidate_map[chromosome]):
                points = candidate_aligned_tracks[track]
                ax.scatter(points[:, xk], points[:, yk], s=2, marker="D" if copy_index == 0 else "P", color=colors[ci], alpha=alpha, linewidths=0)
        ax.set_xlim(*limits[xk]); ax.set_ylim(*limits[yk]); ax.set_aspect("equal", adjustable="box"); ax.set_title(label, fontsize=7); ax.tick_params(labelsize=7); ax.set_xlabel("xyz"[xk], fontsize=7); ax.set_ylabel("xyz"[yk], fontsize=7)
    fig.suptitle("Reference + globally aligned C0 projections; chr1 saturated, others alpha=0.14", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RUN / "plots" / "projections_xyz.png", dpi=300)
    fig.savefig(RUN / "plots" / "projections_xyz.pdf")
    plt.close(fig)

    # 仅显示中心的视图直接解释 190 pair metric。
    fig = plt.figure(figsize=(3.2, 3.5)); ax = fig.add_subplot(111, projection="3d")
    for ci, chromosome in enumerate(EXPECTED_CHROMS):
        alpha = 1.0 if ci == 0 else 0.35
        ax.plot([state["reference_centers"][ci, 0], state["candidate_chr_aligned"][ci, 0]], [state["reference_centers"][ci, 1], state["candidate_chr_aligned"][ci, 1]], [state["reference_centers"][ci, 2], state["candidate_chr_aligned"][ci, 2]], color=colors[ci], alpha=alpha, linewidth=1.3 if ci == 0 else 0.5)
        ax.scatter(*state["reference_centers"][ci], color=colors[ci], marker="o", s=28 if ci == 0 else 13, alpha=alpha)
        ax.scatter(*state["candidate_chr_aligned"][ci], color=colors[ci], marker="X", s=30 if ci == 0 else 14, alpha=alpha)
    style_3d(ax); ax.set_title("20 chromosome centers\n" + title_suffix, fontsize=7)
    ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="None", color="0.2", label="reference center"), Line2D([0], [0], marker="X", linestyle="None", color="0.2", label="C0 center")], fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RUN / "plots" / "centers_only_3d.png", dpi=300)
    fig.savefig(RUN / "plots" / "centers_only_3d.pdf")
    plt.close(fig)

    # Distance scatter 高亮涉及 chr1 的 19 对。
    pairs = list(csv.DictReader((RUN / "pairs190.tsv").open(encoding="utf-8"), delimiter="\t"))
    a20 = metrics["primary_20_chromosome_centers"]["distance_scale_a20"]
    x = a20 * np.asarray([float(row["candidate_distance_raw"]) for row in pairs]); y = np.asarray([float(row["reference_distance"]) for row in pairs]); chr1 = np.asarray([int(row["chr1_involved"]) for row in pairs], dtype=bool)
    lim = max(float(x.max()), float(y.max())) * 1.05
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.scatter(x[~chr1], y[~chr1], s=14, color="0.65", alpha=0.65, label="other 171 pairs")
    ax.scatter(x[chr1], y[chr1], s=24, color=colors[0], edgecolor="black", linewidth=0.3, label="chr1 pairs (19)")
    grid = np.linspace(0, lim, 200)
    ax.plot(grid, grid, color="black", linewidth=0.8, linestyle="--", label="1:1")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel("C0 chromosome-center distance x a20", fontsize=7); ax.set_ylabel("reference chromosome-center distance", fontsize=7)
    ax.set_aspect("equal", adjustable="box"); ax.tick_params(labelsize=7); ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.set_title("190 chromosome-center pairs\nSpearman rho=%.3f; stress=%.3f" % (metrics["primary_20_chromosome_centers"]["spearman"], metrics["primary_20_chromosome_centers"]["normalized_stress_a20"]), fontsize=7)
    fig.tight_layout(); fig.savefig(RUN / "plots" / "center_distance_scatter.png", dpi=300); fig.savefig(RUN / "plots" / "center_distance_scatter.pdf"); plt.close(fig)

    # Distance maps 使用相同的 primary 20-center distance scale，而不是 fit scale。
    ref_map = state["reference_distance_matrix"]; pred_map_scaled = state["candidate_distance_matrix_scaled"]
    vmax = max(float(ref_map.max()), float(pred_map_scaled.max()))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), constrained_layout=True)
    im0 = axes[0].imshow(ref_map, cmap="coolwarm_r", vmin=0, vmax=vmax, interpolation="nearest"); axes[0].set_title("Reference center distances", fontsize=7)
    im1 = axes[1].imshow(pred_map_scaled, cmap="coolwarm_r", vmin=0, vmax=vmax, interpolation="nearest"); axes[1].set_title("C0 distances scaled by a20", fontsize=7)
    tick_labels = [c.replace("chr", "") for c in EXPECTED_CHROMS]
    for ax in axes:
        ax.set_xticks(range(NCHR)); ax.set_yticks(range(NCHR)); ax.set_xticklabels(tick_labels, rotation=90, fontsize=7); ax.set_yticklabels(tick_labels, fontsize=7); ax.set_xlabel("chromosome", fontsize=7); ax.set_ylabel("chromosome", fontsize=7)
    cbar = fig.colorbar(im1, ax=axes, shrink=0.75, label="distance", pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("distance", fontsize=7)
    fig.savefig(RUN / "plots" / "center_distance_maps.png", dpi=300); fig.savefig(RUN / "plots" / "center_distance_maps.pdf"); plt.close(fig)

    # chr1 的 19 个 distance 按 numeric chromosome order 排列。
    chr1_pairs = [row for row in pairs if row["chromosome_i"] == "chr1" or row["chromosome_j"] == "chr1"]
    chr1_pairs.sort(key=lambda row: EXPECTED_CHROMS.index(row["chromosome_j"] if row["chromosome_i"] == "chr1" else row["chromosome_i"]))
    labels = [row["chromosome_j"] if row["chromosome_i"] == "chr1" else row["chromosome_i"] for row in chr1_pairs]
    ref_y = np.asarray([float(row["reference_distance"]) for row in chr1_pairs]); pred_y = np.asarray([float(row["candidate_distance_scaled_by_primary_a"]) for row in chr1_pairs])
    xx = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 2.8)); ax.plot(xx, ref_y, "o-", color="black", linewidth=0.9, markersize=3.5, label="reference"); ax.plot(xx, pred_y, "D-", color=colors[0], linewidth=0.8, markersize=3.2, label="C0 x a20"); ax.set_xticks(xx); ax.set_xticklabels([x.replace("chr", "") for x in labels], fontsize=7); ax.set_xlabel("other chromosome (numeric order)", fontsize=7); ax.set_ylabel("distance", fontsize=7); ax.tick_params(axis="y", labelsize=7); ax.legend(fontsize=7, frameon=False); ax.set_title("chr1-to-other chromosome-center distances (19)", fontsize=7); fig.tight_layout(); fig.savefig(RUN / "plots" / "chr1_distances.png", dpi=300); fig.savefig(RUN / "plots" / "chr1_distances.pdf"); plt.close(fig)

    html_path = RUN / "plots" / "interchrom_centroids.html"
    html_status = write_custom_html(html_path, candidate_aligned_tracks, reference_tracks, candidate_map, reference_map, shared_positions, colors, display_center, max_span, det, fit_scale, distance_scale_a20)

    # 写出每个 plot 后验证 dimensions 和 geometry invariants。
    def png_size(path: Path) -> list[int] | None:
        try:
            with path.open("rb") as handle:
                signature = handle.read(24)
            if signature[:8] != b"\x89PNG\r\n\x1a\n":
                return None
            return [struct.unpack(">II", signature[16:24])[0], struct.unpack(">II", signature[16:24])[1]]
        except Exception:
            return None

    plot_files = sorted(str(path.relative_to(RUN)) for path in (RUN / "plots").iterdir() if path.is_file())
    plot_checks: dict[str, dict[str, Any]] = {}
    for name in plot_files:
        path = RUN / name
        if path.suffix == ".png":
            dimensions = png_size(path)
            plot_checks[name] = {"exists": True, "bytes": path.stat().st_size, "png_dimensions": dimensions, "valid_png_signature": dimensions is not None}
        elif path.suffix == ".pdf":
            plot_checks[name] = {"exists": True, "bytes": path.stat().st_size, "pdf_nonempty": path.stat().st_size > 1000}
        elif path.suffix == ".html":
            text = path.read_text(encoding="utf-8", errors="replace")
            resource_refs = re.findall(r"<(?:script|link)\b[^>]*(?:src|href)\s*=\s*[\"']([^\"']+)", text, flags=re.IGNORECASE)
            external_resources = [ref for ref in resource_refs if re.match(r"^(?:https?:)?//", ref)]
            inline_canvas_viewer = "STRUCTURE_DATA" in text and "reconstruction-canvas" in text and "function render" in text
            inline_bundle = ("Plotly.newPlot" in text and "window.Plotly" in text and len(text) > 100000) or inline_canvas_viewer
            plot_checks[name] = {"exists": True, "bytes": path.stat().st_size, "html_has_plotly": "plotly" in text.lower(), "external_script_or_link_resources": external_resources, "inline_plotly_bundle": inline_bundle, "inline_canvas_viewer": inline_canvas_viewer, "html_self_contained": bool(not external_resources and inline_bundle)}

    position_rows_actual = list(csv.DictReader((RUN / "coordinates" / "positions.tsv").open(encoding="utf-8"), delimiter="\t"))
    pair_rows_actual = list(csv.DictReader((RUN / "pairs190.tsv").open(encoding="utf-8"), delimiter="\t"))
    copy_rows_actual = list(csv.DictReader((RUN / "copy_pairs760.tsv").open(encoding="utf-8"), delimiter="\t"))
    per_rows_actual = list(csv.DictReader((RUN / "per_chromosome.tsv").open(encoding="utf-8"), delimiter="\t"))
    n_centers_actual = int(state["candidate_centers"].shape[0])
    n_copy_centers_actual = int(np.prod(state["candidate_copy_centers"].shape[:2]))
    expected_pairs_actual = n_centers_actual * (n_centers_actual - 1) // 2
    expected_copy_pairs_actual = expected_pairs_actual * 4
    shared_position_count_actual = int(sum(int(row["shared"]) for row in position_rows_actual))
    per_chromosome_counts_actual = [int(row["distance_n"]) for row in per_rows_actual]

    def choose_swaps(copy_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        choices = []
        direct_costs = []
        swapped_costs = []
        chosen_costs = []
        for ci in range(copy_centers.shape[0]):
            direct = float(np.sum((copy_centers[ci, 0] - state["reference_copy_centers"][ci, 0]) ** 2) + np.sum((copy_centers[ci, 1] - state["reference_copy_centers"][ci, 1]) ** 2))
            swapped = float(np.sum((copy_centers[ci, 0] - state["reference_copy_centers"][ci, 1]) ** 2) + np.sum((copy_centers[ci, 1] - state["reference_copy_centers"][ci, 0]) ** 2))
            choice = int(swapped < direct)
            choices.append(choice); direct_costs.append(direct); swapped_costs.append(swapped); chosen_costs.append(min(direct, swapped))
        return np.asarray(choices, dtype=np.int8), np.asarray(direct_costs), np.asarray(swapped_costs), np.asarray(chosen_costs)

    def copy_vectors(copy_centers_aligned: np.ndarray, copy_centers_raw: np.ndarray, choices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        aligned_values = []; raw_values = []; reference_values = []
        for i in range(copy_centers_aligned.shape[0]):
            for j in range(i + 1, copy_centers_aligned.shape[0]):
                for copy_i in (0, 1):
                    for copy_j in (0, 1):
                        ref_i = copy_i if choices[i] == 0 else 1 - copy_i
                        ref_j = copy_j if choices[j] == 0 else 1 - copy_j
                        aligned_values.append(np.linalg.norm(copy_centers_aligned[i, copy_i] - copy_centers_aligned[j, copy_j]))
                        raw_values.append(np.linalg.norm(copy_centers_raw[i, copy_i] - copy_centers_raw[j, copy_j]))
                        reference_values.append(np.linalg.norm(state["reference_copy_centers"][i, ref_i] - state["reference_copy_centers"][j, ref_j]))
        return np.asarray(aligned_values), np.asarray(raw_values), np.asarray(reference_values)

    original_choices, original_direct, original_swapped, original_chosen = choose_swaps(state["candidate_copy_aligned"])
    relabeled_aligned_centers = state["candidate_copy_aligned"][:, ::-1, :]
    relabeled_raw_centers = state["candidate_copy_centers"][:, ::-1, :]
    relabeled_choices, relabeled_direct, relabeled_swapped, relabeled_chosen = choose_swaps(relabeled_aligned_centers)
    original_aligned_vector, original_raw_vector, original_reference_vector = copy_vectors(state["candidate_copy_aligned"], state["candidate_copy_centers"], original_choices)
    relabeled_aligned_vector, relabeled_raw_vector, relabeled_reference_vector = copy_vectors(relabeled_aligned_centers, relabeled_raw_centers, relabeled_choices)
    swap_ties = np.isclose(original_direct, original_swapped, rtol=0.0, atol=1e-15)
    mapping_complement_check = bool(np.all(relabeled_choices[~swap_ties] == (1 - original_choices[~swap_ties])))
    chosen_sse_diff = float(np.max(np.abs(original_chosen - relabeled_chosen)))
    rho_original = correlation(original_aligned_vector, original_reference_vector, "spearman")
    rho_relabeled = correlation(relabeled_aligned_vector, relabeled_reference_vector, "spearman")
    stress_original = scale_and_stress(original_raw_vector, original_reference_vector, metrics["primary_20_chromosome_centers"]["distance_scale_a20"])["stress"]
    stress_relabeled = scale_and_stress(relabeled_raw_vector, relabeled_reference_vector, metrics["primary_20_chromosome_centers"]["distance_scale_a20"])["stress"]
    tsv_aligned_vector = np.asarray([float(row["candidate_distance_after_global_similarity"]) for row in copy_rows_actual])
    tsv_reference_vector = np.asarray([float(row["reference_distance"]) for row in copy_rows_actual])
    tsv_copy_geometry_check = bool(len(copy_rows_actual) == len(original_aligned_vector) and np.max(np.abs(tsv_aligned_vector - original_aligned_vector)) <= 1e-10 and np.max(np.abs(tsv_reference_vector - original_reference_vector)) <= 1e-10)

    input_hashes_rechecked = {"candidate": sha256(CANDIDATE), "reference": sha256(REFERENCE)}
    input_hashes_pass = bool(input_hashes_rechecked["candidate"] == metrics["input_meta"]["candidate"]["sha256"] and input_hashes_rechecked["reference"] == metrics["input_meta"]["reference"]["sha256"])
    matrix_checks = {
        "candidate_symmetric": bool(np.allclose(state["candidate_distance_matrix"], state["candidate_distance_matrix"].T, atol=1e-12)),
        "reference_symmetric": bool(np.allclose(state["reference_distance_matrix"], state["reference_distance_matrix"].T, atol=1e-12)),
        "candidate_zero_diagonal": bool(np.allclose(np.diag(state["candidate_distance_matrix"]), 0.0, atol=1e-12)),
        "reference_zero_diagonal": bool(np.allclose(np.diag(state["reference_distance_matrix"]), 0.0, atol=1e-12)),
        "all_finite": bool(np.isfinite(state["candidate_distance_matrix"]).all() and np.isfinite(state["reference_distance_matrix"]).all()),
    }
    transform_checks = {
        "orthogonal_matrix": bool(np.allclose(np.asarray(alignment_obj["fit"]["rotation_matrix_column_vector"]).T @ np.asarray(alignment_obj["fit"]["rotation_matrix_column_vector"]), np.eye(3), atol=1e-10)),
        "positive_uniform_scale": bool(float(fit_scale) > 0),
        "determinant_abs_one": bool(abs(abs(float(det)) - 1.0) <= 1e-10),
        "single_transform_applied": True,
        "distance_invariance_max_abs_error": float(np.max(np.abs(distance_matrix(state["candidate_chr_aligned"]) - fit_scale * state["candidate_distance_matrix"]))),
        "copy_distance_invariance_max_abs_error": float(np.max(np.abs(distance_matrix(state["candidate_copy_aligned"].reshape(-1, 3)) - fit_scale * distance_matrix(state["candidate_copy_centers"].reshape(-1, 3))))),
    }
    required_checks = {
        "candidate_track_count_40": metrics["input_meta"]["candidate"]["n_tracks"] == 40,
        "reference_track_count_40": metrics["input_meta"]["reference"]["n_tracks"] == 40,
        "candidate_rows_finite": metrics["input_meta"]["candidate"]["nonfinite_rows"] == 0,
        "reference_rows_finite": metrics["input_meta"]["reference"]["nonfinite_rows"] == 0,
        "chromosome_center_count": n_centers_actual == len(EXPECTED_CHROMS),
        "copy_center_count": n_copy_centers_actual == 2 * n_centers_actual,
        "all_chromosomes_retained": set(row["chromosome"] for row in per_rows_actual) == set(EXPECTED_CHROMS),
        "shared_position_positive_each_chromosome": all(int(row["shared_positions"]) > 0 for row in per_rows_actual),
        "pairs190_actual_and_expected": len(pair_rows_actual) == expected_pairs_actual,
        "copy_pairs760_actual_and_expected": len(copy_rows_actual) == expected_copy_pairs_actual,
        "per_chromosome_n_actual": all(value == n_centers_actual - 1 for value in per_chromosome_counts_actual),
        "chr1_n_actual": next((int(row["distance_n"]) for row in per_rows_actual if row["chromosome"] == "chr1"), -1) == n_centers_actual - 1,
        "position_rows_shared_count_consistent": shared_position_count_actual == metrics["mask"]["total_shared_positions"],
        "matrix_checks": all(matrix_checks.values()),
        "transform_checks": all([transform_checks["orthogonal_matrix"], transform_checks["positive_uniform_scale"], transform_checks["determinant_abs_one"], transform_checks["distance_invariance_max_abs_error"] <= 1e-10, transform_checks["copy_distance_invariance_max_abs_error"] <= 1e-10]),
        "copy_tsv_geometry_recomputed": tsv_copy_geometry_check,
        "swap_mapping_complement_after_all_chr_relabel": mapping_complement_check,
        "swap_chosen_sse_invariant": chosen_sse_diff <= 1e-12,
        "swap_copy_rho_invariant": abs(rho_original - rho_relabeled) <= 1e-12,
        "swap_copy_primary_stress_invariant": abs(stress_original - stress_relabeled) <= 1e-12,
        "input_hashes_unchanged": input_hashes_pass,
        "main_png_valid": bool(plot_checks.get("plots/main_3panel.png", {}).get("valid_png_signature", False)),
        "main_pdf_nonempty": bool(plot_checks.get("plots/main_3panel.pdf", {}).get("pdf_nonempty", False)),
        "plot_shared_mask_point_count": plotted_candidate_points == 2 * shared_position_count_actual and plotted_reference_points == 2 * shared_position_count_actual,
        "html_exists_and_self_contained": bool(html_status.get("available") and html_status.get("path") and Path(html_status["path"]).exists() and html_status.get("self_contained")),
        "all_png_dimensions_valid": all(item.get("valid_png_signature", False) and item.get("png_dimensions") for item in plot_checks.values() if "png_dimensions" in item),
        "all_pdf_nonempty": all(item.get("pdf_nonempty", False) for item in plot_checks.values() if "pdf_nonempty" in item),
        "required_static_plot_files": all(plot_checks.get(name, {}).get("exists", False) for name in ["plots/main_3panel.png", "plots/main_3panel.pdf", "plots/center_distance_scatter.png", "plots/center_distance_scatter.pdf", "plots/center_distance_maps.png", "plots/center_distance_maps.pdf", "plots/chr1_distances.png", "plots/chr1_distances.pdf"]),
    }
    validation = {
        "schema": "p9016-c0-interchrom-centroids-validation-v2",
        "actual_counts": {"candidate_tracks": metrics["input_meta"]["candidate"]["n_tracks"], "reference_tracks": metrics["input_meta"]["reference"]["n_tracks"], "candidate_rows": metrics["input_meta"]["candidate"]["n_rows"], "reference_rows": metrics["input_meta"]["reference"]["n_rows"], "chromosome_centers": n_centers_actual, "copy_centers": n_copy_centers_actual, "pairs190_rows": len(pair_rows_actual), "expected_pairs_from_centers": expected_pairs_actual, "copy_pairs760_rows": len(copy_rows_actual), "expected_copy_pairs_from_centers": expected_copy_pairs_actual, "same_chromosome_copy_pairs_excluded": n_centers_actual, "per_chromosome_distance_counts": per_chromosome_counts_actual, "chr1_distance_count": next((int(row["distance_n"]) for row in per_rows_actual if row["chromosome"] == "chr1"), None), "shared_position_rows": shared_position_count_actual, "plot_candidate_points": plotted_candidate_points, "plot_reference_points": plotted_reference_points},
        "mask": {"all_chromosomes_retained": required_checks["all_chromosomes_retained"], "shared_positions_positive_each_chromosome": required_checks["shared_position_positive_each_chromosome"], "total_union_positions": metrics["mask"]["total_union_positions"], "total_shared_positions": metrics["mask"]["total_shared_positions"], "total_excluded_positions": metrics["mask"]["total_excluded_positions"], "candidate_plot_points_expected": 2 * shared_position_count_actual, "reference_plot_points_expected": 2 * shared_position_count_actual, "plot_uses_shared_mask_only": plotted_candidate_points == 2 * shared_position_count_actual and plotted_reference_points == 2 * shared_position_count_actual, "candidate_grid_diagnostics": metrics["input_meta"]["candidate_track_diagnostics"], "reference_grid_diagnostics": metrics["input_meta"]["reference_track_diagnostics"]},
        "distance_matrices": matrix_checks,
        "similarity_transform": {**transform_checks, "determinant": det, "fit_scale": fit_scale, "proper_rotation_sensitivity_rmsd": metrics["global_alignment"]["proper_rotation_only"]["center_rmsd"]},
        "swap_gauge_invariance": {"policy": metrics["copy_supplement"]["swap_policy"], "relabelled_all_chromosomes": True, "original_swap_choices": original_choices.tolist(), "relabelled_swap_choices": relabeled_choices.tolist(), "tie_count": int(np.count_nonzero(swap_ties)), "mapping_complement_check": mapping_complement_check, "chosen_sse_max_abs_difference": chosen_sse_diff, "copy_rho_original": rho_original, "copy_rho_relabelled": rho_relabeled, "copy_rho_abs_difference": abs(rho_original - rho_relabeled), "copy_primary_stress_original": stress_original, "copy_primary_stress_relabelled": stress_relabeled, "copy_primary_stress_abs_difference": abs(stress_original - stress_relabeled), "tsv_geometry_recomputed": tsv_copy_geometry_check},
        "plots": {"files": plot_checks, "html": html_status},
        "input_hashes_rechecked_after_analysis": {**input_hashes_rechecked, "candidate_unchanged": input_hashes_pass, "reference_unchanged": input_hashes_pass},
        "required_checks": required_checks,
        "warnings": ["reference has missing/irregular positions within some observed track envelopes; all excluded positions are explicitly listed in coordinates/positions.tsv; no interpolation or extrapolation was used", "rho and Pearson on 190/760 distances are descriptive structure-level summaries; no independent-pair p-value was computed", "copy A/B swaps are geometry gauge only, not parental assignment", "a20 distance scale and global Procrustes fit scale are distinct quantities", "sorted four-distance gauge-free pooled comparison is structurally rank-correlated by construction and cannot prove consistent copy matching"],
        "overall_pass": bool(all(required_checks.values())),
    }
    write_json(RUN / "validation.json", validation)
    if not validation["overall_pass"]:
        failed = [key for key, value in required_checks.items() if not value]
        raise RuntimeError("validation failed: %s" % ", ".join(failed))

    # 加入 plot status 到 metrics，不改变数值结果。
    metrics["plots"] = {"files": plot_files, "html": html_status, "main": {"png": "plots/main_3panel.png", "pdf": "plots/main_3panel.pdf", "default_chr1_alpha": 1.0, "other_chromosome_alpha": 0.14, "common_limits": limits, "common_camera": {"elev": 20, "azim": -55}}}
    metrics["status"] = "evaluation_complete"
    write_json(RUN / "metrics.json", metrics)

    primary = metrics["primary_20_chromosome_centers"]
    chr1 = metrics["chr1"]
    copy_metric = metrics["copy_supplement"]
    readme = """# 036 C0 染色体间中心评价\n\n状态：`evaluation_complete`。本目录是独立的中心度量，不复用旧的 intra pair mask，也不改变 036 endpoint、reference 或训练代码。\n\n## 结论\n\n在冻结的 P9016 单细胞、20 个染色体中心、190 个无序染色体间 pair 上，C0 `random_joint` 与参考结构的中心距离 Spearman rho 为 **%.6f**，Pearson 为 **%.6f**。用全 190 对统一距离尺度 `a20=%.6f` 拟合后，归一化 stress 为 **%.6f**。这只是几何相似性读出，不应由 exit code 或单个 rho 直接解释为“结构保持”。9999 次完整染色体标签置换的描述性 null 分位数为 2.5%%/50%%/97.5%% = %.6f / %.6f / %.6f，描述性 p=%.6f；距离对相互依赖，未做普通独立 pair p-value。\n\n每条染色体使用 19 个到其他中心的距离。chr1 单列 rho=**%.6f**、Pearson=**%.6f**、同一 `a20` 的 stress=**%.6f**；candidate/reference 前三近邻重叠 **%d/3**，列表见 `per_chromosome.tsv`。\n\n一次以 20 个合并中心等权拟合的全局相似度 Procrustes（平移 + 正交矩阵 + 正的统一尺度，允许反射）实际 det=**%.6f**、拟合尺度=**%.6f**，中心 RMSD=**%.6f**，除以参考中心 Rg 后为 **%.6f**。同一变换应用于全部 candidate beads 和 40 个 copy 中心；仅允许正旋转的敏感性 RMSD=**%.6f**。这里的 fit scale 与上面的 `a20` 距离 stress scale 不同。\n\n40-copy 补充在固定全局变换后逐染色体选择整条染色体 A/B swap（几何规范，非亲本恢复）。跨染色体 760 个 copy-centroid 距离的对齐 rho=**%.6f**、Pearson=**%.6f**；复用 primary `a20` 对原始 copy 距离的 stress=**%.6f**，对齐单位 stress=**%.6f**。对全部 760 个 copy 距离共同拟合的单一尺度为 %.6f、stress=%.6f，仅作附加诊断（不是逐 copy 分别缩放）。190 个染色体 pair 的四距离各自排序后再拼接为 760 值的规范自由 pooled Spearman=**%.6f**、Pearson=**%.6f**、同一 `a20` 的 stress=**%.6f**；排序会带来结构性 rank correlation，不能证明一致的拷贝匹配。结果见 `copy_gauge_free190.tsv`。\n\n## 方法与 mask\n\n- 身份固定为单一 P9016 细胞，生物重复 n=1；candidate 为 `C0-random_joint_base2207`，选择准则为 `count_nll_per_record`。036 1 Mb 阶段 `not_converged / budget_exhausted`（固定 `nit=240`）作为训练终止证据保留。\n- candidate 轨道由文件实际映射为 `c01a/c01b` ... `c20a/c20b`，reference 为 `chrN(mat)/chrN(pat)`；顺序为数值 chr1..19,X。\n- 每 chr 的共有位点是 candidate A/B 和 reference mat/pat 四轨中精确整数基因组坐标的有限 1 Mb 交集；不插补、不外推、不丢染色体。每 copy 中心是共有 beads 算术均值，chr 中心是两 copy 等权均值。总共有位点 %d，并集位点 %d，排除 %d；完整原因与每个位置见 `coordinates/positions.tsv`，汇总见 `per_chromosome.tsv`。\n- 主公式：`a20=sum(pred*ref)/sum(pred^2)`；`stress=sqrt(sum((a20*pred-ref)^2)/sum(ref^2))`。缩放后的距离使用 reference 3DG coordinate units，stress 为无量纲。\n\n## 图件与导航\n\n- `plots/main_3panel.png` / `.pdf`：Reference、globally aligned C0、overlay 三 panel；固定 chromosome colors，chr1 alpha=1，其余 alpha=0.14；两 copy 使用不同 marker；同 limits/camera。标题明确 centroid-based global fit、reflection 与 det。主图只绘制共有 mask，每结构 4894 个点；full-track aligned 坐标仍保存在 coordinates 文件中。\n- `plots/interchrom_centroids.html`：自包含、无外部依赖的 canvas 3D HTML，默认显示所有共有 mask 染色体并突出 chr1，可切换 chr1-only、source/copy markers，支持旋转/缩放。\n- `plots/center_distance_scatter.png` / `.pdf`：190 对，chr1 的 19 点高亮；横轴为 candidate distance × `a20`，仅含 1:1 线，标题给出 rho/stress。\n- `plots/center_distance_maps.png` / `.pdf`：reference 与 candidate×`a20` 的 20x20 maps，色标 `coolwarm_r`。\n- `plots/chr1_distances.png` / `.pdf`：chr1 到其他染色体的 numeric-order 对照。`plots/projections_xyz.*` 和 `plots/centers_only_3d.*` 为遮挡检查与中心解释补充。\n- `coordinates/aligned_coordinates.tsv` 同时保留 source/track/chromosome/copy/position、原始坐标与 aligned 坐标；另有 `candidate_aligned.3dg` 与 `reference.3dg`。\n\n## 限制\n\nreference 某些轨道在其 observed envelope 内存在缺位，已逐位计入 exclusion；因此本结果只声称共有 finite beads 上的中心几何关系。candidate/reference 的 190/760 距离来自同一单细胞的 linked measurements，不是 190 或 760 个独立生物重复。copy swap 是评价坐标 gauge，不能命名 maternal/paternal。高相关、低 stress 或全局 det 都不等于 chromosome-wide allele identity recovery。\n\n关键文件：`config.json`、`input_hashes.json`、`metrics.json`、`alignment.json`、`validation.json`、`terminal_evidence.json`、`pairs190.tsv`、`copy_pairs760.tsv`、`per_chromosome.tsv`、`centers.tsv`、`logs/`、`centroid_analysis.py`。\n""" % (primary["spearman"], primary["pearson"], primary["distance_scale_a20"], primary["normalized_stress_a20"], primary["permutation_null"]["quantiles"]["q025"], primary["permutation_null"]["quantiles"]["q50"], primary["permutation_null"]["quantiles"]["q975"], primary["permutation_null"]["descriptive_p_ge_observed"], chr1["spearman_19"], chr1["pearson_19"], chr1["stress_19_using_primary_a20"], chr1["top3_overlap"], metrics["global_alignment"]["determinant"], metrics["global_alignment"]["fit_scale"], metrics["global_alignment"]["center_rmsd"], metrics["global_alignment"]["center_rmsd_over_reference_rg"], metrics["global_alignment"]["proper_rotation_only"]["center_rmsd"], copy_metric["copy_rho_after_global_alignment"], copy_metric["copy_pearson_after_global_alignment"], copy_metric["copy_primary_a20_stress_raw_distances"], copy_metric["copy_aligned_unit_stress"], copy_metric["copy_best_scale_raw_distances"], copy_metric["copy_best_scale_stress_raw_distances"], copy_metric["gauge_free_pooled_sorted_blocks"]["spearman"], copy_metric["gauge_free_pooled_sorted_blocks"]["pearson"], copy_metric["gauge_free_pooled_sorted_blocks"]["primary_a20_stress"], metrics["mask"]["total_shared_positions"], metrics["mask"]["total_union_positions"], metrics["mask"]["total_excluded_positions"])
    (RUN / "README.md").write_text(readme, encoding="utf-8")
    log("plot and validation stages complete; plots=%d html=%s" % (len(plot_files), html_status["available"]))
    print("PLOT_SUMMARY", json.dumps({"main_png": str(RUN / "plots" / "main_3panel.png"), "html": html_status, "validation_overall_pass": validation["overall_pass"]}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["core", "plots", "all"], default="all")
    args = parser.parse_args()
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    if args.stage in ("core", "all"):
        core_stage()
    if args.stage in ("plots", "all"):
        plot_stage()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log("FAILED: %s: %s" % (type(exc).__name__, exc))
        raise
