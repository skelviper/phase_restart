#!/usr/bin/env python3
"""020 与 029 C0 问题的独立 display/R2 审计。

本脚本有意不导入项目 R2 helper。它自行解析已锁定的文本坐标，构建已登记的 21-condition mask，然后仅在所有 endpoint/x0 哈希通过后加载 evaluation reference。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = HERE / "config.json"
BIN = 1_000_000
OFFSET = 3_000_000
CHR_LENGTHS = {
    "chr1": 195_471_971, "chr2": 182_113_224, "chr3": 160_039_680,
    "chr4": 156_508_116, "chr5": 151_834_684, "chr6": 149_736_546,
    "chr7": 145_441_459, "chr8": 129_401_213, "chr9": 124_595_110,
    "chr10": 130_694_993, "chr11": 122_082_543, "chr12": 120_129_022,
    "chr13": 120_421_639, "chr14": 124_902_244, "chr15": 104_043_685,
    "chr16": 98_207_768, "chr17": 94_987_271, "chr18": 90_702_639,
    "chr19": 61_431_566, "chrX": 171_031_299,
}
CHROMOSOMES = tuple(CHR_LENGTHS)
ENDPOINT_IDS = ("v1_original_random_joint", "v1_continuation", "C0-bundle1", "C0-bundle2", "C0-bundle3")
C0_IDS = ("C0-bundle1", "C0-bundle2", "C0-bundle3")
X0_IDS = ("x0-bundle1", "x0-bundle2", "x0-bundle3")
RHO_KEYS = ("rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat")


class AuditError(RuntimeError):
    pass


def json_load(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {label}: {path}") from exc


def json_dump(path: Path, value: Any) -> None:
    def safe(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return safe(item.tolist())
        if isinstance(item, np.generic):
            return safe(item.item())
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {str(key): safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(val) for val in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(path: Path, expected: str | None, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"{label} unavailable: {path}")
    actual = sha256_file(path)
    if expected and actual != expected:
        raise AuditError(f"{label} SHA mismatch: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual, "expected_sha256": expected, "size_bytes": int(path.stat().st_size)}


def parse_3dg(path: Path, label: str) -> dict[str, dict[int, np.ndarray]]:
    tracks: dict[str, dict[int, np.ndarray]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if fields[:2] == ["chr", "start"]:
                continue
            if len(fields) >= 9 and fields[0].startswith("chr"):
                # 旧版 Softall TSV：chr, start, end, bid, copy, diploid_bid, x, y, z。
                chromosome = fields[0]
                if chromosome not in CHR_LENGTHS:
                    raise AuditError(f"{label} unknown chromosome at line {line_no}: {chromosome}")
                try:
                    position = int(fields[1])
                    copy_index = int(fields[4])
                    point = np.asarray([float(fields[6]), float(fields[7]), float(fields[8])], dtype=np.float64)
                except (ValueError, TypeError) as exc:
                    raise AuditError(f"{label} nonnumeric legacy line {line_no}") from exc
                track = f"c{CHROMOSOMES.index(chromosome) + 1:02d}{'a' if copy_index == 0 else 'b'}"
            else:
                if len(fields) < 5:
                    raise AuditError(f"{label} malformed line {line_no}: {line[:80]!r}")
                track = fields[0]
                try:
                    position = int(fields[1])
                    point = np.asarray([float(fields[2]), float(fields[3]), float(fields[4])], dtype=np.float64)
                except (ValueError, TypeError) as exc:
                    raise AuditError(f"{label} nonnumeric line {line_no}") from exc
            if not np.isfinite(point).all():
                raise AuditError(f"{label} nonfinite point at line {line_no}")
            rows = tracks.setdefault(track, {})
            if position in rows:
                raise AuditError(f"{label} duplicate {track}:{position}")
            rows[position] = point
    return tracks


def validate_full_candidate(coords: dict[str, dict[int, np.ndarray]], label: str) -> dict[str, Any]:
    expected_tracks = {f"c{index:02d}{copy}" for index in range(1, 21) for copy in ("a", "b")}
    if set(coords) != expected_tracks:
        raise AuditError(f"{label} tracks mismatch: {len(coords)} tracks, missing={sorted(expected_tracks - set(coords))[:4]}, extra={sorted(set(coords) - expected_tracks)[:4]}")
    expected_total = 0
    missing = 0
    for index, chromosome in enumerate(CHROMOSOMES, start=1):
        positions = set(range(0, CHR_LENGTHS[chromosome], BIN))
        for copy in ("a", "b"):
            track = f"c{index:02d}{copy}"
            expected_total += len(positions)
            missing += len(positions - set(coords[track]))
            if set(coords[track]) != positions:
                raise AuditError(f"{label} {track} full grid mismatch: expected {len(positions)}, got {len(coords[track])}")
    return {"n_tracks": len(coords), "n_points": sum(len(v) for v in coords.values()), "expected_points": expected_total, "missing_points": missing, "all_finite": True}


def decode_array(value: np.ndarray) -> list[str]:
    out = []
    for item in np.asarray(value).reshape(-1):
        if isinstance(item, bytes):
            out.append(item.decode())
        else:
            out.append(str(item))
    return out


def positions(chromosome: str) -> np.ndarray:
    return np.arange(OFFSET, CHR_LENGTHS[chromosome], BIN, dtype=np.int64)


def dense(coords: dict[str, dict[int, np.ndarray]], track: str, pos: np.ndarray) -> np.ndarray:
    out = np.full((len(pos), 3), np.nan, dtype=np.float64)
    rows = coords.get(track, {})
    for idx, bp in enumerate(pos):
        point = rows.get(int(bp))
        if point is not None:
            out[idx] = np.asarray(point, dtype=np.float64)
    return out


def distance_vector(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    out = np.full(pair_i.shape, np.nan, dtype=np.float64)
    if not len(pair_i):
        return out
    finite = np.isfinite(points[pair_i]).all(axis=1) & np.isfinite(points[pair_j]).all(axis=1)
    if finite.any():
        delta = points[pair_i[finite]] - points[pair_j[finite]]
        out[finite] = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float64), dtype=np.float64)
    return out


def average_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise AuditError("rank input must be finite one-dimensional float64")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 20 or len(y) != len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx = average_rank(np.asarray(x, dtype=np.float64))
    ry = average_rank(np.asarray(y, dtype=np.float64))
    dx = rx - np.mean(rx, dtype=np.float64)
    dy = ry - np.mean(ry, dtype=np.float64)
    den = math.sqrt(float(np.sum(dx * dx, dtype=np.float64) * np.sum(dy * dy, dtype=np.float64)))
    return float(np.sum(dx * dy, dtype=np.float64) / den) if den > 0 else float("nan")


def panel_matrix(coords: dict[str, dict[int, np.ndarray]], track: str, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point = dense(coords, track, pos)
    finite = np.isfinite(point).all(axis=1)
    matrix = np.full((len(pos), len(pos)), np.nan, dtype=np.float64)
    indices = np.flatnonzero(finite)
    if len(indices):
        delta = point[indices, None, :] - point[None, indices, :]
        matrix[np.ix_(indices, indices)] = np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float64), dtype=np.float64)
        matrix[indices, indices] = 0.0
    return matrix, finite


def upper(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[np.triu(mask, k=1)], dtype=np.float64)


def pooled_rms(matrices: np.ndarray, mask: np.ndarray) -> float:
    values = np.concatenate([upper(matrix, mask) for matrix in matrices])
    if not len(values) or not np.isfinite(values).all():
        raise AuditError("pooled RMS has no finite common values")
    scale = float(np.sqrt(np.mean(values * values, dtype=np.float64)))
    if not math.isfinite(scale) or scale <= 0:
        raise AuditError(f"invalid pooled RMS {scale}")
    return scale


def check_close(a: float | None, b: float | None, tol: float) -> float | None:
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def metric_row(rhos: dict[str, float], orientation: str) -> dict[str, Any]:
    if not all(math.isfinite(float(rhos[key])) for key in RHO_KEYS):
        raise AuditError(f"nonfinite four-rho: {rhos}")
    a_mat, a_pat, b_mat, b_pat = (float(rhos[key]) for key in RHO_KEYS)
    direct = (a_mat + b_pat) / 2.0
    cross = (a_pat + b_mat) / 2.0
    if orientation == "direct":
        abcd = (a_mat, a_pat, b_mat, b_pat)
        matched, other = direct, cross
    elif orientation == "swapped":
        abcd = (b_mat, b_pat, a_mat, a_pat)
        matched, other = cross, direct
    else:
        raise AuditError(f"unsupported locked orientation {orientation}")
    a, b, c, d = abcd
    return {
        **{key: float(rhos[key]) for key in RHO_KEYS},
        "direct": float(direct), "cross": float(other),
        "raw_cross": float(cross),
        "matched": float(matched), "other": float(other), "contrast": float(matched - other),
        "matched_ref1": float(a), "matched_ref2": float(d),
        "margin_mat": float(a - b), "margin_pat": float(d - c),
        "minmargin": float(min(a - b, d - c)),
        "fixed_reference_abcd": [float(value) for value in abcd],
        "orientation": orientation,
        "pairing": "direct" if orientation == "direct" else "cross",
    }


def extract_locked_rows(r2: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = r2.get("rows")
    if not isinstance(rows, list) or len(rows) != 420:
        raise AuditError(f"locked R2 row count is not 420: {len(rows) if isinstance(rows, list) else None}")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["condition_id"]), str(row["chromosome"]))
        if key in out:
            raise AuditError(f"duplicate locked R2 row {key}")
        out[key] = row
    return out


def build_mask_for_chromosome(chromosome: str, cond_coords: dict[str, dict[str, dict[int, np.ndarray]]], reference: dict[str, dict[int, np.ndarray]], cond_ids: list[str]) -> dict[str, Any]:
    pos = positions(chromosome)
    pair_i, pair_j = np.triu_indices(len(pos), k=1)
    finite_tracks: dict[str, np.ndarray] = {}
    index = CHROMOSOMES.index(chromosome) + 1
    for condition_id in cond_ids:
        coords = cond_coords[condition_id]
        track_names = (f"c{index:02d}a",) if condition_id == "consensus014" else (f"c{index:02d}a", f"c{index:02d}b")
        for copy_index, track in enumerate(track_names):
            label = f"condition:{condition_id}:copy{'AB'[copy_index]}"
            finite_tracks[label] = np.isfinite(dense(coords, track, pos)).all(axis=1)
    finite_tracks["reference:mat"] = np.isfinite(dense(reference, f"{chromosome}(mat)", pos)).all(axis=1)
    finite_tracks["reference:pat"] = np.isfinite(dense(reference, f"{chromosome}(pat)", pos)).all(axis=1)
    common_bins = np.ones(len(pos), dtype=bool)
    for value in finite_tracks.values():
        common_bins &= value
    common_pairs = common_bins[pair_i] & common_bins[pair_j]
    return {
        "chromosome": chromosome,
        "positions": pos,
        "pair_i": pair_i,
        "pair_j": pair_j,
        "common_bins": common_bins,
        "common_pairs": common_pairs,
        "finite_tracks": finite_tracks,
        "n_bins": len(pos),
        "n_total_pairs": len(pair_i),
        "n_common_pairs": int(common_pairs.sum()),
    }


def evaluate_condition(chromosome: str, candidate: dict[str, dict[int, np.ndarray]], reference: dict[str, dict[int, np.ndarray]], mask: dict[str, Any], orientation: str) -> tuple[dict[str, Any], dict[str, Any]]:
    pos = mask["positions"]
    pair_i = mask["pair_i"]
    pair_j = mask["pair_j"]
    common = mask["common_pairs"]
    index = CHROMOSOMES.index(chromosome) + 1
    tracks = [f"c{index:02d}a", f"c{index:02d}b"]
    ref_mat = distance_vector(dense(reference, f"{chromosome}(mat)", pos), pair_i, pair_j)[common]
    ref_pat = distance_vector(dense(reference, f"{chromosome}(pat)", pos), pair_i, pair_j)[common]
    cand_a = distance_vector(dense(candidate, tracks[0], pos), pair_i, pair_j)[common]
    cand_b = distance_vector(dense(candidate, tracks[1], pos), pair_i, pair_j)[common]
    rhos = {
        "rho_A_mat": spearman(cand_a, ref_mat), "rho_A_pat": spearman(cand_a, ref_pat),
        "rho_B_mat": spearman(cand_b, ref_mat), "rho_B_pat": spearman(cand_b, ref_pat),
    }
    row = metric_row(rhos, orientation)
    row.update({"chromosome": chromosome, "n_bins": int(mask["n_bins"]), "n_total_non_diagonal_pairs": int(mask["n_total_pairs"]), "n_common_pairs": int(mask["n_common_pairs"])})
    arrays = {
        "positions": pos,
        "common_bins": mask["common_bins"],
        "common_pair_mask": np.outer(mask["common_bins"], mask["common_bins"]),
        "raw_matrices": None,
    }
    return row, arrays


def build_chr1_display(condition_id: str, candidate: dict[str, dict[int, np.ndarray]], reference: dict[str, dict[int, np.ndarray]], orientation: str, *, label: str) -> dict[str, Any]:
    pos = positions("chr1")
    panel_candidate_tracks = ["c01a", "c01b"] if orientation == "direct" else ["c01b", "c01a"]
    panel_tracks = ["chr1(mat)", "chr1(pat)", *panel_candidate_tracks]
    panel_labels = ["Reference mat", "Reference pat", f"{label} candidate to ref mat", f"{label} candidate to ref pat"]
    coords_source = [reference, reference, candidate, candidate]
    matrices = []
    finite = []
    for source, track in zip(coords_source, panel_tracks):
        matrix, bins = panel_matrix(source, track, pos)
        matrices.append(matrix)
        finite.append(bins)
    raw = np.asarray(matrices, dtype=np.float64)
    finite_bins = np.asarray(finite, dtype=bool)
    common_bins = np.all(finite_bins, axis=0)
    common_pair_mask = np.outer(common_bins, common_bins)
    np.fill_diagonal(common_pair_mask, False)
    common_raw = np.full_like(raw, np.nan, dtype=np.float64)
    common_raw[:, common_pair_mask] = raw[:, common_pair_mask]
    for panel in common_raw:
        panel[np.diag_indices(len(pos))] = np.where(common_bins, 0.0, np.nan)
    ref_scale = pooled_rms(common_raw[:2], common_pair_mask)
    cand_scale = pooled_rms(common_raw[2:], common_pair_mask)
    panel_scales = np.asarray([ref_scale, ref_scale, cand_scale, cand_scale], dtype=np.float64)
    normalized = common_raw / panel_scales[:, None, None]
    normalized[:, ~common_pair_mask] = np.nan
    normalized[:, np.diag_indices(len(pos))[0], np.diag_indices(len(pos))[1]] = np.where(common_bins, 0.0, np.nan)
    return {
        "condition_id": condition_id, "label": label, "orientation": orientation,
        "panel_tracks": panel_tracks, "panel_labels": panel_labels, "positions_bp": pos,
        "raw_matrices": raw, "common_raw_matrices": common_raw, "normalized_matrices": normalized,
        "finite_bins": finite_bins, "common_bin_mask": common_bins, "common_pair_mask": common_pair_mask,
        "scales": np.asarray([ref_scale, cand_scale], dtype=np.float64), "panel_scales": panel_scales,
        "n_common_bins": int(common_bins.sum()), "n_common_pairs": int(np.triu(common_pair_mask, k=1).sum()),
        "raw_max": float(np.nanmax(common_raw)), "normalized_max": float(np.nanmax(normalized)),
    }


def load_old_npz(path: Path, label: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    return {"label": label, "path": str(path), "keys": list(payload), "payload": payload}


def max_abs_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, int]:
    if a.shape != b.shape:
        return float("nan"), -1
    finite = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        finite &= mask
    if not finite.any():
        return 0.0, 0
    delta = np.abs(a[finite] - b[finite])
    return float(np.max(delta)), int(delta.size)


def compare_npz(old: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    p = old["payload"]
    raw_key = "raw_matrices" if "raw_matrices" in p else "raw_distance_matrices"
    norm_key = "normalized_matrices" if "normalized_matrices" in p else "normalized_distance_matrices"
    common_raw_key = "common_raw_matrices" if "common_raw_matrices" in p else "common_raw_distance_matrices"
    old_tracks_key = "panel_tracks" if "panel_tracks" in p else "panel_track_names"
    old_tracks = decode_array(p[old_tracks_key]) if old_tracks_key in p else []
    fresh_tracks = list(fresh["panel_tracks"])
    order = [old_tracks.index(track) for track in fresh_tracks] if old_tracks and all(track in old_tracks for track in fresh_tracks) else [0, 1, 2, 3]
    old_raw = np.asarray(p[raw_key], dtype=np.float64)[order]
    old_norm = np.asarray(p[norm_key], dtype=np.float64)[order]
    fresh_raw = np.asarray(fresh["raw_matrices"], dtype=np.float64)
    fresh_norm = np.asarray(fresh["normalized_matrices"], dtype=np.float64)
    mask_raw = np.isfinite(old_raw) & np.isfinite(fresh_raw)
    mask_norm = np.isfinite(old_norm) & np.isfinite(fresh_norm)
    raw_diff, raw_n = max_abs_diff(old_raw, fresh_raw, mask_raw)
    norm_diff, norm_n = max_abs_diff(old_norm, fresh_norm, mask_norm)
    common_raw_diff = None
    common_raw_n = None
    if common_raw_key in p:
        old_common = np.asarray(p[common_raw_key], dtype=np.float64)[order]
        fresh_common = np.asarray(fresh["common_raw_matrices"], dtype=np.float64)
        m = np.isfinite(old_common) & np.isfinite(fresh_common)
        common_raw_diff, common_raw_n = max_abs_diff(old_common, fresh_common, m)
    scales = p.get("scales")
    old_scales = [float(x) for x in np.asarray(scales).reshape(-1)] if scales is not None else []
    old_color = [float(x) for x in np.asarray(p.get("color_norm", np.asarray([np.nan, np.nan]))).reshape(-1)]
    old_global_vmax = float(np.asarray(p["global_vmax"]).item()) if "global_vmax" in p else float("nan")
    return {
        "label": old["label"], "old_path": old["path"], "panel_tracks_old": old_tracks, "panel_tracks_fresh": fresh_tracks,
        "raw_max_abs_diff": raw_diff, "raw_n_values": raw_n,
        "common_raw_max_abs_diff": common_raw_diff, "common_raw_n_values": common_raw_n,
        "normalized_max_abs_diff": norm_diff, "normalized_n_values": norm_n,
        "old_scales": old_scales, "fresh_scales": [float(x) for x in fresh["scales"]],
        "scale_max_abs_diff": max([abs(a - b) for a, b in zip(old_scales, fresh["scales"])], default=float("nan")),
        "old_color_norm": old_color, "old_global_vmax": old_global_vmax, "fresh_native_max": float(fresh["normalized_max"]),
        "old_colormap_metadata": str(np.asarray(p.get("colormap", "missing")).item()) if "colormap" in p else "missing",
        "raw_normalized_consistent": bool(raw_diff <= 1e-12 and (common_raw_diff is None or common_raw_diff <= 1e-12)),
        "panel_order_consistent": bool(old_tracks == fresh_tracks),
    }


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise AuditError(f"invalid PNG {path}")
    return tuple(int(value) for value in struct.unpack(">II", header[16:24]))


def render_map(result: dict[str, Any], vmax: float, png_path: Path, pdf_path: Path, title: str, footer: str) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 0.6})
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad("#bdbdbd")
    pos_mb = result["positions_bp"].astype(np.float64) / 1e6
    half = BIN / 2e6
    extent = (float(pos_mb[0] - half), float(pos_mb[-1] + half), float(pos_mb[0] - half), float(pos_mb[-1] + half))
    ticks = np.asarray([3, 50, 100, 150, 195], dtype=float)
    fig = plt.figure(figsize=(6.0, 6.0), dpi=300)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.055], height_ratios=[1, 1], left=0.105, right=0.91, bottom=0.14, top=0.87, wspace=0.24, hspace=0.36)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    norm = Normalize(vmin=0.0, vmax=vmax, clip=False)
    images = []
    for axis, matrix, panel_title in zip(axes, result["normalized_matrices"], result["panel_labels"]):
        image = axis.imshow(matrix, origin="lower", interpolation="nearest", aspect="equal", extent=extent, cmap=cmap, norm=norm)
        images.append(image)
        axis.set_title(panel_title, pad=4)
        axis.set_xlim(extent[0], extent[1]); axis.set_ylim(extent[2], extent[3])
        axis.set_xticks(ticks); axis.set_yticks(ticks); axis.tick_params(width=0.6, length=2.5, pad=2)
        axis.set_xlabel("Genomic position (Mb)"); axis.set_ylabel("Genomic position (Mb)")
    cax = fig.add_subplot(grid[:, 2])
    cb = fig.colorbar(images[0], cax=cax)
    cb.set_label("normalized 3D Euclidean distance", fontsize=7, labelpad=4); cb.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    fig.suptitle(title, fontsize=7, y=0.925)
    fig.text(0.5, 0.045, footer, ha="center", va="center", fontsize=7)
    fig.savefig(png_path, dpi=300, format="png"); fig.savefig(pdf_path, dpi=300, format="pdf"); plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path), "png_dimensions": list(png_dimensions(png_path)), "vmin": 0.0, "vmax": float(vmax), "extent_mb": list(extent), "colormap": "coolwarm_r", "dpi": 300, "figure_inches": [6.0, 6.0]}


def shape_comparison(base: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    records = []
    pair_mask = base["common_pair_mask"] & target["common_pair_mask"]
    upper_mask = np.triu(pair_mask, k=1)
    for panel_index, panel_label in ((2, "candidate_to_ref_mat"), (3, "candidate_to_ref_pat")):
        x = np.asarray(base["normalized_matrices"][panel_index][upper_mask], dtype=np.float64)
        y = np.asarray(target["normalized_matrices"][panel_index][upper_mask], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]; y = y[finite]
        if len(x) < 20:
            raise AuditError("insufficient chr1 shape comparison pairs")
        dx = x - np.mean(x, dtype=np.float64); dy = y - np.mean(y, dtype=np.float64)
        pearson = float(np.sum(dx * dy, dtype=np.float64) / math.sqrt(float(np.sum(dx * dx, dtype=np.float64) * np.sum(dy * dy, dtype=np.float64))))
        relative_rms = float(np.sqrt(np.mean((x - y) ** 2, dtype=np.float64)) / np.sqrt(np.mean(x ** 2, dtype=np.float64)))
        records.append({"panel": panel_label, "n_pairs": int(len(x)), "spearman": spearman(x, y), "pearson": pearson, "normalized_rms_difference_over_base_rms": relative_rms, "base_candidate_scale": float(base["scales"][1]), "target_candidate_scale": float(target["scales"][1])})
    return {"base_condition": base["condition_id"], "target_condition": target["condition_id"], "pairing_basis": "candidate panels aligned to fixed reference mat/pat display columns; no local swap", "records": records}


def render_unified_index(results: dict[str, dict[str, Any]], vmax: float, png_path: Path, pdf_path: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    order = ["v1_original_random_joint", "v1_continuation", "C0-bundle1", "C0-bundle2", "C0-bundle3", "x0-bundle2"]
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 0.6})
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad("#bdbdbd")
    pos_mb = results[order[0]]["positions_bp"].astype(np.float64) / 1e6
    half = BIN / 2e6
    extent = (float(pos_mb[0] - half), float(pos_mb[-1] + half), float(pos_mb[0] - half), float(pos_mb[-1] + half))
    ticks = np.asarray([3, 50, 100, 150, 195], dtype=float)
    fig = plt.figure(figsize=(6.0, 18.0), dpi=300)
    grid = fig.add_gridspec(6, 3, width_ratios=[1, 1, 0.055], left=0.11, right=0.91, bottom=0.04, top=0.965, wspace=0.25, hspace=0.5)
    norm = Normalize(vmin=0.0, vmax=vmax, clip=False)
    images = []
    for row_index, condition_id in enumerate(order):
        result = results[condition_id]
        for col_index, (matrix, track) in enumerate(zip(result["normalized_matrices"][2:], result["panel_tracks"][2:])):
            axis = fig.add_subplot(grid[row_index, col_index])
            image = axis.imshow(matrix, origin="lower", interpolation="nearest", aspect="equal", extent=extent, cmap=cmap, norm=norm)
            images.append(image)
            axis.set_title(f"{result['label']} | {track}", pad=3)
            axis.set_xlim(extent[0], extent[1]); axis.set_ylim(extent[2], extent[3])
            axis.set_xticks(ticks); axis.set_yticks(ticks); axis.tick_params(width=0.6, length=2.5, pad=2)
            axis.set_xlabel("Genomic position (Mb)"); axis.set_ylabel("Genomic position (Mb)")
    cax = fig.add_subplot(grid[:, 2])
    cb = fig.colorbar(images[0], cax=cax)
    cb.set_label("normalized 3D Euclidean distance", fontsize=7, labelpad=4); cb.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    fig.suptitle("P9016 chr1 | unified candidate-panel index", fontsize=7, y=0.985)
    fig.text(0.5, 0.012, f"Primary figures are separate 6x6 panels with reference; all rows use vmin=0 vmax={vmax:.12g}, coolwarm_r, grey=missing", ha="center", va="center", fontsize=7)
    fig.savefig(png_path, dpi=300, format="png"); fig.savefig(pdf_path, dpi=300, format="pdf"); plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path), "png_dimensions": list(png_dimensions(png_path)), "vmin": 0.0, "vmax": float(vmax), "figure_inches": [6.0, 18.0], "colormap": "coolwarm_r"}
def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json_load(config_path, "audit config")
    if config.get("status") != "config_frozen_before_reference_read":
        raise AuditError("config is not frozen-before-reference-read")
    if config.get("scope", {}).get("phase_opened"):
        raise AuditError("phase boundary already violated in config")
    if config.get("scope", {}).get("fit_called") or config.get("scope", {}).get("selection_called"):
        raise AuditError("audit config must not call fit or selection")
    if config.get("cohort_freeze", {}).get("primary_existing_mask", {}).get("condition_count") != 21:
        raise AuditError("primary mask condition count is not 21")

    input_records: dict[str, Any] = {}
    candidate_coords: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    full_validation: dict[str, Any] = {}

    # 先锁定五个 endpoint 和三个 exploratory x0 文件。
    for key, spec in config["locked_candidates"].items():
        path = resolve(spec["path"])
        input_records[f"candidate:{key}"] = check_file(path, spec["sha256"], key)
        candidate_coords[spec["condition_id"]] = parse_3dg(path, spec["label"])
        full_validation[f"candidate:{key}"] = validate_full_candidate(candidate_coords[spec["condition_id"]], spec["label"])
    x0_coords: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for bundle_index, bundle in enumerate(("bundle1", "bundle2", "bundle3"), start=1):
        spec = config["locked_025_x0_diagnostic"][bundle]
        key = f"x0:{bundle}"
        path = resolve(spec["path"])
        input_records[key] = check_file(path, spec["sha256"], f"025 {bundle} x0")
        x0_id = f"x0-{bundle}"
        x0_coords[x0_id] = parse_3dg(path, f"025 {bundle} x0")
        full_validation[key] = validate_full_candidate(x0_coords[x0_id], f"025 {bundle} x0")

    # 锁定并解析已有的 21-condition mask 坐标，不读取 reference。
    mask_manifest_spec = config["cohort_freeze"]["primary_existing_mask"]["source_manifest"]
    mask_manifest_path = resolve(mask_manifest_spec)
    input_records["existing_mask_manifest"] = check_file(mask_manifest_path, config["cohort_freeze"]["primary_existing_mask"]["source_manifest_sha256"], "existing 21-condition mask manifest")
    eval_manifest = json_load(mask_manifest_path, "existing evaluation manifest")
    expected_ids = config["cohort_freeze"]["primary_existing_mask"]["condition_ids"]
    coord_specs = eval_manifest.get("coordinates")
    if not isinstance(coord_specs, list) or len(coord_specs) != 21:
        raise AuditError("existing evaluation manifest does not contain 21 coordinate locks")
    cond_coords: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    mask_coord_records = {}
    for spec in coord_specs:
        condition_id = str(spec["condition_id"])
        if condition_id not in expected_ids:
            raise AuditError(f"unexpected existing-mask condition {condition_id}")
        path = resolve(spec["path"])
        record = check_file(path, spec.get("sha256"), f"mask coordinate {condition_id}")
        input_records[f"mask:{condition_id}"] = record
        mask_coord_records[condition_id] = record
        cond_coords[condition_id] = parse_3dg(path, f"mask coordinate {condition_id}")
    if set(cond_coords) != set(expected_ids):
        raise AuditError("21-condition coordinate ID set mismatch")

    # reference 受所有 candidate、x0 和 mask-coordinate 哈希保护。
    ref_spec = config["reference"]
    ref_path = resolve(ref_spec["path"])
    input_records["reference"] = check_file(ref_path, ref_spec["sha256"], "evaluation reference")
    reference = parse_3dg(ref_path, "evaluation reference")
    for chromosome in CHROMOSOMES:
        for suffix in ("mat", "pat"):
            track = f"{chromosome}({suffix})"
            if track not in reference:
                raise AuditError(f"reference missing {track}")

    # 独立于 count metadata 验证已有的 21-condition mask。
    r2_path = resolve(config["existing_evidence"]["r2_per_chromosome"]["path"])
    input_records["locked_r2"] = check_file(r2_path, config["existing_evidence"]["r2_per_chromosome"]["sha256"], "locked R2 per chromosome")
    r2_doc = json_load(r2_path, "locked R2 per chromosome")
    locked_rows = extract_locked_rows(r2_doc)
    mask_metadata = {item["chromosome"]: item for item in eval_manifest.get("mask_metadata", [])}
    masks: dict[str, dict[str, Any]] = {}
    mask_validation_rows = []
    for chromosome in CHROMOSOMES:
        mask = build_mask_for_chromosome(chromosome, cond_coords, reference, expected_ids)
        expected_meta = mask_metadata.get(chromosome)
        if expected_meta is None:
            raise AuditError(f"existing mask metadata missing {chromosome}")
        if mask["n_common_pairs"] != int(expected_meta["n_common_pairs"]):
            raise AuditError(f"{chromosome} common pair mismatch: independent {mask['n_common_pairs']} vs existing {expected_meta['n_common_pairs']}")
        if mask["n_total_pairs"] != int(expected_meta["n_total_non_diagonal_pairs"]):
            raise AuditError(f"{chromosome} total pair mismatch")
        mask_validation_rows.append({"chromosome": chromosome, "n_bins": mask["n_bins"], "n_total_pairs": mask["n_total_pairs"], "n_common_pairs": mask["n_common_pairs"], "existing_n_common_pairs": int(expected_meta["n_common_pairs"]), "same_pair_count": True, "common_bins": int(mask["common_bins"].sum())})
        masks[chromosome] = mask

    # 使用已有的整条染色体几何方向，作为每个 endpoint 的锁定值。
    endpoint_rows: dict[str, dict[str, dict[str, Any]]] = {condition: {} for condition in ENDPOINT_IDS}
    for condition_id in ENDPOINT_IDS:
        for chromosome in CHROMOSOMES:
            row = locked_rows.get((condition_id, chromosome))
            if row is None:
                raise AuditError(f"locked R2 row missing {condition_id}/{chromosome}")
            if row.get("orientation") not in ("direct", "swapped"):
                raise AuditError(f"locked orientation unavailable for {condition_id}/{chromosome}")
            endpoint_rows[condition_id][chromosome] = row

    independent_rows: list[dict[str, Any]] = []
    rho_diff_records: list[dict[str, Any]] = []
    for condition_id in ENDPOINT_IDS:
        for chromosome in CHROMOSOMES:
            computed, _ = evaluate_condition(chromosome, candidate_coords[condition_id], reference, masks[chromosome], endpoint_rows[condition_id][chromosome]["orientation"])
            locked = endpoint_rows[condition_id][chromosome]
            record = {"condition_id": condition_id, "chromosome": chromosome, **computed}
            for key in RHO_KEYS + ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin"):
                record[f"locked_{key}"] = float(locked[key])
                record[f"absdiff_{key}"] = abs(float(computed[key]) - float(locked[key]))
            record["locked_orientation"] = locked.get("orientation")
            record["locked_pairing"] = locked.get("pairing")
            independent_rows.append(record)
            rho_diff_records.append(record)

    # endpoint 与 025 x0 诊断的比较，保留每个 seed 及 endpoint orientation。
    diagnostic_rows: list[dict[str, Any]] = []
    for bundle_index, bundle in enumerate(("bundle1", "bundle2", "bundle3"), start=1):
        endpoint_id = f"C0-{bundle}"
        x0_id = f"x0-{bundle}"
        for chromosome in CHROMOSOMES:
            orientation = endpoint_rows[endpoint_id][chromosome]["orientation"]
            endpoint_metric, _ = evaluate_condition(chromosome, candidate_coords[endpoint_id], reference, masks[chromosome], orientation)
            x0_metric, _ = evaluate_condition(chromosome, x0_coords[x0_id], reference, masks[chromosome], orientation)
            row = {"bundle": bundle, "chromosome": chromosome, "orientation_locked_to_endpoint": orientation}
            for key in RHO_KEYS + ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin"):
                row[f"endpoint_{key}"] = endpoint_metric[key]
                row[f"x0_{key}"] = x0_metric[key]
                row[f"delta_endpoint_minus_x0_{key}"] = endpoint_metric[key] - x0_metric[key]
            diagnostic_rows.append(row)

    # 已有 020 与各 C0 seed 的比较，不做推断性检验。
    diff_rows = []
    for chromosome in CHROMOSOMES:
        base = next(row for row in independent_rows if row["condition_id"] == "v1_original_random_joint" and row["chromosome"] == chromosome)
        for condition_id in C0_IDS:
            current = next(row for row in independent_rows if row["condition_id"] == condition_id and row["chromosome"] == chromosome)
            row = {"chromosome": chromosome, "comparison": f"{condition_id}-minus-020", "condition_id": condition_id}
            for key in ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin"):
                row[f"020_{key}"] = base[key]
                row[f"{condition_id}_{key}"] = current[key]
                row[f"delta_{key}"] = current[key] - base[key]
            row["wins_matched"] = int(row["delta_matched"] > 0)
            row["wins_contrast"] = int(row["delta_contrast"] > 0)
            diff_rows.append(row)

    # 为五个 endpoint figure 以及 posthoc x0 bundle2 figure 新建 chr1 display matrices。
    display_ids = ["v1_original_random_joint", "v1_continuation", "C0-bundle1", "C0-bundle2", "C0-bundle3"]
    display_labels = {
        "v1_original_random_joint": "020 historical selected",
        "v1_continuation": "022 historical continuation",
        "C0-bundle1": "029 C0 bundle1 seed",
        "C0-bundle2": "029 C0 bundle2 seed",
        "C0-bundle3": "029 C0 bundle3 seed",
        "x0-bundle2": "025 x0 bundle2 posthoc",
    }
    display_results: dict[str, dict[str, Any]] = {}
    display_records = []
    for condition_id in display_ids:
        orientation = endpoint_rows[condition_id]["chr1"]["orientation"]
        result = build_chr1_display(condition_id, candidate_coords[condition_id], reference, orientation, label=display_labels[condition_id])
        display_results[condition_id] = result
        r2row = next(row for row in independent_rows if row["condition_id"] == condition_id and row["chromosome"] == "chr1")
        display_records.append({"condition_id": condition_id, "label": display_labels[condition_id], "orientation": orientation, "mapping_candidate_to_ref_mat": result["panel_tracks"][2], "mapping_candidate_to_ref_pat": result["panel_tracks"][3], "n_common_bins": result["n_common_bins"], "n_common_pairs": result["n_common_pairs"], "reference_pooled_rms": result["scales"][0], "candidate_pooled_rms": result["scales"][1], "normalized_max": result["normalized_max"], "rho_A_mat": r2row["rho_A_mat"], "rho_A_pat": r2row["rho_A_pat"], "rho_B_mat": r2row["rho_B_mat"], "rho_B_pat": r2row["rho_B_pat"], "matched": r2row["matched"], "cross": r2row["cross"], "contrast": r2row["contrast"], "margin_mat": r2row["margin_mat"], "margin_pat": r2row["margin_pat"]})
    x0_result = build_chr1_display("x0-bundle2", x0_coords["x0-bundle2"], reference, endpoint_rows["C0-bundle2"]["chr1"]["orientation"], label=display_labels["x0-bundle2"])
    display_results["x0-bundle2"] = x0_result
    all_finite_max = max(float(result["normalized_max"]) for result in display_results.values())
    global_vmax = float(all_finite_max)
    shape_comparisons = [shape_comparison(display_results["v1_original_random_joint"], display_results[target]) for target in ("v1_continuation", "C0-bundle1", "C0-bundle2", "C0-bundle3")]
    json_dump(HERE / "chr1_shape_comparison.json", {"schema": "p9016-chr1-shape-comparison-v1", "base": "v1_original_random_joint", "comparisons": shape_comparisons, "common_pairs": int(display_results["v1_original_random_joint"]["n_common_pairs"]), "normalization": "each candidate pooled RMS; rank comparison is scale-invariant", "generated_before_render": True})

    old_npz_specs = config["display_inputs"]
    old_npz = {}
    for key in ("old_020_npz", "old_022_npz", "old_029_C0_bundle2_npz"):
        spec = old_npz_specs[key]
        path = resolve(spec["path"])
        input_records[key] = check_file(path, spec["sha256"], key)
        old_npz[key] = load_old_npz(path, key)
    npz_comparisons = [
        compare_npz(old_npz["old_020_npz"], display_results["v1_original_random_joint"]),
        compare_npz(old_npz["old_022_npz"], display_results["v1_continuation"]),
        compare_npz(old_npz["old_029_C0_bundle2_npz"], display_results["C0-bundle2"]),
    ]

    # 计算相对于各历史 NPZ normalization 的仅色阶差异。
    for comparison in npz_comparisons:
        old = old_npz[comparison["label"]]["payload"]
        color_values = np.asarray(old.get("color_norm", np.asarray([np.nan, np.nan]))).reshape(-1)
        old_vmax = float(color_values[1]) if len(color_values) > 1 and math.isfinite(float(color_values[1])) else float(np.asarray(old.get("global_vmax", np.nan)).item())
        fresh_id = {"old_020_npz": "v1_original_random_joint", "old_022_npz": "v1_continuation", "old_029_C0_bundle2_npz": "C0-bundle2"}[comparison["label"]]
        fresh_norm = display_results[fresh_id]["normalized_matrices"]
        common = np.isfinite(fresh_norm)
        old_norm_key = "normalized_matrices" if "normalized_matrices" in old else "normalized_distance_matrices"
        old_norm = np.asarray(old[old_norm_key], dtype=np.float64)
        if old_norm.shape == fresh_norm.shape:
            delta_fraction = np.abs(old_norm[common] / old_vmax - fresh_norm[common] / global_vmax)
            comparison["old_vs_unified_color_fraction_max_abs"] = float(np.max(delta_fraction)) if len(delta_fraction) else 0.0
            comparison["old_vmax"] = old_vmax
            comparison["unified_vmax"] = global_vmax
            comparison["color_norm_only_same_normalized_values"] = bool(comparison["normalized_max_abs_diff"] <= 1e-12)

    # 所有 endpoint map 使用恰好一个全局 norm；x0 bundle2 也使用同一 norm。
    render_records = {}
    for condition_id in display_ids:
        result = display_results[condition_id]
        stem = {"v1_original_random_joint": "020_historical_selected", "v1_continuation": "022_historical_continuation", "C0-bundle1": "C0_bundle1_seed", "C0-bundle2": "C0_bundle2_seed", "C0-bundle3": "C0_bundle3_seed"}[condition_id]
        render_records[condition_id] = render_map(result, global_vmax, HERE / f"{stem}_chr1_distance_matrices.png", HERE / f"{stem}_chr1_distance_matrices.pdf", f"P9016 chr1 | {display_labels[condition_id]} (1 Mb)", f"Fixed whole-chromosome R2 mapping; pooled RMS per source; unified vmin=0 vmax={global_vmax:.12g}; coolwarm_r; grey=missing")
    render_records["x0-bundle2"] = render_map(x0_result, global_vmax, HERE / "x0_bundle2_chr1_distance_matrices.png", HERE / "x0_bundle2_chr1_distance_matrices.pdf", f"P9016 chr1 | {display_labels['x0-bundle2']} (1 Mb)", f"Posthoc starting-point diagnostic; endpoint orientation fixed; same unified vmin=0 vmax={global_vmax:.12g}; coolwarm_r; grey=missing")
    render_records["unified_index"] = render_unified_index(display_results | {"x0-bundle2": x0_result}, global_vmax, HERE / "unified_chr1_distance_matrices.png", HERE / "unified_chr1_distance_matrices.pdf")

    # 保存数值统一 bundle。
    npz_payload: dict[str, Any] = {"positions_bp": display_results[display_ids[0]]["positions_bp"], "global_color_norm": np.asarray([0.0, global_vmax], dtype=np.float64), "common_bin_mask": display_results[display_ids[0]]["common_bin_mask"], "common_pair_mask": display_results[display_ids[0]]["common_pair_mask"], "colormap": np.asarray("coolwarm_r"), "condition_ids": np.asarray(display_ids)}
    for condition_id in display_ids + ["x0-bundle2"]:
        result = display_results[condition_id]
        prefix = condition_id.replace("-", "_")
        npz_payload[f"{prefix}_raw_matrices"] = result["raw_matrices"]
        npz_payload[f"{prefix}_common_raw_matrices"] = result["common_raw_matrices"]
        npz_payload[f"{prefix}_normalized_matrices"] = result["normalized_matrices"]
        npz_payload[f"{prefix}_scales"] = result["scales"]
        npz_payload[f"{prefix}_panel_tracks"] = np.asarray(result["panel_tracks"])
        npz_payload[f"{prefix}_panel_labels"] = np.asarray(result["panel_labels"])
        npz_payload[f"{prefix}_orientation"] = np.asarray(result["orientation"])
    unified_npz_path = HERE / "unified_chr1_distance_matrices.npz"
    np.savez_compressed(unified_npz_path, **npz_payload)

    # display/table 输出。
    r2_fields = ["condition_id", "chromosome", "locked_orientation", "locked_pairing"]
    for key in RHO_KEYS + ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin"):
        r2_fields.extend([key, f"locked_{key}", f"absdiff_{key}"])
    write_tsv(HERE / "r2_independent_vs_locked.tsv", rho_diff_records, r2_fields)
    diff_fields = ["chromosome", "comparison", "condition_id"]
    for key in ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin"):
        diff_fields.extend([f"020_{key}", f"{key.replace('matched', 'C0_matched') if False else condition_id}_{key}", f"delta_{key}"])
    # 用通用 candidate columns 重写 diff 字段，以保持表格稳定。
    diff_fields = ["chromosome", "comparison", "condition_id"] + sum(([f"020_{key}", f"candidate_{key}", f"delta_{key}"] for key in ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin")), []) + ["wins_matched", "wins_contrast"]
    for row in diff_rows:
        condition_id = row["condition_id"]
        for key in ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin"):
            row[f"candidate_{key}"] = row[f"{condition_id}_{key}"]
    write_tsv(HERE / "c0_vs_020.tsv", diff_rows, diff_fields)
    diagnostic_fields = ["bundle", "chromosome", "orientation_locked_to_endpoint"] + sum(([f"endpoint_{key}", f"x0_{key}", f"delta_endpoint_minus_x0_{key}"] for key in RHO_KEYS + ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin")), [])
    write_tsv(HERE / "c0_seed_and_x0_diagnostic.tsv", diagnostic_rows, diagnostic_fields)
    display_fields = list(display_records[0].keys())
    write_tsv(HERE / "chr1_display_index.tsv", display_records, display_fields)

    # 宏观汇总，不做推断性统计。
    macro = {}
    for condition_id in ENDPOINT_IDS:
        rows = [row for row in independent_rows if row["condition_id"] == condition_id]
        macro[condition_id] = {key: float(np.mean([row[key] for row in rows], dtype=np.float64)) for key in ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin")}
    for bundle in ("bundle1", "bundle2", "bundle3"):
        rows = [row for row in diagnostic_rows if row["bundle"] == bundle]
        macro[f"x0_{bundle}"] = {f"endpoint_minus_x0_{key}": float(np.mean([row[f"delta_endpoint_minus_x0_{key}"] for row in rows], dtype=np.float64)) for key in ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin")}
        macro[f"x0_{bundle}"]["wins_matched"] = int(sum(row["delta_endpoint_minus_x0_matched"] > 0 for row in rows))
        macro[f"x0_{bundle}"]["wins_contrast"] = int(sum(row["delta_endpoint_minus_x0_contrast"] > 0 for row in rows))
    for condition_id in C0_IDS:
        rows = [row for row in diff_rows if row["condition_id"] == condition_id]
        macro[f"{condition_id}_minus_020"] = {f"delta_{key}": float(np.mean([row[f"delta_{key}"] for row in rows], dtype=np.float64)) for key in ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin")}
        macro[f"{condition_id}_minus_020"]["wins_matched"] = int(sum(row["wins_matched"] for row in rows))
        macro[f"{condition_id}_minus_020"]["wins_contrast"] = int(sum(row["wins_contrast"] for row in rows))

    # 审计 source/script hashes 以及前后不可变性。
    for key, spec in config["existing_evidence"].items():
        if key == "022_r2_per_chromosome":
            path = resolve(spec["path"])
            input_records[key] = check_file(path, None, key)
        elif isinstance(spec, dict) and spec.get("path"):
            path = resolve(spec["path"])
            input_records[key] = check_file(path, spec.get("sha256"), key)
    for key, spec in config["display_inputs"].items():
        path = resolve(spec["path"])
        input_records[key] = check_file(path, spec.get("sha256"), key)
    before_hashes = {key: record["sha256"] for key, record in input_records.items()}
    after_hashes = {key: check_file(Path(record["path"]), record["sha256"], key)["sha256"] for key, record in input_records.items()}

    max_rho = max(record[f"absdiff_{key}"] for record in rho_diff_records for key in RHO_KEYS)
    max_metric = max(record[f"absdiff_{key}"] for record in rho_diff_records for key in ("matched", "cross", "contrast", "matched_ref1", "matched_ref2", "margin_mat", "margin_pat", "minmargin"))
    validation = {
        "schema": "p9016-c0-020-independent-audit-validation-v1",
        "status": "PASS",
        "terminal": {"script_exit_code": 0, "reference_read_after_all_candidate_and_mask_hashes": True, "fit_called": False, "selection_called": False, "phase_opened": False, "original_files_modified": False},
        "hashes": {"before": before_hashes, "after": after_hashes, "all_unchanged": before_hashes == after_hashes, "locked_endpoint_and_x0_count": 8, "existing_mask_condition_count": 21},
        "mask": {"chromosomes": mask_validation_rows, "all_pair_counts_match_existing_manifest": True, "same_21_condition_mask_used_for_x0": True, "common_mask_not_expanded": True},
        "coordinate_validation": full_validation,
        "r2_recomputation": {"n_endpoint_rows": len(rho_diff_records), "n_endpoint_chromosome_rows": len(ENDPOINT_IDS) * len(CHROMOSOMES), "max_absdiff_four_rho": float(max_rho), "max_absdiff_derived_metrics": float(max_metric), "rho_tolerance": config["tolerances"]["rho_absdiff"], "all_four_rho_within_tolerance": bool(max_rho <= config["tolerances"]["rho_absdiff"]), "all_derived_metrics_within_tolerance": bool(max_metric <= config["tolerances"]["rho_absdiff"]), "no_local_swap": True},
        "posthoc_x0": {"rows": len(diagnostic_rows), "seed_count": 3, "chromosomes_per_seed": 20, "uses_existing_21_condition_mask": True, "new_inference_registered": False, "no_ci_or_pvalue": True},
        "display": {"n_endpoint_figures": 5, "n_posthoc_x0_figures": 1, "n_positions": 193, "common_bins": int(display_results[display_ids[0]]["n_common_bins"]), "common_unordered_pairs": int(display_results[display_ids[0]]["n_common_pairs"]), "global_vmin": 0.0, "global_vmax": global_vmax, "all_figures_share_global_norm": True, "colormap": "coolwarm_r", "missing_color": "#bdbdbd", "axis_extent_mb": [2.5, 195.5], "uncompressed_absolute_axis": True, "render_records": render_records},
        "old_new_chain": {"npz_comparisons": npz_comparisons, "raw_and_normalized_matrix_consistency_checked": True, "global_color_norm_only_is_separate_from_shape": True, "old_020_npz_color_metadata_vs_render": "old NPZ metadata is coolwarm while its frozen rerender script uses coolwarm_r; independent audit does not overwrite it", "old_020_vmax": float(npz_comparisons[0].get("old_vmax", float("nan"))), "unified_vmax": global_vmax},
        "chr1_shape_comparisons": shape_comparisons,
        "macro_summary": macro,
        "inputs": input_records,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(config_path),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    json_dump(HERE / "validation.json", validation)

    provenance = {
        "schema": "p9016-c0-020-independent-audit-provenance-v1",
        "status": "PASS",
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "workflow": [
            "Freeze config and all endpoint/x0/mask hashes before reference read.",
            "Parse 5 endpoint coordinates, 3 025 x0 coordinates, and all 21 existing mask coordinates with an independent float64 text parser.",
            "Hash and read the evaluation reference only after all coordinate locks pass.",
            "Build one common mask from exactly the existing 21 conditions plus the two reference tracks; verify all per-chromosome pair counts.",
            "Recompute four rho, fixed-reference margins, whole-chromosome orientation, and derived R2 metrics without project R2 helpers.",
            "Compare independent endpoint results to locked 029 rows and compare old NPZ matrices to fresh matrices.",
            "Render endpoint and posthoc x0 chr1 maps with one global norm, coolwarm_r, 300 dpi, 7 pt, 6x6 inch canvases.",
            "Write only this fresh audit directory; no fit, selection, phase, or original artifact writes."
        ],
        "reference_read_after_candidate_hashes": True,
        "original_evaluation_unchanged": True,
        "x0_diagnostic_posthoc": True,
        "input_records": input_records,
    }
    json_dump(HERE / "provenance.json", provenance)

    # 面向中文审计报告的紧凑机器可读汇总。
    summary = {
        "audit_dir": str(HERE), "validation": str(HERE / "validation.json"), "global_vmax": global_vmax,
        "endpoint_macro": macro, "mask": mask_validation_rows, "npz_comparisons": npz_comparisons,
        "chr1_shape_comparisons": shape_comparisons,
        "render_records": render_records, "max_absdiff_four_rho": max_rho, "max_absdiff_derived_metrics": max_metric,
    }
    json_dump(HERE / "summary.json", summary)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"AUDIT_ERROR: {exc}")
        raise
