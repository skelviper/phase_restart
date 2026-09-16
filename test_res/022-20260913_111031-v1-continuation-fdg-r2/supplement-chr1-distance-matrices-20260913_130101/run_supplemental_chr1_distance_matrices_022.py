#!/usr/bin/env python3
"""仅根据 `r2_results.json` 重绘 022 chr1 continuation 与 rejected-FDG distance-map 补充结果。

本 evaluator 对 training inputs 只读。它哈希已写出的两个 candidate 3DG 文件和仅供评估的 reference，检查冻结的 022 R2 pairing，并只写本补充及其交付副本。坐标按 float64 文本解析；不会打开或执行 pairs、phase fields、fitting 或 FDG run。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "config.json"
BIN = 1_000_000
OFFSET = 3_000_000
CHROMOSOME = "chr1"
CHROMOSOME_LENGTH_BP = 195_471_971
POSITIONS_BP = np.arange(OFFSET, CHROMOSOME_LENGTH_BP, BIN, dtype=np.int64)
N_BINS = len(POSITIONS_BP)
N_TOTAL_PAIRS = N_BINS * (N_BINS - 1) // 2
EXPECTED_COMMON_BINS = 188
EXPECTED_COMMON_PAIRS = 17_578
COLORMAP = "coolwarm_r"
MISSING_COLOR = "#bdbdbd"
FIGURE_SIZE = (6.0, 6.0)
DPI = 300
FONT_SIZE = 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def file_record(path: Path, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} is unavailable: {path}")
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}")
    record: dict[str, Any] = {
        "path": str(path),
        "actual_sha256": actual,
        "size_bytes": int(path.stat().st_size),
    }
    if expected_sha256 is not None:
        record["expected_sha256"] = expected_sha256
    return record


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}: {path}") from exc


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path, label)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")


def load_track_expectations(path: Path) -> tuple[dict[str, tuple[int, ...]], list[str], dict[str, Any]]:
    document = read_json_object(path, "022 track map")
    rows = document.get("tracks")
    if not isinstance(rows, list) or len(rows) != 40:
        raise RuntimeError("022 track map must contain exactly 40 tracks")
    expected: dict[str, tuple[int, ...]] = {}
    order: list[str] = []
    lengths: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("022 track map row is not an object")
        track = str(row.get("track"))
        chromosome = str(row.get("chromosome_name"))
        length = row.get("length_bp")
        if track in expected or not isinstance(length, int) or length <= 0:
            raise RuntimeError(f"invalid or duplicate track-map row: {row}")
        expected[track] = tuple(range(0, length, BIN))
        order.append(track)
        lengths[track] = length
        if not re.fullmatch(r"c\d{2}[ab]", track):
            raise RuntimeError(f"unexpected native track name: {track}")
        if row.get("copy_index") not in (0, 1) or row.get("chromosome_name") is None:
            raise RuntimeError(f"invalid track-map copy/chromosome metadata: {row}")
        if not chromosome:
            raise RuntimeError(f"empty chromosome name for {track}")
    expected_order = [f"c{index:02d}{copy}" for index in range(1, 21) for copy in ("a", "b")]
    if order != expected_order:
        raise RuntimeError("022 track map order is not c01a,c01b,...,c20a,c20b")
    n_physical_beads = int(sum(len(values) for values in expected.values()))
    n_loci = n_physical_beads // 2
    if n_physical_beads != 5290 or n_loci != 2645:
        raise RuntimeError("022 track map does not describe 2645 loci / 5290 physical beads")
    metadata = {
        "schema_version": document.get("schema_version"),
        "n_tracks": len(expected),
        "n_loci": n_loci,
        "n_physical_beads": n_physical_beads,
        "track_order": order,
        "lengths_bp": lengths,
    }
    return expected, order, metadata


def load_candidate(path: Path, expected_positions: dict[str, tuple[int, ...]], label: str) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    coordinates: dict[str, dict[int, np.ndarray]] = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 5:
                raise RuntimeError(f"{label} line {line_no} has {len(fields)} fields, expected 5")
            track = fields[0]
            if track not in expected_positions:
                raise RuntimeError(f"{label} has unexpected track {track} at line {line_no}")
            try:
                position = int(fields[1])
                point = np.asarray([float(value) for value in fields[2:5]], dtype=np.float64)
            except ValueError as exc:
                raise RuntimeError(f"{label} line {line_no} is not numeric") from exc
            if position < 0 or not np.isfinite(point).all():
                raise RuntimeError(f"{label} line {line_no} is negative or non-finite")
            rows = coordinates.setdefault(track, {})
            if position in rows:
                raise RuntimeError(f"{label} duplicates {track}:{position}")
            rows[position] = point
    if set(coordinates) != set(expected_positions):
        raise RuntimeError(
            f"{label} track mismatch: missing={sorted(set(expected_positions) - set(coordinates))} "
            f"unexpected={sorted(set(coordinates) - set(expected_positions))}"
        )
    for track, positions in expected_positions.items():
        observed = set(coordinates[track])
        if observed != set(positions):
            raise RuntimeError(
                f"{label} full-grid mismatch for {track}: "
                f"missing={len(set(positions) - observed)} unexpected={len(observed - set(positions))}"
            )
    radii = [float(np.linalg.norm(point)) for rows in coordinates.values() for point in rows.values()]
    return coordinates, {
        "n_tracks": len(coordinates),
        "n_beads": int(sum(len(rows) for rows in coordinates.values())),
        "max_radius": float(max(radii)),
        "all_finite": True,
        "float_dtype": "float64",
    }


def dense_track(source: dict[str, dict[int, np.ndarray]], track: str, positions: np.ndarray) -> np.ndarray:
    rows = source.get(track)
    if rows is None:
        raise RuntimeError(f"coordinate track is unavailable: {track}")
    points = np.full((len(positions), 3), np.nan, dtype=np.float64)
    for index, position in enumerate(positions):
        value = rows.get(int(position))
        if value is not None:
            point = np.asarray(value, dtype=np.float64)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise RuntimeError(f"invalid coordinate at {track}:{position}")
            points[index] = point
    return points


def distance_matrix(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_bins = np.isfinite(points).all(axis=1)
    matrix = np.full((len(points), len(points)), np.nan, dtype=np.float64)
    indices = np.flatnonzero(finite_bins)
    if len(indices):
        values = np.asarray(points[indices], dtype=np.float64)
        delta = values[:, None, :] - values[None, :, :]
        matrix[np.ix_(indices, indices)] = np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float64))
    pair_mask = np.outer(finite_bins, finite_bins)
    np.fill_diagonal(pair_mask, False)
    return matrix, finite_bins, pair_mask


def upper_values(matrix: np.ndarray, pair_mask: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[np.triu(pair_mask, k=1)], dtype=np.float64)


def rms_scale(matrices: np.ndarray, pair_mask: np.ndarray) -> float:
    values = np.concatenate([upper_values(matrix, pair_mask) for matrix in matrices])
    if len(values) == 0 or not np.isfinite(values).all():
        raise RuntimeError("RMS scale has no finite common non-diagonal values")
    scale = float(np.sqrt(np.mean(values * values, dtype=np.float64)))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"invalid RMS scale: {scale}")
    return scale


def symmetric_with_nan(matrix: np.ndarray) -> bool:
    finite = np.isfinite(matrix)
    if not np.array_equal(finite, finite.T):
        return False
    return bool(np.allclose(matrix[finite], matrix.T[finite], rtol=0.0, atol=1e-12))


def finite_span(points: np.ndarray) -> dict[str, Any]:
    finite = points[np.isfinite(points).all(axis=1)]
    if not len(finite):
        return {"n_finite_bins": 0, "min": None, "max": None, "span": None}
    minimum = finite.min(axis=0)
    maximum = finite.max(axis=0)
    return {
        "n_finite_bins": int(len(finite)),
        "min": [float(value) for value in minimum],
        "max": [float(value) for value in maximum],
        "span": [float(value) for value in (maximum - minimum)],
    }


def make_candidate_result(candidate_id: str, candidate: dict[str, dict[int, np.ndarray]], reference: dict[str, dict[int, np.ndarray]], titles: list[str]) -> dict[str, Any]:
    panel_tracks = ["chr1(mat)", "chr1(pat)", "c01b", "c01a"]
    panel_labels = ["reference_maternal", "reference_paternal", f"{candidate_id}_copy_B_maternal", f"{candidate_id}_copy_A_paternal"]
    sources = [reference, reference, candidate, candidate]
    points = np.stack(
        [dense_track(source, track, POSITIONS_BP) for source, track in zip(sources, panel_tracks)], axis=0
    )
    raw_matrices = []
    track_bin_mask = []
    track_pair_mask = []
    for panel in points:
        matrix, bin_mask, pair_mask = distance_matrix(panel)
        raw_matrices.append(matrix)
        track_bin_mask.append(bin_mask)
        track_pair_mask.append(pair_mask)
    raw_matrices_array = np.stack(raw_matrices, axis=0)
    track_bin_mask_array = np.stack(track_bin_mask, axis=0)
    track_pair_mask_array = np.stack(track_pair_mask, axis=0)
    common_bin_mask = np.all(track_bin_mask_array, axis=0)
    common_pair_mask = np.outer(common_bin_mask, common_bin_mask)
    np.fill_diagonal(common_pair_mask, False)
    common_raw = np.array(raw_matrices_array, copy=True)
    common_raw[:, ~common_pair_mask] = np.nan
    diagonal = np.diag_indices(N_BINS)
    for panel_index in range(4):
        common_raw[panel_index, diagonal[0], diagonal[1]] = np.where(common_bin_mask, 0.0, np.nan)
    reference_scale = rms_scale(common_raw[:2], common_pair_mask)
    candidate_scale = rms_scale(common_raw[2:], common_pair_mask)
    scales = np.asarray([reference_scale, candidate_scale], dtype=np.float64)
    panel_scales = np.asarray([reference_scale, reference_scale, candidate_scale, candidate_scale], dtype=np.float64)
    normalized = common_raw / panel_scales[:, None, None]
    for panel_index in range(4):
        normalized[panel_index, diagonal[0], diagonal[1]] = np.where(common_bin_mask, 0.0, np.nan)
    common_pair_count = int(np.triu(common_pair_mask, k=1).sum())
    if int(common_bin_mask.sum()) != EXPECTED_COMMON_BINS or common_pair_count != EXPECTED_COMMON_PAIRS:
        raise RuntimeError(
            f"{candidate_id} common mask mismatch: bins={int(common_bin_mask.sum())}, pairs={common_pair_count}"
        )
    return {
        "candidate_id": candidate_id,
        "panel_tracks": panel_tracks,
        "panel_labels": panel_labels,
        "titles": titles,
        "points": points,
        "raw_matrices": raw_matrices_array,
        "common_raw_matrices": common_raw,
        "normalized_matrices": normalized,
        "track_bin_finite_mask": track_bin_mask_array,
        "track_pair_finite_mask": track_pair_mask_array,
        "common_bin_mask": common_bin_mask,
        "common_pair_mask": common_pair_mask,
        "scales": scales,
        "panel_scales": panel_scales,
        "mask": {
            "n_total_bins": N_BINS,
            "n_common_bins": int(common_bin_mask.sum()),
            "n_total_non_diagonal_pairs": N_TOTAL_PAIRS,
            "n_common_non_diagonal_pairs": common_pair_count,
            "missing_bins_by_track": {
                label: int((~track_bin_mask_array[index]).sum()) for index, label in enumerate(panel_labels)
            },
            "missing_pair_count_by_track": {
                label: int(N_TOTAL_PAIRS - np.triu(track_pair_mask_array[index], k=1).sum())
                for index, label in enumerate(panel_labels)
            },
            "common_pair_mask_symmetric": bool(np.array_equal(common_pair_mask, common_pair_mask.T)),
            "common_pair_mask_diagonal_false": bool(not np.any(np.diag(common_pair_mask))),
        },
        "coordinate_summary": {
            "reference_chr1_mat": finite_span(points[0]),
            "reference_chr1_pat": finite_span(points[1]),
            "candidate_c01b": finite_span(points[2]),
            "candidate_c01a": finite_span(points[3]),
        },
    }


def validate_result(result: dict[str, Any], global_vmax: float) -> dict[str, Any]:
    raw = result["raw_matrices"]
    common_raw = result["common_raw_matrices"]
    normalized = result["normalized_matrices"]
    bin_mask = result["track_bin_finite_mask"]
    pair_mask = result["common_pair_mask"]
    positions = POSITIONS_BP
    scales = result["scales"]
    if raw.shape != (4, N_BINS, N_BINS) or common_raw.shape != raw.shape or normalized.shape != raw.shape:
        raise RuntimeError(f"{result['candidate_id']} matrix shape mismatch")
    if bin_mask.shape != (4, N_BINS) or pair_mask.shape != (N_BINS, N_BINS):
        raise RuntimeError(f"{result['candidate_id']} mask shape mismatch")
    expected_pair_mask = np.outer(result["common_bin_mask"], result["common_bin_mask"])
    np.fill_diagonal(expected_pair_mask, False)
    if not np.array_equal(pair_mask, expected_pair_mask):
        raise RuntimeError(f"{result['candidate_id']} pair mask is not derived from common bin mask")
    if not all(symmetric_with_nan(matrix) for matrix in raw):
        raise RuntimeError(f"{result['candidate_id']} raw matrices are not symmetric")
    if not all(symmetric_with_nan(matrix) for matrix in common_raw):
        raise RuntimeError(f"{result['candidate_id']} common raw matrices are not symmetric")
    if not all(symmetric_with_nan(matrix) for matrix in normalized):
        raise RuntimeError(f"{result['candidate_id']} normalized matrices are not symmetric")
    if not np.array_equal(pair_mask, pair_mask.T) or np.any(np.diag(pair_mask)):
        raise RuntimeError(f"{result['candidate_id']} common pair mask is invalid")
    diagonal = np.diag_indices(N_BINS)
    common_bins = result["common_bin_mask"]
    for panel_index in range(4):
        if not np.all(normalized[panel_index, diagonal[0][common_bins], diagonal[1][common_bins]] == 0.0):
            raise RuntimeError(f"{result['candidate_id']} normalized diagonal is not exactly zero")
        if np.any(np.isfinite(normalized[panel_index, diagonal[0][~common_bins], diagonal[1][~common_bins]])):
            raise RuntimeError(f"{result['candidate_id']} missing-bin diagonal is finite")
        allowed = pair_mask | (np.eye(N_BINS, dtype=bool) & common_bins[:, None])
        if np.any(np.isfinite(normalized[panel_index]) & ~allowed):
            raise RuntimeError(f"{result['candidate_id']} normalized matrix has values outside common mask")
    finite = np.isfinite(normalized)
    finite_values = normalized[finite]
    if len(finite_values) == 0 or np.min(finite_values) < -1e-12 or np.max(finite_values) > global_vmax + 1e-12:
        raise RuntimeError(f"{result['candidate_id']} normalized values exceed global color range")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise RuntimeError(f"{result['candidate_id']} RMS scales are not positive finite")
    if not np.array_equal(positions, np.arange(OFFSET, CHROMOSOME_LENGTH_BP, BIN, dtype=np.int64)):
        raise RuntimeError("chr1 position grid changed")
    recomputed_reference = rms_scale(common_raw[:2], pair_mask)
    recomputed_candidate = rms_scale(common_raw[2:], pair_mask)
    if not np.allclose(scales, [recomputed_reference, recomputed_candidate], rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{result['candidate_id']} RMS normalization is not reproducible")
    return {
        "raw_matrices": list(raw.shape),
        "common_raw_matrices": list(common_raw.shape),
        "normalized_matrices": list(normalized.shape),
        "track_bin_finite_mask": list(bin_mask.shape),
        "track_pair_finite_mask": list(result["track_pair_finite_mask"].shape),
        "common_bin_mask": list(common_bins.shape),
        "common_pair_mask": list(pair_mask.shape),
        "positions_bp": list(positions.shape),
        "raw_symmetric": True,
        "common_raw_symmetric": True,
        "normalized_symmetric": True,
        "diagonal_zero_on_common_bins": True,
        "diagonal_excluded_from_pair_mask": True,
        "diagonal_excluded_from_rms": True,
        "n_common_bins": int(common_bins.sum()),
        "n_common_unordered_non_diagonal_pairs": int(np.triu(pair_mask, k=1).sum()),
        "n_total_unordered_non_diagonal_pairs": N_TOTAL_PAIRS,
        "normalized_min": float(np.min(finite_values)),
        "normalized_max": float(np.max(finite_values)),
        "rms_recomputed": True,
        "global_color_vmin": 0.0,
        "global_color_vmax": float(global_vmax),
        "global_color_clip": False,
    }


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"invalid PNG header: {path}")
    return tuple(int(value) for value in __import__("struct").unpack(">II", header[16:24]))


def pdf_page_size(path: Path) -> tuple[float, float]:
    matches = re.findall(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", path.read_bytes())
    if not matches:
        raise RuntimeError(f"cannot parse PDF MediaBox: {path}")
    return float(matches[0][0]), float(matches[0][1])


def render_figure(normalized: np.ndarray, titles: list[str], figure_title: str, caption: str, vmax: float, png_path: Path, pdf_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.linewidth": 0.6,
    })
    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad(MISSING_COLOR)
    if cmap.name != COLORMAP:
        raise RuntimeError(f"unexpected colormap: {cmap.name}")
    norm = Normalize(vmin=0.0, vmax=vmax, clip=False)
    positions_mb = POSITIONS_BP.astype(np.float64) / 1_000_000.0
    half_bin_mb = BIN / 2_000_000.0
    extent = (
        float(positions_mb[0] - half_bin_mb),
        float(positions_mb[-1] + half_bin_mb),
        float(positions_mb[0] - half_bin_mb),
        float(positions_mb[-1] + half_bin_mb),
    )
    ticks = np.asarray([3, 50, 100, 150, 195], dtype=float)
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    grid = fig.add_gridspec(
        2, 3, width_ratios=[1.0, 1.0, 0.055], height_ratios=[1.0, 1.0],
        left=0.105, right=0.91, bottom=0.14, top=0.87, wspace=0.24, hspace=0.36,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    if len(titles) != 4:
        raise RuntimeError("four panel titles are required")
    images = []
    for axis, matrix, title in zip(axes, normalized, titles):
        image = axis.imshow(matrix, origin="lower", interpolation="nearest", aspect="equal", extent=extent, cmap=cmap, norm=norm)
        images.append(image)
        axis.set_title(title, pad=4)
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.tick_params(width=0.6, length=2.5, pad=2)
        axis.set_xlabel("Genomic position (Mb)")
        axis.set_ylabel("Genomic position (Mb)")
    colorbar_axis = fig.add_subplot(grid[:, 2])
    colorbar = fig.colorbar(images[0], cax=colorbar_axis)
    colorbar.set_label("normalized 3D Euclidean distance", fontsize=FONT_SIZE, labelpad=4)
    colorbar.ax.tick_params(labelsize=FONT_SIZE, width=0.6, length=2.5)
    fig.suptitle(figure_title, fontsize=FONT_SIZE, y=0.925)
    fig.text(0.5, 0.045, caption, ha="center", va="center", fontsize=FONT_SIZE)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=DPI, format="png")
    fig.savefig(pdf_path, dpi=DPI, format="pdf")
    plt.close(fig)
    dimensions = png_dimensions(png_path)
    page = pdf_page_size(pdf_path)
    if dimensions != (1800, 1800):
        raise RuntimeError(f"unexpected PNG dimensions for {png_path}: {dimensions}")
    if not np.allclose(page, (432.0, 432.0), rtol=0.0, atol=1e-6):
        raise RuntimeError(f"unexpected PDF page dimensions for {pdf_path}: {page}")
    return {
        "colormap": COLORMAP,
        "missing_color": MISSING_COLOR,
        "vmin": 0.0,
        "vmax": float(vmax),
        "clip": False,
        "figure_size_inches": list(FIGURE_SIZE),
        "dpi": DPI,
        "font_size_pt": FONT_SIZE,
        "png_dimensions_px": list(dimensions),
        "pdf_page_size_points": list(page),
        "axis_extent_mb": list(extent),
        "titles": titles,
        "shared_colorbar": True,
    }


def save_npz(path: Path, result: dict[str, Any], global_vmax: float, pairing: dict[str, Any], metadata: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        raw_matrices=result["raw_matrices"],
        common_raw_matrices=result["common_raw_matrices"],
        normalized_matrices=result["normalized_matrices"],
        track_bin_finite_mask=result["track_bin_finite_mask"],
        track_pair_finite_mask=result["track_pair_finite_mask"],
        common_bin_mask=result["common_bin_mask"],
        common_pair_mask=result["common_pair_mask"],
        positions_bp=POSITIONS_BP,
        panel_tracks=np.asarray(result["panel_tracks"]),
        panel_labels=np.asarray(result["panel_labels"]),
        scales=result["scales"],
        panel_scales=result["panel_scales"],
        color_norm=np.asarray([0.0, global_vmax], dtype=np.float64),
        coordinate_units=np.asarray("candidate_dimensionless_R1_reference_arbitrary_units"),
        colormap=np.asarray(COLORMAP),
        pairing_json=np.asarray(json.dumps(pairing, sort_keys=True)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def verify_r2(config: dict[str, Any]) -> dict[str, Any]:
    r2_cfg = config["r2"]
    json_path = resolve_path(r2_cfg["json_path"])
    tsv_path = resolve_path(r2_cfg["tsv_path"])
    json_record = file_record(json_path, "022 R2 JSON")
    tsv_record = file_record(tsv_path, "022 R2 TSV")
    document = read_json_object(json_path, "022 R2 JSON")
    chromosomes = document.get("chromosomes")
    if not isinstance(chromosomes, list):
        raise RuntimeError("022 R2 JSON has no chromosome list")
    chr1 = next((row for row in chromosomes if isinstance(row, dict) and row.get("chromosome") == CHROMOSOME), None)
    if not isinstance(chr1, dict):
        raise RuntimeError("022 R2 JSON has no chr1 record")
    mask = chr1.get("mask")
    if not isinstance(mask, dict):
        raise RuntimeError("022 R2 chr1 mask record is unavailable")
    expected_total = int(r2_cfg["expected_n_total_non_diagonal_pairs"])
    expected_pairs = int(r2_cfg["expected_n_common_pairs"])
    if int(mask.get("n_bins")) != N_BINS or int(mask.get("n_total_non_diagonal_pairs")) != expected_total or int(mask.get("n_common_pairs")) != expected_pairs:
        raise RuntimeError("022 R2 chr1 mask counts disagree with frozen config")
    conditions = chr1.get("conditions")
    if not isinstance(conditions, dict):
        raise RuntimeError("022 R2 chr1 conditions are unavailable")
    selected: dict[str, Any] = {}
    for condition_id in ("v1_continuation", "fdg_proposal"):
        row = conditions.get(condition_id)
        if not isinstance(row, dict):
            raise RuntimeError(f"022 R2 condition missing: {condition_id}")
        if row.get("pairing") != r2_cfg["expected_pairing"] or row.get("orientation") != r2_cfg["expected_orientation"]:
            raise RuntimeError(f"022 R2 pairing changed for {condition_id}: {row.get('pairing')}/{row.get('orientation')}")
        if row.get("geometry_status") != "cross_maximum" or row.get("track_names") != ["c01a", "c01b"]:
            raise RuntimeError(f"022 R2 geometry record is not the locked cross best-swap for {condition_id}")
        if int(row.get("n_common_pairs")) != expected_pairs or int(row.get("n_total_non_diagonal_pairs")) != expected_total:
            raise RuntimeError(f"022 R2 pair counts changed for {condition_id}")
        selected[condition_id] = {
            "pairing": row["pairing"],
            "orientation": row["orientation"],
            "geometry_status": row["geometry_status"],
            "direct": float(row["direct"]),
            "cross": float(row["cross"]),
            "matched": float(row["matched"]),
            "other": float(row["other"]),
            "n_common_pairs": int(row["n_common_pairs"]),
            "n_total_non_diagonal_pairs": int(row["n_total_non_diagonal_pairs"]),
        }
    with tsv_path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError("022 R2 TSV has no header")
        rows = list(reader)
    tsv_selected: dict[str, Any] = {}
    for condition_id in ("v1_continuation", "fdg_proposal"):
        matches = [row for row in rows if row.get("chromosome") == CHROMOSOME and row.get("condition_id") == condition_id]
        if len(matches) != 1:
            raise RuntimeError(f"022 R2 TSV has {len(matches)} chr1 rows for {condition_id}")
        row = matches[0]
        if row.get("pairing") != r2_cfg["expected_pairing"] or row.get("orientation") != r2_cfg["expected_orientation"]:
            raise RuntimeError(f"022 R2 TSV pairing changed for {condition_id}")
        if int(row["n_common_pairs"]) != expected_pairs or int(row["n_total_non_diagonal_pairs"]) != expected_total:
            raise RuntimeError(f"022 R2 TSV pair counts changed for {condition_id}")
        tsv_selected[condition_id] = {
            "pairing": row["pairing"],
            "orientation": row["orientation"],
            "geometry_status": row["geometry_status"],
            "n_common_pairs": int(row["n_common_pairs"]),
            "n_total_non_diagonal_pairs": int(row["n_total_non_diagonal_pairs"]),
            "display_name": row.get("display_name"),
        }
    return {
        "json": json_record,
        "tsv": tsv_record,
        "chr1_mask": {
            "n_bins": int(mask["n_bins"]),
            "n_total_non_diagonal_pairs": int(mask["n_total_non_diagonal_pairs"]),
            "n_common_pairs": int(mask["n_common_pairs"]),
        },
        "json_selected": selected,
        "tsv_selected_via_csv_dictreader": tsv_selected,
        "pairing_policy_applied": "cross = actual A/c01a -> reference chr1(pat), actual B/c01b -> reference chr1(mat)",
    }


def verify_source_identity(config: dict[str, Any]) -> dict[str, Any]:
    candidates_cfg = config["candidates"]
    continuation_path = resolve_path(candidates_cfg["v1_continuation"]["path"])
    fdg_path = resolve_path(candidates_cfg["fdg_proposal"]["path"])
    reference_path = resolve_path(config["reference"]["path"])
    records = {
        "reference": file_record(reference_path, "reference 3DG", config["reference"]["sha256"]),
        "v1_continuation": file_record(continuation_path, "V1 continuation 3DG", candidates_cfg["v1_continuation"]["sha256"]),
        "fdg_proposal": file_record(fdg_path, "FDG mapped proposal 3DG", candidates_cfg["fdg_proposal"]["sha256"]),
    }
    decision_path = resolve_path(config["fdg_decision"]["path"])
    decision = read_json_object(decision_path, "FDG acceptance decision")
    if decision.get("status") != config["fdg_decision"]["expected_status"]:
        raise RuntimeError("FDG acceptance decision status changed")
    artifacts = decision.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("FDG acceptance decision has no artifact map")
    mapped_path = resolve_path(config["fdg_decision"]["mapped_proposal_path"])
    accepted_path = resolve_path(config["fdg_decision"]["accepted_trial_path"])
    if fdg_path != mapped_path or records["fdg_proposal"]["actual_sha256"] != config["fdg_decision"]["mapped_proposal_sha256"]:
        raise RuntimeError("FDG candidate is not the frozen mapped_proposal.3dg")
    mapped_sha = sha256_file(mapped_path)
    accepted_sha = sha256_file(accepted_path)
    if mapped_sha == accepted_sha or accepted_sha != config["fdg_decision"]["accepted_trial_sha256"]:
        raise RuntimeError("FDG mapped proposal and accepted anchor duplicate identity is inconsistent")
    if artifacts.get("mapped_3dg") != str(mapped_path) or artifacts.get("mapped_3dg_sha256") != mapped_sha:
        raise RuntimeError("FDG decision artifact map disagrees with mapped proposal")
    if artifacts.get("accepted_3dg") != str(accepted_path) or artifacts.get("accepted_3dg_sha256") != accepted_sha:
        raise RuntimeError("FDG decision artifact map disagrees with accepted trial")
    continuation_summary_path = resolve_path(config["continuation_summary"]["path"])
    continuation_summary = read_json_object(continuation_summary_path, "continuation summary")
    if continuation_summary.get("status") != config["continuation_summary"]["expected_status"]:
        raise RuntimeError("continuation summary status changed")
    termination = continuation_summary.get("termination", {})
    if termination.get("status") != config["continuation_summary"]["expected_termination_status"]:
        raise RuntimeError("continuation termination status changed")
    if continuation_summary.get("iterations", {}).get("cumulative_end") != candidates_cfg["v1_continuation"]["cumulative_iterations"]:
        raise RuntimeError("continuation cumulative iteration count changed")
    if continuation_summary.get("final_coordinates", {}).get("sha256") != records["v1_continuation"]["actual_sha256"]:
        raise RuntimeError("continuation summary coordinate hash disagrees with candidate")
    ancillary = {
        "fdg_decision": file_record(decision_path, "FDG acceptance decision"),
        "accepted_trial": file_record(accepted_path, "FDG accepted trial duplicate"),
        "continuation_summary": file_record(continuation_summary_path, "continuation summary"),
    }
    return {
        "candidates": records,
        "fdg_decision": ancillary["fdg_decision"],
        "fdg_candidate_identity": {
            "actual_path": str(fdg_path),
            "mapped_proposal_path": str(mapped_path),
            "mapped_proposal_sha256": mapped_sha,
            "accepted_trial_path": str(accepted_path),
            "accepted_trial_sha256": accepted_sha,
            "accepted_trial_is_anchor_duplicate": True,
            "decision_status": decision["status"],
        },
        "continuation_summary": ancillary["continuation_summary"],
        "continuation_status": {
            "endpoint_status": candidates_cfg["v1_continuation"]["endpoint_status"],
            "added_iterations": int(candidates_cfg["v1_continuation"]["added_iterations"]),
            "cumulative_iterations": int(candidates_cfg["v1_continuation"]["cumulative_iterations"]),
            "termination_status": termination["status"],
        },
        "ancillary_hashes": ancillary,
    }


def render_and_save(config: dict[str, Any], results: dict[str, dict[str, Any]], global_vmax: float, pairing: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    output_cfg = config["outputs"]
    output_paths = {
        "v1_continuation": {
            "npz": HERE / output_cfg["continuation_npz"],
            "png": HERE / output_cfg["continuation_png"],
            "pdf": HERE / output_cfg["continuation_pdf"],
        },
        "fdg_proposal": {
            "npz": HERE / output_cfg["fdg_npz"],
            "png": HERE / output_cfg["fdg_png"],
            "pdf": HERE / output_cfg["fdg_pdf"],
        },
    }
    delivery_paths = {
        "v1_continuation": {
            "png": ROOT / "deliverables" / output_cfg["continuation_png"],
            "pdf": ROOT / "deliverables" / output_cfg["continuation_pdf"],
        },
        "fdg_proposal": {
            "png": ROOT / "deliverables" / output_cfg["fdg_png"],
            "pdf": ROOT / "deliverables" / output_cfg["fdg_pdf"],
        },
    }
    targets = [path for group in output_paths.values() for path in group.values()] + [path for group in delivery_paths.values() for path in group.values()]
    for path in targets:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing 022 supplement/delivery artifact: {path}")
    artifact_records: dict[str, Any] = {}
    render_records: dict[str, Any] = {}
    for candidate_id, result in results.items():
        metadata = {"candidate_id": candidate_id, "sample": "P9016", "chromosome": CHROMOSOME, "panel_tracks": result["panel_tracks"], "panel_labels": result["panel_labels"]}
        save_npz(output_paths[candidate_id]["npz"], result, global_vmax, pairing[candidate_id], metadata)
        figure_title = config["render"]["figure_titles"][candidate_id]
        caption = "Shared pooled RMS per structure; common colorbar across continuation and FDG; red=near, blue=far; grey=missing"
        render_records[candidate_id] = render_figure(
            result["normalized_matrices"], result["titles"], figure_title, caption, global_vmax,
            output_paths[candidate_id]["png"], output_paths[candidate_id]["pdf"],
        )
        delivery_paths[candidate_id]["png"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_paths[candidate_id]["png"], delivery_paths[candidate_id]["png"])
        shutil.copy2(output_paths[candidate_id]["pdf"], delivery_paths[candidate_id]["pdf"])
        artifact_records[candidate_id] = {
            "npz": file_record(output_paths[candidate_id]["npz"], f"{candidate_id} NPZ"),
            "png": file_record(output_paths[candidate_id]["png"], f"{candidate_id} PNG"),
            "pdf": file_record(output_paths[candidate_id]["pdf"], f"{candidate_id} PDF"),
            "delivery_png": file_record(delivery_paths[candidate_id]["png"], f"{candidate_id} delivery PNG"),
            "delivery_pdf": file_record(delivery_paths[candidate_id]["pdf"], f"{candidate_id} delivery PDF"),
        }
    return artifact_records, render_records


def write_readme(path: Path, config: dict[str, Any], validation: dict[str, Any], provenance_path: Path) -> None:
    source = validation["source_identity"]
    mask = validation["mask"]["by_candidate"]["v1_continuation"]
    lines = [
        "# P9016 chr1：022 延续 / FDG 距离矩阵",
        "",
        "**状态：** `PASS`。本目录是 022 的全新、仅供评价的补充；未重训练、未运行 FDG、未读取带 phase 的 pairs、未改变 022 接受决定、R2 数值、旧 020/softall 图件或任何原始输入。",
        "",
        "## 输入与身份",
        "",
        f"- 参考结构仅作评价：`{source['candidates']['reference']['path']}`，SHA256 `{source['candidates']['reference']['actual_sha256']}`；reference 轨迹字面值为 `chr1(mat)` / `chr1(pat)`，坐标保留任意单位。",
        f"- V1 continuation：`{source['candidates']['v1_continuation']['path']}`，SHA256 `{source['candidates']['v1_continuation']['actual_sha256']}`；新增 `{source['continuation_status']['added_iterations']}` 步，累计 `{source['continuation_status']['cumulative_iterations']}` 步，状态 `{source['continuation_status']['endpoint_status']}`。",
        f"- FDG 实际图件输入固定为 mapped full proposal：`{source['candidates']['fdg_proposal']['path']}`，SHA256 `{source['candidates']['fdg_proposal']['actual_sha256']}`；native 1000 步，状态 `{source['fdg_candidate_identity']['decision_status']}`。",
        f"- `accepted_trial.3dg` 明确为 anchor duplicate（SHA256 `{source['fdg_candidate_identity']['accepted_trial_sha256']}`），与 mapped proposal 不同；没有用 accepted trial 代替 proposal。原决定文件：`{source['fdg_decision']['path']}`。",
        "",
        "## 网格、配对与 mask",
        "",
        f"- P9016 单细胞，仅 chr1；1 Mb OFF=3 Mb，starts `3..195 Mb`，共 `{validation['grid']['n_bins']}` 个 bins，轴为绝对基因组位置 (Mb)。",
        f"- 四轨共同有限 mask：`{mask['n_common_bins']}/{mask['n_total_bins']}` bins、`{mask['n_common_non_diagonal_pairs']}` 无序非对角距离对；全网格非对角 pairs `{mask['n_total_non_diagonal_pairs']}`。对角线显示 0 但排除 RMS/统计；共同 mask 外位置以 `{MISSING_COLOR}` 灰色显示。",
        f"- 缺失 bin（两图相同）：`{json.dumps(mask['missing_bins_by_track'], ensure_ascii=False, sort_keys=True)}`；缺失距离对由坐标 mask 直接产生，不造点、不补零、不压缩基因组轴。",
        "- 配对严格复用 022 R2 chr1 四 rho 的几何最佳交换：`cross/swapped`。实际 A/c01a -> reference `chr1(pat)`；实际 B/c01b -> reference `chr1(mat)`。配对不是按图面观感重选，也不代表亲本身份。",
        "",
        "## 尺度与颜色",
        "",
        f"- 每张图的 reference 两轨 pooled RMS 及 candidate 两 copy pooled RMS 分别记录在 `validation.json`；本次具体值：continuation `{validation['candidates']['v1_continuation']['scales']}`，FDG `{validation['candidates']['fdg_proposal']['scales']}`。reference 使用任意坐标单位，candidate 使用无量纲 R1；不称 reference 为 R1 归一化。",
        f"- 两张图共享同一 Normalize：`vmin=0, vmax={validation['global_color']['vmax']:.14g}, clip=False`，取 reference+continuation+FDG 所有显示 finite normalized distances 的共同最大值；colormap `{COLORMAP}`，红=近、蓝=远、缺失=`{MISSING_COLOR}`。该共享色标与旧 020/softall 图的独立色标不保证一致。",
        "",
        "## 输出与验证",
        "",
        f"- continuation 图：`{validation['artifacts']['v1_continuation']['png']['path']}` / PDF；交付副本：`{validation['artifacts']['v1_continuation']['delivery_png']['path']}` / PDF。",
        f"- FDG proposal 图：`{validation['artifacts']['fdg_proposal']['png']['path']}` / PDF；交付副本：`{validation['artifacts']['fdg_proposal']['delivery_png']['path']}` / PDF。",
        f"- 两份 raw/common-raw/normalized matrices NPZ、config、全新 gate、机器可读验证分别在本目录；provenance：`{provenance_path}`。PNG 为 `{validation['render']['v1_continuation']['png_dimensions_px'][0]}x{validation['render']['v1_continuation']['png_dimensions_px'][1]}` px、300 dpi；PDF 页面 `{validation['render']['v1_continuation']['pdf_page_size_points']} pt`，figure 6x6 英寸。",
        "- 已验证：float64 坐标解析、矩阵 symmetry、zero diagonal、共同 mask/missing bins/pairs、pooled RMS normalization、两图 identical global color scale、grid/mapping、PNG/PDF 尺寸；本补充不提出 L2 新结论。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    config = read_json_object(config_path, "supplement config")
    if config.get("status") != "config_frozen_before_compute":
        raise RuntimeError("supplement config is not frozen before compute")
    grid_cfg = config["grid"]
    if (grid_cfg.get("chromosome"), grid_cfg.get("chromosome_length_bp"), grid_cfg.get("bin_size_bp"), grid_cfg.get("offset_bp"), grid_cfg.get("expected_n_bins")) != (CHROMOSOME, CHROMOSOME_LENGTH_BP, BIN, OFFSET, N_BINS):
        raise RuntimeError("supplement grid differs from frozen chr1 grid")
    if config["distance_and_norm"].get("colormap") != COLORMAP or config["distance_and_norm"].get("missing_color") != MISSING_COLOR:
        raise RuntimeError("supplement color policy changed")
    track_map_path = resolve_path(config["track_map"]["path"])
    track_map_record = file_record(track_map_path, "022 track map")
    expected_positions, track_order, track_metadata = load_track_expectations(track_map_path)
    if track_metadata["n_tracks"] != config["track_map"]["expected_tracks"] or track_metadata["n_physical_beads"] != config["track_map"]["expected_physical_beads"]:
        raise RuntimeError("022 track map inventory disagrees with frozen config")
    source_identity = verify_source_identity(config)
    r2_identity = verify_r2(config)
    template_records = {}
    for key, value in config["template_reuse"].items():
        template_records[key] = file_record(resolve_path(value), f"reused template {key}")
    # 只有 source identity 检查完成后才导入项目 gate/reference readers。
    sys.path.insert(0, str(ROOT))
    from pr.gate import EvalGate
    from pr import ref3dg

    gate_path = HERE / config["outputs"]["gate"]
    if gate_path.exists():
        raise FileExistsError(f"refusing to overwrite existing fresh gate: {gate_path}")
    gate = EvalGate(str(gate_path))
    continuation_path = resolve_path(config["candidates"]["v1_continuation"]["path"])
    fdg_path = resolve_path(config["candidates"]["fdg_proposal"]["path"])
    reference_path = resolve_path(config["reference"]["path"])
    if gate.register("022-supplement", "candidate:v1_continuation", str(continuation_path)) != config["candidates"]["v1_continuation"]["sha256"]:
        raise RuntimeError("fresh gate continuation digest mismatch")
    if gate.register("022-supplement", "candidate:fdg_proposal_mapped", str(fdg_path)) != config["candidates"]["fdg_proposal"]["sha256"]:
        raise RuntimeError("fresh gate FDG mapped proposal digest mismatch")
    if gate.register("022-supplement", "reference:P9016.1m.3dg.gz", str(reference_path)) != config["reference"]["sha256"]:
        raise RuntimeError("fresh gate reference digest mismatch")
    gate.arm(ref3dg.STAGE)
    gate.require(ref3dg.STAGE)
    continuation, continuation_inventory = load_candidate(continuation_path, expected_positions, "V1 continuation")
    fdg, fdg_inventory = load_candidate(fdg_path, expected_positions, "FDG mapped proposal")
    reference = ref3dg.load_reference(gate, path=str(reference_path))
    if not isinstance(reference, dict) or "chr1(mat)" not in reference or "chr1(pat)" not in reference:
        raise RuntimeError("reference lacks literal chr1(mat)/chr1(pat) tracks")
    pairing = {
        "v1_continuation": {
            "r2_pairing": r2_identity["json_selected"]["v1_continuation"]["pairing"],
            "r2_orientation": r2_identity["json_selected"]["v1_continuation"]["orientation"],
            "reference_maternal": "chr1(mat)", "reference_paternal": "chr1(pat)",
            "candidate_copy_A": "c01a -> chr1(pat)", "candidate_copy_B": "c01b -> chr1(mat)",
        },
        "fdg_proposal": {
            "r2_pairing": r2_identity["json_selected"]["fdg_proposal"]["pairing"],
            "r2_orientation": r2_identity["json_selected"]["fdg_proposal"]["orientation"],
            "reference_maternal": "chr1(mat)", "reference_paternal": "chr1(pat)",
            "candidate_copy_A": "c01a -> chr1(pat)", "candidate_copy_B": "c01b -> chr1(mat)",
        },
    }
    results = {
        "v1_continuation": make_candidate_result(
            "v1_continuation", continuation, reference, config["render"]["titles"]["v1_continuation"]
        ),
        "fdg_proposal": make_candidate_result(
            "fdg_proposal", fdg, reference, config["render"]["titles"]["fdg_proposal"]
        ),
    }
    all_finite_normalized = np.concatenate([result["normalized_matrices"][np.isfinite(result["normalized_matrices"])] for result in results.values()])
    global_vmax = float(np.max(all_finite_normalized))
    if not math.isfinite(global_vmax) or global_vmax <= 0:
        raise RuntimeError(f"invalid global normalized color maximum: {global_vmax}")
    shape_validation = {candidate_id: validate_result(result, global_vmax) for candidate_id, result in results.items()}
    # 两个 panel 必须使用完全相同的有限归一化端点。
    if not all(np.isfinite(result["normalized_matrices"]).any() for result in results.values()):
        raise RuntimeError("one candidate has no finite normalized values")
    artifact_records, render_records = render_and_save(config, results, global_vmax, pairing)
    validation_path = HERE / config["outputs"]["validation"]
    provenance_path = HERE / config["outputs"]["provenance"]
    readme_path = HERE / config["outputs"]["readme"]
    for path in (validation_path, provenance_path, readme_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite fresh metadata artifact: {path}")
    validation = {
        "schema": "supplemental_chr1_distance_matrices_022.validation.v1",
        "status": "PASS",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": file_record(config_path.resolve(), "supplement config"),
        "script": file_record(Path(__file__).resolve(), "supplement script"),
        "scope_guard": {
            "training_or_selection": False,
            "fdg_run": False,
            "phase_payload_opened": False,
            "pairs_payload_opened": False,
            "reference_loaded_after_all_coordinate_hashes": True,
            "existing_020_and_softall_artifacts_modified": False,
        },
        "template_reuse": template_records,
        "source_identity": source_identity,
        "r2_identity": r2_identity,
        "track_map": {"record": track_map_record, "metadata": track_metadata},
        "gate": {
            "path": str(gate_path.resolve()),
            "sha256": sha256_file(gate_path),
            "entries": gate.entries,
            "armed_stage": ref3dg.STAGE,
            "all_coordinate_files_registered_before_reference_load": True,
        },
        "grid": {
            "chromosome": CHROMOSOME,
            "chromosome_length_bp": CHROMOSOME_LENGTH_BP,
            "bin_size_bp": BIN,
            "offset_bp": OFFSET,
            "n_bins": N_BINS,
            "positions_bp_first_last": [int(POSITIONS_BP[0]), int(POSITIONS_BP[-1])],
            "absolute_mb_axes": True,
            "position_rule": "range(offset_bp, chromosome_length_bp, bin_size_bp)",
        },
        "pairing": pairing,
        "mask": {
            "n_total_bins": N_BINS,
            "n_common_bins": EXPECTED_COMMON_BINS,
            "n_total_non_diagonal_pairs": N_TOTAL_PAIRS,
            "n_common_non_diagonal_pairs": EXPECTED_COMMON_PAIRS,
            "same_across_candidates": True,
            "by_candidate": {candidate_id: result["mask"] for candidate_id, result in results.items()},
        },
        "candidates": {
            "v1_continuation": {
                "inventory": continuation_inventory,
                "coordinate_summary": results["v1_continuation"]["coordinate_summary"],
                "scales": {
                    "reference_pooled_rms": float(results["v1_continuation"]["scales"][0]),
                    "candidate_pooled_rms": float(results["v1_continuation"]["scales"][1]),
                    "panel_scales": results["v1_continuation"]["panel_scales"].tolist(),
                    "definition": "pooled RMS over the two reference panels or two candidate copies on common unordered non-diagonal pairs",
                },
                "shape_validation": shape_validation["v1_continuation"],
            },
            "fdg_proposal": {
                "inventory": fdg_inventory,
                "coordinate_summary": results["fdg_proposal"]["coordinate_summary"],
                "scales": {
                    "reference_pooled_rms": float(results["fdg_proposal"]["scales"][0]),
                    "candidate_pooled_rms": float(results["fdg_proposal"]["scales"][1]),
                    "panel_scales": results["fdg_proposal"]["panel_scales"].tolist(),
                    "definition": "pooled RMS over the two reference panels or two candidate copies on common unordered non-diagonal pairs",
                },
                "shape_validation": shape_validation["fdg_proposal"],
            },
        },
        "global_color": {
            "colormap": COLORMAP,
            "missing_color": MISSING_COLOR,
            "vmin": 0.0,
            "vmax": global_vmax,
            "clip": False,
            "finite_normalized_values_across_reference_continuation_fdg": int(len(all_finite_normalized)),
            "global_max_source": "max of all finite normalized values in both candidate figures (reference panels included)",
            "identical_across_figures": True,
        },
        "render": render_records,
        "artifacts": artifact_records,
        "validation_checks": {
            "symmetry": True,
            "zero_diagonal": True,
            "missing_mask": True,
            "finite_entries": True,
            "rms_normalization": True,
            "global_identical_color_scale": True,
            "grid_and_mapping": True,
            "png_1800px": True,
            "pdf_6inch": True,
        },
        "no_new_inference_statistics": True,
        "no_L2_claim": True,
    }
    write_json(validation_path, validation)
    provenance = {
        "schema": "supplemental_chr1_distance_matrices_022.provenance.v1",
        "status": "COMPLETE",
        "generated_utc": validation["generated_utc"],
        "supplement_root": str(HERE.resolve()),
        "workflow": [
            "Read frozen config, source identities, 022 decision, continuation summary, track map and R2 JSON/TSV; hash all files programmatically.",
            "Register continuation 3DG, mapped FDG proposal 3DG and reference bytes in a fresh EvalGate before loading the reference payload.",
            "Parse both candidate 3DG files directly as float64 and validate all 40 tracks / 5290 physical beads on the origin-zero full grid.",
            "Load data/P9016.1m.3dg.gz only through pr.ref3dg.load_reference after the eval gate is armed; use literal chr1(mat) and chr1(pat).",
            "Apply the frozen 022 R2 cross/swapped best-swap mapping, build the common 188-bin / 17578-pair mask, and compute pooled RMS scales.",
            "Compute one shared vmin=0/vmax global color norm over all finite normalized values in both four-panel figures, then render 6x6 inch PNG/PDF outputs and delivery copies.",
            "Write validation, provenance, README and fresh run log without modifying prior experiments or source inputs.",
        ],
        "source_identity": source_identity,
        "r2_identity": r2_identity,
        "template_reuse": template_records,
        "validation": {"path": str(validation_path.resolve()), "sha256": sha256_file(validation_path)},
        "artifacts": artifact_records,
        "scope_guard": validation["scope_guard"],
    }
    write_json(provenance_path, provenance)
    write_readme(readme_path, config, validation, provenance_path)
    validation["provenance"] = file_record(provenance_path, "provenance")
    validation["readme"] = file_record(readme_path, "README")
    # 最后 metadata records 写出后只更新一次 validation，不改变
    # 已验证的数值 payload 或任何 source input。
    write_json(validation_path, validation)
    print(json.dumps({
        "status": "PASS",
        "supplement": str(HERE.resolve()),
        "common_bins": EXPECTED_COMMON_BINS,
        "common_non_diagonal_pairs": EXPECTED_COMMON_PAIRS,
        "global_vmin": 0.0,
        "global_vmax": global_vmax,
        "pairing": "cross/swapped; c01a(A)->chr1(pat), c01b(B)->chr1(mat)",
        "continuation_png": artifact_records["v1_continuation"]["delivery_png"],
        "continuation_pdf": artifact_records["v1_continuation"]["delivery_pdf"],
        "fdg_png": artifact_records["fdg_proposal"]["delivery_png"],
        "fdg_pdf": artifact_records["fdg_proposal"]["delivery_pdf"],
        "continuation_npz": artifact_records["v1_continuation"]["npz"],
        "fdg_npz": artifact_records["fdg_proposal"]["npz"],
        "validation": str(validation_path.resolve()),
        "provenance": str(provenance_path.resolve()),
        "readme": str(readme_path.resolve()),
    }, ensure_ascii=False, indent=2))
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
