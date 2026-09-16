#!/usr/bin/env python3
"""为锁定的 029 R2 创建仅供显示的 chr1 contrast 和 distance-map 补充结果。

本脚本仅用于 evaluation/display。它读取已锁定哈希的 R2 表格、五个已锁定哈希的候选坐标文件，以及在候选哈希通过后读取的已锁定哈希 evaluation reference。它不打开 phase data、pairs 或任何 fit/selection input，也不调用 fitter 或重新计算 R2。
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parents[1]
PROJECT_ROOT = RUN_ROOT.parent.parent
DEFAULT_CONFIG = HERE / "config.json"
VARIANT_ORDER = ("C0", "C1", "C2-map", "C2-free", "C3")
VARIANT_COLORS = {
    "C0": "#1b9e77",
    "C1": "#d95f02",
    "C2-map": "#7570b3",
    "C2-free": "#e7298a",
    "C3": "#66a61e",
}
CHROMOSOME = "chr1"
CANDIDATE_TRACKS = ("c01a", "c01b")
REFERENCE_TRACKS = ("chr1(mat)", "chr1(pat)")
BIN = 1_000_000
FONT_SIZE = 7.0
DPI = 300
MISSING_COLOR = "#bdbdbd"
COLORMAP = "coolwarm_r"
POSITIONS_BP = np.arange(3_000_000, 195_471_971, BIN, dtype=np.int64)
N_BINS = int(POSITIONS_BP.size)
N_TOTAL_PAIRS = N_BINS * (N_BINS - 1) // 2


class DisplayError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DisplayError(f"cannot read {label}: {path}") from exc


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_value(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_record(path: Path, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise DisplayError(f"{label} is unavailable: {path}")
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise DisplayError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": actual,
        "size_bytes": int(path.stat().st_size),
    }
    if expected_sha256 is not None:
        record["expected_sha256"] = expected_sha256
    return record


def tree_inventory(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(
            {
                "relative_path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return records


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise DisplayError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-12) -> None:
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tol:
        raise DisplayError(f"{label}: expected {expected!r}, got {actual!r}")


def parse_json_field(value: Any, label: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise DisplayError(f"invalid JSON field {label}: {value!r}") from exc
    return value


def lock_records(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    locks = config["input_lock"]
    records: dict[str, Any] = {}
    order: list[str] = []

    # candidate coordinate hashes 必须在 reference 甚至被 hash/解析之前检查，符合 evaluation-side gate 顺序。
    for variant in VARIANT_ORDER:
        spec = locks["candidates"][variant]
        label = f"candidate {variant}"
        records[f"candidate:{variant}"] = file_record(
            resolve_path(spec["path"]), label, spec["sha256"]
        )
        order.append(f"candidate:{variant}")

    reference = locks["reference"]
    records["reference"] = file_record(
        resolve_path(reference["path"]), "evaluation reference", reference["sha256"]
    )
    order.append("reference")

    for name in (
        "release_manifest",
        "r2_per_chromosome",
        "r2_summary",
        "evaluation_manifest",
        "controller_r2_delivery_receipt",
    ):
        spec = locks[name]
        records[name] = file_record(resolve_path(spec["path"]), name, spec["sha256"])
        order.append(name)
    return records, order


def load_candidate_chr1(path: Path) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    coords: dict[str, dict[int, np.ndarray]] = {track: {} for track in CANDIDATE_TRACKS}
    n_lines = 0
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                raise DisplayError(f"malformed candidate line {path}:{line_number}")
            track = fields[0]
            if track not in coords:
                continue
            try:
                position = int(fields[1])
                point = np.asarray([float(fields[2]), float(fields[3]), float(fields[4])], dtype=np.float64)
            except ValueError as exc:
                raise DisplayError(f"non-numeric candidate line {path}:{line_number}") from exc
            if not np.isfinite(point).all():
                raise DisplayError(f"non-finite candidate coordinate {path}:{line_number}")
            if position in coords[track]:
                raise DisplayError(f"duplicate candidate coordinate {track}:{position} in {path}")
            coords[track][position] = point
            n_lines += 1

    expected = {int(position) for position in np.arange(0, 195_471_971, BIN, dtype=np.int64)}
    for track, values in coords.items():
        if set(values) != expected:
            raise DisplayError(
                f"{path} {track} grid mismatch: expected {len(expected)} chr1 bins, got {len(values)}"
            )
    return coords, {
        "path": str(path.resolve()),
        "chr1_tracks": list(CANDIDATE_TRACKS),
        "n_chr1_coordinate_lines": n_lines,
        "n_finite_bins": {track: len(values) for track, values in coords.items()},
    }


def load_reference_chr1(path: Path) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    coords: dict[str, dict[int, np.ndarray]] = {track: {} for track in REFERENCE_TRACKS}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            track = fields[0]
            if track not in coords:
                continue
            try:
                position = int(fields[1])
                point = np.asarray([float(fields[2]), float(fields[3]), float(fields[4])], dtype=np.float64)
            except ValueError as exc:
                raise DisplayError(f"non-numeric reference line {path}:{line_number}") from exc
            if not np.isfinite(point).all():
                raise DisplayError(f"non-finite reference coordinate {path}:{line_number}")
            if position in coords[track]:
                raise DisplayError(f"duplicate reference coordinate {track}:{position}")
            coords[track][position] = point

    expected = {int(position) for position in POSITIONS_BP}
    for track, values in coords.items():
        if not set(values).issubset(expected):
            raise DisplayError(f"reference {track} contains positions outside chr1 display grid")
        if not values:
            raise DisplayError(f"reference {track} has no chr1 coordinates")
    return coords, {
        "path": str(path.resolve()),
        "tracks": list(REFERENCE_TRACKS),
        "n_finite_bins": {track: len(values) for track, values in coords.items()},
        "missing_positions_bp": {
            track: [int(position) for position in POSITIONS_BP if int(position) not in values]
            for track, values in coords.items()
        },
    }


def points_for_track(
    coords: dict[str, dict[int, np.ndarray]], track: str
) -> tuple[np.ndarray, np.ndarray]:
    values = coords.get(track, {})
    points = np.full((N_BINS, 3), np.nan, dtype=np.float64)
    for index, position in enumerate(POSITIONS_BP):
        point = values.get(int(position))
        if point is not None:
            points[index] = point
    return points, np.isfinite(points).all(axis=1)


def distance_matrix(points: np.ndarray) -> np.ndarray:
    if points.shape != (N_BINS, 3) or points.dtype != np.float64:
        raise DisplayError("distance matrix input must be a float64 chr1 grid")
    delta = points[:, None, :] - points[None, :, :]
    matrix = np.linalg.norm(delta, axis=2).astype(np.float64, copy=False)
    finite_bins = np.isfinite(points).all(axis=1)
    matrix[~finite_bins, :] = np.nan
    matrix[:, ~finite_bins] = np.nan
    diagonal = np.diag_indices(N_BINS)
    matrix[diagonal] = np.where(finite_bins, 0.0, np.nan)
    return matrix


def offdiag_mask(bin_mask: np.ndarray) -> np.ndarray:
    return np.logical_and.outer(bin_mask, bin_mask) & ~np.eye(N_BINS, dtype=bool)


def pooled_rms(matrices: np.ndarray, pair_mask: np.ndarray) -> float:
    values = []
    upper = np.triu(pair_mask, k=1)
    for matrix in matrices:
        current = matrix[upper]
        if current.size == 0 or not np.isfinite(current).all():
            raise DisplayError("pooled RMS received missing/non-finite common pairs")
        values.append(current)
    merged = np.concatenate(values)
    return float(np.sqrt(np.mean(np.square(merged, dtype=np.float64), dtype=np.float64)))


def masked_matrix(matrix: np.ndarray, pair_mask: np.ndarray, common_bins: np.ndarray) -> np.ndarray:
    out = np.full_like(matrix, np.nan, dtype=np.float64)
    out[pair_mask] = matrix[pair_mask]
    # 仅对 all-panel common mask 中的 bins 保留 zero diagonal。
    for index in np.flatnonzero(common_bins):
        out[index, index] = 0.0
    return out


def input_coverage(mask_metadata: dict[str, Any], key: str) -> dict[str, Any]:
    coverage = mask_metadata.get("track_coverage", {})
    if key not in coverage:
        raise DisplayError(f"evaluation mask metadata lacks {key}")
    return coverage[key]


def find_chr1_mask(evaluation_manifest: dict[str, Any]) -> dict[str, Any]:
    entries = evaluation_manifest.get("mask_metadata", [])
    matches = [entry for entry in entries if entry.get("chromosome") == CHROMOSOME]
    if len(matches) != 1:
        raise DisplayError(f"expected one chr1 mask metadata entry, found {len(matches)}")
    return matches[0]


def validate_locked_metadata(
    config: dict[str, Any],
    release: dict[str, Any],
    per_chromosome: dict[str, Any],
    summary: dict[str, Any],
    evaluation_manifest: dict[str, Any],
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    locks = config["input_lock"]
    assert_equal(release.get("status"), "locked_for_evaluation", "release status")
    assert_equal(release.get("reference", {}).get("sha256"), locks["reference"]["sha256"], "release reference SHA")
    assert_equal(release.get("within_variant_representatives"), {
        "C0": "C0-bundle2",
        "C1": "C1-bundle2",
        "C2-free": "C2-free-bundle2",
        "C2-map": "C2-map-bundle2",
        "C3": "C3-bundle2",
    }, "release representatives")
    assert_equal(receipt.get("status"), "complete", "controller receipt status")
    scope = receipt.get("scope_audit", {})
    assert_equal(scope.get("r2_only"), True, "controller R2-only audit")
    assert_equal(scope.get("real_fit_started_by_evaluator"), False, "controller fit audit")
    assert_equal(scope.get("phase_payload_opened"), False, "controller phase audit")
    assert_equal(scope.get("selection_recomputed"), False, "controller selection audit")
    assert_equal(evaluation_manifest.get("condition_count"), 21, "evaluation condition count")

    mask = find_chr1_mask(evaluation_manifest)
    grid = config["grid"]
    assert_equal(mask.get("bin_size_bp"), grid["bin_size_bp"], "evaluation chr1 bin size")
    assert_equal(mask.get("offset_bp"), grid["offset_bp"], "evaluation chr1 offset")
    assert_equal(mask.get("n_bins"), N_BINS, "evaluation chr1 n_bins")
    assert_equal(mask.get("n_total_non_diagonal_pairs"), N_TOTAL_PAIRS, "evaluation total pairs")
    assert_equal(mask.get("n_common_pairs"), config["mask_policy"]["expected_common_unordered_offdiag_pairs"], "evaluation common pairs")
    assert_equal(len(mask.get("included_condition_ids", [])), config["mask_policy"]["condition_count"], "evaluation included mask conditions")
    assert_equal(mask.get("excluded_failed_condition_ids"), [], "evaluation excluded failed conditions")

    rows = per_chromosome.get("rows", [])
    if len(rows) != 420:
        raise DisplayError(f"R2 per-chromosome row count expected 420, got {len(rows)}")
    selected: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_ORDER:
        condition_id = locks["candidates"][variant]["condition_id"]
        matches = [row for row in rows if row.get("condition_id") == condition_id and row.get("chromosome") == CHROMOSOME]
        if len(matches) != 1:
            raise DisplayError(f"expected one locked R2 chr1 row for {condition_id}, found {len(matches)}")
        row = matches[0]
        if row.get("bundle_id") != "bundle2":
            raise DisplayError(f"{condition_id} is not bundle2")
        if row.get("metric_status") != "ok" or row.get("formal_metric_status") != "ok":
            raise DisplayError(f"{condition_id} R2 row is not metric_status=ok")
        if not math.isfinite(float(row.get("contrast"))):
            raise DisplayError(f"{condition_id} contrast is not finite")
        assert_equal(row.get("n_bins"), N_BINS, f"{condition_id} R2 n_bins")
        assert_equal(row.get("n_total_non_diagonal_pairs"), N_TOTAL_PAIRS, f"{condition_id} R2 total pairs")
        assert_equal(row.get("n_common_pairs"), mask["n_common_pairs"], f"{condition_id} R2 common pairs")
        pairing = row.get("pairing")
        orientation = row.get("orientation")
        if pairing not in {"direct", "cross"} or orientation not in {"direct", "swapped"}:
            raise DisplayError(f"{condition_id} has unsupported R2 pairing/orientation")
        track_names = parse_json_field(row.get("track_names"), f"{condition_id}.track_names")
        assert_equal(track_names, list(CANDIDATE_TRACKS), f"{condition_id} track names")
        rhos = {
            "A_mat": float(row["rho_A_mat"]),
            "A_pat": float(row["rho_A_pat"]),
            "B_mat": float(row["rho_B_mat"]),
            "B_pat": float(row["rho_B_pat"]),
        }
        if not all(math.isfinite(value) for value in rhos.values()):
            raise DisplayError(f"{condition_id} has non-finite four-rho record")
        summary_row = summary.get("condition_summary", {}).get(condition_id, {})
        assert_equal(summary_row.get("contrast_is_applicable"), True, f"{condition_id} summary contrast applicability")
        assert_equal(summary_row.get("contrast_n_finite"), 20, f"{condition_id} summary contrast count")
        selected[variant] = {
            "condition_id": condition_id,
            "contrast_chr1": float(row["contrast"]),
            "pairing": pairing,
            "orientation": orientation,
            "rhos": rhos,
            "r2_row": row,
        }

    selected_mask_coverage: dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        condition_id = locks["candidates"][variant]["condition_id"]
        for copy_name, key in (("A", f"condition:{condition_id}:copyA"), ("B", f"condition:{condition_id}:copyB")):
            selected_mask_coverage[key] = input_coverage(mask, key)
    selected_mask_coverage["reference:mat"] = input_coverage(mask, "reference:mat")
    selected_mask_coverage["reference:pat"] = input_coverage(mask, "reference:pat")

    return {
        "chr1_mask_metadata": mask,
        "selected_mask_coverage": selected_mask_coverage,
        "r2_selected_rows": selected,
        "r2_row_count": len(rows),
    }, selected


def panel_mapping(r2_row: dict[str, Any]) -> tuple[str, str]:
    orientation = r2_row["orientation"]
    if orientation == "direct":
        return "c01a", "c01b"
    if orientation == "swapped":
        return "c01b", "c01a"
    raise DisplayError(f"unsupported orientation: {orientation}")


def build_variant_matrix(
    variant: str,
    selected: dict[str, Any],
    candidate_coords: dict[str, dict[str, dict[int, np.ndarray]]],
    reference_coords: dict[str, dict[int, np.ndarray]],
    mask_metadata: dict[str, Any],
) -> dict[str, Any]:
    mat_track, pat_track = panel_mapping(selected["r2_row"])
    tracks = ["chr1(mat)", "chr1(pat)", mat_track, pat_track]
    labels = [
        "Reference mat",
        "Reference pat",
        f"{variant} copy {'B' if mat_track == 'c01b' else 'A'} -> ref mat",
        f"{variant} copy {'B' if pat_track == 'c01b' else 'A'} -> ref pat",
    ]
    point_sources: list[tuple[np.ndarray, str]] = []
    for track in REFERENCE_TRACKS:
        point_sources.append((points_for_track(reference_coords, track)[0], track))
    for track in (mat_track, pat_track):
        point_sources.append((points_for_track(candidate_coords[variant], track)[0], track))
    points = np.asarray([item[0] for item in point_sources], dtype=np.float64)
    finite_bins = np.asarray([np.isfinite(item).all(axis=1) for item in points], dtype=bool)
    raw = np.asarray([distance_matrix(item) for item in points], dtype=np.float64)
    common_bins = np.logical_and.reduce(finite_bins, axis=0)
    common_pair_mask = offdiag_mask(common_bins)
    implied_common_bins = int(round((1 + math.sqrt(1 + 8 * mask_metadata["n_common_pairs"])) / 2))
    if int(common_bins.sum()) != implied_common_bins:
        raise DisplayError(
            f"{variant} coordinate-derived common bins={int(common_bins.sum())}, "
            f"evaluation common pairs={mask_metadata['n_common_pairs']} (implied bins {implied_common_bins})"
        )
    # n_common_pairs 是 pair count，因此独立验证精确的 upper-triangle mask，
    # 而不依赖 hard-coded bin count。
    derived_pairs = int(np.triu(common_pair_mask, k=1).sum())
    if derived_pairs != int(mask_metadata["n_common_pairs"]):
        raise DisplayError(f"{variant} derived common pair count mismatch")

    common_raw = np.asarray([masked_matrix(matrix, common_pair_mask, common_bins) for matrix in raw], dtype=np.float64)
    reference_scale = pooled_rms(common_raw[:2], common_pair_mask)
    candidate_scale = pooled_rms(common_raw[2:], common_pair_mask)
    scales = np.asarray([reference_scale, reference_scale, candidate_scale, candidate_scale], dtype=np.float64)
    normalized = common_raw / scales[:, None, None]

    panel_coverage: dict[str, Any] = {}
    for index, (track, finite) in enumerate(zip(tracks, finite_bins)):
        pair_count = int(np.triu(np.isfinite(raw[index]), k=1).sum())
        missing_positions = [int(position) for position, ok in zip(POSITIONS_BP, finite) if not ok]
        panel_coverage[track] = {
            "n_expected_bins": N_BINS,
            "n_finite_bins": int(finite.sum()),
            "n_missing_bins": int((~finite).sum()),
            "n_finite_pairs": pair_count,
            "n_missing_pairs": int(N_TOTAL_PAIRS - pair_count),
            "missing_positions_bp": missing_positions,
        }
        expected_key = (
            "reference:mat" if track == "chr1(mat)" else
            "reference:pat" if track == "chr1(pat)" else
            f"condition:{selected['condition_id']}:copy{'A' if track == 'c01a' else 'B'}"
        )
        expected = mask_metadata["track_coverage"][expected_key]
        for field in ("n_expected_bins", "n_finite_bins", "n_missing_bins", "n_finite_pairs", "n_missing_pairs"):
            assert_equal(panel_coverage[track][field], expected[field], f"{variant} {track} {field}")

    return {
        "variant": variant,
        "condition_id": selected["condition_id"],
        "pairing": selected["pairing"],
        "orientation": selected["orientation"],
        "tracks": tracks,
        "labels": labels,
        "points": points,
        "finite_bins": finite_bins,
        "raw": raw,
        "common_raw": common_raw,
        "normalized": normalized,
        "common_bins": common_bins,
        "common_pair_mask": common_pair_mask,
        "scales": scales,
        "reference_pooled_rms": reference_scale,
        "candidate_pooled_rms": candidate_scale,
        "panel_coverage": panel_coverage,
        "rhos": selected["rhos"],
        "contrast_chr1": selected["contrast_chr1"],
    }


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mpl_colors
    from matplotlib import patheffects
    from matplotlib.text import Text

    matplotlib.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })
    return matplotlib, plt, mpl_colors, patheffects, Text


def check_text_layout(fig: Any, Text: Any, label: str) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside: list[str] = []
    wrong_font: list[dict[str, Any]] = []
    for text in fig.findobj(Text):
        if not text.get_visible():
            continue
        size = float(text.get_fontsize())
        if abs(size - FONT_SIZE) > 1e-6:
            wrong_font.append({"text": text.get_text(), "fontsize": size})
        bbox = text.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        if bbox.x0 < -0.01 or bbox.x1 > 1.01 or bbox.y0 < -0.01 or bbox.y1 > 1.01:
            outside.append(text.get_text())
    if outside:
        raise DisplayError(f"{label} has text outside figure: {outside}")
    if wrong_font:
        raise DisplayError(f"{label} has non-7pt text: {wrong_font[:5]}")
    return {"all_text_inside": True, "all_visible_text_7pt": True, "visible_text_count": len(fig.findobj(Text))}


def check_xlabel_footer_overlap(
    fig: Any,
    axes: list[Any],
    footers: list[Any],
    label: str,
) -> dict[str, Any]:
    """检查已渲染 xlabel/footer 的 bbox，而不仅是 canvas 边界。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    footer_boxes = [
        footer.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        for footer in footers
    ]
    overlaps: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(axes):
        xlabel = axis.xaxis.get_label()
        xlabel_box = xlabel.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        for footer_index, footer_box in enumerate(footer_boxes):
            if xlabel_box.overlaps(footer_box):
                overlaps.append({"axis": axis_index, "footer": footer_index})
    if overlaps:
        raise DisplayError(f"{label} xlabel/footer bbox overlap: {overlaps}")
    return {
        "xlabel_footer_bbox_overlap": False,
        "xlabel_footer_bbox_overlap_count": 0,
        "checked_xlabel_count": len(axes),
        "checked_footer_count": len(footers),
    }


def save_figure(fig: Any, path_png: Path, path_pdf: Path) -> dict[str, Any]:
    fig.savefig(path_png, dpi=DPI, format="png", facecolor="white")
    fig.savefig(path_pdf, dpi=DPI, format="pdf", facecolor="white")
    return {
        "png": file_record(path_png, path_png.name),
        "pdf": file_record(path_pdf, path_pdf.name),
    }


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise DisplayError(f"not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def pdf_media_box(path: Path) -> tuple[float, float]:
    data = path.read_bytes()
    matches = re.findall(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", data)
    if not matches:
        raise DisplayError(f"cannot find PDF MediaBox: {path}")
    return float(matches[0][0]), float(matches[0][1])


def render_boxplot(values: np.ndarray, chromosomes: list[str], out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    matplotlib, plt, _, _, Text = import_matplotlib()
    fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=DPI)
    x = np.arange(1, len(VARIANT_ORDER) + 1, dtype=float)
    for index in range(values.shape[1]):
        ax.plot(x, values[:, index], color="#b0b0b0", linewidth=0.55, alpha=0.8, zorder=1)
    box = ax.boxplot(
        [values[index] for index in range(values.shape[0])],
        positions=x,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 0.8},
        whiskerprops={"color": "#404040", "linewidth": 0.6},
        capprops={"color": "#404040", "linewidth": 0.6},
        boxprops={"linewidth": 0.7},
    )
    for patch, variant in zip(box["boxes"], VARIANT_ORDER):
        patch.set_facecolor(VARIANT_COLORS[variant])
        patch.set_alpha(0.75)
        patch.set_edgecolor("#202020")
    for index, variant in enumerate(VARIANT_ORDER):
        ax.scatter(
            np.full(values.shape[1], x[index]),
            values[index],
            s=9,
            color=VARIANT_COLORS[variant],
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )
    ax.axhline(0.0, color="#505050", linewidth=0.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANT_ORDER)
    ax.set_ylabel("matched - cross")
    ax.set_title("Genome-wide R2 contrast | count-selected bundle2", pad=5.0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.35)
    ax.set_axisbelow(True)
    ax.margins(x=0.08, y=0.12)
    fig.text(
        0.5,
        0.025,
        "gray lines/points are paired chromosomes; each box is a 20-chromosome distribution, not a CI",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.24, top=0.82)
    layout = check_text_layout(fig, Text, "contrast boxplot")
    png_path = out_dir / "genomewide_contrast_boxplot.png"
    pdf_path = out_dir / "genomewide_contrast_boxplot.pdf"
    artifacts = save_figure(fig, png_path, pdf_path)
    dimensions = png_dimensions(png_path)
    media_box = pdf_media_box(pdf_path)
    if dimensions != (1800, 900):
        raise DisplayError(f"contrast PNG dimensions expected 1800x900, got {dimensions}")
    if abs(media_box[0] - 432.0) > 0.1 or abs(media_box[1] - 216.0) > 0.1:
        raise DisplayError(f"contrast PDF MediaBox expected 432x216 pt, got {media_box}")
    plt.close(fig)
    return artifacts, {"png_dimensions": dimensions, "pdf_media_box_pt": media_box, "layout": layout}


def render_heatmap(result: dict[str, Any], global_vmax: float, out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    matplotlib, plt, mpl_colors, _, Text = import_matplotlib()
    cmap = copy.copy(matplotlib.colormaps[COLORMAP])
    cmap.set_bad(MISSING_COLOR)
    norm = mpl_colors.Normalize(vmin=0.0, vmax=global_vmax, clip=False)
    fig = plt.figure(figsize=(6.0, 6.0), dpi=DPI)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.055),
        height_ratios=(1.0, 1.0),
        left=0.11,
        right=0.91,
        bottom=0.145,
        top=0.875,
        wspace=0.27,
        hspace=0.35,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    image = None
    extent = [float(POSITIONS_BP[0] / 1e6 - 0.5), float(POSITIONS_BP[-1] / 1e6 + 0.5), float(POSITIONS_BP[0] / 1e6 - 0.5), float(POSITIONS_BP[-1] / 1e6 + 0.5)]
    ticks = [3, 50, 100, 150, 195]
    for index, (ax, matrix, label) in enumerate(zip(axes, result["normalized"], result["labels"])):
        image = ax.imshow(
            matrix,
            origin="lower",
            interpolation="none",
            extent=extent,
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )
        ax.set_title(label, pad=3.0)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("genomic Mb")
        ax.set_ylabel("genomic Mb")
        ax.tick_params(length=2.0, pad=1.5)
    colorbar_ax = fig.add_subplot(grid[:, 2])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("normalized distance", fontsize=FONT_SIZE, labelpad=3.0)
    colorbar.ax.tick_params(labelsize=FONT_SIZE, length=2.0, pad=1.5)
    rho = result["rhos"]
    rho_text = (
        f"R2 {result['pairing']}/{result['orientation']} | "
        f"rho A(mat/pat)={rho['A_mat']:.3f}/{rho['A_pat']:.3f}; "
        f"B(mat/pat)={rho['B_mat']:.3f}/{rho['B_pat']:.3f}"
    )
    fig.suptitle(f"P9016 chr1 | {result['variant']} | distance matrices", fontsize=FONT_SIZE, y=0.965)
    axis_footer = fig.text(0.5, 0.065, "absolute genomic position (Mb); missing rows/columns = gray NaN", ha="center", va="center", fontsize=FONT_SIZE)
    rho_footer = fig.text(0.5, 0.025, rho_text, ha="center", va="center", fontsize=FONT_SIZE)
    layout = check_text_layout(fig, Text, f"{result['variant']} heatmap")
    layout.update(check_xlabel_footer_overlap(fig, axes, [axis_footer, rho_footer], f"{result['variant']} heatmap"))
    png_path = out_dir / f"{result['variant']}_chr1_distance_matrices.png"
    pdf_path = out_dir / f"{result['variant']}_chr1_distance_matrices.pdf"
    artifacts = save_figure(fig, png_path, pdf_path)
    dimensions = png_dimensions(png_path)
    media_box = pdf_media_box(pdf_path)
    if dimensions != (1800, 1800):
        raise DisplayError(f"{result['variant']} PNG dimensions expected 1800x1800, got {dimensions}")
    if abs(media_box[0] - 432.0) > 0.1 or abs(media_box[1] - 432.0) > 0.1:
        raise DisplayError(f"{result['variant']} PDF MediaBox expected 432x432 pt, got {media_box}")
    plt.close(fig)
    return artifacts, {"png_dimensions": dimensions, "pdf_media_box_pt": media_box, "layout": layout}


def write_readme(
    path: Path,
    config: dict[str, Any],
    lock_before: dict[str, Any],
    lock_after: dict[str, Any],
    metadata: dict[str, Any],
    results: dict[str, dict[str, Any]],
    contrast_artifact: dict[str, Any],
    heatmap_artifacts: dict[str, Any],
    global_vmax: float,
    validation_path: Path,
    provenance_path: Path,
) -> None:
    mask = metadata["chr1_mask_metadata"]
    common_bins = results["C0"]["common_bins"]
    missing = results["C0"]["panel_coverage"]
    contrast_lines = []
    for variant in VARIANT_ORDER:
        item = results[variant]
        r = item["rhos"]
        contrast_lines.append(
            f"| {variant} | `{item['condition_id']}` | {item['pairing']} / {item['orientation']} | "
            f"{r['A_mat']:.12f} | {r['A_pat']:.12f} | {r['B_mat']:.12f} | {r['B_pat']:.12f} |"
        )
    scale_lines = []
    for variant in VARIANT_ORDER:
        item = results[variant]
        scale_lines.append(
            f"| {variant} | {item['reference_pooled_rms']:.15g} | {item['candidate_pooled_rms']:.15g} |"
        )
    artifact_lines = [
        f"- `{contrast_artifact['png']['path']}` SHA256 `{contrast_artifact['png']['sha256']}`",
        f"- `{contrast_artifact['pdf']['path']}` SHA256 `{contrast_artifact['pdf']['sha256']}`",
    ]
    for variant in VARIANT_ORDER:
        artifact_lines.extend([
            f"- `{heatmap_artifacts[variant]['png']['path']}` SHA256 `{heatmap_artifacts[variant]['png']['sha256']}`",
            f"- `{heatmap_artifacts[variant]['pdf']['path']}` SHA256 `{heatmap_artifacts[variant]['pdf']['sha256']}`",
        ])
    missing_text = "; ".join(
        f"{track}: {coverage['n_missing_bins']} ({','.join(f'{position/1e6:.0f}Mb' for position in coverage['missing_positions_bp']) or 'none'})"
        for track, coverage in missing.items()
    )
    readme = f"""# 029 全基因组 contrast / chr1 距离展示补充

**状态：** `PASS`。这是 029 封存终点的仅供评价展示补充；不新拟合、不重算 R2、不重算全 R2、不改变任何原封存、代码或报告文件。

## 范围与锁定

- 真实运行：`{config['scope']['real_run']}`；只使用五个锁定的 count-selected `bundle2` 代表：C0、C1、C2-map、C2-free、C3。
- 训练、选择和 R2 均未在本目录执行；没有打开 phase payload。reference 只在全部五个 candidate 坐标 SHA 核验通过后读取。
- `release_manifest_r2.json` SHA256 `{lock_before['release_manifest']['sha256']}`；`r2_per_chromosome.json` SHA256 `{lock_before['r2_per_chromosome']['sha256']}`；`r2_summary.json` SHA256 `{lock_before['r2_summary']['sha256']}`；`evaluation_manifest.json` SHA256 `{lock_before['evaluation_manifest']['sha256']}`；controller receipt SHA256 `{lock_before['controller_r2_delivery_receipt']['sha256']}`。
- 参考结构 `/mnt/ssd/zliu/phase_restart/data/P9016.1m.3dg.gz` SHA256 `{lock_before['reference']['sha256']}`，仅使用`chr1(mat)`/`chr1(pat)`评价轨迹。

## 全基因组 contrast 箱线图

- 输出是绝对 `matched - cross`，不是相对 C0 的效应量；数值只从锁定 R2 per-chromosome 表读取，没有新算 rho。
- 图标题为`全基因组 R2 contrast | 按计数选择的 bundle2`，明确箱线图使用 20 条染色体而非仅 chr1；五组分别为 C0、C1、C2-map、C2-free、C3；每组 20 个 chr 值，短 xlabel 保留；灰线连接同一条染色体的五个值，点为 20 条染色体；箱体是 20 条染色体的分布，不是 CI，也不是 60 个独立重复。
- 原 R2 相对 C0 的配对效应图仍在 [`../../evaluation-r2/plots/paired_effects/r2_paired_effects_contrast.png`](../../evaluation-r2/plots/paired_effects/r2_paired_effects_contrast.png)，两者定义不同。

## chr1距离热图

- 固定网格为`range(3_000_000, 195_471_971, 1_000_000)`：{N_BINS} bins，完整绝对基因组轴；评价清单的 21-condition 共同 mask 为{int(common_bins.sum())}/{N_BINS} bins、{int(mask['n_common_pairs'])} 个无序非对角距离对；完整非对角距离对数为{N_TOTAL_PAIRS}。
- 共同 mask 从坐标有限位置恢复并与`evaluation_manifest.mask_metadata`/R2 coverage 逐轨核对，不用临时五候选 mask 扩张。缺失行列保留为 NaN 并以`{MISSING_COLOR}`灰色显示，不填 0、不压缩基因组轴。实际缺失位置：{missing_text}。
- 每张图四面板顺序为：参考结构 mat、参考结构 pat；candidate 按锁定 R2 的整条染色体方向对齐到 ref mat、ref pat。该几何对应不声称 candidate copy A/B 的亲本身份，未做局部 swap。

### R2 chr1 四个 rho 与配对方向

| 变体 | 锁定条件 | 配对 / 方向 | rho A(mat) | rho A(pat) | rho B(mat) | rho B(pat) |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(contrast_lines)}

### pooled RMS 与全局色标

- reference 两 copy 在完全相同的共同无序非对角 mask 上共享一个 pooled RMS；每种 candidate 两 copy 共享各自的 pooled RMS，禁止单拷贝归一化。所有距离为 float64 欧氏距离。

| 变体 | reference pooled RMS | candidate pooled RMS |
|---|---:|---:|
{chr(10).join(scale_lines)}

- colormap 为`coolwarm_r`（近=红、远=蓝），`vmin=0`，统一 `vmax={global_vmax:.15g}`，来自 reference 与 5×2 candidate 全部有限归一化值的共同最大值，不做分位数裁剪。
- C2-map 与 C2-free 保留为两个独立变体文件；其锁定坐标 SHA 相同，脚本验证两者 raw/common-raw/normalized 矩阵逐元素完全相同，但未静默合并。

## 输出与验证

- 箱线图：`{contrast_artifact['png']['path']}`、`{contrast_artifact['pdf']['path']}`。
- 五份热图PNG/PDF及五份raw/common-raw/normalized矩阵NPZ见下列路径：
{chr(10).join(artifact_lines)}
- 机器验证：`{validation_path}`；provenance：`{provenance_path}`；配置：`{(HERE / 'config.json').resolve()}`。
- PNG 为 300 dpi：箱线图 1800×900，热图 1800×1800；PDF 分别为 6×3 和 6×6 英寸；所有可见文字按 7 pt 检查并执行文字越界检查。
- 本补充只做展示与轻量数值核验，不推断新的L2/L1/L3结论。
- 输入 hash 前后相同：`{lock_before == lock_after}`；evaluation-r2 原目录只读、未修改。
"""
    path.write_text(readme, encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path, "config")
    out_dir = config_path.parent.resolve()
    if out_dir != HERE:
        raise DisplayError("config must live beside this independent script")
    for output_name in (
        "chr1_contrast_boxplot.png",
        "chr1_contrast_boxplot.pdf",
        "validation.json",
        "provenance.json",
        "terminal_evidence.json",
        "README.md",
    ):
        if (out_dir / output_name).exists():
            raise DisplayError(f"refusing to overwrite existing output: {out_dir / output_name}")

    evaluation_root = resolve_path("test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2")
    eval_inventory_before = tree_inventory(evaluation_root)
    lock_before, hash_order = lock_records(config)

    release = read_json(resolve_path(config["input_lock"]["release_manifest"]["path"]), "release manifest")
    per_chromosome = read_json(resolve_path(config["input_lock"]["r2_per_chromosome"]["path"]), "R2 per chromosome")
    summary = read_json(resolve_path(config["input_lock"]["r2_summary"]["path"]), "R2 summary")
    evaluation_manifest = read_json(resolve_path(config["input_lock"]["evaluation_manifest"]["path"]), "evaluation manifest")
    receipt = read_json(resolve_path(config["input_lock"]["controller_r2_delivery_receipt"]["path"]), "controller receipt")
    metadata, selected = validate_locked_metadata(config, release, per_chromosome, summary, evaluation_manifest, receipt)

    # 只有五个 candidate SHA lock 全部通过后才读取 candidate payload。
    candidate_coords: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    candidate_summaries: dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        spec = config["input_lock"]["candidates"][variant]
        candidate_coords[variant], candidate_summaries[variant] = load_candidate_chr1(resolve_path(spec["path"]))

    # 只有每个 candidate hash 和 metadata lock 均验证后才读取 reference。
    reference_path = resolve_path(config["input_lock"]["reference"]["path"])
    reference_coords, reference_summary = load_reference_chr1(reference_path)

    results: dict[str, dict[str, Any]] = {}
    common_masks: list[np.ndarray] = []
    for variant in VARIANT_ORDER:
        results[variant] = build_variant_matrix(
            variant,
            selected[variant],
            candidate_coords,
            reference_coords,
            metadata["chr1_mask_metadata"],
        )
        common_masks.append(results[variant]["common_bins"])
    for mask in common_masks[1:]:
        if not np.array_equal(mask, common_masks[0]):
            raise DisplayError("five figures do not use one identical common bin mask")
    common_bins = common_masks[0]
    expected_common_bins = int(round((1 + math.sqrt(1 + 8 * metadata["chr1_mask_metadata"]["n_common_pairs"])) / 2))
    if int(common_bins.sum()) != expected_common_bins:
        raise DisplayError(f"derived common bin count {int(common_bins.sum())} != implied evaluation count {expected_common_bins}")
    if int(common_bins.sum()) != int(config["mask_policy"]["expected_common_bins"]):
        raise DisplayError("derived common bin count does not match frozen display expectation")

    all_finite_normalized = np.concatenate([
        result["normalized"][np.isfinite(result["normalized"])] for result in results.values()
    ])
    if all_finite_normalized.size == 0 or not np.isfinite(all_finite_normalized).all():
        raise DisplayError("no finite normalized distances available for global color scale")
    global_vmax = float(np.max(all_finite_normalized))
    if not math.isfinite(global_vmax) or global_vmax <= 0:
        raise DisplayError(f"invalid global vmax: {global_vmax}")

    contrast_values = np.asarray(
        [[selected[variant]["r2_row"]["contrast"] for _ in range(1)] for variant in VARIANT_ORDER],
        dtype=np.float64,
    )
    # boxplot 使用锁定的 20 个 chr 值，而不是 chr1 display scalar。
    all_rows = per_chromosome["rows"]
    contrast_matrix = np.full((len(VARIANT_ORDER), 20), np.nan, dtype=np.float64)
    chromosomes: list[str] = []
    chromosome_names = sorted(
        {row["chromosome"] for row in all_rows},
        key=lambda value: (0, int(value[3:])) if value.startswith("chr") and value[3:].isdigit() else (1, value),
    )
    for chromosome in chromosome_names:
        if chromosome not in chromosomes:
            chromosomes.append(chromosome)
    if len(chromosomes) != 20:
        raise DisplayError(f"expected 20 chromosomes for contrast boxplot, got {len(chromosomes)}")
    for variant_index, variant in enumerate(VARIANT_ORDER):
        condition_id = config["input_lock"]["candidates"][variant]["condition_id"]
        by_chr = {
            row["chromosome"]: row["contrast"]
            for row in all_rows
            if row.get("condition_id") == condition_id
        }
        if set(by_chr) != set(chromosomes) or len(by_chr) != 20:
            raise DisplayError(f"{condition_id} does not have exactly 20 contrast rows")
        contrast_matrix[variant_index] = [float(by_chr[chromosome]) for chromosome in chromosomes]
    if not np.isfinite(contrast_matrix).all():
        raise DisplayError("contrast boxplot contains missing/non-finite R2 values")

    np.savez_compressed(
        out_dir / "contrast_values.npz",
        variant_order=np.asarray(VARIANT_ORDER),
        chromosomes=np.asarray(chromosomes),
        condition_ids=np.asarray([config["input_lock"]["candidates"][variant]["condition_id"] for variant in VARIANT_ORDER]),
        contrast=np.asarray(contrast_matrix, dtype=np.float64),
        definition=np.asarray("locked R2 per-chromosome contrast = matched - cross"),
    )
    contrast_npz_record = file_record(out_dir / "contrast_values.npz", "contrast_values.npz")
    contrast_artifacts, contrast_render = render_boxplot(contrast_matrix, chromosomes, out_dir)

    heatmap_artifacts: dict[str, Any] = {}
    heatmap_render: dict[str, Any] = {}
    npz_records: dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        result = results[variant]
        npz_path = out_dir / f"{variant}_chr1_distance_matrices.npz"
        np.savez_compressed(
            npz_path,
            positions_bp=np.asarray(POSITIONS_BP, dtype=np.int64),
            raw_distance_matrices=np.asarray(result["raw"], dtype=np.float64),
            common_raw_distance_matrices=np.asarray(result["common_raw"], dtype=np.float64),
            normalized_distance_matrices=np.asarray(result["normalized"], dtype=np.float64),
            common_bin_mask=np.asarray(result["common_bins"], dtype=bool),
            common_pair_mask=np.asarray(result["common_pair_mask"], dtype=bool),
            panel_track_names=np.asarray(result["tracks"]),
            panel_labels=np.asarray(result["labels"]),
            panel_scales=np.asarray(result["scales"], dtype=np.float64),
            reference_pooled_rms=np.asarray(result["reference_pooled_rms"], dtype=np.float64),
            candidate_pooled_rms=np.asarray(result["candidate_pooled_rms"], dtype=np.float64),
            global_vmin=np.asarray(0.0, dtype=np.float64),
            global_vmax=np.asarray(global_vmax, dtype=np.float64),
        )
        npz_records[variant] = file_record(npz_path, f"{variant} matrix NPZ")
        heatmap_artifacts[variant], heatmap_render[variant] = render_heatmap(result, global_vmax, out_dir)

    if not np.array_equal(results["C2-map"]["raw"], results["C2-free"]["raw"], equal_nan=True):
        raise DisplayError("C2-map and C2-free raw matrices differ despite locked identical coordinate SHA")
    if not np.array_equal(results["C2-map"]["common_raw"], results["C2-free"]["common_raw"], equal_nan=True):
        raise DisplayError("C2-map and C2-free common raw matrices differ")
    if not np.array_equal(results["C2-map"]["normalized"], results["C2-free"]["normalized"], equal_nan=True):
        raise DisplayError("C2-map and C2-free normalized matrices differ")
    if not np.array_equal(results["C2-map"]["scales"], results["C2-free"]["scales"]):
        raise DisplayError("C2-map and C2-free scales differ")

    lock_after, _ = lock_records(config)
    eval_inventory_after = tree_inventory(evaluation_root)
    if lock_before != lock_after:
        raise DisplayError("one or more locked fit/evaluation inputs changed during display run")
    if eval_inventory_before != eval_inventory_after:
        raise DisplayError("evaluation-r2 original inventory changed during display run")

    artifacts = {
        "contrast_values_npz": contrast_npz_record,
        "contrast_boxplot": contrast_artifacts,
        "heatmap_matrices": npz_records,
        "heatmaps": heatmap_artifacts,
    }
    validation = {
        "schema": "p9016-post020-chr1-contrast-display-validation-v1",
        "status": "PASS",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope_guard": {
            "fit_called": False,
            "r2_recomputed": False,
            "full_r2_recomputed": False,
            "phase_payload_opened": False,
            "reference_loaded_after_all_candidate_hashes": True,
            "original_evaluation_r2_modified": False,
            "no_new_inference": True,
        },
        "input_hash_lock": {
            "hash_order": hash_order,
            "before": lock_before,
            "after": lock_after,
            "unchanged": lock_before == lock_after,
        },
        "evaluation_r2_inventory": {
            "before": eval_inventory_before,
            "after": eval_inventory_after,
            "unchanged": eval_inventory_before == eval_inventory_after,
        },
        "grid": {
            "positions_bp": [int(item) for item in POSITIONS_BP],
            "n_bins": N_BINS,
            "first_position_bp": int(POSITIONS_BP[0]),
            "last_position_bp": int(POSITIONS_BP[-1]),
            "total_unordered_offdiag_pairs": N_TOTAL_PAIRS,
        },
        "mask": {
            "evaluation_manifest_mask_status": metadata["chr1_mask_metadata"].get("status"),
            "included_condition_count": len(metadata["chr1_mask_metadata"]["included_condition_ids"]),
            "evaluation_n_common_pairs": int(metadata["chr1_mask_metadata"]["n_common_pairs"]),
            "derived_common_bins": int(common_bins.sum()),
            "derived_common_pairs": int(np.triu(offdiag_mask(common_bins), k=1).sum()),
            "expected_common_bins": int(config["mask_policy"]["expected_common_bins"]),
            "expected_common_pairs": int(config["mask_policy"]["expected_common_unordered_offdiag_pairs"]),
            "same_common_mask_all_variants": True,
            "panel_coverage": {variant: results[variant]["panel_coverage"] for variant in VARIANT_ORDER},
        },
        "r2_locked_chr1": {
            variant: {
                "condition_id": selected[variant]["condition_id"],
                "contrast": selected[variant]["contrast_chr1"],
                "pairing": selected[variant]["pairing"],
                "orientation": selected[variant]["orientation"],
                "rhos": selected[variant]["rhos"],
            }
            for variant in VARIANT_ORDER
        },
        "contrast_boxplot": {
            "variant_order": list(VARIANT_ORDER),
            "chromosomes": chromosomes,
            "shape": list(contrast_matrix.shape),
            "definition": "locked R2 per-chromosome matched - cross; absolute contrast; 20 chromosome values per variant",
            "uses_ci": False,
            "uses_60_independent_repeats": False,
            "render": contrast_render,
        },
        "distance_matrices": {
            "dtype": "float64",
            "distance": "Euclidean",
            "reference_pooled_rms": {variant: results[variant]["reference_pooled_rms"] for variant in VARIANT_ORDER},
            "candidate_pooled_rms": {variant: results[variant]["candidate_pooled_rms"] for variant in VARIANT_ORDER},
            "panel_scales": {variant: results[variant]["scales"].tolist() for variant in VARIANT_ORDER},
            "global_vmin": 0.0,
            "global_vmax": global_vmax,
            "global_max_source": "all finite normalized values across reference and 5x2 candidate panels",
            "colormap": COLORMAP,
            "missing_color": MISSING_COLOR,
            "clip": False,
            "diagonal_zero_on_valid_bins": True,
            "missing_rows_columns_nan": True,
            "c2_map_free_raw_identical": True,
            "c2_map_free_common_raw_identical": True,
            "c2_map_free_normalized_identical": True,
            "c2_map_free_scales_identical": True,
            "render": heatmap_render,
        },
        "artifacts": artifacts,
        "candidate_coordinate_summaries": candidate_summaries,
        "reference_coordinate_summary": reference_summary,
    }
    validation_path = out_dir / "validation.json"
    provenance_path = out_dir / "provenance.json"
    write_json(validation_path, validation)
    write_json(
        provenance_path,
        {
            "schema": "p9016-post020-chr1-contrast-display-provenance-v1",
            "status": "PASS",
            "generated_utc": validation["generated_utc"],
            "script": file_record(Path(__file__), "display script"),
            "config": file_record(config_path, "display config"),
            "workflow": [
                "Hash-lock all five bundle2 candidate coordinates before reading the evaluation reference.",
                "Read locked release/R2/evaluation/controller JSON only; take chr1 contrast values and four-rho pairing/orientation directly from locked R2 rows.",
                "Parse candidate chr1 c01a/c01b and evaluation-only reference chr1(mat)/chr1(pat) as float64 coordinates.",
                "Recover finite/missing bins from coordinates and check every panel against evaluation_manifest.mask_metadata coverage and the 21-condition common pair count.",
                "Build one common 188-bin/17578-pair mask, pooled RMS normalize reference and candidate copies separately, and use one global vmin=0/vmax across all five heatmaps.",
                "Render one absolute matched-minus-cross boxplot and five independent 2x2 heatmaps with raw/common-raw/normalized NPZ payloads.",
                "Hash-check all locked inputs and the original evaluation-r2 inventory before/after; write only this fresh directory.",
            ],
            "template_references": {
                name: file_record(resolve_path(path), name)
                for name, path in config.get("template_references", {}).items()
            },
            "validation_path": str(validation_path.resolve()),
            "artifacts": artifacts,
            "scope_guard": validation["scope_guard"],
        },
    )
    readme_path = out_dir / "README.md"
    write_readme(
        readme_path,
        config,
        lock_before,
        lock_after,
        metadata,
        results,
        contrast_artifacts,
        heatmap_artifacts,
        global_vmax,
        validation_path,
        provenance_path,
    )
    validation.pop("validation_json", None)
    validation["validation_record"] = {
        "path": str(validation_path.resolve()),
        "hash_scope": "self-referential hash omitted; compute SHA256 from final file bytes",
    }
    validation["provenance_json"] = file_record(provenance_path, "provenance.json")
    validation["readme"] = file_record(readme_path, "README.md")
    write_json(validation_path, validation)
    terminal = {
        "schema": "p9016-post020-chr1-contrast-display-terminal-v1",
        "status": "PASS",
        "exit_code": 0,
        "command": " ".join([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path.resolve())]),
        "threads": 1,
        "output_dir": str(out_dir),
        "validation": str(validation_path),
        "provenance": str(provenance_path),
        "readme": str(readme_path),
        "common_bins": int(common_bins.sum()),
        "common_unordered_offdiag_pairs": int(np.triu(offdiag_mask(common_bins), k=1).sum()),
        "global_vmin": 0.0,
        "global_vmax": global_vmax,
        "artifacts": artifacts,
    }
    write_json(out_dir / "terminal_evidence.json", terminal)
    print(json.dumps(terminal, ensure_ascii=False, indent=2, sort_keys=True))
    return terminal


def rerender_from_saved_npz(config_path: Path) -> dict[str, Any]:
    """仅从已保存的 NPZ/JSON metadata 重绘；绝不重新打开坐标。"""
    config = read_json(config_path, "config")
    out_dir = config_path.parent.resolve()
    validation_path = out_dir / "validation.json"
    provenance_path = out_dir / "provenance.json"
    validation = read_json(validation_path, "existing validation")
    provenance = read_json(provenance_path, "existing provenance")
    if validation.get("status") != "PASS":
        raise DisplayError("saved validation is not PASS; refusing NPZ-only rerender")

    contrast_npz_path = out_dir / "contrast_values.npz"
    npz_paths = {variant: out_dir / f"{variant}_chr1_distance_matrices.npz" for variant in VARIANT_ORDER}
    npz_before = {"contrast_values": file_record(contrast_npz_path, "contrast_values.npz")}
    npz_before.update({variant: file_record(path, f"{variant} matrix NPZ") for variant, path in npz_paths.items()})
    with np.load(contrast_npz_path, allow_pickle=False) as payload:
        contrast_matrix = np.asarray(payload["contrast"], dtype=np.float64).copy()
        chromosomes = [str(item) for item in payload["chromosomes"].tolist()]
    if contrast_matrix.shape != (5, 20) or not np.isfinite(contrast_matrix).all():
        raise DisplayError("saved contrast_values.npz does not contain the locked 5x20 finite matrix")
    contrast_artifacts, contrast_render = render_boxplot(contrast_matrix, chromosomes, out_dir)

    results: dict[str, dict[str, Any]] = {}
    heatmap_artifacts: dict[str, Any] = {}
    heatmap_render: dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        with np.load(npz_paths[variant], allow_pickle=False) as payload:
            normalized = np.asarray(payload["normalized_distance_matrices"], dtype=np.float64).copy()
            labels = [str(item) for item in payload["panel_labels"].tolist()]
            tracks = [str(item) for item in payload["panel_track_names"].tolist()]
            common_bins = np.asarray(payload["common_bin_mask"], dtype=bool).copy()
            scales = np.asarray(payload["panel_scales"], dtype=np.float64).copy()
        locked = validation["r2_locked_chr1"][variant]
        results[variant] = {
            "variant": variant,
            "condition_id": locked["condition_id"],
            "pairing": locked["pairing"],
            "orientation": locked["orientation"],
            "labels": labels,
            "tracks": tracks,
            "normalized": normalized,
            "common_bins": common_bins,
            "scales": scales,
            "rhos": locked["rhos"],
            "contrast_chr1": locked["contrast"],
            "reference_pooled_rms": validation["distance_matrices"]["reference_pooled_rms"][variant],
            "candidate_pooled_rms": validation["distance_matrices"]["candidate_pooled_rms"][variant],
            "panel_coverage": validation["mask"]["panel_coverage"][variant],
        }
        heatmap_artifacts[variant], heatmap_render[variant] = render_heatmap(
            results[variant], validation["distance_matrices"]["global_vmax"], out_dir
        )

    npz_after = {"contrast_values": file_record(contrast_npz_path, "contrast_values.npz")}
    npz_after.update({variant: file_record(path, f"{variant} matrix NPZ") for variant, path in npz_paths.items()})
    if npz_before != npz_after:
        raise DisplayError("NPZ hash changed during NPZ-only rerender")

    generated_utc = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    validation["schema"] = "p9016-post020-chr1-contrast-display-validation-v2"
    validation["generated_utc"] = generated_utc
    validation["render_revision"] = {
        "mode": "NPZ/JSON-only rerender",
        "coordinates_reopened": False,
        "reference_reopened": False,
        "matrices_recomputed": False,
        "boxplot_title": "Genome-wide R2 contrast | count-selected bundle2",
        "boxplot_stem": "genomewide_contrast_boxplot",
        "heatmap_footer_y": [0.065, 0.025],
        "xlabel_footer_bbox_overlap_checked": True,
    }
    validation["npz_hash_lock"] = {"before": npz_before, "after": npz_after, "unchanged": True}
    validation["artifacts"]["contrast_boxplot"] = contrast_artifacts
    validation["artifacts"]["heatmaps"] = heatmap_artifacts
    validation["contrast_boxplot"]["render"] = contrast_render
    validation["distance_matrices"]["render"] = heatmap_render
    write_json(validation_path, validation)

    provenance["schema"] = "p9016-post020-chr1-contrast-display-provenance-v2"
    provenance["generated_utc"] = generated_utc
    provenance["script"] = file_record(Path(__file__), "display script")
    provenance["config"] = file_record(config_path, "display config")
    provenance["rerender_only"] = True
    provenance["artifacts"]["contrast_boxplot"] = contrast_artifacts
    provenance["artifacts"]["heatmaps"] = heatmap_artifacts
    provenance["workflow"] = list(provenance.get("workflow", [])) + [
        "NPZ-only render revision: read saved contrast_values.npz, five saved normalized matrices, and existing validation metadata; did not reopen reference/candidate coordinates or recompute matrices.",
        "Moved heatmap footer rows to y=0.065 and y=0.025 and asserted rendered xlabel/footer bounding boxes do not overlap.",
    ]
    write_json(provenance_path, provenance)

    metadata = {
        "chr1_mask_metadata": {"n_common_pairs": validation["mask"]["evaluation_n_common_pairs"]}
    }
    write_readme(
        out_dir / "README.md",
        config,
        validation["input_hash_lock"]["before"],
        validation["input_hash_lock"]["after"],
        metadata,
        results,
        contrast_artifacts,
        heatmap_artifacts,
        validation["distance_matrices"]["global_vmax"],
        validation_path,
        provenance_path,
    )

    # 只有所有新图和 metadata 写出后才移除旧名称。
    for old_name in ("chr1_contrast_boxplot.png", "chr1_contrast_boxplot.pdf"):
        old_path = out_dir / old_name
        if old_path.exists():
            old_path.unlink()
    validation.pop("validation_json", None)
    validation["validation_record"] = {
        "path": str(validation_path.resolve()),
        "hash_scope": "self-referential hash omitted; compute SHA256 from final file bytes",
    }
    validation["provenance_json"] = file_record(provenance_path, "provenance.json")
    validation["readme"] = file_record(out_dir / "README.md", "README.md")
    validation["old_boxplot_names_absent"] = not any((out_dir / name).exists() for name in ("chr1_contrast_boxplot.png", "chr1_contrast_boxplot.pdf"))
    write_json(validation_path, validation)

    terminal = read_json(out_dir / "terminal_evidence.json", "existing terminal evidence")
    terminal["schema"] = "p9016-post020-chr1-contrast-display-terminal-v2"
    terminal["status"] = "PASS"
    terminal["exit_code"] = 0
    terminal["command"] = " ".join([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path.resolve()), "--rerender"])
    terminal["rerender_only"] = True
    terminal["npz_hashes_unchanged"] = True
    terminal["old_boxplot_names_absent"] = validation["old_boxplot_names_absent"]
    terminal["artifacts"] = validation["artifacts"]
    terminal["validation"] = str(validation_path.resolve())
    terminal["provenance"] = str(provenance_path.resolve())
    terminal["readme"] = str((out_dir / "README.md").resolve())
    write_json(out_dir / "terminal_evidence.json", terminal)
    print(json.dumps(terminal, ensure_ascii=False, indent=2, sort_keys=True))
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rerender", action="store_true", help="redraw only from saved NPZ/JSON metadata")
    args = parser.parse_args()
    try:
        if args.rerender:
            rerender_from_saved_npz(args.config.resolve())
        else:
            run(args.config.resolve())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
