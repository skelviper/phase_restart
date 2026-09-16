#!/usr/bin/env python3
"""创建全新的 chr1 100 kb allele-A/B distance-matrix 后处理输出。

该流程仅供 evaluator 使用。它先哈希所有冻结输入，并在加载任何数值 payload 前检查阶段坐标摘要；随后验证已保存的 100 kb state arrays 和导出的坐标，只计算 chr1 A/B matrices，并使用一个 pooled RMS/color norm 渲染两张独立图。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
EXPECTED_TRACKS = tuple(
    f"c{chromosome:02d}{copy_label}"
    for chromosome in range(1, 21)
    for copy_label in ("a", "b")
)
CHROMOSOME = "chr1"
CHROMOSOME_INDEX = 0
BIN_SIZE_BP = 100_000
CHR1_LENGTH_BP = 195_471_971
EXPECTED_N_BINS = 1_955
EXPECTED_LAST_POSITION_BP = 195_400_000
COLORMAP = "coolwarm_r"
MISSING_COLOR = "#bdbdbd"
DPI = 300
FONT_SIZE_PT = 7
FIGURE_SIZE_INCHES = (4.25, 3.85)
BASE_PANEL_SIZE_INCHES = (3.0, 3.0)


class ValidationError(RuntimeError):
    """冻结 contract 或生成 artifact 无效时抛出。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """run_pipeline.array_sha256 的精确副本（source lines 92-98）。"""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def grid_state_sha256(positions: np.ndarray, chromosomes: np.ndarray) -> str:
    return hashlib.sha256(
        (array_sha256(positions) + array_sha256(chromosomes)).encode("ascii")
    ).hexdigest()


def read_json(path: Path, label: str) -> Any:
    try:
        with path.open("rt", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {label}: {path}") from exc


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = read_json(path, label)
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be a JSON object: {path}")
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value: {value}")
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(value)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def file_record(path: Path, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValidationError(f"{label} is unavailable: {path}")
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValidationError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    record: dict[str, Any] = {
        "path": str(path),
        "actual_sha256": actual,
        "size_bytes": int(path.stat().st_size),
    }
    if expected_sha256 is not None:
        record["expected_sha256"] = expected_sha256
    return record


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValidationError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_fresh(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValidationError(f"refusing to overwrite fresh output(s): {existing}")


def build_full_grid(lengths: np.ndarray, bin_size_bp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions_parts: list[np.ndarray] = []
    chromosome_parts: list[np.ndarray] = []
    offsets = [0]
    for chromosome_index, length in enumerate(lengths):
        count = (int(length) + bin_size_bp - 1) // bin_size_bp
        positions_parts.append(np.arange(count, dtype=np.int64) * bin_size_bp)
        chromosome_parts.append(np.full(count, chromosome_index, dtype=np.int32))
        offsets.append(offsets[-1] + count)
    return (
        np.concatenate(positions_parts),
        np.concatenate(chromosome_parts),
        np.asarray(offsets, dtype=np.int64),
    )


def load_track_mapping(path: Path, state_names: list[str], state_lengths: list[int]) -> dict[str, dict[str, Any]]:
    """仅使用当前 20 kb metadata 进行 native track -> chr/copy 映射。"""
    document = read_json_object(path, "track map")
    assert_equal(document.get("bin_size_bp"), 20_000, "track map metadata bin size")
    rows = document.get("tracks")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValidationError("track map must contain exactly 40 rows")
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValidationError("track map row is not an object")
        track = row.get("track_name")
        chromosome_index = row.get("chromosome_index")
        chromosome_name = row.get("chromosome_name")
        copy_index = row.get("copy_index")
        copy_label = row.get("copy_label")
        if not isinstance(track, str) or track in mapping:
            raise ValidationError(f"invalid or duplicate track map row: {row}")
        if not isinstance(chromosome_index, int) or not isinstance(copy_index, int):
            raise ValidationError(f"invalid track map indices: {row}")
        if chromosome_index < 0 or chromosome_index >= len(state_names) or copy_index not in (0, 1):
            raise ValidationError(f"track map index out of range: {row}")
        expected_track = f"c{chromosome_index + 1:02d}{'ab'[copy_index]}"
        assert_equal(track, expected_track, f"native track name for map row {track}")
        assert_equal(chromosome_name, state_names[chromosome_index], f"chromosome mapping for {track}")
        assert_equal(int(row.get("chromosome_length_bp", -1)), int(state_lengths[chromosome_index]), f"header length mapping for {track}")
        assert_equal(copy_label, "ab"[copy_index], f"copy label mapping for {track}")
        mapping[track] = {
            "track_name": track,
            "chromosome_index": chromosome_index,
            "chromosome_name": chromosome_name,
            "copy_index": copy_index,
            "copy_label": copy_label,
        }
    if tuple(mapping) != EXPECTED_TRACKS:
        raise ValidationError("track map order or inventory is not c01a,c01b,...,c20b")
    return mapping


def parse_coordinate_export(path: Path) -> dict[str, list[tuple[int, np.ndarray]]]:
    """将已哈希的 100 kb export 解析为 float64 rows。"""
    rows: dict[str, list[tuple[int, np.ndarray]]] = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split()
            if len(fields) != 5:
                raise ValidationError(f"coordinate export line {line_no} has {len(fields)} fields")
            track = fields[0]
            try:
                position = int(fields[1])
                point = np.asarray([float(value) for value in fields[2:5]], dtype=np.float64)
            except ValueError as exc:
                raise ValidationError(f"non-numeric coordinate export line {line_no}") from exc
            if position < 0 or not np.isfinite(point).all():
                raise ValidationError(f"negative or non-finite coordinate export line {line_no}")
            rows.setdefault(track, []).append((position, point))
    return rows


def verify_coordinate_export(
    coordinate_rows: dict[str, list[tuple[int, np.ndarray]]],
    mapping: dict[str, dict[str, Any]],
    state_positions: np.ndarray,
    state_chromosomes: np.ndarray,
    state_coordinates: np.ndarray,
    state_names: list[str],
) -> dict[str, Any]:
    if set(coordinate_rows) != set(mapping):
        raise ValidationError(
            f"coordinate export track inventory mismatch: missing={sorted(set(mapping) - set(coordinate_rows))}, "
            f"unexpected={sorted(set(coordinate_rows) - set(mapping))}"
        )
    total_rows = 0
    per_track: dict[str, Any] = {}
    for track in EXPECTED_TRACKS:
        row = mapping[track]
        observed = coordinate_rows[track]
        total_rows += len(observed)
        chromosome_index = int(row["chromosome_index"])
        copy_index = int(row["copy_index"])
        state_indices = np.flatnonzero(state_chromosomes == chromosome_index)
        expected_positions = np.asarray(state_positions[state_indices], dtype=np.int64)
        if len(observed) != len(expected_positions):
            raise ValidationError(f"{track} row count mismatch")
        observed_positions = np.asarray([item[0] for item in observed], dtype=np.int64)
        order = np.argsort(observed_positions, kind="stable")
        sorted_positions = observed_positions[order]
        if not np.array_equal(sorted_positions, expected_positions):
            raise ValidationError(f"{track} exported positions do not match the 100 kb state grid")
        observed_points = np.stack([observed[index][1] for index in order], axis=0)
        expected_points = np.asarray(state_coordinates[copy_index, state_indices], dtype=np.float64)
        if not np.allclose(observed_points, expected_points, rtol=0.0, atol=1e-12):
            raise ValidationError(f"{track} exported coordinates do not match saved state coordinates")
        per_track[track] = {
            "chromosome": state_names[chromosome_index],
            "chromosome_index": chromosome_index,
            "copy_index": copy_index,
            "n_rows": len(observed),
            "positions_match_state": True,
            "coordinates_match_state_atol_1e-12": True,
        }
    if total_rows != int(state_coordinates.shape[1] * 2):
        raise ValidationError(f"coordinate export row total mismatch: {total_rows}")
    return {
        "n_tracks": len(coordinate_rows),
        "n_rows": total_rows,
        "all_tracks_match_state": True,
        "per_track": per_track,
    }


def finite_span(points: np.ndarray) -> dict[str, Any]:
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return {"n_finite_bins": 0, "min": None, "max": None, "span": None}
    minimum = finite.min(axis=0)
    maximum = finite.max(axis=0)
    return {
        "n_finite_bins": int(len(finite)),
        "min": [float(value) for value in minimum],
        "max": [float(value) for value in maximum],
        "span": [float(value) for value in maximum - minimum],
    }


def matrix_summary(matrix: np.ndarray, pair_mask: np.ndarray, finite_bins: np.ndarray) -> dict[str, Any]:
    upper = np.triu(pair_mask, k=1)
    values = np.asarray(matrix[upper], dtype=np.float64)
    diagonal = np.diag(matrix)
    finite = np.isfinite(matrix)
    symmetric = bool(
        np.array_equal(finite, finite.T)
        and np.allclose(matrix[finite], matrix.T[finite], rtol=0.0, atol=1e-12)
    )
    diagonal_zero = bool(np.all(diagonal[finite_bins] == 0.0) and np.all(~np.isfinite(diagonal[~finite_bins])))
    return {
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "symmetric_atol_1e-12": symmetric,
        "nonnegative_finite_entries": bool(len(values) > 0 and np.isfinite(values).all() and np.all(values >= 0.0)),
        "diagonal_zero_on_finite_bins": diagonal_zero,
        "diagonal_excluded_from_pairs": True,
        "finite_bin_count": int(finite_bins.sum()),
        "finite_offdiagonal_unordered_pair_count": int(len(values)),
        "finite_min_offdiagonal": float(np.min(values)),
        "finite_max_offdiagonal": float(np.max(values)),
    }


def png_dimensions_and_dpi(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValidationError(f"invalid PNG signature: {path}")
    width = height = None
    dpi_x = dpi_y = None
    offset = 8
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data_start = offset + 8
        chunk_data_end = chunk_data_start + length
        if chunk_data_end + 4 > len(data):
            raise ValidationError(f"truncated PNG chunk: {path}")
        chunk_data = data[chunk_data_start:chunk_data_end]
        if chunk_type == b"IHDR" and length >= 8:
            width, height = struct.unpack(">II", chunk_data[:8])
        elif chunk_type == b"pHYs" and length >= 9 and chunk_data[8] == 1:
            pixels_per_m_x, pixels_per_m_y = struct.unpack(">II", chunk_data[:8])
            dpi_x = float(pixels_per_m_x) * 0.0254
            dpi_y = float(pixels_per_m_y) * 0.0254
        offset = chunk_data_end + 4
        if chunk_type == b"IEND":
            break
    if width is None or height is None:
        raise ValidationError(f"PNG lacks IHDR dimensions: {path}")
    if dpi_x is None or dpi_y is None:
        raise ValidationError(f"PNG lacks pHYs DPI metadata: {path}")
    if abs(dpi_x - DPI) > 0.2 or abs(dpi_y - DPI) > 0.2:
        raise ValidationError(f"PNG DPI mismatch for {path}: {dpi_x}, {dpi_y}")
    return {
        "dimensions_px": [int(width), int(height)],
        "dpi": [dpi_x, dpi_y],
        "dpi_within_0.2_of_300": True,
    }


def png_pixel_check(path: Path) -> dict[str, Any]:
    import matplotlib.image as mpimg

    image = np.asarray(mpimg.imread(path), dtype=np.float64)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValidationError(f"unexpected decoded PNG shape for {path}: {image.shape}")
    rgb = image[..., :3]
    if not np.isfinite(rgb).all():
        raise ValidationError(f"decoded PNG contains non-finite pixels: {path}")
    channel_range = float(np.max(rgb) - np.min(rgb))
    if channel_range <= 0.01:
        raise ValidationError(f"decoded PNG is blank/constant: {path}")
    return {
        "decoded_shape": list(image.shape),
        "finite_rgb_pixels": True,
        "rgb_min": float(np.min(rgb)),
        "rgb_max": float(np.max(rgb)),
        "rgb_range": channel_range,
        "nonconstant_pixels": True,
    }


def pdf_page_size(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    matches = re.findall(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", data)
    if not matches:
        raise ValidationError(f"PDF MediaBox unavailable: {path}")
    width, height = (float(matches[0][0]), float(matches[0][1]))
    expected = [FIGURE_SIZE_INCHES[0] * 72.0, FIGURE_SIZE_INCHES[1] * 72.0]
    if not np.allclose([width, height], expected, rtol=0.0, atol=1e-5):
        raise ValidationError(f"PDF page size mismatch for {path}: {(width, height)}")
    return {
        "page_size_points": [width, height],
        "expected_page_size_points": expected,
        "page_size_matches_canvas": True,
    }


def render_figure(
    normalized_matrix: np.ndarray,
    positions_bp: np.ndarray,
    color_vmax: float,
    title: str,
    png_path: Path,
    pdf_path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.linewidth": 0.6,
        }
    )
    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad(MISSING_COLOR)
    if cmap.name != COLORMAP:
        raise ValidationError(f"unexpected colormap: {cmap.name}")
    norm = Normalize(vmin=0.0, vmax=color_vmax, clip=False)
    positions_mb = positions_bp.astype(np.float64) / 1_000_000.0
    half_bin_mb = BIN_SIZE_BP / 2_000_000.0
    extent = (
        float(positions_mb[0] - half_bin_mb),
        float(positions_mb[-1] + half_bin_mb),
        float(positions_mb[0] - half_bin_mb),
        float(positions_mb[-1] + half_bin_mb),
    )
    ticks = np.asarray([0.0, 50.0, 100.0, 150.0, 195.0], dtype=np.float64)

    fig = plt.figure(figsize=FIGURE_SIZE_INCHES, dpi=DPI)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.055],
        left=0.14,
        right=0.90,
        bottom=0.12,
        top=0.89,
        wspace=0.14,
    )
    axis = fig.add_subplot(grid[0, 0])
    image = axis.imshow(
        normalized_matrix,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        extent=extent,
        cmap=cmap,
        norm=norm,
    )
    axis.set_title(title, pad=4)
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.tick_params(width=0.6, length=2.5, pad=2)
    axis.set_xlabel("Genomic position (Mb)")
    axis.set_ylabel("Genomic position (Mb)")
    colorbar_axis = fig.add_subplot(grid[0, 1])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Distance / shared RMS", fontsize=FONT_SIZE_PT, labelpad=4)
    colorbar.ax.tick_params(labelsize=FONT_SIZE_PT, width=0.6, length=2.5)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=DPI, format="png")
    fig.savefig(pdf_path, dpi=DPI, format="pdf")
    plt.close(fig)

    png_meta = png_dimensions_and_dpi(png_path)
    pixel_meta = png_pixel_check(png_path)
    pdf_meta = pdf_page_size(pdf_path)
    return {
        "title": title,
        "colormap": COLORMAP,
        "missing_color": MISSING_COLOR,
        "normalize": {"vmin": 0.0, "vmax": float(color_vmax), "clip": False},
        "figure_size_inches": list(FIGURE_SIZE_INCHES),
        "base_panel_size_inches": list(BASE_PANEL_SIZE_INCHES),
        "font_size_pt": FONT_SIZE_PT,
        "dpi": DPI,
        "axis_extent_mb": list(extent),
        "ticks_mb": ticks.tolist(),
        "origin": "lower",
        "interpolation": "nearest",
        "aspect": "equal",
        "colorbar_label": "Distance / shared RMS",
        "png": {**png_meta, **pixel_meta},
        "pdf": pdf_meta,
    }


def output_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def write_readme(
    path: Path,
    validation: dict[str, Any],
    validation_sha256: str,
    manifest_path: Path,
) -> None:
    inputs = validation["inputs"]
    grid = validation["grid"]["chr1"]
    arrays = validation["state_arrays"]
    matrices = validation["matrices"]
    render = validation["render"]
    outputs = validation["artifacts"]
    lines = [
        "# P9016 chr1：100 kb 等位拷贝 A/B 距离矩阵",
        "",
        "**状态：** `PASS`。这是 028 独立 GPU 运行的全新、仅供评价的后处理目录；没有重训练、重跑 GPU、拟合、读取 raw contacts、phase 或 reference，也没有写入旧结果目录。",
        "",
        "## 冻结输入与门控",
        "",
        f"- 配置在矩阵计算前冻结：`{validation['config']['path']}`，SHA256 `{validation['config']['actual_sha256']}`；运行环境为 `analysis` conda。",
        f"- 实际 100 kb stage：`{inputs['stage']['path']}`，SHA256 `{inputs['stage']['actual_sha256']}`；`candidate_id=random_joint`，状态 `{validation['stage']['status']}`，accepted 迭代次数 `{validation['stage']['accepted_iterations']}`。",
        f"- 坐标导出：`{inputs['coordinates']['path']}`；在读取任何数值前已核对其 SHA256 `{inputs['coordinates']['actual_sha256']}` 与 stage `/final_coordinates/sha256` 一致。分析结束后 source hash 仍一致。",
        f"- 100 kb state JSON/NPZ：`{inputs['state_json']['path']}` / `{inputs['state_npz']['path']}`；state 数组按 `run_pipeline.array_sha256`（dtype + shape + C-order bytes）复核。",
        f"- 1 Mb selection 仅用于记录既定候选：`{inputs['selection_1m']['path']}`，selected 候选 `{validation['selection']['selected_candidate']}`；没有据此重选 100 kb 结果。",
        "",
        "## 网格与 native 映射",
        "",
        f"- 使用 state 中的真实完整 header/grid，而不是 20 kb track-map 网格：chr1 header length `{grid['header_length_bp']:,} bp`；位置为 `{grid['first_position_bp']:,}..{grid['last_position_bp']:,} bp`，step `{grid['step_bp']:,} bp`，共 `{grid['n_bins']}` 个 bin。",
        f"- state 全局坐标形状 `{arrays['coordinates']['shape']}`；全局完整 grid 与 header 一致。chr1 两个 copy 的有限 bin 数为 `{matrices['common_finite_bin_count']}/{grid['n_bins']}`，预期全部 `{grid['expected_full_finite_bins']}`。",
        "- 当前 `provenance/track_map.json` 的 20 kb positions/counts 未用于建网格；只使用 track 到 chromosome/copy 的映射：`c01a=copy_index 0=Allele A`，`c01b=copy_index 1=Allele B`。A/B 是 gauge labels，不是 maternal/paternal 判定。",
        f"- 导出的 40 个 native tracks、`{validation['coordinate_export']['n_rows']}` 行均与 100 kb state 坐标逐 track 核对（绝对误差 <=1e-12）。",
        "",
        "## 指标与颜色",
        "",
        "- 每个矩阵的原始定义为 `D_a[i,j] = float64` 三维欧氏距离；用 `scipy.spatial.distance.cdist`，只对 chr1 1955 个 100 kb 点计算，不分配全基因组 N²。",
        f"- 观察单位是共同有限 bins 上的无序非对角距离对 `i<j`；机器计算的距离对分母为 `{matrices['offdiag_unordered_pair_count']}`，`n_bio_cells=1`。对角线显示为 0，但排除 RMS 和距离对计数。",
        f"- 单一 shared RMS：`s = sqrt(mean(concat(D_A[i<j]^2, D_B[i<j]^2))) = {matrices['shared_rms']:.16g}`；A/B 都渲染为 `D/s`。normalized pooled off-diagonal RMS 为 `{matrices['normalized_pooled_offdiag_rms']:.16g}`。",
        f"- 两张图使用完全相同的 `Normalize(vmin=0, vmax={validation['color_norm']['vmax']:.16g}, clip=False)`，vmax 是两张归一化矩阵的全局有限最大值；不做百分位裁剪，也不做逐等位拷贝归一化。",
        f"- colormap 是 `{COLORMAP}`（红=近，蓝=远），missing color `{MISSING_COLOR}`；图像使用 `origin=lower`、`interpolation=nearest`、`aspect=equal`，轴为绝对基因组位置 (Mb)，bin 锚点采用 +/- half-bin extent。",
        "- 本结果是 SNP-free 模型距离的可视化，不是 observed Hi-C 距离；不使用旧 reference crop、`a=pat` assignment、reference comparison/interpolation，也不提出 L2、accuracy 或新的 accuracy 结论。",
        "",
        "## 输出",
        "",
        f"- 等位拷贝 A PNG：`{outputs['allele_A_png']['path']}`，SHA256 `{outputs['allele_A_png']['sha256']}`。",
        f"- 等位拷贝 A PDF：`{outputs['allele_A_pdf']['path']}`，SHA256 `{outputs['allele_A_pdf']['sha256']}`。标题为 `chr1 | 等位拷贝 A | 100 kb`。",
        f"- 等位拷贝 B PNG：`{outputs['allele_B_png']['path']}`，SHA256 `{outputs['allele_B_png']['sha256']}`。",
        f"- 等位拷贝 B PDF：`{outputs['allele_B_pdf']['path']}`，SHA256 `{outputs['allele_B_pdf']['sha256']}`。标题为 `chr1 | 等位拷贝 B | 100 kb`。",
        f"- raw NPZ：`{outputs['raw_npz']['path']}`，SHA256 `{outputs['raw_npz']['sha256']}`；保存 float64 `raw_matrices`、positions、共同 mask、shared RMS、color limits、allele IDs/native IDs。normalized matrices 按规则在读取/绘图时计算，未重复存储。",
        f"- 机器验证：`{validation['validation_file']['path']}`，SHA256 `{validation_sha256}`；preflight：`{validation['preflight']['path']}`；终端输出日志：`{validation['run_log']['path']}`。最终文件清单：`{manifest_path}`。",
        f"- 两张 PNG 均验证为 300 dpi、非空像素；figure canvas `{FIGURE_SIZE_INCHES[0]:g} x {FIGURE_SIZE_INCHES[1]:g}` 英寸，base matrix panel 约 3 x 3 英寸，统一 7 pt 文字。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    if os.environ.get("CONDA_DEFAULT_ENV") != "analysis":
        raise ValidationError(
            f"this computation must run in conda analysis; CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV')!r}"
        )
    config = read_json_object(CONFIG_PATH, "frozen config")
    assert_equal(config.get("status"), "config_frozen_before_compute", "config status")
    assert_equal(config.get("source_stage"), "100k", "config source stage")
    assert_equal(config.get("run_id"), "028-20260913_151456-020-gpu-independent", "config run id")
    assert_equal(config.get("native_mapping", {}).get("track_map_is_20kb_metadata"), True, "track map metadata guard")
    assert_equal(config.get("scope_guard", {}).get("fit"), False, "fit scope guard")
    assert_equal(config.get("scope_guard", {}).get("gpu_eval"), False, "GPU evaluation scope guard")
    assert_equal(config.get("scope_guard", {}).get("reference_3dg_opened"), False, "reference scope guard")

    input_cfg = config["inputs"]
    paths = {
        key: Path(value["path"]).resolve()
        for key, value in input_cfg.items()
        if key != "array_sha256_reference"
    }
    records = {
        key: file_record(paths[key], key, value.get("expected_sha256"))
        for key, value in input_cfg.items()
        if key != "array_sha256_reference"
    }

    # 这是 numeric 计算前的关键 gate：先将 coordinate bytes 与 stage identity 比较。
    stage = read_json_object(paths["stage"], "100 kb stage")
    assert_equal(stage.get("candidate_id"), "random_joint", "stage candidate")
    assert_equal(stage.get("stage"), "100k", "stage label")
    assert_equal(stage.get("bin_size_bp"), BIN_SIZE_BP, "stage bin size")
    assert_equal(stage.get("status"), "budget_not_converged", "stage status")
    assert_equal(stage.get("budget_not_converged"), True, "stage budget flag")
    assert_equal(stage.get("fit", {}).get("nit"), 240, "stage accepted iteration budget")
    final_coordinates = stage.get("final_coordinates")
    if not isinstance(final_coordinates, dict):
        raise ValidationError("stage lacks final_coordinates")
    assert_equal(Path(final_coordinates.get("path", "")).resolve(), paths["coordinates"], "stage coordinate path")
    assert_equal(final_coordinates.get("sha256"), records["coordinates"]["actual_sha256"], "stage coordinate digest")
    assert_equal(final_coordinates.get("full_grid"), True, "stage full-grid flag")
    assert_equal(final_coordinates.get("n_tracks"), 40, "stage track count")
    assert_equal(final_coordinates.get("n_beads"), 52_700, "stage bead count")
    assert_equal(stage.get("resume_state", {}).get("coordinates_sha256"), "5b322672ffc35bfdba477ea14f0f023438a340a23e521267778a6cf7ff379ab6", "stage state coordinate array hash")
    assert_equal(stage.get("resume_state", {}).get("theta_sha256"), "1813ea65bcd5305cb0eb7e89850972158c80fbdecbdced2308a1ef72344cf7fc", "stage state theta hash")
    assert_equal(stage.get("resume_state", {}).get("positions_sha256"), "a412e29dd418fb97e2d31fed1a32aac3fcd98ab89d9f1447553c23fecc04ae5f", "stage state positions hash")
    assert_equal(stage.get("resume_state", {}).get("grid_state_sha256"), "26e4a208808d10f4e3daafc96a6926ab2e36e0a4f6aab3be179a8af956ac2cbb", "stage grid state hash")

    selection = read_json_object(paths["selection_1m"], "1 Mb selection")
    assert_equal(selection.get("selected_candidate"), "random_joint", "selected candidate")
    assert_equal(selection.get("phase_or_reference_opened"), False, "selection phase/reference flag")
    assert_equal(selection.get("config_sha256"), "14deef2734fb0b13798d394d6bf0336c23c6e2a3958ec60f532c48f959e6ae7e", "selection config hash")
    assert_equal(selection.get("input_sha256"), "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa", "selection input hash")

    state_meta = read_json_object(paths["state_json"], "100 kb state metadata")
    assert_equal(state_meta.get("candidate_id"), "random_joint", "state candidate")
    assert_equal(state_meta.get("stage"), "100k", "state stage")
    assert_equal(state_meta.get("bin_size_bp"), BIN_SIZE_BP, "state bin size")
    assert_equal(state_meta.get("phase_or_reference_opened"), False, "state phase/reference flag")
    assert_equal(state_meta.get("coordinates_shape"), [2, 26_350, 3], "state coordinate shape metadata")
    assert_equal(state_meta.get("theta_shape"), [158_101], "state theta shape metadata")
    assert_equal(state_meta.get("coordinates_sha256"), "5b322672ffc35bfdba477ea14f0f023438a340a23e521267778a6cf7ff379ab6", "state coordinate hash metadata")
    assert_equal(state_meta.get("theta_sha256"), "1813ea65bcd5305cb0eb7e89850972158c80fbdecbdced2308a1ef72344cf7fc", "state theta hash metadata")
    assert_equal(state_meta.get("positions_sha256"), "a412e29dd418fb97e2d31fed1a32aac3fcd98ab89d9f1447553c23fecc04ae5f", "state positions hash metadata")
    assert_equal(state_meta.get("grid_state_sha256"), "26e4a208808d10f4e3daafc96a6926ab2e36e0a4f6aab3be179a8af956ac2cbb", "state grid state hash metadata")
    state_names = list(state_meta["chromosome_names"])
    state_lengths = [int(value) for value in state_meta["chromosome_lengths"]]
    assert_equal(state_names[CHROMOSOME_INDEX], CHROMOSOME, "state chr1 header name")
    assert_equal(state_lengths[CHROMOSOME_INDEX], CHR1_LENGTH_BP, "state chr1 header length")
    if len(state_names) != 20 or len(state_lengths) != 20:
        raise ValidationError("state header must contain 20 chromosomes")

    track_map = load_track_mapping(paths["track_map"], state_names, state_lengths)
    assert_equal(track_map["c01a"]["copy_index"], 0, "c01a copy mapping")
    assert_equal(track_map["c01b"]["copy_index"], 1, "c01b copy mapping")
    assert_equal(track_map["c01a"]["chromosome_index"], 0, "c01a chromosome mapping")
    assert_equal(track_map["c01b"]["chromosome_index"], 0, "c01b chromosome mapping")

    output_cfg = config["outputs"]
    output_paths = {
        "raw_npz": HERE / output_cfg["raw_npz"],
        "allele_A_png": HERE / output_cfg["allele_A_png"],
        "allele_A_pdf": HERE / output_cfg["allele_A_pdf"],
        "allele_B_png": HERE / output_cfg["allele_B_png"],
        "allele_B_pdf": HERE / output_cfg["allele_B_pdf"],
        "validation": HERE / output_cfg["validation"],
        "readme": HERE / output_cfg["readme"],
        "preflight": HERE / output_cfg["preflight_log"],
        "run_log": HERE / output_cfg["run_log"],
    }
    assert_fresh([path for key, path in output_paths.items() if key != "run_log"])
    for path in (output_paths["allele_A_png"], output_paths["allele_A_pdf"], output_paths["allele_B_png"], output_paths["allele_B_pdf"]):
        path.parent.mkdir(parents=True, exist_ok=True)

    preflight = {
        "schema": "chr1_100kb_preflight.v1",
        "status": "PASS_before_numeric_payload_load",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": file_record(CONFIG_PATH, "frozen config"),
        "inputs": records,
        "stage_coordinate_digest_matches_before_numeric_read": True,
        "stage_status": stage["status"],
        "stage_accepted_iterations": int(stage["fit"]["nit"]),
        "selected_candidate": selection["selected_candidate"],
        "state_metadata_headers_loaded": True,
        "track_map_mapping_only": True,
        "numeric_state_npz_loaded": False,
        "coordinate_export_numeric_rows_loaded": False,
        "matrix_computation_started": False,
        "scope_guard": config["scope_guard"],
    }
    write_json(output_paths["preflight"], preflight)

    # 只有通过上面的 coordinate byte/hash gate 后才开始加载 numeric state。
    with np.load(paths["state_npz"], allow_pickle=False) as archive:
        required_arrays = {"theta", "coordinates", "positions", "chromosomes"}
        if not required_arrays.issubset(set(archive.files)):
            raise ValidationError(f"state NPZ lacks required arrays: {sorted(required_arrays - set(archive.files))}")
        theta = np.asarray(archive["theta"], dtype=np.float64).copy()
        coordinates = np.asarray(archive["coordinates"], dtype=np.float64).copy()
        positions = np.asarray(archive["positions"], dtype=np.int64).copy()
        chromosomes = np.asarray(archive["chromosomes"], dtype=np.int32).copy()

    if theta.shape != (158_101,) or coordinates.shape != (2, 26_350, 3):
        raise ValidationError(f"unexpected state array shapes: theta={theta.shape}, coordinates={coordinates.shape}")
    if positions.shape != (26_350,) or chromosomes.shape != (26_350,):
        raise ValidationError(f"unexpected state grid array shapes: positions={positions.shape}, chromosomes={chromosomes.shape}")
    if not np.isfinite(theta).all() or not np.isfinite(coordinates).all():
        raise ValidationError("state theta or coordinates contain non-finite values")
    array_hashes = {
        "theta": {"dtype": str(theta.dtype), "shape": list(theta.shape), "sha256": array_sha256(theta), "expected_sha256": state_meta["theta_sha256"]},
        "coordinates": {"dtype": str(coordinates.dtype), "shape": list(coordinates.shape), "sha256": array_sha256(coordinates), "expected_sha256": state_meta["coordinates_sha256"]},
        "positions": {"dtype": str(positions.dtype), "shape": list(positions.shape), "sha256": array_sha256(positions), "expected_sha256": state_meta["positions_sha256"]},
        "chromosomes": {"dtype": str(chromosomes.dtype), "shape": list(chromosomes.shape), "sha256": array_sha256(chromosomes)},
    }
    for name, record in array_hashes.items():
        expected_hash = record.get("expected_sha256")
        if expected_hash is not None and record["sha256"] != expected_hash:
            raise ValidationError(f"state {name} array hash mismatch")
    assert_equal(grid_state_sha256(positions, chromosomes), state_meta["grid_state_sha256"], "computed state grid state hash")
    expected_positions, expected_chromosomes, expected_offsets = build_full_grid(np.asarray(state_lengths, dtype=np.int64), BIN_SIZE_BP)
    if not np.array_equal(positions, expected_positions) or not np.array_equal(chromosomes, expected_chromosomes):
        raise ValidationError("state positions/chromosomes do not describe the complete 100 kb header-order grid")
    if len(expected_positions) != 26_350:
        raise ValidationError("state-derived full grid locus count changed")

    coordinate_rows = parse_coordinate_export(paths["coordinates"])
    coordinate_export_validation = verify_coordinate_export(
        coordinate_rows, track_map, positions, chromosomes, coordinates, state_names
    )

    chr1_indices = np.flatnonzero(chromosomes == CHROMOSOME_INDEX)
    chr1_positions = np.asarray(positions[chr1_indices], dtype=np.int64)
    expected_chr1_positions = np.arange(0, CHR1_LENGTH_BP, BIN_SIZE_BP, dtype=np.int64)
    if len(chr1_positions) != EXPECTED_N_BINS:
        raise ValidationError(f"chr1 state-derived bin count mismatch: {len(chr1_positions)}")
    if not np.array_equal(chr1_positions, expected_chr1_positions):
        raise ValidationError("chr1 state-derived positions differ from the frozen full-grid rule")
    if int(chr1_positions.min()) != 0 or int(chr1_positions.max()) != EXPECTED_LAST_POSITION_BP:
        raise ValidationError("chr1 grid min/max mismatch")
    if not np.all(np.diff(chr1_positions) == BIN_SIZE_BP):
        raise ValidationError("chr1 grid step mismatch")
    points = np.asarray(coordinates[:, chr1_indices, :], dtype=np.float64)
    finite_bins_by_allele = np.isfinite(points).all(axis=2)
    common_finite_bins = finite_bins_by_allele.all(axis=0)
    if not np.array_equal(finite_bins_by_allele[0], finite_bins_by_allele[1]):
        raise ValidationError("A/B finite-bin masks differ")
    common_pair_mask = np.outer(common_finite_bins, common_finite_bins)
    np.fill_diagonal(common_pair_mask, False)
    offdiag_upper = np.triu(common_pair_mask, k=1)
    offdiag_pair_count = int(offdiag_upper.sum())
    if offdiag_pair_count != EXPECTED_N_BINS * (EXPECTED_N_BINS - 1) // 2:
        raise ValidationError("chr1 off-diagonal pair denominator is not the full 1955-bin count")

    from scipy.spatial.distance import cdist

    raw_matrices = np.empty((2, EXPECTED_N_BINS, EXPECTED_N_BINS), dtype=np.float64)
    for allele_index in (0, 1):
        distance = np.asarray(cdist(points[allele_index], points[allele_index], metric="euclidean"), dtype=np.float64)
        distance[~common_pair_mask] = np.nan
        diagonal_indices = np.arange(EXPECTED_N_BINS)
        distance[diagonal_indices, diagonal_indices] = np.where(common_finite_bins, 0.0, np.nan)
        raw_matrices[allele_index] = distance
    if not np.isfinite(raw_matrices[:, offdiag_upper]).all():
        raise ValidationError("raw matrix has non-finite common off-diagonal values")
    pooled_offdiag = np.concatenate(
        [np.asarray(raw_matrices[allele_index][offdiag_upper], dtype=np.float64) for allele_index in (0, 1)]
    )
    shared_rms = float(np.sqrt(np.mean(pooled_offdiag * pooled_offdiag, dtype=np.float64)))
    if not math.isfinite(shared_rms) or shared_rms <= 0.0:
        raise ValidationError(f"shared RMS is not positive finite: {shared_rms}")
    normalized_matrices = raw_matrices / shared_rms
    normalized_matrices[:, ~common_pair_mask] = np.nan
    diagonal_indices = np.arange(EXPECTED_N_BINS)
    normalized_matrices[:, diagonal_indices, diagonal_indices] = np.where(common_finite_bins, 0.0, np.nan)
    normalized_pooled_offdiag = np.concatenate(
        [np.asarray(normalized_matrices[allele_index][offdiag_upper], dtype=np.float64) for allele_index in (0, 1)]
    )
    normalized_pooled_rms = float(np.sqrt(np.mean(normalized_pooled_offdiag * normalized_pooled_offdiag, dtype=np.float64)))
    if not math.isfinite(normalized_pooled_rms) or not np.isclose(normalized_pooled_rms, 1.0, rtol=0.0, atol=1e-12):
        raise ValidationError(f"normalized pooled off-diagonal RMS is not approximately one: {normalized_pooled_rms}")
    color_vmax = float(np.max(normalized_pooled_offdiag))
    if not math.isfinite(color_vmax) or color_vmax <= 0.0:
        raise ValidationError(f"invalid global normalized color maximum: {color_vmax}")
    if np.any(pooled_offdiag < 0.0) or np.any(~np.isfinite(pooled_offdiag)):
        raise ValidationError("raw off-diagonal distances are negative or non-finite")

    matrix_summaries = {
        "A": matrix_summary(raw_matrices[0], common_pair_mask, finite_bins_by_allele[0]),
        "B": matrix_summary(raw_matrices[1], common_pair_mask, finite_bins_by_allele[1]),
    }
    if not all(summary["symmetric_atol_1e-12"] for summary in matrix_summaries.values()):
        raise ValidationError("raw matrices are not symmetric")
    if not all(summary["diagonal_zero_on_finite_bins"] for summary in matrix_summaries.values()):
        raise ValidationError("raw matrix diagonal validation failed")

    np.savez_compressed(
        output_paths["raw_npz"],
        raw_matrices=np.asarray(raw_matrices, dtype=np.float64),
        positions_bp=np.asarray(chr1_positions, dtype=np.int64),
        common_bin_mask=np.asarray(common_finite_bins, dtype=bool),
        common_pair_mask=np.asarray(common_pair_mask, dtype=bool),
        shared_rms=np.asarray(shared_rms, dtype=np.float64),
        color_norm=np.asarray([0.0, color_vmax], dtype=np.float64),
        allele_ids=np.asarray(["A", "B"]),
        native_track_ids=np.asarray(["c01a", "c01b"]),
        chromosome=np.asarray(CHROMOSOME),
        bin_size_bp=np.asarray(BIN_SIZE_BP, dtype=np.int64),
        n_bio_cells=np.asarray(1, dtype=np.int64),
        coordinate_units=np.asarray("dimensionless_R1_coordinates"),
        colormap=np.asarray(COLORMAP),
        color_missing=np.asarray(MISSING_COLOR),
        metric_definition=np.asarray("float64 3D Euclidean distance; unordered off-diagonal i<j"),
    )

    render_records = {
        "A": render_figure(
            normalized_matrices[0],
            chr1_positions,
            color_vmax,
            "chr1 | Allele A | 100 kb",
            output_paths["allele_A_png"],
            output_paths["allele_A_pdf"],
        ),
        "B": render_figure(
            normalized_matrices[1],
            chr1_positions,
            color_vmax,
            "chr1 | Allele B | 100 kb",
            output_paths["allele_B_png"],
            output_paths["allele_B_pdf"],
        ),
    }
    if render_records["A"]["normalize"] != render_records["B"]["normalize"]:
        raise ValidationError("A/B image color norm metadata differ")
    if render_records["A"]["colormap"] != COLORMAP or render_records["B"]["colormap"] != COLORMAP:
        raise ValidationError("render colormap mismatch")

    source_hash_after = {
        key: sha256_file(paths[key])
        for key in ("coordinates", "state_json", "state_npz", "stage", "selection_1m", "track_map", "run_config")
    }
    source_hash_unchanged = {
        key: source_hash_after[key] == records[key]["actual_sha256"] for key in source_hash_after
    }
    if not all(source_hash_unchanged.values()):
        raise ValidationError(f"frozen source hash changed during analysis: {source_hash_unchanged}")

    artifacts = {
        "raw_npz": output_record(output_paths["raw_npz"]),
        "allele_A_png": output_record(output_paths["allele_A_png"]),
        "allele_A_pdf": output_record(output_paths["allele_A_pdf"]),
        "allele_B_png": output_record(output_paths["allele_B_png"]),
        "allele_B_pdf": output_record(output_paths["allele_B_pdf"]),
    }
    validation = {
        "schema": "chr1_100kb_allele_distance_matrices.validation.v1",
        "status": "PASS",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": sys.executable,
            "numpy": np.__version__,
        },
        "config": file_record(CONFIG_PATH, "frozen config"),
        "script": file_record(Path(__file__), "postprocessing script"),
        "inputs": records,
        "stage": {
            "candidate_id": stage["candidate_id"],
            "stage_label": stage["stage"],
            "bin_size_bp": stage["bin_size_bp"],
            "status": stage["status"],
            "budget_not_converged": bool(stage["budget_not_converged"]),
            "accepted_iterations": int(stage["fit"]["nit"]),
            "full_grid": bool(final_coordinates["full_grid"]),
            "n_tracks": int(final_coordinates["n_tracks"]),
            "n_beads": int(final_coordinates["n_beads"]),
            "coordinates_sha256": final_coordinates["sha256"],
        },
        "selection": {
            "selected_candidate": selection["selected_candidate"],
            "criterion": selection.get("criterion"),
            "status": selection.get("status"),
            "phase_or_reference_opened": bool(selection["phase_or_reference_opened"]),
        },
        "state_arrays": array_hashes,
        "state_header": {
            "chromosome_names": state_names,
            "chromosome_lengths_bp": state_lengths,
            "n_chromosomes": len(state_names),
            "n_loci": int(len(positions)),
            "coordinate_axis_order": stage["initialization"]["coordinate_axis_order"],
            "full_grid_from_state": True,
            "expected_offsets": expected_offsets.tolist(),
            "computed_chromosomes_sha256": array_sha256(chromosomes),
            "computed_grid_state_sha256": grid_state_sha256(positions, chromosomes),
        },
        "native_mapping": {
            "track_map_metadata_bin_size_bp": 20_000,
            "track_map_positions_used": False,
            "mapping_used_only": "native track -> chromosome/copy",
            "c01a": {"track": "c01a", "chromosome": CHROMOSOME, "chromosome_index": 0, "copy_index": 0, "allele_id": "A"},
            "c01b": {"track": "c01b", "chromosome": CHROMOSOME, "chromosome_index": 0, "copy_index": 1, "allele_id": "B"},
            "all_40_tracks_mapped": True,
        },
        "coordinate_export": coordinate_export_validation,
        "grid": {
            "source": "100k-state.json and 100k-state.npz",
            "full_grid_rule": "state chromosomes==0, verified against range(0, chr1_header_length_bp, 100000)",
            "chr1": {
                "chromosome": CHROMOSOME,
                "chromosome_index": CHROMOSOME_INDEX,
                "header_length_bp": CHR1_LENGTH_BP,
                "first_position_bp": int(chr1_positions[0]),
                "last_position_bp": int(chr1_positions[-1]),
                "step_bp": BIN_SIZE_BP,
                "n_bins": int(len(chr1_positions)),
                "min_position_bp": int(chr1_positions.min()),
                "max_position_bp": int(chr1_positions.max()),
                "expected_n_bins": EXPECTED_N_BINS,
                "expected_full_finite_bins": EXPECTED_N_BINS,
                "full_extent_bp": [0, CHR1_LENGTH_BP],
            },
        },
        "matrices": {
            "raw_dtype": str(raw_matrices.dtype),
            "raw_shape": list(raw_matrices.shape),
            "A": matrix_summaries["A"],
            "B": matrix_summaries["B"],
            "same_common_finite_bin_mask": True,
            "common_finite_bin_count": int(common_finite_bins.sum()),
            "offdiag_unordered_pair_count": offdiag_pair_count,
            "pooled_offdiag_value_count": int(len(pooled_offdiag)),
            "shared_rms": shared_rms,
            "shared_rms_positive_finite": True,
            "diagonal_display_value": 0.0,
            "diagonal_excluded_from_rms": True,
            "normalized_pooled_offdiag_rms": normalized_pooled_rms,
            "normalized_pooled_offdiag_rms_abs_error": abs(normalized_pooled_rms - 1.0),
            "normalized_matrices_not_saved_duplicate": True,
            "n_bio_cells": 1,
            "reference_oracle_comparison": False,
        },
        "color_norm": {
            "colormap": COLORMAP,
            "missing_color": MISSING_COLOR,
            "vmin": 0.0,
            "vmax": color_vmax,
            "clip": False,
            "global_finite_max_across_A_and_B_normalized": True,
            "same_limits_A_and_B": True,
            "per_allele_normalization": False,
            "percentile_clipping": False,
        },
        "render": render_records,
        "source_hash_after_analysis": source_hash_after,
        "source_hash_unchanged_after_analysis": source_hash_unchanged,
        "artifacts": artifacts,
        "preflight": {
            "path": str(output_paths["preflight"].resolve()),
            "stage_coordinate_hash_checked_before_numeric_read": True,
            "numeric_arrays_loaded_after_preflight": True,
        },
        "run_log": {"path": str(output_paths["run_log"].resolve())},
        "validation_file": {"path": str(output_paths["validation"].resolve())},
        "scope_guard": {
            "fit": False,
            "gpu_eval": False,
            "training": False,
            "raw_contacts_opened": False,
            "phase_files_opened": False,
            "reference_3dg_opened": False,
            "old_mat_pat_assignment": False,
            "old_reference_crop": False,
            "interpolation_or_resampling": False,
            "smoothing": False,
            "new_accuracy_claim": False,
            "new_L2_claim": False,
            "source_inputs_modified": False,
            "old_results_modified": False,
        },
        "validation_checks": {
            "config_frozen_before_matrix_computation": True,
            "coordinate_file_sha_matches_stage_before_numeric_read": True,
            "state_array_hashes_match_canonical_function": True,
            "state_full_grid_and_headers_verified": True,
            "native_mapping_verified": True,
            "coordinate_export_matches_state": True,
            "finite_coordinates": True,
            "matrix_shape_1955x1955": True,
            "matrix_float64": True,
            "matrix_symmetry": True,
            "nonnegative_distances": True,
            "diagonal_zero_and_excluded": True,
            "exact_offdiag_denominator": True,
            "shared_rms_positive_finite": True,
            "normalized_pooled_rms_approximately_one": True,
            "same_color_limits_and_norm_both_images": True,
            "png_pixels_and_dpi_verified": True,
            "output_hashes_recorded": True,
            "source_hash_same_after_analysis": True,
            "json_allow_nan_false": True,
        },
        "interpretation": {
            "model_distance_not_observed_hic": True,
            "gauge_labels_only": True,
            "no_accuracy_or_L2_claim": True,
        },
    }
    write_json(output_paths["validation"], validation)
    validation_sha256 = sha256_file(output_paths["validation"])
    manifest_path = HERE / "logs" / "output_manifest.json"
    write_readme(output_paths["readme"], validation, validation_sha256, manifest_path)
    manifest = {
        "schema": "chr1_100kb_output_manifest.v1",
        "status": "PASS",
        "validation_sha256": validation_sha256,
        "readme_sha256": sha256_file(output_paths["readme"]),
        "artifacts": artifacts,
        "validation_path": str(output_paths["validation"].resolve()),
        "readme_path": str(output_paths["readme"].resolve()),
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "postprocessing_dir": str(HERE.resolve()),
                "chr1_bins": EXPECTED_N_BINS,
                "position_first_last_step_bp": [0, EXPECTED_LAST_POSITION_BP, BIN_SIZE_BP],
                "offdiag_unordered_pair_count": offdiag_pair_count,
                "shared_rms": shared_rms,
                "normalized_pooled_offdiag_rms": normalized_pooled_rms,
                "color_norm": [0.0, color_vmax],
                "allele_A_png": artifacts["allele_A_png"],
                "allele_A_pdf": artifacts["allele_A_pdf"],
                "allele_B_png": artifacts["allele_B_png"],
                "allele_B_pdf": artifacts["allele_B_pdf"],
                "raw_npz": artifacts["raw_npz"],
                "validation": output_record(output_paths["validation"]),
                "readme": output_record(output_paths["readme"]),
                "manifest": output_record(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return validation


def main() -> int:
    try:
        run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
