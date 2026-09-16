#!/usr/bin/env python3
"""渲染冻结的 P9016 chr1 distance-matrix 补充评估。

本脚本仅供 evaluator 使用。它在新的 EvalGate 注册并哈希两个坐标文件后，读取已选择的 020 坐标 payload 和 reference。本处不重新计算 contacts、phase labels、拟合、candidate selection 或全基因组指标。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
BIN = 1_000_000
OFF = 3_000_000
CHROMOSOME = "chr1"
CHROMOSOME_LENGTH_BP = 195_471_971
EXPECTED_N_BINS = 193
EXPECTED_N_TOTAL_PAIRS = EXPECTED_N_BINS * (EXPECTED_N_BINS - 1) // 2
EXPECTED_PAIRING = "a=pat"
EXPECTED_REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
EXPECTED_SELECTED_SHA256 = "afb2d52ae11e342e9b43b3c8042c5581760d177c5e563f2478646ab36b3e7078"
EXPECTED_SELECTION_SHA256 = "5ee6376527d6139eb4fe55ac057b40ca94a6e731289a463989f2343a815252a6"
EXPECTED_CANONICAL_METRICS_SHA256 = "4eac0661698043bc15442fabbec9bec67f262f0a682fd23640fa063e6dcd9f70"
EXPECTED_CANONICAL_GATE_SHA256 = "79119d5088c921021f5e835f0c35f2e87bb057577859cc19c5f99bc146434e4c"
EXPECTED_NPZ_SHA256 = "ff8022d3cbfd9df663d78e0d09c0d4b941be3e8d1608cf822543f68a5f118d44"
RENDER_COLORMAP = "coolwarm_r"
RENDER_FIGURE_SIZE_INCHES = (6.0, 6.0)
RENDER_TITLES = (
    "Reference + SNP | maternal",
    "Reference + SNP | paternal",
    "SNP-free | B -> maternal",
    "SNP-free | A -> paternal",
)
RENDER_CAPTION = (
    "Shared RMS scale per reconstruction; four-panel common colorbar; "
    "red=near, blue=far; grey=missing"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is unavailable: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}")
    return {"path": str(path), "expected_sha256": expected_sha256, "actual_sha256": actual}


def load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}: {path}") from exc


def assert_config(config: dict[str, Any], config_path: Path) -> None:
    if config.get("schema_version") != "supplemental-chr1-distance-matrices-v1":
        raise RuntimeError("unsupported supplemental config schema")
    if config.get("status") != "config_frozen_before_compute":
        raise RuntimeError("supplemental config was not frozen before compute")
    grid = config.get("grid", {})
    expected_grid = {
        "chromosome": CHROMOSOME,
        "chromosome_length_bp": CHROMOSOME_LENGTH_BP,
        "bin_size_bp": BIN,
        "offset_bp": OFF,
        "expected_n_bins": EXPECTED_N_BINS,
    }
    for key, expected in expected_grid.items():
        if grid.get(key) != expected:
            raise RuntimeError(f"config grid.{key} mismatch: expected {expected!r}, got {grid.get(key)!r}")
    pairing = config.get("pairing", {})
    if pairing.get("expected_best_pairing") != EXPECTED_PAIRING:
        raise RuntimeError("config pairing is not the frozen canonical a=pat pairing")
    distance = config.get("distance_and_norm", {})
    if distance.get("colormap") != "coolwarm":
        raise RuntimeError("config must use exact non-reversed coolwarm")
    if distance.get("mask") != "four-track common finite bins and unordered non-diagonal pairs":
        raise RuntimeError("config mask policy changed")
    if config_path.parent != HERE:
        raise RuntimeError("config must live beside the supplemental script")


def dense_track(coords: dict[str, dict[int, np.ndarray]], track: str,
                positions_bp: np.ndarray) -> np.ndarray:
    points = np.full((len(positions_bp), 3), np.nan, dtype=float)
    rows = coords.get(track, {})
    for index, position in enumerate(positions_bp):
        value = rows.get(int(position))
        if value is not None:
            point = np.asarray(value, dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise RuntimeError(f"non-finite or malformed coordinate at {track}:{position}")
            points[index] = point
    return points


def distance_matrix(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    matrix = np.linalg.norm(delta, axis=2)
    finite = np.isfinite(points).all(axis=1)
    matrix[~(finite[:, None] & finite[None, :])] = np.nan
    return matrix


def rms_scale(matrix_pair: np.ndarray, common_pair_mask: np.ndarray) -> float:
    upper = np.triu(common_pair_mask, k=1)
    values = matrix_pair[:, upper]
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError("RMS scale has no finite common non-diagonal values")
    scale = float(np.sqrt(np.mean(values * values)))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"RMS scale is not positive and finite: {scale}")
    return scale


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


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n" or signature[12:16] != b"IHDR":
        raise RuntimeError(f"not a PNG with IHDR header: {path}")
    return struct.unpack(">II", signature[16:24])


def render_figure(normalized: np.ndarray, positions_bp: np.ndarray, vmax: float,
                  png_path: Path, pdf_path: Path, *,
                  colormap_name: str = RENDER_COLORMAP,
                  figure_size_inches: tuple[float, float] = RENDER_FIGURE_SIZE_INCHES,
                  titles: tuple[str, ...] = RENDER_TITLES,
                  caption: str = RENDER_CAPTION) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.6,
    })
    cmap = plt.get_cmap(colormap_name).copy()
    cmap.set_bad("#bdbdbd")
    if cmap.name != colormap_name:
        raise RuntimeError(f"unexpected colormap name: {cmap.name}")

    # 矩阵条目是绝对基因组位置处的 bin-center 值。
    positions_mb = positions_bp.astype(float) / 1_000_000.0
    half_bin_mb = BIN / 2_000_000.0
    extent = (
        float(positions_mb[0] - half_bin_mb),
        float(positions_mb[-1] + half_bin_mb),
        float(positions_mb[0] - half_bin_mb),
        float(positions_mb[-1] + half_bin_mb),
    )
    ticks = np.asarray([3, 50, 100, 150, 195], dtype=float)

    fig = plt.figure(figsize=figure_size_inches, dpi=300)
    grid_spec = fig.add_gridspec(
        2, 3, width_ratios=[1.0, 1.0, 0.055], height_ratios=[1.0, 1.0],
        left=0.105, right=0.91, bottom=0.14, top=0.87,
        wspace=0.24, hspace=0.36,
    )
    axes = [
        fig.add_subplot(grid_spec[0, 0]), fig.add_subplot(grid_spec[0, 1]),
        fig.add_subplot(grid_spec[1, 0]), fig.add_subplot(grid_spec[1, 1]),
    ]
    if len(titles) != len(axes):
        raise RuntimeError("four panel titles are required")
    images = []
    for axis, matrix, title in zip(axes, normalized, titles):
        image = axis.imshow(
            matrix, origin="lower", interpolation="nearest", aspect="equal",
            extent=extent, cmap=cmap, vmin=0.0, vmax=vmax,
        )
        images.append(image)
        axis.set_title(title, pad=4)
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.tick_params(width=0.6, length=2.5, pad=2)
        axis.set_xlabel("Genomic position (Mb)")
        axis.set_ylabel("Genomic position (Mb)")

    colorbar_axis = fig.add_subplot(grid_spec[:, 2])
    colorbar = fig.colorbar(images[0], cax=colorbar_axis)
    colorbar.set_label("normalized 3D Euclidean distance", fontsize=7, labelpad=4)
    colorbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    fig.suptitle("P9016 chr1 | 1 Mb distance matrices", fontsize=7, y=0.925)
    fig.text(
        0.5, 0.045, caption,
        ha="center", va="center", fontsize=7,
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, format="png")
    fig.savefig(pdf_path, dpi=300, format="pdf")
    plt.close(fig)
    if cmap.name != colormap_name:
        raise RuntimeError("figure colormap changed unexpectedly")
    return {
        "colormap": cmap.name,
        "missing_color": "#bdbdbd",
        "vmin": 0.0,
        "vmax": float(vmax),
        "figure_size_inches": [float(value) for value in figure_size_inches],
        "dpi": 300,
        "axis_extent_mb": [float(value) for value in extent],
        "position_start_mb": [float(value) for value in positions_mb],
        "png_dimensions": list(png_dimensions(png_path)),
    }


def output_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def write_readme(path: Path, *, config_path: Path, validation: dict[str, Any],
                 outputs: dict[str, Any]) -> None:
    grid = validation["grid"]
    mask = validation["mask"]
    scales = validation["scales"]
    pairing = validation["pairing"]
    inputs = validation["inputs"]
    figure = validation["figure"]
    lines = [
        "# P9016 chr1 距离矩阵评价补充",
        "",
        "**状态：** `evaluation_complete`。这是 020 canonical 评价下的全新补充图件；没有重训练、重选型或重算全基因组指标。",
        "",
        "## 冻结输入",
        "",
        f"- 参考结构 + SNP：`{inputs['reference']['path']}`，SHA256 `{inputs['reference']['actual_sha256']}`。",
        f"- SNP-free 重构：`{inputs['selected']['path']}`，candidate `random_joint`，SHA256 `{inputs['selected']['actual_sha256']}`。",
        f"- 训练选择：`{inputs['selection']['path']}`，SHA256 `{inputs['selection']['actual_sha256']}`，`status=training_complete`，`selected_id=random_joint`。",
        f"- canonical metrics/gate 仅作既有评价证据：metrics SHA256 `{inputs['canonical_metrics']['actual_sha256']}`；gate SHA256 `{inputs['canonical_gate']['actual_sha256']}`。旧 metrics、selection、坐标、ZIP 未写回。",
        "",
        "## 网格、观测单位与配对",
        "",
        f"- P9016 单细胞；只显示 chr1。观察单位是 1 Mb bin 的无序、非对角距离对；距离为 3D 欧氏距离。",
        f"- 完整显示网格：`OFF=3 Mb`，bin 起点 `{grid['first_position_mb']:.0f}..{grid['last_position_mb']:.0f} Mb`，{grid['n_total_bins']} bins；chr1 header length `{grid['chromosome_length_bp']:,} bp`。轴为绝对基因组位置 (Mb)，没有删除低覆盖区或压缩轴。",
        f"- 四轨共同有限 bin/距离对 mask：common bins `{mask['n_common_bins']}/{mask['n_total_bins']}`；全网格非对角距离对 `{mask['n_total_non_diagonal_pairs']}`，common non-diagonal pairs `{mask['n_common_non_diagonal_pairs']}`；对角线显示 0，但不进入 RMS 或距离对统计。",
        f"- 缺失 bin：{json.dumps(mask['missing_bins_by_track'], ensure_ascii=False, sort_keys=True)}；共同 mask 外的非对角距离对数 `{mask['n_total_non_diagonal_pairs'] - mask['n_common_non_diagonal_pairs']}`，图中以灰色显示。",
        f"- A/B 配对不由本图观感决定：从 canonical metrics 的 `{pairing['source_pointer']}` 读取既有 chr1 R2 geometry `best_pairing={pairing['best_pairing']}`。固定为 `actual copy A (c01a) -> paternal (chr1(pat))`，`actual copy B (c01b) -> maternal (chr1(mat))`。",
        "",
        "## 尺度与颜色",
        "",
        f"- 参考结构两 allele 共同使用一个 RMS scale `{scales['reference']:.12g}`；candidate 两 copy 共同使用另一个 RMS scale `{scales['candidate']:.12g}`。每个 scale 都是相应两轨在四轨共同的非对角距离对 mask 上合并后的 RMS 距离；不做每个 panel 独立 min-max 或独立缩放。",
        f"- 四个归一化 panel 共用一个颜色规范：`0..{figure['vmax']:.12g}`，不裁剪有限极值；精确 colormap 为 `coolwarm`（蓝=近，红=远），missing 为灰色 `#bdbdbd`。图注明确每个重构共用 scale 和 4-panel 共用 colorbar。",
        "- 重构坐标采用无量纲 `R=1` 约束；reference 使用其原始任意坐标单位，二者都未作物理尺度标定。图中各行归一化为共同 RMS 距离，不能解释为微米。`config.grid.coordinate_units` 仅描述 selected reconstruction，不应用到 reference。",
        "",
        "## 输出与复现",
        "",
        f"- 脚本：`{path.parent / 'run_supplemental_chr1_distance_matrices.py'}`；配置：`{config_path}`。运行时会核验输入 SHA、selection、canonical R2 pairing，并用全新 `EvalGate` 在读取 reference 前注册/哈希坐标。",
        f"- NPZ 追溯包：`{outputs['npz']['path']}`；保存 `raw_matrices`、`normalized_matrices`、`track_bin_finite_mask`、`common_bin_mask`、`common_pair_mask`、`positions_bp`、`scales` 和 common color norm。",
        f"- PNG：`{outputs['png']['path']}`；PDF：`{outputs['pdf']['path']}`；快捷副本在 `deliverables/`。PNG 实际尺寸 `{figure['png_dimensions'][0]}x{figure['png_dimensions'][1]} px`，300 dpi，figure canvas 约 7x7 inch，基础 panel 约 3 inch。",
        f"- 轻量验证结果：距离矩阵对称、共同 mask 对称且对角为 false、对角线在共同 bins 为 0、两 RMS scale 为正、四 panel 共用同一 `vmin=0/vmax={figure['vmax']:.12g}`；完整机器可读证据见 `validation.json`。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_render_payload(npz_path: Path) -> dict[str, Any]:
    """仅从冻结 NPZ 加载并验证，不重新计算任何坐标。"""
    npz_record = require_file(npz_path, EXPECTED_NPZ_SHA256, "frozen distance-matrix NPZ")
    required = {
        "normalized_matrices", "positions_bp", "common_bin_mask", "common_pair_mask",
        "scales", "color_norm", "colormap",
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise RuntimeError(f"frozen NPZ lacks render fields: {missing}")
        payload = {
            "normalized_matrices": np.array(archive["normalized_matrices"], copy=True),
            "positions_bp": np.array(archive["positions_bp"], copy=True),
            "common_bin_mask": np.array(archive["common_bin_mask"], copy=True),
            "common_pair_mask": np.array(archive["common_pair_mask"], copy=True),
            "scales": np.array(archive["scales"], copy=True),
            "color_norm": np.array(archive["color_norm"], copy=True),
            "colormap": str(np.asarray(archive["colormap"]).item()),
        }
    normalized = payload["normalized_matrices"]
    positions_bp = payload["positions_bp"]
    common_bin_mask = payload["common_bin_mask"]
    common_pair_mask = payload["common_pair_mask"]
    scales = payload["scales"]
    color_norm = payload["color_norm"]
    if normalized.shape != (4, EXPECTED_N_BINS, EXPECTED_N_BINS):
        raise RuntimeError(f"unexpected normalized matrix shape: {normalized.shape}")
    if positions_bp.shape != (EXPECTED_N_BINS,) or not np.array_equal(
        positions_bp, np.asarray(range(OFF, CHROMOSOME_LENGTH_BP, BIN), dtype=np.int64)
    ):
        raise RuntimeError("frozen NPZ positions do not match the 193-bin chr1 grid")
    if common_bin_mask.shape != (EXPECTED_N_BINS,) or int(common_bin_mask.sum()) != 188:
        raise RuntimeError("frozen NPZ common-bin mask changed")
    if common_pair_mask.shape != (EXPECTED_N_BINS, EXPECTED_N_BINS):
        raise RuntimeError("frozen NPZ common-pair mask shape changed")
    if not np.array_equal(common_pair_mask, common_pair_mask.T) or np.any(np.diag(common_pair_mask)):
        raise RuntimeError("frozen NPZ common-pair mask is invalid")
    if int(np.triu(common_pair_mask, k=1).sum()) != 17578:
        raise RuntimeError("frozen NPZ common-pair count changed")
    if not np.array_equal(scales, np.asarray([1.5529545301005403, 0.3766542938839421])):
        raise RuntimeError("frozen NPZ RMS scales changed")
    if not np.array_equal(color_norm, np.asarray([0.0, 2.4009829719481615])):
        raise RuntimeError("frozen NPZ color norm changed")
    finite = np.isfinite(normalized)
    if not np.allclose(normalized[finite], normalized.transpose(0, 2, 1)[finite], rtol=0.0, atol=1e-12):
        raise RuntimeError("frozen NPZ normalized matrices are not symmetric")
    # NPZ 是初始冻结的数值记录；其旧 colormap metadata
    # 有意保留，本修订只反转视觉映射。
    if payload["colormap"] != "coolwarm":
        raise RuntimeError(f"frozen NPZ colormap metadata changed: {payload['colormap']}")
    payload["npz_record"] = npz_record
    return payload


def render_existing(npz_path: Path) -> dict[str, Any]:
    """仅从冻结 NPZ 渲染，并覆盖授权的图文件。"""
    payload = load_render_payload(npz_path)
    out_dir = HERE
    png_path = out_dir / "P9016-v1-020-chr1-distance-matrices.png"
    pdf_path = out_dir / "P9016-v1-020-chr1-distance-matrices.pdf"
    figure = render_figure(
        payload["normalized_matrices"], payload["positions_bp"],
        float(payload["color_norm"][1]), png_path, pdf_path,
        colormap_name=RENDER_COLORMAP,
        figure_size_inches=RENDER_FIGURE_SIZE_INCHES,
        titles=RENDER_TITLES,
        caption=RENDER_CAPTION,
    )
    deliverables_dir = ROOT / "deliverables"
    deliverables_dir.mkdir(parents=True, exist_ok=True)
    shortcut_png = deliverables_dir / png_path.name
    shortcut_pdf = deliverables_dir / pdf_path.name
    shutil.copy2(png_path, shortcut_png)
    shutil.copy2(pdf_path, shortcut_pdf)
    return {
        "status": "render_revision_complete",
        "input_npz": payload["npz_record"],
        "render": figure,
        "scales": [float(value) for value in payload["scales"]],
        "common_bins": int(payload["common_bin_mask"].sum()),
        "common_non_diagonal_pairs": int(np.triu(payload["common_pair_mask"], k=1).sum()),
        "outputs": {
            "png": output_file_record(png_path),
            "pdf": output_file_record(pdf_path),
            "shortcut_png": output_file_record(shortcut_png),
            "shortcut_pdf": output_file_record(shortcut_pdf),
        },
        "numeric_inputs_unchanged": True,
        "recomputed_coordinates_or_metrics": False,
    }


def run(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path, "supplemental config")
    if not isinstance(config, dict):
        raise RuntimeError("supplemental config must be a JSON object")
    assert_config(config, config_path)

    input_cfg = config["inputs"]
    canonical_cfg = config["canonical_evaluation"]
    ref_path = resolve_path(input_cfg["reference"]["path"])
    selected_path = resolve_path(input_cfg["selected"]["path"])
    selection_path = resolve_path(input_cfg["selection"]["path"])
    canonical_metrics_path = resolve_path(canonical_cfg["metrics_path"])
    canonical_gate_path = resolve_path(canonical_cfg["gate_path"])
    inputs = {
        "reference": require_file(ref_path, EXPECTED_REFERENCE_SHA256, "reference 3DG"),
        "selected": require_file(selected_path, EXPECTED_SELECTED_SHA256, "selected coordinates"),
        "selection": require_file(selection_path, EXPECTED_SELECTION_SHA256, "selection.json"),
        "canonical_metrics": require_file(canonical_metrics_path, EXPECTED_CANONICAL_METRICS_SHA256, "canonical metrics"),
        "canonical_gate": require_file(canonical_gate_path, EXPECTED_CANONICAL_GATE_SHA256, "canonical gate"),
    }
    if input_cfg["reference"].get("sha256") != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("config reference SHA does not match frozen user input")
    if input_cfg["selected"].get("sha256") != EXPECTED_SELECTED_SHA256:
        raise RuntimeError("config selected SHA does not match frozen user input")
    if input_cfg["selection"].get("sha256") != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("config selection SHA does not match frozen user input")

    # canonical gate 作为不可变证据读取；本次运行写出自己的 gate。
    canonical_gate = load_json(canonical_gate_path, "canonical gate")
    if not isinstance(canonical_gate, list) or len(canonical_gate) < 5:
        raise RuntimeError("canonical gate does not contain the expected five registered entries")
    canonical_metrics = load_json(canonical_metrics_path, "canonical metrics")
    try:
        pairing_record = canonical_metrics["per_chromosome"][CHROMOSOME]["candidates"]["random_joint"]["R2"]["by_candidate"]["selected"]
        pairing = {
            "source_pointer": "/per_chromosome/chr1/candidates/random_joint/R2/by_candidate/selected/best_pairing",
            "best_pairing": pairing_record["best_pairing"],
            "geometry_status": pairing_record["gauge"]["status"],
            "geometry_orientation": pairing_record["gauge"]["orientation"],
            "n_common_pairs_in_canonical_record": int(pairing_record["n_common"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("canonical metrics lacks the frozen chr1 selected R2 pairing record") from exc
    if pairing["best_pairing"] != EXPECTED_PAIRING:
        raise RuntimeError(f"canonical chr1 selected R2 pairing changed: {pairing['best_pairing']}")

    # 只有输入/config 检查完成后才导入项目 API。verify_training_complete 只读，
    # 不能打开 phase/reference payload。
    sys.path.insert(0, str(ROOT))
    from pr import ref3dg, refeval, reconstruction_report as report
    from pr.gate import EvalGate

    verified = report.verify_training_complete(selection_path, strict_p9016=True)
    if verified.selected_id != "random_joint":
        raise RuntimeError(f"selection selected_id changed: {verified.selected_id}")
    if verified.document.get("status") != "training_complete":
        raise RuntimeError("selection status is not training_complete")
    if verified.grid.bin_size_bp != BIN or verified.grid.origin_bp != 0:
        raise RuntimeError("verified full grid does not match frozen 1 Mb origin-zero training grid")
    # 用户冻结的 selected.3dg 是已交付的 020 selected payload。其字节
    # 必须与 selection.json 已核验的 selected candidate source 一致。
    selected_source = Path(verified.selected.coordinates_path).resolve()
    selected_source_sha = sha256_file(selected_source)
    if selected_source_sha != EXPECTED_SELECTED_SHA256:
        raise RuntimeError("selection-selected candidate source digest differs from frozen selected.3dg digest")

    out_dir = HERE
    gate_path = out_dir / "gate.json"
    gate = EvalGate(str(gate_path))
    registered_selected_sha = gate.register("supplemental-chr1-distance-matrices", "candidate:selected.3dg", str(selected_path))
    registered_reference_sha = gate.register("supplemental-chr1-distance-matrices", "reference:P9016.1m.3dg.gz", str(ref_path))
    if registered_selected_sha != EXPECTED_SELECTED_SHA256 or registered_reference_sha != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("supplemental gate registered an unexpected coordinate digest")
    gate.arm(refeval.STAGE)
    gate.require(refeval.STAGE)

    # candidate parsing 在注册后进行；reference parsing 由 ref3dg.load_reference 显式受 gate 保护。
    selected = report.read_full_grid_coordinates(str(selected_path), verified.grid, EXPECTED_SELECTED_SHA256)
    reference = ref3dg.load_reference(gate, path=str(ref_path))
    if not isinstance(reference, dict):
        raise RuntimeError("reference loader returned an unexpected object")

    positions_bp = np.asarray(range(OFF, CHROMOSOME_LENGTH_BP, BIN), dtype=np.int64)
    if len(positions_bp) != EXPECTED_N_BINS or int(positions_bp[-1]) != 195_000_000:
        raise RuntimeError("chr1 OFF=3 Mb position grid does not match the frozen 193-bin grid")
    panel_tracks = ("chr1(mat)", "chr1(pat)", "c01b", "c01a")
    panel_labels = (
        "reference_maternal",
        "reference_paternal",
        "reconstruction_copy_B_maternal",
        "reconstruction_copy_A_paternal",
    )
    coordinate_sources = (reference, reference, selected, selected)
    points = np.stack([
        dense_track(source, track, positions_bp)
        for source, track in zip(coordinate_sources, panel_tracks)
    ], axis=0)
    track_bin_finite_mask = np.isfinite(points).all(axis=2)
    common_bin_mask = track_bin_finite_mask.all(axis=0)
    individual_pair_finite = np.zeros((4, len(positions_bp), len(positions_bp)), dtype=bool)
    raw_matrices = np.full((4, len(positions_bp), len(positions_bp)), np.nan, dtype=float)
    diagonal = np.eye(len(positions_bp), dtype=bool)
    for index in range(4):
        matrix = distance_matrix(points[index])
        finite = np.isfinite(matrix)
        individual_pair_finite[index] = finite & ~diagonal
        raw_matrices[index] = matrix
    common_pair_mask = individual_pair_finite.all(axis=0) & ~diagonal
    if not np.array_equal(common_pair_mask, common_pair_mask.T):
        raise RuntimeError("common pair mask is not symmetric")
    if np.any(np.diag(common_pair_mask)):
        raise RuntimeError("common pair mask includes diagonal")
    for index in range(4):
        matrix = raw_matrices[index]
        matrix[~common_pair_mask] = np.nan
        diagonal_values = np.where(common_bin_mask, 0.0, np.nan)
        diagonal_indices = np.arange(len(positions_bp))
        matrix[diagonal_indices, diagonal_indices] = diagonal_values
    common_pair_count = int(np.triu(common_pair_mask, k=1).sum())
    if common_pair_count <= 0:
        raise RuntimeError("common finite non-diagonal pair mask is empty")
    reference_scale = rms_scale(raw_matrices[:2], common_pair_mask)
    candidate_scale = rms_scale(raw_matrices[2:], common_pair_mask)
    scales = np.asarray([reference_scale, candidate_scale], dtype=float)
    panel_scales = np.asarray([reference_scale, reference_scale, candidate_scale, candidate_scale], dtype=float)
    normalized_matrices = raw_matrices / panel_scales[:, None, None]
    normalized_matrices[:, ~common_pair_mask] = np.nan
    diagonal_indices = np.arange(len(positions_bp))
    normalized_matrices[:, diagonal_indices, diagonal_indices] = np.where(
        common_bin_mask, 0.0, np.nan
    )
    finite_normalized = normalized_matrices[np.isfinite(normalized_matrices)]
    if not len(finite_normalized):
        raise RuntimeError("normalized matrices have no finite values")
    color_vmin = 0.0
    color_vmax = float(np.max(finite_normalized))
    if not np.isfinite(color_vmax) or color_vmax <= color_vmin:
        raise RuntimeError(f"invalid common color norm maximum: {color_vmax}")

    # 渲染前进行数值检查。
    finite_raw = np.isfinite(raw_matrices)
    if not np.allclose(raw_matrices[finite_raw], raw_matrices.transpose(0, 2, 1)[finite_raw], rtol=0.0, atol=1e-12):
        raise RuntimeError("raw distance matrices are not symmetric")
    finite_norm = np.isfinite(normalized_matrices)
    if not np.allclose(normalized_matrices[finite_norm], normalized_matrices.transpose(0, 2, 1)[finite_norm], rtol=0.0, atol=1e-12):
        raise RuntimeError("normalized distance matrices are not symmetric")
    diagonal_values = raw_matrices[:, np.arange(len(positions_bp)), np.arange(len(positions_bp))]
    if not np.all(diagonal_values[:, common_bin_mask] == 0.0):
        raise RuntimeError("common-bin diagonal is not exactly zero")
    if np.any(np.isfinite(diagonal_values[:, ~common_bin_mask])):
        raise RuntimeError("missing-bin diagonal must remain NaN")
    if not np.all(scales > 0) or not np.isfinite(scales).all():
        raise RuntimeError("RMS scales are not positive finite values")
    if not np.array_equal(track_bin_finite_mask.all(axis=0), common_bin_mask):
        raise RuntimeError("common bin mask is inconsistent with four-track finite mask")

    npz_path = out_dir / "distance_matrices_chr1.npz"
    png_path = out_dir / "P9016-v1-020-chr1-distance-matrices.png"
    pdf_path = out_dir / "P9016-v1-020-chr1-distance-matrices.pdf"
    validation_path = out_dir / "validation.json"
    readme_path = out_dir / "README.md"
    for path in (npz_path, png_path, pdf_path, validation_path, readme_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing supplemental artifact: {path}")
    np.savez_compressed(
        npz_path,
        raw_matrices=raw_matrices,
        normalized_matrices=normalized_matrices,
        track_bin_finite_mask=track_bin_finite_mask,
        individual_pair_finite=individual_pair_finite,
        common_bin_mask=common_bin_mask,
        common_pair_mask=common_pair_mask,
        positions_bp=positions_bp,
        panel_tracks=np.asarray(panel_tracks),
        panel_labels=np.asarray(panel_labels),
        scales=scales,
        panel_scales=panel_scales,
        color_norm=np.asarray([color_vmin, color_vmax], dtype=float),
        coordinate_units=np.asarray("dimensionless_R1_not_micrometre_calibrated"),
        colormap=np.asarray("coolwarm"),
        pairing=np.asarray(json.dumps(pairing, sort_keys=True)),
    )
    figure = render_figure(
        normalized_matrices, positions_bp, color_vmax, png_path, pdf_path,
        colormap_name="coolwarm",
        figure_size_inches=(7.0, 7.0),
        titles=(
            "Reference + SNP | maternal",
            "Reference + SNP | paternal",
            "SNP-free reconstruction | copy B -> maternal",
            "SNP-free reconstruction | copy A -> paternal",
        ),
        caption=(
            "Shared RMS scale per reconstruction; four-panel common colorbar; "
            "blue=near, red=far; grey=missing"
        ),
    )

    # 不要将面向用户的快捷方式静默替换为其他补充结果。
    deliverables_dir = ROOT / "deliverables"
    deliverables_dir.mkdir(parents=True, exist_ok=True)
    shortcut_png = deliverables_dir / "P9016-v1-020-chr1-distance-matrices.png"
    shortcut_pdf = deliverables_dir / "P9016-v1-020-chr1-distance-matrices.pdf"
    for path in (shortcut_png, shortcut_pdf):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing shortcut: {path}")
    shutil.copy2(png_path, shortcut_png)
    shutil.copy2(pdf_path, shortcut_pdf)

    coordinate_summary = {
        "reference_chr1_mat": finite_span(points[0]),
        "reference_chr1_pat": finite_span(points[1]),
        "selected_c01b": finite_span(points[2]),
        "selected_c01a": finite_span(points[3]),
        "selected_full_grid_inventory": {
            "n_tracks": len(selected),
            "n_beads": int(sum(len(rows) for rows in selected.values())),
            "max_radius": float(max(np.linalg.norm(value) for rows in selected.values() for value in rows.values())),
        },
    }
    mask = {
        "n_total_bins": int(len(positions_bp)),
        "n_common_bins": int(common_bin_mask.sum()),
        "n_total_non_diagonal_pairs": EXPECTED_N_TOTAL_PAIRS,
        "n_common_non_diagonal_pairs": common_pair_count,
        "missing_bins_by_track": {
            label: int((~track_bin_finite_mask[index]).sum())
            for index, label in enumerate(panel_labels)
        },
        "missing_pair_count_by_track": {
            label: int(EXPECTED_N_TOTAL_PAIRS - np.triu(individual_pair_finite[index], k=1).sum())
            for index, label in enumerate(panel_labels)
        },
        "common_pair_mask_symmetric": bool(np.array_equal(common_pair_mask, common_pair_mask.T)),
        "common_pair_mask_diagonal_false": bool(not np.any(np.diag(common_pair_mask))),
    }
    grid = {
        "chromosome": CHROMOSOME,
        "chromosome_length_bp": CHROMOSOME_LENGTH_BP,
        "bin_size_bp": BIN,
        "offset_bp": OFF,
        "first_position_bp": int(positions_bp[0]),
        "last_position_bp": int(positions_bp[-1]),
        "first_position_mb": float(positions_bp[0] / 1_000_000.0),
        "last_position_mb": float(positions_bp[-1] / 1_000_000.0),
        "n_total_bins": int(len(positions_bp)),
        "axis_units": "absolute genomic position (Mb)",
        "coordinate_units": "dimensionless_R1_not_micrometre_calibrated",
    }
    outputs = {
        "npz": output_file_record(npz_path),
        "png": output_file_record(png_path),
        "pdf": output_file_record(pdf_path),
        "shortcut_png": output_file_record(shortcut_png),
        "shortcut_pdf": output_file_record(shortcut_pdf),
        "gate": output_file_record(gate_path),
    }
    validation = {
        "status": "evaluation_complete",
        "inputs": inputs,
        "selection": {
            "status": verified.document.get("status"),
            "selected_id": verified.selected_id,
            "selected_source_path": str(selected_source),
            "selected_source_sha256": selected_source_sha,
        },
        "gate": {
            "path": str(gate_path),
            "armed_stage": refeval.STAGE,
            "entries": gate.entries,
            "all_registered_before_arm": True,
        },
        "pairing": pairing,
        "grid": grid,
        "mask": mask,
        "scales": {
            "reference": reference_scale,
            "candidate": candidate_scale,
            "panel_scales": [float(value) for value in panel_scales],
            "definition": "group RMS over both alleles/copies and four-track common unordered non-diagonal pairs",
            "positive_finite": bool(np.isfinite(scales).all() and np.all(scales > 0)),
        },
        "coordinate_summary": coordinate_summary,
        "shape_validation": {
            "raw_matrices": list(raw_matrices.shape),
            "normalized_matrices": list(normalized_matrices.shape),
            "common_bin_mask": list(common_bin_mask.shape),
            "common_pair_mask": list(common_pair_mask.shape),
            "positions_bp": list(positions_bp.shape),
            "raw_symmetric": True,
            "normalized_symmetric": True,
            "diagonal_zero_on_common_bins": True,
            "diagonal_excluded_from_rms": True,
        },
        "figure": {
            **figure,
            "shared_norm_panels": 4,
            "shared_norm_vmin": color_vmin,
            "shared_norm_vmax": color_vmax,
            "norm_clips_finite_values": False,
            "exact_colormap": "coolwarm",
        },
        "outputs": outputs,
        "no_new_inference_statistics": True,
        "no_micrometre_claim": True,
    }
    validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(readme_path, config_path=config_path, validation=validation, outputs=outputs)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="frozen config.json beside this script")
    parser.add_argument(
        "--render-existing", action="store_true",
        help="render the authorized revision directly from the frozen distance_matrices_chr1.npz",
    )
    parser.add_argument(
        "--npz", default=str(HERE / "distance_matrices_chr1.npz"),
        help="frozen NPZ used by --render-existing",
    )
    args = parser.parse_args()
    if args.render_existing:
        revision = render_existing(Path(args.npz).resolve())
        print(json.dumps(revision, ensure_ascii=False, indent=2))
        return 0
    validation = run(Path(args.config).resolve())
    print(json.dumps({
        "status": validation["status"],
        "supplement_dir": str(HERE),
        "common_bins": validation["mask"]["n_common_bins"],
        "common_non_diagonal_pairs": validation["mask"]["n_common_non_diagonal_pairs"],
        "reference_scale": validation["scales"]["reference"],
        "candidate_scale": validation["scales"]["candidate"],
        "vmax": validation["figure"]["shared_norm_vmax"],
        "png": validation["outputs"]["png"],
        "pdf": validation["outputs"]["pdf"],
        "npz": validation["outputs"]["npz"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
