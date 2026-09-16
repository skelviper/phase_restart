#!/usr/bin/env python3
"""在不拟合或重新评估的情况下渲染锁定的 036 R2 chr1 distance heatmaps。"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/mnt/ssd/zliu/phase_restart")
PREP_DIR = ROOT / "docs/audits/multires-r2-preparation-20260914_041826"
R2_DIR = ROOT / "test_res/036-20260914T064651Z-gpu-multires/evaluation-r2-20260914_072405"
OUT_DIR = R2_DIR / "chr1-heatmaps-20260914_094441"
R2_CODE = PREP_DIR / "multires_r2.py"
CONFIG_PATH = PREP_DIR / "config.json"
MASK_LOCK_PATH = PREP_DIR / "mask_lock.json"
EVAL_RESULTS_PATH = R2_DIR / "evaluation_results.json"
FULL_R2_PATH = R2_DIR / "r2_full_10endpoint_x20chr.tsv"
REP_PATH = R2_DIR / "representatives_by_variant.tsv"
INPUT_PROV_PATH = R2_DIR / "input_provenance.json"
MASK_PROV_PATH = R2_DIR / "mask_provenance.json"
README_SOURCE_PATH = R2_DIR / "README.md"
SELECTION_PATH = ROOT / "test_res/036-20260914T064651Z-gpu-multires/selection.json"
RELEASE_MANIFEST_PATH = ROOT / "test_res/036-20260914T064651Z-gpu-multires/release_ready_manifest.json"
REFERENCE_EXPECTED_SHA = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
EXPECTED_MASK_MANIFEST_SHA = "fa1b26c834c173604c21f954d494cece8e053dd970e7cb6111f561b349574acb"
EXPECTED_MASK_COUNTS = (193, 18528, 17578)
CHROMOSOME = "chr1"
CHR1_LENGTH_BP = 195_471_971
GRID_OFFSET_BP = 3_000_000
BIN_SIZE_BP = 1_000_000
FONT_SIZE_PT = 7
DPI = 300
V_MIN = 0.0
VARIANTS = ("C0", "C1", "C2-map", "C2-free", "C3")
CONDITION_LABELS = (
    "Reference",
    "020 / C0 CPU",
    "C0 GPU",
    "C1 GPU",
    "C2-map GPU",
    "C2-free GPU",
    "C3 GPU",
)
COPY_KEYS = ("mat", "pat")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"missing required input: {path}")
    return path


def load_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def require_close(left: float, right: float, label: str, atol: float = 1e-12) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol):
        raise RuntimeError(f"{label} mismatch: {left!r} vs {right!r}")


def pairwise_distance(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    finite_rows = np.isfinite(points).all(axis=1)
    delta = points[:, None, :] - points[None, :, :]
    values = np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float64), dtype=np.float64)
    valid = finite_rows[:, None] & finite_rows[None, :]
    values[~valid] = np.nan
    diagonal = np.diag_indices(len(points))
    values[diagonal] = 0.0
    return values


def validate_orientation_row(row: dict[str, Any], label: str) -> dict[str, Any]:
    values = {key: float(row[key]) for key in ("rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat")}
    direct = (values["rho_A_mat"] + values["rho_B_pat"]) / 2.0
    cross = (values["rho_A_pat"] + values["rho_B_mat"]) / 2.0
    orientation = str(row["orientation"])
    if abs(direct - cross) <= 1e-12:
        raise RuntimeError(f"{label} has an unresolved chr1 orientation tie")
    expected = "direct" if direct > cross else "swapped"
    if orientation != expected:
        raise RuntimeError(f"{label} stored orientation {orientation!r} disagrees with stored four-rho values ({expected})")
    matched = float(row["matched"])
    require_close(matched, max(direct, cross), f"{label}.matched", atol=1e-12)
    if orientation == "direct":
        mapping = {"mat": "A", "pat": "B"}
    else:
        mapping = {"mat": "B", "pat": "A"}
    return {
        "orientation": orientation,
        "mapping": mapping,
        "rho_A_mat": values["rho_A_mat"],
        "rho_A_pat": values["rho_A_pat"],
        "rho_B_mat": values["rho_B_mat"],
        "rho_B_pat": values["rho_B_pat"],
        "direct": direct,
        "cross": cross,
        "matched": matched,
    }


def set_ticks(axis: Any, positions_mb: np.ndarray) -> None:
    ticks = np.asarray([3, 43, 83, 123, 163, 193], dtype=np.float64)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xlim(float(positions_mb[0] - 0.5), float(positions_mb[-1] + 0.5))
    axis.set_ylim(float(positions_mb[0] - 0.5), float(positions_mb[-1] + 0.5))
    axis.set_xlabel("chr1 position (Mb)", fontsize=FONT_SIZE_PT)
    axis.set_ylabel("chr1 position (Mb)", fontsize=FONT_SIZE_PT)
    axis.tick_params(axis="both", labelsize=FONT_SIZE_PT, length=2, pad=1)


def masked_display(matrix: np.ndarray, pair_mask: np.ndarray) -> np.ma.MaskedArray:
    values = np.asarray(matrix, dtype=np.float64).copy()
    values[~pair_mask] = np.nan
    diagonal = np.diag_indices(values.shape[0])
    values[diagonal] = 0.0
    return np.ma.masked_invalid(values)


def render_heatmaps(
    normalized: np.ndarray,
    pair_mask: np.ndarray,
    positions_mb: np.ndarray,
    condition_labels: tuple[str, ...],
    copy_labels: tuple[tuple[str, str], ...],
    rhos: tuple[float | None, ...],
    vmax: float,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "figure.titlesize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "font.family": "DejaVu Sans",
        }
    )
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad("#d0d0d0")
    extent = (
        float(positions_mb[0] - 0.5),
        float(positions_mb[-1] + 0.5),
        float(positions_mb[0] - 0.5),
        float(positions_mb[-1] + 0.5),
    )
    norm = matplotlib.colors.Normalize(vmin=V_MIN, vmax=vmax, clip=False)
    paths: dict[str, Any] = {"overview": {}, "details": {}}

    fig, axes = plt.subplots(7, 2, figsize=(6.8, 21.0), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.12, right=0.89, top=0.975, bottom=0.035, wspace=0.18, hspace=0.62)
    for row_index, condition_label in enumerate(condition_labels):
        for copy_index, copy_label in enumerate(copy_labels[row_index]):
            axis = axes[row_index, copy_index]
            image = axis.imshow(
                masked_display(normalized[row_index, copy_index], pair_mask),
                cmap=cmap,
                norm=norm,
                origin="lower",
                interpolation="none",
                extent=extent,
                aspect="equal",
            )
            rho = rhos[row_index]
            rho_text = "" if rho is None else f" | chr1 matched rho={rho:.4f}"
            axis.set_title(f"{condition_label}\n{copy_label}{rho_text}", fontsize=FONT_SIZE_PT, pad=2)
            set_ticks(axis, positions_mb)
            if copy_index == 1:
                axis.set_ylabel("")
                axis.tick_params(axis="y", labelleft=False)
        axes[row_index, 0].text(
            -0.36,
            0.5,
            condition_label,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=FONT_SIZE_PT,
        )
    fig.suptitle(
        "P9016 chr1 distance matrices | one cell / 1 Mb / whole-chr best-swap\n"
        "mat-matched / pat-matched = evaluation alignment, not training parent labels",
        fontsize=FONT_SIZE_PT,
        y=0.994,
    )
    cbar_axis = fig.add_axes([0.13, 0.011, 0.73, 0.009])
    colorbar = fig.colorbar(image, cax=cbar_axis, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=FONT_SIZE_PT, length=2, pad=1)
    colorbar.set_label("distance / median distance on frozen R2 pairs", fontsize=FONT_SIZE_PT, labelpad=2)
    overview_png = OUT_DIR / "overview.png"
    overview_pdf = OUT_DIR / "overview.pdf"
    fig.savefig(overview_png, dpi=DPI, format="png")
    fig.savefig(overview_pdf, dpi=DPI, format="pdf")
    plt.close(fig)
    paths["overview"] = {"png": str(overview_png), "pdf": str(overview_pdf), "figsize_inches": [6.8, 21.0]}

    detail_stems = (
        "reference",
        "020_C0_CPU",
        "C0",
        "C1",
        "C2-map",
        "C2-free",
        "C3",
    )
    for row_index, (condition_label, stem) in enumerate(zip(condition_labels, detail_stems)):
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(left=0.09, right=0.91, top=0.72, bottom=0.24, wspace=0.14)
        for copy_index, copy_label in enumerate(copy_labels[row_index]):
            axis = axes[0, copy_index]
            image = axis.imshow(
                masked_display(normalized[row_index, copy_index], pair_mask),
                cmap=cmap,
                norm=norm,
                origin="lower",
                interpolation="none",
                extent=extent,
                aspect="equal",
            )
            rho = rhos[row_index]
            rho_text = "" if rho is None else f" | rho={rho:.4f}"
            axis.set_title(f"{copy_label}{rho_text}", fontsize=FONT_SIZE_PT, pad=2)
            set_ticks(axis, positions_mb)
            if copy_index == 1:
                axis.set_ylabel("")
                axis.tick_params(axis="y", labelleft=False)
        fig.suptitle(f"{condition_label} | chr1 whole-chromosome distance", fontsize=FONT_SIZE_PT, y=0.965)
        cbar_axis = fig.add_axes([0.12, 0.105, 0.76, 0.035])
        colorbar = fig.colorbar(image, cax=cbar_axis, orientation="horizontal")
        colorbar.ax.tick_params(labelsize=FONT_SIZE_PT, length=2, pad=1)
        colorbar.set_label("distance / median distance on frozen R2 pairs", fontsize=FONT_SIZE_PT, labelpad=1)
        png_path = OUT_DIR / f"detail_{stem}.png"
        pdf_path = OUT_DIR / f"detail_{stem}.pdf"
        fig.savefig(png_path, dpi=DPI, format="png")
        fig.savefig(pdf_path, dpi=DPI, format="pdf")
        plt.close(fig)
        paths["details"][condition_label] = {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "figsize_inches": [6.0, 3.0],
        }
    return paths


def image_validation(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else 0}
    if not path.is_file() or path.stat().st_size <= 0:
        return record
    if path.suffix.lower() == ".pdf":
        record["format"] = "PDF"
        record["pdf_header_ok"] = path.read_bytes()[:5] == b"%PDF-"
        return record
    try:
        from PIL import Image

        with Image.open(path) as image:
            record["format"] = image.format
            record["size_px"] = [int(image.width), int(image.height)]
            dpi_value = image.info.get("dpi")
            record["dpi_metadata"] = None if dpi_value is None else [float(dpi_value[0]), float(dpi_value[1])]
            record["dpi_300_ok"] = bool(
                dpi_value is not None and abs(float(dpi_value[0]) - DPI) < 1.0 and abs(float(dpi_value[1]) - DPI) < 1.0
            )
    except Exception as exc:  # pragma: no cover - only a diagnostics fallback
        record["image_validation_error"] = repr(exc)
    return record


def make_readme(metadata: dict[str, Any], validation: dict[str, Any]) -> str:
    figure_lines = [
        f"- 主图：`{Path(metadata['outputs']['overview']['png']).name}` / `{Path(metadata['outputs']['overview']['pdf']).name}`",
    ]
    for condition in metadata["condition_order"]:
        detail = metadata["outputs"]["details"][condition]
        figure_lines.append(f"- {condition} 明细：`{Path(detail['png']).name}` / `{Path(detail['pdf']).name}`")
    return f"""# 036 R2 chr1 距离矩阵热图

状态：`{validation['status']}`。本目录只从已完成并锁定的 036 R2 坐标、R2 TSV/JSON、冻结 21-condition 共同 mask 和评价参考结构生成图像；没有调用训练、优化、native、R1/R3 或新的 R2 endpoint 评价。

## 图形

{chr(10).join(figure_lines)}

`distance_arrays.npz` 保存完整 193×193 原始距离、按冻结 mask 的可视化归一化输入、对角线 0、网格和 mask。mask 外未评价格子在图中明确显示为灰色，不代表已用于 R2。

## 冻结定义

- chr1 网格严格为 `range(3_000_000, 195_471_971, 1_000_000)`，共 193 个 1 Mb bin；矩阵是同一染色体内的欧氏距离，不是 contact matrix。
- 冻结 R2 mask 只使用 17,578 / 18,528 个无序非对角距离对，未因新 endpoint 缩域或重选 mask；归一化和 vmax 只使用这 17,578 对。
- 预测 A/B 列按本轮 R2 已存的整条染色体最佳交换固定为 `mat-matched` / `pat-matched`；这些标签是评价对齐，不是训练提供的亲本标签。参考结构列固定为 maternal / paternal。
- 每张矩阵除以其自身在相同冻结 R2 pairs 上的中位数原始距离；统一 `vmin=0`、统一 99th-percentile `vmax={metadata['normalization']['vmax']:.10g}`，超出部分仅作颜色截断。
- 对角线保留为 0，但不纳入 R2 pair mask、median 或 vmax。

## 与已有 R2 作图规范的实际差异

已有 036 R2 规范是 3 英寸 panel、300 DPI、7 pt 的四指标箱线图，并明确“不适用 distance colormap”；本产物首次提供 chr1 距离矩阵图，沿用 3 英寸/300-DPI/7-pt 基准和 `coolwarm_r`，新增固定 frozen-mask 灰格、每矩阵中位数归一化 和全图统一 vmax，以便比较形状而不是绝对尺度。

## 验证

- `validation.json`：{validation['status']}；shape、对称性、有限 frozen pairs、对角线、mask count、orientation mapping、统一色标和 PNG DPI 均已检查。
- 运行 exit 由父侧命令记录在 `run_exit.json`；本脚本不写入既有训练/评价文件。
"""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PREP_DIR))
    import multires_r2 as r2

    for path in (
        CONFIG_PATH,
        MASK_LOCK_PATH,
        EVAL_RESULTS_PATH,
        FULL_R2_PATH,
        REP_PATH,
        INPUT_PROV_PATH,
        MASK_PROV_PATH,
        README_SOURCE_PATH,
        SELECTION_PATH,
        RELEASE_MANIFEST_PATH,
        R2_CODE,
    ):
        require_file(path)

    config = r2.read_json(CONFIG_PATH)
    mask_lock = r2.read_json(MASK_LOCK_PATH)
    eval_results = read_json(EVAL_RESULTS_PATH)
    input_prov = read_json(INPUT_PROV_PATH)
    mask_prov = read_json(MASK_PROV_PATH)
    representative_rows = load_tsv_rows(REP_PATH)
    full_rows = load_tsv_rows(FULL_R2_PATH)
    representative_by_variant = {row["variant_id"]: row for row in representative_rows}
    selected_endpoint_by_variant = {variant: representative_by_variant[variant]["representative_endpoint_id"] for variant in VARIANTS}
    if any(representative_by_variant[variant]["representative_source_id"] != "random_joint_base2207" for variant in VARIANTS):
        raise RuntimeError("the locked representatives are not all random_joint_base2207")

    candidate_prov = {row["endpoint_id"]: row for row in input_prov["candidate_and_terminal_hashes"]}
    selected_rows: dict[str, dict[str, str]] = {}
    for variant in VARIANTS:
        endpoint_id = selected_endpoint_by_variant[variant]
        if endpoint_id not in candidate_prov:
            raise RuntimeError(f"missing candidate provenance for {endpoint_id}")
        row = next(
            item for item in full_rows if item["endpoint_id"] == endpoint_id and item["chromosome"] == CHROMOSOME
        )
        selected_rows[variant] = row
        if row["n_bins"] != str(EXPECTED_MASK_COUNTS[0]) or row["n_common_pairs_frozen"] != str(EXPECTED_MASK_COUNTS[2]):
            raise RuntimeError(f"R2 chr1 counts changed for {endpoint_id}")

    historical_inputs = {item["condition_id"]: item for item in mask_prov["historical_input_hashes"]}
    cpu_prov = historical_inputs.get("v1_original_random_joint")
    if cpu_prov is None:
        raise RuntimeError("missing locked v1_original_random_joint historical CPU anchor")
    cpu_path = Path(cpu_prov["path"]).resolve()
    require_file(cpu_path)
    cpu_sha = sha256_file(cpu_path)
    if cpu_sha != cpu_prov["sha256"]:
        raise RuntimeError("historical CPU anchor SHA mismatch")
    cpu_row_candidates = eval_results.get("post_release_acceptance", {}).get("historical_cpu_anchor_rows", [])
    if not cpu_row_candidates:
        cpu_row_candidates = eval_results.get("historical_cpu_anchor_rows", [])
    cpu_row = next(item for item in cpu_row_candidates if item.get("chromosome") == CHROMOSOME)
    cpu_orientation = validate_orientation_row(cpu_row, "020 / C0 CPU")

    # 复用 R2 evaluator 中经过审计的 coordinate parser 和精确 frozen-mask builder。
    manifest_path, manifest, mask_rows = r2._mask_manifest_records(config)
    if sha256_file(manifest_path) != EXPECTED_MASK_MANIFEST_SHA:
        raise RuntimeError("published mask manifest SHA mismatch")
    mask_hashes: list[dict[str, str]] = []
    for item in mask_rows:
        path = r2.resolve_path(item["path"], manifest_path.parent)
        require_file(path)
        actual = sha256_file(path)
        if actual != item.get("sha256"):
            raise RuntimeError(f"mask input SHA mismatch: {item['condition_id']}")
        mask_hashes.append({"condition_id": str(item["condition_id"]), "path": str(path), "sha256": actual})
    mask_structures = r2.load_mask_structures(mask_rows, {"mask_hashes": mask_hashes})
    reference_path = Path(eval_results["reference"]["path"]).resolve()
    require_file(reference_path)
    reference_sha = sha256_file(reference_path)
    if reference_sha != REFERENCE_EXPECTED_SHA or reference_sha != eval_results["reference"]["sha256"]:
        raise RuntimeError("reference SHA mismatch")
    reference = r2.load_coordinates(reference_path, "3dg")
    chr1_mask = r2._build_mask_for_chromosome(
        CHROMOSOME,
        0,
        CHR1_LENGTH_BP,
        mask_structures,
        reference,
    )
    if (
        int(chr1_mask["n_bins"]) != EXPECTED_MASK_COUNTS[0]
        or int(chr1_mask["n_total_non_diagonal_pairs"]) != EXPECTED_MASK_COUNTS[1]
        or int(chr1_mask["n_common_pairs"]) != EXPECTED_MASK_COUNTS[2]
    ):
        raise RuntimeError(f"frozen chr1 mask counts changed: {chr1_mask['n_bins']}, {chr1_mask['n_total_non_diagonal_pairs']}, {chr1_mask['n_common_pairs']}")

    positions = np.asarray(chr1_mask["positions"], dtype=np.int64)
    if not np.array_equal(positions, np.arange(GRID_OFFSET_BP, CHR1_LENGTH_BP, BIN_SIZE_BP, dtype=np.int64)):
        raise RuntimeError("chr1 grid does not match the frozen numeric range")
    pair_i = np.asarray(chr1_mask["pair_i"], dtype=np.int64)
    pair_j = np.asarray(chr1_mask["pair_j"], dtype=np.int64)
    common_upper = np.asarray(chr1_mask["common"], dtype=bool)
    pair_mask = np.zeros((len(positions), len(positions)), dtype=bool)
    pair_mask[pair_i[common_upper], pair_j[common_upper]] = True
    pair_mask[pair_j[common_upper], pair_i[common_upper]] = True

    coordinate_paths: dict[str, Path] = {"020 / C0 CPU": cpu_path}
    coordinate_hashes: dict[str, str] = {"020 / C0 CPU": cpu_sha}
    structures_by_condition: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    structures_by_condition["020 / C0 CPU"] = r2.load_coordinates(cpu_path, "3dg")
    orientation_by_condition: dict[str, dict[str, Any]] = {"020 / C0 CPU": cpu_orientation}
    rhos_by_condition: dict[str, float | None] = {"020 / C0 CPU": float(cpu_row["matched"])}
    endpoint_ids: dict[str, str] = {"020 / C0 CPU": "historical_cpu_anchor_020_c0"}

    for variant in VARIANTS:
        endpoint_id = selected_endpoint_by_variant[variant]
        path = Path(candidate_prov[endpoint_id]["coordinate_path"]).resolve()
        require_file(path)
        actual_sha = sha256_file(path)
        if actual_sha != candidate_prov[endpoint_id]["coordinate_sha256"]:
            raise RuntimeError(f"candidate SHA mismatch: {endpoint_id}")
        condition_label = f"{variant} GPU"
        coordinate_paths[condition_label] = path
        coordinate_hashes[condition_label] = actual_sha
        structures_by_condition[condition_label] = r2.load_coordinates(path, "3dg")
        orientation_by_condition[condition_label] = validate_orientation_row(selected_rows[variant], condition_label)
        rhos_by_condition[condition_label] = float(selected_rows[variant]["matched"])
        endpoint_ids[condition_label] = endpoint_id

    grid_points = {"bp": positions.tolist(), "mb": (positions / 1_000_000.0).tolist()}
    raw_matrices: list[list[np.ndarray]] = []
    copy_labels: list[tuple[str, str]] = []
    condition_ids = ["reference", "historical_cpu_anchor_020_c0"] + [selected_endpoint_by_variant[v] for v in VARIANTS]
    condition_names = list(CONDITION_LABELS)

    reference_mat = pairwise_distance(r2.dense_points(reference, "chr1(mat)", positions))
    reference_pat = pairwise_distance(r2.dense_points(reference, "chr1(pat)", positions))
    raw_matrices.append([reference_mat, reference_pat])
    copy_labels.append(("reference maternal", "reference paternal"))

    ordered_condition_labels = ["020 / C0 CPU"] + [f"{variant} GPU" for variant in VARIANTS]
    for condition_label in ordered_condition_labels:
        structures = structures_by_condition[condition_label]
        orientation = orientation_by_condition[condition_label]
        matrix_a = pairwise_distance(r2.dense_points(structures, "c01a", positions))
        matrix_b = pairwise_distance(r2.dense_points(structures, "c01b", positions))
        if orientation["mapping"]["mat"] == "A":
            matrix_mat, matrix_pat = matrix_a, matrix_b
        else:
            matrix_mat, matrix_pat = matrix_b, matrix_a
        raw_matrices.append([matrix_mat, matrix_pat])
        copy_labels.append(("mat-matched", "pat-matched"))

    raw = np.stack([np.stack(row, axis=0) for row in raw_matrices], axis=0).astype(np.float64)
    if raw.shape != (7, 2, 193, 193):
        raise RuntimeError(f"unexpected raw matrix shape: {raw.shape}")
    diagonal = np.diag_indices(193)
    raw[:, :, diagonal[0], diagonal[1]] = 0.0
    divisors: list[list[float]] = []
    normalized = np.full_like(raw, np.nan, dtype=np.float64)
    matrix_stats: list[dict[str, Any]] = []
    all_display_values: list[np.ndarray] = []
    for condition_index, condition_label in enumerate(condition_names):
        condition_divisors: list[float] = []
        for copy_index, copy_name in enumerate(COPY_KEYS):
            values = raw[condition_index, copy_index, pair_i[common_upper], pair_j[common_upper]]
            if not np.isfinite(values).all():
                raise RuntimeError(f"nonfinite {condition_label} {copy_name} value on frozen R2 pairs")
            divisor = float(np.median(values))
            if not math.isfinite(divisor) or divisor <= 0.0:
                raise RuntimeError(f"invalid normalization divisor for {condition_label} {copy_name}: {divisor}")
            condition_divisors.append(divisor)
            normalized[condition_index, copy_index] = raw[condition_index, copy_index] / divisor
            normalized[condition_index, copy_index, diagonal[0], diagonal[1]] = 0.0
            all_display_values.append(values / divisor)
        divisors.append(condition_divisors)

    display_values = np.concatenate(all_display_values)
    if not np.isfinite(display_values).all() or len(display_values) != 7 * 2 * EXPECTED_MASK_COUNTS[2]:
        raise RuntimeError("invalid combined normalized display values")
    vmax = float(np.percentile(display_values, 99.0))
    if not math.isfinite(vmax) or vmax <= V_MIN:
        raise RuntimeError(f"invalid unified vmax: {vmax}")
    clipped_fraction: list[list[float]] = []
    for condition_index in range(7):
        fractions: list[float] = []
        for copy_index in range(2):
            values = normalized[condition_index, copy_index, pair_i[common_upper], pair_j[common_upper]]
            fractions.append(float(np.count_nonzero(values > vmax) / len(values)))
        clipped_fraction.append(fractions)
        matrix_stats.append(
            {
                "condition": condition_names[condition_index],
                "divisors": {COPY_KEYS[i]: divisors[condition_index][i] for i in range(2)},
                "normalization_pair_count": int(len(display_values) // 14),
                "clipped_fraction": {COPY_KEYS[i]: fractions[i] for i in range(2)},
            }
        )

    arrays_path = OUT_DIR / "distance_arrays.npz"
    np.savez_compressed(
        arrays_path,
        raw_distance=raw,
        normalized_distance=normalized,
        frozen_pair_mask=pair_mask,
        frozen_pair_i=pair_i,
        frozen_pair_j=pair_j,
        frozen_pair_common_upper=common_upper,
        grid_bp=positions,
        grid_mb=positions / 1_000_000.0,
        condition_ids=np.asarray(condition_ids),
        condition_labels=np.asarray(condition_names),
        copy_ids=np.asarray(COPY_KEYS),
    )

    rhos = (None, rhos_by_condition["020 / C0 CPU"]) + tuple(rhos_by_condition[f"{variant} GPU"] for variant in VARIANTS)
    outputs = render_heatmaps(
        normalized,
        pair_mask,
        positions / 1_000_000.0,
        tuple(condition_names),
        tuple(copy_labels),
        tuple(rhos),
        vmax,
    )

    source_files: dict[str, dict[str, str]] = {}
    for path in (
        EVAL_RESULTS_PATH,
        FULL_R2_PATH,
        REP_PATH,
        INPUT_PROV_PATH,
        MASK_PROV_PATH,
        README_SOURCE_PATH,
        SELECTION_PATH,
        RELEASE_MANIFEST_PATH,
        R2_CODE,
        CONFIG_PATH,
        MASK_LOCK_PATH,
        manifest_path,
    ):
        source_files[path.name] = {"path": str(path), "sha256": sha256_file(path)}

    orientation_metadata: dict[str, Any] = {}
    for condition_label in ordered_condition_labels:
        orientation_metadata[condition_label] = {
            "endpoint_id": endpoint_ids[condition_label],
            "orientation": orientation_by_condition[condition_label]["orientation"],
            "raw_A_B_to_mat_pat": orientation_by_condition[condition_label]["mapping"],
            "rho_A_mat": orientation_by_condition[condition_label]["rho_A_mat"],
            "rho_A_pat": orientation_by_condition[condition_label]["rho_A_pat"],
            "rho_B_mat": orientation_by_condition[condition_label]["rho_B_mat"],
            "rho_B_pat": orientation_by_condition[condition_label]["rho_B_pat"],
            "stored_chr1_matched_rho": orientation_by_condition[condition_label]["matched"],
        }
    orientation_metadata["Reference"] = {
        "endpoint_id": "P9016.1m reference",
        "orientation": "fixed_reference_columns",
        "raw_A_B_to_mat_pat": {"mat": "chr1(mat)", "pat": "chr1(pat)"},
        "stored_chr1_matched_rho": None,
    }

    metadata = {
        "schema_version": "p9016-036-r2-chr1-distance-heatmaps-v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "complete",
        "source": {
            "experiment": "036 GPU multiresolution C0/C1/C2-map/C2-free/C3 controlled experiment",
            "r2_directory": str(R2_DIR),
            "evaluation_results": str(EVAL_RESULTS_PATH),
            "selection_locked_representatives": True,
            "representatives_all_random_joint": True,
            "no_training_or_optimization_called": True,
            "native_called": False,
            "new_r2_or_r1_or_r3_called": False,
            "source_files": source_files,
            "coordinate_files": {
                condition: {"path": str(coordinate_paths[condition]), "sha256": coordinate_hashes[condition]}
                for condition in coordinate_paths
            },
            "reference": {"path": str(reference_path), "sha256": reference_sha, "columns": ["chr1(mat)", "chr1(pat)"]},
            "r2_code": {"path": str(R2_CODE), "sha256": sha256_file(R2_CODE)},
            "mask_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "condition_count": 21},
            "mask_input_hashes": mask_hashes,
        },
        "condition_order": condition_names,
        "condition_ids": condition_ids,
        "copy_order": ["mat", "pat"],
        "copy_labels": {condition_names[i]: list(copy_labels[i]) for i in range(7)},
        "grid": {
            "chromosome": CHROMOSOME,
            "length_bp": CHR1_LENGTH_BP,
            "rule": "range(3_000_000,195471971,1_000_000)",
            "offset_bp": GRID_OFFSET_BP,
            "bin_size_bp": BIN_SIZE_BP,
            "n_bins": int(len(positions)),
            "grid_bp_first_last": [int(positions[0]), int(positions[-1])],
            "grid_mb_first_last": [float(positions[0] / 1_000_000.0), float(positions[-1] / 1_000_000.0)],
        },
        "frozen_r2_pair_mask": {
            "policy": "exact reuse of published 029 21-condition mask; new 036 endpoints never entered or shrank it",
            "pair_definition": "unordered upper-triangle off-diagonal numeric-grid pairs",
            "n_total_non_diagonal_pairs": int(len(pair_i)),
            "n_common_pairs": int(common_upper.sum()),
            "n_bins": int(len(positions)),
            "mask_matrix_shape": list(pair_mask.shape),
            "mask_outside_plot_color": "#d0d0d0",
            "diagonal_in_mask": False,
        },
        "orientation": {
            "policy": "fixed from existing 036 R2 chr1 rows; one whole-chromosome best-swap, no local swap or representative reselection",
            "labels": "mat-matched / pat-matched are evaluation alignment labels, not training parent labels",
            "by_condition": orientation_metadata,
        },
        "normalization": {
            "formula": "raw distance / median raw distance on the same frozen chr1 R2 unordered offdiag pairs",
            "pair_count_per_matrix": int(common_upper.sum()),
            "divisors_raw_median": {
                condition_names[i]: {COPY_KEYS[j]: divisors[i][j] for j in range(2)} for i in range(7)
            },
            "vmin": V_MIN,
            "vmax": vmax,
            "vmax_rule": "99th percentile of all 14 matrices' normalized values on frozen R2 pairs",
            "combined_valid_normalized_offdiag_count": int(len(display_values)),
            "matrix_stats": matrix_stats,
        },
        "arrays": {
            "path": str(arrays_path),
            "raw_key": "raw_distance",
            "normalized_key": "normalized_distance",
            "mask_key": "frozen_pair_mask",
            "grid_keys": ["grid_bp", "grid_mb"],
            "shape": [7, 2, 193, 193],
            "dtype": "float64",
            "raw_distance_outside_mask_retained": True,
            "plot_values_outside_mask_masked": True,
            "diagonal_value": 0.0,
        },
        "rendering": {
            "colormap": "coolwarm_r",
            "font_size_pt": FONT_SIZE_PT,
            "dpi": DPI,
            "overview_figsize_inches": [6.8, 21.0],
            "detail_figsize_inches": [6.0, 3.0],
            "axis_range_mb": [float(positions[0] / 1_000_000.0 - 0.5), float(positions[-1] / 1_000_000.0 + 0.5)],
            "axis_ticks_mb": [3, 43, 83, 123, 163, 193],
            "colorbar_label": "distance / median distance on frozen R2 pairs",
        },
        "outputs": outputs,
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
    }
    metadata_path = OUT_DIR / "metadata.json"
    write_json(metadata_path, metadata)

    validation_errors: list[str] = []
    if raw.shape != (7, 2, 193, 193) or normalized.shape != (7, 2, 193, 193):
        validation_errors.append("matrix shape")
    for name, array in (("raw", raw), ("normalized", normalized)):
        if not np.allclose(array, np.swapaxes(array, -1, -2), equal_nan=True, atol=0.0, rtol=0.0):
            validation_errors.append(f"{name} symmetry")
        if not np.allclose(array[:, :, diagonal[0], diagonal[1]], 0.0, atol=0.0, rtol=0.0):
            validation_errors.append(f"{name} diagonal")
    if not np.isfinite(raw[:, :, pair_i[common_upper], pair_j[common_upper]]).all():
        validation_errors.append("raw finite frozen pairs")
    if not np.isfinite(normalized[:, :, pair_i[common_upper], pair_j[common_upper]]).all():
        validation_errors.append("normalized finite frozen pairs")
    if int(pair_mask.sum() // 2) != EXPECTED_MASK_COUNTS[2] or int(common_upper.sum()) != EXPECTED_MASK_COUNTS[2]:
        validation_errors.append("mask pair count")
    if len(positions) != EXPECTED_MASK_COUNTS[0] or int(len(pair_i)) != EXPECTED_MASK_COUNTS[1]:
        validation_errors.append("grid/total pair count")
    if any(orientation_by_condition[label]["mapping"] not in ({"mat": "A", "pat": "B"}, {"mat": "B", "pat": "A"}) for label in ordered_condition_labels):
        validation_errors.append("orientation mapping")
    image_records = []
    for path in [Path(outputs["overview"]["png"]), Path(outputs["overview"]["pdf"])] + [
        Path(detail[key]) for detail in outputs["details"].values() for key in ("png", "pdf")
    ]:
        record = image_validation(path)
        image_records.append(record)
        if not record.get("exists") or record.get("size_bytes", 0) <= 0:
            validation_errors.append(f"missing image {path.name}")
        if path.suffix == ".png" and record.get("dpi_300_ok") is False:
            validation_errors.append(f"PNG dpi {path.name}")
    validation = {
        "schema_version": "p9016-036-r2-chr1-distance-heatmaps-validation-v1",
        "status": "PASS" if not validation_errors else "FAIL",
        "exit_code_intended": 0 if not validation_errors else 1,
        "errors": validation_errors,
        "matrix_shape": list(raw.shape),
        "symmetry_checked": True,
        "finite_frozen_pair_values_checked": True,
        "diagonal_zero_checked": True,
        "grid_n_bins": int(len(positions)),
        "mask_n_total_non_diagonal_pairs": int(len(pair_i)),
        "mask_n_common_pairs": int(common_upper.sum()),
        "all_images_same_vmin": True,
        "all_images_same_vmax": True,
        "vmin": V_MIN,
        "vmax": vmax,
        "font_size_pt": FONT_SIZE_PT,
        "dpi": DPI,
        "image_records": image_records,
        "normalization_pair_count_per_matrix": int(common_upper.sum()),
        "no_new_metric_selection": True,
        "no_training_or_optimization": True,
    }
    write_json(OUT_DIR / "validation.json", validation)
    readme = make_readme(metadata, validation)
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"status": validation["status"], "output": str(OUT_DIR), "vmax": vmax, "mask_pairs": int(common_upper.sum())}, ensure_ascii=False))
    if validation_errors:
        raise RuntimeError("validation failed: " + ", ".join(validation_errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
