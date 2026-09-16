#!/usr/bin/env python
"""049 追加图（用户追加请求）：chr1 distance matrices，2 行 x 5 列。

10 个 panel = reference(2 拷贝) + 046 baseline(2) + 三个冻结 display endpoint
（A-ms-random / B-raw-consensus / C-ms-random，各 2 拷贝）；行 = 与 reference
mat / pat 对应的拷贝，对应关系取自 049 已冻结的 chr1 intra-chromosome Pearson
whole-chromosome direct/swapped matching（``evaluation/results/r2_per_chromosome.tsv``），
不做逐 copy 独立匹配。

冻结口径：
- mask 复用 046 ``frozen_legacy_mask_snapshot.npz``（SHA 校验），``global =
  chromosome_offset + positions // 1Mb``；chr1 为 193 个 old21 numeric positions（3..195 Mb），
  17578 个 common upper pairs；只画 common pairs 及其镜像，对角线只在 188 个 common-valid
  bins 置 0（仅展示，评估仍排除），其余格点灰色。
- 尺度：每个 dataset 的两拷贝共用一次 whole-cell Rg（在全部 20 条染色体共同有限支持
  2447 loci / 4894 beads 上计算，与 049 whole-genome 对比图的 gauge scale 同一口径）；
  不对 chr1 单独缩放、不逐 copy 缩放；距离对旋转平移不变，故不再对齐。
- 全部 10 个 panel 共用一个 ``coolwarm_r`` colorbar，vmin=0，vmax=所有 panel 有限值最大值。

本脚本不拟合、不重算评价、不写冻结坐标/manifest/selection/既有报告或原图；
只写 1 张 PNG 与 1 个 metrics JSON。
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 不写 __pycache__：本步骤只应新增 1 张 PNG 与 1 个 metrics JSON，不往 run 树里留派生物
sys.dont_write_bytecode = True
import eval049_inputs as inputs  # noqa: E402
import eval049_lib as lib  # noqa: E402
import eval049_evaluate as evaluate  # noqa: E402
import eval049_plots as plots  # noqa: E402  （导入时设置 MPLCONFIGDIR，并复用其样式常量）

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

PLOTS = lib.EVAL / "plots"
RESULTS = lib.EVAL / "results"
REFERENCE_OPEN_PATH = lib.EVAL / "gates/reference_open.json"
R2_PER_CHROMOSOME = RESULTS / "r2_per_chromosome.tsv"
SELECTION_PATH = lib.RUN / "results/selection_pre_reference.json"
MANIFEST_PATH = lib.RUN / "results/endpoint_manifest_pre_reference.json"
FIGURE_PATH = PLOTS / "049_chr1_distance_matrices.png"
METRICS_PATH = RESULTS / "chr1_distance_matrix_metrics.json"

BASELINE_FIT_ID = "baseline-046-real-extension-G-full-J"
LOSES = ("A", "B", "C")
# 用户追加请求中声明的冻结 display endpoint（必须与 selection 一致，否则报错退出）
EXPECTED_DISPLAY = {"A": "A-ms-random", "B": "B-raw-consensus", "C": "C-ms-random"}
COLUMN_KEYS = ("reference", BASELINE_FIT_ID, "A-ms-random", "B-raw-consensus", "C-ms-random")
COLUMN_LABELS = {
    "reference": "Reference",
    BASELINE_FIT_ID: "046 baseline",
    "A-ms-random": "A marginal G",
    "B-raw-consensus": "B hard observed",
    "C-ms-random": "C max rate",
}
ROW_LABELS = ("copy matched to reference mat", "copy matched to reference pat")
CHROMOSOME = "chr1"
CMAP_NAME = "coolwarm_r"
BAD_COLOR = "#d9d9d9"
XLABEL = "Chr1 position (Mb)"
# 中英分工：图的标题/坐标轴/图例/图注为英文（内部记录保持中文）
CAPTION = (
    "chr1 intra-chromosome distance matrices on the frozen 046 old21 mask: 193 numeric positions (3-195 Mb), "
    "17578 common upper pairs drawn and mirrored.\n"
    "grey = missing bin or non-common pair; the diagonal is set to 0 only at the 188 common-valid bins for display "
    "and stays excluded from every evaluation.\n"
    "Each dataset (both copies jointly) is divided by one whole-cell Rg computed on the shared all-20-chromosome "
    "support (2447 loci / 4894 beads); no per-chromosome or per-copy scaling, no alignment (distances are "
    "rotation/translation invariant).\n"
    "Candidate copy labels come from the frozen chr1 whole-chromosome intra-chromosome Pearson direct/swapped "
    "matching (r2_per_chromosome.tsv); r = Pearson r of that copy against the reference copy named in its row."
)


def sha256(path: Path) -> str:
    return lib.sha256_file(Path(path))


def load_chr1_r2_table(path: Path) -> dict[str, dict[str, Any]]:
    """chr1 Pearson 冻结匹配表：per candidate_id 保留 orientation 与四个 rho。"""
    table: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if str(row["chromosome"]) != CHROMOSOME or str(row["metric"]) != "pearson":
                continue
            rho = lib.parse_rho_cell(str(row["rho"]))
            table[str(row["candidate_id"])] = {
                "orientation": str(row["orientation"]),
                "geometry_tie": str(row["geometry_tie"]).strip().lower() == "true",
                "direct": float(row["direct"]),
                "swapped": float(row["swapped"]),
                "matched": float(row["matched"]),
                "contrast": float(row["contrast"]),
                "n_pairs": int(row["n_pairs"]),
                "rho": {key: float(rho[key]) for key in ("A_mat", "A_pat", "B_mat", "B_pat")},
            }
    return table


def row_copy_assignment(row: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    """按冻结 chr1 orientation 给出 row1(mat)/row2(pat) 各取哪个候选拷贝，以及匹配 r。"""
    if row["geometry_tie"]:
        raise RuntimeError("chr1 matching is an unresolved geometry tie for %s; refuse to assign rows" % dataset_id)
    rho = row["rho"]
    orientation = str(row["orientation"])
    if orientation == "direct":
        mat_copy, pat_copy = 0, 1
    elif orientation == "swapped":
        mat_copy, pat_copy = 1, 0
    else:
        raise RuntimeError("unexpected chr1 orientation %r for %s" % (orientation, dataset_id))
    name = {0: "copyA", 1: "copyB"}
    return {
        "orientation": orientation,
        "matched": float(row["matched"]),
        "contrast": float(row["contrast"]),
        "row_mat": {"copy_index": mat_copy, "copy_label": name[mat_copy],
                    "r": float(rho["%s_mat" % name[mat_copy][-1].upper()])},
        "row_pat": {"copy_index": pat_copy, "copy_label": name[pat_copy],
                    "r": float(rho["%s_pat" % name[pat_copy][-1].upper()])},
        "rho": {key: float(value) for key, value in rho.items()},
    }


def whole_cell_gauge_scale(coords_support: np.ndarray) -> tuple[float, np.ndarray]:
    """049 whole-genome 图同口径的 gauge scale：两拷贝合并去中心 + 单位 Rg。

    与 ``eval049_plots.gauge_points`` 的缩放部分完全相同（那里额外做一次 proper
    Kabsch 旋转；距离对旋转不变，故本图不需要）。
    """
    points = np.asarray(coords_support, dtype=np.float64).reshape(-1, 3)
    centered = points - points.mean(axis=0)
    rg = math.sqrt(float(np.mean(np.sum(centered ** 2, axis=1))))
    if not math.isfinite(rg) or rg <= 0.0:
        raise RuntimeError("non-positive whole-cell Rg")
    return rg, centered


def chr1_matrix(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray, valid_bins: np.ndarray,
                n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """只填 common upper pairs 及其镜像；对角线在 common-valid bins 置 0（仅展示）。"""
    matrix = np.full((n_bins, n_bins), np.nan, dtype=np.float64)
    distances = np.sqrt(np.sum((points[pair_i] - points[pair_j]) ** 2, axis=1))
    matrix[pair_i, pair_j] = distances
    matrix[pair_j, pair_i] = distances
    matrix[valid_bins, valid_bins] = 0.0
    return matrix, distances


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", default=str(FIGURE_PATH))
    parser.add_argument("--metrics", default=str(METRICS_PATH))
    parser.add_argument("--force", action="store_true", help="允许覆盖同名输出（默认拒绝）")
    args = parser.parse_args()
    figure_path, metrics_path = Path(args.figure), Path(args.metrics)
    for path in (figure_path, metrics_path):
        if path.exists() and not args.force:
            raise RuntimeError("refusing to overwrite existing output: %s (use --force)" % path)

    plots.apply_style()
    created = lib.utc_now()

    # 1) 候选/基线 SHA 先算，再打开 reference（科学隔离顺序沿用 049 评价）
    manifest = inputs.load_manifest(MANIFEST_PATH)
    selection = inputs.load_selection(SELECTION_PATH)
    display = {loss: str(selection["display"][loss]["fit_id"]) for loss in LOSES}
    if display != EXPECTED_DISPLAY:
        raise RuntimeError("selection display endpoints changed: %s != %s" % (display, EXPECTED_DISPLAY))
    lut = {str(row["fit_id"]): row for row in manifest["fits"]}
    baseline_manifest = manifest["baseline"]
    if baseline_manifest is None:
        raise RuntimeError("manifest has no baseline record")

    panel_ids = ["reference"] + [COLUMN_KEYS[1]] + [display[loss] for loss in LOSES]
    if list(panel_ids) != list(COLUMN_KEYS):
        raise RuntimeError("panel order changed: %s" % panel_ids)

    npz_paths: dict[str, Path] = {}
    npz_recorded: dict[str, str] = {}
    for fit_id in panel_ids[1:]:
        if fit_id == BASELINE_FIT_ID:
            path, recorded = lib.BASELINE_NPZ, str(baseline_manifest["npz_sha256"])
        else:
            row = lut[fit_id]
            path, recorded = lib.resolve_path(row["npz_path"]), str(row["npz_sha256"])
        npz_paths[fit_id], npz_recorded[fit_id] = path, recorded
    for loss in LOSES:
        declared = selection["display"][loss].get("coords_sha256")
        if declared and str(declared) != npz_recorded[display[loss]]:
            raise RuntimeError("selection coordsSHA disagrees with manifest for %s" % display[loss])
    npz_actual = {fit_id: sha256(npz_paths[fit_id]) for fit_id in panel_ids[1:]}
    mismatched = {fit_id: {"recorded": npz_recorded[fit_id], "actual": npz_actual[fit_id]}
                  for fit_id in npz_actual if npz_actual[fit_id] != npz_recorded[fit_id]}
    if mismatched:
        raise RuntimeError("candidate coordinate SHA mismatch (before opening the reference): %s" % mismatched)
    input_hashes_at_start = {"manifest": manifest["sha256"], "selection": selection["sha256"],
                             "baseline_npz": npz_actual[BASELINE_FIT_ID],
                             **{fit_id: npz_actual[fit_id] for fit_id in display.values()}}

    # 2) reference：只有 gate 记录为 opened、且文件哈希与 gate 记录一致才读
    if not REFERENCE_OPEN_PATH.is_file():
        raise RuntimeError("reference has not been opened (no gates/reference_open.json)")
    open_record = lib.read_json(REFERENCE_OPEN_PATH)
    if str(open_record.get("status")) != "opened":
        raise RuntimeError("reference_open.json status=%r" % open_record.get("status"))
    ref_actual = sha256(lib.REFERENCE_PATH)
    if ref_actual != str(open_record["reference_sha256"]):
        raise RuntimeError("reference SHA changed: %s != %s" % (ref_actual, open_record["reference_sha256"]))
    if ref_actual != lib.REFERENCE_SHA256:
        raise RuntimeError("reference SHA differs from eval049_lib.REFERENCE_SHA256")
    input_hashes_at_start["reference"] = ref_actual

    # 3) mask：复用 049 冻结加载器（内部校验 SHA 与总量）
    data = lib.Aggregate()
    mask_audit = data.audit()
    masks, mask_report = evaluate.load_real_masks(data)
    mask = masks[CHROMOSOME]
    positions = np.asarray(mask["positions"], dtype=np.int64)
    pair_i = np.asarray(mask["pair_i"], dtype=np.int64)
    pair_j = np.asarray(mask["pair_j"], dtype=np.int64)
    common = np.asarray(mask["common"], dtype=bool)
    valid_local = np.asarray(mask["valid_local_bins"], dtype=np.int64)
    if int(common.sum()) != int(mask["n_common_pairs"]) or int(valid_local.size) != 188:
        raise RuntimeError("chr1 mask shape changed: common=%d valid=%d" % (int(common.sum()), int(valid_local.size)))
    global_indices = lib.mask_global_indices(data.offsets, mask)
    input_hashes_at_start["mask_snapshot"] = str(mask_report["snapshot_sha256"])
    input_hashes_at_start["r2_per_chromosome_tsv"] = sha256(R2_PER_CHROMOSOME)

    # 4) 载入 5 个 dataset 的坐标与 reference；共同有限支持（全 20 chr、两拷贝）
    reference = lib.three_dg_to_array(lib.load_3dg(lib.REFERENCE_PATH), data, track_mode="reference")
    coords: dict[str, np.ndarray] = {"reference": reference}
    for fit_id in panel_ids[1:]:
        coords[fit_id] = lib.load_coords_npz(npz_paths[fit_id])["coordinates"]
    finite_masks = [np.isfinite(coords[panel_id]).all(axis=(0, 2)) for panel_id in panel_ids]
    support = np.logical_and.reduce(finite_masks)
    n_support = int(support.sum())
    if n_support != int(mask_report["totals"]["valid_bins"]) or n_support != 2447:
        raise RuntimeError("all-20-chromosome common finite support changed: %d" % n_support)
    support_global = np.zeros(data.n_loci, dtype=bool)
    per_chromosome_support = {}
    for ci, name in enumerate(data.chromosome_names):
        local = np.asarray(masks[str(name)]["valid_local_bins"], dtype=np.int64)
        index = lib.mask_global_indices(data.offsets, masks[str(name)])
        support_global[index[local]] = True
        per_chromosome_support[str(name)] = int(local.size)
    if not np.array_equal(support, support_global):
        raise RuntimeError("common finite support != frozen mask valid bins (all 20 chromosomes)")
    if any(count <= 0 for count in per_chromosome_support.values()):
        raise RuntimeError("some chromosome contributes no locus to the Rg support")

    # 5) 每 dataset 一次 whole-cell Rg（两拷贝合并、全 20 chr 支持）；距离对旋转平移不变，不再对齐
    scaled: dict[str, np.ndarray] = {}
    rg_by_dataset: dict[str, float] = {}
    for panel_id in panel_ids:
        rg, _ = whole_cell_gauge_scale(coords[panel_id][:, support, :])
        rg_by_dataset[panel_id] = rg
        scaled[panel_id] = coords[panel_id] / rg

    # 6) chr1 矩阵（common upper pairs + 镜像；对角线仅对 common-valid bins 置 0）
    n_bins = int(positions.size)
    local_support = support[global_indices]
    if not np.array_equal(np.flatnonzero(local_support), valid_local):
        raise RuntimeError("chr1 mask positions inside the support != common-valid bins")
    matrices: dict[tuple[str, int], np.ndarray] = {}
    pair_distances: dict[tuple[str, int], np.ndarray] = {}
    for panel_id in panel_ids:
        for copy in (0, 1):
            matrix, distances = chr1_matrix(scaled[panel_id][copy][global_indices],
                                            pair_i[common], pair_j[common], valid_local, n_bins)
            matrices[(panel_id, copy)] = matrix
            pair_distances[(panel_id, copy)] = distances

    # 7) 冻结 chr1 匹配 -> 行归属
    r2_table = load_chr1_r2_table(R2_PER_CHROMOSOME)
    assignment: dict[str, Any] = {
        "reference": {"orientation": "not_applicable",
                      "row_mat": {"copy_index": 0, "copy_label": "reference mat", "r": None},
                      "row_pat": {"copy_index": 1, "copy_label": "reference pat", "r": None},
                      "note": "reference panel is the target itself"},
    }
    for fit_id in panel_ids[1:]:
        if fit_id not in r2_table:
            raise RuntimeError("chr1 Pearson row missing in r2_per_chromosome.tsv: %s" % fit_id)
        assignment[fit_id] = row_copy_assignment(r2_table[fit_id], fit_id)

    # 8) 图上标注的 r 必须与正式 chr1 R2 表 <=1e-12 一致；矩阵重算 r 同样核对
    annotation_check: dict[str, Any] = {}
    max_annotation_diff = 0.0
    max_recomputed_diff = 0.0
    for fit_id in panel_ids[1:]:
        entry = assignment[fit_id]
        for row_key, ref_copy in (("row_mat", 0), ("row_pat", 1)):
            label, value = str(entry[row_key]["copy_label"]), float(entry[row_key]["r"])
            key = "%s_%s" % (label[-1].upper(), "mat" if ref_copy == 0 else "pat")
            frozen = float(r2_table[fit_id]["rho"][key])
            gap = abs(value - frozen)
            max_annotation_diff = max(max_annotation_diff, gap)
            recomputed = lib.metric("pearson", pair_distances[(fit_id, int(entry[row_key]["copy_index"]))],
                                    pair_distances[("reference", ref_copy)])
            recomputed_gap = abs(recomputed - frozen)
            max_recomputed_diff = max(max_recomputed_diff, recomputed_gap)
            annotation_check["%s|%s" % (fit_id, row_key)] = {
                "annotated_r": value, "frozen_r2_table_r": frozen, "abs_diff": gap,
                "recomputed_from_plotted_pairs": recomputed, "recomputed_abs_diff": recomputed_gap,
            }
    if max_annotation_diff > 1e-12:
        raise RuntimeError("annotated r disagrees with the frozen chr1 R2 table: %.3e" % max_annotation_diff)
    if max_recomputed_diff > 1e-12:
        raise RuntimeError("matrix-recomputed r disagrees with the frozen chr1 R2 table: %.3e" % max_recomputed_diff)

    # 9) 对称性 / 共同 mask / 共同色标（逐项实测，不写死）
    matrix_checks: dict[str, Any] = {}
    nan_patterns: dict[str, np.ndarray] = {}
    vmax = 0.0
    symmetric_ok = True
    for panel_id in panel_ids:
        for copy in (0, 1):
            matrix = matrices[(panel_id, copy)]
            symmetric_ok &= bool(np.array_equal(matrix, matrix.T, equal_nan=True))
            if not symmetric_ok:
                raise RuntimeError("matrix not symmetric: %s copy%d" % (panel_id, copy))
            finite = matrix[np.isfinite(matrix)]
            if finite.size != 2 * int(common.sum()) + int(valid_local.size):
                raise RuntimeError("unexpected finite cell count for %s copy%d: %d"
                                   % (panel_id, copy, finite.size))
            if float(np.nanmin(matrix)) < 0.0:
                raise RuntimeError("negative distance in %s copy%d" % (panel_id, copy))
            vmax = max(vmax, float(finite.max()))
            tag = "%s|copy%d" % (panel_id, copy)
            nan_patterns[tag] = np.isnan(matrix)
            matrix_checks[tag] = {
                "finite_cells": int(finite.size), "gray_cells": int(matrix.size - finite.size),
                "min": float(finite.min()), "max": float(finite.max()),
                "diagonal_zero_bins": int(valid_local.size),
                "symmetric_exact": True, "common_pairs": int(common.sum()),
            }
    if not math.isfinite(vmax) or vmax <= 0.0:
        raise RuntimeError("non-positive shared colour maximum")
    shared_scale_ok = all(float(record["max"]) <= vmax for record in matrix_checks.values())
    if not shared_scale_ok:
        raise RuntimeError("some panel exceeds the shared colour maximum")
    first_tag = next(iter(nan_patterns))
    mask_pattern_ok = all(np.array_equal(pattern, nan_patterns[first_tag]) for pattern in nan_patterns.values())
    if not mask_pattern_ok:
        raise RuntimeError("grey (missing) cell pattern differs across the 10 panels")
    mask_ok = str(mask_report["snapshot_sha256"]) == lib.MASK_SNAPSHOT_SHA256
    if not mask_ok:
        raise RuntimeError("mask snapshot SHA differs from the frozen constant")
    candidate_ok = all(npz_actual[fit_id] == npz_recorded[fit_id] for fit_id in npz_actual)
    reference_ok = bool(ref_actual == str(open_record["reference_sha256"]) == lib.REFERENCE_SHA256)
    all20_chr_ok = bool(per_chromosome_support and all(count > 0 for count in per_chromosome_support.values()))

    # 10) 绘图：2 行 x 5 列，每基础 panel 3 英寸，300 dpi，英文 7 pt
    #     版式按英寸显式排版：每个 axes box 恰为 base x base，四周留出标题/共享 colorbar/图注空间
    base = float(plots.BASE_PANEL_INCHES)
    dpi = int(plots.DPI)
    wspace, hspace = 0.16, 0.34
    left_band, right_band, top_band, bottom_band = 0.70, 0.10, 0.55, 1.65
    axes_region_w = base * (5.0 + 4.0 * wspace)
    axes_region_h = base * (2.0 + hspace)
    fig_w = left_band + axes_region_w + right_band
    fig_h = top_band + axes_region_h + bottom_band
    axes_x0, axes_x1 = left_band / fig_w, 1.0 - right_band / fig_w
    axes_y0, axes_y1 = bottom_band / fig_h, 1.0 - top_band / fig_h
    figure, axes = plt.subplots(2, 5, figsize=(fig_w, fig_h))
    figure.subplots_adjust(left=axes_x0, right=axes_x1, top=axes_y1, bottom=axes_y0,
                           wspace=wspace, hspace=hspace)
    cmap = plt.get_cmap(CMAP_NAME).copy()
    cmap.set_bad(BAD_COLOR)
    lo_mb, hi_mb = float(positions[0]) / 1e6, float(positions[-1]) / 1e6
    extent = (lo_mb - 0.5, hi_mb + 0.5, lo_mb - 0.5, hi_mb + 0.5)
    row_keys = ("row_mat", "row_pat")
    panel_titles: dict[str, str] = {}
    for column, panel_id in enumerate(panel_ids):
        for row, row_key in enumerate(row_keys):
            ax = axes[row, column]
            entry = assignment[panel_id]
            copy = int(entry[row_key]["copy_index"])
            image = ax.imshow(matrices[(panel_id, copy)], cmap=cmap, vmin=0.0, vmax=vmax,
                              origin="lower", extent=extent, interpolation="nearest", aspect="auto")
            title = COLUMN_LABELS[panel_id]
            label = str(entry[row_key]["copy_label"])
            if entry[row_key]["r"] is None:
                title += "\n%s" % label
            else:
                title += "\n%s, r=%.3f" % (label, float(entry[row_key]["r"]))
            ax.set_title(title, pad=3.0)
            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.tick_params(length=1.5, pad=1.0)
            if row == 1:
                ax.set_xlabel(XLABEL, labelpad=1.0)
            if column == 0:
                ax.set_ylabel(XLABEL, labelpad=1.0)
            panel_titles["%s|%s" % (panel_id, row_key)] = title.replace("\n", " / ")
    colorbar_axes = figure.add_axes([0.36, 0.67 / fig_h, 0.28, 0.14 / fig_h])
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0.0, vmax), cmap=cmap),
                               cax=colorbar_axes, orientation="horizontal")
    colorbar.set_label("chr1 distance, normalised by whole-cell Rg", fontsize=plots.FONT_PT, labelpad=1.5)
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(length=1.5, pad=1.0, labelsize=plots.FONT_PT)
    colorbar.outline.set_linewidth(0.4)
    figure.text(0.5, 1.0 - 0.18 / fig_h,
                "049 chr1 distance matrices: reference, 046 baseline and the three frozen display "
                "endpoints (one shared colour scale, vmin=0, vmax=%.4f Rg)" % vmax,
                ha="center", va="top", fontsize=plots.FONT_PT)
    figure.text(0.5, 0.61 / fig_h, CAPTION, ha="center", va="top", fontsize=plots.FONT_PT, linespacing=1.35)
    for row, row_key in enumerate(row_keys):
        box = axes[row, 0].get_position()
        figure.text(0.006, (box.y0 + box.y1) / 2.0, ROW_LABELS[row], rotation=90, ha="left", va="center",
                    fontsize=plots.FONT_PT)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=dpi)
    axes_box = axes[0, 0].get_position()
    axes_box_inches = [round(axes_box.width * fig_w, 4), round(axes_box.height * fig_h, 4)]
    plt.close(figure)
    if not figure_path.is_file() or figure_path.stat().st_size < 10_000:
        raise RuntimeError("figure output missing or too small: %s" % figure_path)

    # 11) 写 metrics（只此一个 JSON）；随后复核输入哈希未变
    input_hashes_after = {
        "manifest": sha256(MANIFEST_PATH), "selection": sha256(SELECTION_PATH),
        "baseline_npz": sha256(npz_paths[BASELINE_FIT_ID]),
        **{fit_id: sha256(npz_paths[fit_id]) for fit_id in display.values()},
        "reference": sha256(lib.REFERENCE_PATH), "mask_snapshot": sha256(lib.MASK_SNAPSHOT),
        "r2_per_chromosome_tsv": sha256(R2_PER_CHROMOSOME),
    }
    if input_hashes_after != input_hashes_at_start:
        raise RuntimeError("input hashes changed during the run")

    metrics = {
        "schema": "p9016-049-chr1-distance-matrices-v1",
        "created_at_utc": created,
        "run": str(lib.RUN),
        "outputs": {"figure": str(figure_path), "figure_bytes": int(figure_path.stat().st_size),
                    "figure_sha256": sha256(figure_path), "metrics": str(metrics_path)},
        "figure_spec": {
            "layout": "2 rows x 5 columns", "columns": [COLUMN_LABELS[key] for key in panel_ids],
            "rows": list(ROW_LABELS), "base_panel_inches": base, "axes_box_inches": axes_box_inches,
            "figure_inches": [round(fig_w, 4), round(fig_h, 4)], "dpi": dpi, "font_pt": plots.FONT_PT,
            "annotation_language": "English", "png_count": 1,
            "colormap": CMAP_NAME, "vmin": 0.0, "vmax": vmax, "shared_colorbar": True,
            "bad_color": BAD_COLOR, "panel_titles": panel_titles,
        },
        "inputs": {
            "selection": {"path": str(SELECTION_PATH), "sha256": selection["sha256"],
                          "display_endpoints": display},
            "manifest": {"path": str(MANIFEST_PATH), "sha256": manifest["sha256"]},
            "reference": {"path": str(lib.REFERENCE_PATH), "sha256": ref_actual,
                          "gate_record": str(REFERENCE_OPEN_PATH),
                          "first_opened_utc": open_record.get("reference_first_opened_utc"),
                          "candidate_hashes_computed_before_reference_open": True},
            "mask_snapshot": {"path": str(lib.MASK_SNAPSHOT), "sha256": str(mask_report["snapshot_sha256"]),
                              "expected_sha256": lib.MASK_SNAPSHOT_SHA256,
                              "expected_in_049_config": lib.MASK_SNAPSHOT_SHA256_IN_049_CONFIG,
                              "loaded_by": "eval049_evaluate.load_real_masks"},
            "r2_per_chromosome_tsv": {"path": str(R2_PER_CHROMOSOME),
                                      "sha256": input_hashes_at_start["r2_per_chromosome_tsv"]},
            "candidate_npz": {
                fit_id: {"path": str(npz_paths[fit_id]), "sha256_recorded": npz_recorded[fit_id],
                         "sha256_actual": npz_actual[fit_id], "match": True} for fit_id in panel_ids[1:]},
            "hashes_unchanged_after_write": True,
        },
        "mask": {
            "chromosome": CHROMOSOME, "snapshot_key": "chr0",
            "positions": n_bins, "position_first_bp": int(positions[0]), "position_last_bp": int(positions[-1]),
            "total_non_diagonal_pairs": int(pair_i.size), "common_pairs": int(common.sum()),
            "common_valid_bins": int(valid_local.size),
            "non_common_bins": [int(value) for value in positions[np.setdiff1d(np.arange(n_bins), valid_local)]],
            "global_index_rule": "chromosome_offset + positions // 1Mb",
            "valid_bins_all_20_chromosomes": int(mask_report["totals"]["valid_bins"]),
            "aggregate_audit": mask_audit,
            "caption_policy": ("common upper pairs and their mirror only; diagonal 0 at common-valid bins is "
                               "display-only and excluded from every evaluation"),
        },
        "scale": {
            "policy": ("one whole-cell Rg per dataset on the shared all-20-chromosome common finite support; "
                       "both copies pooled; no per-chromosome or per-copy scaling; no alignment applied because "
                       "distances are rotation/translation invariant"),
            "support_loci": n_support, "support_beads": 2 * n_support,
            "support_equals_mask_valid_bins": True,
            "per_chromosome_support_loci": per_chromosome_support,
            "chr1_mask_positions_in_support": int(local_support.sum()),
            "rg_by_dataset": rg_by_dataset,
            "reference_open_record_loci": {"finite_loci": open_record.get("finite_loci"),
                                           "nonfinite_loci": open_record.get("nonfinite_loci")},
        },
        "matching": {
            "source": str(R2_PER_CHROMOSOME),
            "policy": ("frozen chr1 whole-chromosome intra-chromosome Pearson direct/swapped matching; "
                       "the same copy assignment is used for both rows (no per-copy independent matching)"),
            "per_dataset": {
                fit_id: {
                    "orientation": r2_table[fit_id]["orientation"],
                    "matched": r2_table[fit_id]["matched"], "contrast": r2_table[fit_id]["contrast"],
                    "geometry_tie": r2_table[fit_id]["geometry_tie"],
                    "row_mat_copy": assignment[fit_id]["row_mat"]["copy_label"],
                    "row_mat_r": assignment[fit_id]["row_mat"]["r"],
                    "row_pat_copy": assignment[fit_id]["row_pat"]["copy_label"],
                    "row_pat_r": assignment[fit_id]["row_pat"]["r"],
                    "rho": r2_table[fit_id]["rho"],
                } for fit_id in panel_ids[1:]
            },
            "reference": assignment["reference"],
        },
        "matrices": matrix_checks,
        "checks": {
            "candidate_sha_matches_manifest_and_selection": bool(candidate_ok),
            "reference_sha_matches_gate_record": bool(reference_ok),
            "mask_sha_matches_frozen_constant": bool(mask_ok),
            "matrices_symmetric_exact": bool(symmetric_ok),
            "grey_cell_pattern_identical_across_10_panels": bool(mask_pattern_ok),
            "shared_colour_scale_all_10_panels": {"vmin": 0.0, "vmax": vmax, "ok": bool(shared_scale_ok)},
            "all_20_chromosomes_used_for_rg": bool(all20_chr_ok),
            "annotation_vs_frozen_r2_table_max_abs_diff": max_annotation_diff,
            "recomputed_matrix_r_vs_frozen_r2_table_max_abs_diff": max_recomputed_diff,
            "annotated_r_table": annotation_check,
            "input_hashes_unchanged_after_write": True,
        },
        "scope_note": ("user-appended plotting-only step: no refit, no re-evaluation, no prep/gate/R2 rerun; the "
                       "reference is read with the pure parser (lib.load_3dg + three_dg_to_array) after only "
                       "reading the existing gates/reference_open.json record, so the writing open_reference "
                       "wrapper is never called; no change to frozen coordinates, manifest, selection, existing "
                       "reports or the two existing figures; this step writes exactly one PNG and this one "
                       "metrics JSON"),
    }
    lib.write_json(metrics_path, metrics)

    print("CHR1 MATRICES WRITTEN")
    print("  figure  : %s" % figure_path)
    print("  metrics : %s" % metrics_path)
    print("  mask    : chr1 positions=%d common_pairs=%d valid_bins=%d" % (n_bins, int(common.sum()),
                                                                          int(valid_local.size)))
    print("  scale   : support=%d loci / %d beads; Rg per dataset=%s" % (n_support, 2 * n_support,
          {key: round(value, 6) for key, value in rg_by_dataset.items()}))
    print("  colour  : %s vmin=0 vmax=%.6f" % (CMAP_NAME, vmax))
    print("  match   : " + "; ".join("%s %s->%s r=%.6f / %s r=%.6f" % (
        fit_id, r2_table[fit_id]["orientation"], assignment[fit_id]["row_mat"]["copy_label"],
        assignment[fit_id]["row_mat"]["r"], assignment[fit_id]["row_pat"]["copy_label"],
        assignment[fit_id]["row_pat"]["r"]) for fit_id in panel_ids[1:]))
    print("  checks  : annotation<=1e-12 (max %.3e), recomputed r<=1e-12 (max %.3e), symmetric, inputs unchanged"
          % (max_annotation_diff, max_recomputed_diff))
    return 0


if __name__ == "__main__":
    sys.exit(main())
