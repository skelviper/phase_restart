"""054 交付图（只画图，不拟合、不重算评价）：

图 1 plots/four_group_same_cross_spearman.png
    四组 same/cross signed Spearman 双色箱线图，顺序固定为
    Baseline / Extra 20->10->5->2->1 / 本链 40->20->10->5->2->1 Mb / 正确全链 200kb->1Mb；
    每盒叠加 20 条染色体的实际点；只读 eval/per_chromosome_spearman.tsv。

图 2 plots/chr1_distance_matrices.png
    chr1 距离矩阵 2 行 x 4 列，列序 = Reference / Extra / 本链 1Mb 端点 / 全链 200kb->1Mb 粗化；
    行 = matched to reference mat / pat（各列按该列 chr1 的 4 个 Spearman 值做一次整 chr swap）。
    * 共同位点矩阵：轴 = 冻结 mask 的 chr1 positions（193 个 1Mb bin，3-195 Mb），
      只在冻结 common pairs 上有值，其余灰色；有效对角 0；
    * 展示尺度：每个 dataset（每列）用该列两个 copy 在 common pairs 上合并的中位距离做
      一个尺度，画 raw distance / 该中位数；不逐 copy 归一化；
    * 面板顶部写该行展示 copy 的 Spearman R same / cross（来自同一组 4 个 chr1 rho，
      same = 与匹配 ref copy 的 rho，cross = 与另一 ref copy 的 rho）。

规范：coolwarm_r、3 英寸基础面板、300 DPI、统一 7 pt 文字、共享 colorbar、同 Mb 轴。
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
EVAL_DIR = RUN / "eval"
PLOTS = RUN / "plots"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from round_paths_054 import (BOX_GROUPS, CHAIN_1MB_NPZ, CHAIN_COARSE_1MB_NPZ, CHR1_INDEX,  # noqa: E402
                             COLORS, DPI, EVAL_BIN_BP, EXTRA_NPZ, FONT_PT, MASK_SNAPSHOT,
                             PANEL_IN, REFERENCE_PATH, REUSED_AGGREGATES)

TSV = EVAL_DIR / "per_chromosome_spearman.tsv"
MISSING_GREY = "#d9d9d9"
MATRIX_KEYS = ("extra_levels", "chain_1Mb", "chain_coarse_1Mb")
COLUMN_LABELS = {
    "reference": "Reference",
    "extra_levels": "Extra 20\u219210\u21925\u21922\u21921 Mb",
    "chain_1Mb": "Shared chain 40\u219220\u219210\u21925\u21922\u21921 Mb",
    "chain_coarse_1Mb": "Full chain 200 kb \u2192 1 Mb",
}
# 首行面板顶部标题：第四列写明完整链，避免读者分不清 500/200 kb 是否包含在内
COLUMN_TOP_TITLES = {
    "reference": "Reference",
    "extra_levels": "Extra 20\u219210\u21925\u21922\u21921 Mb",
    "chain_1Mb": "Shared chain 40\u219220\u219210\u21925\u21922\u21921 Mb",
    "chain_coarse_1Mb": "Full chain 40\u219220\u219210\u21925\u21922\u21921 Mb\u2192500\u2192200 kb\n"
                        "coarsened to 1 Mb",
}
COLUMN_FOOTNOTE = {
    "reference": "reference 1 Mb coordinates",
    "extra_levels": "1 Mb endpoint, 1902 FG",
    "chain_1Mb": "1 Mb endpoint, 2102 FG",
    "chain_coarse_1Mb": "200 kb endpoint arithmetic-mean coarsened to 1 Mb, 2402 FG",
}


# ---------------------------------------------------------------- 共同工具
def read_tsv() -> tuple[list[dict], dict[str, dict[str, list[float]]]]:
    header, rows = None, []
    with TSV.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if header is None:
                header = fields
                continue
            rows.append(dict(zip(header, fields)))
    data = {group["key"]: {"same": [], "cross": []} for group in BOX_GROUPS}
    for row in rows:
        for key in data:
            for metric in ("same", "cross"):
                raw = row.get("%s_%s" % (key, metric), "NA")
                data[key][metric].append(float("nan") if raw == "NA" else float(raw))
    return rows, data


def load_coordinates(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
    if coordinates.shape != expected_shape:
        raise RuntimeError("candidate %s shape %s != %s" % (path, coordinates.shape, expected_shape))
    return coordinates


def load_reference_chr1(path: Path, chromosome: str, bin_starts: np.ndarray) -> dict[str, np.ndarray]:
    """参考 3DG 的 chr1(mat)/chr1(pat) 坐标按 1Mb bin 起点 bp 对齐；缺失 bin 保持 NaN。"""
    wanted = {int(bp): index for index, bp in enumerate(bin_starts)}
    tracks = {"mat": np.full((len(bin_starts), 3), np.nan), "pat": np.full((len(bin_starts), 3), np.nan)}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5 or not fields[0].startswith(chromosome + "("):
                continue
            side = "mat" if fields[0].endswith("(mat)") else ("pat" if fields[0].endswith("(pat)") else None)
            if side is None:
                continue
            index = wanted.get(int(fields[1]))
            if index is None:
                continue
            point = np.asarray([float(item) for item in fields[2:5]], dtype=np.float64)
            if np.isfinite(point).all():
                tracks[side][index] = point
    return tracks


def pair_distances(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    delta = points[pair_i] - points[pair_j]
    return np.sqrt(np.sum(delta * delta, axis=1))


def masked_matrix(distances: np.ndarray, n_positions: int, pair_i: np.ndarray, pair_j: np.ndarray,
                  used: np.ndarray, diagonal_mask: np.ndarray) -> np.ndarray:
    """distances 已是 used（common）pair 上的值；只在 used 的 pair 上填值（对称），其余 NaN；有效对角 0。"""
    if distances.shape != pair_i[used].shape:
        raise RuntimeError("masked_matrix distance/pair length mismatch")
    matrix = np.full((n_positions, n_positions), np.nan, dtype=np.float64)
    flat = pair_i[used] * n_positions + pair_j[used]
    matrix.ravel()[flat] = distances
    matrix.ravel()[pair_j[used] * n_positions + pair_i[used]] = distances
    index = np.arange(n_positions)
    matrix[index[diagonal_mask], index[diagonal_mask]] = 0.0
    return matrix


def chr1_selection(rows: list[dict], key: str) -> dict:
    row = next(r for r in rows if int(r["chromosome_index"]) == CHR1_INDEX)
    orientation = row["%s_orientation" % key]
    if orientation == "direct":
        mat_copy, pat_copy = 0, 1
    elif orientation in ("swapped", "tie"):
        mat_copy, pat_copy = 1, 0
    else:
        raise RuntimeError("chr1 orientation undefined for %s" % key)
    rho = {name: row["%s_%s" % (key, name)] for name in ("A_mat", "A_pat", "B_mat", "B_pat")}
    rho = {name: (None if value == "NA" else float(value)) for name, value in rho.items()}
    return {"orientation": orientation, "mat_copy": mat_copy, "pat_copy": pat_copy,
            "rho": rho, "chromosome": row["chromosome"],
            "same": None if row["%s_same" % key] == "NA" else float(row["%s_same" % key]),
            "cross": None if row["%s_cross" % key] == "NA" else float(row["%s_cross" % key])}


def place_footer(fig, lines: list[str], x_in: float, y_start_in: float, dy_in: float) -> None:
    """在 figure 坐标里放多行页脚，并断言每行都没有超出图宽（避免被裁掉）。"""
    artists = []
    for index, line in enumerate(lines):
        artists.append(fig.text(x_in / fig.get_figwidth(), (y_start_in - dy_in * index) / fig.get_figheight(),
                                line, ha="left", va="bottom", fontsize=FONT_PT))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width = fig.bbox.width
    height = fig.bbox.height
    for artist, line in zip(artists, lines):
        box = artist.get_window_extent(renderer=renderer)
        if box.x1 > width - 1.0 or box.x0 < 1.0 or box.y0 < 0.5 or box.y1 > height - 0.5:
            raise RuntimeError("footer line does not fit the figure: %r (x1=%.1f / %.1f)" % (line[:60], box.x1, width))


# ---------------------------------------------------------------- 图 1
def figure_boxplot(data: dict) -> Path:
    plt.rcParams.update({"font.size": FONT_PT, "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT,
                         "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT,
                         "legend.fontsize": FONT_PT, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    labels = [group["tick"] for group in BOX_GROUPS]
    keys = [group["key"] for group in BOX_GROUPS]
    fig = plt.figure(figsize=(10.8, 3.6), dpi=DPI)
    ax = fig.add_axes([0.068, 0.30, 0.60, 0.545])
    positions_same = np.arange(len(keys)) - 0.18
    positions_cross = positions_same + 0.36
    rng = np.random.default_rng(20260916)
    for index, key in enumerate(keys):
        for offset, metric in ((positions_same[index], "same"), (positions_cross[index], "cross")):
            values = np.asarray(data[key][metric], dtype=np.float64)
            defined = values[np.isfinite(values)]
            if len(defined):
                box = ax.boxplot([defined], positions=[offset], widths=0.3, patch_artist=True,
                                 showfliers=False, medianprops={"color": "black", "linewidth": 0.8},
                                 whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6},
                                 boxprops={"linewidth": 0.6})
                for patch in box["boxes"]:
                    patch.set_facecolor(COLORS[metric])
                    patch.set_alpha(0.55)
            jitter = rng.uniform(-0.055, 0.055, size=len(values))
            ax.scatter(np.full(len(values), offset) + jitter, values, s=4.0, marker="o",
                       facecolors="none", edgecolors=COLORS[metric], linewidths=0.4, zorder=3)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman rho vs reference (1 Mb)")
    ax.set_title("Four 1 Mb endpoints, one cell, one chromosome per point (n=20); the shared chain starts "
                 "at 40 Mb and passes through 20/10/5/2 Mb", pad=6)
    ax.set_xlim(-0.5, len(keys) - 0.5)
    ax.set_ylim(-0.05, 1.0)
    ax.axhline(0.0, color="0.6", linewidth=0.5, linestyle="--", zorder=1)
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=4.5,
                          markerfacecolor=COLORS["same"], markeredgecolor="none", alpha=0.75,
                          label="same (matched copies)"),
               plt.Line2D([], [], marker="s", linestyle="none", markerfacecolor=COLORS["cross"],
                          markeredgecolor="none", alpha=0.75, label="cross (swapped copies)"),
               plt.Line2D([], [], marker="o", linestyle="none", markersize=3.0,
                          markerfacecolor="none", markeredgecolor="0.2",
                          label="one point = one chromosome (n=20)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              handletextpad=0.4, borderaxespad=0.0, labelspacing=0.5)
    place_footer(fig, [
        "Shared support: 157,529 pairs; unequal budgets; endpoints not converged.",
    ], x_in=0.068 * 10.8, y_start_in=0.26, dy_in=0.115)
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "four_group_same_cross_spearman.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    for key, label in zip(keys, labels):
        same = np.asarray(data[key]["same"], dtype=np.float64)
        cross = np.asarray(data[key]["cross"], dtype=np.float64)
        print("%-16s same n=%2d mean=%.6f | cross n=%2d mean=%.6f" % (
            key, int(np.isfinite(same).sum()),
            float(np.nanmean(same)) if np.isfinite(same).any() else float("nan"),
            int(np.isfinite(cross).sum()),
            float(np.nanmean(cross)) if np.isfinite(cross).any() else float("nan")))
    return out


# ---------------------------------------------------------------- 图 2
def figure_matrices(rows: list[dict]) -> Path:
    coarse_layer = np.load(REUSED_AGGREGATES[1_000_000][0], allow_pickle=False)
    chromosome_names = [str(x) for x in coarse_layer["chromosome_names"]]
    chromosome_lengths = [int(x) for x in coarse_layer["chromosome_lengths"]]
    n_bins_1mb = [(L + EVAL_BIN_BP - 1) // EVAL_BIN_BP for L in chromosome_lengths]
    total_1mb = int(sum(n_bins_1mb))
    offsets_1mb = np.concatenate(([0], np.cumsum(n_bins_1mb[:-1])))
    chromosome = chromosome_names[CHR1_INDEX]
    chr1_length = chromosome_lengths[CHR1_INDEX]

    with np.load(MASK_SNAPSHOT, allow_pickle=False) as archive:
        positions = np.asarray(archive["chr%d_positions" % CHR1_INDEX], dtype=np.int64)
        pair_i = np.asarray(archive["chr%d_pair_i" % CHR1_INDEX], dtype=np.int64)
        pair_j = np.asarray(archive["chr%d_pair_j" % CHR1_INDEX], dtype=np.int64)
        common = np.asarray(archive["chr%d_common" % CHR1_INDEX], dtype=bool)
    if not np.array_equal(positions, np.arange(3_000_000, chr1_length, EVAL_BIN_BP, dtype=np.int64)):
        raise RuntimeError("frozen chr1 positions changed")
    n_positions = int(positions.size)
    used_loci = np.zeros(n_positions, dtype=bool)
    used_loci[np.unique(np.concatenate((pair_i[common], pair_j[common])))] = True
    global_indices = int(offsets_1mb[CHR1_INDEX]) + positions // EVAL_BIN_BP
    if global_indices.max() >= total_1mb:
        raise RuntimeError("chr1 global index escaped the 1Mb grid")

    # —— 参考 chr1（缺失 bin 保持 NaN） ——
    reference = load_reference_chr1(REFERENCE_PATH, chromosome, positions)

    # —— 三个候选列 + 参考列的坐标 ——
    extra_1mb = load_coordinates(EXTRA_NPZ, (2, total_1mb, 3))
    chain_1mb = load_coordinates(CHAIN_1MB_NPZ, (2, total_1mb, 3))
    chain_coarse = load_coordinates(CHAIN_COARSE_1MB_NPZ, (2, total_1mb, 3))
    coords_by_key = {"extra_levels": extra_1mb, "chain_1Mb": chain_1mb, "chain_coarse_1Mb": chain_coarse}

    selections = {key: chr1_selection(rows, key) for key in MATRIX_KEYS}

    # —— 每列一个展示尺度：该列两个 copy 在 common pairs 上合并的 raw distance 中位数 ——
    scales: dict[str, dict] = {}
    matrices: dict[str, dict[int, np.ndarray]] = {}

    def column_scale(dist_parts: list[np.ndarray]) -> float:
        pooled = np.concatenate(dist_parts)
        value = float(np.median(pooled))
        if not (math.isfinite(value) and value > 0.0):
            raise RuntimeError("non-positive pooled median scale")
        return value

    ref_parts = []
    for side in ("mat", "pat"):
        # reference 数组已按 positions（冻结 mask 的 193 个 1Mb bin 起点）对齐，index 即 positions 下标
        points = reference[side]
        ref_parts.append(pair_distances(points, pair_i[common], pair_j[common]))
    scale_reference = column_scale(ref_parts)
    scales["reference"] = {"s_pooled_median_raw_distance": scale_reference,
                           "n_common_pairs_two_copies": int(2 * common.sum()),
                           "rule": "raw distance / pooled median over both copies on frozen common pairs",
                           "displayed_copies": ["reference mat", "reference pat"]}
    matrices["reference"] = {
        0: masked_matrix(ref_parts[0], n_positions, pair_i, pair_j, common, used_loci) / scale_reference,
        1: masked_matrix(ref_parts[1], n_positions, pair_i, pair_j, common, used_loci) / scale_reference,
    }

    for key in MATRIX_KEYS:
        coords = coords_by_key[key]
        parts = [pair_distances(coords[copy, global_indices], pair_i[common], pair_j[common])
                 for copy in (0, 1)]
        scale = column_scale(parts)
        selection = selections[key]
        scales[key] = {"s_pooled_median_raw_distance": scale,
                       "n_common_pairs_two_copies": int(2 * common.sum()),
                       "rule": "raw distance / pooled median over both copies on frozen common pairs",
                       "chr1_best_swap": selection["orientation"],
                       "displayed_copies": ["copyA" if selection["mat_copy"] == 0 else "copyB",
                                            "copyA" if selection["pat_copy"] == 0 else "copyB"],
                       "chr1_same": selection["same"], "chr1_cross": selection["cross"]}
        matrices[key] = {
            0: masked_matrix(parts[selection["mat_copy"]], n_positions, pair_i, pair_j, common, used_loci) / scale,
            1: masked_matrix(parts[selection["pat_copy"]], n_positions, pair_i, pair_j, common, used_loci) / scale,
        }

    # —— 面板顶部 Spearman 标注：该行展示的 copy 对两个 ref copy 的 rho ——
    annotations = []

    def panel_rho(key: str, row_index: int) -> tuple[float | None, float | None, str, str]:
        selection = selections[key]
        copy_index = selection["mat_copy"] if row_index == 0 else selection["pat_copy"]
        prefix = "A" if copy_index == 0 else "B"
        matched_ref = "mat" if row_index == 0 else "pat"
        other_ref = "pat" if row_index == 0 else "mat"
        same = selection["rho"]["%s_%s" % (prefix, matched_ref)]
        cross = selection["rho"]["%s_%s" % (prefix, other_ref)]
        return same, cross, "copyA" if copy_index == 0 else "copyB", matched_ref

    finite_values = np.concatenate([matrix[np.isfinite(matrix)].ravel()
                                    for key in matrices for matrix in matrices[key].values()])
    vmax = float(finite_values.max())

    plt.rcParams.update({"font.size": FONT_PT, "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT,
                         "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    gap_x, gap_y = 0.42, 0.86
    left, right_cbar, bottom, top = 0.78, 1.02, 2.00, 0.72
    width_in = left + 4 * PANEL_IN + 3 * gap_x + 0.34 + right_cbar
    height_in = top + 2 * PANEL_IN + gap_y + bottom
    fig = plt.figure(figsize=(width_in, height_in), dpi=DPI)
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad(MISSING_GREY)
    half_mb = EVAL_BIN_BP / 2e6
    extent = [(positions[0] / 1e6) - half_mb, (positions[-1] / 1e6) + half_mb]

    mesh = None
    for row_index in range(2):
        for col_index, key in enumerate(("reference",) + MATRIX_KEYS):
            x0 = (left + col_index * (PANEL_IN + gap_x)) / width_in
            y0 = (bottom + (1 - row_index) * (PANEL_IN + gap_y)) / height_in
            ax = fig.add_axes([x0, y0, PANEL_IN / width_in, PANEL_IN / height_in])
            mesh = ax.imshow(np.ma.masked_invalid(matrices[key][row_index]), origin="lower",
                             extent=[extent[0], extent[1], extent[0], extent[1]], interpolation="none",
                             aspect="equal", cmap=cmap, vmin=0.0, vmax=vmax)
            if key == "reference":
                title = "Reference 1 Mb\n%s" % ("mat" if row_index == 0 else "pat")
            else:
                same, cross, copy_name, matched_ref = panel_rho(key, row_index)
                if row_index == 0:
                    header = COLUMN_TOP_TITLES[key]
                else:
                    header = "%s (matched to reference pat)" % COLUMN_LABELS[key]
                title = "%s\nSpearman R same = %.3f | cross = %.3f" % (
                    header, float("nan") if same is None else same,
                    float("nan") if cross is None else cross)
                annotations.append({
                    "column_index": col_index + 1, "column": COLUMN_LABELS[key], "candidate_key": key,
                    "row_index": row_index + 1,
                    "row": "matched to reference %s" % matched_ref,
                    "displayed_copy": copy_name,
                    "matched_reference_copy": matched_ref,
                    "spearman_same_single_copy": same, "spearman_cross_single_copy": cross,
                })
            ax.set_title(title, pad=5, fontsize=FONT_PT)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[0], extent[1])
            ax.set_xticks([3, 50, 100, 150, 195])
            ax.set_yticks([3, 50, 100, 150, 195])
            ax.tick_params(pad=1.5)
            if col_index == 0:
                ax.set_ylabel("matched to reference %s\nchr1 position (Mb)"
                              % ("mat" if row_index == 0 else "pat"))
            else:
                ax.set_yticklabels([])
            if row_index == 1:
                ax.set_xlabel("chr1 position (Mb)")
            else:
                ax.set_xticklabels([])

    # 两 copy 平均（chr1）加到同一个小表里，不塞进图内避免拥挤
    for key in MATRIX_KEYS:
        selection = selections[key]
        for record in annotations:
            if record["candidate_key"] == key:
                record["chr1_two_copy_mean_same"] = selection["same"]
                record["chr1_two_copy_mean_cross"] = selection["cross"]
                record["chr1_best_swap"] = selection["orientation"]

    cbar_ax = fig.add_axes([(left + 4 * PANEL_IN + 3 * gap_x + 0.30) / width_in,
                            bottom / height_in, 0.22 / width_in,
                            (2 * PANEL_IN + gap_y) / height_in])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label("chr1 distance / pooled two-copy median\n(0 = same bin)", fontsize=FONT_PT)
    cbar.ax.tick_params(labelsize=FONT_PT, pad=1.5)

    fig.text(0.004, 1.0, "chr1 distance matrices: reference vs Extra, the shared-chain 1 Mb endpoint and its "
                         "200 kb \u2192 1 Mb coarsening", ha="left", va="top", fontsize=FONT_PT)
    footer_lines = [
        "Every column is 1 Mb-scale coordinates. Column 4 is the arithmetic-mean coarsening of the same chain's native "
        "200 kb endpoint; baseline is only in the boxplot.",
        "Panels are the frozen chr1 common-loci matrix (193 1-Mb bins, 3-195 Mb): only frozen common pairs are drawn, "
        "everything else is grey.",
        "Grey also marks reference bins without coordinates (NaN, no interpolation); diagonal = 0 where a bin has valid "
        "common pairs.",
        "Display scale: one pooled median over both copies' common-pair distances per column (raw distance / that median); "
        "no per-copy normalization; rescaling cannot change Spearman rho.",
        "Rows follow each column's chr1 best swap (whole-chromosome A/B choice by the maximum 4-rho pair sum under Spearman).",
        "The numbers above each candidate panel are that displayed copy's own rho: same = matched reference copy, "
        "cross = the other reference copy; all endpoints are budget_not_converged.",
    ]
    place_footer(fig, footer_lines, x_in=0.004 * width_in, y_start_in=1.15, dy_in=0.155)
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "chr1_distance_matrices.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    annotation_path = EVAL_DIR / "panel_annotations.tsv"
    columns = ["column_index", "column", "candidate_key", "row_index", "row", "displayed_copy",
               "matched_reference_copy", "spearman_same_single_copy", "spearman_cross_single_copy",
               "chr1_two_copy_mean_same", "chr1_two_copy_mean_cross", "chr1_best_swap"]
    with annotation_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for record in annotations:
            values = []
            for column in columns:
                value = record.get(column, "")
                if isinstance(value, float):
                    values.append("NA" if not math.isfinite(value) else "%.6f" % value)
                else:
                    values.append(str(value))
            handle.write("\t".join(values) + "\n")

    record = {
        "schema": "p9016-round054-chr1-matrix-scales-v1",
        "roles": "display / evaluation only; no fitting, no re-optimization, no new candidate",
        "chromosome": chromosome, "chr1_length_bp": chr1_length,
        "column_order": ["Reference", "Extra 20\u219210\u21925\u21922\u21921 Mb",
                         "Shared chain 40\u219220\u219210\u21925\u21922\u21921 Mb",
                         "Full chain 200 kb \u2192 1 Mb"],
        "row_order": ["matched to reference mat", "matched to reference pat"],
        "grid": {"source_mask": str(MASK_SNAPSHOT.relative_to(ROOT)), "n_positions": n_positions,
                 "positions_first_last": [int(positions[0]), int(positions[-1])],
                 "bin_bp": EVAL_BIN_BP, "n_common_pairs": int(common.sum()),
                 "n_pairs_total": int(pair_i.size),
                 "rule": "axis = frozen chr1 common-loci positions; only common pairs drawn; "
                         "other cells grey; diagonal 0 where a bin has valid common pairs"},
        "display_scales": scales,
        "color": {"cmap": "coolwarm_r", "vmin": 0.0, "vmax": vmax,
                  "note": "single shared colour scale over all eight panels; no per-panel scaling"},
        "panel_top_annotation": {
            "rule": "each candidate panel states Spearman R same / cross at the top for the copy shown in that row",
            "same_definition": "rho(displayed copy, its matched reference copy) read from the same chr1 four rhos",
            "cross_definition": "rho(displayed copy, the other reference copy) read from the same chr1 four rhos",
            "not_used": ["20-chromosome macro mean pasted onto chr1", "two-copy average presented as a single copy"],
            "two_copy_means_in": str(annotation_path.relative_to(ROOT)),
        },
        "panel_size_inches": PANEL_IN, "dpi": DPI, "font_pt": FONT_PT,
    }
    (EVAL_DIR / "chr1_matrix_scales.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print("vmax", vmax)
    for key, value in scales.items():
        print("  %-18s scale=%.4f" % (key, value["s_pooled_median_raw_distance"]))
    print("annotations ->", annotation_path)
    return out


def main() -> int:
    rows, data = read_tsv()
    print(figure_boxplot(data))
    print(figure_matrices(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
