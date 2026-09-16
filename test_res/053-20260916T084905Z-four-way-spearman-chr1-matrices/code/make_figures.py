"""053 交付图（只画图，不拟合、不重算评价）：

图 1 plots/four_way_same_cross_spearman.png
    四版本 same/cross signed Spearman 双色箱线图（Baseline / Extra levels /
    New chain 1 Mb / 200kb -> 1Mb 粗化）；每盒叠加 20 条染色体的实际点；只读
    eval/per_chromosome_spearman.tsv。

图 2 plots/chr1_distance_matrices.png
    chr1 距离矩阵 2 行 x 4 列：列 = Reference 1Mb / New chain 1Mb /
    New chain 200kb / 200kb->1Mb 粗化；行 = matched to reference mat / pat。
    * 候选面板用各自的原生完整 grid：1Mb 面板 196 个 bin、200kb 面板 978 个 bin，
      pcolormesh 用真实 bp bin edges（末端 edge 截到真实 chr1 长度），四列同一
      x/y 范围，不把 978 个 bin 硬拉成 196 个 bin 的像素边界；
    * 参考缺失坐标保持 NaN（灰色），不填 0、不外推；对角线仅有效点为 0；
    * 直接 1Mb 面板用该候选 chr1 最佳 swap；200kb 与其粗化 1Mb 共用粗化候选的
      chr1 最佳 swap，保证两列显示同一 copy；不给 200kb 虚构直接对 200kb 参考的 rho；
    * 参考无绝对单位校准：直接 1Mb 用两个 copy 合在一起、冻结 chr1 共同 pairs 上
      最小二乘 s=(d.r)/(d.d)；200kb 与其粗化 1Mb 共用由粗化距离求得的同一个 s；
      绘图用 s*d（乘），参考尺度 1；仅展示用缩放，不改变 Spearman，不做每 copy 或
      每格独立缩放。

规范：coolwarm_r、3 英寸基础面板、300 DPI、统一 7 pt 文字。
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
EVAL_DIR = RUN / "eval"
PLOTS = RUN / "plots"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from four_way_paths import (AGGREGATE_200KB, CANDIDATES, CHR1_INDEX, EVAL_BIN_BP,  # noqa: E402
                            MASK_SNAPSHOT, NEW_CHAIN_200KB_NPZ, REFERENCE_PATH, ROOT, VERSION_ORDER,
                            verify_candidate_hashes)

TSV = EVAL_DIR / "per_chromosome_spearman.tsv"
SCALES_JSON = EVAL_DIR / "chr1_matrix_scales.json"
COLORS = {"same": "#2b6cb0", "cross": "#dd6b20"}
MISSING_GREY = "#d9d9d9"
PANEL_IN = 3.0


# ---------------------------------------------------------------- 图 1
def read_tsv() -> tuple[list[dict], dict[str, dict[str, list[float]]]]:
    header, rows = None, []
    with TSV.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if header is None:
                header = fields
                continue
            rows.append(dict(zip(header, fields)))
    data = {name: {"same": [], "cross": []} for name in VERSION_ORDER}
    for row in rows:
        for name in VERSION_ORDER:
            key = CANDIDATES[name]["key"]
            for metric in ("same", "cross"):
                raw = row.get("%s_%s" % (key, metric), "NA")
                data[name][metric].append(float("nan") if raw == "NA" else float(raw))
    return rows, data


def figure_boxplot(data: dict) -> Path:
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig = plt.figure(figsize=(9.2, 3.5), dpi=300)
    ax = fig.add_axes([0.088, 0.315, 0.575, 0.575])
    positions_same = np.arange(len(VERSION_ORDER)) - 0.18
    positions_cross = positions_same + 0.36
    rng = np.random.default_rng(20260916)
    for index, name in enumerate(VERSION_ORDER):
        for offset, metric in ((positions_same[index], "same"), (positions_cross[index], "cross")):
            values = np.asarray(data[name][metric], dtype=np.float64)
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
    ax.set_xticks(np.arange(len(VERSION_ORDER)))
    ax.set_xticklabels([CANDIDATES[name]["tick"] for name in VERSION_ORDER])
    ax.set_ylabel("Spearman rho vs reference (1 Mb)")
    ax.set_title("Four candidate endpoints (all budget_not_converged); one cell, "
                 "20 chromosomes as the observation unit", pad=6)
    ax.set_xlim(-0.5, len(VERSION_ORDER) - 0.5)
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
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.035, 1.0), frameon=False,
              handletextpad=0.4, borderaxespad=0.0, labelspacing=0.5)
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "four_way_same_cross_spearman.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    for name in VERSION_ORDER:
        same = np.asarray(data[name]["same"], dtype=np.float64)
        cross = np.asarray(data[name]["cross"], dtype=np.float64)
        print("%-16s same n=%2d mean=%.4f | cross n=%2d mean=%.4f" % (
            name, int(np.isfinite(same).sum()),
            float(np.nanmean(same)) if np.isfinite(same).any() else float("nan"),
            int(np.isfinite(cross).sum()),
            float(np.nanmean(cross)) if np.isfinite(cross).any() else float("nan")))
    return out


# ---------------------------------------------------------------- 图 2
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


def distance_matrix(points: np.ndarray) -> np.ndarray:
    """完整 N x N 欧氏距离矩阵；坐标缺失的行/列保持 NaN，对角线仅有效点为 0。"""
    finite = np.isfinite(points).all(axis=1)
    matrix = np.full((len(points), len(points)), np.nan, dtype=np.float64)
    if finite.sum() >= 2:
        delta = points[finite][:, None, :] - points[finite][None, :, :]
        sub = np.sqrt(np.sum(delta * delta, axis=2))
        np.fill_diagonal(sub, 0.0)
        matrix[np.ix_(finite, finite)] = sub
    return matrix


def selection_from_tsv(rows: list[dict], key: str) -> dict:
    row = next(r for r in rows if int(r["chromosome_index"]) == CHR1_INDEX)
    orientation = row["%s_orientation" % key]
    if orientation == "direct":
        mat_copy, pat_copy = 0, 1
    elif orientation in ("swapped", "tie"):
        mat_copy, pat_copy = 1, 0
    else:
        raise RuntimeError("chr1 orientation undefined for %s" % key)
    return {"orientation": orientation, "mat_copy": mat_copy, "pat_copy": pat_copy,
            "chromosome": row["chromosome"],
            "same": None if row["%s_same" % key] == "NA" else float(row["%s_same" % key]),
            "cross": None if row["%s_cross" % key] == "NA" else float(row["%s_cross" % key])}


def pair_distances(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    delta = points[pair_i] - points[pair_j]
    return np.sqrt(np.sum(delta * delta, axis=1))


def least_squares_scale(points: np.ndarray, mat_copy: int, pat_copy: int,
                        ref_mat: np.ndarray, ref_pat: np.ndarray,
                        pair_i: np.ndarray, pair_j: np.ndarray) -> dict:
    """冻结 chr1 共同 pairs 上、两个 copy 合在一起的 display 尺度 s = (d.r)/(d.d)。"""
    d_parts, r_parts = [], []
    for copy, ref in ((mat_copy, ref_mat), (pat_copy, ref_pat)):
        finite = np.isfinite(ref).all(axis=1)
        d = pair_distances(points[copy], pair_i, pair_j)
        r = pair_distances(ref, pair_i, pair_j)
        keep = finite[pair_i] & finite[pair_j] & np.isfinite(d) & np.isfinite(r)
        d_parts.append(d[keep])
        r_parts.append(r[keep])
    d = np.concatenate(d_parts)
    r = np.concatenate(r_parts)
    if d.size == 0:
        raise RuntimeError("no frozen chr1 common pairs available for the scale estimate")
    scale = float(np.dot(d, r) / np.dot(d, d))
    if not (math.isfinite(scale) and scale > 0.0):
        raise RuntimeError("non-positive scale")
    residual = r - scale * d
    return {"s": scale, "n_pairs_pooled_two_copies": int(d.size),
            "sum_d_dot_r": float(np.dot(d, r)), "sum_d_dot_d": float(np.dot(d, d)),
            "pearson_r_of_pooled_d_r": float(np.corrcoef(d, r)[0, 1]),
            "residual_rms_reference_units": float(np.sqrt(np.mean(residual * residual)))}


def figure_matrices(rows: list[dict]) -> Path:
    # —— chr1 grid：1Mb（196 bin）与原生 200kb（978 bin），末端 edge 截到真实 chr1 长度 ——
    with np.load(AGGREGATE_200KB, allow_pickle=False) as fine:
        chromosome_names = [str(x) for x in fine["chromosome_names"]]
        chromosome_lengths = [int(x) for x in fine["chromosome_lengths"]]
        fine_n_bins = [int(x) for x in fine["n_bins"]]
        fine_offsets = [int(x) for x in fine["offsets"]]
    chromosome = chromosome_names[CHR1_INDEX]
    chr1_length = chromosome_lengths[CHR1_INDEX]
    n_1mb = (chr1_length + EVAL_BIN_BP - 1) // EVAL_BIN_BP
    edges_1mb = np.minimum(np.arange(n_1mb + 1, dtype=np.float64) * EVAL_BIN_BP, chr1_length)
    n_200kb = fine_n_bins[CHR1_INDEX]
    edges_200kb = np.minimum(np.arange(n_200kb + 1, dtype=np.float64) * 200_000, chr1_length)
    n_bins_1mb_all = [(L + EVAL_BIN_BP - 1) // EVAL_BIN_BP for L in chromosome_lengths]
    offsets_1mb = np.concatenate(([0], np.cumsum(n_bins_1mb_all[:-1])))
    if fine_offsets[CHR1_INDEX] != 0 or offsets_1mb[CHR1_INDEX] != 0:
        raise RuntimeError("chr1 is not the first chromosome in the native grids")

    # —— 参考 chr1：只用于评价/展示；缺失 bin 保持 NaN ——
    reference = load_reference_chr1(REFERENCE_PATH, chromosome, edges_1mb[:-1])
    ref_mat_matrix = distance_matrix(reference["mat"])
    ref_pat_matrix = distance_matrix(reference["pat"])

    # —— 候选坐标（完整原生 grid） ——
    new_1mb = load_coordinates(Path(CANDIDATES["New chain 1 Mb"]["npz"]), (2, int(sum(n_bins_1mb_all)), 3))
    coarse_1mb = load_coordinates(Path(CANDIDATES["200 kb -> 1 Mb"]["npz"]), new_1mb.shape)
    fine_200kb = load_coordinates(Path(NEW_CHAIN_200KB_NPZ), (2, int(sum(fine_n_bins)), 3))
    chr1_new_1mb = new_1mb[:, :n_1mb, :]
    chr1_coarse = coarse_1mb[:, :n_1mb, :]
    chr1_200kb = fine_200kb[:, :n_200kb, :]
    if chr1_new_1mb.shape[1] != 196 or chr1_200kb.shape[1] != 978 or chr1_coarse.shape[1] != 196:
        raise RuntimeError("chr1 grid mismatch: %s %s %s" % (chr1_new_1mb.shape, chr1_200kb.shape,
                                                             chr1_coarse.shape))

    # —— 最佳 swap（来自 053 自己的 chr1 Spearman 行） ——
    sel_1mb = selection_from_tsv(rows, CANDIDATES["New chain 1 Mb"]["key"])
    sel_coarse = selection_from_tsv(rows, CANDIDATES["200 kb -> 1 Mb"]["key"])

    # —— display 尺度：冻结 chr1 共同 pairs、两 copy 合并 ——
    with np.load(MASK_SNAPSHOT, allow_pickle=False) as archive:
        positions = np.asarray(archive["chr%d_positions" % CHR1_INDEX], dtype=np.int64)
        pair_i = np.asarray(archive["chr%d_pair_i" % CHR1_INDEX], dtype=np.int64)
        pair_j = np.asarray(archive["chr%d_pair_j" % CHR1_INDEX], dtype=np.int64)
        common = np.asarray(archive["chr%d_common" % CHR1_INDEX], dtype=bool)
        mask_pairs_total = int(pair_i.size)
    if not np.array_equal(positions, np.arange(3_000_000, chr1_length, EVAL_BIN_BP, dtype=np.int64)):
        raise RuntimeError("frozen chr1 positions changed")
    keep_pair = common
    pair_i_c, pair_j_c = pair_i[keep_pair], pair_j[keep_pair]
    ref_masked_mat = reference["mat"][positions // EVAL_BIN_BP]
    ref_masked_pat = reference["pat"][positions // EVAL_BIN_BP]
    scale_1mb = least_squares_scale(chr1_new_1mb, sel_1mb["mat_copy"], sel_1mb["pat_copy"],
                                    ref_masked_mat, ref_masked_pat, pair_i_c, pair_j_c)
    scale_coarse = least_squares_scale(chr1_coarse, sel_coarse["mat_copy"], sel_coarse["pat_copy"],
                                       ref_masked_mat, ref_masked_pat, pair_i_c, pair_j_c)

    # —— 面板矩阵（候选乘 s；参考尺度 1） ——
    panels = [
        {"title": "Reference 1 Mb\n(chr1 mat / pat, scale 1)",
         "edges": edges_1mb, "rows": [ref_mat_matrix, ref_pat_matrix], "note": "reference, units uncalibrated"},
        {"title": "New chain 1 Mb\n(40 \u2192 10 \u2192 5 \u2192 2 \u2192 1 Mb, 1902 FG)",
         "edges": edges_1mb,
         "rows": [distance_matrix(chr1_new_1mb[sel_1mb["mat_copy"]]) * scale_1mb["s"],
                  distance_matrix(chr1_new_1mb[sel_1mb["pat_copy"]]) * scale_1mb["s"]],
         "note": "s = %.3f (chr1 best swap: %s)" % (scale_1mb["s"], sel_1mb["orientation"])},
        {"title": "New chain 200 kb\n(final fine layer of the same chain, 2202 FG)",
         "edges": edges_200kb,
         "rows": [distance_matrix(chr1_200kb[sel_coarse["mat_copy"]]) * scale_coarse["s"],
                  distance_matrix(chr1_200kb[sel_coarse["pat_copy"]]) * scale_coarse["s"]],
         "note": "native 978 bins, s = %.3f shared with coarsened 1Mb" % scale_coarse["s"]},
        {"title": "200 kb \u2192 1 Mb\n(arithmetic-mean coarsening of column 3)",
         "edges": edges_1mb,
         "rows": [distance_matrix(chr1_coarse[sel_coarse["mat_copy"]]) * scale_coarse["s"],
                  distance_matrix(chr1_coarse[sel_coarse["pat_copy"]]) * scale_coarse["s"]],
         "note": "same copy as column 3, s = %.3f (chr1 best swap: %s)"
                 % (scale_coarse["s"], sel_coarse["orientation"])},
    ]

    finite_values = np.concatenate([m[np.isfinite(m)].ravel() for p in panels for m in p["rows"]])
    vmax = float(finite_values.max())

    # —— 版式：3 英寸基础面板、2 行 x 4 列，色条统一 ——
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    gap_x, gap_y = 0.34, 0.72
    left, right_cbar, bottom, top = 0.72, 0.95, 1.48, 0.62
    width_in = left + 4 * PANEL_IN + 3 * gap_x + 0.30 + right_cbar
    height_in = top + 2 * PANEL_IN + gap_y + bottom
    fig = plt.figure(figsize=(width_in, height_in), dpi=300)
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad(MISSING_GREY)

    panel_axes = []
    for row_index in range(2):
        for col_index in range(4):
            x0 = (left + col_index * (PANEL_IN + gap_x)) / width_in
            y0 = (bottom + (1 - row_index) * (PANEL_IN + gap_y)) / height_in
            ax = fig.add_axes([x0, y0, PANEL_IN / width_in, PANEL_IN / height_in])
            panel = panels[col_index]
            matrix = np.ma.masked_invalid(panel["rows"][row_index])
            mesh = ax.pcolormesh(panel["edges"] / 1e6, panel["edges"] / 1e6, matrix, cmap=cmap,
                                 vmin=0.0, vmax=vmax, shading="flat", rasterized=True)
            ax.set_xlim(0.0, chr1_length / 1e6)
            ax.set_ylim(0.0, chr1_length / 1e6)
            ax.set_aspect("equal")
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_yticks([0, 50, 100, 150, 200])
            ax.tick_params(pad=1.5)
            if row_index == 0:
                ax.set_title(panel["title"], pad=5)
            if col_index == 0:
                ax.set_ylabel("matched to reference %s\nchr1 position (Mb)" % ("mat" if row_index == 0 else "pat"))
            else:
                ax.set_yticklabels([])
            if row_index == 1:
                ax.set_xlabel("chr1 position (Mb)")
            else:
                ax.set_xticklabels([])
            if row_index == 1:
                ax.text(0.0, -0.235, panel["note"], transform=ax.transAxes, fontsize=7,
                        ha="left", va="top")
            panel_axes.append(ax)

    cbar_ax = fig.add_axes([(left + 4 * PANEL_IN + 3 * gap_x + 0.30) / width_in,
                            bottom / height_in, 0.20 / width_in,
                            (2 * PANEL_IN + gap_y) / height_in])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label("chr1 distance (reference units)\ncandidates display-rescaled by s", fontsize=7)
    cbar.ax.tick_params(labelsize=7, pad=1.5)

    fig.text(0.004, 1.0, "chr1 distance matrices: reference vs the 40 \u2192 10 \u2192 5 \u2192 2 \u2192 1 Mb chain "
                         "endpoints (full native grids)", ha="left", va="top", fontsize=7)
    footer_lines = [
        "Grey = reference coordinate missing (NaN, no interpolation / no extrapolation); diagonal = 0 for valid bins. "
        "Reference has no calibrated absolute units; candidates are display-rescaled by a single positive least-squares scale per column.",
        "Candidate panels use their full native grids with real bp bin edges (200 kb panel: 978 bins, not stretched to 196); "
        "all four columns share the same chr1 bp range.",
        "Column 3 (native 200 kb) and column 4 (its 1 Mb coarsening) show the same copy and share one scale; the 200 kb layer is never scored "
        "directly against the 1 Mb reference. Display rescaling does not change Spearman rho. All shown endpoints are budget_not_converged.",
    ]
    for line_index, line in enumerate(footer_lines):
        fig.text(0.004, (0.36 - 0.12 * line_index) / height_in, line, ha="left", va="bottom", fontsize=7)
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "chr1_distance_matrices.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)

    record = {
        "schema": "p9016-round053-chr1-matrix-scales-v1",
        "roles": "display / evaluation only; no fitting, no re-optimization, no new candidate",
        "chromosome": chromosome,
        "chr1_length_bp": chr1_length,
        "column_order": ["Reference 1 Mb", "New chain 1 Mb", "New chain 200 kb", "200 kb -> 1 Mb"],
        "row_order": ["matched to reference mat", "matched to reference pat"],
        "grids": {
            "1Mb": {"n_bins": n_1mb, "bin_bp": EVAL_BIN_BP,
                    "edge_rule": "edges = min(arange(n+1)*1e6, chr1_length)",
                    "last_bin_bp": int(edges_1mb[-1] - edges_1mb[-2])},
            "200kb": {"n_bins": n_200kb, "bin_bp": 200_000,
                      "edge_rule": "edges = min(arange(n+1)*2e5, chr1_length)",
                      "last_bin_bp": int(edges_200kb[-1] - edges_200kb[-2]),
                      "native_endpoint": str(Path(NEW_CHAIN_200KB_NPZ).relative_to(ROOT))},
        },
        "reference": {"path": str(REFERENCE_PATH.relative_to(ROOT)),
                      "missing_bins_are_nan": True,
                      "n_bins_with_coordinates_mat": int(np.isfinite(reference["mat"]).all(axis=1).sum()),
                      "n_bins_with_coordinates_pat": int(np.isfinite(reference["pat"]).all(axis=1).sum())},
        "mapping": {
            "New chain 1 Mb": {"chr1_best_swap": sel_1mb["orientation"], "mat_copy_index": sel_1mb["mat_copy"],
                               "pat_copy_index": sel_1mb["pat_copy"], "chr1_same": sel_1mb["same"],
                               "chr1_cross": sel_1mb["cross"]},
            "New chain 200 kb": {"chr1_best_swap": sel_coarse["orientation"],
                                 "mat_copy_index": sel_coarse["mat_copy"],
                                 "pat_copy_index": sel_coarse["pat_copy"],
                                 "shared_with": "200 kb -> 1 Mb",
                                 "chr1_same": sel_coarse["same"], "chr1_cross": sel_coarse["cross"]},
            "200 kb -> 1 Mb": {"chr1_best_swap": sel_coarse["orientation"],
                               "mat_copy_index": sel_coarse["mat_copy"],
                               "pat_copy_index": sel_coarse["pat_copy"],
                               "shared_with": "New chain 200 kb",
                               "chr1_same": sel_coarse["same"], "chr1_cross": sel_coarse["cross"]},
        },
        "display_scales": {
            "rule": "s = (d . r) / (d . d) pooling both copies on the frozen chr1 common pairs; "
                    "plotted value = s * d; reference scale = 1",
            "frozen_support": {"source": str(MASK_SNAPSHOT.relative_to(ROOT)),
                               "positions": str(positions[:1]) + "..." + str(positions[-1:]),
                               "n_positions": int(positions.size),
                               "n_pairs_total": mask_pairs_total,
                               "n_pairs_common_used": int(keep_pair.sum())},
            "New chain 1 Mb": scale_1mb,
            "200 kb -> 1 Mb": scale_coarse,
            "New chain 200 kb": {"s": scale_coarse["s"], "shared_with": "200 kb -> 1 Mb"},
            "reference": {"s": 1.0},
        },
        "color": {"cmap": "coolwarm_r", "vmin": 0.0, "vmax": vmax,
                  "note": "single shared colour scale over all eight panels; no per-panel scaling"},
        "no_new_spearman_for_200kb": "the 200 kb layer has no 200 kb reference; it is never scored directly "
                                     "against the 1 Mb reference",
        "panel_size_inches": PANEL_IN, "dpi": 300, "font_pt": 7,
    }
    SCALES_JSON.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print("vmax", vmax, "scale_1mb", scale_1mb["s"], "scale_coarse", scale_coarse["s"])
    print("chr1 swap 1Mb:", sel_1mb["orientation"], "| coarsened:", sel_coarse["orientation"])
    for panel in panels:
        for row_index, matrix in enumerate(panel["rows"]):
            finite = matrix[np.isfinite(matrix)]
            print("  %-28s row=%d shape=%s finite_frac=%.3f max=%.3f" % (
                panel["title"].splitlines()[0], row_index, matrix.shape,
                finite.size / matrix.size, finite.max() if finite.size else float("nan")))
    return out


def main() -> int:
    verify_candidate_hashes(EVAL_DIR / "candidate_hashes.json")
    rows, data = read_tsv()
    print(figure_boxplot(data))
    print(figure_matrices(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
