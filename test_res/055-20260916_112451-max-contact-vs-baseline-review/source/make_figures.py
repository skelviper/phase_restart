"""055 轮两张交付图（只读 eval/ 产物，不再打开 reference）。

图 1 plots/same_cross_contrast_boxplot.png：3 面板（same / cross / contrast）x 4 组，每组 20 条染色体实测点。
图 2 plots/chr1_distance_matrices.png：chr1 距离矩阵 4 行（Reference / Baseline / B / C）x 2 列
（映射到 reference mat / pat 的拷贝），距离按该数据全细胞 Rg 归一，统一色标、缺失灰、对角 0。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from review_paths import CHROMOSOMES, EVAL, GROUP_ORDER, JITTER_SEED, PLOTS, RUN  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", str(RUN / "scratch" / "mplcache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
})

SHORT_LABELS = {
    "Reference": "Reference\n(self control)",
    "Baseline-046-G-random": "Baseline 046\nG-random",
    "B-hard-observed": "B hard_observed\n049 raw/cons",
    "C-max-rate": "C max_rate\n049 ms/rand",
}
FULL_LABELS = {
    "Reference": "Reference (self control)",
    "Baseline-046-G-random": "Baseline 046 G-random (1502 FG)",
    "B-hard-observed": "B hard_observed, 049 raw/consensus (1502 FG)",
    "C-max-rate": "C max_rate, 049 ms/random (1502 FG)",
}
COLORS = {
    "Reference": "#6a51a3",
    "Baseline-046-G-random": "#7f7f7f",
    "B-hard-observed": "#e6550d",
    "C-max-rate": "#3182bd",
}
PANEL_KEYS = (("same", "same  =  max(direct, swapped)"),
              ("cross", "cross  =  min(direct, swapped)"),
              ("contrast", "contrast  =  same - cross"))


def figure_boxplot() -> Path:
    payload = json.loads((EVAL / "boxplot_data.json").read_text())
    pearson = payload["pearson"]
    rng = np.random.default_rng(JITTER_SEED)
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.5), constrained_layout=True)
    positions = np.arange(1, len(GROUP_ORDER) + 1, dtype=float)
    for axis, (key, title) in zip(axes, PANEL_KEYS):
        data = [pearson[name][key] for name in GROUP_ORDER]
        box = axis.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                           showfliers=False, medianprops={"color": "black", "linewidth": 0.9},
                           whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7},
                           boxprops={"linewidth": 0.7})
        for patch, name in zip(box["boxes"], GROUP_ORDER):
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(0.35)
            patch.set_edgecolor(COLORS[name])
        for index, name in enumerate(GROUP_ORDER):
            values = np.asarray(data[index], dtype=float)
            jitter = rng.uniform(-0.09, 0.09, size=values.size)
            axis.scatter(np.full(values.size, positions[index]) + jitter, values, s=6,
                         color=COLORS[name], edgecolor="white", linewidth=0.2, zorder=3)
        axis.axhline(0.0, color="black", linewidth=0.5, linestyle="--", zorder=1)
        axis.set_title(title)
        axis.set_xticks(positions)
        axis.set_xticklabels([SHORT_LABELS[name] for name in GROUP_ORDER], rotation=20, ha="right")
        axis.set_ylim(-0.05, 1.08)
        axis.set_ylabel("signed Pearson r" if key != "contrast" else "difference in Pearson r")
        axis.grid(axis="y", linewidth=0.3, alpha=0.4)
        axis.set_axisbelow(True)
    fig.suptitle("Per-chromosome copy-to-reference distance correlation, 20 chromosomes, "
                 "frozen legacy mask (157,529 pairs)", fontsize=7)
    output = PLOTS / "same_cross_contrast_boxplot.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def panel_annotation(name: str, column_index: int, row: dict[str, str]) -> str:
    """单 panel 标题：写清是哪一个候选 copy 对哪一条 reference 单倍型，并给出该 panel 自己的 Pearson r。"""
    if name == "Reference":
        return ("Reference mat\nr = 1.000 (self)" if column_index == 0
                else "Reference pat\nr = 1.000 (self)")
    orientation = row["orientation"]
    if orientation == "swapped":
        copy_label, rho_key = ("copy A", "A_pat") if column_index == 1 else ("copy B", "B_mat")
    else:
        copy_label, rho_key = ("copy B", "B_pat") if column_index == 1 else ("copy A", "A_mat")
    arrow = "-> ref mat" if column_index == 0 else "-> ref pat"
    suffix = "" if orientation != "unresolved_tie" else " (tie)"
    return "%s %s%s\nr = %.3f" % (copy_label, arrow, suffix, float(row[rho_key]))


def figure_chr1() -> Path:
    payload = np.load(EVAL / "chr1_panels.npz", allow_pickle=True)
    bins = payload["bins"].astype(float)
    panels = {
        "Reference": payload["reference"],
        "Baseline-046-G-random": payload["baseline"],
        "B-hard-observed": payload["b_hard_observed"],
        "C-max-rate": payload["c_max_rate"],
    }
    vmax = float(payload["unified_vmax"][0])
    panel_scale = payload["panel_scale_raw_median"]
    per_chr = {}
    with open(EVAL / "per_chromosome.tsv", "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            fields = dict(zip(header, line.rstrip("\n").split("\t")))
            if fields["chr"] == "chr1":
                per_chr[fields["dataset"]] = fields

    edges = np.concatenate(([bins[0] - 500_000.0], bins + 500_000.0)) / 1e6
    extent = [float(edges[0]), float(edges[-1]), float(edges[0]), float(edges[-1])]
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad("#d9d9d9")

    fig = plt.figure(figsize=(7.6, 11.6), constrained_layout=True)
    grid = fig.add_gridspec(len(GROUP_ORDER), 4, width_ratios=[0.16, 1.0, 1.0, 0.055])
    for row_index, name in enumerate(GROUP_ORDER):
        row = per_chr[name]
        label_axis = fig.add_subplot(grid[row_index, 0])
        label_axis.axis("off")
        label_axis.text(0.5, 0.5,
                        "%s\nscale %.3f\nsame %.3f | cross %.3f" % (
                            FULL_LABELS[name], float(panel_scale[GROUP_ORDER.index(name)]),
                            float(row["same"]), float(row["cross"])),
                        rotation=90, ha="center", va="center", fontsize=6)
        for column_index in range(2):
            axis = fig.add_subplot(grid[row_index, column_index + 1])
            matrix = panels[name][column_index]
            mesh = axis.imshow(matrix, origin="lower", extent=extent, cmap=cmap,
                               vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal")
            missing_bins = int(np.isnan(matrix).all(axis=1).sum())
            axis.set_title(panel_annotation(name, column_index, row), fontsize=6)
            axis.set_ylabel("chr1 (Mb)", fontsize=6)
            if row_index == len(GROUP_ORDER) - 1:
                axis.set_xlabel("chr1 (Mb)", fontsize=6)
            axis.tick_params(labelsize=5)
            if missing_bins:
                axis.text(0.02, 0.02, "Missing bins: %d" % missing_bins, transform=axis.transAxes,
                          fontsize=5, color="#404040")
    color_axis = fig.add_subplot(grid[:, 3])
    fig.colorbar(mesh, cax=color_axis, label="distance / raw median scale")
    fig.suptitle("chr1 1Mb distance matrices, one raw median scale per dataset\n"
                 "shared bins, one unified scale (0 to %.2f), grey = missing, diagonal = 0\n"
                 "row label: whole-chromosome same / cross (mean over the two copies)" % vmax,
                 fontsize=7)
    output = PLOTS / "chr1_distance_matrices.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    first = figure_boxplot()
    second = figure_chr1()
    print("wrote", first.relative_to(RUN))
    print("wrote", second.relative_to(RUN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
