"""052 交付图：三版本 same/cross signed Spearman 双色箱线图（唯一 PNG 交付）。

规范：x = Baseline / Extra levels / 200 kb -> 1 Mb；same 蓝、cross 橙；
每个箱叠加 20 条染色体的实际点；统一 y 轴；图内英文；7 pt；300 DPI；3 英寸基础面板；
图例放在坐标区外，避免压住数据。只读 eval/per_chromosome_spearman.tsv。
"""
from __future__ import annotations

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
TSV = EVAL_DIR / "per_chromosome_spearman.tsv"

LABELS = ["Baseline", "Extra levels", "200 kb \u2192 1 Mb"]
KEYS = ["baseline", "extra_levels", "new_200kb_to_1Mb"]
COLORS = {"same": "#2b6cb0", "cross": "#dd6b20"}


def read_tsv() -> dict[str, dict[str, list[float]]]:
    header = None
    rows = []
    with TSV.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if header is None:
                header = fields
                continue
            rows.append(dict(zip(header, fields)))
    data = {key: {"same": [], "cross": []} for key in KEYS}
    for row in rows:
        for key in KEYS:
            for metric in ("same", "cross"):
                raw = row.get("%s_%s" % (key, metric), "NA")
                data[key][metric].append(float("nan") if raw == "NA" else float(raw))
    return data


def main() -> int:
    data = read_tsv()
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    # 3 英寸基础面板；图例放在坐标区外，用固定边距而不是 bbox_inches="tight"，
    # 避免 tight 布局把面板压扁、把刻度标签挤在一起。
    fig = plt.figure(figsize=(6.6, 3.0), dpi=300)
    ax = fig.add_axes([0.128, 0.155, 0.455, 0.815])
    positions_same = np.arange(len(LABELS)) * 1.0 - 0.18
    positions_cross = positions_same + 0.36
    rng = np.random.default_rng(20260916)
    for index, key in enumerate(KEYS):
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
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Spearman rho vs reference (1 Mb coarse)")
    ax.set_xlim(-0.5, len(LABELS) - 0.5)
    ax.axhline(0.0, color="0.6", linewidth=0.5, linestyle="--", zorder=1)
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=4.5,
                          markerfacecolor=COLORS["same"], markeredgecolor="none", alpha=0.75,
                          label="same (matched copies)"),
               plt.Line2D([], [], marker="s", linestyle="none", markersize=4.5,
                          markerfacecolor=COLORS["cross"], markeredgecolor="none", alpha=0.75,
                          label="cross (swapped copies)"),
               plt.Line2D([], [], marker="o", linestyle="none", markersize=3.0,
                          markerfacecolor="none", markeredgecolor="0.2",
                          label="one point = one chromosome (n=20)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False,
              handletextpad=0.4, borderaxespad=0.0, labelspacing=0.5)
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "same_cross_spearman_boxplot.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(out)
    for key, label in zip(KEYS, LABELS):
        same = np.asarray(data[key]["same"], dtype=np.float64)
        cross = np.asarray(data[key]["cross"], dtype=np.float64)
        print("%-14s same n=%d mean=%.4f | cross n=%d mean=%.4f" % (
            label, int(np.isfinite(same).sum()), np.nanmean(same) if np.isfinite(same).any() else float("nan"),
            int(np.isfinite(cross).sum()), np.nanmean(cross) if np.isfinite(cross).any() else float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
