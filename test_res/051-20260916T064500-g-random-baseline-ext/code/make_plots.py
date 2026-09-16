"""051 两张交付图（v2，按主侧审阅意见修正）。

图 1：chr1 distance matrix，2 行 x 5 列。
  列 = Reference / baseline / extra-levels / no-bend / reference-beads，
  行 = matched-to-mat / matched-to-pat。每列按该条件在 chr1 上的最佳 whole-chr mapping
  决定 A/B 与 mat/pat 的对应（mapping=swapped 时该列的 copyA 显示对 pat），因此矩阵数值
  与 per_chromosome.tsv 的 same/cross 计算完全一致。
  公共支持 = 046 冻结 legacy 21-mask；缺失 pair 为 NaN -> 灰色；轴按真实 bp（不压缩缺口）；
  对角线设 0；全图统一色标 [0, max]。
  展示尺度：每个 condition 用**同一个最优正尺度**同时作用在两 copy 上（joint least-squares
  到 chr1 参考距离），参考尺度固定 1；线性缩放不影响 Pearson。

图 2：same/cross signed Pearson 箱线图（x = 4 条件，每组 same/cross 两箱，叠 20 chr 点）。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
STYLE = {"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
         "ytick.labelsize": 6, "legend.fontsize": 6}
CONDITIONS = ("baseline", "extra-levels", "no-bend", "reference-beads")
LABELS = {"baseline": "baseline", "extra-levels": "extra-levels",
          "no-bend": "no-bend", "reference-beads": "reference-beads"}
COLORS = {"same": "#2b6cb0", "cross": "#c05621"}


def _matrix(values, positions, pair_i, pair_j):
    n = len(positions)
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    matrix[pair_i, pair_j] = values
    matrix[pair_j, pair_i] = values
    np.fill_diagonal(matrix, 0.0)
    return matrix


def figure_matrix(evaluation: Path, out_path: Path) -> None:
    payload = np.load(evaluation / "chr1_matrix_input.npz", allow_pickle=False)
    meta = json.loads((evaluation / "chr1_matrix_meta.json").read_text(encoding="utf-8"))
    positions = np.asarray(payload["baseline_positions"], dtype=np.float64)
    pair_i = np.asarray(payload["baseline_pair_i"], dtype=np.int64)
    pair_j = np.asarray(payload["baseline_pair_j"], dtype=np.int64)
    ref_mat = np.asarray(payload["baseline_reference_mat"], dtype=np.float64)
    ref_pat = np.asarray(payload["baseline_reference_pat"], dtype=np.float64)

    columns = [("Reference", {"mat": _matrix(ref_mat, positions, pair_i, pair_j),
                              "pat": _matrix(ref_pat, positions, pair_i, pair_j)},
                {"mat": "reference mat", "pat": "reference pat"})]
    for condition in CONDITIONS:
        cand_a = np.asarray(payload["%s_candidate_copyA" % condition], dtype=np.float64)
        cand_b = np.asarray(payload["%s_candidate_copyB" % condition], dtype=np.float64)
        mapping = str(meta["chr1_mapping"][condition])
        rho = meta["chr1_rho"][condition]
        candidate = np.concatenate([cand_a, cand_b])
        reference = np.concatenate([ref_mat, ref_pat])
        scale = float(np.dot(candidate, reference) / np.dot(candidate, candidate))
        to_mat, to_pat = (cand_a, cand_b) if mapping != "swapped" else (cand_b, cand_a)
        rho_key = ("A_mat", "B_pat") if mapping != "swapped" else ("A_pat", "B_mat")
        columns.append((LABELS[condition],
                        {"mat": _matrix(to_mat * scale, positions, pair_i, pair_j),
                         "pat": _matrix(to_pat * scale, positions, pair_i, pair_j)},
                        {"mat": "matched-to-mat rho=%.4f" % rho[rho_key[0]],
                         "pat": "matched-to-pat rho=%.4f" % rho[rho_key[1]],
                         "scale": scale, "mapping": mapping}))

    grid_mb = positions / 1e6
    bin_mb = float(np.median(np.diff(grid_mb))) / 2.0 if len(grid_mb) > 1 else 0.5
    extent = [float(grid_mb[0] - bin_mb), float(grid_mb[-1] + bin_mb),
              float(grid_mb[0] - bin_mb), float(grid_mb[-1] + bin_mb)]
    panels = [column[1][key] for column in columns for key in ("mat", "pat")]
    visible = np.concatenate([panel[np.isfinite(panel)].ravel() for panel in panels])
    color_limits = (0.0, float(visible.max()))
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad("#bdbdbd")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 5, figsize=(15.0, 6.0), dpi=300, squeeze=False)
        for column_index, (title, matrices, info) in enumerate(columns):
            for row_index, key in enumerate(("mat", "pat")):
                ax = axes[row_index][column_index]
                image = ax.imshow(np.ma.masked_invalid(matrices[key]), origin="lower",
                                  extent=extent, cmap=cmap, vmin=color_limits[0],
                                  vmax=color_limits[1], interpolation="nearest", aspect="equal")
                if row_index == 0:
                    ax.set_title("%s\n%s" % (title, info[key]), pad=2.0)
                else:
                    ax.set_title(info[key], pad=2.0)
                if row_index == 1:
                    ax.set_xlabel("chr1 position (Mb)")
                if column_index == 0:
                    ax.set_ylabel("chr1 position (Mb)")
                ax.set_xticks([grid_mb[0], grid_mb[len(grid_mb) // 2], grid_mb[-1]])
                ax.set_yticks([grid_mb[0], grid_mb[len(grid_mb) // 2], grid_mb[-1]])
                ax.tick_params(length=2, pad=1)
        colorbar = fig.colorbar(image, ax=axes, fraction=0.016, pad=0.012)
        colorbar.set_label("distance (per-condition joint display scale; gray = missing pair)")
        scales_text = ", ".join("%s=%.4g" % (LABELS[name], info["scale"])
                                for name, _m, info in columns[1:])
        fig.suptitle("chr1 distance matrices on the frozen 21-mask public support; columns follow "
                     "each condition's best whole-chromosome mapping\njoint display scales: %s "
                     "(linear rescaling does not change Pearson)" % scales_text, y=1.0)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    print("wrote", out_path)


def figure_boxplot(summary_path: Path, out_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))["summary"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=300)
        width = 0.34
        for index, condition in enumerate(CONDITIONS):
            for offset, key in ((-width / 2.0, "same"), (width / 2.0, "cross")):
                values = np.asarray([value for value in summary[condition][key]["values"]
                                     if value is not None], dtype=np.float64)
                if values.size == 0:
                    continue
                position = index + offset
                box = ax.boxplot([values], positions=[position], widths=width * 0.9,
                                 patch_artist=True, showfliers=False, manage_ticks=False,
                                 medianprops={"color": "black", "linewidth": 0.8},
                                 boxprops={"linewidth": 0.6},
                                 whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6})
                for patch in box["boxes"]:
                    patch.set_facecolor(COLORS[key])
                    patch.set_alpha(0.55)
                jitter = (np.arange(values.size) % 7 - 3) * 0.012
                ax.scatter(np.full(values.size, position) + jitter, values, s=5.0,
                           color=COLORS[key], edgecolors="none", zorder=3)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([LABELS[name] for name in CONDITIONS], rotation=12, ha="right")
        ax.set_ylabel("signed Pearson (matched / swapped)")
        ax.set_title("Same/cross copy consistency on the frozen 21-mask support\n"
                     "(20 chromosomes of one cell; not biological replicates)")
        undefined = {name: (summary[name]["same"]["n_undefined_chr"]
                            + summary[name]["cross"]["n_undefined_chr"]) for name in CONDITIONS}
        if any(value for value in undefined.values()):
            ax.set_xlabel("undefined chromosome counts: %s" % json.dumps(undefined))
        handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=5,
                              color=COLORS["same"], alpha=0.55, label="same (matched)"),
                   plt.Line2D([], [], marker="s", linestyle="none", markersize=5,
                              color=COLORS["cross"], alpha=0.55, label="cross (swapped)")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=2, frameon=False)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
    print("wrote", out_path)


if __name__ == "__main__":
    evaluation = RUN / "evaluation"
    plots = RUN / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    figure_matrix(evaluation, plots / "chr1_distance_matrices.png")
    figure_boxplot(evaluation / "pearson_summary.json", plots / "same_cross_pearson_boxplot.png")
