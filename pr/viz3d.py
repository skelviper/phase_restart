"""染色体空间分区诊断与 legacy 对齐辅助函数。

正式报告图不会把候选结构与 reference 做 similarity 对齐。每个显示结构都独立居中并缩放到单位 RMS 半径，因此不能据各 panel 的视觉接近程度声称它们共享核坐标系。下面较早的 Procrustes 辅助函数仍为 exploratory code 保留兼容性，但正式图入口不会调用它们。
"""
import gzip

import numpy as np

from .genome import track
from .paths import BIN, OFF, PAIRS, REF3DG
from .score import spearman


def chrom_bins(lengths):
    return [int((L - OFF) // BIN) + 1 for L in lengths]


def dense_track(structs, trk, n_bins):
    A = np.full((n_bins, 3), np.nan)
    for p, v in structs.get(trk, {}).items():
        b = (int(p) - OFF) // BIN
        if 0 <= b < n_bins:
            A[b] = v
    return A


def procrustes_scaled(src, dst):
    """计算将 src 点映射到 dst 的旋转 R、统一缩放 s 和平移 t，满足 dst ≈ s R src + t。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    X, Y = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(X.T @ Y)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = float((S * np.array([1.0, 1.0, d])).sum() / (X ** 2).sum())
    return R, s, mu_d - s * (R @ mu_s)


def apply_sim(A, R, s, t):
    out = np.full_like(A, np.nan)
    ok = ~np.isnan(A).any(1)
    if ok.any():
        out[ok] = s * (A[ok] @ R.T) + t
    return out


def gauge_and_align(ours, ref, names, nbins):
    """返回 (每条染色体的 gauge, R, s, t)，使 ours 与 ref(mat) 达到最佳对齐。"""
    gauge = []
    for c in range(len(names)):
        rmat = dense_track(ref, "%s(mat)" % names[c], nbins[c])
        rpat = dense_track(ref, "%s(pat)" % names[c], nbins[c])
        a = dense_track(ours, track(c, 0), nbins[c])
        b = dense_track(ours, track(c, 1), nbins[c])
        i, j = np.triu_indices(nbins[c], 1)

        def rho(X, Y):
            dx, dy = X[i] - X[j], Y[i] - Y[j]
            ok = np.isfinite(dx).all(1) & np.isfinite(dy).all(1)
            if ok.sum() < 50:
                return -9.0
            return spearman(np.linalg.norm(dx[ok], axis=1), np.linalg.norm(dy[ok], axis=1))

        # 联合评分两种配对：(a,mat)+(b,pat) 与 (a,pat)+(b,mat)
        direct = rho(a, rmat) + rho(b, rpat)
        swap = rho(a, rpat) + rho(b, rmat)
        gauge.append(0 if direct >= swap else 1)
    S, D = [], []
    for c in range(len(names)):
        chosen = dense_track(ours, track(c, gauge[c]), nbins[c])
        rmat = dense_track(ref, "%s(mat)" % names[c], nbins[c])
        ok = ~np.isnan(chosen).any(1) & ~np.isnan(rmat).any(1)
        S.append(chosen[ok]); D.append(rmat[ok])
    return gauge, procrustes_scaled(np.vstack(S), np.vstack(D))


def homolog_stats(ours, ref, names, nbins, gauge=None):
    """逐染色体返回每个 copy 的 Rg、质心间距，以及 reference 的同类指标。"""
    rows = []
    for c, nm in enumerate(names):
        o = [dense_track(ours, track(c, k), nbins[c]) for k in (0, 1)]
        r = [dense_track(ref, "%s(%s)" % (nm, w), nbins[c]) for w in ("mat", "pat")]

        def rg(A):
            ok = ~np.isnan(A).any(1)
            return float(np.sqrt(((A[ok] - A[ok].mean(0)) ** 2).sum(1).mean())) if ok.sum() else np.nan

        def sep(A, B):
            oa, ob = ~np.isnan(A).any(1), ~np.isnan(B).any(1)
            if not (oa.any() and ob.any()):
                return np.nan
            return float(np.linalg.norm(A[oa].mean(0) - B[ob].mean(0)))

        rows.append({"chrom": nm,
                     "ours_rg": [rg(o[0]), rg(o[1])],
                     "ref_rg": [rg(r[0]), rg(r[1])],
                     "ours_sep": sep(o[0], o[1]),
                     "ref_sep": sep(r[0], r[1])})
    return rows


def territory_matrix(A_by_chrom):
    n = len(A_by_chrom)
    C = np.full((n, 3), np.nan)
    for i, A in enumerate(A_by_chrom):
        ok = ~np.isnan(A).any(1)
        if ok.sum():
            C[i] = A[ok].mean(0)
    return np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)


# --------------------------------------------------------------------------
# 正式图：独立归一化
# --------------------------------------------------------------------------
def _track_rows(structs, names, kind):
    """将全部有限 bead 返回为 (chromosome, copy, position, xyz) 元组。"""
    rows = []
    for ci, chrom in enumerate(names):
        if kind == "reference":
            specs = (("mat", "%s(mat)" % chrom), ("pat", "%s(pat)" % chrom))
        elif kind == "consensus":
            specs = (("single", track(ci, 0)),)
        else:
            specs = (("a", track(ci, 0)), ("b", track(ci, 1)))
        for copy, trk in specs:
            for pos, xyz in sorted(structs.get(trk, {}).items()):
                xyz = np.asarray(xyz, dtype=float)
                if np.isfinite(xyz).all():
                    rows.append((chrom, copy, int(pos), xyz))
    return rows


def independently_normalised_rows(structs, names, kind):
    """独立地将一整套结构居中，并除以其 RMS 半径。"""
    rows = _track_rows(structs, names, kind)
    if not rows:
        raise ValueError("no finite coordinates for %s" % kind)
    points = np.vstack([row[3] for row in rows])
    center = points.mean(axis=0)
    rms = float(np.sqrt(((points - center) ** 2).sum(axis=1).mean()))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("non-positive RMS radius for %s" % kind)
    norm_rows = [(chrom, copy, pos, xyz, (xyz - center) / rms)
                 for chrom, copy, pos, xyz in rows]
    return norm_rows, {"center": center.tolist(), "rms_radius": rms, "n_beads": len(rows)}


def _chrom_colors(names):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    return {name: cmap(i) for i, name in enumerate(names)}


def write_territory_figure(path, table_path, panels, names, provenance):
    """写出四个独立归一化的 3D scatter panel 和透明行。

    ``panels`` 是按顺序排列的 ``(label, structs, kind)`` 序列，其中 kind 为 ``reference``、``oracle``、``random`` 或 ``consensus``。显示完整坐标轨迹（包括 3 Mb metrics grid 以下的 beads）；数据表中的 provenance 会记录这一差异。
    """
    from . import figs  # 在 pyplot 之前导入 config；保留其 7 pt/300 dpi 规范。
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import json

    del figs
    colors = _chrom_colors(names)
    markers = {"mat": "o", "pat": "^", "a": "o", "b": "^", "single": "s"}
    panel_rows = []
    rows_out = []
    panel_meta = {}
    for label, structs, kind in panels:
        rows, meta = independently_normalised_rows(structs, names, kind)
        panel_meta[label] = meta
        panel_rows.append((label, rows))
        for chrom, copy, pos, raw, norm in rows:
            rows_out.append({"panel": label, "chromosome": chrom, "copy": copy,
                             "position_bp": pos, "raw_x": float(raw[0]),
                             "raw_y": float(raw[1]), "raw_z": float(raw[2]),
                             "normalized_x": float(norm[0]), "normalized_y": float(norm[1]),
                             "normalized_z": float(norm[2]),
                              "color_rgba": ",".join("%.5g" % x for x in colors[chrom]),
                              "marker": markers[copy]})

    # 单位 RMS 只有在各 panel 共享显示范围时才足以保证尺度可比。
    # 该边界是在每个 panel 独立归一化后推导的。
    all_norm = np.vstack([row[4] for _label, rows in panel_rows for row in rows])
    bound = float(np.max(np.abs(all_norm))) * 1.05
    bound = max(bound, 1.0)
    fig = plt.figure(figsize=(6.8, 7.2))
    for panel_index, (label, rows) in enumerate(panel_rows, start=1):
        ax = fig.add_subplot(2, 2, panel_index, projection="3d")
        grouped = {}
        for chrom, copy, _pos, _raw, norm in rows:
            grouped.setdefault((chrom, copy), []).append(norm)
        # 每条染色体/copy 一个 collection，而不是每个 bead 一个。TSV 输出
        # 下面仍按 bead 粒度记录。
        for (chrom, copy), points in grouped.items():
            points = np.vstack(points)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2.2, marker=markers[copy],
                       color=colors[chrom], alpha=0.78, linewidths=0)
        titles = {"oracle": "oracle (fully labeled subset fit)",
                  "consensus": "consensus (single trajectory)"}
        ax.set_title(titles.get(label, label))
        ax.set_xlabel("x", labelpad=-5)
        ax.set_ylabel("y", labelpad=-5)
        ax.set_zlabel("z", labelpad=-5)
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_zlim(-bound, bound)
        ax.tick_params(labelsize=7, pad=-2)
        ax.view_init(elev=20, azim=-55)
        ax.set_box_aspect((1, 1, 1))
    chromosome_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[c],
                                  markersize=3, label=c) for c in names]
    marker_handles = [Line2D([0], [0], marker="o", color="0.2", linestyle="None", markersize=4,
                             label="reference mat"),
                      Line2D([0], [0], marker="^", color="0.2", linestyle="None", markersize=4,
                             label="reference pat"),
                      Line2D([0], [0], marker="o", color="0.2", linestyle="None", markersize=4,
                             label="candidate A (arbitrary gauge)"),
                      Line2D([0], [0], marker="^", color="0.2", linestyle="None", markersize=4,
                             label="candidate B (arbitrary gauge)"),
                      Line2D([0], [0], marker="s", color="0.2", linestyle="None", markersize=4,
                             label="consensus single")]
    chromosome_legend = fig.legend(handles=chromosome_handles, title="Chromosome", ncol=4,
                                   loc="lower center", bbox_to_anchor=(0.5, 0.015), frameon=False,
                                   handletextpad=0.25, columnspacing=0.6)
    fig.add_artist(chromosome_legend)
    fig.legend(handles=marker_handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3,
               frameon=False, handletextpad=0.25, columnspacing=0.9)
    fig.text(0.5, 0.985, "Independent center + unit RMS per panel; no rigid/similarity fit",
             ha="center", va="top", fontsize=7)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.22, top=0.86, wspace=0.02, hspace=0.10)
    fig.savefig(path, dpi=300)
    plt.close(fig)

    columns = ["panel", "chromosome", "copy", "position_bp", "raw_x", "raw_y", "raw_z",
               "normalized_x", "normalized_y", "normalized_z", "color_rgba", "marker"]
    with open(table_path, "w") as out:
        out.write("# provenance=" + json.dumps(provenance, sort_keys=True) + "\n")
        out.write("# normalization=" + json.dumps(panel_meta, sort_keys=True) + "\n")
        out.write("\t".join(columns) + "\n")
        for row in rows_out:
            out.write("\t".join(str(row[key]) for key in columns) + "\n")
    return {"figure": path, "table": table_path, "normalization": panel_meta,
            "n_rows": len(rows_out)}


def _rg(points):
    if len(points) == 0:
        return float("nan")
    center = points.mean(axis=0)
    return float(np.sqrt(((points - center) ** 2).sum(axis=1).mean()))


def _representation_stats(label, structs, names, kind):
    rows, _ = independently_normalised_rows(structs, names, kind)
    by_copy = {}
    for chrom, copy, _pos, _raw, norm in rows:
        by_copy.setdefault((chrom, copy), []).append(norm)
    copy_stats = []
    for (chrom, copy), points in by_copy.items():
        points = np.vstack(points)
        copy_stats.append({"chromosome": chrom, "copy": copy, "points": points,
                           "centroid": points.mean(axis=0), "rg": _rg(points)})
    pair_distances = []
    for i, left in enumerate(copy_stats):
        for right in copy_stats[i + 1:]:
            if left["chromosome"] != right["chromosome"]:
                pair_distances.append(float(np.linalg.norm(left["centroid"] - right["centroid"])))
    mean_interchrom = float(np.mean(pair_distances)) if pair_distances else float("nan")
    n_interchrom_pairs = len(pair_distances)
    out = []
    for chrom in names:
        current = [item for item in copy_stats if item["chromosome"] == chrom]
        all_points = np.vstack([item["points"] for item in current]) if current else np.empty((0, 3))
        copy_rgs = [item["rg"] for item in current]
        mean_copy_rg = float(np.mean(copy_rgs)) if copy_rgs else float("nan")
        if len(current) == 2:
            separation = float(np.linalg.norm(current[0]["centroid"] - current[1]["centroid"]))
            homolog_ratio = separation / mean_copy_rg if mean_copy_rg > 0 else float("nan")
        else:
            separation = float("nan")
            homolog_ratio = float("nan")
        out.append({"candidate": label, "chromosome": chrom, "kind": kind,
                    "n_beads": int(len(all_points)), "n_copies": int(len(current)),
                    "mean_copy_rg": mean_copy_rg,
                    "mean_interchromosome_copy_centroid_distance": mean_interchrom,
                    "n_interchromosome_copy_centroid_pairs": int(n_interchrom_pairs),
                    "mean_copy_rg_over_mean_interchromosome_copy_centroid_distance": (
                        mean_copy_rg / mean_interchrom if mean_interchrom > 0 else float("nan")),
                    "homolog_centroid_separation": separation,
                    "homolog_separation_over_mean_rg": homolog_ratio})
    ratios = [row["mean_copy_rg_over_mean_interchromosome_copy_centroid_distance"] for row in out
              if np.isfinite(row["mean_copy_rg_over_mean_interchromosome_copy_centroid_distance"])]
    homolog_values = [row["homolog_separation_over_mean_rg"] for row in out]
    finite_homolog = [value for value in homolog_values if np.isfinite(value)]
    summary = {"candidate": label, "kind": kind,
               "mean_interchromosome_copy_centroid_distance": mean_interchrom,
               "n_interchromosome_copy_centroid_pairs": int(n_interchrom_pairs),
               "median_mean_copy_rg_over_mean_interchromosome_copy_centroid_distance": (
                   float(np.median(ratios)) if ratios else float("nan")),
               "median_homolog_separation_over_mean_rg": (float(np.median(finite_homolog))
                                                           if finite_homolog else float("nan")),
               "n_chromosomes": len(out)}
    return out, summary


def write_scale_free_diagnostics(path, table_path, panels, names, provenance):
    """写出不依赖尺度的 homolog/territory 诊断，以及包含每个值的 TSV。"""
    from . import figs
    import matplotlib.pyplot as plt
    import json

    del figs
    rows = []
    summary = []
    for label, structs, kind in panels:
        current, aggregate = _representation_stats(label, structs, names, kind)
        rows.extend(current)
        summary.append(aggregate)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    x = np.arange(len(names))
    for candidate, color, marker in (("reference", "0.15", "o"), ("oracle", "C3", "^"),
                                     ("random", "C0", "s")):
        values = [row["homolog_separation_over_mean_rg"] for row in rows
                  if row["candidate"] == candidate]
        if values:
            axes[0].plot(x, values, marker=marker, ms=2.5, color=color, label=candidate)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([name.replace("chr", "") for name in names], rotation=0)
    axes[0].set_xlabel("chromosome")
    axes[0].set_ylabel("homolog centroid separation / mean Rg")
    axes[0].grid(alpha=0.25, lw=0.4)
    axes[0].legend(frameon=False)

    labels = [item["candidate"] for item in summary]
    values = [item["median_mean_copy_rg_over_mean_interchromosome_copy_centroid_distance"]
              for item in summary]
    colors = ["0.2" if label == "reference" else ("C3" if label == "oracle" else
              ("C0" if label == "random" else "0.55")) for label in labels]
    axes[1].bar(np.arange(len(labels)), values, color=colors, width=0.65)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Mean copy Rg / mean interchromosome\ncopy-centroid distance")
    axes[1].set_title("Median across 20 chromosomes")
    axes[1].grid(alpha=0.25, lw=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    columns = ["candidate", "chromosome", "kind", "n_beads", "n_copies", "mean_copy_rg",
               "mean_interchromosome_copy_centroid_distance",
               "n_interchromosome_copy_centroid_pairs",
               "mean_copy_rg_over_mean_interchromosome_copy_centroid_distance",
               "homolog_centroid_separation", "homolog_separation_over_mean_rg"]
    table_provenance = dict(provenance)
    table_provenance.update({
        "fig6_territory_definition": "per-chromosome mean copy Rg divided by mean distance over copy-centroid pairs from different chromosomes",
        "fig6_pair_counts": "diploid 40-track panels exclude 20 homolog pairs: 760 pairs; consensus 20-track panel: 190 pairs",
        "not_directly_comparable_to_014_fig6": "current definition uses copy-specific Rg, all different-chromosome copy-centroid pairs and full beads; the complete 014 figure-generating script/provenance is unavailable, so legacy values are not directly comparable",
    })
    with open(table_path, "w") as out:
        out.write("# provenance=" + json.dumps(table_provenance, sort_keys=True) + "\n")
        out.write("# summary=" + json.dumps(summary, sort_keys=True) + "\n")
        out.write("\t".join(columns) + "\n")
        for row in rows:
            out.write("\t".join(str(row[key]) for key in columns) + "\n")
    return {"figure": path, "table": table_path, "summary": summary, "n_rows": len(rows)}
