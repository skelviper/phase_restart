#!/usr/bin/env python
"""049 评价图：最多两张 PNG（英文、7pt、3-inch 基准面板、300 dpi）。

图1：核心指标比较（12 fit 按 loss 分组，baseline 与两个 initial 对照）。
图2：whole-genome 5 panel = reference + baseline + 3 个预先冻结的 loss display endpoint，
     每个 dataset 只做一次 global gauge（去中心 + 单位 Rg + 单个 proper Kabsch），固定视角、共同 axes。

只有 gate 之后（``gates/reference_open.json`` 存在）才可读取 reference 坐标。``--selfcheck``
使用内部合成坐标，仅验证绘图代码路径，不产生正式图。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval049_inputs as inputs  # noqa: E402
import eval049_lib as lib  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", str(lib.EVAL / "logs/mplconfig"))
(lib.EVAL / "logs/mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLOTS = lib.EVAL / "plots"
RESULTS = lib.EVAL / "results"
REFERENCE_OPEN_PATH = lib.EVAL / "gates/reference_open.json"
BASE_PANEL_INCHES = 3.0
DPI = 300
FONT_PT = 7

LOSS_LABEL = {"A": "A marginal_G", "B": "B hard_observed", "C": "C max_rate"}
COLORS = {"raw-consensus": "#1f77b4", "raw-random": "#ff7f0e", "ms-consensus": "#2ca02c", "ms-random": "#d62728"}
MARKERS = {"raw-consensus": "o", "raw-random": "s", "ms-consensus": "^", "ms-random": "D"}


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": FONT_PT, "axes.titlesize": FONT_PT, "axes.labelsize": FONT_PT,
        "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT,
        "figure.dpi": DPI, "savefig.dpi": DPI, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


def read_results() -> dict[str, Any]:
    evaluation = lib.read_json(RESULTS / "evaluation.json")
    if evaluation.get("smoke"):
        raise RuntimeError("refusing to make formal figures from a smoke evaluation.json")
    return evaluation


def dataset_row(evaluation: Mapping[str, Any], dataset_id: str) -> Mapping[str, Any]:
    entry = evaluation["datasets"][dataset_id]
    mac = entry["r2"]["macro_full_20_all_chromosomes_required"]
    centers = entry["spatial"]["merged_chr_centers"]
    copy_centers = entry["spatial"]["copy_centers"]["primary"]
    return {
        "matched": mac["pearson"]["matched"], "cross": mac["pearson"]["cross"],
        "contrast": mac["pearson"]["contrast"], "min_margin": mac["pearson"]["min_margin"],
        "spearman_matched": mac["spearman"]["matched"],
        "inter_pearson": entry["inter"].get("pearson"), "inter_spearman": entry["inter"].get("spearman"),
        "center_spearman": centers["spearman"], "center_stress": centers["normalized_stress"],
        "center_pearson": centers["pearson"], "copy_center_spearman": copy_centers["spearman"],
        "copy_center_stress": copy_centers["normalized_stress"],
        "terminal": entry["terminal"],
    }


def figure1(evaluation: Mapping[str, Any], out: Path) -> dict[str, Any]:
    fit_ids = [fid for fid in lib.EXPECTED_FIT_IDS]
    rows = {fid: dataset_row(evaluation, fid) for fid in fit_ids}
    baseline = dataset_row(evaluation, "baseline-046-real-extension-G-full-J")
    initials = {name: dataset_row(evaluation, "initial-%s" % name) for name in lib.SOURCES}

    panels = [("matched", "Within-chr matched Pearson r", False),
              ("contrast", "Copy contrast (matched - cross)", False),
              ("center_stress", "Chromosome-center stress", True)]
    figure, axes = plt.subplots(1, 3, figsize=(3 * BASE_PANEL_INCHES, 3.4))
    for ax, (field, title, lower_better) in zip(axes, panels):
        rng = np.random.default_rng(4242)
        for li, loss in enumerate(lib.LOSES):
            for solver in lib.SOLVERS:
                for source in lib.SOURCES:
                    key = "%s-%s" % (solver, source)
                    fit_id = "%s-%s-%s" % (loss, solver, source)
                    value = rows[fit_id][field]
                    x = li + (lib.SOLVERS.index(solver) * 2 + lib.SOURCES.index(source) - 1.5) * 0.18
                    x += rng.normal(0.0, 0.012)
                    face = COLORS[key] if rows[fit_id]["terminal"] == "converged" else "none"
                    ax.scatter([x], [value], s=16, marker=MARKERS[key], facecolors=face, edgecolors=COLORS[key],
                               linewidths=0.6, zorder=3, label=(key if li == 0 else None))
        ax.axhline(baseline[field], color="black", linestyle="--", linewidth=0.7, zorder=1)
        ax.axhline(initials["consensus"][field], color="grey", linestyle=":", linewidth=0.7, zorder=1)
        ax.axhline(initials["random"][field], color="grey", linestyle="-.", linewidth=0.7, zorder=1)
        ax.set_xticks(range(len(lib.LOSES)))
        ax.set_xticklabels([LOSS_LABEL[loss] for loss in lib.LOSES], rotation=12, ha="right")
        ax.set_title(title, pad=4)
        ax.grid(alpha=0.25, linewidth=0.4)
        if lower_better:
            ax.invert_yaxis()
    axes[0].set_ylabel("value")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="black", linestyle="--", linewidth=0.7),
                plt.Line2D([], [], color="grey", linestyle=":", linewidth=0.7),
                plt.Line2D([], [], color="grey", linestyle="-.", linewidth=0.7)]
    labels += ["046 baseline", "initial-consensus", "initial-random"]
    # 共享图例与说明放在面板下方空白带，避免覆盖数据点
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=4,
                  frameon=False, handletextpad=0.35, columnspacing=1.0)
    figure.text(0.5, 0.012,
                "filled marker = converged; open marker = other terminal. R2 = within-chromosome matched Pearson on frozen "
                "common pairs (fixed 20 chr);\ncenter stress = normalized stress of the 190 merged-chromosome center distances.",
                ha="center", va="bottom", fontsize=FONT_PT)
    figure.tight_layout(pad=0.4, rect=(0.0, 0.20, 1.0, 1.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=DPI)
    plt.close(figure)
    return {"path": str(out), "panels": [panel[0] for panel in panels],
            "note": "12 fits grouped by loss; baseline and both blind initials as horizontal references; shared legend below panels"}


def gauge_points(coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """每个 dataset 一次 global gauge：去中心 + 单位 Rg + 单个 proper Kabsch 旋转到 reference。"""
    points = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    points = points - points.mean(axis=0)
    ref = ref - ref.mean(axis=0)
    points = points / math.sqrt(float(np.mean(np.sum(points ** 2, axis=1))))
    ref = ref / math.sqrt(float(np.mean(np.sum(ref ** 2, axis=1))))
    h = points.T @ ref
    u, _, vt = np.linalg.svd(h)
    rot = u @ vt
    if float(np.linalg.det(rot)) < 0:
        u = u.copy()
        u[:, -1] *= -1.0
        rot = u @ vt
    return points @ rot


def chromosome_index_per_locus(data: lib.Aggregate, keep: np.ndarray | None = None) -> np.ndarray:
    """每个 locus 的 chromosome 索引（可选 keep 掩码）。"""
    index = np.zeros(data.n_loci, dtype=np.int64)
    for ci in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(ci)
        index[slc] = ci
    return index if keep is None else index[np.asarray(keep, dtype=bool)]


def chromosome_color_array(data: lib.Aggregate, keep: np.ndarray | None = None) -> np.ndarray:
    """与 coords[:, keep].reshape(-1, 3) 同序（copy-major）的 chromosome 索引数组。"""
    per_locus = chromosome_index_per_locus(data, keep)
    return np.concatenate([per_locus, per_locus])


def figure2(evaluation: Mapping[str, Any], reference: np.ndarray, data: lib.Aggregate,
            manifest: Mapping[str, Any], selection: Mapping[str, Any], out: Path) -> dict[str, Any]:
    display = {loss: selection["display"][loss]["fit_id"] for loss in lib.LOSES}
    panel_ids = ["reference"] + ["baseline-046-real-extension-G-full-J"] + [display[loss] for loss in lib.LOSES]
    titles = ["reference (P9016 1Mb 3DG)"]
    titles.append("046 baseline: %s" % evaluation["datasets"]["baseline-046-real-extension-G-full-J"]["terminal"])
    for loss in lib.LOSES:
        fit_id = display[loss]
        titles.append("%s display: %s\n%s" % (loss, fit_id, evaluation["datasets"][fit_id]["terminal"]))
    coords_by_id: dict[str, np.ndarray] = {"reference": reference}
    lut = {str(row["fit_id"]): row for row in manifest["fits"]}
    for fit_id in panel_ids[1:]:
        if fit_id == "baseline-046-real-extension-G-full-J":
            path = lib.BASELINE_NPZ
        else:
            row = lut[fit_id]
            path = lib.resolve_path(row["npz_path"])
        coords_by_id[fit_id] = lib.load_coords_npz(path)["coordinates"]

    # reference 存在真实缺失 loci：图 2 只在全部 panel 共同 finite 的支持上作图，并写明支持大小
    finite_masks = [np.isfinite(coords_by_id[panel_id]).all(axis=(0, 2)) for panel_id in panel_ids]
    common_finite = np.logical_and.reduce(finite_masks)
    kept = int(common_finite.sum())
    if kept == 0:
        raise RuntimeError("no common finite loci across the figure panels")
    for panel_id in panel_ids:
        coords_by_id[panel_id] = coords_by_id[panel_id][:, common_finite, :]
    reference = coords_by_id["reference"]
    chrom = chromosome_color_array(data, common_finite)
    gauged = {panel_id: gauge_points(coords_by_id[panel_id], reference) for panel_id in panel_ids}
    stacked = np.concatenate([gauged[panel_id] for panel_id in panel_ids], axis=0)
    limit = float(np.abs(stacked).max()) * 1.02
    figure, axes = plt.subplots(1, 5, figsize=(5 * BASE_PANEL_INCHES, 3.2),
                                subplot_kw={"projection": "3d"})
    cmap = plt.get_cmap("tab20")
    for ax, panel_id, title in zip(axes, panel_ids, titles):
        points = gauged[panel_id]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.1, c=chrom, cmap=cmap, vmin=0, vmax=19,
                   alpha=0.85, linewidths=0)
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=30)
        ax.set_title(title, pad=2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
    figure.suptitle("whole-genome gauge only (center + unit Rg + one global proper rotation); no per-chromosome fit; "
                    "common finite support %d/%d loci (%d beads)" % (kept, data.n_loci, 2 * kept),
                    fontsize=FONT_PT, y=0.99)
    # 统一 20 chr 色标（单个共享 colorbar，不重复 20 份图例）
    cax = figure.add_axes([0.30, 0.052, 0.40, 0.020])
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-0.5, vmax=19.5), cmap=cmap),
                               cax=cax, orientation="horizontal", ticks=np.arange(20))
    # 染色体名用聚合输入的原值（chr1..chr19, chrX），不硬编码 chr1..chr20；字号统一 7 pt
    colorbar.ax.set_xticklabels([str(name) for name in data.chromosome_names], rotation=0, fontsize=FONT_PT)
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label("chromosome (both copies)", fontsize=FONT_PT, labelpad=1)
    colorbar.ax.tick_params(length=1.5, pad=1)
    colorbar.outline.set_linewidth(0.4)
    figure.tight_layout(pad=0.4, rect=(0.0, 0.135, 1.0, 0.86))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=DPI)
    plt.close(figure)
    return {"path": str(out), "panels": panel_ids, "titles": titles,
            "display_selection": display, "gauge": "single global proper alignment per dataset; common axes and view",
            "common_finite_support": {"kept_loci": kept, "total_loci": int(data.n_loci), "beads": 2 * kept,
                                      "policy": "panels drawn on the loci finite in every plotted dataset"}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--selection", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    apply_style()
    outdir = Path(args.outdir) if args.outdir else PLOTS

    if args.selfcheck:
        return selfcheck(Path(args.outdir) if args.outdir else lib.EVAL / "codecheck/plots")
    if not REFERENCE_OPEN_PATH.is_file():
        raise RuntimeError("reference has not been opened (no gates/reference_open.json); figures are post-gate only")
    evaluation = read_results()
    manifest = inputs.load_manifest(Path(args.manifest)) if args.manifest else None
    selection = inputs.load_selection(Path(args.selection)) if args.selection else None
    if manifest is None or selection is None:
        raise RuntimeError("formal figures require --manifest and --selection")
    data = lib.Aggregate()
    reference = lib.three_dg_to_array(lib.load_3dg(lib.REFERENCE_PATH), data, track_mode="reference")
    fig1 = figure1(evaluation, outdir / "049_figure1_core_metrics.png")
    fig2 = figure2(evaluation, reference, data, manifest, selection, outdir / "049_figure2_whole_genome.png")
    lib.write_json(RESULTS / "figure_manifest.json",
                   {"schema": "p9016-049-figures-v1", "figure1": fig1, "figure2": fig2,
                    "spec": {"base_panel_inches": BASE_PANEL_INCHES, "dpi": DPI, "font_pt": FONT_PT,
                             "annotation_language": "English", "png_count": 2},
                    "selection_consumed": "pre-frozen display endpoints; figures do not select anything"})
    print("FIGURES WRITTEN")
    print("  %s" % fig1["path"])
    print("  %s" % fig2["path"])
    return 0


def selfcheck(outdir: Path) -> int:
    """合成数据绘图自检：不读 reference、不读正式结果。"""
    rng = np.random.default_rng(5)
    dataset_ids = list(lib.EXPECTED_FIT_IDS) + ["initial-consensus", "initial-random",
                                                "baseline-046-real-extension-G-full-J"]
    datasets = {}
    for index, dataset_id in enumerate(dataset_ids):
        matched = 0.2 + 0.02 * index
        datasets[dataset_id] = {
            "terminal": "budget_not_converged" if index % 2 else "converged",
            "r2": {"macro_full_20_all_chromosomes_required": {
                "pearson": {"matched": matched, "cross": matched - 0.2, "contrast": 0.2,
                            "min_margin": 0.01, "direct": matched, "swapped": matched - 0.2,
                            "copy_A_margin": 0.01, "copy_B_margin": 0.01, "matched_mat_margin": 0.01,
                            "matched_pat_margin": 0.01},
                "spearman": {"matched": matched, "cross": matched - 0.2, "contrast": 0.2, "min_margin": 0.01,
                             "direct": matched, "swapped": matched - 0.2, "copy_A_margin": 0.01,
                             "copy_B_margin": 0.01, "matched_mat_margin": 0.01, "matched_pat_margin": 0.01}}},
            "inter": {"pearson": 0.6 + 0.01 * index, "spearman": 0.65 + 0.01 * index},
            "spatial": {"merged_chr_centers": {"pearson": 0.5, "spearman": 0.5, "normalized_stress": 0.9 - 0.01 * index},
                        "copy_centers": {"primary": {"pearson": 0.5, "spearman": 0.5, "normalized_stress": 0.9}}},
        }
    evaluation = {"smoke": False, "datasets": datasets}
    data = lib.Aggregate()
    reference = rng.normal(size=(2, data.n_loci, 3)) * 0.05
    reference[1, 5:9] = np.nan  # 覆盖“reference 存在真实缺失”的作图路径
    manifest = {"fits": [{"fit_id": fid, "npz_path": str(lib.EVAL / "codecheck/coords" / fid / "1Mb.npz")}
                         for fid in lib.EXPECTED_FIT_IDS]}
    selection = {"display": {loss: {"fit_id": "%s-raw-consensus" % loss} for loss in lib.LOSES}}
    figures = outdir
    figures.mkdir(parents=True, exist_ok=True)
    fig1 = figure1(evaluation, figures / "selfcheck_figure1.png")
    fig2 = figure2(evaluation, reference, data, manifest, selection, figures / "selfcheck_figure2.png")
    for item in (fig1, fig2):
        if not Path(item["path"]).is_file() or Path(item["path"]).stat().st_size < 10_000:
            raise RuntimeError("figure selfcheck output missing or too small: %s" % item["path"])
    print("PLOT SELFCHECK PASS (synthetic coords; formal figures untouched)")
    print("  %s" % fig1["path"])
    print("  %s" % fig2["path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
