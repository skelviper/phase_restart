#!/usr/bin/env python
"""比较两个候选 3DG 与同一 reference 的逐染色体 Pearson 指标，并画三个并排 boxplot。

冻结口径（与本次 048 任务一致）：
- 观察单位：每条染色体一个等权观察（20 条染色体 = 同一细胞的 20 对），不把基因组 pair 当独立重复。
- mask：读 frozen snapshot 的每 chr positions/pair_i/pair_j/common（3Mb 起 1Mb grid），再用
  6 条 track（候选 A 双 copy + 候选 B 双 copy + reference 双 copy）在该 pair 上全部 finite 的交集收窄；
  两个候选使用完全相同的支持，收窄量逐 chr 记录（dropped_from_mask）。
- 指标 Pearson：四个 rho A_mat/A_pat/B_mat/B_pat；direct=(A_mat+B_pat)/2，swapped=(A_pat+B_mat)/2；
  same=matched=max(direct,swapped)，cross=min，contrast=same-cross；|direct-swapped|<=1e-12 记 tie，
  数值取两方向均值、contrast=0。互换按整条染色体统一判定，绝不逐 pair 挑最大。
- undefined 保留 NA 并在 TSV/summary 报告，不丢染色体、不把少于 20 条的均值当完整 macro。
- 距离指标对 3D 刚体朝向不变，不做 Kabsch 配准。

只读两份候选与一份 reference，不重跑评价、不重建 mask、不改冻结数值。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
import plot_3dg_comparison as p3d  # noqa: E402  复用 parse_3dg/load_mask/points_on_grid/distance_vector/corr

RESOLUTION = 1_000_000
TIE_TOLERANCE = 1e-12
METRICS = (
    ("same_matched", "Same (matched) Pearson"),
    ("cross", "Cross Pearson"),
    ("contrast", "Contrast (same - cross)"),
)
GROUP_COLORS = ("#1f77b4", "#ff7f0e")  # A(旧)=蓝, B(新)=橙
JITTER_SEED = 20260915


def utc_stamp() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d_%H%M%S"), now.isoformat(timespec="seconds").replace("+00:00", "Z")


def track_finite(tracks: dict[str, dict[int, np.ndarray]], track: str, positions: np.ndarray) -> np.ndarray:
    return np.isfinite(p3d.points_on_grid(tracks, track, positions)).all(axis=1)


def compare_chromosome(
    candidate_a: dict[str, dict[int, np.ndarray]],
    candidate_b: dict[str, dict[int, np.ndarray]],
    reference: dict[str, dict[int, np.ndarray]],
    mask: dict[str, Any],
    chromosome: str,
) -> dict[str, dict[str, Any]]:
    """返回 {group: metrics}，两个 group 共用同一 pair 支持。"""
    ci = p3d.CHROMOSOMES.index(chromosome)
    positions, pair_i, pair_j = mask["positions"], mask["pair_i"], mask["pair_j"]
    keep = mask["common"].copy()
    for tracks, names in (
        (candidate_a, (p3d.candidate_track(ci, 0), p3d.candidate_track(ci, 1))),
        (candidate_b, (p3d.candidate_track(ci, 0), p3d.candidate_track(ci, 1))),
        (reference, (p3d.reference_track(chromosome, 0), p3d.reference_track(chromosome, 1))),
    ):
        for name in names:
            finite = track_finite(tracks, name, positions)
            keep &= finite[pair_i] & finite[pair_j]
    used_i, used_j = pair_i[keep], pair_j[keep]
    ref_mat = p3d.distance_on_grid(reference, p3d.reference_track(chromosome, 0), positions, used_i, used_j)
    ref_pat = p3d.distance_on_grid(reference, p3d.reference_track(chromosome, 1), positions, used_i, used_j)

    support = {
        "n_positions": int(len(positions)),
        "n_total_pairs": int(len(pair_i)),
        "n_mask_common": int(mask["common"].sum()),
        "n_final_common": int(keep.sum()),
        "n_pairs_used": int(len(used_i)),
        "dropped_from_mask": int(mask["common"].sum() - keep.sum()),
    }
    out: dict[str, dict[str, Any]] = {}
    for group, candidate in (("A", candidate_a), ("B", candidate_b)):
        copy_a = p3d.distance_on_grid(candidate, p3d.candidate_track(ci, 0), positions, used_i, used_j)
        copy_b = p3d.distance_on_grid(candidate, p3d.candidate_track(ci, 1), positions, used_i, used_j)
        rho = {
            "A_mat": p3d.corr(copy_a, ref_mat),
            "A_pat": p3d.corr(copy_a, ref_pat),
            "B_mat": p3d.corr(copy_b, ref_mat),
            "B_pat": p3d.corr(copy_b, ref_pat),
        }
        row: dict[str, Any] = dict(rho)
        direct = (rho["A_mat"] + rho["B_pat"]) / 2.0
        swapped = (rho["A_pat"] + rho["B_mat"]) / 2.0
        row["direct"] = float(direct)
        row["swapped"] = float(swapped)
        if not all(np.isfinite(list(rho.values()))):
            row.update({"orientation": "undefined", "tie": False, "same_matched": float("nan"), "cross": float("nan"), "contrast": float("nan")})
        elif abs(direct - swapped) <= TIE_TOLERANCE:
            mean_both = (direct + swapped) / 2.0
            row.update({"orientation": "tie", "tie": True, "same_matched": float(mean_both), "cross": float(mean_both), "contrast": 0.0})
        else:
            same, cross = (direct, swapped) if direct > swapped else (swapped, direct)
            row.update({"orientation": "direct" if direct > swapped else "swapped", "tie": False, "same_matched": float(same), "cross": float(cross), "contrast": float(same - cross)})
        row.update(support)
        out[group] = row
    return out


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    a = np.asarray([row["A"][key] for row in rows], dtype=np.float64)
    b = np.asarray([row["B"][key] for row in rows], dtype=np.float64)
    pair_ok = np.isfinite(a) & np.isfinite(b)
    diff = a[pair_ok] - b[pair_ok]

    def stats(values: np.ndarray) -> dict[str, Any]:
        good = values[np.isfinite(values)]
        return {
            "n_defined": int(good.size),
            "mean": float(np.mean(good)) if good.size else None,
            "median": float(np.median(good)) if good.size else None,
            "values": [float(v) for v in values],
        }

    tie_mask = np.abs(diff) <= TIE_TOLERANCE
    return {
        "A": stats(a),
        "B": stats(b),
        "paired_diff_A_minus_B": {
            "n_pairs": int(diff.size),
            "mean": float(np.mean(diff)) if diff.size else None,
            "median": float(np.median(diff)) if diff.size else None,
            "n_A_higher": int(np.sum(diff > TIE_TOLERANCE)),
            "n_A_lower": int(np.sum(diff < -TIE_TOLERANCE)),
            "n_equal": int(np.sum(tie_mask)),
            "n_na_excluded": int(np.sum(~pair_ok)),
        },
    }


def plot_panels(rows: list[dict[str, Any]], labels: tuple[str, str], out_path: Path) -> None:
    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
    jitter = np.random.default_rng(JITTER_SEED).uniform(-0.13, 0.13, size=len(rows))
    x_positions = (1.0, 2.0)
    same_cross_values = np.asarray(
        [row[g][k] for row in rows for g in ("A", "B") for k in ("same_matched", "cross")], dtype=np.float64
    )
    contrast_values = np.asarray([row[g]["contrast"] for row in rows for g in ("A", "B")], dtype=np.float64)

    def limits(values: np.ndarray) -> tuple[float, float]:
        good = values[np.isfinite(values)]
        lo, hi = float(np.min(good)), float(np.max(good))
        span = hi - lo if hi > lo else max(abs(hi), 1.0) * 0.01
        return lo - 0.12 * span, hi + 0.12 * span

    shared_limits = limits(same_cross_values)
    contrast_limits = limits(contrast_values)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.9))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.78, bottom=0.145, wspace=0.30)
    for panel, (key, title) in enumerate(METRICS):
        ax = axes[panel]
        datasets, box_colors = [], []
        for group_index, group in enumerate(("A", "B")):
            values = np.asarray([row[group][key] for row in rows], dtype=np.float64)
            datasets.append(values[np.isfinite(values)])
            box_colors.append(GROUP_COLORS[group_index])
        bp = ax.boxplot(
            datasets,
            positions=list(x_positions),
            widths=0.5,
            patch_artist=True,
            showfliers=False,  # 叠点时不再重复画 flier
            whis=1.5,
            medianprops={"color": "#111111", "linewidth": 0.8},
            whiskerprops={"color": "#444444", "linewidth": 0.6},
            capprops={"color": "#444444", "linewidth": 0.6},
            boxprops={"linewidth": 0.6, "edgecolor": "#333333"},
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.30)
        for row, offset in zip(rows, jitter):
            va, vb = row["A"][key], row["B"][key]
            if np.isfinite(va) and np.isfinite(vb):
                ax.plot([x_positions[0] + offset, x_positions[1] + offset], [va, vb], color="0.62", linewidth=0.45, zorder=2)
            for group_index, group in enumerate(("A", "B")):
                value = row[group][key]
                if np.isfinite(value):
                    ax.scatter(
                        x_positions[group_index] + offset,
                        value,
                        s=7,
                        facecolor=GROUP_COLORS[group_index],
                        edgecolor="#222222",
                        linewidth=0.3,
                        alpha=0.9,
                        zorder=3,
                    )
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(list(labels))
        ax.set_title(title, pad=4)
        ax.set_xlim(0.4, 2.6)
        ax.set_ylim(*(contrast_limits if key == "contrast" else shared_limits))
        ax.set_ylabel("same - cross (Pearson r)" if key == "contrast" else "Pearson r")
        ax.grid(axis="y", color="0.90", linewidth=0.4)
        ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=GROUP_COLORS[i], alpha=0.30, edgecolor="#333333", linewidth=0.6, label=label)
        for i, label in enumerate(labels)
    ] + [Line2D([], [], marker="o", linestyle="none", markersize=3.2, markerfacecolor="0.45", markeredgecolor="#222222", markeredgewidth=0.3, label="per-chromosome pair (20 chr, one cell)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncols=3, frameon=False, handlelength=1.4, columnspacing=1.6)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-candidate per-chromosome Pearson boxplots against one reference.")
    parser.add_argument("--candidate-a", required=True, type=Path)
    parser.add_argument("--candidate-b", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path, help="frozen mask snapshot npz (per-chr positions/pair_i/pair_j/common)")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--label-a", default="candidate A")
    parser.add_argument("--label-b", default="candidate B")
    parser.add_argument("--expect-sha-a", default=None, help="若给出则先核对候选 A 的 SHA256，不符即拒绝运行")
    parser.add_argument("--expect-sha-b", default=None)
    parser.add_argument("--png-name", default="metric_boxplots.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # SHA 门：两个候选先核对，再打开 reference。
    sha_a = p3d.sha256_file(args.candidate_a)
    sha_b = p3d.sha256_file(args.candidate_b)
    for label, digest, expected in (("A", sha_a, args.expect_sha_a), ("B", sha_b, args.expect_sha_b)):
        if expected and digest != expected:
            print(f"SHA mismatch for candidate {label}: got {digest}, expected {expected}", file=sys.stderr)
            return 2
    print(f"candidate A sha256={sha_a}", flush=True)
    print(f"candidate B sha256={sha_b}", flush=True)

    candidate_a = p3d.parse_3dg(args.candidate_a)
    candidate_b = p3d.parse_3dg(args.candidate_b)
    reference = p3d.parse_3dg(args.reference)

    rows: list[dict[str, Any]] = []
    extra_drop: list[dict[str, Any]] = []
    for chromosome in p3d.CHROMOSOMES:
        mask = p3d.load_mask(args.mask, chromosome, RESOLUTION)
        per_group = compare_chromosome(candidate_a, candidate_b, reference, mask, chromosome)
        support = per_group["A"]
        rows.append({"chromosome": chromosome, **per_group})
        if support["dropped_from_mask"]:
            extra_drop.append({
                "chromosome": chromosome,
                "n_mask_common": support["n_mask_common"],
                "n_final_common": support["n_final_common"],
                "dropped": support["dropped_from_mask"],
            })
        print(
            f"{chromosome}: mask_common={support['n_mask_common']} final_common={support['n_final_common']} "
            f"A_same={per_group['A']['same_matched']:.4f} A_cross={per_group['A']['cross']:.4f} "
            f"B_same={per_group['B']['same_matched']:.4f} B_cross={per_group['B']['cross']:.4f}",
            flush=True,
        )

    tsv_path = outdir / "per_chromosome.tsv"
    columns = [
        "chromosome", "candidate", "label", "n_positions", "n_total_pairs", "n_mask_common", "n_final_common",
        "n_pairs_used", "dropped_from_mask", "rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat",
        "direct", "swapped", "orientation", "tie", "same_matched", "cross", "contrast",
    ]
    label_of = {"A": args.label_a, "B": args.label_b}
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            for group in ("A", "B"):
                data = row[group]
                values = [row["chromosome"], group, label_of[group]] + [
                    data[key] for key in ("n_positions", "n_total_pairs", "n_mask_common", "n_final_common", "n_pairs_used", "dropped_from_mask")
                ] + [data[key] for key in ("A_mat", "A_pat", "B_mat", "B_pat", "direct", "swapped", "orientation")] + [
                    int(data["tie"]), data["same_matched"], data["cross"], data["contrast"],
                ]
                handle.write("\t".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in values) + "\n")

    sha_a_short, sha_b_short = sha_a[:8], sha_b[:8]
    summary = {
        "schema": "p9016-two-candidate-per-chromosome-metrics-v1",
        "generated_utc": utc_stamp()[1],
        "unit": "one equal-weight observation per chromosome; 20 paired chromosomes from a single real P9016 cell",
        "candidate_a": {"label": args.label_a, "path": str(args.candidate_a.resolve()), "sha256": sha_a},
        "candidate_b": {"label": args.label_b, "path": str(args.candidate_b.resolve()), "sha256": sha_b},
        "reference": {"path": str(args.reference.resolve())},
        "mask": {"path": str(args.mask.resolve()), "resolution_bp": RESOLUTION},
        "n_chromosomes": len(rows),
        "metrics": {key: metric_summary(rows, key) for key, _ in METRICS},
        "support": {
            "mask_common_pairs_total": int(sum(row["A"]["n_mask_common"] for row in rows)),
            "final_common_pairs_total": int(sum(row["A"]["n_final_common"] for row in rows)),
            "extra_dropped_pairs_total": int(sum(row["A"]["dropped_from_mask"] for row in rows)),
            "chromosomes_with_extra_drop": extra_drop,
            "identical_support_across_candidates": True,
        },
        "orientation": {row["chromosome"]: {"A": row["A"]["orientation"], "B": row["B"]["orientation"], "A_tie": bool(row["A"]["tie"]), "B_tie": bool(row["B"]["tie"])} for row in rows},
        "undefined": {
            key: {
                "A_chromosomes": [row["chromosome"] for row in rows if not np.isfinite(row["A"][key])],
                "B_chromosomes": [row["chromosome"] for row in rows if not np.isfinite(row["B"][key])],
            }
            for key, _ in METRICS
        },
        "measurement": {
            "metric": "Pearson r between per-bin distance vectors (1Mb grid, frozen mask)",
            "exchange_rule": "one whole-chromosome A/B swap decision per chromosome (same=matched=max(direct,swapped), cross=min)",
            "tie_tolerance": TIE_TOLERANCE,
            "alignment": "none (distance metrics are invariant to rigid 3D orientation)",
            "sha_a_prefix": sha_a_short,
            "sha_b_prefix": sha_b_short,
        },
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(p3d.jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    png_path = outdir / args.png_name
    plot_panels(rows, (args.label_a, args.label_b), png_path)
    print(f"wrote {png_path}", flush=True)
    print(f"wrote {tsv_path}", flush=True)
    print(f"wrote {summary_path}", flush=True)
    print(f"elapsed_seconds={time.time() - started:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
