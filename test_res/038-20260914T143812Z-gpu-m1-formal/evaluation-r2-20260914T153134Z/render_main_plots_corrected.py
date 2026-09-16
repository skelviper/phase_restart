#!/usr/bin/env python3
"""仅渲染修正五张主 real-R2 图。

读取冻结的 plot arrays，只覆盖主 metric panels 和主 four-panel figure。按 source 分层的补充图有意保持不变。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ("matched", "cross", "contrast", "minmargin")
MAIN_LABELS = ("M0-B1", "M0-B2", "M1-B1", "M1-B2", "036 C0\nanchor")


def _render_one(array: np.ndarray, metric: str, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    array = np.asarray(array, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
    values = [array[index, np.isfinite(array[index])] for index in range(array.shape[0])]
    box = ax.boxplot(
        values,
        positions=np.arange(1, len(values) + 1),
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 0.8},
        whiskerprops={"linewidth": 0.7},
        capprops={"linewidth": 0.7},
    )
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#8a8a8a"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.65)
    rng = np.random.default_rng(7301 + METRICS.index(metric))
    for index in range(array.shape[0]):
        finite = np.isfinite(array[index])
        jitter = rng.uniform(-0.12, 0.12, int(finite.sum()))
        ax.scatter(
            np.full(int(finite.sum()), index + 1, dtype=np.float64) + jitter,
            array[index, finite],
            s=5.0,
            color="#555555" if index == array.shape[0] - 1 else "#222222",
            alpha=0.62,
            linewidths=0.0,
            zorder=3,
        )
    ax.axhline(0.0, color="#777777", linewidth=0.55, linestyle="--", zorder=0)
    ax.set_xticks(np.arange(1, len(MAIN_LABELS) + 1))
    ax.set_xticklabels(MAIN_LABELS, fontsize=7)
    ax.tick_params(axis="y", labelsize=7, width=0.55, length=2.5)
    ax.tick_params(axis="x", width=0.55, length=2.5)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.set_ylabel(metric, fontsize=7)
    ax.set_title("P9016 real R2: " + metric, fontsize=7, pad=4)
    ax.grid(axis="y", color="#dddddd", linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _render_four(arrays: dict[str, np.ndarray], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 6.0), constrained_layout=True)
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#8a8a8a"]
    rng = np.random.default_rng(7311)
    for axis, metric in zip(axes.flat, METRICS):
        array = np.asarray(arrays["main_" + metric], dtype=np.float64)
        values = [array[index, np.isfinite(array[index])] for index in range(array.shape[0])]
        box = axis.boxplot(
            values,
            positions=np.arange(1, len(values) + 1),
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 0.75},
            whiskerprops={"linewidth": 0.65},
            capprops={"linewidth": 0.65},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.6)
        for index in range(array.shape[0]):
            finite = np.isfinite(array[index])
            jitter = rng.uniform(-0.11, 0.11, int(finite.sum()))
            axis.scatter(
                np.full(int(finite.sum()), index + 1, dtype=np.float64) + jitter,
                array[index, finite],
                s=4.0,
                color="#222222",
                alpha=0.62,
                linewidths=0.0,
                zorder=3,
            )
        axis.axhline(0.0, color="#777777", linewidth=0.5, linestyle="--")
        axis.set_xticks(np.arange(1, len(MAIN_LABELS) + 1))
        axis.set_xticklabels(MAIN_LABELS, fontsize=7)
        axis.tick_params(axis="both", labelsize=7, width=0.5, length=2.0)
        axis.yaxis.set_major_locator(ticker.MaxNLocator(4))
        axis.set_ylabel(metric, fontsize=7)
        axis.grid(axis="y", color="#dddddd", linewidth=0.4)
        for spine in axis.spines.values():
            spine.set_linewidth(0.5)
    fig.suptitle("P9016 real R2 selected conditions", fontsize=7)
    fig.savefig(output_dir / "real_r2_main_four_panel.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "real_r2_main_four_panel.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    arrays_path = Path(args.arrays).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(arrays_path) as archive:
        arrays = {"main_" + metric: np.asarray(archive["main_" + metric], dtype=np.float64) for metric in METRICS}
    if any(array.shape != (5, 20) for array in arrays.values()):
        raise ValueError("frozen main arrays must all have shape (5,20)")
    for metric in METRICS:
        _render_one(arrays["main_" + metric], metric, output_dir / ("real_r2_main_" + metric))
    _render_four(arrays, output_dir)
    print(json.dumps({"status": "render_correction_complete", "output_dir": str(output_dir), "written_main_files": 10}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
