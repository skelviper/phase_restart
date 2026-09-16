"""050 轮图件：最多两张，英文标注、7 pt、300 dpi、3 英寸基础面板。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
PLOTS = RUN_DIR / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                     "axes.linewidth": 0.6, "figure.dpi": 300})

SHORT = {
    "046-base-G-random": "046 base\nG-random",
    "046-work-baseline-G-full-J": "046 work\nbaseline",
    "049-A-ms-random": "049 A\nms-random",
    "049-B-raw-consensus": "049 B\nraw-cons",
    "049-C-ms-random": "049 C\nms-random",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def figure_p0() -> Path | None:
    summary_path = RUN_DIR / "p0" / "formal5" / "r1_r3_summary.json"
    if not summary_path.is_file():
        return None
    summary = load(summary_path)
    order = [entry["id"] for entry in json.loads(
        (RUN_DIR / "p0" / "manifest_formal5.json").read_text(encoding="utf-8"))["candidates"]]
    accuracy = [summary["candidates"][cid]["R1_macro_mean_all20"] for cid in order]
    oracle = summary["candidates"][order[0]]["R1_oracle_fit_ceiling_macro_mean_all20"]
    ceiling = summary["candidates"][order[0]]["R1_reference_ceiling_macro_mean_all20"]
    r3 = [summary["candidates"][cid]["R3_frac_consistent_macro_mean_all20"] for cid in order]
    walls = [summary["candidates"][cid]["R3_n_walls_macro_mean_all20"] for cid in order]

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6))
    x = np.arange(len(order))
    axes[0].axhline(ceiling, color="#999999", lw=0.8, ls="--", label="reference ceiling")
    axes[0].axhline(oracle, color="#666666", lw=0.8, ls=":", label="S0 oracle-fit ceiling")
    axes[0].bar(x, accuracy, width=0.62, color="#4c72b0")
    for xi, value in zip(x, accuracy):
        axes[0].text(xi, value + 0.004, "%.3f" % value, ha="center", va="bottom", fontsize=6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SHORT.get(cid, cid) for cid in order])
    axes[0].set_ylabel("R1 accuracy (20-chr macro mean)")
    axes[0].set_ylim(0.45, 0.83)
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].set_title("Per-contact accuracy, common 200,898-record support")

    axes[1].bar(x - 0.17, r3, width=0.34, color="#55a868", label="R3 frac_consistent")
    axes[1].bar(x + 0.17, np.asarray(walls) / 4.0, width=0.34, color="#c44e52",
                label="R3 n_walls / 4")
    for xi, value in zip(x, r3):
        axes[1].text(xi - 0.17, value + 0.005, "%.3f" % value, ha="center", va="bottom", fontsize=6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SHORT.get(cid, cid) for cid in order])
    axes[1].set_ylabel("R3 readout (macro mean)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="lower right", frameon=False)
    axes[1].set_title("20 Mb fragment consistency")
    fig.tight_layout()
    path = PLOTS / "p0_r1_r3_endpoints.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def figure_p1() -> Path | None:
    summary_path = RUN_DIR / "evaluation" / "results" / "p1_evaluation_summary.json"
    bootstrap_path = RUN_DIR / "evaluation" / "results" / "p1_paired_bootstrap.json"
    if not summary_path.is_file() or not bootstrap_path.is_file():
        return None
    summary = load(summary_path)
    rows = load(bootstrap_path)["rows"]
    wanted = [("pearson:matched", "R2 matched (Pearson)"), ("pearson:contrast", "R2 contrast (Pearson)")]
    seen, order = set(), []
    for row in rows:
        if row["label"] not in seen:
            seen.add(row["label"])
            order.append(row["label"])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for axis, (metric, title) in zip(axes, wanted):
        means, lows, highs = [], [], []
        for label in order:
            value = next(r for r in rows if r["label"] == label)["metrics"][metric]
            means.append(value["mean"] if value["mean"] is not None else np.nan)
            lows.append(value["ci95"][0] if value["ci95"][0] is not None else np.nan)
            highs.append(value["ci95"][1] if value["ci95"][1] is not None else np.nan)
        y = np.arange(len(order))[::-1]
        means = np.asarray(means)
        axis.axvline(0.0, color="#999999", lw=0.8)
        axis.errorbar(means, y, xerr=[means - np.asarray(lows), np.asarray(highs) - means],
                      fmt="o", ms=3.0, lw=0.8, capsize=1.6, color="#4c72b0")
        axis.set_yticks(y)
        axis.set_yticklabels(order, fontsize=5.5)
        axis.set_xlabel("delta (left minus right)")
        axis.set_title(title)
    fig.tight_layout()
    path = PLOTS / "p1_paired_deltas.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> int:
    made = [p for p in (figure_p0(), figure_p1()) if p is not None]
    print(json.dumps({"plots": [str(p) for p in made]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
