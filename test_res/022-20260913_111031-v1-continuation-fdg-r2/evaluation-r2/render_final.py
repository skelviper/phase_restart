#!/usr/bin/env python3
"""仅根据 r2_results.json 重绘冻结的 R2 主图。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


CONDITION_IDS = (
    "softall_seed124101",
    "v1_original_random_joint",
    "v1_continuation",
    "fdg_proposal",
)
SHORT_LABELS = {
    "softall_seed124101": "Softall\n124101",
    "v1_original_random_joint": "V1\n240 steps",
    "v1_continuation": "V1\n480 steps",
    "fdg_proposal": "FDG\n1000 steps",
}
COLORS = ("#2878b5", "#d77a1f", "#3a9d5d", "#ba4a62")
NUMERIC_FILENAMES = ("r2_results.json", "r2_summary.json", "r2_per_chromosome.tsv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_values(values: list[Any]) -> list[float]:
    result = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def metric_limits(data: list[list[float]]) -> tuple[float, float]:
    values = [number for series in data for number in series]
    if not values:
        return 0.0, 1.0
    span = max(max(values) - min(values), 0.05)
    lower = min(0.0, min(values) - 0.08 * span)
    upper = max(0.05, max(values) + 0.12 * span)
    return lower, upper if upper > lower else lower + 0.1


def render(results_dir: Path, output_stem: Path, revision_path: Path) -> dict[str, Any]:
    results_dir = results_dir.resolve()
    output_stem = output_stem.resolve()
    revision_path = revision_path.resolve()
    results_json = results_dir / "r2_results.json"
    summary_json = results_dir / "r2_summary.json"
    table_tsv = results_dir / "r2_per_chromosome.tsv"
    old_png = output_stem.with_suffix(".png")
    old_pdf = output_stem.with_suffix(".pdf")
    numeric_paths = (results_json, summary_json, table_tsv)
    numeric_before = {path.name: sha256_file(path) for path in numeric_paths}
    old_hashes = {"png": sha256_file(old_png), "pdf": sha256_file(old_pdf)}

    payload = json.loads(results_json.read_text(encoding="utf-8"))
    chromosomes = payload["chromosomes"]
    names = [str(item["chromosome"]) for item in chromosomes]
    expected_names = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
    if names != expected_names:
        raise RuntimeError(f"unexpected chromosome order: {names}")
    if any(condition_id not in item["conditions"] for item in chromosomes for condition_id in CONDITION_IDS):
        raise RuntimeError("r2_results.json is missing a main condition")

    rows = [item["conditions"] for item in chromosomes]
    metric_data = {
        metric: [finite_values([row[condition_id].get(metric) for row in rows])
                 for condition_id in CONDITION_IDS]
        for metric in ("similarity", "contrast")
    }
    if any(len(series) != len(chromosomes) for series_list in metric_data.values() for series in series_list):
        raise RuntimeError("main figure contains missing/nonfinite values")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), squeeze=False)
    axes = axes[0]
    positions = list(range(1, len(CONDITION_IDS) + 1))
    labels = [SHORT_LABELS[condition_id] for condition_id in CONDITION_IDS]
    panel_specs = (
        (axes[0], "similarity", "Similarity to reference", "Matched Spearman rho"),
        (axes[1], "contrast", "Copy-specific R2 contrast", "Matched minus crossed rho"),
    )
    for ax, metric, title, y_label in panel_specs:
        data = metric_data[metric]
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.8},
            whiskerprops={"linewidth": 0.65},
            capprops={"linewidth": 0.65},
            boxprops={"linewidth": 0.65},
            zorder=2,
        )
        for index, patch in enumerate(box["boxes"]):
            patch.set_facecolor(COLORS[index])
            patch.set_alpha(0.55)
        for chromosome_index, row in enumerate(rows):
            x_values = [positions[index] for index in range(len(CONDITION_IDS))]
            y_values = [float(row[condition_id][metric]) for condition_id in CONDITION_IDS]
            ax.plot(x_values, y_values, color="#999999", linewidth=0.5, alpha=0.65, zorder=1)
            for condition_index, value in enumerate(y_values):
                ax.scatter(
                    [positions[condition_index]], [value], s=10,
                    color=COLORS[condition_index], edgecolors="white", linewidths=0.25, zorder=3,
                )
        lower, upper = metric_limits(data)
        ax.set_ylim(lower, upper)
        ax.set_xlim(0.5, len(CONDITION_IDS) + 0.5)
        ax.set_xticks(positions, labels=labels)
        ax.set_ylabel(y_label, fontsize=7)
        ax.set_title(title, fontsize=7, pad=4)
        ax.tick_params(axis="both", which="major", labelsize=7, pad=2, length=3, width=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.35, alpha=0.65)
        ax.set_axisbelow(True)

    fig.text(
        0.5, 0.055,
        "20 chromosomes; paired lines; FDG rejected by V1 objective",
        ha="center", va="bottom", fontsize=6,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.23, top=0.81, wspace=0.42)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r2-render-", dir=str(output_stem.parent)) as temporary:
        temporary = Path(temporary)
        temporary_png = temporary / "r2_main.png"
        temporary_pdf = temporary / "r2_main.pdf"
        fig.savefig(temporary_png, dpi=300, format="png")
        fig.savefig(temporary_pdf, dpi=300, format="pdf")
        plt.close(fig)
        os.replace(temporary_png, old_png)
        os.replace(temporary_pdf, old_pdf)

    numeric_after = {path.name: sha256_file(path) for path in numeric_paths}
    if numeric_before != numeric_after:
        raise RuntimeError("numeric JSON/TSV changed during render")
    new_hashes = {"png": sha256_file(old_png), "pdf": sha256_file(old_pdf)}
    revision = {
        "schema_version": "p9016-r2-render-revision-v1",
        "source_results_json": str(results_json),
        "reference_read": False,
        "r2_recomputed": False,
        "condition_ids": list(CONDITION_IDS),
        "short_tick_labels": labels,
        "panel_titles": ["Similarity to reference", "Copy-specific R2 contrast"],
        "y_labels": ["Matched Spearman rho", "Matched minus crossed rho"],
        "footnote": "20 chromosomes; paired lines; FDG rejected by V1 objective",
        "figure_size_inches": [6.0, 3.0],
        "dpi": 300,
        "font_sizes_pt": {"title_axis_tick": 7, "footnote": 6},
        "subplots_adjust": {"left": 0.10, "right": 0.98, "bottom": 0.23, "top": 0.81, "wspace": 0.42},
        "legend": "none",
        "old": {"png_sha256": old_hashes["png"], "pdf_sha256": old_hashes["pdf"]},
        "new": {"png_sha256": new_hashes["png"], "pdf_sha256": new_hashes["pdf"]},
        "numeric_hashes_before": numeric_before,
        "numeric_hashes_after": numeric_after,
        "numeric_json_tsv_unchanged": numeric_before == numeric_after,
    }
    revision_path.write_text(json.dumps(revision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return revision


def main() -> int:
    parser = argparse.ArgumentParser(description="Redraw the frozen P9016 R2 main figure")
    here = Path(__file__).resolve().parent
    parser.add_argument("--results-dir", type=Path, default=here / "results")
    parser.add_argument("--output-stem", type=Path, default=here / "results" / "plots" / "r2_main")
    parser.add_argument("--revision-path", type=Path, default=here / "results" / "render_revision.json")
    args = parser.parse_args()
    revision = render(args.results_dir, args.output_stem, args.revision_path)
    print(json.dumps(revision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
