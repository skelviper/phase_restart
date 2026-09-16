#!/usr/bin/env python3
"""将锁定的主 R2 配对效应渲染为紧凑的 mean/CI 图。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = (
    ("similarity", "Similarity (matched)"),
    ("cross", "Cross"),
    ("contrast", "Contrast"),
    ("minmargin", "Min margin"),
)
COMPARISONS = (
    ("C1", "C0"),
    ("C2-map", "C0"),
    ("C2-free", "C0"),
    ("C2-free", "C2-map"),
    ("C3", "C0"),
)
SHORT_VARIANTS = {"C2-map": "C2m", "C2-free": "C2f"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_primary(input_path: Path) -> list[dict[str, object]]:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    wanted_metrics = {metric for metric, _ in METRICS}
    wanted_comparisons = set(COMPARISONS)
    selected: list[dict[str, object]] = []
    for row in rows:
        if row.get("level") != "primary" or row.get("metric") not in wanted_metrics:
            continue
        comparison = (row.get("left_variant", ""), row.get("right_variant", ""))
        if comparison not in wanted_comparisons:
            continue
        if row.get("status") != "ok":
            raise ValueError(f"primary row is not ok: {row.get('label')}")
        if row.get("planned_seed_count") != "3" or row.get("valid_seed_count") != "3":
            raise ValueError(f"primary row is not 3/3 seeds: {row.get('label')}")
        if row.get("n_chromosomes_valid") != "20":
            raise ValueError(f"primary row is not 20 chromosomes: {row.get('label')}")
        try:
            parsed = {
                "left_variant": row["left_variant"],
                "right_variant": row["right_variant"],
                "metric": row["metric"],
                "mean_delta": float(row["mean_delta"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric primary row: {row.get('label')}") from exc
        if not all(math.isfinite(parsed[key]) for key in ("mean_delta", "ci95_low", "ci95_high")):
            raise ValueError(f"non-finite primary row: {row.get('label')}")
        selected.append(parsed)

    expected_keys = {(left, right, metric) for left, right in COMPARISONS for metric, _ in METRICS}
    actual_keys = {(row["left_variant"], row["right_variant"], row["metric"]) for row in selected}
    if len(selected) != 20 or actual_keys != expected_keys:
        raise ValueError(f"expected exactly 20 primary rows, got {len(selected)}")
    return selected


def short_variant(variant: str) -> str:
    return SHORT_VARIANTS.get(variant, variant)


def render(rows: list[dict[str, object]], output_dir: Path) -> None:
    by_key = {
        (row["left_variant"], row["right_variant"], row["metric"]): row for row in rows
    }
    labels = [f"{short_variant(left)}-{short_variant(right)}" for left, right in COMPARISONS]
    x_values = list(range(len(labels)))

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=300)
    axes_flat = axes.ravel()
    point_color = "#285b73"
    error_color = "#285b73"
    zero_color = "#8a8a8a"

    for axis, (metric, title) in zip(axes_flat, METRICS):
        metric_rows = [by_key[(left, right, metric)] for left, right in COMPARISONS]
        means = [row["mean_delta"] for row in metric_rows]
        lows = [row["ci95_low"] for row in metric_rows]
        highs = [row["ci95_high"] for row in metric_rows]
        lower_errors = [mean - low for mean, low in zip(means, lows)]
        upper_errors = [high - mean for mean, high in zip(means, highs)]

        axis.axhline(0.0, color=zero_color, linewidth=0.7, zorder=0)
        axis.errorbar(
            x_values,
            means,
            yerr=[lower_errors, upper_errors],
            fmt="o",
            color=point_color,
            ecolor=error_color,
            markersize=3.0,
            linewidth=0.8,
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.7,
            zorder=2,
        )
        axis.set_title(title, pad=4)
        axis.set_xlabel("left variant - right variant", labelpad=3)
        axis.set_xticks(x_values)
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.tick_params(axis="both", which="major", labelsize=7, pad=2)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.5, alpha=0.8)
        axis.set_axisbelow(True)
        low = min(lows + [0.0])
        high = max(highs + [0.0])
        padding = max((high - low) * 0.12, 0.004)
        axis.set_ylim(low - padding, high + padding)

    figure.suptitle("Primary paired effects: mean and 95% CI", fontsize=7, y=0.985)
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.92, wspace=0.34, hspace=0.42)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "r2_primary_mean_ci.png", dpi=300, facecolor="white")
    figure.savefig(output_dir / "r2_primary_mean_ci.pdf", facecolor="white")
    plt.close(figure)


def write_input_record(input_path: Path, output_dir: Path, rows: list[dict[str, object]]) -> None:
    record = {
        "schema": "post020-r2-summary-figure-input-v1",
        "source_tsv": str(input_path),
        "source_sha256": sha256_file(input_path),
        "selection": {
            "level": "primary",
            "metrics": [metric for metric, _ in METRICS],
            "comparisons_left_minus_right": [
                {"left_variant": left, "right_variant": right} for left, right in COMPARISONS
            ],
            "selected_row_count": len(rows),
            "seed_aggregation": "read existing three-seed primary mean_delta and existing ci95 bounds",
            "bootstrap": "existing seed=9301, n_boot=10000, n_chromosomes=20 matrix; no recalculation",
        },
        "figure": {
            "size_inches": [6, 6],
            "base_panel_inches": [3, 3],
            "dpi": 300,
            "formats": ["png", "pdf"],
            "zero_reference_line": True,
        },
    }
    (output_dir / "input_hashes.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    rows = read_primary(input_path)
    write_input_record(input_path, output_dir, rows)
    render(rows, output_dir)
    print(f"selected_primary_rows={len(rows)}")
    print(f"input_sha256={sha256_file(input_path)}")
    print(f"png={output_dir / 'r2_primary_mean_ci.png'}")
    print(f"pdf={output_dir / 'r2_primary_mean_ci.pdf'}")


if __name__ == "__main__":
    main()
