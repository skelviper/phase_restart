"""渲染 post-evaluation P3 calibration 图形，不改变 metrics。

renderer 只读取 metrics 和 supplemental JSON。它不打开 truth files、coordinates 或 model inputs，也绝不写入 metrics.json。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "test_res" / "019-20260913_064842-v1-calibration-recovery"
CONDITIONS = ("A", "B", "C", "D")
COLORS = {"A": "#2f6690", "B": "#3a7d44", "C": "#b5651d", "D": "#6c567b"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with open(path, "wt") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    return plt


def _render_main(plt, run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "plots" / "r2_r3_summary.png"
    previous_hash = _sha256(path) if path.is_file() else None
    x = list(range(len(CONDITIONS)))
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), dpi=300)

    r2_values = [metrics["conditions"][condition]["r2"]["mean_contrast"]
                 for condition in CONDITIONS]
    r2_finite = [value is not None for value in r2_values]
    axes[0].bar([index for index, ok in zip(x, r2_finite, strict=True) if ok],
                [value for value in r2_values if value is not None],
                color=[COLORS[condition] for condition, ok in zip(CONDITIONS, r2_finite, strict=True) if ok],
                width=0.68)
    axes[0].axhline(0.0, color="black", linewidth=0.6)
    axes[0].set_xticks(x, CONDITIONS)
    axes[0].set_xlim(-0.5, 3.5)
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_xlabel("condition")
    axes[0].set_ylabel("contrast")
    axes[0].set_title("R2 geometry best-swap contrast\n(non-negative by construction)")
    r2_top = max([value for value in r2_values if value is not None] + [0.1])
    axes[0].text(3.0, 0.08 * r2_top, "D: tie", ha="center", va="bottom", fontsize=7)

    r3_values = [metrics["conditions"][condition]["r3"]["mean_frac_consistent"]
                 for condition in CONDITIONS]
    r3_finite = [value is not None for value in r3_values]
    axes[1].bar([index for index, ok in zip(x, r3_finite, strict=True) if ok],
                [value for value in r3_values if value is not None],
                color=[COLORS[condition] for condition, ok in zip(CONDITIONS, r3_finite, strict=True) if ok],
                width=0.68)
    axes[1].set_xticks(x, CONDITIONS)
    axes[1].set_xlim(-0.5, 3.5)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("condition")
    axes[1].set_ylabel("fraction")
    axes[1].set_title("R3 fragment consistency")
    axes[1].text(3.0, 0.08, "D: n/a", ha="center", va="bottom", fontsize=7)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.80, wspace=0.38)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return {
        "path": str(path.relative_to(run_dir)),
        "figure_size_inches": [6.0, 3.0],
        "base_panel_size_inches": [3.0, 3.0],
        "dpi": 300,
        "font_size_pt": 7,
        "pixel_size": [1800, 900],
        "previous_sha256": previous_hash,
        "sha256": _sha256(path),
        "D_R2_annotation": "tie",
        "D_R3_annotation": "n/a",
    }


def _render_supplemental(plt, run_dir: Path, supplemental: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "plots" / "supplemental_controls_paired.png"
    previous_hash = _sha256(path) if path.is_file() else None
    x = list(range(len(CONDITIONS)))
    pair_name = "candidate_vs_random_field"
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), dpi=300)
    for axis, metric_key, title, ylabel in (
        (axes[0], "r2", "Paired R2 contrast: candidate - fixed random-field", "delta geometry contrast"),
        (axes[1], "r3", "Paired R3: candidate - fixed random-field", "delta common-mask fraction"),
    ):
        values = []
        errors = []
        valid = []
        for condition in CONDITIONS:
            metric = supplemental["conditions"][condition]["paired"][pair_name][metric_key]
            mean = metric["mean_delta"]
            if mean is None:
                valid.append(False)
                values.append(0.0)
                errors.append((0.0, 0.0))
            else:
                valid.append(True)
                values.append(mean)
                errors.append((mean - metric["ci_low"], metric["ci_high"] - mean))
        for index, condition in enumerate(CONDITIONS):
            if valid[index]:
                axis.errorbar(index, values[index], yerr=np.asarray(errors[index])[:, None],
                              fmt="none", ecolor=COLORS[condition], capsize=2, linewidth=0.8)
        axis.bar([index for index, ok in zip(x, valid, strict=True) if ok],
                 [value for value, ok in zip(values, valid, strict=True) if ok],
                 color=[COLORS[condition] for condition, ok in zip(CONDITIONS, valid, strict=True) if ok],
                 width=0.68, alpha=0.9)
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_xticks(x, CONDITIONS)
        axis.set_xlim(-0.5, 3.5)
        axis.set_xlabel("condition")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    axes[1].text(3.0, 0.08, "D: n/a", ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.80, wspace=0.38)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return {
        "path": str(path.relative_to(run_dir)),
        "figure_size_inches": [6.0, 3.0],
        "base_panel_size_inches": [3.0, 3.0],
        "dpi": 300,
        "font_size_pt": 7,
        "pixel_size": [1800, 900],
        "previous_sha256": previous_hash,
        "sha256": _sha256(path),
        "D_R3_annotation": "n/a",
        "error_bars": "technical chromosome-bootstrap percentile 2.5/97.5 CI",
    }


def render(run_dir: str | Path = DEFAULT_RUN) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    metrics_path = run_dir / "metrics.json"
    supplemental_path = run_dir / "work" / "supplemental_controls.json"
    metrics_hash_before = _sha256(metrics_path)
    supplemental_hash_before = _sha256(supplemental_path)
    metrics = json.loads(metrics_path.read_text())
    supplemental = json.loads(supplemental_path.read_text())
    plt = _configure_matplotlib()
    figures = {
        "main_r2_r3": _render_main(plt, run_dir, metrics),
        "supplemental_paired_controls": _render_supplemental(plt, run_dir, supplemental),
    }
    metrics_hash_after = _sha256(metrics_path)
    supplemental_hash_after = _sha256(supplemental_path)
    provenance = {
        "schema": "v1-p3-rendering-provenance-v2",
        "renderer": "pr/v1_calibration_reporting.py",
        "renderer_sha256": _sha256(Path(__file__).resolve()),
        "run_id": run_dir.name,
        "truth_artifact_opened": False,
        "metrics_sha256_before": metrics_hash_before,
        "metrics_sha256_after": metrics_hash_after,
        "metrics_unchanged": metrics_hash_before == metrics_hash_after,
        "supplemental_controls_sha256_before": supplemental_hash_before,
        "supplemental_controls_sha256_after": supplemental_hash_after,
        "supplemental_controls_unchanged": supplemental_hash_before == supplemental_hash_after,
        "font_policy": "all visible text 7 pt",
        "main_figure_policy": "two 3x3 inch base panels in one 6x3 inch figure at 300 dpi",
        "D_R3_policy": "explicit n/a annotation and no zero-height bar",
        "figures": figures,
    }
    _write_json(run_dir / "plots" / "rendering_provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Render P3 calibration figures")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    args = parser.parse_args()
    print(json.dumps(render(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
