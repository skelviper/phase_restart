#!/usr/bin/env python3
"""绘制已完成的下一步 real R2 TSV，不打开坐标 payload。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PREP_DIR = Path(__file__).resolve().parent
METRICS = ("matched", "cross", "contrast", "minmargin")
METHODS = ("M0", "M1")
BUDGETS = ("B1", "B2")
SOURCES = ("consensus_joint_base1103", "random_joint_base2207")


class PlotError(RuntimeError):
    """已完成的 R2 结果不能安全绘图时抛出。"""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PlotError("result JSON root is not an object: %s" % path)
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _value(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return float("nan")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _load_arrays(result_path: Path, result: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = result_path.parent
    real_path = root / "r2_real_8endpoint_x20chr.tsv"
    anchor_path = root / "r2_real_historical_036_C0_anchor_x20chr.tsv"
    if not real_path.is_file() or not anchor_path.is_file():
        raise PlotError("real/anchor TSVs are missing beside result JSON")
    real_rows = _read_tsv(real_path)
    anchor_rows = _read_tsv(anchor_path)
    selected = list(result.get("release", {}).get("selected_real_endpoint_ids", []))
    if selected != ["M0-B1-%s" % SOURCES[0], "M0-B2-%s" % SOURCES[0], "M1-B1-%s" % SOURCES[0], "M1-B2-%s" % SOURCES[0]] and len(selected) != 4:
        # Selection 可使用任一 source；只要求冻结的 method/budget 顺序。
        if len(selected) != 4:
            raise PlotError("result lacks four selected real endpoints")
    if len(real_rows) != 160 or len(anchor_rows) != 20:
        raise PlotError("unexpected real/anchor row count: %d/%d" % (len(real_rows), len(anchor_rows)))
    by_key = {(row.get("endpoint_id"), row.get("chromosome")): row for row in real_rows}
    anchor_id = str(result.get("release", {}).get("historical_anchor_id", "historical_036_C0_anchor"))
    anchor_by_chr = {(anchor_id, row.get("chromosome")): row for row in anchor_rows}
    chromosomes = [row.get("chromosome") for row in anchor_rows]
    if len(set(chromosomes)) != 20:
        raise PlotError("anchor TSV does not contain 20 chromosomes")
    arrays: dict[str, np.ndarray] = {}
    selected_labels = ["M0-B1", "M0-B2", "M1-B1", "M1-B2", "036 C0 historical anchor"]
    selected_endpoint_ids = selected + [anchor_id]
    for metric in METRICS:
        array = np.full((5, 20), np.nan, dtype=np.float64)
        for condition_index, endpoint_id in enumerate(selected_endpoint_ids):
            source = anchor_by_chr if condition_index == 4 else by_key
            for chromosome_index, chromosome in enumerate(chromosomes):
                row = source.get((endpoint_id, chromosome))
                if row is None:
                    raise PlotError("missing %s row for %s/%s" % (metric, endpoint_id, chromosome))
                array[condition_index, chromosome_index] = _value(row, metric)
        arrays["main_" + metric] = array
    for source_id in SOURCES:
        for metric in METRICS:
            array = np.full((4, 20), np.nan, dtype=np.float64)
            for method_index, method in enumerate(METHODS):
                for budget_index, budget in enumerate(BUDGETS):
                    endpoint_id = "%s-%s-%s" % (method, budget, source_id)
                    condition_index = method_index * 2 + budget_index
                    for chromosome_index, chromosome in enumerate(chromosomes):
                        row = by_key.get((endpoint_id, chromosome))
                        if row is None:
                            raise PlotError("missing source row for %s/%s" % (metric, endpoint_id))
                        array[condition_index, chromosome_index] = _value(row, metric)
            arrays["%s_%s" % (source_id, metric)] = array
    metadata = {
        "schema_version": "p9016-next-step-r2-plot-arrays-v1",
        "metrics": list(METRICS),
        "main_condition_order": selected_labels,
        "main_endpoint_order": selected_endpoint_ids,
        "source_condition_order": ["M0-B1", "M0-B2", "M1-B1", "M1-B2"],
        "source_ids": list(SOURCES),
        "n_chromosomes": 20,
        "main_array_shape": [5, 20],
        "source_array_shape": [4, 20],
        "base_panel_inches": 3.0,
        "figure_layout": "2x2",
        "dpi": 300,
        "font_size_pt": 7,
        "historical_anchor_style": "gray",
        "one_cell_linked_measurements_not_biological_replicates": True,
        "no_confidence_intervals": True,
        "no_p_values": True,
    }
    return arrays, metadata


def _render_one(arrays: Mapping[str, np.ndarray], metric: str, condition_order: Sequence[str], key: str, output: Path, title: str, anchor: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    array = np.asarray(arrays[key], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
    values = [array[index, np.isfinite(array[index])] for index in range(array.shape[0])]
    box = ax.boxplot(values, positions=np.arange(1, len(values) + 1), widths=0.58, patch_artist=True, showfliers=False, medianprops={"color": "#111111", "linewidth": 0.8}, whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7})
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    if anchor:
        colors.append("#8a8a8a")
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.65)
    rng = np.random.default_rng(7301 + hash(metric) % 1000)
    for index in range(array.shape[0]):
        finite = np.isfinite(array[index])
        jitter = rng.uniform(-0.12, 0.12, int(finite.sum()))
        ax.scatter(np.full(int(finite.sum()), index + 1, dtype=np.float64) + jitter, array[index, finite], s=5.0, color="#222222" if not (anchor and index == array.shape[0] - 1) else "#555555", alpha=0.62, linewidths=0.0, zorder=3)
    ax.axhline(0.0, color="#777777", linewidth=0.55, linestyle="--", zorder=0)
    ax.set_xticks(np.arange(1, len(condition_order) + 1))
    ax.set_xticklabels([str(item).replace(" historical anchor", "\n036 C0 anchor") for item in condition_order], fontsize=7)
    ax.tick_params(axis="y", labelsize=7, width=0.55, length=2.5)
    ax.tick_params(axis="x", width=0.55, length=2.5)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.set_ylabel(metric, fontsize=7)
    ax.set_title(title, fontsize=7, pad=4)
    ax.grid(axis="y", color="#dddddd", linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_plots(results_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    results_path = Path(results_path).resolve()
    output_dir = Path(output_dir).resolve()
    result = _read_json(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays, metadata = _load_arrays(results_path, result)
    np.savez_compressed(output_dir / "plot_arrays.npz", **arrays)
    (output_dir / "plot_arrays_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_endpoint_ids = list(result["release"]["selected_real_endpoint_ids"])
    main_labels = ["M0-B1", "M0-B2", "M1-B1", "M1-B2", "036 C0 historical anchor"]
    for metric in METRICS:
        _render_one(arrays, metric, main_labels, "main_" + metric, output_dir / ("real_r2_main_" + metric), "P9016 real R2: " + metric, anchor=True)
    # 四个指标 panel 也会作为一个紧凑的 2x2 figure 输出，以用于要求的报告。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    def four_panel(keys: Sequence[str], labels: Sequence[str], prefix: str, title: str, anchor: bool) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(6.0, 6.0), constrained_layout=True)
        colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00"] + (["#8a8a8a"] if anchor else [])
        rng = np.random.default_rng(7311)
        for axis, metric, key in zip(axes.flat, METRICS, keys):
            array = np.asarray(arrays[key], dtype=np.float64)
            values = [array[index, np.isfinite(array[index])] for index in range(array.shape[0])]
            box = axis.boxplot(values, positions=np.arange(1, len(values) + 1), widths=0.58, patch_artist=True, showfliers=False, medianprops={"color": "#111111", "linewidth": 0.75}, whiskerprops={"linewidth": 0.65}, capprops={"linewidth": 0.65})
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.78)
                patch.set_edgecolor("#333333")
                patch.set_linewidth(0.6)
            for index in range(array.shape[0]):
                finite = np.isfinite(array[index])
                jitter = rng.uniform(-0.11, 0.11, int(finite.sum()))
                axis.scatter(np.full(int(finite.sum()), index + 1, dtype=np.float64) + jitter, array[index, finite], s=4.0, color="#222222", alpha=0.62, linewidths=0.0, zorder=3)
            axis.axhline(0.0, color="#777777", linewidth=0.5, linestyle="--")
            axis.set_xticks(np.arange(1, len(labels) + 1))
            axis.set_xticklabels([str(item).replace(" historical anchor", "\n036 C0") for item in labels], fontsize=7)
            axis.tick_params(axis="both", labelsize=7, width=0.5, length=2.0)
            axis.yaxis.set_major_locator(ticker.MaxNLocator(4))
            axis.set_ylabel(metric, fontsize=7)
            axis.grid(axis="y", color="#dddddd", linewidth=0.4)
            for spine in axis.spines.values():
                spine.set_linewidth(0.5)
        fig.suptitle(title, fontsize=7)
        fig.savefig(output_dir / (prefix + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / (prefix + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    four_panel(["main_" + metric for metric in METRICS], main_labels, "real_r2_main_four_panel", "P9016 real R2 selected conditions", True)
    for source in SOURCES:
        four_panel([source + "_" + metric for metric in METRICS], ["M0-B1", "M0-B2", "M1-B1", "M1-B2"], "real_r2_%s_four_panel" % source, "P9016 real R2 %s" % source, False)
    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            files[path.name] = _sha256(path)
    return {"status": "plots_complete", "output_dir": str(output_dir), "files": files, "arrays": metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        result = render_plots(args.results, args.output_dir)
    except (PlotError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": result["status"], "output_dir": result["output_dir"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
