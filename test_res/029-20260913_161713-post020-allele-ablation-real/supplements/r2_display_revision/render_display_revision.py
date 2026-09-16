#!/usr/bin/env python3
"""锁定的 real R2 输出的独立、仅供显示的渲染器。

本脚本只读取用于作图的逐染色体锁定行、R2 summary 和 release metadata。它不打开坐标、phase data 或 reference，并在不触碰原始评估的情况下写出十个可读性补充结果。
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
RUN_ROOT = SCRIPT_PATH.parent.parent.parent
PROJECT_ROOT = RUN_ROOT.parent.parent
EVAL_ROOT = RUN_ROOT / "evaluation-r2"
OUTPUT_ROOT = SCRIPT_PATH.parent
CONTROLLER_ROOT = PROJECT_ROOT / "test_res" / "POST020_REAL_CONTROLLER"
PER_CHROMOSOME_PATH = EVAL_ROOT / "r2_per_chromosome.json"
SUMMARY_PATH = EVAL_ROOT / "r2_summary.json"
RELEASE_PATH = RUN_ROOT / "release_manifest_r2.json"
DELIVERY_RECEIPT_PATH = CONTROLLER_ROOT / "r2_delivery_receipt.json"
SOURCE_PATH = PROJECT_ROOT / "pr" / "allele_r2.py"
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "POST020_ALLELE_ABLATION_PROTOCOL.json"

EXPECTED_SOURCE_SHA256 = "56dacfe9a3edc13db4804fa7d5ce7c40b9401a1d1dd62ddbaa66d0d6ad5b1672"
EXPECTED_PROTOCOL_SHA256 = "b281037946775ee4271dbf33dd7e4b17dba5ddf4f9bd49eaf618f7c499c5361f"
EXPECTED_REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
VARIANT_ORDER = ("C0", "C1", "C2-map", "C2-free", "C3")
HISTORICAL_LABEL_ORDER = ("Softall", "020", "022", "FDG", "Random", "Cons.")
METRICS = ("similarity", "contrast", "cross", "margin_mat", "margin_pat")
NEW_COLORS = OrderedDict((
    ("C0", "#1b9e77"),
    ("C1", "#d95f02"),
    ("C2-map", "#7570b3"),
    ("C2-free", "#e7298a"),
    ("C3", "#66a61e"),
))
HISTORICAL_COLORS = OrderedDict((
    ("Softall", "#4c78a8"),
    ("020", "#f58518"),
    ("022", "#54a24b"),
    ("FDG", "#e45756"),
    ("Random", "#72b7b2"),
    ("Cons.", "#b279a2"),
))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError("missing file: %s" % path)
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(_file_record(path))
    return records


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError("temporary output already exists: %s" % temporary)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _historical_label(display_name: str) -> str:
    text = str(display_name)
    if text.startswith("Softall"):
        return "Softall"
    if text.startswith("020"):
        return "020"
    if text.startswith("022 FDG") or "FDG" in text:
        return "FDG"
    if text.startswith("022"):
        return "022"
    if text.startswith("Random"):
        return "Random"
    if text.startswith("Consensus"):
        return "Cons."
    raise RuntimeError("unmapped historical display name: %s" % text)


def _validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str], list[str], dict[str, str], dict[str, str], dict[str, dict[str, dict[str, Any]]]]:
    per_payload = _read_json(PER_CHROMOSOME_PATH)
    summary = _read_json(SUMMARY_PATH)
    release = _read_json(RELEASE_PATH)
    rows = per_payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 420:
        raise RuntimeError("locked per-chromosome rows must contain 420 rows")
    if release.get("status") not in ("locked_for_evaluation", "released"):
        raise RuntimeError("release metadata is not locked")
    if release.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("release protocol SHA mismatch")
    conditions = {str(item["id"]): item for item in release.get("conditions", [])}
    representatives = release.get("within_variant_representatives")
    if not isinstance(representatives, Mapping):
        raise RuntimeError("release representative mapping is missing")
    new_ids = []
    new_labels = {}
    for variant in VARIANT_ORDER:
        condition_id = str(representatives.get(variant, ""))
        item = conditions.get(condition_id)
        if item is None or item.get("role") != "new_variant":
            raise RuntimeError("representative is not a released new variant: %s" % variant)
        if item.get("bundle_id") != "bundle2":
            raise RuntimeError("representative bundle is not bundle2: %s" % variant)
        new_ids.append(condition_id)
        new_labels[condition_id] = variant
    historical_by_label: dict[str, str] = {}
    for item in conditions.values():
        if item.get("role") != "historical_control":
            continue
        label = _historical_label(str(item.get("display_name", item["id"])))
        if label in historical_by_label:
            raise RuntimeError("duplicate historical short label: %s" % label)
        historical_by_label[label] = str(item["id"])
    if tuple(historical_by_label) != HISTORICAL_LABEL_ORDER:
        if set(historical_by_label) != set(HISTORICAL_LABEL_ORDER):
            raise RuntimeError("historical release mapping does not match six frozen controls")
        historical_by_label = {label: historical_by_label[label] for label in HISTORICAL_LABEL_ORDER}
    historical_ids = [historical_by_label[label] for label in HISTORICAL_LABEL_ORDER]
    all_ids = new_ids + historical_ids
    summary_conditions = summary.get("condition_summary", {})
    if not isinstance(summary_conditions, Mapping):
        raise RuntimeError("R2 summary condition_summary is missing")
    for condition_id in all_ids:
        if condition_id not in summary_conditions:
            raise RuntimeError("summary is missing released condition: %s" % condition_id)
        if summary_conditions[condition_id].get("n_chromosomes_expected") != 20:
            raise RuntimeError("summary chromosome count mismatch: %s" % condition_id)
    grid = release.get("grid", {}).get("chromosomes", [])
    chromosomes = [str(item["name"]) for item in grid]
    if len(chromosomes) != 20 or len(set(chromosomes)) != 20:
        raise RuntimeError("release grid must contain 20 unique chromosomes")
    by_condition: dict[str, dict[str, dict[str, Any]]] = {condition_id: {} for condition_id in all_ids}
    for row in rows:
        condition_id = str(row.get("condition_id"))
        chromosome = str(row.get("chromosome"))
        if condition_id not in by_condition:
            continue
        if chromosome in by_condition[condition_id]:
            raise RuntimeError("duplicate condition/chromosome row: %s/%s" % (condition_id, chromosome))
        by_condition[condition_id][chromosome] = row
    for condition_id in all_ids:
        if set(by_condition[condition_id]) != set(chromosomes):
            raise RuntimeError("condition does not cover the release chromosome grid: %s" % condition_id)
    if _sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("authoritative protocol SHA mismatch")
    if _sha256(SOURCE_PATH) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("evaluator source SHA changed before display render")
    if not DELIVERY_RECEIPT_PATH.is_file():
        raise RuntimeError("47-lock delivery receipt is missing")
    return (per_payload, summary, release, new_ids, historical_ids,
            new_labels, historical_by_label, by_condition)


def _configure_matplotlib() -> Any:
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
        "figure.titlesize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def _metric_values(by_condition: Mapping[str, Mapping[str, Mapping[str, Any]]],
                   condition_id: str, chromosomes: Iterable[str], metric: str) -> np.ndarray:
    return np.asarray([
        float(by_condition[condition_id][chromosome][metric])
        if _finite(by_condition[condition_id][chromosome].get(metric)) else np.nan
        for chromosome in chromosomes
    ], dtype=float)


def _axis_limits(arrays: Iterable[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays if np.isfinite(array).any()]) \
        if any(np.isfinite(array).any() for array in arrays) else np.empty(0, dtype=float)
    if not len(finite):
        return 0.0, 1.0
    span = max(float(np.ptp(finite)), 0.05)
    lower = min(0.0, float(finite.min()) - 0.12 * span)
    upper = max(0.05, float(finite.max()) + 0.18 * span)
    if upper <= lower:
        upper = lower + 0.1
    return lower, upper


def _plot_metric(ax: Any, condition_ids: list[str], labels: list[str], chromosomes: list[str],
                 by_condition: Mapping[str, Mapping[str, Mapping[str, Any]]], metric: str,
                 colors: Mapping[str, str], label_to_condition: Mapping[str, str]) -> dict[str, Any]:
    positions = np.arange(1, len(condition_ids) + 1, dtype=float)
    arrays = {condition_id: _metric_values(by_condition, condition_id, chromosomes, metric)
              for condition_id in condition_ids}
    data = [arrays[condition_id][np.isfinite(arrays[condition_id])] for condition_id in condition_ids]
    present = [(index, values) for index, values in enumerate(data) if len(values)]
    if present:
        box = ax.boxplot(
            [values for _index, values in present],
            positions=[positions[index] for index, _values in present],
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.8},
            whiskerprops={"linewidth": 0.65},
            capprops={"linewidth": 0.65},
            boxprops={"linewidth": 0.65},
        )
        for patch, (index, _values) in zip(box["boxes"], present):
            patch.set_facecolor(colors[labels[index]])
            patch.set_alpha(0.55)
    for chromosome_index, chromosome in enumerate(chromosomes):
        x_values = []
        y_values = []
        for condition_index, condition_id in enumerate(condition_ids):
            value = arrays[condition_id][chromosome_index]
            if np.isfinite(value):
                x_values.append(positions[condition_index])
                y_values.append(float(value))
        if len(x_values) >= 2:
            ax.plot(x_values, y_values, color="#999999", linewidth=0.5, alpha=0.58, zorder=1)
        for condition_index, condition_id in enumerate(condition_ids):
            value = arrays[condition_id][chromosome_index]
            if np.isfinite(value):
                ax.scatter(
                    [positions[condition_index]], [float(value)],
                    s=11, color=colors[labels[condition_index]], edgecolors="white",
                    linewidths=0.25, zorder=3,
                )
    missing_labels = [labels[index] for index, values in enumerate(data) if not len(values)]
    for index, values in enumerate(data):
        if not len(values):
            ax.text(positions[index], 0.95, "n/a", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7, color="#555555")
    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", labelsize=7, pad=2)
    ax.tick_params(axis="y", labelsize=7, pad=2)
    ax.set_xlim(0.45, len(condition_ids) + 0.55)
    ax.set_ylim(*_axis_limits(data))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.4, alpha=0.65)
    return {
        "metric": metric,
        "labels": labels,
        "condition_ids": condition_ids,
        "n_finite_by_label": {label: int(np.isfinite(arrays[condition_id]).sum())
                               for label, condition_id in label_to_condition.items()},
        "n_a_labels": missing_labels,
        "chromosomes": chromosomes,
    }


def _assert_no_text_overlap(fig: Any, axes: list[Any], footer: Any) -> dict[str, Any]:
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    checks = {"font_size_all_visible_text_pt": True, "x_tick_labels_nonoverlap": True,
              "text_inside_figure": True, "suptitle_clear_of_axes_titles": True,
              "footer_clear_of_x_ticks": True, "axes_nonoverlap": True}
    visible_text = [text for text in fig.findobj() if hasattr(text, "get_text") and text.get_visible() and text.get_text()]
    for text in visible_text:
        if abs(float(text.get_fontsize()) - 7.0) > 1e-6:
            checks["font_size_all_visible_text_pt"] = False
        box = text.get_window_extent(renderer)
        if box.x0 < -1.0 or box.y0 < -1.0 or box.x1 > width + 1.0 or box.y1 > height + 1.0:
            checks["text_inside_figure"] = False
    for ax in axes:
        tick_boxes = [text.get_window_extent(renderer) for text in ax.get_xticklabels()
                      if text.get_visible() and text.get_text()]
        for left_index, left in enumerate(tick_boxes):
            for right in tick_boxes[left_index + 1:]:
                if left.overlaps(right):
                    checks["x_tick_labels_nonoverlap"] = False
        for title in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if title.get_visible() and title.get_text():
                title_box = title.get_window_extent(renderer)
                if title_box.x0 < -1.0 or title_box.y0 < -1.0 or title_box.x1 > width + 1.0 or title_box.y1 > height + 1.0:
                    checks["text_inside_figure"] = False
    axes_boxes = [ax.get_window_extent(renderer) for ax in axes]
    for left_index, left in enumerate(axes_boxes):
        for right in axes_boxes[left_index + 1:]:
            if left.overlaps(right):
                checks["axes_nonoverlap"] = False
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_visible():
        title_box = suptitle.get_window_extent(renderer)
        for ax in axes:
            if title_box.overlaps(ax.title.get_window_extent(renderer)):
                checks["suptitle_clear_of_axes_titles"] = False
    if footer is not None:
        footer_box = footer.get_window_extent(renderer)
        for ax in axes:
            for text in ax.get_xticklabels():
                if text.get_visible() and text.get_text() and footer_box.overlaps(text.get_window_extent(renderer)):
                    checks["footer_clear_of_x_ticks"] = False
    if not all(checks.values()):
        raise RuntimeError("display layout check failed: %s" % checks)
    return checks


def _render_main(plt: Any, stem: Path, title: str, condition_ids: list[str], labels: list[str],
                 chromosomes: list[str], by_condition: Mapping[str, Mapping[str, Mapping[str, Any]]],
                 colors: Mapping[str, str], label_to_condition: Mapping[str, str]) -> dict[str, Any]:
    fig, axes_array = plt.subplots(1, 2, figsize=(6.0, 3.0), squeeze=False)
    axes = list(axes_array[0])
    panel_specs = (("similarity", "Matched similarity", "Spearman rho"),
                   ("contrast", "Matched minus cross", "Contrast (matched - cross)"))
    panel_metadata = []
    for ax, (metric, panel_title, y_label) in zip(axes, panel_specs):
        panel_metadata.append(_plot_metric(ax, condition_ids, labels, chromosomes, by_condition,
                                           metric, colors, label_to_condition))
        ax.set_title(panel_title, fontsize=7, pad=4)
        ax.set_ylabel(y_label, fontsize=7)
    fig.suptitle(title, fontsize=7, y=0.985)
    footer = fig.text(0.5, 0.015,
                      "Dots = chromosome; gray = same chromosome\nBox = distribution, not CI",
                      ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.24, top=0.80, wspace=0.40)
    fig.canvas.draw()
    layout = _assert_no_text_overlap(fig, axes, footer)
    paths = {}
    for extension in ("png", "pdf"):
        path = stem.with_suffix("." + extension)
        fig.savefig(path, dpi=300)
        paths[extension] = _file_record(path)
    plt.close(fig)
    return {"kind": "main_two_panel", "title": title, "panels": panel_metadata,
            "figure_size_inches": [6.0, 3.0], "dpi": 300, "text_size_pt": 7,
            "layout_checks": layout, "files": paths}


def _render_single(plt: Any, path: Path, title: str, metric: str, y_label: str,
                   condition_ids: list[str], labels: list[str], chromosomes: list[str],
                   by_condition: Mapping[str, Mapping[str, Mapping[str, Any]]],
                   colors: Mapping[str, str], label_to_condition: Mapping[str, str]) -> dict[str, Any]:
    fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
    panel = _plot_metric(ax, condition_ids, labels, chromosomes, by_condition,
                         metric, colors, label_to_condition)
    panel_title, separator, subtitle = title.partition(" | ")
    ax.set_title(panel_title if separator else title, fontsize=7, pad=4)
    ax.set_ylabel(y_label, fontsize=7)
    fig.suptitle(("R2 | " + subtitle) if separator else "R2", fontsize=7, y=0.985)
    footer = fig.text(0.5, 0.015,
                      "Dots = chromosome; gray = same chromosome\nBox = distribution, not CI",
                      ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(left=0.23, right=0.98, bottom=0.27, top=0.80)
    fig.canvas.draw()
    layout = _assert_no_text_overlap(fig, [ax], footer)
    fig.savefig(path, dpi=300)
    record = _file_record(path)
    plt.close(fig)
    return {"kind": "single_metric", "title": title, "panel": panel,
            "figure_size_inches": [3.0, 3.0], "dpi": 300, "text_size_pt": 7,
            "layout_checks": layout, "files": {"png": record}}


def _validate_png_dimensions(path: Path, expected: tuple[int, int]) -> list[int]:
    from matplotlib.image import imread
    image = imread(path)
    actual = (int(image.shape[1]), int(image.shape[0]))
    if actual != expected:
        raise RuntimeError("PNG dimension mismatch for %s: %s != %s" % (path, actual, expected))
    return [actual[0], actual[1]]


def main() -> None:
    expected_figures = (
        "r2_new_representatives.png", "r2_new_representatives.pdf",
        "r2_new_representatives_cross.png", "r2_new_representatives_margin_mat.png",
        "r2_new_representatives_margin_pat.png", "r2_historical_controls.png",
        "r2_historical_controls.pdf", "r2_historical_controls_cross.png",
        "r2_historical_controls_margin_mat.png", "r2_historical_controls_margin_pat.png",
    )
    for name in expected_figures + ("render_provenance.json", "output_validation.json"):
        if (OUTPUT_ROOT / name).exists():
            raise RuntimeError("supplement output already exists: %s" % (OUTPUT_ROOT / name))
    per_payload, summary, release, new_ids, historical_ids, new_labels, historical_by_label, by_condition = _validate_inputs()
    original_inventory_before = _inventory(EVAL_ROOT)
    delivery_before = _file_record(DELIVERY_RECEIPT_PATH)
    source_before = _file_record(SOURCE_PATH)
    input_records = {
        "r2_per_chromosome": _file_record(PER_CHROMOSOME_PATH),
        "r2_summary": _file_record(SUMMARY_PATH),
        "release_manifest": _file_record(RELEASE_PATH),
    }
    if release.get("reference", {}).get("sha256") != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("release reference metadata SHA mismatch")
    chromosomes = [str(item["name"]) for item in release["grid"]["chromosomes"]]
    new_labels_ordered = list(VARIANT_ORDER)
    historical_labels_ordered = list(HISTORICAL_LABEL_ORDER)
    new_label_to_condition = {label: condition_id for condition_id, label in new_labels.items()}
    historical_label_to_condition = {label: condition_id for label, condition_id in historical_by_label.items()}
    plt = _configure_matplotlib()
    rendered = OrderedDict()
    rendered["new_representatives"] = _render_main(
        plt, OUTPUT_ROOT / "r2_new_representatives", "R2 | count-selected; all bundle2",
        new_ids, new_labels_ordered, chromosomes, by_condition, NEW_COLORS, new_label_to_condition)
    rendered["new_representatives_cross"] = _render_single(
        plt, OUTPUT_ROOT / "r2_new_representatives_cross.png", "Cross Spearman rho | count-selected; all bundle2",
        "cross", "Cross Spearman rho", new_ids, new_labels_ordered, chromosomes,
        by_condition, NEW_COLORS, new_label_to_condition)
    rendered["new_representatives_margin_mat"] = _render_single(
        plt, OUTPUT_ROOT / "r2_new_representatives_margin_mat.png", "Maternal margin | count-selected; all bundle2",
        "margin_mat", "Margin to fixed mat", new_ids, new_labels_ordered, chromosomes,
        by_condition, NEW_COLORS, new_label_to_condition)
    rendered["new_representatives_margin_pat"] = _render_single(
        plt, OUTPUT_ROOT / "r2_new_representatives_margin_pat.png", "Paternal margin | count-selected; all bundle2",
        "margin_pat", "Margin to fixed pat", new_ids, new_labels_ordered, chromosomes,
        by_condition, NEW_COLORS, new_label_to_condition)
    rendered["historical_controls"] = _render_main(
        plt, OUTPUT_ROOT / "r2_historical_controls", "R2 | historical controls",
        historical_ids, historical_labels_ordered, chromosomes, by_condition,
        HISTORICAL_COLORS, historical_label_to_condition)
    rendered["historical_controls_cross"] = _render_single(
        plt, OUTPUT_ROOT / "r2_historical_controls_cross.png", "Cross Spearman rho | historical controls",
        "cross", "Cross Spearman rho", historical_ids, historical_labels_ordered, chromosomes,
        by_condition, HISTORICAL_COLORS, historical_label_to_condition)
    rendered["historical_controls_margin_mat"] = _render_single(
        plt, OUTPUT_ROOT / "r2_historical_controls_margin_mat.png", "Maternal margin | historical controls",
        "margin_mat", "Margin to fixed mat", historical_ids, historical_labels_ordered, chromosomes,
        by_condition, HISTORICAL_COLORS, historical_label_to_condition)
    rendered["historical_controls_margin_pat"] = _render_single(
        plt, OUTPUT_ROOT / "r2_historical_controls_margin_pat.png", "Paternal margin | historical controls",
        "margin_pat", "Margin to fixed pat", historical_ids, historical_labels_ordered, chromosomes,
        by_condition, HISTORICAL_COLORS, historical_label_to_condition)
    output_records = []
    png_dimensions = {}
    for name in expected_figures:
        path = OUTPUT_ROOT / name
        record = _file_record(path)
        if path.suffix == ".png":
            expected = (1800, 900) if name.endswith(".png") and (name in {
                "r2_new_representatives.png", "r2_historical_controls.png"}) else (900, 900)
            png_dimensions[name] = _validate_png_dimensions(path, expected)
        output_records.append(record)
    original_inventory_after = _inventory(EVAL_ROOT)
    delivery_after = _file_record(DELIVERY_RECEIPT_PATH)
    source_after = _file_record(SOURCE_PATH)
    if original_inventory_before != original_inventory_after:
        raise RuntimeError("original evaluation-r2 inventory changed during display render")
    if delivery_before != delivery_after:
        raise RuntimeError("47-lock delivery receipt changed during display render")
    if source_before != source_after or source_after["sha256"] != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("locked evaluator source changed during display render")
    provenance = {
        "schema": "post020-r2-display-revision-provenance-v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "script": _file_record(SCRIPT_PATH),
        "input_files": input_records,
        "input_policy": {
            "numeric_plot_inputs": [str(PER_CHROMOSOME_PATH.resolve()), str(SUMMARY_PATH.resolve()), str(RELEASE_PATH.resolve())],
            "coordinates_opened": False,
            "reference_opened": False,
            "phase_opened": False,
        },
        "mapping": {
            "new_representatives": [{"label": label, "condition_id": new_label_to_condition[label], "bundle_id": "bundle2"}
                                    for label in new_labels_ordered],
            "historical_controls": [{"label": label, "condition_id": historical_label_to_condition[label]}
                                     for label in historical_labels_ordered],
            "historical_mapping_source": "release_manifest.conditions.display_name",
        },
        "render_settings": {
            "dpi": 300,
            "text_size_pt": 7,
            "main_figure_size_inches": [6.0, 3.0],
            "single_metric_size_inches": [3.0, 3.0],
            "boxplot_interpretation": "chromosome distribution, not CI",
            "same_chromosome_lines": True,
            "n_a_not_zero": True,
            "new_common_note": "count-selected; all bundle2",
            "new_colors": dict(NEW_COLORS),
            "historical_colors": dict(HISTORICAL_COLORS),
        },
        "immutable_lock_check": {
            "original_evaluation_r2_inventory_before": original_inventory_before,
            "original_evaluation_r2_inventory_after": original_inventory_after,
            "original_evaluation_r2_unchanged": original_inventory_before == original_inventory_after,
            "delivery_receipt_47_lock": {
                "path": delivery_before["path"],
                "sha256_before": delivery_before["sha256"],
                "sha256_after": delivery_after["sha256"],
                "unchanged": delivery_before == delivery_after,
            },
            "evaluator_source": {
                "path": source_before["path"],
                "sha256_before": source_before["sha256"],
                "sha256_after": source_after["sha256"],
                "expected_sha256": EXPECTED_SOURCE_SHA256,
                "unchanged": source_before == source_after,
            },
        },
        "outputs": output_records,
        "rendered_figures": rendered,
    }
    provenance_path = OUTPUT_ROOT / "render_provenance.json"
    _atomic_json(provenance_path, provenance)
    provenance_record = _file_record(provenance_path)
    validation = {
        "schema": "post020-r2-display-revision-validation-v1",
        "status": "complete",
        "provenance": provenance_record,
        "figure_count": len(output_records),
        "figure_files": output_records,
        "png_dimensions": png_dimensions,
        "all_layout_checks_passed": all(
            all(checks.values())
            for item in rendered.values()
            for checks in ([item["layout_checks"]] if "layout_checks" in item else [])
        ),
        "original_evaluation_r2_unchanged": original_inventory_before == original_inventory_after,
        "delivery_receipt_47_lock_unchanged": delivery_before == delivery_after,
        "evaluator_source_unchanged": source_before == source_after,
        "evaluator_source_sha256": source_after["sha256"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "n_a_labels": {
            key: [panel.get("n_a_labels", []) for panel in value.get("panels", [])]
            if "panels" in value else value.get("panel", {}).get("n_a_labels", [])
            for key, value in rendered.items()
        },
    }
    _atomic_json(OUTPUT_ROOT / "output_validation.json", validation)
    print(json.dumps({
        "status": "display_revision_complete",
        "output_root": str(OUTPUT_ROOT),
        "figure_count": len(output_records),
        "png_dimensions": png_dimensions,
        "original_evaluation_r2_unchanged": validation["original_evaluation_r2_unchanged"],
        "delivery_receipt_47_lock_unchanged": validation["delivery_receipt_47_lock_unchanged"],
        "evaluator_source_sha256": source_after["sha256"],
        "provenance": str(provenance_path),
        "validation": str(OUTPUT_ROOT / "output_validation.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
