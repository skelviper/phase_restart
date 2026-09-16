#!/usr/bin/env python
"""Independent post-processing for the completed final evaluation.

This script reads only frozen numeric TSV outputs. It does not import or run
the evaluator, refit coordinates, alter numeric result files, or select a new
endpoint. It overwrites only the two presentation PNGs and writes its own
terminal/audit evidence under evaluation_final/.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parent.parent
OUT = RUN / "evaluation_final"
RESULTS = OUT / "results"
PLOTS = OUT / "plots"
POST = OUT / "postprocess"
R2_TSV = RESULTS / "r2_summary.tsv"
TRAJECTORY_TSV = RESULTS / "trajectory_summary.tsv"
SEALED_HASHES = OUT / "input_code_hashes.json"
RUN_README = RUN / "README.md"
RUN_REAL_PLAN = RUN / "REAL_PLAN.md"
DELIVERY_README = OUT / "README.md"
REPOSITORY_README = RUN.parents[1] / "README.md"
POSTPROCESS_CODE = OUT / "postprocess_figures.py"
R2_PNG = PLOTS / "r2_matched_summary.png"
TRAJECTORY_PNG = PLOTS / "trajectory_map_argmax.png"
TERMINAL = OUT / "postprocess_terminal.json"
AUDIT = POST / "postprocess_audit.json"

SELECTED_BRANCHES = {
    "S": {
        "base random": "real-S-random",
        "extension full-J": "real-extension-S-full-J",
        "extension count-only": "real-extension-S-count-only",
    },
    "G": {
        "base random": "real-G-random",
        "extension full-J": "real-extension-G-full-J",
        "extension count-only": "real-extension-G-count-only",
    },
}
BRANCH_COLORS = {"base random": "#1f77b4", "extension full-J": "#d62728", "extension count-only": "#2ca02c"}
ENDPOINT_LABELS = {
    "real-G-consensus": "G-consensus", "real-G-random": "G-random",
    "real-S-consensus": "S-consensus", "real-S-random": "S-random",
    "real-extension-G-count-only": "ext-G-count", "real-extension-G-full-J": "ext-G-full",
    "real-extension-S-count-only": "ext-S-count", "real-extension-S-full-J": "ext-S-full",
    "initial-consensus": "initial-consensus", "initial-random": "initial-random",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _terminal(return_code: int, *, status: str, started: str, finished: str, error: str | None = None, audit: dict[str, Any] | None = None) -> None:
    record: dict[str, Any] = {
        "schema": "p9016-real-evaluation-final-postprocess-terminal-v1",
        "status": status, "return_code": int(return_code), "started_at_utc": started,
        "finished_at_utc": finished, "command": [sys.executable, str(POSTPROCESS_CODE)],
        "postprocess_code": str(POSTPROCESS_CODE), "postprocess_code_sha256": _sha(POSTPROCESS_CODE),
        "evaluator_called": False, "refit_started": False, "numeric_result_files_modified": False,
        "r2_input": str(R2_TSV), "trajectory_input": str(TRAJECTORY_TSV),
    }
    if error is not None:
        record["error"] = error
    if audit is not None:
        record["audit"] = str(AUDIT)
        record["sealed_input_hash_audit"] = audit.get("sealed_input_hash_audit")
        record["output_hashes"] = audit.get("outputs")
    _write_json(TERMINAL, record)


def _sealed_input_audit() -> dict[str, Any]:
    if not SEALED_HASHES.is_file():
        raise RuntimeError("sealed input_code_hashes.json is missing")
    sealed = json.loads(SEALED_HASHES.read_text(encoding="utf-8"))
    records = []
    for group in ("code", "inputs"):
        values = sealed.get(group)
        if not isinstance(values, dict):
            raise RuntimeError("sealed hash group is missing: %s" % group)
        for raw_path, expected in sorted(values.items()):
            path = Path(raw_path)
            if not path.is_file():
                raise RuntimeError("sealed hash path is missing: %s" % path)
            actual = _sha(path)
            records.append({"group": group, "path": str(path), "expected_sha256": str(expected), "actual_sha256": actual, "match": actual == str(expected)})
    mismatches = [row for row in records if not row["match"]]
    if mismatches:
        raise RuntimeError("sealed code/input hash mismatch: %s" % mismatches[:3])
    document_hashes = {}
    for label, path in (("run_readme", RUN_README), ("run_real_plan", RUN_REAL_PLAN), ("evaluation_final_readme", DELIVERY_README), ("repository_readme", REPOSITORY_README)):
        if not path.is_file():
            raise RuntimeError("documentation path is missing: %s" % path)
        document_hashes[label] = {"path": str(path), "sha256": _sha(path), "in_sealed_input_hashes": str(path) in sealed.get("inputs", {})}
    return {"status": "PASS", "sealed_hash_file": str(SEALED_HASHES), "sealed_hash_file_sha256": _sha(SEALED_HASHES), "checked_count": len(records), "mismatch_count": 0, "records": records, "documentation_hashes": document_hashes}


def _style() -> None:
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 5.8, "ytick.labelsize": 6.2, "legend.fontsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _plot_r2(rows: list[dict[str, str]]) -> dict[str, Any]:
    labels = [ENDPOINT_LABELS.get(row["candidate_id"], row["candidate_id"]) for row in rows]
    matched = [float(row["pearson_matched"]) for row in rows]
    contrast = [float(row["pearson_contrast"]) for row in rows]
    x = list(range(len(rows)))
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6, 2.8), dpi=300, sharey=True)
    for ax, values, title, color in (
        (axes[0], matched, "Pearson matched", "#3b6ea8"),
        (axes[1], contrast, "Pearson contrast", "#c85a3f"),
    ):
        ax.bar(x, values, color=color, width=0.78)
        ax.axhline(0.0, color="#333333", linewidth=0.45)
        ax.set_xticks(x, labels, rotation=55, ha="right")
        ax.set_title(title)
        ax.set_xlabel("P9016 endpoint")
        ax.grid(axis="y", color="#dddddd", linewidth=0.35)
    axes[0].set_ylabel("Pearson R2")
    fig.suptitle("Frozen old21 R2 endpoint summary", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(R2_PNG, dpi=300)
    plt.close(fig)
    return {"path": str(R2_PNG), "metric_fields": ["pearson_matched", "pearson_contrast"], "endpoint_count": len(rows), "panel_count": 2, "figsize_inches": [6.0, 2.8], "dpi": 300, "matched_values_sha256": hashlib.sha256(json.dumps(matched, separators=(",", ":")).encode()).hexdigest(), "contrast_values_sha256": hashlib.sha256(json.dumps(contrast, separators=(",", ":")).encode()).hexdigest()}


def _branch_rows(rows: list[dict[str, str]], fit_id: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["fit_id"] == fit_id and row["stage"] == "1Mb"]
    if not selected:
        raise RuntimeError("missing selected 1Mb trajectory branch: %s" % fit_id)
    selected.sort(key=lambda row: (int(row["nfev"]), int(row["iteration"])))
    nfev = [int(row["nfev"]) for row in selected]
    if any(left > right for left, right in zip(nfev, nfev[1:])):
        raise RuntimeError("nfev is not monotonic within branch: %s" % fit_id)
    return selected


def _plot_trajectory(rows: list[dict[str, str]]) -> dict[str, Any]:
    metrics = (("distance_intra_map", "MAP intra distance"), ("distance_inter_map", "MAP inter distance"), ("whole_cell_Rg", "Whole-cell Rg"))
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=300, sharex="col")
    branch_counts: dict[str, int] = {}
    for row_index, model in enumerate(("S", "G")):
        for col_index, (field, ylabel) in enumerate(metrics):
            ax = axes[row_index, col_index]
            for label, fit_id in SELECTED_BRANCHES[model].items():
                branch = _branch_rows(rows, fit_id)
                x0 = int(branch[0]["nfev"])
                x = [int(item["nfev"]) - x0 for item in branch]
                y = [float(item[field]) for item in branch]
                ax.plot(x, y, color=BRANCH_COLORS[label], linewidth=0.75, marker="o", markersize=1.4, label=label)
                branch_counts[fit_id] = len(branch)
            if row_index == 0:
                ax.set_title(ylabel)
            if col_index == 0:
                ax.set_ylabel("S" if model == "S" else "G")
            if row_index == 1:
                ax.set_xlabel("FG calls within branch")
            ax.grid(color="#dddddd", linewidth=0.35)
            ax.tick_params(axis="x", labelrotation=0)
            if row_index == 0 and col_index == 2:
                ax.legend(loc="best", frameon=False, handlelength=1.8)
    fig.suptitle("Selected random 1Mb branches and full-J/count-only extensions", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(TRAJECTORY_PNG, dpi=300)
    plt.close(fig)
    return {"path": str(TRAJECTORY_PNG), "stage_filter": "1Mb only; all 799 trajectory rows remain in trajectory_summary.tsv", "metric_fields": [field for field, _ in metrics], "models": ["S", "G"], "branches": SELECTED_BRANCHES, "branch_row_counts": branch_counts, "panel_shape": [2, 3], "panel_size_inches": [3.0, 3.0], "figsize_inches": [9.0, 6.0], "dpi": 300, "x_definition": "nfev minus the first nfev within each branch; branches are sorted independently and never connected across stage/branch"}


def main() -> int:
    started = _now()
    try:
        if not R2_TSV.is_file() or not TRAJECTORY_TSV.is_file():
            raise RuntimeError("required completed TSV input is missing")
        r2_rows = _read_tsv(R2_TSV)
        trajectory_rows = _read_tsv(TRAJECTORY_TSV)
        if len(r2_rows) != 10:
            raise RuntimeError("r2_summary.tsv must contain 10 endpoint rows, got %d" % len(r2_rows))
        sealed_input_audit = _sealed_input_audit()
        if len(trajectory_rows) != 799:
            raise RuntimeError("trajectory_summary.tsv must contain 799 rows, got %d" % len(trajectory_rows))
        for row in trajectory_rows:
            if row["stage"] == "1Mb":
                for field in ("distance_intra_map", "distance_inter_map", "whole_cell_Rg"):
                    if not row[field] or not float(row[field]) == float(row[field]):
                        raise RuntimeError("nonfinite 1Mb trajectory metric: %s" % field)
        r2_audit = _plot_r2(r2_rows)
        trajectory_audit = _plot_trajectory(trajectory_rows)
        audit = {
            "schema": "p9016-real-evaluation-final-postprocess-audit-v1", "status": "PASS",
            "started_at_utc": started, "finished_at_utc": _now(), "evaluator_called": False,
            "numeric_result_files_modified": False, "r2_input": str(R2_TSV), "r2_input_sha256": _sha(R2_TSV),
            "trajectory_input": str(TRAJECTORY_TSV), "trajectory_input_sha256": _sha(TRAJECTORY_TSV),
            "trajectory_row_count": len(trajectory_rows), "r2_row_count": len(r2_rows),
            "sealed_input_hash_audit": sealed_input_audit,
            "r2_plot": r2_audit, "trajectory_plot": trajectory_audit,
            "outputs": {"r2_png_sha256": _sha(R2_PNG), "trajectory_png_sha256": _sha(TRAJECTORY_PNG)},
            "postprocess_code_sha256": _sha(POSTPROCESS_CODE),
        }
        _write_json(AUDIT, audit)
        _terminal(0, status="terminal", started=started, finished=audit["finished_at_utc"], audit=audit)
        return 0
    except Exception as exc:
        _terminal(1, status="terminal", started=started, finished=_now(), error=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
