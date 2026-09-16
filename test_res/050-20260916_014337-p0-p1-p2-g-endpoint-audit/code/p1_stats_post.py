"""P1 统计后处理（050 轮）：R1/R3 配对 bootstrap、逐 chr 胜出、R3 片段共同支持身份检查、
按 source×null_kind×metric 分层的 null 汇总，以及 R2 基线回归。

只读取已持久化的 TSV/JSON，不重算任何指标，也不重跑拟合。
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
ROOT = RUN_DIR.parents[1]
S046_EVAL = ROOT / "test_res/046-UTC-real-cell-shared-capture/evaluation_final"
if str(S046_EVAL) not in sys.path:
    sys.path.insert(0, str(S046_EVAL))
import evaluator as ev  # noqa: E402

BOOTSTRAP_INDICES = S046_EVAL / "bootstrap_indices_seed450301_10000x20.npy"
GENOME_ORDER = ["chr%d" % k for k in range(1, 20)] + ["chrX"]

R1R3_PAIRS = [
    ("A-raw-G-consensus", "A-ms-G-consensus", "R1/R3 solver: raw minus ms at the G-consensus base"),
    ("A-raw-G-random", "A-ms-G-random", "R1/R3 solver: raw minus ms at the G-random base"),
    ("A-raw-G-consensus", "046-base-G-consensus", "R1/R3 fork: raw minus base at G-consensus"),
    ("A-ms-G-consensus", "046-base-G-consensus", "R1/R3 fork: ms minus base at G-consensus"),
    ("A-raw-G-random", "046-base-G-random", "R1/R3 fork: raw minus base at G-random"),
    ("A-ms-G-random", "046-base-G-random", "R1/R3 fork: ms minus base at G-random"),
]
R1R3_METRICS = {
    "R1_accuracy": ("r1_accuracy", "higher"),
    "R3_frac_consistent": ("r3_frac_consistent", "higher"),
    "R3_longest_run": ("r3_longest_run", "higher"),
    "R3_n_walls": ("r3_n_walls", "lower"),
}
# 046 已发布的 R2 基线（Pearson macro，046 evaluation_final/README.md 表）
R2_BASELINE_EXPECTED = {
    "046-base-G-consensus": {"matched": 0.548096, "cross": 0.396223, "contrast": 0.151873},
    "046-base-G-random": {"matched": 0.602361, "cross": 0.365617, "contrast": 0.236743},
    "046-work-baseline-G-full-J": {"matched": 0.602331, "cross": 0.364710, "contrast": 0.237621},
}
R2_TOLERANCE = 5e-7


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ev._jsonable(value), indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join("NA" if row.get(key) is None else str(row.get(key))
                                   for key in columns) + "\n")


def load_table(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value in ("", "NA", "None"):
                    parsed[key] = None
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


def series(rows: Sequence[Mapping[str, Any]], candidate: str, field: str) -> list[Any]:
    lookup = {row["chromosome"]: row for row in rows if row["candidate_id"] == candidate}
    return [lookup.get(name, {}).get(field) for name in GENOME_ORDER]


def r1r3_section() -> dict[str, Any]:
    source = RUN_DIR / "evaluation" / "r1r3_arms_and_bases"
    rows = load_table(source / "r1_r3_per_chromosome.tsv")
    fragments = load_table(source / "r3_fragments.tsv")
    indices = np.load(BOOTSTRAP_INDICES)
    if indices.shape != (10_000, 20):
        raise RuntimeError("frozen bootstrap index matrix shape changed")
    out_rows = []
    for left, right, label in R1R3_PAIRS:
        for metric, (field, better) in R1R3_METRICS.items():
            left_values = series(rows, left, field)
            right_values = series(rows, right, field)
            entry = ev._bootstrap_fixed(left_values, right_values, GENOME_ORDER, indices)
            delta = [None if (a is None or b is None) else float(a) - float(b)
                     for a, b in zip(left_values, right_values)]
            higher = sum(1 for value in delta if value is not None and value > 1e-12)
            lower = sum(1 for value in delta if value is not None and value < -1e-12)
            ties = sum(1 for value in delta if value is not None and abs(value) <= 1e-12)
            out_rows.append({
                "comparison": label, "left": left, "right": right, "metric": metric,
                "better_direction": better,
                "mean": entry["mean"], "ci95_low": entry["ci95"][0], "ci95_high": entry["ci95"][1],
                "left_higher": higher, "right_higher": lower, "ties": ties,
                "left_wins": entry["left_wins"], "right_wins": entry["right_wins"],
                "improvement_wins": (higher if better == "higher" else lower),
                "improvement_wins_label": ("left better" if better == "higher" else "left better (fewer walls)"),
                "wins_semantics": ("left_wins/right_wins are counts of HIGHER values only; for R3_n_walls fewer "
                                   "walls is better, so improvement_wins is the count of delta < 0"),
                "defined_chromosomes": entry["defined_chromosomes"],
                "full_20_estimate_defined": entry["full_20_estimate_defined"],
                "undefined_chromosomes": "|".join(entry["undefined_chromosomes"]),
                "defined_only_mean": entry["defined_only_descriptive"]["mean"],
                "defined_only_n": entry["defined_only_descriptive"]["n"],
            })
    write_tsv(RUN_DIR / "evaluation" / "results" / "p1_r1r3_paired_bootstrap.tsv", out_rows)

    # R3 片段共同支持身份检查：同一 (chr, grid_start_bin) 上两个候选都可解析才算共同片段
    by_candidate: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = {}
    for row in fragments:
        by_candidate.setdefault(str(row["candidate_id"]), {})[(str(row["chromosome"]),
                                                               int(row["grid_start_bin"]))] = row
    support_rows = []
    common_candidates = sorted({str(row["candidate_id"]) for row in fragments})
    for left, right, label in R1R3_PAIRS:
        per_chromosome = {}
        for name in GENOME_ORDER:
            keys_left = {key for key in by_candidate.get(left, {}) if key[0] == name}
            keys_right = {key for key in by_candidate.get(right, {}) if key[0] == name}
            common = [key for key in (keys_left & keys_right)
                      if by_candidate[left][key].get("label") is not None
                      and by_candidate[right][key].get("label") is not None]
            total = len(keys_left | keys_right)
            per_chromosome[name] = {"n_common_resolved_fragments": len(common),
                                    "n_fragments_union": total}
        counts = [value["n_common_resolved_fragments"] for value in per_chromosome.values()]
        support_rows.append({
            "comparison": label, "left": left, "right": right,
            "common_resolved_fragments_total": int(sum(counts)),
            "min_per_chromosome": int(min(counts)), "median_per_chromosome": float(np.median(counts)),
            "max_per_chromosome": int(max(counts)),
            "chromosomes_with_zero_common": "|".join(
                name for name, value in per_chromosome.items()
                if value["n_common_resolved_fragments"] == 0),
            "identical_fragment_grid": True,
            "rule": "fragment identity = (chromosome, grid_start_bin); a fragment counts as common support only "
                    "when BOTH candidates resolve it to a label (tied/insufficient fragments break runs and are "
                    "never bridged)",
            "per_chromosome": per_chromosome,
        })
    write_tsv(RUN_DIR / "evaluation" / "results" / "p1_r3_common_support.tsv",
              [{key: value for key, value in row.items() if key != "per_chromosome"} for row in support_rows])
    write_json(RUN_DIR / "evaluation" / "results" / "p1_r3_common_support.json", {
        "schema": "p9016-round050-p1-r3-common-support-v1",
        "candidate_fragment_tables": {name: str((source / "r3_fragments.tsv").relative_to(ROOT))
                                      for name in common_candidates},
        "rows": support_rows})
    return {"rows": out_rows, "support": support_rows}


def support_identity() -> dict[str, Any]:
    """证明主比较的 R1 支持与 R3 片段身份在各候选间完全相同（由持久化表核对）。"""
    source = RUN_DIR / "evaluation" / "r1r3_arms_and_bases"
    rows = load_table(source / "r1_r3_per_chromosome.tsv")
    fragments = load_table(source / "r3_fragments.tsv")
    candidates = sorted({str(row["candidate_id"]) for row in rows})
    per_chr_support: dict[str, dict[str, Any]] = {}
    mismatches = []
    for name in GENOME_ORDER:
        entry = {cid: next((row["r1_n_common"] for row in rows
                            if row["candidate_id"] == cid and row["chromosome"] == name), None)
                 for cid in candidates}
        per_chr_support[name] = entry
        values = [value for value in entry.values() if value is not None]
        if values and len(set(values)) != 1:
            mismatches.append({"chromosome": name, "denominators": entry})
    totals = {cid: sum(int(entry[cid]) for entry in per_chr_support.values() if entry[cid] is not None)
              for cid in candidates}
    labelled: dict[str, set] = {}
    for row in fragments:
        if row.get("label") is not None and str(row.get("status")) == "labelled":
            labelled.setdefault(str(row["candidate_id"]), set()).add(
                (str(row["chromosome"]), int(row["grid_start_bin"])))
    common = set.intersection(*labelled.values()) if labelled else set()
    union = set.union(*labelled.values()) if labelled else set()
    payload = {
        "schema": "p9016-round050-support-identity-v1",
        "candidates": candidates,
        "R1_per_chromosome_denominator_identical": not mismatches,
        "R1_denominator_mismatches": mismatches,
        "R1_denominator_per_chromosome": {name: per_chr_support[name][candidates[0]]
                                          for name in GENOME_ORDER},
        "R1_denominator_total_per_candidate": totals,
        "R1_denominator_total_identical": len(set(totals.values())) == 1,
        "R3_labelled_fragment_set_identical": bool(labelled) and common == union,
        "R3_labelled_fragments_common": len(common),
        "R3_labelled_fragments_union": len(union),
        "R3_labelled_fragments_per_candidate": {cid: len(values) for cid, values in labelled.items()},
        "rule": "the paired R1/R3 statistics are only computed after this identity check passes; the pairing is at "
                "the chromosome level on the frozen 20-chromosome denominator",
    }
    write_json(RUN_DIR / "evaluation" / "results" / "p1_support_identity.json", payload)
    return payload


def grouped_nulls(table: Path, out_json: Path, schema: str, extra: Mapping[str, Any]) -> dict[str, Any]:
    """按 source × null_kind × metric 分层汇总。

    `p1_null_summary.tsv` 每行带 `metric`（pearson / spearman），因此 metric 是分层维度；
    `p0/controls/null_support.tsv` 每个字段本身就是一个 metric（R1_... / R2_... / R3_...），
    因此按字段名分层。两种情况下都不做跨 source 的池化平均。
    """
    rows = load_table(table)
    has_metric_column = "metric" in rows[0]
    value_fields = [key for key in rows[0]
                    if key not in ("source", "null_kind", "seed", "metric", "R3_status_counts",
                                   "undefined_chromosomes")
                    and (rows[0][key] is None or isinstance(rows[0][key], (int, float)))]
    grouped: dict[str, Any] = {}
    strata: list[tuple[str, str, str | None]] = []
    for source in sorted({str(row["source"]) for row in rows}):
        for kind in ("u_zero", "random_u"):
            if has_metric_column:
                for metric in sorted({str(row["metric"]) for row in rows if str(row["source"]) == source}):
                    strata.append((source, kind, metric))
            else:
                strata.append((source, kind, None))
    for source, kind, metric in strata:
        subset = [row for row in rows
                  if str(row["source"]) == source and row["null_kind"] == kind
                  and (metric is None or str(row.get("metric")) == metric)]
        if not subset:
            continue
        seeds = sorted({int(row["seed"]) for row in subset if row["seed"] is not None})
        entry = {"n_draws": (len(seeds) if seeds else (1 if subset else 0)),
                 "n_rows": len(subset),
                 "seeds": seeds,
                 "seed_note": ("u_zero has no seed and counts as one deterministic control draw"
                               if not seeds else None),
                 "metric": metric}
        for field in value_fields:
            values = [float(row[field]) for row in subset if row[field] is not None]
            entry[field] = {"n": len(values),
                            "mean": float(np.mean(values)) if values else None,
                            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                            "min": float(np.min(values)) if values else None,
                            "max": float(np.max(values)) if values else None}
        key = "%s:%s" % (source, kind) if metric is None else "%s:%s:%s" % (source, kind, metric)
        grouped[key] = entry
    payload = {"schema": schema,
               "rule": "nulls are aggregated per source x null_kind x metric; a cross-source pooled mean is NOT "
                       "used as any candidate's null distribution, because each candidate has its own geometry",
               "grouped": grouped, **extra}
    write_json(out_json, payload)
    return payload


def r2_regression() -> dict[str, Any]:
    rows = load_table(RUN_DIR / "evaluation" / "results" / "p1_source_summary.tsv")
    lookup = {(str(row["source"]), str(row["metric"])): row for row in rows}
    results = []
    for source, expected in R2_BASELINE_EXPECTED.items():
        for metric in ("pearson", "spearman"):
            row = lookup.get((source, metric))
            if metric == "spearman":
                continue
            observed = {"matched": row["matched_macro"], "cross": row["cross_macro"],
                        "contrast": row["contrast_macro"]}
            diffs = {key: abs(float(observed[key]) - float(expected[key])) for key in expected}
            results.append({"source": source, "metric": metric, "observed": observed,
                            "expected_046_published": expected, "abs_diff": diffs,
                            "status": "PASS" if all(v <= R2_TOLERANCE for v in diffs.values()) else "FAIL"})
    status = "PASS" if results and all(row["status"] == "PASS" for row in results) else "FAIL"
    payload = {
        "schema": "p9016-round050-r2-baseline-regression-v1", "status": status,
        "tolerance": R2_TOLERANCE,
        "expected_source": "test_res/046-UTC-real-cell-shared-capture/evaluation_final/README.md macro table "
                           "(published values, 6 decimals)",
        "purpose": "confirm that this round's independent old21-mask R2 path reproduces the frozen 046 readouts "
                   "for the three shared reference sources before any new endpoint value is interpreted",
        "rows": results,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_json(RUN_DIR / "evaluation" / "results" / "r2_baseline_regression.json", payload)
    return payload


def main() -> int:
    identity = support_identity()
    if not identity["R1_per_chromosome_denominator_identical"] or \
            not identity["R3_labelled_fragment_set_identical"]:
        raise RuntimeError("main paired support identity check failed: %s" % identity)
    stats = r1r3_section()
    grouped_nulls(RUN_DIR / "evaluation" / "results" / "p1_null_summary.tsv",
                  RUN_DIR / "evaluation" / "results" / "p1_null_aggregate_by_source.json",
                  "p9016-round050-p1-null-aggregate-by-source-v1",
                  {"source_table": "evaluation/results/p1_null_summary.tsv"})
    grouped_nulls(RUN_DIR / "p0" / "controls" / "null_support.tsv",
                  RUN_DIR / "p0" / "controls" / "null_aggregate_by_source.json",
                  "p9016-round050-p0-null-aggregate-by-source-v1",
                  {"source_table": "p0/controls/null_support.tsv"})
    regression = r2_regression()
    print(json.dumps({"r1r3_rows": len(stats["rows"]), "r3_support_rows": len(stats["support"]),
                      "r1_denominator_identical": identity["R1_denominator_total_identical"],
                      "r1_denominator_total": identity["R1_denominator_total_per_candidate"],
                      "r3_common_fragments": identity["R3_labelled_fragments_common"],
                      "r2_regression": regression["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
