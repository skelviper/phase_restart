"""P0 对照的后处理（050 轮）：从已持久化的 R1/R3 结果生成配对 bootstrap 与 null 支持表。

`p0_controls.py` 的正式评价本身已完成并写出 `r1_r3_summary.json` / `r1_r3_per_chromosome.tsv`；
本脚本不重算任何指标，只做统计汇总，因此可以独立重跑。
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
for _path in (str(S046_EVAL),):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import evaluator as ev  # noqa: E402

BOOTSTRAP_INDICES = S046_EVAL / "bootstrap_indices_seed450301_10000x20.npy"
GENOME_ORDER = ["chr%d" % k for k in range(1, 20)] + ["chrX"]
ENDPOINTS = ("046-base-G-random", "046-work-baseline-G-full-J", "049-A-ms-random",
             "049-B-raw-consensus", "049-C-ms-random")
SOURCE_COLUMNS = {
    "fixed_random": {"R1": "r1_fixed_random",
                     "R2_contrast_spearman": "r2_fixed_random_contrast_spearman",
                     "R3_frac_consistent": "r3_fixed_random_frac_consistent",
                     "R3_n_walls": "r3_fixed_random_n_walls"},
    "oracle": {"R1": "r1_oracle_fit_ceiling", "R2_contrast_spearman": None,
               "R3_frac_consistent": None, "R3_n_walls": None},
    "reference_ceiling": {"R1": "r1_reference_ceiling", "R2_contrast_spearman": None,
                          "R3_frac_consistent": None, "R3_n_walls": None},
    "fixed_consensus": {"R1": None, "R2_contrast_spearman": "r2_consensus_contrast_spearman",
                        "R3_frac_consistent": "r3_consensus_frac_consistent",
                        "R3_n_walls": "r3_consensus_n_walls"},
}
CANDIDATE_COLUMNS = {"R1": "r1_accuracy", "R2_contrast_spearman": "r2_selected_contrast_spearman",
                     "R3_frac_consistent": "r3_frac_consistent", "R3_n_walls": "r3_n_walls"}
COMPARISONS = [
    ("046-base-G-random", "fixed_random", "046 base G-random minus fixed 014 random"),
    ("046-work-baseline-G-full-J", "fixed_random", "046 work baseline minus fixed 014 random"),
    ("049-A-ms-random", "fixed_random", "049 A-ms-random minus fixed 014 random"),
    ("049-B-raw-consensus", "fixed_random", "049 B-raw-consensus minus fixed 014 random"),
    ("049-C-ms-random", "fixed_random", "049 C-ms-random minus fixed 014 random"),
    ("049-A-ms-random", "046-work-baseline-G-full-J", "049 A-ms-random minus 046 work baseline"),
    ("049-B-raw-consensus", "046-work-baseline-G-full-J", "049 B-raw-consensus minus 046 work baseline"),
    ("049-C-ms-random", "046-work-baseline-G-full-J", "049 C-ms-random minus 046 work baseline"),
    ("046-base-G-random", "046-work-baseline-G-full-J", "046 base G-random minus 046 work baseline"),
    ("046-base-G-random", "oracle", "046 base G-random minus S0 oracle-fit ceiling"),
    ("046-work-baseline-G-full-J", "oracle", "046 work baseline minus S0 oracle-fit ceiling"),
    ("049-A-ms-random", "oracle", "049 A-ms-random minus S0 oracle-fit ceiling"),
]
NULL_METRICS = ("R1", "R3_frac_consistent")
METRIC_DIRECTION = {"R1": "higher", "R2_contrast_spearman": "higher",
                    "R3_frac_consistent": "higher", "R3_n_walls": "lower"}
TOKEN = {"1": "".join(("1", "")), "NULL": "NA"}


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


def load_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value == "NA" or value == "":
                    parsed[key] = None
                elif key in ("candidate_id", "chromosome", "r1_accuracy_policy", "r1_orientation",
                             "r1_gauge_status", "r1_fixed_random_policy", "r3_global_label",
                             "r3_global_label_tied", "r3_status_summary", "r3_consensus_applicable",
                             "r3_consensus_reason"):
                    parsed[key] = value
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            out.append(parsed)
    return out


def series(rows: Sequence[Mapping[str, Any]], source: str, metric: str) -> list[Any]:
    lookup = {row["chromosome"]: row for row in rows if row["candidate_id"] == source}
    if lookup:
        column = CANDIDATE_COLUMNS[metric]
    else:
        column = SOURCE_COLUMNS[source][metric]
        if column is None:
            return [None] * len(GENOME_ORDER)
        lookup = {}
        for row in rows:
            lookup.setdefault(row["chromosome"], row)
    values = []
    for name in GENOME_ORDER:
        value = lookup.get(name, {}).get(column)
        values.append(float(value) if isinstance(value, (int, float)) else None)
    return values


def main() -> int:
    out = RUN_DIR / "p0" / "controls"
    summary = json.loads((out / "r1_r3_summary.json").read_text(encoding="utf-8"))
    rows = load_rows(out / "r1_r3_per_chromosome.tsv")
    fragment_rows = load_rows(out / "r3_fragments.tsv")
    null_index = json.loads((out / "null_index.json").read_text(encoding="utf-8"))
    indices = np.load(BOOTSTRAP_INDICES)
    if indices.shape != (10_000, 20):
        raise RuntimeError("frozen bootstrap index matrix shape changed")

    # 主端点支持的同一性：R1 逐 chr 分母与 R3 已解析片段集合必须完全一致
    per_chr_denominators: dict[str, dict[str, Any]] = {}
    mismatches = []
    for name in GENOME_ORDER:
        entry = {cid: next((row["r1_n_common"] for row in rows
                            if row["candidate_id"] == cid and row["chromosome"] == name), None)
                 for cid in ENDPOINTS}
        per_chr_denominators[name] = entry
        values = [value for value in entry.values() if value is not None]
        if values and len(set(values)) != 1:
            mismatches.append({"chromosome": name, "denominators": entry})
    totals = {cid: sum(int(entry[cid]) for entry in per_chr_denominators.values() if entry[cid] is not None)
              for cid in ENDPOINTS}
    labelled: dict[str, set] = {}
    for row in fragment_rows:
        # 主端点身份检查只覆盖 5 个冻结端点；null 有自己的支持（例如某些 random-u 在 chr17:80 不可解析），
        # 它们不参与主比较掩码，也不得让主端点身份检查失败。
        if str(row.get("candidate_id")) not in ENDPOINTS:
            continue
        if row.get("label") is not None and str(row.get("status")) == "labelled":
            labelled.setdefault(str(row["candidate_id"]), set()).add(
                (str(row["chromosome"]), int(row["grid_start_bin"])))
    common = set.intersection(*labelled.values()) if labelled else set()
    union = set.union(*labelled.values()) if labelled else set()
    identity = {
        "schema": "p9016-round050-p0-support-identity-v1",
        "endpoints": list(ENDPOINTS),
        "R1_per_chromosome_denominator_identical": not mismatches,
        "R1_denominator_mismatches": mismatches,
        "R1_denominator_total_per_endpoint": totals,
        "R1_denominator_total_identical": len(set(totals.values())) == 1,
        "R3_labelled_fragment_set_identical": bool(labelled) and common == union,
        "R3_labelled_fragments_common": len(common),
        "R3_labelled_fragments_union": len(union),
        "rule": "paired statistics are computed only after this identity check passes; nulls keep their own "
                "resolvable support and are never intersected into this mask",
    }
    write_json(out / "support_identity.json", identity)
    if not identity["R1_per_chromosome_denominator_identical"] or \
            not identity["R3_labelled_fragment_set_identical"]:
        raise RuntimeError("P0 main support identity check failed")

    comparisons, bootstrap_rows = [], []
    for left, right, label in COMPARISONS:
        for metric in ("R1", "R2_contrast_spearman", "R3_frac_consistent", "R3_n_walls"):
            better = METRIC_DIRECTION[metric]
            left_values = series(rows, left, metric)
            right_values = series(rows, right, metric)
            entry = ev._bootstrap_fixed(left_values, right_values, GENOME_ORDER, indices)
            delta = [None if (a is None or b is None) else float(a) - float(b)
                     for a, b in zip(left_values, right_values)]
            higher = sum(1 for value in delta if value is not None and value > 1e-12)
            lower = sum(1 for value in delta if value is not None and value < -1e-12)
            ties = sum(1 for value in delta if value is not None and abs(value) <= 1e-12)
            comparisons.append({"label": label, "left": left, "right": right, "metric": metric,
                                "better_direction": better, "direction": "delta = left minus right",
                                **entry})
            bootstrap_rows.append({
                "comparison": label, "left": left, "right": right, "metric": metric,
                "better_direction": better,
                "mean": entry["mean"], "ci95_low": entry["ci95"][0], "ci95_high": entry["ci95"][1],
                "left_higher": higher, "right_higher": lower, "ties": ties,
                "left_wins": entry["left_wins"], "right_wins": entry["right_wins"],
                "improvement_wins": (higher if better == "higher" else lower),
                "wins_semantics": ("left_wins/right_wins count HIGHER values only; for R3_n_walls fewer walls is "
                                   "better, so improvement_wins counts delta < 0"),
                "defined_chromosomes": entry["defined_chromosomes"],
                "full_20_estimate_defined": entry["full_20_estimate_defined"],
                "undefined_chromosomes": "|".join(entry["undefined_chromosomes"]),
                "defined_only_mean": entry["defined_only_descriptive"]["mean"],
                "defined_only_n": entry["defined_only_descriptive"]["n"],
            })
    write_tsv(out / "paired_bootstrap.tsv", bootstrap_rows)
    write_json(out / "paired_bootstrap.json", {
        "schema": "p9016-round050-p0-paired-bootstrap-v1", "seed": 450301, "draws": 10_000,
        "index_matrix": str(BOOTSTRAP_INDICES.relative_to(ROOT)),
        "unit": "paired chromosome resampling within one cell; technical/structural variation, not biological "
                "replication",
        "rule": "fixed 20-chromosome denominator; if any chromosome is undefined the primary mean/CI is null and "
                "the defined-only descriptive value and n are reported next to it",
        "control_semantics": {
            "fixed_random": "014 fixed random baseline read from the SAME paired_r1 common-record support as the "
                            "candidate, matching the 020 report row",
            "oracle": "014 S0 oracle-fit ceiling, same common support",
            "reference_ceiling": "reference-structure ceiling, same common support",
            "fixed_consensus": "single-trajectory 014 consensus has no two-copy R1 (its R1 column is correctly "
                               "unavailable, not missing); only its R2/R3 readouts are comparable",
        },
        "rows": comparisons})

    null_support = []
    for record in null_index["nulls"]:
        cid = "%s__%s" % (record["source"], "u-zero" if record["seed"] is None else "random-u-%d" % record["seed"])
        entry = summary["candidates"].get(cid)
        if entry is None:
            continue
        null_support.append({
            "source": record["source"], "null_kind": record["null_kind"], "seed": record["seed"],
            "R1_macro_mean_all20": entry["R1_macro_mean_all20"],
            "R1_macro_mean_defined_only": entry["R1_macro_mean_defined_only"],
            "R1_defined_chromosomes": entry["R1_defined_chromosomes"],
            "R1_undefined_chromosomes": "|".join(entry["R1_undefined_chromosomes"]),
            "R1_denominator_total_declared20": entry["R1_denominator_total_declared20"],
            "R1_pooled_over_defined_common_records": entry["R1_pooled_over_defined_common_records"],
            "R2_contrast_spearman_macro_mean_all20": entry["R2_contrast_spearman_macro_mean_all20"],
            "R3_frac_consistent_macro_mean_all20": entry["R3_frac_consistent_macro_mean_all20"],
            "R3_frac_consistent_defined_chromosomes": entry["R3_frac_consistent_defined_chromosomes"],
            "R3_fragments_applicable": entry["R3_fragments_applicable"],
            "R3_fragments_tied": entry["R3_fragments_tied"],
            "R3_status_counts": json.dumps(entry["R3_status_counts"], sort_keys=True),
            "post_scale_radius": record["post_scale_radius"], "global_scale": record["global_scale"],
        })
    write_tsv(out / "null_support.tsv", null_support)
    aggregate = {}
    for kind in ("u_zero", "random_u"):
        for field in ("R1_macro_mean_defined_only", "R2_contrast_spearman_macro_mean_all20",
                      "R3_frac_consistent_macro_mean_all20"):
            values = [row[field] for row in null_support
                      if row["null_kind"] == kind and row[field] is not None]
            aggregate["%s:%s" % (kind, field)] = {
                "n": len(values), "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    write_json(out / "null_aggregate.json", {
        "schema": "p9016-round050-p0-null-aggregate-v1", "aggregate": aggregate,
        "n_null_rows": len(null_support),
        "rule": "each null reports its own resolvable support; null resolvability is never intersected into the "
                "main candidate comparison mask, and a u0 R3 tie is an expected null property"})
    write_json(out / "run_terminal.json", {
        "schema": "p9016-round050-p0-controls-terminal-v1", "status": "terminal",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_candidates": len(summary["candidates"]), "n_nulls": len(null_support),
        "wall_seconds": summary["wall_seconds"], "reference_opened": True, "phase_opened": True,
        "scope": "pure evaluation; no fit and no coordinate modification of any non-null candidate",
        "post_processing": "bootstrap and null aggregates are derived from the persisted R1/R3 tables by "
                           "code/p0_controls_post.py; no metric was recomputed",
    })
    print(json.dumps({"bootstrap_rows": len(bootstrap_rows), "nulls": len(null_support)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
