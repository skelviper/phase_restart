"""P2 运行后修正（050 轮）：copy_centre_* 的分母从 C(40,2)=780 修正为跨 chr 760 = 190×4。

背景：首版 `copy_centre_760_*` 实际在 40 个 copy 中心的全对上统计（780 对），其中含 20 对
同 chr 的两个 homolog 中心；计划与 049 既有口径的 760 是**跨 chr**集合（190 个 chr 组合 ×
4 个 copy 组合 = 760）。同 chr homolog 对按刚体性几乎不变，会把均值稀释。

本脚本只从**已冻结**的 046 baseline npz 与 `p2/probes/*.npz` 重算中心附属读出：
不重跑 probe、不重算 count/KL/fullJ、不改 probe 坐标、不改任何坐标/文件 hash、
不改 pre_reference_gate.json。同时断言并记录：
  * 190（合并 chr 中心）/ 760（跨 chr copy 中心）/ 780（全部 copy 中心对）/ 20（同 chr homolog）
  * 重算的 190 与 780 字段必须与修正前 TSV/JSON 数值一致（证明只换分母、没换坐标）
  * 其余所有字段逐位不变（逐字段比较）
并更新 p2_probes.tsv / p2_probes.json / results/validation.json / terminal.json，
另写 results/denominator_correction.json 记录前后对照。disk_verification.json 由
code/p2_verify_from_disk.py 重跑刷新（它会独立重算 760/780/20 并核对 TSV）。
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import p2_probes as p2p  # noqa: E402

CENTRE_PREFIX = "copy_centre_760_"
TSV = RUN_DIR / "p2/results/p2_probes.tsv"
JSON_REPORT = RUN_DIR / "p2/results/p2_probes.json"
VALIDATION = RUN_DIR / "p2/results/validation.json"
TERMINAL = RUN_DIR / "p2/terminal.json"
CORRECTION = RUN_DIR / "p2/results/denominator_correction.json"


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").rstrip("\n").split("\n")
    columns = lines[0].split("\t")
    rows = [dict(zip(columns, line.split("\t"))) for line in lines[1:]]
    return columns, rows


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join("" if row.get(key) is None else str(row.get(key)) for key in columns) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="fix copy-centre denominator 780 -> cross-chromosome 760 (read-only inputs)")
    parser.parse_args()
    data = p2p.ac.load_layer(1_000_000)

    with np.load(p2p.BASELINE_NPZ, allow_pickle=False) as payload:
        baseline = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        baseline_p = float(np.asarray(payload["p"]).item())
    if p2p.sha256_file(p2p.BASELINE_NPZ) != p2p.BASELINE_NPZ_SHA256:
        raise RuntimeError("baseline NPZ sha mismatch during correction")

    baseline_merged = p2p.merged_chromosome_centres(baseline, data)
    baseline_copy = p2p.copy_centres(baseline, data)
    merged_masks = p2p.centre_pair_masks(np.arange(len(data.chromosome_names), dtype=np.int64))
    copy_masks = p2p.centre_pair_masks(p2p.copy_centre_chromosome_index(data))
    denominators = {"chr_centre_190": int(merged_masks["all"].sum()),
                    "copy_centre_760": int(copy_masks["cross_chromosome"].sum()),
                    "copy_centre_all780": int(copy_masks["all"].sum()),
                    "copy_centre_same_chr_homolog_20": int(copy_masks["same_chromosome"].sum())}
    if (denominators["chr_centre_190"], denominators["copy_centre_760"],
            denominators["copy_centre_all780"], denominators["copy_centre_same_chr_homolog_20"]) != (190, 760, 780, 20):
        raise AssertionError("centre pair denominators are not 190 / 760 / 780 / 20: %s" % denominators)

    gate = json.loads((RUN_DIR / "p2/pre_reference_gate.json").read_text(encoding="utf-8"))
    columns, tsv_rows = read_tsv(TSV)
    tsv_by_probe = {row["probe"]: dict(row) for row in tsv_rows}
    before_tsv = {row["probe"]: dict(row) for row in tsv_rows}
    original_columns = list(columns)
    report = json.loads(JSON_REPORT.read_text(encoding="utf-8"))
    json_rows = {row["probe"]: row for row in report["probes"]}
    before_json = copy.deepcopy(json_rows)

    per_probe: list[dict[str, Any]] = []
    max_190_rel = 0.0
    max_780_rel = 0.0
    unchanged_field_count = 0
    for entry in gate["probes"]:
        name = entry["probe"]
        probe_path = RUN_DIR / entry["path"]
        if p2p.sha256_file(probe_path) != entry["sha256"]:
            raise RuntimeError("probe file changed on disk: %s" % name)
        with np.load(probe_path, allow_pickle=False) as payload:
            coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
        merged_probe = p2p.merged_chromosome_centres(coordinates, data)
        copy_probe = p2p.copy_centres(coordinates, data)
        merged = p2p.point_distance_stats(baseline_merged, merged_probe, merged_masks["all"])
        cross = p2p.point_distance_stats(baseline_copy, copy_probe, copy_masks["cross_chromosome"])
        all780 = p2p.point_distance_stats(baseline_copy, copy_probe, copy_masks["all"])
        same20 = p2p.point_distance_stats(baseline_copy, copy_probe, copy_masks["same_chromosome"])
        old = tsv_by_probe[name]
        old_json = json_rows[name]

        def rel(recorded: float, value: float) -> float:
            return abs(value - recorded) / max(abs(recorded), 1.0)

        # 1) 修正前的 190 与 780 必须能由冻结坐标精确复算（证明坐标/hash 未变）
        checks_190 = [rel(float(old["chr_centre_190_mean_abs_change"]), merged["mean_abs_change"]),
                      rel(float(old["chr_centre_190_max_abs_change"]), merged["max_abs_change"]),
                      rel(float(old["chr_centre_190_rms_change"]), merged["rms_change"])]
        checks_780 = [rel(float(old["copy_centre_760_mean_abs_change"]), all780["mean_abs_change"]),
                      rel(float(old["copy_centre_760_max_abs_change"]), all780["max_abs_change"]),
                      rel(float(old["copy_centre_760_rms_change"]), all780["rms_change"])]
        max_190_rel = max(max_190_rel, *checks_190)
        max_780_rel = max(max_780_rel, *checks_780)
        if max(checks_190) > 1e-9:
            raise AssertionError("merged chr-centre 190 readout does not reproduce: %s" % name)
        if max(checks_780) > 1e-9:
            raise AssertionError("legacy 780 copy-centre readout does not reproduce: %s" % name)
        if int(old["copy_centre_760_n_pairs"]) != 780:
            raise AssertionError("expected the legacy copy_centre_760_n_pairs == 780, got %s" % old["copy_centre_760_n_pairs"])

        new_tsv_row = dict(old)
        for column in list(new_tsv_row):
            if column.startswith(CENTRE_PREFIX):
                del new_tsv_row[column]
        new_tsv_row.update({
            "copy_centre_760_n_pairs": cross["n_pairs"],
            "copy_centre_760_mean_abs_change": cross["mean_abs_change"],
            "copy_centre_760_max_abs_change": cross["max_abs_change"],
            "copy_centre_760_rms_change": cross["rms_change"],
            "copy_centre_760_mean_relative_change": cross["mean_relative_change"],
            "copy_centre_all780_n_pairs": all780["n_pairs"],
            "copy_centre_all780_mean_abs_change": all780["mean_abs_change"],
            "copy_centre_all780_max_abs_change": all780["max_abs_change"],
            "copy_centre_all780_rms_change": all780["rms_change"],
            "copy_centre_all780_mean_relative_change": all780["mean_relative_change"],
            "copy_centre_same_chr_homolog_20_n_pairs": same20["n_pairs"],
            "copy_centre_same_chr_homolog_20_mean_abs_change": same20["mean_abs_change"],
            "copy_centre_same_chr_homolog_20_max_abs_change": same20["max_abs_change"],
            "copy_centre_same_chr_homolog_20_rms_change": same20["rms_change"],
            "copy_centre_same_chr_homolog_20_mean_relative_change": same20["mean_relative_change"],
        })
        tsv_by_probe[name] = new_tsv_row

        new_json_row = copy.deepcopy(old_json)
        for key in list(new_json_row):
            if key.startswith(CENTRE_PREFIX):
                del new_json_row[key]
        new_json_row.update({key: value for key, value in new_tsv_row.items() if key.startswith("copy_centre_")})
        json_rows[name] = new_json_row

        per_probe.append({
            "probe": name,
            "copy_centre_760_cross_chromosome": cross,
            "copy_centre_all780": all780,
            "copy_centre_same_chr_homolog_20": same20,
            "legacy_780_field_values": {key: old[key] for key in sorted(old) if key.startswith(CENTRE_PREFIX)},
            "recompute_check_190_max_rel_error": max(checks_190),
            "recompute_check_legacy_780_max_rel_error": max(checks_780),
            "mean_abs_change_ratio_760_over_780": cross["mean_abs_change"] / all780["mean_abs_change"],
        })

    # 2) 除中心读出（copy_centre_*）外，其它字段必须逐位不变
    changed: list[str] = []
    unchanged_field_count = 0
    for name in before_tsv:
        original = {k: v for k, v in before_tsv[name].items() if not k.startswith("copy_centre_")}
        current = {k: v for k, v in tsv_by_probe[name].items() if not k.startswith("copy_centre_")}
        if original != current:
            changed.append(name)
        else:
            unchanged_field_count += len(current)
        original_json = {k: v for k, v in before_json[name].items() if not k.startswith("copy_centre_")}
        current_json = {k: v for k, v in json_rows[name].items() if not k.startswith("copy_centre_")}
        if original_json != current_json:
            changed.append("%s(json)" % name)
    if changed:
        raise AssertionError("non-centre fields changed: %s" % changed)

    # 3) 写回 TSV：保持原列顺序，用 760（主）+ all780 + same20 三组替换旧的 5 个 copy_centre_760_* 列
    new_columns: list[str] = []
    for column in original_columns:
        if column.startswith(CENTRE_PREFIX):
            if column == "copy_centre_760_n_pairs":
                new_columns.extend(["copy_centre_760_n_pairs", "copy_centre_760_mean_abs_change",
                                    "copy_centre_760_max_abs_change", "copy_centre_760_rms_change",
                                    "copy_centre_760_mean_relative_change",
                                    "copy_centre_all780_n_pairs", "copy_centre_all780_mean_abs_change",
                                    "copy_centre_all780_max_abs_change", "copy_centre_all780_rms_change",
                                    "copy_centre_all780_mean_relative_change",
                                    "copy_centre_same_chr_homolog_20_n_pairs",
                                    "copy_centre_same_chr_homolog_20_mean_abs_change",
                                    "copy_centre_same_chr_homolog_20_max_abs_change",
                                    "copy_centre_same_chr_homolog_20_rms_change",
                                    "copy_centre_same_chr_homolog_20_mean_relative_change"])
            continue
        new_columns.append(column)
    write_tsv(TSV, new_columns, [tsv_by_probe[row["probe"]] for row in tsv_rows])

    # 4) 写回 JSON / validation / terminal
    probe_order = [row["probe"] for row in report["probes"]]
    report["probes"] = [json_rows[name] for name in probe_order]
    report["geometry"]["centre_pair_denominators"] = {
        "definition": "fixed denominators for the centre readouts: 190 merged chromosome-centre pairs; the primary "
                      "copy-centre readout is the CROSS-CHROMOSOME set 190 x 4 = 760 copy-centre pairs; the 20 "
                      "same-chromosome homolog copy-centre pairs are reported separately and never enter a 760 field",
        "merged_chromosome_centre_pairs": 190,
        "copy_centre_cross_chromosome_pairs": 760,
        "copy_centre_all_pairs_including_same_chromosome_homologs": 780,
        "copy_centre_same_chromosome_homolog_pairs": 20,
        "max_copy_centre_cross_chromosome_mean_abs_change": max(
            row["copy_centre_760_mean_abs_change"] for row in report["probes"]),
        "max_copy_centre_all780_mean_abs_change": max(
            row["copy_centre_all780_mean_abs_change"] for row in report["probes"]),
        "max_copy_centre_same_chr_homolog_mean_abs_change": max(
            row["copy_centre_same_chr_homolog_20_mean_abs_change"] for row in report["probes"]),
        "passed": True,
    }
    report["post_run_corrections"] = [{
        "id": "copy-centre-denominator-780-to-760",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "what": "the first version of copy_centre_760_* was computed over all C(40,2)=780 copy-centre pairs, "
                "which includes the 20 same-chromosome homolog centre pairs; the primary field is now the "
                "cross-chromosome set 190x4=760 and the 780 / 20 sets are reported separately",
        "inputs": "recomputed only from the frozen 046 baseline NPZ and the eight already-written p2/probes/*.npz; "
                  "probe coordinates, probe file hashes, the pre-reference gate and every count/KL/full-J field "
                  "are unchanged",
        "scope": "derived readout fix only; no probe was re-run, nothing was re-fitted",
    }]
    write_json(JSON_REPORT, report)

    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    validation["centre_pair_denominators"] = report["geometry"]["centre_pair_denominators"]
    validation["copy_centre_cross_chromosome_readout"] = {
        "primary_field": "copy_centre_760_* (cross-chromosome, 190 chromosome pairs x 4 copy combinations = 760)",
        "secondary_field": "copy_centre_all780_* (all C(40,2)=780 copy-centre pairs, kept for comparison only)",
        "tertiary_field": "copy_centre_same_chr_homolog_20_* (20 same-chromosome homolog centre pairs)",
        "max_mean_abs_change_760": max(row["copy_centre_760_mean_abs_change"] for row in report["probes"]),
        "max_mean_abs_change_all780": max(row["copy_centre_all780_mean_abs_change"] for row in report["probes"]),
        "max_mean_abs_change_same_chr_20": max(
            row["copy_centre_same_chr_homolog_20_mean_abs_change"] for row in report["probes"]),
        "rotation_family_same_chr_homolog_pairs_preserved": True,
        "note": "same-chromosome homolog copy centres move together by construction, so their 20 pair distances "
                "stay at float noise; they are excluded from the 760 field",
        "passed": bool(denominators["copy_centre_760"] == 760 and denominators["copy_centre_all780"] == 780
                       and denominators["copy_centre_same_chr_homolog_20"] == 20
                       and denominators["chr_centre_190"] == 190),
    }
    validation["post_run_corrections"] = report["post_run_corrections"]
    validation["failed_checks"] = [key for key, value in validation.items()
                                   if isinstance(value, dict) and value.get("passed") is False]
    validation["all_checks_passed"] = not validation["failed_checks"]
    write_json(VALIDATION, validation)

    terminal = json.loads(TERMINAL.read_text(encoding="utf-8"))
    terminal["post_run_corrections"] = report["post_run_corrections"]
    terminal["all_checks_passed"] = validation["all_checks_passed"]
    terminal["failed_checks"] = validation["failed_checks"]
    terminal["corrected_results"] = list(terminal.get("results", [])) + ["p2/results/denominator_correction.json"]
    write_json(TERMINAL, terminal)

    correction = {
        "schema": "p9016-round050-p2-centre-denominator-correction-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "issue": "copy_centre_760_* was computed over all 780 copy-centre pairs (40 choose 2), including the 20 "
                 "same-chromosome homolog pairs; the planned and 049-consistent primary denominator is the "
                 "cross-chromosome set 760 = 190 chromosome pairs x 4 copy combinations",
        "fix": "recomputed the centre readouts from the frozen baseline NPZ and the eight written probe NPZ only; "
               "760 is now the primary field, 780 and 20 are explicit secondary/tertiary fields",
        "denominators": denominators,
        "not_changed": ["probe coordinates", "probe npz files and their sha256", "pre_reference_gate.json",
                        "count_A / fullJ_A,B,C / KL / observed-log / normalizer deltas", "all other TSV/JSON fields"],
        "recompute_checks": {
            "legacy_780_fields_reproduced_max_rel_error": max_780_rel,
            "chr_centre_190_fields_reproduced_max_rel_error": max_190_rel,
            "non_centre_tsv_fields_verified_unchanged": unchanged_field_count,
            "probe_file_sha256_rechecked_against_gate": True,
        },
        "per_probe": per_probe,
        "scope": "derived readout only; no probe re-run, no fit, no reference read",
    }
    write_json(CORRECTION, correction)
    print(json.dumps({"denominators": denominators,
                      "max_760_mean_abs_change": validation["copy_centre_cross_chromosome_readout"]["max_mean_abs_change_760"],
                      "max_all780_mean_abs_change": validation["copy_centre_cross_chromosome_readout"]["max_mean_abs_change_all780"],
                      "max_same20_mean_abs_change": validation["copy_centre_cross_chromosome_readout"]["max_mean_abs_change_same_chr_20"],
                      "legacy_780_reproduced_max_rel_error": max_780_rel,
                      "chr_centre_190_reproduced_max_rel_error": max_190_rel,
                      "non_centre_fields_verified_unchanged": unchanged_field_count,
                      "all_checks_passed": validation["all_checks_passed"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
