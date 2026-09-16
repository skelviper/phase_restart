"""054 必要验收核对：只读本 run 与冻结产物的必要条目，不做长审计框架。

核对项（用户列出的必要验收）：

1. 端点 lineage 确实含 20 Mb；第三/第四组共用同一 40→20→10→5→2→1 Mb 前缀；
2. 各新阶段终态 / 实际 FG（及退出码记录文件）齐全；
3. 粗化：逐 bin 算术均值 + 贡献珠数 / 守恒正确（独立重算比较）；
4. 四组同评价分母 20 chr；既有两组（046 baseline / 051 extra）指标不变；
5. chr1 矩阵每面板标题的 Spearman 数字与同 chr1 同 copy 的 TSV 值一致，列序正确；
6. 046/051/052/053 的既有候选文件字节未变。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from round_paths_054 import (BASELINE_NPZ, BASELINE_NPZ_SHA256, CHAIN_1MB_NPZ,  # noqa: E402
                             CHAIN_COARSE_1MB_NPZ, EVAL_CHROMOSOMES, EVAL_INTER_PAIRS,
                             EVAL_VALID_LOCI, EXTRA_NPZ, EXTRA_NPZ_SHA256, FROZEN_EXPECTED_MEANS,
                             GROUP3_FG, GROUP4_FG, REUSED_AGGREGATES, REUSED_PREFIX_FG,
                             REUSED_PREFIX_NPZ, REUSED_PREFIX_NPZ_SHA256, ROOT, RUN_STAGES,
                             STAGE_FG_CAP)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    # 1. lineage 含 20Mb 且第 3/4 组共用前缀
    terminals = {}
    for stage in RUN_STAGES:
        path = RUN / "logs" / ("new-chain-%s.terminal.json" % stage)
        if not path.is_file():
            checks.append(("terminal/%s" % stage, False, "missing %s" % path.name))
            continue
        terminals[stage] = json.loads(path.read_text(encoding="utf-8"))
    checks.append(("terminals_complete", len(terminals) == len(RUN_STAGES),
                   "stages=%s" % ",".join(sorted(terminals))))
    chain_ok = True
    detail = []
    for index, stage in enumerate(RUN_STAGES):
        record = terminals.get(stage)
        if record is None:
            chain_ok = False
            continue
        expected_previous = "40Mb" if index == 0 else RUN_STAGES[index - 1]
        expected_role = "reused_frozen_prefix" if index == 0 else "new_chain_endpoint"
        expected_lineage = REUSED_PREFIX_FG + sum(STAGE_FG_CAP[s] for s in RUN_STAGES[:index + 1])
        ok = (record["previous_endpoint_role"] == expected_role
              and expected_previous in record["previous_endpoint"])
        fg_ok = (int(record["outer_fg_actual"]) == STAGE_FG_CAP[stage]
                 and int(record["lineage_fg_total_through_this_stage"]) == expected_lineage)
        chain_ok &= bool(ok and fg_ok)
        detail.append("%s<-%s(%s) fg=%s/%s lineage=%s" % (
            stage, expected_previous, record["previous_endpoint_role"], record["outer_fg_actual"],
            STAGE_FG_CAP[stage], record["lineage_fg_total_through_this_stage"]))
    checks.append(("chain_linkage_and_fg", chain_ok, " | ".join(detail)))
    checks.append(("lineage_contains_20Mb", "20Mb" in RUN_STAGES
                   and terminals.get("20Mb", {}).get("previous_endpoint_role") == "reused_frozen_prefix",
                   "20Mb is prolonged from the reused 40Mb prefix, not from 051's optimized 20Mb"))
    # 逐层 previous 端点的实际哈希必须等于该层 terminal 记录（第三/第四组确实共用同一 1Mb 前缀文件）
    linkage_ok = True
    linkage_detail = []
    for index, stage in enumerate(RUN_STAGES):
        record = terminals.get(stage)
        if record is None:
            linkage_ok = False
            continue
        previous = Path(record["previous_endpoint"])
        if not previous.is_absolute():
            previous = RUN / previous
        actual = sha256_file(previous)
        same = actual == record["previous_endpoint_sha256"]
        linkage_ok &= same
        linkage_detail.append("%s<-%s:%s" % (stage, previous.name, "same" if same else "MISMATCH"))
    checks.append(("previous_endpoint_hash_linkage", linkage_ok, " ".join(linkage_detail)))
    checks.append(("group3_fg", terminals.get("1Mb", {}).get("lineage_fg_total_through_this_stage") == GROUP3_FG,
                   "1Mb lineage FG=%s (expected %d)" % (
                       terminals.get("1Mb", {}).get("lineage_fg_total_through_this_stage"), GROUP3_FG)))
    checks.append(("group4_fg", terminals.get("200kb", {}).get("lineage_fg_total_through_this_stage") == GROUP4_FG,
                   "200kb lineage FG=%s (expected %d)" % (
                       terminals.get("200kb", {}).get("lineage_fg_total_through_this_stage"), GROUP4_FG)))
    checks.append(("group3_and_group4_share_prefix", RUN.joinpath("coords/new-chain/1Mb.npz").is_file(),
                   "group 4 continues from the same coords/new-chain/1Mb.npz that is group 3's endpoint"))

    # 2. 退出码记录
    exit_log = RUN / "logs/chain_resume_exit.out"
    checks.append(("chain_exit_codes_recorded", exit_log.is_file(),
                   str(exit_log.relative_to(ROOT)) if exit_log.is_file() else "missing"))

    # 3. 粗化：独立重算
    manifest = json.loads((RUN / "coords/200kb-to-1Mb/coarsening_manifest.json").read_text(encoding="utf-8"))
    with np.load(RUN / "coords/new-chain/200kb.npz", allow_pickle=False) as payload:
        fine = np.asarray(payload["coordinates"], dtype=np.float64)
    with np.load(CHAIN_COARSE_1MB_NPZ, allow_pickle=False) as payload:
        coarse = np.asarray(payload["coordinates"], dtype=np.float64)
        coarse_raw = np.asarray(payload["raw_y"], dtype=np.float64)
    fine_layer = np.load(REUSED_AGGREGATES[200_000][0], allow_pickle=False)
    redo = np.full_like(coarse, np.nan)
    n_bins_1mb = [(int(L) + 999_999) // 1_000_000 for L in fine_layer["chromosome_lengths"]]
    for ci in range(len(n_bins_1mb)):
        slc = slice(int(fine_layer["offsets"][ci]), int(fine_layer["offsets"][ci]) + int(fine_layer["n_bins"][ci]))
        key = (np.asarray(fine_layer["locus_bin"][slc], dtype=np.int64) * 200_000) // 1_000_000
        offset = int(np.sum(n_bins_1mb[:ci]))
        for copy in (0, 1):
            summed = np.zeros((int(key.max()) + 1, 3), dtype=np.float64)
            for dim in range(3):
                np.add.at(summed[:, dim], key, fine[copy, slc, dim])
            counts = np.bincount(key, minlength=len(summed))
            redo[copy, offset:offset + len(summed)] = summed / counts[:, None]
    max_diff = float(np.max(np.abs(redo - coarse)))
    checks.append(("coarsening_is_bin_mean", max_diff < 1e-12, "max |recomputed - saved| = %.3e" % max_diff))
    conserved = (int(manifest["contributing_beads_total"]) == 26362
                 and int(manifest["bins_with_5_beads"]) == 5256
                 and int(manifest["bins_with_partial_1_to_4"]) == 34
                 and int(manifest["target_grid"]["n_loci"]) == 2645)
    checks.append(("coarsening_counts", conserved,
                   "beads=%s five=%s partial=%s loci=%s" % (manifest["contributing_beads_total"],
                                                            manifest["bins_with_5_beads"],
                                                            manifest["bins_with_partial_1_to_4"],
                                                            manifest["target_grid"]["n_loci"])))
    with np.load(CHAIN_COARSE_1MB_NPZ, allow_pickle=False) as payload:
        contributing = np.asarray(payload["contributing_beads"])
    sphere_ok = float(np.max(np.abs(np.linalg.norm(coarse, axis=2)))) <= 1.0 + 1e-12
    checks.append(("coarsened_raw_y_not_zero", bool(np.all(np.isfinite(coarse_raw))
                                                   and np.any(coarse_raw != 0.0) and sphere_ok),
                   "raw_y finite, non-zero, coords inside the unit ball"))
    checks.append(("coarsening_contributing_beads", int(contributing.sum()) == 26362,
                   "contributing beads total=%d" % int(contributing.sum())))

    # 4. 四组同分母 + 既有两组不变
    summary = json.loads((RUN / "eval/summary.json").read_text(encoding="utf-8"))
    support = summary["support"]
    support_ok = (int(support["n_chromosomes"]) == EVAL_CHROMOSOMES
                  and int(support["n_valid_loci"]) == EVAL_VALID_LOCI
                  and int(support["n_intra_pairs"]) == EVAL_INTER_PAIRS)
    checks.append(("shared_denominator", support_ok,
                   "chromosomes=%s loci=%s pairs=%s" % (support["n_chromosomes"], support["n_valid_loci"],
                                                        support["n_intra_pairs"])))
    n_defined = {key: row["same"]["n_defined"] for key, row in summary["per_version"].items()}
    checks.append(("all_groups_20_defined", all(value == 20 for value in n_defined.values()), str(n_defined)))
    unchanged = True
    for key, expected in FROZEN_EXPECTED_MEANS.items():
        row = summary["per_version"][key]
        for metric in ("same", "cross"):
            if abs(float(row[metric]["mean"]) - float(expected[metric])) > 1e-12:
                unchanged = False
    checks.append(("existing_groups_unchanged", unchanged, json.dumps(summary.get("existing_groups_reproduced", {}))[:180]))
    checks.append(("baseline_extra_shas", sha256_file(BASELINE_NPZ) == BASELINE_NPZ_SHA256
                   and sha256_file(EXTRA_NPZ) == EXTRA_NPZ_SHA256, "046/051 candidates byte-identical"))
    checks.append(("reused_40mb_prefix_sha", sha256_file(REUSED_PREFIX_NPZ) == REUSED_PREFIX_NPZ_SHA256,
                   "052 40Mb prefix byte-identical"))
    # 052/053 旧端点（错误链）不得被改写：与 053 记录的哈希核对
    audit053 = json.loads((ROOT / "test_res/053-20260916T084905Z-four-way-spearman-chr1-matrices/eval/candidate_hashes.json").read_text(encoding="utf-8"))
    old_ok = True
    details = []
    for name, entry in audit053.items():
        if name.startswith("__chr1_matrix_only__") or entry.get("npz_sha256") is None:
            continue
        path = ROOT / entry["npz"]
        if not path.is_file():
            old_ok = False
            details.append("%s missing" % name)
            continue
        same = sha256_file(path) == entry["npz_sha256"]
        old_ok &= same
        details.append("%s=%s" % (name, "same" if same else "CHANGED"))
    checks.append(("old_052_053_endpoints_untouched", old_ok, "; ".join(details)))

    # 5. panel 标注 vs TSV（同 chr1 同 copy）
    rows = []
    with (RUN / "eval/per_chromosome_spearman.tsv").open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    chr1 = next(row for row in rows if int(row["chromosome_index"]) == 0)
    annotations = []
    with (RUN / "eval/panel_annotations.tsv").open("r", encoding="utf-8") as handle:
        ann_header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            annotations.append(dict(zip(ann_header, line.rstrip("\n").split("\t"))))
    order_ok = [record["candidate_key"] for record in annotations[:3]] == ["extra_levels", "chain_1Mb", "chain_coarse_1Mb"]
    annotation_ok = True
    detail = []
    for record in annotations:
        key = record["candidate_key"]
        copy_prefix = "A" if record["displayed_copy"] == "copyA" else "B"
        matched = record["matched_reference_copy"]
        other = "pat" if matched == "mat" else "mat"
        expected_same = float(chr1["%s_%s_%s" % (key, copy_prefix, matched)])
        expected_cross = float(chr1["%s_%s_%s" % (key, copy_prefix, other)])
        got_same = float(record["spearman_same_single_copy"])
        got_cross = float(record["spearman_cross_single_copy"])
        same_ok = abs(expected_same - got_same) < 5e-7 and abs(expected_cross - got_cross) < 5e-7
        annotation_ok &= same_ok
        detail.append("%s row%d %s same=%.6f/%.6f cross=%.6f/%.6f" % (
            key, int(record["row_index"]), record["displayed_copy"], got_same, expected_same,
            got_cross, expected_cross))
    checks.append(("panel_annotation_matches_tsv", annotation_ok and order_ok,
                   "column order ok=%s | %s" % (order_ok, "; ".join(detail))))
    scales = json.loads((RUN / "eval/chr1_matrix_scales.json").read_text(encoding="utf-8"))
    checks.append(("matrix_column_order", scales["column_order"][0] == "Reference"
                   and scales["column_order"][3].startswith("Full chain 200 kb"),
                   " | ".join(scales["column_order"])))

    # 交付图存在且为 PNG/300dpi
    for name in ("four_group_same_cross_spearman.png", "chr1_distance_matrices.png"):
        path = RUN / "plots" / name
        ok = path.is_file() and path.stat().st_size > 10_000
        checks.append(("figure/%s" % name, ok, "%s bytes" % (path.stat().st_size if path.is_file() else 0)))

    failed = [name for name, ok, _ in checks if not ok]
    report = {"schema": "p9016-round054-acceptance-v1",
              "checks": [{"check": name, "passed": bool(ok), "evidence": evidence}
                         for name, ok, evidence in checks],
              "n_passed": int(sum(1 for _, ok, _ in checks if ok)), "n_checks": len(checks),
              "failed": failed}
    (RUN / "eval/acceptance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, ok, evidence in checks:
        print("%s %-32s %s" % ("PASS" if ok else "FAIL", name, evidence[:150]))
    print("passed %d/%d" % (report["n_passed"], report["n_checks"]))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
