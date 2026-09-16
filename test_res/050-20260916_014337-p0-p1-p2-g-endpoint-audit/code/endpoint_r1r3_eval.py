"""独立 R1/R3 评价入口（050 轮）。

设计边界（与主侧约定一致）：

* 不伪造 020 训练 selection/frozen-code 证据。本入口用**新的运行合同**（`manifest`）
  列出每个待评端点与三个冻结 baseline（fixed random / consensus / S0 oracle），
  只复用 `pr.refeval` 与 `pr.reconstruction_report.evaluate_chromosome` 的**底层指标**。
* 顺序强制：先把所有坐标文件哈希并登记到 `pre_reference_gate.json`，再 `gate.arm`，
  之后才允许读取 phase labels 与 reference 3DG。
* R1 只在同染色体、同 copy 完整标签、非对角、OFF=3 Mb metric grid 内的记录上统计；
  inter 记录不进入标签准确率。候选 accuracy / reference ceiling / oracle-fit ceiling
  在**同一共同记录集**上计算。
* `accuracy` 是 20 条染色体的 macro mean；同时给出按共同分母加权的 pooled accuracy，
  以及 pooled 分母总和。两者明确分开。
* R3 沿用 20 Mb 冻结片段划分；tie/insufficient 片段打断 run 且不进分母。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pr import genome, refeval  # noqa: E402
from pr.gate import EvalGate, sha256_file  # noqa: E402
from pr import reconstruction_report as report  # noqa: E402
from pr.reconstruction_evaluate import (  # noqa: E402
    _augment_metric_row,
    _finite_values,
    _mean,
    _metric_bins,
)

SCHEMA = "p9016-endpoint-r1r3-evaluation-v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            cells = []
            for key in columns:
                value = row.get(key)
                if value is None:
                    cells.append("NA")
                elif isinstance(value, float):
                    cells.append("%.12g" % value)
                else:
                    cells.append(str(value))
            handle.write("\t".join(cells) + "\n")


def build_grid(snpfree_path: str) -> report.FullGrid:
    lengths = genome.chrom_lengths(snpfree_path)
    chromosomes = tuple(report.Chromosome(name, int(length)) for name, length in lengths)
    grid = report.FullGrid(chromosomes, report.FINAL_BIN_BP, report.FULL_GRID_ORIGIN_BP)
    if grid.n_loci != report.FROZEN_P9016["full_grid_loci"] or \
            grid.n_physical_beads != report.FROZEN_P9016["full_grid_physical_beads"]:
        raise RuntimeError("grid mismatch: %d loci / %d beads" % (grid.n_loci, grid.n_physical_beads))
    return grid


def load_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise RuntimeError("manifest schema mismatch: %r" % document.get("schema"))
    for key in ("candidates", "baselines", "raw_pairs_path", "reference_3dg_path"):
        if key not in document:
            raise RuntimeError("manifest is missing %s" % key)
    return document


def pre_reference_gate(manifest: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    """哈希并登记全部坐标；任何 SHA 不匹配都在打开参考之前中止。"""
    gate = EvalGate(str(out_dir / "gate.json"))
    registered = []
    for entry in list(manifest["candidates"]) + list(manifest["baselines"].values()):
        target = Path(entry["path"])
        if not target.is_file():
            raise RuntimeError("coordinate file is missing: %s" % target)
        actual = sha256_file(target)
        expected = entry.get("sha256")
        if expected is not None and actual != expected:
            raise RuntimeError("coordinate SHA256 mismatch for %s: %s != %s" % (target, actual, expected))
        stage = "candidate" if entry.get("role") != "evaluation_ceiling" else "baseline-oracle"
        gate.register("final-evaluation", "%s:%s" % (stage, entry["id"]), str(target))
        registered.append({"id": entry["id"], "role": entry.get("role"),
                           "path": str(target), "sha256": actual,
                           "sha256_matches_manifest": bool(expected is None or expected == actual)})
    raw_pairs = Path(manifest["raw_pairs_path"])
    reference = Path(manifest["reference_3dg_path"])
    snpfree = Path(manifest["snpfree_path"])
    evidence = {
        "schema": "p9016-endpoint-r1r3-pre-reference-gate-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "registered_coordinates": registered,
        "n_registered_coordinates": len(registered),
        "raw_pairs": {"path": str(raw_pairs), "sha256": sha256_file(raw_pairs)},
        "snpfree_pairs": {"path": str(snpfree), "sha256": sha256_file(snpfree)},
        "reference_3dg": {"path": str(reference), "sha256": sha256_file(reference)},
        "hash_only_stage": "bytes were hashed only; no phase column and no reference payload was parsed",
        "reference_bytes_hashed_before_open": True,
        "phase_opened": False,
        "reference_opened": False,
        "note": "raw pairs, SNP-free pairs and reference 3DG were hashed as bytes only; no phase column and "
                "no reference payload was parsed before every candidate coordinate was written and hashed. "
                "The declared frozen digests are compared before the gate is armed.",
    }
    write_json(out_dir / "pre_reference_gate.json", evidence)
    return {"gate": gate, "evidence": evidence}


def evaluate(manifest: Mapping[str, Any], out_dir: Path, *, label: str) -> dict[str, Any]:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = build_grid(manifest["snpfree_path"])
    gate_bundle = pre_reference_gate(manifest, out_dir)
    gate = gate_bundle["gate"]
    raw_pairs = Path(manifest["raw_pairs_path"])
    expected_raw = manifest.get("raw_pairs_sha256")
    if expected_raw is not None and gate_bundle["evidence"]["raw_pairs"]["sha256"] != expected_raw:
        raise RuntimeError("raw pairs SHA256 mismatch")
    snpfree = Path(manifest["snpfree_path"])
    expected_snpfree = manifest.get("snpfree_sha256")
    if expected_snpfree is not None and sha256_file(snpfree) != expected_snpfree:
        raise RuntimeError("SNP-free input SHA256 mismatch")
    expected_reference = manifest.get("reference_3dg_sha256")
    if expected_reference is not None and \
            gate_bundle["evidence"]["reference_3dg"]["sha256"] != expected_reference:
        raise RuntimeError("reference 3DG SHA256 mismatch")
    record_count = int(manifest.get("record_count", 0))
    contacts = genome.load_all(manifest["snpfree_path"])
    if record_count and len(contacts["ci"]) != record_count:
        raise RuntimeError("contact record count mismatch: %d" % len(contacts["ci"]))

    refeval.clear_cache()
    gate.arm(refeval.STAGE)
    gate.require(refeval.STAGE)
    a1, a2 = refeval.load_labels_two(gate, contacts, str(raw_pairs))
    lab = refeval.single_label(a1, a2)
    reference = refeval.load_reference(gate, manifest["reference_3dg_path"])
    write_json(out_dir / "post_reference_gate.json", {
        "schema": "p9016-endpoint-r1r3-post-reference-gate-v1",
        "phase_opened": True, "reference_opened": True,
        "opened_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "labels": {"n_records": int(len(lab)), "n_labelled_single_copy": int((lab >= 0).sum()),
                   "n_crosscopy_or_unknown": int((lab < 0).sum())},
        "reference_tracks": len(reference),
    })

    candidates = {}
    for entry in manifest["candidates"]:
        candidates[entry["id"]] = report.read_full_grid_coordinates(
            entry["path"], grid, entry.get("sha256"))
    baselines = {}
    for tag, entry in manifest["baselines"].items():
        # 与冻结评估器一致：baseline source 用 _read_coordinate_rows（不做完整网格强制）。
        # 014 baseline 有意省略 1 Mb / 2 Mb bin（低于 3 Mb metric grid offset，永不进入 R1/R2/R3）
        # 以及每条染色体的最后一个偏 bin，因此严格 full-grid 读取会失败；metric readout 不受影响。
        target = Path(entry["path"])
        actual_sha = sha256_file(target)
        if entry.get("sha256") is not None and actual_sha != entry["sha256"]:
            raise RuntimeError("baseline SHA256 mismatch for %s" % target)
        baselines[tag] = report._read_coordinate_rows(target)

    chromosome_names = [chromosome.name for chromosome in grid.chromosomes]
    n_bins_by_chromosome = {chromosome.name: _metric_bins(chromosome.length_bp)
                            for chromosome in grid.chromosomes}
    per_chromosome: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    fragment_rows: list[dict[str, Any]] = []
    for chromosome_index, chromosome in enumerate(grid.chromosomes):
        name = chromosome.name
        indices = np.where(contacts["cis"] & (contacts["ci"] == chromosome_index)
                           & (contacts["cj"] == chromosome_index))[0]
        b1 = ((contacts["p1"][indices] - 3_000_000) // 1_000_000).astype(np.int64)
        b2 = ((contacts["p2"][indices] - 3_000_000) // 1_000_000).astype(np.int64)
        n_metric_bins = n_bins_by_chromosome[name]
        evaluated = report.evaluate_preregistered_candidates(
            candidates=candidates,
            selected_id=manifest["candidates"][0]["id"],
            random=baselines["random"], oracle=baselines["oracle"], consensus=baselines["consensus"],
            reference=reference, chromosome_index=chromosome_index, chromosome_name=name,
            labels=lab[indices], b1=b1, b2=b2, n_metric_bins=n_metric_bins,
        )
        per_chromosome[name] = evaluated
        for candidate_id, row in evaluated["candidates"].items():
            row = _augment_metric_row(dict(row))
            r1 = row["R1"]
            denom = r1.get("paired_denominator") or {}
            selected = r1.get("selected") or {}
            r3 = ((row["R3"].get("by_candidate") or {}).get("selected") or {})
            r3_random = ((row["R3"].get("by_candidate") or {}).get("random") or {})
            r3_consensus = ((row["R3"].get("by_candidate") or {}).get("consensus") or {})
            r2_random = ((row["R2"].get("by_candidate") or {}).get("random") or {})
            r2_consensus = ((row["R2"].get("by_candidate") or {}).get("consensus") or {})
            statuses: dict[str, int] = {}
            for item in r3.get("detail", []):
                statuses[str(item.get("status"))] = statuses.get(str(item.get("status")), 0) + 1
            # 逐片段精简表：复用同一内存 detail，不额外重算；用于核验 common fragment 身份、
            # 墙数与 longest_run 的可复算性。
            random_by_bin = {int(item.get("grid_start_bin")): item
                             for item in r3_random.get("detail", [])}
            for item in r3.get("detail", []):
                start_bin = int(item.get("grid_start_bin"))
                partner = random_by_bin.get(start_bin, {})
                fragment_rows.append({
                    "candidate_id": candidate_id,
                    "chromosome": name,
                    "chromosome_index": chromosome_index,
                    "grid_start_bin": start_bin,
                    "frag_start_mb": item.get("frag_start_mb"),
                    "frag_start_bp": item.get("frag_start_bp"),
                    "status": item.get("status"),
                    "label": item.get("label"),
                    "local_score": item.get("score"),
                    "n_common_finite_pairs": item.get("n"),
                    "candidate_global_label": r3.get("global_label"),
                    "candidate_global_label_tied": r3.get("global_label_tied"),
                    "candidate_frac_consistent": r3.get("frac_consistent"),
                    "candidate_longest_run": r3.get("longest_run"),
                    "candidate_n_walls": r3.get("n_walls"),
                    "candidate_n_fragments_applicable": r3.get("n_fragments_applicable"),
                    "fixed_random_status": partner.get("status"),
                    "fixed_random_label": partner.get("label"),
                    "fixed_random_local_score": partner.get("score"),
                    "in_common_with_fixed_random": bool(item.get("label") is not None
                                                        and partner.get("label") is not None),
                })
            rows.append({
                "candidate_id": candidate_id,
                "chromosome": name,
                "chromosome_index": chromosome_index,
                "n_cis_records": int(len(indices)),
                "r1_n_records": denom.get("n_records"),
                "r1_n_common": denom.get("n_common"),
                "r1_n_label_or_crosscopy_excluded": denom.get("n_label_or_crosscopy_excluded"),
                "r1_n_samebin_excluded": denom.get("n_samebin_excluded"),
                "r1_n_out_of_grid_excluded": denom.get("n_out_of_grid_excluded"),
                "r1_n_selected_missing_excluded": denom.get("n_selected_missing_excluded"),
                "r1_n_random_missing_excluded": denom.get("n_random_missing_excluded"),
                "r1_n_oracle_missing_excluded": denom.get("n_oracle_missing_excluded"),
                "r1_n_reference_missing_excluded": denom.get("n_reference_missing_excluded"),
                "r1_accuracy": selected.get("accuracy"),
                "r1_accuracy_policy": selected.get("accuracy_policy"),
                "r1_orientation": ((selected.get("gauge") or {}).get("orientation")),
                "r1_gauge_status": ((selected.get("gauge") or {}).get("status")),
                "r1_reference_ceiling": (r1.get("reference_ceiling") or {}).get("accuracy"),
                "r1_oracle_fit_ceiling": r1.get("oracle_fit_ceiling"),
                "r1_fixed_random": ((r1.get("random") or {}).get("accuracy")),
                "r1_fixed_random_policy": ((r1.get("random") or {}).get("accuracy_policy")),
                "r2_selected_contrast_spearman": ((row["R2"].get("by_candidate") or {})
                                                  .get("selected") or {}).get("contrast"),
                "r2_fixed_random_contrast_spearman": r2_random.get("contrast"),
                "r2_consensus_contrast_spearman": r2_consensus.get("contrast"),
                "r3_frac_consistent": r3.get("frac_consistent"),
                "r3_longest_run": r3.get("longest_run"),
                "r3_n_walls": r3.get("n_walls"),
                "r3_n_fragments_total": r3.get("n_fragments_total"),
                "r3_n_fragments_applicable": r3.get("n_fragments_applicable"),
                "r3_n_fragments_tied": r3.get("n_fragments_tied"),
                "r3_n_fragments_insufficient": r3.get("n_fragments_insufficient"),
                "r3_global_label": r3.get("global_label"),
                "r3_global_label_tied": r3.get("global_label_tied"),
                "r3_fixed_random_frac_consistent": r3_random.get("frac_consistent"),
                "r3_fixed_random_n_walls": r3_random.get("n_walls"),
                "r3_fixed_random_n_fragments_applicable": r3_random.get("n_fragments_applicable"),
                "r3_consensus_frac_consistent": r3_consensus.get("frac_consistent"),
                "r3_consensus_n_walls": r3_consensus.get("n_walls"),
                "r3_consensus_applicable": r3_consensus.get("applicable"),
                "r3_consensus_reason": r3_consensus.get("reason"),
                "r3_status_summary": ",".join("%s=%d" % kv for kv in sorted(statuses.items())),
            })

    summary: dict[str, Any] = {
        "schema": "p9016-endpoint-r1r3-summary-v1",
        "label": label,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metric_grid": {"bin_size_bp": report.FINAL_BIN_BP, "offset_bp": report.METRIC_GRID_OFFSET_BP,
                        "r1_scope": "intra-chromosomal, same-copy fully-labelled, non-diagonal, in-grid records only",
                        "r3_fragment_bp": report.FRAGMENT_BP},
        "chromosomes": chromosome_names,
        "candidate_order": [entry["id"] for entry in manifest["candidates"]],
        "candidate_roles": {entry["id"]: entry.get("role") for entry in manifest["candidates"]},
        "baselines": {tag: {"path": entry["path"], "sha256": entry["sha256"], "role": entry.get("role")}
                      for tag, entry in manifest["baselines"].items()},
        "accuracy_statistic": {
            "primary": "R1_macro_mean_all20 = unweighted mean over ALL 20 chromosomes of the per-chromosome "
                       "geometry-gauged accuracy (ties counted 0.5; the copy gauge is fixed per chromosome by "
                       "the common-finite non-diagonal four-track Spearman score). If ANY chromosome has an "
                       "undefined accuracy the primary value is None, never a mean over a shrunken set.",
            "companion": "R1_macro_mean_defined_only = the same mean restricted to chromosomes with a finite "
                         "accuracy, always reported together with n_defined_chromosomes and the explicit list "
                         "of undefined chromosomes.",
            "pooled": "R1_pooled_over_defined_common_records = sum(accuracy_c * n_common_c) / sum(n_common_c) "
                      "over chromosomes with a finite accuracy. It is a record-pooled number, not the primary "
                      "statistic, and it excludes undefined chromosomes by construction; that exclusion is "
                      "declared, never silent.",
            "denominator": "R1_denominator_total_declared20 sums n_common over all 20 chromosomes (an undefined "
                           "chromosome contributes 0) and is a pooled denominator sum, not an accuracy.",
            "undefined_rule": "a chromosome whose gauge is unavailable (insufficient common finite pairs) keeps "
                              "accuracy=None: it is excluded from R1_macro_mean_defined_only and from the pooled "
                              "numerator/denominator, is listed in R1_undefined_chromosomes, and makes "
                              "R1_macro_mean_all20 None. 0.5 is never substituted for a missing value.",
        },
        "candidates": {},
        "reference_opened": True,
        "phase_opened": True,
    }

    total_cis = int(sum(row["n_cis_records"] for row in rows
                        if row["candidate_id"] == manifest["candidates"][0]["id"]))
    for entry in manifest["candidates"]:
        candidate_id = entry["id"]
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        accuracies = [row["r1_accuracy"] for row in candidate_rows]
        denominators = [row["r1_n_common"] for row in candidate_rows]
        finite_pairs = [(float(a), int(n)) for a, n in zip(accuracies, denominators)
                        if a is not None and n is not None and np.isfinite(float(a))]
        undefined = [row["chromosome"] for row in candidate_rows
                     if row["r1_accuracy"] is None or row["r1_n_common"] is None]
        pooled_denominator = int(sum(n for _a, n in finite_pairs))
        pooled_accuracy = (float(sum(a * n for a, n in finite_pairs) / pooled_denominator)
                           if pooled_denominator else None)
        r3_frac = _macro_all20([row["r3_frac_consistent"] for row in candidate_rows])
        r3_walls = _macro_all20([row["r3_n_walls"] for row in candidate_rows])
        r3_longest = _macro_all20([row["r3_longest_run"] for row in candidate_rows])
        r2_contrast = _macro_all20([row["r2_selected_contrast_spearman"] for row in candidate_rows])
        ref_ceiling = _macro_all20([row["r1_reference_ceiling"] for row in candidate_rows])
        oracle_ceiling = _macro_all20([row["r1_oracle_fit_ceiling"] for row in candidate_rows])
        r3_pooled = _pooled_r3(per_chromosome, candidate_id)
        summary["candidates"][candidate_id] = {
            "role": entry.get("role"),
            "coordinates_path": entry["path"],
            "coordinates_sha256": entry["sha256"],
            "R1_macro_mean_all20": (_mean(accuracies) if not undefined else None),
            "R1_macro_mean_all20_defined": not undefined,
            "R1_macro_mean_defined_only": _mean(accuracies),
            "R1_defined_chromosomes": len(finite_pairs),
            "R1_chromosomes_total": len(chromosome_names),
            "R1_undefined_chromosomes": undefined,
            "R1_pooled_over_defined_common_records": pooled_accuracy,
            "R1_pooled_denominator_records": pooled_denominator,
            "R1_denominator_total_declared20": int(sum(int(n) for n in denominators if n is not None)),
            "R1_reference_ceiling_macro_mean_all20": ref_ceiling["all20"],
            "R1_reference_ceiling_macro_mean_defined_only": ref_ceiling["defined_only"],
            "R1_oracle_fit_ceiling_macro_mean_all20": oracle_ceiling["all20"],
            "R1_oracle_fit_ceiling_macro_mean_defined_only": oracle_ceiling["defined_only"],
            "R1_per_chromosome": {row["chromosome"]: row["r1_accuracy"] for row in candidate_rows},
            "R1_denominator_per_chromosome": {row["chromosome"]: row["r1_n_common"] for row in candidate_rows},
            "R2_contrast_spearman_macro_mean_all20": r2_contrast["all20"],
            "R2_contrast_spearman_macro_mean_defined_only": r2_contrast["defined_only"],
            "R2_contrast_defined_chromosomes": r2_contrast["n_defined"],
            "R3_frac_consistent_macro_mean_all20": r3_frac["all20"],
            "R3_frac_consistent_macro_mean_defined_only": r3_frac["defined_only"],
            "R3_frac_consistent_defined_chromosomes": r3_frac["n_defined"],
            "R3_longest_run_macro_mean_all20": r3_longest["all20"],
            "R3_longest_run_macro_mean_defined_only": r3_longest["defined_only"],
            "R3_n_walls_macro_mean_all20": r3_walls["all20"],
            "R3_n_walls_macro_mean_defined_only": r3_walls["defined_only"],
            "R3_fragments_total": int(sum(int(r["r3_n_fragments_total"] or 0) for r in candidate_rows)),
            "R3_fragments_applicable": int(sum(int(r["r3_n_fragments_applicable"] or 0) for r in candidate_rows)),
            "R3_fragments_tied": int(sum(int(r["r3_n_fragments_tied"] or 0) for r in candidate_rows)),
            "R3_fragments_insufficient": int(sum(int(r["r3_n_fragments_insufficient"] or 0) for r in candidate_rows)),
            "R3_status_counts": _status_totals(per_chromosome, candidate_id),
            "R3_frac_consistent_pooled_over_applicable_fragments": r3_pooled,
            "R3_pooled_definition": "sum(frac_consistent_c * n_fragments_applicable_c) / "
                                    "sum(n_fragments_applicable_c); None when no chromosome has a resolvable "
                                    "fragment. A chromosome whose majority label is tied contributes its own "
                                    "0.5 (refeval.r3_fragments), never 0.",
            "R3_n_walls_definition": "walls are counted only between consecutive resolvable fragments; "
                                     "a tied/insufficient/missing fragment resets the run and never bridges "
                                     "a wall count across it",
        }
    summary["denominator_note"] = (
        "R1 denominators are per-chromosome pooled sums of the joint common support of "
        "candidate + fixed random + oracle + reference on fully-labelled same-copy non-diagonal in-grid "
        "records. The per-candidate total is therefore candidate-dependent by construction; the empty or "
        "shrunken chromosome is reported as NA, never silently dropped from the 20-chromosome declaration.")
    summary["total_cis_records"] = total_cis
    summary["n_raw_records"] = int(len(contacts["ci"]))
    summary["wall_seconds"] = float(time.time() - started)
    summary["gate"] = gate_bundle["evidence"]

    write_tsv(out_dir / "r1_r3_per_chromosome.tsv", rows)
    write_tsv(out_dir / "r3_fragments.tsv", fragment_rows)
    write_json(out_dir / "r1_r3_summary.json", summary)
    return {"summary": summary, "rows": rows, "fragment_rows": fragment_rows,
            "per_chromosome": per_chromosome}


def _status_totals(per_chromosome: Mapping[str, Any], candidate_id: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    for evaluated in per_chromosome.values():
        row = evaluated["candidates"].get(candidate_id)
        if row is None:
            continue
        r3 = ((row.get("R3") or {}).get("by_candidate") or {}).get("selected") or {}
        for item in r3.get("detail", []):
            key = str(item.get("status"))
            totals[key] = totals.get(key, 0) + 1
    return totals


def _macro_all20(values: Sequence[Any]) -> dict[str, Any]:
    """主口径 all-20 macro：任一 chromosome 未定义则整体 None；同时给 defined-only。"""
    finite = _finite_values(values)
    n_total = len(values)
    undefined = [index for index, value in enumerate(values)
                 if value is None or not _is_finite(value)]
    return {
        "all20": (float(np.mean(finite)) if finite and not undefined and n_total == len(finite) else None),
        "defined_only": (float(np.mean(finite)) if finite else None),
        "n_defined": len(finite),
        "n_total": n_total,
    }


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _pooled_r3(per_chromosome: Mapping[str, Any], candidate_id: str) -> float | None:
    """按可解析片段数加权的 pooled 一致比例。

    ``r3_fragments`` 对每条染色体返回 ``frac_consistent = max(n0, n1) / n_valid``，
    多数方向平局时即 0.5。pooled 值因此必须用 Σ(frac_c * n_valid_c) / Σ n_valid_c，
    不能把平局染色体记 0。全部染色体都没有可解析片段时返回 None。
    """
    numerator = 0.0
    denominator = 0
    for evaluated in per_chromosome.values():
        row = evaluated["candidates"].get(candidate_id)
        if row is None:
            continue
        r3 = ((row.get("R3") or {}).get("by_candidate") or {}).get("selected") or {}
        frac = r3.get("frac_consistent")
        n_valid = r3.get("n_fragments_applicable")
        if frac is None or not n_valid:
            continue
        numerator += float(frac) * int(n_valid)
        denominator += int(n_valid)
    return (float(numerator / denominator) if denominator else None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="p0")
    args = parser.parse_args(argv)
    manifest = load_manifest(Path(args.manifest))
    result = evaluate(manifest, Path(args.out), label=args.label)
    summary = result["summary"]
    print(json.dumps({
        "out": args.out,
        "R1_macro_all20": {cid: row["R1_macro_mean_all20"] for cid, row in summary["candidates"].items()},
        "R1_macro_defined_only": {cid: row["R1_macro_mean_defined_only"]
                                  for cid, row in summary["candidates"].items()},
        "R1_pooled_over_defined": {cid: row["R1_pooled_over_defined_common_records"]
                                   for cid, row in summary["candidates"].items()},
        "R1_denominator_declared20": {cid: row["R1_denominator_total_declared20"]
                                      for cid, row in summary["candidates"].items()},
        "R1_reference_ceiling_all20": {cid: row["R1_reference_ceiling_macro_mean_all20"]
                                       for cid, row in summary["candidates"].items()},
        "R1_oracle_ceiling_all20": {cid: row["R1_oracle_fit_ceiling_macro_mean_all20"]
                                    for cid, row in summary["candidates"].items()},
        "R3_frac_all20": {cid: row["R3_frac_consistent_macro_mean_all20"]
                          for cid, row in summary["candidates"].items()},
        "wall_seconds": summary["wall_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
