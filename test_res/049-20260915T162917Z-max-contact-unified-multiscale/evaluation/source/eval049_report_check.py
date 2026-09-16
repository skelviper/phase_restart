#!/usr/bin/env python
"""049 报告生成器的隔离自检：用与 evaluator 相同 schema 的合成 results 生成 REPORT.md，不读 reference。"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval049_lib as lib  # noqa: E402
import eval049_report as report  # noqa: E402

OUT = lib.EVAL / "codecheck/report"


def metric_block(matched: float, cross: float, defined: bool = True) -> dict[str, float | None]:
    fields = ("copy_A_margin", "copy_B_margin", "matched_mat_margin", "matched_pat_margin", "min_margin")
    block = {"matched": matched, "cross": cross, "contrast": matched - cross, "direct": matched, "swapped": cross}
    for field in fields:
        block[field] = 0.01
    if not defined:
        block.update({field: None for field in fields})
    return block


def make_dataset(index: int, kind: str, defined: bool = True) -> dict:
    matched = 0.20 + 0.01 * index
    cross = matched - 0.2
    r2 = {
        "per_chromosome": [], "denominator_chromosomes": 20,
        "macro_full_20_all_chromosomes_required": {
            "pearson": metric_block(matched, cross, defined),
            "spearman": metric_block(matched - 0.01, cross - 0.01, defined)},
        "macro_full_20_defined": {metric: {field: defined for field in
                                           ("matched", "cross", "contrast", "direct", "swapped", "copy_A_margin",
                                            "copy_B_margin", "matched_mat_margin", "matched_pat_margin", "min_margin")}
                                  for metric in ("pearson", "spearman")},
        "macro_equal_chromosome_weight_defined_only": {
            "pearson": metric_block(matched, cross, True), "spearman": metric_block(matched, cross, True)},
        "defined_chromosome_counts": {metric: {field: 20 for field in ("matched", "cross", "contrast")}
                                      for metric in ("pearson", "spearman")},
        "undefined_chromosomes_by_field": {},
    }
    centers = {"pearson": 0.5 + 0.005 * index, "spearman": 0.5 + 0.005 * index, "normalized_stress": 0.9 - 0.005 * index,
               "optimal_single_scale": 1.0, "per_chr_profile_macro_pearson": 0.4, "per_chr_profile_macro_spearman": 0.4,
               "n_centers": 20, "n_distances": 190, "per_chr_19_distance_profile_pearson": {},
               "per_chr_19_distance_profile_spearman": {}, "top3_neighbors": {"mean_overlap_top3": 1.5}}
    primary = {"pearson": 0.45, "spearman": 0.46, "normalized_stress": 0.85, "n_distances": 760,
               "optimal_single_scale": 1.0,
               "procrustes_proper": {"rotation_det": 1.0, "normalized_aligned_rmsd": 0.7},
               "procrustes_reflection": {"rotation_det": -1.0, "normalized_aligned_rmsd": 0.6}}
    return {
        "kind": kind, "terminal": "budget_not_converged" if index % 2 else "converged",
        "outer_fg": 1502, "wall_s": 100.0 + index, "canonical_grad": 1e-4,
        "p": 0.93, "q": 2.66, "q_source": "q_from_p(p_init=0.75)",
        "loss": "A" if kind == "formal_fit" else None,
        "own_count_nll": 9.5 + 0.001 * index, "own_full_j": 9.6 + 0.001 * index,
        "common_G_count": 9.4 + 0.001 * index, "common_G_fullJ": 9.55 + 0.001 * index,
        "npz_path": "codecheck/coords/dummy/1Mb.npz",
        "r2": r2,
        "inter": {"pearson": 0.6 + 0.001 * index, "spearman": 0.65 + 0.001 * index,
                  "eligible_locus_pairs": 2835152, "denominator": 11340608, "valid_bin_count": 2447,
                  "nonfinite_value_count": 0, "status": "ok",
                  "per_order_statistic": {"order_%d" % k: {"pearson": 0.5 + 0.01 * k} for k in range(4)}},
        "spatial": {"merged_chr_centers": centers,
                    "label_permutation_null": {"one_sided_p_ge_observed": 0.001, "draws": 9999,
                                               "spearman_null": {"mean": 0.0}},
                    "copy_centers": {"primary": primary, "unresolved_alternatives": None,
                                     "unresolved_chromosomes": ["chrX"] if index == 3 else []}},
        "geometry": {"whole_cell_Rg": 0.5},
    }


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    order = list(lib.EXPECTED_FIT_IDS) + ["initial-consensus", "initial-random",
                                          "baseline-046-real-extension-G-full-J"]
    datasets = {}
    for index, dataset_id in enumerate(order):
        kind = ("formal_fit" if dataset_id in lib.EXPECTED_FIT_IDS
                else "initial_control" if dataset_id.startswith("initial-") else "baseline")
        datasets[dataset_id] = make_dataset(index, kind, defined=(dataset_id != "initial-random"))
    evaluation = {
        "schema": "p9016-049-evaluation-v1", "smoke": False, "run_id": lib.RUN.name,
        "endpoint_common_g": [{"candidate_id": dataset_id, "kind": datasets[dataset_id]["kind"],
                               "terminal": datasets[dataset_id]["terminal"], "loss": datasets[dataset_id]["loss"],
                               "own_count_nll": datasets[dataset_id]["own_count_nll"],
                               "own_full_j": datasets[dataset_id]["own_full_j"],
                               "common_G_count": datasets[dataset_id]["common_G_count"],
                               "common_G_fullJ": datasets[dataset_id]["common_G_fullJ"]}
                              for dataset_id in order],
        "created_utc": lib.utc_now(), "datasets": datasets,
        "reference": {"reference_first_opened_utc": "2026-09-15T00:00:00+00:00", "reference_sha256": "a" * 64,
                      "track_mode": "reference", "nonfinite_loci": 0, "rows": 5290},
        "mask": {"support_statement": "common 2447 valid loci = 4894 beads", "totals": {}, "per_chromosome": []},
        "aggregate": {}, "gate": {}, "bootstrap_indices": {},
        "null_summary": [
            {"source_candidate_id": source, "null_kind": kind, "draw_count": 1 if kind == "u_zero" else 16,
             "metric": metric, "defined_draws": 1 if kind == "u_zero" else 16,
             "mean": 0.3, "min": 0.25, "max": 0.35, "range": 0.1, "std": 0.01}
            for source in ("A-raw-consensus", "initial-consensus")
            for kind in ("u_zero", "random_u")
            for metric in ("pearson_matched", "pearson_cross", "pearson_contrast", "inter_pearson")],
        "null_draw_count": 153,
        "comparisons": [{"label": "endpoint-minus-baseline", "left": "A-raw-consensus",
                         "right": "baseline-046-real-extension-G-full-J",
                         "metrics": {"pearson:matched": {"mean": 0.01, "ci95": [-0.01, 0.03], "left_wins": 12,
                                                        "right_wins": 8, "ties": 0, "defined_chromosomes": 20},
                                     "pearson:contrast": {"mean": 0.0, "ci95": [-0.02, 0.02], "left_wins": 10,
                                                          "right_wins": 10, "ties": 0, "defined_chromosomes": 20},
                                     "spearman:matched": {"mean": 0.01, "ci95": [-0.01, 0.03], "left_wins": 12,
                                                          "right_wins": 8, "ties": 0, "defined_chromosomes": 20},
                                     "spearman:contrast": {"mean": 0.0, "ci95": [-0.02, 0.02], "left_wins": 10,
                                                           "right_wins": 10, "ties": 0, "defined_chromosomes": 20}}}],
        "null_draws": [],
    }
    validation = {"status": "PASS", "checks": {"reference_sha256_verified": True, "baseline_regression": "PASS",
                                               "null_draws_evaluated": 153, "smoke": False},
                  "baseline_regression": {"atol": 1e-12, "checked_values": 168, "mismatch_count": 0, "status": "PASS"}}
    terminal = {"evaluation_terminal": "complete", "exit_code_zero_is_not_scientific_success": True}
    bootstrap = {"comparisons": evaluation["comparisons"], "indices": {}}
    gate = {"manifest": {"sha256": "b" * 64}, "selection": {"sha256": "c" * 64},
            "nulls_new": [{"source_candidate_id": "s%d" % i} for i in range(136)],
            "nulls_reused_baseline": [{"source_candidate_id": "b%d" % i} for i in range(17)],
            "selection_checks": {"source_selection": {"A-raw": {"selected_source": "consensus", "rule": "min own count",
                                                                "margin": 1e-10}},
                                 "display": {"A": {"declared_fit_id": "A-raw-consensus",
                                                   "recomputed_argmin": "A-raw-consensus",
                                                   "tied_fit_ids": ["A-raw-consensus"]}}},
            "budget": {"deviations": [], "frozen_outer_fg_per_fit": 1502}}
    for name, payload in (("evaluation.json", evaluation), ("validation.json", validation),
                          ("terminal.json", terminal), ("paired_bootstrap.json", bootstrap)):
        lib.write_json(OUT / name, payload)
    gate_path = OUT / "gate.json"
    lib.write_json(gate_path, gate)
    text = report.build(OUT, gate_path)
    (OUT / "REPORT_selfcheck.md").write_text(text, encoding="utf-8")
    for required in ("## 1. 封存、终态与预算", "## 3. R2 结构一致性", "## 4. 染色体间排序四距",
                     "## 5. 空间读出", "## 6. null：u=0 与 16 个 random-u", "## 7. paired bootstrap",
                     "## 8. 验证与回归", "## 9. 限制与不可声称项", "## 10. 产物清单"):
        assert required in text, required
    assert "baseline-046-real-extension-G-full-J" in text
    assert "NA" in text, "未定义的 initial-random 指标必须显示为 NA"
    assert "136 个新文件" in text and "17 个" in text and "153 个 null draw" in text, "null 计数未按 136+17=153 展示"
    assert "共同原 G 重打分" in text and "common-G count" in text and "common-G fullJ" in text
    assert "545.5" in text or "100.0" in text, "wall 时间未渲染"
    assert "q_source" in text and "q_from_p(p_init=0.75)" in text
    assert text.count("|") > 200, "表格内容过少"
    print("REPORT SELFCHECK PASS: markdown built from synthetic results mirroring the evaluator schema "
          "(%d lines, no reference read)" % text.count("\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
