"""生成058精简终态、全部arm表和中文README。"""
from __future__ import annotations

import json
from pathlib import Path
import statistics

RUN = Path(__file__).resolve().parent.parent


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def main():
    A = json.loads((RUN / "logs/A_terminal.json").read_text())
    B = json.loads((RUN / "logs/B_terminal.json").read_text())
    selection = json.loads((RUN / "results/A/selection.json").read_text())
    comps = json.loads((RUN / "evaluation/comparisons.json").read_text())["comparisons"]
    struct = json.loads((RUN / "evaluation/structural_results.json").read_text())
    phase = json.loads((RUN / "evaluation/phase_traceability.json").read_text())
    eval_terminal = json.loads((RUN / "logs/evaluation_terminal.json").read_text())
    dev = json.loads((RUN / "evaluation/dev_results.json").read_text())["results"]

    arms = []
    for row in A["arms"]:
        arms.append({"arm_id": row["arm_id"], "seed": row["seed"],
                     "candidate_id": row["candidate"]["candidate_id"], "arm": row["arm"],
                     "terminal": row["terminal"], "nfev": row["nfev"],
                     "elapsed_seconds": row["elapsed_seconds"],
                     "train_J": row["endpoint"]["value"],
                     "train_count": row["endpoint"]["components"]["count_nll_normalized"],
                     "active_maxgrad": row["endpoint"]["active_maxgrad"],
                     "full_raw_y_maxgrad": row["endpoint"]["full_raw_y_maxgrad"],
                     "q_absgrad": row["endpoint"]["q_absgrad"],
                     "dev_nll": dev[row["arm_id"]]["primary_no_K0_nll_per_offdiag"]})
    B_states = [{"endpoint_id": row["endpoint_id"], "seed": row["seed"], "role": row["role"],
                 "branch": row["metadata"]["branch"], "coordinate_fg": row["metadata"]["coordinate_fg"],
                 "p": row["p"], "train_J": row["value"],
                 "dev_nll": dev[row["endpoint_id"]]["primary_no_K0_nll_per_offdiag"]}
                for row in B["endpoints"]]

    comparisons = []
    for row in comps:
        item = dict(row)
        if row["branch"] == "A":
            chrom = row["candidate"]["chromosome"]
            left = next(x for x in struct["results"][row["control"]]["per_chromosome"]
                        if x["chromosome"] == chrom)
            right = next(x for x in struct["results"][row["candidate_endpoint"]]["per_chromosome"]
                         if x["chromosome"] == chrom)
            item["affected_chr_min_margin_delta"] = right["min_margin"] - left["min_margin"]
            control_arm = next(x for x in arms if x["arm_id"] == row["control"])
            swap_arm = next(x for x in arms if x["arm_id"] == row["candidate_endpoint"])
            ratio = max(control_arm["active_maxgrad"], swap_arm["active_maxgrad"]) / max(
                min(control_arm["active_maxgrad"], swap_arm["active_maxgrad"]), 1e-12)
            item["optimization_gradient_ratio"] = ratio
            item["optimization_imbalance_warning"] = bool(ratio > 10 and max(
                control_arm["active_maxgrad"], swap_arm["active_maxgrad"]) > 1e-6)
        comparisons.append(item)

    group_denominators = {name: {"records": value["records"],
                                 "eligible_pairs": value["eligible_pairs"]}
                          for name, value in next(iter(dev.values()))["groups"].items()}
    result = {
        "schema": "p9016-058-final-results-v1", "status": "complete",
        "decision": {"A": "reject_no_seed_passed_training_gate",
                     "B": "stop_after_profile_0_coordinate_FG_no_meaningful_gain",
                     "baseline_changed": False,
                     "claim": "no candidate improvement; identity_unverified; L2 remains unproven"},
        "A": {"training_fg": A["training_fg"], "arms": arms,
              "selection": selection["selections"],
              "comparisons": [row for row in comparisons if row["branch"] == "A"],
              "total_optimizer_seconds": sum(row["elapsed_seconds"] for row in arms)},
        "B": {"coordinate_triggered": B["coordinate_triggered"],
              "coordinate_training_fg": B["training_fg"],
              "cache_equivalents": B["auxiliary_equivalents"],
              "diagnostics": [{"seed": row["seed"], "p_before": row["original"]["p"],
                               "p_after": row["profiled"]["p"], "train_J_gain": row["J_gain"],
                               "train_count_gain": row["count_gain"],
                               "dJ_dp_before": row["original"]["derivative"],
                               "dJ_dp_after": row["profiled"]["derivative"],
                               "dJ_dq_before": row["original"]["components"]["dJ_dq"],
                               "dJ_dq_after": row["profiled"]["components"]["dJ_dq"],
                               "scalar_unique_calls": row["scalar_unique_calls"],
                               "numeric_stop": row["numeric_stop"]}
                              for row in B["diagnostics"]],
              "states": B_states,
              "comparisons": [row for row in comparisons if row["branch"] == "B"],
              "cache_build_seconds": sum(row["cache_build"]["elapsed_seconds"]
                                         for row in B["diagnostics"])},
        "evaluation": {"group_denominators": group_denominators,
                       "phase_traceability": phase,
                       "auxiliary_equivalents": eval_terminal["nonreserve_auxiliary_equivalents"],
                       "auxiliary_cap": eval_terminal["cap"],
                       "remaining_validation_reserve": eval_terminal["remaining_validation_reserve"],
                       "elapsed_seconds": eval_terminal["elapsed_seconds"]},
        "budgets": {"training_FG_actual": A["training_fg"] + B["training_fg"],
                    "training_FG_cap": 1400,
                    "auxiliary_equivalents_actual": eval_terminal["nonreserve_auxiliary_equivalents"],
                    "auxiliary_equivalents_cap": 64},
    }
    dump(RUN / "FINAL_RESULTS.json", result)

    lines = [
        "# Run 058 最终结果", "", "## 结论", "",
        "1. **A 不通过。** 四个 Swap 在 100 FG 后的训练 J 和 count 都差于各自 Control；两个 seed 均无候选通过训练门，因此没有按 dev 重选。所有 8 个有限末态仍完成了 dev、分层和结构评价。",
        "2. **B 不触发坐标优化。** 两 seed 的 fixed-x p profile 训练 J gain 分别为 `%.3g`、`%.3g`，均低于 `1e-5`，所以按冻结规则以 0 coordinate FG 结束。" % (result["B"]["diagnostics"][0]["train_J_gain"], result["B"]["diagnostics"][1]["train_J_gain"]),
        "3. **没有可采纳改善。** A 的四个 dev gain 全为负；B Profile 的描述性 dev gain为 `%.3g`、`%.3g`，远低于 `0.001` 且方向不一致。baseline 不变。" % (result["B"]["comparisons"][0]["dev_gain"], result["B"]["comparisons"][1]["dev_gain"]),
        "4. **身份未验证。** raw phase 与 SNP-free 七列逐行 `1,703,888/1,703,888` 对齐，但没有独立 provenance 能区分 direct SNP 与 inferred/imputed hard labels。`R1_direct` 和身份退化 veto 均为 `NA`；不能据此声称身份恢复。L2 仍未证明。",
        "", "## A 全部终态", "",
        "| Arm | Terminal | FG | Train J | Active maxgrad | Dev NLL |", "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in arms:
        lines.append("| `%s` | %s | %d | %.9f | %.3g | %.9f |" % (
            row["arm_id"], row["terminal"], row["nfev"], row["train_J"],
            row["active_maxgrad"], row["dev_nll"]))
    lines += ["", "训练选择：seed 560101 与 560102 均为 `training_gate_failed / selected=null`。所有 A 臂均有限但为 `budget_not_converged`。仅 seed 560101 candidate1 出现 active-gradient ratio >10 的 fixed-budget imbalance warning；它不改变已失败的候选门，也不能证明固有结构或全局最优。", "",
              "## B 诊断", "", "| Seed | p before | p after | Train J gain | Dev gain | Scalar calls |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for diag, comp in zip(result["B"]["diagnostics"], result["B"]["comparisons"]):
        lines.append("| %d | %.9f | %.9f | %.3g | %.3g | %d |" % (
            diag["seed"], diag["p_before"], diag["p_after"], diag["train_J_gain"],
            comp["dev_gain"], diag["scalar_unique_calls"]))
    lines += ["", "四个 B 状态是 fixed-x Original/Profile 描述性状态，不是 3×50 FG 策略对照。坐标完全相同，因此结构 same/cross/contrast/min-margin delta 均为 0。", "",
              "## Dev 分层", "", "固定分母：`Noff=253067`。`bin_anchor_genomic_separation` 三个 cis 组精确覆盖全部 cis offdiag：", "",
              "| Group | Records | Eligible pairs |", "| --- | ---: | ---: |"]
    for name in ("cis_1_5Mb", "cis_5_20Mb", "cis_20Mb_inf", "inter"):
        d = group_denominators[name]; lines.append(f"| `{name}` | {d['records']} | {d['eligible_pairs']} |")
    lines += ["", "A 四个 Swap 的 global dev gain 均为负；每个比较的近、中、远、inter contribution 见 `evaluation/comparisons.json`。B 两 seed 的 cis contribution 与 inter contribution 方向相抵，净变化接近 0，不能解释为稳定的近距或远距收益。", "",
              "## 结构与身份", "", "055 公共支持为 20 chr、157,529 pairs；每 endpoint/chr 只用整条染色体 Pearson 几何确定一次 mapping。A 的 all20 same delta 介于 `-0.00697` 与 `+0.000032`，affected-chr min-margin 与完整 cross/contrast 见 `FINAL_RESULTS.json`。四个 Swap 的 fixed-frame return-to-Control 均不满足 `near_control`，但这不挽救训练/dev 门失败。", "", "phase 对齐后的 dev 统计：cis offdiag `139560`；已知同-copy `40939`，其中远距 `16542`、`10320` 个 distinct bin-pair blocks；已知 cross-copy `1055`，未知 `97566`。由于 direct 来源未证，未执行 block bootstrap，不用推断标签替代。", "",
              "## 预算与时间", "", "- Training FG：`800/1400`（A 800，B 0）。", "- Extra full-grid equivalent：`14/64`（B cache 2，12 endpoint dev 12），保留 50。", "- A optimizer 累计：`%.2f s`；B cache build：`%.2f s`；统一评价：`%.2f s`。" % (result["A"]["total_optimizer_seconds"], result["B"]["cache_build_seconds"], result["evaluation"]["elapsed_seconds"]), "", "## 主要证据", "", "- `config.json`：最终冻结方法。", "- `preformal_freeze.json`：正式运行前 fixture 与 config hash。", "- `results/pre_evaluation_hash_gate.json`：12 states / 36 artifacts 的评价前哈希门。", "- `FINAL_RESULTS.json`：机器可读终态、比较、预算与决定。", "- `evaluation/dev_results.json`、`evaluation/structural_results.json`、`evaluation/phase_traceability.json`：原始评价表。", "", "本轮不修改 046 baseline，不 commit/push，不作 L2 或全局最优声明。", ""]
    (RUN / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "A_arms": len(arms),
                      "B_states": len(B_states), "training_FG": result["budgets"]["training_FG_actual"],
                      "aux": result["budgets"]["auxiliary_equivalents_actual"]}))


if __name__ == "__main__":
    main()
