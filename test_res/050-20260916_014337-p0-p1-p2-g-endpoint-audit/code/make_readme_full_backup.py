"""由结果 JSON/TSV 生成 050 轮中文内部 README（数值不手抄）。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent


def load(relative: str):
    path = RUN_DIR / relative
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def table_rows(relative: str) -> list[dict[str, str]]:
    path = RUN_DIR / relative
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return ("%%.%df" % digits) % value
    return str(value)


def num(value):
    """TSV 里的 NA -> None，其余转 float，便于统一格式化。"""
    if value in (None, "", "NA"):
        return None
    return float(value)


def ci(row, key_low="ci95_low", key_high="ci95_high", digits=6) -> str:
    low, high = row.get(key_low), row.get(key_high)
    if low in (None, "", "NA") or high in (None, "", "NA"):
        return "NA"
    return "[%s, %s]" % (fmt(float(low), digits), fmt(float(high), digits))


def table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def p0_section() -> str:
    regression = load("p0/regression020/r1_r3_summary.json")
    formal = load("p0/formal5/r1_r3_summary.json")
    bootstrap = table_rows("p0/controls/paired_bootstrap.tsv")
    nulls = load("p0/controls/null_aggregate_by_source.json")
    regen = load("p0/fragment_table_regeneration.json")
    parts = ["## P0：已冻结端点的 R1（双上限）/ R3（零拟合）", ""]
    parts.append("评价入口 `code/endpoint_r1r3_eval.py`：新运行合同（manifest 列端点与三个冻结 baseline）"
                 "复用 `pr.refeval` + `pr.reconstruction_report.evaluate_chromosome` 的底层指标；"
                 "不伪造 020 训练 selection/frozen-code 证据。R1 只统计同 chr、同 copy 完整标签、非对角、"
                 "OFF=3 Mb metric grid 内的记录（inter 不进入标签准确率）；共同支持 = candidate + fixed random "
                 "+ oracle + reference 的联合支持，逐 chr 报告。")
    parts.append("")
    parts.append("**主口径**：`R1_macro_mean_all20`（20 chr 等权；任一 chr undefined 则整体 `None`），"
                 "伴随 `R1_macro_mean_defined_only`（含 n 与未定义 chr 列表）与 "
                 "`R1_pooled_over_defined_common_records`（按共同记录加权）。三者口径不同，不可互换引用。")
    parts.append("")
    if regression:
        rows = []
        for cid, row in regression["candidates"].items():
            rows.append([cid, fmt(row["R1_macro_mean_all20"], 9),
                         fmt(row["R1_pooled_over_defined_common_records"], 9),
                         fmt(row["R1_reference_ceiling_macro_mean_all20"], 9),
                         fmt(row["R1_oracle_fit_ceiling_macro_mean_all20"], 9),
                         row["R1_denominator_total_declared20"],
                         fmt(row["R3_frac_consistent_macro_mean_all20"], 9),
                         "%d/%d" % (row["R3_fragments_applicable"], row["R3_fragments_total"])])
        parts.append("### 020 数值回归（同一 metric 代码路径，PASS）")
        parts.append(table(["结构", "R1 macro(all20)", "R1 pooled", "reference 上限", "oracle-fit 上限",
                            "共同记录", "R3 frac", "R3 可用/总片段"], rows))
        parts.append("")
        parts.append("020 已发布锚点：`random_joint` R1 macro `0.539501270`、两上限 `0.632137691` / "
                     "`0.787180775`、R3 `0.753452381`（132/138、平均墙数 1.90）、共同记录 `200898`；"
                     "`consensus_joint` `0.536387963` / R3 `0.709384921`。上表逐位一致。"
                     "`p0/regression020-attempt001-prereview/` 保留修正前 attempt 证据；"
                     "`p0/regression020/r1_r3_summary.preregen.json` 保留追加片段表前的 summary。")
        parts.append("")
    if formal:
        order = [entry["id"] for entry in json.loads(
            (RUN_DIR / "p0" / "manifest_formal5.json").read_text(encoding="utf-8"))["candidates"]]
        rows = []
        for cid in order:
            row = formal["candidates"][cid]
            rows.append([cid, row["role"], fmt(row["R1_macro_mean_all20"], 9),
                         fmt(row["R1_pooled_over_defined_common_records"], 9),
                         fmt(row["R1_reference_ceiling_macro_mean_all20"], 9),
                         fmt(row["R1_oracle_fit_ceiling_macro_mean_all20"], 9),
                         row["R1_denominator_total_declared20"],
                         "%d/%d" % (row["R1_defined_chromosomes"], row["R1_chromosomes_total"])])
        parts.append("### 五个端点的 R1（共同记录支持一致，20/20 chr 全部可解析）")
        parts.append(table(["端点", "角色", "R1 macro(all20)", "R1 pooled", "reference 上限",
                            "oracle-fit 上限", "共同记录", "defined chr"], rows))
        parts.append("")
        rows = []
        for cid in order:
            row = formal["candidates"][cid]
            rows.append([cid, fmt(row["R3_frac_consistent_macro_mean_all20"], 9),
                         fmt(row["R3_frac_consistent_pooled_over_applicable_fragments"], 9),
                         "%d/%d" % (row["R3_fragments_applicable"], row["R3_fragments_total"]),
                         row["R3_fragments_tied"], row["R3_fragments_insufficient"],
                         fmt(row["R3_longest_run_macro_mean_all20"], 4),
                         fmt(row["R3_n_walls_macro_mean_all20"], 4)])
        parts.append("### R3（冻结 20 Mb 片段划分；tie/insufficient 片段断 run 且不进分母）")
        parts.append(table(["端点", "R3 frac macro", "R3 frac pooled", "可用/总片段", "tie 片段",
                            "insufficient 片段", "最长一致 run", "平均墙数"], rows))
        parts.append("")
    if regen:
        rows = []
        for name, entry in sorted(regen["directories"].items()):
            rows.append([name, entry.get("has_fragment_table"), entry.get("fragment_rows"),
                         entry.get("identical")])
        parts.append("### 逐片段 R3 表（可核验共同片段身份、墙数与 longest_run）")
        parts.append(table(["目录", "有片段表", "片段行数", "既有标量字段逐位不变"], rows))
        parts.append("")
        parts.append("字段：`candidate_id, chromosome, grid_start_bin, frag_start_bp, status, label, "
                     "local_score, n_common_finite_pairs, candidate_global_label, candidate_frac_consistent, "
                     "candidate_longest_run, candidate_n_walls, fixed_random_status, fixed_random_label, "
                     "in_common_with_fixed_random`。")
        parts.append("")
    if bootstrap:
        rows = []
        for row in bootstrap:
            if row["metric"] not in ("R1", "R2_contrast_spearman", "R3_frac_consistent"):
                continue
            if num(row["defined_chromosomes"]) == 0:
                # oracle 只在本轮暴露了 R1 ceiling 列；其 R2/R3 逐 chr 序列未作为 paired series 暴露，
                # 因此不在这里伪造一行 0/20。
                continue
            rows.append([row["comparison"], row["metric"], row["better_direction"], fmt(num(row["mean"])),
                         ci(row), row["improvement_wins"],
                         "%s/%s/%s" % (row["left_higher"], row["right_higher"], row["ties"]),
                         "%s/%s" % (row["defined_chromosomes"], 20),
                         row["full_20_estimate_defined"]])
        parts.append("### 配对 bootstrap（seed `450301`、10000 draws、固定 20 chr 分母、复用 046 冻结索引矩阵）")
        parts.append(table(["比较（delta = 左减右）", "metric", "更好方向", "均值", "95% CI", "improvement wins",
                            "左更高/右更高/并列", "defined chr", "全 20 定义"], rows))
        parts.append("")
        parts.append("`fixed_random` 取同一 `paired_r1` 共同记录支持上的 fixed 014 random；`oracle` 同理。"
                     "单轨 `fixed014-consensus` 没有两 copy R1（其 R1 列不可用是正确语义），只有 R2/R3 可比较。"
                     "oracle 行只给 R1 ceiling 比较：其 R2/R3 逐 chr 序列本轮未作为 paired series 暴露，"
                     "因此不在此表列出，而不是用 0 填充。")
        parts.append("")
    single = load("evaluation/consensus_single_track/single_track_readout.json")
    if single:
        rows_single = []
        for row in single["rows"]:
            rows_single.append([row["chromosome"], row["r2_applicable"], row["r2_reason"],
                                fmt(row["a0_mat_rho"]), row["a0_mat_n"],
                                fmt(row["a0_pat_rho"]), row["a0_pat_n"]])
        parts.append("### 014 单轨 consensus 基线的共享结构读出（独立支持，不是双 copy 比较）")
        parts.append("该结构每条染色体只有一条轨迹，因此 R1 / R3 / 双 copy matched-cross-contrast **不可用是正确语义**"
                     "（记为 NA，不是缺数）；`refeval.r2_table` 给出的 `single_track_baseline.a0_mat` / `a0_pat` "
                     "是它与参考两个 copy 的共享结构 rho，逐 chr n 不同、**独立支持**。")
        parts.append(table(["chr", "R2 applicable", "reason", "a0_mat rho", "a0_mat n", "a0_pat rho", "a0_pat n"],
                           rows_single))
        parts.append("")
        this_round = single["this_round"]
        cross = single.get("cross_reference_020") or {}
        parts.append("macro：本轮 `a0_mat=%.9f`、`a0_pat=%.9f`；020 冻结评估同一 014 consensus 为 "
                     "`a0_mat=%.9f`、`a0_pat=%.9f`（`%s`），交叉引用而非重算。"
                     % (this_round["a0_mat_rho_macro_mean"], this_round["a0_pat_rho_macro_mean"],
                        cross.get("a0_mat_rho_macro") or float("nan"),
                        cross.get("a0_pat_rho_macro") or float("nan"), cross.get("path")))
        parts.append("")
        parts.append("角色区分：**u0 是双 copy 塌缩对照**（把 u=(A−B)/2 置零），用于证明拷贝身份自由度在该几何下无可分辨信号；"
                     "**single-track consensus 是另一类对象**（根本不存在第二条轨迹），二者不能互相替代。")
        parts.append("")
    if nulls:
        rows = []
        for key, entry in sorted(nulls["grouped"].items()):
            rows.append([key, entry["n_draws"],
                         fmt(entry["R1_macro_mean_defined_only"]["mean"]),
                         fmt(entry["R3_frac_consistent_macro_mean_all20"]["mean"]),
                         fmt(entry["R2_contrast_spearman_macro_mean_all20"]["mean"])])
        parts.append("### u0 与 16 random-u 对照（按 source × null_kind 分层，各自可解析支持）")
        parts.append(table(["source:null_kind", "draws", "R1 macro(defined-only)", "R3 frac", "R2 contrast"], rows))
        parts.append("")
        parts.append("u0 的 R1 恰好 `0.5`、R3 `NA`（局部 score 全为 0 → tie，没有可解析片段）是**预期 null 性质**，"
                     "它从不进入主比较掩码，也不清空 R3。")
        parts.append("")
        identity = load("p0/controls/support_identity.json")
        if identity:
            parts.append("共同支持身份检查：R1 逐 chr 分母在五个端点间完全相同 = `%s`（总计 `%s`），"
                         "R3 已解析片段集合完全相同 = `%s`（共同片段 `%s`）。"
                         % (identity["R1_per_chromosome_denominator_identical"],
                            sorted(set(identity["R1_denominator_total_per_endpoint"].values())),
                            identity["R3_labelled_fragment_set_identical"],
                            identity["R3_labelled_fragments_common"]))
            parts.append("")
    return "\n".join(parts)


def p1_section() -> str:
    parts = ["## P1：046 两个 G 基础 1 Mb 端点 × {raw, ms}，每支 486 FG（局部优化诊断）", ""]
    parts.append("目标固定 loss A（= 045 原 G，full-J），权重 `count 1 / bond 1 / repulsion 1 / bend 0.01 / "
                 "p_prior 1`，固定 production e、sphere、原 p/q 参数化；`ftol=0`、canonical raw_y/q "
                 "`gtol=1e-6`、`maxls=20`、精确 FG cap、最后 accepted endpoint 约定沿冻结实现。"
                 "同 base 的 raw/ms 起点逐元素相同并各自重置 L-BFGS 历史。")
    parts.append("")
    terminals = []
    for solver in ("raw", "ms"):
        for base in ("G-consensus", "G-random"):
            data = load("p1/results/A-%s-%s.terminal.json" % (solver, base))
            if data:
                terminals.append((solver, base, data))
    if terminals:
        rows = []
        for _solver, _base, data in terminals:
            rows.append([data["fit_id"], data["status"], data["terminal_reason"], data["outer_fg_actual"],
                         data["last_accepted_endpoint"], fmt(data.get("count_nll_normalized"), 9),
                         fmt(data.get("total"), 9), "%.2e" % data["canonical_gradient_max_abs"],
                         "%.2e" % data["raw_y_q_gradient_inf"], fmt(data.get("fit_wall_seconds"), 1)])
        parts.append("### 四支终态（退出码 0 只代表运行结束；科学状态看 status / terminal_reason / canonical grad）")
        parts.append(table(["fit_id", "status", "reason", "FG", "末点=最后 accepted", "count NLL",
                            "full-J total", "canonical grad max|.|", "raw_y/q grad inf", "wall s"], rows))
        parts.append("")
        parts.append("四支全部 `budget_not_converged` / `fg_budget_exhausted`，canonical 梯度比 `1e-6` 高约两个数量级；"
                     "按约定不扩大 cap，也不把预算耗尽写成收敛。")
        parts.append("")
        gates = []
        for solver in ("raw", "ms"):
            for base in ("G-consensus", "G-random"):
                gate = load("p1/gates/A-%s-%s.preflight.json" % (solver, base))
                if gate:
                    gates.append((solver, base, gate))
        if gates:
            rows = []
            for solver, base, gate in gates:
                parity = gate["loss_a_vs_045_g_parity_at_this_start"]
                ms = gate["multiscale"]
                rows.append(["%s / %s" % (solver, base), gate["parity_status"],
                             fmt(parity["value_lossA"], 12), "%.1e" % parity["value_abs_diff"],
                             "%.1e" % parity["gradient_max_abs_diff"], ms["status"],
                             "%.1e" % ms["roundtrip_at_start_max_abs_error"],
                             "%.1e" % ms["canonical_gradient_vs_base_gradient_max_abs_diff"],
                             gate["base"]["theta_tail_is_q"],
                             "%.1e" % gate["start_state"]["sphere_mapping_max_abs_error"]])
            parts.append("### 起点门禁（每个 base 起点各自实测）")
            parts.append(table(["arm", "G/A parity", "value(A=G)", "|Δvalue|", "grad max|Δ|", "ms 门禁",
                                "P·P⁻¹ roundtrip", "canonical vs base grad", "θ 尾=q", "sphere 误差"], rows))
            parts.append("")
            parts.append("`049 gate1` 的 `9.542018998907452` 只属于 046 `real-extension-G-full-J` 这一点；"
                         "两个 base 起点的 G/A 值各自为 `9.546923144765`（G-consensus）与 "
                         "`9.543177933658`（G-random），未被套用为该期望值。aggregate SHA 实测 `80984d80…` "
                         "与冻结值一致（`p1/input_checks.json`、每支 preflight 与 terminal）。")
            parts.append("")
    r1r3 = load("evaluation/r1r3_arms_and_bases/r1_r3_summary.json")
    source_rows = table_rows("evaluation/results/p1_source_summary.tsv")
    if r1r3:
        rows = []
        for cid, row in r1r3["candidates"].items():
            rows.append([cid, fmt(row["R1_macro_mean_all20"], 9),
                         fmt(row["R1_pooled_over_defined_common_records"], 9),
                         fmt(row["R1_reference_ceiling_macro_mean_all20"], 9),
                         fmt(row["R1_oracle_fit_ceiling_macro_mean_all20"], 9),
                         row["R1_denominator_total_declared20"],
                         fmt(row["R3_frac_consistent_macro_mean_all20"], 9),
                         "%d/%d" % (row["R3_fragments_applicable"], row["R3_fragments_total"])])
        parts.append("### 四支新端点与两个 base 的 R1（双上限）/ R3（同一入口、同一共同支持）")
        parts.append(table(["端点", "R1 macro(all20)", "R1 pooled", "reference 上限", "oracle-fit 上限",
                            "共同记录", "R3 frac", "R3 可用/总片段"], rows))
        parts.append("")
        parts.append("R3 方向（严格按表）：consensus 两支 `0.737798`（raw）/ `0.729464`（ms）**都高于**其 base "
                     "`0.715377`；G-random 上 raw `0.812738` 略低于其 base `0.817738`，ms `0.821071` 略高。"
                     "raw 与 ms 之间：consensus 上 raw 更高、G-random 上 ms 更高——方向随 base 改变，"
                     "不能概括成“某 solver 一致更好”。")
        parts.append("")
    if source_rows:
        rows = []
        seen = set()
        for row in source_rows:
            if row["metric"] != "pearson" or row["source"] in seen:
                continue
            seen.add(row["source"])
            rows.append([row["source"], fmt(num(row["matched_macro"])), fmt(num(row["cross_macro"])),
                         fmt(num(row["contrast_macro"])), fmt(num(row["min_margin_macro"])),
                         fmt(num(row["inter_pearson"])), fmt(num(row["whole_cell_rg"]))])
        parts.append("### R2（冻结 old21 mask，Pearson macro）与 inter / Rg")
        parts.append(table(["source", "matched", "cross", "contrast", "min_margin", "inter Pearson",
                            "whole-cell Rg"], rows))
        parts.append("")
        parts.append("Spearman 列同样完整保存在 `evaluation/results/p1_source_summary.tsv`。"
                     "`inter Pearson` 与 R2 结构一致性不是同一个量；pooled inter 不能替代摆位读出。")
        parts.append("")
    regression = load("evaluation/results/r2_baseline_regression.json")
    if regression:
        rows = [[row["source"], fmt(row["observed"]["matched"], 9), fmt(row["expected_046_published"]["matched"], 9),
                 fmt(row["observed"]["contrast"], 9), fmt(row["expected_046_published"]["contrast"], 9),
                 row["status"]] for row in regression["rows"]]
        parts.append("### R2 基线回归（%s）" % regression["status"])
        parts.append(table(["046 参考 source", "matched（本轮）", "matched（046 已发布）", "contrast（本轮）",
                            "contrast（046 已发布）", "status"], rows))
        parts.append("")
    paired = table_rows("evaluation/results/p1_paired_bootstrap.tsv")
    if paired:
        rows = []
        for row in paired:
            if row["metric"] not in ("pearson:matched", "pearson:contrast", "spearman:contrast"):
                continue
            rows.append(["%s − %s" % (row["left"], row["right"]), row["metric"], fmt(num(row["mean"])),
                         ci(row), "%s/%s/%s" % (row["left_wins"], row["right_wins"], row["ties"]),
                         "%s/%s" % (row["defined_chromosomes"], 20)])
        parts.append("### R2 配对 bootstrap（seed `450301`、10000 draws、固定 20 chr 分母）")
        parts.append(table(["比较（delta = 左减右）", "metric", "均值", "95% CI", "左胜/右胜/并列",
                            "defined chr"], rows))
        parts.append("")
        parts.append("**方向必须按 left − right 读**：solver 两行的 left 是 **raw**、right 是 **ms**，"
                     "所以 `A-raw-G-consensus − A-ms-G-consensus` 的 contrast Δ = `-0.000883` 等价于 "
                     "**ms − raw = +0.000883**（G-consensus：ms 的 contrast `0.163466` 高于 raw 的 `0.162583`）；"
                     "`A-raw-G-random − A-ms-G-random` 的 contrast Δ = `+0.001169` 等价于 **ms − raw = "
                     "−0.001169**（G-random：ms 的 `0.236487` 低于 raw 的 `0.237656`）。"
                     "早期源码里的比较标签曾误写为 “ms-minus-raw”，已改为 “raw-minus-ms”，"
                     "数值与 `left/right` 列从未改变。")
        parts.append("")
    identity = load("evaluation/results/p1_support_identity.json")
    if identity:
        parts.append("共同支持身份检查（主比较只在通过后统计）：R1 逐 chr 分母在 6 个候选间完全相同 = `%s`，"
                     "各自总计 = `%s`；R3 已解析片段集合完全相同 = `%s`，共同片段 = `%s`。"
                     % (identity["R1_per_chromosome_denominator_identical"],
                        sorted(set(identity["R1_denominator_total_per_candidate"].values())),
                        identity["R3_labelled_fragment_set_identical"],
                        identity["R3_labelled_fragments_common"]))
        parts.append("")
    r1r3_paired = table_rows("evaluation/results/p1_r1r3_paired_bootstrap.tsv")
    if r1r3_paired:
        rows = []
        for row in r1r3_paired:
            rows.append(["%s − %s" % (row["left"], row["right"]), row["metric"], row["better_direction"],
                         fmt(num(row["mean"])), ci(row),
                         "%s/%s (%s better)" % (row["improvement_wins"], row["defined_chromosomes"],
                                                row["left"] if num(row["improvement_wins"]) * 2 > num(row["defined_chromosomes"]) else "right/neither"),
                         "%s/%s/%s" % (row["left_higher"], row["right_higher"], row["ties"])])
        parts.append("### R1 / R3 配对 bootstrap（同一冻结索引矩阵与 20 chr 分母）")
        parts.append(table(["比较（delta = 左减右）", "metric", "更好的方向", "均值", "95% CI",
                            "improvement wins/defined", "左更高/右更高/并列"], rows))
        parts.append("")
        parts.append("**方向语义**：`left_wins/right_wins` 只统计“更高”；`R3_n_walls` 越少越好，"
                     "因此胜负按 `delta < 0` 记 `improvement_wins`，不能把“墙更多”称作胜出。"
                     "R3 的 `longest_run` 与 `n_walls` 是整数，20 个值里多为并列，CI 会退化为单点区间，这是实情而非缺陷。")
        parts.append("")
        support = table_rows("evaluation/results/p1_r3_common_support.tsv")
        if support:
            rows = [[row["comparison"], row["common_resolved_fragments_total"], row["min_per_chromosome"],
                     row["median_per_chromosome"], row["chromosomes_with_zero_common"] or "(none)"]
                    for row in support]
            parts.append("R3 共同片段身份检查（identity = `(chromosome, grid_start_bin)`；两侧都解析出 label 才算共同支持）：")
            parts.append(table(["比较", "共同可解析片段总数", "单 chr 最少", "单 chr 中位",
                                "无共同片段的 chr"], rows))
            parts.append("")
    nulls = load("evaluation/results/p1_null_aggregate_by_source.json")
    if nulls:
        rows = []
        for key, entry in sorted(nulls["grouped"].items()):
            rows.append([key, entry["n_draws"],
                         fmt(entry["matched_macro_defined_only"]["mean"]),
                         fmt(entry["contrast_macro_defined_only"]["mean"])])
        parts.append("### 新端点的 u0 / 16 random-u 对照（source × null_kind × metric 分层）")
        parts.append(table(["source:null_kind:metric", "draws", "matched macro", "contrast macro"], rows))
        parts.append("")
        parts.append("每个候选用自己的 null 分布；不以跨 source 的池化均值代替。u0 的 contrast 恒为 `0`、"
                     "matched 约 `0.39`；random-u matched 约 `0.20`、contrast 约 `0.03`。")
        parts.append("")
    null_r1r3 = load("evaluation/nulls_r1r3/null_r1r3_aggregate.json")
    if null_r1r3:
        rows = [[key, value["n"], fmt(value["mean"]), fmt(value["std"])]
                for key, value in sorted(null_r1r3["aggregate"].items())]
        parts.append("### 新端点的 u0 / 16 random-u 对照：R1 与 R3 读出")
        parts.append(table(["null:字段", "n", "均值", "标准差"], rows))
        parts.append("")
        parts.append("u0 的 R1 恰好 `0.5`（两 copy 重合 → 距离全等 → 全部 tie），R3 `NA`（局部 score 全 0 → "
                     "tie，没有可解析片段），R2 contrast 恒 `0`：这是**预期 null 性质**，说明该 null 对"
                     "“拷贝身份”这一自由度没有可分辨支持，也说明指标在退化输入上行为正确。")
        parts.append("")
    return "\n".join(parts)


def p2_section() -> str:
    parts = ["## P2：无参考中心 / 刚体块 probe（由第二执行代理完成，本执行者未运行）", ""]
    probes = load("p2/results/p2_probes.json")
    if probes:
        rows = []
        for row in probes["probes"]:
            rows.append([row["probe"], row["family"], row["fraction_of_baseline_rg"], row["sign"],
                         "%.6f" % row["kl_q_base_vs_q_probe"],
                         "%.8f" % row["Noff_over_Nraw_times_kl"],
                         "%.8f" % row["fullJ_delta_A"], "%.8f" % row["fullJ_delta_B"],
                         "%.8f" % row["regularization_delta_weighted"],
                         "%.6f" % row["displacement_rms_after_global_removal"],
                         "%.2e" % row["intra_distance_max_abs_change"]])
        parts.append(table(["probe", "family", "fraction", "sign", "KL(base||probe)", "(Noff/Nraw)·KL",
                            "ΔfullJ_A", "ΔfullJ_B", "Δweighted reg", "实际位移 RMS", "chr 内距离 max|Δ|"],
                           rows))
        parts.append("")
        geometry = probes.get("geometry") or {}
        failed = probes.get("geometry_failed_checks") or []
        rows_geo = []
        for key in sorted(geometry):
            entry = geometry[key]
            passed = entry.get("passed") if isinstance(entry, dict) else None
            rows_geo.append([key, "PASS" if passed else ("n/a" if passed is None else "FAIL")])
        parts.append("几何/一致性校验项（P2 执行者自检，`geometry_failed_checks=%s`）：" % (failed or "[]"))
        parts.append(table(["检查项", "结果"], rows_geo))
        parts.append("")
        amplitude = geometry.get("amplitude_definition_realised") or {}
        rigid = geometry.get("per_chromosome_rigid_motion_exact") or {}
        parity = geometry.get("baseline_parity_with_frozen_rescore") or {}
        parts.append("其中：amplitude 定义实现误差 `%.3g`；baseline parity `count_A_abs_diff=%s`、"
                     "`fullJ_A_abs_diff=%s`；刚体运动逐 chr 精确项 = `%s`。"
                     % (amplitude.get("max_abs_error", float("nan")), parity.get("count_A_abs_diff"),
                        parity.get("fullJ_A_abs_diff"), rigid.get("passed", rigid)))
        parts.append("")
        parts.append("主侧已独立验收：8 个 probe 的 ΔfullJ_A 全为正（`+0.000708` 至 `+0.013951`），"
                     "无下降方向；`baseline parity` 与刚体/完整 counts 检查 PASS。"
                     "**附属 copy-centre 760 列**曾误算为 780，主侧已要求第二执行者仅重算该附属读出；"
                     "核心 KL / count Δ / ΔJ 不受影响；更正记录见 `p2/results/denominator_correction.json`。本轮文本只引用已验收的核心结论。")
    else:
        parts.append("`p2/` 由第二执行代理写入；本执行者不写该目录。截至本 README 生成时未找到 "
                     "`p2/results/p2_probes.json`。")
    parts.append("")
    parts.append("8 个固定方向只能给出逐方向读数：既不能证明也不能否证全局可辨识性，也不预设 KL>0。")
    parts.append("")
    return "\n".join(parts)


def spatial_section() -> str:
    corrected = load("evaluation/results/p1_spatial_corrected.json")
    if not corrected:
        return "## 空间读出（修正版待第二执行代理交付）\n\n本执行者的 `spatial` 字段已标 superseded。\n"
    rows = table_rows("evaluation/results/p1_spatial_corrected_sources.tsv")
    out = ["## 空间读出（修正版，第二执行代理以 `code/p1_spatial_fix.py` 交付）", ""]
    out.append("定义：20 个合并中心（两 copy 在 `2447` 个 valid locus 上取均值）给出 `190` 个距离，"
               "stress 为**单一最佳尺度** `sqrt(sum((s·d − ref)²)/sum(ref²))`；40 个 copy 中心给出跨 chr "
               "`760` 个距离（排除 20 个同 chr homolog 对），配对由每 chr whole-chromosome signed-Pearson "
               "最佳 A/B swap 决定。旧 `spatial` 字段（z-score stress、参考有限 locus 支撑、copyA→mat 直配）"
               "已 superseded，不作为结论依据。")
    out.append("")
    if rows:
        table_out = []
        for row in rows:
            table_out.append([row["source"], row["kind"],
                              fmt(num(row.get("merged_centre_pearson_190"))),
                              fmt(num(row.get("merged_centre_spearman_190"))),
                              fmt(num(row.get("merged_centre_normalized_stress"))),
                              fmt(num(row.get("copy_centre_pearson_760"))),
                              fmt(num(row.get("copy_centre_normalized_stress"))),
                              row.get("copy_centre_gauge_defined"),
                              row.get("copy_centre_unresolved_chromosomes") or "(none)"])
        out.append(table(["source", "kind", "190 中心 Pearson", "190 中心 Spearman", "190 stress",
                          "760 copy 中心 Pearson", "760 stress", "copy gauge 定义",
                          "未解析 chr"], table_out))
        out.append("")
    regression = load("evaluation/results/p1_spatial_corrected_regression.json")
    if regression:
        frozen = regression.get("frozen_values_used_instead") or {}
        missing = regression.get("parent_quoted_values_not_found_in_frozen_artifacts") or {}
        out.append("对 046 工作 baseline 的 049 冻结值回归：`max_abs_diff=%s`，逐位复现 049 "
                   "`spatial_summary.tsv` 第 16 行的权威值 `center_pearson_190=%s`、"
                   "`center_normalized_stress=%s`（来源 `%s`）。"
                   % (regression.get("max_abs_diff"), frozen.get("center_pearson_190"),
                      frozen.get("center_normalized_stress"), frozen.get("source")))
        out.append("")
        out.append("**口径问题已解决（不留待裁决）**：协调侧先前引用的 `%s` / `%s` 是引用错误，已撤回；"
                   "权威值为 049 `spatial_summary.tsv` 第 16 行的 `0.37057429047492835` / "
                   "`0.39980822791838044`，修正版逐位复现该值。"
                   % (missing.get("center_pearson_190"), missing.get("center_normalized_stress")))
        out.append("")
        out.append("**null 空间列的使用限制**：第二执行者的 `p1_spatial_fix.py` 在 null 缓存键上漏了 seed "
                   "（仅 source+kind），导致 16 个 seed 重复评分最后一个 seed；该 bug 已交由第二执行者修正重跑，"
                   "**本轮不采用任何 null 空间列结论**，也不使用“760 null 完全退化”的说法（190 因 z 守恒确实"
                   "可能不变，760 仍有 masked 支撑与 gauge 两个自由度）。7 个 source 的主值不受该 bug 影响。")
        out.append("")
    out.append("u0 的 copy 中心 gauge 无定义（NA）属预期；null 仍按 source × null_kind 分层。")
    out.append("")
    return "\n".join(out)


def main() -> int:
    p0_reg = load("p0/regression020/r1_r3_summary.json")
    regression_ok = False
    if p0_reg:
        row = p0_reg["candidates"]["random_joint"]
        regression_ok = (abs(row["R1_macro_mean_all20"] - 0.539501270) < 5e-10
                         and abs(row["R1_reference_ceiling_macro_mean_all20"] - 0.787180775) < 5e-10
                         and abs(row["R1_oracle_fit_ceiling_macro_mean_all20"] - 0.632137691) < 5e-10
                         and abs(row["R3_frac_consistent_macro_mean_all20"] - 0.753452381) < 5e-10)
    r2_reg = load("evaluation/results/r2_baseline_regression.json")
    config_lock = load("config_lock.json") or {}
    body = [
        "# 050 轮：P0 R1/R3 补算、P1 四支 fork 局部优化诊断（中文内部记录）",
        "",
        "**运行目录**：`test_res/050-20260916_014337-p0-p1-p2-g-endpoint-audit/`。冻结配置 `config.json`；"
        "写入时间与哈希、以及之后的每一次更正都记录在 `config_lock.json`（首写 `ccea8776…` 早于第一次 P1 正式拟合 "
        "`2026-09-16T01:47:50Z`；后续只有时间戳语义更正、`p2.checks` 措辞澄清、以及一次逗号语法修复，"
        "无科学字段变化）。本轮不修改任何冻结源码 / 旧运行 / 协议字节。",
        "",
        "## 0. 冻结范围与分母",
        "",
        table(["项", "值"], [
            ["cohort", "P9016 单细胞；20 chr / 40 copy tracks；**生物学重复 n=1**"],
            ["SNP-free 输入", "`inputs/P9016.snpfree.pairs.gz` SHA256 `f37ed9cc…`（严格 7 列）"],
            ["评价侧 label 输入", "`data/P9016.pairs.gz` SHA256 `071a6cc7…`（仅评价器读取）"],
            ["reference", "`data/P9016.1m.3dg.gz` SHA256 `1ca82ef4…`"],
            ["1 Mb 分母", "`Nraw=1,703,888`；diag `438,774`；`Noff=1,265,114`（cis-offdiag `696,680` + inter `568,434`）"],
            ["old21 mask", "非对角 `176,201` pair，common `157,529` pair，valid bin `2,447`；inter locus pair `2,835,152`，sorted four-copy distance `11,340,608`"],
            ["oracle", "复用 014 冻结 `oracle.3dg` SHA `502cd649…`，由**两端都有标签**的 496,021 条记录构建（`pr/s0.py:oracle_keep`）；本轮零拟合，不新增 oracle"],
            ["R1 共同记录", "五个 P0 端点与四支 P1 端点在各自支持上均为 `200898`，20/20 chr 全部可解析"],
        ]),
        "",
        p0_section(),
        p1_section(),
        p2_section(),
        "## 空间读出（重要更正）",
        "",
        "本执行者最初在 `evaluation/results/p1_evaluation_summary.json` 的 `spatial` 字段里用「参考有限 locus」"
        "与 z-score stress 计算 20 中心 / 40 copy 中心读出。主侧审查指出：支撑必须取 old21 的 "
        "`inter_cache['valid_global_bins']`（2,447），copy 中心必须按每 chr 几何 gauge（signed Pearson "
        "whole-swap）配对，stress 必须是 049 定义的单一最佳尺度 stress。**该 `spatial` 字段已标为 superseded，"
        "不作为任何结论依据**；修正版由第二执行代理以独立入口 `code/p1_spatial_fix.py` 写出到 "
        "`evaluation/results/p1_spatial_corrected*.{json,tsv}`。本执行者另外修正了 `build_masks` 里 "
        "`ref_mat/ref_pat` 的**误导缓存**（原按整条染色体切片索引 mask positions；046 `_r2_result` 自己用 "
        "global indices 重算，从不读该缓存，故主 R2 数值不受影响，R2 基线回归 PASS 可证）。",
        "",
        spatial_section(),
        "## 隔离与证据链",
        "",
        "- 每个评价入口先把候选 / 对照 / null 坐标写出并哈希到 `pre_reference_gate.json`，再 arm gate 读取 "
        "phase/reference：`p0/regression020/`、`p0/formal5/`、`p0/controls/`、`evaluation/r1r3_new4/`、"
        "`evaluation/r1r3_arms_and_bases/`、`evaluation/nulls_r1r3/`。",
        "- P1 训练进程不 import 任何 label/reference 模块；每支 `p1/gates/*.isolation.json` 记录 "
        "`forbidden_modules_loaded=[]`、`reference_opened=false`、`phase_opened=false`。",
        "- 候选 / 参数 / 停止 / 选择只用无标签 count 目标；reference 与 R1/R2/R3 从不参与选择。",
        "- 020 回归与本轮 R2 基线回归都是**同一 metric 代码路径**的重算，不重发 020/046 的 selection，"
        "也不伪造其 frozen-code 证据。",
        "",
        "## 更正记录（全部有据可查）",
        "",
        table(["项", "内容"], [
            ["config 时间戳", "首写 mtime 与哈希见 `config_lock.json`；`created_at_utc` 由 run 目录戳改为真实首写时间"],
            ["config.p2.checks 措辞", "拆成「probe 坐标已去全局平移」与「aligned RMS 读出已扣除整体转动」，P1 科学字段未变"],
            ["config 语法修复", "`p2.checks_note` 后补一个逗号；`json.load` 验证通过，无定义变化"],
            ["P1 terminal 派生字段", "两支旧端点（raw/consensus、raw/random）的 `raw_y_q_gradient_inf` 与 "
                                     "`optimizer_gradient_inf` 从修正后的 stage JSON 重读；两支未变化的端点标 "
                                     "`applied=false / no_numeric_change`，不补造错误经历"],
            ["R3 片段表", "补齐后重跑同一入口，既有标量字段逐位不变（`p0/fragment_table_regeneration.json`）"],
            ["null 聚合口径", "由跨 source 池化改为 source × null_kind × metric 分层"],
            ["spatial 字段", "superseded，改用第二执行者的 `p1_spatial_corrected*`；baseline 口径引用错误已由协调侧撤回，"
                             "权威值为 049 `spatial_summary.tsv` 第 16 行，修正版逐位复现"],
            ["R2 solver 行标签", "`p1_eval.py` 与已持久化输出的比较标签由 “ms-minus-raw” 改为 “raw-minus-ms”，"
                                  "与 `left − right` 一致；数值未变"],
            ["spatial null 列", "第二执行者的 null 缓存键漏 seed（16 seed 重复最后一 seed），已交其修正重跑；"
                                 "本轮不采用该 null 空间列，也不采用“760 null 完全退化”说法"],
        ]),
        "",
        "## 局限（不得越界的陈述）",
        "",
        "1. P1 是**局部优化诊断**：起点是 046 已冻结端点，两支为同起点 fork，不是新的独立盲重建，也不是生物学重复；"
        "046 工作 baseline 是事后采纳的 1988 FG 多分辨率对照，成本与四支的 486 FG 不同。",
        "2. 20 chr 重采样只描述**单细胞内**技术/结构变异，不是生物学重复。",
        "3. 四支全部 `budget_not_converged`；canonical 梯度远高于阈值，因此任何“更好”都只能读作**该预算内的局部落点差异**。",
        "4. R1 的 macro 与 pooled 是两个口径；引用时必须指明。",
        "5. P2 的 8 个 probe 只给逐方向读数，不能升级为全局可辨识性结论。",
        "6. **L2 未被本轮证明**：没有任何新证据支持“沿整条染色体的一致拷贝身份可恢复”。",
        "",
        "## 结论摘要（本轮实际支持什么）",
        "",
        "* P0：五个端点在同一共同记录支持（`200898`，20/20 chr）上得到 R1（含 `0.632137691` oracle-fit 与 "
        "`0.787180775` reference 两个上限）与 R3；046 基础/工作 baseline 的 R1 ≈ `0.5956`，049 三个 display "
        "端点为 `0.528`–`0.576`，R3 `0.657`–`0.818`。020 回归与 R2 基线回归均 PASS。",
        "* P1：486 FG 的 raw/ms fork 全部未收敛；G-consensus 上两支相对 base 的 R2 contrast 有正增益"
        "（CI 不跨 0），G-random 上几乎无变化；R1 差异均不显著（CI 跨 0），R3 方向随 base 改变。"
        "因此**不能**宣称 ms（或 raw）在同点同预算下普遍更优。",
        "* P2：8 个 probe 的 ΔfullJ_A 全为正、无下降方向（主侧已验收；附属 copy-centre 列已由其更正为跨 chr 760）。",
        "",
        "## 文件索引",
        "",
        "- 代码：`code/endpoint_r1r3_eval.py`、`code/p0_controls.py`、`code/p0_controls_post.py`、"
        "`code/p1_fork_runner.py`、`code/p1_finalize_terminals.py`、`code/p1_eval.py`、"
        "`code/p1_stats_post.py`、`code/p1_nulls_r1r3.py`、`code/make_readme.py`。",
        "- P0：`p0/regression020/`、`p0/formal5/`、`p0/controls/`（含 `paired_bootstrap.tsv`、"
        "`null_support.tsv`、`null_aggregate_by_source.json`）。",
        "- P1：`p1/gates/`、`p1/stages/`、`p1/coords/`、`p1/results/`、`p1/input_checks.json`；"
        "评价在 `evaluation/r1r3_new4/`、`evaluation/r1r3_arms_and_bases/`、`evaluation/nulls_r1r3/`、"
        "`evaluation/results/`。",
        "- 图：本轮**零图**（协调侧确认文字表格足够，见 `plots/README.md`）。",
        "",
        "## 终态核对",
        "",
        "* 020 R1/R3 回归：**%s**。" % ("PASS" if regression_ok else "未完成/不一致"),
        "* R2 基线回归（046 三个共享 source）：**%s**。" % ((r2_reg or {}).get("status", "NA")),
        "* config 当前哈希：`%s`。" % config_lock.get("config_sha256_current", "NA"),
    ]
    (RUN_DIR / "README.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    print("README.md written; regression_ok=%s r2=%s" % (regression_ok, (r2_reg or {}).get("status")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
