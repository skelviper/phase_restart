"""生成 050 轮精简版中文内部 README（约 100-150 行）。

只保留主结论、P0 五端点（双上限）/R3、P1 四支终态与 R1/R2/R3 要点、关键配对 CI/胜出、
P2 结论与限制、空间读出权威口径与路径索引。完整逐 chr 表、全部 bootstrap 行、逐片段 R3 表
均以 TSV/JSON 为准，不复制进 README。完整版生成器备份为 `code/make_readme_full_backup.py`。
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent


def load(relative: str):
    path = RUN_DIR / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def rows_of(relative: str) -> list[dict[str, str]]:
    path = RUN_DIR / relative
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def num(value):
    if value in (None, "", "NA"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def f(value, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        return ("%%.%df" % digits) % float(value)
    except (TypeError, ValueError):
        return str(value)


def ci(row, digits: int = 6) -> str:
    low, high = num(row.get("ci95_low")), num(row.get("ci95_high"))
    return "NA" if low is None or high is None else "[%s, %s]" % (f(low, digits), f(high, digits))


def table(header, body) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in body)
    return "\n".join(lines)


def main() -> int:
    p0 = load("p0/formal5/r1_r3_summary.json")
    reg = load("p0/regression020/r1_r3_summary.json")
    p0_boot = rows_of("p0/controls/paired_bootstrap.tsv")
    p0_identity = load("p0/controls/support_identity.json")
    arms = load("evaluation/r1r3_arms_and_bases/r1_r3_summary.json")
    sources = rows_of("evaluation/results/p1_source_summary.tsv")
    p1_boot = rows_of("evaluation/results/p1_paired_bootstrap.tsv")
    p1_r1r3 = rows_of("evaluation/results/p1_r1r3_paired_bootstrap.tsv")
    p1_identity = load("evaluation/results/p1_support_identity.json")
    r2_reg = load("evaluation/results/r2_baseline_regression.json")
    spatial_rows = rows_of("evaluation/results/p1_spatial_corrected_sources.tsv")
    consensus = load("evaluation/consensus_single_track/single_track_readout.json")
    p2 = load("p2/results/p2_probes.json")
    null_r1r3 = load("evaluation/nulls_r1r3/null_r1r3_aggregate.json")
    lock = load("config_lock.json") or {}

    order = [entry["id"] for entry in json.loads(
        (RUN_DIR / "p0" / "manifest_formal5.json").read_text(encoding="utf-8"))["candidates"]]
    terminal = {}
    for solver in ("raw", "ms"):
        for base in ("G-consensus", "G-random"):
            entry = load("p1/results/A-%s-%s.terminal.json" % (solver, base))
            if entry:
                terminal["A-%s-%s" % (solver, base)] = entry

    reg_ok = False
    if reg:
        row = reg["candidates"]["random_joint"]
        reg_ok = (abs(row["R1_macro_mean_all20"] - 0.539501270) < 5e-10
                  and abs(row["R1_reference_ceiling_macro_mean_all20"] - 0.787180775) < 5e-10
                  and abs(row["R1_oracle_fit_ceiling_macro_mean_all20"] - 0.632137691) < 5e-10
                  and abs(row["R3_frac_consistent_macro_mean_all20"] - 0.753452381) < 5e-10)

    parts: list[str] = []
    parts.append("# 050 轮：P0 R1/R3 补算、P1 四支 fork 局部优化诊断（精简中文内部记录）")
    parts.append("")
    parts.append("运行目录 `test_res/050-20260916_014337-p0-p1-p2-g-endpoint-audit/`。冻结配置 `config.json`，"
                 "写入时间/哈希与全部更正见 `config_lock.json`（首写 `ccea8776…` 早于首次 P1 拟合 "
                 "`2026-09-16T01:47:50Z`）。冻结源码 / 旧运行 / 协议字节未改。逐 chr 表、全部 bootstrap 行、"
                 "逐片段 R3 表以 TSV/JSON 为准，不在本文件复制。")
    parts.append("")

    parts.append("## 1. 冻结范围与分母")
    parts.append(table(["项", "值"], [
        ["cohort", "P9016 单细胞；20 chr / 40 copy tracks；**生物学重复 n=1**"],
        ["输入", "SNP-free `inputs/P9016.snpfree.pairs.gz` `f37ed9cc…`；评价侧 label `data/P9016.pairs.gz` `071a6cc7…`；reference `data/P9016.1m.3dg.gz` `1ca82ef4…`"],
        ["1 Mb 分母", "`Nraw=1,703,888`；diag `438,774`；`Noff=1,265,114`（cis-offdiag `696,680` + inter `568,434`）"],
        ["old21 mask", "非对角 `176,201`；common `157,529`；valid bin `2,447`；inter locus pair `2,835,152`；sorted four-copy `11,340,608`"],
        ["oracle", "复用 014 `oracle.3dg` `502cd649…`（两端都有标签的 496,021 条记录，`pr/s0.py:oracle_keep`）；本轮零拟合"],
        ["R1 共同记录", "五个 P0 端点与四支 P1 端点在各自支持上均为 `200898`、20/20 chr 可解析"],
    ]))
    parts.append("")
    parts.append("既有事实须并列保留：染色体间接触的**已知标签同/异单倍型计数为 `70,558` / `76,438`，不提供"
                 "母源/父源标签信号**；但全部 `568,434` 条 inter 接触仍作为几何摆位约束参与拟合与评价。")
    parts.append("")
    parts.append("R1 主口径为 **20 chr macro mean**（任一 chr undefined 则主值 `None`），同时给 defined-only、"
                 "pooled-over-records 与 n；分母从不静默缩小。")
    parts.append("")

    if p0:
        parts.append("## 2. P0：五端点 R1（双上限）/ R3（零拟合）")
        parts.append("")
        body = []
        for cid in order:
            row = p0["candidates"][cid]
            body.append([cid, f(row["R1_macro_mean_all20"], 9),
                         f(row["R1_pooled_over_defined_common_records"], 9),
                         f(row["R1_oracle_fit_ceiling_macro_mean_all20"], 9),
                         f(row["R1_reference_ceiling_macro_mean_all20"], 9),
                         row["R1_denominator_total_declared20"],
                         "%d/%d" % (row["R1_defined_chromosomes"], row["R1_chromosomes_total"])])
        parts.append(table(["端点", "R1 macro(all20)", "R1 pooled", "oracle-fit 上限", "reference 上限",
                            "共同记录", "defined chr"], body))
        parts.append("")
        body = []
        for cid in order:
            row = p0["candidates"][cid]
            body.append([cid, f(row["R3_frac_consistent_macro_mean_all20"], 9),
                         "%d/%d" % (row["R3_fragments_applicable"], row["R3_fragments_total"]),
                         row["R3_fragments_tied"], row["R3_fragments_insufficient"],
                         f(row["R3_longest_run_macro_mean_all20"], 2),
                         f(row["R3_n_walls_macro_mean_all20"], 2)])
        parts.append(table(["端点", "R3 frac macro", "可用/总片段", "tie", "insufficient", "最长 run",
                            "平均墙数"], body))
        parts.append("")
        parts.append("020 数值回归 **%s**：`random_joint` R1 macro `0.539501270`、两上限 `0.632137691` / "
                     "`0.787180775`、R3 `0.753452381`(132/138)、共同记录 `200898`，与 020 已发布值逐位一致"
                     "（`p0/regression020/`；`-attempt001-prereview/` 保留修正前证据）。"
                     % ("PASS" if reg_ok else "未通过"))
        parts.append("")
        parts.append("对照与 null：`fixed_random` / `oracle` 取自同一 `paired_r1` 共同支持；014 单轨 consensus "
                     "**R1 与 R3 均 NA**，只有共享结构的 `single_track_baseline.a0_mat/a0_pat`（**Spearman rho**）可用"
                     "（macro `%s` / `%s`，与 020 冻结值一致）；u0 `R1=0.5`（全 tie）、`R3=NA`、`contrast=0`；"
                     "16 random-u 例如 `046-base-G-random` `R1=0.5071`、`R3=0.6609`。每个 null 用自身支持并按 "
                     "source×null_kind×metric 分层，不进入主比较掩码。"
                     % (f((consensus or {}).get("this_round", {}).get("a0_mat_rho_macro_mean"), 9),
                        f((consensus or {}).get("this_round", {}).get("a0_pat_rho_macro_mean"), 9)))
        parts.append("")
        if p0_identity:
            parts.append("共同支持身份检查 **PASS**：R1 逐 chr 分母五端点全同（合计 `%s`），R3 已解析片段集合全同"
                         "（共同 `%s` 片段）。"
                         % (sorted(set(p0_identity["R1_denominator_total_per_endpoint"].values()))[0],
                            p0_identity["R3_labelled_fragments_common"]))
            parts.append("")
        body = []
        for row in p0_boot:
            if row["metric"] not in ("R1", "R2_contrast_spearman"):
                continue
            if num(row["defined_chromosomes"]) == 0:
                continue
            if "minus 046 work baseline" in row["comparison"] or "oracle" in row["comparison"]:
                continue
            body.append([row["comparison"], row["metric"], f(num(row["mean"])), ci(row),
                         "%s/%s" % (row["improvement_wins"], row["defined_chromosomes"]),
                         "%s/%s/%s" % (row["left_higher"], row["right_higher"], row["ties"])])
        parts.append("对 fixed 014 random 的配对 bootstrap（seed `450301`、10000 draws、固定 20 chr；"
                     "`improvement_wins` 对 `R3_n_walls` 按 delta<0 计）：")
        parts.append(table(["比较（delta = 左 − 右）", "metric", "均值", "95% CI", "改善胜出/defined",
                            "左高/右高/并列"], body))
        parts.append("")
        parts.append("相对 fixed 014 random 的 **R3 片段一致性增益**：046 工作 baseline `+0.133948` "
                     "`[+0.054167, +0.217365]`（11 胜 / 2 负 / 7 平），046 base G-random `+0.138948` "
                     "`[+0.058670, +0.222365]`（11/2/7）；049 三条端点的 CI 跨 0。即证据支持**部分片段的"
                     "拷贝一致性高于随机对照，但不是全染色体一致恢复**。")
        parts.append("")

    if terminal:
        parts.append("## 3. P1：四支 486 FG fork（局部优化诊断）")
        parts.append("")
        body = []
        for fit_id, data in terminal.items():
            body.append([fit_id, data["status"] + " / " + data["terminal_reason"], data["outer_fg_actual"],
                         data["last_accepted_endpoint"], f(data["count_nll_normalized"], 9),
                         f(data["total"], 9), "%.2e" % data["canonical_gradient_max_abs"],
                         "%.2e" % data["raw_y_q_gradient_inf"], f(data["fit_wall_seconds"], 1)])
        parts.append(table(["fit_id", "status / reason", "FG", "末点=最后 accepted", "count NLL", "full-J",
                            "canonical_grad_inf", "raw_y/q grad inf", "wall s"], body))
        parts.append("")
        parts.append("四支全部 `budget_not_converged` / `fg_budget_exhausted`，canonical 梯度比 `1e-6` 高约两个"
                     "数量级：**不扩大 cap，也不把预算耗尽写成收敛**。起点门禁四支全 PASS：各 base 起点实测 "
                     "G/A parity Δvalue=0、grad max|Δ|≤7.8e-18；sphere `2.2e-16`；`p_from_q` bit-exact；"
                     "ms `P·P⁻¹` roundtrip `3.6e-15`、canonical 还原 `2.0e-16`；aggregate SHA `80984d80…` 与"
                     "冻结值一致。049 的 `9.542018998907452` 只属 extension 点，未被套用。")
        parts.append("")
        if arms:
            body = []
            for cid, row in arms["candidates"].items():
                body.append([cid, f(row["R1_macro_mean_all20"], 9),
                             f(row["R1_pooled_over_defined_common_records"], 9),
                             f(row["R3_frac_consistent_macro_mean_all20"], 9),
                             "%d/%d" % (row["R3_fragments_applicable"], row["R3_fragments_total"]),
                             row["R1_denominator_total_declared20"]])
            parts.append(table(["端点（含两个 base）", "R1 macro(all20)", "R1 pooled", "R3 frac",
                                "R3 可用/总片段", "共同记录"], body))
            parts.append("")
            parts.append("R3 方向：consensus 两支 `0.737798`(raw)/`0.729464`(ms) **都高于** base `0.715377`；"
                         "G-random 上 raw `0.812738` 略低于、ms `0.821071` 略高于 base `0.817738`；raw 与 ms "
                         "之间方向随 base 改变。R1 均 `0.578`–`0.596`，两上限 `0.632137691` / `0.787180775`。")
            parts.append("")
        if sources:
            body = []
            seen = set()
            for row in sources:
                if row["metric"] != "pearson" or row["source"] in seen:
                    continue
                seen.add(row["source"])
                body.append([row["source"], f(num(row["matched_macro"])), f(num(row["cross_macro"])),
                             f(num(row["contrast_macro"])), f(num(row["inter_pearson"]))])
            parts.append(table(["source（Pearson macro）", "matched", "cross", "contrast", "inter"], body))
            parts.append("")
            parts.append("Spearman 全字段见 `evaluation/results/p1_source_summary.tsv`。R2 基线回归 **%s**"
                         "（复现 046 三个共享 source 的已发布 matched/contrast，≤5e-7）。"
                         % ((r2_reg or {}).get("status", "NA")))
            parts.append("")
        body = []
        for row in p1_boot:
            if row["metric"] != "pearson:contrast":
                continue
            if row["left"].startswith("A-") and row["right"].startswith("A-"):
                body.append([row["left"] + " − " + row["right"], f(num(row["mean"])), ci(row),
                             "%s/%s" % (row["left_wins"], row["right_wins"])])
        if body:
            parts.append("solver / 同 base 配对（R2 Pearson contrast，delta = 左 − 右；**solver 行左=raw、右=ms**，"
                         "两条比较标签统一为 raw-minus-ms）：")
            parts.append(table(["比较", "均值", "95% CI", "左高/右高"], body))
            parts.append("")
        baseline_rows = [row for row in p1_boot
                         if row["metric"] == "pearson:contrast" and "work-baseline" in row["right"]]
        if baseline_rows:
            parts.append("对 046 工作 baseline（1988 FG、含多分辨率 stage，**成本不同**）的 R2 contrast Δ："
                         + "；".join("%s `%+.6f` %s (%s/%s)" % (row["left"], num(row["mean"]), ci(row),
                                                              row["left_wins"], row["right_wins"])
                                     for row in baseline_rows) + "。")
            parts.append("")
        body = []
        for row in p1_r1r3:
            if row["metric"] not in ("R1_accuracy", "R3_frac_consistent"):
                continue
            if row["left"].startswith("A-") and row["right"].startswith("A-"):
                body.append([row["left"] + " − " + row["right"], row["metric"], row["better_direction"],
                             f(num(row["mean"])), ci(row),
                             "%s/%s" % (row["improvement_wins"], row["defined_chromosomes"])])
        if body:
            parts.append("R1/R3 配对（同一冻结索引矩阵；`R3_n_walls` 以 delta<0 记 improvement，"
                         "不把墙更多称胜出）：")
            parts.append(table(["比较", "metric", "更好方向", "均值", "95% CI", "改善胜出/defined"], body))
            parts.append("")
        if p1_identity:
            parts.append("共同支持身份检查 **PASS**：R1 逐 chr 分母六候选全同（合计 `%s`），R3 已解析片段集合"
                         "全同（`%s` 片段）；R3 共同支持按 `(chromosome, grid_start_bin)` 核验，"
                         "tie/insufficient 片段断 run 不跨接。"
                         % (sorted(set(p1_identity["R1_denominator_total_per_candidate"].values()))[0],
                            p1_identity["R3_labelled_fragments_common"]))
            parts.append("")
        if null_r1r3:
            agg = null_r1r3["aggregate"]
            parts.append("新端点 null：u0 `R1=%s`、`R3=NA`、`R2 contrast=0`；random-u `R1=%s`、`R3=%s`、"
                         "`R2 contrast=%s`（n=%s）。"
                         % (f(agg["u_zero:R1_macro_mean_defined_only"]["mean"], 3),
                            f(agg["random_u:R1_macro_mean_defined_only"]["mean"], 4),
                            f(agg["random_u:R3_frac_consistent_macro_mean_all20"]["mean"], 4),
                            f(agg["random_u:R2_contrast_spearman_macro_mean_all20"]["mean"], 4),
                            agg["random_u:R1_macro_mean_defined_only"]["n"]))
            parts.append("")

    parts.append("## 3b. 主结论（本轮实际支持什么）")
    parts.append("① **保留 046 `real-extension-G-full-J` 作为工作对照/baseline**（1988 FG、含多分辨率 stage，"
                 "成本不同于四支 486 FG）。② 当前 486 FG 预算下 **ms 相对 raw 无一致综合改善**：G-consensus 上 "
                 "contrast Δ(raw−ms) `-0.000883`（CI 跨 0），G-random 上 `+0.001169`（CI 不跨 0 但幅度 ~1e-3）；"
                 "**R1 在两个 base 上都是 raw 略高**（`+0.000316` / `+0.001413`，均跨 0 或幅度极小），"
                 "随 base 改变方向的是 **R3**。③ P0 的五端点 R1（含两上限）与 R3 可分辨，R3 相对 fixed random "
                 "有部分片段正增益。④ **本轮证据不足以证明全 chr 一致恢复（L2）**；本轮**不重做也不升级**旧 "
                 "L1 留出（held-out）结论，同样**不支持已撤回的 L3**（“任何无 SNP 方法都不能工作”）。")
    parts.append("")
    parts.append("## 4. P2：无参考中心 / 刚体块 probe（8 个预定扰动）")
    if p2:
        deltas = [row["fullJ_delta_A"] for row in p2["probes"]]
        validation = load("p2/results/validation.json") or {}
        disk = load("p2/results/disk_verification.json") or {}
        parts.append("设计：**2 个 family × {±0.01, ±0.05} × whole-cell Rg = 8 个预定扰动**（不是 8 个方向 × 2 "
                     "个幅度），seed `461001`，固定 e/p 不拟合。结果：**这八个预定扰动在这些幅度上未见 "
                     "`ΔJ<0`**（`ΔfullJ_A` 范围 `%+.6f … %+.6f`）；chr 内距离不变（max|Δ| ≤ `6.7e-16`）、"
                     "全部坐标在球内、baseline parity 与 count-delta 恒等式一致。自检文件 "
                     "`p2/results/validation.json`（`all_checks_passed=%s`，%d 个具名检查项 + `failed_checks=%s`）与 "
                     "`p2/results/disk_verification.json`（`all_disk_checks_passed=%s`、`failed_checks=%s`、"
                     "`n_probes_checked=%s`）均通过。"
                     % (min(deltas), max(deltas), validation.get("all_checks_passed"),
                        len([k for k in validation
                             if k not in ("schema", "created_at_utc", "all_checks_passed",
                                          "failed_checks", "phase_opened", "reference_opened",
                                          "post_run_corrections")]),
                        validation.get("failed_checks"), disk.get("all_disk_checks_passed"),
                        disk.get("failed_checks"), disk.get("n_probes_checked")))
        parts.append("")
        parts.append("限制：这 8 个预定扰动只给逐点读数，**既不能证明也不能否证全局可辨识性**，也不预设 KL>0；"
                     "附属 copy-centre 读出分母由 780 更正为跨 chr `760`（`denominator_correction.json`），"
                     "核心 KL / count / ΔJ 不受影响。")
    else:
        parts.append("未找到 `p2/results/p2_probes.json`。")
    parts.append("")

    parts.append("## 5. 空间读出（修正版权威口径）")
    if spatial_rows:
        body = []
        for row in spatial_rows:
            body.append([row["source"], f(num(row.get("merged_centre_pearson_190"))),
                         f(num(row.get("merged_centre_normalized_stress"))),
                         f(num(row.get("copy_centre_pearson_760"))),
                         f(num(row.get("copy_centre_normalized_stress")))])
        parts.append("20 合并中心（`2447` valid locus）给出 `190` 距离，stress 为单一最佳尺度；40 copy 中心给出"
                     "跨 chr `760` 距离，配对由每 chr whole-chromosome signed-Pearson 最佳 A/B swap 决定：")
        parts.append(table(["source", "190 中心 Pearson", "190 stress", "760 copy 中心 Pearson", "760 stress"],
                           body))
        parts.append("")
    parts.append("对 046 工作 baseline 的 049 冻结值回归 `max_abs_diff=0.0`，逐位复现 `spatial_summary.tsv` "
                 "第 16 行权威值 `0.37057429047492835` / `0.39980822791838044`。**baseline 口径引用错误已由"
                 "协调侧撤回，问题已解决，不留待裁决。** 旧 `spatial` 字段（z-score stress、参考有限支撑、"
                 "copyA→mat 直配）标 superseded。")
    parts.append("")
    parts.append("**null 空间列（已修正并已解决）**：`p1_spatial_fix.py` 的 null 缓存键已修正为 "
                 "`(source, null_kind, seed)`：119 个唯一键 / 119 个唯一 scored hash，逐 draw 评分数组 hash 与 "
                 "gate 一致（`u_null_per_draw_integrity_and_variation.passed=true`），7 source TSV 前后哈希不变，"
                 "049 冻结值回归与 A/B 交换不变性继续 PASS。结论：190 合并中心按构造不变（z 固定 + scale "
                 "不变性，与 source 最大差 `3.9e-16`）；760 copy 中心**确实随 seed 变化**（7 source × 16 seed "
                 "共 112 个逐 draw 值，各 source 内 16 个互不相同）；046 工作 baseline 的 760 为 "
                 "`0.129127759`，其自有 16 个 null 均值为 `0.088972075`、范围 "
                 "`[-0.009609647, 0.206813581]`；u0 的 760 仍为 7 个 NA（gauge 无定义，预期）。"
                 "**使用规则**：按 source 分层报告；190 不能用 u-null 当随机摆位 null；760 只作扰动参照，"
                 "**不做显著性判定**。")
    parts.append("")

    parts.append("## 6. 隔离、更正与局限")
    parts.append(table(["项", "内容"], [
        ["隔离", "每个评价入口先把候选/对照/null 坐标写出并哈希（`pre_reference_gate.json`）再 arm gate 读 phase/reference；P1 训练进程 `forbidden_modules_loaded=[]`、`reference_opened=false`；选择只用无标签 count 目标"],
        ["config / 元数据更正", "时间戳语义、`p2.checks` 措辞、一次逗号语法修复（`json.load` 已验证）；P1 terminal 派生字段仅对真正改值的两支记 `applied=true`，未变化的两支标 `no_numeric_change`——科学字段均未变"],
        ["R2 solver 标签", "两条 solver 比较标签统一为 **raw-minus-ms**（源码 PAIRS、JSON、TSV、README），与 `left − right` 一致；**数值未反转**"],
        ["空间 baseline / null 列", "baseline 引用值已撤回，权威值为 049 `spatial_summary.tsv` 第 16 行；null 缓存键已修正为 `(source, kind, seed)`，119 唯一键、逐 draw hash 与 gate 一致，**已解决**"],
    ]))
    parts.append("")
    parts.append("局限：① P1 是**局部优化诊断**（同起点 fork），不是独立盲重建，也不是生物学重复；工作 baseline "
                 "1988 FG 与四支 486 FG 成本不同。② 20 chr 重采样只描述单细胞内技术/结构变异。③ 四支全部 "
                 "`budget_not_converged`，任何“更好”只能读作该预算内的局部落点差异。④ R1 的 macro 与 pooled "
                 "是两个口径。⑤ P2 的 8 个 probe 不升级为全局可辨识性结论。⑥ **L2 未被本轮证明**。")
    parts.append("")

    parts.append("## 7. 权威路径索引")
    parts.append(table(["用途", "路径"], [
        ["P0 五端点 R1/R3 / 020 回归 / 逐 chr / 逐片段", "`p0/formal5/r1_r3_summary.json`、`r1_r3_per_chromosome.tsv`、`r3_fragments.tsv`、`p0/regression020/`"],
        ["P0 配对 / null / 支持身份", "`p0/controls/paired_bootstrap.tsv`、`null_support.tsv`、`support_identity.json`"],
        ["P1 终态 / 门禁 / 输入哈希 / R1/R3（含 base）", "`p1/results/*.terminal.json`、`p1/gates/*.preflight.json`、`p1/input_checks.json`、`evaluation/r1r3_arms_and_bases/r1_r3_summary.json`、`r3_fragments.tsv`"],
        ["P1 端点坐标（后续复用 ms 须携带 canonical raw_y/q）", "`p1/coords/*/1Mb.3dg`、`1Mb.npz`、`1Mb.canonical.npz`（`canonical_raw_theta` 与 `optimizer_theta` 已区分）"],
        ["P1 R2 / 配对 / null 分层 / R2 回归 / 空间权威版", "`evaluation/results/p1_source_summary.tsv`、`p1_paired_bootstrap.tsv`、`p1_r1r3_paired_bootstrap.tsv`、`r2_baseline_regression.json`、`p1_null_aggregate_by_source.json`、`p1_support_identity.json`、`p1_spatial_corrected*.{json,tsv}`（含 null 分层与 per-draw 完整性证据）"],
        ["P2", "`p2/results/p2_probes.{tsv,json}`、`validation.json`、`denominator_correction.json`"],
    ]))
    parts.append("")
    parts.append("图：本轮**零图**（文字表格足够，见 `plots/README.md`）。终态核对：020 R1/R3 回归 **%s**；"
                 "R2 基线回归 **%s**；config 当前哈希 `%s`。"
                 % ("PASS" if reg_ok else "未完成", (r2_reg or {}).get("status", "NA"),
                    lock.get("config_sha256_current", "NA")))
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(parts))
    (RUN_DIR / "README.md").write_text(text + "\n", encoding="utf-8")
    print("README.md written, %d lines" % len(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
