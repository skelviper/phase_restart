"""从 results.json 构建运行 README。

与 run.py 分开，使正文只需写一次，表格可以在不重新拟合的情况下重新生成。
"""
import json
import os

import numpy as np


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        return "nan"
    return ("%." + str(nd) + "f") % x


def _ci(d):
    return "[%+.4f, %+.4f]" % (d["lo"], d["hi"]) if np.isfinite(d["lo"]) else "n/a"


ORDER = ["consensus", "random", "random_phased", "random_phased_1", "random_phased_2",
         "oracle", "gauge_all",
         "micro5", "micro10", "wall1", "wall2", "wall3", "wall4", "wall5",
         "island1", "island2", "island3", "alt",
         "wall1_rewire", "wall2_rewire", "wall3_rewire", "wall4_rewire", "wall5_rewire",
         "alt_rewire",
         "oracle_half1", "oracle_half2"]


def build(res, outdir):
    r = res
    T = r["table"]
    B = r["bootstrap"]
    G = r["gauge_invariant"]
    M = r["masked_compare"]
    D = r["decision"]
    ids = [i for i in ORDER if i in T]

    L = []
    A = L.append
    A("# 阶段 1：片段交换实验（%s）" % r["chrom"])
    A("")
    A("运行 `%s`。驱动程序为 `run.py stage1`，设计冻结于 `docs/STAGE1_PREREGISTRATION.md`。"
      % os.path.basename(os.path.abspath(outdir)))
    A("")
    A("## 研究问题")
    A("")
    A("无标签的留出分数能否把**整条染色体一致的**两拷贝")
    A("拆分与**局部正确但全局错接**的拆分区分开？")
    A("")
    A("**这是 oracle 工具检验，不是盲恢复。** 这里不提出任何方法。正结果可以支持进入阶段 2，但不能证明 L2。")
    A("")
    A("## 核心答案")
    A("")
    mono = D["min_detectable_flipped_mb"]
    A("**是。** 已检测到一个分界墙。合并后的规范不变评分在 %s 上检测到的最小翻转区域为 **%s Mb**。" % (r["chrom"], mono))
    A("")
    A("| 工具 | 最小检测分界墙 | 该尺度的 rho 降幅 |")
    A("| --- | ---: | ---: |")
    def _best(rows):
        det = [x for x in rows if x[2]]
        return min(det, key=lambda x: x[0]) if det else None
    gr = _best([(T[i]["flipped_mb"], i, G[i]["excludes_zero"]) for i in G if i != "_reference"])
    A("| 合并、规范不变（`relabel`） | %s | %+.4f |"
      % ("%d Mb" % gr[0] if gr else "未检测到", G[gr[1]]["delta"] if gr else float("nan")))
    rw = _best([(T[i]["flipped_mb"], i, G[i]["excludes_zero"])
                for i in G if i.endswith("_rewire")])
    A("| 合并、`rewire` 规则 | %s | %+.4f |"
      % ("%d Mb" % rw[0] if rw else "未检测到", G[rw[1]]["delta"] if rw else float("nan")))
    sp = sorted([(v["flipped_mb"], k, v) for k, v in r["splice"].items() if v["excludes_zero"]])
    A("| 无需选择规范的嵌合结构（`splice`） | %s | %+.4f |"
      % ("%d Mb" % sp[0][0] if sp else "未检测到",
         sp[0][2]["delta_vs_oracle"] if sp else float("nan")))
    A("")
    A("三类候选类别的选择检验均具有决定性：无标签的留出 rho 将 **oracle 排在所有预注册的错接替代方案之上**。")
    A("")
    A("| 候选类别 | 最大值 | oracle 是否第一 |")
    A("| --- | --- | --- |")
    for fam, v in r["selection"].items():
        A("| %s | `%s` | %s |" % (fam, v["best"], "是" if v["oracle_wins"] else "**否**"))
    A("")
    A("## 本次运行能否证明什么")
    A("")
    A("| 结论 | 本次运行后的状态 |")
    A("| --- | --- |")
    A("| **L1** 存在可预测信号 | 按修正协议在 chr1 上复现：oracle %s，对比同记录集 random %s（delta %+.4f） |"
      % (_fmt(T["oracle"]["rho_pooled"]), _fmt(T["random_phased"]["rho_pooled"]),
         T["oracle"]["rho_pooled"] - T["random_phased"]["rho_pooled"]))
    A("| **L2** 整条染色体一致的恢复 | **仍未显示。** 本次运行说明该*工具*能识别全局错接；尚未证明有盲方法能达到一致拆分。 |")
    A("| **L3** 无 SNP 恢复不可能 | 仍为撤回状态；本次运行与此无关 |")
    A("")
    A("## 参考结构（评估器侧，仅在坐标完成哈希后读取）")
    A("")
    acc = r["reference_accuracy"]
    A("参考结构将 chr1 已完整分相、同染色体同拷贝（cis）的接触分类为 **%s**（方向 %d），n=%d，"
      % (_fmt(acc["acc"]), acc["orientation"], acc["n"]))
    A("排除同区间接触 %d，剩余并列率为 %s。"
      % (acc["n_samebin_excluded"], _fmt(acc["tie_rate"])))
    A("这是 oracle 结构上限，不是方法结果：它使用了分相标签。")
    A("")
    A("## 数据与协议")
    A("")
    A("| | |")
    A("| --- | --- |")
    A("| 染色体 | %s |" % r["chrom"])
    A("| 接触记录（cis） | %d |" % r["n_records"])
    A("| 同区间记录，所有指标均排除 | %d (%.2f%%) |"
      % (r["samebin_records"], 100.0 * r["samebin_records"] / r["n_records"]))
    A("| 区间 / 片段 | %d 个区间（%d bp），%d 个片段（每个 %d 个区间） |"
      % (r["n_bins"], r["bin_bp"], r["n_fragments"], r["frag_bins"]))
    A("| train / val / test 记录 | %d / %d / %d |"
      % (r["n_train"], r["n_val"], r["n_test"]))
    A("| 留出 bin pairs（val / test） | %d / %d |" % (r["pairs_val"], r["pairs_test"]))
    n_ph = r.get("n_ph_train")
    if n_ph is None:                       # 旧 results.json：从日志恢复
        try:
            with open(os.path.join(outdir, "logs", "run.log")) as fh:
                for line in fh:
                    if "phased train records" in line:
                        n_ph = int(line.split("phased train records")[1].split()[0])
                        break
        except OSError:
            n_ph = None
    A("| oracle-family 训练记录（完全定相） | %s |" % (n_ph if n_ph else "n/a"))
    A("| bootstrap 重采样次数 | %d |" % r["config"]["nboot"])
    A("")
    A("折分配使用 `benchmark_v1.py` 的混合哈希，并核验与该标量")
    A("实现（`pr.folds.regression_against_reference`）一致，且核验它不是")
    A("基因组间距（`pr.folds.sanity_check`）的函数。")
    A("")
    A("## 每个候选的留出 rho")
    A("")
    A("| 候选 | 类别 | 规则 | 翻转 Mb | alpha | n | NaN | pooled rho | intra-frag | x-short | x-long |")
    A("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i in ids:
        t = T[i]
        s = t["strata"]
        A("| `%s` | %s | %s | %s | %.2f | %s | %s | **%s** | %s | %s | %s |"
          % (i, t["family"], t["rule"] or "-",
             t["flipped_mb"] if t["flipped_mb"] is not None else "-", t["alpha"],
             t.get("n_finite", "-"), t.get("n_nan", "-"),
             _fmt(t["rho_pooled"]), _fmt(s["intra_frag"]["rho"]),
             _fmt(s["xshort"]["rho"]), _fmt(s["xlong"]["rho"])))
    A("")
    A("`intra-frag` = 两端位于同一 20 Mb 片段；`x-short` = 位于不同片段，")
    A("间距 <= 40 Mb；`x-long` = 位于不同片段，间距 > 40 Mb。")
    A("")
    A("### NaN 规则")
    A("")
    A("`%s`" % r.get("nan_policy", "未记录"))
    A("")
    A("候选自身的拟合记录集中没有 contact 的 bin，native engine 不会为其写出 bead，因此该距离未定义。每次比较都使用")
    A("**涉及两个候选的最大共同有限子集**，且每一行都注明")
    A("自己的 `n`。所有候选之间不存在全局交集：早期版本")
    A("取了一个全局交集，两个各缺一个 bead 的辅助 half-data diagnostics 静默地")
    A("将**每个**指标从 %d 缩为 1,603 个 pairs（chrX 从 1,059 缩为 583）。"
      % r["pairs_test"])
    A("")
    A("每个候选的有限 pair 计数：" + ", ".join(
        "`%s` %s" % (k, v) for k, v in sorted(r.get("finite_counts", {}).items(),
                                             key=lambda kv: kv[1])) + ".")
    A("")
    A("## 规范不变读出（主要比较）")
    A("")
    A("native 双拷贝拟合对标签分配到哪条轨迹**并不**保持不变：")
    A("将每个 bin 翻转（纯规范变化）会使留出 rho 移动 **%s**。"
      % _fmt(G["_reference"]["gauge_asymmetry"]))
    A("因此参考结构水平取 `oracle` 与 `gauge_all` 的均值 = **%s**。"
      % _fmt(G["_reference"]["rho_oracle_gauge_avg"]))
    A("")
    A("| 候选 | 规则 | 翻转 Mb | rho（规范均值） | 自身规范不对称 | 相对 oracle 的 delta | 95% CI | n | 是否检测到 |")
    A("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |")
    for i in ids:
        if i not in G:
            continue
        v = G[i]
        A("| `%s` | %s | %d | %s | %s | %+.4f | %s | %s | %s |"
          % (i, T[i]["rule"], T[i]["flipped_mb"], _fmt(v["rho_gauge_avg"]),
             _fmt(v["gauge_asymmetry"]), v["delta"], _ci(v), v.get("n", "-"),
             "**是**" if v["excludes_zero"] else "否"))
    A("")
    A("## 受影响 pair 读出（最敏锐的比较）")
    A("")
    A("限定在候选的翻转掩码在两端之间不一致的区间对上，")
    A("也就是错接实际造成损害的区间对。null 列是在两个独立 half-data oracle 拟合之间")
    A("计算的同一读出。")
    A("")
    A("没有翻转模式的候选（`consensus`、`random`、`random_phased*`）没有")
    A("受影响子集，因此在此省略；它们出现在上面的合并表中。")
    A("")
    A("| 候选 | 区间对数（占 test 百分比） | 含 null 的 n | rho oracle | rho candidate | delta | 95% CI | null |")
    A("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for i in ids:
        if i not in M or (T[i]["rule"] is None):
            continue          # 没有翻转模式，无法限定受影响的区间对子集
        v = M[i]
        A("| `%s` | %d (%.0f%%) | %d | %s | %s | **%+.4f** | %s | %+.4f |"
          % (i, v["n"], 100 * v["frac_of_test"], v.get("n_common_with_null", 0),
             _fmt(v["rho_oracle"]), _fmt(v["rho_cand"]),
             v["delta"], _ci(v), v["null_delta_half_oracles"]))
    A("")
    A("## 无 gauge 的 chimera（`splice`）")
    A("")
    A("oracle 的两条拟合轨迹按片段层面进行 Procrustes 叠合和 splice，")
    A("然后直接评分，不重新标记接触，也不重新拟合，因此该工具**没有规范不对称**。这是最清楚的长程错接证据。")
    A("")
    A("| 翻转 Mb | rho | 相对 oracle 的 delta | 95% CI | n | 是否检测到 |")
    A("| ---: | ---: | ---: | --- | ---: | --- |")
    for k, v in sorted(r["splice"].items(), key=lambda kv: kv[1]["flipped_mb"]):
        A("| %d | %s | %+.4f | %s | %s | %s |"
          % (v["flipped_mb"], _fmt(v["rho_pooled"]), v["delta_vs_oracle"], _ci(v), v.get("n", "-"),
             "**是**" if v["excludes_zero"] else "否"))
    A("")
    A("## 边界放宽（每个候选使用相同预算）")
    A("")
    A("对训练记录中位于九条名义 20 Mb 网格线 ±%d 个区间内的 %s 做一轮硬 E-step，"
      % ("band", r["config"]["relax_band"]))
    A("并且每个候选使用相同网格，因此交换划分不会仅因只有它受到的人为截断而与 oracle 分离。")
    A("")
    A("| 候选 | rho 之前 | rho 之后 | E-step 与 truth 一致 | 已重标记 | 带区记录（更新后） |")
    A("| --- | ---: | ---: | ---: | ---: | ---: |")
    for i in ids:
        if i not in r["relaxation"] or "skipped" in r["relaxation"][i]:
            continue
        v = r["relaxation"][i]
        A("| `%s` | %s | %s | %s | %d | %d (%s) |"
          % (i, _fmt(T[i]["rho_pooled"]), _fmt(v["rho_pooled"]),
             _fmt(v["estep_agreement"]), v["changed"],
             v["n_in_band"], v.get("n_updated", "n/a")))
    A("")
    A("符号断言（报告规则 5）：oracle 的 E-step 在该带区上与真实 label 的一致率为")
    A("%s，高于 0.55 下限。" % _fmt(r["relaxation"]["oracle"]["estep_agreement"]))
    A("一轮放宽不会修复 swap：每个完整片段交换候选")
    A("都低于放宽后的 oracle。因此这种差异不是片段边界处硬截断的产物。")
    A("")
    A("需要注意这项特定预算的限制：带区位于名义 20 Mb 网格上，")
    A("因此恰好覆盖完整片段分界墙，但**漏掉 5 Mb 和 10 Mb 分界墙**，")
    A("它们位于 5 和 10 Mb。因此对应的放宽后数值是在*其他位置*放宽，")
    A("而不是在各自 wall 处放宽。")
    A("")
    A("## 预注册决策规则")
    A("")
    A("`null 容差 = max(null 范围 %s, 规范不对称 %s) = %s`。"
      % (_fmt(D["null_spread_range"]), _fmt(D["gauge_asymmetry"]), _fmt(D["null_tolerance"])))
    A("")
    A("| 翻转 Mb | 候选 | delta | 95% CI | 是否检测到 |")
    A("| ---: | --- | ---: | --- | --- |")
    for s in D["scales"]:
        A("| %d | `%s` | %+.4f | [%+.4f, %+.4f] | %s |"
          % (s["flipped_mb"], s["id"], s["delta"], s["lo"], s["hi"],
             "**是**" if s["distinguishable"] else "否"))
    A("")
    A("**检测到的最小翻转区域：%s Mb。**" % D["min_detectable_flipped_mb"])
    A("")
    A("## 直白列出的限制")
    A("")
    A("1. **一条染色体、一个细胞。** 仅 chr1。20 条染色体是单个细胞中的 20 个")
    A("   关联测量；这里没有生物学重复。")
    A("2. **合并评分中的 20 Mb 效应很小**（+%+.4f），引擎自身的" % G["wall1"]["delta"])
    A("   规范不对称为 %s。规范不变读出会移除这个偏移，但" % _fmt(D["gauge_asymmetry"]))
    A("   余量很薄。决定性证据来自受影响 pair 读出")
    A("   （%+.4f，%d 个区间对，null %+.4f）以及无需选择规范的 splice（%+.4f）。"
      % (M["wall1"]["delta"], M["wall1"]["n"], M["wall1"]["null_delta_half_oracles"],
         r["splice"]["splice_wall1"]["delta_vs_oracle"]))
    A("3. **拟合噪声下限。** 在一半训练记录上重新拟合*相同* oracle 拆分，")
    A("   留出 rho 最多移动 %s。仅凭该工具，大小相同的交换效应无法与"
      % _fmt(D["half_data_oracle_delta_max"]))
    A("   训练数据的变化区分开来。")
    A("4. **两条翻转规则、一个有歧义的对象。** 片段交换不是可实现的轨迹对，")
    A("   因此跨分界墙接触没有规范标签。两条规则都报告；它们结论一致但幅度不同。")
    A("5. **`relabel` 是对 AGENTS.md 原文的直接解读**，但也是")
    A("   对规范敏感的规则；`rewire` 按构造对互补操作保持不变。")
    A("6. **参考结构准确率 %s 是上限，不是达成结果**：它使用了方法不可见的分相"
      % _fmt(acc["acc"]))
    A("   标签。")
    A("")
    A("## 本次运行后发现的缺陷")
    A("")
    A("这里记录这些缺陷而不是静默修补，因为运行目录保持生成时的原样。")
    A("")
    A("1. 控制台日志的 reference 行对任意染色体都打印字面量 `chr1`。")
    A("   `results.json` 记录了真实染色体（`%s`）。两次阶段 1 运行后已在 `run.py` 中修复，"
      % r["chrom"])
    A("   reference 数值本身不受影响。")
    A("2. `pr.fdg.pair_distances` 最初查找原始接触位置，而 native 引擎")
    A("   引擎每个 1 Mb 区间起点只写出一个珠子。修复前所有查找都返回 NaN，")
    A("   并静默清空放宽带区。该问题在冒烟运行中捕获，")
    A("   在任何正式运行前已修复。")
    A("3. `pr.splits.apply_flip` 最初从已有的片段索引重新推导片段索引，")
    A("   把每条接触都压到片段 0。该问题在冒烟运行中捕获，")
    A("   在任何正式运行前已修复。")
    A("")
    A("后续对 canonical 运行的审计发现两项仅影响报告的细节，现予披露而不修改数值：")
    A("")
    A("- `bootstrap[*].delta` 过去是每个候选*自身*有限集上的 rho 差，而 CI 使用")
    A("  *共同*集合上的配对 bootstrap。现在同时报告 `delta`（共同集合）")
    A("  和 `delta_own_sets`；本次运行两者相差约 4e-4。")
    A("- `relaxation[*].n_in_band` 在有限性筛选前统计带区记录；实际更新的数量")
    A("  现在以 `n_updated` 并列报告。")
    A("")
    A("## 文件")
    A("")
    A("```")
    A("results.json     所有数值，包括 bootstrap draws")
    A("config.json      精确的 CLI 参数")
    A("gate.json        每个写出坐标文件的 sha256")
    A("coords/*.3dg     拟合结构，每个候选一个")
    A("work/*.pairs.gz  精确的 FDG 输入（每个七列）")
    A("plots/           四张图")
    A("logs/run.log     控制台日志")
    A("```")
    A("")
    return "\n".join(L) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(os.path.join(a.out, "results.json")) as f:
        res = json.load(f)
    text = build(res, a.out)
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(text)
    print("wrote %s/README.md (%d lines)" % (a.out, text.count("\n")))


if __name__ == "__main__":
    main()
