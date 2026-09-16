"""从 049 运行目录的 JSON 产物装配根 README.md（中文内部）。

只读 results/ diagnostics/ gates/ 与可选 evaluation/；不写其他文件。
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
LOSS_NAME = {"A": "marginal_G", "B": "hard_observed", "C": "max_rate"}


def load(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits=6):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return ("%%.%df" % digits) % value
    return str(value)




def _tsv(path):
    import csv
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _num(value, digits=6):
    if value in (None, "", "None", "NA", "nan"):
        return "NA"
    try:
        return ("%%.%df" % digits) % float(value)
    except (TypeError, ValueError):
        return str(value)


def evaluation_section(add, run) -> None:
    """§6 后置评价：全部数字从 evaluation/results/*.tsv|json 渲染。"""
    results = run / "evaluation" / "results"
    terminal = load(results / "terminal.json", {})
    validation = load(results / "validation.json", {})
    if not terminal:
        add("评价（R2 / inter / 空间 / null）由评价侧在 `evaluation/` 内完成，本文件在收到其结果后合并。"
            "本轮只做 R2 与空间，不做 R1/R3，不声称 L2。")
        return
    r2 = {(row["candidate_id"], row["metric"]): row for row in _tsv(results / "r2_summary.tsv")}
    inter = {row["candidate_id"]: row for row in _tsv(results / "inter_summary.tsv")}
    spatial = {row["candidate_id"]: row for row in _tsv(results / "spatial_summary.tsv")}
    common = {row["candidate_id"]: row for row in _tsv(results / "endpoint_common_g.tsv")}
    nulls = _tsv(results / "null_summary.tsv")
    boot = _tsv(results / "paired_bootstrap.tsv")
    base = "baseline-046-real-extension-G-full-J"
    add("评价终态 `evaluation/results/terminal.json`：`evaluation_terminal=%s`、`validation_status=%s`、"
        "参考首次打开 `%s`（gate 之后）、15 datasets / 153 null / 28 comparisons，"
        "046 baseline 复算 %s 个值零差异，bootstrap 指数矩阵与 046 相同。" % (
            terminal.get("evaluation_terminal"), terminal.get("validation_status"),
            terminal.get("reference_first_opened_utc"),
            (validation.get("checks") or {}).get("baseline_regression_checked_values")))
    add("")
    add("详细报告：[`evaluation/REPORT.md`](evaluation/REPORT.md)；两张图："
        "[`049_figure1_core_metrics.png`](evaluation/plots/049_figure1_core_metrics.png)、"
        "[`049_figure2_whole_genome.png`](evaluation/plots/049_figure2_whole_genome.png)。"
        "下表数字全部由 `evaluation/results/*.tsv` 直接渲染。")
    add("")
    add("### 6.1 15 个 dataset 的主读出")
    add("")
    add("| dataset | kind | terminal | R2 matched | R2 contrast | inter Pearson | inter Spearman | 20-center Pearson | 20-center Spearman | 20-center stress | 760 copy-center Pearson | 760 copy-center stress | 共同 G count |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    order = [row["candidate_id"] for row in _tsv(results / "endpoint_common_g.tsv")]
    for cid in order:
        row = common.get(cid, {})
        m = r2.get((cid, "pearson"), {})
        sp = spatial.get(cid, {})
        it = inter.get(cid, {})
        add("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            cid, row.get("kind"), row.get("terminal"),
            _num(m.get("matched")), _num(m.get("contrast")),
            _num(it.get("pearson")), _num(it.get("spearman")),
            _num(sp.get("center_pearson_190")), _num(sp.get("center_spearman_190")),
            _num(sp.get("center_normalized_stress")),
            _num(sp.get("copy_center_pearson_760")), _num(sp.get("copy_center_normalized_stress")),
            _num(row.get("common_G_count"))))
    add("")
    add("R2 为 Pearson（Spearman 见 `r2_summary.tsv`）；matched/cross/contrast 的拷贝标签按几何 direct/swapped 规范选择，"
        "tie 记为 unresolved。20-center 指标基于 old21 mask 支持的合并染色体中心（gauge-free）；"
        "760 copy-center 为跨染色体拷贝中心距离，按预定 whole-chr intraR2 方向统一映射并做一次 proper global alignment。")
    add("")
    add("### 6.2 关键配对 bootstrap（seed 450301，10000 draws，20 chr 配对）")
    add("")
    add("| 比较 | metric | mean Δ | 95% CI | 左/右胜出 |")
    add("| --- | --- | --- | --- | --- |")
    for row in boot:
        if row["label"] != "endpoint-minus-baseline" or row["metric_field"] != "pearson:matched":
            continue
        add("| `%s` − baseline | pearson:matched | %+0.6f | [%+0.4f, %+0.4f] | %s / %s |" % (
            row["left"], float(row["mean"]), float(row["ci95_low"]), float(row["ci95_high"]),
            row["left_wins"], row["right_wins"]))
    for row in boot:
        if row["label"] == "same-source ms-minus-raw" and row["left"] == "C-ms-random" \
                and row["metric_field"] in ("pearson:matched", "pearson:contrast", "spearman:matched"):
            add("| `C-ms-random` − `C-raw-random` | %s | %+0.6f | [%+0.4f, %+0.4f] | %s / %s |" % (
                row["metric_field"], float(row["mean"]), float(row["ci95_low"]), float(row["ci95_high"]),
                row["left_wins"], row["right_wins"]))
    add("")
    add("### 6.3 结论（平衡且限定）")
    add("")
    add("- **12 条新端点的 R2 matched、contrast 与 pooled inter 全部低于历史 baseline**；其中 11 条的 matched 差 CI 不跨 0，"
        "只有 `A-ms-random`（本轮最好 matched %s vs baseline %s，差 −0.0400，CI[−0.0944,+0.0105]，20 chr 中 9 胜）CI 跨 0。"
        "主方案 `B-raw-consensus`：matched %s / contrast %s / inter %s，对应 baseline %s / %s / %s。" % (
            _num(r2[("A-ms-random", "pearson")]["matched"]), _num(r2[(base, "pearson")]["matched"]),
            _num(r2[("B-raw-consensus", "pearson")]["matched"]),
            _num(r2[("B-raw-consensus", "pearson")]["contrast"]), _num(inter["B-raw-consensus"]["pearson"]),
            _num(r2[(base, "pearson")]["matched"]), _num(r2[(base, "pearson")]["contrast"]),
            _num(inter[base]["pearson"])))
    add("- **局部确有改善**：同 loss 同 source 的 raw→ms 对比中 5/6 组 own count 更低（见 §3）；"
        "`C-ms-random` − `C-raw-random` 的 matched +0.058407、CI[+0.0253,+0.0947]、13/20 胜；"
        "`A-ms-random` 的 760 copy-center Pearson %s vs baseline %s、stress %s vs %s；"
        "`C-raw-consensus` 的 merged-center Spearman %s、stress %s vs baseline %s / %s（但其 R2/inter 退化）。"
        "这些是**后验（reference 之后）空间观察，不能用来改已冻结的 display 选择**。" % (
            _num(spatial["A-ms-random"]["copy_center_pearson_760"]), _num(spatial[base]["copy_center_pearson_760"]),
            _num(spatial["A-ms-random"]["copy_center_normalized_stress"]), _num(spatial[base]["copy_center_normalized_stress"]),
            _num(spatial["C-raw-consensus"]["center_spearman_190"]), _num(spatial["C-raw-consensus"]["center_normalized_stress"]),
            _num(spatial[base]["center_spearman_190"]), _num(spatial[base]["center_normalized_stress"])))
    add("- `B-ms-random` 只有 merged-center Spearman (%s) 高于 baseline (%s)，其 stress %s **差于** baseline %s，"
        "因此不能说它两项都更好。" % (
            _num(spatial["B-ms-random"]["center_spearman_190"]), _num(spatial[base]["center_spearman_190"]),
            _num(spatial["B-ms-random"]["center_normalized_stress"]), _num(spatial[base]["center_normalized_stress"])))
    add("- **重要诊断**：全部 12 条新端点的 20-center Pearson 与 stress 都差于各自冻结 initial "
        "（control center r = %s / %s，stress = %s / %s）。但所有拟合的 pooled inter 明显高于 initial（%s / %s）。"
        "因此**“sorted-four pooled inter 提高”不能直接解读成染色体中心摆位提高**——每个 order statistic 与中心读出必须一起看。" % (
            _num(spatial["initial-consensus"]["center_pearson_190"]), _num(spatial["initial-random"]["center_pearson_190"]),
            _num(spatial["initial-consensus"]["center_normalized_stress"]), _num(spatial["initial-random"]["center_normalized_stress"]),
            _num(inter["initial-consensus"]["pearson"]), _num(inter["initial-random"]["pearson"])))
    baseline_null = [row for row in nulls if row["source_candidate_id"] == base and row["null_kind"] == "random_u"
                     and row["metric"] == "inter_pearson"]
    if baseline_null:
        row = baseline_null[0]
        add("- 不能把 random-u 的高 pooled 值直接说成“u 携带大量正确摆位信息”：16 个 random-u 的 inter Pearson "
            "mean %s（range %s–%s）本身也很高，其中可能包含 4-order 排序分布/尺度差异；"
            "同时 baseline 自身 %s 仍高于其 random-u 均值 %s，**不能与随机等同**。" % (
                _num(row["mean"]), _num(row["min"]), _num(row["max"]),
                _num(inter[base]["pearson"]), _num(row["mean"])))
    add("- **总结**：本轮**未找到能全面替换 baseline 的方法**，而不是所有指标毫无改善；B（用户主方案）语义更确定，"
        "但本轮结构变差。这不否定一切 hard/max 方法或求解器，也不构成 L2 已证明或已否证。")
    add("")
    add("### 6.4 评价侧偏离与 errata（只记录，不改封存产物）")
    add("")
    deviations = terminal.get("deviations") or []
    for item in deviations:
        add("- **%s**：%s（影响：%s）" % (item.get("item"), item.get("detail") or item.get("requirement"),
                                          item.get("impact")))
    add("- **mask SHA 笔误的更正状态（erratum）**：上面第 2 条是评价侧读到的 config 快照记录。"
        "**权威值 = `9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9`**，"
        "与 mask 文件实际哈希、以及 046 创建记录 `evaluation_final/results/mask_validation.json` 的 "
        "`frozen_legacy_mask_worker.output_sha256` 三者一致。本目录 `config.json` 的 `evaluation.mask.sha256` 现为该权威值，"
        "并在 `errata` 段保留本条记录；mask 内容与冻结计数（total 176201 / common 157529 / validbins 2447 / "
        "inter 2835152 locus pairs）未变，未修改任何 mask/参考/坐标文件。")
    add("- **预 gate mask metadata 早读偏离的更正状态**：见上面第 1 条；已如实登记，未补造审计、未改变任何评价数值。")
    add("- **attempt 1 失败（exit 1）与 attempt 2 成功是两件事**（详见 `evaluation/logs/EVALUATION_ATTEMPTS.md`）："
        "attempt 1 在 `2026-09-15T17:30:15.548703+00:00` 首次打开 reference（SHA 校验通过、40 track 解析成功）后，"
        "于 `load_real_masks` 抛出 `RuntimeError: frozen mask snapshot hash mismatch`；**失败原因是评价侧常量沿用了 config.json 中"
        "抄错 1 个字符的 mask sha256（`...1541f55f...` 而非文件实际的 `...1547f55f...`）**。核对 046 创建记录"
        "（`mask_validation.json → frozen_legacy_mask_worker.output_sha256`）并改正常量后，attempt 2 完成完整 15-dataset 评价、"
        "**exit code 0**，并保留 attempt 1 的真实首次打开时间（`open_attempt_count=2`）。诚实说明：attempt 1 的原始 stderr 写在"
        "`logs/evaluate_formal.log`，该文件被 attempt 2 重跑覆盖（未 append），因此**不补造** traceback 原文；上述异常消息与首次打开时间"
        "保存在 `evaluation/gates/reference_open.json`、`results/validation.json` 的 deviations"
        "（`mask_config_sha256_typo_recorded=true`）与 `REPORT.md`。")
    add("- **attempt 2 之后的派生输出路径元数据修复（与上面那次重跑无关，也不是它的失败原因）**：`terminal.json` / `evaluation.json` 中"
        "12 个 fit 的 `npz_path` 曾被写成 CWD 相对的 `evaluation/test_res/049-.../coords/...`（对已是仓库相对形式的路径直接 `rel()`，"
        "未先 `resolve_path`），文件实际不存在。修复从已保存的 `evaluation.json`/`terminal.json` 与冻结 manifest 只重写路径字段，"
        "共 **36 处**（`evaluation.datasets[*]` 12 + `terminal.datasets[*]` 12 + `endpoint_common_g[*]` 12），写入前用 `strip_paths` "
        "深比较断言**数值与其它字段 0 变化**：`numeric_or_other_fields_changed=0`、`recomputed_metrics=false`（before/after sha256 见 "
        "`results/validation.json → path_repair`）；路径存在性检查 348 条、缺失 0（`results/path_check.json` PASS）。"
        "**这一步不重算任何指标。**")


def main() -> int:
    terminal = load(RUN / "results/formal_terminal.json", {"rows": [], "status_counts": {}})
    manifest = load(RUN / "results/endpoint_manifest_pre_reference.json", {"endpoints": {}})
    selection = load(RUN / "results/selection_pre_reference.json", {"per_loss": {}})
    gates = load(RUN / "gates/gates_report.json", {"passed": 0, "gate_count": 0, "failed": []})
    posterior = load(RUN / "diagnostics/posterior_diagnostic.json", {})
    cross = load(RUN / "diagnostics/cross_resolution_closure.json", {})
    probe = load(RUN / "diagnostics/probe_diagnostic.json", {})
    evaluation = load(RUN / "evaluation/evaluation_summary.json", None)

    fit_rows = manifest.get("fits", [])
    by_label = {row.get("fit_id"): row for row in fit_rows}
    by_label.update({row.get("fit_id"): row for row in manifest.get("initials", [])})
    if isinstance(manifest.get("baseline"), dict):
        by_label[manifest["baseline"].get("fit_id")] = manifest["baseline"]
    lines: list[str] = []
    add = lines.append
    add("# 049 max-contact-unified-multiscale 内部报告")
    add("")
    add("运行目录：`test_res/%s/`（run id `%s`）。本文件由 `source/make_readme.py` 从运行产物装配；"
        "表内数值全部来自 `results/`、`diagnostics/`、`gates/`、`evaluation/` 的 JSON/TSV。" % (RUN.name, RUN.name))
    add("")
    add("## 1. 本轮做了什么")
    add("")
    add("- 一个新真实目录，12 条 fit：`3 loss × 2 solver × 2 blind source`，每条固定 **1502 FG**、全程 1Mb full grid。")
    add("- loss：`A marginal_G`（= 045 原 G）、`B hard_observed`（用户主方案，normalizer 仍为 Zsum、观测项取 max）、"
        "`C max_rate`（Zmax 也取 max）。solver：`raw` 原 raw_y/q；`ms` 链内多尺度线性预条件 "
        "`P = I + (I+25L)^-1 + (I+400L)^-1`（DCT-II 对角化，全部 > 0，优化 `z = P^-1 y`，停止用 canonical raw_y/q 梯度）。")
    add("- 两个盲起点为 014 获准无标签 source 的 zero-optimization prolongation；同 source 的 6 条 fit 共享逐元素相同的 raw_y/q。")
    add("- baseline 为 046 `real-extension-G-full-J`（事后采纳的工作对照，非本轮训练初值）。")
    add("")
    add("## 2. 冻结事实与边界")
    add("")
    add("| 项 | 值 |")
    add("| --- | --- |")
    add("| SNP-free 输入 | `inputs/P9016.snpfree.pairs.gz` SHA256 `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`（7 列，拒绝 phase） |")
    add("| 记录 / loci / full grid | 1,703,888 records；2645 loci；3,496,690 零对保留的 offdiag pair |")
    add("| 分母 | `Nraw=1,703,888`；`Noff=696,680+568,434=1,265,114`；diag `438,774` 独立饱和 nuisance |")
    add("| 生物重复 | n=1（同一 P9016 细胞），20 chr 为细胞内关联技术/结构测量 |")
    add("| 工程门 | %d/%d PASS，failed=%s（`gates/gates_report.json`） |" % (gates.get("passed"), gates.get("gate_count"), gates.get("failed")))
    add("| reference | 本轮训练与诊断阶段未打开；由评价侧在候选/initial/baseline/null 全部 hash 之后打开 |")
    add("")
    add("## 3. 12 fits 终态与共同 rescore")
    add("")
    add("**旧 baseline 1988 FG 含 5Mb/2Mb stage，本轮每条 fit 1502 FG 全为 1Mb full-grid；两者成本不同，不能称成本相等。**")
    add("")
    add("| fit | loss | solver | source | FG cap | FG 实际 | wall (s) | terminal | canonical gel max | count_A | count_B | count_C | fullJ(own) | fullJ_A(共同) | Rg |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in sorted(fit_rows, key=lambda item: item.get("fit_id")):
        fit = row.get("fit_id")
        add("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            fit, row.get("loss"), row.get("solver"), row.get("source"), row.get("fg_cap"),
            row.get("outerFG"), fmt(row.get("fit_wall_seconds"), 1), row.get("terminal_reason"),
            fmt(row.get("canonical_gradient_max_abs"), 3), fmt(row.get("count_A")), fmt(row.get("count_B")),
            fmt(row.get("count_C")), fmt(row.get("fullJ")), fmt(row.get("fullJ_A")),
            fmt(row.get("whole_cell_rg"))))
    for label in ("initial-consensus", "initial-random", "baseline-046-G-full-J"):
        row = by_label.get(label, {})
        if not row:
            continue
        add("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            label, row.get("loss") or "-", row.get("solver") or "-", row.get("source"),
            row.get("fg_cap") if row.get("fg_cap") is not None else "-", row.get("outerFG"),
            fmt(row.get("fit_wall_seconds"), 1), row.get("terminal"),
            fmt(row.get("canonical_gradient_max_abs"), 3),
            fmt(row.get("count_A")), fmt(row.get("count_B")), fmt(row.get("count_C")),
            fmt(row.get("fullJ")), fmt(row.get("fullJ_A")), fmt(row.get("whole_cell_rg"))))
    add("")
    if fit_rows and by_label.get("baseline-046-G-full-J"):
        base = by_label["baseline-046-G-full-J"]
        best = min(fit_rows, key=lambda row: float(row["count_A"]))
        per_loss_best = {}
        for loss in ("A", "B", "C"):
            rows_loss = [row for row in fit_rows if row.get("loss") == loss]
            if rows_loss:
                per_loss_best[loss] = min(rows_loss, key=lambda row: float(row["count_A"]))
        _pairs = []
        for loss in ("A", "B", "C"):
            key = {"A": "count_A", "B": "count_B", "C": "count_C"}[loss]
            for source in ("consensus", "random"):
                raw_row = next(r for r in fit_rows if r["loss"] == loss and r["solver"] == "raw" and r["source"] == source)
                ms_row = next(r for r in fit_rows if r["loss"] == loss and r["solver"] == "ms" and r["source"] == source)
                _pairs.append((loss, source, float(raw_row[key]), float(ms_row[key]), float(ms_row[key]) - float(raw_row[key])))
        _ms_better = [item for item in _pairs if item[4] < 0.0]
        _exception = max(_pairs, key=lambda item: item[4])
        add("**共同 count_A 比较（本轮预注册采用的共同目标口径）**：baseline `%s` = %s；"
            "12 条新端点最好的 `%s` = %s（差 %+0.6f）；各 loss 最好：%s。" % (
                base["fit_id"], fmt(base["count_A"]), best["fit_id"], fmt(best["count_A"]),
                float(best["count_A"]) - float(base["count_A"]),
                "；".join("%s %s=%s" % (loss, per_loss_best[loss]["fit_id"], fmt(per_loss_best[loss]["count_A"]))
                          for loss in sorted(per_loss_best))))
        count_verdict = "该口径无改善" if float(best["count_A"]) > float(base["count_A"]) else "该口径有改善"
        add("")
        add("**结论（严格限定）：在本轮预注册的共同原 G count 口径下，12 条新端点的最好者仍未超过 046 baseline（%s）**。"
            "这只说明**共同 count 目标上没有改善**；`count_A` 是本轮预注册采用的共同目标口径（A 与 C 都是同一观测空间的"
            "归一化模型，B 是 classification/MAP 状态目标），因此**不对三 loss 做泛化数学比较**。"
            "本轮**不据此下任何结构结论**：结构恢复是否改善由评价侧的 R2 / inter / 空间读出回答（见 §6），"
            "phase/L2 恢复本轮不作声明。" % count_verdict)
        add("")
        add("**同一 source 下 solver 对比（真实事实，保留）**：6 组 同 source × 同 loss 的 raw vs ms 中，"
            "**5/6 组 multiscale 预条件降低了自身 count**（唯一例外为 %s-%s：ms %s vs raw %s）。逐组差值见下表。" % (
                _exception[0], _exception[1], fmt(_exception[3]), fmt(_exception[2])))
        add("")
        add("| loss | source | raw own count | ms own count | ms − raw | ms 更低 |")
        add("| --- | --- | --- | --- | --- | --- |")
        for loss in ("A", "B", "C"):
            for source in ("consensus", "random"):
                raw_row = next(r for r in fit_rows if r["loss"] == loss and r["solver"] == "raw" and r["source"] == source)
                ms_row = next(r for r in fit_rows if r["loss"] == loss and r["solver"] == "ms" and r["source"] == source)
                key = {"A": "count_A", "B": "count_B", "C": "count_C"}[loss]
                delta = float(ms_row[key]) - float(raw_row[key])
                add("| %s | %s | %s | %s | %+0.6f | %s |" % (loss, source, fmt(raw_row[key]), fmt(ms_row[key]),
                                                             delta, "yes" if delta < 0 else "no"))
        add("")
        add("注意 baseline 是上一轮 1988 混合尺度 FG 的事后采纳工作对照，与本轮 1502 全 fine FG 成本不同，不是等预算比较；"
            "分层的 count_A（A 系 %s–%s、B 系 %s–%s、C 系 %s–%s）只说明各 loss 优化的是不同目标。" % (
                fmt(min(float(r["count_A"]) for r in fit_rows if r["loss"] == "A")),
                fmt(max(float(r["count_A"]) for r in fit_rows if r["loss"] == "A")),
                fmt(min(float(r["count_A"]) for r in fit_rows if r["loss"] == "B")),
                fmt(max(float(r["count_A"]) for r in fit_rows if r["loss"] == "B")),
                fmt(min(float(r["count_A"]) for r in fit_rows if r["loss"] == "C")),
                fmt(max(float(r["count_A"]) for r in fit_rows if r["loss"] == "C"))))
    add("")
    add("恒等式核对（同一坐标/e/p/同常数）：`count_B - count_A = sum_C C*(-log gamma_max)/Nraw >= 0`，"
        "`count_C - count_B = (Noff/Nraw)*log(Zmax/Zsum) <= 0`。逐端点实际值与公式值见 "
        "`results/endpoint_manifest_pre_reference.json` 的 `B_minus_A_formula/actual` 与 `C_minus_B_formula/actual`。")
    add("")
    add("## 4. 预注册选择（open reference 之前冻结）")
    add("")
    add("| loss | solver | 选中的 source | 自身 count 差 | 规则 |")
    add("| --- | --- | --- | --- | --- |")
    for loss, block in sorted(selection.get("per_loss", {}).items()):
        for solver, row in sorted(block.get("source_selection", {}).items()):
            add("| %s %s | %s | %s | %s | %s |" % (loss, LOSS_NAME.get(loss, ""), solver,
                                                   row.get("selected_source"),
                                                   fmt(row.get("margin_consensus_minus_random"), 9),
                                                   row.get("rule")))
    add("")
    add("显示端点（每 loss 从自身 4 个 endpoint 按自身 count 选 1，先于 reference 冻结）：" +
        "；".join("`%s` (own count %s)" % (block["display_endpoint"]["fit_id"],
                                           fmt(block["display_endpoint"].get("count")))
                 for _loss, block in sorted(selection.get("per_loss", {}).items())))
    add("")
    add("三 loss 的数值不可直接互相宣布更优；跨 loss 解释统一使用共同 `count_A`/`fullJ_A` rescore。")
    add("")
    add("## 5. 无 reference 机制诊断")
    add("")
    if posterior:
        summary = posterior.get("summary", {})
        add("### 5.1 posterior 归属")
        add("")
        add("`gamma_s = t_s / sum_s t_s`（原 marginal），全部按 1,265,114 counts 加权；**不是真实 allele accuracy**。")
        add("")
        add("| fit | iter0 H_all | final H_all | final H_intra | final H_inter | final max posterior | final frac(gamma_max>=0.9) | final MAP-switch vs iter0 | Rg(iter0) | Rg(final) |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for fit, block in sorted(summary.items()):
            initial = block.get("iter0", {})
            final = block.get("final", {})
            add("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                fit, fmt(initial.get("entropy_all"), 4), fmt(final.get("entropy_all"), 4),
                fmt(final.get("entropy_intra"), 4), fmt(final.get("entropy_inter"), 4),
                fmt(final.get("max_posterior_all"), 4), fmt(final.get("frac_gamma_max_ge_0.9_all"), 4),
                fmt(final.get("map_switch_all_vs_iter0"), 4),
                fmt(initial.get("rg_whole_cell_all_5290_beads"), 4),
                fmt(final.get("rg_whole_cell_all_5290_beads"), 4)))
        add("")
        add("逐 checkpoint 全表：`diagnostics/posterior_all_states.tsv`；每 fit 一份 `diagnostics/posterior_<fit>.tsv`。")
        add("")
        add("本表数值全部由 `diagnostics/posterior_diagnostic.json` 已封存字段直接渲染（不手抄）。")
    if posterior and selection.get("per_loss"):
        add("")
        add("#### 已冻结 display 端点的 inter 归属距离摘要（距离以自己 whole-cell Rg 归一）")
        add("")
        add("| display endpoint | iter0 posterior inter d/Rg | final **动态** posterior inter d/Rg | final **冻结 iter0** posterior inter d/Rg |")
        add("| --- | --- | --- | --- |")
        for loss in ("A", "B", "C"):
            block = selection["per_loss"].get(loss)
            if not block:
                continue
            fit = block["display_endpoint"]["fit_id"]
            summary_fit = (posterior.get("summary") or {}).get(fit)
            if not summary_fit:
                continue
            add("| `%s` (%s) | %s | %s | %s |" % (
                fit, loss, fmt(summary_fit["iter0"].get("distance_over_rg_inter"), 7),
                fmt(summary_fit["final"].get("distance_over_rg_inter"), 7),
                fmt(summary_fit["final"].get("distance_over_rg_inter_frozen_iter0"), 7)))
        add("")
        add("解释（严格限定）：三种口径的相对大小只说明**后验归属变化会明显影响这一个距离摘要**，而 hard/max 状态本身也会"
            "随时间改变归属。冻结的 iter0 gamma **不是真实标签**，因此不能据此推断“结构改善全部来自投机”或“归属全错”；"
            "这些端点是否真的更接近真实结构，最终仍由 reference 指标（R2 / inter / 空间）判断。")
        add("")
    if cross:
        add("### 5.2 跨分辨率闭合（046 baseline 固定 1Mb coords/p）")
        add("")
        add("| 粗层 | 粗 loci | 映入粗 diag 的细 pair | 映入 diag 的 counts | counts 最大偏差 | KL(fine-sum||coarse-point) | Pearson(prob) | Pearson(log prob) | 细求和 cis/inter 质量 | 粗点 cis/inter 质量 |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for level, row in sorted(cross.get("levels", {}).items()):
            group = row.get("group_mass", {})
            add("| %s bp | %s | %s | %s | %s | %s | %s | %s | %s / %s | %s / %s |" % (
                level, row.get("coarse_n_loci"), row.get("fine_pairs_mapped_to_coarse_diag"),
                row.get("counts_mapped_to_coarse_diag"), fmt(row.get("max_abs_count_mismatch_vs_coarse_aggregate"), 1),
                fmt(row.get("kl_fine_sum_vs_coarse_point"), 5), fmt(row.get("pearson_probability"), 5),
                fmt(row.get("pearson_log_probability"), 5),
                fmt(group.get("fine_sum_cis_mass"), 4), fmt(group.get("fine_sum_inter_mass"), 4),
                fmt(group.get("point_cis_mass"), 4), fmt(group.get("point_inter_mass"), 4)))
        add("")
        add("190 chr-pair 贡献：`diagnostics/cross_resolution_chr_pairs.tsv`。2Mb 与 5Mb 本身不嵌套；不跨网格比绝对 NLL。")
        add("")
    if probe:
        add("### 5.3 局部可观测性 probe（仅 baseline）")
        add("")
        add("baseline whole-cell Rg = %s，p = %s；seed %s；方向先固定、整体去平移、whole-cell displacement RMS 归一；"
            "超 strict ball 则整体缩幅（实测未触发）；固定 e/p、不拟合。" % (
                fmt(probe.get("baseline_whole_cell_rg"), 6), fmt(probe.get("baseline_p"), 4), probe.get("seed")))
        add("")
        add("| probe | 幅度(×baseline Rg) | 实际位移幅度(坐标单位) | Noff/Nraw·KL(q_base‖q_probe) | data count NLL Δ(归一) | fullJ Δ(A) | 加权正则 Δ | rigid 后 RMS 形变 |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in probe.get("probes", []):
            add("| %s | %s%.2f | %s | %s | %s | %s | %s | %s |" % (
                row["probe"], "+" if row["sign"] > 0 else "-", row["fraction_of_baseline_rg"],
                fmt(row["actual_amplitude"], 6), "%.3e" % row["Noff_over_Nraw_times_kl"],
                fmt(row["data_count_nll_delta"], 6), fmt(row["fullJ_delta_A"], 6),
                fmt(row["regularization_delta_weighted"], 6), fmt(row["rms_deformation_after_proper_rigid"], 6)))
        add("")
        add("幅度列口径：`fraction` 是相对 baseline whole-cell Rg（%s）的乘数，`实际位移幅度` 是 probe 的坐标单位位移"
            "（`actual_amplitude`，与 baseline Rg 同单位），两者不是同一个量。" % fmt(probe.get("baseline_whole_cell_rg"), 6))
        add("")
        add("仅为局部方向可观测性，不是全局不可识别证明；probe coords+hash 在 `diagnostics/probe_coords/`。")
        add("")
    add("## 5.4 更正与口径记录")
    add("")
    add("- **PLAN.md §2 起点 seed 文本更正（冻结 PLAN 不改字节）**：PLAN 写 1Mb prolongation seed 为 consensus 5509 / random 4405，")
    add("  实际源 metadata 为：5Mb 层 `approved_014_blind_initialization` inner seed 1103/2207；2Mb 层 `multiresolution_warm_start`")
    add("  inner seed 4406/5510（candidate_base_seed 1103/2207）；1Mb 层 inner seed **4405(consensus)/5509(random)**。")
    add("  `045/source/prepare_inputs.py:274-276` 在 5Mb->2Mb 与 2Mb->1Mb 两步都传 1103/2207 作为 prolongation 参数。")
    add("  gate8 用 1103/2207 从冻结 045 5Mb 文件重跑 zero-optimization prolongation，与两个冻结 1Mb 文件**逐元素误差 0.000e+00**。")
    add("  因此冻结 PLAN 的两处问题：consensus/random 写反、以及把层内扰动 seed 当成了 prolongation 参数。起点身份以 045 文件 SHA 与逐元素复现为准，12 fit 初态未变。")
    add("- **FG 口径**：`18,024 = 12 × 1502` 是**正式矩阵的 FG 上限（并等于实际值，若每条都耗尽 cap）**，")
    add("  **不包含**工程门（含 12 cell × 2 FG 短集成）、已废弃的 launch-1 attempt、以及无 reference 诊断的重评分")
    add("  （posterior 遍历 iter0 + 每 10 accepted checkpoint + 真 final endpoint、端点三目标重评分、cross-resolution、8 个 probe）。")
    add("  这些是额外计算开销，本轮总计算量大于 18,024 FG，不得宣传为只有 18,024。")
    add("- **probe 口径**：`smooth_common_mode` 用每 chr DCT modes 1–3 且每 chr 均值为 0，20 个 chr 中心保持不变，主要测内部平滑形变；")
    add("  `local_independent_copies` 以逐 bead 扰动为主，只带很小随机中心漂移。因此 8 个 probe 的 KL>0 **不能**据此宣布")
    add("  “染色体中心摆位可辨识/充分约束”；本诊断未系统搜索中心摆位弱方向。真实摆位改善由统一 190-center / 760-copy-center 读出来回答。")
    add("- 分析层更正：正则按加权口径（bend 权 0.01）且 raw/weighted 分开；probe 的 count NLL delta 含 normalizer 并与 count_A delta 断言一致；")
    add("  posterior 加入真 final endpoint 并按 all/intra/inter 分列；cross chr-pair 表 20 cis + 190 inter 且逐行 KL 贡献求和等于总 KL；15 条记录统一 `fit_id`。")
    add("")
    add("## 6. 后置评价（评价侧）")
    add("")
    evaluation_section(add, RUN)
    add("")
    add("## 7. 诚实结论与限制")
    add("")
    add("- 本设计不是生物重复：n=1 细胞，20 chr 为关联测量，bootstrap/置换只描述细胞内技术/结构变异。")
    add("- baseline（046 G full-J）是**事后采纳的工作对照**，其 1988 FG 含低分辨率 stage，与本轮 1502 全 fine FG 成本不同。")
    add("- `exit code 0` 不代表科学成功；每条 fit 的真实终态见第 3 节 `terminal` 列。")
    add("- **12/12 fit 均为 `budget_not_converged`**（末态 canonical 梯度 2.3e-4 ~ 1.8e-3，均未达 `1e-6`）；因此本轮没有"
        "“已收敛”端点，任何“哪个方法更好”的结论都带有**预算已耗尽但仍未收敛**这一保留条件。")
    add("- **L2（沿整条染色体的一致拷贝身份恢复）本轮仍未证明**；本轮只做 R2 与空间读出，不做 R1/R3，不声称 L2，"
        "也不把拷贝标签当作母源/父源标签。")
    add("- 参考结构只在候选/initial/baseline/null 全部 hash 之后由评价侧打开（`reference_first_opened_utc` 见 §6），"
        "未参与初始化、停止、source 选择或超参选择；display 选择在打开参考之前冻结，未据 reference 改选。")
    add("- 本轮结果已交父侧验收；**未替换 `docs/CURRENT_BASELINE.md`**（baseline 仍是 046 G full-J 工作对照）。")
    add("")
    add("_本文件由脚本生成，生成时间 %s。_" % dt.datetime.now(dt.timezone.utc).isoformat())
    text = "\n".join(lines) + "\n"
    (RUN / "README.md").write_text(text, encoding="utf-8")
    print(json.dumps({"status": "ok", "path": str((RUN / "README.md")), "lines": len(lines)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
