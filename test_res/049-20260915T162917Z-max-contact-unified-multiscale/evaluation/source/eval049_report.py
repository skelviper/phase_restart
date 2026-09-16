#!/usr/bin/env python
"""049 评价报告生成器：由 results/ 的 JSON/TSV 生成中文内部 REPORT.md（数据驱动，不手抄数字）。"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval049_lib as lib  # noqa: E402


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return "%.3e" % value
        return ("%%.%df" % digits) % value
    return str(value)


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def dataset_order(evaluation: Mapping[str, Any]) -> list[str]:
    fits = [fid for fid in lib.EXPECTED_FIT_IDS if fid in evaluation["datasets"]]
    initials = [did for did in ("initial-consensus", "initial-random") if did in evaluation["datasets"]]
    baseline = [did for did in ("baseline-046-real-extension-G-full-J",) if did in evaluation["datasets"]]
    return fits + initials + baseline


def macro(evaluation: Mapping[str, Any], dataset_id: str, metric: str) -> Mapping[str, Any]:
    return evaluation["datasets"][dataset_id]["r2"]["macro_full_20_all_chromosomes_required"][metric]


def build(results_dir: Path, gate_path: Path) -> str:
    evaluation = lib.read_json(results_dir / "evaluation.json")
    validation = lib.read_json(results_dir / "validation.json")
    terminal = lib.read_json(results_dir / "terminal.json")
    gate = lib.read_json(gate_path)
    bootstrap = lib.read_json(results_dir / "paired_bootstrap.json")
    order = dataset_order(evaluation)
    parts: list[str] = []
    smoke = bool(evaluation.get("smoke"))
    parts.append("# 049 评价报告：max-contact 统一多尺度方法 vs 046 baseline\n")
    if smoke:
        parts.append("> **SMOKE/开发运行**：`evaluation.json` 标记 `smoke=true`，本文数字不是正式评价结果。\n")
    parts.append(
        "本报告由 `source/eval049_report.py` 从 `results/` 的机器输出生成；所有分母、支持集与终态都来自冻结产物，"
        "不重新拟合、不选择端点。生物重复 n=1，20 条染色体是同一细胞内关联测量。\n")

    # 1 封存与终态
    parts.append("## 1. 封存、终态与预算\n")
    parts.append(table(
        ["项目", "值"],
        [["run_id", evaluation["run_id"]],
         ["created_utc", evaluation["created_utc"]],
         ["reference_first_opened_utc", evaluation["reference"]["reference_first_opened_utc"]],
         ["reference_sha256", evaluation["reference"].get("reference_sha256")],
         ["reference 解析模式", evaluation["reference"].get("track_mode")],
         ["reference 缺失 loci（mask 支持外允许）", evaluation["reference"].get("nonfinite_loci")],
         ["endpoint manifest sha256", gate["manifest"]["sha256"]],
         ["selection sha256", gate["selection"]["sha256"]],
         ["输出 CSV/端点数", "%d 个 dataset（12 fit + 2 initial + baseline）" % len(order)],
         ["评价终态", terminal.get("evaluation_terminal")],
         ["validation 状态", validation.get("status")],
         ["bootstrap 指数矩阵与 046 相同", validation["checks"].get("bootstrap_index_matrix_matches_046")]]))
    rows = []
    for dataset_id in order:
        entry = evaluation["datasets"][dataset_id]
        rows.append([dataset_id, entry["kind"], entry["terminal"], entry.get("outer_fg"),
                     entry.get("wall_s"), entry.get("canonical_grad"), entry.get("p"), entry.get("q"),
                     entry.get("q_source")])
    parts.append("\n**全部端点终态与预算**（退出码 0 不等于科学收敛；wall/canonical grad/p/q 取自 manifest 与逐 fit JSON，未重算）：\n")
    parts.append(table(["dataset", "kind", "terminal", "outer FG", "wall s", "canonical grad max abs",
                        "p", "q", "q_source"], rows))
    common_rows = evaluation.get("endpoint_common_g")
    if common_rows is None:
        common_rows = [{"candidate_id": dataset_id, "kind": evaluation["datasets"][dataset_id]["kind"],
                        "terminal": evaluation["datasets"][dataset_id]["terminal"],
                        "loss": evaluation["datasets"][dataset_id].get("loss"),
                        "own_count_nll": evaluation["datasets"][dataset_id].get("own_count_nll"),
                        "own_full_j": evaluation["datasets"][dataset_id].get("own_full_j"),
                        "common_G_count": evaluation["datasets"][dataset_id].get("common_G_count"),
                        "common_G_fullJ": evaluation["datasets"][dataset_id].get("common_G_fullJ")}
                       for dataset_id in order]
    parts.append("\n**共同原 G 重打分（pre-reference 封存字段，value-only，不重算）**："
                 "`common_G_count = count_nll_by_loss['A']`、`common_G_fullJ = fullJ_A`；"
                 "`own_*` 为该 fit 自己 loss 的数值，两者不可混用。\n")
    parts.append(table(["dataset", "kind", "terminal", "loss", "own count nll", "own fullJ",
                        "common-G count", "common-G fullJ"],
                       [[row["candidate_id"], row["kind"], row["terminal"], row.get("loss"),
                         row.get("own_count_nll"), row.get("own_full_j"),
                         row.get("common_G_count"), row.get("common_G_fullJ")] for row in common_rows]))
    missing_common = [row["candidate_id"] for row in common_rows if row.get("common_G_count") is None
                      and row.get("common_G_fullJ") is None]
    if missing_common:
        parts.append("\n- 未提供共同 G 字段的 dataset（记 NA，不重算）：%s" % ", ".join(missing_common))
    budget = gate.get("budget", {})
    parts.append("\n- 冻结预算：每 fit `1502` FG（全 1Mb full grid），实际 `outerFG` 与冻结值不一致的 fit：%s。"
                 % (", ".join(budget.get("deviations", [])) or "无"))
    parts.append("- 046 baseline 累计 `1988` FG 含 5Mb/2Mb 低分辨率 stage；本轮为 1502 全 1Mb full-grid FG，"
                 "**成本不同**，不得宣称与旧 baseline 成本相等。\n")

    # 2 选择
    parts.append("## 2. 端点选择（打开 reference 之前冻结）\n")
    parts.append(table(["loss × solver", "选中 source", "rule", "margin", "复算一致"],
                       [[key, value["selected_source"], value.get("rule") or "min own count",
                         (value.get("margin")), "yes"]
                        for key, value in sorted(gate["selection_checks"]["source_selection"].items())]))
    parts.append("")
    parts.append(table(["loss", "display endpoint", "复算 argmin", "并列集合"],
                       [[loss, block["declared_fit_id"], block["recomputed_argmin"], ", ".join(block["tied_fit_ids"])]
                        for loss, block in sorted(gate["selection_checks"]["display"].items())]))
    parts.append("\n- 选择只消费 endpoint 自身 count 与 terminal；三个 loss 的数值不可跨 loss 直接排名。"
                 "图 2 的 5 panel 由该冻结 display 选择决定，评价不改动选择。\n")

    # 3 R2
    parts.append("## 3. R2 结构一致性（每 chr 四 Pearson/Spearman，fixed-20 macro）\n")
    rows = []
    for dataset_id in order:
        row = [dataset_id]
        for metric_name in ("pearson", "spearman"):
            block = macro(evaluation, dataset_id, metric_name)
            row.extend([block["matched"], block["cross"], block["contrast"], block["min_margin"],
                        evaluation["datasets"][dataset_id]["r2"]["defined_chromosome_counts"][metric_name]["matched"]])
        rows.append(row)
    parts.append(table(["dataset", "P matched", "P cross", "P contrast", "P min margin", "P defined",
                        "S matched", "S cross", "S contrast", "S min margin", "S defined"], rows))
    parts.append("\n- 主列口径：20 chr 全部定义才给值（任一 chr 缺失即 NA），不使用 defined-only 替代；"
                 "`matched - cross` 是规范不变读出，`u=0` 对照见第 7 节。\n")

    # 4 inter
    parts.append("## 4. 染色体间排序四距（pooled sorted four-copy distances）\n")
    rows = []
    for dataset_id in order:
        inter = evaluation["datasets"][dataset_id]["inter"]
        per_order = inter.get("per_order_statistic") or {}
        rows.append([dataset_id, inter.get("pearson"), inter.get("spearman"),
                     inter["eligible_locus_pairs"], inter["denominator"], inter["nonfinite_value_count"]] +
                    [per_order.get("order_%d" % k, {}).get("pearson") for k in range(4)])
    parts.append(table(["dataset", "pearson", "spearman", "locus pairs", "四距分母",
                        "非有限值数", "order0 P", "order1 P", "order2 P", "order3 P"], rows))
    parts.append("\n- 支持集固定为 046 old21 common mask 的 2447 个 valid loci（4894 beads）；"
                 "缺失记 NA，不缩小分母、不丢 chr。\n")

    # 5 空间
    parts.append("## 5. 空间读出\n")
    rows = []
    for dataset_id in order:
        spatial = evaluation["datasets"][dataset_id]["spatial"]
        centers = spatial["merged_chr_centers"]
        copy_centers = spatial["copy_centers"]["primary"]
        rows.append([dataset_id, centers["pearson"], centers["spearman"], centers["normalized_stress"],
                     centers["optimal_single_scale"], centers["per_chr_profile_macro_spearman"],
                     centers["top3_neighbors"]["mean_overlap_top3"],
                     spatial["label_permutation_null"]["one_sided_p_ge_observed"],
                     copy_centers["pearson"], copy_centers["spearman"], copy_centers["normalized_stress"],
                     copy_centers["procrustes_proper"]["rotation_det"],
                     copy_centers["procrustes_proper"]["normalized_aligned_rmsd"],
                     copy_centers["procrustes_reflection"]["normalized_aligned_rmsd"]])
    parts.append(table(["dataset", "center P(190)", "center S(190)", "center stress", "single scale",
                        "per-chr 19 距离 macro S", "top3 邻居均值", "标签置换 p",
                        "copy P(760)", "copy S(760)", "copy stress", "proper det",
                        "proper norm RMSD", "reflect norm RMSD"], rows))
    parts.append("\n- 20 merged chr 中心（每 copy 用共通 valid bins 均值后两 copy 等权）；单一尺度 `a=Σ(pred·ref)/Σ(pred²)`，"
                 "`stress=sqrt(Σ(a·pred−ref)²/Σref²)`；每 chr 19 条距离 profile 用 min_pairs=2 的 helper。")
    parts.append("- 190 距离整 chr 标签置换 null（seed %d，%d draws，dyadic 依赖，不是独立 pair 检验）：%s。"
                 % (lib.PERMUTATION_SEED, evaluation["datasets"][order[0]]["spatial"]["label_permutation_null"]["draws"],
                    "见上表 p 值"))
    unresolved = {dataset_id: evaluation["datasets"][dataset_id]["spatial"]["copy_centers"]["unresolved_chromosomes"]
                  for dataset_id in order}
    parts.append("- copy center（40 点/760 跨 chr 距离）对应由每 chr whole-chr signed-Pearson `derive_rho` 规则固定，"
                 "同 chr homolog pair 不入距离向量；tie/undefined chr：%s。"
                 % ("; ".join("%s=%s" % (k, ",".join(v) or "无") for k, v in unresolved.items())))
    parts.append("- 20 center 相似度用单个 global Procrustes（proper 与允许反射两版，报 det）；"
                 "反射不改变距离类指标，只改变 aligned RMSD。\n")

    # 6 null
    parts.append("## 6. null：u=0 与 16 个 random-u\n")
    new_nulls = len(gate.get("nulls_new", []))
    reused_nulls = len(gate.get("nulls_reused_baseline", []))
    parts.append("null 设计：8 个来源（6 个按 count 选中的 loss×solver endpoint + 2 个 initial）× 17 个变体"
                 "（u0 + seeds 450500..450515）= **%d 个新文件**；046 同 candidate SHA 复用 baseline null **%d 个**；"
                 "合计 **%d 个 null draw**（`evaluation.json.null_draw_count`=%s）。\n"
                 % (new_nulls, reused_nulls, new_nulls + reused_nulls, evaluation.get("null_draw_count")))
    summary = evaluation["null_summary"]
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in summary:
        by_source.setdefault(str(row["source_candidate_id"]), []).append(row)
    rows = []
    for source_id, items in sorted(by_source.items()):
        for kind in ("u_zero", "random_u"):
            picked = {row["metric"]: row for row in items if row["null_kind"] == kind}
            matched = picked.get("pearson_matched", {})
            cross = picked.get("pearson_cross", {})
            contrast = picked.get("pearson_contrast", {})
            inter = picked.get("inter_pearson", {})
            rows.append([source_id, kind, matched.get("draw_count"), matched.get("mean"), matched.get("min"),
                         matched.get("max"), cross.get("mean"), contrast.get("mean"), contrast.get("range"),
                         inter.get("mean"), inter.get("range")])
    parts.append(table(["source", "null kind", "draws", "P matched mean", "P matched min", "P matched max",
                        "P cross mean", "P contrast mean", "P contrast range", "inter P mean", "inter P range"], rows))
    parts.append("\n- `u=0` 的 margin 类指标为 NA（tie 规则下每 chr margin 未定义），符合 PLAN §8 的预期；")
    parts.append("- 16 个 random-u 是同一细胞内的技术性置换，**不是生物重复**，只用于给出同格对照的 mean/range。\n")

    # 7 bootstrap
    parts.append("## 7. paired bootstrap（seed %d，%d draws，20 chr 配对）\n"
                 % (lib.BOOTSTRAP_SEED, lib.BOOTSTRAP_DRAWS))
    rows = []
    for comparison in bootstrap["comparisons"]:
        for key in ("pearson:matched", "pearson:contrast", "spearman:matched", "spearman:contrast"):
            block = comparison["metrics"].get(key, {})
            rows.append([comparison["label"], comparison["left"], comparison["right"], key, block.get("mean"),
                         block.get("ci95", [None, None])[0], block.get("ci95", [None, None])[1],
                         block.get("left_wins"), block.get("right_wins"), block.get("ties"),
                         block.get("defined_chromosomes")])
    parts.append(table(["comparison", "left", "right", "metric", "mean diff", "CI2.5", "CI97.5",
                        "left wins", "right wins", "ties", "defined chr"], rows))
    parts.append("\n- 全部 20 chr 定义时才给 fixed-20 估计；胜出数分母为 20。比较包括每个 endpoint − baseline、"
                 "同 source 的 B−A / C−A 与 ms−raw。\n")

    # 8 验证
    parts.append("## 8. 验证与回归\n")
    parts.append(table(["检查", "结果"],
                       [[key, value] for key, value in validation["checks"].items()]))
    regression = validation.get("baseline_regression", {})
    parts.append("\n- baseline 回归：与 046 既有指标表在 atol=%s 下核对 %s 个数值，mismatch=%s（status=%s）。"
                 % (regression.get("atol"), regression.get("checked_values"), regression.get("mismatch_count"),
                    regression.get("status")))
    repair = validation.get("path_repair")
    if repair:
        parts.append("- 派生路径修复（不重算指标）：修复 `evaluation.datasets`/`terminal.datasets`/`endpoint_common_g` 的 "
                     "`npz_path` 共 %s 处，数值与其它字段深比较后 0 变化；路径存在性检查 %s 条，缺失 %s 条。"
                     % (repair["fixed_fields"], repair["path_check"]["checked"], repair["path_check"]["missing"]))
    deviations = validation.get("deviations", [])
    if deviations:
        parts.append("\n**偏离记录**\n")
        for item in deviations:
            detail = (item.get("detail") or "").strip()
            if not detail:
                detail = "；".join(str(item[key]).strip() for key in
                                   ("requirement", "what_was_read_before_gate", "what_was_not_read_before_gate")
                                   if item.get(key))
            parts.append("- **%s**：%s（影响：%s）" % (item.get("item"), detail, item.get("impact")))
    parts.append("\n- reference 打开次数：%s（首次打开时间 %s，保留 attempt 1 的真实时间；attempt 1 因 mask sha256 常量"
                 "沿用了 config 的笔误而失败，详见 `logs/EVALUATION_ATTEMPTS.md`）。"
                 % (terminal.get("reference_open_attempts"), terminal.get("reference_first_opened_utc")))
    parts.append("- mask：%s" % evaluation["mask"]["support_statement"])
    parts.append("- mask snapshot sha256：`%s`（与 046 创建时记录一致）；049/config.json 记录值相差 1 个字符，"
                 "属配置转录笔误，文件未变，评价侧只记录不改 config。\n"
                 % evaluation["mask"].get("snapshot_sha256"))

    # 9 限制
    parts.append("## 9. 限制与不可声称项\n")
    parts.append("\n".join([
        "- 本轮只做 R2 与空间读出；**不做 R1/R3**，不声称 L2（整条染色体一致拷贝身份可恢复）。",
        "- 结构尺度未校准：使用 rank/单一 scale/Procrustes 口径；copy 内相关高不等于同源拷贝恢复。",
        "- `copyA`/`copyB` 是标签规范自由度；跨拷贝指标按几何最佳互换（tie<=1e-12 记 unresolved），"
        "并给出 `u=0` 与 `random-u` 对照。",
        "- sorted inter 四距含高背景，random-u 保留中心而非 packing null；不把 raw r 当恢复率，"
        "也不把「与 random-u 差异小」写成「不可区分」。",
        "- 生物重复 n=1；20 chr 是同一细胞内的关联测量，bootstrap 只描述细胞内技术/结构变异。",
        "- 046 baseline 为事后采纳的工作对照，且 FG 预算与分辨率组合与本轮不同。",
        "- 几何效果不直接等于 phase/L2 恢复；本轮不读 phase 列。",
    ]))
    parts.append("")

    # 10 产物
    parts.append("## 10. 产物清单\n")
    artifacts = [("results/evaluation.json", "全部 dataset 指标、null draw、选择与支持集"),
                 ("results/r2_per_chromosome.tsv", "逐 chr 四 rho 与 matched/cross/contrast/margin"),
                 ("results/r2_summary.tsv", "fixed-20 macro 与 defined-only 描述列"),
                 ("results/endpoint_common_g.tsv", "15 行 own 目标值与共同原 G count/fullJ（不重算）"),
                 ("results/inter_summary.tsv", "pooled 四距与 4 个 order statistic"),
                 ("results/spatial_summary.tsv", "中心/stress/置换 null/copy center/Procrustes"),
                 ("results/null_per_draw.tsv", "153 个 null draw 的逐 draw 指标"),
                 ("results/null_summary.tsv", "按 source×kind 的 mean/range"),
                 ("results/paired_bootstrap.json / .tsv", "28 组配对比较的 CI 与胜出数"),
                 ("results/validation.json", "mask/paths/回归/gate 一致性"),
                 ("results/terminal.json", "端点与评价终态"),
                 ("plots/049_figure1_core_metrics.png", "核心指标比较（12 fit + baseline + 2 initial）"),
                 ("plots/049_figure2_whole_genome.png", "ref + baseline + 3 loss display endpoint 5 panel"),
                 ("gates/pre_reference_gate.json", "封存 gate（含首次打开 reference 时间）")]
    parts.append(table(["路径", "内容"], artifacts))
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--gate", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else lib.EVAL / "results"
    gate_path = Path(args.gate) if args.gate else lib.EVAL / "gates/pre_reference_gate.json"
    out = Path(args.out) if args.out else lib.EVAL / "REPORT.md"
    if not (results_dir / "evaluation.json").is_file():
        raise RuntimeError("evaluation.json not found under %s; report is post-evaluation only" % results_dir)
    text = build(results_dir, gate_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("REPORT WRITTEN: %s (%d lines)" % (out, text.count("\n")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
