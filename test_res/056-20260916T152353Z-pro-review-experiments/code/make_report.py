"""056最小结果表、三张英文图、预算/哈希清单与中文内部README。"""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(__file__).resolve().parent.parent
ROOT = RUN.parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def style():
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "figure.dpi": 120, "savefig.dpi": 300, "axes.linewidth": 0.6})


def plot_experiment1(exp1):
    original = exp1["rows"][0]
    counts = Counter(row["kind"] for row in exp1["rows"])
    expected = {"original": 1, "whole": 20, "splice": 8, "u0": 1, "random_u": 16}
    if dict(counts) != expected:
        raise RuntimeError("experiment1 plotting inventory changed: %r != %r" % (dict(counts), expected))

    def deltas(rows):
        return (np.asarray([row["offdiag_data_nat_per_contact"]
                            - original["offdiag_data_nat_per_contact"] for row in rows]),
                np.asarray([sum(row["penalties_weighted"].values())
                            - sum(original["penalties_weighted"].values()) for row in rows]))

    splices = [row for row in exp1["rows"] if row["kind"] == "splice"]
    whole = [row for row in exp1["rows"] if row["kind"] == "whole"]
    random_u = [row for row in exp1["rows"] if row["kind"] == "random_u"]
    u0 = [row for row in exp1["rows"] if row["kind"] == "u0"]
    sx, sy = deltas(splices); wx, wy = deltas(whole); rx, ry = deltas(random_u); ux, uy = deltas(u0)
    if not (np.max(np.abs(wx)) < 1e-10 and np.max(np.abs(wy)) < 1e-10):
        raise RuntimeError("whole-chromosome strict controls are not at the invariant origin")
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].scatter(sx, sy, s=18, color="#d55e00", label="8 suffix splices", zorder=3)
    axes[0].scatter([0], [0], marker="*", s=50, color="#333333",
                    label="20 whole-chr swaps (all zero)", zorder=4)
    for row, x, y in zip(splices, sx, sy):
        label = row["state_id"].replace("splice-", "")
        offset, horizontal, vertical = {
            "chr1-065": ((-3, 3), "right", "baseline"),
            "chr1-130": ((-3, 3), "right", "baseline"),
            "chr19-041": ((3, -8), "left", "top"),
            "chrX-057": ((-4, -8), "right", "baseline"),
        }.get(label, ((2, 2), "left", "baseline"))
        axes[0].annotate(label, (x, y), xytext=offset, ha=horizontal, va=vertical,
                         textcoords="offset points", fontsize=7)
    axes[0].axvline(0, color="0.7", lw=0.6); axes[0].axhline(0, color="0.7", lw=0.6)
    axes[0].set(xlabel="Off-diagonal data delta (nat/contact)",
                ylabel="Weighted regularizer delta", title="Strict point-set controls (local scale)")
    axes[0].margins(x=0.08, y=0.15)
    axes[0].legend(frameon=False, loc="best")
    axes[1].scatter(rx, ry, s=18, color="#0072b2", label="16 random-u")
    axes[1].scatter(ux, uy, s=28, marker="s", color="#555555", label="u=0")
    axes[1].axvline(0, color="0.7", lw=0.6); axes[1].axhline(0, color="0.7", lw=0.6)
    axes[1].set(xlabel="Off-diagonal data delta (nat/contact)",
                ylabel="Weighted regularizer delta",
                title="Auxiliary controls (not point-set preserving)")
    axes[1].ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Fixed-state perturbation response; panel scales differ", fontsize=7)
    fig.tight_layout(); fig.savefig(RUN / "plots/experiment1_data_regularizer.png"); plt.close(fig)


def plot_experiment2():
    path = RUN / "results/experiment2_gradients.npz"
    with np.load(path, allow_pickle=False) as payload:
        values = {}
        for denominator in ("Nraw", "Noff"):
            for diag in ("shift", "doubled"):
                for exposure in ("locked_original", "production_recomputed"):
                    base = np.asarray(payload["original__%s__%s" % (exposure, denominator)])
                    changed = np.asarray(payload["%s__%s__%s" % (diag, exposure, denominator)])
                    values[(denominator, diag, exposure)] = float(
                        np.linalg.norm(changed - base) / np.linalg.norm(base))
    if values[("Noff", "doubled", "locked_original")] >= 1e-12:
        raise RuntimeError("locked-e doubled diagonal is not invariant under fixed Noff")
    labels = ["Shift", "Doubled"]
    x = np.arange(2); width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    for ax, denominator, title in zip(axes, ("Nraw", "Noff"),
                                      ("Nraw_variant normalization", "Fixed Noff normalization")):
        ax.bar(x - width / 2,
               [values[(denominator, key, "locked_original")] for key in ("shift", "doubled")],
               width, label="Locked e", color="#999999")
        ax.bar(x + width / 2,
               [values[(denominator, key, "production_recomputed")] for key in ("shift", "doubled")],
               width, label="Recomputed e", color="#009e73")
        ax.set_xticks(x, labels); ax.set_title(title); ax.legend(frameon=False)
    axes[0].set_ylabel("Relative L2 change in physical-x gradient")
    fig.suptitle("Relative to original diagonal state under the same normalization", fontsize=7)
    fig.tight_layout(); fig.savefig(RUN / "plots/experiment2_gradient_response.png"); plt.close(fig)
    return values


def plot_experiment3(nonref, reference):
    heldout = {row["seed"]: row["heldout_gain_original_minus_offdiag_nat_per_contact"]
               for row in nonref["heldout_gains"]}
    structural = {row["seed"]: row["matched_delta_offdiag_minus_original"]
                  for row in reference["experiment3_comparisons"]}
    margins = {row["seed"]: row["min_margin_delta_offdiag_minus_original"]
               for row in reference["experiment3_comparisons"]}
    seeds = sorted(heldout)
    colors = ("#0072b2", "#d55e00")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for seed, color in zip(seeds, colors):
        axes[0].plot([0, 1], [0, heldout[seed]], marker="o", color=color, label=str(seed))
        axes[0].annotate("%+.4f" % heldout[seed], (1, heldout[seed]), xytext=(3, 0),
                         textcoords="offset points", va="center", fontsize=7)
    axes[0].axhline(0.01, color="0.4", ls="--", lw=0.7)
    axes[0].set_xticks([0, 1], ["Original e", "Offdiag e"])
    axes[0].set_ylabel("Held-out NLL gain (nat/contact)"); axes[0].set_title("Held-out contact score")
    for seed, color in zip(seeds, colors):
        axes[1].plot([0, 1], [0, structural[seed]], marker="o", color=color, label=str(seed))
        axes[1].annotate("%+.4f" % structural[seed], (1, structural[seed]), xytext=(3, 0),
                         textcoords="offset points", va="center", fontsize=7)
    axes[1].axhline(-0.02, color="0.4", ls="--", lw=0.7)
    axes[1].set_xticks([0, 1], ["Original e", "Offdiag e"])
    axes[1].set_ylabel("Matched Pearson delta"); axes[1].set_title("20-chromosome macro structure")
    for seed, color in zip(seeds, colors):
        axes[2].plot([0, 1], [0, margins[seed]], marker="o", color=color, label=str(seed))
        axes[2].annotate("%+.4f" % margins[seed], (1, margins[seed]), xytext=(3, 0),
                         textcoords="offset points", va="center", fontsize=7)
    axes[2].axhline(0, color="0.4", ls="--", lw=0.7)
    axes[2].set_xticks([0, 1], ["Original e", "Offdiag e"])
    axes[2].set_ylabel("Minimum copy-margin delta")
    axes[2].set_title("20-chromosome macro weak-copy margin")
    axes[2].legend(frameon=False, title="Init seed")
    gate_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5}
    axes[0].text(0.03, 0.97, "Gate: gain >= +0.01", transform=axes[0].transAxes,
                 va="top", bbox=gate_box)
    axes[1].text(0.03, 0.97, "Gate: delta >= -0.02", transform=axes[1].transAxes,
                 va="top", bbox=gate_box)
    axes[2].text(0.03, 0.97, "Both < -1e-10: consistent degradation",
                 transform=axes[2].transAxes, va="top", bbox=gate_box)
    fig.tight_layout(); fig.savefig(RUN / "plots/experiment3_paired_outcomes.png"); plt.close(fig)


def main():
    style()
    exp1 = read_json(RUN / "results/experiment1.json")
    exp2 = read_json(RUN / "results/experiment2.json")
    nonref = read_json(RUN / "results/experiment3_nonreference.json")
    reference = read_json(RUN / "reference_eval/metrics.json")
    training = read_json(RUN / "logs/training_terminal.json")
    plot_experiment1(exp1)
    gradient_values = plot_experiment2()
    plot_experiment3(nonref, reference)
    nonref_by = {row["fit_id"]: row for row in nonref["rows"]}
    ref_by = {row["candidate_id"]: row for row in reference["summaries"]}
    stage_by = {fit["fit_id"]: fit for fit in training["fits"]}
    with (RUN / "reference_eval/per_chromosome.tsv").open("r", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle, delimiter="\t"))
    best_ties = [row for row in reference_rows if row["best_orientation"] == "unresolved_tie"]
    affected_exp3_ties = [row for row in best_ties if row["experiment"] == "3"]
    if affected_exp3_ties:
        raise RuntimeError("existing experiment3 rows require deterministic tie correction")
    if len(best_ties) != 20 or any(row["candidate_id"] != "exp1/u0" for row in best_ties):
        raise RuntimeError("unexpected existing best-orientation tie inventory")
    tie_audit = {
        "best_orientation_tie_rows": len(best_ties),
        "tie_rows_by_candidate": {"exp1/u0": 20},
        "experiment3_endpoint_or_splice_rows_affected": 0,
        "existing_reference_metrics_changed": False,
        "reason": "all existing ties are exp1/u0 evaluated along the frozen baseline fixed mapping",
    }
    endpoint_rows = []
    for seed in (560101, 560102):
        for condition in ("G-original", "G-offdiag-e"):
            fit_id = "%s-seed%d" % (condition, seed)
            score = nonref_by[fit_id]
            structural = ref_by["exp3/" + fit_id]["macro"]
            terminal = stage_by[fit_id]["stages"][-1]
            endpoint_rows.append({
                "fit_id": fit_id, "condition": condition, "seed": seed,
                "training_fraction": 0.8,
                "heldout_test_Noff": score["test_Noff"],
                "heldout_test_nll_nat_per_offdiag": score["test_primary_nll_nat_per_offdiag"],
                "pearson_same_macro": structural["same"],
                "pearson_cross_macro": structural["cross"],
                "mean_margin_A": structural["margin_A"],
                "mean_margin_B": structural["margin_B"],
                "mean_min_margin": structural["min_margin"],
                "canonical_gradient_max_abs_1Mb": terminal["canonical_gradient_max_abs"],
                "terminal_1Mb": terminal["status"],
                "terminal_reason_1Mb": terminal["terminal_reason"],
            })
    comparisons = []
    for seed in (560101, 560102):
        a = nonref_by["G-original-seed%d" % seed]
        b = nonref_by["G-offdiag-e-seed%d" % seed]
        ra = ref_by["exp3/G-original-seed%d" % seed]
        rb = ref_by["exp3/G-offdiag-e-seed%d" % seed]
        ta = stage_by[a["fit_id"]]["stages"][-1]
        tb = stage_by[b["fit_id"]]["stages"][-1]
        ratio = max(ta["canonical_gradient_max_abs"], tb["canonical_gradient_max_abs"]) / max(
            min(ta["canonical_gradient_max_abs"], tb["canonical_gradient_max_abs"]), 1e-12)
        convergence_mismatch = ((ta["status"] == "converged") != (tb["status"] == "converged"))
        gradient_imbalance = (ratio > 10 and max(ta["canonical_gradient_max_abs"],
                                                tb["canonical_gradient_max_abs"]) > 1e-6)
        comparisons.append({
            "seed": seed,
            "heldout_gain_nat_per_contact": a["test_primary_nll_nat_per_offdiag"] - b["test_primary_nll_nat_per_offdiag"],
            "heldout_gate_ge_0_01": a["test_primary_nll_nat_per_offdiag"] - b["test_primary_nll_nat_per_offdiag"] >= 0.01,
            "matched_delta_offdiag_minus_original": rb["macro"]["same"] - ra["macro"]["same"],
            "matched_gate_ge_minus_0_02": rb["macro"]["same"] - ra["macro"]["same"] >= -0.02,
            "min_margin_delta_offdiag_minus_original": rb["macro"]["min_margin"] - ra["macro"]["min_margin"],
            "canonical_gradient_ratio": ratio,
            "original_1Mb_status": ta["status"],
            "offdiag_1Mb_status": tb["status"],
            "one_converged_other_not": convergence_mismatch,
            "canonical_gradient_imbalance": gradient_imbalance,
            "optimization_imbalance_alarm": bool(convergence_mismatch or gradient_imbalance),
            "original_splice_positive_count": a["splice_positive_count"],
            "offdiag_splice_positive_count": b["splice_positive_count"],
            "original_splice_median": a["splice_median_delta_nat_per_offdiag"],
            "offdiag_splice_median": b["splice_median_delta_nat_per_offdiag"],
        })
    consistent_margin_degradation = all(row["min_margin_delta_offdiag_minus_original"] < -1e-10
                                        for row in comparisons)
    consistent_splice_degradation = (all(row["offdiag_splice_positive_count"]
                                         < row["original_splice_positive_count"] for row in comparisons)
                                     or all(row["offdiag_splice_median"]
                                            < row["original_splice_median"] - 1e-10 for row in comparisons))
    optimization_gate_pass = not any(row["optimization_imbalance_alarm"] for row in comparisons)
    heldout_gate_pass = all(row["heldout_gate_ge_0_01"] for row in comparisons)
    matched_gate_pass = all(row["matched_gate_ge_minus_0_02"] for row in comparisons)
    margin_gate_pass = not consistent_margin_degradation
    splice_gate_pass = not consistent_splice_degradation
    adoption_gates = {
        "heldout_gain_both_seeds": heldout_gate_pass,
        "matched_delta_both_seeds": matched_gate_pass,
        "no_consistent_min_margin_degradation": margin_gate_pass,
        "no_consistent_splice_degradation": splice_gate_pass,
        "no_preregistered_optimization_imbalance_alarm": optimization_gate_pass,
    }
    adoption_recommended = all(adoption_gates.values())
    failure_reasons = [name for name, passed in adoption_gates.items() if not passed]
    status = {
        "schema": "p9016-056-final-status-v2", "experiment1": exp1["status"],
        "experiment2": exp2["status"], "experiment3_training": training["status"],
        "comparisons": comparisons,
        "consistent_min_margin_degradation": consistent_margin_degradation,
        "consistent_splice_degradation": consistent_splice_degradation,
        "adoption_gates": adoption_gates,
        "adoption_recommended": adoption_recommended,
        "adoption_failure_reasons": failure_reasons,
        "method_status": ("offdiag_e_adoption_recommended" if adoption_recommended
                          else "offdiag_e_not_adopted_prespecified_gates_failed"),
        "heldout_prediction_interpretation": "uncertain_threshold_not_met",
        "global_method_falsification_claimed": False,
        "new_l1_claim": False,
        "historical_l1_status": "pre-existing evidence supports chr1; chrX evidence is marginal",
        "l2_supported_by_this_round": False,
        "l3_claimed": False,
        "reference_tie_branch_audit": tie_audit,
        "endpoint_absolute_results": endpoint_rows,
    }
    write_json(RUN / "results/final_status.json", status)
    budget = {
        "experiment1": {"objective_fg": 46, "fg_safety_cap": 48,
                        "contact_forward_sweeps": 46, "repulsion_physics_sweeps": 46,
                        "note": "Each FG computed contact outputs and one separate physics/repulsion pair sweep."},
        "experiment2": {"value_gradient_fg": 12, "contact_forward_backward_sweeps": 24},
        "experiment3_training": {"fg_actual": training["outer_fg_actual"], "fg_cap": 6008,
                                 "fine_fg_actual": training["fine_1Mb_fg_actual"], "fine_fg_cap": 1944,
                                 "contact_forward_backward_sweeps": training["estimated_training_pair_kernel_sweeps"],
                                 "repulsion_full_pair_sweeps": training["outer_fg_actual"],
                                 "total_physical_full_pair_sweeps": 3 * training["outer_fg_actual"],
                                 "launcher_field_scope_note": "estimated_training_pair_kernel_sweeps is contact-only; final ledger adds one repulsion sweep per FG",
                                 "outside_auxiliary_cap": True},
        "experiment3_auxiliary": {"actual_full_pair_sweeps": nonref["auxiliary_total_full_pair_sweeps"],
                                  "cap": nonref["auxiliary_cap"],
                                  "contact_forward": nonref["auxiliary_eligible_contact_pair_forward_sweeps"],
                                  "regularizer": nonref["auxiliary_regularizer_full_pair_sweeps"]},
        "reference": {"cpu_distance_work_separately_accounted": True,
                      "candidate_states": reference["candidate_count"], "pairs_per_state": 157529,
                      "chromosomes_per_state": 20,
                      "distance_arrays_per_chromosome": 4,
                      "distance_array_computations": reference["candidate_count"] * 20 * 4,
                      "pearson_correlations_per_chromosome": 4,
                      "pearson_correlations": reference["candidate_count"] * 20 * 4,
                      "process_cpu_seconds": None,
                      "wall_seconds": None,
                      "timing_status": "not_recorded; no rerun performed",
                      "count_basis": "existing evaluator code computes candidate A/B and reference mat/pat distance arrays, then four Pearson correlations, per chromosome and state"},
    }
    write_json(RUN / "results/budget_ledger.json", budget)
    with (RUN / "results/summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoint_rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(endpoint_rows)
    artifacts = []
    for directory in ("results", "reference_eval", "plots", "logs", "inputs"):
        for path in sorted((RUN / directory).glob("**/*")):
            if path.is_file() and path.name != "artifact_hashes.json":
                artifacts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                  "sha256": sha(path)})
    write_json(RUN / "results/artifact_hashes.json", {"artifacts": artifacts})
    primary_shift = exp2["sensitivity"]["primary_shift_production_Nraw_L2_relative"]
    lines = [
        "# 056 专业审查实验终态", "",
        "## 结论", "",
        "- 实验1和实验2硬门均为 PASS。实验1的8/8固定suffix splice data delta为正，median为 `%.12g nat/offdiag contact`。" % exp1["scientific_gate"]["median_delta_nat_per_offdiag"],
        "- 实验2 production exposure 下shift的physical-x gradient L2相对变化为 `%.6f`；这是高优先级敏感性，不是停止实验3的门。" % primary_shift,
        "- 实验3的两个初始化只表示优化重复，不是生物学重复。四条fit真实终态见 `logs/training_terminal.json`。", "",
        "## 实验3配对结果", "",
        "| seed | held-out gain | matched delta | min-margin delta | grad ratio | optimization imbalance alarm |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in comparisons:
        lines.append("| %d | %.6g | %.6g | %.6g | %.4g | %s |" % (
            row["seed"], row["heldout_gain_nat_per_contact"], row["matched_delta_offdiag_minus_original"],
            row["min_margin_delta_offdiag_minus_original"], row["canonical_gradient_ratio"],
            "yes" if row["optimization_imbalance_alarm"] else "no"))
    lines += ["", "### 四个endpoint绝对值", "",
              "下表四个endpoint均由同一个固定80% record-train fold拟合；test NLL来自共同20% record-test，结构列来自共同055 support。它们不是046 full-data baseline，不能与full-data baseline作训练效果横比。", "",
              "| condition | seed | test NLL | same | cross | mean margin A | mean margin B | mean min-margin | 1Mb canonical grad | terminal |",
              "|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|" ]
    for row in endpoint_rows:
        lines.append("| %s | %d | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.4g | %s/%s |" % (
            row["condition"], row["seed"], row["heldout_test_nll_nat_per_offdiag"],
            row["pearson_same_macro"], row["pearson_cross_macro"], row["mean_margin_A"],
            row["mean_margin_B"], row["mean_min_margin"], row["canonical_gradient_max_abs_1Mb"],
            row["terminal_1Mb"], row["terminal_reason_1Mb"]))
    lines += ["", "本轮预注册门下的采纳决策：`%s`；`adoption_recommended=%s`。" %
              (status["method_status"], str(status["adoption_recommended"]).lower()),
              "失败门：`%s`。" % "`, `".join(status["adoption_failure_reasons"]), "",
              "## 范围与限制", "",
              "- `new_l1_claim=false`：本轮实验1是已拟合baseline上的in-sample固定x/e/p扰动，held-out只比较两个diploid exposure条件且没有1-copy/consensus预测对照，因此不构成新的L1证明。既有chr1支持与chrX边缘证据仅作为历史状态保留。",
              "- 本轮能直接陈述：G数据项排斥预定splices；diag counts会通过e改变physical-x gradient；offdiag-e未满足预注册采纳门。",
              "- held-out方向在两seed不一致且均未达到+0.01，因此预测证据为不确定/未达门；结构与min-margin一致退化使本轮不采纳。该决策不是对offdiag-e方法的全局证伪。",
              "- 本轮不证明整条染色体拷贝身份已恢复（L2），也不提出‘任何无SNP方法都无效’（L3）结论。",
              "- 四条1Mb fit均为`budget_not_converged`。梯度比和收敛状态未触发预注册优化失衡警报，但这不证明两条件优化等价或已经收敛。",
              "- 实验1 perturbation 的same/cross/contrast沿055 frozen baseline mapping（whole-chromosome swap同步transport mapping）计算，因此fixed-map contrast可以为负；`direct/swapped`及其max/min只作为best-global secondary，不能把全部contrast误称为绝对差。",
              "- reference tie branch审计：现有20个best-orientation ties全部为`exp1/u0`且沿frozen fixed mapping评价；4 endpoints和32 splices受影响行数为0，故未重跑reference距离、现有数值不变。未来unresolved tie保留signed per-copy margins。",
              "- 输入全部readID为`.`，因此固定record hash split不能隔离未知分子身份；train/test是record级而非molecule级。",
              "- 四个endpoint及其splices共同使用055 frozen legacy support：20条染色体、每个状态157,529对、Pearson主指标、染色体等权。",
              "- `experiment3_manifest.json`中的`same_raw_position=0`只表示端点bp精确相等，不是分辨率聚合后的diag；各层真正的`raw_same_bin/raw_cis_offdiag/raw_inter`见`results/split_resolution_budgets.json`。",
              "- 实验1的inter逐pair rate、normalized rate与sorted-four distances在GPU内存中完整计算；落盘只保留shape、SHA256与max difference。独立重放完整数组需要重新消耗相应计算预算。",
              "- 训练使用train records在5/2/1 Mb逐层重新聚合；original e包含当层全部endpoint（diag两端），offdiag-e只含当层offdiag endpoint；test primary仅在1 Mb。", "",
              "## checkpoint/export工程边界", "",
              "- `pr/solver_state.py` 与本轮056 runner实际采用 finite immutable `solver_state.npz`、独立bool `presence_mask.npz` 和由统一helper生成的export，并验证export前后state SHA不变。",
              "- `code/budget_runner.py` 是同一SciPy L-BFGS-B算法和冻结参数的预算口径适配；它不是逐字节调用旧wrapper。它移除重复预算外评分，并对两个e条件完全对称。",
              "- 历史 `051/run_chain` 与 `fix_export.py` 未修改；不能宣称旧入口已修复，也不得将旧 `fix_export.py` 用于本轮或未来新checkpoint。未来链式入口必须直接调用同一个 `pr.solver_state` helper。",
              "- exclusive-create是安全契约；完整重跑必须新建新的编号运行目录并将本轮code/protocol复制到新目录后执行，不能覆盖056现有产物。", "",
              "## 预算与隔离", "",
              "- 实验1：46 FG（上限48），物理口径为46 contact forward + 46 separate physics/repulsion sweeps。",
              "- 实验2：12 value+gradient FG、24 contact forward/back sweeps。",
              "- 实验3辅助：%d/64 actual full-pair sweeps；训练FG另列，不混入辅助上限。" % nonref["auxiliary_total_full_pair_sweeps"],
              "- 实验3训练：每FG按2次contact forward/backward和1次repulsion full-pair计，总计%d physical sweeps；launcher中的`estimated_training_pair_kernel_sweeps`仅是contact口径。" % (3 * training["outer_fg_actual"]),
              "- 所有训练和无reference评分记录 `reference_opened=false, phase_opened=false`；统一reference evaluator只在state/hash gate后运行。", "",
              "## 主要产物", "",
              "- `results/final_status.json`, `results/summary.tsv`, `results/budget_ledger.json`",
              "- `reference_eval/metrics.json`, `reference_eval/per_chromosome.tsv`",
              "- `plots/experiment1_data_regularizer.png`, `plots/experiment2_gradient_response.png`, `plots/experiment3_paired_outcomes.png`",
              "- `results/artifact_hashes.json`", "",
              "## 核验命令", "",
              "以下是入口记录；除只读测试外，正式重算应先按上一节创建新运行目录，不能直接覆盖本目录。", "",
              "```bash",
              "conda run -n analysis python -m unittest tests.test_solver_state -v",
              "conda run -n analysis python test_res/056-20260916T152353Z-pro-review-experiments/code/run_nonref.py --experiment both",
              "conda run -n analysis python test_res/056-20260916T152353Z-pro-review-experiments/code/prepare_experiment3.py --prepare",
              "conda run -n analysis python test_res/056-20260916T152353Z-pro-review-experiments/code/run_training.py --all",
              "conda run -n analysis python test_res/056-20260916T152353Z-pro-review-experiments/code/score_experiment3.py",
              "conda run -n analysis python test_res/056-20260916T152353Z-pro-review-experiments/code/evaluate_reference.py",
              "conda run -n analysis python test_res/056-20260916T152353Z-pro-review-experiments/code/make_report.py", "```", ""]
    (RUN / "README.md").write_text("\n".join(lines), encoding="utf-8")
    terminal = {"schema": "p9016-056-final-terminal-v1", "status": "complete",
                "scientific_status": status, "artifact_hash_manifest": "results/artifact_hashes.json",
                "github_pushed": False, "independent_parent_acceptance_pending": True}
    write_json(RUN / "logs/final_terminal.json", terminal)
    print(json.dumps({"status": "complete", "method_status": status["method_status"],
                      "comparisons": comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
