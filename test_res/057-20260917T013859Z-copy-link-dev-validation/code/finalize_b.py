"""汇总057实验B全no-op分支与A/B最终README。"""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent; RUN=HERE.parent; ROOT=RUN.parents[1]
S045=ROOT/"test_res/045-20260915T073310Z-shared-capture-round/source"
for p in (HERE,ROOT,S045):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from b_core import json_dump
from pr.solver_state import sha256_file


def main():
    a=json.loads((RUN/"results.json").read_text()); at=json.loads((RUN/"terminal.json").read_text())
    fixture=json.loads((RUN/"fixture_results.json").read_text()); r=json.loads((RUN/"round1_results.json").read_text())
    led=json.loads((RUN/"ledger_B_round1.json").read_text())
    if at["experiment_A"]!="PASS" or fixture["status"]!="PASS" or r["paired_training_triggered"]:
        raise RuntimeError("finalizer is only valid for A-pass/B-all-no-op branch")
    if any(x["accepted"] for x in r["results"]): raise RuntimeError("accepted scan cannot use no-op finalizer")
    decisions=[]
    for x in r["results"]:
        actual=x["best_full_verification_actual"]
        decisions.append({"seed":x["seed"],"candidate_count":x["candidate_count"],"best":x["best"],
                          "best_exact_full_G_actual":actual,"accepted":False,"decision_reason":x["decision_reason"],
                          "decision_basis":x["decision_basis"],"modified_chromosomes":[],
                          "modified_chromosome_gate":"not_met_empty_set",
                          "source_component_max_abs_error":max(x["source_component_errors"].values()),
                          "whole_chromosome_max_abs_error":max(x["whole_chr_errors"].values()),
                          "best_full_verification_max_abs_error":max(x["best_full_verification_abs_errors"].values()),
                          "scan_tsv":x["scan_tsv"],"scan_tsv_sha256":x["scan_tsv_sha256"]})
    result={"schema":"p9016-057-B-final-v1","status":"complete","experiment_B":"completed_first_round_all_no_op",
            "candidate_generation_count":fixture["candidate_generation_count"],"fixture":"PASS","decisions":decisions,
            "paired_training_triggered":False,"paired_training_performed":False,"training_fg":0,
            "new_development_evaluation_performed":False,"new_reference_evaluation_performed":False,
            "adoption_recommended":False,"classification":"no acceptable first-round discrete swap under frozen training gates",
            "comparison_to_pure_continuation":"NA: paired Control/Swap was not run because both first-round selections were no-op",
            "paired_dev_gain":"NA","paired_structure_differences":"NA",
            "retain_or_stop":"stop B; retain frozen G-original endpoints and existing 046 baseline",
            "l2_supported":False,"l3_claimed":False,"reference_opened":False,"development_opened":False}
    json_dump(RUN/"results_B.json",result)
    ledger={"schema":"p9016-057-B-ledger-v1","status":"complete","fixture_cpu_not_1Mb_fullgrid":True,
            "candidate_rows_scored":sum(x["candidate_count"] for x in r["results"]),
            "partial_cis_kernel_pairs":sum(x["partial_kernel_pairs"] for x in r["results"]),
            "logical_1Mb_fullgrid_calls":led["logical_fullgrid_calls"],"physical_1Mb_fullgrid_sweeps":led["physical_fullgrid_sweeps"],
            "physical_1Mb_fullgrid_cap":32,"training_fg":0,"paired_training_fg_cap_unused":1200,
            "host_wall_seconds_sum":sum(float(x.get("host_wall_seconds",0)) for x in led["entries"]),
            "gpu_event_seconds_sum":sum(float(x.get("gpu_event_seconds",0)) for x in led["entries"]),
            "timing_note":"CUDA event and host wall are measured separately; CPU candidate scan wall is stored per seed in round1_results.json",
            "entries":led["entries"],"jobs":[{"job_id":"bash-124","role":"fixture_and_candidates","exit_code":0,"status":"completed"},
                       {"job_id":"bash-125","role":"round1_scan","exit_code":0,"status":"completed"}]}
    json_dump(RUN/"ledger_B.json",ledger)
    terminal={"schema":"p9016-057-B-terminal-v1","status":"complete","completed_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
              "experiment_A":"PASS","experiment_B":"completed_first_round_all_no_op","fixture":"PASS",
              "round1":"both_seeds_no_op","paired_training":"not_started_by_frozen_branch_rule","training_fg":0,
              "formal_logical_fullgrid_calls":ledger["logical_1Mb_fullgrid_calls"],
              "formal_physical_fullgrid_sweeps":ledger["physical_1Mb_fullgrid_sweeps"],
              "new_development_evaluation":"not_run","new_reference_evaluation":"not_run",
              "adoption_recommended":False,"reference_opened":False,"development_opened":False,
              "l2_supported":False,"l3_claimed":False}
    json_dump(RUN/"terminal_B.json",terminal)
    lines=["# 057 copy 连接实验 A/B 最终记录","",
      "## 直接结论","",
      "- **A 是否通过：通过。** 两个 G-original seed 的 development-validation splice 均为 `8/8` 正 delta；median 分别为 `0.00904292036645` 与 `0.00929004905929` nat/offdiag contact。",
      "- **B 是否运行：已运行固定候选生成、fixture、初始完整核对和首轮训练侧扫描。** 两个 seed 的全局最优候选均不满足门，因此均 no-op。",
      "- **是否优于纯继续：未证明。** 冻结分支规则在两个首轮 no-op 后停止，未触发 control/swap paired continuation。",
      "- **保留还是停止：停止 B，不采纳离散 swap；保留 056 G-original endpoints 与既有 046 baseline。**",
      "","## A：development validation 门","",
      "20% fold 已经看过，因此称 development validation，不称全新 test；它是 record-level split，不保证 molecule isolation。固定 train exposure/x/p，只替换 counts；Noff=253067，Z_full 覆盖 3,496,690 pairs；无 K0、无正则。A 恰好使用18次 contact full-grid forward，训练FG=0。",
      "","## B：首轮扫描","",
      f"候选按真实 header/grid 与 chromosome length 生成，共 `{fixture['candidate_generation_count']}` 个，低于2000，未采样。所有选择只使用训练目标；未读取 development 或 reference。",
    ]
    for d in decisions:
        b=d["best"]
        a=d["best_exact_full_G_actual"]
        lines.append(f"- seed {d['seed']}: cached argmin `{b['candidate_id']}`，cached delta_full_J=`{b['delta_full_J']:.12g}`；exact full-G delta_full_J=`{a['delta_full_J']:.12g}`，delta_count/Nraw=`{a['delta_count_Nraw']:.12g}`，决定=`no-op`（`exact_full_G_checked`）。")
    lines += ["","选择顺序是先在全部候选中取 delta_full_J 最小者，再检查 `delta_full_J<=-1e-8` 与 `delta_count/Nraw<=1e-10`；没有按 count 预筛，也没有改选次优。两个 best 的 delta 均为正。",
      "","## 预算与终态","",
      f"- B 使用 `{ledger['logical_1Mb_fullgrid_calls']}` 个 logical full-grid calls / `{ledger['physical_1Mb_fullgrid_sweeps']}` 个 physical sweeps（cap 32），另计 `{ledger['partial_cis_kernel_pairs']}` 个 cis partial-kernel pairs；training FG=0。",
      "- 因双方首轮 no-op，未启动 paired L-BFGS、未生成四个新终态，也未做无意义的新 development/reference 评价；paired Control/Swap gain 与结构差值均为 `NA`，不是 0。",
      "- 两个 seed 的 modified_chr 集合均为空，预冻结 modified-set margin 门为 `not_met`。",
      "- A 的 `config.json`、`results.json`、`terminal.json` 保持不变。B 见 `config_B.json`、`fixture_results.json`、`round1_results.json`、`results_B.json`、`ledger_B.json`、`terminal_B.json` 与两个完整 scan TSV。",
      "- 046/056、冻结源码、协议、mask、checkpoint 均未修改；未调用 051 fix_export；未 commit/push。",
      "","## 科学解释","",
      "A 支持固定的八个 copy 连接破坏在 development validation 上被排斥。B 的更广训练扫描没有找到可接受的首轮离散 swap，因此不支持进入 paired continuation。L2 仍未证明，不作 L3。两个 seed 是初始化重复，不是生物学重复。"]
    (RUN/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    hashes={p.name:sha256_file(p) for p in (RUN/"results_B.json",RUN/"ledger_B.json",RUN/"terminal_B.json",RUN/"README.md")}
    json_dump(RUN/"final_hashes_B.json",hashes)
    print(json.dumps({"status":"complete","decision":result["retain_or_stop"],"hashes":hashes},sort_keys=True))

if __name__=="__main__": main()
