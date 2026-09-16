# 056 专业审查实验终态

## 结论

- 实验1和实验2硬门均为 PASS。实验1的8/8固定suffix splice data delta为正，median为 `0.0150419041367 nat/offdiag contact`。
- 实验2 production exposure 下shift的physical-x gradient L2相对变化为 `0.237461`；这是高优先级敏感性，不是停止实验3的门。
- 实验3的两个初始化只表示优化重复，不是生物学重复。四条fit真实终态见 `logs/training_terminal.json`。

## 实验3配对结果

| seed | held-out gain | matched delta | min-margin delta | grad ratio | optimization imbalance alarm |
|---:|---:|---:|---:|---:|:---:|
| 560101 | -0.00713769 | -0.0289855 | -0.0463981 | 1.471 | no |
| 560102 | 0.00227718 | -0.0207211 | -0.0270166 | 1.124 | no |

### 四个endpoint绝对值

下表四个endpoint均由同一个固定80% record-train fold拟合；test NLL来自共同20% record-test，结构列来自共同055 support。它们不是046 full-data baseline，不能与full-data baseline作训练效果横比。

| condition | seed | test NLL | same | cross | mean margin A | mean margin B | mean min-margin | 1Mb canonical grad | terminal |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| G-original | 560101 | 13.489889 | 0.598012 | 0.372697 | 0.220102 | 0.230528 | 0.165548 | 0.000432 | budget_not_converged/fg_budget_exhausted |
| G-offdiag-e | 560101 | 13.497027 | 0.569027 | 0.380981 | 0.183706 | 0.192387 | 0.119150 | 0.0002938 | budget_not_converged/fg_budget_exhausted |
| G-original | 560102 | 13.505434 | 0.594509 | 0.391797 | 0.174412 | 0.231012 | 0.118167 | 0.0003926 | budget_not_converged/fg_budget_exhausted |
| G-offdiag-e | 560102 | 13.503157 | 0.573787 | 0.395596 | 0.132429 | 0.223954 | 0.091150 | 0.0003492 | budget_not_converged/fg_budget_exhausted |

本轮预注册门下的采纳决策：`offdiag_e_not_adopted_prespecified_gates_failed`；`adoption_recommended=false`。
失败门：`heldout_gain_both_seeds`, `matched_delta_both_seeds`, `no_consistent_min_margin_degradation`, `no_consistent_splice_degradation`。

## 范围与限制

- `new_l1_claim=false`：本轮实验1是已拟合baseline上的in-sample固定x/e/p扰动，held-out只比较两个diploid exposure条件且没有1-copy/consensus预测对照，因此不构成新的L1证明。既有chr1支持与chrX边缘证据仅作为历史状态保留。
- 本轮能直接陈述：G数据项排斥预定splices；diag counts会通过e改变physical-x gradient；offdiag-e未满足预注册采纳门。
- held-out方向在两seed不一致且均未达到+0.01，因此预测证据为不确定/未达门；结构与min-margin一致退化使本轮不采纳。该决策不是对offdiag-e方法的全局证伪。
- 本轮不证明整条染色体拷贝身份已恢复（L2），也不提出‘任何无SNP方法都无效’（L3）结论。
- 四条1Mb fit均为`budget_not_converged`。梯度比和收敛状态未触发预注册优化失衡警报，但这不证明两条件优化等价或已经收敛。
- 实验1 perturbation 的same/cross/contrast沿055 frozen baseline mapping（whole-chromosome swap同步transport mapping）计算，因此fixed-map contrast可以为负；`direct/swapped`及其max/min只作为best-global secondary，不能把全部contrast误称为绝对差。
- reference tie branch审计：现有20个best-orientation ties全部为`exp1/u0`且沿frozen fixed mapping评价；4 endpoints和32 splices受影响行数为0，故未重跑reference距离、现有数值不变。未来unresolved tie保留signed per-copy margins。
- 输入全部readID为`.`，因此固定record hash split不能隔离未知分子身份；train/test是record级而非molecule级。
- 四个endpoint及其splices共同使用055 frozen legacy support：20条染色体、每个状态157,529对、Pearson主指标、染色体等权。
- `experiment3_manifest.json`中的`same_raw_position=0`只表示端点bp精确相等，不是分辨率聚合后的diag；各层真正的`raw_same_bin/raw_cis_offdiag/raw_inter`见`results/split_resolution_budgets.json`。
- 实验1的inter逐pair rate、normalized rate与sorted-four distances在GPU内存中完整计算；落盘只保留shape、SHA256与max difference。独立重放完整数组需要重新消耗相应计算预算。
- 训练使用train records在5/2/1 Mb逐层重新聚合；original e包含当层全部endpoint（diag两端），offdiag-e只含当层offdiag endpoint；test primary仅在1 Mb。

## checkpoint/export工程边界

- `pr/solver_state.py` 与本轮056 runner实际采用 finite immutable `solver_state.npz`、独立bool `presence_mask.npz` 和由统一helper生成的export，并验证export前后state SHA不变。
- `code/budget_runner.py` 是同一SciPy L-BFGS-B算法和冻结参数的预算口径适配；它不是逐字节调用旧wrapper。它移除重复预算外评分，并对两个e条件完全对称。
- 历史 `051/run_chain` 与 `fix_export.py` 未修改；不能宣称旧入口已修复，也不得将旧 `fix_export.py` 用于本轮或未来新checkpoint。未来链式入口必须直接调用同一个 `pr.solver_state` helper。
- exclusive-create是安全契约；完整重跑必须新建新的编号运行目录并将本轮code/protocol复制到新目录后执行，不能覆盖056现有产物。

## 预算与隔离

- 实验1：46 FG（上限48），物理口径为46 contact forward + 46 separate physics/repulsion sweeps。
- 实验2：12 value+gradient FG、24 contact forward/back sweeps。
- 实验3辅助：40/64 actual full-pair sweeps；训练FG另列，不混入辅助上限。
- 实验3训练：每FG按2次contact forward/backward和1次repulsion full-pair计，总计18024 physical sweeps；launcher中的`estimated_training_pair_kernel_sweeps`仅是contact口径。
- 所有训练和无reference评分记录 `reference_opened=false, phase_opened=false`；统一reference evaluator只在state/hash gate后运行。

## 主要产物

- `results/final_status.json`, `results/summary.tsv`, `results/budget_ledger.json`
- `reference_eval/metrics.json`, `reference_eval/per_chromosome.tsv`
- `plots/experiment1_data_regularizer.png`, `plots/experiment2_gradient_response.png`, `plots/experiment3_paired_outcomes.png`
- `results/artifact_hashes.json`

## 核验命令

以下是入口记录；除只读测试外，正式重算应先按上一节创建新运行目录，不能直接覆盖本目录。

```bash
conda run -n analysis python -m unittest tests.test_solver_state -v
conda run -n analysis python test_res/056-20260916T152353Z-pro-review-experiments/code/run_nonref.py --experiment both
conda run -n analysis python test_res/056-20260916T152353Z-pro-review-experiments/code/prepare_experiment3.py --prepare
conda run -n analysis python test_res/056-20260916T152353Z-pro-review-experiments/code/run_training.py --all
conda run -n analysis python test_res/056-20260916T152353Z-pro-review-experiments/code/score_experiment3.py
conda run -n analysis python test_res/056-20260916T152353Z-pro-review-experiments/code/evaluate_reference.py
conda run -n analysis python test_res/056-20260916T152353Z-pro-review-experiments/code/make_report.py
```
