# Run 058 最终结果

## 结论

1. **A 不通过。** 四个配对的 Swap 经各 100 FG 后，训练 `J`、count 和 dev NLL 全部差于各自 Control；8 个臂均为 `budget_not_converged`，两个 seed 都是 `training_gate_failed / selected=null`，4 个 `near_control=false`。
2. **B 只完成 fixed-x p 诊断，不是 B-Control 3x50 FG 策略比较。** 两 seed 的训练 J gain 为 `1.7971419907780728e-09`、`3.943964976826919e-08`，均小于 `1e-5`；仅用 47/46 次 scalar calls，因此 `coordinate_triggered=false`、coordinate FG=0。`p` 从 `0.24035967246787707` 到 `0.2403934067333627`、从 `0.243843226710218` 到 `0.24368279562818268`，不是 `p<0.5` 错误。
3. **分层结果不支持候选改善。** A 四个 `cis_20Mb_inf` 对 global dev gain 的贡献依次为 `-0.00019152032390135076`、`-0.00028677825133271995`、`-0.0007228799550387244`、`-0.0007444280188662056`。seed 560102 candidate1 的近/中距 global contribution 略好，但被远距/inter 恶化抵消。组内 conditional NLL 与对 global gain 的贡献是不同量，必须同时看已存值，不能断言两者方向必然相同。整体无 candidate improvement，不换 046 baseline，不作 L2 声明。
4. **身份未验证。** raw phase 与 SNP-free 七列逐行 `1,703,888/1,703,888` 对齐，但没有独立 direct/inferred provenance。`R1_direct` 和 identity veto 均为 `NA`；`hard known same-copy=40939` 只是未验证来源标签分类，不是已证 direct SNP 真值，因此不跑 bootstrap，结论为 `identity_unverified`。

## A 全部终态

| Arm | Terminal | FG | Train J | Active maxgrad | Dev NLL |
| --- | --- | ---: | ---: | ---: | ---: |
| `A-seed560101-candidate1-Control` | budget_not_converged | 100 | 9.515268334 | 2.08e-05 | 13.489870016 |
| `A-seed560101-candidate1-Swap` | budget_not_converged | 100 | 9.520386660 | 0.000894 | 13.496579944 |
| `A-seed560101-candidate2-Control` | budget_not_converged | 100 | 9.515233341 | 0.000205 | 13.489896402 |
| `A-seed560101-candidate2-Swap` | budget_not_converged | 100 | 9.517487555 | 0.000795 | 13.492332192 |
| `A-seed560102-candidate1-Control` | budget_not_converged | 100 | 9.526582488 | 5.58e-05 | 13.505268218 |
| `A-seed560102-candidate1-Swap` | budget_not_converged | 100 | 9.527207566 | 0.000358 | 13.506312800 |
| `A-seed560102-candidate2-Control` | budget_not_converged | 100 | 9.526611242 | 5.21e-05 | 13.505404013 |
| `A-seed560102-candidate2-Swap` | budget_not_converged | 100 | 9.528992813 | 0.000248 | 13.508388495 |

训练选择：seed 560101 与 560102 均为 `training_gate_failed / selected=null`。所有 A 臂均有限但为 `budget_not_converged`。仅 seed 560101 candidate1 出现 active-gradient ratio >10 的 fixed-budget imbalance warning；它不改变已失败的候选门，也不能证明固有结构或全局最优。

## B 诊断

| Seed | p before | p after | Train J gain | Dev gain | Scalar calls |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 560101 | 0.240359672 | 0.240393407 | 1.8e-09 | 1.24e-07 | 47 |
| 560102 | 0.243843227 | 0.243682796 | 3.94e-08 | -5.34e-07 | 46 |

四个 B 状态是 fixed-x Original/Profile 描述性状态，不是 3×50 FG 策略对照。坐标完全相同，因此结构 same/cross/contrast/min-margin delta 均为 0。

## Dev 分层

固定分母：`Noff=253067`。`bin_anchor_genomic_separation` 三个 cis 组精确覆盖全部 cis offdiag：

| Group | Records | Eligible pairs |
| --- | ---: | ---: |
| `cis_1_5Mb` | 49424 | 10380 |
| `cis_5_20Mb` | 31007 | 36075 |
| `cis_20Mb_inf` | 59129 | 137561 |
| `inter` | 113507 | 3312674 |

A 四个 Swap 的 global dev gain 均为负。上述近、中、远、inter 分层同时提供两端 `within_group_conditional_nll` 及其 conditional gain，不能用 global contribution 冒充组内 conditional NLL；完整 6 comparisons x 4 groups 标量见 `dev_strata_pairs.tsv`。

## 结构与身份

055 公共支持为 20 chr、157,529 pairs；每 endpoint/chr 只用整条染色体 Pearson 几何确定一次 mapping。A 的 affected-chr min-margin 为三降一升，其中 seed 560102 candidate2 为 `+0.10186136907797211`，不能概括为全部结构恶化；完整 macro20chr 与 affected-chr 的 same/cross/contrast/margin_A/margin_B/min 及两端值和 delta 见 `paired_A.tsv`。四个 Swap 的 fixed-frame return-to-Control 均为 `near_control=false`，但这不挽救训练/dev 门失败。

phase 对齐后的 dev 统计：cis offdiag `139560`；已知同-copy `40939`，其中远距 `16542`、`10320` 个 distinct bin-pair blocks；已知 cross-copy `1055`，未知 `97566`。由于 direct 来源未证，未执行 block bootstrap，不用推断标签替代。

## 配对导出

- `paired_A.tsv`：4 个 A 配对宽表；gain 定义为 `Control - Swap`，结构 delta 定义为 `Swap - Control`，macro20chr 与 affected-chr 明确分列。
- `p_profile_B.tsv`：2 个 fixed-x B profile 配对；gain 定义为 `Original - Profile`，结构 delta 定义为 `Profile - Original`（均为 0），并标记 `diagnostic_only=true`。
- `dev_strata_pairs.tsv`：6 comparisons x 4 groups；conditional gain 与 global-contribution gain 均定义为 `Control/Original - candidate`。
- 复现仅需运行 `PYTHONDONTWRITEBYTECODE=1 conda run -n analysis python code/export_tables.py`；脚本只联表及做已存标量减法。

## 预算与时间

- Source exposure：`trainfixed`；所有 12 个 gate states / 36 artifacts 冻结后统一评价。
- Training FG：`800/1400`（A 800，B 0）。
- Extra full-grid equivalent：`14/64`（B cache 2，12 endpoint dev 12），保留 50。
- A optimizer 累计：`163.6937117157504 s`；B cache build：`0.5192612949758768 s`；统一评价：`10.605994581244886 s`。这些是现存区段时间，不是用户任务总耗时。
- 明细账本见 `budget_ledger.json`；据此停止本版 A 和本轮 B 坐标扩展，保留模型 G 与 046 baseline。

## 主要证据

- `config.json`：最终冻结方法。
- `preformal_freeze.json`：正式运行前 fixture 与 config hash。
- `results/pre_evaluation_hash_gate.json`：12 states / 36 artifacts 的评价前哈希门。
- `FINAL_RESULTS.json`：机器可读终态、比较、预算与决定。
- `evaluation/dev_results.json`、`evaluation/structural_results.json`、`evaluation/phase_traceability.json`：原始评价表。

本轮不修改 046 baseline，不 commit/push，不作 L2 或全局最优声明。
