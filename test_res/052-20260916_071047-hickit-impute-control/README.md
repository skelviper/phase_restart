# 052 Hickit 插补对照

## 当前状态

`budget_complete_convergence_not_established_evaluated`。父代理已验收单一实验臂；全量 `-u`、单次 `-S/-b1m`、候选哈希、实际split核验及事后评估均完成。

## 问题与最小因果对照

本实验检验：一次原生 Hickit phase imputation 加概率阈值拆分，随后使用014固定的单尺度 FDG 预算，是否能得到与固定 reference 更相似的几何。相对于014直接 `-b1m` 的唯一处理改变是 imputation-plus-split 输入路径。这是授权的有标签 SNP 辅助诊断，不是盲法训练，也不支持 L2。

主比较是新 imputed/split fit 对冻结014完整标签子集 fit。不会重跑 random/consensus，因为本实验不声称盲拆分或两结构优于单结构。未计划第二个多尺度臂：本地只找到 chr1/chr2 单 `-b1m` oracle pilot，且它硬编码不可访问的绝对二进制路径，不是可信的 reference 原始命令。

## 冻结对象、输入与预算

- 细胞与观测单位：一个 P9016 细胞；20条染色体是相关的细胞内技术/结构测量，不是生物重复。
- 原始输入：`data/P9016.pairs.gz`，SHA256 `071a6cc76bfad543ea1ace6ee1ce3022b30ac1b1e50a9f0c3a3a1967b9649505`。
- 原始物理接触全集：1,703,888条 = 1,135,454 cis + 568,434 inter；保留20条染色体和拆分后的40条copy轨迹，在同一全基因组拟合中共同处理。
- 拟合网格：1 Mb、原点0。评估网格：与014一致，从3 Mb起的1 Mb网格。
- FDG：seed 1、`-n 1000`、单次 `-b1m`。固定预算完成不等于收敛。
- 014对照：完整标签子集496,021条 = 349,025物理cis + 146,996物理inter；冻结R2宏观 matched/swapped/contrast 为0.4207229 / 0.1349990 / 0.2857239。

## 原生命令语义与顺序

`-u` 用原始 `phase0`/`phase1` 作为种子，为每条既有 pair 写入 `phase_prob00..11`，不会创建新 contact。其冻结原生默认是10 Mb radius、50 kb min radius、50 neighbors、pseudo-count 0.4、cis spatial adjustment 和1,000次内部迭代。原始输入有 `phase0`/`phase1`，但没有 `phase_prob` 列。

`-S` 需要四个概率列，对既有 pair 选最大概率状态，并仅在最大值 `>= -p 0.75` 时保留，所以会过滤不确定 pair。阈值显式冻结为0.75。为确保输入范围确为全量原始接触，两次 `-i` 前均显式使用 `--min-leg-dist=0 --dup-dist=0`，避免原生默认近端腿过滤和去重。`-n 1000` 在立即执行的 `-u` 前；`-S` 在仍为ploidy 2时执行，`-P1` 特意放在 `-S` 后但在 FDG 前；`-s1` 和 FDG 的 `-n1000` 都在立即执行的 `-b1m` 前。

精确命令见 `logs/commands.planned.txt`。`-o` 的 native 输出为普通文本，故输出名不伪装为 `.gz`。

## Gate 与评估

在 FDG 前，`work/pre_fit_gate.py` 只读取已序列化的 `imputed.pairs`：要求记录数1,703,888，报告物理cis/inter和完整标签数，使用未重新归一化的原生三位小数概率预测 `-S` 阈值保留/丢弃及40轨迹。四概率和允许 `abs(sum-1)<=0.0020001` 的序列化舍入误差，同时要求都在[0,1]且有限。由于同一原生命令中的 `-S -o split -b` 会直接开始拟合，fit前结果是对实际split输入的预测；fit结束后已从实际 `split-p075.pairs` 完成核验。

候选坐标写出并记录SHA256后，才允许读取reference。评估使用新候选、冻结014 oracle 和 reference 在每条染色体六轨迹共同有限位点上的距离 Spearman，按整条染色体选择最佳copy互换，报告 direct、swapped、matched 和 matched-minus-swapped contrast、chr胜出数，以及新减014的配对染色体 bootstrap 95% CI（seed 37、10,000 draws）。该CI仅描述同一细胞内技术/结构变异，不是生物重复。

全局诊断不局部扭曲：报告经全基因组 copy Rg 中位数归一化的copy Rg、同源质心分离/平均copy Rg、跨染色体copy质心分离/平均copy Rg，以及基于冻结的逐chr copy matching、各组先居中并单位RMS化后一次全基因组 orthogonal alignment 的RMS residual（允许reflection，因为距离约束无手性；0表示完全一致）。该残差只是辅助诊断，不用于选择拟合、seed、阈值或终点。若作图，只写一张英文标注、3英寸、300dpi、7pt的三面板图；绝不按染色体或局部对齐。

## 完成结果

- Native `-u`：`bash-111` exit 0，1000/1000次内部迭代，CPU 141.115 s。实际序列化 `imputed.pairs` 为完整1,703,888条；四概率均有限且在[0,1]，允许三位小数序列化 `abs(sum-1)<=0.0020001` 后无异常。
- `-S p=.75` 预测并由实际 `split-p075.pairs` 核验：保留1,295,440条 = 998,091物理cis + 297,349物理inter，丢弃408,448条 = 137,363物理cis + 271,085物理inter；20物理chr、40拆分轨迹。全部496,021条完整标签记录仍被保留（349,025 cis + 146,996 inter）。
- FDG：`bash-113` exit 0，精确完成1000/1000步（last logged RMS force 0.0019，CPU 353.582 s）。native没有力阈值早停判据，故终态是 `budget_complete/convergence_not_established`；坐标为native返回的 `best_x`，不能把exit 0或最后日志步写成科学收敛。
- 候选：`coords/hickit-impute-p075.3dg`，SHA256 `8dcf8a0a37a1c70e5bf0e6fc78742d2accb05b8709463a08d36142717cce7ea8`，40轨迹、5,178珠、无非有限值。
- 共同评估：20/20 chr均适用；六轨迹共同支持共2,447个共同1Mb基因组位点（两拷贝合计4,894个物理珠）、157,529个非对角距离对（每chr每对仅计一次）。014数值在这套共同mask上重算，所以可与016冻结表有细微差异；本次正好仍为014 matched/swapped/contrast `0.4207229/0.1349990/0.2857239`。
- R2宏观值：imputed matched/swapped/contrast=`0.6842395/0.2110719/0.4731676`。相对014，matched增加0.2635166（20/20 chr；technical chromosome-bootstrap 95% CI `[0.2416538, 0.2870324]`），contrast增加0.1874437（20/20；`[0.1688550, 0.2072196]`）。swapped增加0.0760730（19更高、1更低）仅作描述，不解释为性能胜出。
- 全局辅助诊断：单位RMS后的单次全基因组正交对齐残差（允许reflection、固定逐chr最佳互换）为imputed `0.2709669`，014 `0.4426444`；同源质心分离/平均copy Rg中位数为imputed `3.72385`、014 `2.56131`、reference `4.15147`，跨chr copy质心分离/平均copy Rg为`3.78853/2.97700/4.22196`。这些量不用于拟合或选择。
- 单图：`plots/whole_genome_reference_oracle_imputed.png`，英文标注，2700x900 px（9x3 in、300 dpi；每面板3 in），全局对齐不做局部扭曲。

本实验支持一个受限结论：在这个单次、SNP辅助的 native Hickit imputation-plus-split 流程和固定 `-b1m` 预算下，候选对固定reference的绝对结构一致性高于014完整标签子集的直接单尺度 fit。它不证明盲法单倍型恢复、L2、reference流程复现或生物学重复效应。

## 解释限制

终态除非有独立原生证据，否则记为 `budget_complete/convergence_not_established`；Hickit输出的是最低每步RMS force的 `best_x`，未必是最后日志步。与reference相似必须看绝对 matched 和几何诊断，不能只看是否超过014。本实验不能证明盲法单倍型恢复、整条染色体身份恢复、生物学重复效应或 reference 流程复现。染色体间接触没有母/父源标签信号，但仍是全基因组共享拟合的几何约束。
