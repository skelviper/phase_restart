# 020 后续实验计划：优先验证 allele 差异信号

## 给主实验进程的交接

这是用户在阅读 020 方法及 021 对照后要求更新的下一步计划。优先级是：**allele 特异结构差异的恢复，高于单纯提高与 reference 的整体 Spearman 相关。** 本文更新后续研究方向，不修改 020 的冻结配置、坐标、选择或已有评估。原版本只写了方向；Revision 1 将父侧冻结的 1 Mb 消融、合成校准、native 起点、预算、评价和失败处理写成可执行 protocol，见 [`POST020_ALLELE_ABLATION_PROTOCOL.md`](POST020_ALLELE_ABLATION_PROTOCOL.md) 与机器可读 [`POST020_ALLELE_ABLATION_PROTOCOL.json`](POST020_ALLELE_ABLATION_PROTOCOL.json)。本文件现已追加 Revision 2 的实际执行记录；prefit 原文另存为 [`archive/PLAN-post020-allele-signal.prefit-3d85720a.md`](archive/PLAN-post020-allele-signal.prefit-3d85720a.md)，二者不能混读为同一时间点。

**Revision 2 / status: `complete`.** Revision 1 的 protocol、freeze 和执行顺序已兑现：023 固定端点诊断、022 的 240→480 warm continuation、024 的既有 R2 派生、025 native validation、026 synthetic calibration、029 的 15 个 real joint fits 和本轮 R2-only evaluation 均已有 sealed outputs。R2 scope 仅为 21 conditions、20 linked chromosomes、420 chromosome rows、40 primary cells、120 seed rows；本轮没有新增 R1/R3。real fits 和 R2 均不读 phase payload，R2 reference 只在坐标 hash lock 完成后读取；所有 real endpoint 的 fixed budget 状态与未收敛边界须保留。prefit 文档快照仍保留在 `docs/archive/PLAN-post020-allele-signal.prefit-3d85720a.md`，其 `frozen_before_fit` 只描述历史时间点，不代表当前状态。最终中文结果见 [`POST020_ALLELE_ABLATION_RESULTS.md`](POST020_ALLELE_ABLATION_RESULTS.md)。

用户提出的方向：弯曲项可能不必要；固定核球约束可能不必要；后续主要采用 random 初始化。下一步应将这些方向落实为可解释的消融，而不是同时改变全部设定。非整倍体与稀疏 SNP 辅助方向继续留在 [后续应用实验](RECONSTRUCTION_FUTURE_APPLICATIONS.md)，不混入本轮严格 SNP-free 优化。新 protocol 采用 C0、C1、C2-map、C2-free、C3 五个单项 arm；没有无 bend + free 组合、没有 r0/先验搜索，也不新跑 020 consensus。

Revision 1 已完成阶段 A、protocol freeze 和执行规则；本 Revision 2 记录实际结果，不再要求启动新的准备或 fit。具体 seed、迭代/函数调用预算、停止规则与候选清单均以冻结 protocol、029 run manifest 和 receipts 为准：480 accepted 是固定的有限预算，不是充分收敛证明。本文不把旧 300/200/240 次称为已校准预算，不增加额外确认门。

## 已有证据与工作假设

共同位置对比较来自 `test_res/021-20260913-softall-v1-reference-spearman/`：Softall 固定 seed124101，本次 V1 固定为 020 已选中的 `random_joint`。20 条染色体、157,529 个共同非对角 bin pairs；每条染色体独立选择一次整体 A/B 对应，不允许局部重新配对以抬高分数。

| 染色体 macro mean | Softall | V1 random_joint |
|---|---:|---:|
| Matched Spearman | 0.546126 | 0.490326 |
| Cross-matched Spearman | 0.458126 | 0.357369 |
| R2 contrast | 0.087999 | 0.132956 |

V1 的平均 contrast 高约 0.044957，而整体 matched 相关低约 0.055800。021 为描述性比较，尚未给出 Softall–V1 contrast 差异的不确定性；不能仅凭均值差宣称显著优越。

020 相对固定 random 基线的 R2 contrast 增益为 +0.111530，染色体重采样 95% CI 为 [+0.068857, +0.157703]。这是当前 allele 相关差异信号的积极证据。R3 增益 CI 仍跨 0，且所有优化层均未收敛，因此不升级为整条染色体 copy identity 已恢复的 L2 结论。单细胞内染色体重采样不是生物学重复。

用户提出的工作假设：Softall 与 reference 共有 hickit 重构流程/先验，可能使整体几何相关更高；V1 的不同 prior（例如 bend、球约束）可能解释部分整体相关损失。**这是假设，不是已证明的因果归因。** 本轮应检验它，不把 hickit reference 视为无误差的物理真值，也不把“画图风格”与生成坐标的约束混为一谈。

## 评价重点：像自己的 allele，并区别于另一个

对每条染色体保留完整的四个 Spearman 相关，按整体几何配对重命名后记为：

| | ref1 | ref2 |
|---|---:|---:|
| 重构1 | a | b |
| 重构2 | c | d |

报告：

- `matched=(a+d)/2`：匹配参考的整体结构相似。
- `cross_matched=(b+c)/2`：交叉对应的相似。
- `contrast=matched-cross_matched`：主要的 allele 差异读数。
- `margin1=a-b`、`margin2=d-c`：分别检查两条链是否都更像自己对应的 reference，避免平均值掩盖一条好、一条差。
- 同时保存每 copy 的 matched 值、逐染色体分母、缺失/tie、整体配对方向；最优整体配对引入的正向选择效应必须保留在解释中。

评价优先 allele 差异，但不能只奖励降低 cross_matched 而牺牲全部 matched 相似。禁止以两重构之间的距离、`1-rho(copyA,copyB)` 或单纯空间分离作恢复分数；无意义差异也能抬高这些数值。本轮正式 real evaluation 只做 R2；不新增 R1/R3，图和报告以四相关、contrast、双 margin 为主。

**训练/选择与评价分开：** 本文的“优先 contrast”是科研评价优先级，不是允许把 reference contrast 放进训练目标、早停或参数搜索。候选坐标及无标签选择在读取 reference/保留 phase 之前固定并 hash；所有预定消融都报告，不只留下 reference 表现最好的结果。用户已看过 P9016 评价，因此本轮属于针对同一样本的后续探索，不能包装成未触碰的独立验证。模型泛化需独立合成条件或后续独立样本检验。

## Revision 2：阶段 A-F 已完成，R2-only 已封存

### 已有结果与实际进展

- **022 + 240 已完成**：这是计划 B 的固定 warm continuation，同一 020 `random_joint` 端点由 240 accepted 延至累计 1 Mb 480；`total J` 为 `9.61001577125 -> 9.60361804197`，`count_nll` 为 `9.59358593129 -> 9.58710105238`，终止状态是 `budget_not_converged`。它是预算诊断，不是独立初始化重复。FDG1000full 提案的原 J 未通过 label-free 规则，状态为 rejected。
- **023 固定端点诊断已完成**：见 [`023 README`](../test_res/023-20260913_212541-post020-allele-signal-diagnostics/README.md)。020/022 的 `maxR` 分别为 `0.65161710/0.68574970`，`>=0.90/0.95/0.99` bead 均为 0；这只能排除这些端点的活跃 hard-wall occupancy，不能说 sphere 参数化没有优化影响。sphere radial attenuation 最小值为 `0.43646488/0.38556995`，tangential attenuation 为 `0.75854806/0.72783744`。weighted bend y-gradient L2 为 `0.01216713/0.01206299`，count y-gradient L2 为 `0.03243707/0.03234493`，比约 `0.375/0.373`，所以 bend=.01 有实际作用。实际 repulsion 是 `.7*l0=.04017409936`；旧 config metadata 中的 `repulsion_radius_l0=2.0` 是错误字段，仅作错误留痕；native `d_r=2.0` 是独立的 native 参数。total gradient L2 为 `.00620367/.00514436`，末 20 accepted total J 降为 `.00091717/.0003898645`；020 六层和 022 都是预算停止，不称充分收敛。023 保存源与当前 `pr` 有差异，历史重现以保存代码的数值结果为准。023 的最终 bash75 与 validator 均 exit 0。
- **024 R2 派生已完成但不再返选训练**：见 [`024 README`](../test_res/024-20260913_133109-r2-allele-signal-derived/README.md)。原 V1 相对 Softall 的 contrast 为 `+0.044957`，CI `[-0.007420,+0.096044]` 跨 0，matched 变化为 `-0.055800`；022 相对 020 的 contrast 为 `-0.000222`，基本持平。原 V1 与 continuation 均为 20/20 finite，`both-positive=13/20`、`one-negative=7/20`，minmargin 分别为 `.049677/.047710`。父侧发现 fixed-reference 列锚定问题：swapped 时应换 candidate 行而不是 ref 列；024 v2 已修复。matched/cross/contrast/minmargin/counts 不受影响，旧 v1 的 absolute per-reference margin 或 contribution share 仍不引用，除非在 v2 下重新生成。
- **025 native validation 已完成**：实际目录为 [`025 native full-grid preflight`](../test_res/025-20260913_135100-random-native-fullgrid-preflight/)。三 bundles、count conservation 和 63 graph-swap checks 通过，native calls、x0、coordinate/graph checks 已由 validation JSON 锁定；它是 native initialization/graph validation，不是 joint fit 完成证据。
- **026 synthetic calibration 已完成**：20 个固定 `cal80` fits、truth-only evaluation、implementation gate 和 C2 audit 均已封存；20/20 为 `budget_not_converged`。结果只用于 failure-mode/calibration，不返选 real 参数，不建立 L2。
- **029 real ablation 已完成**：5 variants x 3 bundles 共 15 个 joint fits，全部 `nit=480`、`budget_not_converged`；variant 内按 `count_nll_normalized` 选择 `bundle2`，不读取 R2 返选。完整 15-terminal archive、termination/worker audit 与 final JSON/coordinate hash 在 [`029 result report`](POST020_ALLELE_ABLATION_RESULTS.md) 和 [`real archive receipt`](../test_res/029-20260913_161713-post020-allele-ablation-real/archives/post020_real_all_endpoints.receipt.json)。
- **029 R2-only 已完成**：[`r2_delivery_receipt.json`](../test_res/POST020_REAL_CONTROLLER/r2_delivery_receipt.json) 为 `complete`；21 conditions、420 chromosome rows、40 primary cells、120 seed rows。40 个 primary rows 为 `3/3` valid seeds、20/20 valid chromosomes，120 个 seed rows 各自对应一个指定 seed、20/20 valid chromosomes；这 40 个 primary + 120 个 seed paired rows 无 n/a。420 个 per-chromosome rows 仍按 metric applicability 保留 n/a；例如 consensus 的 contrast/cross/margins 不适用时为 n/a。本次 15 个新 real candidate 的适用 two-copy metrics 均 finite。实际 20-record mask metadata 汇总为 `157529` common / `176201` total non-diagonal pairs；shared bootstrap 为 seed `9301`, `10000 x 20`。本轮没有新增 R1/R3。

这些结果回答了预算、边界传递、代表选择和 R2 读数的冻结问题，但不回答 L2。最终数值、图表索引和科学结论见 [`POST020_ALLELE_ABLATION_RESULTS.md`](POST020_ALLELE_ABLATION_RESULTS.md)。协议机器文件为 [`POST020_ALLELE_ABLATION_PROTOCOL.json`](POST020_ALLELE_ABLATION_PROTOCOL.json)，完整文字约定为 [`POST020_ALLELE_ABLATION_PROTOCOL.md`](POST020_ALLELE_ABLATION_PROTOCOL.md)。

### 阶段 B：先锁定输入、起点和预算

阶段 B 的冻结要求已在 029 run 实际执行：run-specific `freeze/POST020_ALLELE_ABLATION_FREEZE.json`、machine JSON/input/source/prepared-graph/x0 hashes 和 append-only registry 均作为 sealed provenance 保留。MD/plan 的真实 mtime/hash 若在 native 后追加，只按实际时间记录，不回填成早于 fit；controller provenance 也明确标注启动后补录。未分配 run 时使用的 path pattern 仅是历史规则，不代表当前仍有待分配 run。

正式 run 的路径模式曾为：

```text
test_res/{NNN}-{YYYYMMDD_HHMMSS}-post020-allele-ablation-{real|synthetic}/
```

不猜已有 fixture/run 文件，不把 prepare-only 或 hash 缺失的产物称为 fit。

真实 cohort 固定为一个 P9016 生物细胞，输入 `inputs/P9016.snpfree.pairs.gz`，SHA256 为 `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`。原始记录为 `1,703,888 = 1,135,454 cis + 568,434 inter`；其中 same-bin `438,774` 单独进入 saturated nuisance，structural raw records 为 `1,265,114 = 696,680 cis-offdiag + 568,434 inter`。likelihood observation unit 是聚合 unordered genomic bin-pair `C_ij`；从 header 构造完整 eligible/zero 集合，zero 不能丢。1 Mb grid 使用 origin 0、`ceil(L/B)` 加 terminal partial bin、20 chromosomes、40 tracks、2,645 loci/copy、5,290 physical beads、`3,496,690` eligible pairs，其中 real zero 为 `3,009,436`。

三个 phase-free random native bundles 固定为：

| bundle | assignment seed | native seed |
|---|---:|---:|
| bundle1 | 250101 | 250201 |
| bundle2 | 250102 | 250202 |
| bundle3 | 250103 | 250203 |

assignment 沿 `genome.random_assignment`：cis 两端同一随机 copy，inter 两端独立随机 copy；不读 phase/p_gen，不用 014 换 seed 假充独立。native 使用 explicit full-grid NULL-source bridge，`n_iter=1000`，每 seed 唯一一次，CPU=1，包含 terminal/zero-contact beads；参数沿明确 default，`target_radius=10`、`source=NULL`，native bend=0，不因 C1 改 native bend。图级 canonicalization 对每条 chr 的 AA/BB cis integer edge signature 数值排序，tie 使用忽略 partner-copy 的 incidence signature，再 tie 即 preflight fail、保留 seed、不换 seed。每 bundle 做 20 个 single-chr 加 1 个 all-chr canonical-input byte-invariance audit，合计 63 checks；这只证明 graph/blob invariance，不声称 native equivariant。

native 最终一次性处理全部 40 tracks，再做一次全轨共同 center 和 uniform scale，使 max radius=.8；保存 canonical labels。三个 physical x0 必须在进入模型前锁定并 hash。同一 bundle 的五个 variants 共用同一 physical x0，只允许各自 latent inverse 不同；不得按 variant 再调用 native。

真实每 fit 固定 SciPy L-BFGS-B：`maxiter=480 accepted`、`maxfun=1470`、`maxls=20`、`ftol=1e-10`、`gtol=1e-6`、checkpoint 每 20 accepted。必须记录实际 `nfev/njev`、全部 objective evaluation 和 line-search probe、components、终态 reason、输入/x0/source hashes。预算不是充分收敛承诺，不按 R2 早停或加量。

### 阶段 C：合成 calibration（已完成，固定结果已封存）

合成数据与真实使用同一 header、1 Mb、20 chr、5,290 points 和完整 eligible 构造。四个 fixture 的 identity 和 seed 不可互换：

| fixture | class | truth | draw | exposure | independent initial-shape | generating exposure |
|---|---|---:|---:|---:|---:|---|
| N1 | same internal shape, spatially separated negative | 260101 | 260201 | 260301 | 260401 | ones |
| N2 | same internal shape, spatially separated negative | 260102 | 260202 | 260302 | 260402 | lognormal sigma=.4, mean=1, no dropout |
| P1 | different-shape positive | 260103 | 260203 | 260303 | 260403 | ones |
| P2 | different-shape positive | 260104 | 260204 | 260304 | 260404 | lognormal sigma=.4, mean=1, no dropout |

四个 truth 独立，N1/N2 或 P1/P2 的跨 fixture 差异不能当作纯 exposure effect。生成 kernel 与 V1 相同、`p_gen=.8`；N2/P2 按实际 helper 生成 `np.exp(default_rng(seed).normal(0, sigma, N))`，其中 `sigma=.4`，再除以 realized full-grid mean，不猜 population log mean。每 fixture 固定 integer multinomial totals：diag `438,774`、cis-offdiag `696,680`、inter `568,434`，总 raw `1,703,888`，保留所有 eligible pairs；每 fixture 的实际 zero 数另行记录，不能套用 real `3,009,436`。

C0/C1/C2-map/C2-free 从 synthetic observed endpoints 重算 exposure，C3 只用 ones；generating exposure 不给 worker。每 fixture 的 independent init 一次调用 `generate_truth(startseed,false)` 生成 random chains，统一 normalize 到 maxR=.8；不取 truth、不扰 truth，五 arm 共用同一 x0，p0=.75。先完成 truth/observed-data freeze 和 prepare，再 fit；所有 fit coordinates hash 锁定后才在隔离 evaluator 读 truth。

合成共 `4 x 5 = 20` fits，预算为 `maxiter=80 accepted`、`maxfun=270`，其余 tolerance、maxls=20、checkpoint20 与 real 相同。N fixture 需满足每 chr 两 truth D 一致（rtol `1e-10`）且 centers 间隔 `>1e-6`；P normalized D difference `>1e-6`。失败 fixture 标 `invalid` 并保留，不换 seed。finite/gradient/grid/budget/code/hash 是 implementation gates；truth R2 增加不是放行门，truth 不返选参数。N 的 contrast 近零是同形状构造的代数性质，不是“不虚构分裂”的证明。N/P 四 fixture 的 20 个 cal80 fits、truth-only evaluation、implementation gate 和 C2 audit 已完成；结果只用于 failure-mode/calibration，不返选 real 参数、不建立 L2。详细 artifact 见 [`026 synthetic evaluation`](../test_res/026-20260913_221709-allele-calibration-prepare/results/evaluation.json) 和 [`最终中文报告`](POST020_ALLELE_ABLATION_RESULTS.md)。N 报每 copy 对 common truth 的 shape error：每 chr 使用同一 finite unordered offdiag mask；候选两 copy 共用 `sC=sqrt(mean(两份候选 copy 在该 mask 上的 squared distances))`，truth 两 copy 共用 `sT=sqrt(mean(两份 truth copy 在该 mask 上的 squared distances))`（N 用 common-truth D 复制给两 truth copy），每 copy 计算 `RMS(Dc_copy/sC-Dt_copy/sT)` 的 chr mean/max，保留相对 copy 尺度，不做 per-copy rescale；另报 negative-only `RMS((DcA-DcB)/sC)`，不作 positive score 或 selection。P 同样使用两 copy 共用的 sC/sT，并报 own-truth 四 rho、contrast、margins 和 shape error。

### 阶段 D：五项 real ablation（已完成，终态已封存）

所有 arm 都是同一输入、同一 1 Mb grid、同一 bundle physical x0 和同一预算；共 `5 x 3 = 15` joint fits。除明确列出的改动外，count/p_prior/bond/repulsion 权重均为 1，bend=.01（C1 除外），kernel `epsilon=1e-6`、`r0=2*l0`、repulsion=.7*l0、`p_floor=1e-4`、p-prior strength=`1e-4`、p_init=.75、`q_from_p(.75)`，`l0=(2*N_loci)^(-1/3)`；无层间或多分辨率归一化。

| arm | 唯一主改动 | exposure / parameterization |
|---|---|---|
| C0 | 新 1 Mb V1 original；不复现 020 path | observed-endpoint / current sphere |
| C1 | bend `.01 -> 0`，保留 bond | observed-endpoint / same sphere |
| C2-map | 只换 fixed identity-core `.90` smooth sphere map | observed-endpoint / `R<1` |
| C2-free | direct-x finite-only；无硬球、无 clip、无 rescale | observed-endpoint / direct Cartesian |
| C3 | 只把 full-grid exposure 设为 ones；保留 endpoint audit 和所有 normalizers | ones / same sphere |

C2-map 使用：`r=||y||`；`r<=.90` 时 `s=r`；否则 `t=(r-.90)/.10`、`s=.90+.10*t/sqrt(1+t^2)`、`x=s*y/r`（r=0 时 x=0）。C2-free 的 direct-x 同时改变参数化和可行域，因此 C0→free 不能归因纯 boundary；C2-map 只帮助区分 map sensitivity，仍不做夸大因果。没有无 bend+free 组合、没有 r0/先验搜索，020 consensus 仅保留历史对照不新跑。

每个 variant 内只按 final `count_nll_normalized` 选 3 starts 的 representative；tie `<=1e-12` 按 bundle ID。所有 15 outputs 都评价，不能跨 exposure/prior 按 J 宣称最佳。统计或数值失败、拒绝、未收敛都保留；不得因不利表现重抽 seed 或重跑。只有实现 bug 可在同一冻结 data/seed/budget 上修复重试，完整记录原/新 source hash，retry 不算 independent repeat。实际 15 个 outputs 均完成并为 `budget_not_converged`，每个 `solver.nit=480`，`nfev=482--491`；C0/C1/C2-map/C2-free/C3 均在 variant 内选 bundle2，且 selection 未使用 R2。终点/selection/hash 导航见 [`release_manifest_r2.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/release_manifest_r2.json)、[`final_release_receipt.json`](../test_res/POST020_REAL_CONTROLLER/final_release_receipt.json) 和 [`15-arm archive`](../test_res/029-20260913_161713-post020-allele-ablation-real/archives/post020_real_all_endpoints.receipt.json)。

### 阶段 E：real R2-only 隔离评价（已完成，receipt complete）

所有 15 endpoints、variant 内 selection、coordinates 和 source hashes 已锁定后，evaluator 才读取 evaluation-only reference `data/P9016.1m.3dg.gz`；没有读取真实 phase pairs。与 Softall seed124101、020 random、022 continuation、FDGfull rejected、random014、consensus014 使用同一 fixed controls 和共同 mask。实际输出为 21 conditions（15 real bundles + 6 historical controls）、20 chromosomes、420 chromosome rows、40 primary cells、120 seed rows；R2 receipt 的 `phase_payload_opened=false`、`real_fit_started_by_evaluator=false`、`selection_recomputed=false`、`reference_loaded_after_all_coordinate_hashes=true`。实际 20 条 `mask_metadata` 汇总为 `157529` common / `176201` total non-diagonal pairs；40 个 primary rows 为 `3/3` valid seeds、20/20 valid chromosomes，120 个 seed rows 各自对应一个指定 seed、20/20 valid chromosomes；这 40 个 primary + 120 个 seed paired rows 无 n/a。420 个 per-chromosome rows 仍按 metric applicability 保留 n/a；例如 consensus 的 contrast/cross/margins 不适用时为 n/a。本次 15 个新 real candidate 的适用 two-copy metrics 均 finite。结果导航见 [`POST020_ALLELE_ABLATION_RESULTS.md`](POST020_ALLELE_ABLATION_RESULTS.md)、[`r2_summary.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_summary.json)、[`r2_paired_effects.tsv`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_paired_effects.tsv) 和 [`r2_delivery_receipt.json`](../test_res/POST020_REAL_CONTROLLER/r2_delivery_receipt.json)。

位置范围为 `range(3Mb,L,1Mb)`；逐 chr 使用 finite、unordered、offdiag 的共同 mask。constant 或 nonfinite 输入，或某 chr common pairs `<20` 时该 metric 记 n/a；失败 arm 保留为 n/a，不改变分母、局部换 copy 或从计划消失。本次没有触发 primary/seed n/a，但规则仍是 release contract。

每条 chr 保存原始 `rho_A_mat/rho_A_pat/rho_B_mat/rho_B_pat` 和一次整体 swap 后的 `matched/cross_matched/contrast`、fixed-ref 两个 matched/margins、minmargin、both-positive/one-negative/both-negative/margin-tie/geometry-tie/missing counts。固定 `ref1=mat`、`ref2=pat`，不换 reference 列：

```text
direct: a=A_mat, b=A_pat, c=B_mat, d=B_pat
cross/swapped: a=B_mat, b=B_pat, c=A_mat, d=A_pat
matched=(a+d)/2
cross_matched=(b+c)/2
contrast=matched-cross_matched
margin_ref1=a-b; margin_ref2=d-c; minmargin=min(margin_ref1,margin_ref2)
```

每条 chr 只做一次 best overall swap；best-swap 的正向选择偏置保留并明示，不作自身差异评分。`abs(direct-cross)<=1e-12` 时为 `unresolved_tie`，保留原四 rho、contrast=0、不计 both-positive，fixed-ref margins/minmargin 记 n/a；不做局部换 copy。both-positive/one-negative 只是受 best swap 影响的描述模式；非 tie 时 both-negative=0 是 `contrast>=0` 的代数约束，不是成功证据。

主要 paired comparisons 固定为 `C1-C0`、`C2-map-C0`、`C2-free-C0`、`C2-free-C2-map`、`C3-C0`，再与历史 controls 作描述比较。先在每个 seed 内配对同一 20 chr，再按每条 chr 对全部 3 个 seed 求效应；三 seed 是 optimization repeats，不是 biological replicates，也不把 60 条 chr 当 n=60。本次五个 primary comparisons 均为 planned/valid `3/3`，没有静默降为 available-seed 主结果。

所有 metrics/comparisons 共用同一 seed=9301 的 `10000 x 20` bootstrap index matrix。interval 只表示一个细胞内的 structural/technical variation，不是 biological CI 或 p-value；不能只报最好 seed。本轮正式 real evaluation 只做 R2，不新增 R1/R3；L2 仍未证实。

### 阶段 F：资源与 release 顺序

资源约束已按计划执行：formal real fits 最多 6 个单线程 process，BLAS/OMP=1；029 的实际 15-arm worker coverage、RSS/exit/termination 状态由 worker audit 和 archive receipt 封存。此前观察到的 `192 logical CPU`、MemAvailable 约 `335 GB` 只是运行前快照，不是线性加速保证；real 单 fit 480 的 023 历史 wall 估计也不替代本次实测终态。

release 顺序已完成并可由 receipts readback：machine JSON/input/source/prepared-graph hash → run-specific freeze/registry → fixture/graph/native prepare → x0 hash lock → synthetic fits/calibration → real 15 fits → endpoints/selection/hash lock → isolated real R2 evaluation。protocol MD/plan 的真实 mtime/hash 按实际时间保留；任何 `budget_not_converged`、失败或 n/a 都随原始 artifact 保留，没有跨步骤回填。主报告列出 [`r2_delivery_receipt.json`](../test_res/POST020_REAL_CONTROLLER/r2_delivery_receipt.json)、[`termination_audit.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/termination_audit.json)、[`ARCHIVE_MANIFEST.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/archives/ARCHIVE_MANIFEST.json) 和 R2 输出导航。

### 本 revision 的结论边界

本 revision 保留原计划的动机：检验 bend、sphere 参数化/可行域和 exposure 是否解释 020 相对 Softall 的整体差异，同时优先 allele contrast 而不是只提高整体 Spearman。实际最小实验已完成：C1 检查 bend，C2-map 检查平滑 sphere map，C2-free 检查合并的参数化/无硬球敏感性，C3 检查 full-grid exposure；每项只改一个登记因素，并以合成 negative/positive fixture 识别失败模式。

当前已支持：023 的无 reference 数值语义已复核，024 的既有 R2 派生读数和不利 FDG proposal 已保存，025 的 graph/native prepare-only 检查已过，026 synthetic calibration 已完成，029 real ablation 与 R2-only evaluation 已由 complete receipts 封存。029 R2 的实际结论和 40-cell paired table 见 [`POST020_ALLELE_ABLATION_RESULTS.md`](POST020_ALLELE_ABLATION_RESULTS.md)。

本计划和 freeze 仍不能声称 native equivariance、充分收敛、allele recovery 或 L2 whole-chromosome copy identity。15 个 real endpoints 都是 fixed-budget `budget_not_converged`；C3 的 overall similarity 增加没有转化为清晰的 allele-specific contrast；C2-map 的 `>0.9` 非恒等分支未激活，C2-free 实际最大访问 `r=.8<1` 且没有该 branch，不能回答球外访问是否有用。L1 的历史 chr1 predictive signal、L2 尚未建立、L3 对整个 nonlinear solver class 的撤回必须始终分开。本轮正式 evaluation 只做 R2，不新增 R1/R3。
