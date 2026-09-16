# 020 后续实验计划：优先验证 allele 差异信号

## 给主实验进程的交接

这是用户在阅读 020 方法及 021 对照后要求更新的下一步计划。优先级是：**allele 特异结构差异的恢复，高于单纯提高与 reference 的整体 Spearman 相关。** 本文更新后续研究方向，不修改 020 的冻结配置、坐标、选择或已有评估。原版本只写了方向；本 revision 把父侧已经冻结的 1 Mb 消融、合成校准、native 起点、预算、评价和失败处理写成可执行 protocol，见 [`POST020_ALLELE_ABLATION_PROTOCOL.md`](POST020_ALLELE_ABLATION_PROTOCOL.md) 与机器可读 [`POST020_ALLELE_ABLATION_PROTOCOL.json`](POST020_ALLELE_ABLATION_PROTOCOL.json)。

**Revision 1 / status: `frozen_before_fit`.** 本次文档工作没有启动新 fit，也没有读取真实 reference、phase 或 coordinates。023 的固定端点诊断、022 的 240→480 warm continuation 和 024 的既有 R2 派生已完成；025 只到 prepare-only（native bridge 未调用 hk_fdg）。合成校准、15 个正式 real joint fits 和本轮 R2-only evaluation 仍为 `not_started`，不得写成完成。native prepare 的启动前置是 machine JSON、输入/source hashes 和该阶段 prepared graph（到 x0 阶段则为 locked x0）；MD/plan 的真实 mtime/hash 可在文档可用时追加 registry，若后补必须如实记录且不能阻塞 native。任何后续 fit/evaluation 均不得把后补文档伪称早于 fit；run ID 未落定时只使用 protocol 的 path pattern，不猜目录或产物。

用户提出的方向：弯曲项可能不必要；固定核球约束可能不必要；后续主要采用 random 初始化。下一步应将这些方向落实为可解释的消融，而不是同时改变全部设定。非整倍体与稀疏 SNP 辅助方向继续留在 [后续应用实验](RECONSTRUCTION_FUTURE_APPLICATIONS.md)，不混入本轮严格 SNP-free 优化。新 protocol 采用 C0、C1、C2-map、C2-free、C3 五个单项 arm；没有无 bend + free 组合、没有 r0/先验搜索，也不新跑 020 consensus。

主进程先读取本文和 020 实际配置、保存代码及停止审计，完成下述阶段 A；然后在首次新 fit 前写出具体 seed、迭代/函数调用预算、停止规则与候选清单。**这一旧要求现已由 023 诊断和 protocol freeze 兑现：480 accepted 是固定的有限预算，不是充分收敛证明。** 本文不虚构尚未测量的合适迭代数，也不把旧 300/200/240 次称为已校准预算。已有授权可覆盖的准备与诊断直接推进；本文不增加额外的确认门。

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

评价优先 allele 差异，但不能只奖励降低 cross_matched 而牺牲全部 matched 相似。禁止以两重构之间的距离、`1-rho(copyA,copyB)` 或单纯空间分离作恢复分数；无意义差异也能抬高这些数值。R3 继续作为长程一致性的辅助约束，R1 保留并附对照 ceiling；图和报告以四相关、contrast、双 margin 为主。

**训练/选择与评价分开：** 本文的“优先 contrast”是科研评价优先级，不是允许把 reference contrast 放进训练目标、早停或参数搜索。候选坐标及无标签选择在读取 reference/保留 phase 之前固定并 hash；所有预定消融都报告，不只留下 reference 表现最好的结果。用户已看过 P9016 评价，因此本轮属于针对同一样本的后续探索，不能包装成未触碰的独立验证。模型泛化需独立合成条件或后续独立样本检验。

## Revision 1：阶段 A 已完成，后续 protocol 已冻结

### 已有结果与实际进展

- **022 + 240 已完成**：这是计划 B 的固定 warm continuation，同一 020 `random_joint` 端点由 240 accepted 延至累计 1 Mb 480；`total J` 为 `9.61001577125 -> 9.60361804197`，`count_nll` 为 `9.59358593129 -> 9.58710105238`，终止状态是 `budget_not_converged`。它是预算诊断，不是独立初始化重复。FDG1000full 提案的原 J 未通过 label-free 规则，状态为 rejected。
- **023 固定端点诊断已完成**：见 [`023 README`](../test_res/023-20260913_212541-post020-allele-signal-diagnostics/README.md)。020/022 的 `maxR` 分别为 `0.65161710/0.68574970`，`>=0.90/0.95/0.99` bead 均为 0；这只能排除这些端点的活跃 hard-wall occupancy，不能说 sphere 参数化没有优化影响。sphere radial attenuation 最小值为 `0.43646488/0.38556995`，tangential attenuation 为 `0.75854806/0.72783744`。weighted bend y-gradient L2 为 `0.01216713/0.01206299`，count y-gradient L2 为 `0.03243707/0.03234493`，比约 `0.375/0.373`，所以 bend=.01 有实际作用。实际 repulsion hinge 是 `.7*l0=.04017409936`；旧 config 的 `repulsion_radius_l0=2.0` 和 native `d_r2.0` 是不同单位，不能混用。total gradient L2 为 `.00620367/.00514436`，末 20 accepted total J 降为 `.00091717/.0003898645`；020 六层和 022 都是预算停止，不称充分收敛。023 保存源与当前 `pr` 有差异，历史重现以保存代码的数值结果为准。023 的最终 bash75 与 validator 均 exit 0。
- **024 R2 派生已完成但不再返选训练**：见 [`024 README`](../test_res/024-20260913_133109-r2-allele-signal-derived/README.md)。原 V1 相对 Softall 的 contrast 为 `+0.044957`，CI `[-0.007420,+0.096044]` 跨 0，matched 变化为 `-0.055800`；022 相对 020 的 contrast 为 `-0.000222`，基本持平。原 V1 与 continuation 均为 20/20 finite，`both-positive=13/20`、`one-negative=7/20`，minmargin 分别为 `.049677/.047710`。父侧发现 fixed-reference 列锚定问题：swapped 时应换 candidate 行而不是 ref 列；024 v2 已修复。matched/cross/contrast/minmargin/counts 不受影响，旧 v1 的 absolute per-reference margin 或 contribution share 仍不引用，除非在 v2 下重新生成。
- **025 仅为 prepare-only**：实际目录为 [`025 native full-grid preflight`](../test_res/025-20260913_135100-random-native-fullgrid-preflight/)。父侧确认 3 bundles、count conservation 和 63 graph-swap checks 已通过，未调用 `hk_fdg`；它不是 native fit 完成证据。synthetic calibration、15 个 real fits 与本轮 real R2 evaluation 均仍为 `not_started`。

这些结果回答的是“预算、边界传递和读数语义是否已经能冻结”，没有回答 L2。协议机器文件为 [`POST020_ALLELE_ABLATION_PROTOCOL.json`](POST020_ALLELE_ABLATION_PROTOCOL.json)，完整文字约定为 [`POST020_ALLELE_ABLATION_PROTOCOL.md`](POST020_ALLELE_ABLATION_PROTOCOL.md)。

### 阶段 B：先锁定输入、起点和预算

任何 native prepare 或 joint fit 之前，必须在实际 run 的 `freeze/POST020_ALLELE_ABLATION_FREEZE.json` 记录 machine JSON、输入/source hashes，以及该阶段已生成的 prepared graph（到 x0 阶段则为 locked x0）hash，并向 append-only `test_res/POST020_ALLELE_ABLATION_ARTIFACT_REGISTRY.jsonl` 追加记录。MD/plan 的真实 mtime/hash 在文档可用时追加；若后补必须如实记录、不能回填成早于 fit，也不阻塞 native prepare。run 尚未分配时只使用：

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

### 阶段 C：合成 calibration（已冻结，尚未执行）

合成数据与真实使用同一 header、1 Mb、20 chr、5,290 points 和完整 eligible 构造。四个 fixture 的 identity 和 seed 不可互换：

| fixture | class | truth | draw | exposure | independent initial-shape | generating exposure |
|---|---|---:|---:|---:|---:|---|
| N1 | same internal shape, spatially separated negative | 260101 | 260201 | 260301 | 260401 | ones |
| N2 | same internal shape, spatially separated negative | 260102 | 260202 | 260302 | 260402 | lognormal sigma=.4, mean=1, no dropout |
| P1 | different-shape positive | 260103 | 260203 | 260303 | 260403 | ones |
| P2 | different-shape positive | 260104 | 260204 | 260304 | 260404 | lognormal sigma=.4, mean=1, no dropout |

四个 truth 独立，N1/N2 或 P1/P2 的跨 fixture 差异不能当作纯 exposure effect。生成 kernel 与 V1 相同、`p_gen=.8`；N2/P2 按实际 helper 生成 `np.exp(default_rng(seed).normal(0, sigma, N))`，其中 `sigma=.4`，再除以 realized full-grid mean，不猜 population log mean。每 fixture 固定 integer multinomial totals：diag `438,774`、cis-offdiag `696,680`、inter `568,434`，总 raw `1,703,888`，保留所有 eligible pairs；每 fixture 的实际 zero 数另行记录，不能套用 real `3,009,436`。

C0/C1/C2-map/C2-free 从 synthetic observed endpoints 重算 exposure，C3 只用 ones；generating exposure 不给 worker。每 fixture 的 independent init 一次调用 `generate_truth(startseed,false)` 生成 random chains，统一 normalize 到 maxR=.8；不取 truth、不扰 truth，五 arm 共用同一 x0，p0=.75。先完成 truth/observed-data freeze 和 prepare，再 fit；所有 fit coordinates hash 锁定后才在隔离 evaluator 读 truth。

合成共 `4 x 5 = 20` fits，预算为 `maxiter=80 accepted`、`maxfun=270`，其余 tolerance、maxls=20、checkpoint20 与 real 相同。N fixture 需满足每 chr 两 truth D 一致（rtol `1e-10`）且 centers 间隔 `>1e-6`；P normalized D difference `>1e-6`。失败 fixture 标 `invalid` 并保留，不换 seed。finite/gradient/grid/budget/code/hash 是 implementation gates；truth R2 增加不是放行门，truth 不返选参数。N 的 contrast 近零是同形状构造的代数性质，不是“不虚构分裂”的证明。N 报每 copy 对 common truth 的 shape error：每 chr 使用同一 finite unordered offdiag mask；候选两 copy 共用 `sC=sqrt(mean(两份候选 copy 在该 mask 上的 squared distances))`，truth 两 copy 共用 `sT=sqrt(mean(两份 truth copy 在该 mask 上的 squared distances))`（N 用 common-truth D 复制给两 truth copy），每 copy 计算 `RMS(Dc_copy/sC-Dt_copy/sT)` 的 chr mean/max，保留相对 copy 尺度，不做 per-copy rescale；另报 negative-only `RMS((DcA-DcB)/sC)`，不作 positive score 或 selection。P 同样使用两 copy 共用的 sC/sT，并报 own-truth 四 rho、contrast、margins 和 shape error。

### 阶段 D：五项 real ablation（已冻结，尚未执行）

所有 arm 都是同一输入、同一 1 Mb grid、同一 bundle physical x0 和同一预算；共 `5 x 3 = 15` joint fits。除明确列出的改动外，count/p_prior/bond/repulsion 权重均为 1，bend=.01（C1 除外），kernel `epsilon=1e-6`、`r0=2*l0`、repulsion=.7*l0、`p_floor=1e-4`、p-prior strength=`1e-4`、p_init=.75、`q_from_p(.75)`，`l0=(2*N_loci)^(-1/3)`；无层间或多分辨率归一化。

| arm | 唯一主改动 | exposure / parameterization |
|---|---|---|
| C0 | 新 1 Mb V1 original；不复现 020 path | observed-endpoint / current sphere |
| C1 | bend `.01 -> 0`，保留 bond | observed-endpoint / same sphere |
| C2-map | 只换 fixed identity-core `.90` smooth sphere map | observed-endpoint / `R<1` |
| C2-free | direct-x finite-only；无硬球、无 clip、无 rescale | observed-endpoint / direct Cartesian |
| C3 | 只把 full-grid exposure 设为 ones；保留 endpoint audit 和所有 normalizers | ones / same sphere |

C2-map 使用：`r=||y||`；`r<=.90` 时 `s=r`；否则 `t=(r-.90)/.10`、`s=.90+.10*t/sqrt(1+t^2)`、`x=s*y/r`（r=0 时 x=0）。C2-free 的 direct-x 同时改变参数化和可行域，因此 C0→free 不能归因纯 boundary；C2-map 只帮助区分 map sensitivity，仍不做夸大因果。没有无 bend+free 组合、没有 r0/先验搜索，020 consensus 仅保留历史对照不新跑。

每个 variant 内只按 final `count_nll_normalized` 选 3 starts 的 representative；tie `<=1e-12` 按 bundle ID。所有 15 outputs 都评价，不能跨 exposure/prior 按 J 宣称最佳。统计或数值失败、拒绝、未收敛都保留；不得因不利表现重抽 seed 或重跑。只有实现 bug 可在同一冻结 data/seed/budget 上修复重试，完整记录原/新 source hash，retry 不算 independent repeat。

### 阶段 E：real R2-only 隔离评价（尚未执行）

所有 15 endpoints、variant 内 selection、coordinates 和 source hashes 锁定后，才可读取 evaluation-only reference `data/P9016.1m.3dg.gz`（预登记 SHA256 `1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29`）；不读取真实 phase pairs。与 Softall seed124101、020 random、022 continuation、FDGfull rejected、random014、consensus014 使用同一 fixed controls 和共同 mask。位置范围为 `range(3Mb,L,1Mb)`；逐 chr 使用 finite、unordered、offdiag 的共同 mask，expected `157,529` common / `176,201` total non-diagonal pairs，实际覆盖必须核对。constant 或 nonfinite 输入，或某 chr common pairs `<20` 时该 metric 为 n/a；失败 arm 必须保留为 n/a，不得改变分母、局部换 copy 或从计划消失。

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

主要 paired comparisons 固定为 `C1-C0`、`C2-map-C0`、`C2-free-C0`、`C2-free-C2-map`、`C3-C0`，再与历史 controls 作描述比较。先在每个 seed 内配对同一 20 chr，再按每条 chr 对全部 3 个 seed 求效应；三 seed 是 optimization repeats，不是 biological replicates，也不把 60 条 chr 当 n=60。某比较若任一 seed 失败或没有有效 coordinates，主要三-seed 汇总必须为 n/a，同时保留 `planned_seed_count=3` 与 `valid_seed_count`；另列 available-seed 描述时需明确标注，不能静默变成二-seed主结果。

所有 metrics/comparisons 共用同一 seed=9301 的 `10000 x 20` bootstrap index matrix。interval 只表示一个细胞内的 structural/technical variation，不是 biological CI 或 p-value；不能只报最好 seed。R1/R3 按此前 R2-only 偏好不在本轮新增，L2 仍未证实。

### 阶段 F：资源与 release 顺序

已观察只读快照为 `192 logical CPU`、MemAvailable 约 `335 GB`，当时无当前拟合；这不是线性加速保证。native prepare 最多 2、synthetic fit 最多 4，正式 real fits 最多 6 个单线程 process，全球上限 6，BLAS/OMP=1。运行前记录可用内存和单 worker RSS；若 MemAvailable <32 GiB，不新起 worker，只等待已运行者结束，不改 budget。按 023 历史，real 单 fit 480 约 45--47 分钟，总 CPU 约 11.4 小时，实际 wall 为数小时且不能假设线性缩短。

release 顺序固定为：machine JSON/input/source/prepared-graph hash → run-specific freeze JSON/registry append → fixture/graph/native prepare → x0 hash lock → synthetic fits/calibration → real 15 fits → endpoints/selection/hash lock → isolated real R2 evaluation；protocol MD/plan 若在 native 后完成，追加真实 mtime/hash，不回填、不阻塞 native。任何一步失败都保存原始 artifact 和 n/a 状态，不跨步骤回填或伪称完成。

### 本 revision 的结论边界

本 revision 保留原计划的动机：检验 bend、sphere 参数化/可行域和 exposure 是否解释 020 相对 Softall 的整体差异，同时优先 allele contrast 而不是只提高整体 Spearman。现在把问题收敛为一个最小可解释实验：C1 检查 bend，C2-map 检查平滑 sphere map，C2-free 检查合并的参数化/无硬球敏感性，C3 检查 full-grid exposure；每项只改一个登记因素，并用合成 negative/positive fixture 识别失败模式。

目前只支持：023 的无 reference 数值语义已复核，024 的既有 R2 派生读数和不利 FDG proposal 已保存，025 的 graph/native prepare-only 检查已过。**不能**由本计划或 freeze 声称 synthetic calibration 完成、real ablation 完成、native equivariance、充分收敛、allele recovery 或 L2 whole-chromosome copy identity。L1 的历史 chr1 信号、L2 尚未建立、L3 对整个 nonlinear solver class 的撤回必须始终分开。
