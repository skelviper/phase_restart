# 参考差距独立方法审查

- 审查时间：2026-09-14 UTC
- 审查范围：只读已有协议、源码、保存的 020/022/025/026/029 结果与已完成评估，并纳入已完成的 030 training-side/native audit。
- 本审查没有启动 fit、长运行、参数搜索、R2 重算或大分析；没有修改原代码、原 reports、freeze 或任何既有 `test_res` 目录。
- 外部文献/网络检索没有作为本次归因证据；结论只依赖本地源码、协议、保存结果和已完成的本地实测。
- 新增配套证据矩阵：`evidence_matrix.tsv`、`evidence_matrix.json`；主控制设计纠正见 [`docs/POST020_CONTROL_DESIGN_CORRECTION.md`](../../POST020_CONTROL_DESIGN_CORRECTION.md)。

## 1. 结论先行

### 1.1 029 C0 为什么不同于 020

最强的结论不是某个单一因素已经造成差异，而是：**029 的实现设计相对用户要求发生了明确偏离：它更换了 starting source/native preprocessing，并取消了 020 的 5 Mb -> 2 Mb -> 1 Mb 多分辨率 continuation。** 因此 029 C0 与 020 不是同一 full-020 pipeline 的受控重复；现有差异只能作为不同 source、schedule、p/q path 和有限预算终点的 exploratory comparison。

- 020 的 `random_joint` 在 1 Mb 层不是 fresh 1 Mb random x0：它从批准的 014 blind random source 开始，经 5 Mb -> 2 Mb -> 1 Mb warm start；1 Mb 进入点 `max_radius=0.6127428727`，初始记录 `count_nll_normalized=9.6767139712`、`total=9.8442660571`、`p=0.9618223300`。
- 029 C0 是 1 Mb-only，使用 025 bundle2 的一次 native random-assignment x0；x0 shape 为 `(2,2645,3)`，全局 `max_radius=0.8`，配置的 `p_init=0.75`。029 第一个保存的 iter20 记录仍为 `count_nll_normalized=9.8761862880`、`total=10.0181210634`、`p=0.8284313506`。
- 020 1 Mb 终点为 `count_nll_normalized=9.5935859313`、`total=9.6100157712`、`p=0.9578874283`；029 C0 iter480 为 `9.6532884606`、`9.6748600161`、`p=0.9950156170`。029 C0 的 count NLL 比 020 高约 `0.05970`/record，total 高约 `0.06484`；这是终点差异，不是单独的 init 或 schedule 因果效应。
- 020 的 6 个 stage 与 029 的 15 个 real fit 都在 iteration limit 停止，均没有 `solver_converged` 证据。029 C0 iter480 的保存值 `gradient_norm=0.008399` 是 L2 诊断；SciPy `gtol=1e-6` 使用 infinity-norm，二者不能直接比较。当前未收敛证据来自显式的 maxiter termination；030 training-side audit 已完成并核对了正确的 gradient/状态重算，不能把 native 最后一次 RMS 当作 convergence。
- 最新只读 x0 标准审计按**每个 x0 自身**的 chromosome best-swap 计算，而不是沿 endpoint orientation 固定：C0 bundle1/2/3 的 endpoint-x0 macro matched/contrast 分别为 `+0.27911733/+0.08666839`、`+0.26150224/+0.08319607`、`+0.23176664/+0.06722237`；三者 matched 均为 20/20 wins，contrast wins 为 19/20、15/20、16/20。这证明每条 025 native-x0 -> C0 endpoint 路径在自身标准下确实改善，不能说优化把 025 起点“搞坏”。它**不**比较 025 起点与 020 的 014 source 起点谁更好，后者没有同一标准的因果对照；原 endpoint-fixed x0 读数只保留为 secondary，不用于主结论。
- 因此不能把 029 C0 相对 020 的 R2 差异写成已证实的代码回归，也不能把较低的 measured R2 等同于未知物理真结构更差。当前可写的是：在不同 x0、不同 resolution schedule、不同 p/q path、不同 native 初始化管线和有限 budget 下，两个有限时间 endpoint 不同；029 endpoint 相对自身 x0 有改善，但仍未达到 020。030 已完成的训练/native audit 没有找到 Python objective、grid/count、sphere mapping、3DG serialization、bridge ordering 或 backbone regression；它确认 014 `max_nei` 与 025 `max_nei=0` 是额外的 native initialization/contact-weight 差异。

### 1.2 代码口径与模型口径

对 020 provenance snapshot 与当前工作树的只读核对显示：

- `pr/contact_model.py` 和 `pr/genome.py` 的 hash 在 020 training-code manifest 与 029 source snapshot 中一致；冻结的 V1 observation/objective 也相同。
- `pr/joint_fit.py` 的差异是保存最近 analytic gradient、计算/记录 `gradient_norm` 的审计扩展；没有看到 objective formula 的改变。`pr/reconstruct.py` 的差异是把 iteration/function budget 的终止分类修正为可审计的 `budget_not_converged`，不是改变接触目标。
- 029 C0 的 objective 是 Python V1 连续目标；025 native FDG 只提供 029 的初始化 x0，不是 029 C0 的拟合目标。
- 但 V1 与 native FDG 本来就不是同一个数学目标：V1 使用
  `K(d)=epsilon+(1-epsilon)(1+d^2/r0^2)^(-2)`，`r0=2*l0`，再做 group-total-profiled count likelihood；native FDG 对每个 contact 使用 `n_eff=fdg_contact_count(...)` 和 `d_scale=n_eff^(-1/3)` 的 force target。这个差异是模型事实，**不是已证明的 reference-gap 原因**。
- 030 training/native audit 已完成：contact model/source、full-grid data arrays、bridge payload、track/bead ordering、backbone loop、native actual-call receipt 与 terminal log 均已核对；没有发现把 029 C0 差异归因于 Python objective 或 bridge/backbone regression 的证据。审计同时确认 014 标准 `-b1m` 会自动填 neighbor support（random `max_nei=19`），而 025 raw bridge 为 `max_nei=0`、使其 contact `k=1`；这是已证实的 native initialization pipeline 差异，尚未做成对 native comparison，也不需要在本次主控制设计中引入 025 starts。

### 1.3 为什么 029 与 reference 不同

reference gap 至少包含两个不能混写的层次：

1. **candidate endpoint 层**：029 C0 没有达到数值收敛，而且相对 020 改变了 init、schedule、p trajectory 和 endpoint。C0 与 C3/C2 的 paired R2 也只说明有限预算下的敏感性。
2. **reference/pipeline 层**：reference 是 evaluation-only 的 hickit-derived reconstruction，使用了不同的多分辨率与默认设置，并且 reference 侧先做了 phase imputation。`docs/MEASURED_FACTS.md` 的“Reference is a noisy target”历史 simple single-`-b1m` oracle split（源脚本 `docs/exploration/pilot_oracle.py`）曾记录 ours `Rg` 约 2.5、reference 约 1.1，但那不是 029 V1 的尺度或终点，不能用于当前 029 归因。当前 020/029 展示审计实际记录的 pooled RMS 为 reference `1.5529545301005403`、020 candidate `0.3766542939`、C0-bundle2 candidate `0.2942922368`；这些是各自 candidate/reference 的显示归一化尺度，不是物理单位，也不能用来解释 Spearman R2 差异。相同 common mask 的 shape comparison 还显示 020 vs 022 的两个 ref-aligned matrix rho 为 `0.99430/0.99180`，020 vs C0-bundle2 为 `0.36735/0.62807`；色标归一化坐标变化仅约 `0.7%-3.7%`，所以 020->C0 的矩阵形状变化是真实数值差异，不是色标造成。

chr1 的 locked 四-rho 也说明不能把高 matched 解释成双 copy 都已恢复：020 的 `rho(A,mat/pat)=(0.4894246598,0.7797499717)`、`rho(B,mat/pat)=(0.5601095185,0.5778770697)`，best-swapped 后 matched=`0.6699297451`、cross=`0.5336508647`、contrast=`0.1362788804`，两条 margin 为 `-0.01776755` 与 `+0.29032531`；C0-bundle2 的对应 matched=`0.5038604957`、cross=`0.4564611074`、contrast=`0.0473993883`，两条 margin 为 `-0.03762509` 与 `+0.13242386`。也就是说，matched mean 可被一条强 copy 拉高，同时另一条 copy 仍接近或偏离；这强化了“R2 是结构相似性读数，不是 L2 copy identity”的边界。

因此，V1 bounded kernel、observed-endpoint exposure、profiled likelihood 与 native/reference pipeline 的差异是**待隔离的假设**。正比例尺度变换不改变 Spearman；所以尺度只能解释 raw distance、pooled-RMS 或颜色位置，不能单独解释 R2 gap。R2 差异仍需在 endpoint、shape、pipeline、mask 和 observation/prior 机制之间区分，不能因为某个 kernel 或 exposure 的输出在 R2 上“更像 reference”就选择它。

### 1.4 当前可以和不可以说什么

- **L1 保持支持**：修正后的全基因组 label-free benchmark 中，oracle `0.5856`、random `0.5127`、consensus `0.5087`，oracle-random 为 `+0.0729 +/- 0.0520`，19/20 条染色体获胜；chr1 的证据最强。
- **L2 仍未建立**：029 的正 contrast、较低 count NLL、较高 matched similarity 或多个 separated tracks 都不能证明整条染色体的 long-range copy identity。
- **L3 仍撤回**：已有结果只否定特定 random-init/hard-EM、soft E-step、线性 phase retrieval 或 consensus sign-map 路径，不能否定所有 nonlinear solver。

## 2. 固定科学边界

本审查使用的对象是一个 P9016 单细胞、20 条 linked chromosomes、40 tracks、1,703,888 条 SNP-free records；不是生物学重复。1 Mb full grid 是 2,645 loci/copy、5,290 physical beads、3,496,690 eligible pairs，其中 438,774 same-bin records 由 saturated nuisance layer 单独处理。

- inter contacts 提供共同核内空间摆位和 territory/packing 约束，但没有 maternal/paternal haplotype label signal；不能拿它们做 haplotype accuracy。
- count objective 对每条染色体的 half-difference `u_c=(X_c-Y_c)/2` 有精确的 `u_c -> -u_c` gauge；总的 copy-label gauge 是 `2^20`。inter contacts 不会自动打破它。
- `u=0` 是偶函数目标的一阶驻点；偶性不等于它一定稳定，也不等于所有 solver 都不能逃离。
- 029 R2 固定 `ref1=mat`、`ref2=pat`，每条染色体至多一次 candidate-only whole-chromosome best swap；`contrast=matched-cross`。best-swap bias 和 one-cell bootstrap 必须保留在解释中。

## 3. 020 与 029 C0 的逐项比较

| 维度 | 020 `random_joint` | 029 C0 | 对解释的含义 |
|---|---|---|---|
| cohort/objective data | 1,703,888 records；20 chromosomes/40 tracks；1 Mb full grid；全量 count objective | 同一 SNP-free hash、同一 1 Mb full-grid contract、同一 record budget | 数据身份不是首要差别 |
| 进入 1 Mb 的 start | 从 014 blind random source 经过 5 Mb -> 2 Mb -> 1 Mb warm start；`maxR=0.6127428727` | 025 bundle2 native x0；`maxR=0.8`；shape `(2,2645,3)` | init 与尺度/几何起点不同 |
| start provenance | `candidate_base_seed=2207`；1 Mb initialization mode=`multiresolution_warm_start`；保留上一层 frame/scale | assignment/native seeds 为 `250102/250202`；一次 native call 后 global center/scale normalization；不读 phase | 不是同一个 random seed 的重复 |
| p/q path | 1 Mb 初始记录 `p=0.9618223300`，`q_in=3.2290977504`；说明 p 已由粗层路径带入 | config 固定 `p_init=0.75`；029 C0 第一个保存 iter20 为 `p=0.8284313506` | p 是 nuisance/path indicator，不是 phasing accuracy |
| schedule | `maxiter=300/200/240` for 5m/2m/1m；1 Mb terminal `nit=240` | `one_mb_only=true`；`maxiter_accepted=480` | 029 多了 1 Mb steps，但没有粗层预条件 |
| solver endpoint | 1 Mb `success=false`、iteration-limit；final gradient 未按旧 snapshot 记录 | `nit=480`、`success=false`、`budget_not_converged_maxiter`；C0 保存 `gradient_norm(L2)=0.0083992485` | 两者都不是 converged endpoint；L2 诊断不与 SciPy `gtol` infinity-norm 直接比较 |
| 020 count/total | initial `9.6767139712/9.8442660571`；final `9.5935859313/9.6100157712` | iter20 `9.8761862880/10.0181210634`；iter480 `9.6532884606/9.6748600161` | 029 终点更差，但不能分摊到某一个因素 |
| 020 final components | bend `1.1263446985`、bond `0.0019617233`、repulsion `0.0028836263`、p `0.9578874283` | bend `1.1795193356`、bond `0.0035595800`、repulsion `0.0056861379`、p `0.9950156170` | component 差异是 endpoint/path 结果，不是单项机制证明 |
| label-free selection | 只在 `consensus_joint` 和 `random_joint` 间按 final 1 Mb count NLL；选 `random_joint` | 每个 variant 内三 bundle 按 count NLL；C0 选 bundle2 | selection universe 与 selection boundary 不同 |
| R2 | 020 original matched `0.490326`、contrast `0.132956`；022 continuation 480 matched `0.489435`、contrast `0.132735` | C0-bundle2 matched `0.477405`、contrast `0.112523` | R2 gap 与 path/finite endpoint 混杂；不能当作代码回归 |

表中 020 数值来自 `stages/random_joint/1m.json` 的 initial/final/history；029 数值来自 C0 bundle2 terminal checkpoint、selection 和锁定 R2 report。完整引用见证据矩阵。

## 4. 因果机制审查

### 4.1 优化路径与设计纠正

**已知证据。** 020 使用批准的 014 blind random/consensus source，完整走 5 Mb -> 2 Mb -> 1 Mb warm start；029 C0 改用 025 native x0，直接从 1 Mb 开始。030 已确认 source/objective/grid/bridge ordering 没有回归，但也确认 025 与标准 014 native 输入的 `max_nei` 初始化不同；因此 029 不能当作 020 pipeline 的 controlled ablation。029 的三个 C0 bundle 都是一次 native initialization 后的 paired physical starts，且 020 的六个 stage 和 029 的 15 个 real fits 都 hit iteration limit。022 从 020 的 240 checkpoint 继续到累计 480，count/total 小幅下降，但 R2 matched 和 contrast 基本不变。

**反证/限制。** 029 三个 C0 bundle 的结果不是同一个 endpoint；bundle2 只是 variant 内 count-NLL representative。没有一个已保存的 crossed `init x schedule` factorial，所以不能把 029 的差异分摊到 source、native preprocessing、coarse preconditioning、p/q path 或有限预算中的某一个因素。x0 自身 best-swap 只证明各自 endpoint 相对自身起点改善，不比较 025 起点与 020 的 014 source 起点谁更好。

**可检验预测。** 这不是旧 029 结果可以回答的因果问题。下一轮先要求 C0 使用两条原始 014 source、三层 020 schedule 通过逐层 reproduction gate；只有 gate 通过后，才在每个变体的完整三层轨迹中解释 C1/C2-map/C2-free/C3 的 paired effect。若 C0 gate 不通过，先定位 source、adapter、p/q、objective、数组或 stop-state 差异，不进入变体归因。

**最小验证。** 主线不是旧四臂 pilot，而是严格 020-derived control：`5 variants x 2` 个原始 014 source conditions x `3` 个 resolution layers，共 30 stage attempts/10 trajectories；先跑 C0 两 source 的 6 个 stages，C0 gate 通过后才跑其余 24 个 stages。每个变体都从相同的 014 source 独立走 5 Mb -> 2 Mb -> 1 Mb，保留 full-grid adapter、interpolation/endpoint-fill/jitter、p/q carry、`.7*l0`、020 optimizer options 和原 count selection boundary；C2-map/free 先做 physical interpolation、legal pack/unpack 与 domain preflight。训练先看 label-free J、components、p/q、两种 gradient norm、termination 和 hash，坐标锁定后才做 R2，不做 R1/R3 或 reference-driven selection。旧 source-geometry x schedule 四臂只降为 C0 gate 后的可选诊断，不并入这 30 个主 stage。

### 4.2 V1 observation law 与 native FDG/reference target 的差异

**已知证据。** V1 protocol 明确规定 bounded finite kernel、`r0=2*l0`、observed-endpoint exposure、按 cis/inter group total profile 的 conditional count NLL。native `fdg.c` 在 contact force 中使用 `n_eff` 和 `d_scale=pow(n_eff,-1/3)`，并以 force-energy target 更新 bead distance。reference 的生成 pipeline 属于 hickit/native ancestry，且使用了 phase imputation 和不同的多分辨率/defaults。

**第一层应先做无采样噪声的 expected-count 校准。** 在 exact-V1 generating law、true exposure 和 true `p` 下，固定同一 full grid，对每个 group 令 `C*_ij=M_g rate_true,ij/S_true,g`。按 V1 的 conditional count expression，truth 处的 **count-only analytic gradient 应为 0**；candidate 相对 truth 的 excess count loss 是按 group/record 权重的 KL 型差异（去掉 candidate-independent 常数）。先比较 count-only loss/gradient，再比较 total `J`，因为 default bend/bond/repulsion/p-prior 在 truth 上未必 stationary。此层必须标记为 `synthetic_expected` cross-entropy，不能冒充 raw-integer P9016 或真实采样结果；通过后才进入 known-sampling/short-fit 层。

**反证/限制。** native `d_scale` 是 force/energy target，不是已定义好的 probabilistic contact kernel；029 C0 也没有把 native FDG 当作训练 objective，只把其输出用于 x0。这个差异是模型事实，不能单独推出 R2 gap 因果。`epsilon=1e-6` 目前没有被测量为尾部主导，不能随意当主因；球内距离虽至多约 2、`r0` 约 0.115，但仍需按实际 rate contribution 做 controlled audit。

**可检验预测。** 在 expected-count 层，law-matched V1 truth 的 count gradient 应接近零且 excess loss 为零；在同一 optimizer 下，若 exact-V1 generating counts 可被 V1 objective 恢复，而 native-like generating law 出现系统偏差，则 observation mismatch 是可重复的中介。若两种生成 law 都失败，优先怀疑 optimizer/basin 或 prior。

**最小区分实验。** 在已有 `v1_calibration` full-grid synthetic template 上固定 truth、true exposure、true `p` 和 group totals；先构造 `synthetic_expected` `C*` 并验证 truth count-gradient=0、candidate excess loss 的 KL 方向，再固定同一 x0/optimizer 比较 V1 K 与预先定义的 native-like count adapter。报告 known-truth shape/contrast、count-only loss/gradient、total J 和 pooled RMS/raw-distance diagnostics；后者不是物理尺度，不得以 real reference R2 选 kernel。任何 adapter 必须先明确它把 native `d_scale(n_eff)` 映射到什么 count-generating law，否则实验本身不可解释。

### 4.3 Exposure 与 profiled likelihood

**已知证据。** V1 exposure 是 `sqrt(endpoint_count+10)/mean`，不优化，且协议承认它可能把 capture、mappability、visibility、bin capacity 和 real geometry 混在一起。C3 唯一注册变更是 uniform full-grid exposure；在 029 paired R2 中 C3-C0 matched `+0.013042`，CI `[+0.003714,+0.022705]`，但 contrast `+0.006934` 的 CI `[-0.008487,+0.023061]` 跨 0；所以只显示 overall similarity 变化，不显示确定的 allele-specific gain。

**反证/限制。** C3 是 real finite-budget ablation，仍与 C0 共享 native starts，但 endpoint 未收敛；它不能把 exposure 变化和 reference pipeline、p path 或不同最终几何完全分开。026 实际是 N1/N2/P1/P2，四类均为 `synthetic_integer` 且固定 `maxiter=80`；其中 P1/P2 使用独立 truth 和独立 observation/exposure fixture，不能把 P1/P2 差异当纯 exposure effect，N1/N2 又是 duplicated common truth 的 negative controls。这里不应声称不同 exposure 下的 count NLL 在数学上不可比：在相同 counts `C`、相同 group totals `M_g` 和同一个 conditional expression 下，三个 exposure view 的 count NLL 本身可以比较；029 不跨 variant 选 representative 是冻结的 selection rule，而不是数学不可比。

**可检验预测。** 在同一 counts/truth/x0 下，若 known generating exposure 相对 observed-endpoint exposure 和 uniform exposure 稳定提高 known-truth difference-shape/contrast，exposure misspecification 是重要中介；若只改变 matched 而不改变 difference readout，则它主要影响 shared geometry。

**最小区分实验。** 从一份 exact-V1 synthetic count fixture 派生三种 data view：known generating exposure、production observed-endpoint exposure、uniform ones；counts、`C/M`、truth、x0、objective kernel、p initialization、budget 全固定。以 P positive 与 N same-shape negative 同时评价；固定 truth 只用于 evaluation。若 known-vs-observed 的 difference readout 在两个 positive fixture 中都不改善，停止把 exposure 当第一主因；仍保留其对 shared R2 的影响。

### 4.4 shared structure 与 half-difference `u`

**已知证据。** 当前 V1 是 two-copy joint coordinates，不冻结 consensus；目标对 `u` 是偶函数，whole-chromosome sign 是 gauge，`u=0` 只保证一阶驻点。历史 F6 中 random-init hard-assignment + native-FDG loop 停在近 collapse，oracle initialization 保持 signal；F10 显示 collapse 和 truth basin 都可稳定，F12 的线性 phase retrieval 因 `|delta|/dbar=0.51`、relative error `1.40` 失败。

**反证/限制。** F6/F10 是 hard-EM/native-FDG 或特定 basin diagnostic，不是 029 continuous V1 C0 的直接 stability theorem；偶函数不能推出 `u=0` 必为 local minimum。F12 否定的是线性化求解，不是否定 nonlinear phase retrieval。029 C0 的 R2 结果没有保存一个独立的 `u` basin experiment。026 N1/N2 的 copy-difference RMS 约为 `0.5384-0.5631`，这是尚未控制的伪分裂诊断，不是合格上界或 pass/fail threshold。

**可检验预测。** 在 exact-V1 synthetic positive fixture 中，truth-informed nonzero `u` start 若保持正向 difference readout，而 zero/small random `u` start 退回 shared solution，则 real C0 gap 更像 basin/phase-retrieval failure；若 truth-informed start 也退回，才需要优先重审 observation law 或 prior。

**最小区分实验。** 这是 synthetic calibration-only 的 oracle/warm-start 诊断，不是 blind real fit：generation side 可读取并锁定已知 truth，构造同一 shared `Z` 的 `u=0`、small random、medium nonzero、truth-informed 四种起点；worker 只接收已序列化的起点和无标签 counts，blind random arm 不读 truth，真实 reference 绝不用于 init。先用 known true exposure/true `p` 作为合成校准上界，再单独比较 production observed-endpoint exposure；记录 `u` amplitude、difference contrast、N false-split RMS、count-only NLL 与 gradient trajectory。`u=0` 和 truth-informed start 的结果只用于判断 basin，不可转写成 real recovery。

### 4.5 prior、generator 与跨轨排斥：不能泛称 bond mismatch

**已知证据。** V1 bond prior 的无惩罚区间是 `[0.75,1.25]*l0`，bend weight 为 `0.01`，cross-copy/all-bead repulsion threshold 为 `0.7*l0`。合成 truth 的 `_chain_offsets` 每步明确使用 `Uniform(0.85,1.15)*l0`，并拒绝 territory 外、unit-ball 外和同链 nonadjacent distance 小于 `0.25*l0` 的 proposal；metadata 明确 `prior_equilibrium_claim=false`。所以不能说 synthetic truth 天然违反 bond 区间。

**反证/限制。** generator 没有声明它从 V1 的 bend、跨轨 all-bead repulsion 或 observed-endpoint/profiled count model 的平衡分布抽样；因此 bend mismatch、cross-track repulsion mismatch、exposure/profiled-likelihood mismatch 仍是不同的待检验假设。029 C1 去掉 bend 后 matched effect `+0.000093` 的 CI 跨 0、contrast `-0.013592` 的 CI 也跨 0；这不足以证明 bend 是主因。C2-map 与 C0 的比较是球内 parameterization 变化；C2-free 与 C2-map 才是域限制相关比较（同时记录 free 的 direct-x parameterization）；C2-free 与 C0 是 parameterization + feasible-domain 联合变化，不能称为纯“去球”单因素。029 C2-map/free 在 real480 逐元素相同，不能把它解读成球边界已被隔离。

**可检验预测。** 在 exact-V1 synthetic task 中，若只改变 bend 或只改变 cross-track repulsion，known-truth difference readout 与 N false-split rate 应分别呈现可重复变化；若变化只发生在 shared shape/scale，则不能把它称为 allele recovery mechanism。

**最小区分实验。** 固定 exact-V1 truth、counts、exposure、x0 和 C0 observation law，做最小 `bend={0,0.01} x cross-track-repulsion={off,on}` factorial；同时保持 bond 区间与 kernel 不变。报告 P positive 的 known-truth contrast/shape error 和 N negative 的 false-split RMS，全部 arm 完整保留，不以 real reference 选 model。若某项在 P/N 两类都没有方向一致的效应，停止把该项作为主解释；若只影响 N false split，归为 regularization/suppression 而非 recovery gain。

### 4.6 reference/native prior 与尺度

**已知证据。** 已保存的 native provenance 表明 vendored hickit 与 reference 所用 phase3 FDG implementation byte-identical；但 reference pipeline 先以 `hk_impute` 传播 phase，再运行支持 phase probabilities 的流程。`docs/MEASURED_FACTS.md` 的“Reference is a noisy target”历史 simple single-`-b1m` oracle split（源脚本 `docs/exploration/pilot_oracle.py`）曾记录 ours `Rg` 约 2.5、reference `Rg` 约 1.1；该数值属于特定历史 pipeline，不是 029 V1，不能移作当前定量归因。当前 020/029 展示审计的实际 pooled RMS 为 reference `1.5529545301005403`、020 candidate `0.3766542939`、C0-bundle2 candidate `0.2942922368`；它们是显示归一化尺度，不是物理单位。相同 common mask 的 shape comparison 为：020 vs 022 两个 ref-aligned matrix rho `0.99430/0.99180`，020 vs C0-bundle2 `0.36735/0.62807`；色标归一化坐标变化仅约 `0.7%-3.7%`，不能造成后者的真实 shape difference。

**反证/限制。** 相同 native ancestry 不等于相同输入、相同 phase imputation、相同 resolution schedule、相同 contact target 或相同 scale。正比例尺度不会改变 Spearman R2；pooled RMS/颜色只能解释 raw distance 或显示位置，不能单独解释 R2 gap。单个 reference 与单个 cell 不能分辨 pipeline bias、model misspecification 和 candidate endpoint error。

**可检验预测。** 在 known-truth synthetic data 上，若固定同一 mask 和正比例尺度后 raw distance/pooled-RMS/颜色不同但 Spearman R2 不变，则只能支持显示尺度差异；若 R2 也不同，必须另从 shape、pipeline、mask 或 endpoint 解释，不能归因为 scale。只有在 law-matched 条件下某一 pipeline 的 known-truth agreement 更好，才支持对应 observation/prior mismatch。

**最小区分实验。** 主 pipeline 对照使用同一份无标签 synthetic counts、同一 full-grid/input budget、同一初始化 source geometry 和同一允许信息，分别跑 native-like 40-track pipeline 与 V1 C0；evaluation 侧使用同一 mask、同一 rank/Spearman R2 和同一 raw-distance/pooled-RMS 报告。不得让任一方读取真实 assignment。若额外做带真实 assignment 的 native oracle，只列为独立 ceiling，不能拿它与 blind V1 的差值归因于纯 pipeline。把 truth agreement、cross-copy contrast、R2、pooled RMS 分开；reference 只能作为末端 display，不参与 kernel/prior/endpoint 选择。030 native actual-call/骨架审计已完成；它确认 014/025 `max_nei` 输入差异，但这项成对 native comparison 仍未执行，不把该差异外推为 V1 训练回归。

### 4.7 信息量、噪声与 identifiability

**已知证据。** corrected label-free baseline 已显示 chr1 上 oracle split 高于 random/consensus；同时 inter contacts 的 label counts 近似 same/different 1:1，只提供几何。没有 molecule-level linkage，`readID='.'` 全部如此。20 条 chromosome 是 one-cell linked measurements。R2 positive contrast 还受 candidate-only best-swap bias。

**反证/限制。** L1 predictive signal 不等于 L2 whole-chromosome recovery；chrX 的 phased depth、coverage evenness 和 held-out pair count 都更差。没有独立 biological replicate，也没有已完成的 blind nonlinear solver 证明能将 signal 变成 coherent `u`。

**可检验预测。** 只要在 held-out/known-truth label-free instrument 上保留 oracle>random 的排序，information 尚未被否定；若优化方法在同一 instrument 上不能稳定达到非零 coherent difference，则瓶颈在 basin/solver/model，而不是直接推出无信息。

**最小区分实验。** 对每个候选 solver 先在 exact-V1 synthetic P/N 上做 frozen known-truth calibration，再对 real data 严格沿原 020 的 label-free final count NLL 规则选择 candidate；convergence、u-coherence 和坐标 hash 后的预注册 R2 只作诊断/评价，不新增选择条件。当前默认不新增 R1/R3；若要把 whole-chromosome fragment consistency 作为 L2 主判据，应另立并获授权的任务，不纳入本审查的默认步骤。单个 R2 positive 或 two-track separation 不足。

## 5. 为什么“多跑几步”或“改 learning rate”单独不能回答问题

1. **更多迭代只改变终点，不识别路径来源。** 022 已从 020 的 240 accepted steps 延到累计 480；count NLL/total 确实下降，但 R2 matched 与 contrast 几乎不变，而且 continuation 仍是 `budget_not_converged`。这说明“还没到终点”是可能性，但不说明 R2 gap 来自 init、schedule、objective misspecification 还是 reference mismatch。
2. **L-BFGS 的 iteration limit 不是 convergence。** 029 C0 的 `nit=480`、`success=false` 是未收敛的直接证据；保存的 `gradient_norm=0.008399` 只是 L2 诊断，不能与使用 infinity-norm 的 `gtol=1e-6` 直接比较。030 training-side audit 已重算并核对正确的 gradient/state 记录；native 最后一次 RMS `139-144` 是 last-attempt diagnostic，不是返回 best state，也不是 convergence。继续跑可能沿当前 basin 下降，也可能只改变 shared geometry。一个更低的 J 不能替代 known-truth/locked-R2 readout。
3. **单独改步长会同时改变 trajectory、line-search 接受和 basin。** 若不固定 x0、p/q initialization、resolution schedule、objective law 和 stop rule，learning-rate effect 无法与 path effect 分开。并且本流程的 `scipy L-BFGS-B` 没有一个“learning rate”单参数可以直接解释全部 native/continuous 行为；应记录可重复的 optimizer options、line-search diagnostics 和 accepted trajectory。
4. **reference 不能作为早停或调参目标。** R2 只有在 coordinate hash lock 后才能运行；用“更像 reference”的 endpoint 选择 kernel、exposure、prior 或步长会把 evaluation target 反向泄漏进方法选择。

因此最小先验检验不是“直接把 maxiter 加大”或“找一个更合适的步长”，而是先完成严格的 full-020 C0 reproduction gate：两个 source 各走三层，checkpoint 每 10 accepted steps。只有 gate 通过后，C1/C2-map/C2-free/C3 的跨层受控消融与后续 solver/observation/prior sensitivity 才有可解释基线；旧 crossed `init x schedule` 四臂和 solver-options sensitivity 都只能作为 gate 后的可选诊断。

## 6. 优先级实验清单（最多五项）

以下每项都给出 concrete problem -> importance -> small example -> minimum validation experiment；建议顺序是先把 optimization path 与 observation law 分开，再做 prior/reference calibration。

### E1. 严格复现 020，先建立 C0 control gate

- **问题：** 029 C0 更换了 014 blind source/native preprocessing，取消了 020 的 5 Mb -> 2 Mb -> 1 Mb continuation，并使用了不同的 1 Mb budget 与 selection universe；因此旧 029 C0/C1/C2/C3 不是 020 的受控消融。
- **重要性：** 只有先复现同一 source、full-grid adapter、interpolation/jitter、p/q carry、objective、三层 budget 和 label-free selection，后续 variant 差异才可归因于登记的处理变量。
- **小例子：** 020 `random_joint` 1 Mb endpoint 的 count NLL/total 为 `9.5935859313/9.6100157712`；029 C0 bundle2 iter480 为 `9.6532884606/9.6748600161`，但两者不是同一初值或 schedule。030 已确认 source/objective/grid/bridge ordering 无回归，同时确认 014/025 native `max_nei` 管线差异。
- **主设计：** `5 variants x 2` 个原始 014 source conditions x `3` 个 resolution layers，共 `30` 个 stage attempts、10 条完整 trajectories。先跑 C0 两个 source 的 6 个 stages；逐层 gate 通过后才跑 C1/C2-map/C2-free/C3 的 24 个 stages。每个 variant 都独立从 014 consensus/random source 走 5 Mb -> 2 Mb -> 1 Mb，不喂入其他 variant 的 coarse/final endpoint。
- **固定合同：** 保留 020 的 source SHA/base seed、full-grid adapter、线性 interpolation/endpoint fill、deterministic jitter、单次共同 center/uniform scale、observed-endpoint exposure（C3 才改为 ones）、`.7*l0`（按 layer 更新）、p 首层 `.75` 与 raw q carry、`maxiter/maxfun=300/930 -> 200/630 -> 240/750`、`maxls=20`、`ftol=1e-10`、`gtol=1e-6`、checkpoint 每 10 accepted。C0 的 unit-ball radial clip 由 C1/C3 继承；C2-map/free 只执行各自登记并经预检的合法 domain 操作。
- **C2 解释：** C2-map vs C0 是球内 parameterization 变化；C2-free vs C2-map 才是域限制相关比较；C2-free vs C0 是 parameterization + feasible-domain 联合变化，不能称纯“去球”单因素。C2 跨层先做 physical interpolation、legal pack/unpack、finite、mapping 和 free 旧 sphere writer/clip 隐式限制检查；预检失败就停止。
- **C0 gate：** 不读 reference、不用 R2 放行；逐 checkpoint 比较 source/input/data arrays、entry/exit state、p/q、各 component、J、gradient-L2/gradient-infinity、nfev/nit、termination 和 selected candidate。整数/配置/hash 要 exact，浮点容差在执行前冻结；任一未解释差异先定位，不启动其他 variant。坐标 hash 通过后才做 R2；不做 R1/R3 或 reference-driven selection。
- **可选诊断：** 旧 source-geometry x schedule 四臂 pilot 只在 C0 gate 后另行授权，不计入这 30 个主 stage；它不用于把 025 native 初始化算法与 020 source 作因果优劣比较。

### E2. 已知 generating law 与 native-like target 的 observation isolation

- **问题：** V1 `K`/profiled likelihood 与 native FDG `n_eff^(-1/3)` force target 不是同一目标，reference 又来自 native/imputed pipeline。
- **重要性：** 它决定 reference gap 是 observation mismatch、optimizer/basin 还是显示尺度差异；不能用 real R2 直接挑模型。
- **小例子：** V1 代码在 `contact_model.py:671-684` 计算 bounded `K`；native `fdg.c:390-397` 先计算 `n_eff` 再用 `d_scale=pow(n_eff,-1/3)`。
- **最小验证：** 先做 expected-count 层：固定同一 full-grid truth、true exposure、true `p` 和 group totals，令 `C*_ij=M_g rate_true,ij/S_true,g`；验证 truth 的 `count-gradient=0`（count-only analytic gradient），并检查 candidate excess count loss 的 KL 方向。该层标记为 `synthetic_expected` conditional cross-entropy，不作为 raw-integer 真实数据结果。通过后再固定同一 x0/optimizer 比较 V1 K 与预先定义的 native-like count adapter，报告 known-truth shape/contrast、count-only loss/gradient、total J 和 pooled RMS/raw-distance diagnostics；后者不是物理尺度，不得以 real reference R2 选 kernel。任何 adapter 必须先明确它把 native `d_scale(n_eff)` 映射到什么 count-generating law，否则实验本身不可解释。
- **预测：** 若仅在 law-matched 条件下恢复，则支持 observation mismatch；若两组都失败，则优先回到 E1/basin 或 prior。
- **停止规则：** native-like adapter 无法在生成前定义唯一 count law时不启动；若 V1-generated positive control 也不能恢复，停止把 real gap 归因于 kernel；若两种 law 在 known truth 上同样表现，则不增加 kernel 搜索。

### E3. 同一 count fixture 的 exposure/profile ablation

- **问题：** observed-endpoint exposure 可能把 visibility/capture 与 geometry 混在一起；C3 只显示 matched gain，未显示确定的 difference gain。
- **重要性：** 可直接测试 C3 的 real signal 是 shared geometry 变化还是 allele-specific recovery。
- **小例子：** 029 `C3-C0` matched `+0.013042` 的 CI 不跨 0，但 contrast `+0.006934` 的 CI 跨 0。
- **最小验证：** 一份 exact-V1 synthetic counts 固定不变，派生 `known generating`、`observed-endpoint`、`uniform` 三个 exposure view；同一 truth/x0/objective/budget，P1/P2 positive 与 N negative 都跑。truth 只在 hash-gated evaluation 使用。
- **预测：** known exposure 若稳定提高 difference contrast/shape readout，exposure 是中介；只提高 matched 则主要影响 shared geometry；三者无差别则 exposure 不是首要解释。
- **停止规则：** 两个 positive fixture 都不复现 known-vs-observed 的方向，停止把 exposure 当第一主因；N 出现可重复 false split 时拒绝该 exposure/regularization 组合。

### E4. 非零 half-difference basin 与 `u=0` 对照

- **问题：** objective 的偶性让 `u=0` 一阶驻点，但不能从数学偶性推出 stability；目前没有 029 C0 的 direct basin map。
- **重要性：** 这是区分“信息存在但 blind optimizer 没进入 basin”和“objective/observation 本身无法保持 split”的最小测试。
- **小例子：** F10 的 hard-EM 在 chr1 显示 collapse 与 truth 都是 attractor；F12 线性 phase retrieval relative error 为 `1.40`，但 nonlinear 版本尚未试过。
- **最小验证：** 这是 synthetic calibration-only 的 oracle/warm-start 诊断，不是 blind real fit：generation side 读取并锁定已知 truth，构造同一 shared-Z 的 `u=0`、small random、medium nonzero、truth-informed 四种起点；worker 只接收已序列化的起点和无标签 counts，blind random arm 不读 truth，真实 reference 绝不用于 init。先用 known true exposure/true `p` 作为合成校准上界，再单独比较 production observed-endpoint exposure；记录 u amplitude、P difference contrast、N false-split RMS、count NLL 与 gradient trajectory。truth-informed arm 只用于判断 basin，不可转写成 real recovery。
- **预测：** P 中非零 start 保持 signal 而 zero/random start collapse，支持 basin；truth-informed 也退回则优先查 observation/prior；N false split 上升则 regularizer/start 不可接受。
- **停止规则：** 026 现有 N-only RMS `0.5384-0.5631` 仅是尚未控制的诊断，不设 pass/fail 上界。必须先指定数值容差并建立独立 sampling-noise baseline，再相对该 baseline 报告；若 baseline 尚未校准，只报数值、不设 passflag。任一 start 非有限或不满足 physical domain 即停止该 start family；没有 truth-informed positive retention 时不推进 real blind nonlinear fit。

### E5. bend/repulsion 与 reference/native pipeline 的小型校准

- **问题：** prior/generator、native FDG、reference pipeline 与 V1 scale 的差异尚未分解；不能把它们笼统称为“bond mismatch”。
- **重要性：** 只有在 exact observation law 和 optimization path 已知后，才可判断 shared prior、pipeline/shape 或显示尺度是否造成 residual gap。
- **小例子：** synthetic chain step 是 `0.85-1.15*l0`，在 unit-ball/territory/nonadjacent rejection 下生成；V1 bond 无惩罚区间是 `0.75-1.25*l0`，而 native FDG contact target 另按 `n_eff` 缩放。
- **最小验证：** 先在 exact-V1 synthetic truth 上做 `bend={0,0.01} x cross-track-repulsion={off,on}`，bond/kernel/exposure/x0 固定；再用同一 known truth/count fixture 对 V1 与 native-like 40-track pipeline 做同一 mask 下的 truth agreement、R2、pooled RMS/raw-distance 和 count J 分开比较。R2 只作锁定后的 evaluation，不用 reference 选参；pooled RMS/颜色只作显示尺度诊断，不当物理尺度。
- **预测：** bend/repulsion 若只改变 shape/false split，不能写成 allele recovery；若 raw distance/pooled-RMS/颜色变化而正比例缩放下 Spearman R2 不变，这是显示尺度效应，不能解释 R2 gap；R2 的变化需另由 shape、pipeline、mask 或 endpoint 解释；若只有某个 law-matched pipeline 在 known truth 上恢复，才支持 corresponding observation/prior mechanism。
- **停止规则：** 若不能在同一 fixture 中固定 truth、counts、x0 和 observation law，就不对 prior 变化作因果归因。任何 arm 预算停止都只能作 finite-path sensitivity；若 P/N 没有方向一致的 prior effect，不再扩大 prior 搜索。030 native actual-call/输入骨架/backbone audit 已完成；它确认 014/025 `max_nei` 初始化差异，但尚未做成对 native comparison，不把该差异外推成 V1 训练回归。

## 7. 最终判断

029 C0 相对 020 的差异目前应标为：**设计上更换了 starting source/native preprocessing、取消了多分辨率，并因此形成不同 p/q path 与有限时间终点**。这解释了为什么 029 不是对 020 的受控重现；它不等于发现 Python 代码错误，也不等于证明 029 exploratory 方法失败。x0 自身 best-swap 审计进一步表明三个 025 native x0 到各自 C0 endpoint 都有 macro matched/contrast 改善，因此不能说 C0 优化把起点搞坏；但没有 025-x0 与 020-014-source 的同标准因果对照，不能判断哪个起点本来更好。

V1 与 native/reference 的 contact law、exposure、profiled likelihood、prior 和 scale 差异都值得做受控校准，但目前只能作为可证伪假设。030 training/native audit 已完成：没有发现 Python objective、grid/count、sphere mapping、3DG serialization、bridge ordering 或 backbone regression；同时确认 014 与 025 的 `max_nei` 输入/初始化差异，且 native 固定预算的最后一次 RMS 不是 convergence 证据。尤其不能把 synthetic truth 写成 bond 先验天然失配，也不能把 `epsilon` floor、sphere boundary、bend 或 cross-track repulsion 中任何一个未经隔离的因素升级为主因。

029 的 real R2 只能支持有限预算下的 comparative sensitivity：C3 提高 overall matched similarity，但没有明确 difference-signal improvement；C1/C2 没有建立 beneficial allele recovery；C2-map/free 的 real480 exact tie 不能回答球外访问问题。最终科学状态仍是 L1 保持、L2 未证明、L3 对整个 nonlinear solver class 撤回。

## 8. 主要证据索引

- 主控制设计纠正：`docs/POST020_CONTROL_DESIGN_CORRECTION.md`（严格 full-020 C0 gate、30 stage attempts/10 trajectories、C2 比较边界与未实施边界）。
- 030 训练侧/native 审计：`test_res/030-20260914_c0-020-training-audit/REPORT.zh-CN.md`、`verification.json`、`native_audit.json`。
- B display/R2/x0 audit 与统一图：`docs/audits/c0-020-display-20260914_021033/AUDIT.md`、`unified_chr1_distance_matrices.png/.pdf`。
- 020 1 Mb initial/final/initialization：`test_res/020-20260913_071841-v1-p9016-joint/stages/random_joint/1m.json:6108-6224`。
- 020 全部 iteration-limit 审计：`test_res/020-20260913_071841-v1-p9016-joint/termination_audit.json:11-15,16-166,186-193`。
- 022 warm continuation：`test_res/022-20260913_111031-v1-continuation-fdg-r2/continuation_summary.json:1-34,63-134`。
- 029 contract 与 x0：`test_res/029-20260913_161713-post020-allele-ablation-real/config.json:23-63,241-294`。
- 029 终止与 selection：`test_res/029-20260913_161713-post020-allele-ablation-real/finalize.json:1-31`、`selection.json:1-56,100-263`。
- 029 C0/C3 terminal components：`jobs/C0-bundle2/attempts/attempt-001/final.json:13325-13377`、`jobs/C3-bundle2/attempts/attempt-001/final.json:13325-13377`。
- 029 R2 paired results 与 C2 boundary：`docs/POST020_ALLELE_ABLATION_RESULTS.md:202-253`。
- chr1 四-rho、best-swap 与 margins：020 `test_res/020-20260913_071841-v1-p9016-joint/evaluation-20260913_080300-final/metrics/metrics.json:5580-5604`；029 C0 `test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_per_chromosome.json:59-90`。
- 020/029 C0 展示、shape 与 x0 自身 best-swap 审计：`docs/audits/c0-020-display-20260914_021033/AUDIT.md:72-105,133-136`。
- V1 kernel/exposure/count/prior：`docs/RECONSTRUCTION_V1_PROTOCOL.md:108-239,292-323`；expected-count 实现与 fractional-mode 审计：`pr/contact_model.py:802-851`。
- 生成器链/接受规则/内核/曝光项：`pr/v1_calibration.py:248-343,368-438`。
- 026 synthetic N1/N2/P1/P2 结构、count mode 与 N RMS 边界：`test_res/026-20260913_221709-allele-calibration-prepare/reports/026_synthetic_allele_calibration_zh.md:5-18`。
- native FDG 目标缩放：`native/hickit/fdg.c:316-361,390-399`。
- reference scale/pipeline 与无 molecule linkage：`docs/MEASURED_FACTS.md:626-684,686-717`。
- L1/L2/L3、inter与 `2^20` gauge：`AGENTS.md:18-32,54-77`。
- 偶性与残差/solver边界：`docs/PLAN-genome-allele-split.md:35-60,120-150`。
