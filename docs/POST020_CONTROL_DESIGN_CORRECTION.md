# POST020 控制设计纠正与综合结论

- 状态：**033 已严格复现 020 并完成 C0 gate；CPU 034 因用户要求切 GPU 而停止/正在终止，GPU 迁移待验证；R2 尚未运行**。
- 本文是 POST020 控制口径的主入口。它不启动 fit、不改写原始 fit/protocol/report/freeze，也不把建议写成已运行结果。
- 真实对象是一个 P9016 单细胞：20 条 linked chromosomes、40 条轨迹、`1,703,888` 条 SNP-free records；不是生物学重复。
- 本文只保留本轮授权的 R2 读数、training-side label-free diagnostics 和 synthetic known-truth calibration；R1/R3 不属于本设计默认步骤。

## 一句话结论

用户指出的纠正是实质性的：**029 C0 更换了起点和 native 预处理，并取消了 020 的多分辨率 continuation；这是相对用户意图的设计偏离，不只是命名不清。** 因而 029 C0、C1、C2-map、C2-free、C3 的既有结果不能代表“沿用 020 全流程、只改变指定变量”的受控消融。

029 应保留为**独立的 1 Mb 探索性消融**。它可以说明 025 native starts 上的有限路径敏感性，但不能据其 C1/C2 结果宣称某项改动对 020 全流程无效，也不能用它回答“同一 020 pipeline 下取消 bend、改变 sphere map、去掉 sphere 或改 exposure”的因果问题。

本纠正覆盖旧建议中任何“从 020 已优化终点再接一段 1 Mb 就等于复现 baseline”的说法。一个 020 的 1 Mb endpoint continuation 只是在既有状态上继续优化；它不等于从 014 blind source 重新走 5 Mb -> 2 Mb -> 1 Mb，也不等于 020 的完整 baseline reproduction。

## 1. 已完成审计给出的事实

### 1.1 030 训练侧审计

030 审计只读取 training-side 文件、保存的 020/022/029 锁定状态、025 native receipt/log/binary、014 training-side 坐标和 bridge/native 源码；没有打开 reference、phase 或 R2/evaluation payload，没有调用 optimizer，也没有重新调用 native。

它确认：

- 输入为 `inputs/P9016.snpfree.pairs.gz`，SHA256 为 `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`，共 `1,703,888` 条记录。
- full-grid 是 origin-0、1 Mb、20 chromosomes、2,645 loci/copy、5,290 physical beads、40 tracks；track-major 顺序为 `c01a,c01b,...,c20a,c20b`。
- 当前与归档 `contact_model.py` byte-identical，SHA256 为 `cf19833febc97e346bbaa24f3520c385fdcfb18180957a23b59838f7402e8ace`。
- 020、022、029 C0 锁定状态的 objective、各分项、gradient 和坐标重算最大绝对差为 `0.0`；030 的 5 Mb/2 Mb/1 Mb 数据数组、writer/parser roundtrip、grid/mapping/normalization checks 均通过。
- 因此当前没有证据支持 Python V1 objective、grid/count、sphere mapping、3DG serialization、bridge ordering 或 backbone construction 回归。

这不等于 029 是正确的 020 消融。它只说明：在既有锁定状态上，没有找到把差异归因于上述实现回归的证据；实验设计本身仍改变了 source、预处理和 resolution path。

### 1.2 已证实的 native 初始化管线差异

030 的 native audit 通过 10 项检查，但确认了 014 标准 CLI 与 025 bridge 的输入差异：

- 014 `-b1m` 输入没有 `n_nei` 列时，标准 CLI 会先调用 `hk_pair_count_nei`；random 日志的实际 `max_nei=19`，consensus 为 `69`，oracle 为 `7`。
- 025 bridge 对 raw `hk_pair` 先 `memset`，实际 `max_nei=0`；`fdg.c` 的中位数 `max_nei` 权重逻辑因此使 025 contact 的 `k=1`。
- 这是真实的 native contact-weight/initialization pipeline 差异，不是单纯换 seed，也不是 Python V1 objective bug。它是把 025 starts 排除出主控制消融的额外理由。
- 025 native 每个 bundle 都完成固定的 1000 attempted steps，但 bridge 没有 convergence 判据。CPU `hk_fdg` 返回内部保存的 `best_x`；最后一次尝试的 RMS 约 `139.8-143.9` 不等于返回状态。`returncode=0` 只表示固定预算调用完成，不能解释为 native 已收敛。

### 1.3 B 展示/R2 审计与 x0 口径

B audit 已接受。它确认旧/新矩阵解析、track 映射、common mask、raw/normalized matrix 和 R2 计算一致；020 -> C0 的形状差异不是色标或 panel 排列造成。当前显示尺度应使用 B audit 的 pooled RMS，而不是把历史 native/oracle 的 `Rg` 直接套到 029 V1。

历史 `Rg≈2.5` 对照来自 `docs/MEASURED_FACTS.md` 的 “Reference is a noisy target” simple single-`-b1m` oracle split（源脚本 `docs/exploration/pilot_oracle.py`）；它不是 029 V1 的尺度。B audit 当前 pooled RMS 为 reference `1.5529545301005403`、020 candidate `0.3766542939`、C0 bundle2 `0.2942922368`。正比例 scale 不改变 Spearman，因此 pooled RMS/颜色只解释 raw distance/display，不解释 R2 差异。

**chr1 的 B audit 表（既有 exploratory endpoint，不是纠正后主消融结果）：**

| 条件 | 方向 | candidate pooled RMS | matched | cross | contrast |
|---|---|---:|---:|---:|---:|
| 020 selected | swapped | 0.3766542939 | 0.6699297451 | 0.5336508647 | 0.1362788804 |
| 029 C0 bundle1 | direct | 0.3004858440 | 0.5719791227 | 0.5199398043 | 0.0520393184 |
| 029 C0 bundle2 | swapped | 0.2942922368 | 0.5038604957 | 0.4564611074 | 0.0473993883 |
| 029 C0 bundle3 | direct | 0.2895790947 | 0.5810853146 | 0.5232977633 | 0.0577875513 |

**同一既有 mask 的 20-chr R2 描述表：**

| comparison | matched 的宏观 delta | contrast 的宏观 delta | matched 胜出 / 20 | contrast 胜出 / 20 |
|---|---:|---:|---:|---:|
| C0 bundle1 - 020 | -0.01731854 | -0.01363149 | 7 | 8 |
| C0 bundle2 - 020 | -0.01292096 | -0.02043332 | 8 | 8 |
| C0 bundle3 - 020 | -0.01104117 | -0.03131856 | 9 | 9 |

对应宏平均原值为：020 `matched=0.49032573, contrast=0.13295638`；C0 bundle1 `0.47300719, 0.11932488`；bundle2 `0.47740477, 0.11252305`；bundle3 `0.47928457, 0.10163782`。这些数值是 locked R2 的描述性结果，不是新的显著性检验，也不是受控初始化对照。

**正确的 x0 自身 best-swap 前后表：**

| bundle | x0 与 endpoint 方向不同 | endpoint - x0 matched | matched 胜出 | endpoint - x0 contrast | contrast 胜出 |
|---|---:|---:|---:|---:|---:|
| bundle1 | 10 / 20 | +0.27911733 | 20 / 20 | +0.08666839 | 19 / 20 |
| bundle2 | 9 / 20 | +0.26150224 | 20 / 20 | +0.08319607 | 15 / 20 |
| bundle3 | 7 / 20 | +0.23176664 | 20 / 20 | +0.06722237 | 16 / 20 |

这证明每个 025 native x0 到其 C0 endpoint 的自身 best-swap 读数有改善；不能说优化把起点搞坏。它不比较 025 x0 与 020 的 014 source 起点谁更好，不能把原 endpoint-fixed secondary 表当作主 x0 结论。

相同 common mask 的 shape comparison 为：020 vs 022 的两个 ref-aligned matrix rho `0.99430/0.99180`，020 vs C0 bundle2 为 `0.36735/0.62807`。色标归一化坐标变化约 `0.7%-3.7%`，不能造成后者的真实矩阵 shape difference。

## 2. 为什么 029 不能回答 020 受控消融

| 维度 | 020 基线 | 029 C0 探索运行 | 结论 |
|---|---|---|---|
| source | 014 blind random/consensus | 025 native random-assignment bundles | source 与 native preprocessing 都变了 |
| resolution | 5 Mb -> 2 Mb -> 1 Mb | 1 Mb-only | 多分辨率 continuation 被取消 |
| initial p/q | 首层 `p=.75`，后续 raw `q` carry | 1 Mb `p=.75`，没有 020 粗层 q path | nuisance/path 也改变 |
| 1 Mb budget | `maxiter=240, maxfun=750` | `maxiter=480, maxfun=1470` | 不能把 480 写成 020 的复现 |
| source/native seeds | 014 source + base seed 1103/2207 | 025 assignment/native seed pairs | 不是同一 source 条件 |
| selection | 两候选完成全三层后按 final count NLL 选 | variant 内三个 025 bundle 按 count NLL 选 | selection universe 不同 |
| status | 各 stage 达固定预算，非 convergence | C0 达 480，`budget_not_converged` | 都是 finite-budget endpoint |

因此，029 的 C1/C2-map/C2-free/C3 结果最多是“025 native 1 Mb starts 下的独立 exploratory sensitivity”。它们不能作为 020 全流程中相应消融无效的证据。

## 3. 正确的主控制设计

### 3.1 固定 cohort、source 和物理合同

所有 5 个 variant 都使用同一份全量 SNP-free input：

- path：`inputs/P9016.snpfree.pairs.gz`
- SHA256：`f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`
- raw accounting：`1,703,888 = 1,135,454 cis + 568,434 inter`
- 训练：origin-0 full grid，1 Mb final，2,645 loci/copy，5,290 beads，40 tracks，同一 shared nuclear volume
- same-bin count 继续作为 saturated nuisance；`438,774` 是 1 Mb 聚合下的数值，各层都按自身 bin size 重新聚合，不能把该值套用于 5 Mb 或 2 Mb；所有各层的 eligible/zero pair set 都保留在对应 count grid

主设计只允许 020 已冻结的两个 blind source：

| source 条件 | source 路径 | source SHA256 | 基础 seed |
|---|---|---|---:|
| `consensus_joint` | `test_res/014-20260912_153000-s0-genome-wide-fixed/coords/consensus.3dg` | `e76655732deb6b8386b1b77bc76ff45d7dba1384f6931337fee80d8f4aaa8e02` | 1103 |
| `random_joint` | `test_res/014-20260912_153000-s0-genome-wide-fixed/coords/random.3dg` | `9a48d73e1401e18349d11758e679da4c76da0904dbc467979079cb54bcd567d7` | 2207 |

两者都是 blind initialization conditions，不是两个生物学重复，也不是同一分布下可用于估计 biological variance 的 seed samples。主设计不使用 025 starts，不重新跑 native，不把 025 的 `max_nei=0` 管线带入受控消融。

### 3.2 严格继承 020 初始化、目标与调度

source expansion 与层间 warm start 必须直接继承 020 的实现和记录，不另写一个“等价” adapter：

- 每条 source track 按数值 genomic position 排序；full grid 使用 bracket 之间的线性 interpolation；首个 source row 之前和最后一个 source row 之后使用 nearest endpoint fill；不能丢失 3 Mb 评价起点之前的训练 bins（1 Mb 层的 0/1/2 Mb）或 terminal partial bin。
- consensus 用 source `cXXa` 形成共同 `Z`，再按 020 的 1103 seeded smooth half-difference field 生成两 copy；random 保留 014 blind 的两条 copy axes，seed 为 2207。不得读取 phase 或 reference。
- source expansion 后只做一次全 physical beads 的共同 center/uniform scale 到 `max_radius=0.8`，这是所有 variant 共享的起始准备，不是 C2-free 优化过程中的隐式球约束；不能做 per-chromosome/per-copy rescale。C2-free 的 direct Cartesian feasible domain 从此起点进入，但其优化不再附加 sphere clip。
- 新插入、interpolated、endpoint-filled 或 repeated loci 使用 020 的 deterministic Gaussian jitter，`perturbation_scale=0.025*l0`。层间 warm start 使用同一 exact/interpolated/endpoint-fill audit、同一 deterministic jitter seed `3301 + bin_size//1,000,000 + candidate_base_seed`，不重新 center/rescale；在 C0 原流程中，超出 unit ball 按 `1-1e-6` radial clip 规则处理，C1/C3 继承该 C0 处理；C2-map/C2-free 只执行各自登记且通过预检的合法 domain 操作，不能把 C0 clip 偷带进 free。所有这些已作为 033/034 的冻结训练合同；CPU 034 因切 GPU 而停止/正在终止，GPU 迁移待验证。
- 目标仍为 Python V1 continuous joint objective：bounded `K`、observed-endpoint exposure、profiled conditional count likelihood、same-bin nuisance、bond/bend/repulsion/p-prior 组件。实际 repulsion threshold 在 1 Mb 层为 `.7*l0 = 0.04017409936`；5 Mb/2 Mb 层也使用 `.7*l0`，数值随各层 `l0` 更新。旧 config 的 `repulsion_radius_l0=2.0` 只作 stale metadata 留痕，不能当实际 V1 threshold。

严格继承 020 的 optimizer budget：

| layer | accepted maxiter | maxfun | maxls | ftol | gtol | checkpoint |
|---|---:|---:|---:|---:|---:|---:|
| 5 Mb | 300 | 930 | 20 | `1e-10` | `1e-6` | every 10 accepted |
| 2 Mb | 200 | 630 | 20 | `1e-10` | `1e-6` | every 10 accepted |
| 1 Mb | 240 | 750 | 20 | `1e-10` | `1e-6` | every 10 accepted |

首层 `p=.75`；层间 carry raw bounded-logistic `q`。不得把 1 Mb budget 改为 480，也不得以更长 continuation 替代 020 的 stage contract。所有层应在同一 `analysis` conda 环境和单线程 BLAS/OMP 合同下运行。

### 3.3 变体与运行顺序

variant 只允许下表指定的改变，并且该改变在该 variant 的**所有三层**生效：

| variant | 允许的唯一改变 | 其余保持 |
|---|---|---|
| C0 | 020 V1 原始 | observed-endpoint exposure、当前 sphere、bend `.01` |
| C1 | bend `.01 -> 0` | C0 的 kernel、sphere、exposure、bond、repulsion、p rule |
| C2-map | post020 登记的 identity-core `.90` smooth sphere map | C0 的 objective/weights/exposure，physical `R<1` domain |
| C2-free | direct Cartesian `x`，无 hard sphere、无 clip/rescale | C0 的 contact/exposure/weights；它同时改变 parameterization 和 feasible domain |
| C3 | full-grid exposure=`ones` | C0 的 parameterization、bend、kernel、bond、repulsion、p rule |

C2 的比较关系必须按下列三组解释，不能混写：**C2-map vs C0 是球内的 parameterization 变化**；**C2-free vs C2-map 才是域限制相关的比较**（同时仍需记录 free 的 direct-x parameterization）；**C2-free vs C0 是 parameterization + feasible-domain 的联合变化**，不能把它简称为纯“去球”单因素。这个关系适用于每个 resolution layer，不能只在 1 Mb 末端解释。

总设计是 `5 variants x 2 source conditions x 3 layers = 30 stage attempts`，即 10 条完整 trajectories。033 已完成 C0 的两个 source 条件及 6 个 stage attempts，并通过 C0 gate。CPU 034 的 C1/C2-map/C2-free/C3 24 个 stage attempts 因用户要求切 GPU 而停止/正在终止，GPU 迁移待验证；只有迁移后的完整 terminal/hash evidence 锁定后，才允许进入 R2；R2 不参与训练选择。

C1/C2-map/C2-free/C3 都必须从各自相同的两个原始 014 source conditions 开始，独立走完 5 Mb -> 2 Mb -> 1 Mb。不能把 C0 优化好的 coarse state 或 final state 喂给其他 variant；不能在 variant 内用另一个 variant 的 endpoint 作为 warm start。每个 source 内的 paired comparison 是同 source 配对；consensus/random 只是两种初始化条件。

C2-map、C2-free 的每一层都只允许其登记的处理差异，其他条件跨层一律继承同一 020 contract。跨层开始前，必须先对同一 physical interpolation 结果做各自合法的 pack/unpack、finite、track mapping 和 feasible-domain checks；还要专门检查 free 层间是否仍被旧 sphere writer、`sphere_inverse`、radial clip 或 rescale 隐式限制。若存在该隐式限制，必须在开跑前由预检暴露并停止，不能在运行中临时改代码或补回球面限制。C2-free 不得偷偷经过 `sphere_inverse`、球裁剪或 sphere rescale；C2-map 也不得把 free 的 direct-x 当作 map。共同的 center/`max_radius=0.8` 只属于所有 variant 共用的 physical 起始准备，不构成 free 优化阶段的球约束。起点的物理等价性和 pack roundtrip 未通过时，不能开始该 variant。这里是已冻结的迁移后训练 gate；CPU 034 已停止/正在终止，GPU 迁移及其完整验证仍待完成。

### 3.4 C0 gate：先逐层复现，再允许任何其他 variant

C0 gate 不使用 reference 或 R2 放行。每个 source 都要逐层比较归档 020 的对应状态：

1. **source/input gate：** path、source SHA、gate SHA、cohort/input SHA、full-grid inventory、track order、positions、counts 和 zero-count pair set 必须一致；integer/data arrays 要求 exact/hash equality。
2. **initialization gate：** 每层 entry state 与 020 的 source expansion/warm-start 记录逐项比较；interpolation、endpoint fill、jitter count/seed、normalization、max radius、p/q entry 必须一致。
3. **objective gate：** 每个 checkpoint 和 terminal 比较 count、bond、bend、repulsion、p-prior、total `J`、p、q、gradient-L2 与正确的 gradient-infinity diagnostics。默认 numeric gate 在正式执行前冻结；建议对浮点 scalar/array 使用 `atol=1e-12, rtol=1e-12`，hash/整数/配置字段仍要求 exact。任何超差先定位，不直接运行 C1。
4. **solver/stop gate：** `maxiter`、`maxfun`、`maxls`、checkpoint cadence、`success`/termination message、nfev/nit 和 budget reason 必须按 020 contract 记录。达到 `maxiter` 只能写 finite-budget/not-converged，不能写 solver converged；L2 gradient 不能直接与 SciPy infinity-norm `gtol` 比较。
5. **selection gate：** 两个 C0 source 的三层都完成后，严格按 020 的 final 1 Mb label-free count NLL/count-per-record 规则选择 candidate；不读取 reference，不计算 R2 来返选。selection input、tie rule 和 selected id 都要保存。

C0 gate 的目标是尽可能逐数、逐 checkpoint、逐数组复现 020，不是“R2 看起来接近”。033 已按该口径完成：6 个 stage 的初末 3DG 与 148 个 checkpoint 通过 exact gate，selection 仍为 `random_joint`；原 controller 因缺少 journal 以 exit 2 结束，两个局部 recovery attempt 也以 exit 2 结束，最终 terminal-only recovery 以 exit 0 完成，不能把原 job 改写成 exit 0。CPU 034 的其余 24 个 stage 因用户要求切 GPU 而停止/正在终止，GPU 迁移待验证；在新的完整 terminal/hash evidence 锁定前，不进入 R2。

## 4. 已确定与仍待检验的原因

### 已确定

- 029 相对用户要求确实换了 source/native preprocessing，并取消了 020 多分辨率；这是设计偏离。
- 029 C0 的 existing endpoint 与 020 的 R2/shape 不同，且该差异不能由色标解释；B audit 的 020-vs-C0 shape rho 和 pooled RMS 已锁定。
- 029 C0 的三个 025 x0 按自身 best-swap 都有 endpoint improvement；不能写成优化把起点搞坏，也不能据此比较 025 起点与 020-014 起点优劣。
- 030 没有发现 Python V1 objective、grid/count、sphere mapping、serialization、bridge ordering 或 native/backbone regression。
- 014 `max_nei` 自动支持与 025 `max_nei=0` 是已证实的额外 native initialization/contact-weight 差异；native exit 0 和最后一次 RMS 都不是 convergence 证据。
- 033 已完成严格的 C0 020 reproduction gate；selection 规则仍为 `count_nll_per_record`，包含 diag constants、排除 priors，`tie_tolerance_per_record=1e-9`，source 顺序为 consensus 后 random，selected source 为 `random_joint`。

### 仍待检验

- 034 的 C1/C2-map/C2-free/C3 24 个 stage attempts 是否在 GPU 迁移后全部 terminal、坐标和 provenance hash 是否完整锁定；CPU 034 已因用户要求切 GPU 而停止/正在终止，GPU 迁移尚待验证，不能把 CPU 部分结果写成完整 release。
- 在 GPU 迁移后的完整 release 通过 hash gate 后，C1/C2-map/C2-free/C3 各自从两个 014 source、各自完整多分辨率路径产生的 R2 差异；在此之前不能把 029 的 C1/C2 结果外推为 020 消融结论。
- GPU 迁移后的完整 release 之后才做固定-mask R2、legacy common-mask exact parity 和发布 029 `v1_original_random_joint` 数值 parity；R2 尚未运行。
- native/contact-law、exposure、profiled likelihood、prior 和 basin 对 synthetic known-truth recovery 的相对作用。expected-count `C*` 校准、positive/negative null、true exposure/p 只作为 synthetic calibration boundary，不是 real oracle。
- 更长迭代、learning-rate/solver-options、reference/pipeline 差异等后续因素，只有在正确 C0 baseline 建立后才有可解释性。

## 5. 优先级下一步

1. **C0 full 020 reproduction gate（已完成）。** 033 已冻结上述 source/hash、adapter、interpolation/jitter、objective、p/q carry、`.7*l0` repulsion 和三层 budget，并逐层验证 arrays/components/gradients/stop/selection；原始 controller 的 exit 2 与 recovery final exit 0 分开保留。
2. **受控主消融（034 暂停并切换 GPU）。** C0 gate 通过后，CPU 034 已按授权启动 C1/C2-map/C2-free/C3，但因用户要求切 GPU 而停止/正在终止；GPU 迁移待验证。后续仍要求每个 variant 的两个 014 source 完整走三层，所有 candidate 均保留，不能把 CPU 部分结果写成已完成结果。
3. **R2-only evaluation（待 GPU 迁移后的完整 release）。** 只有迁移后的坐标和 provenance hash 通过 release gate 后，才按预先固定的同一 R2/common-mask policy 评价；不使用 R2 返选，不新增 R1/R3。当前 R2 尚未运行。
4. **合成机制校准。** 在不影响 real 选择的前提下，做 `synthetic_expected` count-gradient/KL 第一层，再做已知 truth 的 null/positive、exposure 和 observation-law 对照；true exposure/p 是校准上界，truth-informed start 只由 generation side 构造并锁定，worker 只收 serialized start 与无标签 counts。
5. **可选后续因果诊断。** 旧四臂 source-geometry x schedule pilot 降为 C0 gate 之后的可选诊断，不并入本次 `30 stage attempts` 主消融预算。任何 L2 fragment-consistency/R1/R3 任务另行授权设计，不作为本轮默认步骤。

## 6. 完整审计与统一图

- 033 C0 严格复现报告：[`033 REPORT.zh-CN.md`](../test_res/033-20260914_044246-c0-controlled-reproduction/REPORT.zh-CN.md)
- 033 C0 数值 gate：[`C0_gate.json`](../test_res/033-20260914_044246-c0-controlled-reproduction/C0_gate.json)
- 033 原始 controller/recovery 分离凭证：[`recovery_receipt.json`](../test_res/033-20260914_044246-c0-controlled-reproduction/recovery_receipt.json)
- 034 variant prep 与执行边界：[`053514 variant prep README`](audits/multires-variant-preflight-20260914_053514/README.zh-CN.md)
- 034 父侧发布授权：[`release_authorization.json`](audits/multires-variant-preflight-20260914_053514/release_authorization.json)
- 034 CPU 训练输出（已停止/正在终止，GPU 迁移待验证）：[`034 config.json`](../test_res/034-20260914_053514-multires-variants/config.json)
- 训练审计总报告：[`030 REPORT.zh-CN.md`](../test_res/030-20260914_c0-020-training-audit/REPORT.zh-CN.md)
- 训练审计机器验证：[`030 verification.json`](../test_res/030-20260914_c0-020-training-audit/verification.json)
- native/backbone/neighbor 审计：[`030 native_audit.json`](../test_res/030-20260914_c0-020-training-audit/native_audit.json)
- B display/R2/x0 审计：[`c0-020-display AUDIT.md`](audits/c0-020-display-20260914_021033/AUDIT.md)
- B 审计统一 chr1 距离矩阵：[`unified_chr1_distance_matrices.png`](audits/c0-020-display-20260914_021033/unified_chr1_distance_matrices.png) / [`PDF`](audits/c0-020-display-20260914_021033/unified_chr1_distance_matrices.pdf)
- B 审计 shape comparison：[`chr1_shape_comparison.json`](audits/c0-020-display-20260914_021033/chr1_shape_comparison.json)
- 既有 POST020 R2-only 结果背景：[`POST020_ALLELE_ABLATION_RESULTS.md`](POST020_ALLELE_ABLATION_RESULTS.md)

本文的最终边界是：**033 的 C0 已严格复现并通过 gate；CPU 034 因用户要求切 GPU 而停止/正在终止，GPU 迁移待验证；完成新的 GPU terminal/hash release 后才运行固定口径 R2。029 existing results 保留为 exploratory，R2 尚未运行，不新增 R1/R3。**
