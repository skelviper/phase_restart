# Reconstruction V1 P3 合成校准协议

## 状态与边界

**状态：正式 P3 合成校准，预注册；尚未运行。** 本协议只授权
`test_res/018-20260913_121446-v1-synthetic-calibration` 中的四个固定预算合成
拟合。它不授权真实 P9016 重建、参数搜索、额外 seed、真实 phase/reference/oracle
读取或任何 L2 成功宣称。

上游 V1 观测协议保持冻结，SHA256 为
`cf2ae36fdd4dec6e05d32001a5ef498add7bb62e5908a4373b2316fc380ed8e5`。本文件
定义该协议的 P3 校准执行，不修改上游协议正文。20 条染色体来自一个细胞的 header
范围；四个合成条件不是生物学重复。

训练侧只读取经 SHA256 核验的 SNP-free 文件表头（染色体名、长度和顺序）以及冻结总预算：1,703,888 条，1 Mb 为 diag 438,774、cis-offdiag 696,680、inter 568,434。绝不以真实 phase、reference、014 坐标或真实接触 endpoint 作为 synthetic truth 或 exposure。

## 网格与合成几何

最终层为 1 Mb、from-zero 的 full grid：2,645 个 locus、5,290 个物理珠子、20 条染色体和 40 条轨迹。保留每个 terminal bin。生成后的 1 Mb synthetic counts 按 genomic bin start 严格 coarsen 到 2/5 Mb；coarse 各组总数由实际 coarsening 得到，不能强行替换为真实 P9016 粗层组数。最终 1 Mb 才是 well-specified 条件的精确目标，coarse 层只是 warm-start 近似。

独立 truth 使用 `seed=4101`。先在半径不超过 0.55 的核球内均匀体积采样 4,096 个 center 候选，用 farthest-point 选择 40 个，再随机分配给轨迹。每条轨迹围绕其 center 生成有界随机链：

- `territory_radius = .65 * (n_chr / N_loci)^(1/3)`；
- 相邻步长为 `Uniform(.85, 1.15) * l0`；
- 方向为 `normalize(.2 * previous_direction + .8 * Gaussian)`；
- 超 territory、超核球或与既有非相邻 bead 距离小于 `.25*l0` 的 proposal 被拒绝；
- 每个 locus 有固定 rejection 上限，失败即报错，绝不缩小 chromosome/grid。

truth 不称为 prior 平衡态；每个 condition 报告自身的 bond/repulsion/bend。所有 truth 和 candidate 坐标必须为 finite 且严格位于 unit ball 内。

D 的 truth 在每条染色体的两份 copy 中具有相同的内部链形状，只平移到两个不同的 territory center。它不要求物理位置重合。连续步长只用于避免无关的数值距离 ties；内部形状的 direct/swap tie 是预期的负对照读数。

## 条件与计数生成

共同参数为 `p_gen=.8`、冻结 epsilon 和完整 1 Mb group budget。每个 condition 都使用
完整 E，未观察 bin-pair 仍进入总 rate。diag 按 `e_gen^2` 分配。

| ID | 条件 | truth / kernel / exposure | 计数 | fitter exposure |
| --- | --- | --- | --- | --- |
| A | `noiseless_expected` | different-shape；V1 kernel；lognormal sigma=.4, seed 4201 | 每组 `M_g * rate/sum(rate)` 的 fractional expected mass | known synthetic `e_gen` |
| B | `known_sampling` | 与 A 同 truth/kernel/exposure | conditional multinomial，seed 6101 | known synthetic `e_gen` |
| C | `capture_misspecified` | 同 normal truth；sigma=.8, seed 4203，随机 15% bin 乘 .02 后归一；`epsilon+(1-epsilon)/(1+(d/r0)^6)` | conditional multinomial，seed 6103 | production `sqrt(endpoint_count+10)` |
| D | `same_shape_null` | 每 chr 同内部形状、不同 center；V1 kernel；sigma=.4, seed 4201 | conditional multinomial，seed 6105 | production endpoint exposure |

A 的 fractional 输入明确标为 `synthetic_expected`，只做 conditional cross-entropy 数值校准，不是 fractional Poisson 概率，factorial constants 不用于选择。B/C/D 标为 `synthetic_integer`，保留 exact integer budget 和 aggregate-derived endpoint accounting。

known-exposure coarse 层将 1 Mb `e_gen` 按 coarse bin 求和后重新归一，明确标为近似，不是粗层完全 well-specified。production exposure 在每层从该层 synthetic aggregate endpoint counts 重新计算。

## 拟合隔离与预算

每个 condition 恰好一个 independent initialization：在 5 Mb grid 上以 `seed=5101` 使用相同的有界随机链算法，不从 synthetic truth warm-start，也不使用 014。5→2→1 使用 `pr.reconstruction_init.warm_start_from_layer`，只插值新点并使用其固定扰动规则。p 的第一层为 `.75`；后续层直接 carry 上一层原始 bounded-logistic `q`，不对已饱和的浮点 p 做逆变换。

| 层 | maxiter | maxfun | maxls | ftol | gtol |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 Mb | 120 | 390 | 20 | 1e-10 | 1e-6 |
| 2 Mb | 80 | 270 | 20 | 1e-10 | 1e-6 |
| 1 Mb | 80 | 270 | 20 | 1e-10 | 1e-6 |

每 10 个 accepted iterations 保存 theta、coords 和 JSON checkpoint；日志每 10 iterations 至少包含 total、count/prior component、p、nfev 和 elapsed。必须区分 `converged`、`budget_terminated` 和其他未收敛状态；各层 final total 不得超过同层 initial total（固定浮点容差被记录）。不允许自动加长预算、换先验、改 kernel 或追加 seed。

worker 只接收该 condition 的无标签 layer counts/exposure 和 independent init；没有 truth path 参数。candidate 坐标先写到 `coords/` 并完成 SHA256 后，离线 evaluator 才读取隔离的 `eval_truth/`。

## 预注册检查门与离线读数

数值检查门：全 cohort/full grid/三组预算守恒、finite/strict ball、解析 objective 可计算、各层 nonincrease。A 还要求 1 Mb true data cross-entropy 不劣于 fixed collapse、random field 和 fragment-swap alternatives；对每条染色体的全局 copy swap，data 项差异不得超过 `1e-8` per record。这些都是目标函数/实现校准检查门，不是恢复结论。

候选完成 hash 后，离线评估为每个 condition 报告：

- 每条 chromosome 的 R2 四轨共同 finite mask、direct/swap、geometry contrast；
- 从 full-grid zero 起点开始的完整 20 Mb fragment R3，含覆盖、local/global tie 与不可判；
- count likelihood、bond/repulsion/bend/p prior、truth/initial/collapse/random/fragment-swap 对照；
- D 的内部形状差异和 `truth_minus_candidate` count noise gain，仅作诊断；
- C 相对 B 的联合 kernel+exposure misspecification sensitivity，不能归因于单一因素。

不生成模拟 record-copy labels，因此 R1 标为不适用。R3 的低 walls、copy difference 大小或 noise gain 不能单独宣称 L2。D 的 R2/R3 tie 或不可判既不算失败也不算成功。任何阴性 recoverability 结果必须如实报告；父代理在收到所有终态后决定真实重构可以作何种解释。
