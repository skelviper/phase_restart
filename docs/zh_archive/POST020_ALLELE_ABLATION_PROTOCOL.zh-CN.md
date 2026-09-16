# POST020 等位基因消融协议

**协议版本：** `1.0`  
**状态：** `frozen_before_fit`  
**范围：** P9016 SNP-free 等位基因信号消融、合成校准和隔离的真实数据 R2 评估  
**权威机器文件：** [`POST020_ALLELE_ABLATION_PROTOCOL.json`](POST020_ALLELE_ABLATION_PROTOCOL.json)

本文件把 020 之后的后续实验冻结为可执行协议。它不是结果报告，也不是拟合授权本身：当前没有新的真实拟合、合成拟合或真实 R2 评估；本次冻结工作没有读取真实参考结构、真实 phase 或新的坐标。025 只到 prepare-only，不能写成 native 拟合已完成。

## 1. 冻结与发布 gate

任何 native prepare 或 joint fit 之前，必须在实际 run 目录写入：

```text
test_res/{NNN}-{YYYYMMDD_HHMMSS}-post020-allele-ablation-{real|synthetic}/freeze/POST020_ALLELE_ABLATION_FREEZE.json
```

native prepare 的启动前置是本机器 JSON、输入/source 哈希，以及该阶段已生成的 prepared graph 哈希（若已到 x0 阶段则记录 locked x0 哈希）。protocol MD/plan 的真实 mtime/hash 在文档可用时追加到只追加 registry；若它们在 native prepare 之后才完成，必须记录真实时间和哈希，不能回填成早于 native 的版本，也不阻塞 native 启动。任何后续联合拟合/评估仍须保留完整的实际哈希/mtime 记录。

freeze JSON 与同一记录必须追加到：

```text
test_res/POST020_ALLELE_ABLATION_ARTIFACT_REGISTRY.jsonl
```

registry 是 append-only：旧记录不覆盖；实现 bug 的重试追加新记录并同时保留原/新 source hash。run ID 尚未分配时，只使用上述 path pattern，不猜已有 fixture/run 文件。协议文档 hash 使用：

```bash
sha256sum docs/PLAN-post020-allele-signal.md \
  docs/POST020_ALLELE_ABLATION_PROTOCOL.md \
  docs/POST020_ALLELE_ABLATION_PROTOCOL.json
```

当前 machine JSON 已先严格解析并标为 `frozen_before_fit`；本 MD 的补写不改变任何方法、seed、预算或执行状态。不能以文件 mtime 代替 freeze，也不能把后补的 MD/plan 说成 pre-fit。

## 2. 历史动机与实际进展

原计划的动机保留：020 相比 Softall 的整体 geometry 相似度较低，但 allele contrast 较高；需要区分 bend、sphere 参数化/可行域和 exposure 的影响，优先验证 allele-specific signal，而不是只追求与 reference 的整体 Spearman。当前新增 protocol 将每项因素单独消融，并用独立 synthetic negative/positive fixture 识别数值失败模式。

### 2.1 022 延续运行

022 是计划 B 的固定 warm continuation：同一 020 `random_joint` 1 Mb endpoint 增加 240 个 accepted steps，累计到 480；不是独立初始化重复。

| 数量 | 020 原始 | 022 延续运行 |
|---|---:|---:|
| total J | 9.61001577125 | 9.60361804197 |
| count NLL | 9.59358593129 | 9.58710105238 |
| accepted steps | 240 | 480 cumulative |
| status | budget-limited | `budget_not_converged` |

FDG1000full 提案原 J 未通过 label-free 规则，保留为 `rejected_no_label_free_improvement`，不把 rejected endpoint 当作新成功条件。

### 2.2 023 固定端点诊断

结果见 [`test_res/023-20260913_212541-post020-allele-signal-diagnostics/README.md`](../test_res/023-20260913_212541-post020-allele-signal-diagnostics/README.md)。023 没有启动新 fit、没有读取 reference/phase；最终 bash75 和 validator 均 exit 0。

| 诊断项 | 020 random 1 Mb | 022 延续 480 |
|---|---:|---:|
| max radius | 0.65161710 | 0.68574970 |
| beads `>=0.90` | 0 | 0 |
| beads `>=0.95` | 0 | 0 |
| beads `>=0.99` | 0 | 0 |
| minimum sphere radial attenuation | 0.43646488 | 0.38556995 |
| minimum sphere tangential attenuation | 0.75854806 | 0.72783744 |
| weighted bend y-gradient L2 | 0.01216713 | 0.01206299 |
| count y-gradient L2 | 0.03243707 | 0.03234493 |
| weighted bend/count y-gradient ratio | about 0.375 | about 0.373 |
| total gradient L2 | 0.00620367 | 0.00514436 |
| last-20 accepted total-J decrease | 0.00091717 | 0.0003898645 |

没有 bead 接近 0.90/0.95/0.99，只能排除这些 endpoint 的 active hard-wall occupancy；不能说 sphere 参数化无优化影响。radial/tangential attenuation 仍改变梯度传递，bend=.01 也有实际梯度作用。

实际 repulsion hinge threshold 是 `.7*l0 = 0.04017409936`。020 旧 config 的 `repulsion_radius_l0=2.0` 是元数据字段，native `d_r2.0` 是另一单位；三者不能混用。023 保存 source 与当前 `pr` 有差异时，历史重现以保存代码的数值结果为准。020 六层和 022 均为固定预算停止，不称充分收敛；本 protocol 保留 480 accepted 的固定有限预算，不自动加量。

### 2.3 024 派生的 R2 证据与更正

结果见 [`test_res/024-20260913_133109-r2-allele-signal-derived/README.md`](../test_res/024-20260913_133109-r2-allele-signal-derived/README.md)。024 只从既有四 rho 派生 R2，没有新 fit，也没有打开真实 reference、phase 或 contacts。

- 020 original 相对 Softall：contrast `+0.044957`，95% CI `[-0.007420,+0.096044]` 跨 0；matched 变化 `-0.055800`。
- 020 original 与 continuation 都是 20/20 finite；两者均为 `both-positive=13/20`、`one-negative=7/20`；minmargin 分别为 `.049677/.047710`。
- continuation 相对 020 的 contrast 为 `-0.000222`，基本持平。
- 父侧发现 fixed-reference 列锚定 bug：swapped 时应交换 candidate 行，不应交换 reference mat/pat 列。024 v2 现已按此修复。已有 matched/cross/contrast/minmargin/counts 不受影响；旧 v1 的 absolute per-reference margin 或 contribution share 仍不引用，除非在 v2 下重新生成。

024 的 best-overall-swap 选择偏置必须保留在解释中；它不能被包装成无偏的 recovery score。

### 2.4 025 仅 prepare 的预检

实际目录为 [`test_res/025-20260913_135100-random-native-fullgrid-preflight/`](../test_res/025-20260913_135100-random-native-fullgrid-preflight/)。父侧确认 3 native bundles、count conservation 和 63 graph-swap checks 已通过，但没有调用 `hk_fdg`。这是输入图和 canonicalization 的 prepare-only 证据，不是 native fit、coordinate endpoint 或 allele recovery 结果。

## 3. 冻结的真实队列与观测单位

- 生物样本：P9016，一个生物细胞；生物学重复层级 = 1。
- 训练输入：`inputs/P9016.snpfree.pairs.gz`。
- 训练 SHA256：`f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`。
- 训练侧绝不读取 `phase0`、`phase1`、`phase_prob00..11`、`p_gen`、参考 3DG 或 014 坐标。

### 3.1 原始记录核算

```text
raw total              = 1,703,888
cis total              = 1,135,454
inter                   =   568,434
same-bin diagonal       =   438,774
cis off-diagonal        =   696,680
structural              = 1,265,114
1,703,888 = 438,774 + 696,680 + 568,434
```

似然的观测单位是聚合后的无序基因组 bin-pair 计数 `C_ij`。同一配对上的重复原始记录会求和。same-bin 记录仍作为每个 bin 独立的饱和 nuisance 层保留；它们不是几何项。

### 3.2 全网格约定

- 分辨率：仅 1 Mb；origin 0。
- 对染色体长度 `L`，使用 `ceil(L/B)` 个 bin，并保留末端不完整 bin，`bin = position // B`。
- 20 条染色体、40 条轨迹、每个拷贝 2,645 个位点、5,290 个物理珠子。
- 完整无序 eligible 集合，包含零计数配对：`3,496,690` 对。
- 真实零计数 eligible 配对：`3,009,436` 对。
- 结构原始分组：cis-offdiag `696,680`，inter `568,434`。
- 同区间 nuisance bin：`2,645` 个。
- 训练中没有 legacy 3 Mb 偏移；不会为了匹配旧坐标清单而丢弃记录。

所有符合条件的配对以及所有零计数配对都保留在速率归一化项中。实现必须断言原始记录守恒，并保留所有归一化项。

### 3.3 固定尺度与计数模型

```text
l0 = (2*N_loci)^(-1/3) = 0.05739157051286577
r0 = 2*l0            = 0.11478314102573153
epsilon              = 1e-6
repulsion threshold  = .7*l0 = 0.04017409936
```

有限接触核为：

```text
K(d) = epsilon + (1-epsilon)*(1 + d^2/r0^2)^(-2)
```

对于 cis 非对角配对：

```text
rate_ij = e_i*e_j * [
    p/2       * (K(X_i,X_j) + K(Y_i,Y_j))
  + (1-p)/2  * (K(X_i,Y_j) + K(Y_i,X_j))
]
```

对于 inter 配对：

```text
rate_ij = e_i*e_j/4 * [
    K(X_i,X_j) + K(X_i,Y_j) + K(Y_i,X_j) + K(Y_i,Y_j)
]
```

`p` 是全基因组共享的 cis nuisance 参数，初值为 `.75`，参数化方式为：

```text
p = p_floor + (1 - 2*p_floor)*logistic(q)
p_floor = 1e-4
p_prior_strength = 1e-4
q_init = q_from_p(.75)
```

候选相关的条件计数 NLL 在每个结构分组中对固定总量建 profile。对角层是独立的、按 bin 饱和的 nuisance 项。零计数配对仍保留在分组速率和中。权重固定为：count=1、bond=1、repulsion=1、p-prior=1，除 C1 外 bend=.01。

## 4. 五个冻结的真实变体

真实实验是 `5 variants x 3 paired starts = 15` 个仅 1 Mb 的 joint fit。C0 是使用 V1 模型的新 1 Mb 起点，不是重放 020 的 5 Mb -> 2 Mb -> 1 Mb 路径。

| arm | 唯一登记的改动 | 参数化 / exposure | 状态 |
|---|---|---|---|
| C0 | 相对于新的 V1 1 Mb 模型无改动 | 当前 sphere / observed-endpoint | planned, not run |
| C1 | bend `.01 -> 0`；保留 bond | same sphere / observed-endpoint | planned, not run |
| C2-map | 只将 map 替换为 identity-core `.90` smooth ball map | physical `R<1` / observed-endpoint | planned, not run |
| C2-free | 直接使用 physical x，仅有限值，不设 hard ball，不 clip/rescale | direct Cartesian / observed-endpoint | planned, not run |
| C3 | exposure 只设为 full-grid ones；保留 endpoint audit 和 normalizers | same sphere / ones | planned, not run |

### 4.1 Exposure

对于 C0/C1/C2-map/C2-free，使用固定的观测端点 exposure：

```text
e_i = sqrt(endpoint_count_i + 10)
      / mean_full_grid(sqrt(endpoint_count + 10))
```

它使用所有原始端点，包括 same-bin 记录，并且不读取 phase 或 reference 计算。它不会被优化。C3 在每个 full-grid locus 使用 `e_i=1`，同时保留 endpoint audit 和每个 normalizer。

### 4.2 C2-map 与 C2-free

对于 C2-map，令 `r=||y||`：

```text
r <= .90:  s = r
r >  .90:  t = (r-.90)/.10
           s = .90 + .10*t/sqrt(1+t^2)
           x = s*y/r                 (r=0 -> x=0)
```

因此 `R=||x||=s<1`；没有 clip 或 rescale。C2-free 直接使用 physical `x` 变量，只对有限值目标进行评估，不设 hard sphere，不 clip，也不 rescale。C0 -> C2-free 的 contrast 同时改变参数化和可行域，因此不能归因于单一边界原因。C2-map 有助于分离 map 敏感性，但两个 contrast 都不是因果证明。

不运行 `bend=0 + free` 组合。不允许搜索 `r0`、先验、exposure 或由 reference 驱动的参数。020 consensus 保留为历史对照，不重新运行。

## 5. 冻结的原生初始化与物理 x0

### 5.1 Bundle 与分配

| bundle | assignment seed | native seed |
|---|---:|---:|
| bundle1 | 250101 | 250201 |
| bundle2 | 250102 | 250202 |
| bundle3 | 250103 | 250203 |

分配使用 `genome.random_assignment`：cis 记录的两个端点接收同一随机拷贝；inter 端点接收独立随机拷贝。不读取 phase、`p_gen`，也不替换为 014 seed。

### 5.2 原生 bridge

每个 seed 恰好使用一次显式的 full-grid NULL-source 原生 bridge：

```text
n_iter=1000
CPU=1
source=NULL
target_radius=10
native bend=0
```

原生默认值其余不变。保留末端和零接触珠子。C1 只改变 joint model 的 bend；不改变 native bend。

### 5.3 规范化与 x0 锁定

对每条染色体，使用固定编码对整数 AA/BB cis 非对角边签名进行数值排序。选取较小的字节 key 作为规范拷贝 A。如果主 key 平局，则使用按其他染色体、自身局部 bin、其他局部 bin 和计数排序的 incidence 签名，并忽略 partner-copy 标签。如果仍然平局，则标记 preflight failure，保留 seed，不静默交换或替换。

每个 bundle 有 20 个单染色体加一个全染色体规范输入字节不变性审计，每个 bundle 21 个，总计 63 个。这些只是 graph/blob 检查；本协议不称原生引擎本身具有 equivariant 性质。

原生阶段一次性对全部 40 条轨迹完成。然后对全部轨迹进行一次共同居中和统一缩放，使最大半径为 `.8`；保存规范标签。在任何 joint model 之前，对三个物理 x0 文件哈希。一个 bundle 的五个变体严格共享同一个物理 x0；它们的 latent inverse 表示可以不同。不得每个变体调用一次 native。

## 6. 真实优化预算与选择

每个真实 fit 使用 SciPy L-BFGS-B，参数如下：

```text
maxiter = 480 accepted iterations
maxfun  = 1470
maxls   = 20
ftol    = 1e-10
gtol    = 1e-6
checkpoint = every 20 accepted iterations
```

记录实际 `nfev`、`njev`、每次目标评估、line-search probe、组件值、checkpoint、source hash、input hash、x0 hash 和终止原因。不得使用 R2 进行提前停止、预算扩展或候选选择。固定预算明确容易导致未充分收敛。

仅在每个 variant 内，按最终 `count_nll_normalized` 选择代表结果；`<=1e-12` 的平局按 bundle ID 打破。全部 15 个输出都保留在评估集合中。不得跨 exposure/prior 按 J 排名并称某一 variant 在全局最佳。

统计失败、数值失败、被拒绝的端点和 `not_converged` 端点都保留在 registry 和最终报告中。不得因为结果不利而重抽 seed 或重跑。实现 bug 只能在同一冻结数据、seed、预算和 variant 上修复后重试；同时记录两个 source hash，并称为 retry，绝不能称为独立重复。

## 7. 合成校准（已冻结，尚未运行）

合成校准用于数值和失败模式检查。它不声称真实数据的原生初始化已恢复，也不声称与生物学匹配。

### 7.1 测试样例与 seed

所有测试样例使用同一完整表头、1 Mb 网格、20 条染色体、40 条轨迹和 5,290 个点。四个真值对象相互独立，不得互换：

| 测试样例 | 类别 | 真值 seed | 生成 seed | exposure seed | 初始形状 seed | 生成 exposure |
|---|---|---:|---:|---:|---:|---| 
| N1 | 内部形状相同、空间分离的负例 | 260101 | 260201 | 260301 | 260401 | ones |
| N2 | 内部形状相同、空间分离的负例 | 260102 | 260202 | 260302 | 260402 | lognormal sigma=.4, mean=1, no dropout |
| P1 | 不同形状的正例 | 260103 | 260203 | 260303 | 260403 | ones |
| P2 | 不同形状的正例 | 260104 | 260204 | 260304 | 260404 | lognormal sigma=.4, mean=1, no dropout |

对于 N2/P2，遵循实际 helper 语义：用 `sigma=0.4` 按 `np.exp(default_rng(seed).normal(0, sigma, N))` 生成，然后除以实际 full-grid mean。不要替换为总体 log mean。N1/N2 与 P1/P2 不是成对的仅 exposure 实验；不能把跨测试样例差异解释成纯 exposure 效应。

### 7.2 生成与 worker 隔离

生成使用 V1 kernel 和 `p_gen=.8`。每个测试样例使用固定的整数 multinomial 总量：

```text
diagonal same-bin = 438,774
cis off-diagonal  = 696,680
inter             = 568,434
raw total         = 1,703,888
```

所有符合条件的配对都保留。记录每个测试样例的实际零计数；绝不能替换为真实数据的 `3,009,436`。

对于 C0/C1/C2-map/C2-free，从合成观测端点重新计算 exposure。C3 使用 ones。生成 exposure 绝不传给 worker。初始形状各自调用一次 `generate_truth(startseed,false)`，然后共享一次最大半径 `.8` 的归一化；它们不使用或扰动真值。一个测试样例内的五个 arm 共享同一 x0，并初始化 `p=.75`。

在拟合前准备并哈希所有真值/观测数据产物。拟合 worker 只接收观测数据。每个拟合的坐标端点都要哈希并锁定，之后隔离 evaluator 才能读取真值。

合成 fit 共 `4 x 5 = 20` 个。使用：

```text
maxiter = 80 accepted iterations
maxfun  = 270
maxls   = 20
ftol    = 1e-10
gtol    = 1e-6
checkpoint = every 20 accepted iterations
```

### 7.3 合成 gate 与指标

对于 N，每条染色体的两份真值距离矩阵必须在 rtol `1e-10` 下相同，且真值质心间距必须大于 `>1e-6`。对于 P，归一化真值距离矩阵必须相差 `>1e-6`。无效测试样例保留为 `invalid`；不替换其 seed。

有限值、梯度、网格、预算、代码和真值坐标哈希检查都是实现 gate。真值 R2 增加不是发布 gate，真值不能选择参数。N 的 contrast 接近零是同形状构造的代数性质，不是方法无法虚构拆分的证明。

对于 N，将每条染色体的共同真值定义为两份真值 D 矩阵的均值。对每条染色体，候选和真值使用同一个有限无序非对角掩码。候选尺度 `sC` 与真值尺度 `sT` 对两份拷贝各自共享：

```text
sC = sqrt(mean of squared distances from candidate copies A and B over the common mask)
sT = sqrt(mean of squared distances from truth copies A and B over the common mask)
```

对于 N，计算 `sT` 时对两份真值拷贝都使用共同真值 D 矩阵。对于 P，使用其自身的真值拷贝矩阵。报告每份候选拷贝的形状误差 `RMS(Dc_copy/sC - Dt_copy/sT)`，并给出每条染色体的均值和最大值。这会保留相对拷贝尺度；不得分别重缩放拷贝。另报告负例专用诊断 `RMS((DcA-DcB)/sC)`，使用同一掩码和共享的 `sC`；它不是正向分数或选择规则。对于 P，报告自身真值的四个 rho、一个整体方向、contrast、两个固定参考 margin 和形状误差。

## 8. 隔离的真实 R2-only 评估

这是已规划但尚未运行的步骤。只有在全部 15 个端点、variant 内选择、坐标和 source hash 都锁定后，评估器才能读取：

```text
data/P9016.1m.3dg.gz
SHA256 1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29
```

不读取真实 phase pairs。本轮不加入 R1/R3；评估范围仅为 R2。既有对照固定为：Softall seed124101、020 random、022 continuation、FDGfull rejected、random014 和 consensus014。它们是描述性对照，不是调参目标。

### 8.1 共同掩码

使用位置 `range(3Mb,L,1Mb)`，并对每条染色体使用一个共同的有限无序非对角掩码。预期总量为：

```text
common mask pairs       = 157,529
total non-diagonal pairs = 176,201
```

必须检查实际覆盖。常数指标输入、任何非有限输入，或某条染色体少于 20 个共同配对，都会使该指标记为 n/a。如果某 arm 失败或没有有效坐标，将其保留为 n/a；不得静默移除或改变分母。这不授权任何局部拷贝交换。

### 8.2 四个 rho 与固定参考方向

始终保留原始值：

```text
rho_A_mat, rho_A_pat, rho_B_mat, rho_B_pat
```

参考列固定为 `ref1=mat`、`ref2=pat`。允许的两种整条染色体映射为：

```text
direct: a=A_mat, b=A_pat, c=B_mat, d=B_pat
cross/swapped: a=B_mat, b=B_pat, c=A_mat, d=A_pat
```

对于所选方向：

```text
matched       = (a+d)/2
cross_matched = (b+c)/2
contrast      = matched - cross_matched
margin_ref1   = a-b
margin_ref2   = d-c
minmargin     = min(margin_ref1, margin_ref2)
```

每条染色体比较一次原始 direct 和 cross 数值，然后取较大者作为一次整体 best swap。这会保留 best-swap 的正向选择偏置，解释中必须明确说明。不做局部拷贝修复。

如果 `abs(direct-cross) <= 1e-12`，标记为 `unresolved_tie`，保留原始四个 rho，将 contrast 设为 0，不计入 `both-positive`，并将固定参考 margins/minmargin 设为 n/a。`both-positive` 和 `one-negative` 是受 best-swap 选择影响的描述性模式。对于非平局，`both-negative=0` 由 contrast 非负代数推出；它不是成功证据。

### 8.3 比较、seed 与 bootstrap

主要配对比较为：

```text
C1-C0
C2-map-C0
C2-free-C0
C2-free-C2-map
C3-C0
```

然后与固定历史对照作描述性比较。先在同一 20 条染色体上配对每个 seed，再对每条染色体的三个 seed 的效应取均值。seed 是优化重复，不是生物学重复；不得将 60 行染色体合并为 n=60，也不得只报告最佳 seed。

主要的三 seed 汇总要求该比较中的所有计划 seed 都有效。如果任何 seed 失败或没有有效坐标，则主三 seed 汇总报告为 n/a，同时保留 `planned_seed_count=3` 和 `valid_seed_count`；只有在明确标注为 available-seed summary 时，才可另列可用 seed 汇总。不得静默变成双 seed 主结果。

所有指标和比较使用同一个 seed-9301 的 `10000 x 20` bootstrap 索引矩阵。区间描述细胞内的结构/技术变异，不是生物学重复，也不是 p-value。

## 9. 资源与产物布局

观测到的只读快照：192 个逻辑 CPU，MemAvailable 约 335 GB，快照时没有活动 fit。这不是加速证据。调度上限为：

```text
native preparation: at most 2 workers
synthetic fits:     at most 4 workers
real fits:          at most 6 workers
global single-thread process limit: 6
BLAS=1, OMP=1
```

启动前记录当前 MemAvailable 和单 worker RSS。如果 MemAvailable 低于 32 GiB，不得启动新 worker；等待正在运行的 worker，也不得改变优化预算。根据 023，一次真实 480-step fit 历史上约需 45--47 分钟；15 个真实 fit 总计约 11.4 CPU 小时，墙钟时间为数小时。不承诺墙钟时间线性加速。

未来运行使用新目录：

```text
test_res/{NNN}-{YYYYMMDD_HHMMSS}-post020-allele-ablation-real/
test_res/{NNN}-{YYYYMMDD_HHMMSS}-post020-allele-ablation-synthetic/
```

每次运行必须保留 `README.md`、`config.json`、`freeze/`、`logs/`、坐标、图、source/input 清单、选择记录、终止审计和隔离评估。本文件工作不修改任何既有 source 或实验目录。

发布顺序固定为：

```text
machine JSON/input/source/prepared-graph hash
-> run-specific freeze JSON and registry append
-> fixture/graph/native prepare
-> physical x0 hash lock
-> synthetic fits and calibration
-> real 15 fits
-> endpoint/selection/hash lock
-> isolated real R2 evaluation
```

如果 protocol MD/plan 在 native preparation 之后完成，则追加其真实 mtime/hash，不回填日期，也不阻塞 native。旧 machine-JSON 快照 SHA256 `6fa2f833ac4a915614b04c85577d1b7ef5528105553c4c8be69c6327af2eec92` 保留为 `prefit_superseded_snapshot`；当前 JSON hash 在这些更正后重新计算。

## 10. 结论边界

本次冻结只确立：下一步实验、输入、参数化、seed、预算和读数规则都是明确且机器可读的。它不声称合成校准已完成、真实消融已完成、原生 equivariance 已成立、数值收敛充分、已恢复等位基因，或已恢复 L2 的整条染色体拷贝身份。

历史 L1 证据、尚未解决的 L2 问题以及对宽泛 L3 主张的撤回仍然分开。一个正 contrast、两条分离的线、更低的训练损失、零退出码或单个高相关，都不能单独报告为等位基因已恢复。
