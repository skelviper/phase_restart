# 049 max-contact-unified-multiscale 冻结计划

本文件在实现之前冻结。实施过程中不改本文件的矩阵、定义、预算与选择规则；任何偏离必须在
`README.md` / `results/` 中作为偏差记录，不得静默修改。

## 1. 运行目录与冻结输入

- 运行目录：`test_res/049-20260915T162917Z-max-contact-unified-multiscale/`
  （run id `049-20260915T162917Z-max-contact-unified-multiscale`，UTC 冻结时间 2026-09-15T16:29:17Z）。
- 唯一真实细胞：P9016，`20 chr / 40 copy tracks`，全部 `1,703,888` 条 records。
- SNP-free 输入：`inputs/P9016.snpfree.pairs.gz`，
  SHA256 `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`（本目录实测一致）。
  加载器仍拒绝任何 `#columns` 含 phase 字段的文件；本目录不读 raw phase 列。
- 1Mb 聚合（只读复用 045 训练侧序列化，不重算）：
  `test_res/045-20260915T073310Z-shared-capture-round/inputs/real_1000000_aggregate.npz`
  （SHA256 `80984d804f8ae0552f6bab34a77a03e778073ac87f3b3ec6f04f96de74137420`）。
  实测 full grid：`n_loci=2645`，`n_pairs=3,496,690`（全部 strict upper triangle，含零计数 pair），
  `cis_pair.sum()=184,016`（同染色体非对角 pair），
  `Σcounts(cis)=696,680`，`Σcounts(inter)=568,434`，`Σcounts=1,265,114`，
  `diag_counts` 合计 `438,774`（独立饱和 nuisance，不进入结构归一化），
  `Nraw=1,703,888` 为目标函数分母。`Noff=696,680+568,434=1,265,114`。
- 训练侧不读 reference，不读 phase。`data/P9016.1m.3dg.gz`（SHA
  `1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29`）只在全部候选/initial/null
  写出并完成 hash 封存后才由评价侧打开。
- 生物重复 `n=1`；20 chr 是同一细胞内的关联技术/结构测量，bootstrap 只描述细胞内变异。

## 2. 两个盲起点（consensus / random）的确切来源

两个 1Mb 盲起点复用 045 训练侧已冻结的 zero-optimization initial controls，本目录逐元素读取，
**不由 reference 构建，也不使用任何已被评价过的结构拼初值**：

- consensus：`045-.../coords/initial/real_consensus_1Mb.npz`
- random：`045-.../coords/initial/real_random_1Mb.npz`

血统核对（写入门禁脚本，结果落 `gates/`）：两文件的 `metadata_json` 均标记
`no_optimization: true`、`source: "014 approved blind root plus zero-optimization prolongation"`，
`source_root: test_res/014-20260912_153000-s0-genome-wide-fixed`，
`gate_stage/gate_tag` 为 `blind`，`source_sha256` 分别为
`e76655732deb6b8386b1b77bc76ff45d7dba1384f6931337fee80d8f4aaa8e02`（consensus）
与 `9a48d73e1401e18349d11758e679da4c76da0904dbc467979079cb54bcd567d7`（random），
即 014 获准的无标签盲 source。perturbation 规则为 `new_or_repeated_coordinates_only`,
seed consensus 5509 / random 4405（1Mb 层），`p_init=0.75`。

同一 source 的全部 6 个 fit（3 loss × 2 solver）共享**逐元素相同**的 `raw_y` 与
`q=q_from_p(0.75)`；门禁对每个 source 计算 `raw_y` 数组 SHA256 并要求 6 条 fit 的 iter0
`raw_y_sha256` 完全一致。`p/q` 每臂独立优化；起点 p 由 `q_init` 唯一决定。

## 3. 三个 loss 的精确定义

对每个非对角 pair `(i,j)`，令 `e_i e_j` 为固定 production exposure 乘积（V0-fixed-production-e，
来自 1Mb aggregate 的 `exposure` 向量，全场算术平均 1），四个 copy 状态
`s ∈ {AA,AB,BA,BB}`：

```
K_s    = 1e-6 + (1-1e-6) * (1 + d_s^2 / r0^2)^(-2),   r0 = 2 * (2*n_loci)^(-1/3) = 2*l0
w_s    = cis  : [0.5p, 0.5(1-p), 0.5(1-p), 0.5p]
         inter: [0.25, 0.25, 0.25, 0.25]
t_s    = w_s * K_s
r_sum  = e_i e_j * Σ_s t_s
r_max  = e_i e_j * max_s t_s
Zsum   = Σ_{all offdiag pairs, 含零计数} r_sum ;  Zmax = 同样求和下的 r_max
C_ij   = counts（整数）
```

- **A `marginal_G`**（= 045 原 G）：`Noff*log(Zsum) - Σ C log r_sum`
- **B `hard_observed`**（用户主方案）：`Noff*log(Zsum) - Σ C log r_max`
- **C `max_rate`**（辅助语义对照）：`Noff*log(Zmax) - Σ C log r_max`

三者相加：
- 同一 `diag_nll`（samebin 饱和 nuisance，`438,774` 条，`Σ(λ - λlogλ)`，integer 模式再加
  `Σlgamma(λ+1)`）；
- 同一 candidate-independent 常数 `K0 = 696680*log(696680/1265114) + 568434*log(568434/1265114)`
  （与 045 原 G 的 `group-mass KL` 展开项完全一致，A 因而与原 G 数值等价）；
- 除以 `Nraw`；
- 再加原完整正则 `weights = (count=1, bond=1, repulsion=1, bend=0.01, p_prior=1)`，
  `p_prior = -1e-4*log(p(1-p))`（内部已含 1e-4，不重复缩放）。

梯度：value 与 gradient 使用同一聚合选择。A/B 的 normalizer 用 `Zsum`（含 max 观测项时
normalizer 仍是 sum）；C 的 `Zmax` 梯度必须取 max 的分支导数。观测项梯度的
`Σ C * d log r_obs` 用同一 `r_obs` 的聚合选择。`p` 的导数包含 `w_s(p)` 的导数
（cis：`dw/dp = [.5,-.5,-.5,.5]`；inter：0），hard derivative 因而含权重对 p 的导数。

**精确并列 ties**：`max_c` 定义在 4 个精确相等的 `t_s` 上取平均 subgradient
（`mark = (t_s == m)`，权重 `mark / mark.sum()`），不使用 first-argmax，保持 A/B 规范对称。
有限差分门禁主体避开非光滑 tie 点，另有独立 tie/gauge 检查。

零接触 pair 永不删除，也不使用标签做 E-step。

## 4. 两个 solver

- `raw`：原 `raw_y/q` 参数化，`sphere_forward(y)=y/sqrt(1+|y|^2)`，`q` 由 bounded-logistic
  映射到 `p`。梯度即 `dF/d(raw_y)`。
- `ms`（multiscale）：固定、可逆、覆盖全部自由度的链内多尺度线性预条件。
  对每条 chr 的每个 copy 的每个 xyz 分量，沿该 chr 全部 bins 的 raw_y 施加
  `P = I + (I + 5^2 L)^(-1) + (I + 20^2 L)^(-1)`，`L` 为该 1Mb 链 path graph 的
  Neumann 离散 Laplacian，长度为该 chr 的全部 bins；用 `scipy.fft.dct/idct(type=2,
  norm="ortho")` 对角化，`lambda_k = 4*sin^2(pi*k/(2n))`，
  `p_k = 1 + 1/(1+25*lambda_k) + 1/(1+400*lambda_k)`，全部 `>0`（`p_0=3`）。
  优化变量 `z = P^{-1} y`（`q` 不变），objective `f(Pz, q)`，gradient_z = `P gradient_y`
  （`P` 对称）。所有频率始终存在；不冻结内部形状，不按 intra/inter 拆分优化，
  不是"先刚体后 bead"。初始化 roundtrip 门禁要求 `< 1e-12`。
  停止判定必须用**原 raw_y/q canonical 梯度**（门禁用非 identity `P` 验证 runner 报告的
  canonical 梯度与真正 raw_y/q 梯度一致），不得用 precond 梯度误称收敛。
  checkpoint/`result.y` 必须是 raw_y，且与 sphere 物理坐标映射一致（门禁验证）。
- 两 solver 共用同一 objective 数值核心、同一 `ftol=0`、同一 `maxls=20`、
  `canonical_gtol=1e-6`、同一 `maxiter=fg_cap+1`、同一 exact FG 记账
  （每个 line-search FG 都计数，含被拒绝的试探点）。

## 5. 冻结 12-fit 矩阵与预算

`3 loss (A/B/C) × 2 solver (raw/ms) × 2 blind source (consensus/random) = 12 fits`。
fit_id 形如 `A-raw-consensus`。

- 每条 fit 固定 **1502 FG**（与旧 `612+404+486=1502` 调用数相同），全程 1Mb级 full grid。
  必须报告 wall；**不得宣称与旧 baseline 成本相等**（旧 1502 FG 含 5Mb/2Mb 低分辨率 stage，
  本轮 1502 FG 全为 1Mb full-grid）。
- 不默认延长预算；若 canonical 收敛则如实 `converged`，否则 `budget_not_converged`。
  退出码 0 不代表科学成功。
- 每 10 accepted 保存 checkpoint（后验诊断需要），另保存 iter0（`accepted-00000.npz`）与
  last-accepted endpoint；每条 fit 完整保存 `.3dg` / `.npz`（含 `coordinates/raw_y/theta/p/q`）。
- 预期总 outer FG `12*1502 = 18,024`。

## 6. 必要工程门（全部通过后才启动正式 12 fit）

1. 原 marginal raw（loss A）在 046 baseline 固定点的 value/gradient 与 045
   `SharedCaptureObjective(model_id="G")` wrapper parity。
2. 三 loss 有限差分：方向导数、多个 p 与多个坐标方向、max 唯一点；另测 exact tie 的
   subgradient value 与 gradient covariance、gauge 不变性。
3. 全 chr A/B 交换及 global rotation 下三 loss 物理值不变。
4. `P` 的 inverse/roundtrip、chain 与 identity 小 fixture parity、`P` 对称性、
   `gradient_z = P gradient_y` 的有限差分验证；非 identity `P` 下 runner 报告的
   canonical raw_y/q 梯度与独立计算的真正 raw_y/q 梯度一致；checkpoint.y 与 sphere 坐标一致。
5. 12 cell 各 2 FG 的同一正式入口 full-grid 短集成（scratch fixture，工程门不冒充正式）。
6. 共享起点位相隔离（同 source 6 条 fit 的 iter0 raw_y hash 相同；两 source 之间不同）、
   diag+offdiag 分母完整（`Noff=1265114`、`Nraw=1703888`、diag=438774）、
   export readback（`.3dg` 回读逐元素等于 endpoint coords）、
   last-accepted / FG cap / gradient stop 状态检查。
7. 门禁前后核对旧 045/046 source、native、既有实验字节未改。

## 7. 无 reference 机制诊断

在训练侧 baseline（046 `real-extension-G-full-J`）与新候选的 hash 都冻结之后、打开 reference 之前执行：

1. **posterior 归属**：每 fit 的 iter0 / 每 10 accepted checkpoint / final，用统一原 marginal
   后验 `gamma_s = t_s / Σ t`（不用 max 伪造确定性）。报告按 `1,265,114` counts 加权的
   all/intra/inter 的 entropy、max posterior、`fraction gamma_max >= 0.9`、
   相对 iter0 的 MAP-switch；动态 posterior 的 expected distance / Rg 与 fixed-iter0
   posterior 同口径距离 / Rg 对比；Rg 用全部 `5290` 物理 beads 统一计算。
   这是模型推断诊断，**不是真实 allele accuracy**。
2. **跨分辨率闭合**：固定 046 baseline 1Mb coords/p，按数值 bin 映射把细 rates 求和到 2Mb/5Mb
   的非对角事件；用同一细 coords 按粗 bin 均值构建 coarse 代表点，按各层实际 `e/l0/r0`
   重算 coarse-point rates。比较同一粗 offdiag 事件集合上归一化概率的 KL、相关、
   cis/inter 总质量、190 chr-pair 贡献。报告细 pair 映入粗 diag 的计数/质量（不隐删）。
   raw 聚合与 1Mb 求和计数必须相等；5Mb 与 2Mb 本身不嵌套。
   coarse R 与 sum fine R 一般不一致，但**不预言数值/因果**，不跨网格比绝对 NLL。
3. **局部可观测性 probe**：baseline 物理 coords，全部 20 chr 两 copy，固定 seed `461001`
   生成低频平滑方向与高频局部随机方向各 1，整体去平移，按 **whole-cell displacement RMS**
   归一化；各取 `±0.01` 与 `±0.05` 倍 baseline whole-cell Rg，共 8 probe。
   方向先固定，所有点沿直线若超 strict ball 则共同缩小幅度到可行并记录 actual（不逐 bead clip）；
   固定 `e`、`p`，不拟合，完整 q 预测。报告相对 baseline 的
   `Noff/Nraw * KL(q_base||q_probe)`、data count NLL delta、full J delta 与正则拆分、
   经单一 whole-cell proper rigid 对齐 baseline 后的 RMS 形变（不用 reference）。
   仅为局部方向可观测性，不是全局不可识别证明；probe coords+hash 归档，不作 formal 选候选。

## 8. 选择与评价冻结

- 每 `loss × solver` 内 consensus/random 只按自身 final count objective 选 source
  （差 `<=1e-9` 取 consensus）。
- 三 loss 的数值不能直接据较小宣布更好；每个 endpoint 额外用**共同原 G count / fullJ**
  value-only rescore 供解释。不据 reference 选终点或超参。
- 显示选择预先冻结：每 loss 从其 4 个 endpoint（2 solver × 2 source）按自身 count 选 1，
  然后加 ref/baseline 共 5 panel。该选择在打开 reference 之前写入 `results/selection_pre_reference.json`。
- 全部 12 endpoint、2 initial control、baseline 及 null 先生成 hash，之后才 open reference。
- 评价 mask：`046/evaluation_final/results/frozen_legacy_mask_snapshot.npz`（评估侧），
  old21 数值规则 `range(3_000_000, chr_len, 1_000_000)`；
  `20 chr / 157,529 within pairs / 176,201 total`，`validbins=2447`，
  inter `2,835,152` locus pairs `= 11,340,608` 四距。不读新 finite 交集偷缩分母，缺失记 NA。
- 每 chr 四 Pearson/Spearman，whole-chr direct/swapped/tie、matched/cross/contrast/minmargin。
  20 chr paired bootstrap `10000` draws、seed `450301`、报胜出数，相对 baseline 及同 source
  的 solver/loss 配对。本轮只做 R2 与空间，不做 R1/R3，不声称 L2。
- null：`u0 + 16 random-u`（seeds `450500..450515`）给 6 个按 count 选的 loss×solver endpoint，
  加 baseline（复用 046 现有同 hash null）与两 initial；每 chr 固定 `z`、permute `u`，
  超球域统一 rescale 并记录。null 全部先 hash 再 open reference。
  随机分裂 initial 与共识作同格对照。R2/inter null draw 表及 mean/range 必报；
  16 draw 不是生物重复。
- 空间评价：pooled sorted 四距 Pearson/Spearman + 各 order-statistic 分层相关；
  20 merged chr-centers / 190 距离的 Spearman、Pearson、optimal single distance scale
  normalized stress、每 chr top3 neighbor overlap。40 copy-center/760 跨 chr 距离按预定
  whole-chr intraR2 方向统一 mapping + proper global alignment；20 center 指标本身 gauge-free。
  global O(3) reflection 敏感性只汇报单个 global transform，不逐 chr fit。
  固定 mask 中心支持；参考/候选同一支持与共通尺度。190 距离整 chr 标签置换 null
  `9999` draws、seed `461100`（dyadic 依赖，不是普通 pair p）。
  几何效果不直接称 phase/L2 恢复。

## 9. 交付

- `README.md`（中文内部）、`PLAN.md`、`config.json`、`source/`、`logs/`、`coords/`、
  `results/`、`plots/`、`gates/`、`diagnostics/`。
- 最多两张 PNG：主要改善指标图（英文 7pt / 3-inch panel / 300dpi），
  whole-genome 对照图（固定视角 / 全局规范对齐 / 共享 axes，5 panel = 3 loss display + ref + baseline）。
  图内标题标明真实选择与终态。
- README 表格列全 12 fits 及 baseline、initial、selection/各 count rescore、R2/inter/centroid、
  预算与实际 terminal。若无改善诚实报告，不自动无依据加预算。
- 明确声明：本设计不是生物重复；baseline 为事后采纳的工作对照；旧 baseline 1988 FG
  与本轮 1502 全 fine FG 成本不同。不更新 `CURRENT_BASELINE`，先交父侧验收。

## 10. 预计耗时

- 工程门：约 20–40 min（含 12 cell × 2 FG 短集成与有限差分）。
- 正式 12 fit：单 FG 1Mb full-grid 约 0.25–0.45 s（依据 046 extension 实测
  4 arm × 486 FG ≈ 519 s），`18,024` FG 串行约 75–135 min；按 3 路并发预期 40–70 min。
- 诊断 + 评价 + 图：30–60 min。总预期 2–4 h wall。
