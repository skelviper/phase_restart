# 049 max-contact-unified-multiscale 内部报告

运行目录：`test_res/049-20260915T162917Z-max-contact-unified-multiscale/`（run id `049-20260915T162917Z-max-contact-unified-multiscale`）。本文件由 `source/make_readme.py` 从运行产物装配；表内数值全部来自 `results/`、`diagnostics/`、`gates/`、`evaluation/` 的 JSON/TSV。

## 1. 本轮做了什么

- 一个新真实目录，12 条 fit：`3 loss × 2 solver × 2 blind source`，每条固定 **1502 FG**、全程 1Mb full grid。
- loss：`A marginal_G`（= 045 原 G）、`B hard_observed`（用户主方案，normalizer 仍为 Zsum、观测项取 max）、`C max_rate`（Zmax 也取 max）。solver：`raw` 原 raw_y/q；`ms` 链内多尺度线性预条件 `P = I + (I+25L)^-1 + (I+400L)^-1`（DCT-II 对角化，全部 > 0，优化 `z = P^-1 y`，停止用 canonical raw_y/q 梯度）。
- 两个盲起点为 014 获准无标签 source 的 zero-optimization prolongation；同 source 的 6 条 fit 共享逐元素相同的 raw_y/q。
- baseline 为 046 `real-extension-G-full-J`（事后采纳的工作对照，非本轮训练初值）。

## 2. 冻结事实与边界

| 项 | 值 |
| --- | --- |
| SNP-free 输入 | `inputs/P9016.snpfree.pairs.gz` SHA256 `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`（7 列，拒绝 phase） |
| 记录 / loci / full grid | 1,703,888 records；2645 loci；3,496,690 零对保留的 offdiag pair |
| 分母 | `Nraw=1,703,888`；`Noff=696,680+568,434=1,265,114`；diag `438,774` 独立饱和 nuisance |
| 生物重复 | n=1（同一 P9016 细胞），20 chr 为细胞内关联技术/结构测量 |
| 工程门 | 9/9 PASS，failed=[]（`gates/gates_report.json`） |
| reference | 本轮训练与诊断阶段未打开；由评价侧在候选/initial/baseline/null 全部 hash 之后打开 |

## 3. 12 fits 终态与共同 rescore

**旧 baseline 1988 FG 含 5Mb/2Mb stage，本轮每条 fit 1502 FG 全为 1Mb full-grid；两者成本不同，不能称成本相等。**

| fit | loss | solver | source | FG cap | FG 实际 | wall (s) | terminal | canonical gel max | count_A | count_B | count_C | fullJ(own) | fullJ_A(共同) | Rg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `A-ms-consensus` | A | ms | consensus | 1502 | 1502 | 545.6 | fg_budget_exhausted | 0.001 | 9.572251 | 9.847853 | 9.660295 | 9.585451 | 9.585451 | 0.538786 |
| `A-ms-random` | A | ms | random | 1502 | 1502 | 534.7 | fg_budget_exhausted | 0.000 | 9.563901 | 9.843195 | 9.649799 | 9.577054 | 9.577054 | 0.527089 |
| `A-raw-consensus` | A | raw | consensus | 1502 | 1502 | 552.8 | fg_budget_exhausted | 0.000 | 9.579651 | 9.859848 | 9.668363 | 9.592576 | 9.592576 | 0.518148 |
| `A-raw-random` | A | raw | random | 1502 | 1502 | 540.4 | fg_budget_exhausted | 0.001 | 9.581205 | 9.866854 | 9.668467 | 9.594414 | 9.594414 | 0.514510 |
| `B-ms-consensus` | B | ms | consensus | 1502 | 1502 | 592.3 | fg_budget_exhausted | 0.000 | 9.622445 | 9.795518 | 9.693933 | 9.810757 | 9.637684 | 0.597184 |
| `B-ms-random` | B | ms | random | 1502 | 1502 | 611.4 | fg_budget_exhausted | 0.002 | 9.616823 | 9.789638 | 9.686693 | 9.804915 | 9.632100 | 0.613861 |
| `B-raw-consensus` | B | raw | consensus | 1502 | 1502 | 591.3 | fg_budget_exhausted | 0.000 | 9.615111 | 9.787837 | 9.683614 | 9.803466 | 9.630740 | 0.586625 |
| `B-raw-random` | B | raw | random | 1502 | 1502 | 588.9 | fg_budget_exhausted | 0.000 | 9.621341 | 9.819899 | 9.695452 | 9.835804 | 9.637245 | 0.560317 |
| `C-ms-consensus` | C | ms | consensus | 1502 | 1502 | 594.3 | fg_budget_exhausted | 0.001 | 9.659670 | 9.961493 | 9.700378 | 9.715767 | 9.675059 | 0.465298 |
| `C-ms-random` | C | ms | random | 1502 | 1502 | 594.8 | fg_budget_exhausted | 0.000 | 9.632386 | 9.947808 | 9.671565 | 9.687042 | 9.647864 | 0.475519 |
| `C-raw-consensus` | C | raw | consensus | 1502 | 1502 | 609.6 | fg_budget_exhausted | 0.000 | 9.677079 | 9.988950 | 9.716185 | 9.732217 | 9.693112 | 0.449820 |
| `C-raw-random` | C | raw | random | 1502 | 1502 | 591.2 | fg_budget_exhausted | 0.001 | 9.662856 | 9.993047 | 9.694232 | 9.710527 | 9.679151 | 0.446554 |
| `initial-consensus` | - | - | consensus | - | 0 | NA | zero_optimization_initial_control | NA | 10.110235 | 10.700507 | 10.131247 | 11.417025 | 11.417025 | 0.244175 |
| `initial-random` | - | - | random | - | 0 | NA | zero_optimization_initial_control | NA | 10.100333 | 10.717910 | 10.150421 | 11.943546 | 11.943546 | 0.240049 |
| `baseline-046-G-full-J` | A | raw | random | - | 1988 | NA | budget_not_converged | NA | 9.529135 | 9.783897 | 9.611809 | 9.542019 | 9.542019 | 0.546922 |

**共同 count_A 比较（本轮预注册采用的共同目标口径）**：baseline `baseline-046-G-full-J` = 9.529135；12 条新端点最好的 `A-ms-random` = 9.563901（差 +0.034766）；各 loss 最好：A A-ms-random=9.563901；B B-raw-consensus=9.615111；C C-ms-random=9.632386。

**结论（严格限定）：在本轮预注册的共同原 G count 口径下，12 条新端点的最好者仍未超过 046 baseline（该口径无改善）**。这只说明**共同 count 目标上没有改善**；`count_A` 是本轮预注册采用的共同目标口径（A 与 C 都是同一观测空间的归一化模型，B 是 classification/MAP 状态目标），因此**不对三 loss 做泛化数学比较**。本轮**不据此下任何结构结论**：结构恢复是否改善由评价侧的 R2 / inter / 空间读出回答（见 §6），phase/L2 恢复本轮不作声明。

**同一 source 下 solver 对比（真实事实，保留）**：6 组 同 source × 同 loss 的 raw vs ms 中，**5/6 组 multiscale 预条件降低了自身 count**（唯一例外为 B-consensus：ms 9.795518 vs raw 9.787837）。逐组差值见下表。

| loss | source | raw own count | ms own count | ms − raw | ms 更低 |
| --- | --- | --- | --- | --- | --- |
| A | consensus | 9.579651 | 9.572251 | -0.007400 | yes |
| A | random | 9.581205 | 9.563901 | -0.017305 | yes |
| B | consensus | 9.787837 | 9.795518 | +0.007681 | no |
| B | random | 9.819899 | 9.789638 | -0.030261 | yes |
| C | consensus | 9.716185 | 9.700378 | -0.015807 | yes |
| C | random | 9.694232 | 9.671565 | -0.022667 | yes |

注意 baseline 是上一轮 1988 混合尺度 FG 的事后采纳工作对照，与本轮 1502 全 fine FG 成本不同，不是等预算比较；分层的 count_A（A 系 9.563901–9.581205、B 系 9.615111–9.622445、C 系 9.632386–9.677079）只说明各 loss 优化的是不同目标。

恒等式核对（同一坐标/e/p/同常数）：`count_B - count_A = sum_C C*(-log gamma_max)/Nraw >= 0`，`count_C - count_B = (Noff/Nraw)*log(Zmax/Zsum) <= 0`。逐端点实际值与公式值见 `results/endpoint_manifest_pre_reference.json` 的 `B_minus_A_formula/actual` 与 `C_minus_B_formula/actual`。

## 4. 预注册选择（open reference 之前冻结）

| loss | solver | 选中的 source | 自身 count 差 | 规则 |
| --- | --- | --- | --- | --- |
| A marginal_G | ms | random | 0.008350709 | minimum own final count objective |
| A marginal_G | raw | consensus | -0.001553994 | minimum own final count objective |
| B hard_observed | ms | random | 0.005879729 | minimum own final count objective |
| B hard_observed | raw | consensus | -0.032062089 | minimum own final count objective |
| C max_rate | ms | random | 0.028812950 | minimum own final count objective |
| C max_rate | raw | random | 0.021952642 | minimum own final count objective |

显示端点（每 loss 从自身 4 个 endpoint 按自身 count 选 1，先于 reference 冻结）：`A-ms-random` (own count 9.563901)；`B-raw-consensus` (own count 9.787837)；`C-ms-random` (own count 9.671565)

三 loss 的数值不可直接互相宣布更优；跨 loss 解释统一使用共同 `count_A`/`fullJ_A` rescore。

## 5. 无 reference 机制诊断

### 5.1 posterior 归属

`gamma_s = t_s / sum_s t_s`（原 marginal），全部按 1,265,114 counts 加权；**不是真实 allele accuracy**。

| fit | iter0 H_all | final H_all | final H_intra | final H_inter | final max posterior | final frac(gamma_max>=0.9) | final MAP-switch vs iter0 | Rg(iter0) | Rg(final) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `A-ms-consensus` | 1.1587 | 0.6183 | 0.4912 | 0.7741 | 0.7183 | 0.2705 | 0.5364 | 0.2442 | 0.5388 |
| `A-ms-random` | 1.1645 | 0.6189 | 0.4771 | 0.7927 | 0.7138 | 0.2521 | 0.5687 | 0.2400 | 0.5271 |
| `A-raw-consensus` | 1.1587 | 0.6227 | 0.4767 | 0.8017 | 0.7137 | 0.2578 | 0.5302 | 0.2442 | 0.5181 |
| `A-raw-random` | 1.1645 | 0.6329 | 0.4773 | 0.8237 | 0.7084 | 0.2429 | 0.5659 | 0.2400 | 0.5145 |
| `B-ms-consensus` | 1.1587 | 0.4410 | 0.3397 | 0.5651 | 0.8134 | 0.4529 | 0.4837 | 0.2442 | 0.5972 |
| `B-ms-random` | 1.1645 | 0.4395 | 0.3398 | 0.5617 | 0.8135 | 0.4516 | 0.5350 | 0.2400 | 0.6139 |
| `B-raw-consensus` | 1.1587 | 0.4444 | 0.3427 | 0.5690 | 0.8134 | 0.4477 | 0.4766 | 0.2442 | 0.5866 |
| `B-raw-random` | 1.1645 | 0.4970 | 0.3572 | 0.6684 | 0.7886 | 0.3928 | 0.5232 | 0.2400 | 0.5603 |
| `C-ms-consensus` | 1.1587 | 0.6754 | 0.4927 | 0.8993 | 0.6923 | 0.1979 | 0.4693 | 0.2442 | 0.4653 |
| `C-ms-random` | 1.1645 | 0.7003 | 0.5390 | 0.8981 | 0.6798 | 0.1753 | 0.5343 | 0.2400 | 0.4755 |
| `C-raw-consensus` | 1.1587 | 0.6961 | 0.4971 | 0.9400 | 0.6831 | 0.1811 | 0.4527 | 0.2442 | 0.4498 |
| `C-raw-random` | 1.1645 | 0.7097 | 0.5038 | 0.9621 | 0.6690 | 0.1723 | 0.5261 | 0.2400 | 0.4466 |

逐 checkpoint 全表：`diagnostics/posterior_all_states.tsv`；每 fit 一份 `diagnostics/posterior_<fit>.tsv`。

本表数值全部由 `diagnostics/posterior_diagnostic.json` 已封存字段直接渲染（不手抄）。

#### 已冻结 display 端点的 inter 归属距离摘要（距离以自己 whole-cell Rg 归一）

| display endpoint | iter0 posterior inter d/Rg | final **动态** posterior inter d/Rg | final **冻结 iter0** posterior inter d/Rg |
| --- | --- | --- | --- |
| `A-ms-random` (A) | 1.0628576 | 0.8174337 | 1.1847265 |
| `B-raw-consensus` (B) | 1.0391001 | 0.6637901 | 1.1366291 |
| `C-ms-random` (C) | 1.0628576 | 0.8676500 | 1.1557206 |

解释（严格限定）：三种口径的相对大小只说明**后验归属变化会明显影响这一个距离摘要**，而 hard/max 状态本身也会随时间改变归属。冻结的 iter0 gamma **不是真实标签**，因此不能据此推断“结构改善全部来自投机”或“归属全错”；这些端点是否真的更接近真实结构，最终仍由 reference 指标（R2 / inter / 空间）判断。

### 5.2 跨分辨率闭合（046 baseline 固定 1Mb coords/p）

| 粗层 | 粗 loci | 映入粗 diag 的细 pair | 映入 diag 的 counts | counts 最大偏差 | KL(fine-sum||coarse-point) | Pearson(prob) | Pearson(log prob) | 细求和 cis/inter 质量 | 粗点 cis/inter 质量 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2000000 bp | 1329 | 1316 | 77272.0 | 0.0 | 0.01948 | 0.99250 | 0.99702 | 0.5500 / 0.4500 | 0.4891 / 0.5109 |
| 5000000 bp | 538 | 5256 | 168778.0 | 0.0 | 0.07817 | 0.97592 | 0.99127 | 0.5261 / 0.4739 | 0.3995 / 0.6005 |

190 chr-pair 贡献：`diagnostics/cross_resolution_chr_pairs.tsv`。2Mb 与 5Mb 本身不嵌套；不跨网格比绝对 NLL。

### 5.3 局部可观测性 probe（仅 baseline）

baseline whole-cell Rg = 0.546922，p = 0.9752；seed 461001；方向先固定、整体去平移、whole-cell displacement RMS 归一；超 strict ball 则整体缩幅（实测未触发）；固定 e/p、不拟合。

| probe | 幅度(×baseline Rg) | 实际位移幅度(坐标单位) | Noff/Nraw·KL(q_base‖q_probe) | data count NLL Δ(归一) | fullJ Δ(A) | 加权正则 Δ | rigid 后 RMS 形变 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smooth_common_mode | +0.01 | 0.005469 | 9.626e-04 | 0.000853 | 0.003184 | 0.002332 | 0.005466 |
| smooth_common_mode | -0.01 | -0.005469 | 9.603e-04 | 0.000857 | 0.003193 | 0.002336 | 0.005466 |
| smooth_common_mode | +0.05 | 0.027346 | 2.345e-02 | 0.020577 | 0.045520 | 0.024944 | 0.027332 |
| smooth_common_mode | -0.05 | -0.027346 | 2.330e-02 | 0.020540 | 0.044053 | 0.023513 | 0.027332 |
| local_independent_copies | +0.01 | 0.005469 | 1.187e-03 | 0.001420 | 0.009596 | 0.008175 | 0.005469 |
| local_independent_copies | -0.01 | -0.005469 | 1.189e-03 | 0.001414 | 0.009635 | 0.008221 | 0.005469 |
| local_independent_copies | +0.05 | 0.027346 | 2.893e-02 | 0.034083 | 0.165947 | 0.131864 | 0.027344 |
| local_independent_copies | -0.05 | -0.027346 | 2.900e-02 | 0.033930 | 0.168009 | 0.134079 | 0.027344 |

幅度列口径：`fraction` 是相对 baseline whole-cell Rg（0.546922）的乘数，`实际位移幅度` 是 probe 的坐标单位位移（`actual_amplitude`，与 baseline Rg 同单位），两者不是同一个量。

仅为局部方向可观测性，不是全局不可识别证明；probe coords+hash 在 `diagnostics/probe_coords/`。

## 5.4 更正与口径记录

- **PLAN.md §2 起点 seed 文本更正（冻结 PLAN 不改字节）**：PLAN 写 1Mb prolongation seed 为 consensus 5509 / random 4405，
  实际源 metadata 为：5Mb 层 `approved_014_blind_initialization` inner seed 1103/2207；2Mb 层 `multiresolution_warm_start`
  inner seed 4406/5510（candidate_base_seed 1103/2207）；1Mb 层 inner seed **4405(consensus)/5509(random)**。
  `045/source/prepare_inputs.py:274-276` 在 5Mb->2Mb 与 2Mb->1Mb 两步都传 1103/2207 作为 prolongation 参数。
  gate8 用 1103/2207 从冻结 045 5Mb 文件重跑 zero-optimization prolongation，与两个冻结 1Mb 文件**逐元素误差 0.000e+00**。
  因此冻结 PLAN 的两处问题：consensus/random 写反、以及把层内扰动 seed 当成了 prolongation 参数。起点身份以 045 文件 SHA 与逐元素复现为准，12 fit 初态未变。
- **FG 口径**：`18,024 = 12 × 1502` 是**正式矩阵的 FG 上限（并等于实际值，若每条都耗尽 cap）**，
  **不包含**工程门（含 12 cell × 2 FG 短集成）、已废弃的 launch-1 attempt、以及无 reference 诊断的重评分
  （posterior 遍历 iter0 + 每 10 accepted checkpoint + 真 final endpoint、端点三目标重评分、cross-resolution、8 个 probe）。
  这些是额外计算开销，本轮总计算量大于 18,024 FG，不得宣传为只有 18,024。
- **probe 口径**：`smooth_common_mode` 用每 chr DCT modes 1–3 且每 chr 均值为 0，20 个 chr 中心保持不变，主要测内部平滑形变；
  `local_independent_copies` 以逐 bead 扰动为主，只带很小随机中心漂移。因此 8 个 probe 的 KL>0 **不能**据此宣布
  “染色体中心摆位可辨识/充分约束”；本诊断未系统搜索中心摆位弱方向。真实摆位改善由统一 190-center / 760-copy-center 读出来回答。
- 分析层更正：正则按加权口径（bend 权 0.01）且 raw/weighted 分开；probe 的 count NLL delta 含 normalizer 并与 count_A delta 断言一致；
  posterior 加入真 final endpoint 并按 all/intra/inter 分列；cross chr-pair 表 20 cis + 190 inter 且逐行 KL 贡献求和等于总 KL；15 条记录统一 `fit_id`。

## 6. 后置评价（评价侧）

评价终态 `evaluation/results/terminal.json`：`evaluation_terminal=complete`、`validation_status=PASS`、参考首次打开 `2026-09-15T17:30:15.548703+00:00`（gate 之后）、15 datasets / 153 null / 28 comparisons，046 baseline 复算 166 个值零差异，bootstrap 指数矩阵与 046 相同。

详细报告：[`evaluation/REPORT.md`](evaluation/REPORT.md)；两张图：[`049_figure1_core_metrics.png`](evaluation/plots/049_figure1_core_metrics.png)、[`049_figure2_whole_genome.png`](evaluation/plots/049_figure2_whole_genome.png)。下表数字全部由 `evaluation/results/*.tsv` 直接渲染。

### 6.1 15 个 dataset 的主读出

| dataset | kind | terminal | R2 matched | R2 contrast | inter Pearson | inter Spearman | 20-center Pearson | 20-center Spearman | 20-center stress | 760 copy-center Pearson | 760 copy-center stress | 共同 G count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `A-raw-consensus` | formal_fit | budget_not_converged | 0.523501 | 0.143732 | 0.644478 | 0.658267 | 0.348842 | 0.323586 | 0.395060 | 0.094815 | 0.432265 | 9.579651 |
| `A-raw-random` | formal_fit | budget_not_converged | 0.540196 | 0.181957 | 0.635556 | 0.653070 | 0.335272 | 0.310671 | 0.401821 | 0.157872 | 0.413659 | 9.581205 |
| `A-ms-consensus` | formal_fit | budget_not_converged | 0.518728 | 0.154634 | 0.654110 | 0.675367 | 0.307481 | 0.307536 | 0.411009 | 0.084941 | 0.443008 | 9.572251 |
| `A-ms-random` | formal_fit | budget_not_converged | 0.562316 | 0.190904 | 0.650005 | 0.668165 | 0.318445 | 0.301027 | 0.405456 | 0.227142 | 0.405273 | 9.563901 |
| `B-raw-consensus` | formal_fit | budget_not_converged | 0.452346 | 0.098575 | 0.665608 | 0.687110 | 0.247421 | 0.243664 | 0.428102 | 0.109368 | 0.455364 | 9.615111 |
| `B-raw-random` | formal_fit | budget_not_converged | 0.440833 | 0.120624 | 0.628973 | 0.653703 | 0.207917 | 0.194088 | 0.433876 | 0.100826 | 0.442271 | 9.621341 |
| `B-ms-consensus` | formal_fit | budget_not_converged | 0.462362 | 0.097578 | 0.672104 | 0.695407 | 0.288454 | 0.302040 | 0.413672 | 0.059296 | 0.475486 | 9.622445 |
| `B-ms-random` | formal_fit | budget_not_converged | 0.459881 | 0.081809 | 0.657402 | 0.678432 | 0.351479 | 0.339002 | 0.400934 | 0.032872 | 0.486767 | 9.616823 |
| `C-raw-consensus` | formal_fit | budget_not_converged | 0.473737 | 0.083008 | 0.588335 | 0.596760 | 0.352557 | 0.367434 | 0.390324 | 0.224009 | 0.404098 | 9.677079 |
| `C-raw-random` | formal_fit | budget_not_converged | 0.484314 | 0.072975 | 0.607303 | 0.620122 | 0.350287 | 0.338264 | 0.396610 | 0.238007 | 0.392389 | 9.662856 |
| `C-ms-consensus` | formal_fit | budget_not_converged | 0.465503 | 0.092395 | 0.588430 | 0.605711 | 0.249651 | 0.247861 | 0.416447 | 0.176586 | 0.414154 | 9.659670 |
| `C-ms-random` | formal_fit | budget_not_converged | 0.542721 | 0.149547 | 0.617642 | 0.640646 | 0.293171 | 0.277424 | 0.413659 | 0.151986 | 0.418762 | 9.632386 |
| `initial-consensus` | initial_control | zero_optimization_initial_control | 0.317453 | 0.051573 | 0.468435 | 0.463602 | 0.395139 | 0.398232 | 0.378622 | 0.268516 | 0.376889 | 10.110235 |
| `initial-random` | initial_control | zero_optimization_initial_control | 0.268685 | 0.032998 | 0.431410 | 0.430695 | 0.386441 | 0.386916 | 0.382257 | 0.224110 | 0.386415 | 10.100333 |
| `baseline-046-real-extension-G-full-J` | baseline | budget_not_converged | 0.602331 | 0.237621 | 0.695058 | 0.718435 | 0.370574 | 0.322372 | 0.399808 | 0.129128 | 0.435203 | 9.529135 |

R2 为 Pearson（Spearman 见 `r2_summary.tsv`）；matched/cross/contrast 的拷贝标签按几何 direct/swapped 规范选择，tie 记为 unresolved。20-center 指标基于 old21 mask 支持的合并染色体中心（gauge-free）；760 copy-center 为跨染色体拷贝中心距离，按预定 whole-chr intraR2 方向统一映射并做一次 proper global alignment。

### 6.2 关键配对 bootstrap（seed 450301，10000 draws，20 chr 配对）

| 比较 | metric | mean Δ | 95% CI | 左/右胜出 |
| --- | --- | --- | --- | --- |
| `A-raw-consensus` − baseline | pearson:matched | -0.078830 | [-0.1265, -0.0332] | 6 / 14 |
| `A-raw-random` − baseline | pearson:matched | -0.062135 | [-0.1076, -0.0115] | 5 / 15 |
| `A-ms-consensus` − baseline | pearson:matched | -0.083604 | [-0.1414, -0.0273] | 6 / 14 |
| `A-ms-random` − baseline | pearson:matched | -0.040015 | [-0.0944, +0.0105] | 9 / 11 |
| `B-raw-consensus` − baseline | pearson:matched | -0.149985 | [-0.2043, -0.1008] | 0 / 20 |
| `B-raw-random` − baseline | pearson:matched | -0.161498 | [-0.2208, -0.1054] | 1 / 19 |
| `B-ms-consensus` − baseline | pearson:matched | -0.139969 | [-0.1904, -0.0912] | 1 / 19 |
| `B-ms-random` − baseline | pearson:matched | -0.142450 | [-0.1924, -0.0956] | 2 / 18 |
| `C-raw-consensus` − baseline | pearson:matched | -0.128594 | [-0.1705, -0.0909] | 1 / 19 |
| `C-raw-random` − baseline | pearson:matched | -0.118017 | [-0.1708, -0.0668] | 3 / 17 |
| `C-ms-consensus` − baseline | pearson:matched | -0.136828 | [-0.1743, -0.0996] | 1 / 19 |
| `C-ms-random` − baseline | pearson:matched | -0.059610 | [-0.1045, -0.0169] | 6 / 14 |
| `C-ms-random` − `C-raw-random` | pearson:matched | +0.058407 | [+0.0253, +0.0947] | 13 / 7 |
| `C-ms-random` − `C-raw-random` | pearson:contrast | +0.076572 | [+0.0348, +0.1167] | 16 / 4 |
| `C-ms-random` − `C-raw-random` | spearman:matched | +0.044920 | [+0.0144, +0.0780] | 13 / 7 |

### 6.3 结论（平衡且限定）

- **12 条新端点的 R2 matched、contrast 与 pooled inter 全部低于历史 baseline**；其中 11 条的 matched 差 CI 不跨 0，只有 `A-ms-random`（本轮最好 matched 0.562316 vs baseline 0.602331，差 −0.0400，CI[−0.0944,+0.0105]，20 chr 中 9 胜）CI 跨 0。主方案 `B-raw-consensus`：matched 0.452346 / contrast 0.098575 / inter 0.665608，对应 baseline 0.602331 / 0.237621 / 0.695058。
- **局部确有改善**：同 loss 同 source 的 raw→ms 对比中 5/6 组 own count 更低（见 §3）；`C-ms-random` − `C-raw-random` 的 matched +0.058407、CI[+0.0253,+0.0947]、13/20 胜；`A-ms-random` 的 760 copy-center Pearson 0.227142 vs baseline 0.129128、stress 0.405273 vs 0.435203；`C-raw-consensus` 的 merged-center Spearman 0.367434、stress 0.390324 vs baseline 0.322372 / 0.399808（但其 R2/inter 退化）。这些是**后验（reference 之后）空间观察，不能用来改已冻结的 display 选择**。
- `B-ms-random` 只有 merged-center Spearman (0.339002) 高于 baseline (0.322372)，其 stress 0.400934 **差于** baseline 0.399808，因此不能说它两项都更好。
- **重要诊断**：全部 12 条新端点的 20-center Pearson 与 stress 都差于各自冻结 initial （control center r = 0.395139 / 0.386441，stress = 0.378622 / 0.382257）。但所有拟合的 pooled inter 明显高于 initial（0.468435 / 0.431410）。因此**“sorted-four pooled inter 提高”不能直接解读成染色体中心摆位提高**——每个 order statistic 与中心读出必须一起看。
- 不能把 random-u 的高 pooled 值直接说成“u 携带大量正确摆位信息”：16 个 random-u 的 inter Pearson mean 0.668457（range 0.666043–0.670334）本身也很高，其中可能包含 4-order 排序分布/尺度差异；同时 baseline 自身 0.695058 仍高于其 random-u 均值 0.668457，**不能与随机等同**。
- **总结**：本轮**未找到能全面替换 baseline 的方法**，而不是所有指标毫无改善；B（用户主方案）语义更确定，但本轮结构变差。这不否定一切 hard/max 方法或求解器，也不构成 L2 已证明或已否证。

### 6.4 评价侧偏离与 errata（只记录，不改封存产物）

- **pre-gate read of frozen mask snapshot metadata**：PLAN/task 要求 mask snapshot 只在评估侧（gate 之后）加载（影响：positions 由 range(3_000_000, chr_len, 1_000_000) 与染色体长度唯一确定，未参与训练、候选选择或任何指标计算；该提前读取不改变任何评价数值，但确实偏离了“整个 mask gate 后 load”的操作要求）
- **049/config.json mask sha256 transcription typo**：config.json 记为 9c551c6a4586a9221541f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9，文件实际为 9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9；文件哈希与 046 evaluation_final/results/mask_validation.json 中创建时记录（frozen_legacy_mask_worker.output_sha256）一致，文件未被修改。评价侧只记录、不改 config（不在写范围）。（影响：mask 内容与冻结计数（176201/157529/2447/2835152）一致，评价数值不受影响）
- **mask SHA 笔误的更正状态（erratum）**：上面第 2 条是评价侧读到的 config 快照记录。**权威值 = `9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9`**，与 mask 文件实际哈希、以及 046 创建记录 `evaluation_final/results/mask_validation.json` 的 `frozen_legacy_mask_worker.output_sha256` 三者一致。本目录 `config.json` 的 `evaluation.mask.sha256` 现为该权威值，并在 `errata` 段保留本条记录；mask 内容与冻结计数（total 176201 / common 157529 / validbins 2447 / inter 2835152 locus pairs）未变，未修改任何 mask/参考/坐标文件。
- **预 gate mask metadata 早读偏离的更正状态**：见上面第 1 条；已如实登记，未补造审计、未改变任何评价数值。
- **attempt 1 失败（exit 1）与 attempt 2 成功是两件事**（详见 `evaluation/logs/EVALUATION_ATTEMPTS.md`）：attempt 1 在 `2026-09-15T17:30:15.548703+00:00` 首次打开 reference（SHA 校验通过、40 track 解析成功）后，于 `load_real_masks` 抛出 `RuntimeError: frozen mask snapshot hash mismatch`；**失败原因是评价侧常量沿用了 config.json 中抄错 1 个字符的 mask sha256（`...1541f55f...` 而非文件实际的 `...1547f55f...`）**。核对 046 创建记录（`mask_validation.json → frozen_legacy_mask_worker.output_sha256`）并改正常量后，attempt 2 完成完整 15-dataset 评价、**exit code 0**，并保留 attempt 1 的真实首次打开时间（`open_attempt_count=2`）。诚实说明：attempt 1 的原始 stderr 写在`logs/evaluate_formal.log`，该文件被 attempt 2 重跑覆盖（未 append），因此**不补造** traceback 原文；上述异常消息与首次打开时间保存在 `evaluation/gates/reference_open.json`、`results/validation.json` 的 deviations（`mask_config_sha256_typo_recorded=true`）与 `REPORT.md`。
- **attempt 2 之后的派生输出路径元数据修复（与上面那次重跑无关，也不是它的失败原因）**：`terminal.json` / `evaluation.json` 中12 个 fit 的 `npz_path` 曾被写成 CWD 相对的 `evaluation/test_res/049-.../coords/...`（对已是仓库相对形式的路径直接 `rel()`，未先 `resolve_path`），文件实际不存在。修复从已保存的 `evaluation.json`/`terminal.json` 与冻结 manifest 只重写路径字段，共 **36 处**（`evaluation.datasets[*]` 12 + `terminal.datasets[*]` 12 + `endpoint_common_g[*]` 12），写入前用 `strip_paths` 深比较断言**数值与其它字段 0 变化**：`numeric_or_other_fields_changed=0`、`recomputed_metrics=false`（before/after sha256 见 `results/validation.json → path_repair`）；路径存在性检查 348 条、缺失 0（`results/path_check.json` PASS）。**这一步不重算任何指标。**

## 7. 诚实结论与限制

- 本设计不是生物重复：n=1 细胞，20 chr 为关联测量，bootstrap/置换只描述细胞内技术/结构变异。
- baseline（046 G full-J）是**事后采纳的工作对照**，其 1988 FG 含低分辨率 stage，与本轮 1502 全 fine FG 成本不同。
- `exit code 0` 不代表科学成功；每条 fit 的真实终态见第 3 节 `terminal` 列。
- **12/12 fit 均为 `budget_not_converged`**（末态 canonical 梯度 2.3e-4 ~ 1.8e-3，均未达 `1e-6`）；因此本轮没有“已收敛”端点，任何“哪个方法更好”的结论都带有**预算已耗尽但仍未收敛**这一保留条件。
- **L2（沿整条染色体的一致拷贝身份恢复）本轮仍未证明**；本轮只做 R2 与空间读出，不做 R1/R3，不声称 L2，也不把拷贝标签当作母源/父源标签。
- 参考结构只在候选/initial/baseline/null 全部 hash 之后由评价侧打开（`reference_first_opened_utc` 见 §6），未参与初始化、停止、source 选择或超参选择；display 选择在打开参考之前冻结，未据 reference 改选。
- 本轮结果已交父侧验收；**未替换 `docs/CURRENT_BASELINE.md`**（baseline 仍是 046 G full-J 工作对照）。

_本文件由脚本生成，生成时间 2026-09-16T00:54:57.557616+00:00。_
