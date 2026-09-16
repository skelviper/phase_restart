# 055 轮：历史 max-contact 端点 vs 当前 baseline vs reference（描述性对照）

运行目录 `test_res/055-20260916_112451-max-contact-vs-baseline-review/`（run id `055-20260916_112451-max-contact-vs-baseline-review`）。
本文件是中文内部记录；图内文案为英文。本轮**不拟合、不重训、不 bootstrap、不做 R1/R3、不改任何旧目录字节**，
只读既有端点做一次冻结口径的描述性比较。

## 1. 本轮做了什么

用户要求比较「历史只保留最高概率 contact 计算 loss」的版本、最新 baseline 与 reference，给出
same/cross 箱线图与 chr1 热图，并核查历史实现的正确性。本轮把"最高概率版本"落到 049 轮预注册的
两个 max-contact loss 端点上（B `hard_observed`、C `max_rate`），与当前 baseline（046 G-random 基础端点）
及 reference 在同一冻结 mask 的公共支持上比较。

| 组 | 端点 | loss | 起点 source | solver | FG | 终态 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline-046-G-random | `046/base_remaining/coords/real-G-random/1Mb.3dg` | A `marginal_G` | 014 无标签 random (seed 2207) | raw（多尺度 5→2→1 Mb） | 612+404+486=1502 | `budget_not_converged` / `fg_budget_exhausted` |
| B-hard-observed | `049/coords/B-raw-consensus/1Mb.3dg` | B `hard_observed` | consensus | raw | 1502（全 1Mb full grid） | `budget_not_converged` / `fg_budget_exhausted` |
| C-max-rate | `049/coords/C-ms-random/1Mb.3dg` | C `max_rate` | random | ms（链内多尺度预条件） | 1502（全 1Mb full grid） | `budget_not_converged` / `fg_budget_exhausted` |
| Reference | `data/P9016.1m.3dg.gz` | — | — | — | — | 结构对照（self control，非拟合） |

B（`B-raw-consensus`）与 C（`C-ms-random`）的选择沿用 049 在打开 reference **之前**冻结的 display selection
（`results/selection_pre_reference.json`，2026-09-15T17:26:49Z；reference 首次打开 17:30:15Z），本轮不按 reference 重选。
**Baseline 不属于该 selection**：它是 2026-09-16 由用户事后指定的工作 baseline（`docs/CURRENT_BASELINE.md` 与 `051/config.json` 一致的
`base_remaining` 端点），未经 049 的无 reference 选择流程。

## 2. 两张图（唯一图形交付）

| 文件 | 内容 |
| --- | --- |
| `plots/same_cross_contrast_boxplot.png` | 3 面板（same / cross / contrast）x 4 组，每组 20 条染色体实测点；Pearson。SHA256 `0b8b6c74c00bbb6a7526eec76ee55b5708c95caaedc7b50f86c2f70070abd595` |
| `plots/chr1_distance_matrices.png` | chr1 距离矩阵 4 行（Reference / Baseline / B / C）x 2 列；每 panel 标注原始 copy（copy A/B 或 Reference mat/pat）→ ref mat/pat 与该 panel 自己的 Pearson r；行左侧标注该行 same/cross 双 copy 均值一次；共享 bin、统一色标不截断、缺失灰、对角 0、英文标注。SHA256 `8973e7a6c30dd7551635c468276481484424652c275b4b5b9ca50c8d7ac47331` |

规范：3 英寸基础面板、300 DPI、统一 7 pt 文字、`coolwarm_r`；图例/标签置于坐标区外，无标题裁切。
标注修订（2026-09-16T11:31Z，仅改绘图脚本与 README，不重算任何指标）：缺失标注改为**按 bin** 计数
（`Missing bins: 4 / 3`，即 reference 两条 copy 各自整行缺失的 bin 数），此前的 `1528 / 1149` 是矩阵网格数
（matrix entries），其口径已在 `eval/summary.json → chr1_panels.nan_counts` 中保留为矩阵计数。

## 3. 冻结口径（same / cross / contrast 的确切定义）

- 支持集：`046/evaluation_final/results/frozen_legacy_mask_snapshot.npz`
  （SHA256 `9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9`）中每 chr 的
  `positions / pair_i / pair_j / common`，再与四个数据集（各自的 mat/pat 或 copy a/b）在 pair 两端**有限**的交集。
  四个组共用同一分母，无组内分母，任何缺失显式报告（本轮 dropped = 0）。
- 距离：候选/reference 1Mb 坐标的欧氏距离，在 mask 的 `positions` 数值位置键上取值（不使用压缩下标，不做字符串比较）。
- 每 chr 4 个相关：`A_mat / A_pat / B_mat / B_pat`（A = 候选 copy a，B = 候选 copy b；mat/pat = reference 两条真实单倍型轨迹）。
- `direct = (A_mat + B_pat)/2`，`swapped = (A_pat + B_mat)/2`；`same = max(direct, swapped)`，
  `cross = min(direct, swapped)`，`contrast = same − cross`；定向是**整条染色体一次** A/B 互换（不是逐 bin 调标签），
  并列（`|direct − swapped| ≤ 1e-12`）记为 `unresolved_tie`。**same/cross 就是 matched/swapped**，不是"同 copy 对 / 跨 copy 对"的距离比较。
- 因此 `same` 是逐 chr 择优后的量、`cross` 是配套的次优量，二者必须成对报告；`contrast` 是对标签互换不变的判别读出。
- Reference self control：`same ≡ 1`（自相关退化，不能当作拟合证据），`cross` 为 mat/pat 距离向量相关，仅作结构对照。
- 次要口径：同一支持与定向规则下的有符号 Spearman（平均秩），用于与 053/054 对齐。

## 4. 数值（`eval/summary.json`；20 条染色体，全部 defined）

| 组 | same 均值 / 中位数 | cross 均值 / 中位数 | contrast 均值 / 中位数 |
| --- | --- | --- | --- |
| Reference（self） | 1.000000 / 1.000000 | 0.261784 / 0.286217 | 0.738216 / 0.713783 |
| Baseline-046-G-random | 0.602361 / 0.617525 | 0.365617 / 0.348783 | 0.236743 / 0.257475 |
| B-hard-observed | 0.452346 / 0.426026 | 0.353771 / 0.350388 | 0.098575 / 0.066040 |
| C-max-rate | 0.542721 / 0.518449 | 0.393174 / 0.414706 | 0.149547 / 0.155479 |

Spearman 均值：Reference `1.000000 / 0.251047`；Baseline `0.604876 / 0.377608`（contrast `0.227268`）；
B `0.459902 / 0.371749`（`0.088152`）；C `0.549744 / 0.412545`（`0.137199`）。

chr1（共同 `17,578` 对，193 个 mask bin）：

| 组 | same | cross | contrast | orientation |
| --- | --- | --- | --- | --- |
| Reference | 1.000000 | 0.341779 | 0.658221 | direct |
| Baseline-046-G-random | 0.699845 | 0.396239 | 0.303606 | swapped |
| B-hard-observed | 0.552730 | 0.431604 | 0.121127 | direct |
| C-max-rate | 0.712885 | 0.487298 | 0.225587 | swapped |

chr1 热图显示尺度（复刻 `scripts/plot_3dg_comparison.py` 的历史口径：该数据集 chr1 两拷贝在**共同 `17,578` 个 pair**
上的距离合并取一个 raw median）：Reference `1.421528`、Baseline `0.278502`、B `0.280991`、C `0.250965`；
八格统一色标 `0 – 5.492823`。显示与指标的分工须分清：**指标（same/cross/contrast）与显示尺度都用四组共同的 `17,578` pair**；
而每个 panel 画的是该数据集在**全部 193 个共享 bin 上自身有限的**两两距离（候选全覆盖，reference 缺 bin 处保持灰色、不外推），
所以八格的可见格并不是同一个掩码。reference 两拷贝各自整行缺失 `4 / 3` 个 bin（矩阵网格计数分别为 `1528 / 1149`，
见 `eval/summary.json → chr1_panels.nan_counts`）。
（全细胞 Rg 仅作散点口径的次要读出：Reference `3.441468`、Baseline `0.547140`、B `0.586625`、C `0.475519`；不用于热图缩放。）

## 5. 验证（`eval/validation.json`，12/12 PASS）

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| `regression_049_r2_pearson_B-hard-observed` | PASS | 与 049 冻结 `evaluation/results/r2_per_chromosome.tsv` 逐 chr `rho_A_mat/A_pat/B_mat/B_pat/matched/cross` 最大绝对差 < 1e-9 |
| `regression_049_r2_pearson_C-max-rate` | PASS | 同上 |
| `regression_051_baseline_pearson` | PASS | 与 051 冻结 `evaluation/per_chromosome.tsv`（condition=baseline）逐 chr 最大绝对差 < 1e-9 |
| `regression_051_baseline_same_mean` | PASS | 20 chr same 均值 `0.6023605799695413`（052/051 发布值 `0.6023605799695414`，差 1e-16 浮点求和顺序） |
| `regression_053_baseline_spearman` | PASS | 与 053 `eval/summary.json` 的 baseline same/cross/contrast 差 < 1e-12（该轮 baseline 的 npz SHA 与本轮一致） |
| `copy_swap_invariance` | PASS | 候选两条 copy 整体互换后 same/cross/contrast 逐位不变 |
| `support_matches_frozen_mask` | PASS | 共享支持合计 `157,529` = mask common 总数，dropped `0` |
| `all_20_chromosomes_defined` | PASS | 20 chr × 4 组全部 same/cross 有限，无 NA |
| `reference_self_same_is_one` | PASS | reference same 恒为 1，cross 有限 |
| `chr1_matrix_symmetry_and_zero_diagonal` | PASS | 对称误差 `< 1e-12`，有效对角恒 0；reference 两 copy 整行缺失 `4 / 3` 个 bin（矩阵网格计数 `1528 / 1149`，见 `eval/summary.json`） |
| `candidate_position_full_coverage` | PASS | 候选 40 条轨迹覆盖全部 `2,585` 个 mask 位置；reference 覆盖 `2,447` 有效 bin（缺失显式保留） |
| `chr1_panel_scale_positive` | PASS | 四个显示尺度均为正有限；显示缩放不进入 same/cross/contrast |

独立复算未复用旧评价代码（`source/evaluate_review.py` 自行解析 3DG、自算 Pearson/Spearman），却与三份冻结表逐值一致；
`eval/pre_reference_hash_gate.json` 先写盘，随后才解析 reference（本轮未见 pre-gate 早读偏离）。

## 6. 历史实现审查："最高概率 contact 计算 loss" 究竟是什么

049 轮的三个 loss 定义在 `049/source/max_contact_objective.py:28-32`（`LOSS_SPEC`）：

| loss | 归一化项 `Z` | 观测项 | 语义 |
| --- | --- | --- | --- |
| A `marginal_G` | 四态求和（`r_sum`） | `r_sum` 求和 | 045 原 G，作为 parity 基准 |
| B `hard_observed` | 四态求和（`r_sum`） | **四态 max**（`r_max`） | 观测项按四态中速率最高的那一态计似然 |
| C `max_rate` | **四态 max**（`Zmax`） | **四态 max** | max 同时进入归一化与观测项 |

代码事实（`_max_count_and_gradient`，行 81–118）：对每个 pair，`r_sum = e·Σ_s w_s K_s`，`r_max = e·max_s w_s K_s`；
观测项是 `self._counts * log(r_obs)`，即**计数值 C 原样保留**；归一化项是
`Noff · log Z`（`Noff = 1,265,114 = 696,680 cis-offdiag + 568,434 inter`），而 `Z` 的求和对**全 1Mb full grid 的
`3,496,690 = C(2645,2)` 个 pair 的速率**进行（含观测计数为 0 的 pair，零计数 pair 保留在归一化里）。

准确说法：B/C 的**观测项**是同一 pair 四个 copy-state 速率的 max（`r_max`），即对**唯一最大分支等价于 hard MAP 插件似然**；
这与「逐 contact 硬分配」在唯一最大时数值上等价，但实现上**没有另行生成逐 contact 的 assignment 文件**：loss 直接对速率取 max，
计数值 C 与 support 不变，也没有按 max posterior 做任何**阈值筛选**（没有丢弃任何 contact）。
B 的归一化项仍用四态求和（`r_sum`），只有 C 把 max 也放进归一化（`Zmax`）。
诊断里的 `gamma_s = t_s / Σ_s t_s` 是**模型后验归属**（全部按 `1,265,114` counts 加权），**不是真实 allele 归属**，也未进入 loss。

并列与梯度：`_tie_weights`（行 61–66）对精确并列的态**均分 subgradient**，`diagnostics()` 明确写
`exact_tie_rule = "equal split subgradient over exactly tied states; no first-argmax"`（行 349）。

已验证的独立证据（049 `gates/gates_report.json`，9/9 PASS，`failed=[]`）：

- `gate1_lossA_vs_045_G_parity`：A 与 045 原 G 数值差 `0.0`、梯度最大差 `7.1e-18`（GPU atomic 次序噪声）。
- `gate2_finite_difference_all_losses`：A/B/C 的方向导数与只对 q 的导数都与数值差分一致（相对误差 1e-9 ~ 1e-7），
  `all_max_unique=true`、`branch_crossing_directions=0`（max 分支在检验点稳定）。
- `gate3_exact_tie_subgradient_and_gauge`：四态并列（p=0.5，`r_max = r_sum/4`、`Z_max = Z_sum/4`）解析式与实现一致，
  tie 权重矩阵与期望逐位相同；copy 互换下 value 差 `0.0`、梯度协变差 ≤ 1.1e-16。
- `gate4_ab_swap_and_rotation_invariance`：A/B 交换与全局旋转下 loss 不变（`det=+1`，差值 ≤ 8.9e-16）。
- `gate8_lineage_and_denominators`：分母守恒 `raw_records 1,703,888 = same_bin 438,774 + cis_offdiag 696,680 + inter 568,434`，
  full grid `n_pairs = 3,496,690 = C(2645,2)`，`count_mode = raw_integer`。
- `gate9_frozen_bytes_unchanged`：`changed = []`。
- 端点恒等式（`results/endpoint_manifest_pre_reference.json`）：B-raw-consensus
  `count_B − count_A = 0.17272591769355472`（公式 `0.17272591769355577`）；C-ms-random
  `count_C − count_B = −0.27624315421165974`（公式 `−0.27624315421166135`），差 < 1e-14。

### 数据范围、零距离与标签规范

- 输入 `inputs/P9016.snpfree.pairs.gz`（7 列，SHA `f37ed9cc…`，拒绝 phase 列）；1Mb 聚合
  `045/inputs/real_1000000_aggregate.npz`（SHA `80984d80…`）：`1,703,888` 条记录，其中同 bin（零距离）
  `438,774` 条作为**独立饱和 nuisance**（`diag_nll`，与坐标无关的常数）不计入结构归一化，off-diagonal
  `1,265,114` 条（`cis_offdiag 696,680` + `inter 568,434`）进入计数目标；`n_pairs = 3,496,690` 含零计数 pair（保留）。
- 结构指标另有一层零距离排除：冻结 mask 只有 `i<j` 的**非对角** pair（本轮已断言 `pair_i < pair_j`，无自对）；
  049 的 off-diagonal 目标与评价 mask 都不含零距离对。
- 标签规范：A/B（或 mat/pat）是规范自由度；本轮 every-chr 一次互换定向 + 交换不变性检查，
  与 AGENTS.md「跨拷贝指标须按几何选择最佳互换」一致。

## 7. 已知历史偏离与限制（必须与结果同时保留）

1. **049 评价侧的三项已登记偏离**（`049/README.md` §6.4，本轮不改）：(a) pre-gate 早读 frozen mask 的 metadata；
   (b) `049/config.json` 的 mask SHA 曾抄错 1 个字符，导致评价 attempt 1 抛 `frozen mask snapshot hash mismatch`、
   被 attempt 2 覆盖 stderr（原始 traceback 未补造），`open_attempt_count=2`；(c) 12 个 fit 的 `npz_path` 派生路径
   36 处修复，写入前深比较断言数值字段 0 变化、未重算任何指标。**所以不能说 049 的历史完全无问题**；
   本轮只引用其已冻结且可复现的数值（B/C 的 R2 与 Rg 均已逐值复现）。
2. **不是等成本、也不是单因素比较**：Baseline 是 5→2→1 Mb 多尺度延长（1502 FG，含粗层），
   B 是 consensus 起点 + raw solver 的全 1Mb 1502 FG，C 是 random 起点 + ms 预条件的全 1Mb 1502 FG。
   三者起点、solver、层结构都不同，**不能把 B/C 与 baseline 的差异单独归因于 loss**。
   本轮限于既有端点的描述性复核，不追加控制实验（这是本轮的执行范围限制，不是对方法学必要性的判断）。
3. **端点全部 `budget_not_converged`**（`fg_budget_exhausted`，末态梯度远高于 `1e-6`）；退出码 0 ≠ 科学成功。
4. **n = 1 细胞**；20 条染色体是同一细胞内的关联测量，不是生物学重复；本轮无 bootstrap、无显著性、无 R1/R3，
   只报告描述性对照；不作 L2（沿整条染色体一致拷贝身份恢复）声明。
5. **baseline 文本有两处不一致（本轮按较新者）**：`docs/CURRENT_BASELINE.md`（2026-09-16 15:31 更新）与
   `051/config.json` 都把 baseline 定义为 `base_remaining/coords/real-G-random/1Mb.3dg`（SHA `ee5eb154…`），
   本轮采用该端点并逐值复现其发布均值 `0.6023605799695414`；`docs/current_baseline.json`（更早，2026-09-15 23:16）
   仍写 `real-extension-G-full-J`（SHA `4301d4df…`，049 的 baseline 行，matched 0.602331）。两者数值差 ~3e-5，
   但身份不同，需由主代理决定是否统一文档。
6. **热图显示尺度更正（主代理更正）**：主代理在生成图之前把口径从「全细胞 Rg」更正为「历史 `build_matrix` 的
   每数据集 chr1 双拷贝合并 raw median」（见 `config.json → heatmap_scale_correction`；该更正由主代理提出，
   不是用户原话）。这是显示缩放，不改变 same/cross/contrast，也不改历史文件。
7. **图标注修订（2026-09-16T11:31Z，仅图/README）**：缺失标注改为按 bin（`Missing bins: 4 / 3`），panel 标题改为
   写明原始 copy（copy A/B 或 Reference mat/pat）→ ref mat/pat 与该 panel 自己的 Pearson r，行 same/cross 双 copy 均值
   移到左侧行标注一次；计算、色标、布局与支持集均未改变，也未重跑评价。

## 8. 文件与复现

```
config.json                       冻结输入路径/SHA、mask/分母、Pearson 定义、terminals
source/review_paths.py            路径与常量
source/evaluate_review.py         hash gate → 读 reference → same/cross/contrast → 回归核对 → 写 eval/
source/make_figures.py            只读 eval/ 产物画两张图
eval/pre_reference_hash_gate.json 四个 3DG + mask 的 SHA，先于 reference 解析写盘
eval/per_chromosome.tsv           20 chr × 4 组逐 chr 读数（含 NA 口径与交换不变性误差）
eval/summary.json                 均值/中位数、chr1 读出、显示尺度、支持集审计
eval/validation.json              12 项检查与偏离记录
eval/boxplot_data.json            箱线图数据
eval/chr1_panels.npz              chr1 八格矩阵 + bin + 显示尺度 + 统一 vmax
logs/evaluate_review.out, logs/make_figures.out, logs/terminal.json
plots/same_cross_contrast_boxplot.png, plots/chr1_distance_matrices.png
```

复现：

```bash
conda activate analysis
cd test_res/055-20260916_112451-max-contact-vs-baseline-review
python source/evaluate_review.py
MPLCONFIGDIR=scratch/mplcache python source/make_figures.py
```

终态：`logs/terminal.json` → `validation_status = PASS`、`checks_failed = []`、`refit = 0`、`bootstrap = 0`、
`phase_parsed = false`。
