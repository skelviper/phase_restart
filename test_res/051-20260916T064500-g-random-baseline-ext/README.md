# 051 轮内部 README：G-random 简化 baseline + 三组对照

本轮只回答用户提出的三个对照问题，并把不带第二轮 1Mb 优化的 046 G-random 基础端点登记为新
baseline。cohort 仍是**一个真实细胞、20 条染色体 / 40 条 trace**；20 条染色体是同一细胞内的
关联测量，**不是生物学重复**，本文件所有 per-chromosome 统计都只描述该细胞内的变异。

## 1. baseline（用户主动指定）

| 项 | 值 |
| --- | --- |
| ID | `P9016-046-G-random-base-1Mb` |
| 3DG | `test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.3dg` |
| 3DG SHA256 | `ee5eb1545db9bfeeabcc24e704707f5f61a5793e6f245091347373442dc0032b` |
| NPZ | `.../base_remaining/coords/real-G-random/1Mb.npz` |
| NPZ SHA256 | `6bb93bf570cf688fd8b02dccf14ca0867c1f831d5e123036ced626856f8d8752` |
| lineage | 014 无标签 random 起点（seed 2207）→ 5Mb 612FG → 2Mb 404FG → 1Mb 486FG，共 1502FG |
| 终态 | `budget_not_converged`（`fg_budget_exhausted`），每层 `last_accepted_endpoint=true` |

不带第二段 1Mb extension（旧的 `real-extension-G-full-J` / `count-only`，各 486FG）——用户明确
取消了第二段。旧运行、冻结 mask、旧评价与冻结协议均未改动；`docs/CURRENT_BASELINE.md` 已更新为
该基础端点，并保留旧 extension 作为历史。

**轻量 sanity（必要）**：在 046 冻结 legacy 21-mask 公共支持上重算，baseline 的 matched
Pearson 20-chr 均值 = `0.6023605799695413`，与已发布值 `0.6023605799695414` 一致（差 1e-16，
浮点求和顺序）。没有重做 050 的回归矩阵。

## 2. 三组对照（用户明确列出的三个）

| 组 | fit_id | 层与 FG cap | 说明 |
| --- | --- | --- | --- |
| extra-levels | `A-extra-levels` | 20Mb 200 → 10Mb 200 → 5Mb 612 → 2Mb 404 → 1Mb 486（共 1902FG，比 baseline 多 400FG） | 新 20/10Mb 是主侧为快速回答设的粗层，**不是等成本比较** |
| no-bend | `B-no-bend` | 5Mb 612 → 2Mb 404 → 1Mb 486（共 1502FG） | 只有 bend 权重 `.01 → 0`，从起点重跑 |
| reference-beads | `C-reference-beads` | 5Mb 612 → 2Mb 404 → 1Mb 486（共 1502FG） | 逐 copy 珠子支持（reference-informed），bend 仍 `.01` |

共同固定：普通 raw L-BFGS、G 共享 offdiag 归一化、完整 `J = count + bond + repulsion + .01·bend + p_prior`、
固定 production e、p/q 优化、球域、`ftol=0`、`canonical_gtol=1e-6`、`maxls=20`、精确 FG cap、
最后 accepted endpoint。`l0=(2·n_loci)^(-1/3)`、`r0=2·l0`、repulsion threshold `0.7·l0` 全部读
`AggregatedContacts` 属性（见 `config.json`）。

起点：B/C 的 5Mb 起点逐元素相同（`coords/initial/base_start_5Mb.npz`，SHA
`c2fdf8a4c6a52cf3acdcfc48485a38c71cc106c15b644423150fb9695e617a85`，与 045 `real_random_5Mb`
逐位一致）。A 的 20/10Mb 由同一 014 random 轨迹直接 expand 到 full grid，5Mb 由 A 自己的 10Mb
端点 prolongation 得到，因此 **A 的 5/2/1Mb 与 baseline 不同支**；B/C 与 baseline 共享 5Mb 起点。

### 三组真实终态（退出码 0 不等于收敛）

| 组 | 20Mb | 10Mb | 5Mb | 2Mb | 1Mb |
| --- | --- | --- | --- | --- | --- |
| A extra-levels | budget_not_converged 200FG | budget_not_converged 200FG | budget_not_converged 612FG | budget_not_converged 404FG | budget_not_converged 486FG |
| B no-bend | — | — | budget_not_converged 612FG | budget_not_converged 404FG | budget_not_converged 486FG |
| C reference-beads | — | — | budget_not_converged 612FG | budget_not_converged 404FG | budget_not_converged 486FG |

全部层都是 `fg_budget_exhausted` + `last_accepted_endpoint=true`；每层实际 FG、最后 accepted 标记、
canonical raw-y/q 梯度、count NLL、p/q 见 `stages/<fit_id>/<stage>.json` 与
`logs/<fit_id>-<stage>.terminal.json`。

最终 1Mb 坐标（本轮交付）：

| 组 | NPZ SHA256 | 3DG SHA256 |
| --- | --- | --- |
| A extra-levels | `60711facfa41e60b7d5283e18f5aa221e227561d007c4a82729c06a36c02757a` | `d8f779e7c85271fca2df89512645b2ddcd9d79134839e6b62e656cc1b74aa29f` |
| B no-bend | `9af861c1e62006919f092a13981ff1e6aa84264b1a9f639d66462a4bdc8805eb` | `e463cede991bd7f7ec666139e721579dad8f29fb2b7d6f2d14b88f6268b874d4` |
| C reference-beads | `f4d42c9fc2f878b87032d2c2b2c4d0feb0aa621ed2be88ea102e0e89e83fb760` | `9c60d02aca38e595d27e753984c92b6428ad3c9731bf45f2c1803398b4c8bc92` |

## 3. reference-beads support（C 的授权边界）

* 训练侧只接受布尔 mask；reference 的 xyz 数值、phase 列都不进入训练进程。
* mask 由 `code/build_support.py` 生成：1Mb 层保留 reference 中**该 copy 存在且 xyz 有限**的
  locus；更粗层把 1Mb 有效 locus 按数值 bin（position // bin_size）归并，粗 bin 内至少含一个
  有效 1Mb locus 即视为存在。20/10/5/2/1Mb 的 mask SHA 见 `config.json`。
* 珠子计数（用 `np.count_nonzero` 直接算，`inputs/reference_support_manifest.json`）：

  | 层 | copyA(mat) | copyB(pat) | 共有 | 仅 A | 仅 B | union |
  | --- | --- | --- | --- | --- | --- | --- |
  | 1Mb | 2460 | 2487 | 2447 | 13 | 40 | 2500 |
  | 2Mb | 1263 | 1268 | 1260 | 3 | 8 | 1271 |
  | 5Mb | 525 | 527 | 525 | 0 | 2 | 527 |

  （1Mb 只有 2500 个 union locus 而不是 2645，是因为 reference track 行只覆盖 3Mb 起的真实
  locus；`B-only = |B| − |A∩B| = 2487 − 2447 = 40`，`|B| − |A| = 27` 不是 B-only。）
* 过滤规则（冻结）：`lane = mask.any(axis=0)`；pair 保留当两端都在 lane；diag 只在该 locus 在
  lane 时保留；**单边存在的 locus 的 contacts 保留**（由存在的那条 copy 解释）。`Nraw`/`Noff`
  用 filtered 值，e 用 filtered endpoint counts 的 production 公式 `sqrt(endpoint+10)` 在 lane
  上归一到均值 1、非 lane 为 0。
* 缺失珠子真正退出目标：`_kernel_block` 按逐 copy mask 对四项 K 及其梯度乘 0/1，bond 只保留
  两端存在且原 bp 相邻的 edge，bend 只保留三珠存在且相邻的 triple，repulsion 的每个物理珠对
  仅在该 pair 对应 copy 的两端珠子都存在时计入（含同 locus A/B 对角项）；归一化按有效珠子数
  `Nactive`（all-ones 时回到 `2·n_loci`）。导出 3DG 只写真实珠子（1Mb 恰好 4947 行，与
  reference 有限点集合逐位相同），NPZ 保留 full grid 但缺失处为 NaN。
* 必要验证（`code/validate_masked.py`，5Mb/2Mb/1Mb 全部 PASS）：all-ones 的
  value / count_nll / bond / bend / repulsion / 梯度与冻结 G 逐位 parity（max|dp| ≤ 5.6e-17）；
  缺失 lane 坐标惰性（value delta 0、active 梯度 delta ≤ 5.6e-17、缺失自由度梯度 0）；缺口例子
  bond/bend 项数按真实删除减少、count 移除对应 pair；单边 locus 的 contacts 保留；方向中心
  有限差分 rel ≤ 3.2e-9；C 跨层 prolongation 在 all-ones 时与原 `warm_start_from_layer` 逐位一致
  （max|d| = 0）。
* **不要**把 C 说成完全 blind，也不要把它的 count NLL 与 A/B 直接排名（有效分母不同）。

## 4. 评价（唯一正式入口 `code/eval_pearson.py`）

* 公共支持 = 046 冻结 legacy 21-mask（`evaluation_final/results/frozen_legacy_mask_snapshot.npz`）。
  四个条件与 reference 在该支持上**同一分母**：157529 个 intra pair，`dropped = 0`
  （`evaluation/shared_support_audit.json`）。C 的 1Mb 支持覆盖全部 legacy 公共 pair
  （`inputs/legacy_mask_containment.json`），所以分母既不扩也不缩。
* 索引映射：`global = offset + position // 1Mb`，用真实数值 position，不用压缩下标当 bin。
* 指标：每 chr 一次 whole-chr signed Pearson 最佳 A/B 互换（`direct` vs `swapped`），
  `same = matched`、`cross = swapped`，两 copy 平均后每 chr 一个 same 和一个 cross；20 chr 分别
  画箱线图。无定义值写 NA，不填 0，不因为 NA 删 chr；本轮 4 组 × 20 chr 全部有定义。
* 表：`evaluation/per_chromosome.tsv`（chr / condition / same / cross / contrast / n_pairs /
  mapping / n_positions / A_mat / A_pat / B_mat / B_pat，另附 legacy 分母列）。
  均值与逐 chr 原值见 `evaluation/pearson_summary.json`。不做 bootstrap、不做显著性检验。

### 主要读数（15 位有效数字见 `pearson_summary.json`）

| 组 | same 均值 | same 中位数 | cross 均值 | cross 中位数 | same−cross 均值 |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.6023606 | 0.6175251 | 0.3656173 | 0.3487830 | 0.2367433 |
| extra-levels | 0.6152262 | 0.6401852 | 0.3803363 | 0.4124673 | 0.2348899 |
| no-bend | 0.6105771 | 0.6141432 | 0.3819714 | 0.3755336 | 0.2286057 |
| reference-beads | 0.5736072 | 0.5868107 | 0.3933622 | 0.3643852 | 0.1802450 |

same−cross 间隔（由上表均值相减，不另跑分析）：baseline `0.2367433`、extra-levels `0.2348899`、
no-bend `0.2286057`、reference-beads `0.1802450`。三组的间隔均**未超过** baseline，即没有任何一组
把「匹配拷贝相关性 − 交换拷贝相关性」的间隔拉大；这仍然不构成 L2 的证据。

对照含义（仅描述该细胞的 20 条染色体）：`extra-levels` 与 `no-bend` 的 same 略高于 baseline
（+0.0129 / +0.0082），`reference-beads` 的 same 略低且 cross 略高，即 same−cross 间隔更小。
这不构成 L2 的证据：same 高不等于沿整条染色体的一致性拷贝身份被恢复。

## 5. 两张交付图（`code/make_plots.py`）

* `plots/chr1_distance_matrices.png`：chr1 distance matrix，2 行 × 5 列（列 = Reference /
  baseline / extra-levels / no-bend / reference-beads，行 = matched-to-mat / matched-to-pat）。
  缺失 pair 灰色（对角线设 0，缺口不压缩成连续 bp 轴），全图统一距离色标。展示尺度：每个
  condition 用**同一个最优正尺度**同时作用在两 copy 上（joint 最小二乘到 chr1 参考距离向量，
  乘而非除），reference 尺度固定 1；线性缩放不影响 Pearson。每列按该条件自己的最佳 whole-chr
  mapping 决定哪条 copy 显示在 mat / pat 行。
* `plots/same_cross_pearson_boxplot.png`：x = 四组，每组 same/cross 两箱，叠 20 个 chr 点，y 同轴。
  20 条染色体不是独立生物学重复。

图内英文标题/轴/图例；本 README 为中文内部记录。

## 6. 无效尝试与边界

* `invalid_C_v1/`：第一版 masked objective 用“坐标置零 + 换分母”冒充退出，科学不正确，
  结果不交付（含 `masked_objective_v1.py`、coords/stages/logs）。
* `invalid_C_v2_intersection_coords/`：C 第二版把 pair 过滤写成两 copy 交集，删掉了单边存在
  locus 的 contacts，不符合冻结 union 规则，结果作废；当前 C 为 union 规则重跑。
* 本轮不做 R1/R3、不做 P0/P2/null、不做 10000×bootstrap，不读 phase，不新增框架/审计快照。
* 三个对照组的 count NLL 分母不同（C 用有效支持），跨组比较只看同一公共支持上的 Pearson。

## 7. 复现命令

```bash
conda activate analysis
cd test_res/051-20260916T064500-g-random-baseline-ext
python code/prep_inputs.py          # 粗层 aggregate + 盲起点轨迹（也可复用已落盘 inputs/）
python code/build_support.py        # C 的 reference support mask（唯一读 reference 的进程）
bash   code/run_chain.sh A          # 20/10/5/2/1Mb
bash   code/run_chain.sh B          # 5/2/1Mb, bend=0
bash   code/run_chain.sh C 5Mb     # 5/2/1Mb, reference beads（2/1Mb 用修复后的 prolongation）
python code/fix_export.py           # C 只导出真实珠子（保留原 bp）
python code/validate_masked.py --stage 5Mb   # 必要检查（2Mb/1Mb 同样）
python code/eval_pearson.py         # 统一公共支持评价
python code/make_plots.py           # 两张 PNG
```
