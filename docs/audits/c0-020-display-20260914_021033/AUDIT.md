# P9016 020/029 C0 独立评价与展示审计

审计目录：`docs/audits/c0-020-display-20260914_021033/`

## 结论摘要

本审计没有发现把 029 C0 图错误地画成与 020 不同结构的解析、轨道映射、common mask、矩阵计算或 R2 计算错误。独立 float64 文本解析得到的候选终点、四个 rho、固定参考列 margin、raw matrix 和 normalized matrix 均与已锁定记录一致。029 C0 与 020 的 chr1 差距不能归因于色标或 panel 排列：归一化候选矩阵的形状本身已经改变，且 029 C0 三个 native 起点在相同既有 mask 上的终点读数都明显改善。

唯一确认的展示/追溯问题是：020 旧 NPZ 的 `colormap` 元数据仍写 `coolwarm`，但其后续冻结的 render revision 实际使用 `coolwarm_r`；该 revision 的 provenance/README 已说明这是仅渲染修订。它不改变旧 raw/normalized 数值，也不是造成 C0/020 形状差异的原因。

## 冻结与边界

- cohort：P9016 单细胞，1,703,888 contacts，5,290 physical beads，40 tracks；20 条染色体是同一细胞内的 linked observations，不是生物学重复。
- 评价：只做 R2；1 Mb grid 从 3 Mb 到染色体长度以内；同 bin 和对角 pair 不进入结构统计；数值 genomic join 使用 chromosome + integer bp 坐标。
- common mask：直接读取并独立重建既有 21-condition mask，不扩大到 025 x0。chr1 为 193 bins、188 common bins、18,528 全部 unordered off-diagonal pairs、17,578 common pairs；20 条染色体的 pair count 全部与既有 `evaluation_manifest.json` 一致。
- 配对：reference 列固定为 `chrN(mat)`/`chrN(pat)`；每条候选每条染色体只使用既有 whole-chromosome best-swap orientation；不做局部 swap、不按图选择方向。025 x0 的标准 post-hoc 指标则按 x0 自身四 rho 每染色体独立 best-swap，tie 容差为 `1e-12`。
- 没有读取 phase 字段，没有 fit，没有 selection，没有改写原始 coords、原始 pr、原始 fit/eval 或原报告。reference 在所有 endpoint、x0 和 21-condition mask 坐标 hash 通过后才读取。

冻结配置：`docs/audits/c0-020-display-20260914_021033/config.json`

## 验证证据

最终命令均在 `analysis` conda 环境、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 下运行：

- `python -m py_compile docs/audits/c0-020-display-20260914_021033/audit_c0_020_display.py`：exit 0。
- 独立审计脚本：exit 0。
- x0 best-swap 后处理脚本：exit 0；它只读已有 60 行四-rho 表，不重读坐标/reference，不重算 rho。
- `pdfinfo` 核验 C0 bundle2 的主 6x6 PDF：exit 0，`432 x 432 pts`；统一 candidate-panel index 另为 `6 x 18 inch`（`432 x 1296 pts`）；所有单条件 PNG 为 `1800 x 1800`。
- footer-only render revision：`rerender_short_footer.py` 从现有 unified NPZ 读取矩阵后 exit 0；NPZ SHA before/after 完全相同，未重读 coords/reference、未重算 rho/R2。6 个单条件图和 unified index 的 visible-text bbox 全部在 canvas 内且无文字 bbox 重叠（`render_revision_short_footer.json` 为 PASS）。

机器验证：`validation.json` 的 `status=PASS`、`all_unchanged=true`、`reference_read_after_all_candidate_and_mask_hashes=true`；100 个 endpoint×chromosome 的四 rho 和派生指标最大绝对差均为 `2.220446049250313e-16`。

主要输入锁：

- 020 selected：`afb2d52ae11e342e9b43b3c8042c5581760d177c5e563f2478646ab36b3e7078`
- 022 continuation：`49501f5b38d699fb2c9c8849616edd02b70fccd5ae39e1efe6036e17f5f620e7`
- 029 C0 bundle1/2/3：`629c8af30a5b924c3a661368874ca000e6f119c60f62f48af1278d4184288d5a` / `6a8b249c1b6bde046183740d5591ae5ee185676991bafdba152827d1581db68f` / `65b27356b43d0271fbd8c3167f198a584822f08d99462ab47418437e2fc2c66f`
- 025 x0 bundle1/2/3：`2c55a6d8d3eba202c6de138ca79264d281900a74a806be935bdfb7389ea9fdd4` / `d74f97dbdd8e0fe7701906e12061ddcf0351c566c9648890e540b28feb620358` / `f3a951e28c41152c630208241b99a5f18094475842e1ab9a6c8591824ca99a94`
- reference：`1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29`
- 既有 21-condition evaluation manifest：`fa1b26c834c173604c21f954d494cece8e053dd970e7cb6111f561b349574acb`

## chr1 R2 与尺度

| 条件 | 方向 | 候选映射 | 候选 pooled RMS | matched | cross | contrast |
|---|---|---|---:|---:|---:|---:|
| 020 selected | swapped | B -> mat, A -> pat | 0.3766542939 | 0.6699297451 | 0.5336508647 | 0.1362788804 |
| 022 continuation | swapped | B -> mat, A -> pat | 0.3792290105 | 0.6779122141 | 0.5300218123 | 0.1478904017 |
| 029 C0 bundle1 | direct | A -> mat, B -> pat | 0.3004858440 | 0.5719791227 | 0.5199398043 | 0.0520393184 |
| 029 C0 bundle2 | swapped | B -> mat, A -> pat | 0.2942922368 | 0.5038604957 | 0.4564611074 | 0.0473993883 |
| 029 C0 bundle3 | direct | A -> mat, B -> pat | 0.2895790947 | 0.5810853146 | 0.5232977633 | 0.0577875513 |

reference 两 copy 的 pooled RMS 始终为 `1.5529545301005403`。候选 RMS 明显不同，但每个候选内部始终使用两 copy 一个 pooled RMS，不是 per-copy 独立缩放。尺度差会改变颜色强度，但对同一候选内部的 distance rank/Spearman 不产生解释力；它不能解释上述真实 R2 形状差异。

## 旧/新展示链逐项核对

独立重算采用 full 193-position grid、188 common bins、17,578 common unordered pairs、NaN 灰色、绝对 Mb 轴、不压缩缺失位置、`origin=lower`、`extent=(2.5,195.5,2.5,195.5)`、`coolwarm_r`。候选和 reference 都先生成 raw Euclidean distance matrix，再在相同 common pair 上计算 pooled RMS 和 normalized matrix。

与旧 NPZ 的有限项比较：

- 020：raw max absdiff `0`；normalized max absdiff `0`；panel order 都是 `[chr1(mat), chr1(pat), c01b, c01a]`；scale 完全相同。
- 022：raw/common-raw/normalized max absdiff 均为 `0`；panel order 和 scale 完全相同。
- 029 C0 bundle2：raw/common-raw/normalized max absdiff 均为 `0`；panel order 完全相同。

因此，旧/新脚本没有在 row/column 对应、中心化、坐标轴、mask、float 精度或矩阵计算上制造 C0/020 形状差。旧色标只影响视觉映射：

- 020 旧 `vmax=2.4009829719`，统一审计色标 `2.4182068229`；normalized color-coordinate 最大差 `0.00712257`。
- 022 旧 `vmax=2.5102779072`，该色标同时覆盖 continuation 与 rejected FDG proposal；统一差异最大 `0.03656615`。
- 029 C0 bundle2 旧 `vmax=2.3456143084`，统一审计色标最大 `2.4182068229`；最大差 `0.02967981`。

这些量化只表示颜色映射改变，不能把全局 `vmax` 变化当作 Spearman/矩阵形状变化。旧 020 NPZ 的 stale metadata 是 `colormap=coolwarm`，而其 `render_revision-20260913_094830.json` 明确记录最终 PNG 已改为 `coolwarm_r`、6x6、300 dpi、共同 mask/RMS/color norm 未变；这属于 provenance 字段不一致，不是数值或排列错误。

进一步的同一 17,578 pair 形状比较（每个候选 panel 先按自身 pooled RMS 归一化、再比较 rank）如下：

- 020 vs 022：两个固定 reference 对齐 panel 的 Spearman 为 `0.99430`、`0.99180`，normalized RMS 差为 `4.55%`、`4.81%`。
- 020 vs C0 bundle1：Spearman `0.67602`、`0.63213`，normalized RMS 差 `33.67%`、`46.67%`。
- 020 vs C0 bundle2：Spearman `0.36735`、`0.62807`，normalized RMS 差 `44.32%`、`52.84%`。
- 020 vs C0 bundle3：Spearman `0.47747`、`0.65527`，normalized RMS 差 `39.65%`、`45.12%`。

这直接区分了尺度/色标效应与真实 shape difference：020→022 的结构形状高度一致，020→029 C0 的形状明显不同。

## 全 20chr 的 020 vs C0 描述

以下是同一既有 mask 上的逐染色体差值 `C0 - 020`；没有注册新显著性检验，没有 p-value/CI，也不能把历史 020 与 C0 当作配对初始化。

| 比较 | macro delta matched | macro delta contrast | matched 胜出 / 20 | contrast 胜出 / 20 |
|---|---:|---:|---:|---:|
| C0 bundle1 - 020 | -0.01731854 | -0.01363149 | 7 | 8 |
| C0 bundle2 - 020 | -0.01292096 | -0.02043332 | 8 | 8 |
| C0 bundle3 - 020 | -0.01104117 | -0.03131856 | 9 | 9 |

宏平均原值：020 `matched=0.49032573, contrast=0.13295638`；C0 bundle1 `0.47300719, 0.11932488`；bundle2 `0.47740477, 0.11252305`；bundle3 `0.47928457, 0.10163782`。这些 genome-wide macro 不能替代 chr1 读数；chr1 仍显示 020 的 matched/contrast 明显高于三条 C0 终点。

逐染色体的所有值和 delta 在 `c0_vs_020.tsv`；100 行 endpoint 独立复算与既有 R2 对照在 `r2_independent_vs_locked.tsv`。

## 025 x0 事后诊断

标准 x0 读数已经从已有的四个 x0 rho 重新派生，每个 seed×chromosome 使用 x0 自身 direct/cross 最大者；reference 列固定，tie 仍按 `1e-12` 规则。没有重读原坐标或重算 rho。

| bundle | x0 与 endpoint orientation 不同 | standard endpoint - x0 matched | matched wins | standard endpoint - x0 contrast | contrast wins |
|---|---:|---:|---:|---:|---:|
| bundle1 | 10 / 20 | +0.27911733 | 20 / 20 | +0.08666839 | 19 / 20 |
| bundle2 | 9 / 20 | +0.26150224 | 20 / 20 | +0.08319607 | 15 / 20 |
| bundle3 | 7 / 20 | +0.23176664 | 20 / 20 | +0.06722237 | 16 / 20 |

这项 post-hoc 只证明 025 native x0 到 C0 endpoint 这条路径本身在固定 mask 上改善；它不作为 020/C0 差别“不是画法”的逻辑证据。020/C0 的非画法结论独立来自旧/新 NPZ 一致性和跨候选形状比较（见上文）：raw/normalized 矩阵复现为 0 差，020→C0 的对应 panel rank correlation 为约 0.37–0.68。`c0_seed_and_x0_diagnostic.tsv` 是标准 x0 主表；`c0_seed_and_x0_fixed_endpoint_direction.tsv` 仅保留“沿 endpoint 方向固定”的 secondary trajectory-aligned 读数，不能与标准 x0 R2 混称。chr1 的 bundle2 x0 自身 best-swap 恰好也是 `swapped`，所以对应 x0 图的 chr1 panel 排列与标准 x0 读数一致。

## 分类结论

### 【已证实错误】

- 020 frozen NPZ 的 colormap metadata 与后续实际渲染 revision 不一致：NPZ 写 `coolwarm`，最终 revision 使用 `coolwarm_r`。这是 provenance/metadata 问题，已被旧 README 和 render provenance 说明；没有证据表明它改变了矩阵数值。

### 【合法展示口径差异】

- 不同条件集使用不同的共同色标是合法的展示口径差异：旧 022 的共同色标覆盖 continuation 与 rejected FDG proposal，旧 C0 的共同色标覆盖其五个 variant；本审计则为 020/022/C0 五个 endpoint 加 025 x0 bundle2 使用一个统一色标。差异已被量化，但不应标成错误，也不是 R2 计算错误。

### 【已修正的本次审计草稿问题】

- 初版 post-hoc x0 表曾沿用 paired endpoint orientation 计算并命名 x0 的 matched/contrast；现已改为 x0 自身每染色体四-rho best-swap，tie 规则仍为 `1e-12`。旧固定 endpoint 方向读数仅作为明确命名的 secondary trajectory-aligned 表保留。原 020/029 endpoint R2 没有这个口径问题。

### 【已排除】

- 候选终点轨道错配、A/B 颠倒、局部 swap、字符串坐标比较、grid 排列变化、缺失 bin 压缩、mask 临时扩大、对角/同 bin 混入、float 精度导致的 shape 差异。
- 旧/新 raw、common-raw、normalized 距离矩阵的计算差异；有限元素 max absdiff 为 0。
- 029 C0 四 rho、matched/cross/contrast、双 margin 与既有锁定 R2 记录的不一致；最大 absdiff 仅为浮点舍入量 `2.22e-16`。

### 【实验本身差异】

- 020 是 `5 Mb -> 2 Mb -> 1 Mb random_joint selected`；029 C0 是三个新 native 起点的 1 Mb/480 endpoint，不是 020 的 paired continuation。
- 三个 C0 seed 的 candidate pooled RMS 也不同（0.30049/0.29429/0.28958），且 chr1 的 panel shape 与 020 的 rank correlation 只有约 0.37–0.68；这是真实终点坐标差异，而非统一展示造成的假象。
- 022 与 020 形状和 R2 更接近（chr1 matched/contrast 分别约 0.678/0.148 与 0.670/0.136），说明 continuation 与 C0 native-start 终点不能直接当成同一轨迹。

### 【仍需训练审计解释】

- 评价/展示链只能确认差异存在，不能解释 029 C0 为什么没有达到 020 的 chr1 终点。可能相关的训练路径、native 起点、1 Mb-only 预算、终止状态和优化动力学需要训练侧单独审计；本分支没有新 fit，也没有把 reference 用于训练选择。
- 20chr macro 不能代替 chr1 的宏观图。所有结论仍是单细胞内 linked chromosome measurements，不是生物学复制结论。

## 新审计产物

核心配置、脚本和机器验证：

- `docs/audits/c0-020-display-20260914_021033/config.json`
- `docs/audits/c0-020-display-20260914_021033/audit_c0_020_display.py`
- `docs/audits/c0-020-display-20260914_021033/postprocess_x0_best_swap.py`
- `docs/audits/c0-020-display-20260914_021033/validation.json`
- `docs/audits/c0-020-display-20260914_021033/provenance.json`
- `docs/audits/c0-020-display-20260914_021033/summary.json`
- `docs/audits/c0-020-display-20260914_021033/chr1_shape_comparison.json`
- `docs/audits/c0-020-display-20260914_021033/x0_best_swap_validation.json`
- `docs/audits/c0-020-display-20260914_021033/rerender_short_footer.py`
- `docs/audits/c0-020-display-20260914_021033/render_revision_short_footer.json`

统一数值包和图：

- `docs/audits/c0-020-display-20260914_021033/unified_chr1_distance_matrices.npz`
- `docs/audits/c0-020-display-20260914_021033/unified_chr1_distance_matrices.png/.pdf`：统一 candidate-panel index；主图为下列独立 6x6 图。
- `docs/audits/c0-020-display-20260914_021033/020_historical_selected_chr1_distance_matrices.png/.pdf`
- `docs/audits/c0-020-display-20260914_021033/022_historical_continuation_chr1_distance_matrices.png/.pdf`
- `docs/audits/c0-020-display-20260914_021033/C0_bundle1_seed_chr1_distance_matrices.png/.pdf`
- `docs/audits/c0-020-display-20260914_021033/C0_bundle2_seed_chr1_distance_matrices.png/.pdf`
- `docs/audits/c0-020-display-20260914_021033/C0_bundle3_seed_chr1_distance_matrices.png/.pdf`
- `docs/audits/c0-020-display-20260914_021033/x0_bundle2_chr1_distance_matrices.png/.pdf`

表格：

- `docs/audits/c0-020-display-20260914_021033/chr1_display_index.tsv`
- `docs/audits/c0-020-display-20260914_021033/r2_independent_vs_locked.tsv`
- `docs/audits/c0-020-display-20260914_021033/c0_vs_020.tsv`
- `docs/audits/c0-020-display-20260914_021033/c0_seed_and_x0_diagnostic.tsv`
- `docs/audits/c0-020-display-20260914_021033/c0_seed_and_x0_fixed_endpoint_direction.tsv`

