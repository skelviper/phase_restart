# 053 四版本 Spearman boxplot + chr1 distance matrices（追加绘图，不拟合）

用户追加请求的执行记录。**只读取既有端点结果做评价/画图，没有重新拟合、没有追加优化、没有其它实验**；
046 / 051 / 052 目录字节未改动。

## 两张图（唯一图形交付）

| 文件 | 内容 |
| --- | --- |
| `plots/four_way_same_cross_spearman.png` | 四版本 same/cross signed Spearman 双色箱线图；每盒叠加 20 条染色体的实际点；顺序 Baseline(5→2→1) / Extra levels(20→10→5→2→1) / New chain 1 Mb(40→10→5→2→1) / 200 kb→1 Mb(完整新链粗化) |
| `plots/chr1_distance_matrices.png` | chr1 距离矩阵 2 行 x 4 列；行 = matched to reference mat / pat；列 = Reference 1 Mb / New chain 1 Mb / New chain 200 kb / 200 kb→1 Mb 粗化 |

规范：`coolwarm_r`、3 英寸基础面板、300 DPI、统一 7 pt 文字；图例置于坐标区外，不遮挡数据。

## 固定评价口径

- 支持集逐字取自 `046/evaluation_final/results/frozen_legacy_mask_snapshot.npz`（SHA `9c551c6a…`）：
  **2447 个有效 loci / 157,529 个 intra pair / 20 条染色体**，四个版本共用同一分母，不静默缩范围。
- 距离 = 候选 1Mb 坐标欧氏距离；指标 = 有符号 Spearman rho（平均秩）；每 chr 用 4 个 rho 的最大配对和
  选择一次 whole-chr A/B swap；same 蓝、cross 橙；NA 明确保留（本任务四组均 20/20 defined）。
- 观察单位是染色体（同一细胞的 20 个关联测量），不是生物学重复；四个端点均 `budget_not_converged`，
  FG 预算 1502 / 1902 / 1902 / 2202 不相等，比较不是等成本比较。

## 数值（`eval/summary.json`，均值 / 中位数，n_defined）

| 版本 | same mean | same median | cross mean | cross median | same−cross mean | n_defined |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (1502 FG) | 0.604876 | 0.621264 | 0.377608 | 0.374225 | 0.227268 | 20 |
| Extra levels (1902 FG) | 0.626619 | 0.659439 | 0.384737 | 0.409241 | 0.241883 | 20 |
| **New chain 1 Mb (1902 FG, 新增)** | **0.587919** | **0.573363** | **0.399012** | **0.392941** | **0.188907** | 20 |
| 200 kb → 1 Mb (2202 FG) | 0.585112 | 0.558666 | 0.364629 | 0.346139 | 0.220483 | 20 |

既有三版本与 `052/eval` 逐 chr 逐值核对一致（420 个数值 + summary mean/median，最大绝对差 `0.0`）；
新增 1Mb 端点与 200kb→1Mb 粗化端点之差：same `+0.002807`、cross `+0.034383`、same−cross `−0.031576`
（20 条染色体上的端点级差异，未做任何检验）。

## chr1 矩阵的口径（`eval/chr1_matrix_scales.json`）

- 候选面板各用**原生完整 grid**：1Mb 面板 196 个 bin，200kb 面板 978 个 bin；`pcolormesh` 用真实 bp
  bin edges（末端 edge 截到真实 chr1 长度 195,471,971 bp），四列同一 x/y 范围，978 个 bin 不硬拉成 196 个。
- 参考只有 1Mb 真坐标（chr1 mat 189 / pat 190 个 bin 有效），缺失 bin 保持 NaN 灰色，不填 0、不外推；
  对角线仅有效点为 0。200kb 层不存在真参考，因此**不直接对 1Mb 参考打 rho**。
- 行映射：直接 1Mb 用该候选 chr1 最佳 swap（swapped → copy B = mat, copy A = pat）；200kb 与其粗化 1Mb
  共用粗化候选的 chr1 最佳 swap（同为 swapped），保证两列显示同一 copy。
- 尺度：参考无绝对单位校准，直接 1Mb 与粗化 1Mb 各用两个 copy 合在一起、冻结 chr1 共同 pairs 上的一次
  最小二乘正尺度 s = (d·r)/(d·d)（各 35,156 pair；s = 2.9668 与 2.9690）；200kb 与粗化共用后者一个 s；
  绘图值 = s·d，参考尺度 1。**这只是展示缩放，不改变 Spearman**；不做每 copy 缩放或每格独立色标
  （八格统一 vmin 0 / vmax 4.192）。

## 文件

- `code/four_way_paths.py` 路径/常量/哈希核对；`code/evaluate_four_way_spearman.py` 四版本评价（沿用
  052 逻辑，只新增候选）+ 与 052 的一致性核对；`code/make_figures.py` 两张图。
- `eval/candidate_hashes.json`（先写盘再读参考）、`eval/per_chromosome_spearman.tsv`、
  `eval/summary.json`、`eval/chr1_matrix_scales.json`；`logs/*.out`。
- 复现：`MPLCONFIGDIR=… python code/evaluate_four_way_spearman.py && python code/make_figures.py`。
