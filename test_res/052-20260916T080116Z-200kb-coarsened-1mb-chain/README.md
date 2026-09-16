# 052 — 新链 40Mb→10Mb→5Mb→2Mb→1Mb→500kb→200kb 与 200kb 粗化 1Mb 的 same/cross 比较

单细胞 P9016，20 条染色体、40 条拷贝轨迹。本 run 只做一件事：跑**一条**新的 coarse→fine 链，
把最终 200kb 物理坐标按原基因组 1Mb bin 取算术均值粗化，然后在 046 冻结的 old21 支持上，
用 signed Spearman 比较三个 1Mb 版本。

## 交付物

| 文件 | 内容 |
| --- | --- |
| `plots/same_cross_spearman_boxplot.png` | 唯一图：三版本 same(蓝)/cross(橙) 箱线 + 每染色体散点，图例在图外 |
| `eval/per_chromosome_spearman.tsv` | 20 条染色体 × 3 版本 × 4 个 rho + same/cross/contrast |
| `eval/summary.json` | same/cross/差值的均值、中位数、n_defined |
| `config.json` | **拟合前**写下的冻结配置（层、预算、objective、粗化、评价） |
| `logs/*.terminal.json` | 每层终态 / FG / last accepted / 退出码与哈希 |
| `coords/new-chain/<stage>.{npz,3dg}` | 新链各层端点（40Mb…200kb） |
| `coords/200kb-to-1Mb/coarsened1Mb.{npz,3dg}` | 粗化后的 1Mb 候选（第三版本） |
| `inputs/prep_manifest.json` | 各层规模、raw 预算、grid 审计、折叠与折叠一致性检查 |

## 冻结方法

* 输入：`inputs/P9016.snpfree.pairs.gz`，SHA `f37ed9cc…c9aa`，全部 **1,703,888** 条记录，无缩水；训练侧不读 phase。
* objective：`G/full-J/raw`，fixed production `e`，`bend=0.01`，count/bond/repulsion/p_prior 权重均 1
  （p_prior 在实现内部已乘 1e-4）。`l0=(2*n_loci)^(-1/3)`，`r0=2*l0` 由每层 data 读出。
* 优化器：普通 raw L-BFGS（非 multiscale），`ftol=0`、canonical `gtol=1e-6`、`maxls=20`，
  取 last callback-confirmed accepted endpoint。
* 每层一段、不续跑；FG 预算：40Mb 200 → 10Mb 200 → 5Mb 612 → 2Mb 404 → 1Mb 486 → 500kb 200 → 200kb 100，**总 2202**。
* 起点：014 无标签 random 盲源（seed 2207）在真实 40,000,000 bp 网格上的展开（78 loci，p=0.75，零优化）；
  其后每层由**本链上一层端点**按冻结 `warm_start_from_layer` 规则 prolongation，5Mb 处**没有**重置 baseline 起点。
* 输入层：1Mb 复用 045 冻结 aggregate（SHA 核对），20/10/5/2Mb 复用 051 冻结 aggregate（SHA 核对），
  40Mb 由冻结 1Mb 精确整数折叠得到，500kb/200kb 由原始 7 列无标签 pair 的**精确 bp**重新聚合
  （绝不把粗层 counts 拆成细 bin）。全部 20 条染色体保留完整 header grid，训练珠子不删。

## 结果（单细胞，20 条染色体，n_defined=20/20，无 NA）

| 版本 | 自身 FG 预算 | same 均值 / 中位数 | cross 均值 / 中位数 | same−cross 均值 |
| --- | --- | --- | --- | --- |
| Baseline | 1502 | 0.6049 / 0.6213 | 0.3776 / 0.3742 | 0.2273 |
| Extra levels | 1902 | 0.6266 / 0.6594 | 0.3847 / 0.4092 | 0.2419 |
| 200 kb → 1 Mb（本链） | 2202 | 0.5851 / 0.5587 | 0.3646 / 0.3461 | 0.2205 |

三条版本 20/20 条染色体都满足 same > cross。新链相对 baseline 的 same 均值差 −0.0198（逐染色体 7/20 更高），
相对 extra-levels 为 −0.0415（7/20 更高）。三个版本的 FG 预算不同，不是等成本比较，此处只作描述，不做显著性或
"更优 / 更差"的结论。

自检：用本 run 的评价实现对冻结 baseline 复算得到 same=0.6049 / cross=0.3776，与 046 已发布的
Spearman 宏平均逐位一致；用参考自身当候选时 20 条染色体 rho 全为 1.0。


## 诚实标注

* 每层都是为速度故意压低预算的，终态一律是 `budget_not_converged`（`fg_budget_exhausted`），
  **不宣称收敛**；退出码 0 只表示进程正常结束。
* 三个版本的 FG 预算**不相同**（baseline 1502 / extra-levels 1902 / 新链 2202），
  这不是等成本比较。
* 只描述单细胞：20 条染色体是同一细胞的关联测量，40 条 copy tracks 不是独立重复；
  无 bootstrap、无显著性、无新 null、无 R1/R3。

## 粗化规则

每 chr、每 copy，按 key = `floor(original_bp / 1_000_000)` 把 200kb 物理 xyz 直接取算术均值；
每个完整 1Mb bin 恰好 5 颗 200kb 珠，末端不足的按实际 1..4 颗平均（部分 bin 数由各 chr 真实长度推得并逐位核对）。
不取每第 5 颗、不平均距离矩阵、不重优化、不重新居中或缩放；保留真实 1Mb bp 坐标与缺口。

## 评价冻结

`Frozen support` = 046 `evaluation_final/results/frozen_legacy_mask_snapshot.npz`（SHA `9c551c6a…28d9`）的
`positions` / `pair_i` / `pair_j` / `common`：2447 个有效 loci、157,529 个 intra pair、20 条染色体，
三版本共用同一支持。候选索引 `global_index = offset(chr) + bp // 1_000_000`，
`offset = cumsum(ceil(chromosome_length / 1e6))`；距离是粗化坐标的欧氏距离。
指标为 signed Spearman rho（平均秩）。每 chr 用 4 个 rho 的最大配对和选一次 whole-chr A/B swap
（本次按 Spearman 选），same = 匹配两 copy 的 rho 平均，cross = 互换两 copy 的 rho 平均。
NA 明确保留，不做 nanmean 后宣称 n=20。

参考 `data/P9016.1m.3dg.gz`（SHA `1ca82ef4…cea29`）只在本 run 三个版本坐标与哈希写完之后才打开，
不参与初始化 / 训练 / 停止 / 选择。

## 复现

```bash
cd test_res/052-20260916T080116Z-200kb-coarsened-1mb-chain
conda activate analysis
python code/prep_inputs_052.py                 # 各层 aggregate + 40Mb 起点 + 一致性检查
bash code/run_chain.sh                         # 40Mb…200kb 链（每层一个进程）
python code/coarsen_200kb_to_1mb.py            # 200kb -> 1Mb 算术均值粗化
python code/evaluate_spearman.py               # 打开参考，算 same/cross Spearman
python code/make_plot.py                       # 唯一 PNG
```
