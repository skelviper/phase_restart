# 054 正确共享链：40→20→10→5→2→1 Mb（→500 kb→200 kb→1 Mb 粗化）

内部记录（中文）；图内文案与交付材料英文。

## 为什么有这一轮

用户指出旧 052 链的设计缺 20 Mb 层（40 → 10 → 5 → …），因此它的 1 Mb / 200 kb 端点
**不能**回答"逐步增加分辨率带来什么影响"；053 曾把 052 的结果当作正确链的端点展示。
本轮只新做一条**正确**的共享链，第三组与第四组共用同一前缀，旧 052/053 的端点结果
不被改标签、不改字节。

## 链与预算

| stage | bin | FG cap | 说明 |
| --- | --- | --- | --- |
| 40Mb | 40,000,000 bp (78 loci) | 200 | **复用** 052 已冻结端点，不重跑 |
| 20Mb | 20,000,000 bp (144 loci) | 200 | 由上述 40Mb 端点 prolongation（052 缺的新层） |
| 10Mb | 10,000,000 bp (276) | 200 | |
| 5Mb | 5,000,000 bp (538) | 612 | |
| 2Mb | 2,000,000 bp (1329) | 404 | |
| 1Mb | 1,000,000 bp (2645) | 486 | **第三组端点**（lineage 2102 FG） |
| 500kb | 500,000 bp (5278) | 200 | |
| 200kb | 200,000 bp (13181) | 100 | **第四组**最终细层端点（lineage 2402 FG） |
| 200kb→1Mb | 1,000,000 bp (2645) | — | 逐 bin 算术均值粗化，**第四组评价端点** |

本轮新计算 2202 FG；含复用 40 前缀的 lineage 共 2402 FG；第三组到 1 Mb 为 2102 FG。
1 Mb 只跑一段，任何端点之后不再追加优化。

## 复用的 40 Mb 前缀（reference-free 冻结前缀，非新运行）

* 源：`test_res/052-20260916T080116Z-200kb-coarsened-1mb-chain/coords/new-chain/40Mb.npz`
* SHA256 `905575fed78d381f3effc751bb87b7ac68c9423a9b994cebd6bd9b8bbe9ad152`（与 052 terminal
  的 `artifact_hashes.coordinate_npz_sha256` 一致）；3DG `f81557e0…`；terminal `5b222080…`
* 核对：40,000,000 bp、78 loci、1,703,888 raw records、200 FG、`budget_not_converged` /
  `fg_budget_exhausted`、`last_accepted_endpoint=true`、`reference_opened=false`、`phase_opened=false`，
  起点为 014 无标签 random 盲源（seed 2207、p=0.75）；与本轮同目标/同数据/同权重。
* 登记文件：`coords/reused/40Mb-prefix-registration.json`；**没有伪造新的 40 Mb 终态**。
* 20 Mb 层由这个 40 Mb 端点 prolongation，**没有**用 051 已优化的 20 Mb 替代。

## 输入（只读复用，不重建、不跑 preflight/smoke）

`inputs/reused_aggregates.json` 逐层记录 path/SHA256/bin_size/sumceil(header/bin)/raw records：
40Mb=78、20Mb=144、10Mb=276、5Mb=538、2Mb=1329、1Mb=2645、500kb=5278、200kb=13181，每层
1,703,888 records 守恒。20/10/5/2Mb 来自 051，1Mb 来自 045，40Mb/500kb/200kb 来自 052。

## 冻结设置

固定 G / full-J / raw L-BFGS（不用 ms 预条件）、固定 production e；count/bond/repulsion/p_prior
权重 1、bend 0.01（p_prior 内部 1e-4）；p/q 经 raw_y 与 q 优化（p 由 q 导出）、球域、ftol=0、
gtol=1e-6、maxls=20、pair_block=262144、tile_rows=32；cap 到即停、保留 last accepted、
如实写 `budget_not_converged`。跨层一律用冻结 `warm_start_from_layer`（真实 bp；seed =
3301 + bin_size//1e6 + 2207；只扰动新增/重复 loci；径向 clip；保留上一层 frame/scale），
不在 5 Mb 重置初始化。phase 列禁止读取；reference 只在候选/粗化写完哈希后评价。

## 运行

```bash
conda activate analysis
python code/register_reused.py      # 40Mb 前缀 + 各层输入核对与登记（拟合前）
bash   code/run_chain.sh 20Mb       # 20Mb -> ... -> 200kb，每层一个进程
python code/coarsen_200kb_to_1mb.py # 第四组：200kb -> 1Mb 算术均值粗化
python code/evaluate_chain_spearman.py  # 打开 reference，算四组 same/cross Spearman
python code/make_figures.py         # 两张 PNG + panel 标注表
```

第一次启动（nohup 于工具 shell 内）在 5 Mb 刚开始时被进程组清理杀掉，未产生任何 5 Mb 端点；
被中断的中间文件已移入 `logs/aborted-kill-20260916T0912Z/`，5Mb..200kb 以受管后台任务从
`coords/new-chain/10Mb.npz` 重新完整计算。

## 评价与交付

四组全部是 1 Mb 坐标，共用 046 `frozen_legacy_mask_snapshot.npz` 的 positions/pair_i/pair_j/common
（2447 个有效 loci、157,529 个 intra pair、20 条染色体），index = offset + bp//1Mb，signed
Spearman（平均秩），每 chr 按 4 个 rho 的最大配对和做一次整 chr A/B swap，same/cross 各两 copy
平均，每箱 20 个点，NA 不静默删。既有两组（046 baseline、051 extra）必须逐位复现冻结均值。

* `plots/four_group_same_cross_spearman.png`：四组（Baseline / Extra 20→10→5→2→1 /
  共享链 40→20→10→5→2→1 / 正确全链 200 kb→1 Mb）same 蓝、cross 橙箱线图，预算 1502/1902/2102/2402。
* `plots/chr1_distance_matrices.png`：chr1 2×4 共同位点距离矩阵，列序
  Reference / Extra / 共享链 1 Mb 端点 / 全链 200 kb 粗化 1 Mb；每列一个"两 copy 合并中位距离"
  展示尺度；缺失灰、有效对角 0；每个候选面板顶部写该行展示 copy 的 Spearman R same/cross。
* 表：`eval/per_chromosome_spearman.tsv`、`eval/summary.json`、`eval/panel_annotations.tsv`、
  `eval/chr1_matrix_scales.json`、`coords/reused/coarsening_manifest.json`。

范围：单细胞、20 条染色体为关联测量、40 条 copy 轨迹不是独立生物学重复；无 bootstrap、
无显著性、无新 null、无 R1/R3。四组预算不等，比较不是等成本比较。
