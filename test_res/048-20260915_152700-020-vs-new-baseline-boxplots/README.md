# 048 — 020 vs 新 baseline 逐染色体 Pearson boxplot（内部记录）

## 复现命令

```bash
cd /mnt/ssd/zliu/phase_restart
MPLCONFIGDIR=/tmp/mplcache-048 /mnt/ssd/zliu/miniforge3/envs/analysis/bin/python scripts/plot_3dg_metric_boxplots.py \
  --candidate-a test_res/020-20260913_071841-v1-p9016-joint/selected.3dg \
  --candidate-b test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.3dg \
  --reference data/P9016.1m.3dg.gz \
  --mask test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/frozen_legacy_mask_snapshot.npz \
  --outdir test_res/048-20260915_152700-020-vs-new-baseline-boxplots \
  --label-a 020 --label-b "New baseline" \
  --expect-sha-a afb2d52ae11e342e9b43b3c8042c5581760d177c5e563f2478646ab36b3e7078 \
  --expect-sha-b 4301d4df6e89c1417690599d6687a9e83b18fa37596de5d6c11a911d867d69ea \
  --png-name 020-vs-new-baseline-metrics.png
```

候选 SHA 在打开 reference 前核对：020 = `afb2d52a…`，新 baseline = `4301d4df…`（与冻结值一致）。实际运行 exit code 0，脚本内 elapsed 1.14 s（`date` 包裹约 2 s）。

## 口径与定义

- 观察单位：每条染色体一个等权观察（20 条：chr1..chr19、chrX，同一真实 P9016 细胞；生物学重复只有此细胞）。基因组 pair 不作为独立重复；不做 p 值、不做 bootstrap。
- 支持（分母）：`frozen_legacy_mask_snapshot.npz` 的每 chr `common`（3 Mb 起 1 Mb grid），再取 6 条 track（020 双 copy + 新 baseline 双 copy + reference 双 copy）在同一 pair 上全部 finite 的交集；两个候选共用完全相同支持。**本次 mask common 合计 157,529 对，最终 common 合计 157,529 对，额外剔除 0 对**，逐 chr 数字见 `per_chromosome.tsv` 的 `n_mask_common` / `n_final_common`。
- 指标：每 chr 每候选四个 Pearson rho（A_mat、A_pat、B_mat、B_pat，距离向量 = 1 Mb bin 对的三维欧氏距离）；`direct=(A_mat+B_pat)/2`，`swapped=(A_pat+B_mat)/2`；`same=matched=max(direct,swapped)`，`cross=min`，`contrast=same-cross`。方向按整条染色体统一判定，不逐 pair 挑最大；`|direct-swapped|<=1e-12` 记 tie（取两方向均值、contrast=0；本次 0 个 tie）。
- undefined 保留 NA：本次 3 个指标、两组各 20/20 均有定义，无 NA，无染色体被丢弃。
- 距离指标对 3D 刚体朝向不变，本次未做 Kabsch 配准，也未重画 3D。

## 结果摘要（A=020，B=New baseline；配对差 = A − B，n=20）

| metric | 020 mean / median | New baseline mean / median | 配对差 mean / median | A 高 / A 低 / 平 |
| --- | --- | --- | --- | --- |
| Same (matched) | 0.4691 / 0.4458 | 0.6023 / 0.6202 | −0.1332 / −0.1373 | 3 / 17 / 0 |
| Cross | 0.3409 / 0.3292 | 0.3647 / 0.3507 | −0.0238 / −0.0286 | 9 / 11 / 0 |
| Contrast (same − cross) | 0.1282 / 0.0953 | 0.2376 / 0.2549 | −0.1094 / −0.1472 | 5 / 15 / 0 |

## 限制（必读）

- 两个候选的算法与优化预算不同：020 是 `random_joint` / `minimize_count_nll_per_record` 的 joint 端点；新 baseline 是 G-random 的 full-J 延长（1988 FG，终态 `budget_not_converged`）。本图只是对**已选端点**的描述性对照，**不是**预算控制的因果比较，也不能把差异归因于任一算法或预算。
- cross 也略有不同（新 baseline 略高），因此 contrast 的差异同时来自两个方向；本图不分解来源。
- 本图只报告拷贝身份方向的 Pearson 对照，**不证明 L2**（沿整条染色体的一致拷贝身份可恢复），也不构成 L1/L3 的新证据。
- 未修改 020 / 046 / 047 的冻结数值，未重跑评价、未重建 mask、未调用旧 worker。

## 文件

- `020-vs-new-baseline-metrics.png`：三个并排 panel（Same matched / Cross / Contrast），每 panel 两箱（旧蓝 020、新橙 New baseline），20 chr 配对散点 + 细灰连线，jitter 按 chr 固定且两组一致；boxplot 用中位数/IQR/1.5IQR 须线，`showfliers=False` 避免叠点重复；同/交叉 panel 共享 y 范围，contrast 独立轴；全部点均在范围内，无 quantile 截断；3 英寸基础 panel、300 dpi、7 pt、英文标签，标题/轴/图例未裁切。
- `per_chromosome.tsv`：40 行（20 chr × 2 候选），含四 rho、direct/swapped、orientation/tie、same/cross/contrast 及 mask 计数。
- `summary.json`：每指标两组 mean/median、配对差 mean/median、增/降/平 chr 数、support 与 undefined 记录。
