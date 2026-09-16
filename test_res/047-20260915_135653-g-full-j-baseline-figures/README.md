# 通用 3DG comparison 制图 run

本目录由 `scripts/plot_3dg_comparison.py` 只读生成；不调用训练、评价入口或 legacy worker。图中文字保持英文，本文档记录通用规则。

本目录的 `plots/chr1_1Mb_distance_matrix.png`、`plots/plot_points.*`、`plots/matrices.npz` 与 `validation.json`/`provenance.json` 仍是当时的冻结记录：图中数值与 `manifest.pre_alignment.json` 里的脚本 SHA256 对应当时的 CLI 版本（带 input lock/provenance/validation 那版）。之后脚本已按用户要求精简（只写两张 PNG 与 `metrics.json`）。

按用户要求（“以后都画对齐的版本，047 的图也改一下”），`plots/whole_genome_1Mb_3d_scatter.png` 已用新 CLI 的默认全局刚体对齐版覆盖：它基于本目录原有的 pre-alignment display 坐标再追加一次全局 Kabsch 旋转 R（`copy_pairs` 逐 chr 对应，det(R)=1，无平移/缩放）。新 metrics 是 `aligned_metrics.json`，保存该 R、`det_rotation`、逐 chr orientation 与 fit point count（以及该次 CLI command 与 wall time）；旧 `plot_points.npz/tsv` 仍是未旋转的 pre-alignment 显示坐标，可用 `--no-align` 复现。

旧 `manifest.json`（含覆盖前 scatter PNG 的哈希）对应先前版本，已改名为 `manifest.pre_alignment.json`；其中 scatter PNG 的哈希此后失效，新图以 `aligned_metrics.json` 为准，不重算全量哈希或旧审计。

- command：`/mnt/ssd/zliu/phase_restart/scripts/plot_3dg_comparison.py --candidate /mnt/ssd/zliu/phase_restart/test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.3dg --reference /mnt/ssd/zliu/phase_restart/data/P9016.1m.3dg.gz --chrom chr1 --resolution 1000000 --mask /mnt/ssd/zliu/phase_restart/test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/frozen_legacy_mask_snapshot.npz --candidate-id real-extension-G-full-J --r2-table /mnt/ssd/zliu/phase_restart/test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/r2_per_chromosome.tsv --reference-sha256 1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29 --baseline-id P9016-046-G-random-full-J-1Mb --outdir /mnt/ssd/zliu/phase_restart/test_res/047-20260915_135653-g-full-j-baseline-figures`
- candidate：`/mnt/ssd/zliu/phase_restart/test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.3dg` SHA256 `4301d4df6e89c1417690599d6687a9e83b18fa37596de5d6c11a911d867d69ea`
- wall time：`1.299 s`（读取、解析和绘图；由 validation/manifest 记录）。
- reference：`/mnt/ssd/zliu/phase_restart/data/P9016.1m.3dg.gz` SHA256 `1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29`
- resolution：`1000000` bp；chromosome matrix：`chr1`；orientation：requested `auto`, used `swapped`
- support：frozen mask common bits；matrix support `17,578/18,528` upper pairs
- distance display：reference mat+pat share one raw-median scale; candidate copyA+copyB share one raw-median scale; `coolwarm_r`, actual visible maximum, no quantile clipping.
- 3D display：each dataset is centered once over all finite beads and divided once by its whole-cell RMS; the candidate panel then carries the default whole-genome global rigid rotation R (`aligned_metrics.json`), reference panel unchanged; fixed view `(elev=20, azim=-55)`, equal aspect, shared axis limits, no cross-chromosome lines.
- scatter grid：numeric union，`5,290` points per dataset（candidate grid `5,290`，reference grid `5,000`；reference outside candidate `0`，candidate outside reference `290`）；finite reference/candidate `4,947` / `5,290`；missing `343` / `0`.
- per-chromosome/per-copy finite counts are recorded in `provenance.json`; every grid row remains in `plot_points.tsv`.
- reference parsing and plotting wall time is recorded in `validation.json` and `manifest.pre_alignment.json`.
- missing reference points remain in `plot_points.tsv` and are not imputed; candidate denominator is not reduced to a finite intersection.

Published `real-extension-G-full-J` `chr1` Pearson orientation: `swapped`; A_mat `0.397953491832805`, A_pat `0.793832262469981`, B_mat `0.600481152559272`, B_pat `0.404338252357019`; matched/cross/contrast `0.697156707514626` / `0.401145872094912` / `0.296010835419714`.
Candidate copyA/copyB are arbitrary labels. Matching to reference mat/pat is not a parental-identity claim.

输出：`plots/chr1_1Mb_distance_matrix.png`、`plots/whole_genome_1Mb_3d_scatter.png`、`plots/matrices.npz`、`plots/plot_points.tsv`、`plots/plot_points.npz`；不输出 HTML/PDF。
validation status（prior 未对齐版本，不是新对齐图的终态）：`PASS`；详细结果在 `validation.json`。
