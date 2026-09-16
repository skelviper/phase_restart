# Next-step R2 独立准备与评价入口

状态：`evaluated_parent_release_complete`。本目录冻结本轮 `M0/M1 × B1/B2 × 2 source` 的 8 个 real endpoint，以及 `P2/N2 × (M0/production-e, M1/production-e, M0/known-e)` 的 6 个 synthetic endpoint；训练阶段未读取 reference/truth，正式评价输出位于 `test_res/038-20260914T143812Z-gpu-m1-formal/evaluation-r2-20260914T153134Z/`。

## 已冻结范围

- P9016 单细胞全部 `1,703,888` contacts，20 chromosomes、40 tracks、1 Mb 5,290 physical beads；metric grid 仍为 `range(3Mb, L, 1Mb)`，numeric bp join、unordered off-diagonal pair。
- Real source 顺序固定为 `consensus_joint_base1103`、`random_joint_base2207`；每个 method × budget 只在两 source 间按最终 `count_nll_per_record` 选择，tie `<=1e-9` 按上述顺序取先者。R2 不参与选择。
- 父侧已冻结 B1 的实际 FG nfev 为 `5/2/1Mb = 306/202/243`，B2 为 `612/404/486`；synthetic 1 Mb FG cap=`243`，scale2 固定。保留共同 `ftol=1e-10`、`maxls=20`；仅关闭 SciPy 依赖变换变量的 `gtol` 早停（`gtol=0`）。canonical C0 raw-y/q accepted callback 用 `gradient_inf <=1e-6` 停止，FG cap 或数值停止据实记账。
- M1 在 `a=(yA+yB)/sqrt(2), b=(yA-yB)/sqrt(2)`、优化 `a,b/2,q`，再可逆还原到物理 `yA,yB`；目标仍是原 C0 物理目标，不强制分离。
- Real mask 只复用 old21condition，未把新 endpoint 加入或缩小 mask。冻结 mask lock SHA 为 `d0c325...e3666c49e`，published manifest SHA 为 `fa1b26...349574acb`，总计 `157,529` common / `176,201` non-diagonal pairs。orientation tie 为 `<=1e-12`。

## known-e 输入

`synthetic_inputs/known_e_worker_input_manifest.json` 是给训练侧的 manifest，不含 truth path 或 truth coordinates。P2/N2 的 exposure 均来自 026 truth NPZ 的唯一 `exposure` 键；archive 只列出 `coordinates.npy`/`exposure.npy`，未读取 coordinates payload。每个向量均为 little-endian float64、length `2645`、finite、strictly positive、mean-one，并与 026 generation metadata 和冻结 exposure SHA 一致。

- P2：`synthetic_inputs/P2_generation_exposure.npz`，文件 SHA `cd2443191826008385911e9008a0b43deae46fde5322fa13f5d5582c927c73b6`，数组 SHA `12b34ae791e0face4fd2d69f36b61de7435f0f91eee584589f4f20cfb15c165f`。
- N2：`synthetic_inputs/N2_generation_exposure.npz`，文件 SHA `ca77e1b4f2f7957bc2c3898b9bdd8fbd99a884c62166918a3bd82777ff99c7c9`，数组 SHA `e2e903d8f9d8d4347897c7ac6bc01e4e65f24d5821ffb6e9e2739da3eb65b34c`。

对应 count snapshot、shared x0、existing production-e C0 layer 的精确路径和 SHA 见上述 worker manifest。`exposure_manifest.json` 和 `exposure_provenance.json` 分开记录了 evaluator-side 来源与读取边界。

## 校准与旧入口

运行：

```bash
source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh
conda activate analysis
python docs/audits/next-step-r2-preparation-20260914T143656Z/r2_entry.py --tiny-check
python docs/audits/next-step-r2-preparation-20260914T143656Z/r2_entry.py --validate-preparation
```

两项当前均为 `PASS`。tiny fixture 只使用自造的两染色体、8-locus 数组，验证：per-chromosome independent whole-swap 不改变 matched/cross/contrast/minmargin；只 swap candidate 时 shape-error 为零；same-shape N2 null 两个 truth rho 列相同且 contrast 恒零；不同 candidate copy 由 `N_only_copy_difference_rms` 识别；tie 和 n/a 对称。

Legacy suite status 已单独记录在 `legacy_suite_status.json`：`PYTHONPATH=. pytest -q tests/test_allele_calibration_evaluator.py docs/audits/multires-r2-preparation-20260914_041826/test_multires_r2.py` 的结果是 `13 passed, 1 failed`。唯一失败测试为 `FrozenContractTests.test_pending_release_is_rejected_before_payload_access`，断言 `assertRaises(r2.PreparationError)` 未触发，因为旧 041826 release contract 当前已不再处于 pending/rejected 状态。这是 nonblocking 的历史 expectation mismatch，不代表本轮 next-step gate 失败，也没有修改旧目录。`preparation_source_hashes.json` 记录了本目录 entry/plotter、config/protocol、依赖及 old-files-modified 状态。

不要调用旧的 `python -m pr.allele_calibration evaluate`：它是 026 的 `4 fixtures × 5 variants` 入口，且旧 CLI/历史 pooled-swap 语义不适用于本轮 `P2/N2 × 3 arms`。本目录 `r2_entry.py` 只绑定现有 `pr.allele_calibration_evaluator.chromosome_metrics`（synthetic）和冻结 `multires_r2.evaluate_endpoint_chromosome`/`pr.r2comparison` mask comparator（real），不修改旧 result/codehash。

## 037/038 schema 映射

037 的 `source/m1_gpu_controller.py` 已只读核对。正式 038 输出的 canonical mapping 为：real `candidate_id=consensus_joint`/`random_joint` 分别对应 source `consensus_joint_base1103`/`random_joint_base2207`；real candidate 坐标来自 `stages/real/<M0|M1>/<B1|B2>/<candidate_id>/coords/final-1m.3dg`，terminal record 是同目录的 `1m.json`。synthetic 坐标来自 `stages/synthetic/<fixture>/<method>/<exposure_label>/coords/final-1m.3dg`，terminal record 是同目录的 `1m.json`；训练 controller 的 `known-generating-e` 在本 evaluator contract 中填写为 canonical `known-e`，两者是同一 arm。`selection.json` 的 `real.per_method_budget` 提供四个 method-budget cell 的 candidate scores、tie order 和 selected candidate；`release_ready_manifest.json` 提供 14 个 final coordinate path/SHA；`terminal_evidence.json` 提供全局训练 terminal gate；`provenance/source_hashes.json` 与 `selection.json` 的路径/SHA 也必须分别写入 `training_controller.source_manifest_*` 和 `selection_evidence`。只有将这些记录连同 source/raw snapshot SHA、每个 endpoint 的 terminal-record SHA 和本目录 code SHA 写入并锁定新 release contract 后，才可进入评价。


```bash
python docs/audits/next-step-r2-preparation-20260914T143656Z/r2_entry.py \
  --validate-release <parent-locked-release.json>
python docs/audits/next-step-r2-preparation-20260914T143656Z/r2_entry.py \
  --evaluate-released <parent-locked-release.json> \
  --output-dir docs/audits/next-step-r2-preparation-20260914T143656Z/release/evaluation-r2-<UTC>
```

`--validate-release` 只读 JSON metadata，不打开 coordinate/reference/truth payload。`--evaluate-released` 先 hash 全部 14 candidate、historical 036 C0 anchor、21 old-mask inputs、source/terminal/selection/code，再 hash/open reference 和 synthetic truth；任何锁不通过立即 abort。评价输出不含 R1/R3、p-value、CI，也不按 R2 选 model/endpoint/hyperparameter。

完成后将生成：

- `r2_real_8endpoint_x20chr.tsv`、`r2_real_summary.tsv`、`r2_real_paired_deltas.tsv`、`r2_real_paired_summaries.tsv`；paired 表分别记录每个 source 的 `M1-M0` 与 `B2-B1 (double-base)`，含 mean/median/positive/negative/zero/n_a。
- `r2_synthetic_6endpoint_x20chr.tsv`、`r2_synthetic_summary.tsv`；P2 为逐 chr whole-swap R2，N2 另外输出 `synthetic_N2_n_only_diagnostic.tsv`，含同 x0 initial、各 final、以及 production-e 的 M1-M0。N2 的 contrast 恒为零，`N_only_copy_difference_rms` 只作 known-null 诊断，不能作正分数；shape-error 只作内部校准辅助。
- `mask_exact_comparison`、hash gate、evaluation manifest、terminal evidence、arrays/metadata 和中文 README；主 real 图为四 panel、每 panel 3 inch、300 dpi、7 pt，含四个 count-selected real conditions 与灰色 `036 C0` 历史锚点；两 source 各有补图。chr1 heatmap 不是主交付门。

## 038 评价交付

- Real 160 rows：`test_res/038-20260914T143812Z-gpu-m1-formal/evaluation-r2-20260914T153134Z/r2_real_8endpoint_x20chr.tsv`；paired method/budget 表与 036 C0 anchor 同目录。
- Synthetic 120 rows：`r2_synthetic_6endpoint_x20chr.tsv` 与 `r2_synthetic_summary.tsv`。N2 reader-facing 20-chromosome 汇总为 `synthetic_N2_n_only_summary.tsv` / `.json`；该汇总只用于 known-null 的 `N_only_copy_difference_rms` 诊断，明确报告同 x0 initial、各 arm final/final-minus-initial、M1-production-e minus M0-production-e、M0-known-e minus M0-production-e 及正/负计数。
- 历史锚点接线审计为 `historical_anchor_r2_parity.json`：038 M0/B1 random 与已发布 036 C0 random 在同一 old21 mask 下逐 chromosome 对应字段 `20/20`，数值 `atol=1e-12` 最大差 `0.0`，exact mismatch `0`。
- 主图 tick 的 render-only 更正记录于 `plot_render_correction.json`；仅覆盖主 4 个 metric panel 与主 four-panel 的 PNG/PDF，source-stratified 补图、TSV、数组和 R2 数值未改变。
