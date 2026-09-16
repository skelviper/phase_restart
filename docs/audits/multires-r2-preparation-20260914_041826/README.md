# P9016 多分辨率 R2 预备

状态：`prepared_only` / `evaluation_not_run`。本目录只冻结统一 GPU 主实验的 10-endpoint R2/display 协议、未来 release contract 和 metadata-only release gate；本目录没有读取新训练坐标、reference 3DG、pairs/rawphase，没有 fit，也没有调用 native。033 的 C0 CPU 结果与旧 020 CPU 结果仅作为独立历史锚点；CPU 034 已停止，统一 GPU 30-stage 方案正在做精度/速度预检，R2 尚未运行。

## 范围

- 变体：`C0`、`C1`、`C2-map`、`C2-free`、`C3`。
- 两个原始 blind source：`consensus_joint(base1103)` 与 `random_joint(base2207)`，共 10 个新 endpoint；**全部 10 个都是 40-track 双 copy 模型（`n_copies=2`）**。这里的 `consensus_joint` 只表示 consensus-derived initialization，不是 014 的 single-copy consensus baseline；所有新条件都计算四 rho 和 dual margins。两种 source 不是 IID seed，也不是生物学重复。
- 训练输入固定为 1,703,888 records、20 chromosomes/40 tracks、origin 0、最终 1 Mb/5,290 beads；三层为 5 Mb×300、2 Mb×200、1 Mb×240，首层 `p=0.75`，后续 `carryq`。统一 GPU 主实验从原 014 起点完整运行 `5 variants × 2 sources × 3 layers = 30 stages`；GPU C0 的 numeric no-reference gate 只约束 GPU 主实验。033/020 CPU 不充当 GPU C0 组。
- 每个 variant 只在两个 source 间按训练侧 `count_nll_per_record`（原 020 final 字段 `count_nll_normalized`）选择代表；该值含 diag 常数、不含 priors。选择 tie 使用独立的 `tie_tolerance_per_record=1e-9`，按预注册顺序 `[consensus_joint_base1103, random_joint_base2207]` 取先出现者；R2 orientation tie 仍独立固定为 `<=1e-12`。R2 不重新选择、不使用 reference 参与训练选择。

## 后端决策冻结

028 的同一 scoped benchmark 约为 CPU `3504 s`、GPU `58.9 s`，因此本轮 30 stages 统一使用 GPU backend，避免把 CPU/GPU 混在 10 个主 endpoint 中。GPU endpoint 不要求与 CPU 逐位一致；CPU 033/020 只保留为独立历史锚点，GPU C0 与 CPU 020 的差异只能作为 diagnostic，不能作为 GPU training、release 或 label-free selection 的拒绝条件。所有主 endpoint 和历史锚点仍使用同一 20-chromosome common-mask 评价口径；mask exact 与旧 020 到 published 20-chromosome parity 仍是硬验收。


位置是 `range(3Mb, L, 1Mb)`，numeric bp join，所有距离和派生值 `float64`；每条染色体使用 unordered off-diagonal pair。固定 reference columns 为 `chrN(mat)` 和 `chrN(pat)`，每个候选每条染色体只做一次 whole-chromosome best-swap：

```text
 direct  = (A_mat, A_pat, B_mat, B_pat)
 swapped = (B_mat, B_pat, A_mat, A_pat)
 matched = (a+d)/2
 cross   = (b+c)/2
 contrast = matched - cross
 margin_mat = a-b
 margin_pat = d-c
 minmargin = min(margin_mat, margin_pat)
```

`direct`/`cross` 差值 `<=1e-12` 时标记 unresolved，`contrast=0`，named margins 为 n/a，且不给 both-positive credit。少于 20 common pairs、nonfinite、constant candidate/reference distance 均为 n/a；两-copy 不能用单一有效 copy 平均。这里的量称为 Spearman-derived R2 geometry readout，不称为统计学 R²。

共同 mask 只能复用已发布的 029 21-condition mask（manifest SHA256 `fa1b26c834c173604c21f954d494cece8e053dd970e7cb6111f561b349574acb`）。新 endpoint 不进入 mask，mask 也不能因新 endpoint 缺失而缩小；未来 evaluator 会在所有 release hash 通过后，使用旧 21 条 mask input 和 reference 重建 pair index，并逐染色体核对 `mask_lock.json` 的 20 个 pair count。

## 发布合同

`release_contract.json` 是父侧未来填写的模板，当前所有新 endpoint/source/selection/terminal 的真实路径和 SHA 均为 `pending_parent_release` 或 `null`，没有伪造 digest。有效 release 必须提供：

1. 10 个 GPU endpoint 的 variant/source 标识、终态记录、坐标路径与 SHA，失败 arm 也要保留 terminal failure reason；
2. 两个 blind source 的 raw source/snapshot hashes；
3. complete GPU terminal evidence（10/10 endpoint、30 stages、无 active job）；
4. train-only `count_nll_per_record`（报告为 final `count_nll_normalized`）selection proof（五个 GPU variant，各只比较两个 source，reference/R2/phase 均未读）；
5. GPU C0 两轨 numeric no-reference gate `PASS`；
6. config、protocol、mask、evaluator、plotter 和 release/source 的 hashes。

只有这些锁全部通过，未来 evaluator 才允许 hash/read reference `data/P9016.1m.3dg.gz`（冻结 digest `1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29`）。evaluator 自身不 fit、不调用 native、不读取正在写的训练坐标。

## GPU 发布 gate

`build_multires_release.py` 已是 schema-aware metadata-only gate：它锁定 035 source 中的 `gpu-multires-selection-v1` / `gpu-multires-terminal-evidence-v1` 成功 payload 形状，默认等待正式 output `test_res/036-20260914T064651Z-gpu-multires`。它只解析 `selection.json` 与 `terminal_evidence.json` 的 metadata，校验 10 candidates、5×2 source、30-stage audit、CUDA probe、selection criterion/tie 和路径/SHA 字段格式，不打开 coordinate bytes 或 reference；036 terminal 未完成时返回 blocked，不增加训练等待：

```bash
python docs/audits/multires-r2-preparation-20260914_041826/build_multires_release.py --check
```

当前 035 GPU preflight gate 已由父侧验收为 `PASS`（`terminal_prep.json` SHA `709d5531...568c03`）。正式 release 仍必须等 036 的 10/10 endpoint、30 stages、source/selection/terminal hashes 和无 active job；CPU byte identity 与 CPU020 R2 match 明确不属于放行条件。

GPU controller 的 complete terminal/hash gate 通过且 schema 已被父侧锁定后，才允许显式实现并运行 `--build`：它必须校验 10 个 GPU endpoint、30 个 stage、source/selection/terminal/code hashes，再生成 evaluator 使用的 `release_contract.json`。在此之前不创建 release，不读取坐标/reference。source 映射固定为 `consensus_joint -> consensus_joint_base1103`、`random_joint -> random_joint_base2207`，所有主 endpoint `n_copies=2`；CPU 033/020 只作为历史锚点，CPU 034 不进入 release。


未来 release 后生成：

- 全 `10 GPU endpoint × 20 chr` 的四 rho 和 `matched/cross/contrast/minmargin` 长表；
- 五个 GPU variant 的 count-selected representative 表；
- 主 2×2 图包含五个 GPU 条件和一个灰色、明确标注 `020 / C0 CPU` 的历史锚点；该锚点不是 GPU endpoint、不是生物学重复，也不进入 GPU 优化选择；
- source 分层图只展示五个 GPU 条件，标签为 `consensus-derived initialization` 和 `random-derived initialization`；不使用 `consensus baseline` 或 single-copy 语义；
- source 内每条染色体 `variant - C0` 四 metric delta TSV；
- mask/input/source/selection provenance、plot arrays NPZ、PNG/PDF、validation、terminal evidence、`post_release_acceptance.json` 和中文 README；
- 图为 2×2、每 panel 3 inch、300 DPI、统一 7 pt、色盲友好条件色、染色体点和明确 n/a 计数；不产生新 p-value/CI，不生成 dummy/simulated boxplot。

`multires_r2.py` 是 gated evaluator/core math，`plot_multires_r2.py` 是只读 evaluation result 的绘图入口，`test_multires_r2.py` 只做轻量 synthetic/config contract checks。

## 发布后的后续验收

真实 release 通过 candidate/mask/terminal/selection/source/code/protocol hash gate 并打开 reference 后，evaluator 还必须运行 `post_release_acceptance()`：

- 用冻结 `pr/allele_r2.py`（SHA `56dacfe9a3edc13db4804fa7d5ce7c40b9401a1d1dd62ddbaa66d0d6ad5b1672`）的原 `_load_coordinates`，以及 SHA `2979118aed25ac571d8f476eec4f6ec7687649f5a3e0fc43618194a293b4ed6e` 锁定的 `pr.r2comparison.build_common_mask`，独立重建 21-condition mask；对每条染色体逐项 exact compare `positions`、`pair_i`、`pair_j`、`common` bits，不能只比较 pair count。
- 只读取已发布的 029 `r2_per_chromosome.tsv`（SHA `5e578b5ef8876f57866f9dfa325a8738adab70ebc4913498c570cc2b07883ed1`）中 `v1_original_random_joint` 的 20 行；在同一冻结 mask 下比较新实现四 rho、direct/swapped、matched/cross/contrast、两 named margins 和 `minmargin`，数值 `atol=1e-12`。
- 明确禁止使用可能来自不同 mask 的 020 原始 evaluation 数值。旧 029 `v1_original_random_joint` 在同一冻结 mask 下的 20-chromosome parity 是硬验收；若锁定的 GPU C0 representative 是 `C0-random_joint_base2207`，则额外记录其与 CPU 020/`020 / C0 CPU` 锚点的逐染色体比较，但该 CPU↔GPU 差异仅为 diagnostic，不能拒绝 GPU training、GPU endpoint release 或 GPU label-free selection；若不是 random，则记录 `NOT_APPLICABLE`。

hook 输出 `post_release_acceptance.json`，失败时阻止正式 evaluation 完成。当前 hook 不执行，不读取坐标/reference。

## 当前证据

当前仅能验证 GPU preparation schema、冻结 mask metadata 摘要和 synthetic metric invariants；033/020 CPU 证据只作为历史锚点，CPU 034 已停止，统一 GPU 30-stage controller/output schema 尚待 child 终态。generic adapter 尚未 build，也没有真实 R2 数值、箱线图或代表选择后的 evaluation 结果。完成状态必须保持 `prepared_only/evaluation_not_run`，等待 GPU schema/path 与完整 terminal/hash release。
