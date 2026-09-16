# POST020 等位基因消融结果报告

**状态：`complete`**  
**用途：最终中文报告；本轮正式真实数据评估只包含 R2**  
**冻结依据：** [`docs/POST020_ALLELE_ABLATION_PROTOCOL.json`](POST020_ALLELE_ABLATION_PROTOCOL.json)，SHA256 `b281037946775ee4271dbf33dd7e4b17dba5ddf4f9bd49eaf618f7c499c5361f`

本报告整理已完成的 protocol、诊断、初始化证据、026 合成校准、029 的 15 项真实 fit（5 个 variants x 3 个 seeds）和仅 R2 评估。R1/R3 没有在本轮新增；所有真实数值均来自已锁定的 029/R2 封存输出，不重新计算 R2，也不用 R2 返选终点或调参。

## 0. 结果导读

本轮没有证明等位基因特异差异恢复提升。C3 提高 matched similarity，但 contrast 和 minmargin 的区间都跨 0；C2 相对 C0 的 matched similarity 与 contrast 下降；5 个 variants x 3 个 seeds 的 15 项真实 fit 全部在固定预算停止、未达到收敛。

## 1. 当前状态

| 阶段 | 当前状态 | 已有证据 | 本稿不声称 |
|---|---|---|---|
| 协议 / 冻结 | complete | protocol JSON/MD/PLAN 已冻结 | 不等于 fit 完成 |
| 022 延续 | complete | 020 endpoint 延至累计 480 accepted | 不是独立重复，不是充分收敛 |
| 023 固定端点诊断 | complete | bash75 与 validator 均 exit 0 | 不提供新的 R2 或 L2 证据 |
| 024 R2 派生 | complete, revision v2 | 既有四 rho 的 6-condition 派生已修复 reference anchor | 不替代 029 的 21-condition R2 |
| 025 原生初始化 | validated | 三 bundle native call、x0、count/graph validation | 不是 joint ablation fit |
| 026 合成校准 | complete | 20/20 terminal，均 `budget_not_converged`；evaluation gate 与 truth-only evaluator 完成 | 不是 real recovery 或 L2 |
| 15 项真实 joint fit | complete | 029 final release/termination/worker audit；15/15 `budget_not_converged` | 不等于充分收敛或 L2 |
| 真实 R2 | complete | 已锁定 21 conditions / 420 rows / 40 primary cells / 120 seed rows | 不等于 allele recovery 或 L2 |

026 的 prepare snapshot 本身记录 `actual_fit_count=0`；那是 prepare 阶段状态。随后 20 个受管 fit 均到达固定预算，评估在 candidate hash gate 通过后完成。两阶段不能混写为一次 fit 计数。

## 2. 历史证据：为什么做这组消融

### 2.1 022 延长没有带来 R2 改善

022 是同一 020 `random_joint` endpoint 的固定 warm continuation，不是新初始化。总目标和 count NLL 下降，但所有结果仍是有限预算产物：

| 条件 | total J | count NLL | R2 matched | R2 cross | R2 contrast |
|---|---:|---:|---:|---:|---:|
| 020 original | 9.61001577125 | 9.59358593129 | 0.490326 | 0.357369 | 0.132956 |
| 022 continuation 480 | 9.60361804197 | 9.58710105238 | 0.489435 | 0.356700 | 0.132735 |
| 022 - 020 | -0.00639772928 | -0.00648487891 | -0.000891 | -0.000669 | -0.000222 |

022 的 R2 contrast CI 跨 0，不能把更低的 label-free objective 写成 allele recovery。两端均为 `budget_not_converged`。

FDG1000full 提案在训练侧原 J 规则下被拒绝；024 v2 的派生读数也显示相对 022 的 matched/cross/contrast 均下降（contrast delta `-0.017458`）。它保留为不利的历史条件，不是后续选择依据。

### 2.2 023 说明 bend 与 sphere map 都不能凭权重或 occupancy 直接排除

结果见 [`023 README`](../test_res/023-20260913_212541-post020-allele-signal-diagnostics/README.md)、[`data_budget.json`](../test_res/023-20260913_212541-post020-allele-signal-diagnostics/results/data_budget.json) 和 [`recommendations.json`](../test_res/023-20260913_212541-post020-allele-signal-diagnostics/results/recommendations.json)。

- 020/022 endpoint `maxR` 为 `0.65161710/0.68574970`，在 `0.90/0.95/0.99` 阈值以上均为 0 个珠子。该结果只排除这些 endpoint 的激活硬墙占用，不能说明 sphere 参数化没有影响。
- 最小 sphere 径向 attenuation 为 `0.43646488/0.38556995`，切向 attenuation 为 `0.75854806/0.72783744`，因此不能用 boundary 不活跃来替代解释 map 的梯度传递。
- weighted bend y-gradient L2 为 `0.01216713/0.01206299`，count y-gradient L2 为 `0.03243707/0.03234493`，比例约 `.375/.373`。bend=.01 虽小仍有作用，因此登记 C1。
- 023 的 total gradient L2 为 `.00620367/.00514436`，最后 20 个 accepted total-J decrease 为 `.00091717/.0003898645`；020 六层和 022 都按预算停止，不能称充分收敛。
- 实际 repulsion 是 `.7*l0=.04017409936`；历史 config 元数据中的 `repulsion_radius_l0=2.0` 是错误字段，仅作错误留痕；native `d_r=2.0` 是独立的 native 参数，不是旧 metadata 的单位换算。保存的 source 与当前 `pr` 有差异时，历史数值以保存的 source 为准。

因此 C1 测试 bend 是否限制等位基因特异形状；C2-map 测试另一种固定的 smooth sphere map；C2-free 测试 direct-x、无硬球的联合参数化/物理域敏感性。C2-free 同时改变 parameterization 和 feasible domain，必须用 `C2-free - C2-map` 辅助解读，不能把 `C2-free - C0` 单独归因于球边界。不同参数化下的 gradient norm 也不能直接比较“谁的约束更强”。

### 2.3 024 v2 的历史 R2 基线

结果见 [`024 revision-v2 README`](../test_res/024-20260913_133109-r2-allele-signal-derived/README.md) 和 [`allele_signal_results.json`](../test_res/024-20260913_133109-r2-allele-signal-derived/results/allele_signal_results.json)。它只派生既有结果，不是新 fit。

- 020 original 相对 Softall：matched delta `-0.055800`，cross delta `-0.100757`，contrast delta `+0.044957`，95% CI `[-0.007420,+0.096044]` 跨 0。
- 020 original 与 022 continuation 都是 20/20 finite；两者 `both-positive=13/20`、`one-negative=7/20`，minmargin `.049677/.047710`。这些 pattern 受 best overall swap 影响，不是成功率。
- v2 修正 swapped 情形为只换 candidate 行，不换固定的 `ref1=mat/ref2=pat` reference 列；matched/cross/contrast/minmargin/orientation/pattern 数值不变，per-reference fields 已重算。
- 024 的旧 6-condition mask 是 `157529` common / `176201` total non-diagonal pairs。它只能作为历史 control，不能冒充 029 的 21-condition 新 R2 输出。

## 3. 冻结设计摘要

真实 cohort 是一个 P9016 生物细胞，训练输入为 `inputs/P9016.snpfree.pairs.gz`，SHA256 `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`。raw accounting 为 `1,703,888 = 1,135,454 cis + 568,434 inter`；same-bin `438,774` 是独立 saturated nuisance，structural raw 为 `1,265,114 = 696,680 cis-offdiag + 568,434 inter`。观测单位是聚合 unordered genomic bin-pair `Cij`，完整 origin-0 1 Mb grid 保留所有 eligible/zero pair：20 chromosomes、40 tracks、2,645 loci/copy、5,290 beads、3,496,690 eligible pairs，real zero `3,009,436`。

### 3.1 五个真实变体

| 变体 | 唯一变更 | exposure / parameterization | 当前状态 |
|---|---|---|---|
| C0 | 新 1 Mb V1 原始 | observed-endpoint / 当前 sphere | complete, 3 bundles |
| C1 | bend `.01 -> 0`，保留 bond | observed-endpoint / same sphere | complete, 3 bundles |
| C2-map | identity-core 平滑球面映射 | observed-endpoint / physical `R<1` | complete, 3 bundles |
| C2-free | direct-x finite-only，无硬球、无 clip/rescale | observed-endpoint / direct Cartesian | complete, 3 bundles |
| C3 | full-grid exposure=ones，保留 endpoint audit/normalizers | ones / same sphere | complete, 3 bundles |

所有其他 count/p-prior/bond/repulsion 权重固定为 1，`epsilon=1e-6`、`r0=2*l0`、repulsion `.7*l0`、`p_floor=1e-4`、p-prior strength `1e-4`、`p_init=.75`。无 bend+free 组合、无 r0/先验搜索、无跨 exposure/prior 的 J 选优。总量为 `5 variants x 3 paired starts = 15` 个真实 joint fit，全部输出都要评价；只在 variant 内按最终 `count_nll_normalized` 选择代表，tie `<=1e-12` 按 bundle ID 打破。

### 3.2 原生起点

三 bundle 的 assignment/native seeds 固定为 `(250101,250201)`、`(250102,250202)`、`(250103,250203)`。assignment 沿 `genome.random_assignment`：cis 两端使用同一随机 copy，inter 两端独立；不读取 phase/p_gen，也不使用 014 seed 代替独立性。每个 bundle 只运行一次 native，`n_iter=1000`、CPU1、`source=NULL`、`target_radius=10`，保留全部 terminal/zero-contact beads。native bend=0，不因 C1 改变。

### 3.3 合成校准

N1/N2 是内部形状相同、空间分离的负例；P1/P2 是不同形状的正例。四个真值相互独立。每个测试样例的 5 个变体共享独立的随机链 x0，不取真值、不扰动真值；C0/C1/C2-map/C2-free 从观测端点重算 exposure，C3 使用 ones，生成 exposure 不传给 worker。生成总数固定为 diag `438774`、cis-offdiag `696680`、inter `568434`，每个测试样例的实际 zero 数单独记录。

合成预算为 `maxiter=80 accepted/maxfun=270/maxls=20`，其他 tolerance 与真实 fit 相同。N 的 contrast 近零是同形状构造的代数事实，必须用 N-only false-split shape error 诊断；不能把它当作方法不能虚构分裂的证明。每条 chromosome 的 candidate 两份 copy 共用同一个 `sC`，truth 两份 copy 共用同一个 `sT`，二者使用同一 finite unordered offdiag mask 的两份 copy pooled `sqrt(mean squared distances)`；保留相对 copy scale，不做 per-copy rescale。P 报告自身 truth 的四个 rho、contrast、margins 和 shape error，不按有利 truth 选择实验。

## 4. 025 原生初始化验证

权威验证文件为 [`POST020_ALLELE_ABLATION_NATIVE_VALIDATION.json`](../test_res/025-20260913_135100-random-native-fullgrid-preflight/validation/POST020_ALLELE_ABLATION_NATIVE_VALIDATION.json)，辅以 [`manifest.json`](../test_res/025-20260913_135100-random-native-fullgrid-preflight/manifest.json) 和 [`config.json`](../test_res/025-20260913_135100-random-native-fullgrid-preflight/config.json)。较早 README 的 `prepared_native_pending` 是 prepare checkpoint；本报告以 validation JSON 的 `status=validated`、`native_calls_total=3` 为当前 native 状态。

| bundle | assignment/native seed | native calls | n_iter | n_binned_pairs | x0 shape | global maxR |
|---|---|---:|---:|---:|---|---:|
| bundle1 | 250101 / 250201 | 1 | 1000 | 610101 | `(2,2645,3)` | 0.8000000000000002 |
| bundle2 | 250102 / 250202 | 1 | 1000 | 609813 | `(2,2645,3)` | 0.8 |
| bundle3 | 250103 / 250203 | 1 | 1000 | 609714 | `(2,2645,3)` | 0.8 |

三份 x0 均 finite、pairwise distinct，经过全 40-track 一次 global center/uniform scale；validation 的 count conservation、full mapping、iter=1000、source/binary/input graph hashes、以及 63 graph-swap checks 均通过。`native_equivariance_asserted=false` 仍保留：通过的是 graph/blob canonical-input invariance，不是 native 引擎等变性证明。

### 4.1 b1 后处理修复留痕

bundle1 曾出现 Python 后处理 bead-order 错误：`normalize_x0` 返回 `(5290,3)` 的 bead-order，但旧代码按 `(copy,locus,xyz)` 索引，产生 `(0,)` 到 `(183,3)` 的 broadcast 问题。修复复用了原 native 输出，没有重新运行 native，也没有修改 native C 引擎。验证记录保留：

- native 调用处的旧 Python driver (`validation.old_native_source_sha256`)：`69b3bc77dc0e397b34a9e7be1fb625ac0cc8fd5cc6b41695822e6c69a74ede90`
- 当前 Python driver/source (`validation.current_source_sha256`)：`cf1c7392933078bd1f0fb9c7346b93e97c19a1f13a7e6b95357a61610437652a`
- bundle1 复用的 native 输出 SHA256：`1039ebfb9c4bad89094ca59bde3b6d339868b6ab864af1fdb17005ee3d881e26`
- `python_postprocess_failure_recovered_without_native_rerun=true`

旧/新 driver hashes 和 correction freeze 都保留在 validation/provenance 中。该修复是 Python bead-order postprocessing 修复，不应写成 native C engine 改变。

## 5. 026 合成校准：评估已完成

结果文件：[`evaluation.json`](../test_res/026-20260913_221709-allele-calibration-prepare/results/evaluation.json)，SHA256 `c1000043074fbd2ac9aa0b5c91fabf9979c471dde0add6d8f7d107002504642e`。该文件 `status=complete`、`candidate_count=20`，且 candidate hash gate 在 truth access 前通过。evaluation source freeze 为 `59044744faa080cd69af069779d6dcddf93f8a0aaa80e79aa79f4eff1c3da3d6`，evaluator source 为 `5d52e40e36731a8b06cf694fd27c6f98374be1d56da574ab580901b76f5a21e4`，fit manifest 为 `95e58cc834562097bf6c6d022a6cedf7b49745de34d526c5351800b1a0ec9007`。

最终调度器为 `bash-101`，exit code `0`，4 个进程各使用 1 个 BLAS/OMP 线程，调度器墙钟时间 `2322.789260 s`；20 个独立 fit 的已用时间总和为 `9250.292618 s`。预算设置为 `maxiter=80/maxfun=270`，实际 20 个 fit 均为 `nit=80` 并因 `maxiter` 停止，终止原因均为 `budget_not_converged`；这不是充分收敛声明。20 个 candidate endpoint 均通过 finite/hash/full-grid evaluator gate；可定义的 primary metrics 均有 valid rows，N 的 named-reference margins/minmargin 按 duplicated common truth 与 geometry ties 规则为 n/a。每个 final 均保留 iter20/40/60/80 checkpoints、17-digit coordinates、components、gradient、RSS、input hashes 和 map diagnostics；终止审计的 final JSON/coordinate/map/checkpoint/finite checks 均为 20/20。evaluation 遵守：同一 full finite mask、每 chr 一次整体 swap、N 使用 common truth matrix、N-only copy difference 不是正向分数、truth-distance rho 不用于 seed/model selection，也不作 biological replicate claim。

四个在 logging 修复前已有 iter20 checkpoint 的 task，其旧 checkpoint 与 retry 的 `theta`、raw-coordinate、physical-coordinate arrays 均逐数组 byte-identical（4/4，maximum absolute difference `0`）。因此 retry 只改变 logging/metadata，不改变 optimizer math 或 shared-start trajectory。`bash-93`（source-boundary，4 fits）、`bash-98`（checkpoint logging collision，8 fits）和 `bash-99`（manifest state-machine，4 fits）均作为 aborted attempts 单独留存；`bash-97` 的 stale-manifest refusal exit `2` 且没有提交 fit。它们不混入通过的 20-fit 结果。

026 独立 implementation gate 的 `status=passed` 只表示 implementation/integrity checks 通过，没有 performance threshold，也没有把 budget stop 标为 convergence。完整中文伴随报告与 compact 6x6/300dpi、2x2 图见 [`026 合成校准报告`](../test_res/026-20260913_221709-allele-calibration-prepare/reports/026_synthetic_allele_calibration_zh.md) 与 [`中文紧凑图`](../test_res/026-20260913_221709-allele-calibration-prepare/reports/026_synthetic_allele_calibration_zh.png)；原 CPU-bar 图不作为最终图。

### 5.1 N 负例测试样例

N1/N2 的真值两份拷贝具有相同内部形状，只做空间分离。因此将共同真值矩阵复制到两份真值拷贝后，`contrast=0` 是构造的代数事实；20/20 条染色体都是 `unresolved_tie`，固定参考 margins/minmargin 为 n/a。它不能证明方法不会制造拆分，必须同时查看 false-split shape error 和 negative-only copy-difference 诊断。下表中的 shape-error 两列都是 20-chromosome arithmetic macro：第一列是每条 chromosome 两份拷贝误差均值再跨 chromosome 平均，第二列是每条 chromosome 两份拷贝中较大误差再跨 chromosome 平均，不是合并所有 chromosome 后取一个全局最大值。

| 测试样例/模型 | matched = cross | contrast | 形状误差均值 / 每 chr 最大值均值（20-chr macro） | N-only copy-difference RMS | 方向 D/S/U |
|---|---:|---:|---:|---:|---:|
| N1/C0 | 0.693342 | 0 | 0.366930 / 0.407235 | 0.563111 | 0 / 0 / 20 |
| N1/C1 | 0.694753 | 0 | 0.366010 / 0.408426 | 0.560885 | 0 / 0 / 20 |
| N1/C2-map | 0.707107 | 0 | 0.353053 / 0.391487 | 0.545198 | 0 / 0 / 20 |
| N1/C2-free | 0.707110 | 0 | 0.353050 / 0.391486 | 0.545193 | 0 / 0 / 20 |
| N1/C3 | 0.703143 | 0 | 0.355871 / 0.398045 | 0.540875 | 0 / 0 / 20 |
| N2/C0 | 0.687059 | 0 | 0.360021 / 0.396812 | 0.538391 | 0 / 0 / 20 |
| N2/C1 | 0.687575 | 0 | 0.357963 / 0.395768 | 0.542121 | 0 / 0 / 20 |
| N2/C2-map | 0.698436 | 0 | 0.355242 / 0.389279 | 0.540658 | 0 / 0 / 20 |
| N2/C2-free | 0.698287 | 0 | 0.355334 / 0.389367 | 0.540787 | 0 / 0 / 20 |
| N2/C3 | 0.669847 | 0 | 0.371514 / 0.411739 | 0.550125 | 0 / 0 / 20 |

N-only copy-difference RMS 是负向诊断；可以比较阴性误差并按“越低越好”解读，但不作为“越大越好”的指标，也不作为选模型或调参依据。N 的零 contrast 仍不能作为正向恢复结果。

### 5.2 P 正例测试样例

P1/P2 的真值是不同的内部形状。下表是每个测试样例独立真值的 20-chromosome 宏平均；`D/S` 是 direct/swapped 染色体计数。shape-error 两列仍都是 20-chromosome arithmetic macro：第一列为每条 chromosome 两份拷贝误差均值的跨 chromosome 平均，第二列为每条 chromosome 两份拷贝较大误差的跨 chromosome 平均，不是全 genome 的跨 chromosome 最大值。

| 测试样例/模型 | matched | cross | contrast | margin ref1 / ref2 | minmargin | 形状误差均值 / 每 chr 最大值均值（20-chr macro） | D/S |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1/C0 | 0.530810 | 0.365906 | 0.164904 | 0.153349 / 0.176459 | 0.101340 | 0.429772 / 0.490488 | 10 / 10 |
| P1/C1 | 0.545271 | 0.373330 | 0.171941 | 0.162195 / 0.181688 | 0.096042 | 0.424688 / 0.478039 | 9 / 11 |
| P1/C2-map | 0.552501 | 0.371809 | 0.180692 | 0.188484 / 0.172900 | 0.115952 | 0.424451 / 0.471453 | 10 / 10 |
| P1/C2-free | 0.552501 | 0.371809 | 0.180692 | 0.188484 / 0.172900 | 0.115952 | 0.424451 / 0.471453 | 10 / 10 |
| P1/C3 | 0.537878 | 0.367197 | 0.170681 | 0.162226 / 0.179136 | 0.104666 | 0.429517 / 0.479942 | 10 / 10 |
| P2/C0 | 0.563361 | 0.396372 | 0.166989 | 0.171893 / 0.162086 | 0.086610 | 0.406466 / 0.450938 | 10 / 10 |
| P2/C1 | 0.559076 | 0.402595 | 0.156482 | 0.160442 / 0.152521 | 0.069923 | 0.407252 / 0.454530 | 11 / 9 |
| P2/C2-map | 0.566076 | 0.405122 | 0.160954 | 0.168256 / 0.153652 | 0.066268 | 0.412381 / 0.451708 | 12 / 8 |
| P2/C2-free | 0.566075 | 0.405122 | 0.160954 | 0.168256 / 0.153652 | 0.066267 | 0.412382 / 0.451708 | 12 / 8 |
| P2/C3 | 0.538905 | 0.383525 | 0.155380 | 0.156103 / 0.154656 | 0.062673 | 0.421179 / 0.458882 | 11 / 9 |

P1 中 C2-map/free 的描述性 contrast 最高（`.180692`），P2 中 C0 最高（`.166989`）；C1、C2-map/free、C3 没有跨 P1/P2 一致的改善。C2-free 与 C2-map 的宏观 contrast 在 P1 精确到 6 位小数相同，P2 只相差约 `7e-7`，因此这些合成数值不支持将 C2-free 的差异单独归因于球边界；C2-free 仍是参数化+物理域的联合敏感性。这里的正 contrast 只描述 best-swap 后观测 matched/cross 的差值；随机 candidate 也可能产生正值，本实验未登记或计算 initial-random 阳性指标作为对照，因此不能单凭正 contrast 声称已读出或拟合出阳性信号，更不等同真实等位基因恢复或 L2。四个测试样例相互独立，N1/N2 或 P1/P2 的差异不能当作纯 exposure 因果比较。

### 5.3 C2 map 活动审计

独立只读 audit 见 [`026_c2_mapping_activity.md`](../test_res/026-20260913_221709-allele-calibration-prepare/reports/026_c2_mapping_activity.md) 和 [`026_c2_mapping_activity.tsv`](../test_res/026-20260913_221709-allele-calibration-prepare/reports/026_c2_mapping_activity.tsv)。`nonidentity/total` 是 objective evaluation 次数；计数包含 line-search probes 以及 accepted/endpoint calls，不是只统计 rejected probes。对于 C2-map，nonidentity 表示至少一个 bead 的 raw radius `>0.9`。

| 测试样例 | C2-map fit-result nonidentity/total | C2-map terminal nonidentity/total |
|---|---:|---:|
| N1 | 23/82 | 24/83 |
| N2 | 37/82 | 38/83 |
| P1 | 0/82 | 0/83 |
| P2 | 22/82 | 23/83 |

C2-free 在四条 calibration 轨迹的全部记录评估中，最大 physical radius 分别为：N1 `.940681908`、N2 `.962391541`、P1 `.859920347`、P2 `.925934542`，均小于 1。因此这些 `cal80` 轨迹没有实际访问球外，不能因为最终输出接近边界就说 map 未活动，也不能把差异归因于访问球外；N1、N2、P2 的 C2-map 明确进入过 `>0.9` nonidentity branch。P1 的 C2-map 始终为 identity，且 C2-free/C2-map final-state SHA256 完全相同：`c57ddcf29469e95d0360c59c4f33546b7c6a1de862c673387fc52f9f3d3ea79a`。

该 audit 只描述这四条已完成 calibration 轨迹的 map 调用和半径范围，没有重新 fit 或重新评价，不改变 C2-free 是 parameterization+physical-domain 联合敏感性的预注册定义，也不外推到 real `480` accepted steps。

## 6. 029 真实联合拟合与已完成的 R2-only 评估

### 6.1 运行、终态与无标签代表选择

029 的真实 run 已完成并封存。终态审计于 `2026-09-13T18:36:52.569145Z` 完成，15 项 fit（5 variants x 3 seeds）的 archive 于 `2026-09-13T18:36:58.145671Z` 创建，R2 receipt 于 `2026-09-13T18:47:17.811368Z` 创建。主凭证是 [`r2_delivery_receipt.json`](../test_res/POST020_REAL_CONTROLLER/r2_delivery_receipt.json)（SHA256 `7a2914e27fcf11059c710c2976c8cb33129c35b0a63906361854f3d4faed68f8`）和 [`post020_real_all_endpoints.receipt.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/archives/post020_real_all_endpoints.receipt.json)；两份均为 `complete`。

15 项 real fit（5 variants x 3 seeds）全部调用并完成固定 `nit=480` budget，状态均为 `budget_not_converged`、停止原因均为 `budget_not_converged_maxiter`；sealed final JSON 的实际优化器 `nfev` 范围为 `482--491`。15 个 worker 的 exit code 全为 0，但 exit 0 不等于 scientific convergence。026 synthetic 的 20 个 calibration fit 也均为固定 `nit=80`、`budget_not_converged`。因此本报告不把任何 endpoint 写成充分收敛或 L2 recovery。

| arm | 终态计数 | 固定预算 | 终态状态 | 实际 nfev |
|---|---:|---:|---|---:|
| 026 synthetic calibration | 20 | `nit=80` | 20/20 `budget_not_converged` | 见 [`evaluation.json`](../test_res/026-20260913_221709-allele-calibration-prepare/results/evaluation.json) |
| 029 真实联合拟合 | 15 | `nit=480` | 15/15 `budget_not_converged` | `482--491` |

真实 endpoint 的无标签选择只在 variant 内按 `count_nll_normalized` 选择代表，不读取 R2 进行选择或调参；五个 variant 都选择 `bundle2`：

| 变体 | 选定 job | count NLL | 选定 3DG | 选定最终 JSON |
|---|---|---:|---|---|
| C0 | C0-bundle2 | 9.653288461 | [`final_coordinates.3dg`](../test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C0-bundle2/attempts/attempt-001/final_coordinates.3dg) | [`final.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C0-bundle2/attempts/attempt-001/final.json) |
| C1 | C1-bundle2 | 9.650544512 | [`final_coordinates.3dg`](../test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C1-bundle2/attempts/attempt-001/final_coordinates.3dg) | [`final.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C1-bundle2/attempts/attempt-001/final.json) |
| C2-map | C2-map-bundle2 | 9.652368628 | [`final_coordinates.3dg`](../test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C2-map-bundle2/attempts/attempt-001/final_coordinates.3dg) | [`final.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C2-map-bundle2/attempts/attempt-001/final.json) |
| C2-free | C2-free-bundle2 | 9.652368628 | [`final_coordinates.3dg`](../test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C2-free-bundle2/attempts/attempt-001/final_coordinates.3dg) | [`final.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C2-free-bundle2/attempts/attempt-001/final.json) |
| C3 | C3-bundle2 | 9.664644128 | [`final_coordinates.3dg`](../test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C3-bundle2/attempts/attempt-001/final_coordinates.3dg) | [`final.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C3-bundle2/attempts/attempt-001/final.json) |

所有 15 个终态的完整归档是 [`post020_real_all_endpoints.tar`](../test_res/029-20260913_161713-post020-allele-ablation-real/archives/post020_real_all_endpoints.tar)，共 60 members（59 business members + `ARCHIVE_MANIFEST.json`）；archive receipt 的 tar readback、manifest member 和 15/15 coordinates、15/15 final JSON、15/15 job coverage 均通过。15 个 job ID 为：`C0-bundle1/2/3`、`C1-bundle1/2/3`、`C2-map-bundle1/2/3`、`C2-free-bundle1/2/3`、`C3-bundle1/2/3`。完整成员 hash 见 [`ARCHIVE_MANIFEST.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/archives/ARCHIVE_MANIFEST.json)。

### 6.2 R2 合同、共同 mask 与数据完整性

R2 评估器只读取已锁定的 candidate/reference 输出，`phase_payload_opened=false`、`real_fit_started_by_evaluator=false`、`selection_recomputed=false`，且只有在所有 candidate 坐标 hash lock 完成后才读取 reference；CLI receipt 为 exit 0。输出范围是 21 个 conditions（15 个真实 variant bundles + 6 个历史 controls）、20 条染色体，共 420 个 chromosome rows；5 个预注册 variant comparisons 的 8 个 metrics 共 40 个 primary cells，并保留 120 个 seed rows。40 个 primary rows 都是 `3/3` valid seeds、20/20 valid chromosomes；120 个 seed rows 各自对应一个指定 seed、20/20 valid chromosomes；这 40 个 primary + 120 个 seed paired rows 无 n/a。420 个 per-chromosome rows 仍按 metric applicability 保留 n/a；例如 consensus 的 contrast/cross/margins 不适用时为 n/a。本次 15 个新真实 candidate 的适用 two-copy metrics 均为 finite。

实际共同 mask 直接从 [`evaluation_manifest.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/evaluation_manifest.json) 的 20 条 `mask_metadata` 汇总，而不是对 420 行重复累计。逐 chromosome 的 `common/total non-diagonal` 为：`chr1 17578/18528`、`chr2 14878/16110`、`chr3 12090/12403`、`chr4 10731/11781`、`chr5 10296/11026`、`chr6 10585/10731`、`chr7 8128/10153`、`chr8 7503/8001`、`chr9 7021/7381`、`chr10 7875/8128`、`chr11 7021/7140`、`chr12 5995/6903`、`chr13 6328/6903`、`chr14 5778/7381`、`chr15 4753/5151`、`chr16 4278/4560`、`chr17 3916/4186`、`chr18 3741/3828`、`chr19 1653/1711`、`chrX 7381/14196`；合计 `157529` common pairs / `176201` total non-diagonal pairs。各 mask 的状态为 `partial_coordinate_coverage`，但 primary rows 没有因 constant、nonfinite、low-pair 或失败而成为 n/a。

固定 `ref1=mat`、`ref2=pat`；每条 chromosome 最多一次 candidate 整体 best swap，只交换 candidate 行，不交换 reference 列，也不做 local copy repair。`contrast=matched-cross`，best-swap 的选择偏差保留在解释中；正 contrast 不能单独作为恢复证据。

### 6.3 选定代表的绝对 R2 宏观摘要

以下数值来自锁定的 [`r2_summary.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_summary.json)，是每条 chromosome 先按固定规则计算后，再对 20 条关联染色体取 arithmetic macro mean；不是 biological replicate summary。

| 代表结果 | similarity (matched) | cross | contrast | minmargin | direct/swapped |
|---|---:|---:|---:|---:|---:|
| C0-bundle2 | 0.477405 | 0.364882 | 0.112523 | 0.030963 | 9 / 11 |
| C1-bundle2 | 0.473841 | 0.379768 | 0.094072 | 0.020905 | 11 / 9 |
| C2-map-bundle2 | 0.473080 | 0.366354 | 0.106726 | 0.032795 | 9 / 11 |
| C2-free-bundle2 | 0.473080 | 0.366354 | 0.106726 | 0.032795 | 9 / 11 |
| C3-bundle2 | 0.493561 | 0.370152 | 0.123409 | 0.040306 | 10 / 10 |

### 6.4 五个预注册比较的主要配对单元

主分析严格使用同一 chromosome 的 paired seed rows：每条 chromosome 先对三个计划 seed 的 delta 取均值，再对 20 条关联染色体作宏观摘要；95% CI 直接使用已保存的 seed=`9301`、`10000 x 20` bootstrap index matrix。它不是 60 个独立重复，也不是 biological CI 或 p-value。下表从预注册的 8 个 metrics 中取 4 项作摘要展示；完整 40-cell/120-seed 表见 [`r2_paired_effects.tsv`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_paired_effects.tsv) 和 [`r2_paired_effects.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_paired_effects.json)。

| 比较（左−右） | 指标 | 平均 delta | 95% CI | 胜 / 平 / 负 |
|---|---|---:|---:|---:|
| C1-C0 | similarity (matched) | +0.000093 | [-0.008406, +0.008901] | 10 / 0 / 10 |
| C1-C0 | cross | +0.013685 | [+0.002895, +0.023979] | 14 / 0 / 6 |
| C1-C0 | contrast | -0.013592 | [-0.029814, +0.001484] | 9 / 0 / 11 |
| C1-C0 | minmargin | -0.007946 | [-0.027245, +0.009563] | 9 / 0 / 11 |
| C2-map-C0 | 相似度（matched） | -0.013198 | [-0.026363, -0.000562] | 7 / 0 / 13 |
| C2-map-C0 | cross | +0.007829 | [-0.001624, +0.017430] | 14 / 0 / 6 |
| C2-map-C0 | contrast | -0.021026 | [-0.041315, -0.000773] | 8 / 0 / 12 |
| C2-map-C0 | minmargin | -0.013286 | [-0.034309, +0.007508] | 9 / 0 / 11 |
| C2-free-C0 | similarity (matched) | -0.013198 | [-0.026363, -0.000562] | 7 / 0 / 13 |
| C2-free-C0 | cross | +0.007829 | [-0.001624, +0.017430] | 14 / 0 / 6 |
| C2-free-C0 | contrast | -0.021026 | [-0.041315, -0.000773] | 8 / 0 / 12 |
| C2-free-C0 | minmargin | -0.013286 | [-0.034309, +0.007508] | 9 / 0 / 11 |
| C2-free-C2-map | 相似度（matched） | 0.000000 | [0.000000, 0.000000] | 0 / 20 / 0 |
| C2-free-C2-map | cross | 0.000000 | [0.000000, 0.000000] | 0 / 20 / 0 |
| C2-free-C2-map | contrast | 0.000000 | [0.000000, 0.000000] | 0 / 20 / 0 |
| C2-free-C2-map | minmargin | 0.000000 | [0.000000, 0.000000] | 0 / 20 / 0 |
| C3-C0 | similarity (matched) | +0.013042 | [+0.003714, +0.022705] | 16 / 0 / 4 |
| C3-C0 | cross | +0.006109 | [-0.003448, +0.015434] | 14 / 0 / 6 |
| C3-C0 | contrast | +0.006934 | [-0.008487, +0.023061] | 13 / 0 / 7 |
| C3-C0 | minmargin | +0.008461 | [-0.011133, +0.027591] | 12 / 0 / 8 |

`C2-map-C0` 与 `C2-free-C0` 的 8 个 primary metric cells 实际相同；`C2-free-C2-map` 的全部 8 个 metrics 都是 20/20 平局。CI 是否跨 0 只表示该单细胞 chromosome-bootstrap 的区间是否覆盖 0，不能把跨 0 称为等价或无效应，也不能把不跨 0 的正向数值直接称为生物学显著。`cross` 较高不是目标改善，因此 C1/C2/C3 的 cross 增加不作正向 recovery 解读。

### 6.5 真实 C2 map 活动补充审计

### 6.5 真实 C2 map 活动补充审计

真实 C2 supplement（[`c2_mapping_activity.md`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/c2_mapping_activity/c2_mapping_activity.md)、[`c2_mapping_activity.tsv`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/c2_mapping_activity/c2_mapping_activity.tsv)、[`validation.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/c2_mapping_activity/validation.json)）是只读审计，没有新 fit/R2，也没有读取 reference/phase。三个 C2-map bundle 的非恒等映射次数/总 objective 次数为 `0/493`、`0/489`、`0/491`，optimizer `nfev` 为 `491/487/489`，每条有 2 个 endpoint extra evaluations；C2-map 和 C2-free 的 raw/physical coordinates、theta、p、NPZ 与 3DG 对每个 bundle 都逐元素/逐字节相同。所有 5290 个 beads 的最大访问 raw/physical radius 都是 `.8<.9`，C2-free 没有访问 unit sphere 外部；但 C2-free 是 `direct_physical_unbounded`，没有 `>0.9` threshold branch，零计数不能解释为“没有越过阈值”。因此本轮 free-minus-map 不能回答球外访问是否有用；C2-map/free 与 C0 的差异仍包含原 sphere-forward 参数化变化，不能说 C2 没有改变任何东西。

## 7. 图表与交付索引

### 7.1 主报告优先使用的展示修订版

使用已锁定的 R2 数值重画了 10 个仅用于展示的图，原始 `evaluation-r2` 文件 inventory 与 delivery receipt locks 未改，`output_validation.json` 报告 10/10、7 pt、300 DPI，且布局检查通过；这些图只修正版式，不改变数据。主报告使用以下修正版：

- 新 variant representative：[`PNG`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/r2_new_representatives.png)、[`PDF`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/r2_new_representatives.pdf)、[`cross PNG`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/r2_new_representatives_cross.png)、[`margin_mat PNG`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/r2_new_representatives_margin_mat.png)、[`margin_pat PNG`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/r2_new_representatives_margin_pat.png)。
- 历史 controls：[`PNG`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/r2_historical_controls.png)、[`PDF`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/r2_historical_controls.pdf)、[`cross PNG`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/r2_historical_controls_cross.png)、[`margin_mat PNG`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/r2_historical_controls_margin_mat.png)、[`margin_pat PNG`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/r2_historical_controls_margin_pat.png)。
- 修正版式 provenance/validation：[`render_provenance.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/render_provenance.json)、[`output_validation.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/output_validation.json)。

### 7.2 四面板均值+95% CI 摘要图

[`r2_primary_mean_ci.png`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_summary_figures/r2_primary_mean_ci.png)（SHA256 `9c5ad2c3415630a120961cc693727f10615504123eeeb7486cbbd439dc69870e`）和 [`r2_primary_mean_ci.pdf`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_summary_figures/r2_primary_mean_ci.pdf)（SHA256 `594d97dfd6f25c49d5be6f27ce56e9860d486fa36f9f6bda1c68295bac305273`）是独立的新展示补充。图为 6 x 6 inch、2 x 2 个 3-inch 基础面板、300 DPI、7 pt；只从已锁定 [`r2_paired_effects.tsv`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_paired_effects.tsv) 的 20 个 headline primary rows 读取既有 `mean_delta/ci95_low/ci95_high`，没有新统计或 R2 重算。渲染脚本和输入记录是 [`render_r2_summary.py`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_summary_figures/render_r2_summary.py)（输入 TSV SHA `8ee23863c8f2be9c4b28f378d4b9fe3aeb6312e64a1429cca68cb537956e1d56`）和 [`input_hashes.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_summary_figures/input_hashes.json)。图的横轴是五个比较的分类标签（按 left variant - right variant 命名），不是数值轴；纵轴是 mean delta，误差线是 TSV 中已有的 95% CI。短标签 `C2m=C2-map`、`C2f=C2-free`；灰色水平线是 0。图内不标“右边更好”，因为 cross 较高不是目标改善。中文解释放在本段和 6.4：三个 seed 先按 chr 平均，再以 20 条关联染色体做 bootstrap，不是 biological replication。

原始 R2 目录中的 26 个图仍完整保留，索引如下；它们不是本次版式修订的替代数据：

- 原始 representatives：[`r2_new_representatives.png`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/r2_new_representatives.png)、[`r2_new_representatives.pdf`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/r2_new_representatives.pdf)、[`r2_new_representatives_cross.png`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/r2_new_representatives_cross.png)、[`r2_new_representatives_margin_mat.png`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/r2_new_representatives_margin_mat.png)、[`r2_new_representatives_margin_pat.png`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/r2_new_representatives_margin_pat.png)。
- 原始 historical controls：[`r2_historical_controls.png`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/r2_historical_controls.png)、[`r2_historical_controls.pdf`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/r2_historical_controls.pdf)、[`r2_historical_controls_cross.png`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/r2_historical_controls_cross.png)、[`r2_historical_controls_margin_mat.png`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/r2_historical_controls_margin_mat.png)、[`r2_historical_controls_margin_pat.png`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/r2_historical_controls_margin_pat.png)。
- 原始 paired effects：[`similarity`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/paired_effects/r2_paired_effects_similarity.png)、[`cross`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/paired_effects/r2_paired_effects_cross.png)、[`contrast`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/paired_effects/r2_paired_effects_contrast.png)、[`matched_ref1`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/paired_effects/r2_paired_effects_matched_ref1.png)、[`matched_ref2`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/paired_effects/r2_paired_effects_matched_ref2.png)、[`margin_mat`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/paired_effects/r2_paired_effects_margin_mat.png)、[`margin_pat`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/paired_effects/r2_paired_effects_margin_pat.png)、[`minmargin`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/paired_effects/r2_paired_effects_minmargin.png)。
- 原始 seed stability：[`similarity`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/seed_stability/r2_seed_stability_similarity.png)、[`cross`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/seed_stability/r2_seed_stability_cross.png)、[`contrast`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/seed_stability/r2_seed_stability_contrast.png)、[`matched_ref1`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/seed_stability/r2_seed_stability_matched_ref1.png)、[`matched_ref2`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/seed_stability/r2_seed_stability_matched_ref2.png)、[`margin_mat`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/seed_stability/r2_seed_stability_margin_mat.png)、[`margin_pat`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/seed_stability/r2_seed_stability_margin_pat.png)、[`minmargin`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/plots/seed_stability/r2_seed_stability_minmargin.png)。

原始 paired effects 图的箱线和须线表示 chromosome-level delta 分布，不是 bootstrap CI；原始 seed stability 图中的 seed 点/线也不是 CI。只有 7.2 的四面板图直接绘制 primary `mean_delta` 和已有的 95% bootstrap bounds。

## 8. 结论边界与实质性限制

### 8.1 本轮主要结论

1. 本轮没有证明等位基因特异差异恢复得到改善。主要依据是 5 个预注册 comparisons 的 40 个 primary cells，而不是单个最优 endpoint。
2. C1 相对 C0 的 matched similarity 几乎不变（`+0.000093`，CI 跨 0），contrast 为负但 CI 仍跨 0；不能据此说去掉 bend 改善等位基因信号。C1 的 cross 上升不是目标改善。
3. C2-map 和 C2-free 相对 C0 的 similarity 与 contrast 都下降；两者在 3 个 seeds 上逐元素相同，free-minus-map 的 8 个 primary metrics 全为 20/20 平局。C2 audit 只表明本次 real480 轨迹没有进入 `>0.9` map branch，也没有访问 unit sphere 外部，不能回答球外访问是否有用。
4. C3 相对 C0 提高 overall matched similarity（`+0.013042`，CI `[+0.003714,+0.022705]`），但 contrast 的 CI 为 `[-0.008487,+0.023061]`、minmargin 的 CI 为 `[-0.011133,+0.027591]`；因此只能写整体相似度提高，不能写等位基因特异信号明确改善。
5. 这些结果不建立 L2 整条染色体拷贝身份，也不证明真实等位基因恢复；L1 的历史 chr1 predictive signal、L2 尚未建立、L3 对整个 nonlinear solver class 的撤回必须分开。

### 8.2 预注册与解释限制

- 新 C0 endpoint 包含 025 native initialization 和 480-step joint optimization 的共同贡献；21 条预注册 R2 conditions 不包含 025 三个 native x0 的独立 R2，所以不能分离 initialization effect，也不能把 C0 的全部结构表现单独归因于 480-step optimization。
- 020/022 多分辨率与 continuation 是历史描述，不是与新 C0 等预算、等初值的因果对照；022 的旧 continuation 只说明有限延长没有改善既有 R2。
- 029 选择在 variant 内只按无标签 `count_nll_normalized` 进行；R2 没有返选 endpoint、改参数或新增候选。best overall swap 的正向选择偏差仍存在；positive contrast、较低 J 或较高 matched 都不等同 L2。
- 20 条染色体来自同一个 P9016 cell，是关联测量；三个 bundle 是 optimization starts，不是 biological replicates。bootstrap 只表示该单细胞内的 structural/technical variation。
- reference `data/P9016.1m.3dg.gz` 是仅用于评估的重构几何；R2 可以比较 candidate 与其几何轨迹的相似度，但不能验证染色体间放置的真实生物学正确性。
- 15 个 real endpoints 都在固定 budget 停止，`budget_not_converged`；exit 0、低 objective、正 contrast 或跨 0 的区间都不能被包装成充分收敛、等价性或生物学显著。
- R2 使用固定 `ref1=mat/ref2=pat`、unordered offdiag finite common mask、一次整体 swap 和失败/invalid/n/a 保留规则；本轮没有新增 R1/R3，也没有 025 x0 独立 R2。

## 9. 来源

- 冻结协议：[`POST020_ALLELE_ABLATION_PROTOCOL.json`](POST020_ALLELE_ABLATION_PROTOCOL.json)、[`POST020_ALLELE_ABLATION_PROTOCOL.md`](POST020_ALLELE_ABLATION_PROTOCOL.md)。
- prefit 计划与留档：[`PLAN-post020-allele-signal.md`](PLAN-post020-allele-signal.md)、[`prefit snapshot`](archive/PLAN-post020-allele-signal.prefit-3d85720a.md)、[`snapshot provenance`](archive/PLAN-post020-allele-signal.prefit-3d85720a.provenance.json)。prefit snapshot SHA256 `3d85720af477b4a87cffdd4e19780a2e9b3e0937f04579bfca2bea2516cb2215`；现行 PLAN 的 Revision 2 状态与结果导航在该文件开头及阶段 E/F。
- 023 诊断：[`README.md`](../test_res/023-20260913_212541-post020-allele-signal-diagnostics/README.md)、[`data_budget.json`](../test_res/023-20260913_212541-post020-allele-signal-diagnostics/results/data_budget.json)、[`recommendations.json`](../test_res/023-20260913_212541-post020-allele-signal-diagnostics/results/recommendations.json)。
- 024 更正后的历史 R2：[`README.md`](../test_res/024-20260913_133109-r2-allele-signal-derived/README.md)、[`allele_signal_results.json`](../test_res/024-20260913_133109-r2-allele-signal-derived/results/allele_signal_results.json)。
- 025 原生验证：[`POST020_ALLELE_ABLATION_NATIVE_VALIDATION.json`](../test_res/025-20260913_135100-random-native-fullgrid-preflight/validation/POST020_ALLELE_ABLATION_NATIVE_VALIDATION.json)、[`manifest.json`](../test_res/025-20260913_135100-random-native-fullgrid-preflight/manifest.json)、[`config.json`](../test_res/025-20260913_135100-random-native-fullgrid-preflight/config.json)。
- 026 合成校准：[`evaluation.json`](../test_res/026-20260913_221709-allele-calibration-prepare/results/evaluation.json)、[`中文 companion report`](../test_res/026-20260913_221709-allele-calibration-prepare/reports/026_synthetic_allele_calibration_zh.md)、[`C2 map audit`](../test_res/026-20260913_221709-allele-calibration-prepare/reports/026_c2_mapping_activity.md)。
- 029 endpoint/release 凭证：[`r2_delivery_receipt.json`](../test_res/POST020_REAL_CONTROLLER/r2_delivery_receipt.json)、[`release_manifest_r2.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/release_manifest_r2.json)、[`final_release_receipt.json`](../test_res/POST020_REAL_CONTROLLER/final_release_receipt.json)、[`termination_audit.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/termination_audit.json)、[`worker_exit_audit.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/provenance/worker_exit_audit.json)、[`post020_real_all_endpoints.receipt.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/archives/post020_real_all_endpoints.receipt.json)、[`ARCHIVE_MANIFEST.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/archives/ARCHIVE_MANIFEST.json)、[`post020_real_all_endpoints.tar`](../test_res/029-20260913_161713-post020-allele-ablation-real/archives/post020_real_all_endpoints.tar)。
- 029 R2 输出：[`README.md`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/README.md)、[`evaluation_manifest.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/evaluation_manifest.json)、[`r2_summary.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_summary.json)、[`r2_per_chromosome.tsv`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_per_chromosome.tsv)、[`r2_paired_effects.tsv`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_paired_effects.tsv)、[`r2_results.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_results.json)。
- 029 C2 真实补充：[`c2_mapping_activity.md`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/c2_mapping_activity/c2_mapping_activity.md)、[`c2_mapping_activity.tsv`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/c2_mapping_activity/c2_mapping_activity.tsv)、[`validation.json`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/c2_mapping_activity/validation.json)。
- 029 展示修订版：[`r2_display_revision`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_display_revision/output_validation.json)；新的 CI 汇总：[`r2_summary_figures`](../test_res/029-20260913_161713-post020-allele-ablation-real/supplements/r2_summary_figures/input_hashes.json)。

### 9.1 封存哈希索引

| 产物 | SHA256 |
|---|---|
| R2 delivery receipt | `7a2914e27fcf11059c710c2976c8cb33129c35b0a63906361854f3d4faed68f8` |
| 真实 archive receipt | `844e911da6e865de49fac8c5be8c563ff7fc580e9fec30114b8d07c7e94de40b` |
| release manifest R2 | `aa3b729e81f745b23b3267f2011fe556ed61c235ff722e78d28a6fc25f2da16b` |
| 最终 release receipt | `15e1348dec0044677ac3a1751359d2e9869789209ad2429a8721b984c3498d54` |
| 终止审计 / worker 退出审计 | `6d641290baf676627fda6c4b709be47bd6bf4d9819015560089dab25d130745b` / `f842ec6c1ca223fd8f703716d837bbc4cfa56a38dd4f66ef8de14d1e2b4f0d52` |
| archive manifest / 15-arm tar | `b962897da7630739dfab88057166c67e13bbb0133a5cccf7d28fb138fddcbb1d` / `9cf5ec0f0e0be82caf7fff813ed5004edfe874079a5eb865d42e7d0d31fb9550` |
| evaluator source `pr/allele_r2.py` | `56dacfe9a3edc13db4804fa7d5ce7c40b9401a1d1dd62ddbaa66d0d6ad5b1672` |
| evaluation manifest / `r2_results.json` | `fa1b26c834c173604c21f954d494cece8e053dd970e7cb6111f561b349574acb` / `647821a0760be20f5e3656136a421e6124ada3c4f8d99e7a9a590c74e0720d55` |
| `r2_summary.json` | `877cbee683c3a31b007eefd8a363b8630e817ff6e9faafc7b28e7c30c46f4896` |
| `r2_paired_effects.tsv` / `.json` | `8ee23863c8f2be9c4b28f378d4b9fe3aeb6312e64a1429cca68cb537956e1d56` / `37b1c4ca8883de15b5d413578fc6be79c9474d5df8cfd05d70a4ab42896aa2c9` |
| `r2_per_chromosome.tsv` / `.json` / `.csv` | `5e578b5ef8876f57866f9dfa325a8738adab70ebc4913498c570cc2b07883ed1` / `ae29cf14b213c5652d10d90cf57fc5d3c8865902c7b78147fe7bfae8adaa4c16` / `425c023e37066b6a4c42c135efe250619d2ffb512b0bf579159596c1499c0c6a` |
| C2 validation | `cf95685d0b81bc4f6b67e1b02ac361e1d83ae1e5066f843b9202dd24438a79a1` |
| CI 渲染器 / 输入记录 | `fca52a888d33928471796e4e0edefb2dfb96cc0c5dfcbb5b2a69eb8a055fdf95` / `71283960abbbb8394a6ec3ae28ea315bece6e06e6df09a7b2ee8caa78240f9a1` |
| CI PNG / PDF | `9c5ad2c3415630a120961cc693727f10615504123eeeb7486cbbd439dc69870e` / `594d97dfd6f25c49d5be6f27ce56e9860d486fa36f9f6bda1c68295bac305273` |
| 展示修订验证 / provenance | `4f1e2dc2223a838a60f9b5ebbac0bf9fc41ff3bd66fe56bc3268b5292179ea89` / `e101a542dceede9990e049ad02c5990a7d508182f56ad553184643ca787e2d10` |
