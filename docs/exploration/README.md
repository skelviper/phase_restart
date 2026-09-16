# 探索脚本状态

## `pr/` 包和 `run.py` 已成为项目主体，不再属于探索代码

Stage 1 已移出 `docs/exploration/`，现在由已纳入项目的驱动器实现：

| 路径 | 作用 |
| --- | --- |
| `run.py prepare` | 将 `phase0`/`phase1` 去掉，写出 `inputs/P9016.snpfree.pairs.gz`（7 列，sha256 在 `inputs/manifest.json`） |
| `run.py stage1` | 片段交换实验；规范运行是 `test_res/007-…`（chr1）、`008-…`（chrX）、`009-…`（chr1 使用 chrX 深度）；001/002 和 004/005/006 已被取代 |
| `pr/pairs7.py` | SNP-free 加载器；若文件的 `#columns` 提到任何 phase 字段则**硬拒绝** |
| `pr/labels.py`、`pr/ref3dg.py` | 唯一读取 phase 列/参考结构的模块；只有在写出并哈希坐标的阶段通过 `pr.gate.EvalGate` armed 后才允许读取 |
| `pr/folds.py` | fold 哈希；已核验与 `benchmark_v1.py` 的标量哈希完全一致，也已核验它不是基因组距离的函数 |
| `pr/fdg.py`、`pr/splits.py`、`pr/score.py`、`pr/splice.py`、`pr/report.py`、`pr/figs.py` | 引擎封装、候选拆分、留出评分、无 gauge chimera、README 构建器和图形 |

设计冻结在 `../STAGE1_PREREGISTRATION.md`。数字见 `../MEASURED_FACTS.md` 的 F13–F20。

`MEASURED_FACTS.md` 历史上记录过两个引擎事实：在测试 seed 下观察到 bit-identical 的 `.3dg` 文件（F14），以及双拷贝拟合在 chr1 上存在 0.0212 的标签规范不对称 held-out rho。S2 审查发现旧 Stage 1 封装把 `-s` 放在 `-b` 后面，因此 F14 不能证明正确排序的 native seed 无效。正确 seed 仍然只是优化初始化重复；独立对照仍必须改变数据。所有双拷贝比较都必须保持 gauge-invariant。

**使用任何数字前，先阅读 `../MEASURED_FACTS.md` 的“审查更正”部分。** 外部审查发现探索代码有四个缺陷，并已用数值核验：

| 缺陷 | 影响 |
| --- | --- |
| 坐标按字符串比较（`c[2] > c[4]`） | 丢掉 **147,005 / 1,135,454** 条染色体内接触（12.95%）；chr1 为 14.10% |
| 拆分哈希 `(i*7919 + j*104729) % 2` | 等于 `(i+j) % 2`，即基因组距离的奇偶性；所有同 bin 对都落入 test fold |
| 在 test fold 上选择距离指数 | 在评估集上选择参数 |
| 交换准确率取 `1 - acc` | 把每个平局都计入；主要平局来源是同 bin 零距离接触 |

字符串比较已在原处机械修复。其余三个问题没有回填修复；受影响脚本已被取代。

## 有效脚本

| 脚本 | 能确立什么 |
| --- | --- |
| `benchmark_v1.py` | **更正后的基线。** 数值比较、mixed-hash 60/20/20 拆分、在 validation 上选择指数、排除 same-bin、对 test bin pair 计算 bootstrap CI。取代 `id_test_all.py`。其 fold 哈希由 `pr/folds.py` 逐字节复现。 |
| `explore_estep_fixed.py` | 更正后的参考结构准确率：去掉 same-bin 接触后为 **76.2–81.2%**（chr1 为 79.03%）。取代 `explore_estep.py`。 |
| `verify_review.py` | 四个缺陷的数值核验。 |
| `basin_radius3.py`、`basin_radius4.py` | 第 2.1 步 basin sweep。作为**诊断**有效，但见下方限制。 |
| `sim_identifiability.py` | 参考结构上的模拟对照。 |

## 已取代或带限制

| 脚本 | 问题 |
| --- | --- |
| `id_test_all.py`、`id_test.py` | 字符串比较、奇偶哈希、在 test 上选择指数、包含 same-bin。已由 `benchmark_v1.py` 取代。 |
| `explore_estep.py` | `1 - acc` 的平局处理和 same-bin 膨胀。已由 `explore_estep_fixed.py` 取代。 |
| `explore_control.py`、`explore_haps.py`、`explore_selfconsistency.py`、`explore_feasibility.py`、`explore_bins.py` | 接触过滤使用字符串比较；只能作定性结论。 |
| `spectral_*.py`、`signmap_check*.py`、`construct_check.py` | 使用字符串比较。它们的发现——FDG 共识不是中点、其几何不携带符号信息——是 FDG 拟合的性质，不是过滤器的性质；已在 `signmap_check2.py` 中用数值过滤重新推导。 |
| `pilot_*.py`、`basin_radius*.py` | 接触过滤使用字符串比较。 |

## basin 实验的限制

- 该实验使用的“真值”是**在训练 fold 上重新拟合的 native oracle**，不是 `data/P9016.1m.3dg.gz`。
- 其 `contrast` 按每条接触记录计算（重复的 bin pair 会重复计数），且不限于 test fold。
- 因此，“达到真值的 65%”应理解为达到该**诊断量**的 65%，不是恢复真实染色体结构的 65%。
