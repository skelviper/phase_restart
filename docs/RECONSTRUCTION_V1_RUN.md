# Reconstruction V1 正式运行

## 状态与授权

本文件冻结第一轮连续、无 phase Reconstruction V1 训练运行的生产运行器配置。它只是一份实现和 provenance 规范。它不授权真实 P9016 拟合、参考侧评估、读取 phase、参数搜索或自动重试。在执行下面的命令前，必须先获得上游阶段发布许可。

运行器为 `pr/reconstruct.py`；CLI 为：

```bash
python run.py reconstruct --out NEW_FORMAL_OUTPUT --workers 2 --threads 1
```

输出路径必须不存在。命令只创建一次，且绝不复用或覆盖正式运行目录。候选工作进程数可设置为 `1` 或 `2`；同样的 `1` 或 `2` 上限也会应用于 `OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS` 和 `NUMEXPR_NUM_THREADS`。如果这些变量尚未达到上限，CLI 会重新执行自身一次，以便在该限制下初始化 NumPy/BLAS。多工作进程运行使用显式的 Python `spawn` 多进程上下文；工作进程只有在继承线程上限后才导入训练路径。

## 冻结的训练总体

| 项目 | 冻结值 |
| --- | --- |
| 样本 | P9016；一个生物样本/细胞 |
| 输入 | `inputs/P9016.snpfree.pairs.gz` |
| SNP-free SHA256 | `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa` |
| 原始来源 SHA256 | `071a6cc76bfad543ea1ace6ee1ce3022b30ac1b1e50a9f0c3a3a1967b9649505` |
| 接触记录 | 1,703,888 = 1,135,454 intra + 568,434 inter |
| 基因组范围 | 原始的 20 个 SNP-free 染色体表头，每条有两份无标签拷贝 |
| 最终训练网格 | 1 Mb，起点 0，每份拷贝 2,645 个位点，5,290 个物理珠子，40 条轨迹 |
| 坐标 | 无量纲原始 `R=1` 单位球坐标 |

每次加载无 phase 输入都会核验冻结的 SNP-free 摘要和七列格式。每个分辨率都保留完整输入记录预算、完整的全网格可用配对集合 `E`、对角干扰项层、cis 和 inter 计数、零计数可用配对以及端点预算。以 0 为起点的完整训练网格不同于从 3 Mb 开始的旧指标网格；训练期间不运行指标。

## 预注册候选与起点

候选按以下确切顺序列出并选择：

1. `consensus_joint`：已批准的 014 盲 `consensus` 来源，基础 seed `1103`。
2. `random_joint`：已批准的 014 盲 `random` 来源，基础 seed `2207`。

唯一允许的启动工件是通过哈希核验的 014 盲来源文件及其匹配的 `gate.json` 条目。运行器记录来源路径、来源哈希、source-gate 哈希和基线元数据。它可以记录 oracle source-gate 元数据和摘要，但训练期间不会打开、哈希、解析或以其他方式加载 oracle 坐标载荷。

在 5 Mb 层，每个候选调用已批准的来源初始化器。在 2 Mb 和 1 Mb 层，调用前一候选层冻结的完整网格热启动。热启动保留输入的共享坐标框架和尺度，只扰动新引入/重复的位点，并使用规定的原始 `q` 传递规则。候选起点是优化初始化，不是数据重复或生物学重复。

## 冻结的优化器与模型设置

分辨率顺序和数值预算固定如下：

| 层 | bin 大小 | `maxiter` | `maxfun` | `maxls` |
| --- | ---: | ---: | ---: | ---: |
| `5m` | 5,000,000 bp | 300 | 930 | 20 |
| `2m` | 2,000,000 bp | 200 | 630 | 20 |
| `1m` | 1,000,000 bp | 240 | 750 | 20 |

所有层使用 `ftol=1e-10` 和 `gtol=1e-6` 的 L-BFGS-B。第一层使用 `p_init=0.75`。细层直接传递上一层的原始有界 logistic `q` 值，而不是对四舍五入后的 `p` 重新求逆。

目标仍然是冻结的 V1 全网格条件计数模型，加上固定的对角干扰项层和固定先验：

- 有限有界核，`epsilon=1e-6` 且 `r0=2*l0`；
- 观测端点暴露量 `sqrt(endpoint_count + 10) / full_grid_mean`；
- `count=1`、`p_prior=1`、`bond=1`、`repulsion=1`、`bend=0.01`；
- 固定的有界 cis 混合参数化和固定的 `p` 先验；
- 全拷贝单位球变换、轨迹内 bond/bend 以及所有物理珠子的排斥。

P9016 运行期间不选择模型系列、核、暴露量、先验、正则化权重、来源系列或停止预算。

## 不可变 provenance 与可变状态

在任何优化器调用前，正式输出会写入：

- 不可变的 `config.json`，包含候选/尝试预注册、顺序、优化器设置、冻结模型契约、完整网格约定和输入身份；
- `track_map.json`，严格按原始表头顺序；
- `input_version.json`，记录未修改 SNP-free 输入的字节哈希身份；
- `provenance/training-code/` 下的来源快照，以及 schema 为 `reconstruction-training-code-snapshot-v1` 的 `provenance/training-code-manifest.json`；
- `provenance/protocol/` 下的协议快照和 `provenance/protocol-manifest.json`。

`training-code` 清单只包含参与无 phase 训练路径的代码，不包含评估渲染器资源。每个成员列出原始 `source_path`、相对运行目录的 `snapshot_path` 以及快照字节 SHA256。`selection guard` 会对清单哈希，并核验每个快照成员。协议清单也按成员检查，包括冻结协议文档 SHA256。来源路径只用于历史 provenance，核验时绝不从后来编辑过的工作区重新读取。

`selection.json` 中的 `frozen_provenance` 恰好有 `protocol`、`config`、`code` 和 `data` 四个条目。protocol/config/data 使用 `hash_source=file_bytes`；code 使用 `hash_source=snapshot_manifest`。可变进度绝不出现在该冻结对象中。运行时状态单独存放在 `run_status/run_status.json`、每个候选的状态文件和 `run_status/attempts.jsonl` 中；日志按预注册的候选/阶段顺序，对每次尝试恰好包含一行最终记录。

## 每层工件与失败行为

对于每个候选/层，运行器保存：

- 完整网格初始坐标和 SHA256，并标记为不受参考侧 gate 约束；
- 完整网格最终坐标和 SHA256，写入新路径；
- 原始/聚合/端点预算审计、完整 `E` 大小、零配对数量和对角层审计；
- 初始和最终总目标、完整的最终组成诊断、`p`、输入/输出原始 `q`、L-BFGS 的 status/message/nit/nfev/njev/elapsed time，以及适用时明确的 `budget_exhausted`/`not_converged` 状态；
- `logs/` 中的已接受迭代进度，以及每 10 个已接受迭代保存一次的检查点；每个检查点都保存坐标、`theta`、`y`、网格索引和截至该检查点的完整历史。

已完成的 5 Mb 或 2 Mb 尝试在候选层级仍不是终态。每个候选恰好记录一个终态尝试：已完成的 1 Mb 层，或第一次致命失败。失败候选仍保留在日志和 selection 中，使用 `coordinates=null`、`count_nll_per_record=null` 以及失败原因。失败候选绝不能被静默删除或选中。如果两个候选都失败，运行器写出终止训练失败摘要，并且不创建 `training_complete` selection。

在单个层内，最终总目标不得超过自身初始总目标；由于维度不同，绝不跨分辨率比较目标值。

## 无标签选择与完成保护

两个候选都达到训练终态后，所有未失败的 1 Mb 候选都会从新写入的坐标文件重新读入。运行器使用该候选拟合得到的原始 `q`（固定不变）重新计算同一个全数据目标。它恰好记录以下 `count_model` 字段：

```text
count_nll_normalized
conditional_nll_raw
diag_profiled_nll_raw
count_nll_raw
p
bond
repulsion
bend
p_prior
total
```

只有 `count_nll_normalized` 可以用于选择。先验和 `total` 不是选择值。每条记录的 count NLL 最小者获胜；每条记录差值在 `1e-9` 内时，使用冻结的预注册候选顺序。选中的坐标逐字节复制为 `selected.3dg`，并记录其来源候选和两份哈希。

最终 `selection.json` 使用 schema `reconstruction-selection-v1`，状态 `training_complete`，以及冻结的目标/模型契约。它标识所有候选尝试，并把 014 consensus/random/oracle 基线资源保留在候选清单之外。运行器只调用 `reconstruction_report.verify_training_complete()`，作为纯 provenance/网格检查。它绝不调用评估加载器、`guarded_evaluation_inputs`、phase readers、reference readers、oracle coordinate loaders、report metrics 或 renderers。评估必须在该输出完成并通过哈希核验后，作为后续隔离阶段进行。
