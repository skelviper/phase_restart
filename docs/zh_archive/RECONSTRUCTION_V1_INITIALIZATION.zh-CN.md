# Reconstruction V1 初始化

## 状态与边界

本文件只冻结第一版连续 Reconstruction V1 原型的**初始化变换**。它不授权运行 P9016 拟合、参数搜索、模型选择、分数计算或参考侧评估。输出是优化起始状态，不是独立拟合、生物学重复或 L2 恢复结论。

实现位置：`pr/reconstruction_init.py`。

该模块绝不打开带 phase 的配对文件或 `data/P9016.1m.3dg.gz`。其数组级 API 不进行文件 I/O。真实数据加载器只允许使用下面两个通过哈希核验的盲来源 014 坐标。

## 冻结的盲来源

| 候选 | 来源路径 | gate 要求 | SHA256 | 基础 seed |
| --- | --- | --- | --- | ---: |
| `consensus` | `test_res/014-20260912_153000-s0-genome-wide-fixed/coords/consensus.3dg` | `stage=blind`, `tag=consensus` | `e76655732deb6b8386b1b77bc76ff45d7dba1384f6931337fee80d8f4aaa8e02` | 1103 |
| `random` | `test_res/014-20260912_153000-s0-genome-wide-fixed/coords/random.3dg` | `stage=blind`, `tag=random` | `9a48d73e1401e18349d11758e679da4c76da0904dbc467979079cb54bcd567d7` | 2207 |

`initialize_approved_candidate()` 在解析坐标行之前，会核验以下全部条件：

1. 候选必须恰好是 `consensus` 或 `random`；`oracle`、任意路径、带 phase 的文件和参考文件都不是可接受的 API 来源。
2. 实际坐标 SHA256 必须等于冻结的候选摘要。
3. `test_res/014-20260912_153000-s0-genome-wide-fixed/gate.json` 必须包含一条与冻结摘要匹配的记录，且具有已解析的坐标路径、`stage=blind` 和候选 tag。
4. 来源轨迹清单以及每个 native `#chromosome` 表头长度，必须与所提供 SNP-free 表头的顺序和长度一致。

加载器不使用 014 oracle 坐标，也不打开 phase 或参考数据。

## 坐标契约

所有公开的初始化都使用 copy-first 的稠密数组：

```text
coords.shape == (2, N_loci, 3)
axis 0：0 = 任意拷贝 a 的规范，1 = 任意拷贝 b 的规范
axis 1：按所提供表头顺序串接染色体位点
axis 2: x, y, z
```

对应的 `positions.shape == (N_loci,)` 和 `chromosome_index.shape == (N_loci,)` 与位点轴对齐。`positions` 是基因组 bin 起点，不是旧版评估偏移。`locus_offsets` 给出按表头顺序的边界；元数据 `metadata.track_mapping` 显式把每个 `c01a`、`c01b`、...、`c20a`、`c20b` 轨迹映射到 `(chromosome_index, chromosome_name, copy_index)`。

在 bin 大小 `B` 下，表头长度为 `L` 的染色体恰好有 `ceil(L / B)` 个位点，位置为 `0, B, ..., (ceil(L/B)-1)*B`。末端不完整 bin 保留。对 P9016 的 1 Mb 网格，每份拷贝有 2,645 个位点，共 5,290 个物理珠子。这与旧版 3 Mb 偏移评估网格（每份拷贝 2,585 个位点）分开。

每个结果都断言：坐标为有限值；每条轨迹内的位置按数值排序；包含完整网格；20-header 基因组有 40 条轨迹；每个坐标的 `norm < 1`。

## 已批准的来源扩展

对每条染色体和每条来源轨迹，先按基因组位置数值排序 native 坐标行。输出完整网格通过括住目标位置的两行进行线性插值得到。低于第一行 native 坐标或高于最后一行 native 坐标的 bin，使用最近端点填充。来源不能丢弃任何完整网格位点，包括 3 Mb 之前的 bin 和末端不完整 bin。

对于每条目标轨迹，元数据记录：

- `exact`、`interpolated` 和 `endpoint_filled` 目标位点数量；
- 来源行数、唯一行数和重复行数；
- 对应重复来源位置的目标位置；
- 来源和目标轨迹名称；
- 是否将 consensus 来源轨迹复用于两份拷贝；
- 接收规定小幅扰动的坐标数量。

对于 consensus 来源，每条 native `cXXa` 轨迹为两份拷贝提供共同的初始 `Z`。对于 random，native `cXXa` 和 `cXXb` 分别提供匹配的拷贝轴。不作母本/父本解释。`swap_copy_first(coords)` 是显式的整细胞 A/B 规范补集。

## Consensus 和 random 起点

来源扩展后，在**全部**物理珠子上做一次中心化，再统一缩放到最大半径 0.8。绝不进行按染色体或按拷贝的中心化、旋转、对齐或缩放；这样可以保留盲来源携带的染色体间相对几何关系。

### Consensus

`consensus` 把 `Z` 扩展成两份带非零平滑半差的拷贝：

```text
X = Z + u
Y = Z - u
```

对每条染色体，`u` 是带随机种子 `seed` 的随机 3D 控制点场。控制点每 20 Mb 间隔放置，并包含位置零点以及最后一个完整网格位点。控制点场线性插值到各位点，并归一化为染色体 RMS 大小 `0.06 * R`，其中冻结的核球半径为 `R = 1`。来源归一化目标 0.8 是另一个最终全细胞显示/起始尺度常数。冻结 seed 为 1103。元数据记录所有控制点数量、最终前 RMS 值和 `post_final_field_rms`。加入 `u` 和扰动后，再做一次共同的最终中心化/统一重缩放，将最大半径恢复为 0.8；因此不会暗示重缩放前的 RMS 保持不变。

### Random

`random` 保留两条 native 盲轨迹。冻结 seed 为 2207。不使用 phase 标签、oracle 坐标或参考几何来选择任一拷贝轴。

对于两类候选，新插值/端点填充的位点和重复来源坐标都会接收确定性的高斯扰动。Consensus 来源坐标会复制到两个候选拷贝，因此两条拷贝轴都接收该小扰动。最终归一化之前的尺度为

```text
l0 = (2 * N_loci)^(-1/3)
perturbation_scale = 0.025 * l0
```

最终操作仍然只是一次全细胞中心化/统一缩放到半径 0.8。元数据会说明这些 seed 是优化初始化选择，不是数据重复或生物学重复。

## 多分辨率热启动

`warm_start_from_layer(coords, positions, chromosome_index, names,
header_lengths, bin_size, candidate_base_seed)` 直接接收上一层的 copy-first 40 条轨迹。`positions` 和 `chromosome_index` 可以是共享的 `(N,)` 数组，也可以是按拷贝的 `(2, N)` 数组。每条轨迹都使用与已批准来源变换相同的 exact/interpolated/endpoint-fill 审计，数值插值到更细的完整网格。

热启动有意不重新中心化或重缩放单条染色体、单份拷贝或整个细胞。它保留输入的全细胞坐标框架和尺度。只有新引入/重复的坐标接收规定扰动，使用：

```text
seed = 3301 + bin_size // 1_000_000 + candidate_base_seed
perturbation_scale = 0.025 * (2 * N_loci)^(-1/3)
```

任何位于或超出单位球的坐标都会径向裁剪到 `1 - 1e-6`。元数据报告输入最大半径、裁剪上限以及被裁剪的物理坐标数量。因此输出是有限值、完整网格，并严格位于球内，同时不会独立修改染色体坐标框架。

## API 摘要

```python
from pr import reconstruction_init as init

layout = init.full_grid_layout(names, header_lengths, bin_size)
result = init.initialize_approved_candidate("consensus", names, header_lengths, bin_size)
# result["coords"]: (2, N_loci, 3)
# result["positions"], result["chromosome_index"], result["metadata"]

warm = init.warm_start_from_layer(
    result["coords"], result["positions"], result["chromosome_index"],
    names, header_lengths, finer_bin_size, candidate_base_seed=1103,
)
```

`expand_tracks_to_full_grid()` 和 `initialize_from_tracks()` 接收显式的内存内轨迹，可用于合成测试样例和未来适配器。它们不读取文件。真实 P9016 使用必须保留在 `initialize_approved_candidate()` 上，这样不能绕过盲来源白名单和 gate/hash 核验。

## 必需审计字段

每个初始化结果都提供包含以下内容的 `metadata`：候选/模式、seed、完整网格标志、`l0`、扰动规则和数量、所有归一化或热启动裁剪细节、每条轨迹的插值审计、精确坐标形状、轨迹映射、位点偏移、有限值状态、最大半径、耗时，以及“结果是优化初始化而不是生物学重复”的声明。已批准来源的结果还额外包含来源路径、来源 SHA256、gate 路径、已解析的 gate 条目、gate stage 和 gate tag。
