# 051 — oracle / baseline / reference 的 chr1 距离矩阵（内部记录）

本目录只做**只读制图**：不拟合、不重跑评价、不改任何冻结产物。输入是两个已冻结的 3DG 与一份已冻结 mask。

## 命令

复用既有双 3DG CLI（`scripts/plot_3dg_comparison.py`，第二位固定按 reference 命名 `chrN(mat)/chrN(pat)` 读）：

```bash
python scripts/plot_3dg_comparison.py \
  test_res/014-20260912_153000-s0-genome-wide-fixed/coords/oracle.3dg data/P9016.1m.3dg.gz \
  --mask test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/frozen_legacy_mask_snapshot.npz \
  --candidate-id oracle --outdir test_res/051-20260916_142401-oracle-vs-baseline-chr1-matrix/oracle_vs_reference

python scripts/plot_3dg_comparison.py \
  test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.3dg data/P9016.1m.3dg.gz \
  --mask test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/frozen_legacy_mask_snapshot.npz \
  --candidate-id baseline-046-G-full-J --outdir test_res/051-20260916_142401-oracle-vs-baseline-chr1-matrix/baseline_vs_reference
```

两个候选结构**直接对比**时既有 CLI 不适用：它把第二个文件按 reference 命名读，而 oracle 与 baseline 的 track 都叫 `c01a/c01b`，会全部读成 NaN 并在 `build_matrix` 抛 `Pearson comparison is not finite`。为此新增一个同风格的小入口（本目录图 3）：

```bash
python scripts/plot_3dg_pair.py \
  test_res/014-20260912_153000-s0-genome-wide-fixed/coords/oracle.3dg \
  test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.3dg \
  --a-label "oracle-fit (true labels)" --b-label "baseline 046 G-full-J" \
  --mask test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/frozen_legacy_mask_snapshot.npz \
  --outdir test_res/051-20260916_142401-oracle-vs-baseline-chr1-matrix/oracle_vs_baseline
```

## 图

| 图 | 内容 | matched / cross（chr1，17,578 共同 pair） |
| --- | --- | --- |
| `oracle_vs_reference/plots/chr1_1Mb_distance_matrix.png` | reference mat/pat + oracle 两拷贝 | 0.411694 / 0.118066 |
| `baseline_vs_reference/plots/chr1_1Mb_distance_matrix.png` | reference mat/pat + baseline 两拷贝 | 0.697157 / 0.401146 |
| `oracle_vs_baseline/plots/chr1_1Mb_pair_distance_matrix.png` | oracle 两拷贝 + baseline 两拷贝（同列 = 同一侧） | 0.260440 / 0.146200 |

前两张图各自的四个 panel 共用**该图内**的一个尺度（reference 取 `mat+pat` 中位数、candidate 取两拷贝中位数），所以图与图之间的深浅不可比；第三张四条 track 共用一个尺度，panel 之间可比。灰色 = 缺 bin 或非同源 pair。方向 `auto` 按 Pearson 均值择大。

## 机器读数（chr1，17,578 个共同有限非对角 pair，全为 Pearson）

| 量 | oracle | baseline | reference |
| --- | --- | --- | --- |
| 同一结构内部两拷贝（copyA~copyB / mat~pat） | **+0.142** | +0.367 | +0.342 |
| 距离向量 raw std | 0.246 / 0.267 | 0.118 / 0.144 | 0.574 / 0.597 |

四方向相关：

- oracle × baseline：A~A `+0.355`，A~B `+0.148`，B~A `+0.144`，B~B `+0.166`
- oracle × reference：A~mat `+0.110`，A~pat `+0.405`，B~mat `+0.418`，B~pat `+0.126`
- baseline × reference：A~mat `+0.398`，A~pat `+0.794`，B~mat `+0.601`，B~pat `+0.404`

20 条染色体 macro：oracle 内部两拷贝 `0.217`、baseline `0.226`、reference `0.262`；baseline 四条相关 A~mat `0.469`、B~mat `0.458`、A~pat `0.505`、B~pat `0.501`（mat 与 pat 之差 `0.039`，只有 1/20 chr 出现"两拷贝都更像同一侧"）。

## 限制

- **第三张图的方向标签与 reference 标签无关**：`oracleA↔baselineA` 是在 oracle/baseline 之间按相关择大的，而 oracleA 在 reference 里更靠 pat（`A~pat 0.405 > A~mat 0.110`）。不要把 panel 标题里的 copyA/copyB 直接当 mat/pat。
- baseline 的距离矩阵接近均匀（raw std 约 0.12–0.14，面板上几乎整片同色），这是"两拷贝被压到接近共识几何"的视觉表现，不是绘图归一化造成的。
- 这是对**已冻结端点**的描述性对照，不是预算受控的因果比较，也不改变任何 L1/L2/L3 结论；L2 仍未证明。
- 中间失败证据：曾把 baseline 当第二个位置传入既有 CLI，报 `ValueError: Pearson comparison is not finite for this grid/mask`（`oracle_vs_baseline.err` 已随目录删除，原因记录在本文件与 `scripts/plot_3dg_pair.py` docstring）。
