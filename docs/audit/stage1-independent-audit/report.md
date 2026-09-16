# 独立审计——Stage 1（chr1 片段交换），从原始输入重新推导数字

`audit/verify_independent.py` 只根据协议常数重新实现全部步骤（pairs 解析、uint64 fold 哈希、留出 bin-pair 计数、参考分类器、验证集选择的指数、留出 Spearman、Kabsch 嵌合拼接、列审计和哈希 gate）。它**不**从 `pr/` 或 `run.py` 导入任何内容。原始标准输出：
`audit/verify_independent.out`；机器可读结果：`audit/verify_independent.json`。
审计从 `results.json`、`logs/run.log` 和 `gate.json` 读取主张。

核心结论：**8 项检查中有 5 项（1、2、3、4、7）完全一致，2 项（5、6）只在运行未声明的缩小评分总体上一致，1 项（8）失败。**所有差异都已精确解释到最后一位打印数字。

## 判定表

| # | 指标 | 运行主张 | 独立重新推导 | 判定 |
|---|---|---|---|---|
| 1 | `data/P9016.pairs.gz` 中的 chr1 cis 记录 | 91,703 | 91,703 | **AGREE** |
| 1 | same 1 Mb bin | 31,827 | 31,827 | **AGREE** |
| 1 | 完全定相（phase0/phase1 均为 0 或均为 1） | 28,606 (`n00` 15,638 + `n11` 12,968) | 28,606 | **AGREE** |
| 2 | train / val / test records | 54,420 / 18,155 / 19,128 | 54,420 / 18,155 / 19,128 | **AGREE** |
| 2 | fully phased train records | 16,928 | 16,928 | **AGREE** |
| 3 | held-out unique bin pairs, val / test | 1,619 / 1,622 | 1,619 / 1,622 | **AGREE** |
| 4 | chr1 cis 参考准确率，排除同区间 | 0.7903 | 0.790303 (n=18,150) | **AGREE** |
| 4 | tie rate | 0.0000 | 0.000000 (0 ties) | **AGREE** |
| 5 | oracle 留出 rho | 0.6402 | 0.6402315207838114 (1,603 pairs) / 0.6388918168625625 (all 1,622) | **AGREE\*** |
| 5 | `gauge_all` held-out rho | 0.6191 | 0.6190719928703486 (1,603) / 0.6168765369827813 (1,622) | **AGREE\*** |
| 5 | gauge asymmetry | 0.0212 | 0.02115952791346276 (1,603) / 0.022015279879781136 (1,622) | **AGREE\*** |
| 5 | 单个文件内的轨迹顺序影响 | 未报告 | exactly 0 | **AGREE** |
| 6 | `splice_wall1` rho | 0.6105 | 0.610461539425788 (1,603) / 0.6074848012069358 (1,622) | **AGREE\*** |
| 6 | Procrustes RMSD (`cc00b`→`cc00a`, 194 beads) | 2.301162823 | 2.301162823163265 | **AGREE** |
| 7 | `work/` + `coords/` 中无 `phase`/`prob` 字段、7 列 | 无违反 | 扫描 73 + 73 个文件，0 项违反 | **AGREE** |
| 8 | `coords/` 下每个 `.3dg` 的 `gate.json` sha256 | 必需 | 73 个中 37 个存在，37 个全部匹配，**36 个无哈希** | **DISAGREE** |

`AGREE*` 表示：主张值被精确复现，但只是在运行实际使用的 1,603 个配对评分总体上，而不是协议和日志描述的 1,622 个留出配对上（差异 D1）。

子检查合计：18 项 AGREE、5 项 AGREE*、1 项 DISAGREE。

## D1——留出分数使用 1,603/1,622 个配对（检查 5 和 6）

`run.py` 在 `MASK_IDX["all"]` 上为每个候选评分，即使用**所有**候选有限距离掩码的交集。两个半数据诊断拟合各缺少一个珠子，因此 19 个 test 配对（1.17%）会从整个比较中静默丢弃，包括 oracle 参考和 bootstrap。

```
-- run scoring universe --
  candidates declared in results.json: 37
  candidates with non-finite test distances: {'oracle_half1': 13, 'oracle_half2': 6}
  run's MASK_IDX['all'] would be 1603 of 1622 test pairs (results.json stratum_sizes.all=1603, log says 1622)
    oracle_half1/cc00a: 194 beads, missing bin starts=0 []
    oracle_half1/cc00b: 193 beads, missing bin starts=1 [8000000]
    oracle_half2/cc00a: 193 beads, missing bin starts=1 [192000000]
    oracle_half2/cc00b: 194 beads, missing bin starts=0 []
-- validation-selected exponents --
  oracle   : alpha=2.50 (val rho 0.6149) -> test rho 0.638892 on all 1622 pairs, 0.640232 on the run's 1603-pair universe  [run 0.6402]
  gauge_all: alpha=2.50 (val rho 0.5973) -> test rho 0.616877 on all 1622 pairs, 0.619072 on the run's 1603-pair universe  [run 0.6191]
  ...
  splice_wall1 rho=0.607485 on all 1622 pairs, 0.610462 on the run's 1603 pairs (run 0.6105)
```

我在这 1,603 个配对上的数值与 `results.json` 逐位相同（`0.6402315207838114`、`0.6190719928703486`、`0.02115952791346276`、`0.610461539425788`），因此算术是正确的；错误在于评分总体。后果如下：

- `logs/run.log` 打印 `stratum all n= 1622`，但 `results.json` 的 `stratum_sizes.all` 是 `1603`；二者描述的是不同对象，且都没有标注清楚。
- `masked_compare[*].frac_of_test` 以 `len(idx_all)=1603` 计算，因此在缺少 19 个配对时仍会报告候选覆盖 `1.0`（“100% of test”）。
- 偏差方向并不统一：丢弃这 19 个配对会使每个 rho 分别升高 +0.0013（oracle）、+0.0022（`gauge_all`）和 +0.0030（`splice_wall1`），从而缩小 oracle−候选差距（`splice_wall1` 为 0.0314 → 0.0298；规范不对称为 0.0220 → 0.0212）。影响虽小，但这是未声明的自由参数，并且会让运行的标题差值偏向自身。
- 任何地方都没有声明 NaN 政策；一个候选只要在单个 bin 失败，就会静默改变其他所有候选的分数。
- 所有下游数字继承同一评分总体：`root`、所有 `bootstrap` CI、`common_alpha` 表、`gauge_invariant`、`masked_compare` 和 `splice` 都在同一 1,603 个配对上计算。标题 L1 差距（交叉引用为 F16，`random_phased` delta `+0.076 [+0.048, +0.104]`）也在其中。

## D2——`gate.json` 未覆盖 36 个松弛坐标文件（检查 8）

```
-- check 8: gate.json sha256 coverage --
  gate.json entries=37 ; .3dg files on disk=73
  hashed and matching: 37 ; missing from gate.json: 36 ; hash mismatch: 0 ; stale entries: 0
```

每个已记录的摘要都与磁盘文件一致（用独立的 `hashlib` sha256 核验），也没有过期条目。但 73 个文件中有 36 个完全没有条目——它们正是 `run.py:relax_candidate` 写出的 `coords/*_relax.3dg` 文件（该函数调用 `fdg.run_fdg` 和 `evaluate`，却从未调用 `gate.register`）。这些结构在 `results.json["relaxation"]`（`rho_pooled`、`estep_agreement`）中被评分和报告，因此评估阶段内部生成的坐标文件绕过了 gate 要求的“写出后再哈希”规则。

## D3——次要问题：“测得而非假定”的方向分支是死代码（检查 4）

`pr/ref3dg.reference_accuracy` 循环 `for orient in (0, 1)`，但循环体除平局项外从未使用 `orient`，因此 `acc_orient1` 是 `acc_orient0` 的副本（`results.json`：`acc_orient0 == acc_orient1 == 0.7903030303030303`，`orientation = 0`）。
我正确交换后的计算给出 `acc(orient1) = 0.209697 = 1 - acc0`（无平局），因此方向 0 确实更好，报告的 0.7903 不受影响。但“方向是测得而非假定”的主张仍未在实现中成立。

## D4——次要问题：运行目录的 provenance

- `results.json` 没有 `n_ph_train` key，尽管 `run.py:598` 会写入它。因此运行的 `results.json` 早于当前 `run.py`（mtime 12:03；运行在 12:02 开始）。
- 定相训练主张（16,928）取自 `logs/run.log`，并已独立复现。

## 检查 5——轨迹顺序效应有多大？

- **在同一个文件内它精确为零。**`d_A^-a + d_B^-a` 具有对称性；交换两条轨迹后，我对 `oracle.3dg` 和 `gauge_all.3dg` 都测得 `|Δrho| = 0.000e+00`。
- **两个文件确实互为规范补集。**`work/oracle.pairs.gz` 和 `work/gauge_all.pairs.gz` 的记录 key 完全相同，且 **100.0000%** 的记录轨迹标签不同。
- **结构层面的拷贝对应得到确认：**匹配配对（oracle `cc00a` 对 gauge_all `cc00b`，二者在同一接触总体上拟合）给出配对距离矩阵 Spearman **0.989146**；同名配对只有 **0.188825**。
- **报告的 0.0212 不对称性来自两个独立 FDG 拟合，而不是指标中的轨迹排序。**两个拟合的结构相关为 0.989；两个近似相同运行之间的残余摆动才是要求读者当作零假设容差的部分。
- 单拷贝读数（alpha = 2.5）：`O_cc00a` 0.4742、`O_cc00b` 0.4523、`G_cc00a` 0.4307、`G_cc00b` 0.4748；两拷贝和为 0.6389/0.6169。跨文件组合：`O_a+G_a` 0.6191、`O_b+G_b` 0.6365、`O_a+G_b` 0.4755、`O_b+G_a` 0.4520。最后两项是匹配配对，并降到单拷贝水平——当两项描述的是**同一结构**时这是预期结果：两拷贝增益（相对单拷贝 rho 约 +0.17）来自求和两份真正不同的拷贝，而不是来自轨迹标签。

## 检查 6——splice 仪器，独立复现

在 194 个共同珠子上执行自有 Kabsch（不缩放、修正反射），构造嵌合体 `A' = cc00a`，除 bin 0..19（20 Mb）使用叠加后的 `cc00b` 外其余位置不变；`B'` 交换两者角色；使用 oracle 的 validation alpha（2.5）评分：

```
  Kabsch cc00b->cc00a on 194 common beads: rmsd=2.301162823 (run 2.301162823)
  Kabsch cc00a->cc00b (roles exchanged): rmsd=2.301162823
  chimera scored at the oracle's validation alpha 2.50
  splice_wall1 rho=0.607485 on all 1622 pairs, 0.610462 on the run's 1603 pairs (run 0.6105)
  oracle rho=0.638892 (1622) / 0.640232 (1603) (run 0.6402) | delta_vs_oracle=0.029770 (1603)
  splice_wall1 rho over the alpha grid (run's 1603-pair universe): 0.50:0.6110 0.75:0.6121 1.00:0.6128 1.25:0.6128 1.50:0.6128 2.00:0.6118 2.50:0.6105 3.00:0.6089
  best-on-test alpha would give 0.6128 at a=1.25 (not the protocol's choice)
```

RMSD 精确到机器精度一致；使用运行的 1,603 配对总体后，rho 也一致到最后一位。指数选择不是解释：整个网格只跨越 0.0039，且协议指数（正确地在验证集上选择）是网格中倒数第二差的点，即 splice 差值不是指数伪影。

## 次要观察（不在要求的 8 项检查内）：chrX 运行继承了同一缺陷，而且更严重

在重建评分总体时，我检查了兄弟运行 `test_res/002-20260912_121000-stage1-fragment-swap-chrX/results.json` 中的同两个数字：

```
002-...-chrX     pairs_test = 1059     stratum_sizes.all = 583
```

也就是说，**1,059 个留出 chrX 配对中有 476 个（45%）从每个报告的 chrX 指标中丢弃**，包括 F19a 引用的 oracle−random 差距。机制相同：`common = 所有候选有限掩码的 AND`；但 chrX 中不是两个诊断拟合的问题，几乎每个候选的 FDG 输出都缺少珠子（例如 `oracle`：`cc19a` 缺少 168 个 bin 起点中的 13 个，`cc19b` 缺少 22 个；半数据拟合缺少 23 和 33 个）。chr1 中同一规则只损失 19 个配对，因为 `oracle.3dg` 和 `gauge_all.3dg` 都有完整珠子集合。我只核验了这个数量，没有重新推导 chrX 运行的其余部分。

## 无法复现的部分

- 在重建运行评分总体后，检查 1–7 没有任何一项无法复现；没有未解释的残差。
- 我没有重新运行 FDG，因此无法核验 `coords/*.3dg` 是否正是引擎从对应 `work/*.pairs.gz` 生成的结果；我改为核验配对文件确实是所称分配，并核验所有已记录坐标哈希与磁盘一致。
- 运行将带有不一致 phase 列（`01`/`10`）的 512 条 chr1 cis 记录视为未定相；任务措辞（“两列都为 0 或都为 1”）与此一致，但较宽松的“两列都有值”解读会得到 29,118，而不是 28,606。

## 文件

| 文件 | 内容 |
|---|---|
| `audit/verify_independent.py` | 独立重新推导（stdlib + numpy + scipy） |
| `audit/verify_independent.out` | 上述运行的原始标准输出 |
| `audit/verify_independent.json` | 机器可读主张 / 派生值 / 详细信息 |

复现：

```bash
source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh && conda activate analysis
cd /mnt/ssd/zliu/phase_restart
python test_res/001-20260912_120223-stage1-fragment-swap-chr1/audit/verify_independent.py
```

任一子检查不一致时退出状态为 1（当前为 `gate.json` 覆盖检查）。
