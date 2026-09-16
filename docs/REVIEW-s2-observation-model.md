# S2 与观测模型独立审查

## 范围、结论与边界

本审查只读检查了当前工作树的入口、`pr/`、测试、native FDG 源码和最新冻结结果；没有修改生产代码、已有数据或结果，也没有运行正式重建、FDG 长运行或参数搜索。所有 Python 命令均在 `analysis` conda 环境中运行。

**核心结论：**当前没有 S2 非线性求解器。`run.py split` 实际实现的是 S0（consensus/random/oracle）而非计划中的 S2/S3；且当前真实 `split` 入口在任何 FDG 运行前就因误传原始含 phase 文件而失败。S2 的残差反演、Poisson 求和域、曝光项、same-bin 处理和 S3 整数分配均仍是计划级方法缺口，不能描述为已经运行过的实现 bug。

最新 `017` 是对 `014` 已有坐标的有效、无 FDG 重拟合的评估修正，不是 S2 结果，也不会因当前 `split` 入口故障而失效。

## 冻结数据与实际完成度

| 项目 | 本次实测/源码证据 |
| --- | --- |
| 原始输入 | `data/P9016.pairs.gz`，SHA256 `071a6cc7...b9649505`，20 个染色体 header、1,703,888 条原始接触记录；1,135,454 cis、568,434 inter；列含 `phase0/phase1`。 |
| 训练输入 | `inputs/P9016.snpfree.pairs.gz`，SHA256 `f37ed9cc...a3c9aa`，同样 1,703,888 条、20 个 header、7 列；与 `inputs/manifest.json` 的 source/output hash 和 record 数均一致。 |
| 同 bin cis | 438,774 / 1,135,454 = 0.38643045；非对角 cis 为 696,680。该数是原始接触记录数，不是 unique bin-pair 数。 |
| 当前盲拟合加载器 | `pr/genome.py:52-83` 的 `load_all(SNPFREE)` 逐行读 7 列、保留 cis/inter，并只对 cis 数值规范化端点。`pr/s0.py:25-42` 对 consensus/random 以 `keep=None` 传全部记录给 20/40-track wrapper。 |
| 当前 S2 完成度 | `run.py:809-957` 注释和实现均为 Stage S0；CLI 只有 `prepare`、`stage1`、`split`、`reevaluate`（`run.py:971-1009`）。没有 S2 objective、梯度、eligible-pair aggregator、posterior 或 S3 整数分配模块。 |
| 最新结果 | `test_res/017-20260912_200040-s0-reevaluated` 的 `config.json:2-7` 是 `reevaluate`，`fdg_refit_started=false`。它先校验 manifest 和 014 坐标 hash，再读 phase/reference（`pr/reevaluate.py:424-465`）。 |

`014` 的历史配置固定为 `bin_size: "1m"`（`test_res/014-20260912_153000-s0-genome-wide-fixed/config.json:1-7`）；其日志记录 consensus/random 均用 1,703,888 条记录、random/oracle 均为 40 track（`logs/run.log:1-8`）。`017` 再次盘点为 random 40 track/5,185 beads、oracle 40 track/5,149 beads，且 hash 逐项匹配（`017/.../input-coordinate-verification.json:38-199`）。

## 确认的可复现代码问题

### 1. P0：当前 `split` 入口把原始含 phase 文件交给训练侧 schema gate

**问题。** `pr/paths.py:6` 将 `PAIRS` 定义为 `data/P9016.pairs.gz`。但 `run.py:818-820` 调用 `genome.chrom_lengths(PAIRS)`，而 `pr/genome.py:18-21` 无条件调用 `pairs7.assert_snpfree`。后者按设计拒绝 `#columns` 中的 phase 字段（`pr/pairs7.py:61-70`）。随后 `contacts = genome.load_all()` 本来会正确使用 `SNPFREE`，但永远到不了这一步。

**重要性。** 这是当前实现的阻塞性入口错误；不泄漏 phase，恰恰是 schema gate 正确工作，但调用者给了错误的文件。任一新 S0/S2 运行都在坐标、hash gate、FDG 和 evaluator 之前终止。

**可复现小例。** 在临时目录运行而后删除目录：

```bash
source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh
conda activate analysis
PYTHONDONTWRITEBYTECODE=1 python run.py split --out "$(mktemp -d)" --n-iter 1
```

内部进程退出码为 **1**，栈终止于：

```text
pr.pairs7.PhaseLeakError: refusing to load .../data/P9016.pairs.gz:
#columns mentions ['phase0', 'phase1']
```

该检查未启动 FDG。现有唯一 split 测试把 `run.genome.chrom_lengths` mock 掉（`tests/test_reevaluation.py:236-248`），故 11 项测试全部通过仍不能发现真实入口错误。

**最小验证（修复后才执行）。** 增加不 mock loader 的集成测试：断言 `cmd_split` 的 chromosome header 和 contacts 都来自 `SNPFREE`，且在 phase/reference gate 被 arm 前可以写出并 hash 盲候选坐标。此项必须保持 20 染色体/全部记录的真实输入，不应用单 chr fixture 代替实际入口检查。

### 2. P1：`--bin-size` 接受非 1 Mb，但评估网格固定为 1 Mb

**问题。** `run.py:999-1000` 接受任意字符串 `--bin-size`，经 `run.py:837-839`、`pr/s0.py:40` 传给 `pr/gwfdg.py:61-66` 的 native `-b`。与此同时，`pr/paths.py:11-13`、`pr/genome.py:86-91`、`pr/refeval.py:105-116` 和 `pr/refeval.py:405-418` 均硬编码 1 Mb bin/20 个 bin 的 R3 片段。若将 2 Mb 或 5 Mb 的**最终** FDG bead 坐标读回该 evaluator，它仍会以固定的 1 Mb index 解释，而原始记录也仍以 1 Mb bin 映射。

**重要性。** 只有把 2/5 Mb 的**最终** FDG 坐标直接交给固定 1 Mb evaluator（而不是仅作为 warm start、随后再完成 1 Mb 最终拟合）时，R1/R2 的坐标格与记录格会不兼容，R3 内的候选 bead 覆盖也会稀疏/错位。R3 的物理片段边界仍是 20 Mb；当前源码能推知 finite mask/分母会变化，但不能据此断言具体缺失量或片段边界本身改变。

**小例。** `--bin-size 2m` 可以被 parser 接受并传入 `hickit -b2m`，但 `dense()` 对 position 的索引仍是 `(position - 3 Mb) // 1 Mb`（`pr/refeval.py:111-116`）。

**最小验证。** 要么在入口只允许 `1m`，要么让 bin size 成为 coordinates、`bin_of`、R1/R2/R3、fragment bp 边界和 provenance 的一个同源参数，并用一个 2 Mb 的合成坐标/记录 fixture 验证每条记录映射至正确 bead。此问题不影响 014/017，因为它们的固定配置为 1 Mb。

### 3. P2：正式 `--out` 目录可被重用并覆盖

**问题。** `cmd_stage1` 和 `cmd_split` 都只用 `os.makedirs(..., exist_ok=True)`（`run.py:145-146`、`814-815`），固定写 `coords/{consensus,random,oracle}.3dg`、`work/*.pairs.gz`、`results.json` 和日志。与此相对，`reevaluate` 在创建目录前明确拒绝重用（`pr/reevaluate.py:404-410`）。

**重要性。** 一旦 P0 修复，重复传入某一 formal run 目录会覆盖坐标和 gate/provenance，违反“一次 formal experiment 一个新目录”的项目规则。

**小例。** 同一 `--out test_res/014-...` 没有任何 `exists()` 拒绝路径；输出文件名是候选名而非带运行 id 的新文件名。

**最小验证。** 对已有 formal 目录调用 `split`/`stage1` 必须在读取数据前失败；对新目录才允许写入。此处仅提出验证，不在本审查中修改运行策略。

### 4. P3：Stage 1 的 `--relax-rounds > 1` 并非多轮 E/M

**问题。** `relax_candidate` 在 `run.py:763-788` 的 `for r in range(rounds)` 内一直用进入函数时的 `structs` 算距离；FDG 写 pairs 和 `run_fdg` 只在循环外一次（`run.py:788-794`）。因此 rounds>1 时第二轮没有使用上一轮的新结构，返回值却记录传入的 `rounds`。

**重要性。** 只影响 `stage1 --relax-rounds 2+` 的可选路径；默认是 1（`run.py:981-984`），所以不应把它倒灌为 canonical Stage 1 运行的历史 bug，也与尚未实现的 S2 无关。

**小例。** 令 `rounds=2`，两次 E-step 的 `d0/d1` 均来自同一个 `structs`，而 native refit 调用次数仍为 1。

**最小验证。** 使用 mock `fdg.run_fdg` 断言 multi-round 选项要么每轮都有一次 refit/结构更新，要么 CLI 明确拒绝 `rounds>1`。

### 5. P4：旧单染色体 FDG wrapper 将 `-n` 和 `-s` 放在 `-b` 后；`max_iter` 与 seed 均可能无效

**问题。** `pr/fdg.py:44-48` 先构造 `... -b1m -O OUT -s SEED`，仅在 `max_iter` 非空时再追加 `-n N`。native parser 在读到 `-n` 时设置 `fdg_opt.n_iter`（`native/hickit/main.c:166-168`），在读到 `-s` 时才重设 RNG（`:276-278`），但读到 `-b` 就立刻执行 `hk_fdg`（`:240-263`）。因此两个参数都来得太晚。

**重要性。** `max_iter` 未来若被传入会保留 `-b` 时的默认 1000 次。更重要的是，Stage 1 的 `fit_candidate` 将 seed 传给该 wrapper（`run.py:96-102`），但该 seed 不会影响此次 fit。故“`hickit -s` 天然给出 bit-identical 输出”不能从后置 `-s` 的调用推出为 native 引擎性质。没有在本审查中重建 F14 的完整历史命令，故应复核其 provenance，而不能断言所有历史 F14 都必由同一原因造成。该发现不否定默认 1000 次的既有 014/017 坐标，也不影响真正改变 record split 或 disjoint half-data 的 Stage 1 控制；它只使 seed 不变不能再当作引擎 seed-invariance 的证据。正确排序的 seed 至多是优化初始值重复，仍不是独立数据或生物学重复。

**本次最小 native 复现。** 用无 label/reference 的 6-pair、5-bead fixture，强制 CPU、`-n 1`，然后删除所有 fixture/coordinate：

```text
seed 在 -b 前：  seed 1 -> d8ed57d7388a..., seed 7 -> b030cf33be49... （不同）
seed 在 -b 后：  seed 1 -> d8ed57d7388a..., seed 7 -> d8ed57d7388a... （字节相同）
```

这直接验证了选项时序；并不声称充分迭代的真实 P9016 fit 必然对正确排序的 seed 不同。

**最小验证/修复后测试。** wrapper 必须把 `-n` 和 `-s` 都放在第一个 `-b` 前；用 mock `subprocess.run` 锁定命令顺序，再保留一个 1-iteration native fixture 断言 pre-`-b` seeds 的输出不同。当前 S0 使用的是顺序正确的 `pr/gwfdg.py:63-66`，但该函数没有 seed 参数，故这条问题主要影响旧 Stage 1 wrapper/其 F14 证据解释。

### 6. P5：`split --seed` 未传入 genome-wide native FDG

**问题。** `cmd_split` 把 `args.seed` 传给 `s0.fit`（`run.py:837-839`、`856-858`），但 `pr/s0.py:25-42` 的 `seed` 形参没有被使用；`pr/gwfdg.py:61-75` 也没有 seed 参数或 `-s`。当前 `--seed` 只会影响 random candidate 的 `genome.random_assignment`（`run.py:832-836`、`pr/genome.py:100-106`），不会改变 consensus/random/oracle 的 native 初始化。

**重要性。** 这是 CLI 语义和 reproducibility provenance 的风险：不同 `--seed` 不能被解释成不同 native fit seed。它不使 014/017 无效，也不能替代 P4 的 Stage 1 后置 `-s` 问题。

**最小验证。** 在 run config/log 中分别记录 assignment seed 与 native seed；若需要后者，`gwfdg.run` 应显式接受并在 `-b` 前传递它，并以短 fixture 检查命令顺序。

## S2 计划与观测模型缺陷（不是已实现代码 bug）

### 1. 残差反演使用的二阶式不完整，且计划内部的归一化不一致

令 `v = Z_i - Z_j`、`a = u_i-u_j`、`b = u_i+u_j`，`d=||v||`、`vhat=v/d`，以及 `f(q)=||q||^{-alpha}`。对任意向量 `w`，正确的偶对称二阶式为：

```text
[f(v+w)+f(v-w)] / 2
= d^-alpha * {1 + alpha/(2d^2) * [(alpha+2)(vhat·w)^2 - ||w||^2]} + O(||w||^4).
```

因此计划定义的 cis 混合（`docs/PLAN-genome-allele-split.md:101-110`）应为：

```text
lambda_cis = s_cis * [p_cis * g(u_i-u_j) + (1-p_cis) * g(u_i+u_j)],
```

其中 `g` 是上式。计划 `:124-128` 只从正残差反演一个标量 `|delta|`，既漏掉横向 `-||w||^2`，也漏掉 cross-copy 的 `u_i+u_j` 项；而且按该计划 `:104-105` 的 `/2` 权重，`u=0` 时是 `lambda_0=s_cis*d^-alpha`，不是 `:126` 写出的 `2*s*d^-alpha`（除非另行重定义 `s`）。在实际实现前必须固定这个参数化。

**为什么重要。** 单细胞计数稀疏时，正残差再开平方会系统性选择正噪声；真实的横向几何扰动反而产生负残差。把任何正残差解释为几何幅度，会混进 consensus misspecification、曝光差和抽样波动。

**本次合成数值复现。** 不读 phase/reference，仅用 `alpha=1.5`、`v=(10,0,0)`：

| 扰动 | 精确相对变化 | 上述二阶式 | 计划标量投影 |
| --- | ---: | ---: | ---: |
| `w=(1,0,0)`（平行） | +0.018999060 | +0.018750000 | 1 |
| `w=(0,1,0)`（横向） | -0.007434971 | -0.007500000 | 0 |
| `u_i=u_j=(0,1,0)`：same 项 | 0 | 0 | `u_i-u_j=0` |
| 同一例的 cross 项 | -0.028987109 | 需要 `u_i+u_j` | 未被 scalar same-copy 式表示 |
| `p_cis=0.5` 的混合 | -0.014493555 | 同上 | 不能由 `u_i-u_j` 单独恢复 |

这是一条代数/数值反例，不是对 P9016 的新拟合结果。

**历史佐证的适用边界。** `docs/MEASURED_FACTS.md:556-580` 的 F12 记录了三个独立失败机制：`|delta|/dbar=0.51` 非小扰动；用 FDG consensus 当 `Z` 但代入 true `u` 的 matched contrast 是 -0.007，而 exact midpoint 时为 +0.2665；残差估计 rms 8.99、true rms 0.615。`docs/exploration/README.md:46-54` 将 spectral/signmap/construct 脚本列为 caveated（旧 string filter），仅说明 midpoint 结论在 `signmap_check2.py` 数值过滤后重推。本审查没有重算 F12，故把它作为历史诊断，不能当作当前全基因组校准。

### 2. `p_cis=0.5` 时，计数似然对逐 bin 局部 A/B 交换也不变

当 `p_cis=0.5` 时，cis 的四个组合均为 `s_cis/4`；在任一 bin 将 `X_i,Y_i` 互换只重排四项。trans 本来就是四项等权，亦有同样的局部置换不变性。此时接触似然不只具有每条染色体整体 2^20 gauge，而对局部符号也没有区分力；链平滑/聚合物 prior 才承担连贯 `u` 场的主要约束。

**重要性。** 不能仅因优化收敛就声称 contact 证据恢复了 L2。`p_cis` 必须预先定义可辨识的拟合方式，报告 profile likelihood/不确定性和其与 0.5 的距离；否则 coherent output 可能主要由 prior 决定。

**最小验证。** 在不读 phase/reference 的 40-track 合成数据上，对一个单 bin local swap 数值比较完整 likelihood；分别固定 `p=0.5` 和一个有明确置信区间的 `p!=0.5`，确认哪一部分数据项而非正则项产生差异。

### 3. Poisson 写法未定义观测单位、eligible-pair 零项和曝光

计划 `:99-113` 写了 `sum_p [lambda_p-c_p log lambda_p]`，但没有定义 `p` 是每条原始接触记录还是聚合的 bin-pair，也没有定义 eligible bin-pair 集 `E`。这是决定性问题：

- 若用聚合计数 `C_ij`，Poisson NLL 必须对全部 `E` 求和，零计数对需要包含 `lambda_ij` 项；只有观察到的正 count pairs 不足以给出正确归一化。
- 若条件在总原始接触记录数 `M`，应使用 multinomial/point-process 形式；所有未观测 pair 通过 `sum_(ij in E) e_ij K(d_ij)` 的归一化进入，而不是当作“硬远距/必为零”。
- 每行 pairs 是原始接触记录；相同 bin-pair 可以有 0、1、2... 条，不能把 contact 当二元。用户举的是不同距离可能产生同一 0/1 观测的例子；概率模型允许这种现象。例如，**仅作假设**，若某一 bin-pair 的期望计数 `mu=0.5`，则 `P(C=1|0.5)=0.303265`；若另一个 pair 的 `mu=0.4`，则 `P(C=0|0.4)=0.670320`。这些 `mu` 不是由用户给出的距离 `1/2`、`6/15` 换算而来，也不是对任何特定 P9016 pair 的估计。
- `s_cis/s_inter` 不能替代 bin-pair exposure：需要明确 `e_ij`（可及性、ligation/mappability、bin 容量和可能的 genomic-distance nuisance）及其约束，否则残差混合几何与测序/选择偏差。

**最小验证。** 先用固定 20 染色体、40 track、1 Mb bin 边界和全部 1,703,888 总分子数的 P3 合成数据，明确生成 `E`、exposure 和 0/1/multiple counts；比较完整 Poisson/条件 multinomial 目标与“只对 observed positives 求和”的目标。全程不读真实 phase/reference 来选模型。

### 4. `f(d)=d^-alpha` 与“全部接触”在 same-bin 处未定义；核应有限且有曝光项

计划 `:191-197` 要求使用全部记录，`S2` 又定义 `f(a,b)=|a-b|^-alpha`（`:107-110`）。对 same-bin 理想 bead 距离为零，强度发散。native FDG 确实读取这些 pairs，但在实际 contact-force loop 明确跳过同一 bead（`native/hickit/fdg.c:390-399` 的 `if (p->bid[0] == p->bid[1]) continue`）；这是“保留记录”不等于“每条记录都有结构力”，不是 native bug。

**建议的最小模型实验。** same-bin 计数作为独立的非结构 nuisance likelihood/offset 保留并单独报告；非对角 pair 使用有限核（例如有限 bin 积分核或带 `r0` 的 bounded kernel）及 exposure。不要静默删掉 438,774 条记录，也不要把它们当作 `d=0` 的无限几何证据。

### 5. native 有跨染色体短程排斥，但没有显式有限半径核球约束

`native/hickit/fdg.c:185-203` 默认 `target_radius=10`、`k_confine=0`、`d_confine=12`。对 native C 源码的 `k_confine|d_confine|target_radius` 全文搜索只有这两个 confinement 字段的赋值；没有径向 confinement force。CPU repulsion 在 `fdg.c:402-447` 的所有 bead 间搜索中确实没有 chromosome 判据，故 40 tracks 共享短程 excluded volume；backbone 是按 track 内相邻 bead 加的（`fdg.c:376-389`）。

**重要性。** `docs/PLAN-genome-allele-split.md:22` 和 `pr/gwfdg.py:7-9` 把“跨染色体排斥”说成“引擎已经实现有限核球”，说法过强。它能形成共同的排斥体积/紧凑云团，但没有 `||x||<=R` 或 radial potential；`target_radius` 是初始化和单位尺度，不是硬边界。S2/S3 若需要已知核半径或显式核内先验，必须另行定义并测试，不能声称 native 已提供。

### 6. S3 的软 posterior 到整数分配不是概率 EM

计划 `:137-143` 提议将软 posterior 分配成 40 track 的整数接触。对一条 `c=1` 的记录，posterior 例如 0.55/0.45（或四组合）最终仍要变成 1/0；native FDG 的 count-dependent target distance（`fdg.c:390-399`）也不是上述 Poisson likelihood 的 M-step。若接受/拒绝在固定、完整定义的同一 objective 上执行，已接受候选序列可以保证不变差；但 native proposal 本身不保证改善，该过程不能称为概率 EM 或假设 EM 收敛，更不能因此保证 L2。

**最小验证。** P3 中必须独立画出“连续 observation likelihood”与“整数 FDG proposal 后 likelihood”的差值，报告所有 proposal/拒绝，而非只报告被接受的候选。更稳妥的定位是 FDG 为聚合物先验下的候选提议/初始化，直到联合目标被明确定义。

## R1/R2/R3、gate 和 reevaluation：已审查且未发现当前实现错误

- **phase gate / provenance：**`pr/reevaluate.py:71-95` 先核验 manifest 的 raw/SNP-free hash；`155-189` 验证 source gate 中 consensus/random/oracle coordinate hashes 和预期 track inventory；之后才 `gate.arm` 和读取 labels/reference（`424-465`）。017 的 log 与 JSON 均记录此顺序。
- **40-track wrapper：**`pr/gwfdg.py:25-58` 为两端独立 copy track 写 cis/inter；`pr/s0.py:33-40` 的 single 与 diploid 路径正确分开。`gwfdg.run` 将 `-n` 放在 `-b` 前（`61-75`）。
- **R1：**`pr/refeval.py:245-343` 用 labelled、non-diagonal、in-grid、candidate/reference/oracle 都 finite 的同一记录分母；排除原因分别计数，几何 gauge 不用真标签破 tie。
- **R2：**`pr/refeval.py:154-199` 对四条距离轨迹的共同 finite 非对角 bin-pair 集选择 gauge；017 的 oracle-vs-random comparison 再建六轨共同 mask（`pr/reevaluate.py:227-264`）。
- **R3：**`pr/refeval.py:405-500` 正确把 unavailable/tie fragment 从可判分母剔除并断开 run/wall；`frac_consistent=max(n0,n1)/n_valid`，local score 仅超过 `1e-12` 即标记。这不是代码漏报，但解释有风险：`n_walls` 很低可能只是可判 fragment 少/有 gap，且 random 的 majority baseline 天然大于 0.5。017 已给出正确的配对对照：random `frac=0.6787897, walls=2.45`，oracle `0.9536310, 0.55`（`017/.../results/summary.json:16-82`）。未来 L2 报告必须同时给 `n_fragments_total/applicable/tied/insufficient`、paired random/control 差值和单细胞内技术/结构不确定性；一个低 walls 值本身不是 recovery。
- **生物学重复：**017 的 bootstrap 明确是 20 条 linked chromosomes within one cell，不能解释为 biological replication（`summary.json:85-188`）。

## 已检查但不能/不应当声称的问题

1. 没有现存 S2 solver，故不能称“实现漏掉未观测零”“S3 排序错”或“整数分配错”；这些是必须在实现前冻结的设计要求。
2. `docs/PLAN-genome-allele-split.md:168` 承诺 `run.py split --genome`，实际 parser 没有 `--genome`（`run.py:996-1001`）。本次直接调用内部退出 **2**：`unrecognized arguments: --genome`。这是文档/接口不一致；当前 `split` 本意已经是 genome-wide S0。
3. `pr/s0.py:7-9` 的“所有候选共享坐标 frame、可直接与 reference 比较”措辞不严谨；独立 fit 不会共享 reference frame。017 README 和 `pr/viz3d.py:1-8` 已正确按独立中心化/单位 RMS 展示，R1/R2 的距离秩相关也不依赖全局刚体 frame。因此列为文档修辞问题，不列为 017 指标 bug。
4. `docs/MEASURED_FACTS.md:626-650` 仍保留“只依赖 delta^2 / generically determined”的旧叙述，与 `AGENTS.md:143-159` 已纠正的完整向量偶性相冲突；应在方法文档上统一，但不把废弃探索代码冒充当前执行 bug。

## 运行与测试证据

执行环境均为：

```bash
source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh
conda activate analysis
```

| 检查 | 结果 |
| --- | --- |
| `pwd` | `/mnt/ssd/zliu/phase_restart` |
| `git status --short --branch` | `master`；工作树已有用户/项目的未跟踪 `pr/`、`run.py`、`tests/`、plan 等，本审查未回滚或改写它们。 |
| raw/SNP-free hash、header、全记录/cis/inter scan | 两个 hash 均与 manifest 一致；完整 1,703,888/1,135,454/568,434，20 chromosomes。 |
| synthetic vector/Poisson check | 得到本报告列出的 +0.018999、-0.007435、-0.014494、0.303265、0.670320；不使用 P9016 phase/reference。 |
| native seed-order fixture | 6 pairs、5 beads、CPU、1 iteration：pre-`-b` 的 seed 1/7 hash 不同，post-`-b` hash 相同；所有临时 fixture 和坐标已清理。 |
| real `split` 临时入口检查 | 内部 exit 1，在 raw schema gate 终止；临时目录已删除；无 FDG。 |
| documented `--genome` flag | 内部 exit 2，parser 拒绝。 |
| `python -m unittest discover -s tests -v` | **11 tests, 0 failures**, 0.069 s。测试内含 mock 的 tiny `cmd_split` 路径，不能覆盖 P0。 |
| `folds.sanity_check()` / `folds.regression_against_reference()` | 都返回 `True`。 |
| 017 已有 regression log | 同样记录 11 tests、OK（`test_res/017-.../logs/regression-tests.log:1-31`）。 |

没有运行正式 FDG、正式重建、parameter search 或 phase/reference 训练选择；没有在 `scratch/` 留下脚本。

## 建议的最小下一步顺序

1. **先修复/测试 P0 和 formal output reuse，再重新允许任何盲 fit。** 这是运行完整性，不是方法创新；修后仍固定 `SNPFREE`、20 chromosomes、1,703,888 条原始接触记录和 all-fit policy。
2. **冻结 observation model，同时预先限定模型容量和正则预算。** 明确原始接触记录与聚合 `C_ij` 的关系、eligible set `E`、Poisson 或 conditional multinomial、cis/inter/same-bin exposure、bounded kernel、`p_cis` 参数化与其 0.5 可辨识诊断；same-bin 写为独立 nuisance 项而不是无声丢弃。仅靠真实 P9016 的训练 count likelihood 不能同时选择任意模型复杂度和正则强度；在 all-fit 政策下，应在独立合成校准中预先固定容量/正则预算，真实 reference 不参与选择。
3. **先做目标函数校准和最小联合拟合原型，再做正式 P9016 S2。** 原型允许 `Z` 与 `u`（或 `X,Y`）同时移动，且把聚合物/核体积先验和 observation likelihood 写进同一个明确目标；最小梯度/数值 fixture 只用于实现核对，不作为生物学或全基因组结论。
4. **用 P3 作为原型的决策级校准。** 在全 20 染色体/40 tracks、相同 bin 边界和总记录/稀疏度的无 phase/reference 合成数据上，按零噪声、已知采样噪声、真实稀疏度、两 copy 相同的负对照四种条件，验证完整目标是否能区分可恢复、不可恢复和不确定；记录所有初值、完整 likelihood 和 prior contribution。
5. **校准通过后才运行正式全基因组 S2/S3。** `Z` 不应被未经验证地当真 midpoint；初值不应由正残差开方单独决定。S3 若使用 FDG proposal，以固定的 label-free objective 接受/拒绝并报告全部 proposal/拒绝。多初值和数据重采样只报告优化/技术稳定性，不冒充生物学重复；R1/R2/R3 仅在坐标 hash 后报告，reference 不参与选择。

这些步骤不会把单细胞内的 20 条染色体误称为生物学重复，也不会用 phase/reference 来训练、早停或选择模型。
