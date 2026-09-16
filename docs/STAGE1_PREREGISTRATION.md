# Stage 1 预注册：片段交换实验

**在正式运行 `test_res/001-20260912_120223-stage1-fragment-swap-chr1` 之前冻结。**
下述设计是在当时只有 `scratch/` 上的 smoke runs（chr1、bootstrap 100–300）时固定的。正式运行只改变 bootstrap 重采样次数（2000）和输出目录。看到正式数值之后，没有新增或删除任何候选、分层、对照或阈值。

## 问题

无标签 held-out 分数能否区分**整条染色体一致**的拆分与**局部正确但全局错接**的拆分？这决定可以声称什么（L2），也决定 Stage 2 应优化什么。

这是**oracle 仪器检查，不是盲恢复**。这里不提出任何方法。

## 固定要素

| 要素 | 数值 |
| --- | --- |
| 染色体 | chr1 |
| bins | 1 Mb，偏移 3,000,000；193 个 bin（0..192） |
| 片段 | 固定 20 bins = 20 Mb；10 个片段，最后一个含 13 个 bin |
| folds | `benchmark_v1.py` 的 mixed hash % 10；train 0–5、val 6–7、test 8–9 |
| 评分范围 | 至少有 1 条 test contact 的 held-out 唯一 bin pair，且 `b1 != b2` |
| same-bin pairs | 从所有结构指标中排除，单独报告 |
| predictor | `d_a^-α + d_b^-α` 对观测计数；每个候选在 **validation** fold 上选择 α |
| bootstrap | test bin pair 的配对重采样，2000 次，seed 0 |
| oracle record set | 仅 fully labelled train records（报告规则 2） |

## 候选

盲候选（仅 SNP-free 表）：`consensus`（一个结构，全部 train records）、`random`
（随机 50/50，全部 train records）。

零假设系列（无标签、与 oracle 相同的 record set）：`random_phased`、
`random_phased_1`、`random_phased_2`——对同一批 16,928 条定相 train records 做三次独立 iid 50/50 拆分。它们之间的离散范围称为“repartitioning wobble”。

Oracle 仪器（使用标签；仅在 phased train fold 上重新拟合）：`oracle`、`gauge_all`
（所有 bin 重新标号——纯 gauge 变化）、`micro5`、`micro10`（子片段墙）、
`wall1..wall5`（翻转前 m 个完整片段）、`island1..island3`（内部区块）、
`alt`（每隔一个片段），以及第二种 flip rule 下的同一 wall/alt 系列。

预注册两种接触级 flip rule，是因为 domain wall 两侧的接触确实有歧义（chimera 不是可实现的轨迹对）：

- **`relabel`**——仅当较小位置端的 mask 被设置时翻转。这是“交换完整 20 Mb 片段 A/B 标签”的字面解释。
- **`rewire`**——仅当两端的 mask 不一致时翻转。每个片段保留自己的正确内部几何；只打乱跨墙连接。

两种预注册读数都报告：

- **gauge-invariant**——取某个 pattern 及其补集的均值；参考层同样取 `oracle` 和 `gauge_all` 的均值。之所以需要它，是因为 native engine 的双拷贝拟合并不保持“哪个标签分到哪条 track”不变。
- **affected-pair**——只限于候选 mask 在两端不一致的 bin pair（`tb[b1] != tb[b2]`），并在同一批 pair 上使用匹配的零假设。

## 独立对照

- 零假设离散：三次 `random_phased` 拟合的 held-out rho 的范围。
- 引擎 gauge 非对称：`|rho(oracle) − rho(gauge_all)|`。
- 半数据 oracle：phased train records 的两个互不重叠随机半集，标签相同。
- 无 gauge 的 chimera（`splice`）：将 `oracle` 的两条拟合轨迹做 Procrustes 叠加，并在片段层面拼接；不重新标记接触，也不重新拟合，因此完全不携带 gauge 非对称。
- 参考：`data/P9016.1m.3dg.gz`，仅在所有坐标写出并完成哈希之后读取。

## 边界松弛（每个候选预算相同）

进行一轮 hard E-step，只限于每条名义 20 Mb 网格线两侧 ±2 bins 的带状区域（所有候选使用相同的 9 条网格线，无论是否真的有墙）；根据候选自身拟合结构，将带内 train records 重新分配给较近的拷贝，然后重新拟合。记录 sign assertion：使用 oracle 结构时，E-step 在该带上必须与真实标签一致（实测 0.730；规则 5 的阈值为 0.55）。

## 预注册决策规则

`null tolerance = max(range of the three random_phased rhos, engine gauge asymmetry)`。

候选 *X* 只有在以下两个条件同时满足时才算**检测到**：`rho_gi(oracle) − rho_gi(X)` 的配对 bootstrap CI 排除 0，且点估计超过 null tolerance。

**支持的片段尺度**是最小的、能够被检测到的翻转宽度（5、10、20、40、60、80、93、100 Mb）。如果没有任何尺度被检测到，目标必须限制在数据支持的最粗尺度，并明确这样说明。

半数据 oracle 重拟合**不用于** null tolerance：它是另一个候选（数据系统性更少），不是零假设。它的效应量单独作为限制报告，因为它与最小可检测交换处于同一数量级。

## 选择检验（次要、预注册）

在每个系列（`relabel` gauge-invariant、`rewire`、`splice`）内，从 {oracle + 所有错接替代} 的 held-out rho 中取 argmax。只有当 argmax 是 oracle 时，该仪器才对 Stage 2 有用。

## 事后新增项（已记录，但未预注册）

预注册运行之后新增了两项，并明确标注为事后内容：

1. **深度对照**（运行 `009`）：使用 `--ph-subsample 5209` 的 chr1，即 chrX 的定相 train record 数。它是由 chrX 表现不同触发的，没有预注册，属于后续实验而非冻结设计的一部分。
2. **`splice` 仪器扩展**：在无 gauge 构造建立后，将墙的范围从 20–100 Mb 向下扩展到 5 和 10 Mb。

两者都没有改变任何预注册候选、分层、对照或阈值；它们只是同一实验的额外读数。

## 本实验无法说明什么

它只能说明**分数**是否对全局错接敏感。它不能说明任何盲方法能否**找到**一致拆分。即使结果完全为正，**L2 仍未证明**。
