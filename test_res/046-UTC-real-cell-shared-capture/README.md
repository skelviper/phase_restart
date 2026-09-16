# P9016 真实细胞 shared-capture 内部 README

本目录是用户范围修订后的 P9016 真实细胞实验。cohort 只有一个真实细胞，20 条染色体是同一细胞内的关联测量，不是生物学重复；所有 chromosome bootstrap 只描述技术/结构变异。禁止合成输入、truth、synthetic formal fit 和 synthetic evaluation。045 中原已完成的合成检查仅作工程校准归档，不用于本轮恢复结论。

本轮输入为 `1,703,888` 条 raw records、20 条染色体/40 条 SNP-free tracks；SNP-free 输入 SHA `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`。1Mb 固定计数分母 `1,265,114`，其中 diag `438,774`、cis-offdiag `696,680`、inter `568,434`。S/G 的 separate-cis/inter 与 shared-offdiag/group-mass-KL 公式沿用045已验收 PLAN，不在本轮重定义。


基础四 fit 为 `S/G × consensus/random`，每个 `5Mb -> 2Mb -> 1Mb`，caps `612/404/486`，正式 outer FG `6008`。每个模型只按基础 1Mb 的无标签 count NLL 选择 source；差值 `<=1e-9` 取 consensus。延长四 arm 从该模型 selected 1Mb 的同一保存 `raw_y/q` fork，分别 `full-J` 和 `count-only`，每 arm cap `486`，额外 `1944` FG。两 arm 均重置 L-BFGS 历史，属于额外预算的正则/优化诊断，不是 continuation，也不是生物学重复。总计 `8 fits / 16 stages / 7952 FG`。

数值参数固定：`ftol=0`、`canonical_gtol=1e-6`、`maxls=20`、同一 sphere；full-J 权重 `(1,1,1,0.01,1)`，count-only 权重 `(1,0,0,0,0)`。不根据 reference 改参数、预算、seed 或 source。

过程诊断的主 Rg 为整细胞所有物理 beads 的统一读出：`Rg=sqrt(mean_{a,i} ||X_ai-mean_{b,j}X_bj||²)`，mean 覆盖两 copy 的全部 `5290` beads、无 counts 权重。per-chromosome 或跨染色体拼接的 copy Rg 只能明确标作辅助读出，不能替代 whole-cell Rg。


`reused_base/snapshot_manifest.json` 是 cutoff `2026-09-15T10:47:18.820402Z` 的只读 lineage，含 8 个已核对完整 stage：S-consensus 三 stage、S-random 三 stage、G-consensus 5Mb/2Mb，共复用 `4020 FG`。`base_remaining/` 只运行 G-consensus 1Mb（从冻结 G2、seed 1103 prolongation）和 G-random 5/2/1Mb（原始 random start、seed 2207），共 `1988 FG`。最终由 `combined_base_manifest.json` 单一消费入口索引 12 个基础 stage 和 4 个 1Mb endpoint；source selection、extension 起点与后置评价不得读取045动态候选或 selection。

045 的 bash-39 gate reject、bash-40 首个旧14-fit尝试、bash-41/b42 中断、bash-48/49 writer 交错都保留原始证据。未知取消调用不计入正式 FG，只列可证实上下界和 `exit unavailable`。新 base_remaining 与 extension wrapper 使用 `fcntl.flock(LOCK_EX|LOCK_NB)`，FD贯穿 wrapper/子进程生命周期，另写 PID、启动时间和实际 return code。

## 评价边界与分母

唯一正式评价入口是 `evaluation_final/run_evaluation.py evaluate`；旧 `evaluation/` 和旧 `evaluation_attempts/` 仅作失败审计，禁止调用。新 wrapper 用 `fcntl.flock(LOCK_EX|LOCK_NB)` 保证单 writer，并在 `evaluation_final/wrapper_terminal.json` 与 `attempts/` 中持久记录实际子进程 `return_code`、stdout/stderr 和 phase/reference 状态。执行前可用 `evaluation_final/run_evaluation.py validate` 复核冻结输入；正式评价只从本目录冻结清单读取，不改坐标、不重拟合。

正式评价先封存 8 个 candidate endpoints、2 个真实 initial controls、每个 source 的 `u0` 与 16 个 random-u null（seeds `450500..450515`）、评价代码与输入 hashes，再打开 `data/P9016.1m.3dg.gz`。old21 mask 使用 `range(3000000, chromosome_length_bp, 1000000)`，严格上三角总 pair `176201`、common pair `157529`；逐 chromosome 比较 `positions/pair_i/pair_j/common` 与 frozen legacy helper，并验证 `common == valid[i] & valid[j]`。每条染色体记录 `A_mat/A_pat/B_mat/B_pat` 四 rho；direct/cross 差 `<=1e-12` 标 `unresolved_tie`，报告 matched、cross、contrast、copy-A/copy-B 以及方向匹配后的 mat/pat/min margins。inter 只用 old21 common-mask valid bins 的 sorted-four-copy distances，不扩展 finite 交集分母，并给 locus-pair 与 `4x` 分母。

MAP 对每个非对角 pair 使用四个 `wK` 的逐 pair copy-axis argmax：cis 权重 `[.5p,.5(1-p),.5(1-p),.5p]`、inter 权重 `.25`，`K=1e-6+(1-1e-6)(1+d^2/r0^2)^-2`、`r0=2*(2*n_loci)^(-1/3)`；之后按全部 raw counts 加权 intra/inter/all，diag `438,774` 排除。只在 `counts>0` 行计算贡献是严格等价优化，并报告 positive-pair 行数。整细胞 Rg 为 `sqrt(mean(sum((coords.reshape(-1,3)-全体均值)^2)))`，覆盖两 copy 的 `5290` physical beads，不使用 per-copy Rg。

paired-source、selected S/G 和 extension 对照都用全 20 染色体固定分母；undefined chromosome 单独报告，任何缺失都会令 full-20 mean/CI 为 `null`，另给明确标注的 defined-only 描述统计。bootstrap seed `450301`、`10000x20` 固定 index matrix，并输出逐 chromosome wins。全部 170 null 都写出 draw 级与 16-draw summary，包含 matched/cross/contrast/margins 和 inter；null 只固定 z 做同染色体 u permutation，越 sphere 时才做一次统一 global rescale。

最终 JSON/TSV/PNG、MAP/Rg regression、old036 Spearman regression、trajectory checkpoint 与 NA 证据均索引于 `evaluation_final/results/evaluation.json`。图使用英文标签、最多两张、3-inch panel、300 DPI、7 pt。reference 只作评价，不能参与训练、source selection、初始化、停止或超参数选择；所有 stage 的 `budget_not_converged` 原样展示。
