# Shared-Capture Round 内部执行计划

## 1. 本轮目标与边界

运行目录：`test_res/045-20260915T073310Z-shared-capture-round/`。

本阶段只先完成准备、实现和数值门，formal 尚未启动。用户已授权全部正式拟合与后置评价；数值门通过、父侧快速验收后，同一执行通道自动启动 14 fits，随后按候选/初始/null 封存、reference/truth gate、evaluation 和报告流程完成本轮。

固定科学问题：

- A：共享捕获强度是否改善 inter 空间几何和 homolog 结构恢复。
- B：已知真值下 near 与 blind 的差异来自优化/预算，还是联合目标与正则偏差。

homolog 主目标固定为每条染色体 strict upper triangle 上的 matched Pearson 相关、matched-cross contrast、两 copy margin 和 min margin；中心距离、1-rho、假设性的片段交换不作为 homolog 恢复证据。Spearman 同口径并列保留。

合成真值来自 `pr/v1_calibration.py:generate_truth` 的纯生成逻辑，不读取 reference；该生成器的 `prior_equilibrium_claim=False`。生成链不是完整正则 prior 的平衡分布抽样，因此 full-J 漂移只能定位本夹具上的 prior 偏置，不能直接推出真实生物结构的 prior 错误。

## 2. 固定模型与目标

完整 eligible offdiag grid 始终保留零计数 pairs；same-bin 是独立饱和 nuisance，不进入 shared 结构归一化。`n_loci` 为实际 full header grid，`r0=2*(2*n_loci)^(-1/3)`，

`K=1e-6+(1-1e-6)*(1+d^2/r0^2)^(-2)`。

`rate=e_i e_j*[cis 0.5*(p*(KAA+KBB)+(1-p)*(KAB+KBA)); inter 0.25*(KAA+KAB+KBA+KBB)]`，`p` 使用原 bounded-q 参数化。

- S：`L_S=(Ncis*log Scis + Ninter*log Sinter - sum C*log r + diag_nll)/Nraw`。
- G：`L_G=L_S+sum_g Ng*log[(Ng/Noff)/(Sg/(Scis+Sinter))]/Nraw`。
- G 新项为 group-mass KL；允许仅有 float 舍入微小负值。G 梯度的每组 normalizer 系数为 `Noff/(Scis+Sinter)`，S 为 `Ng/Sg`；observed-log 项严格相同。
- `J=L_count+1*bond+1*repulsion+0.01*bend+1*p_prior`。`p_prior` 内部已有 `1e-4`，不额外乘第二遍。
- real 使用每层固定 production `e`；synthetic 全部使用 manifest 授权的 known generating `e`；不做 visibility profile、不做 M1 预条件。
- count-only synthetic 仅将 bond/repulsion/bend/p_prior 置 0，保留球坐标与可优化 `p`；不是另一套 counts。

## 3. 输入与起点

真实输入：`inputs/P9016.snpfree.pairs.gz`，SHA256 固定为 `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`，全 1,703,888 条、20 chr、40 tracks，不拆 train/test。准备器序列化 5Mb/2Mb/1Mb 三个完整 aggregate NPZ，包含完整 pair index、cis mask、零 counts、diag、exposure 和所有 budget metadata。

真实层 budget：

- 5Mb：diag 607,552，cis-offdiag 527,902，inter 568,434。
- 2Mb：diag 516,046，cis-offdiag 619,408，inter 568,434。
- 1Mb：diag 438,774，cis-offdiag 696,680，inter 568,434。

real S/G 的 consensus/random 在同一 source 上共享精确 5Mb 坐标、raw-y、q 和 seed；S 本轮重跑，不复制 041 历史轨迹。5Mb 初始来自 014 approved blind source（consensus seed 1103，random seed 2207），之后按原 continuation 插值并 bit-exact carry q；不把旧优化 1Mb endpoint 当新 5Mb 初始。

synthetic 固定 P2/N2，1Mb 2,645 loci/5,290 beads：

- P2 truth seed 260104，`same_shape=False`；N2 truth seed 260102，`same_shape=True` 且空间分离；`p_gen=.8`。
- 生成 exposure sigma=.4、无 dropout，P2 seed 260304，N2 seed 260302；与既有 known-e manifest 向量逐元素核对并复制到新 run。
- joint shared rates 直接使用 `Cij=1,265,114*r_ij/sum_all_offdiag(r)`；不强制 cis/inter quota。diag 复制 SNP-free 1Mb template diag 向量（438,774，nuisance）。counts 保持 float64，严禁 int cast；记录实际 expected group mass，Nraw 目标为 1,703,888。2026-09-15 preparation correction：曾发现并删除按 real cis/inter 配额二次缩放的错误实现；本次重生成仅允许一个全局倍乘（实际 P2/N2 `group_scale_ratio_ptp=1.42e-14`，`group_quota_rescaling=false`），truth/near/blind seeds 与坐标未改变，修正后 expected/endpoints/manifests/hash 已重新写出。
- near：`y0=sphere_inverse(x_truth)+Normal(0,.05*l0)`，P2/N2 noise seeds 450101/450102，再 `sphere_forward(y0)`。这是 raw-y 噪声的 oracle-informed calibration 起点，不是盲恢复；无拒绝重抽、缩放或调 seed，直接 assert inside unit ball，并记录起点误差。
- blind：独立 `generate_truth` seed P2 260404/N2 260402、`same_shape=False`，按旧规则全局居中后缩放 maxradius=.8，再转 raw-y；不读取 truth 路径。两个起点 p_init 均为 .8，S/G 共用精确 x0/raw-y/q 和 count/e hash。

truth 只写 `eval_truth/`，formal worker manifest 不含 truth path/token；controller 不 import evaluator。

## 4. Formal matrix 与预算

正式矩阵共 14 fits、22 stages：

- real 4 fits：`S/G × consensus/random`，每 fit `5Mb -> 2Mb -> 1Mb`，FG caps `612/404/486`，共 12 stages。
- synthetic full-J 8 fits：`P2/N2 × S/G × near/blind`，每 fit 1Mb FG cap 486。
- synthetic count-only 2 fits：`P2 × S/G × near`，每 fit 1Mb FG cap 486。

总 hard outer FG 上限：`4*(612+404+486)+10*486=10,868`。所有 formal arms：`ftol=0`、`canonical_gtol=1e-6`、`maxls=20`、`maxiter=FGcap+1`，使用冻结 037 `run_budgeted_lbfgs` 的 exact FG accounting；不设额外 300/200/240 accepted-iteration 早停。终点是最后 callback-confirmed accepted state；未接受 line-search 点丢弃。终态如实标记 `converged`、`budget_not_converged`、`not_converged` 或 `failure`。

每个 accepted state 保存标量 `fun/group KL/Scis/Sinter/Scis+Sinter/profiled intensities/p/weighted components/gradnorm`；每 10 accepted 保存物理 coords，另保存 iter0 和最终完整 40-track `.3dg`、NPZ、raw-y/q 和 hash。异常恢复不得静默接续失败 run。

## 5. 实现文件

- `source/shared_capture_objective.py`：继承 041 float64 backend，仅替换 S/G count normalizer 与可配置 penalty weights；包含坐标与 p 链式梯度、G KL audit、count-only。
- `source/data_io.py`：full-grid aggregate/start 序列化和 array/file hash。
- `source/prepare_inputs.py`：real full-scale aggregates、纯 synthetic truth/rates/expected counts、known-e、near/blind 起点、real zero-optimization initial controls、worker/evaluation manifests；不调用 optimizer。
- `source/formal_controller.py`：统一 14-fit/22-stage formal 入口，明确拒绝非 `READY_FOR_FORMAL` 状态；不 import evaluator。
- `source/post_evaluator.py`：候选/initial/null hash gate 后读取 synthetic truth 或真实 reference 的后置评价器。
- `source/visibility_profile_base.py`、`source/gpu_variant_backend.py`、`source/m1_preconditioner.py`、`source/frozen_pr/`：只读字节复制，source hash 写入 manifest。

## 6. 必要预检

1. source hash 与 041/037/frozen-pr manifest 核对；S 与 041 V0 fixed objective 在相同 real 5Mb 与小 synthetic 上 value/gradient parity，不能改旧 041。
2. G value 独立核对 `S+KL`；G raw-y/q gradient 与独立 autograd 或避开 hinge 边界的 central difference 核对。
3. 验证 G 只改 normalizer correction，不改 observed-log 项；samebin 不变；全体 rate/common exposure 倍乘时 S/G profile prediction 与结构 gradient 不变；仅 inter rate 倍乘时 S 不变、G 一般改变；counts 整体倍乘只要求归一化结构梯度/去数据常数后的 CE 不变，samebin Poisson 常数可变。
4. 每条 chromosome A/B whole-swap gauge、坐标有限/球域、完整 grid/零 pairs、q carry、3DG roundtrip/schema/hash。
5. expected truth count-only gradient 近 0、KL 近 0（P2/N2，S/G）作为 calibration，不冒充恢复实验；Float64 mass 守恒且无 int cast。
6. 短 integration 只走同一正式 `stage_fit` 入口：real 两 source、real S/G 主要 branch、synthetic near/blind/fullJ/count-only 取 FG=2；结果只写 integration 子目录，不扩大 formal budget。
7. shell 每条命令记录实际 exit code；preflight 前后核对旧 source/native/旧结果未改，reference/phase/evaluation 未打开。

## 7. 后置评价预注册

候选、real 初始（每个 014 source 的实际 5Mb root 经同一 5->2->1 zero-optimization prolongation）和 null 坐标都在首次 reference 读取前写出并完成 hash。真实 R2 不按新候选动态增减 mask，而复用既有 frozen old-21 mask：位置为数值 `range(3000000, chromosome_length_bp, 1000000)`，每 chr 对 `positions` 取 unordered strict upper triangle，固定 `pair_i/pair_j/common`；来源为 `docs/audits/multires-r2-preparation-20260914_041826/mask_lock.json` 及其 `test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/evaluation_manifest.json`，lock SHA256 分别为 `d0c325dea1289374152a327557605f8bbd52267b025cd7b57689be3e3666c49e` 和 `fa1b26c834c173604c21f954d494cece8e053dd970e7cb6111f561b349574acb`。固定 21 condition、176,201 个全非对角 pair、157,529 个 common pair、20 chr；候选不进入 mask，mask payload 仅在全部新候选/初始/null hashes 之后读取，并按每 chr `positions/pair_i/pair_j/common` 和 expected counts 重建核对。训练仍使用 origin-0 完整 grid；synthetic 评价仍使用 origin-0 full grid，不套 real 缺失 mask。

真实 R2 对每个 frozen common mask 做 Pearson 四相关、matched/cross/contrast、copy-A/B margin/minmargin 和 macro；Spearman 同规则。若 direct/cross mean 差 `<=1e-12`，标记 `unresolved_tie`，matched=cross=两方向均值、contrast=0，named margins/minmargin 为 null；MIN_COMMON_PAIRS=20，报告每个 metric defined/undefined 的 20-chr 分母，不把 null 当改善。G-S 逐 chr 胜出数及 paired chromosome bootstrap seed 450301、10,000 draws、95% CI；20 chr 是一个细胞的技术/结构测量，不是生物重复。

inter 使用完整公共 inter locus-pair grid 的四 copy 距离排序，denominator 固定 `4*eligible common inter locus-pairs`；同口径比较 S/G、相应初始、u=0、16 个 fixed random-u（seed 450500..450515）。排序是 order-statistic，绝对 inter r 不代表 packing 恢复。

N2 评价使用两真值距离矩阵均值作为 common truth matrix，避免浮点 tie；同时报告每 copy 距离 RMS、common-scale 两 copy RMS 差、各自 RMS 归一后的矩阵差、相对 common truth 的 shape error。不得仅凭 common-scale RMS 增大宣称假形状拆分。所有恢复解释坚持 matched/cross/margins，不使用 `1-rho`。

random-u：每 chr 对 `u=(XA-XB)/2` 做固定 seed 的 locus permutation，`z=(XA+XB)/2` 保持；超球域只允许一次共同 global scaling，相关性不变。u=0 与 random-u 仅后置评价，不拟合。

## 8. 预期结算

本阶段结束时 `config.json` 应由 `prepared_no_optimizer` 经所有数值门通过后冻结为 `READY_FOR_FORMAL`，并记录：精确 run/PLAN/config 路径、source/input hashes、14-fit/22-stage matrix、FG 10,868、每项检查结果和实际 exit code、实测 benchmark（若有）、以及具体未解阻塞。父侧快速验收后自动启动已授权 formal，完成 14 fits、端点/初始/null hash 封存、后置 reference/truth evaluation、图表和最终报告；不因阶段性 READY 状态把整个实验写成“未授权”。
