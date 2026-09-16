# 056 GPT Pro 审查实验冻结协议

## 1. 目的与边界

本轮只研究 P9016 单细胞一个生物样本（`n=1 cell`），20 条染色体、40 条 copy 轨迹。
两条初始化 seed 是优化重复，不是生物学重复。训练输入只允许七列 SNP-free pairs；训练、初始化、
停止和 checkpoint 选择不得读取 phase 或 reference。所有候选与对照状态先写出并完成 SHA256，之后才
由独立 reference 评价入口读取 reference。L1、L2 与同细胞 held-out contact prediction 分开叙述。

冻结输入：

- 七列 contacts：`inputs/P9016.snpfree.pairs.gz`，SHA256
  `f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa`，1,703,888 records。
- baseline solver state：
  `test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.npz`，SHA256
  `6bb93bf570cf688fd8b02dccf14ca0867c1f831d5e123036ced626856f8d8752`。
- baseline 3DG：同目录 `1Mb.3dg`，SHA256
  `ee5eb1545db9bfeeabcc24e704707f5f61a5793e6f245091347373442dc0032b`。
- baseline 是 046 `base_remaining/real-G-random`，不是 extension；终态
  `budget_not_converged`，本轮实验 1/2 只固定使用其 `x/e/p`，不继续优化。
- 1 Mb full-header grid：2,645 loci/copy，5,290 physical beads，3,496,690 off-diagonal
  eligible locus pairs。全部 20 条染色体均保留。

`readID` 已核验全部为 `.`；按完整 canonical record 分组也全部为 singleton。因此 molecule identity
不可用，无法保证未知同分子记录不跨 fold。本轮按用户选择 A 继续，明确记录该限制，不宣称 molecule
isolation。

## 2. 公共数值与模型

- Python 使用 `analysis` conda 环境；GPU 为单卡顺序执行；Torch objective 显式 float64。
- 复用 045 original G/full-J/raw objective、fixed-e、sphere parameterization、penalty weights、
  p prior 和 L-BFGS runner；不修改 045/049/051 冻结或历史文件，不引入不同 solver。
- weights：count=1、bond=1、repulsion=1、bend=0.01、p_prior=1。
- optimizer：`ftol=0`、`maxls=20`、canonical raw-y/q `gtol=1e-6`；只保留 finite last-accepted state。
- hard invariant tolerance：absolute `<1e-10`。若 GPU reduction 首先失败，固定状态下用相同公式的
  CPU float64 诊断；不得改变阈值、支持集或算法后继续解释。
- 计数：1 Mb baseline `Noff=1,265,114`（cis offdiag 696,680；inter 568,434）。
- `K0 = Ncis*log(Ncis/Noff) + Ninter*log(Ninter/Noff)`。
- 主 offdiag data 指标（nat/offdiag contact）：
  `(Noff*log(Zall) - sum_offdiag C_ij*log(rate_ij) + K0) / Noff`。
  同时记录不含 K0 的值；K0 不影响任何同 counts 的 delta。diag nuisance 不进入此指标。

## 3. 实验 1：严格 copy 连接对照

固定 baseline `x/e/p`，零拟合，共 46 个且仅 46 个 1 Mb full-grid evaluations：

1. original：1。
2. single-whole-chromosome A/B swap：20，每次只交换一条染色体的完整 copy 轨迹。
3. inclusive suffix swap：8。对 chr1、chr8、chr19、chrX，从 0-based local bin index `s`
   起交换 A/B，冻结 `(floor(n/3), floor(2n/3))`：chr1 `(65,130)`、chr8 `(43,86)`、
   chr19 `(20,41)`、chrX `(57,114)`；1 Mb bp 分别为对应 index × 1,000,000。
4. u0：1。`z=(A+B)/2`，输出 `(z,z)`。
5. random-u：16。`u=(A-B)/2`，每条染色体内独立排列 u，seed 450500..450515，z 固定。

strict swap 不做 scale/project。u0/random-u 只有在最大半径 `>=1` 时统一全细胞缩放到
`1-1e-6`，并逐状态报告缩放；它们是辅助 null，不与严格 pair-preserving 对照混称。

每个状态一次 forward full-grid 遍历同时输出 shared-Zall G data、Nraw count、raw/weighted
bond/bend/repulsion/p_prior、total、Zcis/Zinter、per-inter-pair mixture rate、sorted four distances
与 inter rate/Zinter distribution。不得为组件重复遍历 full grid。

硬门：

- 20 whole-chromosome swaps 的 normalized total、per-record data 和各 penalty 相对 original 的
  absolute error 均 `<1e-10`。
- 28 whole/local swaps 的 pointwise unordered two-point multiset 守恒，坐标字节 multiset 守恒；
  inter per-pair mixture rates、sorted four distances、normalized inter distribution max absolute error
  `<1e-10`；Zinter 同时报 absolute/relative error，absolute error 必须 `<1e-10`。
- 任一硬门失败，停止后续正式流程并保留证据。

科学门：8 个 suffix splice 的 offdiag data delta 中至少 7 个 `>0`，且 median delta
`>=0.001 nat/offdiag contact`。通过只说明这些连接破坏被数据排斥，不证明 L2。

## 4. 实验 2：diag -> exposure -> geometry-gradient 路径

仍固定 baseline `x/p` 和 offdiag counts，零拟合。三个 diag 状态：

- original；
- all-chromosome cyclic shift：每条染色体 slice 内 `np.roll(diag, floor(n_bins/3))`，冻结 shift
  `[65,61,53,52,50,50,48,43,41,43,41,40,40,41,35,33,31,30,20,57]`；
- doubled：`2*diag_original`。

两个 e：locked original baseline e；production recomputed
`sqrt(offdiag_endpoint_counts + 2*diag_variant + 10)/mean`。两个 normalization denominator：
该 variant 的 `Nraw_variant` 与固定 `Noff`。共 `3*2*2=12` 次 value+gradient full-grid evaluation。

主梯度为 count-only physical-coordinate `dL/dx`，shape `(2,2645,3)`，不含 p、raw-y pullback
或任何 regularizer。主相对变化为 Nraw denominator 下 shift production vs original production 的
L2：`||g_shift-g_original||2/max(||g_original||2,1e-300)`；Linf 为辅助。doubled、Noff 下变化为
预注册 secondary。

硬门：locked-e/Nraw original vs shift 的 physical-x gradient max absolute difference `<1e-10`。
同一 raw objective 更换 denominator 时，value 和 gradient 必须按 `Nraw/Noff` 缩放，误差 `<1e-10`；
value 校验使用相同 diag constant，并另报 coordinate-dependent conditional 项。

production shift 主相对变化 `>10%` 表示优先机制，`<1%` 表示低优先级，中间为 uncertain；该解释
不取消已授权实验 3。不得更换 shift 或挑选格子。

## 5. 实验 3：train-only exposure 消融

### 5.1 固定 80/20 split

canonical endpoint 编码为 big-endian：`uint16 chromosome_header_index`、`uint64 position_bp`、
`uint8 strand_code`，strand code 冻结为 `+ -> 1`、`- -> 2`、其他 ASCII 单字节 -> `128+byte`；
两个 endpoint 按三元组 lexicographic 排序。canonical record bytes 为：

`b"P9016-contact-fold-v1\0" + uint64_be(560301) + endpoint_lo + endpoint_hi`。

hash 为 `BLAKE2b(digest_size=8)` 的 big-endian unsigned integer。阈值为
`floor(0.8 * 2^64) = 14757395258967642112`；`hash < threshold` 为 train，其余 test。不使用 Python
内置 hash，不使用旧 genomic parity，不强凑精确 80%。完全重复 canonical record 必须同 fold；
输出 group overlap=0、实际 train/test records 及 cis/inter/diag 分母。训练只读 train split；test
文件和评分由隔离评价入口读取。完整 header grid 始终保留，test 不决定 loci/support。

### 5.2 两个条件与初始化

- G-original：`e=sqrt(all_train_endpoints+10)/mean`。
- G-offdiag-e：`e=sqrt(offdiag_train_endpoints+10)/mean`。

除 e 外，train counts、Nraw、regularizers、solver、5->2->1 Mb schedule 全同。

5 Mb polymer 初始化复用 `pr/v1_calibration.py` 的纯几何 bounded-chain 规则，seeds 560101、560102，
`same_shape=False`、`p_init=0.75`。adapter 只允许 header/grid/n_loci/l0，不允许 counts、exposure、
test、phase 或 reference。必须以 counts 全零和置换 counts 证明同 seed 输出逐位相同；两个 e 条件同
seed 的 5 Mb x/raw-y/theta 逐位相同。禁止 baseline、014 或任何 contact-derived initialization。
后续 warm start 仅来自本条件上一 train stage endpoint；双方使用同 seed/stage 的同一扰动规则。

4 fits 单卡顺序。每 fit cap：5 Mb 612 FG、2 Mb 404 FG、1 Mb 486 FG，最多 1502 FG；总训练
最多 6008 FG，其中 1 Mb 最多 1944。某 stage canonical gradient `<=1e-6` 可 early converge，
然后按协议进入细层，不补花剩余预算。每个 endpoint 写全 finite、一致、不可覆盖的
`solver_state.npz`；presence mask 与 export 分离。

held-out primary 是真正 `q_ij=r_ij/Zall` cross-entropy：
`log(Zall) - sum_test_offdiag Ctest*log(r_ij)/Ntest_offdiag`。无 diag、factorial 或 test 拟合；e/x/p
完全 train-fixed；Zall 覆盖完整 eligible full grid，绝不能只覆盖 test observed pairs。另报加 K0_test
的旧口径，但 candidate-independent K0 不进入预测 gain。

### 5.3 辅助预算

辅助 full-grid traversals 上限 64，预分配 52：12 stage initial checks、4 terminal readback/train
score、4 test score、32 endpoint splice checks（4 endpoints × 实验1固定8 splices）。剩余最多12只用于
失败诊断，使用前必须记账。分别记录 objective FG calls 与 physical full-grid kernel passes；不得把
辅助 evaluate 藏入 6008。reference 距离 CPU 成本另账。

## 6. 状态、导出与 reference 评价

`solver_state.npz` 必须全 finite，且 `theta[:-1] == raw_y.ravel()`、sphere_forward(raw_y)==coordinates、
q->p 一致；使用 exclusive create，拒绝覆盖。`presence_mask.npz` 独立保存 bool `(2,n_loci)`。
`export.3dg` 只能由统一 helper 使用 `data.track_specs`、chromosome slice/offset、`locus_bin*bin_size`
导出 mask=True 行。导出前后 solver-state SHA 必须不变。CPU fixture 为两条染色体×两种分辨率，
覆盖不同 chromosome/copy drop、位置、readback、拒绝覆盖与 inconsistent state。

reference 评价只在所有待评价 state/endpoint/export 已写出 SHA 后启动。训练进程必须记录
`reference_opened=false, phase_opened=false`。

- 实验1/2 baseline whole-chromosome mapping 从 055 当前 base baseline 的20 chr orientation读取，
  核对 baseline SHA 与 mask；复算 baseline 必须复现 055。local splice 沿该 mapping，不逐窗口重选。
  whole-chromosome gauge control同步 transport mapping。secondary 可报告 best-global same/cross/contrast。
- 每条染色体报告 fixed-map same/cross、margin_A、margin_B、min_margin 完整表；sorted-four inter
  correlation 与严格不变量分开。
- 实验3每个原始 endpoint 在评价侧各定一次 whole-chromosome geometry mapping，其8个 splice 固定
  沿用；mapping 不反馈训练。

## 7. 实验3判据

- 两个 seed 的 held-out gain `NLL(original)-NLL(offdiag-e) >=0.01 nat/contact`。
- 每 seed 20-chr macro matched difference `offdiag-original >= -0.02`。
- min_margin primary 为20-chr macro mean；一致退化定义为两 seed delta 均 `<-1e-10`。
- 连接支持逐 endpoint 记录8 splices positive_count与median data delta；任一指标在两 seed均比original
  更差（tol `1e-10`）即 consistent degradation，不采纳。
- optimization incomparability：同seed一方1Mb converged而另一方未converged；或最终 canonical maxgrad
  ratio `>10`（denominator floor `1e-12`）且至少一方 `>1e-6`。出现时机制结论为 uncertain。
- test/reference 不选择 seed、checkpoint、超参或延长。全部细项保留，不只输出 success boolean。

## 8. 终态与报告

退出码0不等于科学成功。每 stage/fit 必须写 `converged`、`not_converged`、
`budget_not_converged`、failure/rejected/cancelled 的真实状态。输出最小 JSON/TSV、预算 ledger、artifact
hashes、中文内部 README 和必要英文图：实验1 data-vs-regularizer paired 图、实验2 gradient response、
实验3 paired held-out/structure 图；3 inch基础面板、300 DPI、7 pt，不制作重复热图。

本轮完成后先由父代理独立验收，暂不更新 GitHub。验收后才同步必要代码与结果到独立快照；旧历史
文件仍不修改。
