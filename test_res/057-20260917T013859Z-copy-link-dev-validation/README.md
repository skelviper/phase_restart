# 057 copy 连接实验 A/B 最终记录

## 直接结论

- **A 是否通过：通过。** 两个 G-original seed 的 development-validation splice 均为 `8/8` 正 delta；median 分别为 `0.00904292036645` 与 `0.00929004905929` nat/offdiag contact。
- **B 是否运行：已运行固定候选生成、fixture、初始完整核对和首轮训练侧扫描。** 两个 seed 的全局最优候选均不满足门，因此均 no-op。
- **是否优于纯继续：未证明。** 冻结分支规则在两个首轮 no-op 后停止，未触发 control/swap paired continuation。
- **保留还是停止：停止 B，不采纳离散 swap；保留 056 G-original endpoints 与既有 046 baseline。**

## A：development validation 门

20% fold 已经看过，因此称 development validation，不称全新 test；它是 record-level split，不保证 molecule isolation。固定 train exposure/x/p，只替换 counts；Noff=253067，Z_full 覆盖 3,496,690 pairs；无 K0、无正则。A 恰好使用18次 contact full-grid forward，训练FG=0。

## B：首轮扫描

候选按真实 header/grid 与 chromosome length 生成，共 `1992` 个，低于2000，未采样。所有选择只使用训练目标；未读取 development 或 reference。
- seed 560101: cached argmin（rejected）`chr5 [130000000,151834684) bp`，cached delta_full_J=`0.0621613797192`；exact full-G delta_full_J=`0.0621613797192`，delta_count/Nraw=`0.00354786245118`，决定=`no-op`（`exact_full_G_checked`），实际 modified 集合为空。
- seed 560102: cached argmin（rejected）`chr2 [60000000,182113224) bp`，cached delta_full_J=`0.0161964674997`；exact full-G delta_full_J=`0.0161964674997`，delta_count/Nraw=`0.00457990329961`，决定=`no-op`（`exact_full_G_checked`），实际 modified 集合为空。

选择顺序是先在全部候选中取 delta_full_J 最小者，再检查 `delta_full_J<=-1e-8` 与 `delta_count/Nraw<=1e-10`；没有按 count 预筛，也没有改选次优。两个 best 的 delta 均为正。

## 预算与终态

- B 使用 `6` 个 logical full-grid calls / `12` 个 physical sweeps（cap 32），另计 `368032` 个 cis partial-kernel pairs；training FG=0。
- CPU 候选扫描墙钟：seed 560101 为 `0.149914 s`，seed 560102 为 `0.095205 s`。A 与 B 的 GPU event/host wall 分别见 `ledger.json` 与 `ledger_B.json`；这些计时口径分开记录，不作为等墙钟性能比较。
- 因双方首轮 no-op，未启动 paired L-BFGS、未生成四个新终态，也未做无意义的新 development/reference 评价；paired Control/Swap gain 与结构差值均为 `NA`，不是 0。
- 两个 seed 的 modified_chr 集合均为空，预冻结 modified-set margin 门为 `not_met`。
- A 的 `config.json`、`results.json`、`terminal.json` 保持不变。B 见 `config_B.json`、`fixture_results.json`、`round1_results.json`、`results_B.json`、`ledger_B.json`、`terminal_B.json` 与两个完整 scan TSV。
- 046/056、冻结源码、协议、mask、checkpoint 均未修改；未调用 051 fix_export；未 commit/push。

## 科学解释

A 支持固定的八个 copy 连接破坏在 development validation 上被排斥。B 的更广训练扫描没有找到可接受的首轮离散 swap，因此不支持进入 paired continuation。L2 仍未证明，不作 L3。两个 seed 是初始化重复，不是生物学重复。
