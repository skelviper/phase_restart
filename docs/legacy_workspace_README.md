# phase_restart

从去掉单倍型列的接触数据重建 P9016 小鼠单细胞二倍体三维结构：20 条染色体、40 条拷贝轨迹，在一个共享核体积中联合拟合全部 `1,703,888` 条接触，并诚实评估真实单倍型拆分的可恢复程度。

## 阅读顺序

| 文件 | 内容 |
| --- | --- |
| `AGENTS.md` | 中文常驻规则、科学边界、报告底线和运行约束 |
| `docs/PROJECT_CONTEXT.md` | 历史阶段、关键数值、旧协议和适用边界的中文参考 |
| `docs/PLAN-genome-allele-split.md` | 当前全基因组 Reconstruction V1 计划 |
| `docs/MEASURED_FACTS.md` | 证据日志、数值和协议更正 |
| `docs/exploration/README.md` | 探索脚本的有效性与已替代状态 |
| `docs/STAGE1_PREREGISTRATION.md` | 已冻结的第 1 阶段评估工具设计 |
| `docs/audit/stage1-independent-audit/report.md` | 第 1 阶段独立审计 |
| `docs/zh_archive/README.md` | 冻结或哈希锁定英文原件的中文阅读映射 |
| `docs/zh_archive/RECONSTRUCTION_V1_PROTOCOL.zh-CN.md` | 哈希锁定 V1 protocol 的中文阅读版本 |

## 当前状态

- **第 0 阶段修正版基线：完成。** `docs/exploration/benchmark_v1.py` 是冻结回归基线，`pr/folds.py` 的折哈希已与标量实现逐字节一致。
- **第 1 阶段片段交换评估检验：完成。** chr1 的无标签判据在三类评估工具中都把染色体范围内一致拆分排在预注册全局错接方案之前；敏感度由分相深度、覆盖均匀性和留出对数量共同决定。这是评估检验，不是盲恢复。正式运行目录为 `test_res/007-…`、`008-…`、`009-…`。
- **Reconstruction V1 盲训练 020：完成记录。** `run.py reconstruct` 完成 `5 Mb -> 2 Mb -> 1 Mb` 全数据训练，5,290 个最终物理珠、40 条轨迹。`random_joint` 以最终 1 Mb `count_nll_normalized` `9.593585931292237` 选择，优于 `consensus_joint` 的 `9.598781292397067`；所有阶段到固定预算后为 `not_converged`，不等于数值收敛或 L2 证明。019 的 `iter50`/注册 `80` 步边界见 `docs/PROJECT_CONTEXT.md`。
- **022 V1 延续拟合与 FDG/R2：完成记录。** 接受 `240 -> 480` 的 240 步，终态 `budget_not_converged`；FDG 提案因 `rejected_no_label_free_improvement` 被拒。R2 20/20 条染色体的 `similarity` 差值为 `-0.000890817628`（CI `[-0.004060498973, 0.002427151962]`），`contrast` 差值为 `-0.000221703680`（CI `[-0.004748627202, 0.004092863791]`），均 9/20 胜出；不改变 L2 状态。详见 `test_res/022-20260913_111031-v1-continuation-fdg-r2/`；R2 报告正文为 `test_res/022-20260913_111031-v1-continuation-fdg-r2/exports/P9016-022-R2-comparison.md`。
- **S0 冻结坐标重评估：完成。** `python run.py reevaluate --source test_res/014-20260912_153000-s0-genome-wide-fixed` 只做经哈希验证的评估，不重新拟合；正式结果为 `test_res/017-20260912_200040-s0-reevaluated`，`016` 仅因 fig6 坐标轴标签被裁切而标为已替代，`015` 取消目录不应报告。V1 与旧版 S0 的留出协议分开。
- **046 P9016 真实细胞 shared-capture：8 个已验收拟合的唯一后置评价入口。** 全部 8 fits/16 stages/7952 outer FG 已冻结且为 `budget_not_converged`；正式评价只允许调用 `test_res/046-UTC-real-cell-shared-capture/evaluation_final/run_evaluation.py evaluate`，结果索引为 `test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/evaluation.json`。旧 045 `evaluation/real_only` 是动态废弃目录，不纳入后续候选。
- **当前 post-046 working baseline：** `P9016-046-G-random-full-J-1Mb`；G-random 按同模型 count NLL 选择，配套图和可复用双 3DG 制图入口见 `docs/CURRENT_BASELINE.md`、`docs/current_baseline.json` 与 `test_res/047-20260915_135653-g-full-j-baseline-figures/plots/`。reference 未用于 source selection、拟合或停止。

### 当前 baseline 制图

```bash
python scripts/plot_3dg_comparison.py \
  test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.3dg \
  data/P9016.1m.3dg.gz \
  --mask test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/frozen_legacy_mask_snapshot.npz
```

需要项目 Python 环境时先 `conda activate analysis`。默认自动比较 direct/swapped，并在此基础上对散点做 whole-genome 整体刚体对齐：逐 chromosome 按共同有限 position 的双拷贝内部距离 Pearson 定 copy 对应，再用全部对应点的 display 坐标拟合一次全局 Kabsch 旋转 R（`det=+1`、不镜像、不缩放、不逐 chr 旋转、不加平移），`--no-align` 可关闭以复现历史未旋转版本。普通调用只写出 `plots/*.png` 两张图与 `metrics.json`（参数、方向、匹配/交叉 Pearson、finite 计数、candidate SHA、`alignment` 块含 R 与逐 chr 方向/拟合点数）。默认输出目录是 `<repo>/scratch/plot_3dg_comparison`，相对 `--outdir` 也锚定 repo 根，不会在根目录堆产物。047 的散点图已按用户要求更新为对齐版，位于 `test_res/047-20260915_135653-g-full-j-baseline-figures/plots/whole_genome_1Mb_3d_scatter.png`，chr1 矩阵图仍是旧记录，对齐细节见该目录 `aligned_metrics.json`。

`docs/RECONSTRUCTION_V1_PROTOCOL.md` 虽在活动目录，却由 `pr/reconstruct.py:421-423` 和 `continuation.py:650-658` 按哈希锁定；机器命令仍使用原路径，中文阅读版本为 `docs/zh_archive/RECONSTRUCTION_V1_PROTOCOL.zh-CN.md`，由 `docs/zh_archive/README.md` 索引。

## 快速开始

```bash
source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh
conda activate analysis
cd native/hickit && make hickit && cd ../..
python run.py prepare          # 写出 inputs/P9016.snpfree.pairs.gz 和 inputs/manifest.json
```

`hickit -b` 会在解析时进行 FDG 拟合；`-P1` 必须先于输入加载，`-s`/`-n` 必须先于 `-b`，例如 `-P1 -s 7 -i pairs.gz -n 3000 -b1m`。显式原生随机种子只是初始化重复；独立对照必须更换数据，例如使用不相交的记录半集。

## 目录

```text
data/            输入，不修改
native/hickit/   随仓库提供的原生 FDG 引擎与 PROVENANCE.md
inputs/          派生的去单倍型列接触文件与 sha256 清单
pr/, run.py      驱动器：prepare、stage1、split、reevaluate、reconstruct
docs/            计划、证据、预注册、审计和中文上下文
docs/zh_archive/ 冻结或哈希锁定英文文档的去重中文阅读映射（索引文件为 README.md）
test_res/        每个正式实验一个目录，保留历史状态与终态
test_res/*/exports/  该实验的交付导出层（原先放在根 deliverables/ 的文件已按归属迁入）
scratch/         冒烟检查夹具；明确标记的 READ_LATER 笔记保留
```

不设根 `deliverables/` 和根编号实验目录；导出件放在对应 `test_res/{NNN}-…/exports/`，临时产物放在 `scratch/`。

## 020 评估交付

最终评估报告是 `test_res/020-20260913_071841-v1-p9016-joint/evaluation-20260913_080300-final/README.md`，交付入口与打包件在该 run 的 `exports/`（`START_HERE.md`、`P9016-020-final-delivery-MANIFEST.md`、`P9016-v1-020.zip`）。报告在同一细胞 20 条染色体的共同 `200,898` 条记录上汇总：

- R1：`selected` `53.95%`，`fixed-random` `50.72%`，`oracle-fit` `63.21%`，`reference` `78.72%`。
- 配对 R1：`+0.032277`，CI `[+0.009545, +0.054825]`，胜出数 `14/20`。
- R2：`delta=+0.111530`，CI `[+0.068857, +0.157703]`，胜出数 `18/20`。
- R3：`delta=+0.074663`，CI `[-0.010580, +0.161255]`，`132/138` 个片段，平均 `1.90` 道分界墙。

这些区间描述同一细胞内的技术/结构变异，不是生物学重复。六个 020 训练阶段都以固定预算 `not_converged`；评估报告提供读出，但 L2 仍未证明。020 是 Reconstruction V1，与旧版 S0 及旧留出协议分开；评估器只做报告整理，不应预期新拟合或指标改变。

完整规则、历史数值、阶段边界和冻结文件处理见 `AGENTS.md` 与 `docs/PROJECT_CONTEXT.md`。按用途交付：面向外部的正式交付物、报告、图注、摘要和交付 README 使用英文，内部阅读材料、代码注释和 docstring 使用中文；机器接口/协议字符串和命令字符串保持原样，面向人的展示文案按用途选择语言。必要机器可读结果照常保留。文字或表格足够时可以不出图，一张足够时不做第二张，HTML 仅在用户明确要求时生成。