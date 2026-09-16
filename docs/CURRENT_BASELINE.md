# 当前 Baseline

- **Baseline ID**：`P9016-046-G-random-base-1Mb`
- **用户授权（2026-09-16）**：把不带第二轮 1Mb 优化的 046 G-random **基础端点**设为 baseline。
- **用途**：供后续工作比较，不改旧盲 source 选择记录，也不改任何旧运行或冻结协议。
- **候选 endpoint**：`real-G-random`（base_remaining），模型 `G` / `full-J`，`fixed_e`。
- **候选 3DG**：`test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.3dg`
- **3DG SHA256**：`ee5eb1545db9bfeeabcc24e704707f5f61a5793e6f245091347373442dc0032b`
- **候选 NPZ**：`test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.npz`
- **NPZ SHA256**：`6bb93bf570cf688fd8b02dccf14ca0867c1f831d5e123036ced626856f8d8752`
- **lineage**：014 无标签 random 起点（seed 2207）→ 5Mb 612FG → 2Mb 404FG → 1Mb 486FG，共 `612+404+486=1502` FG；终态 `budget_not_converged` (`fg_budget_exhausted`)。
- **历史记录（不改字节）**：046 的第二段 1Mb extension（`real-extension-G-full-J` / `real-extension-G-count-only`，各 486FG，ID `P9016-046-G-random-full-J-1Mb`）仍作为历史保留；它不再是当前 baseline。旧运行、冻结 mask、评价结果与冻结协议均未改动。
- **科学状态**：L2（沿整条染色体的一致拷贝身份恢复）尚未证明。
- **记录事实**：这是用户主动指定的简化 baseline（评价后的工作 baseline 采纳），不追溯宣称为盲选 G 模型；旧 source/fit/stop 记录没有使用 reference。
- **轻量 sanity（可选复现）**：在 046 冻结 legacy 21-mask 公共支持上，该 baseline 的 matched Pearson 在 20 条染色体上的均值为 `0.6023605799695414`（对每条染色体取最佳 whole-chr A/B 互换后两拷贝平均）。该数值由 `test_res/051-20260916T064500-g-random-baseline-ext/code/eval_pearson.py` 的 `same_legacy` / `same` 列复现。
- **绘图 CLI（旧记录）**：[`scripts/plot_3dg_comparison.py`](../scripts/plot_3dg_comparison.py) 仍可用于只读比较；051 轮的正式图由 `test_res/051-20260916T064500-g-random-baseline-ext/code/make_plots.py` 生成，使用同一 legacy mask 与统一 2x5 面板规范。

  ```bash
  python scripts/plot_3dg_comparison.py \
    test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.3dg \
    data/P9016.1m.3dg.gz \
    --mask test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/frozen_legacy_mask_snapshot.npz
  ```

- **旧 baseline 的正式图与验证**：[`test_res/047-20260915_135653-g-full-j-baseline-figures/`](../test_res/047-20260915_135653-g-full-j-baseline-figures/) 仍指向历史 extension 端点，不改字节、不改叙述时态。
