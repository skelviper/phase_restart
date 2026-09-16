"""phase_restart：P9016 的 SNP-free 二倍体重建。

包布局
    paths.py    常量（bin 大小、bin 偏移、染色体顺序）
    gate.py     保护每次 label 读取的坐标哈希 gate
    pairs7.py   SNP-free 读取器/写入器（严格七列，硬性拒绝 phase）
    labels.py   唯一打开 phase 列的模块；受 gate 保护
    folds.py    对数值 bin 对进行混合哈希 train/val/test 划分
    fdg.py      随仓库提供的 native hickit FDG engine 封装
    splits.py   候选 contact 划分（oracle、fragment swap、random、consensus）
    score.py    留出 Spearman、分层读出、配对 bootstrap CI
    figs.py     图形辅助函数（3 英寸 panel、300 DPI、7 pt）
"""
