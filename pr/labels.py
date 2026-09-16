"""读取 phase 列的唯一模块。

本模块受 `pr.gate.EvalGate` 保护：调用者必须提供一个已经为待评价坐标所属 stage 打开的 gate。训练路径的任何部分都不导入本模块。
"""
import gzip

import numpy as np

from .paths import PAIRS
from .pairs7 import canonical_cis

STAGE = "eval"


def load_phase_labels(gate, chrom, pairs_path=PAIRS):
    """返回指定染色体的 cis 记录的 (pos1, pos2, label)。

    label：phase 为 '00' 时取 0，phase 为 '11' 时取 1；非完整的 '00'/'11'（包括两端异拷贝或分相未知）时取 -1。
    `pos1 <= pos2` 按数值保证。只有 phase 完整的接触记录才可作为 oracle（AGENTS.md 报告规则 2）。
    """
    gate.require(STAGE)
    p1, p2, lb = [], [], []
    with gzip.open(pairs_path, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.rstrip("\n").split("\t")
            if c[1] != c[3] or c[1] != chrom:
                continue
            a, b, swapped = canonical_cis(c[2], c[4])
            phase0, phase1 = c[7], c[8]
            if swapped:
                phase0, phase1 = phase1, phase0
            ph = phase0 + phase1
            p1.append(a)
            p2.append(b)
            lb.append(0 if ph == "00" else (1 if ph == "11" else -1))
    return (np.asarray(p1, dtype=np.int64), np.asarray(p2, dtype=np.int64),
            np.asarray(lb, dtype=np.int8))


def label_counts(lb):
    return {"n00": int((lb == 0).sum()), "n11": int((lb == 1).sum()),
            "n_unphased": int((lb < 0).sum())}
