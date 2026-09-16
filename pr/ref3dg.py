"""reference 3DG 读取器：仅供评估器使用，并受 gate 保护。

`data/P9016.1m.3dg.gz` 是目标，绝不能作为任何拟合的输入。gate 必须已打开（即某个 stage 已写出并完成坐标哈希），本模块才会打开该文件。
"""
import gzip

import numpy as np

from .paths import BIN, OFF, REF3DG

STAGE = "eval"
# F3 / 修正后的 header block：phase0 -> chrN(pat)，phase1 -> chrN(mat)。
LABEL_TRACK = {0: "pat", 1: "mat"}


def load_reference(gate, path=REF3DG):
    gate.require(STAGE)
    coords = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            a = line.split()
            if len(a) < 5:
                continue
            coords.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
    return coords


def track(coords, chrom, suffix):
    return coords["%s(%s)" % (chrom, suffix)]


def distance(coords, key, b1, b2):
    """对 bin 索引数组计算距离；缺少珠子时返回 NaN。"""
    m = coords.get(key, {})
    out = np.full(len(b1), np.nan)
    for i, (x, y) in enumerate(zip(b1, b2)):
        va = m.get(int(OFF + x * BIN))
        vb = m.get(int(OFF + y * BIN))
        if va is None or vb is None:
            continue
        out[i] = float(np.linalg.norm(va - vb))
    return out


def reference_accuracy(coords, chrom, bins1, bins2, labels):
    """计算染色体内接触中真实拷贝更近的比例（排除同区间对）。

    phase 约定已冻结：phase0 -> pat，phase1 -> mat。反向方向只保留为命名的旧版诊断，不能用于选择报告的上限。平局按 1/2 计。
    """
    keep = (bins1 != bins2) & (labels >= 0)
    b1, b2, lb = bins1[keep], bins2[keep], labels[keep]
    d_pat = distance(coords, "%s(pat)" % chrom, b1, b2)
    d_mat = distance(coords, "%s(mat)" % chrom, b1, b2)
    ok = np.isfinite(d_pat) & np.isfinite(d_mat)
    d_pat, d_mat, lb = d_pat[ok], d_mat[ok], lb[ok]
    n = len(lb)
    tie = d_pat == d_mat
    res = {"n": int(n), "n_samebin_excluded": int((bins1 == bins2).sum()),
           "tie_rate": float(tie.mean()) if n else float("nan")}
    own = np.where(lb == 0, d_pat, d_mat)
    other = np.where(lb == 0, d_mat, d_pat)
    fixed = float(((own < other).astype(float) + 0.5 * tie).mean()) if n else float("nan")
    inverse = float(((other < own).astype(float) + 0.5 * tie).mean()) if n else float("nan")
    res.update({"acc": fixed, "orientation": 0,
                "acc_fixed_phase0_pat_phase1_mat": fixed,
                "acc_legacy_truth_reversed": inverse,
                "acc_legacy_truth_max": max(fixed, inverse) if n else float("nan")})
    return res
