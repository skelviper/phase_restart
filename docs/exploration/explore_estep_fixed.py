"""修正后的 reference-structure accuracy（评审问题 4 与 same-bin trap）。

初始数字有两个独立问题：
  (a) swapped accuracy 被写成 1 - acc，导致所有 tie 都被静默计入；
  (b) 主要的“tie”来源是 same-bin contacts，此时距离按定义为 0。它们约占所有 intra contacts 的 39%，完全不携带结构信息。

本脚本先排除 same-bin contacts，再诚实处理剩余的 ties。
"""
import gzip, collections
import numpy as np

BIN = 1_000_000; OFF = 3_000_000
coords = {}
with gzip.open("data/P9016.1m.3dg.gz", "rt") as f:
    for line in f:
        a = line.split()
        coords.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
rows = collections.defaultdict(list)
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != c[3]: continue
        a, b = int(c[2]), int(c[4])
        if a > b: a, b = b, a
        ph = c[7] + c[8]
        if ph not in ("00", "11"): continue
        rows[c[1]].append((a, b, 0 if ph == "00" else 1))

print("%-6s %7s %7s %7s | %8s %11s %11s %11s" % (
    "chr", "n00", "n11", "same-bin", "tie rate", "acc(ties=.5)", "acc(no ties)", "old inflated"))
for chrom in ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chrX"]:
    mk, pk = chrom + "(mat)", chrom + "(pat)"
    if mk not in coords: continue
    rs = rows[chrom]
    same = sum(1 for a, b, _ in rs if (a - OFF)//BIN == (b - OFF)//BIN)
    rs = [(a, b, t) for a, b, t in rs if (a - OFF)//BIN != (b - OFF)//BIN]
    def d(key, a, b):
        m = coords[key]
        ba = (a - OFF)//BIN*BIN + OFF; bb = (b - OFF)//BIN*BIN + OFF
        return float(np.linalg.norm(m[ba] - m[bb])) if (ba in m and bb in m) else np.nan
    s = np.array([np.log(max(d(pk, a, b), 1e-6)) - np.log(max(d(mk, a, b), 1e-6)) for a, b, t in rs])
    t = np.array([x[2] for x in rs])
    ok = np.isfinite(s); s, t = s[ok], t[ok]
    predB = (s < 0)                       # label 0 -> (pat) 轨迹；方向沿用之前的测量
    tie = (s == 0)
    acc_half = (np.sum(predB & (t == 0)) + np.sum(~predB & (t == 1)) + 0.5*np.sum(tie)) / len(s)
    nt = ~tie
    acc_nt = (np.sum(predB[nt] & (t[nt] == 0)) + np.sum(~predB[nt] & (t[nt] == 1))) / nt.sum()
    old = 1.0 - float(np.mean((s > 0) == (t == 0)))    # 原始的 1 - acc 定义，在同一数据集上计算
    print("%-6s %7d %7d %7d | %8.4f %11.4f %11.4f %11.4f" % (
        chrom, int((t == 0).sum()), int((t == 1).sum()), same, tie.mean(), acc_half, acc_nt, old))
