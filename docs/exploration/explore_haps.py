import gzip, collections
import numpy as np

BIN = 1_000_000
OFF = 3_000_000
path = "data/P9016.pairs.gz"

mats = {}   # (chrom, phaskey) -> dict (i,j)->count
nbins = {"chr1": 193}
for k in ("00", "11", "01", "10", "u"):
    mats[k] = collections.Counter()

with gzip.open(path, "rt") as f:
    for line in f:
        if line[0] == "#":
            continue
        c = line.rstrip("\n").split("\t")
        if c[1] != "chr1" or c[3] != "chr1":
            continue
        b1, b2 = (int(c[2]) - OFF) // BIN, (int(c[4]) - OFF) // BIN
        p0, p1 = c[7], c[8]
        if p0 == "0" and p1 == "0": k = "00"
        elif p0 == "1" and p1 == "1": k = "11"
        elif p0 == "0" and p1 == "1": k = "01"
        elif p0 == "1" and p1 == "0": k = "10"
        else: k = "u"
        mats[k][(b1, b2)] += 1

n = 193
def tomat(cnt):
    M = np.zeros((n, n))
    for (i, j), v in cnt.items():
        M[i, j] += v; M[j, i] += v
    return M

M00, M11, M01, M10, Mu = (tomat(mats[k]) for k in ("00", "11", "01", "10", "u"))
tot = M00 + M11 + M01 + M10
print("chr1 totals: 00=%d 11=%d 01=%d 10=%d unphased=%d" % (M00.sum()/2, M11.sum()/2, M01.sum()/2, M10.sum()/2, Mu.sum()/2))
iu = np.triu_indices(n, 1)
a, b = M00[iu], M11[iu]
print("corr(M00, M11) upper-tri (all):", round(float(np.corrcoef(a, b)[0,1]), 4))
# 只使用两个 counts 都有信息的 pairs
def dscale(M, dmin=1):
    out = np.zeros((n, n))
    for d in range(1, n):
        idx = np.arange(n - d)
        v = M[idx, idx + d]
        out[idx, idx + d] = v
    return out
# 更好的做法：在距离衰减归一化后比较
def dd_norm(M):
    A = np.zeros_like(M)
    for d in range(n):
        idx = np.arange(n - d)
        v = M[idx, idx + d]
        mu = v.mean()
        if mu > 0:
            A[idx, idx + d] = v / mu
    A = A + A.T
    np.fill_diagonal(A, 0)
    return A
A00, A11 = dd_norm(M00), dd_norm(M11)
print("corr(ddnorm M00, ddnorm M11):", round(float(np.corrcoef(A00[iu], A11[iu])[0,1]), 4))
mask = (M00[iu] + M11[iu]) >= 2
print("n pairs with >=2 phased:", int(mask.sum()), "of", len(a))
print("corr on those:", round(float(np.corrcoef(a[mask], b[mask])[0,1]), 4))
print("sum 00 in mask:", int(a[mask].sum()), "sum 11:", int(b[mask].sum()))
# log ratio 分布
lr = np.log2((a[mask] + 0.5) / (b[mask] + 0.5))
print("log2 ratio 00/11: std", round(float(lr.std()), 3), "mean", round(float(lr.mean()), 3))
