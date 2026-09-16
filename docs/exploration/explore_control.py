import gzip
import numpy as np

BIN = 1_000_000; OFF = 3_000_000
rows = []
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != "chr1" or c[3] != "chr1": continue
        rows.append(((int(c[2])-OFF)//BIN, (int(c[4])-OFF)//BIN, c[7]+c[8]))
n = 193
iu = np.triu_indices(n, 1)

def build(sel):
    M = np.zeros((n, n))
    for i, j, p in rows:
        if sel(i, j, p):
            M[i, j] += 1; M[j, i] += 1
    return M

def ddn(M):
    A = np.zeros_like(M)
    for d in range(n):
        idx = np.arange(n - d); v = M[idx, idx + d]; mu = v.mean()
        if mu > 0: A[idx, idx + d] = v / mu
    return A + A.T

M00 = build(lambda i, j, p: p == "00")
M11 = build(lambda i, j, p: p == "11")
print("counts 00=%d 11=%d" % (M00.sum()/2, M11.sum()/2))

rng = np.random.default_rng(1)
n00, n11 = int(M00.sum()/2), int(M11.sum()/2)
reps = 5
cs = []
for r in range(reps):
    A = np.zeros((n, n)); B = np.zeros((n, n))
    perm = rng.permutation(len(rows))
    for k, idx in enumerate(perm):
        i, j, p = rows[idx]
        if k < n00: A[i, j] += 1; A[j, i] += 1
        elif k < n00 + n11: B[i, j] += 1; B[j, i] += 1
    cs.append(float(np.corrcoef(ddn(A)[iu], ddn(B)[iu])[0, 1]))
print("MATCHED-COUNT random split control corr: %.4f (sd %.4f)" % (np.mean(cs), np.std(cs)))
print("REAL 00 vs 11 corr: %.4f" % float(np.corrcoef(ddn(M00)[iu], ddn(M11)[iu])[0, 1]))

# 3dg
coords = {}
with gzip.open("data/P9016.1m.3dg.gz", "rt") as f:
    for line in f:
        a = line.split()
        coords.setdefault(a[0], {}).setdefault(int(a[1]), (float(a[2]), float(a[3]), float(a[4])))
mk = "chr1(mat)"; pk = "chr1(pat)"
pos = sorted(set(coords[mk]) & set(coords[pk]))
print("common chr1 bins in 3dg:", len(pos))
X = np.array([coords[mk][p] for p in pos]); Y = np.array([coords[pk][p] for p in pos])
m = len(pos); iu2 = np.triu_indices(m, 1)
D1 = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
D2 = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=-1)
print("corr(mat dist, pat dist) 3dg:", round(float(np.corrcoef(D1[iu2], D2[iu2])[0, 1]), 4))
print("mat Rg %.3f pat Rg %.3f" % (np.sqrt(((X - X.mean(0))**2).sum(1).mean()), np.sqrt(((Y - Y.mean(0))**2).sum(1).mean())))
print("centroid distance:", round(float(np.linalg.norm(X.mean(0) - Y.mean(0))), 3))
# 所有染色体拷贝之间的 centroid distance
import itertools
keys = sorted(coords)
cens = {}
for k in keys:
    v = np.array(list(coords[k].values()))
    cens[k] = v.mean(0)
print("\nwithin-chrom mat-pat centroid distances:")
for c in ["chr1","chr2","chr5","chr10","chrX"]:
    print(" ", c, round(float(np.linalg.norm(cens[c+"(mat)"] - cens[c+"(pat)"])), 3))
# overlap：mat bins 中最近的非自身结构是 pat 的比例
