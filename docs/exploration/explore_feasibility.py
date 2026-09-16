import gzip, collections
import numpy as np

BIN = 1_000_000; OFF = 3_000_000
path = "data/P9016.pairs.gz"
chr1_rows = []
with gzip.open(path, "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != "chr1" or c[3] != "chr1": continue
        chr1_rows.append(((int(c[2])-OFF)//BIN, (int(c[4])-OFF)//BIN, c[7]+c[8]))
n = 193
print("chr1 intra rows:", len(chr1_rows))

def build(rows):
    M = np.zeros((n, n))
    for i, j, p in rows:
        if p == "00": M[i,j]+=1; M[j,i]+=1
        elif p == "11": pass
    return M

def build_all(rows, sel):
    M = np.zeros((n, n))
    for i, j, p in rows:
        if sel(p):
            M[i,j]+=1; M[j,i]+=1
    return M

def dd_norm(M):
    A = np.zeros_like(M)
    for d in range(n):
        idx = np.arange(n - d)
        v = M[idx, idx+d]
        mu = v.mean()
        if mu > 0: A[idx, idx+d] = v/mu
    return A + A.T

M00 = build_all(chr1_rows, lambda p: p == "00")
M11 = build_all(chr1_rows, lambda p: p == "11")
Mall = build_all(chr1_rows, lambda p: True)
iu = np.triu_indices(n, 1)

# --- 与 M00+M11 总接触数相同的随机拆分对照
import random
rng = np.random.default_rng(0)
phased = [r for r in chr1_rows if r[2] in ("00","11")]
rand = [r for r in chr1_rows]
A = np.zeros((n,n)); B = np.zeros((n,n))
for i,j,p in rand:
    if rng.random() < 0.5: A[i,j]+=1; A[j,i]+=1
    else: B[i,j]+=1; B[j,i]+=1
An, Bn = dd_norm(A), dd_norm(B)
print("CONTROL random split of ALL chr1 contacts (equal sizes): corr =", round(float(np.corrcoef(An[iu], Bn[iu])[0,1]), 4))
An2, Bn2 = dd_norm(M00), dd_norm(M11)
print("REAL 00 vs 11: corr =", round(float(np.corrcoef(An2[iu], Bn2[iu])[0,1]), 4))

# 5Mb 分箱
B5 = 5
n5 = n // B5
def rebin(M, k):
    m = n // k
    out = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            out[i,j] = M[i*k:(i+1)*k, j*k:(j+1)*k].sum()
    return out
A5, B5m = rebin(M00, B5), rebin(M11, B5)
def dd_norm_k(M, m):
    A = np.zeros_like(M)
    for d in range(m):
        idx = np.arange(m-d); v = M[idx, idx+d]; mu = v.mean()
        if mu > 0: A[idx, idx+d] = v/mu
    return A + A.T
iu5 = np.triu_indices(n5, 1)
print("5Mb 00 vs 11 corr:", round(float(np.corrcoef(dd_norm_k(A5,n5)[iu5], dd_norm_k(B5m,n5)[iu5])[0,1]), 4))
R5a = rebin(A, B5); R5b = rebin(B, B5)
print("5Mb random-split control corr:", round(float(np.corrcoef(dd_norm_k(R5a,n5)[iu5], dd_norm_k(R5b,n5)[iu5])[0,1]), 4))

# --- 3dg 结构
coords = {}
with gzip.open("data/P9016.1m.3dg.gz", "rt") as f:
    for line in f:
        a = line.split()
        coords.setdefault(a[0], []).append((int(a[1]), float(a[2]), float(a[3]), float(a[4])))
for k in ("chr1(mat)", "chr1(pat)"):
    print(k, "nbins", len(coords[k]), "first pos", coords[k][0][0], "last", coords[k][-1][0])
def xyz(k):
    v = sorted(coords[k])
    return np.array([x[1:] for x in v])
X = xyz("chr1(mat)"); Y = xyz("chr1(pat)")
print("mat shape", X.shape, "pat shape", Y.shape)
D1 = np.linalg.norm(X[:,None,:]-X[None,:,:], axis=-1)
D2 = np.linalg.norm(Y[:,None,:]-Y[None,:,:], axis=-1)
print("corr(mat dist, pat dist):", round(float(np.corrcoef(D1[iu], D2[iu])[0,1]), 4))
print("mat Rg", round(float(np.sqrt(((X-X.mean(0))**2).sum(1).mean())), 3), "pat Rg", round(float(np.sqrt(((Y-Y.mean(0))**2).sum(1).mean())), 3))
# 混合 contact 对距离的预测效果
Mu = Mall.copy(); np.fill_diagonal(Mu, 0)
d1 = D1[iu]; d2 = D2[iu]; mm = Mu[iu]; m00 = M00[iu]; m11 = M11[iu]
print("corr(log mixed, log dist_mat):", round(float(np.corrcoef(np.log(mm+0.5), np.log(d1+1))[0,1]), 4))
print("corr(log 00, log dist_mat):", round(float(np.corrcoef(np.log(m00+0.5), np.log(d1+1))[0,1]), 4))
print("corr(log 11, log dist_pat):", round(float(np.corrcoef(np.log(m11+0.5), np.log(d2+1))[0,1]), 4))
print("cross: corr(log 00, log dist_pat):", round(float(np.corrcoef(np.log(m00+0.5), np.log(d2+1))[0,1]), 4))
print("cross: corr(log 11, log dist_mat):", round(float(np.corrcoef(np.log(m11+0.5), np.log(d1+1))[0,1]), 4))
