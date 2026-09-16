"""模拟：从无标签 contacts 中是否能辨识单倍型拆分？

真实值 = reference chr1 mat/pat structures。使用已知的拷贝标签从它们生成 contacts，然后隐藏标签。

  (1) sum test      ：预测留出的 TOTAL counts n_ij
  (2) label test    ：预测留出的 copy-specific counts n^A_ij（阳性对照）

如果 (1) 显示 oracle == random，而 (2) 显示 oracle >> random，则说明拆分存在于数据中，但对于只能看到总量的方法不可见。
"""
import gzip, os, subprocess
import numpy as np
from scipy.stats import spearmanr

HICKIT = "native/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"; CLEN = 195471971
WORK = "scratch/sim"; os.makedirs(WORK, exist_ok=True)

ref = {}
with gzip.open("data/P9016.1m.3dg.gz", "rt") as f:
    for line in f:
        a = line.split()
        ref.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
mk, pk = CHROM + "(mat)", CHROM + "(pat)"
pos = sorted(set(ref[mk]) & set(ref[pk]))
X = np.array([ref[mk][p] for p in pos]); Y = np.array([ref[pk][p] for p in pos])
n = len(pos)
print("bins:", n, "=> bin pairs:", n*(n-1)//2)
DM = np.linalg.norm(X[:,None]-X[None,:], axis=-1)
DP = np.linalg.norm(Y[:,None]-Y[None,:], axis=-1)

i, j = np.triu_indices(n, 1)
dA = np.minimum(DM[i, j], DP[i, j])          # 将按 copy 替换
dA = np.maximum(DM[i, j], 1e-3); dB = np.maximum(DP[i, j], 1e-3)
wA = dA ** -2.0; wB = dB ** -2.0
wA /= wA.sum(); wB /= wB.sum()

M = 60000
rng = np.random.default_rng(2024)
copy = rng.integers(0, 2, M)
pick = np.where(copy == 0, rng.choice(len(i), M, p=wA), rng.choice(len(i), M, p=wB))
ci, cj = i[pick], j[pick]
print("generated contacts:", M, "copy0:", int((copy == 0).sum()), "copy1:", int((copy == 1).sum()))

pairkey = ci * (n + 1) + cj
hsh = ((ci * 7919 + cj * 104729) % 2).astype(bool)   # True 表示 train
tr = hsh; te = ~hsh
print("train contacts %d / test contacts %d" % (tr.sum(), te.sum()))

PI = lambda p: pos[p]
def write(mask, assign, tag, split):
    names = (CHROM,) if not split else (CHROM + "a", CHROM + "b")
    path = WORK + "/%s.pairs.gz" % tag
    with gzip.open(path, "wt") as f:
        f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
        for nm in names: f.write("#chromosome: %s %d\n" % (nm, CLEN))
        f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
        for k in np.where(mask)[0]:
            nm = CHROM if not split else (CHROM + "a" if assign[k] == 0 else CHROM + "b")
            p, q = sorted((PI(ci[k]), PI(cj[k])))
            f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, p, nm, q))
    out = WORK + "/%s.3dg" % tag
    r = subprocess.run([HICKIT, "-i", path, "-P1", "-b1m", "-O", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    d = {}
    with open(out) as f:
        for line in f:
            if line[0] == "#": continue
            a = line.split()
            if len(a) < 5: continue
            d.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
    return d

# test fold 聚合
tkeys = {}
for k in np.where(te)[0]:
    tkeys.setdefault((ci[k], cj[k]), [0, 0])[copy[k]] += 1
TK = np.array(sorted(tkeys)); TC = np.array([tkeys[tuple(k)] for k in TK], float)
print("test-fold bin pairs:", len(TK))

def tvec(d, key):
    m = d[key]; out = np.full(len(TK), np.nan)
    for r in range(len(TK)):
        a, b = PI(int(TK[r, 0])), PI(int(TK[r, 1]))
        if a in m and b in m: out[r] = np.linalg.norm(m[a] - m[b])
    return out

print("fitting structures on the TRAIN fold ...")
dcons = tvec(write(tr, np.zeros(M, int), "cons", False), CHROM)
do = write(tr, copy, "oracle", True)
dOA, dOB = tvec(do, CHROM + "a"), tvec(do, CHROM + "b")
dr = write(tr, rng.integers(0, 2, M), "rand", True)
dRA, dRB = tvec(dr, CHROM + "a"), tvec(dr, CHROM + "b")

Mx = lambda x: np.maximum(x, 1e-3)
AL = [0.25, 0.5, 1.0, 2.0, 3.0]
ok = np.isfinite(dcons) & np.isfinite(dOA) & np.isfinite(dOB) & np.isfinite(dRA) & np.isfinite(dRB)
tot = TC.sum(1)
print("scored test bin pairs:", int(ok.sum()), "contacts:", int(tot[ok].sum()))
print()
print("(1) predict held-out TOTAL counts  n_ij = n^A + n^B")
for tag, a, b in (("oracle split", dOA, dOB), ("random split", dRA, dRB)):
    print("    %-14s two-struct rho = %+.4f" % (tag, spearmanr(Mx(a[ok])**-1.5 + Mx(b[ok])**-1.5, tot[ok])[0]))
print("    %-14s one-struct  rho = %+.4f" % ("consensus", spearmanr(Mx(dcons[ok])**-1.5, tot[ok])[0]))
print()
print("(2) POSITIVE CONTROL: predict held-out COPY-SPECIFIC counts n^A_ij / n^B_ij")
nA, nB = TC[:, 0], TC[:, 1]
have = ok & ((nA + nB) >= 1)
for tag, a, b in (("oracle split", dOA, dOB), ("random split", dRA, dRB)):
    rA = spearmanr(Mx(a[have])**-1.5, nA[have])[0]; rB = spearmanr(Mx(b[have])**-1.5, nB[have])[0]
    rA2 = spearmanr(Mx(b[have])**-1.5, nA[have])[0]; rB2 = spearmanr(Mx(a[have])**-1.5, nB[have])[0]
    print("    %-14s own-copy rho = (%+.4f, %+.4f)   swapped = (%+.4f, %+.4f)" % (tag, rA, rB, rA2, rB2))
