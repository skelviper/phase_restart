"""决定性的可辨识性检验，不发生信息泄漏。

结构只使用 train bin-pair contacts 拟合。alpha 也在 train pairs 上拟合。所有结果都在留出的 bin pairs 上评分。如果由 TRUE labels 构建的双结构模型，在预测留出 counts 时不优于由 RANDOM split 构建的模型，则单倍型拆分没有超出 lambda_A + lambda_B 总和的信息。
"""
import gzip, os, subprocess
import numpy as np
from scipy.stats import spearmanr

HICKIT = "native/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"; CLEN = 195471971
WORK = "scratch/id_test"; os.makedirs(WORK, exist_ok=True)

p1, p2, lab = [], [], []
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != CHROM or c[3] != CHROM or int(c[2]) > int(c[4]): continue
        p1.append(int(c[2])); p2.append(int(c[4]))
        ph = c[7] + c[8]
        lab.append(0 if ph == "00" else (1 if ph == "11" else -1))
p1 = np.array(p1); p2 = np.array(p2); lab = np.array(lab); N = len(p1)
I1 = (p1 - OFF)//BIN; I2 = (p2 - OFF)//BIN

# 在 bin pairs 上进行确定性的 train/test 拆分
hsh = ((I1 * 7919 + I2 * 104729) % 2).astype(bool)   # True 表示 train
n_tr = int(hsh.sum()); n_te = N - n_tr
print("contacts: total %d  train %d  test %d" % (N, n_tr, n_te))
pc = {}
for k in range(N):
    pc[(I1[k], I2[k])] = pc.get((I1[k], I2[k]), 0) + 1
K = np.array(sorted(pc)); C = np.array([pc[tuple(k)] for k in K], float)
hK = ((K[:, 0] * 7919 + K[:, 1] * 104729) % 2).astype(bool)
print("bin pairs: total %d  train %d  test %d" % (len(K), int(hK.sum()), int((~hK).sum())))

def write(rows, tag, split):
    path = WORK + "/%s.pairs.gz" % tag
    names = (CHROM,) if not split else (CHROM + "a", CHROM + "b")
    with gzip.open(path, "wt") as f:
        f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
        for nm in names:
            f.write("#chromosome: %s %d\n" % (nm, CLEN))
        f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
        for k, asg in rows:
            nm = CHROM if not split else (CHROM + "a" if asg == 0 else CHROM + "b")
            f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, p1[k], nm, p2[k]))
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

def dv(d, key):
    m = d[key]; out = np.full(len(K), np.nan)
    for r in range(len(K)):
        a, b = int(K[r, 0])*BIN + OFF, int(K[r, 1])*BIN + OFF
        if a in m and b in m: out[r] = np.linalg.norm(m[a] - m[b])
    return out

tr_idx = [k for k in range(N) if hsh[k]]
print("building structures from TRAIN contacts only ...")
d_cons = dv(write([(k, 0) for k in tr_idx], "cons", False), CHROM)
do = write([(k, lab[k] if lab[k] >= 0 else (k % 2)) for k in tr_idx], "oracle", True)
d_oa, d_ob = dv(do, CHROM + "a"), dv(do, CHROM + "b")

ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
M = lambda x: np.maximum(x, 1e-3)

def score(pred, mask):
    a, b = pred[mask], C[mask]
    return spearmanr(a, b)[0] if np.std(a) > 0 else float("nan")

ok0 = np.isfinite(d_cons) & np.isfinite(d_oa) & np.isfinite(d_ob)
print("common support pairs: %d (train %d / test %d)" % (ok0.sum(), (ok0 & hK).sum(), (ok0 & ~hK).sum()))

def evaluate(dA, dB, tag):
    tr = ok0 & hK; te = ok0 & ~hK; best = (-9, None, None)
    for a in ALPHAS:
        rtr = score(M(dA)**-a, tr)
        if rtr > best[0]: best = (rtr, a, score(M(dA)**-a + M(dB)**-a, te))
    _, a_2, rte_2 = best
    # 在同一 support 上的 one-structure reference
    best1 = (-9, None, None)
    for a in ALPHAS:
        rtr = score(M(d_cons)**-a, tr)
        if rtr > best1[0]: best1 = (rtr, a, score(M(d_cons)**-a, te))
    print("  %-26s  two-struct alpha*=%.2f  test_rho=%.4f   | one-struct test_rho=%.4f" %
          (tag, a_2, rte_2, best1[2]))
    return rte_2, best1[2]

print()
o2, o1 = evaluate(d_oa, d_ob, "oracle split (train-fit)")
print()
rs = []
for seed in range(5):
    rng = np.random.default_rng(1000 + seed)
    dr = write([(k, int(rng.integers(0, 2))) for k in tr_idx], "rand%d" % seed, True)
    r2, _ = evaluate(dv(dr, CHROM + "a"), dv(dr, CHROM + "b"), "random split seed %d" % seed)
    rs.append(r2)
rs = np.array(rs)
print()
print("=== SUMMARY (held-out bin-pair Spearman) ===")
print("  one structure (consensus)        : %.4f" % o1)
print("  two structures, oracle split     : %.4f" % o2)
print("  two structures, random split     : %.4f +- %.4f  (n=5)" % (rs.mean(), rs.std()))
print("  oracle - random                  : %+.4f" % (o2 - rs.mean()))
print("  random - one structure           : %+.4f" % (rs.mean() - o1))
