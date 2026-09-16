"""步骤 1（v3，已验证符号）：在结构层面测量吸引盆半径。

contrast = mean(rho(dA,dA*), rho(dB,dB*)) - mean(rho(dA,dB*), rho(dB,dA*))
  0        -> collapsed，结构中没有单倍型信息
  0.1993   -> exact truth（在本数据上测得）

E-step：sc = log(dB) - log(dA)。sc > 0 表示更接近 copy A。
  hard：若 sc > 0，则分配到 A
  soft：P(A) = 1 / (1 + exp(-alpha * sc))       [= dA^-a / (dA^-a + dB^-a)]
"""
import gzip, os, subprocess, sys
import numpy as np
from scipy.stats import spearmanr

HICKIT = "native/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"; NA, NB = "c1a", "c1b"
WORK = "scratch/basin5"; os.makedirs(WORK, exist_ok=True)
Mx = lambda x: np.maximum(x, 1e-3)

p1, p2, lb = [], [], []
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != c[3] or c[1] != CHROM or int(c[2]) > int(c[4]): continue
        ph = c[7] + c[8]
        if ph not in ("00", "11"): continue
        p1.append(int(c[2])); p2.append(int(c[4])); lb.append(0 if ph == "00" else 1)
p1 = np.array(p1); p2 = np.array(p2); lb = np.array(lb); N = len(p1)
I1 = (p1 - OFF)//BIN; I2 = (p2 - OFF)//BIN
P1, P2 = I1*BIN + OFF, I2*BIN + OFF
hsh = ((I1 * 7919 + I2 * 104729) % 2).astype(bool)
trk = np.where(hsh)[0]; tek = np.where(~hsh)[0]

def run(tag, rows, split=True):
    path = WORK + "/%s.pairs.gz" % tag
    with gzip.open(path, "wt") as f:
        f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
        for nm in ((CHROM,) if not split else (NA, NB)):
            f.write("#chromosome: %s %d\n" % (nm, 300000000))
        f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
        for p, q, a in rows:
            nm = CHROM if not split else (NA if a == 0 else NB)
            f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, p, nm, q))
    out = WORK + "/%s.3dg" % tag
    r = subprocess.run([HICKIT, "-i", path, "-P1", "-b1m", "-O", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    d = {}
    with open(out) as f:
        for line in f:
            if line[0] == "#": continue
            a = line.split()
            if len(a) < 5: continue
            d.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
    return d

def dv(d, key):
    m = d.get(key, {}); out = np.full(N, np.nan)
    for k in range(N):
        if P1[k] in m and P2[k] in m:
            out[k] = np.linalg.norm(m[P1[k]] - m[P2[k]])
    return out

truth_rows = [(int(p1[i]), int(p2[i]), int(lb[i])) for i in trk]
dA_star = dv(run("truth", truth_rows), NA); dB_star = dv(run("truth2", truth_rows), NB)
base = np.isfinite(dA_star) & np.isfinite(dB_star)
sp = lambda a, b, m: spearmanr(a[m], b[m])[0]

def contrast(dA, dB):
    m = base & np.isfinite(dA) & np.isfinite(dB)
    if m.sum() < 500: return float("nan"), int(m.sum())
    match = (sp(dA, dA_star, m) + sp(dB, dB_star, m)) / 2
    cross = (sp(dA, dB_star, m) + sp(dB, dA_star, m)) / 2
    return match - cross, int(m.sum())

c_truth, n_use = contrast(dA_star, dB_star)
dC = dv(run("cons", [(int(p1[i]), int(p2[i]), 0) for i in trk], False), CHROM)
c_cons, _ = contrast(dC, dC)
print("usable bin pairs: %d" % n_use)
print("SANITY  contrast(truth) = %+.4f   contrast(consensus) = %+.4f" % (c_truth, c_cons))

# 符号检查：使用真实结构时，hard assignment 必须优于随机机会
sc0 = np.log(Mx(dB_star)) - np.log(Mx(dA_star))
hard_acc = float(np.mean((((sc0 <= 0).astype(int)) == lb)[base]))
print("SANITY  hard assignment from the TRUE structures: test accuracy = %.4f  (chance 0.5)" % hard_acc)
assert hard_acc > 0.55, "E-step sign looks inverted"

FS = [0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48]; ROUNDS = 6; ALPHA = 1.0
SEEDS = [11, 12]
for mode in ["hard"]:
    print()
    print("=== E-step = %s (alpha=%.1f) ===" % (mode, ALPHA))
    print("%5s | %s" % ("f", "  ".join("k%-7d" % k for k in range(ROUNDS + 1))))
    for f in FS:
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            asg = np.where(rng.random(N) < f, 1 - lb, lb).astype(int)
            vals = []
            for k in range(ROUNDS + 1):
                d = run("m", [(int(p1[i]), int(p2[i]), int(asg[i])) for i in trk])
                dA, dB = dv(d, NA), dv(d, NB)
                c, _ = contrast(dA, dB); vals.append(c)
                if k == ROUNDS: break
                sc = np.log(Mx(dB)) - np.log(Mx(dA))
                sc = np.where(np.isfinite(sc), sc, 0.0)
                asg = (sc <= 0).astype(int)
            print("%5.2f s%d | %s" % (f, sd, "  ".join("%+.4f" % v for v in vals)))
            sys.stdout.flush()
print()
print("reference: truth = %+.4f, consensus = %+.4f" % (c_truth, c_cons))