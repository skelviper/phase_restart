"""步骤 1（修正版）：在结构层面测量吸引盆半径。

初版使用“分配到真实拷贝的接触比例”作为指标。该指标有上限：E-step 对每个 bin pair 只做一次决策，因此即使结构完美，可达到的 agreement 也只有 E[max(pi, 1-pi)]，约为 0.6-0.7。
因此，某一轮从 1.00 降到 0.61 可能意味着“达到上限”，而不是“发生塌缩”。

正确指标是：拟合距离场与真实距离场之间的 matched-minus-swapped 对比度。

  match    = mean( rho(dA, dA*), rho(dB, dB*) )
  cross    = mean( rho(dA, dB*), rho(dB, dA*) )
  contrast = match - cross        (0 = collapsed, ~0.66 = exact truth)

同时比较 HARD E-step 与 STOCHASTIC（soft）E-step。
"""
import gzip, os, subprocess, sys
import numpy as np
from scipy.stats import spearmanr

HICKIT = "native/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"
NA, NB = "c1a", "c1b"
WORK = "scratch/basin2"; os.makedirs(WORK, exist_ok=True)
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
p1 = np.array(p1); p2 = np.array(p2); lb = np.array(lb)
I1 = (p1 - OFF)//BIN; I2 = (p2 - OFF)//BIN
P1, P2 = I1*BIN + OFF, I2*BIN + OFF
hsh = ((I1 * 7919 + I2 * 104729) % 2).astype(bool)
trk = np.where(hsh)[0]; tek = np.where(~hsh)[0]
print("phased cis contacts %d (train %d / test %d)" % (len(p1), len(trk), len(tek)))

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
    m = d.get(key, {}); out = np.full(len(p1), np.nan)
    for k in range(len(p1)):
        if P1[k] in m and P2[k] in m:
            out[k] = np.linalg.norm(m[P1[k]] - m[P2[k]])
    return out

# 真实结构（train fold、true labels）和 consensus
dT = run("truth", [(int(p1[i]), int(p2[i]), int(lb[i])) for i in trk])
dA_star, dB_star = dv(dT, NA), dv(dT, NB)
dC = dv(run("cons", [(int(p1[i]), int(p2[i]), 0) for i in trk], False), CHROM)
mo = np.isfinite(dA_star) & np.isfinite(dB_star) & np.isfinite(dC)
print("pairs usable for the structure metric: %d" % mo.sum())
sp = lambda a, b: spearmanr(a[mo], b[mo])[0]
print("sanity: contrast(dA*,dB*) pairs = %.4f ; consensus contrast = %.4f" % (
    (sp(dA_star, dA_star) + sp(dB_star, dB_star) - sp(dA_star, dB_star) - sp(dB_star, dA_star)) / 2,
    (sp(dC, dA_star) + sp(dC, dB_star) - sp(dC, dB_star) - sp(dC, dA_star)) / 2))

def contrast(d):
    dA, dB = dv(d, NA), dv(d, NB)
    if not (np.isfinite(dA[mo]).all() and np.isfinite(dB[mo]).all()): return float("nan")
    match = (sp(dA, dA_star) + sp(dB, dB_star)) / 2
    cross = (sp(dA, dB_star) + sp(dB, dA_star)) / 2
    return match - cross

FS = [0.0, 0.2, 0.3, 0.4, 0.5]
ROUNDS = 8
print()
print("contrast: 0 = collapsed (no haplotype info), %.3f = exact truth" % 0.66)
for mode in ["hard", "soft"]:
    print()
    print("=== E-step = %s ===" % mode)
    print("%5s | %s" % ("f", "  ".join("k%-6d" % k for k in range(ROUNDS + 1))))
    for f in FS:
        rng = np.random.default_rng(11)
        asg = np.where(rng.random(len(lb)) < f, 1 - lb, lb).astype(int)
        vals = []
        for k in range(ROUNDS + 1):
            if k == 0:
                d = run("m0", [(int(p1[i]), int(p2[i]), int(asg[i])) for i in trk])
            vals.append(contrast(d))
            if k == ROUNDS: break
            dA, dB = dv(d, NA)[mo], dv(d, NB)[mo]
            sc = np.log(Mx(dB)) - np.log(Mx(dA))
            pA = 1.0 / (1.0 + np.exp(np.clip(sc, -30, 30)))     # P(copy A)
            full_pA = np.full(len(p1), 0.5); full_pA[mo] = pA
            if mode == "hard":
                asg = (full_pA <= 0.5).astype(int)
            else:
                asg = (rng.random(len(p1)) < full_pA).astype(int)
            d = run("m%d" % (k + 1), [(int(p1[i]), int(p2[i]), int(asg[i])) for i in trk])
        print("%5.2f | %s" % (f, "  ".join("%+.4f" % v for v in vals)))
        sys.stdout.flush()
