"""步骤 1：单倍型拆分的吸引盆半径，以及得分分解。

(A) 得分分解：留出数据的 Spearman 得分中，有多少仅来自基因组距离衰减，多少来自共识三维形状，多少来自拆分？
(B) 吸引盆半径：按比例 f 扰动真实拆分，运行若干轮 EM，并跟踪分配是否回到真实结果。

所有内容都只在 bin pair 的 TRAIN 折上拟合；所有报告数字都来自 TEST 折。
"""
import gzip, os, subprocess, sys
import numpy as np
from scipy.stats import spearmanr

HICKIT = "native/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000
WORK = "scratch/basin"; os.makedirs(WORK, exist_ok=True)
SAFE = {"chr1": ("c1a", "c1b"), "chr2": ("c2a", "c2b"), "chr3": ("c3a", "c3b")}
Mx = lambda x: np.maximum(x, 1e-3)
AL = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

def load(chrom):
    p1, p2, lb = [], [], []
    with gzip.open("data/P9016.pairs.gz", "rt") as f:
        for line in f:
            if line[0] == "#": continue
            c = line.rstrip("\n").split("\t")
            if c[1] != c[3] or c[1] != chrom or int(c[2]) > int(c[4]): continue
            ph = c[7] + c[8]
            if ph not in ("00", "11"): continue
            p1.append(int(c[2])); p2.append(int(c[4])); lb.append(0 if ph == "00" else 1)
    return np.array(p1), np.array(p2), np.array(lb)

def write_run(rows, chrom, tag, split):
    a_n, b_n = SAFE[chrom]
    path = WORK + "/%s.pairs.gz" % tag
    with gzip.open(path, "wt") as f:
        f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
        for nm in ((chrom,) if not split else (a_n, b_n)):
            f.write("#chromosome: %s %d\n" % (nm, 300000000))
        f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
        for p, q, asg in rows:
            nm = chrom if not split else (a_n if asg == 0 else b_n)
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

def dists(d, key, P1, P2):
    m = d.get(key, {}); out = np.full(len(P1), np.nan)
    for k in range(len(P1)):
        if P1[k] in m and P2[k] in m:
            out[k] = np.linalg.norm(m[P1[k]] - m[P2[k]])
    return out

# ---------------- (A) 得分分解 ----------------
print("=== (A) where does the held-out score come from? ===")
print("%-6s %8s %10s %10s %10s %10s" % ("chr", "pairs", "genome|i-j|", "consensus", "oracle2", "oracle-cons"))
for chrom in ["chr1", "chr2", "chr3"]:
    p1, p2, lb = load(chrom)
    I1 = (p1 - OFF)//BIN; I2 = (p2 - OFF)//BIN
    P1, P2 = I1*BIN + OFF, I2*BIN + OFF
    hsh = ((I1 * 7919 + I2 * 104729) % 2).astype(bool)
    trk = np.where(hsh)[0]; tek = np.where(~hsh)[0]
    tr_rows = [(int(p1[k]), int(p2[k]), int(lb[k])) for k in trk]
    dgen = np.abs(I1 - I2).astype(float)
    dcons = dists(write_run([(a, b, 0) for a, b, _ in tr_rows], chrom, "c_" + chrom, False), chrom, P1, P2)
    do = write_run(tr_rows, chrom, "o_" + chrom, True)
    dA = dists(do, SAFE[chrom][0], P1, P2); dB = dists(do, SAFE[chrom][1], P1, P2)
    te = {}
    for k in tek: te.setdefault((I1[k], I2[k]), 0)
    TK = np.array(sorted(te)); nte = np.array([sum(1 for k in tek if (I1[k], I2[k]) == tuple(t)) for t in TK], float)
    te_mask = np.zeros(len(p1), bool)
    def tvec(v):
        m = {(I1[k], I2[k]): v[k] for k in tek}
        return np.array([m[tuple(t)] for t in TK])
    ok = np.isfinite(tvec(dcons)) & np.isfinite(tvec(dA)) & np.isfinite(tvec(dB))
    def fit_alpha(pred):
        best = (-9, None)
        trmask = ok
        for a in AL:
            r = spearmanr(pred(a)[trmask], nte[trmask])[0]
            if r > best[0]: best = (r, a)
        return best[1]
    tg = np.array([np.abs(t[0]-t[1]) for t in TK], float)
    def gen(a): return tg ** -a
    ag = fit_alpha(gen); ac = fit_alpha(lambda a: Mx(tvec(dcons))**-a)
    ao = fit_alpha(lambda a: Mx(tvec(dA))**-a + Mx(tvec(dB))**-a)
    rg = spearmanr(gen(ag)[ok], nte[ok])[0]
    rc = spearmanr(Mx(tvec(dcons))[ok]**-ac, nte[ok])[0]
    ro = spearmanr((Mx(tvec(dA))**-ao + Mx(tvec(dB))**-ao)[ok], nte[ok])[0]
    print("%-6s %8d %10.4f %10.4f %10.4f %+10.4f" % (chrom, int(ok.sum()), rg, rc, ro, ro - rc))
sys.stdout.flush()

# ---------------- (B) 吸引盆半径 ----------------
print()
print("=== (B) basin radius on chr1: does the EM climb back to the truth? ===")
chrom = "chr1"
p1, p2, lb = load(chrom)
I1 = (p1 - OFF)//BIN; I2 = (p2 - OFF)//BIN
P1, P2 = I1*BIN + OFF, I2*BIN + OFF
hsh = ((I1 * 7919 + I2 * 104729) % 2).astype(bool)
trk = np.where(hsh)[0]; tek = np.where(~hsh)[0]
ROUNDS = 8
FS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
SEEDS = [0, 1]
print("agree = fraction of TEST contacts assigned to the same copy as the truth")
print("%5s %5s | %s" % ("f", "seed", "  ".join("k%-5d" % k for k in range(ROUNDS + 1))))
curves = {}
for f in FS:
    for sd in SEEDS:
        rng = np.random.default_rng(100 + sd)
        flip = rng.random(len(lb)) < f
        asg = np.where(flip, 1 - lb, lb).astype(int)
        agree = [float(np.mean(asg[tek] == lb[tek]))]
        for k in range(ROUNDS):
            rows = [(int(p1[i]), int(p2[i]), int(asg[i])) for i in trk]
            d = write_run(rows, chrom, "b_%s" % chrom, True)
            dA = dists(d, SAFE[chrom][0], P1, P2); dB = dists(d, SAFE[chrom][1], P1, P2)
            sc = np.log(Mx(dB)) - np.log(Mx(dA))
            new = np.where(np.isfinite(sc), (sc <= 0).astype(int), asg)
            asg = new
            agree.append(float(np.mean(new[tek] == lb[tek])))
        curves[(f, sd)] = agree
        print("%5.2f %5d | %s" % (f, sd, "  ".join("%.3f" % v for v in agree)))
        sys.stdout.flush()
print()
print("=== summary: agreement at k=0 and at k=%d ===" % ROUNDS)
print("%5s %10s %10s %10s" % ("f", "k=0", "k=%d" % ROUNDS, "change"))
for f in FS:
    a0 = np.mean([curves[(f, s)][0] for s in SEEDS])
    aR = np.mean([curves[(f, s)][ROUNDS] for s in SEEDS])
    print("%5.2f %10.3f %10.3f %+10.3f" % (f, a0, aR, aR - a0))
