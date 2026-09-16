"""试运行 v3：围绕原生 hickit FDG 的实际迭代 EM 循环。

使用全部 chr1 染色体内 contacts（拆分标签对循环隐藏）。phased 子集只用于在每轮之后评估恢复的拆分。
"""
import gzip, os, subprocess, sys, json
import numpy as np
from scipy.stats import spearmanr

HICKIT = "/work/phase3/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000
CHROM = "chr1"; CLEN = 195471971
WORK = "scratch/pilot_em"; os.makedirs(WORK, exist_ok=True)
NA, NB = "C1a", "C1b"

p1, p2, lab = [], [], []
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != CHROM or c[3] != CHROM or int(c[2]) > int(c[4]): continue
        p1.append(int(c[2])); p2.append(int(c[4]))
        ph = c[7] + c[8]
        lab.append(0 if ph == "00" else (1 if ph == "11" else -1))
p1 = np.array(p1); p2 = np.array(p2); lab = np.array(lab)
N = len(p1)
phased = lab >= 0
print("contacts=%d phased=%d" % (N, int(phased.sum())))

HDR = ("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n"
       "#chromosome: %s %d\n#chromosome: %s %d\n"
       "#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n" % (NA, CLEN, NB, CLEN))

def write_pairs(path, assign):
    with gzip.open(path, "wt") as f:
        f.write(HDR)
        for k in range(N):
            a = NA if assign[k] == 0 else NB
            f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (a, p1[k], a, p2[k]))

def run_fdg(pg, out):
    r = subprocess.run([HICKIT, "-i", pg, "-P1", "-b1m", "-O", out], capture_output=True, text=True)
    return r.returncode, r.stderr

def read_3dg(path):
    d = {}
    with open(path) as f:
        for line in f:
            if line[0] == "#": continue
            a = line.split()
            if len(a) < 5: continue
            d.setdefault(a[0], {})[int(a[1])] = np.array([float(a[2]), float(a[3]), float(a[4])])
    return d

binof = lambda p: (p - OFF)//BIN*BIN + OFF
B1, B2 = binof(p1), binof(p2)

def dists(d, key):
    m = d.get(key, {}); out = np.full(N, np.nan)
    for k in range(N):
        if B1[k] in m and B2[k] in m:
            out[k] = np.linalg.norm(m[B1[k]] - m[B2[k]])
    return out

def estep(assign, alpha):
    write_pairs(WORK + "/j.pairs.gz", assign)
    rc, err = run_fdg(WORK + "/j.pairs.gz", WORK + "/j.3dg")
    assert rc == 0, err[-800:]
    d = read_3dg(WORK + "/j.3dg")
    dA, dB = dists(d, NA), dists(d, NB)
    score = alpha * (np.log(dB + 1e-3) - np.log(dA + 1e-3))   # >0 => 属于 A
    nanm = ~np.isfinite(dA) & np.isfinite(dB)
    nanA = np.isfinite(dA) & ~np.isfinite(dB)
    score[nanm] = 1e6; score[nanA] = -1e6
    bad = ~np.isfinite(dA) & ~np.isfinite(dB)
    return score, bad

def report(tag, score, bad):
    s = score[phased & ~bad]
    t = lab[phased & ~bad]
    rho = spearmanr(s, 1 - t)[0] if len(s) > 10 else float("nan")
    acc = float(np.mean((s > 0) == (t == 0)))
    aA = float(np.mean((s[t == 0] > 0))); aB = float(np.mean((s[t == 1] < 0)))
    print("  %-22s rho=%+.4f  hard_acc=%.4f (A=%.3f B=%.3f)  nan=%d" % (tag, rho, acc, aA, aB, int(bad.sum())))
    return rho

rng = np.random.default_rng(3)
print("\n[A] ORACLE init (true labels on phased contacts, random on the rest):")
init = np.where(phased, lab, rng.integers(0, 2, N))
for it in range(6):
    sc, bad = estep(init, 1.0)
    report("iter %d" % it, sc, bad)
    init = (sc <= 0).astype(np.int64)   # 贪心 hard EM
print("\n[B] RANDOM init:")
init = rng.integers(0, 2, N)
for it in range(8):
    sc, bad = estep(init, 1.0)
    report("iter %d" % it, sc, bad)
    init = (sc <= 0).astype(np.int64)
