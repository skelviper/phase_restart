"""GATE 0（稳健版本）：双结构 mixture 对留出 contacts 的预测是否优于单一共识结构？

在聚合后的 bin-pair counts 上使用基于 rank 的（Spearman）评分，避免 Poisson MLE 病态。对照包括随机拆分结构和最佳单结构。
"""
import gzip, os, subprocess
import numpy as np
from scipy.stats import spearmanr

HICKIT = "/work/phase3/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"; CLEN = 195471971
WORK = "scratch/pilot_gate0b"; os.makedirs(WORK, exist_ok=True)
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
p1 = np.array(p1); p2 = np.array(p2); lab = np.array(lab); N = len(p1)
B1i = (p1 - OFF)//BIN; B2i = (p2 - OFF)//BIN
B1 = B1i*BIN + OFF; B2 = B2i*BIN + OFF

def write(assign, tag, single=False):
    path = WORK + "/%s.pairs.gz" % tag
    with gzip.open(path, "wt") as f:
        f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
        if single:
            f.write("#chromosome: %s %d\n" % (CHROM, CLEN))
            f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
            for k in range(N):
                f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (CHROM, p1[k], CHROM, p2[k]))
        else:
            f.write("#chromosome: %s %d\n#chromosome: %s %d\n" % (NA, CLEN, NB, CLEN))
            f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
            for k in range(N):
                nm = NA if assign[k] == 0 else NB
                f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, p1[k], nm, p2[k]))
    out = WORK + "/%s.3dg" % tag
    r = subprocess.run([HICKIT, "-i", path, "-P1", "-b1m", "-O", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    d = {}
    with open(out) as f:
        for line in f:
            if line[0] == "#": continue
            a = line.split()
            if len(a) < 5: continue
            d.setdefault(a[0], {})[int(a[1])] = np.array([float(a[2]), float(a[3]), float(a[4])])
    return d

# 按 bin pair 聚合 counts，并保留每个 pair 的一个代表 contact index
pair_count = {}
for k in range(N):
    pair_count[(B1i[k], B2i[k])] = pair_count.get((B1[k], B2[k]), 0) + 1
keys = list(pair_count)
print("distinct bin pairs:", len(keys), "contacts:", N)
keyarr = np.array(keys); ncnt = np.array([pair_count[k] for k in keys], float)

def dvec(d, key):
    m = d[key]; out = np.full(len(keyarr), np.nan)
    for r in range(len(keyarr)):
        i, j = int(keyarr[r][0]), int(keyarr[r][1])
        pi, pj = i*BIN + OFF, j*BIN + OFF
        if pi in m and pj in m: out[r] = np.linalg.norm(m[pi] - m[pj])
    return out

dc = write(np.zeros(N), "cons", single=True)
d_cons = dvec(dc, CHROM)
do = write(np.where(lab == 1, 1, 0), "oracle")
d_A = dvec(do, NA); d_B = dvec(do, NB)
dr = write(np.random.default_rng(5).integers(0, 2, N), "rand")
d_A2 = dvec(dr, NA); d_B2 = dvec(dr, NB)
ok = np.isfinite(d_cons) & np.isfinite(d_A) & np.isfinite(d_B) & np.isfinite(d_A2) & np.isfinite(d_B2)
print("usable pairs:", int(ok.sum()))

# 在 bin pairs 上进行确定性的 train/test 拆分
h = ((keyarr[:, 0] * 7919 + keyarr[:, 1] * 104729) % 2).astype(bool)
tr = ok & h; te = ok & ~h
print("train pairs %d (contacts %d) / test pairs %d (contacts %d)" % (tr.sum(), ncnt[tr].sum(), te.sum(), ncnt[te].sum()))

alphas = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
def best_alpha(pred_fn):
    best, ba = -9, None
    for a in alphas:
        r = spearmanr(pred_fn(a)[tr], ncnt[tr])[0]
        if r > best: best, ba = r, a
    return ba, best

a1, s1 = best_alpha(lambda a: d_cons ** -a)
a2, s2 = best_alpha(lambda a: d_A ** -a + d_B ** -a)
a3, s3 = best_alpha(lambda a: d_A2 ** -a + d_B2 ** -a)
print()
print("%-34s %8s %10s %10s" % ("predictor", "alpha*", "train_rho", "test_rho"))
rows = [
    ("1 structure (consensus)", lambda a: d_cons ** -a),
    ("2 structures (oracle split)", lambda a: d_A ** -a + d_B ** -a),
    ("2 structures (random split)", lambda a: d_A2 ** -a + d_B2 ** -a),
    ("1 structure = oracle copy A only", lambda a: d_A ** -a),
    ("1 structure = oracle copy B only", lambda a: d_B ** -a),
]
out = {}
for name, fn in rows:
    a, str_ = best_alpha(fn)
    te_r = spearmanr(fn(a)[te], ncnt[te])[0]
    out[name] = te_r
    print("%-34s %8.2f %10.4f %10.4f" % (name, a, str_, te_r))
print()
print("Delta test rho: oracle-2struct - consensus-1struct = %+.4f" % (out["2 structures (oracle split)"] - out["1 structure (consensus)"]))
print("Delta test rho: random-2struct - consensus-1struct = %+.4f" % (out["2 structures (random split)"] - out["1 structure (consensus)"]))
print("Delta test rho: oracle-2struct - random-2struct     = %+.4f" % (out["2 structures (oracle split)"] - out["2 structures (random split)"]))