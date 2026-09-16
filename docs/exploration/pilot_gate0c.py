import gzip, os, subprocess
import numpy as np
from scipy.stats import spearmanr

HICKIT = "/work/phase3/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"; CLEN = 195471971
WORK = "scratch/pilot_gate0b"
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
I1 = (p1 - OFF)//BIN; I2 = (p2 - OFF)//BIN

def run(assign, tag, single=False):
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

# 不同的 bin pairs
pc = {}
for k in range(N):
    pc[(I1[k], I2[k])] = pc.get((I1[k], I2[k]), 0) + 1
K = np.array(sorted(pc)); C = np.array([pc[tuple(k)] for k in K], float)
print("distinct bin pairs %d, contacts %d, contacts/pair mean %.2f max %d" % (len(K), int(C.sum()), C.mean(), int(C.max())))

def dv(d, key):
    m = d[key]; out = np.full(len(K), np.nan)
    for r in range(len(K)):
        a, b = int(K[r, 0])*BIN + OFF, int(K[r, 1])*BIN + OFF
        if a in m and b in m: out[r] = np.linalg.norm(m[a] - m[b])
    return out

d_cons = dv(run(np.zeros(N), "cons", single=True), CHROM)
do = run(np.where(lab == 1, 1, 0), "oracle")
d_A, d_B = dv(do, NA), dv(do, NB)
dr = run(np.random.default_rng(5).integers(0, 2, N), "rand")
d_A2, d_B2 = dv(dr, NA), dv(dr, NB)
ok = np.isfinite(d_cons) & np.isfinite(d_A) & np.isfinite(d_B) & np.isfinite(d_A2) & np.isfinite(d_B2)
print("usable pairs %d (contacts %d)" % (ok.sum(), int(C[ok].sum())))
h = ((K[:, 0]*7919 + K[:, 1]*104729) % 2).astype(bool)
tr, te = ok & h, ok & ~h
print("train pairs %d contacts %d | test pairs %d contacts %d" % (tr.sum(), int(C[tr].sum()), te.sum(), int(C[te].sum())))

def sp(pred, mask):
    a, b = pred[mask], C[mask]
    if np.std(a) == 0 or np.std(b) == 0: return float("nan")
    return spearmanr(a, b)[0]

print()
print("%-38s %-28s %10s %10s" % ("predictor", "alpha", "train_rho", "test_rho"))
PRED = [
    ("1 struct: consensus", lambda a: np.maximum(d_cons, 1e-3) ** -a),
    ("2 struct: oracle split (sum)", lambda a: np.maximum(d_A, 1e-3)**-a + np.maximum(d_B, 1e-3)**-a),
    ("2 struct: random split (sum)", lambda a: np.maximum(d_A2, 1e-3)**-a + np.maximum(d_B2, 1e-3)**-a),
    ("1 struct: oracle copy A", lambda a: np.maximum(d_A, 1e-3) ** -a),
    ("1 struct: oracle copy B", lambda a: np.maximum(d_B, 1e-3) ** -a),
]
res = {}
for name, fn in PRED:
    best = (-9, None, None)
    for a in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        rtr = sp(fn(a), tr); rte = sp(fn(a), te)
        if rtr == rtr and rtr > best[0]: best = (rtr, a, rte)
    res[name] = (best[1], best[2])
    print("%-38s %-28s %10.4f %10.4f" % (name, "alpha=%.1f" % best[1], best[0], best[2]))
print()
o = res["2 struct: oracle split (sum)"][1]; c1 = res["1 struct: consensus"][1]; rnd = res["2 struct: random split (sum)"][1]
print("test rho: consensus-1struct      = %+.4f" % c1)
print("test rho: oracle-2struct         = %+.4f   (delta vs consensus = %+.4f)" % (o, o - c1))
print("test rho: random-2struct         = %+.4f   (delta vs consensus = %+.4f)" % (rnd, rnd - c1))
print("test rho: oracle-2 - random-2    = %+.4f" % (o - rnd))
