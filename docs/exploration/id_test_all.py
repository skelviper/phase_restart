"""干净的真实数据可辨识性检验，覆盖所有染色体。

GOLD = 两端都已 phased 的 contacts（因此 oracle split 是精确的）。
结构只在 TRAIN 折上拟合；所有结果都在 TEST 折上评分。

  sum test   ：预测留出的每个 bin pair 的 TOTAL counts（无标签判据）
  label test ：预测留出的 copy-specific counts（阳性对照）
"""
import gzip, os, subprocess, sys
import numpy as np
from scipy.stats import spearmanr

HICKIT = "native/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000
WORK = "scratch/id2"; os.makedirs(WORK, exist_ok=True)
CHRS = ["chr%d" % k for k in range(1, 20)] + ["chrX"]

data = {}
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != c[3] or c[1] not in CHRS or int(c[2]) > int(c[4]): continue
        ph = c[7] + c[8]
        if ph not in ("00", "11"): continue
        t = 0 if ph == "00" else 1
        data.setdefault(c[1], []).append((int(c[2]), int(c[4]), t))

SAFE = {}
for _i, _c in enumerate(CHRS):
    SAFE[_c] = ("cc%02da" % _i, "cc%02db" % _i)

def write_pairs(path, rows, split, chrom):
    names = (chrom,) if not split else SAFE[chrom]
    with gzip.open(path, "wt") as f:
        f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
        for nm in names: f.write("#chromosome: %s %d\n" % (nm, 300000000))
        f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
        for a, b, asg in rows:
            nm = chrom if not split else (SAFE[chrom][0] if asg == 0 else SAFE[chrom][1])
            f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, a, nm, b))

def run(pairs, out):
    r = subprocess.run([HICKIT, "-i", pairs, "-P1", "-b1m", "-O", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    d = {}
    with open(out) as f:
        for line in f:
            if line[0] == "#": continue
            a = line.split()
            if len(a) < 5: continue
            d.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
    return d

Mx = lambda x: np.maximum(x, 1e-3)
AL = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
res = []
for chrom in CHRS:
    rows = data.get(chrom, [])
    if len(rows) < 3000: continue
    p1 = np.array([r[0] for r in rows], dtype=np.int64)
    p2 = np.array([r[1] for r in rows], dtype=np.int64)
    lb = np.array([r[2] for r in rows])
    I1 = (p1 - OFF)//BIN; I2 = (p2 - OFF)//BIN
    hsh = ((I1 * 7919 + I2 * 104729) % 2).astype(bool)      # True 表示 train
    tr = [(int(p1[k]), int(p2[k]), int(lb[k])) for k in np.where(hsh)[0]]
    te = {}
    for k in np.where(~hsh)[0]:
        te.setdefault((I1[k], I2[k]), [0, 0])[lb[k]] += 1
    TK = np.array(sorted(te)); TC = np.array([te[tuple(k)] for k in TK], float)
    if len(TK) < 200: continue

    def tv(d, key):
        m = d[key]; out = np.full(len(TK), np.nan)
        for r_ in range(len(TK)):
            a, b = int(TK[r_, 0])*BIN + OFF, int(TK[r_, 1])*BIN + OFF
            if a in m and b in m: out[r_] = np.linalg.norm(m[a] - m[b])
        return out

    dcons = tv(run(*["%s/cons_%s.pairs.gz" % (WORK, chrom),
                    "%s/cons_%s.3dg" % (WORK, chrom)]) if False else
               (lambda pp, oo: (write_pairs(pp, [(a, b, 0) for a, b, _ in tr], False, chrom), run(pp, oo))[1])(
                   "%s/cons_%s.pairs.gz" % (WORK, chrom), "%s/cons_%s.3dg" % (WORK, chrom)), chrom)
    do = write_pairs("%s/or_%s.pairs.gz" % (WORK, chrom), tr, True, chrom)
    do = run("%s/or_%s.pairs.gz" % (WORK, chrom), "%s/or_%s.3dg" % (WORK, chrom))
    dOA, dOB = tv(do, SAFE[chrom][0]), tv(do, SAFE[chrom][1])

    seeds = [0, 1, 2]
    rand_sums = []
    rng = np.random.default_rng(7)
    for s in seeds:
        asg = rng.integers(0, 2, len(tr))
        rr = [(tr[k][0], tr[k][1], int(asg[k])) for k in range(len(tr))]
        write_pairs("%s/rd_%s_%d.pairs.gz" % (WORK, chrom, s), rr, True, chrom)
        dr = run("%s/rd_%s_%d.pairs.gz" % (WORK, chrom, s), "%s/rd_%s_%d.3dg" % (WORK, chrom, s))
        dRA, dRB = tv(dr, SAFE[chrom][0]), tv(dr, SAFE[chrom][1])
        if s == 0: dRA0, dRB0 = dRA, dRB
        rand_sums.append((dRA, dRB))

    ok = np.isfinite(dcons) & np.isfinite(dOA) & np.isfinite(dOB)
    for a, b in rand_sums: ok &= np.isfinite(a) & np.isfinite(b)
    tot = TC.sum(1); nA, nB = TC[:, 0], TC[:, 1]

    def rho_sum(a, b, alpha):
        return spearmanr(Mx(a[ok])**-alpha + Mx(b[ok])**-alpha, tot[ok])[0]
    def rho_lab(a, b, alpha):
        return (spearmanr(Mx(a[ok])**-alpha, nA[ok])[0], spearmanr(Mx(b[ok])**-alpha, nB[ok])[0])

    bo = max(((rho_sum(dOA, dOB, a), a) for a in AL))
    bc = max(((spearmanr(Mx(dcons[ok])**-a, tot[ok])[0], a) for a in AL))
    br = max((np.mean([rho_sum(x, y, a) for x, y in rand_sums]), a) for a in AL)
    lo = rho_lab(dOA, dOB, bo[1])
    lr = rho_lab(dRA0, dRB0, bo[1])
    lc = spearmanr(Mx(dcons[ok])**-bo[1], nA[ok])[0]
    res.append((chrom, int(ok.sum()), int(tot[ok].sum()), bo[0], br[0], bc[0], lo, lr, lc))
    print("%-5s pairs=%5d cts=%6d | SUM oracle=%.4f random=%.4f cons=%.4f | LABEL oracle=(%.3f,%.3f) random=(%.3f,%.3f) consA=%.3f" %
          (chrom, int(ok.sum()), int(tot[ok].sum()), bo[0], br[0], bc[0], lo[0], lo[1], lr[0], lr[1], lc))
    sys.stdout.flush()

A = np.array([[r[3], r[4], r[5]] for r in res])
L = np.array([[r[6][0], r[6][1], r[7][0], r[7][1], r[8]] for r in res])
print()
print("=== %d chromosomes, held-out bin-pair Spearman ===" % len(res))
print("SUM   oracle  mean %.4f (sd %.4f)" % (A[:,0].mean(), A[:,0].std()))
print("SUM   random  mean %.4f (sd %.4f)" % (A[:,1].mean(), A[:,1].std()))
print("SUM   cons-1  mean %.4f (sd %.4f)" % (A[:,2].mean(), A[:,2].std()))
d = A[:,0] - A[:,1]
print("SUM   oracle - random : mean %+.4f  sd %.4f  paired t = %.2f  (n=%d)" % (d.mean(), d.std(ddof=1), d.mean()/(d.std(ddof=1)/np.sqrt(len(d))), len(d)))
print()
print("LABEL own-copy oracle mean (%.4f, %.4f) | random (%.4f, %.4f) | consensus-vs-A %.4f" %
      (L[:,0].mean(), L[:,1].mean(), L[:,2].mean(), L[:,3].mean(), L[:,4].mean()))