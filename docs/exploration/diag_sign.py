import gzip, os, subprocess
import numpy as np
from scipy.stats import spearmanr
HICKIT = "native/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"; NA, NB = "c1a", "c1b"
WORK = "scratch/diag"; os.makedirs(WORK, exist_ok=True)
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
trk = np.where(hsh)[0]

path = WORK + "/t.pairs.gz"
with gzip.open(path, "wt") as f:
    f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
    for nm in (NA, NB): f.write("#chromosome: %s %d\n" % (nm, 300000000))
    f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
    for i in trk:
        nm = NA if lb[i] == 0 else NB
        f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, p1[i], nm, p2[i]))
out = WORK + "/t.3dg"
r = subprocess.run([HICKIT, "-i", path, "-P1", "-b1m", "-O", out], capture_output=True, text=True)
print("rc", r.returncode)
for ln in r.stderr.splitlines():
    if "merge" in ln or "beads" in ln or "unit" in ln: print(" ", ln)
d = {}
with open(out) as f:
    for line in f:
        if line.startswith("#unit"): print("unit:", line.split()[1])
        if line[0] == "#": continue
        a = line.split()
        if len(a) < 5: continue
        d.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
def dv(key):
    m = d[key]; o = np.full(N, np.nan)
    for k in range(N):
        if P1[k] in m and P2[k] in m: o[k] = np.linalg.norm(m[P1[k]] - m[P2[k]])
    return o
dA, dB = dv(NA), dv(NB)
ok = np.isfinite(dA) & np.isfinite(dB)
print("beads: %s=%d %s=%d" % (NA, len(d[NA]), NB, len(d[NB])))
print()
print("mean distance in structure built from phase00 contacts (c1a):")
for t, nm in ((0, "phase00"), (1, "phase11")):
    s = ok & (lb == t)
    print("  %s contacts: mean d(c1a)=%.4f  mean d(c1b)=%.4f   n=%d" % (nm, dA[s].mean(), dB[s].mean(), s.sum()))
print()
sc = np.log(np.maximum(dB,1e-3)) - np.log(np.maximum(dA,1e-3))
print("hard acc assigning label 0 -> c1a  (pred = sc>0)      : %.4f" % np.mean(((sc > 0).astype(int) == lb)[ok]))
print("hard acc assigning label 0 -> c1b  (pred = sc<=0)     : %.4f" % np.mean(((sc <= 0).astype(int) == lb)[ok]))
print()
# 接触计数与距离的方向是否正确？
pc = {}
for k in range(N):
    if ok[k]: pc.setdefault((I1[k], I2[k]), 0)
for k in range(N):
    if ok[k]: pc[(I1[k], I2[k])] += 1
K = np.array(sorted(pc)); C = np.array([pc[tuple(k)] for k in K], float)
tmp = {}
for k in range(N):
    if ok[k]: tmp[(I1[k], I2[k])] = (dA[k], dB[k])
DA = np.array([tmp[tuple(k)][0] for k in K]); DB = np.array([tmp[tuple(k)][1] for k in K])
print("corr(log count, log d_c1a) = %+.4f  (negative expected if contacts attract)" % spearmanr(np.log(C), np.log(DA))[0])
print("corr(log count, log d_c1b) = %+.4f" % spearmanr(np.log(C), np.log(DB))[0])
