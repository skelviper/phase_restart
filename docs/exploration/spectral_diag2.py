"""联合坐标系诊断：在一次 FDG 运行中拟合 consensus 和两个真实拷贝。"""
import gzip, os, subprocess
import numpy as np
from scipy.stats import spearmanr
HICKIT = "native/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"
NC, NA, NB = "c1c", "c1a", "c1b"
WORK = "scratch/spec3"; os.makedirs(WORK, exist_ok=True)
NB_ = 193

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
kb = (p1 - OFF)//BIN; kc = (p2 - OFF)//BIN

rows = []
for i in range(N):
    rows.append((int(p1[i]), int(p2[i]), NC))
    rows.append((int(p1[i]), int(p2[i]), NA if lb[i] == 0 else NB))
path = WORK + "/joint.pairs.gz"
with gzip.open(path, "wt") as f:
    f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
    for nm in (NC, NA, NB): f.write("#chromosome: %s %d\n" % (nm, 300000000))
    f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
    for p, q, nm in rows:
        f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, p, nm, q))
r = subprocess.run([HICKIT, "-i", path, "-P1", "-b1m", "-O", WORK + "/joint.3dg"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-400:]
d = {}
with open(WORK + "/joint.3dg") as f:
    for line in f:
        if line[0] == "#": continue
        a = line.split()
        if len(a) < 5: continue
        d.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
print("beads:", {k: len(d[k]) for k in (NC, NA, NB)})

def to_grid(key):
    starts = np.array(sorted(d[key])); pos = np.array([d[key][p] for p in starts])
    out = np.full((NB_, 3), np.nan)
    for i in range(NB_):
        j = np.searchsorted(starts, OFF + i*BIN, side="right") - 1
        if j >= 0: out[i] = pos[j]
    return out
Gz, Ga, Gb = to_grid(NC), to_grid(NA), to_grid(NB)
ok = np.isfinite(Gz).all(1) & np.isfinite(Ga).all(1) & np.isfinite(Gb).all(1)
print("bins usable: %d/193" % ok.sum())
dm = lambda G: np.linalg.norm(G[:, None, :] - G[None, :, :], axis=-1)
DZ, DA, DB = dm(Gz), dm(Ga), dm(Gb)
iu = np.triu_indices(NB_, 1)
sp = lambda a, b: spearmanr(a, b)[0]
TRUTH = 1 - sp(DA[iu], DB[iu])
print("truth contrast = %+.4f" % TRUTH)

cnt = np.zeros((NB_, NB_))
for i in range(N): cnt[kb[i], kc[i]] += 1
cnt = cnt + cnt.T
kk, ll = np.where(np.triu(cnt > 0, 1))
dbar = DZ[kk, ll]
delta_true = (DA[kk, ll] - DB[kk, ll]) / 2.0
V = Gz[kk] - Gz[ll]
Vh = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
S = np.zeros((len(kk), 3*NB_)); row = np.arange(len(kk))
for c in range(3):
    S[row, 3*kk + c] += Vh[:, c]; S[row, 3*ll + c] -= Vh[:, c]
print("pairs %d | mean dbar %.3f | mean|delta_true| %.3f | mean|delta/dbar| %.3f" % (
    len(kk), dbar.mean(), np.abs(delta_true).mean(), np.abs(delta_true/dbar).mean()))
u_grid = (Ga - Gb).reshape(-1) / 2.0
pred = S @ u_grid
print("LINEARISATION self-check: rel err = %.4f ; corr(pred, delta_true) = %.4f" % (
    np.linalg.norm(pred - delta_true)/np.linalg.norm(delta_true), sp(pred, delta_true)))
rmsu = np.sqrt((u_grid**2).mean()); print("rms |u_grid| = %.3f ; rms dbar = %.3f" % (rmsu, np.sqrt((dbar**2).mean())))
