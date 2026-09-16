"""Oracle 检查：按真实 phase labels 拆分 chr1，运行联合 FDG，并与 reference 3dg 比较。"""
import gzip, os, subprocess
import numpy as np
from scipy.stats import spearmanr

HICKIT = "/work/phase3/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000
WORK = "scratch/pilot_oracle"; os.makedirs(WORK, exist_ok=True)
CHROMS = [("chr1", 195471971), ("chr2", 182113224)]
NAME = {}
for c, _ in CHROMS:
    NAME[c] = (c + "A", c + "B")

rows = {c: [] for c, _ in CHROMS}
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        x = line.rstrip("\n").split("\t")
        if x[1] != x[3] or x[1] not in rows: continue
        if int(x[2]) > int(x[4]): continue
        p = x[7] + x[8]
        if p == "00": t = 0
        elif p == "11": t = 1
        else: continue
        rows[x[1]].append((int(x[2]), int(x[4]), t))

hdr = ["## pairs format v1.0", "#sorted: chr1-chr2-pos1-pos2", "#shape: upper triangle"]
for c, L in CHROMS:
    hdr.append("#chromosome: %s %d" % (NAME[c][0], L))
    hdr.append("#chromosome: %s %d" % (NAME[c][1], L))
hdr.append("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2")
with gzip.open(WORK + "/oracle.pairs.gz", "wt") as f:
    f.write("\n".join(hdr) + "\n")
    for c, _ in CHROMS:
        for a, b, t in rows[c]:
            nm = NAME[c][t]
            f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, a, nm, b))
r = subprocess.run([HICKIT, "-i", WORK + "/oracle.pairs.gz", "-P1", "-b1m", "-O", WORK + "/oracle.3dg"],
                   capture_output=True, text=True)
print("hickit rc", r.returncode)
got = {}
with open(WORK + "/oracle.3dg") as f:
    for line in f:
        if line[0] == "#": continue
        a = line.split()
        if len(a) < 5: continue
        got.setdefault(a[0], {})[int(a[1])] = np.array([float(a[2]), float(a[3]), float(a[4])])
ref = {}
with gzip.open("data/P9016.1m.3dg.gz", "rt") as f:
    for line in f:
        a = line.split()
        ref.setdefault(a[0], {})[int(a[1])] = np.array([float(a[2]), float(a[3]), float(a[4])])

def dmat(d, pos):
    X = np.array([d[p] for p in pos]); return np.linalg.norm(X[:, None] - X[None, :], axis=-1)

for c, L in CHROMS:
    for our, refk in ((NAME[c][0], c + "(pat)"), (NAME[c][1], c + "(mat)")):
        common = sorted(set(got[our]) & set(ref[refk]))
        if len(common) < 20: print(c, our, "too few common beads", len(common)); continue
        A = dmat(got[our], common); B = dmat(ref[refk], common)
        iu = np.triu_indices(len(common), 1)
        rgA = np.sqrt(((np.array([got[our][p] for p in common]) - np.array([got[our][p] for p in common]).mean(0))**2).sum(1).mean())
        rgB = np.sqrt(((np.array([ref[refk][p] for p in common]) - np.array([ref[refk][p] for p in common]).mean(0))**2).sum(1).mean())
        print("%s %-8s vs %-10s beads=%3d  spearman(D)=%.4f  pearson=%.4f  Rg ours=%.2f ref=%.2f" % (
            c, our, refk, len(common), spearmanr(A[iu], B[iu])[0], np.corrcoef(A[iu], B[iu])[0, 1], rgA, rgB))
    # 交叉检查
    for our, refk in ((NAME[c][0], c + "(mat)"), (NAME[c][1], c + "(pat)")):
        common = sorted(set(got[our]) & set(ref[refk]))
        A = dmat(got[our], common); B = dmat(ref[refk], common); iu = np.triu_indices(len(common), 1)
        print("  [cross] %s %-8s vs %-10s spearman=%.4f" % (c, our, refk, spearmanr(A[iu], B[iu])[0]))
