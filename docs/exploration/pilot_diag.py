"""诊断 A/B 不对称性：结构尺度与距离分布。"""
import gzip, os, subprocess, sys
import numpy as np
HICKIT = "/work/phase3/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000
CHROM = "chr1"; CHROM_LEN = 195471971
WORK = "scratch/pilot3"; os.makedirs(WORK, exist_ok=True)
NA, NB = CHROM + "(copyA)", CHROM + "(copyB)"
recs = []
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != CHROM or c[3] != CHROM or int(c[2]) > int(c[4]): continue
        if c[7] == "0" and c[8] == "0": t = 0
        elif c[7] == "1" and c[8] == "1": t = 1
        else: continue
        recs.append((int(c[2]), int(c[4]), t))
truth = np.array([r[2] for r in recs]); p1 = np.array([r[0] for r in recs]); p2 = np.array([r[1] for r in recs])
print("n00=%d n11=%d" % ((truth==0).sum(), (truth==1).sum()))
HDR = ("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n"
       "#chromosome: %s %d\n#chromosome: %s %d\n#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n"
       % (NA, CHROM_LEN, NB, CHROM_LEN))
with gzip.open(WORK + "/j.pairs.gz", "wt") as f:
    f.write(HDR)
    for k in range(len(recs)):
        a = NA if truth[k] == 0 else NB
        f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (a, p1[k], a, p2[k]))
r = subprocess.run([HICKIT, "-i", WORK + "/j.pairs.gz", "-P1", "-b1m", "-O", WORK + "/j.3dg", "-v3"], capture_output=True, text=True)
print("hickit rc", r.returncode)
for ln in r.stderr.splitlines():
    if "unit" in ln.lower() or "bead" in ln.lower() or "FDG" in ln or "fdg" in ln: print("  ", ln)
d = {}
unit = None
with open(WORK + "/j.3dg") as f:
    for line in f:
        if line.startswith("#unit"): unit = float(line.split()[1])
        if line[0] == "#": continue
        a = line.split()
        d.setdefault(a[0], {})[int(a[1])] = np.array([float(a[2]), float(a[3]), float(a[4])])
print("unit:", unit)
for k in (NA, NB):
    X = np.array(list(d[k].values()))
    print("%s beads=%d Rg=%.4f mean pairwise=%.4f centroid=%s" % (
        k, len(X), np.sqrt(((X - X.mean(0))**2).sum(1).mean()),
        np.linalg.norm(X[:,None]-X[None,:],axis=-1).mean(),
        np.round(X.mean(0), 3)))
binof = lambda p: (p - OFF)//BIN*BIN + OFF
def ds(key):
    m = d[key]; out = np.full(len(p1), np.nan)
    for k in range(len(p1)):
        b1, b2 = binof(p1[k]), binof(p2[k])
        if b1 in m and b2 in m: out[k] = np.linalg.norm(m[b1]-m[b2])
    return out
dA, dB = ds(NA), ds(NB)
ok = np.isfinite(dA) & np.isfinite(dB)
for t, nm in ((0, "truth0"), (1, "truth1")):
    s = ok & (truth == t)
    print("%s: mean dA=%.4f mean dB=%.4f mean(dA-dB)=%+.4f  frac dA<dB=%.4f" % (
        nm, dA[s].mean(), dB[s].mean(), (dA[s]-dB[s]).mean(), np.mean(dA[s] < dB[s])))
print("paired dA-dB: mean %+.4f sd %.4f" % ((dA[ok]-dB[ok]).mean(), (dA[ok]-dB[ok]).std()))
# 按尺度归一化的分类
XA = np.array(list(d[NA].values())); XB = np.array(list(d[NB].values()))
rgA = np.sqrt(((XA-XA.mean(0))**2).sum(1).mean()); rgB = np.sqrt(((XB-XB.mean(0))**2).sum(1).mean())
pred = np.where(dA/rgA <= dB/rgB, 0, 1)
print("scale-normalized acc=%.4f accA=%.4f accB=%.4f" % (
    np.mean(pred[ok]==truth[ok]), np.mean(pred[ok&(truth==0)]==0), np.mean(pred[ok&(truth==1)]==1)))
# rank-normalized：使用距离的全局 rank
ra = np.argsort(np.argsort(dA[ok]))/ok.sum(); rb = np.argsort(np.argsort(dB[ok]))/ok.sum()
pred2 = np.where(ra <= rb, 0, 1)
print("rank-normalized acc=%.4f" % np.mean(pred2 == truth[ok]))
