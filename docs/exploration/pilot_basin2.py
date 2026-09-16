"""试运行 v2：联合二倍体 FDG（两个拷贝位于同一公共坐标系）。

将 contacts 重命名为 chr1(copyA)/chr1(copyB)，使单次 hickit FDG 在同一 shared unit、shared frame 和 inter-copy repulsion 下联合放置 2N 个 beads。
"""
import gzip, os, subprocess, sys
import numpy as np

HICKIT = "/work/phase3/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000
CHROM = "chr1"; CHROM_LEN = 195471971
WORK = "scratch/pilot2"; os.makedirs(WORK, exist_ok=True)
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
truth = np.array([r[2] for r in recs])
p1 = np.array([r[0] for r in recs]); p2 = np.array([r[1] for r in recs])
print("phased cis contacts:", len(recs))

HDR = ("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n"
       "#chromosome: %s %d\n#chromosome: %s %d\n"
       "#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n"
       % (NA, CHROM_LEN, NB, CHROM_LEN))

def write_pairs(path, assign):
    with gzip.open(path, "wt") as f:
        f.write(HDR)
        for k in range(len(recs)):
            a = NA if assign[k] == 0 else NB
            f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (a, p1[k], a, p2[k]))

def run_fdg(pg, out):
    r = subprocess.run([HICKIT, "-i", pg, "-P1", "-b1m", "-O", out], capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL", r.stderr[-1500:]); sys.exit(1)

def read_3dg(path):
    d = {}
    with open(path) as f:
        for line in f:
            if line[0] == "#": continue
            a = line.split()
            if len(a) < 5: continue
            d.setdefault(a[0], {})[int(a[1])] = np.array([float(a[2]), float(a[3]), float(a[4])])
    return d

def dists(d, p1, p2):
    binof = lambda p: (p - OFF) // BIN * BIN + OFF
    out = np.full(len(p1), np.nan)
    for k in range(len(p1)):
        b1, b2 = binof(p1[k]), binof(p2[k])
        a = d.get(NA, {}); b = d.get(NB, {})
        if b1 in a and b2 in a: out[k] = np.linalg.norm(a[b1] - a[b2])
    return out

rng = np.random.default_rng(11)
print("%6s %8s %8s %8s %8s %8s" % ("noise", "acc", "accA", "accB", "nA", "nB"))
for f in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    assign = np.where(rng.random(len(truth)) < f, 1 - truth, truth)
    write_pairs(WORK + "/joint.pairs.gz", assign)
    run_fdg(WORK + "/joint.pairs.gz", WORK + "/joint.3dg")
    d3 = read_3dg(WORK + "/joint.3dg")
    nA = len(d3.get(NA, {})); nB = len(d3.get(NB, {}))
    dA = dists(d3, p1, p2)
    # same-code distances：chr1(copyA) 与 chr1(copyB) 都按 NA/NB 索引；需要分别查找
    def dists_key(d, key, p1, p2):
        binof = lambda p: (p - OFF) // BIN * BIN + OFF
        m = d.get(key, {}); out = np.full(len(p1), np.nan)
        for k in range(len(p1)):
            b1, b2 = binof(p1[k]), binof(p2[k])
            if b1 in m and b2 in m: out[k] = np.linalg.norm(m[b1] - m[b2])
        return out
    dA = dists_key(d3, NA, p1, p2); dB = dists_key(d3, NB, p1, p2)
    ok = np.isfinite(dA) & np.isfinite(dB)
    pred = np.where(dA <= dB, 0, 1)
    acc = float(np.mean(pred[ok] == truth[ok]))
    aA = float(np.mean(pred[ok & (truth == 0)] == 0)); aB = float(np.mean(pred[ok & (truth == 1)] == 1))
    print("%6.2f %8.4f %8.4f %8.4f %8d %8d" % (f, acc, aA, aB, nA, nB))
