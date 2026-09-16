"""试运行：测量单倍型拆分的吸引盆。

给定带噪声的真实逐接触单倍型分配，分别为两个拷贝拟合一个 FDG 结构，并依据哪个结构给出的距离更近来重新分类 contacts。扫描噪声比例并测量恢复情况。
"""
import gzip, os, subprocess, sys
import numpy as np

HICKIT = "/work/phase3/hickit/hickit"
BIN = 1_000_000
CHROM = "chr1"
CHROM_LEN = 195471971
WORK = "scratch/pilot1"
os.makedirs(WORK, exist_ok=True)

# ---- 读取带 phase labels 的 chr1 染色体内 contacts
recs = []   # (pos1, pos2, truth)；truth=0 表示 phase00，1 表示 phase11
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != CHROM or c[3] != CHROM: continue
        if int(c[2]) > int(c[4]): continue
        if c[7] == "0" and c[8] == "0": t = 0
        elif c[7] == "1" and c[8] == "1": t = 1
        else: continue
        recs.append((int(c[2]), int(c[4]), t))
print("phased cis contacts on %s: %d" % (CHROM, len(recs)))

# ---- 同时加载后续“全数据”变体所需的 unphased contacts
HDR = ("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n"
       "#chromosome: %s %d\n#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n"
       % (CHROM, CHROM_LEN))

def write_pairs(path, rows):
    with gzip.open(path, "wt") as f:
        f.write(HDR)
        for p1, p2 in rows:
            f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (CHROM, p1, CHROM, p2))

def run_fdg(pairs_gz, out_3dg):
    cmd = [HICKIT, "-i", pairs_gz, "-P1", "-b1m", "-O", out_3dg]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("hickit failed:", r.stderr[-800:]); sys.exit(1)

def read_3dg(path):
    pos, xyz = [], []
    with open(path) as f:
        for line in f:
            if line[0] == "#": continue
            a = line.split()
            if len(a) < 5: continue
            pos.append(int(a[1])); xyz.append([float(a[2]), float(a[3]), float(a[4])])
    return np.array(pos), np.array(xyz)

def dist_lookup(pos, xyz, p1, p2):
    idx = {p: i for i, p in enumerate(pos)}
    def binof(p): return (p - 3_000_000) // BIN * BIN + 3_000_000
    out = np.full(len(p1), np.nan)
    for k in range(len(p1)):
        b1, b2 = binof(p1[k]), binof(p2[k])
        i, j = idx.get(b1), idx.get(b2)
        if i is None or j is None: continue
        out[k] = np.linalg.norm(xyz[i] - xyz[j])
    return out

rng = np.random.default_rng(7)
truth = np.array([r[2] for r in recs])
p1 = np.array([r[0] for r in recs]); p2 = np.array([r[1] for r in recs])

print("%6s %8s %8s %8s %8s" % ("noise", "acc", "accA", "accB", "n_nan"))
for f in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    flip = rng.random(len(truth)) < f
    init = np.where(flip, 1 - truth, truth)
    rowsA = [(p1[k], p2[k]) for k in range(len(recs)) if init[k] == 0]
    rowsB = [(p1[k], p2[k]) for k in range(len(recs)) if init[k] == 1]
    write_pairs(WORK + "/A.pairs.gz", rowsA)
    write_pairs(WORK + "/B.pairs.gz", rowsB)
    run_fdg(WORK + "/A.pairs.gz", WORK + "/A.3dg")
    run_fdg(WORK + "/B.pairs.gz", WORK + "/B.3dg")
    pA, xA = read_3dg(WORK + "/A.3dg"); pB, xB = read_3dg(WORK + "/B.3dg")
    print("  structure A beads=%d  B beads=%d" % (len(pA), len(pB)))
    dA = dist_lookup(pA, xA, p1, p2); dB = dist_lookup(pB, xB, p1, p2)
    ok = np.isfinite(dA) & np.isfinite(dB)
    pred = np.where(dA <= dB, 0, 1)
    acc = float(np.mean(pred[ok] == truth[ok]))
    accA = float(np.mean(pred[ok & (truth == 0)] == 0))
    accB = float(np.mean(pred[ok & (truth == 1)] == 1))
    print("%6.2f %8.4f %8.4f %8.4f %8d" % (f, acc, accA, accB, int((~ok).sum())))
