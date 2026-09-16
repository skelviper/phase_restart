import gzip
import numpy as np
from scipy.stats import spearmanr

BIN = 1_000_000; OFF = 3_000_000
coords = {}
with gzip.open("data/P9016.1m.3dg.gz", "rt") as f:
    for line in f:
        a = line.split()
        coords.setdefault(a[0], {})[int(a[1])] = (float(a[2]), float(a[3]), float(a[4]))
rows = {}
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != c[3]: continue
        rows.setdefault(c[1], []).append(((int(c[2])-OFF)//BIN, (int(c[4])-OFF)//BIN, c[7]+c[8]))

def build(rs, n, keys):
    M = np.zeros((n, n))
    for i, j, p in rs:
        if p in keys:
            M[i, j] += 1; M[j, i] += 1
    return M

print("=== A) given the TRUE reference structures: can we classify 00 vs 11 contacts? ===")
print("%-6s %6s %6s | %8s %8s | %8s | %8s" % ("chr","n00","n11","acc(lab0->mat)","acc(lab0->pat)","lr~dd corr","acc_swap_oracle"))
for c in ["chr1","chr2","chr3","chr4","chr5","chr7","chrX"]:
    mk, pk = c + "(mat)", c + "(pat)"
    allrows = rows[c]
    n = max(max(r[0], r[1]) for r in allrows) + 1
    keep = [i for i in range(n) if (i*BIN+OFF) in coords[mk] and (i*BIN+OFF) in coords[pk]]
    X = np.array([coords[mk][i*BIN+OFF] for i in keep]); Y = np.array([coords[pk][i*BIN+OFF] for i in keep])
    Dm = np.linalg.norm(X[:,None]-X[None,:], axis=-1); Dp = np.linalg.norm(Y[:,None]-Y[None,:], axis=-1)
    idx = {b: k for k, b in enumerate(keep)}
    r00 = [(idx[i], idx[j]) for i, j, p in allrows if p == "00" and i in idx and j in idx]
    r11 = [(idx[i], idx[j]) for i, j, p in allrows if p == "11" and i in idx and j in idx]
    # label 0 = mat 的 log-odds score：alpha*log(Dp/Dm)；正值 => 更接近 mat
    lr = np.log(Dp + 0.01) - np.log(Dm + 0.01)
    s00 = np.array([lr[i, j] for i, j in r00]); s11 = np.array([lr[i, j] for i, j in r11])
    acc_a = float(np.mean(np.concatenate([s00 > 0, s11 < 0])))
    acc_b = 1.0 - acc_a
    # point-biserial correlation between sign score 与真实 label
    lab = np.concatenate([np.zeros(len(s00)), np.ones(len(s11))]); sc = np.concatenate([s00, s11])
    rho = spearmanr(sc, lab)[0]
    print("%-6s %6d %6d | %14.3f %14.3f | %14.3f | %14.3f" % (c, len(s00), len(s11), acc_a, acc_b, rho, max(acc_a, acc_b)))
