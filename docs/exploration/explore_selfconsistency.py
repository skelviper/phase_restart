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
        rows.setdefault(c[1], []).append(((int(c[2]) - OFF)//BIN, (int(c[4]) - OFF)//BIN, c[7] + c[8]))

def mat(rs, n_):
    M = np.zeros((n_, n_))
    for i, j, p in rs:
        M[i, j] += 1; M[j, i] += 1
    return M

sp = lambda a, b: spearmanr(np.log(a + 0.5), np.log(b + 1))[0]
print("%-6s %5s %7s | %8s %8s | %8s %8s | %8s %8s" % ("chr","nbins","ncts","M00~Dmat","M00~Dpat","M11~Dmat","M11~Dpat","Mall~Dmat","Mall~Dpat"))
for c in ["chr1","chr2","chr3","chr4","chr5","chr7","chrX"]:
    mk, pk = c + "(mat)", c + "(pat)"
    if mk not in coords: continue
    allrows = rows[c]
    n = max(max(r[0], r[1]) for r in allrows) + 1
    M   = mat(allrows, n)
    M00 = mat([r for r in allrows if r[2] == "00"], n)
    M11 = mat([r for r in allrows if r[2] == "11"], n)
    keep = [i for i in range(n) if (i*BIN + OFF) in coords[mk] and (i*BIN + OFF) in coords[pk]]
    X = np.array([coords[mk][i*BIN+OFF] for i in keep]); Y = np.array([coords[pk][i*BIN+OFF] for i in keep])
    Dm = np.linalg.norm(X[:,None]-X[None,:], axis=-1); Dp = np.linalg.norm(Y[:,None]-Y[None,:], axis=-1)
    iu = np.triu_indices(len(keep), 1)
    dm, dp = Dm[iu], Dp[iu]
    f = lambda M_: M_[np.ix_(keep, keep)][iu]
    m00, m11, mall = f(M00), f(M11), f(M)
    print("%-6s %5d %7d | %8.3f %8.3f | %8.3f %8.3f | %8.3f %8.3f" % (
        c, len(keep), int(mall.sum()), sp(m00,dm), sp(m00,dp), sp(m11,dm), sp(m11,dp), sp(mall,dm), sp(mall,dp)))
