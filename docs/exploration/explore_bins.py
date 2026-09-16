import gzip, collections
import numpy as np

path = "data/P9016.pairs.gz"
BIN = 1_000_000
counts = collections.Counter()
nbins = collections.Counter()
minpos = {}
maxpos = {}
with gzip.open(path, "rt") as f:
    for line in f:
        if line[0] == "#":
            continue
        c = line.rstrip("\n").split("\t")
        c1, p1, c2, p2 = c[1], int(c[2]), c[3], int(c[4])
        b1, b2 = (p1 - 3_000_000) // BIN, (p2 - 3_000_000) // BIN
        if c1 == c2:
            counts[(c1, b1, b2)] += 1
            nb = max(b1, b2)
            if nb + 1 > nbins[c1]:
                nbins[c1] = nb + 1
        for cc, pp in ((c1, p1), (c2, p2)):
            if pp < minpos.get(cc, 1 << 60): minpos[cc] = pp
            if pp > maxpos.get(cc, 0): maxpos[cc] = pp
print("bins per chrom:", dict(sorted(nbins.items())))
print("min pos:", {k: minpos[k] for k in sorted(minpos)})
print("max pos:", {k: maxpos[k] for k in sorted(maxpos)})
print("total intra bin-pairs:", len(counts), "contacts:", sum(counts.values()))
n = nbins["chr1"]
M = np.zeros((n, n))
for (c, i, j), v in counts.items():
    if c == "chr1":
        M[i, j] += v
        M[j, i] += v
print("chr1 bins:", n, "chr1 contacts:", M.sum() / 2)
up = M[np.triu_indices(n, 1)]
print("chr1 nonzero frac (upper):", float(np.mean(up > 0)))
print("chr1 diag sums d=0..9:", [int(M[np.arange(n - d), np.arange(d, n)].sum()) for d in range(10)])
print("upper count: max", up.max(), "mean", round(float(up.mean()), 2), "median", float(np.median(up)))
np.save("/tmp/chr1_M.npy", M)
