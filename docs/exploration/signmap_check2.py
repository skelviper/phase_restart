"""sign map 出问题了，还是只有共识几何有问题？

计算以下两种情况下的 sign agreement：(a) 来自真实 midpoint 的 vhat；(b) 来自干净拟合的 consensus，并通过 Procrustes 对齐到真实坐标系。
"""
import gzip, os, subprocess
import numpy as np
HICKIT = "native/hickit/hickit"
NB_ = 193; BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"
WORK = "scratch/sign4"; os.makedirs(WORK, exist_ok=True)

p1, p2, lb = [], [], []
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != c[3] or c[1] != CHROM or int(c[2]) > int(c[4]): continue
        ph = c[7] + c[8]
        if ph not in ("00", "11"): continue
        p1.append(int(c[2])); p2.append(int(c[4])); lb.append(0 if ph == "00" else 1)
p1 = np.array(p1); p2 = np.array(p2); lb = np.array(lb)
kb = (p1 - OFF)//BIN; kc = (p2 - OFF)//BIN
keep = (kb < kc) & (kb >= 0) & (kc < NB_)
p1, p2, lb, kb, kc = p1[keep], p2[keep], lb[keep], kb[keep], kc[keep]
N = len(p1); print("contacts: %d" % N)

def write(tag, triples):
    path = WORK + "/%s.pairs.gz" % tag
    with gzip.open(path, "wt") as f:
        f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
        for nm in sorted(set(t[2] for t in triples)):
            f.write("#chromosome: %s %d\n" % (nm, 300000000))
        f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
        for a, b, nm in triples:
            f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, a, nm, b))
    out = WORK + "/%s.3dg" % tag
    r = subprocess.run([HICKIT, "-i", path, "-P1", "-b1m", "-O", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    d = {}
    with open(out) as f:
        for line in f:
            if line[0] == "#": continue
            a = line.split()
            if len(a) < 5: continue
            d.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
    return d

def grid(d, key):
    st = np.array(sorted(d[key])); pos = np.array([d[key][p] for p in st])
    out = np.full((NB_, 3), np.nan)
    for i in range(NB_):
        j = np.searchsorted(st, OFF + i*BIN, side="right") - 1
        if j >= 0: out[i] = pos[j]
    return out

# truth copies：在干净的 two-copy run 中拟合；consensus 在其独立 run 中拟合
dt = write("truth", [(int(p1[i]), int(p2[i]), "c1a" if lb[i] == 0 else "c1b") for i in range(N)])
Ga, Gb = grid(dt, "c1a"), grid(dt, "c1b")
Gz = grid(write("cons", [(int(p1[i]), int(p2[i]), "c1z") for i in range(N)]), "c1z")
Gmid = (Ga + Gb) / 2.0

def procrustes(A, B):
    """将 A 刚性映射到 B。"""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S_, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca
R, t = procrustes(Gz, Gmid)
Gz_al = (R @ Gz.T).T + t
print("Procrustes: rms residual after align = %.4f ; Rg(Z)=%.3f Rg(mid)=%.3f" % (
    np.sqrt((((R @ Gz.T).T + t - Gmid)**2).sum(1).mean()),
    np.sqrt(((Gz - Gz.mean(0))**2).sum(1).mean()), np.sqrt(((Gmid - Gmid.mean(0))**2).sum(1).mean())))

u_true = (Ga - Gb) / 2.0
delta_true = np.linalg.norm(Ga[:, None] - Ga[None, :], axis=-1) - np.linalg.norm(Gb[:, None] - Gb[None, :], axis=-1)

pair_index = {}
cnt = np.zeros((NB_, NB_))
for i in range(N): cnt[kb[i], kc[i]] += 1
kk, ll = np.where(cnt > 0)
for r_ in range(len(kk)): pair_index[(int(kk[r_]), int(ll[r_]))] = r_
cidx = np.array([pair_index[(int(kb[i]), int(kc[i]))] for i in range(N)])
true_sign = lb == 0

def agree(Gbase):
    v = Gbase[kk] - Gbase[ll]
    dv = (u_true[kk] - u_true[ll])
    s = (dv * v).sum(1)
    return float(np.mean((s[cidx] > 0) == true_sign)), s

for name, Gb_ in (("true midpoint", Gmid), ("consensus (raw frame)", Gz), ("consensus (Procrustes-aligned)", Gz_al)):
    a, s = agree(Gb_)
    dtrue = delta_true[kk, ll]
    print("%-32s sign agreement = %.4f ; corr(sign-score, delta_true) = %+.4f" % (
        name, a, np.corrcoef(s, dtrue)[0, 1]))
print()
print("ceiling: sign(delta_true) vs the true label = %.4f" % np.mean((delta_true[kk, ll][cidx] > 0) == true_sign))
