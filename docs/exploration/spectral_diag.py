"""用阳性对照诊断 phase-retrieval 步骤。

h_true = 来自真实结构的 delta*^2（无噪声上界）
h_est  = 共识残差估计（实际使用的量）
h_rand = 打乱后的 h_est（对照）
"""
import gzip, os, subprocess
import numpy as np
from scipy.stats import spearmanr
HICKIT = "native/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"; NA, NB = "c1a", "c1b"
WORK = "scratch/spec2"; os.makedirs(WORK, exist_ok=True)
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

def run(tag, rows, split=True):
    path = WORK + "/%s.pairs.gz" % tag
    with gzip.open(path, "wt") as f:
        f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
        for nm in ((CHROM,) if not split else (NA, NB)):
            f.write("#chromosome: %s %d\n" % (nm, 300000000))
        f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
        for p, q, a in rows:
            nm = CHROM if not split else (NA if a == 0 else NB)
            f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, p, nm, q))
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

def to_grid(d, key):
    starts = np.array(sorted(d[key])); pos = np.array([d[key][p] for p in starts])
    out = np.full((NB_, 3), np.nan)
    for i in range(NB_):
        j = np.searchsorted(starts, OFF + i*BIN, side="right") - 1
        if j >= 0: out[i] = pos[j]
    return out

ALL = np.arange(N)
Gz = to_grid(run("cons", [(int(p1[i]), int(p2[i]), 0) for i in ALL], False), CHROM)
Gt = run("truth", [(int(p1[i]), int(p2[i]), int(lb[i])) for i in ALL])
Ga, Gb = to_grid(Gt, NA), to_grid(Gt, NB)
dm = lambda G: np.linalg.norm(G[:, None, :] - G[None, :, :], axis=-1)
DZ, DA, DB = dm(Gz), dm(Ga), dm(Gb)
iu = np.triu_indices(NB_, 1)
sp = lambda a, b: spearmanr(a, b)[0]
TRUTH = (1 - sp(DA[iu], DB[iu]))
print("truth contrast (1 - rho(DA,DB)) = %+.4f" % TRUTH)

cnt = np.zeros((NB_, NB_))
for i in ALL: cnt[kb[i], kc[i]] += 1
cnt = cnt + cnt.T
sel = np.triu(cnt > 0, 1)
kk, ll = np.where(sel)
dbar = DZ[kk, ll]
delta_true = (DA[kk, ll] - DB[kk, ll]) / 2.0
print("measured pairs: %d ; delta_true: mean|.|=%.4f rms=%.4f" % (len(kk), np.abs(delta_true).mean(), delta_true.std()))

V = Gz[kk] - Gz[ll]
Vh = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
S = np.zeros((len(kk), 3*NB_)); row = np.arange(len(kk))
for c in range(3):
    S[row, 3*kk + c] += Vh[:, c]; S[row, 3*ll + c] -= Vh[:, c]
G = S.T @ S
print("self-check  ||S u_true_grid - delta_true|| / ||delta_true|| = %.4f" % (
    np.linalg.norm(S @ (Ga - Gb).reshape(-1)/2 - delta_true) / np.linalg.norm(delta_true)))

# 刚性投影
idx = np.arange(NB_)
rig = []
for c in range(3):
    t = np.zeros((3*NB_, 3)); t[3*idx + c, c] = 1.0; rig.append(t)
    e = np.zeros(3); e[c] = 1.0
    rot = np.cross(np.tile(e, (NB_, 1)), np.nan_to_num(Gz))
    t = np.zeros((3*NB_, 3)); t[idx[:, None]*3 + np.arange(3)[None, :], c] = rot
    rig.append(t)
B = np.concatenate(rig, 1); Q, _ = np.linalg.qr(B)
proj = lambda v: v - Q @ (Q.T @ v)
Gpinv = np.linalg.pinv(proj(G), rcond=1e-10)

def solve(h, iters=300):
    H = S.T @ (S * h[:, None])
    if not np.isfinite(H).all(): return None, None, "H not finite"
    ev, evec = np.linalg.eigh(H)
    u = proj(evec[:, -1]); u /= np.linalg.norm(u)
    spec = u.copy()
    tgt = np.sqrt(np.maximum(h, 0))
    for it in range(iters):
        s = S @ u
        u = proj(Gpinv @ (S.T @ np.where(s >= 0, tgt, -tgt)))
        u /= max(np.linalg.norm(u), 1e-12)
    return spec, u, None

def contrast_of(u, t):
    if u is None: return float("nan")
    d = (S @ u)
    dA = np.full((NB_, NB_), np.nan); dB = np.full((NB_, NB_), np.nan)
    dA[kk, ll] = dbar + t*d; dB[kk, ll] = dbar - t*d
    dA = dA + dA.T; dB = dB + dB.T
    m = np.isfinite(dA[iu]) & np.isfinite(dB[iu])
    if m.sum() < 200: return float("nan")
    a, b = dA[iu][m], dB[iu][m]
    if np.std(a) == 0 or np.std(b) == 0: return float("nan")
    return 1 - sp(a, b)

h_true = delta_true**2
rng = np.random.default_rng(0)
h_rand = rng.permutation(h_true)
# 基于 residual 的 h（同前）
n_obs = cnt[kk, ll]; gsep = np.abs(kk - ll).astype(float)
feat = np.stack([np.log(gsep + 1), np.log(dbar + 1e-6)], 1)
KNN = 100; nhat = np.zeros(len(kk)); wb = np.zeros(len(kk))
w = None
for a in range(len(kk)):
    dd = ((feat - feat[a])**2).sum(1); ii = np.argpartition(dd, KNN)[:KNN]
    nhat[a] = n_obs[ii].mean()
w = (n_obs - nhat)**2 / np.maximum(nhat, 1e-6)
for a in range(len(kk)):
    dd = ((feat - feat[a])**2).sum(1); ii = np.argpartition(dd, KNN)[:KNN]
    wb[a] = w[ii].mean()
h_est = np.maximum(0.0, w - wb)
print("h_true: rms %.3f ; h_est: rms %.3f, nonzero %d ; h_rand rms %.3f" % (
    np.sqrt((h_true**2).mean()), np.sqrt((h_est**2).mean()), int((h_est>0).sum()), np.sqrt((h_rand**2).mean())))
print()
print("%-10s %-22s %-22s" % ("h source", "spectral contrast", "GS contrast"))
for name, h in (("true", h_true), ("random", h_rand), ("residual", h_est)):
    spec, u, err = solve(h)
    if err: print("%-10s ERROR %s" % (name, err)); continue
    cs = max((contrast_of(spec, t), t) for t in [0.1, 0.3, 0.6, 1.0, 2.0, 4.0])
    cg = max((contrast_of(u, t), t) for t in [0.1, 0.3, 0.6, 1.0, 2.0, 4.0])
    print("%-10s %+.4f (t=%.1f)      %+.4f (t=%.1f)" % (name, cs[0], cs[1], cg[0], cg[1]))
print()
print("truth = %+.4f   (perfect start)" % TRUTH)
print("best eps sweep for residual h:")
spec, u, _ = solve(h_est)
for t in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.0]:
    print("   t=%-5.2f spectral %+.4f   GS %+.4f" % (t, contrast_of(spec, t), contrast_of(u, t)))
