import gzip, os, subprocess
import numpy as np
from scipy.stats import spearmanr
HICKIT = "native/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"
NC, NA, NB = "c1c", "c1a", "c1b"
WORK = "scratch/spec3"; NB_ = 193
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
d = {}
with open(WORK + "/joint.3dg") as f:
    for line in f:
        if line[0] == "#": continue
        a = line.split()
        if len(a) < 5: continue
        d.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
def to_grid(key):
    starts = np.array(sorted(d[key])); pos = np.array([d[key][p] for p in starts])
    out = np.full((NB_, 3), np.nan)
    for i in range(NB_):
        j = np.searchsorted(starts, OFF + i*BIN, side="right") - 1
        if j >= 0: out[i] = pos[j]
    return out
Gz, Ga, Gb = to_grid(NC), to_grid(NA), to_grid(NB)
dm = lambda G: np.linalg.norm(G[:, None, :] - G[None, :, :], axis=-1)
DZ, DA, DB = dm(Gz), dm(Ga), dm(Gb)
iu = np.triu_indices(NB_, 1)
sp = lambda a, b: spearmanr(a, b)[0]
TRUTH = 1 - sp(DA[iu], DB[iu])
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
idx = np.arange(NB_); rig = []
for c in range(3):
    t = np.zeros((3*NB_, 3)); t[3*idx + c, c] = 1.0; rig.append(t)
    e = np.zeros(3); e[c] = 1.0
    rot = np.cross(np.tile(e, (NB_, 1)), np.nan_to_num(Gz))
    t = np.zeros((3*NB_, 3)); t[idx[:, None]*3 + np.arange(3)[None, :], c] = rot
    rig.append(t)
B = np.concatenate(rig, 1); Q, _ = np.linalg.qr(B)
proj = lambda v: v - Q @ (Q.T @ v)
print("Q orthonormal:", np.allclose(Q.T@Q, np.eye(Q.shape[1]), atol=1e-8), " rank", np.linalg.matrix_rank(B))
G = S.T @ S
Gp = proj(G)
print("G finite", np.isfinite(G).all(), " Gp finite", np.isfinite(Gp).all())
ev = np.linalg.eigvalsh(Gp); print("Gp eigenvalues: min %.3e max %.3e ; #>1e-6: %d" % (ev.min(), ev.max(), (ev > 1e-6).sum()))
Gpinv = np.linalg.pinv(Gp, rcond=1e-10)
print("Gpinv finite", np.isfinite(Gpinv).all(), " max %.3e" % np.abs(Gpinv).max())

def contrast_of(u, t):
    dlt = S @ u
    dA = np.full((NB_, NB_), np.nan); dB = np.full((NB_, NB_), np.nan)
    dA[kk, ll] = dbar + t*dlt; dA[ll, kk] = dbar + t*dlt
    dB[kk, ll] = dbar - t*dlt; dB[ll, kk] = dbar - t*dlt
    m = np.isfinite(dA[iu]) & np.isfinite(dB[iu])
    a, b = dA[iu][m], dB[iu][m]
    if len(a) < 200 or np.std(a) == 0 or np.std(b) == 0: return float("nan"), len(a)
    return 1 - sp(a, b), len(a)

cnt_obs = cnt[kk, ll]
gsep = np.abs(kk - ll).astype(float)
feat = np.stack([np.log(gsep + 1), np.log(np.maximum(dbar, 1e-6))], 1)
KNN = 100
nhat = np.zeros(len(kk))
for a in range(len(kk)):
    dd = ((feat - feat[a])**2).sum(1); ii = np.argpartition(dd, KNN)[:KNN]
    nhat[a] = cnt_obs[ii].mean()
w = (cnt_obs - nhat)**2 / np.maximum(nhat, 1e-6)
wb = np.zeros(len(kk))
for a in range(len(kk)):
    dd = ((feat - feat[a])**2).sum(1); ii = np.argpartition(dd, KNN)[:KNN]
    wb[a] = w[ii].mean()
h_est = np.maximum(0.0, w - wb)
h_rand = np.random.default_rng(0).permutation(delta_true**2)
print("h_true rms %.3f | h_est rms %.3f, nonzero %d | h_rand rms %.3f" % (
    np.sqrt((delta_true**4).mean()), np.sqrt((h_est**2).mean()), int((h_est>0).sum()), np.sqrt(((delta_true**2)**2).mean())))

def run_case(h, tag):
    H = S.T @ (S * h[:, None])
    evh, evec = np.linalg.eigh(H)
    u = proj(evec[:, -1]); u = u/np.linalg.norm(u)
    tgt = np.sqrt(h)
    uu = u.copy()
    for it in range(300):
        s = S @ uu
        uu = proj(Gpinv @ (S.T @ np.where(s >= 0, tgt, -tgt)))
        uu = uu/max(np.linalg.norm(uu), 1e-12)
    print("  %-8s" % tag, end="")
    for t in TS:
        print("  t=%.1f: spec %+.3f GS %+.3f" % (t, contrast_of(u, t)[0], contrast_of(uu, t)[0]), end="")
    print()

TS = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5]
for tag, h in (("h_true", delta_true**2), ("h_est", h_est), ("h_rand", h_rand)):
    run_case(h, tag)
print("truth contrast = %+.4f" % TRUTH)
evh, evec = np.linalg.eigh(H)
u = proj(evec[:, -1]); print("u spec finite", np.isfinite(u).all(), "norm %.3e" % np.linalg.norm(u))
u = u/np.linalg.norm(u)
tgt = np.sqrt(h_true)
uu = u.copy()
for it in range(300):
    s = S @ uu
    uu = proj(Gpinv @ (S.T @ np.where(s >= 0, tgt, -tgt)))
    uu = uu/max(np.linalg.norm(uu), 1e-12)
print("truth contrast = %+.4f" % TRUTH)