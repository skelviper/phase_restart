"""Spectral / GS 起点，使用正确指标评分。

contrast_matched(u) = mean(rho(dA,DA), rho(dB,DB)) - mean(rho(dA,DB), rho(dB,DA))
  ~ +0.27 表示真实值，~0 表示随机场，u = 0 也约为 ~0。
（此前的 1 - rho(dA,dB) 是错误的：它会奖励任意大的随机扰动。）
"""
import gzip, os, subprocess
import numpy as np
from scipy.stats import spearmanr
NB_ = 193; BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"
NC, NA, NBB = "c1c", "c1a", "c1b"
d = {}
with open("scratch/spec3/joint.3dg") as f:
    for line in f:
        if line[0] == "#": continue
        a = line.split()
        if len(a) < 5: continue
        d.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
def grid(key):
    st = np.array(sorted(d[key])); pos = np.array([d[key][p] for p in st])
    out = np.full((NB_, 3), np.nan)
    for i in range(NB_):
        j = np.searchsorted(st, OFF + i*BIN, side="right") - 1
        if j >= 0: out[i] = pos[j]
    return out
Gz, Ga, Gb = grid(NC), grid(NA), grid(NBB)
dm = lambda G: np.linalg.norm(G[:, None, :] - G[None, :, :], axis=-1)
DA, DB, DZ = dm(Ga), dm(Gb), dm(Gz)
iu = np.triu_indices(NB_, 1)
sp = lambda a, b: spearmanr(a, b)[0]
Vall = Gz[:, None, :] - Gz[None, :, :]
TRUTH = 1 - sp(DA[iu], DB[iu])
print("truth contrast_matched = %+.4f" % TRUTH)

def cm(u, t=1.0):
    U = u.reshape(NB_, 3)
    Ut = U[:, None, :] - U[None, :, :]
    dA = np.linalg.norm(Vall + t*Ut, axis=-1); dB = np.linalg.norm(Vall - t*Ut, axis=-1)
    m = (sp(dA[iu], DA[iu]) + sp(dB[iu], DB[iu])) / 2
    c = (sp(dA[iu], DB[iu]) + sp(dB[iu], DA[iu])) / 2
    return m - c

rng = np.random.default_rng(0)
ur = rng.normal(size=3*NB_); ur *= np.sqrt(((Ga-Gb)**2).mean()/4)/np.sqrt((ur**2).mean())
print("control: u = 0 -> %+.4f ; random u -> %+.4f (t=1)" % (cm(np.zeros(3*NB_)), cm(ur)))
umid = (Ga - Gb).reshape(-1) / 2
print("control: u = true half-difference -> %+.4f (t=1)" % cm(umid))

# measured pairs 与 S 算子
p1, p2 = [], []
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != c[3] or c[1] != CHROM or int(c[2]) > int(c[4]): continue
        ph = c[7] + c[8]
        if ph not in ("00", "11"): continue
        p1.append(int(c[2])); p2.append(int(c[4]))
p1 = np.array(p1); p2 = np.array(p2)
cnt = np.zeros((NB_, NB_))
for i in range(len(p1)): cnt[(p1[i]-OFF)//BIN, (p2[i]-OFF)//BIN] += 1
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
    t = np.zeros((3*NB_, 3)); t[idx[:, None]*3 + np.arange(3)[None, :], c] = rot; rig.append(t)
Q, _ = np.linalg.qr(np.concatenate(rig, 1))
proj = lambda v: v - Q @ (Q.T @ v)
Gpinv = np.linalg.pinv(proj(S.T @ S), rcond=1e-10)

# 基于 residual 的 h
n_obs = cnt[kk, ll]; gsep = np.abs(kk - ll).astype(float)
feat = np.stack([np.log(gsep + 1), np.log(np.maximum(dbar, 1e-6))], 1)
KNN = 100; nhat = np.zeros(len(kk))
for a in range(len(kk)):
    dd = ((feat - feat[a])**2).sum(1); ii = np.argpartition(dd, KNN)[:KNN]; nhat[a] = n_obs[ii].mean()
w = (n_obs - nhat)**2 / np.maximum(nhat, 1e-6); wb = np.zeros(len(kk))
for a in range(len(kk)):
    dd = ((feat - feat[a])**2).sum(1); ii = np.argpartition(dd, KNN)[:KNN]; wb[a] = w[ii].mean()
h_est = np.maximum(0.0, w - wb)
h_true = delta_true**2
h_rand = np.random.default_rng(0).permutation(h_true)

print()
print("matched-minus-swapped contrast, best over t in [0.25,0.5,1,2,4]:")
print("%-8s %-22s %-22s" % ("h source", "spectral", "Gerchberg-Saxton"))
for tag, h in (("h_true", h_true), ("h_est", h_est), ("h_rand", h_rand)):
    H = S.T @ (S * h[:, None])
    ev, evec = np.linalg.eigh(H)
    u = proj(evec[:, -1]); u /= np.linalg.norm(u)
    tgt = np.sqrt(h); uu = u.copy()
    for it in range(300):
        s = S @ uu
        uu = proj(Gpinv @ (S.T @ np.where(s >= 0, tgt, -tgt))); uu /= max(np.linalg.norm(uu), 1e-12)
    bs = max((cm(u, t), t) for t in [0.25, 0.5, 1.0, 2.0, 4.0])
    bg = max((cm(uu, t), t) for t in [0.25, 0.5, 1.0, 2.0, 4.0])
    print("%-8s %+.4f (t=%.2f)      %+.4f (t=%.2f)" % (tag, bs[0], bs[1], bg[0], bg[1]))
print()
print("truth = %+.4f ; target from step 2.1 = +0.10" % TRUTH)
