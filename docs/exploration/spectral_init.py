"""阶段 2.2：单一确定性的 phase-retrieval 起点。

counts 给出 |delta_ij|，而不是 sign(delta_ij)。将 S 写成线性映射
(S u)_ij = (u_i - u_j) . vhat_ij，把半差场 u 映射到差值 delta。然后：

  spectral init   u0 = H = sum_ij h_ij S_ij^T S_ij 的主特征向量
  Gerchberg-Saxton  重复：s = S u；s' = sqrt(h) * sign(s)；u <- pinv(S^T S) S^T s'

其中 h_ij 是根据 consensus residual 对 delta_ij^2 的估计。整个过程是确定性的：一次起点，不使用 multi-start，也不使用 multi-resolution。

contrast 根据在同一 train fold 上拟合的真实结构测量。它对坐标系不变（使用距离矩阵的 rank），因此 Z 与 truth 不需要共享坐标系。
"""
import gzip, os, subprocess, sys
import numpy as np
from scipy.stats import spearmanr

HICKIT = "native/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"; NA, NB = "c1a", "c1b"
WORK = "scratch/spec2"; os.makedirs(WORK, exist_ok=True)

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
NB_ = 193
kb = (p1 - OFF)//BIN; kc = (p2 - OFF)//BIN
hsh = ((kb * 7919 + kc * 104729) % 2).astype(bool)
trk = np.where(hsh)[0]; tek = np.where(~hsh)[0]
print("chr1 phased cis: %d contacts (train %d / test %d)" % (N, len(trk), len(tek)))

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
    """将每个 1 Mb bin（start = OFF + i*BIN）映射到 bead 坐标，或 NaN。"""
    starts = np.array(sorted(d[key])); pos = np.array([d[key][p] for p in starts])
    out = np.full((NB_, 3), np.nan)
    for i in range(NB_):
        b = OFF + i*BIN
        j = np.searchsorted(starts, b, side="right") - 1
        if j >= 0: out[i] = pos[j]
    return out

print("fitting consensus Z and the true split on the train fold ...")
ALL = np.arange(N)
Gz = to_grid(run("cons", [(int(p1[i]), int(p2[i]), 0) for i in ALL], False), CHROM)
Gt = run("truth", [(int(p1[i]), int(p2[i]), int(lb[i])) for i in ALL])
Ga, Gb = to_grid(Gt, NA), to_grid(Gt, NB)
okb = np.isfinite(Gz).all(1) & np.isfinite(Ga).all(1) & np.isfinite(Gb).all(1)
print("bins usable in all three structures: %d / %d" % (okb.sum(), NB_))

def dmat(G):
    return np.linalg.norm(G[:, None, :] - G[None, :, :], axis=-1)

DZ, DA, DB = dmat(Gz), dmat(Ga), dmat(Gb)
iu = np.triu_indices(NB_, 1)
sp = lambda a, b: spearmanr(a, b)[0]
TRUTH = (sp(DA[iu], DA[iu]) + sp(DB[iu], DB[iu]) - sp(DA[iu], DB[iu]) - sp(DB[iu], DA[iu])) / 2
print("reference contrast on this grid: truth = %+.4f" % TRUTH)

# --- 可观测的差分能量 h_ij ~ delta_ij^2，来自 consensus residual
cnt = np.zeros((NB_, NB_))
for i in ALL: cnt[kb[i], kc[i]] += 1
cnt = cnt + cnt.T
sel = np.triu(cnt > 0, 1)
kk, ll = np.where(sel)
n_obs = cnt[kk, ll]
dbar = DZ[kk, ll]
gsep = np.abs(kk - ll).astype(float)
nhat = np.zeros(len(kk))
feat = np.stack([np.log(gsep + 1), np.log(dbar + 1e-6)], 1)
KNN = 100
for a in range(len(kk)):
    dd = ((feat - feat[a])**2).sum(1)
    idx = np.argpartition(dd, KNN)[:KNN]
    nhat[a] = n_obs[idx].mean()
w = (n_obs - nhat)**2 / np.maximum(nhat, 1e-6)
# 用同一 smoother 去除系统性过度离散，再进行截断
wbase = np.zeros(len(kk))
for a in range(len(kk)):
    dd = ((feat - feat[a])**2).sum(1)
    idx = np.argpartition(dd, KNN)[:KNN]
    wbase[a] = w[idx].mean()
h = np.maximum(0.0, w - wbase)
print("h_ij: %d pairs, %d with h>0, mean h = %.3f" % (len(h), int((h > 0).sum()), h.mean()))

V = Gz[kk] - Gz[ll]
Vh = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
S = np.zeros((len(kk), 3*NB_))
row = np.arange(len(kk))
for c in range(3):
    S[row, 3*kk + c] += Vh[:, c]
    S[row, 3*ll + c] -= Vh[:, c]
G = S.T @ S
H = S.T @ (S * h[:, None])

# 刚性运动构成精确的零空间；从任意 field 中投影出去
Zc = Gz[okb]
rig = [np.zeros((3*NB_, 3))]
for c in range(3):
    t = np.zeros((3*NB_, 3)); t[3*np.where(okb)[0] + c, c] = 1.0; rig.append(t)
    w_ = np.zeros(3); w_[c] = 1.0
    rot = np.cross(np.tile(w_, (NB_, 1)), np.nan_to_num(Gz))
    t = np.zeros((3*NB_, 3))
    t[np.where(okb)[0][:, None]*3 + np.arange(3)[None, :], c] = rot[okb]
    rig.append(t)
B = np.concatenate(rig, 1)
Q, _ = np.linalg.qr(B)
proj = lambda v: v - Q @ (Q.T @ v)

ev, evec = np.linalg.eigh(H)
u_spec = proj(evec[:, -1]); u_spec /= np.linalg.norm(u_spec)
Gp = proj(G)
Gpinv = np.linalg.pinv(Gp, rcond=1e-8)

u = u_spec.copy()
target = np.sqrt(h)
hist = []
for it in range(200):
    s = S @ u
    sp_ = np.where(s >= 0, target, -target)
    u = proj(Gpinv @ (S.T @ sp_))
    u /= max(np.linalg.norm(u), 1e-12)
    resid = np.linalg.norm(np.abs(S @ u) - target)
    hist.append(resid)
print("Gerchberg-Saxton residual: start %.4f -> end %.4f" % (hist[0], hist[-1]))

def contrast_of_field(u, eps):
    d = (S @ u).reshape(-1)
    dA = np.full((NB_, NB_), np.nan); dB = np.full((NB_, NB_), np.nan)
    dA[kk, ll] = dbar + eps*d; dB[kk, ll] = dbar - eps*d
    dA = dA + dA.T; dB = dB + dB.T
    m = np.isfinite(dA[iu]) & np.isfinite(dB[iu]) & np.isfinite(DA[iu]) & np.isfinite(DB[iu])
    a, b = dA[iu][m], dB[iu][m]
    return (sp(a, DA[iu][m]) + sp(b, DB[iu][m]) - sp(a, DB[iu][m]) - sp(b, DA[iu][m])) / 2, int(m.sum())

Rg = float(np.sqrt(((Gz[okb] - Gz[okb].mean(0))**2).sum(1).mean()))
print()
print("Rg(Z) = %.3f ; eps is in units of Rg(Z)" % Rg)
print("%8s | %10s %10s %10s" % ("eps/Rg", "spectral", "GS", "random-u ctrl"))
rng = np.random.default_rng(0)
u_rnd = proj(rng.normal(size=3*NB_)); u_rnd /= np.linalg.norm(u_rnd)
best = (0, 0, None)
for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]:
    eps = t * Rg
    cs, _ = contrast_of_field(u_spec, eps)
    cg, _ = contrast_of_field(u, eps)
    cr, _ = contrast_of_field(u_rnd, eps)
    if abs(cg) > abs(best[0]): best = (cg, t, u.copy())
    print("%8.2f | %+10.4f %+10.4f %+10.4f" % (t, cs, cg, cr))
print()
print("truth contrast = %+.4f" % TRUTH)
print("BEST GS contrast = %+.4f at eps/Rg = %.2f  (%.0f%% of truth)" % (best[0], best[1], 100*best[0]/TRUTH))
np.save(WORK + "/u_gs.npy", best[2]); np.save(WORK + "/u_spec.npy", u_spec)