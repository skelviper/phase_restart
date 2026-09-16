"""2.2' 步骤 A：共识几何的 sign map 是否至少可用？

X = Z + u、Y = Z - u 产生的 assignment 只依赖 sign((u_i - u_j).vhat_ij)，不依赖幅度。因此，即使 FDG 共识无法表示真实结构，由它构建的 sign map 仍可能产生可用的拆分。先测量这一点，再进行搜索。

另外，截断到 S^T S 的前 K 个非刚性特征向量后，真实 half-difference field 是否仍能给出良好拆分？
"""
import gzip, os, subprocess
import numpy as np
from scipy.stats import spearmanr
HICKIT = "native/hickit/hickit"
NB_ = 193; BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"
NC, NA, NBB = "c1c", "c1a", "c1b"
WORK = "scratch/spec3"; os.makedirs(WORK, exist_ok=True)

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
keep = (kb < kc) & (kb >= 0) & (kc < NB_)          # 去掉同 bin contacts：没有距离信息
p1, p2, lb, kb, kc = p1[keep], p2[keep], lb[keep], kb[keep], kc[keep]
N = len(p1); print("contacts after dropping same-bin: %d" % N)

d = {}
with open(WORK + "/joint.3dg") as f:
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
u_true = (Ga - Gb).reshape(-1) / 2.0

cnt = np.zeros((NB_, NB_))
for i in range(N): cnt[kb[i], kc[i]] += 1
cnt = cnt + cnt.T
kk, ll = np.where(np.triu(cnt > 0, 1))
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

# 将每个 contact 映射到 measured list 中对应的 pair index
pair_index = {}
for r_ in range(len(kk)): pair_index[(int(kk[r_]), int(ll[r_]))] = r_
cidx = np.array([pair_index[(int(kb[i]), int(kc[i]))] for i in range(N)])

def sign_of(u):
    return (S @ u)[cidx] > 0

true_sign = lb == 0
sp_true = sign_of(u_true)
print("=== diagnostic: sign map from the consensus geometry ===")
print("u = u_true            : per-contact sign agreement = %.4f  (chance 0.500)" % np.mean(sp_true == true_sign))
pm = np.array([np.mean(sp_true[cidx == r_] == true_sign[cidx == r_]) for r_ in range(len(kk))])
print("u = u_true            : per-bin-pair majority agreement = %.4f" % np.mean((pm > 0.5) == (
    np.array([np.mean(true_sign[cidx == r_] == (lb[cidx == r_] == 0)) for r_ in range(len(kk))]) > 0.5)))
for K in [5, 10, 20, 40, 80]:
    G = proj(S.T @ S)
    ev, evec = np.linalg.eigh(G)
    Bk = evec[:, -K:]
    coef = Bk.T @ u_true
    u_trunc = Bk @ coef
    sp_k = sign_of(u_trunc)
    print("u_true truncated to K=%-3d: sign agreement = %.4f ; captured energy = %.3f" % (
        K, np.mean(sp_k == true_sign), np.linalg.norm(coef)**2 / np.linalg.norm(u_true)**2))
rng = np.random.default_rng(0)
ur = proj(rng.normal(size=3*NB_))
print("u = random            : sign agreement = %.4f" % np.mean(sign_of(ur) == true_sign))
np.save(WORK + "/u_true.npy", u_true); np.save(WORK + "/kk.npy", kk); np.save(WORK + "/ll.npy", ll)