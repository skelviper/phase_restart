"""GATE 0：双结构二倍体模型在预测上是否优于单结构？

无标签评分：在一半训练 contacts 上拟合距离衰减 contact model，然后用以下模型对留出 contacts 评分：(a) 一个共识结构，(b) 两个结构的 50/50 mixture。对照：从 contacts 的随机拆分得到的两个结构。
如果随机拆分结构的增益与真实（oracle）拆分一样大，则增益只来自额外灵活性，而不是单倍型信息。
"""
import gzip, os, subprocess
import numpy as np
from scipy.optimize import minimize

HICKIT = "/work/phase3/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000; CHROM = "chr1"; CLEN = 195471971
WORK = "scratch/pilot_gate0"; os.makedirs(WORK, exist_ok=True)
NA, NB = "C1a", "C1b"

p1, p2, lab = [], [], []
with gzip.open("data/P9016.pairs.gz", "rt") as f:
    for line in f:
        if line[0] == "#": continue
        c = line.rstrip("\n").split("\t")
        if c[1] != CHROM or c[3] != CHROM or int(c[2]) > int(c[4]): continue
        p1.append(int(c[2])); p2.append(int(c[4]))
        ph = c[7] + c[8]
        lab.append(0 if ph == "00" else (1 if ph == "11" else -1))
p1 = np.array(p1); p2 = np.array(p2); lab = np.array(lab); N = len(p1)
binof = lambda p: (p - OFF)//BIN*BIN + OFF
B1, B2 = binof(p1), binof(p2)
B1i = (p1 - OFF)//BIN; B2i = (p2 - OFF)//BIN

def write(assign, tag):
    path = WORK + "/%s.pairs.gz" % tag
    names = {}
    with gzip.open(path, "wt") as f:
        f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
        if np.isscalar(assign) or assign.max() <= 0:
            f.write("#chromosome: %s %d\n" % (CHROM, CLEN))
            f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
            for k in range(N):
                f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (CHROM, p1[k], CHROM, p2[k]))
        else:
            f.write("#chromosome: %s %d\n#chromosome: %s %d\n" % (NA, CLEN, NB, CLEN))
            f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
            for k in range(N):
                nm = NA if assign[k] == 0 else NB
                f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, p1[k], nm, p2[k]))
    out = WORK + "/%s.3dg" % tag
    r = subprocess.run([HICKIT, "-i", path, "-P1", "-b1m", "-O", out], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-600:]
    d = {}
    with open(out) as f:
        for line in f:
            if line[0] == "#": continue
            a = line.split()
            if len(a) < 5: continue
            d.setdefault(a[0], {})[int(a[1])] = np.array([float(a[2]), float(a[3]), float(a[4])])
    return d

def dists(d, key):
    m = d[key]; out = np.full(N, np.nan)
    for k in range(N):
        if B1[k] in m and B2[k] in m: out[k] = np.linalg.norm(m[B1[k]] - m[B2[k]])
    return out

# consensus（单结构）
dc = write(np.zeros(N), "cons")
d1 = dists(dc, CHROM)
# oracle 拆分
do = write(np.where(lab == 1, 1, 0), "oracle")
dA = dists(do, NA); dB = dists(do, NB)
# random split 对照（大小相同）
rng = np.random.default_rng(5)
ra = rng.integers(0, 2, N)
dr = write(ra, "rand")
dA2 = dists(dr, NA); dB2 = dists(dr, NB)
print("nan counts: cons=%d oracle=%d rand=%d" % (np.isnan(d1).sum(), np.isnan(dA).sum(), np.isnan(dA2).sum()))

ok = np.isfinite(d1) & np.isfinite(dA) & np.isfinite(dB) & np.isfinite(dA2) & np.isfinite(dB2)
# 将评分限制在有信息的 contacts 上（两个结构都已定义）
key = ((B1i * 131 + B2i) * 2654435761) % 1000
tr = ok & (key < 500); te = ok & (key >= 500)
print("train %d test %d" % (tr.sum(), te.sum()))

# 按 fold 聚合每个 bin pair 的 counts
def agg(mask, dvec):
    n = {}
    for k in np.where(mask)[0]:
        kk = (B1[k], B2[k])
        n.setdefault(kk, [0, 0.0, 0.0, 0.0, 0])
        v = n[kk]; v[0] += 1
    return n
# 更简单：按 contact 使用 Poisson，lambda = c * f(d)（contacts 是独立抽样）
def nll(params, ds, counts, alpha_free=True):
    c = np.exp(params[0]); 
    if alpha_free: alpha = np.exp(params[1])
    else: alpha = params[1]
    lam = c * np.power(np.maximum(ds, 1e-3), -alpha)
    return -(np.sum(counts * np.log(lam + 1e-12) - lam))

def fit1(ds, counts, x0=(0.0, 1.0)):
    r = minimize(nll, np.array(x0), args=(ds, counts, True), method="Nelder-Mead",
                 options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
    return r.x, -r.fun

def nll2(params, dA_, dB_, counts):
    c = np.exp(params[0]); alpha = np.exp(params[1]); w = 1.0/(1.0+np.exp(-params[2]))
    lam = c * (w * np.power(np.maximum(dA_, 1e-3), -alpha) + (1 - w) * np.power(np.maximum(dB_, 1e-3), -alpha))
    return -(np.sum(counts * np.log(lam + 1e-12) - lam))

def fit2(dA_, dB_, counts, x0=(0.0, 1.0, 0.0)):
    r = minimize(nll2, np.array(x0), args=(dA_, dB_, counts), method="Nelder-Mead",
                 options=dict(maxiter=8000, xatol=1e-6, fatol=1e-8))
    return r.x, -r.fun

cnt = np.ones(len(p1))
res = {}
for name, ds in (("one_consensus", d1), ("two_random", (dA2, dB2)), ("two_oracle", (dA, dB))):
    if name == "one_consensus":
        x, f = fit1(ds[tr], cnt[tr])
        ll_te = -nll(x, ds[te], cnt[te], True)
        k = 2
    else:
        x, f = fit2(ds[0][tr], ds[1][tr], cnt[tr])
        ll_te = -nll2(x, ds[0][te], ds[1][te], cnt[te])
        k = 3
    res[name] = (f, ll_te, k)
    print("%-14s train_LL=%10.1f  test_LL=%10.1f  test_LL/contact=%+.5f  k=%d" % (
        name, f, ll_te, ll_te/te.sum(), k))
o = res["one_consensus"]; rnd = res["two_random"]; orc = res["two_oracle"]
print()
print("delta test LL: two_oracle - one_consensus = %+.1f  (%+.5f/contact, %d params)" % (
    orc[1]-o[1], (orc[1]-o[1])/te.sum(), orc[2]-o[2]))
print("delta test LL: two_random - one_consensus = %+.1f  (%+.5f/contact)" % (rnd[1]-o[1], (rnd[1]-o[1])/te.sum()))
print("delta test LL: two_oracle - two_random     = %+.1f  (%+.5f/contact)" % (orc[1]-rnd[1], (orc[1]-rnd[1])/te.sum()))
print("AIC delta (oracle vs one): %+.1f" % (2*(o[2]-orc[2]) + 2*(orc[1]-o[1])))
print("AIC delta (random vs one): %+.1f" % (2*(o[2]-rnd[2]) + 2*(rnd[1]-o[1])))