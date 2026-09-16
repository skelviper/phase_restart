"""修正后的无 SNP benchmark（v1）。

修复评审指出的四个问题：
  1. 坐标按数值比较，而不是按字符串比较
  2. train/val/test 按数值 bin pair 的混合哈希拆分，而不是按 (i+j) 奇偶拆分，避免与基因组间距混杂，也不会把所有同区间接触推入同一折
  3. 距离指数在 VALIDATION 折上选择，绝不使用 test 折
  4. 同区间（零距离）pair 从结构指标中排除并单独报告，因为零距离不携带结构信息

结构只在 TRAIN 折上拟合，包含三类 predictor：
  oracle   - 按真实 phase labels 拆分（阳性对照，不是盲方法）
  random   - 随机 50/50 拆分（控制两套结构带来的额外灵活性）
  consensus- 单一结构
"""
import gzip, os, subprocess, sys
import numpy as np
from scipy.stats import spearmanr

HICKIT = "native/hickit/hickit"
BIN = 1_000_000; OFF = 3_000_000
WORK = "scratch/bench_v1"; os.makedirs(WORK, exist_ok=True)
CHRS = ["chr%d" % k for k in range(1, 20)] + ["chrX"]
SAFE = {c: ("cc%02da" % i, "cc%02db" % i) for i, c in enumerate(CHRS)}

def fold_of(i, j):
    """根据数值 bin pair 生成确定性的 0..9 fold；不依赖 i+j。"""
    h = (np.uint64(i) * np.uint64(0x9E3779B1)) ^ (np.uint64(j) * np.uint64(0x85EBCA77))
    h ^= np.uint64(i * j + 0x165667B1)
    h = np.uint64(h) ^ (np.uint64(h) >> np.uint64(15))
    h = np.uint64(h) * np.uint64(0x2545F491)
    h = np.uint64(h) ^ (np.uint64(h) >> np.uint64(13))
    return int(h % np.uint64(10))

def load(chrom):
    p1, p2, lb = [], [], []
    with gzip.open("data/P9016.pairs.gz", "rt") as f:
        for line in f:
            if line[0] == "#": continue
            c = line.rstrip("\n").split("\t")
            if c[1] != c[3] or c[1] != chrom: continue
            a, b = int(c[2]), int(c[4])          # FIX 1：数值比较
            if a > b: a, b = b, a                 # 上三角
            p1.append(a); p2.append(b)
            ph = c[7] + c[8]
            lb.append(0 if ph == "00" else (1 if ph == "11" else -1))
    return np.array(p1), np.array(p2), np.array(lb)

def run_fdg(tag, rows, chrom, split):
    na, nb = SAFE[chrom]
    path = WORK + "/%s.pairs.gz" % tag
    with gzip.open(path, "wt") as f:
        f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
        for nm in ((chrom,) if not split else (na, nb)):
            f.write("#chromosome: %s %d\n" % (nm, 300000000))
        f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
        for a, b, asg in rows:
            nm = chrom if not split else (na if asg == 0 else nb)
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

def pair_dist(d, key, KA, KB):
    m = d.get(key, {})
    if KA in m and KB in m: return float(np.linalg.norm(m[KA] - m[KB]))
    return np.nan

def analyse(chrom, seed=0):
    p1, p2, lb = load(chrom)
    I1 = (p1 - OFF)//BIN; I2 = (p2 - OFF)//BIN
    folds = np.array([fold_of(int(i), int(j)) for i, j in zip(I1, I2)])
    trk = np.where(folds < 6)[0]; vak = np.where((folds >= 6) & (folds < 8))[0]; tek = np.where(folds >= 8)[0]
    # 用于构建 oracle 和报告的两端均 phased contacts
    ph_m = lb >= 0
    orc = np.where(ph_m & (folds < 6))[0]
    rows_or = [(int(p1[i]), int(p2[i]), int(lb[i])) for i in orc]
    rows_rn = [(int(p1[i]), int(p2[i]), 0) for i in trk]
    if len(rows_or) < 500: return None
    dO = run_fdg("o_%s" % chrom, rows_or, chrom, True)
    na, nb = SAFE[chrom]
    rng = np.random.default_rng(seed)
    rn_asg = rng.integers(0, 2, len(trk))
    dR = run_fdg("r_%s_%d" % (chrom, seed), [(int(p1[i]), int(p2[i]), int(rn_asg[k]))
                                            for k, i in enumerate(trk)], chrom, True)
    dC = run_fdg("c_%s" % chrom, [(int(p1[i]), int(p2[i]), 0) for i in trk], chrom, False)

    # 结构指标的候选全集：该 fold 中至少有 1 个 contact 的 bin pairs，且 i != j
    def pairs_of(idx):
        s = {}
        for i in idx:
            if I1[i] == I2[i]: continue
            s[(int(I1[i]), int(I2[i]))] = s.get((int(I1[i]), int(I2[i])), 0) + 1
        return s
    def score(fold_idx, alpha):
        s = pairs_of(fold_idx)
        K = np.array(sorted(s)); C = np.array([s[tuple(k)] for k in K], float)
        A = np.array([pair_dist(dO, na, OFF + k[0]*BIN, OFF + k[1]*BIN) for k in K])
        B = np.array([pair_dist(dO, nb, OFF + k[0]*BIN, OFF + k[1]*BIN) for k in K])
        R1 = np.array([pair_dist(dR, na, OFF + k[0]*BIN, OFF + k[1]*BIN) for k in K])
        R2 = np.array([pair_dist(dR, nb, OFF + k[0]*BIN, OFF + k[1]*BIN) for k in K])
        Z = np.array([pair_dist(dC, chrom, OFF + k[0]*BIN, OFF + k[1]*BIN) for k in K])
        m = np.isfinite(A) & np.isfinite(B) & np.isfinite(R1) & np.isfinite(R2) & np.isfinite(Z)
        return K, C, A, B, R1, R2, Z, m
    KO, CO, AO, BO, RO1, RO2, ZO, mO = score(vak, None)
    # FIX 3：在 validation 上选择 alpha
    AL = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    okv = mO
    a_or = max(AL, key=lambda a: spearmanr(np.maximum(AO[okv], 1e-3)**-a + np.maximum(BO[okv], 1e-3)**-a, CO[okv])[0])
    a_rn = max(AL, key=lambda a: spearmanr(np.maximum(RO1[okv], 1e-3)**-a + np.maximum(RO2[okv], 1e-3)**-a, CO[okv])[0])
    a_cs = max(AL, key=lambda a: spearmanr(np.maximum(ZO[okv], 1e-3)**-a, CO[okv])[0])
    KT, CT, AT, BT, RT1, RT2, ZT, mT = score(tek, None)
    if mT.sum() < 100: return None
    so = spearmanr(np.maximum(AT[mT], 1e-3)**-a_or + np.maximum(BT[mT], 1e-3)**-a_or, CT[mT])[0]
    sr = spearmanr(np.maximum(RT1[mT], 1e-3)**-a_rn + np.maximum(RT2[mT], 1e-3)**-a_rn, CT[mT])[0]
    sc = spearmanr(np.maximum(ZT[mT], 1e-3)**-a_cs, CT[mT])[0]
    # 对 test bin pairs 的配对差值进行 bootstrap
    idx = np.where(mT)[0]; rng2 = np.random.default_rng(seed + 99)
    diffs = []
    po = np.maximum(AT, 1e-3)**-a_or + np.maximum(BT, 1e-3)**-a_or
    pr = np.maximum(RT1, 1e-3)**-a_rn + np.maximum(RT2, 1e-3)**-a_rn
    for _ in range(2000):
        b = rng2.choice(idx, len(idx), replace=True)
        if np.std(po[b]) == 0 or np.std(pr[b]) == 0 or np.std(CT[b]) == 0: continue
        diffs.append(spearmanr(po[b], CT[b])[0] - spearmanr(pr[b], CT[b])[0])
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))
    n_samebin_tr = int(np.sum(I1[trk] == I2[trk]))
    return dict(chrom=chrom, n=len(p1), n_tr=len(trk), n_va=len(vak), n_te=len(tek),
                n_samebin_tr=n_samebin_tr, pairs_te=int(mT.sum()),
                oracle=so, random=sr, consensus=sc, diff=so - sr, lo=lo, hi=hi,
                a_or=a_or, a_rn=a_rn, a_cs=a_cs)

CH = [c for c in sys.argv[1:]] or CHRS
res = []
for c in CH:
    r = analyse(c)
    if r is None: print("%-5s skipped" % c); continue
    res.append(r)
    print("%-5s pairs=%5d cts=%6d | same-bin in train %5d | oracle %.4f random %.4f cons %.4f | diff %+.4f [%+.4f,%+.4f]"
          % (c, r["pairs_te"], -1, r["n_samebin_tr"], r["oracle"], r["random"], r["consensus"], r["diff"], r["lo"], r["hi"]))
    sys.stdout.flush()
if res:
    d = np.array([r["diff"] for r in res])
    print()
    print("=== %d chromosomes ===" % len(res))
    print("oracle %.4f | random %.4f | consensus %.4f" % (
        np.mean([r["oracle"] for r in res]), np.mean([r["random"] for r in res]), np.mean([r["consensus"] for r in res])))
    print("oracle - random: mean %+.4f  sd %.4f  wins %d/%d" % (d.mean(), d.std(ddof=1), int((d > 0).sum()), len(d)))
