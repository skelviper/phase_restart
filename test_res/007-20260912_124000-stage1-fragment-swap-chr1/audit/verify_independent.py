#!/usr/bin/env python
"""运行 `007-20260912_124000-stage1-fragment-swap-chr1` 的独立审计。

这里的一切都由原始输入和自包含代码重新推导。不会导入或执行 `pr/`、`run.py`；读取这些文件仅为了解运行声明的内容及其约定（bin 网格、fold 哈希、NaN 策略、splice 定义）。

原始输入
    ../../data/P9016.pairs.gz            8 列，含 phase0/phase1
    ../../inputs/P9016.snpfree.pairs.gz  7 列，已删除 phase 列
    coords/*.3dg                         运行写出的坐标（已哈希）
    results.json, gate.json              待检验的声明

输出
    audit/audit_numbers.json   此处计算的每个数值，与声明并列保存
    audit/report.md            人类可读的发现（由本脚本写出）

仅使用标准库、numpy 和 scipy。
"""
import gzip
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(RUN))

PAIRS = os.path.join(ROOT, "data", "P9016.pairs.gz")
SNPFREE = os.path.join(ROOT, "inputs", "P9016.snpfree.pairs.gz")
REF3DG = os.path.join(ROOT, "data", "P9016.1m.3dg.gz")
COORDS = os.path.join(RUN, "coords")

CHROM = "chr1"
BIN = 1_000_000
OFF = 3_000_000
FRAG_BINS = 20
ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
TRAIN_FOLDS = (0, 1, 2, 3, 4, 5)
VAL_FOLDS = (6, 7)
TEST_FOLDS = (8, 9)
NBOOT = 2000
BOOT_SEED = 0

TRACK_A = "cc00a"
TRACK_B = "cc00b"

CLAIMS = json.load(open(os.path.join(RUN, "results.json")))
GATE = json.load(open(os.path.join(RUN, "gate.json")))

# --------------------------------------------------------------------------
# 检查记账
# --------------------------------------------------------------------------
CHECKS = []


def check(section, name, computed, claimed, note="", tol=None, agree=None):
    """记录一个 computed-vs-claimed 比较；tol 表示数值接近程度。"""
    if agree is None:
        if tol is not None:
            if computed is None or claimed is None:
                agree = False
            elif isinstance(computed, (int, float)) and isinstance(claimed, (int, float)):
                if np.isnan(computed) and np.isnan(claimed):
                    agree = True
                else:
                    agree = abs(computed - claimed) <= tol
            else:
                agree = computed == claimed
        else:
            agree = computed == claimed
    row = {"section": section, "check": name,
           "computed": (computed.tolist() if isinstance(computed, np.ndarray) else computed),
           "claimed": claimed, "agree": bool(agree), "note": note}
    CHECKS.append(row)
    flag = "AGREE   " if agree else "DISAGREE"
    print("  [%s] %-52s computed=%s claimed=%s %s"
          % (flag, name, _f(computed), _f(claimed), ("| " + note) if note else ""))
    return bool(agree)


def _f(x):
    if isinstance(x, float):
        return "%.10g" % x
    return str(x)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 0. 原始输入
# --------------------------------------------------------------------------
def load_snpfree(chrom):
    """从 7 列 SNP-free 文件读取 (pos1, pos2) int64，并按数值排序。"""
    cols = None
    p1, p2 = [], []
    with gzip.open(SNPFREE, "rt") as f:
        for line in f:
            if line[0] == "#":
                if line.startswith("#columns:"):
                    cols = line.rstrip("\n").split(":", 1)[1].strip().split("\t")
                continue
            if cols is None:
                raise RuntimeError("no #columns header")
            c = line.rstrip("\n").split("\t")
            if len(c) != 7:
                raise RuntimeError("expected 7 columns, got %d" % len(c))
            if c[1] != c[3] or c[1] != chrom:
                continue
            a, b = int(c[2]), int(c[4])
            if a > b:
                a, b = b, a
            p1.append(a)
            p2.append(b)
    return (cols, np.asarray(p1, dtype=np.int64), np.asarray(p2, dtype=np.int64))


def load_phase(chrom):
    """从 8 列 evidence 文件读取 (pos1, pos2, phase-string pair)。"""
    cols = None
    p1, p2, ph = [], [], []
    with gzip.open(PAIRS, "rt") as f:
        for line in f:
            if line[0] == "#":
                if line.startswith("#columns:"):
                    cols = line.rstrip("\n").split(":", 1)[1].strip().split("\t")
                continue
            c = line.rstrip("\n").split("\t")
            if c[1] != c[3] or c[1] != chrom:
                continue
            a, b = int(c[2]), int(c[4])
            if a > b:
                a, b = b, a
            p1.append(a)
            p2.append(b)
            ph.append(c[7] + c[8])
    return (cols, np.asarray(p1, dtype=np.int64), np.asarray(p2, dtype=np.int64),
            np.asarray(ph))


def bin_of(pos):
    return (np.asarray(pos, dtype=np.int64) - OFF) // BIN


def fold_of_array(i, j):
    """使用 uint64 回绕算术，根据 mixed hash 返回 fold id。"""
    i = np.asarray(i, dtype=np.uint64)
    j = np.asarray(j, dtype=np.uint64)
    h = i * np.uint64(0x9E3779B1) ^ j * np.uint64(0x85EBCA77)
    h = h ^ (i * j + np.uint64(0x165667B1))
    h = h ^ (h >> np.uint64(15))
    h = h * np.uint64(0x2545F491)
    h = h ^ (h >> np.uint64(13))
    return (h % np.uint64(10)).astype(np.int64)


# --------------------------------------------------------------------------
# 1. 结构和距离（自有 reader 与 lookup）
# --------------------------------------------------------------------------
_STRUCT_CACHE = {}


def read_3dg(path):
    """返回 {track: {bin_start_pos: xyz}}。"""
    d = {}
    with open(path) as f:
        for line in f:
            if line[0] == "#":
                continue
            a = line.split()
            if len(a) < 5:
                continue
            d.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
    return d


def structures(tag):
    if tag not in _STRUCT_CACHE:
        _STRUCT_CACHE[tag] = read_3dg(os.path.join(COORDS, tag + ".3dg"))
    return _STRUCT_CACHE[tag]


def distances(structs, track, b1, b2):
    """计算每个 bin pair 的距离；缺少 bead 时为 NaN（bins 从 0 开始）。"""
    m = structs.get(track, {})
    out = np.full(len(b1), np.nan)
    cache = {}
    for k, (x, y) in enumerate(zip(b1, b2)):
        pa, pb = OFF + int(x) * BIN, OFF + int(y) * BIN
        if pa not in cache:
            cache[pa] = m.get(pa)
        if pb not in cache:
            cache[pb] = m.get(pb)
        va, vb = cache[pa], cache[pb]
        if va is None or vb is None:
            continue
        out[k] = float(np.linalg.norm(va - vb))
    return out


def spearman(x, y):
    rx = rankdata(x)
    ry = rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def predictor(d, alpha, floor=1e-3):
    return np.maximum(d, floor) ** (-alpha)


def candidate_parts(tag):
    """返回已写出 `tag` 对应 .3dg 中每条轨迹的距离数组列表。"""
    st = structures(tag)
    tracks = [t for t in st if t.startswith("cc")]
    if not tracks:                      # 单轨迹盲 consensus 拟合
        tracks = [t for t in st]
    return st, sorted(tracks)


def pair_distances_for(tag, b1, b2):
    st, tracks = candidate_parts(tag)
    return [distances(st, t, b1, b2) for t in tracks]


def finite_mask(tag, b1, b2):
    parts = pair_distances_for(tag, b1, b2)
    ok = np.ones(len(b1), dtype=bool)
    for p in parts:
        ok &= np.isfinite(p)
    return ok


def pooled_prediction(tag, b1, b2, alpha):
    parts = pair_distances_for(tag, b1, b2)
    pred = np.zeros(len(b1))
    ok = np.ones(len(b1), dtype=bool)
    for p in parts:
        ok &= np.isfinite(p)
        pred = pred + predictor(p, alpha)
    pred[~ok] = np.nan
    return pred, ok


def select_alpha(tag, vb1, vb2, vC):
    parts = pair_distances_for(tag, vb1, vb2)
    ok = np.ones(len(vb1), dtype=bool)
    for p in parts:
        ok &= np.isfinite(p)
    best, best_a = -np.inf, ALPHAS[0]
    for a in ALPHAS:
        pred = np.zeros(int(ok.sum()))
        for p in parts:
            pred = pred + predictor(p[ok], a)
        r = spearman(pred, vC[ok])
        if np.isfinite(r) and r > best:
            best, best_a = r, a
    return best_a, best


# --------------------------------------------------------------------------
# 2. bootstrap（相同 estimator，自有实现）
# --------------------------------------------------------------------------
def paired_bootstrap(C, ref, alt, n_boot=NBOOT, seed=BOOT_SEED):
    n = len(C)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    k = 0
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        r1 = spearman(ref[b], C[b])
        r2 = spearman(alt[b], C[b])
        if np.isfinite(r1) and np.isfinite(r2):
            d[k] = r1 - r2
            k += 1
    if k < 100:
        return float("nan"), float("nan"), 0
    d = d[:k]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi), int(k)


def paired_bootstrap_gaugeavg(C, ra, rb, aa, ab, n_boot=NBOOT, seed=BOOT_SEED):
    n = len(C)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    k = 0
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        r1 = (spearman(ra[b], C[b]) + spearman(rb[b], C[b])) / 2.0
        r2 = (spearman(aa[b], C[b]) + spearman(ab[b], C[b])) / 2.0
        if np.isfinite(r1) and np.isfinite(r2):
            d[k] = r1 - r2
            k += 1
    if k < 100:
        return float("nan"), float("nan"), 0
    d = d[:k]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi), int(k)


# --------------------------------------------------------------------------
# 3. 自有 Procrustes + splice（chimera 评估工具）
# --------------------------------------------------------------------------
def procrustes(src, dst):
    """刚性变换 (R, t)，不缩放，将 src 映射到 dst。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(A)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, mu_d - R @ mu_s


def align(moving, fixed):
    """在共同 beads 上将 `moving`（dict pos->xyz）叠合到 `fixed`。"""
    common = sorted(set(moving) & set(fixed))
    if len(common) < 3:
        return None, None
    src = np.array([moving[p] for p in common], dtype=float)
    dst = np.array([fixed[p] for p in common], dtype=float)
    R, t = procrustes(src, dst)
    out = {p: R @ v + t for p, v in moving.items()}
    rmsd = float(np.sqrt((((src @ R.T + t) - dst) ** 2).sum(1).mean()))
    return out, rmsd


def splice_chimera(structs, key_a, key_b, block_pos):
    """在 block_pos 上使用 key_a 的坐标及 key_b 的对齐坐标。"""
    aligned_b, rmsd = align(structs[key_b], structs[key_a])
    if aligned_b is None:
        return None, None, None
    out = {p: v.copy() for p, v in structs[key_a].items()}
    for p in block_pos:
        if p in aligned_b:
            out[p] = aligned_b[p].copy()
    return out, rmsd, aligned_b


def bin_sign(kind, n_bins):
    """返回逐 bin 的 boolean flip mask，按文档中的规则自行实现。"""
    tb = np.zeros(n_bins, dtype=bool)
    nf = (n_bins + FRAG_BINS - 1) // FRAG_BINS
    if kind == "oracle":
        pass
    elif kind == "gauge_all":
        tb[:] = True
    elif kind.startswith("wall"):
        tb[:min(int(kind[4:]) * FRAG_BINS, n_bins)] = True
    elif kind.startswith("island"):
        c = (nf // 2) * FRAG_BINS
        tb[c:min(c + int(kind[6:]) * FRAG_BINS, n_bins)] = True
    elif kind == "alt":
        tb[:] = (np.arange(n_bins) // FRAG_BINS) % 2 == 1
    elif kind.startswith("micro"):
        tb[:min(int(kind[5:]), n_bins)] = True
    else:
        raise ValueError(kind)
    return tb


# --------------------------------------------------------------------------
# 4. reference 结构
# --------------------------------------------------------------------------
def load_ref():
    d = {}
    with gzip.open(REF3DG, "rt") as f:
        for line in f:
            a = line.split()
            if len(a) < 5:
                continue
            d.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
    return d


def ref_dist(refc, key, b1, b2):
    m = refc.get(key, {})
    out = np.full(len(b1), np.nan)
    for k, (x, y) in enumerate(zip(b1, b2)):
        va = m.get(int(OFF + x * BIN))
        vb = m.get(int(OFF + y * BIN))
        if va is None or vb is None:
            continue
        out[k] = float(np.linalg.norm(va - vb))
    return out


def main():
    t0 = time.time()
    report = {"sections": OrderedDict(), "checks": CHECKS}
    print("=" * 78)
    print("INDEPENDENT AUDIT of %s" % os.path.basename(RUN))
    print("=" * 78)

    # ---------------- 0. 原始 counts（section D prerequisites） --------------
    print("\n## 0. raw inputs")
    cols7, p1, p2 = load_snpfree(CHROM)
    colsf, q1, q2, ph = load_phase(CHROM)
    print("  snpfree columns: %s" % cols7)
    print("  evidence columns: %s" % colsf)
    check("D", "snpfree #columns == 7 expected names",
          cols7, ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"])
    n_rec = len(p1)
    check("D", "chr1 cis record count (snpfree)", n_rec, CLAIMS["n_records"])
    check("D", "chr1 cis record count (evidence file)", len(q1), CLAIMS["n_records"])
    check("D", "snpfree and evidence records line up",
          bool(len(p1) == len(q1) and np.array_equal(p1, q1) and np.array_equal(p2, q2)),
          True, note="needed for label/oracle alignment")

    I1, I2 = bin_of(p1), bin_of(p2)
    samebin = int((I1 == I2).sum())
    check("D", "same-bin cis records", samebin, CLAIMS["samebin_records"])
    n_bins = int(max(I1.max(), I2.max())) + 1
    check("D", "n_bins (max bin index + 1)", n_bins, CLAIMS["n_bins"])
    nfrag = (n_bins + FRAG_BINS - 1) // FRAG_BINS
    check("D", "n_fragments of 20 bins", nfrag, CLAIMS["n_fragments"])

    lb = np.where(ph == "00", 0, np.where(ph == "11", 1, -1)).astype(np.int8)
    counts = {"n00": int((lb == 0).sum()), "n11": int((lb == 1).sum()),
              "n_unphased": int((lb < 0).sum())}
    check("D", "label counts n00/n11/n_unphased", counts, CLAIMS["label_counts"])
    check("D", "fully phased cis records (00+11)", counts["n00"] + counts["n11"],
          counts["n00"] + counts["n11"],
          note="reported pair; no separate run claim")

    f = fold_of_array(I1, I2)
    tr, va, te = np.isin(f, TRAIN_FOLDS), np.isin(f, VAL_FOLDS), np.isin(f, TEST_FOLDS)
    check("D", "train records (folds 0-5)", int(tr.sum()), CLAIMS["n_train"])
    check("D", "val records (folds 6-7)", int(va.sum()), CLAIMS["n_val"])
    check("D", "test records (folds 8-9)", int(te.sum()), CLAIMS["n_test"])
    check("D", "fold partition covers every record",
          int(tr.sum() + va.sum() + te.sum()), n_rec)

    ph_tr = tr & (lb >= 0)
    check("D", "fully phased train records", int(ph_tr.sum()), CLAIMS["n_ph_train"])

    def bin_pairs(mask):
        keep = mask & (I1 != I2)
        s = {}
        for a, b in zip(I1[keep], I2[keep]):
            k = (int(a), int(b))
            s[k] = s.get(k, 0) + 1
        K = np.array(sorted(s), dtype=np.int64)
        C = np.array([s[tuple(k)] for k in K], dtype=float)
        return K[:, 0], K[:, 1], C

    vb1, vb2, vC = bin_pairs(va)
    tb1, tb2, tC = bin_pairs(te)
    check("D", "held-out val bin pairs (same-bin excluded)", len(vb1), CLAIMS["pairs_val"])
    check("D", "held-out test bin pairs (same-bin excluded)", len(tb1), CLAIMS["pairs_test"])
    check("A", "scoring universe == claimed test bin pairs",
          len(tb1), CLAIMS["pairs_test"])

    # strata（根据文档规则得到的 fragment indices）
    def frag_of(bins):
        return np.clip(bins // FRAG_BINS, 0, nfrag - 1)

    ft1, ft2 = frag_of(tb1), frag_of(tb2)
    sep = np.abs(tb1 - tb2)
    strata = {"all": np.ones(len(tb1), dtype=bool),
              "intra_frag": ft1 == ft2,
              "xshort": (ft1 != ft2) & (sep <= 40),
              "xlong": (ft1 != ft2) & (sep > 40)}
    check("D", "test stratum sizes all/intra/xshort/xlong",
          {k: int(v.sum()) for k, v in strata.items()}, CLAIMS["stratum_sizes"])

    # ---------------- 1. reference 分类 accuracy -----------------
    print("\n## 1. reference-structure classification accuracy (D)")
    refc = load_ref()
    keep = (I1 != I2) & (lb >= 0)
    rb1, rb2, rlb = I1[keep], I2[keep], lb[keep]
    d_pat = ref_dist(refc, "%s(pat)" % CHROM, rb1, rb2)
    d_mat = ref_dist(refc, "%s(mat)" % CHROM, rb1, rb2)
    ok = np.isfinite(d_pat) & np.isfinite(d_mat)
    d_pat, d_mat, rlb = d_pat[ok], d_mat[ok], rlb[ok]
    tie = d_pat == d_mat
    accs = {}
    for orient, (first, second) in enumerate(((d_pat, d_mat), (d_mat, d_pat))):
        own = np.where(rlb == 0, first, second)
        other = np.where(rlb == 0, second, first)
        accs[orient] = float(((own < other).astype(float) + 0.5 * tie).mean())
    acc = max(accs.values())
    acc_orient = int(accs[1] > accs[0])
    ra = CLAIMS["reference_accuracy"]
    check("D", "reference accuracy (same-bin excluded)", acc, ra["acc"], tol=1e-12)
    check("D", "reference accuracy orientation", acc_orient, ra["orientation"])
    check("D", "reference accuracy n (phased, non-same-bin)", int(len(rlb)), ra["n"])
    check("D", "same-bin excluded count in reference readout",
          int((I1 == I2).sum()), ra["n_samebin_excluded"])
    check("D", "reference tie rate", float(tie.mean()), ra["tie_rate"], tol=1e-12)
    ref_report = {"acc": acc, "acc_orient0": accs[0], "acc_orient1": accs[1],
                  "n": int(len(rlb)), "tie_rate": float(tie.mean())}

    # ---------------- 2. NaN / 评分全集（A） -------------------------
    print("\n## 2. NaN policy and scoring universe (A)")
    tags = list(CLAIMS["table"].keys())
    fin = {}
    for tag in tags:
        fin[tag] = finite_mask(tag, tb1, tb2)
    check("A", "finite test pairs: oracle", int(fin["oracle"].sum()),
          CLAIMS["finite_counts"]["oracle"])
    check("A", "finite test pairs: oracle_half1", int(fin["oracle_half1"].sum()),
          CLAIMS["finite_counts"]["oracle_half1"])
    check("A", "finite test pairs: oracle_half2", int(fin["oracle_half2"].sum()),
          CLAIMS["finite_counts"]["oracle_half2"])
    n_fin_claim = {k: v["n_finite"] for k, v in CLAIMS["table"].items()}
    n_nan_claim = {k: v["n_nan"] for k, v in CLAIMS["table"].items()}
    bad_fin = {k: (int(fin[k].sum()), n_fin_claim[k]) for k in tags
               if int(fin[k].sum()) != n_fin_claim[k]}
    bad_nan = {k: (int((~fin[k]).sum()), n_nan_claim[k]) for k in tags
               if int((~fin[k]).sum()) != n_nan_claim[k]}
    check("A", "table[*].n_finite matches for all %d candidates" % len(tags),
          bad_fin, {})
    check("A", "table[*].n_nan matches for all %d candidates" % len(tags),
          bad_nan, {})
    # 哪些 bins 对 half-data oracles 缺失？
    missing = {}
    for tag in ("oracle", "oracle_half1", "oracle_half2"):
        st, tracks = candidate_parts(tag)
        have = set(st[tracks[0]]) & set(st[tracks[1]]) if len(tracks) > 1 else set(st[tracks[0]])
        miss = sorted({int(b) for k in np.where(~fin[tag])[0] for b in (tb1[k], tb2[k])
                       if OFF + int(b) * BIN not in have})
        missing[tag] = miss
    print("  bins with no bead: %s" % missing)
    check("A", "oracle has a bead for every test bin",
          missing["oracle"], [])
    report["missing_bins"] = missing

    # 对 oracle，bootstrap n == true common-set size
    boot_n_claim = {k: v["n"] for k, v in CLAIMS["bootstrap"].items()}
    boot_n_mine = {k: int((fin["oracle"] & fin[k]).sum()) for k in boot_n_claim}
    bad_boot = {k: (boot_n_mine[k], boot_n_claim[k]) for k in boot_n_claim
                if boot_n_mine[k] != boot_n_claim[k]}
    check("A", "bootstrap[*].n == |oracle-finite & candidate-finite|",
          bad_boot, {})
    check("A", "bootstrap[oracle_half1].n", boot_n_mine["oracle_half1"],
          boot_n_claim["oracle_half1"],
          note="1609 not 1603/811")
    check("A", "bootstrap[oracle_half2].n", boot_n_mine["oracle_half2"],
          boot_n_claim["oracle_half2"])
    report["section_A"] = {
        "finite": {k: int(fin[k].sum()) for k in tags},
        "nan": {k: int((~fin[k]).sum()) for k in tags},
        "missing_bins": missing,
        "bootstrap_n_mine": boot_n_mine,
    }
    # 检查已废弃的 global-intersection 策略会产生什么结果
    glob = np.ones(len(tb1), dtype=bool)
    for t in tags:
        glob &= fin[t]
    h1h2 = fin["oracle_half1"] & fin["oracle_half2"]
    print("  retired global-intersection sizes: all-candidate %d, half-oracles %d"
          % (int(glob.sum()), int(h1h2.sum())))
    report["section_A"]["global_intersection_all"] = int(glob.sum())
    report["section_A"]["intersection_half_oracles"] = int(h1h2.sum())

    # ---------------- 3. gate 覆盖（B） ---------------------------------
    print("\n## 3. gate coverage of written coordinates (B)")
    files = sorted(x for x in os.listdir(COORDS) if x.endswith(".3dg"))
    on_disk = {os.path.join("coords", x) for x in files}
    # 规范化：gate paths 相对于 ROOT，例如 test_res/<run>/coords/x.3dg
    gate_by_rel = {}
    for e in GATE:
        rp = os.path.relpath(os.path.join(ROOT, e["path"]), RUN)
        gate_by_rel[rp] = e
    missing_in_gate = sorted(on_disk - set(gate_by_rel))
    extra_in_gate = sorted(set(gate_by_rel) - on_disk)
    mismatched = []
    for rel in sorted(on_disk & set(gate_by_rel)):
        got = sha256_file(os.path.join(RUN, rel))
        want = gate_by_rel[rel]["sha256"]
        if got != want:
            mismatched.append({"file": rel, "on_disk": got, "in_gate": want})
    n_relax = sum(1 for x in files if x.endswith("_relax.3dg"))
    check("B", "number of .3dg files under coords/", len(files), len(GATE),
          note="%d of them are *_relax.3dg" % n_relax)
    check("B", "every .3dg file has a gate.json entry", missing_in_gate, [])
    check("B", "every gate.json entry has a file on disk", extra_in_gate, [])
    check("B", "every recorded sha256 matches the file on disk", mismatched, [])
    check("B", "*_relax.3dg files registered in gate.json", n_relax,
          sum(1 for r in gate_by_rel if r.endswith("_relax.3dg")))
    check("B", "gate stage/tag pairs are unique",
          len({(e["stage"], e["tag"]) for e in GATE}), len(GATE))
    report["section_B"] = {"n_files": len(files), "n_gate_entries": len(GATE),
                           "n_relax_files": n_relax,
                           "missing_in_gate": missing_in_gate,
                           "extra_in_gate": extra_in_gate,
                           "mismatched": mismatched}

    # ---------------- 4. 表格复现（D） -----------------------------------
    print("\n## 4. per-candidate pooled rho (D)")
    mine = {}
    agree = {}
    for tag in tags:
        a, vr = select_alpha(tag, vb1, vb2, vC)
        pred, ok = pooled_prediction(tag, tb1, tb2, a)
        idx = np.where(ok)[0]
        rho = spearman(pred[idx], tC[idx]) if len(idx) >= 20 else float("nan")
        mine[tag] = {"alpha": a, "rho_pooled": rho, "n": int(len(idx)),
                     "alpha_val_rho": vr}
        claim = CLAIMS["table"][tag]
        agree[tag] = (abs(rho - claim["rho_pooled"]) <= 1e-12 and a == claim["alpha"])
    bad = {k: (mine[k], CLAIMS["table"][k]["rho_pooled"], CLAIMS["table"][k]["alpha"])
           for k in tags if not agree[k]}
    check("D", "table[*].rho_pooled and alpha reproduced for all %d candidates"
          % len(tags), bad, {})
    for k in ("oracle", "gauge_all", "consensus", "random_phased"):
        check("D", "table[%s].rho_pooled" % k, mine[k]["rho_pooled"],
              CLAIMS["table"][k]["rho_pooled"], tol=1e-12,
              note="alpha %s" % mine[k]["alpha"])
    report["section_D_table"] = mine

    # ---------------- 5. splice 评估工具（C） -----------------------------
    print("\n## 5. splice instrument: index-array vs boolean-mask bug (C)")
    n_test = len(tb1)
    orc_structs = structures("oracle")
    orc_alpha = CLAIMS["table"]["oracle"]["alpha"]
    my_orc_alpha = mine["oracle"]["alpha"]
    check("C", "oracle alpha (validation-selected)", my_orc_alpha, orc_alpha)
    orc_pred, orc_ok = pooled_prediction("oracle", tb1, tb2, my_orc_alpha)
    splice_res = {}
    splice_claim = CLAIMS["splice"]
    parity_value = {}
    for m in ["micro5", "micro10", "wall1", "wall2", "wall3", "wall4", "wall5"]:
        tb = bin_sign(m, n_bins)
        blk = [OFF + b * BIN for b in range(n_bins) if tb[b]]
        A2, rms_a, aligned_b = splice_chimera(orc_structs, TRACK_A, TRACK_B, blk)
        B2, _, _ = splice_chimera(orc_structs, TRACK_B, TRACK_A, blk)
        st = {TRACK_A: A2, TRACK_B: B2}
        dA = distances(st, TRACK_A, tb1, tb2)
        dB = distances(st, TRACK_B, tb1, tb2)
        ok = np.isfinite(dA) & np.isfinite(dB)
        pred = predictor(dA, my_orc_alpha) + predictor(dB, my_orc_alpha)
        pred = np.where(ok, pred, np.nan)
        # 运行的全集：oracle-finite 且 splice-finite
        idx = np.where(orc_ok & ok)[0]
        rho = spearman(pred[idx], tC[idx])
        do = spearman(orc_pred[idx], tC[idx]) - rho
        lo, hi, k = paired_bootstrap(tC[idx], orc_pred[idx], pred[idx])
        # 已废弃的 `np.where(index_array & bool_mask)` 缺陷会产生什么结果：
        parity_idx = np.where(np.arange(n_test) & orc_ok & ok)[0]
        even_idx = np.where(((np.arange(n_test) & 1) == 0) & orc_ok & ok)[0]
        parity_value[m] = {
            "buggy_odd_rows_n": int(len(parity_idx)),
            "buggy_odd_rows_rho": spearman(pred[parity_idx], tC[parity_idx]),
            "even_rows_n": int(len(even_idx)),
            "even_rows_rho": spearman(pred[even_idx], tC[even_idx]),
        }
        splice_res["splice_%s" % m] = {
            "rho_pooled": rho, "delta_vs_oracle": do, "lo": lo, "hi": hi,
            "n": int(len(idx)), "flipped_mb": int(tb.sum()),
            "procrustes_rmsd": rms_a, "n_boot_finite": k,
        }
    for k, v in splice_res.items():
        c = splice_claim[k]
        check("C", "%s n == full common finite set" % k, v["n"], c["n"],
              note="half-set would be ~%d; buggy parity subset n=%d"
                   % (n_test // 2, parity_value[k[7:]]["buggy_odd_rows_n"]))
        check("C", "%s rho_pooled" % k, v["rho_pooled"], c["rho_pooled"], tol=1e-12)
        check("C", "%s delta_vs_oracle" % k, v["delta_vs_oracle"],
              c["delta_vs_oracle"], tol=1e-12)
        check("C", "%s bootstrap CI [lo, hi]" % k,
              [round(v["lo"], 10), round(v["hi"], 10)],
              [round(c["lo"], 10), round(c["hi"], 10)], tol=1e-9)
        check("C", "%s procrustes rmsd" % k, v["procrustes_rmsd"],
              c["procrustes_rmsd"], tol=1e-9)
    check("C", "splice_wall1 rho matches the claimed 20 Mb value",
          splice_res["splice_wall1"]["rho_pooled"],
          splice_claim["splice_wall1"]["rho_pooled"], tol=1e-12,
          note="claimed value is NOT the parity-subset rho %.6f"
               % parity_value["wall1"]["buggy_odd_rows_rho"])
    report["section_C"] = {"splice": splice_res, "parity_counterfactual": parity_value}

    # ---------------- 6. 规范不变读出（D） -----------------------
    print("\n## 6. gauge-invariant readout and wall1 delta (D)")
    rho_o = mine["oracle"]["rho_pooled"]
    rho_ga = mine["gauge_all"]["rho_pooled"]
    ref_gi = (rho_o + rho_ga) / 2.0
    asym_engine = abs(rho_o - rho_ga)
    gi_claim = CLAIMS["gauge_invariant"]
    check("D", "engine gauge asymmetry |rho(oracle)-rho(gauge_all)|",
          asym_engine, gi_claim["_reference"]["gauge_asymmetry"], tol=1e-12)
    check("D", "oracle gauge-averaged reference level", ref_gi,
          gi_claim["_reference"]["rho_oracle_gauge_avg"], tol=1e-12)
    gi_mine = {}
    for cid, claim in gi_claim.items():
        if cid == "_reference":
            continue
        if cid + "_gc" in mine:
            r_b = (mine[cid]["rho_pooled"] + mine[cid + "_gc"]["rho_pooled"]) / 2.0
            ids = ["oracle", "gauge_all", cid, cid + "_gc"]
            asym = abs(mine[cid]["rho_pooled"] - mine[cid + "_gc"]["rho_pooled"])
        else:
            r_b = mine[cid]["rho_pooled"]
            ids = ["oracle", "gauge_all", cid]
            asym = 0.0
        m = np.ones(n_test, dtype=bool)
        for t in ids:
            m &= fin[t]
        idx = np.where(m)[0]
        preds = {}
        for t in ids:
            pr, _ = pooled_prediction(t, tb1, tb2, mine[t]["alpha"])
            preds[t] = pr
        lo, hi, k = paired_bootstrap_gaugeavg(
            tC[idx], preds["oracle"][idx], preds["gauge_all"][idx],
            preds[cid][idx], preds[cid + "_gc"][idx] if cid + "_gc" in preds
            else preds[cid][idx])
        gi_mine[cid] = {"rho_gauge_avg": r_b, "gauge_asymmetry": asym,
                        "n": int(len(idx)), "delta": ref_gi - r_b,
                        "lo": lo, "hi": hi}
        check("D", "gauge_invariant[%s] rho_avg/n/asym" % cid,
              [round(r_b, 12), int(len(idx)), round(asym, 12)],
              [round(claim["rho_gauge_avg"], 12), claim["n"],
               round(claim["gauge_asymmetry"], 12)], tol=1e-9)
        check("D", "gauge_invariant[%s] delta and CI" % cid,
              [round(ref_gi - r_b, 10), round(lo, 10), round(hi, 10)],
              [round(claim["delta"], 10), round(claim["lo"], 10),
               round(claim["hi"], 10)], tol=1e-9)
    report["section_D_gauge"] = {"ref_gi": ref_gi, "asym": asym_engine,
                                 "per_candidate": gi_mine}

    # ---------------- 7. 关键配对 bootstrap 断言 --------------------------
    print("\n## 7. paired bootstrap vs oracle (spot checks)")
    for cid in ("consensus", "random_phased", "gauge_all", "oracle_half1",
                "oracle_half2"):
        m = fin["oracle"] & fin[cid]
        idx = np.where(m)[0]
        pr, _ = pooled_prediction(cid, tb1, tb2, mine[cid]["alpha"])
        lo, hi, k = paired_bootstrap(tC[idx], orc_pred[idx], pr[idx])
        c = CLAIMS["bootstrap"][cid]
        check("D", "bootstrap[%s] n / CI" % cid,
              [int(len(idx)), round(lo, 9), round(hi, 9)],
              [c["n"], round(c["lo"], 9), round(c["hi"], 9)], tol=1e-8)

    # ---------------- 8. 缺陷类别的静态检查 -------------------------------
    print("\n## 8. static scan for index-array & boolean-mask combinations")
    import re
    hits = []
    for fn in ["run.py"] + [os.path.join("pr", x) for x in sorted(os.listdir(os.path.join(ROOT, "pr")))
                            if x.endswith(".py")]:
        p = os.path.join(ROOT, fn)
        for ln, line in enumerate(open(p), 1):
            if re.search(r"np\.where\([^)]*&", line) or re.search(r"STRATUM_IDX", line):
                hits.append("%s:%d: %s" % (fn, ln, line.strip()))
    for h in hits:
        print("   " + h)
    report["static_scan"] = hits

    # ---------------- 9. 逐 strata 的 n 和 rho（A 的一部分） ------------
    print("\n## 9. every table row reports its own per-stratum n and rho (A)")

    def tb_of(tag):
        """根据 candidate tag 推断逐 bin flip mask，完全自行重建。

        对于完全没有 flip pattern 的 candidate（按运行约定，unaffected 与 affected 都是全集）返回 None。
        """
        if tag in ("consensus", "random", "random_phased", "random_phased_1",
                   "random_phased_2"):
            return None
        if tag in ("oracle", "oracle_half1", "oracle_half2"):
            return np.zeros(n_bins, dtype=bool)
        if tag == "gauge_all":
            return np.ones(n_bins, dtype=bool)
        kind = tag
        if kind.endswith("_gc"):
            return ~bin_sign(kind[:-3], n_bins)
        if kind.endswith("_rewire"):
            kind = kind[:-7]
        return bin_sign(kind, n_bins)

    strata_bad = {}
    for tag in tags:
        tb = tb_of(tag)
        if tb is None:
            masks = dict(strata)
            masks["unaffected"] = np.ones(len(tb1), dtype=bool)
            masks["affected"] = np.ones(len(tb1), dtype=bool)
        else:
            a, b = tb[tb1], tb[tb2]
            masks = dict(strata)
            masks["unaffected"] = (a == b)
            masks["affected"] = (a != b)
        claim = CLAIMS["table"][tag]["strata"]
        pr = pooled_prediction(tag, tb1, tb2, mine[tag]["alpha"])[0]
        for m, mm in masks.items():
            idx = np.where(mm & fin[tag])[0]
            rho = spearman(pr[idx], tC[idx]) if len(idx) >= 20 else float("nan")
            c = claim[m]
            okn = int(len(idx)) == c["n"]
            okr = (np.isnan(rho) and (c["rho"] is None or
                                      (isinstance(c["rho"], float) and np.isnan(c["rho"]))))
            if not okr and not np.isnan(rho):
                okr = abs(rho - c["rho"]) <= 1e-12
            if not (okn and okr):
                strata_bad["%s/%s" % (tag, m)] = {
                    "n": [int(len(idx)), c["n"]],
                    "rho": [rho, c["rho"]]}
    check("A", "strata n (all/intra/xshort/xlong/una/aff) for %d candidates"
          % len(tags), strata_bad, {})
    report["section_A_strata_bad"] = strata_bad

    # NaN 机制的严格检查：哪些 bins 缺失，以及它们移除了多少
    # 留出 pairs
    print("  missing-bead mechanism:")
    for tag, b in (("oracle_half1", 5), ("oracle_half2", 189)):
        n_pairs_b = int(((tb1 == b) | (tb2 == b)).sum())
        st, tracks = candidate_parts(tag)
        have = set(st[tracks[0]]) & set(st[tracks[1]])
        allbins = sorted(set(int(x) for x in tb1) | set(int(x) for x in tb2))
        miss = [x for x in allbins if OFF + x * BIN not in have]
        print("    %s: missing bead bins %s; test pairs touching bin %d = %d; NaN = %d"
              % (tag, miss, b, n_pairs_b, int((~fin[tag]).sum())))
        check("A", "%s NaN count == test pairs touching its missing bin" % tag,
              int((~fin[tag]).sum()), n_pairs_b)

    # 明确说明：声明的 splice rho 不等于任何 parity-subset value
    print("  splice parity counterfactual (C):")
    for m, v in parity_value.items():
        c = splice_claim["splice_%s" % m]
        print("    %-8s full n=%d rho=%.9f | odd-rows n=%d rho=%.9f | even-rows n=%d rho=%.9f"
              % (m, splice_res["splice_%s" % m]["n"],
                 splice_res["splice_%s" % m]["rho_pooled"],
                 v["buggy_odd_rows_n"], v["buggy_odd_rows_rho"],
                 v["even_rows_n"], v["even_rows_rho"]))
        check("C", "%s claimed rho != odd-row (parity) rho" % m,
              abs(c["rho_pooled"] - v["buggy_odd_rows_rho"]) > 1e-9, True)
        check("C", "%s claimed rho != even-row rho" % m,
              abs(c["rho_pooled"] - v["even_rows_rho"]) > 1e-9, True)

    # ---------------- 10. 评分输入是否为文档所述输入？ --------------------
    print("\n## 10. work/*.pairs.gz == documented construction from raw data")
    # 为每个非 relax candidate 重建精确的 FDG input text
    o_lb = lb[ph_tr]
    o_p1, o_p2 = p1[ph_tr], p2[ph_tr]
    b1_tr, b2_tr = I1[ph_tr], I2[ph_tr]
    order = np.random.default_rng(0).permutation(int(ph_tr.sum()))
    halves = {"oracle_half1": order[:int(ph_tr.sum()) // 2],
              "oracle_half2": order[int(ph_tr.sum()) // 2:]}

    def assign_for(tag):
        """按文档返回 (record p1, p2, assignment 或 None, single-track flag)。"""
        if tag == "consensus":
            return p1[tr], p2[tr], None
        if tag == "random":
            return p1[tr], p2[tr], np.random.default_rng(0).integers(0, 2, int(tr.sum()))
        if tag.startswith("random_phased"):
            k = 0 if tag == "random_phased" else int(tag.rsplit("_", 1)[1])
            return o_p1, o_p2, np.random.default_rng(5 + k).integers(0, 2, len(o_p1))
        if tag in halves:
            sub = halves[tag]
            return o_p1[sub], o_p2[sub], o_lb[sub]
        tb = tb_of(tag)
        assert tb is not None
        a = tb[b1_tr]
        flip = a if not tag.endswith("_rewire") else (a != tb[b2_tr])
        return o_p1, o_p2, (o_lb ^ flip.astype(o_lb.dtype))

    def pairs_text(tag, rp1, rp2, assign):
        two = assign is not None
        tracks = [TRACK_A, TRACK_B] if two else [CHROM]
        out = ["## pairs format v1.0\n", "#sorted: chr1-chr2-pos1-pos2\n",
               "#shape: upper triangle\n"]
        for t in tracks:
            out.append("#chromosome: %s %d\n" % (t, 300_000_000))
        out.append("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
        if two:
            for x, y, k in zip(rp1, rp2, assign):
                nm = tracks[int(k)]
                out.append(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, x, nm, y))
        else:
            for x, y in zip(rp1, rp2):
                out.append(".\t%s\t%d\t%s\t%d\t+\t+\n" % (CHROM, x, CHROM, y))
        return "".join(out)

    pairs_bad = {}
    recon = {}
    for tag in tags:
        rp1, rp2, assign = assign_for(tag)
        expect = pairs_text(tag, rp1, rp2, assign)
        with gzip.open(os.path.join(RUN, "work", tag + ".pairs.gz"), "rt") as f:
            got = f.read()
        n_expect = len(rp1)
        n_got = got.count("\n") - 6
        same = got == expect
        recon[tag] = {"n_records": n_expect, "identical_text": bool(same),
                      "n_records_on_disk": n_got}
        if not same:
            pairs_bad[tag] = {"n_expect": n_expect, "n_disk": n_got,
                              "first_diff": next((i for i in range(min(len(got), len(expect)))
                                                  if got[i] != expect[i]), None)}
    check("D", "work/*.pairs.gz reproduced from raw data for %d candidates" % len(tags),
          pairs_bad, {})
    # 运行日志记录的每个 split 分给 copy a 的训练 contacts 数
    logtxt = open(os.path.join(RUN, "logs", "run.log")).read()
    for tag in ("oracle", "wall1", "wall1_rewire", "gauge_all"):
        _, _, assign = assign_for(tag)
        key = "%s flipped" % tag if not tag.endswith("_rewire") else "%s flipped" % tag
        import re as _re
        mline = _re.search(r"^\[[\d:]+\] %s\s+flipped.*copy-a gets (\d+)/(\d+) train contacts"
                           % _re.escape(tag), logtxt, _re.M)
        if mline:
            check("D", "%s copy-a contact count matches run.log" % tag,
                  [int(assign.sum()), len(assign)],
                  [int(mline.group(1)), int(mline.group(2))])
    report["section_pairs_recon"] = recon

    # ---------------- 11. native refit 抽查 --------------------------
    print("\n## 11. native refit reproduces the scored coordinates (binding check)")
    import subprocess
    HICKIT = os.path.join(ROOT, "native", "hickit", "hickit")
    tmp = os.path.join(HERE, "tmp")
    os.makedirs(tmp, exist_ok=True)
    refit = {}
    if os.path.exists(HICKIT):
        for tag in ("oracle", "wall1", "oracle_half1"):
            inp = os.path.join(RUN, "work", tag + ".pairs.gz")
            outp = os.path.join(tmp, tag + ".refit.3dg")
            r = subprocess.run([HICKIT, "-i", inp, "-P1", "-b1m", "-O", outp, "-s", "1"],
                               capture_output=True, text=True, cwd=tmp)
            if r.returncode != 0:
                refit[tag] = {"error": r.stderr[-300:]}
                continue
            a = open(outp).read()
            b = open(os.path.join(COORDS, tag + ".3dg")).read()
            refit[tag] = {"identical": a == b, "n_lines_refit": a.count("\n"),
                          "n_lines_disk": b.count("\n")}
            check("D", "coords/%s.3dg == fresh native fit of work/%s.pairs.gz"
                  % (tag, tag), refit[tag]["identical"], True,
                  note="seed 1, -P1 -b1m")
    else:
        print("  native engine not found; binding check skipped")
    report["section_refit"] = refit

    # ---------------- 12. fold hash sanity（支持 D） -----------------------
    print("\n## 12. fold hash sanity (the counts in D rest on it)")
    fa = fold_of_array(I1, I2)          # `f` 被上面的 file handles 遮蔽
    same_bin_fold = np.bincount(fa[I1 == I2], minlength=10)
    check("D", "every fold receives same-bin pairs (old (i+j)%2 bug)",
          int((same_bin_fold > 0).sum()), 10, note=str(same_bin_fold.tolist()))
    fr = []
    for k in range(10):
        sel = fa == k
        fr.append(float(((I1 == I2) & sel).sum() / max(1, sel.sum())))
    # 文档中的回归判据（pr.folds.sanity_check）：固定 separation 不应
# 塌缩到少数几个 folds
    distinct_by_sep = {}
    sep_all = np.abs(I1 - I2)
    for d in (1, 2, 5, 20, 50):
        distinct_by_sep[d] = int(len(np.unique(fa[sep_all == d])))
    check("D", "distinct folds used at fixed separation (>=5 required)",
          min(distinct_by_sep.values()), 5, tol=None,
          agree=min(distinct_by_sep.values()) >= 5,
          note=str(distinct_by_sep))
    parity_spread = {}
    for p in (0, 1):
        sel = (sep_all % 2) == p
        c = np.bincount(fa[sel], minlength=10) / max(1, sel.sum())
        parity_spread[p] = float(c.max() - c.min())
    info = {
        "fold_totals": np.bincount(fa, minlength=10).tolist(),
        "samebin_per_fold": same_bin_fold.tolist(),
        "samebin_fraction_per_fold": [round(x, 4) for x in fr],
        "parity_fold_share_max_minus_min": parity_spread,
        "distinct_folds_at_fixed_separation": distinct_by_sep,
        "mean_separation_per_fold": [round(float(sep_all[fa == k].mean()), 3)
                                     for k in range(10)],
    }
    # val 与 test 的 strata 组成（检查 alpha 是否能从 val 迁移到 test）
    fv1, fv2 = frag_of(vb1), frag_of(vb2)
    vs_masks = {"intra_frag": fv1 == fv2,
                "xshort": (fv1 != fv2) & (np.abs(vb1 - vb2) <= 40),
                "xlong": (fv1 != fv2) & (np.abs(vb1 - vb2) > 40)}
    info["val_stratum_sizes"] = {k: int(v.sum()) for k, v in vs_masks.items()}
    info["test_stratum_sizes"] = {k: int(v.sum()) for k, v in strata.items()
                                  if k != "all"}
    info["val_vs_test_frac"] = {
        k: [round(info["val_stratum_sizes"][k] / len(vb1), 4),
            round(info["test_stratum_sizes"][k] / len(tb1), 4)]
        for k in vs_masks}
    print("  INFO val vs test stratum fractions: %s" % info["val_vs_test_frac"])
    print("  INFO fold totals: %s" % info["fold_totals"])
    print("  INFO same-bin fraction per fold: %s" % info["samebin_fraction_per_fold"])
    print("  INFO mean |i-j| per fold: %s" % info["mean_separation_per_fold"])
    print("  INFO fold-share spread across 10 folds within each separation parity: %s"
          % {k: round(v, 4) for k, v in parity_spread.items()})
    report["section_fold_info"] = info
    # README 中记录的已废弃 global-intersection 策略大小
    check("A", "retired global-intersection size matches README (1603)",
          int(glob.sum()), 1603)

    # ---------------- 13. README.md 与 results.json（声明一致性） --------
    print("\n## 13. README.md tables vs results.json")
    readme = open(os.path.join(RUN, "README.md")).read().splitlines()

    def md_tables(lines):
        out, cur = [], []
        for ln in lines:
            s = ln.strip()
            if s.startswith("|"):
                cur.append(s)
            elif cur:
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        parsed = []
        for t in out:
            cells = [[c.strip() for c in row.strip("|").split("|")] for row in t]
            if len(cells) >= 2 and all(set(c) <= set("-: ") for c in cells[1]):
                parsed.append((cells[0], cells[2:]))
        return parsed

    def num(s):
        s = s.replace("*", "").replace("`", "").replace("+", "").strip()
        if s in ("-", "", "nan", "NaN", "yes", "no"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    tables = md_tables(readme)
    print("  parsed %d markdown tables" % len(tables))
    t_rho = t_gauge = t_aff = t_splice = t_relax = t_decision = None
    for hdr, rows in tables:
        if "pooled rho" in hdr:
            t_rho = rows
        elif any("gauge-avg" in h for h in hdr):
            t_gauge = rows
        elif any("rho oracle" in h for h in hdr):
            t_aff = rows
        elif hdr and hdr[0].strip().lower().startswith("flipped mb"):
            if any("n" == h.strip() for h in hdr):
                t_splice = rows
            else:
                t_decision = rows
        elif any("rho before" in h for h in hdr):
            t_relax = rows

    readme_bad = {}

    def cmp_row(key, field, got, want, tol=1e-4):
        # README 对未定义 cell 打印 "nan"；results.json 保存 NaN
        if isinstance(got, float) and np.isnan(got):
            got = None
        if want is not None and isinstance(want, float) and np.isnan(want):
            want = None
        if want is None or got is None:
            if not (want is None and got is None):
                readme_bad[key] = {"readme": want, "results": got, "field": field}
            return
        if abs(got - want) > tol:
            readme_bad[key] = {"readme": want, "results": got, "field": field}

    # 表 1：每个 candidate 的留出 rho
    n_rho_rows = 0
    for r in (t_rho or []):
        tag = r[0].replace("`", "")
        if tag not in CLAIMS["table"]:
            continue
        n_rho_rows += 1
        c = CLAIMS["table"][tag]
        cmp_row(tag, "alpha", c["alpha"], num(r[4]))
        cmp_row(tag, "n", c["n_finite"], num(r[5]))
        cmp_row(tag, "nan", c["n_nan"], num(r[6]))
        cmp_row(tag, "rho_pooled", c["rho_pooled"], num(r[7]))
        cmp_row(tag, "intra_frag", c["strata"]["intra_frag"]["rho"], num(r[8]))
        cmp_row(tag, "xshort", c["strata"]["xshort"]["rho"], num(r[9]))
        cmp_row(tag, "xlong", c["strata"]["xlong"]["rho"], num(r[10]))
    check("D", "README held-out table matches results.json (%d rows)" % n_rho_rows,
          {k: v for k, v in readme_bad.items()}, {})

    # 表 2：gauge-invariant
    n_g = 0
    for r in (t_gauge or []):
        tag = r[0].replace("`", "")
        if tag not in CLAIMS["gauge_invariant"]:
            continue
        n_g += 1
        c = CLAIMS["gauge_invariant"][tag]
        cmp_row(tag, "rho_gauge_avg", c["rho_gauge_avg"], num(r[3]))
        cmp_row(tag, "gauge_asymmetry", c["gauge_asymmetry"], num(r[4]))
        cmp_row(tag, "delta", c["delta"], num(r[5]))
        cmp_row(tag, "n", c["n"], num(r[7]))
        ci = r[6].strip("[]").split(",")
        cmp_row(tag, "lo", c["lo"], num(ci[0]))
        cmp_row(tag, "hi", c["hi"], num(ci[1]))
        cmp_row(tag, "detected", 1.0 if c["excludes_zero"] else 0.0,
                1.0 if "yes" in r[8] else 0.0)
    check("D", "README gauge-invariant table matches (%d rows)" % n_g,
          {k: v for k, v in readme_bad.items()}, {})

    # 表 3：affected-pair readout
    n_a = 0
    for r in (t_aff or []):
        tag = r[0].replace("`", "")
        if tag not in CLAIMS["masked_compare"]:
            continue
        n_a += 1
        c = CLAIMS["masked_compare"][tag]
        cmp_row(tag, "n", c["n"], num(r[1].split()[0]))
        cmp_row(tag, "n_common_with_null", c["n_common_with_null"], num(r[2]))
        cmp_row(tag, "rho_oracle", c["rho_oracle"], num(r[3]))
        cmp_row(tag, "rho_cand", c["rho_cand"], num(r[4]))
        cmp_row(tag, "delta", c["delta"], num(r[5]))
        ci = r[6].strip("[]").split(",")
        cmp_row(tag, "lo", c["lo"], num(ci[0]))
        cmp_row(tag, "hi", c["hi"], num(ci[1]))
        cmp_row(tag, "null", c["null_delta_half_oracles"], num(r[7]))
    check("D", "README affected-pair table matches (%d rows)" % n_a,
          {k: v for k, v in readme_bad.items()}, {})

    # 表 4：splice
    n_s = 0
    for r in (t_splice or []):
        mb = num(r[0])
        key = None
        for k, v in CLAIMS["splice"].items():
            if v["flipped_mb"] == mb and k.startswith("splice_"):
                key = k
        if key is None:
            continue
        n_s += 1
        c = CLAIMS["splice"][key]
        cmp_row(key, "rho", c["rho_pooled"], num(r[1]))
        cmp_row(key, "delta", c["delta_vs_oracle"], num(r[2]))
        ci = r[3].strip("[]").split(",")
        cmp_row(key, "lo", c["lo"], num(ci[0]))
        cmp_row(key, "hi", c["hi"], num(ci[1]))
        cmp_row(key, "n", c["n"], num(r[4]))
        cmp_row(key, "detected", 1.0 if c["excludes_zero"] else 0.0,
                1.0 if "yes" in r[5] else 0.0)
    check("D", "README splice table matches (%d rows)" % n_s,
          {k: v for k, v in readme_bad.items()}, {})

    # 表 5：relaxation
    n_r = 0
    for r in (t_relax or []):
        tag = r[0].replace("`", "")
        if tag not in CLAIMS["relaxation"]:
            continue
        c = CLAIMS["relaxation"][tag]
        if "rho_pooled" not in c:
            continue
        n_r += 1
        cmp_row(tag, "rho_before", CLAIMS["table"][tag]["rho_pooled"], num(r[1]))
        cmp_row(tag, "rho_after", c["rho_pooled"], num(r[2]))
        cmp_row(tag, "estep_agreement", c["estep_agreement"], num(r[3]))
        cmp_row(tag, "changed", c["changed"], num(r[4]))
    check("D", "README relaxation table matches (%d rows)" % n_r,
          {k: v for k, v in readme_bad.items()}, {})

    # 表 6：decision rule
    n_d = 0
    for r in (t_decision or []):
        mb = num(r[0])
        tag = r[1].replace("`", "")
        if tag not in CLAIMS["gauge_invariant"]:
            continue
        n_d += 1
        c = CLAIMS["gauge_invariant"][tag]
        cmp_row(tag, "delta", c["delta"], num(r[2]))
        ci = r[3].strip("[]").split(",")
        cmp_row(tag, "lo", c["lo"], num(ci[0]))
        cmp_row(tag, "hi", c["hi"], num(ci[1]))
    check("D", "README decision-rule table matches (%d rows)" % n_d,
          {k: v for k, v in readme_bad.items()}, {})

    # README prose / data table 中的 headline numbers
    head = {
        "cis records 91703": ("91703", str(CLAIMS["n_records"])),
        "same-bin 31827": ("31827", str(CLAIMS["samebin_records"])),
        "train 54420": ("54420", str(CLAIMS["n_train"])),
        "val 18155": ("18155", str(CLAIMS["n_val"])),
        "test 19128": ("19128", str(CLAIMS["n_test"])),
        "val pairs 1619": ("1619", str(CLAIMS["pairs_val"])),
        "test pairs 1622": ("1622", str(CLAIMS["pairs_test"])),
        "phased train 16928": ("16928", str(CLAIMS["n_ph_train"])),
        "acc 0.7903": ("0.7903", "%.4f" % CLAIMS["reference_accuracy"]["acc"]),
        "ref n 18150": ("18150", str(CLAIMS["reference_accuracy"]["n"])),
        "null tolerance 0.0220": ("0.0220", "%.4f" % CLAIMS["decision"]["null_tolerance"]),
    }
    readme_text = "\n".join(readme)
    head_bad = {k: v for k, v in head.items() if v[0] not in readme_text}
    check("D", "README headline numbers all present and consistent", head_bad, {})
    report["section_readme_bad"] = readme_bad

    # ---------------- 14. affected-pair readout（核心证据） --------
    print("\n## 14. affected-pair readout reproduced (masked_compare)")
    masked_bad = {}
    for tag, c in CLAIMS["masked_compare"].items():
        tb = tb_of(tag)
        aff = tb[tb1] != tb[tb2]
        idx = np.where(aff & fin["oracle"] & fin[tag])[0]
        idn = np.where(aff & fin["oracle"] & fin[tag] & fin["oracle_half1"]
                       & fin["oracle_half2"])[0]
        po, _ = pooled_prediction("oracle", tb1, tb2, mine["oracle"]["alpha"])
        pc, _ = pooled_prediction(tag, tb1, tb2, mine[tag]["alpha"])
        ph1, _ = pooled_prediction("oracle_half1", tb1, tb2, mine["oracle_half1"]["alpha"])
        ph2, _ = pooled_prediction("oracle_half2", tb1, tb2, mine["oracle_half2"]["alpha"])
        r_ref = spearman(po[idx], tC[idx])
        r_alt = spearman(pc[idx], tC[idx])
        lo, hi, _ = paired_bootstrap(tC[idx], po[idx], pc[idx])
        nd = nlo = nhi = float("nan")
        if len(idn) >= 30:
            nlo, nhi, _ = paired_bootstrap(tC[idn], ph1[idn], ph2[idn])
            nd = spearman(ph1[idn], tC[idn]) - spearman(ph2[idn], tC[idn])
        got = {"n": int(len(idx)), "n_common_with_null": int(len(idn)),
               "rho_oracle": r_ref, "rho_cand": r_alt, "delta": r_ref - r_alt,
               "lo": lo, "hi": hi, "null_delta_half_oracles": nd,
               "null_lo": nlo, "null_hi": nhi}
        for k, v in got.items():
            w = c[k]
            if isinstance(v, float) and isinstance(w, (int, float)):
                if not (abs(v - w) <= 1e-9 or (np.isnan(v) and np.isnan(w))):
                    masked_bad["%s/%s" % (tag, k)] = [v, w]
            elif v != w:
                masked_bad["%s/%s" % (tag, k)] = [v, w]
    check("D", "masked_compare reproduced for all %d candidates"
          % len(CLAIMS["masked_compare"]), masked_bad, {})
    report["section_masked_bad"] = masked_bad

    # ---------------- 15. selection test 与 decision rule ------------------
    print("\n## 15. selection test and pre-registered decision rule")
    sel = {}
    gi_vals = {cid: v["rho_gauge_avg"] for cid, v in gi_mine.items()}
    fam_relabel = [("oracle", ref_gi)] + list(gi_vals.items())
    fam_rewire = [("oracle", ref_gi)] + [(t, mine[t]["rho_pooled"]) for t in tags
                                         if t.endswith("_rewire")]
    fam_splice = [("oracle", mine["oracle"]["rho_pooled"])] + [
        (k, v["rho_pooled"]) for k, v in splice_res.items()]
    for fam, rows in (("relabel (gauge-invariant)", fam_relabel),
                      ("rewire", fam_rewire),
                      ("splice (label-free chimera)", fam_splice)):
        best = max(rows, key=lambda r: r[1])
        sel[fam] = {"best": best[0], "rho": best[1], "oracle_wins": best[0] == "oracle"}
        claim = CLAIMS["selection"][fam]
        check("D", "selection[%s] argmax" % fam,
              [best[0], round(best[1], 12), best[0] == "oracle"],
              [claim["best"], round(claim["rho"], 12), claim["oracle_wins"]],
              tol=1e-9)
    # null spread 和 tolerance
    null_ids = CLAIMS["null_spread"]["ids"]
    nr = np.array([mine[k]["rho_pooled"] for k in null_ids])
    check("D", "null spread rhos/sd/range",
          [round(float(nr.std(ddof=1)), 12), round(float(nr.max() - nr.min()), 12)],
          [round(CLAIMS["null_spread"]["sd"], 12),
           round(CLAIMS["null_spread"]["range"], 12)], tol=1e-9)
    check("D", "null_tolerance = max(null spread, gauge asymmetry)",
          round(max(float(nr.max() - nr.min()), asym_engine), 12),
          round(CLAIMS["decision"]["null_tolerance"], 12), tol=1e-9)
    scales = CLAIMS["decision"]["scales"]
    scales_bad = {}
    for s in scales:
        gi = gi_mine[s["id"]] if s["id"] != "oracle" else {"delta": 0.0, "lo": 0.0, "hi": 0.0}
        if abs(gi["delta"] - s["delta"]) > 1e-9 or abs(gi["lo"] - s["lo"]) > 1e-9 \
                or abs(gi["hi"] - s["hi"]) > 1e-9:
            scales_bad[s["id"]] = [gi, s]
    check("D", "decision scales match gauge-invariant deltas/CI", scales_bad, {})
    det = sorted([s["flipped_mb"] for s in scales if s["distinguishable"]])
    check("D", "min detectable flipped region (Mb)", det[0] if det else None,
          CLAIMS["decision"]["min_detectable_flipped_mb"])
    check("D", "half-data oracle max |delta|",
          round(max(abs(CLAIMS["bootstrap"][k]["delta"])
                    for k in ("oracle_half1", "oracle_half2")), 12),
          round(CLAIMS["decision"]["half_data_oracle_delta_max"], 12), tol=1e-9)
    report["section_selection"] = sel

    # ---------------- 16. *_relax.3dg 文件复现 relaxation 表 ------------
    print("\n## 16. relaxed coordinates reproduce the relaxation table")
    relax_bad = {}
    n_relax_checked = 0
    for tag, c in CLAIMS["relaxation"].items():
        if "rho_pooled" not in c:
            continue
        rtag = tag                        # relax 文件写在相同 id 下
        path = os.path.join(COORDS, rtag + "_relax.3dg")
        if not os.path.exists(path):
            relax_bad[tag] = "missing coords/%s_relax.3dg" % rtag
            continue
        n_relax_checked += 1
        a, _ = select_alpha(rtag + "_relax", vb1, vb2, vC)
        pred, ok = pooled_prediction(rtag + "_relax", tb1, tb2, a)
        idx = np.where(ok)[0]
        rho = spearman(pred[idx], tC[idx])
        if abs(rho - c["rho_pooled"]) > 1e-9:
            relax_bad[tag] = [rho, c["rho_pooled"]]
    check("D", "relaxed .3dg reproduce relaxation rho (%d checked)" % n_relax_checked,
          relax_bad, {})
    report["section_relax"] = {"n_checked": n_relax_checked, "bad": relax_bad}

    # ---------------- 17. boundary-relaxation E-step 重新推导 -----------
    print("\n## 17. boundary-relaxation E-step re-derived from the pre-relax coordinates")
    band = np.zeros(n_bins, dtype=bool)
    for g in [FRAG_BINS * k for k in range(1, nfrag)]:
        band[max(0, g - 2):min(n_bins, g + 3)] = True
    es_bad = {}
    n_es = 0
    for tag, c in CLAIMS["relaxation"].items():
        if "rho_pooled" not in c:
            continue
        rp1, rp2, assign = assign_for(tag)
        if tag in ("consensus", "random"):
            continue
        # 与该 candidate 自身 record set 对齐的真实 label array
        ltr = o_lb[halves[tag]] if tag in halves else o_lb
        inband = band[bin_of(rp1)] | band[bin_of(rp2)]
        st, tracks = candidate_parts(tag)
        d0 = distances(st, tracks[0], bin_of(rp1), bin_of(rp2))
        d1 = distances(st, tracks[1], bin_of(rp1), bin_of(rp2))
        ok = np.isfinite(d0) & np.isfinite(d1)
        sel = inband & ok
        new = (d1[sel] < d0[sel]).astype(assign.dtype)
        changed = int((new != assign[sel]).sum())
        agree = float((new == ltr[sel]).mean()) if sel.sum() else float("nan")
        n_es += 1
        # run.py 报告 n_in_band = |inband|，发生在 finite filter（sel）之前
        if changed != c["changed"] or abs(agree - c["estep_agreement"]) > 1e-12 \
                or int(inband.sum()) != c["n_in_band"]:
            es_bad[tag] = {"changed": [changed, c["changed"]],
                           "agreement": [agree, c["estep_agreement"]],
                           "n_in_band": [int(inband.sum()), c["n_in_band"]]}
        if int(sel.sum()) != int(inband.sum()):
            info.setdefault("relax_band_after_finite_filter", {})[tag] = [
                int(sel.sum()), int(inband.sum())]
    check("D", "relaxation changed/agreement/n_in_band re-derived (%d candidates)"
          % n_es, es_bad, {})
    report["section_estep"] = {"n_checked": n_es, "bad": es_bad}

    # ---------------- 汇总 ----------------------------------------------
    n_agree = sum(1 for c in CHECKS if c["agree"])
    n_total = len(CHECKS)
    bad = [c for c in CHECKS if not c["agree"]]
    print("\n" + "=" * 78)
    print("SUMMARY: %d/%d checks AGREE" % (n_agree, n_total))
    for c in bad:
        print("  DISAGREE [%s] %s: computed=%s claimed=%s"
              % (c["section"], c["check"], _f(c["computed"]), _f(c["claimed"])))
    print("elapsed %.1f s" % (time.time() - t0))

    report["summary"] = {"n_agree": n_agree, "n_total": n_total,
                         "disagreements": bad,
                         "elapsed_sec": time.time() - t0}
    with open(os.path.join(HERE, "audit_numbers.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("wrote %s" % os.path.join(HERE, "audit_numbers.json"))
    return 0 if n_total else 1


if __name__ == "__main__":
    sys.exit(main())
