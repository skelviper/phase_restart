#!/usr/bin/env python
"""第 1 阶段（chr1 片段交换）数值的独立重算。

本脚本有意不从 `pr/` 或 `run.py` 导入任何内容。它根据任务中给出的协议常量，自行实现所需步骤：pairs 解析、mixed-hash fold 分配、留出 bin-pair counts、reference-structure classifier、使用 validation 选择指数的留出 Spearman 得分、gauge（track-order）检查、结构层面的 splice 评估工具、七列审计和坐标哈希 gate 审计。

用法：
    source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh && conda activate analysis
    python test_res/001-20260912_120223-stage1-fragment-swap-chr1/audit/verify_independent.py
"""
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import rankdata

# --------------------------------------------------------------------------
# 常量（来自任务说明，而不是 pr/paths.py）
# --------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUN = os.path.join(ROOT, "test_res", "001-20260912_120223-stage1-fragment-swap-chr1")
PAIRS = os.path.join(ROOT, "data", "P9016.pairs.gz")
SNPFREE = os.path.join(ROOT, "inputs", "P9016.snpfree.pairs.gz")
REF3DG = os.path.join(ROOT, "data", "P9016.1m.3dg.gz")
COORDS = os.path.join(RUN, "coords")
WORK = os.path.join(RUN, "work")
GATE = os.path.join(RUN, "gate.json")
RESULTS = os.path.join(RUN, "results.json")

CHROM = "chr1"
BIN = 1_000_000
OFF = 3_000_000
FRAG_BINS = 20
ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
FLOOR = 1e-3
TRAIN_FOLDS = (0, 1, 2, 3, 4, 5)
VAL_FOLDS = (6, 7)
TEST_FOLDS = (8, 9)

REPORT = []          # 记录格式：(check_id, claimed, derived, verdict)
DETAIL = {}


def emit(check, claimed, derived, verdict, note=""):
    REPORT.append((check, claimed, derived, verdict))
    line = "[%s] %-34s claimed=%-12s derived=%-12s %s" % (
        verdict, check, claimed, derived, note)
    print(line)
    sys.stdout.flush()

# --------------------------------------------------------------------------
# 基础检查
# --------------------------------------------------------------------------
def bin_of(p):
    return (np.asarray(p, dtype=np.int64) - OFF) // BIN


def fold_of(i, j):
    """协议哈希，uint64 回绕，向量化实现。"""
    i = np.asarray(i, dtype=np.uint64)
    j = np.asarray(j, dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = (i * np.uint64(0x9E3779B1)) ^ (j * np.uint64(0x85EBCA77))
        h = h ^ (i * j + np.uint64(0x165667B1))
        h = h ^ (h >> np.uint64(15))
        h = h * np.uint64(0x2545F491)
        h = h ^ (h >> np.uint64(13))
    return (h % np.uint64(10)).astype(np.int64)


def spearman(x, y):
    """自计算的 Spearman（平均 rank、总体标准差）；任一侧为常量时返回 NaN。"""
    rx = rankdata(x)
    ry = rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def predictor(d, alpha):
    return np.maximum(np.asarray(d, dtype=float), FLOOR) ** (-alpha)


def select_alpha(dist_parts, counts):
    """在 validation fold 上选择指数；按指定规则，首次达到严格最大值者胜出。"""
    best, best_a = -np.inf, ALPHAS[0]
    for a in ALPHAS:
        p = np.zeros(len(counts))
        for d in dist_parts:
            p = p + predictor(d, a)
        r = spearman(p, counts)
        if np.isfinite(r) and r > best:
            best, best_a = r, a
    return best_a, best


def read_3dg(path):
    """返回 {track: {pos: xyz}}；支持 gz 或普通文件。"""
    op = gzip.open if path.endswith(".gz") else open
    d = {}
    with op(path, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            a = line.split()
            if len(a) < 5:
                continue
            d.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
    return d


def pair_dist(m, b1, b2):
    """计算轨迹 `m` 中两个 bin 起点 OFF+b*BIN 的距离；缺少 bead 时为 NaN。"""
    out = np.full(len(b1), np.nan)
    for k, (x, y) in enumerate(zip(b1, b2)):
        va = m.get(int(OFF + int(x) * BIN))
        vb = m.get(int(OFF + int(y) * BIN))
        if va is None or vb is None:
            continue
        out[k] = float(np.linalg.norm(va - vb))
    return out


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
# 1. 原始 pairs -> chr1 cis records
# --------------------------------------------------------------------------
def load_chr1_cis(path, ncol):
    """返回 (pos1 sorted, pos2 sorted, label, n_discordant, n_both_filled)。label：'00' 为 0，'11' 为 1；任一列为 '.' 或两列不一致（'01'/'10'）时为 -1。"""
    p1, p2, lb = [], [], []
    n_disc = n_both = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.rstrip("\n").split("\t")
            if c[1] != c[3] or c[1] != CHROM:
                continue
            a, b = int(c[2]), int(c[4])
            if a > b:
                a, b = b, a
            p1.append(a)
            p2.append(b)
            if ncol == 9:
                ph = c[7] + c[8]
                if ph in ("00", "11"):
                    n_both += 1
                    lb.append(0 if ph == "00" else 1)
                else:
                    if c[7] != "." and c[8] != ".":
                        n_disc += 1
                    lb.append(-1)
            else:
                lb.append(-1)
    return (np.asarray(p1, dtype=np.int64), np.asarray(p2, dtype=np.int64),
            np.asarray(lb, dtype=np.int8), n_disc, n_both)


def unique_pairs_counts(i, j, keep):
    """在布尔 `keep` 条件下返回唯一的 (i<j) bin pairs 及其记录 counts。"""
    ii, jj = i[keep], j[keep]
    sel = ii != jj
    ii, jj = ii[sel], jj[sel]
    key = ii.astype(np.int64) * 1_000_000 + jj
    uk, cnt = np.unique(key, return_counts=True)
    return (uk // 1_000_000).astype(np.int64), (uk % 1_000_000).astype(np.int64), cnt.astype(float)


def main():
    print("=" * 78)
    print("independent re-derivation -- no pr/ or run.py imports")
    print("root:", ROOT)
    print("=" * 78)

    res = json.load(open(RESULTS))
    # phased-train count 在日志中声明；results.json（由运行写出）根本不包含该 key——此处作为 provenance 记录。
    log_text = open(os.path.join(RUN, "logs", "run.log")).read()
    m = re.search(r"phased train records (\d+) of (\d+) train records", log_text)
    claim_ph_train = int(m.group(1)) if m else None
    claim_train_log = int(m.group(2)) if m else None
    has_key = "n_ph_train" in res
    print("provenance: results.json has n_ph_train=%s (log claims %s);"
          " run.py line 598 does write it -> results.json predates the current run.py"
          % (has_key, claim_ph_train))
    claims = {
        "n_records": res["n_records"], "samebin": res["samebin_records"],
        "n_train": res["n_train"], "n_val": res["n_val"], "n_test": res["n_test"],
        "n_ph_train": claim_ph_train, "pairs_val": res["pairs_val"],
        "pairs_test": res["pairs_test"], "acc": res["reference_accuracy"]["acc"],
        "tie": res["reference_accuracy"]["tie_rate"],
        "n_acc": res["reference_accuracy"]["n"],
        "rho_oracle": res["table"]["oracle"]["rho_pooled"],
        "alpha_oracle": res["table"]["oracle"]["alpha"],
        "rho_gauge": res["table"]["gauge_all"]["rho_pooled"],
        "alpha_gauge": res["table"]["gauge_all"]["alpha"],
        "gauge_asym": res["gauge_invariant"]["_reference"]["gauge_asymmetry"],
        "splice_wall1": res["splice"]["splice_wall1"]["rho_pooled"],
        "splice_rmsd": res["splice"]["splice_wall1"]["procrustes_rmsd"],
    }

    # ---------------- 检查 1：chr1 cis records ---------------------------
    p1, p2, lb, n_disc, n_both = load_chr1_cis(PAIRS, ncol=9)
    print("\n-- check 1: chr1 cis records in data/P9016.pairs.gz --")
    same_bin = bin_of(p1) == bin_of(p2)
    n_ph = int((lb >= 0).sum())
    n00, n11 = int((lb == 0).sum()), int((lb == 1).sum())

    q1_raw = load_chr1_cis(SNPFREE, ncol=7)
    ident = (np.array_equal(q1_raw[0], p1) and np.array_equal(q1_raw[1], p2))
    print("  snpfree file has identical chr1 cis positions and order: %s" % ident)
    print("  records with BOTH phase columns filled (incl. discordant): %d" % n_both)
    print("  discordant phase pairs (01/10): %d" % n_disc)
    print("  label counts n00=%d n11=%d n_unphased(incl. discordant)=%d" % (n00, n11, int((lb < 0).sum())))
    emit("1a chr1 cis records", claims["n_records"], len(p1),
         "AGREE" if len(p1) == claims["n_records"] else "DISAGREE")
    emit("1b same-bin records", claims["samebin"], int(same_bin.sum()),
         "AGREE" if int(same_bin.sum()) == claims["samebin"] else "DISAGREE")
    emit("1c fully phased (00/11)", "n00+n11=%d" % (res["label_counts"]["n00"] + res["label_counts"]["n11"]),
         n_ph, "AGREE" if n_ph == res["label_counts"]["n00"] + res["label_counts"]["n11"] else "DISAGREE")
    emit("1d phase-label counts", "n00=%d n11=%d" % (res["label_counts"]["n00"], res["label_counts"]["n11"]),
         "n00=%d n11=%d" % (n00, n11),
         "AGREE" if (n00 == res["label_counts"]["n00"] and n11 == res["label_counts"]["n11"]) else "DISAGREE")
    DETAIL["check1"] = {"n_records": int(len(p1)), "same_bin": int(same_bin.sum()),
                        "n_fully_phased": n_ph, "n00": n00, "n11": n11,
                        "both_filled": int(n_both), "discordant": int(n_disc)}

    # ---------------- 检查 2：folds ---------------------------------------
    I1, I2 = bin_of(p1), bin_of(p2)
    f = fold_of(I1, I2)
    tr = np.isin(f, TRAIN_FOLDS)
    va = np.isin(f, VAL_FOLDS)
    te = np.isin(f, TEST_FOLDS)
    ph_tr = tr & (lb >= 0)
    print("\n-- check 2: folds --")
    print("  train/val/test = %d/%d/%d ; phased train = %d"
          % (tr.sum(), va.sum(), te.sum(), ph_tr.sum()))
    for name, cl, got in (("2a train records", claims["n_train"], int(tr.sum())),
                          ("2b val records", claims["n_val"], int(va.sum())),
                          ("2c test records", claims["n_test"], int(te.sum())),
                          ("2d phased train records", claims["n_ph_train"], int(ph_tr.sum()))):
        emit(name, cl, got, "AGREE" if cl == got else "DISAGREE")
    DETAIL["check2"] = {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                        "phased_train": int(ph_tr.sum())}

    vb1, vb2, vC = unique_pairs_counts(I1, I2, va)
    tb1, tb2, tC = unique_pairs_counts(I1, I2, te)
    print("\n-- check 3: held-out unique bin pairs (same-bin excluded) --")
    emit("3a val bin pairs", claims["pairs_val"], len(vb1),
         "AGREE" if len(vb1) == claims["pairs_val"] else "DISAGREE")
    emit("3b test bin pairs", claims["pairs_test"], len(tb1),
         "AGREE" if len(tb1) == claims["pairs_test"] else "DISAGREE")
    DETAIL["check3"] = {"val_pairs": int(len(vb1)), "test_pairs": int(len(tb1)),
                        "val_contacts": float(vC.sum()), "test_contacts": float(tC.sum())}
    # 为完整性记录各 fold 的 same-bin records
    sb = same_bin
    DETAIL["samebin_by_fold"] = {"train": int((sb & tr).sum()), "val": int((sb & va).sum()),
                                 "test": int((sb & te).sum())}

    # ---------------- 检查 4：reference accuracy --------------------------
    refc = read_3dg(REF3DG)
    d_pat = pair_dist(refc["%s(pat)" % CHROM], I1, I2)
    d_mat = pair_dist(refc["%s(mat)" % CHROM], I1, I2)
    keep = (~same_bin) & (lb >= 0) & np.isfinite(d_pat) & np.isfinite(d_mat)
    dp, dm, ll = d_pat[keep], d_mat[keep], lb[keep]
    tie = dp == dm
    n_acc = int(keep.sum())
    # 方向 0：label 0 -> pat，label 1 -> mat
    hit0 = np.where(ll == 0, dp < dm, dm < dp)
    # 方向 1：label 0 -> mat，label 1 -> pat
    hit1 = np.where(ll == 0, dm < dp, dp < dm)
    acc0, acc1 = float(hit0.mean()), float(hit1.mean())
    print("\n-- check 4: reference-structure classification (chr1 cis, same-bin excluded) --")
    print("  n=%d  ties=%d (rate %.6f)" % (n_acc, int(tie.sum()), float(tie.mean())))
    print("  acc(orient0: 00->pat, 11->mat)=%.6f   acc(orient1: 00->mat, 11->pat)=%.6f"
          % (acc0, acc1))
    print("  n_samebin_excluded (records dropped as same-bin) = %d" % int(same_bin.sum()))
    emit("4a reference accuracy", "%.4f" % claims["acc"], "%.6f" % max(acc0, acc1),
         "AGREE" if abs(max(acc0, acc1) - claims["acc"]) < 5e-4 else "DISAGREE")
    emit("4b tie rate", "%.4f" % claims["tie"], "%.6f" % float(tie.mean()),
         "AGREE" if abs(float(tie.mean()) - claims["tie"]) < 5e-4 else "DISAGREE")
    emit("4c classification n", claims["n_acc"], n_acc,
         "AGREE" if n_acc == claims["n_acc"] else "DISAGREE")
    # 运行自身的 orientation loop 不依赖 `orient`；同时报告正确计算的 orientation-1 accuracy。
    run_orient0 = res["reference_accuracy"]["acc_orient0"]
    run_orient1 = res["reference_accuracy"]["acc_orient1"]
    print("  run's acc_orient0=%.6f acc_orient1=%.6f orientation=%d"
          % (run_orient0, run_orient1, res["reference_accuracy"]["orientation"]))
    print("  NOTE: the run's two orientation branches use the same (own,other) pair,"
          " so acc_orient1 is a copy of acc_orient0; my orient1 = 1 - acc0 (ties=0)"
          " = %.6f. Orientation 0 is nevertheless the better one, so acc is unaffected."
          % acc1)
    DETAIL["check4"] = {"n": n_acc, "ties": int(tie.sum()), "tie_rate": float(tie.mean()),
                        "acc_orient0": acc0, "acc_orient1": acc1,
                        "run_acc_orient0": run_orient0, "run_acc_orient1": run_orient1,
                        "run_orientation": res["reference_accuracy"]["orientation"]}

    # ---------------- 评分辅助函数 --------------------------------------
    # 运行在 MASK_IDX["all"] 上为每个 candidate 评分，该集合是
    # 所有 candidate 有限距离掩码的交集。两个诊断 candidate
    #（oracle_half1/half2，即半数据 refit）可能缺少 beads，从而静默
    # 缩小该全集。此处根据 results.json 自身声明的 candidate list 复现这一行为。
    cand_ids = list(res["table"].keys())
    run_mask = np.ones(len(tb1), dtype=bool)
    nan_report = {}
    for cid in cand_ids:
        s = read_3dg(os.path.join(COORDS, cid + ".3dg"))
        ok = np.ones(len(tb1), dtype=bool)
        for trk in s:
            ok &= np.isfinite(pair_dist(s[trk], tb1, tb2))
        if not ok.all():
            nan_report[cid] = int((~ok).sum())
        run_mask &= ok
    print("\n-- run scoring universe --")
    print("  candidates declared in results.json: %d" % len(cand_ids))
    print("  candidates with non-finite test distances: %s" % (nan_report or "none"))
    print("  run's MASK_IDX['all'] would be %d of %d test pairs"
          " (results.json stratum_sizes.all=%d, log says %d)"
          % (int(run_mask.sum()), len(tb1), res["stratum_sizes"]["all"], claims["pairs_test"]))
    for cid in nan_report:
        s = read_3dg(os.path.join(COORDS, cid + ".3dg"))
        want = set(int(OFF + b * BIN) for b in range(int(max(I1.max(), I2.max())) + 1))
        for trk in sorted(s):
            miss = sorted(want - set(s[trk]))
            print("    %s/%s: %d beads, missing bin starts=%d %s"
                  % (cid, trk, len(s[trk]), len(miss), miss[:5]))

    orc = read_3dg(os.path.join(COORDS, "oracle.3dg"))
    gau = read_3dg(os.path.join(COORDS, "gauge_all.3dg"))
    Oa = pair_dist(orc["cc00a"], tb1, tb2)
    Ob = pair_dist(orc["cc00b"], tb1, tb2)
    Ga = pair_dist(gau["cc00a"], tb1, tb2)
    Gb = pair_dist(gau["cc00b"], tb1, tb2)
    V_Oa = pair_dist(orc["cc00a"], vb1, vb2)
    V_Ob = pair_dist(orc["cc00b"], vb1, vb2)
    V_Ga = pair_dist(gau["cc00a"], vb1, vb2)
    V_Gb = pair_dist(gau["cc00b"], vb1, vb2)
    for nm, arr in (("Oa", Oa), ("Ob", Ob), ("Ga", Ga), ("Gb", Gb),
                    ("V_Oa", V_Oa), ("V_Ob", V_Ob), ("V_Ga", V_Ga), ("V_Gb", V_Gb)):
        assert np.isfinite(arr).all(), (nm, int((~np.isfinite(arr)).sum()))

    a_orc, rv_orc = select_alpha([V_Oa, V_Ob], vC)
    a_gau, rv_gau = select_alpha([V_Ga, V_Gb], vC)
    rho_orc = spearman(predictor(Oa, a_orc) + predictor(Ob, a_orc), tC)
    rho_gau = spearman(predictor(Ga, a_gau) + predictor(Gb, a_gau), tC)
    rho_orc_run = spearman(predictor(Oa[run_mask], a_orc) + predictor(Ob[run_mask], a_orc), tC[run_mask])
    rho_gau_run = spearman(predictor(Ga[run_mask], a_gau) + predictor(Gb[run_mask], a_gau), tC[run_mask])
    print("\n-- validation-selected exponents --")
    print("  oracle   : alpha=%.2f (val rho %.4f) -> test rho %.6f on all %d pairs,"
          " %.6f on the run's %d-pair universe  [run %.4f]"
          % (a_orc, rv_orc, rho_orc, len(tb1), rho_orc_run, int(run_mask.sum()), claims["rho_oracle"]))
    print("  gauge_all: alpha=%.2f (val rho %.4f) -> test rho %.6f on all %d pairs,"
          " %.6f on the run's %d-pair universe  [run %.4f]"
          % (a_gau, rv_gau, rho_gau, len(tb1), rho_gau_run, int(run_mask.sum()), claims["rho_gauge"]))
    emit("5a oracle alpha", "%.2f" % claims["alpha_oracle"], "%.2f" % a_orc,
         "AGREE" if abs(a_orc - claims["alpha_oracle"]) < 1e-9 else "DISAGREE")
    emit("5b gauge_all alpha", "%.2f" % claims["alpha_gauge"], "%.2f" % a_gau,
         "AGREE" if abs(a_gau - claims["alpha_gauge"]) < 1e-9 else "DISAGREE")
    emit("5c oracle held-out rho", "%.4f" % claims["rho_oracle"],
         "%.6f (1622) / %.6f (1603)" % (rho_orc, rho_orc_run),
         "AGREE*" if abs(rho_orc_run - claims["rho_oracle"]) < 5e-4 else "DISAGREE",
         "matches only on the run's reduced universe")
    emit("5d gauge_all held-out rho", "%.4f" % claims["rho_gauge"],
         "%.6f (1622) / %.6f (1603)" % (rho_gau, rho_gau_run),
         "AGREE*" if abs(rho_gau_run - claims["rho_gauge"]) < 5e-4 else "DISAGREE",
         "matches only on the run's reduced universe")

    # ---------------- 检查 5：gauge / track-order ------------------------
    print("\n-- check 5: gauge (track-order) check, oracle vs gauge_all --")

    def rho_of(pa, pb, alpha=a_orc):
        return spearman(predictor(pa, alpha) + predictor(pb, alpha), tC)

    # 文件内交换：two-copy predictor 对其两个参数对称
    same_file_swap = abs(rho_of(Oa, Ob) - rho_of(Ob, Oa))
    same_file_swap_g = abs(rho_of(Ga, Gb) - rho_of(Gb, Ga))
    # 单拷贝读出，分别检查两个 orientation
    single = {}
    for nm, d in (("O_cc00a", Oa), ("O_cc00b", Ob), ("G_cc00a", Ga), ("G_cc00b", Gb)):
        single[nm] = spearman(predictor(d, a_orc), tC)
    scale = {nm: (float(np.mean(d)), float(np.min(d)), float(np.max(d)))
             for nm, d in (("O_cc00a", Oa), ("O_cc00b", Ob), ("G_cc00a", Ga), ("G_cc00b", Gb))}
    # 跨文件：matched pairing（oracle cc00a <-> gauge_all cc00b）与同名 pairing
    combos = {
        "O_a+G_a (same-name)": rho_of(Oa, Ga),
        "O_a+G_b (matched)": rho_of(Oa, Gb),
        "O_b+G_a (matched)": rho_of(Ob, Ga),
        "O_b+G_b (same-name)": rho_of(Ob, Gb),
    }
    # 距离矩阵对应关系，比较 matched 与 swapped
    dm_matched = spearman(Oa, Gb)
    dm_swapped = spearman(Oa, Ga)
    run_asym = abs(claims["rho_oracle"] - claims["rho_gauge"])
    my_asym = abs(rho_orc - rho_gau)
    my_asym_run = abs(rho_orc_run - rho_gau_run)
    print("  within-file track swap (predictor is symmetric):"
          " |d rho| oracle=%.3e gauge_all=%.3e" % (same_file_swap, same_file_swap_g))
    for k, v in single.items():
        print("  single-copy rho (d^-a alone) %-8s = %.6f   mean dist %.2f" % (k, v, scale[k][0]))
    for k, v in combos.items():
        print("  cross-file mixture %-22s rho=%.6f" % (k, v))
    print("  distance-matrix Spearman  O_cc00a vs G_cc00b (matched) = %.6f" % dm_matched)
    print("  distance-matrix Spearman  O_cc00a vs G_cc00a (swapped) = %.6f" % dm_swapped)
    print("  run's gauge asymmetry |rho(oracle)-rho(gauge_all)| = %.6f ; mine = %.6f (1622),"
          " %.6f (1603)" % (run_asym, my_asym, my_asym_run))
    print("  -> the entire track-order effect lives in the two FDG FITS, not in the scoring:"
          " with the pairing matched (O_a vs G_b) the two distance matrices agree at"
          " rho=%.4f, mismatched (O_a vs G_a) they agree at only rho=%.4f."
          % (dm_matched, dm_swapped))
    # 检查 gauge_all 是否确实为 oracle assignment 的 complement
    def assign_of(path):
        out = {}
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line[0] == "#":
                    continue
                c = line.rstrip("\n").split("\t")
                out[(int(c[2]), int(c[4]))] = c[1]
        return out
    ao, ag = assign_of(os.path.join(WORK, "oracle.pairs.gz")), assign_of(os.path.join(WORK, "gauge_all.pairs.gz"))
    keys_same = set(ao) == set(ag)
    diff_frac = float(np.mean([ao[k] != ag[k] for k in ao])) if keys_same else float("nan")
    print("  work/oracle.pairs.gz vs work/gauge_all.pairs.gz: same record keys=%s,"
          " fraction with a different track label=%.6f" % (keys_same, diff_frac))
    emit("5e gauge asymmetry", "%.4f" % claims["gauge_asym"],
         "%.6f (1622) / %.6f (1603)" % (my_asym, my_asym_run),
         "AGREE*" if abs(my_asym_run - claims["gauge_asym"]) < 5e-4 else "DISAGREE",
         "matches on the run's universe")
    emit("5f within-file track order effect", "0 (not reported)", "%.3e" % max(same_file_swap, same_file_swap_g),
         "AGREE" if max(same_file_swap, same_file_swap_g) == 0 else "DISAGREE",
         "(predictor symmetric: track order inside one file has literally no effect)")
    # 将自己的 Spearman 与 scipy 在 oracle prediction 上的结果交叉核对
    try:
        from scipy.stats import spearmanr
        sc = float(spearmanr(predictor(Oa, a_orc) + predictor(Ob, a_orc), tC).statistic)
        print("  scipy.spearmanr cross-check on the oracle prediction: %.12f (mine %.12f)"
              % (sc, rho_orc))
    except Exception as exc:                                   # pragma: no cover
        sc = None
        print("  scipy cross-check failed: %r" % (exc,))
    DETAIL["check5"] = {"rho_oracle": rho_orc, "rho_gauge": rho_gau,
                        "rho_oracle_run_mask": rho_orc_run, "rho_gauge_run_mask": rho_gau_run,
                        "run_mask_n": int(run_mask.sum()), "nan_candidates": nan_report,
                        "within_file_swap": [same_file_swap, same_file_swap_g],
                        "cross_file": combos, "dist_matched": dm_matched,
                        "dist_swapped": dm_swapped, "labels_differ_frac": diff_frac,
                        "same_keys": bool(keys_same), "gauge_asym_1622": my_asym,
                        "gauge_asym_1603": my_asym_run, "single_copy_rho": single,
                        "mean_dist": {k: v[0] for k, v in scale.items()},
                        "scipy_rho_oracle": sc}

    # ---------------- 检查 6：splice 评估工具 ------------------------------
    print("\n-- check 6: splice instrument (20 Mb wall), independent implementation --")

    def kabsch(src, dst):
        mu_s, mu_d = src.mean(0), dst.mean(0)
        A = (src - mu_s).T @ (dst - mu_d)
        U, _, Vt = np.linalg.svd(A)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        return R, mu_d - R @ mu_s

    pos = sorted(set(orc["cc00a"]) & set(orc["cc00b"]))
    src = np.array([orc["cc00b"][p] for p in pos])
    dst = np.array([orc["cc00a"][p] for p in pos])
    R, t = kabsch(src, dst)
    rmsd = float(np.sqrt((((src @ R.T + t) - dst) ** 2).sum(1).mean()))
    b_align = {p: R @ v + t for p, v in orc["cc00b"].items()}
    # 角色交换：第二次独立的 Kabsch solve，cc00a -> cc00b
    src2 = np.array([orc["cc00a"][p] for p in pos])
    dst2 = np.array([orc["cc00b"][p] for p in pos])
    R2, t2 = kabsch(src2, dst2)
    rmsd2 = float(np.sqrt((((src2 @ R2.T + t2) - dst2) ** 2).sum(1).mean()))
    a_align = {p: R2 @ v + t2 for p, v in orc["cc00a"].items()}
    block = [OFF + b * BIN for b in range(0, 20)]           # bins 0..19 = 20 Mb 分界墙
    A_chim = {p: v.copy() for p, v in orc["cc00a"].items()}
    for p in block:
        if p in b_align:
            A_chim[p] = b_align[p].copy()
    B_chim = {p: v.copy() for p, v in orc["cc00b"].items()}
    for p in block:
        if p in a_align:
            B_chim[p] = a_align[p].copy()
    dA = pair_dist(A_chim, tb1, tb2)
    dB = pair_dist(B_chim, tb1, tb2)
    assert np.isfinite(dA).all() and np.isfinite(dB).all()
    rho_splice = spearman(predictor(dA, a_orc) + predictor(dB, a_orc), tC)
    rho_splice_run = spearman(predictor(dA[run_mask], a_orc) + predictor(dB[run_mask], a_orc),
                              tC[run_mask])
    print("  Kabsch cc00b->cc00a on %d common beads: rmsd=%.9f (run %.9f)"
          % (len(pos), rmsd, claims["splice_rmsd"]))
    print("  Kabsch cc00a->cc00b (roles exchanged): rmsd=%.9f" % rmsd2)
    print("  chimera scored at the oracle's validation alpha %.2f" % a_orc)
    print("  splice_wall1 rho=%.6f on all %d pairs, %.6f on the run's %d pairs (run %.4f)"
          % (rho_splice, len(tb1), rho_splice_run, int(run_mask.sum()), claims["splice_wall1"]))
    print("  oracle rho=%.6f (1622) / %.6f (1603) (run %.4f) | delta_vs_oracle=%.6f (1603)"
          % (rho_orc, rho_orc_run, claims["rho_oracle"],
             rho_orc_run - rho_splice_run))
    # 检查 splice 数值对指数选择的敏感性
    sweep = {}
    for a in ALPHAS:
        da = pair_dist(A_chim, tb1, tb2)
        db = pair_dist(B_chim, tb1, tb2)
        sweep[a] = spearman(predictor(da[run_mask], a) + predictor(db[run_mask], a), tC[run_mask])
    print("  splice_wall1 rho over the alpha grid (run's %d-pair universe): " % int(run_mask.sum()) +
          " ".join("%.2f:%.4f" % (a, sweep[a]) for a in ALPHAS))
    print("  best-on-test alpha would give %.4f at a=%.2f (not the protocol's choice)"
          % (max(sweep.values()), max(sweep, key=sweep.get)))
    emit("6a splice_wall1 rho", "%.4f" % claims["splice_wall1"],
         "%.6f (1622) / %.6f (1603)" % (rho_splice, rho_splice_run),
         "AGREE*" if abs(rho_splice_run - claims["splice_wall1"]) < 5e-4 else "DISAGREE",
         "matches on the run's reduced universe")
    emit("6b procrustes rmsd", "%.9f" % claims["splice_rmsd"], "%.9f" % rmsd,
         "AGREE" if abs(rmsd - claims["splice_rmsd"]) < 1e-6 else "DISAGREE")
    emit("6c oracle rho (splice reference)", "%.4f" % claims["rho_oracle"],
         "%.6f (1622) / %.6f (1603)" % (rho_orc, rho_orc_run),
         "AGREE*" if abs(rho_orc_run - claims["rho_oracle"]) < 5e-4 else "DISAGREE")
    DETAIL["check6"] = {"rmsd": rmsd, "rmsd_reverse": rmsd2, "rho_splice_wall1": rho_splice,
                        "rho_splice_wall1_run_mask": rho_splice_run, "rho_oracle": rho_orc,
                        "rho_oracle_run_mask": rho_orc_run, "alpha_used": a_orc,
                        "alpha_sweep": sweep}

    # ---------------- 检查 7：七列 ----------------------------------------
    print("\n-- check 7: seven-column audit of work/*.pairs.gz and coords/*.3dg --")
    EXPECT = ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"]
    violations = []
    n_work = n_coords = 0
    for fn in sorted(os.listdir(WORK)):
        if not fn.endswith(".pairs.gz"):
            continue
        n_work += 1
        path = os.path.join(WORK, fn)
        col_line = None
        bad_cols = bad_ncol = 0
        blob = []
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line[0] == "#":
                    if line.startswith("#columns"):
                        col_line = line.rstrip("\n").split(":", 1)[1]
                    continue
                n = line.count("\t") + 1
                if n != 7:
                    bad_ncol += 1
                blob.append(line)
        text = "".join(blob)
        if col_line is None:
            violations.append((fn, "no #columns line"))
        else:
            cols = col_line.split("\t")
            if cols != EXPECT:
                violations.append((fn, "#columns=%r" % cols))
        if bad_ncol:
            violations.append((fn, "%d data lines without exactly 7 fields" % bad_ncol))
        if re.search(r"phase|prob", text, re.I):
            violations.append((fn, "body matches /phase|prob/i"))
    for fn in sorted(os.listdir(COORDS)):
        if not fn.endswith(".3dg"):
            continue
        n_coords += 1
        with open(os.path.join(COORDS, fn), "r", errors="replace") as fh:
            text = fh.read()
        if re.search(r"phase|prob", text, re.I):
            violations.append((fn, "matches /phase|prob/i"))
    print("  scanned %d work/*.pairs.gz and %d coords/*.3dg files" % (n_work, n_coords))
    print("  violations: %s" % (violations if violations else "none"))
    emit("7 columns clean", "0 violations", "%d violations" % len(violations),
         "AGREE" if not violations else "DISAGREE", str(violations[:5]))
    DETAIL["check7"] = {"n_work": n_work, "n_coords": n_coords, "violations": violations}

    # ---------------- 检查 8：gate.json 哈希 -------------------------------
    print("\n-- check 8: gate.json sha256 coverage --")
    gate = json.load(open(GATE))
    by_path = {}
    for e in gate:
        p = e["path"] if os.path.isabs(e["path"]) else os.path.join(ROOT, e["path"])
        by_path[os.path.abspath(p)] = e["sha256"]
    files = sorted(f for f in os.listdir(COORDS) if f.endswith(".3dg"))
    missing, mismatch, ok = [], [], []
    for fn in files:
        p = os.path.abspath(os.path.join(COORDS, fn))
        rec = by_path.get(p)
        if rec is None:
            missing.append(fn)
            continue
        got = sha256_file(p)
        if got == rec:
            ok.append(fn)
        else:
            mismatch.append((fn, rec, got))
    extra = [p for p in by_path if os.path.basename(p) not in files]
    print("  gate.json entries=%d ; .3dg files on disk=%d" % (len(gate), len(files)))
    print("  hashed and matching: %d ; missing from gate.json: %d ; hash mismatch: %d ; stale entries: %d"
          % (len(ok), len(missing), len(mismatch), len(extra)))
    if missing:
        print("  MISSING (first 10): %s" % missing[:10])
    if extra:
        print("  STALE: %s" % extra)
    verdict = "AGREE" if (not missing and not mismatch and not extra) else "DISAGREE"
    emit("8 gate.json hashes", "sha256 for every .3dg", "%d/%d present, %d match" %
         (len(ok), len(files), len(ok) - len(mismatch)), verdict)
    DETAIL["check8"] = {"entries": len(gate), "files": len(files), "ok": len(ok),
                        "missing": missing, "mismatch": [m[0] for m in mismatch],
                        "stale": extra}

    # ---------------- 汇总 ----------------------------------------------
    print("\n" + "=" * 78)
    agree = sum(1 for r in REPORT if r[3] == "AGREE")
    cond = sum(1 for r in REPORT if r[3] == "AGREE*")
    bad = [r for r in REPORT if r[3] == "DISAGREE"]
    print("SUMMARY: %d/%d sub-checks AGREE, %d AGREE only on the run's reduced"
          " 1603-pair universe, %d DISAGREE" % (agree, len(REPORT), cond, len(bad)))
    for cid, cl, de, v in REPORT:
        if v not in ("AGREE",):
            print("  %-9s %s claimed=%s derived=%s" % (v, cid, cl, de))
    print("=" * 78)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "verify_independent.json"), "w") as fh:
        json.dump({"report": REPORT, "detail": DETAIL, "claims": claims}, fh, indent=2)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
