"""面向评估器、受 gate 保护的现有坐标 reference 读出。

R1 使用每条染色体的几何 gauge，而不是真实 labels，决定 candidate copy ``a`` 对应 reference 的 ``mat`` 还是 ``pat``。gauge 在四条轨迹共同有限、非对角的 bin 对集合上计算。几何平局不根据 truth 解决：R1 取两种方向的均值（因此不受 gauge 影响）；几何信息不足时，主要值不可用。

R2 使用相同的四条轨迹和共同 mask 比较。单轨 consensus 只是结构基线，不是退化的双拷贝重建。R3 保留原有的 20 Mb 局部 label 定义，但缺失或平局片段会打断 run，并从报告的分母中排除。
"""
import gzip

import numpy as np

from .genome import track
from .paths import BIN, OFF, PAIRS, REF3DG
from .score import spearman

STAGE = "eval"
REF_SUFFIX = ("mat", "pat")
FRAG_BINS = 20
MIN_RHO_PAIRS = 20
GEOMETRY_TIE_TOL = 1e-12


# --------------------------------------------------------------------------
# label（受 gate 保护）
# --------------------------------------------------------------------------
def _records(path=PAIRS):
    with gzip.open(path, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            yield line.rstrip("\n").split("\t")


def load_labels_two(gate, contacts, pairs_path=PAIRS):
    """返回与规范化 SNP-free records 对齐的逐端 phase labels。

    ``phase0`` 和 ``phase1`` 分别属于端点 1 和端点 2。如果 cis 行将 ``pos1 > pos2`` 按数值规范化为 ``pos1 <= pos2``，phase 字段也会同步交换。当前正式 P9016 文件中没有此类行，但 loader 仍须正确处理合法的重排输入。
    """
    gate.require(STAGE)
    n = len(contacts["ci"])
    a1 = np.full(n, -1, dtype=np.int8)
    a2 = np.full(n, -1, dtype=np.int8)
    names = contacts["names"]
    idx = {c: i for i, c in enumerate(names)}
    k = 0
    for c in _records(pairs_path):
        if len(c) < 9:
            raise RuntimeError("phase-bearing pairs row %d has fewer than nine columns" % k)
        a, b = idx[c[1]], idx[c[3]]
        x, y = int(c[2]), int(c[4])
        phase0, phase1 = c[7], c[8]
        if a == b and x > y:
            x, y = y, x
            phase0, phase1 = phase1, phase0
        if (a != contacts["ci"][k] or x != contacts["p1"][k]
                or b != contacts["cj"][k] or y != contacts["p2"][k]):
            raise RuntimeError("label records do not line up at row %d" % k)
        a1[k] = 0 if phase0 == "0" else (1 if phase0 == "1" else -1)
        a2[k] = 0 if phase1 == "0" else (1 if phase1 == "1" else -1)
        k += 1
    if k != n:
        raise RuntimeError("pairs file has %d records, expected %d" % (k, n))
    return a1, a2


def single_label(a1, a2):
    """仅当 cis 两个端点都属于同一个已知 copy 时，返回该记录的 label。"""
    return np.where(a1 == a2, a1, np.int8(-1)).astype(np.int8)


# --------------------------------------------------------------------------
# reference 与 dense coordinates
# --------------------------------------------------------------------------
def load_reference(gate, path=REF3DG):
    gate.require(STAGE)
    coords = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            a = line.split()
            if len(a) < 5:
                continue
            coords.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
    return coords


_CACHE = {}


def clear_cache():
    """清除候选之间的 dense 坐标数组。"""
    _CACHE.clear()


def dense(structs, trk, n_bins):
    """返回密集的 (n_bins, 3) 数组；网格外 bead 的位置不计入指标 bin。"""
    key = (id(structs), trk, n_bins)
    ent = _CACHE.get(key)
    if ent is not None and ent[0] is structs:
        return ent[1]
    A = np.full((n_bins, 3), np.nan)
    for p, v in structs.get(trk, {}).items():
        b = (int(p) - OFF) // BIN
        if 0 <= b < n_bins:
            A[b] = v
    _CACHE[key] = (structs, A)
    return A


def _dists(A, b1, b2):
    """先检查边界再索引并计算距离；无效 bin 保持为 NaN。"""
    b1 = np.asarray(b1, dtype=np.int64)
    b2 = np.asarray(b2, dtype=np.int64)
    out = np.full(len(b1), np.nan)
    in_grid = ((b1 >= 0) & (b2 >= 0) & (b1 < len(A)) & (b2 < len(A)))
    if not in_grid.any():
        return out
    ii = np.where(in_grid)[0]
    d = A[b1[ii]] - A[b2[ii]]
    bad = np.isnan(d).any(1)
    out[ii] = np.where(bad, np.nan, np.sqrt((d * d).sum(1)))
    return out


def ours_dists(structs, chrom_index, k, b1, b2, n_bins):
    return _dists(dense(structs, track(chrom_index, k), n_bins), b1, b2)


def ref_dists(ref, chrom_name, which, b1, b2, n_bins):
    return _dists(dense(ref, "%s(%s)" % (chrom_name, REF_SUFFIX[which]), n_bins), b1, b2)


def has_two_copies(structs, chrom_index):
    return bool(structs.get(track(chrom_index, 0))) and bool(structs.get(track(chrom_index, 1)))


def _rho(x, y):
    return spearman(x, y) if len(x) >= MIN_RHO_PAIRS else float("nan")


# --------------------------------------------------------------------------
# 每条染色体一个几何 gauge
# --------------------------------------------------------------------------
def geometry_gauge(structs, chrom_index, chrom_name, ref, n_bins):
    """仅依据四条轨迹共同的有限距离向量解析 copy gauge。

    ``direct`` 将候选 ``a,b`` 对应到 reference ``mat,pat``；``swapped`` 则对应到 ``pat,mat``。返回的判定同时供 R1 与 R2 使用。几何平局有意保持未决；R1 会对两种 label 得分取均值，而不是查 truth 后选择方向。
    """
    out = {
        "policy": "common-finite-non-diagonal-four-track-Spearman",
        "n_total_non_diagonal_pairs": int(n_bins * (n_bins - 1) // 2),
    }
    if not has_two_copies(structs, chrom_index):
        out.update({"applicable": False, "status": "single_trajectory", "orientation": None,
                    "n_common": 0})
        return out
    i, j = np.triu_indices(n_bins, k=1)
    ours = [ours_dists(structs, chrom_index, k, i, j, n_bins) for k in (0, 1)]
    refs = [ref_dists(ref, chrom_name, k, i, j, n_bins) for k in (0, 1)]
    ok = np.ones(len(i), dtype=bool)
    for value in ours + refs:
        ok &= np.isfinite(value)
    out["n_common"] = int(ok.sum())
    if ok.sum() < MIN_RHO_PAIRS:
        out.update({"applicable": False, "status": "insufficient_common_finite_pairs",
                    "orientation": None})
        return out
    rho = {
        "a0_mat": _rho(ours[0][ok], refs[0][ok]),
        "a0_pat": _rho(ours[0][ok], refs[1][ok]),
        "a1_mat": _rho(ours[1][ok], refs[0][ok]),
        "a1_pat": _rho(ours[1][ok], refs[1][ok]),
    }
    direct = rho["a0_mat"] + rho["a1_pat"]
    swapped = rho["a0_pat"] + rho["a1_mat"]
    out.update({"rho": rho, "score_direct": float(direct), "score_swapped": float(swapped)})
    if not (np.isfinite(direct) and np.isfinite(swapped)):
        out.update({"applicable": False, "status": "nonfinite_geometry_score", "orientation": None})
    elif np.isclose(direct, swapped, rtol=0.0, atol=GEOMETRY_TIE_TOL):
        out.update({"applicable": True, "status": "tie_average_orientations", "orientation": None,
                    "tied": True})
    elif direct > swapped:
        out.update({"applicable": True, "status": "direct", "orientation": "direct", "tied": False})
    else:
        out.update({"applicable": True, "status": "swapped", "orientation": "swapped", "tied": False})
    return out


def _phase_accuracy(d_a, d_b, labels, phase_to_copy):
    copies = (d_a, d_b)
    own = np.where(labels == 0, copies[phase_to_copy[0]], copies[phase_to_copy[1]])
    other = np.where(labels == 0, copies[1 - phase_to_copy[0]], copies[1 - phase_to_copy[1]])
    tie = own == other
    hit = (own < other).astype(float) + 0.5 * tie
    return float(hit.mean()), tie


def _reference_accuracy(d_mat, d_pat, labels):
    # 冻结的对应关系：phase0 -> pat，phase1 -> mat；不按 truth 反转。
    own = np.where(labels == 0, d_pat, d_mat)
    other = np.where(labels == 0, d_mat, d_pat)
    tie = own == other
    hit = (own < other).astype(float) + 0.5 * tie
    return float(hit.mean()), tie


def _primary_accuracy(d_a, d_b, labels, gauge):
    """返回按几何 gauge 选择的 R1 与取 truth 最大值的诊断，不可反过来使用。"""
    direct, ties = _phase_accuracy(d_a, d_b, labels, (1, 0))
    swapped, _ = _phase_accuracy(d_a, d_b, labels, (0, 1))
    result = {
        "orientation_direct": direct,
        "orientation_swapped": swapped,
        "legacy_truth_max": max(direct, swapped),
        "ties": ties,
    }
    if not gauge.get("applicable"):
        result.update({"accuracy": None, "policy": "unavailable_insufficient_geometry"})
    elif gauge.get("orientation") == "direct":
        result.update({"accuracy": direct, "policy": "geometry_direct"})
    elif gauge.get("orientation") == "swapped":
        result.update({"accuracy": swapped, "policy": "geometry_swapped"})
    else:
        result.update({"accuracy": float((direct + swapped) / 2.0),
                       "policy": "geometry_tie_average_orientations"})
    return result


# --------------------------------------------------------------------------
# R1
# --------------------------------------------------------------------------
def r1_accuracy(structs, chrom_index, chrom_name, lab, b1, b2, n_bins, ref=None,
                gauge=None, oracle_structs=None, oracle_gauge=None):
    """在明确声明的一组共同记录上计算 R1。

    分母要求：同 copy 的 phase label、非对角 contact、bin 在网格内、候选的两条轨迹和 reference 的两条轨迹都存在；如果提供 oracle-fit，还要求其两条轨迹都存在。因此候选 accuracy、reference ceiling 和 oracle-fit ceiling 只在完全相同的记录上计算。
    """
    lab = np.asarray(lab, dtype=np.int8)
    b1 = np.asarray(b1, dtype=np.int64)
    b2 = np.asarray(b2, dtype=np.int64)
    out = {
        "applicable": False,
        "n_records": int(len(lab)),
        "n_out_of_grid_records": int(((b1 < 0) | (b2 < 0) | (b1 >= n_bins) | (b2 >= n_bins)).sum()),
        "n_label_or_crosscopy_excluded": int((lab < 0).sum()),
        "n_samebin_excluded": int(((lab >= 0) & (b1 == b2)).sum()),
        "n_out_of_grid_excluded": int(((lab >= 0) & (b1 != b2)
                                         & ((b1 < 0) | (b2 < 0) | (b1 >= n_bins) | (b2 >= n_bins))).sum()),
    }
    if ref is None:
        out["reason"] = "reference_required_for_geometry_gauge"
        return out
    if not has_two_copies(structs, chrom_index):
        out.update({"reason": "single_trajectory_has_no_two-copy_R1", "accuracy": None,
                    "reference_ceiling": None, "oracle_fit_ceiling": None})
        return out

    labelled = lab >= 0
    non_diagonal = b1 != b2
    in_grid = ((b1 >= 0) & (b2 >= 0) & (b1 < n_bins) & (b2 < n_bins))
    eligible = labelled & non_diagonal & in_grid
    da = ours_dists(structs, chrom_index, 0, b1, b2, n_bins)
    db = ours_dists(structs, chrom_index, 1, b1, b2, n_bins)
    dm = ref_dists(ref, chrom_name, 0, b1, b2, n_bins)
    dp = ref_dists(ref, chrom_name, 1, b1, b2, n_bins)
    candidate_ok = np.isfinite(da) & np.isfinite(db)
    ref_ok = np.isfinite(dm) & np.isfinite(dp)

    if oracle_structs is not None:
        oda = ours_dists(oracle_structs, chrom_index, 0, b1, b2, n_bins)
        odb = ours_dists(oracle_structs, chrom_index, 1, b1, b2, n_bins)
        oracle_ok = np.isfinite(oda) & np.isfinite(odb)
    else:
        oda = odb = None
        oracle_ok = np.ones(len(lab), dtype=bool)

    out["n_candidate_missing_excluded"] = int((eligible & ~candidate_ok).sum())
    out["n_oracle_missing_excluded"] = int((eligible & candidate_ok & ~oracle_ok).sum())
    out["n_reference_missing_excluded"] = int((eligible & candidate_ok & oracle_ok & ~ref_ok).sum())
    common = eligible & candidate_ok & oracle_ok & ref_ok
    out["n_common"] = int(common.sum())
    out["n_denominator"] = int(common.sum())
    if not common.any():
        out.update({"reason": "no_common_finite_labeled_non_diagonal_records", "accuracy": None,
                    "reference_ceiling": None, "oracle_fit_ceiling": None})
        return out

    g = gauge if gauge is not None else geometry_gauge(structs, chrom_index, chrom_name, ref, n_bins)
    primary = _primary_accuracy(da[common], db[common], lab[common], g)
    ref_acc, ref_tie = _reference_accuracy(dm[common], dp[common], lab[common])
    ref_inverse, _ = _reference_accuracy(dp[common], dm[common], lab[common])

    out.update({
        "gauge": g,
        "accuracy": primary["accuracy"],
        "accuracy_policy": primary["policy"],
        "orientation_direct": primary["orientation_direct"],
        "orientation_swapped": primary["orientation_swapped"],
        "legacy_truth_max": primary["legacy_truth_max"],
        "reference_ceiling": ref_acc,
        "reference_ceiling_policy": "fixed_phase0_pat_phase1_mat",
        "reference_ceiling_legacy_truth_max": max(ref_acc, ref_inverse),
        "candidate_ties": int(primary["ties"].sum()),
        "candidate_tie_rate": float(primary["ties"].mean()),
        "reference_ties": int(ref_tie.sum()),
        "reference_tie_rate": float(ref_tie.mean()),
    })
    # backward-compatible aliases 有意指向新的 geometry-gauged 值。
    out["acc"] = out["accuracy"]
    out["ceil_ref"] = out["reference_ceiling"]

    if oracle_structs is not None:
        og = oracle_gauge if oracle_gauge is not None else geometry_gauge(
            oracle_structs, chrom_index, chrom_name, ref, n_bins)
        oracle_primary = _primary_accuracy(oda[common], odb[common], lab[common], og)
        out.update({
            "oracle_gauge": og,
            "oracle_fit_ceiling": oracle_primary["accuracy"],
            "oracle_fit_ceiling_policy": oracle_primary["policy"],
            "oracle_fit_legacy_truth_max": oracle_primary["legacy_truth_max"],
            "oracle_fit_ties": int(oracle_primary["ties"].sum()),
            "oracle_fit_tie_rate": float(oracle_primary["ties"].mean()),
        })
    out["applicable"] = out["accuracy"] is not None
    if not out["applicable"]:
        out["reason"] = "insufficient_geometry_for_gauge"
    return out


# --------------------------------------------------------------------------
# R2
# --------------------------------------------------------------------------
def r2_table(structs, chrom_index, chrom_name, ref, n_bins, gauge=None):
    """四条轨迹的 R2 表：使用一组共同的非对角有限 bin 对 mask。"""
    if not has_two_copies(structs, chrom_index):
        i, j = np.triu_indices(n_bins, k=1)
        a = ours_dists(structs, chrom_index, 0, i, j, n_bins)
        mat = ref_dists(ref, chrom_name, 0, i, j, n_bins)
        pat = ref_dists(ref, chrom_name, 1, i, j, n_bins)
        base = {"applicable": False, "reason": "single_trajectory_has_no_two-copy_R2",
                "contrast": None, "matched": None, "swapped": None, "gauge": None,
                "n_common": 0,
                "single_track_baseline": {}}
        for suffix, values in (("mat", mat), ("pat", pat)):
            ok = np.isfinite(a) & np.isfinite(values)
            base["single_track_baseline"]["a0_%s" % suffix] = {
                "rho": _rho(a[ok], values[ok]), "n": int(ok.sum())}
        return base

    g = gauge if gauge is not None else geometry_gauge(structs, chrom_index, chrom_name, ref, n_bins)
    if not g.get("applicable"):
        return {"applicable": False, "reason": g.get("status"), "contrast": None,
                "matched": None, "swapped": None, "gauge": g,
                "n_common": int(g.get("n_common", 0))}

    n = int(g["n_common"])
    rho = g["rho"]
    table = {key: {"rho": float(value), "n": n} for key, value in rho.items()}
    direct = float((rho["a0_mat"] + rho["a1_pat"]) / 2.0)
    swapped = float((rho["a0_pat"] + rho["a1_mat"]) / 2.0)
    if g.get("orientation") == "direct":
        matched, other, pairing = direct, swapped, "a=mat"
    elif g.get("orientation") == "swapped":
        matched, other, pairing = swapped, direct, "a=pat"
    else:
        # 几何 score 在容差内相等。显式平均两种方向，避免浮点残差产生带符号的 contrast，
        # 或选择依赖 truth 的方向。
        matched = other = float((direct + swapped) / 2.0)
        pairing = None
    table.update({
        "applicable": True,
        "n_common": n,
        "gauge": g,
        "matched": matched,
        "swapped": other,
        "contrast": float(matched - other),
        "best_pairing": pairing,
        # 为旧 reader 保留这些名称；现在两者都基于共同 mask。
        "matched_a_mat": direct,
        "matched_a_pat": swapped,
    })
    return table


# --------------------------------------------------------------------------
# R3
# --------------------------------------------------------------------------
def r3_fragments(structs, chrom_index, chrom_name, ref, n_bins, frag_bins=FRAG_BINS):
    """局部 20 Mb labels；明确记录不可用/平局片段，并在 gap 处打断连续段。"""
    if not has_two_copies(structs, chrom_index):
        return {"applicable": False, "reason": "single_trajectory_has_no_two-copy_R3",
                "frac_consistent": None, "longest_run": None, "n_walls": None,
                "n_fragments": 0, "n_fragments_applicable": 0,
                "n_fragments_total": int((n_bins + frag_bins - 1) // frag_bins),
                "n_fragments_tied": 0, "n_fragments_insufficient": 0, "detail": []}

    detail = []
    for f0 in range(0, n_bins, frag_bins):
        bins = np.arange(f0, min(f0 + frag_bins, n_bins))
        item = {"frag_start_mb": int(f0),  # 3 Mb metrics grid 的 legacy offset
                "grid_start_bin": int(f0), "frag_start_bp": int(OFF + f0 * BIN), "label": None}
        if len(bins) < 4:
            item.update({"status": "insufficient_fragment_length", "n": 0})
            detail.append(item)
            continue
        i1, i2 = np.triu_indices(len(bins), k=1)
        b1, b2 = bins[i1], bins[i2]
        ours = [ours_dists(structs, chrom_index, k, b1, b2, n_bins) for k in (0, 1)]
        refs = [ref_dists(ref, chrom_name, k, b1, b2, n_bins) for k in (0, 1)]
        ok = np.ones(len(b1), dtype=bool)
        for value in ours + refs:
            ok &= np.isfinite(value)
        item["n"] = int(ok.sum())
        if ok.sum() < MIN_RHO_PAIRS:
            item["status"] = "insufficient_common_finite_pairs"
            detail.append(item)
            continue
        direct = _rho(ours[0][ok], refs[0][ok]) + _rho(ours[1][ok], refs[1][ok])
        swapped = _rho(ours[0][ok], refs[1][ok]) + _rho(ours[1][ok], refs[0][ok])
        score = float(direct - swapped)
        item["score"] = score
        if not np.isfinite(score):
            item["status"] = "nonfinite_local_score"
        elif np.isclose(score, 0.0, rtol=0.0, atol=GEOMETRY_TIE_TOL):
            item["status"] = "local_geometry_tie"
        else:
            item.update({"status": "labelled", "label": int(score < 0.0)})
        detail.append(item)

    labelled = [item for item in detail if item["status"] == "labelled"]
    n_total = len(detail)
    n_valid = len(labelled)
    n_tied = sum(item["status"] == "local_geometry_tie" for item in detail)
    n_insufficient = n_total - n_valid - n_tied
    result = {
        "n_fragments_total": int(n_total),
        "n_fragments": int(n_valid),
        "n_fragments_applicable": int(n_valid),
        "n_fragments_tied": int(n_tied),
        "n_fragments_insufficient": int(n_insufficient),
        "detail": detail,
        "labels": [item["label"] for item in labelled],
        "frag_start_mb": [item["frag_start_mb"] for item in labelled],
        "legacy_grid_offset_mb": [item["frag_start_mb"] for item in labelled],
        "grid_start_bin": [item["grid_start_bin"] for item in labelled],
        "frag_start_bp": [item["frag_start_bp"] for item in labelled],
        "labels_by_fragment": [item["label"] for item in detail],
    }
    if not labelled:
        result.update({"applicable": False, "reason": "no_resolvable_fragments",
                       "global_label": None, "global_label_tied": False,
                       "frac_consistent": None, "longest_run": None, "n_walls": None})
        return result

    labels = np.array([item["label"] for item in labelled], dtype=np.int8)
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    global_tie = n0 == n1
    global_label = None if global_tie else (0 if n0 > n1 else 1)
    frac = float(max(n0, n1) / n_valid)
    best_by_label = {0: 0, 1: 0}
    run_by_label = {0: 0, 1: 0}
    walls = 0
    previous = None
    for item in detail:
        label = item["label"]
        if label is None:
            previous = None
            run_by_label = {0: 0, 1: 0}
            continue
        if previous is not None and label != previous:
            walls += 1
        previous = label
        for possible in (0, 1):
            run_by_label[possible] = run_by_label[possible] + 1 if label == possible else 0
            best_by_label[possible] = max(best_by_label[possible], run_by_label[possible])
    # label 数量相等时没有多数 label。报告两个并列多数 run 中较长者，保证全局 A/B 交换不改变结果。
    longest = max(best_by_label.values()) if global_tie else best_by_label[global_label]
    result.update({"applicable": True, "global_label": global_label,
                   "global_label_tied": bool(global_tie), "frac_consistent": frac,
                   "longest_run": int(longest), "n_walls": int(walls),
                   "longest_run_by_label": best_by_label})
    return result
