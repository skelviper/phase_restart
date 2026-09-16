"""C 组 support 数据构造：在冻结 aggregate 上真正过滤缺失珠子相关的 contacts。

不做任何坐标技巧：被排除的 pair 的 counts/diag_counts 置零并重算
Nraw / Noff / conditional_factorial_constant / diag_factorial_sum；e 用同一 production
公式（sqrt(endpoint_counts + 10)，在有效支持上归一到均值 1）在有效端点上重算。
缺失珠子在 e 上恰好为 0，因此合法地退出 count。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import gammaln


def _positive_log_factorial_sum(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        return 0.0
    return float(gammaln(positive + 1.0).sum())


def build_filtered(data, mask: np.ndarray):
    """返回 (filtered_data, exposure_valid, audit)。"""
    mask = np.asarray(mask, dtype=bool)
    pair_i = np.asarray(data.pair_i, dtype=np.int64)
    pair_j = np.asarray(data.pair_j, dtype=np.int64)
    cis = np.asarray(data.cis_pair, dtype=bool)
    counts = np.asarray(data.counts, dtype=np.float64)
    # eligible 规则（冻结）：两端都落在 lane = mask.any(axis=0) 即可保留；单边存在的
    # locus 的 contacts 仍由存在的那条 copy 解释，因此不能取两 copy 交集。
    lane = mask.any(axis=0)
    eligible = lane[pair_i] & lane[pair_j]
    pair_ok = np.zeros((2, pair_i.shape[0]), dtype=bool)
    for copy in (0, 1):
        pair_ok[copy] = mask[copy][pair_i] & mask[copy][pair_j]
    kept = counts * eligible
    # diag 层：same-bin 记录只在“该 locus 在两个 copy 都存在”时保留（缺失珠子的
    # same-bin 记录随珠子一起退出），保证 raw/aggregate/endpoint 三项守恒。
    diag_kept = np.asarray(data.diag_counts, dtype=np.float64) * lane
    raw_cis = float(kept[cis].sum())
    raw_inter = float(kept[~cis].sum())
    raw_diag = float(diag_kept.sum())
    n_raw = raw_diag + raw_cis + raw_inter
    conditional_constant = 0.0
    for group_mask, group_total in ((cis, raw_cis), (~cis, raw_inter)):
        if group_total > 0.0:
            conditional_constant += -float(gammaln(group_total + 1.0))
            conditional_constant += _positive_log_factorial_sum(kept[group_mask])
    kept_counts = np.where(eligible, counts, 0.0)
    endpoints_total = np.zeros(int(data.n_loci), dtype=np.int64)
    counts_int = np.asarray(data.counts, dtype=np.int64)
    np.add.at(endpoints_total, pair_i[eligible], counts_int[eligible])
    np.add.at(endpoints_total, pair_j[eligible], counts_int[eligible])
    endpoints_total[lane] += 2 * np.asarray(data.diag_counts, dtype=np.int64)[lane]
    filtered = replace(
        data,
        endpoint_counts=endpoints_total,
        counts=kept_counts.astype(np.asarray(data.counts).dtype, copy=False),
        diag_counts=diag_kept.astype(np.asarray(data.diag_counts).dtype, copy=False),
        raw_records=n_raw,
        raw_same_bin=raw_diag,
        raw_cis_offdiag=raw_cis,
        raw_inter=raw_inter,
        conditional_factorial_constant=conditional_constant,
        diag_factorial_sum=_positive_log_factorial_sum(diag_kept),
    )
    # 训练用 e：有效支持上沿用完整数据的 production e（frozen aggregate 的
    # sqrt(endpoint_counts+10) 归一化向量），缺失珠子处补 0；有效支持内不再重缩放。
    # all-ones 时与冻结 G 逐位一致（parity 检查）。该层的 production endpoint 计数与
    # C 的有效 endpoint 计数审计见 audit 的 exposure_* 字段。
    endpoint_counts_masked = np.zeros(int(data.n_loci), dtype=np.float64)
    np.add.at(endpoint_counts_masked, pair_i[eligible], kept_counts[eligible])
    np.add.at(endpoint_counts_masked, pair_j[eligible], kept_counts[eligible])
    endpoint_counts_masked[lane] += 2.0 * np.asarray(data.diag_counts, dtype=np.float64)[lane]
    exposure_production = np.sqrt(endpoint_counts_masked + 10.0)
    exposure_production[~lane] = 0.0
    exposure_production[lane] /= float(exposure_production[lane].mean())
    # 训练用的 e：按冻结要求用**筛后数据**的 production 公式 sqrt(endpoint+10) 在 lane 上
    # 归一到均值 1，非 lane 为 0（不是沿用 full-data e）。
    e_shared = exposure_production
    audit = {
        "n_loci": int(data.n_loci),
        "n_valid_copyA": int(mask[0].sum()),
        "n_valid_copyB": int(mask[1].sum()),
        "n_union_loci": int(lane.sum()),
        "n_shared_loci": int(np.count_nonzero(mask[0] & mask[1])),
        "original_raw_records": float(data.raw_records),
        "filtered_raw_records": n_raw,
        "original_raw_cis_offdiag": float(data.raw_cis_offdiag),
        "filtered_raw_cis_offdiag": raw_cis,
        "original_raw_inter": float(data.raw_inter),
        "filtered_raw_inter": raw_inter,
        "original_raw_same_bin": float(data.raw_same_bin),
        "filtered_raw_same_bin": raw_diag,
        "removed_pairs_cis": int(np.count_nonzero(cis & ~eligible)),
        "removed_pairs_inter": int(np.count_nonzero(~cis & ~eligible)),
        "removed_counts_cis": float(data.raw_cis_offdiag) - raw_cis,
        "removed_counts_inter": float(data.raw_inter) - raw_inter,
        "exposure_mean_valid_shared": float(e_shared[lane].mean()),
        "exposure_min_valid_shared": float(e_shared[lane].min()),
        "exposure_max_valid_shared": float(e_shared[lane].max()),
        "filtered_endpoint_total": float(endpoint_counts_masked.sum()),
        "full_data_endpoint_total": float(np.asarray(data.endpoint_counts).sum()),
    }
    return filtered, e_shared, audit
