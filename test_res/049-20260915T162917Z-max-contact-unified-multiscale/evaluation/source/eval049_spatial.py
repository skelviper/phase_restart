#!/usr/bin/env python
"""049 空间评价：merged chr 中心、单一尺度 stress、标签置换 null、Procrustes、copy 中心。

仅在 evaluation 侧、reference gate 之后调用。支撑 = 046 冻结 old21 common mask 的
2447 个 valid loci（4894 beads）；不引入新的 finite 交集、不缩小分母。
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

import eval049_lib as lib


def valid_global_bins(data: lib.Aggregate, masks: Mapping[str, Mapping[str, Any]],
                      expected_valid_bins: int | None = None) -> np.ndarray:
    valid = np.zeros(data.n_loci, dtype=bool)
    expected: list[np.ndarray] = []
    for name in data.chromosome_names:
        mask = masks[str(name)]
        global_indices = lib.mask_global_indices(data.offsets, mask)
        selected = global_indices[np.asarray(mask["valid_local_bins"], dtype=np.int64)]
        valid[selected] = True
        expected.append(selected)
    actual = np.flatnonzero(valid)
    if not np.array_equal(actual, np.sort(np.concatenate(expected))):
        raise RuntimeError("valid-bin global locus set disagrees with mask positions")
    if expected_valid_bins is not None and int(valid.sum()) != int(expected_valid_bins):
        raise RuntimeError("valid bin count %d != %d" % (int(valid.sum()), int(expected_valid_bins)))
    return valid


def per_chromosome_valid_indices(data: lib.Aggregate,
                                 masks: Mapping[str, Mapping[str, Any]]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for name in data.chromosome_names:
        mask = masks[str(name)]
        global_indices = lib.mask_global_indices(data.offsets, mask)
        out.append(global_indices[np.asarray(mask["valid_local_bins"], dtype=np.int64)])
    return out


def merged_chr_centers(coords: np.ndarray,
                       per_chr_valid: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """每 chr：两 copy 各取 valid bins 均值，再两 copy 等权平均。返回 (20,3) 与 (20,2,3)。"""
    copy_centers = np.zeros((len(per_chr_valid), 2, 3), dtype=np.float64)
    for ci, indices in enumerate(per_chr_valid):
        copy_centers[ci, 0] = coords[0, indices].mean(axis=0)
        copy_centers[ci, 1] = coords[1, indices].mean(axis=0)
    return copy_centers.mean(axis=1), copy_centers


def pairwise_upper(points: np.ndarray) -> np.ndarray:
    diff = np.asarray(points, dtype=np.float64)[:, None, :] - np.asarray(points, dtype=np.float64)[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    i, j = np.triu_indices(len(points), k=1)
    return dist[i, j]


def upper_pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def pairwise_upper_filtered(points: np.ndarray, keep_pair: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (distances, pair_a, pair_b)；keep_pair(a, b) 为可选的 pair 过滤。"""
    i, j = np.triu_indices(len(points), k=1)
    if keep_pair is not None:
        keep = np.asarray(keep_pair(i, j), dtype=bool)
        i, j = i[keep], j[keep]
    diff = np.asarray(points, dtype=np.float64)[i] - np.asarray(points, dtype=np.float64)[j]
    return np.sqrt(np.sum(diff * diff, axis=1)), i, j


def optimal_scale(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.sum(np.asarray(pred, dtype=np.float64) ** 2))
    return float(np.sum(np.asarray(pred, dtype=np.float64) * np.asarray(ref, dtype=np.float64)) / denom) if denom else float("nan")


def stress(pred: np.ndarray, ref: np.ndarray, scale: float | None = None) -> float:
    if scale is None:
        scale = optimal_scale(pred, ref)
    resid = scale * np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    denom = float(np.sum(np.asarray(ref, dtype=np.float64) ** 2))
    return float(math.sqrt(float(np.sum(resid * resid)) / denom)) if denom else float("nan")


def neighbor_sets(points: np.ndarray, names: Sequence[str], k: int = 3) -> list[set[str]]:
    out: list[set[str]] = []
    for i in range(len(points)):
        d = np.sqrt(np.sum((np.asarray(points) - np.asarray(points)[i]) ** 2, axis=1))
        order = [int(x) for x in np.argsort(d, kind="stable") if int(x) != i][:k]
        out.append({str(names[x]) for x in order})
    return out


def top3_neighbor_overlap(cand: np.ndarray, ref: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    cand_sets, ref_sets = neighbor_sets(cand, names), neighbor_sets(ref, names)
    overlaps = [len(cand_sets[i] & ref_sets[i]) for i in range(len(cand_sets))]
    return {
        "per_chromosome": {str(names[i]): int(overlaps[i]) for i in range(len(names))},
        "mean_overlap_top3": float(np.mean(overlaps)),
        "denominator_chromosomes": len(names),
    }


def center_metrics(cand_centers: np.ndarray, ref_centers: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    if len(cand_centers) != 20 or len(ref_centers) != 20:
        raise RuntimeError("merged chr center metrics require exactly 20 chromosomes, got %d/%d"
                           % (len(cand_centers), len(ref_centers)))
    cand_d, ref_d = pairwise_upper(cand_centers), pairwise_upper(ref_centers)
    pair_a, pair_b = upper_pair_indices(len(cand_centers))
    if len(cand_d) != 190:
        raise RuntimeError("merged chr centers must yield exactly 190 distances, got %d" % len(cand_d))
    scale = optimal_scale(cand_d, ref_d)
    per_chr_pearson: dict[str, float] = {}
    per_chr_spearman: dict[str, float] = {}
    for i, name in enumerate(names):
        keep = (pair_a == i) | (pair_b == i)
        if len(names) == 20 and int(np.count_nonzero(keep)) != 19:
            raise RuntimeError("per-chromosome distance profile must contain exactly 19 distances, got %d"
                               % int(np.count_nonzero(keep)))
        # 每 chr profile 只有 19 条距离：用 min=2 的非恒定 helper，不套用 R2 的 >=20 门槛
        per_chr_pearson[str(name)] = lib.metric_min("pearson", cand_d[keep], ref_d[keep], 2)
        per_chr_spearman[str(name)] = lib.metric_min("spearman", cand_d[keep], ref_d[keep], 2)
    finite_p = [v for v in per_chr_pearson.values() if math.isfinite(v)]
    finite_s = [v for v in per_chr_spearman.values() if math.isfinite(v)]
    return {
        "n_centers": int(len(cand_centers)), "n_distances": int(len(cand_d)),
        "pearson": lib.metric("pearson", cand_d, ref_d),
        "spearman": lib.metric("spearman", cand_d, ref_d),
        "optimal_single_scale": scale,
        "normalized_stress": stress(cand_d, ref_d, scale),
        "per_chr_19_distance_profile_pearson": per_chr_pearson,
        "per_chr_19_distance_profile_spearman": per_chr_spearman,
        "per_chr_profile_macro_pearson": float(np.mean(finite_p)) if finite_p else None,
        "per_chr_profile_macro_spearman": float(np.mean(finite_s)) if finite_s else None,
        "top3_neighbors": top3_neighbor_overlap(cand_centers, ref_centers, names),
        "per_chr_profile_min_pairs": 2,
        "per_chr_profile_note": "19 distances per chromosome; helper uses min_pairs=2, R2 keeps the >=20 rule",
        "scale_policy": "one single optimal distance scale for the whole 190-vector; no per-pair or per-chr fit",
    }


def label_permutation_null(cand_centers: np.ndarray, ref_centers: np.ndarray, observed_spearman: float,
                           draws: int = lib.PERMUTATION_DRAWS, seed: int = lib.PERMUTATION_SEED) -> dict[str, Any]:
    """整 chr 标签置换（dyadic 依赖），不是独立 pair 检验。"""
    ref_d = pairwise_upper(ref_centers)
    ref_rank = rankdata(ref_d)
    rng = np.random.default_rng(seed)
    spearman_draws = np.empty(draws, dtype=np.float64)
    stress_draws = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        order = rng.permutation(len(cand_centers))
        cand_d = pairwise_upper(np.asarray(cand_centers)[order])
        spearman_draws[draw] = lib.spearman_ranked(cand_d, ref_rank)
        stress_draws[draw] = stress(cand_d, ref_d)
    p_value = None
    if observed_spearman is not None and math.isfinite(float(observed_spearman)):
        p_value = float((int(np.count_nonzero(spearman_draws >= float(observed_spearman))) + 1) / (draws + 1))
    return {
        "draws": int(draws), "seed": int(seed),
        "unit": "whole 20-chromosome label permutation of the 190-distance vector",
        "note": "dyadic dependence; not an independent pair test",
        "observed_spearman": observed_spearman,
        "spearman_null": {"mean": float(spearman_draws.mean()), "std": float(spearman_draws.std(ddof=1)),
                          "min": float(spearman_draws.min()), "max": float(spearman_draws.max()),
                          "p2_5": float(np.percentile(spearman_draws, 2.5)),
                          "p97_5": float(np.percentile(spearman_draws, 97.5))},
        "stress_null": {"mean": float(stress_draws.mean()), "std": float(stress_draws.std(ddof=1)),
                        "min": float(stress_draws.min()), "max": float(stress_draws.max())},
        "one_sided_p_ge_observed": p_value,
    }


def procrustes_global(cand: np.ndarray, ref: np.ndarray, *, allow_reflection: bool) -> dict[str, Any]:
    """单个 global transform（平移+旋转+统一缩放）；correspondence 固定为输入顺序。"""
    x = np.asarray(cand, dtype=np.float64)
    y = np.asarray(ref, dtype=np.float64)
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    h = xc.T @ yc
    u, s, vt = np.linalg.svd(h)
    # row-vector 约定：最小化 ||s * xc @ R - yc||，H = xc.T @ yc = U S V^T 的最优 R = U V^T
    rot = u @ vt
    det = float(np.linalg.det(rot))
    if not allow_reflection and det < 0:
        u = u.copy()
        u[:, -1] *= -1.0
        rot = u @ vt
        s = s.copy()
        s[-1] *= -1.0
        det = float(np.linalg.det(rot))
        if det <= 0.0:
            raise RuntimeError("proper Procrustes failed to produce det=+1 (det=%r)" % det)
    denom = float(np.sum(xc * xc))
    scale = float(np.sum(s) / denom) if denom else float("nan")
    aligned = scale * (xc @ rot)
    resid = aligned - yc
    rmsd = float(math.sqrt(float(np.mean(np.sum(resid * resid, axis=1)))))
    ref_rms = float(math.sqrt(float(np.mean(np.sum(yc * yc, axis=1)))))
    return {
        "allow_reflection": bool(allow_reflection), "rotation_det": det,
        "optimal_uniform_scale": scale, "aligned_rmsd": rmsd, "reference_rms_radius": ref_rms,
        "normalized_aligned_rmsd": float(rmsd / ref_rms) if ref_rms else float("nan"),
        "rotation_matrix": rot.tolist(),
        "policy": "single global transform for all centers; no per-chr fit",
    }


def mapping_from_rho(rho: Mapping[str, float]) -> dict[str, Any]:
    """用 lib.derive_rho 的既有规则（signed rho、direct/swapped、tie<=1e-12）决定 whole-chr 对应。"""
    derived = lib.derive_rho(rho)
    if not derived["derived_defined"]:
        return {"defined": False, "reason": derived["undefined_reason"], "orientation": "undefined",
                "copy0_to_reference": None, "copy1_to_reference": None, "derived": derived}
    if derived["geometry_tie"]:
        return {"defined": False, "reason": "unresolved_geometry_tie_within_1e-12", "orientation": "unresolved_tie",
                "copy0_to_reference": None, "copy1_to_reference": None, "derived": derived}
    direct = derived["orientation"] == "direct"
    return {"defined": True, "reason": None, "orientation": derived["orientation"],
            "copy0_to_reference": 0 if direct else 1, "copy1_to_reference": 1 if direct else 0, "derived": derived}


def copy_center_metrics(coords: np.ndarray, reference: np.ndarray, per_chr_valid: Sequence[np.ndarray],
                        masks: Mapping[str, Mapping[str, Any]], names: Sequence[str],
                        offsets: np.ndarray) -> dict[str, Any]:
    """40 copy center / 760 跨 chr 距离；每 chr 由 whole-chr signed-Pearson derive_rho 规则固定对应。

    同 chr 的 20 个 homolog pair 不进入距离向量（距离只保留跨 chr 的 760 对）；
    Procrustes 仍对全部 40 个点做单个 global transform（correspondence 固定）。
    """
    n = len(names)
    if n != 20:
        raise RuntimeError("copy-center metrics require exactly 20 chromosomes, got %d" % n)
    cand_copy = np.zeros((n, 2, 3), dtype=np.float64)
    ref_copy = np.zeros((n, 2, 3), dtype=np.float64)
    mapping: dict[str, Any] = {}
    unresolved: list[str] = []
    for ci, name in enumerate(names):
        indices = per_chr_valid[ci]
        for copy in (0, 1):
            cand_copy[ci, copy] = coords[copy, indices].mean(axis=0)
            ref_copy[ci, copy] = reference[copy, indices].mean(axis=0)
        mask = masks[str(name)]
        global_indices = lib.mask_global_indices(offsets, mask)
        pair_i = np.asarray(mask["pair_i"], dtype=np.int64)
        pair_j = np.asarray(mask["pair_j"], dtype=np.int64)
        common = np.asarray(mask["common"], dtype=bool)
        rho: dict[str, float] = {}
        for local_copy, local_tag in enumerate(("A", "B")):
            cand_d = lib.distance(coords[local_copy][global_indices], pair_i[common], pair_j[common])
            for ref_copy_index, ref_tag in enumerate(("mat", "pat")):
                ref_d = lib.distance(reference[ref_copy_index][global_indices], pair_i[common], pair_j[common])
                rho["%s_%s" % (local_tag, ref_tag)] = lib.metric("pearson", cand_d, ref_d)
        decision = mapping_from_rho(rho)
        if not decision["defined"]:
            unresolved.append(str(name))
        mapping[str(name)] = {
            "signed_pearson_rho": rho, "orientation": decision["orientation"], "mapping_defined": decision["defined"],
            "reason": decision["reason"], "copy0_to_reference": decision["copy0_to_reference"],
            "copy1_to_reference": decision["copy1_to_reference"],
            "direct": decision["derived"]["direct"], "swapped": decision["derived"]["swapped"],
            "contrast": decision["derived"]["contrast"], "derived": decision["derived"],
            "criterion": ("fixed whole-chromosome signed intra-copy Pearson rho through lib.derive_rho "
                          "(direct/swapped, tie<=1e-12); not a squared R2 and not a copy-A-only preference"),
        }

    chr_index = np.repeat(np.arange(n), 2)
    keep_pair = (lambda a, b: chr_index[a] != chr_index[b])

    def cloud(flip: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
        cand_points = np.zeros((2 * n, 3), dtype=np.float64)
        ref_points = np.zeros((2 * n, 3), dtype=np.float64)
        for ci, name in enumerate(names):
            entry = mapping[str(name)]
            if entry["mapping_defined"]:
                first, second = entry["copy0_to_reference"], entry["copy1_to_reference"]
            else:
                first, second = (1, 0) if flip[ci] else (0, 1)
            cand_points[2 * ci + 0] = cand_copy[ci, 0]
            cand_points[2 * ci + 1] = cand_copy[ci, 1]
            ref_points[2 * ci + 0] = ref_copy[ci, first]
            ref_points[2 * ci + 1] = ref_copy[ci, second]
        return cand_points, ref_points

    def cloud_metrics(flip: Sequence[bool]) -> dict[str, Any]:
        cand_points, ref_points = cloud(flip)
        cand_d, pair_a, pair_b = pairwise_upper_filtered(cand_points, keep_pair)
        ref_d, _, _ = pairwise_upper_filtered(ref_points, keep_pair)
        if len(cand_d) != 760:
            raise RuntimeError("copy-center cross-chromosome distances must be 760, got %d" % len(cand_d))
        if np.any(chr_index[pair_a] == chr_index[pair_b]):
            raise RuntimeError("same-chromosome homolog pairs leaked into the copy-center distance vector")
        scale = optimal_scale(cand_d, ref_d)
        return {
            "n_distances": int(len(cand_d)),
            "pearson": lib.metric("pearson", cand_d, ref_d),
            "spearman": lib.metric("spearman", cand_d, ref_d),
            "optimal_single_scale": scale,
            "normalized_stress": stress(cand_d, ref_d, scale),
            "procrustes_proper": procrustes_global(cand_points, ref_points, allow_reflection=False),
            "procrustes_reflection": procrustes_global(cand_points, ref_points, allow_reflection=True),
        }

    identity_flip = [False] * n
    primary = cloud_metrics(identity_flip)
    alternative = cloud_metrics([str(name) in set(unresolved) for name in names]) if unresolved else None
    _, ref_points = cloud(identity_flip)
    return {
        "n_copy_centers": int(2 * n),
        "primary": primary,
        "unresolved_alternative": alternative,
        "unresolved_chromosomes": unresolved,
        "mapping_policy": ("per-chromosome whole-chromosome signed-Pearson derive_rho mapping; "
                           "unresolved/undefined chromosomes use identity as deterministic fallback and the "
                           "alternative (flipped) variant is reported alongside"),
        "per_chromosome_mapping": mapping,
        "distance_policy": "760 cross-chromosome copy-center distances only; the 20 same-chr homolog pairs are excluded",
        "policy": "one unified scale and one global fit for all 760 distances; no per-pair best fit",
        "reflection_note": "distance metrics are reflection invariant; only aligned RMSD differs",
        "ref_points_checksum": float(np.sum(ref_points)),
    }
