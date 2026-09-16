"""049 共用分析核：端点三 loss audit、posterior 表、细 rate 表。

只读训练侧数据与已冻结坐标；不读 reference，不读 phase。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from frozen_imports import contact_model, data_io
from max_contact_objective import MaxContactObjective
from round_paths import AGGREGATE_1MB, PAIR_BLOCK, ROOT, SOURCE_045
from shared_capture_objective import PenaltyWeights

FULL_WEIGHTS = PenaltyWeights()
AGGREGATES = {
    5_000_000: SOURCE_045 / "inputs/real_5000000_aggregate.npz",
    2_000_000: SOURCE_045 / "inputs/real_2000000_aggregate.npz",
    1_000_000: AGGREGATE_1MB,
}


def load_layer(bin_size: int):
    data = data_io.load_aggregate(AGGREGATES[int(bin_size)])
    data.assert_consistent()
    return data


def build_objective(data: Any, loss: str = "B", pair_block: int = PAIR_BLOCK) -> MaxContactObjective:
    return MaxContactObjective(data, loss, weights=FULL_WEIGHTS, mode="V0-fixed-production-e",
                               device="cuda", pair_block=pair_block, inner_cap=80, cg_cap=80,
                               profile_warm_start=True, known_e=None)


def _theta_from(coords: np.ndarray, p: float) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    contact_model.assert_inside_unit_ball(coords)
    raw_y = contact_model.sphere_inverse(coords)
    q = float(contact_model.q_from_p(float(p)))
    return np.concatenate((raw_y.reshape(-1), np.asarray([q], dtype=np.float64)))


def three_loss_components(data: Any, coords: np.ndarray, p: float) -> dict[str, Any]:
    """一次 full-grid 扫描得到 Zsum/Zmax、观测项、三 loss count 与两个恒等式。"""
    obj = build_objective(data, "B")
    theta = _theta_from(coords, p)
    values = torch.as_tensor(theta, dtype=torch.float64, device="cuda")
    x = obj._physics._map_raw(values[:-1].reshape(2, int(data.n_loci), 3))
    obj._physics.validate_physical_coordinates_tensor(x)
    p_value, _dpdq = contact_model.p_from_q(float(theta[-1]))
    p_t = torch.as_tensor(p_value, dtype=torch.float64, device="cuda")
    eprod = obj._fixed_e[obj._pair_i] * obj._fixed_e[obj._pair_j]
    normalizer = float(data.raw_records)
    n_cis = float(data.raw_cis_offdiag)
    n_inter = float(data.raw_inter)
    n_off = n_cis + n_inter
    z_sum = 0.0
    z_max = 0.0
    obs_sum = 0.0
    obs_max = 0.0
    obs_log_gamma = 0.0
    observed = obj._counts > 0.0
    for start in range(0, int(data.n_pairs), obj.pair_block):
        stop = min(start + obj.pair_block, int(data.n_pairs))
        i = obj._pair_i[start:stop]
        j = obj._pair_j[start:stop]
        kernels, _ = obj._kernel_block(x, i, j, p_t, with_gradient=False)
        stacked, _w = obj._block_mixture(kernels, p_t, obj._cis[start:stop])
        local_e = eprod[start:stop]
        r_sum = local_e * (stacked[0] + stacked[1] + stacked[2] + stacked[3])
        r_max = local_e * stacked.max(dim=0).values
        z_sum += float(r_sum.sum().item())
        z_max += float(r_max.sum().item())
        mask = observed[start:stop]
        if bool(torch.any(mask)):
            counts = obj._counts[start:stop][mask]
            obs_sum += float((counts * torch.log(r_sum[mask])).sum().item())
            obs_max += float((counts * torch.log(r_max[mask])).sum().item())
            obs_log_gamma += float((counts * torch.log(r_max[mask] / r_sum[mask])).sum().item())
    k0 = 0.0
    if n_off > 0.0:
        if n_cis > 0.0:
            k0 += n_cis * math.log(n_cis / n_off)
        if n_inter > 0.0:
            k0 += n_inter * math.log(n_inter / n_off)
    positive = np.asarray(data.diag_counts, dtype=np.float64)
    positive = positive[positive > 0.0]
    diag = float((positive - positive * np.log(positive)).sum())
    if data.count_mode in ("raw_integer", "synthetic_integer"):
        from scipy.special import gammaln
        diag += float(gammaln(positive + 1.0).sum())
    count_a_raw = n_off * math.log(z_sum) - obs_sum + k0 + diag
    count_b_raw = n_off * math.log(z_sum) - obs_max + k0 + diag
    count_c_raw = n_off * math.log(z_max) - obs_max + k0 + diag
    # 物理正则：raw 值与按冻结权重的加权值分开报告（bend 权重 0.01）
    objectives = {loss: build_objective(data, loss) for loss in ("A", "B", "C")}
    direct: dict[str, dict[str, Any]] = {}
    for loss, obj in objectives.items():
        loss_value, _gradient, components = obj.evaluate(_theta_from(coords, p), True)
        direct[loss] = {
            "total": float(loss_value),
            "count_nll_normalized": float(components["count_nll_normalized"]),
            "raw": {key: float(components[key]) for key in ("bond", "repulsion", "bend", "p_prior")},
            "weighted": {
                "bond": float(components["weighted_bond"]),
                "repulsion": float(components["weighted_repulsion"]),
                "bend": float(components["weighted_bend"]),
                "p_prior": float(components["weighted_p_prior"]),
            },
            "penalty_weights": dict(components["penalty_weights"]),
            "p": float(components["p"]),
        }
    # 三条直线 count 与对应 objective 的固定点 rescore 必须一致
    for loss, computed in (("A", count_a_raw / normalizer), ("B", count_b_raw / normalizer),
                           ("C", count_c_raw / normalizer)):
        difference = abs(computed - direct[loss]["count_nll_normalized"])
        if difference > 1e-9:
            raise AssertionError("count_%s rescore mismatch vs objective %s: %.3e" % (loss, loss, difference))
    regularizers_raw = dict(direct["A"]["raw"])
    regularizers_weighted = dict(direct["A"]["weighted"])
    regularization = float(sum(regularizers_weighted.values()))
    for loss in ("A", "B", "C"):
        expected = count_a_raw / normalizer if loss == "A" else (
            count_b_raw / normalizer if loss == "B" else count_c_raw / normalizer)
        difference = abs((expected + regularization) - direct[loss]["total"])
        if difference > 1e-9:
            raise AssertionError("fullJ_%s rescore mismatch: %.3e" % (loss, difference))
    result = {
        "n_off": n_off, "Nraw": normalizer, "k0_raw": k0, "diag_nll_raw": diag,
        "Zsum": z_sum, "Zmax": z_max,
        "observed_log_r_sum": obs_sum, "observed_log_r_max": obs_max,
        "sum_C_log_gamma_max": obs_log_gamma,
        "count_A_raw": count_a_raw, "count_B_raw": count_b_raw, "count_C_raw": count_c_raw,
        "count_A": count_a_raw / normalizer, "count_B": count_b_raw / normalizer,
        "count_C": count_c_raw / normalizer,
        "count_nll_by_loss": {"A": count_a_raw / normalizer, "B": count_b_raw / normalizer,
                              "C": count_c_raw / normalizer},
        "B_minus_A_formula": -obs_log_gamma / normalizer,
        "B_minus_A_actual": (count_b_raw - count_a_raw) / normalizer,
        "C_minus_B_formula": n_off * math.log(z_max / z_sum) / normalizer,
        "C_minus_B_actual": (count_c_raw - count_b_raw) / normalizer,
        "regularizers_raw": regularizers_raw,
        "regularizers_weighted": regularizers_weighted,
        "regularization_total": regularization,
        "fullJ_A": count_a_raw / normalizer + regularization,
        "fullJ_B": count_b_raw / normalizer + regularization,
        "fullJ_C": count_c_raw / normalizer + regularization,
        "fullJ_by_loss": {"A": count_a_raw / normalizer + regularization,
                          "B": count_b_raw / normalizer + regularization,
                          "C": count_c_raw / normalizer + regularization},
        "total_A_direct": direct["A"]["total"],
        "total_B_direct": direct["B"]["total"],
        "total_C_direct": direct["C"]["total"],
        "objective_direct": direct,
        "p": direct["A"]["p"],
    }
    return result


def observed_pair_tables(data: Any, coords: np.ndarray, p: float, loss: str = "B") -> dict[str, np.ndarray]:
    """只在 counts>0 的 pair 上返回四状态 t、gamma 与四距离（后续统计全用 counts 加权）。"""
    obj = build_objective(data, loss)
    theta = _theta_from(coords, p)
    values = torch.as_tensor(theta, dtype=torch.float64, device="cuda")
    x = obj._physics._map_raw(values[:-1].reshape(2, int(data.n_loci), 3))
    p_value, _ = contact_model.p_from_q(float(theta[-1]))
    p_t = torch.as_tensor(p_value, dtype=torch.float64, device="cuda")
    eprod = obj._fixed_e[obj._pair_i] * obj._fixed_e[obj._pair_j]
    observed_index = torch.nonzero(obj._counts > 0.0, as_tuple=False).reshape(-1)
    gamma_blocks: list[np.ndarray] = []
    distance_blocks: list[np.ndarray] = []
    for start in range(0, int(observed_index.numel()), obj.pair_block):
        index = observed_index[start:start + obj.pair_block]
        i = obj._pair_i[index]
        j = obj._pair_j[index]
        aa = x[0, i] - x[0, j]
        ab = x[0, i] - x[1, j]
        ba = x[1, i] - x[0, j]
        bb = x[1, i] - x[1, j]
        distances = torch.stack((torch.linalg.vector_norm(aa, dim=-1), torch.linalg.vector_norm(ab, dim=-1),
                                 torch.linalg.vector_norm(ba, dim=-1), torch.linalg.vector_norm(bb, dim=-1)), dim=0)
        kernels, _ = obj._kernel_block(x, i, j, p_t, with_gradient=False)
        stacked, _w = obj._block_mixture(kernels, p_t, obj._cis[index])
        total = stacked.sum(dim=0, keepdim=True)
        gamma_blocks.append((stacked / total).detach().cpu().numpy().astype(np.float64))
        distance_blocks.append(distances.detach().cpu().numpy().astype(np.float64))
    gamma = np.concatenate(gamma_blocks, axis=1)
    distances = np.concatenate(distance_blocks, axis=1)
    pair_i = obj._pair_i[observed_index].detach().cpu().numpy().astype(np.int64)
    pair_j = obj._pair_j[observed_index].detach().cpu().numpy().astype(np.int64)
    counts = obj._counts[observed_index].detach().cpu().numpy().astype(np.float64)
    cis = obj._cis[observed_index].detach().cpu().numpy().astype(bool)
    return {"gamma": gamma, "distances": distances, "pair_i": pair_i, "pair_j": pair_j,
            "counts": counts, "cis": cis, "p": np.asarray([p_value])}


def posterior_summary(tables: dict[str, np.ndarray]) -> dict[str, Any]:
    gamma = tables["gamma"]
    distances = tables["distances"]
    counts = tables["counts"]
    cis = tables["cis"]
    maximum = gamma.max(axis=0)
    entropy = -np.sum(np.where(gamma > 0.0, gamma * np.log(np.maximum(gamma, 1e-300)), 0.0), axis=0)
    argmax = gamma.argmax(axis=0)
    expected_distance = np.sum(gamma * distances, axis=0)
    out: dict[str, Any] = {"pair_count": int(len(counts)), "total_counts": float(counts.sum())}
    for label, mask in (("all", np.ones_like(cis)), ("intra", cis), ("inter", ~cis)):
        weight = counts[mask]
        total = float(weight.sum())
        out[label] = {
            "counts": total,
            "mean_entropy": float(np.sum(weight * entropy[mask]) / total) if total else None,
            "mean_max_posterior": float(np.sum(weight * maximum[mask]) / total) if total else None,
            "fraction_gamma_max_ge_0.9": float(np.sum(weight * (maximum[mask] >= 0.9)) / total) if total else None,
            "mean_expected_distance": float(np.sum(weight * expected_distance[mask]) / total) if total else None,
        }
    out["argmax"] = argmax
    out["counts"] = counts
    out["cis"] = cis
    out["expected_distance"] = expected_distance
    out["gamma"] = gamma
    return out


def map_switch_fraction(current_argmax: np.ndarray, reference_argmax: np.ndarray, counts: np.ndarray,
                        cis: np.ndarray) -> dict[str, Any]:
    switch = current_argmax != reference_argmax
    out = {}
    for label, mask in (("all", np.ones_like(cis)), ("intra", cis), ("inter", ~cis)):
        total = float(counts[mask].sum())
        out[label] = {"counts": total,
                      "map_switch_fraction": float(np.sum(counts[mask] * switch[mask]) / total) if total else None}
    return out


def whole_cell_rg(coordinates: np.ndarray) -> float:
    beads = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    center = beads.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((beads - center) ** 2, axis=1))))


def fine_normalized_rates(data: Any, coords: np.ndarray, p: float) -> np.ndarray:
    """marginal(G) 语义下每个 full-grid offdiag pair 的 r_sum。"""
    obj = build_objective(data, "A")
    theta = _theta_from(coords, p)
    values = torch.as_tensor(theta, dtype=torch.float64, device="cuda")
    x = obj._physics._map_raw(values[:-1].reshape(2, int(data.n_loci), 3))
    p_value, _ = contact_model.p_from_q(float(theta[-1]))
    p_t = torch.as_tensor(p_value, dtype=torch.float64, device="cuda")
    eprod = obj._fixed_e[obj._pair_i] * obj._fixed_e[obj._pair_j]
    rates = np.empty(int(data.n_pairs), dtype=np.float64)
    for start in range(0, int(data.n_pairs), obj.pair_block):
        stop = min(start + obj.pair_block, int(data.n_pairs))
        i = obj._pair_i[start:stop]
        j = obj._pair_j[start:stop]
        kernels, _ = obj._kernel_block(x, i, j, p_t, with_gradient=False)
        stacked, _w = obj._block_mixture(kernels, p_t, obj._cis[start:stop])
        total = stacked[0] + stacked[1] + stacked[2] + stacked[3]
        rates[start:stop] = (eprod[start:stop] * total).detach().cpu().numpy()
    return rates


def coarse_pair_index(data_coarse: Any, chromosome: np.ndarray, bin_start_bp: np.ndarray) -> np.ndarray:
    """数值 bin 映射：细 locus 的基因组起点 -> 粗层全局 locus 索引。"""
    local = (np.asarray(bin_start_bp, dtype=np.int64) // int(data_coarse.bin_size))
    offsets = np.asarray(data_coarse.offsets, dtype=np.int64)
    n_bins = np.asarray(data_coarse.n_bins, dtype=np.int64)
    local = np.minimum(local, n_bins[np.asarray(chromosome, dtype=np.int64)] - 1)
    return offsets[np.asarray(chromosome, dtype=np.int64)] + local


def pair_linear_index(data: Any, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """strict upper triangle 的线性下标 (i<j)。"""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    n = int(data.n_loci)
    return lo * (2 * n - lo - 1) // 2 + (hi - lo - 1)


__all__ = ["build_objective", "three_loss_components", "observed_pair_tables", "posterior_summary",
           "map_switch_fraction", "whole_cell_rg", "fine_normalized_rates", "coarse_pair_index",
           "pair_linear_index", "load_layer", "FULL_WEIGHTS", "AGGREGATES", "ROOT"]
