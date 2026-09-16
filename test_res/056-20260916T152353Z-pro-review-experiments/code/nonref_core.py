"""056 实验1/2的训练侧固定状态 full-grid 数值核。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.special import gammaln
import torch

ROOT = Path(__file__).resolve().parents[3]
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
if str(S045) not in sys.path:
    sys.path.insert(0, str(S045))

import data_io  # noqa: E402
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective  # noqa: E402
from pr import contact_model  # noqa: E402

PAIR_BLOCK = 262_144
WEIGHTS = PenaltyWeights()


def array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, order="C")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite JSON value")
        return float(value)
    return value


def make_objective(data: Any) -> SharedCaptureObjective:
    return SharedCaptureObjective(
        data, "G", weights=WEIGHTS, mode="V0-fixed-production-e", device="cuda",
        pair_block=PAIR_BLOCK, inner_cap=80, cg_cap=80, profile_warm_start=True,
        known_e=None,
    )


def fixed_state_fullgrid(data: Any, coords: np.ndarray, p: float,
                         *, keep_inter_arrays: bool,
                         include_penalties: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """单次 pair-grid forward 同时计算G data、inter不变量和物理penalty。"""
    coords = np.asarray(coords, dtype=np.float64)
    contact_model.assert_inside_unit_ball(coords)
    obj = make_objective(data)
    raw_y = contact_model.sphere_inverse(coords)
    raw = torch.as_tensor(raw_y, dtype=torch.float64, device="cuda")
    x = obj._physics._map_raw(raw)
    obj._physics.validate_physical_coordinates_tensor(x)
    p_t = torch.as_tensor(float(p), dtype=torch.float64, device="cuda")
    eprod = obj._fixed_e[obj._pair_i] * obj._fixed_e[obj._pair_j]
    counts = obj._counts
    z_cis = torch.zeros((), dtype=torch.float64, device="cuda")
    z_inter = torch.zeros((), dtype=torch.float64, device="cuda")
    observed_log = torch.zeros((), dtype=torch.float64, device="cuda")
    inter_rates: list[np.ndarray] = []
    inter_sorted_distances: list[np.ndarray] = []
    quarter = torch.as_tensor(0.25, dtype=torch.float64, device="cuda")
    half = torch.as_tensor(0.5, dtype=torch.float64, device="cuda")
    for start in range(0, int(data.n_pairs), PAIR_BLOCK):
        stop = min(start + PAIR_BLOCK, int(data.n_pairs))
        i = obj._pair_i[start:stop]
        j = obj._pair_j[start:stop]
        cis = obj._cis[start:stop]
        kernels, _ = obj._kernel_block(x, i, j, p_t, with_gradient=False)
        kaa, kab, kba, kbb = kernels[:4]
        w_same = torch.where(cis, half * p_t, quarter)
        w_cross = torch.where(cis, half * (1.0 - p_t), quarter)
        rate = eprod[start:stop] * (w_same * (kaa + kbb) + w_cross * (kab + kba))
        z_cis = z_cis + rate[cis].sum()
        z_inter = z_inter + rate[~cis].sum()
        positive = counts[start:stop] > 0.0
        if bool(torch.any(positive)):
            observed_log = observed_log + (counts[start:stop][positive] * torch.log(rate[positive])).sum()
        if keep_inter_arrays and bool(torch.any(~cis)):
            local_i = i[~cis]
            local_j = j[~cis]
            delta = torch.stack((
                x[0, local_i] - x[0, local_j],
                x[0, local_i] - x[1, local_j],
                x[1, local_i] - x[0, local_j],
                x[1, local_i] - x[1, local_j],
            ), dim=0)
            distance = torch.linalg.vector_norm(delta, dim=2)
            inter_rates.append(rate[~cis].detach().cpu().numpy().astype(np.float64))
            inter_sorted_distances.append(
                torch.sort(distance, dim=0).values.detach().cpu().numpy().astype(np.float64))
    obj.synchronize()
    zc = float(z_cis.detach().cpu().item())
    zi = float(z_inter.detach().cpu().item())
    zo = zc + zi
    obs = float(observed_log.detach().cpu().item())
    n_cis = float(data.raw_cis_offdiag)
    n_inter = float(data.raw_inter)
    n_off = n_cis + n_inter
    k0 = n_cis * math.log(n_cis / n_off) + n_inter * math.log(n_inter / n_off)
    conditional_no_k0 = n_off * math.log(zo) - obs
    conditional = conditional_no_k0 + k0
    positive_diag = np.asarray(data.diag_counts, dtype=np.float64)
    positive_diag = positive_diag[positive_diag > 0.0]
    diag = float((positive_diag - positive_diag * np.log(positive_diag)
                  + gammaln(positive_diag + 1.0)).sum())
    count_raw = conditional + diag
    count_per_raw = count_raw / float(data.raw_records)
    offdiag_per_contact = conditional / n_off
    offdiag_no_k0_per_contact = conditional_no_k0 / n_off
    raw_penalties = {}
    weighted = {}
    total = count_per_raw
    if include_penalties:
        bond, _bg, repulsion, _rg, bend, _beg = obj._physics_layers(x)
        p_prior = -1e-4 * torch.log(p_t * (1.0 - p_t))
        raw_penalties = {
            "bond": float(bond.detach().cpu().item()),
            "repulsion": float(repulsion.detach().cpu().item()),
            "bend": float(bend.detach().cpu().item()),
            "p_prior": float(p_prior.detach().cpu().item()),
        }
        weighted = {
            "bond": raw_penalties["bond"],
            "repulsion": raw_penalties["repulsion"],
            "bend": 0.01 * raw_penalties["bend"],
            "p_prior": raw_penalties["p_prior"],
        }
        total += sum(weighted.values())
    record = {
        "p": float(p), "Nraw": float(data.raw_records), "Noff": n_off,
        "Ncis": n_cis, "Ninter": n_inter, "K0": k0,
        "Zcis": zc, "Zinter": zi, "Zall": zo,
        "observed_log_rate_offdiag": obs,
        "conditional_no_K0_raw": conditional_no_k0,
        "conditional_with_K0_raw": conditional,
        "offdiag_data_no_K0_nat_per_contact": offdiag_no_k0_per_contact,
        "offdiag_data_nat_per_contact": offdiag_per_contact,
        "diag_profiled_nll_raw": diag,
        "count_nll_raw": count_raw,
        "count_nll_per_raw_record": count_per_raw,
        "penalties_raw": raw_penalties,
        "penalties_weighted": weighted,
        "total_normalized": total,
        "objective_fg_calls": 1,
        "pair_kernel_forward_passes": 1,
        "regularizer_full_pair_passes": 1 if include_penalties else 0,
    }
    arrays: dict[str, np.ndarray] = {}
    if keep_inter_arrays:
        rates = np.concatenate(inter_rates)
        distances = np.concatenate(inter_sorted_distances, axis=1)
        arrays = {
            "inter_mixture_rate": rates,
            "inter_normalized_rate": rates / zi,
            "inter_sorted_four_distances": distances,
        }
        record["inter_arrays"] = {
            key: {"shape": list(value.shape), "sha256": array_sha256(value)}
            for key, value in arrays.items()
        }
    return record, arrays


def strict_pointset_equal(original: np.ndarray, changed: np.ndarray) -> bool:
    original = np.asarray(original)
    changed = np.asarray(changed)
    direct = np.all(changed[0] == original[0], axis=1) & np.all(changed[1] == original[1], axis=1)
    swapped = np.all(changed[0] == original[1], axis=1) & np.all(changed[1] == original[0], axis=1)
    return bool(np.all(direct | swapped))


def diag_variant_data(data: Any, diag: np.ndarray, exposure: np.ndarray) -> Any:
    diag = np.asarray(diag, dtype=np.int64)
    exposure = np.asarray(exposure, dtype=np.float64)
    offdiag_endpoints = np.asarray(data.endpoint_counts, dtype=np.int64) - 2 * np.asarray(data.diag_counts, dtype=np.int64)
    endpoints = offdiag_endpoints + 2 * diag
    raw_same = int(diag.sum())
    diag_factorial = float(gammaln(diag[diag > 1].astype(np.float64) + 1.0).sum())
    result = replace(
        data, diag_counts=diag.copy(), endpoint_counts=endpoints,
        exposure=exposure.copy(), raw_same_bin=raw_same,
        raw_records=int(data.raw_cis_offdiag) + int(data.raw_inter) + raw_same,
        diag_factorial_sum=diag_factorial,
    )
    result.assert_consistent()
    return result


def physical_count_cell(data: Any, coords: np.ndarray, p: float,
                        denominator: float) -> tuple[dict[str, Any], np.ndarray]:
    """一次 frozen G value+gradient call，返回 physical-x count-only gradient。"""
    obj = make_objective(data)
    raw_y = contact_model.sphere_inverse(np.asarray(coords, dtype=np.float64))
    q = float(contact_model.q_from_p(float(p)))
    theta = np.concatenate((raw_y.reshape(-1), np.asarray([q], dtype=np.float64)))
    _theta_t, _raw, _x, _p, _dpdq, p_t, k, e, _visibility = obj._prepare_count(theta)
    components, gradient_x, _gradient_p = obj._count_and_gradient(_x, p_t, k, e, need_gradient=True)
    native_denominator = float(data.raw_records)
    scale = native_denominator / float(denominator)
    gradient = gradient_x.detach().cpu().numpy().astype(np.float64) * scale
    value = float(components["count_nll_normalized"]) * scale
    conditional = float(components["conditional_nll_raw"]) / float(denominator)
    return {
        "denominator": float(denominator), "native_Nraw": native_denominator,
        "scale_from_native": scale, "count_value": value,
        "conditional_value": conditional,
        "count_nll_raw": float(components["count_nll_raw"]),
        "conditional_nll_raw": float(components["conditional_nll_raw"]),
        "gradient_l2": float(np.linalg.norm(gradient)),
        "gradient_linf": float(np.max(np.abs(gradient))),
        "gradient_sha256": array_sha256(gradient),
        "objective_fg_calls": 1,
        "pair_kernel_passes": 2,
    }, gradient


__all__ = [
    "array_sha256", "data_io", "diag_variant_data", "fixed_state_fullgrid",
    "jsonable", "physical_count_cell", "strict_pointset_equal",
]
