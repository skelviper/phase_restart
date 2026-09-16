"""S/G shared-capture objective wrapper.

本模块不修改 041 backend。S 是 cis/inter 分组各自完整 grid normalization，G
是全 offdiag 完整 grid 的共同 normalization；两者保留相同 observed-log rate 项，
G 比 S 多出 group-mass KL。所有物理层、球坐标和 fixed-e/known-e 支持复用冻结
visibility_profile_base.py。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch

from visibility_profile_base import (
    EPSILON,
    P_PRIOR_STRENGTH,
    PROFILE_RESIDUAL_TOL,
    VisibilityError,
    VisibilityGPUObjective,
    _finite_scalar,
)
from pr import contact_model


@dataclass(frozen=True)
class PenaltyWeights:
    """J 的显式组件权重；p_prior 自带 1e-4，不在此处重复缩放。"""

    count: float = 1.0
    bond: float = 1.0
    repulsion: float = 1.0
    bend: float = 0.01
    p_prior: float = 1.0

    def validate(self) -> None:
        values = (self.count, self.bond, self.repulsion, self.bend, self.p_prior)
        if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
            raise ValueError("penalty weights must be finite and nonnegative")

    def as_dict(self) -> dict[str, float]:
        return {
            "count": float(self.count),
            "bond": float(self.bond),
            "repulsion": float(self.repulsion),
            "bend": float(self.bend),
            "p_prior": float(self.p_prior),
        }


class SharedCaptureObjective(VisibilityGPUObjective):
    """可配置 S/G count objective 加冻结物理 penalty。"""

    MODEL_IDS = frozenset(("S", "G"))

    def __init__(self, data: Any, model_id: str, *, weights: PenaltyWeights | None = None,
                 **kwargs: Any):
        model_id = str(model_id).upper()
        if model_id not in self.MODEL_IDS:
            raise ValueError("model_id must be S or G")
        self.model_id = model_id
        self.weights = weights if weights is not None else PenaltyWeights()
        self.weights.validate()
        super().__init__(data, **kwargs)

    def _normalizer_coefficients(self, z_cis: torch.Tensor, z_inter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_cis = self._group_total_cis
        n_inter = self._group_total_inter
        n_off = n_cis + n_inter
        total = z_cis + z_inter
        if _finite_scalar(total) <= 0.0:
            raise VisibilityError("shared offdiag full-grid rate sum is nonpositive")
        if self.model_id == "S":
            cis_coeff = n_cis / z_cis if n_cis > 0.0 else torch.zeros_like(z_cis)
            inter_coeff = n_inter / z_inter if n_inter > 0.0 else torch.zeros_like(z_inter)
        else:
            shared = n_off / total
            cis_coeff = shared
            inter_coeff = shared
        return cis_coeff, inter_coeff, total

    def _count_and_gradient(self, x: torch.Tensor, p: torch.Tensor, k: torch.Tensor,
                            e: torch.Tensor, need_gradient: bool):
        """S/G 共享计数目标；完整 eligible grid 和 zero pairs 始终保留。"""
        eprod = e[self._pair_i] * e[self._pair_j]
        rate = eprod * k
        z_cis = rate[self._cis].sum()
        z_inter = rate[~self._cis].sum()
        if (self._group_total_cis > 0.0 and _finite_scalar(z_cis) <= 0.0) or (
                self._group_total_inter > 0.0 and _finite_scalar(z_inter) <= 0.0):
            raise VisibilityError("positive contact group has nonpositive full-grid rate sum")
        observed_log_cis = torch.zeros((), dtype=self.dtype, device=self.device)
        observed_log_inter = torch.zeros((), dtype=self.dtype, device=self.device)
        observed = self._counts > 0.0
        if bool(torch.any(observed & self._cis)):
            observed_log_cis = (self._counts[observed & self._cis]
                                * torch.log(rate[observed & self._cis])).sum()
        if bool(torch.any(observed & ~self._cis)):
            observed_log_inter = (self._counts[observed & ~self._cis]
                                  * torch.log(rate[observed & ~self._cis])).sum()
        separate_conditional = (
            self._group_total_cis * torch.log(z_cis) - observed_log_cis
            + self._group_total_inter * torch.log(z_inter) - observed_log_inter
        )
        _, _, z_total = self._normalizer_coefficients(z_cis, z_inter)
        n_off = self._group_total_cis + self._group_total_inter
        group_kl_raw = torch.zeros((), dtype=self.dtype, device=self.device)
        if n_off > 0.0:
            for n_group, z_group in ((self._group_total_cis, z_cis),
                                     (self._group_total_inter, z_inter)):
                if n_group > 0.0:
                    group_kl_raw = group_kl_raw + n_group * (
                        torch.log(n_group / n_off) - torch.log(z_group / z_total))
        effective_conditional = separate_conditional + (
            group_kl_raw if self.model_id == "G" else torch.zeros_like(group_kl_raw)
        )
        positive = self._diag_counts[self._diag_counts > 0.0]
        diag = (positive - positive * torch.log(positive)).sum()
        if self.data.count_mode in ("raw_integer", "synthetic_integer"):
            diag = diag + torch.lgamma(positive + 1.0).sum()
        count_raw = effective_conditional + diag
        normalizer = float(self.data.raw_records)
        cis_mass = self._group_total_cis / n_off if n_off > 0.0 else torch.zeros_like(n_off)
        inter_mass = self._group_total_inter / n_off if n_off > 0.0 else torch.zeros_like(n_off)
        predicted_cis_mass = z_cis / z_total
        predicted_inter_mass = z_inter / z_total
        components = {
            "count_mode": str(self.data.count_mode),
            "normalization_model": self.model_id,
            "count_factorial_status": (
                "integer_count_constants_recorded"
                if self.data.count_mode in ("raw_integer", "synthetic_integer")
                else "not_defined_for_fractional_expected_cross_entropy"
            ),
            "sum_rate_cis_offdiag": _finite_scalar(z_cis),
            "sum_rate_inter": _finite_scalar(z_inter),
            "sum_rate_offdiag": _finite_scalar(z_total),
            "observed_log_rate_cis_offdiag": _finite_scalar(observed_log_cis),
            "observed_log_rate_inter": _finite_scalar(observed_log_inter),
            "observed_log_rate_total": _finite_scalar(observed_log_cis + observed_log_inter),
            "separate_conditional_nll_raw": _finite_scalar(separate_conditional),
            "group_mass_kl_raw": _finite_scalar(group_kl_raw),
            "group_mass_kl_normalized": _finite_scalar(group_kl_raw / normalizer),
            "observed_cis_mass": _finite_scalar(cis_mass),
            "observed_inter_mass": _finite_scalar(inter_mass),
            "predicted_cis_mass": _finite_scalar(predicted_cis_mass),
            "predicted_inter_mass": _finite_scalar(predicted_inter_mass),
            "profiled_intensity_cis": _finite_scalar(self._group_total_cis / z_cis),
            "profiled_intensity_inter": _finite_scalar(self._group_total_inter / z_inter),
            "shared_profiled_intensity": _finite_scalar(n_off / z_total),
            "conditional_nll_raw": _finite_scalar(effective_conditional),
            "conditional_factorial_constant_omitted": (
                float(self.data.conditional_factorial_constant)
                if math.isfinite(float(self.data.conditional_factorial_constant)) else None
            ),
            "diag_factorial_sum_included": (
                float(self.data.diag_factorial_sum)
                if math.isfinite(float(self.data.diag_factorial_sum)) else None
            ),
            "diag_profiled_nll_raw": _finite_scalar(diag),
            "count_nll_raw": _finite_scalar(count_raw),
            "count_nll_normalized": _finite_scalar(count_raw / normalizer),
        }
        if not need_gradient:
            return components, None, None

        cis_coeff, inter_coeff, _ = self._normalizer_coefficients(z_cis, z_inter)
        gradient_x = torch.zeros_like(x)
        gradient_p = torch.zeros((), dtype=self.dtype, device=self.device)
        weights = torch.zeros_like(rate)
        pair_positive = eprod > 0.0
        if self.model_id == "G" or self._group_total_cis > 0.0:
            mask = pair_positive & self._cis
            weights[mask] = cis_coeff - self._counts[mask] / rate[mask]
        if self.model_id == "G" or self._group_total_inter > 0.0:
            mask = pair_positive & ~self._cis
            weights[mask] = inter_coeff - self._counts[mask] / rate[mask]
        quarter = torch.as_tensor(0.25, dtype=self.dtype, device=self.device)
        for start in range(0, self.data.n_pairs, self.pair_block):
            stop = min(start + self.pair_block, self.data.n_pairs)
            i = self._pair_i[start:stop]
            j = self._pair_j[start:stop]
            kernels, gradients = self._kernel_block(x, i, j, p, with_gradient=True)
            kaa, kab, kba, kbb, _same, _cross = kernels
            gaa, gab, gba, gbb = gradients
            local_e = eprod[start:stop]
            local_weights = weights[start:stop]
            local_cis = self._cis[start:stop]
            same_coefficient = local_e * torch.where(
                local_cis, 0.5 * p, quarter) * local_weights
            cross_coefficient = local_e * torch.where(
                local_cis, 0.5 * (1.0 - p), quarter) * local_weights
            for copy_i, copy_j, kernel_gradient, coefficient in (
                    (0, 0, gaa, same_coefficient), (0, 1, gab, cross_coefficient),
                    (1, 0, gba, cross_coefficient), (1, 1, gbb, same_coefficient)):
                term = coefficient.unsqueeze(-1) * kernel_gradient
                gradient_x[copy_i].index_add_(0, i, term)
                gradient_x[copy_j].index_add_(0, j, -term)
            derivative_p = 0.5 * local_e * (kaa + kbb - kab - kba)
            if bool(torch.any(local_cis)):
                gradient_p = gradient_p + (
                    local_weights[local_cis] * derivative_p[local_cis]).sum()
        return components, gradient_x / normalizer, gradient_p / normalizer

    def _prepare_count(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,) or not np.all(np.isfinite(theta)):
            raise VisibilityError("theta has wrong shape or nonfinite values")
        theta_t = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        raw = theta_t[:-1].reshape(2, self.data.n_loci, 3)
        x = self._physics._map_raw(raw)
        self._physics.validate_physical_coordinates_tensor(x)
        p, dpdq = contact_model.p_from_q(float(theta[-1]))
        p_t = torch.as_tensor(p, dtype=self.dtype, device=self.device)
        k = self._build_k(x, p_t)
        self._sync()
        if self.mode == "V1-profile-e":
            warm = self._warm_eta if self.profile_warm_start else None
            eta, visibility = self._profile(k, warm_eta=warm)
            self._warm_eta = eta.detach().clone()
            e = torch.as_tensor(visibility["e"], dtype=self.dtype, device=self.device)
        else:
            _fixed_values, visibility = self._fixed_visibility()
            e = self._fixed_e
            assert e is not None
        return theta_t, raw, x, p, dpdq, p_t, k, e, visibility

    def evaluate(self, theta: np.ndarray, need_gradient: bool = True):
        """返回 S/G J、raw-y/q gradient、组件和 normalization audit。"""
        started = time.perf_counter()
        _theta_t, raw, x, p, dpdq, p_t, k, e, visibility = self._prepare_count(theta)
        count_components, count_gradient_x, count_gradient_p = self._count_and_gradient(
            x, p_t, k, e, need_gradient)
        bond, bond_gradient, repulsion, repulsion_gradient, bend, bend_gradient = self._physics_layers(x)
        p_prior = -P_PRIOR_STRENGTH * torch.log(p_t * (1.0 - p_t))
        p_prior_derivative = P_PRIOR_STRENGTH * (1.0 / (1.0 - p_t) - 1.0 / p_t)
        components = dict(count_components)
        components.update({
            "visibility_mode": self.mode,
            "model_id": self.model_id,
            "p": float(p),
            "p_prior": _finite_scalar(p_prior),
            "bond": _finite_scalar(bond),
            "repulsion": _finite_scalar(repulsion),
            "bend": _finite_scalar(bend),
            "repulsion_threshold_l0": 0.7,
            "penalty_weights": self.weights.as_dict(),
            "profile": {key: value for key, value in visibility.items()
                         if key not in ("eta", "e", "predicted_degree", "degree")},
        })
        total = (
            self.weights.count * torch.as_tensor(components["count_nll_normalized"], dtype=self.dtype, device=self.device)
            + self.weights.p_prior * p_prior
            + self.weights.bond * bond
            + self.weights.repulsion * repulsion
            + self.weights.bend * bend
        )
        components["weighted_count"] = float(self.weights.count) * components["count_nll_normalized"]
        components["weighted_p_prior"] = float(self.weights.p_prior) * components["p_prior"]
        components["weighted_bond"] = float(self.weights.bond) * components["bond"]
        components["weighted_repulsion"] = float(self.weights.repulsion) * components["repulsion"]
        components["weighted_bend"] = float(self.weights.bend) * components["bend"]
        components["total"] = _finite_scalar(total)
        self._counters["outer_evaluations"] += 1
        if not need_gradient:
            self._last_visibility = visibility
            self._sync()
            self._counters["outer_wall_seconds"] += time.perf_counter() - started
            return components["total"], None, components
        combined_x = (
            self.weights.count * count_gradient_x
            + self.weights.bond * bond_gradient
            + self.weights.repulsion * repulsion_gradient
            + self.weights.bend * bend_gradient
        )
        gradient_raw = self._physics._pullback(raw, combined_x)
        gradient_p_physical = self.weights.count * count_gradient_p + self.weights.p_prior * p_prior_derivative
        self._last_physical_gradient = torch.cat(
            (combined_x.reshape(-1), gradient_p_physical.reshape(1))
        ).detach().cpu().numpy().astype(np.float64, copy=True)
        gradient_q = torch.as_tensor(dpdq, dtype=self.dtype, device=self.device) * gradient_p_physical
        gradient = torch.cat((gradient_raw.reshape(-1), gradient_q.reshape(1)))
        gradient_np = gradient.detach().cpu().numpy().astype(np.float64, copy=False)
        if not np.all(np.isfinite(gradient_np)) or not math.isfinite(components["total"]):
            raise FloatingPointError("nonfinite shared-capture objective or gradient")
        self._last_visibility = visibility
        self._sync()
        self._counters["outer_wall_seconds"] += time.perf_counter() - started
        return components["total"], gradient_np.copy(), components

    def count_value_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """返回只含 S/G count normalization 的 envelope gradient。"""
        _theta_t, raw, x, _p, _dpdq, p_t, k, e, visibility = self._prepare_count(theta)
        components, gradient_x, gradient_p = self._count_and_gradient(
            x, p_t, k, e, need_gradient=True)
        gradient_raw = self._physics._pullback(raw, gradient_x)
        _p_value, dpdq = contact_model.p_from_q(float(theta[-1]))
        gradient_q = torch.as_tensor(dpdq, dtype=self.dtype, device=self.device) * gradient_p
        gradient = torch.cat((gradient_raw.reshape(-1), gradient_q.reshape(1)))
        result = gradient.detach().cpu().numpy().astype(np.float64, copy=True)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite shared-capture count gradient")
        self._last_visibility = visibility
        self._counters["outer_evaluations"] += 1
        return float(components["count_nll_normalized"]), result

    def diagnostics(self) -> dict[str, Any]:
        result = super().diagnostics()
        result.update({
            "model_id": self.model_id,
            "penalty_weights": self.weights.as_dict(),
            "normalization_definition": (
                "S: Ncis/Scis and Ninter/Sinter; "
                "G: Noff/(Scis+Sinter), with group-mass KL added to S"
            ),
            "observed_log_rate_unchanged_between_S_G": True,
        })
        return result


__all__ = ["PenaltyWeights", "SharedCaptureObjective", "PROFILE_RESIDUAL_TOL"]
