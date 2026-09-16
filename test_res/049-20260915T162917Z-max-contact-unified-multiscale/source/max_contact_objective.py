"""三个 loss 的统一 max-contact objective。

A marginal_G      : Noff*log(Zsum) - sum_C C*log(r_sum)          （= 045 原 G）
B hard_observed   : Noff*log(Zsum) - sum_C C*log(r_max)          （分类/MAP 状态目标）
C max_rate        : Noff*log(Zmax) - sum_C C*log(r_max)          （max 也进入归一化）

所有 loss 共用同一 e、K、p、diag_nll、candidate-independent 常数、物理正则。
max 的精确并列用均分 subgradient，不用 first-argmax。
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from frozen_imports import contact_model, shared_capture_objective  # noqa: F401
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective
from visibility_profile_base import (
    P_PRIOR_STRENGTH,
    VisibilityError,
    _finite_scalar,
)

LOSS_IDS = ("A", "B", "C")
LOSS_SPEC = {
    "A": {"name": "marginal_G", "z_aggregation": "sum", "observed_aggregation": "sum"},
    "B": {"name": "hard_observed", "z_aggregation": "sum", "observed_aggregation": "max"},
    "C": {"name": "max_rate", "z_aggregation": "max", "observed_aggregation": "max"},
}


class MaxContactObjective(SharedCaptureObjective):
    """A/B/C 三 loss 共用同一数值核心；loss A 精确委托 045 G 实现以保证 parity。"""

    def __init__(self, data: Any, loss_id: str, *, weights: PenaltyWeights | None = None, **kwargs: Any):
        loss_id = str(loss_id).upper()
        if loss_id not in LOSS_IDS:
            raise ValueError("loss_id must be one of A/B/C")
        self.loss_id = loss_id
        self.spec = dict(LOSS_SPEC[loss_id])
        # 父类只接受 S/G；loss A 需要父类 G 分支，B/C 用自己的聚合路径。
        super().__init__(data, model_id="G", weights=weights, **kwargs)

    # ---------------------------------------------------------------- helpers
    def _constant_k0(self) -> float:
        n_cis = float(self.data.raw_cis_offdiag)
        n_inter = float(self.data.raw_inter)
        n_off = n_cis + n_inter
        if n_off <= 0.0:
            return 0.0
        value = 0.0
        if n_cis > 0.0:
            value += n_cis * math.log(n_cis / n_off)
        if n_inter > 0.0:
            value += n_inter * math.log(n_inter / n_off)
        return float(value)

    @staticmethod
    def _tie_weights(stacked: torch.Tensor) -> torch.Tensor:
        """stacked: (4, B) -> 精确并列均分权重 (4, B)。"""
        maximum = stacked.max(dim=0, keepdim=True).values
        marked = (stacked == maximum).to(stacked.dtype)
        return marked / marked.sum(dim=0, keepdim=True)

    def _block_mixture(self, kernels, p_t: torch.Tensor, local_cis: torch.Tensor):
        kaa, kab, kba, kbb = kernels[0], kernels[1], kernels[2], kernels[3]
        quarter = torch.as_tensor(0.25, dtype=self.dtype, device=self.device)
        half = torch.as_tensor(0.5, dtype=self.dtype, device=self.device)
        w_same = torch.where(local_cis, half * p_t, quarter)
        w_cross = torch.where(local_cis, half * (1.0 - p_t), quarter)
        t0 = w_same * kaa
        t1 = w_cross * kab
        t2 = w_cross * kba
        t3 = w_same * kbb
        return torch.stack((t0, t1, t2, t3), dim=0), (w_same, w_cross)

    # ------------------------------------------------------- B/C value+gradient
    def _max_count_and_gradient(self, x: torch.Tensor, p_t: torch.Tensor, e: torch.Tensor,
                                need_gradient: bool):
        z_max = self.spec["z_aggregation"] == "max"
        obs_max = self.spec["observed_aggregation"] == "max"
        eprod = e[self._pair_i] * e[self._pair_j]
        n_off = float(self._group_total_cis) + float(self._group_total_inter)
        normalizer = float(self.data.raw_records)

        z_cis = torch.zeros((), dtype=self.dtype, device=self.device)
        z_inter = torch.zeros((), dtype=self.dtype, device=self.device)
        observed_log_cis = torch.zeros((), dtype=self.dtype, device=self.device)
        observed_log_inter = torch.zeros((), dtype=self.dtype, device=self.device)
        observed = self._counts > 0.0
        for start in range(0, self.data.n_pairs, self.pair_block):
            stop = min(start + self.pair_block, self.data.n_pairs)
            i = self._pair_i[start:stop]
            j = self._pair_j[start:stop]
            kernels, _ = self._kernel_block(x, i, j, p_t, with_gradient=False)
            local_cis = self._cis[start:stop]
            stacked, _ = self._block_mixture(kernels, p_t, local_cis)
            local_e = eprod[start:stop]
            r_sum = local_e * (stacked[0] + stacked[1] + stacked[2] + stacked[3])
            if z_max or obs_max:
                r_max = local_e * stacked.max(dim=0).values
            else:
                r_max = None
            r_z = r_max if z_max else r_sum
            z_cis = z_cis + r_z[local_cis].sum()
            z_inter = z_inter + r_z[~local_cis].sum()
            r_obs = r_max if obs_max else r_sum
            mask_cis = observed[start:stop] & local_cis
            mask_inter = observed[start:stop] & ~local_cis
            if bool(torch.any(mask_cis)):
                observed_log_cis = observed_log_cis + (
                    self._counts[start:stop][mask_cis] * torch.log(r_obs[mask_cis])).sum()
            if bool(torch.any(mask_inter)):
                observed_log_inter = observed_log_inter + (
                    self._counts[start:stop][mask_inter] * torch.log(r_obs[mask_inter])).sum()

        z_total = z_cis + z_inter
        if _finite_scalar(z_total) <= 0.0:
            raise VisibilityError("max-contact normalizer is nonpositive")
        observed_log_total = observed_log_cis + observed_log_inter
        conditional = n_off * torch.log(z_total) - observed_log_total
        constant_k0 = self._constant_k0()
        positive = self._diag_counts[self._diag_counts > 0.0]
        diag = (positive - positive * torch.log(positive)).sum()
        if self.data.count_mode in ("raw_integer", "synthetic_integer"):
            diag = diag + torch.lgamma(positive + 1.0).sum()
        count_raw = conditional + constant_k0 + diag

        predicted_cis_mass = z_cis / z_total
        predicted_inter_mass = z_inter / z_total
        components = {
            "count_mode": str(self.data.count_mode),
            "model_id": self.loss_id,
            "loss_name": self.spec["name"],
            "normalization_model": self.loss_id,
            "normalization_aggregation": self.spec["z_aggregation"],
            "observed_aggregation": self.spec["observed_aggregation"],
            "count_factorial_status": (
                "integer_count_constants_recorded"
                if self.data.count_mode in ("raw_integer", "synthetic_integer")
                else "not_defined_for_fractional_expected_cross_entropy"),
            "sum_rate_cis_offdiag": _finite_scalar(z_cis),
            "sum_rate_inter": _finite_scalar(z_inter),
            "sum_rate_offdiag": _finite_scalar(z_total),
            "observed_log_rate_cis_offdiag": _finite_scalar(observed_log_cis),
            "observed_log_rate_inter": _finite_scalar(observed_log_inter),
            "observed_log_rate_total": _finite_scalar(observed_log_total),
            "separate_conditional_nll_raw": _finite_scalar(conditional),
            "candidate_independent_constant_raw": float(constant_k0),
            "candidate_independent_constant_included": True,
            "group_mass_kl_raw": None,
            "group_mass_kl_normalized": None,
            "observed_cis_mass": (float(self.data.raw_cis_offdiag) / n_off) if n_off > 0.0 else 0.0,
            "observed_inter_mass": (float(self.data.raw_inter) / n_off) if n_off > 0.0 else 0.0,
            "predicted_cis_mass": _finite_scalar(predicted_cis_mass),
            "predicted_inter_mass": _finite_scalar(predicted_inter_mass),
            "profiled_intensity_cis": None,
            "profiled_intensity_inter": None,
            "shared_profiled_intensity": _finite_scalar(n_off / z_total),
            "conditional_nll_raw": _finite_scalar(conditional),
            "conditional_factorial_constant_omitted": (
                float(self.data.conditional_factorial_constant)
                if math.isfinite(float(self.data.conditional_factorial_constant)) else None),
            "diag_factorial_sum_included": (
                float(self.data.diag_factorial_sum)
                if math.isfinite(float(self.data.diag_factorial_sum)) else None),
            "diag_profiled_nll_raw": _finite_scalar(diag),
            "count_nll_raw": _finite_scalar(count_raw),
            "count_nll_normalized": _finite_scalar(count_raw / normalizer),
        }
        if not need_gradient:
            return components, None, None

        z_coefficient = n_off / z_total
        gradient_x = torch.zeros_like(x)
        gradient_p = torch.zeros((), dtype=self.dtype, device=self.device)
        half = torch.as_tensor(0.5, dtype=self.dtype, device=self.device)
        for start in range(0, self.data.n_pairs, self.pair_block):
            stop = min(start + self.pair_block, self.data.n_pairs)
            i = self._pair_i[start:stop]
            j = self._pair_j[start:stop]
            kernels, gradients = self._kernel_block(x, i, j, p_t, with_gradient=True)
            kaa, kab, kba, kbb = kernels[0], kernels[1], kernels[2], kernels[3]
            gaa, gab, gba, gbb = gradients[0], gradients[1], gradients[2], gradients[3]
            local_cis = self._cis[start:stop]
            stacked, _weights = self._block_mixture(kernels, p_t, local_cis)
            local_e = eprod[start:stop]
            r_sum = local_e * (stacked[0] + stacked[1] + stacked[2] + stacked[3])
            if z_max or obs_max:
                tie = self._tie_weights(stacked)
                r_max = local_e * stacked.max(dim=0).values
            else:
                tie = None
                r_max = None
            r_obs = r_max if obs_max else r_sum
            local_counts = self._counts[start:stop]
            observed_weight = local_counts / r_obs  # C=0 处为 0，保持有限
            # z 聚合的 subgradient 权重 a_s 与观测聚合的 b_s（sum 分支为全 1）
            use_tie_for_z = z_max
            use_tie_for_o = obs_max
            tie_stack = tie if (use_tie_for_z or use_tie_for_o) else None
            one = torch.as_tensor(1.0, dtype=self.dtype, device=self.device)

            def subgradient_weights(use_tie: bool):
                if not use_tie:
                    return (one, one, one, one)
                return (tie_stack[0], tie_stack[1], tie_stack[2], tie_stack[3])

            a_weights = subgradient_weights(use_tie_for_z)
            b_weights = subgradient_weights(use_tie_for_o)
            quarter = torch.as_tensor(0.25, dtype=self.dtype, device=self.device)
            w_same = torch.where(local_cis, half * p_t, quarter)
            w_cross = torch.where(local_cis, half * (1.0 - p_t), quarter)
            w_states = (w_same, w_cross, w_cross, w_same)
            # d r_agg/dx 的核系数：eprod * w_s * subgradient_s
            coefficients = []
            for state in range(4):
                coefficients.append(local_e * (z_coefficient * w_states[state] * a_weights[state]
                                               - observed_weight * w_states[state] * b_weights[state]))
            for copy_i, copy_j, kernel_gradient, coefficient_value in (
                    (0, 0, gaa, coefficients[0]), (0, 1, gab, coefficients[1]),
                    (1, 0, gba, coefficients[2]), (1, 1, gbb, coefficients[3])):
                term = coefficient_value.unsqueeze(-1) * kernel_gradient
                gradient_x[copy_i].index_add_(0, i, term)
                gradient_x[copy_j].index_add_(0, j, -term)
            if bool(torch.any(local_cis)):
                # d r_agg/dp = eprod * sum_s subgradient_s * (dw_s/dp) * K_s，
                # cis 的 dw/dp = [+.5, -.5, -.5, +.5]，inter 为 0。
                # 注意：这里不能再乘 w_s —— 乘了 w_s 会把 a_s 替换成 w_s*a_s。
                dp_z = local_e * half * (a_weights[0] * kaa - a_weights[1] * kab
                                         - a_weights[2] * kba + a_weights[3] * kbb)
                dp_o = local_e * half * (b_weights[0] * kaa - b_weights[1] * kab
                                         - b_weights[2] * kba + b_weights[3] * kbb)
                local_dp = (z_coefficient * dp_z - observed_weight * dp_o)[local_cis]
                gradient_p = gradient_p + local_dp.sum()
        return components, gradient_x / normalizer, gradient_p / normalizer

    # --------------------------------------------------------------- dispatch
    def _count_layer(self, x: torch.Tensor, p_t: torch.Tensor, e: torch.Tensor, need_gradient: bool):
        if self.loss_id == "A":
            mixture = self._build_k(x, p_t)
            return SharedCaptureObjective._count_and_gradient(self, x, p_t, mixture, e, need_gradient)
        return self._max_count_and_gradient(x, p_t, e, need_gradient)

    def evaluate(self, theta: np.ndarray, need_gradient: bool = True):
        started = time.perf_counter()
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,) or not np.all(np.isfinite(theta)):
            raise VisibilityError("theta has wrong shape or nonfinite values")
        theta_t = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        raw = theta_t[:-1].reshape(2, self.data.n_loci, 3)
        x = self._physics._map_raw(raw)
        self._physics.validate_physical_coordinates_tensor(x)
        p, dpdq = contact_model.p_from_q(float(theta[-1]))
        p_t = torch.as_tensor(p, dtype=self.dtype, device=self.device)
        count_components, count_gradient_x, count_gradient_p = self._count_layer(x, p_t, self._fixed_e, need_gradient)
        bond, bond_gradient, repulsion, repulsion_gradient, bend, bend_gradient = self._physics_layers(x)
        p_prior = -P_PRIOR_STRENGTH * torch.log(p_t * (1.0 - p_t))
        p_prior_derivative = P_PRIOR_STRENGTH * (1.0 / (1.0 - p_t) - 1.0 / p_t)
        _fixed_values, visibility = self._fixed_visibility()
        components = dict(count_components)
        components.update({
            "visibility_mode": self.mode,
            "model_id": self.loss_id,
            "loss_name": self.spec["name"],
            "normalization_model": self.loss_id,
            "normalization_aggregation": self.spec["z_aggregation"],
            "observed_aggregation": self.spec["observed_aggregation"],
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
            raise FloatingPointError("nonfinite max-contact objective or gradient")
        self._last_visibility = visibility
        self._sync()
        self._counters["outer_wall_seconds"] += time.perf_counter() - started
        return components["total"], gradient_np.copy(), components

    def count_value_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,) or not np.all(np.isfinite(theta)):
            raise VisibilityError("theta has wrong shape or nonfinite values")
        theta_t = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        raw = theta_t[:-1].reshape(2, self.data.n_loci, 3)
        x = self._physics._map_raw(raw)
        self._physics.validate_physical_coordinates_tensor(x)
        p, dpdq = contact_model.p_from_q(float(theta[-1]))
        p_t = torch.as_tensor(p, dtype=self.dtype, device=self.device)
        components, gradient_x, gradient_p = self._count_layer(x, p_t, self._fixed_e, True)
        gradient_raw = self._physics._pullback(raw, gradient_x)
        gradient_q = torch.as_tensor(dpdq, dtype=self.dtype, device=self.device) * gradient_p
        gradient = torch.cat((gradient_raw.reshape(-1), gradient_q.reshape(1)))
        result = gradient.detach().cpu().numpy().astype(np.float64, copy=True)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite max-contact count gradient")
        self._counters["outer_evaluations"] += 1
        return float(components["count_nll_normalized"]), result

    def diagnostics(self) -> dict[str, Any]:
        result = super().diagnostics()
        result.update({
            "loss_id": self.loss_id,
            "loss_name": self.spec["name"],
            "z_aggregation": self.spec["z_aggregation"],
            "observed_aggregation": self.spec["observed_aggregation"],
            "exact_tie_rule": "equal split subgradient over exactly tied states; no first-argmax",
            "zero_pairs_retained": True,
        })
        return result


__all__ = ["MaxContactObjective", "LOSS_SPEC", "LOSS_IDS", "PenaltyWeights"]
