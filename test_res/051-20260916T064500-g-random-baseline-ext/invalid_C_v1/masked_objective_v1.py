"""C 组（reference-beads support ablation）的逐 copy 珠子支持适配器。

授权范围与设计（用户明确授权）：

* 训练进程只接受 ``support.npz`` 里的**布尔 mask**（每 copy 每层珠子是否存在于
  reference 且 xyz 有限），不接触 reference 的 xyz 数值、也不接触 phase。
* 缺失珠子不参与 count kernel / bond / bend / repulsion：把缺失珠子的物理坐标置零。
  count 的四项 kernel、bond hinge、bend 二阶差分、repulsion 的 pairwise hinge 在
  delta=0 处取值均为零，因此置零等价于“该珠子不参与”，并保留原基因组索引/间隔。
* count 目标按有效集合精确重算（G 的 shared-offdiag 归一化不变）：
  prefix 总数 = mask 后仍在两 copy 都有效的 pair 计数；observed log-rate 只保留有效
  pair；group-mass KL 用 masked 组总数与 masked z_cis/z_inter；diag 只按有效 pair 计入。
  x 空间梯度用逐 pair 的一阶修正精确给出：Δw * Σ_grad + Δcoef * kernel。
* bond / bend / repulsion 的父类值在置零坐标上已等于 masked 贡献，只需把 normalization
  分母换成有效 edge / triple / (copy,locus-pair) 计数；三者线性缩放，梯度同系数缩放。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from shared_capture_objective import EPSILON, P_PRIOR_STRENGTH, SharedCaptureObjective  # noqa: E402
from visibility_profile_base import VisibilityError  # noqa: E402


class PerCopySupportObjective(SharedCaptureObjective):
    """G 目标 + 逐 copy 珠子支持；缺失珠子坐标置零并退出 count/bond/bend/repulsion。"""

    def __init__(self, data: Any, model_id: str, *, mask: np.ndarray,
                 exposure: np.ndarray, **kwargs: Any):
        super().__init__(data, model_id, **kwargs)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (2, int(data.n_loci)):
            raise ValueError("per-copy support mask must have shape (2, n_loci)")
        if not mask.any():
            raise ValueError("per-copy support mask is empty")
        self.support_mask = mask.copy()
        self._mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        self._n_valid = (int(mask[0].sum()), int(mask[1].sum()))
        self._n_shared_loci = int(np.count_nonzero(mask[0] & mask[1]))
        self._epsilon = float(EPSILON)
        self._r0sq = float(data.r0 * data.r0)

        bond_valid = 0
        bend_valid = 0
        bond_full = 0
        bend_full = 0
        n_chromosomes = len(data.chromosome_names)
        for chrom in range(n_chromosomes):
            slc = data.chromosome_slice(chrom)
            n_bins = int(data.n_bins[chrom])
            bond_full += 2 * max(0, n_bins - 1)
            bend_full += 2 * max(0, n_bins - 2)
            for copy in (0, 1):
                m = self.support_mask[copy, slc]
                bond_valid += int(np.count_nonzero(m[:-1] & m[1:]))
                if n_bins >= 3:
                    bend_valid += int(np.count_nonzero(m[:-2] & m[1:-1] & m[2:]))
        self._bond_terms, self._bond_terms_full = int(bond_valid), int(bond_full)
        self._bend_terms, self._bend_terms_full = int(bend_valid), int(bend_full)
        if self._bond_terms <= 0 or self._bend_terms <= 0:
            raise ValueError("support mask leaves no bond/bend terms")

        pair_i = np.asarray(data.pair_i, dtype=np.int64)
        pair_j = np.asarray(data.pair_j, dtype=np.int64)
        cis = np.asarray(data.cis_pair, dtype=bool)
        counts = np.asarray(data.counts, dtype=np.float64)
        pair_ok = np.zeros((2, pair_i.shape[0]), dtype=bool)
        for copy in (0, 1):
            pair_ok[copy] = mask[copy][pair_i] & mask[copy][pair_j]
        both_ok = pair_ok[0] & pair_ok[1]
        self._pair_ok, self._both_ok = pair_ok, both_ok
        self._cis_np = np.asarray(cis, dtype=bool)
        self._cis = torch.as_tensor(cis, dtype=torch.bool, device=self.device)
        self._group_total = (float(counts[cis & both_ok].sum()), float(counts[~cis & both_ok].sum()))
        self._original_group_total = (float(counts[cis].sum()), float(counts[~cis].sum()))
        self._removed_counts = (self._original_group_total[0] - self._group_total[0],
                                self._original_group_total[1] - self._group_total[1])
        self._removed_pairs = (int(np.count_nonzero(cis & ~both_ok)),
                               int(np.count_nonzero(~cis & ~both_ok)))
        self._observed_removed = int(np.count_nonzero((counts > 0.0) & ~both_ok))
        self._diag_valid_np = np.where(both_ok, counts, 0.0)
        self._diag_valid = torch.as_tensor(self._diag_valid_np, dtype=self.dtype, device=self.device)
        self._diag_nll_valid = self._diag_term(self._diag_valid)
        self._diag_nll_full = self._diag_term(self._diag_counts)

        n_pairs, n_loci = int(data.n_pairs), int(data.n_loci)
        same_copy = int(np.count_nonzero(pair_ok[0])) + int(np.count_nonzero(pair_ok[1]))
        diag_pairs = self._n_valid[0] ** 2 + self._n_valid[1] ** 2
        cross_valid = 0
        for chrom in range(n_chromosomes):
            slc = data.chromosome_slice(chrom)
            a = mask[0, slc]
            b = mask[1, slc]
            cross_valid += int(np.count_nonzero(a[:, None] & b[None, :]))
        self._repulsion_terms_valid = float(2 * (same_copy + diag_pairs + cross_valid))
        self._repulsion_terms_full = float(2 * (2 * n_pairs + 2 * n_loci ** 2))
        self._ratios = {
            "bond": self._bond_terms_full / float(self._bond_terms),
            "bend": self._bend_terms_full / float(self._bend_terms),
            "repulsion": self._repulsion_terms_full / float(self._repulsion_terms_valid),
        }

        exposure = np.asarray(exposure, dtype=np.float64).copy()
        if exposure.shape != (n_loci,) or not np.all(np.isfinite(exposure)):
            raise ValueError("per-copy exposure must be a finite length-n_loci vector")
        positive = exposure[exposure > 0.0]
        if positive.size == 0:
            raise ValueError("per-copy exposure has no positive entry")
        imputed = int(np.count_nonzero(exposure <= 0.0))
        if imputed:
            exposure[exposure <= 0.0] = float(positive.min())
        self._exposure_imputed = imputed
        self._true_e = torch.as_tensor(exposure, dtype=self.dtype, device=self.device)
        self._exposure_np = exposure
        # 父类要求 full-grid e 全正且均值恰为 1（它是 A/B 的 gauge）。C 的 e 在缺失珠子上
        # 取零，全网格均值小于 1，因此父类只放一个“全正 + 均值 1”的占位 e 通过形状校验；
        # 真正的 count 计算在本类 _count_and_gradient 里用 _true_e 完成。
        placeholder = np.asarray(exposure, dtype=np.float64).copy()
        any_valid = mask.any(axis=0)
        placeholder[~any_valid] = float(exposure[any_valid].mean())
        placeholder = placeholder / placeholder.mean()
        self._exposure_placeholder = torch.as_tensor(placeholder, dtype=self.dtype, device=self.device)
        self._pair_delta: torch.Tensor | None = None

    # ------------------------------------------------------------------ helpers
    def _scalar(self, value) -> float:
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        value = float(value)
        if not math.isfinite(value):
            raise VisibilityError("nonfinite per-copy support component")
        return value

    def _diag_term(self, diag_counts: torch.Tensor) -> float:
        positive = diag_counts[diag_counts > 0.0]
        value = (positive - positive * torch.log(positive)).sum()
        if self.data.count_mode in ("raw_integer", "synthetic_integer"):
            value = value + torch.lgamma(positive + 1.0).sum()
        return float(value.detach().cpu().item())

    def reason_terms(self) -> dict[str, float]:
        return dict(self._ratios)

    def support_audit(self) -> dict[str, Any]:
        return {
            "n_loci": int(self.data.n_loci),
            "n_valid_loci_copyA": self._n_valid[0],
            "n_valid_loci_copyB": self._n_valid[1],
            "n_missing_loci_copyA": int(self.data.n_loci - self._n_valid[0]),
            "n_missing_loci_copyB": int(self.data.n_loci - self._n_valid[1]),
            "n_shared_loci": self._n_shared_loci,
            "bond_terms_valid": self._bond_terms, "bond_terms_full": self._bond_terms_full,
            "bend_terms_valid": self._bend_terms, "bend_terms_full": self._bend_terms_full,
            "repulsion_terms_valid": self._repulsion_terms_valid,
            "repulsion_terms_full": self._repulsion_terms_full,
            "normalizer_ratios": dict(self._ratios),
            "group_total_valid_cis": self._group_total[0],
            "group_total_valid_inter": self._group_total[1],
            "removed_counts_cis": self._removed_counts[0],
            "removed_counts_inter": self._removed_counts[1],
            "removed_pairs_cis": self._removed_pairs[0],
            "removed_pairs_inter": self._removed_pairs[1],
            "observed_positive_pairs_removed": self._observed_removed,
            "original_cis_offdiag": self._original_group_total[0],
            "original_inter": self._original_group_total[1],
            "original_raw_records": float(self.data.raw_records),
            "count_denominator_valid_offdiag": float(self._group_total[0] + self._group_total[1]),
            "exposure_imputed_nonpositive_loci": self._exposure_imputed,
            "exposure_mean": float(self._exposure_np.mean()),
            "exposure_min": float(self._exposure_np.min()),
            "exposure_max": float(self._exposure_np.max()),
        }

    # ------------------------------------------------------------------ count objective
    def _kernel_terms(self, xs: torch.Tensor, i: torch.Tensor, j: torch.Tensor):
        aa = xs[0, i] - xs[0, j]
        ab = xs[0, i] - xs[1, j]
        ba = xs[1, i] - xs[0, j]
        bb = xs[1, i] - xs[1, j]
        eps = self._epsilon
        one_minus = 1.0 - eps
        r0sq = self._r0sq

        def kernel(delta):
            base = 1.0 + torch.sum(delta * delta, dim=-1) / r0sq
            value = eps + one_minus * base.pow(-2)
            gradient = (-4.0 * one_minus / r0sq) * delta * base.pow(-3).unsqueeze(-1)
            return value, gradient

        kaa, gaa = kernel(aa)
        kab, gab = kernel(ab)
        kba, gba = kernel(ba)
        kbb, gbb = kernel(bb)
        return (kaa, kab, kba, kbb, kaa + kbb, kab + kba), (gaa, gab, gba, gbb)

    def _pair_delta_tensor(self) -> torch.Tensor:
        if self._pair_delta is None:
            counts = np.asarray(self.data.counts, dtype=np.float64)
            cis = self._cis_np
            both = self._both_ok
            coeff_orig = np.where(cis, 1.0 / float(self.data.raw_cis_offdiag),
                                  1.0 / float(self.data.raw_inter))
            coeff_mask = np.where(cis, 1.0 / self._group_total[0], 1.0 / self._group_total[1])
            k_orig = counts * coeff_orig
            k_mask = counts * coeff_mask
            delta = np.where(both, k_orig - k_mask, P_PRIOR_STRENGTH * math.log(4.0) - k_orig)
            self._pair_delta = torch.as_tensor(delta, dtype=self.dtype, device=self.device)
        return self._pair_delta

    def evaluate(self, theta: np.ndarray, need_gradient: bool = True):
        backup = self._fixed_e
        self._fixed_e = self._exposure_placeholder
        try:
            return super().evaluate(theta, need_gradient=need_gradient)
        finally:
            self._fixed_e = backup

    def _count_and_gradient(self, x: torch.Tensor, p: torch.Tensor, k: torch.Tensor,
                            e_placeholder: torch.Tensor, need_gradient: bool):
        e_in = self._true_e
        xs = x * self._mask.unsqueeze(-1).to(x.dtype)
        normalizer = float(self.data.raw_records)
        p_value = float(p) if not torch.is_tensor(p) else float(p.detach().cpu().item())
        n_cis_m, n_inter_m = self._group_total
        n_off_m = n_cis_m + n_inter_m
        blocks: list[torch.Tensor] = []
        kernels_cache: list[tuple] = []
        start = 0
        while start < int(self.data.n_pairs):
            stop = min(start + self.pair_block, int(self.data.n_pairs))
            kernels, _ = self._kernel_terms(xs, self._pair_i[start:stop], self._pair_j[start:stop])
            kaa, kab, kba, kbb, same, cross = kernels
            local_cis = self._cis[start:stop]
            rate = e_in[self._pair_i[start:stop]] * e_in[self._pair_j[start:stop]] * torch.where(
                local_cis, 0.25 * p_value * same + 0.25 * (1.0 - p_value) * cross,
                0.25 * (same + cross))
            blocks.append(rate)
            kernels_cache.append((kernels, local_cis))
            start = stop
        rate_of_k = torch.cat(blocks)
        z_cis = rate_of_k[self._cis].sum()
        z_inter = rate_of_k[~self._cis].sum()
        if float(z_cis.detach().cpu().item()) <= 0.0 or float(z_inter.detach().cpu().item()) <= 0.0:
            raise VisibilityError("masked positive contact group has nonpositive rate sum")
        z_total = z_cis + z_inter
        observed = self._counts > 0.0
        observed_log_cis = (self._counts[observed & self._cis]
                            * torch.log(rate_of_k[observed & self._cis])).sum()
        observed_log_inter = (self._counts[observed & ~self._cis]
                              * torch.log(rate_of_k[observed & ~self._cis])).sum()
        separate = (n_cis_m * torch.log(z_cis) - observed_log_cis
                    + n_inter_m * torch.log(z_inter) - observed_log_inter)
        group_mass = torch.zeros((), dtype=self.dtype, device=self.device)
        if n_off_m > 0.0:
            for n_group, z_group in ((n_cis_m, z_cis), (n_inter_m, z_inter)):
                if n_group > 0.0:
                    group_mass = group_mass + n_group * (
                        math.log(n_group / n_off_m) - torch.log(z_group / z_total))
        count_raw = separate + group_mass + self._diag_nll_valid
        components = {
            "count_mode": str(self.data.count_mode),
            "normalization_model": self.model_id,
            "count_factorial_status": (
                "integer_count_constants_recorded"
                if self.data.count_mode in ("raw_integer", "synthetic_integer")
                else "not_defined_for_fractional_expected_cross_entropy"),
            "sum_rate_cis_offdiag": self._scalar(z_cis),
            "sum_rate_inter": self._scalar(z_inter),
            "sum_rate_offdiag": self._scalar(z_total),
            "observed_log_rate_cis_offdiag": self._scalar(observed_log_cis),
            "observed_log_rate_inter": self._scalar(observed_log_inter),
            "observed_log_rate_total": self._scalar(observed_log_cis + observed_log_inter),
            "separate_conditional_nll_raw": self._scalar(separate),
            "group_mass_kl_raw": self._scalar(group_mass),
            "group_mass_kl_normalized": self._scalar(group_mass / normalizer),
            "observed_cis_mass": self._scalar(n_cis_m / n_off_m),
            "observed_inter_mass": self._scalar(n_inter_m / n_off_m),
            "predicted_cis_mass": self._scalar(z_cis / z_total),
            "predicted_inter_mass": self._scalar(z_inter / z_total),
            "profiled_intensity_cis": self._scalar(n_cis_m / z_cis),
            "profiled_intensity_inter": self._scalar(n_inter_m / z_inter),
            "shared_profiled_intensity": self._scalar(n_off_m / z_total),
            "conditional_nll_raw": self._scalar(separate),
            "conditional_factorial_constant_omitted": (
                float(self.data.conditional_factorial_constant)
                if math.isfinite(float(self.data.conditional_factorial_constant)) else None),
            "diag_factorial_sum_included": (
                float(self.data.diag_factorial_sum)
                if math.isfinite(float(self.data.diag_factorial_sum)) else None),
            "diag_profiled_nll_raw": self._scalar(self._diag_nll_valid),
            "count_nll_raw": self._scalar(count_raw),
            "count_nll_normalized": self._scalar(count_raw / normalizer),
            "per_copy_support": self.support_audit(),
        }
        if not need_gradient:
            return components, None, None

        gradient_x = torch.zeros_like(x)
        gradient_p = torch.zeros((), dtype=self.dtype, device=self.device)
        delta = self._pair_delta_tensor()
        start = 0
        block_index = 0
        while start < int(self.data.n_pairs):
            stop = min(start + self.pair_block, int(self.data.n_pairs))
            i = self._pair_i[start:stop]
            j = self._pair_j[start:stop]
            (kaa, kab, kba, kbb, same, cross), gradients = self._kernel_terms(xs, i, j)
            gaa, gab, gba, gbb = gradients
            local_cis = kernels_cache[block_index][1]
            local_e = e_in[i] * e_in[j]
            mix = torch.where(local_cis, 0.5 * p_value, 0.25)
            rate = rate_of_k[start:stop]
            local_delta = delta[start:stop]
            local_observed = observed[start:stop]
            safe = (rate > 0.0) & (local_delta != 0.0)
            observed_gradient = torch.zeros_like(rate)
            if bool(torch.any(local_observed & safe)):
                sel = local_observed & safe
                observed_gradient[sel] = (self._counts[start:stop][sel] * mix[sel] / rate[sel])
            weight = (local_delta + mix * local_e - observed_gradient) * local_e
            raw_gradient = weight * mix
            same_coefficient = torch.where(local_cis, raw_gradient, torch.zeros_like(raw_gradient))
            cross_coefficient = torch.where(local_cis, -raw_gradient, torch.zeros_like(raw_gradient))
            for copy_i, copy_j, kernel_gradient, coefficient in (
                    (0, 0, gaa, same_coefficient), (0, 1, gab, cross_coefficient),
                    (1, 0, gba, cross_coefficient), (1, 1, gbb, same_coefficient)):
                term_grad = coefficient.unsqueeze(-1) * kernel_gradient
                gradient_x[copy_i].index_add_(0, i, term_grad)
                gradient_x[copy_j].index_add_(0, j, -term_grad)
            if bool(torch.any(local_cis & safe)):
                sel = local_cis & safe
                dp = (0.5 * local_e[sel] * (same[sel] - cross[sel]) / rate[sel]
                      * self._counts[start:stop][sel])
                gradient_p = gradient_p + (local_delta[sel] + mix[sel] * local_e[sel]).mul(dp).sum()
            start = stop
            block_index += 1
        return components, gradient_x / normalizer, gradient_p / normalizer

