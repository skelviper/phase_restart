"""统一 C0/V1 技术可见性 Torch float64 后端。

本模块只处理 SNP-free 聚合 counts 和物理几何。V0 固定生产 e、V1
profile-e、以及 synthetic known-e 经过同一份 pair-kernel、几何梯度和物理
先验代码；V1 的 eta 内层在固定 X,p 下完整求解，不把一次 degree 校正当作
profile likelihood。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

for _thread_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_key] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE_035 = Path(__file__).resolve().parent / "frozen_035"
FROZEN_PR = Path(__file__).resolve().parent / "frozen_pr"
for _path in (ROOT, SOURCE_035, FROZEN_PR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gpu_variant_backend import GPUVariantObjective  # type: ignore  # noqa: E402
from pr import contact_model  # noqa: E402

EPSILON = 1e-6
P_FLOOR = 1e-4
P_PRIOR_STRENGTH = 1e-4
PROFILE_RESIDUAL_TOL = 1e-10
DEFAULT_INNER_CAP = 80
DEFAULT_CG_CAP = 80
DEFAULT_PAIR_BLOCK = 262_144


class VisibilityError(RuntimeError):
    """技术可见性支持、内层收敛或几何数值契约失败。"""


class ProfileConvergenceError(VisibilityError):
    """固定几何下的 eta profile 未在有限内层预算内收敛。"""


@dataclass(frozen=True)
class SupportAudit:
    """由冻结 offdiag counts 决定的 active/zero support 和充分内点条件。"""

    degree: np.ndarray
    active_mask: np.ndarray
    zero_mask: np.ndarray
    per_chromosome: tuple[dict[str, Any], ...]
    total_offdiag_count: float
    all_conditions_pass: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "active_count": int(self.active_mask.sum()),
            "zero_count": int(self.zero_mask.sum()),
            "total_offdiag_count": float(self.total_offdiag_count),
            "all_conditions_pass": bool(self.all_conditions_pass),
            "per_chromosome": [dict(row) for row in self.per_chromosome],
            "condition_definition": {
                "minimum_active_per_chromosome": 3,
                "strict_degree_balance": "2*max_bin_degree_chr < sum_degree_chr",
                "strict_global_room": "sum_degree_chr < total_offdiag_count",
                "full_k_positive": True,
                "zero_degree_extended_mle": "e_i=0; no floor or pseudocount",
            },
        }


def _support_audit(data: contact_model.AggregatedContacts) -> SupportAudit:
    """从 offdiag aggregate counts 计算 support；same-bin 不进入 degree。"""
    degree = np.zeros(data.n_loci, dtype=np.float64)
    counts = np.asarray(data.counts, dtype=np.float64)
    np.add.at(degree, np.asarray(data.pair_i, dtype=np.int64), counts)
    np.add.at(degree, np.asarray(data.pair_j, dtype=np.int64), counts)
    if not np.all(np.isfinite(degree)) or np.any(degree < 0.0):
        raise VisibilityError("offdiag degree is nonfinite or negative")
    active = degree > 0.0
    rows: list[dict[str, Any]] = []
    total = float(data.raw_cis_offdiag + data.raw_inter)
    for chromosome_index, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(chromosome_index)
        values = degree[slc]
        positive = values[values > 0.0]
        sum_degree = float(values.sum())
        max_degree = float(positive.max(initial=0.0))
        row = {
            "chromosome_index": int(chromosome_index),
            "chromosome": str(name),
            "active_count": int(len(positive)),
            "sum_degree": sum_degree,
            "max_bin_degree": max_degree,
            "minimum_active_pass": bool(len(positive) >= 3),
            "strict_degree_balance_pass": bool(2.0 * max_degree < sum_degree),
            "strict_global_room_pass": bool(sum_degree < total),
        }
        rows.append(row)
    passed = all(
        row["minimum_active_pass"]
        and row["strict_degree_balance_pass"]
        and row["strict_global_room_pass"]
        for row in rows
    )
    return SupportAudit(
        degree=degree,
        active_mask=active,
        zero_mask=~active,
        per_chromosome=tuple(rows),
        total_offdiag_count=total,
        all_conditions_pass=bool(passed),
    )


def _finite_scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite value cannot enter JSON")
        return float(value)
    return value


class VisibilityGPUObjective:
    """统一几何 objective，支持固定 e、profile-e 和 known-e。

    ``mode`` 只能是 ``V0-fixed-production-e``、``VZ-zero-degree-fixed-e``、
    ``V1-profile-e`` 或 ``V0-known-generating-e``。V1 inner profile 使用 active
    degree support；zero-degree loci 保留在完整 grid 和 pair kernel 中，但其 e 恰为零。
    """

    MODES = frozenset(("V0-fixed-production-e", "VZ-zero-degree-fixed-e", "V1-profile-e", "V0-known-generating-e"))

    def __init__(self, data: contact_model.AggregatedContacts, mode: str,
                 device: str = "cuda", pair_block: int = DEFAULT_PAIR_BLOCK,
                 inner_cap: int = DEFAULT_INNER_CAP, cg_cap: int = DEFAULT_CG_CAP,
                 profile_warm_start: bool = True, known_e: np.ndarray | None = None):
        data.assert_consistent()
        mode = str(mode)
        if mode not in self.MODES:
            raise ValueError("unknown visibility mode %r" % mode)
        if mode == "V0-known-generating-e" and known_e is None:
            raise ValueError("known-e mode requires the authorized generating exposure vector")
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise VisibilityError("CUDA requested but no CUDA device is available")
        if requested.type not in ("cpu", "cuda"):
            raise VisibilityError("device must be cpu or cuda")
        if int(pair_block) <= 0 or int(inner_cap) <= 0 or int(cg_cap) <= 0:
            raise ValueError("pair_block, inner_cap, and cg_cap must be positive")
        self.data = data
        self.mode = mode
        self.device = requested
        self.dtype = torch.float64
        self.pair_block = int(pair_block)
        self.inner_cap = int(inner_cap)
        self.cg_cap = int(cg_cap)
        self.profile_warm_start = bool(profile_warm_start)
        self.support = _support_audit(data)
        if mode == "V1-profile-e" and not self.support.all_conditions_pass:
            raise VisibilityError("degree support fails the preregistered sufficient interiority conditions")

        # 035 的 C0 Torch layers 仅作为同一 geometry backend 的物理层；不调用
        # 028 fused count 层。expected-count mock 没有整数 Poisson budget，物理层
        # 使用同一 full-grid metadata 的零 counts 影子对象，不触碰其科学 counts。
        physics_data = data
        if data.count_mode == "synthetic_expected":
            physics_data = replace(
                data,
                counts=np.zeros(data.n_pairs, dtype=np.int64),
                diag_counts=np.zeros(data.n_loci, dtype=np.int64),
                endpoint_counts=np.zeros(data.n_loci, dtype=np.int64),
                raw_records=0, raw_same_bin=0, raw_cis_offdiag=0, raw_inter=0,
                conditional_factorial_constant=0.0, diag_factorial_sum=0.0,
                count_mode="raw_integer", exposure_mode="synthetic_expected_physics",
            )
        self._physics = GPUVariantObjective(
            physics_data, model_id="C0", device=str(requested), tile_rows=32,
            dtype=torch.float64, use_fused=False, diagnostics=False,
        )
        self.n_parameters = 6 * data.n_loci + 1
        self._pair_i = torch.as_tensor(data.pair_i, dtype=torch.int64, device=requested)
        self._pair_j = torch.as_tensor(data.pair_j, dtype=torch.int64, device=requested)
        self._cis = torch.as_tensor(data.cis_pair, dtype=torch.bool, device=requested)
        self._counts = torch.as_tensor(np.asarray(data.counts, dtype=np.float64), dtype=self.dtype, device=requested)
        self._degree = torch.as_tensor(self.support.degree, dtype=self.dtype, device=requested)
        self._active = torch.as_tensor(self.support.active_mask, dtype=torch.bool, device=requested)
        self._pair_active = self._active[self._pair_i] & self._active[self._pair_j]
        self._group_total_cis = torch.as_tensor(float(data.raw_cis_offdiag), dtype=self.dtype, device=requested)
        self._group_total_inter = torch.as_tensor(float(data.raw_inter), dtype=self.dtype, device=requested)
        self._diag_counts = torch.as_tensor(np.asarray(data.diag_counts, dtype=np.float64), dtype=self.dtype, device=requested)
        self._fixed_e = None
        if mode == "V0-fixed-production-e":
            values = np.asarray(data.exposure, dtype=np.float64)
            self._fixed_e = self._validate_e(values, "production e", allow_zero=False)
        elif mode == "VZ-zero-degree-fixed-e":
            values = np.asarray(data.exposure, dtype=np.float64).copy()
            values[self.support.zero_mask] = 0.0
            mean = float(values.mean())
            if not math.isfinite(mean) or mean <= 0.0:
                raise VisibilityError("VZ zero-degree support leaves no positive exposure")
            values /= mean
            self._fixed_e = self._validate_e(values, "VZ zero-degree fixed e", allow_zero=True)
        elif mode == "V0-known-generating-e":
            self._fixed_e = self._validate_e(np.asarray(known_e, dtype=np.float64), "known-generating e", allow_zero=False)
        self._warm_eta: torch.Tensor | None = None
        self._last_theta: np.ndarray | None = None
        self._last_value: float | None = None
        self._last_gradient: np.ndarray | None = None
        self._last_components: dict[str, Any] | None = None
        self._last_visibility: dict[str, Any] | None = None
        self._counters = {
            "outer_evaluations": 0,
            "geometry_kernel_builds": 0,
            "profile_calls": 0,
            "profile_inner_iterations": 0,
            "profile_cg_iterations": 0,
            "profile_matvecs": 0,
            "profile_backtracks": 0,
            "profile_seconds": 0.0,
            "geometry_seconds": 0.0,
            "count_gradient_seconds": 0.0,
            "physics_seconds": 0.0,
            "outer_wall_seconds": 0.0,
        }
        self._sync()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def synchronize(self) -> None:
        """等待当前 GPU stream，供预检的分段计时使用。"""
        self._sync()

    def _validate_e(self, values: np.ndarray, label: str, allow_zero: bool) -> torch.Tensor:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        valid = np.all(np.isfinite(values)) and (np.all(values >= 0.0) if allow_zero else np.all(values > 0.0))
        if values.shape != (self.data.n_loci,) or not valid:
            raise ValueError("%s must be a finite length-n_loci vector with the declared support" % label)
        if not np.isclose(float(values.mean()), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("%s must have full-grid arithmetic mean one" % label)
        if allow_zero and not np.any(values > 0.0):
            raise ValueError("%s must retain a positive support" % label)
        return torch.as_tensor(values, dtype=self.dtype, device=self.device)

    @property
    def n_loci(self) -> int:
        return int(self.data.n_loci)

    def _kernel_block(self, x: torch.Tensor, i: torch.Tensor, j: torch.Tensor,
                      p: torch.Tensor, with_gradient: bool):
        aa = x[0, i] - x[0, j]
        ab = x[0, i] - x[1, j]
        ba = x[1, i] - x[0, j]
        bb = x[1, i] - x[1, j]
        r0sq = float(self.data.r0 * self.data.r0)

        def kernel(delta: torch.Tensor):
            base = 1.0 + torch.sum(delta * delta, dim=-1) / r0sq
            value = EPSILON + (1.0 - EPSILON) * base.pow(-2)
            if not with_gradient:
                return value, None
            gradient = (-4.0 * (1.0 - EPSILON) / r0sq) * delta * base.pow(-3).unsqueeze(-1)
            return value, gradient

        kaa, gaa = kernel(aa)
        kab, gab = kernel(ab)
        kba, gba = kernel(ba)
        kbb, gbb = kernel(bb)
        same = kaa + kbb
        cross = kab + kba
        return (kaa, kab, kba, kbb, same, cross), (gaa, gab, gba, gbb)

    def _build_k(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """在完整 upper-triangle 上构造不含 e_i e_j 的 mixture k。"""
        values = torch.empty(self.data.n_pairs, dtype=self.dtype, device=self.device)
        for start in range(0, self.data.n_pairs, self.pair_block):
            stop = min(start + self.pair_block, self.data.n_pairs)
            i = self._pair_i[start:stop]
            j = self._pair_j[start:stop]
            kernels, _ = self._kernel_block(x, i, j, p, with_gradient=False)
            kaa, kab, kba, kbb, same, cross = kernels
            values[start:stop] = torch.where(self._cis[start:stop],
                                              0.5 * (p * same + (1.0 - p) * cross),
                                              0.25 * (same + cross))
        self._counters["geometry_kernel_builds"] += 1
        return values

    def _profile_stats(self, k: torch.Tensor, eta: torch.Tensor):
        """给定 eta 计算两组 Z、预测 degree、梯度及 F。"""
        active_pair = self._pair_active
        factor = torch.exp(eta[self._pair_i] + eta[self._pair_j])
        weight = torch.where(active_pair, factor * k, torch.zeros_like(k))
        z_cis = weight[self._cis].sum()
        z_inter = weight[~self._cis].sum()
        if (self._group_total_cis > 0.0 and float(z_cis.detach().cpu().item()) <= 0.0) or (
                self._group_total_inter > 0.0 and float(z_inter.detach().cpu().item()) <= 0.0):
            raise VisibilityError("profile group has nonpositive Z")
        probability = torch.zeros_like(weight)
        if self._group_total_cis > 0.0:
            probability[self._cis] = weight[self._cis] * (self._group_total_cis / z_cis)
        if self._group_total_inter > 0.0:
            probability[~self._cis] = weight[~self._cis] * (self._group_total_inter / z_inter)
        predicted = torch.zeros(self.data.n_loci, dtype=self.dtype, device=self.device)
        predicted.index_add_(0, self._pair_i, probability)
        predicted.index_add_(0, self._pair_j, probability)
        log_k_term = (self._counts * torch.log(k)).sum()
        objective = self._group_total_cis * torch.log(z_cis) + self._group_total_inter * torch.log(z_inter)
        objective = objective - (self._degree * eta).sum() - log_k_term
        residual = predicted - self._degree
        denominator = torch.clamp(self._degree, min=1.0)
        relative_residual = torch.max(torch.abs(residual) / denominator)
        mu_cis = torch.zeros_like(predicted)
        mu_inter = torch.zeros_like(predicted)
        mu_cis.index_add_(0, self._pair_i[self._cis], probability[self._cis])
        mu_cis.index_add_(0, self._pair_j[self._cis], probability[self._cis])
        mu_inter.index_add_(0, self._pair_i[~self._cis], probability[~self._cis])
        mu_inter.index_add_(0, self._pair_j[~self._cis], probability[~self._cis])
        return {
            "weight": weight,
            "probability": probability,
            "z_cis": z_cis,
            "z_inter": z_inter,
            "predicted_degree": predicted,
            "residual": residual,
            "relative_residual": relative_residual,
            "mu_cis": mu_cis,
            "mu_inter": mu_inter,
            "objective": objective,
        }

    def _project_active(self, values: torch.Tensor) -> torch.Tensor:
        result = torch.where(self._active, values, torch.zeros_like(values))
        active_count = max(1, int(self.support.active_mask.sum()))
        return torch.where(self._active, result - result[self._active].sum() / float(active_count), result)

    def _hessian_vector(self, probability: torch.Tensor, mu_cis: torch.Tensor,
                        mu_inter: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """精确 profile Hessian-vector product，不对 inner 迭代求导。"""
        pair_dot = vector[self._pair_i] + vector[self._pair_j]
        result = torch.zeros(self.data.n_loci, dtype=self.dtype, device=self.device)
        term = probability * pair_dot
        result.index_add_(0, self._pair_i, term)
        result.index_add_(0, self._pair_j, term)
        if self._group_total_cis > 0.0:
            scalar = (mu_cis * vector).sum() / self._group_total_cis
            result = result - mu_cis * scalar
        if self._group_total_inter > 0.0:
            scalar = (mu_inter * vector).sum() / self._group_total_inter
            result = result - mu_inter * scalar
        return self._project_active(result)

    def _newton_direction(self, stats: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        """在 active gauge 子空间用无 ridge 的预条件 CG 解 Newton 方程。"""
        gradient = self._project_active(stats["residual"])
        probability = stats["probability"]
        mu_cis = stats["mu_cis"]
        mu_inter = stats["mu_inter"]
        diagonal = stats["predicted_degree"].clone()
        if self._group_total_cis > 0.0:
            diagonal = diagonal - mu_cis * mu_cis / self._group_total_cis
        if self._group_total_inter > 0.0:
            diagonal = diagonal - mu_inter * mu_inter / self._group_total_inter
        diagonal = torch.where(self._active, torch.clamp(diagonal, min=torch.finfo(self.dtype).tiny), torch.ones_like(diagonal))
        rhs = self._project_active(-gradient)
        direction = torch.zeros_like(rhs)
        residual = rhs.clone()
        preconditioned = residual / diagonal
        search = preconditioned.clone()
        rr = torch.dot(residual, preconditioned)
        rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(self.dtype).tiny)
        cg_used = 0
        for cg_index in range(self.cg_cap):
            hv = self._hessian_vector(probability, mu_cis, mu_inter, search)
            self._counters["profile_matvecs"] += 1
            denominator = torch.dot(search, hv)
            if not torch.isfinite(denominator) or float(denominator.detach().cpu().item()) <= 0.0:
                break
            alpha = rr / denominator
            direction = self._project_active(direction + alpha * search)
            residual = self._project_active(residual - alpha * hv)
            preconditioned = residual / diagonal
            new_rr = torch.dot(residual, preconditioned)
            cg_used = cg_index + 1
            if float(torch.linalg.vector_norm(residual).detach().cpu().item()) <= 1e-12 * float(rhs_norm.detach().cpu().item()):
                break
            beta = new_rr / torch.clamp(rr, min=torch.finfo(self.dtype).tiny)
            search = self._project_active(preconditioned + beta * search)
            rr = new_rr
        self._counters["profile_cg_iterations"] += cg_used
        if not torch.all(torch.isfinite(direction)):
            raise VisibilityError("profile Newton direction is nonfinite")
        # 方向应为下降方向；若 Hessian 数值误差破坏它，退回 majorization 方向。
        directional = torch.dot(gradient, direction)
        if float(directional.detach().cpu().item()) >= 0.0 or cg_used == 0:
            ratio = torch.where(self._active,
                                torch.log(torch.clamp(self._degree, min=torch.finfo(self.dtype).tiny)
                                          / torch.clamp(stats["predicted_degree"], min=torch.finfo(self.dtype).tiny)),
                                torch.zeros_like(self._degree))
            direction = self._project_active(0.5 * ratio)
        return direction, cg_used

    def _profile(self, k: torch.Tensor, warm_eta: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """完整 profile inner；未达 residual 门时抛出硬错误。"""
        started = time.perf_counter()
        matvec_start = int(self._counters["profile_matvecs"])
        cg_start = int(self._counters["profile_cg_iterations"])
        self._counters["profile_calls"] += 1
        if warm_eta is not None:
            eta = torch.where(self._active, warm_eta, torch.zeros_like(warm_eta)).clone()
            eta = self._project_active(eta)
            start_kind = "warm"
        else:
            eta = torch.zeros(self.data.n_loci, dtype=self.dtype, device=self.device)
            start_kind = "cold"
        stats = self._profile_stats(k, eta)
        converged = False
        backtracks = 0
        iterations = 0
        for iteration in range(self.inner_cap + 1):
            iterations = iteration
            residual_value = _finite_scalar(stats["relative_residual"])
            if residual_value <= PROFILE_RESIDUAL_TOL:
                converged = True
                break
            if iteration >= self.inner_cap:
                break
            direction, _cg = self._newton_direction(stats)
            current_value = _finite_scalar(stats["objective"])
            gradient = self._project_active(stats["residual"])
            directional = _finite_scalar(torch.dot(gradient, direction))
            if not math.isfinite(directional) or directional >= 0.0:
                raise ProfileConvergenceError("profile direction is not strictly descending")
            step = 1.0
            accepted = False
            for _ in range(30):
                trial_eta = self._project_active(eta + step * direction)
                trial_stats = self._profile_stats(k, trial_eta)
                trial_value = _finite_scalar(trial_stats["objective"])
                if math.isfinite(trial_value) and trial_value <= current_value + 1e-14 * max(1.0, abs(current_value)):
                    eta, stats = trial_eta, trial_stats
                    accepted = True
                    break
                step *= 0.5
                backtracks += 1
            if not accepted:
                raise ProfileConvergenceError("profile line search failed to decrease F")
        self._sync()
        elapsed = time.perf_counter() - started
        self._counters["profile_inner_iterations"] += iterations
        self._counters["profile_backtracks"] += backtracks
        self._counters["profile_seconds"] += elapsed
        residual_value = _finite_scalar(stats["relative_residual"])
        if not converged:
            raise ProfileConvergenceError(
                "profile residual %.17g > %.17g after inner cap %d (%s start)"
                % (residual_value, PROFILE_RESIDUAL_TOL, self.inner_cap, start_kind))
        visibility = torch.where(self._active, torch.exp(eta), torch.zeros_like(eta))
        full_mean = visibility.mean()
        if not torch.isfinite(full_mean) or float(full_mean.detach().cpu().item()) <= 0.0:
            raise ProfileConvergenceError("profile e mean is nonpositive")
        normalized_visibility = visibility / full_mean
        self._counters["profile_seconds"] += 0.0
        return eta, {
            "start": start_kind,
            "inner_iterations": int(iterations),
            "inner_cap": int(self.inner_cap),
            "cg_cap": int(self.cg_cap),
            "matvecs": int(self._counters["profile_matvecs"] - matvec_start),
            "cg_iterations": int(self._counters["profile_cg_iterations"] - cg_start),
            "backtracks": int(backtracks),
            "relative_degree_residual": residual_value,
            "degree_residual_inf": _finite_scalar(torch.abs(stats["residual"])[self._active].max()),
            "converged": True,
            "gauge": "sum_active_eta_zero",
            "active_count": int(self.support.active_mask.sum()),
            "zero_count": int(self.support.zero_mask.sum()),
            "F": _finite_scalar(stats["objective"]),
            "z_cis": _finite_scalar(stats["z_cis"]),
            "z_inter": _finite_scalar(stats["z_inter"]),
            "e_fullgrid_mean_before_normalization": _finite_scalar(full_mean),
            "e_fullgrid_mean_after_normalization": _finite_scalar(normalized_visibility.mean()),
            "eta": eta.detach().cpu().numpy().astype(np.float64, copy=True),
            "e": normalized_visibility.detach().cpu().numpy().astype(np.float64, copy=True),
            "predicted_degree": stats["predicted_degree"].detach().cpu().numpy().astype(np.float64, copy=True),
            "degree": self.support.degree.copy(),
        }

    def _fixed_visibility(self) -> tuple[torch.Tensor, dict[str, Any]]:
        assert self._fixed_e is not None
        values = self._fixed_e
        values_np = values.detach().cpu().numpy().astype(np.float64, copy=True)
        eta_np = np.full_like(values_np, -np.inf)
        np.log(values_np, out=eta_np, where=values_np > 0.0)
        return values, {
            "start": "fixed",
            "inner_iterations": 0,
            "inner_cap": int(self.inner_cap),
            "cg_cap": int(self.cg_cap),
            "matvecs": 0,
            "backtracks": 0,
            "relative_degree_residual": None,
            "degree_residual_inf": None,
            "converged": True,
            "gauge": "fixed_e",
            "active_count": int(self.support.active_mask.sum()),
            "zero_count": int(self.support.zero_mask.sum()),
            "F": None,
            "z_cis": None,
            "z_inter": None,
            "e_fullgrid_mean_before_normalization": 1.0,
            "e_fullgrid_mean_after_normalization": 1.0,
            "eta": eta_np,
            "e": values.detach().cpu().numpy().copy(),
            "predicted_degree": None,
            "degree": self.support.degree.copy(),
        }

    def _count_and_gradient(self, x: torch.Tensor, p: torch.Tensor, k: torch.Tensor,
                            e: torch.Tensor, need_gradient: bool):
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
            observed_log_cis = (self._counts[observed & self._cis] * torch.log(rate[observed & self._cis])).sum()
        if bool(torch.any(observed & ~self._cis)):
            observed_log_inter = (self._counts[observed & ~self._cis] * torch.log(rate[observed & ~self._cis])).sum()
        conditional = (self._group_total_cis * torch.log(z_cis) - observed_log_cis
                       + self._group_total_inter * torch.log(z_inter) - observed_log_inter)
        positive = self._diag_counts[self._diag_counts > 0.0]
        diag = (positive - positive * torch.log(positive)).sum()
        if self.data.count_mode in ("raw_integer", "synthetic_integer"):
            diag = diag + torch.lgamma(positive + 1.0).sum()
        count_raw = conditional + diag
        components = {
            "count_mode": str(self.data.count_mode),
            "count_factorial_status": (
                "integer_count_constants_recorded"
                if self.data.count_mode in ("raw_integer", "synthetic_integer")
                else "not_defined_for_fractional_expected_cross_entropy"),
            "sum_rate_cis_offdiag": _finite_scalar(z_cis),
            "sum_rate_inter": _finite_scalar(z_inter),
            "observed_log_rate_cis_offdiag": _finite_scalar(observed_log_cis),
            "observed_log_rate_inter": _finite_scalar(observed_log_inter),
            "conditional_nll_raw": _finite_scalar(conditional),
            "conditional_factorial_constant_omitted": float(self.data.conditional_factorial_constant),
            "diag_factorial_sum_included": float(self.data.diag_factorial_sum),
            "diag_profiled_nll_raw": _finite_scalar(diag),
            "count_nll_raw": _finite_scalar(count_raw),
            "count_nll_normalized": _finite_scalar(count_raw / float(self.data.raw_records)),
        }
        if not need_gradient:
            return components, None, None

        gradient_x = torch.zeros_like(x)
        gradient_p = torch.zeros((), dtype=self.dtype, device=self.device)
        weights = torch.zeros_like(rate)
        pair_positive = eprod > 0.0
        if self._group_total_cis > 0.0:
            mask = pair_positive & self._cis
            weights[mask] = self._group_total_cis / z_cis - self._counts[mask] / rate[mask]
        if self._group_total_inter > 0.0:
            mask = pair_positive & ~self._cis
            weights[mask] = self._group_total_inter / z_inter - self._counts[mask] / rate[mask]
        for start in range(0, self.data.n_pairs, self.pair_block):
            stop = min(start + self.pair_block, self.data.n_pairs)
            i = self._pair_i[start:stop]
            j = self._pair_j[start:stop]
            kernels, gradients = self._kernel_block(x, i, j, p, with_gradient=True)
            kaa, kab, kba, kbb, same, cross = kernels
            gaa, gab, gba, gbb = gradients
            local_e = eprod[start:stop]
            local_weights = weights[start:stop]
            local_cis = self._cis[start:stop]
            same_coefficient = local_e * torch.where(local_cis, 0.5 * p,
                                                     torch.as_tensor(0.25, dtype=self.dtype, device=self.device)) * local_weights
            cross_coefficient = local_e * torch.where(local_cis, 0.5 * (1.0 - p),
                                                      torch.as_tensor(0.25, dtype=self.dtype, device=self.device)) * local_weights
            for copy_i, copy_j, kernel_gradient, coefficient in (
                    (0, 0, gaa, same_coefficient), (0, 1, gab, cross_coefficient),
                    (1, 0, gba, cross_coefficient), (1, 1, gbb, same_coefficient)):
                term = coefficient.unsqueeze(-1) * kernel_gradient
                gradient_x[copy_i].index_add_(0, i, term)
                gradient_x[copy_j].index_add_(0, j, -term)
            derivative_p = 0.5 * local_e * (kaa + kbb - kab - kba)
            cis_local = local_cis
            if bool(torch.any(cis_local)):
                gradient_p = gradient_p + (local_weights[cis_local] * derivative_p[cis_local]).sum()
        normalizer = float(self.data.raw_records)
        return components, gradient_x / normalizer, gradient_p / normalizer

    def _physics_layers(self, x: torch.Tensor):
        started = time.perf_counter()
        backend = self._physics._backend
        bond, bond_gradient = backend._bond_layer(x)
        repulsion, repulsion_gradient = backend._repulsion_layer(x)
        bend, bend_gradient = backend._bend_layer(x)
        self._sync()
        self._counters["physics_seconds"] += time.perf_counter() - started
        return bond, bond_gradient, repulsion, repulsion_gradient, bend, bend_gradient

    def evaluate(self, theta: np.ndarray, need_gradient: bool = True):
        """返回 V0/V1 总目标、raw-y/q 梯度、components 和 profile 审计。"""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,) or not np.all(np.isfinite(theta)):
            raise VisibilityError("theta has wrong shape or nonfinite values")
        started = time.perf_counter()
        geometry_started = started
        theta_t = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        raw = theta_t[:-1].reshape(2, self.data.n_loci, 3)
        x = self._physics._map_raw(raw)
        self._physics.validate_physical_coordinates_tensor(x)
        p, dpdq = contact_model.p_from_q(float(theta[-1]))
        p_t = torch.as_tensor(p, dtype=self.dtype, device=self.device)
        k = self._build_k(x, p_t)
        self._sync()
        self._counters["geometry_seconds"] += time.perf_counter() - geometry_started
        if self.mode == "V1-profile-e":
            warm = self._warm_eta if self.profile_warm_start else None
            eta, visibility = self._profile(k, warm_eta=warm)
            self._warm_eta = eta.detach().clone()
            e = torch.as_tensor(visibility["e"], dtype=self.dtype, device=self.device)
        else:
            _fixed_values, visibility = self._fixed_visibility()
            e = self._fixed_e
            assert e is not None
        count_started = time.perf_counter()
        count_components, count_gradient_x, count_gradient_p = self._count_and_gradient(
            x, p_t, k, e, need_gradient)
        self._sync()
        self._counters["count_gradient_seconds"] += time.perf_counter() - count_started
        bond, bond_gradient, repulsion, repulsion_gradient, bend, bend_gradient = self._physics_layers(x)
        p_prior = -P_PRIOR_STRENGTH * torch.log(p_t * (1.0 - p_t))
        p_prior_derivative = P_PRIOR_STRENGTH * (1.0 / (1.0 - p_t) - 1.0 / p_t)
        components = dict(count_components)
        components.update({
            "visibility_mode": self.mode,
            "p": float(p),
            "p_prior": _finite_scalar(p_prior),
            "bond": _finite_scalar(bond),
            "repulsion": _finite_scalar(repulsion),
            "bend": _finite_scalar(bend),
            "repulsion_threshold_l0": 0.7,
            "profile": {key: value for key, value in visibility.items() if key not in ("eta", "e", "predicted_degree", "degree")},
        })
        total = (torch.as_tensor(components["count_nll_normalized"], dtype=self.dtype, device=self.device)
                 + p_prior + bond + repulsion + 0.01 * bend)
        components["total"] = _finite_scalar(total)
        self._counters["outer_evaluations"] += 1
        if not need_gradient:
            self._last_visibility = visibility
            self._sync()
            self._counters["outer_wall_seconds"] += time.perf_counter() - started
            return components["total"], None, components
        combined_x = count_gradient_x + bond_gradient + repulsion_gradient + 0.01 * bend_gradient
        gradient_raw = self._physics._pullback(raw, combined_x)
        gradient_q = torch.as_tensor(dpdq, dtype=self.dtype, device=self.device) * (count_gradient_p + p_prior_derivative)
        gradient = torch.cat((gradient_raw.reshape(-1), gradient_q.reshape(1)))
        gradient_np = gradient.detach().cpu().numpy().astype(np.float64, copy=False)
        if not np.all(np.isfinite(gradient_np)) or not math.isfinite(components["total"]):
            raise FloatingPointError("nonfinite visibility objective or gradient")
        self._last_visibility = visibility
        self._sync()
        self._counters["outer_wall_seconds"] += time.perf_counter() - started
        return components["total"], gradient_np.copy(), components

    def count_value_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """只返回 profile count NLL 及其 envelope raw-y/q 梯度，排除所有 priors。"""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,) or not np.all(np.isfinite(theta)):
            raise VisibilityError("theta has wrong shape or nonfinite values")
        started = time.perf_counter()
        geometry_started = started
        theta_t = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        raw = theta_t[:-1].reshape(2, self.data.n_loci, 3)
        x = self._physics._map_raw(raw)
        self._physics.validate_physical_coordinates_tensor(x)
        p, dpdq = contact_model.p_from_q(float(theta[-1]))
        p_t = torch.as_tensor(p, dtype=self.dtype, device=self.device)
        k = self._build_k(x, p_t)
        self._sync()
        self._counters["geometry_seconds"] += time.perf_counter() - geometry_started
        if self.mode == "V1-profile-e":
            warm = self._warm_eta if self.profile_warm_start else None
            eta, visibility = self._profile(k, warm_eta=warm)
            self._warm_eta = eta.detach().clone()
            e = torch.as_tensor(visibility["e"], dtype=self.dtype, device=self.device)
        else:
            _fixed_values, visibility = self._fixed_visibility()
            e = self._fixed_e
            assert e is not None
        count_started = time.perf_counter()
        count_components, count_gradient_x, count_gradient_p = self._count_and_gradient(
            x, p_t, k, e, need_gradient=True)
        self._sync()
        self._counters["count_gradient_seconds"] += time.perf_counter() - count_started
        gradient_raw = self._physics._pullback(raw, count_gradient_x)
        gradient_q = torch.as_tensor(dpdq, dtype=self.dtype, device=self.device) * count_gradient_p
        gradient = torch.cat((gradient_raw.reshape(-1), gradient_q.reshape(1)))
        gradient_np = gradient.detach().cpu().numpy().astype(np.float64, copy=True)
        if not np.all(np.isfinite(gradient_np)):
            raise FloatingPointError("nonfinite count-only envelope gradient")
        self._last_visibility = visibility
        self._counters["outer_evaluations"] += 1
        self._sync()
        self._counters["outer_wall_seconds"] += time.perf_counter() - started
        return float(count_components["count_nll_normalized"]), gradient_np

    def value_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, components = self.evaluate(theta, need_gradient=True)
        self._last_theta = np.asarray(theta, dtype=np.float64).copy()
        self._last_value = float(value)
        self._last_gradient = np.asarray(gradient, dtype=np.float64).copy()
        self._last_components = dict(components)
        return self._last_value, self._last_gradient.copy()

    def cached_value_and_components(self, theta: np.ndarray):
        if self._last_theta is None:
            return None
        candidate = np.asarray(theta, dtype=np.float64)
        if candidate.shape != self._last_theta.shape or not np.array_equal(candidate, self._last_theta):
            return None
        return float(self._last_value), dict(self._last_components or {})

    def cached_value_and_gradient(self, theta: np.ndarray):
        if self._last_theta is None:
            return None
        candidate = np.asarray(theta, dtype=np.float64)
        if candidate.shape != self._last_theta.shape or not np.array_equal(candidate, self._last_theta):
            return None
        return float(self._last_value), self._last_gradient.copy(), dict(self._last_components or {})

    def components(self, theta: np.ndarray) -> dict[str, Any]:
        cached = self.cached_value_and_components(theta)
        if cached is not None:
            return cached[1]
        return self.evaluate(theta, need_gradient=False)[2]

    def raw_from_physical(self, coordinates: np.ndarray) -> np.ndarray:
        return self._physics.raw_from_physical(np.asarray(coordinates, dtype=np.float64))

    def physical_coordinates_from_raw(self, raw: np.ndarray) -> np.ndarray:
        return self._physics.physical_coordinates_from_raw(np.asarray(raw, dtype=np.float64))

    def pack(self, raw: np.ndarray, p: float = 0.75) -> np.ndarray:
        return self._physics.pack(np.asarray(raw, dtype=np.float64), p=float(p))

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        return self._physics.unpack(np.asarray(theta, dtype=np.float64))

    def coordinates_and_p(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        return self._physics.coordinates_and_p(np.asarray(theta, dtype=np.float64))

    def visibility_state(self) -> dict[str, Any] | None:
        if self._last_visibility is None:
            return None
        return {key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in self._last_visibility.items()}

    def diagnostics(self) -> dict[str, Any]:
        result = {key: value for key, value in self._counters.items()}
        result.update({
            "mode": self.mode,
            "device": str(self.device),
            "dtype": "torch.float64",
            "mixed_precision": False,
            "sampling": False,
            "pair_block": int(self.pair_block),
            "n_loci": int(self.data.n_loci),
            "n_pairs": int(self.data.n_pairs),
            "full_grid": True,
            "active_count": int(self.support.active_mask.sum()),
            "zero_count": int(self.support.zero_mask.sum()),
            "inner_residual_tolerance": PROFILE_RESIDUAL_TOL,
            "timing_semantics": {
                "geometry_seconds": "sphere map plus k matrix construction; excludes profile, count_gradient, physics",
                "profile_seconds": "eta inner stats, HVP and line search; excludes k construction and physics",
                "count_gradient_seconds": "count envelope gradient after profile; excludes profile and physics",
                "physics_seconds": "bond/repulsion/bend layers; excludes count and profile",
                "outer_wall_seconds": "host wall after final CUDA synchronize; inclusive total evaluate time",
            },
            "last_profile": None if self._last_visibility is None else {
                key: value for key, value in self._last_visibility.items()
                if key not in ("eta", "e", "predicted_degree", "degree")
            },
        })
        return _jsonable(result)

    def profile_arrays(self) -> dict[str, np.ndarray] | None:
        """导出最后一次 profile 的 eta/e/degree，不读取任何 truth 坐标。"""
        if self._last_visibility is None:
            return None
        return {
            "eta": np.asarray(self._last_visibility["eta"], dtype=np.float64).copy(),
            "e": np.asarray(self._last_visibility["e"], dtype=np.float64).copy(),
            "degree": np.asarray(self._last_visibility["degree"], dtype=np.float64).copy(),
            "predicted_degree": (np.asarray(self._last_visibility["predicted_degree"], dtype=np.float64).copy()
                                 if self._last_visibility["predicted_degree"] is not None
                                 else np.zeros(self.data.n_loci, dtype=np.float64)),
            "active_mask": self.support.active_mask.astype(np.bool_, copy=True),
            "zero_mask": self.support.zero_mask.astype(np.bool_, copy=True),
        }


__all__ = [
    "DEFAULT_CG_CAP", "DEFAULT_INNER_CAP", "DEFAULT_PAIR_BLOCK", "PROFILE_RESIDUAL_TOL",
    "ProfileConvergenceError", "SupportAudit", "VisibilityError", "VisibilityGPUObjective",
]
