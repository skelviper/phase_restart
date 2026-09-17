"""058 预注册方法的纯 NumPy 核心；不加载真实数据或评价标签。"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


class FrozenMethodError(RuntimeError):
    """冻结接口或数值契约不满足。"""


def select_two_candidates(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """按 058 冻结规则选择 full-J 第一名和排除后的 count 第一名。"""
    values = [dict(row) for row in rows]
    if len(values) < 2:
        raise FrozenMethodError("candidate scan must contain at least two rows")
    first = min(values, key=lambda row: (float(row["delta_full_J"]), int(row["candidate_order"])))
    second = min(
        (row for row in values if int(row["candidate_order"]) != int(first["candidate_order"])),
        key=lambda row: (float(row["delta_count_Nraw"]), int(row["candidate_order"])),
    )
    return first, second


def swap_complete_interval(coordinates: np.ndarray, raw_y: np.ndarray, start: int, stop: int):
    """同时交换两 copy 的 xyz/raw-y，并保持逐 locus 无序点集。"""
    x = np.asarray(coordinates, dtype=np.float64)
    y = np.asarray(raw_y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3 or x.shape[0] != 2 or x.shape[2] != 3:
        raise FrozenMethodError("coordinates/raw_y must share shape (2,n_loci,3)")
    if not (0 <= int(start) < int(stop) <= x.shape[1]):
        raise FrozenMethodError("invalid half-open swap interval")
    changed_x = x.copy()
    changed_y = y.copy()
    changed_x[:, start:stop] = changed_x[::-1, start:stop]
    changed_y[:, start:stop] = changed_y[::-1, start:stop]
    outside = np.ones(x.shape[1], dtype=bool)
    outside[start:stop] = False
    if not np.array_equal(changed_x[:, outside], x[:, outside]):
        raise FrozenMethodError("swap changed coordinates outside interval")
    if not np.array_equal(changed_y[:, outside], y[:, outside]):
        raise FrozenMethodError("swap changed raw_y outside interval")
    direct = np.all(changed_x[:, start:stop] == x[:, start:stop], axis=2)
    swapped = np.all(changed_x[:, start:stop] == x[::-1, start:stop], axis=2)
    if not np.all(direct | swapped):
        raise FrozenMethodError("swap changed per-locus complete-3D coordinate point sets")
    return changed_x, changed_y


class RestrictedObjective:
    """把 active 向量装回 full theta，并只返回对应 gradient。"""

    def __init__(self, full_objective: Any, source_theta: np.ndarray, active_indices: np.ndarray):
        self.full_objective = full_objective
        self.source_theta = np.asarray(source_theta, dtype=np.float64).copy()
        self.active_indices = np.asarray(active_indices, dtype=np.int64).copy()
        if self.source_theta.ndim != 1 or not np.isfinite(self.source_theta).all():
            raise FrozenMethodError("source_theta must be finite and one-dimensional")
        if self.active_indices.ndim != 1 or len(np.unique(self.active_indices)) != len(self.active_indices):
            raise FrozenMethodError("active indices must be one-dimensional and unique")
        if np.any(self.active_indices < 0) or np.any(self.active_indices >= len(self.source_theta) - 1):
            raise FrozenMethodError("active indices must select raw-y only, never q")
        self._cache = None

    def initial_active(self) -> np.ndarray:
        return self.source_theta[self.active_indices].copy()

    def full_theta(self, active_y: np.ndarray) -> np.ndarray:
        active_y = np.asarray(active_y, dtype=np.float64)
        if active_y.shape != self.active_indices.shape or not np.isfinite(active_y).all():
            raise FrozenMethodError("active_y has wrong shape or nonfinite values")
        theta = self.source_theta.copy()
        theta[self.active_indices] = active_y
        return theta

    def value_and_grad(self, active_y: np.ndarray):
        theta = self.full_theta(active_y)
        value, gradient, components = self.full_objective.evaluate(theta, need_gradient=True)
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.shape != theta.shape or not math.isfinite(float(value)) or not np.isfinite(gradient).all():
            raise FrozenMethodError("full objective returned invalid value/gradient")
        self._cache = {
            "active_y": np.asarray(active_y, dtype=np.float64).copy(),
            "full_theta": theta,
            "value": float(value),
            "full_gradient": gradient.copy(),
            "components": dict(components),
        }
        return float(value), gradient[self.active_indices].copy()

    def cached_full(self, active_y: np.ndarray):
        if self._cache is None or not np.array_equal(np.asarray(active_y), self._cache["active_y"]):
            return None
        return self._cache


@dataclass(frozen=True)
class FixedXPCache:
    """固定 x/e 后的完整 G 标量 p profile cache。

    只有正计数 cis 项保留逐 pair 数组；所有 zero-count cis 与全部 inter 项在构造时预聚合。
    """

    cis_a_observed: np.ndarray
    cis_b_observed: np.ndarray
    cis_counts_observed: np.ndarray
    sum_b_all_cis: float
    sum_delta_all_cis: float
    z_inter: float
    sum_Cinter_log_rate: float
    n_off: float
    n_raw: float
    count_constant_raw: float
    physical_constant: float
    p_prior_strength: float = 1e-4

    def __post_init__(self):
        arrays = tuple(np.asarray(x, dtype=np.float64) for x in (
            self.cis_a_observed, self.cis_b_observed, self.cis_counts_observed))
        a, b, counts = arrays
        if a.shape != b.shape or a.shape != counts.shape:
            raise FrozenMethodError("observed cis p-cache arrays have inconsistent shapes")
        if any(x.ndim != 1 or not np.isfinite(x).all() for x in arrays):
            raise FrozenMethodError("observed cis p-cache arrays must be finite vectors")
        if np.any(a <= 0) or np.any(b <= 0) or np.any(counts <= 0):
            raise FrozenMethodError("observed cis rates/counts must be positive")
        scalars = (self.sum_b_all_cis, self.sum_delta_all_cis, self.z_inter,
                   self.sum_Cinter_log_rate, self.n_off, self.n_raw,
                   self.count_constant_raw, self.physical_constant)
        if not all(math.isfinite(float(value)) for value in scalars):
            raise FrozenMethodError("p-cache aggregates must be finite")
        if self.sum_b_all_cis <= 0 or self.z_inter <= 0 or self.n_off <= 0 or self.n_raw <= 0:
            raise FrozenMethodError("p-cache positive aggregates are outside domain")

    @classmethod
    def from_pair_arrays(cls, *, cis_a: np.ndarray, cis_b: np.ndarray,
                         cis_counts: np.ndarray, inter_rates: np.ndarray,
                         inter_counts: np.ndarray, n_raw: float,
                         count_constant_raw: float, physical_constant: float,
                         p_prior_strength: float = 1e-4):
        """一次预聚合完整 pair arrays；正式 profile 不再扫描 inter 或 zero-count cis。"""
        a = np.asarray(cis_a, dtype=np.float64)
        b = np.asarray(cis_b, dtype=np.float64)
        counts = np.asarray(cis_counts, dtype=np.float64)
        inter_rates = np.asarray(inter_rates, dtype=np.float64)
        inter_counts = np.asarray(inter_counts, dtype=np.float64)
        if a.shape != b.shape or a.shape != counts.shape or inter_rates.shape != inter_counts.shape:
            raise FrozenMethodError("full p-cache pair arrays have inconsistent shapes")
        if any(x.ndim != 1 or not np.isfinite(x).all()
               for x in (a, b, counts, inter_rates, inter_counts)):
            raise FrozenMethodError("full p-cache pair arrays must be finite vectors")
        if (np.any(a <= 0) or np.any(b <= 0) or np.any(inter_rates <= 0)
                or np.any(counts < 0) or np.any(inter_counts < 0)):
            raise FrozenMethodError("full p-cache rates/counts are outside domain")
        observed = counts > 0
        inter_observed = inter_counts > 0
        return cls(
            cis_a_observed=a[observed].copy(),
            cis_b_observed=b[observed].copy(),
            cis_counts_observed=counts[observed].copy(),
            sum_b_all_cis=float(b.sum()),
            sum_delta_all_cis=float((a - b).sum()),
            z_inter=float(inter_rates.sum()),
            sum_Cinter_log_rate=float(np.dot(inter_counts[inter_observed],
                                               np.log(inter_rates[inter_observed]))),
            n_off=float(counts.sum() + inter_counts.sum()),
            n_raw=float(n_raw),
            count_constant_raw=float(count_constant_raw),
            physical_constant=float(physical_constant),
            p_prior_strength=float(p_prior_strength),
        )

    def value_derivative(self, p: float) -> tuple[float, float, dict[str, float]]:
        p = float(p)
        p_lo = float(np.nextafter(0.0001, 1.0))
        p_hi = 0.9998999999999998
        if not p_lo <= p <= p_hi:
            raise FrozenMethodError("p outside frozen numerical interior interval")
        a = np.asarray(self.cis_a_observed, dtype=np.float64)
        b = np.asarray(self.cis_b_observed, dtype=np.float64)
        counts = np.asarray(self.cis_counts_observed, dtype=np.float64)
        delta = a - b
        observed_rate = b + p * delta
        if np.any(observed_rate <= 0):
            raise FrozenMethodError("nonpositive profiled observed cis rate")
        z_cis = float(self.sum_b_all_cis + p * self.sum_delta_all_cis)
        z_all = z_cis + float(self.z_inter)
        observed_log = float(np.dot(counts, np.log(observed_rate))
                             + self.sum_Cinter_log_rate)
        count = (self.n_off * math.log(z_all) - observed_log
                 + float(self.count_constant_raw)) / self.n_raw
        prior = -self.p_prior_strength * math.log(p * (1.0 - p))
        value = count + prior + float(self.physical_constant)
        derivative = (
            (self.n_off / z_all) * self.sum_delta_all_cis
            - float(np.dot(counts, delta / observed_rate))
        ) / self.n_raw + self.p_prior_strength * (1.0 / (1.0 - p) - 1.0 / p)
        if not (math.isfinite(value) and math.isfinite(derivative)):
            raise FrozenMethodError("nonfinite p profile result")
        span = 1.0 - 2.0e-4
        dpdq = (p - 1.0e-4) * (0.9999 - p) / span
        return value, derivative, {"p": p, "count": count, "p_prior": prior,
                                   "physical_constant": float(self.physical_constant),
                                   "count_constant_raw": float(self.count_constant_raw),
                                   "dJ_dp": derivative, "dp_dq": dpdq,
                                   "dJ_dq": derivative * dpdq,
                                   "Zcis": z_cis, "Zinter": float(self.z_inter),
                                   "Zall": z_all}
