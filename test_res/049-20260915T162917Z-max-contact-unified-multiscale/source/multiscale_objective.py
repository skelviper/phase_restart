"""multiscale solver 的优化变量包装：z = P^-1 raw_y，q 不变。

objective f(Pz, q)；gradient_z = P gradient_y（P 对称）；canonical 梯度始终是真正的
dF/d(raw_y), dF/dq。`unpack`/`raw_unpack` 都返回 **raw_y**，使旧 runner 的
checkpoint.raw_y 与 BudgetedFitResult.y 不会误报成 z。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from multiscale_precond import ChainPreconditioner


class MultiscaleObjective:
    """把 base objective（raw_y 参数化）搬到 z = P^-1 y 坐标系。"""

    solver_id = "ms"

    def __init__(self, base: Any, preconditioner: ChainPreconditioner):
        self.base = base
        self.preconditioner = preconditioner
        self.data = base.data
        self.n_parameters = int(base.n_parameters)
        self.n_loci = int(base.data.n_loci)
        self.dtype = getattr(base, "dtype", None)
        self.device = getattr(base, "device", None)
        self.model_id = getattr(base, "model_id", "G")
        self._last_theta: np.ndarray | None = None
        self._last_value: float | None = None
        self._last_gradient: np.ndarray | None = None
        self._last_raw_gradient: np.ndarray | None = None
        self._last_components: dict[str, Any] | None = None

    # ------------------------------------------------------------ 坐标变换
    def _split(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        values = np.asarray(theta, dtype=np.float64)
        if values.shape != (self.n_parameters,) or not np.all(np.isfinite(values)):
            raise ValueError("multiscale theta has wrong shape or nonfinite values")
        z = values[:-1].reshape(2, self.n_loci, 3)
        return z, float(values[-1])

    def _raw_theta(self, theta: np.ndarray) -> np.ndarray:
        z, q = self._split(theta)
        y = self.preconditioner.P_apply(z)
        return np.concatenate((y.reshape(-1), np.asarray([q], dtype=np.float64)))

    def _optimizer_theta(self, raw_theta: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_theta, dtype=np.float64)
        y = values[:-1].reshape(2, self.n_loci, 3)
        z = self.preconditioner.Pinv_apply(y)
        return np.concatenate((z.reshape(-1), values[-1:]))

    def pack(self, raw_y: np.ndarray, p: float = 0.75) -> np.ndarray:
        raw_theta = self.base.pack(np.asarray(raw_y, dtype=np.float64), p=float(p))
        return self._optimizer_theta(raw_theta)

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        """返回 canonical raw_y 与 q（不是 z）。"""
        z, q = self._split(theta)
        return self.preconditioner.P_apply(z), q

    def raw_unpack(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        return self.unpack(theta)

    def optimizer_theta(self, theta: np.ndarray) -> np.ndarray:
        z, q = self._split(theta)
        return np.concatenate((z.reshape(-1), np.asarray([q], dtype=np.float64)))

    def canonical_raw_theta(self, theta: np.ndarray) -> np.ndarray:
        return self._raw_theta(theta)

    def raw_from_physical(self, coordinates: np.ndarray) -> np.ndarray:
        return self.base.raw_from_physical(np.asarray(coordinates, dtype=np.float64))

    def physical_coordinates_from_raw(self, raw: np.ndarray) -> np.ndarray:
        return self.base.physical_coordinates_from_raw(np.asarray(raw, dtype=np.float64))

    def coordinates_and_p(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        return self.base.coordinates_and_p(self._raw_theta(theta))

    def validate_physical_coordinates(self, coordinates: np.ndarray) -> None:
        self.base.validate_physical_coordinates(np.asarray(coordinates, dtype=np.float64))

    def physical_gradient(self):
        return self.base.physical_gradient()

    # ---------------------------------------------------------------- 目标值
    def evaluate(self, theta: np.ndarray, need_gradient: bool = True):
        raw_theta = self._raw_theta(theta)
        value, raw_gradient, components = self.base.evaluate(raw_theta, need_gradient=need_gradient)
        if not need_gradient:
            return float(value), None, dict(components)
        raw_gradient = np.asarray(raw_gradient, dtype=np.float64)
        raw_y_gradient = raw_gradient[:-1].reshape(2, self.n_loci, 3)
        optimizer_y_gradient = self.preconditioner.P_apply(raw_y_gradient)
        optimizer_gradient = np.concatenate((optimizer_y_gradient.reshape(-1), raw_gradient[-1:]))
        if not np.all(np.isfinite(optimizer_gradient)):
            raise FloatingPointError("multiscale transformed gradient is nonfinite")
        self._last_theta = np.asarray(theta, dtype=np.float64).copy()
        self._last_value = float(value)
        self._last_gradient = optimizer_gradient.copy()
        self._last_raw_gradient = raw_gradient.copy()
        self._last_components = dict(components)
        return float(value), optimizer_gradient, dict(components)

    def value_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _components = self.evaluate(theta, need_gradient=True)
        return float(value), np.asarray(gradient, dtype=np.float64).copy()

    def cached_value_and_components(self, theta: np.ndarray):
        if self._last_theta is None:
            return None
        values = np.asarray(theta, dtype=np.float64)
        if values.shape != self._last_theta.shape or not np.array_equal(values, self._last_theta):
            return None
        return float(self._last_value), dict(self._last_components or {})

    def cached_value_and_gradient(self, theta: np.ndarray):
        if self._last_theta is None:
            return None
        values = np.asarray(theta, dtype=np.float64)
        if values.shape != self._last_theta.shape or not np.array_equal(values, self._last_theta):
            return None
        return float(self._last_value), self._last_gradient.copy(), self._last_raw_gradient.copy()

    def components(self, theta: np.ndarray) -> dict[str, Any]:
        cached = self.cached_value_and_components(theta)
        if cached is not None:
            return cached[1]
        return self.evaluate(theta, need_gradient=False)[2]

    def canonical_raw_gradient(self, theta: np.ndarray, optimizer_gradient: np.ndarray | None = None) -> np.ndarray:
        """把 z 空间梯度还原为真正的 dF/d(raw_y), dF/dq。

        z = P^-1 y  =>  y = P z  =>  grad_z = P grad_y（P 对称）
        因此 grad_y = P^-1 grad_z，必须用 Pinv_apply，不能用 P_apply。
        """
        if optimizer_gradient is None:
            cached = self.cached_value_and_gradient(theta)
            if cached is None:
                _, optimizer_gradient, _ = self.evaluate(theta, need_gradient=True)
            else:
                _, optimizer_gradient, _ = cached
        values = np.asarray(optimizer_gradient, dtype=np.float64)
        z_gradient = values[:-1].reshape(2, self.n_loci, 3)
        raw_y_gradient = self.preconditioner.Pinv_apply(z_gradient)
        return np.concatenate((raw_y_gradient.reshape(-1), values[-1:]))

    def count_value_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        raw_theta = self._raw_theta(theta)
        value, raw_gradient = self.base.count_value_and_grad(raw_theta)
        raw_gradient = np.asarray(raw_gradient, dtype=np.float64)
        y_gradient = self.preconditioner.P_apply(raw_gradient[:-1].reshape(2, self.n_loci, 3))
        return float(value), np.concatenate((y_gradient.reshape(-1), raw_gradient[-1:]))

    def map_diagnostics(self) -> dict[str, Any]:
        return {
            "solver_id": self.solver_id,
            "optimizer_coordinate_parameterization": "z = P^-1 raw_y, q unchanged",
            "canonical_gradient_space": "raw_y/q",
            "physical_objective_unchanged": True,
            **self.preconditioner.diagnostics(),
        }

    def backend_metadata(self) -> dict[str, Any]:
        return {
            "solver_id": self.solver_id,
            "preconditioner": "chain_multiscale_neumann_path_graph",
            "physical_objective_unchanged": True,
            "symbol_min": self.preconditioner.min_symbol,
            "symbol_max": self.preconditioner.max_symbol,
        }

    def raw_theta_roundtrip_error(self, theta: np.ndarray) -> float:
        raw_theta = self._raw_theta(theta)
        return float(np.max(np.abs(self._optimizer_theta(raw_theta) - np.asarray(theta, dtype=np.float64))))


def make_preconditioner(data: Any, *, identity: bool = False) -> ChainPreconditioner:
    return ChainPreconditioner(data, identity=identity)


def check_roundtrip(data: Any, tolerance: float = 1e-12) -> float:
    """初始化 roundtrip 门禁：Pinv(P(y)) == y。"""
    rng = np.random.default_rng(24901)
    y = rng.normal(size=(2, int(data.n_loci), 3)) * 0.3
    pre = ChainPreconditioner(data)
    error = float(np.max(np.abs(pre.Pinv_apply(pre.P_apply(y)) - y)))
    if not math.isfinite(error) or error > tolerance:
        raise AssertionError("multiscale roundtrip error %.3e exceeds %.1e" % (error, tolerance))
    return error


__all__ = ["MultiscaleObjective", "make_preconditioner", "check_roundtrip"]
