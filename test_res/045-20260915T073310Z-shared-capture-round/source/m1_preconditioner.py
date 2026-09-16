"""M1 raw-y common/difference preconditioner and hard-budget L-BFGS.

The wrapped objective remains the unchanged C0 physical objective.  M1 only
changes the optimizer coordinates:

    a = (yA + yB) / sqrt(2)
    v = (yA - yB) / (2*sqrt(2))

so that yA=(a+2v)/sqrt(2), yB=(a-2v)/sqrt(2).  The q coordinate is copied.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping

import numpy as np
from scipy.optimize import OptimizeResult, minimize


SQRT2 = math.sqrt(2.0)
M1_SCALE = 2.0


class FGBudgetExceeded(RuntimeError):
    """Raised before an objective call would exceed the declared FG cap."""


class CanonicalGTolReached(RuntimeError):
    """Raised after an accepted state reaches the canonical raw-y/q tolerance."""


def raw_to_common_difference(y: np.ndarray, scale: float = M1_SCALE) -> np.ndarray:
    """Map raw ``(yA,yB)`` to optimizer ``(a,v)`` with an explicit scale."""
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 2 or values.shape[2] != 3:
        raise ValueError("raw y must have shape (2,n_loci,3)")
    if not np.all(np.isfinite(values)) or not math.isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("raw y and scale must be finite; scale must be positive")
    common = (values[0] + values[1]) / SQRT2
    differential = (values[0] - values[1]) / (SQRT2 * float(scale))
    return np.stack((common, differential), axis=0)


def common_difference_to_raw(z: np.ndarray, scale: float = M1_SCALE) -> np.ndarray:
    """Reconstruct raw ``(yA,yB)`` from optimizer ``(a,v)``."""
    values = np.asarray(z, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 2 or values.shape[2] != 3:
        raise ValueError("common/difference coordinates must have shape (2,n_loci,3)")
    if not np.all(np.isfinite(values)) or not math.isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("optimizer coordinates and scale must be finite; scale must be positive")
    common, differential = values
    return np.stack(
        ((common + float(scale) * differential) / SQRT2,
         (common - float(scale) * differential) / SQRT2),
        axis=0,
    )


def raw_gradient_to_common_difference(gradient_y: np.ndarray, scale: float = M1_SCALE) -> np.ndarray:
    """Apply the exact chain rule from raw-y gradient to optimizer gradient."""
    values = np.asarray(gradient_y, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 2 or values.shape[2] != 3:
        raise ValueError("raw gradient must have shape (2,n_loci,3)")
    if not np.all(np.isfinite(values)) or not math.isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("gradient and scale must be finite; scale must be positive")
    common = (values[0] + values[1]) / SQRT2
    differential = float(scale) * (values[0] - values[1]) / SQRT2
    return np.stack((common, differential), axis=0)


def common_difference_gradient_to_raw(gradient_z: np.ndarray, scale: float = M1_SCALE) -> np.ndarray:
    """Invert the M1 gradient chain rule for canonical stopping diagnostics."""
    values = np.asarray(gradient_z, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 2 or values.shape[2] != 3:
        raise ValueError("optimizer gradient must have shape (2,n_loci,3)")
    if not np.all(np.isfinite(values)) or not math.isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("gradient and scale must be finite; scale must be positive")
    common, differential = values
    return np.stack(
        ((common + differential / float(scale)) / SQRT2,
         (common - differential / float(scale)) / SQRT2),
        axis=0,
    )


def orthogonal_raw_to_ab(y: np.ndarray) -> np.ndarray:
    """Scale-one orthogonal transform used only by the mathematical precheck."""
    return raw_to_common_difference(y, scale=1.0)


def orthogonal_ab_to_raw(z: np.ndarray) -> np.ndarray:
    """Inverse of the scale-one orthogonal mathematical precheck transform."""
    return common_difference_to_raw(z, scale=1.0)


class M1Objective:
    """Thin adapter exposing unchanged C0 physics through M1 optimizer variables."""

    model_id = "C0"
    preconditioner = "raw_y_common_difference"
    scale = M1_SCALE

    def __init__(self, base_objective: Any):
        if str(getattr(base_objective, "model_id", "")) != "C0":
            raise ValueError("M1 is frozen as a C0-only optimizer preconditioner")
        self.base = base_objective
        self.data = base_objective.data
        self.n_parameters = int(base_objective.n_parameters)
        self.dtype = getattr(base_objective, "dtype", None)
        self.device = getattr(base_objective, "device", None)
        self.backend = getattr(base_objective, "backend", None)
        self._last_theta: np.ndarray | None = None
        self._last_value: float | None = None
        self._last_gradient: np.ndarray | None = None
        self._last_raw_gradient: np.ndarray | None = None
        self._last_components: dict[str, Any] | None = None

    @property
    def n_loci(self) -> int:
        return int(self.data.n_loci)

    def _validate_theta(self, theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype=np.float64)
        if values.shape != (self.n_parameters,) or not np.all(np.isfinite(values)):
            raise ValueError("M1 optimizer theta has wrong shape or nonfinite values")
        return values

    def _to_raw_theta(self, theta: np.ndarray) -> np.ndarray:
        values = self._validate_theta(theta)
        z = values[:-1].reshape(2, self.n_loci, 3)
        raw = common_difference_to_raw(z, scale=self.scale)
        return np.concatenate((raw.reshape(-1), values[-1:]))

    def _from_raw_theta(self, raw_theta: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_theta, dtype=np.float64)
        if values.shape != (self.n_parameters,) or not np.all(np.isfinite(values)):
            raise ValueError("raw theta has wrong shape or nonfinite values")
        raw = values[:-1].reshape(2, self.n_loci, 3)
        z = raw_to_common_difference(raw, scale=self.scale)
        return np.concatenate((z.reshape(-1), values[-1:]))

    def raw_theta(self, theta: np.ndarray) -> np.ndarray:
        """Return the exact C0 raw-y/q theta corresponding to M1 theta."""
        return self._to_raw_theta(theta).copy()

    def optimizer_theta_from_raw_theta(self, raw_theta: np.ndarray) -> np.ndarray:
        """Convert a raw-y/q theta to M1 variables without changing q."""
        return self._from_raw_theta(raw_theta).copy()

    def pack(self, raw_y: np.ndarray, p: float = 0.75) -> np.ndarray:
        raw_theta = self.base.pack(np.asarray(raw_y, dtype=np.float64), p=float(p))
        return self._from_raw_theta(raw_theta)

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        values = self._validate_theta(theta)
        return values[:-1].reshape(2, self.n_loci, 3).copy(), float(values[-1])

    def raw_unpack(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        raw = self._to_raw_theta(theta)
        return raw[:-1].reshape(2, self.n_loci, 3), float(raw[-1])

    def raw_from_physical(self, coordinates: np.ndarray) -> np.ndarray:
        return self.base.raw_from_physical(np.asarray(coordinates, dtype=np.float64))

    def physical_coordinates_from_raw(self, raw: np.ndarray) -> np.ndarray:
        return self.base.physical_coordinates_from_raw(np.asarray(raw, dtype=np.float64))

    def coordinates_and_p(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        raw_theta = self._to_raw_theta(theta)
        return self.base.coordinates_and_p(raw_theta)

    def validate_physical_coordinates(self, coordinates: np.ndarray) -> None:
        self.base.validate_physical_coordinates(np.asarray(coordinates, dtype=np.float64))

    def evaluate(self, theta: np.ndarray, need_gradient: bool = True):
        raw_theta = self._to_raw_theta(theta)
        value, raw_gradient, components = self.base.evaluate(raw_theta, need_gradient=need_gradient)
        if not need_gradient:
            return float(value), None, dict(components)
        raw_gradient = np.asarray(raw_gradient, dtype=np.float64)
        raw_y_gradient = raw_gradient[:-1].reshape(2, self.n_loci, 3)
        optimizer_y_gradient = raw_gradient_to_common_difference(raw_y_gradient, scale=self.scale)
        optimizer_gradient = np.concatenate((optimizer_y_gradient.reshape(-1), raw_gradient[-1:]))
        if not np.all(np.isfinite(optimizer_gradient)):
            raise FloatingPointError("M1 transformed gradient is nonfinite")
        return float(value), optimizer_gradient, dict(components)

    def value_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        values = self._validate_theta(theta)
        raw_theta = self._to_raw_theta(values)
        value, raw_gradient = self.base.value_and_grad(raw_theta)
        raw_gradient = np.asarray(raw_gradient, dtype=np.float64)
        raw_y_gradient = raw_gradient[:-1].reshape(2, self.n_loci, 3)
        optimizer_y_gradient = raw_gradient_to_common_difference(raw_y_gradient, scale=self.scale)
        optimizer_gradient = np.concatenate((optimizer_y_gradient.reshape(-1), raw_gradient[-1:]))
        components_cached = self.base.cached_value_and_components(raw_theta)
        if components_cached is None:
            _, _, components = self.base.evaluate(raw_theta, need_gradient=False)
        else:
            _, components = components_cached
        self._last_theta = values.copy()
        self._last_value = float(value)
        self._last_gradient = optimizer_gradient.copy()
        self._last_raw_gradient = raw_gradient.copy()
        self._last_components = dict(components)
        return float(value), optimizer_gradient.copy()

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

    def canonical_raw_gradient(self, theta: np.ndarray, optimizer_gradient: np.ndarray | None = None) -> np.ndarray:
        if optimizer_gradient is None:
            cached = self.cached_value_and_gradient(theta)
            if cached is None:
                _, optimizer_gradient, _ = self.evaluate(theta, need_gradient=True)
            else:
                _, optimizer_gradient, _ = cached
        values = np.asarray(optimizer_gradient, dtype=np.float64)
        y_gradient = values[:-1].reshape(2, self.n_loci, 3)
        raw_y_gradient = common_difference_gradient_to_raw(y_gradient, scale=self.scale)
        return np.concatenate((raw_y_gradient.reshape(-1), values[-1:]))

    def components(self, theta: np.ndarray) -> dict[str, Any]:
        cached = self.cached_value_and_components(theta)
        if cached is not None:
            return cached[1]
        return self.evaluate(theta, need_gradient=False)[2]

    def map_diagnostics(self) -> dict[str, Any]:
        result = dict(self.base.map_diagnostics())
        result.update({
            "optimizer_coordinate_parameterization": "(a=(yA+yB)/sqrt2, v=(yA-yB)/(2*sqrt2), q)",
            "preconditioner": self.preconditioner,
            "fixed_scale": float(self.scale),
            "physical_objective_unchanged": True,
        })
        return result

    def backend_metadata(self) -> dict[str, Any]:
        result = dict(self.base.backend_metadata())
        result.update({
            "model_id": "C0",
            "optimizer_coordinate_parameterization": "raw_y_common_difference",
            "preconditioner": self.preconditioner,
            "fixed_scale": float(self.scale),
            "physical_objective_unchanged": True,
        })
        return result


@dataclass
class BudgetedFitResult:
    """Result whose endpoint is explicitly a callback-confirmed accepted state."""

    objective: Any
    theta: np.ndarray
    y: np.ndarray
    coordinates: np.ndarray
    p: float
    fun: float
    components: dict[str, Any]
    success: bool
    status: int
    message: str
    terminal_reason: str
    budget_exhausted: bool
    nit: int
    nfev: int
    actual_nfev: int
    scipy_nfev: int
    njev: int
    elapsed_seconds: float
    history: list[dict[str, Any]]
    canonical_gradient_max_abs: float
    canonical_gradient_norm: float
    initial_total: float
    initial_components: dict[str, Any]
    validation_calls: int
    endpoint_was_last_accepted: bool
    raw_scipy_result: OptimizeResult | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "status": int(self.status),
            "message": str(self.message),
            "terminal_reason": str(self.terminal_reason),
            "budget_exhausted": bool(self.budget_exhausted),
            "nit": int(self.nit),
            "nfev": int(self.nfev),
            "actual_nfev": int(self.actual_nfev),
            "scipy_nfev": int(self.scipy_nfev),
            "njev": int(self.njev),
            "elapsed_seconds": float(self.elapsed_seconds),
            "canonical_gradient_max_abs": float(self.canonical_gradient_max_abs),
            "canonical_gradient_norm": float(self.canonical_gradient_norm),
            "initial_total": float(self.initial_total),
            "final_total": float(self.fun),
            "validation_calls": int(self.validation_calls),
            "endpoint_was_last_accepted": bool(self.endpoint_was_last_accepted),
        }


def _canonical_gradient(objective: Any, theta: np.ndarray, optimizer_gradient: np.ndarray) -> np.ndarray:
    if isinstance(objective, M1Objective):
        return objective.canonical_raw_gradient(theta, optimizer_gradient)
    return np.asarray(optimizer_gradient, dtype=np.float64).copy()


def _canonical_metrics(gradient: np.ndarray) -> tuple[float, float]:
    values = np.asarray(gradient, dtype=np.float64)
    return float(np.max(np.abs(values), initial=0.0)), float(np.linalg.norm(values))


def run_budgeted_lbfgs(
    objective: Any,
    initial_y: np.ndarray,
    *,
    p_init: float = 0.75,
    q_init: float | None = None,
    maxfun: int,
    maxiter: int | None = None,
    maxls: int = 20,
    ftol: float = 1e-10,
    canonical_gtol: float = 1e-6,
    checkpoint_every: int | None = None,
    checkpoint_hook: Callable[[dict[str, Any]], None] | None = None,
    accepted_callback: Callable[[dict[str, Any]], None] | None = None,
) -> BudgetedFitResult:
    """Run L-BFGS-B with an exact FG cap and canonical raw-y/q stopping.

    ``maxfun`` counts every SciPy value+analytic-gradient call, including line
    search probes.  The initial value-only diagnostic and final readback are
    outside that counter and are recorded separately.  A cap exception discards
    any unaccepted trial and returns the last callback-confirmed state.
    """
    initial_y = np.asarray(initial_y, dtype=np.float64)
    if initial_y.shape != (2, objective.data.n_loci, 3) or not np.all(np.isfinite(initial_y)):
        raise ValueError("initial_y must be finite with shape (2,n_loci,3)")
    if not isinstance(maxfun, (int, np.integer)) or int(maxfun) <= 0:
        raise ValueError("maxfun must be a positive integer")
    if maxiter is None:
        maxiter = int(maxfun) + 1
    if not isinstance(maxiter, (int, np.integer)) or int(maxiter) <= 0:
        raise ValueError("maxiter must be a positive integer")
    if not isinstance(maxls, (int, np.integer)) or int(maxls) <= 0:
        raise ValueError("maxls must be a positive integer")
    if not math.isfinite(float(ftol)) or float(ftol) < 0.0:
        raise ValueError("ftol must be finite and nonnegative")
    if not math.isfinite(float(canonical_gtol)) or float(canonical_gtol) < 0.0:
        raise ValueError("canonical_gtol must be finite and nonnegative")
    if checkpoint_every is not None and int(checkpoint_every) <= 0:
        raise ValueError("checkpoint_every must be positive when supplied")

    theta0 = objective.pack(initial_y, p=float(p_init))
    if q_init is not None:
        if not math.isfinite(float(q_init)):
            raise ValueError("q_init must be finite")
        theta0[-1] = float(q_init)
    theta0 = np.asarray(theta0, dtype=np.float64)
    initial_total, _, initial_components = objective.evaluate(theta0, need_gradient=False)
    started = time.perf_counter()
    history: list[dict[str, Any]] = [{
        "iteration": 0,
        "nfev": 0,
        "elapsed_seconds": 0.0,
        "fun": float(initial_total),
        "p": float(initial_components["p"]),
        "components": dict(initial_components),
        "state_status": "accepted_initial",
    }]
    nfev = 0
    njev = 0
    validation_calls = 0
    last_eval_theta: np.ndarray | None = None
    last_eval_gradient: np.ndarray | None = None
    last_eval_value: float | None = None
    accepted_theta = theta0.copy()
    accepted_iterations = 0
    stop_exception: Exception | None = None

    def loss_and_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal nfev, njev, last_eval_theta, last_eval_gradient, last_eval_value
        if nfev >= int(maxfun):
            raise FGBudgetExceeded("fg_budget_exhausted_before_next_objective_call")
        nfev += 1
        njev += 1
        value, gradient = objective.value_and_grad(np.asarray(theta, dtype=np.float64))
        gradient = np.asarray(gradient, dtype=np.float64)
        if not np.isfinite(float(value)) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("nonfinite value or gradient from objective")
        last_eval_theta = np.asarray(theta, dtype=np.float64).copy()
        last_eval_gradient = gradient.copy()
        last_eval_value = float(value)
        return float(value), gradient.copy()

    def recorder(theta: np.ndarray) -> None:
        nonlocal accepted_theta, accepted_iterations, validation_calls
        theta = np.asarray(theta, dtype=np.float64).copy()
        accepted_theta = theta.copy()
        accepted_iterations += 1
        cached = objective.cached_value_and_components(theta)
        if cached is not None:
            value, components = cached
        elif last_eval_theta is not None and np.array_equal(theta, last_eval_theta):
            value = float(last_eval_value)
            components = objective.components(theta)
        else:
            validation_calls += 1
            value, _, components = objective.evaluate(theta, need_gradient=True)
        if last_eval_theta is not None and last_eval_gradient is not None and np.array_equal(theta, last_eval_theta):
            optimizer_gradient = last_eval_gradient.copy()
        else:
            validation_calls += 1
            _, optimizer_gradient, _ = objective.evaluate(theta, need_gradient=True)
        canonical = _canonical_gradient(objective, theta, optimizer_gradient)
        max_abs, norm = _canonical_metrics(canonical)
        entry = {
            "iteration": int(accepted_iterations),
            "nfev": int(nfev),
            "elapsed_seconds": time.perf_counter() - started,
            "fun": float(value),
            "p": float(components["p"]),
            "canonical_gradient_max_abs": max_abs,
            "canonical_gradient_norm": norm,
            "components": dict(components),
            "state_status": "accepted",
        }
        history.append(entry)
        if checkpoint_hook is not None and checkpoint_every is not None and accepted_iterations % int(checkpoint_every) == 0:
            checkpoint_hook({
                "iteration": int(accepted_iterations),
                "nfev": int(nfev),
                "elapsed_seconds": float(entry["elapsed_seconds"]),
                "fun": float(value),
                "p": float(components["p"]),
                "theta": theta.copy(),
                "raw_y": objective.raw_unpack(theta)[0].copy() if isinstance(objective, M1Objective)
                else objective.unpack(theta)[0].copy(),
                "coordinates": objective.coordinates_and_p(theta)[0].copy(),
                "components": dict(components),
                "optimizer_gradient": optimizer_gradient.copy(),
                "canonical_gradient": canonical.copy(),
            })
        if accepted_callback is not None:
            accepted_callback(dict(entry))
        if max_abs <= float(canonical_gtol):
            raise CanonicalGTolReached("canonical_raw_y_q_gtol_reached")

    options = {
        "maxiter": int(maxiter),
        "maxfun": int(maxfun),
        "maxls": int(maxls),
        "ftol": float(ftol),
        # Disable SciPy's coordinate-dependent projected-gradient stop. The
        # canonical raw-y/q test above is shared by M0 and M1.
        "gtol": 0.0,
    }
    scipy_result: OptimizeResult | None = None
    try:
        scipy_result = minimize(
            loss_and_gradient,
            theta0,
            method="L-BFGS-B",
            jac=True,
            callback=recorder,
            options=options,
        )
    except (FGBudgetExceeded, CanonicalGTolReached) as exc:
        stop_exception = exc

    endpoint_theta = accepted_theta.copy()
    endpoint_was_last_accepted = True
    if scipy_result is not None:
        raw_x = np.asarray(getattr(scipy_result, "x", endpoint_theta), dtype=np.float64)
        if raw_x.shape == endpoint_theta.shape and np.array_equal(raw_x, endpoint_theta):
            endpoint_was_last_accepted = True
        else:
            endpoint_was_last_accepted = False
    endpoint_cached = objective.cached_value_and_components(endpoint_theta)
    if endpoint_cached is not None:
        endpoint_value, endpoint_components = endpoint_cached
    elif last_eval_theta is not None and np.array_equal(endpoint_theta, last_eval_theta):
        endpoint_value = float(last_eval_value)
        endpoint_components = objective.components(endpoint_theta)
    else:
        validation_calls += 1
        endpoint_value, _, endpoint_components = objective.evaluate(endpoint_theta, need_gradient=True)
    endpoint_gradient_cached = getattr(objective, "cached_value_and_gradient", lambda _theta: None)(endpoint_theta)
    if endpoint_gradient_cached is not None:
        _, endpoint_optimizer_gradient, _endpoint_raw = endpoint_gradient_cached
    elif last_eval_theta is not None and last_eval_gradient is not None and np.array_equal(endpoint_theta, last_eval_theta):
        endpoint_optimizer_gradient = last_eval_gradient.copy()
    else:
        validation_calls += 1
        _, endpoint_optimizer_gradient, _ = objective.evaluate(endpoint_theta, need_gradient=True)
    endpoint_canonical = _canonical_gradient(objective, endpoint_theta, endpoint_optimizer_gradient)
    canonical_max_abs, canonical_norm = _canonical_metrics(endpoint_canonical)
    endpoint_coordinates, endpoint_p = objective.coordinates_and_p(endpoint_theta)

    if stop_exception is not None:
        if isinstance(stop_exception, FGBudgetExceeded):
            success = False
            status = 1
            message = str(stop_exception)
            terminal_reason = "fg_budget_exhausted"
            budget_exhausted = True
        else:
            success = True
            status = 0
            message = str(stop_exception)
            terminal_reason = "canonical_gtol"
            budget_exhausted = False
        scipy_nfev = int(nfev)
        scipy_nit = int(accepted_iterations)
        scipy_njev = int(njev)
    else:
        assert scipy_result is not None
        scipy_nfev = int(getattr(scipy_result, "nfev", nfev) or nfev)
        scipy_nit = int(accepted_iterations)
        scipy_njev = int(getattr(scipy_result, "njev", njev) or njev)
        success = bool(getattr(scipy_result, "success", False))
        status = int(getattr(scipy_result, "status", 0) or 0)
        message = str(getattr(scipy_result, "message", ""))
        if nfev >= int(maxfun):
            terminal_reason = "fg_budget_exhausted"
            budget_exhausted = True
            success = False
        elif accepted_iterations >= int(maxiter):
            terminal_reason = "accepted_iteration_guard"
            budget_exhausted = True
            success = False
        elif success:
            message_upper = message.upper()
            if canonical_max_abs <= float(canonical_gtol):
                terminal_reason = "canonical_gtol"
            elif "REL_REDUCTION" in message_upper or "FUNCTION" in message_upper:
                terminal_reason = "ftol_numeric_stop"
            else:
                terminal_reason = "solver_reported_success"
            budget_exhausted = False
        else:
            terminal_reason = "solver_reported_nonconvergence"
            budget_exhausted = False

    elapsed = time.perf_counter() - started
    if not history or not np.isclose(float(history[-1]["fun"]), float(endpoint_value), rtol=0.0, atol=0.0):
        history.append({
            "iteration": int(accepted_iterations),
            "nfev": int(nfev),
            "elapsed_seconds": elapsed,
            "fun": float(endpoint_value),
            "p": float(endpoint_components["p"]),
            "canonical_gradient_max_abs": float(canonical_max_abs),
            "canonical_gradient_norm": float(canonical_norm),
            "components": dict(endpoint_components),
            "state_status": "accepted_endpoint_readback",
        })
    return BudgetedFitResult(
        objective=objective,
        theta=endpoint_theta,
        y=(objective.raw_unpack(endpoint_theta)[0].copy() if isinstance(objective, M1Objective)
           else objective.unpack(endpoint_theta)[0].copy()),
        coordinates=np.asarray(endpoint_coordinates, dtype=np.float64).copy(),
        p=float(endpoint_p),
        fun=float(endpoint_value),
        components=dict(endpoint_components),
        success=bool(success),
        status=int(status),
        message=message,
        terminal_reason=terminal_reason,
        budget_exhausted=bool(budget_exhausted),
        nit=int(accepted_iterations),
        nfev=int(nfev),
        actual_nfev=int(nfev),
        scipy_nfev=int(scipy_nfev),
        njev=int(scipy_njev),
        elapsed_seconds=float(elapsed),
        history=history,
        canonical_gradient_max_abs=float(canonical_max_abs),
        canonical_gradient_norm=float(canonical_norm),
        initial_total=float(initial_total),
        initial_components=dict(initial_components),
        validation_calls=int(validation_calls),
        endpoint_was_last_accepted=bool(endpoint_was_last_accepted),
        raw_scipy_result=scipy_result,
    )


__all__ = [
    "CanonicalGTolReached",
    "FGBudgetExceeded",
    "M1Objective",
    "M1_SCALE",
    "SQRT2",
    "BudgetedFitResult",
    "common_difference_gradient_to_raw",
    "common_difference_to_raw",
    "orthogonal_ab_to_raw",
    "orthogonal_raw_to_ab",
    "raw_gradient_to_common_difference",
    "raw_to_common_difference",
    "run_budgeted_lbfgs",
]
