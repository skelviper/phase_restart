"""045 raw L-BFGS的056预算口径适配：不做预算外full-grid init/readback。"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize

ROOT = Path(__file__).resolve().parents[3]
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
if str(S045) not in sys.path:
    sys.path.insert(0, str(S045))

from m1_preconditioner import (BudgetedFitResult, CanonicalGTolReached,
                               FGBudgetExceeded, M1Objective,
                               _canonical_gradient, _canonical_metrics)


def run_budgeted_lbfgs_no_extra(
    objective: Any,
    initial_y: np.ndarray,
    *,
    p_init: float,
    q_init: float,
    maxfun: int,
    maxiter: int,
    maxls: int,
    ftol: float,
    canonical_gtol: float,
    checkpoint_every: int = 10,
    checkpoint_hook: Callable[[dict[str, Any]], None] | None = None,
) -> BudgetedFitResult:
    """保持045 SciPy算法与参数，但全部full-grid调用都计入maxfun。"""
    initial_y = np.asarray(initial_y, dtype=np.float64)
    if initial_y.shape != (2, objective.data.n_loci, 3) or not np.isfinite(initial_y).all():
        raise ValueError("initial_y must be finite with shape (2,n_loci,3)")
    theta0 = np.asarray(objective.pack(initial_y, p=float(p_init)), dtype=np.float64)
    theta0[-1] = float(q_init)
    started = time.perf_counter()
    nfev = 0
    njev = 0
    accepted_iterations = 0
    accepted_theta = theta0.copy()
    accepted_value: float | None = None
    accepted_components: dict[str, Any] | None = None
    accepted_gradient: np.ndarray | None = None
    initial_total: float | None = None
    initial_components: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    last_eval_theta = None
    last_eval_gradient = None
    last_eval_value = None
    last_eval_components = None
    stop_exception: Exception | None = None

    def cache_components(theta, value):
        cached = objective.cached_value_and_components(theta)
        if cached is None:
            raise RuntimeError("objective did not cache components for current evaluated theta")
        cached_value, components = cached
        if float(cached_value) != float(value):
            raise RuntimeError("cached value differs from evaluated value")
        return dict(components)

    def loss_and_gradient(theta):
        nonlocal nfev, njev, last_eval_theta, last_eval_gradient, last_eval_value
        nonlocal last_eval_components, accepted_value, accepted_components, accepted_gradient
        nonlocal initial_total, initial_components
        if nfev >= int(maxfun):
            raise FGBudgetExceeded("fg_budget_exhausted_before_next_objective_call")
        nfev += 1
        njev += 1
        theta = np.asarray(theta, dtype=np.float64)
        value, gradient = objective.value_and_grad(theta)
        gradient = np.asarray(gradient, dtype=np.float64)
        if not math.isfinite(float(value)) or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite value or gradient from objective")
        components = cache_components(theta, value)
        last_eval_theta = theta.copy()
        last_eval_gradient = gradient.copy()
        last_eval_value = float(value)
        last_eval_components = components
        if nfev == 1:
            initial_total = float(value)
            initial_components = dict(components)
            accepted_value = float(value)
            accepted_components = dict(components)
            accepted_gradient = gradient.copy()
            canonical = _canonical_gradient(objective, theta, gradient)
            max_abs, norm = _canonical_metrics(canonical)
            history.append({
                "iteration": 0, "nfev": 1, "elapsed_seconds": time.perf_counter() - started,
                "fun": float(value), "p": float(components["p"]),
                "canonical_gradient_max_abs": max_abs, "canonical_gradient_norm": norm,
                "components": dict(components), "state_status": "accepted_initial_budgeted",
            })
            if checkpoint_hook is not None:
                checkpoint_hook({
                    "iteration": 0, "nfev": 1, "elapsed_seconds": history[-1]["elapsed_seconds"],
                    "fun": float(value), "p": float(components["p"]), "theta": theta.copy(),
                    "raw_y": objective.unpack(theta)[0].copy(),
                    "coordinates": objective.coordinates_and_p(theta)[0].copy(),
                    "components": dict(components), "optimizer_gradient": gradient.copy(),
                    "canonical_gradient": canonical.copy(),
                })
            if max_abs <= float(canonical_gtol):
                raise CanonicalGTolReached("canonical_raw_y_q_gtol_reached_at_initial")
        return float(value), gradient.copy()

    def recorder(theta):
        nonlocal accepted_theta, accepted_iterations, accepted_value, accepted_components, accepted_gradient
        theta = np.asarray(theta, dtype=np.float64).copy()
        if last_eval_theta is None or not np.array_equal(theta, last_eval_theta):
            raise RuntimeError("SciPy accepted state is not the last budgeted objective evaluation")
        accepted_theta = theta
        accepted_iterations += 1
        accepted_value = float(last_eval_value)
        accepted_components = dict(last_eval_components)
        accepted_gradient = np.asarray(last_eval_gradient, dtype=np.float64).copy()
        canonical = _canonical_gradient(objective, theta, accepted_gradient)
        max_abs, norm = _canonical_metrics(canonical)
        entry = {
            "iteration": accepted_iterations, "nfev": nfev,
            "elapsed_seconds": time.perf_counter() - started,
            "fun": accepted_value, "p": float(accepted_components["p"]),
            "canonical_gradient_max_abs": max_abs, "canonical_gradient_norm": norm,
            "components": dict(accepted_components), "state_status": "accepted",
        }
        history.append(entry)
        if checkpoint_hook is not None and accepted_iterations % int(checkpoint_every) == 0:
            checkpoint_hook({
                "iteration": accepted_iterations, "nfev": nfev,
                "elapsed_seconds": entry["elapsed_seconds"], "fun": accepted_value,
                "p": float(accepted_components["p"]), "theta": theta.copy(),
                "raw_y": objective.unpack(theta)[0].copy(),
                "coordinates": objective.coordinates_and_p(theta)[0].copy(),
                "components": dict(accepted_components),
                "optimizer_gradient": accepted_gradient.copy(),
                "canonical_gradient": canonical.copy(),
            })
        if max_abs <= float(canonical_gtol):
            raise CanonicalGTolReached("canonical_raw_y_q_gtol_reached")

    scipy_result: OptimizeResult | None = None
    try:
        scipy_result = minimize(
            loss_and_gradient, theta0, method="L-BFGS-B", jac=True, callback=recorder,
            options={"maxiter": int(maxiter), "maxfun": int(maxfun), "maxls": int(maxls),
                     "ftol": float(ftol), "gtol": 0.0},
        )
    except (FGBudgetExceeded, CanonicalGTolReached) as error:
        stop_exception = error
    if accepted_value is None or accepted_components is None or accepted_gradient is None:
        raise RuntimeError("solver produced no budgeted finite initial evaluation")
    endpoint_canonical = _canonical_gradient(objective, accepted_theta, accepted_gradient)
    canonical_max_abs, canonical_norm = _canonical_metrics(endpoint_canonical)
    endpoint_coordinates, endpoint_p = objective.coordinates_and_p(accepted_theta)
    if stop_exception is not None:
        if isinstance(stop_exception, FGBudgetExceeded):
            success, status, terminal_reason, budget_exhausted = False, 1, "fg_budget_exhausted", True
        else:
            success, status, terminal_reason, budget_exhausted = True, 0, "canonical_gtol", False
        message = str(stop_exception)
    else:
        if scipy_result is None:
            raise RuntimeError("SciPy result missing")
        success = bool(scipy_result.success)
        status = int(scipy_result.status or 0)
        message = str(scipy_result.message)
        if nfev >= int(maxfun):
            terminal_reason, budget_exhausted, success = "fg_budget_exhausted", True, False
        elif accepted_iterations >= int(maxiter):
            terminal_reason, budget_exhausted, success = "accepted_iteration_guard", True, False
        elif success and canonical_max_abs <= float(canonical_gtol):
            terminal_reason, budget_exhausted = "canonical_gtol", False
        elif success and ("REL_REDUCTION" in message.upper() or "FUNCTION" in message.upper()):
            terminal_reason, budget_exhausted = "ftol_numeric_stop", False
        elif success:
            terminal_reason, budget_exhausted = "solver_reported_success", False
        else:
            terminal_reason, budget_exhausted = "solver_reported_nonconvergence", False
    return BudgetedFitResult(
        objective=objective, theta=accepted_theta.copy(),
        y=objective.unpack(accepted_theta)[0].copy(),
        coordinates=np.asarray(endpoint_coordinates, dtype=np.float64).copy(),
        p=float(endpoint_p), fun=float(accepted_value), components=dict(accepted_components),
        success=bool(success), status=int(status), message=message,
        terminal_reason=terminal_reason, budget_exhausted=bool(budget_exhausted),
        nit=int(accepted_iterations), nfev=int(nfev), actual_nfev=int(nfev),
        scipy_nfev=int(nfev), njev=int(njev), elapsed_seconds=time.perf_counter() - started,
        history=history, canonical_gradient_max_abs=float(canonical_max_abs),
        canonical_gradient_norm=float(canonical_norm), initial_total=float(initial_total),
        initial_components=dict(initial_components), validation_calls=0,
        endpoint_was_last_accepted=True, raw_scipy_result=scipy_result,
    )


__all__ = ["run_budgeted_lbfgs_no_extra"]
