"""连续 V1 joint 目标函数的最小 SciPy L-BFGS 接口。

本模块有意不提供 CLI，也不自动运行真实数据。调用者必须提供已获授权的无 phase 初始化。
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .contact_model import (AggregatedContacts, JointObjective, ObjectiveWeights,
                            assert_inside_unit_ball)


@dataclass
class JointFitResult:
    """仅保存数值结果；不作生物学或 reference 侧解释。"""

    objective: JointObjective
    theta: np.ndarray
    y: np.ndarray
    coordinates: np.ndarray
    p: float
    fun: float
    components: dict
    success: bool
    status: int
    message: str
    nit: int
    nfev: int
    actual_nfev: int
    scipy_nfev: int
    njev: int
    elapsed_seconds: float
    history: list[dict]


@dataclass(frozen=True)
class JointCheckpoint:
    """供调用者持久化 hook 使用的可安全复制的周期快照。"""

    iteration: int
    nfev: int
    elapsed_seconds: float
    fun: float
    p: float
    theta: np.ndarray
    y: np.ndarray
    coordinates: np.ndarray
    components: dict
    gradient_norm: float = 0.0


def random_initial_y(data: AggregatedContacts, seed: int = 0,
                     scale: float = 0.05) -> np.ndarray:
    """为 synthetic checks 创建有界尺度的技术初始化。"""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return np.random.default_rng(seed).normal(
        0.0, scale, size=(2, data.n_loci, 3)).astype(np.float64)


def fit_joint(objective: JointObjective, initial_y: np.ndarray, p_init: float = 0.75,
              q_init: float | None = None, maxiter: int = 50, maxfun: int | None = None,
              maxls: int = 20, ftol: float = 1e-12, gtol: float = 1e-7,
              callback: Callable[[dict], None] | None = None,
              checkpoint_every: int | None = None,
              checkpoint_hook: Callable[[JointCheckpoint], None] | None = None) -> JointFitResult:
    """运行有界坐标的 L-BFGS-B 数值优化。

    坐标变量本身无约束，因为 ``JointObjective`` 应用 sphere transform。``maxiter`` 默认是短 prototype run，不是正式 P9016 fit budget。若可用，``q_init`` 携带上一层精确的 bounded-logistic 坐标，避免重新求逆已饱和的浮点 ``p``。``checkpoint_hook`` 由调用者持有，本身从不写文件；若未提供 interval，则每 10 个接受的 iteration 接收一个可安全复制的快照。"""
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    if maxfun is not None and maxfun <= 0:
        raise ValueError("maxfun must be positive when supplied")
    if maxls <= 0:
        raise ValueError("maxls must be positive")
    if q_init is not None and not np.isfinite(q_init):
        raise ValueError("q_init must be finite when supplied")
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive when supplied")
    if checkpoint_hook is not None and checkpoint_every is None:
        checkpoint_every = 10
    started = time.perf_counter()
    theta0 = objective.pack(initial_y, p=p_init)
    if q_init is not None:
        theta0[-1] = float(q_init)
    initial_value, _, initial_components = objective.evaluate(theta0, need_gradient=False)
    history = [{"iteration": 0, "nfev": 0, "elapsed_seconds": time.perf_counter() - started,
                "fun": float(initial_value), "p": float(initial_components["p"]),
                "components": initial_components}]
    actual_nfev = 0
    last_gradient = None
    last_theta = None

    def loss_and_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal actual_nfev, last_gradient, last_theta
        actual_nfev += 1
        value, gradient = objective.value_and_grad(theta)
        last_theta = np.asarray(theta, dtype=np.float64).copy()
        last_gradient = np.asarray(gradient, dtype=np.float64).copy()
        return value, gradient

    def recorder(theta: np.ndarray) -> None:
        cached = objective.cached_value_and_components(theta)
        if cached is None:
            value, _, components = objective.evaluate(theta, need_gradient=False)
        else:
            value, components = cached
        if (last_theta is not None and last_gradient is not None
                and np.array_equal(np.asarray(theta, dtype=np.float64), last_theta)):
            gradient_norm = float(np.linalg.norm(last_gradient))
        else:
            _, gradient, _ = objective.evaluate(theta, need_gradient=True)
            gradient_norm = float(np.linalg.norm(gradient))
        iteration = len(history)
        elapsed = time.perf_counter() - started
        entry = {"iteration": iteration, "nfev": actual_nfev,
                 "elapsed_seconds": elapsed, "fun": float(value),
                 "p": float(components["p"]), "gradient_norm": gradient_norm,
                 "components": components}
        history.append(entry)
        if checkpoint_hook is not None and iteration % checkpoint_every == 0:
            y, _ = objective.unpack(theta)
            coordinates, p = objective.coordinates_and_p(theta)
            checkpoint_hook(JointCheckpoint(
                iteration=iteration,
                nfev=actual_nfev,
                elapsed_seconds=elapsed,
                fun=float(value),
                p=float(p),
                theta=np.asarray(theta, dtype=np.float64).copy(),
                y=y.copy(),
                coordinates=coordinates.copy(),
                components=dict(components),
                gradient_norm=gradient_norm,
            ))
        if callback is not None:
            callback(entry)

    options = {"maxiter": int(maxiter), "maxls": int(maxls),
               "ftol": float(ftol), "gtol": float(gtol)}
    if maxfun is not None:
        options["maxfun"] = int(maxfun)
    raw: OptimizeResult = minimize(
        loss_and_gradient,
        theta0,
        method="L-BFGS-B",
        jac=True,
        callback=recorder,
        options=options,
    )
    y, _ = objective.unpack(raw.x)
    coordinates, p = objective.coordinates_and_p(raw.x)
    assert_inside_unit_ball(coordinates)
    cached = objective.cached_value_and_components(raw.x)
    if cached is None:
        final_value, _, components = objective.evaluate(raw.x, need_gradient=False)
    else:
        final_value, components = cached
    elapsed = time.perf_counter() - started
    if not history or history[-1]["fun"] != float(final_value):
        history.append({"iteration": len(history), "nfev": actual_nfev,
                        "elapsed_seconds": elapsed, "fun": float(final_value),
                        "p": float(p), "components": components})
    scipy_nfev = int(getattr(raw, "nfev", actual_nfev))
    return JointFitResult(
        objective=objective,
        theta=np.asarray(raw.x, dtype=np.float64).copy(),
        y=y.copy(),
        coordinates=coordinates.copy(),
        p=float(p),
        fun=float(final_value),
        components=components,
        success=bool(raw.success),
        status=int(raw.status),
        message=str(raw.message),
        nit=int(raw.nit),
        nfev=int(actual_nfev),
        actual_nfev=int(actual_nfev),
        scipy_nfev=scipy_nfev,
        njev=int(raw.njev) if raw.njev is not None else 0,
        elapsed_seconds=elapsed,
        history=history,
    )


def fit_from_data(data: AggregatedContacts, initial_y: np.ndarray, p_init: float = 0.75,
                   q_init: float | None = None,
                  weights: ObjectiveWeights | None = None, block_size: int = 65_536,
                  repulsion_block_size: int | None = None, maxiter: int = 50,
                   maxfun: int | None = None, maxls: int = 20,
                  ftol: float = 1e-12, gtol: float = 1e-7,
                  callback: Callable[[dict], None] | None = None,
                  checkpoint_every: int | None = None,
                  checkpoint_hook: Callable[[JointCheckpoint], None] | None = None) -> JointFitResult:
    """构建 objective 并运行 :func:`fit_joint`，不写文件。"""
    objective = JointObjective(
        data,
        weights=weights,
        block_size=block_size,
        repulsion_block_size=repulsion_block_size,
    )
    return fit_joint(objective, initial_y, p_init=p_init, q_init=q_init, maxiter=maxiter,
                      maxfun=maxfun, maxls=maxls,
                     ftol=ftol, gtol=gtol, callback=callback,
                     checkpoint_every=checkpoint_every, checkpoint_hook=checkpoint_hook)
