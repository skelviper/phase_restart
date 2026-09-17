"""058 冻结接口的纯 CPU fixture；不加载 P9016 正式 arrays。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from frozen_core import FixedXPCache, RestrictedObjective, swap_complete_interval


class QuadraticFullObjective:
    def __init__(self, center):
        self.center = np.asarray(center, dtype=np.float64)

    def evaluate(self, theta, need_gradient=True):
        theta = np.asarray(theta, dtype=np.float64)
        delta = theta - self.center
        value = 0.5 * float(np.dot(delta, delta))
        return value, delta.copy(), {"total": value, "p": 0.24}


def main():
    rng = np.random.default_rng(580001)
    coordinates = rng.normal(size=(2, 7, 3)) * 0.05
    raw_y = rng.normal(size=(2, 7, 3)) * 0.1
    changed_x, changed_y = swap_complete_interval(coordinates, raw_y, 2, 5)

    theta = np.concatenate((raw_y.ravel(), np.array([-1.2])))
    active = np.array([0, 1, 6, 7, 18, 19, 24, 25], dtype=np.int64)
    center = rng.normal(size=len(theta)) * 0.2
    wrapped = RestrictedObjective(QuadraticFullObjective(center), theta, active)
    y0 = wrapped.initial_active()
    value, gradient = wrapped.value_and_grad(y0)
    eps = 1e-6
    fd = np.empty_like(gradient)
    for k in range(len(y0)):
        yp = y0.copy(); ym = y0.copy()
        yp[k] += eps; ym[k] -= eps
        vp = wrapped.full_objective.evaluate(wrapped.full_theta(yp), need_gradient=False)[0]
        vm = wrapped.full_objective.evaluate(wrapped.full_theta(ym), need_gradient=False)[0]
        fd[k] = (vp - vm) / (2.0 * eps)
    fd_error = float(np.max(np.abs(fd - gradient)))
    inactive = np.ones(len(theta), dtype=bool); inactive[active] = False
    frozen_error = float(np.max(np.abs(wrapped.full_theta(y0)[inactive] - theta[inactive])))

    cache = FixedXPCache.from_pair_arrays(
        cis_a=np.array([0.8, 1.4, 0.6, 1.1]),
        cis_b=np.array([1.2, 0.7, 1.5, 0.9]),
        cis_counts=np.array([3.0, 1.0, 4.0, 2.0]),
        inter_rates=np.array([0.5, 0.9, 1.3]),
        inter_counts=np.array([2.0, 0.0, 5.0]),
        n_raw=25.0,
        count_constant_raw=1.7,
        physical_constant=0.03,
    )
    p = 0.24
    p_value, analytic, components = cache.value_derivative(p)
    vp = cache.value_derivative(p + eps)[0]
    vm = cache.value_derivative(p - eps)[0]
    p_fd = (vp - vm) / (2.0 * eps)
    p_fd_error = float(abs(p_fd - analytic))

    checks = {
        "active_fd_max_abs_error": fd_error,
        "active_inactive_freeze_max_abs_error": frozen_error,
        "active_cache_exact_hit": wrapped.cached_full(y0) is not None,
        "active_cache_rejects_other": wrapped.cached_full(y0 + eps) is None,
        "swap_xyz_exact": bool(np.array_equal(changed_x[:, 2:5], coordinates[::-1, 2:5])),
        "swap_raw_y_exact": bool(np.array_equal(changed_y[:, 2:5], raw_y[::-1, 2:5])),
        "p_derivative_fd_abs_error": p_fd_error,
        "p_value": p_value,
        "p_components": components,
    }
    checks["passed"] = bool(
        fd_error <= 1e-8 and frozen_error == 0.0
        and checks["active_cache_exact_hit"] and checks["active_cache_rejects_other"]
        and checks["swap_xyz_exact"] and checks["swap_raw_y_exact"]
        and p_fd_error <= 1e-8
    )
    out = HERE.parent / "fixture_results.json"
    out.write_text(json.dumps(checks, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(checks, sort_keys=True))
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
