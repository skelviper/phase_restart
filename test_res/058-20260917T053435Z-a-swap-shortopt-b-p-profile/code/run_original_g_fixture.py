"""原 SharedCaptureObjective 的双染色体 CPU fixture。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
for path in (HERE, ROOT, S045):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frozen_core import FixedXPCache, RestrictedObjective
from pr import contact_model
from pr.solver_state import (export_present_3dg, load_solver_state, sha256_file,
                             write_presence_mask, write_solver_state)
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective


def make_data():
    names = ("chr1", "chr2")
    lengths = (3_000_000, 2_000_000)
    # diag、cis offdiag、inter 都存在，并含重复记录形成正计数权重。
    ci = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1], dtype=np.int64)
    p1 = np.array([100, 100, 100, 1_000_100, 1_000_100, 2_000_100,
                   100, 100, 1_000_100, 100, 2_000_100, 100, 1_000_100, 1_000_100, 100])
    cj = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0], dtype=np.int64)
    p2 = np.array([100, 1_000_100, 1_000_100, 2_000_100, 2_000_100, 2_000_100,
                   100, 1_000_100, 1_000_100, 100, 1_000_100, 2_000_100, 1_000_100, 100, 100])
    return contact_model.aggregate_from_arrays(names, lengths, ci, p1, cj, p2, 1_000_000)


def make_objective(data):
    return SharedCaptureObjective(
        data, "G", weights=PenaltyWeights(count=1.0, bond=1.0, repulsion=1.0,
                                            bend=0.01, p_prior=1.0),
        mode="V0-fixed-production-e", device="cpu", pair_block=64,
        inner_cap=20, cg_cap=20, profile_warm_start=True, known_e=None)


def build_numpy_cache(data, coordinates, components):
    i = np.asarray(data.pair_i, dtype=np.int64)
    j = np.asarray(data.pair_j, dtype=np.int64)
    eprod = np.asarray(data.exposure)[i] * np.asarray(data.exposure)[j]
    kaa = contact_model.bounded_kernel(coordinates[0, i] - coordinates[0, j], data.r0)
    kab = contact_model.bounded_kernel(coordinates[0, i] - coordinates[1, j], data.r0)
    kba = contact_model.bounded_kernel(coordinates[1, i] - coordinates[0, j], data.r0)
    kbb = contact_model.bounded_kernel(coordinates[1, i] - coordinates[1, j], data.r0)
    cis = np.asarray(data.cis_pair, dtype=bool)
    a = eprod[cis] * 0.5 * (kaa[cis] + kbb[cis])
    b = eprod[cis] * 0.5 * (kab[cis] + kba[cis])
    inter_rate = eprod[~cis] * 0.25 * (kaa[~cis] + kab[~cis] + kba[~cis] + kbb[~cis])
    n_cis = float(data.raw_cis_offdiag)
    n_inter = float(data.raw_inter)
    n_off = n_cis + n_inter
    k0 = n_cis * math.log(n_cis / n_off) + n_inter * math.log(n_inter / n_off)
    count_constant_raw = k0 + float(components["diag_profiled_nll_raw"])
    physical_constant = (float(components["weighted_bond"])
                         + float(components["weighted_repulsion"])
                         + float(components["weighted_bend"]))
    return FixedXPCache.from_pair_arrays(
        cis_a=a, cis_b=b, cis_counts=np.asarray(data.counts)[cis],
        inter_rates=inter_rate, inter_counts=np.asarray(data.counts)[~cis],
        n_raw=float(data.raw_records), count_constant_raw=count_constant_raw,
        physical_constant=physical_constant)


def main():
    data = make_data()
    rng = np.random.default_rng(580002)
    raw_y = rng.normal(0.0, 0.18, size=(2, data.n_loci, 3))
    coordinates = contact_model.sphere_forward(raw_y)
    p0 = 0.24
    q0 = contact_model.q_from_p(p0)
    theta = np.concatenate((raw_y.ravel(), np.array([q0])))
    objective = make_objective(data)
    full0, gradient0, components0 = objective.evaluate(theta, need_gradient=True)
    cache = build_numpy_cache(data, coordinates, components0)

    p_errors = {}
    for p in (p0, 0.61):
        q = contact_model.q_from_p(p)
        local_theta = theta.copy(); local_theta[-1] = q
        expected, _, expected_components = objective.evaluate(local_theta, need_gradient=False)
        actual, derivative, actual_components = cache.value_derivative(p)
        p_errors[str(p)] = {
            "full_value": float(expected), "cache_value": float(actual),
            "value_abs_error": float(abs(expected - actual)),
            "full_count": float(expected_components["count_nll_normalized"]),
            "cache_count": float(actual_components["count"]),
            "count_abs_error": float(abs(expected_components["count_nll_normalized"]
                                         - actual_components["count"])),
            "dJ_dp": float(derivative), "dJ_dq": float(actual_components["dJ_dq"]),
        }

    chr0 = data.chromosome_slice(0)
    active_mask = np.zeros(raw_y.shape, dtype=bool)
    active_mask[:, chr0, :] = True
    active_indices = np.flatnonzero(active_mask.ravel())
    wrapped = RestrictedObjective(objective, theta, active_indices)
    active0 = wrapped.initial_active()
    wrapped_value, wrapped_gradient = wrapped.value_and_grad(active0)
    eps = 1e-6
    fd = np.empty_like(wrapped_gradient)
    for k in range(len(active0)):
        plus = active0.copy(); minus = active0.copy()
        plus[k] += eps; minus[k] -= eps
        vp = objective.evaluate(wrapped.full_theta(plus), need_gradient=False)[0]
        vm = objective.evaluate(wrapped.full_theta(minus), need_gradient=False)[0]
        fd[k] = (vp - vm) / (2.0 * eps)
    active_fd_error = float(np.max(np.abs(fd - wrapped_gradient)))
    inactive = np.ones(len(theta), dtype=bool); inactive[active_indices] = False
    inactive_error = float(np.max(np.abs(wrapped.full_theta(active0)[inactive] - theta[inactive])))
    q_bitwise_frozen = bool(wrapped.full_theta(active0)[-1].tobytes() == theta[-1].tobytes())

    fixture_dir = RUN / "fixture_original_g"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    state_path = fixture_dir / "solver_state.npz"
    mask_path = fixture_dir / "presence_mask.npz"
    export_path = fixture_dir / "coordinates.3dg"
    for path in (state_path, mask_path, export_path):
        if path.exists():
            path.unlink()
    state = write_solver_state(state_path, coordinates=coordinates, raw_y=raw_y,
                               theta=theta, p=p0, q=q0)
    presence = np.ones((2, data.n_loci), dtype=bool)
    mask = write_presence_mask(mask_path, presence, data.n_loci)
    before = sha256_file(state_path)
    export = export_present_3dg(export_path, data, coordinates, presence)
    after = sha256_file(state_path)
    loaded = load_solver_state(state_path)

    max_p_error = max(row["value_abs_error"] for row in p_errors.values())
    max_count_error = max(row["count_abs_error"] for row in p_errors.values())
    result = {
        "schema": "p9016-058-original-G-cpu-fixture-v1",
        "data": data.budget(),
        "p_cache": p_errors,
        "p_cache_max_full_value_abs_error": max_p_error,
        "p_cache_max_count_abs_error": max_count_error,
        "active_full_value_abs_error": float(abs(wrapped_value - full0)),
        "active_gradient_fd_max_abs_error": active_fd_error,
        "inactive_theta_max_abs_error": inactive_error,
        "q_bitwise_frozen": q_bitwise_frozen,
        "state": state,
        "state_readback_sha256": loaded["sha256"],
        "mask": mask,
        "export": export,
        "export_left_state_sha_unchanged": before == after,
    }
    result["passed"] = bool(
        max_p_error <= 1e-9 and max_count_error <= 1e-9
        and result["active_full_value_abs_error"] <= 1e-12
        and active_fd_error <= 1e-7 and inactive_error == 0.0 and q_bitwise_frozen
        and before == after and loaded["audit"]["finite"])
    out = RUN / "fixture_original_g_results.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "p_cache_max_error": max_p_error,
                      "active_fd_max_error": active_fd_error,
                      "q_bitwise_frozen": q_bitwise_frozen}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
