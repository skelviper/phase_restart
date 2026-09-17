"""058 B：独立 fixed-x p profile 与条件触发的配对 joint 优化。"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
S056 = ROOT / "test_res/056-20260916T152353Z-pro-review-experiments"
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
for path in (HERE, ROOT, S045):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import data_io
from frozen_core import FixedXPCache
from pr import contact_model
from pr.solver_state import (export_present_3dg, load_solver_state, sha256_file,
                             write_presence_mask, write_solver_state)
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective

CONFIG = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
WEIGHTS = PenaltyWeights(count=1.0, bond=1.0, repulsion=1.0, bend=0.01, p_prior=1.0)
P_LO = 0.00010000000000000002
P_HI = 0.9998999999999998


class BudgetStop(RuntimeError):
    pass


class FullConverged(RuntimeError):
    pass


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def make_objective(data):
    return SharedCaptureObjective(data, "G", weights=WEIGHTS,
                                  mode="V0-fixed-production-e", device="cuda",
                                  pair_block=262_144, inner_cap=80, cg_cap=80,
                                  profile_warm_start=True, known_e=None)


def build_p_cache(data, coordinates, trusted_components):
    started = time.perf_counter()
    obj = make_objective(data)
    raw = torch.as_tensor(contact_model.sphere_inverse(coordinates), dtype=torch.float64,
                          device="cuda")
    x = obj._physics._map_raw(raw)
    e = obj._fixed_e
    cis_a = []; cis_b = []; cis_counts = []
    sum_b = torch.zeros((), dtype=torch.float64, device="cuda")
    sum_delta = torch.zeros((), dtype=torch.float64, device="cuda")
    z_inter = torch.zeros((), dtype=torch.float64, device="cuda")
    inter_obs_log = torch.zeros((), dtype=torch.float64, device="cuda")
    for start in range(0, data.n_pairs, obj.pair_block):
        stop = min(start + obj.pair_block, data.n_pairs)
        i = obj._pair_i[start:stop]; j = obj._pair_j[start:stop]
        local_cis = obj._cis[start:stop]
        counts = obj._counts[start:stop]
        kernels, _ = obj._kernel_block(x, i, j,
                                       torch.as_tensor(0.5, dtype=torch.float64, device="cuda"),
                                       with_gradient=False)
        kaa, kab, kba, kbb = kernels[:4]
        eprod = e[i] * e[j]
        a = eprod * 0.5 * (kaa + kbb)
        b = eprod * 0.5 * (kab + kba)
        if bool(torch.any(local_cis)):
            aa = a[local_cis]; bb = b[local_cis]; cc = counts[local_cis]
            sum_b = sum_b + bb.sum(); sum_delta = sum_delta + (aa - bb).sum()
            observed = cc > 0
            if bool(torch.any(observed)):
                cis_a.append(aa[observed].detach().cpu().numpy())
                cis_b.append(bb[observed].detach().cpu().numpy())
                cis_counts.append(cc[observed].detach().cpu().numpy())
        if bool(torch.any(~local_cis)):
            rate = eprod[~local_cis] * 0.25 * (
                kaa[~local_cis] + kab[~local_cis] + kba[~local_cis] + kbb[~local_cis])
            cc = counts[~local_cis]
            z_inter = z_inter + rate.sum()
            observed = cc > 0
            if bool(torch.any(observed)):
                inter_obs_log = inter_obs_log + (cc[observed] * torch.log(rate[observed])).sum()
    obj.synchronize()
    n_cis = float(data.raw_cis_offdiag); n_inter = float(data.raw_inter); n_off = n_cis + n_inter
    k0 = n_cis * math.log(n_cis / n_off) + n_inter * math.log(n_inter / n_off)
    cache = FixedXPCache(
        cis_a_observed=np.concatenate(cis_a), cis_b_observed=np.concatenate(cis_b),
        cis_counts_observed=np.concatenate(cis_counts),
        sum_b_all_cis=float(sum_b.item()), sum_delta_all_cis=float(sum_delta.item()),
        z_inter=float(z_inter.item()), sum_Cinter_log_rate=float(inter_obs_log.item()),
        n_off=n_off, n_raw=float(data.raw_records),
        count_constant_raw=k0 + float(trusted_components["diag_profiled_nll_raw"]),
        physical_constant=(float(trusted_components["weighted_bond"])
                           + float(trusted_components["weighted_repulsion"])
                           + float(trusted_components["weighted_bend"])))
    return cache, {"elapsed_seconds": time.perf_counter() - started,
                   "observed_cis_pairs": int(len(cache.cis_counts_observed)),
                   "equivalents": 1}


def profile_p(cache, original_p):
    calls = {}
    def evaluate(p):
        p = float(p); key = p.hex()
        if key not in calls:
            if len(calls) >= 128:
                raise RuntimeError("p profile scalar evaluation cap exhausted")
            value, derivative, components = cache.value_derivative(p)
            calls[key] = {"p": p, "value": value, "derivative": derivative,
                          "components": components}
        return calls[key]
    original = evaluate(original_p)
    grid = np.linspace(P_LO, P_HI, 33)
    grid_rows = [evaluate(float(p)) for p in grid]
    best_grid_i = min(range(len(grid_rows)), key=lambda i: (grid_rows[i]["value"], i))
    lo_i = max(0, best_grid_i - 1); hi_i = min(len(grid) - 1, best_grid_i + 1)
    if lo_i == hi_i:
        lo_i, hi_i = (0, 1) if best_grid_i == 0 else (len(grid) - 2, len(grid) - 1)
    result = minimize_scalar(lambda p: evaluate(float(p))["value"], bounds=(float(grid[lo_i]), float(grid[hi_i])),
                             method="bounded", options={"xatol": 1e-10, "maxiter": 80})
    bounded = evaluate(float(result.x))
    candidates = [original, *grid_rows, bounded]
    best = min(candidates, key=lambda row: (row["value"],
                                             0 if row["p"] == original_p else 1,
                                             row["p"]))
    # Explicit diagnostics are cache hits and do not increment.
    before = evaluate(original_p); after = evaluate(best["p"])
    return {"original": before, "profiled": after,
            "J_gain": before["value"] - after["value"],
            "count_gain": before["components"]["count"] - after["components"]["count"],
            "bounded": {"success": bool(result.success), "message": str(result.message),
                         "x": float(result.x), "fun": float(result.fun), "nfev": int(result.nfev),
                         "interval": [float(grid[lo_i]), float(grid[hi_i])]},
            "numeric_stop": "bounded_profile_numeric_stop" if result.success else "bounded_profile_not_converged",
            "scalar_unique_calls": len(calls), "scalar_cap": 128,
            "all_calls": sorted(calls.values(), key=lambda row: row["p"])}


def run_full_segment(objective, theta0, maxfun=50):
    theta0 = np.asarray(theta0, dtype=np.float64)
    nfev = 0; nit = 0; last_trial = None; accepted = None; history = []
    started = time.perf_counter(); stop = None; scipy_result = None
    def evaluate(theta):
        nonlocal nfev, last_trial, accepted
        if nfev >= maxfun:
            raise BudgetStop("fg_budget_exhausted_before_next_objective_call")
        nfev += 1
        theta = np.asarray(theta, dtype=np.float64)
        value, gradient, components = objective.evaluate(theta, need_gradient=True)
        gradient = np.asarray(gradient, dtype=np.float64)
        if not math.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite full B value/gradient")
        last_trial = {"theta": theta.copy(), "value": float(value),
                      "gradient": gradient.copy(), "components": dict(components)}
        if nfev == 1:
            accepted = {key: (v.copy() if isinstance(v, np.ndarray) else dict(v) if isinstance(v, dict) else v)
                        for key, v in last_trial.items()}
            history.append({"iteration": 0, "nfev": 1, "value": float(value),
                            "full_maxgrad": float(np.max(np.abs(gradient))),
                            "status": "accepted_initial_budgeted"})
            if history[-1]["full_maxgrad"] <= 1e-6:
                raise FullConverged("full_raw_y_q_gtol_at_initial")
        return float(value), gradient.copy()
    def callback(theta):
        nonlocal nit, accepted
        theta = np.asarray(theta, dtype=np.float64)
        if last_trial is None or not np.array_equal(theta, last_trial["theta"]):
            raise RuntimeError("B accepted state is not last budgeted FG")
        nit += 1
        accepted = {key: (v.copy() if isinstance(v, np.ndarray) else dict(v) if isinstance(v, dict) else v)
                    for key, v in last_trial.items()}
        maxgrad = float(np.max(np.abs(accepted["gradient"])))
        history.append({"iteration": nit, "nfev": nfev, "value": accepted["value"],
                        "full_maxgrad": maxgrad, "status": "accepted"})
        if maxgrad <= 1e-6:
            raise FullConverged("full_raw_y_q_gtol")
    try:
        scipy_result = minimize(evaluate, theta0, jac=True, method="L-BFGS-B", callback=callback,
                                options={"maxiter": maxfun + 1, "maxfun": maxfun,
                                         "maxls": 20, "ftol": 0.0, "gtol": 0.0})
    except (BudgetStop, FullConverged) as exc:
        stop = exc
    if accepted is None:
        raise RuntimeError("B segment produced no accepted endpoint")
    maxgrad = float(np.max(np.abs(accepted["gradient"])))
    terminal = ("converged" if isinstance(stop, FullConverged) or maxgrad <= 1e-6 else
                "budget_not_converged" if isinstance(stop, BudgetStop) or nfev >= maxfun else
                "not_converged")
    return {"terminal": terminal, "reason": str(stop) if stop else str(scipy_result.message),
            "nfev": nfev, "nit": nit, "elapsed_seconds": time.perf_counter() - started,
            "history": history, "endpoint": accepted}


def freeze_endpoint(data, endpoint_id, theta, components, gradient, role, seed, metadata):
    raw = np.asarray(theta[:-1], dtype=np.float64).reshape(2, data.n_loci, 3)
    coords = contact_model.sphere_forward(raw)
    q = float(theta[-1]); p = contact_model.p_from_q(q)[0]
    outdir = RUN / "states/B" / endpoint_id
    state = write_solver_state(outdir / "solver_state.npz", coordinates=coords, raw_y=raw,
                               theta=np.asarray(theta), p=p, q=q)
    presence = np.ones((2, data.n_loci), dtype=bool)
    mask = write_presence_mask(outdir / "presence_mask.npz", presence, data.n_loci)
    before = state["sha256"]
    export = export_present_3dg(RUN / "exports/B" / f"{endpoint_id}.3dg", data, coords, presence)
    if before != sha256_file(outdir / "solver_state.npz"):
        raise RuntimeError("B export changed solver state")
    return {"endpoint_id": endpoint_id, "seed": seed, "role": role,
            "p": p, "q": q, "value": float(components["total"]),
            "components": dict(components),
            "full_raw_y_maxgrad": None if gradient is None else float(np.max(np.abs(gradient[:-1]))),
            "q_absgrad": None if gradient is None else float(abs(gradient[-1])),
            "metadata": metadata,
            "artifacts": {"solver_state": state, "presence_mask": mask, "export": export,
                          "state_sha_unchanged_after_export": True}}


def main():
    terminal_path = RUN / "logs/B_terminal.json"
    terminal = {"schema": "p9016-058-B-terminal-v1", "status": "running",
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "reference_opened": False, "phase_opened": False,
                "training_fg": 0, "auxiliary_equivalents": 0, "diagnostics": [], "endpoints": []}
    dump(terminal_path, terminal)
    data = data_io.load_aggregate(S056 / "inputs/G-original_1000000_aggregate.npz")
    sources = {}; diagnostics = {}
    try:
        for seed in (560101, 560102):
            source = load_solver_state(S056 / f"states/experiment3/G-original-seed{seed}/1Mb/solver_state.npz")
            stage = json.loads((S056 / f"stages/G-original-seed{seed}/1Mb.json").read_text())
            components = stage["endpoint"]["components"]
            cache, ledger = build_p_cache(data, source["coordinates"], components)
            terminal["auxiliary_equivalents"] += 1
            diagnostic = profile_p(cache, source["p"])
            if abs(diagnostic["original"]["value"] - float(components["total"])) > 1e-9:
                raise RuntimeError(f"real p-cache original full-J mismatch seed {seed}")
            if abs(diagnostic["original"]["components"]["count"]
                   - float(components["count_nll_normalized"])) > 1e-9:
                raise RuntimeError(f"real p-cache original count mismatch seed {seed}")
            diagnostic.update({"seed": seed, "cache_build": ledger,
                               "source_state_sha256": source["sha256"]})
            terminal["diagnostics"].append(diagnostic)
            diagnostics[seed] = diagnostic; sources[seed] = (source, components, cache)
            dump(RUN / f"results/B/profile_seed{seed}.json", diagnostic)
            dump(terminal_path, terminal)
            print(json.dumps({"event": "B_profile", "seed": seed,
                              "p_before": source["p"], "p_after": diagnostic["profiled"]["p"],
                              "J_gain": diagnostic["J_gain"],
                              "calls": diagnostic["scalar_unique_calls"]}), flush=True)
        trigger = not all(diagnostics[seed]["J_gain"] < 1e-5 for seed in diagnostics)
        terminal["coordinate_triggered"] = trigger
        terminal["branch"] = "paired_3x50FG" if trigger else "fixed_x_descriptive_0FG"
        for seed in (560101, 560102):
            source, source_components, initial_cache = sources[seed]
            if not trigger:
                for role, p in (("Original", source["p"]),
                                ("Profile", diagnostics[seed]["profiled"]["p"])):
                    theta = source["theta"].copy(); theta[-1] = contact_model.q_from_p(p)
                    value, _, comp = initial_cache.value_derivative(p)
                    components = dict(source_components)
                    components.update({"p": p, "p_prior": comp["p_prior"],
                                       "weighted_p_prior": comp["p_prior"],
                                       "count_nll_normalized": comp["count"],
                                       "weighted_count": comp["count"], "total": value,
                                       "sum_rate_cis_offdiag": comp["Zcis"],
                                       "sum_rate_inter": comp["Zinter"],
                                       "sum_rate_offdiag": comp["Zall"]})
                    endpoint_id = f"B-seed{seed}-{role}"
                    frozen = freeze_endpoint(data, endpoint_id, theta, components, None,
                                             role, seed, {"branch": terminal["branch"],
                                                          "coordinate_fg": 0,
                                                          "descriptive_not_3x50_control": True})
                    terminal["endpoints"].append(frozen)
                    dump(RUN / "results/B/endpoints" / f"{endpoint_id}.json", frozen)
            else:
                for role in ("Control", "Profile"):
                    theta = source["theta"].copy()
                    segments = []
                    trusted = source_components
                    for segment in (1, 2, 3):
                        preprofile = None
                        if role == "Profile":
                            if segment == 1:
                                preprofile = diagnostics[seed]
                            else:
                                current_raw = theta[:-1].reshape(2, data.n_loci, 3)
                                cache, ledger = build_p_cache(data, contact_model.sphere_forward(current_raw), trusted)
                                terminal["auxiliary_equivalents"] += 1
                                preprofile = profile_p(cache, contact_model.p_from_q(float(theta[-1]))[0])
                            theta[-1] = contact_model.q_from_p(preprofile["profiled"]["p"])
                        objective = make_objective(data)
                        result = run_full_segment(objective, theta, maxfun=50)
                        terminal["training_fg"] += result["nfev"]
                        endpoint = result.pop("endpoint")
                        theta = endpoint["theta"]
                        trusted = endpoint["components"]
                        segments.append({**result, "preprofile": preprofile,
                                         "endpoint_value": endpoint["value"],
                                         "endpoint_p": trusted["p"],
                                         "full_raw_y_maxgrad": float(np.max(np.abs(endpoint["gradient"][:-1]))),
                                         "q_absgrad": float(abs(endpoint["gradient"][-1]))})
                        del objective
                        torch.cuda.empty_cache()
                    endpoint_id = f"B-seed{seed}-{role}"
                    frozen = freeze_endpoint(data, endpoint_id, theta, trusted, endpoint["gradient"],
                                             role, seed, {"branch": terminal["branch"],
                                                          "segments": segments,
                                                          "coordinate_fg": sum(x["nfev"] for x in segments)})
                    terminal["endpoints"].append(frozen)
                    dump(RUN / "results/B/endpoints" / f"{endpoint_id}.json", frozen)
                    dump(terminal_path, terminal)
                    print(json.dumps({"event": "B_endpoint", "endpoint_id": endpoint_id,
                                      "fg": frozen["metadata"]["coordinate_fg"],
                                      "p": frozen["p"], "value": frozen["value"]}), flush=True)
        if terminal["training_fg"] > 600 or terminal["auxiliary_equivalents"] > 6:
            raise RuntimeError("B budget exceeded")
        terminal.update({"status": "complete",
                         "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        dump(terminal_path, terminal)
        return 0
    except Exception as exc:
        terminal.update({"status": "failure", "error_type": type(exc).__name__, "error": str(exc),
                         "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        dump(terminal_path, terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
