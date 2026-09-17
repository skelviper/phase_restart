"""058 A：两候选 Control/Swap 的 restricted raw-y 正式短优化。"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
S056 = ROOT / "test_res/056-20260916T152353Z-pro-review-experiments"
S057 = ROOT / "test_res/057-20260917T013859Z-copy-link-dev-validation"
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
for path in (HERE, ROOT, S045):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import data_io
from frozen_core import FrozenMethodError, RestrictedObjective, swap_complete_interval
from pr import contact_model
from pr.solver_state import (export_present_3dg, load_solver_state, sha256_file,
                             write_presence_mask, write_solver_state)
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective

CONFIG = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
WEIGHTS = PenaltyWeights(count=1.0, bond=1.0, repulsion=1.0, bend=0.01, p_prior=1.0)
TOL = 1e-9


class BudgetStop(RuntimeError):
    pass


class ActiveConverged(RuntimeError):
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


def active_indices_for_chromosome(data, chromosome_index):
    raw_shape = (2, data.n_loci, 3)
    mask = np.zeros(raw_shape, dtype=bool)
    slc = data.chromosome_slice(int(chromosome_index))
    mask[:, slc, :] = True
    indices = np.flatnonzero(mask.ravel())
    expected = 2 * (slc.stop - slc.start) * 3
    if len(indices) != expected or len(np.unique(indices)) != expected:
        raise FrozenMethodError("active index count/uniqueness mismatch")
    recovered = np.zeros(raw_shape, dtype=bool)
    recovered.reshape(-1)[indices] = True
    if not np.array_equal(recovered, mask):
        raise FrozenMethodError("active indices are not exactly both target-chromosome xyz blocks")
    return indices


def expected_initial(source_components, candidate, arm):
    keys = ("total", "count_nll_normalized", "bond", "bend", "repulsion", "p_prior")
    expected = {key: float(source_components[key]) for key in keys}
    if arm == "Swap":
        expected["total"] += float(candidate["source_delta_full_J"])
        expected["count_nll_normalized"] += float(candidate["source_delta_count_Nraw"])
        expected["bond"] += float(candidate["source_delta_bond"])
        expected["bend"] += float(candidate["source_delta_bend"])
        expected["repulsion"] += float(candidate.get("source_delta_repulsion", 0.0))
        expected["p_prior"] += float(candidate.get("source_delta_p_prior", 0.0))
    return expected


def run_restricted(objective, base_theta, active_indices, expected, maxfun=100):
    wrapped = RestrictedObjective(objective, base_theta, active_indices)
    x0 = wrapped.initial_active()
    nfev = 0
    nit = 0
    last_trial = None
    accepted = None
    history = []
    initial_errors = None
    started = time.perf_counter()

    def evaluate(active):
        nonlocal nfev, last_trial, accepted, initial_errors
        if nfev >= maxfun:
            raise BudgetStop("fg_budget_exhausted_before_next_objective_call")
        nfev += 1
        value, gradient = wrapped.value_and_grad(active)
        cache = wrapped.cached_full(active)
        if cache is None:
            raise RuntimeError("restricted full cache missing after FG")
        active_gradient = np.asarray(gradient, dtype=np.float64)
        full_gradient = np.asarray(cache["full_gradient"], dtype=np.float64)
        record = {
            "active": np.asarray(active, dtype=np.float64).copy(),
            "full_theta": np.asarray(cache["full_theta"], dtype=np.float64).copy(),
            "value": float(value),
            "active_gradient": active_gradient.copy(),
            "full_gradient": full_gradient.copy(),
            "components": dict(cache["components"]),
        }
        if nfev == 1:
            initial_errors = {key: abs(float(record["components"][key]) - float(want))
                              for key, want in expected.items()}
            if max(initial_errors.values()) >= TOL:
                raise RuntimeError("initial full-G component mismatch: %r" % initial_errors)
            accepted = record
            history.append({"iteration": 0, "nfev": 1, "value": float(value),
                            "active_maxgrad": float(np.max(np.abs(active_gradient))),
                            "full_raw_y_maxgrad": float(np.max(np.abs(full_gradient[:-1]))),
                            "q_absgrad": float(abs(full_gradient[-1])),
                            "status": "accepted_initial_budgeted"})
            if history[-1]["active_maxgrad"] <= 1e-6:
                raise ActiveConverged("active_gtol_at_initial")
        last_trial = record
        return float(value), active_gradient.copy()

    def callback(active):
        nonlocal nit, accepted
        active = np.asarray(active, dtype=np.float64)
        if last_trial is None or not np.array_equal(active, last_trial["active"]):
            raise RuntimeError("SciPy accepted state is not last budgeted FG")
        nit += 1
        accepted = {key: (value.copy() if isinstance(value, np.ndarray) else
                          dict(value) if isinstance(value, dict) else value)
                    for key, value in last_trial.items()}
        active_max = float(np.max(np.abs(accepted["active_gradient"])))
        history.append({"iteration": nit, "nfev": nfev, "value": accepted["value"],
                        "active_maxgrad": active_max,
                        "full_raw_y_maxgrad": float(np.max(np.abs(accepted["full_gradient"][:-1]))),
                        "q_absgrad": float(abs(accepted["full_gradient"][-1])),
                        "status": "accepted"})
        if active_max <= 1e-6:
            raise ActiveConverged("active_gtol")

    result = None
    stop = None
    try:
        result = minimize(evaluate, x0, jac=True, method="L-BFGS-B", callback=callback,
                          options={"maxiter": maxfun + 1, "maxfun": maxfun,
                                   "maxls": 20, "ftol": 0.0, "gtol": 0.0})
    except (BudgetStop, ActiveConverged) as exc:
        stop = exc
    if accepted is None:
        raise RuntimeError("no finite accepted endpoint")
    active_max = float(np.max(np.abs(accepted["active_gradient"])))
    if isinstance(stop, ActiveConverged) or active_max <= 1e-6:
        terminal = "restricted_converged"
    elif isinstance(stop, BudgetStop) or nfev >= maxfun:
        terminal = "budget_not_converged"
    else:
        terminal = "not_converged"
    return {
        "terminal": terminal, "terminal_reason": str(stop) if stop else str(result.message),
        "nfev": nfev, "nit": nit, "elapsed_seconds": time.perf_counter() - started,
        "initial_component_abs_errors": initial_errors, "history": history,
        "endpoint": accepted,
    }


def main():
    terminal_path = RUN / "logs/A_terminal.json"
    terminal = {"schema": "p9016-058-A-terminal-v1", "status": "running",
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "reference_opened": False, "phase_opened": False, "arms": []}
    dump(terminal_path, terminal)
    data = data_io.load_aggregate(S056 / "inputs/G-original_1000000_aggregate.npz")
    round1 = json.loads((S057 / "round1_results.json").read_text(encoding="utf-8"))
    round_by_seed = {int(row["seed"]): row for row in round1["results"]}
    total_fg = 0
    try:
        for seed_text, candidates in CONFIG["A"]["selection"]["candidates"].items():
            seed = int(seed_text)
            source_path = S056 / f"states/experiment3/G-original-seed{seed}/1Mb/solver_state.npz"
            source = load_solver_state(source_path)
            if source["sha256"] != CONFIG["frozen_hashes"][f"source_state_{seed}"]:
                raise RuntimeError(f"source state hash mismatch seed {seed}")
            stage = json.loads((S056 / f"stages/G-original-seed{seed}/1Mb.json").read_text())
            source_components = stage["endpoint"]["components"]
            source_theta = source["theta"]
            for candidate in candidates:
                ci = int(candidate["chromosome_index"])
                slc = data.chromosome_slice(ci)
                g0 = int(candidate["global_locus_start"]); g1 = int(candidate["global_locus_end"])
                if (g0, g1) != (slc.start + int(candidate["bead_start"]),
                                slc.start + int(candidate["bead_end"])):
                    raise RuntimeError("candidate global interval mismatch")
                scan_path = S057 / f"scan_round1_seed{seed}.tsv"
                # Frozen config values came from this hash-locked scan; deltas needed for all components.
                import csv
                scan_rows = list(csv.DictReader(scan_path.open(encoding="utf-8"), delimiter="\t"))
                scan = next(row for row in scan_rows
                            if int(row["candidate_order"]) == int(candidate["candidate_order"]))
                candidate = dict(candidate)
                for key in ("delta_bond", "delta_bend", "delta_repulsion", "delta_p_prior"):
                    candidate["source_" + key] = float(scan[key])
                for arm in ("Control", "Swap"):
                    arm_id = f"A-seed{seed}-candidate{candidate['slot']}-{arm}"
                    coords = source["coordinates"].copy()
                    raw_y = source["raw_y"].copy()
                    if arm == "Swap":
                        coords, raw_y = swap_complete_interval(coords, raw_y, g0, g1)
                        mapped_error = float(np.max(np.abs(contact_model.sphere_forward(raw_y) - coords)))
                        if mapped_error > 1e-12:
                            raise RuntimeError("swap raw-y/xyz mapping mismatch")
                    theta = np.concatenate((raw_y.ravel(), np.array([source["q"]])))
                    if theta[-1].tobytes() != source_theta[-1].tobytes():
                        raise RuntimeError("A q changed before optimization")
                    indices = active_indices_for_chromosome(data, ci)
                    objective = make_objective(data)
                    expected = expected_initial(source_components, candidate, arm)
                    result = run_restricted(objective, theta, indices, expected, maxfun=100)
                    total_fg += int(result["nfev"])
                    endpoint = result.pop("endpoint")
                    full_theta = endpoint["full_theta"]
                    endpoint_raw = full_theta[:-1].reshape(2, data.n_loci, 3)
                    endpoint_coords = contact_model.sphere_forward(endpoint_raw)
                    endpoint_p = contact_model.p_from_q(float(full_theta[-1]))[0]
                    if full_theta[-1].tobytes() != source_theta[-1].tobytes() or endpoint_p != source["p"]:
                        raise RuntimeError("A changed frozen q/p")
                    outdir = RUN / "states/A" / arm_id
                    state = write_solver_state(outdir / "solver_state.npz",
                                               coordinates=endpoint_coords, raw_y=endpoint_raw,
                                               theta=full_theta, p=endpoint_p, q=float(full_theta[-1]))
                    presence = np.ones((2, data.n_loci), dtype=bool)
                    mask = write_presence_mask(outdir / "presence_mask.npz", presence, data.n_loci)
                    before = state["sha256"]
                    export = export_present_3dg(RUN / "exports/A" / f"{arm_id}.3dg",
                                                data, endpoint_coords, presence)
                    after = sha256_file(outdir / "solver_state.npz")
                    if before != after:
                        raise RuntimeError("A export changed solver state")
                    full_gradient = endpoint.pop("full_gradient")
                    active_gradient = endpoint.pop("active_gradient")
                    endpoint.pop("active")
                    endpoint.pop("full_theta")
                    record = {
                        "arm_id": arm_id, "seed": seed, "candidate": candidate,
                        "arm": arm, **result,
                        "endpoint": {"value": endpoint["value"],
                                     "components": endpoint["components"],
                                     "active_maxgrad": float(np.max(np.abs(active_gradient))),
                                     "active_grad_norm": float(np.linalg.norm(active_gradient)),
                                     "full_raw_y_maxgrad": float(np.max(np.abs(full_gradient[:-1]))),
                                     "q_absgrad": float(abs(full_gradient[-1])),
                                     "p": endpoint_p, "q": float(full_theta[-1])},
                        "artifacts": {"solver_state": state, "presence_mask": mask,
                                      "export": export, "state_sha_unchanged_after_export": True},
                        "reference_opened": False, "phase_opened": False,
                    }
                    dump(RUN / "results/A/arms" / f"{arm_id}.json", record)
                    terminal["arms"].append(record)
                    terminal["training_fg"] = total_fg
                    dump(terminal_path, terminal)
                    print(json.dumps({"event": "A_arm_complete", "arm_id": arm_id,
                                      "terminal": record["terminal"], "nfev": record["nfev"],
                                      "value": record["endpoint"]["value"],
                                      "active_maxgrad": record["endpoint"]["active_maxgrad"]}), flush=True)
                    del objective
                    torch.cuda.empty_cache()
        if total_fg > 800:
            raise RuntimeError("A FG cap exceeded")
        by = {(row["seed"], row["candidate"]["slot"], row["arm"]): row
              for row in terminal["arms"]}
        selections = []
        for seed_text, candidates in CONFIG["A"]["selection"]["candidates"].items():
            seed = int(seed_text); scored = []
            for candidate in candidates:
                control = by[(seed, candidate["slot"], "Control")]
                swap = by[(seed, candidate["slot"], "Swap")]
                j_gain = control["endpoint"]["value"] - swap["endpoint"]["value"]
                count_control = control["endpoint"]["components"]["count_nll_normalized"]
                count_swap = swap["endpoint"]["components"]["count_nll_normalized"]
                passed = j_gain >= 1e-5 and count_swap <= count_control
                scored.append({"seed": seed, "slot": candidate["slot"],
                               "candidate_order": candidate["candidate_order"],
                               "candidate_id": candidate["candidate_id"],
                               "chromosome": candidate["chromosome"],
                               "J_control_minus_swap": j_gain,
                               "count_control_minus_swap": count_control - count_swap,
                               "training_gate_passed": bool(passed)})
            passing = [row for row in scored if row["training_gate_passed"]]
            selected = max(passing, key=lambda row: (row["J_control_minus_swap"],
                                                     -row["candidate_order"])) if passing else None
            selections.append({"seed": seed, "candidates": scored, "selected": selected,
                               "status": "selected" if selected else "training_gate_failed"})
        dump(RUN / "results/A/selection.json", {"schema": "p9016-058-A-selection-v1",
                                                 "selections": selections,
                                                 "development_opened": False,
                                                 "reference_opened": False})
        terminal.update({"status": "complete", "training_fg": total_fg,
                         "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                         "selections": selections})
        dump(terminal_path, terminal)
        return 0
    except Exception as exc:
        terminal.update({"status": "failure", "error_type": type(exc).__name__,
                         "error": str(exc), "training_fg": total_fg,
                         "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        dump(terminal_path, terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
