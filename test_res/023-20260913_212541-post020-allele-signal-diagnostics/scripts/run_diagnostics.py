#!/usr/bin/env python3
"""针对冻结的 020/022 V1 endpoints 的 Phase-A diagnostics。

本脚本只读取冻结的 SNP-free aggregate、已保存的训练 snapshots 和固定 endpoint checkpoints。它不导入 phase/reference/Softall loaders，也不调用 optimizer。source package 明确选自 020 provenance snapshot，因此 objective 和 gradients 是已发布实现。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "test_res/023-20260913_212541-post020-allele-signal-diagnostics"
RUN020 = REPO / "test_res/020-20260913_071841-v1-p9016-joint"
RUN022 = REPO / "test_res/022-20260913_111031-v1-continuation-fdg-r2"
INPUT = REPO / "inputs/P9016.snpfree.pairs.gz"
CODE020 = RUN020 / "provenance/training-code"
CODE022 = RUN022 / "provenance/training-code"

# 使用已保存的 package，而不是可变的 working-tree package。
sys.path.insert(0, str(CODE020))
from pr import contact_model  # noqa: E402

if Path(contact_model.__file__).resolve() != (CODE020 / "pr/contact_model.py").resolve():
    raise RuntimeError("saved 020 contact_model snapshot was not imported")
# snapshot 的 paths.py 以其 provenance copy 为根；只绑定明确授权的 SNP-free input path，不改变冻结 source。
contact_model.SNPFREE = str(INPUT)


COMPONENT_NAMES = ("count", "bond", "repulsion", "bend", "p_prior")
VALUE_KEYS = {
    "count": "count_nll_normalized",
    "bond": "bond",
    "repulsion": "repulsion",
    "bend": "bend",
    "p_prior": "p_prior",
}
WEIGHT_KEYS = {"count": "count", "bond": "bond", "repulsion": "repulsion", "bend": "bend", "p_prior": "p_prior"}
HISTORY_KEYS = ("total", "count_nll_normalized", "bond", "repulsion", "bend", "p_prior", "p")
QUANTILE_LEVELS = (0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0)
QUANTILE_LABELS = ("q00", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "q100")


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value in JSON payload")
        return value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def stats(values: Iterable[float] | np.ndarray) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64).ravel()
    if len(a) == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None,
                "quantiles": {label: None for label in QUANTILE_LABELS}}
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite diagnostic values")
    q = np.quantile(a, QUANTILE_LEVELS)
    return {
        "n": int(len(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "quantiles": {label: float(value) for label, value in zip(QUANTILE_LABELS, q)},
    }


def norm_summary(array: np.ndarray) -> dict[str, float]:
    a = np.asarray(array, dtype=np.float64)
    absolute = np.abs(a)
    return {"l2": float(np.linalg.norm(a.ravel())), "infinity": float(absolute.max())}


def scalar_norm(value: float) -> dict[str, float]:
    value = float(value)
    return {"l2": abs(value), "infinity": abs(value)}


def track_name(chromosome_index: int, copy_index: int) -> str:
    return "c%02d%s" % (chromosome_index + 1, "ab"[copy_index])


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def endpoint_metadata() -> dict[str, dict[str, Any]]:
    stage = load_json(RUN020 / "stages/random_joint/1m.json")
    return {
        "020_random_joint_1m": {
            "endpoint_kind": "020_selected_random_joint_1m_checkpoint",
            "npz_path": RUN020 / "checkpoints/random_joint/1m-accepted-0240.npz",
            "coordinate_path": RUN020 / "selected.3dg",
            "expected_coordinate_sha256": "afb2d52ae11e342e9b43b3c8042c5581760d177c5e563f2478646ab36b3e7078",
            "expected_checkpoint_sha256": "c6a8890e2cff53767a8bddd49d4cb2d77892eee6c89f15895b3a5d8ba600b723",
            "published_components": stage["fit"]["final_components"],
            "published_total": stage["fit"]["final_total"],
            "source_iteration": 240,
            "source_run": "020",
        },
        "022_random_joint_continuation_480": {
            "endpoint_kind": "022_random_joint_1m_continuation_endpoint",
            "npz_path": RUN022 / "theta/final-theta.npz",
            "coordinate_path": RUN022 / "coords/random_joint/final-1m-continuation-61da99669694b822.3dg",
            "expected_coordinate_sha256": "49501f5b38d699fb2c9c8849616edd02b70fccd5ae39e1efe6036e17f5f620e7",
            "expected_theta_file_sha256": "588bafce1d7652fa8cf17cb060d23807dd6098c3e072fd9fc52784a145860a57",
            "published_components": load_json(RUN022 / "final_components.json")["components"],
            "published_total": load_json(RUN022 / "final_components.json")["components"]["total"],
            "source_iteration": 480,
            "source_run": "022",
        },
    }


def history_row(source: str, candidate: str, resolution: str, scope: str,
                entry: dict[str, Any]) -> dict[str, Any]:
    comp = entry.get("components", entry)
    local_iteration = int(entry["iteration"])
    return {
        "source": source,
        "candidate": candidate,
        "resolution": resolution,
        "iteration_scope": scope,
        "iteration": local_iteration,
        "cumulative_iteration": int(entry.get("cumulative_iteration", local_iteration)),
        "nfev": int(entry.get("nfev", entry.get("actual_nfev", 0))),
        "cumulative_nfev": int(entry.get("cumulative_nfev", entry.get("nfev", entry.get("actual_nfev", 0)))),
        "elapsed_seconds": float(entry.get("elapsed_seconds", 0.0)),
        "total": float(entry.get("fun", comp.get("total"))),
        "count_nll_normalized": float(comp["count_nll_normalized"]),
        "bond": float(comp["bond"]),
        "repulsion": float(comp["repulsion"]),
        "bend": float(comp["bend"]),
        "p_prior": float(comp["p_prior"]),
        "p": float(entry.get("p", comp.get("p"))),
    }


def improvement_summary(rows: list[dict[str, Any]], n_tail: int = 20) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty history")
    first = rows[0]
    last = rows[-1]
    result: dict[str, Any] = {
        "n_history_entries": len(rows),
        "start_iteration": int(first["iteration"]),
        "end_iteration": int(last["iteration"]),
        "start_elapsed_seconds": float(first["elapsed_seconds"]),
        "end_elapsed_seconds": float(last["elapsed_seconds"]),
        "start": {key: float(first[key]) for key in HISTORY_KEYS},
        "end": {key: float(last[key]) for key in HISTORY_KEYS},
        "delta_start_to_end_decrease": {
            key: float(first[key] - last[key]) for key in HISTORY_KEYS if key != "p"
        },
        "p_delta_start_to_end": float(last["p"] - first["p"]),
    }
    if len(rows) > n_tail:
        tail_start = rows[-n_tail - 1]
        actual_n = int(last["iteration"] - tail_start["iteration"])
        result["last_n_accepted_iterations"] = actual_n
        result["last_interval_start_iteration"] = int(tail_start["iteration"])
        result["last_interval_start"] = {key: float(tail_start[key]) for key in HISTORY_KEYS}
        result["last_interval_end"] = {key: float(last[key]) for key in HISTORY_KEYS}
        result["last_interval_decrease"] = {
            key: float(tail_start[key] - last[key]) for key in HISTORY_KEYS if key != "p"
        }
        result["last_interval_relative_decrease_per_iteration"] = {
            key: float((tail_start[key] - last[key]) / max(abs(tail_start[key]), 1.0) / max(actual_n, 1))
            for key in HISTORY_KEYS if key != "p"
        }
        result["last_interval_p_delta"] = float(last["p"] - tail_start["p"])
    else:
        result["last_n_accepted_iterations"] = None
    return result


def load_histories() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    term_audit = load_json(RUN020 / "termination_audit.json")
    term_by_key = {(r["candidate_id"], r["stage"]): r for r in term_audit["attempts"]}
    for path in sorted((RUN020 / "stages").glob("*/*.json")):
        candidate = path.parent.name
        resolution = path.stem
        record = load_json(path)
        fit = record["fit"]
        stage_rows = [history_row("020", candidate, resolution, "stage_local", e) for e in fit["history"]]
        rows.extend(stage_rows)
        raw = term_by_key[(candidate, resolution)]
        solver = dict(raw["raw_scipy"])
        # audit 提供 classification；stage record 提供权威的 wall time
        # 和 accepted-evaluation count。
        recorded_solver = fit["solver"]
        solver.update({
            "elapsed_seconds": float(recorded_solver["elapsed_seconds"]),
            "nfev": int(recorded_solver["nfev"]),
            "nit": int(recorded_solver["nit"]),
            "status": int(recorded_solver["status"]),
            "success": bool(recorded_solver["success"]),
            "message": str(recorded_solver["message"]),
            "actual_nfev": int(raw["raw_scipy"]["actual_nfev"]),
        })
        summaries.append({
            "source": "020",
            "candidate": candidate,
            "resolution": resolution,
            "iteration_scope": "stage_local",
            "history": improvement_summary(stage_rows),
            "final_components": fit["final_components"],
            "published_final_total": float(fit["final_total"]),
            "solver": solver,
            "derived_termination": raw["derived"],
            "budget": {"maxiter": int(solver["nit"]), "maxfun": int(raw["configured"]["maxfun"])},
            "elapsed_seconds": float(recorded_solver["elapsed_seconds"]),
        })

    continuation_rows = []
    with (RUN022 / "logs/progress.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                continuation_rows.append(json.loads(line))
    cont_rows = [history_row("022", "random_joint", "1m", "continuation_local", e) for e in continuation_rows]
    rows.extend(cont_rows)
    cont_audit = load_json(RUN022 / "terminal_audit.json")
    summaries.append({
        "source": "022",
        "candidate": "random_joint",
        "resolution": "1m",
        "iteration_scope": "continuation_local",
        "iteration_origin": 240,
        "cumulative_iteration_range": [240, 480],
        "history": improvement_summary(cont_rows),
        "final_components": cont_audit["objective_components_final"],
        "published_final_total": float(cont_audit["before_after"]["final"]["total"]),
        "solver": cont_audit["termination"],
        "derived_termination": {
            "classification": "iteration_budget_not_converged",
            "iteration_limit_reached": bool(cont_audit["termination"]["iteration_limit_reached"]),
            "function_limit_reached": bool(cont_audit["termination"]["function_limit_reached"]),
        },
        "budget": {"maxiter": int(cont_audit["termination"]["maxiter"]), "maxfun": int(cont_audit["termination"]["maxfun"])},
        "elapsed_seconds": float(cont_audit["termination"]["elapsed_seconds"]),
    })
    return rows, summaries


def write_history_tsv(rows: list[dict[str, Any]]) -> None:
    path = OUT / "results/history.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("source", "candidate", "resolution", "iteration_scope", "iteration",
               "cumulative_iteration", "nfev", "cumulative_nfev", "elapsed_seconds",
               "total", "count_nll_normalized", "bond", "repulsion", "bend", "p_prior", "p")
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row[c]) for c in columns))
    path.write_text("\n".join(lines) + "\n")


def model_weights(obj: Any) -> dict[str, float]:
    return {name: float(getattr(obj.weights, WEIGHT_KEYS[name])) for name in COMPONENT_NAMES}


def endpoint_gradient_diagnostic(data: Any, name: str, meta: dict[str, Any]) -> dict[str, Any]:
    npz_path = meta["npz_path"]
    expected_array_sha = meta.get("expected_checkpoint_sha256", meta.get("expected_theta_file_sha256"))
    if sha256_file(npz_path) != expected_array_sha:
        raise RuntimeError("endpoint array file hash mismatch for %s" % name)
    coordinate_sha = sha256_file(meta["coordinate_path"])
    if coordinate_sha != meta["expected_coordinate_sha256"]:
        raise RuntimeError("endpoint coordinate hash mismatch for %s" % name)
    with np.load(npz_path, allow_pickle=False) as payload:
        theta = np.asarray(payload["theta"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.float64)
        x_saved = np.asarray(payload["coordinates"], dtype=np.float64)
        positions = np.asarray(payload["positions"], dtype=np.int64)
        chromosome_index = np.asarray(payload["chromosome_index"], dtype=np.int64)
    if theta.shape != (6 * data.n_loci + 1,):
        raise ValueError("unexpected theta shape for %s: %s" % (name, theta.shape))
    if y.shape != (2, data.n_loci, 3) or x_saved.shape != y.shape:
        raise ValueError("unexpected endpoint array shape for %s" % name)
    x_forward = contact_model.sphere_forward(y)
    x_error = float(np.max(np.abs(x_forward - x_saved)))
    if x_error > 1e-12:
        raise ValueError("theta/y/coordinates mismatch for %s: %.3g" % (name, x_error))
    p, dpdq = contact_model.p_from_q(float(theta[-1]))

    class CapturingJointObjective(contact_model.JointObjective):
        """捕获一次已发布 evaluation 的精确 component gradients。"""

        def _count_nll_and_gradient(self, x, p, need_gradient):
            result = super()._count_nll_and_gradient(x, p, need_gradient)
            if need_gradient:
                self.captured_count = result
            return result

        def _bond_and_gradient(self, x):
            result = super()._bond_and_gradient(x)
            self.captured_bond = result
            return result

        def _repulsion_and_gradient(self, x):
            result = super()._repulsion_and_gradient(x)
            self.captured_repulsion = result
            return result

        def _bend_and_gradient(self, x):
            result = super()._bend_and_gradient(x)
            self.captured_bend = result
            return result

    obj = CapturingJointObjective(data)
    total, actual_gradient, components = obj.evaluate(theta, need_gradient=True)
    count_components, g_count_x, g_count_p = obj.captured_count
    bond, g_bond_x = obj.captured_bond
    repulsion, g_rep_x = obj.captured_repulsion
    bend, g_bend_x = obj.captured_bend
    p_prior = -contact_model.P_PRIOR_STRENGTH * math.log(p * (1.0 - p))
    p_prior_derivative_p = contact_model.P_PRIOR_STRENGTH * (1.0 / (1.0 - p) - 1.0 / p)
    zero_x = np.zeros_like(x_saved)
    raw_values = {
        "count": float(count_components["count_nll_normalized"]),
        "bond": float(bond),
        "repulsion": float(repulsion),
        "bend": float(bend),
        "p_prior": float(p_prior),
    }
    raw_grad_x = {
        "count": g_count_x,
        "bond": g_bond_x,
        "repulsion": g_rep_x,
        "bend": g_bend_x,
        "p_prior": zero_x,
    }
    raw_grad_p = {
        "count": float(g_count_p),
        "bond": 0.0,
        "repulsion": 0.0,
        "bend": 0.0,
        "p_prior": float(p_prior_derivative_p),
    }
    weights = model_weights(obj)
    term_payload: dict[str, Any] = {}
    combined_x_weighted = np.zeros_like(x_saved)
    combined_x_unweighted = np.zeros_like(x_saved)
    combined_p_weighted = 0.0
    combined_p_unweighted = 0.0
    weighted_total = 0.0
    unweighted_total = 0.0
    for component in COMPONENT_NAMES:
        gx = raw_grad_x[component]
        gp = raw_grad_p[component]
        gy = contact_model.sphere_pullback(y, gx)
        gq = float(dpdq * gp)
        w = weights[component]
        weighted_total += w * raw_values[component]
        unweighted_total += raw_values[component]
        combined_x_weighted += w * gx
        combined_x_unweighted += gx
        combined_p_weighted += w * gp
        combined_p_unweighted += gp
        term_payload[component] = {
            "value_unweighted": raw_values[component],
            "weight": w,
            "value_weighted_contribution": float(w * raw_values[component]),
            "physical_coordinate_gradient": norm_summary(gx),
            "physical_p_gradient": scalar_norm(gp),
            "sphere_parameter_y_gradient": norm_summary(gy),
            "sphere_parameter_q_gradient": scalar_norm(gq),
            "gradient_zero_physical_x": bool(np.all(gx == 0.0)),
            "gradient_zero_p": bool(gp == 0.0),
        }
    gy_weighted = contact_model.sphere_pullback(y, combined_x_weighted)
    gy_unweighted = contact_model.sphere_pullback(y, combined_x_unweighted)
    gq_weighted = float(dpdq * combined_p_weighted)
    gq_unweighted = float(dpdq * combined_p_unweighted)
    manual_weighted = np.concatenate((gy_weighted.ravel(), np.array([gq_weighted])))
    manual_unweighted = np.concatenate((gy_unweighted.ravel(), np.array([gq_unweighted])))
    recomposed_components = {
        "count_nll_normalized": raw_values["count"],
        "bond": raw_values["bond"],
        "repulsion": raw_values["repulsion"],
        "bend": raw_values["bend"],
        "p_prior": raw_values["p_prior"],
        "total": float(weighted_total),
    }
    component_diffs = {
        key: float(recomposed_components[key] - components[key])
        for key in ("count_nll_normalized", "bond", "repulsion", "bend", "p_prior", "total")
    }
    return {
        "endpoint": name,
        "endpoint_kind": meta["endpoint_kind"],
        "checkpoint_or_theta_path": rel(npz_path),
        "checkpoint_or_theta_sha256": sha256_file(npz_path),
        "coordinate_path": rel(meta["coordinate_path"]),
        "coordinate_sha256": sha256_file(meta["coordinate_path"]),
        "source_run": meta["source_run"],
        "source_iteration": int(meta["source_iteration"]),
        "input_budget": data.budget(),
        "n_loci": int(data.n_loci),
        "n_pairs": int(data.n_pairs),
        "l0": float(data.l0),
        "r0": float(data.r0),
        "theta_shape": list(theta.shape),
        "y_shape": list(y.shape),
        "coordinates_shape": list(x_saved.shape),
        "q": float(theta[-1]),
        "p": float(p),
        "dp_dq": float(dpdq),
        "theta_y_coordinate_consistency_max_abs": x_error,
        "published_components": meta["published_components"],
        "recomputed_components": components,
        "component_recomposition_differences": component_diffs,
        "objective_values": {
            "weighted_actual_joint_objective": float(total),
            "weighted_manual_sum": float(weighted_total),
            "unweighted_sum_of_normalized_terms": float(unweighted_total),
        },
        "weights": weights,
        "terms": term_payload,
        "combined": {
            "weighted": {
                "value": float(weighted_total),
                "physical_coordinate_gradient": norm_summary(combined_x_weighted),
                "physical_p_gradient": scalar_norm(combined_p_weighted),
                "sphere_parameter_y_gradient": norm_summary(gy_weighted),
                "sphere_parameter_q_gradient": scalar_norm(gq_weighted),
            },
            "unweighted": {
                "value": float(unweighted_total),
                "physical_coordinate_gradient": norm_summary(combined_x_unweighted),
                "physical_p_gradient": scalar_norm(combined_p_unweighted),
                "sphere_parameter_y_gradient": norm_summary(gy_unweighted),
                "sphere_parameter_q_gradient": scalar_norm(gq_unweighted),
            },
        },
        "joint_objective_gradient_recomposition": {
            "max_abs_difference": float(np.max(np.abs(manual_weighted - actual_gradient))),
            "l2_difference": float(np.linalg.norm(manual_weighted - actual_gradient)),
            "actual_gradient_l2": float(np.linalg.norm(actual_gradient)),
            "actual_gradient_infinity": float(np.max(np.abs(actual_gradient))),
            "passed_abs_1e-10": bool(np.max(np.abs(manual_weighted - actual_gradient)) <= 1e-10),
            "method": "exact analytic component recomposition; no large finite-difference rerun",
        },
        "key_gradient_validation_reuse": {
            "source": rel(RUN022 / "preflight.json"),
            "source_gradient_norm": load_json(RUN022 / "preflight.json")["gradient_diagnostic"]["gradient_norm"],
            "source_value": load_json(RUN022 / "preflight.json")["gradient_diagnostic"]["value"],
            "note": "The frozen preflight analytic gradient check is reused as evidence; this diagnostic only checks exact component recomposition against JointObjective.evaluate.",
        },
        "arrays_for_geometry": {"theta": theta, "y": y, "x": x_saved, "positions": positions, "chromosome_index": chromosome_index},
    }


def sphere_radius_diagnostic(data: Any, endpoint: str, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    radii = np.linalg.norm(x, axis=2)
    inv_radius = 1.0 / np.sqrt(1.0 + np.sum(y * y, axis=2))
    radial_attenuation = inv_radius ** 3
    tangential_attenuation = inv_radius
    if np.any(radii >= 1.0):
        raise ValueError("endpoint is not strictly inside unit ball")
    all_x = x.reshape(-1, 3)
    all_r = radii.reshape(-1)
    all_radial = radial_attenuation.reshape(-1)
    all_tangential = tangential_attenuation.reshape(-1)
    max_flat = int(np.argmax(all_r))
    max_copy, max_locus = divmod(max_flat, data.n_loci)
    max_ci = int(data.locus_chromosome[max_locus])
    global_summary = {
        "n_physical_beads": int(len(all_r)),
        "centroid": all_x.mean(axis=0),
        "centroid_norm": float(np.linalg.norm(all_x.mean(axis=0))),
        "radius": stats(all_r),
        "max_radius": float(all_r[max_flat]),
        "max_radius_bead": {
            "flat_index": max_flat,
            "copy_index": max_copy,
            "copy": "ab"[max_copy],
            "locus_index": max_locus,
            "chromosome": data.chromosome_names[max_ci],
            "track": track_name(max_ci, max_copy),
            "position_bp": int(data.locus_bin[max_locus] * data.bin_size),
            "coordinate": all_x[max_flat],
        },
        "boundary_counts": {
            "radius_ge_0.90": int(np.count_nonzero(all_r >= 0.90)),
            "radius_ge_0.95": int(np.count_nonzero(all_r >= 0.95)),
            "radius_ge_0.99": int(np.count_nonzero(all_r >= 0.99)),
            "radius_ge_1.00": int(np.count_nonzero(all_r >= 1.0)),
        },
        "boundary_fractions": {
            "radius_ge_0.90": float(np.mean(all_r >= 0.90)),
            "radius_ge_0.95": float(np.mean(all_r >= 0.95)),
            "radius_ge_0.99": float(np.mean(all_r >= 0.99)),
            "radius_ge_1.00": float(np.mean(all_r >= 1.0)),
        },
        "sphere_jacobian_singular_value_bounds": {
            "radial_attenuation": stats(all_radial),
            "tangential_attenuation": stats(all_tangential),
            "jacobian_operator_norm_equals_tangential_max": float(all_tangential.max()),
            "jacobian_min_singular_equals_radial_min": float(all_radial.min()),
        },
    }
    per_track = []
    for ci, _name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(ci)
        for copy_index in (0, 1):
            rr = radii[copy_index, slc]
            ra = radial_attenuation[copy_index, slc]
            ta = tangential_attenuation[copy_index, slc]
            coords = x[copy_index, slc]
            per_track.append({
                "track": track_name(ci, copy_index),
                "chromosome": data.chromosome_names[ci],
                "copy": "ab"[copy_index],
                "n_beads": int(len(rr)),
                "radius": stats(rr),
                "fraction_radius_ge_0.90": float(np.mean(rr >= 0.90)),
                "fraction_radius_ge_0.95": float(np.mean(rr >= 0.95)),
                "fraction_radius_ge_0.99": float(np.mean(rr >= 0.99)),
                "fraction_radius_ge_1.00": float(np.mean(rr >= 1.0)),
                "centroid": coords.mean(axis=0),
                "centroid_norm": float(np.linalg.norm(coords.mean(axis=0))),
                "sphere_jacobian_radial_attenuation": stats(ra),
                "sphere_jacobian_tangential_attenuation": stats(ta),
            })
    return {
        "endpoint": endpoint,
        "l0": float(data.l0),
        "r0": float(data.r0),
        "physical_coordinate_units": "dimensionless_R1",
        "global": global_summary,
        "per_track": per_track,
        "interpretation_boundary": "No bead at or above the unit sphere is allowed by the transform; threshold counts are descriptive and do not establish that the parameterization is irrelevant.",
        "arrays_for_plot": {
            "radius": all_r,
            "radial_attenuation": all_radial,
            "tangential_attenuation": all_tangential,
        },
    }


def add_summary_row(rows: list[dict[str, Any]], endpoint: str, scope: str, scope_id: str,
                    metric: str, values: np.ndarray, numerator: int | None = None,
                    denominator: int | None = None, ratio: float | None = None) -> None:
    row: dict[str, Any] = {"endpoint": endpoint, "scope": scope, "scope_id": scope_id, "metric": metric}
    row.update(stats(values))
    row["numerator"] = "" if numerator is None else int(numerator)
    row["denominator"] = "" if denominator is None else int(denominator)
    row["ratio"] = "" if ratio is None else float(ratio)
    rows.append(row)


def repulsion_geometry(data: Any, x: np.ndarray) -> dict[str, Any]:
    n = data.n_loci
    n_pairs = data.n_pairs
    threshold = 0.7 * data.l0
    n_terms = 4 * n_pairs + n
    all_distances = np.empty(n_terms, dtype=np.float64)
    nearest = np.full((2, n), np.inf, dtype=np.float64)
    near_track = np.zeros(40, dtype=np.int64)
    term_track = np.zeros(40, dtype=np.int64)
    write_at = 0
    block_size = 65536
    for start in range(0, n_pairs, block_size):
        stop = min(start + block_size, n_pairs)
        i = data.pair_i[start:stop]
        j = data.pair_j[start:stop]
        for copy_i, copy_j in ((0, 0), (0, 1), (1, 0), (1, 1)):
            d = np.linalg.norm(x[copy_i, i] - x[copy_j, j], axis=1)
            all_distances[write_at:write_at + len(d)] = d
            write_at += len(d)
            np.minimum.at(nearest[copy_i], i, d)
            np.minimum.at(nearest[copy_j], j, d)
            near = d < threshold
            ti = 2 * data.locus_chromosome[i] + copy_i
            tj = 2 * data.locus_chromosome[j] + copy_j
            np.add.at(near_track, ti, near.astype(np.int64))
            np.add.at(near_track, tj, near.astype(np.int64))
            np.add.at(term_track, ti, 1)
            np.add.at(term_track, tj, 1)
    homolog = np.linalg.norm(x[0] - x[1], axis=1)
    all_distances[write_at:write_at + n] = homolog
    write_at += n
    homolog_near = homolog < threshold
    near_track += np.bincount(2 * data.locus_chromosome, weights=homolog_near.astype(np.int64), minlength=40).astype(np.int64)
    near_track += np.bincount(2 * data.locus_chromosome + 1, weights=homolog_near.astype(np.int64), minlength=40).astype(np.int64)
    term_track += np.bincount(2 * data.locus_chromosome, minlength=40).astype(np.int64)
    term_track += np.bincount(2 * data.locus_chromosome + 1, minlength=40).astype(np.int64)
    if write_at != n_terms or np.any(~np.isfinite(nearest)):
        raise RuntimeError("repulsion model distance inventory is incomplete")
    expected_track_terms = (2 * n - 1) * np.repeat(
        np.bincount(data.locus_chromosome, minlength=20), 2)
    if not np.all(term_track == expected_track_terms):
        raise RuntimeError("repulsion endpoint-incidence denominator mismatch")
    near_total = int(np.count_nonzero(all_distances < threshold))
    return {
        "threshold": float(threshold),
        "threshold_definition": "0.7*l0; hinge is active only for distance < threshold",
        "model_distance_term_denominator": int(n_terms),
        "model_distance_term_formula": "4*n_eligible_unordered_locus_pairs + n_loci homolog terms",
        "endpoint_incidence_denominator_per_track": expected_track_terms,
        "near_total": near_total,
        "near_fraction_total": float(near_total / n_terms),
        "all_model_distances": all_distances,
        "nearest_by_bead": nearest,
        "near_track": near_track,
        "term_track": term_track,
        "homolog_distances": homolog,
    }


def geometry_diagnostic(data: Any, endpoint: str, x: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    l0 = float(data.l0)
    bond_all = []
    bend_all = []
    angle_all = []
    track_bond: dict[str, np.ndarray] = {}
    track_bend: dict[str, np.ndarray] = {}
    track_angle: dict[str, np.ndarray] = {}
    for ci, _name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(ci)
        for copy_index in (0, 1):
            name = track_name(ci, copy_index)
            coords = x[copy_index, slc]
            bond = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            second = coords[2:] - 2.0 * coords[1:-1] + coords[:-2]
            bend = np.linalg.norm(second, axis=1) / l0
            v1 = coords[1:-1] - coords[:-2]
            v2 = coords[2:] - coords[1:-1]
            denominator = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            cosine = np.divide(np.sum(v1 * v2, axis=1), denominator, out=np.zeros_like(denominator), where=denominator > 0.0)
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            bond_all.append(bond)
            bend_all.append(bend)
            angle_all.append(angle)
            track_bond[name] = bond
            track_bend[name] = bend
            track_angle[name] = angle
            add_summary_row(rows, endpoint, "track", name, "bond_distance_R", bond)
            add_summary_row(rows, endpoint, "track", name, "bond_distance_over_l0", bond / l0)
            add_summary_row(rows, endpoint, "track", name, "bend_second_difference_over_l0", bend)
            add_summary_row(rows, endpoint, "track", name, "bend_turn_angle_rad", angle)
    bond_flat = np.concatenate(bond_all)
    bend_flat = np.concatenate(bend_all)
    angle_flat = np.concatenate(angle_all)
    add_summary_row(rows, endpoint, "overall", "all_tracks", "bond_distance_R", bond_flat)
    add_summary_row(rows, endpoint, "overall", "all_tracks", "bond_distance_over_l0", bond_flat / l0)
    add_summary_row(rows, endpoint, "overall", "all_tracks", "bend_second_difference_over_l0", bend_flat)
    add_summary_row(rows, endpoint, "overall", "all_tracks", "bend_turn_angle_rad", angle_flat)

    rep = repulsion_geometry(data, x)
    add_summary_row(rows, endpoint, "overall", "all_model_terms", "repulsion_model_distance_R", rep["all_model_distances"], rep["near_total"], rep["model_distance_term_denominator"], rep["near_fraction_total"])
    nearest_flat = rep["nearest_by_bead"].ravel()
    add_summary_row(rows, endpoint, "overall", "all_beads", "repulsion_nearest_bead_distance_R", nearest_flat)
    for ci, _name in enumerate(data.chromosome_names):
        for copy_index in (0, 1):
            name = track_name(ci, copy_index)
            ti = 2 * ci + copy_index
            nearest_track = rep["nearest_by_bead"][copy_index, data.chromosome_slice(ci)]
            near = int(rep["near_track"][ti])
            denom = int(rep["term_track"][ti])
            add_summary_row(rows, endpoint, "track", name, "repulsion_nearest_bead_distance_R", nearest_track, near, denom, near / denom)
            # track-level indicator 使表格中的分母明确。
            add_summary_row(rows, endpoint, "track", name, "repulsion_near_fraction", np.asarray([near / denom]), near, denom, near / denom)

    details = {
        "endpoint": endpoint,
        "l0": l0,
        "repulsion_threshold": rep["threshold"],
        "bond_terms_per_track_formula": "n_loci_in_track - 1",
        "bend_terms_per_track_formula": "n_loci_in_track - 2",
        "repulsion_exclusion_rule": "full eligible unordered locus-pair grid; all four copy combinations plus the homologous same-locus pair; no same-copy same-locus self term",
        "repulsion": {
            "model_distance_term_denominator": rep["model_distance_term_denominator"],
            "model_distance_term_formula": rep["model_distance_term_formula"],
            "endpoint_incidence_denominator_per_track": rep["endpoint_incidence_denominator_per_track"],
            "near_total": rep["near_total"],
            "near_fraction_total": rep["near_fraction_total"],
            "homolog_distance": stats(rep["homolog_distances"]),
        },
        "overall": {
            "bond_distance_R": stats(bond_flat),
            "bond_distance_over_l0": stats(bond_flat / l0),
            "bend_second_difference_over_l0": stats(bend_flat),
            "bend_turn_angle_rad": stats(angle_flat),
            "repulsion_model_distance_R": stats(rep["all_model_distances"]),
            "repulsion_nearest_bead_distance_R": stats(nearest_flat),
        },
    }
    samples = {
        "bond_over_l0": bond_flat[::max(1, len(bond_flat) // 200000)],
        "bend_over_l0": bend_flat[::max(1, len(bend_flat) // 200000)],
        "repulsion_model_distance": rep["all_model_distances"][::max(1, len(rep["all_model_distances"]) // 400000)],
        "repulsion_nearest_distance": nearest_flat,
    }
    return rows, details, samples


def source_provenance() -> dict[str, Any]:
    source_files = [
        RUN020 / "config.json",
        RUN020 / "selection.json",
        RUN020 / "termination_audit.json",
        RUN022 / "config.json",
        RUN022 / "continuation_summary.json",
        RUN022 / "terminal_audit.json",
        RUN022 / "terminal_audit_independent.json",
        RUN022 / "preflight.json",
        RUN022 / "protocol-freeze.json",
        RUN020 / "provenance/training-code-manifest.json",
        RUN022 / "provenance/training-code-manifest.json",
    ]
    code_files = [
        CODE020 / "pr/contact_model.py",
        CODE020 / "pr/joint_fit.py",
        CODE020 / "pr/genome.py",
        CODE020 / "pr/pairs7.py",
        CODE020 / "pr/paths.py",
        CODE022 / "pr/contact_model.py",
        CODE022 / "pr/joint_fit.py",
        CODE022 / "pr/continuation.py",
    ]
    return {
        "input": {"path": rel(INPUT), "sha256": sha256_file(INPUT), "expected_sha256": "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"},
        "source_snapshots": {
            rel(path): {"path": rel(path), "sha256": sha256_file(path)}
            for path in code_files
        },
        "frozen_metadata_files": {
            rel(path): {"path": rel(path), "sha256": sha256_file(path)}
            for path in source_files
        },
        "analysis_script": {"path": rel(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        "training_boundary": {
            "phase_used": False,
            "reference_used": False,
            "softall_coordinates_opened": False,
            "native_fdg_started": False,
            "new_fit_started": False,
        },
    }


def static_config(data: Any, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "post020-allele-signal-diagnostics-v1",
        "purpose": "phase-A fixed-endpoint diagnostics; no optimization",
        "input": {
            "path": rel(INPUT),
            "sha256": provenance["input"]["sha256"],
            "raw_record_budget": 1703888,
            "raw_record_accounting": {"cis": 1135454, "inter": 568434},
            "same_bin_records": 438774,
            "cis_offdiag_structural_records": 696680,
            "structural_records": 1265114,
        },
        "likelihood_observation_unit": "aggregated unordered genomic bin-pair count C_ij over the complete eligible full grid; same-bin counts are a separate saturated nuisance layer",
        "diagnostic_distribution_units": {
            "raw_input": "raw records",
            "bond": "physical bead-to-bead backbone bond terms",
            "bend": "physical bead-centered second differences/turn angles",
            "repulsion": "eligible physical pair terms under the model exclusion rule, with explicit term and endpoint-incidence denominators",
        },
        "coordinate_grid": {
            "origin_bp": 0,
            "bin_size_bp": 1000000,
            "n_loci": int(data.n_loci),
            "physical_beads": int(2 * data.n_loci),
            "tracks": 40,
            "coordinate_units": "dimensionless_R1",
            "nuclear_radius_parameter": 1.0,
        },
        "frozen_model": {
            "weights": {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0},
            "kernel": {"epsilon": 1e-6, "exponent": 4, "r0": "2*l0"},
            "exposure": "sqrt(endpoint_count + 10), normalized to full-grid mean one",
            "same_bin_layer": "per_bin_saturated_poisson_nuisance",
            "p_floor": 1e-4,
            "p_prior_strength": 1e-4,
            "bond_target_interval_over_l0": [0.75, 1.25],
            "repulsion_implementation_threshold": "0.7*l0",
            "config_field_repulsion_radius_l0": 2.0,
        },
        "endpoints": {
            "020": {"path": rel(RUN020 / "checkpoints/random_joint/1m-accepted-0240.npz"), "checkpoint_sha256": "c6a8890e2cff53767a8bddd49d4cb2d77892eee6c89f15895b3a5d8ba600b723"},
            "022": {"path": rel(RUN022 / "theta/final-theta.npz"), "file_sha256": "588bafce1d7652fa8cf17cb060d23807dd6098c3e072fd9fc52784a145860a57"},
        },
        "source_provenance": provenance,
    }


def model_semantics_audit(data: Any, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "post020-model-semantics-audit-v1",
        "source_code": rel(CODE020 / "pr/contact_model.py"),
        "source_code_sha256": provenance["source_snapshots"][rel(CODE020 / "pr/contact_model.py")]["sha256"],
        "config_source": rel(RUN020 / "config.json"),
        "config_source_sha256": provenance["frozen_metadata_files"][rel(RUN020 / "config.json")]["sha256"],
        "l0": float(data.l0),
        "contact_kernel": {
            "epsilon": 1e-6,
            "formula": "epsilon + (1-epsilon)*(1+d2/r0^2)^-2",
            "r0": float(data.r0),
            "r0_over_l0": 2.0,
            "implementation_source_lines": "671-684",
        },
        "repulsion_prior": {
            "config_field_repulsion_radius_l0": 2.0,
            "implementation_threshold": float(0.7 * data.l0),
            "implementation_threshold_over_l0": 0.7,
            "implementation_source_lines": "923-952; threshold = 0.7 * data.l0",
            "config_source_field": "model_definition.prior_configuration.repulsion_radius_l0",
            "warning": "The config field 2.0 is not the implemented repulsion hinge threshold; do not use 2*l0 as the repulsion exclusion radius.",
        },
        "weights": {"bend": 0.01, "bond": 1.0, "repulsion": 1.0, "count": 1.0, "p_prior": 1.0},
        "p_initialization": {"first_layer_p_init": 0.75, "note": "No phased p near 0.98 prior is imported by this diagnostic."},
    }


def make_plots(history_rows: list[dict[str, Any]], radius_payloads: dict[str, dict[str, Any]],
               geometry_samples: dict[str, dict[str, np.ndarray]], threshold: float) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    out_paths: list[str] = []
    stage_resolutions = ("5m", "2m", "1m")
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    for col, resolution in enumerate(stage_resolutions):
        ax = axes[0, col]
        for candidate, color in (("consensus_joint", "#3366aa"), ("random_joint", "#cc5533")):
            sub = [r for r in history_rows if r["source"] == "020" and r["resolution"] == resolution and r["candidate"] == candidate]
            xx = np.asarray([r["iteration"] for r in sub])
            ax.plot(xx, [r["total"] for r in sub], color=color, lw=0.8, label=candidate.replace("_joint", "") + " total")
            ax.plot(xx, [r["count_nll_normalized"] for r in sub], color=color, lw=0.8, ls="--", label=candidate.replace("_joint", "") + " count")
        ax.set_title("020 %s (stage-local)" % resolution)
        ax.set_xlabel("accepted iteration")
        ax.set_ylabel("objective / count NLL")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if col == 2:
            ax.legend(loc="best", frameon=False, ncol=2)
        ax.grid(alpha=0.2)
        ax = axes[1, col]
        for candidate, color in (("consensus_joint", "#3366aa"), ("random_joint", "#cc5533")):
            sub = [r for r in history_rows if r["source"] == "020" and r["resolution"] == resolution and r["candidate"] == candidate]
            xx = np.asarray([r["iteration"] for r in sub])
            ax.plot(xx, [r["p"] for r in sub], color=color, lw=0.8, label=candidate.replace("_joint", ""))
        ax.set_title("p curve")
        ax.set_xlabel("accepted iteration")
        ax.set_ylabel("p")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(alpha=0.2)
    path = OUT / "plots/objective_history_stages.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    out_paths.append(rel(path))

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    sub = [r for r in history_rows if r["source"] == "022"]
    xx = np.asarray([r["iteration"] for r in sub])
    axes[0].plot(xx, [r["total"] for r in sub], color="#222222", lw=0.8, label="total")
    axes[0].plot(xx, [r["count_nll_normalized"] for r in sub], color="#3366aa", lw=0.8, ls="--", label="count NLL")
    axes[0].set_title("022 continuation (local 0-240)")
    axes[0].set_xlabel("local accepted iteration")
    axes[0].set_ylabel("objective / count NLL")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    for key, color in (("bend", "#cc5533"), ("bond", "#3366aa"), ("repulsion", "#228855"), ("p_prior", "#8855aa")):
        axes[1].plot(xx, [r[key] * (0.01 if key == "bend" else 1.0) for r in sub], lw=0.8, label="weighted " + key)
    axes[1].set_title("weighted priors")
    axes[1].set_xlabel("local accepted iteration")
    axes[1].set_ylabel("contribution")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(alpha=0.2)
    path = OUT / "plots/continuation_history.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    out_paths.append(rel(path))

    colors = {"020_random_joint_1m": "#3366aa", "022_random_joint_continuation_480": "#cc5533"}
    fig, axes = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)
    for endpoint, payload in radius_payloads.items():
        r = np.sort(payload["arrays_for_plot"]["radius"])
        ycdf = np.arange(1, len(r) + 1) / len(r)
        axes[0, 0].plot(r, ycdf, lw=0.8, color=colors[endpoint], label=endpoint.split("_")[0])
        sample_r = payload["arrays_for_plot"]["radius"][::max(1, len(payload["arrays_for_plot"]["radius"]) // 50000)]
        sample_radial = payload["arrays_for_plot"]["radial_attenuation"][::max(1, len(payload["arrays_for_plot"]["radial_attenuation"]) // 50000)]
        sample_tangential = payload["arrays_for_plot"]["tangential_attenuation"][::max(1, len(payload["arrays_for_plot"]["tangential_attenuation"]) // 50000)]
        order = np.argsort(sample_r)
        axes[0, 1].plot(sample_r[order], sample_radial[order], lw=0.5, color=colors[endpoint], label=endpoint.split("_")[0] + " radial")
        axes[0, 1].plot(sample_r[order], sample_tangential[order], lw=0.5, ls="--", color=colors[endpoint], label=endpoint.split("_")[0] + " tangential")
        samples = geometry_samples[endpoint]
        axes[1, 0].hist(samples["bond_over_l0"], bins=50, density=True, histtype="step", lw=0.8, color=colors[endpoint], label=endpoint.split("_")[0])
        axes[1, 1].hist(samples["repulsion_nearest_distance"], bins=50, density=True, histtype="step", lw=0.8, color=colors[endpoint], label=endpoint.split("_")[0])
    axes[0, 0].axvline(0.9, color="#666666", lw=0.5, ls=":")
    axes[0, 0].axvline(0.99, color="#666666", lw=0.5, ls="--")
    axes[0, 0].set_title("radius CDF")
    axes[0, 0].set_xlabel("radius")
    axes[0, 0].set_ylabel("fraction beads")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set_title("sphere Jacobian attenuation")
    axes[0, 1].set_xlabel("radius")
    axes[0, 1].set_ylabel("singular attenuation")
    axes[0, 1].legend(frameon=False, ncol=2)
    axes[1, 0].axvspan(0.75, 1.25, color="#dddddd", alpha=0.4)
    axes[1, 0].set_title("backbone bond")
    axes[1, 0].set_xlabel("distance / l0")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].axvline(threshold, color="#222222", lw=0.7, ls="--", label="0.7 l0")
    axes[1, 1].set_title("nearest repulsion distance")
    axes[1, 1].set_xlabel("distance")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].legend(frameon=False)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    path = OUT / "plots/endpoint_geometry_diagnostics.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    out_paths.append(rel(path))
    return out_paths


def recommendations(history_summaries: list[dict[str, Any]], endpoint_gradients: dict[str, Any],
                    radius_payloads: dict[str, dict[str, Any]], semantics: dict[str, Any]) -> dict[str, Any]:
    stage_times = {(row["source"], row["candidate"], row["resolution"]): float(row["elapsed_seconds"]) for row in history_summaries}
    random_path_seconds = sum(stage_times[("020", "random_joint", r)] for r in ("5m", "2m", "1m"))
    consensus_path_seconds = sum(stage_times[("020", "consensus_joint", r)] for r in ("5m", "2m", "1m"))
    continuation_seconds = float(next(row["elapsed_seconds"] for row in history_summaries if row["source"] == "022"))
    max_radius = {key: float(value["global"]["max_radius"]) for key, value in radius_payloads.items()}
    gradients = {}
    for key, payload in endpoint_gradients.items():
        weighted = payload["combined"]["weighted"]
        count = payload["terms"]["count"]["sphere_parameter_y_gradient"]["l2"]
        bend = payload["terms"]["bend"]["sphere_parameter_y_gradient"]["l2"]
        gradients[key] = {
            "weighted_y_l2": weighted["sphere_parameter_y_gradient"]["l2"],
            "count_y_l2": count,
            "bend_y_l2": bend,
            "bend_to_count_y_l2_ratio": float(bend / count) if count > 0 else None,
            "weighted_physical_x_l2": weighted["physical_coordinate_gradient"]["l2"],
            "count_physical_x_l2": payload["terms"]["count"]["physical_coordinate_gradient"]["l2"],
            "bend_physical_x_l2": payload["terms"]["bend"]["physical_coordinate_gradient"]["l2"],
            "bend_to_count_physical_x_l2_ratio": float(payload["terms"]["bend"]["physical_coordinate_gradient"]["l2"] / payload["terms"]["count"]["physical_coordinate_gradient"]["l2"]),
            "bend_weighted_y_l2": float(0.01 * bend),
        }
    return {
        "schema_version": "post020-stage-a-recommendations-v1",
        "selection_rule": "No reference/phase criterion used; times and fixed budgets only.",
        "measured_wall_time_seconds": {
            "020_random_joint_5m_2m_1m_serial_sum": random_path_seconds,
            "020_consensus_joint_5m_2m_1m_serial_sum": consensus_path_seconds,
            "022_continuation_240_local_iterations": continuation_seconds,
            "020_runtime_candidate_workers": 2,
        },
        "fixed_follow_up_options": {
            "C0_022_warm_continuation": {
                "status": "recommend_fixed_budget_only",
                "same_initial_endpoint": "022 cumulative iteration 480",
                "additional_1m_accepted_iterations": 240,
                "solver_restart": True,
                "estimated_wall_seconds_one_worker": continuation_seconds,
                "reason": "separate budget sufficiency from initialization; not an independent repeat",
            },
            "independent_random_joint": {
                "status": "recommend_two_preregistered_independent_initialization_seeds",
                "n_fits": 2,
                "schedule_per_fit": {"5m": 300, "2m": 200, "1m": 240},
                "estimated_serial_seconds_per_fit": random_path_seconds,
                "estimated_cpu_seconds_for_two": 2.0 * random_path_seconds,
                "estimated_wall_seconds_with_two_workers": random_path_seconds,
                "parallelism_basis": "020 ran two candidate workers; this is optimization-repeat capacity, not biological/data replication",
                "seed_and_initialization_note": "Use new random assignment/native seeds explicitly recorded by the future fit; do not import phased p as an initializer.",
            },
        },
        "boundary_activity": {
            "max_radius_by_endpoint": max_radius,
            "all_endpoints_have_zero_beads_at_ge_0.90": all(value["global"]["boundary_counts"]["radius_ge_0.90"] == 0 for value in radius_payloads.values()),
            "conclusion": "Active hard-wall occupancy/clipping can be excluded at these endpoints (max radius < 0.70 and zero beads at 0.90/0.95/0.99/1.00); this does not prove the sphere parameterization has no optimization effect.",
        },
        "bend_relative_gradient": gradients,
        "recommended_C2_minimal_factor_split": {
            "purpose": "separate physical radial boundary from coordinate parameterization without changing contact-kernel or prior scales",
            "fixed_scales": {"r0": "2*l0", "epsilon": 1e-6, "repulsion_threshold": "0.7*l0", "bond_target_interval": ["0.75*l0", "1.25*l0"]},
            "factorial_cells": [
                {"cell": "sphere_none", "parameterization": "current sphere_forward", "radial_potential": "none", "role": "frozen baseline"},
                {"cell": "sphere_soft", "parameterization": "same sphere_forward", "radial_potential": "fixed soft penalty beyond R_soft=0.90", "role": "boundary term under current parameterization"},
                {"cell": "cartesian_none", "parameterization": "centered Cartesian coordinates", "radial_potential": "none", "role": "parameterization change without radial penalty"},
                {"cell": "cartesian_soft", "parameterization": "centered Cartesian coordinates", "radial_potential": "same fixed soft penalty beyond R_soft=0.90", "role": "physical boundary under Cartesian parameterization"},
            ],
            "minimum_protocol": "Run the 2x2 cells from the same phase-free random endpoint/seed with fixed budgets; center Cartesian coordinates to remove translation gauge, keep all other scales/weights fixed, and choose R_soft/penalty before reading any reference score.",
            "interpretation": "cartesian_soft - cartesian_none estimates the radial-potential effect at fixed parameterization; sphere_soft - sphere_none is the same check for the current map; sphere versus Cartesian at matched radial-potential state estimates parameterization sensitivity.",
        },
    }


def write_radius_tsv(radius_payloads: dict[str, dict[str, Any]]) -> None:
    path = OUT / "results/radius_sphere.tsv"
    columns = ["endpoint", "track", "chromosome", "copy", "n_beads",
               "radius_min", "radius_q01", "radius_q05", "radius_q25", "radius_q50",
               "radius_q75", "radius_q95", "radius_q99", "radius_q100", "radius_max",
               "fraction_ge_0.90", "fraction_ge_0.95", "fraction_ge_0.99", "fraction_ge_1.00",
               "centroid_x", "centroid_y", "centroid_z", "centroid_norm"]
    lines = ["\t".join(columns)]
    for endpoint, payload in radius_payloads.items():
        for track in payload["per_track"]:
            q = track["radius"]["quantiles"]
            values = [endpoint, track["track"], track["chromosome"], track["copy"], track["n_beads"],
                      track["radius"]["min"], q["q01"], q["q05"], q["q25"], q["q50"],
                      q["q75"], q["q95"], q["q99"], q["q100"], track["radius"]["max"],
                      track["fraction_radius_ge_0.90"], track["fraction_radius_ge_0.95"],
                      track["fraction_radius_ge_0.99"], track["fraction_radius_ge_1.00"],
                      track["centroid"][0], track["centroid"][1], track["centroid"][2], track["centroid_norm"]]
            lines.append("\t".join(str(value) for value in values))
    path.write_text("\n".join(lines) + "\n")


def write_gradient_tsv(endpoint_payloads: dict[str, dict[str, Any]]) -> None:
    path = OUT / "results/gradient_summary.tsv"
    columns = ["endpoint", "p", "q", "objective_total", "gradient_y_l2", "gradient_y_infinity", "gradient_q_abs",
               "component", "value_unweighted", "value_weighted", "gradient_x_l2", "gradient_x_infinity",
               "gradient_y_l2_component", "gradient_y_infinity_component", "gradient_q_abs_component"]
    lines = ["\t".join(columns)]
    for endpoint, payload in endpoint_payloads.items():
        combined = payload["combined"]["weighted"]
        for component in COMPONENT_NAMES:
            term = payload["terms"][component]
            lines.append("\t".join(str(value) for value in [
                endpoint, payload["p"], payload["q"], payload["objective_values"]["weighted_actual_joint_objective"],
                combined["sphere_parameter_y_gradient"]["l2"], combined["sphere_parameter_y_gradient"]["infinity"],
                combined["sphere_parameter_q_gradient"]["l2"], component, term["value_unweighted"],
                term["value_weighted_contribution"], term["physical_coordinate_gradient"]["l2"],
                term["physical_coordinate_gradient"]["infinity"], term["sphere_parameter_y_gradient"]["l2"],
                term["sphere_parameter_y_gradient"]["infinity"], term["sphere_parameter_q_gradient"]["l2"]
            ]))
    path.write_text("\n".join(lines) + "\n")


def write_geometry_tsv(rows: list[dict[str, Any]]) -> None:
    path = OUT / "results/geometry_summary.tsv"
    columns = ["endpoint", "scope", "scope_id", "metric", "n", "min", "max", "mean", "std"] + list(QUANTILE_LABELS) + ["numerator", "denominator", "ratio"]
    lines = ["\t".join(columns)]
    for row in rows:
        values = [row.get("n", ""), row.get("min", ""), row.get("max", ""), row.get("mean", ""), row.get("std", "")]
        values.extend(row.get("quantiles", {}).get(label, "") for label in QUANTILE_LABELS)
        values.extend([row.get("numerator", ""), row.get("denominator", ""), row.get("ratio", "")])
        lines.append("\t".join([str(row.get(c, "")) for c in columns[:4]] + [str(v) for v in values]))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = source_provenance()
    if provenance["input"]["sha256"] != provenance["input"]["expected_sha256"]:
        raise RuntimeError("SNP-free input SHA mismatch")
    data = contact_model.load_frozen_p9016_aggregate(1_000_000)
    if data.n_loci != 2645 or data.n_pairs != 3496690:
        raise RuntimeError("frozen full-grid dimensions changed")
    write_json(OUT / "config.json", static_config(data, provenance))
    write_json(OUT / "results/provenance.json", provenance)
    write_json(OUT / "results/data_budget.json", data.budget())
    write_json(OUT / "results/model_semantics_audit.json", model_semantics_audit(data, provenance))

    history_rows, history_summaries = load_histories()
    write_history_tsv(history_rows)
    write_json(OUT / "results/history_summary.json", {
        "schema_version": "post020-history-summary-v1",
        "observation_unit_note": "Raw records are a budget/accounting unit; count likelihood entries are aggregated unordered eligible genomic bin-pair counts.",
        "stage_summaries": history_summaries,
    })

    endpoint_payloads: dict[str, dict[str, Any]] = {}
    radius_payloads: dict[str, dict[str, Any]] = {}
    radius_plot_payloads: dict[str, dict[str, Any]] = {}
    geometry_samples: dict[str, dict[str, np.ndarray]] = {}
    geometry_rows: list[dict[str, Any]] = []
    geometry_details: dict[str, Any] = {}
    endpoint_meta = endpoint_metadata()
    for name, meta in endpoint_meta.items():
        payload = endpoint_gradient_diagnostic(data, name, meta)
        endpoint_payloads[name] = {k: v for k, v in payload.items() if k != "arrays_for_geometry"}
        arrays = payload["arrays_for_geometry"]
        radius = sphere_radius_diagnostic(data, name, arrays["x"], arrays["y"])
        radius_payloads[name] = {k: v for k, v in radius.items() if k != "arrays_for_plot"}
        radius_plot_payloads[name] = {"arrays_for_plot": radius["arrays_for_plot"]}
        rows, details, samples = geometry_diagnostic(data, name, arrays["x"])
        geometry_rows.extend(rows)
        geometry_details[name] = details
        geometry_samples[name] = samples
    write_json(OUT / "results/endpoint_gradients.json", endpoint_payloads)
    write_json(OUT / "results/radius_sphere.json", radius_payloads)
    write_gradient_tsv(endpoint_payloads)
    write_radius_tsv(radius_payloads)
    write_geometry_tsv(geometry_rows)
    write_json(OUT / "results/geometry_details.json", geometry_details)
    recommendations_payload = recommendations(history_summaries, endpoint_payloads, radius_payloads, load_json(OUT / "results/model_semantics_audit.json"))
    write_json(OUT / "results/recommendations.json", recommendations_payload)
    plot_paths = make_plots(history_rows, radius_plot_payloads, geometry_samples, 0.7 * data.l0)
    write_json(OUT / "results/plot_manifest.json", {"plots": plot_paths, "dpi": 300, "base_panel_inches": 3, "text_size_pt": 7, "reference_or_R2_plots": False})
    runtime = {"status": "completed", "started_epoch": started, "ended_epoch": time.time(), "elapsed_seconds": time.time() - started, "python": sys.executable, "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"), "command": "source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh && conda activate analysis && python scripts/run_diagnostics.py"}
    write_json(OUT / "logs/analysis_runtime.json", runtime)
    print(json.dumps({"status": "completed", "output_root": str(OUT), "history_rows": len(history_rows), "geometry_rows": len(geometry_rows), "plots": plot_paths, "elapsed_seconds": runtime["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
