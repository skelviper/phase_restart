"""Formal M0/M1 GPU controller for the frozen P9016 comparison.

This module is training-side only.  It imports the read-only 035 adapter and
028 fused backend, plus phase-free ``pr`` loaders.  It never opens evaluation
or reference payloads.  The formal matrix is:

* real: M0/M1 x B1/B2 x consensus/random, each 5m -> 2m -> 1m;
* synthetic: P2/N2 x (M0/production-e, M1/production-e,
  M0/known-generating-e), one 1m layer each.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "test_res" / "037-20260914T143812Z-gpu-m1-preflight"
SOURCE_DIR = Path(__file__).resolve().parent
SOURCE_035 = ROOT / "test_res" / "035-20260914T060945Z-gpu-multires-preflight" / "source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_035) not in sys.path:
    sys.path.insert(0, str(SOURCE_035))

from gpu_variant_backend import GPUVariantObjective, cuda_probe  # noqa: E402
from m1_preconditioner import M1Objective, run_budgeted_lbfgs  # noqa: E402
from pr import contact_model, genome, multires_variant_runner as warm, paired_run  # noqa: E402
from pr import reconstruction_init, v1_calibration  # noqa: E402


REAL_STAGES = (
    {"label": "5m", "bin_size_bp": 5_000_000, "b1_fg": 306, "b2_fg": 612},
    {"label": "2m", "bin_size_bp": 2_000_000, "b1_fg": 202, "b2_fg": 404},
    {"label": "1m", "bin_size_bp": 1_000_000, "b1_fg": 243, "b2_fg": 486},
)
CANDIDATES = (
    {"candidate_id": "consensus_joint", "initialization_candidate": "consensus", "base_seed": 1103},
    {"candidate_id": "random_joint", "initialization_candidate": "random", "base_seed": 2207},
)
METHODS = ("M0", "M1")
BUDGETS = ("B1", "B2")
SYNTHETIC_FIXTURES = ("P2", "N2")
PREPARED_RUN = ROOT / "test_res" / "026-20260913_221709-allele-calibration-prepare"
KNOWN_E_ROOT = ROOT / "docs/audits/next-step-r2-preparation-20260914T143656Z/synthetic_inputs"
KNOWN_E_MANIFEST = KNOWN_E_ROOT / "known_e_worker_input_manifest.json"
KNOWN_E_PATHS = {
    "P2": KNOWN_E_ROOT / "P2_generation_exposure.npz",
    "N2": KNOWN_E_ROOT / "N2_generation_exposure.npz",
}
KNOWN_E_EXPECTED_SHA = {
    "P2": "cd2443191826008385911e9008a0b43deae46fde5322fa13f5d5582c927c73b6",
    "N2": "ca77e1b4f2f7957bc2c3898b9bdd8fbd99a884c62166918a3bd82777ff99c7c9",
}
INPUT_PATH = ROOT / "inputs" / "P9016.snpfree.pairs.gz"
EXPECTED_INPUT_SHA = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"
EXPECTED_REAL_RECORDS = {"raw_records": 1_703_888, "raw_inter": 568_434}
EXPECTED_REAL_CIS_OFFDIAG = {5_000_000: 527_902, 2_000_000: 619_408, 1_000_000: 696_680}
FTOL = 1e-10
CANONICAL_GTOL = 1e-6
MAXLS = 20
CHECKPOINT_EVERY = 10
SELECTION_TOL = 1e-9
FORMAL_OUTPUT_SUGGESTION = ROOT / "test_res" / "038-20260914T143812Z-gpu-m1-formal"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite value cannot enter formal JSON")
        return float(value)
    return value


def _json_bytes(value: Any, indent: int = 2) -> bytes:
    return (json.dumps(_jsonable(value), sort_keys=True, indent=indent, allow_nan=False) + "\n").encode()


def write_json(path: Path, value: Any, exclusive: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    payload = _json_bytes(value)
    with path.open(mode) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _source_hashes() -> dict[str, str]:
    paths = {
        "m1_preconditioner": SOURCE_DIR / "m1_preconditioner.py",
        "m1_preflight": SOURCE_DIR / "m1_preflight.py",
        "m1_controller": SOURCE_DIR / "m1_gpu_controller.py",
        "035_gpu_variant_backend": SOURCE_035 / "gpu_variant_backend.py",
        "035_gpu_multires_controller": SOURCE_035 / "gpu_multires_controller.py",
        "028_gpu_backend": ROOT / "test_res/028-20260913_151456-020-gpu-independent/source/gpu_backend.py",
        "028_fused_objective": ROOT / "test_res/028-20260913_151456-020-gpu-independent/source/fused/fused_objective.py",
        "028_cuda_pair_kernel": ROOT / "test_res/028-20260913_151456-020-gpu-independent/source/fused/cuda_pair_objective.cu",
        "pr_contact_model": ROOT / "pr/contact_model.py",
        "pr_reconstruction_init": ROOT / "pr/reconstruction_init.py",
        "pr_multires_variant_runner": ROOT / "pr/multires_variant_runner.py",
        "pr_v1_calibration": ROOT / "pr/v1_calibration.py",
        "pr_paired_run": ROOT / "pr/paired_run.py",
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def _load_036_reference() -> dict[str, Any]:
    """Read only 036 C0 training stage solver metadata."""
    result: dict[str, Any] = {}
    root = ROOT / "test_res/036-20260914T064651Z-gpu-multires/stages/C0"
    for source in ("consensus_joint", "random_joint"):
        result[source] = {}
        for stage in ("5m", "2m", "1m"):
            path = root / source / (stage + ".json")
            with path.open(encoding="utf-8") as handle:
                row = json.load(handle)
            solver = row["fit"]["solver"]
            result[source][stage] = {
                "path": _relative(ROOT, path),
                "solver_nfev": int(solver["nfev"]),
                "solver_actual_nfev": int(solver["actual_nfev"]),
                "solver_nit": int(solver["nit"]),
                "solver_njev": int(solver.get("njev", 0)),
                "maxiter": int(solver["maxiter"]),
                "maxfun": int(solver["maxfun"]),
                "termination_reason": str(row["fit"]["termination_reason"]),
                "status": str(row["fit"]["status"]),
            }
    return result


def _real_headers() -> tuple[tuple[str, int], ...]:
    identity = contact_model.verify_frozen_snpfree(INPUT_PATH)
    if identity.get("snpfree_sha256") != EXPECTED_INPUT_SHA:
        raise RuntimeError("real input hash mismatch")
    return tuple((str(name), int(length)) for name, length in genome.chrom_lengths(str(INPUT_PATH)))


def _load_real_data(cache: dict[int, Any], bin_size_bp: int):
    bin_size_bp = int(bin_size_bp)
    if bin_size_bp not in cache:
        cache[bin_size_bp] = contact_model.load_frozen_p9016_aggregate(bin_size_bp)
    data = cache[bin_size_bp]
    audit = data.budget()
    for key, expected in EXPECTED_REAL_RECORDS.items():
        if int(audit[key]) != int(expected):
            raise RuntimeError("real %s audit changed: %s" % (key, audit[key]))
    expected_cis = EXPECTED_REAL_CIS_OFFDIAG.get(bin_size_bp)
    if expected_cis is None or int(audit["raw_cis_offdiag"]) != int(expected_cis):
        raise RuntimeError("real raw_cis_offdiag audit changed at %d: %s" % (bin_size_bp, audit["raw_cis_offdiag"]))
    if data.exposure_mode != "observed_endpoint":
        raise RuntimeError("real exposure is not production observed-endpoint exposure")
    return data


def _load_prepared_synthetic(fixture: str) -> tuple[Any, paired_run.PairedStart, dict[str, Any]]:
    fixture = str(fixture)
    manifest_path = PREPARED_RUN / "work" / "prepared_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("optimizer_started") is not False or manifest.get("status") != "prepared_no_optimizer":
        raise RuntimeError("synthetic prepared manifest is not the frozen no-optimizer input")
    matching = [row for row in manifest["runs"] if row.get("fixture_id") == fixture and row.get("model_id") == "C0"]
    if len(matching) != 1:
        raise RuntimeError("missing unique C0 prepared row for %s" % fixture)
    row = matching[0]
    data_path = PREPARED_RUN / "work" / (fixture + "_C0_1mb_layer.npz")
    start_path = PREPARED_RUN / "work" / (fixture + "_shared_start.npz")
    if sha256_file(data_path) != row["data_sha256"]:
        raise RuntimeError("synthetic C0 data hash mismatch for %s" % fixture)
    start = paired_run.load_paired_start(start_path)
    if start.coordinate_sha256 != row["initial_coordinate_sha256"]:
        raise RuntimeError("synthetic start hash mismatch for %s" % fixture)
    data = v1_calibration.load_layer(data_path)
    data.assert_consistent()
    if data.exposure_mode != "observed_endpoint_synthetic_recomputed":
        raise RuntimeError("synthetic production exposure mode changed for %s" % fixture)
    return data, start, {
        "fixture_id": fixture,
        "data_path": _relative(ROOT, data_path),
        "data_sha256": sha256_file(data_path),
        "start_path": _relative(ROOT, start_path),
        "start_sha256": start.coordinate_sha256,
        "prepared_row": row,
    }


def load_known_exposure(path: str | os.PathLike[str], fixture: str, n_loci: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a supplied synthetic-only generation exposure vector, never truth."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("known-generating exposure is unavailable: %s" % path)
    values: np.ndarray
    keys: list[str] = []
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            keys = list(payload.files)
            candidates = [key for key in ("known_exposure", "exposure", "generation_exposure") if key in payload]
            if len(candidates) != 1:
                raise ValueError("known exposure NPZ must have exactly one recognized exposure key")
            values = np.asarray(payload[candidates[0]], dtype=np.float64).copy()
    elif path.suffix.lower() == ".npy":
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).copy()
    else:
        values = np.asarray(np.loadtxt(path, dtype=np.float64), dtype=np.float64)
    values = values.reshape(-1)
    if values.shape != (int(n_loci),) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("known-generating exposure must be positive finite vector of length %d" % n_loci)
    if not np.isclose(float(values.mean()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("known-generating exposure must have mean one")
    return values, {
        "fixture_id": str(fixture),
        "path": str(path),
        "sha256": sha256_file(path),
        "n_loci": int(n_loci),
        "mean": float(values.mean()),
        "recognized_npz_keys": keys,
        "training_use": "synthetic_only_known_generating_exposure",
    }


def _known_data(production_data: Any, exposure: np.ndarray) -> Any:
    audit = production_data.budget()
    return contact_model.synthetic_integer_clone(
        production_data,
        production_data.counts,
        production_data.diag_counts,
        np.asarray(exposure, dtype=np.float64),
        group_totals={"diag": int(audit["aggregate_same_bin"]),
                      "cis_offdiag": int(audit["aggregate_cis_offdiag"]),
                      "inter": int(audit["aggregate_inter"])},
        exposure_mode="known_generating_e",
        endpoint_counts=production_data.endpoint_counts,
    )


def _make_objective(data: Any, method: str, device: str) -> Any:
    if method not in METHODS:
        raise ValueError("unknown method %s" % method)
    base = GPUVariantObjective(data, model_id="C0", device=device, dtype=__import__("torch").float64,
                               use_fused=True, diagnostics=True)
    return base if method == "M0" else M1Objective(base)


def _raw_gradient(objective: Any, theta: np.ndarray) -> tuple[np.ndarray, int]:
    """Return canonical raw-y/q gradient and one explicit validation count."""
    value, gradient, _components = objective.evaluate(theta, need_gradient=True)
    if not math.isfinite(float(value)):
        raise FloatingPointError("endpoint validation returned nonfinite objective")
    gradient = np.asarray(gradient, dtype=np.float64)
    if isinstance(objective, M1Objective):
        gradient = objective.canonical_raw_gradient(theta, gradient)
    return gradient, 1


def _physical_gradient_from_raw(raw_y: np.ndarray, raw_gradient_theta: np.ndarray) -> np.ndarray:
    """Invert the C0 sphere Jacobian for an extra physical-coordinate diagnostic."""
    raw_y = np.asarray(raw_y, dtype=np.float64)
    grad = np.asarray(raw_gradient_theta[:-1], dtype=np.float64).reshape(raw_y.shape)
    one_plus = 1.0 + np.sum(raw_y * raw_y, axis=-1, keepdims=True)
    dot = np.sum(raw_y * grad, axis=-1, keepdims=True)
    physical = np.sqrt(one_plus) * (grad + raw_y * dot)
    if not np.all(np.isfinite(physical)):
        raise FloatingPointError("physical coordinate gradient is nonfinite")
    return physical


def _fit_state(fit: Any) -> tuple[str, str]:
    """Map solver termination to an explicit scientific status."""
    reason = str(fit.terminal_reason)
    canonical_ok = reason == "canonical_gtol" and float(fit.canonical_gradient_max_abs) <= CANONICAL_GTOL
    if canonical_ok:
        return "converged", "canonical_gtol"
    if bool(fit.budget_exhausted) or reason == "fg_budget_exhausted":
        return "budget_not_converged", reason
    if reason == "ftol_numeric_stop":
        return "numeric_stop_not_converged", reason
    if not bool(fit.success):
        return "solver_not_converged", reason
    return "numeric_stop_not_converged", reason


def _completion_gate(real_rows: list[Mapping[str, Any]], synthetic_rows: list[Mapping[str, Any]],
                     attempts: list[Mapping[str, Any]], real_selection: Mapping[str, Any]) -> dict[str, Any]:
    """Require every planned arm/stage and every label-free rescore before release."""
    executed = [row for row in attempts if row.get("status") != "not_run_after_prior_failure"]
    attempt_ids = [str(row.get("attempt_id")) for row in executed]
    stage_records = [str(row.get("stage_record")) for row in executed if row.get("stage_record")]
    executed_statuses = {"converged", "budget_not_converged", "numeric_stop_not_converged", "solver_not_converged"}
    valid_fit_statuses = {"converged", "budget_not_converged", "numeric_stop_not_converged"}

    def endpoint_valid(row: Mapping[str, Any]) -> bool:
        endpoint = row.get("final_coordinates")
        rescore = row.get("selection_rescore")
        status = row.get("optimization", {}).get("terminal_status")
        return (status in valid_fit_statuses and isinstance(endpoint, Mapping)
                and bool(endpoint.get("path")) and bool(endpoint.get("sha256"))
                and isinstance(rescore, Mapping) and rescore.get("criterion") == "count_nll_per_record"
                and isinstance(rescore.get("calls"), int) and rescore.get("calls") >= 1
                and isinstance(row.get("count_nll_per_record"), (int, float))
                and math.isfinite(float(row["count_nll_per_record"])))

    real_valid = [row for row in real_rows if endpoint_valid(row)]
    synthetic_valid = [row for row in synthetic_rows if endpoint_valid(row)]
    groups = real_selection.get("per_method_budget", {}) if isinstance(real_selection, Mapping) else {}
    complete_groups = [name for name, value in groups.items()
                       if value.get("status") == "selected"
                       and len(value.get("candidate_scores", {})) == len(CANDIDATES)]
    nonexecuted = [row for row in attempts if row.get("status") == "not_run_after_prior_failure"]
    duplicate_attempt_ids = sorted({key for key in attempt_ids if attempt_ids.count(key) > 1})
    duplicate_stage_records = sorted({key for key in stage_records if stage_records.count(key) > 1})
    conditions = {
        "planned_stage_attempts_30": len(attempts) == 30,
        "executed_stage_attempts_30": len(executed) == 30,
        "executed_statuses_are_stage_statuses": all(row.get("status") in executed_statuses for row in executed),
        "unique_attempt_ids_30": len(attempt_ids) == 30 and len(set(attempt_ids)) == 30,
        "unique_stage_records_30": len(stage_records) == 30 and len(set(stage_records)) == 30,
        "real_rows_8": len(real_rows) == 8,
        "real_valid_endpoints_8": len(real_valid) == 8,
        "synthetic_rows_6": len(synthetic_rows) == 6,
        "synthetic_valid_endpoints_6": len(synthetic_valid) == 6,
        "rescore_passed_endpoints_14": len(real_valid) + len(synthetic_valid) == 14,
        "real_selection_groups_4": len(complete_groups) == 4,
        "real_selection_has_both_sources": all(
            len(groups[name].get("candidate_scores", {})) == len(CANDIDATES) for name in complete_groups),
        "no_not_run_after_prior_failure": len(nonexecuted) == 0,
    }
    return {
        "status": "release_ready" if all(conditions.values()) else "incomplete_with_failures",
        "conditions": conditions, "all_conditions_pass": all(conditions.values()),
        "planned_stage_attempts": 30, "executed_stage_attempts": len(executed),
        "not_run_after_prior_failure": len(nonexecuted),
        "unique_attempt_ids": len(set(attempt_ids)), "unique_stage_records": len(set(stage_records)),
        "duplicate_attempt_ids": duplicate_attempt_ids, "duplicate_stage_records": duplicate_stage_records,
        "real_valid_endpoint_count": len(real_valid), "synthetic_valid_endpoint_count": len(synthetic_valid),
        "rescore_passed_endpoint_count": len(real_valid) + len(synthetic_valid),
        "real_selection_group_count": len(complete_groups),
        "selection_groups": complete_groups,
        "valid_fit_statuses": sorted(valid_fit_statuses),
        "numeric_stop_counts_as_completed_not_converged": True,
    }


def _solver_elapsed_seconds(root: Path, attempts: list[Mapping[str, Any]]) -> float:
    total = 0.0
    for attempt in attempts:
        record = attempt.get("stage_record")
        if attempt.get("status") == "not_run_after_prior_failure" or not record:
            continue
        path = root / str(record)
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        fit = payload.get("fit", {})
        if isinstance(fit, Mapping) and isinstance(fit.get("elapsed_seconds"), (int, float)):
            total += float(fit["elapsed_seconds"])
    return total


def _write_checkpoint(root: Path, path: Path, payload: Mapping[str, Any], positions: np.ndarray,
                      chromosome_index: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    history = json.dumps(_jsonable(payload["history"]), sort_keys=True, allow_nan=False)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            coordinates=np.asarray(payload["coordinates"], dtype=np.float64),
            theta=np.asarray(payload["theta"], dtype=np.float64),
            y=np.asarray(payload["raw_y"], dtype=np.float64),
            positions=np.asarray(positions, dtype=np.int64),
            chromosome_index=np.asarray(chromosome_index, dtype=np.int64),
            fullhistory_json=np.asarray(history),
        )
        handle.flush()
        os.fsync(handle.fileno())
    return _relative(root, path)


def _fit_stage(root: Path, stage_record_root: Path, method: str, budget_id: str,
               candidate: Mapping[str, Any], stage: Mapping[str, Any], data: Any,
               initial_coordinates: np.ndarray, positions: np.ndarray,
               chromosome_index: np.ndarray, q_in: float | None, device: str) -> tuple[Any, dict[str, Any], np.ndarray]:
    started_wall = time.perf_counter()
    objective = _make_objective(data, method, device)
    raw_initial = objective.raw_from_physical(np.asarray(initial_coordinates, dtype=np.float64))
    cap = int(stage["b1_fg"] if budget_id == "B1" else stage["b2_fg"])
    fit = None
    live_history: list[dict[str, Any]] = []
    checkpoint_paths: list[str] = []

    def accepted(entry: Mapping[str, Any]) -> None:
        live_history.append(_jsonable(dict(entry)))

    def checkpoint(entry: Mapping[str, Any]) -> None:
        history = list(live_history)
        if not history or int(history[-1]["iteration"]) != int(entry["iteration"]):
            history.append({
                "iteration": int(entry["iteration"]), "nfev": int(entry["nfev"]),
                "elapsed_seconds": float(entry["elapsed_seconds"]), "fun": float(entry["fun"]),
                "p": float(entry["p"]), "components": _jsonable(dict(entry["components"])),
                "state_status": "accepted_checkpoint",
            })
        checkpoint_path = stage_record_root / "checkpoints" / (
            "%s-accepted-%04d.npz" % (stage["label"], int(entry["iteration"])))
        checkpoint_paths.append(_write_checkpoint(root, checkpoint_path, {**entry, "history": history},
                                                  positions, chromosome_index))

    fit = run_budgeted_lbfgs(
        objective, raw_initial, p_init=0.75, q_init=q_in,
        maxfun=cap, maxiter=cap + 1, maxls=MAXLS, ftol=FTOL,
        canonical_gtol=CANONICAL_GTOL, checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_hook=checkpoint, accepted_callback=accepted,
    )
    final_coordinates = np.asarray(fit.coordinates, dtype=np.float64)
    objective.validate_physical_coordinates(final_coordinates)
    final_gradient, postfit_validation_calls = _raw_gradient(objective, fit.theta)
    raw_final = np.asarray(fit.y, dtype=np.float64)
    physical_gradient = _physical_gradient_from_raw(raw_final, final_gradient)
    final_diag = warm._map_diagnostics(objective)
    fit_payload = fit.as_dict()
    terminal_status, terminal_status_reason = _fit_state(fit)
    fit_payload.update({
        "terminal_status": terminal_status,
        "terminal_status_reason": terminal_status_reason,
        "method": str(method),
        "budget_id": str(budget_id),
        "budget_unit": "optimizer value+analytic-gradient FG calls; all line-search probes included",
        "budget_cap_fg": cap,
        "initial_q": None if q_in is None else float(q_in),
        "final_q": float(fit.theta[-1]),
        "initial_components": _jsonable(dict(fit.initial_components)),
        "final_components": _jsonable(dict(fit.components)),
        "canonical_gradient": {
            "space": "C0 raw-y/q",
            "inf_norm": float(fit.canonical_gradient_max_abs),
            "l2_norm": float(fit.canonical_gradient_norm),
            "gtol": CANONICAL_GTOL,
        },
        "physical_coordinate_gradient": {
            "space": "C0 physical coordinates x; q omitted",
            "inf_norm": float(np.max(np.abs(physical_gradient), initial=0.0)),
            "l2_norm": float(np.linalg.norm(physical_gradient)),
        },
        "calls": {
            "optimizer_fg_calls": int(fit.nfev),
            "solver_nfev": int(fit.nfev),
            "solver_njev": int(fit.njev),
            "validation_calls_inside_solver": int(fit.validation_calls),
            "postfit_validation_calls": int(postfit_validation_calls),
            "cuda_objective_eval_count": int(final_diag.get("objective_eval_count", 0) or 0),
            "line_search_probes_included_in_optimizer_fg_calls": True,
            "initial_value_validation_excluded_from_optimizer_fg": True,
        },
        "solver": {
            "maxfun": cap,
            "maxiter_internal_guard": cap + 1,
            "maxiter_is_not_a_formal_resource_limit": True,
            "maxls": MAXLS,
            "ftol": FTOL,
            "scipy_gtol": 0.0,
            "canonical_gtol": CANONICAL_GTOL,
            "endpoint_last_accepted": bool(fit.endpoint_was_last_accepted),
        },
        "checkpoint_paths": checkpoint_paths,
        "mapping_diagnostics": final_diag,
        "backend": objective.backend_metadata(),
        "stage_wall_seconds": float(time.perf_counter() - started_wall),
    })
    if fit.budget_exhausted and int(fit.nfev) > cap:
        raise RuntimeError("FG cap was exceeded")
    if not fit.endpoint_was_last_accepted:
        raise RuntimeError("solver endpoint was not a callback-confirmed accepted state")
    if float(fit.fun) > float(fit.initial_total) + FTOL * max(1.0, abs(float(fit.initial_total))):
        raise RuntimeError("same-layer total increased")
    return fit, fit_payload, final_coordinates


def _append_attempt(root: Path, row: Mapping[str, Any]) -> None:
    path = root / "run_status" / "attempts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("at", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(dict(row)), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_coordinates(root: Path, path: Path, data: Any, coordinates: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = paired_run.write_coordinates(path, data, "C0", np.asarray(coordinates, dtype=np.float64))
    info["path"] = _relative(root, path)
    return info


def _new_stage_record(root: Path, path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return _relative(root, path) if write_json(path, payload, exclusive=True) else _relative(root, path)


def _validate_initialization(data: Any, state: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    coords = np.asarray(state["coords"], dtype=np.float64)
    if coords.shape != (2, data.n_loci, 3) or not np.all(np.isfinite(coords)):
        raise RuntimeError("initial coordinates are invalid")
    data_positions = np.concatenate([np.arange(n, dtype=np.int64) * data.bin_size for n in data.n_bins])
    expected_positions = data_positions
    positions = np.asarray(state["positions"], dtype=np.int64)
    chromosome_index = np.asarray(state["chromosome_index"], dtype=np.int64)
    if not np.array_equal(positions, expected_positions):
        raise RuntimeError("initialization is not complete origin-0 grid")
    expected_chromosome = np.repeat(np.arange(len(data.chromosome_names), dtype=np.int64), data.n_bins)
    if not np.array_equal(chromosome_index, expected_chromosome):
        raise RuntimeError("initialization chromosome grid changed")
    if tuple(state["names"]) != tuple(data.chromosome_names):
        raise RuntimeError("initialization names changed")
    if tuple(int(v) for v in state["header_lengths"]) != tuple(int(v) for v in data.chromosome_lengths):
        raise RuntimeError("initialization header lengths changed")
    if int(state["bin_size"]) != int(data.bin_size):
        raise RuntimeError("initialization bin size changed")
    contact_model.assert_inside_unit_ball(coords)
    return coords, positions, chromosome_index, _jsonable(dict(state.get("metadata", {})))


def _initial_state(candidate: Mapping[str, Any], data: Any, previous: Mapping[str, Any] | None,
                   headers: tuple[tuple[str, int], ...]) -> Mapping[str, Any]:
    names = tuple(name for name, _ in headers)
    lengths = tuple(int(length) for _, length in headers)
    if previous is None:
        return reconstruction_init.initialize_approved_candidate(
            str(candidate["initialization_candidate"]), names, lengths, int(data.bin_size))
    return warm.transfer_layer(
        np.asarray(previous["coords"], dtype=np.float64),
        np.asarray(previous["positions"], dtype=np.int64),
        np.asarray(previous["chromosome_index"], dtype=np.int64),
        names, lengths, int(data.bin_size), int(candidate["base_seed"]), "C0",
    )


def _run_real_chain(root: Path, data_cache: dict[int, Any], headers: tuple[tuple[str, int], ...],
                    method: str, budget_id: str, candidate: Mapping[str, Any], device: str) -> dict[str, Any]:
    chain_started = time.perf_counter()
    chain_root = root / "stages" / "real" / method / budget_id / str(candidate["candidate_id"])
    attempts: list[dict[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    q_in: float | None = None
    final_info = None
    final_q = None
    failed = False
    for stage in REAL_STAGES:
        attempt_id = "real-%s-%s-%s-%s" % (method, budget_id, candidate["candidate_id"], stage["label"])
        stage_path = chain_root / (stage["label"] + ".json")
        payload: dict[str, Any] = {
            "attempt_id": attempt_id, "kind": "real", "method": method, "budget_id": budget_id,
            "candidate_id": candidate["candidate_id"], "stage": stage["label"],
            "bin_size_bp": int(stage["bin_size_bp"]), "started_at_utc": _utc_now(),
            "independent_chain": {"from_014_source": True, "resume_from_B1": False,
                                  "resume_from_other_candidate": False},
        }
        if failed:
            payload.update({"status": "not_run_after_prior_failure", "failure_reason": "prior stage failed",
                            "completed_at_utc": _utc_now()})
            record_path = _new_stage_record(root, stage_path, payload)
            attempts.append({"attempt_id": attempt_id, "status": "not_run_after_prior_failure",
                             "is_terminal": False, "stage_record": record_path})
            continue
        try:
            data = _load_real_data(data_cache, int(stage["bin_size_bp"]))
            state = _initial_state(candidate, data, previous, headers)
            initial_coordinates, positions, chromosome_index, init_meta = _validate_initialization(data, state)
            initial_path = chain_root / "coords" / ("initial-%s.3dg" % stage["label"])
            final_path = chain_root / "coords" / ("final-%s.3dg" % stage["label"])
            initial_info = _write_coordinates(root, initial_path, data, initial_coordinates)
            fit, fit_payload, final_coordinates = _fit_stage(
                root, chain_root, method, budget_id, candidate, stage, data,
                initial_coordinates, positions, chromosome_index, q_in, device)
            final_info = _write_coordinates(root, final_path, data, final_coordinates)
            payload.update({
                "status": str(fit_payload["terminal_status"]),
                "completed_at_utc": _utc_now(), "data_budget": data.budget(),
                "initialization": init_meta,
                "initial_coordinates": initial_info,
                "final_coordinates": final_info,
                "fit": fit_payload,
            })
            record_path = _new_stage_record(root, stage_path, payload)
            attempts.append({"attempt_id": attempt_id, "status": payload["status"], "is_terminal": stage["label"] == "1m",
                             "stage_record": record_path, "budget_exhausted": bool(fit.budget_exhausted),
                             "terminal_reason": fit_payload["terminal_status_reason"],
                             "solver_nfev": int(fit.nfev), "solver_nit": int(fit.nit)})
            previous = {"coords": final_coordinates, "positions": positions, "chromosome_index": chromosome_index}
            q_in = float(fit.theta[-1])
            if stage["label"] == "1m":
                final_q = q_in
        except Exception as exc:
            failed = True
            reason = "%s: %s" % (type(exc).__name__, exc)
            payload.update({"status": "failed", "failed_at_utc": _utc_now(), "failure_reason": reason,
                            "traceback": traceback.format_exc(limit=20)})
            record_path = _new_stage_record(root, stage_path, payload)
            attempts.append({"attempt_id": attempt_id, "status": "failed", "is_terminal": True,
                             "stage_record": record_path, "failure_reason": reason})
        _append_attempt(root, {"attempt_id": attempt_id, "kind": "real", "method": method,
                               "budget_id": budget_id, "candidate_id": candidate["candidate_id"],
                               "stage": stage["label"], "status": attempts[-1]["status"]})
    if failed or final_info is None or final_q is None:
        return {"kind": "real", "method": method, "budget_id": budget_id,
                "candidate_id": candidate["candidate_id"], "source": candidate["initialization_candidate"],
                "base_seed": int(candidate["base_seed"]), "attempts": attempts,
                "optimization": {"terminal_status": "failed"}, "failure_reason": next(
                    (row.get("failure_reason") for row in attempts if row["status"] == "failed"),
                    "chain failed before final 1m"), "wall_seconds": time.perf_counter() - chain_started}
    attempts[-1]["is_terminal"] = True
    return {"kind": "real", "method": method, "budget_id": budget_id,
            "candidate_id": candidate["candidate_id"], "source": candidate["initialization_candidate"],
            "base_seed": int(candidate["base_seed"]), "attempts": attempts,
            "final_coordinates": final_info, "final_q": float(final_q),
            "optimization": {"terminal_status": attempts[-1]["status"],
                             "terminal_attempt_id": attempts[-1]["attempt_id"]},
            "wall_seconds": time.perf_counter() - chain_started}


def _read_back_coordinates(root: Path, info: Mapping[str, Any], data: Any) -> np.ndarray:
    path = root / str(info["path"])
    if sha256_file(path) != info["sha256"]:
        raise RuntimeError("coordinate hash changed before rescore: %s" % path)
    return warm._read_model_coordinates(path, data, "C0", expected_sha=str(info["sha256"]))


def _rescore_real(root: Path, data: Any, row: dict[str, Any], device: str) -> dict[str, Any]:
    if row.get("optimization", {}).get("terminal_status") == "failed":
        return row
    base = GPUVariantObjective(data, model_id="C0", device=device, dtype=__import__("torch").float64,
                               use_fused=True, diagnostics=True)
    coordinates = _read_back_coordinates(root, row["final_coordinates"], data)
    raw = base.raw_from_physical(coordinates)
    theta = base.pack(raw, p=0.75)
    theta[-1] = float(row["final_q"])
    started = time.perf_counter()
    total, _, components = base.evaluate(theta, need_gradient=False)
    row["count_nll_per_record"] = float(components["count_nll_normalized"])
    row["count_model"] = {key: float(components[key]) for key in (
        "conditional_nll_raw", "diag_profiled_nll_raw", "count_nll_raw", "count_nll_normalized")}
    row["selection_rescore"] = {
        "criterion": "count_nll_per_record", "selection_uses_priors": False,
        "coordinate_readback": True, "q_fixed_from_final_fit": float(row["final_q"]),
        "calls": 1, "total": float(total), "components": _jsonable(dict(components)),
        "wall_seconds": float(time.perf_counter() - started), "reference_used": False, "phase_used": False,
    }
    return row


def _select_real(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    by_group: dict[str, dict[str, Any]] = {}
    selected_files: dict[str, Any] = {}
    for method in METHODS:
        for budget in BUDGETS:
            group = "%s__%s" % (method, budget)
            expected_candidates = {candidate["candidate_id"] for candidate in CANDIDATES}
            usable = [row for row in rows if row.get("method") == method and row.get("budget_id") == budget
                      and row.get("candidate_id") in expected_candidates
                      and row.get("count_nll_per_record") is not None
                      and row.get("selection_rescore", {}).get("criterion") == "count_nll_per_record"
                      and row.get("optimization", {}).get("terminal_status") in
                      ("converged", "budget_not_converged", "numeric_stop_not_converged")]
            if len(usable) != len(CANDIDATES) or {row["candidate_id"] for row in usable} != expected_candidates:
                by_group[group] = {
                    "status": "incomplete_missing_source", "criterion": "count_nll_per_record",
                    "expected_candidate_ids": sorted(expected_candidates),
                    "rescored_candidate_ids": sorted(row["candidate_id"] for row in usable),
                    "candidate_statuses": {row["candidate_id"]: row.get("optimization", {}).get("terminal_status")
                                           for row in rows if row.get("method") == method and row.get("budget_id") == budget},
                }
                continue
            minimum = min(float(row["count_nll_per_record"]) for row in usable)
            eligible = [row for row in usable if float(row["count_nll_per_record"]) <= minimum + SELECTION_TOL]
            order = {candidate["candidate_id"]: index for index, candidate in enumerate(CANDIDATES)}
            selected = min(eligible, key=lambda row: order[str(row["candidate_id"])])
            target = root / "selected" / (method + "__" + budget + ".3dg")
            target.parent.mkdir(parents=True, exist_ok=True)
            with (root / selected["final_coordinates"]["path"]).open("rb") as source, target.open("xb") as dest:
                shutil.copyfileobj(source, dest)
                dest.flush()
                os.fsync(dest.fileno())
            target_sha = sha256_file(target)
            selected_files[group] = {"path": _relative(root, target), "sha256": target_sha,
                                     "source_candidate_id": selected["candidate_id"],
                                     "source_coordinate_path": selected["final_coordinates"]["path"],
                                     "source_coordinate_sha256": selected["final_coordinates"]["sha256"]}
            by_group[group] = {
                "status": "selected", "criterion": "count_nll_per_record", "direction": "minimize",
                "tie_tolerance": SELECTION_TOL, "tie_order": [x["candidate_id"] for x in CANDIDATES],
                "selected_candidate_id": selected["candidate_id"],
                "candidate_scores": {row["candidate_id"]: row.get("count_nll_per_record") for row in usable},
                "reference_used": False, "phase_used": False,
            }
    return {"per_method_budget": by_group, "selected_files": selected_files}


def _run_synthetic_arm(root: Path, fixture: str, method: str, exposure_label: str,
                       data: Any, start: paired_run.PairedStart, device: str,
                       input_meta: Mapping[str, Any]) -> dict[str, Any]:
    arm_root = root / "stages" / "synthetic" / fixture / method / exposure_label
    stage = {"label": "1m", "bin_size_bp": 1_000_000, "b1_fg": 243, "b2_fg": 486}
    candidate = {"candidate_id": fixture, "initialization_candidate": "prepared_shared_start", "base_seed": 0}
    attempt_id = "synthetic-%s-%s-%s-1m" % (fixture, method, exposure_label)
    stage_path = arm_root / "1m.json"
    payload: dict[str, Any] = {"attempt_id": attempt_id, "kind": "synthetic", "fixture_id": fixture,
                               "method": method, "exposure_label": exposure_label, "stage": "1m",
                               "bin_size_bp": 1_000_000, "started_at_utc": _utc_now(),
                               "input": dict(input_meta), "source_start_id": start.start_id,
                               "no_source_selection": True, "no_r2_selection": True}
    try:
        initial_coordinates = np.asarray(start.coordinates, dtype=np.float64)
        positions = np.concatenate([np.arange(n, dtype=np.int64) * data.bin_size for n in data.n_bins])
        chromosome_index = np.repeat(np.arange(len(data.chromosome_names), dtype=np.int64), data.n_bins)
        initial_path = arm_root / "coords" / "initial-1m.3dg"
        final_path = arm_root / "coords" / "final-1m.3dg"
        initial_info = _write_coordinates(root, initial_path, data, initial_coordinates)
        fit, fit_payload, final_coordinates = _fit_stage(
            root, arm_root, method, "B1", candidate, stage, data,
            initial_coordinates, positions, chromosome_index, float(contact_model.q_from_p(start.p_init)), device)
        final_info = _write_coordinates(root, final_path, data, final_coordinates)
        payload.update({"status": str(fit_payload["terminal_status"]),
                        "completed_at_utc": _utc_now(), "data_budget": data.budget(),
                        "initialization": start.as_dict(), "initial_coordinates": initial_info,
                        "final_coordinates": final_info, "final_q": float(fit.theta[-1]), "fit": fit_payload})
        record_path = _new_stage_record(root, stage_path, payload)
        _append_attempt(root, {"attempt_id": attempt_id, "kind": "synthetic", "fixture_id": fixture,
                               "method": method, "exposure_label": exposure_label, "stage": "1m",
                               "status": payload["status"]})
        return {"kind": "synthetic", "fixture_id": fixture, "method": method,
                "exposure_label": exposure_label, "attempts": [{"attempt_id": attempt_id,
                "status": payload["status"], "is_terminal": True, "stage_record": record_path,
                "solver_nfev": int(fit.nfev), "solver_nit": int(fit.nit)}],
                "final_coordinates": final_info, "final_q": float(fit.theta[-1]),
                "optimization": {"terminal_status": payload["status"], "terminal_attempt_id": attempt_id},
                "input": dict(input_meta)}
    except Exception as exc:
        reason = "%s: %s" % (type(exc).__name__, exc)
        payload.update({"status": "failed", "failed_at_utc": _utc_now(), "failure_reason": reason,
                        "traceback": traceback.format_exc(limit=20)})
        record_path = _new_stage_record(root, stage_path, payload)
        _append_attempt(root, {"attempt_id": attempt_id, "kind": "synthetic", "fixture_id": fixture,
                               "method": method, "exposure_label": exposure_label, "stage": "1m",
                               "status": "failed"})
        return {"kind": "synthetic", "fixture_id": fixture, "method": method,
                "exposure_label": exposure_label, "attempts": [{"attempt_id": attempt_id,
                "status": "failed", "is_terminal": True, "stage_record": record_path,
                "failure_reason": reason}], "optimization": {"terminal_status": "failed",
                "terminal_attempt_id": attempt_id}, "failure_reason": reason, "input": dict(input_meta)}


def _rescore_synthetic(root: Path, row: dict[str, Any], data: Any, device: str) -> dict[str, Any]:
    if row.get("optimization", {}).get("terminal_status") == "failed":
        return row
    base = GPUVariantObjective(data, model_id="C0", device=device, dtype=__import__("torch").float64,
                               use_fused=True, diagnostics=True)
    coordinates = _read_back_coordinates(root, row["final_coordinates"], data)
    raw = base.raw_from_physical(coordinates)
    theta = base.pack(raw, p=0.75)
    theta[-1] = float(row["final_q"])
    total, _, components = base.evaluate(theta, need_gradient=False)
    row["count_nll_per_record"] = float(components["count_nll_normalized"])
    row["count_model"] = {key: float(components[key]) for key in (
        "conditional_nll_raw", "diag_profiled_nll_raw", "count_nll_raw", "count_nll_normalized")}
    row["selection_rescore"] = {"criterion": "count_nll_per_record", "selection_uses_priors": False,
                                 "coordinate_readback": True, "calls": 1, "total": float(total),
                                 "reference_used": False, "phase_used": False}
    return row


def _formal_protocol(preflight_sha: str) -> dict[str, Any]:
    refs = _load_036_reference()
    b1 = {stage["label"]: int(stage["b1_fg"]) for stage in REAL_STAGES}
    b2 = {stage["label"]: int(stage["b2_fg"]) for stage in REAL_STAGES}
    return {
        "schema": "gpu-m1-preconditioner-protocol-v1", "status": "frozen",
        "created_at_utc": _utc_now(), "scope": "phase-free C0 training only",
        "input": {"path": str(INPUT_PATH), "sha256": EXPECTED_INPUT_SHA,
                  "raw_records": 1_703_888, "intra_records": 1_135_454, "inter_records": 568_434,
                  "training_columns": ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"],
                  "observational_unit": "bin-pair count", "samebin_nuisance": "retained",
                  "fullgrid_zero_pairs": True, "no_folds": True},
        "initialization": {
            "source_run": "test_res/014-20260912_153000-s0-genome-wide-fixed",
            "sources": {candidate["candidate_id"]: {
                "initialization_candidate": candidate["initialization_candidate"],
                "base_seed": candidate["base_seed"],
                "path": str(reconstruction_init.APPROVED_SOURCES[candidate["initialization_candidate"]]["path"]),
                "sha256": reconstruction_init.APPROVED_SOURCES[candidate["initialization_candidate"]]["sha256"],
            } for candidate in CANDIDATES},
            "first_layer": "036 center/maxR .8, interpolation/jitter/seed/p=.75/q conversion retained",
            "warm_transfer": "036 exact numeric genomic interpolation + endpoint fill + jitter; preserve frame/scale",
            "q_carry": "exact raw q between 5m->2m->1m; first layer p=0.75",
            "b2_independent": "fresh 014 source and fresh optimizer state; never B1 endpoint/checkpoint/L-BFGS history",
        },
        "physical_model": {
            "model_id": "C0", "count_weight": 1.0, "p_prior_weight": 1.0,
            "bond_weight": 1.0, "repulsion_weight": 1.0, "bend_weight": 0.01,
            "repulsion_threshold": "0.7*l0", "finite_kernel": {"epsilon": 1e-6, "r0": "2*l0", "tail": 4},
            "p_init": 0.75, "p_floor": 1e-4, "p_prior_strength": 1e-4,
            "sphere": "x=y/sqrt(1+||y||^2)", "exposure": "sqrt(endpoint+10)/full-grid mean",
            "selection_count_only": True, "selection_includes_samebin_profile": True,
        },
        "method": {
            "M0": {"optimizer_coordinates": "raw-y/q", "description": "C0 raw-y L-BFGS control"},
            "M1": {"optimizer_coordinates": "a,v,q", "a": "(yA+yB)/sqrt2",
                   "v": "(yA-yB)/(2*sqrt2)", "reconstruction": "yA=(a+2v)/sqrt2; yB=(a-2v)/sqrt2",
                   "gradient": "g_a=(gA+gB)/sqrt2; g_v=2*(gA-gB)/sqrt2; g_q unchanged",
                   "fixed_scale": 2.0, "physical_objective_unchanged": True,
                   "no_force_separation": True, "no_consensus_freeze": True,
                   "no_sphere_domain_kernel_prior_change": True},
            "scale_1": "orthogonal transform is mathematical precheck only; not a formal arm",
        },
        "budget": {
            "reference_036_C0": refs,
            "unit": "optimizer value+analytic-gradient request (FG); all line-search/direction calls included",
            "B1_fg_cap": b1, "B2_fg_cap": b2, "synthetic_1m_fg_cap": 243,
            "B2_exactly_2x_B1": True, "baseline_nit_not_budget": True,
            "maxiter_internal": "cap+1 only, non-binding guard; no 036 300/200/240 accepted-iteration cap",
            "cap_behavior": "raise before B+1 request; freeze last callback-confirmed accepted state; discard trial",
            "validation_rescore_independent": True,
        },
        "stopping": {
            "ftol": FTOL, "maxls": MAXLS, "scipy_gtol": 0.0,
            "canonical_space": "C0 raw-y/q", "canonical_gtol_inf_norm": CANONICAL_GTOL,
            "physical_gradient": "diagnostic only; not stopping or selection",
            "ftol_without_canonical_gtol": "numeric_stop_only, never gradient_converged",
            "budget_terminal_reason": "fg_budget_exhausted",
            "accepted_endpoint_required": True,
        },
        "synthetic_worker_inputs": {
            "worker_manifest": {"path": _relative(ROOT, KNOWN_E_MANIFEST), "sha256": sha256_file(KNOWN_E_MANIFEST)},
            "known_exposure": {fixture: {"path": _relative(ROOT, path), "sha256": KNOWN_E_EXPECTED_SHA[fixture]}
                               for fixture, path in KNOWN_E_PATHS.items()},
            "truth_payloads_opened": False,
        },
        "matrix": {
            "real": [{"method": method, "budget_id": budget, "candidate_ids": [c["candidate_id"] for c in CANDIDATES],
                      "layers": [s["label"] for s in REAL_STAGES], "independent_chain": True}
                     for method in METHODS for budget in BUDGETS],
            "synthetic": [{"fixture": fixture, "arms": ["M0/production-e", "M1/production-e", "M0/known-generating-e"],
                           "budget": "B1/1m=243", "one_prepared_start": True, "r2_selection": False}
                          for fixture in SYNTHETIC_FIXTURES],
            "real_trajectory_count": 8, "synthetic_arm_count": 6,
        },
        "selection": {"real": "within each method x budget, min final count_nll_per_record over consensus then random tie order; tie=1e-9",
                       "synthetic": "one existing start per arm; no source/R2 selection", "reference_used": False, "phase_used": False},
        "backend": {"device": "cuda required", "dtype": "float64", "mixed_precision": False,
                     "sampling": False, "implementation": "035 adapter + immutable 028 fused ordered-row CUDA",
                     "full_grid": True, "all_pairs": True, "rawq": True},
        "source_code_sha256": _source_hashes(),
        "preflight": {"path": "preflight.json", "sha256": preflight_sha},
        "formal_output_suggestion": str(FORMAL_OUTPUT_SUGGESTION),
        "training_boundary": {"phase_used": False, "reference_used": False,
                              "synthetic_truth_opened": False, "real_reference_opened": False,
                              "evaluation_r2_opened": False, "native_fdg_started": False},
        "release": {"status": "parent_method_accepted", "endpoint_hash_before_evaluation": True},
    }


def prepare(preflight_path: Path) -> dict[str, Any]:
    if preflight_path.resolve() != (ARTIFACT / "preflight.json").resolve():
        raise RuntimeError("preflight must be the new M1 preparation artifact")
    with preflight_path.open(encoding="utf-8") as handle:
        preflight = json.load(handle)
    if preflight.get("status") != "passed":
        raise RuntimeError("M1 preflight is not passed")
    protocol = _formal_protocol(sha256_file(preflight_path))
    protocol_sha = write_json(ARTIFACT / "protocol.json", protocol)
    config = {
        "schema": "gpu-m1-preconditioner-preflight-config-v1",
        "protocol_sha256": protocol_sha, "preflight_sha256": sha256_file(preflight_path),
        "status": "prepared_pending_formal_launch", "formal_output": str(FORMAL_OUTPUT_SUGGESTION),
        "real_trajectory_count": 8, "synthetic_arm_count": 6,
        "budget": protocol["budget"], "stopping": protocol["stopping"],
        "source_code_sha256": protocol["source_code_sha256"],
        "input_sha256": EXPECTED_INPUT_SHA,
        "training_boundary": protocol["training_boundary"],
        "formal_training_started": False,
    }
    config_sha = write_json(ARTIFACT / "config.json", config)
    source_manifest = {"schema": "gpu-m1-source-hashes-v1", "source_code_sha256": _source_hashes(),
                       "protocol_sha256": protocol_sha, "preflight_sha256": sha256_file(preflight_path),
                       "input_sha256": EXPECTED_INPUT_SHA}
    source_sha = write_json(ARTIFACT / "provenance" / "source_hashes.json", source_manifest)
    terminal = {"schema": "gpu-m1-preparation-terminal-v1", "status": "prepared_pending_formal_launch",
                "formal_training_started": False, "optimizer_calls_in_preparation": int(preflight.get("tests", {}).get("budget", {}).get("optimizer_calls", 0)),
                "preflight_sha256": sha256_file(preflight_path), "protocol_sha256": protocol_sha,
                "config_sha256": config_sha, "source_manifest_sha256": source_sha,
                "training_boundary": protocol["training_boundary"]}
    terminal_sha = write_json(ARTIFACT / "terminal_prep.json", terminal)
    readme = """# 037 M1 Preflight\n\nThis is a preparation-only record for the frozen M0/M1 C0 comparison. The\npreflight uses only the SNP-free input, approved 014 blind starts, 036 C0\ntraining metadata, and synthetic unlabeled counts/starts. No formal P9016\ntrajectory was launched here.\n\nFormal output is reserved at `test_res/038-20260914T143812Z-gpu-m1-formal`.\nThe accepted-state FG hard stop is implemented in\n`source/m1_preconditioner.py::run_budgeted_lbfgs`.\n"""
    (ARTIFACT / "README.md").write_text(readme, encoding="utf-8")
    return {"status": "prepared", "artifact": str(ARTIFACT), "protocol_sha256": protocol_sha,
            "config_sha256": config_sha, "source_manifest_sha256": source_sha, "terminal_sha256": terminal_sha}


def run_formal(output: Path, prep_root: Path, known_exposure_paths: Mapping[str, Path], device: str = "cuda") -> dict[str, Any]:
    """Run all frozen formal arms serially on one CUDA device."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("formal output must be fresh: %s" % output)
    run_started_utc = _utc_now()
    run_started_perf = time.perf_counter()
    run_started_perf_counter = run_started_perf
    output.mkdir(parents=True, exist_ok=False)
    for relative in ("coords", "checkpoints", "logs", "stages", "selected", "run_status", "provenance"):
        (output / relative).mkdir(parents=True, exist_ok=False)
    prep_root = prep_root.resolve()
    protocol_path = prep_root / "protocol.json"
    config_path = prep_root / "config.json"
    preflight_path = prep_root / "preflight.json"
    with protocol_path.open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    with config_path.open(encoding="utf-8") as handle:
        prep_config = json.load(handle)
    with preflight_path.open(encoding="utf-8") as handle:
        preflight = json.load(handle)
    if protocol.get("status") != "frozen" or preflight.get("status") != "passed":
        raise RuntimeError("formal launch requires frozen protocol and passed preflight")
    current_hashes = _source_hashes()
    if current_hashes != protocol["source_code_sha256"]:
        raise RuntimeError("source changed after preparation; refuse formal launch")
    if sha256_file(INPUT_PATH) != EXPECTED_INPUT_SHA:
        raise RuntimeError("real input changed before formal launch")
    expected_synthetic_inputs = protocol["synthetic_worker_inputs"]
    if sha256_file(KNOWN_E_MANIFEST) != expected_synthetic_inputs["worker_manifest"]["sha256"]:
        raise RuntimeError("known-e worker manifest changed before formal launch")
    for fixture, expected in expected_synthetic_inputs["known_exposure"].items():
        supplied = Path(known_exposure_paths[fixture]).resolve()
        if sha256_file(supplied) != expected["sha256"]:
            raise RuntimeError("known-e exposure hash mismatch before formal launch for %s" % fixture)
    probe = cuda_probe()
    if probe.get("status") != "available":
        raise RuntimeError("CUDA is unavailable at formal launch")
    with (output / "provenance" / "protocol.json").open("xb") as dest, protocol_path.open("rb") as source:
        shutil.copyfileobj(source, dest)
    with (output / "provenance" / "preflight.json").open("xb") as dest, preflight_path.open("rb") as source:
        shutil.copyfileobj(source, dest)
    write_json(output / "config.json", {
        "schema": "gpu-m1-formal-run-config-v1", "status": "running", "created_at_utc": run_started_utc,
        "run_started_utc": run_started_utc, "run_started_perf_counter": run_started_perf_counter,
        "protocol_sha256": sha256_file(protocol_path), "preflight_sha256": sha256_file(preflight_path),
        "prepared_config_sha256": sha256_file(config_path), "source_code_sha256": current_hashes,
        "input": {"path": str(INPUT_PATH), "sha256": EXPECTED_INPUT_SHA, "uses_all_records": True},
        "device": device, "dtype": "float64", "matrix": protocol["matrix"],
        "training_boundary": protocol["training_boundary"],
        "known_exposure_paths": {fixture: str(Path(path).resolve()) for fixture, path in known_exposure_paths.items()},
    })
    write_json(output / "provenance" / "source_hashes.json", {
        "schema": "gpu-m1-formal-source-hashes-v1", "source_code_sha256": current_hashes,
        "protocol_sha256": sha256_file(protocol_path), "preflight_sha256": sha256_file(preflight_path),
        "input_sha256": EXPECTED_INPUT_SHA,
    })
    write_json(output / "runtime_probe.json", {"status": "passed", "recorded_at_utc": _utc_now(),
                                                 "cuda": probe, "dtype": "float64", "device": device,
                                                 "training_boundary": protocol["training_boundary"]})
    write_json(output / "run_status" / "run_status.json", {"status": "training", "started_at_utc": run_started_utc,
                                                               "run_started_utc": run_started_utc,
                                                               "run_started_perf_counter": run_started_perf_counter})
    headers = _real_headers()
    data_cache: dict[int, Any] = {}
    real_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for budget in BUDGETS:
            for candidate in CANDIDATES:
                real_rows.append(_run_real_chain(output, data_cache, headers, method, budget, candidate, device))
    final_real_data = _load_real_data(data_cache, 1_000_000)
    finalized_real: list[dict[str, Any]] = []
    for row in real_rows:
        try:
            finalized_real.append(_rescore_real(output, final_real_data, row, device))
        except Exception as exc:
            row = dict(row)
            row.update({"optimization": {"terminal_status": "failed"},
                       "failure_reason": "final_count_rescore: %s: %s" % (type(exc).__name__, exc)})
            finalized_real.append(row)
    synthetic_rows: list[dict[str, Any]] = []
    synthetic_inputs: dict[str, Any] = {}
    for fixture in SYNTHETIC_FIXTURES:
        production_data, start, input_meta = _load_prepared_synthetic(fixture)
        known_path = Path(known_exposure_paths[fixture]).resolve()
        known_exposure, known_meta = load_known_exposure(known_path, fixture, production_data.n_loci)
        known_data = _known_data(production_data, known_exposure)
        synthetic_inputs[fixture] = {"production": input_meta, "known_exposure": known_meta,
                                     "known_data_exposure_mode": known_data.exposure_mode}
        for method, exposure_label, data in (("M0", "production-e", production_data),
                                              ("M1", "production-e", production_data),
                                              ("M0", "known-generating-e", known_data)):
            synthetic_rows.append(_run_synthetic_arm(output, fixture, method, exposure_label,
                                                     data, start, device,
                                                     {**input_meta, "exposure": synthetic_inputs[fixture]["known_exposure"]
                                                      if exposure_label == "known-generating-e" else {
                                                          "mode": "production_observed_endpoint_synthetic_recomputed"}}))
    finalized_synth: list[dict[str, Any]] = []
    for row in synthetic_rows:
        fixture = str(row["fixture_id"])
        exposure_label = str(row["exposure_label"])
        base_data, _start, _meta = _load_prepared_synthetic(fixture)
        if exposure_label == "known-generating-e":
            known_exposure, _ = load_known_exposure(Path(known_exposure_paths[fixture]), fixture, base_data.n_loci)
            score_data = _known_data(base_data, known_exposure)
        else:
            score_data = base_data
        try:
            finalized_synth.append(_rescore_synthetic(output, row, score_data, device))
        except Exception as exc:
            row = dict(row)
            row.update({"optimization": {"terminal_status": "failed"},
                       "failure_reason": "final_count_rescore: %s: %s" % (type(exc).__name__, exc)})
            finalized_synth.append(row)
    run_end_utc = _utc_now()
    run_end_perf_counter = time.perf_counter()
    run_wall_seconds = float(run_end_perf_counter - run_started_perf)
    real_selection = _select_real(finalized_real, output)
    attempts = [attempt for row in finalized_real + finalized_synth for attempt in row.get("attempts", [])]
    completion = _completion_gate(finalized_real, finalized_synth, attempts, real_selection)
    run_status = str(completion["status"])
    solver_elapsed_seconds = _solver_elapsed_seconds(output, attempts)
    selection = {
        "schema": "gpu-m1-formal-selection-v1", "status": run_status,
        "created_at_utc": run_end_utc, "run_started_utc": run_started_utc,
        "run_started_perf_counter": run_started_perf_counter, "run_end_utc": run_end_utc,
        "run_end_perf_counter": run_end_perf_counter, "real": real_selection, "synthetic": {
            "status": "no_selection_one_start_per_arm", "arms": [
                {"fixture_id": row["fixture_id"], "method": row["method"],
                 "exposure_label": row["exposure_label"], "count_nll_per_record": row.get("count_nll_per_record"),
                 "coordinate_path": None if "final_coordinates" not in row else row["final_coordinates"]["path"],
                 "coordinate_sha256": None if "final_coordinates" not in row else row["final_coordinates"]["sha256"],
                 "optimization_status": row.get("optimization", {}).get("terminal_status")}
                for row in finalized_synth]},
        "candidates": finalized_real, "synthetic_arms": finalized_synth,
        "attempts": attempts, "completion_gate": completion, "selection_uses_priors": False,
        "reference_used": False, "phase_used": False, "r2_used": False,
        "training_boundary": protocol["training_boundary"],
    }
    selection_sha = write_json(output / "selection.json", selection)
    endpoint_rows = finalized_real + finalized_synth
    endpoints = []
    for row in endpoint_rows:
        if row.get("final_coordinates"):
            endpoints.append({"kind": row["kind"], "method": row.get("method"), "budget_id": row.get("budget_id"),
                              "fixture_id": row.get("fixture_id"), "candidate_id": row.get("candidate_id"),
                              "exposure_label": row.get("exposure_label"),
                              "path": row["final_coordinates"]["path"], "sha256": row["final_coordinates"]["sha256"],
                              "final_q": row.get("final_q"), "count_nll_per_record": row.get("count_nll_per_record"),
                              "optimization_status": row.get("optimization", {}).get("terminal_status")})
    manifest = {
        "schema": "gpu-m1-release-ready-manifest-v1", "status": run_status,
        "created_at_utc": run_end_utc, "run_started_utc": run_started_utc,
        "run_started_perf_counter": run_started_perf_counter, "run_end_utc": run_end_utc,
        "run_end_perf_counter": run_end_perf_counter,
        "run_wall_seconds": run_wall_seconds, "solver_elapsed_seconds": solver_elapsed_seconds,
        "protocol_sha256": sha256_file(protocol_path), "preflight_sha256": sha256_file(preflight_path),
        "source_code_sha256": current_hashes, "input_sha256": EXPECTED_INPUT_SHA,
        "device_probe": probe, "dtype": "float64", "endpoints": endpoints,
        "endpoint_count_expected": 14, "endpoint_count_present": len(endpoints),
        "completion_gate": completion, "real_selection": real_selection, "synthetic_inputs": synthetic_inputs,
        "endpoint_hash_before_evaluation": True, "reference_used": False, "phase_used": False,
        "evaluation_r2_opened": False, "training_boundary": protocol["training_boundary"],
    }
    manifest_sha = write_json(output / "release_ready_manifest.json", manifest)
    report = {
        "schema": "gpu-m1-training-report-v1", "status": run_status,
        "run_started_utc": run_started_utc, "run_started_perf_counter": run_started_perf_counter,
        "run_end_utc": run_end_utc, "run_end_perf_counter": run_end_perf_counter,
        "run_wall_seconds": run_wall_seconds, "solver_elapsed_seconds": solver_elapsed_seconds,
        "summary": {"real_trajectories": 8, "synthetic_arms": 6, "real_stage_attempts": 24,
                    "synthetic_stage_attempts": 6, "executed_stage_attempts": completion["executed_stage_attempts"],
                    "all_endpoints_saved": len(endpoints) == 14, "completion_gate": completion,
                    "selection": "real method x budget by final count_nll_per_record; synthetic no selection"},
        "real": finalized_real, "synthetic": finalized_synth,
        "training_boundary": protocol["training_boundary"], "reference_used": False, "phase_used": False,
    }
    report_sha = write_json(output / "report.json", report)
    markdown = "# M1 formal training report\\n\\nTraining-only endpoints and label-free count selection are recorded in `report.json`, `selection.json`, and `release_ready_manifest.json`. Reference/phase evaluation was not opened by this runner.\\n"
    with (output / "report.md").open("xb") as handle:
        handle.write(markdown.encode())
        handle.flush()
        os.fsync(handle.fileno())
    completed_statuses = {"converged", "budget_not_converged", "numeric_stop_not_converged", "solver_not_converged"}
    terminal = {
        "schema": "gpu-m1-terminal-evidence-v1", "status": run_status,
        "runner_exit_code": 0 if completion["all_conditions_pass"] else 1,
        "run_started_utc": run_started_utc, "run_started_perf_counter": run_started_perf_counter,
        "run_end_utc": run_end_utc, "run_end_perf_counter": run_end_perf_counter,
        "run_wall_seconds": run_wall_seconds, "solver_elapsed_seconds": solver_elapsed_seconds,
        "planned_real_trajectory_count": 8, "planned_synthetic_arm_count": 6,
        "planned_stage_attempt_count": 30, "executed_stage_attempt_count": completion["executed_stage_attempts"],
        "completed_stage_attempt_count": sum(1 for x in attempts if x.get("status") in completed_statuses),
        "failed_stage_attempt_count": sum(1 for x in attempts if x.get("status") == "failed"),
        "nonconverged_stage_attempt_count": sum(1 for x in attempts if x.get("status") in completed_statuses - {"converged"}),
        "attempts": attempts, "completion_gate": completion,
        "selection_sha256": selection_sha, "release_manifest_sha256": manifest_sha,
        "report_sha256": report_sha, "all_endpoints_saved": len(endpoints) == 14,
        "training_boundary": protocol["training_boundary"],
    }
    terminal_sha = write_json(output / "terminal_evidence.json", terminal)
    exit_code = 0 if completion["all_conditions_pass"] else 1
    receipt = {"schema": "gpu-m1-formal-run-receipt-v1", "status": run_status,
               "runner_exit_code": exit_code, "outdir": str(output),
               "run_started_utc": run_started_utc, "run_started_perf_counter": run_started_perf_counter,
                "run_end_utc": run_end_utc, "run_end_perf_counter": run_end_perf_counter,
               "run_wall_seconds": run_wall_seconds, "solver_elapsed_seconds": solver_elapsed_seconds,
               "completion_gate": completion,
               "selection": {"path": "selection.json", "sha256": selection_sha},
               "release_ready_manifest": {"path": "release_ready_manifest.json", "sha256": manifest_sha},
               "report": {"path": "report.json", "sha256": report_sha},
               "terminal": {"path": "terminal_evidence.json", "sha256": terminal_sha},
               "endpoint_count": len(endpoints), "training_boundary": protocol["training_boundary"]}
    receipt_sha = write_json(output / "run_receipt.json", receipt)
    write_json(output / "run_status" / "run_status.json", {
        "status": run_status, "run_started_utc": run_started_utc, "run_started_perf_counter": run_started_perf_counter,
                "run_end_utc": run_end_utc, "run_end_perf_counter": run_end_perf_counter,
        "run_wall_seconds": run_wall_seconds, "solver_elapsed_seconds": solver_elapsed_seconds,
        "receipt": "run_receipt.json", "receipt_sha256": receipt_sha,
        "completion_gate": completion,
    }, exclusive=False)
    return {"status": run_status, "exit_code": exit_code, "output": str(output),
            "endpoint_count": len(endpoints), "completion_gate": completion,
            "selection_sha256": selection_sha, "manifest_sha256": manifest_sha,
            "report_sha256": report_sha, "terminal_sha256": terminal_sha, "receipt_sha256": receipt_sha}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--preflight", type=Path, default=ARTIFACT / "preflight.json")
    run = sub.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--prep-root", type=Path, default=ARTIFACT)
    run.add_argument("--known-exposure-p2", type=Path, required=True)
    run.add_argument("--known-exposure-n2", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.preflight.resolve())
    else:
        result = run_formal(args.output.resolve(), args.prep_root.resolve(),
                            {"P2": args.known_exposure_p2.resolve(), "N2": args.known_exposure_n2.resolve()}, args.device)
    print(json.dumps(_jsonable(result), sort_keys=True))
    return int(result.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
