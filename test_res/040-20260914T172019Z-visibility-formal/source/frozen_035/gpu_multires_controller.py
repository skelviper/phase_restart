"""GPU-only multiresolution controller for the frozen five-variant contract.

The controller is a thin orchestration layer.  Initialization, interpolation,
coordinate serialization, and label-free selection come from the existing
training-side modules; only objective/value-gradient calls are replaced by
``gpu_variant_backend.GPUVariantObjective``.  It never opens phase/reference
payloads and refuses to run without a visible CUDA device.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

for _thread_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_key] = "1"

import numpy as np
from scipy.optimize import OptimizeResult, minimize

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from pr import allele_models, contact_model, joint_fit, multires_variant_runner as cpu_runner
from pr import paired_run, reconstruct, reconstruction_init
from gpu_variant_backend import GPUVariantObjective, VARIANTS, cuda_probe, sha256_file

CANDIDATES = tuple(reconstruct.DEFAULT_CANDIDATES)
STAGES = tuple(reconstruct.DEFAULT_STAGES)
P_INIT = 0.75
CHECKPOINT_EVERY = 10
TIE_TOLERANCE = 1e-9


def _jsonable(value: Any) -> Any:
    return cpu_runner._jsonable(value)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_exclusive(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(payload)
    with path.open("xb") as handle:
        handle.write(data)
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _utc_now() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _source_hashes() -> dict[str, str]:
    paths = {
        "adapter": SOURCE_DIR / "gpu_variant_backend.py",
        "controller": SOURCE_DIR / "gpu_multires_controller.py",
        "028_gpu_backend": ROOT / "test_res/028-20260913_151456-020-gpu-independent/source/gpu_backend.py",
        "028_fused_objective": ROOT / "test_res/028-20260913_151456-020-gpu-independent/source/fused/fused_objective.py",
        "028_cuda_pair_kernel": ROOT / "test_res/028-20260913_151456-020-gpu-independent/source/fused/cuda_pair_objective.cu",
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def _planned_rows() -> list[dict[str, Any]]:
    return [{
        "attempt_id": "%s-%s-%s" % (model, candidate.candidate_id, stage.label),
        "variant": model,
        "candidate_id": candidate.candidate_id,
        "stage": stage.label,
        "bin_size_bp": int(stage.bin_size),
        "maxiter": int(stage.maxiter),
        "maxfun": int(stage.maxfun),
    } for model in VARIANTS for candidate in CANDIDATES for stage in STAGES]


def _protocol(context: reconstruct.RunContext, preflight_path: Path,
              artifact_root: Path) -> dict[str, Any]:
    preflight_sha = sha256_file(preflight_path)
    base = cpu_runner._build_protocol(context, preflight_sha, artifact_root)
    planned = _planned_rows()
    base.update({
        "schema": "gpu-multires-variant-protocol-v1",
        "status": "frozen",
        "created_at_utc": _utc_now(),
        "scope": "phase-free/reference-free GPU training from approved 014 blind starts",
        "variants": [allele_models.model_spec(model).as_dict() for model in VARIANTS],
        "remaining_variants": list(VARIANTS),
        "attempt_plan": {
            "planned_variant_count": len(VARIANTS),
            "planned_candidate_count": len(CANDIDATES),
            "planned_stage_count": len(STAGES),
            "planned_trajectory_count": len(VARIANTS) * len(CANDIDATES),
            "planned_stage_attempt_count": len(planned),
            "planned_attempt_count": len(planned),
            "planned_attempts": planned,
            "executed_at_prepare": 0,
            "independent_chain_per_variant_source": True,
            "start_policy": "all 10 variant/source trajectories start from approved 014 sources; each has three stages (30 stage attempts total); no 034 checkpoint resume",
        },
        "backend": {
            "adapter": "gpu_variant_backend.GPUVariantObjective",
            "device": "cuda required",
            "implementation": "028 fused ordered-row CUDA kernel when available",
            "dtype": "float64",
            "mixed_precision": False,
            "sampling": False,
            "pair_grid": "implicit complete upper triangle plus sparse observed CSR",
            "reduction": "ordered-row double precision warp reduction; no global coordinate atomics",
            "candidate_workers": 1,
            "synchronize_each_objective": True,
        },
        "optimization": {
            "schedule": [{"label": stage.label, "bin_size_bp": int(stage.bin_size),
                           "maxiter": int(stage.maxiter), "maxfun": int(stage.maxfun),
                           "maxls": 20, "ftol": 1e-10, "gtol": 1e-6,
                           "checkpoint_every_accepted": CHECKPOINT_EVERY} for stage in STAGES],
            "p_init_first_layer": P_INIT,
            "carry_raw_q_between_layers": True,
            "budget_not_convergence": True,
            "solver": "SciPy L-BFGS-B; objective/gradient on CUDA, line search on CPU",
        },
        "release": {
            "status": "prepared_pending_parent_method_acceptance",
            "c0_cpu_anchor": "test_res/033-20260914_044246-c0-controlled-reproduction/C0_gate.json",
            "c0_gpu_is_not_cpu_byte_identity": True,
            "034_resume": "forbidden; 034 ordinary coordinate checkpoints lack L-BFGS s/y/history",
            "launch_only_after": ["CUDA device probe passes", "fixed-state numeric gate passes",
                                   "parent method acceptance"],
        },
        "preflight": {"path": _relative(artifact_root, preflight_path), "sha256": preflight_sha},
        "source_code_sha256": _source_hashes(),
    })
    return base


def _config(context: reconstruct.RunContext, protocol: Mapping[str, Any],
            protocol_sha: str, output_path: Path) -> dict[str, Any]:
    final_loci = int(protocol["grid"]["n_loci"])
    return {
        "schema": "gpu-multires-variant-config-v1",
        "purpose": "formal_phase_free_gpu_training_only",
        "protocol_sha256": protocol_sha,
        "preflight_sha256": str(protocol["preflight"]["sha256"]),
        "cohort": dict(context.cohort),
        "coordinate_grid": dict(protocol["grid"]),
        "input": dict(protocol["input"]),
        "variants": list(VARIANTS),
        "candidate_order": [candidate.candidate_id for candidate in CANDIDATES],
        "models": {model: cpu_runner._model_contract(model, final_loci) for model in VARIANTS},
        "optimization": dict(protocol["optimization"]),
        "selection": {
            "criterion": "count_nll_per_record",
            "direction": "minimize",
            "tie_tolerance_per_record": TIE_TOLERANCE,
            "tie_break": "preregistered_order",
            "order": [candidate.candidate_id for candidate in CANDIDATES],
            "within_each_variant_only": True,
            "includes_candidate_invariant_diagonal_layer": True,
            "includes_priors": False,
            "reference_used": False,
            "phase_used": False,
        },
        "backend": dict(protocol["backend"]),
        "runtime": {"candidate_workers": 1, "threads": 1,
                    "thread_env": {name: "1" for name in cpu_runner.THREAD_ENV}},
        "formal_output_path": str(output_path),
        "training_boundary": {"phase_used": False, "reference_used": False,
                               "oracle_coordinates_opened": False, "evaluation_outputs_opened": False,
                               "native_fdg_started": False},
        "prepared_not_launched": True,
    }


def prepare(outdir: Path, preflight_path: Path, formal_output: Path) -> dict[str, Any]:
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError("GPU artifact directory is not fresh: %s" % outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    context = reconstruct.production_context()
    cpu_runner._validate_context(context)
    preflight = cpu_runner._read_json(preflight_path)
    if preflight.get("schema") != "gpu-multires-preflight-v1" or preflight.get("status") != "passed":
        raise ValueError("preflight must be a passed gpu-multires-preflight-v1 document")
    protocol = _protocol(context, preflight_path.resolve(), outdir)
    protocol_sha_written = _write_exclusive(outdir / "protocol.json", protocol)
    config = _config(context, protocol, protocol_sha_written, formal_output.resolve())
    config_sha = _write_exclusive(outdir / "config.json", config)
    _write_exclusive(outdir / "provenance" / "source_hashes.json", {
        "schema": "gpu-multires-source-hashes-v1",
        "source_code_sha256": _source_hashes(),
        "protocol_sha256": protocol_sha_written,
        "config_sha256": config_sha,
        "input_sha256": str(context.data_sha256),
        "approved_014_sources": {key: value["sha256"] for key, value in context.source_assets.items()},
        "c0_gate_sha256": sha256_file(ROOT / "test_res/033-20260914_044246-c0-controlled-reproduction/C0_gate.json"),
    })
    (outdir / "launch_command.txt").write_text(
        "source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh\n"
        "conda activate analysis\n"
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n"
        "python %s run --artifact %s --output %s --device cuda --workers 1\n" % (
            str((outdir / "source" / "gpu_multires_controller.py").resolve()),
            str(outdir.resolve()), str(formal_output.resolve())),
        encoding="ascii")
    return {"status": "prepared", "artifact": str(outdir), "protocol_sha256": protocol_sha_written,
            "config_sha256": config_sha, "planned_attempts": len(_planned_rows()),
            "cuda_probe": cuda_probe(), "formal_output": str(formal_output.resolve())}


def _solver_status(result: Any, stage: reconstruct.StageSpec) -> tuple[str, bool, str]:
    return cpu_runner._solver_status(result, stage)


def _fit_gpu(root: Path, model: str, candidate: reconstruct.CandidateSpec,
             stage: reconstruct.StageSpec, data: contact_model.AggregatedContacts,
             initial_coordinates: np.ndarray, positions: np.ndarray,
             chromosome_index: np.ndarray, q_init: float | None,
             device: str) -> tuple[joint_fit.JointFitResult, dict[str, Any], np.ndarray]:
    objective = GPUVariantObjective(data, model, device=device, tile_rows=32,
                                    use_fused=True, diagnostics=True)
    objective.validate_physical_coordinates(initial_coordinates)
    raw_initial = objective.raw_from_physical(initial_coordinates)
    theta0 = objective.pack(raw_initial, p=P_INIT)
    if q_init is not None:
        theta0[-1] = float(q_init)
    initial_total, _, initial_components = objective.evaluate(theta0, need_gradient=False)
    started = time.perf_counter()
    history: list[dict[str, Any]] = [{"iteration": 0, "nfev": 0, "elapsed_seconds": 0.0,
                                      "fun": float(initial_total), "p": float(initial_components["p"]),
                                      "components": _jsonable(initial_components)}]
    checkpoint_paths: list[str] = []
    accepted_activity: list[dict[str, Any]] = []
    actual_nfev = 0
    last_theta: np.ndarray | None = None
    last_gradient: np.ndarray | None = None

    def loss_and_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal actual_nfev, last_theta, last_gradient
        actual_nfev += 1
        value, gradient = objective.value_and_grad(theta)
        last_theta = np.asarray(theta, dtype=np.float64).copy()
        last_gradient = np.asarray(gradient, dtype=np.float64).copy()
        return float(value), last_gradient.copy()

    def checkpoint_hook(checkpoint: joint_fit.JointCheckpoint) -> None:
        entry = {"iteration": int(checkpoint.iteration), "nfev": int(checkpoint.nfev),
                 "elapsed_seconds": float(checkpoint.elapsed_seconds), "fun": float(checkpoint.fun),
                 "p": float(checkpoint.p), "components": _jsonable(checkpoint.components)}
        snapshot_history = list(history)
        if not snapshot_history or snapshot_history[-1]["iteration"] != entry["iteration"]:
            snapshot_history.append(entry)
        path = root / "checkpoints" / model / candidate.candidate_id / (
            "%s-accepted-%04d.npz" % (stage.label, checkpoint.iteration))
        cpu_runner._checkpoint_write(path, checkpoint, snapshot_history, positions, chromosome_index)
        checkpoint_paths.append(_relative(root, path))

    def callback(_entry: Mapping[str, Any]) -> None:
        theta = np.asarray(_entry.get("theta", []), dtype=np.float64) if "theta" in _entry else None
        # SciPy callback supplies x only through the closure in this controller.
        if theta is None or theta.shape != (objective.n_parameters,):
            return
        cached = objective.cached_value_and_components(theta)
        if cached is None:
            value, gradient, components = objective.evaluate(theta, need_gradient=True)
        else:
            value, components = cached
            gradient = last_gradient if last_theta is not None and np.array_equal(theta, last_theta) else objective.evaluate(theta, True)[1]
        iteration = len(history)
        p = float(components["p"])
        entry = {"iteration": iteration, "nfev": actual_nfev,
                 "elapsed_seconds": time.perf_counter() - started, "fun": float(value), "p": p,
                 "gradient_norm": float(np.linalg.norm(gradient)), "components": _jsonable(components),
                 "map_diagnostics": objective.map_diagnostics()}
        history.append(entry)
        accepted_activity.append({"iteration": iteration, "nfev": actual_nfev,
                                  "map_diagnostics": objective.map_diagnostics()})
        if CHECKPOINT_EVERY and iteration % CHECKPOINT_EVERY == 0:
            raw, _ = objective.unpack(theta)
            coordinates, p = objective.coordinates_and_p(theta)
            checkpoint_hook(joint_fit.JointCheckpoint(
                iteration=iteration, nfev=actual_nfev,
                elapsed_seconds=float(entry["elapsed_seconds"]), fun=float(value), p=float(p),
                theta=theta.copy(), y=raw.copy(), coordinates=coordinates.copy(),
                components=dict(components), gradient_norm=float(np.linalg.norm(gradient))))

    def scipy_callback(theta: np.ndarray) -> None:
        callback({"theta": np.asarray(theta, dtype=np.float64)})

    options = {"maxiter": int(stage.maxiter), "maxfun": int(stage.maxfun), "maxls": 20,
               "ftol": 1e-10, "gtol": 1e-6}
    raw_result: OptimizeResult = minimize(loss_and_gradient, theta0, method="L-BFGS-B", jac=True,
                                           callback=scipy_callback, options=options)
    final_theta = np.asarray(raw_result.x, dtype=np.float64)
    final_coordinates, p = objective.coordinates_and_p(final_theta)
    final_value, final_gradient, final_components = objective.evaluate(final_theta, need_gradient=True)
    if not history or history[-1].get("fun") != float(final_value):
        history.append({"iteration": len(history), "nfev": actual_nfev,
                        "elapsed_seconds": time.perf_counter() - started,
                        "fun": float(final_value), "p": float(p),
                        "gradient_norm": float(np.linalg.norm(final_gradient)),
                        "components": _jsonable(final_components),
                        "map_diagnostics": objective.map_diagnostics()})
    elapsed = time.perf_counter() - started
    result = joint_fit.JointFitResult(
        objective=objective, theta=final_theta.copy(), y=final_theta[:-1].reshape(2, data.n_loci, 3).copy(),
        coordinates=final_coordinates.copy(), p=float(p), fun=float(final_value),
        components=dict(final_components), success=bool(raw_result.success), status=int(raw_result.status),
        message=str(raw_result.message), nit=int(raw_result.nit), nfev=int(actual_nfev),
        actual_nfev=int(actual_nfev), scipy_nfev=int(getattr(raw_result, "nfev", actual_nfev)),
        njev=int(getattr(raw_result, "njev", 0) or 0), elapsed_seconds=float(elapsed), history=history)
    status, budget_exhausted, reason = _solver_status(result, stage)
    payload = {
        "status": status,
        "termination_reason": reason,
        "budget_exhausted": budget_exhausted,
        "initial_total": float(initial_total),
        "final_total": float(final_value),
        "initial_components": _jsonable(initial_components),
        "final_components": _jsonable(final_components),
        "p": float(p), "q_in": None if q_init is None else float(q_init), "q_out": float(final_theta[-1]),
        "solver": {"success": bool(raw_result.success), "status": int(raw_result.status),
                   "message": str(raw_result.message), "nit": int(raw_result.nit), "nfev": int(actual_nfev),
                   "actual_nfev": int(actual_nfev), "scipy_nfev": int(getattr(raw_result, "nfev", actual_nfev)),
                   "njev": int(getattr(raw_result, "njev", 0) or 0), "elapsed_seconds": float(elapsed),
                   "maxiter": stage.maxiter, "maxfun": stage.maxfun, "maxls": 20,
                   "ftol": 1e-10, "gtol": 1e-6},
        "checkpoint_paths": checkpoint_paths,
        "history": _jsonable(history),
        "mapping_diagnostics": _jsonable(objective.map_diagnostics()),
        "backend": objective.backend_metadata(),
        "model_id": model,
        "physical_domain": allele_models.model_spec(model).physical_domain,
        "coordinate_parameterization": allele_models.model_spec(model).coordinate_parameterization,
    }
    return result, payload, final_coordinates


def _run_chain(root: Path, context: reconstruct.RunContext, config: Mapping[str, Any],
               model: str, candidate: reconstruct.CandidateSpec, device: str,
               data_cache: dict[int, contact_model.AggregatedContacts]) -> dict[str, Any]:
    names = tuple(name for name, _ in context.headers)
    lengths = tuple(int(length) for _, length in context.headers)
    previous_coordinates = None
    previous_positions = None
    previous_chromosome = None
    carried_q = None
    attempts = []
    failure_reason: str | None = None
    for index, stage in enumerate(STAGES):
        attempt_id = "%s-%s-%s" % (model, candidate.candidate_id, stage.label)
        stage_payload: dict[str, Any] = {"model_id": model, "candidate_id": candidate.candidate_id,
                                         "attempt_id": attempt_id, "stage": stage.label,
                                         "bin_size_bp": int(stage.bin_size), "started_at_utc": _utc_now()}
        if failure_reason is not None:
            stage_payload.update({"status": "not_run_after_prior_failure", "completed_at_utc": _utc_now(),
                                  "failure_reason": failure_reason})
            stage_path = root / "stages" / model / candidate.candidate_id / (stage.label + ".json")
            record_path = _write_exclusive(stage_path, stage_payload)
            attempts.append({"attempt_id": attempt_id, "model_id": model,
                             "candidate_id": candidate.candidate_id, "stage": stage.label,
                             "status": "not_run_after_prior_failure", "is_terminal": index == len(STAGES) - 1,
                             "stage_record": _relative(root, stage_path), "failure_reason": failure_reason})
            continue
        try:
            bin_size = int(stage.bin_size)
            if bin_size not in data_cache:
                data_cache[bin_size] = contact_model.load_frozen_p9016_aggregate(bin_size)
            data = data_cache[bin_size]
            budget = cpu_runner._validate_layer(data, context, stage)
            if previous_coordinates is None:
                initialized = reconstruction_init.initialize_approved_candidate(
                    candidate.initialization_candidate, names, lengths, stage.bin_size)
            else:
                initialized = cpu_runner.transfer_layer(
                    previous_coordinates, previous_positions, previous_chromosome,
                    names, lengths, stage.bin_size, candidate.base_seed, model)
            initial_coordinates, positions, chromosome_index, init_meta = cpu_runner._validate_initialization(
                initialized, data, model)
            initial_path = root / "coords" / model / candidate.candidate_id / ("initial-%s.3dg" % stage.label)
            initial_path.parent.mkdir(parents=True, exist_ok=True)
            initial_info = paired_run.write_coordinates(initial_path, data, model, initial_coordinates)
            result, fit_payload, final_coordinates = _fit_gpu(
                root, model, candidate, stage, data, initial_coordinates,
                positions, chromosome_index, carried_q, device)
            final_path = root / "coords" / model / candidate.candidate_id / ("final-%s.3dg" % stage.label)
            final_info = paired_run.write_coordinates(final_path, data, model, final_coordinates)
            stage_payload.update({"status": fit_payload["status"], "completed_at_utc": _utc_now(),
                                  "data_budget": budget, "initialization": init_meta,
                                  "initial_coordinates": {**initial_info, "path": _relative(root, initial_path)},
                                  "final_coordinates": {**final_info, "path": _relative(root, final_path)},
                                  "fit": fit_payload})
            record_path = _write_exclusive(root / "stages" / model / candidate.candidate_id / (stage.label + ".json"), stage_payload)
            attempts.append({"attempt_id": attempt_id, "model_id": model,
                             "candidate_id": candidate.candidate_id, "stage": stage.label,
                             "status": fit_payload["status"], "is_terminal": index == len(STAGES) - 1,
                             "stage_record": _relative(root, root / "stages" / model / candidate.candidate_id / (stage.label + ".json")),
                             "budget_exhausted": fit_payload["budget_exhausted"]})
            previous_coordinates, previous_positions, previous_chromosome = final_coordinates, positions, chromosome_index
            carried_q = float(result.theta[-1])
        except Exception as exc:
            failure_reason = "%s: %s" % (type(exc).__name__, exc)
            stage_payload.update({"status": "failed", "failed_at_utc": _utc_now(),
                                  "failure_reason": failure_reason})
            stage_path = root / "stages" / model / candidate.candidate_id / (stage.label + ".json")
            _write_exclusive(stage_path, stage_payload)
            attempts.append({"attempt_id": attempt_id, "model_id": model, "candidate_id": candidate.candidate_id,
                             "stage": stage.label, "status": "failed", "is_terminal": index == len(STAGES) - 1,
                             "stage_record": _relative(root, stage_path), "failure_reason": failure_reason})
    if failure_reason is not None:
        return {"id": candidate.candidate_id, "model_id": model,
                "initialization": candidate.initialization_candidate, "base_seed": candidate.base_seed,
                "final_coordinates_path": None, "final_coordinates_sha256": None, "final_q": None,
                "attempts": attempts, "failure_reason": failure_reason,
                "optimization": {"terminal_status": "failed", "terminal_attempt_id": attempts[-1]["attempt_id"]}}
    final_attempt = attempts[-1]
    final_stage_path = root / "stages" / model / candidate.candidate_id / (STAGES[-1].label + ".json")
    final_stage = cpu_runner._read_json(final_stage_path)
    final_coordinates = final_stage["final_coordinates"]
    return {"id": candidate.candidate_id, "model_id": model,
            "initialization": candidate.initialization_candidate, "base_seed": candidate.base_seed,
            "final_coordinates_path": final_coordinates["path"],
            "final_coordinates_sha256": final_coordinates["sha256"], "final_q": final_stage["fit"]["q_out"],
            "attempts": attempts,
            "optimization": {"terminal_status": final_attempt["status"],
                              "terminal_attempt_id": final_attempt["attempt_id"]}}


def _attempt_audit(candidates: Sequence[Mapping[str, Any]],
                   protocol: Mapping[str, Any]) -> dict[str, Any]:
    planned = list(protocol.get("attempt_plan", {}).get("planned_attempts", []))
    expected_ids = [str(row["attempt_id"]) for row in planned]
    actual = [dict(attempt) for candidate in candidates for attempt in candidate.get("attempts", [])]
    actual_ids = [str(row.get("attempt_id")) for row in actual]
    id_counts = {}
    for attempt_id in actual_ids:
        id_counts[attempt_id] = id_counts.get(attempt_id, 0) + 1
    duplicate_ids = sorted(attempt_id for attempt_id, count in id_counts.items() if count > 1)
    expected_set = set(expected_ids)
    actual_set = set(actual_ids)
    missing_ids = sorted(expected_set - actual_set)
    unexpected_ids = sorted(actual_set - expected_set)
    failed_attempts = [row for row in actual
                       if row.get("status") in ("failed", "not_run_after_prior_failure")]
    completed_attempts = [row for row in actual
                          if row.get("status") not in ("failed", "not_run_after_prior_failure")]
    trajectories = []
    for candidate in candidates:
        attempts = list(candidate.get("attempts", []))
        complete_stage_rows = [row for row in attempts
                               if row.get("status") not in ("failed", "not_run_after_prior_failure")]
        endpoint = bool(candidate.get("final_coordinates_path")
                        and candidate.get("final_coordinates_sha256")
                        and candidate.get("final_q") is not None
                        and candidate.get("optimization", {}).get("terminal_status") != "failed")
        trajectory_pass = len(attempts) == len(STAGES) and len(complete_stage_rows) == len(STAGES) and endpoint
        trajectories.append({
            "model_id": candidate.get("model_id"),
            "candidate_id": candidate.get("id"),
            "attempt_count": len(attempts),
            "completed_stage_count": len(complete_stage_rows),
            "has_final_1m_endpoint": endpoint,
            "pass": trajectory_pass,
        })
    expected_trajectories = len(VARIANTS) * len(CANDIDATES)
    passed = bool(
        len(planned) == len(VARIANTS) * len(CANDIDATES) * len(STAGES)
        and len(actual) == len(planned)
        and len(actual_ids) == len(set(actual_ids))
        and not missing_ids and not unexpected_ids and not duplicate_ids
        and not failed_attempts
        and len(trajectories) == expected_trajectories
        and all(row["pass"] for row in trajectories)
    )
    return {
        "schema": "gpu-multires-attempt-audit-v1",
        "pass": passed,
        "planned_trajectory_count": expected_trajectories,
        "actual_trajectory_count": len(trajectories),
        "planned_stage_attempt_count": len(planned),
        "actual_stage_attempt_count": len(actual),
        "completed_stage_attempt_count": len(completed_attempts),
        "failed_or_not_run_stage_attempt_count": len(failed_attempts),
        "expected_attempt_ids": expected_ids,
        "actual_attempt_ids": actual_ids,
        "missing_attempt_ids": missing_ids,
        "unexpected_attempt_ids": unexpected_ids,
        "duplicate_attempt_ids": duplicate_ids,
        "trajectories": trajectories,
        "failure_rows": failed_attempts,
    }


def _artifact_member(artifact: Path, relative: str) -> Path:
    path = (artifact / str(relative)).resolve()
    try:
        path.relative_to(artifact.resolve())
    except ValueError as exc:
        raise ValueError("artifact member escapes frozen artifact: %s" % relative) from exc
    return path


def _validate_frozen_artifact(artifact: Path, protocol: Mapping[str, Any],
                              config: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("schema") != "gpu-multires-variant-protocol-v1" or protocol.get("status") != "frozen":
        raise ValueError("protocol is not a frozen gpu-multires-variant-protocol-v1 document")
    if config.get("schema") != "gpu-multires-variant-config-v1":
        raise ValueError("config schema is not gpu-multires-variant-config-v1")
    preflight_ref = protocol.get("preflight", {})
    preflight_path = _artifact_member(artifact, str(preflight_ref.get("path", "")))
    if not preflight_path.is_file():
        raise ValueError("frozen preflight file is missing")
    preflight_sha = sha256_file(preflight_path)
    if preflight_sha != str(preflight_ref.get("sha256")):
        raise ValueError("frozen preflight hash mismatch")
    preflight = cpu_runner._read_json(preflight_path)
    if preflight.get("schema") != "gpu-multires-preflight-v1" or preflight.get("status") != "passed":
        raise ValueError("frozen preflight is not passed")
    protocol_path = artifact / "protocol.json"
    config_path = artifact / "config.json"
    protocol_sha = sha256_file(protocol_path)
    config_sha = sha256_file(config_path)
    if protocol_sha != str(config.get("protocol_sha256")):
        raise ValueError("config protocol_sha256 does not match protocol bytes")
    if str(config.get("preflight_sha256")) != preflight_sha:
        raise ValueError("config preflight_sha256 does not match preflight bytes")
    source_manifest_path = artifact / "provenance" / "source_hashes.json"
    if not source_manifest_path.is_file():
        raise ValueError("frozen source hash manifest is missing")
    source_manifest = cpu_runner._read_json(source_manifest_path)
    if source_manifest.get("protocol_sha256") != protocol_sha:
        raise ValueError("source manifest protocol hash mismatch")
    if source_manifest.get("config_sha256") != config_sha:
        raise ValueError("source manifest config hash mismatch")
    current_sources = _source_hashes()
    if source_manifest.get("source_code_sha256") != current_sources:
        raise ValueError("source code hash manifest does not match current frozen sources")
    if source_manifest.get("input_sha256") != protocol.get("input", {}).get("sha256"):
        raise ValueError("source manifest input hash mismatch")
    if config.get("input", {}).get("sha256") != protocol.get("input", {}).get("sha256"):
        raise ValueError("config and protocol input hashes disagree")
    planned = protocol.get("attempt_plan", {}).get("planned_attempts", [])
    if len(planned) != len(VARIANTS) * len(CANDIDATES) * len(STAGES):
        raise ValueError("frozen protocol does not contain exactly 30 stage attempts")
    return {
        "protocol_sha256": protocol_sha,
        "config_sha256": config_sha,
        "preflight_sha256": preflight_sha,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_code_sha256": current_sources,
    }


def _rescore(root: Path, candidate: dict[str, Any], data: contact_model.AggregatedContacts,
             device: str) -> dict[str, Any]:
    if candidate.get("optimization", {}).get("terminal_status") == "failed":
        return candidate
    path = root / str(candidate["final_coordinates_path"])
    coordinates = cpu_runner._read_model_coordinates(path, data, candidate["model_id"], candidate["final_coordinates_sha256"])
    objective = GPUVariantObjective(data, candidate["model_id"], device=device, tile_rows=32,
                                    use_fused=True, diagnostics=True)
    theta = objective.pack(objective.raw_from_physical(coordinates), p=P_INIT)
    theta[-1] = float(candidate["final_q"])
    total, _, components = objective.evaluate(theta, need_gradient=False)
    candidate.update({"coordinates": {"path": candidate["final_coordinates_path"],
                                       "sha256": candidate["final_coordinates_sha256"],
                                       "model_id": candidate["model_id"],
                                       "physical_domain": allele_models.model_spec(candidate["model_id"]).physical_domain},
                      "count_nll_per_record": float(components["count_nll_normalized"]),
                      "count_model": {key: float(components[key]) for key in (
                          "conditional_nll_raw", "diag_profiled_nll_raw", "count_nll_raw", "count_nll_normalized")},
                      "selection_rescore": {"coordinate_readback": True, "q_fixed_from_final_fit": float(candidate["final_q"]),
                                             "total": float(total), "backend": objective.backend_metadata()},
                      "optimization": {"terminal_status": candidate["attempts"][-1]["status"],
                                       "terminal_attempt_id": candidate["attempts"][-1]["attempt_id"]}})
    return candidate


def _failed_candidate_record(root: Path, model: str, candidate: reconstruct.CandidateSpec,
                             reason: str) -> dict[str, Any]:
    """Materialize all missing chain rows after an outer orchestration failure."""
    attempts = []
    prior_reason = reason
    for index, stage in enumerate(STAGES):
        path = root / "stages" / model / candidate.candidate_id / (stage.label + ".json")
        attempt_id = "%s-%s-%s" % (model, candidate.candidate_id, stage.label)
        if path.is_file():
            payload = cpu_runner._read_json(path)
            status = str(payload.get("status", "failed"))
            row = {"attempt_id": attempt_id, "model_id": model, "candidate_id": candidate.candidate_id,
                   "stage": stage.label, "status": status, "is_terminal": index == len(STAGES) - 1,
                   "stage_record": _relative(root, path)}
            if status in ("failed", "not_run_after_prior_failure"):
                row["failure_reason"] = str(payload.get("failure_reason", prior_reason))
            attempts.append(row)
            continue
        status = "failed" if index == 0 else "not_run_after_prior_failure"
        payload = {"model_id": model, "candidate_id": candidate.candidate_id,
                   "attempt_id": attempt_id, "stage": stage.label,
                   "bin_size_bp": int(stage.bin_size), "status": status,
                   "failure_reason": prior_reason, "failed_at_utc": _utc_now()}
        _write_exclusive(path, payload)
        attempts.append({"attempt_id": attempt_id, "model_id": model, "candidate_id": candidate.candidate_id,
                         "stage": stage.label, "status": status, "is_terminal": index == len(STAGES) - 1,
                         "stage_record": _relative(root, path), "failure_reason": prior_reason})
    return {"id": candidate.candidate_id, "model_id": model,
            "initialization": candidate.initialization_candidate, "base_seed": candidate.base_seed,
            "final_coordinates_path": None, "final_coordinates_sha256": None, "final_q": None,
            "attempts": attempts, "failure_reason": reason,
            "optimization": {"terminal_status": "failed", "terminal_attempt_id": attempts[-1]["attempt_id"]}}


def run(artifact: Path, output: Path, device: str = "cuda", workers: int = 1) -> dict[str, Any]:
    if device != "cuda":
        raise ValueError("formal GPU controller requires --device cuda; CPU execution is forbidden")
    if workers != 1:
        raise ValueError("GPU controller freezes one candidate chain at a time")
    protocol = cpu_runner._read_json(artifact / "protocol.json")
    config = cpu_runner._read_json(artifact / "config.json")
    integrity = _validate_frozen_artifact(artifact, protocol, config)
    expected_output = Path(str(config.get("formal_output_path", ""))).resolve()
    if output.resolve() != expected_output:
        raise ValueError("formal output path does not match frozen config formal_output_path")
    probe = cuda_probe()
    if probe.get("status") != "available":
        raise RuntimeError("CUDA device unavailable; refusing formal run: %s" % probe)
    planned = protocol.get("attempt_plan", {}).get("planned_attempts", [])
    if len(planned) != len(VARIANTS) * len(CANDIDATES) * len(STAGES):
        raise ValueError("formal GPU protocol must contain exactly 30 stage attempts")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("formal GPU output must be fresh: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    context = reconstruct.production_context()
    cpu_runner._validate_context(context)
    for relative in ("coords", "checkpoints", "logs", "stages", "selected", "run_status", "provenance"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    frozen_copies = {}
    for relative in ("protocol.json", "config.json", "provenance/source_hashes.json"):
        source_path = artifact / relative
        destination = output / "provenance" / Path(relative).name
        shutil.copyfile(source_path, destination)
        copied_sha = sha256_file(destination)
        if copied_sha != sha256_file(source_path):
            raise RuntimeError("frozen artifact copy hash mismatch: %s" % relative)
        frozen_copies[relative] = {"path": _relative(output, destination), "sha256": copied_sha}
    _write_exclusive(output / "runtime_probe.json", {"status": "passed", "cuda": probe,
                                                       "recorded_at_utc": _utc_now(),
                                                       "source_artifact": str(artifact.resolve()),
                                                       "artifact_integrity": integrity,
                                                       "frozen_copies": frozen_copies})
    data_cache: dict[int, contact_model.AggregatedContacts] = {}
    candidates = []
    started = _utc_now()
    for model in VARIANTS:
        for candidate in CANDIDATES:
            try:
                record = _run_chain(output, context, config, model, candidate, device, data_cache)
            except Exception as exc:
                record = _failed_candidate_record(
                    output, model, candidate, "%s: %s" % (type(exc).__name__, exc))
            candidates.append(record)
    if 1_000_000 not in data_cache:
        data_cache[1_000_000] = contact_model.load_frozen_p9016_aggregate(1_000_000)
    final_data = data_cache[1_000_000]
    attempt_audit = _attempt_audit(candidates, protocol)
    finalized = []
    rescore_errors = []
    for candidate in candidates:
        if candidate.get("optimization", {}).get("terminal_status") == "failed":
            finalized.append(candidate)
            continue
        try:
            finalized.append(_rescore(output, candidate, final_data, device))
        except Exception as exc:
            failed = dict(candidate)
            reason = "%s: %s" % (type(exc).__name__, exc)
            failed.update({"failure_reason": reason, "failure_phase": "rescore",
                           "optimization": {"terminal_status": "failed",
                                             "terminal_attempt_id": candidate.get("attempts", [{}])[-1].get("attempt_id")}})
            finalized.append(failed)
            rescore_errors.append({"model_id": candidate.get("model_id"), "candidate_id": candidate.get("id"),
                                   "failure_reason": reason})
    selections = {}
    selected_files = {}
    selection_errors = []
    for model in VARIANTS:
        subset = [item for item in finalized if item.get("model_id") == model]
        try:
            selected_id = cpu_runner.select_variant_candidate(subset)
            selected = next(item for item in subset if item["id"] == selected_id)
            source = output / str(selected["coordinates"]["path"])
            target = output / "selected" / (model + ".3dg")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            selected_files[model] = {"path": _relative(output, target), "sha256": sha256_file(target),
                                     "source_candidate_id": selected_id,
                                     "source_coordinate_path": selected["coordinates"]["path"]}
            selections[model] = {"status": "selected", "selected_id": selected_id,
                                 "candidate_scores": {item["id"]: item.get("count_nll_per_record") for item in subset},
                                 "criterion": "count_nll_per_record", "tie_tolerance_per_record": TIE_TOLERANCE,
                                 "reference_used": False, "phase_used": False}
        except Exception as exc:
            reason = "%s: %s" % (type(exc).__name__, exc)
            selections[model] = {"status": "failed", "failure_reason": reason,
                                 "criterion": "count_nll_per_record", "reference_used": False, "phase_used": False}
            selection_errors.append({"model_id": model, "failure_reason": reason})
    complete = bool(attempt_audit["pass"] and not rescore_errors and not selection_errors
                    and len(selected_files) == len(VARIANTS)
                    and all(row.get("status") == "selected" for row in selections.values()))
    selection_status = "training_complete" if complete else "incomplete"
    selection = {"schema": "gpu-multires-selection-v1", "status": selection_status,
                 "created_at_utc": _utc_now(), "started_at_utc": started,
                 "candidates": _jsonable(finalized),
                 "selection": {"per_variant": selections, "selected_files": selected_files,
                               "all_attempts_accounted_for": bool(attempt_audit["pass"]),
                               "attempt_audit": attempt_audit,
                               "rescore_errors": rescore_errors, "selection_errors": selection_errors,
                               "reference_used": False, "phase_used": False},
                 "training_boundary": dict(config["training_boundary"]),
                 "artifact_integrity": integrity}
    selection_sha = _write_exclusive(output / "selection.json", selection)
    terminal = {"schema": "gpu-multires-terminal-evidence-v1",
                 "status": selection_status,
                 "runner_exit_code": 0 if complete else 2,
                 "started_at_utc": started, "ended_at_utc": _utc_now(),
                 "planned_trajectory_count": len(VARIANTS) * len(CANDIDATES),
                 "planned_stage_attempt_count": len(planned),
                 "executed_stage_attempt_count": attempt_audit["actual_stage_attempt_count"],
                 "completed_stage_attempt_count": attempt_audit["completed_stage_attempt_count"],
                 "failed_or_not_run_stage_attempt_count": attempt_audit["failed_or_not_run_stage_attempt_count"],
                 "attempt_audit": attempt_audit, "selection_sha256": selection_sha,
                 "rescore_errors": rescore_errors, "selection_errors": selection_errors,
                 "artifact_integrity": integrity, "cuda": probe,
                 "reference_used": False, "phase_used": False}
    terminal_sha = _write_exclusive(output / "terminal_evidence.json", terminal)
    if not complete:
        raise RuntimeError("formal GPU run incomplete; terminal evidence written: %s (sha256=%s)" %
                           (output / "terminal_evidence.json", terminal_sha))
    return {"status": "training_complete", "output": str(output), "selection_sha256": selection_sha,
            "terminal_evidence_sha256": terminal_sha, "cuda": probe,
            "trajectories": len(VARIANTS) * len(CANDIDATES),
            "stage_attempts": len(planned)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--artifact", required=True, type=Path)
    prep.add_argument("--preflight", required=True, type=Path)
    prep.add_argument("--formal-output", required=True, type=Path)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--artifact", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--workers", default=1, type=int)
    args = parser.parse_args(argv)
    try:
        result = (prepare(args.artifact, args.preflight, args.formal_output)
                  if args.command == "prepare"
                  else run(args.artifact, args.output, args.device, args.workers))
        print(json.dumps(_jsonable(result), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
