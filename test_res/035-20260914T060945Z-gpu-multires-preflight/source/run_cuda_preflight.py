"""运行 CUDA migration preflight，不启动正式训练。

本脚本只评估已批准的 phase-free sources 和 SNP-free aggregate。所有 numerical/runtime gates 通过后，才在 035 preflight 目录下写出不可变 JSON evidence 并准备 protocol/config/launch metadata。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "test_res" / "035-20260914T060945Z-gpu-multires-preflight"
SOURCE = ARTIFACT / "source"
PREFLIGHT_DIR = ARTIFACT / "preflight"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

# 为声明的比较将 host-side 数值库限制为单线程。
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from gpu_variant_backend import GPUVariantObjective, VARIANTS, sha256_file  # noqa: E402
import gpu_multires_controller as gpu_controller  # noqa: E402
from pr import allele_models, contact_model, paired_run, reconstruct  # noqa: E402
from pr import reconstruction_init  # noqa: E402
from pr import multires_variant_runner as cpu_runner  # noqa: E402


CANDIDATES = tuple(reconstruct.DEFAULT_CANDIDATES)
STAGES = tuple(reconstruct.DEFAULT_STAGES)
PARITY_TOLERANCES = {
    "normalized_atol": 1e-10,
    "normalized_rtol": 1e-10,
    "raw_atol": 1e-8,
    "raw_rtol": 1e-12,
    "gradient_atol": 1e-10,
    "gradient_rtol": 1e-9,
}
NORMALIZED_FIELDS = (
    "count_nll_normalized", "p", "p_prior", "bond", "repulsion", "bend", "total",
)
RAW_FIELDS = (
    "sum_rate_cis_offdiag", "sum_rate_inter", "observed_log_rate_cis_offdiag",
    "observed_log_rate_inter", "conditional_nll_raw", "diag_profiled_nll_raw",
    "count_nll_raw",
)
ARRAY_FIELDS = (
    "pair_i", "pair_j", "cis_pair", "counts", "diag_counts", "endpoint_counts",
    "exposure", "locus_chromosome", "locus_bin", "n_bins", "offsets",
)
EXPECTED_INPUT_SHA = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"
EXPECTED_SOURCE_SHA = {
    "consensus": "e76655732deb6b8386b1b77bc76ff45d7dba1384f6931337fee80d8f4aaa8e02",
    "random": "9a48d73e1401e18349d11758e679da4c76da0904dbc467979079cb54bcd567d7",
}
EXPECTED_BACKEND_SHA = {
    "gpu_backend.py": "67d5231a9a8b8b1609dab369304df4c973015b90039a4aeb88ee2b047fea1c6f",
    "fused_objective.py": "958397a8ee938716d98e49425e360363399483357739f4a63d4bb663c89736c4",
    "cuda_pair_objective.cu": "f100d691ead66d3ffade72d027d1456fe36f167619e892eb0e076cd31acfcdd7",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("nonfinite metadata value")
        return float(value)
    return value


def _array_sha(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(_jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _max_abs(a: Any, b: Any) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right), initial=0.0))


def _max_rel(a: Any, b: Any) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    denominator = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-300)
    return float(np.max(np.abs(left - right) / denominator, initial=0.0))


def _within(a: float, b: float, atol: float, rtol: float) -> bool:
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=rtol))


def _state_hashes(state: dict[str, Any], coordinates: np.ndarray) -> dict[str, Any]:
    positions = np.asarray(state["positions"], dtype=np.int64)
    chromosomes = np.asarray(state["chromosome_index"], dtype=np.int64)
    return {
        "coordinates_float64_c_order": _array_sha(coordinates.astype("<f8", copy=False)),
        "positions_int64_c_order": _array_sha(positions.astype("<i8", copy=False)),
        "chromosome_index_int64_c_order": _array_sha(chromosomes.astype("<i8", copy=False)),
        "coordinates_shape": [int(v) for v in coordinates.shape],
        "positions_shape": [int(v) for v in positions.shape],
        "chromosome_index_shape": [int(v) for v in chromosomes.shape],
        "positions_min": int(positions.min()),
        "positions_max": int(positions.max()),
        "max_physical_radius": float(np.linalg.norm(coordinates, axis=2).max()),
    }


def _load_states(context: reconstruct.RunContext, source_name: str, model: str,
                 data_by_bin: dict[int, contact_model.AggregatedContacts]) -> dict[str, dict[str, Any]]:
    names = tuple(name for name, _ in context.headers)
    lengths = tuple(int(length) for _, length in context.headers)
    candidate = next(c for c in CANDIDATES if c.initialization_candidate == source_name)
    result: dict[str, dict[str, Any]] = {}
    previous: dict[str, Any] | None = None
    for stage in STAGES:
        data = data_by_bin[int(stage.bin_size)]
        if previous is None:
            state = reconstruction_init.initialize_approved_candidate(
                source_name, names, lengths, int(stage.bin_size))
        else:
            state = cpu_runner.transfer_layer(
                previous["coords"], previous["positions"], previous["chromosome_index"],
                names, lengths, int(stage.bin_size), candidate.base_seed, model)
        coordinates = np.asarray(state["coords"], dtype=np.float64)
        positions = np.asarray(state["positions"], dtype=np.int64)
        chromosome_index = np.asarray(state["chromosome_index"], dtype=np.int64)
        expected_positions = np.asarray(data.locus_bin, dtype=np.int64) * int(data.bin_size)
        expected_chromosome = np.asarray(data.locus_chromosome, dtype=np.int64)
        if coordinates.shape != (2, data.n_loci, 3):
            raise AssertionError(f"{model}/{source_name}/{stage.label}: coordinate shape changed")
        if positions.shape == (data.n_loci,):
            positions = np.broadcast_to(positions, (2, data.n_loci)).copy()
        if chromosome_index.shape == (data.n_loci,):
            chromosome_index = np.broadcast_to(chromosome_index, (2, data.n_loci)).copy()
        if not np.array_equal(positions[0], expected_positions) or not np.array_equal(positions[1], expected_positions):
            raise AssertionError(f"{model}/{source_name}/{stage.label}: position grid changed")
        if not np.array_equal(chromosome_index[0], expected_chromosome) or not np.array_equal(chromosome_index[1], expected_chromosome):
            raise AssertionError(f"{model}/{source_name}/{stage.label}: chromosome grid changed")
        allele_models.validate_physical_for_model(model, coordinates)
        state = dict(state)
        state["positions"] = positions
        state["chromosome_index"] = chromosome_index
        state["coords"] = coordinates
        state["state_hashes"] = _state_hashes(state, coordinates)
        state["source_name"] = source_name
        state["model"] = model
        state["stage"] = stage.label
        result[stage.label] = state
        previous = state
    return result


def _data_manifest(context: reconstruct.RunContext,
                   data_by_bin: dict[int, contact_model.AggregatedContacts]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in STAGES:
        data = data_by_bin[int(stage.bin_size)]
        budget = cpu_runner._validate_layer(data, context, stage)
        result[stage.label] = {
            "bin_size_bp": int(data.bin_size),
            "budget": _jsonable(budget),
            "array_sha256": {name: _array_sha(getattr(data, name)) for name in ARRAY_FIELDS},
        }
    return result


def _parity(context: reconstruct.RunContext,
            data_by_bin: dict[int, contact_model.AggregatedContacts]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model in VARIANTS:
        for stage in STAGES:
            data = data_by_bin[int(stage.bin_size)]
            cpu_objective = allele_models.objective_for_model(data, model)
            gpu_objective = GPUVariantObjective(
                data, model, device="cuda", tile_rows=32, use_fused=True, diagnostics=True)
            for source_name in ("consensus", "random"):
                state = _load_states(context, source_name, model, data_by_bin)[stage.label]
                coordinates = np.asarray(state["coords"], dtype=np.float64)
                raw = allele_models.raw_coordinates_from_physical(cpu_objective, coordinates)
                theta = cpu_objective.pack(raw, p=0.75)
                started = time.perf_counter()
                cpu_value, cpu_gradient, cpu_components = cpu_objective.evaluate(theta, need_gradient=True)
                cpu_elapsed = time.perf_counter() - started
                started = time.perf_counter()
                gpu_value, gpu_gradient, gpu_components = gpu_objective.evaluate(theta, need_gradient=True)
                gpu_elapsed = time.perf_counter() - started
                component_abs: dict[str, float] = {}
                component_rel: dict[str, float] = {}
                for field in NORMALIZED_FIELDS + RAW_FIELDS:
                    if field not in cpu_components or field not in gpu_components:
                        raise AssertionError(f"missing parity field {field}")
                    component_abs[field] = abs(float(cpu_components[field]) - float(gpu_components[field]))
                    component_rel[field] = _max_rel([cpu_components[field]], [gpu_components[field]])
                normalized_pass = all(_within(cpu_components[field], gpu_components[field],
                                              PARITY_TOLERANCES["normalized_atol"],
                                              PARITY_TOLERANCES["normalized_rtol"])
                                      for field in NORMALIZED_FIELDS)
                raw_pass = all(_within(cpu_components[field], gpu_components[field],
                                       PARITY_TOLERANCES["raw_atol"], PARITY_TOLERANCES["raw_rtol"])
                                for field in RAW_FIELDS)
                gradient_abs = _max_abs(cpu_gradient, gpu_gradient)
                gradient_rel = _max_rel(cpu_gradient, gpu_gradient)
                gradient_pass = bool(np.allclose(
                    cpu_gradient, gpu_gradient,
                    atol=PARITY_TOLERANCES["gradient_atol"],
                    rtol=PARITY_TOLERANCES["gradient_rtol"],
                ))
                row = {
                    "model": model,
                    "source": source_name,
                    "candidate_id": "consensus_joint" if source_name == "consensus" else "random_joint",
                    "stage": stage.label,
                    "bin_size_bp": int(stage.bin_size),
                    "n_loci": int(data.n_loci),
                    "state_hashes": state["state_hashes"],
                    "cpu_backend": "pr.allele_models.objective_for_model",
                    "gpu_backend": gpu_objective.backend_metadata(),
                    "cpu_elapsed_seconds": float(cpu_elapsed),
                    "gpu_elapsed_seconds": float(gpu_elapsed),
                    "cpu_value": float(cpu_value),
                    "gpu_value": float(gpu_value),
                    "value_abs_difference": abs(float(cpu_value) - float(gpu_value)),
                    "value_rel_difference": _max_rel([cpu_value], [gpu_value]),
                    "component_abs_difference": component_abs,
                    "component_rel_difference": component_rel,
                    "gradient_max_abs_difference": float(gradient_abs),
                    "gradient_max_rel_difference": float(gradient_rel),
                    "normalized_components_pass": normalized_pass,
                    "raw_components_pass": raw_pass,
                    "gradient_pass": gradient_pass,
                    "pass": bool(normalized_pass and raw_pass and gradient_pass),
                }
                rows.append(row)
                print("PARITY %s %s %s abs_total=%.3g abs_grad=%.3g pass=%s" % (
                    model, source_name, stage.label, row["value_abs_difference"],
                    row["gradient_max_abs_difference"], row["pass"]), flush=True)
    if not all(row["pass"] for row in rows):
        failed = [row for row in rows if not row["pass"]]
        raise RuntimeError("CUDA parity gate failed: " + json.dumps(failed[:2], sort_keys=True))
    return {
        "schema": "gpu-multires-cuda-parity-v1",
        "status": "passed",
        "rows": rows,
        "row_count": len(rows),
        "expected_row_count": len(VARIANTS) * len(CANDIDATES) * len(STAGES),
        "tolerances": dict(PARITY_TOLERANCES),
        "same_raw_theta_per_cpu_gpu": True,
        "selection_or_hyperparameter_use": False,
        "phase_used": False,
        "reference_used": False,
    }


def _c2_diagnostics(context: reconstruct.RunContext,
                    data_by_bin: dict[int, contact_model.AggregatedContacts]) -> dict[str, Any]:
    data = data_by_bin[1_000_000]
    names = tuple(name for name, _ in context.headers)
    lengths = tuple(int(length) for _, length in context.headers)
    state = reconstruction_init.initialize_approved_candidate("consensus", names, lengths, 1_000_000)
    base = np.asarray(state["coords"], dtype=np.float64)
    base_objective = allele_models.objective_for_model(data, "C2-map")
    base_raw = allele_models.raw_coordinates_from_physical(base_objective, base)
    map_raw = base_raw.copy()
    map_raw[0, 0] = np.asarray([1.2, 0.0, 0.0], dtype=np.float64)
    map_gpu = GPUVariantObjective(data, "C2-map", device="cuda", tile_rows=32, use_fused=True, diagnostics=True)
    map_theta = map_gpu.pack(map_raw, p=0.75)
    map_value, map_gradient, map_components = map_gpu.evaluate(map_theta, need_gradient=True)
    map_physical = map_gpu.physical_coordinates_from_raw(map_raw)
    map_diag = map_gpu.map_diagnostics()
    map_pass = bool(
        np.isfinite(map_value) and np.all(np.isfinite(map_gradient))
        and float(np.linalg.norm(map_raw, axis=2).max()) > 1.0
        and float(np.linalg.norm(map_physical, axis=2).max()) < 1.0
        and map_diag["nonidentity_map_eval_count"] >= 1
        and float(map_diag["max_physical_radius_seen"]) < 1.0
    )

    free_raw = base.copy()
    free_raw[0, 0] = np.asarray([1.2, 0.0, 0.0], dtype=np.float64)
    free_gpu = GPUVariantObjective(data, "C2-free", device="cuda", tile_rows=32, use_fused=True, diagnostics=True)
    free_theta = free_gpu.pack(free_raw, p=0.75)
    free_value, free_gradient, free_components = free_gpu.evaluate(free_theta, need_gradient=True)
    free_physical = free_gpu.physical_coordinates_from_raw(free_raw)
    free_diag = free_gpu.map_diagnostics()
    free_roundtrip_abs = _max_abs(free_physical, free_raw)
    free_pass = bool(
        np.isfinite(free_value) and np.all(np.isfinite(free_gradient))
        and float(np.linalg.norm(free_raw, axis=2).max()) > 1.0
        and free_roundtrip_abs == 0.0
        and float(np.linalg.norm(free_physical, axis=2).max()) > 1.0
        and free_diag["max_physical_radius_seen"] > 1.0
    )
    result = {
        "schema": "gpu-multires-c2-domain-diagnostics-v1",
        "status": "passed" if map_pass and free_pass else "failed",
        "c2_map": {
            "raw_outside_radius": float(np.linalg.norm(map_raw, axis=2).max()),
            "physical_max_radius": float(np.linalg.norm(map_physical, axis=2).max()),
            "value": float(map_value),
            "gradient_max_abs": float(np.max(np.abs(map_gradient))),
            "components": map_components,
            "map_diagnostics": map_diag,
            "physical_strict_unit_ball": True,
            "no_clip_or_rescale": True,
            "pass": map_pass,
        },
        "c2_free": {
            "raw_max_radius": float(np.linalg.norm(free_raw, axis=2).max()),
            "physical_max_radius": float(np.linalg.norm(free_physical, axis=2).max()),
            "roundtrip_max_abs": float(free_roundtrip_abs),
            "value": float(free_value),
            "gradient_max_abs": float(np.max(np.abs(free_gradient))),
            "components": free_components,
            "map_diagnostics": free_diag,
            "free_domain_preserved": True,
            "clip_applied": False,
            "pass": free_pass,
        },
        "phase_used": False,
        "reference_used": False,
        "selection_or_hyperparameter_use": False,
    }
    if result["status"] != "passed":
        raise RuntimeError("C2 domain diagnostic failed: " + json.dumps(result, sort_keys=True))
    return result


def _benchmark(context: reconstruct.RunContext,
               data_by_bin: dict[int, contact_model.AggregatedContacts],
               states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import torch
    from fused.fused_objective import load_fused_extension

    data = data_by_bin[1_000_000]
    cpu_objective = allele_models.objective_for_model(data, "C0")
    state = states["consensus"]["1m"]
    raw = allele_models.raw_coordinates_from_physical(cpu_objective, np.asarray(state["coords"], dtype=np.float64))
    theta = cpu_objective.pack(raw, p=0.75)
    compile_start = time.perf_counter()
    load_fused_extension(verbose=False)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_start
    torch.cuda.reset_peak_memory_stats()
    gpu_ctor_start = time.perf_counter()
    gpu_objective = GPUVariantObjective(data, "C0", device="cuda", tile_rows=32, use_fused=True, diagnostics=True)
    gpu_objective.synchronize()
    gpu_constructor_seconds = time.perf_counter() - gpu_ctor_start
    cpu_ctor_start = time.perf_counter()
    cpu_objective = allele_models.objective_for_model(data, "C0")
    cpu_constructor_seconds = time.perf_counter() - cpu_ctor_start

    # 使用不同的 q 值和直接 evaluate calls，确保两个 backend 都不能复用 cached result。
    thetas = []
    for index in range(1, 6):
        candidate = theta.copy()
        candidate[-1] += index * 1e-7
        thetas.append(candidate)
    for candidate in thetas[:2]:
        cpu_objective.evaluate(candidate, need_gradient=True)
        gpu_objective.evaluate(candidate, need_gradient=True)
    cpu_times = []
    gpu_times = []
    cpu_outputs = []
    gpu_outputs = []
    gpu_count_before = int(gpu_objective.map_diagnostics()["objective_eval_count"])
    for candidate in thetas[2:5]:
        cpu_start = time.perf_counter()
        cpu_value, cpu_gradient, _ = cpu_objective.evaluate(candidate, need_gradient=True)
        cpu_times.append(time.perf_counter() - cpu_start)
        cpu_outputs.append((float(cpu_value), float(np.max(np.abs(cpu_gradient)))))
        gpu_objective.synchronize()
        gpu_start = time.perf_counter()
        gpu_value, gpu_gradient, _ = gpu_objective.evaluate(candidate, need_gradient=True)
        gpu_objective.synchronize()
        gpu_times.append(time.perf_counter() - gpu_start)
        gpu_outputs.append((float(gpu_value), float(np.max(np.abs(gpu_gradient)))))
    gpu_count_after = int(gpu_objective.map_diagnostics()["objective_eval_count"])
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gpu_median = float(np.median(gpu_times))
    cpu_median = float(np.median(cpu_times))
    speedup = cpu_median / gpu_median
    result = {
        "schema": "gpu-multires-synchronized-benchmark-v1",
        "status": "passed",
        "scope": "real P9016 1 Mb C0 fixed-state value+gradient only",
        "device": {
            "name": torch.cuda.get_device_name(0),
            "index": 0,
            "total_memory_bytes": int(total_bytes),
            "free_memory_bytes_after": int(free_bytes),
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "compile_load_seconds_separate": float(compile_seconds),
        "gpu_constructor_upload_seconds": float(gpu_constructor_seconds),
        "cpu_constructor_seconds": float(cpu_constructor_seconds),
        "data_resident_on_gpu": bool(gpu_objective.backend_metadata()["data_resident_on_device"]),
        "warmup_count": 2,
        "timed_eval_count": 3,
        "timed_calls_need_gradient": True,
        "timed_calls_synchronized": True,
        "cache_hit_avoidance": {
            "direct_evaluate_used": True,
            "distinct_theta_q_offsets": [3e-7, 4e-7, 5e-7],
            "gpu_objective_eval_count_before": gpu_count_before,
            "gpu_objective_eval_count_after": gpu_count_after,
            "actual_gpu_eval_increment": gpu_count_after - gpu_count_before,
            "expected_gpu_eval_increment": 3,
        },
        "cpu_times_seconds": [float(v) for v in cpu_times],
        "gpu_times_seconds": [float(v) for v in gpu_times],
        "cpu_median_seconds": cpu_median,
        "gpu_median_seconds": gpu_median,
        "cpu_mean_seconds": float(np.mean(cpu_times)),
        "gpu_mean_seconds": float(np.mean(gpu_times)),
        "cpu_gpu_median_speedup": float(speedup),
        "cpu_outputs": cpu_outputs,
        "gpu_outputs": gpu_outputs,
        "backend": gpu_objective.backend_metadata(),
        "phase_used": False,
        "reference_used": False,
        "selection_or_hyperparameter_use": False,
    }
    if result["cache_hit_avoidance"]["actual_gpu_eval_increment"] != 3:
        raise RuntimeError("benchmark did not execute exactly three timed GPU evaluations")
    return result


def _controller_smoke(context: reconstruct.RunContext,
                      data_by_bin: dict[int, contact_model.AggregatedContacts]) -> dict[str, Any]:
    import json as _json

    root = PREFLIGHT_DIR / "controller_smoke"
    if root.exists():
        raise FileExistsError(f"controller smoke directory already exists: {root}")
    root.mkdir(parents=True)
    data = data_by_bin[5_000_000]
    candidate = CANDIDATES[0]
    names = tuple(name for name, _ in context.headers)
    lengths = tuple(int(length) for _, length in context.headers)
    state = reconstruction_init.initialize_approved_candidate("consensus", names, lengths, 5_000_000)
    positions = np.asarray(data.locus_bin, dtype=np.int64) * int(data.bin_size)
    chromosome_index = np.asarray(data.locus_chromosome, dtype=np.int64)
    stage = SimpleNamespace(label="smoke", bin_size=5_000_000, maxiter=1, maxfun=2)
    old_checkpoint = gpu_controller.CHECKPOINT_EVERY
    gpu_controller.CHECKPOINT_EVERY = 1
    try:
        result, fit_payload, final_coordinates = gpu_controller._fit_gpu(
            root, "C0", candidate, stage, data, np.asarray(state["coords"], dtype=np.float64),
            positions, chromosome_index, None, "cuda")
    finally:
        gpu_controller.CHECKPOINT_EVERY = old_checkpoint
    initial_path = root / "coords" / "initial-smoke.3dg"
    final_path = root / "coords" / "final-smoke.3dg"
    initial_path.parent.mkdir(parents=True, exist_ok=True)
    initial_info = paired_run.write_coordinates(initial_path, data, "C0", np.asarray(state["coords"], dtype=np.float64))
    final_info = paired_run.write_coordinates(final_path, data, "C0", np.asarray(final_coordinates, dtype=np.float64))
    checkpoint_paths = [root / relative for relative in fit_payload["checkpoint_paths"]]
    checkpoint_checks = []
    for checkpoint in checkpoint_paths:
        with np.load(checkpoint, allow_pickle=False) as payload:
            required = {"coordinates", "theta", "y", "positions", "chromosome_index", "fullhistory_json"}
            keys = set(payload.files)
            if keys != required:
                raise AssertionError(f"checkpoint schema changed: {keys}")
            checkpoint_checks.append({
                "path": str(checkpoint.relative_to(root)),
                "keys": sorted(keys),
                "coordinates_shape": [int(v) for v in payload["coordinates"].shape],
                "theta_shape": [int(v) for v in payload["theta"].shape],
                "positions_shape": [int(v) for v in payload["positions"].shape],
                "chromosome_index_shape": [int(v) for v in payload["chromosome_index"].shape],
            })
    stage_path = root / "stage-smoke.json"
    stage_record = {
        "schema": "gpu-multires-controller-smoke-stage-v1",
        "fit": _jsonable(fit_payload),
        "initial_coordinates": {**initial_info, "path": str(initial_path.relative_to(root))},
        "final_coordinates": {**final_info, "path": str(final_path.relative_to(root))},
    }
    _write_json(stage_path, stage_record)
    result_payload = {
        "schema": "gpu-multires-controller-smoke-v1",
        "status": "passed",
        "scope": "one real 5 Mb C0 stage, maxiter=1/maxfun=2, checkpoint_every=1",
        "stage_record": str(stage_path.relative_to(root)),
        "checkpoint_checks": checkpoint_checks,
        "checkpoint_count": len(checkpoint_checks),
        "initial_coordinates": {**initial_info, "path": str(initial_path.relative_to(root))},
        "final_coordinates": {**final_info, "path": str(final_path.relative_to(root))},
        "solver": {
            "status": fit_payload["status"],
            "budget_exhausted": bool(fit_payload["budget_exhausted"]),
            "actual_nfev": int(fit_payload["solver"]["actual_nfev"]),
            "q_out": float(fit_payload["q_out"]),
        },
        "phase_used": False,
        "reference_used": False,
        "formal_training_started": False,
    }
    result_sha = _write_json(root / "smoke_result.json", result_payload)
    result_payload["result_sha256"] = result_sha
    return result_payload


def _prepare_terminal_metadata(context: reconstruct.RunContext, preflight_path: Path) -> dict[str, Any]:
    # 这里仅准备 metadata。30-stage formal output 保持为空且不触碰。
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    formal_output = ROOT / "test_res" / f"036-{stamp}-gpu-multires"
    if formal_output.exists():
        raise FileExistsError(f"formal output suggestion already exists: {formal_output}")
    protocol = gpu_controller._protocol(context, preflight_path.resolve(), ARTIFACT)
    config_path = ARTIFACT / "config.json"
    protocol_path = ARTIFACT / "protocol.json"
    source_hash_path = ARTIFACT / "provenance" / "source_hashes.json"
    protocol_sha_written = _write_json(protocol_path, protocol)
    config = gpu_controller._config(context, protocol, protocol_sha_written, formal_output)
    config_sha = _write_json(config_path, config)
    source_hashes = {
        "schema": "gpu-multires-source-hashes-v1",
        "source_code_sha256": gpu_controller._source_hashes(),
        "protocol_sha256": protocol_sha_written,
        "config_sha256": config_sha,
        "input_sha256": str(context.data_sha256),
        "approved_014_sources": {key: value["sha256"] for key, value in context.source_assets.items()},
        "c0_gate_sha256": sha256_file(ROOT / "test_res/033-20260914_044246-c0-controlled-reproduction/C0_gate.json"),
    }
    source_hash_sha = _write_json(source_hash_path, source_hashes)
    launch_path = ARTIFACT / "launch_command.txt"
    launch_payload = (
        "source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh\n"
        "conda activate analysis\n"
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n"
        f"python {str((SOURCE / 'gpu_multires_controller.py').resolve())} run "
        f"--artifact {str(ARTIFACT.resolve())} --output {str(formal_output.resolve())} "
        "--device cuda --workers 1\n"
    ).encode("ascii")
    with launch_path.open("xb") as handle:
        handle.write(launch_payload)
        handle.flush()
        os.fsync(handle.fileno())
    launch_sha = hashlib.sha256(launch_payload).hexdigest()
    return {
        "formal_output_suggestion": str(formal_output),
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha_written},
        "config": {"path": str(config_path), "sha256": config_sha},
        "source_hashes": {"path": str(source_hash_path), "sha256": source_hash_sha},
        "launch_command": {"path": str(launch_path), "sha256": launch_sha},
        "planned_attempt_count": len(gpu_controller._planned_rows()),
        "protocol_sha256_recomputed": protocol_sha_written,
    }


def main() -> int:
    context = reconstruct.production_context()
    reconstruct._validate_context(context)
    actual_input_sha = sha256_file(Path(context.data_path))
    if actual_input_sha != EXPECTED_INPUT_SHA:
        raise RuntimeError(f"input SHA changed: {actual_input_sha}")
    for name, expected in EXPECTED_SOURCE_SHA.items():
        source_path = Path(context.source_assets[name]["path"])
        actual = sha256_file(source_path)
        if actual != expected:
            raise RuntimeError(f"014 {name} SHA changed: {actual}")
    mature_paths = {
        "gpu_backend.py": ROOT / "test_res/028-20260913_151456-020-gpu-independent/source/gpu_backend.py",
        "fused_objective.py": ROOT / "test_res/028-20260913_151456-020-gpu-independent/source/fused/fused_objective.py",
        "cuda_pair_objective.cu": ROOT / "test_res/028-20260913_151456-020-gpu-independent/source/fused/cuda_pair_objective.cu",
    }
    mature_hashes = {name: sha256_file(path) for name, path in mature_paths.items()}
    if mature_hashes != EXPECTED_BACKEND_SHA:
        raise RuntimeError(f"mature backend hashes changed: {mature_hashes}")

    data_by_bin = {int(stage.bin_size): contact_model.load_frozen_p9016_aggregate(int(stage.bin_size))
                   for stage in STAGES}
    data_manifest = _data_manifest(context, data_by_bin)
    print("DATA budgets and array hashes passed", flush=True)
    parity = _parity(context, data_by_bin)
    c2 = _c2_diagnostics(context, data_by_bin)
    states_for_benchmark = {
        "consensus": _load_states(context, "consensus", "C0", data_by_bin),
    }
    benchmark = _benchmark(context, data_by_bin, states_for_benchmark)
    print("BENCHMARK median_cpu=%.6fs median_gpu=%.6fs speedup=%.3fx" % (
        benchmark["cpu_median_seconds"], benchmark["gpu_median_seconds"], benchmark["cpu_gpu_median_speedup"]), flush=True)
    smoke = _controller_smoke(context, data_by_bin)
    preflight = {
        "schema": "gpu-multires-preflight-v1",
        "status": "passed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "CUDA scientific/performance/controller preflight only; formal 30-stage run not launched",
        "input": {
            "path": str(Path(context.data_path).resolve()),
            "sha256": actual_input_sha,
            "records": int(context.cohort["raw_contacts"]),
            "cis_records": int(context.cohort["intra_contacts"]),
            "inter_records": int(context.cohort["inter_contacts"]),
            "training_boundary": {"phase_used": False, "reference_used": False, "r2_used": False, "evaluation_used": False},
        },
        "approved_sources": {
            name: {"path": str(context.source_assets[name]["path"]), "sha256": digest,
                   "candidate_id": "consensus_joint" if name == "consensus" else "random_joint",
                   "base_seed": 1103 if name == "consensus" else 2207}
            for name, digest in EXPECTED_SOURCE_SHA.items()
        },
        "mature_backend_sha256": mature_hashes,
        "data_manifest": data_manifest,
        "parity": {"path": str(PREFLIGHT_DIR / "cuda_parity.json"), "summary": parity},
        "c2_diagnostics": {"path": str(PREFLIGHT_DIR / "c2_domain_diagnostics.json"), "summary": c2},
        "benchmark": {"path": str(PREFLIGHT_DIR / "synchronized_benchmark.json"), "summary": benchmark},
        "controller_smoke": {"path": str(PREFLIGHT_DIR / "controller_smoke/smoke_result.json"), "summary": smoke},
        "tolerances": dict(PARITY_TOLERANCES),
        "formal_run": {"launched": False, "planned_variants": list(VARIANTS), "planned_sources": [c.candidate_id for c in CANDIDATES], "planned_stages": [s.label for s in STAGES]},
    }
    # 先写出 component evidence；随后 main preflight document 不可变。
    parity_sha = _write_json(PREFLIGHT_DIR / "cuda_parity.json", parity)
    c2_sha = _write_json(PREFLIGHT_DIR / "c2_domain_diagnostics.json", c2)
    benchmark_sha = _write_json(PREFLIGHT_DIR / "synchronized_benchmark.json", benchmark)
    preflight["evidence_sha256"] = {
        "cuda_parity.json": parity_sha,
        "c2_domain_diagnostics.json": c2_sha,
        "synchronized_benchmark.json": benchmark_sha,
        "controller_smoke/smoke_result.json": smoke.get("result_sha256"),
    }
    preflight_path = ARTIFACT / "preflight.json"
    preflight_sha = _write_json(preflight_path, preflight)
    metadata = _prepare_terminal_metadata(context, preflight_path)
    terminal = {
        "schema": "gpu-multires-preflight-terminal-v1",
        "status": "preflight_passed_formal_pending_parent_acceptance",
        "preflight": {"path": str(preflight_path), "sha256": preflight_sha},
        "metadata": metadata,
        "formal_run_launched": False,
        "planned_attempt_count": len(gpu_controller._planned_rows()),
        "phase_used": False,
        "reference_used": False,
    }
    terminal_sha = _write_json(ARTIFACT / "terminal_prep.json", terminal)
    print(json.dumps({
        "status": "passed",
        "preflight": str(preflight_path),
        "preflight_sha256": preflight_sha,
        "terminal_prep": str(ARTIFACT / "terminal_prep.json"),
        "terminal_prep_sha256": terminal_sha,
        "metadata": metadata,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise
