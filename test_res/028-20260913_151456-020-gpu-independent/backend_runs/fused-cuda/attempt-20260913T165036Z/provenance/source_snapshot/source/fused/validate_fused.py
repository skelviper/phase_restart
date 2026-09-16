"""GPU exactness gates for the fused ordered-row objective.

The driver is intentionally separate from the formal pipeline. It evaluates a
small fixture and one phase-free real 020 checkpoint, compares to archived NumPy,
and writes a machine-generated audit in the exclusive fused log scope.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import torch

RUN = Path(__file__).resolve().parents[2]
WORKSPACE = RUN.parents[1]
SOURCE = RUN / "source"
ARCHIVED = SOURCE / "archived020"
LOG = RUN / "logs" / "fused-validation.json"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ARCHIVED))

from fused.fused_objective import FusedObjective, load_fused_extension  # noqa: E402
from gpu_backend import load_sparse_aggregate  # noqa: E402
from validate_backend import fixture_sparse  # noqa: E402
from pr.contact_model import JointObjective as ArchivedObjective  # noqa: E402
from pr.contact_model import load_aggregate  # noqa: E402

INPUT = WORKSPACE / "inputs" / "P9016.snpfree.pairs.gz"
CHECKPOINT = WORKSPACE / "test_res" / "020-20260913_071841-v1-p9016-joint" / "checkpoints" / "random_joint" / "1m-accepted-0240.npz"


def numeric_components(components):
    return {str(k): float(v) for k, v in components.items() if isinstance(v, (float, int))}


def component_errors(left, right):
    left_num = numeric_components(left)
    right_num = numeric_components(right)
    common = sorted(left_num.keys() & right_num.keys())
    return {key: float(left_num[key] - right_num[key]) for key in common}


def max_abs(values):
    return float(max((abs(float(value)) for value in values.values()), default=0.0))


def evaluate_fixture(device: str):
    data, extra = fixture_sparse()
    archived_data = extra["archived"]
    rng = np.random.default_rng(9417)
    y = rng.normal(0.0, 0.13, size=(2, data.n_loci, 3)).astype(np.float64)
    archived = ArchivedObjective(archived_data, block_size=3, repulsion_block_size=3)
    theta = archived.pack(y, p=0.67)
    archived_value, archived_gradient, archived_components = archived.evaluate(theta, need_gradient=True)
    fused = FusedObjective(data, device=device, tile_rows=2, dtype=torch.float64)
    fused_value, fused_gradient, fused_components = fused.evaluate(theta, need_gradient=True)
    nondefault_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(nondefault_stream):
        stream_value, stream_gradient, stream_components = fused.evaluate(theta, need_gradient=True)
    nondefault_stream.synchronize()
    stream_component_errors = component_errors(stream_components, fused_components)
    stream_observed = {
        "value_abs": abs(float(stream_value - fused_value)),
        "gradient_max_abs": float(np.max(np.abs(stream_gradient - fused_gradient))),
        "component_errors": stream_component_errors,
        "component_max_abs": max_abs(stream_component_errors),
    }
    errors = component_errors(fused_components, archived_components)
    finite_difference = {}
    for index in np.linspace(0, len(theta) - 1, num=min(12, len(theta)), dtype=int):
        step = 1.0e-6
        plus = theta.copy(); plus[index] += step
        minus = theta.copy(); minus[index] -= step
        f_plus = fused.evaluate(plus, need_gradient=False)[0]
        f_minus = fused.evaluate(minus, need_gradient=False)[0]
        finite_difference[str(int(index))] = float(
            (f_plus - f_minus) / (2.0 * step) - fused_gradient[index])

    zero_y = np.zeros_like(y)
    zero_theta = archived.pack(zero_y, p=0.5)
    zero_archived_value, zero_archived_gradient, zero_archived_components = archived.evaluate(
        zero_theta, need_gradient=True)
    zero_fused_value, zero_fused_gradient, zero_fused_components = fused.evaluate(
        zero_theta, need_gradient=True)
    zero_component_errors = component_errors(zero_fused_components, zero_archived_components)

    chromosome_swap_y = y.copy()
    chromosome_mask = data.locus_chromosome == data.locus_chromosome[0]
    chromosome_swap_y[:, chromosome_mask] = y[::-1, chromosome_mask]
    chromosome_swap_theta = archived.pack(chromosome_swap_y, p=0.67)
    chromosome_swap_value, _, chromosome_swap_components = fused.evaluate(
        chromosome_swap_theta, need_gradient=False)

    local_swap_y = y.copy()
    local_swap_y[:, 0] = y[::-1, 0]
    half_theta = archived.pack(y, p=0.5)
    half_value, _, half_components = fused.evaluate(half_theta, need_gradient=False)
    local_swap_theta = archived.pack(local_swap_y, p=0.5)
    local_swap_value, _, local_swap_components = fused.evaluate(local_swap_theta, need_gradient=False)

    fixture_thresholds = {
        "value_abs": 1.0e-11,
        "gradient_max_abs": 1.0e-9,
        "finite_difference_max_abs": 2.0e-6,
        "copy_swap_value_abs": 1.0e-11,
        "zero_distance_value_abs": 1.0e-11,
        "zero_distance_gradient_max_abs": 1.0e-9,
        "zero_distance_component_max_abs": 1.0e-11,
        "p_half_local_swap_count_abs": 1.0e-11,
        "nondefault_stream_value_abs": 1.0e-11,
        "nondefault_stream_gradient_max_abs": 1.0e-9,
        "nondefault_stream_component_max_abs": 1.0e-11,
    }
    fixture_observed = {
        "value_abs": abs(float(fused_value - archived_value)),
        "gradient_max_abs": float(np.max(np.abs(fused_gradient - archived_gradient))),
        "component_errors": errors,
        "component_max_abs": max_abs(errors),
        "finite_difference_errors": finite_difference,
        "finite_difference_max_abs": max_abs(finite_difference),
        "copy_swap_value_abs": abs(float(chromosome_swap_value - fused_value)),
        "zero_distance_value_abs": abs(float(zero_fused_value - zero_archived_value)),
        "zero_distance_gradient_max_abs": float(np.max(np.abs(zero_fused_gradient - zero_archived_gradient))),
        "zero_distance_component_errors": zero_component_errors,
        "zero_distance_component_max_abs": max_abs(zero_component_errors),
        "p_half_local_swap_count_abs": abs(float(
            local_swap_components["count_nll_normalized"] - half_components["count_nll_normalized"])),
        "p_half_local_swap_total_abs": abs(float(local_swap_value - half_value)),
        "nondefault_stream": stream_observed,
        "zero_distance_components": numeric_components(zero_fused_components),
    }
    fixture_passed = (
        fixture_observed["value_abs"] <= fixture_thresholds["value_abs"]
        and fixture_observed["gradient_max_abs"] <= fixture_thresholds["gradient_max_abs"]
        and fixture_observed["finite_difference_max_abs"] <= fixture_thresholds["finite_difference_max_abs"]
        and fixture_observed["copy_swap_value_abs"] <= fixture_thresholds["copy_swap_value_abs"]
        and fixture_observed["zero_distance_value_abs"] <= fixture_thresholds["zero_distance_value_abs"]
        and fixture_observed["zero_distance_gradient_max_abs"] <= fixture_thresholds["zero_distance_gradient_max_abs"]
        and fixture_observed["zero_distance_component_max_abs"] <= fixture_thresholds["zero_distance_component_max_abs"]
        and fixture_observed["p_half_local_swap_count_abs"] <= fixture_thresholds["p_half_local_swap_count_abs"]
        and fixture_observed["nondefault_stream"]["value_abs"] <= fixture_thresholds["nondefault_stream_value_abs"]
        and fixture_observed["nondefault_stream"]["gradient_max_abs"] <= fixture_thresholds["nondefault_stream_gradient_max_abs"]
        and fixture_observed["nondefault_stream"]["component_max_abs"] <= fixture_thresholds["nondefault_stream_component_max_abs"]
    )
    return {
        "passed": bool(fixture_passed),
        "budget": data.budget(),
        "thresholds": fixture_thresholds,
        "observed": fixture_observed,
        "phase_or_reference_opened": False,
    }


def evaluate_real_1m(device: str):
    sparse = load_sparse_aggregate(INPUT, 1_000_000, verify_hash=True)
    archived_data = load_aggregate(INPUT, bin_size=1_000_000, verify_frozen_hash=True)
    checkpoint = np.load(CHECKPOINT, allow_pickle=False)
    theta = np.asarray(checkpoint["theta"], dtype=np.float64)
    fused = FusedObjective(sparse, device=device, tile_rows=32, dtype=torch.float64)
    fused.synchronize()
    warmup_started = time.perf_counter()
    warmup_value, warmup_gradient, warmup_components = fused.evaluate(theta, need_gradient=True)
    fused.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started
    steady = []
    for repeat in range(3):
        fused.synchronize()
        started = time.perf_counter()
        value, gradient, components = fused.evaluate(theta, need_gradient=True)
        fused.synchronize()
        elapsed = time.perf_counter() - started
        steady.append({
            "repeat": repeat + 1,
            "elapsed_seconds": elapsed,
            "value": float(value),
            "gradient_l2": float(np.linalg.norm(gradient)),
            "gradient_max_abs": float(np.max(np.abs(gradient))),
            "components": numeric_components(components),
        })

    archived = ArchivedObjective(archived_data, block_size=65_536, repulsion_block_size=65_536)
    cpu_started = time.perf_counter()
    cpu_value, cpu_gradient, cpu_components = archived.evaluate(theta, need_gradient=True)
    cpu_seconds = time.perf_counter() - cpu_started
    fused_components = numeric_components(warmup_components)
    archived_components = numeric_components(cpu_components)
    errors = {key: float(fused_components[key] - archived_components[key])
              for key in sorted(fused_components.keys() & archived_components.keys())}
    raw_keys = (
        "sum_rate_cis_offdiag", "sum_rate_inter", "observed_log_rate_cis_offdiag",
        "observed_log_rate_inter", "conditional_nll_raw", "diag_profiled_nll_raw",
        "count_nll_raw",
    )
    raw_tolerances = {
        key: 1.0e-10 * max(1.0, abs(archived_components[key])) for key in raw_keys
    }
    raw_component_strict_max_abs = max(abs(errors.get(key, 0.0)) for key in raw_keys)
    normalized_error = abs(float(fused_components["count_nll_normalized"] - archived_components["count_nll_normalized"]))
    gradient_error = float(np.max(np.abs(warmup_gradient - cpu_gradient)))
    real_thresholds = {
        "count_nll_normalized_abs": 1.0e-10,
        "gradient_max_abs": 1.0e-8,
        "raw_component_relative": 1.0e-10,
        "raw_component_abs_strict": 1.0e-6,
    }
    real_passed = (
        normalized_error <= real_thresholds["count_nll_normalized_abs"]
        and gradient_error <= real_thresholds["gradient_max_abs"]
        and raw_component_strict_max_abs <= real_thresholds["raw_component_abs_strict"]
        and all(abs(errors.get(key, 0.0)) <= raw_tolerances[key] for key in raw_keys)
    )
    return {
        "passed": bool(real_passed),
        "checkpoint": str(CHECKPOINT),
        "theta_shape": list(theta.shape),
        "budget": sparse.budget(),
        "thresholds": real_thresholds,
        "raw_component_tolerances": raw_tolerances,
        "warmup": {
            "elapsed_seconds": warmup_seconds,
            "value": float(warmup_value),
            "gradient_l2": float(np.linalg.norm(warmup_gradient)),
            "gradient_max_abs": float(np.max(np.abs(warmup_gradient))),
            "components": numeric_components(warmup_components),
        },
        "steady": steady,
        "steady_mean_seconds": float(np.mean([row["elapsed_seconds"] for row in steady])),
        "steady_min_seconds": float(np.min([row["elapsed_seconds"] for row in steady])),
        "archived_cpu": {
            "block_size": 65_536,
            "repulsion_block_size": 65_536,
            "elapsed_seconds": cpu_seconds,
            "value": float(cpu_value),
            "gradient_l2": float(np.linalg.norm(cpu_gradient)),
            "gradient_max_abs": float(np.max(np.abs(cpu_gradient))),
            "components": archived_components,
        },
        "fused_minus_archived": {
            "value": float(warmup_value - cpu_value),
            "count_nll_normalized_abs": normalized_error,
            "gradient_max_abs": gradient_error,
            "raw_component_strict_max_abs": float(raw_component_strict_max_abs),
            "components": errors,
        },
        "backend": fused.backend_metadata(),
        "phase_or_reference_opened": False,
    }


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("fused validation requires exactly one visible CUDA device")
    torch.set_num_threads(1)
    extension = load_fused_extension(verbose=False)
    del extension
    started = time.perf_counter()
    fixture = evaluate_fixture("cuda")
    if not fixture["passed"]:
        payload = {
            "status": "failed_fixture",
            "fixture": fixture,
            "elapsed_seconds": time.perf_counter() - started,
            "formal_pipeline_started": False,
            "phase_or_reference_opened": False,
        }
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        LOG.write_text(text, encoding="utf-8")
        print(text, end="")
        raise SystemExit(1)
    real = evaluate_real_1m("cuda")
    payload = {
        "status": "passed" if real["passed"] else "failed_real_1m",
        "fixture": fixture,
        "real_1m": real,
        "elapsed_seconds": time.perf_counter() - started,
        "formal_pipeline_started": False,
        "phase_or_reference_opened": False,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    LOG.write_text(text, encoding="utf-8")
    print(text, end="")
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
