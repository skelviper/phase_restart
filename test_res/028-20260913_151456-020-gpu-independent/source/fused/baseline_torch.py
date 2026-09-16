"""Run the authorized existing TorchObjective CUDA 1 Mb baseline.

This is a diagnostic baseline only. It does not modify shared code or start the
formal two-candidate pipeline. Output is printed so the caller can persist the
JSON under the exclusive fused log scope.
"""
from __future__ import annotations

import json
import os
import argparse
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
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ARCHIVED))

from gpu_backend import TorchObjective, load_sparse_aggregate  # noqa: E402
from pr.contact_model import JointObjective as ArchivedObjective  # noqa: E402
from pr.contact_model import load_aggregate  # noqa: E402

INPUT = WORKSPACE / "inputs" / "P9016.snpfree.pairs.gz"
CHECKPOINT = WORKSPACE / "test_res" / "020-20260913_071841-v1-p9016-joint" / "checkpoints" / "random_joint" / "1m-accepted-0240.npz"


def json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def numeric_components(components):
    return {str(k): float(v) for k, v in components.items() if isinstance(v, (float, int))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN / "logs" / "fused-torch-baseline-1m-machine.json",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("authorized GPU baseline requires exactly one visible CUDA device")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    checkpoint = np.load(CHECKPOINT, allow_pickle=False)
    theta = np.asarray(checkpoint["theta"], dtype=np.float64)
    if theta.ndim != 1:
        raise RuntimeError(f"checkpoint theta has invalid shape {theta.shape}")

    preprocess_started = time.perf_counter()
    sparse = load_sparse_aggregate(INPUT, 1_000_000, verify_hash=True)
    preprocess_seconds = time.perf_counter() - preprocess_started

    torch.cuda.synchronize(device)
    upload_started = time.perf_counter()
    gpu_objective = TorchObjective(sparse, device="cuda", tile_rows=32, dtype=torch.float64)
    torch.cuda.synchronize(device)
    upload_seconds = time.perf_counter() - upload_started

    torch.cuda.reset_peak_memory_stats(device)
    gpu_objective.synchronize()
    warmup_started = time.perf_counter()
    warmup_value, warmup_gradient, warmup_components = gpu_objective.evaluate(theta, need_gradient=True)
    gpu_objective.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started
    if warmup_gradient is None or not np.all(np.isfinite(warmup_gradient)):
        raise RuntimeError("Torch CUDA warmup produced a nonfinite gradient")

    steady = []
    for repeat in range(3):
        gpu_objective.synchronize()
        started = time.perf_counter()
        value, gradient, components = gpu_objective.evaluate(theta, need_gradient=True)
        gpu_objective.synchronize()
        elapsed = time.perf_counter() - started
        steady.append({
            "repeat": repeat + 1,
            "elapsed_seconds": elapsed,
            "value": float(value),
            "gradient_l2": float(np.linalg.norm(gradient)),
            "gradient_max_abs": float(np.max(np.abs(gradient))),
            "components": numeric_components(components),
        })

    cpu_load_started = time.perf_counter()
    archived_data = load_aggregate(INPUT, bin_size=1_000_000, verify_frozen_hash=True)
    cpu_load_seconds = time.perf_counter() - cpu_load_started
    archived_objective = ArchivedObjective(archived_data, block_size=65_536, repulsion_block_size=65_536)
    cpu_started = time.perf_counter()
    cpu_value, cpu_gradient, cpu_components = archived_objective.evaluate(theta, need_gradient=True)
    cpu_elapsed = time.perf_counter() - cpu_started
    final = steady[-1]
    gpu_components = final["components"]
    component_differences = {
        key: float(gpu_components[key] - numeric_components(cpu_components)[key])
        for key in gpu_components.keys() & numeric_components(cpu_components).keys()
    }
    component_max_abs = float(max(abs(v) for v in component_differences.values()))
    gradient_max_abs = float(np.max(np.abs(np.asarray(warmup_gradient) - cpu_gradient)))
    if component_max_abs > 1.0e-6 or gradient_max_abs > 1.0e-8:
        raise RuntimeError(
            "existing Torch CUDA baseline disagrees with archived CPU: "
            f"component={component_max_abs:.17g}, gradient={gradient_max_abs:.17g}")
    payload = {
        "artifact_integrity": "machine_generated_direct_json_dump",
        "status": "completed",
        "purpose": "authorized_existing_torch_cuda_baseline_before_fused_extension",
        "input": str(INPUT),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_theta_shape": list(theta.shape),
        "device": {"index": 0, "name": props.name, "total_memory": int(props.total_memory)},
        "dtype": "torch.float64",
        "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "sparse_budget": json_ready(sparse.budget()),
        "timing": {
            "preprocess_seconds": preprocess_seconds,
            "upload_seconds": upload_seconds,
            "warmup_seconds_synchronized": warmup_seconds,
            "steady_evaluations": steady,
            "steady_mean_seconds": float(np.mean([row["elapsed_seconds"] for row in steady])),
            "steady_min_seconds": float(np.min([row["elapsed_seconds"] for row in steady])),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "archived_cpu_load_seconds": cpu_load_seconds,
            "archived_cpu_block_size": 65_536,
            "archived_cpu_repulsion_block_size": 65_536,
            "archived_cpu_evaluate_seconds": cpu_elapsed,
        },
        "warmup": {
            "value": float(warmup_value),
            "gradient_l2": float(np.linalg.norm(warmup_gradient)),
            "gradient_max_abs": float(np.max(np.abs(warmup_gradient))),
            "components": numeric_components(warmup_components),
        },
        "gpu_final": {"value": float(final["value"]), "components": gpu_components},
        "archived_cpu": {
            "value": float(cpu_value),
            "gradient_l2": float(np.linalg.norm(cpu_gradient)),
            "gradient_max_abs": float(np.max(np.abs(cpu_gradient))),
            "components": numeric_components(cpu_components),
        },
        "gpu_minus_archived_cpu": {
            "value": float(final["value"] - cpu_value),
            "max_abs_component": float(max(abs(v) for v in component_differences.values())),
            "components": component_differences,
            "gradient_max_abs": gradient_max_abs,
        },
        "gradient_comparison": {
            "max_abs": float(np.max(np.abs(np.asarray(warmup_gradient) - cpu_gradient))),
            "gpu_l2": float(np.linalg.norm(warmup_gradient)),
            "cpu_l2": float(np.linalg.norm(cpu_gradient)),
        },
        "phase_or_reference_opened": False,
        "formal_pipeline_started": False,
    }
    text = json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
