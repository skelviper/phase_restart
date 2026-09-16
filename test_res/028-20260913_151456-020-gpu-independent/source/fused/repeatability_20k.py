"""Post-fit repeatability gate for the saved 20 kb fused result.

This is read-only with respect to the formal attempt: it loads the saved theta
and coordinates, performs three identical full objective+gradient evaluations,
and writes only a diagnostic log outside the attempt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

RUN = Path(__file__).resolve().parents[2]
WORKSPACE = RUN.parents[1]
SOURCE = RUN / "source"
ATTEMPT = RUN / "backend_runs" / "fused-cuda" / "attempt-20260913T165036Z"
BACKEND = ATTEMPT / "backend_runs" / "fused-cuda"
LOG = RUN / "logs" / "fused-final-20k-repeatability.json"
INPUT = WORKSPACE / "inputs" / "P9016.snpfree.pairs.gz"
STATE_NPZ = BACKEND / "resume" / "random_joint" / "20k-state.npz"
STATE_JSON = BACKEND / "resume" / "random_joint" / "20k-state.json"
STAGE_JSON = BACKEND / "stages" / "random_joint" / "20k.json"
COORDS_FILE = BACKEND / "coords" / "random_joint" / "final-20k-14deef2734fb.3dg"
CONFIG = ATTEMPT / "config.json"
BUILD_LOG = RUN / "logs" / "fused-build.json"
EXPECTED_CONFIG = "14deef2734fb0b13798d394d6bf0336c23c6e2a3958ec60f532c48f959e6ae7e"
EXPECTED_SOURCE = "f100d691ead66d3ffade72d027d1456fe36f167619e892eb0e076cd31acfcdd7"
sys.path.insert(0, str(SOURCE))

from fused import FusedObjective, load_fused_extension  # noqa: E402
from gpu_backend import load_sparse_aggregate  # noqa: E402
from run_pipeline import array_sha256  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_components(components: dict) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in components.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def component_errors(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(left) | set(right))
    return {key: float(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys}


def max_abs(values: dict[str, float]) -> float:
    return max((abs(value) for value in values.values()), default=0.0)


def main() -> None:
    started = time.perf_counter()
    required = (INPUT, STATE_NPZ, STATE_JSON, STAGE_JSON, COORDS_FILE, CONFIG, BUILD_LOG)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("missing saved 20 kb repeatability input/artifact")
    state = json.loads(STATE_JSON.read_text(encoding="utf-8"))
    stage = json.loads(STAGE_JSON.read_text(encoding="utf-8"))
    saved = np.load(STATE_NPZ, allow_pickle=False)
    theta = np.asarray(saved["theta"], dtype=np.float64)
    saved_coordinates = np.asarray(saved["coordinates"], dtype=np.float64)
    if theta.shape != (790201,) or saved_coordinates.shape != (2, 131700, 3):
        raise RuntimeError(f"saved shape mismatch: theta={theta.shape}, coordinates={saved_coordinates.shape}")
    theta_hash = array_sha256(theta)
    saved_coordinates_hash = array_sha256(saved_coordinates)
    if theta_hash != state["theta_sha256"]:
        raise RuntimeError("saved theta canonical hash disagrees with state metadata")
    if saved_coordinates_hash != state["coordinates_sha256"]:
        raise RuntimeError("saved coordinates canonical hash disagrees with state metadata")

    load_started = time.perf_counter()
    data = load_sparse_aggregate(INPUT, 20_000, verify_hash=True)
    load_seconds = time.perf_counter() - load_started
    extension_started = time.perf_counter()
    load_fused_extension(verbose=False)
    extension_load_seconds = time.perf_counter() - extension_started
    upload_started = time.perf_counter()
    objective = FusedObjective(data, device="cuda", tile_rows=32, dtype=torch.float64)
    objective.synchronize()
    upload_seconds = time.perf_counter() - upload_started

    coordinate_started = time.perf_counter()
    reconstructed_coordinates, reconstructed_p = objective.coordinates_and_p(theta)
    objective.synchronize()
    coordinate_seconds = time.perf_counter() - coordinate_started
    coordinate_diff = reconstructed_coordinates - saved_coordinates
    coordinate_max_abs = float(np.max(np.abs(coordinate_diff)))
    coordinate_bitwise = bool(np.array_equal(reconstructed_coordinates, saved_coordinates))
    reconstructed_coordinates_hash = array_sha256(reconstructed_coordinates)
    radii = np.linalg.norm(reconstructed_coordinates, axis=-1)
    finite_ball = bool(np.isfinite(reconstructed_coordinates).all() and (radii < 1.0).all())

    torch.cuda.reset_peak_memory_stats(objective.device)
    objective.synchronize()
    repeats = []
    for repeat in range(3):
        objective_started = time.perf_counter()
        value, gradient, components = objective.evaluate(theta, need_gradient=True)
        objective.synchronize()
        elapsed = time.perf_counter() - objective_started
        if gradient is None:
            raise RuntimeError("repeatability evaluation returned no gradient")
        gradient = np.asarray(gradient, dtype=np.float64)
        components_numeric = numeric_components(components)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise RuntimeError(f"nonfinite repeatability evaluation at repeat {repeat}")
        repeats.append({
            "repeat": repeat,
            "elapsed_seconds": elapsed,
            "value": float(value),
            "gradient": gradient,
            "gradient_l2": float(np.linalg.norm(gradient)),
            "gradient_max_abs": float(np.max(np.abs(gradient))),
            "components": components_numeric,
        })
    objective.synchronize()

    repeat_pairs = []
    for right_index in (1, 2):
        left, right = repeats[0], repeats[right_index]
        component_error = component_errors(right["components"], left["components"])
        gradient_delta = right["gradient"] - left["gradient"]
        repeat_pairs.append({
            "left_repeat": 0,
            "right_repeat": right_index,
            "value_abs": abs(right["value"] - left["value"]),
            "gradient_max_abs": float(np.max(np.abs(gradient_delta))),
            "gradient_l2": float(np.linalg.norm(gradient_delta)),
            "component_errors": component_error,
            "component_max_abs": max_abs(component_error),
            "bitwise_value": bool(right["value"] == left["value"]),
            "bitwise_gradient": bool(np.array_equal(right["gradient"], left["gradient"])),
            "bitwise_components": bool(all(right["components"].get(key) == left["components"].get(key) for key in set(right["components"]) | set(left["components"]))),
        })
    repeatability_bitwise = all(
        pair["bitwise_value"] and pair["bitwise_gradient"] and pair["bitwise_components"]
        for pair in repeat_pairs
    )
    repeat_value_max_abs = max(pair["value_abs"] for pair in repeat_pairs)
    repeat_gradient_max_abs = max(pair["gradient_max_abs"] for pair in repeat_pairs)
    repeat_component_max_abs = max(pair["component_max_abs"] for pair in repeat_pairs)

    saved_components = numeric_components(stage["fit"]["final_components"])
    stored_component_errors = component_errors(repeats[0]["components"], saved_components)
    stored_consistency = {
        "stored_final_total": float(stage["fit"]["final_total"]),
        "evaluated_total": repeats[0]["value"],
        "total_abs": abs(repeats[0]["value"] - float(stage["fit"]["final_total"])),
        "stored_final_gradient_l2": float(stage["fit"]["final_gradient_l2"]),
        "evaluated_gradient_l2": repeats[0]["gradient_l2"],
        "gradient_l2_abs": abs(repeats[0]["gradient_l2"] - float(stage["fit"]["final_gradient_l2"])),
        "stored_final_gradient_max_abs": float(stage["fit"]["final_gradient_max_abs"]),
        "evaluated_gradient_max_abs": repeats[0]["gradient_max_abs"],
        "gradient_max_abs_abs": abs(repeats[0]["gradient_max_abs"] - float(stage["fit"]["final_gradient_max_abs"])),
        "stored_components": saved_components,
        "evaluated_components": repeats[0]["components"],
        "component_errors": stored_component_errors,
        "component_max_abs": max_abs(stored_component_errors),
    }
    thresholds = {
        "repeat_value_abs": 1.0e-12,
        "repeat_gradient_max_abs": 1.0e-12,
        "repeat_component_max_abs": 1.0e-10,
        "stored_total_abs": 1.0e-10,
        "stored_gradient_l2_abs": 1.0e-12,
        "stored_gradient_max_abs_abs": 1.0e-12,
        "stored_component_max_abs": 1.0e-10,
        "coordinate_max_abs": 0.0,
    }
    repeat_within_tolerance = (
        repeat_value_max_abs <= thresholds["repeat_value_abs"]
        and repeat_gradient_max_abs <= thresholds["repeat_gradient_max_abs"]
        and repeat_component_max_abs <= thresholds["repeat_component_max_abs"]
    )
    stored_within_tolerance = (
        stored_consistency["total_abs"] <= thresholds["stored_total_abs"]
        and stored_consistency["gradient_l2_abs"] <= thresholds["stored_gradient_l2_abs"]
        and stored_consistency["gradient_max_abs_abs"] <= thresholds["stored_gradient_max_abs_abs"]
        and stored_consistency["component_max_abs"] <= thresholds["stored_component_max_abs"]
    )
    metadata = objective.backend_metadata()
    build = json.loads(BUILD_LOG.read_text(encoding="utf-8"))
    config_hash = sha256(CONFIG)
    source_hash = metadata["source_sha256"]
    checks = {
        "config_hash": config_hash == EXPECTED_CONFIG,
        "source_hash": source_hash == EXPECTED_SOURCE,
        "build_source_hash": build["source_sha256"] == EXPECTED_SOURCE,
        "saved_theta_hash": theta_hash == state["theta_sha256"],
        "saved_coordinates_hash": saved_coordinates_hash == state["coordinates_sha256"],
        "coordinates_reconstructed_bitwise": coordinate_bitwise,
        "finite_full_grid_ball": finite_ball,
        "three_full_evaluations_finite": len(repeats) == 3,
        "repeatability_bitwise": repeatability_bitwise,
        "repeatability_within_tolerance": repeat_within_tolerance,
        "stored_final_consistency": stored_within_tolerance,
        "budget_exact": (
            data.n_loci == 131700
            and data.n_pairs == 8672379150
            and data.raw_records == 1703888
        ),
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "repeatability_mode": "bitwise_deterministic" if repeatability_bitwise else "tolerance_bounded_nonbitwise",
        "diagnostic": "three_same_theta_full_value_gradient_evaluations_on_saved_20kb_state",
        "started_elapsed_seconds": time.perf_counter() - started,
        "attempt": str(ATTEMPT),
        "input": {"path": str(INPUT), "sha256": sha256(INPUT)},
        "config_sha256": config_hash,
        "source_sha256": source_hash,
        "build_record": {
            "path": str(BUILD_LOG),
            "sha256": sha256(BUILD_LOG),
            "source_sha256": build["source_sha256"],
        },
        "saved_state": {
            "state_npz": str(STATE_NPZ),
            "state_npz_sha256": sha256(STATE_NPZ),
            "state_json": str(STATE_JSON),
            "state_json_sha256": sha256(STATE_JSON),
            "stage_json": str(STAGE_JSON),
            "stage_json_sha256": sha256(STAGE_JSON),
            "coordinates_file": str(COORDS_FILE),
            "coordinates_file_sha256": sha256(COORDS_FILE),
            "theta_sha256": theta_hash,
            "coordinates_array_sha256": saved_coordinates_hash,
            "theta_shape": list(theta.shape),
            "coordinates_shape": list(saved_coordinates.shape),
        },
        "grid_budget": data.budget(),
        "timing": {
            "load_20kb_seconds": load_seconds,
            "extension_load_seconds": extension_load_seconds,
            "upload_seconds_synchronized": upload_seconds,
            "coordinate_reconstruct_seconds": coordinate_seconds,
            "repeat_elapsed_seconds": [row["elapsed_seconds"] for row in repeats],
            "repeat_total_seconds": sum(row["elapsed_seconds"] for row in repeats),
            "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(objective.device)),
            "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(objective.device)),
        },
        "coordinates": {
            "saved_max_radius": float(np.linalg.norm(saved_coordinates, axis=-1).max()),
            "reconstructed_max_radius": float(radii.max()),
            "finite_ball": finite_ball,
            "reconstructed_array_sha256": reconstructed_coordinates_hash,
            "saved_array_sha256": saved_coordinates_hash,
            "bitwise": coordinate_bitwise,
            "max_abs": coordinate_max_abs,
            "p_reconstructed": float(reconstructed_p),
        },
        "repeats": [
            {key: value for key, value in row.items() if key != "gradient"}
            for row in repeats
        ],
        "repeat_pairs": repeat_pairs,
        "repeatability_max_abs": {
            "value": repeat_value_max_abs,
            "gradient": repeat_gradient_max_abs,
            "components": repeat_component_max_abs,
        },
        "stored_final_consistency": stored_consistency,
        "thresholds": thresholds,
        "checks": checks,
        "formal_optimization_started": True,
        "phase_or_reference_opened": False,
    }
    LOG.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
