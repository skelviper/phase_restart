"""One authorized 20 kb fused objective+gradient performance probe.

The 1 Mb accepted checkpoint is used only to construct a phase-free warm-start
coordinate array. This script never runs an optimizer and never builds a dense
eligible-pair array or a CPU 20 kb objective.
"""
from __future__ import annotations

import hashlib
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
LOG = RUN / "logs" / "fused-20k-probe.json"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ARCHIVED))

from fused.fused_objective import FusedObjective, load_fused_extension  # noqa: E402
from gpu_backend import initial_theta_from_coordinates, load_sparse_aggregate  # noqa: E402
from pr.reconstruction_init import warm_start_from_layer  # noqa: E402

INPUT = WORKSPACE / "inputs" / "P9016.snpfree.pairs.gz"
CHECKPOINT = WORKSPACE / "test_res" / "020-20260913_071841-v1-p9016-joint" / "checkpoints" / "random_joint" / "1m-accepted-0240.npz"
BUILD_LOG = RUN / "logs" / "fused-build.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def full_positions(lengths: np.ndarray, bin_size: int):
    positions = []
    chromosome = []
    offsets = [0]
    for index, length in enumerate(lengths):
        n = (int(length) + bin_size - 1) // bin_size
        positions.append(np.arange(n, dtype=np.int64) * bin_size)
        chromosome.append(np.full(n, index, dtype=np.int32))
        offsets.append(offsets[-1] + n)
    return (
        np.concatenate(positions),
        np.concatenate(chromosome),
        np.asarray(offsets, dtype=np.int64),
    )


def main():
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("20 kb probe requires exactly one visible CUDA device")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    if not INPUT.is_file() or not CHECKPOINT.is_file():
        raise FileNotFoundError("frozen probe input or checkpoint is missing")

    load_started = time.perf_counter()
    data = load_sparse_aggregate(INPUT, 20_000, verify_hash=True)
    load_seconds = time.perf_counter() - load_started
    if data.n_loci != 131_700 or data.n_pairs != 8_672_379_150:
        raise RuntimeError(f"20 kb grid mismatch: N={data.n_loci}, E={data.n_pairs}")
    if data.raw_records != 1_703_888:
        raise RuntimeError(f"raw record budget mismatch: {data.raw_records}")

    checkpoint = np.load(CHECKPOINT, allow_pickle=False)
    source_coords = np.asarray(checkpoint["coordinates"], dtype=np.float64)
    if source_coords.shape[0] != 2 or source_coords.shape[2] != 3:
        raise RuntimeError(f"checkpoint coordinate shape mismatch: {source_coords.shape}")
    source_positions, source_chromosome, _ = full_positions(data.chromosome_lengths, 1_000_000)
    if source_coords.shape[1] != len(source_positions):
        raise RuntimeError("checkpoint is not the expected 1 Mb full grid")
    names = tuple(data.chromosome_names)

    warm_started = time.perf_counter()
    warm = warm_start_from_layer(
        source_coords, source_positions, source_chromosome,
        names, tuple(int(v) for v in data.chromosome_lengths), 20_000,
        candidate_base_seed=2207,
    )
    warm_seconds = time.perf_counter() - warm_started
    coordinates = np.asarray(warm["coords"], dtype=np.float64)
    positions = np.asarray(warm["positions"], dtype=np.int64)
    chromosome = np.asarray(warm["chromosome_index"], dtype=np.int32)
    if coordinates.shape != (2, data.n_loci, 3):
        raise RuntimeError(f"warm-start coordinate shape mismatch: {coordinates.shape}")
    if not np.isfinite(coordinates).all() or not (np.linalg.norm(coordinates, axis=-1) < 1.0).all():
        raise RuntimeError("warm-start coordinates are not finite and strictly inside the ball")
    target_positions, target_chromosome, _ = full_positions(data.chromosome_lengths, 20_000)
    if not np.array_equal(positions, target_positions) or not np.array_equal(chromosome, target_chromosome):
        raise RuntimeError("warm-start grid does not match the 20 kb full grid")

    build_started = time.perf_counter()
    load_fused_extension(verbose=False)
    extension_load_seconds = time.perf_counter() - build_started
    upload_started = time.perf_counter()
    objective = FusedObjective(data, device="cuda", tile_rows=32, dtype=torch.float64)
    objective.synchronize()
    upload_seconds = time.perf_counter() - upload_started
    theta = initial_theta_from_coordinates(data, coordinates, p=0.75)
    if theta.shape != (6 * data.n_loci + 1,):
        raise RuntimeError(f"probe theta shape mismatch: {theta.shape}")

    torch.cuda.reset_peak_memory_stats(device)
    objective.synchronize()
    eval_started = time.perf_counter()
    value, gradient, components = objective.evaluate(theta, need_gradient=True)
    objective.synchronize()
    eval_seconds = time.perf_counter() - eval_started
    if gradient is None or not np.isfinite(value) or not np.isfinite(gradient).all():
        raise RuntimeError("20 kb probe produced a nonfinite objective or gradient")

    budget = data.budget()
    expected_beads = 2 * data.n_loci
    expected_tracks = 2 * len(data.chromosome_names)
    if expected_beads != 263_400 or expected_tracks != 40:
        raise RuntimeError("20 kb bead/track budget mismatch")
    if budget["endpoint_total"] != 2 * budget["raw_records"]:
        raise RuntimeError("20 kb endpoint conservation failed")
    if budget["raw_cis_offdiag"] + budget["raw_same_bin"] + budget["raw_inter"] != budget["raw_records"]:
        raise RuntimeError("20 kb raw cis/diag/inter conservation failed")
    if budget["n_zero_eligible_pairs"] != budget["n_eligible_pairs"] - budget["n_observed_nonzero_pairs"]:
        raise RuntimeError("20 kb zero eligible-pair accounting failed")
    build_payload = json.loads(BUILD_LOG.read_text(encoding="utf-8")) if BUILD_LOG.is_file() else None
    metadata = objective.backend_metadata()
    payload = {
        "status": "completed",
        "purpose": "single_authorized_20kb_full_fused_objective_gradient_probe",
        "formal_optimization_started": False,
        "phase_or_reference_opened": False,
        "input": {"path": str(INPUT), "sha256": sha256_file(INPUT)},
        "source_initialization": {
            "path": str(CHECKPOINT),
            "sha256": sha256_file(CHECKPOINT),
            "candidate": "random_joint",
            "source_stage": "1m-accepted-0240",
            "use": "phase_free_performance_initialization_only",
            "warm_start_method": "archived020.warm_start_from_layer",
            "warm_start_metadata": warm["metadata"],
        },
        "grid": {
            "bin_size_bp": 20_000,
            "n_loci": int(data.n_loci),
            "n_beads": expected_beads,
            "n_tracks": expected_tracks,
            "eligible_unordered_pairs": int(data.n_pairs),
            "pair_index_max_if_flat": int(data.n_pairs - 1),
            "pair_index_fits_int64": bool(data.n_pairs - 1 < 2**63),
            "dense_pair_array_materialized": False,
            "cpu_pair_indices_materialized": False,
            "indexing": "implicit ordered row j!=i; int64 audit arithmetic",
        },
        "budget": budget,
        "csr": {
            "nonzero_edges": int(len(objective._csr.out_j)),
            "out_ptr_dtype": str(objective._csr.out_ptr.dtype),
            "out_j_dtype": str(objective._csr.out_j.dtype),
            "out_count_dtype": str(objective._csr.out_count.dtype),
            "in_ptr_dtype": str(objective._csr.in_ptr.dtype),
            "in_i_dtype": str(objective._csr.in_i.dtype),
            "in_count_dtype": str(objective._csr.in_count.dtype),
            "out_terminal": int(objective._csr.out_ptr[-1]),
            "in_terminal": int(objective._csr.in_ptr[-1]),
        },
        "timing": {
            "load_20kb_seconds": load_seconds,
            "warm_start_seconds": warm_seconds,
            "extension_load_seconds": extension_load_seconds,
            "upload_seconds_synchronized": upload_seconds,
            "objective_gradient_seconds_synchronized": eval_seconds,
            "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "objective": {
            "value": float(value),
            "gradient_l2": float(np.linalg.norm(gradient)),
            "gradient_max_abs": float(np.max(np.abs(gradient))),
            "components": {str(k): float(v) for k, v in components.items() if isinstance(v, (float, int))},
        },
        "backend": metadata,
        "build_record": build_payload,
        "theta": {
            "shape": list(theta.shape),
            "dtype": str(theta.dtype),
            "p_init": 0.75,
            "finite": bool(np.isfinite(theta).all()),
        },
        "coordinates": {
            "shape": list(coordinates.shape),
            "max_radius": float(np.linalg.norm(coordinates, axis=-1).max()),
            "finite_ball": bool((np.linalg.norm(coordinates, axis=-1) < 1.0).all()),
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    LOG.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
