"""Independent 020 driver with isolated Torch and archived-CPU backends.

The driver never overwrites a completed backend run.  A fine-resolution request
first requires a completed, hashed two-candidate 1 Mb selection, then resumes
only the selected candidate from its saved 1 Mb theta/q and complete grid state.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Set native thread caps before importing NumPy/SciPy/Torch.
for _thread_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_key] = "1"

import numpy as np
import torch

RUN = Path(__file__).resolve().parents[1]
WORKSPACE = RUN.parents[1]
INPUT_DEFAULT = WORKSPACE / "inputs" / "P9016.snpfree.pairs.gz"
ARCHIVED_ROOT = RUN / "source" / "archived020"
sys.path.insert(0, str(ARCHIVED_ROOT))

from gpu_backend import (  # noqa: E402
    BackendError,
    TorchObjective,
    fit_l_bfgs,
    initial_theta_from_coordinates,
    load_sparse_aggregate,
    require_cuda,
    sha256_file,
    threadpool_audit,
)
from pr.contact_model import JointObjective as ArchivedObjective  # noqa: E402
from pr.contact_model import load_aggregate as load_archived_aggregate  # noqa: E402
from pr.contact_model import sphere_inverse as archived_sphere_inverse  # noqa: E402
from pr.reconstruction_init import warm_start_from_layer as archived_warm_start  # noqa: E402
import pr.joint_fit  # noqa: E402,F401

for _module_name in ("pr.contact_model", "pr.joint_fit", "pr.reconstruction_init"):
    _module = sys.modules.get(_module_name)
    if _module is None or not Path(_module.__file__).resolve().is_relative_to(ARCHIVED_ROOT):
        raise BackendError(f"archived source isolation failed for {_module_name}: {_module}")


CANDIDATES = (("consensus_joint", "consensus", 1103), ("random_joint", "random", 2207))
STAGES = (
    ("5m", 5_000_000, 300, 930),
    ("2m", 2_000_000, 200, 630),
    ("1m", 1_000_000, 240, 750),
    ("500k", 500_000, 240, 750),
    ("200k", 200_000, 240, 750),
    ("100k", 100_000, 240, 750),
    ("50k", 50_000, 240, 750),
    ("20k", 20_000, 240, 750),
)
ARCHIVED_CPU_BLOCK_SIZE = 65_536
FROZEN_INPUT_SHA256 = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"


# ---------- small serialization and identity helpers ----------


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


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


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def track_name(chromosome: int, copy: int) -> str:
    return f"c{chromosome + 1:02d}{'ab'[copy]}"


def backend_label(backend: str, device: str) -> str:
    normalized_device = device.replace(":", "-")
    if backend == "archived_cpu":
        return "archived-cpu"
    if backend == "fused_cuda":
        return "fused-cuda-" + normalized_device.removeprefix("cuda-") if normalized_device != "cuda" else "fused-cuda"
    return "torch-" + normalized_device


def assert_absent(path: Path, description: str) -> None:
    if path.exists():
        raise BackendError(f"refusing to overwrite existing {description}: {path}")


# ---------- exact grid and phase-free coordinate handling ----------


def full_positions(lengths: np.ndarray, bin_size: int):
    positions = []
    chromosomes = []
    offsets = [0]
    for chromosome, length in enumerate(lengths):
        count = (int(length) + bin_size - 1) // bin_size
        positions.append(np.arange(count, dtype=np.int64) * bin_size)
        chromosomes.append(np.full(count, chromosome, dtype=np.int32))
        offsets.append(offsets[-1] + count)
    return np.concatenate(positions), np.concatenate(chromosomes), np.asarray(offsets, dtype=np.int64)


def parse_3dg(path: Path, lengths: np.ndarray):
    tracks: dict[str, tuple[list[int], list[np.ndarray]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split()
            if len(fields) < 5:
                raise BackendError(f"malformed coordinate row {line_no}: {path}")
            try:
                track, position = fields[0], int(fields[1])
                xyz = np.asarray([float(v) for v in fields[2:5]], dtype=np.float64)
            except ValueError as exc:
                raise BackendError(f"invalid coordinate row {line_no}: {path}") from exc
            tracks.setdefault(track, ([], []))
            tracks[track][0].append(position)
            tracks[track][1].append(xyz)
    normalized = {}
    for track, (raw_positions, raw_coords) in tracks.items():
        positions = np.asarray(raw_positions, dtype=np.int64)
        coords = np.asarray(raw_coords, dtype=np.float64)
        order = np.argsort(positions, kind="stable")
        positions, coords = positions[order], coords[order]
        unique, starts, counts = np.unique(positions, return_index=True, return_counts=True)
        averaged = np.asarray(
            [coords[start:start + count].mean(axis=0) for start, count in zip(starts, counts)],
            dtype=np.float64,
        )
        normalized[track] = (unique, averaged)
    expected = {track_name(c, k) for c in range(len(lengths)) for k in (0, 1)}
    if set(normalized) != expected:
        raise BackendError(f"coordinate track inventory mismatch in {path}")
    for chromosome, length in enumerate(lengths):
        for copy in (0, 1):
            positions, coords = normalized[track_name(chromosome, copy)]
            if np.any(positions < 0) or np.any(positions >= int(length)):
                raise BackendError(f"coordinate outside chromosome header in {path}")
            if not np.all(np.isfinite(coords)):
                raise BackendError(f"nonfinite coordinate in {path}")
    return normalized


def expand_tracks(tracks, lengths: np.ndarray, bin_size: int):
    positions, chromosomes, offsets = full_positions(lengths, bin_size)
    coords = np.empty((2, len(positions), 3), dtype=np.float64)
    for chromosome, length in enumerate(lengths):
        start, stop = int(offsets[chromosome]), int(offsets[chromosome + 1])
        target = positions[start:stop]
        for copy in (0, 1):
            source_positions, source_coords = tracks[track_name(chromosome, copy)]
            for dimension in range(3):
                coords[copy, start:stop, dimension] = np.interp(
                    target, source_positions, source_coords[:, dimension])
    return coords, positions, chromosomes


def warm_start_exact(coords: np.ndarray, source_positions: np.ndarray, source_chromosomes: np.ndarray,
                     names: tuple[str, ...], lengths: np.ndarray, target_bin: int, base_seed: int):
    result = archived_warm_start(
        coords,
        source_positions,
        source_chromosomes,
        names,
        tuple(int(value) for value in lengths),
        target_bin,
        base_seed,
    )
    return (np.asarray(result["coords"], dtype=np.float64),
            np.asarray(result["positions"], dtype=np.int64),
            np.asarray(result["chromosome_index"], dtype=np.int32),
            dict(result["metadata"]))


def write_coordinates_with_offsets(path: Path, coords: np.ndarray, positions: np.ndarray,
                                   lengths: np.ndarray, bin_size: int) -> dict:
    expected_positions, expected_chromosomes, offsets = full_positions(lengths, bin_size)
    coords = np.asarray(coords, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    if coords.shape != (2, len(expected_positions), 3):
        raise BackendError(f"coordinate shape does not match grid: {coords.shape}")
    if not np.array_equal(positions, expected_positions):
        raise BackendError("export positions do not match complete header-order grid")
    if not np.all(np.isfinite(coords)) or np.any(np.linalg.norm(coords, axis=-1) >= 1.0):
        raise BackendError("export coordinates are nonfinite or outside the unit ball")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chromosome in range(len(lengths)):
            start, stop = int(offsets[chromosome]), int(offsets[chromosome + 1])
            for copy in (0, 1):
                for index in range(start, stop):
                    xyz = coords[copy, index]
                    handle.write(
                        f"{track_name(chromosome, copy)}\t{int(positions[index])}\t"
                        f"{xyz[0]:.17g}\t{xyz[1]:.17g}\t{xyz[2]:.17g}\n"
                    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "max_radius": float(np.linalg.norm(coords, axis=-1).max()),
        "n_tracks": 2 * len(lengths),
        "n_beads": int(coords.shape[1] * 2),
        "full_grid": True,
        "chromosome_order": list(range(len(lengths))),
        "positions_sha256": array_sha256(expected_positions),
        "chromosomes_sha256": array_sha256(expected_chromosomes),
    }


# ---------- backend adapters ----------


class ArchivedObjectiveAdapter:
    """Normalize the archived objective to the driver callback contract."""

    def __init__(self, objective):
        self._objective = objective

    def __getattr__(self, name):
        return getattr(self._objective, name)

    def components(self, theta):
        cached = getattr(self._objective, "cached_value_and_components", lambda value: None)(theta)
        if cached is not None:
            return cached[1]
        return self._objective.evaluate(theta, need_gradient=False)[2]


class BackendAdapter:
    def __init__(self, kind: str, device: str, tile_rows: int):
        if kind not in ("torch", "fused_cuda", "archived_cpu"):
            raise BackendError(f"unknown backend: {kind}")
        if kind == "archived_cpu" and device != "cpu":
            raise BackendError("archived_cpu backend only supports --device cpu")
        if kind == "fused_cuda" and not device.startswith("cuda"):
            raise BackendError("fused_cuda backend requires --device cuda or cuda:N")
        if tile_rows <= 0:
            raise BackendError("tile_rows must be positive")
        self.kind = kind
        self.device = device
        self.tile_rows = int(tile_rows)
        self.label = backend_label(kind, device)

    def load_data(self, input_path: Path, bin_size: int):
        if self.kind == "archived_cpu":
            return load_archived_aggregate(input_path, bin_size=bin_size, verify_frozen_hash=True)
        return load_sparse_aggregate(input_path, bin_size=bin_size, verify_hash=True)

    def objective(self, data):
        if self.kind == "archived_cpu":
            return ArchivedObjectiveAdapter(ArchivedObjective(
                data,
                block_size=ARCHIVED_CPU_BLOCK_SIZE,
                repulsion_block_size=ARCHIVED_CPU_BLOCK_SIZE,
            ))
        if self.kind == "fused_cuda":
            from fused import FusedObjective
            return FusedObjective(data, device=self.device, tile_rows=self.tile_rows,
                                  dtype=torch.float64)
        return TorchObjective(data, device=self.device, tile_rows=self.tile_rows)

    def pack_theta(self, objective, data, coordinates: np.ndarray, p: float) -> np.ndarray:
        if self.kind == "archived_cpu":
            return objective.pack(archived_sphere_inverse(coordinates), p=p)
        return initial_theta_from_coordinates(data, coordinates, p=p)

    def synchronize(self, objective) -> None:
        if hasattr(objective, "synchronize"):
            objective.synchronize()

    def setup_timing_name(self) -> str:
        return "upload_seconds" if self.kind in ("torch", "fused_cuda") else "archived_constructor_seconds"


# ---------- resumable state and selection ----------


def state_paths(backend_dir: Path, candidate_id: str, label: str):
    stem = backend_dir / "resume" / candidate_id / f"{label}-state"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def save_state(backend_dir: Path, candidate_id: str, label: str, bin_size: int,
               data, coordinates: np.ndarray, positions: np.ndarray,
               chromosomes: np.ndarray, theta: np.ndarray, config_hash: str,
               input_sha256: str) -> dict:
    npz_path, json_path = state_paths(backend_dir, candidate_id, label)
    assert_absent(npz_path, f"resume state {candidate_id}/{label}")
    assert_absent(json_path, f"resume state metadata {candidate_id}/{label}")
    payload = {
        "candidate_id": candidate_id,
        "stage": label,
        "bin_size_bp": int(bin_size),
        "theta_sha256": array_sha256(theta),
        "coordinates_sha256": array_sha256(coordinates),
        "positions_sha256": array_sha256(positions),
        "chromosomes_sha256": array_sha256(chromosomes),
        "grid_state_sha256": hashlib.sha256(
            (array_sha256(positions) + array_sha256(chromosomes)).encode("ascii")
        ).hexdigest(),
        "theta_shape": list(np.asarray(theta).shape),
        "coordinates_shape": list(np.asarray(coordinates).shape),
        "n_loci": int(data.n_loci),
        "chromosome_names": list(data.chromosome_names),
        "chromosome_lengths": [int(value) for value in data.chromosome_lengths],
        "input_sha256": input_sha256,
        "config_sha256": config_hash,
        "carried_q": float(theta[-1]),
        "phase_or_reference_opened": False,
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        theta=np.asarray(theta, dtype=np.float64),
        coordinates=np.asarray(coordinates, dtype=np.float64),
        positions=np.asarray(positions, dtype=np.int64),
        chromosomes=np.asarray(chromosomes, dtype=np.int32),
    )
    write_json(json_path, payload)
    return {"npz": str(npz_path.relative_to(backend_dir)),
            "json": str(json_path.relative_to(backend_dir)), **payload}


def load_state(backend_dir: Path, ref: dict, expected_candidate: str, expected_label: str,
               expected_bin: int, expected_data, config_hash: str, input_sha256: str) -> dict:
    npz_path = backend_dir / ref["npz"]
    json_path = backend_dir / ref["json"]
    if not npz_path.exists() or not json_path.exists():
        raise BackendError(f"missing saved state for {expected_candidate}/{expected_label}")
    with json_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    for key, value in (
        ("candidate_id", expected_candidate), ("stage", expected_label),
        ("bin_size_bp", int(expected_bin)), ("input_sha256", input_sha256),
        ("config_sha256", config_hash),
    ):
        if meta.get(key) != value:
            raise BackendError(f"resume metadata mismatch for {expected_candidate}/{expected_label}: {key}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        theta = np.asarray(arrays["theta"], dtype=np.float64)
        coordinates = np.asarray(arrays["coordinates"], dtype=np.float64)
        positions = np.asarray(arrays["positions"], dtype=np.int64)
        chromosomes = np.asarray(arrays["chromosomes"], dtype=np.int32)
    expected_positions, expected_chromosomes, _ = full_positions(expected_data.chromosome_lengths, expected_bin)
    if theta.shape != tuple(meta["theta_shape"]) or coordinates.shape != tuple(meta["coordinates_shape"]):
        raise BackendError("resume array shape metadata mismatch")
    if coordinates.shape != (2, expected_data.n_loci, 3) or theta.shape != (6 * expected_data.n_loci + 1,):
        raise BackendError("resume state does not match complete saved grid")
    if not np.array_equal(positions, expected_positions) or not np.array_equal(chromosomes, expected_chromosomes):
        raise BackendError("resume state grid does not match frozen header-order grid")
    if (meta["theta_sha256"] != array_sha256(theta)
            or meta["coordinates_sha256"] != array_sha256(coordinates)
            or meta["positions_sha256"] != array_sha256(positions)
            or meta["chromosomes_sha256"] != array_sha256(chromosomes)):
        raise BackendError("resume state array hash mismatch")
    if float(meta["carried_q"]) != float(theta[-1]):
        raise BackendError("resume carried q does not equal theta[-1]")
    return {"candidate_id": expected_candidate, "stage": expected_label,
            "bin_size_bp": int(expected_bin), "coordinates": coordinates,
            "positions": positions, "chromosomes": chromosomes,
            "theta": theta, "carried_q": float(theta[-1]),
            "state_ref": ref, "meta": meta}


def choose_selection(results: list[dict], config_hash: str, input_sha256: str) -> dict:
    if len(results) != 2:
        raise BackendError("selection requires both frozen candidates")
    order = {candidate_id: index for index, (candidate_id, _, _) in enumerate(CANDIDATES)}
    selected = min(results, key=lambda row: (row["final_count_nll_normalized"], order[row["candidate_id"]]))
    return {
        "status": "selected_after_both_candidates_1m",
        "criterion": "count_nll_normalized",
        "candidate_ids": [row["candidate_id"] for row in results],
        "values": {row["candidate_id"]: row["final_count_nll_normalized"] for row in results},
        "selected_candidate": selected["candidate_id"],
        "selected_value": selected["final_count_nll_normalized"],
        "selected_state": selected["state_ref"],
        "config_sha256": config_hash,
        "input_sha256": input_sha256,
        "phase_or_reference_opened": False,
        "recorded_at_utc": utc_now(),
    }


def read_stage(backend_dir: Path, candidate_id: str, label: str) -> dict:
    path = backend_dir / "stages" / candidate_id / f"{label}.json"
    if not path.exists():
        raise BackendError(f"missing completed stage {candidate_id}/{label}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# ---------- one candidate and the two-phase schedule ----------


def run_candidate(backend_dir: Path, input_path: Path, input_sha256: str, config_hash: str,
                  adapter: BackendAdapter, candidate_id: str, initialization: str,
                  base_seed: int, start_stage: int, stop_stage: int,
                  initial_state: dict | None = None):
    lengths = None if initial_state is None else np.asarray(initial_state["meta"]["chromosome_lengths"], dtype=np.int64)
    names = None if initial_state is None else tuple(initial_state["meta"]["chromosome_names"])
    state = initial_state
    rows = []
    for stage_index in range(start_stage, stop_stage + 1):
        label, bin_size, maxiter, maxfun = STAGES[stage_index]
        stage_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        data = adapter.load_data(input_path, bin_size)
        preprocess_seconds = time.perf_counter() - preprocess_started
        if lengths is None:
            lengths = data.chromosome_lengths.copy()
            names = tuple(data.chromosome_names)
        elif tuple(data.chromosome_names) != names or not np.array_equal(data.chromosome_lengths, lengths):
            raise BackendError("chromosome header changed between stages")

        init_started = time.perf_counter()
        if state is None:
            initial_path = RUN / "coords" / "initial_sources" / f"{initialization}_joint-initial-5m.3dg"
            initial_tracks = parse_3dg(initial_path, data.chromosome_lengths)
            coordinates, positions, chromosomes = expand_tracks(initial_tracks, data.chromosome_lengths, bin_size)
            init_meta = {"mode": "archived_020_phase_free_initial_5m", "path": str(initial_path),
                         "sha256": sha256_file(initial_path), "base_seed": int(base_seed)}
            carried_q = None
        else:
            expected_previous_label, expected_previous_bin, *_ = STAGES[stage_index - 1]
            if state["stage"] != expected_previous_label or int(state["bin_size_bp"]) != expected_previous_bin:
                raise BackendError(
                    f"invalid immediate predecessor for {candidate_id}/{label}: "
                    f"state={state['stage']}/{state['bin_size_bp']} expected="
                    f"{expected_previous_label}/{expected_previous_bin}")
            coordinates, positions, chromosomes, init_meta = warm_start_exact(
                state["coordinates"], state["positions"], state["chromosomes"],
                names, lengths, bin_size, base_seed,
            )
            init_meta["mode"] = "archived_020_exact_multiresolution_warm_start"
            init_meta["source_stage"] = state["stage"]
            carried_q = float(state["carried_q"])
        init_seconds = time.perf_counter() - init_started
        if coordinates.shape != (2, data.n_loci, 3) or not np.all(np.isfinite(coordinates)):
            raise BackendError(f"invalid initialization for {candidate_id}/{label}")
        expected_positions, expected_chromosomes, _ = full_positions(data.chromosome_lengths, bin_size)
        if not np.array_equal(positions, expected_positions) or not np.array_equal(chromosomes, expected_chromosomes):
            raise BackendError(f"initialization grid mismatch for {candidate_id}/{label}")

        setup_started = time.perf_counter()
        if adapter.device.startswith("cuda"):
            torch.cuda.synchronize(adapter.device)
        objective = adapter.objective(data)
        adapter.synchronize(objective)
        setup_seconds = time.perf_counter() - setup_started
        p_init = 0.75
        theta0 = adapter.pack_theta(objective, data, coordinates, p=p_init)
        if carried_q is not None:
            theta0[-1] = carried_q

        warmup_started = time.perf_counter()
        objective.evaluate(theta0, need_gradient=True)
        adapter.synchronize(objective)
        warmup_seconds = time.perf_counter() - warmup_started

        checkpoint_dir = backend_dir / "checkpoints" / candidate_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if any(checkpoint_dir.glob(f"{label}-accepted-*.npz")):
            raise BackendError(f"refusing to reuse checkpoint directory for {candidate_id}/{label}")
        accepted = []

        def checkpoint_hook(payload):
            checkpoint_path = checkpoint_dir / f"{label}-accepted-{int(payload['iteration']):04d}.npz"
            assert_absent(checkpoint_path, f"checkpoint {candidate_id}/{label}")
            np.savez_compressed(
                checkpoint_path,
                coordinates=payload["coordinates"], theta=payload["theta"],
                iteration=payload["iteration"], nfev=payload["nfev"], p=payload["p"],
                fullhistory_json=np.asarray(json.dumps(json_ready(payload["history"]), sort_keys=True)),
            )
            accepted.append(str(checkpoint_path.relative_to(backend_dir)))

        fit_started = time.perf_counter()
        fit = fit_l_bfgs(
            objective, theta0, maxiter=maxiter, maxfun=maxfun, maxls=20,
            ftol=1e-10, gtol=1e-6, checkpoint_every=10, checkpoint_hook=checkpoint_hook,
        )
        adapter.synchronize(objective)
        fit_seconds = time.perf_counter() - fit_started

        stage_path = backend_dir / "stages" / candidate_id / f"{label}.json"
        assert_absent(stage_path, f"stage summary {candidate_id}/{label}")
        final_path = backend_dir / "coords" / candidate_id / f"final-{label}-{config_hash[:12]}.3dg"
        assert_absent(final_path, f"coordinate export {candidate_id}/{label}")
        export_started = time.perf_counter()
        final_info = write_coordinates_with_offsets(
            final_path, fit["coordinates"], positions, data.chromosome_lengths, bin_size)
        export_seconds = time.perf_counter() - export_started
        state_ref = save_state(
            backend_dir, candidate_id, label, bin_size, data,
            fit["coordinates"], positions, chromosomes, fit["theta"], config_hash, input_sha256,
        )
        stage_row = {
            "candidate_id": candidate_id,
            "stage": label,
            "bin_size_bp": bin_size,
            "backend": adapter.label,
            "data_budget": data.budget(),
            "initialization": init_meta,
            "fit": {key: value for key, value in fit.items() if key not in ("theta", "coordinates", "history")},
            "accepted_history": fit["history"],
            "checkpoint_paths": accepted,
            "final_coordinates": final_info,
            "resume_state": state_ref,
            "timing": {
                "preprocess_seconds": preprocess_seconds,
                "initialization_seconds": init_seconds,
                "backend_constructor_seconds": setup_seconds,
                adapter.setup_timing_name(): setup_seconds if adapter.kind in ("torch", "fused_cuda") else 0.0,
                "warmup_seconds": warmup_seconds,
                "optimizer_only_seconds": fit_seconds,
                "export_seconds": export_seconds,
                "scoped_end_to_end_seconds": time.perf_counter() - stage_started,
                "objective_evaluations": fit["nfev"],
                "accepted_iterations": fit["nit"],
                "thread_pools": threadpool_audit(),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())
                if adapter.device.startswith("cuda") else None,
            },
            "status": fit["termination_class"],
            "budget_not_converged": fit["termination_class"] == "budget_not_converged",
            "line_search_abort": fit["termination_class"] == "line_search_abort",
        }
        write_json(stage_path, stage_row)
        rows.append(stage_row)
        state = {
            "candidate_id": candidate_id, "stage": label, "bin_size_bp": bin_size,
            "coordinates": np.asarray(fit["coordinates"], dtype=np.float64),
            "positions": np.asarray(positions, dtype=np.int64),
            "chromosomes": np.asarray(chromosomes, dtype=np.int32),
            "theta": np.asarray(fit["theta"], dtype=np.float64),
            "carried_q": float(fit["theta"][-1]), "state_ref": state_ref,
            "meta": state_ref,
        }
    final = rows[-1]
    return {
        "candidate_id": candidate_id,
        "stages": rows,
        "final_count_nll_normalized": final["fit"]["final_components"]["count_nll_normalized"],
        "final_coordinates": final["final_coordinates"],
        "carried_q": final["resume_state"]["carried_q"],
        "state_ref": final["resume_state"],
    }


def coarse_result_from_disk(backend_dir: Path, candidate_id: str) -> dict:
    rows = [read_stage(backend_dir, candidate_id, label) for label, *_ in STAGES[:3]]
    final = rows[-1]
    return {
        "candidate_id": candidate_id,
        "stages": rows,
        "final_count_nll_normalized": final["fit"]["final_components"]["count_nll_normalized"],
        "final_coordinates": final["final_coordinates"],
        "carried_q": final["resume_state"]["carried_q"],
        "state_ref": final["resume_state"],
    }


def load_selection(backend_dir: Path, config_hash: str, input_sha256: str) -> dict:
    path = backend_dir / "selection-1m.json"
    if not path.exists():
        raise BackendError(f"missing frozen 1 Mb selection: {path}")
    with path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("status") != "selected_after_both_candidates_1m":
        raise BackendError("invalid selection status")
    if selection.get("config_sha256") != config_hash or selection.get("input_sha256") != input_sha256:
        raise BackendError("selection provenance hash mismatch")
    if selection.get("selected_candidate") not in {candidate[0] for candidate in CANDIDATES}:
        raise BackendError("selection candidate is not frozen")
    return selection


def create_or_load_selection(backend_dir: Path, results: list[dict], config_hash: str,
                             input_sha256: str) -> dict:
    path = backend_dir / "selection-1m.json"
    if path.exists():
        return load_selection(backend_dir, config_hash, input_sha256)
    selection = choose_selection(results, config_hash, input_sha256)
    assert_absent(path, "1 Mb selection")
    write_json(path, selection)
    return selection


def highest_completed_fine(backend_dir: Path, candidate_id: str, target_index: int,
                           data_loader, input_path: Path, config_hash: str, input_sha256: str):
    current_state = None
    current_index = 2
    # The selection's 1 Mb state is the only valid starting point.
    one_m_data = data_loader(1_000_000)
    one_m_stage = read_stage(backend_dir, candidate_id, "1m")
    current_state = load_state(
        backend_dir, one_m_stage["resume_state"], candidate_id, "1m", 1_000_000,
        one_m_data, config_hash, input_sha256,
    )
    for index in range(3, target_index + 1):
        label, bin_size, *_ = STAGES[index]
        stage_path = backend_dir / "stages" / candidate_id / f"{label}.json"
        if not stage_path.exists():
            break
        stage = read_stage(backend_dir, candidate_id, label)
        data = data_loader(bin_size)
        current_state = load_state(
            backend_dir, stage["resume_state"], candidate_id, label, bin_size,
            data, config_hash, input_sha256,
        )
        current_index = index
    return current_index, current_state


# ---------- command entry point ----------


def execute(args) -> dict:
    if sha256_file(args.input) != FROZEN_INPUT_SHA256:
        raise BackendError("input SHA256 does not match frozen P9016 SNP-free cohort")
    if args.backend == "archived_cpu" and args.device != "cpu":
        raise BackendError("archived_cpu requires --device cpu")
    if args.backend == "fused_cuda" and not args.device.startswith("cuda"):
        raise BackendError("fused_cuda requires --device cuda or cuda:N")
    if args.backend == "torch" and args.device == "cpu" and not args.allow_cpu:
        raise BackendError("refusing independent Torch CPU full pipeline without --allow-cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        blocked = {
            "status": "gpu_unavailable",
            "backend": args.backend,
            "device": args.device,
            "probe": require_cuda(),
            "terminal_benchmark_run": False,
            "reason": "CUDA device visibility is required before GPU timing or pipeline",
        }
        backend_dir = args.run_dir / "backend_runs" / backend_label(args.backend, args.device)
        write_json(backend_dir / "pipeline-blocked.json", blocked)
        return blocked

    adapter = BackendAdapter(args.backend, args.device, args.tile_rows)
    backend_dir = args.run_dir / "backend_runs" / adapter.label
    backend_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_file(args.run_dir / "config.json")
    input_sha256 = sha256_file(args.input)
    target_index = next(index for index, row in enumerate(STAGES) if row[0] == args.through)
    results = []

    if target_index <= 2:
        for candidate_id, initialization, base_seed in CANDIDATES:
            results.append(run_candidate(
                backend_dir, args.input, input_sha256, config_hash, adapter,
                candidate_id, initialization, base_seed, 0, target_index,
            ))
        selection = create_or_load_selection(backend_dir, results, config_hash, input_sha256) if target_index == 2 else None
    else:
        coarse_ready = all(
            (backend_dir / "stages" / candidate_id / "1m.json").exists()
            for candidate_id, _, _ in CANDIDATES
        )
        if coarse_ready:
            results = [coarse_result_from_disk(backend_dir, candidate_id) for candidate_id, _, _ in CANDIDATES]
        else:
            if any((backend_dir / "stages" / candidate_id / "1m.json").exists()
                   for candidate_id, _, _ in CANDIDATES):
                raise BackendError("partial coarse run present; refusing to rerun or mix candidates")
            for candidate_id, initialization, base_seed in CANDIDATES:
                results.append(run_candidate(
                    backend_dir, args.input, input_sha256, config_hash, adapter,
                    candidate_id, initialization, base_seed, 0, 2,
                ))
        selection = create_or_load_selection(backend_dir, results, config_hash, input_sha256)
        selected_id = selection["selected_candidate"]
        selected_spec = next(row for row in CANDIDATES if row[0] == selected_id)
        current_index, current_state = highest_completed_fine(
            backend_dir, selected_id, target_index,
            lambda bin_size: adapter.load_data(args.input, bin_size),
            args.input, config_hash, input_sha256,
        )
        if current_index >= target_index:
            raise BackendError(f"target stage {STAGES[target_index][0]} already exists; refusing overwrite")
        continuation = run_candidate(
            backend_dir, args.input, input_sha256, config_hash, adapter,
            selected_spec[0], selected_spec[1], selected_spec[2],
            current_index + 1, target_index, initial_state=current_state,
        )
        results.append(continuation)

    payload = {
        "status": "training_complete" if target_index >= 2 else "partial_through_" + STAGES[target_index][0],
        "backend": args.backend,
        "backend_label": adapter.label,
        "device": args.device,
        "through": STAGES[target_index][0],
        "backend_directory": str(backend_dir),
        "input_sha256": input_sha256,
        "config_sha256": config_hash,
        "thread_pools": threadpool_audit(),
        "candidates": results,
        "selection": selection,
        "continuation_policy": "both_candidates_through_1m_then_selected_only_fine",
        "recorded_at_utc": utc_now(),
    }
    write_json(backend_dir / "pipeline-summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--backend", choices=("torch", "fused_cuda", "archived_cpu"), default="torch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-rows", type=int, default=32)
    parser.add_argument("--through", choices=[row[0] for row in STAGES], default="1m")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
        limiter.__enter__()
    except Exception:
        limiter = None
    try:
        payload = execute(args)
    except (BackendError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        raise SystemExit(1)
    finally:
        if limiter is not None:
            limiter.__exit__(None, None, None)
    print(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    if payload.get("status") == "gpu_unavailable":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
