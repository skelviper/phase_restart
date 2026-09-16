"""040正式技术visibility运行控制器。

本控制器只负责已授权的12 fits、24 stages和严格FG预算的编排与记录。
所有训练输入来自SNP-free counts、014 blind初始化或manifest授权的synthetic
worker inputs；不读取phase、reference、evaluation或038。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent
FORMAL_ROOT = SOURCE.parent
for path in (SOURCE, SOURCE / "frozen_035", SOURCE / "frozen_pr", SOURCE / "frozen_037"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m1_preconditioner import run_budgeted_lbfgs  # noqa: E402
from pr import contact_model, reconstruction_init  # noqa: E402
from visibility_profile import ProfileConvergenceError, VisibilityGPUObjective  # noqa: E402

INPUT_PATH = (ROOT / "inputs" / "P9016.snpfree.pairs.gz").resolve()
MANIFEST_PATH = (ROOT / "docs/audits/next-step-r2-preparation-20260914T143656Z/synthetic_inputs/known_e_worker_input_manifest.json").resolve()
PREFLIGHT_PATH = (ROOT / "test_res/039-20260914T164022Z-visibility-preflight/preflight.json").resolve()
PARITY_PATH = (ROOT / "test_res/039-20260914T164022Z-visibility-preflight/real_c0_parity_audit.json").resolve()
APPROVED_014_ROOT = (ROOT / "test_res/014-20260912_153000-s0-genome-wide-fixed").resolve()

REAL_STAGES = (
    {"stage": "5Mb", "bin_size_bp": 5_000_000, "fg_cap": 612},
    {"stage": "2Mb", "bin_size_bp": 2_000_000, "fg_cap": 404},
    {"stage": "1Mb", "bin_size_bp": 1_000_000, "fg_cap": 486},
)
SYNTHETIC_STAGE = {"stage": "1Mb", "bin_size_bp": 1_000_000, "fg_cap": 243}
REAL_MODE_SPECS = (
    ("V0", "V0-fixed-production-e", None),
    ("VZ", "VZ-zero-degree-fixed-e", None),
    ("V1", "V1-profile-e", None),
)
SYNTHETIC_MODE_SPECS = (
    ("V0-production-e", "V0-fixed-production-e", "production_e"),
    ("V1-profile-e", "V1-profile-e", None),
    ("V0-known-generating-e", "V0-known-generating-e", "known_e"),
)
REAL_CANDIDATES = ("consensus", "random")
FIXED_E_RESCORE_MODE = "V0-fixed-production-e"
PROFILE_RESIDUAL_TOL = 1e-10
CANONICAL_GTOL = 1e-6
FTOL = 1e-10
SOURCE_TIE_TOL = 1e-9

INNER_CAP = 80
PAIR_BLOCK = 262_144
EXPECTED_FITS = 12
EXPECTED_STAGES = 24
EXPECTED_OUTER_FG = 10_470
ACTIVE_EXPECTED_OUTER_FG = EXPECTED_OUTER_FG


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def append_event(event: dict[str, Any]) -> None:
    with (RUN / "logs" / "formal.jsonl").open("a") as handle:
        handle.write(json.dumps(jsonable(event), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def configure_014_paths() -> None:
    """把冻结014初始化的相对路径锚定到当前工作树，不改变其内容。"""
    reconstruction_init.ROOT = str(ROOT)
    gate = APPROVED_014_ROOT / "gate.json"
    coords = APPROVED_014_ROOT / "coords"
    reconstruction_init.DEFAULT_GATE_PATH = str(gate)
    reconstruction_init.DEFAULT_COORD_DIR = str(coords)
    for candidate, spec in list(reconstruction_init.APPROVED_SOURCES.items()):
        patched = dict(spec)
        patched["path"] = str(coords / (candidate + ".3dg"))
        patched["gate_path"] = str(gate)
        reconstruction_init.APPROVED_SOURCES[candidate] = patched


def source_hashes() -> dict[str, str]:
    result = {}
    for path in sorted(SOURCE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        result[str(path.relative_to(FORMAL_ROOT))] = sha256_file(path)
    return result


def stage_record_path(attempt_id: str, stage: str) -> Path:
    return RUN / "stages" / attempt_id / (stage + ".json")


def endpoint_paths(attempt_id: str, stage: str) -> tuple[Path, Path, Path]:
    coord = RUN / "coords" / attempt_id / (stage + ".3dg")
    sidecar = RUN / "sidecars" / attempt_id / (stage + ".visibility.npz")
    history = RUN / "stages" / attempt_id / (stage + ".accepted_history.json")
    coord.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    history.parent.mkdir(parents=True, exist_ok=True)
    return coord, sidecar, history


def real_data(bin_size_bp: int):
    data = contact_model.load_aggregate(INPUT_PATH, bin_size=bin_size_bp, verify_frozen_hash=True)
    data.assert_consistent()
    return data


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def manifest_path(entry: dict[str, Any]) -> Path:
    path = (ROOT / entry["path"]).resolve()
    expected = str(entry.get("sha256", ""))
    if expected and sha256_file(path) != expected:
        raise RuntimeError("manifest hash mismatch: %s" % path)
    return path


def load_synthetic_inputs(fixture: str, template: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    spec = manifest["fixtures"][fixture]
    production_path = manifest_path(spec["production_e_layer"])
    counts_path = manifest_path(spec["count_snapshot"])
    start_path = manifest_path(spec["shared_start"])
    known_path = manifest_path(spec["known_e_exposure_npy"])
    known_json_path = manifest_path(spec["known_e_exposure_json"])
    with np.load(production_path, allow_pickle=False) as prod:
        production_counts = np.asarray(prod["counts"], dtype=np.int64)
        production_diag = np.asarray(prod["diag_counts"], dtype=np.int64)
        production_exposure = np.asarray(prod["exposure"], dtype=np.float64)
        production_endpoint = np.asarray(prod["endpoint_counts"], dtype=np.int64)
        production_totals = {
            "diag": int(prod["group_diag"]),
            "cis_offdiag": int(prod["group_cis_offdiag"]),
            "inter": int(prod["group_inter"]),
        }
    with np.load(counts_path, allow_pickle=False) as counts_file:
        counts = np.asarray(counts_file["counts"], dtype=np.int64)
        diag_counts = np.asarray(counts_file["diag_counts"], dtype=np.int64)
        endpoint_counts = np.asarray(counts_file["endpoint_counts"], dtype=np.int64)
        totals = {
            "diag": int(counts_file["group_diag"]),
            "cis_offdiag": int(counts_file["group_cis_offdiag"]),
            "inter": int(counts_file["group_inter"]),
        }
    with np.load(start_path, allow_pickle=False) as start_file:
        coordinates = np.asarray(start_file["coordinates"], dtype=np.float64)
        p_init = float(start_file["p_init"])
    known_e = np.asarray(np.load(known_path, allow_pickle=False), dtype=np.float64)
    if not np.array_equal(counts, production_counts) or not np.array_equal(diag_counts, production_diag):
        raise RuntimeError("synthetic count snapshot disagrees with production layer")
    if not np.array_equal(endpoint_counts, production_endpoint) or totals != production_totals:
        raise RuntimeError("synthetic aggregate metadata disagrees with production layer")
    if coordinates.shape != (2, template.n_loci, 3) or not np.isfinite(coordinates).all():
        raise RuntimeError("synthetic shared start has wrong coordinates")
    if not (0.0 < p_init < 1.0):
        raise RuntimeError("synthetic shared start p_init outside (0,1)")
    data = contact_model.synthetic_integer_clone(
        template, counts, diag_counts, production_exposure,
        group_totals=totals, exposure_mode="synthetic_production_e",
        endpoint_counts=endpoint_counts,
    )
    return {
        "fixture": fixture,
        "data": data,
        "coordinates": coordinates,
        "p_init": p_init,
        "production_e": production_exposure,
        "known_e": known_e,
        "input_hashes": {
            "production_e_layer": sha256_file(production_path),
            "count_snapshot": sha256_file(counts_path),
            "shared_start": sha256_file(start_path),
            "known_e_exposure_npy": sha256_file(known_path),
            "known_e_exposure_json": sha256_file(known_json_path),
        },
    }


def read_full_tracks(path: Path, data: Any) -> np.ndarray:
    """独立解析已写3DG，确保sidecar来自写回后的完整full grid。"""
    coords = np.full((2, data.n_loci, 3), np.nan, dtype=np.float64)
    name_to_spec = {spec.name: spec for spec in data.track_specs}
    seen = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 5 or fields[0] not in name_to_spec:
                raise RuntimeError("invalid 3DG row at %s:%d" % (path, line_number))
            name, position_text, *xyz_text = fields
            spec = name_to_spec[name]
            position = int(position_text)
            local_bin = position // int(data.bin_size)
            slc = data.chromosome_slice(spec.chromosome_index)
            local = np.flatnonzero(data.locus_bin[slc] == local_bin)
            if len(local) != 1:
                raise RuntimeError("3DG row does not map to one full-grid bin")
            global_index = int(slc.start + local[0])
            key = (spec.copy_index, global_index)
            if key in seen:
                raise RuntimeError("duplicate 3DG row")
            seen.add(key)
            coords[spec.copy_index, global_index] = np.asarray(xyz_text, dtype=np.float64)
    if len(seen) != 2 * data.n_loci or not np.isfinite(coords).all():
        raise RuntimeError("3DG did not contain every full-grid copy/bin")
    return coords


def save_visibility_sidecar(path: Path, data: Any, objective: VisibilityGPUObjective) -> dict[str, Any]:
    visibility = objective.profile_arrays()
    e = np.asarray(visibility["e"], dtype=np.float64)
    active_mask = np.asarray(objective.support.active_mask, dtype=np.bool_)
    zero_mask = np.asarray(objective.support.zero_mask, dtype=np.bool_)
    eta = np.full(e.shape, np.nan, dtype=np.float64)
    active_log_e = np.log(e[active_mask])
    eta[active_mask] = active_log_e - float(active_log_e.mean())
    eta_placeholder = objective.mode == "V1-profile-e"
    if eta_placeholder:
        eta[~active_mask] = 0.0
    arrays = {
        "eta": eta,
        "e": e,
        "active_mask": active_mask,
        "zero_mask": zero_mask,
        "numeric_position": (np.asarray(data.locus_bin, dtype=np.int64) * int(data.bin_size)),
        "chromosome_index": np.asarray(data.locus_chromosome, dtype=np.int32),
        "eta_inactive_placeholder": np.asarray(eta_placeholder, dtype=np.bool_),
    }
    has_predicted_degree = objective.mode == "V1-profile-e"
    if has_predicted_degree:
        arrays["degree"] = np.asarray(visibility["degree"], dtype=np.float64)
        arrays["predicted_degree"] = np.asarray(visibility["predicted_degree"], dtype=np.float64)
    np.savez_compressed(path, **arrays)
    active_count = int(np.count_nonzero(active_mask))
    return {
        "e_vector_length": int(len(e)),
        "eta_active_count": active_count,
        "free_dimension": int(active_count - 1) if has_predicted_degree else None,
        "eta_semantics": "eta_active=log(e_active)-mean_active(log(e)); e was separately arithmetic-full-grid normalized",
        "eta_inactive_semantics": ("zero placeholder, not log(e=0)" if eta_placeholder
                                    else "not computed; stored as NaN"),
        "degree_status": "computed" if has_predicted_degree else "not_computed",
        "predicted_degree_status": "computed" if has_predicted_degree else "not_computed",
        "e_mean_full_grid": float(np.mean(e)),
        "e_zero_count": int(np.count_nonzero(zero_mask & (e == 0.0))),
        "active_count": active_count,
        "zero_count": int(np.count_nonzero(zero_mask)),
        "profile_relative_residual": (None if not has_predicted_degree
                                       else float(objective.diagnostics()["last_profile"]["relative_degree_residual"])),
    }


def objective_for(data: Any, backend_mode: str, known_e: np.ndarray | None) -> VisibilityGPUObjective:
    return VisibilityGPUObjective(
        data, backend_mode, device="cuda", pair_block=PAIR_BLOCK,
        inner_cap=INNER_CAP, cg_cap=INNER_CAP, profile_warm_start=True,
        known_e=known_e,
    )


def classify_fit(result: Any) -> str:
    if result.terminal_reason == "canonical_gtol":
        return "converged"
    if result.terminal_reason == "fg_budget_exhausted":
        return "budget_not_converged"
    if result.terminal_reason in ("ftol_numeric_stop", "scipy_stop"):
        return "not_converged"
    if result.canonical_gradient_max_abs <= CANONICAL_GTOL:
        return "converged"
    return "not_converged"


def select_candidate_by_count(rows: list[dict[str, Any]]) -> tuple[str | None, str]:
    """同一模型两source的count NLL选择；1e-9内近tie固定选consensus。"""
    valid = [row for row in rows if row.get("status") in ("converged", "not_converged", "budget_not_converged")
             and row.get("count_nll_normalized") is not None]
    if not valid:
        return None, "consensus wins when |count_nll_consensus-count_nll_random| <= 1e-9"
    best = min(valid, key=lambda row: float(row["count_nll_normalized"]))
    consensus = next((row for row in valid if row.get("candidate") == "consensus"), None)
    if consensus is not None and abs(float(consensus["count_nll_normalized"]) - float(best["count_nll_normalized"])) <= SOURCE_TIE_TOL:
        return "consensus", "consensus wins when |count_nll_consensus-count_nll_random| <= 1e-9"
    return str(best["candidate"]), "consensus wins when |count_nll_consensus-count_nll_random| <= 1e-9"


def stage_fit(
    attempt_id: str,
    stage_spec: dict[str, Any],
    data: Any,
    backend_mode: str,
    initial_coordinates: np.ndarray,
    p_init: float,
    q_init: float | None,
    known_e: np.ndarray | None,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    stage = str(stage_spec["stage"])
    fg_cap = int(stage_spec["fg_cap"])
    record_path = stage_record_path(attempt_id, stage)
    coord_path, sidecar_path, history_path = endpoint_paths(attempt_id, stage)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    base = {
        "attempt_id": attempt_id,
        "stage": stage,
        "bin_size_bp": int(stage_spec["bin_size_bp"]),
        "backend_mode": backend_mode,
        "input_provenance": provenance,
        "initial_p": float(p_init),
        "initial_q": None if q_init is None else float(q_init),
        "fg_cap": fg_cap,
        "maxiter": fg_cap + 1,
        "ftol": FTOL,
        "canonical_gtol": CANONICAL_GTOL,
        "inner_residual_tolerance": PROFILE_RESIDUAL_TOL,
        "status": "running",
        "started_at_utc": started_at,
    }
    write_json(record_path, base)
    append_event({"event": "stage_start", "attempt_id": attempt_id, "stage": stage,
                  "fg_cap": fg_cap, "at_utc": started_at})
    objective = None
    try:
        objective = objective_for(data, backend_mode, known_e)
        initial_coordinates_array = np.asarray(initial_coordinates, dtype=np.float64)
        initial_y = objective.raw_from_physical(initial_coordinates_array)
        initial_theta = objective.pack(initial_y, p=float(p_init))
        if q_init is not None:
            initial_theta[-1] = float(q_init)
        effective_initial_q = float(initial_theta[-1])
        base["initial_q"] = effective_initial_q
        base["initial_state_hashes"] = {
            "coordinates_sha256": hashlib.sha256(initial_coordinates_array.tobytes()).hexdigest(),
            "raw_y_sha256": hashlib.sha256(np.asarray(initial_y, dtype=np.float64).tobytes()).hexdigest(),
            "theta_sha256": hashlib.sha256(np.asarray(initial_theta, dtype=np.float64).tobytes()).hexdigest(),
            "q_float64_sha256": hashlib.sha256(np.asarray(effective_initial_q, dtype=np.float64).tobytes()).hexdigest(),
        }
        base["initial_positions_bp"] = {
            "unit": "bp", "shape": [int(data.n_loci)],
            "sha256": hashlib.sha256((np.asarray(data.locus_bin, dtype=np.int64) * int(data.bin_size)).tobytes()).hexdigest(),
        }
        write_json(record_path, base)
        accepted_history = []

        def accepted_callback(entry: dict[str, Any]) -> None:
            accepted_history.append(dict(entry))
            write_json(history_path, accepted_history)

        fit_started = time.perf_counter()
        result = run_budgeted_lbfgs(
            objective, initial_y, p_init=float(p_init), q_init=q_init, maxfun=fg_cap,
            maxiter=fg_cap + 1, maxls=20, ftol=FTOL,
            canonical_gtol=CANONICAL_GTOL, checkpoint_every=25,
            accepted_callback=accepted_callback,
        )
        fit_wall = time.perf_counter() - fit_started
        optimizer_diagnostics = objective.diagnostics()
        endpoint_coords = np.asarray(result.coordinates, dtype=np.float64).copy()
        # 先写full-grid 3DG，再独立读回，再做最终profile和rescore。
        contact_model.write_full_tracks(coord_path, data, endpoint_coords)
        readback_coords = read_full_tracks(coord_path, data)
        if not np.array_equal(endpoint_coords, readback_coords):
            raise RuntimeError("3DG coordinate write/readback changed endpoint coordinates")
        readback_started = time.perf_counter()
        readback_objective = objective_for(data, backend_mode, known_e)
        readback_raw = readback_objective.raw_from_physical(readback_coords)
        readback_theta = readback_objective.pack(readback_raw, p=float(result.p))
        readback_theta[-1] = float(result.theta[-1])
        readback_value, readback_gradient, readback_components = readback_objective.evaluate(
            readback_theta, need_gradient=True)
        readback_physical_gradient = readback_objective.physical_gradient()
        if readback_physical_gradient is None:
            raise RuntimeError("readback objective did not expose physical gradient")
        if abs(float(readback_components["p"]) - float(result.p)) > 1e-14:
            raise RuntimeError("readback p does not match endpoint q")
        readback_wall = time.perf_counter() - readback_started
        sidecar_summary = save_visibility_sidecar(sidecar_path, data, readback_objective)
        fixed_started = time.perf_counter()
        fixed_objective = objective_for(data, FIXED_E_RESCORE_MODE, None)
        fixed_theta = fixed_objective.pack(fixed_objective.raw_from_physical(readback_coords), p=float(result.p))
        fixed_theta[-1] = float(result.theta[-1])
        fixed_value, fixed_gradient, fixed_components = fixed_objective.evaluate(fixed_theta, need_gradient=True)
        fixed_objective.synchronize()
        fixed_wall = time.perf_counter() - fixed_started
        readback_value_abs = abs(float(readback_value) - float(result.fun))
        if readback_value_abs > 1e-8:
            raise RuntimeError("readback total differs from accepted endpoint beyond tolerance")
        readback_diagnostics = readback_objective.diagnostics()
        fixed_diagnostics = fixed_objective.diagnostics()
        profile = readback_diagnostics.get("last_profile") or {}
        stage_status = classify_fit(result)
        if backend_mode == "V1-profile-e" and (
                profile.get("relative_degree_residual") is None
                or float(profile["relative_degree_residual"]) > PROFILE_RESIDUAL_TOL):
            raise RuntimeError("readback profile residual exceeded hard tolerance")
        record = {
            **base,
            "status": stage_status,
            "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fit_wall_seconds": float(fit_wall),
            "optimizer": result.as_dict(),
            "terminal_classification": stage_status,
            "nfev_within_cap": bool(result.nfev <= fg_cap),
            "outer_fg_actual": int(result.nfev),
            "outer_fg_budget_accounted": int(result.nfev),
            "last_accepted_endpoint": bool(result.endpoint_was_last_accepted),
            "endpoint": {
                "p": float(result.p),
                "q": float(result.theta[-1]),
                "coordinates_shape": list(endpoint_coords.shape),
                "raw_y_q_gradient_inf": float(np.max(np.abs(readback_gradient))),
                "canonical_gradient_max_abs": float(result.canonical_gradient_max_abs),
                "canonical_gradient_norm": float(result.canonical_gradient_norm),
                "physical_gradient_inf": float(np.max(np.abs(readback_physical_gradient))),
                "total": float(readback_value),
                "count_nll_normalized": float(readback_components["count_nll_normalized"]),
                "profile": profile,
                "visibility": sidecar_summary,
            },
            "fixed_e_rescore": {
                "mode": FIXED_E_RESCORE_MODE,
                "total": float(fixed_value),
                "count_nll_normalized": float(fixed_components["count_nll_normalized"]),
                "gradient_inf": float(np.max(np.abs(fixed_gradient))),
                "p": float(fixed_components["p"]),
                "q": float(fixed_theta[-1]),
                "wall_seconds": float(fixed_wall),
            },
            "readback": {
                "coordinate_equal": True,
                "value_abs_vs_optimizer_endpoint": float(readback_value_abs),
                "component_total_abs_vs_optimizer_endpoint": abs(float(readback_components["total"]) - float(result.fun)),
                "wall_seconds": float(readback_wall),
                "sidecar_path": str(sidecar_path.relative_to(RUN)),
                "coordinate_path": str(coord_path.relative_to(RUN)),
                "accepted_history_path": str(history_path.relative_to(RUN)),
            },
            "optimizer_diagnostics": optimizer_diagnostics,
            "readback_diagnostics": readback_diagnostics,
            "fixed_e_rescore_diagnostics": fixed_diagnostics,
            "diagnostic_scopes": {
                "optimizer_diagnostics": "cumulative objective calls during run_budgeted_lbfgs, including initial value-only and non-FG endpoint/cache work",
                "readback_diagnostics": "one independent final write/readback value+gradient/profile call",
                "fixed_e_rescore_diagnostics": "one independent fixed-e auxiliary rescore call; excluded from optimizer FG budget",
            },
            "input_provenance": provenance,
        }
        if not history_path.exists():
            write_json(history_path, accepted_history)
        record["artifact_hashes"] = {
            "coordinates_sha256": sha256_file(coord_path),
            "visibility_sidecar_sha256": sha256_file(sidecar_path),
            "accepted_history_sha256": sha256_file(history_path),
        }
        write_json(record_path, record)
        append_event({"event": "stage_end", "attempt_id": attempt_id, "stage": stage,
                      "status": stage_status, "outer_fg_actual": int(result.nfev),
                      "at_utc": record["completed_at_utc"]})
        return record
    except ProfileConvergenceError as error:
        record = {**base, "status": "inner_error", "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "error_type": type(error).__name__, "error": str(error),
                  "hard_error": True, "inner_failure_is_not_convergence": True}
        write_json(record_path, record)
        append_event({"event": "stage_error", "attempt_id": attempt_id, "stage": stage,
                      "status": "inner_error", "error": str(error)})
        return record
    except Exception as error:
        record = {**base, "status": "error", "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "error_type": type(error).__name__, "error": str(error),
                  "hard_error": True}
        write_json(record_path, record)
        append_event({"event": "stage_error", "attempt_id": attempt_id, "stage": stage,
                      "status": "error", "error": str(error)})
        return record


def build_matrix(real_template: Any) -> list[dict[str, Any]]:
    matrix = []
    for model, backend, _ in REAL_MODE_SPECS:
        for candidate in REAL_CANDIDATES:
            matrix.append({"fit_id": "real-%s-%s" % (model, candidate), "kind": "real",
                           "model": model, "candidate": candidate, "backend_mode": backend,
                           "stage_specs": list(REAL_STAGES)})
    for fixture in ("P2", "N2"):
        for model, backend, exposure_key in SYNTHETIC_MODE_SPECS:
            matrix.append({"fit_id": "synthetic-%s-%s" % (fixture, model), "kind": "synthetic",
                           "fixture": fixture, "model": model, "backend_mode": backend,
                           "exposure_key": exposure_key, "stage_specs": [dict(SYNTHETIC_STAGE)]})
    return matrix


def prepare_manifest() -> dict[str, Any]:
    configure_014_paths()
    manifest = load_manifest()
    real_template = real_data(1_000_000)
    matrix = build_matrix(real_template)
    expected_stage_rows = [(fit["fit_id"], stage) for fit in matrix for stage in fit["stage_specs"]]
    if len(matrix) != EXPECTED_FITS or len(expected_stage_rows) != EXPECTED_STAGES:
        raise RuntimeError("formal matrix cardinality self-check failed")
    total_fg = sum(int(stage["fg_cap"]) for _, stage in expected_stage_rows)
    if total_fg != EXPECTED_OUTER_FG:
        raise RuntimeError("formal outer FG self-check failed: %d" % total_fg)
    input_hashes = {"snpfree": sha256_file(INPUT_PATH), "synthetic_manifest": sha256_file(MANIFEST_PATH),
                    "preflight": sha256_file(PREFLIGHT_PATH), "real_c0_parity_audit": sha256_file(PARITY_PATH)}
    synthetic_input_hashes = {}
    for fixture in ("P2", "N2"):
        spec = manifest["fixtures"][fixture]
        synthetic_input_hashes[fixture] = {
            key: sha256_file(manifest_path(spec[key]))
            for key in ("production_e_layer", "count_snapshot", "shared_start",
                        "known_e_exposure_npy", "known_e_exposure_json")
        }
    source = source_hashes()
    config = {
        "schema_version": "p9016-visibility-formal-v1",
        "run_id": RUN.name,
        "status": "prepared",
        "authorization": {"parent_authorized": True, "formal_started": False},
        "matrix": {
            "expected_fits": EXPECTED_FITS, "expected_stages": EXPECTED_STAGES,
            "expected_outer_fg": EXPECTED_OUTER_FG, "real_fit_count": 6,
            "synthetic_fit_count": 6, "stage_specs": list(REAL_STAGES) + [SYNTHETIC_STAGE],
            "fit_ids": [fit["fit_id"] for fit in matrix],
        },
        "optimizer": {
            "backend": "frozen_037.run_budgeted_lbfgs",
            "maxiter_rule": "fg_cap_plus_one", "ftol": FTOL,
            "canonical_gtol": CANONICAL_GTOL, "maxls": 20,
            "inner_cap": INNER_CAP, "profile_residual_tolerance": PROFILE_RESIDUAL_TOL,
            "dtype": "torch.float64", "mixed_precision": False, "pair_sampling": False,
            "accepted_endpoint_rule": "last callback-confirmed finite state",
            "ftol_is_not_gradient_convergence": True,
        },
        "data_boundary": {
            "snpfree_path": str(INPUT_PATH), "snpfree_sha256": input_hashes["snpfree"],
            "reference_opened": False, "phase_opened": False, "evaluation_opened": False,
            "038_opened": False, "synthetic_truth_coordinates_opened": False,
            "real_cis_total": 1_135_454, "real_inter": 568_434,
            "real_layer_breakdown": {
                "5Mb": {"cis_offdiag": 527902, "same_bin_diag": 607552},
                "2Mb": {"cis_offdiag": 619408, "same_bin_diag": 516046},
                "1Mb": {"cis_offdiag": 696680, "same_bin_diag": 438774},
            },
        },
        "hashes": {"inputs": input_hashes, "synthetic_inputs": synthetic_input_hashes,
                   "formal_source": source},
        "preflight_gate_paths": {
            "math": str(PREFLIGHT_PATH.relative_to(ROOT)),
            "real_c0_v0_parity": str(PARITY_PATH.relative_to(ROOT)),
        },
        "matrix_manifest": "matrix_manifest.json",
        "source_hash_manifest": "source_hashes.json",
        "endpoint_hash_manifest": "endpoint_hashes.json",
    }
    write_json(RUN / "matrix_manifest.json", {"matrix": matrix, "stage_rows": expected_stage_rows,
                                               "expected_fits": EXPECTED_FITS,
                                               "expected_stages": EXPECTED_STAGES,
                                               "expected_outer_fg": EXPECTED_OUTER_FG})
    write_json(RUN / "source_hashes.json", source)
    integration_summary = RUN / "integration_check" / "summary.json"
    if integration_summary.exists():
        config["integration_check"] = {
            "status": "passed", "path": "integration_check/summary.json",
            "sha256": sha256_file(integration_summary),
        }
    write_json(RUN / "config.json", config)
    return {"config": config, "matrix": matrix, "real_template": real_template,
            "manifest": manifest, "synthetic_input_hashes": synthetic_input_hashes}



def legacy_integration_check(prepared: dict[str, Any]) -> int:
    """用两条014 source做三层2-FG链，验证bp位置、q carry和写回契约。"""
    rows = []
    required = {"p", "q", "visibility", "coordinate_path", "sidecar_path", "artifact_hashes"}
    for candidate in REAL_CANDIDATES:
        previous = None
        p_init = 0.75
        q_init = contact_model.q_from_p(p_init)
        previous_q = None
        candidate_rows = []
        q_checks = []
        for stage_spec in REAL_STAGES:
            data = real_data(int(stage_spec["bin_size_bp"]))
            names = tuple(data.chromosome_names)
            lengths = tuple(int(length) for length in data.chromosome_lengths)
            if previous is None:
                state = reconstruction_init.initialize_approved_candidate(
                    candidate, names, lengths, int(stage_spec["bin_size_bp"]))
            else:
                state = reconstruction_init.warm_start_from_layer(
                    previous["coordinates"], previous["positions"], previous["chromosome_index"],
                    names, lengths, int(stage_spec["bin_size_bp"]),
                    1103 if candidate == "consensus" else 2207)
            expected_positions = np.broadcast_to(
                data.locus_bin * int(data.bin_size), (2, data.n_loci)).copy()
            positions = np.asarray(state["positions"], dtype=np.int64)
            if not np.array_equal(positions, expected_positions):
                raise RuntimeError("integration positions are not numeric bp")
            if previous_q is not None and q_init != previous_q:
                raise RuntimeError("integration q carry is not bit-exact")
            objective = objective_for(data, "V0-fixed-production-e", None)
            initial_y = objective.raw_from_physical(np.asarray(state["coords"], dtype=np.float64))
            accepted = []
            result = run_budgeted_lbfgs(
                objective, initial_y, p_init=float(p_init), q_init=float(q_init),
                maxfun=2, maxiter=3, maxls=20, ftol=FTOL,
                canonical_gtol=CANONICAL_GTOL,
                accepted_callback=lambda entry: accepted.append(dict(entry)),
            )
            if result.nfev != 2 or not result.endpoint_was_last_accepted:
                raise RuntimeError("integration 2-FG cap contract failed")
            q_out = float(result.theta[-1])
            p_from_q, _ = contact_model.p_from_q(q_out)
            if p_from_q != float(result.p):
                raise RuntimeError("integration p/q endpoint is not bit-exact")
            coord_path = RUN / "integration_check" / (candidate + "-" + stage_spec["stage"] + ".3dg")
            sidecar_path = RUN / "integration_check" / (candidate + "-" + stage_spec["stage"] + ".visibility.npz")
            record_path = RUN / "integration_check" / (candidate + "-" + stage_spec["stage"] + ".json")
            contact_model.write_full_tracks(coord_path, data, result.coordinates)
            readback = read_full_tracks(coord_path, data)
            if not np.array_equal(readback, result.coordinates):
                raise RuntimeError("integration 3DG readback changed coordinates")
            readback_objective = objective_for(data, "V0-fixed-production-e", None)
            raw = readback_objective.raw_from_physical(readback)
            theta = readback_objective.pack(raw, p=float(result.p))
            value, gradient, components = readback_objective.evaluate(theta, need_gradient=True)
            sidecar = save_visibility_sidecar(sidecar_path, data, readback_objective)
            record = {
                "attempt_id": "integration-%s" % candidate, "candidate": candidate,
                "stage": stage_spec["stage"], "fg_cap": 2, "outer_fg_actual": int(result.nfev),
                "initial_p": float(p_init), "initial_q": float(q_init),
                "p": float(result.p), "q": q_out,
                "p_from_q_equal": bool(p_from_q == float(result.p)),
                "positions_bp_equal": True, "coordinate_readback_equal": True,
                "visibility": sidecar, "coordinate_path": str(coord_path.relative_to(RUN)),
                "sidecar_path": str(sidecar_path.relative_to(RUN)),
                "record_fields_complete": True, "total": float(value),
                "gradient_inf": float(np.max(np.abs(gradient))),
                "count_nll_normalized": float(components["count_nll_normalized"]),
                "accepted_states": len(accepted),
                "artifact_hashes": {"coordinates_sha256": sha256_file(coord_path),
                                    "visibility_sidecar_sha256": sha256_file(sidecar_path)},
            }
            if not required.issubset(record):
                raise RuntimeError("integration record fields incomplete")
            write_json(record_path, record)
            candidate_rows.append(record)
            q_checks.append(previous_q is None or q_init == previous_q)
            previous = {
                "coordinates": readback, "positions": positions,
                "chromosome_index": np.asarray(state["chromosome_index"], dtype=np.int32),
            }
            p_init = float(result.p)
            previous_q = q_out
            q_init = q_out
        rows.append({"candidate": candidate, "stages": candidate_rows,
                     "q_carry_checks": q_checks, "all_q_carry_exact": all(q_checks)})
    result = {
        "schema": "visibility-formal-integration-check-v1", "status": "passed",
        "source_candidates": list(REAL_CANDIDATES),
        "layers": [stage["stage"] for stage in REAL_STAGES],
        "outer_fg_per_stage": 2, "positions_unit": "bp",
        "q_carry": "exact float value passed as q_init; p is not used to reconstruct q",
        "rows": rows, "matrix_manifest_sha256": sha256_file(RUN / "matrix_manifest.json"),
    }
    write_json(RUN / "integration_check" / "summary.json", result)
    config_path = RUN / "config.json"
    config = json.loads(config_path.read_text())
    config["integration_check"] = {
        "status": "passed", "path": "integration_check/summary.json",
        "sha256": sha256_file(RUN / "integration_check" / "summary.json"),
    }
    write_json(config_path, config)
    return 0


def integration_check(prepared: dict[str, Any]) -> int:
    """只跑接口层短链：real V0两source、V1一source、synthetic三分支各一案。"""
    required = {"p", "q", "visibility", "coordinate_path", "sidecar_path", "artifact_hashes"}
    rows: list[dict[str, Any]] = []
    initial_hashes: dict[str, str] = {}

    def run_case(case_id: str, data: Any, backend_mode: str, coordinates: np.ndarray,
                 p_init: float, q_init: float, stage_specs: list[dict[str, Any]],
                 candidate: str | None, provenance: dict[str, Any],
                  initial_positions: np.ndarray | None = None,
                  initial_chromosome_index: np.ndarray | None = None) -> dict[str, Any]:
        previous = None
        previous_q = None
        case_rows = []
        for stage_spec in stage_specs:
            bin_size = int(stage_spec["bin_size_bp"])
            if previous is None:
                stage_data = data
                state_metadata = {"mode": "initial_source"}
            else:
                stage_data = real_data(bin_size)
                names = tuple(stage_data.chromosome_names)
                lengths = tuple(int(length) for length in stage_data.chromosome_lengths)
                state = reconstruction_init.warm_start_from_layer(
                    previous["coordinates"], previous["positions"], previous["chromosome_index"],
                    names, lengths, bin_size, 1103 if candidate == "consensus" else 2207)
                state_metadata = state["metadata"]
                coordinates = np.asarray(state["coords"], dtype=np.float64).copy()
            expected_positions = (stage_data.locus_bin * int(stage_data.bin_size)).copy()
            if previous is None:
                positions = (np.asarray(initial_positions, dtype=np.int64).copy()
                             if initial_positions is not None else expected_positions.copy())
                chromosome_index = (np.asarray(initial_chromosome_index, dtype=np.int32).copy()
                                    if initial_chromosome_index is not None
                                    else np.asarray(stage_data.locus_chromosome, dtype=np.int32).copy())
            else:
                positions = np.asarray(state["positions"], dtype=np.int64)
                chromosome_index = np.asarray(state["chromosome_index"], dtype=np.int32)
            if positions.shape == (stage_data.n_loci,):
                positions_equal = np.array_equal(positions, expected_positions)
            elif positions.shape == (2, stage_data.n_loci):
                positions_equal = np.array_equal(positions, np.broadcast_to(expected_positions, positions.shape))
            else:
                positions_equal = False
            if not positions_equal:
                raise RuntimeError("integration positions are not exact numeric bp")
            if previous_q is not None and q_init != previous_q:
                raise RuntimeError("integration q carry is not bit-exact")
            initial_hash = hashlib.sha256(np.asarray(coordinates, dtype=np.float64).tobytes()).hexdigest()
            initial_key = case_id + "/" + str(stage_spec["stage"])
            initial_hashes[initial_key] = initial_hash
            objective = objective_for(stage_data, backend_mode, provenance.get("known_e"))
            initial_y = objective.raw_from_physical(coordinates)
            accepted: list[dict[str, Any]] = []
            result = run_budgeted_lbfgs(
                objective, initial_y, p_init=float(p_init), q_init=float(q_init),
                maxfun=2, maxiter=3, maxls=20, ftol=FTOL,
                canonical_gtol=CANONICAL_GTOL,
                accepted_callback=lambda entry: accepted.append(dict(entry)),
            )
            if result.nfev != 2 or not result.endpoint_was_last_accepted:
                raise RuntimeError("integration 2-FG cap contract failed")
            q_out = float(result.theta[-1])
            p_from_q, _ = contact_model.p_from_q(q_out)
            if p_from_q != float(result.p):
                raise RuntimeError("integration p/q endpoint is not exact")
            base = RUN / "integration_check" / case_id
            base.mkdir(parents=True, exist_ok=True)
            coord_path = base / (str(stage_spec["stage"]) + ".3dg")
            sidecar_path = base / (str(stage_spec["stage"]) + ".visibility.npz")
            record_path = base / (str(stage_spec["stage"]) + ".json")
            contact_model.write_full_tracks(coord_path, stage_data, result.coordinates)
            readback = read_full_tracks(coord_path, stage_data)
            if not np.array_equal(readback, result.coordinates):
                raise RuntimeError("integration 3DG readback changed coordinates")
            optimizer_diagnostics = objective.diagnostics()
            readback_objective = objective_for(stage_data, backend_mode, provenance.get("known_e"))
            raw = readback_objective.raw_from_physical(readback)
            theta = readback_objective.pack(raw, p=float(result.p))
            theta[-1] = q_out
            value, gradient, components = readback_objective.evaluate(theta, need_gradient=True)
            readback_physical_gradient = readback_objective.physical_gradient()
            if readback_physical_gradient is None:
                raise RuntimeError("integration readback physical gradient missing")
            readback_value_abs = abs(float(value) - float(result.fun))
            if readback_value_abs > 1e-8:
                raise RuntimeError("integration readback F mismatch")
            readback_diagnostics = readback_objective.diagnostics()
            visibility = save_visibility_sidecar(sidecar_path, stage_data, readback_objective)
            profile = readback_diagnostics.get("last_profile") or {}
            if backend_mode == "V1-profile-e" and float(profile["relative_degree_residual"]) > PROFILE_RESIDUAL_TOL:
                raise RuntimeError("integration V1 profile residual failed")
            fixed_objective = objective_for(stage_data, FIXED_E_RESCORE_MODE, None)
            fixed_theta = fixed_objective.pack(fixed_objective.raw_from_physical(readback), p=float(result.p))
            fixed_theta[-1] = q_out
            fixed_value, fixed_gradient, fixed_components = fixed_objective.evaluate(fixed_theta, need_gradient=True)
            fixed_diagnostics = fixed_objective.diagnostics()
            record = {
                "attempt_id": case_id, "stage": stage_spec["stage"],
                "backend_mode": backend_mode, "fg_cap": 2, "outer_fg_actual": int(result.nfev),
                "initial_p": float(p_init), "initial_q": float(q_init),
                "initial_coordinate_sha256": initial_hash,
                "initialization_metadata": state_metadata,
                "p": float(result.p), "q": q_out,
                "p_from_q_equal": bool(p_from_q == float(result.p)),
                "positions_bp_equal": True, "coordinate_readback_equal": True,
                "visibility": visibility, "profile": profile,
                "coordinate_path": str(coord_path.relative_to(RUN)),
                "sidecar_path": str(sidecar_path.relative_to(RUN)),
                "record_fields_complete": True, "total": float(value),
                "readback_value_abs": float(readback_value_abs),
                "raw_y_q_gradient_inf": float(np.max(np.abs(gradient))),
                "physical_gradient_inf": float(np.max(np.abs(readback_physical_gradient))),
                "canonical_gradient_max_abs": float(result.canonical_gradient_max_abs),
                "count_nll_normalized": float(components["count_nll_normalized"]),
                "fixed_e_rescore": {
                    "total": float(fixed_value), "count_nll_normalized": float(fixed_components["count_nll_normalized"]),
                    "p": float(fixed_components["p"]), "q": float(fixed_theta[-1]),
                    "gradient_inf": float(np.max(np.abs(fixed_gradient))),
                },
                "optimizer_diagnostics": optimizer_diagnostics,
                "readback_diagnostics": readback_diagnostics,
                "fixed_e_rescore_diagnostics": fixed_diagnostics,
                "accepted_states": len(accepted), "input_provenance": provenance,
                "artifact_hashes": {"coordinates_sha256": sha256_file(coord_path),
                                    "visibility_sidecar_sha256": sha256_file(sidecar_path)},
            }
            if not required.issubset(record):
                raise RuntimeError("integration record fields incomplete")
            write_json(record_path, record)
            case_rows.append(record)
            previous = {"coordinates": readback, "positions": positions,
                        "chromosome_index": chromosome_index}
            p_init = float(result.p)
            previous_q = q_out
            q_init = q_out
        return {"case_id": case_id, "backend_mode": backend_mode,
                "stages": case_rows, "input_provenance": provenance,
                "all_positions_bp": True,
                "all_q_carry_exact": all(row["p_from_q_equal"] for row in case_rows)}

    configure_014_paths()
    for candidate in REAL_CANDIDATES:
        first = real_data(5_000_000)
        names = tuple(first.chromosome_names)
        lengths = tuple(int(length) for length in first.chromosome_lengths)
        source_state = reconstruction_init.initialize_approved_candidate(candidate, names, lengths, 5_000_000)
        start_coords = np.asarray(source_state["coords"], dtype=np.float64)
        source_hash = str(source_state["metadata"]["source"]["source_sha256"])
        common = {"kind": "real", "candidate": candidate, "input_sha256": sha256_file(INPUT_PATH),
                  "initial_source_sha256": source_hash}
        rows.append(run_case("real-V0-" + candidate, first, "V0-fixed-production-e", start_coords.copy(),
                             0.75, contact_model.q_from_p(0.75), list(REAL_STAGES), candidate, common,
                             source_state["positions"], source_state["chromosome_index"]))
        initial_hashes["real-V0-" + candidate + "/5Mb"] = initial_hashes["real-V0-" + candidate + "/5Mb"]
        rows.append(run_case("real-V1-" + candidate, first, "V1-profile-e", start_coords.copy(),
                             0.75, contact_model.q_from_p(0.75), list(REAL_STAGES), candidate, common,
                             source_state["positions"], source_state["chromosome_index"]))
    manifest = prepared["manifest"]
    template = prepared["real_template"]
    synthetic_initials: dict[str, str] = {}
    for fixture in ("P2", "N2"):
        synth = load_synthetic_inputs(fixture, template, manifest)
        for model, backend, exposure_key in SYNTHETIC_MODE_SPECS:
            case_id = "synthetic-%s-%s" % (fixture, model)
            known = synth["known_e"] if exposure_key == "known_e" else None
            provenance = {"kind": "synthetic", "fixture": fixture,
                          "input_hashes": synth["input_hashes"], "known_e": known}
            row = run_case(case_id, synth["data"], backend, synth["coordinates"].copy(),
                           float(synth["p_init"]), contact_model.q_from_p(float(synth["p_init"])),
                           [dict(SYNTHETIC_STAGE)], None, provenance)
            rows.append(row)
            synthetic_initials[case_id] = row["stages"][0]["initial_coordinate_sha256"]
    for fixture in ("P2", "N2"):
        values = {case_id: digest for case_id, digest in synthetic_initials.items() if case_id.startswith("synthetic-%s-" % fixture)}
        if len(set(values.values())) != 1:
            raise RuntimeError("synthetic modes do not share the manifest start coordinates")
    for candidate in REAL_CANDIDATES:
        v0 = initial_hashes["real-V0-" + candidate + "/5Mb"]
        v1 = initial_hashes["real-V1-" + candidate + "/5Mb"]
        if v0 != v1:
            raise RuntimeError("real modes do not share the same 014 source start")
    tie_near = [{"candidate": "consensus", "status": "budget_not_converged", "count_nll_normalized": 1.0},
                {"candidate": "random", "status": "budget_not_converged", "count_nll_normalized": 1.0 + 5e-10}]
    tie_far = [{"candidate": "consensus", "status": "budget_not_converged", "count_nll_normalized": 1.0},
               {"candidate": "random", "status": "budget_not_converged", "count_nll_normalized": 1.0 + 2e-9}]
    near_selected, tie_rule = select_candidate_by_count(tie_near)
    far_selected, _ = select_candidate_by_count(tie_far)
    if near_selected != "consensus" or far_selected != "random":
        raise RuntimeError("source count-NLL tie rule integration check failed")
    tie_rule_check = {"near_difference": 5e-10, "far_difference": 2e-9,
                      "near_selected": near_selected, "far_selected": far_selected,
                      "tolerance": SOURCE_TIE_TOL, "rule": tie_rule}
    result = {
        "schema": "visibility-formal-integration-check-v2", "status": "passed",
        "real_cases": ["real-V0-consensus", "real-V0-random", "real-V1-consensus", "real-V1-random"],
        "synthetic_cases": [row["case_id"] for row in rows if row["case_id"].startswith("synthetic-")],
        "real_layers": [stage["stage"] for stage in REAL_STAGES],
        "synthetic_layers": [SYNTHETIC_STAGE["stage"]], "outer_fg_per_stage": 2,
        "positions_unit": "bp", "q_carry": "direct exact q_init from prior theta[-1]",
        "initial_mode_same_source": True, "synthetic_mode_same_start": True,
        "selection_tie_rule": tie_rule_check,
        "rows": rows, "matrix_manifest_sha256": sha256_file(RUN / "matrix_manifest.json"),
    }
    write_json(RUN / "integration_check" / "summary.json", result)
    config_path = RUN / "config.json"
    config = json.loads(config_path.read_text())
    config["integration_check"] = {"status": "passed", "path": "integration_check/summary.json",
                                    "sha256": sha256_file(RUN / "integration_check" / "summary.json")}
    write_json(config_path, config)
    return 0


def reusable_stage(attempt_id: str, stage_spec: dict[str, Any], data: Any) -> tuple[dict[str, Any], np.ndarray] | None:
    """复用本controller已完整写出且hash仍一致的accepted stage。"""
    path = stage_record_path(attempt_id, str(stage_spec["stage"]))
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("status") not in ("converged", "not_converged", "budget_not_converged"):
        return None
    hashes = record.get("artifact_hashes", {})
    readback = record.get("readback", {})
    coordinate_rel = readback.get("coordinate_path")
    sidecar_rel = readback.get("sidecar_path")
    if not coordinate_rel or not sidecar_rel:
        return None
    coordinate_path = RUN / coordinate_rel
    sidecar_path = RUN / sidecar_rel
    if not coordinate_path.exists() or not sidecar_path.exists():
        return None
    if sha256_file(coordinate_path) != hashes.get("coordinates_sha256"):
        return None
    if sha256_file(sidecar_path) != hashes.get("visibility_sidecar_sha256"):
        return None
    coordinates = read_full_tracks(coordinate_path, data)
    # 333的已完成首stage曾把rescore字段名写成q；修复记录语义而不改变数值/坐标。
    if "fixed_e_rescore" in record and "endpoint" in record:
        endpoint_p = record["endpoint"].get("p")
        if endpoint_p is not None and record["fixed_e_rescore"].get("p") != endpoint_p:
            record["fixed_e_rescore"]["p"] = float(endpoint_p)
            write_json(path, record)
    return record, coordinates


def run_formal(prepared: dict[str, Any]) -> int:
    matrix = prepared["matrix"]
    manifest = prepared["manifest"]
    real_template = prepared["real_template"]
    previous_config = json.loads((RUN / "config.json").read_text())
    previous_config["status"] = "running"
    previous_config["authorization"]["formal_started"] = True
    previous_config["started_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(RUN / "config.json", previous_config)
    append_event({"event": "formal_start", "run_id": RUN.name,
                  "expected_fits": EXPECTED_FITS, "expected_stages": EXPECTED_STAGES,
                  "expected_outer_fg": ACTIVE_EXPECTED_OUTER_FG,
                  "at_utc": previous_config["started_at_utc"]})
    fit_summaries = []
    actual_outer_fg = 0
    real_cache: dict[int, Any] = {1_000_000: real_template}
    for fit in matrix:
        fit_id = fit["fit_id"]
        candidate = fit.get("candidate")
        if fit["kind"] == "real":
            coordinates = None
            p_init = 0.75
            q_init = None
            provenance = {"kind": "real", "candidate": candidate,
                          "initialization": "014 blind consensus/random",
                          "input_sha256": sha256_file(INPUT_PATH)}
            known_e = None
            mode_data = None
        else:
            synth = load_synthetic_inputs(fit["fixture"], real_template, manifest)
            coordinates = synth["coordinates"].copy()
            p_init = float(synth["p_init"])
            q_init = contact_model.q_from_p(p_init)
            known_e = synth["known_e"] if fit["exposure_key"] == "known_e" else None
            mode_data = synth["data"]
            provenance = {"kind": "synthetic", "fixture": fit["fixture"],
                          "start_id": "manifest_shared_start", "input_hashes": synth["input_hashes"],
                          "production_exposure_mode": "synthetic_production_e"}
        fit_stage_records = []
        blocked = False
        previous_result = None
        for stage_spec in fit["stage_specs"]:
            bin_size = int(stage_spec["bin_size_bp"])
            if blocked:
                record = {"attempt_id": fit_id, "stage": stage_spec["stage"],
                          "bin_size_bp": bin_size, "status": "blocked_by_previous_stage",
                          "reason": "previous_stage_hard_error"}
                write_json(stage_record_path(fit_id, stage_spec["stage"]), record)
                fit_stage_records.append(record)
                continue
            if fit["kind"] == "real":
                data = real_cache.setdefault(bin_size, real_data(bin_size))
            else:
                data = mode_data
            reusable = reusable_stage(fit_id, stage_spec, data)
            if reusable is not None:
                record, endpoint_coords = reusable
                fit_stage_records.append(record)
                actual_outer_fg += int(record.get("outer_fg_actual", 0))
                previous_result = {
                    "coordinates": endpoint_coords,
                    "positions": (data.locus_bin * int(data.bin_size)).copy(),
                    "chromosome_index": np.asarray(data.locus_chromosome, dtype=np.int32).copy(),
                }
                coordinates = endpoint_coords
                p_init = float(record["endpoint"]["p"])
                q_init = float(record["endpoint"]["q"])
                append_event({"event": "stage_reused", "attempt_id": fit_id,
                              "stage": stage_spec["stage"], "outer_fg_actual": record.get("outer_fg_actual", 0)})
                continue
            if fit["kind"] == "real":
                if coordinates is None:
                    names = tuple(data.chromosome_names)
                    lengths = tuple(int(length) for length in data.chromosome_lengths)
                    initial = reconstruction_init.initialize_approved_candidate(candidate, names, lengths, bin_size)
                    coordinates = np.asarray(initial["coords"], dtype=np.float64).copy()
                    provenance["initial_source_hash"] = initial["metadata"]["source"]["source_sha256"]
                elif previous_result is not None:
                    names = tuple(data.chromosome_names)
                    lengths = tuple(int(length) for length in data.chromosome_lengths)
                    warm = reconstruction_init.warm_start_from_layer(
                        previous_result["coordinates"], previous_result["positions"],
                        previous_result["chromosome_index"], names, lengths, bin_size,
                        1103 if candidate == "consensus" else 2207)
                    coordinates = np.asarray(warm["coords"], dtype=np.float64).copy()
                    provenance["warm_start"] = warm["metadata"]
            record = stage_fit(
                fit_id, stage_spec, data, fit["backend_mode"], coordinates, p_init,
                q_init, known_e, provenance,
            )
            fit_stage_records.append(record)
            if record.get("status") in ("inner_error", "error"):
                blocked = True
                previous_result = None
            else:
                actual_outer_fg += int(record.get("outer_fg_actual", 0))
                # Reload the just-written endpoint rather than retaining an optimizer object.
                endpoint_coords = read_full_tracks(
                    RUN / record["readback"]["coordinate_path"], data)
                previous_result = {
                    "coordinates": endpoint_coords,
                    "positions": (data.locus_bin * int(data.bin_size)).copy(),
                    "chromosome_index": np.asarray(data.locus_chromosome, dtype=np.int32).copy(),
                }
                coordinates = endpoint_coords
                p_init = float(record["endpoint"]["p"])
                q_init = float(record["endpoint"]["q"])
        fit_status = "completed" if not blocked else "hard_error"
        fit_summary = {"fit_id": fit_id, "kind": fit["kind"], "model": fit["model"],
                       "candidate": candidate, "fixture": fit.get("fixture"),
                       "status": fit_status, "stage_status": [row.get("status") for row in fit_stage_records],
                       "outer_fg_actual": sum(int(row.get("outer_fg_actual", 0)) for row in fit_stage_records)}
        fit_summaries.append(fit_summary)
        write_json(RUN / "stages" / fit_id / "fit_summary.json", fit_summary)
        append_event({"event": "fit_end", **fit_summary})
    endpoint_rows = []
    for stage_path in sorted((RUN / "stages").glob("*/[125]Mb.json")):
        record = json.loads(stage_path.read_text())
        endpoint_rows.append({
            "attempt_id": record.get("attempt_id"), "stage": record.get("stage"),
            "status": record.get("status"), "outer_fg_actual": record.get("outer_fg_actual", 0),
            "coordinate_sha256": record.get("artifact_hashes", {}).get("coordinates_sha256"),
            "visibility_sidecar_sha256": record.get("artifact_hashes", {}).get("visibility_sidecar_sha256"),
            "stage_json_sha256": sha256_file(stage_path),
        })
    endpoint_manifest = {"schema": "p9016-visibility-formal-endpoints-v1",
                         "run_id": RUN.name, "expected_stages": EXPECTED_STAGES,
                         "actual_stages": len(endpoint_rows), "rows": endpoint_rows,
                         "all_stage_records_present": len(endpoint_rows) == EXPECTED_STAGES,
                         "all_endpoints_have_hashes": all(row["coordinate_sha256"] and row["visibility_sidecar_sha256"]
                                                          for row in endpoint_rows)}
    write_json(RUN / "endpoint_hashes.json", endpoint_manifest)
    selections = {}
    for model in ("V0", "VZ", "V1"):
        candidates = []
        for candidate in REAL_CANDIDATES:
            stage_record = json.loads(stage_record_path("real-%s-%s" % (model, candidate), "1Mb").read_text())
            candidates.append({"candidate": candidate, "status": stage_record.get("status"),
                               "count_nll_normalized": stage_record.get("endpoint", {}).get("count_nll_normalized"),
                               "coordinate_sha256": stage_record.get("artifact_hashes", {}).get("coordinates_sha256")})
        selected, tie_rule = select_candidate_by_count(candidates)
        selections[model] = {"candidates": candidates, "selected_by_own_count_nll": selected,
                             "tie_rule": tie_rule, "tie_tolerance": SOURCE_TIE_TOL}
    write_json(RUN / "selection.json", {"schema": "visibility-formal-selection-v1", "models": selections,
                                        "reference_used": False, "evaluation_used": False})
    final_status = "completed" if all(item["status"] == "completed" for item in fit_summaries) else "completed_with_hard_errors"
    final_config = json.loads((RUN / "config.json").read_text())
    final_config.update({"status": final_status, "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                         "actual_outer_fg": int(actual_outer_fg), "fit_summaries": fit_summaries,
                         "endpoint_hashes": endpoint_manifest, "endpoint_hashes_sha256": sha256_file(RUN / "endpoint_hashes.json")})
    write_json(RUN / "config.json", final_config)
    append_event({"event": "formal_end", "status": final_status,
                  "actual_outer_fg": actual_outer_fg, "at_utc": final_config["completed_at_utc"]})
    return 0 if final_status == "completed" else 2


def integration_formal_check() -> int:
    """在独立scratch root调用同一run_formal/stage_fit入口，所有24 stage各2 FG。"""
    global RUN, ACTIVE_EXPECTED_OUTER_FG
    original_run = RUN
    scratch = original_run / "integration_formal_check_v2"
    if scratch.exists():
        raise RuntimeError("integration scratch root already exists; refuse to mix runs")
    scratch.mkdir(parents=True)
    for name in ("logs", "coords", "plots", "stages", "sidecars", "rescores"):
        (scratch / name).mkdir()
    summary = None
    try:
        RUN = scratch
        ACTIVE_EXPECTED_OUTER_FG = 48
        prepared = prepare_manifest()
        for fit in prepared["matrix"]:
            for stage in fit["stage_specs"]:
                stage["fg_cap"] = 2
        write_json(RUN / "matrix_manifest.json", {
            "matrix": prepared["matrix"], "expected_fits": EXPECTED_FITS,
            "expected_stages": EXPECTED_STAGES, "expected_outer_fg": 48,
            "integration_budget_override": "all stage caps set to 2 only in scratch",
        })
        config = json.loads((RUN / "config.json").read_text())
        config["matrix"]["expected_outer_fg"] = 48
        config["integration_budget_override"] = {"all_stage_fg_cap": 2, "expected_outer_fg": 48}
        write_json(RUN / "config.json", config)
        run_status = run_formal(prepared)
        endpoint = json.loads((RUN / "endpoint_hashes.json").read_text())
        records = []
        previous_q: dict[str, float] = {}
        sidecar_checks = []
        for fit in prepared["matrix"]:
            fit_id = fit["fit_id"]
            for stage_spec in fit["stage_specs"]:
                stage_path = stage_record_path(fit_id, stage_spec["stage"])
                record = json.loads(stage_path.read_text())
                if record.get("status") not in ("budget_not_converged", "not_converged", "converged"):
                    raise RuntimeError("scratch stage did not produce a normal terminal status")
                if record.get("outer_fg_actual") != 2 or not record.get("initial_state_hashes"):
                    raise RuntimeError("scratch stage missing 2-FG or initial-state hash contract")
                if fit_id in previous_q and record.get("initial_q") != previous_q[fit_id]:
                    raise RuntimeError("scratch q carry is not exact")
                previous_q[fit_id] = float(record["endpoint"]["q"])
                sidecar_path = RUN / record["readback"]["sidecar_path"]
                with np.load(sidecar_path, allow_pickle=False) as sidecar:
                    expected_n = int(record["bin_size_bp"] and (2645 if record["bin_size_bp"] == 1_000_000 else
                                                                 1329 if record["bin_size_bp"] == 2_000_000 else 538))
                    positions = np.asarray(sidecar["numeric_position"], dtype=np.int64)
                    chromosome_index = np.asarray(sidecar["chromosome_index"], dtype=np.int32)
                    expected_positions = real_data(int(record["bin_size_bp"]))
                    if len(positions) != expected_n or not np.array_equal(
                            positions, expected_positions.locus_bin * int(expected_positions.bin_size)):
                        raise RuntimeError("scratch sidecar numeric positions failed")
                    if not np.array_equal(chromosome_index, expected_positions.locus_chromosome):
                        raise RuntimeError("scratch sidecar chromosome index failed")
                    is_v1 = record["backend_mode"] == "V1-profile-e"
                    if is_v1 and ("predicted_degree" not in sidecar.files or "degree" not in sidecar.files):
                        raise RuntimeError("scratch V1 sidecar omitted predicted degree")
                    if not is_v1 and ("predicted_degree" in sidecar.files or "degree" in sidecar.files):
                        raise RuntimeError("scratch fixed-e sidecar fabricated degree arrays")
                    sidecar_checks.append({"fit_id": fit_id, "stage": record["stage"],
                                           "degree_arrays": "computed" if is_v1 else "not_computed",
                                           "positions_bp": True})
                records.append({"fit_id": fit_id, "stage": record["stage"],
                                "status": record["status"], "outer_fg_actual": record["outer_fg_actual"],
                                "initial_state_hashes": record["initial_state_hashes"],
                                "endpoint_q": record["endpoint"]["q"]})
        selection = json.loads((RUN / "selection.json").read_text())
        for model, selected in selection["models"].items():
            if selected.get("tie_tolerance") != SOURCE_TIE_TOL:
                raise RuntimeError("scratch source selection tie tolerance mismatch")
        tie_near = [{"candidate": "consensus", "status": "budget_not_converged", "count_nll_normalized": 1.0},
                    {"candidate": "random", "status": "budget_not_converged", "count_nll_normalized": 1.0 + 5e-10}]
        tie_far = [{"candidate": "consensus", "status": "budget_not_converged", "count_nll_normalized": 1.0 + 2e-9},
                   {"candidate": "random", "status": "budget_not_converged", "count_nll_normalized": 1.0}]
        near_selected, tie_rule = select_candidate_by_count(tie_near)
        far_selected, _ = select_candidate_by_count(tie_far)
        if near_selected != "consensus" or far_selected != "random":
            raise RuntimeError("scratch source selection near-tie rule failed")
        tie_check = {"near_difference": 5e-10, "far_difference": 2e-9,
                      "near_selected": near_selected, "far_selected": far_selected,
                      "tolerance": SOURCE_TIE_TOL, "rule": tie_rule}
        if run_status != 0 or endpoint["actual_stages"] != EXPECTED_STAGES:
            raise RuntimeError("scratch run_formal did not complete 24 stages")
        summary = {
            "schema": "visibility-formal-integration-formal-entry-v1", "status": "passed",
            "scratch_root": str(scratch.relative_to(original_run)), "run_status": run_status,
            "expected_fits": EXPECTED_FITS, "expected_stages": EXPECTED_STAGES,
            "actual_stages": endpoint["actual_stages"], "outer_fg_per_stage": 2,
            "actual_outer_fg": int(sum(row["outer_fg_actual"] for row in records)),
            "positions_unit": "bp", "q_carry": "record initial_q equals previous endpoint q exactly",
            "sidecar_checks": sidecar_checks, "records": records,
            "selection": selection, "selection_tie_check": tie_check,
            "source_hashes": json.loads((RUN / "source_hashes.json").read_text()),
            "formal_cap_matrix_restored_after_check": True,
        }
    finally:
        RUN = original_run
        ACTIVE_EXPECTED_OUTER_FG = EXPECTED_OUTER_FG
    write_json(original_run / "integration_check" / "summary.json", summary)
    config_path = original_run / "config.json"
    config = json.loads(config_path.read_text())
    config["integration_check"] = {"status": "passed", "path": "integration_check/summary.json",
                                    "sha256": sha256_file(original_run / "integration_check" / "summary.json"),
                                    "scratch_root": "integration_formal_check_v2"}
    write_json(config_path, config)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-check", "integration-check", "run"))
    args = parser.parse_args()
    if args.command == "integration-check":
        return integration_formal_check()
    prepared = prepare_manifest()
    if args.command == "self-check":
        print(json.dumps({"run": str(RUN), "fits": len(prepared["matrix"]),
                          "stages": sum(len(fit["stage_specs"]) for fit in prepared["matrix"]),
                          "outer_fg": EXPECTED_OUTER_FG, "source_files": len(source_hashes())},
                         sort_keys=True))
        return 0
    return run_formal(prepared)


if __name__ == "__main__":
    raise SystemExit(main())
