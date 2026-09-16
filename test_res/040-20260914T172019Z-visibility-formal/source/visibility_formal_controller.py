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
INNER_CAP = 80
PAIR_BLOCK = 262_144
EXPECTED_FITS = 12
EXPECTED_STAGES = 24
EXPECTED_OUTER_FG = 10_470


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
        result[str(path.relative_to(RUN))] = sha256_file(path)
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
    arrays = {
        "eta": np.asarray(visibility["eta"], dtype=np.float64),
        "e": np.asarray(visibility["e"], dtype=np.float64),
        "degree": np.asarray(visibility["degree"], dtype=np.float64),
        "predicted_degree": np.asarray(visibility["predicted_degree"], dtype=np.float64),
        "active_mask": np.asarray(objective.support.active_mask, dtype=np.bool_),
        "zero_mask": np.asarray(objective.support.zero_mask, dtype=np.bool_),
        "numeric_position": (np.asarray(data.locus_bin, dtype=np.int64) * int(data.bin_size)),
        "chromosome_index": np.asarray(data.locus_chromosome, dtype=np.int32),
        "eta_inactive_placeholder": np.asarray(objective.mode == "V1-profile-e", dtype=np.bool_),
    }
    np.savez_compressed(path, **arrays)
    active_count = int(np.count_nonzero(arrays["active_mask"]))
    eta_semantics = {
        "V1-profile-e": "eta_active=log(e_active) after sum_active_eta_zero; inactive eta entries are zero placeholders, not log(e=0)",
        "VZ-zero-degree-fixed-e": "eta=log(e) on positive entries and -inf on exact zero-degree e=0 entries",
    }.get(objective.mode, "eta=log(fixed positive e)")
    return {
        "e_vector_length": int(len(arrays["e"])),
        "eta_active_count": active_count,
        "free_dimension": int(active_count - 1) if objective.mode == "V1-profile-e" else None,
        "eta_semantics": eta_semantics,
        "e_mean_full_grid": float(np.mean(arrays["e"])),
        "e_zero_count": int(np.count_nonzero(arrays["zero_mask"] & (arrays["e"] == 0.0))),
        "active_count": int(np.count_nonzero(arrays["active_mask"])),
        "zero_count": int(np.count_nonzero(arrays["zero_mask"])),
        "profile_relative_residual": (None if objective.mode != "V1-profile-e"
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


def stage_fit(
    attempt_id: str,
    stage_spec: dict[str, Any],
    data: Any,
    backend_mode: str,
    initial_coordinates: np.ndarray,
    p_init: float,
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
        initial_y = objective.raw_from_physical(np.asarray(initial_coordinates, dtype=np.float64))
        accepted_history = []

        def accepted_callback(entry: dict[str, Any]) -> None:
            accepted_history.append(dict(entry))
            write_json(history_path, accepted_history)

        fit_started = time.perf_counter()
        result = run_budgeted_lbfgs(
            objective, initial_y, p_init=float(p_init), maxfun=fg_cap,
            maxiter=fg_cap + 1, maxls=20, ftol=FTOL,
            canonical_gtol=CANONICAL_GTOL, checkpoint_every=25,
            accepted_callback=accepted_callback,
        )
        fit_wall = time.perf_counter() - fit_started
        endpoint_coords = np.asarray(result.coordinates, dtype=np.float64).copy()
        result_theta = np.asarray(result.theta, dtype=np.float64).copy()
        result_physical_gradient = objective.physical_gradient()
        if result_physical_gradient is None:
            raise RuntimeError("optimizer endpoint did not expose physical gradient")
        # 先写full-grid 3DG，再独立读回，再做最终profile和rescore。
        contact_model.write_full_tracks(coord_path, data, endpoint_coords)
        readback_coords = read_full_tracks(coord_path, data)
        if not np.array_equal(endpoint_coords, readback_coords):
            raise RuntimeError("3DG coordinate write/readback changed endpoint coordinates")
        readback_objective = objective_for(data, backend_mode, known_e)
        readback_raw = readback_objective.raw_from_physical(readback_coords)
        readback_theta = readback_objective.pack(readback_raw, p=float(result.p))
        readback_value, readback_gradient, readback_components = readback_objective.evaluate(
            readback_theta, need_gradient=True)
        readback_physical_gradient = readback_objective.physical_gradient()
        if readback_physical_gradient is None:
            raise RuntimeError("readback objective did not expose physical gradient")
        sidecar_summary = save_visibility_sidecar(sidecar_path, data, readback_objective)
        fixed_objective = objective_for(data, FIXED_E_RESCORE_MODE, None)
        fixed_theta = fixed_objective.pack(fixed_objective.raw_from_physical(readback_coords), p=float(result.p))
        fixed_value, fixed_gradient, fixed_components = fixed_objective.evaluate(fixed_theta, need_gradient=True)
        fixed_objective.synchronize()
        diagnostics = readback_objective.diagnostics()
        profile = diagnostics.get("last_profile") or {}
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
                "optimizer_endpoint_physical_gradient_inf": float(np.max(np.abs(result_physical_gradient))),
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
            },
            "readback": {
                "coordinate_equal": True,
                "value_abs_vs_optimizer_endpoint": abs(float(readback_value) - float(result.fun)),
                "component_total_abs_vs_optimizer_endpoint": abs(float(readback_components["total"]) - float(result.fun)),
                "sidecar_path": str(sidecar_path.relative_to(RUN)),
                "coordinate_path": str(coord_path.relative_to(RUN)),
                "accepted_history_path": str(history_path.relative_to(RUN)),
            },
            "diagnostics": diagnostics,
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
    write_json(RUN / "config.json", config)
    return {"config": config, "matrix": matrix, "real_template": real_template,
            "manifest": manifest, "synthetic_input_hashes": synthetic_input_hashes}



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
                  "expected_outer_fg": EXPECTED_OUTER_FG,
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
            provenance = {"kind": "real", "candidate": candidate,
                          "initialization": "014 blind consensus/random",
                          "input_sha256": sha256_file(INPUT_PATH)}
            known_e = None
            mode_data = None
        else:
            synth = load_synthetic_inputs(fit["fixture"], real_template, manifest)
            coordinates = synth["coordinates"].copy()
            p_init = float(synth["p_init"])
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
                    "positions": np.broadcast_to(data.locus_bin, (2, data.n_loci)).copy(),
                    "chromosome_index": np.broadcast_to(data.locus_chromosome, (2, data.n_loci)).copy(),
                }
                coordinates = endpoint_coords
                p_init = float(record["endpoint"]["p"])
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
                known_e, provenance,
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
                    "positions": np.broadcast_to(data.locus_bin, (2, data.n_loci)).copy(),
                    "chromosome_index": np.broadcast_to(data.locus_chromosome, (2, data.n_loci)).copy(),
                }
                coordinates = endpoint_coords
                p_init = float(record["endpoint"]["p"])
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
        valid = [row for row in candidates if row["status"] in ("converged", "budget_not_converged", "not_converged")
                 and row["count_nll_normalized"] is not None]
        selected = None
        if valid:
            selected = sorted(valid, key=lambda row: (float(row["count_nll_normalized"]),
                                                      0 if row["candidate"] == "consensus" else 1))[0]["candidate"]
        selections[model] = {"candidates": candidates, "selected_by_own_count_nll": selected,
                             "tie_rule": "consensus wins exact numeric tie"}
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-check", "run"))
    args = parser.parse_args()
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
