"""已发布 020 random_joint 1 Mb fit 的无 phase warm continuation。

本 runner 有意只处理一个候选，且仅用于 training。它只导入冻结的 contact objective 和 SciPy optimizer，绝不导入评价 loader 或 native FDG。checkpoint 包含 theta/y/coordinates 以及本地和累计 iteration audit；不会尝试从 parent run 重建 L-BFGS memory。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from typing import Any, Mapping

import numpy as np

from . import contact_model, genome, joint_fit
from .gate import sha256_file
from .paths import ROOT, SNPFREE


ROOT_PATH = Path(ROOT)
SNPFREE_PATH = Path(SNPFREE)
RUN_SCHEMA = "reconstruction-v1-continuation-fdg-r2-v1"
PARENT_RUN = ROOT_PATH / "test_res/020-20260913_071841-v1-p9016-joint"
START_CHECKPOINT = PARENT_RUN / "checkpoints/random_joint/1m-accepted-0240.npz"
START_FINAL = PARENT_RUN / "coords/random_joint/final-1m-660fed4c165342de93414a68e730a55a.3dg"
START_SELECTED = PARENT_RUN / "selected.3dg"
PARENT_SELECTION = PARENT_RUN / "selection.json"
OUT_RELATIVE = "test_res/022-20260913_111031-v1-continuation-fdg-r2"
SOURCE_ACCEPTED_ITERATION = 240
MAXITER = 240
MAXFUN = 750
MAXLS = 20
FTOL = 1e-10
GTOL = 1e-6
CHECKPOINT_EVERY = 20
EXPECTED_SELECTED_SHA = "afb2d52ae11e342e9b43b3c8042c5581760d177c5e563f2478646ab36b3e7078"
EXPECTED_SELECTION_SHA = "5ee6376527d6139eb4fe55ac057b40ca94a6e731289a463989f2343a815252a6"
EXPECTED_INPUT_SHA = contact_model.FROZEN_P9016_SNPFREE_SHA256
EXPECTED_PROTOCOL_SHA = "cf2ae36fdd4dec6e05d32001a5ef498add7bb62e5908a4373b2316fc380ed8e5"


class ContinuationError(RuntimeError):
    """冻结 continuation 不变量违反时抛出。"""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContinuationError("formal metadata contains a non-finite float")
        return value
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    _write_atomic(path, _json_bytes(value))


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(_json_ready(value), sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _array_sha256(array: np.ndarray) -> str:
    raw = np.ascontiguousarray(np.asarray(array)).view(np.uint8).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative(outdir: Path, path: Path) -> str:
    return str(path.resolve().relative_to(outdir.resolve()))


def _weighted_priors(components: Mapping[str, Any]) -> dict[str, float]:
    return {
        "bond": float(components["bond"]),
        "repulsion": float(components["repulsion"]),
        "bend": 0.01 * float(components["bend"]),
        "p_prior": float(components["p_prior"]),
    }


def _progress_entry(raw: Mapping[str, Any], source_actual_nfev: int) -> dict[str, Any]:
    iteration = int(raw["iteration"])
    components = dict(raw["components"])
    return {
        "iteration": iteration,
        "cumulative_iteration": SOURCE_ACCEPTED_ITERATION + iteration,
        "solver_restart": True,
        "iteration_origin": SOURCE_ACCEPTED_ITERATION,
        "nfev": int(raw["nfev"]),
        "actual_nfev": int(raw["nfev"]),
        "cumulative_nfev": int(source_actual_nfev + int(raw["nfev"])),
        "elapsed_seconds": float(raw["elapsed_seconds"]),
        "fun": float(raw["fun"]),
        "p": float(raw["p"]),
        "gradient_norm": float(raw.get("gradient_norm", 0.0)),
        "count_nll_normalized": float(components["count_nll_normalized"]),
        "conditional_nll_raw": float(components["conditional_nll_raw"]),
        "diag_profiled_nll_raw": float(components["diag_profiled_nll_raw"]),
        "count_nll_raw": float(components["count_nll_raw"]),
        "weighted_priors": _weighted_priors(components),
        "components": _json_ready(components),
    }


def _classify_termination(*, success: bool, status: int, message: str,
                          nit: int, actual_nfev: int, maxiter: int = MAXITER,
                          maxfun: int = MAXFUN) -> dict[str, Any]:
    upper = str(message).upper()
    iteration_limit = int(nit) >= int(maxiter) or "TOTAL NO. OF ITERATIONS REACHED LIMIT" in upper
    function_limit = int(actual_nfev) >= int(maxfun) or (
        "TOTAL NO. OF F" in upper and "EVALUATIONS EXCEEDS LIMIT" in upper)
    budget_exhausted = bool(iteration_limit or function_limit)
    if bool(success):
        state, reason = "converged", "solver_reported_success"
        budget_exhausted = False
    elif budget_exhausted:
        state, reason = "not_converged", "budget_exhausted"
    else:
        state, reason = "not_converged", "solver_reported_nonconvergence"
    return {
        "status": state,
        "termination_reason": reason,
        "budget_exhausted": budget_exhausted,
        "iteration_limit_reached": bool(iteration_limit),
        "function_limit_reached": bool(function_limit),
        "success": bool(success),
        "scipy_status": int(status),
        "message": str(message),
        "nit": int(nit),
        "actual_nfev": int(actual_nfev),
        "maxiter": int(maxiter),
        "maxfun": int(maxfun),
    }


def _read_parent_json() -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (PARENT_SELECTION, START_CHECKPOINT, START_FINAL, START_SELECTED):
        if not path.is_file():
            raise ContinuationError("required 020 start artifact is unavailable: %s" % path)
    selection_sha = sha256_file(PARENT_SELECTION)
    if selection_sha != EXPECTED_SELECTION_SHA:
        raise ContinuationError("020 selection SHA mismatch: %s" % selection_sha)
    selected_sha = sha256_file(START_SELECTED)
    final_sha = sha256_file(START_FINAL)
    if selected_sha != EXPECTED_SELECTED_SHA or final_sha != EXPECTED_SELECTED_SHA:
        raise ContinuationError("020 selected/final coordinate SHA mismatch")
    with PARENT_SELECTION.open() as handle:
        selection = json.load(handle)
    with (PARENT_RUN / "config.json").open() as handle:
        config = json.load(handle)
    candidates = {row.get("id"): row for row in selection.get("candidates", [])}
    random = candidates.get("random_joint")
    if not isinstance(random, dict):
        raise ContinuationError("020 selection has no random_joint candidate")
    if selection.get("selection", {}).get("selected_id") != "random_joint":
        raise ContinuationError("020 selected candidate is not random_joint")
    selected_path = random.get("coordinates", {}).get("path")
    if selected_path != "coords/random_joint/final-1m-660fed4c165342de93414a68e730a55a.3dg":
        raise ContinuationError("020 random_joint final path changed")
    expected = {
        "p": 0.9578874282644998,
        "q": 3.1266569616552804,
        "count_nll_normalized": 9.593585931292237,
        "total": 9.610015771246907,
    }
    score = random.get("count_model", {})
    q = random.get("selection_rescore", {}).get("q_fixed_from_final_fit")
    if q is None or not math.isclose(float(q), expected["q"], rel_tol=0.0, abs_tol=1e-14):
        raise ContinuationError("020 random_joint q start metadata changed")
    for key in ("p", "count_nll_normalized", "total"):
        if not math.isclose(float(score[key]), expected[key], rel_tol=0.0, abs_tol=1e-14):
            raise ContinuationError("020 random_joint %s start metadata changed" % key)
    return config, {
        "selection_sha256": selection_sha,
        "selected_sha256": selected_sha,
        "final_sha256": final_sha,
        "checkpoint_sha256": sha256_file(START_CHECKPOINT),
        "q": expected["q"],
        "p": expected["p"],
        "count_nll_normalized": expected["count_nll_normalized"],
        "total": expected["total"],
        "selected_path": str(START_SELECTED),
        "final_path": str(START_FINAL),
    }


def _snapshot_files(outdir: Path, root_name: str, sources: tuple[str, ...], schema: str) -> dict[str, Any]:
    rows = []
    for source_name in sources:
        source = ROOT_PATH / source_name
        if not source.is_file():
            raise ContinuationError("cannot snapshot missing source %s" % source)
        destination = outdir / root_name / source_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        rows.append({"source_path": source_name, "snapshot_path": _relative(outdir, destination),
                     "sha256": sha256_file(destination)})
    manifest = {"schema_version": schema, "files": rows}
    manifest_path = outdir / ("provenance/%s-manifest.json" % ("training-code" if "training-code" in root_name else "protocol"))
    _write_json(manifest_path, manifest)
    return {"path": _relative(outdir, manifest_path), "sha256": sha256_file(manifest_path),
            "schema_version": schema, "files": rows}


def _dependency_snapshot(outdir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "numpy": np.__version__,
    }
    try:
        import scipy
        payload["scipy"] = scipy.__version__
    except Exception as exc:  # pragma: no cover - runner 依赖 scipy
        payload["scipy_error"] = "%s: %s" % (type(exc).__name__, exc)
    try:
        completed = subprocess.run(["conda", "list", "--explicit"], check=False,
                                   capture_output=True, text=True)
        explicit = completed.stdout
        payload["conda_list_returncode"] = int(completed.returncode)
        payload["conda_list_sha256"] = hashlib.sha256(explicit.encode()).hexdigest()
        (outdir / "provenance" / "conda-list-explicit.txt").write_text(explicit)
    except OSError as exc:
        payload["conda_list_error"] = "%s: %s" % (type(exc).__name__, exc)
    path = outdir / "provenance/dependency-versions.json"
    _write_json(path, payload)
    payload["path"] = _relative(outdir, path)
    payload["sha256"] = sha256_file(path)
    return payload


def _build_config(parent_config: Mapping[str, Any], start: Mapping[str, Any],
                  code: Mapping[str, Any], protocol: Mapping[str, Any],
                  dependencies: Mapping[str, Any]) -> dict[str, Any]:
    grid = parent_config["coordinate_grid"]
    model_definition = dict(parent_config["model_definition"])
    model_definition["joint_fit_source_sha256"] = sha256_file(ROOT_PATH / "pr/joint_fit.py")
    model_definition["continuation_runner_source_sha256"] = sha256_file(ROOT_PATH / "pr/continuation.py")
    return {
        "schema_version": RUN_SCHEMA,
        "purpose": "formal_phase_free_training_only",
        "parent_release": "handoff-accepted continuation from 020 random_joint 1 Mb endpoint",
        "parent_run": str(PARENT_RUN),
        "cohort": {
            "sample_id": "P9016",
            "biological_samples": 1,
            "raw_contacts": contact_model.FROZEN_P9016_RECORDS,
            "intra_contacts": contact_model.FROZEN_P9016_CIS_RECORDS,
            "inter_contacts": contact_model.FROZEN_P9016_INTER_RECORDS,
            "snpfree_sha256": EXPECTED_INPUT_SHA,
        },
        "input": {
            "path": str(SNPFREE_PATH.resolve()),
            "sha256": EXPECTED_INPUT_SHA,
            "training_columns": ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"],
            "uses_all_records": True,
        },
        "coordinate_grid": {
            "training_origin_bp": 0,
            "full_grid": True,
            "bin_size_bp": 1_000_000,
            "expected_loci": 2645,
            "expected_physical_beads": 5290,
            "n_tracks": 40,
            "nuclear_radius": 1.0,
            "coordinate_units": "dimensionless_R1",
            "chromosomes": grid["chromosomes"],
        },
        "model_contract": parent_config["model_contract"],
        "model_definition": model_definition,
        "starting_endpoint": {
            "candidate_id": "random_joint",
            "checkpoint_path": str(START_CHECKPOINT),
            "checkpoint_sha256": start["checkpoint_sha256"],
            "final_coordinate_path": str(START_FINAL),
            "final_coordinate_sha256": start["final_sha256"],
            "selected_coordinate_path": str(START_SELECTED),
            "selected_coordinate_sha256": start["selected_sha256"],
            "accepted_iteration": SOURCE_ACCEPTED_ITERATION,
            "q": start["q"],
            "p": start["p"],
            "count_nll_normalized": start["count_nll_normalized"],
            "total": start["total"],
            "restart_kind": "warm_restart_no_lbfgs_memory",
        },
        "optimization": {
            "solver": "scipy.optimize.minimize:L-BFGS-B",
            "maxiter": MAXITER,
            "maxfun": MAXFUN,
            "maxls": MAXLS,
            "ftol": FTOL,
            "gtol": GTOL,
            "checkpoint_every_accepted_iterations": CHECKPOINT_EVERY,
            "local_iteration_range": [0, MAXITER],
            "cumulative_iteration_range": [SOURCE_ACCEPTED_ITERATION,
                                            SOURCE_ACCEPTED_ITERATION + MAXITER],
            "solver_restart": True,
            "q_init_exact_raw": True,
            "selection_or_checkpoint_choice": "none; endpoint is solver final accepted state",
        },
        "checkpoint_contract": {
            "arrays": ["theta", "y", "coordinates", "positions", "chromosome_index"],
            "history_fields": ["iteration", "cumulative_iteration", "nfev", "cumulative_nfev",
                               "fun", "count_nll_normalized", "weighted_priors",
                               "gradient_norm", "elapsed_seconds"],
            "checkpoint_hash": "sha256(file_bytes) plus theta/coordinates raw-array SHA256",
        },
        "frozen_model_settings": {
            "kernel": "bounded_kernel epsilon=1e-6 exponent=4 radius=2*l0",
            "exposure": "observed_endpoint sqrt(endpoint_count+10), full-grid mean one",
            "count_groups": ["cis_offdiag", "inter"],
            "same_bin_layer": "per_bin_saturated_poisson_nuisance",
            "weights": {"count": 1.0, "p_prior": 1.0, "bond": 1.0,
                        "repulsion": 1.0, "bend": 0.01},
            "all_cis_inter_samebin_zero_eligible_sets": "frozen V1 aggregate, unchanged",
        },
        "provenance": {
            "training_code_manifest": code["path"],
            "protocol_manifest": protocol["path"],
            "dependency_versions": dependencies["path"],
        },
        "training_boundary": {
            "phase_used": False,
            "reference_used": False,
            "oracle_coordinates_opened": False,
            "softall_coordinates_opened": False,
            "native_fdg_started": False,
            "fdg_or_r2_evaluation_performed": False,
        },
    }


def _write_track_map(outdir: Path, parent_config: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for chromosome_index, chromosome in enumerate(parent_config["coordinate_grid"]["chromosomes"]):
        for copy_index, copy_name in enumerate(("a", "b")):
            rows.append({"track": "c%02d%s" % (chromosome_index + 1, copy_name),
                         "chromosome_index": chromosome_index,
                         "chromosome_name": chromosome["name"],
                         "copy_index": copy_index,
                         "length_bp": chromosome["length_bp"]})
    payload = {"schema_version": "reconstruction-v1-continuation-track-map-v1",
               "order": "original SNP-free header order, copy a then b", "tracks": rows}
    path = outdir / "track_map.json"
    _write_json(path, payload)
    return {"path": _relative(outdir, path), "sha256": sha256_file(path), "n_tracks": len(rows)}


def _read_coordinate_file(path: Path, data: contact_model.AggregatedContacts) -> np.ndarray:
    expected = {}
    for spec in data.track_specs:
        slc = data.chromosome_slice(spec.chromosome_index)
        for global_index in range(slc.start, slc.stop):
            expected[(spec.name, int(data.locus_bin[global_index] * data.bin_size))] = (
                spec.copy_index, global_index)
    result = np.empty((2, data.n_loci, 3), dtype=np.float64)
    seen = set()
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ContinuationError("coordinate line %d has %d fields" % (line_no, len(fields)))
            track = fields[0]
            try:
                position = int(fields[1])
                xyz = np.asarray([float(value) for value in fields[2:5]], dtype=np.float64)
            except ValueError as exc:
                raise ContinuationError("coordinate line %d is not numeric" % line_no) from exc
            key = (track, position)
            if key not in expected or key in seen or not np.isfinite(xyz).all():
                raise ContinuationError("invalid, duplicate, or unexpected coordinate at line %d" % line_no)
            copy_index, global_index = expected[key]
            result[copy_index, global_index] = xyz
            seen.add(key)
    if seen != set(expected):
        raise ContinuationError("coordinate file is not the complete 40-track 1 Mb full grid")
    contact_model.assert_inside_unit_ball(result)
    return result


def _write_coordinate_file(path: Path, data: contact_model.AggregatedContacts,
                           coordinates: np.ndarray) -> dict[str, Any]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (2, data.n_loci, 3):
        raise ContinuationError("coordinate output shape is invalid")
    contact_model.assert_inside_unit_ball(coordinates)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for spec in data.track_specs:
            slc = data.chromosome_slice(spec.chromosome_index)
            for global_index in range(slc.start, slc.stop):
                position = int(data.locus_bin[global_index] * data.bin_size)
                xyz = coordinates[spec.copy_index, global_index]
                handle.write(("%s\t%d\t%.17g\t%.17g\t%.17g\n" %
                              (spec.name, position, xyz[0], xyz[1], xyz[2])).encode())
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": _relative(path.parent.parent.parent, path), "sha256": sha256_file(path),
            "n_tracks": len(data.track_specs), "n_beads": int(2 * data.n_loci),
            "max_radius": float(np.linalg.norm(coordinates, axis=2).max())}


def _load_start(data: contact_model.AggregatedContacts, start: Mapping[str, Any]) -> dict[str, Any]:
    with np.load(START_CHECKPOINT, allow_pickle=False) as payload:
        fields = set(payload.files)
        required = {"coordinates", "theta", "y", "positions", "chromosome_index", "fullhistory_json"}
        if not required.issubset(fields):
            raise ContinuationError("start checkpoint lacks required arrays")
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        theta = np.asarray(payload["theta"], dtype=np.float64).copy()
        y = np.asarray(payload["y"], dtype=np.float64).copy()
        positions = np.asarray(payload["positions"], dtype=np.int64).copy()
        chromosome_index = np.asarray(payload["chromosome_index"], dtype=np.int64).copy()
        old_history = json.loads(str(payload["fullhistory_json"].item()))
    expected_positions = np.asarray(data.locus_bin, dtype=np.int64) * int(data.bin_size)
    expected_chromosome = np.asarray(data.locus_chromosome, dtype=np.int64)
    if coordinates.shape != (2, data.n_loci, 3) or y.shape != coordinates.shape:
        raise ContinuationError("start checkpoint coordinate/y shape is not (2,2645,3)")
    if theta.shape != (6 * data.n_loci + 1,) or not np.isfinite(theta).all():
        raise ContinuationError("start checkpoint theta shape or finiteness is invalid")
    if not np.array_equal(theta[:-1].reshape(y.shape), y):
        raise ContinuationError("start checkpoint theta/y arrays disagree")
    if not np.array_equal(positions, expected_positions) or not np.array_equal(chromosome_index, expected_chromosome):
        raise ContinuationError("start checkpoint coordinate grid arrays disagree with full 0-based grid")
    if not np.isfinite(coordinates).all() or not np.isfinite(y).all():
        raise ContinuationError("start checkpoint contains non-finite coordinates")
    objective = contact_model.JointObjective(data)
    recomputed, p = objective.coordinates_and_p(theta)
    if not np.allclose(recomputed, coordinates, rtol=0.0, atol=0.0):
        raise ContinuationError("theta -> coordinates is not exact for the saved checkpoint")
    old_coordinates = _read_coordinate_file(START_FINAL, data)
    old_coordinate_max_abs_diff = float(np.max(np.abs(old_coordinates - coordinates)))
    if old_coordinate_max_abs_diff > 1e-12:
        raise ContinuationError("checkpoint coordinates differ from released 020 final coordinates")
    value, _, components = objective.evaluate(theta, need_gradient=False)
    expected_start = {
        "p": float(start["p"]),
        "count_nll_normalized": float(start["count_nll_normalized"]),
        "total": float(start["total"]),
    }
    for key in expected_start:
        if not math.isclose(float(components[key] if key != "p" else p), expected_start[key],
                            rel_tol=0.0, abs_tol=2e-12):
            raise ContinuationError("recomputed start %s conflicts with 020 record" % key)
    theta_before = theta.copy()
    coordinates_before = coordinates.copy()
    diagnostic_value, diagnostic_gradient = objective.value_and_grad(theta.copy())
    diagnostic_norm = float(np.linalg.norm(diagnostic_gradient))
    gradient_safe = (np.array_equal(theta, theta_before)
                     and np.array_equal(coordinates, coordinates_before)
                     and np.isfinite(diagnostic_value)
                     and np.isfinite(diagnostic_gradient).all())
    if not gradient_safe:
        raise ContinuationError("gradient diagnostic changed start state or became non-finite")
    termination_tests = {
        "iteration_budget": _classify_termination(
            success=False, status=1, message="STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
            nit=MAXITER, actual_nfev=MAXITER + 1),
        "function_budget": _classify_termination(
            success=False, status=1, message="STOP: TOTAL NO. OF F EVALUATIONS EXCEEDS LIMIT",
            nit=17, actual_nfev=MAXFUN),
        "solver_success": _classify_termination(
            success=True, status=0, message="CONVERGENCE", nit=MAXITER, actual_nfev=MAXITER + 1),
        "ordinary_failure": _classify_termination(
            success=False, status=2, message="ABNORMAL_TERMINATION_IN_LNSRCH", nit=17, actual_nfev=18),
    }
    if (termination_tests["iteration_budget"]["status"] != "not_converged"
            or not termination_tests["iteration_budget"]["budget_exhausted"]
            or termination_tests["solver_success"]["status"] != "converged"
            or termination_tests["ordinary_failure"]["budget_exhausted"]):
        raise ContinuationError("termination classifier preflight failed")
    return {
        "data": data,
        "objective": objective,
        "theta": theta,
        "y": y,
        "coordinates": coordinates,
        "positions": positions,
        "chromosome_index": chromosome_index,
        "p": float(p),
        "value": float(value),
        "components": components,
        "old_history_entries": int(len(old_history)),
        "old_coordinate_max_abs_diff": old_coordinate_max_abs_diff,
        "gradient_norm": diagnostic_norm,
        "gradient_value": float(diagnostic_value),
        "termination_tests": termination_tests,
    }


def _write_preflight(outdir: Path, state: Mapping[str, Any], start: Mapping[str, Any],
                     static: Mapping[str, Any]) -> None:
    data: contact_model.AggregatedContacts = state["data"]
    budget = data.budget()
    expected = {
        "raw_records": contact_model.FROZEN_P9016_RECORDS,
        "raw_inter": contact_model.FROZEN_P9016_INTER_RECORDS,
        "raw_same_bin": 438774,
        "raw_cis_offdiag": 696680,
        "n_loci": 2645,
        "n_eligible_pairs": 3496690,
        "n_zero_eligible_pairs": 3009436,
    }
    for key, value in expected.items():
        if int(budget[key]) != int(value):
            raise ContinuationError("preflight budget mismatch for %s" % key)
    start_path = outdir / "coords/random_joint/start-1m.3dg"
    if start_path.exists():
        raise ContinuationError("start coordinate artifact already exists")
    _write_coordinate_file(start_path, data, state["coordinates"])
    start_npz = outdir / "provenance/start-checkpoint-0240.npz"
    if start_npz.exists():
        raise ContinuationError("start checkpoint copy already exists")
    shutil.copyfile(START_CHECKPOINT, start_npz)
    start_npz_sha = sha256_file(start_npz)
    if start_npz_sha != start["checkpoint_sha256"]:
        raise ContinuationError("start checkpoint provenance copy changed")
    preflight = {
        "schema_version": "reconstruction-v1-continuation-preflight-v1",
        "status": "passed",
        "timestamp_utc": _utc_now(),
        "input": {"path": str(SNPFREE_PATH.resolve()), "sha256": EXPECTED_INPUT_SHA,
                  "budget": budget, "expected_budget": expected},
        "start_checkpoint": {
            "source_path": str(START_CHECKPOINT), "source_sha256": start["checkpoint_sha256"],
            "run_copy": _relative(outdir, start_npz), "run_copy_sha256": start_npz_sha,
            "required_fields": ["coordinates", "theta", "y", "positions", "chromosome_index", "fullhistory_json"],
            "old_history_entries": state["old_history_entries"],
            "accepted_iteration": SOURCE_ACCEPTED_ITERATION,
            "restart_kind": "warm_restart_no_lbfgs_memory",
        },
        "theta_to_coordinates": {
            "theta_shape": list(state["theta"].shape), "y_shape": list(state["y"].shape),
            "coordinates_shape": list(state["coordinates"].shape),
            "exact_sphere_transform": True,
            "max_radius": float(np.linalg.norm(state["coordinates"], axis=2).max()),
            "old_final_coordinate_max_abs_diff": state["old_coordinate_max_abs_diff"],
            "start_coordinate_path": _relative(outdir, start_path),
            "start_coordinate_sha256": sha256_file(start_path),
        },
        "score_recompute": {
            "q": float(state["theta"][-1]), "p": state["p"],
            "count_nll_normalized": float(state["components"]["count_nll_normalized"]),
            "conditional_nll_raw": float(state["components"]["conditional_nll_raw"]),
            "diag_profiled_nll_raw": float(state["components"]["diag_profiled_nll_raw"]),
            "count_nll_raw": float(state["components"]["count_nll_raw"]),
            "weighted_priors": _weighted_priors(state["components"]),
            "total": float(state["components"]["total"]),
            "published_start": {"q": start["q"], "p": start["p"],
                                "count_nll_normalized": start["count_nll_normalized"],
                                "total": start["total"]},
        },
        "gradient_diagnostic": {
            "value": state["gradient_value"], "gradient_norm": state["gradient_norm"],
            "theta_unchanged": True, "coordinates_unchanged": True, "finite": True,
            "non_side_effect": True,
        },
        "termination_classifier_tests": state["termination_tests"],
        "training_boundary": {
            "phase_used": False, "reference_used": False,
            "oracle_coordinates_opened": False, "softall_coordinates_opened": False,
            "native_fdg_started": False, "fdg_or_r2_evaluation_performed": False,
            "forbidden_coordinate_payloads_opened": [],
        },
        "static_provenance": static,
    }
    _write_json(outdir / "preflight.json", preflight)
    _append_jsonl(outdir / "logs/preflight_attempts.jsonl", {
        "attempt": "continuation_preflight", "status": "passed",
        "input_sha256": EXPECTED_INPUT_SHA, "start_coordinate_sha256": sha256_file(start_path),
        "gradient_norm": state["gradient_norm"], "budget_conserved": True,
    })


def _prepare(outdir: Path, parent_config: Mapping[str, Any], start: Mapping[str, Any]) -> dict[str, Any]:
    for relative in ("checkpoints/random_joint", "coords/random_joint", "theta", "provenance", "run_status"):
        (outdir / relative).mkdir(parents=True, exist_ok=True)
    code = _snapshot_files(
        outdir, "provenance/training-code",
        ("pr/continuation.py", "pr/contact_model.py", "pr/joint_fit.py", "pr/genome.py",
         "pr/paths.py", "pr/gate.py"),
        "reconstruction-v1-continuation-training-code-v1",
    )
    protocol = _snapshot_files(
        outdir, "provenance/protocol",
        ("AGENTS.md", "docs/RECONSTRUCTION_V1_PROTOCOL.md", "docs/RECONSTRUCTION_V1_RUN.md"),
        "reconstruction-v1-continuation-protocol-v1",
    )
    protocol_member = next(row for row in protocol["files"]
                           if row["source_path"] == "docs/RECONSTRUCTION_V1_PROTOCOL.md")
    if protocol_member["sha256"] != EXPECTED_PROTOCOL_SHA:
        raise ContinuationError("current protocol bytes are not the frozen V1 protocol")
    dependencies = _dependency_snapshot(outdir)
    config = _build_config(parent_config, start, code, protocol, dependencies)
    protocol_freeze = {
        "schema_version": "reconstruction-v1-continuation-protocol-freeze-v1",
        "status": "parent_release_handoff",
        "parent_release": "authorized handoff for continuation training",
        "scope": "020 random_joint 1 Mb endpoint plus exactly 240 accepted-step budget",
        "input": {"path": str(SNPFREE_PATH.resolve()), "sha256": EXPECTED_INPUT_SHA,
                  "records": contact_model.FROZEN_P9016_RECORDS,
                  "intra": contact_model.FROZEN_P9016_CIS_RECORDS,
                  "inter": contact_model.FROZEN_P9016_INTER_RECORDS},
        "grid": {"bin_size_bp": 1_000_000, "origin_bp": 0, "loci": 2645,
                 "physical_beads": 5290, "tracks": 40},
        "start": config["starting_endpoint"],
        "solver": config["optimization"],
        "model": config["frozen_model_settings"],
        "selection": {"criterion": "none during continuation; no reference-based checkpoint choice",
                      "endpoint": "valid final accepted solver state"},
        "boundary": config["training_boundary"],
        "root_RUN_md": {"present": False, "actual_run_document": "docs/RECONSTRUCTION_V1_RUN.md"},
        "source_protocol_sha256": EXPECTED_PROTOCOL_SHA,
    }
    _write_json(outdir / "protocol-freeze.json", protocol_freeze)
    config["protocol_freeze"] = {"path": "protocol-freeze.json",
                                  "sha256": sha256_file(outdir / "protocol-freeze.json")}
    _write_json(outdir / "config.json", config)
    track_map = _write_track_map(outdir, parent_config)
    input_version = {"path": str(SNPFREE_PATH.resolve()), "sha256": EXPECTED_INPUT_SHA,
                     "hash_source": "file_bytes", "records": contact_model.FROZEN_P9016_RECORDS}
    _write_json(outdir / "input_version.json", input_version)
    static = {
        "config": {"path": "config.json", "sha256": sha256_file(outdir / "config.json")},
        "protocol_freeze": {"path": "protocol-freeze.json",
                             "sha256": sha256_file(outdir / "protocol-freeze.json")},
        "training_code": {"path": code["path"], "sha256": code["sha256"]},
        "protocol": {"path": protocol["path"], "sha256": protocol["sha256"]},
        "dependencies": {"path": dependencies["path"], "sha256": dependencies["sha256"]},
        "input_version": {"path": "input_version.json", "sha256": sha256_file(outdir / "input_version.json")},
        "track_map": track_map,
    }
    _write_json(outdir / "provenance/static-manifest.json", static)
    readme = """# 022 V1 延续训练

这是从已发布的 020 `random_joint` 1 Mb endpoint warm restart 得到的无 phase 训练记录。checkpoint 包含 theta/y/coordinates，但不含 L-BFGS memory，因此 solver restart 在每条 progress 行和 checkpoint 中都显式记录。本次运行在全部 1,703,888 条 SNP-free 记录上使用未改变的 V1 完整网格 contact model，严格执行已注册的 240 步 continuation budget。

训练 runner 不打开带 phase 的输入、reference 3DG、oracle/softall 坐标 payload 或 native FDG，也不执行 FDG 或 R2 评价。独立的 FDG/R2 branch 只有在本目录完成哈希审计后，才接收最终坐标路径。

参见 `config.json`、`protocol-freeze.json`、`preflight.json`、`logs/progress.jsonl`、`checkpoints/`、`final_theta.npz` 和 `terminal_audit.json`。

"""
    _write_atomic(outdir / "README.md", readme.encode())
    _write_json(outdir / "run_status/run_status.json", {
        "status": "prepared", "created_at_utc": _utc_now(),
        "start_checkpoint_sha256": start["checkpoint_sha256"], "candidate": "random_joint",
        "solver_restart": True,
    })
    _write_json(outdir / "provenance/source-inputs.json", {
        "parent_run": str(PARENT_RUN), "parent_selection_sha256": start["selection_sha256"],
        "parent_selected_sha256": start["selected_sha256"], "parent_final_sha256": start["final_sha256"],
        "parent_checkpoint_sha256": start["checkpoint_sha256"],
        "reference_used": False, "phase_used": False,
    })
    return static


def _log(outdir: Path, message: str) -> None:
    line = "[%s] %s" % (dt.datetime.now().strftime("%H:%M:%S"), message)
    print(line, flush=True)
    with (outdir / "logs/continuation.log").open("at") as handle:
        handle.write(line + "\n")
        handle.flush()


def _run_optimizer(outdir: Path, state: Mapping[str, Any], start: Mapping[str, Any]) -> dict[str, Any]:
    data: contact_model.AggregatedContacts = state["data"]
    objective: contact_model.JointObjective = state["objective"]
    source_actual_nfev = 241
    progress_path = outdir / "logs/progress.jsonl"
    checkpoint_manifest_path = outdir / "logs/checkpoints.jsonl"
    initial_raw = {
        "iteration": 0, "nfev": 0, "elapsed_seconds": 0.0,
        "fun": state["value"], "p": state["p"], "gradient_norm": state["gradient_norm"],
        "components": state["components"],
    }
    progress_history = [_progress_entry(initial_raw, source_actual_nfev)]
    _append_jsonl(progress_path, progress_history[0])
    _log(outdir, "start local_iteration=0 cumulative_iteration=240 nfev=0 total=%.12g count=%.12g grad=%.8g" %
         (state["value"], state["components"]["count_nll_normalized"], state["gradient_norm"]))
    checkpoint_records: list[dict[str, Any]] = []

    def callback(entry: Mapping[str, Any]) -> None:
        item = _progress_entry(entry, source_actual_nfev)
        if progress_history and item["iteration"] == progress_history[-1]["iteration"]:
            progress_history[-1] = item
        else:
            progress_history.append(item)
        _append_jsonl(progress_path, item)
        _log(outdir, "accepted local_iteration=%d cumulative_iteration=%d nfev=%d total=%.12g count=%.12g grad=%.8g" %
             (item["iteration"], item["cumulative_iteration"], item["actual_nfev"],
              item["fun"], item["count_nll_normalized"], item["gradient_norm"]))

    def checkpoint_hook(checkpoint: joint_fit.JointCheckpoint) -> None:
        raw = {
            "iteration": checkpoint.iteration, "nfev": checkpoint.nfev,
            "elapsed_seconds": checkpoint.elapsed_seconds, "fun": checkpoint.fun,
            "p": checkpoint.p, "gradient_norm": checkpoint.gradient_norm,
            "components": checkpoint.components,
        }
        item = _progress_entry(raw, source_actual_nfev)
        history = list(progress_history)
        if not history or history[-1]["iteration"] != item["iteration"]:
            history.append(item)
        local_iteration = int(checkpoint.iteration)
        checkpoint_path = outdir / ("checkpoints/random_joint/1m-continuation-accepted-%04d.npz" % local_iteration)
        restart_metadata = {
            "solver_restart": True, "source_accepted_iteration": SOURCE_ACCEPTED_ITERATION,
            "local_iteration": local_iteration,
            "cumulative_iteration": SOURCE_ACCEPTED_ITERATION + local_iteration,
            "source_checkpoint_sha256": start["checkpoint_sha256"],
            "history_is_local_since_restart": True,
        }
        _write_npz(
            checkpoint_path,
            coordinates=np.asarray(checkpoint.coordinates, dtype=np.float64),
            theta=np.asarray(checkpoint.theta, dtype=np.float64),
            y=np.asarray(checkpoint.y, dtype=np.float64),
            positions=np.asarray(state["positions"], dtype=np.int64),
            chromosome_index=np.asarray(state["chromosome_index"], dtype=np.int64),
            fullhistory_json=np.asarray(json.dumps(_json_ready(history), sort_keys=True)),
            restart_metadata_json=np.asarray(json.dumps(restart_metadata, sort_keys=True)),
        )
        record = {
            "path": _relative(outdir, checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "local_iteration": local_iteration,
            "cumulative_iteration": SOURCE_ACCEPTED_ITERATION + local_iteration,
            "nfev": int(checkpoint.nfev),
            "cumulative_nfev": source_actual_nfev + int(checkpoint.nfev),
            "theta_sha256": _array_sha256(checkpoint.theta),
            "coordinates_sha256": _array_sha256(checkpoint.coordinates),
            "fun": float(checkpoint.fun),
            "count_nll_normalized": float(checkpoint.components["count_nll_normalized"]),
            "gradient_norm": float(checkpoint.gradient_norm),
            "elapsed_seconds": float(checkpoint.elapsed_seconds),
            "solver_restart": True,
        }
        checkpoint_records.append(record)
        _append_jsonl(checkpoint_manifest_path, record)
        _log(outdir, "checkpoint local_iteration=%d cumulative_iteration=%d sha256=%s" %
             (local_iteration, SOURCE_ACCEPTED_ITERATION + local_iteration, record["sha256"]))

    _log(outdir, "solver start maxiter=%d maxfun=%d maxls=%d ftol=%.1g gtol=%.1g q_init=%.17g warm_restart=true" %
         (MAXITER, MAXFUN, MAXLS, FTOL, GTOL, float(state["theta"][-1])))
    try:
        result = joint_fit.fit_joint(
            objective,
            state["y"],
            p_init=0.75,
            q_init=float(state["theta"][-1]),
            maxiter=MAXITER,
            maxfun=MAXFUN,
            maxls=MAXLS,
            ftol=FTOL,
            gtol=GTOL,
            callback=callback,
            checkpoint_every=CHECKPOINT_EVERY,
            checkpoint_hook=checkpoint_hook,
        )
    except Exception as exc:
        failure = {
            "status": "failed",
            "phase": "optimization",
            "timestamp_utc": _utc_now(),
            "failure": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
            "start_preserved": True,
            "progress_entries": len(progress_history),
            "checkpoint_records": checkpoint_records,
        }
        _write_json(outdir / "failure.json", failure)
        _write_json(outdir / "terminal_audit.json", failure)
        _write_json(outdir / "run_status/run_status.json", {"status": "failed", **failure})
        _log(outdir, "solver failed; start checkpoint and all partial checkpoints retained")
        raise

    final_theta = np.asarray(result.theta, dtype=np.float64).copy()
    final_coordinates, final_p = objective.coordinates_and_p(final_theta)
    final_value, final_gradient, final_components = objective.evaluate(final_theta, need_gradient=True)
    final_gradient_norm = float(np.linalg.norm(final_gradient))
    termination = _classify_termination(
        success=bool(result.success), status=int(result.status), message=str(result.message),
        nit=int(result.nit), actual_nfev=int(result.actual_nfev),
    )
    theta_sha = _array_sha256(final_theta)
    final_theta_path = outdir / "theta/final-theta.npz"
    _write_npz(
        final_theta_path,
        theta=final_theta,
        y=final_theta[:-1].reshape(2, data.n_loci, 3),
        coordinates=final_coordinates,
        positions=np.asarray(state["positions"], dtype=np.int64),
        chromosome_index=np.asarray(state["chromosome_index"], dtype=np.int64),
        restart_metadata_json=np.asarray(json.dumps({
            "solver_restart": True, "source_accepted_iteration": SOURCE_ACCEPTED_ITERATION,
            "final_local_iteration": int(result.nit),
            "final_cumulative_iteration": SOURCE_ACCEPTED_ITERATION + int(result.nit),
        }, sort_keys=True)),
    )
    final_theta_file_sha = sha256_file(final_theta_path)
    final_coordinate_path = outdir / ("coords/random_joint/final-1m-continuation-%s.3dg" % theta_sha[:16])
    coordinate_info = _write_coordinate_file(final_coordinate_path, data, final_coordinates)
    serialized_coordinates = _read_coordinate_file(final_coordinate_path, data)
    serialization_max_abs_diff = float(np.max(np.abs(serialized_coordinates - final_coordinates)))
    norms = np.linalg.norm(serialized_coordinates, axis=2)
    if serialization_max_abs_diff > 1e-15:
        raise ContinuationError("final coordinate serialization changed the optimizer coordinates")
    if not bool(np.all(norms < 1.0)):
        raise ContinuationError("final coordinates do not prove strict R<1")
    if coordinate_info["n_tracks"] != 40 or coordinate_info["n_beads"] != 5290:
        raise ContinuationError("final coordinate inventory is not 40 tracks/5290 beads")
    endpoint_record = {
        "record_type": "terminal_endpoint",
        "iteration": int(result.nit),
        "cumulative_iteration": SOURCE_ACCEPTED_ITERATION + int(result.nit),
        "solver_restart": True,
        "accepted_endpoint": True,
        "nfev": int(result.actual_nfev),
        "cumulative_nfev": source_actual_nfev + int(result.actual_nfev),
        "elapsed_seconds": float(result.elapsed_seconds),
        "fun": float(final_value),
        "p": float(final_p),
        "gradient_norm": final_gradient_norm,
        "count_nll_normalized": float(final_components["count_nll_normalized"]),
        "conditional_nll_raw": float(final_components["conditional_nll_raw"]),
        "diag_profiled_nll_raw": float(final_components["diag_profiled_nll_raw"]),
        "count_nll_raw": float(final_components["count_nll_raw"]),
        "weighted_priors": _weighted_priors(final_components),
        "components": _json_ready(final_components),
    }
    _append_jsonl(outdir / "logs/terminal-endpoint.jsonl", endpoint_record)
    noninferior = bool(final_value <= state["value"] + 1e-10 * max(1.0, abs(state["value"])))
    audit = {
        "schema_version": "reconstruction-v1-continuation-terminal-audit-v1",
        "status": "complete",
        "timestamp_utc": _utc_now(),
        "candidate_id": "random_joint",
        "solver_restart": True,
        "source": {
            "parent_run": str(PARENT_RUN), "checkpoint_path": str(START_CHECKPOINT),
            "checkpoint_sha256": start["checkpoint_sha256"],
            "source_accepted_iteration": SOURCE_ACCEPTED_ITERATION,
            "restart_kind": "warm_restart_no_lbfgs_memory",
        },
        "termination": {
            **termination,
            "scipy_nfev": int(result.scipy_nfev),
            "njev": int(result.njev),
            "elapsed_seconds": float(result.elapsed_seconds),
            "final_gradient_norm": final_gradient_norm,
        },
        "iterations": {
            "local_start": 0, "local_end": int(result.nit),
            "cumulative_start": SOURCE_ACCEPTED_ITERATION,
            "cumulative_end": SOURCE_ACCEPTED_ITERATION + int(result.nit),
            "accepted_checkpoint_local_iterations": [record["local_iteration"] for record in checkpoint_records],
            "checkpoint_count": len(checkpoint_records),
            "checkpoint_every": CHECKPOINT_EVERY,
        },
        "before_after": {
            "start": {"q": state["theta"][-1], "p": state["p"],
                      "count_nll_normalized": state["components"]["count_nll_normalized"],
                      "conditional_nll_raw": state["components"]["conditional_nll_raw"],
                      "diag_profiled_nll_raw": state["components"]["diag_profiled_nll_raw"],
                      "count_nll_raw": state["components"]["count_nll_raw"],
                      "weighted_priors": _weighted_priors(state["components"]),
                      "total": state["components"]["total"]},
            "final": {"q": float(final_theta[-1]), "p": float(final_p),
                      "count_nll_normalized": float(final_components["count_nll_normalized"]),
                      "conditional_nll_raw": float(final_components["conditional_nll_raw"]),
                      "diag_profiled_nll_raw": float(final_components["diag_profiled_nll_raw"]),
                      "count_nll_raw": float(final_components["count_nll_raw"]),
                      "weighted_priors": _weighted_priors(final_components),
                      "total": float(final_value)},
            "final_noninferior_to_start": noninferior,
            "note": "No reference/R2 criterion was used to choose an endpoint or checkpoint.",
        },
        "final_theta": {"path": _relative(outdir, final_theta_path),
                        "file_sha256": final_theta_file_sha, "raw_array_sha256": theta_sha,
                        "q": float(final_theta[-1]), "p": float(final_p)},
        "final_coordinates": {**coordinate_info, "path": _relative(outdir, final_coordinate_path),
                               "theta_to_coordinates_max_abs_diff": serialization_max_abs_diff},
        "full_grid_r_leq_1_proof": {
            "n_tracks": coordinate_info["n_tracks"], "n_beads": coordinate_info["n_beads"],
            "n_loci_per_copy": data.n_loci, "all_finite": bool(np.isfinite(serialized_coordinates).all()),
            "max_radius": float(norms.max()), "min_radius": float(norms.min()),
            "all_norms_strictly_less_than_one": bool(np.all(norms < 1.0)),
            "full_grid_origin_bp": 0, "bin_size_bp": 1_000_000,
        },
        "objective_components_final": _json_ready(final_components),
        "logs": {"progress": "logs/progress.jsonl", "checkpoints": "logs/checkpoints.jsonl",
                 "terminal_endpoint": "logs/terminal-endpoint.jsonl"},
        "training_boundary": {"phase_used": False, "reference_used": False,
                               "oracle_coordinates_opened": False, "softall_coordinates_opened": False,
                               "native_fdg_started": False, "fdg_or_r2_evaluation_performed": False},
    }
    _write_json(outdir / "final_components.json", {
        "components": final_components, "weighted_priors": _weighted_priors(final_components),
        "gradient_norm": final_gradient_norm, "actual_nfev": int(result.actual_nfev),
    })
    _write_json(outdir / "continuation_summary.json", audit)
    _write_json(outdir / "terminal_audit.json", audit)
    _write_json(outdir / "run_status/run_status.json", {
        "status": "complete", "updated_at_utc": _utc_now(), "candidate": "random_joint",
        "termination_status": termination["status"], "termination_reason": termination["termination_reason"],
        "final_theta": _relative(outdir, final_theta_path),
        "final_coordinates": _relative(outdir, final_coordinate_path),
    })
    _log(outdir, "finished status=%s reason=%s nit=%d actual_nfev=%d final_total=%.12g final_count=%.12g" %
         (termination["status"], termination["termination_reason"], int(result.nit),
          int(result.actual_nfev), final_value, final_components["count_nll_normalized"]))
    return audit


def run(outdir: str | os.PathLike[str]) -> dict[str, Any]:
    outdir = Path(outdir).resolve()
    if not outdir.is_dir():
        raise FileNotFoundError("fresh formal directory must exist before continuation: %s" % outdir)
    parent_config, start = _read_parent_json()
    static = _prepare(outdir, parent_config, start)
    _write_json(outdir / "run_status/run_status.json", {
        "status": "preflight", "updated_at_utc": _utc_now(), "candidate": "random_joint",
        "solver_restart": True,
    })
    _log(outdir, "formal directory=%s" % outdir)
    _log(outdir, "static config/protocol/source snapshots written before real objective load")
    _log(outdir, "start checkpoint sha256=%s selected/final sha256=%s" %
         (start["checkpoint_sha256"], start["selected_sha256"]))
    try:
        data = contact_model.load_frozen_p9016_aggregate(1_000_000)
        state = _load_start(data, start)
        _write_preflight(outdir, state, start, static)
    except Exception as exc:
        failure = {"status": "failed", "phase": "preflight", "timestamp_utc": _utc_now(),
                   "failure": "%s: %s" % (type(exc).__name__, exc),
                   "traceback": traceback.format_exc()}
        _write_json(outdir / "preflight_failure.json", failure)
        _write_json(outdir / "terminal_audit.json", failure)
        _write_json(outdir / "run_status/run_status.json", failure)
        _log(outdir, "preflight failed: %s" % failure["failure"])
        raise
    _write_json(outdir / "run_status/run_status.json", {
        "status": "training", "updated_at_utc": _utc_now(), "candidate": "random_joint",
        "preflight": "passed", "solver_restart": True,
    })
    return _run_optimizer(outdir, state, start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the authorized phase-free 020 random_joint continuation")
    parser.add_argument("--out", default=str(ROOT_PATH / OUT_RELATIVE),
                        help="fresh formal output directory created before this command")
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({
        "outdir": str(Path(args.out).resolve()),
        "status": result["status"],
        "final_theta": result["final_theta"],
        "final_coordinates": result["final_coordinates"],
        "termination": result["termination"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
