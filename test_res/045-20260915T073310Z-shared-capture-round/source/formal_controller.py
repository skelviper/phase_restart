"""S/G shared-capture formal controller。

默认只在 config 已被 preflight 标记 READY_FOR_FORMAL 时启动。controller 不导入
post evaluator、不读取 eval_truth、phase 或真实 reference；候选 endpoint 先写出并
哈希，后续 evaluator 才能按独立 gate 读取 evaluation-only 数据。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent
for path in (ROOT, SOURCE, SOURCE / "frozen_035", SOURCE / "frozen_037", SOURCE / "frozen_pr"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_io import load_aggregate, load_start, sha256_file, write_json  # noqa: E402
from m1_preconditioner import run_budgeted_lbfgs  # noqa: E402
from pr import contact_model, reconstruction_init  # noqa: E402
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective  # noqa: E402

REAL_STAGES = (
    {"stage": "5Mb", "bin_size_bp": 5_000_000, "fg_cap": 612},
    {"stage": "2Mb", "bin_size_bp": 2_000_000, "fg_cap": 404},
    {"stage": "1Mb", "bin_size_bp": 1_000_000, "fg_cap": 486},
)
SYNTHETIC_STAGE = {"stage": "1Mb", "bin_size_bp": 1_000_000, "fg_cap": 486}
CANONICAL_GTOL = 1e-6
FTOL = 0.0
MAXLS = 20
PAIR_BLOCK = 262_144
INNER_CAP = 80
CG_CAP = 80
SOURCE_TIE_TOL = 1e-9


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    write_json(path, _jsonable(value))


def _event(event: dict[str, Any]) -> None:
    path = RUN / "logs" / "formal.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(event), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _configure_init() -> None:
    root014 = ROOT / "test_res/014-20260912_153000-s0-genome-wide-fixed"
    reconstruction_init.ROOT = str(ROOT)
    reconstruction_init.DEFAULT_GATE_PATH = str(root014 / "gate.json")
    reconstruction_init.DEFAULT_COORD_DIR = str(root014 / "coords")
    for candidate, spec in list(reconstruction_init.APPROVED_SOURCES.items()):
        patched = dict(spec)
        patched["path"] = str(root014 / "coords" / (candidate + ".3dg"))
        patched["gate_path"] = str(root014 / "gate.json")
        reconstruction_init.APPROVED_SOURCES[candidate] = patched


def _load_manifest() -> dict[str, Any]:
    path = RUN / "inputs" / "formal_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "p9016-shared-capture-worker-manifest-v1":
        raise RuntimeError("formal manifest schema mismatch")
    if value.get("optimizer_started"):
        raise RuntimeError("formal manifest already marks optimizer_started")
    return value


def _resolve(relative: str | Path) -> Path:
    path = (RUN / Path(relative)).resolve()
    try:
        path.relative_to(RUN.resolve())
    except ValueError as exc:
        raise RuntimeError("manifest path escapes run directory: %s" % relative) from exc
    return path


def _load_real_data(manifest: dict[str, Any], bin_size: int):
    row = manifest["real_input_paths"][str(int(bin_size))]
    data = load_aggregate(_resolve(row))
    data.assert_consistent()
    return data


def _load_synthetic_data(row: dict[str, Any]):
    path = _resolve(row["data_path"])
    with np.load(path, allow_pickle=False) as payload:
        counts = np.asarray(payload["counts"], dtype=np.float64).copy()
        diag = np.asarray(payload["diag_counts"], dtype=np.float64).copy()
        endpoints = np.asarray(payload["endpoint_counts"], dtype=np.float64).copy()
        exposure = np.asarray(payload["exposure"], dtype=np.float64).copy()
    template = _load_real_data(_load_manifest(), 1_000_000)
    data = contact_model.synthetic_expected_clone(
        template, counts, diag, exposure,
        expected_group_totals={"diag": float(diag.sum()),
                               "cis_offdiag": float(counts[template.cis_pair].sum()),
                               "inter": float(counts[~template.cis_pair].sum())},
        exposure_mode="synthetic_known_generating_e",
        endpoint_counts=endpoints, rtol=1e-10, atol=1e-8,
    )
    return data


def _load_known_e(row: dict[str, Any]) -> np.ndarray:
    values = np.asarray(np.load(_resolve(row["known_e_path"]), allow_pickle=False), dtype=np.float64)
    if values.shape != (2645,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise RuntimeError("known-e vector failed shape/domain gate")
    if not np.isclose(float(values.mean()), 1.0, rtol=0.0, atol=1e-14):
        raise RuntimeError("known-e vector mean is not one")
    return values


def _weights(row: dict[str, Any]) -> PenaltyWeights:
    values = row.get("weights")
    if not isinstance(values, dict):
        raise RuntimeError("fit row lacks explicit penalty weights")
    result = PenaltyWeights(**{key: float(values[key]) for key in ("count", "bond", "repulsion", "bend", "p_prior")})
    result.validate()
    return result


def _objective(data: Any, model_id: str, weights: PenaltyWeights, known_e: np.ndarray | None) -> SharedCaptureObjective:
    if data.count_mode == "synthetic_expected":
        if known_e is None:
            raise RuntimeError("synthetic expected fit requires known-e")
        mode = "V0-known-generating-e"
    elif data.count_mode == "raw_integer":
        if known_e is not None:
            raise RuntimeError("real fit cannot receive known-e")
        mode = "V0-fixed-production-e"
    else:
        raise RuntimeError("unsupported fit count mode: %s" % data.count_mode)
    return SharedCaptureObjective(
        data, model_id=model_id, weights=weights, mode=mode,
        device="cuda", pair_block=PAIR_BLOCK, inner_cap=INNER_CAP,
        cg_cap=CG_CAP, profile_warm_start=True, known_e=known_e,
    )


def _read_full_tracks(path: Path, data: Any) -> np.ndarray:
    coords = np.full((2, data.n_loci, 3), np.nan, dtype=np.float64)
    by_name = {spec.name: spec for spec in data.track_specs}
    seen: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5 or fields[0] not in by_name:
                raise RuntimeError("invalid full-track row at %s:%d" % (path, line_number))
            spec = by_name[fields[0]]
            position = int(fields[1])
            local_bin = position // int(data.bin_size)
            slc = data.chromosome_slice(spec.chromosome_index)
            matches = np.flatnonzero(data.locus_bin[slc] == local_bin)
            if len(matches) != 1:
                raise RuntimeError("track row does not map to one grid locus at %s:%d" % (path, line_number))
            index = int(slc.start + matches[0])
            key = (spec.copy_index, index)
            if key in seen:
                raise RuntimeError("duplicate full-track row at %s:%d" % (path, line_number))
            seen.add(key)
            coords[spec.copy_index, index] = np.asarray(fields[2:5], dtype=np.float64)
    if len(seen) != 2 * data.n_loci or not np.all(np.isfinite(coords)):
        raise RuntimeError("full-track endpoint is incomplete: %s" % path)
    contact_model.assert_inside_unit_ball(coords)
    return coords


def _hash_array(values: np.ndarray, dtype: str = "<f8") -> str:
    array = np.asarray(values, dtype=dtype, order="C")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _checkpoint_hook(fit_id: str, stage: str):
    directory = RUN / "checkpoints" / fit_id / stage
    directory.mkdir(parents=True, exist_ok=True)

    def save(entry: dict[str, Any]) -> None:
        iteration = int(entry["iteration"])
        path = directory / ("accepted-%05d.npz" % iteration)
        np.savez_compressed(
            path, iteration=np.asarray(iteration, dtype=np.int64), nfev=np.asarray(entry["nfev"], dtype=np.int64),
            theta=np.asarray(entry["theta"], dtype=np.float64), raw_y=np.asarray(entry["raw_y"], dtype=np.float64),
            coordinates=np.asarray(entry["coordinates"], dtype=np.float64),
            optimizer_gradient=np.asarray(entry["optimizer_gradient"], dtype=np.float64),
            canonical_gradient=np.asarray(entry["canonical_gradient"], dtype=np.float64),
        )
    return save


def _classify(result: Any) -> str:
    if result.terminal_reason == "canonical_gtol" or result.canonical_gradient_max_abs <= CANONICAL_GTOL:
        return "converged"
    if result.terminal_reason in ("fg_budget_exhausted", "accepted_iteration_guard"):
        return "budget_not_converged"
    if result.terminal_reason in ("ftol_numeric_stop", "scipy_stop", "solver_reported_success", "solver_reported_nonconvergence"):
        return "not_converged"
    return "failure"


def _stage_paths(fit_id: str, stage: str) -> tuple[Path, Path, Path, Path]:
    base = RUN / "coords" / fit_id
    base.mkdir(parents=True, exist_ok=True)
    endpoint = base / (stage + ".3dg")
    endpoint_npz = base / (stage + ".npz")
    record = RUN / "stages" / fit_id / (stage + ".json")
    history = RUN / "stages" / fit_id / (stage + ".accepted_history.json")
    record.parent.mkdir(parents=True, exist_ok=True)
    return endpoint, endpoint_npz, record, history


def stage_fit(fit: dict[str, Any], stage_spec: dict[str, Any], data: Any,
              initial_coordinates: np.ndarray, p_init: float, q_init: float,
              known_e: np.ndarray | None, initial_raw_y: np.ndarray | None = None) -> dict[str, Any]:
    fit_id = str(fit["fit_id"])
    stage = str(stage_spec["stage"])
    fg_cap = int(stage_spec["fg_cap"])
    endpoint_path, endpoint_npz, record_path, history_path = _stage_paths(fit_id, stage)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    base = {
        "fit_id": fit_id, "stage": stage, "bin_size_bp": int(stage_spec.get("bin_size_bp", data.bin_size)),
        "model_id": fit["model_id"], "kind": fit["kind"], "fixture": fit.get("fixture"),
        "candidate": fit.get("candidate"), "start_name": fit.get("start_name"),
        "objective_variant": fit.get("objective_variant", "full-J"), "fg_cap": fg_cap,
        "ftol": FTOL, "canonical_gtol": CANONICAL_GTOL, "maxls": MAXLS,
        "maxiter": fg_cap + 1, "started_at_utc": started, "status": "running",
        "weights": fit.get("weights", {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}),
        "input_count_mode": data.count_mode, "input_exposure_mode": data.exposure_mode,
    }
    _write_json(record_path, base)
    accepted: list[dict[str, Any]] = []
    try:
        weights = _weights(fit)
        objective = _objective(data, str(fit["model_id"]), weights, known_e)
        coordinates = np.asarray(initial_coordinates, dtype=np.float64)
        contact_model.assert_inside_unit_ball(coordinates)
        if initial_raw_y is None:
            initial_y = objective.raw_from_physical(coordinates)
            raw_y_source = "physical_to_raw_at_stage_boundary"
        else:
            initial_y = np.asarray(initial_raw_y, dtype=np.float64).copy()
            if initial_y.shape != coordinates.shape or not np.all(np.isfinite(initial_y)):
                raise RuntimeError("serialized start raw_y shape/domain mismatch")
            mapped = objective.physical_coordinates_from_raw(initial_y)
            mapping_error = float(np.max(np.abs(mapped - coordinates)))
            if mapping_error > 1e-10:
                raise RuntimeError("serialized start raw_y mapping error %.17g exceeds tolerance" % mapping_error)
            raw_y_source = "serialized_start_raw_y"
        mapped_initial = objective.physical_coordinates_from_raw(initial_y)
        mapping_error = float(np.max(np.abs(mapped_initial - coordinates)))
        initial_theta = objective.pack(initial_y, p=float(p_init))
        initial_theta[-1] = float(q_init)
        base["initial_state_hashes"] = {
            "coordinates_sha256": _hash_array(coordinates),
            "raw_y_sha256": _hash_array(initial_y),
            "theta_sha256": _hash_array(initial_theta),
            "q_float64_sha256": _hash_array(np.asarray([q_init], dtype=np.float64)),
        }
        base["initial_p"] = float(p_init)
        base["initial_q"] = float(q_init)
        base["initial_raw_y_source"] = raw_y_source
        base["initial_raw_y_mapping_max_abs_error"] = mapping_error
        base["initial_raw_y_mapping_tolerance"] = 1e-10
        _write_json(record_path, base)

        # Frozen runner history starts at iteration 0; persist that state before optimizer entry.
        initial_value, initial_gradient, initial_components = objective.evaluate(initial_theta, need_gradient=True)
        iter0_path = RUN / "checkpoints" / fit_id / stage / "accepted-00000.npz"
        iter0_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            iter0_path, iteration=np.asarray(0, dtype=np.int64), nfev=np.asarray(0, dtype=np.int64),
            theta=np.asarray(initial_theta, dtype=np.float64), raw_y=np.asarray(initial_y, dtype=np.float64),
            coordinates=np.asarray(coordinates, dtype=np.float64),
            optimizer_gradient=np.asarray(initial_gradient, dtype=np.float64),
            canonical_gradient=np.asarray(initial_gradient, dtype=np.float64),
        )
        base["iter0_artifact"] = {
            "path": str(iter0_path.relative_to(RUN)), "sha256": sha256_file(iter0_path),
            "fun": float(initial_value), "components": initial_components,
        }
        _write_json(record_path, base)

        def accepted_callback(entry: dict[str, Any]) -> None:
            row = dict(entry)
            accepted.append(row)
            _write_json(history_path, accepted)

        fit_started = time.perf_counter()
        result = run_budgeted_lbfgs(
            objective, initial_y, p_init=float(p_init), q_init=float(q_init),
            maxfun=fg_cap, maxiter=fg_cap + 1, maxls=MAXLS, ftol=FTOL,
            canonical_gtol=CANONICAL_GTOL, checkpoint_every=10,
            checkpoint_hook=_checkpoint_hook(fit_id, stage),
            accepted_callback=accepted_callback,
        )
        # The frozen runner owns the authoritative 0..nit history; callback rows alone omit iter0.
        _write_json(history_path, result.history)
        if not result.history or int(result.history[0].get("iteration", -1)) != 0:
            raise RuntimeError("result.history does not contain iteration 0")
        fit_wall = time.perf_counter() - fit_started
        endpoint_coords = np.asarray(result.coordinates, dtype=np.float64).copy()
        contact_model.assert_inside_unit_ball(endpoint_coords)
        contact_model.write_full_tracks(endpoint_path, data, endpoint_coords)
        readback = _read_full_tracks(endpoint_path, data)
        if not np.array_equal(endpoint_coords, readback):
            raise RuntimeError("endpoint 3DG write/readback changed coordinates")
        np.savez_compressed(endpoint_npz, coordinates=endpoint_coords,
                            raw_y=np.asarray(result.y, dtype=np.float64),
                            theta=np.asarray(result.theta, dtype=np.float64),
                            p=np.asarray(result.p, dtype=np.float64),
                            q=np.asarray(result.theta[-1], dtype=np.float64))
        readback_objective = _objective(data, str(fit["model_id"]), weights, known_e)
        endpoint_raw = readback_objective.raw_from_physical(readback)
        endpoint_theta = readback_objective.pack(endpoint_raw, p=float(result.p))
        endpoint_theta[-1] = float(result.theta[-1])
        value, gradient, components = readback_objective.evaluate(endpoint_theta, need_gradient=True)
        physical_gradient = readback_objective.physical_gradient()
        if physical_gradient is None:
            raise RuntimeError("endpoint physical gradient missing")
        if abs(float(value) - float(result.fun)) > 1e-8:
            raise RuntimeError("endpoint readback total differs from hard-budget endpoint")
        p_from_q, _ = contact_model.p_from_q(float(endpoint_theta[-1]))
        if p_from_q != float(result.p):
            raise RuntimeError("endpoint q does not map bit-exactly to endpoint p")
        status = _classify(result)
        record = {
            **base, "status": status, "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fit_wall_seconds": float(fit_wall), "optimizer": result.as_dict(),
            "terminal_reason": result.terminal_reason, "outer_fg_actual": int(result.nfev),
            "last_accepted_endpoint": bool(result.endpoint_was_last_accepted),
            "endpoint": {
                "p": float(result.p), "q": float(result.theta[-1]), "coordinates_shape": list(endpoint_coords.shape),
                "total": float(value), "raw_y_q_gradient_inf": float(np.max(np.abs(gradient))),
                "physical_gradient_inf": float(np.max(np.abs(physical_gradient))),
                "canonical_gradient_max_abs": float(result.canonical_gradient_max_abs),
                "canonical_gradient_norm": float(result.canonical_gradient_norm),
                "components": components,
            },
            "history": {"path": str(history_path.relative_to(RUN)), "accepted_rows": len(accepted),
                        "history_rows": len(result.history), "first_iteration": int(result.history[0]["iteration"]),
                        "last_iteration": int(result.history[-1]["iteration"]), "records_iter0_and_accepted": True,
                        "history_source": "result.history", "iter0_artifact": str(iter0_path.relative_to(RUN)),
                        "coordinates_every_10_accepted_plus_endpoint": True},
            "artifact_hashes": {"coordinate_3dg_sha256": sha256_file(endpoint_path),
                                "coordinate_npz_sha256": sha256_file(endpoint_npz),
                                "history_sha256": sha256_file(history_path) if history_path.exists() else None,
                                 "iter0_npz_sha256": sha256_file(iter0_path)},
            "accepted_endpoint_rule": "last callback-confirmed finite accepted state; unaccepted line-search trial discarded",
        }
        _write_json(record_path, record)
        _event({"event": "stage_end", "fit_id": fit_id, "stage": stage,
                "status": status, "outer_fg_actual": int(result.nfev),
                "at_utc": record["completed_at_utc"]})
        return record
    except Exception as error:
        record = {**base, "status": "failure", "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "error_type": type(error).__name__, "error": str(error), "hard_error": True,
                  "optimizer_started": bool(accepted)}
        _write_json(record_path, record)
        _event({"event": "stage_failure", "fit_id": fit_id, "stage": stage, "error": str(error)})
        return record


def _select_real_sources(results: list[dict[str, Any]]) -> dict[str, Any]:
    selections = {}
    for model in ("S", "G"):
        rows = []
        for candidate in ("consensus", "random"):
            fit_id = "real-%s-%s" % (model, candidate)
            path = RUN / "stages" / fit_id / "1Mb.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            endpoint = record.get("endpoint", {})
            rows.append({"candidate": candidate, "status": record.get("status"),
                         "count_nll_normalized": endpoint.get("components", {}).get("count_nll_normalized"),
                         "coordinate_sha256": record.get("artifact_hashes", {}).get("coordinate_3dg_sha256")})
        valid = [row for row in rows if row["status"] in ("converged", "not_converged", "budget_not_converged")
                 and row["count_nll_normalized"] is not None]
        selected = None
        if valid:
            best = min(valid, key=lambda row: float(row["count_nll_normalized"]))
            consensus = next((row for row in valid if row["candidate"] == "consensus"), None)
            if consensus is not None and abs(float(consensus["count_nll_normalized"]) - float(best["count_nll_normalized"])) <= SOURCE_TIE_TOL:
                selected = "consensus"
            else:
                selected = best["candidate"]
        selections[model] = {"candidates": rows, "selected_by_own_count_nll": selected,
                             "tie_tolerance": SOURCE_TIE_TOL,
                             "selection_rule": "within 1e-9 consensus; otherwise minimum same-model 1Mb count NLL"}
    return selections


def _write_candidate_manifest(manifest: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for row in summaries:
        if row["kind"] == "real":
            stage = "1Mb"
        else:
            stage = "1Mb"
        record_path = RUN / "stages" / row["fit_id"] / (stage + ".json")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") not in ("converged", "not_converged", "budget_not_converged"):
            continue
        endpoint_path = RUN / "coords" / row["fit_id"] / (stage + ".npz")
        with np.load(endpoint_path, allow_pickle=False) as payload:
            coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
            p = float(np.asarray(payload["p"]).item())
        candidates.append({
            "candidate_id": row["fit_id"], "fit_id": row["fit_id"], "kind": row["kind"],
            "fixture": row.get("fixture"), "model_id": row["model_id"], "candidate": row.get("candidate"),
            "start_name": row.get("start_name"), "objective_variant": row.get("objective_variant"),
            "coordinate_path": str(endpoint_path.relative_to(RUN)),
            "coordinate_sha256": sha256_file(endpoint_path),
            "coordinate_array_sha256": _hash_array(coordinates), "coordinate_shape": list(coordinates.shape),
            "p": p, "stage_status": record.get("status"),
        })
    output = {"schema": "p9016-shared-capture-candidate-manifest-v1", "run_id": RUN.name,
              "candidate_hash_gate": "all listed endpoints have complete endpoint hashes",
              "candidates": candidates}
    _write_json(RUN / "results" / "candidate_manifest.json", output)
    return output


def _verify_immutable_inputs(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Refuse optimizer entry if any prepared input/source/protocol byte changed."""
    frozen = config.get("immutable_hashes")
    if not isinstance(frozen, dict) or not frozen:
        raise RuntimeError("config lacks immutable_hashes; optimizer entry is refused")
    required: dict[str, Path] = {
        "inputs/formal_manifest.json": RUN / "inputs/formal_manifest.json",
        "inputs/source_input_manifest.json": RUN / "inputs/source_input_manifest.json",
        "inputs/evaluation_manifest.json": RUN / "inputs/evaluation_manifest.json",
    }
    for key, value in manifest.get("real_input_paths", {}).items():
        path = Path(str(value))
        required["inputs/" + path.name] = path.resolve()
    for fit in manifest.get("matrix", []):
        for key in ("data_path", "known_e_path", "start_path", "start_5Mb"):
            if key not in fit:
                continue
            path = Path(str(fit[key]))
            required[str(path.relative_to(RUN)) if path.is_absolute() and path.is_relative_to(RUN) else str(path)] = path.resolve() if path.is_absolute() else (RUN / path).resolve()
    for key in manifest.get("source_hashes", {}):
        source_path = (RUN / str(key)).resolve()
        if not source_path.is_file():
            source_path = (ROOT / str(key)).resolve()
        if not source_path.is_file():
            source_path = (ROOT / "docs/audits/next-step-r2-preparation-20260914T143656Z/synthetic_inputs" / str(key)).resolve()
        required[str(key)] = source_path
    # The config's immutable protocol map must cover every file the worker can read.
    missing = sorted(set(required) - set(frozen))
    if missing:
        raise RuntimeError("immutable hash map missing %d required paths: %s" % (len(missing), missing[:5]))
    mismatches = []
    paths_to_check: dict[str, Path] = dict(required)
    for key in frozen:
        if key in paths_to_check:
            continue
        run_path = (RUN / key).resolve()
        root_path = (ROOT / key).resolve()
        if run_path.is_file():
            paths_to_check[key] = run_path
        elif root_path.is_file():
            paths_to_check[key] = root_path
        else:
            known_path = (ROOT / "docs/audits/next-step-r2-preparation-20260914T143656Z/synthetic_inputs" / key).resolve()
            if known_path.is_file():
                paths_to_check[key] = known_path
            else:
                raise RuntimeError("immutable hash map points to missing path: %s" % key)
    for key, path in paths_to_check.items():
        expected = str(frozen[key])
        actual = sha256_file(path)
        if actual != expected:
            mismatches.append({"path": key, "expected": expected, "actual": actual})
    if mismatches:
        raise RuntimeError("immutable input/source hash gate failed: %s" % mismatches[:3])
    for key, expected in manifest.get("source_hashes", {}).items():
        if str(frozen.get(key)) != str(expected):
            raise RuntimeError("formal manifest source hash disagrees with config immutable hash: %s" % key)
    protocol = config.get("immutable_protocol", {})
    expected_map_hash = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode("utf-8")).hexdigest()
    if protocol.get("hash_map_sha256") != expected_map_hash:
        raise RuntimeError("immutable protocol hash-map digest mismatch")
    if protocol.get("formal_manifest_sha256") != frozen.get("inputs/formal_manifest.json"):
        raise RuntimeError("immutable protocol formal manifest digest mismatch")
    return {"status": "PASS", "checked": len(paths_to_check), "paths": sorted(paths_to_check)}


def run_formal() -> int:
    config_path = RUN / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "READY_FOR_FORMAL":
        raise RuntimeError("formal run requires config status READY_FOR_FORMAL")
    if config.get("authorization", {}).get("formal_started"):
        raise RuntimeError("formal run already started; refuse silent resume")
    manifest = _load_manifest()
    if int(manifest.get("expected_fits")) != 14 or int(manifest.get("expected_stages")) != 22:
        raise RuntimeError("formal manifest cardinality mismatch")
    immutable_gate = _verify_immutable_inputs(config, manifest)
    config["immutable_hash_gate"] = immutable_gate
    config["status"] = "formal_running"
    config.setdefault("authorization", {})["formal_started"] = True
    config["started_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_json(config_path, config)
    _event({"event": "formal_start", "fits": 14, "stages": 22, "expected_fg": 10_868,
            "at_utc": config["started_at_utc"]})
    _configure_init()
    summaries: list[dict[str, Any]] = []
    actual_fg = 0
    for fit in manifest["matrix"]:
        fit_id = str(fit["fit_id"])
        _event({"event": "fit_start", "fit_id": fit_id, "kind": fit["kind"]})
        fit_rows = []
        blocked = False
        previous = None
        if fit["kind"] == "real":
            start_path = _resolve(fit["start_5Mb"])
            coordinates, raw_y, p_init, _meta = load_start(start_path)
            q_init = contact_model.q_from_p(float(p_init))
        else:
            start_path = _resolve(fit["start_path"])
            coordinates, raw_y, p_init, _meta = load_start(start_path)
            q_init = contact_model.q_from_p(float(p_init))
        known_e = _load_known_e(fit) if fit["kind"] == "synthetic" else None
        for stage_index, stage_spec in enumerate(fit.get("stages", [fit.get("stage")])):
            if blocked:
                row = {"fit_id": fit_id, "stage": stage_spec["stage"], "status": "blocked_by_previous_failure"}
                _write_json(RUN / "stages" / fit_id / (stage_spec["stage"] + ".json"), row)
                fit_rows.append(row)
                continue
            bin_size = int(stage_spec["bin_size_bp"])
            if fit["kind"] == "real":
                data = _load_real_data(manifest, bin_size)
                if stage_index > 0:
                    previous_data = _load_real_data(manifest, int(fit["stages"][stage_index - 1]["bin_size_bp"]))
                    warm = reconstruction_init.warm_start_from_layer(
                        previous["coordinates"], previous["positions"], previous["chromosome_index"],
                        tuple(data.chromosome_names), tuple(int(v) for v in data.chromosome_lengths), bin_size,
                        1103 if fit["candidate"] == "consensus" else 2207,
                    )
                    coordinates = np.asarray(warm["coords"], dtype=np.float64)
                    raw_y = None
                    _event({"event": "prolongation", "fit_id": fit_id, "from_stage": fit["stages"][stage_index - 1]["stage"],
                            "to_stage": stage_spec["stage"], "q_carry": q_init})
            else:
                data = _load_synthetic_data(fit)
            row = stage_fit(fit, stage_spec, data, coordinates, float(p_init), float(q_init), known_e, initial_raw_y=raw_y)
            fit_rows.append(row)
            if row.get("status") == "failure":
                blocked = True
            else:
                actual_fg += int(row.get("outer_fg_actual", 0))
                endpoint_npz = RUN / "coords" / fit_id / (stage_spec["stage"] + ".npz")
                with np.load(endpoint_npz, allow_pickle=False) as payload:
                    coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
                raw_y = None
                p_init = float(row["endpoint"]["p"])
                q_init = float(row["endpoint"]["q"])
                previous = {"coordinates": coordinates,
                            "positions": (data.locus_bin * int(data.bin_size)).copy(),
                            "chromosome_index": np.asarray(data.locus_chromosome, dtype=np.int32).copy()}
        fit_status = "completed" if not blocked else "failure"
        summary = {"fit_id": fit_id, "kind": fit["kind"], "model_id": fit["model_id"],
                   "candidate": fit.get("candidate"), "fixture": fit.get("fixture"),
                   "status": fit_status, "stage_status": [row.get("status") for row in fit_rows],
                   "outer_fg_actual": sum(int(row.get("outer_fg_actual", 0)) for row in fit_rows)}
        summaries.append(summary)
        _write_json(RUN / "stages" / fit_id / "fit_summary.json", summary)
        _event({"event": "fit_end", **summary})
    candidate_manifest = _write_candidate_manifest(manifest, summaries)
    selections = _select_real_sources(summaries) if all(
        (RUN / "stages" / fit_id / "1Mb.json").exists()
        for fit_id in ("real-S-consensus", "real-S-random", "real-G-consensus", "real-G-random")
    ) else {}
    _write_json(RUN / "results" / "real_source_selection.json", {
        "schema": "p9016-shared-capture-real-source-selection-v1", "models": selections,
        "reference_used": False, "evaluation_used": False,
    })
    final_status = "formal_complete" if all(row["status"] == "completed" for row in summaries) else "formal_complete_with_failures"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"status": final_status, "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "actual_outer_fg": int(actual_fg), "fit_summaries": summaries,
                   "candidate_count": len(candidate_manifest["candidates"]),
                   "real_source_selection": "results/real_source_selection.json"})
    _write_json(config_path, config)
    _event({"event": "formal_end", "status": final_status, "actual_outer_fg": actual_fg,
            "at_utc": config["completed_at_utc"]})
    return 0 if final_status == "formal_complete" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S/G shared-capture formal fits")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("command", choices=("run",))
    args = parser.parse_args()
    global RUN
    if args.run_dir is not None:
        RUN = Path(args.run_dir).resolve()
    status = run_formal()
    print(json.dumps({"status": "complete" if status == 0 else "complete_with_failures", "exit_code": status}, indent=2))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
