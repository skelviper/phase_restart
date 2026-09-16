#!/usr/bin/env python
"""P9016 real-cell final evaluator.

This module is deliberately independent of the abandoned ``evaluation/`` writer.
It reads the frozen 046 manifests, writes only under ``evaluation_final/``, and
opens the phase-bearing reference only after the candidate/control/null hash gate.
No fitting, selection, stopping, or coordinate mutation is performed here.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import rankdata

RUN = Path(__file__).resolve().parent.parent
OUT = RUN / "evaluation_final"
RESULTS = OUT / "results"
SOURCE_RUN = RUN.parent / "045-20260915T073310Z-shared-capture-round"
SOURCE = SOURCE_RUN / "source"
ROOT = RUN.parents[1]
for _path in (ROOT, SOURCE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# These are the validated 045 data/objective helpers.  They are imported only
# for loading frozen aggregates and value-only scoring; no optimizer is called.
from data_io import load_aggregate  # noqa: E402
from formal_controller import _objective as frozen_objective  # noqa: E402
from shared_capture_objective import PenaltyWeights  # noqa: E402
from pr import contact_model, r2comparison  # noqa: E402
from pr.score import spearman  # noqa: E402

REFERENCE_PATH = ROOT / "data/P9016.1m.3dg.gz"
REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
RAW_PAIRS_PATH = ROOT / "inputs/P9016.snpfree.pairs.gz"
RAW_PAIRS_SHA256 = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"
MASK_LOCK_PATH = ROOT / "docs/audits/multires-r2-preparation-20260914_041826/mask_lock.json"
MASK_MANIFEST_PATH = ROOT / "test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/evaluation_manifest.json"
MASK_LOCK_SHA256 = "d0c325dea1289374152a327557605f8bbd52267b025cd7b57689be3e3666c49e"
MASK_MANIFEST_SHA256 = "fa1b26c834c173604c21f954d494cece8e053dd970e7cb6111f561b349574acb"
OLD036_ANCHOR_PATH = ROOT / "test_res/036-20260914T064651Z-gpu-multires/coords/C0/random_joint/final-1m.3dg"
OLD036_ANCHOR_SHA256 = "18c06860f78525e55045e911db05981dd431a86c3bf9ee3ebb5046845b0ccc55"
OLD038_TSV = ROOT / "test_res/038-20260914T143812Z-gpu-m1-formal/evaluation-r2-20260914T153134Z/r2_real_historical_036_C0_anchor_x20chr.tsv"
OLD038_TSV_SHA256 = "d44bfe73155edea0f6bb107ed0e41488428a03b230648e381c80066ed4651e21"
OLD036_CHECKPOINT = ROOT / "test_res/036-20260914T064651Z-gpu-multires/checkpoints/C0/random_joint/1m-accepted-0010.npz"
OLD036_CHECKPOINT_SHA256 = "d9acc41f94c7295425fb44714db3eb13fb7be80d064ebc7c06e6ef546e1289a8"
MAP_REGRESSION_METRICS = ROOT / "test_res/043-20260915T053019Z-1mb-map-contact-rg/metrics.tsv"

BIN_SIZE_BP = 1_000_000
MASK_OFFSET_BP = 3_000_000
MIN_COMMON_PAIRS = 20
GEOMETRY_TIE_TOL = 1e-12
BOOTSTRAP_SEED = 450301
BOOTSTRAP_DRAWS = 10_000
RANDOM_U_SEEDS = tuple(range(450500, 450516))
FULL_J_WEIGHTS = {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}
MAP_EPSILON = 1e-6
MAP_BLOCK = 131_072
MAP_REGRESSION_ATOL = 1e-12
STAGE_BIN_SIZES = {"5Mb": 5_000_000, "2Mb": 2_000_000, "1Mb": 1_000_000}
FROZEN_R2_PYC = ROOT / "docs/audits/visibility-r2-preparation-20260914T164603Z/frozen_pr/r2comparison.pyc"
FROZEN_R2_PYC_SHA256 = "a76d71b75df123348edff9690d58469946c1e63cedf98bcc79330732046525a9"
LIVE_R2_EXPECTED_SHA256 = "2979118aed25ac571d8f476eec4f6ec7687649f5a3e0fc43618194a293b4ed6e"
FROZEN_MASK_WORKER = OUT / "frozen_mask_worker.py"
FROZEN_MASK_WORKER_PYTHON = Path("/mnt/ssd/zliu/miniforge3/bin/python3.13")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_array(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(RUN.resolve()))
    except ValueError:
        return str(path.resolve())


def _update_state(**updates: Any) -> None:
    state_path = OUT / "state.json"
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    state.setdefault("schema", "p9016-real-evaluation-final-state-v1")
    state.setdefault("phase_opened", False)
    state.setdefault("reference_opened", False)
    state.update(_jsonable(updates))
    state["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_json(state_path, state)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON object required: %s" % path)
    return value


def _metric(metric: str, x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < MIN_COMMON_PAIRS or len(x) != len(y) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    if metric == "pearson":
        xc, yc = x - x.mean(), y - y.mean()
        denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
        return float(np.dot(xc, yc) / denom) if denom else float("nan")
    value = float(spearman(x, y))
    return value if math.isfinite(value) else float("nan")


def _distance(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    delta = np.asarray(points[pair_i] - points[pair_j], dtype=np.float64)
    return np.sqrt(np.sum(delta * delta, axis=1))


def _four_distances(coords: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    return np.stack((
        _distance(coords[0], pair_i, pair_j),
        np.sqrt(np.sum((coords[0, pair_i] - coords[1, pair_j]) ** 2, axis=1)),
        np.sqrt(np.sum((coords[1, pair_i] - coords[0, pair_j]) ** 2, axis=1)),
        _distance(coords[1], pair_i, pair_j),
    ), axis=0)


def _load_tracks(path: Path, format_name: str, chromosome_names: tuple[str, ...]) -> dict[str, dict[int, np.ndarray]]:
    """Parse frozen 3DG/native TSV while preserving nonfinite rows as NaN."""
    tracks: dict[str, dict[int, np.ndarray]] = {}
    if format_name in ("native_tsv", "coords_tsv"):
        index = {name: i for i, name in enumerate(chromosome_names)}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"chr", "copy", "start", "x", "y", "z"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise RuntimeError("native_tsv missing required fields: %s" % sorted(required))
            for line_no, row in enumerate(reader, start=2):
                try:
                    chromosome = str(row["chr"])
                    ci = index[chromosome]
                    copy = int(row["copy"])
                    position = int(row["start"])
                    point = np.asarray([float(row[a]) for a in ("x", "y", "z")], dtype=np.float64)
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError("invalid native TSV row %d in %s" % (line_no, path)) from exc
                if copy not in (0, 1) or position < 0 or position % BIN_SIZE_BP != 0:
                    raise RuntimeError("invalid native TSV chromosome/copy/start row %d" % line_no)
                if not np.isfinite(point).all():
                    point = np.full(3, np.nan, dtype=np.float64)
                track = "c%02d%s" % (ci + 1, "ab"[copy])
                target = tracks.setdefault(track, {})
                if position in target:
                    raise RuntimeError("duplicate coordinate %s:%d" % (track, position))
                target[position] = point
        return tracks
    if format_name not in ("3dg", "3dg_text"):
        raise RuntimeError("unsupported coordinate format %s" % format_name)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                track = str(fields[0])
                position = int(fields[1])
                point = np.asarray([float(item) for item in fields[2:5]], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("invalid 3DG row %d in %s" % (line_no, path)) from exc
            if not np.isfinite(point).all():
                point = np.full(3, np.nan, dtype=np.float64)
            target = tracks.setdefault(track, {})
            if position in target:
                raise RuntimeError("duplicate coordinate %s:%d" % (track, position))
            target[position] = point
    return tracks


def _dense_points(structures: Mapping[str, Mapping[int, np.ndarray]], track: str, positions: np.ndarray) -> np.ndarray:
    points = np.full((len(positions), 3), np.nan, dtype=np.float64)
    rows = structures.get(track, {})
    for index, position in enumerate(positions):
        value = rows.get(int(position))
        if value is not None:
            point = np.asarray(value, dtype=np.float64)
            if point.shape == (3,) and np.isfinite(point).all():
                points[index] = point
    return points


def _load_data() -> Any:
    path = SOURCE_RUN / "inputs/real_1000000_aggregate.npz"
    data = load_aggregate(path)
    data.assert_consistent()
    audit = data.budget()
    expected = {
        "raw_records": 1_703_888,
        "raw_same_bin": 438_774,
        "raw_cis_offdiag": 696_680,
        "raw_inter": 568_434,
        "n_loci": 2645,
        "n_chromosomes": 20,
    }
    for key, value in expected.items():
        if int(audit[key]) != value:
            raise RuntimeError("frozen 1Mb aggregate mismatch %s=%s expected=%s" % (key, audit[key], value))
    if int(audit["aggregate_cis_offdiag"]) != expected["raw_cis_offdiag"] or int(audit["aggregate_inter"]) != expected["raw_inter"]:
        raise RuntimeError("1Mb aggregate count budget mismatch")
    return data


def _load_candidate_npz(path: Path, expected_shape: tuple[int, int, int]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"coordinates", "raw_y", "theta", "p", "q"}
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError("candidate NPZ missing fields %s: %s" % (sorted(missing), path))
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        raw_y = np.asarray(payload["raw_y"], dtype=np.float64).copy()
        theta = np.asarray(payload["theta"], dtype=np.float64).copy()
        p = float(np.asarray(payload["p"]).item())
        q = float(np.asarray(payload["q"]).item())
    if coordinates.shape != expected_shape or raw_y.shape != expected_shape or theta.shape != (6 * expected_shape[1] + 1,):
        raise RuntimeError("candidate payload shape mismatch: %s" % path)
    if not all(np.isfinite(a).all() for a in (coordinates, raw_y, theta, np.asarray([p, q]))):
        raise RuntimeError("candidate payload nonfinite: %s" % path)
    contact_model.assert_inside_unit_ball(coordinates)
    return {"coordinates": coordinates, "raw_y": raw_y, "theta": theta, "p": p, "q": q}


def _load_candidates(data: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    combined = _read_json(RUN / "combined_base_manifest.json")
    extension = _read_json(RUN / "extension_manifest.json")
    candidate_manifest = _read_json(RUN / "results/candidate_manifest.json")
    if combined.get("schema") != "p9016-real-combined-base-manifest-v1" or int(combined.get("endpoint_count", -1)) != 4:
        raise RuntimeError("combined base manifest schema/count failure")
    if extension.get("schema") != "p9016-real-extension-manifest-v1" or int(extension.get("formal_totals", {}).get("extension_fits", -1)) != 4:
        raise RuntimeError("extension manifest schema/count failure")
    if candidate_manifest.get("schema") != "p9016-real-extension-candidate-manifest-v1" or len(candidate_manifest.get("candidates", [])) != 4:
        raise RuntimeError("candidate manifest schema/count failure")
    expected_shape = (2, int(data.n_loci), 3)
    candidates: list[dict[str, Any]] = []
    for fit_id, item in combined["endpoints"].items():
        path = (RUN / str(item["endpoint_npz"])).resolve()
        if _sha(path) != str(item["endpoint_npz_sha256"]):
            raise RuntimeError("base endpoint hash mismatch: %s" % path)
        payload = _load_candidate_npz(path, expected_shape)
        array_sha = _hash_array(payload["coordinates"])
        candidates.append({
            "candidate_id": str(fit_id), "fit_id": str(fit_id), "kind": "real_base",
            "model_id": str(item["model_id"]), "candidate": str(item["candidate"]),
            "variant": "base-full-J", "stage_status": str(item.get("status")),
            "path": path, "file_sha256": _sha(path), "coordinate_array_sha256": array_sha,
            **payload,
        })
    for item in candidate_manifest["candidates"]:
        path = (RUN / str(item["coordinate_path"])).resolve()
        if _sha(path) != str(item["coordinate_sha256"]):
            raise RuntimeError("extension endpoint hash mismatch: %s" % path)
        payload = _load_candidate_npz(path, expected_shape)
        array_sha = _hash_array(payload["coordinates"])
        if array_sha != str(item["coordinate_array_sha256"]):
            raise RuntimeError("extension coordinate array hash mismatch: %s" % path)
        candidates.append({
            "candidate_id": str(item["candidate_id"]), "fit_id": str(item["candidate_id"]),
            "kind": "real_extension", "model_id": str(item["model_id"]),
            "candidate": str(item.get("candidate", "random")), "variant": str(item["variant"]),
            "stage_status": str(item.get("stage_status")), "path": path,
            "file_sha256": _sha(path), "coordinate_array_sha256": array_sha,
            **payload,
        })
    if len(candidates) != 8 or len({row["candidate_id"] for row in candidates}) != 8:
        raise RuntimeError("exactly eight unique real candidates are required")
    for row in candidates:
        if row.get("stage_status") != "budget_not_converged":
            raise RuntimeError("candidate status changed from budget_not_converged: %s" % row["candidate_id"])
    candidates.sort(key=lambda row: row["candidate_id"])
    formal = _read_json(SOURCE_RUN / "inputs/formal_manifest.json")
    initials: list[dict[str, Any]] = []
    for name in ("consensus", "random"):
        spec = formal["real_initial_paths"][name]["stages"]["1Mb"]
        path = Path(str(spec["path"])).resolve()
        if _sha(path) != str(spec["sha256"]):
            raise RuntimeError("initial endpoint hash mismatch: %s" % path)
        with np.load(path, allow_pickle=False) as payload:
            coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
            raw_y = np.asarray(payload["raw_y"], dtype=np.float64).copy()
            p = float(np.asarray(payload["p_init"]).item())
        if coordinates.shape != expected_shape or raw_y.shape != expected_shape or not np.isfinite(coordinates).all():
            raise RuntimeError("initial endpoint shape/domain failure: %s" % path)
        contact_model.assert_inside_unit_ball(coordinates)
        initials.append({
            "candidate_id": "initial-%s" % name, "fit_id": "initial-%s" % name,
            "kind": "real_initial", "model_id": "initial", "candidate": name,
            "variant": "initial-zero-optimization", "path": path, "file_sha256": _sha(path),
            "coordinate_array_sha256": _hash_array(coordinates), "coordinates": coordinates,
            "raw_y": raw_y, "p": p, "q": float(contact_model.q_from_p(p)),
        })
    return candidates, initials, {"combined": combined, "extension": extension, "candidate_manifest": candidate_manifest, "formal": formal}


def _validate_legacy_evidence() -> dict[str, Any]:
    terminal = _read_json(RUN / "evaluation_terminal.json")
    launch = _read_json(RUN / "evaluation_launch.json")
    stderr_path = Path(str(terminal.get("stderr", "")))
    stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.is_file() else ""
    passed = (
        terminal.get("status") == "terminal" and int(terminal.get("return_code", -999)) == 1
        and launch.get("status") != "running" and "NameError: name '_make_u_zero' is not defined" in stderr
    )
    return {
        "status": "PASS" if passed else "FAIL", "old_evaluation_terminal": terminal,
        "old_evaluation_launch": launch, "old_stderr_path": str(stderr_path),
        "old_stderr_contains_first_null_failure": "NameError: name '_make_u_zero' is not defined" in stderr,
        "old_writer_reused": False, "new_writer": _rel(OUT / "run_evaluation.py"),
    }


def validate() -> dict[str, Any]:
    """Read-only preflight; no reference, phase, null, or fitting access."""
    OUT.mkdir(parents=True, exist_ok=True)
    _update_state(phase_opened=False, reference_opened=False, command="validate")
    data = _load_data()
    candidates, initials, manifests = _load_candidates(data)
    combined = manifests["combined"]
    extension = manifests["extension"]
    trajectory = _read_json(RUN / "trajectory_inputs/manifest.json")
    base_terminal = _read_json(RUN / "base_remaining/wrapper_terminal.json")
    extension_terminal = _read_json(RUN / "extension_terminal.json")
    evidence = _validate_legacy_evidence()
    result = {
        "schema": "p9016-real-evaluation-final-validation-v1", "status": "PASS",
        "reference_opened": False, "phase_opened": False, "synthetic_count": 0,
        "data_budget": data.budget(), "raw_pairs_sha256": _sha(RAW_PAIRS_PATH),
        "raw_pairs_sha_expected": RAW_PAIRS_SHA256,
        "candidate_count": len(candidates), "initial_count": len(initials),
        "candidate_ids": [row["candidate_id"] for row in candidates],
        "candidate_statuses": {row["candidate_id"]: row["stage_status"] for row in candidates},
        "formal_totals": extension.get("formal_totals"),
        "combined_stage_count": int(combined.get("stage_count", -1)),
        "trajectory_manifest": {
            "sha256": _sha(RUN / "trajectory_inputs/manifest.json"),
            "verified_checkpoints": int(sum(int(item.get("verified_checkpoint_count", 0)) for item in trajectory.get("stages", []))),
            "missing_or_unproven": int(sum(len(item.get("missing_or_unproven", [])) for item in trajectory.get("stages", []))),
        },
        "base_remaining_terminal": {"return_code": base_terminal.get("return_code"), "status": base_terminal.get("status")},
        "extension_terminal": {"return_code": extension_terminal.get("return_code"), "status": extension_terminal.get("status")},
        "legacy_attempt_evidence": evidence,
        "checks": {
            "raw_pairs_hash": _sha(RAW_PAIRS_PATH) == RAW_PAIRS_SHA256,
            "eight_real_candidates": len(candidates) == 8,
            "two_real_initial_controls": len(initials) == 2,
            "all_budget_not_converged": all(row["stage_status"] == "budget_not_converged" for row in candidates),
            "formal_total_7952": extension.get("formal_totals", {}).get("outer_fg") == 7952,
            "base_remaining_return_code_0": base_terminal.get("return_code") == 0,
            "extension_return_code_0": extension_terminal.get("return_code") == 0,
            "legacy_attempt_terminal_return_code_1": evidence["status"] == "PASS",
            "frozen_trajectory_counts": int(sum(int(item.get("verified_checkpoint_count", 0)) for item in trajectory.get("stages", []))) == 404 and int(sum(len(item.get("missing_or_unproven", [])) for item in trajectory.get("stages", []))) == 3588,
        },
    }
    if not all(result["checks"].values()):
        result["status"] = "FAIL"
    _write_json(RESULTS / "validation.json", result)
    return result


def _make_u_zero(coords: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    z = (coords[0] + coords[1]) / 2.0
    raw = np.stack((z, z), axis=0)
    pre = float(np.linalg.norm(raw.reshape(-1, 3), axis=1).max())
    scale = 1.0 if pre < 1.0 else (1.0 - 1e-6) / pre
    result = raw * scale
    contact_model.assert_inside_unit_ball(result)
    return result, {"global_scale": scale, "pre_scale_radius": pre, "post_scale_radius": float(np.linalg.norm(result.reshape(-1, 3), axis=1).max()), "permutation": "u_zero", "z_fixed": True}


def _make_random_u(coords: np.ndarray, data: Any, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    z = (coords[0] + coords[1]) / 2.0
    u = (coords[0] - coords[1]) / 2.0
    permuted = np.zeros_like(u)
    rng = np.random.default_rng(seed)
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        permuted[slc] = u[slc][rng.permutation(slc.stop - slc.start)]
    raw = np.stack((z + permuted, z - permuted), axis=0)
    pre = float(np.linalg.norm(raw.reshape(-1, 3), axis=1).max())
    scale = 1.0 if pre < 1.0 else (1.0 - 1e-6) / pre
    result = raw * scale
    contact_model.assert_inside_unit_ball(result)
    return result, {"global_scale": scale, "pre_scale_radius": pre, "post_scale_radius": float(np.linalg.norm(result.reshape(-1, 3), axis=1).max()), "permutation": "within_chromosome_u", "z_fixed": True}


def _write_nulls(sources: list[dict[str, Any]], data: Any) -> list[dict[str, Any]]:
    null_dir = OUT / "nulls"
    null_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source in sources:
        variants: list[tuple[str, int | None]] = [("u_zero", None)] + [("random_u", seed) for seed in RANDOM_U_SEEDS]
        for null_kind, seed in variants:
            coords, audit = _make_u_zero(source["coordinates"]) if seed is None else _make_random_u(source["coordinates"], data, seed)
            suffix = "u-zero" if seed is None else "random-u-%d" % seed
            path = null_dir / (source["candidate_id"] + "__" + suffix + ".npz")
            np.savez_compressed(path, coordinates=coords)
            with np.load(path, allow_pickle=False) as payload:
                readback = np.asarray(payload["coordinates"], dtype=np.float64).copy()
            if readback.shape != coords.shape or not np.array_equal(readback, coords) or not np.isfinite(readback).all():
                raise RuntimeError("null readback failure: %s" % path)
            records.append({
                "source_candidate_id": source["candidate_id"], "null_kind": null_kind, "seed": seed,
                "path": _rel(path), "sha256": _sha(path), "coordinate_array_sha256": _hash_array(coords), **audit,
            })
    if len(records) != 170:
        raise RuntimeError("null count must equal 170, got %d" % len(records))
    return records


def _hash_inputs_and_code() -> dict[str, Any]:
    code_paths = [
        OUT / "evaluator.py", OUT / "run_evaluation.py", OUT / "dependency_audit.py", OUT / "frozen_mask_worker.py",
        SOURCE / "data_io.py", SOURCE / "formal_controller.py", SOURCE / "shared_capture_objective.py",
        SOURCE / "visibility_profile_base.py", ROOT / "pr/contact_model.py", ROOT / "pr/r2comparison.py",
        ROOT / "pr/score.py", ROOT / "docs/audits/visibility-r2-preparation-20260914T164603Z/frozen_pr/allele_r2.py",
        FROZEN_R2_PYC,
    ]
    input_paths = [
        RUN / "combined_base_manifest.json", RUN / "extension_manifest.json", RUN / "results/candidate_manifest.json",
        SOURCE_RUN / "inputs/formal_manifest.json", SOURCE_RUN / "inputs/source_input_manifest.json",
        RUN / "trajectory_inputs/manifest.json", RUN / "scope_manifest.json", RUN / "reused_base/snapshot_manifest.json",
        RUN / "base_remaining/wrapper_terminal.json", RUN / "extension_terminal.json", RUN / "REAL_PLAN.md", RUN / "README.md",
        SOURCE_RUN / "inputs/real_1000000_aggregate.npz", SOURCE_RUN / "inputs/real_2000000_aggregate.npz", SOURCE_RUN / "inputs/real_5000000_aggregate.npz",
        RAW_PAIRS_PATH, MASK_LOCK_PATH, MASK_MANIFEST_PATH, OLD036_ANCHOR_PATH, OLD038_TSV, OLD036_CHECKPOINT, MAP_REGRESSION_METRICS,
    ]
    code = {str(path): _sha(path) for path in code_paths}
    inputs = {str(path): _sha(path) for path in input_paths}
    if inputs[str(RAW_PAIRS_PATH)] != RAW_PAIRS_SHA256:
        raise RuntimeError("raw pairs hash gate mismatch")
    if inputs[str(MASK_LOCK_PATH)] != MASK_LOCK_SHA256 or inputs[str(MASK_MANIFEST_PATH)] != MASK_MANIFEST_SHA256:
        raise RuntimeError("old21 mask input hash gate mismatch")
    if code[str(FROZEN_R2_PYC)] != FROZEN_R2_PYC_SHA256:
        raise RuntimeError("frozen legacy r2comparison.pyc hash gate mismatch")
    result = {
        "schema": "p9016-real-evaluation-final-input-code-hashes-v1", "phase_opened": False,
        "reference_opened": False, "code": code, "inputs": inputs,
        "reference": {"path": str(REFERENCE_PATH), "sha256_expected": REFERENCE_SHA256, "hash_deferred_until_gate": True},
    }
    _write_json(OUT / "input_code_hashes.json", result)
    return result


def _dependency_provenance() -> dict[str, Any]:
    live_modules = []
    for module in (r2comparison, contact_model):
        path = Path(str(module.__file__)).resolve()
        live_modules.append({"module": getattr(module, "__name__", str(module)), "module_file": str(path), "sha256": _sha(path)})
    score_module = importlib.import_module("pr.score")
    score_path = Path(str(score_module.__file__)).resolve()
    live_modules.append({"module": score_module.__name__, "module_file": str(score_path), "sha256": _sha(score_path)})
    result = {
        "schema": "p9016-real-evaluation-final-dependency-provenance-v1",
        "live_imports": live_modules,
        "frozen_legacy_builder": {"module": "evaluation_frozen_worker_pr.r2comparison", "module_file": str(FROZEN_R2_PYC), "sha256": _sha(FROZEN_R2_PYC), "expected_sha256": FROZEN_R2_PYC_SHA256, "selected": True, "loaded_by": "frozen_mask_worker under CPython 3.13"},
        "live_builder_expected_sha256": LIVE_R2_EXPECTED_SHA256,
        "live_builder_used_for_legacy_parity": False,
        "value_only_objective_sources": [str(SOURCE / "formal_controller.py"), str(SOURCE / "shared_capture_objective.py"), str(SOURCE / "visibility_profile_base.py")],
    }
    _write_json(RESULTS / "dependency_provenance.json", result)
    return result


def _pre_reference_gate(candidates: list[dict[str, Any]], initials: list[dict[str, Any]], data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = candidates + initials
    null_records = _write_nulls(sources, data)
    hashes = _hash_inputs_and_code()
    records: list[dict[str, Any]] = []
    for source in sources:
        if _sha(source["path"]) != source["file_sha256"]:
            raise RuntimeError("source file changed during pre-reference gate: %s" % source["path"])
        records.append({"candidate_id": source["candidate_id"], "kind": source["kind"], "path": _rel(source["path"]), "sha256": source["file_sha256"], "coordinate_array_sha256": source["coordinate_array_sha256"]})
    for record in null_records:
        path = (RUN / record["path"]).resolve()
        if _sha(path) != record["sha256"]:
            raise RuntimeError("null file changed during pre-reference gate: %s" % path)
        records.append({key: record[key] for key in ("source_candidate_id", "null_kind", "seed", "path", "sha256", "coordinate_array_sha256")})
    gate = {
        "schema": "p9016-real-evaluation-final-pre-reference-hash-gate-v1", "status": "PASS",
        "reference_opened": False, "phase_opened": False, "synthetic_count": 0,
        "candidate_count": len(candidates), "initial_count": len(initials), "null_count": len(null_records),
        "expected_null_count": 170, "records": records,
        "input_code_hashes_path": _rel(OUT / "input_code_hashes.json"),
        "input_code_hashes_sha256": _sha(OUT / "input_code_hashes.json"),
        "input_code_hashes": hashes, "frozen_raw_records": int(data.raw_records),
    }
    _write_json(RESULTS / "pre_reference_hash_gate.json", gate)
    _update_state(phase_opened=False, reference_opened=False, pre_reference_gate="PASS", null_count=len(null_records))
    return null_records, gate


def _load_mask_structures(data: Any) -> tuple[dict[str, dict[str, dict[int, np.ndarray]]], dict[str, Any], list[dict[str, Any]]]:
    lock = _read_json(MASK_LOCK_PATH)
    manifest = _read_json(MASK_MANIFEST_PATH)
    rows = [row for row in manifest.get("coordinates", []) if row.get("mask_included") is True]
    if len(rows) != 21 or int(lock.get("condition_count", -1)) != 21:
        raise RuntimeError("old21 mask condition count changed")
    ids = {str(row["condition_id"]) for row in rows}
    if "consensus014" not in ids or len(ids) != 21:
        raise RuntimeError("old21 conditions incomplete")
    structures: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    hashes: list[dict[str, Any]] = []
    for row in rows:
        path = Path(str(row["path"])).resolve()
        actual = _sha(path)
        if actual != str(row["sha256"]):
            raise RuntimeError("old21 coordinate hash mismatch: %s" % path)
        structures[str(row["condition_id"])] = _load_tracks(path, str(row.get("format", "3dg")), tuple(data.chromosome_names))
        hashes.append({"condition_id": str(row["condition_id"]), "path": str(path), "format": str(row.get("format", "3dg")), "sha256": actual, "n_copies": 1 if row["condition_id"] == "consensus014" else 2})
    return structures, lock, hashes


def _run_frozen_mask_worker(data: Any, mask_hashes: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    if not FROZEN_MASK_WORKER_PYTHON.is_file():
        raise RuntimeError("CPython 3.13 frozen-mask worker runtime is missing")
    config_path = RESULTS / "frozen_mask_worker_config.json"
    output_path = RESULTS / "frozen_legacy_mask_snapshot.npz"
    meta_path = RESULTS / "frozen_legacy_mask_worker.json"
    config = {
        "chromosome_names": [str(x) for x in data.chromosome_names],
        "chromosome_lengths": [int(x) for x in data.chromosome_lengths],
        "conditions": [{"condition_id": str(row["condition_id"]), "path": str(row["path"]), "format": str(row["format"]), "n_copies": int(row["n_copies"])} for row in mask_hashes],
        "reference_path": str(REFERENCE_PATH),
    }
    _write_json(config_path, config)
    stdout_path = RESULTS / "frozen_mask_worker.stdout.log"
    stderr_path = RESULTS / "frozen_mask_worker.stderr.log"
    command = [str(FROZEN_MASK_WORKER_PYTHON), str(FROZEN_MASK_WORKER), "--config", str(config_path), "--output", str(output_path), "--meta", str(meta_path)]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    evidence = {
        "schema": "p9016-frozen-legacy-mask-worker-launch-v1", "command": command,
        "return_code": int(completed.returncode), "python": str(FROZEN_MASK_WORKER_PYTHON),
        "worker_code": str(FROZEN_MASK_WORKER), "worker_code_sha256": _sha(FROZEN_MASK_WORKER),
        "pyc": str(FROZEN_R2_PYC), "pyc_sha256": _sha(FROZEN_R2_PYC),
        "stdout": str(stdout_path), "stderr": str(stderr_path), "config": str(config_path),
        "output": str(output_path), "meta": str(meta_path),
    }
    _write_json(RESULTS / "frozen_mask_worker_launch.json", evidence)
    if completed.returncode != 0:
        raise RuntimeError("frozen legacy mask worker failed with return code %d" % completed.returncode)
    meta = _read_json(meta_path)
    if str(meta.get("frozen_r2comparison_sha256")) != FROZEN_R2_PYC_SHA256:
        raise RuntimeError("frozen legacy mask worker used unexpected r2comparison.pyc")
    if str(meta.get("output_sha256")) != _sha(output_path):
        raise RuntimeError("frozen legacy mask worker output hash changed")
    with np.load(output_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    worker_masks = {}
    for ci, name in enumerate(data.chromosome_names):
        worker_masks[str(name)] = {field: arrays["chr%d_%s" % (ci, field)] for field in ("positions", "pair_i", "pair_j", "common")}
    meta["launch"] = evidence
    return worker_masks, meta


def _build_masks(data: Any, structures: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]], lock: Mapping[str, Any], reference_tracks: Mapping[str, Mapping[int, np.ndarray]], mask_hashes: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    expected = {str(row["chromosome"]): row for row in lock["expected_by_chromosome"]}
    worker_masks, worker_meta = _run_frozen_mask_worker(data, mask_hashes)
    copy_counts = {str(row["condition_id"]): int(row["n_copies"]) for row in mask_hashes}
    masks: dict[str, dict[str, Any]] = {}
    exact_rows: list[dict[str, Any]] = []
    for ci, name in enumerate(data.chromosome_names):
        length = int(data.chromosome_lengths[ci])
        positions = np.arange(MASK_OFFSET_BP, length, BIN_SIZE_BP, dtype=np.int64)
        pair_i, pair_j = np.triu_indices(len(positions), k=1)
        common = np.ones(len(pair_i), dtype=bool)
        for cid, n_copies in copy_counts.items():
            suffixes = ("a",) if n_copies == 1 else ("a", "b")
            for suffix in suffixes:
                points = _dense_points(structures[cid], "c%02d%s" % (ci + 1, suffix), positions)
                common &= np.isfinite(_distance(points, pair_i, pair_j))
        ref_mat_points = _dense_points(reference_tracks, "%s(mat)" % name, positions)
        ref_pat_points = _dense_points(reference_tracks, "%s(pat)" % name, positions)
        ref_mat = _distance(ref_mat_points, pair_i, pair_j)
        ref_pat = _distance(ref_pat_points, pair_i, pair_j)
        common &= np.isfinite(ref_mat) & np.isfinite(ref_pat)
        valid = np.zeros(len(positions), dtype=bool)
        valid[pair_i[common]] = True
        valid[pair_j[common]] = True
        factor = common == (valid[pair_i] & valid[pair_j])
        factor_mismatches = int(np.count_nonzero(~factor))
        if factor_mismatches:
            raise RuntimeError("old21 common mask does not factorize for %s" % name)
        worker = worker_masks[str(name)]
        checks = {}
        for field, left, right in (
            ("positions", positions, np.asarray(worker["positions"], dtype=np.int64)),
            ("pair_i", pair_i, np.asarray(worker["pair_i"], dtype=np.int64)),
            ("pair_j", pair_j, np.asarray(worker["pair_j"], dtype=np.int64)),
            ("common", common, np.asarray(worker["common"], dtype=bool)),
        ):
            checks[field] = {"same": bool(np.array_equal(left, right)), "mismatch_count": None if left.shape != right.shape else int(np.count_nonzero(left != right))}
        if not all(item["same"] for item in checks.values()):
            raise RuntimeError("frozen legacy mask bit parity failure for %s" % name)
        row = expected.get(str(name))
        counts = {"n_bins": int(len(positions)), "n_total_non_diagonal_pairs": int(len(pair_i)), "n_common_pairs": int(common.sum())}
        if row is None or any(counts[key] != int(row[key]) for key in counts):
            raise RuntimeError("frozen old21 count mismatch for %s: %s vs %s" % (name, counts, row))
        masks[name] = {
            "chromosome": name, "chromosome_index": ci, "positions": positions,
            "pair_i": pair_i, "pair_j": pair_j, "common": common,
            "valid_local_bins": np.flatnonzero(valid), "ref_mat": ref_mat, "ref_pat": ref_pat,
            **counts,
        }
        exact_rows.append({"chromosome": name, "checks": checks, "factorization": {"same": True, "mismatch_count": factor_mismatches}, **counts})
    totals = {"n_total_non_diagonal_pairs": sum(row["n_total_non_diagonal_pairs"] for row in masks.values()), "n_common_pairs": sum(row["n_common_pairs"] for row in masks.values())}
    if totals != {"n_total_non_diagonal_pairs": 176201, "n_common_pairs": 157529}:
        raise RuntimeError("frozen old21 totals changed: %s" % totals)
    validation = {
        "schema": "p9016-real-evaluation-final-old21-mask-validation-v1", "status": "PASS",
        "condition_count": len(structures), "condition_ids": sorted(structures),
        "lock_sha256": _sha(MASK_LOCK_PATH), "manifest_sha256": _sha(MASK_MANIFEST_PATH),
        "per_chromosome": exact_rows, "totals": totals,
        "frozen_legacy_mask_worker": worker_meta,
        "exact_fields": ["positions", "pair_i", "pair_j", "common"],
        "factorization": "common == valid_local_bin[pair_i] & valid_local_bin[pair_j]",
    }
    return masks, validation


def _load_reference(data: Any) -> tuple[dict[str, dict[int, np.ndarray]], np.ndarray, str]:
    actual = _sha(REFERENCE_PATH)
    if actual != REFERENCE_SHA256:
        raise RuntimeError("reference SHA mismatch")
    tracks = _load_tracks(REFERENCE_PATH, "3dg", tuple(data.chromosome_names))
    reference = np.full((2, data.n_loci, 3), np.nan, dtype=np.float64)
    for ci, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(ci)
        positions = data.locus_bin[slc] * int(data.bin_size)
        for copy, suffix in enumerate(("mat", "pat")):
            rows = tracks.get("%s(%s)" % (name, suffix), {})
            for local, position in enumerate(positions):
                point = rows.get(int(position))
                if point is not None:
                    reference[copy, slc.start + local] = point
    return tracks, reference, actual


def _derive_rho(rho: dict[str, float]) -> dict[str, Any]:
    required = ("A_mat", "A_pat", "B_mat", "B_pat")
    if not all(math.isfinite(float(rho[key])) for key in required):
        return {
            "rho": {key: (float(rho[key]) if math.isfinite(float(rho[key])) else None) for key in required},
            "direct": None, "swapped": None, "matched": None, "cross": None, "contrast": None,
            "orientation": "undefined", "geometry_tie": False,
            "copy_A_margin": None, "copy_B_margin": None,
            "matched_mat_margin": None, "matched_pat_margin": None, "min_margin": None,
            "derived_defined": False, "undefined_reason": "one_or_more_of_four_rho_nonfinite",
        }
    direct = (rho["A_mat"] + rho["B_pat"]) / 2.0
    swapped = (rho["A_pat"] + rho["B_mat"]) / 2.0
    base = {"rho": {key: float(rho[key]) for key in required}, "direct": float(direct), "swapped": float(swapped)}
    if abs(direct - swapped) <= GEOMETRY_TIE_TOL:
        return {**base, "matched": float((direct + swapped) / 2.0), "cross": float((direct + swapped) / 2.0), "contrast": 0.0,
                "orientation": "unresolved_tie", "geometry_tie": True,
                "copy_A_margin": None, "copy_B_margin": None, "matched_mat_margin": None,
                "matched_pat_margin": None, "min_margin": None, "derived_defined": True, "undefined_reason": None}
    if direct > swapped:
        copy_a = rho["A_mat"] - rho["A_pat"]
        copy_b = rho["B_pat"] - rho["B_mat"]
        return {**base, "matched": float(direct), "cross": float(swapped), "contrast": float(direct - swapped),
                "orientation": "direct", "geometry_tie": False,
                "copy_A_margin": float(copy_a), "copy_B_margin": float(copy_b),
                "matched_mat_margin": float(copy_a), "matched_pat_margin": float(copy_b),
                "min_margin": float(min(copy_a, copy_b)), "derived_defined": True, "undefined_reason": None}
    copy_a = rho["A_pat"] - rho["A_mat"]
    copy_b = rho["B_mat"] - rho["B_pat"]
    return {**base, "matched": float(swapped), "cross": float(direct), "contrast": float(swapped - direct),
            "orientation": "swapped", "geometry_tie": False,
            "copy_A_margin": float(copy_a), "copy_B_margin": float(copy_b),
            "matched_mat_margin": float(copy_b), "matched_pat_margin": float(copy_a),
            "min_margin": float(min(copy_a, copy_b)), "derived_defined": True, "undefined_reason": None}


def _r2_result(coords: np.ndarray, reference: np.ndarray, data: Any, masks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    per_chromosome: list[dict[str, Any]] = []
    for ci, name in enumerate(data.chromosome_names):
        mask = masks[str(name)]
        slc = data.chromosome_slice(ci)
        global_indices = _mask_global_indices(data, mask)
        candidate = coords[:, global_indices]
        ref = reference[:, global_indices]
        pair_i = np.asarray(mask["pair_i"], dtype=np.int64)
        pair_j = np.asarray(mask["pair_j"], dtype=np.int64)
        common = np.asarray(mask["common"], dtype=bool)
        candidate_a = _distance(candidate[0], pair_i[common], pair_j[common])
        candidate_b = _distance(candidate[1], pair_i[common], pair_j[common])
        reference_mat = _distance(ref[0], pair_i[common], pair_j[common])
        reference_pat = _distance(ref[1], pair_i[common], pair_j[common])
        rhos = {"A_mat": _metric("pearson", candidate_a, reference_mat), "A_pat": _metric("pearson", candidate_a, reference_pat),
                "B_mat": _metric("pearson", candidate_b, reference_mat), "B_pat": _metric("pearson", candidate_b, reference_pat)}
        pearson_row = _derive_rho(rhos)
        rhos_s = {"A_mat": _metric("spearman", candidate_a, reference_mat), "A_pat": _metric("spearman", candidate_a, reference_pat),
                  "B_mat": _metric("spearman", candidate_b, reference_mat), "B_pat": _metric("spearman", candidate_b, reference_pat)}
        spearman_row = _derive_rho(rhos_s)
        per_chromosome.append({"chromosome": str(name), "chromosome_index": ci, "n_pairs": int(mask["n_common_pairs"]),
                               "n_total_non_diagonal_pairs": int(mask["n_total_non_diagonal_pairs"]),
                               "metrics": {"pearson": pearson_row, "spearman": spearman_row},
                               "status": "ok" if pearson_row["derived_defined"] and spearman_row["derived_defined"] else "undefined"})
    fields = ("matched", "cross", "contrast", "copy_A_margin", "copy_B_margin", "matched_mat_margin", "matched_pat_margin", "min_margin")
    macro: dict[str, dict[str, float | None]] = {}
    defined: dict[str, dict[str, int]] = {}
    undefined: dict[str, dict[str, list[str]]] = {}
    for metric in ("pearson", "spearman"):
        macro[metric], defined[metric], undefined[metric] = {}, {}, {}
        for field in fields:
            values = []
            missing = []
            for row in per_chromosome:
                value = row["metrics"][metric].get(field)
                if value is not None and math.isfinite(float(value)):
                    values.append(float(value))
                else:
                    missing.append(str(row["chromosome"]))
            macro[metric][field] = float(np.mean(values)) if values else None
            defined[metric][field] = len(values)
            undefined[metric][field] = missing
    return {
        "per_chromosome": per_chromosome, "macro_equal_chromosome_weight_defined_only": macro,
        "defined_chromosome_counts": defined, "undefined_chromosomes_by_field": undefined,
        "denominator_chromosomes": len(data.chromosome_names), "metric_definition": "frozen old21 common finite pairs",
        "primary_margins": "matched_mat_margin/matched_pat_margin/min_margin; candidate A/B labels are not fixed parental labels",
    }


def _mask_global_indices(data: Any, mask: Mapping[str, Any]) -> np.ndarray:
    chromosome_index = int(mask["chromosome_index"])
    slc = data.chromosome_slice(chromosome_index)
    positions = np.asarray(mask["positions"], dtype=np.int64)
    return slc.start + positions // int(data.bin_size)


def _valid_global_bins(data: Any, masks: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    valid = np.zeros(data.n_loci, dtype=bool)
    expected_indices: list[np.ndarray] = []
    for name in data.chromosome_names:
        mask = masks[str(name)]
        global_indices = _mask_global_indices(data, mask)
        valid_indices = np.asarray(mask["valid_local_bins"], dtype=np.int64)
        selected = global_indices[valid_indices]
        valid[selected] = True
        expected_indices.append(selected)
    expected = np.sort(np.concatenate(expected_indices))
    actual = np.flatnonzero(valid)
    if not np.array_equal(actual, expected):
        raise RuntimeError("valid-bin global locus index set disagrees with R2 mask positions")
    return valid


def _build_inter_cache(data: Any, masks: Mapping[str, Mapping[str, Any]], reference: np.ndarray) -> dict[str, Any]:
    valid = _valid_global_bins(data, masks)
    valid_bin_count = int(valid.sum())
    within_valid_pair_count = int(sum(math.comb(len(np.asarray(masks[str(name)]["valid_local_bins"])), 2) for name in data.chromosome_names))
    if within_valid_pair_count != 157529:
        raise RuntimeError("valid-bin within-chromosome pair count changed: %d" % within_valid_pair_count)
    expected_inter_pairs = math.comb(valid_bin_count, 2) - within_valid_pair_count
    all_inter = ~np.asarray(data.cis_pair, dtype=bool)
    pair_i = np.asarray(data.pair_i[all_inter], dtype=np.int64)
    pair_j = np.asarray(data.pair_j[all_inter], dtype=np.int64)
    keep = valid[pair_i] & valid[pair_j]
    pair_i, pair_j = pair_i[keep], pair_j[keep]
    if len(pair_i) != expected_inter_pairs:
        raise RuntimeError("inter valid-bin pair count mismatch: %d != %d" % (len(pair_i), expected_inter_pairs))
    ref_dist = _four_distances(reference, pair_i, pair_j)
    if not np.isfinite(ref_dist).all():
        raise RuntimeError("reference nonfinite on frozen inter valid-bin set")
    ref_sorted = np.sort(ref_dist, axis=0).ravel()
    return {
        "valid_global_bins": valid, "pair_i": pair_i, "pair_j": pair_j,
        "reference_sorted_four_copy": ref_sorted, "reference_rank": rankdata(ref_sorted),
        "valid_bin_count": valid_bin_count, "within_valid_pair_count": within_valid_pair_count,
        "eligible_locus_pairs": int(len(pair_i)), "denominator": int(4 * len(pair_i)),
        "expected_eligible_locus_pairs": int(expected_inter_pairs),
        "global_locus_index_assertion": True,
    }


def _spearman_ranked(x: np.ndarray, reference_rank: np.ndarray) -> float:
    rx = rankdata(x)
    if np.ptp(rx) == 0.0 or np.ptp(reference_rank) == 0.0:
        return float("nan")
    xc, yc = rx - rx.mean(), reference_rank - reference_rank.mean()
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    return float(np.dot(xc, yc) / denom) if denom else float("nan")


def _inter_result(coords: np.ndarray, inter_cache: Mapping[str, Any]) -> dict[str, Any]:
    pair_i = np.asarray(inter_cache["pair_i"], dtype=np.int64)
    pair_j = np.asarray(inter_cache["pair_j"], dtype=np.int64)
    candidate_dist = _four_distances(coords, pair_i, pair_j)
    if not np.isfinite(candidate_dist).all():
        raise RuntimeError("candidate nonfinite on frozen inter valid-bin set")
    candidate_sorted = np.sort(candidate_dist, axis=0).ravel()
    reference_sorted = np.asarray(inter_cache["reference_sorted_four_copy"], dtype=np.float64)
    result = {
        "pearson": _metric("pearson", candidate_sorted, reference_sorted),
        "spearman": _spearman_ranked(candidate_sorted, np.asarray(inter_cache["reference_rank"], dtype=np.float64)),
        "eligible_locus_pairs": int(len(pair_i)), "denominator": int(4 * len(pair_i)),
        "valid_bin_count": int(np.asarray(inter_cache["valid_global_bins"], dtype=bool).sum()),
        "valid_bin_policy": "old21 common-mask participating bins only; no reference-finite network expansion",
        "order_statistic_vector": "sorted four-copy distances per inter-chromosome locus pair",
    }
    return {key: (float(value) if isinstance(value, (float, np.floating)) and math.isfinite(float(value)) else value) for key, value in result.items()}


def _rg(coords: np.ndarray) -> tuple[float, np.ndarray, float]:
    beads = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    center = beads.mean(axis=0)
    radius = np.sqrt(np.sum((beads - center) ** 2, axis=1))
    return float(np.sqrt(np.mean(radius ** 2))), center, float(np.linalg.norm(beads, axis=1).max())


def _map_metrics(coords: np.ndarray, p: float, data: Any, *, argmax_scope: str = "all") -> dict[str, Any]:
    if argmax_scope not in ("all", "positive_offdiag"):
        raise ValueError("unknown MAP argmax scope")
    counts = np.asarray(data.counts, dtype=np.float64)
    cis = np.asarray(data.cis_pair, dtype=bool)
    positive = counts > 0.0
    weighted_indices = np.flatnonzero(positive)
    eval_indices = np.arange(data.n_pairs, dtype=np.int64) if argmax_scope == "all" else weighted_indices
    sums = {"intra": 0.0, "inter": 0.0}
    denom = {"intra": float(counts[cis].sum()), "inter": float(counts[~cis].sum())}
    argmax_score = -float("inf")
    argmax_pair = (-1, -1)
    argmax_choice = None
    argmax_distance = None
    for start in range(0, len(eval_indices), MAP_BLOCK):
        local_indices = eval_indices[start:start + MAP_BLOCK]
        i = np.asarray(data.pair_i[local_indices], dtype=np.int64)
        j = np.asarray(data.pair_j[local_indices], dtype=np.int64)
        local_cis = cis[local_indices]
        delta = np.stack((coords[0, i] - coords[0, j], coords[0, i] - coords[1, j], coords[1, i] - coords[0, j], coords[1, i] - coords[1, j]), axis=0)
        squared = np.sum(delta * delta, axis=2)
        distances = np.sqrt(squared)
        kernels = MAP_EPSILON + (1.0 - MAP_EPSILON) * (1.0 + squared / (data.r0 * data.r0)) ** -2
        weights = np.stack((np.where(local_cis, 0.5 * p, 0.25), np.where(local_cis, 0.5 * (1.0 - p), 0.25), np.where(local_cis, 0.5 * (1.0 - p), 0.25), np.where(local_cis, 0.5 * p, 0.25)), axis=0)
        scores = weights * kernels
        choice = np.argmax(scores, axis=0)
        columns = np.arange(len(local_indices))
        selected = distances[choice, columns]
        selected_scores = scores[choice, columns]
        if not (np.isfinite(distances).all() and np.isfinite(scores).all()):
            raise RuntimeError("nonfinite MAP distance/score")
        positive_here = positive[local_indices]
        if np.any(positive_here):
            weight_counts = counts[local_indices[positive_here]]
            chosen = selected[positive_here]
            groups = local_cis[positive_here]
            sums["intra"] += float(np.dot(weight_counts[groups], chosen[groups]))
            sums["inter"] += float(np.dot(weight_counts[~groups], chosen[~groups]))
        block_max = int(np.argmax(selected_scores))
        score_max = float(selected_scores[block_max])
        if score_max > argmax_score:
            argmax_score = score_max
            argmax_pair = (int(i[block_max]), int(j[block_max]))
            argmax_choice = int(choice[block_max])
            argmax_distance = float(selected[block_max])
    rg, center, max_radius = _rg(coords)
    total_denom = denom["intra"] + denom["inter"]
    if denom["intra"] <= 0 or denom["inter"] <= 0 or total_denom <= 0:
        raise RuntimeError("MAP count denominator is empty")
    return {
        "p": float(p), "r0": float(data.r0),
        "distance_intra_map": sums["intra"] / denom["intra"],
        "distance_inter_map": sums["inter"] / denom["inter"],
        "distance_all_map": (sums["intra"] + sums["inter"]) / total_denom,
        "count_intra": int(round(denom["intra"])), "count_inter": int(round(denom["inter"])),
        "count_all_offdiag": int(round(total_denom)),
        "positive_pair_rows_intra": int(np.count_nonzero(positive & cis)),
        "positive_pair_rows_inter": int(np.count_nonzero(positive & ~cis)),
        "positive_pair_rows_all_offdiag": int(np.count_nonzero(positive)),
        "same_bin_excluded": True,
        "map_definition": "per-pair argmax over AA/AB/BA/BB of wK; selected distance then raw-count weighted",
        "choice_order": ["AA", "AB", "BA", "BB"],
        "weights_intra": [0.5 * float(p), 0.5 * (1.0 - float(p)), 0.5 * (1.0 - float(p)), 0.5 * float(p)],
        "weights_inter": [0.25, 0.25, 0.25, 0.25],
        "argmax_scope": argmax_scope, "argmax_wK": argmax_score,
        "argmax_pair_i": argmax_pair[0], "argmax_pair_j": argmax_pair[1],
        "argmax_choice": ["AA", "AB", "BA", "BB"][argmax_choice] if argmax_choice is not None else None,
        "argmax_selected_distance": argmax_distance,
        "whole_cell_Rg": rg, "whole_cell_beads": int(coords.shape[0] * data.n_loci),
        "Rg_center": center, "max_bead_radius": max_radius,
        "Rg_definition": "sqrt(mean(sum((coords.reshape(-1,3)-whole_cell_mean)^2))); all physical beads equal weight",
        "counts_weighting": "all raw offdiag record counts; zero-count pairs contribute exactly zero",
    }


def _map_regression(data: Any) -> dict[str, Any]:
    if _sha(OLD036_CHECKPOINT) != OLD036_CHECKPOINT_SHA256:
        raise RuntimeError("old036 regression checkpoint hash mismatch")
    with np.load(OLD036_CHECKPOINT, allow_pickle=False) as payload:
        coords = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        theta = np.asarray(payload["theta"], dtype=np.float64).copy()
    p = float(contact_model.p_from_q(float(theta[-1]))[0])
    observed = _map_metrics(coords, p, data, argmax_scope="all")
    expected = {
        "distance_intra_map": 0.10172084148686669,
        "distance_inter_map": 0.2500344111682071,
        "distance_all_map": 0.16836027134713466,
        "whole_cell_Rg": 0.34507274075430555,
    }
    mismatches = []
    for key, value in expected.items():
        if abs(float(observed[key]) - value) > MAP_REGRESSION_ATOL:
            mismatches.append({"field": key, "expected": value, "observed": observed[key], "abs_diff": abs(float(observed[key]) - value)})
    result = {
        "schema": "p9016-real-map-rg-regression-v1", "status": "PASS" if not mismatches else "FAIL",
        "checkpoint": str(OLD036_CHECKPOINT), "checkpoint_sha256": OLD036_CHECKPOINT_SHA256,
        "metrics_tsv": str(MAP_REGRESSION_METRICS), "metrics_tsv_sha256": _sha(MAP_REGRESSION_METRICS),
        "numeric_atol": MAP_REGRESSION_ATOL, "expected": expected, "observed": observed,
        "mismatches": mismatches, "fit_started": False, "synthetic": False,
    }
    _write_json(RESULTS / "map_rg_regression.json", result)
    if mismatches:
        raise RuntimeError("043 MAP/Rg regression failed")
    return result


def _full_j_diagnostics(candidates: list[dict[str, Any]], data: Any) -> list[dict[str, Any]]:
    objectives: dict[str, Any] = {}
    output: list[dict[str, Any]] = []
    for row in candidates:
        model = str(row["model_id"])
        if model not in objectives:
            weights = PenaltyWeights(**{key: float(value) for key, value in FULL_J_WEIGHTS.items()})
            objectives[model] = frozen_objective(data, model, weights, None)
        theta = np.concatenate((np.asarray(row["raw_y"], dtype=np.float64).ravel(), np.asarray([row["q"]], dtype=np.float64)))
        value, _gradient, components = objectives[model].evaluate(theta, need_gradient=False)
        output.append({
            "candidate_id": row["candidate_id"], "kind": row["kind"], "model_id": model,
            "candidate": row.get("candidate"), "variant": row.get("variant"),
            "counterfactual_full_J": row.get("variant") == "count-only",
            "full_J_total": float(value), "full_J_weights": FULL_J_WEIGHTS,
            "components": components, "count_nll_normalized": components.get("count_nll_normalized"),
            "diagnostic_only": True, "used_for_stop_or_selection": False,
        })
    _write_json(RESULTS / "full_j_endpoint_diagnostics.json", {
        "schema": "p9016-real-endpoint-full-j-diagnostics-v1", "diagnostic_only": True,
        "used_for_stop_or_selection": False, "weights": FULL_J_WEIGHTS, "rows": output,
        "reference_opened": True, "synthetic_count": 0,
    })
    return output


def _bootstrap_indices() -> np.ndarray:
    path = OUT / "bootstrap_indices_seed450301_10000x20.npy"
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 20, size=(BOOTSTRAP_DRAWS, 20), dtype=np.int64)
    np.save(path, indices)
    return indices


def _bootstrap_fixed(left: list[Any], right: list[Any], names: list[str], indices: np.ndarray) -> dict[str, Any]:
    a = np.asarray([float(value) if value is not None and math.isfinite(float(value)) else np.nan for value in left], dtype=np.float64)
    b = np.asarray([float(value) if value is not None and math.isfinite(float(value)) else np.nan for value in right], dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    undefined = [str(name) for name, keep in zip(names, finite) if not keep]
    diff = a - b
    output: dict[str, Any] = {
        "denominator_chromosomes": 20, "defined_chromosomes": int(np.count_nonzero(finite)),
        "defined_chromosome_names": [str(name) for name, keep in zip(names, finite) if keep],
        "undefined_chromosomes": undefined, "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
        "unit": "paired chromosome technical/structural bootstrap; not biological replicates",
        "fixed_index_matrix_shape": list(indices.shape),
        "full_20_estimate_defined": bool(np.all(finite)),
    }
    if np.all(finite):
        draw_values = diff[indices].mean(axis=1)
        output.update({"mean": float(diff.mean()), "ci95": [float(np.percentile(draw_values, 2.5)), float(np.percentile(draw_values, 97.5))], "draw_summary": {"mean": float(draw_values.mean()), "std": float(draw_values.std(ddof=1)), "min": float(draw_values.min()), "max": float(draw_values.max())}})
    else:
        output.update({"mean": None, "ci95": [None, None], "draw_summary": None})
    defined_diff = diff[finite]
    output["defined_only_descriptive"] = {
        "label": "defined-only descriptive; not the fixed-20 estimate",
        "n": int(len(defined_diff)), "mean": float(defined_diff.mean()) if len(defined_diff) else None,
        "std": float(defined_diff.std(ddof=1)) if len(defined_diff) > 1 else None,
    }
    output["left_wins"] = int(np.count_nonzero(finite & (a > b + GEOMETRY_TIE_TOL)))
    output["right_wins"] = int(np.count_nonzero(finite & (b > a + GEOMETRY_TIE_TOL)))
    output["ties"] = int(np.count_nonzero(finite & (np.abs(a - b) <= GEOMETRY_TIE_TOL)))
    return output


def _paired(real_results: list[dict[str, Any]], left_id: str, right_id: str, label: str, indices: np.ndarray) -> dict[str, Any]:
    by_id = {row["candidate_id"]: row for row in real_results}
    left, right = by_id.get(left_id), by_id.get(right_id)
    if left is None or right is None:
        raise RuntimeError("paired candidate missing: %s / %s" % (left_id, right_id))
    names = [str(row["chromosome"]) for row in left["r2"]["per_chromosome"]]
    left_rows = {row["chromosome"]: row for row in left["r2"]["per_chromosome"]}
    right_rows = {row["chromosome"]: row for row in right["r2"]["per_chromosome"]}
    fields = ("matched", "cross", "contrast", "copy_A_margin", "copy_B_margin", "matched_mat_margin", "matched_pat_margin", "min_margin")
    output: dict[str, Any] = {"label": label, "left": left_id, "right": right_id, "denominator_chromosomes": 20, "metrics": {}}
    for metric in ("pearson", "spearman"):
        for field in fields:
            left_values = [left_rows[name]["metrics"][metric].get(field) for name in names]
            right_values = [right_rows[name]["metrics"][metric].get(field) for name in names]
            output["metrics"][metric + ":" + field] = _bootstrap_fixed(left_values, right_values, names, indices)
    return output


def _trajectory_sources(manifests: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = list(manifests["combined"]["stages"])
    items.extend({"fit_id": arm["fit_id"], "stage": "1Mb", "lineage": "extension"} for arm in manifests["extension"].get("arms", []))
    return items


def _trajectory_rows(data_cache: Mapping[int, Any], manifests: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trajectory_manifest = _read_json(RUN / "trajectory_inputs/manifest.json")
    frozen_stage_by_key = {(item["fit_id"], item["stage"]): item for item in trajectory_manifest.get("stages", [])}
    rows: list[dict[str, Any]] = []
    na_rows: list[dict[str, Any]] = []
    verified_frozen = 0
    for item in _trajectory_sources(manifests):
        fit_id, stage = str(item["fit_id"]), str(item["stage"])
        lineage = str(item.get("lineage", ""))
        if lineage == "reused_base":
            checkpoint_dir = RUN / "trajectory_inputs/reused_base/checkpoints" / fit_id / stage
        elif lineage == "base_remaining":
            checkpoint_dir = RUN / "base_remaining/checkpoints" / fit_id / stage
        elif lineage == "extension":
            checkpoint_dir = RUN / "checkpoints" / fit_id / stage
        else:
            raise RuntimeError("unknown trajectory lineage: %s" % lineage)
        frozen_item = frozen_stage_by_key.get((fit_id, stage))
        expected_by_target = {}
        if frozen_item is not None:
            for record in frozen_item.get("verified_checkpoints", []):
                expected_by_target[str(Path(record["target"]).resolve())] = record
        checkpoints = sorted(checkpoint_dir.glob("accepted-*.npz"))
        if not checkpoints:
            raise RuntimeError("trajectory checkpoint directory empty: %s" % checkpoint_dir)
        data = data_cache[STAGE_BIN_SIZES[stage]]
        for checkpoint in checkpoints:
            if expected_by_target:
                evidence = expected_by_target.get(str(checkpoint.resolve()))
                if evidence is None or _sha(checkpoint) != str(evidence["sha256"]):
                    raise RuntimeError("frozen trajectory checkpoint hash mismatch: %s" % checkpoint)
                verified_frozen += 1
            with np.load(checkpoint, allow_pickle=False) as payload:
                coords = np.asarray(payload["coordinates"], dtype=np.float64).copy()
                theta = np.asarray(payload["theta"], dtype=np.float64).copy()
                iteration = int(np.asarray(payload["iteration"]).item())
                nfev = int(np.asarray(payload["nfev"]).item()) if "nfev" in payload else None
            if coords.shape != (2, data.n_loci, 3) or theta.shape != (6 * data.n_loci + 1,) or not np.isfinite(coords).all() or not np.isfinite(theta).all():
                raise RuntimeError("trajectory checkpoint shape/domain failure: %s" % checkpoint)
            p = float(contact_model.p_from_q(float(theta[-1]))[0])
            map_row = _map_metrics(coords, p, data, argmax_scope="positive_offdiag")
            parts = fit_id.split("-")
            if fit_id.startswith("real-extension-"):
                model_id, candidate = parts[2], "-".join(parts[3:])
            else:
                model_id, candidate = parts[1], parts[2]
            rows.append({"fit_id": fit_id, "model_id": model_id, "candidate": candidate, "stage": stage,
                         "lineage": lineage, "iteration": iteration, "nfev": nfev,
                         "checkpoint": _rel(checkpoint), "checkpoint_sha256": _sha(checkpoint), **map_row})
        if frozen_item is not None:
            for missing in frozen_item.get("missing_or_unproven", []):
                na_rows.append({"fit_id": fit_id, "stage": stage, "iteration": int(missing["iteration"]), "nfev": int(missing["nfev"]), "status": "NA", "reason": str(missing["reason"]), "lineage": "reused_base"})
    declared_verified = int(sum(int(item.get("verified_checkpoint_count", 0)) for item in trajectory_manifest.get("stages", [])))
    declared_na = int(sum(len(item.get("missing_or_unproven", [])) for item in trajectory_manifest.get("stages", [])))
    if verified_frozen != declared_verified or len(na_rows) != declared_na:
        raise RuntimeError("trajectory frozen counts changed")
    audit = {"trajectory_manifest_sha256": _sha(RUN / "trajectory_inputs/manifest.json"), "frozen_reused_verified_checkpoints": verified_frozen, "frozen_reused_na_points": len(na_rows), "all_stage_checkpoint_rows": len(rows), "base_remaining_and_extension_checkpoint_rows_included": True}
    return rows, na_rows, audit


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_jsonable(row) for row in rows)


def _plot(real_results: list[dict[str, Any]], trajectory_rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        _write_json(RESULTS / "plot_status.json", {"status": "unavailable", "error": str(exc)})
        return []
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5})
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    labels = [row["candidate_id"].replace("real-extension-", "ext-").replace("real-", "") for row in real_results]
    values = [row["r2"]["macro_equal_chromosome_weight_defined_only"]["pearson"]["matched"] for row in real_results]
    fig, ax = plt.subplots(figsize=(3, 2.25), dpi=300)
    ax.bar(np.arange(len(labels)), values, color="#3b6ea8")
    ax.set_ylabel("Pearson matched")
    ax.set_xlabel("P9016 endpoint")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right")
    ax.set_title("Frozen old21 R2")
    fig.tight_layout()
    path = plot_dir / "r2_matched_summary.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(_rel(path))
    fig, ax = plt.subplots(figsize=(3, 2.25), dpi=300)
    for fit_id in sorted({row["fit_id"] for row in trajectory_rows}):
        subset = [row for row in trajectory_rows if row["fit_id"] == fit_id]
        ax.plot([row["iteration"] for row in subset], [row["argmax_wK"] for row in subset], linewidth=0.55, label=fit_id.replace("real-extension-", "ext-").replace("real-", ""))
    ax.set_xlabel("Accepted iteration")
    ax.set_ylabel("argmax(wK), positive pairs")
    ax.set_title("Frozen checkpoint MAP trajectory")
    if trajectory_rows:
        ax.legend(ncol=2, fontsize=4, loc="best")
    fig.tight_layout()
    path = plot_dir / "trajectory_map_argmax.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(_rel(path))
    return paths


def _artifact_hashes() -> dict[str, Any]:
    records = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name in {"artifact_hashes.json", "lock"}:
            continue
        records.append({"path": _rel(path), "sha256": _sha(path), "bytes": int(path.stat().st_size)})
    result = {"schema": "p9016-real-evaluation-final-artifact-hashes-v1", "artifact_count": len(records), "records": records}
    _write_json(OUT / "artifact_hashes.json", result)
    return result


def evaluate() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _update_state(phase_opened=False, reference_opened=False, command="evaluate", status="running")
    data = _load_data()
    candidates, initials, manifests = _load_candidates(data)
    null_records, gate = _pre_reference_gate(candidates, initials, data)
    # The gate is persisted before any reference path is opened.
    if gate.get("status") != "PASS" or gate.get("reference_opened"):
        raise RuntimeError("pre-reference gate did not pass")
    dependency_provenance = _dependency_provenance()
    structures, mask_lock, mask_hashes = _load_mask_structures(data)
    reference_tracks, reference, reference_sha = _load_reference(data)
    _update_state(phase_opened=False, reference_opened=True, reference_sha256=reference_sha, status="reference_opened")
    masks, mask_validation = _build_masks(data, structures, mask_lock, reference_tracks, mask_hashes)
    _write_json(RESULTS / "mask_validation.json", {**mask_validation, "payload_hashes": mask_hashes})
    inter_cache = _build_inter_cache(data, masks, reference)
    _write_json(RESULTS / "inter_cache_audit.json", {
        "schema": "p9016-real-evaluation-final-inter-cache-v1", "eligible_locus_pairs": inter_cache["eligible_locus_pairs"],
        "denominator": inter_cache["denominator"], "valid_bin_count": int(np.asarray(inter_cache["valid_global_bins"]).sum()),
        "within_valid_pair_count": inter_cache["within_valid_pair_count"], "expected_eligible_locus_pairs": inter_cache["expected_eligible_locus_pairs"],
        "global_locus_index_assertion": inter_cache["global_locus_index_assertion"],
        "reference_sorted_four_copy_sha256": _hash_array(np.asarray(inter_cache["reference_sorted_four_copy"], dtype=np.float64)),
        "reference_rank_sha256": _hash_array(np.asarray(inter_cache["reference_rank"], dtype=np.float64)),
        "valid_bin_policy": "common-mask participating endpoints only; common factorization verified",
    })
    real_results: list[dict[str, Any]] = []
    all_sources = candidates + initials
    for source in all_sources:
        r2 = _r2_result(source["coordinates"], reference, data, masks)
        inter = _inter_result(source["coordinates"], inter_cache)
        map_row = _map_metrics(source["coordinates"], source["p"], data, argmax_scope="all")
        real_results.append({"candidate_id": source["candidate_id"], "kind": source["kind"], "model_id": source["model_id"], "candidate": source.get("candidate"), "variant": source.get("variant"), "coordinate_file_sha256": source["file_sha256"], "coordinate_array_sha256": source["coordinate_array_sha256"], "r2": r2, "inter": inter, "map": map_row})
    regression = _map_regression(data)
    full_j = _full_j_diagnostics(candidates, data)
    data_cache = {int(size): load_aggregate(Path(path)) for size, path in manifests["formal"]["real_input_paths"].items()}
    trajectory_rows, trajectory_na, trajectory_audit = _trajectory_rows(data_cache, manifests)
    _write_tsv(RESULTS / "trajectory_summary.tsv", trajectory_rows)
    _write_tsv(RESULTS / "trajectory_na.tsv", trajectory_na)
    _write_json(RESULTS / "trajectory_audit.json", trajectory_audit)
    bootstrap_indices = _bootstrap_indices()
    bootstrap_hash = _sha(OUT / "bootstrap_indices_seed450301_10000x20.npy")
    selected_sources = manifests["combined"].get("selected_sources", {})
    selected_ids = {model: ("real-%s-%s" % (model, selected_sources.get(model, {}).get("selected_by_own_count_nll")) if selected_sources.get(model, {}).get("selected_by_own_count_nll") in ("consensus", "random") else None) for model in ("S", "G")}
    pairs: dict[str, dict[str, Any]] = {}
    for candidate in ("consensus", "random"):
        pairs["G_minus_S-%s" % candidate] = _paired(real_results, "real-G-%s" % candidate, "real-S-%s" % candidate, "same-source-%s" % candidate, bootstrap_indices)
    if selected_ids["G"] and selected_ids["S"]:
        pairs["G_selected_minus_S_selected"] = _paired(real_results, selected_ids["G"], selected_ids["S"], "count-selected-G-minus-S", bootstrap_indices)
    for model in ("S", "G"):
        base_id = selected_ids[model]
        if base_id is None:
            continue
        for variant in ("full-J", "count-only"):
            ext_id = "real-extension-%s-%s" % (model, variant)
            pairs["%s_vs_selected_base" % ext_id] = _paired(real_results, ext_id, base_id, "%s-vs-selected-base" % ext_id, bootstrap_indices)
        pairs["real-extension-%s-count-only_vs_full-J" % model] = _paired(real_results, "real-extension-%s-count-only" % model, "real-extension-%s-full-J" % model, "%s-count-only-minus-full-J" % model, bootstrap_indices)
    _write_json(RESULTS / "paired_bootstrap.json", {"schema": "p9016-real-evaluation-final-paired-bootstrap-v1", "indices_sha256": bootstrap_hash, "indices_shape": list(bootstrap_indices.shape), "pairs": pairs})
    paired_rows = []
    for label, item in pairs.items():
        for metric_key, summary in item["metrics"].items():
            row = {"comparison": label, "left": item["left"], "right": item["right"], "metric": metric_key, **summary}
            paired_rows.append(row)
    _write_tsv(RESULTS / "paired_bootstrap.tsv", paired_rows)
    null_results: list[dict[str, Any]] = []
    for record in null_records:
        with np.load((RUN / record["path"]).resolve(), allow_pickle=False) as payload:
            coords = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        source = next(row for row in all_sources if row["candidate_id"] == record["source_candidate_id"])
        r2 = _r2_result(coords, reference, data, masks)
        inter = _inter_result(coords, inter_cache)
        null_results.append({**record, "source_kind": source["kind"], "r2": r2, "inter": inter})
    _write_json(RESULTS / "null_evaluations.json", {"schema": "p9016-real-evaluation-final-null-evaluations-v1", "source_count": 10, "draws_per_source": 17, "null_count": len(null_results), "rows": null_results})
    null_rows = []
    for row in null_results:
        macro = row["r2"]["macro_equal_chromosome_weight_defined_only"]
        null_rows.append({
            "source_candidate_id": row["source_candidate_id"], "source_kind": row["source_kind"], "null_kind": row["null_kind"], "seed": row["seed"],
            "global_scale": row["global_scale"], "pre_scale_radius": row["pre_scale_radius"], "post_scale_radius": row["post_scale_radius"],
            "r2_pearson_matched": macro["pearson"]["matched"], "r2_pearson_cross": macro["pearson"]["cross"], "r2_pearson_contrast": macro["pearson"]["contrast"],
            "r2_pearson_copy_A_margin": macro["pearson"]["copy_A_margin"], "r2_pearson_copy_B_margin": macro["pearson"]["copy_B_margin"],
            "r2_pearson_matched_mat_margin": macro["pearson"]["matched_mat_margin"], "r2_pearson_matched_pat_margin": macro["pearson"]["matched_pat_margin"], "r2_pearson_min_margin": macro["pearson"]["min_margin"],
            "r2_spearman_matched": macro["spearman"]["matched"], "r2_spearman_cross": macro["spearman"]["cross"], "r2_spearman_contrast": macro["spearman"]["contrast"],
            "r2_spearman_copy_A_margin": macro["spearman"]["copy_A_margin"], "r2_spearman_copy_B_margin": macro["spearman"]["copy_B_margin"],
            "r2_spearman_matched_mat_margin": macro["spearman"]["matched_mat_margin"], "r2_spearman_matched_pat_margin": macro["spearman"]["matched_pat_margin"], "r2_spearman_min_margin": macro["spearman"]["min_margin"],
            "inter_pearson": row["inter"]["pearson"], "inter_spearman": row["inter"]["spearman"], "inter_eligible_locus_pairs": row["inter"]["eligible_locus_pairs"], "inter_denominator": row["inter"]["denominator"],
        })
    _write_tsv(RESULTS / "null_per_draw.tsv", null_rows)
    summary_rows = []
    for source_id in sorted({row["source_candidate_id"] for row in null_results}):
        for null_kind in ("u_zero", "random_u"):
            subset = [row for row in null_rows if row["source_candidate_id"] == source_id and row["null_kind"] == null_kind]
            for field in ("r2_pearson_matched", "r2_pearson_cross", "r2_pearson_contrast", "r2_pearson_matched_mat_margin", "r2_pearson_matched_pat_margin", "r2_pearson_min_margin", "r2_spearman_matched", "r2_spearman_cross", "r2_spearman_contrast", "r2_spearman_matched_mat_margin", "r2_spearman_matched_pat_margin", "r2_spearman_min_margin", "inter_pearson", "inter_spearman"):
                values = np.asarray([row[field] if row[field] is not None else np.nan for row in subset], dtype=np.float64)
                finite = values[np.isfinite(values)]
                summary_rows.append({"source_candidate_id": source_id, "null_kind": null_kind, "draw_count": len(subset), "metric": field, "defined_draws": len(finite), "mean": float(finite.mean()) if len(finite) else None, "std": float(finite.std(ddof=1)) if len(finite) > 1 else None, "min": float(finite.min()) if len(finite) else None, "max": float(finite.max()) if len(finite) else None})
    _write_tsv(RESULTS / "null_distribution_summary.tsv", summary_rows)
    r2_rows = []
    summary_rows = []
    for result in real_results:
        for row in result["r2"]["per_chromosome"]:
            for metric in ("pearson", "spearman"):
                item = {"candidate_id": result["candidate_id"], "kind": result["kind"], "model_id": result["model_id"], "variant": result.get("variant"), "chromosome": row["chromosome"], "metric": metric, "n_pairs": row["n_pairs"], **row["metrics"][metric]}
                item["rho_A_mat"] = row["metrics"][metric]["rho"]["A_mat"]
                item["rho_A_pat"] = row["metrics"][metric]["rho"]["A_pat"]
                item["rho_B_mat"] = row["metrics"][metric]["rho"]["B_mat"]
                item["rho_B_pat"] = row["metrics"][metric]["rho"]["B_pat"]
                r2_rows.append(item)
        macro = result["r2"]["macro_equal_chromosome_weight_defined_only"]
        summary_rows.append({"candidate_id": result["candidate_id"], "kind": result["kind"], "model_id": result["model_id"], "variant": result.get("variant"),
                             "pearson_matched": macro["pearson"]["matched"], "pearson_cross": macro["pearson"]["cross"], "pearson_contrast": macro["pearson"]["contrast"], "pearson_matched_mat_margin": macro["pearson"]["matched_mat_margin"], "pearson_matched_pat_margin": macro["pearson"]["matched_pat_margin"], "pearson_min_margin": macro["pearson"]["min_margin"],
                             "spearman_matched": macro["spearman"]["matched"], "spearman_cross": macro["spearman"]["cross"], "spearman_contrast": macro["spearman"]["contrast"], "spearman_matched_mat_margin": macro["spearman"]["matched_mat_margin"], "spearman_matched_pat_margin": macro["spearman"]["matched_pat_margin"], "spearman_min_margin": macro["spearman"]["min_margin"],
                             "inter_pearson": result["inter"]["pearson"], "inter_spearman": result["inter"]["spearman"], "inter_eligible_locus_pairs": result["inter"]["eligible_locus_pairs"], "inter_denominator": result["inter"]["denominator"],
                             "map_distance_intra": result["map"]["distance_intra_map"], "map_distance_inter": result["map"]["distance_inter_map"], "map_distance_all": result["map"]["distance_all_map"], "whole_cell_Rg": result["map"]["whole_cell_Rg"], "whole_cell_beads": result["map"]["whole_cell_beads"], "map_positive_pair_rows_all_offdiag": result["map"]["positive_pair_rows_all_offdiag"]})
    _write_tsv(RESULTS / "r2_per_chromosome.tsv", r2_rows)
    _write_tsv(RESULTS / "r2_summary.tsv", summary_rows)
    diagnostic_rows = []
    for item in full_j:
        row = {"candidate_id": item["candidate_id"], "kind": item["kind"], "model_id": item["model_id"], "candidate": item.get("candidate"), "variant": item.get("variant"), "counterfactual_full_J": item["counterfactual_full_J"], "full_J_total": item["full_J_total"], "count_nll_normalized": item.get("count_nll_normalized")}
        for key, value in item["components"].items():
            if not isinstance(value, (dict, list, tuple)):
                row["component:" + str(key)] = value
        diagnostic_rows.append(row)
    _write_tsv(RESULTS / "full_j_endpoint_diagnostics.tsv", diagnostic_rows)
    old036 = _old036_regression(data, reference, masks)
    result = {
        "schema": "p9016-real-only-evaluation-final-v1", "status": "complete", "scope": "P9016 real cell only",
        "synthetic_evaluation": False, "phase_opened": False, "reference_opened": True,
        "reference": {"path": _rel(REFERENCE_PATH), "sha256": reference_sha, "read_stage": "after 8 endpoints + 2 initial controls + 170 nulls + code/input hash gate"},
        "pre_reference_hash_gate": gate, "old21_mask_validation": mask_validation,
        "candidate_count": 8, "initial_count": 2, "null_count": len(null_results), "real_results": real_results,
        "full_J_endpoint_diagnostics": full_j, "paired_bootstrap": pairs, "selected_sources": selected_ids,
        "null_summary": {"sources": 10, "draws_per_source": 17, "total_draws": len(null_results), "per_draw_tsv": _rel(RESULTS / "null_per_draw.tsv"), "distribution_summary_tsv": _rel(RESULTS / "null_distribution_summary.tsv")},
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "denominator_chromosomes": 20, "indices_path": _rel(OUT / "bootstrap_indices_seed450301_10000x20.npy"), "indices_sha256": bootstrap_hash, "biological_replicate_claim": False},
        "trajectory": trajectory_audit, "old036_spearman_regression": old036,
        "map_rg_regression": regression, "dependency_provenance": dependency_provenance, "input_code_hashes": _read_json(OUT / "input_code_hashes.json"),
        "metric_rules": {"map": "AA/AB/BA/BB per-pair score argmax; raw-count weighted intra/inter/all offdiag; diag excluded; Rg whole-cell 5290 physical beads", "r2": "old21 common mask at numeric 3Mb offset; four rho; per-chromosome direct/swapped selection; tie <=1e-12 unresolved", "inter": "common-mask valid-bin factorization; sorted four-copy distance per inter locus pair; denominator 4x pair count", "bootstrap": "fixed 10000x20 index matrix; undefined chromosomes retained and full-20 estimate/CI set to null", "null": "u0 plus 16 within-chromosome u permutations per source, z fixed, one global rescale only if needed"},
        "output_index": {"evaluation_json": _rel(RESULTS / "evaluation.json"), "r2_summary": _rel(RESULTS / "r2_summary.tsv"), "paired_bootstrap": _rel(RESULTS / "paired_bootstrap.tsv"), "null_per_draw": _rel(RESULTS / "null_per_draw.tsv"), "trajectory_summary": _rel(RESULTS / "trajectory_summary.tsv")},
    }
    _write_json(RESULTS / "evaluation.json", result)
    plot_paths = _plot(real_results, trajectory_rows)
    result["plot_paths"] = plot_paths
    _write_json(RESULTS / "evaluation.json", result)
    _artifact_hashes()
    _update_state(phase_opened=False, reference_opened=True, status="complete", output_evaluation_sha256=_sha(RESULTS / "evaluation.json"))
    return result


def _old036_regression(data: Any, reference: np.ndarray, masks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if _sha(OLD036_ANCHOR_PATH) != OLD036_ANCHOR_SHA256 or _sha(OLD038_TSV) != OLD038_TSV_SHA256:
        raise RuntimeError("old036/038 regression input hash mismatch")
    tracks = _load_tracks(OLD036_ANCHOR_PATH, "3dg", tuple(data.chromosome_names))
    anchor = np.full((2, data.n_loci, 3), np.nan, dtype=np.float64)
    for ci, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(ci)
        positions = data.locus_bin[slc] * int(data.bin_size)
        for copy, suffix in enumerate(("a", "b")):
            rows = tracks.get("c%02d%s" % (ci + 1, suffix), {})
            for local, position in enumerate(positions):
                if int(position) in rows:
                    anchor[copy, slc.start + local] = rows[int(position)]
    if not np.isfinite(anchor).all():
        raise RuntimeError("old036 anchor is incomplete")
    new_result = _r2_result(anchor, reference, data, masks)
    new_by_chr = {row["chromosome"]: row for row in new_result["per_chromosome"]}
    with OLD038_TSV.open("r", encoding="utf-8", newline="") as handle:
        old_rows = list(csv.DictReader(handle, delimiter="\t"))
    fields = ("n_bins", "n_total_non_diagonal_pairs", "n_common_pairs_frozen", "rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat", "direct", "swapped", "matched", "cross", "contrast", "margin_mat", "margin_pat", "minmargin")
    mismatches = []
    for old in old_rows:
        name = str(old["chromosome"])
        row = new_by_chr.get(name)
        if row is None:
            mismatches.append({"chromosome": name, "field": "row_missing"})
            continue
        metric = row["metrics"]["spearman"]
        mask = masks[name]
        values = {"n_bins": len(mask["positions"]), "n_total_non_diagonal_pairs": mask["n_total_non_diagonal_pairs"], "n_common_pairs_frozen": mask["n_common_pairs"],
                  "rho_A_mat": metric["rho"]["A_mat"], "rho_A_pat": metric["rho"]["A_pat"], "rho_B_mat": metric["rho"]["B_mat"], "rho_B_pat": metric["rho"]["B_pat"],
                  "direct": metric["direct"], "swapped": metric["swapped"], "matched": metric["matched"], "cross": metric["cross"], "contrast": metric["contrast"],
                  "margin_mat": metric["matched_mat_margin"], "margin_pat": metric["matched_pat_margin"], "minmargin": metric["min_margin"]}
        for field in fields:
            expected = old[field]
            actual = values[field]
            if field in ("n_bins", "n_total_non_diagonal_pairs", "n_common_pairs_frozen"):
                if int(actual) != int(expected):
                    mismatches.append({"chromosome": name, "field": field, "old": expected, "new": actual})
            else:
                if actual is None or abs(float(actual) - float(expected)) > 1e-12:
                    mismatches.append({"chromosome": name, "field": field, "old": expected, "new": actual, "abs_diff": None if actual is None else abs(float(actual) - float(expected))})
        if str(metric["orientation"]) != str(old["orientation"]):
            mismatches.append({"chromosome": name, "field": "orientation", "old": old["orientation"], "new": metric["orientation"]})
    result = {"schema": "p9016-real-only-old036-spearman-regression-v1", "status": "PASS" if not mismatches else "FAIL", "numeric_atol": 1e-12, "numeric_rtol": 0.0, "mismatch_count": len(mismatches), "mismatches": mismatches[:100], "old_anchor_sha256": OLD036_ANCHOR_SHA256, "old038_tsv_sha256": OLD038_TSV_SHA256, "not_used_for_selection": True, "reference_opened_before_regression": True}
    _write_json(RESULTS / "old036_spearman_regression.json", result)
    if mismatches:
        raise RuntimeError("old036 Spearman regression failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "evaluate"))
    args = parser.parse_args()
    output = validate() if args.command == "validate" else evaluate()
    print(json.dumps({"status": output["status"], "reference_opened": output.get("reference_opened", False), "candidate_count": output.get("candidate_count"), "null_count": output.get("null_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
