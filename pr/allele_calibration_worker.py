"""哈希锁定的 synthetic R2 worker 和四进程调度器。

准备 artifact 由 :mod:`pr.allele_calibration` 持有，这里绝不重写。task 只包含 prepare root、fixture/model identifiers、显式 solver budget 和已发布 protocol hash。worker 不接收或打开 truth 路径。``run-all`` 在提交任何 optimizer 调用前写出 freeze 记录，保留 20 个 arms 中的每一个，并且只在所有 terminal 输出完成 hash lock 后创建 evaluator candidate manifest。
"""
from __future__ import annotations

import os

# 在导入 numpy 或 objective 前设置 numerical-library thread limits。
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
for _key, _value in _THREAD_ENV.items():
    os.environ[_key] = _value

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from multiprocessing import get_context
from typing import Any, Mapping, Sequence

import numpy as np

from . import allele_calibration as prepare
from .allele_models import MODEL_IDS, model_spec, validate_physical_for_model
from .paired_run import FitConfig, load_paired_start, plan_paired_runs, run_one_fit, write_coordinates


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_JSON = ROOT / "docs" / "POST020_ALLELE_ABLATION_PROTOCOL.json"
PROTOCOL_MD = ROOT / "docs" / "POST020_ALLELE_ABLATION_PROTOCOL.md"
PREPARE_SOURCE_MANIFEST = "provenance/source_manifest.json"
PREPARED_MANIFEST = "work/prepared_manifest.json"
FREEZE_RELATIVE = Path("freeze") / "POST020_ALLELE_ABLATION_FREEZE.json"
FIT_MANIFEST_RELATIVE = Path("results") / "fit_manifest.json"
CANDIDATE_MANIFEST_RELATIVE = Path("results") / "candidate_manifest.json"
WORKER_SCHEMA = "p9016-r2-allele-calibration-worker-task-v1"
FIT_MANIFEST_SCHEMA = "p9016-r2-allele-calibration-fit-manifest-v1"
FREEZE_SCHEMA = "p9016-r2-allele-calibration-fit-freeze-v1"
REGISTRY_PATH = ROOT / "test_res" / "POST020_ALLELE_ABLATION_ARTIFACT_REGISTRY.jsonl"
MAX_WORKERS = 4
MIN_AVAILABLE_BYTES = 32 * (1 << 30)
TERMINAL_STATUSES = {
    "solver_converged", "budget_not_converged", "solver_failed",
    "failed_source_snapshot", "failed_input", "failed_nonfinite", "failed_exception",
    "failed_worker_exception",
}
FINITE_ENDPOINT_STATUSES = {"solver_converged", "budget_not_converged", "solver_failed"}
NONTERMINAL_STATUSES = {"pending", "submitted", "running", "blocked_memavailable", "deferred_claim"}
ALL_STATUSES = TERMINAL_STATUSES | NONTERMINAL_STATUSES


class WorkerError(RuntimeError):
    """无法履行 synthetic worker 契约时抛出。"""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise WorkerError("nonfinite value cannot enter worker JSON")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorkerError("missing JSON artifact: %s" % path) from exc
    except json.JSONDecodeError as exc:
        raise WorkerError("invalid JSON artifact: %s" % path) from exc
    if not isinstance(value, dict):
        raise WorkerError("JSON root must be an object: %s" % path)
    return value


def _write_json(path: Path, payload: Mapping[str, Any], *, replace_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.%s.tmp" % (path.name, os.getpid(), time.time_ns()))
    encoded = json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if replace_existing:
            os.replace(temporary, path)
        else:
            if path.exists():
                raise FileExistsError(path)
            os.rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_text_exact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz_exact(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8", order="C")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _available_memory_bytes() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024
    return None


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_inside(path: str | Path, root: Path) -> Path:
    candidate = (root / Path(path)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise WorkerError("path escapes run directory: %s" % path) from exc
    return candidate


def _forbidden_task_values(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in ("truth", "reference", "phase", "eval_truth")):
                found.append(prefix + str(key))
            found.extend(_forbidden_task_values(item, prefix + str(key) + "."))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_task_values(item, prefix + str(index) + "."))
    elif isinstance(value, str):
        text = value.lower()
        if any(token in text for token in ("eval_truth", "truth_path", "truth_file", "truth_npz")):
            found.append(prefix.rstrip("."))
    return found


def _task_id(fixture_id: str, model_id: str) -> str:
    return "%s_%s" % (fixture_id, model_id.replace("-", "_"))


def _fit_config_dict() -> dict[str, Any]:
    return prepare.FIT_CONFIG.as_dict()


def _verify_prepare(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared_path = run_dir / PREPARED_MANIFEST
    prepared = _read_json(prepared_path)
    if prepared.get("schema") != prepare.PREPARED_SCHEMA:
        raise WorkerError("prepared manifest schema mismatch")
    if prepared.get("optimizer_started") or prepared.get("status") not in {"prepared_no_optimizer", "fit_running", "fit_complete"}:
        raise WorkerError("prepared manifest is not a permitted no-fit input")
    if len(prepared.get("runs", [])) != len(prepare.FIXTURES) * len(MODEL_IDS):
        raise WorkerError("prepared manifest does not contain 20 rows")
    config_path = run_dir / "config.json"
    if _sha256_file(config_path) != str(prepared.get("config_sha256")):
        raise WorkerError("prepared config hash changed")
    source_manifest_path = run_dir / PREPARE_SOURCE_MANIFEST
    source_manifest = _read_json(source_manifest_path)
    for relative, expected in source_manifest.get("code_sha256", {}).items():
        path = ROOT / str(relative)
        if not path.is_file() or _sha256_file(path) != str(expected):
            raise WorkerError("prepared source hash changed: %s" % relative)
    expected_keys = {(fixture_id, model_id) for fixture_id in prepare.FIXTURES for model_id in MODEL_IDS}
    actual_keys = {(str(row.get("fixture_id")), str(row.get("model_id"))) for row in prepared.get("runs", [])}
    if actual_keys != expected_keys:
        raise WorkerError("prepared manifest fixture/model coverage changed")
    return prepared, source_manifest


def _verify_protocol(protocol_json: Path, expected_sha256: str) -> dict[str, Any]:
    protocol_json = protocol_json.resolve()
    if not protocol_json.is_file():
        raise WorkerError("protocol JSON is missing: %s" % protocol_json)
    actual = _sha256_file(protocol_json)
    expected = str(expected_sha256).lower()
    if actual != expected:
        raise WorkerError("protocol release SHA mismatch: got %s expected %s" % (actual, expected))
    payload = _read_json(protocol_json)
    if payload.get("schema_version") != "post020-allele-ablation-protocol-v1":
        raise WorkerError("protocol schema is not the released post020 protocol")
    return {
        "path": str(protocol_json),
        "sha256": actual,
        "schema_version": payload.get("schema_version"),
        "protocol_version": payload.get("protocol_version"),
        "status": payload.get("status"),
    }


def build_tasks(run_dir: str | Path, *, protocol_sha256: str | None = None,
                release_id: str | None = None) -> list[dict[str, Any]]:
    """从 prepared manifest 精确构建 20 个无 truth 的 worker 任务。"""
    root = Path(run_dir).resolve()
    prepared, _source_manifest = _verify_prepare(root)
    by_key = {(str(row["fixture_id"]), str(row["model_id"])): row for row in prepared["runs"]}
    tasks: list[dict[str, Any]] = []
    for fixture_id in prepare.FIXTURES:
        for model_id in MODEL_IDS:
            row = by_key[(fixture_id, model_id)]
            task: dict[str, Any] = {
                "schema": WORKER_SCHEMA,
                "task_id": _task_id(fixture_id, model_id),
                "prepare_root": str(root),
                "fixture_id": fixture_id,
                "model_id": model_id,
                "fit_config": _fit_config_dict(),
            }
            if protocol_sha256 is not None:
                task["protocol_release_sha256"] = str(protocol_sha256).lower()
            if release_id is not None:
                task["release_id"] = str(release_id)
            if row.get("fit_config") != _fit_config_dict():
                raise WorkerError("prepared fit config differs for %s/%s" % (fixture_id, model_id))
            forbidden = _forbidden_task_values(task)
            if forbidden:
                raise WorkerError("task exposes forbidden training values: %s" % forbidden)
            tasks.append(task)
    if len(tasks) != 20:
        raise WorkerError("expected 20 tasks")
    return tasks


def preflight(run_dir: str | Path) -> dict[str, Any]:
    """核验全部 worker 输入和 B 规划，不调用 ``run_one_fit``。"""
    root = Path(run_dir).resolve()
    prepared, source_manifest = _verify_prepare(root)
    tasks = build_tasks(root)
    starts: dict[str, str] = {}
    layer_hashes: dict[str, str] = {}
    for task in tasks:
        data, start, row = prepare.load_prepared_data(root, task["fixture_id"], task["model_id"])
        plan = plan_paired_runs(data, [start], model_ids=(task["model_id"],))
        if len(plan) != 1 or not plan[0].get("fit_not_run"):
            raise WorkerError("B planning unexpectedly evaluated a fit for %s" % task["task_id"])
        starts[task["fixture_id"]] = start.coordinate_sha256
        layer_hashes[task["task_id"]] = str(row["data_sha256"])
    return {
        "schema": "p9016-r2-allele-calibration-worker-preflight-v1",
        "run_id": root.name,
        "status": "ready_for_released_worker",
        "task_count": len(tasks),
        "fit_started": False,
        "optimizer_started": False,
        "source_manifest_sha256": _sha256_file(root / PREPARE_SOURCE_MANIFEST),
        "worker_source_sha256": _sha256_file(Path(__file__).resolve()),
        "prepared_manifest_sha256": _sha256_file(root / PREPARED_MANIFEST),
        "shared_start_hashes": starts,
        "layer_hashes": layer_hashes,
        "source_code_hashes": dict(source_manifest.get("code_sha256", {})),
        "truth_access": "none",
    }


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        for block in iter(lambda: source_handle.read(1 << 20), b""):
            destination_handle.write(block)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())


def _fit_source_hashes(source_manifest: Mapping[str, Any]) -> dict[str, str]:
    paths = dict(source_manifest.get("code_sha256", {}))
    paths["pr/allele_calibration_worker.py"] = _sha256_file(Path(__file__).resolve())
    for relative in tuple(paths):
        path = ROOT / relative
        if not path.is_file() or _sha256_file(path) != str(paths[relative]):
            raise WorkerError("fit source hash changed: %s" % relative)
    return paths


def _append_registry(record: Mapping[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def create_freeze(run_dir: str | Path, *, protocol_json: str | Path = PROTOCOL_JSON,
                  protocol_sha256: str, release_id: str | None = None) -> dict[str, Any]:
    """在任何 optimizer 进程启动前写出 fit freeze/source 快照。"""
    root = Path(run_dir).resolve()
    prepared, source_manifest = _verify_prepare(root)
    protocol = _verify_protocol(Path(protocol_json), protocol_sha256)
    freeze_dir = root / "freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = root / FREEZE_RELATIVE
    if freeze_path.exists():
        existing = _read_json(freeze_path)
        if existing.get("protocol_release", {}).get("sha256") != protocol["sha256"]:
            raise WorkerError("existing freeze has a different protocol release")
        return existing
    prepare_snapshot_path = freeze_dir / "prepare_source_manifest.json"
    _copy_exact(root / PREPARE_SOURCE_MANIFEST, prepare_snapshot_path)
    fit_sources = _fit_source_hashes(source_manifest)
    fit_source_path = freeze_dir / "fit_source_hashes.json"
    _write_json(fit_source_path, {
        "schema": "p9016-r2-allele-calibration-fit-source-hashes-v1",
        "source_hashes": fit_sources,
        "worker_source_sha256": fit_sources["pr/allele_calibration_worker.py"],
    })
    task_inventory = []
    for row in prepared["runs"]:
        layer = _resolve_inside(row["data_path"], root)
        start = _resolve_inside(row["start_path"], root)
        task_inventory.append({
            "fixture_id": row["fixture_id"],
            "model_id": row["model_id"],
            "data_path": _relative(layer, root),
            "data_sha256": _sha256_file(layer),
            "start_path": _relative(start, root),
            "start_file_sha256": _sha256_file(start),
            "start_coordinate_sha256": row["initial_coordinate_sha256"],
        })
    protocol_documents = {"json": protocol}
    for label, path in (("md", PROTOCOL_MD), ("plan", ROOT / "docs" / "PLAN-post020-allele-signal.md")):
        if path.is_file():
            protocol_documents[label] = {"path": str(path.resolve()), "sha256": _sha256_file(path)}
    payload = {
        "schema": FREEZE_SCHEMA,
        "status": "fit_authorized_before_execution",
        "run_id": root.name,
        "created_utc": _utc_now(),
        "release_id": release_id,
        "protocol_release": protocol,
        "protocol_documents": protocol_documents,
        "prepare": {
            "config_path": _relative(root / "config.json", root),
            "config_sha256": _sha256_file(root / "config.json"),
            "prepared_manifest_path": PREPARED_MANIFEST,
            "prepared_manifest_sha256": _sha256_file(root / PREPARED_MANIFEST),
            "source_manifest_path": PREPARE_SOURCE_MANIFEST,
            "source_manifest_sha256": _sha256_file(root / PREPARE_SOURCE_MANIFEST),
            "source_snapshot_copy": _relative(prepare_snapshot_path, root),
            "source_snapshot_copy_sha256": _sha256_file(prepare_snapshot_path),
            "source_code_sha256": dict(source_manifest.get("code_sha256", {})),
        },
        "fit_source": {
            "path": _relative(fit_source_path, root),
            "sha256": _sha256_file(fit_source_path),
            "source_hashes": fit_sources,
        },
        "worker_inputs": {
            "task_count": 20,
            "fit_config": _fit_config_dict(),
            "all_eligible_pairs_and_exposure_locked": True,
            "truth_access": "none",
            "inventory": task_inventory,
        },
        "resource_policy": {
            "max_parallel_workers": MAX_WORKERS,
            "min_mem_available_bytes": MIN_AVAILABLE_BYTES,
            "omp_threads": 1,
            "blas_threads": 1,
        },
        "execution": {
            "fit_started": False,
            "optimizer_started": False,
            "reference_or_phase_read": False,
        },
    }
    _write_json(freeze_path, payload)
    _append_registry({
        "record_id": "%s-fit-freeze-%s" % (root.name, fit_sources["pr/allele_calibration_worker.py"][:12]),
        "run_id": root.name,
        "scope": "synthetic",
        "variant_id": "all-4x5",
        "fixture_or_bundle_id": "N1,N2,P1,P2",
        "status": payload["status"],
        "path": _relative(freeze_path, ROOT),
        "sha256": _sha256_file(freeze_path),
        "source_hashes": fit_sources,
        "timestamp": payload["created_utc"],
    })
    return payload


def _log_event(log_path: Path, event_name: str, **fields: Any) -> None:
    payload = {"utc": _utc_now(), "event": event_name, **fields}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_checkpoint(root: Path, attempt_root: Path, model_id: str,
                      fixture_id: str, checkpoint: Any) -> dict[str, Any]:
    checkpoint_name = "iter-%06d" % int(checkpoint.iteration)
    npz_path = attempt_root / "checkpoints" / (checkpoint_name + ".npz")
    json_path = attempt_root / "checkpoints" / (checkpoint_name + ".json")
    theta = np.asarray(checkpoint.theta, dtype=np.float64)
    raw_coordinates = np.asarray(checkpoint.y, dtype=np.float64)
    coordinates = np.asarray(checkpoint.coordinates, dtype=np.float64)
    _write_npz_exact(npz_path, theta=theta, raw_coordinates=raw_coordinates, coordinates=coordinates)
    payload = {
        "schema": "p9016-r2-allele-calibration-checkpoint-v1",
        "fixture_id": fixture_id,
        "model_id": model_id,
        "iteration": int(checkpoint.iteration),
        "nfev": int(checkpoint.nfev),
        "elapsed_seconds": float(checkpoint.elapsed_seconds),
        "fun": float(checkpoint.fun),
        "p": float(checkpoint.p),
        "gradient_l2": float(checkpoint.gradient_norm),
        "components": dict(checkpoint.components),
        "npz_path": _relative(npz_path, root),
        "npz_sha256": _sha256_file(npz_path),
        "theta_sha256": _sha256_array(theta),
        "raw_coordinates_sha256": _sha256_array(raw_coordinates),
        "coordinates_sha256": _sha256_array(coordinates),
    }
    _write_json(json_path, payload)
    return {**payload, "path": _relative(json_path, root), "sha256": _sha256_file(json_path)}


def _map_diagnostics_for_result(result: Any) -> dict[str, Any]:
    """在最终 payload 中保留 fit 末端和 terminal-objective map 审计。"""
    fit_diagnostics = getattr(result, "map_diagnostics", {})
    if fit_diagnostics is None:
        fit_diagnostics = {}
    objective = getattr(result, "objective", None)
    objective_method = getattr(objective, "map_diagnostics", None)
    objective_diagnostics = dict(objective_method()) if callable(objective_method) else {}
    return {
        "fit_result_map_diagnostics": dict(fit_diagnostics),
        "terminal_objective_map_diagnostics": objective_diagnostics,
    }


def _stop_reason(result: Any, fit_config: FitConfig) -> str:
    if bool(result.success):
        return "solver_converged"
    if int(result.nfev) >= int(fit_config.maxfun):
        return "budget_not_converged_maxfun"
    if int(result.nit) >= int(fit_config.maxiter):
        return "budget_not_converged_maxiter"
    return "solver_failed"


def _terminal_status(stop_reason: str) -> str:
    if stop_reason == "solver_converged":
        return "solver_converged"
    if stop_reason.startswith("budget_not_converged"):
        return "budget_not_converged"
    return "solver_failed"


def _failure_status(exc: BaseException) -> str:
    text = str(exc).lower()
    if "nonfinite" in text or isinstance(exc, FloatingPointError):
        return "failed_nonfinite"
    if "source" in text or "hash" in text or "protocol" in text:
        return "failed_source_snapshot"
    if any(token in text for token in ("coordinate", "unit-ball", "prepared", "input", "layer", "start")):
        return "failed_input"
    return "failed_exception"


def _claim_path(job_root: Path) -> Path:
    return job_root / ".claim"


def _status_path(job_root: Path) -> Path:
    return job_root / "status.json"


def _read_existing_status(job_root: Path) -> dict[str, Any] | None:
    path = _status_path(job_root)
    return _read_json(path) if path.is_file() else None


def _new_attempt(job_root: Path) -> tuple[Path, int]:
    attempts_root = job_root / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in attempts_root.glob("attempt-*"):
        try:
            numbers.append(int(path.name.split("-")[-1]))
        except ValueError:
            continue
    number = max(numbers, default=0) + 1
    attempt = attempts_root / ("attempt-%03d" % number)
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt, number


def _verify_freeze_for_task(root: Path, task: Mapping[str, Any]) -> dict[str, Any]:
    freeze_path = root / FREEZE_RELATIVE
    freeze = _read_json(freeze_path)
    release_sha = str(task.get("protocol_release_sha256", "")).lower()
    frozen_sha = str(freeze.get("protocol_release", {}).get("sha256", "")).lower()
    if not release_sha or release_sha != frozen_sha:
        raise WorkerError("task protocol release differs from fit freeze")
    protocol_path = Path(str(freeze["protocol_release"]["path"])).resolve()
    if _sha256_file(protocol_path) != frozen_sha:
        raise WorkerError("released protocol changed after freeze")
    if freeze.get("worker_inputs", {}).get("task_count") != 20:
        raise WorkerError("fit freeze task count is not 20")
    return freeze


def _validate_task(task: Mapping[str, Any]) -> None:
    required = {"schema", "task_id", "prepare_root", "fixture_id", "model_id", "fit_config", "protocol_release_sha256"}
    missing = required - set(task)
    if missing:
        raise WorkerError("task missing required fields: %s" % sorted(missing))
    if task.get("schema") != WORKER_SCHEMA:
        raise WorkerError("worker task schema mismatch")
    fixture_id = str(task["fixture_id"])
    model_id = str(task["model_id"])
    if fixture_id not in prepare.FIXTURES or model_id not in MODEL_IDS:
        raise WorkerError("unknown fixture/model task")
    if str(task["task_id"]) != _task_id(fixture_id, model_id):
        raise WorkerError("task id does not match fixture/model")
    if dict(task["fit_config"]) != _fit_config_dict():
        raise WorkerError("task fit budget differs from frozen synthetic budget")
    forbidden = _forbidden_task_values(task)
    if forbidden:
        raise WorkerError("worker task exposes forbidden values: %s" % forbidden)


def run_task(task_path: str | Path) -> dict[str, Any]:
    """运行一个哈希锁定的 synthetic task，并持久化 terminal 结果。"""
    task_file = Path(task_path).resolve()
    task = _read_json(task_file)
    _validate_task(task)
    root = Path(str(task["prepare_root"])).resolve()
    task_id = str(task["task_id"])
    job_root = root / "fit" / "jobs" / task_id
    expected_task_path = (job_root / "task.json").resolve()
    if task_file != expected_task_path:
        raise WorkerError("task path is not the scheduler-owned task config")
    fit_manifest_path = root / FIT_MANIFEST_RELATIVE
    fit_manifest = _read_json(fit_manifest_path)
    rows = {str(row["task_id"]): row for row in fit_manifest.get("jobs", [])}
    manifest_row = rows.get(task_id)
    if manifest_row is None or manifest_row.get("task_sha256") != _sha256_file(task_file):
        raise WorkerError("task is not the hash-locked fit manifest version")
    existing = _read_existing_status(job_root)
    if existing is not None and existing.get("status") in TERMINAL_STATUSES:
        return existing
    claim = _claim_path(job_root)
    try:
        with claim.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "utc": _utc_now()}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return {"task_id": task_id, "status": "deferred_claim", "fit_called": False, "selection_eligible": False}
    started_clock = time.perf_counter()
    started_utc = _utc_now()
    rss_start = _rss_bytes()
    attempt_root: Path | None = None
    log_path: Path | None = None
    fit_called = False
    try:
        attempt_root, attempt_number = _new_attempt(job_root)
        log_path = attempt_root / "worker.jsonl"
        _write_json(attempt_root / "status.json", {
            "schema": "p9016-r2-allele-calibration-fit-status-v1",
            "status": "running",
            "task_id": task_id,
            "fixture_id": task["fixture_id"],
            "model_id": task["model_id"],
            "attempt": attempt_number,
            "pid": os.getpid(),
            "started_utc": started_utc,
            "fit_called": False,
            "selection_eligible": False,
        })
        _write_json(_status_path(job_root), {
            "task_id": task_id,
            "status": "running",
            "attempt": attempt_number,
            "attempt_path": _relative(attempt_root, root),
            "fit_called": False,
            "selection_eligible": False,
            "updated_utc": _utc_now(),
        }, replace_existing=True)
        _log_event(log_path, "worker_started", task_id=task_id, attempt=attempt_number,
                   rss_start_bytes=rss_start, mem_available_bytes=_available_memory_bytes(),
                   thread_env=_THREAD_ENV)
        os.environ.update(_THREAD_ENV)
        freeze = _verify_freeze_for_task(root, task)
        prepared, source_manifest = _verify_prepare(root)
        data, start, prepared_row = prepare.load_prepared_data(root, task["fixture_id"], task["model_id"])
        if prepared_row.get("data_sha256") != _sha256_file(_resolve_inside(prepared_row["data_path"], root)):
            raise WorkerError("prepared layer hash changed during worker load")
        plan = plan_paired_runs(data, [start], model_ids=(task["model_id"],))
        if len(plan) != 1 or not plan[0].get("fit_not_run"):
            raise WorkerError("worker preflight plan is not fit-free")
        if start.coordinate_sha256 != prepared_row.get("initial_coordinate_sha256"):
            raise WorkerError("prepared start hash changed during worker load")
        fit_config = FitConfig(**dict(task["fit_config"]))
        if fit_config.as_dict() != _fit_config_dict():
            raise WorkerError("worker budget differs from frozen config")
        fit_sources = _fit_source_hashes(source_manifest)
        _log_event(log_path, "inputs_verified", data_sha256=prepared_row["data_sha256"],
                   start_coordinate_sha256=start.coordinate_sha256,
                   source_manifest_sha256=_sha256_file(root / PREPARE_SOURCE_MANIFEST),
                   freeze_sha256=_sha256_file(root / FREEZE_RELATIVE),
                   fit_source_hashes=fit_sources)
        checkpoints: list[dict[str, Any]] = []

        def checkpoint_hook(checkpoint: Any) -> None:
            record = _write_checkpoint(root, attempt_root, task["model_id"], task["fixture_id"], checkpoint)
            checkpoints.append(record)
            _log_event(log_path, "checkpoint", iteration=record["iteration"], nfev=record["nfev"],
                       path=record["path"], sha256=record["sha256"])

        def accepted_callback(entry: dict[str, Any]) -> None:
            _log_event(log_path, "accepted", **entry)

        _log_event(log_path, "fit_start", fit=fit_config.as_dict())
        fit_called = True
        _write_json(_status_path(job_root), {
            "task_id": task_id,
            "status": "running",
            "attempt": attempt_number,
            "attempt_path": _relative(attempt_root, root),
            "fit_called": True,
            "selection_eligible": False,
            "updated_utc": _utc_now(),
        }, replace_existing=True)
        result = run_one_fit(
            data, start, task["model_id"], fit_config,
            checkpoint_hook=checkpoint_hook, callback=accepted_callback,
        )
        validate_physical_for_model(task["model_id"], result.coordinates)
        endpoint_value, endpoint_gradient, endpoint_components = result.objective.evaluate(
            result.theta, need_gradient=True)
        endpoint_gradient = np.asarray(endpoint_gradient, dtype=np.float64)
        finite = (
            math.isfinite(float(endpoint_value))
            and np.all(np.isfinite(result.theta))
            and np.all(np.isfinite(result.coordinates))
            and np.all(np.isfinite(endpoint_gradient))
        )
        if not finite:
            raise WorkerError("terminal endpoint contains a nonfinite value")
        gradient_l2 = float(np.linalg.norm(endpoint_gradient))
        gradient_inf = float(np.max(np.abs(endpoint_gradient)))
        if not math.isfinite(gradient_l2) or not math.isfinite(gradient_inf):
            raise WorkerError("terminal gradient norm is nonfinite")
        result_map_diagnostics = _map_diagnostics_for_result(result)
        theta_path = attempt_root / "final_theta.txt"
        _write_text_exact(theta_path, "".join("%.17g\n" % value for value in result.theta))
        state_path = attempt_root / "final_state.npz"
        _write_npz_exact(
            state_path,
            theta=np.asarray(result.theta, dtype=np.float64),
            raw_coordinates=np.asarray(result.raw_coordinates, dtype=np.float64),
            coordinates=np.asarray(result.coordinates, dtype=np.float64),
            endpoint_gradient=endpoint_gradient,
        )
        coordinates_path = attempt_root / "final_coordinates.3dg"
        coordinate_record = write_coordinates(coordinates_path, data, task["model_id"], result.coordinates)
        coordinate_record["path"] = _relative(coordinates_path, root)
        stop_reason = _stop_reason(result, fit_config)
        terminal_status = _terminal_status(stop_reason)
        final_payload: dict[str, Any] = {
            "schema": "p9016-r2-allele-calibration-fit-final-v1",
            "status": terminal_status,
            "task_id": task_id,
            "run_id": root.name,
            "fixture_id": task["fixture_id"],
            "model_id": task["model_id"],
            "attempt": attempt_number,
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started_clock),
            "rss_start_bytes": int(rss_start),
            "rss_max_bytes": int(_rss_bytes()),
            "mem_available_end_bytes": _available_memory_bytes(),
            "protocol_release_sha256": task["protocol_release_sha256"],
            "release_id": task.get("release_id"),
            "prepare_source_manifest": {
                "path": PREPARE_SOURCE_MANIFEST,
                "sha256": _sha256_file(root / PREPARE_SOURCE_MANIFEST),
            },
            "fit_source_hashes": fit_sources,
            "map_diagnostics": result_map_diagnostics,
            "freeze": {
                "path": _relative(root / FREEZE_RELATIVE, root),
                "sha256": _sha256_file(root / FREEZE_RELATIVE),
            },
            "input": {
                "data_path": prepared_row["data_path"],
                "data_sha256": prepared_row["data_sha256"],
                "start_path": prepared_row["start_path"],
                "start_file_sha256": _sha256_file(_resolve_inside(prepared_row["start_path"], root)),
                "start_coordinate_sha256": start.coordinate_sha256,
                "initial_total": float(result.initial_total),
                "initial_roundtrip_max_abs": float(result.initial_roundtrip_max_abs),
            },
            "solver": {
                "fit": fit_config.as_dict(),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "nit": int(result.nit),
                "actual_nfev": int(result.nfev),
                "scipy_nfev": int(result.scipy_nfev),
                "njev": int(result.njev),
                "actual_stop_reason": stop_reason,
            },
            "final": {
                "fun": float(endpoint_value),
                "p": float(result.p),
                "q": float(result.theta[-1]),
                "components": dict(endpoint_components),
                "gradient_l2": gradient_l2,
                "gradient_inf": gradient_inf,
                "endpoint_eval_with_gradient": True,
                "endpoint_value_exact_match": bool(float(endpoint_value) == float(result.fun)),
                "theta_sha256": _sha256_array(result.theta),
                "coordinates_sha256": _sha256_array(result.coordinates),
                "theta_text_path": _relative(theta_path, root),
                "theta_text_sha256": _sha256_file(theta_path),
                "state_npz_path": _relative(state_path, root),
                "state_npz_sha256": _sha256_file(state_path),
                "coordinates_file": coordinate_record,
            },
            "history": result.history,
            "checkpoints": checkpoints,
            "selection_eligible": terminal_status in FINITE_ENDPOINT_STATUSES,
            "solver_abnormal": terminal_status != "solver_converged",
            "reference_or_phase_read": False,
        }
        final_path = attempt_root / "final.json"
        _write_json(final_path, final_payload)
        final_json_sha256 = _sha256_file(final_path)
        final_payload["final_json_sha256"] = final_json_sha256
        _write_json(attempt_root / "status.json", final_payload, replace_existing=True)
        job_status = {
            "task_id": task_id,
            "status": terminal_status,
            "attempt": attempt_number,
            "attempt_path": _relative(attempt_root, root),
            "final_json_path": _relative(final_path, root),
            "final_json_sha256": final_json_sha256,
            "coordinate_path": coordinate_record["path"],
            "coordinate_file_sha256": coordinate_record["sha256"],
            "fit_called": True,
            "selection_eligible": bool(final_payload["selection_eligible"]),
            "updated_utc": _utc_now(),
        }
        _write_json(_status_path(job_root), job_status, replace_existing=True)
        _log_event(log_path, "fit_terminal", status=terminal_status,
                   actual_stop_reason=stop_reason, actual_nfev=int(result.nfev), njev=int(result.njev),
                   gradient_l2=gradient_l2, gradient_inf=gradient_inf)
        return {**final_payload, **job_status}
    except Exception as exc:
        status = _failure_status(exc)
        failure_payload: dict[str, Any] = {
            "schema": "p9016-r2-allele-calibration-fit-failure-v1",
            "status": status,
            "task_id": task_id,
            "fixture_id": task.get("fixture_id"),
            "model_id": task.get("model_id"),
            "started_utc": started_utc,
            "ended_utc": _utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started_clock),
            "rss_start_bytes": int(rss_start),
            "rss_max_bytes": int(_rss_bytes()),
            "fit_called": bool(fit_called),
            "selection_eligible": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if attempt_root is not None:
            failure_payload["attempt"] = int(attempt_root.name.split("-")[-1])
            failure_payload["attempt_path"] = _relative(attempt_root, root)
            _write_json(attempt_root / "failure.json", failure_payload)
            _write_json(attempt_root / "status.json", failure_payload, replace_existing=True)
            _write_json(_status_path(job_root), failure_payload, replace_existing=True)
            if log_path is not None:
                _log_event(log_path, "worker_failed", status=status,
                           error_type=type(exc).__name__, error=str(exc))
        return failure_payload
    finally:
        if claim.exists():
            claim.unlink()


def _worker_entry(task_path: str) -> dict[str, Any]:
    return run_task(task_path)


def _initial_fit_manifest(root: Path, tasks: Sequence[Mapping[str, Any]], freeze: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for task in tasks:
        task_path = root / "fit" / "jobs" / str(task["task_id"]) / "task.json"
        rows.append({
            "task_id": task["task_id"],
            "fixture_id": task["fixture_id"],
            "model_id": task["model_id"],
            "task_path": _relative(task_path, root),
            "task_sha256": _sha256_file(task_path),
            "status": "pending",
            "fit_called": False,
            "selection_eligible": False,
        })
    return {
        "schema": FIT_MANIFEST_SCHEMA,
        "run_id": root.name,
        "status": "pending",
        "created_utc": _utc_now(),
        "protocol_release_sha256": freeze["protocol_release"]["sha256"],
        "release_id": freeze.get("release_id"),
        "freeze_path": _relative(root / FREEZE_RELATIVE, root),
        "freeze_sha256": _sha256_file(root / FREEZE_RELATIVE),
        "worker_source_sha256": _sha256_file(Path(__file__).resolve()),
        "prepared_manifest_sha256": _sha256_file(root / PREPARED_MANIFEST),
        "fit_config": _fit_config_dict(),
        "max_workers": MAX_WORKERS,
        "status_sets": {
            "terminal": sorted(TERMINAL_STATUSES),
            "finite_endpoint": sorted(FINITE_ENDPOINT_STATUSES),
            "solver_converged": ["solver_converged"],
            "nonterminal": sorted(NONTERMINAL_STATUSES),
        },
        "all_registered_arms_preserved": True,
        "jobs": rows,
        "candidate_manifest": None,
        "fit_started": False,
        "optimizer_started": False,
        "actual_fit_count": 0,
        "all_endpoints_hash_locked": False,
    }


def _refresh_manifest_from_jobs(root: Path, manifest: dict[str, Any]) -> None:
    """复制 factual child status，不将 submission 或 claims 当作 fit。"""
    by_task = {str(row["task_id"]): row for row in manifest.get("jobs", [])}
    fields = (
        "status", "fit_called", "selection_eligible", "attempt", "attempt_path",
        "final_json_path", "final_json_sha256", "coordinate_path", "coordinate_file_sha256",
        "error_type", "error", "elapsed_seconds", "rss_start_bytes", "rss_max_bytes",
    )
    for task_id, row in by_task.items():
        status_path = root / "fit" / "jobs" / task_id / "status.json"
        if not status_path.is_file():
            continue
        try:
            child = _read_json(status_path)
        except WorkerError:
            continue
        if child.get("status") not in ALL_STATUSES | {"running"}:
            continue
        for field in fields:
            if field in child:
                row[field] = child[field]
        row["child_status_updated_utc"] = child.get("updated_utc", _utc_now())


def _manifest_progress(root: Path, manifest: dict[str, Any], status: str | None = None) -> dict[str, Any]:
    _refresh_manifest_from_jobs(root, manifest)
    jobs = manifest.get("jobs", [])
    manifest["actual_fit_count"] = int(sum(bool(row.get("fit_called")) for row in jobs))
    manifest["terminal_job_count"] = int(sum(row.get("status") in TERMINAL_STATUSES for row in jobs))
    # 已提交 process 不是 optimizer start；child status 才是事实来源。
    manifest["fit_started"] = bool(
        manifest.get("fit_started")
        or manifest["actual_fit_count"]
        or any(row.get("status") == "running" and row.get("fit_called") for row in jobs)
    )
    manifest["optimizer_started"] = bool(manifest["fit_started"])
    if status is not None:
        manifest["status"] = status
    return manifest


def _write_scheduler_event(root: Path, event_name: str, **fields: Any) -> None:
    _log_event(root / "fit" / "scheduler.jsonl", event_name, **fields)


def _write_candidate_manifest(root: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[Path, str]:
    records = []
    for row in rows:
        if row.get("status") not in TERMINAL_STATUSES or not row.get("selection_eligible"):
            raise WorkerError("cannot lock a candidate from a non-eligible fit row")
        final_path = _resolve_inside(str(row["final_json_path"]), root)
        final = _read_json(final_path)
        coordinate = final.get("final", {}).get("coordinates_file", {})
        records.append({
            "fixture_id": row["fixture_id"],
            "model_id": row["model_id"],
            "coordinate_path": coordinate["path"],
            "coordinate_sha256": coordinate["sha256"],
            "p": final.get("final", {}).get("p"),
            "fit_status": row["status"],
            "final_json_path": row["final_json_path"],
            "final_json_sha256": row["final_json_sha256"],
        })
    records.sort(key=lambda item: (str(item["fixture_id"]), str(item["model_id"])))
    payload = {
        "schema": prepare.CANDIDATE_SCHEMA,
        "run_id": root.name,
        "status": "hash_locked_all_20_endpoints",
        "candidate_count": len(records),
        "candidates": records,
        "truth_access": "none",
    }
    path = root / CANDIDATE_MANIFEST_RELATIVE
    if path.exists():
        raise WorkerError("candidate manifest already exists")
    _write_json(path, payload)
    return path, _sha256_file(path)


def _scheduler_failure(root: Path, task: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    job_root = root / "fit" / "jobs" / str(task["task_id"])
    payload = {
        "schema": "p9016-r2-allele-calibration-fit-failure-v1",
        "status": "failed_worker_exception",
        "task_id": task["task_id"],
        "fixture_id": task["fixture_id"],
        "model_id": task["model_id"],
        "fit_called": False,
        "selection_eligible": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    _write_json(job_root / "scheduler_failure.json", payload)
    _write_json(_status_path(job_root), payload, replace_existing=True)
    return payload


def run_all(run_dir: str | Path, *, protocol_json: str | Path = PROTOCOL_JSON,
            protocol_sha256: str, release_id: str | None = None,
            max_workers: int = MAX_WORKERS) -> dict[str, Any]:
    """冻结并运行全部 20 个 synthetic tasks，最多使用四个 processes。"""
    if not isinstance(max_workers, int) or max_workers < 1 or max_workers > MAX_WORKERS:
        raise WorkerError("max_workers must be in [1,%d]" % MAX_WORKERS)
    root = Path(run_dir).resolve()
    preflight_result = preflight(root)
    freeze = create_freeze(root, protocol_json=protocol_json,
                           protocol_sha256=protocol_sha256, release_id=release_id)
    tasks = build_tasks(root, protocol_sha256=freeze["protocol_release"]["sha256"], release_id=release_id)
    fit_root = root / "fit"
    fit_manifest_path = root / FIT_MANIFEST_RELATIVE
    if fit_manifest_path.exists() or fit_root.exists() and any(fit_root.iterdir()):
        raise WorkerError("fit scheduler artifacts already exist; refusing to reuse run")
    for task in tasks:
        job_root = fit_root / "jobs" / str(task["task_id"])
        job_root.mkdir(parents=True, exist_ok=False)
        _write_json(job_root / "task.json", task)
    manifest = _initial_fit_manifest(root, tasks, freeze)
    _write_json(fit_manifest_path, manifest)
    _write_scheduler_event(root, "scheduler_created", task_count=len(tasks), max_workers=max_workers,
                           preflight_status=preflight_result["status"], freeze_sha256=manifest["freeze_sha256"])
    task_paths = {str(task["task_id"]): str(root / "fit" / "jobs" / str(task["task_id"]) / "task.json") for task in tasks}
    pending = list(tasks)
    running: dict[Any, Mapping[str, Any]] = {}
    blocked_reason: str | None = None
    process_context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=process_context) as pool:
        while pending or running:
            while pending and len(running) < max_workers:
                available = _available_memory_bytes()
                if available is not None and available < MIN_AVAILABLE_BYTES:
                    blocked_reason = "mem_available_below_32GiB"
                    _write_scheduler_event(root, "memory_gate_blocked", mem_available_bytes=available,
                                           pending_count=len(pending), running_count=len(running))
                    break
                task = pending.pop(0)
                future = pool.submit(_worker_entry, task_paths[str(task["task_id"])])
                running[future] = task
                submitted_row = next(row for row in manifest["jobs"] if row["task_id"] == task["task_id"])
                submitted_row["status"] = "submitted"
                submitted_row["fit_called"] = False
                submitted_row["selection_eligible"] = False
                submitted_row["submitted_utc"] = _utc_now()
                _manifest_progress(root, manifest, "running")
                _write_json(fit_manifest_path, manifest, replace_existing=True)
                _write_scheduler_event(root, "task_submitted", task_id=task["task_id"],
                                       running_count=len(running), pending_count=len(pending))
            if not running:
                if pending:
                    blocked_reason = blocked_reason or "scheduler_could_not_submit_pending_tasks"
                    break
                continue
            done, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = _scheduler_failure(root, task, exc)
                by_task = {str(row["task_id"]): row for row in manifest["jobs"]}
                row = by_task[str(task["task_id"])]
                for key in (
                    "status", "fit_called", "selection_eligible", "attempt", "attempt_path",
                    "final_json_path", "final_json_sha256", "coordinate_path", "coordinate_file_sha256",
                    "error_type", "error", "elapsed_seconds", "rss_start_bytes", "rss_max_bytes",
                ):
                    if key in result:
                        row[key] = result[key]
                row["updated_utc"] = _utc_now()
                _manifest_progress(root, manifest, "running" if pending or running else None)
                _write_json(fit_manifest_path, manifest, replace_existing=True)
                _write_scheduler_event(root, "task_terminal", task_id=task["task_id"],
                                       status=result.get("status"), fit_called=result.get("fit_called", False))
    if pending:
        _manifest_progress(root, manifest, "blocked_memavailable" if blocked_reason == "mem_available_below_32GiB" else "incomplete")
    else:
        eligible = all(
            row.get("status") in TERMINAL_STATUSES and row.get("selection_eligible") and row.get("final_json_sha256")
            for row in manifest["jobs"]
        )
        if eligible:
            candidate_path, candidate_sha = _write_candidate_manifest(root, manifest["jobs"])
            manifest["candidate_manifest"] = {"path": _relative(candidate_path, root), "sha256": candidate_sha}
            manifest["all_endpoints_hash_locked"] = True
            _manifest_progress(root, manifest, "complete_terminal")
            _write_scheduler_event(root, "all_endpoints_hash_locked", candidate_manifest_sha256=candidate_sha)
        else:
            all_terminal = all(row.get("status") in TERMINAL_STATUSES for row in manifest["jobs"])
            _manifest_progress(root, manifest, "complete_with_failures" if all_terminal else "incomplete")
    _write_json(fit_manifest_path, manifest, replace_existing=True)
    _write_scheduler_event(root, "scheduler_terminal", status=manifest["status"],
                           terminal_job_count=manifest.get("terminal_job_count"),
                           actual_fit_count=manifest.get("actual_fit_count"),
                           all_endpoints_hash_locked=manifest["all_endpoints_hash_locked"])
    return {
        "schema": "p9016-r2-allele-calibration-worker-run-v1",
        "run_id": root.name,
        "status": manifest["status"],
        "fit_manifest": _relative(fit_manifest_path, root),
        "candidate_manifest": manifest.get("candidate_manifest"),
        "planned_fit_count": 20,
        "actual_fit_count": manifest.get("actual_fit_count", 0),
        "terminal_job_count": manifest.get("terminal_job_count", 0),
        "all_endpoints_hash_locked": bool(manifest.get("all_endpoints_hash_locked")),
        "max_workers": max_workers,
        "preflight": preflight_result,
    }


def _lock_evaluator_source(root: Path) -> dict[str, Any]:
    """仅在 20 个 endpoint/output hashes 全部完成后锁定 evaluator bytes。"""
    fit_manifest_path = root / FIT_MANIFEST_RELATIVE
    fit_manifest = _read_json(fit_manifest_path)
    if fit_manifest.get("schema") != FIT_MANIFEST_SCHEMA or fit_manifest.get("status") != "complete_terminal":
        raise WorkerError("evaluation source cannot be locked before complete terminal fit manifest")
    if not fit_manifest.get("all_endpoints_hash_locked") or not isinstance(fit_manifest.get("candidate_manifest"), Mapping):
        raise WorkerError("evaluation source cannot be locked before all endpoint hashes")
    evaluator_path = ROOT / "pr" / "allele_calibration_evaluator.py"
    if not evaluator_path.is_file():
        raise WorkerError("corrected evaluator source is missing")
    payload = {
        "schema": "p9016-r2-allele-calibration-evaluator-source-freeze-v1",
        "status": "locked_after_all_endpoint_hashes",
        "run_id": root.name,
        "created_utc": _utc_now(),
        "fit_manifest_path": _relative(fit_manifest_path, root),
        "fit_manifest_sha256": _sha256_file(fit_manifest_path),
        "candidate_manifest": dict(fit_manifest["candidate_manifest"]),
        "evaluator_source": {
            "path": _relative(evaluator_path, ROOT),
            "sha256": _sha256_file(evaluator_path),
        },
        "truth_access_before_lock": False,
    }
    path = root / "freeze" / "evaluation_source_hashes.json"
    if path.exists():
        existing = _read_json(path)
        if existing.get("evaluator_source", {}).get("sha256") != payload["evaluator_source"]["sha256"]:
            raise WorkerError("evaluation source changed after its freeze")
        return existing
    _write_json(path, payload)
    _append_registry({
        "record_id": "%s-evaluation-source-freeze" % root.name,
        "run_id": root.name,
        "scope": "synthetic",
        "variant_id": "evaluation",
        "fixture_or_bundle_id": "all-4x5",
        "status": payload["status"],
        "path": _relative(path, ROOT),
        "sha256": _sha256_file(path),
        "source_hashes": {"pr/allele_calibration_evaluator.py": payload["evaluator_source"]["sha256"]},
        "timestamp": payload["created_utc"],
    })
    return payload


def evaluate_locked(run_dir: str | Path) -> dict[str, Any]:
    """锁定 evaluator bytes，并在 endpoint hashes 完成后进入修正后的 evaluation。"""
    root = Path(run_dir).resolve()
    _lock_evaluator_source(root)
    from .allele_calibration_evaluator import evaluate_run
    return evaluate_run(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight_parser = sub.add_parser("preflight", help="validate all 20 inputs without optimizer")
    preflight_parser.add_argument("--run-dir", required=True)
    freeze_parser = sub.add_parser("freeze", help="write release/source freeze without optimizer")
    freeze_parser.add_argument("--run-dir", required=True)
    freeze_parser.add_argument("--protocol-json", default=str(PROTOCOL_JSON))
    freeze_parser.add_argument("--protocol-sha256", required=True)
    freeze_parser.add_argument("--release-id", default=None)
    task_parser = sub.add_parser("run-task", help="run one released fit task")
    task_parser.add_argument("--task", required=True)
    all_parser = sub.add_parser("run-all", help="run 20 released synthetic fits with <=4 workers")
    all_parser.add_argument("--run-dir", required=True)
    all_parser.add_argument("--protocol-json", default=str(PROTOCOL_JSON))
    all_parser.add_argument("--protocol-sha256", required=True)
    all_parser.add_argument("--release-id", default=None)
    all_parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    eval_parser = sub.add_parser("evaluate", help="evaluate hash-locked endpoints")
    eval_parser.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.run_dir)
            print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
            return 0
        if args.command == "freeze":
            result = create_freeze(args.run_dir, protocol_json=args.protocol_json,
                                   protocol_sha256=args.protocol_sha256, release_id=args.release_id)
            print(json.dumps(_jsonable({"status": result["status"], "path": str(Path(args.run_dir).resolve() / FREEZE_RELATIVE),
                                        "protocol_sha256": result["protocol_release"]["sha256"]}), sort_keys=True))
            return 0
        if args.command == "run-task":
            result = run_task(args.task)
            print(json.dumps(_jsonable(result), sort_keys=True))
            return 0 if result.get("status") in TERMINAL_STATUSES else 1
        if args.command == "run-all":
            result = run_all(args.run_dir, protocol_json=args.protocol_json,
                             protocol_sha256=args.protocol_sha256, release_id=args.release_id,
                             max_workers=args.max_workers)
            print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
            return 0 if result["status"] == "complete_terminal" else 1
        result = evaluate_locked(args.run_dir)
        print(json.dumps(_jsonable({"status": result["status"], "candidate_count": result["candidate_count"]}), sort_keys=True))
        return 0
    except (WorkerError, prepare.CalibrationError, AssertionError, ValueError, FileExistsError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


__all__ = [
    "FIT_MANIFEST_SCHEMA", "FREEZE_SCHEMA", "MAX_WORKERS", "WorkerError",
    "build_tasks", "create_freeze", "evaluate_locked", "main", "preflight", "run_all", "run_task",
]


if __name__ == "__main__":
    raise SystemExit(main())
