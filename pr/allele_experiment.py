"""已发布 real 15-fit study 的 controller、worker 和 finalizer。

prepare 有意保持 inert：核验调用者提供的 protocol release hash、冻结的 SNP-free source 和三个外部生成的 native x0 bundle，然后写出完整的 15-job manifest。只有 ``worker`` 可以调用 ``run_one_fit``。这里不加载 reference 3DG、带 phase 的 payload、synthetic calibration 或 native initializer。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from . import contact_model
from . import paths
from .allele_models import MODEL_IDS, model_spec
from .paired_run import (
    FitConfig,
    PairedRunError,
    PairedStart,
    SELECTION_TIE_TOL,
    load_paired_start,
    run_one_fit,
    select_best_start,
    write_coordinates,
)


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
TEST_RES_ROOT = ROOT / "test_res"
PROTOCOL_JSON = ROOT / "docs" / "POST020_ALLELE_ABLATION_PROTOCOL.json"
PROTOCOL_MD = ROOT / "docs" / "POST020_ALLELE_ABLATION_PROTOCOL.md"
PROTOCOL_PLAN = ROOT / "docs" / "PLAN-post020-allele-signal.md"
DEFAULT_SOURCE_PATH = Path(paths.SNPFREE).resolve()
DEFAULT_START_ROOT = (
    TEST_RES_ROOT / "025-20260913_135100-random-native-fullgrid-preflight" / "bundles"
)
SYNTHETIC_GATE_FILENAME = "implementation_gate.json"
INPUT_MANIFEST_SOURCE = ROOT / "inputs" / "manifest.json"
ARTIFACT_REGISTRY_NAME = "POST020_ALLELE_ABLATION_ARTIFACT_REGISTRY.jsonl"

REAL_STUDY_NUMBER = 27
STUDY_SLUG = "post020-allele-ablation-real"
BIN_SIZE = 1_000_000
BUNDLE_IDS = ("bundle1", "bundle2", "bundle3")
N_JOBS = len(MODEL_IDS) * len(BUNDLE_IDS)
N_STARTS = len(BUNDLE_IDS)
N_VARIANTS = len(MODEL_IDS)
MAX_WORKERS = 6
MIN_AVAILABLE_BYTES = 32 * (1 << 30)
EXPECTED_RAW_RECORDS = 1_703_888
EXPECTED_CIS_RECORDS = 1_135_454
EXPECTED_SAME_BIN_RECORDS = 438_774
EXPECTED_INTER_RECORDS = 568_434
EXPECTED_N_LOCI = 2_645
EXPECTED_N_BEADS = 5_290
EXPECTED_ZERO_ELIGIBLE_PAIRS = 3_009_436
EXPECTED_SOURCE_SHA256 = contact_model.FROZEN_P9016_SNPFREE_SHA256

# 这些值是已发布 real-study 契约，不是隐含的 solver 默认值。
REAL_FIT_CONFIG = FitConfig(
    maxiter=480,
    maxfun=1470,
    maxls=20,
    ftol=1e-10,
    gtol=1e-6,
    checkpoint_every=20,
)

TRACKED_SOURCE_FILES = (
    ROOT / "pr" / "__init__.py",
    ROOT / "pr" / "paths.py",
    ROOT / "pr" / "genome.py",
    ROOT / "pr" / "contact_model.py",
    ROOT / "pr" / "allele_models.py",
    ROOT / "pr" / "paired_run.py",
    ROOT / "pr" / "joint_fit.py",
    ROOT / "pr" / "allele_experiment.py",
)

TERMINAL_STATUSES = {
    "solver_converged",
    "budget_not_converged",
    "solver_failed",
    "failed_nonfinite",
    "failed_exception",
    "failed_source_snapshot",
    "failed_input",
}
RERUN_ALLOWED_STATUSES = {
    "failed_exception",
    "failed_source_snapshot",
    "failed_input",
}


class ExperimentError(RuntimeError):
    """controller/worker 契约无法履行时抛出。"""


@dataclass(frozen=True)
class X0Record:
    bundle_id: str
    path: Path
    status: str
    file_sha256: str | None
    coordinate_sha256: str | None
    shape: tuple[int, ...] | None
    max_radius: float | None
    p_init: float | None
    error: str | None = None
    start: PairedStart | None = None
    source_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "path": str(self.path),
            "source_path": None if self.source_path is None else str(self.source_path),
            "status": self.status,
            "file_sha256": self.file_sha256,
            "coordinate_sha256": self.coordinate_sha256,
            "shape": None if self.shape is None else list(self.shape),
            "max_radius": self.max_radius,
            "p_init": self.p_init,
            "error": self.error,
        }


class _JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("x", encoding="utf-8")
        self.path = path

    def write(self, event: str, **fields: Any) -> None:
        payload = {"utc": utc_now(), "event": event, **fields}
        self.handle.write(json.dumps(_jsonable(payload), sort_keys=True,
                                     allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8", order="C")
    return sha256_bytes(array.tobytes(order="C"))


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite value cannot enter experiment JSON")
    return value


def write_json(path: Path, payload: Any, replace_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.%s.tmp" % (path.name, os.getpid(), uuid.uuid4().hex))
    encoded = json.dumps(_jsonable(payload), indent=2, sort_keys=True,
                         allow_nan=False) + "\n"
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


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExperimentError("missing JSON artifact: %s" % path) from exc
    except json.JSONDecodeError as exc:
        raise ExperimentError("invalid JSON artifact: %s" % path) from exc
    if not isinstance(payload, dict):
        raise ExperimentError("JSON artifact must be an object: %s" % path)
    return payload


def relpath(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _source_snapshot_payload(source_path: Path) -> dict[str, Any]:
    paths_to_hash = tuple(TRACKED_SOURCE_FILES) + (source_path.resolve(),)
    files = []
    for path in paths_to_hash:
        path = path.resolve()
        if not path.is_file():
            raise ExperimentError("tracked source file missing: %s" % path)
        files.append({"path": str(path), "sha256": sha256_file(path)})
    return {
        "schema": "post020-source-hash-snapshot-v1",
        "workspace_root": str(ROOT),
        "files": files,
        "source_snpfree_path": str(source_path.resolve()),
        "source_snpfree_sha256": sha256_file(source_path),
        "phase_or_reference_read": False,
    }


def build_source_snapshot(source_path: Path) -> dict[str, Any]:
    payload = _source_snapshot_payload(source_path)
    return {**payload, "snapshot_sha256": sha256_bytes(_canonical_json(payload))}


def verify_source_snapshot(path: Path, expected_snapshot_sha256: str | None = None) -> dict[str, Any]:
    snapshot = read_json(path)
    supplied = snapshot.pop("snapshot_sha256", None)
    if not isinstance(supplied, str):
        raise ExperimentError("source snapshot lacks snapshot_sha256")
    actual_snapshot = sha256_bytes(_canonical_json(snapshot))
    if supplied != actual_snapshot:
        raise ExperimentError("source snapshot self-hash mismatch")
    if expected_snapshot_sha256 is not None and supplied != expected_snapshot_sha256:
        raise ExperimentError("source snapshot hash differs from job manifest")
    for item in snapshot.get("files", []):
        file_path = Path(str(item["path"])).resolve()
        expected = str(item["sha256"])
        if not file_path.is_file() or sha256_file(file_path) != expected:
            raise ExperimentError("hash-locked source changed: %s" % file_path)
    source_path = Path(str(snapshot["source_snpfree_path"])).resolve()
    if not source_path.is_file() or sha256_file(source_path) != snapshot["source_snpfree_sha256"]:
        raise ExperimentError("hash-locked SNP-free source changed: %s" % source_path)
    return {**snapshot, "snapshot_sha256": supplied}


def _walk_key_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = "%s.%s" % (prefix, key) if prefix else str(key)
            rows.extend(_walk_key_values(item, name))
            rows.append((str(key).lower().replace("-", "_").replace(" ", "_"), item))
    return rows


def _validate_protocol_payload(payload: Mapping[str, Any]) -> None:
    """核验 real-fit 字段，不将其与 synthetic 预算混淆。"""
    def check(mapping: Mapping[str, Any], key: str, expected: int | float,
              label: str) -> None:
        if key not in mapping:
            raise ExperimentError("released protocol lacks %s" % label)
        try:
            actual = float(mapping[key])
        except (TypeError, ValueError) as exc:
            raise ExperimentError("protocol %s is not numeric" % label) from exc
        if not math.isfinite(actual) or not math.isclose(
                actual, float(expected), rel_tol=0.0, abs_tol=0.0):
            raise ExperimentError(
                "protocol %s=%r differs from frozen value %r" %
                (label, mapping[key], expected))

    real_budget = payload.get("real_fit_budget")
    if isinstance(real_budget, Mapping):
        for key, expected in (
            ("fit_count", N_JOBS),
            ("maxiter_accepted", REAL_FIT_CONFIG.maxiter),
            ("maxfun", REAL_FIT_CONFIG.maxfun),
            ("maxls", REAL_FIT_CONFIG.maxls),
            ("ftol", REAL_FIT_CONFIG.ftol),
            ("gtol", REAL_FIT_CONFIG.gtol),
            ("checkpoint_accepted_iterations", REAL_FIT_CONFIG.checkpoint_every or 20),
        ):
            check(real_budget, key, expected, "real_fit_budget.%s" % key)
    else:
        # 小型 no-fit 测试使用的 fixture protocol 可以采用紧凑 schema。
        aliases: dict[str, tuple[str, ...]] = {
            "bin_size": ("bin_size", "bin_size_bp", "resolution_bp"),
            "maxiter": ("maxiter", "max_iter", "maxiter_accepted"),
            "maxfun": ("maxfun", "max_fun"),
            "maxls": ("maxls", "max_ls"),
            "ftol": ("ftol",),
            "gtol": ("gtol",),
            "checkpoint_every": ("checkpoint_every", "checkpoint_interval", "checkpoint_accepted_iterations"),
            "n_jobs": ("n_jobs", "job_count", "fit_count"),
            "n_variants": ("n_variants", "variant_count"),
            "n_starts": ("n_starts", "start_count", "bundle_count"),
        }
        expected: dict[str, int | float] = {
            "bin_size": BIN_SIZE,
            "maxiter": REAL_FIT_CONFIG.maxiter,
            "maxfun": REAL_FIT_CONFIG.maxfun,
            "maxls": REAL_FIT_CONFIG.maxls,
            "ftol": REAL_FIT_CONFIG.ftol,
            "gtol": REAL_FIT_CONFIG.gtol,
            "checkpoint_every": REAL_FIT_CONFIG.checkpoint_every or 20,
            "n_jobs": N_JOBS,
            "n_variants": N_VARIANTS,
            "n_starts": N_STARTS,
        }
        found = _walk_key_values(payload)
        for contract_name, names in aliases.items():
            matches = [(key, value) for key, value in found if key in names]
            for key, value in matches:
                try:
                    actual = float(value)
                except (TypeError, ValueError) as exc:
                    raise ExperimentError("protocol %s is not numeric" % key) from exc
                if not math.isfinite(actual) or not math.isclose(
                        actual, float(expected[contract_name]), rel_tol=0.0, abs_tol=0.0):
                    raise ExperimentError(
                        "protocol %s=%r differs from frozen value %r" %
                        (key, value, expected[contract_name]))

    real_cohort = payload.get("real_cohort_freeze")
    if isinstance(real_cohort, Mapping):
        grid = real_cohort.get("grid")
        if isinstance(grid, Mapping):
            check(grid, "resolution_bp", BIN_SIZE, "real_cohort_freeze.grid.resolution_bp")
            check(grid, "n_loci_per_copy", EXPECTED_N_LOCI,
                  "real_cohort_freeze.grid.n_loci_per_copy")
            check(grid, "n_physical_beads", EXPECTED_N_BEADS,
                  "real_cohort_freeze.grid.n_physical_beads")
            check(grid, "real_zero_eligible_pairs", EXPECTED_ZERO_ELIGIBLE_PAIRS,
                  "real_cohort_freeze.grid.real_zero_eligible_pairs")
        raw = real_cohort.get("raw_records")
        if isinstance(raw, Mapping):
            for key, expected in (("total", EXPECTED_RAW_RECORDS),
                                  ("cis_total", EXPECTED_CIS_RECORDS),
                                  ("inter", EXPECTED_INTER_RECORDS),
                                  ("same_bin_diagonal", EXPECTED_SAME_BIN_RECORDS)):
                check(raw, key, expected, "real_cohort_freeze.raw_records.%s" % key)

    model_freeze = payload.get("model_freeze")
    if isinstance(model_freeze, Mapping) and "variants" in model_freeze:
        variants = model_freeze["variants"]
        if not isinstance(variants, list) or [item.get("variant_id") for item in variants] != list(MODEL_IDS):
            raise ExperimentError("released protocol variant order is not the five frozen arms")
    native_freeze = payload.get("native_initialization_freeze")
    if isinstance(native_freeze, Mapping) and "bundles" in native_freeze:
        bundles = native_freeze["bundles"]
        if not isinstance(bundles, list) or [item.get("bundle_id") for item in bundles] != list(BUNDLE_IDS):
            raise ExperimentError("released protocol bundle order is not the three frozen starts")

    status = str(payload.get("status", "")).lower()
    if status in {"draft", "pending", "unreleased", "not_released", "superseded"}:
        raise ExperimentError("protocol is not released: %s" % status)


def verify_protocol_release(protocol_json: Path, release_sha256: str,
                            protocol_md: Path | None = PROTOCOL_MD) -> dict[str, Any]:
    protocol_json = protocol_json.resolve()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(release_sha256)):
        raise ExperimentError("--release must be the 64-hex protocol JSON SHA256")
    if not protocol_json.is_file():
        raise ExperimentError("released protocol JSON is missing: %s" % protocol_json)
    actual = sha256_file(protocol_json)
    expected = str(release_sha256).lower()
    if actual != expected:
        raise ExperimentError(
            "protocol release SHA mismatch: got %s, expected %s" % (actual, expected))
    try:
        payload = json.loads(protocol_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentError("released protocol JSON is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ExperimentError("released protocol JSON must be an object")
    _validate_protocol_payload(payload)
    md_path = None if protocol_md is None else Path(protocol_md).resolve()
    return {
        "path": str(protocol_json),
        "sha256": actual,
        "schema": payload.get("schema", payload.get("schema_version")),
        "md_path": str(md_path) if md_path is not None and md_path.is_file() else None,
        "md_sha256": sha256_file(md_path) if md_path is not None and md_path.is_file() else None,
    }


def _available_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024
    return None


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _single_thread_env() -> dict[str, str]:
    return {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def _used_study_numbers(test_res_root: Path) -> set[int]:
    if not test_res_root.exists():
        return set()
    values = set()
    for item in test_res_root.iterdir():
        if not item.is_dir():
            continue
        match = re.match(r"^(\d{3})-", item.name)
        if match:
            values.add(int(match.group(1)))
    return values


def allocate_study_root(test_res_root: Path = TEST_RES_ROOT) -> Path:
    used = _used_study_numbers(test_res_root)
    number = REAL_STUDY_NUMBER
    while number in used:
        number += 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return test_res_root / ("%03d-%s-%s" % (number, stamp, STUDY_SLUG))


def _study_number(root: Path) -> int | None:
    match = re.match(r"^(\d{3})-", root.name)
    return None if match is None else int(match.group(1))


def _create_fresh_root(root: Path) -> None:
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise ExperimentError("refusing to reuse non-empty study directory: %s" % root)
    else:
        root.mkdir(parents=True)
    for name in (
        "jobs", "active", "logs", "checkpoints", "results", "provenance",
        "initialization", "freeze", "coordinates", "plots", "evaluation",
    ):
        (root / name).mkdir(exist_ok=True)


def _discover_x0(starts_root: Path) -> tuple[X0Record, ...]:
    records = []
    for bundle_id in BUNDLE_IDS:
        path = (starts_root / bundle_id / "x0_normalized.npz").resolve()
        if not path.is_file():
            records.append(X0Record(bundle_id, path, "missing", None, None, None, None, None,
                                    "x0_normalized.npz is not present", source_path=path))
            continue
        file_sha = sha256_file(path)
        try:
            start = load_paired_start(path, start_id=bundle_id)
            values = np.asarray(start.coordinates, dtype=np.float64)
            records.append(X0Record(
                bundle_id=bundle_id,
                path=path,
                status="ready",
                file_sha256=file_sha,
                coordinate_sha256=start.coordinate_sha256,
                shape=tuple(int(x) for x in values.shape),
                max_radius=start.max_radius,
                p_init=float(start.p_init),
                start=start,
                source_path=path,
            ))
        except Exception as exc:
            records.append(X0Record(bundle_id, path, "invalid", file_sha, None, None, None, None,
                                    "%s: %s" % (type(exc).__name__, exc), source_path=path))
    return tuple(records)


def _copy_file_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        for block in iter(lambda: source_handle.read(1 << 20), b""):
            destination_handle.write(block)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())


def _materialize_x0(root: Path, records: Sequence[X0Record]) -> tuple[X0Record, ...]:
    """将已准备好的 025 x0 bytes 复制到新 run 的 initialization tree。"""
    materialized = []
    for record in records:
        destination = root / "initialization" / record.bundle_id / "x0_normalized.npz"
        if record.status == "ready":
            if record.source_path is None or not record.source_path.is_file():
                raise ExperimentError("ready x0 has no source path: %s" % record.bundle_id)
            _copy_file_exact(record.source_path, destination)
            if sha256_file(destination) != record.file_sha256:
                raise ExperimentError("x0 copy hash mismatch: %s" % record.bundle_id)
            materialized.append(X0Record(
                bundle_id=record.bundle_id,
                path=destination,
                status=record.status,
                file_sha256=record.file_sha256,
                coordinate_sha256=record.coordinate_sha256,
                shape=record.shape,
                max_radius=record.max_radius,
                p_init=record.p_init,
                error=record.error,
                start=record.start,
                source_path=record.source_path,
            ))
        else:
            materialized.append(X0Record(
                bundle_id=record.bundle_id,
                path=destination,
                status=record.status,
                file_sha256=record.file_sha256,
                coordinate_sha256=record.coordinate_sha256,
                shape=record.shape,
                max_radius=record.max_radius,
                p_init=record.p_init,
                error=record.error,
                start=record.start,
                source_path=record.source_path,
            ))
    return tuple(materialized)


def _resolve_gate_artifact(value: Any, gate_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    for base in (ROOT, gate_path.parent, gate_path.parent.parent):
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    return (ROOT / candidate).resolve()


def _synthetic_gate(path: str | Path | None = None,
                    protocol_sha256: str | None = None) -> dict[str, Any]:
    """只检查已发布的 026 synthetic terminal implementation gate。"""
    if path is None:
        candidates = sorted(
            directory / "results" / SYNTHETIC_GATE_FILENAME
            for directory in TEST_RES_ROOT.iterdir()
            if directory.is_dir() and directory.name.startswith("026-")
            and (directory / "results" / SYNTHETIC_GATE_FILENAME).is_file()
        ) if TEST_RES_ROOT.is_dir() else []
        if len(candidates) > 1:
            return {
                "status": "invalid",
                "path": None,
                "sha256": None,
                "error": "multiple 026 synthetic implementation gates are present",
                "candidate_paths": [str(candidate) for candidate in candidates],
            }
        path_obj = candidates[0] if candidates else None
    else:
        path_obj = Path(path).resolve()
    if path_obj is None or not path_obj.is_file():
        return {
            "status": "missing",
            "path": None if path_obj is None else str(path_obj),
            "sha256": None,
            "error": "026/results/implementation_gate.json is not present",
        }
    if (path_obj.name != SYNTHETIC_GATE_FILENAME or path_obj.parent.name != "results"
            or not path_obj.parent.parent.name.startswith("026-")):
        return {
            "status": "invalid",
            "path": str(path_obj),
            "sha256": sha256_file(path_obj),
            "error": "synthetic gate path is outside the 026/results contract",
        }
    file_sha = sha256_file(path_obj)
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "path": str(path_obj),
            "sha256": file_sha,
            "error": "%s: %s" % (type(exc).__name__, exc),
        }
    errors = []
    if not isinstance(payload, Mapping):
        errors.append("gate JSON must be an object")
        payload = {}
    if payload.get("schema") != "post020-synthetic-terminal-gate-v1":
        errors.append("schema is not post020-synthetic-terminal-gate-v1")
    if payload.get("status") != "passed":
        errors.append("top-level status is not passed")
    supplied_protocol_sha = payload.get("protocol_json_sha256")
    if not isinstance(supplied_protocol_sha, str):
        errors.append("protocol_json_sha256 is missing")
    elif protocol_sha256 is not None and supplied_protocol_sha != protocol_sha256:
        errors.append("protocol_json_sha256 differs from released protocol")
    elif not re.fullmatch(r"[0-9a-fA-F]{64}", supplied_protocol_sha):
        errors.append("protocol_json_sha256 is not a 64-hex SHA256")
    for key, expected in (
        ("planned_fit_count", 20),
        ("terminal_fit_count", 20),
        ("finite_valid_endpoint_count", 20),
    ):
        if payload.get(key) != expected:
            errors.append("%s is not %s" % (key, expected))
    for key in ("coordinates_hash_locked", "numerical_checks_passed", "evaluation_complete"):
        if payload.get(key) is not True:
            errors.append("%s is not true" % key)

    references = {
        "terminal_manifest_path": "terminal_manifest_sha256",
        "evaluation_path": "evaluation_sha256",
    }
    reference_hashes = {}
    for path_key, hash_key in references.items():
        artifact = _resolve_gate_artifact(payload.get(path_key), path_obj)
        supplied_hash = payload.get(hash_key)
        if artifact is None or not artifact.is_file():
            errors.append("%s is missing" % path_key)
            reference_hashes[path_key] = {"path": None, "sha256": supplied_hash}
        elif not isinstance(supplied_hash, str) or sha256_file(artifact) != supplied_hash:
            errors.append("%s hash mismatch" % path_key)
            reference_hashes[path_key] = {"path": str(artifact), "sha256": supplied_hash}
        else:
            reference_hashes[path_key] = {"path": str(artifact), "sha256": supplied_hash}
    return {
        "status": "ready" if not errors else "invalid",
        "path": str(path_obj),
        "sha256": file_sha,
        "schema": payload.get("schema"),
        "protocol_json_sha256": payload.get("protocol_json_sha256"),
        "planned_fit_count": payload.get("planned_fit_count"),
        "terminal_fit_count": payload.get("terminal_fit_count"),
        "finite_valid_endpoint_count": payload.get("finite_valid_endpoint_count"),
        "coordinates_hash_locked": payload.get("coordinates_hash_locked"),
        "numerical_checks_passed": payload.get("numerical_checks_passed"),
        "evaluation_complete": payload.get("evaluation_complete"),
        **reference_hashes,
        "error": None if not errors else "; ".join(errors),
    }


def _load_real_cohort(source_path: Path) -> tuple[contact_model.AggregatedContacts, dict[str, Any]]:
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise ExperimentError("SNP-free source is missing: %s" % source_path)
    source_sha = sha256_file(source_path)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise ExperimentError(
            "SNP-free source SHA differs from frozen input: got %s, expected %s" %
            (source_sha, EXPECTED_SOURCE_SHA256))
    data = contact_model.load_aggregate(source_path, bin_size=BIN_SIZE,
                                        verify_frozen_hash=True)
    audit = data.budget()
    expected = {
        "raw_records": EXPECTED_RAW_RECORDS,
        "raw_same_bin": EXPECTED_SAME_BIN_RECORDS,
        "raw_cis_offdiag": EXPECTED_CIS_RECORDS - EXPECTED_SAME_BIN_RECORDS,
        "raw_inter": EXPECTED_INTER_RECORDS,
        "n_loci": EXPECTED_N_LOCI,
        "n_zero_eligible_pairs": EXPECTED_ZERO_ELIGIBLE_PAIRS,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ExperimentError("frozen cohort %s=%r, observed %r" %
                                  (key, value, audit.get(key)))
    if 2 * data.n_loci != EXPECTED_N_BEADS:
        raise ExperimentError("frozen cohort bead inventory changed")
    return data, {
        "schema": "post020-real-snpfree-cohort-v1",
        "source_path": str(source_path),
        "source_sha256": source_sha,
        "bin_size_bp": BIN_SIZE,
        "raw_records": EXPECTED_RAW_RECORDS,
        "cis_records": EXPECTED_CIS_RECORDS,
        "same_bin_records": EXPECTED_SAME_BIN_RECORDS,
        "cis_offdiag_records": EXPECTED_CIS_RECORDS - EXPECTED_SAME_BIN_RECORDS,
        "inter_records": EXPECTED_INTER_RECORDS,
        "n_loci": data.n_loci,
        "n_physical_beads": 2 * data.n_loci,
        "n_eligible_pairs": data.n_pairs,
        "n_zero_eligible_pairs": int((data.counts == 0).sum()),
        "exposure_mode": data.exposure_mode,
        "count_mode": data.count_mode,
        "phase_or_reference_read": False,
        "training_payload": "seven-column SNP-free pairs only",
    }


def _append_registry(root: Path, record: Mapping[str, Any]) -> Path:
    path = root.parent / ARTIFACT_REGISTRY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_jsonable(dict(record)), sort_keys=True,
                      allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _input_manifest(root: Path, source_sha: str) -> dict[str, Any]:
    if not INPUT_MANIFEST_SOURCE.is_file():
        raise ExperimentError("frozen inputs/manifest.json is missing")
    payload = read_json(INPUT_MANIFEST_SOURCE)
    if payload.get("output_sha256") != source_sha:
        raise ExperimentError("inputs manifest does not bind the frozen SNP-free SHA")
    destination = root / "input_manifest.json"
    _copy_file_exact(INPUT_MANIFEST_SOURCE, destination)
    return {
        "source_path": str(INPUT_MANIFEST_SOURCE),
        "path": str(destination),
        "sha256": sha256_file(destination),
        "output_sha256": payload.get("output_sha256"),
        "records": payload.get("records"),
        "columns": payload.get("columns"),
    }


def _native_input_artifacts(starts: Sequence[X0Record]) -> dict[str, Any]:
    preflight_root = DEFAULT_START_ROOT.parent
    records = []
    for item in starts:
        source_path = item.source_path or item.path
        records.append({
            "bundle_id": item.bundle_id,
            "x0_source_path": str(source_path),
            "x0_source_sha256": item.file_sha256,
        })
    config_candidates = (preflight_root / "config.json", preflight_root / "config_snapshot.json")
    config_path = next((path for path in config_candidates if path.is_file()), None)
    graph_artifacts = []
    for bundle_id in BUNDLE_IDS:
        graph_root = preflight_root / "bundles" / bundle_id / "graph"
        for name in ("graph_manifest.json", "canonical_edges.npz", "assignment_ledger.npz", "bridge_input.rndbin"):
            artifact = graph_root / name
            graph_artifacts.append({
                "bundle_id": bundle_id,
                "path": str(artifact),
                "sha256": sha256_file(artifact) if artifact.is_file() else None,
            })
    return {
        "preflight_root": str(preflight_root),
        "preflight_config_path": None if config_path is None else str(config_path),
        "preflight_config_sha256": None if config_path is None else sha256_file(config_path),
        "bundles": records,
        "graph_artifacts": graph_artifacts,
        "native_generation_in_this_module": False,
    }


def _write_freeze_record(root: Path, release: Mapping[str, Any],
                         manifest_paths: Mapping[str, Path],
                         input_manifest: Mapping[str, Any],
                         starts: Sequence[X0Record],
                         gate: Mapping[str, Any],
                         status: str) -> tuple[Path, str]:
    freeze_path = root / "freeze" / "POST020_ALLELE_ABLATION_FREEZE.json"
    payload = {
        "schema": "post020-allele-ablation-freeze-v1",
        "status": status,
        "fit_called": False,
        "frozen_utc": utc_now(),
        "protocol_json": {
            "path": release["path"],
            "sha256": release["sha256"],
        },
        "protocol_md": {
            "path": release.get("md_path"),
            "sha256": release.get("md_sha256"),
        },
        "protocol_plan": {
            "path": str(PROTOCOL_PLAN) if PROTOCOL_PLAN.is_file() else None,
            "sha256": sha256_file(PROTOCOL_PLAN) if PROTOCOL_PLAN.is_file() else None,
        },
        "input_manifest": dict(input_manifest),
        "source_snapshot": {
            "path": relpath(manifest_paths["source_snapshot"], root),
            "sha256": sha256_file(manifest_paths["source_snapshot"]),
        },
        "prepared_x0": {
            "manifest_path": relpath(manifest_paths["x0_manifest"], root),
            "manifest_sha256": sha256_file(manifest_paths["x0_manifest"]),
            "bundles": [item.as_dict() for item in starts],
        },
        "synthetic_implementation_gate": dict(gate),
        "native_preflight_artifacts": _native_input_artifacts(starts),
    }
    write_json(freeze_path, payload)
    return freeze_path, sha256_file(freeze_path)


def _write_text_exact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _config_contract() -> dict[str, Any]:
    return {
        "one_mb_only": True,
        "multiresolution": False,
        "variants": list(MODEL_IDS),
        "bundles": list(BUNDLE_IDS),
        "jobs": N_JOBS,
        "starts": N_STARTS,
        "paired_physical_start_across_variants": True,
        "p_init": 0.75,
        "fit": {
            "maxiter_accepted": REAL_FIT_CONFIG.maxiter,
            "maxfun": REAL_FIT_CONFIG.maxfun,
            "maxls": REAL_FIT_CONFIG.maxls,
            "ftol": REAL_FIT_CONFIG.ftol,
            "gtol": REAL_FIT_CONFIG.gtol,
            "checkpoint_accepted_iterations": REAL_FIT_CONFIG.checkpoint_every,
        },
        "fit_budget_is_explicit": True,
        "selection": {
            "within_variant_only": True,
            "criterion": "count_nll_normalized",
            "tie_tol": SELECTION_TIE_TOL,
            "stable_tiebreak": "bundle_id",
            "nonfinite_rejected": True,
            "budget_not_converged_eligible": True,
        },
        "resource_policy": {
            "max_parallel_workers": MAX_WORKERS,
            "min_mem_available_bytes": MIN_AVAILABLE_BYTES,
            "blas_omp_threads_per_worker": 1,
        },
    }


def _job_id(model_id: str, bundle_id: str) -> str:
    return "%s-%s" % (model_id, bundle_id)


def _forbidden_training_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(token in key_text for token in ("reference", "truth", "phase")):
                found.append(prefix + str(key))
            found.extend(_forbidden_training_keys(item, prefix + str(key) + "."))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_training_keys(item, prefix + str(index) + "."))
    return found


def _worker_job_config(root: Path, model_id: str, x0: X0Record,
                       source_path: Path, source_sha: str,
                       source_snapshot_sha: str, protocol_release_sha: str,
                       synthetic_gate: Mapping[str, Any]) -> dict[str, Any]:
    job_id = _job_id(model_id, x0.bundle_id)
    job_root = root / "jobs" / job_id
    attempt_root = job_root / "attempts"
    return {
        "schema": "post020-real-allele-training-job-v1",
        "job_id": job_id,
        "study_root": str(root),
        "model_id": model_id,
        "bundle_id": x0.bundle_id,
        "start_id": x0.bundle_id,
        "x0_path": str(x0.path),
        "x0_file_sha256": x0.file_sha256,
        "x0_coordinate_sha256": x0.coordinate_sha256,
        "x0_status": x0.status,
        "source_snpfree_path": str(source_path.resolve()),
        "source_snpfree_sha256": source_sha,
        "source_snapshot_path": str(root / "source_hashes.json"),
        "source_snapshot_sha256": source_snapshot_sha,
        "protocol_release_sha256": protocol_release_sha,
        "synthetic_gate_path": synthetic_gate.get("path"),
        "synthetic_gate_sha256": synthetic_gate.get("sha256"),
        "synthetic_gate_status": synthetic_gate.get("status"),
        "bin_size_bp": BIN_SIZE,
        "fit": REAL_FIT_CONFIG.as_dict(),
        "resource_policy": {
            "max_parallel_workers": MAX_WORKERS,
            "min_mem_available_bytes": MIN_AVAILABLE_BYTES,
            "blas_omp_threads_per_worker": 1,
        },
        "outputs": {
            "job_root": str(job_root),
            "attempts_root": str(attempt_root),
            "central_status": str(job_root / "status.json"),
        },
        "worker_role": "training_only",
    }


def _prepare_readme(root: Path, status: str) -> str:
    return """# Post-020 真实等位拷贝消融研究（027）

本目录是由控制器管理的全新、仅 1 Mb 的研究，包含五个变体和三个配对物理起点。准备阶段从不运行优化器。每个工作进程单独以 `python -m pr.allele_experiment --worker --job JOB_JSON` 启动。

作业 JSON 只包含哈希锁定的七列无 SNP 训练源、实体化后的物理 x0 配对组、已发布的数值预算和输出路径。不包含参考结构或定相载荷路径。`finalize` 保留放行清单中的全部 15 个分支，包括 pending、failed 和 not-applicable 作业。

准备时状态：%s
""" % status


def prepare(*, run_root: str | Path | None = None,
            release_sha256: str | None = None,
            protocol_json: str | Path = PROTOCOL_JSON,
            protocol_md: str | Path | None = PROTOCOL_MD,
            synthetic_gate_path: str | Path | None = None,
            starts_root: str | Path = DEFAULT_START_ROOT,
            source_path: str | Path = DEFAULT_SOURCE_PATH) -> dict[str, Any]:
    """准备新的真实研究目录，不启动任何拟合。"""
    if release_sha256 is None:
        raise ExperimentError("prepare requires an explicit --release protocol SHA256")
    release = verify_protocol_release(Path(protocol_json), release_sha256, None if protocol_md is None else Path(protocol_md))
    source_path = Path(source_path).resolve()
    source_sha = sha256_file(source_path) if source_path.is_file() else None
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise ExperimentError("SNP-free source is missing or does not match frozen SHA")
    data, cohort = _load_real_cohort(source_path)
    starts = _discover_x0(Path(starts_root).resolve())
    gate = _synthetic_gate(synthetic_gate_path, release["sha256"])
    ready_starts = [item.start for item in starts if item.status == "ready" and item.start is not None]
    starts_complete = len(ready_starts) == N_STARTS
    if starts_complete:
        for start in ready_starts:
            if start.coordinates.shape != (2, data.n_loci, 3):
                starts_complete = False
        try:
            # 该检查验证 common physical-start gate，不评估 objective。
            from .paired_run import plan_paired_runs
            plan_paired_runs(data, ready_starts)
        except (AssertionError, PairedRunError, ValueError):
            starts_complete = False
    status = "ready_for_worker" if starts_complete and gate["status"] == "ready" else (
        "pending_inputs" if not starts_complete else "pending_synthetic_gate"
    )
    ready_for_worker = status == "ready_for_worker"
    root = allocate_study_root() if run_root is None else Path(run_root).resolve()
    _create_fresh_root(root)
    starts = _materialize_x0(root, starts)
    input_manifest = _input_manifest(root, source_sha)

    source_snapshot = build_source_snapshot(source_path)
    source_snapshot_path = root / "source_hashes.json"
    write_json(source_snapshot_path, source_snapshot)
    source_manifest_path = root / "source_manifest.json"
    _copy_file_exact(source_snapshot_path, source_manifest_path)
    source_snapshot_sha = str(source_snapshot["snapshot_sha256"])

    x0_payload = {
        "schema": "post020-paired-x0-manifest-v1",
        "status": "complete" if starts_complete else "pending_inputs",
        "required_bundles": list(BUNDLE_IDS),
        "source_study": "025-random-native-fullgrid-preflight",
        "bundles": [item.as_dict() for item in starts],
        "same_physical_x0_across_variants": True,
        "native_generation_in_this_module": False,
    }
    x0_path = root / "x0_manifest.json"
    write_json(x0_path, x0_payload)

    cohort_path = root / "cohort.json"
    write_json(cohort_path, cohort)
    config_payload = {
        "schema": "post020-real-allele-ablation-config-v1",
        "status": status,
        "study_number": _study_number(root),
        "study_root": str(root),
        "prepared_utc": utc_now(),
        "protocol_release": release,
        "source": {
            "path": str(source_path),
            "sha256": source_sha,
            "hash_locked": True,
            "provenance_snapshot": relpath(source_snapshot_path, root),
        },
        "input_manifest": input_manifest,
        "cohort": {
            "path": relpath(cohort_path, root),
            "sha256": sha256_file(cohort_path),
            **cohort,
        },
        "x0_manifest": {
            "path": relpath(x0_path, root),
            "sha256": sha256_file(x0_path),
            "bundles": [item.as_dict() for item in starts],
        },
        "synthetic_implementation_gate": gate,
        "contract": _config_contract(),
        "model_specs": [model_spec(model_id).as_dict() for model_id in MODEL_IDS],
        "x0_complete": starts_complete,
        "synthetic_gate_ready": gate["status"] == "ready",
        "inputs_complete": ready_for_worker,
        "fit_called": False,
        "forbidden_to_training": [
            "phase0/phase1/phase_prob payloads",
            "reference 3DG",
            "old coordinate snapshots",
        ],
    }
    config_path = root / "config.json"
    config_snapshot_path = root / "config_snapshot.json"
    write_json(config_path, config_payload)
    write_json(config_snapshot_path, config_payload)

    job_rows = []
    for model_id in MODEL_IDS:
        for x0 in starts:
            job_id = _job_id(model_id, x0.bundle_id)
            job_root = root / "jobs" / job_id
            (job_root / "attempts").mkdir(parents=True, exist_ok=False)
            job_payload = _worker_job_config(
                root, model_id, x0, source_path, source_sha,
                source_snapshot_sha, release_sha256.lower(), gate,
            )
            job_path = job_root / "job.json"
            write_json(job_path, job_payload)
            job_rows.append({
                "job_id": job_id,
                "model_id": model_id,
                "bundle_id": x0.bundle_id,
                "status": "pending" if ready_for_worker else "pending_input",
                "job_config": relpath(job_path, root),
                "job_config_sha256": sha256_file(job_path),
                "x0_file_sha256": x0.file_sha256,
                "x0_coordinate_sha256": x0.coordinate_sha256,
                "synthetic_gate_sha256": gate.get("sha256"),
                "fit_called": False,
            })
    jobs_manifest = {
        "schema": "post020-real-allele-job-manifest-v1",
        "status": status,
        "job_count": N_JOBS,
        "all_registered_arms_preserved": True,
        "jobs": job_rows,
    }
    jobs_manifest_path = root / "jobs" / "manifest.json"
    write_json(jobs_manifest_path, jobs_manifest)

    freeze_path, freeze_sha256 = _write_freeze_record(
        root, release,
        {"source_snapshot": source_snapshot_path, "x0_manifest": x0_path},
        input_manifest, starts, gate, status,
    )

    manifest = {
        "schema": "post020-real-allele-ablation-manifest-v1",
        "status": status,
        "ready_for_worker": ready_for_worker,
        "study_number": _study_number(root),
        "study_root": str(root),
        "prepared_utc": config_payload["prepared_utc"],
        "protocol_release": release,
        "source_hashes": {
            "path": relpath(source_snapshot_path, root),
            "sha256": sha256_file(source_snapshot_path),
            "input_path": str(source_path),
            "input_sha256": source_sha,
        },
        "source_manifest": {
            "path": relpath(source_manifest_path, root),
            "sha256": sha256_file(source_manifest_path),
        },
        "input_manifest": {
            "path": relpath(Path(str(input_manifest["path"])), root),
            "sha256": input_manifest["sha256"],
        },
        "cohort": {"path": relpath(cohort_path, root), "sha256": sha256_file(cohort_path)},
        "config": {"path": relpath(config_path, root), "sha256": sha256_file(config_path)},
        "config_snapshot": {"path": relpath(config_snapshot_path, root), "sha256": sha256_file(config_snapshot_path)},
        "freeze": {"path": relpath(freeze_path, root), "sha256": freeze_sha256},
        "x0_manifest": {"path": relpath(x0_path, root), "sha256": sha256_file(x0_path)},
        "synthetic_implementation_gate": gate,
        "jobs_manifest": {
            "path": relpath(jobs_manifest_path, root),
            "sha256": sha256_file(jobs_manifest_path),
        },
        "jobs": job_rows,
        "expected": {
            "variants": list(MODEL_IDS),
            "bundles": list(BUNDLE_IDS),
            "job_count": N_JOBS,
            "raw_records": EXPECTED_RAW_RECORDS,
            "same_bin_records": EXPECTED_SAME_BIN_RECORDS,
            "n_physical_beads": EXPECTED_N_BEADS,
            "n_zero_eligible_pairs": EXPECTED_ZERO_ELIGIBLE_PAIRS,
        },
        "fit_called": False,
        "native_initialization_called": False,
        "reference_or_phase_read": False,
    }
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    write_json(root / "status.json", {
        "schema": "post020-real-allele-status-v1",
        "status": status,
        "updated_utc": utc_now(),
        "fit_called": False,
        "jobs": {row["job_id"]: row["status"] for row in job_rows},
    })
    _write_text_exact(root / "README.md", _prepare_readme(root, status))
    registry_path = _append_registry(root, {
        "schema": "post020-allele-ablation-artifact-registry-v1",
        "recorded_utc": utc_now(),
        "run_root": str(root),
        "scope": "real",
        "variant_id": "all",
        "bundle_or_fixture_id": "all",
        "status": status,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_json_sha256": release["sha256"],
        "source_snapshot_sha256": source_snapshot_sha,
        "fit_called": False,
    })
    return {
        "status": status,
        "study_root": str(root),
        "manifest": str(manifest_path),
        "config_snapshot": str(config_snapshot_path),
        "config": str(config_path),
        "config_snapshot_path": str(config_snapshot_path),
        "jobs_manifest": str(jobs_manifest_path),
        "freeze": str(freeze_path),
        "registry": str(registry_path),
        "source_sha256": source_sha,
        "synthetic_implementation_gate": gate,
        "x0": [item.as_dict() for item in starts],
        "job_count": N_JOBS,
        "fit_called": False,
    }


def _claim_worker(root: Path, job_id: str) -> Path | None:
    available = _available_memory_bytes()
    if available is not None and available < MIN_AVAILABLE_BYTES:
        return None
    active_dir = root / "active"
    active_dir.mkdir(exist_ok=True)
    active_count = len(tuple(active_dir.glob("*.json")))
    if active_count >= MAX_WORKERS:
        return None
    claim = active_dir / (job_id + ".json")
    try:
        with claim.open("x", encoding="utf-8") as handle:
            json.dump({"job_id": job_id, "pid": os.getpid(), "started_utc": utc_now()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ExperimentError("duplicate running job claim: %s" % job_id) from exc
    return claim


def _latest_attempt(job_root: Path) -> tuple[Path, dict[str, Any]] | None:
    attempts = sorted(job_root.glob("attempts/attempt-*"))
    for path in reversed(attempts):
        status_path = path / "status.json"
        if status_path.is_file():
            return path, read_json(status_path)
    return None


def _next_attempt(job_root: Path, allow_rerun: bool, rerun_reason: str | None) -> tuple[Path, int, dict[str, Any] | None]:
    previous = _latest_attempt(job_root)
    if previous is not None:
        previous_path, previous_status = previous
        previous_name = previous_path.name
        if previous_status.get("status") == "running":
            raise ExperimentError("job already has a running attempt: %s" % job_root.name)
        if not allow_rerun:
            raise ExperimentError("refusing duplicate terminal job: %s" % job_root.name)
        if previous_status.get("status") not in RERUN_ALLOWED_STATUSES:
            raise ExperimentError(
                "rerun is forbidden for terminal status %s" % previous_status.get("status"))
        if not rerun_reason:
            raise ExperimentError("rerun requires an explicit source/code-fix reason")
        match = re.fullmatch(r"attempt-(\d+)", previous_name)
        number = int(match.group(1)) + 1 if match else 1
    else:
        number = 1
    attempt = job_root / "attempts" / ("attempt-%03d" % number)
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt, number, None if previous is None else previous[1]


def _write_npz_exact(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _write_checkpoint(root: Path, model_id: str, bundle_id: str,
                      attempt_root: Path, checkpoint: Any) -> dict[str, Any]:
    checkpoint_name = "iter-%06d" % int(checkpoint.iteration)
    npz_path = attempt_root / "checkpoints" / (checkpoint_name + ".npz")
    json_path = attempt_root / "checkpoints" / (checkpoint_name + ".json")
    _write_npz_exact(
        npz_path,
        theta=np.asarray(checkpoint.theta, dtype=np.float64),
        raw_coordinates=np.asarray(checkpoint.y, dtype=np.float64),
        coordinates=np.asarray(checkpoint.coordinates, dtype=np.float64),
    )
    payload = {
        "schema": "post020-fit-checkpoint-v1",
        "model_id": model_id,
        "bundle_id": bundle_id,
        "iteration": int(checkpoint.iteration),
        "nfev": int(checkpoint.nfev),
        "elapsed_seconds": float(checkpoint.elapsed_seconds),
        "fun": float(checkpoint.fun),
        "p": float(checkpoint.p),
        "gradient_l2": float(checkpoint.gradient_norm),
        "components": dict(checkpoint.components),
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
    }
    write_json(json_path, payload)
    return {"path": str(json_path), "sha256": sha256_file(json_path), **payload}


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


def _write_failure(attempt_root: Path, job: Mapping[str, Any], status: str,
                   error: BaseException, started_utc: str, rss_start: int,
                   runtime_seconds: float, fit_called: bool) -> dict[str, Any]:
    payload = {
        "schema": "post020-fit-failure-v1",
        "status": status,
        "job_id": job["job_id"],
        "model_id": job["model_id"],
        "bundle_id": job["bundle_id"],
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "runtime_seconds": float(runtime_seconds),
        "rss_start_bytes": int(rss_start),
        "rss_max_bytes": int(_rss_bytes()),
        "fit_called": bool(fit_called),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "selection_eligible": False,
    }
    write_json(attempt_root / "failure.json", payload)
    write_json(attempt_root / "status.json", payload, replace_existing=True)
    return payload


def worker(job: str | Path, *, allow_rerun: bool = False,
           rerun_reason: str | None = None) -> dict[str, Any]:
    """运行一个声明的训练 job，并持久化 terminal attempt 记录。"""
    job_path = Path(job).resolve()
    job_payload = read_json(job_path)
    forbidden_keys = _forbidden_training_keys(job_payload)
    if forbidden_keys:
        raise ExperimentError("worker job contains forbidden payload keys: %s" % forbidden_keys)
    required = ("job_id", "study_root", "model_id", "bundle_id", "source_snpfree_path",
                "source_snpfree_sha256", "source_snapshot_path", "source_snapshot_sha256",
                "synthetic_gate_path", "synthetic_gate_sha256", "synthetic_gate_status",
                "x0_path", "x0_file_sha256", "x0_coordinate_sha256", "fit")
    missing = [key for key in required if key not in job_payload]
    if missing:
        raise ExperimentError("job config missing required fields: %s" % missing)
    root = Path(str(job_payload["study_root"])).resolve()
    job_id = str(job_payload["job_id"])
    job_root = root / "jobs" / job_id
    if job_path != (job_root / "job.json").resolve():
        raise ExperimentError("job path is not the manifest-owned job config")
    manifest = read_json(root / "manifest.json")
    if not manifest.get("ready_for_worker", False):
        return {"status": manifest.get("status", "pending_inputs"), "job_id": job_id, "fit_called": False}
    manifest_rows = {str(row["job_id"]): row for row in manifest.get("jobs", [])}
    row = manifest_rows.get(job_id)
    if row is None or row.get("job_config_sha256") != sha256_file(job_path):
        raise ExperimentError("job config is not the hash-locked manifest version")
    manifest_release = str(manifest.get("protocol_release", {}).get("sha256", ""))
    if str(job_payload["protocol_release_sha256"]) != manifest_release:
        raise ExperimentError("job protocol release differs from controller manifest")
    if row.get("synthetic_gate_sha256") != job_payload.get("synthetic_gate_sha256"):
        raise ExperimentError("job synthetic gate differs from controller manifest")
    if str(job_payload["model_id"]) not in MODEL_IDS:
        raise ExperimentError("unknown model in job config")
    source_path = Path(str(job_payload["source_snpfree_path"])).resolve()
    if source_path != Path(str(manifest["source_hashes"]["input_path"])).resolve():
        raise ExperimentError("worker source path differs from controller source binding")

    claim = _claim_worker(root, job_id)
    if claim is None:
        available = _available_memory_bytes()
        status = "deferred_memavailable" if available is not None and available < MIN_AVAILABLE_BYTES else "deferred_max_parallel"
        payload = {"status": status, "job_id": job_id, "fit_called": False,
                   "mem_available_bytes": available, "max_parallel_workers": MAX_WORKERS}
        write_json(job_root / "deferred.json", payload, replace_existing=True)
        return payload
    started_clock = time.perf_counter()
    started_utc = utc_now()
    rss_start = _rss_bytes()
    logger: _JsonlLogger | None = None
    attempt_root: Path | None = None
    fit_called = False
    try:
        attempt_root, attempt_number, previous_status = _next_attempt(
            job_root, allow_rerun, rerun_reason)
        logger = _JsonlLogger(attempt_root / "worker.jsonl")
        active_snapshot_path = Path(str(job_payload["source_snapshot_path"])).resolve()
        active_snapshot_sha = str(job_payload["source_snapshot_sha256"])
        if previous_status is not None and allow_rerun:
            rerun_snapshot = build_source_snapshot(source_path)
            active_snapshot_path = attempt_root / "source_hashes.json"
            write_json(active_snapshot_path, rerun_snapshot)
            active_snapshot_sha = str(rerun_snapshot["snapshot_sha256"])
        running_payload = {
            "schema": "post020-fit-status-v1",
            "status": "running",
            "job_id": job_id,
            "model_id": job_payload["model_id"],
            "bundle_id": job_payload["bundle_id"],
            "attempt": attempt_number,
            "pid": os.getpid(),
            "started_utc": started_utc,
            "source_snapshot_sha256": active_snapshot_sha,
            "fit_called": False,
            "rerun_reason": rerun_reason,
        }
        write_json(attempt_root / "status.json", running_payload)
        write_json(job_root / "status.json", {
            "job_id": job_id,
            "status": "running",
            "attempt": attempt_number,
            "attempt_path": str(attempt_root),
            "selection_eligible": False,
            "fit_called": False,
            "updated_utc": utc_now(),
        }, replace_existing=True)
        logger.write("worker_started", attempt=attempt_number,
                     rss_start_bytes=rss_start,
                     mem_available_bytes=_available_memory_bytes(),
                     thread_env=_single_thread_env())
        os.environ.update(_single_thread_env())

        snapshot_path = active_snapshot_path
        if previous_status is None and sha256_file(snapshot_path) != str(manifest["source_hashes"]["sha256"]):
            raise ExperimentError("source snapshot differs from controller manifest")
        verify_source_snapshot(snapshot_path, active_snapshot_sha)
        gate = _synthetic_gate(str(job_payload["synthetic_gate_path"]),
                               str(job_payload["protocol_release_sha256"]))
        if gate.get("status") != "ready" or gate.get("sha256") != str(job_payload["synthetic_gate_sha256"]):
            raise ExperimentError("synthetic implementation gate is missing, invalid, or changed")
        if sha256_file(source_path) != str(job_payload["source_snpfree_sha256"]):
            raise ExperimentError("SNP-free source hash changed before worker load")
        data = contact_model.load_aggregate(source_path, bin_size=BIN_SIZE,
                                            verify_frozen_hash=True)
        if sha256_file(source_path) != str(job_payload["source_snpfree_sha256"]):
            raise ExperimentError("SNP-free source changed during worker load")
        cohort_audit = data.budget()
        if cohort_audit.get("raw_records") != EXPECTED_RAW_RECORDS or data.n_loci != EXPECTED_N_LOCI:
            raise ExperimentError("worker loaded a non-frozen cohort")
        x0_path = Path(str(job_payload["x0_path"])).resolve()
        if str(job_payload.get("x0_status")) != "ready" or not x0_path.is_file():
            raise ExperimentError("x0 input is not ready")
        if sha256_file(x0_path) != str(job_payload["x0_file_sha256"]):
            raise ExperimentError("x0 file hash changed before worker load")
        start = load_paired_start(x0_path, start_id=str(job_payload["bundle_id"]))
        if start.coordinate_sha256 != str(job_payload["x0_coordinate_sha256"]):
            raise ExperimentError("x0 coordinate hash changed before worker load")
        if start.coordinates.shape != (2, data.n_loci, 3):
            raise ExperimentError("x0 coordinate shape does not match frozen cohort")
        if float(start.p_init) != 0.75:
            raise ExperimentError("x0 p_init differs from frozen .75")
        logger.write("inputs_verified", source_sha256=sha256_file(source_path),
                     synthetic_gate_sha256=gate["sha256"],
                     x0_file_sha256=sha256_file(x0_path),
                     x0_coordinate_sha256=start.coordinate_sha256,
                     x0_max_radius=start.max_radius)
        fit_config = FitConfig(**dict(job_payload["fit"]))
        if fit_config.as_dict() != REAL_FIT_CONFIG.as_dict():
            raise ExperimentError("job fit config differs from released shared budget")
        logger.write("fit_start", fit=fit_config.as_dict())

        checkpoints: list[dict[str, Any]] = []

        def checkpoint_hook(checkpoint: Any) -> None:
            record = _write_checkpoint(
                root, str(job_payload["model_id"]), str(job_payload["bundle_id"]),
                attempt_root, checkpoint,
            )
            checkpoints.append(record)
            logger.write("checkpoint", iteration=int(checkpoint.iteration),
                         nfev=int(checkpoint.nfev), path=record["path"])

        def accepted_callback(entry: dict[str, Any]) -> None:
            logger.write("accepted", **entry)

        fit_called = True
        result = run_one_fit(
            data,
            start,
            str(job_payload["model_id"]),
            fit_config,
            checkpoint_hook=checkpoint_hook,
            callback=accepted_callback,
        )
        result.objective.validate_physical_coordinates(result.coordinates) if hasattr(result.objective, "validate_physical_coordinates") else contact_model.assert_inside_unit_ball(result.coordinates)
        endpoint_value, endpoint_gradient, endpoint_components = result.objective.evaluate(
            result.theta, need_gradient=True)
        if endpoint_gradient is None:
            raise ExperimentError("endpoint gradient evaluation returned None")
        endpoint_gradient = np.asarray(endpoint_gradient, dtype=np.float64)
        finite = (
            np.isfinite(float(endpoint_value))
            and np.all(np.isfinite(result.theta))
            and np.all(np.isfinite(result.coordinates))
            and np.all(np.isfinite(endpoint_gradient))
        )
        if not finite:
            raise ExperimentError("terminal endpoint contains a nonfinite value")
        gradient_l2 = float(np.linalg.norm(endpoint_gradient))
        gradient_inf = float(np.max(np.abs(endpoint_gradient)))
        if not math.isfinite(gradient_l2) or not math.isfinite(gradient_inf):
            raise ExperimentError("terminal gradient norm is nonfinite")
        theta_path = attempt_root / "final_theta.txt"
        _write_text_exact(theta_path, "".join("%.17g\n" % value for value in result.theta))
        npz_path = attempt_root / "final_state.npz"
        _write_npz_exact(
            npz_path,
            theta=np.asarray(result.theta, dtype=np.float64),
            raw_coordinates=np.asarray(result.raw_coordinates, dtype=np.float64),
            coordinates=np.asarray(result.coordinates, dtype=np.float64),
            endpoint_gradient=endpoint_gradient,
        )
        coordinate_path = attempt_root / "final_coordinates.3dg"
        coordinate_record = write_coordinates(
            coordinate_path, data, str(job_payload["model_id"]), result.coordinates,
        )
        stop_reason = _stop_reason(result, fit_config)
        terminal_status = _terminal_status(stop_reason)
        final_payload = {
            "schema": "post020-real-allele-fit-final-v1",
            "status": terminal_status,
            "job_id": job_id,
            "model_id": str(job_payload["model_id"]),
            "bundle_id": str(job_payload["bundle_id"]),
            "attempt": attempt_number,
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "runtime_seconds": float(time.perf_counter() - started_clock),
            "rss_start_bytes": int(rss_start),
            "rss_max_bytes": int(_rss_bytes()),
            "mem_available_end_bytes": _available_memory_bytes(),
            "source": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "snapshot_path": str(snapshot_path),
                "snapshot_sha256": active_snapshot_sha,
            },
            "synthetic_gate": {
                "path": str(job_payload["synthetic_gate_path"]),
                "sha256": str(job_payload["synthetic_gate_sha256"]),
            },
            "x0": {
                "path": str(x0_path),
                "file_sha256": sha256_file(x0_path),
                "coordinate_sha256": start.coordinate_sha256,
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
                "count_nll_normalized": float(endpoint_components["count_nll_normalized"]),
                "components": dict(endpoint_components),
                "gradient_l2": gradient_l2,
                "gradient_inf": gradient_inf,
                "endpoint_eval_with_gradient": True,
                "endpoint_value_exact_match": bool(float(endpoint_value) == float(result.fun)),
                "theta_sha256": sha256_array(result.theta),
                "coordinates_sha256": sha256_array(result.coordinates),
                "theta_text_sha256": sha256_file(theta_path),
                "state_npz_sha256": sha256_file(npz_path),
                "coordinates_file": coordinate_record,
                "map_diagnostics": getattr(result.objective, "map_diagnostics", lambda: {})(),
            },
            "history": result.history,
            "checkpoints": checkpoints,
            "selection_eligible": True,
            "solver_abnormal": terminal_status not in {"solver_converged", "budget_not_converged"},
            "reference_or_phase_read": False,
        }
        final_path = attempt_root / "final.json"
        write_json(final_path, final_payload)
        final_json_sha256 = sha256_file(final_path)
        status_payload = dict(final_payload)
        status_payload["final_json_sha256"] = final_json_sha256
        write_json(attempt_root / "status.json", status_payload, replace_existing=True)
        write_json(job_root / "status.json", {
            "job_id": job_id,
            "status": terminal_status,
            "attempt": attempt_number,
            "attempt_path": str(attempt_root),
            "final_path": str(final_path),
            "final_json_sha256": final_json_sha256,
            "selection_eligible": True,
            "fit_called": True,
            "updated_utc": utc_now(),
        }, replace_existing=True)
        logger.write("fit_terminal", status=terminal_status,
                     actual_stop_reason=stop_reason,
                     actual_nfev=int(result.nfev), njev=int(result.njev),
                     gradient_l2=gradient_l2, gradient_inf=gradient_inf)
        _append_registry(root, {
            "schema": "post020-allele-ablation-artifact-registry-v1",
            "recorded_utc": utc_now(),
            "run_root": str(root),
            "scope": "real",
            "variant_id": str(job_payload["model_id"]),
            "bundle_or_fixture_id": str(job_payload["bundle_id"]),
            "status": terminal_status,
            "attempt": attempt_number,
            "artifact_path": str(final_path),
            "artifact_sha256": final_json_sha256,
            "source_snapshot_sha256": active_snapshot_sha,
            "x0_file_sha256": sha256_file(x0_path),
            "fit_called": True,
        })
        return final_payload
    except Exception as exc:
        if attempt_root is None:
            raise
        failure_text = str(exc).lower()
        if "nonfinite" in failure_text or isinstance(exc, FloatingPointError):
            failure_status = "failed_nonfinite"
        elif "source" in failure_text or "hash" in failure_text:
            failure_status = "failed_source_snapshot"
        elif "coordinate" in failure_text or "unit-ball" in failure_text or "x0" in failure_text:
            failure_status = "failed_input"
        else:
            failure_status = "failed_exception"
        payload = _write_failure(
            attempt_root, job_payload, failure_status, exc, started_utc,
            rss_start, time.perf_counter() - started_clock, fit_called,
        )
        write_json(job_root / "status.json", {
            "job_id": job_id,
            "status": failure_status,
            "attempt": int(attempt_root.name.split("-")[-1]),
            "attempt_path": str(attempt_root),
            "selection_eligible": False,
            "fit_called": fit_called,
            "updated_utc": utc_now(),
        }, replace_existing=True)
        if logger is not None:
            logger.write("worker_failed", status=failure_status,
                         error_type=type(exc).__name__, error=str(exc))
        _append_registry(root, {
            "schema": "post020-allele-ablation-artifact-registry-v1",
            "recorded_utc": utc_now(),
            "run_root": str(root),
            "scope": "real",
            "variant_id": str(job_payload["model_id"]),
            "bundle_or_fixture_id": str(job_payload["bundle_id"]),
            "status": failure_status,
            "attempt": int(attempt_root.name.split("-")[-1]),
            "artifact_path": str(attempt_root / "failure.json"),
            "artifact_sha256": sha256_file(attempt_root / "failure.json"),
            "source_snapshot_sha256": locals().get(
                "active_snapshot_sha", job_payload.get("source_snapshot_sha256")),
            "fit_called": fit_called,
        })
        return payload
    finally:
        if logger is not None:
            logger.close()
        if claim is not None and claim.exists():
            claim.unlink()


def _job_status(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    job_id = str(row["job_id"])
    path = root / "jobs" / job_id / "status.json"
    if path.is_file():
        return read_json(path)
    deferred = root / "jobs" / job_id / "deferred.json"
    if deferred.is_file():
        return read_json(deferred)
    return {
        "job_id": job_id,
        "status": row.get("status", "pending"),
        "fit_called": False,
        "selection_eligible": False,
    }


def _load_final_for_status(root: Path, status: Mapping[str, Any]) -> dict[str, Any] | None:
    final_path = status.get("final_path")
    if not final_path:
        attempt_path = status.get("attempt_path")
        if attempt_path:
            final_path = str(Path(str(attempt_path)) / "final.json")
    if not final_path or not Path(str(final_path)).is_file():
        return None
    return read_json(Path(str(final_path)))


def _selection_row(row: Mapping[str, Any], status: Mapping[str, Any],
                   final: Mapping[str, Any] | None) -> dict[str, Any]:
    base = {
        "job_id": row["job_id"],
        "model_id": row["model_id"],
        "bundle_id": row["bundle_id"],
        "status": status.get("status", row.get("status", "n/a")),
        "fit_called": bool(status.get("fit_called", False)),
        "selection_eligible": False,
        "solver_abnormal": None,
        "count_nll_normalized": None,
        "actual_nfev": None,
        "actual_stop_reason": None,
        "excluded_reason": "missing_terminal_result",
    }
    if final is None:
        return base
    final_block = final.get("final", {})
    value = final_block.get("count_nll_normalized")
    finite_count = False
    try:
        finite_count = math.isfinite(float(value))
    except (TypeError, ValueError):
        finite_count = False
    eligible = bool(final.get("selection_eligible", False)) and finite_count
    base.update({
        "status": final.get("status", base["status"]),
        "fit_called": True,
        "selection_eligible": eligible,
        "solver_abnormal": bool(final.get("solver_abnormal", False)),
        "count_nll_normalized": float(value) if finite_count else None,
        "actual_nfev": final.get("solver", {}).get("actual_nfev"),
        "actual_stop_reason": final.get("solver", {}).get("actual_stop_reason"),
        "excluded_reason": None if eligible else (
            "nonfinite_count" if not finite_count else "no_valid_coords_or_endpoint"),
    })
    return base


def finalize(run_root: str | Path) -> dict[str, Any]:
    """收集每个已注册 arm，并只在变体内选择代表项。"""
    root = Path(run_root).resolve()
    manifest = read_json(root / "manifest.json")
    rows = list(manifest.get("jobs", []))
    if len(rows) != N_JOBS:
        raise ExperimentError("manifest does not retain all 15 registered jobs")
    statuses = []
    for row in rows:
        status = _job_status(root, row)
        final = _load_final_for_status(root, status)
        statuses.append(_selection_row(row, status, final))
    selections = []
    for model_id in MODEL_IDS:
        candidates = [item for item in statuses if item["model_id"] == model_id]
        finite = [item for item in candidates if item["selection_eligible"]]
        selected = None
        if finite:
            proxies = [SimpleNamespace(
                model_id=item["model_id"],
                start_id=item["bundle_id"],
                count_nll_normalized=item["count_nll_normalized"],
            ) for item in finite]
            try:
                selected_proxy = select_best_start(proxies, tie_tol=SELECTION_TIE_TOL)
                selected = next(item for item in finite
                                if item["bundle_id"] == selected_proxy.start_id)
            except (PairedRunError, ValueError) as exc:
                selected = None
                selection_error = str(exc)
            else:
                selection_error = None
        else:
            selection_error = "no finite valid terminal candidates"
        selections.append({
            "model_id": model_id,
            "status": "selected" if selected is not None else "n/a",
            "selected_bundle_id": None if selected is None else selected["bundle_id"],
            "selected_job_id": None if selected is None else selected["job_id"],
            "selected_solver_abnormal": None if selected is None else selected["solver_abnormal"],
            "tie_tol": SELECTION_TIE_TOL,
            "criterion": "count_nll_normalized",
            "selection_error": selection_error,
            "candidates": candidates,
        })
    active = sorted(path.name for path in (root / "active").glob("*.json"))
    terminal = all(item["status"] in TERMINAL_STATUSES for item in statuses)
    payload = {
        "schema": "post020-real-allele-finalize-v1",
        "status": "complete" if terminal and not active else "partial",
        "finalized_utc": utc_now(),
        "study_root": str(root),
        "all_15_manifest_arms_preserved": True,
        "terminal_job_count": int(sum(item["status"] in TERMINAL_STATUSES for item in statuses)),
        "job_count": len(statuses),
        "active_jobs": active,
        "jobs": statuses,
        "selection": selections,
        "reference_or_phase_read": False,
    }
    write_json(root / "selection.json", {
        "schema": "post020-real-allele-selection-v1",
        "status": payload["status"],
        "tie_tol": SELECTION_TIE_TOL,
        "within_variant_only": True,
        "variants": selections,
        "all_15_manifest_arms_preserved": True,
    }, replace_existing=True)
    write_json(root / "finalize.json", payload, replace_existing=True)
    write_json(root / "termination_audit.json", {
        "schema": "post020-real-allele-termination-audit-v1",
        "status": payload["status"],
        "finalized_utc": payload["finalized_utc"],
        "job_count": payload["job_count"],
        "terminal_job_count": payload["terminal_job_count"],
        "active_jobs": active,
        "jobs": [
            {
                "job_id": item["job_id"],
                "status": item["status"],
                "fit_called": item["fit_called"],
                "selection_eligible": item["selection_eligible"],
                "actual_nfev": item["actual_nfev"],
                "actual_stop_reason": item["actual_stop_reason"],
            }
            for item in statuses
        ],
        "all_15_manifest_arms_preserved": True,
    }, replace_existing=True)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="write a no-fit 15-job manifest")
    prepare_parser.add_argument("--release", required=True, dest="release_sha256")
    prepare_parser.add_argument("--protocol-json", default=str(PROTOCOL_JSON))
    prepare_parser.add_argument("--protocol-md", default=str(PROTOCOL_MD))
    prepare_parser.add_argument("--synthetic-gate", default=None)
    prepare_parser.add_argument("--starts-root", default=str(DEFAULT_START_ROOT))
    prepare_parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    prepare_parser.add_argument("--run-root", default=None)
    worker_parser = sub.add_parser("worker", help="run one hash-locked training job")
    worker_parser.add_argument("--job", required=True)
    worker_parser.add_argument("--rerun-reason", default=None)
    finalize_parser = sub.add_parser("finalize", help="collect terminal job attempts")
    finalize_parser.add_argument("--run-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(
                run_root=args.run_root,
                release_sha256=args.release_sha256,
                protocol_json=args.protocol_json,
                protocol_md=args.protocol_md,
                synthetic_gate_path=args.synthetic_gate,
                starts_root=args.starts_root,
                source_path=args.source,
            )
            print(json.dumps(_jsonable(result), sort_keys=True))
            return 0
        if args.command == "worker":
            result = worker(args.job, allow_rerun=args.rerun_reason is not None,
                            rerun_reason=args.rerun_reason)
            print(json.dumps(_jsonable(result), sort_keys=True))
            return 0 if not str(result.get("status", "")).startswith("failed") else 1
        result = finalize(args.run_root)
        print(json.dumps(_jsonable({
            "status": result["status"],
            "study_root": result["study_root"],
            "terminal_job_count": result["terminal_job_count"],
            "job_count": result["job_count"],
        }), sort_keys=True))
        return 0
    except (ExperimentError, PairedRunError, AssertionError, ValueError, FileExistsError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


__all__ = [
    "BUNDLE_IDS",
    "EXPECTED_SOURCE_SHA256",
    "ExperimentError",
    "REAL_FIT_CONFIG",
    "X0Record",
    "allocate_study_root",
    "build_source_snapshot",
    "finalize",
    "load_paired_start",
    "main",
    "prepare",
    "sha256_array",
    "sha256_file",
    "verify_protocol_release",
    "verify_source_snapshot",
    "worker",
]


if __name__ == "__main__":
    raise SystemExit(main())
