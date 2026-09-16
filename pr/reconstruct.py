"""正式的、无 phase Reconstruction V1 生产运行器。

本模块只准备并执行冻结的全 contact 坐标训练路径。绝不导入 phase labels、reference 坐标、oracle 坐标 loader 或评估侧指标函数。候选选择严格使用最终坐标序列化并回读后的冻结无 label count objective。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import datetime as dt
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid
from typing import Any, Callable, Mapping

import numpy as np

from . import contact_model, genome, joint_fit, reconstruction_init, reconstruction_report
from .gate import sha256_file
from .paths import ROOT, SNPFREE


RUN_CONFIG_SCHEMA_VERSION = "reconstruction-v1-production-run-v1"
PROTOCOL_MANIFEST_SCHEMA_VERSION = "reconstruction-protocol-snapshot-v1"
FROZEN_PROTOCOL_SHA256 = "cf2ae36fdd4dec6e05d32001a5ef498add7bb62e5908a4373b2316fc380ed8e5"
SELECTION_TIE_TOLERANCE = 1e-9
CHECKPOINT_EVERY = 10
FINAL_BIN_SIZE = 1_000_000
STRICT_INTERIOR_LIMIT = 1.0 - 1e-12


class ReconstructionError(RuntimeError):
    """冻结的正式训练不变量违反时抛出。"""


@dataclass(frozen=True)
class StageSpec:
    """一次预注册的多分辨率优化层。"""

    label: str
    bin_size: int
    maxiter: int

    @property
    def maxfun(self) -> int:
        return 3 * self.maxiter + 30


@dataclass(frozen=True)
class CandidateSpec:
    """预注册的无 label 初始化 family。"""

    candidate_id: str
    initialization_candidate: str
    base_seed: int


DEFAULT_STAGES = (
    StageSpec("5m", 5_000_000, 300),
    StageSpec("2m", 2_000_000, 200),
    StageSpec("1m", 1_000_000, 240),
)
DEFAULT_CANDIDATES = (
    CandidateSpec("consensus_joint", "consensus", 1103),
    CandidateSpec("random_joint", "random", 2207),
)
TRAINING_CODE_SOURCES = (
    "run.py",
    "pr/reconstruct.py",
    "pr/contact_model.py",
    "pr/joint_fit.py",
    "pr/reconstruction_init.py",
    "pr/genome.py",
    "pr/pairs7.py",
    "pr/paths.py",
    "pr/gate.py",
)
PROTOCOL_SOURCES = (
    "docs/RECONSTRUCTION_V1_PROTOCOL.md",
    "docs/RECONSTRUCTION_V1_INITIALIZATION.md",
    "docs/RECONSTRUCTION_V1_RUN.md",
)


@dataclass(frozen=True)
class RunContext:
    """通用 runner 和测试 fixtures 所需的全部不可变输入。"""

    headers: tuple[tuple[str, int], ...]
    data_path: str
    data_sha256: str
    cohort: Mapping[str, Any]
    source_assets: Mapping[str, Mapping[str, Any]]
    baseline_assets: tuple[Mapping[str, Any], ...]
    stages: tuple[StageSpec, ...] = DEFAULT_STAGES
    candidates: tuple[CandidateSpec, ...] = DEFAULT_CANDIDATES
    weights: Mapping[str, float] = field(default_factory=lambda: {
        "count": 1.0,
        "p_prior": 1.0,
        "bond": 1.0,
        "repulsion": 1.0,
        "bend": 0.01,
    })
    strict_p9016: bool = True


@dataclass(frozen=True)
class RunnerHooks:
    """仅供 synthetic tests 使用的可注入、无 phase 协作者。"""

    load_data: Callable[[int], contact_model.AggregatedContacts]
    initialize: Callable[[str, tuple[str, ...], tuple[int, ...], int], Mapping[str, Any]]
    warm_start: Callable[[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[int, ...], int, int], Mapping[str, Any]]
    fit: Callable[..., Any]


def _default_load_data(bin_size: int) -> contact_model.AggregatedContacts:
    return contact_model.load_frozen_p9016_aggregate(bin_size)


def _default_initialize(candidate: str, names: tuple[str, ...], lengths: tuple[int, ...],
                        bin_size: int) -> Mapping[str, Any]:
    return reconstruction_init.initialize_approved_candidate(candidate, names, lengths, bin_size)


def _default_warm_start(coords: np.ndarray, positions: np.ndarray, chromosome_index: np.ndarray,
                        names: tuple[str, ...], lengths: tuple[int, ...], bin_size: int,
                        base_seed: int) -> Mapping[str, Any]:
    return reconstruction_init.warm_start_from_layer(
        coords, positions, chromosome_index, names, lengths, bin_size, base_seed
    )


def _default_fit(objective: contact_model.JointObjective, initial_y: np.ndarray, **kwargs: Any) -> Any:
    return joint_fit.fit_joint(objective, initial_y, **kwargs)


DEFAULT_HOOKS = RunnerHooks(_default_load_data, _default_initialize, _default_warm_start, _default_fit)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _json_ready(value: Any) -> Any:
    """转换 NumPy 值，并拒绝非有限的正式 metadata。"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ReconstructionError("formal metadata cannot contain non-finite values")
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


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_new_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    _write_new_bytes(path, _json_bytes(payload))


def _atomic_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.%s.tmp" % (path.name, os.getpid(), uuid.uuid4().hex))
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative_to_run(outdir: Path, path: Path) -> str:
    return str(path.resolve().relative_to(outdir.resolve()))


def _path_from_run(outdir: Path, path_text: str) -> Path:
    candidate = Path(path_text)
    return candidate if candidate.is_absolute() else outdir / candidate


def _copy_new(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, target.open("xb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
        target_handle.flush()
        os.fsync(target_handle.fileno())


def _snapshot_files(outdir: Path, *, manifest_name: str, snapshot_root: str,
                    schema_version: str, source_paths: tuple[str, ...]) -> dict[str, Any]:
    rows = []
    for source_path in source_paths:
        source = Path(ROOT) / source_path
        if not source.is_file():
            raise ReconstructionError("cannot snapshot missing frozen source %s" % source)
        destination = outdir / snapshot_root / source_path
        _copy_new(source, destination)
        rows.append({
            "source_path": source_path,
            "snapshot_path": _relative_to_run(outdir, destination),
            "sha256": sha256_file(destination),
        })
    manifest = {"schema_version": schema_version, "files": rows}
    manifest_path = outdir / "provenance" / manifest_name
    _write_new_json(manifest_path, manifest)
    return {
        "path": _relative_to_run(outdir, manifest_path),
        "sha256": sha256_file(manifest_path),
        "manifest": manifest,
    }


def _run_relative_path(path_text: Any, label: str) -> str:
    """核验 manifest 路径，不查阅其历史 source。"""
    if not isinstance(path_text, str) or not path_text:
        raise ReconstructionError("%s must be a non-empty run-relative path" % label)
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise ReconstructionError("%s must be run-relative: %s" % (label, path_text))
    return path_text


def _verify_snapshot_manifest(outdir: Path, reference: Mapping[str, Any], schema_version: str) -> dict[str, Any]:
    try:
        manifest_path_text = _run_relative_path(reference["path"], "snapshot manifest path")
        expected_manifest_sha = str(reference["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionError("snapshot manifest reference is malformed") from exc
    path = outdir / manifest_path_text
    try:
        path.resolve().relative_to(outdir.resolve())
    except ValueError as exc:
        raise ReconstructionError("snapshot manifest escapes formal run directory") from exc
    if not path.is_file():
        raise ReconstructionError("snapshot manifest is unavailable: %s" % path)
    if sha256_file(path) != expected_manifest_sha:
        raise ReconstructionError("snapshot manifest hash mismatch: %s" % path)
    try:
        with path.open() as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionError("cannot read snapshot manifest %s: %s" % (path, exc)) from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != schema_version:
        raise ReconstructionError("invalid snapshot manifest %s" % path)
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ReconstructionError("snapshot manifest files must be a list: %s" % path)
    checked = []
    seen_source = set()
    seen_snapshot = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ReconstructionError("snapshot manifest member %d is invalid" % index)
        try:
            source_path = _run_relative_path(row["source_path"],
                                             "snapshot manifest member %d source_path" % index)
            snapshot_path_text = _run_relative_path(row["snapshot_path"],
                                                    "snapshot manifest member %d snapshot_path" % index)
            expected_sha = str(row["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReconstructionError("snapshot manifest member %d is invalid" % index) from exc
        if (len(expected_sha) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha)):
            raise ReconstructionError("snapshot manifest member %d has an invalid SHA256" % index)
        if source_path in seen_source or snapshot_path_text in seen_snapshot:
            raise ReconstructionError("snapshot manifest contains duplicate source or snapshot paths")
        snapshot = outdir / snapshot_path_text
        try:
            snapshot.resolve().relative_to(outdir.resolve())
        except ValueError as exc:
            raise ReconstructionError("snapshot member escapes formal run directory") from exc
        if not snapshot.is_file() or sha256_file(snapshot) != expected_sha:
            raise ReconstructionError("snapshot member hash mismatch: %s" % snapshot)
        seen_source.add(source_path)
        seen_snapshot.add(snapshot_path_text)
        checked.append({"source_path": source_path, "snapshot_path": snapshot_path_text,
                        "sha256": expected_sha})
    if not checked:
        raise ReconstructionError("snapshot manifest must contain at least one member")
    return {"path": str(path), "members": checked}


def _frozen_source_assets() -> tuple[dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
    """只读取 014 gate metadata；绝不打开 oracle 坐标。"""
    specs = reconstruction_init.APPROVED_SOURCES
    source_assets: dict[str, dict[str, Any]] = {}
    gate_paths = {Path(str(spec["gate_path"])).resolve() for spec in specs.values()}
    if len(gate_paths) != 1:
        raise ReconstructionError("approved initialization candidates must share one frozen source gate")
    gate_path = next(iter(gate_paths))
    if not gate_path.is_file():
        raise ReconstructionError("frozen 014 source gate is unavailable: %s" % gate_path)
    gate_sha = sha256_file(gate_path)
    with gate_path.open() as handle:
        gate_rows = json.load(handle)
    if not isinstance(gate_rows, list):
        raise ReconstructionError("frozen 014 source gate must be a list")
    by_tag = {row.get("tag"): row for row in gate_rows if isinstance(row, dict)}
    for candidate, spec in specs.items():
        row = by_tag.get(candidate)
        if row is None:
            raise ReconstructionError("frozen source gate has no %s entry" % candidate)
        source_path = Path(str(spec["path"])).resolve()
        if row.get("stage") != "blind" or row.get("sha256") != spec["sha256"]:
            raise ReconstructionError("frozen source gate does not match approved %s metadata" % candidate)
        if sha256_file(source_path) != spec["sha256"]:
            raise ReconstructionError("approved blind source SHA256 mismatch: %s" % source_path)
        source_assets[candidate] = {
            "tag": candidate,
            "stage": "blind",
            "path": str(source_path),
            "sha256": str(spec["sha256"]),
            "gate_path": str(gate_path),
            "gate_sha256": gate_sha,
            "base_seed": int(spec["base_seed"]),
        }
    baseline_assets = []
    for tag, role in (("consensus", "blind_baseline"), ("random", "blind_baseline"),
                      ("oracle", "evaluation_ceiling")):
        row = by_tag.get(tag)
        if row is None:
            raise ReconstructionError("frozen source gate has no %s baseline metadata" % tag)
        # 坐标路径/摘要仅从 source gate 复制为 provenance。
        # 尤其不会打开或哈希 oracle 坐标数据。
        baseline_assets.append({
            "id": "baseline_014_%s" % tag,
            "tag": tag,
            "role": role,
            "source_gate": {"path": str(gate_path), "sha256": gate_sha},
            "source_coordinate": {
                "path": str(row.get("path")),
                "sha256": str(row.get("sha256")),
                "stage": str(row.get("stage")),
            },
        })
    return source_assets, tuple(baseline_assets)


def production_context() -> RunContext:
    """构建冻结 V1 runner 唯一允许的真实数据 context。"""
    input_identity = contact_model.verify_frozen_snpfree(SNPFREE)
    headers = tuple((str(name), int(length)) for name, length in genome.chrom_lengths(SNPFREE))
    source_assets, baseline_assets = _frozen_source_assets()
    cohort = {
        "sample_id": "P9016",
        "biological_samples": 1,
        "raw_contacts": contact_model.FROZEN_P9016_RECORDS,
        "intra_contacts": contact_model.FROZEN_P9016_CIS_RECORDS,
        "inter_contacts": contact_model.FROZEN_P9016_INTER_RECORDS,
        "snpfree_sha256": input_identity["snpfree_sha256"],
    }
    return RunContext(
        headers=headers,
        data_path=str(Path(SNPFREE).resolve()),
        data_sha256=input_identity["snpfree_sha256"],
        cohort=cohort,
        source_assets=source_assets,
        baseline_assets=baseline_assets,
        strict_p9016=True,
    )


def _validate_context(context: RunContext) -> None:
    if not context.headers or len({name for name, _ in context.headers}) != len(context.headers):
        raise ReconstructionError("formal context must retain unique header-order chromosomes")
    if any(int(length) <= 0 for _name, length in context.headers):
        raise ReconstructionError("formal context has a nonpositive chromosome length")
    if not context.stages or context.stages[-1].bin_size != FINAL_BIN_SIZE:
        raise ReconstructionError("the frozen schedule must end at the 1 Mb final grid")
    if tuple(stage.label for stage in context.stages) != ("5m", "2m", "1m"):
        raise ReconstructionError("the frozen V1 schedule must be 5m, 2m, then 1m")
    if tuple(stage.maxiter for stage in context.stages) != (300, 200, 240):
        raise ReconstructionError("the frozen V1 iteration budgets are 300/200/240")
    if tuple(candidate.candidate_id for candidate in context.candidates) != (
            "consensus_joint", "random_joint"):
        raise ReconstructionError("candidate order must be consensus_joint then random_joint")
    if tuple(candidate.initialization_candidate for candidate in context.candidates) != (
            "consensus", "random"):
        raise ReconstructionError("candidate initialization families are frozen")
    if tuple(candidate.base_seed for candidate in context.candidates) != (1103, 2207):
        raise ReconstructionError("candidate base seeds are frozen")
    if set(context.weights) != {"count", "p_prior", "bond", "repulsion", "bend"}:
        raise ReconstructionError("objective weight inventory changed")
    expected_weights = {"count": 1.0, "p_prior": 1.0, "bond": 1.0,
                        "repulsion": 1.0, "bend": 0.01}
    if {key: float(value) for key, value in context.weights.items()} != expected_weights:
        raise ReconstructionError("V1 objective weights are frozen")
    protocol_path = Path(ROOT) / "docs/RECONSTRUCTION_V1_PROTOCOL.md"
    if not protocol_path.is_file() or sha256_file(protocol_path) != FROZEN_PROTOCOL_SHA256:
        raise ReconstructionError("frozen Reconstruction V1 protocol SHA256 conflict")
    data_path = Path(context.data_path)
    if not data_path.is_file() or sha256_file(data_path) != context.data_sha256:
        raise ReconstructionError("SNP-free input SHA256 conflict before formal output creation")
    if context.cohort.get("snpfree_sha256") != context.data_sha256:
        raise ReconstructionError("cohort and input SHA256 disagree")
    if context.strict_p9016:
        expected = {
            "sample_id": "P9016",
            "biological_samples": 1,
            "raw_contacts": contact_model.FROZEN_P9016_RECORDS,
            "intra_contacts": contact_model.FROZEN_P9016_CIS_RECORDS,
            "inter_contacts": contact_model.FROZEN_P9016_INTER_RECORDS,
            "snpfree_sha256": contact_model.FROZEN_P9016_SNPFREE_SHA256,
        }
        if dict(context.cohort) != expected:
            raise ReconstructionError("strict P9016 cohort identity changed")
        if len(context.headers) != 20:
            raise ReconstructionError("strict P9016 run must retain all 20 chromosomes")

    expected_tags = {"consensus", "random"}
    if set(context.source_assets) != expected_tags:
        raise ReconstructionError("formal source inventory must contain consensus and random only")
    for candidate in context.candidates:
        source = context.source_assets[candidate.initialization_candidate]
        if (source.get("tag") != candidate.initialization_candidate or source.get("stage") != "blind"
                or int(source.get("base_seed", -1)) != candidate.base_seed):
            raise ReconstructionError("candidate source metadata conflicts with frozen preregistration")
        source_path = Path(str(source.get("path", "")))
        gate_path = Path(str(source.get("gate_path", "")))
        if not source_path.is_file() or not gate_path.is_file():
            raise ReconstructionError("candidate source or source gate is unavailable")
        if sha256_file(source_path) != source.get("sha256"):
            raise ReconstructionError("candidate source SHA256 conflict for %s" % candidate.candidate_id)
        if sha256_file(gate_path) != source.get("gate_sha256"):
            raise ReconstructionError("candidate source gate SHA256 conflict for %s" % candidate.candidate_id)
        with gate_path.open() as handle:
            entries = json.load(handle)
        matching = [row for row in entries if isinstance(row, dict)
                    and row.get("tag") == candidate.initialization_candidate]
        if len(matching) != 1 or matching[0].get("stage") != "blind" or matching[0].get("sha256") != source["sha256"]:
            raise ReconstructionError("candidate source gate entry conflicts with frozen source")
        gate_coordinate_path = Path(str(matching[0].get("path", "")))
        if not gate_coordinate_path.is_absolute():
            gate_coordinate_path = Path(ROOT) / gate_coordinate_path
        if gate_coordinate_path.resolve() != source_path.resolve():
            raise ReconstructionError("candidate source gate path conflicts with frozen source")
        if context.strict_p9016:
            approved = reconstruction_init.APPROVED_SOURCES[candidate.initialization_candidate]
            if (source_path.resolve() != Path(str(approved["path"])).resolve()
                    or gate_path.resolve() != Path(str(approved["gate_path"])).resolve()
                    or source.get("sha256") != approved["sha256"]):
                raise ReconstructionError("strict P9016 source is not the approved 014 blind asset")
    baseline_tags = {entry.get("tag") for entry in context.baseline_assets}
    if baseline_tags != {"consensus", "random", "oracle"}:
        raise ReconstructionError("baseline provenance must name consensus/random/oracle exactly once")


def _model_contract(context: RunContext) -> tuple[dict[str, Any], dict[str, Any]]:
    final_loci = sum((int(length) + FINAL_BIN_SIZE - 1) // FINAL_BIN_SIZE
                      for _name, length in context.headers)
    weights = {key: float(value) for key, value in context.weights.items()}
    prior_configuration = {
        "objective_weights": weights,
        "p_floor": float(contact_model.P_FLOOR),
        "p_prior_strength": float(contact_model.P_PRIOR_STRENGTH),
        "prior_component_names": list(reconstruction_report.PRIOR_REPORT_FIELDS),
        "bond_target_interval_l0": [0.75, 1.25],
        "repulsion_radius_l0": 2.0,
        "bend_weight": weights["bend"],
    }
    model_definition = {
        "model_name": "reconstruction_v1_continuous_joint_objective",
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "contact_model_source_sha256": sha256_file(Path(ROOT) / "pr/contact_model.py"),
        "joint_fit_source_sha256": sha256_file(Path(ROOT) / "pr/joint_fit.py"),
        "finite_kernel": {
            "name": "bounded_kernel",
            "epsilon": float(contact_model.EPSILON),
            "radius": "2*l0",
            "formula": "epsilon + (1-epsilon)*(1+||delta||^2/r0^2)^-2",
        },
        "exposure": {
            "mode": "observed_endpoint_sqrt_plus_10_mean_one",
            "formula": "sqrt(endpoint_count + 10) / full_grid_mean",
        },
        "same_bin_layer": "per_bin_saturated_poisson_nuisance",
        "coordinate_transform": "sphere_forward_unit_ball",
        "prior_configuration": prior_configuration,
    }
    prior_sha = _sha256_json(prior_configuration)
    model_signature = _sha256_json(model_definition)
    contract = {
        "model_signature": model_signature,
        "prior_config_sha256": prior_sha,
        "parameter_dimension": 6 * final_loci + 1,
        "bin_size_bp": FINAL_BIN_SIZE,
        "prior_component_names": list(reconstruction_report.PRIOR_REPORT_FIELDS),
    }
    return contract, model_definition


def _build_config(context: RunContext, *, workers: int, threads: int) -> dict[str, Any]:
    model_contract, model_definition = _model_contract(context)
    return {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "purpose": "formal_phase_free_training_only",
        "frozen_protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "cohort": dict(context.cohort),
        "coordinate_grid": {
            "training_origin_bp": 0,
            "full_grid": True,
            "final_bin_size_bp": FINAL_BIN_SIZE,
            "nuclear_radius": 1.0,
            "coordinate_units": "dimensionless_R1",
            "metric_grid": {
                "origin_bp": 3_000_000,
                "used_for_training": False,
                "note": "legacy evaluation grid remains separate from the full 0-based training grid",
            },
            "chromosomes": [{"name": name, "length_bp": int(length)} for name, length in context.headers],
        },
        "input": {
            "path": str(Path(context.data_path).resolve()),
            "sha256": context.data_sha256,
            "raw_source_sha256": contact_model.FROZEN_P9016_RAW_SHA256 if context.strict_p9016 else None,
            "training_columns": ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"],
            "uses_all_records": True,
        },
        "initialization_sources": _json_ready(context.source_assets),
        "baseline_assets": _json_ready(context.baseline_assets),
        "preregistered_candidate_ids": [candidate.candidate_id for candidate in context.candidates],
        "preregistered_attempts": [
            {"attempt_id": "%s-%s" % (candidate.candidate_id, stage.label),
             "candidate_id": candidate.candidate_id, "stage": stage.label}
            for candidate in context.candidates for stage in context.stages
        ],
        "candidates": [
            {"id": candidate.candidate_id, "initialization": candidate.initialization_candidate,
             "base_seed": candidate.base_seed}
            for candidate in context.candidates
        ],
        "optimization": {
            "stages": [
                {"label": stage.label, "bin_size_bp": stage.bin_size, "maxiter": stage.maxiter,
                 "maxfun": stage.maxfun, "maxls": 20, "ftol": 1e-10, "gtol": 1e-6,
                 "checkpoint_every_accepted_iterations": CHECKPOINT_EVERY}
                for stage in context.stages
            ],
            "p_init_first_layer": 0.75,
            "carry_q_raw_between_layers": True,
            "selection_after_all_candidates": True,
        },
        "model_definition": model_definition,
        "model_contract": model_contract,
        "runtime": {
            "candidate_workers": workers,
            "thread_env": {name: str(threads) for name in
                           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
        "training_boundary": {
            "phase_used": False,
            "reference_used": False,
            "oracle_coordinates_opened": False,
            "native_fdg_started": False,
        },
    }


def _make_grid(headers: tuple[tuple[str, int], ...]) -> reconstruction_report.FullGrid:
    return reconstruction_report.FullGrid(
        tuple(reconstruction_report.Chromosome(name, int(length)) for name, length in headers),
        FINAL_BIN_SIZE,
        0,
    )


def _configure_threads(threads: int) -> None:
    if threads not in (1, 2):
        raise ReconstructionError("formal runner permits only one or two BLAS/OpenMP threads")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)


def _create_fresh_layout(outdir: Path) -> None:
    if os.path.lexists(outdir):
        raise FileExistsError("refusing to reuse existing formal output directory: %s" % outdir)
    outdir.mkdir(parents=True)
    for relative in ("coords", "checkpoints", "logs", "work", "plots", "stages", "run_status/candidates",
                     "provenance/training-code", "provenance/protocol"):
        (outdir / relative).mkdir(parents=True, exist_ok=False)
    _write_new_bytes(outdir / "run_status" / "attempts.jsonl", b"")


def _write_static_run_files(outdir: Path, context: RunContext, config: Mapping[str, Any]) -> dict[str, Any]:
    config_path = outdir / "config.json"
    _write_new_json(config_path, dict(config))
    code_snapshot = _snapshot_files(
        outdir,
        manifest_name="training-code-manifest.json",
        snapshot_root="provenance/training-code",
        schema_version=reconstruction_report.TRAINING_CODE_MANIFEST_SCHEMA_VERSION,
        source_paths=TRAINING_CODE_SOURCES,
    )
    protocol_snapshot = _snapshot_files(
        outdir,
        manifest_name="protocol-manifest.json",
        snapshot_root="provenance/protocol",
        schema_version=PROTOCOL_MANIFEST_SCHEMA_VERSION,
        source_paths=PROTOCOL_SOURCES,
    )
    grid = _make_grid(context.headers)
    track_map = reconstruction_report.write_track_map(outdir / "track_map.json", grid)
    input_version = {
        "path": str(Path(context.data_path).resolve()),
        "sha256": context.data_sha256,
        "hash_source": "file_bytes",
        "cohort": dict(context.cohort),
        "raw_source_sha256": contact_model.FROZEN_P9016_RAW_SHA256 if context.strict_p9016 else None,
    }
    _write_new_json(outdir / "input_version.json", input_version)
    readme = """# Reconstruction V1 正式训练记录\n\n本目录是无 phase 的坐标训练记录，使用冻结的 SNP-free P9016 输入、预注册的 `consensus_joint` 与 `random_joint` 初始化，以及 5 Mb -> 2 Mb -> 1 Mb 调度。动态执行状态位于 `run_status/`；不可变训练配置和快照位于 `config.json` 与 `provenance/`。训练 runner 不会打开 phase labels、reference 坐标或 oracle 坐标 payload。\n"""
    _write_new_bytes(outdir / "README.md", readme.encode("utf-8"))
    _atomic_json(outdir / "run_status" / "run_status.json", {
        "status": "prepared",
        "created_at_utc": _utc_now(),
        "candidates": {candidate.candidate_id: "pending" for candidate in context.candidates},
    })
    return {
        "config": {"path": _relative_to_run(outdir, config_path), "sha256": sha256_file(config_path),
                   "hash_source": "file_bytes"},
        "code": {"path": code_snapshot["path"], "sha256": code_snapshot["sha256"],
                 "hash_source": "snapshot_manifest"},
        "protocol": {"path": protocol_snapshot["path"], "sha256": protocol_snapshot["sha256"],
                     "hash_source": "file_bytes", "manifest_schema": PROTOCOL_MANIFEST_SCHEMA_VERSION},
        "data": {"path": str(Path(context.data_path).resolve()), "sha256": context.data_sha256,
                 "hash_source": "file_bytes"},
        "track_map": {"path": _relative_to_run(outdir, Path(track_map["path"])),
                      "sha256": track_map["sha256"]},
        "grid": grid,
    }


def _write_candidate_status(outdir: Path, candidate_id: str, payload: Mapping[str, Any]) -> None:
    _atomic_json(outdir / "run_status" / "candidates" / (candidate_id + ".json"), dict(payload))


def _write_global_status(outdir: Path, status: str, candidates: Mapping[str, str], **extra: Any) -> None:
    payload = {"status": status, "updated_at_utc": _utc_now(), "candidates": dict(candidates)}
    payload.update(extra)
    _atomic_json(outdir / "run_status" / "run_status.json", payload)


def _append_attempt_event(outdir: Path, event: str, attempt: Mapping[str, Any]) -> None:
    path = outdir / "run_status" / "attempts.jsonl"
    record = dict(_json_ready(attempt))
    record["journal_event"] = event
    record["recorded_at_utc"] = _utc_now()
    payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _attempt_journal_event(attempt: Mapping[str, Any]) -> str:
    """为每条最终 attempt 记录选择一个稳定的 journal event。"""
    if attempt.get("status") == "failed":
        return "stage_failed"
    if attempt.get("is_terminal"):
        return "candidate_terminal"
    return "stage_completed"


def _stage_log(outdir: Path, candidate_id: str, stage: str, message: str) -> None:
    line = "[%s] %s %s: %s" % (dt.datetime.now().strftime("%H:%M:%S"), candidate_id, stage, message)
    print(line, flush=True)
    with (outdir / "logs" / (candidate_id + "-" + stage + ".log")).open("at", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def _validate_data_layer(data: contact_model.AggregatedContacts, context: RunContext,
                         stage: StageSpec) -> dict[str, Any]:
    if int(data.bin_size) != stage.bin_size:
        raise ReconstructionError("%s aggregate bin size disagrees with the frozen schedule" % stage.label)
    if tuple(data.chromosome_names) != tuple(name for name, _length in context.headers):
        raise ReconstructionError("%s aggregate chromosome header order changed" % stage.label)
    if tuple(int(value) for value in data.chromosome_lengths) != tuple(length for _name, length in context.headers):
        raise ReconstructionError("%s aggregate chromosome lengths changed" % stage.label)
    data.assert_consistent()
    audit = data.budget()
    if not (audit["raw_conserved"] and audit["aggregate_conserved"] and audit["endpoint_conserved"]):
        raise ReconstructionError("%s full-grid count budget is not conserved" % stage.label)
    if int(audit["raw_records"]) != int(context.cohort["raw_contacts"]):
        raise ReconstructionError("%s does not retain the full raw contact budget" % stage.label)
    if int(audit["raw_same_bin"]) + int(audit["raw_cis_offdiag"]) != int(context.cohort["intra_contacts"]):
        raise ReconstructionError("%s cis/diagonal budget conflict" % stage.label)
    if int(audit["raw_inter"]) != int(context.cohort["inter_contacts"]):
        raise ReconstructionError("%s inter-chromosome budget conflict" % stage.label)
    expected_pairs = data.n_loci * (data.n_loci - 1) // 2
    if int(audit["n_eligible_pairs"]) != expected_pairs or int(audit["n_diag_bins"]) != data.n_loci:
        raise ReconstructionError("%s eligible-pair or diagonal layer is incomplete" % stage.label)
    if context.strict_p9016 and (data.count_mode != "raw_integer" or len(data.chromosome_names) != 20):
        raise ReconstructionError("strict P9016 layer must use raw integer counts across all 20 chromosomes")
    return _json_ready(audit)


def _expected_positions(data: contact_model.AggregatedContacts) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(data.locus_bin, dtype=np.int64) * int(data.bin_size)
    chromosome_index = np.asarray(data.locus_chromosome, dtype=np.int64)
    return positions, chromosome_index


def _validated_initialization(state: Mapping[str, Any], data: contact_model.AggregatedContacts) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        coordinates = np.asarray(state["coords"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionError("initialization did not return finite copy-first coordinates") from exc
    expected_shape = (2, data.n_loci, 3)
    if coordinates.shape != expected_shape or not np.isfinite(coordinates).all():
        raise ReconstructionError("initialization coordinates have shape %s, expected %s" % (coordinates.shape, expected_shape))
    positions, chromosome_index = _expected_positions(data)
    supplied_positions = np.asarray(state.get("positions", positions), dtype=np.int64)
    supplied_chromosome = np.asarray(state.get("chromosome_index", chromosome_index), dtype=np.int64)
    if not np.array_equal(supplied_positions, positions) or not np.array_equal(supplied_chromosome, chromosome_index):
        raise ReconstructionError("initialization full-grid positions do not match the aggregate header grid")
    if tuple(state.get("names", data.chromosome_names)) != tuple(data.chromosome_names):
        raise ReconstructionError("initialization chromosome header order changed")
    if tuple(int(value) for value in state.get("header_lengths", data.chromosome_lengths)) != tuple(int(value) for value in data.chromosome_lengths):
        raise ReconstructionError("initialization chromosome lengths changed")
    if int(state.get("bin_size", data.bin_size)) != int(data.bin_size):
        raise ReconstructionError("initialization bin size changed")
    coordinates, clipping = _strict_interior(coordinates)
    metadata = dict(state.get("metadata", {}))
    metadata["serialization_clip"] = clipping
    return coordinates, positions, chromosome_index, _json_ready(metadata)


def _strict_interior(coordinates: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.asarray(coordinates, dtype=np.float64).copy()
    radii = np.linalg.norm(result, axis=2)
    if not np.isfinite(radii).all():
        raise ReconstructionError("coordinates are non-finite")
    clipped = radii >= STRICT_INTERIOR_LIMIT
    if clipped.any():
        result[clipped] *= (STRICT_INTERIOR_LIMIT / radii[clipped])[:, None]
    contact_model.assert_inside_unit_ball(result)
    return result, {
        "limit": STRICT_INTERIOR_LIMIT,
        "clipped_coordinates": int(clipped.sum()),
        "max_radius": float(np.linalg.norm(result, axis=2).max()),
    }


def _write_coordinates_exclusive(path: Path, data: contact_model.AggregatedContacts,
                                 coordinates: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    coordinates, clipping = _strict_interior(coordinates)
    with path.open("x", encoding="utf-8") as handle:
        for spec in data.track_specs:
            chromosome_slice = data.chromosome_slice(spec.chromosome_index)
            for global_index in range(chromosome_slice.start, chromosome_slice.stop):
                position = int(data.locus_bin[global_index] * data.bin_size)
                xyz = coordinates[spec.copy_index, global_index]
                handle.write("%s\t%d\t%.17g\t%.17g\t%.17g\n" %
                             (spec.name, position, xyz[0], xyz[1], xyz[2]))
        handle.flush()
        os.fsync(handle.fileno())
    return coordinates, {
        "path": str(path),
        "sha256": sha256_file(path),
        "n_tracks": len(data.track_specs),
        "n_beads": int(2 * data.n_loci),
        "full_grid": True,
        "max_radius": clipping["max_radius"],
        "serialization_clip": clipping,
    }


def _checkpoint_write(path: Path, checkpoint: Any, history: list[Mapping[str, Any]],
                      positions: np.ndarray, chromosome_index: np.ndarray) -> None:
    payload = json.dumps(_json_ready(history), sort_keys=True, allow_nan=False)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            coordinates=np.asarray(checkpoint.coordinates, dtype=np.float64),
            theta=np.asarray(checkpoint.theta, dtype=np.float64),
            y=np.asarray(checkpoint.y, dtype=np.float64),
            positions=np.asarray(positions, dtype=np.int64),
            chromosome_index=np.asarray(chromosome_index, dtype=np.int64),
            fullhistory_json=np.asarray(payload),
        )
        handle.flush()
        os.fsync(handle.fileno())


def _result_components(result: Any) -> dict[str, Any]:
    components = getattr(result, "components", None)
    if not isinstance(components, Mapping):
        raise ReconstructionError("optimizer result lacks auditable objective components")
    if "total" not in components or "p" not in components:
        raise ReconstructionError("optimizer result lacks total/p components")
    return _json_ready(dict(components))


def _solver_status(result: Any, stage: StageSpec) -> tuple[str, bool, str]:
    """分类 SciPy 成功状态，同时保留 iteration/function limit 证据。"""
    nit = int(getattr(result, "nit", 0) or 0)
    actual_nfev = getattr(result, "actual_nfev", None)
    if actual_nfev is None:
        actual_nfev = getattr(result, "nfev", 0)
    actual_nfev = int(actual_nfev or 0)
    message = str(getattr(result, "message", ""))
    message_upper = message.upper()
    iteration_limit = (
        nit >= stage.maxiter
        or "TOTAL NO. OF ITERATIONS REACHED LIMIT" in message_upper
    )
    function_limit = (
        actual_nfev >= stage.maxfun
        or ("TOTAL NO. OF F" in message_upper and "EVALUATIONS EXCEEDS LIMIT" in message_upper)
    )
    budget_exhausted = iteration_limit or function_limit
    # 即使落在数值边界，真正的 SciPy success 仍保持为 converged。
    if bool(getattr(result, "success", False)):
        return "converged", False, "solver_reported_success"
    if budget_exhausted:
        return "not_converged", True, "budget_exhausted"
    return "not_converged", False, "solver_reported_nonconvergence"


def _fit_layer(outdir: Path, context: RunContext, candidate: CandidateSpec, stage: StageSpec,
               data: contact_model.AggregatedContacts, initial_coordinates: np.ndarray,
               positions: np.ndarray, chromosome_index: np.ndarray, q_init: float | None,
               hooks: RunnerHooks) -> tuple[Any, dict[str, Any], np.ndarray]:
    weights = contact_model.ObjectiveWeights(**{key: float(value) for key, value in context.weights.items()})
    objective = contact_model.JointObjective(data, weights=weights)
    initial_y = contact_model.sphere_inverse(initial_coordinates)
    theta0 = objective.pack(initial_y, p=0.75)
    if q_init is not None:
        theta0[-1] = float(q_init)
    initial_total, _unused_gradient, initial_components = objective.evaluate(theta0, need_gradient=False)
    initial_history = [{
        "iteration": 0,
        "nfev": 0,
        "elapsed_seconds": 0.0,
        "fun": float(initial_total),
        "p": float(initial_components["p"]),
        "components": _json_ready(dict(initial_components)),
    }]
    live_history: list[dict[str, Any]] = list(initial_history)
    checkpoint_paths: list[str] = []

    def callback(entry: Mapping[str, Any]) -> None:
        item = _json_ready(dict(entry))
        live_history.append(item)
        _stage_log(outdir, candidate.candidate_id, stage.label,
                   "accepted iteration=%d nfev=%d total=%.12g p=%.9g" %
                   (int(item["iteration"]), int(item["nfev"]), float(item["fun"]), float(item["p"])))

    def checkpoint_hook(checkpoint: Any) -> None:
        entry = {
            "iteration": int(checkpoint.iteration),
            "nfev": int(checkpoint.nfev),
            "elapsed_seconds": float(checkpoint.elapsed_seconds),
            "fun": float(checkpoint.fun),
            "p": float(checkpoint.p),
            "components": _json_ready(dict(checkpoint.components)),
        }
        history = list(live_history)
        if not history or int(history[-1]["iteration"]) != entry["iteration"]:
            history.append(entry)
        checkpoint_dir = outdir / "checkpoints" / candidate.candidate_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / ("%s-accepted-%04d.npz" % (stage.label, checkpoint.iteration))
        _checkpoint_write(checkpoint_path, checkpoint, history, positions, chromosome_index)
        checkpoint_paths.append(_relative_to_run(outdir, checkpoint_path))
        _stage_log(outdir, candidate.candidate_id, stage.label,
                   "checkpoint accepted iteration=%d" % checkpoint.iteration)

    _stage_log(outdir, candidate.candidate_id, stage.label,
               "start maxiter=%d maxfun=%d maxls=20 q_init=%s" %
               (stage.maxiter, stage.maxfun, "none" if q_init is None else "carried"))
    started = time.perf_counter()
    result = hooks.fit(
        objective,
        initial_y,
        p_init=0.75,
        q_init=q_init,
        maxiter=stage.maxiter,
        maxfun=stage.maxfun,
        maxls=20,
        ftol=1e-10,
        gtol=1e-6,
        callback=callback,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_hook=checkpoint_hook,
    )
    elapsed = time.perf_counter() - started
    final_coordinates = np.asarray(getattr(result, "coordinates"), dtype=np.float64)
    if final_coordinates.shape != initial_coordinates.shape:
        raise ReconstructionError("optimizer returned coordinates with an invalid shape")
    final_coordinates, serialization_clip = _strict_interior(final_coordinates)
    result_theta = np.asarray(getattr(result, "theta"), dtype=np.float64)
    if result_theta.shape != (objective.n_parameters,) or not np.isfinite(result_theta).all():
        raise ReconstructionError("optimizer returned invalid raw theta")
    final_components = _result_components(result)
    final_total = float(getattr(result, "fun"))
    tolerance = 1e-10 * max(1.0, abs(float(initial_total)))
    if final_total > float(initial_total) + tolerance:
        raise ReconstructionError("same-layer final total increased above its initial total")
    status, budget_exhausted, reason = _solver_status(result, stage)
    history = _json_ready(getattr(result, "history", live_history))
    final_payload = {
        "status": status,
        "termination_reason": reason,
        "budget_exhausted": budget_exhausted,
        "initial_total": float(initial_total),
        "final_total": final_total,
        "initial_components": _json_ready(dict(initial_components)),
        "final_components": final_components,
        "p": float(getattr(result, "p")),
        "q_in": None if q_init is None else float(q_init),
        "q_out": float(result_theta[-1]),
        "solver": {
            "success": bool(getattr(result, "success")),
            "status": int(getattr(result, "status", 0)),
            "message": str(getattr(result, "message", "")),
            "nit": int(getattr(result, "nit", 0)),
            "nfev": int(getattr(result, "nfev", 0)),
            "actual_nfev": int(getattr(result, "actual_nfev", getattr(result, "nfev", 0))),
            "scipy_nfev": int(getattr(result, "scipy_nfev", getattr(result, "nfev", 0))),
            "njev": int(getattr(result, "njev", 0)),
            "elapsed_seconds": float(getattr(result, "elapsed_seconds", elapsed)),
            "maxiter": stage.maxiter,
            "maxfun": stage.maxfun,
            "maxls": 20,
            "ftol": 1e-10,
            "gtol": 1e-6,
        },
        "checkpoint_paths": checkpoint_paths,
        "history": history,
        "serialization_clip": serialization_clip,
    }
    _stage_log(outdir, candidate.candidate_id, stage.label,
               "finished status=%s nfev=%d total=%.12g" %
               (status, final_payload["solver"]["nfev"], final_total))
    return result, final_payload, final_coordinates


def _stage_path(outdir: Path, candidate_id: str, stage: str) -> Path:
    return outdir / "stages" / candidate_id / (stage + ".json")


def _write_stage(outdir: Path, candidate_id: str, stage: str, payload: Mapping[str, Any]) -> str:
    path = _stage_path(outdir, candidate_id, stage)
    _write_new_json(path, dict(payload))
    return _relative_to_run(outdir, path)


def _candidate_failure(candidate: CandidateSpec, model_contract: Mapping[str, Any],
                       attempts: list[dict[str, Any]], layers: list[dict[str, Any]],
                       reason: str) -> dict[str, Any]:
    if not attempts or not attempts[-1]["is_terminal"]:
        raise ReconstructionError("failed candidate must retain a terminal failed attempt")
    return {
        "id": candidate.candidate_id,
        "initialization": candidate.initialization_candidate,
        "base_seed": candidate.base_seed,
        "model_signature": model_contract["model_signature"],
        "prior_config_sha256": model_contract["prior_config_sha256"],
        "parameter_dimension": model_contract["parameter_dimension"],
        "coordinates": None,
        "count_nll_per_record": None,
        "failure_reason": reason,
        "attempts": attempts,
        "layers": layers,
        "optimization": {"terminal_status": "failed", "terminal_attempt_id": attempts[-1]["attempt_id"]},
    }


def _run_candidate(outdir: Path, context: RunContext, candidate: CandidateSpec,
                   config: Mapping[str, Any], hooks: RunnerHooks) -> dict[str, Any]:
    model_contract = config["model_contract"]
    names = tuple(name for name, _length in context.headers)
    lengths = tuple(int(length) for _name, length in context.headers)
    attempts: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    previous_coordinates: np.ndarray | None = None
    previous_positions: np.ndarray | None = None
    previous_chromosome: np.ndarray | None = None
    carried_q: float | None = None
    final_coordinates_path: str | None = None
    final_coordinates_sha: str | None = None
    final_q: float | None = None
    _write_candidate_status(outdir, candidate.candidate_id, {
        "status": "running", "started_at_utc": _utc_now(), "current_stage": None,
    })
    for stage_index, stage in enumerate(context.stages):
        attempt_id = "%s-%s" % (candidate.candidate_id, stage.label)
        stage_payload: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "attempt_id": attempt_id,
            "stage": stage.label,
            "bin_size_bp": stage.bin_size,
            "started_at_utc": _utc_now(),
        }
        _write_candidate_status(outdir, candidate.candidate_id, {
            "status": "running", "updated_at_utc": _utc_now(), "current_stage": stage.label,
        })
        try:
            data = hooks.load_data(stage.bin_size)
            budget = _validate_data_layer(data, context, stage)
            if previous_coordinates is None:
                initialized = hooks.initialize(candidate.initialization_candidate, names, lengths, stage.bin_size)
            else:
                initialized = hooks.warm_start(
                    previous_coordinates, previous_positions, previous_chromosome,
                    names, lengths, stage.bin_size, candidate.base_seed,
                )
            initial_coordinates, positions, chromosome_index, initialization_metadata = _validated_initialization(initialized, data)
            initial_path = outdir / "coords" / candidate.candidate_id / ("initial-%s.3dg" % stage.label)
            initial_path.parent.mkdir(parents=True, exist_ok=True)
            initial_coordinates, initial_info = _write_coordinates_exclusive(initial_path, data, initial_coordinates)
            result, fit_payload, final_coordinates = _fit_layer(
                outdir, context, candidate, stage, data, initial_coordinates, positions,
                chromosome_index, carried_q, hooks,
            )
            final_path = outdir / "coords" / candidate.candidate_id / (
                "final-%s-%s.3dg" % (stage.label, uuid.uuid4().hex)
            )
            final_coordinates, final_info = _write_coordinates_exclusive(final_path, data, final_coordinates)
            stage_payload.update({
                "completed_at_utc": _utc_now(),
                "data_budget": budget,
                "initialization": initialization_metadata,
                "initial_coordinates": {**initial_info, "path": _relative_to_run(outdir, initial_path),
                                        "reference_gate": "not_used"},
                "final_coordinates": {**final_info, "path": _relative_to_run(outdir, final_path)},
                "fit": fit_payload,
            })
            stage_file = _write_stage(outdir, candidate.candidate_id, stage.label, stage_payload)
            status = fit_payload["status"]
            is_last_stage = stage_index == len(context.stages) - 1
            attempts.append({
                "attempt_id": attempt_id,
                "candidate_id": candidate.candidate_id,
                "stage": stage.label,
                "status": status,
                "is_terminal": False,
                "stage_record": stage_file,
                "budget_exhausted": fit_payload["budget_exhausted"],
            })
            layers.append({"stage": stage.label, "path": stage_file, "status": status,
                           "data_budget": budget})
            previous_coordinates = final_coordinates
            previous_positions = positions
            previous_chromosome = chromosome_index
            carried_q = float(np.asarray(result.theta, dtype=np.float64)[-1])
            if is_last_stage:
                final_coordinates_path = _relative_to_run(outdir, final_path)
                final_coordinates_sha = final_info["sha256"]
                final_q = carried_q
        except Exception as exc:
            reason = "%s: %s" % (type(exc).__name__, exc)
            stage_payload.update({
                "failed_at_utc": _utc_now(),
                "failure_reason": reason,
                "traceback": traceback.format_exc(limit=12),
            })
            try:
                stage_file = _write_stage(outdir, candidate.candidate_id, stage.label, stage_payload)
            except FileExistsError:
                stage_file = _relative_to_run(outdir, _stage_path(outdir, candidate.candidate_id, stage.label))
            attempts.append({
                "attempt_id": attempt_id,
                "candidate_id": candidate.candidate_id,
                "stage": stage.label,
                "status": "failed",
                "is_terminal": True,
                "stage_record": stage_file,
                "failure_reason": reason,
            })
            layers.append({"stage": stage.label, "path": stage_file, "status": "failed"})
            _write_candidate_status(outdir, candidate.candidate_id, {
                "status": "failed", "updated_at_utc": _utc_now(), "current_stage": stage.label,
                "failure_reason": reason,
            })
            return _candidate_failure(candidate, model_contract, attempts, layers, reason)
    if final_coordinates_path is None or final_coordinates_sha is None or final_q is None:
        reason = "final 1 Mb coordinate output was not produced"
        attempts[-1]["status"] = "failed"
        attempts[-1]["is_terminal"] = True
        attempts[-1]["failure_reason"] = reason
        return _candidate_failure(candidate, model_contract, attempts, layers, reason)
    _write_candidate_status(outdir, candidate.candidate_id, {
        "status": "awaiting_final_count_rescore", "updated_at_utc": _utc_now(), "current_stage": "1m",
    })
    return {
        "id": candidate.candidate_id,
        "initialization": candidate.initialization_candidate,
        "base_seed": candidate.base_seed,
        "model_signature": model_contract["model_signature"],
        "prior_config_sha256": model_contract["prior_config_sha256"],
        "parameter_dimension": model_contract["parameter_dimension"],
        "final_coordinates_path": final_coordinates_path,
        "final_coordinates_sha256": final_coordinates_sha,
        "final_q": final_q,
        "attempts": attempts,
        "layers": layers,
    }


def _worker_entry(out_text: str, context: RunContext, candidate: CandidateSpec,
                  config: Mapping[str, Any]) -> dict[str, Any]:
    return _run_candidate(Path(out_text), context, candidate, config, DEFAULT_HOOKS)


def _run_candidates(outdir: Path, context: RunContext, config: Mapping[str, Any],
                    hooks: RunnerHooks, workers: int, use_processes: bool) -> list[dict[str, Any]]:
    if workers == 1:
        return [_run_candidate(outdir, context, candidate, config, hooks) for candidate in context.candidates]
    if not use_processes:
        raise ReconstructionError("parallel workers require the default production collaborators")
    results: dict[str, dict[str, Any]] = {}
    process_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=process_context) as pool:
        futures = {
            pool.submit(_worker_entry, str(outdir), context, candidate, config): candidate
            for candidate in context.candidates
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                results[candidate.candidate_id] = future.result()
            except Exception as exc:
                reason = "worker_failure: %s: %s" % (type(exc).__name__, exc)
                attempt_id = "%s-worker" % candidate.candidate_id
                results[candidate.candidate_id] = _candidate_failure(
                    candidate, config["model_contract"], [{
                        "attempt_id": attempt_id, "candidate_id": candidate.candidate_id,
                        "stage": "worker", "status": "failed", "is_terminal": True,
                        "failure_reason": reason,
                    }], [], reason,
                )
    return [results[candidate.candidate_id] for candidate in context.candidates]


def _coordinates_from_structures(data: contact_model.AggregatedContacts,
                                 structures: Mapping[str, Mapping[int, np.ndarray]]) -> np.ndarray:
    coordinates = np.empty((2, data.n_loci, 3), dtype=np.float64)
    for spec in data.track_specs:
        chromosome_slice = data.chromosome_slice(spec.chromosome_index)
        for global_index in range(chromosome_slice.start, chromosome_slice.stop):
            position = int(data.locus_bin[global_index] * data.bin_size)
            coordinates[spec.copy_index, global_index] = structures[spec.name][position]
    contact_model.assert_inside_unit_ball(coordinates)
    return coordinates


def _rescore_final_candidate(outdir: Path, context: RunContext, config: Mapping[str, Any],
                             candidate: dict[str, Any], hooks: RunnerHooks) -> dict[str, Any]:
    final_stage = context.stages[-1]
    data = hooks.load_data(final_stage.bin_size)
    budget = _validate_data_layer(data, context, final_stage)
    grid = _make_grid(context.headers)
    coordinate_path = _path_from_run(outdir, candidate["final_coordinates_path"])
    structures = reconstruction_report.read_full_grid_coordinates(
        coordinate_path, grid, candidate["final_coordinates_sha256"]
    )
    coordinates = _coordinates_from_structures(data, structures)
    y = contact_model.sphere_inverse(coordinates)
    weights = contact_model.ObjectiveWeights(**{key: float(value) for key, value in context.weights.items()})
    objective = contact_model.JointObjective(data, weights=weights)
    theta = objective.pack(y, p=0.75)
    theta[-1] = float(candidate["final_q"])
    total, _gradient, components = objective.evaluate(theta, need_gradient=False)
    count_model = {field: float(components[field]) for field in reconstruction_report.COUNT_MODEL_FIELDS}
    extra_diagnostics = {key: _json_ready(value) for key, value in components.items()
                         if key not in reconstruction_report.COUNT_MODEL_FIELDS}
    if not np.isfinite(count_model["count_nll_normalized"]):
        raise ReconstructionError("final readback count score is non-finite")
    candidate.update({
        "coordinates": {
            "path": candidate["final_coordinates_path"],
            "sha256": candidate["final_coordinates_sha256"],
        },
        "count_nll_per_record": count_model["count_nll_normalized"],
        "count_model": count_model,
        "selection_rescore": {
            "coordinate_readback": True,
            "q_fixed_from_final_fit": float(candidate["final_q"]),
            "total": float(total),
            "data_budget": budget,
            "diagnostics": extra_diagnostics,
        },
    })
    candidate["attempts"][-1]["is_terminal"] = True
    candidate["optimization"] = {
        "terminal_status": candidate["attempts"][-1]["status"],
        "terminal_attempt_id": candidate["attempts"][-1]["attempt_id"],
    }
    return candidate


def _mark_rescore_failure(candidate: dict[str, Any], reason: str) -> dict[str, Any]:
    if candidate["attempts"]:
        candidate["attempts"][-1]["is_terminal"] = False
    terminal = {
        "attempt_id": "%s-final_count_rescore" % candidate["id"],
        "candidate_id": candidate["id"],
        "stage": "final_count_rescore",
        "status": "failed",
        "is_terminal": True,
        "failure_reason": reason,
    }
    candidate["attempts"].append(terminal)
    candidate.pop("coordinates", None)
    candidate.pop("count_nll_per_record", None)
    candidate.pop("count_model", None)
    candidate["coordinates"] = None
    candidate["count_nll_per_record"] = None
    candidate["failure_reason"] = reason
    candidate["optimization"] = {"terminal_status": "failed", "terminal_attempt_id": terminal["attempt_id"]}
    return candidate


def select_candidate(candidates: list[Mapping[str, Any]], preregistered_order: tuple[str, ...] | list[str]) -> str:
    """只按最终 count NLL 选择，并使用冻结的逐 record tie 规则。"""
    by_id = {candidate["id"]: candidate for candidate in candidates}
    usable = [by_id[candidate_id] for candidate_id in preregistered_order
              if by_id[candidate_id].get("optimization", {}).get("terminal_status") != "failed"
              and by_id[candidate_id].get("count_nll_per_record") is not None]
    if not usable:
        raise ReconstructionError("all pre-registered candidates failed before selection")
    best = min(float(candidate["count_nll_per_record"]) for candidate in usable)
    for candidate in usable:
        if float(candidate["count_nll_per_record"]) <= best + SELECTION_TIE_TOLERANCE:
            return str(candidate["id"])
    raise AssertionError("unreachable frozen tie selection")


def _copy_selected_coordinates(outdir: Path, selected: Mapping[str, Any]) -> dict[str, Any]:
    source = _path_from_run(outdir, selected["coordinates"]["path"])
    target = outdir / "selected.3dg"
    _copy_new(source, target)
    digest = sha256_file(target)
    if digest != selected["coordinates"]["sha256"]:
        raise ReconstructionError("selected coordinate copy changed after hashing")
    return {
        "path": _relative_to_run(outdir, target),
        "sha256": digest,
        "source_candidate_id": selected["id"],
        "source_coordinate_path": selected["coordinates"]["path"],
        "source_coordinate_sha256": selected["coordinates"]["sha256"],
    }


def _selection_document(outdir: Path, context: RunContext, config: Mapping[str, Any],
                        static_files: Mapping[str, Any], candidates: list[dict[str, Any]],
                        attempts: list[dict[str, Any]], selected_id: str,
                        selected_coordinate: Mapping[str, Any]) -> dict[str, Any]:
    model_contract = config["model_contract"]
    candidate_records = []
    for raw in candidates:
        record = {
            "id": raw["id"],
            "initialization": raw["initialization"],
            "base_seed": raw["base_seed"],
            "model_signature": raw["model_signature"],
            "prior_config_sha256": raw["prior_config_sha256"],
            "parameter_dimension": raw["parameter_dimension"],
            "attempt_ids": [attempt["attempt_id"] for attempt in raw["attempts"]],
            "optimization": raw["optimization"],
            "layers": raw.get("layers", []),
        }
        if raw["optimization"]["terminal_status"] == "failed":
            record.update({
                "coordinates": None,
                "count_nll_per_record": None,
                "failure_reason": raw["failure_reason"],
            })
        else:
            record.update({
                "coordinates": raw["coordinates"],
                "count_nll_per_record": raw["count_nll_per_record"],
                # 该 map 必须恰好包含 report.COUNT_MODEL_FIELDS。
                "count_model": raw["count_model"],
                "selection_rescore": raw["selection_rescore"],
            })
        candidate_records.append(record)
    grid = static_files["grid"]
    return {
        "schema_version": reconstruction_report.SELECTION_SCHEMA_VERSION,
        "status": "training_complete",
        "cohort": dict(context.cohort),
        "coordinate_grid": {
            "bin_size_bp": FINAL_BIN_SIZE,
            "origin_bp": 0,
            "full_grid": True,
            "nuclear_radius": 1.0,
            "coordinate_units": "dimensionless_R1",
            "chromosomes": [{"name": chromosome.name, "length_bp": chromosome.length_bp}
                            for chromosome in grid.chromosomes],
            "expected_loci": grid.n_loci,
            "expected_physical_beads": grid.n_physical_beads,
        },
        "track_map": dict(static_files["track_map"]),
        "frozen_provenance": {
            "protocol": dict(static_files["protocol"]),
            "config": dict(static_files["config"]),
            "code": dict(static_files["code"]),
            "data": dict(static_files["data"]),
        },
        "objective_contract": {
            "criterion": "count_nll_per_record",
            "direction": "minimize",
            "conditional_count_objective": True,
            "includes_candidate_invariant_diagonal_layer": True,
            "includes_priors": False,
            "uses_all_contacts": True,
            "selection_is_label_free": True,
            "reference_used_for_selection": False,
            "phase_used_for_selection": False,
            "bin_size_bp": FINAL_BIN_SIZE,
            "tie_tolerance_per_record": SELECTION_TIE_TOLERANCE,
            "tie_break": "preregistered_order",
            "count_model_fields": list(reconstruction_report.COUNT_MODEL_FIELDS),
            "prior_component_names": list(reconstruction_report.PRIOR_REPORT_FIELDS),
            "model_contract": dict(model_contract),
        },
        "preregistered_candidate_ids": [candidate.candidate_id for candidate in context.candidates],
        "candidates": candidate_records,
        "attempts": attempts,
        "baseline_assets": _json_ready(context.baseline_assets),
        "selection": {
            "selected_id": selected_id,
            "rule": "minimize_count_nll_per_record",
            "reference_used": False,
            "phase_used": False,
            "all_attempts_accounted_for": True,
            "tie_break": "preregistered_order",
            "selected_coordinate": dict(selected_coordinate),
        },
    }


def verify_completed_training(outdir: str | os.PathLike[str], *, strict_p9016: bool = True) -> Any:
    """只运行纯 provenance/inventory 守门；绝不调用 evaluator loaders。"""
    root = Path(outdir).resolve()
    selection_path = root / "selection.json"
    try:
        with selection_path.open() as handle:
            selection_document = json.load(handle)
        protocol_reference = selection_document["frozen_provenance"]["protocol"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ReconstructionError("completed training selection lacks a protocol manifest reference") from exc
    protocol_manifest = _verify_snapshot_manifest(
        root, protocol_reference, PROTOCOL_MANIFEST_SCHEMA_VERSION)
    protocol_members = [row for row in protocol_manifest["members"]
                        if row["source_path"] == "docs/RECONSTRUCTION_V1_PROTOCOL.md"]
    if len(protocol_members) != 1 or protocol_members[0]["sha256"] != FROZEN_PROTOCOL_SHA256:
        raise ReconstructionError("protocol snapshot does not contain the frozen protocol bytes")
    return reconstruction_report.verify_training_complete(selection_path, strict_p9016=strict_p9016)


def run_training(outdir: str | os.PathLike[str], context: RunContext, *, hooks: RunnerHooks | None = None,
                 workers: int = 1, threads: int = 1) -> dict[str, Any]:
    """执行冻结 context，供生产和 synthetic runner tests 使用。

    真实入口是 :func:`run_production`；注入 hooks 仅用于让 tests 在不优化 P9016 的情况下检查 journaling 和 provenance。
    """
    if workers not in (1, 2):
        raise ReconstructionError("formal runner permits one or two candidate workers")
    requested = Path(os.fspath(outdir))
    if os.path.lexists(requested):
        raise FileExistsError("refusing to reuse existing formal output directory: %s" % requested)
    root = requested.resolve()
    if os.path.lexists(root):
        raise FileExistsError("refusing to reuse existing formal output directory: %s" % root)
    _validate_context(context)
    _configure_threads(threads)
    _create_fresh_layout(root)
    active_hooks = DEFAULT_HOOKS if hooks is None else hooks
    config = _build_config(context, workers=workers, threads=threads)
    static_files = _write_static_run_files(root, context, config)
    _write_global_status(root, "training", {candidate.candidate_id: "pending" for candidate in context.candidates})
    results = _run_candidates(root, context, config, active_hooks, workers, hooks is None)

    final_status: dict[str, str] = {}
    for result in results:
        if result.get("optimization", {}).get("terminal_status") == "failed":
            final_status[result["id"]] = "failed"
            continue
        try:
            _rescore_final_candidate(root, context, config, result, active_hooks)
            final_status[result["id"]] = result["optimization"]["terminal_status"]
            _write_candidate_status(root, result["id"], {
                "status": result["optimization"]["terminal_status"], "updated_at_utc": _utc_now(),
                "current_stage": "1m", "final_count_rescore": "complete",
            })
        except Exception as exc:
            reason = "final_count_rescore: %s: %s" % (type(exc).__name__, exc)
            _mark_rescore_failure(result, reason)
            final_status[result["id"]] = "failed"
            _write_candidate_status(root, result["id"], {
                "status": "failed", "updated_at_utc": _utc_now(), "current_stage": "final_count_rescore",
                "failure_reason": reason,
            })

    attempts = [attempt for result in results for attempt in result["attempts"]]
    # candidate-level rescore 确定每个 candidate 的唯一终态标志后，每次 attempt
    # 只持久化一行最终、确定性的 journal 记录。
    for attempt in attempts:
        _append_attempt_event(root, _attempt_journal_event(attempt), attempt)
    usable = [result for result in results if result.get("optimization", {}).get("terminal_status") != "failed"
              and result.get("count_nll_per_record") is not None]
    if not usable:
        summary = {"status": "training_failed", "candidates": results, "attempts": attempts}
        _write_new_json(root / "training_summary.json", summary)
        _write_global_status(root, "training_failed", final_status,
                             failure_reason="all pre-registered candidates failed")
        raise ReconstructionError("all pre-registered candidates failed; no selection was written")

    selected_id = select_candidate(results, tuple(candidate.candidate_id for candidate in context.candidates))
    selected = next(result for result in results if result["id"] == selected_id)
    selected_coordinate = _copy_selected_coordinates(root, selected)
    selection = _selection_document(root, context, config, static_files, results, attempts,
                                    selected_id, selected_coordinate)
    selection_path = root / "selection.json"
    _write_new_json(selection_path, selection)
    # 该守门只读取 snapshots、data hashes、track map、selection metadata 和最终
    # candidate 坐标。它不会接收 labels/reference/oracle loader 函数。
    verified = verify_completed_training(root, strict_p9016=context.strict_p9016)
    _write_global_status(root, "training_complete", final_status,
                         selection_path=_relative_to_run(root, selection_path), selected_id=selected_id)
    return {
        "outdir": str(root),
        "selection_path": str(selection_path),
        "selected_id": selected_id,
        "selected_coordinates": selected_coordinate,
        "verified_selection": verified,
        "candidates": results,
    }


def run_production(outdir: str | os.PathLike[str], *, workers: int = 1, threads: int = 1) -> dict[str, Any]:
    """在父流程释放后运行授权的 P9016 V1 training path。"""
    return run_training(outdir, production_context(), hooks=None, workers=workers, threads=threads)
