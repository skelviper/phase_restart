"""针对修正版 post-020 设计的受控 C0 复现 runner。

本模块只用于 training 侧。它读取冻结的七列 SNP-free 输入和两个 blind 014 坐标源，本阶段只运行 C0；绝不打开带 phase 的 payload、参考坐标、评价输出或 oracle 坐标数据。其他四个变体保留在冻结的 30-attempt 计划中，但在父侧释放前有意阻止。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from . import allele_models, contact_model, genome, paired_run, reconstruction_init, reconstruction_report
from . import reconstruct
from .gate import sha256_file


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve()
OLD_RUN = ROOT / "test_res" / "020-20260913_071841-v1-p9016-joint"
RUN_NUMBER = 33
RUN_SLUG = "c0-controlled-reproduction"
RUN_SCHEMA = "c0-controlled-reproduction-v1"
PROTOCOL_SCHEMA = "c0-controlled-protocol-v1"
PREFLIGHT_SCHEMA = "c0-controlled-preflight-v1"
GATE_SCHEMA = "c0-reference-free-numeric-gate-v1"
TERMINAL_SCHEMA = "c0-terminal-evidence-v1"
REPORT_SCHEMA = "c0-training-report-v1"

CANDIDATES = (
    reconstruct.CandidateSpec("consensus_joint", "consensus", 1103),
    reconstruct.CandidateSpec("random_joint", "random", 2207),
)
STAGES = (
    reconstruct.StageSpec("5m", 5_000_000, 300),
    reconstruct.StageSpec("2m", 2_000_000, 200),
    reconstruct.StageSpec("1m", 1_000_000, 240),
)
VARIANTS = tuple(allele_models.MODEL_IDS)
RELEASED_VARIANT = "C0"
BLOCKED_VARIANTS = tuple(model for model in VARIANTS if model != RELEASED_VARIANT)
SELECTION_TIE_TOLERANCE = 1e-9
STRICT_ATOL = 1e-12
STRICT_RTOL = 1e-10
FD_ATOL = 1e-5
FD_RTOL = 5e-5
EXPECTED_INPUT_SHA256 = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"
EXPECTED_RAW_SHA256 = "071a6cc76bfad543ea1ace6ee1ce3022b30ac1b1e50a9f0c3a3a1967b9649505"
EXPECTED_GATE_SHA256 = "cef694b899039aed36cc26b169f1ba2b9df170b40249742ddd2f28b38cd35f19"
EXPECTED_SOURCE_SHA256 = {
    "consensus": "e76655732deb6b8386b1b77bc76ff45d7dba1384f6931337fee80d8f4aaa8e02",
    "random": "9a48d73e1401e18349d11758e679da4c76da0904dbc467979079cb54bcd567d7",
}
EXPECTED_LAYER_BUDGETS = {
    "5m": {
        "bin_size_bp": 5_000_000,
        "n_loci": 538,
        "n_eligible_pairs": 144_453,
        "n_zero_eligible_pairs": 22_184,
        "raw_same_bin": 607_552,
        "raw_cis_offdiag": 527_902,
        "raw_inter": 568_434,
    },
    "2m": {
        "bin_size_bp": 2_000_000,
        "n_loci": 1_329,
        "n_eligible_pairs": 882_456,
        "n_zero_eligible_pairs": 534_496,
        "raw_same_bin": 516_046,
        "raw_cis_offdiag": 619_408,
        "raw_inter": 568_434,
    },
    "1m": {
        "bin_size_bp": 1_000_000,
        "n_loci": 2_645,
        "n_eligible_pairs": 3_496_690,
        "n_zero_eligible_pairs": 3_009_436,
        "raw_same_bin": 438_774,
        "raw_cis_offdiag": 696_680,
        "raw_inter": 568_434,
    },
}
TRAINING_SOURCES = tuple(dict.fromkeys(
    tuple(reconstruct.TRAINING_CODE_SOURCES)
    + ("pr/allele_models.py", "pr/paired_run.py", "pr/c0_controlled_runner.py")
))


class ControlledRunnerError(RuntimeError):
    """无法履行修正版 C0 契约时抛出。"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ControlledRunnerError("nonfinite value cannot enter formal evidence")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json_exclusive(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _write_bytes_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _sha256_array(value: np.ndarray, dtype: str | None = None) -> str:
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(np.dtype(dtype), copy=False)
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _float_diff(left: float, right: float) -> float:
    return abs(float(left) - float(right))


def _within(left: float, right: float, atol: float = STRICT_ATOL,
            rtol: float = STRICT_RTOL) -> bool:
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=rtol))


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right), initial=0.0))


def _set_threads(threads: int) -> None:
    if int(threads) != 1:
        raise ControlledRunnerError("this controlled run is frozen to one BLAS/OpenMP thread")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _fresh_root(outdir: str | os.PathLike[str]) -> Path:
    requested = Path(outdir)
    if requested.is_absolute():
        root = requested.resolve()
    else:
        root = (ROOT / requested).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError("refusing to reuse existing controlled run: %s" % root)
    root.mkdir(parents=True)
    for relative in ("coords", "checkpoints", "logs", "work", "plots", "stages",
                     "run_status/candidates", "provenance/training-code",
                     "provenance/protocol", "provenance/controlled-source"):
        (root / relative).mkdir(parents=True, exist_ok=False)
    return root


def _snapshot_controlled_sources(root: Path) -> dict[str, Any]:
    rows = []
    for source_text in TRAINING_SOURCES:
        source = ROOT / source_text
        if not source.is_file():
            raise ControlledRunnerError("missing controlled source: %s" % source)
        destination = root / "provenance" / "controlled-source" / source_text
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_handle, destination.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        rows.append({
            "source_path": source_text,
            "snapshot_path": str(destination.relative_to(root)),
            "sha256": sha256_file(destination),
        })
    manifest = {"schema_version": "controlled-source-snapshot-v1", "files": rows}
    manifest_path = root / "provenance" / "controlled-source-manifest.json"
    manifest_sha = _write_json_exclusive(manifest_path, manifest)
    return {"path": str(manifest_path.relative_to(root)), "sha256": manifest_sha, "files": rows}


def _source_contract(context: reconstruct.RunContext) -> list[dict[str, Any]]:
    result = []
    for candidate in CANDIDATES:
        source = context.source_assets[candidate.initialization_candidate]
        result.append({
            "candidate_id": candidate.candidate_id,
            "initialization": candidate.initialization_candidate,
            "base_seed": candidate.base_seed,
            "stage": "blind",
            "path": str(source["path"]),
            "sha256": str(source["sha256"]),
            "gate_path": str(source["gate_path"]),
            "gate_sha256": str(source["gate_sha256"]),
        })
    return result


def _layer_array_hashes(data: contact_model.AggregatedContacts) -> dict[str, str]:
    return {
        "pair_i": _sha256_array(data.pair_i, "<i4"),
        "pair_j": _sha256_array(data.pair_j, "<i4"),
        "cis_pair": _sha256_array(data.cis_pair, "|b1"),
        "counts": _sha256_array(data.counts, "<i8"),
        "diag_counts": _sha256_array(data.diag_counts, "<i8"),
        "endpoint_counts": _sha256_array(data.endpoint_counts, "<i8"),
        "exposure": _sha256_array(data.exposure, "<f8"),
        "locus_chromosome": _sha256_array(data.locus_chromosome, "<i4"),
        "locus_bin": _sha256_array(data.locus_bin, "<i8"),
        "n_bins": _sha256_array(data.n_bins, "<i8"),
        "offsets": _sha256_array(data.offsets, "<i8"),
    }


def _check_layer_budget(data: contact_model.AggregatedContacts,
                        stage: reconstruct.StageSpec) -> dict[str, Any]:
    data.assert_consistent()
    audit = data.budget()
    expected = EXPECTED_LAYER_BUDGETS[stage.label]
    if int(data.bin_size) != int(expected["bin_size_bp"]):
        raise ControlledRunnerError(
            "%s budget mismatch for bin_size_bp: got %d expected %d" %
            (stage.label, int(data.bin_size), int(expected["bin_size_bp"])))
    for key, value in expected.items():
        if key == "bin_size_bp":
            continue
        actual = int(audit[key])
        if actual != int(value):
            raise ControlledRunnerError(
                "%s budget mismatch for %s: got %d expected %d" %
                (stage.label, key, actual, int(value)))
    required = {
        "raw_records": 1_703_888,
        "raw_same_bin": expected["raw_same_bin"],
        "raw_cis_offdiag": expected["raw_cis_offdiag"],
        "raw_inter": 568_434,
        "aggregate_same_bin": expected["raw_same_bin"],
        "aggregate_cis_offdiag": expected["raw_cis_offdiag"],
        "aggregate_inter": 568_434,
        "endpoint_total": 3_407_776,
        "n_diag_bins": expected["n_loci"],
    }
    for key, value in required.items():
        if int(audit[key]) != int(value):
            raise ControlledRunnerError("%s budget field %s changed" % (stage.label, key))
    if not (audit["raw_conserved"] and audit["aggregate_conserved"] and audit["endpoint_conserved"]):
        raise ControlledRunnerError("%s raw/aggregate/endpoint conservation failed" % stage.label)
    expected_exposure = np.sqrt(data.endpoint_counts.astype(np.float64) + 10.0)
    expected_exposure /= expected_exposure.mean()
    exposure_error = _max_abs(data.exposure, expected_exposure)
    if exposure_error != 0.0:
        raise ControlledRunnerError("%s exposure formula changed (max abs %.17g)" %
                                    (stage.label, exposure_error))
    return {
        "audit": _jsonable(audit),
        "array_sha256": _layer_array_hashes(data),
        "exposure_formula_max_abs": exposure_error,
        "denominator_contract": {
            "eligible_pairs": "all unordered global locus pairs i<j including zero counts",
            "same_bin": "separate per-bin saturated Poisson nuisance",
            "diag_nuisance_parameters": int(data.n_loci),
            "bin_size_bp": int(data.bin_size),
        },
    }


def _state_array(state: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(state["coords"], dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 2 or values.shape[2] != 3:
        raise ControlledRunnerError("initialization state has invalid coordinate shape")
    if not np.isfinite(values).all():
        raise ControlledRunnerError("initialization state is non-finite")
    contact_model.assert_inside_unit_ball(values)
    return values


def _state_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(state.get("metadata", {}))
    return {
        "mode": metadata.get("mode"),
        "candidate": metadata.get("candidate"),
        "seed": metadata.get("seed"),
        "candidate_base_seed": metadata.get("candidate_base_seed"),
        "l0": metadata.get("l0"),
        "perturbation": metadata.get("perturbation"),
        "normalization": metadata.get("normalization"),
        "initial_normalization": metadata.get("initial_normalization"),
        "final_normalization": metadata.get("final_normalization"),
        "source": metadata.get("source"),
    }


def _build_common_starts(context: reconstruct.RunContext,
                         data_by_label: Mapping[str, contact_model.AggregatedContacts]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    names = tuple(name for name, _ in context.headers)
    lengths = tuple(int(length) for _name, length in context.headers)
    states: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        candidate_states = {}
        state = reconstruction_init.initialize_approved_candidate(
            candidate.initialization_candidate, names, lengths, STAGES[0].bin_size)
        candidate_states[STAGES[0].label] = state
        previous = state
        for stage in STAGES[1:]:
            previous = reconstruction_init.warm_start_from_layer(
                previous["coords"], previous["positions"], previous["chromosome_index"],
                names, lengths, stage.bin_size, candidate.base_seed)
            candidate_states[stage.label] = previous
        states[candidate.candidate_id] = candidate_states

    rows: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_rows = {}
        for stage in STAGES:
            data = data_by_label[stage.label]
            state = states[candidate.candidate_id][stage.label]
            coords = _state_array(state)
            expected_positions = np.asarray(data.locus_bin, dtype=np.int64) * stage.bin_size
            expected_chromosome = np.asarray(data.locus_chromosome, dtype=np.int64)
            if not np.array_equal(np.asarray(state["positions"], dtype=np.int64), expected_positions):
                raise ControlledRunnerError("%s/%s positions are not the complete origin-0 grid" %
                                            (candidate.candidate_id, stage.label))
            if not np.array_equal(np.asarray(state["chromosome_index"], dtype=np.int64), expected_chromosome):
                raise ControlledRunnerError("%s/%s chromosome grid changed" %
                                            (candidate.candidate_id, stage.label))
            base_hash = _sha256_array(coords, "<f8")
            variant_rows = {}
            for model_id in VARIANTS:
                objective = allele_models.objective_for_model(data, model_id)
                raw = allele_models.raw_coordinates_from_physical(objective, coords)
                roundtrip = allele_models.physical_coordinates_from_raw(objective, raw)
                error = _max_abs(roundtrip, coords)
                if error > STRICT_ATOL + STRICT_RTOL * max(1.0, float(np.max(np.abs(coords)))):
                    raise ControlledRunnerError(
                        "%s/%s/%s physical start round-trip %.17g exceeds frozen strict tolerance" %
                        (candidate.candidate_id, stage.label, model_id, error))
                allele_models.validate_physical_for_model(model_id, roundtrip)
                theta = objective.pack(raw, p=0.75)
                packed_roundtrip, packed_p = objective.coordinates_and_p(theta)
                packed_error = _max_abs(packed_roundtrip, coords)
                if packed_error > STRICT_ATOL + STRICT_RTOL * max(1.0, float(np.max(np.abs(coords)))):
                    raise ControlledRunnerError(
                        "%s/%s/%s legal pack/unpack changed the shared physical start" %
                        (candidate.candidate_id, stage.label, model_id))
                variant_rows[model_id] = {
                    "input_physical_sha256": base_hash,
                    "raw_sha256": _sha256_array(raw, "<f8"),
                    "roundtrip_max_abs": error,
                    "packed_roundtrip_max_abs": packed_error,
                    "p_roundtrip": float(packed_p),
                    "parameter_dimension": int(objective.n_parameters),
                    "coordinate_parameterization": allele_models.model_spec(model_id).coordinate_parameterization,
                    "physical_domain": allele_models.model_spec(model_id).physical_domain,
                }
            candidate_rows[stage.label] = {
                "input_physical_sha256": base_hash,
                "shape": list(coords.shape),
                "max_radius": float(np.linalg.norm(coords, axis=2).max()),
                "metadata": _state_metadata(state),
                "variants": variant_rows,
            }
        rows[candidate.candidate_id] = candidate_rows
    # 同一 candidate 的 physical start 逐字节传给每个 variant。
    for candidate_id, candidate_rows in rows.items():
        for stage_label, stage_row in candidate_rows.items():
            hashes = {row["input_physical_sha256"] for row in stage_row["variants"].values()}
            if hashes != {stage_row["input_physical_sha256"]}:
                raise ControlledRunnerError("shared physical start hash differs across variants")
    return rows, states


def _fixed_objective_checks(data: contact_model.AggregatedContacts,
                            coords: np.ndarray) -> dict[str, Any]:
    raw = contact_model.sphere_inverse(coords)
    theta = contact_model.JointObjective(data).pack(raw, p=0.75)
    base = contact_model.JointObjective(data)
    c0 = allele_models.objective_for_model(data, "C0")
    total_base, grad_base, comp_base = base.evaluate(theta, need_gradient=True)
    total_c0, grad_c0, comp_c0 = c0.evaluate(theta, need_gradient=True)
    component_diffs = {
        key: _float_diff(comp_base[key], comp_c0[key])
        for key in sorted(set(comp_base) | set(comp_c0))
        if key in comp_base and key in comp_c0 and isinstance(comp_base[key], (int, float))
    }
    gradient_diff = _max_abs(grad_base, grad_c0)
    if not _within(total_base, total_c0) or gradient_diff > STRICT_ATOL + STRICT_RTOL * max(1.0, float(np.linalg.norm(grad_base))):
        raise ControlledRunnerError("C0 fixed-state value/gradient is not equivalent to V1")
    if max(component_diffs.values(), default=0.0) > STRICT_ATOL + STRICT_RTOL:
        raise ControlledRunnerError("C0 fixed-state component equivalence failed")
    return {
        "value_max_abs": _float_diff(total_base, total_c0),
        "gradient_max_abs": gradient_diff,
        "component_max_abs": max(component_diffs.values(), default=0.0),
        "component_diffs": component_diffs,
        "base_total": float(total_base),
        "c0_total": float(total_c0),
        "base_gradient_l2": float(np.linalg.norm(grad_base)),
        "c0_gradient_l2": float(np.linalg.norm(grad_c0)),
    }


def _c1_bend_check(data: contact_model.AggregatedContacts,
                   coords: np.ndarray) -> dict[str, Any]:
    raw = contact_model.sphere_inverse(coords)
    c0 = allele_models.objective_for_model(data, "C0")
    c1 = allele_models.objective_for_model(data, "C1")
    theta = c0.pack(raw, p=0.75)
    total0, grad0, comp0 = c0.evaluate(theta, need_gradient=True)
    total1, grad1, comp1 = c1.evaluate(theta, need_gradient=True)
    same_keys = ("count_nll_normalized", "conditional_nll_raw", "diag_profiled_nll_raw",
                 "count_nll_raw", "p", "bond", "repulsion", "p_prior", "bend")
    component_diffs = {key: _float_diff(comp0[key], comp1[key]) for key in same_keys}
    expected_total_delta = c0.weights.bend * float(comp0["bend"])
    observed_total_delta = float(total0 - total1)
    bend_value, bend_gradient = c0._bend_and_gradient(coords)
    expected_gradient_delta = np.concatenate((
        (c0.weights.bend * contact_model.sphere_pullback(raw, bend_gradient)).ravel(),
        np.asarray([0.0], dtype=np.float64),
    ))
    gradient_residual = _max_abs(np.asarray(grad0) - np.asarray(grad1), expected_gradient_delta)
    if max(component_diffs.values(), default=0.0) > STRICT_ATOL + STRICT_RTOL:
        raise ControlledRunnerError("C1 changed a non-bend component")
    if not _within(observed_total_delta, expected_total_delta):
        raise ControlledRunnerError("C1 total does not remove exactly the bend contribution")
    if gradient_residual > STRICT_ATOL + STRICT_RTOL * max(1.0, float(np.linalg.norm(grad0))):
        raise ControlledRunnerError("C1 gradient does not remove exactly the bend gradient")
    return {
        "nonbend_component_max_abs": max(component_diffs.values(), default=0.0),
        "component_diffs": component_diffs,
        "bend_value": float(bend_value),
        "expected_total_delta": expected_total_delta,
        "observed_total_delta": observed_total_delta,
        "gradient_residual_max_abs": gradient_residual,
    }


def _c3_exposure_check(data: contact_model.AggregatedContacts) -> dict[str, Any]:
    c3_data = allele_models.data_for_model(data, "C3")
    array_fields = ("pair_i", "pair_j", "cis_pair", "counts", "diag_counts",
                    "endpoint_counts", "locus_chromosome", "locus_bin", "n_bins", "offsets")
    changed = {}
    for field in array_fields:
        left = np.asarray(getattr(data, field))
        right = np.asarray(getattr(c3_data, field))
        changed[field] = bool(not np.array_equal(left, right))
        if changed[field]:
            raise ControlledRunnerError("C3 changed data array %s" % field)
    exposure = np.asarray(c3_data.exposure, dtype=np.float64)
    max_deviation = float(np.max(np.abs(exposure - 1.0), initial=0.0))
    if max_deviation != 0.0 or c3_data.exposure_mode != "uniform":
        raise ControlledRunnerError("C3 exposure is not the frozen all-ones view")
    base_budget = data.budget()
    variant_budget = c3_data.budget()
    budget_fields = tuple(key for key in base_budget if key != "exposure_mode")
    budget_diffs = {
        key: _float_diff(variant_budget[key], base_budget[key])
        if isinstance(base_budget[key], (int, float)) else (0.0 if variant_budget[key] == base_budget[key] else float("inf"))
        for key in budget_fields
    }
    if any(value != 0.0 for value in budget_diffs.values()):
        raise ControlledRunnerError("C3 changed the count budget")
    return {
        "unchanged_array_fields": array_fields,
        "changed_array_fields": [field for field, changed_value in changed.items() if changed_value],
        "input_exposure_mode": data.exposure_mode,
        "variant_exposure_mode": c3_data.exposure_mode,
        "uniform_exposure_max_abs_from_one": max_deviation,
        "budget_equal": True,
        "exposure_sha256": _sha256_array(exposure, "<f8"),
    }


def _c2_domain_checks(data: contact_model.AggregatedContacts, root: Path) -> dict[str, Any]:
    directions = np.asarray([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 2.0, 3.0],
        [-2.0, 1.0, 1.0], [1.0, -1.0, 2.0],
    ], dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = np.asarray([0.0, 0.4, 0.9, 0.900001, 1.25, 3.0], dtype=np.float64)
    probes = np.zeros((len(radii), 3), dtype=np.float64)
    probes[1:] = directions[:len(radii) - 1] * radii[1:, None]
    mapped = allele_models.identity_interior_forward(probes)
    inverse = allele_models.identity_interior_inverse(mapped)
    inverse_error = _max_abs(inverse, probes)
    if inverse_error > 2e-12 or not np.all(np.linalg.norm(mapped, axis=1) < 1.0):
        raise ControlledRunnerError("C2-map forward/inverse probe failed")

    # 在真实的 phase-free layer 上使用 C2-map objective，并使用 identity interior 外的 raw point
    # 触发 adapter 的 pullback。
    c2_data = data
    c2 = allele_models.objective_for_model(c2_data, "C2-map")
    base = contact_model.sphere_inverse(np.zeros((2, data.n_loci, 3), dtype=np.float64))
    base[0, 0] = [1.25, 0.0, 0.0]
    theta = c2.pack(base, p=0.75)
    total, gradient, _components = c2.evaluate(theta, need_gradient=True)
    direction = np.zeros_like(theta)
    direction[0] = 1.0
    step = 1e-6
    plus = c2.evaluate(theta + step * direction, need_gradient=False)[0]
    minus = c2.evaluate(theta - step * direction, need_gradient=False)[0]
    map_activity = c2.map_diagnostics()
    numeric = float((plus - minus) / (2.0 * step))
    analytic = float(gradient[0])
    if not np.isclose(analytic, numeric, atol=FD_ATOL, rtol=FD_RTOL):
        raise ControlledRunnerError(
            "C2-map objective gradient finite-difference check failed: %.17g vs %.17g" %
            (analytic, numeric))

    # C2-free 必须在 pack、evaluate 和 model-specific writer 中保留半径 > 1 的有限 physical point；
    # 不允许 sphere inverse/clip。
    free = allele_models.objective_for_model(data, "C2-free")
    outside = np.zeros((2, data.n_loci, 3), dtype=np.float64)
    outside[0, 0] = [1.2, 0.0, 0.0]
    raw = allele_models.raw_coordinates_from_physical(free, outside)
    returned = allele_models.physical_coordinates_from_raw(free, raw)
    free_roundtrip_error = _max_abs(returned, outside)
    if free_roundtrip_error != 0.0:
        raise ControlledRunnerError("C2-free outside-one round-trip changed coordinates")
    free_theta = free.pack(raw, p=0.75)
    free_coordinates, _free_p = free.coordinates_and_p(free_theta)
    free.validate_physical_coordinates(free_coordinates)
    _free_total, _free_gradient, _free_components = free.evaluate(free_theta, need_gradient=False)
    free_map_activity = free.map_diagnostics()
    writer_path = root / "work" / "preflight" / "c2-free-outside1.3dg"
    writer_path.parent.mkdir(parents=True, exist_ok=True)
    writer_info = paired_run.write_coordinates(writer_path, data, "C2-free", outside)
    preserved_line = None
    with writer_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("c01a\t0\t"):
                preserved_line = line.rstrip("\n")
                break
    if preserved_line is None or "1.2" not in preserved_line:
        raise ControlledRunnerError("C2-free writer did not preserve the outside-one point")
    if writer_info["serialization_clip"]["applied"] is not False:
        raise ControlledRunnerError("C2-free writer reports an unexpected clip")
    return {
        "map_probe_inverse_max_abs": inverse_error,
        "map_probe_max_radius": float(np.linalg.norm(mapped, axis=1).max()),
        "map_gradient": {
            "raw_probe_radius": 1.25,
            "finite_difference_step": step,
            "analytic": analytic,
            "central_difference": numeric,
            "abs_difference": abs(analytic - numeric),
            "atol": FD_ATOL,
            "rtol": FD_RTOL,
        },
        "map_activity": {
            "objective_calls": map_activity,
            "accepted_trial_split": "not exposed by current L-BFGS callback; line-search probes are included",
            "used_for_selection_or_budget": False,
        },
        "free_outside_one": {
            "input_radius": 1.2,
            "roundtrip_max_abs": free_roundtrip_error,
            "evaluate_radius": float(np.linalg.norm(free_coordinates[0, 0])),
            "map_activity": free_map_activity,
            "ever_physical_radius_gt1": bool(free_map_activity["max_physical_radius_seen"] > 1.0),
            "writer_path": str(writer_path.relative_to(root)),
            "writer_sha256": writer_info["sha256"],
            "writer_domain": writer_info["physical_domain"],
            "serialization_clip": writer_info["serialization_clip"],
            "preserved_line": preserved_line,
        },
    }


def run_preflight(context: reconstruct.RunContext, root: Path) -> dict[str, Any]:
    data_by_label: dict[str, contact_model.AggregatedContacts] = {}
    layers = {}
    for stage in STAGES:
        data = contact_model.load_frozen_p9016_aggregate(stage.bin_size)
        data_by_label[stage.label] = data
        layers[stage.label] = _check_layer_budget(data, stage)
    starts, state_objects = _build_common_starts(context, data_by_label)
    fixed = {}
    c1 = {}
    c3 = {}
    for candidate in CANDIDATES:
        for stage in STAGES:
            data = data_by_label[stage.label]
            if stage == STAGES[0]:
                state = reconstruction_init.initialize_approved_candidate(
                    candidate.initialization_candidate,
                    tuple(name for name, _ in context.headers),
                    tuple(int(length) for _name, length in context.headers),
                    stage.bin_size,
                )
            else:
                state = state_objects[candidate.candidate_id][stage.label]
            coords = _state_array(state)
            # 上面的完整 common-start chain 就是 layer-interface fixture。
            # 所有 objective checks 都复用其实际坐标。
            key = "%s/%s" % (candidate.candidate_id, stage.label)
            fixed[key] = _fixed_objective_checks(data, coords)
            c1[key] = _c1_bend_check(data, coords)
            c3[key] = _c3_exposure_check(data)
    c2 = _c2_domain_checks(data_by_label["5m"], root)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "created_at_utc": _utc_now(),
        "tolerances": {
            "strict_atol": STRICT_ATOL,
            "strict_rtol": STRICT_RTOL,
            "finite_difference_atol": FD_ATOL,
            "finite_difference_rtol": FD_RTOL,
            "finite_difference_note": "central-difference diagnostic only; strict run gate remains 1e-12/1e-10",
        },
        "checks": {
            "full_grid_aggregate_and_denominator_conservation": True,
            "all_variant_physical_starts_shared": True,
            "c0_fixed_state_value_gradient_equivalence": True,
            "c1_exact_bend_removal": True,
            "c3_only_exposure_changes": True,
            "c2_map_inverse_gradient_and_free_domain": True,
            "no_optimization_called": True,
            "no_native_fdg_called": True,
            "phase_or_reference_opened": False,
        },
        "layers": layers,
        "shared_starts": starts,
        "c0_fixed_state": fixed,
        "c1_bend": c1,
        "c3_exposure": c3,
        "c2_domain": c2,
    }


def _protocol_payload(context: reconstruct.RunContext, preflight_sha: str,
                      source_manifest: Mapping[str, Any], threads: int) -> dict[str, Any]:
    source_hashes = {source: sha256_file(ROOT / source) for source in TRAINING_SOURCES}
    planned = [
        {"attempt_id": "%s-%s-%s" % (variant, candidate.candidate_id, stage.label),
         "variant": variant, "candidate_id": candidate.candidate_id, "stage": stage.label,
         "bin_size_bp": stage.bin_size, "maxiter": stage.maxiter, "maxfun": stage.maxfun}
        for variant in VARIANTS for candidate in CANDIDATES for stage in STAGES
    ]
    released = [row for row in planned if row["variant"] == RELEASED_VARIANT]
    comparator_paths = [OLD_RUN / "config.json", OLD_RUN / "selection.json", OLD_RUN / "termination_audit.json"]
    comparator_paths.extend(
        OLD_RUN / "stages" / candidate.candidate_id / (stage.label + ".json")
        for candidate in CANDIDATES for stage in STAGES
    )
    comparator = {
        "run": str(OLD_RUN.relative_to(ROOT)),
        "files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in comparator_paths
        ],
    }
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "frozen",
        "scope": "reference-free phase-free training; C0 only in this release",
        "created_at_utc": _utc_now(),
        "cohort": dict(context.cohort),
        "input": {
            "path": str(Path(context.data_path).resolve()),
            "sha256": context.data_sha256,
            "raw_source_sha256": EXPECTED_RAW_SHA256,
            "training_columns": ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"],
            "uses_all_1703888_records": True,
        },
        "grid": {
            "origin_bp": 0,
            "full_grid": True,
            "final_bin_size_bp": 1_000_000,
            "n_loci": 2_645,
            "n_physical_beads": 5_290,
            "n_tracks": 40,
            "coordinate_units": "dimensionless_R1",
        },
        "sources": _source_contract(context),
        "source_code_sha256": source_hashes,
        "controlled_source_manifest": dict(source_manifest),
        "020_comparator": comparator,
        "variants": [allele_models.model_spec(model).as_dict() for model in VARIANTS],
        "objective_contract": {
            "c0_is_v1": True,
            "epsilon": 1e-6,
            "r0": "2*l0",
            "l0": "(2*n_loci)^(-1/3)",
            "repulsion_threshold": "0.7*l0",
            "weights": {"count": 1.0, "p_prior": 1.0, "bond": 1.0,
                        "repulsion": 1.0, "bend": 0.01},
            "p_floor": 1e-4,
            "p_prior_strength": 1e-4,
            "exposure": "sqrt(endpoint_count + 10) / full_grid_mean",
            "same_bin": "per-bin saturated Poisson nuisance",
        },
        "mapping_diagnostics": {
            "source": "pr.allele_models._MappedJointObjective.map_diagnostics",
            "record_fields": ["objective_eval_count", "nonidentity_map_eval_count",
                              "nonidentity_bead_eval_count", "max_raw_radius_seen",
                              "max_physical_radius_seen"],
            "accepted_trial_split": "not exposed by current L-BFGS callback; all line-search probes are counted",
            "free_domain_field": "ever_physical_radius_gt1",
            "used_for_selection_or_budget": False,
        },
        "initialization_contract": {
            "consensus_base_seed": 1103,
            "random_base_seed": 2207,
            "target_radius": 0.8,
            "expansion": "numeric genomic sorting, bracket linear interpolation, outside nearest endpoint fill",
            "consensus": "cXXa shared Z plus seed-1103 smooth half-difference",
            "random": "retain both source copies with base seed 2207",
            "global_frame": "one common physical center and uniform scale; maxR=0.8",
            "jitter": "0.025*l0 on new/interpolated/endpoint-filled/repeated loci",
            "warm_seed": "3301 + bin_size//1e6 + candidate_base_seed",
            "warm_clip": "outside-unit-ball radial clip to 1-1e-6",
            "additional_center_rescale_between_layers": False,
        },
        "optimization": {
            "schedule": [
                {"label": stage.label, "bin_size_bp": stage.bin_size,
                 "maxiter": stage.maxiter, "maxfun": stage.maxfun,
                 "maxls": 20, "ftol": 1e-10, "gtol": 1e-6,
                 "checkpoint_every_accepted": 10}
                for stage in STAGES
            ],
            "p_init_first_layer": 0.75,
            "carry_raw_q_between_layers": True,
            "budget_not_convergence": True,
        },
        "selection": {
            "criterion": "count_nll_per_record",
            "direction": "minimize",
            "tie_tolerance_per_record": SELECTION_TIE_TOLERANCE,
            "tie_break": "preregistered_order",
            "order": [candidate.candidate_id for candidate in CANDIDATES],
            "includes_candidate_invariant_diagonal_layer": True,
            "includes_priors": False,
            "reference_used": False,
            "phase_used": False,
        },
        "attempt_plan": {
            "planned_variant_count": len(VARIANTS),
            "planned_candidate_count": len(CANDIDATES),
            "planned_stage_count": len(STAGES),
            "planned_attempt_count": len(planned),
            "released_variant": RELEASED_VARIANT,
            "released_attempt_count": len(released),
            "released_attempts": released,
            "blocked_variants": list(BLOCKED_VARIANTS),
            "blocked_attempt_count": len(planned) - len(released),
            "runner_refuses_blocked_variants": True,
        },
        "preflight": {"path": "preflight.json", "sha256": preflight_sha},
        "runtime": {
            "workers": 2,
            "threads": threads,
            "thread_env": {name: "1" for name in
                           ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "native_fdg_started": False,
        },
        "boundary": {"phase_used": False, "reference_used": False,
                      "oracle_coordinates_opened": False, "evaluation_outputs_opened": False},
    }


def _augment_config(base: Mapping[str, Any], protocol_sha: str, preflight_sha: str,
                    source_manifest: Mapping[str, Any], threads: int) -> dict[str, Any]:
    config = dict(base)
    config["controlled_runner"] = {
        "schema": RUN_SCHEMA,
        "runner_source": "pr/c0_controlled_runner.py",
        "runner_sha256": sha256_file(HERE),
        "protocol_path": "controlled_protocol.json",
        "protocol_sha256": protocol_sha,
        "preflight_path": "preflight.json",
        "preflight_sha256": preflight_sha,
        "controlled_source_manifest": dict(source_manifest),
        "planned_attempts": 30,
        "released_attempts": 6,
        "released_variant": RELEASED_VARIANT,
        "blocked_variants": list(BLOCKED_VARIANTS),
    }
    config["selection_contract"] = {
        "criterion": "count_nll_per_record",
        "tie_tolerance_per_record": SELECTION_TIE_TOLERANCE,
        "tie_break": "preregistered_order",
        "order": [candidate.candidate_id for candidate in CANDIDATES],
        "includes_candidate_invariant_diagonal_layer": True,
        "includes_priors": False,
        "reference_used": False,
        "phase_used": False,
    }
    config["runtime"]["candidate_workers"] = 2
    config["runtime"]["thread_env"] = {name: "1" for name in
                                         ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    config["training_boundary"].update({
        "phase_used": False,
        "reference_used": False,
        "oracle_coordinates_opened": False,
        "evaluation_outputs_opened": False,
        "native_fdg_started": False,
    })
    return config


def _stage_history_common(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in history:
        item = {}
        for key in ("iteration", "nfev", "fun", "p", "components", "gradient_norm"):
            if key in row:
                item[key] = row[key]
        result.append(item)
    return result


def _compare_json_numeric(actual: Any, expected: Any, path: str,
                          differences: list[str]) -> None:
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            differences.append("%s keys differ" % path)
            return
        for key in sorted(actual):
            _compare_json_numeric(actual[key], expected[key], "%s.%s" % (path, key), differences)
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            differences.append("%s length %d != %d" % (path, len(actual), len(expected)))
            return
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_json_numeric(left, right, "%s[%d]" % (path, index), differences)
        return
    if isinstance(actual, (int, float, np.integer, np.floating)) and isinstance(expected, (int, float, np.integer, np.floating)):
        if not _within(float(actual), float(expected)):
            differences.append("%s %.17g != %.17g" % (path, float(actual), float(expected)))
        return
    if actual != expected:
        differences.append("%s %r != %r" % (path, actual, expected))


def _grid_for_stage(context: reconstruct.RunContext, stage: reconstruct.StageSpec) -> reconstruction_report.FullGrid:
    return reconstruction_report.FullGrid(
        tuple(reconstruction_report.Chromosome(name, int(length)) for name, length in context.headers),
        stage.bin_size, 0,
    )


def _structures_to_array(structures: Mapping[str, Mapping[int, np.ndarray]],
                         data: contact_model.AggregatedContacts) -> np.ndarray:
    values = np.empty((2, data.n_loci, 3), dtype=np.float64)
    for spec in data.track_specs:
        slc = data.chromosome_slice(spec.chromosome_index)
        track = structures[spec.name]
        for global_index, local_bin in zip(range(slc.start, slc.stop), data.locus_bin[slc], strict=True):
            values[spec.copy_index, global_index] = track[int(local_bin * stage.bin_size)]
    return values


def _load_stage_coordinates(stage_doc: Mapping[str, Any], root: Path,
                             context: reconstruct.RunContext,
                             stage: reconstruct.StageSpec,
                             data: contact_model.AggregatedContacts,
                             key: str) -> tuple[np.ndarray, str]:
    info = stage_doc[key]
    path = Path(str(info["path"]))
    if not path.is_absolute():
        path = root / path
    expected_sha = str(info["sha256"])
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ControlledRunnerError("%s coordinate hash mismatch" % key)
    structures = reconstruction_report.read_full_grid_coordinates(
        path, _grid_for_stage(context, stage), expected_sha)
    return _structures_to_array(structures, data), actual_sha


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as payload:
        result: dict[str, np.ndarray | str] = {}
        for key in payload.files:
            value = payload[key]
            result[key] = str(value.item()) if key == "fullhistory_json" else value.copy()
    return result


def _compare_checkpoint(old_path: Path, new_path: Path, old_history: Any,
                        new_history: Any, differences: list[str]) -> dict[str, Any]:
    old = _load_npz_arrays(old_path)
    new = _load_npz_arrays(new_path)
    state_fields = ("coordinates", "theta", "y", "positions", "chromosome_index")
    state_diffs = {}
    state_exact = {}
    for field in state_fields:
        left = np.asarray(old[field])
        right = np.asarray(new[field])
        diff = _max_abs(left, right) if left.dtype.kind == "f" else (0.0 if np.array_equal(left, right) else float("inf"))
        state_diffs[field] = diff
        state_exact[field] = bool(np.array_equal(left, right))
        if not state_exact[field]:
            differences.append("checkpoint %s state %s is not byte-identical" % (new_path.name, field))
    try:
        old_hist = json.loads(str(old["fullhistory_json"]))
        new_hist = json.loads(str(new["fullhistory_json"]))
        common_old = [
            {key: value for key, value in row.items() if key != "gradient_norm"}
            for row in _stage_history_common(old_hist)
        ]
        common_new = [
            {key: value for key, value in row.items() if key != "gradient_norm"}
            for row in _stage_history_common(new_hist)
        ]
        history_differences: list[str] = []
        _compare_json_numeric(common_new, common_old, "checkpoint.history", history_differences)
        # 有意排除 elapsed time；新 runner 保留新增的 gradient 字段，020 record 中没有该字段。
        history_differences = [item for item in history_differences if ".gradient_norm" not in item]
        differences.extend("%s: %s" % (new_path.name, item) for item in history_differences)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        differences.append("checkpoint %s history parse failed: %s" % (new_path.name, exc))
    return {"state_max_abs": state_diffs, "state_exact": state_exact}


def build_c0_gate(root: Path, context: reconstruct.RunContext,
                  results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    differences: list[str] = []
    stage_rows = []
    data_cache = {stage.label: contact_model.load_frozen_p9016_aggregate(stage.bin_size)
                  for stage in STAGES}
    per_candidate = {}
    for candidate in CANDIDATES:
        result = next(item for item in results if item["id"] == candidate.candidate_id)
        candidate_rows = []
        old_candidate_root = OLD_RUN
        for stage in STAGES:
            old_stage_path = OLD_RUN / "stages" / candidate.candidate_id / (stage.label + ".json")
            new_stage_path = root / "stages" / candidate.candidate_id / (stage.label + ".json")
            with old_stage_path.open() as handle:
                old_stage = json.load(handle)
            with new_stage_path.open() as handle:
                new_stage = json.load(handle)
            old_data_budget = old_stage["data_budget"]
            new_data_budget = new_stage["data_budget"]
            budget_diffs: list[str] = []
            _compare_json_numeric(new_data_budget, old_data_budget, "data_budget", budget_diffs)
            differences.extend("%s/%s: %s" % (candidate.candidate_id, stage.label, item)
                               for item in budget_diffs)
            initial_new, initial_new_sha = _load_stage_coordinates(
                new_stage, root, context, stage, data_cache[stage.label], "initial_coordinates")
            initial_old, initial_old_sha = _load_stage_coordinates(
                old_stage, OLD_RUN, context, stage, data_cache[stage.label], "initial_coordinates")
            final_new, final_new_sha = _load_stage_coordinates(
                new_stage, root, context, stage, data_cache[stage.label], "final_coordinates")
            final_old, final_old_sha = _load_stage_coordinates(
                old_stage, OLD_RUN, context, stage, data_cache[stage.label], "final_coordinates")
            initial_diff = _max_abs(initial_new, initial_old)
            final_diff = _max_abs(final_new, final_old)
            initial_exact = bool(np.array_equal(initial_new, initial_old))
            final_exact = bool(np.array_equal(final_new, final_old))
            if not initial_exact:
                differences.append("%s/%s initial physical state is not byte-identical" %
                                   (candidate.candidate_id, stage.label))
            if not final_exact:
                differences.append("%s/%s final physical state is not byte-identical" %
                                   (candidate.candidate_id, stage.label))
            old_fit = old_stage["fit"]
            new_fit = new_stage["fit"]
            solver_fields = ("success", "status", "message", "nit", "nfev", "actual_nfev",
                             "scipy_nfev", "njev", "maxiter", "maxfun", "maxls", "ftol", "gtol")
            solver_diffs = []
            for field in solver_fields:
                if field in old_fit["solver"] and field in new_fit["solver"]:
                    _compare_json_numeric(new_fit["solver"][field], old_fit["solver"][field],
                                          "solver.%s" % field, solver_diffs)
            differences.extend("%s/%s: %s" % (candidate.candidate_id, stage.label, item)
                               for item in solver_diffs)
            fit_fields = ("initial_total", "final_total", "initial_components", "final_components",
                          "p", "q_in", "q_out")
            fit_diffs = []
            for field in fit_fields:
                if field in old_fit and field in new_fit:
                    _compare_json_numeric(new_fit[field], old_fit[field], "fit.%s" % field, fit_diffs)
            differences.extend("%s/%s: %s" % (candidate.candidate_id, stage.label, item)
                               for item in fit_diffs)
            old_hist = _stage_history_common(old_fit["history"])
            new_hist = _stage_history_common(new_fit["history"])
            history_diffs: list[str] = []
            # Gradient norm 是新增 bookkeeping；elapsed 有意缺失。
            common_old = [{key: value for key, value in row.items() if key != "gradient_norm"}
                          for row in old_hist]
            common_new = [{key: value for key, value in row.items() if key != "gradient_norm"}
                          for row in new_hist]
            _compare_json_numeric(common_new, common_old, "fit.history", history_diffs)
            differences.extend("%s/%s: %s" % (candidate.candidate_id, stage.label, item)
                               for item in history_diffs)
            old_cp = old_fit.get("checkpoint_paths", [])
            new_cp = new_fit.get("checkpoint_paths", [])
            if len(old_cp) != len(new_cp):
                differences.append("%s/%s checkpoint count %d != %d" %
                                   (candidate.candidate_id, stage.label, len(new_cp), len(old_cp)))
            checkpoint_rows = []
            for old_cp_text, new_cp_text in zip(old_cp, new_cp):
                old_cp_path = OLD_RUN / old_cp_text
                new_cp_path = root / new_cp_text
                checkpoint_rows.append(_compare_checkpoint(
                    old_cp_path, new_cp_path, old_fit["history"], new_fit["history"], differences))
            candidate_rows.append({
                "stage": stage.label,
                "old_initial_sha256": initial_old_sha,
                "new_initial_sha256": initial_new_sha,
                "initial_sha256_equal": initial_old_sha == initial_new_sha,
                "initial_max_abs": initial_diff,
                "initial_byte_equal": initial_exact,
                "old_final_sha256": final_old_sha,
                "new_final_sha256": final_new_sha,
                "final_sha256_equal": final_old_sha == final_new_sha,
                "final_max_abs": final_diff,
                "final_byte_equal": final_exact,
                "checkpoint_count": len(new_cp),
                "checkpoint_state_rows": checkpoint_rows,
                "old_metadata": {
                    "status": old_fit.get("status"),
                    "budget_exhausted": old_fit.get("budget_exhausted"),
                    "termination_reason": old_fit.get("termination_reason"),
                },
                "new_metadata": {
                    "status": new_fit.get("status"),
                    "budget_exhausted": new_fit.get("budget_exhausted"),
                    "termination_reason": new_fit.get("termination_reason"),
                },
            })
            stage_rows.append(candidate_rows[-1])
        per_candidate[candidate.candidate_id] = candidate_rows

    with (root / "selection.json").open() as handle:
        selection = json.load(handle)
    if selection["selection"]["selected_id"] != "random_joint":
        differences.append("new selection did not select random_joint")
    if selection["objective_contract"]["tie_tolerance_per_record"] != SELECTION_TIE_TOLERANCE:
        differences.append("new selection tie tolerance changed")
    if selection["selection"]["rule"] != "minimize_count_nll_per_record":
        differences.append("new selection rule changed")
    old_selection_path = OLD_RUN / "selection.json"
    with old_selection_path.open() as handle:
        old_selection = json.load(handle)
    if selection["selection"]["selected_id"] != old_selection["selection"]["selected_id"]:
        differences.append("selected candidate differs from 020")
    new_scores = {row["id"]: float(row["count_nll_per_record"]) for row in selection["candidates"]}
    old_scores = {row["id"]: float(row["count_nll_per_record"]) for row in old_selection["candidates"]}
    score_diffs = {key: abs(new_scores[key] - old_scores[key]) for key in new_scores}
    for key, value in score_diffs.items():
        if value > STRICT_ATOL + STRICT_RTOL * max(1.0, abs(old_scores[key])):
            differences.append("selection score %s differs by %.17g" % (key, value))
    gate = {
        "schema": GATE_SCHEMA,
        "status": "passed" if not differences else "failed",
        "created_at_utc": _utc_now(),
        "reference_free": True,
        "phase_free": True,
        "old_training_comparator": str(OLD_RUN.relative_to(ROOT)),
        "selection_rule": {
            "criterion": "count_nll_per_record",
            "tie_tolerance_per_record": SELECTION_TIE_TOLERANCE,
            "tie_break": "preregistered_order",
            "order": [candidate.candidate_id for candidate in CANDIDATES],
        },
        "tolerances": {
            "state_gate": "byte_equal_required",
            "numeric_atol": STRICT_ATOL,
            "numeric_rtol": STRICT_RTOL,
            "elapsed_seconds": "excluded_from_reproduction_comparison",
            "historical_metadata": {
                "budget_exhausted": "allowed classification correction from old 020 record",
                "termination_reason": "raw solver fields are gated; textual classification may differ",
                "gradient_norm": "new checkpoint bookkeeping field; not present in old 020",
            },
        },
        "checks": {
            "six_stage_records_present": len(stage_rows) == 6,
            "all_initial_physical_states_byte_equal": all(row["initial_byte_equal"] for row in stage_rows),
            "all_final_physical_states_byte_equal": all(row["final_byte_equal"] for row in stage_rows),
            "all_initial_coordinate_file_sha256_equal": all(row["initial_sha256_equal"] for row in stage_rows),
            "all_final_coordinate_file_sha256_equal": all(row["final_sha256_equal"] for row in stage_rows),
            "all_checkpoint_states_byte_equal": not any(
                not state for row in stage_rows for checkpoint in row["checkpoint_state_rows"]
                for state in checkpoint["state_exact"].values()),
            "all_solver_raw_fields_match": not any("solver." in item for item in differences),
            "all_history_objective_fields_match": not any("fit.history" in item for item in differences),
            "all_final_objective_fields_match": not any("fit." in item for item in differences),
            "selection_same_and_rule_frozen": not any("selection" in item for item in differences),
        },
        "selection": {
            "new_selected_id": selection["selection"]["selected_id"],
            "old_selected_id": old_selection["selection"]["selected_id"],
            "new_scores": new_scores,
            "old_scores": old_scores,
            "score_max_abs": max(score_diffs.values(), default=0.0),
        },
        "stage_comparisons": per_candidate,
        "differences": differences,
    }
    return gate


def _write_report(root: Path, gate: Mapping[str, Any], terminal: Mapping[str, Any],
                  preflight: Mapping[str, Any], run_result: Mapping[str, Any]) -> None:
    status = str(gate.get("status"))
    lines = [
        "# C0 受控训练复现报告",
        "",
        "- schema: `%s`" % REPORT_SCHEMA,
        "- 状态: **%s**" % ("通过" if status == "passed" else "失败"),
        "- 范围：仅不使用定相数据和参考结构的 C0 训练；不含 R1/R2/R3 评价。",
        "- 当前放行: `consensus_joint` 与 `random_joint`，每个完整执行 `5m -> 2m -> 1m`。",
        "- 其余变体：`C1`、`C2-map`、`C2-free`、`C3` 的 24 次阶段尝试已冻结但阻塞，等待父侧验收。",
        "",
        "## 冻结合同",
        "",
        "- 输入：`inputs/P9016.snpfree.pairs.gz`，SHA256 `%s`，全量 1,703,888 条记录。" % EXPECTED_INPUT_SHA256,
        "- 网格：起点 0；最终 2,645 个位点、5,290 个物理珠点、40 条轨迹；保留所有分箱，包括零分箱和末端分箱。",
        "- 优化预算：`5m 300/930 -> 2m 200/630 -> 1m 240/750`，`maxls=20`、`ftol=1e-10`、`gtol=1e-6`，每 10 个已接受的检查点。",
        "- 首层 `p=0.75`，后续层携带原始 `q`；选择为全量无标签的 `count_nll_per_record` 最小化，平局容差为 `1e-9`，按预注册顺序破平局，不含先验，包含候选不变的对角层。",
        "",
        "## 预检",
        "",
        "- 聚合与分母守恒：5m/2m/1m 均通过，分别使用本层分箱尺寸的对角干扰层。",
        "- C0 固定状态的值/梯度等价、C1 精确扣除弯曲项、C3 仅改变曝光、C2 map/free 域钩子与写出器检查：全部通过。",
        "- C2-free 的半径 1.2 点经合法打包、评估和写出流程保留，未套用球面反变换、裁剪或重缩放。",
        "",
        "## 数值 gate",
        "",
        "- C0 六个阶段全部有终态记录、检查点、入口/终点物理状态，并与 020 历史记录逐阶段比较。",
        "- 状态门控要求逐字节一致；数值字段使用 abs `1e-12` / rel `1e-10`；耗时秒数不作复现判定。",
        "- 门控状态：`%s`；差异条数：`%d`。" % (status, len(gate.get("differences", []))),
        "- 选中结果：`%s`；新旧最终计数 NLL 差的最大绝对值 `%.17g`。" % (
            run_result.get("selected_id"), float(gate["selection"]["score_max_abs"])),
        "",
        "## 产物",
        "",
        "- `controlled_protocol.json`：完整 30 次尝试的冻结计划与 C0 放行门。",
        "- `preflight.json`：无优化针对性预检证据。",
        "- `C0_gate.json`：不使用参考结构和定相数据的历史数值复现门控。",
        "- `terminal_evidence.json`：六个阶段的求解器原始字段、停止原因、退出状态与哈希值。",
        "- `selection.json`：无标签候选选择；`selected.3dg` 仅为训练输出，不代表评价结论。",
        "",
        "本报告不读取或生成参考结构、定相数据、评价数据和 R2 数据；预算结束被记录为求解器未收敛/预算终止时，不解释为收敛成功。",
    ]
    _write_bytes_exclusive(root / "REPORT.zh-CN.md", ("\n".join(lines) + "\n").encode("utf-8"))


def _terminal_evidence(root: Path, gate: Mapping[str, Any],
                       results: Sequence[Mapping[str, Any]], started_utc: str,
                       ended_utc: str, protocol_sha: str, preflight_sha: str) -> dict[str, Any]:
    attempts = []
    for candidate in CANDIDATES:
        result = next(item for item in results if item["id"] == candidate.candidate_id)
        for layer in result.get("layers", []):
            stage = layer["stage"]
            stage_path = root / layer["path"]
            with stage_path.open() as handle:
                record = json.load(handle)
            fit = record["fit"]
            solver = fit["solver"]
            attempts.append({
                "attempt_id": "%s-%s-%s" % (RELEASED_VARIANT, candidate.candidate_id, stage),
                "variant": RELEASED_VARIANT,
                "candidate_id": candidate.candidate_id,
                "stage": stage,
                "stage_record": layer["path"],
                "stage_record_sha256": sha256_file(stage_path),
                "stage_exit_code": 0,
                "status": fit["status"],
                "stop_reason": fit["termination_reason"],
                "budget_exhausted": bool(fit["budget_exhausted"]),
                "solver": {
                    "success": solver["success"], "status": solver["status"],
                    "message": solver["message"], "nit": solver["nit"],
                    "nfev": solver["nfev"], "actual_nfev": solver["actual_nfev"],
                    "scipy_nfev": solver["scipy_nfev"], "elapsed_seconds": solver["elapsed_seconds"],
                },
                "initial_coordinates_sha256": record["initial_coordinates"]["sha256"],
                "final_coordinates_sha256": record["final_coordinates"]["sha256"],
                "checkpoint_count": len(fit.get("checkpoint_paths", [])),
                "initial_total": fit["initial_total"],
                "final_total": fit["final_total"],
                "p": fit["p"], "q_in": fit["q_in"], "q_out": fit["q_out"],
            })
    return {
        "schema": TERMINAL_SCHEMA,
        "status": "passed" if gate.get("status") == "passed" else "failed",
        "runner_exit_code": 0,
        "started_at_utc": started_utc,
        "ended_at_utc": ended_utc,
        "protocol_sha256": protocol_sha,
        "preflight_sha256": preflight_sha,
        "variant_release": RELEASED_VARIANT,
        "planned_stage_attempt_count": 30,
        "executed_stage_attempt_count": len(attempts),
        "all_six_stage_attempts_complete": len(attempts) == 6,
        "gate_status": gate.get("status"),
        "attempts": attempts,
        "selection": gate.get("selection"),
        "training_boundary": {
            "phase_used": False,
            "reference_used": False,
            "evaluation_outputs_opened": False,
            "native_fdg_started": False,
        },
    }


def _finalize_training(root: Path, context: reconstruct.RunContext,
                       config: Mapping[str, Any], static_files: Mapping[str, Any],
                       results: list[dict[str, Any]]) -> dict[str, Any]:
    final_status: dict[str, str] = {}
    for result in results:
        if result.get("optimization", {}).get("terminal_status") == "failed":
            final_status[result["id"]] = "failed"
            continue
        try:
            reconstruct._rescore_final_candidate(root, context, config, result, reconstruct.DEFAULT_HOOKS)
            final_status[result["id"]] = result["optimization"]["terminal_status"]
            reconstruct._write_candidate_status(root, result["id"], {
                "status": result["optimization"]["terminal_status"], "updated_at_utc": reconstruct._utc_now(),
                "current_stage": "1m", "final_count_rescore": "complete",
            })
        except Exception as exc:
            reason = "final_count_rescore: %s: %s" % (type(exc).__name__, exc)
            reconstruct._mark_rescore_failure(result, reason)
            final_status[result["id"]] = "failed"
            reconstruct._write_candidate_status(root, result["id"], {
                "status": "failed", "updated_at_utc": reconstruct._utc_now(),
                "current_stage": "final_count_rescore", "failure_reason": reason,
            })
    attempts = [attempt for result in results for attempt in result["attempts"]]
    for attempt in attempts:
        reconstruct._append_attempt_event(root, reconstruct._attempt_journal_event(attempt), attempt)
    usable = [result for result in results if result.get("optimization", {}).get("terminal_status") != "failed"
              and result.get("count_nll_per_record") is not None]
    if not usable:
        reconstruct._write_new_json(root / "training_summary.json", {
            "status": "training_failed", "candidates": results, "attempts": attempts,
        })
        reconstruct._write_global_status(root, "training_failed", final_status,
                                         failure_reason="all pre-registered candidates failed")
        raise ControlledRunnerError("all C0 candidates failed before selection")
    selected_id = reconstruct.select_candidate(results, tuple(candidate.candidate_id for candidate in CANDIDATES))
    selected = next(result for result in results if result["id"] == selected_id)
    selected_coordinate = reconstruct._copy_selected_coordinates(root, selected)
    selection = reconstruct._selection_document(root, context, config, static_files, results, attempts,
                                                selected_id, selected_coordinate)
    selection_path = root / "selection.json"
    reconstruct._write_new_json(selection_path, selection)
    verified = reconstruct.verify_completed_training(root, strict_p9016=True)
    reconstruct._write_global_status(root, "training_complete", final_status,
                                     selection_path=reconstruct._relative_to_run(root, selection_path),
                                     selected_id=selected_id)
    return {
        "outdir": str(root),
        "selection_path": str(selection_path),
        "selected_id": selected_id,
        "selected_coordinates": selected_coordinate,
        "verified_selection": verified,
        "candidates": results,
    }


def run_c0(outdir: str | os.PathLike[str], *, workers: int = 2, threads: int = 1) -> dict[str, Any]:
    """精确冻结并执行三个 layer 上的两个 C0 candidates。"""
    if int(workers) != 2:
        raise ControlledRunnerError("C0 release is frozen to two independent candidate workers")
    _set_threads(threads)
    context = reconstruct.production_context()
    reconstruct._validate_context(context)
    root = _fresh_root(outdir)
    started_utc = _utc_now()
    try:
        source_manifest = _snapshot_controlled_sources(root)
        preflight = run_preflight(context, root)
        preflight_sha = _write_json_exclusive(root / "preflight.json", preflight)
        protocol = _protocol_payload(context, preflight_sha, source_manifest, threads)
        protocol_sha = _write_json_exclusive(root / "controlled_protocol.json", protocol)
        config = _augment_config(
            reconstruct._build_config(context, workers=workers, threads=threads),
            protocol_sha, preflight_sha, source_manifest, threads,
        )
        static_files = reconstruct._write_static_run_files(root, context, config)
        reconstruct._write_global_status(root, "training", {candidate.candidate_id: "pending" for candidate in CANDIDATES})
        results = reconstruct._run_candidates(root, context, config, reconstruct.DEFAULT_HOOKS, workers, True)
        result = _finalize_training(root, context, config, static_files, results)
        gate = build_c0_gate(root, context, results)
        _write_json_exclusive(root / "C0_gate.json", gate)
        ended_utc = _utc_now()
        terminal = _terminal_evidence(root, gate, results, started_utc, ended_utc,
                                      protocol_sha, preflight_sha)
        _write_json_exclusive(root / "terminal_evidence.json", terminal)
        _write_report(root, gate, terminal, preflight, result)
        _write_json_exclusive(root / "run_receipt.json", {
            "schema": "c0-run-receipt-v1",
            "status": "passed" if gate["status"] == "passed" else "failed_gate",
            "runner_exit_code": 0 if gate["status"] == "passed" else 3,
            "started_at_utc": started_utc,
            "ended_at_utc": ended_utc,
            "outdir": str(root),
            "protocol": {"path": "controlled_protocol.json", "sha256": protocol_sha},
            "preflight": {"path": "preflight.json", "sha256": preflight_sha},
            "gate": {"path": "C0_gate.json", "sha256": sha256_file(root / "C0_gate.json")},
            "terminal": {"path": "terminal_evidence.json", "sha256": sha256_file(root / "terminal_evidence.json")},
            "planned_stage_attempt_count": 30,
            "executed_stage_attempt_count": 6,
            "remaining_blocked_stage_attempt_count": 24,
            "selected_id": result["selected_id"],
        })
        if gate["status"] != "passed":
            raise ControlledRunnerError("C0 numerical gate failed; remaining variants stay blocked")
        return {
            "status": "passed",
            "outdir": str(root),
            "selected_id": result["selected_id"],
            "gate_path": str(root / "C0_gate.json"),
            "terminal_path": str(root / "terminal_evidence.json"),
            "report_path": str(root / "REPORT.zh-CN.md"),
        }
    except Exception as exc:
        failure = {
            "schema": "c0-run-failure-v1",
            "status": "failed_before_gate",
            "created_at_utc": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "remaining_blocked_stage_attempt_count": 24,
            "training_boundary": {"phase_used": False, "reference_used": False,
                                  "evaluation_outputs_opened": False, "native_fdg_started": False},
        }
        failure_path = root / "run_failure.json"
        if not failure_path.exists():
            _write_json_exclusive(failure_path, failure)
        raise


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen phase-free C0 reproduction")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run-c0")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--workers", type=int, default=2)
    run_parser.add_argument("--threads", type=int, default=1)
    blocked = sub.add_parser("run-variant")
    blocked.add_argument("variant")
    blocked.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run-variant":
            if args.variant != RELEASED_VARIANT:
                raise ControlledRunnerError(
                    "variant %s is blocked until parent acceptance; only run-c0 is released" % args.variant)
            raise ControlledRunnerError("use run-c0 for the released C0 variant")
        result = run_c0(args.out, workers=args.workers, threads=args.threads)
        print(json.dumps(_jsonable(result), sort_keys=True))
        return 0
    except (ControlledRunnerError, FileExistsError, reconstruct.ReconstructionError,
            ValueError, OSError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
