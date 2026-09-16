"""完成持久化的 P3 checkpoints，不再运行优化。

该 adapter 保持冻结的 P3 runtime 不变。它创建新的 formal recovery run，保存原始 inputs/checkpoints 快照，选择共同的 1 Mb iter50 checkpoints，记录 pre-truth gates，然后才评估隔离的 synthetic truth。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np

from .contact_model import (
    FROZEN_P9016_CIS_RECORDS,
    FROZEN_P9016_INTER_RECORDS,
    FROZEN_P9016_RECORDS,
    JointObjective,
    assert_inside_unit_ball,
    sha256_file,
    tracks_from_coordinates,
    write_full_tracks,
)
from .reconstruction_init import warm_start_from_layer
from .v1_calibration import (
    CONDITIONS as FROZEN_CONDITIONS,
    FINAL_BIN,
    FULL_FINAL_LOCI,
    N_CHROMOSOMES,
    N_TRACKS,
    P_GENERATING,
    P_INITIAL,
    _control_count_terms,
    load_layer,
    _copy_shape_diagnostics,
    _load_truth,
    synthetic_r2,
    synthetic_r3,
)


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RUN = ROOT / "test_res" / "018-20260913_121446-v1-synthetic-calibration"
CONDITION_IDS = tuple(FROZEN_CONDITIONS)
MAIN_ITERATION = 50
PLANNED_FINAL_ITERATIONS = 80
PLANNED_FINAL_MAXFUN = 270
EXPECTED_FINAL_LOCI = FULL_FINAL_LOCI
EXPECTED_PHYSICAL_BEADS = 2 * EXPECTED_FINAL_LOCI


class RecoveryError(RuntimeError):
    """持久化 recovery 或 evaluation gate 无法核验时抛出。"""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wt") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "at") as handle:
        handle.write(json.dumps(_jsonable(value), sort_keys=True) + "\n")


def _load_json(path: str | Path) -> dict:
    with open(path, "rt") as handle:
        return json.load(handle)


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RecoveryError("invalid JSONL at %s:%d" % (path, line_number)) from exc
    return rows


def _copy_and_hash(source: Path, destination: Path) -> dict:
    if not source.is_file():
        raise RecoveryError("missing source artifact: %s" % source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise RecoveryError("snapshot hash mismatch: %s" % source)
    return {
        "source_path": str(source),
        "snapshot_path": str(destination),
        "sha256": source_hash,
        "bytes": int(source.stat().st_size),
    }


def _verify_registered_sources(config: Mapping[str, Any]) -> None:
    source_hashes = config.get("source_hashes", {})
    if not source_hashes:
        raise RecoveryError("recovery config lacks the original source hash manifest")
    for relative, expected in source_hashes.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RecoveryError("registered source changed: %s" % relative)
    for name, protocol in config.get("protocols", {}).items():
        relative = protocol["path"]
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != protocol["sha256"]:
            raise RecoveryError("registered protocol changed: %s" % name)


def _verify_recovery_manifest(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    frozen_config_path = run_dir / "frozen_protocol" / "config.json"
    if not config_path.is_file() or not frozen_config_path.is_file():
        raise RecoveryError("recovery config and frozen copy must exist")
    config_hash = sha256_file(config_path)
    if config_hash != sha256_file(frozen_config_path):
        raise RecoveryError("recovery config differs from frozen copy")
    config = _load_json(config_path)
    _verify_registered_sources(config)
    for name, protocol in config.get("protocols", {}).items():
        relative = protocol["path"]
        frozen_path = run_dir / "frozen_protocol" / Path(relative).name
        if (not frozen_path.is_file()
                or sha256_file(frozen_path) != protocol["sha256"]):
            raise RecoveryError("frozen protocol copy changed: %s" % name)
    return {
        "config_sha256": config_hash,
        "n_source_hashes": len(config.get("source_hashes", {})),
        "n_protocol_hashes": len(config.get("protocols", {})),
    }


def _original_config() -> dict:
    path = ORIGINAL_RUN / "config.json"
    if not path.is_file():
        raise RecoveryError("original 018 config is absent")
    return _load_json(path)


def _copy_original_snapshot(run_dir: Path, original_config: Mapping[str, Any]) -> dict:
    """复制 provenance 和非 truth artifacts；这里绝不复制 eval_truth。"""
    snapshot: dict[str, dict] = {}

    def copy_named(source: Path, relative_destination: str) -> None:
        entry = _copy_and_hash(run_dir / "source_snapshot" / relative_destination, source)
        # 上面的 helper call 使用了相反的语义参数顺序。
        snapshot[str(source.relative_to(ROOT))] = entry

    # 保持原始 formal metadata 和冻结 copies 逐字节不变。
    fixed_sources = [
        ORIGINAL_RUN / "config.json",
        ORIGINAL_RUN / "preregistration.md",
        ORIGINAL_RUN / "frozen_protocol" / "config.json",
        ORIGINAL_RUN / "frozen_protocol" / "source_manifest.json",
        ORIGINAL_RUN / "frozen_protocol" / "RECONSTRUCTION_V1_PROTOCOL.md",
        ORIGINAL_RUN / "frozen_protocol" / "RECONSTRUCTION_V1_CALIBRATION.md",
        ORIGINAL_RUN / "frozen_protocol" / "RECONSTRUCTION_V1_INITIALIZATION.md",
        ORIGINAL_RUN / "work" / "preparation.json",
        ORIGINAL_RUN / "logs" / "status.jsonl",
    ]
    # 直接复制固定 sources，使 source/snapshot hashes 可审计。
    for source in fixed_sources:
        relative = source.relative_to(ORIGINAL_RUN)
        destination = run_dir / "source_snapshot" / "018" / relative
        info = _copy_and_hash(source, destination)
        snapshot[str(source.relative_to(ROOT))] = {
            **info,
            "snapshot_path": str(destination.relative_to(run_dir)),
        }

    # 保存 018 使用的每个 synthetic layer 和 independent blind initialization。
    for source in sorted((ORIGINAL_RUN / "work").glob("*_layer.npz")):
        destination = run_dir / "inputs_snapshot" / source.name
        info = _copy_and_hash(source, destination)
        snapshot[str(source.relative_to(ROOT))] = {
            **info,
            "snapshot_path": str(destination.relative_to(run_dir)),
        }
    for source in sorted((ORIGINAL_RUN / "work").glob("*_independent_init_5mb.npz")):
        destination = run_dir / "inputs_snapshot" / source.name
        info = _copy_and_hash(source, destination)
        snapshot[str(source.relative_to(ROOT))] = {
            **info,
            "snapshot_path": str(destination.relative_to(run_dir)),
        }
    for source in sorted((ORIGINAL_RUN / "work").glob("*_history.json")):
        destination = run_dir / "source_snapshot" / "018" / "work" / source.name
        info = _copy_and_hash(source, destination)
        snapshot[str(source.relative_to(ROOT))] = {
            **info,
            "snapshot_path": str(destination.relative_to(run_dir)),
        }

    # 保留全部 layer log metadata 和空的 worker stdout artifacts。
    for source in sorted((ORIGINAL_RUN / "logs").glob("*")):
        if not source.is_file() or source.name == "status.jsonl":
            continue
        destination = run_dir / "source_snapshot" / "018" / "logs" / source.name
        info = _copy_and_hash(source, destination)
        snapshot[str(source.relative_to(ROOT))] = {
            **info,
            "snapshot_path": str(destination.relative_to(run_dir)),
        }

    # Main analysis checkpoints 是共同且持久的 iter50 files。
    for condition in CONDITION_IDS:
        for suffix in ("npz", "json"):
            source = ORIGINAL_RUN / "checkpoints" / (
                "%s_1mb_iter%04d.%s" % (condition, MAIN_ITERATION, suffix))
            destination = run_dir / "checkpoints_source" / "main" / source.name
            info = _copy_and_hash(source, destination)
            snapshot[str(source.relative_to(ROOT))] = {
                **info,
                "snapshot_path": str(destination.relative_to(run_dir)),
            }
    # 后续 A/B/D iter70 files 仅用于 diagnostics，绝不用于 main analysis。
    for condition in ("A", "B", "D"):
        for suffix in ("npz", "json"):
            source = ORIGINAL_RUN / "checkpoints" / (
                "%s_1mb_iter0070.%s" % (condition, suffix))
            destination = run_dir / "checkpoints_source" / "later_diagnostic" / source.name
            info = _copy_and_hash(source, destination)
            snapshot[str(source.relative_to(ROOT))] = {
                **info,
                "snapshot_path": str(destination.relative_to(run_dir)),
            }

    _write_json(run_dir / "source_snapshot_hashes.json", {
        "schema": "v1-recovery-source-snapshot-hashes",
        "original_run": str(ORIGINAL_RUN),
        "truth_artifacts_copied": False,
        "entries": snapshot,
    })
    # 单独保留原始 manifest，并与 recovery manifest 分开。
    original_manifest = _load_json(ORIGINAL_RUN / "frozen_protocol" / "source_manifest.json")
    _write_json(run_dir / "frozen_protocol" / "source_manifest_018.json", original_manifest)
    return snapshot


def _write_recovery_protocol(run_dir: Path) -> None:
    text = """# V1 P3 checkpoint recovery protocol

本 formal run 只完成 018-20260913_121446-v1-synthetic-calibration 中共同且持久的 1 Mb iter50 checkpoint；不再运行 optimizer
step。此前的 5 Mb（120 iterations）和 2 Mb（80 iterations）layers 已达到各自固定 budget。1 Mb 的 planned cap 为 80 iterations，但四个旧
workers 没有留下共同的 terminal artifact：A/B/D 有持久的 iter70，
C 有持久的 iter50。因此 main analysis 对每个
condition 都使用 iter50；该 checkpoint 在读取 synthetic truth 前选择，且与结果无关。

旧 worker 的 termination cause 未知。最后一个持久 checkpoint 之后的任何 line-search work，以及全部 L-BFGS internal history，均未知或已丢失。recovery terminal state 为 `interrupted_checkpoint_finalized`，而不是
`converged`，也不声称完成了 80-iteration fit。A/B/D iter70 files
仅复制到 `checkpoints_source/later_diagnostic/`；它们不用于
B-versus-C main misspecification comparison。

candidate 在读取原始 018 `eval_truth/` directory 下的隔离 synthetic truth 前写出并完成 SHA256 核验。由于没有生成 synthetic record-copy labels，R1 不适用。R2/R3 仅是 calibration diagnostics，不能证明 biological L2 recovery。
"""
    path = run_dir / "frozen_protocol" / "RECOVERY_PROTOCOL.md"
    path.write_text(text, encoding="utf-8")


def initialize_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "config.json").exists():
        raise RecoveryError("recovery run already initialized: %s" % run_dir)
    original_config = _original_config()
    _verify_registered_sources(original_config)
    for condition in CONDITION_IDS:
        if condition not in FROZEN_CONDITIONS:
            raise RecoveryError("unknown frozen condition %s" % condition)
    snapshot = _copy_original_snapshot(run_dir, original_config)
    protocol_entries = dict(original_config["protocols"])
    config = {
        "schema": "reconstruction-v1-p3-synthetic-calibration-recovery-v1",
        "run_id": run_dir.name,
        "status_at_freeze": "interrupted_checkpoint_finalized_before_truth_evaluation",
        "original_run": {
            "run_id": ORIGINAL_RUN.name,
            "path": str(ORIGINAL_RUN),
            "termination_cause": "unknown",
            "workers_alive_at_recovery_start": False,
            "background_jobs_at_recovery_start": 0,
            "original_files_modified": False,
        },
        "scope": {
            "biological_samples": 1,
            "synthetic_conditions_are_biological_replicates": False,
            "chromosomes": N_CHROMOSOMES,
            "tracks": N_TRACKS,
            "final_bin_bp": FINAL_BIN,
            "final_full_grid_loci": EXPECTED_FINAL_LOCI,
            "final_physical_beads": EXPECTED_PHYSICAL_BEADS,
            "training_side_prohibitions": [
                "P9016 phase columns",
                "P9016 reference 3DG",
                "oracle coordinates",
                "P9016 production fit",
                "truth-path access before pretruth gate",
            ],
        },
        "recovery_decision": {
            "main_checkpoint_iteration": MAIN_ITERATION,
            "main_checkpoint_selection": "common earliest durable 1 Mb checkpoint, selected before synthetic truth read",
            "planned_1mb_maxiter": PLANNED_FINAL_ITERATIONS,
            "planned_1mb_maxfun": PLANNED_FINAL_MAXFUN,
            "planned_cap_completed": False,
            "terminal_label": "interrupted_checkpoint_finalized",
            "solver_converged": False,
            "solver_convergence_status": "unknown",
            "lost_work_after_checkpoint": "unknown",
            "lbfgs_internal_history_available": False,
            "previous_layers": {
                "5mb": {"accepted_iterations": 120, "terminal": "budget_terminated"},
                "2mb": {"accepted_iterations": 80, "terminal": "budget_terminated"},
            },
            "later_diagnostic_checkpoints": {
                "A": 70,
                "B": 70,
                "D": 70,
                "C": None,
            },
            "additional_optimization": False,
            "additional_seed": False,
        },
        "frozen_input": original_config["frozen_input"],
        "protocols": protocol_entries,
        "source_hashes": original_config["source_hashes"],
        "original_018_source_manifest_sha256": sha256_file(
            ORIGINAL_RUN / "frozen_protocol" / "source_manifest.json"),
        "recovery_code_sha256": sha256_file(Path(__file__).resolve()),
        "snapshot_entry_count": len(snapshot),
    }
    _write_json(run_dir / "config.json", config)
    shutil.copy2(run_dir / "config.json", run_dir / "frozen_protocol" / "config.json")
    for protocol in protocol_entries.values():
        source = ROOT / protocol["path"]
        destination = run_dir / "frozen_protocol" / source.name
        _copy_and_hash(source, destination)
    _write_recovery_protocol(run_dir)
    _append_jsonl(run_dir / "logs" / "status.jsonl", {
        "event": "recovery_initialized",
        "run_id": run_dir.name,
        "main_checkpoint_iteration": MAIN_ITERATION,
        "truth_artifacts_copied": False,
        "termination_cause": "unknown",
    })
    return {
        "run_dir": str(run_dir),
        "config_sha256": sha256_file(run_dir / "config.json"),
        "source_snapshot_hashes": str(run_dir / "source_snapshot_hashes.json"),
        "truth_artifacts_copied": False,
    }


def _layer_path(run_dir: Path, condition: str, bin_size: int) -> Path:
    return run_dir / "inputs_snapshot" / (
        "%s_%dmb_layer.npz" % (condition, bin_size // 1_000_000))


def _checkpoint_path(run_dir: Path, condition: str, iteration: int, suffix: str) -> Path:
    return run_dir / "checkpoints_source" / "main" / (
        "%s_1mb_iter%04d.%s" % (condition, iteration, suffix))


def _later_checkpoint_path(run_dir: Path, condition: str, iteration: int, suffix: str) -> Path:
    return run_dir / "checkpoints_source" / "later_diagnostic" / (
        "%s_1mb_iter%04d.%s" % (condition, iteration, suffix))


def _layer_event(condition: str, bin_size: int, event: str) -> dict:
    path = ORIGINAL_RUN / "logs" / ("%s_%dmb.jsonl" % (condition, bin_size // 1_000_000))
    for row in _load_jsonl(path):
        if row.get("event") == event:
            return row
    raise RecoveryError("missing %s event for %s at %d Mb" % (event, condition, bin_size // 1_000_000))


def _terminal_layer_record(condition: str, bin_size: int) -> dict:
    return _layer_event(condition, bin_size, "layer_terminal")


def _load_checkpoint(path_npz: Path, path_json: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path_npz, allow_pickle=False) as payload:
        theta = payload["theta"].copy()
        y = payload["y"].copy()
        checkpoint_coordinates = payload["coordinates"].copy()
    metadata = _load_json(path_json)
    return theta, y, checkpoint_coordinates, metadata


def _numeric_close(left: float, right: float, scale: float = 1.0) -> tuple[bool, float, float]:
    difference = abs(float(left) - float(right))
    tolerance = 1e-10 * max(1.0, abs(float(left)), abs(float(right)), abs(float(scale)))
    return difference <= tolerance, difference, tolerance


def _audit_data(data) -> dict:
    audit = data.budget()
    expected = {
        "raw_records": FROZEN_P9016_RECORDS,
        "raw_inter": FROZEN_P9016_INTER_RECORDS,
        "raw_same_bin_plus_cis_offdiag": FROZEN_P9016_CIS_RECORDS,
    }
    checks = {
        "raw_records": bool(np.isclose(audit["raw_records"], expected["raw_records"], rtol=0.0, atol=0.0)),
        "raw_inter": bool(np.isclose(audit["raw_inter"], expected["raw_inter"], rtol=0.0, atol=0.0)),
        "raw_cis_total": bool(np.isclose(
            audit["raw_same_bin"] + audit["raw_cis_offdiag"],
            expected["raw_same_bin_plus_cis_offdiag"], rtol=0.0, atol=0.0)),
        "raw_conserved": bool(audit["raw_conserved"]),
        "aggregate_conserved": bool(audit["aggregate_conserved"]),
        "full_grid_pair_set": bool(audit["n_eligible_pairs"] == data.n_loci * (data.n_loci - 1) // 2),
    }
    return {"audit": audit, "checks": checks, "all_pass": bool(all(checks.values()))}


def _candidate_training_record(run_dir: Path, condition: str, main_record: dict) -> dict:
    layers = [
        _terminal_layer_record(condition, 5_000_000),
        _terminal_layer_record(condition, 2_000_000),
        main_record,
    ]
    npz_path = run_dir / "coords" / (condition + "_1mb_theta.npz")
    text_path = run_dir / "coords" / (condition + "_1mb.3dg")
    return {
        "condition": condition,
        "condition_name": FROZEN_CONDITIONS[condition]["name"],
        "status": "completed",
        "artifact_status": "interrupted_checkpoint_finalized",
        "terminal": "interrupted_checkpoint_finalized",
        "solver_converged": False,
        "solver_convergence_status": "unknown",
        "training_side_truth_access": False,
        "initialization_seed": 5101,
        "main_analysis_checkpoint_iteration": MAIN_ITERATION,
        "planned_1mb_maxiter": PLANNED_FINAL_ITERATIONS,
        "planned_1mb_maxfun": PLANNED_FINAL_MAXFUN,
        "lost_work_after_checkpoint": "unknown",
        "lbfgs_internal_history_available": False,
        "layers": layers,
        "candidate_coordinate_npz": str(npz_path.relative_to(run_dir)),
        "candidate_coordinate_npz_sha256": sha256_file(npz_path),
        "candidate_coordinate_3dg": str(text_path.relative_to(run_dir)),
        "candidate_coordinate_3dg_sha256": sha256_file(text_path),
        "final_finite": bool(main_record["candidate_finite"]),
        "final_max_radius": float(main_record["candidate_max_radius"]),
        "recovery_note": "Saved durable checkpoint nit50 finalized; this is not a solver terminal result.",
    }


def prepare_candidates(run_dir: str | Path) -> dict:
    """导出全部 main candidates，并在读取任何 truth 前写出 gate。"""
    run_dir = Path(run_dir).resolve()
    manifest = _verify_recovery_manifest(run_dir)
    gate_path = run_dir / "work" / "pretruth_gate.json"
    if gate_path.exists():
        raise RecoveryError("pre-truth gate already exists; refusing to overwrite")
    per_condition = {}
    for condition in CONDITION_IDS:
        data_path = _layer_path(run_dir, condition, FINAL_BIN)
        data = load_layer(data_path)
        data_audit = _audit_data(data)
        if data.n_loci != EXPECTED_FINAL_LOCI or len(data.track_specs) != N_TRACKS:
            raise RecoveryError("full grid/track count mismatch for %s" % condition)

        checkpoint_npz = _checkpoint_path(run_dir, condition, MAIN_ITERATION, "npz")
        checkpoint_json = _checkpoint_path(run_dir, condition, MAIN_ITERATION, "json")
        theta, checkpoint_y, checkpoint_coordinates, checkpoint_meta = _load_checkpoint(
            checkpoint_npz, checkpoint_json)
        objective = JointObjective(data)
        unpacked_y, q = objective.unpack(theta)
        y_match = bool(np.array_equal(unpacked_y, checkpoint_y))
        if not y_match:
            raise RecoveryError("checkpoint theta/y mismatch for %s" % condition)
        coordinates, p = objective.coordinates_and_p(theta)
        coordinate_difference = float(np.max(np.abs(coordinates - checkpoint_coordinates)))
        coordinate_exact = bool(np.array_equal(coordinates, checkpoint_coordinates))
        coordinate_pass = bool(coordinate_difference <= 1e-12)
        if not coordinate_pass:
            raise RecoveryError("checkpoint theta/coordinates mismatch for %s" % condition)

        recomputed_total, _, recomputed_components = objective.evaluate(theta, need_gradient=False)
        objective_fun_pass, objective_difference, objective_tolerance = _numeric_close(
            recomputed_total, checkpoint_meta["fun"])
        component_differences = {}
        component_pass = True
        for key, expected in checkpoint_meta["components"].items():
            if key not in recomputed_components:
                component_pass = False
                component_differences[key] = None
                continue
            actual = recomputed_components[key]
            if isinstance(actual, str) or isinstance(expected, str):
                equal = bool(actual == expected)
                component_differences[key] = {
                    "expected": expected,
                    "actual": actual,
                    "pass": equal,
                }
                component_pass = component_pass and equal
                continue
            difference = abs(float(actual) - float(expected))
            tolerance = 1e-10 * max(1.0, abs(float(actual)), abs(float(expected)))
            component_differences[key] = {
                "difference": difference,
                "tolerance": tolerance,
                "pass": bool(difference <= tolerance),
            }
            component_pass = component_pass and difference <= tolerance

        tracks = tracks_from_coordinates(data, coordinates)
        track_lengths = {name: len(track) for name, track in tracks.items()}
        expected_track_lengths = {
            spec.name: int(data.chromosome_slice(spec.chromosome_index).stop
                           - data.chromosome_slice(spec.chromosome_index).start)
            for spec in data.track_specs
        }
        full_track_pass = bool(
            len(tracks) == N_TRACKS
            and track_lengths == expected_track_lengths
            and sum(track_lengths.values()) == EXPECTED_PHYSICAL_BEADS)
        finite_pass = bool(np.all(np.isfinite(coordinates)))
        strict_ball_pass = bool(np.linalg.norm(coordinates, axis=2).max() < 1.0)
        assert_inside_unit_ball(coordinates)
        if not (full_track_pass and finite_pass and strict_ball_pass):
            raise RecoveryError("candidate grid/finite/unit-ball gate failed for %s" % condition)

        candidate_npz = run_dir / "coords" / (condition + "_1mb_theta.npz")
        candidate_text = run_dir / "coords" / (condition + "_1mb.3dg")
        np.savez_compressed(candidate_npz, theta=theta, coordinates=coordinates)
        write_full_tracks(candidate_text, data, coordinates)
        candidate_npz_hash = sha256_file(candidate_npz)
        candidate_text_hash = sha256_file(candidate_text)
        if candidate_npz_hash != sha256_file(candidate_npz) or candidate_text_hash != sha256_file(candidate_text):
            raise RecoveryError("candidate hash changed during write for %s" % condition)

        initial_event = _layer_event(condition, FINAL_BIN, "layer_initial")
        initial_total = float(initial_event["total"])
        nonincrease_tolerance = 1e-10 * max(1.0, abs(initial_total))
        nonincrease_pass = bool(recomputed_total <= initial_total + nonincrease_tolerance)
        source_input_hash = sha256_file(data_path)
        original_input = ORIGINAL_RUN / "work" / data_path.name
        input_hash_match = bool(source_input_hash == sha256_file(original_input))
        checkpoint_source_hash = sha256_file(checkpoint_npz)
        original_checkpoint = ORIGINAL_RUN / "checkpoints" / checkpoint_npz.name
        checkpoint_hash_match = bool(checkpoint_source_hash == sha256_file(original_checkpoint))
        checkpoint_iteration_pass = bool(
            checkpoint_meta.get("iteration") == MAIN_ITERATION
            and checkpoint_meta.get("nfev") == 51)
        p_match = abs(float(p) - float(checkpoint_meta["p"])) <= 1e-12
        gate = {
            "condition": condition,
            "main_analysis": True,
            "checkpoint_iteration": MAIN_ITERATION,
            "checkpoint_nfev_recorded": int(checkpoint_meta["nfev"]),
            "checkpoint_fun": float(checkpoint_meta["fun"]),
            "checkpoint_p": float(checkpoint_meta["p"]),
            "theta_y_exact_match": y_match,
            "theta_coordinates_exact_match": coordinate_exact,
            "theta_coordinates_max_abs_difference": coordinate_difference,
            "theta_coordinates_pass": coordinate_pass,
            "objective_recomputed_total": float(recomputed_total),
            "objective_checkpoint_difference": objective_difference,
            "objective_checkpoint_tolerance": objective_tolerance,
            "objective_recomputed_match": bool(objective_fun_pass and component_pass),
            "objective_component_differences": component_differences,
            "p_recomputed_match": bool(p_match),
            "full_grid_loci": int(data.n_loci),
            "full_physical_beads": int(sum(track_lengths.values())),
            "n_tracks": int(len(tracks)),
            "track_lengths": track_lengths,
            "full_track_grid_pass": full_track_pass,
            "candidate_finite": finite_pass,
            "candidate_strict_unit_ball": strict_ball_pass,
            "candidate_max_radius": float(np.linalg.norm(coordinates, axis=2).max()),
            "data_budget": data_audit,
            "source_input_hash": source_input_hash,
            "source_input_hash_matches_018": input_hash_match,
            "source_checkpoint_npz_hash": checkpoint_source_hash,
            "source_checkpoint_hash_matches_018": checkpoint_hash_match,
            "checkpoint_metadata_iteration_pass": checkpoint_iteration_pass,
            "original_layer_initial_total": initial_total,
            "recovered_checkpoint_total": float(recomputed_total),
            "final_not_worse_than_original_layer_initial": nonincrease_pass,
            "nonincrease_tolerance": nonincrease_tolerance,
            "candidate_coordinate_npz": str(candidate_npz.relative_to(run_dir)),
            "candidate_coordinate_npz_sha256": candidate_npz_hash,
            "candidate_coordinate_3dg": str(candidate_text.relative_to(run_dir)),
            "candidate_coordinate_3dg_sha256": candidate_text_hash,
            "r1": {"applicable": False, "reason": "synthetic record-copy labels were not generated"},
            "terminal": "interrupted_checkpoint_finalized",
            "solver_converged": False,
            "solver_convergence_status": "unknown",
            "lost_work_after_checkpoint": "unknown",
            "lbfgs_internal_history_available": False,
        }
        gate["all_pretruth_condition_gates_pass"] = bool(all([
            data_audit["all_pass"],
            y_match,
            coordinate_pass,
            objective_fun_pass,
            component_pass,
            p_match,
            full_track_pass,
            finite_pass,
            strict_ball_pass,
            input_hash_match,
            checkpoint_hash_match,
            checkpoint_iteration_pass,
            nonincrease_pass,
            bool(candidate_npz_hash),
            bool(candidate_text_hash),
        ]))
        per_condition[condition] = gate

        main_record = {
            "condition": condition,
            "bin_size": FINAL_BIN,
            "initial_total": initial_total,
            "resume_checkpoint_total": float(recomputed_total),
            "final_total": float(recomputed_total),
            "final_not_worse_than_initial": nonincrease_pass,
            "nonincrease_tolerance": nonincrease_tolerance,
            "optimizer_success": False,
            "solver_converged": False,
            "solver_convergence_status": "unknown",
            "termination": "interrupted_checkpoint_finalized",
            "nit": MAIN_ITERATION,
            "nit_semantics": "durable checkpoint accepted iterations, not solver terminal nit",
            "recorded_nfev": int(checkpoint_meta["nfev"]),
            "actual_nfev": "unknown",
            "scipy_nfev": "unknown",
            "elapsed_seconds": float(checkpoint_meta["elapsed_seconds"]),
            "p": float(p),
            "q": float(theta[-1]),
            "components": recomputed_components,
            "candidate_finite": finite_pass,
            "candidate_max_radius": float(np.linalg.norm(coordinates, axis=2).max()),
            "recovery_checkpoint": str(checkpoint_npz.relative_to(run_dir)),
            "recovery_checkpoint_json": str(checkpoint_json.relative_to(run_dir)),
            "recovery_checkpoint_sha256": checkpoint_source_hash,
            "lost_work_after_checkpoint": "unknown",
            "lbfgs_internal_history_available": False,
            "candidate_coordinate_hashes": {
                "npz": candidate_npz_hash,
                "3dg": candidate_text_hash,
            },
        }
        training = _candidate_training_record(run_dir, condition, main_record)
        _write_json(run_dir / "work" / (condition + "_training.json"), training)
        gate["training_json_sha256"] = sha256_file(run_dir / "work" / (condition + "_training.json"))

    all_pass = bool(all(row["all_pretruth_condition_gates_pass"] for row in per_condition.values()))
    pretruth = {
        "schema": "v1-p3-recovery-pretruth-gate-v1",
        "run_id": run_dir.name,
        "created_at_unix": time.time(),
        "truth_read_before_gate": False,
        "truth_read_authorized": all_pass,
        "main_checkpoint_iteration": MAIN_ITERATION,
        "planned_final_iteration_cap": PLANNED_FINAL_ITERATIONS,
        "terminal": "interrupted_checkpoint_finalized",
        "solver_converged": False,
        "solver_convergence_status": "unknown",
        "termination_cause": "unknown",
        "lost_work_after_checkpoint": "unknown",
        "lbfgs_internal_history_available": False,
        "all_pretruth_gates_pass": all_pass,
        "conditions": per_condition,
        "r1_applicable": False,
        "r1_reason": "synthetic record-copy labels were not generated",
        "source_snapshot_hashes": str((run_dir / "source_snapshot_hashes.json").relative_to(run_dir)),
        "recovery_manifest": manifest,
    }
    _write_json(gate_path, pretruth)
    _append_jsonl(run_dir / "logs" / "status.jsonl", {
        "event": "pretruth_candidate_gate_recorded",
        "conditions": list(CONDITION_IDS),
        "main_checkpoint_iteration": MAIN_ITERATION,
        "all_pretruth_gates_pass": all_pass,
        "truth_read_before_gate": False,
    })
    if not all_pass:
        raise RecoveryError("pre-truth candidate gate failed")
    return pretruth


def _load_verified_candidate(run_dir: Path, training: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / training["candidate_coordinate_npz"]
    expected_hash = training["candidate_coordinate_npz_sha256"]
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RecoveryError("candidate hash changed before truth evaluation: %s" % path)
    with np.load(path, allow_pickle=False) as payload:
        theta = payload["theta"].copy()
        coordinates = payload["coordinates"].copy()
    assert_inside_unit_ball(coordinates)
    return theta, coordinates


def _load_initial_5mb(run_dir: Path, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    path = run_dir / "inputs_snapshot" / (condition + "_independent_init_5mb.npz")
    with np.load(path, allow_pickle=False) as payload:
        coords = payload["coords"].copy()
        positions = payload["positions"].copy()
        chromosome_index = payload["chromosome_index"].copy()
        seed = int(payload["seed"])
    assert_inside_unit_ball(coords)
    return coords, positions, chromosome_index, seed


def _evaluate_condition(run_dir: Path, condition: str, training: Mapping[str, Any],
                        truth: np.ndarray, truth_exposure: np.ndarray,
                        truth_metadata: Mapping[str, Any]) -> dict:
    data = load_layer(_layer_path(run_dir, condition, FINAL_BIN))
    theta, candidate = _load_verified_candidate(run_dir, training)
    objective = JointObjective(data)
    _candidate_coordinates, candidate_p = objective.coordinates_and_p(theta)
    if not np.array_equal(_candidate_coordinates, candidate):
        raise RecoveryError("training coordinate payload differs from theta for %s" % condition)
    candidate_count = objective.count_components_for_coordinates(candidate, candidate_p)
    truth_count = objective.count_components_for_coordinates(truth, P_GENERATING)
    init_coords, positions, chromosome_index, init_seed = _load_initial_5mb(run_dir, condition)
    warm_initial = warm_start_from_layer(
        init_coords, positions, chromosome_index,
        data.chromosome_names, data.chromosome_lengths, FINAL_BIN,
        candidate_base_seed=init_seed,
    )["coords"]
    initial_objective = JointObjective(data)
    initial_count = initial_objective.count_components_for_coordinates(warm_initial, P_INITIAL)
    controls = _control_count_terms(data, truth, P_GENERATING)
    r2 = synthetic_r2(candidate, truth, data)
    r3 = synthetic_r3(candidate, truth, data)
    initial_r2 = synthetic_r2(warm_initial, truth, data)
    initial_r3 = synthetic_r3(warm_initial, truth, data)
    audit = _audit_data(data)
    all_layers_nonincrease = bool(all(
        layer["final_not_worse_than_initial"] for layer in training["layers"]))
    symmetry_pass = bool(
        controls["max_abs_per_chromosome_copy_swap_delta_per_record"] <= 1e-8)
    numeric_gates = {
        "all_layers_nonincreasing": all_layers_nonincrease,
        "candidate_finite": bool(np.all(np.isfinite(candidate))),
        "candidate_strict_unit_ball": bool(np.linalg.norm(candidate, axis=2).max() < 1.0),
        "candidate_full_grid": bool(candidate.shape == (2, EXPECTED_FINAL_LOCI, 3)),
        "input_budget": audit,
        "control_data_symmetry_pass": symmetry_pass,
        "noiseless_truth_control_pass": (
            controls["true_not_worse_than_fixed_controls"] if condition == "A" else None),
        "r1_applicable": False,
    }
    numeric_gates["all_numeric_gates_pass"] = bool(
        all_layers_nonincrease
        and numeric_gates["candidate_finite"]
        and numeric_gates["candidate_strict_unit_ball"]
        and numeric_gates["candidate_full_grid"]
        and audit["all_pass"]
        and symmetry_pass
        and (condition != "A" or numeric_gates["noiseless_truth_control_pass"])
    )
    result = {
        "condition": condition,
        "condition_name": FROZEN_CONDITIONS[condition]["name"],
        "main_analysis_checkpoint_iteration": MAIN_ITERATION,
        "terminal": "interrupted_checkpoint_finalized",
        "solver_converged": False,
        "solver_convergence_status": "unknown",
        "termination_cause": "unknown",
        "lost_work_after_checkpoint": "unknown",
        "lbfgs_internal_history_available": False,
        "truth_read_after_candidate_hash_gate": True,
        "candidate_coordinate_sha256": training["candidate_coordinate_npz_sha256"],
        "candidate_coordinate_3dg_sha256": training["candidate_coordinate_3dg_sha256"],
        "synthetic_truth_sha256": sha256_file(ORIGINAL_RUN / "eval_truth" / (condition + "_truth_1m.npz")),
        "numeric_gates": numeric_gates,
        "count_and_prior": {
            "candidate_count_nll_normalized": candidate_count["count_nll_normalized"],
            "truth_count_nll_normalized_at_p_gen": truth_count["count_nll_normalized"],
            "initial_count_nll_normalized_at_p_init": initial_count["count_nll_normalized"],
            "candidate_minus_truth_count_nll": (
                candidate_count["count_nll_normalized"] - truth_count["count_nll_normalized"]),
            "truth_minus_candidate_noise_gain_diagnostic": (
                truth_count["count_nll_normalized"] - candidate_count["count_nll_normalized"]),
            "candidate_final_components": training["layers"][-1]["components"],
            "truth_metadata": truth_metadata,
            "truth_exposure_shape": list(np.asarray(truth_exposure).shape),
        },
        "controls": controls,
        "r2": r2,
        "r3": r3,
        "initial_r2": initial_r2,
        "initial_r3": initial_r3,
        "same_shape_null_diagnostics": (
            _copy_shape_diagnostics(truth, data) if condition == "D" else None),
        "r1": {
            "applicable": False,
            "reason": "synthetic record-copy labels were not generated",
        },
        "interpretation_boundary": (
            "finite/count/budget checks are implementation calibration only; "
            "R2/R3 and count gain do not establish biological L2 recovery"
        ),
    }
    _write_json(run_dir / "work" / (condition + "_evaluation.json"), result)
    return result


def _evaluate_later_diagnostics(run_dir: Path, truth_cache: Mapping[str, tuple[np.ndarray, np.ndarray, dict]]) -> dict:
    diagnostics = {}
    for condition in ("A", "B", "D"):
        cp_npz = _later_checkpoint_path(run_dir, condition, 70, "npz")
        cp_json = _later_checkpoint_path(run_dir, condition, 70, "json")
        theta, checkpoint_y, checkpoint_coordinates, metadata = _load_checkpoint(cp_npz, cp_json)
        data = load_layer(_layer_path(run_dir, condition, FINAL_BIN))
        objective = JointObjective(data)
        y, _q = objective.unpack(theta)
        coords, p = objective.coordinates_and_p(theta)
        if not np.array_equal(y, checkpoint_y) or not np.array_equal(coords, checkpoint_coordinates):
            raise RecoveryError("later diagnostic theta/coordinate mismatch for %s" % condition)
        truth, _exposure, _truth_meta = truth_cache[condition]
        count = objective.count_components_for_coordinates(coords, p)
        diagnostics[condition] = {
            "condition": condition,
            "role": "later_diagnostic_only",
            "checkpoint_iteration": 70,
            "checkpoint_nfev_recorded": int(metadata["nfev"]),
            "checkpoint_fun": float(metadata["fun"]),
            "p": float(p),
            "candidate_count_nll_normalized": count["count_nll_normalized"],
            "r2": synthetic_r2(coords, truth, data),
            "r3": synthetic_r3(coords, truth, data),
            "terminal": "interrupted_checkpoint_diagnostic",
            "solver_converged": False,
            "solver_convergence_status": "unknown",
            "not_used_for_main_B_vs_C_comparison": True,
            "checkpoint_sha256": sha256_file(cp_npz),
        }
        _write_json(run_dir / "work" / (condition + "_iter0070_later_diagnostic.json"), diagnostics[condition])
    diagnostics["C"] = {
        "condition": "C",
        "role": "later_diagnostic_only",
        "checkpoint_iteration": None,
        "reason": "no durable C iter70 checkpoint exists; C main analysis remains iter50",
        "not_used_for_main_B_vs_C_comparison": True,
    }
    _write_json(run_dir / "work" / "later_diagnostic_summary.json", diagnostics)
    return diagnostics


def _write_summary_tsv(run_dir: Path, metrics: Mapping[str, Any]) -> Path:
    path = run_dir / "plots" / "calibration_summary.tsv"
    with open(path, "wt") as handle:
        handle.write(
            "condition\tcount_mode\tcheckpoint_iteration\tterminal\tfinal_total\t"
            "candidate_minus_truth_count_nll\tr2_mean_contrast\tr3_n_applicable\t"
            "r3_mean_frac_consistent\tall_numeric_gates_pass\n")
        for condition in CONDITION_IDS:
            result = metrics["conditions"][condition]
            final_components = result["count_and_prior"]["candidate_final_components"]
            r2_value = result["r2"]["mean_contrast"]
            r3_value = result["r3"]["mean_frac_consistent"]
            handle.write("%s\t%s\t%d\t%s\t%.17g\t%.17g\t%s\t%d\t%s\t%s\n" % (
                condition,
                FROZEN_CONDITIONS[condition]["count_mode"],
                MAIN_ITERATION,
                result["terminal"],
                final_components["total"],
                result["count_and_prior"]["candidate_minus_truth_count_nll"],
                "" if r2_value is None else "%.17g" % r2_value,
                result["r3"]["n_applicable"],
                "" if r3_value is None else "%.17g" % r3_value,
                result["numeric_gates"]["all_numeric_gates_pass"],
            ))
    return path


def _write_later_summary_tsv(run_dir: Path, diagnostics: Mapping[str, Any]) -> Path:
    path = run_dir / "plots" / "later_diagnostic_summary.tsv"
    with open(path, "wt") as handle:
        handle.write("condition\tcheckpoint_iteration\trole\tr2_mean_contrast\tr3_n_applicable\tr3_mean_frac_consistent\n")
        for condition in ("A", "B", "D"):
            result = diagnostics[condition]
            handle.write("%s\t%d\t%s\t%s\t%d\t%s\n" % (
                condition,
                result["checkpoint_iteration"],
                result["role"],
                "" if result["r2"]["mean_contrast"] is None else "%.17g" % result["r2"]["mean_contrast"],
                result["r3"]["n_applicable"],
                "" if result["r3"]["mean_frac_consistent"] is None else "%.17g" % result["r3"]["mean_frac_consistent"],
            ))
    return path


def _write_plots(run_dir: Path, metrics: Mapping[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(CONDITION_IDS)
    colors = {"A": "#2f6690", "B": "#3a7d44", "C": "#b5651d", "D": "#6c567b"}
    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })
    paths = []
    gains = [metrics["conditions"][c]["count_and_prior"]["candidate_minus_truth_count_nll"] for c in conditions]
    fig, ax = plt.subplots(figsize=(3.0, 2.25), dpi=300)
    ax.bar(conditions, gains, color=[colors[c] for c in conditions], width=0.68)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("condition")
    ax.set_ylabel("candidate - truth count NLL")
    ax.set_title("P3 recovery: common iter50")
    fig.tight_layout(pad=0.5)
    path = run_dir / "plots" / "count_nll_gain.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(str(path.relative_to(run_dir)))

    r2 = [metrics["conditions"][c]["r2"]["mean_contrast"] for c in conditions]
    r3 = [metrics["conditions"][c]["r3"]["mean_frac_consistent"] for c in conditions]
    fig, axes = plt.subplots(1, 2, figsize=(3.0, 1.8), dpi=300)
    axes[0].bar(conditions, [0.0 if value is None else value for value in r2],
                color=[colors[c] for c in conditions], width=0.68)
    axes[0].axhline(0.0, color="black", linewidth=0.6)
    axes[0].set_title("R2 geometry contrast")
    axes[0].set_xlabel("condition")
    axes[0].set_ylabel("direct/swap")
    axes[1].bar(conditions, [0.0 if value is None else value for value in r3],
                color=[colors[c] for c in conditions], width=0.68)
    axes[1].set_title("R3 fragment consistency")
    axes[1].set_xlabel("condition")
    axes[1].set_ylabel("fraction")
    fig.tight_layout(pad=0.5, w_pad=0.8)
    path = run_dir / "plots" / "r2_r3_summary.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(str(path.relative_to(run_dir)))
    return paths


def evaluate_run(run_dir: str | Path) -> dict:
    """仅在每个 candidate hash 和 pre-truth gate 通过后读取 truth。"""
    run_dir = Path(run_dir).resolve()
    manifest = _verify_recovery_manifest(run_dir)
    gate = _load_json(run_dir / "work" / "pretruth_gate.json")
    if not gate.get("all_pretruth_gates_pass") or not gate.get("truth_read_authorized"):
        raise RecoveryError("truth read is not authorized by the recorded pre-truth gate")

    # 在打开 eval_truth 前核验每个 candidate 及每个记录的 hash。
    trainings = {}
    candidates = {}
    for condition in CONDITION_IDS:
        training = _load_json(run_dir / "work" / (condition + "_training.json"))
        if training.get("status") != "completed" or training.get("terminal") != "interrupted_checkpoint_finalized":
            raise RecoveryError("incomplete recovery artifact for %s" % condition)
        theta, coordinates = _load_verified_candidate(run_dir, training)
        text_path = run_dir / training["candidate_coordinate_3dg"]
        if sha256_file(text_path) != training["candidate_coordinate_3dg_sha256"]:
            raise RecoveryError("candidate 3dg hash changed before truth evaluation: %s" % text_path)
        trainings[condition] = training
        candidates[condition] = (theta, coordinates)
    truth_read_authorized = True

    # 这是首次访问原始的隔离 synthetic truth。
    truth_cache = {}
    for condition in CONDITION_IDS:
        truth_cache[condition] = _load_truth(
            ORIGINAL_RUN / "eval_truth" / (condition + "_truth_1m.npz"))
    results = {}
    for condition in CONDITION_IDS:
        truth, exposure, metadata = truth_cache[condition]
        results[condition] = _evaluate_condition(
            run_dir, condition, trainings[condition], truth, exposure, metadata)

    later_diagnostics = _evaluate_later_diagnostics(run_dir, truth_cache)
    main_comparison = {
        "B_checkpoint_iteration": MAIN_ITERATION,
        "C_checkpoint_iteration": MAIN_ITERATION,
        "same_matched_budget": True,
        "B_candidate_minus_truth_count_nll": results["B"]["count_and_prior"]["candidate_minus_truth_count_nll"],
        "C_candidate_minus_truth_count_nll": results["C"]["count_and_prior"]["candidate_minus_truth_count_nll"],
        "interpretation": "joint kernel-plus-exposure misspecification sensitivity; not an attribution to either factor",
        "A_B_D_iter70_excluded": True,
    }
    metrics = {
        "schema": "v1-p3-synthetic-calibration-recovery-metrics-v1",
        "run_id": run_dir.name,
        "frozen_manifest": manifest,
        "truth_read_after_candidate_hash_gate": truth_read_authorized,
        "truth_read_before_pretruth_gate": False,
        "main_analysis_checkpoint_iteration": MAIN_ITERATION,
        "planned_1mb_iteration_cap": PLANNED_FINAL_ITERATIONS,
        "terminal": "interrupted_checkpoint_finalized",
        "solver_converged": False,
        "solver_convergence_status": "unknown",
        "termination_cause": "unknown",
        "lost_work_after_checkpoint": "unknown",
        "lbfgs_internal_history_available": False,
        "conditions": results,
        "later_diagnostics": later_diagnostics,
        "main_B_vs_C_comparison": main_comparison,
        "global_gates": {
            "all_conditions_numeric_gates_pass": bool(all(
                result["numeric_gates"]["all_numeric_gates_pass"] for result in results.values())),
            "A_truth_control_pass": bool(results["A"]["numeric_gates"]["noiseless_truth_control_pass"]),
            "R1_applicable": False,
            "R1_reason": "synthetic record-copy labels were not generated",
            "R2_R3_are_calibration_diagnostics": True,
            "no_L2_claim": True,
        },
        "biological_replicates": 1,
        "calibration_conditions_are_not_biological_replicates": True,
    }
    _write_json(run_dir / "metrics.json", metrics)
    summary_path = _write_summary_tsv(run_dir, metrics)
    later_summary_path = _write_later_summary_tsv(run_dir, later_diagnostics)
    plot_paths = _write_plots(run_dir, metrics)
    metrics["summary_tsv"] = str(summary_path.relative_to(run_dir))
    metrics["later_summary_tsv"] = str(later_summary_path.relative_to(run_dir))
    metrics["plot_paths"] = plot_paths
    _write_json(run_dir / "metrics.json", metrics)
    _append_jsonl(run_dir / "logs" / "status.jsonl", {
        "event": "synthetic_recovery_evaluation_completed",
        "conditions": list(CONDITION_IDS),
        "main_checkpoint_iteration": MAIN_ITERATION,
        "truth_read_after_candidate_hash_gate": True,
        "all_numeric_gates_pass": metrics["global_gates"]["all_conditions_numeric_gates_pass"],
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 durable checkpoint recovery adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "prepare", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if args.command == "init":
        result = initialize_run(run_dir)
    elif args.command == "prepare":
        result = prepare_candidates(run_dir)
    else:
        result = evaluate_run(run_dir)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
