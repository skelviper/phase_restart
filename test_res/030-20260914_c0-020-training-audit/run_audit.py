#!/usr/bin/env python
"""对 020 与 029 C0 执行无 reference 的固定状态审计。

本脚本只读取明确 allow-list 中的 SNP-free inputs、training-side checkpoints/coordinates、native preflight outputs 和 source snapshots。它不导入 reference/evaluation 侧，也不调用 optimizer。
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import difflib
import hashlib
import importlib
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import struct
import sys
import types
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "test_res" / "030-20260914_c0-020-training-audit"
INPUT = ROOT / "inputs" / "P9016.snpfree.pairs.gz"
OLD_PR = AUDIT.parent / "020-20260913_071841-v1-p9016-joint" / "provenance" / "training-code" / "pr"
RUN020 = ROOT / "test_res" / "020-20260913_071841-v1-p9016-joint"
RUN022 = ROOT / "test_res" / "022-20260913_111031-v1-continuation-fdg-r2"
RUN025 = ROOT / "test_res" / "025-20260913_135100-random-native-fullgrid-preflight"
RUN029 = ROOT / "test_res" / "029-20260913_161713-post020-allele-ablation-real"
CURRENT_PR = ROOT / "pr"

# 以下路径均不得包含任何 evaluation/reference payload。
FORBIDDEN_PATH_MARKERS = (
    "evaluation-r2",
    "evaluation",
    "reference",
    "phase",
)
FORBIDDEN_EXACT_FILENAMES = (
    "P9016.1m.3dg.gz",
    "P9016.pairs.gz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray, dtype: str = "<f8") -> str:
    array = np.asarray(values, dtype=dtype, order="C")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def is_forbidden_path(path: Path) -> bool:
    return any(marker in path.parts for marker in FORBIDDEN_PATH_MARKERS) or path.name in FORBIDDEN_EXACT_FILENAMES


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def import_isolated_package(alias: str, package_dir: Path) -> types.ModuleType:
    package = types.ModuleType(alias)
    package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    package.__package__ = alias
    package.__spec__ = ModuleSpec(alias, loader=None, is_package=True)
    sys.modules[alias] = package
    return package


def component_numeric_diff(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, str | None]:
    maximum = 0.0
    key_at_max = None
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
            delta = abs(float(a) - float(b))
            if delta > maximum:
                maximum, key_at_max = delta, key
    return maximum, key_at_max


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def parse_3dg(path: Path, data: Any) -> dict[str, Any]:
    expected = {spec.name: spec for spec in data.track_specs}
    coordinates = np.full((2, data.n_loci, 3), np.nan, dtype=np.float64)
    seen = np.zeros((2, data.n_loci), dtype=bool)
    order: list[str] = []
    rows = 0
    with path.open("rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_no}: expected five columns")
            name, position_text, *xyz_text = fields
            if name not in expected:
                raise ValueError(f"{path}:{line_no}: unexpected track {name}")
            spec = expected[name]
            position = int(position_text)
            if position < 0 or position % int(data.bin_size) != 0:
                raise ValueError(f"{path}:{line_no}: invalid origin-0 position {position}")
            local_bin = position // int(data.bin_size)
            global_index = int(data.offsets[spec.chromosome_index] + local_bin)
            slc = data.chromosome_slice(spec.chromosome_index)
            if global_index < slc.start or global_index >= slc.stop:
                raise ValueError(f"{path}:{line_no}: position outside chromosome grid")
            if seen[spec.copy_index, global_index]:
                raise ValueError(f"{path}:{line_no}: duplicate coordinate")
            xyz = np.asarray([float(v) for v in xyz_text], dtype=np.float64)
            if not np.isfinite(xyz).all():
                raise ValueError(f"{path}:{line_no}: nonfinite coordinate")
            coordinates[spec.copy_index, global_index] = xyz
            seen[spec.copy_index, global_index] = True
            if not order or order[-1] != name:
                order.append(name)
            rows += 1
    expected_order = [spec.name for spec in data.track_specs]
    complete = bool(seen.all())
    if not complete:
        raise ValueError(f"{path}: incomplete coordinate grid ({int(seen.sum())}/{seen.size})")
    return {
        "coordinates": coordinates,
        "rows": rows,
        "complete": complete,
        "track_order": order,
        "expected_track_order": expected_order,
        "track_order_ok": order == expected_order,
        "file_sha256": sha256_file(path),
    }


def endpoint_from_npz(path: Path) -> dict[str, Any]:
    payload = load_npz(path)
    theta = np.asarray(payload["theta"], dtype=np.float64)
    if theta.ndim != 1 or theta.size < 1:
        raise ValueError(f"{path}: unexpected theta shape {theta.shape}")
    if "coordinates" not in payload:
        raise ValueError(f"{path}: endpoint lacks coordinates")
    coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[0] != 2 or coordinates.shape[2] != 3:
        raise ValueError(f"{path}: unexpected coordinate shape {coordinates.shape}")
    expected_size = 6 * int(coordinates.shape[1]) + 1
    if theta.size != expected_size:
        raise ValueError(f"{path}: theta size {theta.size} != {expected_size}")
    if "raw_coordinates" in payload:
        raw = np.asarray(payload["raw_coordinates"], dtype=np.float64)
    elif "y" in payload:
        raw = np.asarray(payload["y"], dtype=np.float64)
    else:
        raw = theta[:-1].reshape(coordinates.shape)
    if raw.shape != coordinates.shape:
        raise ValueError(f"{path}: raw shape {raw.shape} != coordinates {coordinates.shape}")
    return {"path": path, "payload": payload, "theta": theta, "raw": raw, "coordinates": coordinates}


def freeze_allow_list() -> tuple[dict[str, Any], dict[str, Path]]:
    stage_paths: list[Path] = []
    stage_npz: list[Path] = []
    for candidate in ("consensus_joint", "random_joint"):
        for stage in ("5m", "2m", "1m"):
            record = RUN020 / "stages" / candidate / f"{stage}.json"
            stage_paths.append(record)
            stage_npz.append(RUN020 / read_json(record)["fit"]["checkpoint_paths"][-1])
    paths: dict[str, Path] = {
        "input": INPUT,
        "020_config": RUN020 / "config.json",
        "020_input_version": RUN020 / "input_version.json",
        "020_selection": RUN020 / "selection.json",
        "020_termination": RUN020 / "termination_audit.json",
        "020_provenance_manifest": RUN020 / "provenance" / "training-code-manifest.json",
        "020_protocol_manifest": RUN020 / "provenance" / "protocol-manifest.json",
        "022_final_components": RUN022 / "final_components.json",
        "022_continuation_summary": RUN022 / "continuation_summary.json",
        "022_terminal_audit": RUN022 / "terminal_audit_independent.json",
        "022_progress": RUN022 / "logs" / "progress.jsonl",
        "022_final_theta": RUN022 / "theta" / "final-theta.npz",
        "029_config": RUN029 / "config.json",
        "029_config_snapshot": RUN029 / "config_snapshot.json",
        "029_input_manifest": RUN029 / "input_manifest.json",
        "029_x0_manifest": RUN029 / "x0_manifest.json",
        "029_source_hashes": RUN029 / "source_hashes.json",
        "029_selection": RUN029 / "selection.json",
        "029_termination": RUN029 / "termination_audit.json",
        "025_config": RUN025 / "config.json",
        "025_validation": RUN025 / "validation" / "POST020_ALLELE_ABLATION_NATIVE_VALIDATION.json",
        "025_grid": RUN025 / "grid" / "fullgrid.npz",
        "old_contact_model": OLD_PR / "contact_model.py",
        "old_genome": OLD_PR / "genome.py",
        "old_pairs7": OLD_PR / "pairs7.py",
        "old_paths": OLD_PR / "paths.py",
        "old_joint_fit": OLD_PR / "joint_fit.py",
        "old_reconstruct": OLD_PR / "reconstruct.py",
        "old_reconstruction_init": OLD_PR / "reconstruction_init.py",
        "current_contact_model": CURRENT_PR / "contact_model.py",
        "current_genome": CURRENT_PR / "genome.py",
        "current_pairs7": CURRENT_PR / "pairs7.py",
        "current_paths": CURRENT_PR / "paths.py",
        "current_joint_fit": CURRENT_PR / "joint_fit.py",
        "current_allele_models": CURRENT_PR / "allele_models.py",
        "current_paired_run": CURRENT_PR / "paired_run.py",
        "current_allele_experiment": CURRENT_PR / "allele_experiment.py",
        "current_init": CURRENT_PR / "__init__.py",
        "020_selected_3dg": RUN020 / "selected.3dg",
        "020_selected_npz": RUN020 / "checkpoints" / "random_joint" / "1m-accepted-0240.npz",
        "029_C0_bundle2_npz": RUN029 / "jobs" / "C0-bundle2" / "attempts" / "attempt-001" / "checkpoints" / "iter-000480.npz",
        "029_C0_bundle2_3dg": RUN029 / "jobs" / "C0-bundle2" / "attempts" / "attempt-001" / "final_coordinates.3dg",
    }
    for key, path in zip(("020_consensus_5m_npz", "020_consensus_2m_npz", "020_consensus_1m_npz", "020_random_5m_npz", "020_random_2m_npz", "020_random_1m_npz"), [stage_npz[0], stage_npz[1], stage_npz[2], stage_npz[3], stage_npz[4], stage_npz[5]], strict=True):
        paths[key] = path
    for bundle in ("bundle1", "bundle2", "bundle3"):
        paths[f"025_{bundle}_raw_output"] = RUN025 / "bundles" / bundle / "native_output.rndout"
        paths[f"025_{bundle}_raw_npz"] = RUN025 / "bundles" / bundle / "raw_native.npz"
        paths[f"025_{bundle}_raw_3dg"] = RUN025 / "bundles" / bundle / "raw_native.3dg"
        paths[f"025_{bundle}_native_init_3dg"] = RUN025 / "bundles" / bundle / "native_init.3dg"
        paths[f"025_{bundle}_x0_npz"] = RUN025 / "bundles" / bundle / "x0_normalized.npz"
        paths[f"025_{bundle}_x0_3dg"] = RUN025 / "bundles" / bundle / "x0_normalized.3dg"
        paths[f"029_{bundle}_x0_npz"] = RUN029 / "initialization" / bundle / "x0_normalized.npz"
        paths[f"029_C0_{bundle}_final"] = RUN029 / "jobs" / f"C0-{bundle}" / "attempts" / "attempt-001" / "final.json"
        paths[f"029_C0_{bundle}_status"] = RUN029 / "jobs" / f"C0-{bundle}" / "status.json"
        paths[f"029_C0_{bundle}_job"] = RUN029 / "jobs" / f"C0-{bundle}" / "job.json"
        paths[f"029_C0_{bundle}_npz"] = RUN029 / "jobs" / f"C0-{bundle}" / "attempts" / "attempt-001" / "checkpoints" / "iter-000480.npz"
        paths[f"029_C0_{bundle}_3dg"] = RUN029 / "jobs" / f"C0-{bundle}" / "attempts" / "attempt-001" / "final_coordinates.3dg"
    # 在解析其 final checkpoint paths 后加入 stage records。
    for index, path in enumerate(stage_paths, start=1):
        paths[f"020_stage_record_{index:02d}"] = path
    for path in paths.values():
        if is_forbidden_path(path):
            raise AssertionError(f"forbidden path entered allow-list: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    files = [{"key": key, "path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for key, path in sorted(paths.items())]
    payload = {
        "schema": "reference-free-training-audit-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "working_root": str(ROOT),
        "input_boundary": {
            "phase_free_input": rel(INPUT),
            "reference_opened": False,
            "phase_opened": False,
            "evaluation_outputs_opened": False,
            "optimizer_called": False,
            "native_fdg_called": False,
        },
        "authorized_file_count": len(files),
        "files": files,
        "environment_contract": {
            "conda": "analysis",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "script_sha256": sha256_file(AUDIT / "run_audit.py"),
    }
    payload["freeze_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    write_json(AUDIT / "frozen_inputs.json", payload)
    return payload, paths


def native_output_audit(raw_output_path: Path, raw_npz_path: Path, x0_npz_path: Path, x0_3dg_path: Path, data: Any, parse_3dg_fn: Any) -> dict[str, Any]:
    raw = load_npz(raw_npz_path)
    x0 = load_npz(x0_npz_path)
    header = struct.Struct("<8s6I4f")
    binary = raw_output_path.read_bytes()
    if len(binary) < header.size:
        raise ValueError(f"truncated native output {raw_output_path}")
    magic, version, n_beads, n_binned, n_raw, n_iter, flags, native_unit, init_max, final_max, reserved = header.unpack_from(binary)
    n_values = int(n_beads) * 3
    expected_bytes = header.size + 2 * n_values * 4
    if len(binary) != expected_bytes:
        raise ValueError(f"native output byte size mismatch: {len(binary)} != {expected_bytes}")
    values = np.frombuffer(binary, dtype="<f4", count=2 * n_values, offset=header.size).copy()
    init_bead = values[:n_values].reshape(n_beads, 3).astype(np.float64)
    final_bead = values[n_values:].reshape(n_beads, 3).astype(np.float64)
    track_offsets = np.concatenate(([0], np.cumsum(np.repeat(data.n_bins, 2))))
    def bead_to_copyfirst(beads: np.ndarray) -> np.ndarray:
        result = np.empty((2, data.n_loci, 3), dtype=np.float64)
        for chromosome in range(len(data.chromosome_names)):
            slc = data.chromosome_slice(chromosome)
            for copy in (0, 1):
                track = 2 * chromosome + copy
                result[copy, slc] = beads[track_offsets[track]:track_offsets[track + 1]]
        return result
    init_cf = bead_to_copyfirst(init_bead)
    final_cf = bead_to_copyfirst(final_bead)
    normalized_bead = np.asarray(x0["bead_order"], dtype=np.float64)
    normalized_cf = np.asarray(x0["coordinates"], dtype=np.float64)
    center = final_bead.mean(axis=0)
    centered = final_bead - center
    raw_radius = float(np.linalg.norm(centered, axis=1).max())
    scale = 0.8 / raw_radius
    expected_x0_bead = centered * scale
    expected_x0_cf = bead_to_copyfirst(expected_x0_bead)
    x0_text = parse_3dg_fn(x0_3dg_path, data)
    return {
        "raw_output": {
            "magic_ascii": magic.rstrip(b"\\0").decode("ascii", errors="replace"),
            "version": int(version), "n_beads": int(n_beads), "n_binned_pairs": int(n_binned),
            "n_raw": int(n_raw), "n_iter": int(n_iter), "flags": int(flags),
            "native_unit": float(native_unit), "header_init_max_norm": float(init_max),
            "header_final_max_norm": float(final_max), "reserved": int(reserved),
            "byte_size": len(binary), "sha256": sha256_file(raw_output_path),
            "raw_npz_init_bead_max_abs": float(np.max(np.abs(init_bead - raw["native_init_bead_order"]))),
            "raw_npz_final_bead_max_abs": float(np.max(np.abs(final_bead - raw["native_final_bead_order"]))),
        },
        "grid_mapping": {
            "track_order": [spec.name for spec in data.track_specs],
            "track_offsets": [int(v) for v in track_offsets],
            "init_bead_to_copyfirst_max_abs": float(np.max(np.abs(init_cf - raw["native_init_coordinates"]))),
            "final_bead_to_copyfirst_max_abs": float(np.max(np.abs(final_cf - raw["native_final_coordinates"]))),
            "init_shape": list(init_cf.shape), "final_shape": list(final_cf.shape),
        },
        "normalization": {
            "center": center.tolist(), "raw_max_radius": raw_radius, "scale": float(scale),
            "stored_center_max_abs": float(np.max(np.abs(center - x0["center"]))),
            "stored_scale_abs": abs(float(scale) - float(x0["scale"].ravel()[0])),
            "expected_x0_bead_max_abs": float(np.max(np.abs(expected_x0_bead - normalized_bead))),
            "expected_x0_copyfirst_max_abs": float(np.max(np.abs(expected_x0_cf - normalized_cf))),
            "stored_target_max_radius": float(x0["target_max_radius"].ravel()[0]),
            "normalized_max_radius": float(np.linalg.norm(normalized_bead, axis=1).max()),
        },
        "x0_text_roundtrip": {
            "file_sha256": x0_text["file_sha256"], "rows": x0_text["rows"],
            "track_order_ok": x0_text["track_order_ok"],
            "parsed_vs_npz_max_abs": float(np.max(np.abs(x0_text["coordinates"] - normalized_cf))),
        },
        "finite": bool(np.isfinite(init_bead).all() and np.isfinite(final_bead).all() and np.isfinite(normalized_cf).all()),
        "historical_postprocess_bug": {
            "recorded": True,
            "description": "old normalize_x0 returned bead-order (5290,3) but indexed it as (copy,locus,xyz), causing shape (0,) to (183,3) broadcast",
            "recovered_without_native_rerun": True,
            "correct_mapping": "track-major bead order -> (copy, global_locus, xyz) using track=2*chromosome+copy",
        },
    }


def evaluate_fixed(obj: Any, theta: np.ndarray) -> dict[str, Any]:
    value, gradient, components = obj.evaluate(theta, need_gradient=True)
    gradient = np.asarray(gradient, dtype=np.float64)
    return {
        "value": float(value),
        "gradient_l2": float(np.linalg.norm(gradient)),
        "gradient_inf": float(np.max(np.abs(gradient))),
        "gradient_sha256": sha256_array(gradient),
        "gradient": gradient,
        "components": dict(components),
    }


def compare_fixed(old_obj: Any, new_obj: Any, theta: np.ndarray, state_coordinates: np.ndarray, state_name: str) -> dict[str, Any]:
    old_eval = evaluate_fixed(old_obj, theta)
    new_eval = evaluate_fixed(new_obj, theta)
    component_diff, component_key = component_numeric_diff(old_eval["components"], new_eval["components"])
    gradient_diff = float(np.max(np.abs(old_eval["gradient"] - new_eval["gradient"])))
    old_x, old_p = old_obj.coordinates_and_p(theta)
    new_x, new_p = new_obj.coordinates_and_p(theta)
    return {
        "state": state_name,
        "theta_size": int(theta.size),
        "saved_coordinate_max_radius": float(np.linalg.norm(state_coordinates, axis=2).max()),
        "saved_coordinate_finite": bool(np.isfinite(state_coordinates).all()),
        "old": {k: v for k, v in old_eval.items() if k not in ("gradient",)},
        "new_c0": {k: v for k, v in new_eval.items() if k not in ("gradient",)},
        "old_new_value_abs_diff": abs(old_eval["value"] - new_eval["value"]),
        "old_new_gradient_max_abs_diff": gradient_diff,
        "old_new_component_max_abs_diff": component_diff,
        "old_new_component_key_at_max": component_key,
        "old_saved_coordinate_max_abs_diff": float(np.max(np.abs(old_x - state_coordinates))),
        "new_saved_coordinate_max_abs_diff": float(np.max(np.abs(new_x - state_coordinates))),
        "old_new_coordinate_max_abs_diff": float(np.max(np.abs(old_x - new_x))),
        "old_p": float(old_p), "new_p": float(new_p),
        "p_abs_diff": abs(float(old_p) - float(new_p)),
    }


def trace_row(run_id: str, source_kind: str, entry: dict[str, Any], max_radius: float | None = None, cumulative_iteration: Any = None, cumulative_nfev: Any = None) -> dict[str, Any]:
    components = entry.get("components", entry)
    return {
        "run_id": run_id, "source_kind": source_kind,
        "iteration": entry.get("iteration"), "cumulative_iteration": cumulative_iteration,
        "nfev": entry.get("nfev", entry.get("actual_nfev")), "cumulative_nfev": cumulative_nfev,
        "elapsed_seconds": entry.get("elapsed_seconds"), "fun": entry.get("fun", components.get("total")),
        "count_nll_normalized": components.get("count_nll_normalized"), "p": components.get("p", entry.get("p")),
        "p_prior": components.get("p_prior"), "bond": components.get("bond"),
        "bend": components.get("bend"), "repulsion": components.get("repulsion"),
        "gradient_l2": entry.get("gradient_norm", entry.get("gradient_l2")),
    }


def stage_endpoint_rows(old_cm: Any, data_by_bin: dict[int, Any], stage_defs: list[dict[str, Any]], history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in stage_defs:
        record = item["record"]
        final = record["fit"]["final_components"]
        npz = endpoint_from_npz(item["npz"])
        data = data_by_bin[int(record["bin_size_bp"])]
        obj = old_cm.JointObjective(data, block_size=65_536, repulsion_block_size=65_536)
        fixed = evaluate_fixed(obj, npz["theta"])
        max_radius = float(np.linalg.norm(npz["coordinates"], axis=2).max())
        diff, diff_key = component_numeric_diff(final, fixed["components"])
        history = record["fit"].get("history", [])
        for entry in history:
            history_rows.append(trace_row(item["run_id"], "020_stage_record", entry, max_radius=max_radius))
        termination = record["fit"].get("solver", record["fit"].get("termination", {}))
        rows.append({
            "run_id": item["run_id"], "source_kind": "020", "candidate": record["candidate_id"],
            "start_or_stage": record["bin_size_bp"], "resolution_bp": record["bin_size_bp"],
            "p": final.get("p"), "count_nll_normalized": final.get("count_nll_normalized"),
            "total": final.get("total"), "bond": final.get("bond"), "bend": final.get("bend"),
            "repulsion": final.get("repulsion"), "p_prior": final.get("p_prior"),
            "gradient_l2": fixed["gradient_l2"], "gradient_inf": fixed["gradient_inf"],
            "max_radius": max_radius, "nit": termination.get("nit"), "nfev": termination.get("actual_nfev", termination.get("nfev")),
            "budget": f"maxiter={termination.get('maxiter')};maxfun={termination.get('maxfun')}",
            "termination": record["fit"].get("termination_reason", termination.get("message", record.get("status"))),
            "status": record.get("status", record["fit"].get("status")),
            "recorded_recompute_component_max_abs_diff": diff,
            "recorded_recompute_component_key_at_max": diff_key,
            "coordinate_sha256": sha256_array(npz["coordinates"]),
        })
    return rows


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    freeze, paths = freeze_allow_list()

    # 以互不相同的名称导入 archived 和 current packages。两个 package 都不导入 reference/evaluation code；只执行 contact aggregation。
    old_pkg = import_isolated_package("audit_old_pr", OLD_PR)
    new_pkg = import_isolated_package("audit_new_pr", CURRENT_PR)
    old_cm = importlib.import_module("audit_old_pr.contact_model")
    new_cm = importlib.import_module("audit_new_pr.contact_model")
    new_allele = importlib.import_module("audit_new_pr.allele_models")
    new_paired = importlib.import_module("audit_new_pr.paired_run")

    source_comparison: dict[str, Any] = {
        "contact_model_byte_identical": paths["old_contact_model"].read_bytes() == paths["current_contact_model"].read_bytes(),
        "old_contact_model_sha256": sha256_file(paths["old_contact_model"]),
        "current_contact_model_sha256": sha256_file(paths["current_contact_model"]),
        "old_joint_fit_sha256": sha256_file(paths["old_joint_fit"]),
        "current_joint_fit_sha256": sha256_file(paths["current_joint_fit"]),
    }
    old_joint_text = paths["old_joint_fit"].read_text(encoding="utf-8").splitlines(keepends=True)
    current_joint_text = paths["current_joint_fit"].read_text(encoding="utf-8").splitlines(keepends=True)
    joint_diff = list(difflib.unified_diff(old_joint_text, current_joint_text, fromfile=rel(paths["old_joint_fit"]), tofile=rel(paths["current_joint_fit"])))
    (AUDIT / "joint_fit.diff").write_text("".join(joint_diff), encoding="utf-8")
    source_comparison.update({
        "joint_fit_unified_diff_lines": len(joint_diff),
        "joint_fit_added_lines": sum(line.startswith("+") and not line.startswith("+++") for line in joint_diff),
        "joint_fit_removed_lines": sum(line.startswith("-") and not line.startswith("---") for line in joint_diff),
        "joint_fit_change_markers": [marker for marker in ("gradient_norm", "last_gradient", "last_theta") if any(marker in line for line in joint_diff)],
        "objective_source_change_scope": "contact_model byte-identical; joint_fit change is checkpoint/callback gradient bookkeeping",
    })

    data_by_bin_old: dict[int, Any] = {}
    data_by_bin_new: dict[int, Any] = {}
    data_comparison = []
    for bin_size in (5_000_000, 2_000_000, 1_000_000):
        old_data = old_cm.load_aggregate(INPUT, bin_size=bin_size, verify_frozen_hash=True)
        new_data = new_cm.load_aggregate(INPUT, bin_size=bin_size, verify_frozen_hash=True)
        old_budget = old_data.budget()
        new_budget = new_data.budget()
        data_by_bin_old[bin_size] = old_data
        data_by_bin_new[bin_size] = new_data
        arrays_equal = {
            "pair_i": bool(np.array_equal(old_data.pair_i, new_data.pair_i)),
            "pair_j": bool(np.array_equal(old_data.pair_j, new_data.pair_j)),
            "cis_pair": bool(np.array_equal(old_data.cis_pair, new_data.cis_pair)),
            "counts": bool(np.array_equal(old_data.counts, new_data.counts)),
            "diag_counts": bool(np.array_equal(old_data.diag_counts, new_data.diag_counts)),
            "endpoint_counts": bool(np.array_equal(old_data.endpoint_counts, new_data.endpoint_counts)),
            "exposure": bool(np.array_equal(old_data.exposure, new_data.exposure)),
        }
        data_comparison.append({"bin_size_bp": bin_size, "old_budget": old_budget, "new_budget": new_budget, "arrays_equal": arrays_equal, "all_arrays_equal": all(arrays_equal.values())})

    stage_defs = []
    for candidate in ("consensus_joint", "random_joint"):
        for stage in ("5m", "2m", "1m"):
            record_path = RUN020 / "stages" / candidate / f"{stage}.json"
            record = read_json(record_path)
            npz_path = RUN020 / record["fit"]["checkpoint_paths"][-1]
            stage_defs.append({"run_id": f"020-{candidate}-{stage}", "record": record, "record_path": record_path, "npz": npz_path})

    history_rows: list[dict[str, Any]] = []
    comparison_rows = stage_endpoint_rows(old_cm, data_by_bin_old, stage_defs, history_rows)

    # 022 是从 020 random 1 Mb endpoint 开始的 restart，不是独立 start。
    summary022 = read_json(paths["022_continuation_summary"])
    final022 = read_json(paths["022_final_components"])
    npz022 = endpoint_from_npz(paths["022_final_theta"])
    obj022 = old_cm.JointObjective(data_by_bin_old[1_000_000], block_size=65_536, repulsion_block_size=65_536)
    fixed022 = evaluate_fixed(obj022, npz022["theta"])
    progress_rows = []
    for line in paths["022_progress"].read_text(encoding="utf-8").splitlines():
        if line.strip():
            progress_rows.append(json.loads(line))
    for entry in progress_rows:
        history_rows.append(trace_row("022-random-continuation", "022_progress", entry, cumulative_iteration=entry.get("cumulative_iteration"), cumulative_nfev=entry.get("cumulative_nfev")))
    final022_components = final022["components"]
    diff022, diff022_key = component_numeric_diff(final022_components, fixed022["components"])
    comparison_rows.append({
        "run_id": "022-random-continuation", "source_kind": "022", "candidate": "random_joint",
        "start_or_stage": "continuation_240_to_480", "resolution_bp": 1_000_000,
        "p": final022_components.get("p"), "count_nll_normalized": final022_components.get("count_nll_normalized"),
        "total": final022_components.get("total"), "bond": final022_components.get("bond"), "bend": final022_components.get("bend"),
        "repulsion": final022_components.get("repulsion"), "p_prior": final022_components.get("p_prior"),
        "gradient_l2": final022.get("gradient_norm"), "gradient_inf": fixed022["gradient_inf"],
        "max_radius": float(np.linalg.norm(npz022["coordinates"], axis=2).max()), "nit": summary022["termination"].get("nit"),
        "nfev": summary022["termination"].get("actual_nfev"), "budget": f"maxiter={summary022['termination'].get('maxiter')};maxfun={summary022['termination'].get('maxfun')}",
        "termination": summary022["termination"].get("termination_reason"), "status": summary022["termination"].get("status"),
        "recorded_recompute_component_max_abs_diff": diff022, "recorded_recompute_component_key_at_max": diff022_key,
        "coordinate_sha256": sha256_array(npz022["coordinates"]),
        "parent_endpoint": rel(RUN020 / "checkpoints" / "random_joint" / "1m-accepted-0240.npz"),
    })

    # 三个 C0 endpoint rows 和完整 accepted history 都是 training-side JSON。
    final029_payloads = {}
    for bundle in ("bundle1", "bundle2", "bundle3"):
        final_path = paths[f"029_C0_{bundle}_final"]
        final_payload = read_json(final_path)
        final029_payloads[bundle] = final_payload
        npz = endpoint_from_npz(paths[f"029_C0_{bundle}_npz"])
        obj = new_allele.objective_for_model(data_by_bin_new[1_000_000], "C0", block_size=65_536, repulsion_block_size=65_536)
        fixed = evaluate_fixed(obj, npz["theta"])
        final_components = final_payload["final"]["components"]
        diff, diff_key = component_numeric_diff(final_components, fixed["components"])
        for entry in final_payload.get("history", []):
            history_rows.append(trace_row(f"029-C0-{bundle}", "029_C0_final_history", entry))
        comparison_rows.append({
            "run_id": f"029-C0-{bundle}", "source_kind": "029", "candidate": "C0",
            "start_or_stage": bundle, "resolution_bp": 1_000_000,
            "p": final_components.get("p"), "count_nll_normalized": final_components.get("count_nll_normalized"),
            "total": final_components.get("total"), "bond": final_components.get("bond"), "bend": final_components.get("bend"),
            "repulsion": final_components.get("repulsion"), "p_prior": final_components.get("p_prior"),
            "gradient_l2": final_components.get("gradient_l2", final_payload["final"].get("gradient_l2")),
            "gradient_inf": final_payload["final"].get("gradient_inf", fixed["gradient_inf"]),
            "max_radius": float(np.linalg.norm(npz["coordinates"], axis=2).max()), "nit": final_payload["solver"].get("nit"),
            "nfev": final_payload["solver"].get("actual_nfev"), "budget": f"maxiter={final_payload['solver']['fit'].get('maxiter')};maxfun={final_payload['solver']['fit'].get('maxfun')}",
            "termination": final_payload["solver"].get("actual_stop_reason"), "status": final_payload.get("status"),
            "recorded_recompute_component_max_abs_diff": diff, "recorded_recompute_component_key_at_max": diff_key,
            "coordinate_sha256": sha256_array(npz["coordinates"]),
        })

    fields = ["run_id", "source_kind", "candidate", "start_or_stage", "resolution_bp", "p", "count_nll_normalized", "total", "bond", "bend", "repulsion", "p_prior", "gradient_l2", "gradient_inf", "max_radius", "nit", "nfev", "budget", "termination", "status", "recorded_recompute_component_max_abs_diff", "recorded_recompute_component_key_at_max", "coordinate_sha256", "parent_endpoint"]
    write_tsv(AUDIT / "comparison.tsv", fields, comparison_rows)
    history_fields = ["run_id", "source_kind", "iteration", "cumulative_iteration", "nfev", "cumulative_nfev", "elapsed_seconds", "fun", "count_nll_normalized", "p", "p_prior", "bond", "bend", "repulsion", "gradient_l2"]
    write_tsv(AUDIT / "history_metrics.tsv", history_fields, history_rows)

    # 解析保存的文本坐标，并执行 writer/parser round trips。
    data1_old = data_by_bin_old[1_000_000]
    data1_new = data_by_bin_new[1_000_000]
    npz020 = endpoint_from_npz(paths["020_selected_npz"])
    npz029 = endpoint_from_npz(paths["029_C0_bundle2_npz"])
    parsed020 = parse_3dg(paths["020_selected_3dg"], data1_old)
    parsed029 = parse_3dg(paths["029_C0_bundle2_3dg"], data1_new)
    roundtrip_old_path = AUDIT / "roundtrip_archived_writer.3dg"
    roundtrip_new_path = AUDIT / "roundtrip_current_writer.3dg"
    for generated_path in (roundtrip_old_path, roundtrip_new_path):
        if generated_path.exists():
            generated_path.unlink()
    old_cm.write_full_tracks(roundtrip_old_path, data1_old, npz020["coordinates"])
    new_paired.write_coordinates(roundtrip_new_path, data1_new, "C0", npz029["coordinates"])
    parsed_round_old = parse_3dg(roundtrip_old_path, data1_old)
    parsed_round_new = parse_3dg(roundtrip_new_path, data1_new)
    mapping_audit = {
        "track_order_expected": [spec.name for spec in data1_old.track_specs],
        "020_selected_text": {k: v for k, v in parsed020.items() if k != "coordinates"},
        "020_selected_text_vs_npz_max_abs": float(np.max(np.abs(parsed020["coordinates"] - npz020["coordinates"]))),
        "029_C0_bundle2_text": {k: v for k, v in parsed029.items() if k != "coordinates"},
        "029_C0_bundle2_text_vs_npz_max_abs": float(np.max(np.abs(parsed029["coordinates"] - npz029["coordinates"]))),
        "archived_writer": {k: v for k, v in parsed_round_old.items() if k != "coordinates"},
        "current_writer": {k: v for k, v in parsed_round_new.items() if k != "coordinates"},
        "archived_writer_vs_input_max_abs": float(np.max(np.abs(parsed_round_old["coordinates"] - npz020["coordinates"]))),
        "current_writer_vs_input_max_abs": float(np.max(np.abs(parsed_round_new["coordinates"] - npz029["coordinates"]))),
        "writer_file_sha256_equal": sha256_file(roundtrip_old_path) == sha256_file(roundtrip_new_path),
        "writer_file_sha256": {"archived": sha256_file(roundtrip_old_path), "current": sha256_file(roundtrip_new_path)},
    }

    # 对两个锁定 endpoint states 检查 mapping round trips。
    mapping_states = {}
    for name, endpoint, objective in (("020_selected_random_1m", npz020, old_cm.JointObjective(data1_old, block_size=65_536, repulsion_block_size=65_536)), ("029_C0_bundle2_iter480", npz029, new_allele.objective_for_model(data1_new, "C0", block_size=65_536, repulsion_block_size=65_536))):
        raw = endpoint["raw"]
        coordinates = endpoint["coordinates"]
        forward_old = old_cm.sphere_forward(raw)
        forward_new = new_cm.sphere_forward(raw)
        inverse_old = old_cm.sphere_inverse(coordinates)
        inverse_new = new_cm.sphere_inverse(coordinates)
        mapping_states[name] = {
            "raw_shape": list(raw.shape), "coordinates_shape": list(coordinates.shape),
            "raw_finite": bool(np.isfinite(raw).all()), "coordinates_finite": bool(np.isfinite(coordinates).all()),
            "strict_unit_ball": bool(np.all(np.linalg.norm(coordinates, axis=2) < 1.0)),
            "max_radius": float(np.linalg.norm(coordinates, axis=2).max()),
            "forward_old_vs_saved_max_abs": float(np.max(np.abs(forward_old - coordinates))),
            "forward_new_vs_saved_max_abs": float(np.max(np.abs(forward_new - coordinates))),
            "forward_old_vs_new_max_abs": float(np.max(np.abs(forward_old - forward_new))),
            "inverse_old_vs_raw_max_abs": float(np.max(np.abs(inverse_old - raw))),
            "inverse_new_vs_raw_max_abs": float(np.max(np.abs(inverse_new - raw))),
            "theta_raw_vs_payload_raw_max_abs": float(np.max(np.abs(endpoint["theta"][:-1].reshape(2, 2645, 3) - raw))),
            "q": float(endpoint["theta"][-1]),
            "p": float(old_cm.p_from_q(float(endpoint["theta"][-1]))[0]),
            "p_derivative_q": float(old_cm.p_from_q(float(endpoint["theta"][-1]))[1]),
        }

    fixed_state_comparisons = []
    old_obj_selected = old_cm.JointObjective(data1_old, block_size=65_536, repulsion_block_size=65_536)
    new_obj_selected = new_allele.objective_for_model(data1_new, "C0", block_size=65_536, repulsion_block_size=65_536)
    fixed_state_comparisons.append(compare_fixed(old_obj_selected, new_obj_selected, npz020["theta"], npz020["coordinates"], "020_selected_random_1m"))
    old_obj_c0 = old_cm.JointObjective(data1_old, block_size=65_536, repulsion_block_size=65_536)
    new_obj_c0 = new_allele.objective_for_model(data1_new, "C0", block_size=65_536, repulsion_block_size=65_536)
    fixed_state_comparisons.append(compare_fixed(old_obj_c0, new_obj_c0, npz029["theta"], npz029["coordinates"], "029_C0_bundle2_iter480"))

    # 显式验证三个 025 native outputs、track-major mapping 以及 one-center/.8 scaling。
    native_audits = {}
    for bundle in ("bundle1", "bundle2", "bundle3"):
        native_audits[bundle] = native_output_audit(paths[f"025_{bundle}_raw_output"], paths[f"025_{bundle}_raw_npz"], paths[f"025_{bundle}_x0_npz"], paths[f"025_{bundle}_x0_3dg"], data1_old, parse_3dg)
        start = new_paired.load_paired_start(paths[f"029_{bundle}_x0_npz"], start_id=bundle)
        native_audits[bundle]["paired_start"] = {
            "coordinate_sha256": start.coordinate_sha256,
            "shape": list(start.coordinates.shape), "p_init": float(start.p_init),
            "max_radius": start.max_radius, "finite": bool(np.isfinite(start.coordinates).all()),
            "strict_unit_ball": bool(np.all(np.linalg.norm(start.coordinates, axis=2) < 1.0)),
            "coordinate_vs_x0_npz_max_abs": float(np.max(np.abs(start.coordinates - load_npz(paths[f"025_{bundle}_x0_npz"])["coordinates"]))),
            "025_vs_029_x0_file_sha256_equal": sha256_file(paths[f"025_{bundle}_x0_npz"]) == sha256_file(paths[f"029_{bundle}_x0_npz"]),
        }
    starts = [load_npz(paths[f"029_{bundle}_x0_npz"])["coordinates"] for bundle in ("bundle1", "bundle2", "bundle3")]
    pairwise_starts = {}
    for i, left in enumerate(("bundle1", "bundle2", "bundle3")):
        for j, right in enumerate(("bundle1", "bundle2", "bundle3")):
            if j <= i:
                continue
            pairwise_starts[f"{left}_vs_{right}"] = {
                "max_abs": float(np.max(np.abs(starts[i] - starts[j]))),
                "rms": float(np.sqrt(np.mean((starts[i] - starts[j]) ** 2))),
            }

    # 不需要打开额外 payload 的 endpoint 和 source metadata 检查。
    selected020 = read_json(paths["020_selection"])
    selected_random = next(item for item in selected020["candidates"] if item["id"] == "random_joint")
    selected020_selection = selected020["selection"]
    c0_selection = read_json(paths["029_selection"])
    c0_rows = next(item for item in c0_selection["variants"] if item["model_id"] == "C0")["candidates"]
    c0_endpoints = {row["bundle_id"]: row for row in c0_rows}
    endpoint_metadata = {
        "020_selected_candidate": {
            "id": selected020_selection["selected_id"],
            "rule": selected020_selection["rule"],
            "count_nll_normalized": selected_random["count_model"]["count_nll_normalized"],
            "total": selected_random["count_model"]["total"],
            "p": selected_random["count_model"]["p"],
            "q_fixed_from_final_fit": selected_random["selection_rescore"]["q_fixed_from_final_fit"],
            "coordinate_sha256": selected020_selection["selected_coordinate"]["source_coordinate_sha256"],
        },
        "029_C0_candidates": c0_endpoints,
        "029_fit_called_status": {bundle: read_json(paths[f"029_C0_{bundle}_status"])["fit_called"] for bundle in ("bundle1", "bundle2", "bundle3")},
        "029_x0_job_hashes": {bundle: read_json(paths[f"029_C0_{bundle}_job"])["x0_coordinate_sha256"] for bundle in ("bundle1", "bundle2", "bundle3")},
    }

    # 最终验证使用声明的 tolerances；未发生 optimizer 或 native 调用。
    tolerances = {
        "source_contact_model_byte_identity": True,
        "data_arrays_exact": True,
        "mapping_coordinate_abs": 5e-12,
        "objective_value_abs": 1e-8,
        "objective_gradient_abs": 1e-8,
        "objective_component_abs": 1e-8,
        "recorded_recompute_component_abs": 1e-7,
    }
    checks = {
        "source_contact_model_byte_identical": source_comparison["contact_model_byte_identical"],
        "all_data_arrays_exact_at_5m_2m_1m": all(item["all_arrays_equal"] for item in data_comparison),
        "all_input_budgets_conserved": all(item["old_budget"].get("raw_conserved") and item["old_budget"].get("aggregate_conserved") and item["old_budget"].get("endpoint_conserved") for item in data_comparison),
        "020_selected_text_roundtrip": mapping_audit["020_selected_text_vs_npz_max_abs"] <= tolerances["mapping_coordinate_abs"] and mapping_audit["020_selected_text"]["track_order_ok"],
        "029_C0_text_roundtrip": mapping_audit["029_C0_bundle2_text_vs_npz_max_abs"] <= tolerances["mapping_coordinate_abs"] and mapping_audit["029_C0_bundle2_text"]["track_order_ok"],
        "writer_parser_roundtrip": mapping_audit["archived_writer_vs_input_max_abs"] <= tolerances["mapping_coordinate_abs"] and mapping_audit["current_writer_vs_input_max_abs"] <= tolerances["mapping_coordinate_abs"],
        "native_mapping_all_bundles": all(native_audits[b]["grid_mapping"]["final_bead_to_copyfirst_max_abs"] <= 1e-12 for b in native_audits),
        "native_normalization_all_bundles": all(native_audits[b]["normalization"]["expected_x0_copyfirst_max_abs"] <= 1e-12 for b in native_audits),
        "all_starts_shape_finite_strict_ball": all(native_audits[b]["paired_start"]["shape"] == [2, 2645, 3] and native_audits[b]["paired_start"]["finite"] and native_audits[b]["paired_start"]["strict_unit_ball"] for b in native_audits),
        "all_020_022_029_recomputed_records_match": all(row["recorded_recompute_component_max_abs_diff"] <= tolerances["recorded_recompute_component_abs"] for row in comparison_rows),
        "old_current_fixed_state_objective_equivalence": all(item["old_new_value_abs_diff"] <= tolerances["objective_value_abs"] and item["old_new_gradient_max_abs_diff"] <= tolerances["objective_gradient_abs"] and item["old_new_component_max_abs_diff"] <= tolerances["objective_component_abs"] for item in fixed_state_comparisons),
        "029_actual_fit_called_for_all_c0": all(endpoint_metadata["029_fit_called_status"].values()),
        "no_forbidden_allow_list_path": all(not is_forbidden_path(path) for path in paths.values()),
    }
    verification = {
        "schema": "reference-free-fixed-state-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "execution": {
            "command": "source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh && conda activate analysis && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python test_res/030-20260914_c0-020-training-audit/run_audit.py",
            "exit_code": 0 if all(checks.values()) else 1,
            "optimizer_called": False, "native_fdg_called": False,
            "reference_opened": False, "phase_opened": False, "evaluation_outputs_opened": False,
        },
        "freeze": {"path": rel(AUDIT / "frozen_inputs.json"), "sha256": freeze["freeze_sha256"], "authorized_file_count": freeze["authorized_file_count"]},
        "tolerances": tolerances,
        "checks": checks,
        "source_comparison": source_comparison,
        "data_comparison": data_comparison,
        "endpoint_metadata": endpoint_metadata,
        "fixed_state_objective_comparison": fixed_state_comparisons,
        "mapping_roundtrips": mapping_states,
        "coordinate_serialization": mapping_audit,
        "native_preflight_mapping": native_audits,
        "pairwise_start_distances": pairwise_starts,
        "historical_postprocess_bug": "confirmed from 025 validation metadata; recovered without native rerun; current x0 mapping independently rechecked",
        "limitations": [
            "Fixed-state evaluation cannot establish which nonconvex basin a longer fit would reach.",
            "No reference/phase/evaluation read was permitted, so no biological recovery or L2 claim is made.",
            "The three 025 native starts are technical starts from one native protocol, not biological replicates.",
            "020 stage histories predate explicit gradient_norm recording; endpoint gradients in comparison.tsv are recomputed only at locked endpoints.",
        ],
    }
    write_json(AUDIT / "verification.json", verification)
    print(json.dumps({"status": verification["status"], "audit": rel(AUDIT), "checks": checks}, sort_keys=True))
    return int(verification["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
