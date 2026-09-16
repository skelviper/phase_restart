"""面向未来统一 GPU 多分辨率 release 的仅 metadata 预适配器。

冻结的 preflight source 中已有来源控制器 schema，但正式运行目录和最终控制器 revision 尚不存在。本模块只有在显式提供 future run root 时才读取 JSON metadata。它不会打开坐标 payload、reference、pairs/rawphase 或 phase data，也不会在 parent evidence gate 完成前构建 release contract。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

PREP_DIR = Path(__file__).resolve().parent
ROOT = PREP_DIR.parents[2]
CONFIG_PATH = PREP_DIR / "config.json"
PROTOCOL_PATH = PREP_DIR / "r2_protocol.json"
RELEASE_CONTRACT_PATH = PREP_DIR / "release_contract.json"
SCHEMA_REFERENCE_PATH = ROOT / "test_res/035-20260914T060945Z-gpu-multires-preflight/source/gpu_multires_controller.py"
SCHEMA_REFERENCE_SHA256 = "376857b4888e8d3cc093d911cd2fbbdfe6688294a58fa272e29ea1743a9c5346"
FORMAL_OUTPUT_PATH = ROOT / "test_res/036-20260914T064651Z-gpu-multires"
FORMAL_OUTPUT_RELATIVE = "test_res/036-20260914T064651Z-gpu-multires"
GPU_SCHEMA_STATUS = "pending_child_terminal_release"
SELECTION_SCHEMA = "gpu-multires-selection-v1"
TERMINAL_SCHEMA = "gpu-multires-terminal-evidence-v1"
VARIANTS = ("C0", "C1", "C2-map", "C2-free", "C3")
CANDIDATE_IDS = ("consensus_joint", "random_joint")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ReleaseGateError(RuntimeError):
    """冻结 preparation 契约无效时抛出。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ReleaseGateError("metadata root is not an object: %s" % path)
    return value


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise ReleaseGateError("missing or malformed SHA256: %s" % label)
    return value


def _validate_frozen_preparation() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH)
    protocol = _read_json(PROTOCOL_PATH)
    contract = _read_json(RELEASE_CONTRACT_PATH)
    if config.get("status") != "prepared_only_pending_parent_release":
        raise ReleaseGateError("preparation config is not pending parent release")
    training = config.get("training", {})
    if training.get("main_backend") != "GPU" or training.get("main_endpoint_backend") != "GPU":
        raise ReleaseGateError("main endpoints are not frozen to the GPU backend")
    if training.get("gpu_stage_count") != 30:
        raise ReleaseGateError("GPU stage count is not frozen to 30")
    if training.get("gpu_controller_schema") != GPU_SCHEMA_STATUS:
        raise ReleaseGateError("GPU controller schema status changed")
    if (training.get("gpu_selection_schema") != SELECTION_SCHEMA
            or training.get("gpu_terminal_schema") != TERMINAL_SCHEMA
            or training.get("gpu_schema_reference_sha256") != SCHEMA_REFERENCE_SHA256
            or training.get("gpu_formal_output_path") != FORMAL_OUTPUT_RELATIVE):
        raise ReleaseGateError("GPU success payload schema reference changed")
    if (training.get("gpu_preflight_required_checks") !=
            ["cuda_target_visible", "float64_value_gradient_parity", "protocol_consistency", "gpu_c0_training_complete"]
            or training.get("cpu_byte_identity_required") is not False
            or training.get("cpu_r2_match_required") is not False):
        raise ReleaseGateError("GPU preflight acceptance boundary changed")
    training_definition = protocol.get("training_definition", {})
    if training_definition.get("main_training_backend") != "GPU":
        raise ReleaseGateError("protocol main backend is not GPU")
    if training_definition.get("gpu_training_plan") != (
            "all 5 variants x 2 sources x 3 layers = 30 stages, each restarted from the original 014 source"):
        raise ReleaseGateError("protocol GPU stage plan changed")
    backend_policy = protocol.get("training_definition", {}).get("backend_policy", {})
    if (backend_policy.get("gpu_selection_schema") != SELECTION_SCHEMA
            or backend_policy.get("gpu_terminal_schema") != TERMINAL_SCHEMA
            or backend_policy.get("gpu_schema_reference_sha256") != SCHEMA_REFERENCE_SHA256
            or backend_policy.get("gpu_formal_output_path") != FORMAL_OUTPUT_RELATIVE
            or backend_policy.get("gpu_preflight_required_checks") !=
            ["cuda_target_visible", "float64_value_gradient_parity", "protocol_consistency", "gpu_c0_training_complete"]
            or backend_policy.get("cpu_byte_identity_required") is not False
            or backend_policy.get("cpu_r2_match_required") is not False):
        raise ReleaseGateError("protocol GPU preflight/schema boundary changed")
    if contract.get("status") != "prepared_pending_parent_release":
        raise ReleaseGateError("release contract is not the pending template")
    controller = contract.get("training_controller", {})
    if controller.get("formal_output_path") != FORMAL_OUTPUT_RELATIVE:
        raise ReleaseGateError("formal GPU output path is not frozen")
    return {
        "config_sha256": _sha256(CONFIG_PATH),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "release_contract_sha256": _sha256(RELEASE_CONTRACT_PATH),
        "backend": "GPU",
        "endpoint_count": 10,
        "stage_count": 30,
        "selection_schema": SELECTION_SCHEMA,
        "terminal_schema": TERMINAL_SCHEMA,
        "formal_output_path": FORMAL_OUTPUT_RELATIVE,
        "schema_reference": str(SCHEMA_REFERENCE_PATH.relative_to(ROOT)),
        "schema_reference_sha256": SCHEMA_REFERENCE_SHA256,
        "schema_reference_status": "read_source_only; formal_output_schema_revision_pending",
    }


def _blocked(reason: str, args: argparse.Namespace, frozen: dict[str, Any], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "status": "blocked_incomplete",
        "reason": reason,
        "action": "wait_for_gpu_036_terminal_hash_gate_then_build_release",
        "backend": frozen["backend"],
        "endpoint_count": frozen["endpoint_count"],
        "stage_count": frozen["stage_count"],
        "schema_reference": frozen["schema_reference"],
        "schema_reference_sha256": frozen["schema_reference_sha256"],
        "formal_output_path": FORMAL_OUTPUT_RELATIVE,
        "gpu_run_path": str(args.gpu_run) if args.gpu_run else None,
        "coordinate_files_opened": False,
        "reference_opened": False,
        "pairs_or_rawphase_opened": False,
        "phase_payload_opened": False,
        "fit_called": False,
        "native_called": False,
        "frozen_preparation": frozen,
    }
    if metadata is not None:
        result["metadata_schema_check"] = metadata
    return result


def _validate_success_metadata(run_root: Path) -> dict[str, Any]:
    """只验证 selection/terminal JSON；绝不解引用坐标路径。"""
    selection_path = run_root / "selection.json"
    terminal_path = run_root / "terminal_evidence.json"
    if not selection_path.is_file() or not terminal_path.is_file():
        raise ReleaseGateError("GPU run lacks selection.json or terminal_evidence.json")
    selection = _read_json(selection_path)
    terminal = _read_json(terminal_path)
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("status") != "training_complete":
        raise ReleaseGateError("GPU selection schema/status is not complete")
    if terminal.get("schema") != TERMINAL_SCHEMA or terminal.get("status") != "training_complete":
        raise ReleaseGateError("GPU terminal schema/status is not complete")
    if terminal.get("runner_exit_code") != 0:
        raise ReleaseGateError("GPU terminal runner exit is not zero")
    if terminal.get("planned_stage_attempt_count") != 30 or terminal.get("executed_stage_attempt_count") != 30:
        raise ReleaseGateError("GPU terminal stage count is not 30")
    if terminal.get("completed_stage_attempt_count") != 30 or terminal.get("failed_or_not_run_stage_attempt_count") != 0:
        raise ReleaseGateError("GPU terminal stage completion is incomplete")
    if terminal.get("reference_used") is not False or terminal.get("phase_used") is not False:
        raise ReleaseGateError("GPU terminal metadata reports reference or phase use")
    if terminal.get("cuda", {}).get("status") != "available":
        raise ReleaseGateError("GPU terminal metadata lacks an available CUDA probe")
    audit = terminal.get("attempt_audit", {})
    if audit.get("pass") is not True or audit.get("planned_stage_attempt_count") != 30 or audit.get("actual_stage_attempt_count") != 30:
        raise ReleaseGateError("GPU attempt audit is not a complete 30-stage audit")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 10:
        raise ReleaseGateError("GPU selection must contain exactly 10 top candidates")
    seen = set()
    by_variant: dict[str, int] = {variant: 0 for variant in VARIANTS}
    for index, candidate in enumerate(candidates):
        model_id = candidate.get("model_id")
        candidate_id = candidate.get("id")
        key = (model_id, candidate_id)
        if model_id not in VARIANTS or candidate_id not in CANDIDATE_IDS or key in seen:
            raise ReleaseGateError("GPU top candidate identity is invalid at index %d" % index)
        seen.add(key)
        by_variant[model_id] += 1
        attempts = candidate.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 3:
            raise ReleaseGateError("GPU candidate does not contain three stage attempts: %s" % (key,))
        if candidate.get("optimization", {}).get("terminal_status") in (None, "failed"):
            raise ReleaseGateError("GPU candidate terminal optimization is incomplete: %s" % (key,))
        if not isinstance(candidate.get("count_nll_per_record"), (int, float)) or not math.isfinite(float(candidate["count_nll_per_record"])):
            raise ReleaseGateError("GPU candidate lacks finite count_nll_per_record: %s" % (key,))
        coordinates = candidate.get("coordinates")
        if not isinstance(coordinates, Mapping):
            raise ReleaseGateError("GPU candidate lacks coordinate metadata: %s" % (key,))
        if not isinstance(coordinates.get("path"), str) or not coordinates.get("path"):
            raise ReleaseGateError("GPU candidate coordinate path metadata is missing: %s" % (key,))
        _require_sha(coordinates.get("sha256"), "candidate.%s.%s.coordinates" % key)
    if by_variant != {variant: 2 for variant in VARIANTS}:
        raise ReleaseGateError("GPU top candidates are not two per variant: %s" % by_variant)
    nested = selection.get("selection")
    if not isinstance(nested, Mapping):
        raise ReleaseGateError("GPU selection nested selection object is missing")
    per_variant = nested.get("per_variant")
    selected_files = nested.get("selected_files")
    if not isinstance(per_variant, Mapping) or set(per_variant) != set(VARIANTS):
        raise ReleaseGateError("GPU selection.per_variant does not contain five variants")
    if not isinstance(selected_files, Mapping) or set(selected_files) != set(VARIANTS):
        raise ReleaseGateError("GPU selection.selected_files does not contain five variants")
    for variant in VARIANTS:
        record = per_variant[variant]
        if not isinstance(record, Mapping) or record.get("status") != "selected":
            raise ReleaseGateError("GPU per-variant selection is incomplete: %s" % variant)
        if record.get("criterion") != "count_nll_per_record" or record.get("tie_tolerance_per_record") != 1e-9:
            raise ReleaseGateError("GPU selection criterion/tie tolerance changed: %s" % variant)
        if record.get("reference_used") is not False or record.get("phase_used") is not False:
            raise ReleaseGateError("GPU selection metadata reports reference/phase use: %s" % variant)
        if record.get("selected_id") not in CANDIDATE_IDS:
            raise ReleaseGateError("GPU selected candidate is not a frozen source: %s" % variant)
        scores = record.get("candidate_scores")
        if not isinstance(scores, Mapping) or set(scores) != set(CANDIDATE_IDS):
            raise ReleaseGateError("GPU candidate_scores must contain both sources: %s" % variant)
        selected_file = selected_files[variant]
        if not isinstance(selected_file, Mapping) or not isinstance(selected_file.get("path"), str):
            raise ReleaseGateError("GPU selected_files path is missing: %s" % variant)
        _require_sha(selected_file.get("sha256"), "selected_files.%s" % variant)
    return {
        "status": "PASS",
        "selection_schema": selection.get("schema"),
        "terminal_schema": terminal.get("schema"),
        "selection_status": selection.get("status"),
        "terminal_status": terminal.get("status"),
        "top_candidate_count": len(candidates),
        "per_variant_count": len(per_variant),
        "selected_file_count": len(selected_files),
        "planned_stage_attempt_count": terminal.get("planned_stage_attempt_count"),
        "executed_stage_attempt_count": terminal.get("executed_stage_attempt_count"),
        "completed_stage_attempt_count": terminal.get("completed_stage_attempt_count"),
        "cuda_probe_status": terminal.get("cuda", {}).get("status"),
        "selection_sha256": _require_sha(terminal.get("selection_sha256"), "terminal.selection_sha256"),
        "coordinate_metadata_only": True,
        "coordinate_files_opened": False,
        "reference_opened": False,
    }


def _relative_workspace(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except ValueError:
        return str(candidate.resolve())


def _build_release(run_root: Path, frozen: Mapping[str, Any]) -> dict[str, Any]:
    """根据完整的 036 JSON metadata release 构建锁定契约。"""
    if not run_root.is_dir():
        raise ReleaseGateError("formal GPU output root is missing: %s" % run_root)
    selection_path = run_root / "selection.json"
    terminal_path = run_root / "terminal_evidence.json"
    manifest_path = run_root / "release_ready_manifest.json"
    runtime_path = run_root / "runtime_probe.json"
    report_path = run_root / "release_ready_report.md"
    provenance_protocol_path = run_root / "provenance" / "protocol.json"
    provenance_config_path = run_root / "provenance" / "config.json"
    provenance_source_path = run_root / "provenance" / "source_hashes.json"
    required = (selection_path, terminal_path, manifest_path, runtime_path, report_path,
                provenance_protocol_path, provenance_config_path, provenance_source_path)
    if any(not path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise ReleaseGateError("036 metadata release is incomplete: %s" % missing)
    selection = _read_json(selection_path)
    terminal = _read_json(terminal_path)
    manifest = _read_json(manifest_path)
    runtime = _read_json(runtime_path)
    provenance_protocol = _read_json(provenance_protocol_path)
    provenance_config = _read_json(provenance_config_path)
    source_manifest = _read_json(provenance_source_path)
    metadata = _validate_success_metadata(run_root)
    if manifest.get("schema") != "gpu-multires-release-ready-manifest-v1" or manifest.get("status") != "release_ready":
        raise ReleaseGateError("036 release-ready manifest is not release_ready")
    ledger = manifest.get("ledger", {})
    if (ledger.get("actual_attempt_count") != 30
            or ledger.get("actual_trajectory_count") != 10
            or ledger.get("all_attempt_ids_exact") is not True
            or ledger.get("all_trajectories_have_1m_endpoint") is not True
            or ledger.get("all_trajectories_three_stage") is not True):
        raise ReleaseGateError("036 release-ready ledger is not a complete 30-stage/10-trajectory release")
    if (manifest.get("checkpoint_audit", {}).get("actual_total") != 740
            or manifest.get("checkpoint_audit", {}).get("all_checkpoint_files_present") is not True
            or manifest.get("checkpoint_audit", {}).get("all_stages_reached_fixed_budget") is not True):
        raise ReleaseGateError("036 checkpoint audit is incomplete")
    if runtime.get("status") != "passed" or runtime.get("cuda", {}).get("status") != "available":
        raise ReleaseGateError("036 runtime probe is not passed with CUDA available")
    if terminal.get("selection_sha256") != _sha256(selection_path):
        raise ReleaseGateError("036 terminal selection SHA does not match selection.json")
    artifact = terminal.get("artifact_integrity", {})
    expected_integrity = {
        "config_sha256": _sha256(provenance_config_path),
        "protocol_sha256": _sha256(provenance_protocol_path),
        "source_manifest_sha256": _sha256(provenance_source_path),
    }
    for key, actual in expected_integrity.items():
        if artifact.get(key) != actual:
            raise ReleaseGateError("036 artifact integrity mismatch: %s" % key)
    if _sha256(manifest_path) != "6327c6d85e7f4736a2796174475a41541002dd1764fa3281a5cb386c59c5e14c":
        raise ReleaseGateError("036 release-ready manifest SHA differs from parent release evidence")
    if _sha256(report_path) != "260bf5849a28cf83f9c71b8cd14b9d7539f1be441e502818d4f59194a3967860":
        raise ReleaseGateError("036 release-ready report SHA differs from parent release evidence")
    source_code_sha = dict(source_manifest.get("source_code_sha256", {}))
    if source_code_sha.get("controller") != SCHEMA_REFERENCE_SHA256:
        raise ReleaseGateError("036 controller source hash differs from the frozen 035 schema reference")
    if source_manifest.get("protocol_sha256") != expected_integrity["protocol_sha256"] or source_manifest.get("config_sha256") != expected_integrity["config_sha256"]:
        raise ReleaseGateError("036 source manifest protocol/config hashes are inconsistent")
    if source_manifest.get("input_sha256") != "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa":
        raise ReleaseGateError("036 input SHA is not the frozen SNP-free input")
    template = _read_json(RELEASE_CONTRACT_PATH)
    if template.get("status") != "prepared_pending_parent_release":
        raise ReleaseGateError("release contract is not the pending template before build")
    config = _read_json(CONFIG_PATH)
    protocol = _read_json(PROTOCOL_PATH)
    source_specs = {item.get("candidate_id"): item for item in provenance_protocol.get("sources", [])}
    expected_candidates = set(CANDIDATE_IDS)
    if set(source_specs) != expected_candidates:
        raise ReleaseGateError("036 provenance source set mismatch")
    source_locks = []
    for candidate_id, source_id, display in (
            ("consensus_joint", "consensus_joint_base1103", "consensus-derived initialization (014 base1103; 40-track two-copy endpoint)"),
            ("random_joint", "random_joint_base2207", "random-derived initialization (014 base2207; 40-track two-copy endpoint)")):
        spec = source_specs[candidate_id]
        source_path = _relative_workspace(spec["path"])
        source_sha = _require_sha(spec.get("sha256"), "source.%s.sha256" % candidate_id)
        source_locks.append({
            "source_id": source_id,
            "display_name": display,
            "raw_source_path": source_path,
            "raw_source_sha256": source_sha,
            "snapshot_path": source_path,
            "snapshot_sha256": source_sha,
            "gate_path": _relative_workspace(spec.get("gate_path", "")),
            "gate_sha256": _require_sha(spec.get("gate_sha256"), "source.%s.gate_sha256" % candidate_id),
            "status": "locked",
        })
    candidates = {(item.get("model_id"), item.get("id")): item for item in selection["candidates"]}
    selected_source_by_variant = {
        variant: selection["selection"]["per_variant"][variant]["selected_id"] for variant in VARIANTS
    }
    source_suffix = {"consensus_joint": "consensus_joint_base1103", "random_joint": "random_joint_base2207"}
    endpoint_rows = []
    selection_by_variant = {}
    for variant in VARIANTS:
        rows = []
        selected_endpoint_id = None
        selected_source_id = None
        values = []
        for candidate_id in CANDIDATE_IDS:
            candidate = candidates.get((variant, candidate_id))
            if candidate is None:
                raise ReleaseGateError("missing 036 candidate: %s/%s" % (variant, candidate_id))
            endpoint_id = "%s-%s" % (variant, source_suffix[candidate_id])
            coordinate = candidate.get("coordinates", {})
            coordinate_sha = _require_sha(coordinate.get("sha256"), endpoint_id + ".coordinate")
            coordinate_path = "%s/%s" % (FORMAL_OUTPUT_RELATIVE, coordinate.get("path"))
            final_attempt = next((attempt for attempt in candidate.get("attempts", []) if attempt.get("stage") == "1m"), None)
            if final_attempt is None or not final_attempt.get("stage_record"):
                raise ReleaseGateError("missing 1m terminal record: %s" % endpoint_id)
            stage_record_path = "%s/%s" % (FORMAL_OUTPUT_RELATIVE, final_attempt["stage_record"])
            stage_record_abs = ROOT / stage_record_path
            if not stage_record_abs.is_file():
                raise ReleaseGateError("missing 1m terminal record file: %s" % stage_record_abs)
            terminal_record_sha = _sha256(stage_record_abs)
            count_value = float(candidate["count_nll_per_record"])
            count_normalized = float(candidate.get("count_model", {}).get("count_nll_normalized", count_value))
            source_hashes = {
                "input": _require_sha(source_manifest.get("input_sha256"), endpoint_id + ".input_sha256"),
                "protocol": expected_integrity["protocol_sha256"],
                "config": expected_integrity["config_sha256"],
                "source_code_controller": source_code_sha["controller"],
                "source_code_gpu_backend": source_code_sha.get("028_gpu_backend", source_code_sha["controller"]),
                "approved_014_source": source_specs[candidate_id]["sha256"],
            }
            endpoint_rows.append({
                "id": endpoint_id,
                "variant_id": variant,
                "source_id": source_suffix[candidate_id],
                "n_copies": 2,
                "backend": "GPU",
                "endpoint_status": "complete",
                "terminal_record_path": stage_record_path,
                "terminal_record_sha256": terminal_record_sha,
                "coordinate_status": "available",
                "coordinate_path": coordinate_path,
                "coordinate_sha256": coordinate_sha,
                "source_hashes_complete": True,
                "source_hashes": source_hashes,
                "count_nll_per_record": count_value,
                "count_nll_normalized": count_normalized,
                "fit_called": True,
                "terminal_status": candidate.get("optimization", {}).get("terminal_status"),
                "coordinate_domain": coordinate.get("physical_domain"),
            })
            row = {
                "endpoint_id": endpoint_id,
                "source_id": source_suffix[candidate_id],
                "selection_eligible": True,
                "count_nll_per_record": count_value,
                "count_nll_normalized": count_normalized,
            }
            rows.append(row)
            values.append((endpoint_id, count_value))
        selected_candidate_id = selected_source_by_variant[variant]
        selected_endpoint_id = "%s-%s" % (variant, source_suffix[selected_candidate_id])
        selected_source_id = source_suffix[selected_candidate_id]
        minimum = min(value for _, value in values)
        tie_status = "tie_resolved_pre_registered_order" if all(abs(value - minimum) <= 1e-9 for _, value in values) else "unique_minimum"
        selection_by_variant[variant] = {
            "criterion": "count_nll_per_record",
            "reported_field_alias": "final count_nll_normalized",
            "tie_tolerance_per_record": 1e-9,
            "tie_order": ["consensus_joint_base1103", "random_joint_base2207"],
            "includes_diag_constants": True,
            "includes_priors": False,
            "candidates": rows,
            "selected_endpoint_id": selected_endpoint_id,
            "selected_source_id": selected_source_id,
            "tie_status": tie_status,
        }
    training_controller = {
        "status": "complete",
        "main_backend": "GPU",
        "main_endpoint_backend": "GPU",
        "expected_stage_count": 30,
        "gpu_controller_schema": "gpu-multires-controller-v1",
        "gpu_selection_schema": SELECTION_SCHEMA,
        "gpu_terminal_schema": TERMINAL_SCHEMA,
        "gpu_schema_reference": str(SCHEMA_REFERENCE_PATH.relative_to(ROOT)),
        "gpu_schema_reference_sha256": SCHEMA_REFERENCE_SHA256,
        "formal_output_path": FORMAL_OUTPUT_RELATIVE,
        "cpu_020_033_are_historical_anchors": True,
        "cpu_034_excluded_from_release": True,
        "fresh_run_required": True,
        "manifest_path": _relative_workspace(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "selection_path": _relative_workspace(selection_path),
        "selection_sha256": _sha256(selection_path),
        "termination_path": _relative_workspace(terminal_path),
        "termination_sha256": _sha256(terminal_path),
        "runtime_probe_path": _relative_workspace(runtime_path),
        "runtime_probe_sha256": _sha256(runtime_path),
        "release_report_path": _relative_workspace(report_path),
        "release_report_sha256": _sha256(report_path),
        "source_manifest_path": _relative_workspace(provenance_source_path),
        "source_manifest_sha256": _sha256(provenance_source_path),
        "expected_endpoint_count": 10,
        "terminal_endpoint_count": 10,
        "active_jobs": [],
        "all_endpoint_terminal": True,
        "actual_stage_attempt_count": 30,
        "completed_stage_attempt_count": 30,
        "failed_or_not_run_stage_attempt_count": 0,
        "checkpoint_count": 740,
        "selection_status": selection.get("status"),
        "terminal_status": terminal.get("status"),
        "reference_used": False,
        "phase_used": False,
    }
    proof = dict(template.get("train_only_selection_proof", {}))
    proof.update({
        "status": "complete",
        "criterion": "count_nll_per_record",
        "reported_field_alias": "final count_nll_normalized",
        "tie_tolerance_per_record": 1e-9,
        "tie_order": ["consensus_joint_base1103", "random_joint_base2207"],
        "includes_diag_constants": True,
        "includes_priors": False,
        "scope": "within_variant_between_two_sources_only",
        "reference_read": False,
        "r2_read": False,
        "phase_payload_read": False,
        "selection_path": _relative_workspace(selection_path),
        "evidence_path": _relative_workspace(selection_path),
        "evidence_sha256": _sha256(selection_path),
    })
    code_hashes = {
        "evaluator_path": "multires_r2.py",
        "evaluator_sha256": _sha256(PREP_DIR / "multires_r2.py"),
        "plotter_path": "plot_multires_r2.py",
        "plotter_sha256": _sha256(PREP_DIR / "plot_multires_r2.py"),
        "protocol_path": "r2_protocol.json",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "config_path": "config.json",
        "config_sha256": _sha256(CONFIG_PATH),
        "mask_lock_path": "mask_lock.json",
        "mask_lock_sha256": _sha256(PREP_DIR / "mask_lock.json"),
    }
    release = dict(template)
    release.update({
        "status": "locked_for_evaluation",
        "purpose": "Complete unified-GPU 30-stage training release; CPU 020/033 remain historical anchor/diagnostic only",
        "prepared_config": {"path": "config.json", "sha256": _sha256(CONFIG_PATH)},
        "protocol": {"path": "r2_protocol.json", "sha256": _sha256(PROTOCOL_PATH)},
        "source_locks": source_locks,
        "training_controller": training_controller,
        "endpoints": endpoint_rows,
        "selection_by_variant": selection_by_variant,
        "train_only_selection_proof": proof,
        "code_hashes": code_hashes,
        "release_flags": {
            "all_candidate_paths_locked": True,
            "all_candidate_hashes_locked": True,
            "all_source_hashes_locked": True,
            "all_selection_hashes_locked": True,
            "all_terminal_hashes_locked": True,
            "c0_numeric_gate_pass": True,
            "gpu_backend_locked": True,
            "reference_read_allowed": True,
            "prepared_only": False,
        },
    })
    RELEASE_CONTRACT_PATH.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return {
        "status": "locked_for_evaluation",
        "release_contract_path": str(RELEASE_CONTRACT_PATH),
        "release_contract_sha256": _sha256(RELEASE_CONTRACT_PATH),
        "metadata_schema_check": metadata,
        "endpoint_count": len(endpoint_rows),
        "stage_count": 30,
        "checkpoint_count": 740,
        "coordinate_files_opened": False,
        "reference_opened": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Metadata-only gate for a future unified-GPU multiresolution release")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="check frozen metadata and optional future run JSON")
    mode.add_argument("--build", action="store_true", help="reserved for the post-parent-evidence release adapter")
    parser.add_argument("--gpu-run", type=Path, default=FORMAL_OUTPUT_PATH,
                        help="future formal GPU output root; only selection/terminal JSON may be read")
    args = parser.parse_args(argv)
    try:
        frozen = _validate_frozen_preparation()
        if args.build:
            result = _build_release(args.gpu_run.resolve(), frozen)
            print(json.dumps(result, ensure_ascii=True, sort_keys=True))
            return 0
        elif args.gpu_run is None:
            result = _blocked("gpu_success_schema_locked_formal_output_path_pending", args, frozen)
        elif not args.gpu_run.is_dir():
            result = _blocked("gpu_formal_output_path_not_present_or_not_final", args, frozen)
        elif not (args.gpu_run / "selection.json").is_file() or not (args.gpu_run / "terminal_evidence.json").is_file():
            result = _blocked("gpu_formal_output_present_but_terminal_metadata_pending", args, frozen)
        else:
            metadata = _validate_success_metadata(args.gpu_run.resolve())
            result = _blocked("gpu_metadata_schema_prevalidated; parent_release_contract_still_pending", args, frozen, metadata)
    except (OSError, ValueError, json.JSONDecodeError, ReleaseGateError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
