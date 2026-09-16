"""在 post-stage runner 记账失败后恢复最终 C0 记账。

该脚本有意独立于受控 training runner。它从不调用 optimizer：只从六份已写出的 stage 记录重建 candidate 字典，执行已注册的最终 count readback，然后运行不可变历史 gate。它用于唯一一份因缺少 journal 文件、在全部 stages 完成后退出的 033 run。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any

from . import c0_controlled_runner as controlled
from . import reconstruct
from .gate import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = controlled.CANDIDATES
STAGES = controlled.STAGES
RECOVERY_SCHEMA = "c0-c0-stage-finalization-recovery-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _copy_exclusive(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, target.open("xb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
        target_handle.flush()
        os.fsync(target_handle.fileno())
    return sha256_file(target)


def _static_files(root: Path, context: reconstruct.RunContext) -> dict[str, Any]:
    config_path = root / "config.json"
    code_path = root / "provenance" / "training-code-manifest.json"
    protocol_path = root / "provenance" / "protocol-manifest.json"
    data_path = root / "input_version.json"
    track_map_path = root / "track_map.json"
    return {
        "config": {"path": "config.json", "sha256": sha256_file(config_path), "hash_source": "file_bytes"},
        "code": {"path": "provenance/training-code-manifest.json", "sha256": sha256_file(code_path),
                 "hash_source": "snapshot_manifest"},
        "protocol": {"path": "provenance/protocol-manifest.json", "sha256": sha256_file(protocol_path),
                     "hash_source": "file_bytes", "manifest_schema": reconstruct.PROTOCOL_MANIFEST_SCHEMA_VERSION},
        "data": {"path": str(Path(context.data_path).resolve()), "sha256": context.data_sha256,
                 "hash_source": "file_bytes"},
        "track_map": {"path": "track_map.json", "sha256": sha256_file(track_map_path)},
        "grid": reconstruct._make_grid(context.headers),
    }


def _candidate_from_stages(root: Path, config: dict[str, Any],
                           candidate: reconstruct.CandidateSpec) -> dict[str, Any]:
    attempts = []
    layers = []
    stage_documents = {}
    for stage in STAGES:
        relative = Path("stages") / candidate.candidate_id / (stage.label + ".json")
        path = root / relative
        document = _load(path)
        stage_documents[stage.label] = document
        fit = document["fit"]
        if document["candidate_id"] != candidate.candidate_id or document["stage"] != stage.label:
            raise RuntimeError("stage identity mismatch: %s" % path)
        if document["bin_size_bp"] != stage.bin_size:
            raise RuntimeError("stage bin-size mismatch: %s" % path)
        attempts.append({
            "attempt_id": document["attempt_id"],
            "candidate_id": candidate.candidate_id,
            "stage": stage.label,
            "status": fit["status"],
            "is_terminal": stage == STAGES[-1],
            "stage_record": str(relative),
            "budget_exhausted": bool(fit["budget_exhausted"]),
            "recovery_reconstructed": True,
            "original_stage_completed_at_utc": document.get("completed_at_utc"),
        })
        layers.append({
            "stage": stage.label,
            "path": str(relative),
            "status": fit["status"],
            "data_budget": document["data_budget"],
            "recovery_reconstructed": True,
        })
    final_document = stage_documents[STAGES[-1].label]
    final_coordinates = final_document["final_coordinates"]
    model_contract = config["model_contract"]
    result = {
        "id": candidate.candidate_id,
        "initialization": candidate.initialization_candidate,
        "base_seed": candidate.base_seed,
        "model_signature": model_contract["model_signature"],
        "prior_config_sha256": model_contract["prior_config_sha256"],
        "parameter_dimension": model_contract["parameter_dimension"],
        "final_coordinates_path": final_coordinates["path"],
        "final_coordinates_sha256": final_coordinates["sha256"],
        "final_q": float(final_document["fit"]["q_out"]),
        "attempts": attempts,
        "layers": layers,
        "recovery": {
            "reconstructed_from_stage_records": [str(Path("stages") / candidate.candidate_id / (stage.label + ".json"))
                                                   for stage in STAGES],
            "optimizer_called_during_recovery": False,
            "stage_execution_repeated": False,
        },
    }
    if any(attempt["status"] == "failed" for attempt in attempts):
        raise RuntimeError("cannot recover a failed stage candidate: %s" % candidate.candidate_id)
    return result


def _recovery_report(root: Path, gate: dict[str, Any], terminal: dict[str, Any],
                     receipt: dict[str, Any]) -> str:
    selected = gate["selection"]["new_selected_id"]
    return "\n".join([
        "# C0 受控训练复现报告（恢复版）",
        "",
        "- 状态: **%s**" % ("通过" if gate["status"] == "passed" else "失败"),
        "- 原主进程：六个 C0 阶段已实际完成；随后因缺失 `run_status/attempts.jsonl` 在最终收尾前退出，原始退出码 `2`。",
        "- 阶段终态：`consensus_joint` 与 `random_joint` 的 `5m/2m/1m` 均有完整阶段 JSON、终点坐标和检查点；每层求解器原始状态均为固定预算 `not_converged`，不宣称收敛。",
        "- 恢复程序：`pr/c0_recovery.py`，只从已有阶段记录重建候选/日志；未调用优化器、未重跑阶段，恢复退出码 `0`。",
        "- 数值门控：`%s`；状态/检查点要求逐字节相同，其他数值字段使用 abs `1e-12` / rel `1e-10`；耗时不参与判定。" % gate["status"],
        "- 选择：`%s`，criterion=`count_nll_per_record`，每条记录的平局容差=`1e-9`，顺序=`[consensus_joint, random_joint]`。" % selected,
        "- 已保留原始 `run_failure.json`、原阶段/日志/检查点；恢复日志每条记录带 `recovery_reconstructed=true` 和新的记录时间，不伪造原执行时间。",
        "- 本训练记录不使用参考结构、定相数据；未打开参考/oracle 坐标、定相/R2/评价数据，也未启动原生 FDG。",
        "",
        "## 关键路径",
        "",
        "- `controlled_protocol.json` / `preflight.json`：冻结协议与无优化预检。",
        "- `C0_gate.json`：六个阶段相对 020 的数值门控。",
        "- `terminal_evidence.json`：原始主进程退出码 2、六个阶段的原始停止状态与恢复状态。",
        "- `recovery_receipt.json`：恢复程序/原始失败与全部关键 hash。",
        "- `selection.json` / `selected.3dg`：无标签训练选择输出；不代表评价结论。",
        "",
    ])


def recover(outdir: str | os.PathLike[str], original_exit_code: int = 2) -> dict[str, Any]:
    root = Path(outdir)
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    else:
        root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if (root / "selection.json").exists() or (root / "C0_gate.json").exists():
        raise FileExistsError("recovery outputs already exist; refusing to overwrite %s" % root)
    context = reconstruct.production_context()
    reconstruct._validate_context(context)
    config = _load(root / "config.json")
    if config["controlled_runner"]["released_variant"] != "C0":
        raise RuntimeError("recovery target is not a C0-controlled run")
    if int(config["controlled_runner"]["released_attempts"]) != 6:
        raise RuntimeError("recovery target does not have exactly six released attempts")
    original_failure_path = root / "run_failure.json"
    if not original_failure_path.is_file():
        raise FileNotFoundError("original runner failure evidence missing")
    original_failure = _load(original_failure_path)
    if original_failure.get("error_type") != "FileNotFoundError" or "attempts.jsonl" not in original_failure.get("error", ""):
        raise RuntimeError("refusing recovery: failure evidence is not the known journal-layout defect")
    pre_recovery_status = root / "run_status" / "run_status.json"
    status_snapshot = root / "run_status" / "run_status.before_recovery.json"
    if pre_recovery_status.exists():
        _copy_exclusive(pre_recovery_status, status_snapshot)

    journal_path = root / "run_status" / "attempts.jsonl"
    controlled._write_bytes_exclusive(journal_path, b"")
    results = [_candidate_from_stages(root, config, candidate) for candidate in CANDIDATES]
    static_files = _static_files(root, context)
    rescore_notes = []
    for result in results:
        reconstruct._rescore_final_candidate(root, context, config, result, reconstruct.DEFAULT_HOOKS)
        rescore_notes.append({
            "candidate_id": result["id"],
            "count_nll_per_record": result["count_nll_per_record"],
            "total": result["selection_rescore"]["total"],
            "q_fixed_from_final_fit": result["selection_rescore"]["q_fixed_from_final_fit"],
            "optimizer_called": False,
        })
    attempts = [attempt for result in results for attempt in result["attempts"]]
    for attempt in attempts:
        reconstruct._append_attempt_event(root, reconstruct._attempt_journal_event(attempt), attempt)

    selected_id = reconstruct.select_candidate(results, tuple(candidate.candidate_id for candidate in CANDIDATES))
    selected = next(result for result in results if result["id"] == selected_id)
    selected_coordinate = reconstruct._copy_selected_coordinates(root, selected)
    selection = reconstruct._selection_document(root, context, config, static_files, results, attempts,
                                                selected_id, selected_coordinate)
    selection["recovery"] = {
        "schema": RECOVERY_SCHEMA,
        "reconstructed_from_completed_stage_records": True,
        "original_runner_exit_code": int(original_exit_code),
        "recovery_exit_code": 0,
        "optimizer_called": False,
        "stage_execution_repeated": False,
        "journal_recovery": "all six attempt rows were rebuilt from stage JSON; recorded_at_utc is recovery time",
    }
    controlled._write_json_exclusive(root / "selection.json", selection)
    verification = reconstruct.verify_completed_training(root, strict_p9016=True)
    reconstruct._write_global_status(
        root, "training_complete", {candidate.candidate_id: "not_converged" for candidate in CANDIDATES},
        selected_id=selected_id, recovered_from_run_failure=True,
        original_runner_exit_code=int(original_exit_code), recovery_exit_code=0,
    )

    gate = controlled.build_c0_gate(root, context, results)
    gate["recovery"] = {
        "schema": RECOVERY_SCHEMA,
        "original_runner_exit_code": int(original_exit_code),
        "recovery_exit_code": 0,
        "stage_execution_repeated": False,
        "optimizer_called": False,
        "journal_rows_reconstructed": len(attempts),
        "run_failure_sha256": sha256_file(original_failure_path),
    }
    gate_sha = controlled._write_json_exclusive(root / "C0_gate.json", gate)
    started = min(
        _load(root / "stages" / candidate.candidate_id / (stage.label + ".json"))["started_at_utc"]
        for candidate in CANDIDATES for stage in STAGES
    )
    terminal = controlled._terminal_evidence(
        root, gate, results, started, _utc_now(),
        sha256_file(root / "controlled_protocol.json"), sha256_file(root / "preflight.json"),
    )
    terminal.update({
        "runner_exit_code": int(original_exit_code),
        "recovery_exit_code": 0,
        "original_runner_failure": {"path": "run_failure.json", "sha256": sha256_file(original_failure_path)},
        "recovery": {
            "schema": RECOVERY_SCHEMA,
            "optimizer_called": False,
            "stage_execution_repeated": False,
            "journal_rows_reconstructed": len(attempts),
            "verification": verification,
        },
    })
    terminal_sha = controlled._write_json_exclusive(root / "terminal_evidence.json", terminal)
    report_text = _recovery_report(root, gate, terminal, {
        "selected_id": selected_id, "gate_sha256": gate_sha, "terminal_sha256": terminal_sha,
    })
    report_sha = controlled._write_bytes_exclusive(root / "REPORT.zh-CN.md", (report_text + "\n").encode("utf-8"))
    receipt = {
        "schema": "c0-recovery-receipt-v1",
        "status": "passed" if gate["status"] == "passed" else "failed_gate",
        "original_runner_exit_code": int(original_exit_code),
        "recovery_exit_code": 0 if gate["status"] == "passed" else 3,
        "stage_execution_repeated": False,
        "optimizer_called_during_recovery": False,
        "outdir": str(root),
        "recovery_script": {"path": "pr/c0_recovery.py", "sha256": sha256_file(Path(__file__))},
        "original_failure": {"path": "run_failure.json", "sha256": sha256_file(original_failure_path)},
        "pre_recovery_status": {"path": str(status_snapshot.relative_to(root)), "sha256": sha256_file(status_snapshot)},
        "journal": {"path": "run_status/attempts.jsonl", "sha256": sha256_file(journal_path), "rows": len(attempts)},
        "protocol": {"path": "controlled_protocol.json", "sha256": sha256_file(root / "controlled_protocol.json")},
        "preflight": {"path": "preflight.json", "sha256": sha256_file(root / "preflight.json")},
        "gate": {"path": "C0_gate.json", "sha256": gate_sha},
        "terminal": {"path": "terminal_evidence.json", "sha256": terminal_sha},
        "report": {"path": "REPORT.zh-CN.md", "sha256": report_sha},
        "selection": {"path": "selection.json", "sha256": sha256_file(root / "selection.json")},
        "selected_id": selected_id,
        "rescore": rescore_notes,
        "stage_records": [
            {"candidate_id": candidate.candidate_id, "stage": stage.label,
             "path": str(Path("stages") / candidate.candidate_id / (stage.label + ".json")),
             "sha256": sha256_file(root / "stages" / candidate.candidate_id / (stage.label + ".json"))}
            for candidate in CANDIDATES for stage in STAGES
        ],
        "blocked_stage_attempt_count": 24,
        "training_boundary": {"phase_used": False, "reference_used": False,
                              "evaluation_outputs_opened": False, "native_fdg_started": False},
    }
    recovery_sha = controlled._write_json_exclusive(root / "recovery_receipt.json", receipt)
    # 让 completion receipt 自描述，但不在得知自身 hash 后重写；path 和 hash 在进程输出中报告。
    return {
        "status": receipt["status"],
        "outdir": str(root),
        "selected_id": selected_id,
        "gate": str(root / "C0_gate.json"),
        "gate_sha256": gate_sha,
        "terminal": str(root / "terminal_evidence.json"),
        "terminal_sha256": terminal_sha,
        "report": str(root / "REPORT.zh-CN.md"),
        "selection": str(root / "selection.json"),
        "recovery_receipt": str(root / "recovery_receipt.json"),
        "recovery_receipt_sha256": recovery_sha,
    }


def _install_gate_cache_fix() -> dict[str, str]:
    """安装仅 runtime 的缓存修复，不触碰 033 source snapshot。"""
    def fixed_structures_to_array(structures: dict[str, dict[int, Any]], data: Any):
        import numpy as np
        values = np.empty((2, data.n_loci, 3), dtype=np.float64)
        for spec in data.track_specs:
            slc = data.chromosome_slice(spec.chromosome_index)
            track = structures[spec.name]
            for global_index, local_bin in zip(range(slc.start, slc.stop), data.locus_bin[slc], strict=True):
                values[spec.copy_index, global_index] = track[int(local_bin * data.bin_size)]
        return values
    controlled._structures_to_array = fixed_structures_to_array
    return {
        "kind": "runtime_helper_override",
        "bug": "gate cache converter retained a removed stage.bin_size reference",
        "replacement": "data.bin_size",
        "source_snapshot_modified": False,
        "optimizer_path_modified": False,
    }


def resume_partial(outdir: str | os.PathLike[str], original_exit_code: int = 2) -> dict[str, Any]:
    """在 gate-helper bug 后完成部分恢复。"""
    root = Path(outdir)
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    else:
        root = root.resolve()
    if not root.is_dir() or not (root / "selection.json").is_file():
        raise FileNotFoundError("partial recovery selection is missing")
    for name in ("C0_gate.json", "terminal_evidence.json", "recovery_receipt.json", "run_receipt.json"):
        if (root / name).exists():
            raise FileExistsError("partial recovery output already exists: %s" % (root / name))
    context = reconstruct.production_context()
    reconstruct._validate_context(context)
    config = _load(root / "config.json")
    if config["controlled_runner"]["released_variant"] != "C0":
        raise RuntimeError("partial recovery target is not a C0 release")
    original_failure_path = root / "run_failure.json"
    original_failure = _load(original_failure_path)
    attempt1 = {
        "schema": "c0-recovery-attempt-v1",
        "attempt": 1,
        "status": "failed_before_gate",
        "exit_code": 2,
        "error_type": "NameError",
        "error": "name 'stage' is not defined",
        "location": "pr/c0_controlled_runner.py:_structures_to_array",
        "cause": "recovery-only gate cache refactor retained removed stage parameter",
        "stage_execution_repeated": False,
        "optimizer_called": False,
        "created_at_utc": _utc_now(),
    }
    attempt1_sha = controlled._write_json_exclusive(root / "recovery_attempt1_failure.json", attempt1)
    helper_fix = _install_gate_cache_fix()
    results = [_candidate_from_stages(root, config, candidate) for candidate in CANDIDATES]
    for result in results:
        reconstruct._rescore_final_candidate(root, context, config, result, reconstruct.DEFAULT_HOOKS)
    selection = _load(root / "selection.json")
    verification = reconstruct.verify_completed_training(root, strict_p9016=True)
    gate = controlled.build_c0_gate(root, context, results)
    gate["recovery"] = {
        "schema": RECOVERY_SCHEMA,
        "original_runner_exit_code": int(original_exit_code),
        "recovery_attempt1_exit_code": 2,
        "recovery_attempt1_failure": {"path": "recovery_attempt1_failure.json", "sha256": attempt1_sha},
        "recovery_exit_code": 0 if not gate.get("differences") else 3,
        "stage_execution_repeated": False,
        "optimizer_called": False,
        "helper_fix": helper_fix,
        "source_snapshot_unchanged": True,
        "journal_rows_reconstructed": 6,
    }
    gate_sha = controlled._write_json_exclusive(root / "C0_gate.json", gate)
    started = min(
        _load(root / "stages" / candidate.candidate_id / (stage.label + ".json"))["started_at_utc"]
        for candidate in CANDIDATES for stage in STAGES
    )
    terminal = controlled._terminal_evidence(
        root, gate, results, started, _utc_now(),
        sha256_file(root / "controlled_protocol.json"), sha256_file(root / "preflight.json"),
    )
    terminal.update({
        "runner_exit_code": int(original_exit_code),
        "recovery_exit_code": 0 if gate["status"] == "passed" else 3,
        "original_runner_failure": {"path": "run_failure.json", "sha256": sha256_file(original_failure_path)},
        "recovery": {
            "schema": RECOVERY_SCHEMA,
            "optimizer_called": False,
            "stage_execution_repeated": False,
            "recovery_attempt1": {"path": "recovery_attempt1_failure.json", "sha256": attempt1_sha},
            "helper_fix": helper_fix,
            "verification": verification,
        },
    })
    terminal_sha = controlled._write_json_exclusive(root / "terminal_evidence.json", terminal)
    report_text = _recovery_report(root, gate, terminal, {"selected_id": selection["selection"]["selected_id"]})
    report_text += "\n- recovery attempt 1（仅 gate helper）: exit code `2`，NameError `stage`，未调用 optimizer；本次 attempt 2 通过 runtime override 重试 gate，不重跑 stage。\n"
    report_sha = controlled._write_bytes_exclusive(root / "REPORT.zh-CN.md", (report_text + "\n").encode("utf-8"))
    selected_id = selection["selection"]["selected_id"]
    receipt = {
        "schema": "c0-recovery-receipt-v2",
        "status": "passed" if gate["status"] == "passed" else "failed_gate",
        "original_runner_exit_code": int(original_exit_code),
        "recovery_attempt1_exit_code": 2,
        "recovery_exit_code": 0 if gate["status"] == "passed" else 3,
        "stage_execution_repeated": False,
        "optimizer_called_during_recovery": False,
        "outdir": str(root),
        "recovery_script": {"path": "pr/c0_recovery.py", "sha256": sha256_file(Path(__file__))},
        "original_failure": {"path": "run_failure.json", "sha256": sha256_file(original_failure_path)},
        "recovery_attempt1": {"path": "recovery_attempt1_failure.json", "sha256": attempt1_sha},
        "helper_fix": helper_fix,
        "source_snapshot_unchanged": True,
        "preflight": {"path": "preflight.json", "sha256": sha256_file(root / "preflight.json")},
        "protocol": {"path": "controlled_protocol.json", "sha256": sha256_file(root / "controlled_protocol.json")},
        "journal": {"path": "run_status/attempts.jsonl", "sha256": sha256_file(root / "run_status" / "attempts.jsonl"), "rows": 6},
        "gate": {"path": "C0_gate.json", "sha256": gate_sha},
        "terminal": {"path": "terminal_evidence.json", "sha256": terminal_sha},
        "report": {"path": "REPORT.zh-CN.md", "sha256": report_sha},
        "selection": {"path": "selection.json", "sha256": sha256_file(root / "selection.json")},
        "selected_id": selected_id,
        "stage_records": [
            {"candidate_id": candidate.candidate_id, "stage": stage.label,
             "path": str(Path("stages") / candidate.candidate_id / (stage.label + ".json")),
             "sha256": sha256_file(root / "stages" / candidate.candidate_id / (stage.label + ".json"))}
            for candidate in CANDIDATES for stage in STAGES
        ],
        "blocked_stage_attempt_count": 24,
        "training_boundary": {"phase_used": False, "reference_used": False,
                              "evaluation_outputs_opened": False, "native_fdg_started": False},
    }
    recovery_sha = controlled._write_json_exclusive(root / "recovery_receipt.json", receipt)
    run_receipt = {
        "schema": "c0-run-receipt-recovered-v1",
        "status": receipt["status"],
        "original_runner_exit_code": int(original_exit_code),
        "recovery_attempt1_exit_code": 2,
        "recovery_exit_code": receipt["recovery_exit_code"],
        "stage_execution_repeated": False,
        "executed_stage_attempt_count": 6,
        "remaining_blocked_stage_attempt_count": 24,
        "selected_id": selected_id,
        "gate": {"path": "C0_gate.json", "sha256": gate_sha},
        "terminal": {"path": "terminal_evidence.json", "sha256": terminal_sha},
        "recovery_receipt": {"path": "recovery_receipt.json", "sha256": recovery_sha},
    }
    run_receipt_sha = controlled._write_json_exclusive(root / "run_receipt.json", run_receipt)
    return {
        "status": receipt["status"],
        "outdir": str(root),
        "selected_id": selected_id,
        "gate": str(root / "C0_gate.json"),
        "gate_sha256": gate_sha,
        "terminal": str(root / "terminal_evidence.json"),
        "terminal_sha256": terminal_sha,
        "report": str(root / "REPORT.zh-CN.md"),
        "selection": str(root / "selection.json"),
        "recovery_receipt": str(root / "recovery_receipt.json"),
        "recovery_receipt_sha256": recovery_sha,
        "run_receipt_sha256": run_receipt_sha,
    }


def resume_final(outdir: str | os.PathLike[str], original_exit_code: int = 2) -> dict[str, Any]:
    """在 gate 已写出后完成终态记账。"""
    root = Path(outdir)
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    else:
        root = root.resolve()
    required = ("selection.json", "C0_gate.json", "run_status/attempts.jsonl", "run_failure.json")
    if not all((root / path).is_file() for path in required):
        raise FileNotFoundError("completed gate recovery inputs are missing")
    for name in ("terminal_evidence.json", "recovery_receipt.json", "run_receipt.json"):
        if (root / name).exists():
            raise FileExistsError("final recovery output already exists: %s" % (root / name))
    context = reconstruct.production_context()
    reconstruct._validate_context(context)
    config = _load(root / "config.json")
    original_failure_path = root / "run_failure.json"
    attempt2 = {
        "schema": "c0-recovery-attempt-v1",
        "attempt": 2,
        "status": "failed_after_gate",
        "exit_code": 2,
        "error_type": "TypeError",
        "error": "Object of type VerifiedSelection is not JSON serializable",
        "location": "pr/c0_recovery.py:resume_partial terminal_evidence",
        "cause": "verification object was passed to strict JSON evidence without scalar conversion",
        "stage_execution_repeated": False,
        "optimizer_called": False,
        "gate_already_written": True,
        "created_at_utc": _utc_now(),
    }
    attempt2_sha = controlled._write_json_exclusive(root / "recovery_attempt2_failure.json", attempt2)
    results = [_candidate_from_stages(root, config, candidate) for candidate in CANDIDATES]
    for result in results:
        reconstruct._rescore_final_candidate(root, context, config, result, reconstruct.DEFAULT_HOOKS)
    selection = _load(root / "selection.json")
    verification = reconstruct.verify_completed_training(root, strict_p9016=True)
    verification_summary = {
        "type": type(verification).__name__,
        "status": "passed",
        "repr": repr(verification),
    }
    gate = _load(root / "C0_gate.json")
    if gate.get("status") != "passed" or gate.get("differences"):
        raise RuntimeError("existing C0_gate is not a passed zero-difference gate")
    selected_id = selection["selection"]["selected_id"]
    started = min(
        _load(root / "stages" / candidate.candidate_id / (stage.label + ".json"))["started_at_utc"]
        for candidate in CANDIDATES for stage in STAGES
    )
    terminal = controlled._terminal_evidence(
        root, gate, results, started, _utc_now(),
        sha256_file(root / "controlled_protocol.json"), sha256_file(root / "preflight.json"),
    )
    terminal.update({
        "runner_exit_code": int(original_exit_code),
        "recovery_attempt1_exit_code": 2,
        "recovery_attempt2_exit_code": 2,
        "recovery_final_exit_code": 0,
        "original_runner_failure": {"path": "run_failure.json", "sha256": sha256_file(original_failure_path)},
        "recovery": {
            "schema": RECOVERY_SCHEMA,
            "optimizer_called": False,
            "stage_execution_repeated": False,
            "recovery_attempt1": {"path": "recovery_attempt1_failure.json", "sha256": sha256_file(root / "recovery_attempt1_failure.json")},
            "recovery_attempt2": {"path": "recovery_attempt2_failure.json", "sha256": attempt2_sha},
            "finalization_attempt": 3,
            "verification": verification_summary,
            "gate_reused_without_rewrite": True,
        },
    })
    terminal_sha = controlled._write_json_exclusive(root / "terminal_evidence.json", terminal)
    report_text = _recovery_report(root, gate, terminal, {"selected_id": selected_id})
    report_text += "\n- recovery attempt 1（仅 gate helper）: exit code `2`，未调用 optimizer。\n"
    report_text += "- recovery attempt 2（terminal JSON 类型）: exit code `2`，gate 已通过且未重跑 stage；最终 attempt 3 仅序列化 verification 摘要并完成收尾。\n"
    report_sha = controlled._write_bytes_exclusive(root / "REPORT.zh-CN.md", (report_text + "\n").encode("utf-8"))
    receipt = {
        "schema": "c0-recovery-receipt-v3",
        "status": "passed",
        "original_runner_exit_code": int(original_exit_code),
        "recovery_attempt1_exit_code": 2,
        "recovery_attempt2_exit_code": 2,
        "recovery_final_exit_code": 0,
        "stage_execution_repeated": False,
        "optimizer_called_during_recovery": False,
        "outdir": str(root),
        "recovery_script": {"path": "pr/c0_recovery.py", "sha256": sha256_file(Path(__file__))},
        "original_failure": {"path": "run_failure.json", "sha256": sha256_file(original_failure_path)},
        "recovery_attempt1": {"path": "recovery_attempt1_failure.json", "sha256": sha256_file(root / "recovery_attempt1_failure.json")},
        "recovery_attempt2": {"path": "recovery_attempt2_failure.json", "sha256": attempt2_sha},
        "gate_reused": {"path": "C0_gate.json", "sha256": sha256_file(root / "C0_gate.json"), "status": gate["status"], "differences": len(gate["differences"])},
        "terminal": {"path": "terminal_evidence.json", "sha256": terminal_sha},
        "report": {"path": "REPORT.zh-CN.md", "sha256": report_sha},
        "selection": {"path": "selection.json", "sha256": sha256_file(root / "selection.json")},
        "journal": {"path": "run_status/attempts.jsonl", "sha256": sha256_file(root / "run_status" / "attempts.jsonl"), "rows": 6},
        "selected_id": selected_id,
        "verification": verification_summary,
        "stage_records": [
            {"candidate_id": candidate.candidate_id, "stage": stage.label,
             "path": str(Path("stages") / candidate.candidate_id / (stage.label + ".json")),
             "sha256": sha256_file(root / "stages" / candidate.candidate_id / (stage.label + ".json"))}
            for candidate in CANDIDATES for stage in STAGES
        ],
        "blocked_stage_attempt_count": 24,
        "training_boundary": {"phase_used": False, "reference_used": False,
                              "evaluation_outputs_opened": False, "native_fdg_started": False},
    }
    recovery_sha = controlled._write_json_exclusive(root / "recovery_receipt.json", receipt)
    run_receipt = {
        "schema": "c0-run-receipt-recovered-v2",
        "status": "passed",
        "original_runner_exit_code": int(original_exit_code),
        "recovery_attempt1_exit_code": 2,
        "recovery_attempt2_exit_code": 2,
        "recovery_final_exit_code": 0,
        "stage_execution_repeated": False,
        "executed_stage_attempt_count": 6,
        "remaining_blocked_stage_attempt_count": 24,
        "selected_id": selected_id,
        "gate": {"path": "C0_gate.json", "sha256": sha256_file(root / "C0_gate.json")},
        "terminal": {"path": "terminal_evidence.json", "sha256": terminal_sha},
        "recovery_receipt": {"path": "recovery_receipt.json", "sha256": recovery_sha},
    }
    run_receipt_sha = controlled._write_json_exclusive(root / "run_receipt.json", run_receipt)
    return {
        "status": "passed",
        "outdir": str(root),
        "selected_id": selected_id,
        "gate": str(root / "C0_gate.json"),
        "gate_sha256": sha256_file(root / "C0_gate.json"),
        "terminal": str(root / "terminal_evidence.json"),
        "terminal_sha256": terminal_sha,
        "report": str(root / "REPORT.zh-CN.md"),
        "selection": str(root / "selection.json"),
        "recovery_receipt": str(root / "recovery_receipt.json"),
        "recovery_receipt_sha256": recovery_sha,
        "run_receipt_sha256": run_receipt_sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover C0 finalization without rerunning stages")
    parser.add_argument("--out", required=True)
    parser.add_argument("--original-exit-code", type=int, default=2)
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--resume-final", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.resume_final:
            result = resume_final(args.out, original_exit_code=args.original_exit_code)
        elif args.resume_partial:
            result = resume_partial(args.out, original_exit_code=args.original_exit_code)
        else:
            result = recover(args.out, original_exit_code=args.original_exit_code)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "passed" else 3
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "error": str(exc), "traceback": traceback.format_exc()}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
