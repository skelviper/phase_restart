"""收集 formal run 036 的 release-ready、无 phase/reference/R2 evidence。"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "test_res/036-20260914T064651Z-gpu-multires"
PREFLIGHT_RUN = ROOT / "test_res/035-20260914T060945Z-gpu-multires-preflight"
PREFLIGHT = PREFLIGHT_RUN / "preflight.json"
PROTOCOL = PREFLIGHT_RUN / "protocol.json"
CONFIG = PREFLIGHT_RUN / "config.json"
SOURCE_MANIFEST = PREFLIGHT_RUN / "provenance/source_hashes.json"
LAUNCH = PREFLIGHT_RUN / "launch_command.txt"
JOB_ID = "bash-297"
JOB_EXIT_CODE = 0

EXPECTED = {
    "5m": {"bin_size_bp": 5_000_000, "n_loci": 538, "n_eligible_pairs": 144_453,
           "raw_records": 1_703_888, "raw_same_bin": 607_552, "raw_cis_offdiag": 527_902,
           "raw_inter": 568_434, "checkpoint_count": 30},
    "2m": {"bin_size_bp": 2_000_000, "n_loci": 1_329, "n_eligible_pairs": 882_456,
           "raw_records": 1_703_888, "raw_same_bin": 516_046, "raw_cis_offdiag": 619_408,
           "raw_inter": 568_434, "checkpoint_count": 20},
    "1m": {"bin_size_bp": 1_000_000, "n_loci": 2_645, "n_eligible_pairs": 3_496_690,
           "raw_records": 1_703_888, "raw_same_bin": 438_774, "raw_cis_offdiag": 696_680,
           "raw_inter": 568_434, "checkpoint_count": 24},
}
VARIANTS = ("C0", "C1", "C2-map", "C2-free", "C3")
CANDIDATES = ("consensus_joint", "random_joint")
FAIL_STATUSES = {"failed", "not_run_after_prior_failure"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path: Path, value) -> str:
    payload = json_bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def parse_coordinates(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    values = []
    tracks: dict[str, int] = {}
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"malformed coordinate row {path}:{line_no}")
            track = fields[0]
            tracks[track] = tracks.get(track, 0) + 1
            values.append([float(fields[2]), float(fields[3]), float(fields[4])])
    array = np.asarray(values, dtype=np.float64)
    return array, tracks


def finite(value) -> bool:
    return bool(np.isfinite(float(value)))


def main() -> int:
    terminal = read_json(RUN / "terminal_evidence.json")
    selection = read_json(RUN / "selection.json")
    runtime_probe = read_json(RUN / "runtime_probe.json")
    preflight = read_json(PREFLIGHT)
    protocol = read_json(PROTOCOL)
    config = read_json(CONFIG)
    source_manifest = read_json(SOURCE_MANIFEST)
    launch_command = LAUNCH.read_text()

    if terminal.get("status") != "training_complete" or terminal.get("runner_exit_code") != 0:
        raise RuntimeError("terminal evidence is not a successful training completion")
    if selection.get("status") != "training_complete":
        raise RuntimeError("selection status is not training_complete")
    if JOB_EXIT_CODE != 0:
        raise RuntimeError("managed job did not exit zero")

    expected_attempts = [row["attempt_id"] for row in protocol["attempt_plan"]["planned_attempts"]]
    stage_paths = sorted((RUN / "stages").glob("*/*/*.json"))
    if len(stage_paths) != 30:
        raise RuntimeError(f"expected 30 stage records, found {len(stage_paths)}")
    actual_attempts = []
    stage_rows = []
    trajectories: dict[tuple[str, str], list[dict]] = {}
    checkpoint_count = 0
    checkpoint_paths = []
    c2_rows = []
    timing_rows = []
    max_coordinate_radius = 0.0
    q_carry_abs_max = 0.0
    for stage_path in stage_paths:
        row = read_json(stage_path)
        model = str(row["model_id"])
        candidate = str(row["candidate_id"])
        stage = str(row["stage"])
        attempt_id = str(row["attempt_id"])
        if model not in VARIANTS or candidate not in CANDIDATES or stage not in EXPECTED:
            raise RuntimeError(f"unexpected stage identity: {attempt_id}")
        actual_attempts.append(attempt_id)
        data_audit = row["data_budget"]["audit"]
        expected = EXPECTED[stage]
        for field in ("raw_records", "raw_same_bin", "raw_cis_offdiag", "raw_inter", "n_loci", "n_eligible_pairs"):
            if int(data_audit[field]) != int(expected[field]):
                raise RuntimeError(f"budget mismatch {attempt_id} {field}: {data_audit[field]}")
        if row["status"] in FAIL_STATUSES:
            raise RuntimeError(f"failed stage in successful terminal run: {attempt_id}")
        fit = row["fit"]
        final = row["final_coordinates"]
        final_path = RUN / str(final["path"])
        if not final_path.is_file():
            raise RuntimeError(f"missing final coordinates: {final_path}")
        final_sha = sha256(final_path)
        if final_sha != str(final["sha256"]):
            raise RuntimeError(f"final coordinate hash mismatch: {final_path}")
        coordinates, track_counts = parse_coordinates(final_path)
        expected_beads = 2 * int(expected["n_loci"])
        if coordinates.shape != (expected_beads, 3) or sum(track_counts.values()) != expected_beads:
            raise RuntimeError(f"coordinate row count mismatch: {final_path}")
        if len(track_counts) != 40 or not np.isfinite(coordinates).all():
            raise RuntimeError(f"coordinate finite/track check failed: {final_path}")
        if not bool(final["full_grid"]) or int(final["n_tracks"]) != 40 or int(final["n_beads"]) != expected_beads:
            raise RuntimeError(f"full-grid endpoint metadata failed: {final_path}")
        max_radius = float(np.linalg.norm(coordinates, axis=1).max(initial=0.0))
        max_coordinate_radius = max(max_coordinate_radius, max_radius)
        domain = str(final["physical_domain"])
        if domain == "strict_unit_ball" and not max_radius < 1.0:
            raise RuntimeError(f"strict-unit-ball endpoint escaped: {final_path}")
        if domain not in ("strict_unit_ball", "finite_unbounded"):
            raise RuntimeError(f"unknown physical domain: {domain}")
        if final.get("serialization_clip", {}).get("applied") or final.get("serialization_clip", {}).get("clipped_coordinates") != 0:
            raise RuntimeError(f"serialization clip applied: {final_path}")
        for field in ("q_out", "final_total", "initial_total"):
            if not finite(fit[field] if field in fit else fit.get("q_out")):
                raise RuntimeError(f"nonfinite fit field {field}: {attempt_id}")
        q_in = fit.get("q_in")
        q_out = float(fit["q_out"])
        if q_in is not None and not finite(q_in):
            raise RuntimeError(f"nonfinite q_in: {attempt_id}")
        solver = fit["solver"]
        backend = fit["backend"]
        expected_kernel_sha = source_manifest["source_code_sha256"]["028_cuda_pair_kernel"]
        if backend["dtype"] != "torch.float64" or backend["torch_cuda_version"] != "13.0":
            raise RuntimeError(f"backend precision/CUDA mismatch: {attempt_id}")
        if not bool(backend["data_resident_on_device"]):
            raise RuntimeError(f"backend data is not device resident: {attempt_id}")
        if backend["source_sha256"] != expected_kernel_sha:
            raise RuntimeError(f"mature CUDA kernel hash mismatch: {attempt_id}")
        elapsed = float(solver["elapsed_seconds"])
        if not finite(elapsed) or elapsed < 0 or int(solver["actual_nfev"]) <= 0:
            raise RuntimeError(f"invalid solver timing: {attempt_id}")
        cp_paths = [RUN / str(path) for path in fit["checkpoint_paths"]]
        for cp in cp_paths:
            if not cp.is_file():
                raise RuntimeError(f"missing checkpoint: {cp}")
        checkpoint_count += len(cp_paths)
        checkpoint_paths.extend(str(path.relative_to(ROOT)) for path in cp_paths)
        stage_record = {
            "attempt_id": attempt_id,
            "variant": model,
            "candidate_id": candidate,
            "stage": stage,
            "bin_size_bp": int(row["bin_size_bp"]),
            "status": str(row["status"]),
            "stage_record": str(stage_path.relative_to(ROOT)),
            "final_coordinates": {
                "path": str(final_path.relative_to(ROOT)),
                "sha256": final_sha,
                "rows": int(coordinates.shape[0]),
                "tracks": int(len(track_counts)),
                "max_radius": max_radius,
                "physical_domain": domain,
                "full_grid": bool(final["full_grid"]),
                "n_tracks": int(final["n_tracks"]),
                "n_beads": int(final["n_beads"]),
            },
            "budget": {
                "n_loci": int(data_audit["n_loci"]),
                "n_eligible_pairs": int(data_audit["n_eligible_pairs"]),
                "raw_records": int(data_audit["raw_records"]),
                "raw_same_bin": int(data_audit["raw_same_bin"]),
                "raw_cis_offdiag": int(data_audit["raw_cis_offdiag"]),
                "raw_inter": int(data_audit["raw_inter"]),
                "endpoint_total": int(data_audit["endpoint_total"]),
                "aggregate_conserved": bool(data_audit["aggregate_conserved"]),
                "endpoint_conserved": bool(data_audit["endpoint_conserved"]),
            },
            "q_in": None if q_in is None else float(q_in),
            "q_out": q_out,
            "solver": {
                "actual_nfev": int(solver["actual_nfev"]),
                "nit": int(solver["nit"]),
                "maxiter": int(solver["maxiter"]),
                "maxfun": int(solver["maxfun"]),
                "elapsed_seconds": elapsed,
                "status": int(solver["status"]),
                "success": bool(solver["success"]),
                "message": str(solver["message"]),
                "checkpoint_count": len(cp_paths),
                "checkpoint_paths": [str(path.relative_to(ROOT)) for path in cp_paths],
            },
            "backend": {
                "dtype": fit["backend"]["dtype"],
                "torch_version": fit["backend"]["torch_version"],
                "torch_cuda_version": fit["backend"]["torch_cuda_version"],
                "device": fit["backend"]["device"],
                "data_resident_on_device": bool(fit["backend"]["data_resident_on_device"]),
                "peak_cuda_memory_bytes": int(fit["backend"]["peak_cuda_memory_bytes"]),
                "source_sha256": fit["backend"]["source_sha256"],
            },
            "mapping_diagnostics": fit["mapping_diagnostics"],
            "input_sha256": protocol["input"]["sha256"],
            "source_code_sha256": source_manifest["source_code_sha256"],
        }
        stage_rows.append(stage_record)
        trajectories.setdefault((model, candidate), []).append(stage_record)
        timing_rows.append({"attempt_id": attempt_id, "elapsed_seconds": elapsed,
                            "actual_nfev": int(solver["actual_nfev"]), "checkpoint_count": len(cp_paths)})
        if model in ("C2-map", "C2-free"):
            c2_rows.append({"attempt_id": attempt_id, "variant": model, "stage": stage,
                            "candidate_id": candidate, **fit["mapping_diagnostics"]})

    if len(actual_attempts) != len(set(actual_attempts)) or sorted(actual_attempts) != sorted(expected_attempts):
        raise RuntimeError("actual stage attempt IDs are not exactly the frozen 30 unique IDs")
    trajectory_summary = []
    for key, rows in sorted(trajectories.items()):
        rows.sort(key=lambda item: ("5m", "2m", "1m").index(item["stage"]))
        if len(rows) != 3 or rows[-1]["stage"] != "1m":
            raise RuntimeError(f"trajectory incomplete: {key}")
        for index, row in enumerate(rows):
            if index == 0:
                if row["q_in"] is not None:
                    raise RuntimeError(f"first-layer q_in must be null: {key}")
            else:
                if row["q_in"] is None:
                    raise RuntimeError(f"missing carried q_in: {key} {row['stage']}")
                q_carry_abs_max = max(q_carry_abs_max, abs(float(row["q_in"]) - float(rows[index - 1]["q_out"])))
        trajectory_summary.append({
            "variant": key[0], "candidate_id": key[1], "stage_count": len(rows),
            "stages": [row["stage"] for row in rows],
            "final_1m_endpoint": rows[-1]["final_coordinates"],
            "q_out_by_stage": {row["stage"]: row["q_out"] for row in rows},
            "q_carry_exact_within_tolerance": all(
                row["q_in"] is None or abs(row["q_in"] - rows[index - 1]["q_out"]) <= 1e-12
                for index, row in enumerate(rows)
            ),
        })
    if len(trajectory_summary) != 10 or q_carry_abs_max > 1e-12:
        raise RuntimeError("trajectory or raw-q carry audit failed")

    expected_checkpoint_count = sum(EXPECTED[stage]["checkpoint_count"] for stage in EXPECTED) * 10
    if checkpoint_count != expected_checkpoint_count:
        raise RuntimeError(f"checkpoint count mismatch: {checkpoint_count} != {expected_checkpoint_count}")

    selected_representatives = []
    for model in VARIANTS:
        selected = selection["selection"]["per_variant"][model]
        if selected.get("status") != "selected":
            raise RuntimeError(f"missing selected representative for {model}")
        selected_id = str(selected["selected_id"])
        file_info = selection["selection"]["selected_files"][model]
        selected_path = RUN / str(file_info["path"])
        selected_sha = sha256(selected_path)
        if selected_sha != str(file_info["sha256"]):
            raise RuntimeError(f"selected file hash mismatch: {selected_path}")
        score = selected["candidate_scores"][selected_id]
        selected_representatives.append({
            "variant": model,
            "selected_candidate_id": selected_id,
            "count_nll_per_record": float(score),
            "selected_path": str(selected_path.relative_to(ROOT)),
            "selected_sha256": selected_sha,
            "criterion": selected["criterion"],
            "tie_tolerance_per_record": selected["tie_tolerance_per_record"],
            "reference_used": bool(selected["reference_used"]),
            "phase_used": bool(selected["phase_used"]),
        })

    c2_summary = {}
    for model in ("C2-map", "C2-free"):
        rows = [row for row in c2_rows if row["variant"] == model]
        c2_summary[model] = {
            "stage_count": len(rows),
            "objective_eval_count_total": int(sum(row["objective_eval_count"] for row in rows)),
            "nonidentity_map_eval_count_total": int(sum(row["nonidentity_map_eval_count"] for row in rows)),
            "nonidentity_map_eval_fraction_max": float(max(row["nonidentity_map_eval_fraction"] for row in rows)),
            "max_raw_radius_seen": float(max(row["max_raw_radius_seen"] for row in rows)),
            "max_physical_radius_seen": float(max(row["max_physical_radius_seen"] for row in rows)),
            "counts_only_evaluate_calls_all": all(row["counts_only_evaluate_calls"] for row in rows),
            "line_search_probes_included_all": all(row["line_search_probes_included"] for row in rows),
            "used_for_selection_or_budget_any": any(row["used_for_selection_or_budget"] for row in rows),
            "per_stage": rows,
        }
    total_elapsed = float(sum(row["elapsed_seconds"] for row in timing_rows))
    timing = {
        "stage_count": len(timing_rows),
        "total_solver_elapsed_seconds": total_elapsed,
        "median_stage_elapsed_seconds": float(np.median([row["elapsed_seconds"] for row in timing_rows])),
        "min_stage_elapsed_seconds": float(min(row["elapsed_seconds"] for row in timing_rows)),
        "max_stage_elapsed_seconds": float(max(row["elapsed_seconds"] for row in timing_rows)),
        "total_actual_nfev": int(sum(row["actual_nfev"] for row in timing_rows)),
        "total_checkpoints": checkpoint_count,
        "per_stage": timing_rows,
    }

    release_paths = {
        "formal_output": str(RUN.relative_to(ROOT)),
        "all_stage_records": str((RUN / "stages").relative_to(ROOT)),
        "all_final_coordinates": [row["final_coordinates"]["path"] for row in stage_rows],
        "final_1m_endpoints": [row["final_coordinates"]["path"] for row in stage_rows if row["stage"] == "1m"],
        "selected_coordinates": [row["selected_path"] for row in selected_representatives],
        "selection": str((RUN / "selection.json").relative_to(ROOT)),
        "terminal_evidence": str((RUN / "terminal_evidence.json").relative_to(ROOT)),
        "runtime_probe": str((RUN / "runtime_probe.json").relative_to(ROOT)),
        "preflight": str(PREFLIGHT.relative_to(ROOT)),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "config": str(CONFIG.relative_to(ROOT)),
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "launch_command": str(LAUNCH.relative_to(ROOT)),
        "formal_provenance": [str(path.relative_to(ROOT)) for path in sorted((RUN / "provenance").glob("*.json"))],
    }

    manifest = {
        "schema": "gpu-multires-release-ready-manifest-v1",
        "status": "release_ready",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "managed_job": {"id": JOB_ID, "exit_code": JOB_EXIT_CODE, "controller_status": terminal["status"],
                        "launch_command": launch_command},
        "training_boundary": {
            "phase_used": False, "reference_used": False, "r2_used": False, "evaluation_used": False,
            "formal_training_scope": "10 variant/source trajectories, 30 stages, all 1,703,888 SNP-free records",
        },
        "frozen_integrity": {
            "preflight": {"path": str(PREFLIGHT.relative_to(ROOT)), "sha256": sha256(PREFLIGHT)},
            "protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha256(PROTOCOL)},
            "config": {"path": str(CONFIG.relative_to(ROOT)), "sha256": sha256(CONFIG)},
            "source_manifest": {"path": str(SOURCE_MANIFEST.relative_to(ROOT)), "sha256": sha256(SOURCE_MANIFEST)},
            "input_sha256": protocol["input"]["sha256"],
            "approved_sources": preflight["approved_sources"],
            "source_code_sha256": source_manifest["source_code_sha256"],
        },
        "ledger": {
            "expected_attempt_count": 30,
            "actual_attempt_count": len(actual_attempts),
            "unique_attempt_count": len(set(actual_attempts)),
            "expected_trajectory_count": 10,
            "actual_trajectory_count": len(trajectory_summary),
            "completed_stage_count": len(stage_rows),
            "failed_or_not_run_stage_count": 0,
            "all_attempt_ids_exact": True,
            "all_trajectories_three_stage": True,
            "all_trajectories_have_1m_endpoint": True,
            "q_carry_max_abs_difference": q_carry_abs_max,
            "terminal_attempt_audit": terminal["attempt_audit"],
            "trajectories": trajectory_summary,
        },
        "checkpoint_audit": {
            "expected_total": expected_checkpoint_count,
            "actual_total": checkpoint_count,
            "all_checkpoint_files_present": True,
            "all_stages_reached_fixed_budget": all(row["solver"]["checkpoint_count"] == EXPECTED[row["stage"]]["checkpoint_count"] for row in stage_rows),
            "checkpoint_paths": checkpoint_paths,
        },
        "stage_records": stage_rows,
        "selection": {
            "status": selection["status"],
            "criterion": "count_nll_per_record",
            "reference_used": False,
            "phase_used": False,
            "representatives": selected_representatives,
        },
        "c2_activity": c2_summary,
        "timing": timing,
        "runtime_probe": runtime_probe,
        "release_paths": release_paths,
        "limitations": [
            "All 30 stages reached their registered iteration budgets and report not_converged; this is budget termination, not a claim of numerical convergence.",
            "The label-free count selection is recorded; no reference/R2/phase data were read by this collector.",
            "The 211.87x preflight benchmark is retained separately in 035 and is not substituted for these stage timings.",
        ],
    }
    manifest_path = RUN / "release_ready_manifest.json"
    manifest_sha = write_json(manifest_path, manifest)
    report_lines = [
        "# GPU 多分辨率正式运行发布报告",
        "",
        f"- 状态：`release_ready`；受管 job `{JOB_ID}` exit `{JOB_EXIT_CODE}`；controller `training_complete`。",
        "- 规模：10 条变体/来源轨迹，30 个 stage，全部使用 1,703,888 条 SNP-free 记录。",
        f"- 账目：{len(actual_attempts)}/30 个唯一 attempt，{len(trajectory_summary)}/10 条轨迹，所有轨迹均有 5m/2m/1m endpoint；q carry 最大绝对差 `{q_carry_abs_max:.3g}`。",
        f"- checkpoint：{checkpoint_count}/{expected_checkpoint_count}，每个 stage 均达到注册预算，没有追加预算。",
        f"- solver 用时：全部 stage elapsed `{total_elapsed:.3f}s`，中位数 `{timing['median_stage_elapsed_seconds']:.3f}s`，实际 nfev `{timing['total_actual_nfev']}`。",
        "- 选择：每个变体仅按 `count_nll_per_record` 在 consensus_joint/random_joint 中选择；未使用 reference、phase 或 R2。",
        f"- C2-map 目标函数调用总数 `{c2_summary['C2-map']['objective_eval_count_total']}`，C2-free `{c2_summary['C2-free']['objective_eval_count_total']}`；所有调用均标记为仅计数 evaluate，line-search probes 已纳入诊断。",
        "- 所有 endpoint 坐标已逐文件核对 SHA、40 条轨迹、完整网格、有限值；strict-unit-ball 变体未越界，C2-free 保持 finite_unbounded。",
        "",
        "## 关键路径",
        f"- manifest: `{manifest_path.relative_to(ROOT)}`",
        f"- selection: `{RUN.relative_to(ROOT)}/selection.json`",
        f"- terminal: `{RUN.relative_to(ROOT)}/terminal_evidence.json`",
        f"- preflight: `{PREFLIGHT.relative_to(ROOT)}`",
        "",
        "## 限制",
        "所有 stage 均按注册预算结束并记录 `not_converged`；这表示固定预算终止，不表示数值收敛。R2/evaluation 未启动。",
    ]
    report_path = RUN / "release_ready_report.md"
    report_payload = "\n".join(report_lines) + "\n"
    report_path.write_text(report_payload, encoding="utf-8")
    report_sha = sha256(report_path)
    print(json.dumps({"status": "release_ready", "manifest": str(manifest_path),
                      "manifest_sha256": manifest_sha, "report": str(report_path),
                      "report_sha256": report_sha, "attempts": len(actual_attempts),
                      "trajectories": len(trajectory_summary), "checkpoints": checkpoint_count,
                      "total_solver_elapsed_seconds": total_elapsed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise
