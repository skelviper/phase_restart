#!/usr/bin/env python3
"""低频 gated post-020 真实运行 controller。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from pr import allele_experiment as ex

ROOT = Path(__file__).resolve().parents[1]
TEST_RES = ROOT / "test_res"
PRE025 = TEST_RES / "025-20260913_135100-random-native-fullgrid-preflight"
PRE025_VALIDATION = PRE025 / "validation" / "POST020_ALLELE_ABLATION_NATIVE_VALIDATION.json"
D_ROOT = TEST_RES / "028-20260913_151456-020-gpu-independent"
PROTOCOL_SHA = "b281037946775ee4271dbf33dd7e4b17dba5ddf4f9bd49eaf618f7c499c5361f"
SOURCE = ROOT / "inputs" / "P9016.snpfree.pairs.gz"
STARTS = PRE025 / "bundles"
WAIT_ROOT = TEST_RES / "POST020_REAL_CONTROLLER"
WAIT_LOG = WAIT_ROOT / "waiting.jsonl"
STATE = WAIT_ROOT / "status.json"
POLL_SECONDS = 30


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def record(event: str, **fields) -> dict:
    WAIT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "post020-real-controller-v1", "event": event,
               "utc": utc_now(), **fields}
    with WAIT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    STATE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def check_025() -> dict:
    if not PRE025_VALIDATION.is_file():
        return {"ready": False, "reason": "025 validation report missing"}
    try:
        report = read_json(PRE025_VALIDATION)
        if report.get("schema") != "POST020_ALLELE_ABLATION_NATIVE_VALIDATION-v1":
            return {"ready": False, "reason": "025 validation schema mismatch"}
        if report.get("status") != "validated":
            return {"ready": False, "reason": "025 validation status is not validated"}
        required_checks = (
            "3dg_17g_exact_and_full_mapping", "actual_stderr_iter_1000",
            "attempt_started_then_terminal", "global_center_and_max_radius_0_8",
            "native_headers_1000", "raw_arrays_finite_and_mapped",
            "registry_completion_record", "source_binary_input_graph_hashes",
            "starts_pairwise_distinct", "x0_arrays_finite_shape_2x2645x3",
            "x0_pairwise_distinct",
        )
        if any(report.get("checks", {}).get(key) is not True for key in required_checks):
            return {"ready": False, "reason": "025 validation checks are incomplete"}
        contract = report.get("native_contract", {})
        if contract != {
            "backend": "CPU", "n_beads": 5290, "n_iter": 1000,
            "n_raw": 1265114, "one_call_per_bundle": True,
            "retry_allowed": False, "source": None, "target_radius": 10.0,
            "threads": 1,
        }:
            return {"ready": False, "reason": "025 native contract mismatch"}
        for item in report.get("bundle_results", []):
            bid = str(item["bundle_id"])
            npz = PRE025 / "bundles" / bid / "x0_normalized.npz"
            three = PRE025 / "bundles" / bid / "x0_normalized.3dg"
            if sha256(npz) != item["x0_normalized_npz_sha256"] or sha256(three) != item["x0_normalized_3dg_sha256"]:
                return {"ready": False, "reason": "%s x0 hash changed" % bid}
            with np.load(npz, allow_pickle=False) as payload:
                coords = np.asarray(payload["coordinates"])
                if coords.shape != (2, 2645, 3) or coords.dtype != np.float64 or not np.isfinite(coords).all():
                    return {"ready": False, "reason": "%s x0 shape/finite check failed" % bid}
                max_radius = float(np.linalg.norm(coords, axis=2).max())
                if abs(max_radius - 0.8) > 5e-12:
                    return {"ready": False, "reason": "%s x0 max radius changed" % bid}
        return {"ready": True, "validation_sha256": sha256(PRE025_VALIDATION),
                "validation_path": str(PRE025_VALIDATION)}
    except Exception as exc:
        return {"ready": False, "reason": "%s: %s" % (type(exc).__name__, exc)}


def check_d() -> dict:
    try:
        checks = {}
        for name in ("initialization_validation.json", "real_1m_checkpoint_validation.json"):
            payload = read_json(D_ROOT / "logs" / name)
            checks[name] = payload.get("status") == "passed" and payload.get("phase_or_reference_opened") is False
        if not all(checks.values()):
            return {"ready": False, "reason": "D technical validation is not passed", "checks": checks}
        return {"ready": True, "checks": checks}
    except Exception as exc:
        return {"ready": False, "reason": "%s: %s" % (type(exc).__name__, exc)}


def check_026() -> dict:
    candidates = sorted(TEST_RES.glob("026-*/results/implementation_gate.json"))
    if len(candidates) != 1:
        return {"ready": False, "reason": "expected one released 026 gate, found %d" % len(candidates),
                "candidate_paths": [str(path) for path in candidates]}
    gate_path = candidates[0]
    gate = ex._synthetic_gate(gate_path, PROTOCOL_SHA)
    if gate.get("status") != "ready":
        return {"ready": False, "reason": gate.get("error", "026 strict gate is not ready"),
                "gate": gate}
    return {"ready": True, "path": str(gate_path), "gate": gate}


def all_gates() -> dict:
    native = check_025()
    d = check_d()
    synthetic = check_026()
    return {"ready": native["ready"] and d["ready"] and synthetic["ready"],
            "native_025": native, "D_validation": d, "synthetic_026": synthetic}


def verify_prepared(result: dict, gate: dict) -> dict:
    root = Path(result["study_root"]).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("status") != "ready_for_worker" or manifest.get("ready_for_worker") is not True:
        raise RuntimeError("fresh prepare did not produce ready_for_worker")
    rows = manifest.get("jobs", [])
    if len(rows) != 15 or any(row.get("status") != "pending" or row.get("fit_called") for row in rows):
        raise RuntimeError("fresh prepare did not freeze 15 pending jobs")
    if manifest.get("protocol_release", {}).get("sha256") != PROTOCOL_SHA:
        raise RuntimeError("fresh manifest protocol hash mismatch")
    if manifest.get("synthetic_implementation_gate", {}).get("sha256") != gate["gate"]["sha256"]:
        raise RuntimeError("fresh manifest gate hash mismatch")
    source_manifest = root / manifest["source_manifest"]["path"]
    if sha256(source_manifest) != manifest["source_manifest"]["sha256"]:
        raise RuntimeError("fresh source manifest file hash mismatch")
    source_hashes = root / manifest["source_hashes"]["path"]
    if sha256(source_hashes) != manifest["source_hashes"]["sha256"]:
        raise RuntimeError("fresh source snapshot file hash mismatch")
    first_job = root / rows[0]["job_config"]
    job = read_json(first_job)
    ex.verify_source_snapshot(source_hashes, job["source_snapshot_sha256"])
    for key in ("input_manifest", "config", "config_snapshot", "freeze", "x0_manifest", "jobs_manifest"):
        item = manifest[key]
        path = root / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError("fresh %s hash mismatch" % key)
    for row in rows:
        job_path = root / row["job_config"]
        if not job_path.is_file() or sha256(job_path) != row["job_config_sha256"]:
            raise RuntimeError("fresh job hash mismatch: %s" % row["job_id"])
        if ex._forbidden_training_keys(read_json(job_path)):
            raise RuntimeError("forbidden training key in %s" % row["job_id"])
    registry = Path(result["registry"])
    records = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines() if line.strip()]
    matching = [item for item in records if item.get("run_root") == str(root)]
    if not matching or matching[-1].get("fit_called") is not False:
        raise RuntimeError("fresh prepare registry record missing or fit_called is not false")
    return {"run_root": str(root), "manifest_sha256": sha256(root / "manifest.json"),
            "freeze_sha256": manifest["freeze"]["sha256"],
            "jobs_manifest_sha256": manifest["jobs_manifest"]["sha256"],
            "job_count": len(rows), "ready_for_worker": True,
            "registry": str(registry), "registry_record_fit_called": False}


def terminal_progress(root: Path) -> dict:
    statuses = []
    for row in read_json(root / "manifest.json").get("jobs", []):
        path = root / "jobs" / row["job_id"] / "status.json"
        if path.is_file():
            statuses.append(read_json(path).get("status"))
        else:
            statuses.append("pending")
    terminal = sum(status in ex.TERMINAL_STATUSES for status in statuses)
    return {"job_count": len(statuses), "terminal_job_count": terminal,
            "pending_job_count": len(statuses) - terminal,
            "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))}}


def main() -> int:
    record("controller_started", poll_seconds=POLL_SECONDS, protocol_sha256=PROTOCOL_SHA,
           validation_path=str(PRE025_VALIDATION))
    while True:
        gates = all_gates()
        record("waiting" if not gates["ready"] else "gates_ready", gates=gates)
        if gates["ready"]:
            break
        time.sleep(POLL_SECONDS)

    gate = gates["synthetic_026"]
    try:
        prepared = ex.prepare(
            release_sha256=PROTOCOL_SHA,
            synthetic_gate_path=gate["path"],
            starts_root=STARTS,
            source_path=SOURCE,
        )
        prepared_check = verify_prepared(prepared, gate)
        record("prepared", prepare=prepared_check)
        (WAIT_ROOT / "prepare_receipt.json").write_text(
            json.dumps({"prepare": prepared, "verification": prepared_check}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    except Exception as exc:
        record("prepare_failed", error_type=type(exc).__name__, error=str(exc))
        return 2

    run_root = Path(prepared_check["run_root"])
    launcher_log = WAIT_ROOT / "launcher.stdout"
    launcher_err = WAIT_ROOT / "launcher.stderr"
    env = os.environ.copy()
    env.update(ex._single_thread_env())
    command = [sys.executable, "-m", "pr.launch_allele_workers", "--run-root",
               str(run_root), "--max-workers", "6"]
    record("launcher_starting", run_root=str(run_root), command=command)
    with launcher_log.open("w", encoding="utf-8") as stdout, launcher_err.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr)
        record("launcher_started", run_root=str(run_root), launcher_pid=process.pid,
               command=command)
        return_code = process.wait()
    record("launcher_exited", run_root=str(run_root), launcher_pid=process.pid,
           launcher_returncode=return_code,
           launcher_receipt=str(run_root / "logs" / "worker_launcher.json"))

    while True:
        progress = terminal_progress(run_root)
        record("terminal_progress", run_root=str(run_root), progress=progress)
        if progress["job_count"] == 15 and progress["terminal_job_count"] == 15:
            break
        time.sleep(POLL_SECONDS)
    try:
        final = ex.finalize(run_root)
        record("finalized", run_root=str(run_root), status=final["status"],
               terminal_job_count=final["terminal_job_count"], job_count=final["job_count"],
               finalize_path=str(run_root / "finalize.json"),
               selection_path=str(run_root / "selection.json"),
               termination_audit_path=str(run_root / "termination_audit.json"))
        (WAIT_ROOT / "final_receipt.json").write_text(
            json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        record("finalize_failed", run_root=str(run_root), error_type=type(exc).__name__, error=str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
