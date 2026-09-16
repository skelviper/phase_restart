#!/usr/bin/env python3
"""Corrected low-frequency gated post-020 real-run controller."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from scratch import post020_real_controller as base

# 023 was the already accepted model/numerical preflight source for this task.
# It is recorded for provenance; the execution hard gates are 025 and 026 only.
NUMERICAL_PREFLIGHT = (
    base.TEST_RES
    / "023-20260913_212541-post020-allele-signal-diagnostics"
    / "model-preflight-20260913_221712"
    / "results"
    / "model_preflight.json"
)


def all_gates() -> dict:
    native = base.check_025()
    synthetic = base.check_026()
    return {
        "ready": native["ready"] and synthetic["ready"],
        "native_025": native,
        "synthetic_026": synthetic,
        "numerical_preflight": {
            "path": str(NUMERICAL_PREFLIGHT),
            "accepted_before_controller": True,
        },
    }


def live_claims(root: Path) -> list[dict]:
    claims = []
    for path in sorted((root / "active").glob("*.json")):
        try:
            payload = base.read_json(path)
        except Exception as exc:
            claims.append({"path": str(path), "error": "%s: %s" % (type(exc).__name__, exc)})
            continue
        payload["path"] = str(path)
        claims.append(payload)
    return claims


def main() -> int:
    base.record(
        "controller_started_v2",
        poll_seconds=base.POLL_SECONDS,
        protocol_sha256=base.PROTOCOL_SHA,
        validation_path=str(base.PRE025_VALIDATION),
        numerical_preflight_path=str(NUMERICAL_PREFLIGHT),
        hard_gates=("025_native_validation", "026_strict_terminal_gate"),
        replaced_controller="bash-105",
    )
    while True:
        gates = all_gates()
        base.record("waiting_v2" if not gates["ready"] else "gates_ready_v2", gates=gates)
        if gates["ready"]:
            break
        base.time.sleep(base.POLL_SECONDS)

    gate = gates["synthetic_026"]
    try:
        prepared = base.ex.prepare(
            release_sha256=base.PROTOCOL_SHA,
            synthetic_gate_path=gate["path"],
            starts_root=base.STARTS,
            source_path=base.SOURCE,
        )
        prepared_check = base.verify_prepared(prepared, gate)
        base.record("prepared_v2", prepare=prepared_check)
        (base.WAIT_ROOT / "prepare_receipt.json").write_text(
            json.dumps({"prepare": prepared, "verification": prepared_check}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        base.record("prepare_failed_v2", error_type=type(exc).__name__, error=str(exc))
        return 2

    run_root = Path(prepared_check["run_root"])
    launcher_log = base.WAIT_ROOT / "launcher.stdout"
    launcher_err = base.WAIT_ROOT / "launcher.stderr"
    env = os.environ.copy()
    env.update(base.ex._single_thread_env())
    command = [
        sys.executable,
        "-m",
        "pr.launch_allele_workers",
        "--run-root",
        str(run_root),
        "--max-workers",
        "6",
    ]
    base.record("launcher_starting_v2", run_root=str(run_root), command=command)
    try:
        with launcher_log.open("w", encoding="utf-8") as stdout, launcher_err.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=str(base.ROOT), env=env, stdout=stdout, stderr=stderr)
            base.record("launcher_started_v2", run_root=str(run_root), launcher_pid=process.pid, command=command)
            return_code = process.wait()
    except Exception as exc:
        base.record("launcher_failed_v2", run_root=str(run_root), error_type=type(exc).__name__, error=str(exc))
        return 4

    launcher_receipt_path = run_root / "logs" / "worker_launcher.json"
    base.record(
        "launcher_exited_v2",
        run_root=str(run_root),
        launcher_pid=process.pid,
        launcher_returncode=return_code,
        launcher_receipt=str(launcher_receipt_path),
    )
    progress = base.terminal_progress(run_root)
    active = live_claims(run_root)
    if progress["job_count"] != 15 or progress["terminal_job_count"] != 15 or active:
        partial = {
            "status": "partial",
            "error": "launcher exited before all 15 jobs reached terminal state",
            "launcher_returncode": return_code,
            "progress": progress,
            "active_claims": active,
            "launcher_receipt": str(launcher_receipt_path),
            "run_root": str(run_root),
        }
        (base.WAIT_ROOT / "partial_receipt.json").write_text(
            json.dumps(partial, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        base.record("launcher_partial_v2", **partial)
        return 4

    try:
        final = base.ex.finalize(run_root)
        base.record(
            "finalized_v2",
            run_root=str(run_root),
            status=final["status"],
            terminal_job_count=final["terminal_job_count"],
            job_count=final["job_count"],
            finalize_path=str(run_root / "finalize.json"),
            selection_path=str(run_root / "selection.json"),
            termination_audit_path=str(run_root / "termination_audit.json"),
        )
        (base.WAIT_ROOT / "final_receipt.json").write_text(
            json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0
    except Exception as exc:
        base.record("finalize_failed_v2", run_root=str(run_root), error_type=type(exc).__name__, error=str(exc))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
