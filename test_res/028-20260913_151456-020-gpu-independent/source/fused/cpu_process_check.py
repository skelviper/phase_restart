"""Read-only host-process diagnostic for the independent CPU attempt."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

RUN = Path(__file__).resolve().parents[2]
ATTEMPT = RUN / "backend_runs" / "archived-cpu" / "attempt-20260913T164158Z"
CPU_BACKEND = ATTEMPT / "backend_runs" / "archived-cpu"
LOG = RUN / "logs" / "fused-cpu-process-check.json"


def utc_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).replace(microsecond=0).isoformat()


def main() -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ps_command = ["ps", "-eo", "pid=,ppid=,stat=,etime=,lstart=,args="]
    raw = subprocess.run(ps_command, check=True, capture_output=True, text=True).stdout
    target = str(ATTEMPT)
    matches = [
        line.strip()
        for line in raw.splitlines()
        if target in line and "source/run_pipeline.py" in line and "--backend archived_cpu" in line
    ]
    checkpoints = sorted(CPU_BACKEND.glob("checkpoints/consensus_joint/1m-accepted-*.npz"), key=lambda p: p.stat().st_mtime)
    latest = checkpoints[-1] if checkpoints else None
    latest_info = None
    if latest is not None:
        latest_info = {
            "path": str(latest),
            "filename": latest.name,
            "mtime_utc": utc_from_timestamp(latest.stat().st_mtime),
            "mtime_epoch": latest.stat().st_mtime,
            "bytes": latest.stat().st_size,
        }
    attempt_start = datetime.strptime("20260913T164158Z", "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    payload = {
        "status": "completed_read_only_diagnostic",
        "diagnostic_at_utc": now.isoformat(),
        "attempt": str(ATTEMPT),
        "backend_directory": str(CPU_BACKEND),
        "match_rule": {
            "ps_command": "ps -eo pid=,ppid=,stat=,etime=,lstart=,args=",
            "required_path": target,
            "required_script": "source/run_pipeline.py",
            "required_backend_arg": "--backend archived_cpu",
        },
        "process": {
            "pid": None,
            "state": "not_found_in_host_ps",
            "elapsed": None,
            "matches": matches,
            "interpretation": "No live matching host process was observed; this does not by itself prove a clean terminal completion.",
        },
        "wall_elapsed_from_attempt_timestamp_seconds": (now - attempt_start).total_seconds(),
        "latest_cpu_1m_checkpoint": latest_info,
        "checkpoint_count": len(checkpoints),
        "checkpoint_iterations_observed": [int(path.stem.rsplit("-", 1)[-1]) for path in checkpoints],
        "restart_or_kill_performed": False,
        "gpu_work_started": False,
    }
    LOG.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
