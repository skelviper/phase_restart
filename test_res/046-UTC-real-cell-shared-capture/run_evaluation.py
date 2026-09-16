"""Persisted launcher for the post-fit real-only evaluation."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent


def main() -> int:
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    command = [sys.executable, str(RUN / "evaluation/evaluator.py"), "evaluate"]
    launch = {"schema": "p9016-real-evaluation-launch-v1", "wrapper_pid": os.getpid(), "started_at_utc": started,
              "command": command, "reference_opened": "deferred_until_hash_gate", "phase_opened": "deferred_until_hash_gate",
              "synthetic_evaluation": False, "stdout": str(RUN / "logs/evaluation.stdout.log"),
              "stderr": str(RUN / "logs/evaluation.stderr.log")}
    (RUN / "evaluation_launch.json").write_text(json.dumps(launch, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with (RUN / "logs/evaluation.stdout.log").open("w", encoding="utf-8") as stdout, (RUN / "logs/evaluation.stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=str(RUN), stdout=stdout, stderr=stderr, env=os.environ.copy())
        launch["child_pid"] = process.pid
        launch["child_started_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (RUN / "evaluation_launch.json").write_text(json.dumps(launch, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        code = process.wait()
    terminal = {"schema": "p9016-real-evaluation-terminal-v1", **launch, "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "return_code": int(code), "status": "terminal"}
    (RUN / "evaluation_terminal.json").write_text(json.dumps(terminal, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
