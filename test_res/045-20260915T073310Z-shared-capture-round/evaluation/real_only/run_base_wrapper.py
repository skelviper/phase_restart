"""Single-writer managed recovery launcher for the 045 real-only base."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent


def main() -> int:
    lock = RUN / "base_writer.lock"
    launch = {"schema": "p9016-real-base-launch-v1", "pid": os.getpid(),
              "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "command": "real_only_controller.py run", "reference_opened": False, "phase_opened": False, "synthetic_fits": 0}
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit("base writer lock exists; refuse duplicate writer")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(launch, handle, sort_keys=True, indent=2); handle.write("\n")
    (RUN / "base_launch.json").write_text(json.dumps(launch, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    stdout_path, stderr_path = RUN / "logs/base_recovery.stdout.log", RUN / "logs/base_recovery.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(RUN.parent / "real_only_controller.py"), "run"]
    env = os.environ.copy()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=str(RUN.parents[2]), stdout=stdout, stderr=stderr, env=env)
        launch.update({"child_pid": process.pid, "child_started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "stdout": str(stdout_path), "stderr": str(stderr_path)})
        (RUN / "base_launch.json").write_text(json.dumps(launch, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return_code = process.wait()
    terminal = {"schema": "p9016-real-base-terminal-v1", **launch, "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "return_code": int(return_code), "status": "terminal"}
    (RUN / "base_terminal.json").write_text(json.dumps(terminal, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
