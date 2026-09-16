"""Managed launcher that persists the actual extension subprocess return code."""
from __future__ import annotations
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent


def main() -> int:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    command = [sys.executable, str(RUN / "extension_controller.py"), "run"]
    launch = {"schema": "p9016-real-extension-launch-v1", "wrapper_pid": os.getpid(), "started_at_utc": started,
              "command": command, "reference_opened": False, "phase_opened": False, "synthetic_fits": 0,
              "stdout": str(RUN / "logs/extension.stdout.log"), "stderr": str(RUN / "logs/extension.stderr.log")}
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    lock_handle = (RUN / "extension_writer.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise SystemExit("extension writer lock is held by another process") from exc
    launch["lock_method"] = "fcntl.flock(LOCK_EX|LOCK_NB)"
    launch["lock_fd"] = lock_handle.fileno()
    (RUN / "extension_launch.json").write_text(json.dumps(launch, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with (RUN / "logs/extension.stdout.log").open("w", encoding="utf-8") as stdout, (RUN / "logs/extension.stderr.log").open("w", encoding="utf-8") as stderr:
        child_env = os.environ.copy()
        child_env["P9016_EXTENSION_LOCK_HELD"] = "1"
        process = subprocess.Popen(command, cwd=str(RUN), stdout=stdout, stderr=stderr, env=child_env,
                                   pass_fds=(lock_handle.fileno(),))
        launch["child_pid"] = process.pid
        launch["child_started_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (RUN / "extension_launch.json").write_text(json.dumps(launch, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return_code = process.wait()
    finished = dt.datetime.now(dt.timezone.utc).isoformat()
    terminal = {"schema": "p9016-real-extension-terminal-v1", "wrapper_pid": os.getpid(), "child_pid": launch["child_pid"],
                "started_at_utc": started, "finished_at_utc": finished, "return_code": int(return_code),
                "status": "terminal", "command": command, "reference_opened": False, "phase_opened": False,
                "synthetic_fits": 0, "stdout": launch["stdout"], "stderr": launch["stderr"]}
    (RUN / "extension_terminal.json").write_text(json.dumps(terminal, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
