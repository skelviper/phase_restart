"""Single-writer wrapper for isolated 046 base_remaining fits."""
from __future__ import annotations
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent
OUT = RUN / "base_remaining"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "writer.lock"
    handle = lock.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SystemExit("base_remaining writer lock is held") from exc
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    command = [sys.executable, str(RUN / "base_remaining_controller.py"), "run"]
    launch = {"schema": "p9016-real-base-remaining-launch-v1", "wrapper_pid": os.getpid(),
              "started_at_utc": started, "command": command, "lock_method": "fcntl.flock(LOCK_EX|LOCK_NB)",
              "reference_opened": False, "phase_opened": False, "synthetic_fits": 0,
              "stdout": str(OUT / "wrapper.stdout.log"), "stderr": str(OUT / "wrapper.stderr.log")}
    (OUT / "wrapper_launch.json").write_text(json.dumps(launch, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with (OUT / "wrapper.stdout.log").open("w", encoding="utf-8") as stdout, (OUT / "wrapper.stderr.log").open("w", encoding="utf-8") as stderr:
        env = os.environ.copy()
        env["P9016_BASE_REMAINING_LOCK_HELD"] = "1"
        child = subprocess.Popen(command, cwd=str(RUN), stdout=stdout, stderr=stderr, env=env, pass_fds=(handle.fileno(),))
        launch["child_pid"] = child.pid
        launch["child_started_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (OUT / "wrapper_launch.json").write_text(json.dumps(launch, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        code = child.wait()
    terminal = {"schema": "p9016-real-base-remaining-terminal-v1", **launch,
                "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "return_code": int(code), "status": "terminal"}
    (OUT / "wrapper_terminal.json").write_text(json.dumps(terminal, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
