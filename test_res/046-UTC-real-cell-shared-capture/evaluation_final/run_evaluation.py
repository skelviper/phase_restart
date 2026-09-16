#!/usr/bin/env python
"""Single-writer persistent wrapper for evaluation_final."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent.parent
OUT = RUN / "evaluation_final"
LOCK = OUT / "evaluation.lock"
EVALUATOR = OUT / "evaluator.py"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _next_attempt() -> Path:
    attempts = []
    for path in (OUT / "attempts").glob("attempt-*"):
        try:
            attempts.append(int(path.name.split("-")[-1]))
        except ValueError:
            pass
    number = max(attempts, default=0) + 1
    path = OUT / "attempts" / ("attempt-%03d" % number)
    path.mkdir(parents=True, exist_ok=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "evaluate"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LOCK.touch(exist_ok=True)
    attempt = _next_attempt()
    started = _now()
    command = [sys.executable, str(EVALUATOR), args.command]
    launch = {
        "schema": "p9016-real-evaluation-final-launch-v1", "status": "running", "command": command,
        "wrapper_pid": os.getpid(), "started_at_utc": started, "phase_opened": False, "reference_opened": False,
        "synthetic_evaluation": False, "attempt_dir": str(attempt),
        "stdout": str(attempt / "stdout.log"), "stderr": str(attempt / "stderr.log"),
    }
    _write(attempt / "launch.json", launch)
    try:
        lock_handle = LOCK.open("r+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            terminal = {**launch, "status": "terminal", "return_code": 73, "finished_at_utc": _now(), "lock_busy": True}
            _write(attempt / "terminal.json", terminal)
            _write(OUT / "wrapper_terminal.json", terminal)
            return 73
        try:
            with (attempt / "stdout.log").open("w", encoding="utf-8") as stdout, (attempt / "stderr.log").open("w", encoding="utf-8") as stderr:
                child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                launch["child_pid"] = child.pid
                _write(attempt / "launch.json", launch)
                return_code = child.wait()
            state_path = OUT / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            terminal = {**launch, "status": "terminal", "return_code": int(return_code), "finished_at_utc": _now(),
                        "phase_opened": bool(state.get("phase_opened", False)), "reference_opened": bool(state.get("reference_opened", False)),
                        "state_path": str(state_path), "lock_method": "fcntl.flock(LOCK_EX|LOCK_NB)", "lock_held_until_terminal": True}
            _write(attempt / "terminal.json", terminal)
            _write(OUT / "wrapper_terminal.json", terminal)
            return int(return_code)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
    except Exception as exc:
        terminal = {**launch, "status": "terminal", "return_code": 1, "finished_at_utc": _now(), "wrapper_exception": repr(exc)}
        _write(attempt / "terminal.json", terminal)
        _write(OUT / "wrapper_terminal.json", terminal)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
