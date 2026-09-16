"""049 正式 12 fit 调度器。

每条 fit 固定 1502 FG、1Mb full grid，走同一 stage_fit 入口；子进程之间完全隔离，
每个子进程独立 objective / 独立 GPU 上下文。并发度受 --workers 控制。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from round_paths import FIT_FG_CAP, LOSSES, RUN, SOLVERS, SOURCES, all_fit_ids, fit_id

SOURCE = RUN / "source"
LOGS = RUN / "logs"
PYTHON = "/mnt/ssd/zliu/miniforge3/envs/analysis/bin/python"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", default=None, help="comma separated fit ids")
    args = parser.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    (RUN / "results" / "fits").mkdir(parents=True, exist_ok=True)
    targets = args.only.split(",") if args.only else all_fit_ids()
    if args.only and targets == [""]:
        targets = all_fit_ids()

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    # 调度器自身的持久日志（stdout 也写入真实文件，便于父侧直接读取）
    LOGS.mkdir(parents=True, exist_ok=True)
    run_log = (LOGS / "formal_run.log").open("w", encoding="utf-8")

    def emit(message: str) -> None:
        print(message, flush=True)
        run_log.write(message + "\n")
        run_log.flush()

    state_path = RUN / "results" / "formal_runtime.json"
    state_path.write_text(json.dumps({"schema": "p9016-max-contact-formal-runtime-v1", "status": "running",
                                      "started_at_utc": started, "workers": int(args.workers),
                                      "fit_count": len(targets), "fg_per_fit": FIT_FG_CAP,
                                      "reference_opened": False, "phase_opened": False},
                                     indent=2, sort_keys=True) + "\n", encoding="utf-8")

    running: list[dict] = []
    pending = list(targets)
    rows: list[dict] = []
    while pending or running:
        while pending and len(running) < int(args.workers):
            target = pending.pop(0)
            loss, solver, source = target.split("-")
            stdout_path = LOGS / ("fit_%s.stdout.log" % target)
            stderr_path = LOGS / ("fit_%s.stderr.log" % target)
            handle_out = stdout_path.open("w", encoding="utf-8")
            handle_err = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen([PYTHON, str(SOURCE / "run_fit.py"), loss, solver, source],
                                       cwd=str(RUN), env=environment, stdout=handle_out, stderr=handle_err)
            running.append({"fit_id": target, "process": process, "started": time.time(),
                            "stdout": stdout_path, "stderr": stderr_path,
                            "handle_out": handle_out, "handle_err": handle_err})
            emit("[start] %s pid=%d" % (target, process.pid))
        time.sleep(2.0)
        for entry in list(running):
            code = entry["process"].poll()
            if code is None:
                continue
            entry["handle_out"].close()
            entry["handle_err"].close()
            running.remove(entry)
            wall = time.time() - entry["started"]
            summary_path = RUN / "results" / "fits" / (entry["fit_id"] + ".json")
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
            row = {"fit_id": entry["fit_id"], "returncode": int(code), "wall_seconds": wall,
                   "status": (summary or {}).get("status"), "terminal_reason": (summary or {}).get("terminal_reason"),
                   "outer_fg_actual": (summary or {}).get("outer_fg_actual"),
                   "fit_wall_seconds": (summary or {}).get("fit_wall_seconds"),
                   "count_nll_normalized": (summary or {}).get("count_nll_normalized"),
                   "total": (summary or {}).get("total"),
                   "canonical_gradient_max_abs": (summary or {}).get("canonical_gradient_max_abs"),
                   "stdout": str(entry["stdout"].relative_to(RUN)), "stderr": str(entry["stderr"].relative_to(RUN))}
            rows.append(row)
            emit("[done ] %s rc=%d status=%s fg=%s wall=%.1fs" % (
                entry["fit_id"], code, row["status"], row["outer_fg_actual"], wall))
            state_path.write_text(json.dumps({"schema": "p9016-max-contact-formal-runtime-v1",
                                              "status": "running", "started_at_utc": started,
                                              "workers": int(args.workers), "completed": len(rows),
                                              "fit_count": len(targets), "fg_per_fit": FIT_FG_CAP,
                                              "rows": rows, "reference_opened": False, "phase_opened": False},
                                             indent=2, sort_keys=True) + "\n", encoding="utf-8")

    completed = dt.datetime.now(dt.timezone.utc).isoformat()
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[str(row["status"])] = by_status.get(str(row["status"]), 0) + 1
    terminal = {"schema": "p9016-max-contact-formal-terminal-v1", "started_at_utc": started,
                "completed_at_utc": completed, "fit_count": len(targets), "workers": int(args.workers),
                "fg_per_fit": FIT_FG_CAP, "total_fg_actual": sum(int(row["outer_fg_actual"] or 0) for row in rows),
                "status_counts": by_status, "rows": rows,
                "returncode_zero_is_not_scientific_success": True,
                "reference_opened": False, "phase_opened": False}
    (RUN / "results" / "formal_terminal.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    emit(json.dumps({"status_counts": by_status,
                     "total_fg_actual": terminal["total_fg_actual"],
                     "completed_at_utc": completed}))
    run_log.close()
    failed = [row for row in rows if row["status"] in (None, "failure")]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
