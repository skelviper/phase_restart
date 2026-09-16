"""启动最多六个哈希锁定的 post-020 real worker。

这是一个有意保持很小的进程启动器。它不准备输入、不重新生成 native starts、不重试终态任务，也不自行调用 optimizer。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from .allele_experiment import (
    MAX_WORKERS,
    MIN_AVAILABLE_BYTES,
    ExperimentError,
    _available_memory_bytes,
    _jsonable,
    _single_thread_env,
    read_json,
    utc_now,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
MEMORY_POLL_SECONDS = 30


def _job_is_terminal(root: Path, job_id: str) -> bool:
    status_path = root / "jobs" / job_id / "status.json"
    if not status_path.is_file():
        return False
    status = str(read_json(status_path).get("status", ""))
    return status in {
        "solver_converged",
        "budget_not_converged",
        "solver_failed",
        "failed_nonfinite",
        "failed_exception",
        "failed_source_snapshot",
        "failed_input",
    }


def _eligible_jobs(root: Path, manifest: dict[str, Any]) -> list[tuple[str, Path]]:
    jobs = []
    active = {path.stem for path in (root / "active").glob("*.json")}
    for row in manifest.get("jobs", []):
        job_id = str(row["job_id"])
        if job_id in active or _job_is_terminal(root, job_id):
            continue
        job_path = root / str(row["job_config"])
        if not job_path.is_file():
            raise ExperimentError("manifest job config is missing: %s" % job_path)
        jobs.append((job_id, job_path))
    return jobs


def launch(run_root: str | Path, *, max_workers: int = MAX_WORKERS) -> dict[str, Any]:
    """在有界 pool 中运行待处理的 manifest 任务，并返回 receipt。"""
    if not 1 <= int(max_workers) <= MAX_WORKERS:
        raise ExperimentError("max_workers must be between 1 and %d" % MAX_WORKERS)
    root = Path(run_root).resolve()
    manifest = read_json(root / "manifest.json")
    if len(manifest.get("jobs", [])) != 15:
        raise ExperimentError("worker launcher requires all 15 manifest arms")
    receipt: dict[str, Any] = {
        "schema": "post020-real-worker-launcher-v1",
        "run_root": str(root),
        "started_utc": utc_now(),
        "max_workers": int(max_workers),
        "min_mem_available_bytes": MIN_AVAILABLE_BYTES,
        "memory_poll_seconds": MEMORY_POLL_SECONDS,
        "fit_called": False,
        "started_jobs": [],
        "deferred_jobs": [],
        "waiting_events": [],
        "deferred_events": [],
        "return_codes": {},
    }
    if not manifest.get("ready_for_worker", False):
        receipt.update({
            "status": manifest.get("status", "pending_inputs"),
            "reason": "manifest is not ready_for_worker",
            "ended_utc": utc_now(),
        })
        write_json(root / "logs" / "worker_launcher.json", receipt, replace_existing=True)
        return receipt

    pending = _eligible_jobs(root, manifest)
    running: list[tuple[str, Path, subprocess.Popen[Any], Any, Any]] = []
    env = os.environ.copy()
    env.update(_single_thread_env())

    def reap(index: int = 0) -> None:
        job_id, job_path, process, stdout, stderr = running.pop(index)
        code = process.wait()
        stdout.close()
        stderr.close()
        receipt["return_codes"][job_id] = int(code)
        deferred_path = root / "jobs" / job_id / "deferred.json"
        if deferred_path.is_file():
            deferred = read_json(deferred_path)
            deferred_status = str(deferred.get("status", ""))
            if deferred_status.startswith("deferred_") and not _job_is_terminal(root, job_id):
                pending.append((job_id, job_path))
                receipt["deferred_events"].append({
                    "job_id": job_id,
                    "status": deferred_status,
                    "utc": utc_now(),
                    "return_code": int(code),
                })

    while pending or running:
        if pending and len(running) < int(max_workers):
            available = _available_memory_bytes()
            if available is not None and available < MIN_AVAILABLE_BYTES:
                receipt["status"] = "waiting_memavailable"
                receipt["waiting_events"].append({
                    "event": "waiting_memavailable",
                    "utc": utc_now(),
                    "mem_available_bytes": available,
                    "pending_jobs": len(pending),
                    "running_jobs": len(running),
                })
                # 持久化等待状态，不创建第二次 launcher invocation。
                write_json(root / "logs" / "worker_launcher.json", receipt,
                           replace_existing=True)
                completed_index = next(
                    (index for index, item in enumerate(running)
                     if item[2].poll() is not None),
                    None,
                )
                if completed_index is None:
                    time.sleep(MEMORY_POLL_SECONDS)
                else:
                    reap(completed_index)
                continue
            job_id, job_path = pending.pop(0)
            safe_job_id = job_id.replace("/", "_")
            stdout = (root / "logs" / ("launcher-%s.stdout" % safe_job_id)).open("w", encoding="utf-8")
            stderr = (root / "logs" / ("launcher-%s.stderr" % safe_job_id)).open("w", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, "-m", "pr.allele_experiment", "worker", "--job", str(job_path)],
                cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr,
            )
            running.append((job_id, job_path, process, stdout, stderr))
            receipt["started_jobs"].append(job_id)
            receipt["status"] = "running"
            continue
        if running:
            reap()
            continue
        break

    receipt["ended_utc"] = utc_now()
    receipt["fit_called"] = any(
        bool(read_json(root / "jobs" / job_id / "status.json").get("fit_called", False))
        for job_id in receipt["started_jobs"]
        if (root / "jobs" / job_id / "status.json").is_file()
    )
    if any(code != 0 for code in receipt["return_codes"].values()):
        receipt["status"] = "partial"
    elif receipt["started_jobs"]:
        receipt["status"] = "complete"
    else:
        receipt["status"] = "no_pending_jobs"
    write_json(root / "logs" / "worker_launcher.json", receipt, replace_existing=True)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = launch(args.run_root, max_workers=args.max_workers)
    except (ExperimentError, FileNotFoundError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(_jsonable(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
