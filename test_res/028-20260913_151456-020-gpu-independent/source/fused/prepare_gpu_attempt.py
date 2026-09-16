"""Create an immutable provenance snapshot for one fused CUDA formal attempt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

RUN = Path(__file__).resolve().parents[2]
WORKSPACE = RUN.parents[1]
SOURCE = RUN / "source"
ATTEMPT = RUN / "backend_runs" / "fused-cuda" / "attempt-20260913T165036Z"
CPU_CONFIG = RUN / "backend_runs" / "archived-cpu" / "attempt-20260913T164158Z" / "config.json"
ROOT_CONFIG = RUN / "config.json"
SNAPSHOT = ATTEMPT / "provenance" / "source_snapshot"
MANIFEST = ATTEMPT / "provenance" / "used-source-manifest.json"
PRELAUNCH = ATTEMPT / "provenance" / "prelaunch.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, relative: Path, role: str, entries: list[dict]) -> None:
    target = SNAPSHOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    entries.append({
        "role": role,
        "source": str(source),
        "snapshot": str(target),
        "relative": str(relative),
        "sha256": sha256(source),
        "bytes": source.stat().st_size,
    })


def main() -> None:
    if ATTEMPT.exists():
        raise RuntimeError(f"refusing to reuse existing attempt: {ATTEMPT}")
    if not ROOT_CONFIG.is_file() or not CPU_CONFIG.is_file():
        raise FileNotFoundError("root or CPU frozen config is missing")
    root_hash = sha256(ROOT_CONFIG)
    cpu_hash = sha256(CPU_CONFIG)
    if root_hash != cpu_hash:
        raise RuntimeError(f"root/CPU frozen config hash mismatch: {root_hash} != {cpu_hash}")
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT_CONFIG, ATTEMPT / "config.json")
    entries: list[dict] = []
    copy_file(SOURCE / "run_pipeline.py", Path("source/run_pipeline.py"), "primary_pipeline_readonly_snapshot", entries)
    copy_file(SOURCE / "gpu_backend.py", Path("source/gpu_backend.py"), "primary_backend_readonly_snapshot", entries)
    for path in sorted((SOURCE / "fused").glob("*")):
        if path.is_file() and path.suffix in {".py", ".cu"}:
            copy_file(path, Path("source/fused") / path.name, "fused_backend_source", entries)
    for path in sorted((SOURCE / "archived020").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            copy_file(path, Path("source/archived020") / path.relative_to(SOURCE / "archived020"), "archived020_source_snapshot", entries)
    design = RUN / "provenance" / "FUSED_KERNEL_DESIGN.md"
    if design.is_file():
        copy_file(design, Path("provenance/FUSED_KERNEL_DESIGN.md"), "fused_design_snapshot", entries)
    payload = {
        "status": "prepared",
        "attempt": str(ATTEMPT),
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config": {
            "source_root": str(ROOT_CONFIG),
            "source_cpu_frozen": str(CPU_CONFIG),
            "root_sha256": root_hash,
            "cpu_sha256": cpu_hash,
            "attempt_sha256": sha256(ATTEMPT / "config.json"),
            "match": True,
        },
        "input": {
            "path": str(WORKSPACE / "inputs" / "P9016.snpfree.pairs.gz"),
            "sha256": "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa",
        },
        "source_entries": entries,
        "source_count": len(entries),
        "fused_source_hash": sha256(SOURCE / "fused" / "cuda_pair_objective.cu"),
        "formal_command": (
            "conda run --no-capture-output -n analysis python "
            f"{SOURCE / 'run_pipeline.py'} --run-dir {ATTEMPT} "
            "--backend fused_cuda --device cuda --through 1m --tile-rows 32"
        ),
        "initial_sources_root": str(RUN / "coords" / "initial_sources"),
        "gpu_attempt_output": str(ATTEMPT / "backend_runs" / "fused-cuda"),
        "formal_pipeline_started": False,
        "phase_or_reference_training_input": False,
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(text, encoding="utf-8")
    PRELAUNCH.write_text(json.dumps({
        "status": "ready_to_launch",
        "attempt": str(ATTEMPT),
        "config_sha256": root_hash,
        "used_source_manifest": str(MANIFEST),
        "source_count": len(entries),
        "fused_cuda_source_sha256": sha256(SOURCE / "fused" / "cuda_pair_objective.cu"),
        "formal_pipeline_started": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
