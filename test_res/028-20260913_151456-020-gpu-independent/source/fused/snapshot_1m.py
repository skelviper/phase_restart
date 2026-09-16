"""Freeze the completed two-candidate through-1m record before fine continuation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

RUN = Path(__file__).resolve().parents[2]
ATTEMPT = RUN / "backend_runs" / "fused-cuda" / "attempt-20260913T165036Z"
BACKEND = ATTEMPT / "backend_runs" / "fused-cuda"
SNAPSHOT = ATTEMPT / "provenance" / "through-1m"
AUDIT = ATTEMPT / "provenance" / "through-1m-snapshot.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if AUDIT.exists():
        raise RuntimeError(f"refusing to overwrite existing 1m snapshot: {AUDIT}")
    required = [BACKEND / "pipeline-summary.json", BACKEND / "selection-1m.json"]
    for candidate in ("consensus_joint", "random_joint"):
        for stage in ("5m", "2m", "1m"):
            required.extend([
                BACKEND / "stages" / candidate / f"{stage}.json",
                BACKEND / "resume" / candidate / f"{stage}-state.json",
                BACKEND / "resume" / candidate / f"{stage}-state.npz",
            ])
        required.append(next((BACKEND / "coords" / candidate).glob("final-1m-*.3dg")))
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError("missing through-1m artifact(s): " + ", ".join(missing))
    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    entries = []
    for source in required:
        relative = source.relative_to(BACKEND)
        target = SNAPSHOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append({
            "source": str(source),
            "snapshot": str(target),
            "relative": str(relative),
            "sha256": sha256(source),
            "bytes": source.stat().st_size,
        })
    summary = json.loads((BACKEND / "pipeline-summary.json").read_text(encoding="utf-8"))
    selection = json.loads((BACKEND / "selection-1m.json").read_text(encoding="utf-8"))
    manifest = ATTEMPT / "provenance" / "used-source-manifest.json"
    config = ATTEMPT / "config.json"
    payload = {
        "status": "frozen_before_fine_continuation",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "attempt": str(ATTEMPT),
        "backend_directory": str(BACKEND),
        "config_sha256": sha256(config),
        "used_source_manifest_sha256": sha256(manifest),
        "pipeline_summary_sha256": sha256(BACKEND / "pipeline-summary.json"),
        "selection_sha256": sha256(BACKEND / "selection-1m.json"),
        "through": "1m",
        "selection": selection,
        "candidate_stage_status": {
            candidate: {
                stage: {
                    "status": next(item for item in summary["candidates"] if item["candidate_id"] == candidate)["stages"][{"5m": 0, "2m": 1, "1m": 2}[stage]]["status"],
                    "accepted_iterations": next(item for item in summary["candidates"] if item["candidate_id"] == candidate)["stages"][{"5m": 0, "2m": 1, "1m": 2}[stage]]["timing"]["accepted_iterations"],
                }
                for stage in ("5m", "2m", "1m")
            }
            for candidate in ("consensus_joint", "random_joint")
        },
        "snapshot_count": len(entries),
        "entries": entries,
        "fine_continuation_policy": "selected random_joint only; fixed 240 iterations and 750 function calls per fine stage",
        "formal_optimization_started": True,
        "phase_or_reference_opened": False,
    }
    AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
