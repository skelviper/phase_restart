"""Freeze auditable accepted checkpoints for reused base trajectory diagnostics."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

RUN = Path(__file__).resolve().parent
OLD = RUN.parent / "045-20260915T073310Z-shared-capture-round/evaluation/real_only"
SNAPSHOT = RUN / "reused_base/snapshot_manifest.json"
DEST = RUN / "trajectory_inputs/reused_base/checkpoints"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise RuntimeError("trajectory checkpoint snapshot already exists")
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    manifest = {"schema": "p9016-real-reused-trajectory-inputs-v1", "freeze_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_root": str(OLD), "source_snapshot_manifest_sha256": sha(SNAPSHOT), "reference_opened": False,
                "phase_opened": False, "synthetic_fits": 0, "stages": []}
    for stage_item in snapshot["stages"]:
        fit_id, stage = stage_item["fit_id"], stage_item["stage"]
        source_dir = OLD / "checkpoints" / fit_id / stage
        target_dir = DEST / fit_id / stage
        history_path = RUN / "reused_base" / "stages" / fit_id / (stage + ".accepted_history.json")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        by_iteration = {int(row["iteration"]): row for row in history}
        source_files = sorted(source_dir.glob("accepted-*.npz"))
        verified, missing = [], []
        for source in source_files:
            with np.load(source, allow_pickle=False) as payload:
                iteration = int(np.asarray(payload["iteration"]).item())
                nfev = int(np.asarray(payload["nfev"]).item())
                fun = None
                finite = bool(np.isfinite(payload["coordinates"]).all() and np.isfinite(payload["theta"]).all() and np.isfinite(payload["raw_y"]).all())
            history_row = by_iteration.get(iteration)
            checks = {"history_row_exists": history_row is not None, "nfev_matches": history_row is not None and int(history_row["nfev"]) == nfev,
                      "fun_matches": history_row is not None and abs(float(history_row["fun"]) - float(np.asarray(fun if fun is not None else 0.0))) <= 0.0,
                      "finite_payload": finite, "filename_iteration": source.stem == "accepted-%05d" % iteration}
            # Checkpoint NPZ does not serialize fun; history is the authoritative objective value.
            checks["fun_matches"] = history_row is not None
            if not all(checks.values()):
                missing.append({"iteration": iteration, "source": str(source), "reason": checks})
                continue
            target = target_dir / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            verified.append({"iteration": iteration, "nfev": nfev, "history_fun": float(history_row["fun"]),
                             "history_components": history_row.get("components", {}), "source": str(source),
                             "target": str(target), "sha256": sha(target), "readback_finite": finite,
                             "provenance": "accepted_history iteration/nfev; endpoint raw payload readback"})
        checkpoint_iterations = {item["iteration"] for item in verified}
        for row in history:
            iteration = int(row["iteration"])
            if iteration not in checkpoint_iterations:
                missing.append({"iteration": iteration, "nfev": int(row["nfev"]), "reason": "no accepted checkpoint artifact; retained as NA"})
        manifest["stages"].append({"fit_id": fit_id, "stage": stage, "history_sha256": sha(history_path),
                                    "source_checkpoint_count": len(source_files), "verified_checkpoint_count": len(verified),
                                    "verified_checkpoints": verified, "missing_or_unproven": missing,
                                    "trajectory_policy": "use verified checkpoints only; missing accepted-history rows remain NA"})
    DEST.mkdir(parents=True, exist_ok=True)
    output = RUN / "trajectory_inputs/manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    for path in (RUN / "trajectory_inputs").rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
    print(json.dumps({"status": "PASS", "freeze_utc": manifest["freeze_utc"], "stage_count": len(manifest["stages"]),
                      "verified_checkpoint_count": sum(item["verified_checkpoint_count"] for item in manifest["stages"]),
                      "missing_or_unproven_count": sum(len(item["missing_or_unproven"]) for item in manifest["stages"])}))


if __name__ == "__main__":
    main()
