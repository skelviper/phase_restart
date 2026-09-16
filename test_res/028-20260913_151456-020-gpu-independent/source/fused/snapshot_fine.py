"""Write a separate immutable summary for the selected fine continuation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

RUN = Path(__file__).resolve().parents[2]
ATTEMPT = RUN / "backend_runs" / "fused-cuda" / "attempt-20260913T165036Z"
BACKEND = ATTEMPT / "backend_runs" / "fused-cuda"
OUT = ATTEMPT / "provenance" / "fine-continuation-summary.json"
CONFIG = ATTEMPT / "config.json"
MANIFEST = ATTEMPT / "provenance" / "used-source-manifest.json"
CPU_CHECK = RUN / "logs" / "fused-cpu-process-check.json"
STAGES = ("500k", "200k", "100k", "50k", "20k")
CANDIDATE = "random_joint"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite fine summary: {OUT}")
    rows = []
    for stage in STAGES:
        stage_path = BACKEND / "stages" / CANDIDATE / f"{stage}.json"
        state_json = BACKEND / "resume" / CANDIDATE / f"{stage}-state.json"
        state_npz = BACKEND / "resume" / CANDIDATE / f"{stage}-state.npz"
        coordinate_path = next((BACKEND / "coords" / CANDIDATE).glob(f"final-{stage}-*.3dg"), None)
        if coordinate_path is None or not all(path.is_file() for path in (stage_path, state_json, state_npz)):
            raise FileNotFoundError(f"incomplete fine stage: {stage}")
        item = json.loads(stage_path.read_text(encoding="utf-8"))
        state_metadata = json.loads(state_json.read_text(encoding="utf-8"))
        final_coordinates = dict(item["final_coordinates"])
        final_coordinates["file_sha256"] = final_coordinates.pop("sha256")
        final_coordinates["array_sha256"] = state_metadata["coordinates_sha256"]
        fit = item["fit"]
        timing = item["timing"]
        budget = item["data_budget"]
        if item["status"] not in ("budget_not_converged", "converged"):
            raise RuntimeError(f"fine stage {stage} has non-eligible status {item['status']}")
        rows.append({
            "stage": stage,
            "status": item["status"],
            "accepted_iterations": int(timing["accepted_iterations"]),
            "objective_evaluations": int(timing["objective_evaluations"]),
            "nit": int(fit["nit"]),
            "nfev": int(fit["nfev"]),
            "termination_class": fit["termination_class"],
            "success": bool(fit["success"]),
            "initial_total": float(fit["initial_total"]),
            "final_total": float(fit["final_total"]),
            "final_count_nll_normalized": float(fit["final_components"]["count_nll_normalized"]),
            "final_gradient_l2": float(fit["final_gradient_l2"]),
            "final_gradient_max_abs": float(fit["final_gradient_max_abs"]),
            "timing": {
                key: float(timing[key])
                for key in ("preprocess_seconds", "upload_seconds", "warmup_seconds",
                            "optimizer_only_seconds", "export_seconds", "scoped_end_to_end_seconds",
                            "peak_cuda_memory_bytes")
            },
            "budget": {
                key: int(budget[key])
                for key in ("n_loci", "n_eligible_pairs", "n_observed_nonzero_pairs",
                            "n_zero_eligible_pairs", "raw_records", "raw_cis_offdiag",
                            "raw_same_bin", "raw_inter", "endpoint_total")
            },
            "final_coordinates": final_coordinates,
            "artifact_hashes": {
                "stage_json": sha256(stage_path),
                "resume_state_json": sha256(state_json),
                "resume_state_npz": sha256(state_npz),
                "final_coordinates_3dg": sha256(coordinate_path),
            },
            "artifact_paths": {
                "stage_json": str(stage_path),
                "resume_state_json": str(state_json),
                "resume_state_npz": str(state_npz),
                "final_coordinates_3dg": str(coordinate_path),
            },
            "phase_or_reference_opened": False,
        })
    pipeline_summary = BACKEND / "pipeline-summary.json"
    selection = BACKEND / "selection-1m.json"
    payload = {
        "status": "completed_fine_continuation",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "attempt": str(ATTEMPT),
        "backend_directory": str(BACKEND),
        "selected_candidate": CANDIDATE,
        "stages": rows,
        "coordinate_hash_semantics": {
            "array_sha256": "canonical run_pipeline.array_sha256 of the saved coordinate array in resume state",
            "file_sha256": "SHA256 of the emitted .3dg coordinate file bytes",
        },
        "config_sha256": sha256(CONFIG),
        "used_source_manifest_sha256": sha256(MANIFEST),
        "fused_cuda_source_sha256": next(
            entry["sha256"] for entry in json.loads(MANIFEST.read_text(encoding="utf-8"))["source_entries"]
            if entry["relative"] == "source/fused/cuda_pair_objective.cu"
        ),
        "selection_sha256": sha256(selection),
        "final_pipeline_summary_sha256": sha256(pipeline_summary),
        "cpu_process_diagnostic": {
            "path": str(CPU_CHECK),
            "sha256": sha256(CPU_CHECK) if CPU_CHECK.is_file() else None,
            "interpretation": "CPU background process was not found by narrow host ps; no CPU/GPU process was killed or restarted by this run.",
        },
        "formal_optimization_started": True,
        "phase_or_reference_opened": False,
        "continuation_policy": "random_joint selected after both candidates through 1m; fixed 240 iterations and 750 maxfun at every fine stage",
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
