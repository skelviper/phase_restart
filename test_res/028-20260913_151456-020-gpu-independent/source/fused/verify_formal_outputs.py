"""Read-only integrity audit for the completed fused CUDA formal attempt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

RUN = Path(__file__).resolve().parents[2]
ATTEMPT = RUN / "backend_runs" / "fused-cuda" / "attempt-20260913T165036Z"
BACKEND = ATTEMPT / "backend_runs" / "fused-cuda"
LOG = RUN / "logs" / "fused-formal-integrity.json"
EXPECTED_CONFIG = "14deef2734fb0b13798d394d6bf0336c23c6e2a3958ec60f532c48f959e6ae7e"
EXPECTED_SOURCE = "f100d691ead66d3ffade72d027d1456fe36f167619e892eb0e076cd31acfcdd7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    fine = json.loads((ATTEMPT / "provenance" / "fine-continuation-summary.json").read_text(encoding="utf-8"))
    through = json.loads((ATTEMPT / "provenance" / "through-1m-snapshot.json").read_text(encoding="utf-8"))
    summary = json.loads((BACKEND / "pipeline-summary.json").read_text(encoding="utf-8"))
    selection = json.loads((BACKEND / "selection-1m.json").read_text(encoding="utf-8"))
    build = json.loads((RUN / "logs" / "fused-build.json").read_text(encoding="utf-8"))
    validation = json.loads((RUN / "logs" / "fused-validation.json").read_text(encoding="utf-8"))
    probe = json.loads((RUN / "logs" / "fused-20k-probe.json").read_text(encoding="utf-8"))
    checks = {}
    checks["config_hash"] = sha256(ATTEMPT / "config.json") == EXPECTED_CONFIG
    checks["source_hash"] = fine["fused_cuda_source_sha256"] == EXPECTED_SOURCE == build["source_sha256"]
    checks["build_compiled"] = build["status"] == "compiled"
    checks["validation_passed"] = validation["status"] == "passed"
    checks["probe_completed"] = probe["status"] == "completed" and probe["formal_optimization_started"] is False
    checks["through_1m_frozen"] = through["status"] == "frozen_before_fine_continuation" and through["selection"]["selected_candidate"] == "random_joint"
    checks["pipeline_final"] = summary["status"] == "training_complete" and summary["through"] == "20k"
    checks["selection_unchanged"] = selection["selected_candidate"] == "random_joint" and selection["status"] == "selected_after_both_candidates_1m"
    checks["phase_reference_closed"] = not fine["phase_or_reference_opened"] and all(not row["phase_or_reference_opened"] for row in fine["stages"])
    checks["all_fine_stages_eligible"] = all(
        row["status"] in ("budget_not_converged", "converged")
        and row["termination_class"] in ("budget_not_converged", "converged")
        and not row["success"] if row["status"] == "budget_not_converged" else True
        for row in fine["stages"]
    )
    checks["all_fine_budgets"] = all(
        row["accepted_iterations"] == 240 and row["nit"] == 240 and row["objective_evaluations"] <= 243
        and row["budget"]["raw_records"] == 1_703_888
        and row["budget"]["endpoint_total"] == 3_407_776
        and row["budget"]["n_zero_eligible_pairs"] == row["budget"]["n_eligible_pairs"] - row["budget"]["n_observed_nonzero_pairs"]
        for row in fine["stages"]
    )
    checkpoint_counts = {}
    stage_file_hashes_ok = True
    coordinate_file_hashes_ok = True
    for row in fine["stages"]:
        stage = row["stage"]
        checkpoints = sorted((BACKEND / "checkpoints" / "random_joint").glob(f"{stage}-accepted-*.npz"))
        checkpoint_counts[stage] = len(checkpoints)
        stage_file_hashes_ok &= sha256(Path(row["artifact_paths"]["stage_json"])) == row["artifact_hashes"]["stage_json"]
        stage_file_hashes_ok &= sha256(Path(row["artifact_paths"]["resume_state_json"])) == row["artifact_hashes"]["resume_state_json"]
        stage_file_hashes_ok &= sha256(Path(row["artifact_paths"]["resume_state_npz"])) == row["artifact_hashes"]["resume_state_npz"]
        coordinate_file_hashes_ok &= sha256(Path(row["artifact_paths"]["final_coordinates_3dg"])) == row["artifact_hashes"]["final_coordinates_3dg"]
    checks["fine_checkpoint_counts"] = all(value == 24 for value in checkpoint_counts.values())
    checks["fine_artifact_hashes"] = stage_file_hashes_ok and coordinate_file_hashes_ok
    final20 = next(row for row in fine["stages"] if row["stage"] == "20k")
    final20_state_path = Path(final20["artifact_paths"]["resume_state_json"])
    final20_state = json.loads(final20_state_path.read_text(encoding="utf-8"))
    checks["final_20k_budget"] = (
        final20["budget"]["n_loci"] == 131700
        and final20["budget"]["n_eligible_pairs"] == 8672379150
        and final20["budget"]["n_observed_nonzero_pairs"] == 1458278
        and final20["budget"]["n_zero_eligible_pairs"] == 8670920872
    )
    checks["final_20k_array_file_hashes_distinct"] = (
        final20_state["coordinates_sha256"] != final20["artifact_hashes"]["final_coordinates_3dg"]
    )
    checks["cpu_diagnostic_present"] = Path(fine["cpu_process_diagnostic"]["path"]).is_file()
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "audit_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "attempt": str(ATTEMPT),
        "checks": checks,
        "checkpoint_counts": checkpoint_counts,
        "final_20k": {
            "coordinates_path": final20["artifact_paths"]["final_coordinates_3dg"],
            "coordinates_file_sha256": final20["artifact_hashes"]["final_coordinates_3dg"],
            "coordinates_array_sha256": final20_state["coordinates_sha256"],
            "coordinates_state_json_sha256": sha256(final20_state_path),
            "n_loci": final20["budget"]["n_loci"],
            "n_eligible_pairs": final20["budget"]["n_eligible_pairs"],
            "scoped_end_to_end_seconds": final20["timing"]["scoped_end_to_end_seconds"],
            "final_count_nll_normalized": final20["final_count_nll_normalized"],
            "final_total": final20["final_total"],
            "max_radius": final20["final_coordinates"]["max_radius"],
        },
        "formal_optimization_started": True,
        "phase_or_reference_opened": False,
    }
    LOG.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
