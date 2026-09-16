"""Repair only derived coordinate-hash labels in the fine summary.

Formal stage outputs are read-only. The summary is rewritten so canonical array
hashes from resume metadata cannot be confused with raw .3dg file hashes.
"""
from __future__ import annotations

import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[2]
ATTEMPT = RUN / "backend_runs" / "fused-cuda" / "attempt-20260913T165036Z"
SUMMARY = ATTEMPT / "provenance" / "fine-continuation-summary.json"


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for row in payload["stages"]:
        state_path = Path(row["artifact_paths"]["resume_state_json"])
        state = json.loads(state_path.read_text(encoding="utf-8"))
        coordinates = dict(row["final_coordinates"])
        file_hash = coordinates.pop("sha256", None)
        if file_hash is None:
            file_hash = coordinates.get("file_sha256", row["artifact_hashes"]["final_coordinates_3dg"])
        coordinates["file_sha256"] = file_hash
        coordinates["array_sha256"] = state["coordinates_sha256"]
        row["final_coordinates"] = coordinates
    payload["coordinate_hash_semantics"] = {
        "array_sha256": "canonical run_pipeline.array_sha256 of the saved coordinate array in resume state",
        "file_sha256": "SHA256 of the emitted .3dg coordinate file bytes",
    }
    payload["derived_hash_labels_repaired"] = True
    SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "repaired_derived_summary_only",
        "summary": str(SUMMARY),
        "stages": {
            row["stage"]: {
                "array_sha256": row["final_coordinates"]["array_sha256"],
                "file_sha256": row["final_coordinates"]["file_sha256"],
            }
            for row in payload["stages"]
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
