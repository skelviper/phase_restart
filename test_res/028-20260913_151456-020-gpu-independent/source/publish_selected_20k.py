"""Publish the accepted selected 20 kb export without mutating the GPU attempt."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

RUN = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_ATTEMPT = RUN / "backend_runs" / "fused-cuda" / "attempt-20260913T165036Z"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--stage", type=Path)
    args = parser.parse_args()
    attempt = args.attempt if args.attempt.is_absolute() else WORKSPACE / args.attempt
    backend_root = attempt / "backend_runs" / "fused-cuda"
    selection_path = backend_root / "selection-1m.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"missing 1 Mb selection: {selection_path}")
    selection = load(selection_path)
    if selection.get("status") != "selected_after_both_candidates_1m":
        raise ValueError("selection is not the frozen two-candidate 1 Mb selection")
    selected = selection.get("selected_candidate")
    stage_path = args.stage if args.stage is not None else backend_root / "stages" / selected / "20k.json"
    if not stage_path.is_absolute():
        stage_path = WORKSPACE / stage_path
    stage = load(stage_path)
    if stage.get("stage") != "20k" or stage.get("candidate_id") != selected:
        raise ValueError("stage is not the selected accepted 20 kb stage")
    if stage.get("status") not in {"converged", "budget_not_converged"}:
        raise ValueError(f"20 kb stage is not an accepted terminal optimizer result: {stage.get('status')}")
    final = stage.get("final_coordinates", {})
    source_coord = Path(final.get("path", ""))
    if not source_coord.is_absolute():
        source_coord = WORKSPACE / source_coord
    if final.get("n_tracks") != 40 or final.get("full_grid") is not True:
        raise ValueError("selected 20 kb stage does not report a complete 40-track grid")
    if not source_coord.exists() or file_sha256(source_coord) != final.get("sha256"):
        raise ValueError("selected 20 kb coordinate export is missing or hash-invalid")
    state_ref = stage.get("resume_state", {})
    stage_backend_root = stage_path.parents[2]
    source_state_npz = stage_backend_root / state_ref["npz"]
    source_state_json = stage_backend_root / state_ref["json"]
    if not source_state_npz.exists() or not source_state_json.exists():
        raise FileNotFoundError("selected 20 kb saved state is missing")
    destination_coord = RUN / "coords" / "selected-final-20k.3dg"
    destination_npz = RUN / "coords" / "selected-final-20k-state.npz"
    destination_json = RUN / "coords" / "selected-final-20k-state.json"
    for destination in (destination_coord, destination_npz, destination_json):
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite friendly output: {destination}")
    destination_coord.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_coord, destination_coord)
    shutil.copyfile(source_state_npz, destination_npz)
    shutil.copyfile(source_state_json, destination_json)
    payload = {
        "status": "published_byte_identical_selected_20k",
        "artifact_integrity": "machine_generated_direct_json_dump",
        "attempt": str(attempt),
        "stage": str(stage_path),
        "selected_candidate": selected,
        "input_sha256": selection.get("input_sha256"),
        "config_sha256": selection.get("config_sha256"),
        "phase_or_reference_opened": stage.get("phase_or_reference_opened") is False,
        "source": {
            "coordinates": str(source_coord),
            "coordinates_sha256": file_sha256(source_coord),
            "state_npz": str(source_state_npz),
            "state_npz_sha256": file_sha256(source_state_npz),
            "state_json": str(source_state_json),
            "state_json_sha256": file_sha256(source_state_json),
        },
        "friendly_outputs": {
            "coordinates": str(destination_coord),
            "coordinates_sha256": file_sha256(destination_coord),
            "state_npz": str(destination_npz),
            "state_npz_sha256": file_sha256(destination_npz),
            "state_json": str(destination_json),
            "state_json_sha256": file_sha256(destination_json),
        },
        "copy_hashes_match": (
            file_sha256(source_coord) == file_sha256(destination_coord)
            and file_sha256(source_state_npz) == file_sha256(destination_npz)
            and file_sha256(source_state_json) == file_sha256(destination_json)
        ),
        "n_tracks": final.get("n_tracks"),
        "full_grid": final.get("full_grid"),
        "n_beads": final.get("n_beads"),
        "bin_size_bp": stage.get("bin_size_bp"),
    }
    if not payload["copy_hashes_match"]:
        raise RuntimeError("friendly output copy hash mismatch")
    out = RUN / "logs" / "selected_final_20k_publish.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "coordinates": str(destination_coord), "provenance": str(out)}, indent=2))


if __name__ == "__main__":
    main()
