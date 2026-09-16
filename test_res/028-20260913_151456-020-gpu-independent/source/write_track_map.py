"""Generate the auditable 40-track map and validate the 20 kb full grid."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_STATE = RUN / "backend_runs" / "fused-cuda" / "attempt-20260913T165036Z" / "backend_runs" / "fused-cuda" / "resume" / "random_joint" / "1m-state.json"
DEFAULT_JSON = RUN / "provenance" / "track_map.json"
DEFAULT_TSV = RUN / "provenance" / "track_map.tsv"
DEFAULT_VALIDATION = RUN / "logs" / "grid_20kb_validation.json"
BIN_SIZE = 20_000


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    names = state.get("chromosome_names")
    lengths = state.get("chromosome_lengths")
    if not isinstance(names, list) or not isinstance(lengths, list) or len(names) != len(lengths):
        raise ValueError("state does not contain matching chromosome header arrays")
    if len(names) != 20 or names[-1] != "chrX":
        raise ValueError("state header is not the frozen 20-chromosome header")
    if state.get("phase_or_reference_opened") is not False:
        raise ValueError("state phase/reference provenance flag is not false")
    return state


def build_mapping(state: dict) -> list[dict]:
    rows = []
    locus_offset = 0
    track_index = 0
    for chromosome_index, (name, raw_length) in enumerate(zip(state["chromosome_names"], state["chromosome_lengths"], strict=True)):
        length = int(raw_length)
        n_bins = (length + BIN_SIZE - 1) // BIN_SIZE
        for copy_index, copy_label in enumerate(("a", "b")):
            rows.append({
                "track_index": track_index,
                "track_name": f"c{chromosome_index + 1:02d}{copy_label}",
                "chromosome_index": chromosome_index,
                "chromosome_name": name,
                "chromosome_length_bp": length,
                "copy_index": copy_index,
                "copy_label": copy_label,
                "bin_size_bp": BIN_SIZE,
                "first_position_bp": 0,
                "terminal_bin_index": n_bins - 1,
                "terminal_position_bp": (n_bins - 1) * BIN_SIZE,
                "terminal_bin_end_bp": min(length, n_bins * BIN_SIZE),
                "n_bins_20kb": n_bins,
                "locus_offset_20kb": locus_offset,
                "locus_index_first": locus_offset,
                "locus_index_last": locus_offset + n_bins - 1,
            })
            track_index += 1
        locus_offset += n_bins
    return rows


def validate_terminal_grid(stage_path: Path, rows: list[dict]) -> dict:
    expected_by_track = {row["track_name"]: row for row in rows}
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    final = stage.get("final_coordinates", {})
    coordinate_path = Path(final.get("path", ""))
    if not coordinate_path.is_absolute():
        coordinate_path = WORKSPACE / coordinate_path
    validation = {
        "stage_path": str(stage_path),
        "coordinate_path": str(coordinate_path),
        "stage": stage.get("stage"),
        "status": "failed",
        "n_tracks": final.get("n_tracks"),
        "full_grid": final.get("full_grid"),
        "track_counts": {},
        "errors": [],
    }
    if stage.get("stage") != "20k":
        validation["errors"].append("stage is not 20k")
    if final.get("n_tracks") != 40 or final.get("full_grid") is not True:
        validation["errors"].append("stage does not report full 40-track grid")
    if not coordinate_path.exists():
        validation["errors"].append("coordinate export is missing")
    else:
        if final.get("sha256") != file_sha256(coordinate_path):
            validation["errors"].append("coordinate export SHA256 mismatch")
        seen = {name: [] for name in expected_by_track}
        with coordinate_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 5 or fields[0] not in seen:
                    validation["errors"].append(f"unexpected row at line {line_no}")
                    continue
                try:
                    seen[fields[0]].append(int(fields[1]))
                    xyz = [float(value) for value in fields[2:5]]
                    if any(value != value for value in xyz):
                        validation["errors"].append(f"nonfinite coordinate at line {line_no}")
                except ValueError:
                    validation["errors"].append(f"malformed row at line {line_no}")
        for name, positions in seen.items():
            expected = expected_by_track[name]
            wanted = list(range(0, expected["n_bins_20kb"] * BIN_SIZE, BIN_SIZE))
            validation["track_counts"][name] = {
                "observed": len(positions),
                "expected": expected["n_bins_20kb"],
                "first_position_bp": positions[0] if positions else None,
                "last_position_bp": positions[-1] if positions else None,
                "expected_terminal_position_bp": expected["terminal_position_bp"],
                "positions_match_0_grid": positions == wanted,
            }
            if positions != wanted:
                validation["errors"].append(f"{name}: positions do not match 0-based 20 kb grid")
    validation["status"] = "passed" if not validation["errors"] else "failed"
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    args = parser.parse_args()
    state_path = args.state if args.state.is_absolute() else WORKSPACE / args.state
    state = load_state(state_path)
    rows = build_mapping(state)
    if len(rows) != 40 or len({row["track_name"] for row in rows}) != 40:
        raise ValueError("track map is not a unique 40-track map")
    for row in rows:
        if row["track_name"] != f"c{row['chromosome_index'] + 1:02d}{row['copy_label']}" or row["copy_index"] not in (0, 1):
            raise ValueError(f"track map order mismatch: {row}")
    map_payload = {
        "status": "header_derived",
        "source_state": str(state_path),
        "source_state_sha256": file_sha256(state_path),
        "input_sha256": state.get("input_sha256"),
        "config_sha256": state.get("config_sha256"),
        "header": [{"chromosome_index": i, "chromosome_name": name, "length_bp": int(length)}
                   for i, (name, length) in enumerate(zip(state["chromosome_names"], state["chromosome_lengths"], strict=True))],
        "copy_order": {"copy_index_0": "a", "copy_index_1": "b"},
        "export_order": "chromosome_index ascending, then copy_index 0/1; c01a..c20b",
        "bin_size_bp": BIN_SIZE,
        "total_20kb_loci": sum(row["n_bins_20kb"] for row in rows[::2]),
        "tracks": rows,
    }
    json_path = args.json if args.json.is_absolute() else WORKSPACE / args.json
    tsv_path = args.tsv if args.tsv.is_absolute() else WORKSPACE / args.tsv
    validation_path = args.validation if args.validation.is_absolute() else WORKSPACE / args.validation
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(map_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    columns = list(rows[0])
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[column]) for column in columns) + "\n")
    if args.stage is None:
        validation = {
            "status": "pending_terminal_20k",
            "reason": "header-derived 0-based grid contract written; supply --stage after 20k export",
            "expected_tracks": 40,
            "expected_total_20kb_loci": map_payload["total_20kb_loci"],
            "source_state": str(state_path),
        }
    else:
        stage_path = args.stage if args.stage.is_absolute() else WORKSPACE / args.stage
        validation = validate_terminal_grid(stage_path, rows)
        validation["source_state"] = str(state_path)
        validation["expected_total_20kb_loci"] = map_payload["total_20kb_loci"]
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": validation["status"], "json": str(json_path), "tsv": str(tsv_path), "validation": str(validation_path)}, indent=2))
    if validation["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
