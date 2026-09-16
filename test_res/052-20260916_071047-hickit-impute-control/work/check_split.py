#!/usr/bin/env python3
"""Verify that the serialized split input equals the pre-fit probability prediction."""
import json
import sys
from collections import Counter
from pathlib import Path


def physical_name(track: str) -> str:
    if not track.endswith(("a", "b")):
        raise RuntimeError("split track lacks a/b suffix: %s" % track)
    return track[:-1]


def main(split_text: str, predicted_text: str) -> int:
    split_path, predicted_path = Path(split_text), Path(predicted_text)
    predicted = json.loads(predicted_path.read_text(encoding="utf-8"))["split_threshold"]["retained"]
    counts, tracks = Counter(), set()
    columns = None
    with split_path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#columns:"):
                columns = line.split(":", 1)[1].split()
                continue
            if not line or line.startswith("#"):
                continue
            if columns is None:
                raise RuntimeError("record before split #columns declaration")
            fields = line.rstrip("\n").split("\t")
            name1, name2 = fields[columns.index("chr1")], fields[columns.index("chr2")]
            physical1, physical2 = physical_name(name1), physical_name(name2)
            counts["physical_cis" if physical1 == physical2 else "physical_inter"] += 1
            tracks.update((name1, name2))
    actual = {key: counts[key] for key in ("physical_cis", "physical_inter")}
    report = {"input": str(split_path), "header_columns": columns, "physical_counts": actual, "records": sum(actual.values()), "tracks": sorted(tracks), "track_count": len(tracks), "predicted_from_serialized_imputation": predicted}
    report["gate_pass"] = actual == predicted and len(tracks) == 40
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_split.py SPLIT_PAIRS PRE_FIT_GATE_JSON")
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
