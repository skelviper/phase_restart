#!/usr/bin/env python3
"""Validate serialized Hickit imputation output before the single FDG fit."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

EXPECTED_TOTAL = 1_703_888
THRESHOLD = 0.75
SERIALIZED_SUM_TOLERANCE = 0.0020001
MAXIMUM_BINS = (0.25, 0.50, 0.75, 0.90, 0.99, 1.0000001)


def bucket(chr1: str, chr2: str) -> str:
    return "physical_cis" if chr1 == chr2 else "physical_inter"


def main(path_text: str) -> int:
    path = Path(path_text)
    columns = None
    indices = None
    totals = Counter()
    retained = Counter()
    fully_labelled = Counter()
    fully_labelled_retained = Counter()
    probability_sum_bad = 0
    probability_outside_unit_interval = 0
    nonfinite_probability = 0
    maximum_histogram = Counter()
    chromosomes = set()
    split_tracks = set()
    rows = 0
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#columns:"):
                columns = line.rstrip("\n").split(":", 1)[1].split()
                required = ["chr1", "chr2", "phase0", "phase1", "phase_prob00", "phase_prob01", "phase_prob10", "phase_prob11"]
                missing = [name for name in required if name not in columns]
                if missing:
                    raise RuntimeError("missing required columns: %s" % ",".join(missing))
                indices = {name: columns.index(name) for name in required}
                continue
            if not line or line.startswith("#"):
                continue
            if columns is None or indices is None:
                raise RuntimeError("record before #columns declaration")
            fields = line.rstrip("\n").split("\t")
            chr1 = fields[indices["chr1"]]
            chr2 = fields[indices["chr2"]]
            kind = bucket(chr1, chr2)
            probabilities = [float(fields[indices[name]]) for name in ("phase_prob00", "phase_prob01", "phase_prob10", "phase_prob11")]
            rows += 1
            totals[kind] += 1
            chromosomes.update((chr1, chr2))
            is_fully_labelled = fields[indices["phase0"]] in {"0", "1"} and fields[indices["phase1"]] in {"0", "1"}
            if is_fully_labelled:
                fully_labelled[kind] += 1
            if not all(math.isfinite(value) for value in probabilities):
                nonfinite_probability += 1
                continue
            if any(value < 0.0 or value > 1.0 for value in probabilities):
                probability_outside_unit_interval += 1
            if abs(sum(probabilities) - 1.0) > SERIALIZED_SUM_TOLERANCE:
                probability_sum_bad += 1
            maximum = max(probabilities)
            for upper in MAXIMUM_BINS:
                if maximum < upper:
                    maximum_histogram["<%.2f" % upper] += 1
                    break
            if maximum >= THRESHOLD:
                retained[kind] += 1
                if is_fully_labelled:
                    fully_labelled_retained[kind] += 1
                state = probabilities.index(maximum)
                split_tracks.add(chr1 + ("a" if (state >> 1) == 0 else "b"))
                split_tracks.add(chr2 + ("a" if (state & 1) == 0 else "b"))
    excluded = {kind: totals[kind] - retained[kind] for kind in ("physical_cis", "physical_inter")}
    report = {
        "input": str(path),
        "threshold": THRESHOLD,
        "header_columns": columns,
        "records": rows,
        "physical_counts": {kind: totals[kind] for kind in ("physical_cis", "physical_inter")},
        "fully_labelled_physical_counts": {kind: fully_labelled[kind] for kind in ("physical_cis", "physical_inter")},
        "split_threshold": {
            "retained": {kind: retained[kind] for kind in ("physical_cis", "physical_inter")},
            "retained_fully_labelled": {kind: fully_labelled_retained[kind] for kind in ("physical_cis", "physical_inter")},
            "excluded": excluded,
            "retained_total": sum(retained.values()),
            "excluded_total": sum(excluded.values())
        },
        "probability_checks": {
            "serialization_sum_tolerance": SERIALIZED_SUM_TOLERANCE,
            "nonfinite_rows": nonfinite_probability,
            "outside_closed_unit_interval_rows": probability_outside_unit_interval,
            "sum_outside_serialization_tolerance_rows": probability_sum_bad,
            "max_probability_histogram": {"<%.2f" % upper: maximum_histogram["<%.2f" % upper] for upper in MAXIMUM_BINS}
        },
        "physical_chromosomes": sorted(chromosomes),
        "physical_chromosome_count": len(chromosomes),
        "expected_split_tracks_from_serialized_probabilities": sorted(split_tracks),
        "expected_split_track_count": len(split_tracks)
    }
    report["gate_pass"] = (
        rows == EXPECTED_TOTAL
        and len(chromosomes) == 20
        and len(split_tracks) == 40
        and nonfinite_probability == 0
        and probability_outside_unit_interval == 0
        and probability_sum_bad == 0
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: pre_fit_gate.py IMPUTED_PAIRS")
    raise SystemExit(main(sys.argv[1]))
