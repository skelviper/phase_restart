"""Verify 020 warm-start output against the byte-copied archived implementation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

RUN = Path(__file__).resolve().parents[1]
WORKSPACE = RUN.parents[1]
sys.path.insert(0, str(RUN / "source"))
sys.path.insert(0, str(RUN / "source" / "archived020"))

from gpu_backend import _header_and_records  # noqa: E402
from run_pipeline import (  # noqa: E402
    expand_tracks,
    parse_3dg,
    warm_start_exact,
)
from pr.reconstruction_init import warm_start_from_layer  # noqa: E402

INPUT = WORKSPACE / "inputs" / "P9016.snpfree.pairs.gz"
OUT = RUN / "logs" / "warm_start_validation.json"


def main() -> None:
    names, lengths = _header_and_records(INPUT)
    results = []
    for candidate, base_seed in (("consensus", 1103), ("random", 2207)):
        source = RUN / "coords" / "initial_sources" / f"{candidate}_joint-initial-5m.3dg"
        tracks = parse_3dg(source, lengths)
        coords, positions, chromosomes = expand_tracks(tracks, lengths, 5_000_000)
        for target_bin in (2_000_000, 1_000_000):
            ours = warm_start_exact(coords, positions, chromosomes, names, lengths, target_bin, base_seed)
            archived = warm_start_from_layer(
                coords, positions, chromosomes, names, tuple(int(v) for v in lengths),
                target_bin, base_seed,
            )
            checks = {
                "candidate": candidate,
                "base_seed": base_seed,
                "source_bin_size": 5_000_000 if target_bin == 2_000_000 else 2_000_000,
                "target_bin_size": target_bin,
                "coordinates_max_abs_difference": float(np.max(np.abs(ours[0] - archived["coords"]))),
                "positions_identical": bool(np.array_equal(ours[1], archived["positions"])),
                "chromosomes_identical": bool(np.array_equal(ours[2], archived["chromosome_index"])),
                "metadata_seed_identical": int(ours[3]["seed"]) == int(archived["metadata"]["seed"]),
                "metadata_perturbed_identical": int(ours[3]["perturbation"]["perturbed_coordinates"])
                == int(archived["metadata"]["perturbation"]["perturbed_coordinates"]),
            }
            checks["passed"] = (
                checks["coordinates_max_abs_difference"] == 0.0
                and checks["positions_identical"]
                and checks["chromosomes_identical"]
                and checks["metadata_seed_identical"]
                and checks["metadata_perturbed_identical"]
            )
            results.append(checks)
            coords, positions, chromosomes = ours[:3]
    payload = {
        "status": "passed" if all(row["passed"] for row in results) else "failed",
        "source_implementation": "source/archived020/pr/reconstruction_init.py",
        "transitions": results,
        "phase_or_reference_opened": False,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
