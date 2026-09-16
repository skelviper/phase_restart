"""Audit frozen P9016 header and raw budget at every registered grid."""
from __future__ import annotations

import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parents[1]
WORKSPACE = RUN.parents[1]
sys.path.insert(0, str(RUN / "source"))
from gpu_backend import (  # noqa: E402
    BackendError,
    FROZEN_BIN_SIZES,
    FROZEN_CHROMOSOME_HEADER,
    FROZEN_INTRA,
    FROZEN_INTER,
    FROZEN_RECORDS,
    FROZEN_SNPFREE_SHA256,
    load_sparse_aggregate,
    sha256_file,
)

INPUT = WORKSPACE / "inputs" / "P9016.snpfree.pairs.gz"
OUT = RUN / "logs" / "cohort_resolution_validation.json"


def main() -> None:
    actual_hash = sha256_file(INPUT)
    if actual_hash != FROZEN_SNPFREE_SHA256:
        raise BackendError("input hash mismatch before resolution audit")
    rows = {}
    for bin_size in FROZEN_BIN_SIZES:
        data = load_sparse_aggregate(INPUT, bin_size=bin_size, verify_hash=True)
        header = tuple(zip(data.chromosome_names, (int(v) for v in data.chromosome_lengths)))
        audit = data.budget()
        checks = {
            "header_identical": header == FROZEN_CHROMOSOME_HEADER,
            "raw_records": audit["raw_records"],
            "raw_records_identical": audit["raw_records"] == FROZEN_RECORDS,
            "raw_intra_identical": audit["raw_same_bin"] + audit["raw_cis_offdiag"] == FROZEN_INTRA,
            "raw_inter_identical": audit["raw_inter"] == FROZEN_INTER,
            "aggregate_same_bin_identical": audit["aggregate_same_bin"] == audit["raw_same_bin"],
            "aggregate_offdiag_identical": audit["aggregate_offdiag"] == audit["raw_cis_offdiag"] + audit["raw_inter"],
            "full_grid": audit["n_eligible_pairs"] == data.n_loci * (data.n_loci - 1) // 2,
            "factorial_constants_recorded": audit["factorial_constant_status"] == "integer_count_constants_recorded",
            "budget": audit,
        }
        checks["passed"] = all(value for key, value in checks.items() if key not in ("budget", "raw_records"))
        rows[str(bin_size)] = checks
    invalid_rejection = None
    try:
        load_sparse_aggregate(INPUT, bin_size=123_456, verify_hash=False)
    except BackendError as exc:
        invalid_rejection = str(exc)
    payload = {
        "status": "passed" if all(row["passed"] for row in rows.values()) and invalid_rejection else "failed",
        "input_sha256": actual_hash,
        "frozen_header": list(FROZEN_CHROMOSOME_HEADER),
        "frozen_records": FROZEN_RECORDS,
        "frozen_intra": FROZEN_INTRA,
        "frozen_inter": FROZEN_INTER,
        "resolutions": rows,
        "invalid_bin_size_rejection": invalid_rejection,
        "phase_or_reference_opened": False,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
