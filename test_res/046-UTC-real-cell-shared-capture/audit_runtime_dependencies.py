"""Read-only runtime dependency/hash audit for 045 real-only base reuse."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parents[1] / "045-20260915T073310Z-shared-capture-round"
SOURCE = RUN / "source"
ROOT = RUN.parents[1]
for path in (SOURCE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import formal_controller as controller  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    frozen = config["immutable_hashes"]
    import data_io
    import shared_capture_objective
    import visibility_profile_base
    import gpu_variant_backend
    import m1_preconditioner
    import pr.contact_model
    import pr.reconstruction_init
    modules = {
        "source/formal_controller.py": controller,
        "source/data_io.py": data_io,
        "source/shared_capture_objective.py": shared_capture_objective,
        "source/visibility_profile_base.py": visibility_profile_base,
        "source/frozen_035/gpu_variant_backend.py": gpu_variant_backend,
        "source/frozen_037/m1_preconditioner.py": m1_preconditioner,
        "source/frozen_pr/pr/contact_model.py": sys.modules["pr.contact_model"],
        "source/frozen_pr/pr/reconstruction_init.py": sys.modules["pr.reconstruction_init"],
    }
    records = []
    mismatches = []
    for expected_key, module in modules.items():
        path = Path(str(module.__file__)).resolve()
        actual = sha(path)
        expected = frozen.get(expected_key)
        record = {"expected_key": expected_key, "runtime_file": str(path), "runtime_sha256": actual,
                  "pinned_sha256": expected, "sha_match": expected == actual}
        records.append(record)
        if expected != actual:
            mismatches.append(record)
    output = {"schema": "p9016-real-runtime-dependency-audit-v1", "status": "PASS" if not mismatches else "FAIL",
              "source_run": str(RUN), "modules": records, "mismatch_count": len(mismatches), "mismatches": mismatches,
              "synthetic_paths_opened": False, "reference_opened": False, "phase_opened": False,
              "formal_matrix_source": "045 preflight immutable_hashes; real-only runner does not import evaluator"}
    path = Path(__file__).resolve().parent / "runtime_dependency_audit.json"
    path.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "checked": len(records), "mismatch_count": len(mismatches)}))


if __name__ == "__main__":
    main()
