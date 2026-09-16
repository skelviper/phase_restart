#!/usr/bin/env python
"""Read-only dependency provenance audit for evaluation_final."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent.parent
ROOT = RUN.parents[1]
SOURCE = RUN.parent / "045-20260915T073310Z-shared-capture-round" / "source"
for path in (ROOT, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    import pr.r2comparison as live_r2
    import pr.contact_model as live_contact
    import pr.score as live_score
    modules = (live_r2, live_contact, live_score)
    rows = []
    for module in modules:
        path = Path(module.__file__).resolve()
        rows.append({"module": module.__name__, "module_file": str(path), "sha256": sha(path)})
    legacy_r2 = ROOT / "pr/r2comparison.py"
    legacy_allele = ROOT / "docs/audits/visibility-r2-preparation-20260914T164603Z/frozen_pr/allele_r2.py"
    frozen_pyc = ROOT / "docs/audits/visibility-r2-preparation-20260914T164603Z/frozen_pr/r2comparison.pyc"
    rows.append({"module": "legacy_builder_expected", "module_file": str(legacy_r2), "actual_sha256": sha(legacy_r2), "frozen_expected_sha256": "2979118aed25ac571d8f476eec4f6ec7687649f5a3e0fc43618194a293b4ed6e", "byte_identical": sha(legacy_r2) == "2979118aed25ac571d8f476eec4f6ec7687649f5a3e0fc43618194a293b4ed6e"})
    rows.append({"module": "frozen_coordinate_parser", "module_file": str(legacy_allele), "sha256": sha(legacy_allele), "expected_sha256": "56dacfe9a3edc13db4804fa7d5ce7c40b9401a1d1dd62ddbaa66d0d6ad5b1672", "byte_identical": sha(legacy_allele) == "56dacfe9a3edc13db4804fa7d5ce7c40b9401a1d1dd62ddbaa66d0d6ad5b1672"})
    rows.append({"module": "frozen_legacy_builder", "module_file": str(frozen_pyc), "sha256": sha(frozen_pyc), "loaded_by": "SourcelessFileLoader-compatible snapshot loader", "selected_because_live_r2comparison_mismatch": True})
    result = {"schema": "p9016-real-evaluation-final-dependency-audit-v1", "read_only": True, "live_modules": rows, "legacy_builder_policy": "live r2comparison is not byte-identical to frozen expected; final evaluator must use frozen r2comparison.pyc for legacy parity", "training_not_started": True, "reference_not_read": True}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if rows[3]["byte_identical"] is False and rows[4]["byte_identical"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
