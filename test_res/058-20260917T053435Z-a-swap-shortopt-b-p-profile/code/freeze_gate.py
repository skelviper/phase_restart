"""058 所有正式终态的 evaluation 前哈希门。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent.parent
ROOT = RUN.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pr.solver_state import load_solver_state, sha256_file


def main():
    A = json.loads((RUN / "logs/A_terminal.json").read_text())
    B = json.loads((RUN / "logs/B_terminal.json").read_text())
    if A.get("status") != "complete" or B.get("status") != "complete":
        raise RuntimeError("formal A/B terminals are incomplete")
    artifacts = []
    endpoint_count = 0
    for branch, rows in (("A", A["arms"]), ("B", B["endpoints"])):
        for row in rows:
            endpoint_id = row.get("arm_id", row.get("endpoint_id"))
            endpoint_count += 1
            art = row["artifacts"]
            for role, key in (("solver_state", "solver_state"),
                              ("presence_mask", "presence_mask"), ("export", "export")):
                path = Path(art[key]["path"])
                if not path.is_absolute():
                    path = ROOT / path
                actual = sha256_file(path)
                expected = art[key]["sha256"]
                if actual != expected:
                    raise RuntimeError(f"artifact hash mismatch {endpoint_id}/{role}")
                artifacts.append({"branch": branch, "endpoint_id": endpoint_id,
                                  "role": role, "path": str(path.relative_to(ROOT)),
                                  "sha256": actual})
            state_path = ROOT / artifacts[-3]["path"]
            loaded = load_solver_state(state_path)
            if not loaded["audit"]["finite"]:
                raise RuntimeError(f"nonfinite endpoint {endpoint_id}")
    if endpoint_count != 12 or len(artifacts) != 36:
        raise RuntimeError("endpoint/artifact inventory is not 12/36")
    selection = RUN / "results/A/selection.json"
    records = [RUN / "logs/A_terminal.json", RUN / "logs/B_terminal.json", selection]
    payload = {
        "schema": "p9016-058-pre-evaluation-hash-gate-v1", "status": "PASS",
        "endpoint_count": endpoint_count, "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "training_selection_records": [{"path": str(p.relative_to(ROOT)),
                                         "sha256": sha256_file(p)} for p in records],
        "all_actual_artifact_hashes_matched": True,
        "all_pending_states_hashed": True,
        "reference_opened": False, "phase_body_opened": False,
    }
    out = RUN / "results/pre_evaluation_hash_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "endpoints": endpoint_count,
                      "artifacts": len(artifacts)}))


if __name__ == "__main__":
    main()
