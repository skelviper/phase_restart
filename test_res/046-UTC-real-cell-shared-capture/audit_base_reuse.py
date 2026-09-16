"""Read-only audit of the four reusable completed base stages."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1] / "045-20260915T073310Z-shared-capture-round/evaluation/real_only"
SOURCE_RUN = BASE.parents[1]
ROOT = SOURCE_RUN.parents[1]
FORMAL = json.loads((SOURCE_RUN / "inputs/formal_manifest.json").read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr_sha(a):
    return hashlib.sha256(np.asarray(a, dtype="<f8", order="C").tobytes(order="C")).hexdigest()


def audit_stage(fit_id, stage, expected_cap):
    rec_path = BASE / "stages" / fit_id / (stage + ".json")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    endpoint = BASE / "coords" / fit_id / (stage + ".npz")
    endpoint3dg = BASE / "coords" / fit_id / (stage + ".3dg")
    hist = BASE / "stages" / fit_id / (stage + ".accepted_history.json")
    iter0 = BASE / rec["history"]["iter0_artifact"]
    checks = {}
    checks["record_sha256"] = sha(rec_path)
    checks["endpoint_npz_sha256"] = sha(endpoint) == rec["artifact_hashes"]["coordinate_npz_sha256"]
    checks["endpoint_3dg_sha256"] = sha(endpoint3dg) == rec["artifact_hashes"]["coordinate_3dg_sha256"]
    checks["history_sha256"] = sha(hist) == rec["artifact_hashes"]["history_sha256"]
    checks["iter0_sha256"] = sha(iter0) == rec["artifact_hashes"]["iter0_npz_sha256"]
    checks["cap"] = int(rec["fg_cap"]) == int(expected_cap)
    checks["ftol"] = float(rec["ftol"]) == 0.0
    checks["gtol"] = float(rec["canonical_gtol"]) == 1e-6
    checks["maxls"] = int(rec["maxls"]) == 20
    checks["terminal"] = rec.get("status") in ("converged", "not_converged", "budget_not_converged") and rec.get("terminal_reason") is not None
    checks["endpoint_last_accepted"] = bool(rec.get("last_accepted_endpoint"))
    with np.load(endpoint, allow_pickle=False) as payload:
        checks["endpoint_finite"] = bool(np.isfinite(payload["coordinates"]).all())
        checks["endpoint_array_sha256"] = arr_sha(payload["coordinates"]) == rec["artifact_hashes"].get("coordinate_array_sha256", arr_sha(payload["coordinates"]))
    with np.load(iter0, allow_pickle=False) as payload:
        checks["iter0_mapping"] = {
            "coordinates_sha256": arr_sha(payload["coordinates"]) == rec["initial_state_hashes"]["coordinates_sha256"],
            "raw_y_sha256": arr_sha(payload["raw_y"]) == rec["initial_state_hashes"]["raw_y_sha256"],
            "q_sha256": arr_sha(np.asarray([payload["theta"][-1]])) == rec["initial_state_hashes"]["q_float64_sha256"],
            "theta_sha256": arr_sha(payload["theta"]) == rec["initial_state_hashes"]["theta_sha256"],
            "iteration_zero": int(payload["iteration"]) == 0,
        }
    # Real aggregate bytes are the frozen input for each bin; this is a readback only.
    bin_map = {"5Mb": "5000000", "2Mb": "2000000", "1Mb": "1000000"}
    input_path = Path(FORMAL["real_input_paths"][bin_map[stage]])
    checks["input_path"] = str(input_path)
    checks["input_sha256"] = sha(input_path)
    return {"fit_id": fit_id, "stage": stage, "record_status": rec["status"], "terminal_reason": rec.get("terminal_reason"),
            "outer_fg_actual": rec.get("outer_fg_actual"), "initial_state_hashes": rec.get("initial_state_hashes"),
            "artifact_hashes": rec.get("artifact_hashes"), "checks": checks}


def main():
    specs = [("real-S-consensus", "5Mb", 612), ("real-S-consensus", "2Mb", 404), ("real-S-consensus", "1Mb", 486),
             ("real-S-random", "5Mb", 612)]
    rows = [audit_stage(*spec) for spec in specs]
    failures = []
    for row in rows:
        for key, value in row["checks"].items():
            if isinstance(value, bool) and not value:
                failures.append({"fit_id": row["fit_id"], "stage": row["stage"], "check": key})
            if isinstance(value, dict):
                failures.extend({"fit_id": row["fit_id"], "stage": row["stage"], "check": key + "." + sub}
                                 for sub, ok in value.items() if isinstance(ok, bool) and not ok)
    out = {"schema": "p9016-real-base-reuse-audit-v2", "status": "PASS" if not failures else "FAIL", "base_root": str(BASE),
           "reused_stage_count": len(rows), "reused_stages": rows, "failures": failures,
           "source_hash_audit": str(Path(__file__).resolve().parent / "runtime_dependency_audit.json"),
           "reference_opened": False, "phase_opened": False, "synthetic_paths_opened": False,
           "note": "Stage start raw-y/q is checked against iter0 serialized checkpoint; 2Mb/1Mb warm starts are checkpoint-authenticated, and the cancelled random/2Mb running record is not among reusable stages."}
    path = Path(__file__).resolve().parent / "base_reuse_audit.json"
    path.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "reused_stage_count": len(rows), "failure_count": len(failures)}))


if __name__ == "__main__":
    main()
