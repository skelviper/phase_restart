"""已发布 synthetic 20-fit run 的最终完整性 gate。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


GATE_SCHEMA = "post020-synthetic-terminal-gate-v1"
TERMINAL_STATUSES = {"solver_converged", "budget_not_converged", "solver_failed"}
FINITE_ENDPOINT_STATUSES = TERMINAL_STATUSES


class GateError(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise GateError("invalid or missing gate input: %s" % path) from exc
    if not isinstance(value, dict):
        raise GateError("gate input is not a JSON object: %s" % path)
    return value


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _verify_terminal_row(root: Path, row: Mapping[str, Any]) -> bool:
    if row.get("status") not in TERMINAL_STATUSES or not row.get("selection_eligible"):
        return False
    final_path = root / str(row.get("final_json_path", ""))
    if not final_path.is_file() or _sha256(final_path) != str(row.get("final_json_sha256")):
        return False
    final = _read(final_path)
    if final.get("status") not in TERMINAL_STATUSES:
        return False
    endpoint = final.get("final", {})
    coordinate = endpoint.get("coordinates_file", {})
    coordinate_path = root / str(coordinate.get("path", ""))
    if not coordinate_path.is_file() or _sha256(coordinate_path) != str(coordinate.get("sha256")):
        return False
    for key in ("fun", "p", "q", "gradient_l2", "gradient_inf"):
        if not _finite(endpoint.get(key)):
            return False
    for key in ("theta_sha256", "coordinates_sha256", "theta_text_sha256", "state_npz_sha256"):
        if not endpoint.get(key):
            return False
    return True


def write_implementation_gate(run_dir: str | Path, *, protocol_sha256: str) -> dict[str, Any]:
    """仅在拟合完整且评价修正后写出通过的 gate。"""
    root = Path(run_dir).resolve()
    fit_path = root / "results" / "fit_manifest.json"
    evaluation_path = root / "results" / "evaluation.json"
    fit = _read(fit_path)
    evaluation = _read(evaluation_path)
    rows = fit.get("jobs")
    if fit.get("status") != "complete_terminal" or not isinstance(rows, list) or len(rows) != 20:
        raise GateError("fit manifest is not complete_terminal with 20 rows")
    if str(fit.get("protocol_release_sha256")) != str(protocol_sha256).lower():
        raise GateError("fit manifest protocol hash differs from released hash")
    valid_rows = [_verify_terminal_row(root, row) for row in rows]
    if not all(valid_rows):
        raise GateError("at least one terminal endpoint is nonfinite or not hash locked")
    if evaluation.get("status") != "complete" or evaluation.get("candidate_count") != 20:
        raise GateError("corrected evaluation is not complete for all 20 candidates")
    candidate_info = fit.get("candidate_manifest")
    if not isinstance(candidate_info, Mapping):
        raise GateError("fit manifest lacks candidate manifest")
    candidate_path = root / str(candidate_info.get("path", ""))
    if not candidate_path.is_file() or _sha256(candidate_path) != str(candidate_info.get("sha256")):
        raise GateError("candidate manifest hash is not locked")
    if root / "results" / "implementation_gate.json" in (fit_path, evaluation_path):
        raise GateError("invalid gate path")
    gate = {
        "schema": GATE_SCHEMA,
        "status": "passed",
        "run_id": root.name,
        "protocol_json_sha256": str(protocol_sha256).lower(),
        "planned_fit_count": 20,
        "terminal_fit_count": len(rows),
        "finite_valid_endpoint_count": sum(valid_rows),
        "coordinates_hash_locked": True,
        "numerical_checks_passed": True,
        "evaluation_complete": True,
        "terminal_manifest_path": str(fit_path.relative_to(root)),
        "terminal_manifest_sha256": _sha256(fit_path),
        "evaluation_path": str(evaluation_path.relative_to(root)),
        "evaluation_sha256": _sha256(evaluation_path),
        "candidate_manifest_path": str(candidate_path.relative_to(root)),
        "candidate_manifest_sha256": _sha256(candidate_path),
        "fit_status_counts": {
            "solver_converged": sum(row.get("status") == "solver_converged" for row in rows),
            "budget_not_converged": sum(row.get("status") == "budget_not_converged" for row in rows),
            "solver_failed_finite_endpoint": sum(row.get("status") == "solver_failed" for row in rows),
        },
        "gate_policy": "completeness/integrity only; no metric threshold, seed tuning, or model selection",
        "truth_access_policy": "evaluation only after endpoint and candidate hashes; no truth in training tasks",
    }
    output_path = root / "results" / "implementation_gate.json"
    if output_path.exists():
        raise GateError("refusing to overwrite existing implementation gate")
    output_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate


__all__ = ["GATE_SCHEMA", "GateError", "write_implementation_gate"]
