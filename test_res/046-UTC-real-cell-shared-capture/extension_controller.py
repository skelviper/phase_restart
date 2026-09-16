"""Run the four real P9016 extension arms for scope-revised run 046."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

RUN = Path(__file__).resolve().parent
SOURCE_RUN = RUN.parent / "045-20260915T073310Z-shared-capture-round"
SOURCE = SOURCE_RUN / "source"
ROOT = RUN.parents[2]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import formal_controller as controller  # noqa: E402

FULL_WEIGHTS = {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}
COUNT_WEIGHTS = {"count": 1.0, "bond": 0.0, "repulsion": 0.0, "bend": 0.0, "p_prior": 0.0}
_LOCK_HANDLE = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _acquire_single_writer_lock() -> dict[str, Any]:
    global _LOCK_HANDLE
    lock_path = RUN / "extension_writer.lock"
    launch = {"pid": os.getpid(), "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "pid_start_ticks": None, "command": "extension_controller.py run"}
    try:
        launch["pid_start_ticks"] = (Path("/proc") / str(os.getpid()) / "stat").read_text(encoding="utf-8").split()[21]
    except (FileNotFoundError, IndexError, OSError):
        launch["pid_start_ticks"] = "unavailable"
    if os.environ.get("P9016_EXTENSION_LOCK_HELD") == "1":
        _LOCK_HANDLE = None  # The wrapper's inherited descriptor owns the flock.
    else:
        handle = lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError("extension writer lock is held by another process") from exc
        _LOCK_HANDLE = handle
    lock_path.touch(exist_ok=True)
    _write(RUN / "extension_runtime.json", {"status": "running", **launch, "lock": str(lock_path),
                                               "lock_method": "fcntl.flock(LOCK_EX|LOCK_NB)"})
    return launch


def _audit_base(old_manifest: dict[str, Any], base_root: Path) -> dict[str, Any]:
    base_config = json.loads((base_root / "config.json").read_text(encoding="utf-8"))
    if base_config.get("status") != "real_formal_complete" or int(base_config.get("candidate_count", 0)) != 4:
        raise RuntimeError("base real-only run is not complete with four endpoints")
    rows = []
    for fit in old_manifest["matrix"]:
        if fit.get("kind") != "real":
            continue
        fit_id = fit["fit_id"]
        for stage in ("5Mb", "2Mb", "1Mb"):
            record_path = base_root / "stages" / fit_id / (stage + ".json")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            endpoint_npz = base_root / "coords" / fit_id / (stage + ".npz")
            endpoint_3dg = base_root / "coords" / fit_id / (stage + ".3dg")
            if record.get("status") not in ("converged", "not_converged", "budget_not_converged") or not endpoint_npz.is_file() or not endpoint_3dg.is_file():
                raise RuntimeError("base stage incomplete: %s/%s" % (fit_id, stage))
            start_record = record.get("initial_state_hashes", {})
            rows.append({"fit_id": fit_id, "stage": stage, "status": record["status"], "fg_cap": record["fg_cap"],
                         "outer_fg_actual": record.get("outer_fg_actual"), "terminal_reason": record.get("terminal_reason"),
                         "initial_raw_y_sha256": start_record.get("raw_y_sha256"), "initial_q": record.get("initial_q"),
                         "endpoint_npz": str(endpoint_npz), "endpoint_npz_sha256": _sha(endpoint_npz),
                         "endpoint_3dg": str(endpoint_3dg), "endpoint_3dg_sha256": _sha(endpoint_3dg),
                         "history_sha256": record.get("artifact_hashes", {}).get("history_sha256"),
                         "iter0_sha256": record.get("artifact_hashes", {}).get("iter0_npz_sha256")})
    audit = {"schema": "p9016-real-base-audit-v1", "source_run": str(base_root), "base_config_status": base_config["status"],
             "candidate_manifest_sha256": _sha(base_root / "results/candidate_manifest.json"), "stage_count": len(rows),
             "stages": rows, "reference_opened": False, "phase_opened": False}
    _write(RUN / "base_audit.json", audit)
    return audit


def _selected_start(base_root: Path, model: str, selection: dict[str, Any]) -> dict[str, Any]:
    selected = selection["models"][model]["selected_by_own_count_nll"]
    if selected not in ("consensus", "random"):
        raise RuntimeError("no count-selected base source for model %s" % model)
    fit_id = "real-%s-%s" % (model, selected)
    endpoint = base_root / "coords" / fit_id / "1Mb.npz"
    with np.load(endpoint, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        raw_y = np.asarray(payload["raw_y"], dtype=np.float64).copy()
        p = float(np.asarray(payload["p"]).item())
        q = float(np.asarray(payload["q"]).item())
    if not np.isfinite(coordinates).all() or not np.isfinite(raw_y).all():
        raise RuntimeError("selected base endpoint nonfinite")
    return {"model": model, "candidate": selected, "fit_id": fit_id, "endpoint": endpoint,
            "endpoint_sha256": _sha(endpoint), "coordinate_array_sha256": controller._hash_array(coordinates),
            "raw_y_sha256": controller._hash_array(raw_y), "coordinates": coordinates, "raw_y": raw_y, "p": p, "q": q}


def _audit_combined(combined: dict[str, Any]) -> dict[str, Any]:
    if combined.get("schema") != "p9016-real-combined-base-manifest-v1" or combined.get("stage_count") != 12 or combined.get("endpoint_count") != 4:
        raise RuntimeError("combined_base_manifest schema/count mismatch")
    rows = []
    for item in combined["stages"]:
        record_path, endpoint, endpoint3dg = RUN / item["record_path"], RUN / item["endpoint_npz"], RUN / item["endpoint_3dg"]
        if item.get("status") not in ("converged", "not_converged", "budget_not_converged"):
            raise RuntimeError("combined base stage is not terminal: %s/%s" % (item["fit_id"], item["stage"]))
        checks = {"record": _sha(record_path) == item["record_sha256"], "endpoint": _sha(endpoint) == item["endpoint_npz_sha256"],
                  "endpoint_3dg": _sha(endpoint3dg) == item["endpoint_3dg_sha256"]}
        if not all(checks.values()):
            raise RuntimeError("combined base hash failure %s/%s: %r" % (item["fit_id"], item["stage"], checks))
        rows.append({"fit_id": item["fit_id"], "stage": item["stage"], "status": item["status"],
                     "record": str(record_path), "endpoint": str(endpoint), "record_sha256": _sha(record_path),
                     "endpoint_sha256": _sha(endpoint), "outer_fg_actual": item["outer_fg_actual"], "fg_cap": item["fg_cap"]})
    audit = {"schema": "p9016-real-combined-base-audit-v1", "manifest": "combined_base_manifest.json", "stage_count": len(rows),
             "stages": rows, "reference_opened": False, "phase_opened": False, "synthetic_fits": 0}
    _write(RUN / "combined_base_audit.json", audit)
    return audit


def _selected_start_combined(combined: dict[str, Any], model: str) -> dict[str, Any]:
    selected = combined["selected_sources"][model]["selected_by_own_count_nll"]
    if selected not in ("consensus", "random"):
        raise RuntimeError("no combined source for model %s" % model)
    fit_id = "real-%s-%s" % (model, selected)
    endpoint = RUN / combined["endpoints"][fit_id]["endpoint_npz"]
    if _sha(endpoint) != combined["endpoints"][fit_id]["endpoint_npz_sha256"]:
        raise RuntimeError("selected combined endpoint hash mismatch")
    with np.load(endpoint, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        raw_y = np.asarray(payload["raw_y"], dtype=np.float64).copy()
        p, q = float(np.asarray(payload["p"]).item()), float(np.asarray(payload["q"]).item())
    if not np.isfinite(coordinates).all() or not np.isfinite(raw_y).all():
        raise RuntimeError("selected combined endpoint nonfinite")
    return {"model": model, "candidate": selected, "fit_id": fit_id, "endpoint": endpoint,
            "endpoint_sha256": _sha(endpoint), "coordinate_array_sha256": controller._hash_array(coordinates),
            "raw_y_sha256": controller._hash_array(raw_y), "coordinates": coordinates, "raw_y": raw_y, "p": p, "q": q}


def run() -> int:
    launch = _acquire_single_writer_lock()
    combined = json.loads((RUN / "combined_base_manifest.json").read_text(encoding="utf-8"))
    old_manifest = json.loads((SOURCE_RUN / "inputs/formal_manifest.json").read_text(encoding="utf-8"))
    base_audit = _audit_combined(combined)
    starts = [_selected_start_combined(combined, model) for model in ("S", "G")]
    _write(RUN / "extension_manifest.json", {"schema": "p9016-real-extension-manifest-v1", "scope": "P9016 real cell only",
                                               "base_manifest": "combined_base_manifest.json", "base_manifest_sha256": _sha(RUN / "combined_base_manifest.json"),
                                               "base_audit": "combined_base_audit.json", "base_audit_sha256": _sha(RUN / "combined_base_audit.json"),
                                               "selected_sources": {item["model"]: item["candidate"] for item in starts},
                                               "arms": [{"fit_id": "real-extension-%s-%s" % (item["model"], variant), "model": item["model"],
                                                         "source_candidate": item["candidate"], "variant": variant, "stage": "1Mb", "fg_cap": 486,
                                                         "weights": FULL_WEIGHTS if variant == "full-J" else COUNT_WEIGHTS,
                                                         "start_endpoint": str(item["endpoint"]), "start_endpoint_sha256": item["endpoint_sha256"],
                                                         "start_raw_y_sha256": item["raw_y_sha256"], "start_q": item["q"]}
                                                        for item in starts for variant in ("full-J", "count-only")],
                                               "formal_totals": {"base_fits": 4, "base_stages": 12, "base_outer_fg": 6008,
                                                                 "extension_fits": 4, "extension_stages": 4, "extension_outer_fg": 1944,
                                                                 "fits": 8, "stages": 16, "outer_fg": 7952},
                                               "reference_opened": False, "phase_opened": False, "synthetic_fits": 0})
    _write(RUN / "config.json", {"schema": "p9016-real-cell-shared-capture-046-v1", "status": "prepared_extension",
                                  "extension_manifest": "extension_manifest.json", "base_manifest": "combined_base_manifest.json",
                                  "formal_started": False, "reference_opened": False, "phase_opened": False,
                                  "synthetic_fits": 0, "fits": 8, "stages": 16, "outer_fg_cap": 7952})
    controller.RUN = SOURCE_RUN
    data = controller._load_real_data(old_manifest, 1_000_000)
    controller.RUN = RUN
    for name in ("coords", "stages", "checkpoints", "logs", "results"):
        (RUN / name).mkdir(parents=True, exist_ok=True)
    controller._event({"event": "extension_start", "fits": 4, "stages": 4, "outer_fg": 1944, "synthetic_fits": 0,
                       "reference_opened": False, "at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    summaries = []
    for item in starts:
        for variant in ("full-J", "count-only"):
            fit_id = "real-extension-%s-%s" % (item["model"], variant)
            fit = {"fit_id": fit_id, "kind": "real", "model_id": item["model"], "candidate": item["candidate"],
                   "objective_variant": variant, "weights": FULL_WEIGHTS if variant == "full-J" else COUNT_WEIGHTS}
            stage = {"stage": "1Mb", "bin_size_bp": 1_000_000, "fg_cap": 486}
            controller._event({"event": "extension_stage_start", "fit_id": fit_id, "source_fit": item["fit_id"], "variant": variant})
            record = controller.stage_fit(fit, stage, data, item["coordinates"], item["p"], item["q"], None, initial_raw_y=item["raw_y"])
            summary = {"fit_id": fit_id, "kind": "real_extension", "model_id": item["model"], "candidate": item["candidate"],
                       "variant": variant, "status": record.get("status"), "terminal_reason": record.get("terminal_reason"),
                       "outer_fg_actual": int(record.get("outer_fg_actual", 0)),
                       "endpoint_npz_sha256": _sha(RUN / "coords" / fit_id / "1Mb.npz") if (RUN / "coords" / fit_id / "1Mb.npz").is_file() else None,
                       "initial_raw_y_sha256": record.get("initial_state_hashes", {}).get("raw_y_sha256"),
                       "initial_q": record.get("initial_q"), "start_endpoint_sha256": item["endpoint_sha256"],
                       "fresh_lbfgs_history": True, "reference_opened": False, "phase_opened": False}
            summaries.append(summary)
            _write(RUN / "stages" / fit_id / "fit_summary.json", summary)
            controller._event({"event": "extension_fit_end", **summary})
    candidates = []
    for summary in summaries:
        endpoint = RUN / "coords" / summary["fit_id"] / "1Mb.npz"
        with np.load(endpoint, allow_pickle=False) as payload:
            coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
            p_value = float(np.asarray(payload["p"]).item())
        candidates.append({"candidate_id": summary["fit_id"], "fit_id": summary["fit_id"], "kind": "real_extension",
                           "model_id": summary["model_id"], "candidate": summary["candidate"], "variant": summary["variant"],
                           "coordinate_path": str(endpoint.relative_to(RUN)), "coordinate_sha256": _sha(endpoint),
                           "coordinate_array_sha256": controller._hash_array(coordinates), "coordinate_shape": list(coordinates.shape),
                           "p": p_value, "stage_status": summary["status"], "start_endpoint_sha256": summary["start_endpoint_sha256"]})
    _write(RUN / "results/candidate_manifest.json", {"schema": "p9016-real-extension-candidate-manifest-v1", "candidate_count": len(candidates),
                                                        "candidates": candidates, "reference_opened": False, "synthetic_candidates": 0})
    actual_fg = sum(item["outer_fg_actual"] for item in summaries)
    status = "extension_complete" if all(item["status"] in ("converged", "not_converged", "budget_not_converged") for item in summaries) else "extension_complete_with_failures"
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    config.update({"status": status, "formal_started": True, "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "actual_outer_fg_extension": actual_fg, "extension_summaries": summaries, "reference_opened": False, "phase_opened": False})
    _write(RUN / "config.json", config)
    controller._event({"event": "extension_end", "status": status, "actual_outer_fg_extension": actual_fg,
                       "formal_outer_fg_total_cap": 7952, "synthetic_fits": 0, "reference_opened": False,
                       "at_utc": config["completed_at_utc"]})
    _write(RUN / "extension_runtime.json", {"status": status, "return_code": 0 if status == "extension_complete" else 2,
                                               "completed_at_utc": config["completed_at_utc"], "pid": launch["pid"],
                                               "started_at_utc": launch["started_at_utc"], "lock": str(RUN / "extension_writer.lock")})
    return 0 if status == "extension_complete" else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.parse_args()
    raise SystemExit(run())


if __name__ == "__main__":
    main()
