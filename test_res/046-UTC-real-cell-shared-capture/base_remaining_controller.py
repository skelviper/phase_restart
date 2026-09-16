"""Isolated remaining real base stages for run 046.

Consumes only the frozen 046/reused_base snapshot plus the original real inputs;
never writes to or reads dynamic 045/evaluation/real_only stage results.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

RUN = Path(__file__).resolve().parent
SOURCE_RUN = RUN.parent / "045-20260915T073310Z-shared-capture-round"
SOURCE = SOURCE_RUN / "source"
ROOT = RUN.parent.parent
for path in (SOURCE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import formal_controller as controller  # noqa: E402

OUT = RUN / "base_remaining"
FULL_WEIGHTS = {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}
_LOCK_HANDLE = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _acquire_lock() -> dict[str, Any]:
    global _LOCK_HANDLE
    lock_path = OUT / "writer.lock"
    OUT.mkdir(parents=True, exist_ok=True)
    info = {"pid": os.getpid(), "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "lock_method": "fcntl.flock(LOCK_EX|LOCK_NB)", "reference_opened": False, "phase_opened": False, "synthetic_fits": 0}
    try:
        info["pid_start_ticks"] = (Path("/proc") / str(os.getpid()) / "stat").read_text(encoding="utf-8").split()[21]
    except (FileNotFoundError, IndexError, OSError):
        info["pid_start_ticks"] = "unavailable"
    if os.environ.get("P9016_BASE_REMAINING_LOCK_HELD") == "1":
        _LOCK_HANDLE = None
    else:
        handle = lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError("base_remaining writer lock held") from exc
        _LOCK_HANDLE = handle
    _write(OUT / "runtime.json", {"status": "running", **info})
    return info


def _fit_record(old_manifest: dict[str, Any], fit_id: str) -> dict[str, Any]:
    return next(row for row in old_manifest["matrix"] if row["fit_id"] == fit_id)


def _endpoint(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    with np.load(path, allow_pickle=False) as payload:
        coords = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        raw_y = np.asarray(payload["raw_y"], dtype=np.float64).copy()
        p = float(np.asarray(payload["p"]).item())
        q = float(np.asarray(payload["q"]).item())
    if not np.isfinite(coords).all() or not np.isfinite(raw_y).all():
        raise RuntimeError("nonfinite frozen endpoint")
    return coords, raw_y, p, q


def _stage_specs():
    return ({"stage": "5Mb", "bin_size_bp": 5_000_000, "fg_cap": 612},
            {"stage": "2Mb", "bin_size_bp": 2_000_000, "fg_cap": 404},
            {"stage": "1Mb", "bin_size_bp": 1_000_000, "fg_cap": 486})


def _run_stage(fit: dict[str, Any], stage: dict[str, Any], data: Any, coordinates: np.ndarray, p: float, q: float,
               raw_y: np.ndarray | None) -> tuple[dict[str, Any], np.ndarray, float, float]:
    row = controller.stage_fit(fit, stage, data, coordinates, p, q, None, initial_raw_y=raw_y)
    endpoint = OUT / "coords" / fit["fit_id"] / (stage["stage"] + ".npz")
    with np.load(endpoint, allow_pickle=False) as payload:
        next_coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        next_p = float(np.asarray(payload["p"]).item())
        next_q = float(np.asarray(payload["q"]).item())
    return row, next_coordinates, next_p, next_q


def _record_stage(fit_id: str, stage: str) -> dict[str, Any]:
    rec_path = OUT / "stages" / fit_id / (stage + ".json")
    endpoint = OUT / "coords" / fit_id / (stage + ".npz")
    endpoint3dg = OUT / "coords" / fit_id / (stage + ".3dg")
    record = json.loads(rec_path.read_text(encoding="utf-8"))
    return {"fit_id": fit_id, "stage": stage, "status": record["status"], "terminal_reason": record.get("terminal_reason"),
            "outer_fg_actual": record["outer_fg_actual"], "fg_cap": record["fg_cap"], "record_path": str(rec_path.relative_to(RUN)),
            "endpoint_npz": str(endpoint.relative_to(RUN)), "endpoint_npz_sha256": _sha(endpoint),
            "endpoint_3dg": str(endpoint3dg.relative_to(RUN)), "endpoint_3dg_sha256": _sha(endpoint3dg),
            "history_sha256": record["artifact_hashes"]["history_sha256"], "iter0_sha256": record["artifact_hashes"]["iter0_npz_sha256"],
            "initial_state_hashes": record["initial_state_hashes"], "record_sha256": _sha(rec_path), "weights": record["weights"]}


def _write_combined(old_manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    reused = json.loads((RUN / "reused_base/snapshot_manifest.json").read_text(encoding="utf-8"))
    all_rows = []
    for item in reused["stages"]:
        all_rows.append({"lineage": "reused_base", "fit_id": item["fit_id"], "stage": item["stage"],
                         "status": item["status"], "outer_fg_actual": item["outer_fg_actual"], "fg_cap": item["fg_cap"],
                         "record_path": str(Path("reused_base") / "stages" / item["fit_id"] / (item["stage"] + ".json")),
                         "endpoint_npz": str(Path("reused_base") / "coords" / item["fit_id"] / (item["stage"] + ".npz")),
                         "endpoint_npz_sha256": item["source_hashes"]["endpoint_npz"], "endpoint_3dg": str(Path("reused_base") / "coords" / item["fit_id"] / (item["stage"] + ".3dg")),
                         "endpoint_3dg_sha256": item["source_hashes"]["endpoint_3dg"], "history_sha256": item["source_hashes"]["history"],
                         "iter0_sha256": item["source_hashes"]["iter0"], "record_sha256": item["source_hashes"]["record"],
                         "initial_state_hashes": item["initial_state_hashes"], "weights": {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}})
    all_rows.extend({"lineage": "base_remaining", **row} for row in rows)
    if len(all_rows) != 12:
        raise RuntimeError("combined base stage count mismatch")
    endpoints = {}
    for model in ("S", "G"):
        for candidate in ("consensus", "random"):
            fit_id = "real-%s-%s" % (model, candidate)
            one = next(item for item in all_rows if item["fit_id"] == fit_id and item["stage"] == "1Mb")
            record_path = RUN / one["record_path"]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            endpoints[fit_id] = {"fit_id": fit_id, "model_id": model, "candidate": candidate, "endpoint_npz": one["endpoint_npz"],
                                 "endpoint_npz_sha256": one["endpoint_npz_sha256"], "count_nll_normalized": record["endpoint"]["components"]["count_nll_normalized"],
                                 "status": record["status"]}
    selected = {}
    for model in ("S", "G"):
        c = endpoints["real-%s-consensus" % model]
        r = endpoints["real-%s-random" % model]
        if abs(float(c["count_nll_normalized"]) - float(r["count_nll_normalized"])) <= 1e-9:
            chosen = "consensus"
        else:
            chosen = "consensus" if float(c["count_nll_normalized"]) < float(r["count_nll_normalized"]) else "random"
        selected[model] = {"selected_by_own_count_nll": chosen, "candidates": [c, r], "tie_tolerance": 1e-9,
                           "selection_rule": "minimum same-model base 1Mb count NLL; consensus on <=1e-9 tie", "reference_used": False}
    combined = {"schema": "p9016-real-combined-base-manifest-v1", "scope": "P9016 real cell only", "source_lineage_frozen": True,
                "stage_count": 12, "endpoint_count": 4, "outer_fg_cap": 6008, "reused_outer_fg": 4020, "remaining_outer_fg": 1988,
                "stages": all_rows, "endpoints": endpoints, "selected_sources": selected, "reference_opened": False,
                "phase_opened": False, "synthetic_fits": 0, "old_source_dynamic_results_excluded": True}
    _write(RUN / "combined_base_manifest.json", combined)
    _write(RUN / "base_source_selection.json", {"schema": "p9016-real-combined-base-selection-v1", "models": selected,
                                                   "reference_used": False, "source_manifest": "combined_base_manifest.json"})
    return combined


def run() -> int:
    launch = _acquire_lock()
    old_manifest = json.loads((SOURCE_RUN / "inputs/formal_manifest.json").read_text(encoding="utf-8"))
    snapshot = json.loads((RUN / "reused_base/snapshot_manifest.json").read_text(encoding="utf-8"))
    if snapshot["stage_count"] != 8 or snapshot["reused_outer_fg"] != 4020:
        raise RuntimeError("frozen reused_base snapshot mismatch")
    for name in ("coords", "stages", "checkpoints", "logs", "results"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    controller.RUN = SOURCE_RUN
    data = {int(stage["bin_size_bp"]): controller._load_real_data(old_manifest, int(stage["bin_size_bp"])) for stage in _stage_specs()}
    controller.RUN = OUT
    controller._event({"event": "base_remaining_start", "fits": 2, "stages": 4, "outer_fg": 1988, "reused_outer_fg": 4020,
                       "reference_opened": False, "phase_opened": False, "synthetic_fits": 0, "at_utc": launch["started_at_utc"]})
    rows = []
    # G-consensus 1Mb begins from the frozen G-consensus 2Mb endpoint with seed 1103 prolongation.
    frozen_g2 = RUN / "reused_base/coords/real-G-consensus/2Mb.npz"
    coords, raw_y, p, q = _endpoint(frozen_g2)
    warm = controller.reconstruction_init.warm_start_from_layer(
        coords, (data[2_000_000].locus_bin * int(data[2_000_000].bin_size)), data[2_000_000].locus_chromosome,
        tuple(data[1_000_000].chromosome_names), tuple(int(v) for v in data[1_000_000].chromosome_lengths), 1_000_000, 1103)
    fit = dict(_fit_record(old_manifest, "real-G-consensus")); fit["stages"] = [_stage_specs()[2]]; fit["weights"] = FULL_WEIGHTS
    stage = _stage_specs()[2]
    controller._event({"event": "base_remaining_stage_start", "fit_id": fit["fit_id"], "stage": "1Mb", "source": str(frozen_g2), "seed": 1103})
    rec, coords, p, q = _run_stage(fit, stage, data[1_000_000], np.asarray(warm["coords"], dtype=np.float64), p, q, None)
    rows.append(_record_stage("real-G-consensus", "1Mb"))
    # G-random is a fresh full 5->2->1 run from the frozen original serialized 5Mb start.
    random_fit = dict(_fit_record(old_manifest, "real-G-random")); random_fit["stages"] = list(_stage_specs()); random_fit["weights"] = FULL_WEIGHTS
    start_path = Path(random_fit["start_5Mb"])
    if not start_path.is_absolute():
        start_path = SOURCE_RUN / start_path
    coords, raw_y, p, _meta = controller.load_start(start_path)
    q = float(controller.contact_model.q_from_p(p))
    previous_data = None
    for index, stage in enumerate(_stage_specs()):
        if index > 0:
            previous_data = data[int(_stage_specs()[index - 1]["bin_size_bp"])]
            warm = controller.reconstruction_init.warm_start_from_layer(
                coords, (previous_data.locus_bin * int(previous_data.bin_size)), previous_data.locus_chromosome,
                tuple(data[int(stage["bin_size_bp"])].chromosome_names), tuple(int(v) for v in data[int(stage["bin_size_bp"])].chromosome_lengths), int(stage["bin_size_bp"]), 2207)
            coords = np.asarray(warm["coords"], dtype=np.float64)
            raw_y = None
        controller._event({"event": "base_remaining_stage_start", "fit_id": random_fit["fit_id"], "stage": stage["stage"], "seed": 2207})
        rec, coords, p, q = _run_stage(random_fit, stage, data[int(stage["bin_size_bp"])], coords, p, q, raw_y)
        rows.append(_record_stage("real-G-random", stage["stage"]))
    combined = _write_combined(old_manifest, rows)
    status = "base_remaining_complete" if all(row["status"] in ("converged", "not_converged", "budget_not_converged") for row in rows) else "base_remaining_with_failures"
    config = {"schema": "p9016-real-base-remaining-v1", "status": status, "formal_started": True, "reference_opened": False,
              "phase_opened": False, "synthetic_fits": 0, "new_stage_count": 4, "new_outer_fg_actual": sum(int(row["outer_fg_actual"]) for row in rows),
              "reused_outer_fg": 4020, "combined_base_outer_fg": 6008, "combined_base_manifest": "combined_base_manifest.json",
              "source_selection": "base_source_selection.json", "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    _write(OUT / "config.json", config)
    _write(OUT / "runtime.json", {"status": status, "pid": launch["pid"], "started_at_utc": launch["started_at_utc"],
                                    "completed_at_utc": config["completed_at_utc"], "return_code": 0 if status == "base_remaining_complete" else 2,
                                    "lock_method": "fcntl.flock(LOCK_EX|LOCK_NB)"})
    controller._event({"event": "base_remaining_end", "status": status, "new_outer_fg_actual": config["new_outer_fg_actual"],
                       "combined_base_outer_fg": 6008, "reference_opened": False, "phase_opened": False,
                       "at_utc": config["completed_at_utc"]})
    return 0 if status == "base_remaining_complete" else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.parse_args()
    raise SystemExit(run())


if __name__ == "__main__":
    main()
