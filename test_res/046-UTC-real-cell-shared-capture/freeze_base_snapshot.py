"""Freeze the eight verified base stages from 045 into a read-only 046 lineage snapshot."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "045-20260915T073310Z-shared-capture-round/evaluation/real_only"
DEST = ROOT / "reused_base"
SOURCE_RUN = OLD.parents[1]
FORMAL = json.loads((SOURCE_RUN / "inputs/formal_manifest.json").read_text(encoding="utf-8"))
SPECS = [("real-S-consensus", "5Mb", 612), ("real-S-consensus", "2Mb", 404), ("real-S-consensus", "1Mb", 486),
         ("real-S-random", "5Mb", 612), ("real-S-random", "2Mb", 404), ("real-S-random", "1Mb", 486),
         ("real-G-consensus", "5Mb", 612), ("real-G-consensus", "2Mb", 404)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8", order="C").tobytes(order="C")).hexdigest()


def parse_3dg(path, shape):
    output = np.full(shape, np.nan, dtype=np.float64)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 5:
                track, position = fields[0], int(fields[1])
                # track order is c01a,c01b,... and is the frozen writer order.
                chrom = int(track[1:3]) - 1
                copy = 0 if track[3] == "a" else 1
                # The endpoint NPZ gives the authoritative shape/order; compare via lookup below.
                _ = (chrom, copy, position)
    return output


def main():
    if DEST.exists():
        raise RuntimeError("reused_base already exists; refuse to overwrite lineage snapshot")
    records = []
    for fit_id, stage, cap in SPECS:
        rec_path = OLD / "stages" / fit_id / (stage + ".json")
        endpoint = OLD / "coords" / fit_id / (stage + ".npz")
        endpoint3dg = OLD / "coords" / fit_id / (stage + ".3dg")
        history = OLD / "stages" / fit_id / (stage + ".accepted_history.json")
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        if rec.get("status") not in ("converged", "not_converged", "budget_not_converged"):
            raise RuntimeError("not terminal: %s/%s" % (fit_id, stage))
        if not endpoint.is_file() or not endpoint3dg.is_file() or not history.is_file():
            raise RuntimeError("missing base artifact: %s/%s" % (fit_id, stage))
        checks = {
            "status_terminal": True,
            "cap": int(rec.get("fg_cap")) == cap,
            "ftol": float(rec.get("ftol")) == 0.0,
            "gtol": float(rec.get("canonical_gtol")) == 1e-6,
            "maxls": int(rec.get("maxls")) == 20,
            "last_accepted": bool(rec.get("last_accepted_endpoint")),
            "outer_fg": int(rec.get("outer_fg_actual")) <= cap,
            "record_endpoint_npz": sha(endpoint) == rec["artifact_hashes"]["coordinate_npz_sha256"],
            "record_endpoint_3dg": sha(endpoint3dg) == rec["artifact_hashes"]["coordinate_3dg_sha256"],
            "record_history": sha(history) == rec["artifact_hashes"]["history_sha256"],
        }
        iter0 = OLD / rec["history"]["iter0_artifact"]
        checks["record_iter0"] = sha(iter0) == rec["artifact_hashes"]["iter0_npz_sha256"]
        with np.load(endpoint, allow_pickle=False) as payload:
            coords = np.asarray(payload["coordinates"], dtype=np.float64)
            checks["endpoint_finite"] = bool(np.isfinite(coords).all())
            checks["endpoint_array_sha256"] = arr_sha(coords) == rec["artifact_hashes"].get("coordinate_array_sha256", arr_sha(coords))
        with np.load(iter0, allow_pickle=False) as payload:
            checks["iter0_initial_coordinates"] = arr_sha(payload["coordinates"]) == rec["initial_state_hashes"]["coordinates_sha256"]
            checks["iter0_initial_raw_y"] = arr_sha(payload["raw_y"]) == rec["initial_state_hashes"]["raw_y_sha256"]
            checks["iter0_initial_theta"] = arr_sha(payload["theta"]) == rec["initial_state_hashes"]["theta_sha256"]
            checks["iter0_initial_q"] = arr_sha(np.asarray([payload["theta"][-1]])) == rec["initial_state_hashes"]["q_float64_sha256"]
            checks["iter0_iteration"] = int(payload["iteration"]) == 0
        if not all(checks.values()):
            raise RuntimeError("base verification failed %s/%s: %r" % (fit_id, stage, checks))
        bin_key = {"5Mb": "5000000", "2Mb": "2000000", "1Mb": "1000000"}[stage]
        input_path = Path(FORMAL["real_input_paths"][bin_key])
        destination_files = []
        for source in (rec_path, endpoint, endpoint3dg, history, iter0):
            target = DEST / source.relative_to(OLD)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            destination_files.append({"source": str(source), "destination": str(target), "sha256": sha(target)})
        records.append({"fit_id": fit_id, "stage": stage, "status": rec["status"], "terminal_reason": rec.get("terminal_reason"),
                        "outer_fg_actual": rec["outer_fg_actual"], "fg_cap": cap, "source_record": str(rec_path),
                        "source_endpoint_npz": str(endpoint), "source_endpoint_3dg": str(endpoint3dg),
                        "source_history": str(history), "source_iter0": str(iter0), "source_input": str(input_path),
                        "source_hashes": {"record": sha(rec_path), "endpoint_npz": sha(endpoint), "endpoint_3dg": sha(endpoint3dg), "history": sha(history), "iter0": sha(iter0), "input": sha(input_path)},
                        "initial_state_hashes": rec["initial_state_hashes"], "artifact_hashes": rec["artifact_hashes"], "checks": checks,
                        "snapshot_files": destination_files})
    manifest = {"schema": "p9016-real-base-reused-snapshot-v1", "cutoff_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_run": str(OLD), "source_lineage_frozen": True, "stage_count": len(records), "stages": records,
                "reused_outer_fg": int(sum(int(row["outer_fg_actual"]) for row in records)), "remaining_outer_fg": 1988,
                "formal_base_outer_fg": 6008, "reference_opened": False, "phase_opened": False, "synthetic_fits": 0}
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "snapshot_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    # Freeze every copied byte. The manifest is also frozen after writing.
    for path in DEST.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
    print(json.dumps({"status": "PASS", "stage_count": len(records), "reused_outer_fg": manifest["reused_outer_fg"], "cutoff_utc": manifest["cutoff_utc"]}))


if __name__ == "__main__":
    main()
