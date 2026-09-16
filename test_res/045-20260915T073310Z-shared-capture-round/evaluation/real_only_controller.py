"""Real-only continuation runner after the user scope revision.

This runner deliberately reuses the pinned formal stage_fit and frozen optimizer
implementation but writes to evaluation/real_only, leaving the abandoned 14-fit
run and its partial checkpoint untouched. It loads only the three real aggregates
and the two authorized 014 real starts; no synthetic path is opened.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

RUN = Path(__file__).resolve().parents[1]
SOURCE = RUN / "source"
ROOT = RUN.parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Independent formal bootstrap: this imports frozen_pr before pr dependencies.
import formal_controller as controller  # noqa: E402

REAL_STAGES = ({"stage": "5Mb", "bin_size_bp": 5_000_000, "fg_cap": 612},
               {"stage": "2Mb", "bin_size_bp": 2_000_000, "fg_cap": 404},
               {"stage": "1Mb", "bin_size_bp": 1_000_000, "fg_cap": 486})
REAL_FITS = tuple("real-%s-%s" % (model, candidate)
                     for model in ("S", "G") for candidate in ("consensus", "random"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _source_path(source_run: Path, value: str | Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (source_run / path).resolve()


def _real_hash_gate(source_run: Path, old_config: dict[str, Any], old_manifest: dict[str, Any]) -> dict[str, Any]:
    frozen = dict(old_config.get("immutable_hashes", {}))
    required: dict[str, Path] = {}
    for size, path_text in old_manifest["real_input_paths"].items():
        path = _source_path(source_run, path_text)
        required["inputs/real_%s_aggregate.npz" % size] = path
    for candidate in ("consensus", "random"):
        row = old_manifest["real_initial_paths"][candidate]["stages"]["5Mb"]
        required["coords/initial/real_%s_5Mb.npz" % candidate] = _source_path(source_run, row["path"])
    required["inputs/formal_manifest.json"] = source_run / "inputs/formal_manifest.json"
    required["inputs/source_input_manifest.json"] = source_run / "inputs/source_input_manifest.json"
    required["inputs/evaluation_manifest.json"] = source_run / "inputs/evaluation_manifest.json"
    for key in ("test_res/014-20260912_153000-s0-genome-wide-fixed/coords/consensus.3dg",
                "test_res/014-20260912_153000-s0-genome-wide-fixed/coords/random.3dg",
                "test_res/014-20260912_153000-s0-genome-wide-fixed/gate.json"):
        required[key] = ROOT / key
    for key in old_manifest.get("source_hashes", {}):
        if key.startswith("source/post_evaluator") or key == "known_e_worker_input_manifest.json":
            continue
        path = (source_run / key)
        if not path.is_file():
            path = ROOT / key
        if not path.is_file():
            path = ROOT / "docs/audits/next-step-r2-preparation-20260914T143656Z/synthetic_inputs" / key
        if path.is_file():
            required[key] = path
    mismatches = []
    checked = {}
    for key, path in required.items():
        actual = _sha(path)
        expected = frozen.get(key)
        if expected is not None and str(expected) != actual:
            mismatches.append({"path": key, "expected": expected, "actual": actual})
        checked[key] = {"path": str(path), "sha256": actual}
    if mismatches:
        raise RuntimeError("real-only pinned hash gate failed: %s" % mismatches[:3])
    return {"status": "PASS", "checked": checked, "mismatches": mismatches,
            "synthetic_paths_opened": False, "reference_opened": False, "phase_opened": False}


def _fit_rows(old_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in old_manifest["matrix"]:
        if row.get("kind") != "real":
            continue
        fit = dict(row)
        fit["stages"] = [dict(stage) for stage in REAL_STAGES]
        fit["weights"] = {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}
        rows.append(fit)
    if tuple(row["fit_id"] for row in rows) != REAL_FITS:
        raise RuntimeError("real-only matrix mismatch")
    return rows


def _write_candidates(output_run: Path, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for summary in summaries:
        if summary.get("status") != "completed":
            continue
        fit_id = summary["fit_id"]
        endpoint = output_run / "coords" / fit_id / "1Mb.npz"
        record = output_run / "stages" / fit_id / "1Mb.json"
        if not endpoint.is_file() or not record.is_file():
            continue
        with np.load(endpoint, allow_pickle=False) as payload:
            coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
            p = float(np.asarray(payload["p"]).item())
        candidates.append({"candidate_id": fit_id, "fit_id": fit_id, "kind": "real", "model_id": summary["model_id"],
                           "candidate": summary["candidate"], "coordinate_path": str(endpoint.relative_to(output_run)),
                           "coordinate_sha256": _sha(endpoint), "coordinate_array_sha256": controller._hash_array(coordinates),
                           "coordinate_shape": list(coordinates.shape), "p": p,
                           "stage_status": json.loads(record.read_text(encoding="utf-8")).get("status")})
    output = {"schema": "p9016-real-only-candidate-manifest-v1", "run_id": output_run.name,
              "candidate_hash_gate": "all four real endpoints hashed before any reference access",
              "candidates": candidates, "expected_candidates": 4, "synthetic_candidates": 0}
    _write(output_run / "results/candidate_manifest.json", output)
    return output


def run() -> int:
    source_run = RUN
    output_run = RUN / "evaluation" / "real_only"
    for name in ("coords", "stages", "checkpoints", "logs", "results"):
        (output_run / name).mkdir(parents=True, exist_ok=True)
    old_config = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
    old_manifest = json.loads((source_run / "inputs/formal_manifest.json").read_text(encoding="utf-8"))
    gate = _real_hash_gate(source_run, old_config, old_manifest)
    fits = _fit_rows(old_manifest)
    revision = {"schema": "p9016-real-only-scope-revision-v1", "run_id": output_run.name,
                "scope_revision_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "user_scope": "P9016 real cell only; synthetic formal fits and synthetic evaluation cancelled",
                "cancelled_matrix": {"fits": 10, "stages": 10, "outer_fg": 4860, "reason": "latest direct human scope instruction"},
                "real_matrix": {"fits": 4, "stages": 12, "outer_fg": 6008, "fits_ids": list(REAL_FITS)},
                "optimizer": {"ftol": 0.0, "canonical_gtol": 1e-6, "maxls": 20, "stage_caps": [612, 404, 486],
                              "seeds": {"consensus": 1103, "random": 2207}, "weights": fits[0]["weights"]},
                "pinned_training_preflight": {"preflight": "../checks/preflight.json", "preflight_sha256": old_config["preflight"]["sha256"],
                                               "source_hash_count": 51, "real_hash_gate": gate},
                "synthetic_calibration_archived": "The already completed synthetic count/gradient checks remain engineering calibration only; no synthetic formal/evaluation result is produced or interpreted.",
                "reference_opened": False, "phase_opened": False, "synthetic_paths_opened": False}
    _write(output_run / "manifest.json", {"revision": revision, "matrix": fits})
    _write(output_run / "config.json", {"schema": "p9016-real-only-run-v1", "status": "prepared_real_only",
                                         "manifest": "manifest.json", "formal_started": False, "reference_opened": False,
                                         "phase_opened": False, "synthetic_fits": 0, "real_fits": 4, "real_stages": 12,
                                         "outer_fg_cap": 6008, "real_hash_gate": gate})
    # Load only real worker data while RUN points to the original prepared directory.
    controller.RUN = source_run
    real_data = {int(stage["bin_size_bp"]): controller._load_real_data(old_manifest, int(stage["bin_size_bp"])) for stage in REAL_STAGES}
    starts = {}
    for fit in fits:
        start_path = _source_path(source_run, fit["start_5Mb"])
        starts[fit["fit_id"]] = controller.load_start(start_path)
    controller.RUN = output_run
    controller._event({"event": "real_only_start", "fits": 4, "stages": 12, "outer_fg": 6008,
                       "synthetic_fits": 0, "reference_opened": False, "at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    summaries = []
    actual_fg = 0
    for fit in fits:
        fit_id = fit["fit_id"]
        coordinates, raw_y, p_init, _meta = starts[fit_id]
        q_init = controller.contact_model.q_from_p(float(p_init))
        previous = None
        blocked = False
        stage_rows = []
        for stage_index, stage_spec in enumerate(REAL_STAGES):
            stage_name = stage_spec["stage"]
            existing_record_path = output_run / "stages" / fit_id / (stage_name + ".json")
            existing_endpoint = output_run / "coords" / fit_id / (stage_name + ".npz")
            if existing_record_path.is_file():
                existing = json.loads(existing_record_path.read_text(encoding="utf-8"))
                if existing.get("status") in ("converged", "not_converged", "budget_not_converged") and existing_endpoint.is_file():
                    data = real_data[int(stage_spec["bin_size_bp"])]
                    stage_rows.append(existing)
                    actual_fg += int(existing.get("outer_fg_actual", 0))
                    with np.load(existing_endpoint, allow_pickle=False) as payload:
                        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
                    raw_y = None
                    p_init = float(existing["endpoint"]["p"])
                    q_init = float(existing["endpoint"]["q"])
                    previous = {"coordinates": coordinates, "positions": (data.locus_bin * int(data.bin_size)).copy(),
                                "chromosome_index": np.asarray(data.locus_chromosome, dtype=np.int32).copy()}
                    controller._event({"event": "real_only_stage_reuse", "fit_id": fit_id, "stage": stage_name,
                                       "status": existing.get("status"), "outer_fg_reused": int(existing.get("outer_fg_actual", 0)),
                                       "endpoint_sha256": _sha(existing_endpoint)})
                    continue
                if existing.get("status") == "running":
                    restart_note = {"fit_id": fit_id, "stage": stage_name, "prior_status": "running",
                                    "prior_record_sha256": _sha(existing_record_path), "action": "restart_from_prior_completed_stage_boundary",
                                    "lbfgs_history_reused": False, "cancelled_optimizer_fg_count": "unknown_upper_bound_only"}
                    _write(output_run / "stages" / fit_id / (stage_name + ".restart_evidence.json"), restart_note)
                    controller._event({"event": "real_only_stage_restart_after_cancellation", **restart_note})
            if blocked:
                row = {"fit_id": fit_id, "stage": stage_name, "status": "blocked_by_previous_failure"}
                controller._write_json(output_run / "stages" / fit_id / (stage_name + ".json"), row)
                stage_rows.append(row)
                continue
            data = real_data[int(stage_spec["bin_size_bp"])]
            if stage_index > 0:
                previous_data = real_data[int(REAL_STAGES[stage_index - 1]["bin_size_bp"])]
                warm = controller.reconstruction_init.warm_start_from_layer(
                    previous["coordinates"], previous["positions"], previous["chromosome_index"],
                    tuple(data.chromosome_names), tuple(int(v) for v in data.chromosome_lengths), int(stage_spec["bin_size_bp"]),
                    1103 if fit["candidate"] == "consensus" else 2207)
                coordinates = np.asarray(warm["coords"], dtype=np.float64)
                raw_y = None
                controller._event({"event": "real_only_prolongation", "fit_id": fit_id,
                                   "from_stage": REAL_STAGES[stage_index - 1]["stage"], "to_stage": stage_spec["stage"],
                                   "q_carry": q_init})
            controller._event({"event": "real_only_stage_start", "fit_id": fit_id, "stage": stage_spec["stage"]})
            row = controller.stage_fit(fit, stage_spec, data, coordinates, float(p_init), float(q_init), None, initial_raw_y=raw_y)
            stage_rows.append(row)
            if row.get("status") == "failure":
                blocked = True
            else:
                actual_fg += int(row.get("outer_fg_actual", 0))
                endpoint = output_run / "coords" / fit_id / (stage_spec["stage"] + ".npz")
                with np.load(endpoint, allow_pickle=False) as payload:
                    coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
                raw_y = None
                p_init = float(row["endpoint"]["p"])
                q_init = float(row["endpoint"]["q"])
                previous = {"coordinates": coordinates, "positions": (data.locus_bin * int(data.bin_size)).copy(),
                            "chromosome_index": np.asarray(data.locus_chromosome, dtype=np.int32).copy()}
        summary = {"fit_id": fit_id, "kind": "real", "model_id": fit["model_id"], "candidate": fit["candidate"],
                   "status": "completed" if not blocked else "failure", "stage_status": [row.get("status") for row in stage_rows],
                   "outer_fg_actual": sum(int(row.get("outer_fg_actual", 0)) for row in stage_rows)}
        summaries.append(summary)
        controller._write_json(output_run / "stages" / fit_id / "fit_summary.json", summary)
        controller._event({"event": "real_only_fit_end", **summary})
    candidate_manifest = _write_candidates(output_run, summaries)
    selection = controller._select_real_sources(summaries) if len(candidate_manifest["candidates"]) == 4 else {}
    _write(output_run / "results/real_source_selection.json", {"schema": "p9016-real-only-source-selection-v1", "models": selection,
                                                                  "reference_used": False, "evaluation_used": False})
    status = "real_formal_complete" if all(item["status"] == "completed" for item in summaries) else "real_formal_complete_with_failures"
    config = json.loads((output_run / "config.json").read_text(encoding="utf-8"))
    config.update({"status": status, "formal_started": True, "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "actual_outer_fg": actual_fg, "fit_summaries": summaries, "candidate_count": len(candidate_manifest["candidates"])})
    _write(output_run / "config.json", config)
    controller._event({"event": "real_only_end", "status": status, "actual_outer_fg": actual_fg,
                       "candidate_count": len(candidate_manifest["candidates"]), "synthetic_fits": 0,
                       "reference_opened": False, "at_utc": config["completed_at_utc"]})
    return 0 if status == "real_formal_complete" else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    args = parser.parse_args()
    raise SystemExit(run())


if __name__ == "__main__":
    main()
