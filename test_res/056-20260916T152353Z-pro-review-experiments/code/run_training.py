"""056实验3：四条train-only G/full-J/raw多分辨率fit，单卡严格串行。"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
for path in (ROOT, S045):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import data_io  # noqa: E402
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective  # noqa: E402
from budget_runner import run_budgeted_lbfgs_no_extra  # noqa: E402
from pr import contact_model, reconstruction_init  # noqa: E402
from pr.solver_state import (export_present_3dg, load_solver_state,
                             write_presence_mask, write_solver_state)  # noqa: E402

CONDITIONS = ("G-original", "G-offdiag-e")
SEEDS = (560101, 560102)
STAGES = (("5Mb", 5_000_000, 612), ("2Mb", 2_000_000, 404), ("1Mb", 1_000_000, 486))
WEIGHTS = {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def load_data(condition: str, bin_size: int):
    path = RUN / "inputs" / ("%s_%d_aggregate.npz" % (condition, bin_size))
    data = data_io.load_aggregate(path)
    data.assert_consistent()
    return data


def initial_start(seed: int):
    path = RUN / "states/experiment3/initial" / ("seed-%d" % seed) / "5Mb/solver_state.npz"
    state = load_solver_state(path)
    return state["coordinates"], state["raw_y"], state["p"], state["q"], {
        "mode": "contact_independent_polymer", "path": str(path.relative_to(ROOT)),
        "sha256": state["sha256"], "seed": seed,
    }


def warm_start(previous_state: Path, previous_data, target_data, target_bin: int, seed: int):
    state = load_solver_state(previous_state)
    coordinates = state["coordinates"]
    positions = np.broadcast_to(
        (np.asarray(previous_data.locus_bin) * int(previous_data.bin_size)).astype(np.int64),
        (2, previous_data.n_loci)).copy()
    chromosomes = np.broadcast_to(np.asarray(previous_data.locus_chromosome, dtype=np.int32),
                                  (2, previous_data.n_loci)).copy()
    warm = reconstruction_init.warm_start_from_layer(
        coordinates, positions, chromosomes, tuple(target_data.chromosome_names),
        tuple(int(value) for value in target_data.chromosome_lengths), int(target_bin), int(seed))
    target_coordinates = np.asarray(warm["coords"], dtype=np.float64)
    target_raw_y = contact_model.sphere_inverse(target_coordinates)
    return target_coordinates, target_raw_y, state["p"], state["q"], warm["metadata"]


def freeze_stage(fit_id: str, stage: str, data, legacy_endpoint: Path):
    with np.load(legacy_endpoint, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        raw_y = np.asarray(payload["raw_y"], dtype=np.float64).copy()
        theta = np.asarray(payload["theta"], dtype=np.float64).copy()
        p = float(np.asarray(payload["p"]).item())
        q = float(np.asarray(payload["q"]).item())
    directory = RUN / "states/experiment3" / fit_id / stage
    state = write_solver_state(directory / "solver_state.npz", coordinates=coordinates,
                               raw_y=raw_y, theta=theta, p=p, q=q)
    presence = np.ones((2, data.n_loci), dtype=bool)
    mask = write_presence_mask(directory / "presence_mask.npz", presence, data.n_loci)
    before = state["sha256"]
    export = export_present_3dg(RUN / "exports/experiment3" / fit_id / (stage + ".3dg"),
                                data, coordinates, presence)
    after = load_solver_state(directory / "solver_state.npz")["sha256"]
    if before != after:
        raise RuntimeError("export changed immutable solver state")
    return {"solver_state": state, "presence_mask": mask, "export": export,
            "solver_sha_unchanged_after_export": True}


def checkpoint_hook(fit_id: str, stage: str):
    directory = RUN / "checkpoints" / fit_id / stage
    directory.mkdir(parents=True, exist_ok=True)

    def save(entry):
        path = directory / ("accepted-%05d.npz" % int(entry["iteration"]))
        if path.exists():
            raise RuntimeError("refusing to overwrite checkpoint")
        with path.open("xb") as handle:
            np.savez_compressed(
                handle, iteration=np.asarray(entry["iteration"], dtype=np.int64),
                nfev=np.asarray(entry["nfev"], dtype=np.int64),
                theta=np.asarray(entry["theta"], dtype=np.float64),
                raw_y=np.asarray(entry["raw_y"], dtype=np.float64),
                coordinates=np.asarray(entry["coordinates"], dtype=np.float64),
                optimizer_gradient=np.asarray(entry["optimizer_gradient"], dtype=np.float64),
                canonical_gradient=np.asarray(entry["canonical_gradient"], dtype=np.float64),
            )
    return save


def classify(result):
    if result.terminal_reason == "canonical_gtol" or result.canonical_gradient_max_abs <= 1e-6:
        return "converged"
    if result.terminal_reason in ("fg_budget_exhausted", "accepted_iteration_guard"):
        return "budget_not_converged"
    if result.terminal_reason in ("ftol_numeric_stop", "solver_reported_success", "solver_reported_nonconvergence"):
        return "not_converged"
    return "failure"


def run_fit(condition: str, seed: int):
    fit_id = "%s-seed%d" % (condition, seed)
    fit_started = time.perf_counter()
    rows = []
    previous_state = None
    previous_data = None
    for stage_index, (stage, bin_size, cap) in enumerate(STAGES):
        data = load_data(condition, bin_size)
        if stage_index == 0:
            coordinates, raw_y, p_init, q_init, start_meta = initial_start(seed)
        else:
            coordinates, raw_y, p_init, q_init, start_meta = warm_start(
                previous_state, previous_data, data, bin_size, seed)
        if coordinates.shape != (2, data.n_loci, 3):
            raise RuntimeError("start shape differs from target full grid")
        mapping_error = float(np.max(np.abs(contact_model.sphere_forward(raw_y) - coordinates)))
        if mapping_error > 1e-12:
            raise RuntimeError("stage start raw_y mapping error")
        objective = SharedCaptureObjective(
            data, "G", weights=PenaltyWeights(**WEIGHTS), mode="V0-fixed-production-e",
            device="cuda", pair_block=262_144, inner_cap=80, cg_cap=80,
            profile_warm_start=True, known_e=None)
        stage_started = time.perf_counter()
        result = run_budgeted_lbfgs_no_extra(
            objective, raw_y, p_init=p_init, q_init=q_init, maxfun=cap,
            maxiter=cap + 1, maxls=20, ftol=0.0, canonical_gtol=1e-6,
            checkpoint_every=10, checkpoint_hook=checkpoint_hook(fit_id, stage))
        status = classify(result)
        endpoint_dir = RUN / "coords" / fit_id
        endpoint_dir.mkdir(parents=True, exist_ok=True)
        legacy = endpoint_dir / (stage + ".npz")
        if legacy.exists():
            raise RuntimeError("refusing to overwrite endpoint")
        with legacy.open("xb") as handle:
            np.savez_compressed(
                handle, coordinates=np.asarray(result.coordinates, dtype=np.float64),
                raw_y=np.asarray(result.y, dtype=np.float64),
                theta=np.asarray(result.theta, dtype=np.float64),
                p=np.asarray(result.p, dtype=np.float64),
                q=np.asarray(result.theta[-1], dtype=np.float64))
        history_path = RUN / "stages" / fit_id / (stage + ".accepted_history.json")
        write_json(history_path, result.history)
        record = {
            "schema": "p9016-056-exact-budget-stage-v1", "fit_id": fit_id,
            "condition": condition, "seed": seed, "stage": stage, "bin_size_bp": bin_size,
            "status": status, "terminal_reason": result.terminal_reason,
            "fg_cap": cap, "outer_fg_actual": int(result.nfev),
            "optimizer": result.as_dict(), "endpoint": {
                "p": float(result.p), "q": float(result.theta[-1]),
                "total": float(result.fun), "components": result.components,
                "canonical_gradient_max_abs": float(result.canonical_gradient_max_abs),
                "canonical_gradient_norm": float(result.canonical_gradient_norm),
            },
            "fit_wall_seconds": float(time.perf_counter() - stage_started),
            "start_metadata": start_meta, "start_mapping_max_abs": mapping_error,
            "weights": WEIGHTS, "history_path": str(history_path.relative_to(ROOT)),
            "full_grid_accounting": {
                "training_fg_calls": int(result.nfev),
                "budget_extra_init_calls": 0, "budget_extra_endpoint_readback_calls": 0,
                "estimated_pair_kernel_sweeps_per_training_fg": 2,
                "validation_calls": int(result.validation_calls),
            },
            "reference_opened": False, "phase_opened": False,
        }
        stage_record = RUN / "stages" / fit_id / (stage + ".json")
        write_json(stage_record, record)
        if record.get("status") not in ("converged", "not_converged", "budget_not_converged"):
            raise RuntimeError("stage failed: %s/%s: %s" % (fit_id, stage, record.get("error")))
        frozen = freeze_stage(fit_id, stage, data, legacy)
        freeze_record = RUN / "states/experiment3" / fit_id / stage / "artifacts.json"
        write_json(freeze_record, frozen)
        rows.append({
            "stage": stage, "bin_size": bin_size, "fg_cap": cap,
            "outer_fg_actual": int(record["outer_fg_actual"]),
            "status": record["status"], "terminal_reason": record["terminal_reason"],
            "canonical_gradient_max_abs": float(record["endpoint"]["canonical_gradient_max_abs"]),
            "fit_wall_seconds": float(record["fit_wall_seconds"]),
            "immutable_artifacts": frozen,
        })
        previous_state = RUN / "states/experiment3" / fit_id / stage / "solver_state.npz"
        previous_data = data
        print(json.dumps({"event": "056_stage_end", "fit_id": fit_id, "stage": stage,
                          "status": record["status"], "outer_fg_actual": record["outer_fg_actual"],
                          "canonical_gradient_max_abs": record["endpoint"]["canonical_gradient_max_abs"]},
                         sort_keys=True), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        "fit_id": fit_id, "condition": condition, "seed": seed,
        "status": "complete", "stages": rows,
        "outer_fg_actual": int(sum(row["outer_fg_actual"] for row in rows)),
        "outer_fg_cap": 1502,
        "wall_seconds": float(time.perf_counter() - fit_started),
        "reference_opened": False, "phase_opened": False,
        "initialization_replicate_semantics": "optimization initialization only; not biological replication",
    }
    write_json(RUN / "results/training" / (fit_id + ".json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", required=True)
    parser.parse_args()
    terminal_path = RUN / "logs/training_terminal.json"
    terminal = {"schema": "p9016-056-training-terminal-v1", "status": "running",
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "reference_opened": False, "phase_opened": False, "fits": []}
    write_json(terminal_path, terminal)
    try:
        # Condition order is fixed; each condition uses the same immutable seed-specific 5Mb state.
        for seed in SEEDS:
            initial_hashes = []
            for condition in CONDITIONS:
                initial_hashes.append(initial_start(seed)[4]["sha256"])
            if len(set(initial_hashes)) != 1:
                raise RuntimeError("same-seed condition starts differ")
            for condition in CONDITIONS:
                summary = run_fit(condition, seed)
                terminal["fits"].append(summary)
                write_json(terminal_path, terminal)
        total = int(sum(item["outer_fg_actual"] for item in terminal["fits"]))
        fine = int(sum(row["outer_fg_actual"] for item in terminal["fits"]
                       for row in item["stages"] if row["stage"] == "1Mb"))
        if total > 6008 or fine > 1944:
            raise RuntimeError("training FG budget exceeded")
        terminal.update({"status": "complete", "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                         "outer_fg_actual": total, "outer_fg_cap": 6008,
                         "fine_1Mb_fg_actual": fine, "fine_1Mb_fg_cap": 1944,
                         "stage_cpu_initial_checks": 12,
                         "auxiliary_full_grid_calls_during_training": 0,
                         "estimated_training_pair_kernel_sweeps": 2 * total})
        write_json(terminal_path, terminal)
        return 0
    except Exception as error:
        terminal.update({"status": "failure", "error_type": type(error).__name__, "error": str(error),
                         "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        write_json(terminal_path, terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
