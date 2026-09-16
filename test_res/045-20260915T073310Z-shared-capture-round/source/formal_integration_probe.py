"""Fresh-process short integrations through the actual formal stage_fit branches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

# Match an independent formal_controller process, including frozen-pr precedence.
import formal_controller as controller  # noqa: E402
from data_io import save_start, sha256_file  # noqa: E402


def _tiny_data():
    ci = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    cj = np.asarray([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int64)
    p1 = np.asarray([0, 1, 0, 1, 2, 3, 3, 2], dtype=np.int64)
    p2 = np.asarray([1, 2, 1, 2, 3, 0, 0, 3], dtype=np.int64)
    return controller.contact_model.aggregate_from_arrays(("chr1", "chr2"), (4, 4), ci, p1, cj, p2, 1)


def _module_record(module: object) -> dict[str, str]:
    path = Path(str(getattr(module, "__file__", ""))).resolve()
    return {"file": str(path), "sha256": sha256_file(path)}


def _schema_check(integration_root: Path, fit_id: str, stage: str, record: dict[str, Any]) -> dict[str, Any]:
    npz = integration_root / f"coords/{fit_id}/{stage}.npz"
    history = integration_root / f"stages/{fit_id}/{stage}.accepted_history.json"
    iter0 = integration_root / f"checkpoints/{fit_id}/{stage}/accepted-00000.npz"
    with np.load(npz, allow_pickle=False) as payload:
        required = ("coordinates", "raw_y", "theta", "p", "q")
        finite = all(np.all(np.isfinite(np.asarray(payload[key], dtype=np.float64))) for key in required)
        keys = sorted(payload.files)
        q = float(np.asarray(payload["q"]).item())
    with np.load(iter0, allow_pickle=False) as iter0_payload:
        iter0_required = ("coordinates", "raw_y", "theta", "optimizer_gradient", "canonical_gradient")
        iter0_finite = all(np.all(np.isfinite(np.asarray(iter0_payload[key], dtype=np.float64))) for key in iter0_required)
    history_rows = json.loads(history.read_text(encoding="utf-8"))
    result = {"npz_keys": keys, "finite": finite, "iter0_finite": iter0_finite, "history_rows": len(history_rows),
              "history_first_iteration": int(history_rows[0]["iteration"]), "iter0_artifact_exists": iter0.is_file(),
              "iter0_sha256": sha256_file(iter0), "record_status": record["status"],
              "terminal_reason": record.get("terminal_reason"), "outer_fg_actual": int(record["outer_fg_actual"]),
              "initial_raw_y_source": record.get("initial_raw_y_source"),
              "initial_raw_y_mapping_max_abs_error": record.get("initial_raw_y_mapping_max_abs_error"),
              "endpoint_q": q, "record_endpoint_q": float(record["endpoint"]["q"])}
    if not finite or not iter0_finite or result["history_first_iteration"] != 0 or not result["iter0_artifact_exists"]:
        raise RuntimeError(f"integration schema failed for {fit_id}/{stage}")
    return result


def _run_actual_branch(integration_root: Path, source_run: Path, manifest: dict[str, Any], row: dict[str, Any], data: Any,
                       known_e: np.ndarray | None, label: str) -> dict[str, Any]:
    fit_id = f"integration-{label}"
    start_key = "start_5Mb" if row["kind"] == "real" else "start_path"
    start_path_value = Path(str(row[start_key]))
    start_path = start_path_value.resolve() if start_path_value.is_absolute() else (source_run / start_path_value).resolve()
    coordinates, raw_y, p_init, _meta = controller.load_start(start_path)
    stage = f"{label}-1Mb" if row["kind"] == "synthetic" else f"{label}-5Mb"
    spec = {"stage": stage, "bin_size_bp": int(data.bin_size), "fg_cap": 2}
    fit = dict(row)
    fit["fit_id"] = fit_id
    record = controller.stage_fit(fit, spec, data, coordinates, p_init,
                                  controller.contact_model.q_from_p(p_init), known_e,
                                  initial_raw_y=raw_y)
    schema = _schema_check(integration_root, fit_id, stage, record)
    return {"fit_id": fit_id, "source_row": row["fit_id"], "branch": label,
            "data_count_mode": data.count_mode, "model_id": row["model_id"],
            "start_name": row.get("start_name"), "status": record["status"],
            "terminal_reason": record.get("terminal_reason"), "schema": schema}


def run(run_dir: Path) -> dict[str, Any]:
    original_run = controller.RUN
    source_run = run_dir.resolve()
    integration_root = (source_run / "checks" / "integration_stage_fit").resolve()
    for name in ("coords", "stages", "checkpoints", "logs"):
        (integration_root / name).mkdir(parents=True, exist_ok=True)
    controller.RUN = source_run
    manifest = controller._load_manifest()
    real_5mb = controller._load_real_data(manifest, 5_000_000)
    actual_rows = []
    for row in manifest["matrix"]:
        if row.get("kind") == "real":
            actual_rows.append((row, real_5mb, None, f"real-{row['model_id']}-{row['candidate']}"))
    for row in manifest["matrix"]:
        if row.get("kind") != "synthetic":
            continue
        if row.get("objective_variant") == "full-J" and row.get("start_name") in ("near", "blind"):
            actual_rows.append((row, controller._load_synthetic_data(row), controller._load_known_e(row),
                                row["fit_id"].replace("synthetic-", "syn-")))
        elif row.get("objective_variant") == "count-only":
            actual_rows.append((row, controller._load_synthetic_data(row), controller._load_known_e(row),
                                row["fit_id"].replace("synthetic-", "syn-")))
    controller.RUN = integration_root
    integration_results = []
    for row, data, known_e, label in actual_rows:
        integration_results.append(_run_actual_branch(integration_root, source_run, manifest, row, data, known_e, label))
    rng = np.random.default_rng(450002)
    coordinates = rng.normal(size=(2, data.n_loci, 3)).astype(np.float64)
    coordinates *= 0.45 / float(np.linalg.norm(coordinates, axis=2).max())
    objective = controller.SharedCaptureObjective(
        data, model_id="S", weights=controller.PenaltyWeights(), mode="V0-fixed-production-e",
        device="cuda", pair_block=64, inner_cap=8, cg_cap=8, profile_warm_start=False,
    )
    raw_y = objective.raw_from_physical(coordinates)
    start_path = integration_root / "start.npz"
    start_record = save_start(start_path, coordinates, raw_y, 0.8,
                              {"integration": True, "raw_y_serialized": True})
    loaded_coordinates, loaded_raw_y, p_init, _ = controller.load_start(start_path)
    tiny_fit = {"fit_id": "integration-tiny-S", "kind": "real", "model_id": "S", "candidate": "consensus",
                "weights": {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}}
    tiny_known_e = np.asarray(data.exposure, dtype=np.float64).copy()
    tiny1 = controller.stage_fit(tiny_fit, {"stage": "mock1", "bin_size_bp": 1, "fg_cap": 2}, data,
                                 loaded_coordinates, p_init, controller.contact_model.q_from_p(p_init), tiny_known_e,
                                 initial_raw_y=loaded_raw_y)
    endpoint1 = integration_root / "coords/integration-tiny-S/mock1.npz"
    with np.load(endpoint1, allow_pickle=False) as payload:
        coords1 = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        p1 = float(np.asarray(payload["p"]).item())
        q1 = float(np.asarray(payload["q"]).item())
    tiny2 = controller.stage_fit(tiny_fit, {"stage": "mock2", "bin_size_bp": 1, "fg_cap": 2}, data,
                                 coords1, p1, q1, tiny_known_e, initial_raw_y=None)
    tiny_schema = {
        "mock1": _schema_check(integration_root, "integration-tiny-S", "mock1", tiny1),
        "mock2": _schema_check(integration_root, "integration-tiny-S", "mock2", tiny2),
    }
    classification = {}
    class Result:
        canonical_gradient_max_abs = 1.0
        def __init__(self, reason):
            self.terminal_reason = reason
    reasons = ("canonical_gtol", "fg_budget_exhausted", "accepted_iteration_guard", "ftol_numeric_stop",
               "scipy_stop", "solver_reported_success", "solver_reported_nonconvergence", "unknown_error")
    for reason in reasons:
        classification[reason] = controller._classify(Result(reason))
    if classification["solver_reported_nonconvergence"] != "not_converged":
        raise RuntimeError("solver_reported_nonconvergence classification regression")
    if float(tiny2["initial_q"]) != q1:
        raise RuntimeError("q carry is not exact")
    if len(integration_results) != 14:
        raise RuntimeError(f"expected 14 actual branch integrations, got {len(integration_results)}")
    output = {"schema": "p9016-formal-stage-integration-v2", "status": "PASS", "fresh_process": True,
              "optimizer_scope": "scratch integration only; actual branches FG=2; formal matrix untouched",
              "actual_branch_count": len(integration_results), "actual_branches": integration_results,
              "tiny_continuation": {"start_record": start_record, "schema": tiny_schema,
                                    "q_carry_exact": True, "actual_fg_total": int(tiny1["outer_fg_actual"] + tiny2["outer_fg_actual"])},
              "classification": classification,
              "modules": {"formal_controller": _module_record(controller),
                          "pr.contact_model": _module_record(sys.modules["pr.contact_model"]),
                          "pr.reconstruction_init": _module_record(sys.modules["pr.reconstruction_init"]),
                          "visibility_profile_base": _module_record(sys.modules["visibility_profile_base"]),
                          "gpu_variant_backend": _module_record(sys.modules["gpu_variant_backend"]),
                          "m1_preconditioner": _module_record(sys.modules["m1_preconditioner"])} }
    controller.RUN = original_run
    (run_dir / "checks/integration_stage_fit.json").write_text(json.dumps(output, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = run(Path(args.run_dir).resolve())
    print(json.dumps({"status": result["status"], "actual_branch_count": result["actual_branch_count"], "fresh_process": result["fresh_process"]}))


if __name__ == "__main__":
    main()
