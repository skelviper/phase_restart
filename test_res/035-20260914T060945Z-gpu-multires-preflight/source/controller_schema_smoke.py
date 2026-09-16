"""仅控制器层面的 fake-chain 和 frozen-artifact schema 检查。

这里不会启动真实 optimizer 或全数据拟合。fake chain 用于检验缓存计数、失败后的继续执行、attempt 审计和终态守门。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "test_res/035-20260914T060945Z-gpu-multires-preflight"
SOURCE = ARTIFACT / "source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import gpu_multires_controller as controller  # noqa: E402
from pr import reconstruct  # noqa: E402


def _write_json(path: Path, value):
    payload = (json.dumps(controller._jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _fake_chain_checks(context):
    original_load = controller.contact_model.load_frozen_p9016_aggregate
    original_validate_layer = controller.cpu_runner._validate_layer
    original_initialize = controller.reconstruction_init.initialize_approved_candidate
    original_transfer = controller.cpu_runner.transfer_layer
    original_validate_init = controller.cpu_runner._validate_initialization
    original_writer = controller.paired_run.write_coordinates
    original_fit = controller._fit_gpu
    loads = []
    coords = np.zeros((2, 1, 3), dtype=np.float64)
    positions = np.zeros(1, dtype=np.int64)
    chromosome = np.zeros(1, dtype=np.int64)
    fake_data = SimpleNamespace(n_loci=1)
    candidate = reconstruct.DEFAULT_CANDIDATES[0]

    def fake_load(bin_size):
        loads.append(int(bin_size))
        return fake_data

    def fake_state(*args, **kwargs):
        bin_size = next(int(value) for value in reversed(args)
                        if isinstance(value, (int, np.integer))
                        and int(value) in (5_000_000, 2_000_000, 1_000_000))
        return {"coords": coords.copy(), "positions": positions.copy(),
                "chromosome_index": chromosome.copy(), "names": tuple(name for name, _ in context.headers),
                "header_lengths": tuple(length for _, length in context.headers), "bin_size": bin_size}

    def fake_validate_init(state, data, model):
        return coords.copy(), positions.copy(), chromosome.copy(), {"fake": True}

    def fake_writer(path, data, model, values):
        return {"path": str(path), "sha256": "fake-coordinate-sha", "model_id": model,
                "physical_domain": "strict_unit_ball", "full_grid": True,
                "n_tracks": 40, "n_beads": 2, "max_radius": 0.0,
                "serialization_clip": {"clipped_coordinates": 0, "applied": False}}

    def fake_fit(root, model, cand, stage, data, initial_coordinates, positions_arg,
                 chromosome_arg, q_init, device):
        fit_payload = {
            "status": "not_converged", "budget_exhausted": True,
            "q_out": 0.0, "solver": {"actual_nfev": 1},
        }
        result = SimpleNamespace(theta=np.asarray([0.0], dtype=np.float64))
        return result, fit_payload, np.asarray(initial_coordinates, dtype=np.float64).copy()

    controller.contact_model.load_frozen_p9016_aggregate = fake_load
    controller.cpu_runner._validate_layer = lambda data, ctx, stage: {"fake": True}
    controller.reconstruction_init.initialize_approved_candidate = fake_state
    controller.cpu_runner.transfer_layer = fake_state
    controller.cpu_runner._validate_initialization = fake_validate_init
    controller.paired_run.write_coordinates = fake_writer
    controller._fit_gpu = fake_fit
    try:
        with tempfile.TemporaryDirectory(prefix="gpu-controller-chain-") as temp:
            root = Path(temp)
            cache = {}
            success = controller._run_chain(root, context, {}, "C0", candidate, "cuda", cache)
            success_cache = {str(key): int(loads.count(key)) for key in sorted(set(loads))}
            if sorted(loads) != [1_000_000, 2_000_000, 5_000_000]:
                raise AssertionError(f"data cache loaded unexpected keys/order: {loads}; chain={success}")
            if any(count != 1 for count in success_cache.values()):
                raise AssertionError(f"data cache loaded a layer more than once: {success_cache}")
            if len(success["attempts"]) != 3 or success["optimization"]["terminal_status"] == "failed":
                raise AssertionError("fake successful chain did not produce three completed attempts")

        loads.clear()
        def failing_fit(root, model, cand, stage, data, initial_coordinates, positions_arg,
                        chromosome_arg, q_init, device):
            if stage.label == "2m":
                raise RuntimeError("injected stage failure")
            return fake_fit(root, model, cand, stage, data, initial_coordinates,
                            positions_arg, chromosome_arg, q_init, device)
        controller._fit_gpu = failing_fit
        with tempfile.TemporaryDirectory(prefix="gpu-controller-failure-") as temp:
            failed = controller._run_chain(Path(temp), context, {}, "C0", candidate, "cuda", {})
            statuses = [row["status"] for row in failed["attempts"]]
            if statuses != ["not_converged", "failed", "not_run_after_prior_failure"]:
                raise AssertionError(f"failure continuation statuses changed: {statuses}")
            if failed["optimization"]["terminal_status"] != "failed":
                raise AssertionError("failed chain was not marked failed")
        return {
            "success_chain": {"attempt_count": 3, "load_count_by_bin": success_cache},
            "failure_chain": {"statuses": statuses, "attempt_count": len(statuses)},
        }
    finally:
        controller.contact_model.load_frozen_p9016_aggregate = original_load
        controller.cpu_runner._validate_layer = original_validate_layer
        controller.reconstruction_init.initialize_approved_candidate = original_initialize
        controller.cpu_runner.transfer_layer = original_transfer
        controller.cpu_runner._validate_initialization = original_validate_init
        controller.paired_run.write_coordinates = original_writer
        controller._fit_gpu = original_fit


def _attempt_audit_checks():
    planned = controller._planned_rows()
    candidates = []
    for model in controller.VARIANTS:
        for candidate in controller.CANDIDATES:
            attempts = []
            for row in planned:
                if row["variant"] == model and row["candidate_id"] == candidate.candidate_id:
                    attempts.append({"attempt_id": row["attempt_id"], "model_id": model,
                                     "candidate_id": candidate.candidate_id, "stage": row["stage"],
                                     "status": "not_converged", "stage_record": "stage.json"})
            candidates.append({"id": candidate.candidate_id, "model_id": model, "attempts": attempts,
                               "final_coordinates_path": "coords/final-1m.3dg",
                               "final_coordinates_sha256": "coordinate-sha", "final_q": 0.0,
                               "optimization": {"terminal_status": "not_converged"}})
    passing = controller._attempt_audit(candidates, {"attempt_plan": {"planned_attempts": planned}})
    if not passing["pass"] or passing["actual_stage_attempt_count"] != 30 or passing["actual_trajectory_count"] != 10:
        raise AssertionError(f"valid attempt audit failed: {passing}")
    candidates[0]["attempts"][1]["status"] = "failed"
    candidates[0]["attempts"][2]["status"] = "not_run_after_prior_failure"
    failing = controller._attempt_audit(candidates, {"attempt_plan": {"planned_attempts": planned}})
    if failing["pass"] or failing["failed_or_not_run_stage_attempt_count"] != 2:
        raise AssertionError(f"failed attempt audit was not rejected: {failing}")
    return {
        "passing_audit": {"pass": passing["pass"], "actual_stage_attempt_count": passing["actual_stage_attempt_count"],
                          "actual_trajectory_count": passing["actual_trajectory_count"]},
        "failing_audit": {"pass": failing["pass"],
                          "failed_or_not_run_stage_attempt_count": failing["failed_or_not_run_stage_attempt_count"]},
    }


def main():
    context = reconstruct.production_context()
    reconstruct._validate_context(context)
    protocol = controller.cpu_runner._read_json(ARTIFACT / "protocol.json")
    config = controller.cpu_runner._read_json(ARTIFACT / "config.json")
    integrity = controller._validate_frozen_artifact(ARTIFACT, protocol, config)
    with tempfile.TemporaryDirectory(prefix="gpu-controller-device-") as temp:
        try:
            controller.run(ARTIFACT, Path(temp) / "forbidden-cpu-output", device="cpu", workers=1)
        except ValueError as exc:
            cpu_rejection = str(exc)
        else:
            raise AssertionError("controller accepted --device cpu")
    chain = _fake_chain_checks(context)
    audit = _attempt_audit_checks()
    evidence = {
        "schema": "gpu-multires-controller-schema-test-v1",
        "status": "passed",
        "frozen_artifact_integrity": integrity,
        "device_cpu_rejection": {"passed": True, "message": cpu_rejection},
        "fake_chain": chain,
        "attempt_audit": audit,
        "formal_training_started": False,
        "real_optimizer_called": False,
        "real_data_loader_called": False,
    }
    path = ARTIFACT / "preflight" / "controller_schema_test.json"
    digest = _write_json(path, evidence)
    print(json.dumps({"status": "passed", "path": str(path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
