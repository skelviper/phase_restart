"""Substantive M1 preflight; no formal P9016 optimizer is launched."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "test_res" / "037-20260914T143812Z-gpu-m1-preflight"
SOURCE_DIR = Path(__file__).resolve().parent
SOURCE_035 = ROOT / "test_res" / "035-20260914T060945Z-gpu-multires-preflight" / "source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_035) not in sys.path:
    sys.path.insert(0, str(SOURCE_035))
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from gpu_variant_backend import GPUVariantObjective, cuda_probe  # noqa: E402
from m1_preconditioner import (  # noqa: E402
    M1Objective,
    common_difference_gradient_to_raw,
    common_difference_to_raw,
    orthogonal_ab_to_raw,
    orthogonal_raw_to_ab,
    raw_gradient_to_common_difference,
    raw_to_common_difference,
    run_budgeted_lbfgs,
)
import m1_gpu_controller as controller  # noqa: E402
from pr import contact_model, genome, paired_run, reconstruction_init  # noqa: E402
from pr import multires_variant_runner as warm  # noqa: E402

EXPECTED_KNOWN_SHA = {
    "P2": "cd2443191826008385911e9008a0b43deae46fde5322fa13f5d5582c927c73b6",
    "N2": "ca77e1b4f2f7957bc2c3898b9bdd8fbd99a884c62166918a3bd82777ff99c7c9",
}
KNOWN_E_ROOT = ROOT / "docs/audits/next-step-r2-preparation-20260914T143656Z/synthetic_inputs"
KNOWN_E_MANIFEST = KNOWN_E_ROOT / "known_e_worker_input_manifest.json"
KNOWN_E_PATHS = {
    "P2": KNOWN_E_ROOT / "P2_generation_exposure.npz",
    "N2": KNOWN_E_ROOT / "N2_generation_exposure.npz",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _max_numeric_diff(left: dict, right: dict) -> float:
    values = []
    for key in set(left) & set(right):
        a, b = left[key], right[key]
        if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
            values.append(abs(float(a) - float(b)))
    return max(values, default=0.0)


def _probe() -> dict:
    result = dict(cuda_probe())
    result["dtype_requested"] = "torch.float64"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device)
    value = (x * x).sum()
    torch.cuda.synchronize(device)
    _assert(x.dtype is torch.float64, "CUDA probe tensor did not retain float64")
    _assert(value.dtype is torch.float64, "CUDA probe result did not retain float64")
    result.update({
        "device_index": 0, "device_name": props.name, "total_memory": int(props.total_memory),
        "tensor_device": str(x.device), "tensor_dtype": str(x.dtype),
        "tensor_result_dtype": str(value.dtype), "tensor_result": float(value.cpu().item()),
        "float64_exact": True, "status": "passed",
    })
    return result


def _real_fixed_state_tests() -> dict:
    headers = controller._real_headers()
    names = tuple(name for name, _ in headers)
    lengths = tuple(int(length) for _, length in headers)
    cache: dict[int, object] = {}
    previous = None
    layers = []
    for stage in controller.REAL_STAGES:
        data = controller._load_real_data(cache, stage["bin_size_bp"])
        candidate = controller.CANDIDATES[0]
        state = controller._initial_state(candidate, data, previous, headers)
        coordinates, positions, chromosome_index, init_meta = controller._validate_initialization(data, state)
        base = GPUVariantObjective(data, model_id="C0", device="cuda", dtype=torch.float64,
                                   use_fused=True, diagnostics=True)
        m1 = M1Objective(base)
        raw = base.raw_from_physical(coordinates)
        theta0 = base.pack(raw, p=0.75)
        theta1 = m1.optimizer_theta_from_raw_theta(theta0)
        value0, gradient0, components0 = base.evaluate(theta0, need_gradient=True)
        value1, gradient1, components1 = m1.evaluate(theta1, need_gradient=True)
        coordinates1, p1 = m1.coordinates_and_p(theta1)
        canonical1 = m1.canonical_raw_gradient(theta1, gradient1)
        _assert(np.array_equal(theta0[-1:], theta1[-1:]), "M1 changed q")
        _assert(np.max(np.abs(coordinates - coordinates1)) < 2e-14, "M1 physical reconstruction is not reversible")
        _assert(abs(float(value0) - float(value1)) < 2e-10, "M0/M1 objective value mismatch")
        component_diff = _max_numeric_diff(components0, components1)
        _assert(component_diff < 2e-8, "M0/M1 component mismatch %.3g" % component_diff)
        gradient_diff = float(np.max(np.abs(np.asarray(gradient0) - canonical1)))
        _assert(gradient_diff < 3e-8, "M0/M1 canonical gradient mismatch %.3g" % gradient_diff)
        _assert(abs(float(p1) - 0.75) < 1e-14, "M1 p conversion changed")
        _assert(base.data.n_pairs == data.n_loci * (data.n_loci - 1) // 2, "pair grid is not complete")
        audit = data.budget()
        _assert(audit["raw_records"] == 1_703_888, "real raw budget changed")
        expected_cis = controller.EXPECTED_REAL_CIS_OFFDIAG[stage["bin_size_bp"]]
        _assert(audit["raw_cis_offdiag"] == expected_cis and audit["raw_inter"] == 568_434,
                "real group budget changed at %s" % stage["label"])
        layers.append({
            "stage": stage["label"], "bin_size_bp": stage["bin_size_bp"], "n_loci": int(data.n_loci),
            "n_pairs": int(data.n_pairs), "zero_eligible_pairs": int(data.n_pairs - len(base.data.observed_flat)),
            "value_abs_diff": abs(float(value0) - float(value1)), "components_max_abs_diff": component_diff,
            "canonical_gradient_max_abs_diff": gradient_diff,
            "physical_reconstruction_max_abs": float(np.max(np.abs(coordinates - coordinates1))),
            "q_abs_diff": abs(float(theta0[-1]) - float(theta1[-1])),
            "p": float(p1), "full_grid": True, "dtype": str(base.dtype), "device": str(base.device),
            "initialization_metadata": init_meta,
        })
        previous = {"coords": coordinates, "positions": positions, "chromosome_index": chromosome_index}
    return {"status": "passed", "layers": layers, "warm_transfer_checked": True,
            "same_physical_state_m0_m1": True, "same_q": True, "all_pairs": True}


def _transform_tests() -> dict:
    rng = np.random.default_rng(3701)
    y = rng.normal(size=(2, 7, 3)).astype(np.float64)
    z = raw_to_common_difference(y, scale=2.0)
    y_back = common_difference_to_raw(z, scale=2.0)
    _assert(float(np.max(np.abs(y - y_back))) < 5e-16, "scale-2 transform is not reversible")
    gy = rng.normal(size=y.shape).astype(np.float64)
    gz = raw_gradient_to_common_difference(gy, scale=2.0)
    gy_back = common_difference_gradient_to_raw(gz, scale=2.0)
    _assert(float(np.max(np.abs(gy - gy_back))) < 5e-16, "scale-2 gradient chain is not reversible")
    y1 = orthogonal_ab_to_raw(orthogonal_raw_to_ab(y))
    g1 = common_difference_gradient_to_raw(raw_gradient_to_common_difference(gy, scale=1.0), scale=1.0)
    _assert(float(np.max(np.abs(y - y1))) < 5e-16, "scale-1 orthogonal transform failed")
    _assert(float(np.max(np.abs(gy - g1))) < 5e-16, "scale-1 orthogonal gradient failed")
    direction = rng.normal(size=y.shape)
    lhs = float(np.dot(gy.reshape(-1), direction.reshape(-1)))
    rhs = float(np.dot(raw_gradient_to_common_difference(gy, scale=1.0).reshape(-1),
                       orthogonal_raw_to_ab(direction).reshape(-1)))
    _assert(abs(lhs - rhs) < 2e-14, "scale-1 directional chain rule failed")
    return {"status": "passed", "scale2_roundtrip_max_abs": float(np.max(np.abs(y - y_back))),
            "scale2_gradient_roundtrip_max_abs": float(np.max(np.abs(gy - gy_back))),
            "scale1_orthogonal_roundtrip_max_abs": float(np.max(np.abs(y - y1))),
            "scale1_directional_abs_diff": abs(lhs - rhs), "formal_scale1_arm": False}


def _finite_difference_test() -> dict:
    data = warm._synthetic_data(10)
    base = GPUVariantObjective(data, model_id="C0", device="cuda", dtype=torch.float64,
                               use_fused=True, diagnostics=True)
    m1 = M1Objective(base)
    rng = np.random.default_rng(3711)
    raw = rng.normal(scale=0.08, size=(2, data.n_loci, 3)).astype(np.float64)
    theta = m1.pack(raw, p=0.75)
    value, gradient, _ = m1.evaluate(theta, need_gradient=True)
    indices = [0, 1, 3, 7, len(theta) // 2, len(theta) - 2, len(theta) - 1]
    errors = []
    step = 2e-6
    for index in indices:
        plus = theta.copy(); plus[index] += step
        minus = theta.copy(); minus[index] -= step
        value_plus = m1.evaluate(plus, need_gradient=False)[0]
        value_minus = m1.evaluate(minus, need_gradient=False)[0]
        finite_difference = (float(value_plus) - float(value_minus)) / (2.0 * step)
        analytic = float(gradient[index])
        error = abs(analytic - finite_difference) / max(1e-7, abs(analytic), abs(finite_difference))
        errors.append({"index": int(index), "analytic": analytic, "finite_difference": finite_difference,
                       "relative_error": error})
    max_error = max(item["relative_error"] for item in errors)
    _assert(max_error < 3e-4, "M1 finite-difference relative error %.3g" % max_error)
    return {"status": "passed", "fixture": "035 tiny synthetic no truth", "value": float(value),
            "indices": errors, "max_relative_error": max_error}


def _budget_test() -> dict:
    data = warm._synthetic_data(10)
    base = GPUVariantObjective(data, model_id="C0", device="cuda", dtype=torch.float64,
                               use_fused=True, diagnostics=True)
    objective = M1Objective(base)
    rng = np.random.default_rng(3721)
    initial = rng.normal(scale=0.05, size=(2, data.n_loci, 3)).astype(np.float64)
    result = run_budgeted_lbfgs(objective, initial, p_init=0.75, maxfun=2, maxiter=3,
                                maxls=20, ftol=1e-10, canonical_gtol=1e-20)
    _assert(result.nfev <= 2, "hard FG cap exceeded")
    _assert(result.budget_exhausted and result.terminal_reason == "fg_budget_exhausted",
            "small fixture did not hard-stop at FG cap")
    _assert(result.endpoint_was_last_accepted, "budget endpoint is not accepted state")
    _assert(result.history[-1]["state_status"] in ("accepted_initial", "accepted", "accepted_endpoint_readback"),
            "budget endpoint history is not accepted")
    return {"status": "passed", "cap": 2, "optimizer_fg_calls": int(result.nfev),
            "accepted_iterations": int(result.nit), "terminal_reason": result.terminal_reason,
            "endpoint_was_last_accepted": bool(result.endpoint_was_last_accepted),
            "validation_calls": int(result.validation_calls), "formal_fit_started": False}


def _status_mapping_test() -> dict:
    from types import SimpleNamespace
    cases = [
        (SimpleNamespace(terminal_reason="canonical_gtol", canonical_gradient_max_abs=1e-6,
                         budget_exhausted=False, success=True), "converged"),
        (SimpleNamespace(terminal_reason="fg_budget_exhausted", canonical_gradient_max_abs=1.0,
                         budget_exhausted=True, success=False), "budget_not_converged"),
        (SimpleNamespace(terminal_reason="ftol_numeric_stop", canonical_gradient_max_abs=1.0,
                         budget_exhausted=False, success=True), "numeric_stop_not_converged"),
        (SimpleNamespace(terminal_reason="solver_reported_nonconvergence", canonical_gradient_max_abs=1.0,
                         budget_exhausted=False, success=False), "solver_not_converged"),
        (SimpleNamespace(terminal_reason="solver_reported_success", canonical_gradient_max_abs=1.0,
                         budget_exhausted=False, success=True), "numeric_stop_not_converged"),
    ]
    observed = []
    for fit, expected in cases:
        status, reason = controller._fit_state(fit)
        _assert(status == expected, "%s mapped to %s, expected %s" % (fit.terminal_reason, status, expected))
        observed.append({"terminal_reason": fit.terminal_reason, "status": status, "status_reason": reason})
    return {"status": "passed", "cases": observed,
            "gradient_converged_requires_exact_canonical_reason": True}


def _completion_gate_test() -> dict:
    def fake_row(kind: str, index: int) -> dict:
        return {"kind": kind, "method": "M0", "budget_id": "B1", "candidate_id": "c%d" % index,
                "optimization": {"terminal_status": "budget_not_converged"},
                "final_coordinates": {"path": "coords/%d.3dg" % index, "sha256": "a" * 64},
                "selection_rescore": {"criterion": "count_nll_per_record", "calls": 1},
                "count_nll_per_record": float(index)}
    real = [fake_row("real", index) for index in range(8)]
    synthetic = [fake_row("synthetic", index + 8) for index in range(6)]
    attempts = [{"attempt_id": "a%d" % index, "stage_record": "s/%d.json" % index,
                 "status": "budget_not_converged"} for index in range(30)]
    selection = {"per_method_budget": {
        "%s__%s" % (method, budget): {"status": "selected", "candidate_scores": {
            "consensus_joint": 1.0, "random_joint": 2.0}}
        for method in ("M0", "M1") for budget in ("B1", "B2")}}
    passed = controller._completion_gate(real, synthetic, attempts, selection)
    _assert(passed["status"] == "release_ready" and passed["all_conditions_pass"],
            "complete synthetic gate fixture did not pass")
    incomplete = controller._completion_gate(real, synthetic, attempts[:-1] +
                                               [{"attempt_id": "notrun", "stage_record": "s/notrun.json",
                                                 "status": "not_run_after_prior_failure"}], selection)
    _assert(incomplete["status"] == "incomplete_with_failures" and
            incomplete["conditions"]["executed_stage_attempts_30"] is False,
            "not-run fixture did not fail completion gate")
    return {"status": "passed", "complete_status": passed["status"],
            "incomplete_status": incomplete["status"], "gate_requires_30_executed_unique_stages": True,
            "numeric_stop_counts_as_completed": passed["numeric_stop_counts_as_completed_not_converged"]}


def _known_e_test() -> dict:
    if not KNOWN_E_MANIFEST.is_file():
        raise FileNotFoundError(KNOWN_E_MANIFEST)
    manifest_sha = controller.sha256_file(KNOWN_E_MANIFEST)
    entries = {}
    for fixture, path in KNOWN_E_PATHS.items():
        _assert(controller.sha256_file(path) == EXPECTED_KNOWN_SHA[fixture], "known-e hash changed for %s" % fixture)
        values, metadata = controller.load_known_exposure(path, fixture, 2645)
        entries[fixture] = {"path": str(path), "sha256": controller.sha256_file(path),
                            "shape": list(values.shape), "dtype": str(values.dtype),
                            "mean": float(values.mean()), "min": float(values.min()),
                            "max": float(values.max()), "metadata": metadata}
    return {"status": "passed", "worker_manifest_path": str(KNOWN_E_MANIFEST),
            "worker_manifest_sha256": manifest_sha, "vectors": entries,
            "truth_opened": False}


def main() -> int:
    started = time.perf_counter()
    result = {
        "schema": "gpu-m1-preconditioner-preflight-v1", "status": "failed",
        "created_at_utc": controller._utc_now(), "artifact": str(ARTIFACT),
        "training_boundary": {"phase_used": False, "reference_used": False,
                              "synthetic_truth_opened": False, "real_reference_opened": False,
                              "evaluation_r2_opened": False, "native_fdg_started": False,
                              "formal_p9016_fit_started": False},
    }
    try:
        result["runtime"] = _probe()
        result["budget_reference"] = controller._load_036_reference()
        result["tests"] = {
            "transform": _transform_tests(),
            "fixed_state": _real_fixed_state_tests(),
            "finite_difference": _finite_difference_test(),
            "budget": _budget_test(),
            "status_mapping": _status_mapping_test(),
            "completion_gate": _completion_gate_test(),
            "known_e": _known_e_test(),
        }
        result["status"] = "passed"
        result["elapsed_seconds"] = float(time.perf_counter() - started)
        result["optimizer_calls_in_preparation"] = int(result["tests"]["budget"]["optimizer_fg_calls"])
        result["formal_training_started"] = False
        result["source_code_sha256"] = controller._source_hashes()
        digest = controller.write_json(ARTIFACT / "preflight.json", result)
        print(json.dumps({"status": "passed", "path": str(ARTIFACT / "preflight.json"),
                          "sha256": digest, "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))
        return 0
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = __import__("traceback").format_exc(limit=20)
        result["elapsed_seconds"] = float(time.perf_counter() - started)
        controller.write_json(ARTIFACT / "preflight_failed.json", result, exclusive=False)
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
