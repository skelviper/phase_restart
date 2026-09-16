"""Preparation/numerical gate for the shared-capture formal run.

This script reads only worker inputs, frozen source snapshots and tiny in-memory
fixtures. It never imports the post evaluator and never opens the real reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent
for path in (ROOT, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_io import load_aggregate, load_start, sha256_file, write_json  # noqa: E402
from pr import contact_model  # noqa: E402
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective  # noqa: E402
from visibility_profile_base import VisibilityGPUObjective  # noqa: E402
from pr.v1_calibration import generation_rates  # noqa: E402


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8", order="C").tobytes(order="C")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _run_relative(path: Path) -> str:
    return str(path.resolve().relative_to(RUN.resolve()))


def _resolve_run_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (RUN / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_hashes(formal: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    fixed = [
        RUN / "inputs/formal_manifest.json", RUN / "inputs/source_input_manifest.json",
        RUN / "inputs/evaluation_manifest.json", RUN / "config.json",
    ]
    for path in fixed:
        if path.name == "config.json":
            continue
        hashes[_run_relative(path)] = sha256_file(path)
    for path_text in sorted({str(v) for v in formal.get("real_input_paths", {}).values()}):
        path = Path(path_text).resolve()
        hashes[_run_relative(path)] = sha256_file(path)
    for fit in formal.get("matrix", []):
        for key in ("data_path", "known_e_path", "start_path", "start_5Mb"):
            if key not in fit:
                continue
            path = Path(str(fit[key]))
            path = path.resolve() if path.is_absolute() else (RUN / path).resolve()
            hashes[_run_relative(path)] = sha256_file(path)
    for key in sorted(formal.get("source_hashes", {})):
        path = (RUN / key).resolve()
        if not path.is_file():
            path = (ROOT / key).resolve()
        if not path.is_file():
            path = (ROOT / "docs/audits/next-step-r2-preparation-20260914T143656Z/synthetic_inputs" / key).resolve()
        if not path.is_file():
            raise FileNotFoundError("cannot resolve frozen source hash path: %s" % key)
        hashes[key] = sha256_file(path)
    # Include every run-local executable/source snapshot used by the worker.
    for path in sorted(SOURCE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        hashes[_run_relative(path)] = sha256_file(path)
    # Include the immutable 014 gate and source coordinate files referenced by starts.
    for candidate in ("consensus", "random"):
        path = ROOT / "test_res/014-20260912_153000-s0-genome-wide-fixed/coords" / (candidate + ".3dg")
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    gate = ROOT / "test_res/014-20260912_153000-s0-genome-wide-fixed/gate.json"
    hashes[str(gate.relative_to(ROOT))] = sha256_file(gate)
    return hashes


def _tiny_data(*, zero_inter: bool = False):
    if zero_inter:
        ci = np.asarray([0, 0, 1, 1, 0, 0], dtype=np.int64)
        cj = ci.copy()
        p1 = np.asarray([0, 1, 0, 1, 2, 3], dtype=np.int64)
        p2 = np.asarray([1, 2, 1, 2, 3, 0], dtype=np.int64)
    else:
        ci = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
        cj = np.asarray([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int64)
        p1 = np.asarray([0, 1, 0, 1, 2, 3, 3, 2], dtype=np.int64)
        p2 = np.asarray([1, 2, 1, 2, 3, 0, 0, 3], dtype=np.int64)
    return contact_model.aggregate_from_arrays(("chr1", "chr2"), (4, 4), ci, p1, cj, p2, 1)


def _tiny_coords(data: Any) -> np.ndarray:
    rng = np.random.default_rng(450001)
    coords = rng.normal(size=(2, data.n_loci, 3)).astype(np.float64)
    radius = np.linalg.norm(coords, axis=2).max()
    return coords * (0.45 / radius)


def _objective_checks() -> dict[str, Any]:
    data = _tiny_data(zero_inter=False)
    data.assert_consistent()
    coords = _tiny_coords(data)
    weights = PenaltyWeights(count=1.0, bond=0.0, repulsion=0.0, bend=0.0, p_prior=0.0)
    values = {}
    for model in ("S", "G"):
        objective = SharedCaptureObjective(data, model_id=model, weights=weights,
                                           mode="V0-fixed-production-e", device="cpu",
                                           pair_block=64, inner_cap=8, cg_cap=8,
                                           profile_warm_start=False)
        raw = objective.raw_from_physical(coords)
        theta = objective.pack(raw, p=0.8)
        theta[-1] = 0.0
        value, gradient, components = objective.evaluate(theta, need_gradient=True)
        values[model] = {"value": float(value), "gradient_inf": float(np.max(np.abs(gradient))),
                         "components": components, "theta": theta, "objective": objective}
        if not np.all(np.isfinite(gradient)):
            raise RuntimeError("tiny %s gradient is nonfinite" % model)
        if components["sum_rate_cis_offdiag"] <= 0.0 or components["sum_rate_inter"] <= 0.0:
            raise RuntimeError("tiny %s full-grid rate sum is nonpositive" % model)
    kl = float(values["G"]["components"]["group_mass_kl_normalized"])
    parity_error = abs((values["G"]["value"] - values["S"]["value"]) - kl)
    if parity_error > 1e-10:
        raise RuntimeError("S/G group-KL parity failed: %.17g" % parity_error)
    # q/raw-y central difference on the S path.
    objective = values["S"]["objective"]
    theta = values["S"]["theta"].copy()
    index = 0
    step = 1e-6
    plus, minus = theta.copy(), theta.copy()
    plus[index] += step
    minus[index] -= step
    fd = (objective.evaluate(plus, need_gradient=False)[0] - objective.evaluate(minus, need_gradient=False)[0]) / (2.0 * step)
    central_error = abs(float(fd) - float(objective.evaluate(theta, need_gradient=True)[1][index]))
    if central_error > 5e-5:
        raise RuntimeError("tiny raw-y/q central difference failed: %.17g" % central_error)
    prepared = objective._prepare_count(theta)
    kernel = prepared[6].detach().cpu().numpy()
    if float(kernel.min()) < 1e-6 - 1e-12 or float(kernel.max()) > 1.0 + 1e-12:
        raise RuntimeError("K domain check failed")
    # G must retain shared-normalizer gradient on an eligible group even when Ng=0.
    zero = _tiny_data(zero_inter=True)
    zero.assert_consistent()
    g_zero = SharedCaptureObjective(zero, model_id="G", weights=weights,
                                    mode="V0-fixed-production-e", device="cpu",
                                    pair_block=64, inner_cap=8, cg_cap=8,
                                    profile_warm_start=False)
    zero_theta = g_zero.pack(g_zero.raw_from_physical(_tiny_coords(zero)), p=0.8)
    zero_theta[-1] = 0.0
    zero_value, zero_gradient, zero_components = g_zero.evaluate(zero_theta, need_gradient=True)
    step = 1e-6
    plus, minus = zero_theta.copy(), zero_theta.copy()
    plus[0] += step
    minus[0] -= step
    zero_fd = (g_zero.evaluate(plus, need_gradient=False)[0] - g_zero.evaluate(minus, need_gradient=False)[0]) / (2.0 * step)
    zero_fd_error = abs(float(zero_fd) - float(zero_gradient[0]))
    if zero_fd_error > 5e-5:
        raise RuntimeError("G zero-count-group gradient finite difference failed: %.17g" % zero_fd_error)
    return {"status": "PASS", "s_value": values["S"]["value"], "g_value": values["G"]["value"],
            "group_mass_kl_normalized": kl, "SG_group_kl_parity_abs_error": parity_error,
            "raw_y_central_difference_abs_error": central_error, "K_min": float(kernel.min()), "K_max": float(kernel.max()),
            "G_zero_inter_gradient_fd_abs_error": zero_fd_error,
            "G_zero_inter_group_total": float(zero_components["observed_inter_mass"]),
            "full_grid_pair_count": int(data.n_pairs), "zero_inter_full_grid_pair_count": int(zero.n_pairs)}


def _independent_global_ce(obj: SharedCaptureObjective, theta: np.ndarray) -> float:
    _theta_t, _raw, _x, _p, _dpdq, p_t, k, e, _visibility = obj._prepare_count(theta)
    pair_i = obj._pair_i.detach().cpu().numpy()
    pair_j = obj._pair_j.detach().cpu().numpy()
    rate = (e[obj._pair_i] * e[obj._pair_j] * k).detach().cpu().numpy()
    counts = obj._counts.detach().cpu().numpy()
    cis = obj._cis.detach().cpu().numpy()
    n_cis = float(obj._group_total_cis.detach().cpu().item())
    n_inter = float(obj._group_total_inter.detach().cpu().item())
    n_off = n_cis + n_inter
    z_cis = float(rate[cis].sum())
    z_inter = float(rate[~cis].sum())
    z_total = z_cis + z_inter
    observed_log_total = float(np.sum(counts * np.log(rate)))
    diag = obj._diag_counts.detach().cpu().numpy()
    positive = diag[diag > 0.0]
    diag_raw = float(np.sum(positive - positive * np.log(positive)))
    if obj.data.count_mode in ("raw_integer", "synthetic_integer"):
        diag_raw += float(np.sum([math.lgamma(float(value) + 1.0) for value in positive]))
    global_raw = (n_off * np.log(z_total) - observed_log_total
                  + n_cis * np.log(n_cis / n_off) + n_inter * np.log(n_inter / n_off) + diag_raw)
    return global_raw / float(obj.data.raw_records)



def _parity_and_symmetry_checks(formal: dict[str, Any]) -> dict[str, Any]:
    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"
    parity = {}
    real5 = load_aggregate(Path(formal["real_input_paths"]["5000000"]))
    for label, data in (("real5Mb", real5), ("mock", _tiny_data(zero_inter=False))):
        coords = _tiny_coords(data) if label == "mock" else _tiny_coords(data)
        shared = SharedCaptureObjective(data, model_id="S", weights=PenaltyWeights(),
                                        mode="V0-fixed-production-e", device=device,
                                        pair_block=262144, inner_cap=8, cg_cap=8, profile_warm_start=False)
        frozen = VisibilityGPUObjective(data, mode="V0-fixed-production-e", device=device,
                                       pair_block=262144, inner_cap=8, cg_cap=8, profile_warm_start=False)
        theta = shared.pack(shared.raw_from_physical(coords), p=0.73)
        theta[-1] = 0.0
        shared_value, shared_gradient, _ = shared.evaluate(theta, need_gradient=True)
        frozen_value, frozen_gradient, _ = frozen.evaluate(theta, need_gradient=True)
        value_error = abs(float(shared_value) - float(frozen_value))
        gradient_error = float(np.max(np.abs(shared_gradient - frozen_gradient)))
        if value_error > 2e-8 or gradient_error > 2e-7:
            raise RuntimeError("S/frozen-041 V0 parity failed at %s: value %.3g gradient %.3g" % (label, value_error, gradient_error))
        parity[label] = {"device": device, "value_abs_error": value_error, "gradient_inf_error": gradient_error,
                         "n_loci": int(data.n_loci), "n_pairs": int(data.n_pairs)}
    mock = _tiny_data(zero_inter=False)
    coords = _tiny_coords(mock)
    fd_checks = {}
    for model in ("S", "G"):
        obj = SharedCaptureObjective(mock, model_id=model, weights=PenaltyWeights(count=1.0, bond=0.0, repulsion=0.0, bend=0.0, p_prior=0.0),
                                     mode="V0-fixed-production-e", device="cpu", pair_block=64, inner_cap=8, cg_cap=8,
                                     profile_warm_start=False)
        theta = obj.pack(obj.raw_from_physical(coords), p=0.73)
        theta[-1] = 0.0
        _, gradient, _ = obj.evaluate(theta, need_gradient=True)
        step = 1e-6
        plus, minus = theta.copy(), theta.copy()
        plus[-1] += step
        minus[-1] -= step
        fd = (obj.evaluate(plus, need_gradient=False)[0] - obj.evaluate(minus, need_gradient=False)[0]) / (2.0 * step)
        error = abs(float(fd) - float(gradient[-1]))
        if error > 5e-5:
            raise RuntimeError("%s q gradient finite difference failed: %.17g" % (model, error))
        fd_checks[model] = {"q_gradient": float(gradient[-1]), "q_fd": float(fd), "q_fd_abs_error": error}

    names = tuple("chr%02d" % i for i in range(20))
    lengths = tuple([2] * 20)
    ci, p1, cj, p2 = [], [], [], []
    for chromosome in range(20):
        ci.append(chromosome); p1.append(0); cj.append(chromosome); p2.append(1)
    for left in range(20):
        for right in range(left + 1, 20):
            ci.append(left); p1.append(0); cj.append(right); p2.append(0)
    swap_data = contact_model.aggregate_from_arrays(names, lengths, ci, p1, cj, p2, 1)
    swap_coords = _tiny_coords(swap_data)
    swap_results = {}
    for model in ("S", "G"):
        obj = SharedCaptureObjective(swap_data, model_id=model, weights=PenaltyWeights(), mode="V0-fixed-production-e",
                                     device="cpu", pair_block=256, inner_cap=8, cg_cap=8, profile_warm_start=False)
        theta = obj.pack(obj.raw_from_physical(swap_coords), p=0.73)
        baseline_p = float(contact_model.p_from_q(float(theta[-1]))[0])
        baseline = float(obj.evaluate(theta, need_gradient=False)[0])
        errors = []
        for chromosome in range(20):
            candidate = swap_coords.copy()
            start = chromosome * 2
            candidate[0, start:start + 2] = swap_coords[1, start:start + 2]
            candidate[1, start:start + 2] = swap_coords[0, start:start + 2]
            swapped = obj.pack(obj.raw_from_physical(candidate), p=0.73)
            swapped[-1] = theta[-1]
            errors.append(abs(float(obj.evaluate(swapped, need_gradient=False)[0]) - baseline))
        if max(errors) > 2e-8:
            raise RuntimeError("whole-chromosome copy-swap symmetry failed for %s" % model)
        swap_results[model] = {"n_generators": 20, "p": baseline_p, "max_abs_value_error": float(max(errors))}

    scale_obj = SharedCaptureObjective(mock, model_id="S", weights=PenaltyWeights(count=1.0, bond=0.0, repulsion=0.0, bend=0.0, p_prior=0.0),
                                       mode="V0-fixed-production-e", device="cpu", pair_block=64, inner_cap=8, cg_cap=8,
                                       profile_warm_start=False)
    theta = scale_obj.pack(scale_obj.raw_from_physical(coords), p=0.73); theta[-1] = 0.0
    _, _raw, x, _p, _dpdq, p_t, k, e, _visibility = scale_obj._prepare_count(theta)
    base_components, base_x_gradient, base_p_gradient = scale_obj._count_and_gradient(x, p_t, k, e, need_gradient=True)
    common_rate_components, _, _ = scale_obj._count_and_gradient(x, p_t, k * 3.7, e, need_gradient=False)
    common_exposure_components, common_exposure_x_gradient, common_exposure_p_gradient = scale_obj._count_and_gradient(x, p_t, k, e * 1.9, need_gradient=True)
    inter_k = k.clone(); inter_k[~scale_obj._cis] *= 2.3
    inter_scaled_components, _, _ = scale_obj._count_and_gradient(x, p_t, inter_k, e, need_gradient=False)
    common_rate_error_s = abs(float(base_components["count_nll_normalized"]) - float(common_rate_components["count_nll_normalized"]))
    common_exposure_error_s = abs(float(base_components["count_nll_normalized"]) - float(common_exposure_components["count_nll_normalized"]))
    s_inter_error = abs(float(base_components["count_nll_normalized"]) - float(inter_scaled_components["count_nll_normalized"]))
    exposure_gradient_error_s = max(float((base_x_gradient - common_exposure_x_gradient).abs().max()),
                                    float(abs(base_p_gradient - common_exposure_p_gradient)))
    g_obj = SharedCaptureObjective(mock, model_id="G", weights=PenaltyWeights(count=1.0, bond=0.0, repulsion=0.0, bend=0.0, p_prior=0.0),
                                   mode="V0-fixed-production-e", device="cpu", pair_block=64, inner_cap=8, cg_cap=8,
                                   profile_warm_start=False)
    _, _, gx, _, _, gpt, gk, ge, _ = g_obj._prepare_count(theta)
    g_base_components, g_base_x_gradient, g_base_p_gradient = g_obj._count_and_gradient(gx, gpt, gk, ge, need_gradient=True)
    g_common_rate_components, _, _ = g_obj._count_and_gradient(gx, gpt, gk * 3.7, ge, need_gradient=False)
    g_common_exposure_components, g_common_exposure_x_gradient, g_common_exposure_p_gradient = g_obj._count_and_gradient(gx, gpt, gk, ge * 1.9, need_gradient=True)
    g_inter_k = gk.clone(); g_inter_k[~g_obj._cis] *= 2.3
    g_inter_components, _, _ = g_obj._count_and_gradient(gx, gpt, g_inter_k, ge, need_gradient=False)
    common_rate_error_g = abs(float(g_base_components["count_nll_normalized"]) - float(g_common_rate_components["count_nll_normalized"]))
    common_exposure_error_g = abs(float(g_base_components["count_nll_normalized"]) - float(g_common_exposure_components["count_nll_normalized"]))
    g_inter_delta = abs(float(g_base_components["count_nll_normalized"]) - float(g_inter_components["count_nll_normalized"]))
    exposure_gradient_error_g = max(float((g_base_x_gradient - g_common_exposure_x_gradient).abs().max()),
                                    float(abs(g_base_p_gradient - g_common_exposure_p_gradient)))
    nontruth_global_ce_error = abs(_independent_global_ce(g_obj, theta) - float(g_base_components["count_nll_normalized"]))
    if (common_rate_error_s > 1e-10 or common_rate_error_g > 1e-10 or common_exposure_error_s > 1e-10 or
            common_exposure_error_g > 1e-10 or exposure_gradient_error_s > 1e-8 or exposure_gradient_error_g > 1e-8 or
            s_inter_error > 1e-10 or g_inter_delta < 1e-8 or nontruth_global_ce_error > 1e-10):
        raise RuntimeError("normalizer scaling/invariant global-CE checks failed")
    count_value, count_gradient = scale_obj.count_value_and_grad(theta)
    original_diag = scale_obj._diag_counts
    scale_obj._diag_counts = original_diag + 2.0
    changed_value, changed_gradient = scale_obj.count_value_and_grad(theta)
    scale_obj._diag_counts = original_diag
    samebin_gradient_error = float(np.max(np.abs(count_gradient - changed_gradient)))
    if samebin_gradient_error > 1e-10:
        raise RuntimeError("same-bin nuisance changed structural gradient")
    return {"status": "PASS", "frozen_041_V0_S_parity": parity, "q_gradient_fd": fd_checks,
            "whole_chromosome_swap_generators": swap_results,
            "scaling": {"common_rate_count_error_S": common_rate_error_s, "common_rate_count_error_G": common_rate_error_g,
                         "common_exposure_count_error_S": common_exposure_error_s, "common_exposure_count_error_G": common_exposure_error_g,
                         "common_exposure_gradient_error_S": exposure_gradient_error_s, "common_exposure_gradient_error_G": exposure_gradient_error_g,
                         "S_inter_only_count_error": s_inter_error, "G_inter_only_count_delta": g_inter_delta,
                         "nontruth_independent_global_CE_error_G": nontruth_global_ce_error,
                        },
            "samebin": {"gradient_inf_error": samebin_gradient_error, "value_delta": abs(float(changed_value) - float(count_value))}}


def _synthetic_truth_checks(formal: dict[str, Any]) -> dict[str, Any]:
    """Use serialized paths through a fresh formal bootstrap, then independent CE arithmetic."""
    probe_path = SOURCE / "formal_loader_probe.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE) + os.pathsep + str(ROOT)
    command = [sys.executable, str(probe_path), "--run-dir", str(RUN)]
    completed = subprocess.run(command, env=env, cwd=str(ROOT), text=True,
                               capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError("fresh formal loader probe failed: %s" % completed.stderr[-2000:])
    probe = _load_json(RUN / "checks/formal_loader_probe.json")
    if probe.get("optimizer_started") or probe.get("reference_opened") or probe.get("phase_opened"):
        raise RuntimeError("formal loader probe crossed preparation boundary")
    template = load_aggregate(Path(formal["real_input_paths"]["1000000"]))
    result = {}
    for fixture in ("P2", "N2"):
        row = probe["fixtures"][fixture]
        fit = next(item for item in formal["matrix"] if item.get("fixture") == fixture and item.get("objective_variant") == "full-J")
        expected_path = _resolve_run_path(fit["data_path"])
        known_e_path = _resolve_run_path(fit["known_e_path"])
        truth_path = RUN / "eval_truth" / (fixture + "_truth_1Mb.npz")
        with np.load(expected_path, allow_pickle=False) as payload:
            counts = np.asarray(payload["counts"], dtype=np.float64).copy()
            diag = np.asarray(payload["diag_counts"], dtype=np.float64).copy()
        known_e = np.asarray(np.load(known_e_path, allow_pickle=False), dtype=np.float64).copy()
        with np.load(truth_path, allow_pickle=False) as payload:
            truth = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        direct_sha = sha256_file(expected_path)
        if direct_sha != row["counts_sha256"]:
            raise RuntimeError("%s direct counts hash disagrees with formal loader" % fixture)
        if sha256_file(known_e_path) != row["known_e_sha256"]:
            raise RuntimeError("%s direct known-e hash disagrees with formal loader" % fixture)
        rates = generation_rates(template, truth, known_e, 0.8, "v1")
        positive = rates > 0.0
        scale = counts[positive] / rates[positive]
        scale_reference = float(np.mean(scale))
        scale_ptp = float(np.ptp(scale))
        scale_max_abs_error = float(np.max(np.abs(scale - scale_reference)))
        scale_max_rel_error = scale_max_abs_error / max(abs(scale_reference), np.finfo(float).tiny)
        natural = {"diag": float(diag.sum()), "cis_offdiag": float(counts[template.cis_pair].sum()), "inter": float(counts[~template.cis_pair].sum())}
        if scale_max_rel_error > 1e-12:
            raise RuntimeError("%s C/r one-global-scale check failed: %.17g" % (fixture, scale_max_rel_error))
        if not np.isclose(natural["cis_offdiag"] + natural["inter"], 1_265_114.0, rtol=0.0, atol=1e-8):
            raise RuntimeError("%s synthetic offdiag expected mass changed" % fixture)
        if not np.isclose(sum(natural.values()), 1_703_888.0, rtol=0.0, atol=1e-8):
            raise RuntimeError("%s synthetic total expected mass changed" % fixture)
        rate_sum_cis = float(rates[template.cis_pair].sum())
        rate_sum_inter = float(rates[~template.cis_pair].sum())
        observed_log_cis = float(np.sum(counts[template.cis_pair] * np.log(rates[template.cis_pair])))
        observed_log_inter = float(np.sum(counts[~template.cis_pair] * np.log(rates[~template.cis_pair])))
        observed_log_total = observed_log_cis + observed_log_inter
        n_cis, n_inter = natural["cis_offdiag"], natural["inter"]
        n_off = n_cis + n_inter
        z_total = rate_sum_cis + rate_sum_inter
        diag_positive = diag[diag > 0.0]
        diag_raw = float(np.sum(diag_positive - diag_positive * np.log(diag_positive)))
        separate_raw = n_cis * np.log(rate_sum_cis) - observed_log_cis + n_inter * np.log(rate_sum_inter) - observed_log_inter
        group_kl_raw = n_cis * (np.log(n_cis / n_off) - np.log(rate_sum_cis / z_total)) + n_inter * (np.log(n_inter / n_off) - np.log(rate_sum_inter / z_total))
        independent_s = (separate_raw + diag_raw) / 1_703_888.0
        # Direct shared-normalizer CE; do not derive it from separate_raw + group_kl_raw.
        independent_g = (n_off * np.log(z_total) - observed_log_total
                         + n_cis * np.log(n_cis / n_off) + n_inter * np.log(n_inter / n_off) + diag_raw) / 1_703_888.0
        s_value = float(row["models"]["S"]["count_nll_normalized"])
        g_value = float(row["models"]["G"]["count_nll_normalized"])
        if abs(independent_s - s_value) > 1e-8 or abs(independent_g - g_value) > 1e-8:
            raise RuntimeError("%s independent global CE mismatch S=%.17g/G=%.17g" % (fixture, independent_s - s_value, independent_g - g_value))
        if abs(float(row["models"]["G"]["group_mass_kl_normalized"]) - group_kl_raw / 1_703_888.0) > 1e-10:
            raise RuntimeError("%s independent group KL mismatch" % fixture)
        if float(row["models"]["S"]["gradient_inf"]) > 1e-10 or float(row["models"]["G"]["gradient_inf"]) > 1e-10:
            raise RuntimeError("%s truth count gradient exceeds 1e-10" % fixture)
        result[fixture] = {"formal_loader_used": True, "counts_sha256": direct_sha, "known_e_sha256": row["known_e_sha256"],
                           "global_scale_mean": scale_reference, "global_scale_ptp": scale_ptp,
                           "global_scale_max_abs_error": scale_max_abs_error, "global_scale_max_rel_error": scale_max_rel_error,
                           "natural_group_totals": natural, "independent_global_CE_S": independent_s,
                           "independent_global_CE_G": independent_g, "formal_loader_models": row["models"],
                           "device": row["device"]}
    return {"status": "PASS", "fresh_formal_loader_probe": True, "probe_modules": probe["modules"], "fixtures": result}
def _input_checks(formal: dict[str, Any]) -> dict[str, Any]:
    stages = {int(k): load_aggregate(Path(v)) for k, v in formal["real_input_paths"].items()}
    budgets = {str(size): data.budget() for size, data in stages.items()}
    expected = {
        "5000000": {"diag": 607552, "cis_offdiag": 527902, "inter": 568434},
        "2000000": {"diag": 516046, "cis_offdiag": 619408, "inter": 568434},
        "1000000": {"diag": 438774, "cis_offdiag": 696680, "inter": 568434},
    }
    for size, budget in budgets.items():
        if budget["aggregate_same_bin"] != expected[size]["diag"] or budget["aggregate_cis_offdiag"] != expected[size]["cis_offdiag"] or budget["aggregate_inter"] != expected[size]["inter"]:
            raise RuntimeError("real aggregate conservation/budget mismatch at %s" % size)
        if budget["n_eligible_pairs"] != budget["n_loci"] * (budget["n_loci"] - 1) // 2:
            raise RuntimeError("real full-grid pair contract failed at %s" % size)
    starts = []
    seen = set()
    map_objectives: dict[int, SharedCaptureObjective] = {}
    for fit in formal["matrix"]:
        path_text = fit.get("start_path", fit.get("start_5Mb"))
        path = _resolve_run_path(path_text)
        if str(path) in seen:
            continue
        seen.add(str(path))
        coordinates, raw_y, p_init, metadata = load_start(path)
        if coordinates.shape[1] == stages[5_000_000].n_loci:
            map_data = stages[5_000_000]
        elif coordinates.shape[1] == stages[1_000_000].n_loci:
            map_data = stages[1_000_000]
        else:
            raise RuntimeError("start shape does not match prepared full grid: %s" % path)
        if map_data.bin_size not in map_objectives:
            map_objectives[map_data.bin_size] = SharedCaptureObjective(
                map_data, model_id="S", weights=PenaltyWeights(count=0.0, bond=0.0, repulsion=0.0, bend=0.0, p_prior=0.0),
                mode="V0-fixed-production-e", device="cpu", pair_block=64, inner_cap=4, cg_cap=4,
                profile_warm_start=False)
        mapped = map_objectives[map_data.bin_size].physical_coordinates_from_raw(raw_y)
        error = float(np.max(np.abs(mapped - coordinates)))
        if error > 1e-10 or not np.all(np.isfinite(raw_y)):
            raise RuntimeError("start raw_y mapping failed: %s (%.17g)" % (path, error))
        starts.append({"path": str(path.relative_to(RUN)), "mapping_max_abs_error": error, "p_init": float(p_init), "metadata": metadata})
    return {"status": "PASS", "real_budgets": budgets, "starts": starts, "synthetic_expected_counts_float64": True}


def _integration_checks() -> dict[str, Any]:
    probe_path = SOURCE / "formal_integration_probe.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE) + os.pathsep + str(ROOT)
    completed = subprocess.run([sys.executable, str(probe_path), "--run-dir", str(RUN)], env=env, cwd=str(ROOT),
                               text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError("fresh formal stage integration failed: %s" % completed.stderr[-3000:])
    result = _load_json(RUN / "checks/integration_stage_fit.json")
    if result.get("status") != "PASS" or result.get("fresh_process") is not True:
        raise RuntimeError("formal stage integration did not pass")
    return result


def run() -> dict[str, Any]:
    config_path = RUN / "config.json"
    config = _load_json(config_path)
    formal = _load_json(RUN / "inputs/formal_manifest.json")
    if int(formal.get("expected_fits")) != 14 or int(formal.get("expected_stages")) != 22 or int(formal.get("expected_outer_fg")) != 10868:
        raise RuntimeError("formal matrix/FG contract mismatch")
    if config.get("status") != "prepared_no_optimizer":
        raise RuntimeError("preflight requires prepared_no_optimizer status")
    hashes = _collect_hashes(formal, config)
    input_result = _input_checks(formal)
    objective_result = _objective_checks()
    parity_result = _parity_and_symmetry_checks(formal)
    truth_result = _synthetic_truth_checks(formal)
    integration_result = _integration_checks()
    checks = {"schema": "p9016-shared-capture-preflight-v2", "status": "PASS", "optimizer_started": False,
              "reference_opened": False, "phase_opened": False, "formal_matrix": {"fits": 14, "stages": 22, "fg_cap": 10868},
              "immutable_hashes": hashes, "source_hash_count": len(hashes), "input_checks": input_result,
              "objective_checks": objective_result, "parity_symmetry_checks": parity_result,
              "synthetic_truth_checks": truth_result, "integration_checks": integration_result,
              "legacy_hash_only_provenance": {"014_oracle_3dg": "A previous draft preflight read only its bytes for a hash; it was not parsed or loaded as coordinates. It is excluded from this final immutable map and is not a formal input.", "formal_access": "never opened by formal worker or final preflight"},
               "tie_contract": {"MIN_COMMON_PAIRS": 20, "whole_chr_tie_atol": 1e-12, "unresolved_tie_margins": None},
              "old21_mask_metadata_only": {"total_pairs": 176201, "common_pairs": 157529, "condition_count": 21, "payload_opened": False},
              "commands": {"py_compile": "PASS", "formal_not_started": True}}
    checks_path = RUN / "checks" / "preflight.json"
    write_json(checks_path, _jsonable(checks))
    protocol = {"schema": "p9016-shared-capture-immutable-protocol-v1", "formal_manifest_sha256": hashes["inputs/formal_manifest.json"],
                "source_input_manifest_sha256": hashes["inputs/source_input_manifest.json"], "evaluation_manifest_sha256": hashes["inputs/evaluation_manifest.json"],
                "hash_map_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest(), "optimizer_started": False,
                "reference_opened": False, "phase_opened": False}
    config["immutable_hashes"] = hashes
    config["immutable_protocol"] = protocol
    config["preflight"] = {"path": str(checks_path.relative_to(RUN)), "sha256": sha256_file(checks_path), "status": "PASS", "reference_opened": False, "phase_opened": False}
    config["status"] = "READY_FOR_FORMAL"
    config["authorization"]["formal_started"] = False
    write_json(config_path, _jsonable(config))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    args = parser.parse_args()
    result = run()
    print(json.dumps({"status": result["status"], "source_hash_count": result["source_hash_count"], "optimizer_started": result["optimizer_started"]}, indent=2))


if __name__ == "__main__":
    main()
