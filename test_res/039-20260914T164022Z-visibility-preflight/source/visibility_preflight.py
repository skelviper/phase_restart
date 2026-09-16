"""039 技术可见性预检与真实三层短 benchmark。

本脚本只打开 SNP-free counts、014 blind consensus 起点和 manifest 明确允许的
synthetic counts/known-e 向量。它不启动 formal，也不读取 reference、phase、038
或任何 evaluation/R2 结果。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
for _path in (SOURCE, SOURCE / "frozen_035", SOURCE / "frozen_pr", SOURCE / "frozen_037"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from frozen_037.m1_preconditioner import run_budgeted_lbfgs  # noqa: E402
from gpu_variant_backend import GPUVariantObjective  # noqa: E402
from pr import contact_model, reconstruction_init  # noqa: E402
from visibility_profile import (  # noqa: E402
    PROFILE_RESIDUAL_TOL,
    VisibilityGPUObjective,
    _support_audit,
)

INPUT_PATH = (ROOT / "inputs" / "P9016.snpfree.pairs.gz").resolve()
MANIFEST_PATH = (ROOT / "docs/audits/next-step-r2-preparation-20260914T143656Z/synthetic_inputs/known_e_worker_input_manifest.json").resolve()
KNOWN_E_ROOT = MANIFEST_PATH.parent
APPROVED_014_ROOT = (ROOT / "test_res/014-20260912_153000-s0-genome-wide-fixed").resolve()
EXPECTED_INPUT_SHA = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"
REAL_EXPECTED = {
    5_000_000: {"n_loci": 538, "raw_same_bin": 607_552, "raw_cis_offdiag": 527_902, "raw_inter": 568_434},
    2_000_000: {"n_loci": 1329, "raw_same_bin": 516_046, "raw_cis_offdiag": 619_408, "raw_inter": 568_434},
    1_000_000: {"n_loci": 2645, "raw_same_bin": 438_774, "raw_cis_offdiag": 696_680, "raw_inter": 568_434},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("nonfinite value in preflight JSON")
        return float(value)
    return value


# 避免在类型检查之外引入一个庞大的通用框架；这里只需 Mapping 的运行时识别。
from collections.abc import Mapping  # noqa: E402, C0413


def write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
    return hashlib.sha256(payload).hexdigest()


def configure_014_paths() -> None:
    """把快照 reconstruction_init 的 root/path 指向绝对的014 blind文件。"""
    reconstruction_init.ROOT = str(ROOT)
    gate = APPROVED_014_ROOT / "gate.json"
    coords = APPROVED_014_ROOT / "coords"
    reconstruction_init.DEFAULT_GATE_PATH = str(gate)
    reconstruction_init.DEFAULT_COORD_DIR = str(coords)
    for candidate, spec in list(reconstruction_init.APPROVED_SOURCES.items()):
        patched = dict(spec)
        patched["path"] = str(coords / (candidate + ".3dg"))
        patched["gate_path"] = str(gate)
        reconstruction_init.APPROVED_SOURCES[candidate] = patched


def make_mock_data(zero_degree: bool = False) -> contact_model.AggregatedContacts:
    """构造不含任何真实标签的三染色体小 full-grid counts。"""
    names = ("chr1", "chr2", "chr3")
    lengths = (4, 4, 4) if zero_degree else (3, 3, 3)
    n_loci = int(sum(lengths))
    zero = {0} if zero_degree else set()
    ci: list[int] = []
    p1: list[int] = []
    cj: list[int] = []
    p2: list[int] = []
    offsets = np.cumsum((0,) + lengths[:-1])
    for first in range(n_loci):
        if first in zero:
            continue
        first_chr = int(np.searchsorted(offsets[1:], first, side="right"))
        first_bin = first - int(offsets[first_chr])
        for second in range(first + 1, n_loci):
            if second in zero:
                continue
            second_chr = int(np.searchsorted(offsets[1:], second, side="right"))
            second_bin = second - int(offsets[second_chr])
            ci.append(first_chr); p1.append(first_bin)
            cj.append(second_chr); p2.append(second_bin)
    for locus in range(n_loci):
        chromosome = int(np.searchsorted(offsets[1:], locus, side="right"))
        local_bin = locus - int(offsets[chromosome])
        ci.append(chromosome); p1.append(local_bin)
        cj.append(chromosome); p2.append(local_bin)
    return contact_model.aggregate_from_arrays(names, lengths, ci, p1, cj, p2, 1)


def mock_coordinates(n_loci: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coordinates = rng.normal(scale=0.08, size=(2, n_loci, 3))
    coordinates -= coordinates.mean(axis=(0, 1), keepdims=True)
    coordinates *= 0.25 / np.linalg.norm(coordinates, axis=2).max()
    return coordinates.astype(np.float64)


def mock_expected_data(template: contact_model.AggregatedContacts, x: np.ndarray,
                       p: float, e_true: np.ndarray) -> contact_model.AggregatedContacts:
    """用固定 mock geometry 生成 expected counts，只验证数学恢复，不打开 truth。"""
    pair_i = template.pair_i
    pair_j = template.pair_j
    kaa = contact_model.bounded_kernel(x[0, pair_i] - x[0, pair_j], template.r0)
    kab = contact_model.bounded_kernel(x[0, pair_i] - x[1, pair_j], template.r0)
    kba = contact_model.bounded_kernel(x[1, pair_i] - x[0, pair_j], template.r0)
    kbb = contact_model.bounded_kernel(x[1, pair_i] - x[1, pair_j], template.r0)
    cis = template.cis_pair
    k = np.where(cis, 0.5 * (p * (kaa + kbb) + (1.0 - p) * (kab + kba)),
                 0.25 * (kaa + kab + kba + kbb))
    edge_mass = e_true[pair_i] * e_true[pair_j] * k
    cis_total = 100.0
    inter_total = 100.0
    counts = np.zeros_like(edge_mass, dtype=np.float64)
    counts[cis] = cis_total * edge_mass[cis] / edge_mass[cis].sum()
    counts[~cis] = inter_total * edge_mass[~cis] / edge_mass[~cis].sum()
    diag = np.ones(template.n_loci, dtype=np.float64)
    return contact_model.synthetic_expected_clone(
        template, counts, diag, np.ones(template.n_loci, dtype=np.float64),
        expected_group_totals={"diag": float(diag.sum()), "cis_offdiag": cis_total, "inter": inter_total},
        exposure_mode="mock_expected", endpoint_counts=contact_model.endpoint_counts_from_aggregates(template, counts, diag),
    )


def theta_for(objective: VisibilityGPUObjective, coordinates: np.ndarray, p: float) -> np.ndarray:
    return objective.pack(objective.raw_from_physical(coordinates), p=p)


def parity_precheck() -> dict[str, Any]:
    data = make_mock_data(False)
    coordinates = mock_coordinates(data.n_loci)
    old = GPUVariantObjective(data, "C0", device="cuda", tile_rows=32, use_fused=False, diagnostics=False)
    new = VisibilityGPUObjective(data, "V0-fixed-production-e", device="cuda", pair_block=32)
    theta = theta_for(new, coordinates, 0.71)
    old_value, old_gradient, old_components = old.evaluate(theta, need_gradient=True)
    new_value, new_gradient, new_components = new.evaluate(theta, need_gradient=True)
    return {
        "value_abs": abs(float(old_value) - float(new_value)),
        "gradient_max_abs": float(np.max(np.abs(old_gradient - new_gradient))),
        "component_abs": {key: abs(float(old_components[key]) - float(new_components[key]))
                          for key in ("count_nll_normalized", "bond", "repulsion", "bend", "total")},
        "pass": bool(old_value == new_value and np.allclose(old_gradient, new_gradient, rtol=0.0, atol=2e-13)),
    }


def profile_math_precheck() -> dict[str, Any]:
    template = make_mock_data(False)
    coordinates = mock_coordinates(template.n_loci, seed=44)
    p = 0.69
    e_true = np.exp(np.linspace(-0.3, 0.3, template.n_loci))
    e_true /= e_true.mean()
    expected = mock_expected_data(template, coordinates, p, e_true)
    objective = VisibilityGPUObjective(expected, "V1-profile-e", device="cuda", pair_block=32,
                                       inner_cap=80, profile_warm_start=True)
    theta = theta_for(objective, coordinates, p)
    value, gradient, components = objective.evaluate(theta, need_gradient=True)
    state = objective.visibility_state()
    assert state is not None
    eta = torch.as_tensor(state["eta"], dtype=torch.float64, device=objective.device)
    p_t = torch.as_tensor(p, dtype=torch.float64, device=objective.device)
    x_t = torch.as_tensor(coordinates, dtype=torch.float64, device=objective.device)
    k = objective._build_k(x_t, p_t)
    base_stats = objective._profile_stats(k, eta)
    shifted_stats = objective._profile_stats(k, eta + 0.37)
    factor = torch.exp(0.1 * torch.sin(torch.arange(template.n_loci, dtype=torch.float64, device=objective.device)))
    k_factor = k * factor[objective._pair_i] * factor[objective._pair_j]
    transformed_eta = eta - torch.log(factor)
    transformed_stats = objective._profile_stats(k_factor, transformed_eta)
    eta2 = eta + 0.15 * torch.cos(torch.arange(template.n_loci, dtype=torch.float64, device=objective.device))
    eta2 = objective._project_active(eta2)
    stats2 = objective._profile_stats(k, eta2)
    stats_mid = objective._profile_stats(k, objective._project_active((eta + eta2) / 2.0))
    warm_value, warm_gradient, warm_components = objective.evaluate(theta, need_gradient=True)
    # q 与一个 raw-y 坐标的 envelope finite difference；每次都禁用 warm state。
    def fresh_value(candidate: np.ndarray) -> float:
        fresh = VisibilityGPUObjective(expected, "V1-profile-e", device="cuda", pair_block=32,
                                       inner_cap=80, profile_warm_start=False)
        return float(fresh.evaluate(candidate, need_gradient=False)[0])
    h_q = 1e-5
    theta_plus = theta.copy(); theta_plus[-1] += h_q
    theta_minus = theta.copy(); theta_minus[-1] -= h_q
    fd_q = (fresh_value(theta_plus) - fresh_value(theta_minus)) / (2.0 * h_q)
    h_y = 1e-6
    theta_plus_y = theta.copy(); theta_plus_y[0] += h_y
    theta_minus_y = theta.copy(); theta_minus_y[0] -= h_y
    fd_y = (fresh_value(theta_plus_y) - fresh_value(theta_minus_y)) / (2.0 * h_y)
    fd_rows = {
        "q": {"analytic": float(gradient[-1]), "finite_difference": float(fd_q),
              "abs_error": abs(float(gradient[-1]) - float(fd_q))},
        "raw_y_0": {"analytic": float(gradient[0]), "finite_difference": float(fd_y),
                     "abs_error": abs(float(gradient[0]) - float(fd_y))},
    }
    # 在同一 counts 上扰动两 copy 的 raw-y 和 q，避免只在 known-e 驻点检查
    # count envelope。这里 FD 与 analytic 都只取 profile count NLL，不含 priors。
    rng = np.random.default_rng(204)
    theta_ns = theta.copy()
    theta_ns[:-1] += 0.15 * rng.normal(size=theta_ns.size - 1)
    theta_ns[-1] += 0.27
    ns_objective = VisibilityGPUObjective(expected, "V1-profile-e", device="cuda", pair_block=32,
                                          inner_cap=80, profile_warm_start=False)
    ns_count_value, ns_count_gradient = ns_objective.count_value_and_grad(theta_ns)
    ns_state = ns_objective.visibility_state()
    assert ns_state is not None

    def fresh_count(candidate: np.ndarray) -> float:
        fresh = VisibilityGPUObjective(expected, "V1-profile-e", device="cuda", pair_block=32,
                                       inner_cap=80, profile_warm_start=False)
        return float(fresh.count_value_and_grad(candidate)[0])

    def directional_fd(direction: np.ndarray, step: float) -> dict[str, float]:
        candidate_plus = theta_ns + step * direction
        candidate_minus = theta_ns - step * direction
        finite_difference = (fresh_count(candidate_plus) - fresh_count(candidate_minus)) / (2.0 * step)
        analytic = float(np.dot(ns_count_gradient, direction))
        scale = max(1e-12, abs(analytic), abs(finite_difference))
        return {
            "analytic": analytic,
            "finite_difference": float(finite_difference),
            "abs_error": abs(analytic - float(finite_difference)),
            "relative_error": abs(analytic - float(finite_difference)) / scale,
            "nondegenerate": bool(abs(analytic) > 1e-6),
        }

    raw_direction_rows = []
    for seed in (205, 206):
        direction = np.random.default_rng(seed).normal(size=theta_ns.size - 1)
        direction /= np.linalg.norm(direction)
        full_direction = np.concatenate((direction, np.zeros(1, dtype=np.float64)))
        raw_direction_rows.append(directional_fd(full_direction, 1e-6))
    q_direction = np.zeros_like(theta_ns)
    q_direction[-1] = 1.0
    q_row = directional_fd(q_direction, 1e-5)
    perturbed_warm = VisibilityGPUObjective(expected, "V1-profile-e", device="cuda", pair_block=32,
                                            inner_cap=80, profile_warm_start=True)
    perturbed_warm._warm_eta = torch.as_tensor(ns_state["eta"], dtype=torch.float64,
                                                device=perturbed_warm.device).clone()
    perturbed_warm._warm_eta += 0.4 * torch.as_tensor(
        np.random.default_rng(207).normal(size=template.n_loci), dtype=torch.float64,
        device=perturbed_warm.device)
    warm_ns_count, warm_ns_gradient = perturbed_warm.count_value_and_grad(theta_ns)
    warm_ns_state = perturbed_warm.visibility_state()
    assert warm_ns_state is not None
    nonstationary = {
        "count_value": float(ns_count_value),
        "count_gradient_inf": float(np.max(np.abs(ns_count_gradient))),
        "q_gradient_abs": abs(float(ns_count_gradient[-1])),
        "raw_direction_fd": raw_direction_rows,
        "q_fd": q_row,
        "cold_warm_count_abs": abs(float(ns_count_value) - float(warm_ns_count)),
        "cold_warm_gradient_max_abs": float(np.max(np.abs(ns_count_gradient - warm_ns_gradient))),
        "cold_relative_residual": float(ns_state["relative_degree_residual"]),
        "warm_relative_residual": float(warm_ns_state["relative_degree_residual"]),
    }
    nonstationary["pass"] = bool(
        nonstationary["count_gradient_inf"] > 1e-6
        and nonstationary["q_gradient_abs"] > 1e-6
        and all(row["nondegenerate"] and row["abs_error"] <= 2e-5 + 2e-4 * max(abs(row["analytic"]), abs(row["finite_difference"]))
                and row["relative_error"] <= 2e-4 for row in raw_direction_rows)
        and q_row["nondegenerate"]
        and q_row["abs_error"] <= 2e-5 + 2e-4 * max(abs(q_row["analytic"]), abs(q_row["finite_difference"]))
        and q_row["relative_error"] <= 2e-4
        and nonstationary["cold_warm_count_abs"] <= 1e-9
        and nonstationary["cold_warm_gradient_max_abs"] <= 1e-8
        and nonstationary["cold_relative_residual"] <= PROFILE_RESIDUAL_TOL
        and nonstationary["warm_relative_residual"] <= PROFILE_RESIDUAL_TOL
    )
    return {
        "profile_value": float(value),
        "profile_relative_residual": float(components["profile"]["relative_degree_residual"]),
        "profile_degree_residual_inf": float(components["profile"]["degree_residual_inf"]),
        "cold_warm_value_abs": abs(float(value) - float(warm_value)),
        "cold_warm_gradient_max_abs": float(np.max(np.abs(gradient - warm_gradient))),
        "cold_warm_relative_residual": float(warm_components["profile"]["relative_degree_residual"]),
        "translation_F_abs": abs(_finite(base_stats["objective"]) - _finite(shifted_stats["objective"])),
        "translation_gradient_sum_abs": abs(_finite(base_stats["residual"].sum())),
        "node_factor_F_abs": abs(_finite(base_stats["objective"]) - _finite(transformed_stats["objective"])),
        "convexity_jensen_gap": _finite(stats_mid["objective"]) - 0.5 * (_finite(stats2["objective"]) + _finite(base_stats["objective"])),
        "envelope_fd": fd_rows,
        "nonstationary_count_envelope": nonstationary,
        "known_e_max_abs": float(np.max(np.abs(state["e"] - e_true))),
        "known_e_recovery_pass": bool(np.max(np.abs(state["e"] - e_true)) <= 5e-9),
        "inner_cap": 80,
        "pass": bool(
            components["profile"]["relative_degree_residual"] <= PROFILE_RESIDUAL_TOL
            and abs(float(value) - float(warm_value)) <= 1e-10
            and float(np.max(np.abs(gradient - warm_gradient))) <= 1e-8
            and abs(_finite(base_stats["objective"]) - _finite(shifted_stats["objective"])) <= 1e-10
            and abs(_finite(base_stats["objective"]) - _finite(transformed_stats["objective"])) <= 1e-9
            and _finite(stats_mid["objective"]) <= 0.5 * (_finite(stats2["objective"]) + _finite(base_stats["objective"])) + 1e-10
            and fd_rows["q"]["abs_error"] <= 5e-6
            and fd_rows["raw_y_0"]["abs_error"] <= 5e-5
            and nonstationary["pass"]
             and float(np.max(np.abs(state["e"] - e_true))) <= 5e-9
        ),
    }


def _finite(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item()) if value.ndim == 0 else float(value.detach().cpu().numpy().sum())


def zero_face_precheck() -> dict[str, Any]:
    data = make_mock_data(True)
    coordinates = mock_coordinates(data.n_loci, seed=45)
    profile = VisibilityGPUObjective(data, "V1-profile-e", device="cuda", pair_block=32, inner_cap=80)
    theta = theta_for(profile, coordinates, 0.7)
    value, gradient, components = profile.evaluate(theta, need_gradient=True)
    state = profile.visibility_state()
    assert state is not None
    zero = profile.support.zero_mask
    vz = VisibilityGPUObjective(data, "VZ-zero-degree-fixed-e", device="cuda", pair_block=32)
    vz_value, vz_gradient, _ = vz.evaluate(theta, need_gradient=True)
    return {
        "n_loci": int(data.n_loci),
        "n_pairs": int(data.n_pairs),
        "zero_degree": int(zero.sum()),
        "profile_zero_e_exact": bool(np.all(state["e"][zero] == 0.0)),
        "profile_active_e_positive": bool(np.all(state["e"][~zero] > 0.0)),
        "profile_relative_residual": float(components["profile"]["relative_degree_residual"]),
        "vz_zero_e_exact": bool(np.all(vz.visibility_state()["e"][zero] == 0.0)),
        "pass": bool(np.all(state["e"][zero] == 0.0) and np.all(state["e"][~zero] > 0.0)
                    and len(state["e"]) == data.n_loci and int(data.n_pairs) == data.n_loci * (data.n_loci - 1) // 2),
        "profile_total": float(value),
        "vz_total": float(vz_value),
        "vz_gradient_finite": bool(np.all(np.isfinite(vz_gradient))),
    }


def budget_precheck() -> dict[str, Any]:
    data = make_mock_data(False)
    coordinates = mock_coordinates(data.n_loci, seed=46)
    objective = VisibilityGPUObjective(data, "V0-fixed-production-e", device="cuda", pair_block=32)
    try:
        fit = run_budgeted_lbfgs(
            objective, objective.raw_from_physical(coordinates), p_init=0.75,
            maxfun=2, maxiter=3, maxls=20, ftol=1e-10, canonical_gtol=0.0,
        )
        result = fit.as_dict()
        result["fg_cap_pass"] = bool(fit.nfev <= 2)
        result["accepted_endpoint_pass"] = bool(fit.endpoint_was_last_accepted)
        result["last_accepted_finite"] = bool(np.all(np.isfinite(fit.theta)) and np.isfinite(fit.fun))
        result["pass"] = bool(result["fg_cap_pass"] and result["accepted_endpoint_pass"] and result["last_accepted_finite"])
        return result
    except Exception as exc:
        return {"pass": False, "error_type": type(exc).__name__, "error": str(exc)}


def synthetic_support_precheck() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("truth_coordinates_exposed") is not False or manifest.get("optimizer_started") is not False:
        raise RuntimeError("known-e manifest boundary changed")
    template = contact_model.load_aggregate(INPUT_PATH, bin_size=1_000_000, verify_frozen_hash=True)
    rows = {}
    for fixture in ("P2", "N2"):
        snapshot = ROOT / manifest["fixtures"][fixture]["count_snapshot"]["path"]
        if sha256_file(snapshot) != manifest["fixtures"][fixture]["count_snapshot"]["sha256"]:
            raise RuntimeError("synthetic count snapshot hash mismatch for %s" % fixture)
        with np.load(snapshot, allow_pickle=False) as payload:
            counts = np.asarray(payload["counts"], dtype=np.int64)
            diag = np.asarray(payload["diag_counts"], dtype=np.int64)
            degree = np.zeros(template.n_loci, dtype=np.int64)
            np.add.at(degree, template.pair_i, counts)
            np.add.at(degree, template.pair_j, counts)
            rows[fixture] = {
                "count_snapshot_sha256": sha256_file(snapshot),
                "n_loci": int(len(degree)),
                "zero_degree": int((degree == 0).sum()),
                "active_degree": int((degree > 0).sum()),
                "zero_degree_diag_zero": int(((degree == 0) & (diag == 0)).sum()),
                "zero_degree_diag_positive": int(((degree == 0) & (diag > 0)).sum()),
                "vz_same_as_v0": bool((degree == 0).sum() == 0),
            }
        for key in ("known_e_exposure",):
            known = ROOT / manifest["fixtures"][fixture][key]["path"]
            if sha256_file(known) != manifest["fixtures"][fixture][key]["sha256"]:
                raise RuntimeError("known-e vector hash mismatch for %s" % fixture)
            with np.load(known, allow_pickle=False) as payload:
                values = np.asarray(payload[payload.files[0]], dtype=np.float64).reshape(-1)
            rows[fixture]["known_e_n"] = int(len(values))
            rows[fixture]["known_e_mean"] = float(values.mean())
            rows[fixture]["known_e_positive"] = bool(np.all(values > 0.0))
    return {"fixtures": rows, "pass": all(row["zero_degree"] == 0 for row in rows.values())}


def real_benchmark() -> list[dict[str, Any]]:
    configure_014_paths()
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    names_lengths: tuple[tuple[str, int], ...] | None = None
    for bin_size in (5_000_000, 2_000_000, 1_000_000):
        load_started = time.perf_counter()
        data = contact_model.load_aggregate(INPUT_PATH, bin_size=bin_size, verify_frozen_hash=True)
        load_seconds = time.perf_counter() - load_started
        audit = data.budget()
        expected = REAL_EXPECTED[bin_size]
        for key, expected_value in expected.items():
            actual = int(data.n_loci) if key == "n_loci" else int(audit[key])
            if actual != expected_value:
                raise RuntimeError("real %d %s changed: %s" % (bin_size, key, actual))
        if names_lengths is None:
            names_lengths = tuple(zip(data.chromosome_names, data.chromosome_lengths.tolist()))
        if previous is None:
            state = reconstruction_init.initialize_approved_candidate(
                "consensus", tuple(name for name, _ in names_lengths),
                tuple(int(length) for _, length in names_lengths), bin_size)
        else:
            state = reconstruction_init.warm_start_from_layer(
                previous["coords"], previous["positions"], previous["chromosome_index"],
                tuple(name for name, _ in names_lengths), tuple(int(length) for _, length in names_lengths),
                bin_size, 1103)
        previous = state
        layer_row: dict[str, Any] = {
            "bin_size_bp": int(bin_size),
            "load_seconds": float(load_seconds),
            "budget": audit,
            "support": _support_audit(data).as_dict(),
            "modes": {},
        }
        raw_by_mode: dict[str, np.ndarray] = {}
        for mode in ("V0-fixed-production-e", "VZ-zero-degree-fixed-e", "V1-profile-e"):
            objective = VisibilityGPUObjective(data, mode, device="cuda", pair_block=262_144,
                                               inner_cap=80, cg_cap=80, profile_warm_start=True)
            raw = objective.raw_from_physical(np.asarray(state["coords"], dtype=np.float64))
            raw_by_mode[mode] = raw
            theta = objective.pack(raw, p=0.75)
            objective.synchronize()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                value, gradient = objective.value_and_grad(theta)
                cached = objective.cached_value_and_components(theta)
                if cached is None:
                    raise RuntimeError("last accepted value/components cache missing")
                _cached_value, components = cached
                objective.synchronize()
                wall = time.perf_counter() - started
                arrays = objective.profile_arrays()
                layer_row["modes"][mode] = {
                    "status": "completed",
                    "total": float(value),
                    "gradient_inf": float(np.max(np.abs(gradient))),
                    "count_nll_per_record": float(components["count_nll_normalized"]),
                    "profile": components["profile"],
                    "diagnostics": objective.diagnostics(),
                    "wall_seconds": float(wall),
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                    "e_zero_count": int((arrays["e"] == 0.0).sum()) if arrays is not None else None,
                }
            except Exception as exc:
                objective.synchronize()
                layer_row["modes"][mode] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "wall_seconds": float(time.perf_counter() - started),
                    "diagnostics": objective.diagnostics(),
                }
        rows.append(layer_row)
    return rows


def source_manifest() -> dict[str, Any]:
    paths = [
        SOURCE / "visibility_profile.py",
        SOURCE / "visibility_preflight.py",
        SOURCE / "frozen_035" / "gpu_variant_backend.py",
        SOURCE / "frozen_035" / "gpu_multires_controller.py",
        SOURCE / "frozen_037" / "m1_preconditioner.py",
        SOURCE / "frozen_037" / "m1_gpu_controller.py",
    ]
    paths.extend(sorted((SOURCE / "frozen_pr" / "pr").glob("*.py")))
    paths.extend([
        ROOT / "test_res/035-20260914T060945Z-gpu-multires-preflight/source/gpu_variant_backend.py",
        ROOT / "test_res/035-20260914T060945Z-gpu-multires-preflight/source/gpu_multires_controller.py",
        ROOT / "test_res/037-20260914T143812Z-gpu-m1-preflight/source/m1_preconditioner.py",
        ROOT / "test_res/037-20260914T143812Z-gpu-m1-preflight/source/m1_gpu_controller.py",
        ROOT / "pr/contact_model.py", ROOT / "pr/reconstruction_init.py",
        ROOT / "pr/multires_variant_runner.py", ROOT / "pr/genome.py",
    ])
    return {str(path.resolve().relative_to(ROOT)): sha256_file(path) for path in paths}


def main() -> int:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    configure_014_paths()
    out = SOURCE.parent
    cuda = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "available": bool(torch.cuda.is_available()), "device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cuda.update({"device_name": props.name, "total_memory": int(props.total_memory)})
    preflight: dict[str, Any] = {
        "schema": "visibility-preflight-v1",
        "status": "running",
        "started_at_utc": started,
        "scope": "training-side math/numerical preflight and one real 5/2/1Mb benchmark; no formal",
        "input_boundary": {
            "snpfree_path": str(INPUT_PATH), "snpfree_sha256": sha256_file(INPUT_PATH),
            "snpfree_expected_sha256": EXPECTED_INPUT_SHA,
            "reference_opened": False, "phase_opened": False, "evaluation_opened": False,
            "038_opened": False, "synthetic_truth_coordinates_opened": False,
            "pair_sampling": False, "mixed_precision": False,
        },
        "cuda": cuda,
        "source_sha256": source_manifest(),
    }
    if not cuda["available"]:
        preflight["status"] = "failed"
        preflight["error"] = "CUDA unavailable"
        write_json(out / "preflight.json", preflight)
        return 2
    try:
        preflight["synthetic_support"] = synthetic_support_precheck()
        preflight["math"] = {
            "fixed_e_old_c0_parity": parity_precheck(),
            "profile_convex_gauge_factor_envelope_known_e": profile_math_precheck(),
            "zero_degree_fixed_face": zero_face_precheck(),
            "accepted_state_fg_cap": budget_precheck(),
        }
        preflight["real_benchmark"] = real_benchmark()
        all_math = all(bool(item.get("pass")) for item in preflight["math"].values())
        synthetic_ok = bool(preflight["synthetic_support"]["pass"])
        real_ok = all(all(mode.get("status") == "completed" for mode in row["modes"].values())
                      for row in preflight["real_benchmark"])
        preflight["status"] = "passed" if all_math and synthetic_ok and real_ok else "failed"
        preflight["completion"] = {
            "math_pass": all_math, "synthetic_support_pass": synthetic_ok,
            "real_benchmark_completed": real_ok, "formal_started": False,
            "formal_launch_authorized_by_parent": False,
        }
    except Exception as exc:
        preflight["status"] = "failed"
        preflight["error_type"] = type(exc).__name__
        preflight["error"] = str(exc)
    preflight["ended_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(out / "preflight.json", preflight)
    write_json(out / "source_hashes.json", {"source_sha256": preflight["source_sha256"],
                                             "input_sha256": preflight["input_boundary"]["snpfree_sha256"]})
    return 0 if preflight["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
