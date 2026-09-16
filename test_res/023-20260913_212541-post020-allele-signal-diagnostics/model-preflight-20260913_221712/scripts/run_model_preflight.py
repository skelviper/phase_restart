#!/usr/bin/env python3
"""post-020 allele model variants 的独立全规模预检。

本脚本只做固定值/梯度检查。有意不调用 paired runner 的 optimizer，也不导入 reference、phase、Softall 或 native FDG 代码。
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import types
import importlib.util
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[4]
SEALED = REPO / "test_res/023-20260913_212541-post020-allele-signal-diagnostics"
OUT = Path(__file__).resolve().parents[1]
INPUT = REPO / "inputs/P9016.snpfree.pairs.gz"
RUN020 = REPO / "test_res/020-20260913_071841-v1-p9016-joint"
RUN022 = REPO / "test_res/022-20260913_111031-v1-continuation-fdg-r2"
CURRENT_PR = REPO / "pr"
FROZEN_PR = RUN020 / "provenance/training-code/pr"

sys.path.insert(0, str(REPO))
from pr import contact_model as current_contact_model  # noqa: E402
from pr import allele_models  # noqa: E402
from pr import paired_run  # noqa: E402

current_contact_model.SNPFREE = str(INPUT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite value in JSON payload")
        return value
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")


def max_abs(a: Any, b: Any) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        raise AssertionError("shape mismatch: %s vs %s" % (left.shape, right.shape))
    return float(np.max(np.abs(left - right), initial=0.0))


def numeric_component_diff(first: dict[str, Any], second: dict[str, Any],
                           keys: tuple[str, ...] = ("count_nll_normalized", "bond", "repulsion", "bend", "p_prior", "p", "total")) -> dict[str, float]:
    return {key: abs(float(first[key]) - float(second[key])) for key in keys}


def assert_finite_components(components: dict[str, Any]) -> None:
    for key, value in components.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            if not math.isfinite(float(value)):
                raise AssertionError("nonfinite component %s" % key)


def import_frozen_contact_model():
    package_name = "frozen_pr"
    package = types.ModuleType(package_name)
    package.__path__ = [str(FROZEN_PR)]
    package.__package__ = package_name
    package.__spec__ = importlib.util.spec_from_loader(package_name, loader=None, is_package=True)
    sys.modules[package_name] = package
    module_name = package_name + ".contact_model"
    spec = importlib.util.spec_from_file_location(module_name, FROZEN_PR / "contact_model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen contact_model snapshot")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.SNPFREE = str(INPUT)
    if Path(module.__file__).resolve() != (FROZEN_PR / "contact_model.py").resolve():
        raise AssertionError("frozen contact_model import path mismatch")
    return module


def read_endpoint(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        theta = np.asarray(payload["theta"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.float64)
        x = np.asarray(payload["coordinates"], dtype=np.float64)
    return theta, y, x


def theta_for_raw(raw: np.ndarray, q: float) -> np.ndarray:
    return np.concatenate((np.asarray(raw, dtype=np.float64).ravel(), np.asarray([q], dtype=np.float64)))


def physical_recomposition(module: Any, objective: Any, x: np.ndarray, p: float) -> dict[str, Any]:
    count_components, count_x, count_p = objective._count_nll_and_gradient(x, p, need_gradient=True)
    bond, bond_x = objective._bond_and_gradient(x)
    repulsion, repulsion_x = objective._repulsion_and_gradient(x)
    bend, bend_x = objective._bend_and_gradient(x)
    p_prior = -module.P_PRIOR_STRENGTH * math.log(p * (1.0 - p))
    p_prior_p = module.P_PRIOR_STRENGTH * (1.0 / (1.0 - p) - 1.0 / p)
    weights = module.ObjectiveWeights()
    # snapshot 在 module scope 暴露这些 constants/classes，而不是暴露在 instance 上。
    weighted_x = (weights.count * count_x + weights.bond * bond_x
                  + weights.repulsion * repulsion_x + weights.bend * bend_x)
    weighted_p = weights.count * count_p + weights.p_prior * p_prior_p
    values = {
        "count_nll_normalized": float(count_components["count_nll_normalized"]),
        "bond": float(bond),
        "repulsion": float(repulsion),
        "bend": float(bend),
        "p_prior": float(p_prior),
    }
    values["total"] = float(values["count_nll_normalized"] + values["bond"] + values["repulsion"] + 0.01 * values["bend"] + values["p_prior"])
    return {"values": values, "gradient_x": weighted_x, "gradient_p": float(weighted_p)}


def data_field_comparison(original: Any, changed: Any) -> dict[str, Any]:
    preserved = []
    changed_fields = []
    for field in fields(original):
        name = field.name
        left = getattr(original, name)
        right = getattr(changed, name)
        if name in ("exposure", "exposure_mode"):
            changed_fields.append(name)
            continue
        if isinstance(left, np.ndarray):
            same = np.array_equal(left, right)
        else:
            same = left == right
        if not same:
            raise AssertionError("C3 changed field outside exposure: %s" % name)
        preserved.append(name)
    return {"preserved_fields": preserved, "declared_changed_fields": changed_fields}


def rss_peak() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux ru_maxrss 的单位是 KiB。
    result = {
        "ru_maxrss_kib": int(usage.ru_maxrss),
        "ru_maxrss_gib": float(usage.ru_maxrss / (1024.0 * 1024.0)),
        "source": "resource.getrusage(RUSAGE_SELF), Linux KiB",
    }
    status = Path("/proc/self/status")
    if status.exists():
        values = {}
        for line in status.read_text().splitlines():
            if line.startswith(("VmHWM:", "VmRSS:")):
                key, number, unit = line.split()[:3]
                values[key.rstrip(":")] = {"value_kib": int(number), "unit": unit}
        result["proc_status_at_end"] = values
    return result


def main() -> None:
    started = time.perf_counter()
    sealed_manifest_before = sha256_file(SEALED / "results/manifest.json")
    frozen_contact_model = import_frozen_contact_model()
    source_paths = {
        "current_pr/allele_models.py": CURRENT_PR / "allele_models.py",
        "current_pr/paired_run.py": CURRENT_PR / "paired_run.py",
        "current_pr/contact_model.py": CURRENT_PR / "contact_model.py",
        "current_pr/joint_fit.py": CURRENT_PR / "joint_fit.py",
        "frozen_020/contact_model.py": FROZEN_PR / "contact_model.py",
        "frozen_020/joint_fit.py": FROZEN_PR / "joint_fit.py",
    }
    source_hashes = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in source_paths.items()}
    endpoint_specs = {
        "020_random_joint_1m": {
            "npz": RUN020 / "checkpoints/random_joint/1m-accepted-0240.npz",
            "coordinate": RUN020 / "selected.3dg",
            "npz_sha256": "c6a8890e2cff53767a8bddd49d4cb2d77892eee6c89f15895b3a5d8ba600b723",
            "coordinate_sha256": "afb2d52ae11e342e9b43b3c8042c5581760d177c5e563f2478646ab36b3e7078",
        },
        "022_random_joint_continuation_480": {
            "npz": RUN022 / "theta/final-theta.npz",
            "coordinate": RUN022 / "coords/random_joint/final-1m-continuation-61da99669694b822.3dg",
            "npz_sha256": "588bafce1d7652fa8cf17cb060d23807dd6098c3e072fd9fc52784a145860a57",
            "coordinate_sha256": "49501f5b38d699fb2c9c8849616edd02b70fccd5ae39e1efe6036e17f5f620e7",
        },
    }
    data = current_contact_model.load_frozen_p9016_aggregate(1_000_000)
    if data.n_loci != 2645 or data.n_pairs != 3496690:
        raise AssertionError("unexpected full-grid dimensions")
    budget = data.budget()
    if budget["raw_records"] != 1703888 or budget["n_zero_eligible_pairs"] != 3009436:
        raise AssertionError("input budget mismatch")

    endpoint_results: dict[str, Any] = {}
    all_x_below_identity = True
    c3_data = allele_models.data_for_model(data, "C3")
    c3_contract = data_field_comparison(data, c3_data)
    c3_budget_original = data.budget()
    c3_budget = c3_data.budget()
    budget_preserved = {key: c3_budget[key] == c3_budget_original[key]
                        for key in c3_budget_original if key != "exposure_mode"}
    if not all(budget_preserved.values()):
        raise AssertionError("C3 changed a budget field outside exposure_mode")
    if not np.all(c3_data.exposure == 1.0) or c3_data.exposure_mode != "uniform":
        raise AssertionError("C3 exposure view is not uniform")

    for endpoint_name, spec in endpoint_specs.items():
        if sha256_file(spec["npz"]) != spec["npz_sha256"]:
            raise AssertionError("endpoint npz hash mismatch: %s" % endpoint_name)
        if sha256_file(spec["coordinate"]) != spec["coordinate_sha256"]:
            raise AssertionError("endpoint coordinate hash mismatch: %s" % endpoint_name)
        theta, y, x = read_endpoint(spec["npz"])
        if theta.shape != (6 * data.n_loci + 1,) or y.shape != (2, data.n_loci, 3) or x.shape != y.shape:
            raise AssertionError("endpoint shape mismatch: %s" % endpoint_name)
        if max_abs(frozen_contact_model.sphere_forward(y), x) > 1e-12:
            raise AssertionError("endpoint sphere roundtrip mismatch: %s" % endpoint_name)
        q = float(theta[-1])
        p, dpdq = current_contact_model.p_from_q(q)
        radius = np.linalg.norm(x, axis=2)
        max_radius = float(radius.max())
        all_x_below_identity = all_x_below_identity and bool(max_radius < allele_models.IDENTITY_RADIUS)

        frozen_objective = frozen_contact_model.JointObjective(data)
        c0_objective = allele_models.objective_for_model(data, "C0")
        c1_objective = allele_models.objective_for_model(data, "C1")
        c2_map_objective = allele_models.objective_for_model(data, "C2-map")
        c2_free_objective = allele_models.objective_for_model(data, "C2-free")
        c3_objective = allele_models.objective_for_model(data, "C3")

        original_total, original_gradient, original_components = frozen_objective.evaluate(theta, need_gradient=True)
        c0_total, c0_gradient, c0_components = c0_objective.evaluate(theta, need_gradient=True)
        c0_component_diff = numeric_component_diff(original_components, c0_components)
        c0_gradient_diff = max_abs(original_gradient, c0_gradient)
        if max(c0_component_diff.values(), default=0.0) > 1e-12 or c0_gradient_diff > 1e-12:
            raise AssertionError("C0 mismatch: %s" % endpoint_name)

        physical = physical_recomposition(frozen_contact_model, frozen_contact_model.JointObjective(data), x, float(p))
        raw_map = allele_models.raw_coordinates_from_physical(c2_map_objective, x)
        raw_free = allele_models.raw_coordinates_from_physical(c2_free_objective, x)
        if max_abs(raw_map, x) > 1e-14 or max_abs(raw_free, x) > 0.0:
            raise AssertionError("C2 raw physical conversion is not identity in the interior")
        theta_map = theta_for_raw(raw_map, q)
        theta_free = theta_for_raw(raw_free, q)
        c2_map_total, c2_map_gradient, c2_map_components = c2_map_objective.evaluate(theta_map, need_gradient=True)
        c2_free_total, c2_free_gradient, c2_free_components = c2_free_objective.evaluate(theta_free, need_gradient=True)
        c2_map_gradient_x = c2_map_gradient[:-1].reshape(x.shape)
        c2_free_gradient_x = c2_free_gradient[:-1].reshape(x.shape)
        c2_map_value_diff = numeric_component_diff(original_components, c2_map_components)
        c2_free_value_diff = numeric_component_diff(original_components, c2_free_components)
        c2_map_physical_gradient_diff = max_abs(c2_map_gradient_x, physical["gradient_x"])
        c2_free_physical_gradient_diff = max_abs(c2_free_gradient_x, physical["gradient_x"])
        c2_map_q_diff = abs(float(c2_map_gradient[-1]) - float(dpdq * physical["gradient_p"]))
        c2_free_q_diff = abs(float(c2_free_gradient[-1]) - float(dpdq * physical["gradient_p"]))
        c2_map_free_value_diff = max_abs(np.asarray([c2_map_total]), np.asarray([c2_free_total]))
        c2_map_free_gradient_diff = max_abs(c2_map_gradient, c2_free_gradient)
        map_diagnostics = c2_map_objective.map_diagnostics()
        if max(c2_map_value_diff.values(), default=0.0) > 1e-12 or max(c2_free_value_diff.values(), default=0.0) > 1e-12:
            raise AssertionError("C2 value mismatch: %s" % endpoint_name)
        if max(c2_map_physical_gradient_diff, c2_free_physical_gradient_diff, c2_map_q_diff, c2_free_q_diff, c2_map_free_gradient_diff) > 1e-10:
            raise AssertionError("C2 physical gradient mismatch: %s" % endpoint_name)
        if map_diagnostics["nonidentity_map_eval_count"] != 0 or map_diagnostics["nonidentity_bead_eval_count"] != 0:
            raise AssertionError("C2-map unexpectedly left identity interior: %s" % endpoint_name)

        c1_total, c1_gradient, c1_components = c1_objective.evaluate(theta, need_gradient=True)
        c1_expected_total = float(original_total - 0.01 * original_components["bend"])
        c1_other_diff = numeric_component_diff(original_components, c1_components,
                                                keys=("count_nll_normalized", "bond", "repulsion", "bend", "p_prior", "p"))
        c1_total_error = abs(float(c1_total) - c1_expected_total)
        # value/component contract 是首要条件；同时要求 C1 gradient 有限。
        assert_finite_components(c1_components)
        if c1_total_error > 1e-12 or max(c1_other_diff.values(), default=0.0) > 1e-12:
            raise AssertionError("C1 contract mismatch: %s" % endpoint_name)

        c3_total, c3_gradient, c3_components = c3_objective.evaluate(theta, need_gradient=True)
        assert_finite_components(c3_components)
        if not math.isfinite(float(c3_total)) or not np.all(np.isfinite(c3_gradient)):
            raise AssertionError("C3 objective/gradient is nonfinite: %s" % endpoint_name)

        endpoint_results[endpoint_name] = {
            "p": float(p),
            "q": q,
            "dp_dq": float(dpdq),
            "max_physical_radius": max_radius,
            "all_physical_radii_lt_0.90": bool(max_radius < allele_models.IDENTITY_RADIUS),
            "original_v1": {
                "total": float(original_total),
                "components": original_components,
                "gradient_l2": float(np.linalg.norm(original_gradient)),
                "physical_gradient_x_l2": float(np.linalg.norm(physical["gradient_x"])),
                "physical_gradient_p": float(physical["gradient_p"]),
            },
            "C0_same_p_x_as_frozen_v1": {
                "passed": True,
                "max_component_abs_difference": max(c0_component_diff.values(), default=0.0),
                "gradient_max_abs_difference": c0_gradient_diff,
                "total": float(c0_total),
            },
            "C1_bend_removed_only": {
                "passed": True,
                "total": float(c1_total),
                "expected_total": c1_expected_total,
                "total_error": c1_total_error,
                "component_abs_differences_except_total": c1_other_diff,
                "gradient_finite": bool(np.all(np.isfinite(c1_gradient))),
            },
            "C2_map_vs_free_and_physical_recomposition": {
                "passed": True,
                "map_total": float(c2_map_total),
                "free_total": float(c2_free_total),
                "map_component_abs_differences_vs_v1": c2_map_value_diff,
                "free_component_abs_differences_vs_v1": c2_free_value_diff,
                "map_vs_free_total_abs_difference": c2_map_free_value_diff,
                "map_vs_free_gradient_max_abs_difference": c2_map_free_gradient_diff,
                "map_physical_x_gradient_max_abs_difference": c2_map_physical_gradient_diff,
                "free_physical_x_gradient_max_abs_difference": c2_free_physical_gradient_diff,
                "map_q_gradient_abs_difference": c2_map_q_diff,
                "free_q_gradient_abs_difference": c2_free_q_diff,
                "map_diagnostics": map_diagnostics,
            },
            "C3_finite": {
                "passed": True,
                "total": float(c3_total),
                "gradient_l2": float(np.linalg.norm(c3_gradient)),
                "components": c3_components,
            },
        }

    fixture = np.zeros((2, data.n_loci, 3), dtype=np.float64)
    fixture[0, 0, 0] = 1.2
    fixture_path = OUT / "results/free-writer-fixture.3dg"
    writer_metadata = paired_run.write_coordinates(fixture_path, data, "C2-free", fixture)
    fixture_text = fixture_path.read_text()
    bounded_path = OUT / "results/bounded-writer-rejection.3dg"
    bounded_rejected = False
    try:
        paired_run.write_coordinates(bounded_path, data, "C0", fixture)
    except AssertionError:
        bounded_rejected = True
    if not bounded_rejected or bounded_path.exists():
        raise AssertionError("bounded writer did not reject outside-ball fixture")
    if writer_metadata["serialization_clip"]["clipped_coordinates"] != 0 or abs(writer_metadata["max_radius"] - 1.2) > 1e-14:
        raise AssertionError("C2-free writer clipped the outside-ball fixture")
    if "1.2" not in fixture_text:
        raise AssertionError("C2-free writer did not serialize the outside-ball fixture")

    sealed_manifest_after = sha256_file(SEALED / "results/manifest.json")
    if sealed_manifest_before != sealed_manifest_after:
        raise AssertionError("sealed 023 manifest changed during preflight")
    forbidden_modules = sorted(name for name in sys.modules
                               if name.startswith(("pr.ref", "pr.viz", "pr.refeval", "frozen_pr.ref")))
    if forbidden_modules:
        raise AssertionError("forbidden evaluation modules imported: %s" % forbidden_modules)
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "post020-model-preflight-v1",
        "status": "passed",
        "preflight_root": str(OUT),
        "scope": "real-scale fixed endpoints only; no optimizer, native FDG, reference, phase, or Softall",
        "input": {"path": str(INPUT), "sha256": sha256_file(INPUT), "raw_records": int(budget["raw_records"]), "n_loci": int(data.n_loci), "n_pairs": int(data.n_pairs), "n_zero_eligible_pairs": int(budget["n_zero_eligible_pairs"])},
        "source_hashes": source_hashes,
        "sealed_023_manifest": {"sha256_before": sealed_manifest_before, "sha256_after": sealed_manifest_after, "unchanged": sealed_manifest_before == sealed_manifest_after},
        "C3_data_contract": {"only_exposure_field_changed": True, "contract": c3_contract, "budget_fields_preserved_except_exposure_mode": budget_preserved, "uniform_exposure": bool(np.all(c3_data.exposure == 1.0)), "n_pairs": int(c3_data.n_pairs), "n_diag_bins": int(len(c3_data.diag_counts))},
        "all_real_endpoints_r_lt_0.90": all_x_below_identity,
        "endpoints": endpoint_results,
        "free_writer_fixture": {"coordinate": [1.2, 0.0, 0.0], "writer_metadata": writer_metadata, "serialized_1.2_present": "1.2" in fixture_text, "bounded_C0_rejected": bounded_rejected, "implicit_clip_detected": False},
        "optimizer_called": False,
        "forbidden_modules_imported": forbidden_modules,
        "rss_peak": rss_peak(),
        "elapsed_seconds": float(elapsed),
    }
    write_json(OUT / "results/model_preflight.json", result)
    write_json(OUT / "logs/preflight_runtime.json", {"status": "completed", "elapsed_seconds": elapsed, "python": sys.executable, "conda_environment": __import__("os").environ.get("CONDA_DEFAULT_ENV")})
    print(json.dumps({"status": "passed", "output": str(OUT), "endpoint_count": len(endpoint_results), "rss_peak_gib": result["rss_peak"]["ru_maxrss_gib"], "elapsed_seconds": elapsed}, sort_keys=True))


if __name__ == "__main__":
    main()
