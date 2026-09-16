#!/usr/bin/env python3
"""8 个 real 和 6 个 synthetic endpoint 的受 gate 控制的下一步 R2 入口。

Preparation 和小型 calibration fixture 不会打开 real coordinates、synthetic truth coordinates 或 reference。只有 ``--evaluate-released`` 路径会打开这些 payload，并且它首先锁定每个 candidate、source、terminal、selection、code 和 old-21 mask 输入的哈希。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PREP_DIR = Path(__file__).resolve().parent
ROOT = PREP_DIR.parents[2]
OLD_PREP_DIR = ROOT / "docs" / "audits" / "multires-r2-preparation-20260914_041826"
CONFIG_PATH = PREP_DIR / "config.json"
PROTOCOL_PATH = PREP_DIR / "r2_protocol.json"
CONTRACT_PATH = PREP_DIR / "release_contract.json"
WORKER_MANIFEST_PATH = PREP_DIR / "synthetic_inputs" / "known_e_worker_input_manifest.json"
EXPOSURE_MANIFEST_PATH = PREP_DIR / "synthetic_inputs" / "exposure_manifest.json"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
REAL_METHODS = ("M0", "M1")
REAL_BUDGETS = ("B1", "B2")
SOURCES = ("consensus_joint_base1103", "random_joint_base2207")
REAL_ENDPOINT_IDS = tuple(
    "%s-%s-%s" % (method, budget, source)
    for method in REAL_METHODS
    for budget in REAL_BUDGETS
    for source in SOURCES
)
SYNTHETIC_ENDPOINT_IDS = (
    "P2-M0-production-e", "P2-M1-production-e", "P2-M0-known-e",
    "N2-M0-production-e", "N2-M1-production-e", "N2-M0-known-e",
)
ALL_ENDPOINT_IDS = REAL_ENDPOINT_IDS + SYNTHETIC_ENDPOINT_IDS
REAL_METRICS = ("matched", "cross", "contrast", "minmargin")
SYNTHETIC_METRICS = ("matched", "cross", "contrast", "minmargin")
TIE_TOL = 1e-12
SELECTION_TIE_TOL = 1e-9
MIN_COMMON_PAIRS = 20
REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
MASK_SHA256 = "d0c325dea1289374152a327557605f8bbd52267b025cd7b57689be3e3666c49e"
MASK_MANIFEST_SHA256 = "fa1b26c834c173604c21f954d494cece8e053dd970e7cb6111f561b349574acb"


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pr import allele_calibration as synthetic_prepare  # noqa: E402
from pr import allele_calibration_evaluator as synthetic_evaluator  # noqa: E402
from pr import r2comparison  # noqa: E402


def _load_old_adapter():
    path = OLD_PREP_DIR / "multires_r2.py"
    spec = importlib.util.spec_from_file_location("p9016_frozen_multires_r2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen real-R2 adapter: %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OLD = _load_old_adapter()


class EntryError(RuntimeError):
    """Preparation 或 parent release 违反契约时抛出。"""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EntryError("cannot read JSON metadata: %s" % path) from exc
    if not isinstance(value, Mapping):
        raise EntryError("JSON root is not an object: %s" % path)
    return dict(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise EntryError("cannot hash file: %s" % path) from exc
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise EntryError("missing or malformed SHA256: %s" % label)
    return value


def _resolve(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] in {"data", "test_res", "docs", "pr", "inputs"}:
        return (ROOT / path).resolve()
    return (base / path).resolve()


def _resolve_release_path(value: str | Path, release_path: Path) -> Path:
    return _resolve(value, release_path.parent)


def _resolve_preparation_bound_path(value: str | Path, release_path: Path) -> Path:
    """解析 release-relative paths；对本地 code/config 文件提供 fallback。"""
    primary = _resolve_release_path(value, release_path)
    if primary.is_file():
        return primary
    fallback = _resolve(value, PREP_DIR)
    if fallback.is_file():
        return fallback
    return primary


def _hash_expected(path: Path, expected: Any, label: str) -> dict[str, Any]:
    declared = _require_sha(expected, label + ".declared")
    if not path.is_file():
        raise EntryError("missing locked file: %s" % path)
    actual = _sha256(path)
    if actual != declared:
        raise EntryError("SHA mismatch for %s: %s != %s" % (label, actual, declared))
    return {"path": str(path), "sha256": actual}


def _write_tsv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row.get(column) is None else _jsonable(row.get(column)) for column in columns})
            count += 1
    return count


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _assert_close(left: Any, right: Any, tolerance: float, label: str) -> None:
    if not (_finite(left) and _finite(right)) or not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
        raise EntryError("%s differs: %r != %r" % (label, left, right))


def validate_preparation() -> dict[str, Any]:
    """只验证 metadata 和 exposure outputs；不读取 candidate/truth/reference 字节。"""
    config = _read_json(CONFIG_PATH)
    protocol = _read_json(PROTOCOL_PATH)
    contract = _read_json(CONTRACT_PATH)
    if config.get("schema_version") != "p9016-next-step-r2-config-v1":
        raise EntryError("unexpected next-step config schema")
    if config.get("status") != "prepared_only_pending_parent_release":
        raise EntryError("config is not prepared-only pending release")
    if protocol.get("schema_version") != "p9016-next-step-r2-protocol-v1":
        raise EntryError("unexpected next-step protocol schema")
    if protocol.get("status") != "frozen_preparation_pending_parent_release":
        raise EntryError("protocol is not frozen preparation")
    protocol_stop = protocol.get("training_definition", {}).get("canonical_stop_rule", {})
    if (protocol_stop.get("scipy_gtol") != 0.0
            or protocol_stop.get("accepted_callback_gradient_inf_max") != 1e-6
            or protocol_stop.get("common_ftol") != 1e-10
            or protocol_stop.get("common_maxls") != 20):
        raise EntryError("protocol stop parameters changed")
    if contract.get("schema_version") != "p9016-next-step-r2-release-v1":
        raise EntryError("unexpected release contract schema")
    scope = config.get("scope", {})
    expected_scope = {
        "raw_contact_count": 1703888,
        "chromosome_count": 20,
        "track_count": 40,
        "physical_beads_1mb": 5290,
        "real_endpoint_count": 8,
        "real_selected_endpoint_count": 4,
        "synthetic_endpoint_count": 6,
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            raise EntryError("scope %s changed: %r" % (key, scope.get(key)))
    training = config.get("training_release", {})
    if training.get("backend") != "GPU" or training.get("scale2") is not True:
        raise EntryError("real training backend/scale2 contract changed")
    if training.get("b1_nfev_by_layer") != {"5mb": 306, "2mb": 202, "1mb": 243}:
        raise EntryError("B1 nfev freeze changed")
    if training.get("b2_nfev_by_layer") != {"5mb": 612, "2mb": 404, "1mb": 486}:
        raise EntryError("B2 nfev freeze changed")
    if training.get("synthetic_1mb_fgcap") != 243 or training.get("scipy_gtol") != 0.0:
        raise EntryError("synthetic FG cap or gtol freeze changed")
    if (training.get("common_ftol") != 1e-10 or training.get("common_maxls") != 20
            or training.get("shared_ftol_or_maxls") is not True):
        raise EntryError("shared ftol/maxls freeze changed")
    selection = training.get("real_selection", {})
    if (selection.get("criterion") != "final count_nll_per_record"
            or selection.get("tie_tolerance") != SELECTION_TIE_TOL
            or selection.get("tie_order") != list(SOURCES)
            or selection.get("reference_read") is not False
            or selection.get("r2_read") is not False
            or selection.get("phase_read") is not False):
        raise EntryError("real train-only selection contract changed")
    evaluation = protocol.get("evaluation_definition", {})
    if evaluation.get("real_common_mask", {}).get("new_endpoint_inclusion") is not False:
        raise EntryError("new endpoints must not enter the common mask")
    if evaluation.get("two_copy_policy", {}).get("tie_tolerance") != TIE_TOL:
        raise EntryError("R2 tie tolerance changed")
    if evaluation.get("prohibited_outputs") != ["R1", "R3", "p-values", "confidence intervals", "R2-based model/endpoint/hyperparameter selection"]:
        raise EntryError("prohibited R2 outputs changed")
    mask_spec = config.get("mask_lock", {})
    mask_path = _resolve(mask_spec["lock_path"])
    mask_lock_hash = _hash_expected(mask_path, mask_spec.get("lock_sha256"), "mask_lock")
    manifest_path = _resolve(mask_spec["manifest_path"])
    manifest_hash = _hash_expected(manifest_path, mask_spec.get("manifest_sha256"), "published_mask_manifest")
    lock = _read_json(mask_path)
    expected = lock.get("expected_by_chromosome", [])
    if lock.get("condition_count") != 21 or lock.get("new_endpoint_inclusion") is not False or len(expected) != 20:
        raise EntryError("old21 mask metadata changed")
    total_common = sum(int(item["n_common_pairs"]) for item in expected)
    total_pairs = sum(int(item["n_total_non_diagonal_pairs"]) for item in expected)
    if (total_common, total_pairs) != (157529, 176201):
        raise EntryError("old21 mask total counts changed")
    exposure_manifest = _read_json(EXPOSURE_MANIFEST_PATH)
    if (exposure_manifest.get("status") != "prepared_only"
            or exposure_manifest.get("coordinates_payload_read") is not False
            or exposure_manifest.get("reference_payload_read") is not False
            or exposure_manifest.get("optimizer_started") is not False
            or exposure_manifest.get("native_called") is not False):
        raise EntryError("exposure preparation boundary is not clean")
    worker_manifest = _read_json(WORKER_MANIFEST_PATH)
    if worker_manifest.get("truth_coordinates_exposed") is not False or worker_manifest.get("optimizer_started") is not False:
        raise EntryError("worker-facing exposure manifest boundary changed")
    worker_serialized = json.dumps(worker_manifest, sort_keys=True).lower()
    if "eval_truth" in worker_serialized or "truth_npz" in worker_serialized:
        raise EntryError("worker-facing manifest exposes truth path")
    exposure_checks = {}
    for fixture_id in ("P2", "N2"):
        record = exposure_manifest.get("fixtures", {}).get(fixture_id)
        if not isinstance(record, Mapping):
            raise EntryError("missing exposure record: %s" % fixture_id)
        audit = record.get("audit", {})
        if (audit.get("length") != 2645 or audit.get("dtype") != "<f8"
                or audit.get("finite") is not True or audit.get("strictly_positive") is not True
                or audit.get("matches_generation_metadata") is not True
                or audit.get("matches_frozen_truth_manifest_exposure_sha256") is not True):
            raise EntryError("exposure audit failed: %s" % fixture_id)
        output_hashes = {}
        for kind in ("npy", "npz", "json"):
            item = record.get("outputs", {}).get(kind, {})
            path = _resolve(item.get("path"))
            output_hashes[kind] = _hash_expected(path, item.get("sha256"), "%s_exposure_%s" % (fixture_id, kind))
        exposure_checks[fixture_id] = {"audit": dict(audit), "output_hashes": output_hashes}
    dependency_checks = {}
    for key in ("synthetic_evaluator", "r2comparison", "real_mask_adapter", "legacy_coordinate_parser"):
        path = _resolve(config["dependencies"][key + "_path"])
        dependency_checks[key] = _hash_expected(path, config["dependencies"][key + "_sha256"], key)
    result = {
        "schema_version": "p9016-next-step-r2-preparation-validation-v1",
        "status": "PASS",
        "prepared_only": True,
        "evaluation_not_run": True,
        "candidate_coordinates_opened": 0,
        "synthetic_truth_coordinates_opened": False,
        "reference_opened": False,
        "phase_payload_opened": False,
        "fit_called": False,
        "native_called": False,
        "config_sha256": _sha256(CONFIG_PATH),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "release_contract_sha256": _sha256(CONTRACT_PATH),
        "entry_sha256": _sha256(PREP_DIR / "r2_entry.py"),
        "plotter_sha256": _sha256(PREP_DIR / "plot_next_step_r2.py"),
        "mask_lock": mask_lock_hash,
        "published_mask_manifest": manifest_hash,
        "mask_condition_count": 21,
        "mask_total_common_pairs": total_common,
        "mask_total_non_diagonal_pairs": total_pairs,
        "exposure_checks": exposure_checks,
        "dependency_checks": dependency_checks,
        "real_endpoint_count": 8,
        "synthetic_endpoint_count": 6,
        "real_selected_endpoint_count": 4,
        "source_order": list(SOURCES),
        "b1_nfev_by_layer": training["b1_nfev_by_layer"],
        "b2_nfev_by_layer": training["b2_nfev_by_layer"],
        "synthetic_1mb_fgcap": training["synthetic_1mb_fgcap"],
        "accepted_callback_gradient_inf_max": 1e-6,
        "scipy_gtol": 0.0,
        "common_ftol": 1e-10,
        "common_maxls": 20,
        "shared_ftol_or_maxls": True,
        "legacy_suite_status": {
            "status": "nonblocking_expected_failure",
            "command": "PYTHONPATH=. pytest -q tests/test_allele_calibration_evaluator.py docs/audits/multires-r2-preparation-20260914_041826/test_multires_r2.py",
            "passed": 13,
            "failed": 1,
            "test": "FrozenContractTests.test_pending_release_is_rejected_before_payload_access",
            "assertion": "assertRaises(r2.PreparationError) did not raise because the historical 041826 release contract is no longer pending/rejected",
            "scope": "legacy 041826 expectation only; no next-step code or old files changed",
        },
        "old_cli_warning": "pr.allele_calibration evaluate remains the 026 4-fixture x 5-variant pooled-swap entry; do not invoke it for this 2-fixture x 3-arm scope",
    }
    source_hashes = {
        "schema_version": "p9016-next-step-r2-preparation-source-hashes-v1",
        "config": {"path": str(CONFIG_PATH), "sha256": _sha256(CONFIG_PATH)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": _sha256(PROTOCOL_PATH)},
        "release_contract": {"path": str(CONTRACT_PATH), "sha256": _sha256(CONTRACT_PATH)},
        "entry": {"path": str(PREP_DIR / "r2_entry.py"), "sha256": _sha256(PREP_DIR / "r2_entry.py")},
        "plotter": {"path": str(PREP_DIR / "plot_next_step_r2.py"), "sha256": _sha256(PREP_DIR / "plot_next_step_r2.py")},
        "dependencies": dependency_checks,
        "old_files_modified": False,
    }
    _write_json(PREP_DIR / "preparation_source_hashes.json", source_hashes)
    result["preparation_source_hashes_path"] = "preparation_source_hashes.json"
    result["preparation_source_hashes_sha256"] = _sha256(PREP_DIR / "preparation_source_hashes.json")
    _write_json(PREP_DIR / "preparation_validation.json", result)
    _write_json(PREP_DIR / "legacy_suite_status.json", result["legacy_suite_status"])
    terminal = {
        "schema_version": "p9016-next-step-r2-preparation-terminal-v1",
        "status": "prepared_only",
        "exit_code": 0,
        "fit_called": False,
        "native_called": False,
        "candidate_coordinates_opened": 0,
        "synthetic_truth_coordinates_opened": False,
        "reference_opened": False,
        "exposure_key_only": True,
        "exposure_source_members_listed_without_coordinates_payload_read": True,
        "validation_path": "preparation_validation.json",
        "validation_sha256": _sha256(PREP_DIR / "preparation_validation.json"),
        "source_hashes_path": "preparation_source_hashes.json",
        "source_hashes_sha256": _sha256(PREP_DIR / "preparation_source_hashes.json"),
    }
    _write_json(PREP_DIR / "preparation_terminal_evidence.json", terminal)
    return result


class _TinyTemplate:
    chromosome_names = ("chr1", "chr2")

    def chromosome_slice(self, chromosome: int) -> slice:
        return slice(8 * chromosome, 8 * (chromosome + 1))


def _tiny_line(values: Sequence[float], offset: float = 0.0) -> np.ndarray:
    coordinates = np.zeros((len(values), 3), dtype=np.float64)
    coordinates[:, 0] = np.asarray(values, dtype=np.float64)
    coordinates[:, 1] = offset
    return coordinates


def run_tiny_fixture() -> dict[str, Any]:
    """使用不包含项目 candidate 或 truth payload 的 calibration fixture 检查 adapter contract。"""
    template = _TinyTemplate()
    mat = np.concatenate([_tiny_line([0, 1, 3, 6, 10, 15, 21, 28]), _tiny_line([0, 1, 3, 6, 10, 15, 21, 28], 10)])
    pat = np.concatenate([_tiny_line([0, 2, 5, 9, 14, 20, 27, 35]), _tiny_line([0, 2, 5, 9, 14, 20, 27, 35], 20)])
    truth_p = np.stack([mat, pat])
    candidate_direct = np.stack([mat, pat])
    candidate_global_swap = np.stack([pat, mat])
    per_chromosome_direct = [
        synthetic_evaluator.chromosome_metrics(template, candidate_direct, truth_p, same_shape=False, chromosome=index)
        for index in range(2)
    ]
    per_chromosome_swap = [
        synthetic_evaluator.chromosome_metrics(template, candidate_global_swap, truth_p, same_shape=False, chromosome=index)
        for index in range(2)
    ]
    independent_swap_invariance = []
    for index in range(2):
        left = per_chromosome_direct[index]
        right = per_chromosome_swap[index]
        checks = {}
        for key in ("matched", "cross", "contrast", "minmargin"):
            checks[key] = bool(_finite(left[key]) and _finite(right[key]) and abs(float(left[key]) - float(right[key])) <= 1e-12)
        checks["candidate_only_shape_error_zero"] = bool(
            left["shape_error"]["mean"] == 0.0 and right["shape_error"]["mean"] == 0.0
        )
        independent_swap_invariance.append({"chromosome": template.chromosome_names[index], "checks": checks})
    truth_n = np.stack([mat, mat])
    initial_n = np.stack([mat, mat])
    final_n = np.stack([mat, pat])
    initial_records = [
        synthetic_evaluator.chromosome_metrics(template, initial_n, truth_n, same_shape=True, chromosome=index)
        for index in range(2)
    ]
    final_record = synthetic_evaluator.chromosome_metrics(template, final_n, truth_n, same_shape=True, chromosome=0)
    null_rho_equal = all(
        _finite(final_record["raw_four_rho"][key_a])
        and _finite(final_record["raw_four_rho"][key_b])
        and abs(float(final_record["raw_four_rho"][key_a]) - float(final_record["raw_four_rho"][key_b])) <= 1e-12
        for key_a, key_b in (("rho_A_mat", "rho_A_pat"), ("rho_B_mat", "rho_B_pat"))
    )
    n_only_initial = [record["N_only_copy_difference_rms"] for record in initial_records]
    n_only_final = final_record["N_only_copy_difference_rms"]
    n_only_diagnostic = bool(all(value == 0.0 for value in n_only_initial) and _finite(n_only_final) and n_only_final > 0.0)
    null_contrast_zero = bool(final_record["contrast"] == 0.0)
    tied_candidate = np.stack([mat, mat])
    tie_record = synthetic_evaluator.chromosome_metrics(template, tied_candidate, truth_p, same_shape=False, chromosome=0)
    tie_symmetric = bool(
        tie_record["orientation"] == "unresolved_tie"
        and tie_record["contrast"] == 0.0
        and tie_record["margin_ref1_mat"] is None
        and tie_record["margin_ref2_pat"] is None
    )
    constant_candidate = np.stack([np.zeros_like(mat), pat])
    constant_swapped = np.stack([pat, np.zeros_like(mat)])
    na_left = synthetic_evaluator.chromosome_metrics(template, constant_candidate, truth_p, same_shape=False, chromosome=0)
    na_right = synthetic_evaluator.chromosome_metrics(template, constant_swapped, truth_p, same_shape=False, chromosome=0)
    na_count_left = sum(value is None for value in na_left["raw_four_rho"].values())
    na_count_right = sum(value is None for value in na_right["raw_four_rho"].values())
    na_symmetric = bool(na_count_left == na_count_right and na_count_left > 0 and na_left["orientation"] == "unresolved_missing" and na_right["orientation"] == "unresolved_missing")
    result = {
        "schema_version": "p9016-next-step-r2-tiny-fixture-v1",
        "status": "PASS" if (
            all(all(item["checks"].values()) for item in independent_swap_invariance)
            and null_rho_equal and null_contrast_zero and n_only_diagnostic
            and tie_symmetric and na_symmetric
        ) else "FAIL",
        "project_payloads_opened": [],
        "real_reference_opened": False,
        "synthetic_truth_coordinates_opened": False,
        "independent_per_chromosome_swap": independent_swap_invariance,
        "candidate_only_shape_error_zero": True,
        "N2_null": {
            "rho_columns_equal": null_rho_equal,
            "contrast_zero": null_contrast_zero,
            "same_x0_initial_N_only": n_only_initial,
            "final_N_only": n_only_final,
            "different_structure_diagnostic": n_only_diagnostic,
            "N_only_is_not_a_positive_score": True,
        },
        "tie": {"symmetric": tie_symmetric, "orientation": tie_record["orientation"]},
        "n_a": {"symmetric": na_symmetric, "left_count": na_count_left, "right_count": na_count_right},
        "metric_contract": {
            "whole_chromosome_swap": True,
            "fixed_reference_columns": ["mat", "pat"],
            "tie_tolerance": TIE_TOL,
            "minimum_rho_pairs": 20,
        },
    }
    _write_json(PREP_DIR / "tiny_fixture_validation.json", result)
    return result


def _endpoint_record_map(release: Mapping[str, Any], key: str, expected_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    rows = release.get(key)
    if not isinstance(rows, list) or [str(item.get("id")) for item in rows] != list(expected_ids):
        raise EntryError("%s must contain the frozen endpoint order" % key)
    result = {}
    for item in rows:
        if not isinstance(item, Mapping):
            raise EntryError("non-object %s record" % key)
        endpoint_id = str(item.get("id"))
        if endpoint_id in result:
            raise EntryError("duplicate endpoint: %s" % endpoint_id)
        result[endpoint_id] = dict(item)
    return result


def validate_release_metadata(release_path: str | Path) -> dict[str, Any]:
    """只验证 parent JSON；不读取 candidate/reference payload。"""
    release_file = Path(release_path).resolve()
    release = _read_json(release_file)
    if release.get("schema_version") != "p9016-next-step-r2-release-v1":
        raise EntryError("parent release schema mismatch")
    if release.get("status") not in {"released", "locked_for_evaluation"}:
        raise EntryError("parent release is not locked/released")
    config_ref = release.get("prepared_config", {})
    protocol_ref = release.get("protocol", {})
    if config_ref.get("path") != "config.json" or protocol_ref.get("path") != "r2_protocol.json":
        raise EntryError("release must bind this preparation config/protocol")
    if _require_sha(config_ref.get("sha256"), "prepared_config.sha256") != _sha256(CONFIG_PATH):
        raise EntryError("prepared config SHA mismatch")
    if _require_sha(protocol_ref.get("sha256"), "protocol.sha256") != _sha256(PROTOCOL_PATH):
        raise EntryError("protocol SHA mismatch")
    real = _endpoint_record_map(release, "real_endpoints", REAL_ENDPOINT_IDS)
    synthetic = _endpoint_record_map(release, "synthetic_endpoints", SYNTHETIC_ENDPOINT_IDS)
    for endpoint_id, endpoint in {**real, **synthetic}.items():
        if endpoint.get("coordinate_status") != "available":
            raise EntryError("endpoint is not available: %s" % endpoint_id)
        _require_sha(endpoint.get("coordinate_sha256"), endpoint_id + ".coordinate_sha256")
        if not endpoint.get("coordinate_path"):
            raise EntryError("missing coordinate path: %s" % endpoint_id)
        if endpoint.get("n_copies") != 2:
            raise EntryError("endpoint is not two-copy: %s" % endpoint_id)
        _require_sha(endpoint.get("terminal_record_sha256"), endpoint_id + ".terminal_record_sha256")
        if not endpoint.get("terminal_record_path"):
            raise EntryError("missing terminal record path: %s" % endpoint_id)
    for endpoint_id, endpoint in real.items():
        if endpoint.get("backend") != "GPU" or endpoint.get("method_id") not in REAL_METHODS or endpoint.get("budget_id") not in REAL_BUDGETS or endpoint.get("source_id") not in SOURCES:
            raise EntryError("real endpoint identity/backend mismatch: %s" % endpoint_id)
        if not _finite(endpoint.get("final_count_nll_per_record")):
            raise EntryError("real endpoint lacks finite final count_nll_per_record: %s" % endpoint_id)
    for endpoint_id, endpoint in synthetic.items():
        if endpoint.get("fixture_id") not in {"P2", "N2"} or endpoint.get("method_id") not in REAL_METHODS or endpoint.get("exposure_policy") not in {"production-e", "known-e", "known-generating-e"}:
            raise EntryError("synthetic endpoint identity mismatch: %s" % endpoint_id)
        if endpoint.get("coordinate_format") not in {"npz", "3dg", "3dg_text", "native_tsv", "coords_tsv", "npz_or_3dg"}:
            raise EntryError("unsupported synthetic coordinate format: %s" % endpoint_id)
    source_locks = release.get("source_locks")
    if not isinstance(source_locks, list) or [item.get("source_id") for item in source_locks] != list(SOURCES):
        raise EntryError("source locks must follow frozen order")
    for item in source_locks:
        for key in ("raw_source_path", "snapshot_path"):
            if not item.get(key):
                raise EntryError("missing source lock path: %s/%s" % (item.get("source_id"), key))
        for key in ("raw_source_sha256", "snapshot_sha256"):
            _require_sha(item.get(key), "source.%s.%s" % (item.get("source_id"), key))
    controller = release.get("training_controller", {})
    if (controller.get("status") != "complete"
            or controller.get("backend") != "GPU"
            or controller.get("real_endpoint_count") != 8
            or controller.get("synthetic_endpoint_count") != 6
            or controller.get("total_fit_count") != 14
            or controller.get("active_jobs") != []
            or controller.get("reference_used") is not False
            or controller.get("phase_used") is not False):
        raise EntryError("training terminal/controller gate is incomplete")
    for key in ("terminal_evidence_path", "terminal_evidence_sha256", "source_manifest_path", "source_manifest_sha256"):
        if not controller.get(key):
            raise EntryError("training controller lacks %s" % key)
    _require_sha(controller.get("terminal_evidence_sha256"), "training_controller.terminal_evidence_sha256")
    _require_sha(controller.get("source_manifest_sha256"), "training_controller.source_manifest_sha256")
    selection_rows = release.get("selection_by_method_budget")
    expected_selection_ids = ["M0-B1", "M0-B2", "M1-B1", "M1-B2"]
    if not isinstance(selection_rows, list) or ["%s-%s" % (item.get("method_id"), item.get("budget_id")) for item in selection_rows] != expected_selection_ids:
        raise EntryError("selection rows do not cover four method-budget cells")
    for item in selection_rows:
        candidate_ids = item.get("candidate_endpoint_ids")
        expected_ids = ["%s-%s-%s" % (item.get("method_id"), item.get("budget_id"), source) for source in SOURCES]
        if candidate_ids != expected_ids:
            raise EntryError("selection candidate order changed: %s" % item)
        selected_id = item.get("selected_endpoint_id")
        if selected_id not in expected_ids or item.get("selected_source_id") not in SOURCES:
            raise EntryError("selection endpoint/source missing: %s" % item)
        if selected_id.rsplit("-", 1)[-1] != item.get("selected_source_id"):
            raise EntryError("selection source disagrees with selected endpoint: %s" % item)
        if item.get("criterion") != "final count_nll_per_record" or item.get("tie_tolerance") != SELECTION_TIE_TOL or item.get("tie_order") != list(SOURCES):
            raise EntryError("selection criterion/tie changed")
    selection_evidence = release.get("selection_evidence", {})
    if selection_evidence.get("format") != "gpu-m1-formal-selection-v1" or not selection_evidence.get("path"):
        raise EntryError("selection evidence path/format missing")
    _require_sha(selection_evidence.get("sha256"), "selection_evidence.sha256")
    proof = release.get("train_only_selection_proof", {})
    if (proof.get("status") != "complete"
            or proof.get("criterion") != "final count_nll_per_record"
            or proof.get("within_method_budget_only") is not True
            or proof.get("reference_read") is not False
            or proof.get("r2_read") is not False
            or proof.get("phase_read") is not False):
        raise EntryError("train-only selection proof is invalid")
    _require_sha(proof.get("evidence_sha256"), "train_only_selection_proof.evidence_sha256")
    if not proof.get("evidence_path"):
        raise EntryError("selection proof evidence path missing")
    mask = release.get("mask_lock", {})
    if mask.get("path") != "docs/audits/multires-r2-preparation-20260914_041826/mask_lock.json" or mask.get("sha256") != MASK_SHA256 or mask.get("published_manifest_sha256") != MASK_MANIFEST_SHA256 or mask.get("new_endpoint_inclusion") is not False:
        raise EntryError("release mask lock does not bind old21 mask")
    reference = release.get("reference", {})
    if reference.get("path") != "data/P9016.1m.3dg.gz" or reference.get("sha256") != REFERENCE_SHA256 or reference.get("role") != "evaluation_only":
        raise EntryError("reference lock changed")
    anchor = release.get("historical_anchor", {})
    if (anchor.get("id") != "historical_036_C0_anchor"
            or anchor.get("condition_id") != "C0-random_joint_base2207"
            or anchor.get("coordinate_status") != "available"
            or anchor.get("n_copies") != 2
            or not anchor.get("coordinate_path")):
        raise EntryError("036 C0 historical anchor is incomplete")
    _require_sha(anchor.get("coordinate_sha256"), "historical_anchor.coordinate_sha256")
    flags = release.get("release_flags", {})
    required_flags = ("all_14_candidate_paths_locked", "all_14_candidate_hashes_locked", "all_source_hashes_locked", "all_terminal_hashes_locked", "all_selection_hashes_locked", "reference_read_allowed")
    if any(flags.get(key) is not True for key in required_flags) or flags.get("prepared_only") is not False:
        raise EntryError("release flags have not been armed")
    return {"release_file": release_file, "release": release, "real": real, "synthetic": synthetic}


def _hash_locked_inputs(release_info: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    release_file = Path(release_info["release_file"])
    release = release_info["release"]
    hashes: dict[str, Any] = {"candidate_files": [], "terminal_files": [], "sources": [], "code": {}, "mask": [], "evidence": []}
    all_endpoints = list(release_info["real"].values()) + list(release_info["synthetic"].values())
    for endpoint in all_endpoints:
        coord = _hash_expected(_resolve_release_path(endpoint["coordinate_path"], release_file), endpoint["coordinate_sha256"], endpoint["id"] + ".coordinate")
        terminal = _hash_expected(_resolve_release_path(endpoint["terminal_record_path"], release_file), endpoint["terminal_record_sha256"], endpoint["id"] + ".terminal")
        hashes["candidate_files"].append({"endpoint_id": endpoint["id"], **coord})
        hashes["terminal_files"].append({"endpoint_id": endpoint["id"], **terminal})
    anchor = release["historical_anchor"]
    anchor_hash = _hash_expected(_resolve_release_path(anchor["coordinate_path"], release_file), anchor["coordinate_sha256"], "historical_036_C0_anchor.coordinate")
    hashes["historical_anchor"] = {"id": anchor["id"], **anchor_hash}
    for source in release["source_locks"]:
        raw = _hash_expected(_resolve_release_path(source["raw_source_path"], release_file), source["raw_source_sha256"], "source.%s.raw" % source["source_id"])
        snapshot = _hash_expected(_resolve_release_path(source["snapshot_path"], release_file), source["snapshot_sha256"], "source.%s.snapshot" % source["source_id"])
        hashes["sources"].append({"source_id": source["source_id"], "raw": raw, "snapshot": snapshot})
    controller = release["training_controller"]
    hashes["evidence"].append(_hash_expected(_resolve_preparation_bound_path(controller["terminal_evidence_path"], release_file), controller["terminal_evidence_sha256"], "training_terminal_evidence"))
    hashes["evidence"].append(_hash_expected(_resolve_preparation_bound_path(controller["source_manifest_path"], release_file), controller["source_manifest_sha256"], "training_source_manifest"))
    selection_evidence = release["selection_evidence"]
    hashes["evidence"].append(_hash_expected(_resolve_preparation_bound_path(selection_evidence["path"], release_file), selection_evidence["sha256"], "selection_evidence"))
    proof = release["train_only_selection_proof"]
    hashes["evidence"].append(_hash_expected(_resolve_preparation_bound_path(proof["evidence_path"], release_file), proof["evidence_sha256"], "train_only_selection_proof"))
    for key, path_key, sha_key in (("prepared_config", "path", "sha256"), ("protocol", "path", "sha256")):
        hashes["evidence"].append(_hash_expected(_resolve_preparation_bound_path(release[key][path_key], release_file), release[key][sha_key], key))
    code = release.get("code_hashes", {})
    for name, path_key, sha_key in (("entry", "entry_path", "entry_sha256"), ("plotter", "plotter_path", "plotter_sha256"), ("synthetic_evaluator", "synthetic_evaluator_path", "synthetic_evaluator_sha256"), ("real_mask_adapter", "real_mask_adapter_path", "real_mask_adapter_sha256"), ("r2comparison", "r2comparison_path", "r2comparison_sha256"), ("legacy_coordinate_parser", "legacy_coordinate_parser_path", "legacy_coordinate_parser_sha256")):
        path = _resolve_preparation_bound_path(code[path_key], release_file)
        hashes["code"][name] = _hash_expected(path, code[sha_key], "code.%s" % name)
    old_config = OLD.read_json(OLD.CONFIG_PATH)
    old_mask_path, _old_manifest, mask_rows = OLD._mask_manifest_records(old_config)
    hashes["mask_manifest"] = _hash_expected(old_mask_path, MASK_MANIFEST_SHA256, "published_mask_manifest")
    for item in mask_rows:
        path = OLD.resolve_path(item["path"], old_mask_path.parent)
        hashes["mask"].append({"condition_id": item["condition_id"], **_hash_expected(path, item.get("sha256"), "mask.%s" % item["condition_id"])})
    hashes["mask_manifest_path"] = str(old_mask_path)
    hashes["mask_rows"] = mask_rows
    hashes["candidate_hashes_complete"] = len(hashes["candidate_files"]) == 14
    hashes["terminal_hashes_complete"] = len(hashes["terminal_files"]) == 14
    hashes["mask_hashes_complete"] = len(hashes["mask"]) == 21
    return hashes


def _hash_truth_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for fixture_id in ("P2", "N2"):
        spec = config["synthetic_truth"][fixture_id]
        truth_path = _resolve(spec["npz_path"])
        metadata_path = _resolve(spec["metadata_path"])
        result[fixture_id] = {
            "truth": _hash_expected(truth_path, spec["npz_sha256"], fixture_id + ".truth_npz"),
            "metadata": _hash_expected(metadata_path, spec["metadata_sha256"], fixture_id + ".truth_metadata"),
            "same_shape": bool(spec["same_shape"]),
            "truth_seed": spec["truth_seed"],
        }
    return result


def _load_mask_and_reference(config: Mapping[str, Any], locked: Mapping[str, Any], reference_path: Path, reference_hash: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    old_config = OLD.read_json(OLD.CONFIG_PATH)
    mask_lock = OLD._load_mask_lock(old_config)
    mask_structures = OLD.load_mask_structures(locked["mask_rows"], {"mask_hashes": locked["mask"]})
    reference = OLD.load_coordinates(reference_path, "3dg")
    masks = OLD.build_frozen_masks(old_config, mask_structures, reference, mask_lock)
    legacy_masks, legacy_structures, legacy_provenance = OLD._legacy_rebuild_masks_after_gate(
        old_config, {"mask_rows": locked["mask_rows"], "mask_hashes": locked["mask"]}
    )
    mask_exact = OLD.compare_mask_exact(masks, legacy_masks)
    if mask_exact.get("status") != "PASS":
        raise EntryError("old21 mask exact comparison failed")
    return masks, mask_structures, reference, {
        "mask_exact_comparison": mask_exact,
        "legacy_provenance": legacy_provenance,
        "reference_path": str(reference_path),
        "reference_sha256": reference_hash["sha256"],
    }


def _validate_real_structures(structures: Mapping[str, Mapping[int, np.ndarray]], endpoint_id: str) -> None:
    for chromosome_index, (chromosome, length_bp) in enumerate(OLD.CHROMOSOMES):
        positions = OLD.grid_positions(length_bp)
        for suffix in ("a", "b"):
            track = "c%02d%s" % (chromosome_index + 1, suffix)
            values = structures.get(track)
            if not isinstance(values, Mapping):
                raise EntryError("%s lacks track %s" % (endpoint_id, track))
            missing = [int(position) for position in positions if int(position) not in values]
            if missing:
                raise EntryError("%s lacks %d metric-grid positions on %s" % (endpoint_id, len(missing), chromosome))
            for position in positions:
                point = np.asarray(values[int(position)], dtype=np.float64)
                if point.shape != (3,) or not np.all(np.isfinite(point)):
                    raise EntryError("%s has nonfinite point %s:%d" % (endpoint_id, track, int(position)))


def _load_synthetic_candidate(path: Path, endpoint_id: str, template: Any) -> np.ndarray:
    coordinates = synthetic_prepare._candidate_file_coordinates(path, template)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (2, 2645, 3) or not np.all(np.isfinite(coordinates)):
        raise EntryError("%s synthetic candidate is not finite full-grid shape (2,2645,3)" % endpoint_id)
    return coordinates


def _real_rows(release_info: Mapping[str, Any], masks: Mapping[str, Mapping[str, Any]], release_hashes: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    release_file = Path(release_info["release_file"])
    rows: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, np.ndarray]] = {}
    by_endpoint = {item["endpoint_id"]: item for item in release_hashes["candidate_files"]}
    for endpoint_id in REAL_ENDPOINT_IDS:
        endpoint = release_info["real"][endpoint_id]
        path = _resolve_release_path(endpoint["coordinate_path"], release_file)
        format_name = str(endpoint.get("coordinate_format", "3dg"))
        structures = OLD.load_coordinates(path, format_name)
        _validate_real_structures(structures, endpoint_id)
        loaded[endpoint_id] = structures
        eval_endpoint = dict(endpoint)
        eval_endpoint.setdefault("variant_id", "%s-%s" % (endpoint["method_id"], endpoint["budget_id"]))
        eval_endpoint.setdefault("endpoint_status", "accepted")
        eval_endpoint["coordinate_status"] = "available"
        for chromosome, _length in OLD.CHROMOSOMES:
            row = OLD.evaluate_endpoint_chromosome(eval_endpoint, structures, masks[chromosome])
            row["cross_original"] = row.get("swapped")
            row.update({"method_id": endpoint["method_id"], "budget_id": endpoint["budget_id"], "source_id": endpoint["source_id"], "endpoint_sha256": by_endpoint[endpoint_id]["sha256"]})
            rows.append(row)
    return rows, loaded


def _anchor_rows(release_info: Mapping[str, Any], masks: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    release_file = Path(release_info["release_file"])
    anchor = release_info["release"]["historical_anchor"]
    structures = OLD.load_coordinates(_resolve_release_path(anchor["coordinate_path"], release_file), str(anchor.get("format", "3dg")))
    _validate_real_structures(structures, str(anchor["id"]))
    endpoint = {
        "id": str(anchor["id"]),
        "variant_id": "historical_036_C0",
        "source_id": "random_joint_base2207",
        "endpoint_status": "historical_anchor",
        "coordinate_status": "historical_anchor",
    }
    rows = []
    for chromosome, _length in OLD.CHROMOSOMES:
        row = OLD.evaluate_endpoint_chromosome(endpoint, structures, masks[chromosome])
        row["cross_original"] = row.get("swapped")
        row.update({"method_id": "historical", "budget_id": "C0", "source_id": "historical_036_C0", "endpoint_sha256": anchor["coordinate_sha256"]})
        rows.append(row)
    return rows


def _summary_rows(rows: Sequence[Mapping[str, Any]], endpoint_ids: Sequence[str], metrics: Sequence[str], kind: str) -> list[dict[str, Any]]:
    result = []
    for endpoint_id in endpoint_ids:
        selected = [row for row in rows if row.get("endpoint_id") == endpoint_id]
        for metric in metrics:
            values = [float(row[metric]) for row in selected if _finite(row.get(metric))]
            result.append({
                "kind": kind,
                "endpoint_id": endpoint_id,
                "metric": metric,
                "n_chromosomes": len(selected),
                "n_finite": len(values),
                "n_a": len(selected) - len(values),
                "mean": None if not values else float(np.mean(values)),
                "median": None if not values else float(np.median(values)),
                "positive": int(sum(value > 0.0 for value in values)),
                "negative": int(sum(value < 0.0 for value in values)),
                "zero": int(sum(value == 0.0 for value in values)),
            })
    return result


def _paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["endpoint_id"], row["chromosome"]): row for row in rows}
    result = []
    for source in SOURCES:
        for budget in REAL_BUDGETS:
            left_id = "M1-%s-%s" % (budget, source)
            right_id = "M0-%s-%s" % (budget, source)
            for chromosome, _length in OLD.CHROMOSOMES:
                left = by_key[(left_id, chromosome)]
                right = by_key[(right_id, chromosome)]
                for metric in REAL_METRICS:
                    left_value, right_value = left.get(metric), right.get(metric)
                    result.append({
                        "comparison": "M1-M0",
                        "source_id": source,
                        "budget_id": budget,
                        "method_id": "M1_vs_M0",
                        "left_endpoint_id": left_id,
                        "right_endpoint_id": right_id,
                        "chromosome": chromosome,
                        "metric": metric,
                        "delta": float(left_value) - float(right_value) if _finite(left_value) and _finite(right_value) else None,
                        "status": "ok" if _finite(left_value) and _finite(right_value) else "n/a",
                    })
        for method in REAL_METHODS:
            left_id = "%s-B2-%s" % (method, source)
            right_id = "%s-B1-%s" % (method, source)
            for chromosome, _length in OLD.CHROMOSOMES:
                left = by_key[(left_id, chromosome)]
                right = by_key[(right_id, chromosome)]
                for metric in REAL_METRICS:
                    left_value, right_value = left.get(metric), right.get(metric)
                    result.append({
                        "comparison": "double-base",
                        "source_id": source,
                        "budget_id": "B2_minus_B1",
                        "method_id": method,
                        "left_endpoint_id": left_id,
                        "right_endpoint_id": right_id,
                        "chromosome": chromosome,
                        "metric": metric,
                        "delta": float(left_value) - float(right_value) if _finite(left_value) and _finite(right_value) else None,
                        "status": "ok" if _finite(left_value) and _finite(right_value) else "n/a",
                    })
    return result


def _paired_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    total: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        key = (str(row["comparison"]), str(row["source_id"]), str(row["budget_id"]), str(row["method_id"]), str(row["metric"]))
        total[key] = total.get(key, 0) + 1
        if _finite(row.get("delta")):
            grouped.setdefault(key, []).append(float(row["delta"]))
    result = []
    for key in sorted(total):
        values = grouped.get(key, [])
        result.append({
            "comparison": key[0], "source_id": key[1], "budget_id": key[2], "method_id": key[3], "metric": key[4],
            "n_chromosomes": total[key], "n_finite": len(values), "n_a": total[key] - len(values),
            "mean": None if not values else float(np.mean(values)), "median": None if not values else float(np.median(values)),
            "positive": int(sum(value > 0.0 for value in values)), "negative": int(sum(value < 0.0 for value in values)), "zero": int(sum(value == 0.0 for value in values)),
        })
    return result


def _synthetic_rows(release_info: Mapping[str, Any], config: Mapping[str, Any], truth_hashes: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    template = synthetic_prepare.header_templates()[synthetic_prepare.FINAL_BIN]
    release_file = Path(release_info["release_file"])
    truths: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    candidate_coords: dict[str, np.ndarray] = {}
    for fixture_id in ("P2", "N2"):
        truth_path = _resolve(config["synthetic_truth"][fixture_id]["npz_path"])
        truth, _exposure, truth_metadata = synthetic_prepare._load_truth(truth_path)
        truth = np.asarray(truth, dtype=np.float64)
        if truth.shape != (2, 2645, 3) or not np.all(np.isfinite(truth)):
            raise EntryError("synthetic truth shape/finite check failed: %s" % fixture_id)
        if str(truth_metadata.get("fixture_id")) != fixture_id:
            raise EntryError("synthetic truth metadata fixture mismatch: %s" % fixture_id)
        truths[fixture_id] = truth
        same_shape = bool(config["synthetic_truth"][fixture_id]["same_shape"])
        for endpoint_id in SYNTHETIC_ENDPOINT_IDS:
            if not endpoint_id.startswith(fixture_id + "-"):
                continue
            endpoint = release_info["synthetic"][endpoint_id]
            path = _resolve_release_path(endpoint["coordinate_path"], release_file)
            coords = _load_synthetic_candidate(path, endpoint_id, template)
            candidate_coords[endpoint_id] = coords
            for chromosome in range(20):
                record = synthetic_evaluator.chromosome_metrics(template, coords, truth, same_shape=same_shape, chromosome=chromosome)
                output = {
                    "endpoint_id": endpoint_id,
                    "fixture_id": fixture_id,
                    "method_id": endpoint["method_id"],
                    "exposure_policy": endpoint["exposure_policy"],
                    "chromosome": template.chromosome_names[chromosome],
                    "chromosome_index": chromosome,
                    "n_offdiag_pairs_total": record["n_offdiag_pairs_total"],
                    "n_offdiag_pairs_finite_common_mask": record["n_offdiag_pairs_finite_common_mask"],
                    "same_finite_common_mask": record["same_finite_common_mask"],
                    "common_truth_matrix": record["common_truth_matrix"],
                    "rho_A_mat": record["raw_four_rho"]["rho_A_mat"],
                    "rho_A_pat": record["raw_four_rho"]["rho_A_pat"],
                    "rho_B_mat": record["raw_four_rho"]["rho_B_mat"],
                    "rho_B_pat": record["raw_four_rho"]["rho_B_pat"],
                    "matched": record["matched"],
                    "cross": record["cross"],
                    "contrast": record["contrast"],
                    "margin_mat": record["margin_ref1_mat"],
                    "margin_pat": record["margin_ref2_pat"],
                    "minmargin": record["minmargin"],
                    "orientation": record["orientation"],
                    "geometry_tie": record["geometry_tie"],
                    "shape_error_mean_auxiliary": record["shape_error"]["mean"],
                    "shape_error_max_auxiliary": record["shape_error"]["max"],
                    "N_only_copy_difference_rms": record["N_only_copy_difference_rms"],
                    "N_only_is_not_a_positive_score": True,
                }
                rows.append(output)
    # N2 same-x0 诊断有意不进行第二次 R2 评分。
    n_only_rows: list[dict[str, Any]] = []
    worker_manifest = _read_json(WORKER_MANIFEST_PATH)
    for fixture_id in ("N2",):
        start_path = _resolve(worker_manifest["fixtures"][fixture_id]["shared_start"]["path"])
        start = synthetic_prepare.load_paired_start(start_path).coordinates
        if start.shape != (2, 2645, 3) or not np.all(np.isfinite(start)):
            raise EntryError("N2 shared start shape/finite check failed")
        for chromosome in range(20):
            start_record = synthetic_evaluator.chromosome_metrics(template, start, truths[fixture_id], same_shape=True, chromosome=chromosome)
            initial = start_record["N_only_copy_difference_rms"]
            for arm in ("M0-production-e", "M1-production-e", "M0-known-e"):
                endpoint_id = fixture_id + "-" + arm
                final_record = next(item for item in rows if item["endpoint_id"] == endpoint_id and item["chromosome_index"] == chromosome)
                n_only_rows.append({
                    "fixture_id": fixture_id,
                    "comparison": "same_x0_initial_to_final",
                    "arm": arm,
                    "chromosome": template.chromosome_names[chromosome],
                    "initial_n_only_copy_difference_rms": initial,
                    "final_n_only_copy_difference_rms": final_record["N_only_copy_difference_rms"],
                    "final_minus_initial": None if not (_finite(initial) and _finite(final_record["N_only_copy_difference_rms"])) else float(final_record["N_only_copy_difference_rms"]) - float(initial),
                    "N_only_is_not_a_positive_score": True,
                })
            m0 = next(item for item in rows if item["endpoint_id"] == "N2-M0-production-e" and item["chromosome_index"] == chromosome)
            m1 = next(item for item in rows if item["endpoint_id"] == "N2-M1-production-e" and item["chromosome_index"] == chromosome)
            n_only_rows.append({
                "fixture_id": fixture_id,
                "comparison": "M1-M0",
                "arm": "M1-production-e_minus_M0-production-e",
                "chromosome": template.chromosome_names[chromosome],
                "initial_n_only_copy_difference_rms": None,
                "final_n_only_copy_difference_rms": None,
                "final_minus_initial": None if not (_finite(m0["N_only_copy_difference_rms"]) and _finite(m1["N_only_copy_difference_rms"])) else float(m1["N_only_copy_difference_rms"]) - float(m0["N_only_copy_difference_rms"]),
                "N_only_is_not_a_positive_score": True,
            })
    return rows, candidate_coords, n_only_rows


def _check_n2_null(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n2_rows = [row for row in rows if row.get("fixture_id") == "N2"]
    contrast_nonzero = []
    rho_mismatch = []
    for row in n2_rows:
        if _finite(row.get("contrast")) and abs(float(row["contrast"])) > 1e-12:
            contrast_nonzero.append(row["endpoint_id"] + ":" + row["chromosome"])
        for left, right in (("rho_A_mat", "rho_A_pat"), ("rho_B_mat", "rho_B_pat")):
            if _finite(row.get(left)) and _finite(row.get(right)) and abs(float(row[left]) - float(row[right])) > 1e-12:
                rho_mismatch.append(row["endpoint_id"] + ":" + row["chromosome"] + ":" + left)
    return {
        "status": "PASS" if not contrast_nonzero and not rho_mismatch else "FAIL",
        "n_rows": len(n2_rows),
        "contrast_nonzero": contrast_nonzero,
        "same_truth_rho_mismatches": rho_mismatch,
        "contrast_is_algebraically_zero": True,
        "N_only_copy_difference_is_not_a_positive_score": True,
    }


def evaluate_released(release_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise EntryError("evaluation output must be fresh and empty: %s" % output)
    config = _read_json(CONFIG_PATH)
    release_info = validate_release_metadata(release_path)
    output.mkdir(parents=True, exist_ok=False)
    locked = _hash_locked_inputs(release_info, config)
    truth_hashes = _hash_truth_inputs(config)
    # release/hash gate 证据会在任何 reference 或 truth
    # coordinate payload 打开前持久化。
    hash_gate = {
        "schema_version": "p9016-next-step-r2-hash-gate-v1",
        "status": "PASS",
        "candidate_count": len(locked["candidate_files"]),
        "terminal_count": len(locked["terminal_files"]),
        "source_count": len(locked["sources"]),
        "mask_input_count": len(locked["mask"]),
        "truth_archive_hash_count": len(truth_hashes),
        "reference_opened": False,
        "synthetic_truth_coordinates_opened": False,
        "candidate_r2_computed": False,
        "all_candidate_and_mask_hashes_passed": True,
        "all_release_evidence_hashes_passed": True,
        "release_path": str(Path(release_path).resolve()),
        "release_sha256": _sha256(Path(release_path).resolve()),
        "locked_inputs": locked,
        "truth_archive_hashes": truth_hashes,
    }
    _write_json(output / "hash_gate_evidence.json", hash_gate)
    reference_path = _resolve(config["reference"]["path"])
    reference_hash = _hash_expected(reference_path, config["reference"]["sha256"], "reference")
    _write_json(output / "reference_hash_gate.json", {
        "schema_version": "p9016-next-step-r2-reference-hash-gate-v1",
        "status": "PASS",
        "reference_path": str(reference_path),
        "reference_sha256": reference_hash["sha256"],
        "opened_before_hash": False,
        "opened_after_candidate_mask_source_terminal_selection_code_hashes": True,
        "payload_parsing_started": False,
    })
    masks, mask_structures, reference, mask_provenance = _load_mask_and_reference(config, locked, reference_path, reference_hash)
    real_rows, real_loaded = _real_rows(release_info, masks, locked)
    anchor_rows = _anchor_rows(release_info, masks)
    synthetic_rows, synthetic_loaded, n_only_rows = _synthetic_rows(release_info, config, truth_hashes)
    n2_check = _check_n2_null(synthetic_rows)
    if n2_check["status"] != "PASS":
        raise EntryError("N2 null algebra check failed")
    paired_rows = _paired_rows(real_rows)
    selected_ids = [item["selected_endpoint_id"] for item in release_info["release"]["selection_by_method_budget"]]
    real_summary = _summary_rows(real_rows, REAL_ENDPOINT_IDS, REAL_METRICS, "real")
    selected_summary = _summary_rows(real_rows, selected_ids, REAL_METRICS, "selected_real")
    anchor_summary = _summary_rows(anchor_rows, ["historical_036_C0_anchor"], REAL_METRICS, "historical_anchor")
    paired_summary = _paired_summary_rows(paired_rows)
    synthetic_summary = _summary_rows(synthetic_rows, SYNTHETIC_ENDPOINT_IDS, SYNTHETIC_METRICS, "synthetic")
    # 将旧 adapter rows 转换为带有新 scope fields 的稳定 endpoint table。
    real_columns = ["endpoint_id", "variant_id", "method_id", "budget_id", "source_id", "source_display", "chromosome", "chromosome_index", "endpoint_status", "coordinate_status", "n_copies", "n_bins", "n_total_non_diagonal_pairs", "n_common_pairs_frozen", "metric_status", "reason", "rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat", "direct", "swapped", "cross_original", "matched", "cross", "contrast", "margin_mat", "margin_pat", "minmargin", "orientation", "geometry_tie", "both_positive", "margin_class", "rho_reasons_json", "endpoint_sha256"]
    paired_columns = ["comparison", "source_id", "budget_id", "method_id", "left_endpoint_id", "right_endpoint_id", "chromosome", "metric", "delta", "status"]
    synthetic_columns = ["endpoint_id", "fixture_id", "method_id", "exposure_policy", "chromosome", "chromosome_index", "n_offdiag_pairs_total", "n_offdiag_pairs_finite_common_mask", "same_finite_common_mask", "common_truth_matrix", "rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat", "matched", "cross", "contrast", "margin_mat", "margin_pat", "minmargin", "orientation", "geometry_tie", "shape_error_mean_auxiliary", "shape_error_max_auxiliary", "N_only_copy_difference_rms", "N_only_is_not_a_positive_score"]
    summary_columns = ["kind", "endpoint_id", "metric", "n_chromosomes", "n_finite", "n_a", "mean", "median", "positive", "negative", "zero"]
    paired_summary_columns = ["comparison", "source_id", "budget_id", "method_id", "metric", "n_chromosomes", "n_finite", "n_a", "mean", "median", "positive", "negative", "zero"]
    n_only_columns = ["fixture_id", "comparison", "arm", "chromosome", "initial_n_only_copy_difference_rms", "final_n_only_copy_difference_rms", "final_minus_initial", "N_only_is_not_a_positive_score"]
    _write_tsv(output / "r2_real_8endpoint_x20chr.tsv", real_columns, real_rows)
    _write_tsv(output / "r2_real_historical_036_C0_anchor_x20chr.tsv", real_columns, anchor_rows)
    _write_tsv(output / "r2_real_paired_deltas.tsv", paired_columns, paired_rows)
    _write_tsv(output / "r2_real_paired_summaries.tsv", paired_summary_columns, paired_summary)
    _write_tsv(output / "r2_real_summary.tsv", summary_columns, real_summary + selected_summary + anchor_summary)
    _write_tsv(output / "r2_synthetic_6endpoint_x20chr.tsv", synthetic_columns, synthetic_rows)
    _write_tsv(output / "r2_synthetic_summary.tsv", summary_columns, synthetic_summary)
    _write_tsv(output / "synthetic_N2_n_only_diagnostic.tsv", n_only_columns, n_only_rows)
    result = {
        "schema_version": "p9016-next-step-r2-evaluation-v1",
        "status": "evaluation_complete",
        "scope": {
            "real_endpoint_count": 8,
            "real_rows": len(real_rows),
            "synthetic_endpoint_count": 6,
            "synthetic_rows": len(synthetic_rows),
            "historical_anchor_rows": len(anchor_rows),
            "chromosome_count": 20,
            "one_cell_linked_measurements_not_biological_replicates": True,
            "no_r1": True,
            "no_r3": True,
            "no_p_values": True,
            "no_confidence_intervals": True,
            "no_r2_based_selection": True,
            "fit_called": False,
            "native_called": False,
        },
        "release": {
            "path": str(Path(release_path).resolve()),
            "sha256": _sha256(Path(release_path).resolve()),
            "selection_by_method_budget": release_info["release"]["selection_by_method_budget"],
            "selected_real_endpoint_ids": selected_ids,
            "historical_anchor_id": "historical_036_C0_anchor",
        },
        "mask": mask_provenance,
        "hash_gate": {
            "path": "hash_gate_evidence.json",
            "sha256": _sha256(output / "hash_gate_evidence.json"),
            "reference_opened_after_gate": True,
            "synthetic_truth_opened_after_gate": True,
            "all_candidate_hashes_passed_before_r2": True,
        },
        "real_summary": real_summary + selected_summary + anchor_summary,
        "paired_summary": paired_summary,
        "synthetic_summary": synthetic_summary,
        "N2_null_check": n2_check,
        "N2_N_only_policy": {
            "diagnostic_only": True,
            "same_x0_initial_and_final_reported": True,
            "M1_minus_M0_reported": True,
            "not_a_positive_score": True,
        },
        "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "evaluation_results.json":
            result["files"][path.name] = _sha256(path)
    _write_json(output / "evaluation_results.json", result)
    # 有意在所有 primary R2 rows 和验证之后才调用绘图。
    from plot_next_step_r2 import render_plots  # 本地导入使 preparation 只处理 metadata
    plot_result = render_plots(output / "evaluation_results.json", output / "plots")
    result["plot_result"] = plot_result
    _write_json(output / "evaluation_results.json", result)
    _write_json(output / "terminal_evidence.json", {
        "schema_version": "p9016-next-step-r2-terminal-v1",
        "status": "evaluation_complete",
        "exit_code": 0,
        "candidate_hash_gate_passed": True,
        "reference_loaded_after_all_candidate_and_mask_hashes": True,
        "synthetic_truth_loaded_after_all_candidate_hashes": True,
        "real_rows": len(real_rows),
        "synthetic_rows": len(synthetic_rows),
        "anchor_rows": len(anchor_rows),
        "fit_called": False,
        "native_called": False,
        "r1_r3_pvalues_ci": False,
        "evaluation_results_sha256": _sha256(output / "evaluation_results.json"),
    })
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-preparation", action="store_true")
    parser.add_argument("--tiny-check", action="store_true")
    parser.add_argument("--validate-release", metavar="RELEASE_JSON")
    parser.add_argument("--evaluate-released", metavar="RELEASE_JSON")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    try:
        if args.validate_preparation:
            result = validate_preparation()
            print(json.dumps({"status": result["status"], "validation": str(PREP_DIR / "preparation_validation.json")}, sort_keys=True))
            return 0
        if args.tiny_check:
            result = run_tiny_fixture()
            print(json.dumps({"status": result["status"], "validation": str(PREP_DIR / "tiny_fixture_validation.json")}, sort_keys=True))
            return 0
        if args.validate_release:
            result = validate_release_metadata(args.validate_release)
            print(json.dumps({"status": "metadata_gate_pass", "release": str(result["release_file"])}, sort_keys=True))
            return 0
        if args.evaluate_released:
            if not args.output_dir:
                raise EntryError("--output-dir is required with --evaluate-released")
            result = evaluate_released(args.evaluate_released, args.output_dir)
            print(json.dumps({"status": result["status"], "output_dir": str(Path(args.output_dir).resolve()), "real_rows": result["scope"]["real_rows"], "synthetic_rows": result["scope"]["synthetic_rows"]}, sort_keys=True))
            return 0
        raise EntryError("choose one of --validate-preparation, --tiny-check, --validate-release, --evaluate-released")
    except (EntryError, OSError, ValueError, KeyError, AssertionError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
