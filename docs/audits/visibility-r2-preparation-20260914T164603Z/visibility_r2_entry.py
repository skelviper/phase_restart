"""Visibility-only R2 adapter with a strict pre-release boundary.

本模块只在 ``--evaluate-released`` 下读取候选、reference 或 synthetic truth。
``--validate-preparation`` 与 ``--population-kl-feasibility`` 只读取 JSON、源码和
冻结 mask metadata，不启动 optimizer/native，也不打开任何坐标 payload。
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import marshal
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PREP_DIR = Path(__file__).resolve().parent
ROOT = PREP_DIR.parents[2]
FROZEN_PR_DIR = PREP_DIR / "frozen_pr"
FROZEN_PR_PACKAGE = "visibility_frozen_pr"
CONFIG_PATH = PREP_DIR / "config.json"
PROTOCOL_PATH = PREP_DIR / "visibility_protocol.json"
CONTRACT_TEMPLATE_PATH = PREP_DIR / "release_contract.template.json"
SOURCE_SNAPSHOT_PATH = PREP_DIR / "source_snapshot.json"
FEASIBILITY_PATH = PREP_DIR / "population_kl_feasibility.json"
INTEGRATION_SCHEMA_PATH = PREP_DIR / "integration_schema_validation.json"
INTEGRATION_V2_VZ_PATH = PREP_DIR / "integration_v2_real_vz_schema_validation.json"
VALIDATION_PATH = PREP_DIR / "preparation_validation.json"
TERMINAL_EVIDENCE_PATH = PREP_DIR / "preparation_terminal_evidence.json"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

REAL_ENDPOINT_IDS = (
    "V0-consensus_joint_base1103",
    "V0-random_joint_base2207",
    "VZ-consensus_joint_base1103",
    "VZ-random_joint_base2207",
    "V1-consensus_joint_base1103",
    "V1-random_joint_base2207",
)
SYNTHETIC_ENDPOINT_IDS = (
    "P2-V0-production",
    "P2-V1-profile",
    "P2-V0-known-generating-e",
    "N2-V0-production",
    "N2-V1-profile",
    "N2-V0-known-generating-e",
)
METHOD_IDS = ("V0", "VZ", "V1")
SOURCE_IDS = ("consensus_joint_base1103", "random_joint_base2207")
SYNTHETIC_FIXTURES = ("P2", "N2")
R2_METRICS = ("matched", "cross", "contrast", "minmargin")
RAW_RHOS = ("rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat")
EXPECTED_MASK_TOTAL = 176201
EXPECTED_MASK_COMMON = 157529
GROUP_TOTALS = {"cis_offdiag": 696680, "inter": 568434}
GROUP_DENOMINATOR = sum(GROUP_TOTALS.values())
EXPECTED_SYNTHETIC_METADATA_SHA = {
    "P2": "88887f397829fc8f9abbd8e5615bebed48ee3bbc7076fc99cbd3a4ddfcff04b9",
    "N2": "35c2bda6c8004d29cefa4ff101a5c2fbb921ba485f01a199c0fa4cc2fa1b05c9",
}


class EntryError(RuntimeError):
    """准备或 release contract 不满足时抛出的错误。"""


class _SnapshotLoader(importlib.abc.Loader):
    """从实体快照执行模块，同时保留项目 root 的默认 ``__file__`` 语义。"""

    def __init__(self, source_path: Path, module_name: str) -> None:
        self.source_path = source_path
        self.module_name = module_name

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: Any) -> None:
        virtual_path = ROOT / "pr" / (self.module_name + ".py")
        module.__file__ = str(virtual_path)
        module.__package__ = FROZEN_PR_PACKAGE
        module.__loader__ = self
        try:
            source = self.source_path.read_bytes()
        except OSError as exc:
            raise EntryError("cannot read frozen pr snapshot: %s" % self.source_path) from exc
        code = compile(source, str(self.source_path), "exec")
        exec(code, module.__dict__)


class _SnapshotBytecodeLoader(importlib.abc.Loader):
    """执行旧冻结 helper 的 pyc 字节，同时保留虚拟项目 root。"""

    def __init__(self, bytecode_path: Path, module_name: str) -> None:
        self.bytecode_path = bytecode_path
        self.module_name = module_name

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: Any) -> None:
        virtual_path = ROOT / "pr" / (self.module_name + ".py")
        module.__file__ = str(virtual_path)
        module.__package__ = FROZEN_PR_PACKAGE
        module.__loader__ = self
        try:
            payload = self.bytecode_path.read_bytes()
            code = marshal.loads(payload[16:])
        except (OSError, EOFError, ValueError, TypeError) as exc:
            raise EntryError("cannot read frozen helper bytecode: %s" % self.bytecode_path) from exc
        if not isinstance(code, type(compile("", "", "exec"))):
            raise EntryError("frozen helper bytecode does not contain a code object: %s" % self.bytecode_path)
        exec(code, module.__dict__)


class _SnapshotFinder(importlib.abc.MetaPathFinder):
    """把 ``visibility_frozen_pr`` 的相对 import 固定到 prep 快照。"""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        prefix = FROZEN_PR_PACKAGE + "."
        if not fullname.startswith(prefix):
            return None
        module_name = fullname[len(prefix):]
        if not module_name or "." in module_name:
            return None
        source_path = FROZEN_PR_DIR / (module_name + ".py")
        if source_path.is_file():
            loader = _SnapshotLoader(source_path, module_name)
            return importlib.machinery.ModuleSpec(fullname, loader, origin=str(source_path))
        bytecode_path = FROZEN_PR_DIR / (module_name + ".pyc")
        if bytecode_path.is_file():
            loader = _SnapshotBytecodeLoader(bytecode_path, module_name)
            return importlib.machinery.ModuleSpec(fullname, loader, origin=str(bytecode_path))
        return None


def _ensure_snapshot_package() -> None:
    if not FROZEN_PR_DIR.is_dir():
        raise EntryError("frozen pr snapshot directory is missing: %s" % FROZEN_PR_DIR)
    if not any(isinstance(item, _SnapshotFinder) for item in sys.meta_path):
        sys.meta_path.insert(0, _SnapshotFinder())
    if FROZEN_PR_PACKAGE not in sys.modules:
        package_spec = importlib.machinery.ModuleSpec(FROZEN_PR_PACKAGE, loader=None, is_package=True)
        package = importlib.util.module_from_spec(package_spec)
        package.__file__ = str(ROOT / "pr" / "__init__.py")
        package.__path__ = [str(FROZEN_PR_DIR)]
        package.__package__ = FROZEN_PR_PACKAGE
        sys.modules[FROZEN_PR_PACKAGE] = package


def _load_snapshot_module(module_name: str) -> Any:
    """加载 prep 内实体快照，禁止回退到可变的 live ``pr`` 包。"""
    _ensure_snapshot_package()
    return importlib.import_module(FROZEN_PR_PACKAGE + "." + module_name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value) if math.isinf(value) else None
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EntryError("cannot read JSON metadata: %s" % target) from exc
    if not isinstance(value, Mapping):
        raise EntryError("JSON root is not an object: %s" % target)
    return dict(value)


def _write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise EntryError("cannot hash file: %s" % target) from exc
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise EntryError("missing or malformed SHA256: %s" % label)
    return value


def _resolve(value: str | Path, base: str | Path = ROOT) -> Path:
    target = Path(value)
    if target.is_absolute():
        return target.resolve()
    if target.parts and target.parts[0] in {"data", "docs", "inputs", "pr", "test_res", "native"}:
        return (ROOT / target).resolve()
    return (Path(base) / target).resolve()


def _resolve_contract(value: str | Path, contract_path: Path) -> Path:
    return _resolve(value, contract_path.parent)


def _hash_expected(path: Path, expected: Any, label: str) -> dict[str, Any]:
    declared = _require_sha(expected, label + ".declared")
    if not path.is_file():
        raise EntryError("missing locked file: %s" % path)
    actual = _sha256(path)
    if actual != declared:
        raise EntryError("SHA mismatch for %s: %s != %s" % (label, actual, declared))
    return {"path": str(path), "sha256": actual}


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _assert_close(left: Any, right: Any, tolerance: float, label: str) -> None:
    if not (_finite(left) and _finite(right)) or not math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=tolerance
    ):
        raise EntryError("%s differs: %r != %r" % (label, left, right))


def _load_static_files() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _read_json(CONFIG_PATH)
    protocol = _read_json(PROTOCOL_PATH)
    contract_template = _read_json(CONTRACT_TEMPLATE_PATH)
    source_snapshot = _read_json(SOURCE_SNAPSHOT_PATH)
    if config.get("schema_version") != "p9016-visibility-r2-config-v1":
        raise EntryError("visibility config schema mismatch")
    if protocol.get("schema_version") != "p9016-visibility-r2-protocol-v1":
        raise EntryError("visibility protocol schema mismatch")
    if contract_template.get("schema_version") != "p9016-visibility-r2-release-v1":
        raise EntryError("release contract template schema mismatch")
    if source_snapshot.get("schema_version") != "p9016-visibility-r2-source-snapshot-v1":
        raise EntryError("source snapshot schema mismatch")
    return config, protocol, contract_template, source_snapshot


def _validate_grid(config: Mapping[str, Any]) -> dict[str, Any]:
    grid = config.get("grid")
    if not isinstance(grid, Mapping):
        raise EntryError("config grid is missing")
    rows = grid.get("chromosomes")
    expected_names = ["chr%d" % index for index in range(1, 20)] + ["chrX"]
    if not isinstance(rows, list) or len(rows) != 20:
        raise EntryError("config grid must contain 20 chromosomes")
    names = [str(row.get("name")) for row in rows]
    if names != expected_names:
        raise EntryError("config chromosome order changed")
    lengths = [int(row.get("length_bp")) for row in rows]
    if any(length <= 0 for length in lengths):
        raise EntryError("config chromosome length is not positive")
    if int(grid.get("n_loci_1mb_full_grid")) != 2645:
        raise EntryError("full-grid locus count changed")
    if int(grid.get("n_eligible_cis_offdiag_pairs")) != 184016:
        raise EntryError("cis eligible pair count changed")
    if int(grid.get("n_eligible_inter_pairs")) != 3312674:
        raise EntryError("inter eligible pair count changed")
    return {"names": names, "lengths": lengths}


def _validate_source_snapshot(source_snapshot: Mapping[str, Any], *, allow_null_entry: bool = True) -> dict[str, Any]:
    rows = source_snapshot.get("files")
    if not isinstance(rows, list) or not rows:
        raise EntryError("source snapshot has no files")
    observed = []
    for item in rows:
        if not isinstance(item, Mapping):
            raise EntryError("source snapshot row is not an object")
        path = _resolve(str(item.get("path")), ROOT)
        declared = item.get("sha256")
        if declared is None and allow_null_entry and item.get("role") == "visibility_entry":
            if not path.is_file():
                raise EntryError("visibility entry is missing: %s" % path)
            observed.append({"path": str(path), "sha256": None, "role": item.get("role")})
            continue
        actual = _hash_expected(path, declared, "source.%s" % item.get("path"))
        observed.append({"path": str(path), "sha256": actual["sha256"], "role": item.get("role")})
    compatibility = source_snapshot.get("frozen_compatibility", {})
    if source_snapshot.get("snapshot_status") != "entity_bytes_frozen_before_parent_release":
        raise EntryError("source snapshot is not an entity freeze")
    import_policy = source_snapshot.get("import_policy", {})
    if import_policy.get("synthetic_modules") != "visibility_frozen_pr.allele_calibration and visibility_frozen_pr.allele_calibration_evaluator":
        raise EntryError("synthetic import policy is not frozen_pr")
    if import_policy.get("population_generation") != "visibility_frozen_pr.v1_calibration":
        raise EntryError("population generation import policy is not frozen_pr")
    if import_policy.get("live_pr_imports_for_formal_evaluation") is not False:
        raise EntryError("formal evaluator may import live pr")
    if compatibility.get("all_three_old_038_hashes_match") is not True:
        raise EntryError("old 038 frozen R2 dependency compatibility is not explicit")
    runtime = compatibility.get("old_038_r2comparison_runtime", {})
    python_version = "%d.%d.%d" % sys.version_info[:3]
    if runtime.get("artifact_type") != "CPython pyc" or runtime.get("python_version") != python_version:
        raise EntryError("old r2comparison runtime Python provenance changed")
    if runtime.get("magic_number_hex") != importlib.util.MAGIC_NUMBER.hex() or runtime.get("source_equivalence_claim") is not False:
        raise EntryError("old r2comparison bytecode provenance is incomplete")
    runtime_path = _resolve(str(runtime.get("path")), ROOT)
    runtime_hash = _hash_expected(runtime_path, runtime.get("sha256"), "old_038_r2comparison_runtime")
    if runtime_hash["sha256"] != runtime.get("sha256"):
        raise EntryError("old r2comparison runtime bytecode hash changed")
    if runtime.get("source_provenance_path") != "pr/r2comparison.py" or runtime.get("source_provenance_sha256") != compatibility.get("old_038_r2comparison_sha256"):
        raise EntryError("old r2comparison source provenance changed")
    if compatibility.get("old_026_manifest_genome_score_hashes_are_not_reused") is not True:
        raise EntryError("source snapshot silently reused stale 026 hashes")
    return {"files": observed, "compatibility": dict(compatibility)}


def _validate_mask_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    mask = config.get("evaluation_policy", {}).get("r2_mask")
    # The new config carries the immutable counts in its grid; paths live in the
    # release template so preflight does not need to infer a new mask.
    if mask is not None:
        if mask.get("expected_common_pairs") != EXPECTED_MASK_COMMON:
            raise EntryError("visibility mask common-pair count changed")
    template = _read_json(CONTRACT_TEMPLATE_PATH)
    lock_spec = template.get("mask_lock", {})
    lock_path = _resolve(str(lock_spec.get("lock_path")), PREP_DIR)
    manifest_path = _resolve(str(lock_spec.get("manifest_path")), PREP_DIR)
    lock_hash = _hash_expected(lock_path, lock_spec.get("lock_sha256"), "mask_lock")
    manifest_hash = _hash_expected(manifest_path, lock_spec.get("manifest_sha256"), "mask_manifest")
    lock = _read_json(lock_path)
    if lock.get("schema_version") != "p9016-multires-r2-mask-lock-v1":
        raise EntryError("old21 mask lock schema changed")
    if lock.get("manifest_sha256") != manifest_hash["sha256"]:
        raise EntryError("mask lock does not bind the published manifest")
    if lock.get("condition_count") != 21 or lock.get("new_endpoint_inclusion") is not False:
        raise EntryError("mask is not the frozen old21 mask")
    expected = lock.get("expected_by_chromosome")
    if not isinstance(expected, list) or len(expected) != 20:
        raise EntryError("mask lock lacks 20 expected chromosome records")
    total = sum(int(row["n_total_non_diagonal_pairs"]) for row in expected)
    common = sum(int(row["n_common_pairs"]) for row in expected)
    if (total, common) != (EXPECTED_MASK_TOTAL, EXPECTED_MASK_COMMON):
        raise EntryError("mask lock aggregate counts changed")
    manifest = _read_json(manifest_path)
    coordinates = manifest.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != 21:
        raise EntryError("published mask manifest lacks 21 coordinate records")
    return {
        "lock_path": str(lock_path),
        "lock_sha256": lock_hash["sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash["sha256"],
        "condition_count": 21,
        "total_non_diagonal_pairs": total,
        "common_pairs": common,
        "condition_ids": [str(row.get("condition_id")) for row in coordinates],
    }


def _metadata_path(fixture_id: str) -> Path:
    return ROOT / "test_res/026-20260913_221709-allele-calibration-prepare/eval_truth" / (fixture_id + "_truth_1mb.npz.json")


def population_kl_feasibility() -> dict[str, Any]:
    """只依据生成源码和 metadata 判断 population KL 能否精确定义。"""
    source_path = FROZEN_PR_DIR / "v1_calibration.py"
    try:
        source_text = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source_text, filename=str(source_path))
    except (OSError, SyntaxError) as exc:
        raise EntryError("cannot inspect generation source") from exc
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = ("_kernel_values", "generation_rates", "expected_counts")
    missing = [name for name in required if name not in functions]
    metadata = {}
    for fixture_id in SYNTHETIC_FIXTURES:
        path = _metadata_path(fixture_id)
        expected_sha = EXPECTED_SYNTHETIC_METADATA_SHA[fixture_id]
        _hash_expected(path, expected_sha, fixture_id + ".truth_metadata")
        value = _read_json(path)
        metadata[fixture_id] = {
            "path": str(path),
            "sha256": expected_sha,
            "fixture_id": value.get("fixture_id"),
            "p_gen": value.get("p_gen"),
            "generation_kernel": value.get("generation_kernel"),
            "generation_r0": value.get("generation_r0"),
            "generation_epsilon": value.get("generation_epsilon"),
            "generation_exposure_metadata": value.get("generation_exposure"),
        }
        if value.get("fixture_id") != fixture_id:
            missing.append(fixture_id + ":fixture_id")
        if value.get("p_gen") != 0.8:
            missing.append(fixture_id + ":p_gen")
        if value.get("generation_kernel") != "v1":
            missing.append(fixture_id + ":kernel")
        if value.get("generation_r0") != "2*l0":
            missing.append(fixture_id + ":r0")
        if value.get("generation_epsilon") != 1e-6:
            missing.append(fixture_id + ":epsilon")
    source_checks = {
        "required_functions_present": not missing[:3],
        "v1_kernel_branch": 'kernel == "v1"' in source_text,
        "epsilon_kernel_is_finite": "EPSILON + (1.0 - EPSILON) * (1.0 + ratio2) ** -2" in source_text,
        "cis_and_inter_group_rates": "rates[mask].sum()" in source_text,
        "same_bin_separate_nuisance": "diag_weights = exposure * exposure" in source_text,
        "no_sampling_needed": "sample_conditional_counts" in source_text,
    }
    lengths_path = ROOT / "test_res/026-20260913_221709-allele-calibration-prepare/provenance/source_manifest.json"
    source_manifest = _read_json(lengths_path)
    lengths = [int(value) for value in source_manifest["allowed_header_summary"]["chromosome_lengths"]]
    bins = [(length + 999999) // 1000000 for length in lengths]
    cis_pairs = sum(n * (n - 1) // 2 for n in bins)
    all_pairs = sum(bins) * (sum(bins) - 1) // 2
    inter_pairs = all_pairs - cis_pairs
    if (sum(bins), cis_pairs, inter_pairs) != (2645, 184016, 3312674):
        missing.append("eligible_pair_denominators")
    source_checks["full_grid_denominators"] = (sum(bins), cis_pairs, inter_pairs) == (2645, 184016, 3312674)
    feasible = not missing and all(source_checks.values())
    return {
        "schema_version": "p9016-visibility-population-kl-feasibility-v1",
        "status": "FEASIBLE_EXACT_GENERATION_RATES" if feasible else "INFEASIBLE_OR_INCOMPLETE",
        "exact": bool(feasible),
        "preparation_did_not_open_coordinate_payloads": True,
        "source_path": str(source_path),
        "source_functions": {name: bool(name in functions) for name in required},
        "source_checks": source_checks,
        "metadata": metadata,
        "group_totals": dict(GROUP_TOTALS),
        "weighted_denominator": GROUP_DENOMINATOR,
        "eligible_pair_denominators": {"cis_offdiag": cis_pairs, "inter": inter_pairs},
        "definition": {
            "scope": "synthetic-only",
            "not_current_sampled_counts_training_nll": True,
            "eta_improvement_and_zero_coverage_are_separate": True,
            "pi_true": "normalize exact generation_rates(truth X, generating e, p_gen=0.8, kernel=v1) separately over cis_offdiag and inter",
            "pi_fit": "normalize the same rate formula at locked candidate X, locked p, and model final e separately over the same groups",
            "kl": "sum(pi_true*log(pi_true/pi_fit)) over every eligible pair, including observed count-zero pairs",
            "weighted": "(696680*KL_cis + 568434*KL_inter)/1265114",
            "same_bin": "excluded as saturated nuisance",
            "zero_fit_support": "pi_true>0 and pi_fit=0 is +inf; no clipping",
            "refit_or_sampling": "neither"
        },
        "limitations": [
            "truth coordinate and generation exposure arrays are bound only by the future release hash gate",
            "candidate final e sidecars must expose one full-grid value per genomic bin",
            "VZ e=0 support is handled by the visibility adapter's nonnegative-rate path, not by changing the frozen generator"
        ],
        "missing_or_failed_checks": missing,
    }


def _validate_static_contract() -> dict[str, Any]:
    config, protocol, template, source_snapshot = _load_static_files()
    if config.get("status_at_freeze") != "prepared_only_pending_parent_release":
        raise EntryError("preparation is not marked pending parent release")
    if protocol.get("status") != "frozen_preparation_pending_parent_release":
        raise EntryError("protocol is not frozen preparation-only")
    scope = config.get("scope", {})
    checks = {
        "raw_contact_count": scope.get("raw_contact_count") == 1703888,
        "chromosome_count": scope.get("chromosome_count") == 20,
        "track_count": scope.get("track_count") == 40,
        "physical_beads_1mb": scope.get("physical_beads_1mb") == 5290,
        "real_endpoint_count": scope.get("real_endpoint_count") == 6,
        "synthetic_endpoint_count": scope.get("synthetic_endpoint_count") == 6,
        "total_endpoint_count": scope.get("total_endpoint_count") == 12,
        "prohibited_metrics": set(scope.get("prohibited_metrics", [])) == {"R1", "R3", "p-values", "confidence intervals"},
    }
    if not all(checks.values()):
        raise EntryError("visibility scope changed: %r" % checks)
    grid = _validate_grid(config)
    methods = config["real_training_release"]["methods"]
    if set(methods) != set(METHOD_IDS):
        raise EntryError("visibility methods changed")
    real_ids = config["real_training_release"]["endpoint_ids"]
    synthetic_ids = config["synthetic_training_release"]["endpoint_ids"]
    if tuple(real_ids) != REAL_ENDPOINT_IDS or tuple(synthetic_ids) != SYNTHETIC_ENDPOINT_IDS:
        raise EntryError("visibility endpoint order changed")
    training = config["real_training_release"]
    if training.get("required_real_budget_id") != "B2":
        raise EntryError("real release must use B2 only")
    if training.get("B1_historical_reference_only_nfev_by_layer") != {"5mb": 306, "2mb": 202, "1mb": 243}:
        raise EntryError("historical B1 reference changed")
    if training.get("B2_required_nfev_by_layer") != {"5mb": 612, "2mb": 404, "1mb": 486}:
        raise EntryError("real B2 budget changed")
    if training.get("maximum_total_outer_fg") != 10470 or training.get("early_stop_allowed") is not True:
        raise EntryError("real outer-FG maximum/early-stop policy changed")
    if training.get("v1_inner_profile_must_converge") is not True:
        raise EntryError("V1 inner convergence requirement changed")
    if config["synthetic_training_release"]["one_mb_fg_cap"] != 243:
        raise EntryError("synthetic one-Mb FG cap changed")
    if config["synthetic_training_release"]["same_x0"] is not True or config["synthetic_training_release"]["same_p0"] != 0.75:
        raise EntryError("synthetic shared x0/p0 contract changed")
    v1_config = training["methods"]["V1"]
    expected_v1_dimensions = {
        "e_vector_length": 2645,
        "eta_active_count_real": 2581,
        "eta_free_dimension_real": 2580,
        "eta_active_count_synthetic": 2645,
        "eta_free_dimension_synthetic": 2644,
    }
    if any(v1_config.get(key) != value for key, value in expected_v1_dimensions.items()):
        raise EntryError("V1 e-vector/eta dimensions changed")
    if not v1_config.get("eta_gauge") or v1_config.get("eta_parameter_count") is not None:
        raise EntryError("V1 must name active eta dimensions, not claim 2645 independent eta")
    vz = training["methods"]["VZ"]
    if vz.get("synthetic_zero_degree_precheck") != {"P2": 0, "N2": 0} or vz.get("synthetic_extra_fit_count") != 0:
        raise EntryError("synthetic VZ zero-degree precheck changed")
    if vz.get("real_zero_degree_bins") != 64 or vz.get("real_active_degree_positive_bins") != 2581:
        raise EntryError("real VZ degree partition changed")
    sidecar_config = config.get("evaluation_policy", {}).get("sidecar_schema", {})
    if sidecar_config.get("name") != "p9016-visibility-profile-sidecars-v1" or sidecar_config.get("alignment_fields") != ["numeric_position", "chromosome_index"]:
        raise EntryError("sidecar schema alignment contract changed")
    if sidecar_config.get("v1_degree_fields") != ["degree", "predicted_degree"] or sidecar_config.get("v1_degree_residual_tolerance") != 1e-10:
        raise EntryError("V1 degree sidecar contract changed")
    if protocol["selection_protocol"]["real"]["tie_tolerance"] != 1e-9:
        raise EntryError("source-selection tie tolerance changed")
    if protocol["r2_protocol"]["real_mask"]["expected_common_pairs"] != EXPECTED_MASK_COMMON:
        raise EntryError("R2 common mask count changed")
    if protocol["r2_protocol"]["real_mask"]["expected_total_non_diagonal_pairs"] != EXPECTED_MASK_TOTAL:
        raise EntryError("R2 total mask count changed")
    regression_gate = protocol["release_gate"].get("historical_regression_gate", {})
    if regression_gate.get("gate_id") != "old038-m0-b2-random-r2-regression-v1" or regression_gate.get("fixed_historical_condition_id") != "v1_original_random_joint":
        raise EntryError("historical regression gate identity changed")
    if regression_gate.get("compare_numeric_atol") != 1e-12 or regression_gate.get("require_exact_frozen_mask_fields") != ["positions", "pair_i", "pair_j", "common"]:
        raise EntryError("historical regression comparison contract changed")
    if regression_gate.get("new_science_metric") is not False or regression_gate.get("selection_input") is not False or regression_gate.get("failure_policy") != "pause_new_metric_interpretation_and_diagnose":
        raise EntryError("historical regression gate is not diagnostic-only")
    if regression_gate.get("runtime_provenance", {}).get("source_equivalence_claim") is not False:
        raise EntryError("historical regression gate must not claim source/bytecode equivalence")
    source_info = _validate_source_snapshot(source_snapshot)
    source_config = config.get("source_snapshot", {})
    if source_config.get("entity_snapshot_dir") != "docs/audits/visibility-r2-preparation-20260914T164603Z/frozen_pr" or source_config.get("import_package") != "visibility_frozen_pr":
        raise EntryError("frozen_pr import contract changed")
    if source_config.get("live_pr_imports_for_formal_evaluation") is not False or source_config.get("old_frozen_helpers_keep_original_paths") is not True:
        raise EntryError("formal evaluation may not fall back to live pr")
    historical = config["evaluation_policy"]["historical_comparison"]
    if historical.get("new_v0_same_torch_backend") is not True or historical.get("old_038_table_recomputed") is not False or historical.get("old_038_reference_used_for_selection") is not False:
        raise EntryError("historical 038 comparison policy changed")
    if historical.get("old_038_regression_gate_registered") is not True or historical.get("old_038_regression_gate_id") != "old038-m0-b2-random-r2-regression-v1" or historical.get("old_038_regression_gate_is_not_new_metric_or_selection") is not True:
        raise EntryError("historical regression gate registration changed")
    if historical.get("old_038_regression_gate_failure_policy") != "pause_new_metric_interpretation_and_diagnose" or historical.get("old_038_regression_gate_run_after_all_12_hashes") is not True:
        raise EntryError("historical regression gate ordering/policy changed")
    mask_info = _validate_mask_metadata(config)
    feasibility = population_kl_feasibility()
    if feasibility["status"] != "FEASIBLE_EXACT_GENERATION_RATES":
        raise EntryError("population KL exact feasibility check failed")
    return {
        "config": config,
        "protocol": protocol,
        "template": template,
        "source_snapshot": source_snapshot,
        "source_info": source_info,
        "mask_info": mask_info,
        "grid": grid,
        "feasibility": feasibility,
        "scope_checks": checks,
    }


def validate_preparation() -> dict[str, Any]:
    """执行只读 metadata preflight；明确不打开任何坐标 payload。"""
    static = _validate_static_contract()
    _write_json(FEASIBILITY_PATH, static["feasibility"])
    result = {
        "schema_version": "p9016-visibility-preparation-validation-v1",
        "status": "PASS",
        "prepared_only": True,
        "evaluation_not_run": True,
        "optimizer_called": False,
        "native_called": False,
        "candidate_coordinate_payloads_opened": 0,
        "reference_opened": False,
        "synthetic_truth_coordinate_payloads_opened": False,
        "phase_payload_opened": False,
        "endpoint_count": 12,
        "real_endpoint_count": 6,
        "synthetic_endpoint_count": 6,
        "config_sha256": _sha256(CONFIG_PATH),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "contract_template_sha256": _sha256(CONTRACT_TEMPLATE_PATH),
        "source_snapshot_sha256": _sha256(SOURCE_SNAPSHOT_PATH),
        "entry_sha256": _sha256(PREP_DIR / "visibility_r2_entry.py"),
        "mask": static["mask_info"],
        "population_kl": {
            "status": static["feasibility"]["status"],
            "exact": static["feasibility"]["exact"],
            "weighted_denominator": GROUP_DENOMINATOR,
            "eligible_pair_denominators": static["feasibility"]["eligible_pair_denominators"],
        },
        "source_compatibility": static["source_info"]["compatibility"],
        "historical_regression_gate": _historical_regression_gate_registration(static["template"], static),
        "frozen_pr_directory": str(FROZEN_PR_DIR),
        "synthetic_import_package": FROZEN_PR_PACKAGE,
        "integration_schema_check": {
            "path": str(INTEGRATION_SCHEMA_PATH),
            "sha256": _sha256(INTEGRATION_SCHEMA_PATH) if INTEGRATION_SCHEMA_PATH.is_file() else None,
            "status": "PASS" if INTEGRATION_SCHEMA_PATH.is_file() else "not_present",
            "formal_release": False,
        },
        "integration_v2_real_vz_schema_check": {
            "path": str(INTEGRATION_V2_VZ_PATH),
            "sha256": _sha256(INTEGRATION_V2_VZ_PATH) if INTEGRATION_V2_VZ_PATH.is_file() else None,
            "status": "PASS" if INTEGRATION_V2_VZ_PATH.is_file() else "not_present",
            "formal_release": False,
        },
        "access_order": [
            "preflight JSON/source metadata only",
            "parent release contract",
            "all 12 candidate/required-sidecar (56)/terminal hashes",
            "parent source/selection and frozen old21 mask hashes",
            "registered old038 historical regression gate (execution deferred)",
            "reference/truth hash and payload access",
            "R2 and synthetic population KL",
        ],
    }
    _write_json(VALIDATION_PATH, result)
    terminal = {
        "schema_version": "p9016-visibility-preparation-terminal-evidence-v1",
        "status": "PASS",
        "exit_code": 0,
        "fit_called": False,
        "native_called": False,
        "evaluation_started": False,
        "candidate_coordinate_payloads_opened": 0,
        "reference_opened": False,
        "truth_coordinate_payloads_opened": False,
        "prepared_artifacts": {
            "config": str(CONFIG_PATH),
            "protocol": str(PROTOCOL_PATH),
            "source_snapshot": str(SOURCE_SNAPSHOT_PATH),
            "frozen_pr_directory": str(FROZEN_PR_DIR),
            "population_kl_feasibility": str(FEASIBILITY_PATH),
            "integration_schema_validation": str(INTEGRATION_SCHEMA_PATH),
            "integration_v2_real_vz_schema_validation": str(INTEGRATION_V2_VZ_PATH),
            "release_contract_template": str(CONTRACT_TEMPLATE_PATH),
            "adapter": str(PREP_DIR / "visibility_r2_entry.py"),
        },
        "hashes": {
            "config": result["config_sha256"],
            "protocol": result["protocol_sha256"],
            "contract_template": result["contract_template_sha256"],
            "source_snapshot": result["source_snapshot_sha256"],
            "adapter": result["entry_sha256"],
        },
        "historical_regression_gate": result["historical_regression_gate"],
    }
    _write_json(TERMINAL_EVIDENCE_PATH, terminal)
    return result


def _expected_endpoint_kind(endpoint_id: str) -> tuple[str, str, str | None]:
    if endpoint_id in REAL_ENDPOINT_IDS:
        method, source = endpoint_id.split("-", 1)
        return "real", method, source
    if endpoint_id in SYNTHETIC_ENDPOINT_IDS:
        fixture, arm = endpoint_id.split("-", 1)
        method = "V1" if arm == "V1-profile" else "V0"
        return "synthetic", method, fixture
    raise EntryError("unknown endpoint id: %s" % endpoint_id)


def _validate_contract_endpoint(endpoint: Mapping[str, Any]) -> None:
    endpoint_id = str(endpoint.get("endpoint_id"))
    kind, method, extra = _expected_endpoint_kind(endpoint_id)
    if endpoint.get("kind") != kind or endpoint.get("method_id") != method:
        raise EntryError("endpoint kind/method mismatch: %s" % endpoint_id)
    common = (
        "final_3dg_path", "final_3dg_sha256", "terminal_record_path", "terminal_record_sha256",
        "p_path", "p_sha256", "q_path", "q_sha256", "final_e_path", "final_e_sha256",
        "active_mask_path", "active_mask_sha256",
    )
    for key in common:
        if key.endswith("sha256"):
            _require_sha(endpoint.get(key), endpoint_id + "." + key)
        elif not isinstance(endpoint.get(key), str) or not endpoint.get(key):
            raise EntryError("endpoint lacks %s: %s" % (key, endpoint_id))
    if not _finite(endpoint.get("count_nll_per_record")):
        raise EntryError("endpoint count NLL is not finite: %s" % endpoint_id)
    if endpoint.get("endpoint_status") not in {"converged", "budget_not_converged", "not_converged", "completed"}:
        raise EntryError("endpoint is not terminal: %s" % endpoint_id)
    if endpoint.get("sidecar_schema") != "p9016-visibility-profile-sidecars-v1":
        raise EntryError("endpoint sidecar schema mismatch: %s" % endpoint_id)
    actual_fg = endpoint.get("actual_outer_fg_total")
    maximum_fg = endpoint.get("maximum_outer_fg_total")
    if not isinstance(actual_fg, int) or isinstance(actual_fg, bool) or actual_fg < 0:
        raise EntryError("endpoint actual outer-FG total is invalid: %s" % endpoint_id)
    expected_maximum_fg = 1502 if kind == "real" else 243
    if maximum_fg != expected_maximum_fg or actual_fg > maximum_fg:
        raise EntryError("endpoint outer-FG maximum/actual mismatch: %s" % endpoint_id)
    if endpoint.get("e_vector_length") != 2645:
        raise EntryError("endpoint full e-vector length mismatch: %s" % endpoint_id)
    if kind == "real":
        if endpoint.get("source_id") != extra:
            raise EntryError("real source mismatch: %s" % endpoint_id)
        if endpoint.get("all_contacts_used") is not True or endpoint.get("includes_inter") is not True:
            raise EntryError("real endpoint does not bind all-contact training: %s" % endpoint_id)
        expected_budget = {"5mb": 612, "2mb": 404, "1mb": 486}
        if endpoint.get("budget_id") != "B2" or endpoint.get("budget_nfev_by_layer") != expected_budget:
            raise EntryError("real endpoint must use the B2 budget only: %s" % endpoint_id)
        if not isinstance(endpoint.get("start_014_path"), str) or not endpoint.get("start_014_path"):
            raise EntryError("real endpoint lacks independent 014 start: %s" % endpoint_id)
        _require_sha(endpoint.get("start_014_sha256"), endpoint_id + ".start_014_sha256")
    else:
        if endpoint.get("fixture_id") != extra:
            raise EntryError("synthetic fixture mismatch: %s" % endpoint_id)
        if endpoint.get("same_x0") is not True or endpoint.get("p0") != 0.75:
            raise EntryError("synthetic shared x0/p0 mismatch: %s" % endpoint_id)
        if endpoint.get("one_mb_fg_cap") != 243:
            raise EntryError("synthetic FG cap mismatch: %s" % endpoint_id)
        if endpoint.get("exposure_policy") not in {"V0-production", "V1-profile", "V0-known-generating-e"}:
            raise EntryError("synthetic exposure policy mismatch: %s" % endpoint_id)
    if method == "V0":
        if not endpoint.get("e_policy") or not endpoint.get("e_formula"):
            raise EntryError("V0 endpoint lacks fixed-e policy: %s" % endpoint_id)
    elif method == "VZ":
        if not isinstance(endpoint.get("zero_degree_count"), int) or not isinstance(endpoint.get("active_degree_positive_count"), int):
            raise EntryError("VZ endpoint lacks degree accounting: %s" % endpoint_id)
        if endpoint["zero_degree_count"] + endpoint["active_degree_positive_count"] != 2645:
            raise EntryError("VZ degree partition does not cover 2645 bins: %s" % endpoint_id)
        if kind == "real" and (endpoint["zero_degree_count"] != 64 or endpoint["active_degree_positive_count"] != 2581):
            raise EntryError("real VZ degree partition mismatch: %s" % endpoint_id)
        if not endpoint.get("zero_degree_rule") or not isinstance(endpoint.get("vz_equivalent_to_v0"), bool):
            raise EntryError("VZ boundary metadata incomplete: %s" % endpoint_id)
    else:
        for key in ("degree_path", "predicted_degree_path"):
            if not isinstance(endpoint.get(key), str) or not endpoint.get(key):
                raise EntryError("V1 endpoint lacks %s: %s" % (key, endpoint_id))
        for key in ("degree_sha256", "predicted_degree_sha256"):
            _require_sha(endpoint.get(key), endpoint_id + "." + key)
        expected = {"active": 2581, "free": 2580, "zero": 64} if kind == "real" else {"active": 2645, "free": 2644, "zero": 0}
        if endpoint.get("eta_active_count") != expected["active"]:
            raise EntryError("V1 eta active count mismatch: %s" % endpoint_id)
        if endpoint.get("eta_free_dimension") != expected["free"]:
            raise EntryError("V1 eta free dimension mismatch: %s" % endpoint_id)
        if endpoint.get("zero_degree_count") != expected["zero"]:
            raise EntryError("V1 zero-degree count mismatch: %s" % endpoint_id)
        if endpoint.get("eta_free_dimension") != endpoint.get("eta_active_count", -1) - 1:
            raise EntryError("V1 active gauge dimension mismatch: %s" % endpoint_id)
        if not isinstance(endpoint.get("eta_gauge"), str) or not endpoint["eta_gauge"]:
            raise EntryError("V1 eta gauge is missing: %s" % endpoint_id)
        if endpoint.get("eta_parameter_count") is not None:
            raise EntryError("V1 must not claim 2645 independent eta parameters: %s" % endpoint_id)
        if endpoint.get("same_bin_policy") != "saturated nuisance; excluded from eta profiling":
            raise EntryError("V1 same-bin policy changed: %s" % endpoint_id)
        if not endpoint.get("profile_likelihood"):
            raise EntryError("V1 profile likelihood is missing: %s" % endpoint_id)
        if endpoint.get("inner_profile_converged") is not True:
            raise EntryError("V1 inner profile did not fully converge: %s" % endpoint_id)


def _validate_release_contract(contract_path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    raw_path = Path(contract_path)
    if not raw_path.is_absolute():
        raise EntryError("release contract input must be an absolute path")
    path = raw_path.resolve()
    contract = _read_json(path)
    config, protocol, _template, source_snapshot = _load_static_files()
    if contract.get("schema_version") != "p9016-visibility-r2-release-v1":
        raise EntryError("release contract schema mismatch")
    if contract.get("status") != "released" or contract.get("release_authorized") is not True:
        raise EntryError("release contract is not authorized")
    flags = contract.get("release_flags", {})
    required_flags = (
        "all_12_endpoint_paths_locked", "all_12_endpoint_hashes_locked", "all_source_hashes_locked",
        "all_terminal_hashes_locked", "all_selection_hashes_locked", "reference_read_allowed",
    )
    if any(flags.get(key) is not True for key in required_flags) or flags.get("prepared_only") is not False:
        raise EntryError("release flags are not armed")
    scope = contract.get("scope", {})
    if tuple(scope.get("real_endpoint_ids", [])) != REAL_ENDPOINT_IDS:
        raise EntryError("release real endpoint order changed")
    if tuple(scope.get("synthetic_endpoint_ids", [])) != SYNTHETIC_ENDPOINT_IDS:
        raise EntryError("release synthetic endpoint order changed")
    records = contract.get("endpoint_records")
    if not isinstance(records, list) or len(records) != 12:
        raise EntryError("release does not contain exactly 12 endpoint records")
    by_id = {}
    for endpoint in records:
        if not isinstance(endpoint, Mapping):
            raise EntryError("release endpoint record is not an object")
        endpoint_id = str(endpoint.get("endpoint_id"))
        if endpoint_id in by_id:
            raise EntryError("duplicate endpoint record: %s" % endpoint_id)
        by_id[endpoint_id] = dict(endpoint)
        _validate_contract_endpoint(endpoint)
    if set(by_id) != set(REAL_ENDPOINT_IDS + SYNTHETIC_ENDPOINT_IDS):
        raise EntryError("release endpoint set changed")
    accounting = contract.get("outer_fg_accounting", {})
    if accounting.get("maximum_total_outer_fg") != 10470 or accounting.get("early_stop_allowed") is not True or accounting.get("actual_total_outer_fg_required") is not True:
        raise EntryError("outer-FG accounting policy changed")
    expected_maximum_total = sum(int(item["maximum_outer_fg_total"]) for item in by_id.values())
    if expected_maximum_total != 10470:
        raise EntryError("endpoint maximum outer-FG totals do not equal 10470")
    actual_total = sum(int(item["actual_outer_fg_total"]) for item in by_id.values())
    if actual_total < 0 or actual_total > 10470:
        raise EntryError("actual outer-FG total exceeds the maximum")
    selection = contract.get("real_source_selection", {})
    if selection.get("criterion") != "final count_nll_per_record" or selection.get("tie_tolerance") != 1e-9:
        raise EntryError("release source-selection rule changed")
    if selection.get("tie_order") != list(SOURCE_IDS) or selection.get("cross_model_selection") is not False:
        raise EntryError("release source-selection order/scope changed")
    selected = selection.get("selected_endpoint_by_method")
    if not isinstance(selected, Mapping) or set(selected) != set(METHOD_IDS):
        raise EntryError("release lacks one selected source for each method")
    for method in METHOD_IDS:
        candidates = [by_id[item] for item in REAL_ENDPOINT_IDS if by_id[item]["method_id"] == method]
        values = {item["endpoint_id"]: float(item["count_nll_per_record"]) for item in candidates}
        minimum = min(values.values())
        tied = [item for item in SOURCE_IDS if abs(values[method + "-" + item] - minimum) <= 1e-9]
        expected = method + "-" + tied[0]
        if selected.get(method) != expected:
            raise EntryError("source selection is not deterministic for %s" % method)
    parent = contract.get("parent_evidence", {})
    for key in ("terminal_evidence_path", "source_manifest_path", "run_receipt_path", "preflight_path"):
        if not isinstance(parent.get(key), str) or not parent.get(key):
            raise EntryError("parent evidence path missing: %s" % key)
        _require_sha(parent.get(key.replace("_path", "_sha256")), "parent." + key)
    source_lock = contract.get("source_lock", {})
    source_snapshot_path = _resolve_contract(source_lock.get("source_snapshot_path"), path)
    entry_path = _resolve_contract(source_lock.get("entry_path"), path)
    protocol_path = _resolve_contract(source_lock.get("protocol_path"), path)
    config_path = _resolve_contract(source_lock.get("config_path"), path)
    _require_sha(source_lock.get("source_snapshot_sha256"), "source_lock.source_snapshot_sha256")
    _require_sha(source_lock.get("entry_sha256"), "source_lock.entry_sha256")
    _require_sha(source_lock.get("protocol_sha256"), "source_lock.protocol_sha256")
    _require_sha(source_lock.get("config_sha256"), "source_lock.config_sha256")
    for target, key in ((source_snapshot_path, "source_snapshot_sha256"), (entry_path, "entry_sha256"), (protocol_path, "protocol_sha256"), (config_path, "config_sha256")):
        if not target.is_file():
            raise EntryError("source lock path is missing: %s" % target)
    mask = contract.get("mask_lock", {})
    if mask.get("condition_count") != 21 or mask.get("new_endpoint_inclusion") is not False:
        raise EntryError("release old21 mask contract changed")
    if mask.get("expected_common_pairs") != EXPECTED_MASK_COMMON or mask.get("expected_total_non_diagonal_pairs") != EXPECTED_MASK_TOTAL:
        raise EntryError("release old21 mask counts changed")
    for key in ("manifest_sha256", "lock_sha256"):
        _require_sha(mask.get(key), "mask_lock." + key)
    historical_gate = contract.get("historical_regression_gate", {})
    if historical_gate.get("gate_id") != "old038-m0-b2-random-r2-regression-v1" or historical_gate.get("status") != "registered_pending_12_endpoint_hashes":
        raise EntryError("historical regression gate is not registered pending the full hash gate")
    if historical_gate.get("required_before_new_metric_interpretation") is not True or historical_gate.get("fixed_historical_condition_id") != "v1_original_random_joint":
        raise EntryError("historical regression gate scope changed")
    if historical_gate.get("published_table_path") != "test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_per_chromosome.tsv" or historical_gate.get("published_table_sha256") != "5e578b5ef8876f57866f9dfa325a8738adab70ebc4913498c570cc2b07883ed1":
        raise EntryError("historical published R2 table lock changed")
    if historical_gate.get("compare_chromosomes") != 20 or historical_gate.get("compare_numeric_atol") != 1e-12 or historical_gate.get("require_exact_frozen_mask_fields") != ["positions", "pair_i", "pair_j", "common"]:
        raise EntryError("historical regression field comparison changed")
    if historical_gate.get("new_science_metric") is not False or historical_gate.get("selection_input") is not False or historical_gate.get("failure_policy") != "pause_new_metric_interpretation_and_diagnose":
        raise EntryError("historical regression gate cannot affect new metric interpretation/selection")
    if historical_gate.get("executed") is not False or historical_gate.get("passed") is not None:
        raise EntryError("historical regression gate must remain pending before evaluation")
    expected_runtime = source_snapshot["frozen_compatibility"]["old_038_r2comparison_runtime"]
    if historical_gate.get("runtime_provenance") != {
        "r2comparison_artifact_type": expected_runtime["artifact_type"],
        "r2comparison_path": expected_runtime["path"],
        "r2comparison_sha256": expected_runtime["sha256"],
        "python_version": expected_runtime["python_version"],
        "source_provenance_path": expected_runtime["source_provenance_path"],
        "source_provenance_sha256": expected_runtime["source_provenance_sha256"],
        "source_equivalence_claim": False,
    }:
        raise EntryError("historical regression runtime provenance changed")
    reference = contract.get("reference", {})
    if reference.get("role") != "evaluation_only" or reference.get("read_after_candidate_and_mask_hash_gate") is not True:
        raise EntryError("reference access boundary changed")
    _require_sha(reference.get("sha256"), "reference.sha256")
    truth = contract.get("synthetic_truth", {})
    for fixture_id in SYNTHETIC_FIXTURES:
        item = truth.get(fixture_id, {})
        _require_sha(item.get("metadata_sha256"), fixture_id + ".metadata_sha256")
        if not isinstance(item.get("metadata_path"), str) or not isinstance(item.get("coordinate_path"), str):
            raise EntryError("synthetic truth paths are incomplete: %s" % fixture_id)
        _require_sha(item.get("coordinate_sha256"), fixture_id + ".coordinate_sha256")
        if item.get("opened_after_hash_gate") is not True:
            raise EntryError("synthetic truth access boundary changed: %s" % fixture_id)
    return path, contract, {"config": config, "protocol": protocol, "source_snapshot": source_snapshot, "endpoints": by_id}


def validate_release(contract_path: str | Path) -> dict[str, Any]:
    """只验证已发布 contract 的结构，不读取候选/reference/truth payload。"""
    path, contract, static = _validate_release_contract(contract_path)
    source_lock = contract["source_lock"]
    result = {
        "schema_version": "p9016-visibility-release-validation-v1",
        "status": "PASS",
        "release_contract": str(path),
        "endpoint_count": 12,
        "metadata_only": True,
        "candidate_coordinate_payloads_opened": 0,
        "reference_opened": False,
        "synthetic_truth_coordinate_payloads_opened": False,
        "source_paths_checked_for_presence_only": [
            str(_resolve_contract(source_lock["source_snapshot_path"], path)),
            str(_resolve_contract(source_lock["entry_path"], path)),
            str(_resolve_contract(source_lock["protocol_path"], path)),
            str(_resolve_contract(source_lock["config_path"], path)),
        ],
        "formal_evaluation_allowed": True,
        "formal_evaluation_started": False,
    }
    output = path.parent / "release_validation.json"
    _write_json(output, result)
    return result


def _historical_regression_gate_registration(contract: Mapping[str, Any], static: Mapping[str, Any]) -> dict[str, Any]:
    """只登记历史回归 gate；不打开历史 endpoint、reference 或已发布表。"""
    configured = contract["historical_regression_gate"]
    runtime = static["source_snapshot"]["frozen_compatibility"]["old_038_r2comparison_runtime"]
    return {
        "schema_version": "p9016-old038-r2-regression-gate-registration-v1",
        "gate_id": configured["gate_id"],
        "status": "registered_pending_12_endpoint_hashes",
        "executed": False,
        "passed": None,
        "fixed_historical_condition_id": configured["fixed_historical_condition_id"],
        "fixed_historical_label": configured["fixed_historical_label"],
        "run_after_hash_gate": True,
        "new_science_metric": False,
        "selection_input": False,
        "failure_policy": configured["failure_policy"],
        "reference_opened": False,
        "historical_coordinate_opened": False,
        "published_table_opened": False,
        "exact_frozen_mask_required": list(configured["require_exact_frozen_mask_fields"]),
        "runtime_provenance": {
            "artifact_type": runtime["artifact_type"],
            "bytecode_path": runtime["path"],
            "bytecode_sha256": runtime["sha256"],
            "python_version": runtime["python_version"],
            "source_provenance_path": runtime["source_provenance_path"],
            "source_provenance_sha256": runtime["source_provenance_sha256"],
            "source_equivalence_claim": False,
        },
        "note": "Registration only; execute after all 12 endpoint/terminal/source/selection/mask hashes and before interpreting new R2/KL metrics.",
    }


def _hash_release_gate(contract_path: Path, contract: Mapping[str, Any], static: Mapping[str, Any]) -> dict[str, Any]:
    """按冻结顺序 hash 候选/证据/mask；此函数不解析任何坐标。"""
    endpoints = static["endpoints"]
    candidate_files = []
    terminal_files = []
    sidecar_files = []
    for endpoint_id in REAL_ENDPOINT_IDS + SYNTHETIC_ENDPOINT_IDS:
        endpoint = endpoints[endpoint_id]
        coord = _hash_expected(_resolve_contract(endpoint["final_3dg_path"], contract_path), endpoint["final_3dg_sha256"], endpoint_id + ".final_3dg")
        terminal = _hash_expected(_resolve_contract(endpoint["terminal_record_path"], contract_path), endpoint["terminal_record_sha256"], endpoint_id + ".terminal")
        candidate_files.append({"endpoint_id": endpoint_id, **coord})
        terminal_files.append({"endpoint_id": endpoint_id, **terminal})
        for label, path_key, sha_key in (
            ("p", "p_path", "p_sha256"), ("q", "q_path", "q_sha256"),
            ("final_e", "final_e_path", "final_e_sha256"),
            ("active_mask", "active_mask_path", "active_mask_sha256"),
        ):
            item = _hash_expected(_resolve_contract(endpoint[path_key], contract_path), endpoint[sha_key], endpoint_id + "." + label)
            sidecar_files.append({"endpoint_id": endpoint_id, "kind": label, **item})
        if endpoint["method_id"] == "V1":
            for label, path_key, sha_key in (("degree", "degree_path", "degree_sha256"), ("predicted_degree", "predicted_degree_path", "predicted_degree_sha256")):
                item = _hash_expected(_resolve_contract(endpoint[path_key], contract_path), endpoint[sha_key], endpoint_id + "." + label)
                sidecar_files.append({"endpoint_id": endpoint_id, "kind": label, **item})
    parent_hashes = []
    parent = contract["parent_evidence"]
    for path_key, sha_key in (("terminal_evidence_path", "terminal_evidence_sha256"), ("source_manifest_path", "source_manifest_sha256"), ("run_receipt_path", "run_receipt_sha256"), ("preflight_path", "preflight_sha256")):
        item = _hash_expected(_resolve_contract(parent[path_key], contract_path), parent[sha_key], "parent." + path_key)
        parent_hashes.append({"kind": path_key, **item})
    source_lock = contract["source_lock"]
    source_hashes = []
    for path_key, sha_key in (("source_snapshot_path", "source_snapshot_sha256"), ("entry_path", "entry_sha256"), ("protocol_path", "protocol_sha256"), ("config_path", "config_sha256")):
        item = _hash_expected(_resolve_contract(source_lock[path_key], contract_path), source_lock[sha_key], "source." + path_key)
        source_hashes.append({"kind": path_key, **item})
    snapshot_path = _resolve_contract(source_lock["source_snapshot_path"], contract_path)
    snapshot = _read_json(snapshot_path)
    dependency_hashes = []
    for dependency in snapshot.get("files", []):
        declared = dependency.get("sha256")
        if declared is None:
            raise EntryError("source snapshot contains an unbound dependency: %s" % dependency.get("path"))
        target = _resolve(str(dependency.get("path")), ROOT)
        item = _hash_expected(target, declared, "source_snapshot." + str(dependency.get("path")))
        dependency_hashes.append({"role": dependency.get("role"), **item})
    if not dependency_hashes:
        raise EntryError("source snapshot dependency list is empty")
    selection = contract["real_source_selection"]
    selection_evidence = _hash_expected(_resolve_contract(selection["selection_evidence_path"], contract_path), selection["selection_evidence_sha256"], "selection_evidence")
    mask_spec = contract["mask_lock"]
    mask_manifest = _hash_expected(_resolve_contract(mask_spec["manifest_path"], contract_path), mask_spec["manifest_sha256"], "mask_manifest")
    mask_lock = _hash_expected(_resolve_contract(mask_spec["lock_path"], contract_path), mask_spec["lock_sha256"], "mask_lock")
    # mask coordinate files are hash-checked only after all endpoint/sidecar hashes.
    old_module = _load_frozen_multires()
    legacy_config = {
        "grid": static["config"]["grid"],
        "mask_lock": {"manifest_path": mask_spec["manifest_path"], "expected_pair_counts_path": mask_spec["lock_path"]},
        "reference": {"path": contract["reference"]["path"]},
    }
    manifest_path, _manifest, mask_rows = old_module._mask_manifest_records(legacy_config)
    mask_files = []
    for row in mask_rows:
        target = old_module.resolve_path(row["path"], manifest_path.parent)
        item = _hash_expected(target, row.get("sha256"), "mask." + str(row["condition_id"]))
        mask_files.append({"condition_id": row["condition_id"], **item})
    return {
        "schema_version": "p9016-visibility-hash-gate-v1",
        "candidate_files": candidate_files,
        "terminal_files": terminal_files,
        "sidecar_files": sidecar_files,
        "parent_evidence": parent_hashes,
        "source_files": source_hashes,
        "source_dependency_files": dependency_hashes,
        "selection_evidence": {"path": str(selection_evidence["path"]), "sha256": selection_evidence["sha256"]},
        "mask_manifest": {"path": str(mask_manifest["path"]), "sha256": mask_manifest["sha256"]},
        "mask_lock": {"path": str(mask_lock["path"]), "sha256": mask_lock["sha256"]},
        "mask_files": mask_files,
        "all_12_candidate_hashes_complete": len(candidate_files) == 12,
        "all_12_terminal_hashes_complete": len(terminal_files) == 12,
        "all_required_sidecar_hashes_complete": len(sidecar_files) == 56,
        "sidecar_hash_count": len(sidecar_files),
        "all_21_mask_hashes_complete": len(mask_files) == 21,
        "historical_regression_gate": _historical_regression_gate_registration(contract, static),
        "reference_opened": False,
        "truth_coordinates_opened": False,
    }


def hash_gate(contract_path: str | Path) -> dict[str, Any]:
    path, contract, static = _validate_release_contract(contract_path)
    gate = _hash_release_gate(path, contract, static)
    gate["sidecar_validation"] = _validate_release_sidecars(path, static)
    output = path.parent / "hash_gate_evidence.json"
    _write_json(output, gate)
    return gate


def _load_frozen_multires() -> Any:
    source_path = ROOT / "docs/audits/multires-r2-preparation-20260914_041826/multires_r2.py"
    spec = importlib.util.spec_from_file_location("visibility_frozen_multires", source_path)
    if spec is None or spec.loader is None:
        raise EntryError("cannot load frozen multires R2 adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The legacy module imports ``pr.r2comparison``/``pr.allele_r2`` by name.
    # Install only the frozen snapshot modules under those names before the
    # legacy parity call; no mutable live helper is allowed to resolve there.
    snapshot_score = _load_snapshot_module("score")
    snapshot_gate = _load_snapshot_module("gate")
    snapshot_r2comparison = _load_snapshot_module("r2comparison")
    snapshot_allele_r2 = _load_snapshot_module("allele_r2")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    live_package = importlib.import_module("pr")
    for name, value in (("score", snapshot_score), ("gate", snapshot_gate), ("r2comparison", snapshot_r2comparison), ("allele_r2", snapshot_allele_r2)):
        sys.modules["pr." + name] = value
        setattr(live_package, name, value)
    module.LEGACY_ALLELE_R2_PATH = FROZEN_PR_DIR / "allele_r2.py"
    module.LEGACY_ALLELE_R2_SHA256 = _sha256(module.LEGACY_ALLELE_R2_PATH)
    module.LEGACY_R2COMPARISON_PATH = FROZEN_PR_DIR / "r2comparison.pyc"
    module.LEGACY_R2COMPARISON_SHA256 = _sha256(module.LEGACY_R2COMPARISON_PATH)
    return module


def _validate_real_structures(old_module: Any, structures: Mapping[str, Mapping[int, np.ndarray]], endpoint_id: str) -> None:
    for chromosome_index, (chromosome, length_bp) in enumerate(old_module.CHROMOSOMES):
        positions = old_module.grid_positions(length_bp)
        for suffix in ("a", "b"):
            track = "c%02d%s" % (chromosome_index + 1, suffix)
            values = structures.get(track)
            if not isinstance(values, Mapping):
                raise EntryError("%s lacks track %s" % (endpoint_id, track))
            for position in positions:
                if int(position) not in values:
                    raise EntryError("%s lacks %s:%d" % (endpoint_id, track, int(position)))
                point = np.asarray(values[int(position)], dtype=np.float64)
                if point.shape != (3,) or not np.all(np.isfinite(point)):
                    raise EntryError("%s has invalid point %s:%d" % (endpoint_id, track, int(position)))


def _load_synthetic_modules() -> tuple[Any, Any]:
    """只从 frozen_pr 实体快照加载 synthetic evaluator 及其依赖。"""
    synthetic_prepare = _load_snapshot_module("allele_calibration")
    synthetic_evaluator = _load_snapshot_module("allele_calibration_evaluator")
    return synthetic_prepare, synthetic_evaluator


def _load_synthetic_candidate(synthetic_prepare: Any, path: Path, endpoint_id: str, template: Any) -> np.ndarray:
    coordinates = synthetic_prepare._candidate_file_coordinates(path, template)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (2, 2645, 3) or not np.all(np.isfinite(coordinates)):
        raise EntryError("%s synthetic candidate shape/finite check failed" % endpoint_id)
    return coordinates


def _hash_truth_files(contract_path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for fixture_id in SYNTHETIC_FIXTURES:
        item = contract["synthetic_truth"][fixture_id]
        coordinate = _hash_expected(_resolve_contract(item["coordinate_path"], contract_path), item["coordinate_sha256"], fixture_id + ".truth_coordinates")
        metadata = _hash_expected(_resolve_contract(item["metadata_path"], contract_path), item["metadata_sha256"], fixture_id + ".truth_metadata")
        result[fixture_id] = {"coordinate": coordinate, "metadata": metadata}
    return result


def _scalar_from_object(value: Any, key_names: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        for key in key_names:
            if key in value:
                return value[key]
        for child in value.values():
            if isinstance(child, Mapping):
                found = _scalar_from_object(child, key_names)
                if found is not None:
                    return found
    return None


def _load_scalar(path: Path, key_names: Sequence[str]) -> float:
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = _read_json(path)
        raw = _scalar_from_object(value, key_names)
        if raw is None:
            raise EntryError("scalar keys missing in %s" % path)
        try:
            parsed = float(raw)
        except (TypeError, ValueError) as exc:
            raise EntryError("scalar value is invalid in %s" % path) from exc
        if not math.isfinite(parsed):
            raise EntryError("nonfinite scalar in %s" % path)
        return parsed
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            for key in key_names:
                if key in payload:
                    values = np.asarray(payload[key]).reshape(-1)
                    if len(values) == 1 and math.isfinite(float(values[0])):
                        return float(values[0])
        raise EntryError("scalar keys missing in %s" % path)
    try:
        value = float(path.read_text(encoding="utf-8").strip().split()[0])
    except (OSError, ValueError, IndexError) as exc:
        raise EntryError("cannot read scalar %s" % path) from exc
    if not math.isfinite(value):
        raise EntryError("nonfinite scalar %s" % path)
    return value


def _mapping_array(mapping: Mapping[str, Any], aliases: Sequence[str], path: Path, label: str) -> Any:
    for key in aliases:
        if key in mapping:
            return mapping[key]
    raise EntryError("%s field is missing in %s" % (label, path))


def _record_array(records: Sequence[Any], aliases: Sequence[str], path: Path, label: str) -> list[Any]:
    values = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise EntryError("%s record %d is not an object in %s" % (label, index, path))
        values.append(_mapping_array(record, aliases, path, label))
    return values


def _sidecar_arrays(path: Path, value_aliases: Sequence[str], value_label: str) -> tuple[Any, Any, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            positions = _mapping_array(payload, ("numeric_position", "positions", "numeric_positions", "position_bp"), path, "positions")
            chromosome_index = _mapping_array(payload, ("chromosome_index", "chrindex", "chr_index"), path, "chromosome_index")
            values = _mapping_array(payload, value_aliases, path, value_label)
            return np.asarray(positions).copy(), np.asarray(chromosome_index).copy(), np.asarray(values).copy()
    if suffix == ".json":
        payload = _read_json(path)
        records = payload.get("records")
        if isinstance(records, list):
            positions = _record_array(records, ("positions", "position", "numeric_position", "position_bp"), path, "positions")
            chromosome_index = _record_array(records, ("chromosome_index", "chrindex", "chr_index"), path, "chromosome_index")
            values = _record_array(records, value_aliases, path, value_label)
            return np.asarray(positions), np.asarray(chromosome_index), np.asarray(values)
        positions = _mapping_array(payload, ("numeric_position", "positions", "numeric_positions", "position_bp"), path, "positions")
        chromosome_index = _mapping_array(payload, ("chromosome_index", "chrindex", "chr_index"), path, "chromosome_index")
        values = _mapping_array(payload, value_aliases, path, value_label)
        return np.asarray(positions), np.asarray(chromosome_index), np.asarray(values)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = tuple(reader.fieldnames or ())
            if not fields:
                raise EntryError("sidecar lacks header: %s" % path)
            position_key = next((key for key in ("positions", "position", "numeric_position", "position_bp") if key in fields), None)
            chromosome_key = next((key for key in ("chromosome_index", "chrindex", "chr_index") if key in fields), None)
            value_key = next((key for key in value_aliases if key in fields), None)
            if position_key is None or chromosome_key is None or value_key is None:
                raise EntryError("sidecar alignment/value columns are incomplete: %s" % path)
            rows = list(reader)
        return (
            np.asarray([row[position_key] for row in rows]),
            np.asarray([row[chromosome_key] for row in rows]),
            np.asarray([row[value_key] for row in rows]),
        )
    except OSError as exc:
        raise EntryError("cannot read sidecar: %s" % path) from exc


def _integer_vector(raw: Any, path: Path, label: str) -> np.ndarray:
    try:
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise EntryError("%s is not numeric in %s" % (label, path)) from exc
    if values.shape != (2645,) or not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise EntryError("%s must contain 2645 finite integer values in %s" % (label, path))
    return values.astype(np.int64)


def _float_vector(raw: Any, path: Path, label: str) -> np.ndarray:
    try:
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise EntryError("%s is not numeric in %s" % (label, path)) from exc
    if values.shape != (2645,) or not np.all(np.isfinite(values)):
        raise EntryError("%s must contain 2645 finite values in %s" % (label, path))
    return values


def _mask_vector(raw: Any, path: Path) -> np.ndarray:
    values = np.asarray(raw).reshape(-1)
    if values.shape != (2645,):
        raise EntryError("active_mask must contain 2645 values in %s" % path)
    if values.dtype.kind in "OUS":
        normalized = {str(value).strip().lower() for value in values.tolist()}
        if not normalized <= {"0", "1", "false", "true"}:
            raise EntryError("active_mask contains nonbinary values in %s" % path)
        return np.asarray([str(value).strip().lower() in {"1", "true"} for value in values.tolist()], dtype=bool)
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise EntryError("active_mask is not numeric in %s" % path) from exc
    if not np.all(np.isfinite(numeric)) or not np.all(np.isin(numeric, [0.0, 1.0])):
        raise EntryError("active_mask contains nonbinary values in %s" % path)
    return numeric.astype(bool)


def _template_alignment(template: Any) -> tuple[np.ndarray, np.ndarray]:
    positions = (np.asarray(template.locus_bin, dtype=np.int64) * int(template.bin_size)).reshape(-1)
    chromosome_index = np.asarray(template.locus_chromosome, dtype=np.int64).reshape(-1)
    if positions.shape != (2645,) or chromosome_index.shape != (2645,):
        raise EntryError("full template does not contain 2645 loci")
    return positions, chromosome_index


def _load_aligned_sidecar(path: Path, template: Any, value_aliases: Sequence[str], value_label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions_raw, chromosome_raw, values_raw = _sidecar_arrays(path, value_aliases, value_label)
    positions = _integer_vector(positions_raw, path, "positions")
    chromosome_index = _integer_vector(chromosome_raw, path, "chromosome_index")
    expected_positions, expected_chromosome_index = _template_alignment(template)
    if not np.array_equal(positions, expected_positions):
        raise EntryError("%s positions do not match full template order: %s" % (value_label, path))
    if not np.array_equal(chromosome_index, expected_chromosome_index):
        raise EntryError("%s chromosome_index does not match full template order: %s" % (value_label, path))
    return positions, chromosome_index, np.asarray(values_raw)


def _load_profile(path: Path, template: Any | None = None) -> np.ndarray:
    if template is not None:
        _positions, _chromosome_index, raw_values = _load_aligned_sidecar(path, template, ("e",), "e")
        values = _float_vector(raw_values, path, "e")
    else:
        suffix = path.suffix.lower()
        values: np.ndarray | None = None
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                if "e" in payload:
                    values = np.asarray(payload["e"], dtype=np.float64).reshape(-1)
        elif suffix == ".json":
            value = _read_json(path)
            if "e" in value:
                values = np.asarray(value["e"], dtype=np.float64).reshape(-1)
        else:
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    if not reader.fieldnames or "e" not in reader.fieldnames:
                        raise EntryError("profile sidecar lacks e column: %s" % path)
                    values = np.asarray([float(row["e"]) for row in reader], dtype=np.float64)
            except OSError as exc:
                raise EntryError("cannot read profile sidecar: %s" % path) from exc
        if values is None:
            raise EntryError("profile sidecar lacks e values: %s" % path)
    if values.shape != (2645,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise EntryError("profile sidecar must contain 2645 finite nonnegative e values: %s" % path)
    if not np.any(values > 0.0):
        raise EntryError("profile sidecar has zero total exposure: %s" % path)
    return values


def _load_degree_pair(template: Any, degree_path: Path, predicted_path: Path) -> dict[str, Any]:
    degree_positions, degree_chromosomes, degree_raw = _load_aligned_sidecar(degree_path, template, ("degree",), "degree")
    predicted_positions, predicted_chromosomes, predicted_raw = _load_aligned_sidecar(predicted_path, template, ("predicted_degree",), "predicted_degree")
    if not np.array_equal(degree_positions, predicted_positions) or not np.array_equal(degree_chromosomes, predicted_chromosomes):
        raise EntryError("degree and predicted_degree sidecars are not aligned")
    degree = _float_vector(degree_raw, degree_path, "degree")
    predicted = _float_vector(predicted_raw, predicted_path, "predicted_degree")
    if np.any(degree < 0.0) or np.any(predicted < 0.0):
        raise EntryError("degree and predicted_degree must be nonnegative")
    residual = degree - predicted
    relative = np.abs(residual) / np.maximum(1.0, degree)
    max_abs = float(np.max(np.abs(residual)))
    max_relative = float(np.max(relative))
    if max_relative > 1e-10:
        raise EntryError("degree residual exceeds 1e-10 relative tolerance")
    return {
        "degree": degree,
        "predicted_degree": predicted,
        "degree_residual": residual,
        "max_abs_residual": max_abs,
        "max_relative_residual": max_relative,
        "positions_order_match": True,
        "chromosome_index_order_match": True,
    }


def _validate_release_sidecars(contract_path: Path, static: Mapping[str, Any]) -> dict[str, Any]:
    """在候选 hash gate 后只读验证 sidecar 对齐和终态 p/q 一致性。"""
    frozen_generation = _load_snapshot_module("v1_calibration")
    template = frozen_generation.header_templates()[frozen_generation.FINAL_BIN]
    expected_positions, expected_chromosome_index = _template_alignment(template)
    records = []
    for endpoint_id in REAL_ENDPOINT_IDS + SYNTHETIC_ENDPOINT_IDS:
        endpoint = static["endpoints"][endpoint_id]
        e_path = _resolve_contract(endpoint["final_e_path"], contract_path)
        active_path = _resolve_contract(endpoint["active_mask_path"], contract_path)
        e_positions, e_chromosomes, e_raw = _load_aligned_sidecar(e_path, template, ("e",), "e")
        active_positions, active_chromosomes, active_raw = _load_aligned_sidecar(active_path, template, ("active_mask",), "active_mask")
        if not np.array_equal(e_positions, expected_positions) or not np.array_equal(e_chromosomes, expected_chromosome_index):
            raise EntryError("final e sidecar alignment changed: %s" % endpoint_id)
        if not np.array_equal(active_positions, expected_positions) or not np.array_equal(active_chromosomes, expected_chromosome_index):
            raise EntryError("active mask sidecar alignment changed: %s" % endpoint_id)
        e_values = _float_vector(e_raw, e_path, "e")
        if np.any(e_values < 0.0) or not np.any(e_values > 0.0):
            raise EntryError("final e sidecar has invalid support: %s" % endpoint_id)
        active = _mask_vector(active_raw, active_path)
        active_count = int(np.count_nonzero(active))
        if endpoint["method_id"] in {"VZ", "V1"}:
            if not np.array_equal(e_values > 0.0, active):
                raise EntryError("zero-exposure and active-mask support differ: %s" % endpoint_id)
        degree_result: dict[str, Any] = {
            "status": "not_applicable",
            "max_abs_residual": None,
            "max_relative_residual": None,
            "positions_order_match": None,
            "chromosome_index_order_match": None,
        }
        if endpoint["method_id"] == "V1":
            degree_result = _load_degree_pair(
                template,
                _resolve_contract(endpoint["degree_path"], contract_path),
                _resolve_contract(endpoint["predicted_degree_path"], contract_path),
            )
            degree_result["status"] = "PASS"
            if active_count != endpoint["eta_active_count"]:
                raise EntryError("V1 active eta count disagrees with active mask: %s" % endpoint_id)
            if int(np.count_nonzero(e_values <= 0.0)) != endpoint["zero_degree_count"]:
                raise EntryError("V1 zero-degree count disagrees with e sidecar: %s" % endpoint_id)
        if endpoint["method_id"] == "VZ":
            if active_count != endpoint["active_degree_positive_count"]:
                raise EntryError("VZ active count disagrees with active mask: %s" % endpoint_id)
        if "active_degree_positive_count" in endpoint and endpoint["method_id"] == "V0":
            if active_count != endpoint["active_degree_positive_count"]:
                raise EntryError("V0 active count disagrees with active mask: %s" % endpoint_id)
        p_path = _resolve_contract(endpoint["p_path"], contract_path)
        q_path = _resolve_contract(endpoint["q_path"], contract_path)
        terminal_path = _resolve_contract(endpoint["terminal_record_path"], contract_path)
        p_sidecar = _load_scalar(p_path, ("p", "final_p", "value"))
        q_sidecar = _load_scalar(q_path, ("q", "raw_q", "rawq", "final_q", "value"))
        p_terminal = _load_scalar(terminal_path, ("final_p", "p"))
        q_terminal = _load_scalar(terminal_path, ("final_q", "q", "raw_q", "rawq"))
        if not math.isfinite(q_sidecar):
            raise EntryError("q sidecar is not finite: %s" % endpoint_id)
        contact_model = _load_snapshot_module("contact_model")
        p_from_q, _dpdq = contact_model.p_from_q(q_sidecar)
        p_floor = float(contact_model.P_FLOOR)
        if not (p_floor < p_sidecar < 1.0 - p_floor):
            raise EntryError("p sidecar is outside bounded domain: %s" % endpoint_id)
        _assert_close(p_sidecar, p_from_q, 1e-12, endpoint_id + ".p_from_q")
        _assert_close(p_sidecar, p_terminal, 1e-12, endpoint_id + ".p_terminal")
        _assert_close(q_sidecar, q_terminal, 1e-12, endpoint_id + ".q_terminal")
        if "final_p" in endpoint:
            _assert_close(p_sidecar, endpoint["final_p"], 1e-12, endpoint_id + ".p_contract")
        records.append({
            "endpoint_id": endpoint_id,
            "e_vector_length": int(e_values.size),
            "active_mask_count": active_count,
            "zero_e_count": int(np.count_nonzero(e_values <= 0.0)),
            "degree_residual_status": degree_result["status"],
            "max_abs_degree_residual": degree_result["max_abs_residual"],
            "max_relative_degree_residual": degree_result["max_relative_residual"],
            "positions_order_match": True,
            "chromosome_index_order_match": True,
            "p_from_q_match": True,
            "p_q_terminal_match": True,
        })
    return {
        "schema_version": "p9016-visibility-sidecar-validation-v1",
        "status": "PASS",
        "full_template_loci": 2645,
        "records": records,
        "reference_opened": False,
        "truth_coordinates_opened": False,
    }


def _allow_zero_generation_rates(template: Any, coordinates: np.ndarray, exposure: np.ndarray, p: float, kernel: str) -> np.ndarray:
    """复制 frozen v1 rate 公式但保留 VZ 的合法零 exposure support。"""
    if kernel != "v1":
        raise EntryError("population KL only accepts the frozen v1 kernel")
    if coordinates.shape != (2, template.n_loci, 3) or exposure.shape != (template.n_loci,):
        raise EntryError("candidate generation inputs do not match full grid")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(exposure)) or np.any(exposure < 0.0):
        raise EntryError("candidate generation inputs are not finite/nonnegative")
    frozen_generation = _load_snapshot_module("v1_calibration")
    _kernel_values = frozen_generation._kernel_values
    rates = np.empty(template.n_pairs, dtype=np.float64)
    for start in range(0, template.n_pairs, 65536):
        stop = min(start + 65536, template.n_pairs)
        i = template.pair_i[start:stop]
        j = template.pair_j[start:stop]
        cis = template.cis_pair[start:stop]
        kaa = _kernel_values(coordinates[0, i] - coordinates[0, j], template.r0, kernel)
        kab = _kernel_values(coordinates[0, i] - coordinates[1, j], template.r0, kernel)
        kba = _kernel_values(coordinates[1, i] - coordinates[0, j], template.r0, kernel)
        kbb = _kernel_values(coordinates[1, i] - coordinates[1, j], template.r0, kernel)
        mixture = np.where(cis, 0.5 * (p * (kaa + kbb) + (1.0 - p) * (kab + kba)), 0.25 * (kaa + kab + kba + kbb))
        rates[start:stop] = exposure[i] * exposure[j] * mixture
    if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise EntryError("candidate population rates are invalid")
    return rates


def _normalized_group(rates: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(rates[mask], dtype=np.float64)
    denominator = float(values.sum())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise EntryError("population rate group has no positive support: %s" % label)
    return values / denominator


def _kl_from_support(pi_true: np.ndarray, pi_fit: np.ndarray, label: str) -> tuple[float, int, int]:
    if pi_true.shape != pi_fit.shape:
        raise EntryError("population distributions have different shapes: %s" % label)
    positive = pi_true > 0.0
    zero_fit = positive & (pi_fit <= 0.0)
    if np.any(zero_fit):
        return math.inf, int(positive.sum()), int(zero_fit.sum())
    value = float(np.sum(pi_true[positive] * np.log(pi_true[positive] / pi_fit[positive])))
    if value < 0.0 and value > -1e-12:
        value = 0.0
    if not math.isfinite(value) or value < 0.0:
        raise EntryError("population KL is invalid: %s" % label)
    return value, int(positive.sum()), int(np.count_nonzero(pi_fit <= 0.0))


def _population_kl_for_endpoint(
    endpoint_id: str,
    endpoint: Mapping[str, Any],
    fixture_id: str,
    contract_path: Path,
    truth: np.ndarray,
    generating_exposure: np.ndarray,
    truth_metadata: Mapping[str, Any],
    candidate_coordinates: np.ndarray,
    template: Any,
) -> dict[str, Any]:
    frozen_generation = _load_snapshot_module("v1_calibration")
    generation_rates = frozen_generation.generation_rates
    p_true = float(truth_metadata.get("p_gen"))
    kernel = str(truth_metadata.get("generation_kernel"))
    if p_true != 0.8 or kernel != "v1":
        raise EntryError("synthetic truth generation metadata changed: %s" % fixture_id)
    p_fit = _load_scalar(_resolve_contract(endpoint["p_path"], contract_path), ("p", "final_p"))
    if not (0.0 < p_fit < 1.0):
        raise EntryError("candidate p is outside the open probability interval: %s" % endpoint_id)
    e_fit = _load_profile(_resolve_contract(endpoint["final_e_path"], contract_path), template)
    true_rates = generation_rates(template, truth, generating_exposure, p_true, kernel)
    fit_rates = _allow_zero_generation_rates(template, candidate_coordinates, e_fit, p_fit, kernel)
    groups = {
        "cis_offdiag": np.asarray(template.cis_pair, dtype=bool),
        "inter": ~np.asarray(template.cis_pair, dtype=bool),
    }
    rows = []
    for group, mask in groups.items():
        pi_true = _normalized_group(true_rates, mask, fixture_id + ".true." + group)
        pi_fit = _normalized_group(fit_rates, mask, endpoint_id + ".fit." + group)
        kl, n_positive_true, n_zero_fit = _kl_from_support(pi_true, pi_fit, endpoint_id + "." + group)
        rows.append({
            "endpoint_id": endpoint_id,
            "fixture_id": fixture_id,
            "group": group,
            "n_eligible_pairs": int(mask.sum()),
            "n_positive_true": n_positive_true,
            "n_zero_fit_on_true_support": n_zero_fit,
            "KL_nats": kl,
            "group_weight": GROUP_TOTALS[group],
            "same_bin_excluded": True,
            "zero_observed_pairs_included": True,
            "refit_e": False,
            "sampled_new_counts": False,
        })
    weighted = float("inf") if any(math.isinf(float(row["KL_nats"])) for row in rows) else float(sum(GROUP_TOTALS[row["group"]] * float(row["KL_nats"]) for row in rows) / GROUP_DENOMINATOR)
    rows.append({
        "endpoint_id": endpoint_id,
        "fixture_id": fixture_id,
        "group": "weighted_offdiag",
        "n_eligible_pairs": 184016 + 3312674,
        "n_positive_true": None,
        "n_zero_fit_on_true_support": None,
        "KL_nats": weighted,
        "group_weight": GROUP_DENOMINATOR,
        "same_bin_excluded": True,
        "zero_observed_pairs_included": True,
        "refit_e": False,
        "sampled_new_counts": False,
    })
    return {"rows": rows, "weighted_KL_nats_per_record": weighted, "p_fit": p_fit, "e_fit_positive_bins": int(np.count_nonzero(e_fit > 0.0))}


def _real_rows(contract_path: Path, contract: Mapping[str, Any], static: Mapping[str, Any], old_module: Any, masks: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for endpoint_id in REAL_ENDPOINT_IDS:
        endpoint = static["endpoints"][endpoint_id]
        structures = old_module.load_coordinates(_resolve_contract(endpoint["final_3dg_path"], contract_path), str(endpoint.get("coordinate_format", "3dg")))
        _validate_real_structures(old_module, structures, endpoint_id)
        eval_endpoint = {
            "id": endpoint_id,
            "variant_id": endpoint["method_id"],
            "source_id": endpoint["source_id"],
            "endpoint_status": endpoint["endpoint_status"],
            "coordinate_status": "available",
        }
        for chromosome, _length in old_module.CHROMOSOMES:
            row = old_module.evaluate_endpoint_chromosome(eval_endpoint, structures, masks[chromosome])
            row.update({"method_id": endpoint["method_id"], "source_id": endpoint["source_id"], "endpoint_sha256": endpoint["final_3dg_sha256"]})
            rows.append(row)
    return rows


def _synthetic_rows(contract_path: Path, contract: Mapping[str, Any], static: Mapping[str, Any], synthetic_prepare: Any, synthetic_evaluator: Any, truths: Mapping[str, Any]) -> list[dict[str, Any]]:
    template = synthetic_prepare.header_templates()[synthetic_prepare.FINAL_BIN]
    rows = []
    for endpoint_id in SYNTHETIC_ENDPOINT_IDS:
        endpoint = static["endpoints"][endpoint_id]
        fixture_id = endpoint["fixture_id"]
        truth = truths[fixture_id]["coordinates"]
        coords = _load_synthetic_candidate(synthetic_prepare, _resolve_contract(endpoint["final_3dg_path"], contract_path), endpoint_id, template)
        same_shape = fixture_id == "N2"
        for chromosome in range(20):
            record = synthetic_evaluator.chromosome_metrics(template, coords, truth, same_shape=same_shape, chromosome=chromosome)
            rows.append({
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
                "N_only_copy_difference_rms": record["N_only_copy_difference_rms"],
                "N_only_is_not_a_positive_score": True,
            })
    return rows


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise EntryError("cannot write an empty TSV")
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else _jsonable(row.get(key)) for key in keys})
    return len(rows)


def _run_historical_regression_gate(
    contract_path: Path,
    contract: Mapping[str, Any],
    static: Mapping[str, Any],
    old_module: Any,
    masks: Mapping[str, Mapping[str, Any]],
    legacy_structures: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    mask_exact: Mapping[str, Any],
) -> dict[str, Any]:
    """在完整 hash gate 后复算固定 old038 端点，仅作行为回归检查。"""
    configured = contract["historical_regression_gate"]
    registration = _historical_regression_gate_registration(contract, static)
    result = dict(registration)
    result["execution_phase"] = "after_all_endpoint_source_selection_and_frozen_mask_hashes"
    result["exact_frozen_mask_comparison"] = dict(mask_exact)
    result["published_table_path"] = configured["published_table_path"]
    result["published_table_sha256"] = configured["published_table_sha256"]
    if mask_exact.get("status") != "PASS" or mask_exact.get("same_positions_and_pair_bits") is not True:
        result.update({"status": "FAIL", "executed": False, "passed": False, "failure": "frozen mask positions/pair bits differ"})
        return result
    condition_id = str(configured["fixed_historical_condition_id"])
    if condition_id != str(old_module.HISTORICAL_020_ID):
        result.update({"status": "FAIL", "executed": False, "passed": False, "failure": "frozen historical condition id disagrees with legacy helper"})
        return result
    historical_structures = legacy_structures.get(condition_id)
    if historical_structures is None:
        result.update({"status": "FAIL", "executed": False, "passed": False, "failure": "fixed historical condition is missing from frozen mask structures"})
        return result
    published_path = _resolve_contract(configured["published_table_path"], contract_path)
    published_hash = _hash_expected(published_path, configured["published_table_sha256"], "historical_regression.published_table")
    if published_path.resolve() != Path(old_module.PUBLISHED_R2_PATH).resolve():
        raise EntryError("historical regression table path disagrees with frozen legacy helper")
    published_rows, published_provenance = old_module._read_published_r2_after_gate()
    endpoint = {
        "id": "old038-M0-B2-random",
        "variant_id": "old038-M0-B2",
        "source_id": "random_joint_base2207",
        "n_copies": 2,
        "endpoint_status": "historical_locked",
        "coordinate_status": "available",
    }
    historical_rows = [
        old_module.evaluate_endpoint_chromosome(endpoint, historical_structures, masks[chromosome])
        for chromosome, _length in old_module.CHROMOSOMES
    ]
    parity = old_module.compare_published_r2_rows(historical_rows, published_rows, endpoint_label=endpoint["id"])
    passed = parity.get("status") == "PASS" and mask_exact.get("same_positions_and_pair_bits") is True
    result.update({
        "status": "PASS" if passed else "FAIL",
        "executed": True,
        "passed": passed,
        "historical_coordinate_source": "frozen mask row for " + condition_id,
        "historical_coordinate_opened_after_hash_gate": True,
        "published_table_opened_after_hash_gate": True,
        "published_table_hash": published_hash["sha256"],
        "published_table_provenance": published_provenance,
        "chromosome_count": len(historical_rows),
        "compare_numeric_atol": configured["compare_numeric_atol"],
        "compared_fields": [list(item) for item in old_module.POST_ACCEPTANCE_NUMERIC_FIELDS],
        "parity": parity,
        "new_science_metric": False,
        "selection_input": False,
        "interpretation_status": "continue" if passed else "PAUSE_AND_DIAGNOSE",
    })
    return result


def _load_truth_payloads(contract_path: Path, contract: Mapping[str, Any], synthetic_prepare: Any) -> dict[str, dict[str, Any]]:
    truth_hashes = _hash_truth_files(contract_path, contract)
    result = {}
    for fixture_id in SYNTHETIC_FIXTURES:
        path = _resolve_contract(contract["synthetic_truth"][fixture_id]["coordinate_path"], contract_path)
        coordinates, exposure, metadata = synthetic_prepare._load_truth(path)
        coordinates = np.asarray(coordinates, dtype=np.float64)
        exposure = np.asarray(exposure, dtype=np.float64)
        if coordinates.shape != (2, 2645, 3) or exposure.shape != (2645,):
            raise EntryError("synthetic truth shape mismatch: %s" % fixture_id)
        result[fixture_id] = {"coordinates": coordinates, "exposure": exposure, "metadata": metadata, "hashes": truth_hashes[fixture_id]}
    return result


def evaluate_released(contract_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """执行正式 release；所有 payload 访问均位于 candidate/mask hash gate 之后。"""
    path, contract, static = _validate_release_contract(contract_path)
    gate = _hash_release_gate(path, contract, static)
    gate["sidecar_validation"] = _validate_release_sidecars(path, static)
    gate_path = path.parent / "hash_gate_evidence.json"
    _write_json(gate_path, gate)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sidecar_validation = gate["sidecar_validation"]
    _write_json(output / "sidecar_validation.json", sidecar_validation)
    old_module = _load_frozen_multires()
    mask_spec = contract["mask_lock"]
    legacy_config = {
        "grid": static["config"]["grid"],
        "mask_lock": {"manifest_path": mask_spec["manifest_path"], "expected_pair_counts_path": mask_spec["lock_path"]},
        "reference": {"path": contract["reference"]["path"]},
    }
    mask_lock = old_module._load_mask_lock(legacy_config)
    manifest_path, _manifest, mask_rows = old_module._mask_manifest_records(legacy_config)
    mask_structures = old_module.load_mask_structures(mask_rows, {"mask_hashes": gate["mask_files"]})
    reference_path = _resolve_contract(contract["reference"]["path"], path)
    reference_hash = _hash_expected(reference_path, contract["reference"]["sha256"], "reference")
    reference = old_module.load_coordinates(reference_path, "3dg")
    masks = old_module.build_frozen_masks(legacy_config, mask_structures, reference, mask_lock)
    legacy_masks, legacy_structures, legacy_provenance = old_module._legacy_rebuild_masks_after_gate(legacy_config, {"mask_rows": mask_rows, "mask_hashes": gate["mask_files"]})
    mask_exact = old_module.compare_mask_exact(masks, legacy_masks)
    if mask_exact.get("status") != "PASS":
        raise EntryError("old21 mask exact comparison failed")
    historical_regression = _run_historical_regression_gate(path, contract, static, old_module, masks, legacy_structures, mask_exact)
    _write_json(output / "historical_regression_gate.json", historical_regression)
    if historical_regression.get("status") != "PASS" or historical_regression.get("passed") is not True:
        raise EntryError("old038 historical regression gate failed; new metric interpretation is paused")
    synthetic_prepare, synthetic_evaluator = _load_synthetic_modules()
    truths = _load_truth_payloads(path, contract, synthetic_prepare)
    real_rows = _real_rows(path, contract, static, old_module, masks)
    synthetic_rows = _synthetic_rows(path, contract, static, synthetic_prepare, synthetic_evaluator, truths)
    kl_rows = []
    kl_summary = []
    template = synthetic_prepare.header_templates()[synthetic_prepare.FINAL_BIN]
    for endpoint_id in SYNTHETIC_ENDPOINT_IDS:
        endpoint = static["endpoints"][endpoint_id]
        fixture_id = endpoint["fixture_id"]
        coords = _load_synthetic_candidate(synthetic_prepare, _resolve_contract(endpoint["final_3dg_path"], path), endpoint_id, template)
        kl = _population_kl_for_endpoint(endpoint_id, endpoint, fixture_id, path, truths[fixture_id]["coordinates"], truths[fixture_id]["exposure"], truths[fixture_id]["metadata"], coords, template)
        kl_rows.extend(kl["rows"])
        kl_summary.append({"endpoint_id": endpoint_id, "fixture_id": fixture_id, "weighted_KL_nats_per_record": kl["weighted_KL_nats_per_record"], "p_fit": kl["p_fit"], "e_fit_positive_bins": kl["e_fit_positive_bins"], "used_for_selection": False})
    selected = contract["real_source_selection"]["selected_endpoint_by_method"]
    real_path = output / "r2_visibility_real_6endpoint_x20chr.tsv"
    synthetic_path = output / "r2_visibility_synthetic_6endpoint_x20chr.tsv"
    kl_path = output / "synthetic_population_predictive_kl.tsv"
    _write_tsv(real_path, real_rows)
    _write_tsv(synthetic_path, synthetic_rows)
    _write_tsv(kl_path, kl_rows)
    result = {
        "schema_version": "p9016-visibility-r2-evaluation-v1",
        "status": "completed",
        "release_contract": str(path),
        "scope": {"real_rows": len(real_rows), "synthetic_rows": len(synthetic_rows), "population_kl_rows": len(kl_rows)},
        "source_selected_endpoint_by_method": dict(selected),
        "main_comparisons": ["VZ-V0", "V1-VZ", "V1-V0"],
        "r2": {"metric_scope": list(R2_METRICS), "candidate_best_swap_per_chromosome": True, "reference_columns_fixed": True, "mask_exact": mask_exact, "mask_legacy_provenance": legacy_provenance},
        "sidecar_validation": sidecar_validation,
        "historical_regression_gate": historical_regression,
        "population_kl": {"synthetic_only": True, "not_current_sampled_counts_training_nll": True, "eta_improvement_and_zero_coverage_are_separate": True, "group_weights": dict(GROUP_TOTALS), "denominator": GROUP_DENOMINATOR, "same_bin_excluded": True, "all_eligible_zero_observed_pairs_included": True, "no_refit": True, "no_new_sampling": True, "summary": kl_summary},
        "historical_comparison": {"primary": "same-backend V0/VZ/V1", "old_038_m0_b2_same_source_r2": "historical reference only", "old_038_regression_executed": True, "old_038_recomputed_for_new_metric": False, "old_038_used_for_selection": False},
        "hash_gate": gate,
        "reference": {"path": str(reference_path), "sha256": reference_hash["sha256"], "opened_after_hash_gate": True},
        "truth": {fixture: {"coordinate_sha256": truths[fixture]["hashes"]["coordinate"]["sha256"], "metadata_sha256": truths[fixture]["hashes"]["metadata"]["sha256"], "opened_after_hash_gate": True} for fixture in SYNTHETIC_FIXTURES},
        "fit_called": False,
        "native_called": False,
        "output_paths": {"real_tsv": str(real_path), "synthetic_tsv": str(synthetic_path), "population_kl_tsv": str(kl_path), "sidecar_validation": str(output / "sidecar_validation.json")},
    }
    _write_json(output / "evaluation_results.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-preparation", action="store_true")
    group.add_argument("--population-kl-feasibility", action="store_true")
    group.add_argument("--validate-release")
    group.add_argument("--hash-gate")
    group.add_argument("--evaluate-released")
    parser.add_argument("--output-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate_preparation:
            result = validate_preparation()
            print(json.dumps({"status": result["status"], "prepared_only": result["prepared_only"], "population_kl": result["population_kl"]}, sort_keys=True))
            return 0
        if args.population_kl_feasibility:
            result = population_kl_feasibility()
            _write_json(FEASIBILITY_PATH, result)
            print(json.dumps({"status": result["status"], "exact": result["exact"], "weighted_denominator": result["weighted_denominator"]}, sort_keys=True))
            return 0 if result["exact"] else 2
        if args.validate_release:
            result = validate_release(args.validate_release)
            print(json.dumps({"status": result["status"], "metadata_only": result["metadata_only"]}, sort_keys=True))
            return 0
        if args.hash_gate:
            result = hash_gate(args.hash_gate)
            print(json.dumps({"status": "PASS", "candidate_count": len(result["candidate_files"]), "sidecar_status": result["sidecar_validation"]["status"], "reference_opened": result["reference_opened"]}, sort_keys=True))
            return 0
        if args.evaluate_released:
            if not args.output_dir:
                raise EntryError("--output-dir is required with --evaluate-released")
            result = evaluate_released(args.evaluate_released, args.output_dir)
            print(json.dumps({"status": result["status"], "real_rows": result["scope"]["real_rows"], "synthetic_rows": result["scope"]["synthetic_rows"]}, sort_keys=True))
            return 0
    except (EntryError, OSError, ValueError, KeyError, AssertionError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    raise EntryError("no operation selected")


if __name__ == "__main__":
    raise SystemExit(main())
