"""独立的 post-020 real-R2 评价器准备和 release gate。

本模块负责 ablation 侧的锁定/配置和结果整形。它复用 ``pr.r2comparison`` 中的纯 mask 与全基因组评价函数，不修改该模块。准备阶段从不打开坐标或 reference file。已发布路径会先哈希所有可用 candidate/control，再 arm evaluation gate 并加载 reference。
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from . import r2comparison
from .gate import EvalGate, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "docs/POST020_ALLELE_ABLATION_PROTOCOL.json"
AUTHORITATIVE_PROTOCOL_SHA256 = "b281037946775ee4271dbf33dd7e4b17dba5ddf4f9bd49eaf618f7c499c5361f"
REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
CONFIG_SCHEMA = "p9016-allele-r2-preparation-v1"
RELEASE_SCHEMA = "p9016-allele-r2-release-manifest-v1"
EVALUATION_SCHEMA = "p9016-allele-r2-evaluation-v1"
PREPARATION_SCHEMA = "p9016-allele-r2-preparation-manifest-v1"
EVAL_STAGE = "eval"
TIE_TOL = 1e-12
MIN_RHO_PAIRS = 20
BOOTSTRAP_SEED = 9301
BOOTSTRAP_DRAWS = 10_000
BIN_SIZE_BP = 1_000_000
GRID_OFFSET_BP = 3_000_000
VARIANT_IDS = ("C0", "C1", "C2-map", "C2-free", "C3")
BUNDLE_IDS = ("bundle1", "bundle2", "bundle3")
PRIMARY_VARIANT_COMPARISONS = (
    ("C1", "C0"),
    ("C2-map", "C0"),
    ("C2-free", "C0"),
    ("C2-free", "C2-map"),
    ("C3", "C0"),
)
PRIMARY_PAIR_METRICS = (
    "similarity", "cross", "contrast", "matched_ref1", "matched_ref2",
    "margin_mat", "margin_pat", "minmargin",
)

    # 这些只是不读取 coordinate files 的 lock records。准备阶段写出 paths 和 expected digests，
    # 但有意不 stat/hash/open 任何 coordinate files。
HISTORICAL_CONTROLS = (
    {
        "id": "softall_seed124101",
        "display_name": "Softall seed124101",
        "role": "historical_control",
        "endpoint_status": "accepted",
        "accepted_as": "softall_seed124101",
        "n_copies": 2,
        "format": "native_tsv",
        "path": "/work/phase3/test_res/124-20260905_185119-p9016_step_policy_1m/native/fixed_seed124101/softall/p9016_full.coords.tsv",
        "sha256": "b5366dcdcb5faaba25ad7ec1cf1ecd0de3e920982db233b1a4b20ee346605704",
    },
    {
        "id": "v1_original_random_joint",
        "display_name": "020 V1 random_joint",
        "role": "historical_control",
        "endpoint_status": "accepted",
        "accepted_as": "v1_original_random_joint",
        "n_copies": 2,
        "format": "3dg",
        "path": "/mnt/ssd/zliu/phase_restart/test_res/020-20260913_071841-v1-p9016-joint/selected.3dg",
        "sha256": "afb2d52ae11e342e9b43b3c8042c5581760d177c5e563f2478646ab36b3e7078",
    },
    {
        "id": "v1_continuation",
        "display_name": "022 V1 continuation 480",
        "role": "historical_control",
        "endpoint_status": "budget_not_converged",
        "accepted_as": "v1_continuation",
        "n_copies": 2,
        "format": "3dg",
        "path": "/mnt/ssd/zliu/phase_restart/test_res/022-20260913_111031-v1-continuation-fdg-r2/coords/random_joint/final-1m-continuation-61da99669694b822.3dg",
        "sha256": "49501f5b38d699fb2c9c8849616edd02b70fccd5ae39e1efe6036e17f5f620e7",
    },
    {
        "id": "fdg_proposal",
        "display_name": "022 FDG full proposal (rejected)",
        "role": "historical_control",
        "endpoint_status": "rejected_no_label_free_improvement",
        "accepted_as": "anchor",
        "anchor_condition_id": "v1_continuation",
        "n_copies": 2,
        "format": "3dg",
        "path": "/mnt/ssd/zliu/phase_restart/test_res/022-20260913_111031-v1-continuation-fdg-r2/fdg/mapped_proposal.3dg",
        "sha256": "44d8577de6faa034f002075b0b57e7e40de22a98468d05a822d89b22254361e5",
    },
    {
        "id": "random014",
        "display_name": "Random 014 control",
        "role": "historical_control",
        "endpoint_status": "accepted",
        "accepted_as": "random014",
        "n_copies": 2,
        "format": "3dg",
        "path": "/mnt/ssd/zliu/phase_restart/test_res/014-20260912_153000-s0-genome-wide-fixed/coords/random.3dg",
        "sha256": "9a48d73e1401e18349d11758e679da4c76da0904dbc467979079cb54bcd567d7",
    },
    {
        "id": "consensus014",
        "display_name": "Consensus 014 control",
        "role": "historical_control",
        "endpoint_status": "accepted",
        "accepted_as": "consensus014",
        "n_copies": 1,
        "format": "3dg",
        "path": "/mnt/ssd/zliu/phase_restart/test_res/014-20260912_153000-s0-genome-wide-fixed/coords/consensus.3dg",
        "sha256": "e76655732deb6b8386b1b77bc76ff45d7dba1384f6931337fee80d8f4aaa8e02",
        "similarity_policy": "mean(rho_A_mat,rho_A_pat), both finite required; contrast=n/a",
    },
)
HISTORICAL_IDS = tuple(item["id"] for item in HISTORICAL_CONTROLS)
FROZEN_CHROMOSOME_LENGTHS = (
    ("chr1", 195_471_971), ("chr2", 182_113_224), ("chr3", 160_039_680),
    ("chr4", 156_508_116), ("chr5", 151_834_684), ("chr6", 149_736_546),
    ("chr7", 145_441_459), ("chr8", 129_401_213), ("chr9", 124_595_110),
    ("chr10", 130_694_993), ("chr11", 122_082_543), ("chr12", 120_129_022),
    ("chr13", 120_421_639), ("chr14", 124_902_244), ("chr15", 104_043_685),
    ("chr16", 98_207_768), ("chr17", 94_987_271), ("chr18", 90_702_639),
    ("chr19", 61_431_566), ("chrX", 171_031_299),
)
TERMINAL_CONTROLLER_STATUSES = {
    "solver_converged", "budget_not_converged", "solver_failed", "failed_nonfinite",
    "failed_exception", "failed_source_snapshot", "failed_input",
}
FORMAL_ENDPOINT_SUCCESS_STATUSES = {"solver_converged", "budget_not_converged"}
FORMAL_ENDPOINT_FAILURE_STATUSES = {
    "solver_failed", "failed_nonfinite", "failed_exception", "failed_source_snapshot", "failed_input",
}
NEW_CONDITION_IDS = tuple(
    "%s-%s" % (variant_id, bundle_id)
    for variant_id in VARIANT_IDS for bundle_id in BUNDLE_IDS
)
ALL_CONDITION_IDS = NEW_CONDITION_IDS + HISTORICAL_IDS


class AlleleR2Error(RuntimeError):
    """ablation R2 preparation/release 契约无效时抛出。"""


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
        return value if math.isfinite(value) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except FileNotFoundError as exc:
        raise AlleleR2Error("missing JSON artifact: %s" % path) from exc
    except json.JSONDecodeError as exc:
        raise AlleleR2Error("invalid strict JSON artifact: %s" % path) from exc
    if not isinstance(value, Mapping):
        raise AlleleR2Error("JSON artifact must be an object: %s" % path)
    return dict(value)


def _reject_json_constant(token: str) -> Any:
    raise ValueError("non-finite JSON constant is forbidden: %s" % token)


def _write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise AlleleR2Error("refusing to overwrite artifact: %s" % path)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AlleleR2Error("%s must be a 64-character lowercase SHA256" % label)
    if any(char not in "0123456789abcdef" for char in value):
        raise AlleleR2Error("%s must be a lowercase SHA256" % label)
    return value


def _resolve(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _resolve_reference_path(path_text: str, base: Path) -> Path:
    """按 protocol 的 ``data/...`` 写法相对于本 checkout root 解析。"""
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "data":
        return (ROOT / path).resolve()
    return (base / path).resolve()


def _verify_authoritative_protocol(protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    actual = sha256_file(protocol_path)
    if actual != AUTHORITATIVE_PROTOCOL_SHA256:
        raise AlleleR2Error("authoritative protocol SHA256 mismatch: %s != %s" %
                            (actual, AUTHORITATIVE_PROTOCOL_SHA256))
    protocol = _read_json(protocol_path)
    if protocol.get("schema_version") != "post020-allele-ablation-protocol-v1":
        raise AlleleR2Error("unexpected protocol schema")
    if protocol.get("real_r2_only_evaluation_freeze", {}).get("reference", {}).get("sha256") != REFERENCE_SHA256:
        raise AlleleR2Error("protocol reference digest does not match frozen digest")
    return {"path": str(protocol_path.resolve()), "sha256": actual,
            "schema_version": protocol["schema_version"]}


def _new_condition_record(variant_id: str, bundle_id: str) -> dict[str, Any]:
    condition_id = "%s-%s" % (variant_id, bundle_id)
    return {
        "id": condition_id,
        "display_name": "%s %s" % (variant_id, bundle_id),
        "variant_id": variant_id,
        "bundle_id": bundle_id,
        "role": "new_variant",
        "endpoint_status": "awaiting_parent_release",
        "accepted_as": None,
        "n_copies": 2,
        "format": "3dg",
        "coordinate_status": "awaiting_release",
        "path": None,
        "sha256": None,
        "source_hashes_complete": False,
        "source_hashes": None,
        "selection": None,
        "failure_reason": None,
    }


def _historical_condition_record(item: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(item)
    record.update({
        "variant_id": None,
        "bundle_id": None,
        "coordinate_status": "awaiting_release",
        "source_hashes_complete": False,
        "source_hashes": None,
        "selection": {"status": "historical_locked", "criterion": "protocol_control"},
        "failure_reason": None,
    })
    return record


def build_prepared_config(protocol_info: Mapping[str, Any]) -> dict[str, Any]:
    conditions = [_new_condition_record(variant, bundle)
                  for variant in VARIANT_IDS for bundle in BUNDLE_IDS]
    conditions.extend(_historical_condition_record(item) for item in HISTORICAL_CONTROLS)
    return {
        "schema_version": CONFIG_SCHEMA,
        "status": "prepared_pending_release",
        "protocol_sha256": AUTHORITATIVE_PROTOCOL_SHA256,
        "protocol_path": protocol_info["path"],
        "cohort": {
            "sample_id": "P9016",
            "biological_samples": 1,
            "n_chromosomes": 20,
            "resolution_bp": BIN_SIZE_BP,
            "offset_bp": GRID_OFFSET_BP,
        },
        "grid": {
            "bin_size_bp": BIN_SIZE_BP,
            "offset_bp": GRID_OFFSET_BP,
            "position_rule": "range(3Mb, chromosome_length, 1Mb)",
            "chromosomes": [],
        },
        "conditions": conditions,
        "condition_ids": list(ALL_CONDITION_IDS),
        "new_condition_ids": list(NEW_CONDITION_IDS),
        "historical_control_ids": list(HISTORICAL_IDS),
        "comparisons": [
            {
                "label": "%s-%s" % (left, right),
                "left_variant": left,
                "right_variant": right,
                 "metric": "similarity",
                 "metrics": list(PRIMARY_PAIR_METRICS),
                 "primary": True,
            }
            for left, right in PRIMARY_VARIANT_COMPARISONS
        ],
        "primary_pair_metrics": list(PRIMARY_PAIR_METRICS),
        "primary_pair_count": len(PRIMARY_VARIANT_COMPARISONS) * len(PRIMARY_PAIR_METRICS),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "n_boot": BOOTSTRAP_DRAWS,
            "unit": "chromosome",
            "shared_matrix_required": True,
        },
        "reference": {
            "path": str((ROOT / "data/P9016.1m.3dg.gz").resolve()),
            "sha256": REFERENCE_SHA256,
            "role": "evaluation_only",
            "read_after": "all_21_condition_locks_and_available_coordinate_hashes",
        },
        "within_variant_representatives": None,
        "release_manifest_schema": RELEASE_SCHEMA,
        "output_subdirectory": "evaluation-r2",
        "read_boundary": {
            "prepare_only_opens": [str(PROTOCOL_PATH.resolve())],
            "prepare_only_does_not_open": ["candidate/control coordinate files", "reference 3DG", "phase pairs"],
            "reference_read_is_release_gated": True,
        },
        "planned_counts": {
            "new_variants": 5,
            "bundles_per_variant": 3,
            "new_condition_count": 15,
            "historical_control_count": 6,
            "condition_count": 21,
            "rows_after_release": "20 chromosomes x 21 conditions = 420, including failed n/a rows",
        },
    }


def prepare_only(output_dir: str | Path, *, protocol_path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    """写出不读取坐标的 preparation bundle。"""
    output = Path(output_dir).resolve()
    if output.exists():
        if any(output.iterdir()):
            raise AlleleR2Error("preparation output must be fresh and empty: %s" % output)
        raise AlleleR2Error("preparation output directory already exists: %s" % output)
    protocol_info = _verify_authoritative_protocol(Path(protocol_path).resolve())
    output.mkdir(parents=True, exist_ok=False)
    config = build_prepared_config(protocol_info)
    config_path = output / "config.json"
    _write_json(config_path, config)
    source_path = Path(__file__).resolve()
    manifest = {
        "schema_version": PREPARATION_SCHEMA,
        "status": "prepare_only_complete",
        "output_dir": str(output),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "protocol": protocol_info,
        "source": {"allele_r2.py": str(source_path), "sha256": sha256_file(source_path)},
        "expected_condition_count": len(ALL_CONDITION_IDS),
        "expected_new_condition_count": len(NEW_CONDITION_IDS),
        "expected_historical_control_count": len(HISTORICAL_IDS),
        "coordinate_files_opened": 0,
        "reference_opened": False,
        "phase_payload_opened": False,
        "optimizer_called": False,
        "release_required": True,
        "release_manifest_schema": RELEASE_SCHEMA,
        "note": "Paths and historical SHA values are lock metadata only; no coordinate/reference file was opened in prepare-only.",
    }
    _write_json(output / "prepare_manifest.json", manifest)
    readme = """# P9016 allele 消融 R2 评价器准备

状态：`prepare_only_complete`。本目录只准备锁和接口，没有打开真实 candidate/control 坐标、reference 3DG、phase pairs，也没有运行 fit 或 R2。

`config.json` 固定 15 个新条件（C0/C1/C2-map/C2-free/C3 × bundle1/2/3）和 6 个历史对照。父侧锁定全部 15 个端点后，生成 `release_manifest.json`，其中必须保存每个端点的终态、selection/source/coordinate 哈希、失败原因（若失败）以及每个变体的 count-selected representative。

`--evaluate-released` 会先验证权威 protocol SHA、prepared config SHA、21 条条件锁、1 Mb 网格、共同 bootstrap 规则和所有可用坐标 SHA；全部通过后才 arm evaluation gate 并读取 reference。失败的计划 arm 保留在 420 行输出中为 n/a，不从条件表或共同 mask 中消失。

主比较按 bundle 配对：先在每个 seed/20 条染色体上计算差值，再对三个 seed 在同一染色体求平均，最后计算宏观汇总；固定输出 `similarity(=matched)`、`cross`、`contrast`、`matched_ref1`、`matched_ref2`、`margin_mat`、`margin_pat`、`minmargin` 八个读数，即 5 个变体比较 × 8 = 40 项，每项保留三个 seed 和三 seed 汇总。任何计划 seed 失败时主三 seed 结果为 n/a，可用项结果另行标记。R2 不重新选择、不调参，不产生 R1/R3。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return manifest



def _terminal_job_record(experiment_root: Path, job_row: Mapping[str, Any],
                         terminal_by_id: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    job_id = str(job_row["job_id"])
    status_record = dict(terminal_by_id.get(job_id, {}))
    status_path = experiment_root / "jobs" / job_id / "status.json"
    if status_path.is_file():
        live_record = _read_json(status_path)
        merged = dict(status_record)
        merged.update(live_record)
        status_record = merged
    if not status_record:
        status_record = {"job_id": job_id, "status": job_row.get("status", "missing_terminal_record"),
                         "fit_called": bool(job_row.get("fit_called", False))}
    status = str(status_record.get("status", ""))
    if status not in TERMINAL_CONTROLLER_STATUSES:
        raise AlleleR2Error("experiment job %s is not terminal: %s" % (job_id, status or "missing"))
    final_path = status_record.get("final_path")
    if not final_path and status_record.get("attempt_path"):
        final_path = str(Path(str(status_record["attempt_path"])) / "final.json")
    final = None
    if final_path and Path(str(final_path)).is_file():
        final = _read_json(Path(str(final_path)))
    return status_record, final


def _source_hash_record(job_row: Mapping[str, Any], final: Mapping[str, Any] | None,
                        manifest_row: Mapping[str, Any]) -> dict[str, str]:
    source: dict[str, str] = {}
    for key, output_key in (
        ("source_snpfree_sha256", "source_snpfree"),
        ("source_snapshot_sha256", "source_snapshot"),
        ("synthetic_gate_sha256", "synthetic_gate"),
        ("x0_file_sha256", "x0_file"),
        ("x0_coordinate_sha256", "x0_coordinate"),
        ("protocol_release_sha256", "protocol_release"),
    ):
        value = job_row.get(key)
        if value is not None:
            source[output_key] = _require_sha(value, "job.%s" % key)
    if final:
        final_source = final.get("source", {})
        final_gate = final.get("synthetic_gate", {})
        final_x0 = final.get("x0", {})
        for output_key, block, key in (
            ("source_snpfree", final_source, "sha256"),
            ("source_snapshot", final_source, "snapshot_sha256"),
            ("synthetic_gate", final_gate, "sha256"),
            ("x0_file", final_x0, "file_sha256"),
            ("x0_coordinate", final_x0, "coordinate_sha256"),
        ):
            value = block.get(key) if isinstance(block, Mapping) else None
            if value is not None:
                source_value = _require_sha(value, "final.%s" % output_key)
                if output_key in source and source[output_key] != source_value:
                    raise AlleleR2Error("source hash disagreement for %s" % output_key)
                source[output_key] = source_value
    job_config_sha = manifest_row.get("job_config_sha256")
    if job_config_sha is not None:
        source["job_config"] = _require_sha(job_config_sha, "manifest.job_config_sha256")
    return source


def build_release_manifest_from_experiment(
    experiment_root: str | Path,
    prepared_config_path: str | Path,
    *,
    grid: Sequence[Mapping[str, Any]] = tuple({"name": name, "length_bp": length}
                                                for name, length in FROZEN_CHROMOSOME_LENGTHS),
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """将 B 的 finalized controller artifacts 适配为 R2 release lock。

    该适配器有意独立于 training controller。它拒绝保留的 027 ``PREDECESSOR_NO_FIT.json`` 以及任何非全新或非终态 experiment。它只读取 controller/final JSON 和未来全新运行中的 candidate coordinate bytes；绝不读取 phase pairs 或 reference。返回的 lock 即使某个 terminal failure 没有 coordinate，也保留全部 15 个 arms，并明确将该 arm 标记为 ``failed``。
    """
    root = Path(experiment_root).resolve()
    config_path = Path(prepared_config_path).resolve()
    config = _read_json(config_path)
    _validate_config(config, config_path)
    predecessor_path = root / "PREDECESSOR_NO_FIT.json"
    if predecessor_path.is_file():
        predecessor = _read_json(predecessor_path)
        if (predecessor.get("reusable_as_final_real_run") is False or
                predecessor.get("current_controller_requires_fresh_run") is True or
                predecessor.get("role") == "no_fit_preparation_predecessor"):
            raise AlleleR2Error("refusing to activate retained no-fit predecessor: %s" % root)
    manifest_path = root / "manifest.json"
    selection_path = root / "selection.json"
    termination_path = root / "termination_audit.json"
    manifest = _read_json(manifest_path)
    selection = _read_json(selection_path)
    termination = _read_json(termination_path)
    if manifest.get("status") in ("pending_inputs", "partial"):
        raise AlleleR2Error("experiment is no-fit/pending or partial; fresh terminal run required")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(NEW_CONDITION_IDS):
        raise AlleleR2Error("experiment manifest must retain all 15 jobs")
    job_ids = [str(item.get("job_id")) for item in jobs]
    if set(job_ids) != set(NEW_CONDITION_IDS):
        raise AlleleR2Error("experiment job IDs do not match C*-bundle1/2/3 contract")
    if termination.get("status") != "complete" or termination.get("job_count") != 15 or termination.get("terminal_job_count") != 15:
        raise AlleleR2Error("termination_audit is not a complete 15-job terminal record")
    if termination.get("active_jobs"):
        raise AlleleR2Error("termination_audit still has active jobs")
    terminal_by_id = {str(item["job_id"]): item for item in termination.get("jobs", [])}
    if set(terminal_by_id) != set(NEW_CONDITION_IDS):
        raise AlleleR2Error("termination_audit does not preserve all 15 job statuses")
    selected_by_variant: dict[str, str | None] = {}
    selection_by_id: dict[str, Mapping[str, Any]] = {}
    selection_variant_by_id: dict[str, Mapping[str, Any]] = {}
    for variant in selection.get("variants", []):
        variant_id = str(variant.get("model_id"))
        selected_bundle = variant.get("selected_bundle_id")
        selected_by_variant[variant_id] = None if selected_bundle is None else str(selected_bundle)
        selection_variant_by_id[variant_id] = variant
        for candidate in variant.get("candidates", []):
            condition_id = "%s-%s" % (variant_id, candidate.get("bundle_id"))
            selection_by_id[condition_id] = candidate
    if set(selected_by_variant) != set(VARIANT_IDS):
        raise AlleleR2Error("selection.json does not retain all five within-variant groups")
    release_conditions: dict[str, dict[str, Any]] = {}
    jobs_by_id = {str(item["job_id"]): item for item in jobs}
    for condition_id in NEW_CONDITION_IDS:
        job = jobs_by_id[condition_id]
        job_config_path = _resolve(str(job.get("job_config")), root)
        if not job_config_path.is_file():
            raise AlleleR2Error("missing controller job config for %s" % condition_id)
        job_payload = _read_json(job_config_path)
        variant_id = str(job["model_id"])
        bundle_id = str(job["bundle_id"])
        status_record, final = _terminal_job_record(root, job, terminal_by_id)
        source_hashes = _source_hash_record(job_payload, final, job)
        if not source_hashes:
            raise AlleleR2Error("job %s has no source hashes" % condition_id)
        candidate = selection_by_id.get(condition_id, {})
        selected = selected_by_variant.get(variant_id) == bundle_id
        selection_record = {
            "status": "count_selected" if selected else "terminal_candidate",
            "criterion": str(selection_variant_by_id[variant_id].get("criterion", "count_nll_normalized")),
            "planned_seed_count": 3,
            "bundle_id": bundle_id,
            "selected_for_variant": bool(selected),
            "selection_eligible": bool(candidate.get("selection_eligible", status_record.get("selection_eligible", False))),
            "count_nll_normalized": candidate.get("count_nll_normalized"),
            "excluded_reason": candidate.get("excluded_reason"),
        }
        coordinate_record = final.get("final", {}).get("coordinates_file") if final else None
        coordinate = None
        coordinate_status = "failed"
        failure_reason = candidate.get("excluded_reason") or status_record.get("status") or "no_terminal_coordinate"
        if isinstance(coordinate_record, Mapping) and coordinate_record.get("path") and coordinate_record.get("sha256"):
            coordinate_path = Path(str(coordinate_record["path"])).resolve()
            if not coordinate_path.is_file():
                raise AlleleR2Error("terminal coordinate missing for %s: %s" % (condition_id, coordinate_path))
            actual_coordinate_sha = sha256_file(coordinate_path)
            recorded_coordinate_sha = _require_sha(coordinate_record["sha256"], "%s.coordinates_file.sha256" % condition_id)
            if actual_coordinate_sha != recorded_coordinate_sha:
                raise AlleleR2Error("terminal coordinate hash mismatch for %s" % condition_id)
            coordinate = {"path": str(coordinate_path), "sha256": actual_coordinate_sha,
                          "format": "3dg"}
            coordinate_status = "available"
            failure_reason = None
        release_conditions[condition_id] = {
            "id": condition_id,
            "display_name": "%s %s" % (variant_id, bundle_id),
            "variant_id": variant_id,
            "bundle_id": bundle_id,
            "role": "new_variant",
            "endpoint_status": str(status_record.get("status")),
            "accepted_as": condition_id,
            "n_copies": 2,
            "format": "3dg",
            "coordinate_status": coordinate_status,
            "coordinate": coordinate,
            "source_hashes_complete": True,
            "source_hashes": source_hashes,
            "selection": selection_record,
            "failure_reason": failure_reason,
            "controller_job_id": condition_id,
            "fit_called": bool(status_record.get("fit_called", final.get("fit_called", False) if final else False)),
        }
    for historical in HISTORICAL_CONTROLS:
        release_conditions[historical["id"]] = {
            **_historical_condition_record(historical),
            "coordinate_status": "available",
            "coordinate": {"path": historical["path"], "sha256": historical["sha256"], "format": historical["format"]},
            "source_hashes_complete": True,
            "source_hashes": {"historical_coordinate": historical["sha256"]},
            "selection": {"status": "historical_locked", "criterion": "protocol_control"},
        }
    representative_lock = {}
    for variant_id in VARIANT_IDS:
        bundle_id = selected_by_variant.get(variant_id)
        if bundle_id is None:
            representative_lock[variant_id] = None
        else:
            representative_lock[variant_id] = "%s-%s" % (variant_id, bundle_id)
    chromosome_grid = [{"name": str(item["name"]), "length_bp": int(item["length_bp"])} for item in grid]
    if len(chromosome_grid) != 20:
        raise AlleleR2Error("release grid must contain 20 chromosomes")
    release = {
        "schema_version": RELEASE_SCHEMA,
        "status": "locked_for_evaluation",
        "prepared_config_sha256": sha256_file(config_path),
        "protocol_sha256": AUTHORITATIVE_PROTOCOL_SHA256,
        "controller": {
            "experiment_root": str(root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "selection_path": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
            "termination_audit_path": str(termination_path),
            "termination_audit_sha256": sha256_file(termination_path),
            "fresh_run_required": True,
            "predecessor_rejected": not predecessor_path.is_file(),
            "all_15_manifest_arms_preserved": True,
        },
        "grid": {"bin_size_bp": BIN_SIZE_BP, "offset_bp": GRID_OFFSET_BP,
                 "position_rule": "range(3Mb, chromosome_length, 1Mb)", "chromosomes": chromosome_grid},
        "conditions": [release_conditions[item] for item in ALL_CONDITION_IDS],
        "within_variant_representatives": representative_lock,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "n_boot": BOOTSTRAP_DRAWS,
                      "unit": "chromosome", "shared_matrix_required": True},
        "reference": {
            "path": config["reference"]["path"], "sha256": REFERENCE_SHA256,
            "read_after_candidate_hash_lock": True,
        },
        "r2_policy": {
            "reselect": False,
            "retune": False,
            "reference_columns_fixed": ["mat", "pat"],
            "failed_planned_arm_policy": "retain_n/a_and_exclude_from_common_mask_with_reason",
        },
    }
    if output_path is not None:
        _write_json(Path(output_path).resolve(), release)
    return release



def _validate_config(config: Mapping[str, Any], config_path: Path) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise AlleleR2Error("unsupported preparation config schema")
    if config.get("status") != "prepared_pending_release":
        raise AlleleR2Error("config must remain prepared_pending_release before release")
    current_protocol = _verify_authoritative_protocol()
    if config.get("protocol_sha256") != current_protocol["sha256"]:
        raise AlleleR2Error("prepared config protocol SHA does not match current authoritative protocol")
    if config.get("cohort", {}).get("sample_id") != "P9016":
        raise AlleleR2Error("config cohort must be P9016")
    grid = config.get("grid", {})
    if grid.get("bin_size_bp") != BIN_SIZE_BP or grid.get("offset_bp") != GRID_OFFSET_BP:
        raise AlleleR2Error("config grid must be 1 Mb with 3 Mb offset")
    raw_conditions = config.get("conditions")
    if not isinstance(raw_conditions, list) or [item.get("id") for item in raw_conditions] != list(ALL_CONDITION_IDS):
        raise AlleleR2Error("config must preserve all 21 conditions in frozen order")
    reference = config.get("reference", {})
    if reference.get("sha256") != REFERENCE_SHA256:
        raise AlleleR2Error("config reference SHA does not match frozen P9016 reference")
    if config_path.name != "config.json":
        raise AlleleR2Error("release config must be the prepared config.json")
    if config.get("primary_pair_metrics") != list(PRIMARY_PAIR_METRICS):
        raise AlleleR2Error("config primary pair metrics do not match the frozen eight-metric scope")
    if config.get("primary_pair_count") != len(PRIMARY_VARIANT_COMPARISONS) * len(PRIMARY_PAIR_METRICS):
        raise AlleleR2Error("config primary pair count does not match the frozen scope")


def _condition_release_coordinate(item: Mapping[str, Any], label: str) -> dict[str, Any] | None:
    coordinate = item.get("coordinate")
    if isinstance(coordinate, Mapping):
        path = coordinate.get("path")
        digest = coordinate.get("sha256")
        format_name = coordinate.get("format", item.get("format", "3dg"))
    else:
        path = item.get("path")
        digest = item.get("sha256")
        format_name = item.get("format", "3dg")
    if path is None and digest is None:
        return None
    if not isinstance(path, str) or not path:
        raise AlleleR2Error("%s coordinate.path is required when coordinate is present" % label)
    return {"path": path, "sha256": _require_sha(digest, "%s.coordinate.sha256" % label),
            "format": str(format_name)}


def _validate_source_hashes(item: Mapping[str, Any], label: str) -> None:
    if item.get("source_hashes_complete") is not True:
        raise AlleleR2Error("%s source_hashes_complete must be true" % label)
    hashes = item.get("source_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        raise AlleleR2Error("%s source_hashes must be a non-empty mapping" % label)
    for key, value in hashes.items():
        _require_sha(value, "%s.source_hashes.%s" % (label, key))


def _validate_release_manifest(config: Mapping[str, Any], config_path: Path,
                               release_path: Path) -> dict[str, Any]:
    release = _read_json(release_path)
    if release.get("schema_version") != RELEASE_SCHEMA:
        raise AlleleR2Error("release manifest schema mismatch")
    if release.get("status") not in ("locked_for_evaluation", "released"):
        raise AlleleR2Error("release manifest must be locked_for_evaluation/released")
    current_protocol = _verify_authoritative_protocol()
    if release.get("protocol_sha256") != current_protocol["sha256"]:
        raise AlleleR2Error("release protocol SHA mismatch")
    expected_config_sha = _require_sha(release.get("prepared_config_sha256"), "prepared_config_sha256")
    actual_config_sha = sha256_file(config_path)
    if expected_config_sha != actual_config_sha:
        raise AlleleR2Error("prepared config SHA mismatch: %s != %s" %
                            (actual_config_sha, expected_config_sha))
    grid = release.get("grid")
    if not isinstance(grid, Mapping):
        raise AlleleR2Error("release grid is required")
    if grid.get("bin_size_bp") != BIN_SIZE_BP or grid.get("offset_bp") != GRID_OFFSET_BP:
        raise AlleleR2Error("release grid must be 1 Mb with 3 Mb offset")
    chromosomes = grid.get("chromosomes")
    if not isinstance(chromosomes, list) or len(chromosomes) != 20:
        raise AlleleR2Error("release grid must contain exactly 20 chromosomes")
    expected_names = ["chr%d" % index for index in range(1, 20)] + ["chrX"]
    if [item.get("name") for item in chromosomes] != expected_names:
        raise AlleleR2Error("release chromosome order must be chr1..chr19,chrX")
    for item in chromosomes:
        if not isinstance(item.get("length_bp"), (int, np.integer)) or int(item["length_bp"]) <= GRID_OFFSET_BP:
            raise AlleleR2Error("invalid chromosome length in release grid")
    bootstrap = release.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or bootstrap.get("seed") != BOOTSTRAP_SEED or bootstrap.get("n_boot") != BOOTSTRAP_DRAWS:
        raise AlleleR2Error("release bootstrap must be seed9301/n_boot10000")
    reference = release.get("reference")
    config_reference = config.get("reference", {})
    if not isinstance(reference, Mapping):
        raise AlleleR2Error("release reference lock is required")
    release_reference_path = _resolve_reference_path(str(reference.get("path", "")), release_path.parent)
    config_reference_path = _resolve_reference_path(str(config_reference.get("path", "")), ROOT)
    if release_reference_path != config_reference_path:
        raise AlleleR2Error("release reference path differs from prepared config")
    if reference.get("sha256") != REFERENCE_SHA256 or reference.get("read_after_candidate_hash_lock") is not True:
        raise AlleleR2Error("release reference lock is invalid")

    raw_config = {item["id"]: item for item in config["conditions"]}
    raw_release = release.get("conditions")
    if not isinstance(raw_release, list) or [item.get("id") for item in raw_release] != list(ALL_CONDITION_IDS):
        raise AlleleR2Error("release must preserve all 21 condition records in frozen order")
    conditions = []
    for item in raw_release:
        condition_id = str(item.get("id"))
        base = raw_config[condition_id]
        label = "condition:%s" % condition_id
        if item.get("n_copies") != base.get("n_copies"):
            raise AlleleR2Error("%s n_copies changed from prepared config" % label)
        if not isinstance(item.get("endpoint_status"), str) or not item["endpoint_status"]:
            raise AlleleR2Error("%s endpoint_status is required" % label)
        _validate_source_hashes(item, label)
        selection = item.get("selection")
        if not isinstance(selection, Mapping) or not isinstance(selection.get("status"), str) or not selection.get("status"):
            raise AlleleR2Error("%s selection status is required" % label)
        if not isinstance(selection.get("criterion"), str) or not selection.get("criterion"):
            raise AlleleR2Error("%s selection criterion is required" % label)
        coordinate_status = str(item.get("coordinate_status", ""))
        if coordinate_status not in ("available", "failed"):
            raise AlleleR2Error("%s coordinate_status must be available or failed" % label)
        coordinate = _condition_release_coordinate(item, label)
        if coordinate_status == "available" and coordinate is None:
            raise AlleleR2Error("%s available condition lacks coordinate hash" % label)
        if coordinate_status == "failed":
            if condition_id in HISTORICAL_IDS:
                raise AlleleR2Error("historical control cannot be marked failed: %s" % condition_id)
            if not isinstance(item.get("failure_reason"), str) or not item["failure_reason"]:
                raise AlleleR2Error("%s failed condition must retain failure_reason" % label)
            if coordinate is not None:
                raise AlleleR2Error("%s failed condition must not supply a coordinate" % label)
        else:
            if condition_id in NEW_CONDITION_IDS:
                if selection.get("planned_seed_count") != 3:
                    raise AlleleR2Error("%s must retain planned_seed_count=3" % label)
                if selection.get("bundle_id") != base.get("bundle_id"):
                    raise AlleleR2Error("%s selection bundle does not match prepared bundle" % label)
        normalized = dict(item)
        normalized.update({"id": condition_id, "coordinate": coordinate,
                           "coordinate_status": coordinate_status,
                           "selection": dict(selection),
                           "config_record": base})
        conditions.append(normalized)

    representatives = release.get("within_variant_representatives")
    if not isinstance(representatives, Mapping) or set(representatives) != set(VARIANT_IDS):
        raise AlleleR2Error("release must provide one representative for each of five variants")
    by_id = {item["id"]: item for item in conditions}
    representative_records = {}
    for variant_id in VARIANT_IDS:
        representative_id = representatives[variant_id]
        if representative_id is None:
            representative_records[variant_id] = {
                "condition_id": None,
                "bundle_id": None,
                "selection": None,
                "status": "n/a",
                "reason": "no_count_selected_terminal_candidate",
            }
            continue
        if not isinstance(representative_id, str) or representative_id not in NEW_CONDITION_IDS or not representative_id.startswith(variant_id + "-"):
            raise AlleleR2Error("invalid representative for %s: %s" % (variant_id, representative_id))
        item = by_id[representative_id]
        selection = item["selection"]
        status_text = str(selection.get("status", "")).lower().replace("-", "_")
        criterion = str(selection.get("criterion", "")).lower()
        if "count" not in status_text or "count" not in criterion or "nll" not in criterion:
            raise AlleleR2Error("representative %s is not explicitly count-selected" % representative_id)
        representative_records[variant_id] = {
            "condition_id": representative_id,
            "bundle_id": item.get("bundle_id"),
            "selection": dict(selection),
        }
    return {
        "manifest": release,
        "protocol": current_protocol,
        "grid": [{"name": str(item["name"]), "length_bp": int(item["length_bp"])} for item in chromosomes],
        "conditions": conditions,
        "condition_by_id": by_id,
        "representatives": representative_records,
        "reference": dict(reference),
    }


def _parse_3dg(path: Path) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                track = str(fields[0])
                position = int(fields[1])
                point = np.asarray([float(item) for item in fields[2:5]], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise AlleleR2Error("invalid 3dg row %d in %s" % (line_no, path)) from exc
            if point.shape != (3,):
                raise AlleleR2Error("invalid 3dg coordinate shape row %d in %s" % (line_no, path))
            if not np.isfinite(point).all():
                # 保留该行的非有限状态，让 shared-mask evaluator 报告该 arm 为 n/a，
                # 而不是中止整个 run。
                point = np.full(3, np.nan, dtype=np.float64)
            target = result.setdefault(track, {})
            if position in target:
                raise AlleleR2Error("duplicate coordinate %s:%d" % (track, position))
            target[position] = point
    return result


def _parse_native_tsv(path: Path, chromosome_names: Sequence[str]) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {}
    index = {name: index for index, name in enumerate(chromosome_names)}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chr", "copy", "start", "x", "y", "z"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise AlleleR2Error("native TSV lacks required fields: %s" % sorted(required))
        for line_no, row in enumerate(reader, start=2):
            try:
                chromosome = str(row["chr"])
                chromosome_index = index[chromosome]
                copy_index = int(row["copy"])
                position = int(row["start"])
                point = np.asarray([float(row[axis]) for axis in ("x", "y", "z")], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise AlleleR2Error("invalid native TSV row %d in %s" % (line_no, path)) from exc
            if copy_index not in (0, 1) or position < 0 or position % BIN_SIZE_BP != 0:
                raise AlleleR2Error("invalid native TSV chromosome/copy/start row %d" % line_no)
            if not np.isfinite(point).all():
                # 在 common mask 中保留缺失的 bead；不要丢弃 planned arm，
                # 也不要让其他 conditions 的 mask 变空。
                point = np.full(3, np.nan, dtype=np.float64)
            track = "c%02d%s" % (chromosome_index + 1, "ab"[copy_index])
            target = result.setdefault(track, {})
            if position in target:
                raise AlleleR2Error("duplicate coordinate %s:%d" % (track, position))
            target[position] = point
    return result


def _load_coordinates(path: Path, format_name: str,
                      chromosome_names: Sequence[str]) -> dict[str, dict[int, np.ndarray]]:
    if format_name in ("3dg", "3dg_text"):
        return _parse_3dg(path)
    if format_name in ("native_tsv", "coords_tsv"):
        return _parse_native_tsv(path, chromosome_names)
    raise AlleleR2Error("unsupported coordinate format: %s" % format_name)


def _lock_candidate_files(locked: Mapping[str, Any], output_dir: Path) -> tuple[dict[str, dict[str, dict[int, np.ndarray]]], EvalGate, dict[str, Any], dict[str, dict[int, np.ndarray]]]:
    """在解析任何内容或打开 reference 前，对所有可用 candidate/control 计算哈希。"""
    conditions = locked["conditions"]
    gate = EvalGate(str(output_dir / "gate.json"))
    resolved: dict[str, tuple[Path, str]] = {}
    lock_records = []
    excluded_from_mask = []
    # 此循环只读取 candidate/control 哈希。在整个循环完成前，不导入或访问 reference path。
    for item in conditions:
        if item["coordinate_status"] != "available":
            continue
        coordinate = item["coordinate"]
        path = _resolve(coordinate["path"], output_dir)
        if not path.is_file():
            raise AlleleR2Error("missing locked coordinate for %s: %s" % (item["id"], path))
        actual = sha256_file(path)
        if actual != coordinate["sha256"]:
            raise AlleleR2Error("coordinate SHA mismatch for %s: %s != %s" %
                                (item["id"], actual, coordinate["sha256"]))
        registered = gate.register(EVAL_STAGE, "r2:%s" % item["id"], str(path))
        if registered != actual:
            raise AlleleR2Error("gate digest mismatch for %s" % item["id"])
        if _release_mask_eligible(item):
            resolved[item["id"]] = (path, coordinate["format"])
        else:
            excluded_from_mask.append({
                "condition_id": item["id"],
                "reason": "endpoint_not_formally_successful_for_primary_mask",
                "endpoint_status": item["endpoint_status"],
                "coordinate_hash_verified": True,
            })
        lock_records.append({
            "condition_id": item["id"],
            "path": str(path),
            "sha256": actual,
            "format": coordinate["format"],
            "source_hashes": item["source_hashes"],
            "selection": item["selection"],
            "endpoint_status": item["endpoint_status"],
            "mask_included": _release_mask_eligible(item),
        })
    if not any(item["id"] in HISTORICAL_IDS for item in conditions if item["coordinate_status"] == "available"):
        raise AlleleR2Error("at least one historical control must be hash-locked before reference access")
    chromosome_names = [item["name"] for item in locked["grid"]]
    # 所有 hashes 通过后，才解析 coordinate payloads。
    loaded = {
        condition_id: _load_coordinates(path, format_name, chromosome_names)
        for condition_id, (path, format_name) in resolved.items()
    }
    gate.arm(EVAL_STAGE)
    reference_spec = locked["reference"]
    reference_path = _resolve_reference_path(str(reference_spec["path"]), output_dir)
    if not reference_path.is_file():
        raise AlleleR2Error("locked reference file is missing: %s" % reference_path)
    reference_actual = sha256_file(reference_path)
    if reference_actual != reference_spec["sha256"]:
        raise AlleleR2Error("reference SHA mismatch: %s != %s" %
                            (reference_actual, reference_spec["sha256"]))
    # 延迟 import 和 open：只有所有 candidate hashes 与 coordinate parsing 成功后才 arm gate。
    from . import ref3dg  # pylint: disable=import-outside-toplevel
    reference = ref3dg.load_reference(gate, path=str(reference_path))
    return loaded, gate, {
        "coordinates": lock_records,
        "excluded_from_mask": excluded_from_mask,
        "reference": {"path": str(reference_path), "sha256": reference_actual,
                       "loaded_after_gate_arm": True},
        "reference_loaded_after_all_coordinate_hashes": True,
    }, reference


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean_optional(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if _finite(value)]
    return None if not finite else float(np.mean(finite))


def _is_formal_endpoint_failure(row: Mapping[str, Any]) -> bool:
    status = row.get("endpoint_status")
    if status is None:
        return False
    status_text = str(status)
    return status_text in FORMAL_ENDPOINT_FAILURE_STATUSES or status_text.startswith("failed")


def _is_formal_endpoint_success(row: Mapping[str, Any]) -> bool:
    status = row.get("endpoint_status")
    # Synthetic rows 有意省略 controller endpoint status；real rows 必须是
    # solver_converged 或 budget_not_converged 才能进入 primary R2。
    return status is None or str(status) in FORMAL_ENDPOINT_SUCCESS_STATUSES


def _metric_is_formally_usable(row: Mapping[str, Any], metric: str) -> bool:
    if _is_formal_endpoint_failure(row) or row.get("descriptive_only"):
        return False
    value = row.get(metric)
    if not _finite(value):
        return False
    if metric in ("margin_mat", "margin_pat", "minmargin") and row.get("geometry_tie"):
        return False
    return True


def _release_mask_eligible(condition: Mapping[str, Any]) -> bool:
    if condition.get("coordinate_status") != "available":
        return False
    if condition.get("id") in HISTORICAL_IDS:
        return True
    return str(condition.get("endpoint_status", "")) in FORMAL_ENDPOINT_SUCCESS_STATUSES


def _raw_four(row: Mapping[str, Any]) -> dict[str, float | None]:
    return {key: (float(row[key]) if _finite(row.get(key)) else None)
            for key in ("rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat")}



def _derive_two_copy(row: dict[str, Any]) -> dict[str, Any]:
    raw = _raw_four(row)
    row["raw_four_rho"] = raw
    row["direct_original"] = None
    row["cross_original"] = None
    row["cross_matched"] = None
    row["matched_ref1"] = None
    row["matched_ref2"] = None
    row["fixed_reference_abcd"] = None
    row["margin_mat"] = None
    row["margin_pat"] = None
    row["margin_ref1_mat"] = None
    row["margin_ref2_pat"] = None
    row["minmargin"] = None
    row["matched_rho_candidate_A"] = None
    row["matched_rho_candidate_B"] = None
    row["copy_matched"] = None
    row["both_positive"] = False
    row["one_negative"] = False
    row["both_negative"] = False
    row["margin_tie"] = False
    row["geometry_tie"] = False
    row["margin_class"] = None
    row["missing_count"] = sum(value is None for value in raw.values())
    if row["missing_count"]:
        row.update({"matched": None, "cross": None, "other": None, "contrast": None,
                    "similarity": None, "orientation": "unresolved_missing",
                    "pairing": None, "geometry_status": "unavailable",
                    "metric_status": "unavailable"})
        if not row.get("reason"):
            row["reason"] = "; ".join("%s=%s" % (key, row.get("rho_reasons", {}).get(key, "missing"))
                                      for key, value in raw.items() if value is None)
        return row
    a_mat, a_pat, b_mat, b_pat = (raw[key] for key in ("rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat"))
    direct = (a_mat + b_pat) / 2.0
    cross = (a_pat + b_mat) / 2.0
    row["direct_original"] = float(direct)
    row["cross_original"] = float(cross)
    if abs(direct - cross) <= TIE_TOL:
        symmetric = float((direct + cross) / 2.0)
        row.update({
            "matched": symmetric,
            "cross": symmetric,
            "other": symmetric,
            "contrast": 0.0,
            "similarity": symmetric,
            "orientation": "unresolved_tie",
            "pairing": "tie_average",
            "geometry_status": "unresolved_tie",
            "metric_status": "ok",
            "geometry_tie": True,
            "cross_matched": symmetric,
            "matched_ref1": None,
            "matched_ref2": None,
            "margin_class": "geometry_tie",
            "reason": "direct_cross_tie_within_1e-12",
        })
        return row
    if direct > cross:
        # Candidate rows A/B 保持原始顺序。
        abcd = (a_mat, a_pat, b_mat, b_pat)
        matched, other = direct, cross
        matched_a, matched_b = a_mat, b_pat
        orientation, pairing = "direct", "direct"
    else:
        # 只移动 candidate rows；固定的 reference columns 保持 mat/pat。
        abcd = (b_mat, b_pat, a_mat, a_pat)
        matched, other = cross, direct
        matched_a, matched_b = a_pat, b_mat
        orientation, pairing = "swapped", "cross"
    a, b, c, d = abcd
    margin_mat = a - b
    margin_pat = d - c
    row.update({
        "fixed_reference_abcd": [float(value) for value in abcd],
        "cross_matched": float(other),
        "matched_ref1": float(a),
        "matched_ref2": float(d),
        "matched": float(matched),
        "cross": float(other),
        "other": float(other),
        "contrast": float(matched - other),
        "similarity": float(matched),
        "orientation": orientation,
        "pairing": pairing,
        "geometry_status": "direct_maximum" if orientation == "direct" else "cross_maximum",
        "metric_status": "ok",
        "margin_mat": float(margin_mat),
        "margin_pat": float(margin_pat),
        "margin_ref1_mat": float(margin_mat),
        "margin_ref2_pat": float(margin_pat),
        "minmargin": float(min(margin_mat, margin_pat)),
        "matched_rho_candidate_A": float(matched_a),
        "matched_rho_candidate_B": float(matched_b),
        "copy_matched": {"A": float(matched_a), "B": float(matched_b)},
    })
    if abs(margin_mat) <= TIE_TOL or abs(margin_pat) <= TIE_TOL:
        row["margin_tie"] = True
        row["margin_class"] = "margin_tie"
    elif margin_mat > TIE_TOL and margin_pat > TIE_TOL:
        row["both_positive"] = True
        row["margin_class"] = "both_positive"
    elif margin_mat < -TIE_TOL and margin_pat < -TIE_TOL:
        row["both_negative"] = True
        row["margin_class"] = "both_negative"
    else:
        row["one_negative"] = True
        row["margin_class"] = "one_negative"
    return row


def _derive_single(row: dict[str, Any]) -> dict[str, Any]:
    row["raw_four_rho"] = {
        "rho_A_mat": float(row["rho_A_mat"]) if _finite(row.get("rho_A_mat")) else None,
        "rho_A_pat": float(row["rho_A_pat"]) if _finite(row.get("rho_A_pat")) else None,
        "rho_B_mat": None,
        "rho_B_pat": None,
    }
    row.update({
        "direct_original": None,
        "cross_original": None,
        "cross_matched": None,
        "matched_ref1": None,
        "matched_ref2": None,
        "fixed_reference_abcd": None,
        "margin_mat": None,
        "margin_pat": None,
        "margin_ref1_mat": None,
        "margin_ref2_pat": None,
        "minmargin": None,
        "matched_rho_candidate_A": None,
        "matched_rho_candidate_B": None,
        "copy_matched": None,
        "both_positive": False,
        "one_negative": False,
        "both_negative": False,
        "margin_tie": False,
        "geometry_tie": False,
        "margin_class": "not_applicable",
        "missing_count": int((not _finite(row.get("rho_A_mat"))) + (not _finite(row.get("rho_A_pat")))),
        "contrast": None,
        "cross": None,
        "other": None,
        "pairing": "reference_mean" if _finite(row.get("similarity")) else None,
        "geometry_status": "reference_mean" if _finite(row.get("similarity")) else "unavailable",
    })
    return row


def _failed_row(condition: Mapping[str, Any], chromosome: Mapping[str, Any]) -> dict[str, Any]:
    mask = chromosome.get("mask", {})
    n_copies = int(condition.get("n_copies", 2))
    tracks = list(r2comparison.default_track_names(int(chromosome["chromosome_index"]), n_copies))
    return {
        "chromosome": chromosome["chromosome"],
        "condition_id": condition["id"],
        "display_name": condition.get("display_name", condition["id"]),
        "role": condition.get("role", "new_variant"),
        "endpoint_status": condition.get("endpoint_status"),
        "accepted_as": condition.get("accepted_as"),
        "variant_id": condition.get("variant_id"),
        "bundle_id": condition.get("bundle_id"),
        "coordinate_status": str(condition.get("coordinate_status", "failed")),
        "n_copies": n_copies,
        "two_copy": n_copies == 2,
        "track_names": tracks,
        "n_bins": mask.get("n_bins"),
        "n_total_non_diagonal_pairs": mask.get("n_total_non_diagonal_pairs"),
        "n_common_pairs": mask.get("n_common_pairs"),
        "mask_status": mask.get("status"),
        "metric_status": "unavailable",
        "formal_metric_status": "unavailable",
        "descriptive_only": False,
        "reason": condition.get("failure_reason", "planned_endpoint_failed_before_R2"),
        "rho_reasons": {},
        "rho_A_mat": None,
        "rho_A_pat": None,
        "rho_B_mat": None,
        "rho_B_pat": None,
        "raw_four_rho": {key: None for key in ("rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat")},
        "direct_original": None,
        "cross_original": None,
        "cross_matched": None,
        "matched_ref1": None,
        "matched_ref2": None,
        "fixed_reference_abcd": None,
        "matched": None,
        "cross": None,
        "other": None,
        "contrast": None,
        "similarity": None,
        "pairing": None,
        "orientation": "failed",
        "geometry_status": "failed",
        "fixed_reference_abcd": None,
        "margin_mat": None,
        "margin_pat": None,
        "margin_ref1_mat": None,
        "margin_ref2_pat": None,
        "minmargin": None,
        "matched_rho_candidate_A": None,
        "matched_rho_candidate_B": None,
        "copy_matched": None,
        "both_positive": False,
        "one_negative": False,
        "both_negative": False,
        "margin_tie": False,
        "geometry_tie": False,
        "margin_class": "not_applicable",
        "missing_count": 4,
        "coverage": {},
        "failure_reason": condition.get("failure_reason"),
    }


def evaluate_in_memory(
    chromosomes: Sequence[Mapping[str, Any] | Sequence[Any]],
    conditions: Mapping[str, r2comparison.R2Condition] | Sequence[r2comparison.R2Condition],
    reference: Mapping[str, Mapping[int, np.ndarray]],
    *,
    failed_conditions: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """使用冻结的纯 R2 函数评估 synthetic/内存中的 conditions。"""
    failed_by_id: dict[str, Mapping[str, Any]]
    if isinstance(failed_conditions, Mapping):
        failed_by_id = {str(key): value for key, value in failed_conditions.items()}
    else:
        failed_by_id = {str(item["id"]): item for item in failed_conditions}
    if isinstance(conditions, Mapping):
        valid = {key: value for key, value in conditions.items() if key not in failed_by_id}
    else:
        valid = [item for item in conditions if item.condition_id not in failed_by_id]
    if not valid:
        raise AlleleR2Error("at least one finite condition is required for a common mask")
    evaluated = r2comparison.evaluate_genome(chromosomes, valid, reference)
    for chromosome in evaluated:
        for condition_id, row in list(chromosome.get("conditions", {}).items()):
            row = dict(row)
            if int(row.get("n_copies", 2)) == 2:
                row = _derive_two_copy(row)
            else:
                row = _derive_single(row)
            if _is_formal_endpoint_failure(row):
                row["descriptive_only"] = bool(row.get("metric_status") == "ok")
                row["formal_metric_status"] = ("failure_only_descriptive"
                                                if row["descriptive_only"] else "unavailable")
                if row["descriptive_only"]:
                    row["reason"] = "solver_failed_failure_only_descriptive"
            else:
                row["descriptive_only"] = False
                row["formal_metric_status"] = row.get("metric_status")
            chromosome["conditions"][condition_id] = row
        chromosome["mask"]["included_condition_ids"] = list(valid.keys()) if isinstance(valid, Mapping) else [item.condition_id for item in valid]
        chromosome["mask"]["excluded_failed_condition_ids"] = sorted(failed_by_id)
        chromosome["mask"]["excluded_conditions"] = {
            condition_id: {
                "coordinate_status": str(condition.get("coordinate_status", "failed")),
                "reason": condition.get("failure_reason", "planned_endpoint_failed_before_R2"),
            }
            for condition_id, condition in sorted(failed_by_id.items())
        }
        for condition_id, condition in failed_by_id.items():
            chromosome["conditions"][condition_id] = _failed_row(condition, chromosome)
    return evaluated


def make_bootstrap_index_matrix(n_chromosomes: int = 20, *, seed: int = BOOTSTRAP_SEED,
                                n_boot: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    if n_chromosomes <= 0 or n_boot <= 0:
        raise AlleleR2Error("bootstrap dimensions must be positive")
    if seed != BOOTSTRAP_SEED or n_boot != BOOTSTRAP_DRAWS:
        raise AlleleR2Error("release bootstrap must use seed9301 and n_boot10000")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_chromosomes, size=(n_boot, n_chromosomes), dtype=np.int64)


def bootstrap_index_sha256(indices: np.ndarray) -> str:
    """对 canonical 连续 int64 索引数组计算 hash，而非对 artifact file 计算 hash。"""
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def bootstrap_file_sha256(path: str | Path) -> str:
    """对精确的 .npy artifact bytes 计算 hash，供 ``sha256sum -c`` 证据使用。"""
    return sha256_file(Path(path))


def _validate_bootstrap_indices(indices: np.ndarray, n_chromosomes: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if values.shape[1:] != (n_chromosomes,) or values.ndim != 2:
        raise AlleleR2Error("shared bootstrap matrix shape must be (n_boot,n_chromosomes)")
    if values.size and (values.min() < 0 or values.max() >= n_chromosomes):
        raise AlleleR2Error("bootstrap matrix contains an out-of-range chromosome index")
    return values


def _bootstrap_values(values: Sequence[float], indices: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    matrix = _validate_bootstrap_indices(indices, len(vector))
    draws = vector[matrix].mean(axis=1)
    return {
        "n_chromosomes": int(len(vector)),
        "mean": float(vector.mean()),
        "median": float(np.median(vector)),
        "ci95": [float(item) for item in np.percentile(draws, [2.5, 97.5])],
        "seed": BOOTSTRAP_SEED,
        "n_boot": int(len(draws)),
        "matrix_array_sha256": bootstrap_index_sha256(matrix),
        # 向后兼容的 alias；明确命名内存中的 object。
        "index_matrix_array_sha256": bootstrap_index_sha256(matrix),
        "unit": "chromosome",
        "interpretation": "technical/structural chromosome bootstrap within one cell; not biological replication or a p-value",
    }


def _row_lookup(evaluated: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(item["chromosome"]): item for item in evaluated}


def _delta_value(row: Mapping[str, Any], left: str, right: str, metric: str) -> float | None:
    conditions = row.get("conditions", {})
    left_row = conditions.get(left, {})
    right_row = conditions.get(right, {})
    left_value = left_row.get(metric)
    right_value = right_row.get(metric)
    if not _finite(left_value) or not _finite(right_value):
        return None
    delta = float(left_value) - float(right_value)
    return delta if math.isfinite(delta) else None


def summarise_paired_seed_effects(
    per_seed_evaluated: Mapping[str, Sequence[Mapping[str, Any]]],
    left_condition_by_seed: Mapping[str, str],
    right_condition_by_seed: Mapping[str, str],
    *,
    metric: str = "similarity",
    planned_seed_ids: Sequence[str] = BUNDLE_IDS,
    bootstrap_indices: np.ndarray,
    label: str | None = None,
) -> dict[str, Any]:
    """先按 seed 配对，再求每条染色体的三个 seed delta 均值。

    planned seed 中缺失或失败的 seed 会使主结果明确为 n/a。完整的 per-seed 记录仍在返回值中；``available_only`` 单独标记，绝不替代主结果。
    """
    seed_ids = [str(item) for item in planned_seed_ids]
    if len(set(seed_ids)) != len(seed_ids):
        raise AlleleR2Error("planned seed ids must be unique")
    first = next((per_seed_evaluated[item] for item in seed_ids if item in per_seed_evaluated), None)
    if first is None:
        raise AlleleR2Error("no planned seed result supplied")
    chromosome_names = [str(item["chromosome"]) for item in first]
    if len(set(chromosome_names)) != len(chromosome_names) or not chromosome_names:
        raise AlleleR2Error("seed result must contain unique chromosomes")
    index_matrix = _validate_bootstrap_indices(bootstrap_indices, len(chromosome_names))
    per_seed = []
    delta_by_seed: dict[str, dict[str, float | None]] = {}
    for seed_id in seed_ids:
        rows = per_seed_evaluated.get(seed_id)
        left_id = left_condition_by_seed.get(seed_id)
        right_id = right_condition_by_seed.get(seed_id)
        descriptive_deltas = {chromosome: None for chromosome in chromosome_names}
        if rows is None or left_id is None or right_id is None:
            deltas = {chromosome: None for chromosome in chromosome_names}
            reason = "fit_missing"
        else:
            lookup = _row_lookup(rows)
            deltas = {}
            missing_reasons = []
            for chromosome in chromosome_names:
                if chromosome not in lookup:
                    deltas[chromosome] = None
                    missing_reasons.append("fit_missing")
                    continue
                row = lookup[chromosome]
                left_row = row.get("conditions", {}).get(left_id, {})
                right_row = row.get("conditions", {}).get(right_id, {})
                value = _delta_value(row, left_id, right_id, metric)
                descriptive_deltas[chromosome] = value
                left_failed = _is_formal_endpoint_failure(left_row)
                right_failed = _is_formal_endpoint_failure(right_row)
                if left_failed or right_failed:
                    # 保留有限的 residual-coordinate delta 供 diagnostics 使用，
                    # 但绝不让 solver-failed arm 进入正式 primary R2。
                    deltas[chromosome] = None
                    missing_reasons.append("solver_failed")
                    continue
                if not _metric_is_formally_usable(left_row, metric) or not _metric_is_formally_usable(right_row, metric):
                    deltas[chromosome] = None
                    tie_missing = ((metric in ("margin_mat", "margin_pat", "minmargin")) and
                                   (left_row.get("geometry_tie") or right_row.get("geometry_tie")))
                    missing_reasons.append("metric_tie_missing" if tie_missing else "metric_missing")
                    continue
                deltas[chromosome] = value
                if value is None:
                    missing_reasons.append("metric_missing")
            if missing_reasons:
                if "fit_missing" in missing_reasons:
                    reason = "fit_missing"
                elif "solver_failed" in missing_reasons:
                    reason = "solver_failed"
                elif "metric_tie_missing" in missing_reasons:
                    reason = "metric_tie_missing"
                else:
                    reason = "metric_missing"
            else:
                reason = None
        valid_count = sum(value is not None for value in deltas.values())
        valid = valid_count == len(chromosome_names)
        seed_record = {
            "seed_id": seed_id,
            "left_condition": left_id,
            "right_condition": right_id,
            "metric": metric,
            "valid": valid,
            "n_chromosomes_expected": len(chromosome_names),
            "n_chromosomes_valid": valid_count,
            "reason": reason,
            "per_chromosome_delta": deltas,
            "descriptive_per_chromosome_delta": descriptive_deltas,
            "descriptive_only": bool(reason == "solver_failed" and any(value is not None for value in descriptive_deltas.values())),
            "mean_delta": None,
            "median_delta": None,
            "ci95": None,
            "wins": None,
            "ties": None,
            "losses": None,
            "bootstrap": None,
        }
        if valid:
            seed_values = [float(deltas[chromosome]) for chromosome in chromosome_names]
            seed_bootstrap = _bootstrap_values(seed_values, index_matrix)
            seed_wins = sum(value > TIE_TOL for value in seed_values)
            seed_losses = sum(value < -TIE_TOL for value in seed_values)
            seed_record.update({
                "mean_delta": seed_bootstrap["mean"],
                "median_delta": seed_bootstrap["median"],
                "ci95": seed_bootstrap["ci95"],
                "wins": int(seed_wins),
                "ties": int(len(seed_values) - seed_wins - seed_losses),
                "losses": int(seed_losses),
                "bootstrap": seed_bootstrap,
            })
        delta_by_seed[seed_id] = deltas
        per_seed.append(seed_record)
    valid_seed_ids = [item["seed_id"] for item in per_seed if item["valid"]]
    per_chromosome_available = []
    for chromosome in chromosome_names:
        values = [delta_by_seed[seed_id][chromosome] for seed_id in valid_seed_ids]
        finite = [float(value) for value in values if value is not None]
        per_chromosome_available.append({
            "chromosome": chromosome,
            "n_valid_seeds": len(finite),
            "delta_by_seed": {seed_id: delta_by_seed[seed_id][chromosome] for seed_id in seed_ids},
            "descriptive_delta_by_seed": {seed_id: per_seed[seed_ids.index(seed_id)]["descriptive_per_chromosome_delta"][chromosome]
                                          for seed_id in seed_ids},
            "mean_delta": float(np.mean(finite)) if finite else None,
        })
    available_values = [item["mean_delta"] for item in per_chromosome_available
                        if item["mean_delta"] is not None]
    available_summary = None
    if available_values:
        available_bootstrap = _bootstrap_values(available_values, index_matrix[:, :len(available_values)]) \
            if len(available_values) == len(chromosome_names) else None
        available_summary = {
            "status": "available_only",
            "seed_ids": valid_seed_ids,
            "planned_seed_count": len(seed_ids),
            "valid_seed_count": len(valid_seed_ids),
            "per_chromosome": per_chromosome_available,
            "mean_delta": float(np.mean(available_values)),
            "median_delta": float(np.median(available_values)),
            "ci95": available_bootstrap["ci95"] if available_bootstrap else None,
            "bootstrap": available_bootstrap,
        }
    primary_ok = len(valid_seed_ids) == len(seed_ids)
    primary_per_chromosome = None
    primary_summary = {
        "status": "n/a_planned_seed_failure" if not primary_ok else "ok",
        "planned_seed_count": len(seed_ids),
        "valid_seed_count": len(valid_seed_ids),
        "seed_ids": seed_ids,
        "valid_seed_ids": valid_seed_ids,
        "per_chromosome": None,
        "mean_delta": None,
        "median_delta": None,
        "ci95": None,
        "wins": None,
        "ties": None,
        "losses": None,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "n_boot": int(index_matrix.shape[0]),
            "matrix_array_sha256": bootstrap_index_sha256(index_matrix),
            "index_matrix_array_sha256": bootstrap_index_sha256(index_matrix),
            "shared": True,
        },
    }
    if primary_ok:
        primary_per_chromosome = []
        values = []
        for chromosome in chromosome_names:
            by_seed = {seed_id: delta_by_seed[seed_id][chromosome] for seed_id in seed_ids}
            mean_delta = float(np.mean([by_seed[seed_id] for seed_id in seed_ids]))
            primary_per_chromosome.append({"chromosome": chromosome, "delta_by_seed": by_seed,
                                          "mean_delta": mean_delta})
            values.append(mean_delta)
        boot = _bootstrap_values(values, index_matrix)
        wins = sum(value > TIE_TOL for value in values)
        losses = sum(value < -TIE_TOL for value in values)
        primary_summary.update({
            "per_chromosome": primary_per_chromosome,
            "mean_delta": boot["mean"],
            "median_delta": boot["median"],
            "ci95": boot["ci95"],
            "wins": int(wins),
            "ties": int(len(values) - wins - losses),
            "losses": int(losses),
            "bootstrap": boot,
        })
    return {
        "label": label or "%s_minus_%s" % (left_condition_by_seed.get(seed_ids[0]), right_condition_by_seed.get(seed_ids[0])),
        "metric": metric,
        "direction": "left_minus_right",
        "primary": primary_summary,
        "available_only": available_summary,
        "per_seed": per_seed,
        "n_chromosomes_expected": len(chromosome_names),
        "note": "Primary pairs each planned seed first, then averages all planned seeds per chromosome; failed planned seeds force primary n/a.",
    }


def condition_summary(evaluated: Sequence[Mapping[str, Any]], condition_ids: Sequence[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    expected = [str(item["chromosome"]) for item in evaluated]
    for condition_id in condition_ids:
        rows = [item.get("conditions", {}).get(condition_id, {}) for item in evaluated]
        formal_rows = [row for row in rows if not row.get("descriptive_only") and not _is_formal_endpoint_failure(row)]
        similarity = [row.get("similarity") for row in formal_rows if _finite(row.get("similarity"))]
        contrast = [row.get("contrast") for row in formal_rows if _finite(row.get("contrast"))]
        descriptive_similarity = [row.get("similarity") for row in rows
                                  if row.get("descriptive_only") and _finite(row.get("similarity"))]
        descriptive_contrast = [row.get("contrast") for row in rows
                                if row.get("descriptive_only") and _finite(row.get("contrast"))]
        metric_summaries = {}
        for metric in PRIMARY_PAIR_METRICS:
            metric_values = [row.get(metric) for row in formal_rows if _metric_is_formally_usable(row, metric)]
            metric_summaries[metric] = {
                "macro_mean": float(np.mean(metric_values)) if metric_values else None,
                "median": float(np.median(metric_values)) if metric_values else None,
                "n_finite": len(metric_values),
                "missing_chromosomes": [chromosome for chromosome, row in zip(expected, rows)
                                        if not _metric_is_formally_usable(row, metric)],
            }
        def _count(key: str) -> int:
            return int(sum(bool(row.get(key)) for row in rows))
        output[condition_id] = {
            "condition_id": condition_id,
            "n_chromosomes_expected": len(expected),
            "similarity_macro_mean": float(np.mean(similarity)) if similarity else None,
            "similarity_median": float(np.median(similarity)) if similarity else None,
            "similarity_n_finite": len(similarity),
            "similarity_descriptive_only_n_finite": len(descriptive_similarity),
            "similarity_missing_chromosomes": [chromosome for chromosome, row in zip(expected, rows)
                                               if not _metric_is_formally_usable(row, "similarity")],
            "contrast_macro_mean": float(np.mean(contrast)) if contrast else None,
            "contrast_median": float(np.median(contrast)) if contrast else None,
            "contrast_n_finite": len(contrast),
            "contrast_descriptive_only_n_finite": len(descriptive_contrast),
            "contrast_missing_chromosomes": [chromosome for chromosome, row in zip(expected, rows)
                                             if not _metric_is_formally_usable(row, "contrast")],
            "contrast_is_applicable": bool(contrast),
            "metric_summary": metric_summaries,
            "orientation_counts": {
                orientation: int(sum(row.get("orientation") == orientation for row in rows))
                for orientation in ("direct", "swapped", "unresolved_tie", "unresolved_missing", "failed")
            },
            "both_positive_count": _count("both_positive"),
            "one_negative_count": _count("one_negative"),
            "both_negative_count": _count("both_negative"),
            "margin_tie_count": _count("margin_tie"),
            "geometry_tie_count": _count("geometry_tie"),
            "failed_count": int(sum(row.get("coordinate_status") == "failed" or _is_formal_endpoint_failure(row)
                                    for row in rows)),
        }
    return output


R2_COLUMNS = (
    "chromosome", "condition_id", "variant_id", "bundle_id", "display_name", "role",
    "endpoint_status", "accepted_as", "coordinate_status", "n_copies", "two_copy",
    "track_names", "n_bins", "n_total_non_diagonal_pairs", "n_common_pairs", "mask_status",
    "metric_status", "formal_metric_status", "descriptive_only", "reason", "rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat",
    "direct_original", "cross_original", "cross_matched", "matched_ref1", "matched_ref2",
    "matched", "cross", "other", "contrast", "similarity",
    "margin_mat", "margin_pat", "minmargin", "matched_rho_candidate_A", "matched_rho_candidate_B",
    "both_positive", "one_negative", "both_negative", "margin_tie", "geometry_tie", "margin_class",
    "pairing", "orientation", "geometry_status", "coverage_json", "rho_reasons_json",
    "fixed_reference_abcd_json", "copy_matched_json", "failure_reason",
)


def flatten_rows(evaluated: Sequence[Mapping[str, Any]], condition_ids: Sequence[str],
                 condition_meta: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for chromosome in evaluated:
        for condition_id in condition_ids:
            raw = dict(chromosome.get("conditions", {}).get(condition_id, {}))
            meta = condition_meta.get(condition_id, {})
            row = {column: raw.get(column) for column in R2_COLUMNS}
            row["variant_id"] = meta.get("variant_id", raw.get("variant_id"))
            row["bundle_id"] = meta.get("bundle_id", raw.get("bundle_id"))
            row["coordinate_status"] = meta.get("coordinate_status", raw.get("coordinate_status", "available"))
            row["track_names"] = json.dumps(_jsonable(raw.get("track_names", [])), separators=(",", ":"))
            row["coverage_json"] = json.dumps(_jsonable(raw.get("coverage", {})), sort_keys=True, separators=(",", ":"))
            row["rho_reasons_json"] = json.dumps(_jsonable(raw.get("rho_reasons", {})), sort_keys=True, separators=(",", ":"))
            row["fixed_reference_abcd_json"] = json.dumps(_jsonable(raw.get("fixed_reference_abcd")), separators=(",", ":"))
            row["copy_matched_json"] = json.dumps(_jsonable(raw.get("copy_matched")), sort_keys=True, separators=(",", ":"))
            rows.append(row)
    return rows


def write_table(path: Path, rows: Sequence[Mapping[str, Any]], *, delimiter: str) -> dict[str, Any]:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(R2_COLUMNS), delimiter=delimiter,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row.get(column) is None else _jsonable(row.get(column))
                             for column in R2_COLUMNS})
    return {"path": str(path), "rows": len(rows), "columns": list(R2_COLUMNS), "delimiter": delimiter}


PAIRED_COLUMNS = (
    "label", "left_variant", "right_variant", "metric", "level", "seed_id", "status",
    "planned_seed_count", "valid_seed_count", "n_chromosomes_expected", "n_chromosomes_valid",
    "mean_delta", "median_delta", "ci95_low", "ci95_high", "wins", "ties", "losses",
    "reason", "descriptive_only", "bootstrap_matrix_array_sha256",
)


def flatten_paired_rows(comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in comparisons:
        label = str(item.get("label", ""))
        left_variant = item.get("left_variant")
        right_variant = item.get("right_variant")
        metric = item.get("metric")
        primary = item.get("primary", {})
        bootstrap = primary.get("bootstrap", {}) or {}
        ci95 = primary.get("ci95") or [None, None]
        rows.append({
            "label": label, "left_variant": left_variant, "right_variant": right_variant,
            "metric": metric, "level": "primary", "seed_id": None,
            "status": primary.get("status"), "planned_seed_count": primary.get("planned_seed_count"),
            "valid_seed_count": primary.get("valid_seed_count"),
            "n_chromosomes_expected": item.get("n_chromosomes_expected"),
            "n_chromosomes_valid": item.get("n_chromosomes_expected") if primary.get("status") == "ok" else None,
            "mean_delta": primary.get("mean_delta"), "median_delta": primary.get("median_delta"),
            "ci95_low": ci95[0], "ci95_high": ci95[1], "wins": primary.get("wins"),
            "ties": primary.get("ties"), "losses": primary.get("losses"),
            "reason": None if primary.get("status") == "ok" else "planned_seed_failure",
            "descriptive_only": False,
            "bootstrap_matrix_array_sha256": bootstrap.get("matrix_array_sha256"),
        })
        for seed in item.get("per_seed", []):
            seed_bootstrap = seed.get("bootstrap", {}) or {}
            seed_ci = seed.get("ci95") or [None, None]
            rows.append({
                "label": label, "left_variant": left_variant, "right_variant": right_variant,
                "metric": metric, "level": "seed", "seed_id": seed.get("seed_id"),
                "status": "ok" if seed.get("valid") else "n/a",
                "planned_seed_count": len(item.get("per_seed", [])),
                "valid_seed_count": int(seed.get("valid", False)),
                "n_chromosomes_expected": seed.get("n_chromosomes_expected"),
                "n_chromosomes_valid": seed.get("n_chromosomes_valid"),
                "mean_delta": seed.get("mean_delta"), "median_delta": seed.get("median_delta"),
                "ci95_low": seed_ci[0], "ci95_high": seed_ci[1], "wins": seed.get("wins"),
                "ties": seed.get("ties"), "losses": seed.get("losses"),
                "reason": seed.get("reason"), "descriptive_only": seed.get("descriptive_only", False),
                "bootstrap_matrix_array_sha256": seed_bootstrap.get("matrix_array_sha256"),
            })
    return rows


def write_paired_table(path: Path, rows: Sequence[Mapping[str, Any]], *, delimiter: str = "\t") -> dict[str, Any]:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PAIRED_COLUMNS), delimiter=delimiter,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row.get(column) is None else _jsonable(row.get(column))
                             for column in PAIRED_COLUMNS})
    return {"path": str(path), "rows": len(rows), "columns": list(PAIRED_COLUMNS), "delimiter": delimiter}


def _short_condition_ids(condition_meta: Mapping[str, Mapping[str, Any]], *, role: str) -> list[str]:
    return [condition_id for condition_id in ALL_CONDITION_IDS
            if condition_meta.get(condition_id, {}).get("role") == role]


def _short_na_reason(primary: Mapping[str, Any]) -> str:
    reason = primary.get("status") or "invalid"
    if reason.startswith("n/a_"):
        reason = reason[4:]
    return str(reason).replace("_", " ")


def _render_paired_effects(plot_dir: Path, comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    plot_dir.mkdir(parents=True, exist_ok=True)
    output = {}
    for metric in PRIMARY_PAIR_METRICS:
        metric_comparisons = [item for item in comparisons if item.get("metric") == metric]
        labels = [str(item.get("label", "comparison")).split(":", 1)[0]
                  for item in metric_comparisons]
        values = []
        valid_labels = []
        na_reasons = []
        for item, label in zip(metric_comparisons, labels):
            primary = item.get("primary", {})
            per_chr = primary.get("per_chromosome")
            if primary.get("status") == "ok" and per_chr:
                values.append([float(row["mean_delta"]) for row in per_chr])
                valid_labels.append(label)
                na_reasons.append(None)
            else:
                # 保留数值占位符，使失败比较保留其 x slot，
                # 下面可标记为 n/a。
                values.append([np.nan])
                na_reasons.append(_short_na_reason(primary))
        path = plot_dir / ("r2_paired_effects_%s.png" % metric)
        fig, ax = plt.subplots(figsize=(6, 3))
        if metric_comparisons:
            try:
                ax.boxplot(values, tick_labels=labels, showfliers=False)
            except TypeError:  # matplotlib <3.9
                ax.boxplot(values, labels=labels, showfliers=False)
            ax.axhline(0.0, color="#555555", lw=0.5, ls="--")
            for index, reason in enumerate(na_reasons, start=1):
                if reason is not None:
                    ax.text(index, 0.98, "n/a",
                            ha="center", va="top", transform=ax.get_xaxis_transform(),
                            fontsize=7, color="#666666")
        else:
            ax.text(0.5, 0.5, "no registered primary comparison", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
        ax.set_title("paired chromosome effects | %s" % metric, fontsize=7)
        ax.set_ylabel("left minus right", fontsize=7)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        output[metric] = {
            "path": str(path), "labels": labels, "comparisons_with_values": valid_labels,
            "n_comparisons": len(metric_comparisons),
            "n_valid": len(valid_labels), "n_na": len(metric_comparisons) - len(valid_labels),
            "na_reasons": {label: reason for label, reason in zip(labels, na_reasons) if reason is not None},
            "dpi": 300, "figsize_inches": [6, 3], "text_size_pt": 7,
        }
    return output


def _render_seed_stability(plot_dir: Path, comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    plot_dir.mkdir(parents=True, exist_ok=True)
    output = {}
    for metric in PRIMARY_PAIR_METRICS:
        metric_comparisons = [item for item in comparisons if item.get("metric") == metric]
        path = plot_dir / ("r2_seed_stability_%s.png" % metric)
        fig, ax = plt.subplots(figsize=(6, 3))
        plotted = 0
        valid_seed_counts = {}
        for index, item in enumerate(metric_comparisons, start=1):
            values = [float(seed["mean_delta"]) for seed in item.get("per_seed", [])
                      if _finite(seed.get("mean_delta"))]
            planned = len(item.get("per_seed", []))
            valid_seed_counts[str(item.get("label", "comparison")).split(":", 1)[0]] = {
                "valid": len(values), "planned": planned,
            }
            if values:
                plotted += len(values)
                offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0.0]
                ax.plot([index + float(offset) for offset in offsets], values,
                        "o", ms=3, color="#2878b5")
            if planned:
                ax.text(index, 0.98, "%d/%d" % (len(values), planned),
                        ha="center", va="top", transform=ax.get_xaxis_transform(),
                        fontsize=7, color="#666666")
        ax.axhline(0.0, color="#555555", lw=0.5, ls="--")
        ax.set_title("paired seed stability | %s" % metric, fontsize=7)
        ax.set_ylabel("per-seed mean delta", fontsize=7)
        ax.set_xticks(range(1, len(metric_comparisons) + 1),
                      [str(item.get("label", "comparison")).split(":", 1)[0]
                       for item in metric_comparisons], rotation=25, ha="right")
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        output[metric] = {"path": str(path), "n_points": plotted,
                          "n_comparisons": len(metric_comparisons),
                          "valid_seed_counts": valid_seed_counts,
                          "dpi": 300, "figsize_inches": [6, 3], "text_size_pt": 7}
    return output


def _render_outputs(output_dir: Path, evaluated: Sequence[Mapping[str, Any]],
                    condition_ids: Sequence[str], condition_meta: Mapping[str, Mapping[str, Any]],
                    summaries: Mapping[str, Any], paired: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """复用已有的紧凑渲染器，按小型条件组使用。"""
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {}
    new_representatives = [item["condition_id"] for item in summaries["representatives"].values()
                           if item.get("condition_id") is not None]
    controls = [condition_id for condition_id in condition_ids if condition_id in HISTORICAL_IDS]
    for key, ids, title in (("new_representatives", new_representatives, "P9016 allele ablation R2 | count-selected representatives"),
                            ("historical_controls", controls, "P9016 R2 | historical controls")):
        if ids:
            paths[key] = r2comparison.render_figure_pair(
                plots / ("r2_" + key), evaluated, ids, title=title,
                similarity_title="Matched Spearman rho", contrast_title="Matched minus cross rho",
            )
            paths[key + "_cross"] = r2comparison.render_boxplot_figure(
                plots / ("r2_" + key + "_cross.png"), evaluated, ids, metric="cross",
                title=title + " | cross", y_label="Cross Spearman rho",
            )
            paths[key + "_margin_mat"] = r2comparison.render_boxplot_figure(
                plots / ("r2_" + key + "_margin_mat.png"), evaluated, ids, metric="margin_mat",
                title=title + " | mat margin", y_label="margin to fixed mat",
            )
            paths[key + "_margin_pat"] = r2comparison.render_boxplot_figure(
                plots / ("r2_" + key + "_margin_pat.png"), evaluated, ids, metric="margin_pat",
                title=title + " | pat margin", y_label="margin to fixed pat",
            )
    paths["paired_effects"] = _render_paired_effects(plots / "paired_effects", paired)
    paths["seed_stability"] = _render_seed_stability(plots / "seed_stability", paired)
    return paths


def _build_primary_pairs(evaluated_by_seed: Mapping[str, Sequence[Mapping[str, Any]]],
                         bootstrap_indices: np.ndarray) -> list[dict[str, Any]]:
    pairs = []
    for left_variant, right_variant in PRIMARY_VARIANT_COMPARISONS:
        left_map = {bundle: "%s-%s" % (left_variant, bundle) for bundle in BUNDLE_IDS}
        right_map = {bundle: "%s-%s" % (right_variant, bundle) for bundle in BUNDLE_IDS}
        for metric in PRIMARY_PAIR_METRICS:
            result = summarise_paired_seed_effects(
                evaluated_by_seed, left_map, right_map, metric=metric,
                planned_seed_ids=BUNDLE_IDS, bootstrap_indices=bootstrap_indices,
                label="%s-%s:%s" % (left_variant, right_variant, metric),
            )
            result["left_variant"] = left_variant
            result["right_variant"] = right_variant
            pairs.append(result)
    return pairs


def evaluate_released(config_path: str | Path, release_manifest_path: str | Path,
                      output_dir: str | Path) -> dict[str, Any]:
    """评估 parent 已发布的 21-condition lock；参考结构访问受 gate 控制。"""
    config_file = Path(config_path).resolve()
    release_file = Path(release_manifest_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        if any(output.iterdir()):
            raise AlleleR2Error("evaluation output must be fresh and empty: %s" % output)
        raise AlleleR2Error("evaluation output directory already exists: %s" % output)
    config = _read_json(config_file)
    _validate_config(config, config_file)
    locked = _validate_release_manifest(config, config_file, release_file)
    output.mkdir(parents=True, exist_ok=False)
    loaded, gate, gate_provenance, reference = _lock_candidate_files(locked, output)
    condition_meta = locked["condition_by_id"]
    finite_conditions = {}
    failed_conditions = {}
    for item in locked["conditions"]:
        if _release_mask_eligible(item):
            meta = dict(item)
            meta["structures"] = loaded[item["id"]]
            finite_conditions[item["id"]] = r2comparison.R2Condition(
                condition_id=item["id"], display_name=str(item.get("display_name", item["id"])),
                structures=loaded[item["id"]], n_copies=int(item["n_copies"]),
                role=str(item.get("role", "main")), endpoint_status=str(item["endpoint_status"]),
                accepted_as=item.get("accepted_as"),
            )
        else:
            failed_item = dict(item)
            if item.get("coordinate_status") == "available" and item.get("id") not in HISTORICAL_IDS:
                failed_item["failure_reason"] = "solver_failed_failure_only_descriptive_excluded_from_primary_mask"
            elif not failed_item.get("failure_reason"):
                failed_item["failure_reason"] = "missing_locked_coordinate_excluded_from_primary_mask"
            failed_conditions[item["id"]] = failed_item
    chromosomes = locked["grid"]
    evaluated = evaluate_in_memory(chromosomes, finite_conditions, reference,
                                   failed_conditions=failed_conditions)
    condition_ids = list(ALL_CONDITION_IDS)
    bootstrap_indices = make_bootstrap_index_matrix(len(chromosomes))
    bootstrap_path = output / "bootstrap_index_matrix_seed9301.npy"
    np.save(bootstrap_path, bootstrap_indices)
    bootstrap_file_digest = bootstrap_file_sha256(bootstrap_path)
    roundtrip_indices = np.load(bootstrap_path, allow_pickle=False)
    if not np.array_equal(roundtrip_indices, bootstrap_indices):
        raise AlleleR2Error("saved shared bootstrap matrix failed byte-level roundtrip")
    # 每个 condition 的相同行用于所有 bundle-paired comparisons。
    # 已发布 run 只有一个共同的 geometry evaluation；bundle pairing 选择对应行，绝不重新选择代表项。
    evaluated_by_seed = {bundle: evaluated for bundle in BUNDLE_IDS}
    paired = _build_primary_pairs(evaluated_by_seed, bootstrap_indices)
    summaries = {
        "condition_summary": condition_summary(evaluated, condition_ids),
        "representatives": locked["representatives"],
        "paired_comparisons": paired,
    }
    rows = flatten_rows(evaluated, condition_ids, condition_meta)
    paired_rows = flatten_paired_rows(paired)
    paired_table_tsv = write_paired_table(output / "r2_paired_effects.tsv", paired_rows)
    paired_table_json_path = output / "r2_paired_effects.json"
    _write_json(paired_table_json_path, {"rows": paired_rows, "comparison_count": len(paired),
                                         "metric_count": len(PRIMARY_PAIR_METRICS),
                                         "shared_bootstrap_matrix_array_sha256": bootstrap_index_sha256(bootstrap_indices)})
    summary_payload = {
        "schema_version": EVALUATION_SCHEMA,
        "metric_scope": "R2 only",
        "status": "evaluation_complete",
        "condition_summary": summaries["condition_summary"],
        "representatives": summaries["representatives"],
        "paired_comparisons": paired,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "n_boot": BOOTSTRAP_DRAWS,
                      "matrix_array_sha256": bootstrap_index_sha256(bootstrap_indices),
                      "file_sha256": bootstrap_file_digest,
                      "path": str(bootstrap_path),
                      "shared_across_all_comparisons": True,
                      "unit": "chromosome",
                      "interpretation": "technical/structural within one cell; not biological replication or p-value"},
    }
    _write_json(output / "r2_summary.json", summary_payload)
    _write_json(output / "r2_results.json", {
        "schema_version": EVALUATION_SCHEMA,
        "status": "evaluation_complete",
        "method": {
            "reused_pure_functions": ["pr.r2comparison.R2Condition", "build_common_mask", "evaluate_genome"],
            "grid": "range(3Mb, chromosome_length, 1Mb)",
            "mask": "one finite unordered off-diagonal mask per chromosome across all available finite conditions and both fixed reference tracks",
            "minimum_common_pairs": MIN_RHO_PAIRS,
            "primary_pair_metrics": list(PRIMARY_PAIR_METRICS),
            "primary_pair_count": len(PRIMARY_VARIANT_COMPARISONS) * len(PRIMARY_PAIR_METRICS),
            "orientation": "fixed reference columns mat/pat; candidate rows only may swap once per chromosome; tie <=1e-12 unresolved",
            "consensus": "single-copy similarity is mean(rho_A_mat,rho_A_pat), both finite required; contrast n/a",
        },
        "release_manifest": locked["manifest"],
        "gate": {"stage": EVAL_STAGE, "entries": gate.entries, "armed": sorted(gate.armed)},
        "gate_provenance": gate_provenance,
        "conditions": condition_ids,
        "chromosomes": evaluated,
        "summary": summary_payload,
        "phase_payload_opened": False,
        "reference_loaded_after_all_coordinate_hashes": True,
        "selection_recomputed": False,
        "real_fit_started_by_evaluator": False,
        "bootstrap_index_matrix_path": str(bootstrap_path),
        "bootstrap_index_matrix_array_sha256": bootstrap_index_sha256(bootstrap_indices),
        "bootstrap_index_matrix_file_sha256": bootstrap_file_digest,
    })
    _write_json(output / "r2_per_chromosome.json", {"rows": rows})
    table_tsv = write_table(output / "r2_per_chromosome.tsv", rows, delimiter="\t")
    table_csv = write_table(output / "r2_per_chromosome.csv", rows, delimiter=",")
    plot_metadata = _render_outputs(output, evaluated, condition_ids, condition_meta, summaries, paired)
    evaluation_manifest = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "evaluation_complete",
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "release_manifest_path": str(release_file),
        "release_manifest_sha256": sha256_file(release_file),
        "protocol_sha256": AUTHORITATIVE_PROTOCOL_SHA256,
        "reference": gate_provenance["reference"],
        "coordinates": gate_provenance["coordinates"],
        "failed_conditions": [item["id"] for item in locked["conditions"] if item["coordinate_status"] == "failed"],
        "condition_count": len(condition_ids),
        "primary_comparison_count": len(paired),
        "primary_comparison_metrics": list(PRIMARY_PAIR_METRICS),
        "row_count": len(rows),
        "mask_metadata": [item["mask"] for item in evaluated],
        "bootstrap": {"path": str(bootstrap_path), "sha256": bootstrap_file_digest,
                      "matrix_array_sha256": bootstrap_index_sha256(bootstrap_indices),
                      "seed": BOOTSTRAP_SEED, "n_boot": BOOTSTRAP_DRAWS},
        "selection_recomputed": False,
        "phase_payload_opened": False,
        "r1_r3_added": False,
        "code": {
            "allele_r2": str(Path(__file__).resolve()),
            "allele_r2_sha256": sha256_file(Path(__file__).resolve()),
            "r2comparison": str(ROOT / "pr/r2comparison.py"),
            "r2comparison_sha256": sha256_file(ROOT / "pr/r2comparison.py"),
        },
        "outputs": {"summary": str(output / "r2_summary.json"), "results": str(output / "r2_results.json"),
                    "rows_json": str(output / "r2_per_chromosome.json"),
                    "paired_tsv": paired_table_tsv,
                    "paired_json": str(paired_table_json_path),
                    "tsv": table_tsv, "csv": table_csv, "plots": plot_metadata},
    }
    _write_json(output / "evaluation_manifest.json", evaluation_manifest)
    _write_json(output / "config.json", config)
    readme = _evaluation_readme(summary_payload, evaluation_manifest)
    (output / "README.md").write_text(readme, encoding="utf-8")
    return {"output_dir": str(output), "condition_count": len(condition_ids),
            "row_count": len(rows), "failed_conditions": evaluation_manifest["failed_conditions"],
            "manifest": evaluation_manifest}


def _evaluation_readme(summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    lines = [
        "# P9016 Allele Ablation: R2-Only Evaluation",
        "",
        "**Status:** `evaluation_complete`. This directory evaluates R2 only; it does not run or report R1/R3.",
        "",
        "- 21 conditions: 15 new variant×bundle conditions and 6 historical controls; the output retains 20 chromosomes×21 conditions=420 rows. Failed planned arms remain n/a and are not removed from the table or common mask.",
        "- For each chromosome, the common mask uses unordered, non-diagonal 1 Mb pairs across all available finite candidate/control and both reference tracks; positions follow `range(3Mb, L, 1Mb)`. Actual coverage is written to each mask's metadata; expected pair counts cannot replace the audit.",
        "- The fixed reference columns are mat/pat. Each chromosome permits at most one whole-candidate swap; when the direct/cross difference is at most 1e-12, orientation is unresolved, matched/cross use the symmetric mean, `contrast=0`, named margins are n/a, and the result is not counted as both-positive.",
        "- Consensus reports single-track mean similarity only when both reference rho values are finite; contrast is n/a. Constant, nonfinite, or fewer than 20 common pairs all yield n/a with the reason retained.",
        "- The primary comparison first pairs each seed's 20 chromosome deltas by bundle, then averages the three planned seeds on the same chromosome, and finally computes the macro summary. If any planned seed fails, primary is n/a; available-only results are marked separately.",
        "- Bootstrap uses a fixed seed=9301 and a 10000×20 index matrix shared by every comparison; CI describes technical/structural variation within one cell, not biological replicates or a p-value.",
        "",
        "## Gating",
        "",
        "- Every available candidate/control first undergoes coordinate SHA verification and gate registration, then reference is read; `evaluation_manifest.json` stores gate, source, coordinate, reference, and bootstrap hashes.",
        "- R2 does not reselect or tune parameters; the within-variant count-selected representative is read only from the locked release manifest record.",
        "",
        "## Outputs",
        "",
        "- `r2_per_chromosome.tsv/.csv/.json`: all 420 condition records, four rho values, matched/cross/contrast, both margins, orientation, and coverage/status.",
        "- The primary pairs table `r2_paired_effects.tsv/.json` contains 5 preregistered variant comparisons × 8 metrics (similarity/matched, cross, contrast, matched_ref1/2, margin_mat/pat, minmargin) = 40 metric comparisons, retaining records for all three seeds and the three-seed aggregate.",
        "- `plots/`: matched, cross, contrast, both margins, paired effects, and seed-stability figures split by representative condition/historical control, all at 300 DPI and 7 pt.",
        "",
        "This run's config SHA: `%s`; release manifest SHA: `%s`; protocol SHA: `%s`." %
        (manifest["config_sha256"], manifest["release_manifest_sha256"], manifest["protocol_sha256"]),
    ]
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or release-gate P9016 allele ablation R2 evaluation")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true", help="write lock/config only; never open real coordinates/reference")
    modes.add_argument("--evaluate-released", action="store_true", help="verify release manifest, then run gated R2")
    parser.add_argument("--output-dir", required=True, help="fresh preparation or final evaluation-r2 output directory")
    parser.add_argument("--config", help="prepared config.json; required for --evaluate-released")
    parser.add_argument("--release-manifest", help="parent-locked release manifest; required for --evaluate-released")
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH), help="authoritative protocol JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.prepare_only:
            if args.config or args.release_manifest:
                raise AlleleR2Error("--config/--release-manifest are not used with --prepare-only")
            result = prepare_only(args.output_dir, protocol_path=args.protocol)
            print(json.dumps({"status": result["status"], "output_dir": result["output_dir"],
                              "condition_count": result["expected_condition_count"],
                              "coordinate_files_opened": result["coordinate_files_opened"],
                              "reference_opened": result["reference_opened"]}, sort_keys=True))
            return 0
        if not args.config or not args.release_manifest:
            raise AlleleR2Error("--config and --release-manifest are required with --evaluate-released")
        result = evaluate_released(args.config, args.release_manifest, args.output_dir)
        print(json.dumps({"status": "evaluation_complete", "output_dir": result["output_dir"],
                          "condition_count": result["condition_count"], "row_count": result["row_count"],
                          "failed_conditions": result["failed_conditions"]}, sort_keys=True))
        return 0
    except (AlleleR2Error, AssertionError, ValueError, OSError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


__all__ = [
    "ALL_CONDITION_IDS", "AUTHORITATIVE_PROTOCOL_SHA256", "BUNDLE_IDS", "CONFIG_SCHEMA",
    "EVALUATION_SCHEMA", "HISTORICAL_IDS", "MIN_RHO_PAIRS", "NEW_CONDITION_IDS",
    "PRIMARY_VARIANT_COMPARISONS", "PRIMARY_PAIR_METRICS", "RELEASE_SCHEMA", "VARIANT_IDS", "AlleleR2Error",
    "bootstrap_index_sha256", "bootstrap_file_sha256", "build_prepared_config", "build_release_manifest_from_experiment", "condition_summary", "evaluate_in_memory",
    "evaluate_released", "flatten_rows", "flatten_paired_rows", "write_paired_table", "main", "make_bootstrap_index_matrix",
    "prepare_only", "summarise_paired_seed_effects",
]


if __name__ == "__main__":
    raise SystemExit(main())
