"""P3 recovery run 的独立 endpoint accounting audit。

该 audit 使用独立的 upper-triangle index，根据已保存的 aggregate counts 和 diagonal counts 重新构建每个 locus 的 endpoint mass。它不使用 ``endpoint_counts_from_aggregates``，也不修改 pre-truth gate 或任何冻结的 model source。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GROUP_TOTALS = {
    "diag": 438_774,
    "cis_offdiag": 696_680,
    "inter": 568_434,
}
EXPECTED_RECORDS = sum(EXPECTED_GROUP_TOTALS.values())
EXPECTED_ENDPOINTS = 2 * EXPECTED_RECORDS
CONDITIONS = ("A", "B", "C", "D")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: dict) -> None:
    with open(path, "wt") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _append_status(path: Path, value: dict) -> None:
    with open(path, "at") as handle:
        handle.write(json.dumps(_jsonable(value), sort_keys=True) + "\n")


def _raw_layer(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "chromosome_names": tuple(str(value) for value in payload["chromosome_names"].tolist()),
            "chromosome_lengths": payload["chromosome_lengths"].astype(np.int64, copy=True),
            "bin_size": int(payload["bin_size"]),
            "counts": payload["counts"].copy(),
            "diag_counts": payload["diag_counts"].copy(),
            "endpoint_counts": payload["endpoint_counts"].copy(),
            "exposure": payload["exposure"].astype(np.float64, copy=True),
            "count_mode": str(payload["count_mode"].item()),
            "exposure_mode": str(payload["exposure_mode"].item()),
        }


def _audit_condition(run_dir: Path, condition: str, source_manifest: dict) -> dict:
    layer_path = run_dir / "inputs_snapshot" / (condition + "_1mb_layer.npz")
    layer = _raw_layer(layer_path)
    names = layer["chromosome_names"]
    lengths = layer["chromosome_lengths"]
    n_bins = (lengths + layer["bin_size"] - 1) // layer["bin_size"]
    n_loci = int(n_bins.sum())
    if len(names) != 20 or n_loci != 2645:
        raise RuntimeError("unexpected header grid for %s" % condition)
    pair_i, pair_j = np.triu_indices(n_loci, k=1)
    counts = np.asarray(layer["counts"])
    diag = np.asarray(layer["diag_counts"])
    saved = np.asarray(layer["endpoint_counts"])
    if counts.shape != pair_i.shape or diag.shape != (n_loci,) or saved.shape != (n_loci,):
        raise RuntimeError("aggregate shape mismatch for %s" % condition)
    locus_chromosome = np.repeat(np.arange(len(names), dtype=np.int64), n_bins)
    cis_mask = locus_chromosome[pair_i] == locus_chromosome[pair_j]
    # 独立重建：不调用项目的 endpoint helper。
    if np.issubdtype(counts.dtype, np.integer) and np.issubdtype(diag.dtype, np.integer):
        derived = (2 * diag.astype(np.int64, copy=False)).copy()
        np.add.at(derived, pair_i, counts.astype(np.int64, copy=False))
        np.add.at(derived, pair_j, counts.astype(np.int64, copy=False))
        comparison_mode = "exact_integer"
        endpoint_match = bool(np.array_equal(saved, derived))
        max_abs_difference = int(np.max(np.abs(saved.astype(np.int64) - derived.astype(np.int64))))
        mismatch_count = int(np.count_nonzero(saved != derived))
        endpoint_total_match = bool(int(saved.sum()) == EXPECTED_ENDPOINTS)
        endpoint_total_difference = int(saved.sum()) - EXPECTED_ENDPOINTS
    else:
        derived = (2.0 * diag.astype(np.float64, copy=False)).copy()
        np.add.at(derived, pair_i, counts.astype(np.float64, copy=False))
        np.add.at(derived, pair_j, counts.astype(np.float64, copy=False))
        comparison_mode = "tight_float"
        difference = np.abs(saved.astype(np.float64) - derived)
        endpoint_match = bool(np.allclose(saved, derived, rtol=1e-10, atol=1e-10))
        max_abs_difference = float(difference.max())
        mismatch_count = int(np.count_nonzero(difference > 1e-10))
        saved_total = float(saved.sum())
        endpoint_total_difference = saved_total - EXPECTED_ENDPOINTS
        endpoint_total_match = bool(np.isclose(saved_total, EXPECTED_ENDPOINTS, rtol=1e-10, atol=1e-10))

    group_totals = {
        "diag": float(diag.sum()),
        "cis_offdiag": float(counts[cis_mask].sum()),
        "inter": float(counts[~cis_mask].sum()),
    }
    group_differences = {
        key: float(group_totals[key] - EXPECTED_GROUP_TOTALS[key])
        for key in EXPECTED_GROUP_TOTALS
    }
    if comparison_mode == "exact_integer":
        group_budget_pass = bool(all(group_totals[key] == EXPECTED_GROUP_TOTALS[key]
                                     for key in EXPECTED_GROUP_TOTALS))
    else:
        group_budget_pass = bool(all(np.isclose(group_totals[key], EXPECTED_GROUP_TOTALS[key],
                                                 rtol=1e-10, atol=1e-10)
                                     for key in EXPECTED_GROUP_TOTALS))

    source_key = "test_res/018-20260913_121446-v1-synthetic-calibration/work/%s_1mb_layer.npz" % condition
    source_entry = source_manifest["entries"][source_key]
    source_hash = _sha256(ROOT / "test_res/018-20260913_121446-v1-synthetic-calibration/work" / (condition + "_1mb_layer.npz"))
    snapshot_hash = _sha256(layer_path)
    source_hash_pass = bool(source_hash == source_entry["sha256"] == snapshot_hash)

    observed_exposure = np.sqrt(np.asarray(derived, dtype=np.float64) + 10.0)
    observed_exposure /= observed_exposure.mean()
    exposure_difference = np.abs(layer["exposure"] - observed_exposure)
    endpoint_exposure_required = layer["exposure_mode"] in {
        "production_observed_endpoint",
        "production_observed_endpoint_recomputed",
    }
    if endpoint_exposure_required:
        exposure_match = bool(np.allclose(layer["exposure"], observed_exposure, rtol=0.0, atol=1e-14))
        exposure_gate = exposure_match
        exposure_policy = "required_and_checked"
    else:
        # 已知的 synthetic exposure 有意不要求等于 sqrt endpoints。
        exposure_match = None
        exposure_gate = True
        exposure_policy = "not_required_known_exposure_is_distinct"

    # 常规 API 对 synthetic_expected endpoint conservation 返回 null；
    # 这个独立计算补充缺失的显式检查。
    api_endpoint_conserved = None if layer["count_mode"] == "synthetic_expected" else endpoint_total_match
    checks = {
        "source_hash": source_hash_pass,
        "full_grid": bool(n_loci == 2645 and len(pair_i) == n_loci * (n_loci - 1) // 2),
        "group_budget": group_budget_pass,
        "endpoint_vector_match": endpoint_match,
        "endpoint_total_2M": endpoint_total_match,
        "endpoint_exposure_policy": exposure_gate,
    }
    return {
        "condition": condition,
        "count_mode": layer["count_mode"],
        "exposure_mode": layer["exposure_mode"],
        "n_loci": n_loci,
        "n_pairs": int(len(pair_i)),
        "group_totals_rebuilt": group_totals,
        "group_total_differences_from_frozen": group_differences,
        "saved_endpoint_dtype": str(saved.dtype),
        "comparison_mode": comparison_mode,
        "endpoint_vector_match": endpoint_match,
        "endpoint_vector_max_abs_difference": max_abs_difference,
        "endpoint_vector_mismatch_count": mismatch_count,
        "saved_endpoint_total": _jsonable(saved.sum()),
        "derived_endpoint_total": _jsonable(derived.sum()),
        "expected_endpoint_total_2M": EXPECTED_ENDPOINTS,
        "endpoint_total_difference": endpoint_total_difference,
        "endpoint_total_match": endpoint_total_match,
        "contact_model_budget_endpoint_conserved": api_endpoint_conserved,
        "independent_endpoint_conservation_check": endpoint_total_match,
        "saved_exposure_mode": layer["exposure_mode"],
        "observed_endpoint_exposure_formula": "normalize(sqrt(endpoint_counts + 10))",
        "saved_vs_observed_endpoint_exposure_max_abs_difference": float(exposure_difference.max()),
        "endpoint_exposure_required": endpoint_exposure_required,
        "endpoint_exposure_match": exposure_match,
        "endpoint_exposure_policy": exposure_policy,
        "source_hash": source_hash,
        "snapshot_hash": snapshot_hash,
        "source_hash_matches_018": source_hash_pass,
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }


def audit_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir).resolve()
    source_manifest = json.loads((run_dir / "source_snapshot_hashes.json").read_text())
    if source_manifest.get("truth_artifacts_copied") is not False:
        raise RuntimeError("source snapshot unexpectedly includes truth artifacts")
    conditions = {
        condition: _audit_condition(run_dir, condition, source_manifest)
        for condition in CONDITIONS
    }
    result = {
        "schema": "v1-p3-independent-endpoint-audit-v1",
        "run_id": run_dir.name,
        "method": "independent reconstruction from saved counts and diag_counts using np.triu_indices and np.add.at",
        "frozen_group_totals": EXPECTED_GROUP_TOTALS,
        "frozen_records_M": EXPECTED_RECORDS,
        "expected_endpoint_total_2M": EXPECTED_ENDPOINTS,
        "contact_model_endpoint_conserved_api_note": "synthetic_expected returns null; this audit explicitly checks it",
        "known_exposure_policy_note": "known synthetic exposure is not required to equal sqrt(endpoint_counts + 10); production endpoint exposure is checked when required",
        "conditions": conditions,
        "all_conditions_pass": bool(all(row["all_checks_pass"] for row in conditions.values())),
    }
    output = run_dir / "work" / "posttruth_endpoint_audit.json"
    _write_json(output, result)
    _append_status(run_dir / "logs" / "status.jsonl", {
        "event": "independent_endpoint_audit_completed",
        "artifact": str(output.relative_to(run_dir)),
        "all_conditions_pass": result["all_conditions_pass"],
        "expected_endpoint_total_2M": EXPECTED_ENDPOINTS,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent P3 endpoint accounting audit")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(_jsonable(audit_run(args.run_dir)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
