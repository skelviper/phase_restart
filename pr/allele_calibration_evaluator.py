"""按染色体评估 synthetic R2 allele calibration。

本模块有意不调用历史的 ``allele_calibration.evaluate_run`` 实现。每条染色体独立进行 candidate rows 的全局交换，而 truth reference columns 固定为 mat（ref1）和 pat（ref2）。只有 fit manifest 和每个 candidate coordinate 哈希通过后，才会打开 truth 文件。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from . import allele_calibration as prepare
from .score import spearman


ROOT = Path(__file__).resolve().parents[1]
PREPARED_SCHEMA = prepare.PREPARED_SCHEMA
CANDIDATE_SCHEMA = prepare.CANDIDATE_SCHEMA
FIT_MANIFEST_SCHEMA = "p9016-r2-allele-calibration-fit-manifest-v1"
EVALUATION_SCHEMA = "p9016-r2-allele-calibration-evaluation-v2"
FIXTURES = prepare.FIXTURES
MODEL_IDS = tuple(prepare.MODEL_IDS)
TIE_TOL = 1e-12
FIT_ENDPOINT_STATUSES = {"solver_converged", "budget_not_converged", "solver_failed"}


class EvaluationError(RuntimeError):
    """评估侧 hash 或指标契约无效时抛出。"""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise EvaluationError("nonfinite value cannot enter evaluation JSON")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError("missing evaluation artifact: %s" % path) from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError("invalid evaluation JSON: %s" % path) from exc
    if not isinstance(value, dict):
        raise EvaluationError("evaluation artifact must be an object: %s" % path)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_inside(path: str | Path, root: Path) -> Path:
    candidate = (root / Path(path)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise EvaluationError("path escapes run directory: %s" % path) from exc
    return candidate


def _bounded_rho(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 20 or right.size < 20 or left.size != right.size:
        return None
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return None
    if np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = float(spearman(left, right))
    if not math.isfinite(value):
        return None
    return float(np.clip(value, -1.0, 1.0))


def _required_mean(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        return None
    return float((float(left) + float(right)) / 2.0)


def _optional_mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else float(np.mean(finite))


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _distance_vectors(coordinates: np.ndarray, chromosome_slice: slice) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape[0] != 2 or values.shape[2] != 3:
        raise EvaluationError("coordinates must have shape (2, n_loci, 3)")
    n = chromosome_slice.stop - chromosome_slice.start
    local_i, local_j = np.triu_indices(n, k=1)
    indices_i = chromosome_slice.start + local_i
    indices_j = chromosome_slice.start + local_j
    distance_a = np.linalg.norm(values[0, indices_i] - values[0, indices_j], axis=1)
    distance_b = np.linalg.norm(values[1, indices_i] - values[1, indices_j], axis=1)
    return distance_a.astype(np.float64, copy=False), distance_b.astype(np.float64, copy=False)


def _fixed_abcd(raw: Mapping[str, float | None], orientation: str) -> tuple[float | None, ...] | None:
    if orientation == "direct":
        return (raw["rho_A_mat"], raw["rho_A_pat"], raw["rho_B_mat"], raw["rho_B_pat"])
    if orientation == "swapped":
        # 只交换 candidate rows；truth reference columns 保持 mat/pat。
        return (raw["rho_B_mat"], raw["rho_B_pat"], raw["rho_A_mat"], raw["rho_A_pat"])
    return None


def _derived_from_abcd(abcd: tuple[float | None, ...] | None) -> dict[str, float | None]:
    if abcd is None or any(value is None for value in abcd):
        return {"matched": None, "cross": None, "contrast": None,
                "margin_ref1": None, "margin_ref2": None, "minmargin": None}
    a, b, c, d = (float(value) for value in abcd)
    matched = (a + d) / 2.0
    cross = (b + c) / 2.0
    return {
        "matched": matched,
        "cross": cross,
        "contrast": matched - cross,
        "margin_ref1": a - b,
        "margin_ref2": d - c,
        "minmargin": min(a - b, d - c),
    }


def _shape_errors(candidate_a: np.ndarray, candidate_b: np.ndarray,
                  truth_a: np.ndarray, truth_b: np.ndarray,
                  *, same_shape: bool, orientation: str) -> dict[str, float | None]:
    candidate_scale = _rms(np.concatenate([candidate_a, candidate_b]))
    truth_scale = _rms(np.concatenate([truth_a, truth_b]))
    if not math.isfinite(candidate_scale) or candidate_scale <= 0.0:
        candidate_scale = float("nan")
    if not math.isfinite(truth_scale) or truth_scale <= 0.0:
        truth_scale = float("nan")
    if same_shape:
        # N 没有可识别的 truth copy。两个 candidate rows 都与同一个共同 truth matrix 比较，
        # 不推断 identity。
        assigned_a, assigned_b = truth_a, truth_b
    elif orientation == "direct":
        assigned_a, assigned_b = truth_a, truth_b
    elif orientation == "swapped":
        # 对跨 orientation 的比较，交换 candidate rows，而不是 truth rows。
        candidate_a, candidate_b = candidate_b, candidate_a
        assigned_a, assigned_b = truth_a, truth_b
    else:
        return {
            "candidate_offdiag_rms": None,
            "truth_offdiag_rms": None,
            "copy_first": None,
            "copy_second": None,
            "mean": None,
            "max": None,
        }
    if not math.isfinite(candidate_scale) or not math.isfinite(truth_scale):
        return {
            "candidate_offdiag_rms": None,
            "truth_offdiag_rms": None,
            "copy_first": None,
            "copy_second": None,
            "mean": None,
            "max": None,
        }
    error_a = _rms(candidate_a / candidate_scale - assigned_a / truth_scale)
    error_b = _rms(candidate_b / candidate_scale - assigned_b / truth_scale)
    return {
        "candidate_offdiag_rms": candidate_scale,
        "truth_offdiag_rms": truth_scale,
        "copy_first": error_a,
        "copy_second": error_b,
        "mean": (error_a + error_b) / 2.0,
        "max": max(error_a, error_b),
    }


def chromosome_metrics(template: Any, candidate: np.ndarray, truth: np.ndarray,
                       *, same_shape: bool, chromosome: int,
                       common_mask: np.ndarray | None = None) -> dict[str, Any]:
    """计算一条染色体的 raw/derived/fixed-reference 指标。"""
    chromosome_slice = template.chromosome_slice(chromosome)
    candidate_a, candidate_b = _distance_vectors(candidate, chromosome_slice)
    truth_a, truth_b = _distance_vectors(truth, chromosome_slice)
    finite_mask = (
        np.isfinite(candidate_a) & np.isfinite(candidate_b)
        & np.isfinite(truth_a) & np.isfinite(truth_b)
    )
    if common_mask is not None:
        common_mask = np.asarray(common_mask, dtype=bool)
        if common_mask.shape != finite_mask.shape or not np.array_equal(common_mask, finite_mask):
            raise EvaluationError("candidate/truth common finite mask changed across variants")
        finite_mask = common_mask
    candidate_a = candidate_a[finite_mask]
    candidate_b = candidate_b[finite_mask]
    truth_a = truth_a[finite_mask]
    truth_b = truth_b[finite_mask]
    if same_shape:
        common_truth = (truth_a + truth_b) / 2.0
        truth_a_eval, truth_b_eval = common_truth, common_truth
    else:
        common_truth = None
        truth_a_eval, truth_b_eval = truth_a, truth_b
    raw = {
        "rho_A_mat": _bounded_rho(candidate_a, truth_a_eval),
        "rho_A_pat": _bounded_rho(candidate_a, truth_b_eval),
        "rho_B_mat": _bounded_rho(candidate_b, truth_a_eval),
        "rho_B_pat": _bounded_rho(candidate_b, truth_b_eval),
    }
    direct = _required_mean(raw["rho_A_mat"], raw["rho_B_pat"])
    cross = _required_mean(raw["rho_A_pat"], raw["rho_B_mat"])
    tie = direct is not None and cross is not None and abs(direct - cross) <= TIE_TOL
    if tie:
        orientation = "unresolved_tie"
    elif direct is None or cross is None:
        orientation = "unresolved_missing"
    elif direct > cross:
        orientation = "direct"
    else:
        orientation = "swapped"
    abcd = _fixed_abcd(raw, orientation)
    derived = _derived_from_abcd(abcd)
    if tie:
        # 保留 raw four-rho，但不选择命名的 reference orientation。
        tie_value = (float(direct) + float(cross)) / 2.0
        derived = {
            "matched": tie_value,
            "cross": tie_value,
            "contrast": 0.0,
            "margin_ref1": None,
            "margin_ref2": None,
            "minmargin": None,
        }
    errors = _shape_errors(
        candidate_a, candidate_b, truth_a_eval, truth_b_eval,
        same_shape=same_shape, orientation=orientation,
    )
    n_only = None
    if same_shape and errors["candidate_offdiag_rms"] is not None:
        n_only = _rms((candidate_a - candidate_b) / float(errors["candidate_offdiag_rms"]))
    record: dict[str, Any] = {
        "chromosome": template.chromosome_names[chromosome],
        "chromosome_index": chromosome,
        "n_offdiag_pairs_total": int(len(finite_mask)),
        "n_offdiag_pairs_finite_common_mask": int(np.sum(finite_mask)),
        "same_finite_common_mask": bool(np.all(finite_mask)),
        "common_truth_matrix": bool(same_shape),
        "raw_four_rho": raw,
        "direct_original": direct,
        "cross_original": cross,
        "orientation": orientation,
        "geometry_tie": bool(tie),
        "tie_tolerance": TIE_TOL,
        "fixed_reference_abcd": None if tie else abcd,
        "matched": derived["matched"],
        "cross": derived["cross"],
        "contrast": derived["contrast"],
        "margin_ref1_mat": derived["margin_ref1"],
        "margin_ref2_pat": derived["margin_ref2"],
        "minmargin": derived["minmargin"],
        "pooled_candidate_offdiag_rms": errors["candidate_offdiag_rms"],
        "pooled_truth_offdiag_rms": errors["truth_offdiag_rms"],
        "shape_error": {
            "copy_first_or_assigned_first": errors["copy_first"],
            "copy_second_or_assigned_second": errors["copy_second"],
            "mean": errors["mean"],
            "max": errors["max"],
        },
        "N_only_copy_difference_rms": n_only,
        "N_only_copy_difference_is_not_a_positive_score": True,
        "truth_matrix_policy": "common mean duplicated for N" if same_shape else "own mat/pat for P",
        "reference_columns_fixed": {"ref1": "mat", "ref2": "pat"},
        "candidate_local_swap": False,
    }
    return record


def _macro_mean_count(records: Sequence[Mapping[str, Any]], path: Sequence[str]) -> tuple[float | None, int]:
    values: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        try:
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        except (TypeError, ValueError):
            continue
    return (None, 0) if not values else (float(np.mean(values)), len(values))


def _macro_summary(records: Sequence[Mapping[str, Any]], same_shape: bool) -> dict[str, Any]:
    orientation_counts = {"direct": 0, "swapped": 0, "unresolved_tie": 0, "unresolved_missing": 0}
    for record in records:
        orientation = str(record.get("orientation"))
        orientation_counts[orientation] = orientation_counts.get(orientation, 0) + 1
    raw_values = {}
    for key in ("rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat"):
        value, count = _macro_mean_count(records, ("raw_four_rho", key))
        raw_values[key] = value
        raw_values[key + "_valid_chr_count"] = count
    numeric_fields = (
        "direct_original", "cross_original", "matched", "cross", "contrast",
        "margin_ref1_mat", "margin_ref2_pat", "minmargin",
        "pooled_candidate_offdiag_rms", "pooled_truth_offdiag_rms",
    )
    shape_fields = ("copy_first_or_assigned_first", "copy_second_or_assigned_second", "mean", "max")
    output: dict[str, Any] = {
        "n_chromosomes": len(records),
        "selection_policy": "one independent whole-chromosome swap per chromosome; no pooled genome swap",
        "raw_four_rho_macro": raw_values,
        "orientation_counts": orientation_counts,
        "common_truth_matrix": bool(same_shape),
    }
    for field in numeric_fields:
        value, count = _macro_mean_count(records, (field,))
        output[field + "_macro"] = value
        output[field + "_valid_chr_count"] = count
    shape_summary: dict[str, Any] = {}
    for field in shape_fields:
        value, count = _macro_mean_count(records, ("shape_error", field))
        shape_summary[field] = value
        shape_summary[field + "_valid_chr_count"] = count
    output["shape_error_macro"] = shape_summary
    n_only_value, n_only_count = _macro_mean_count(records, ("N_only_copy_difference_rms",))
    output["N_only_copy_difference_rms_macro"] = n_only_value if same_shape else None
    output["N_only_copy_difference_rms_valid_chr_count"] = n_only_count if same_shape else 0
    if same_shape:
        if output["contrast_valid_chr_count"]:
            output["contrast_algebraic_policy"] = "0 for each valid N chromosome because the common truth is duplicated; not a method score"
        else:
            output["contrast_algebraic_policy"] = "n/a: no valid N chromosome contrast"
    return output


def _verify_fit_lock(run_dir: Path) -> tuple[dict[str, Any], Path]:
    fit_manifest_path = run_dir / "results" / "fit_manifest.json"
    fit_manifest = _read_json(fit_manifest_path)
    if fit_manifest.get("schema") != FIT_MANIFEST_SCHEMA:
        raise EvaluationError("fit manifest schema mismatch")
    if fit_manifest.get("status") != "complete_terminal":
        raise EvaluationError("evaluation requires complete_terminal fit manifest")
    rows = fit_manifest.get("jobs")
    if not isinstance(rows, list) or len(rows) != len(FIXTURES) * len(MODEL_IDS):
        raise EvaluationError("fit manifest does not preserve all 20 arms")
    for row in rows:
        if row.get("status") not in FIT_ENDPOINT_STATUSES:
            raise EvaluationError("fit arm is not an evaluation-eligible terminal endpoint")
        if not row.get("selection_eligible") or not row.get("final_json_sha256"):
            raise EvaluationError("fit endpoint is not hash locked")
        final_path = _resolve_inside(str(row["final_json_path"]), run_dir)
        if _sha256_file(final_path) != str(row["final_json_sha256"]):
            raise EvaluationError("fit final JSON hash changed: %s" % final_path)
        final = _read_json(final_path)
        coordinate = final.get("final", {}).get("coordinates_file", {})
        coordinate_path = _resolve_inside(str(coordinate.get("path", "")), run_dir)
        if not coordinate_path.is_file() or _sha256_file(coordinate_path) != str(coordinate.get("sha256")):
            raise EvaluationError("fit coordinate output hash changed for %s" % row.get("job_id"))
    candidate_info = fit_manifest.get("candidate_manifest")
    if not isinstance(candidate_info, Mapping):
        raise EvaluationError("fit manifest lacks locked candidate manifest")
    candidate_path = _resolve_inside(str(candidate_info.get("path", "")), run_dir)
    if _sha256_file(candidate_path) != str(candidate_info.get("sha256")):
        raise EvaluationError("candidate manifest hash changed")
    return fit_manifest, candidate_path


def _verify_evaluation_source_lock(run_dir: Path, fit_manifest: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = run_dir / "freeze" / "evaluation_source_hashes.json"
    lock = _read_json(lock_path)
    source = lock.get("evaluator_source")
    if not isinstance(source, Mapping):
        raise EvaluationError("evaluation source freeze is missing evaluator hash")
    source_path = ROOT / str(source.get("path", ""))
    if not source_path.is_file() or _sha256_file(source_path) != str(source.get("sha256")):
        raise EvaluationError("evaluator source changed after source freeze")
    if lock.get("fit_manifest_sha256") != _sha256_file(run_dir / "results" / "fit_manifest.json"):
        raise EvaluationError("fit manifest changed after evaluator source freeze")
    if lock.get("candidate_manifest") != fit_manifest.get("candidate_manifest"):
        raise EvaluationError("candidate lock changed after evaluator source freeze")
    return lock


def _finite_distance_mask(template: Any, candidate: np.ndarray, truth: np.ndarray,
                          chromosome: int) -> np.ndarray:
    chromosome_slice = template.chromosome_slice(chromosome)
    candidate_a, candidate_b = _distance_vectors(candidate, chromosome_slice)
    truth_a, truth_b = _distance_vectors(truth, chromosome_slice)
    return (
        np.isfinite(candidate_a) & np.isfinite(candidate_b)
        & np.isfinite(truth_a) & np.isfinite(truth_b)
    )


def evaluate_run(run_dir: str | Path, candidate_manifest_path: str | Path | None = None) -> dict[str, Any]:
    """使用每条染色体独立的 swap 评估锁定的 endpoints。"""
    run_dir = Path(run_dir).resolve()
    prepared = _read_json(run_dir / "work" / "prepared_manifest.json")
    if prepared.get("schema") != PREPARED_SCHEMA or prepared.get("optimizer_started"):
        raise EvaluationError("prepared manifest is not a clean worker-facing manifest")
    fit_manifest, locked_candidate_path = _verify_fit_lock(run_dir)
    evaluation_source_lock = _verify_evaluation_source_lock(run_dir, fit_manifest)
    candidate_path = locked_candidate_path if candidate_manifest_path is None else _resolve_inside(candidate_manifest_path, run_dir)
    if candidate_path != locked_candidate_path:
        raise EvaluationError("evaluation candidate path differs from fit-locked candidate manifest")
    candidate_manifest = _read_json(candidate_path)
    template = prepare._validate_templates(prepare.header_templates())
    # 在打开第一个 truth path 前，已核验全部 20 个 candidate file 的 hash/domain。
    candidates = prepare._validate_candidate_manifest(run_dir, candidate_manifest, template)
    by_fixture: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_fixture.setdefault(str(candidate["fixture_id"]), []).append(candidate)
    truth_manifest = _read_json(run_dir / "provenance" / "truth_manifest.json")
    truth_records = truth_manifest.get("fixtures", {})
    results: list[dict[str, Any]] = []
    for fixture_id in prepare.FIXTURES:
        fixture_candidates = by_fixture.get(fixture_id, [])
        if len(fixture_candidates) != len(MODEL_IDS):
            raise EvaluationError("fixture %s does not contain all five candidate variants" % fixture_id)
        truth_record = truth_records.get(fixture_id)
        if not isinstance(truth_record, Mapping):
            raise EvaluationError("truth manifest lacks fixture %s" % fixture_id)
        truth_path = _resolve_inside(str(truth_record["npz"]), run_dir)
        if _sha256_file(truth_path) != str(truth_record["npz_sha256"]):
            raise EvaluationError("truth coordinate hash changed for %s" % fixture_id)
        metadata_path = _resolve_inside(str(truth_record["metadata_json"]), run_dir)
        if _sha256_file(metadata_path) != str(truth_record["metadata_sha256"]):
            raise EvaluationError("truth metadata hash changed for %s" % fixture_id)
        truth, _generation_exposure, truth_metadata = prepare._load_truth(truth_path)
        masks_by_chromosome: list[np.ndarray] = []
        for chromosome in range(len(template.chromosome_names)):
            masks = [
                _finite_distance_mask(template, candidate["coordinates"], truth, chromosome)
                for candidate in fixture_candidates
            ]
            if any(not np.array_equal(masks[0], mask) for mask in masks[1:]):
                raise EvaluationError("fixture %s chromosome %d has variant-dependent finite mask" % (fixture_id, chromosome))
            if not np.all(masks[0]):
                raise EvaluationError("fixture %s chromosome %d is not full-finite on the shared mask" % (fixture_id, chromosome))
            masks_by_chromosome.append(masks[0])
        for candidate in fixture_candidates:
            same_shape = bool(FIXTURES[fixture_id]["same_shape"])
            per_chromosome = [
                chromosome_metrics(
                    template, candidate["coordinates"], truth,
                    same_shape=same_shape, chromosome=chromosome,
                    common_mask=masks_by_chromosome[chromosome],
                )
                for chromosome in range(len(template.chromosome_names))
            ]
            results.append({
                "fixture_id": fixture_id,
                "model_id": candidate["model_id"],
                "candidate_file_sha256": candidate["file_sha256"],
                "truth_file_sha256": str(truth_record["npz_sha256"]),
                "truth_seed": truth_metadata.get("truth_seed"),
                "p": candidate["p"],
                "selection_policy": "independent one-whole-chromosome swap",
                "primary_macro": _macro_summary(per_chromosome, same_shape),
                "per_chromosome": per_chromosome,
            })
    output = {
        "schema": EVALUATION_SCHEMA,
        "run_id": run_dir.name,
        "status": "complete",
        "fit_manifest_sha256": _sha256_file(run_dir / "results" / "fit_manifest.json"),
        "evaluation_source_freeze_sha256": _sha256_file(run_dir / "freeze" / "evaluation_source_hashes.json"),
        "evaluator_source_sha256": evaluation_source_lock["evaluator_source"]["sha256"],
        "candidate_hash_gate": "passed_before_truth_access",
        "candidate_count": len(results),
        "results": results,
        "interpretation": {
            "per_chromosome_global_swap": True,
            "pooled_genome_swap_used": False,
            "N_common_truth_matrix": True,
            "N_only_copy_difference_not_a_positive_score": True,
            "truth_distance_rho_not_used_for_seed_or_model_selection": True,
            "no_biological_replicate_claim": True,
            "shared_full_finite_mask_across_five_variants": True,
        },
    }
    output_path = run_dir / "results" / "evaluation.json"
    if output_path.exists():
        raise EvaluationError("refusing to overwrite existing evaluation: %s" % output_path)
    _write_json(output_path, output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = evaluate_run(args.run_dir)
    except (EvaluationError, prepare.CalibrationError, AssertionError, ValueError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(_jsonable({"status": result["status"], "candidate_count": result["candidate_count"]}), sort_keys=True))
    return 0


__all__ = ["EVALUATION_SCHEMA", "EvaluationError", "chromosome_metrics", "evaluate_run", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
