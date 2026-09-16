"""Prepare and evaluate the post-020 synthetic allele calibration.

The prepare stage creates four fixed integer-count fixtures and five model views per
fixture. It never calls an optimizer. Worker-facing artifacts contain only unlabeled
counts, fit exposure, and one shared independent physical start; isolated synthetic
truth is written separately and is not listed in the worker manifest.

The evaluation stage verifies candidate file hashes before it opens any truth file.
It accepts either an NPZ containing ``coordinates``/``coords`` or the text format
written by :func:`pr.paired_run.write_coordinates`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import genome
from .allele_models import MODEL_IDS, data_for_model, model_spec, validate_physical_for_model
from .contact_model import (
    EPSILON,
    FROZEN_P9016_CIS_RECORDS,
    FROZEN_P9016_INTER_RECORDS,
    FROZEN_P9016_RECORDS,
    JointObjective,
    assert_inside_unit_ball,
    endpoint_counts_from_aggregates,
    q_from_p,
    sha256_file,
    synthetic_integer_clone,
)
from .paired_run import (
    FitConfig,
    PairedStart,
    load_paired_start,
    plan_paired_runs,
    save_paired_start,
)
from .score import spearman
from .v1_calibration import (
    FINAL_BIN,
    FULL_FINAL_LOCI,
    N_CHROMOSOMES,
    N_TRACKS,
    TOTALS_1MB as V1_TOTALS_1MB,
    _load_truth,
    _save_truth,
    generate_truth,
    generation_rates,
    header_templates,
    load_layer,
    observed_endpoint_exposure,
    sample_conditional_counts,
    save_layer,
    synthetic_exposure,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_SCHEMA = "p9016-r2-allele-calibration-v1"
PREPARED_SCHEMA = "p9016-r2-allele-calibration-prepared-v1"
CANDIDATE_SCHEMA = "p9016-r2-allele-calibration-candidates-v1"
TRUTH_SCHEMA = "p9016-r2-allele-calibration-truth-v1"

TOTALS_1MB = {
    "diag": 438_774,
    "cis_offdiag": 696_680,
    "inter": 568_434,
}
assert TOTALS_1MB == V1_TOTALS_1MB
assert sum(TOTALS_1MB.values()) == FROZEN_P9016_RECORDS
assert TOTALS_1MB["diag"] + TOTALS_1MB["cis_offdiag"] == FROZEN_P9016_CIS_RECORDS
assert TOTALS_1MB["inter"] == FROZEN_P9016_INTER_RECORDS

P_GEN = 0.8
P_INIT = 0.75
MAX_RADIUS = 0.8
FIT_CONFIG = FitConfig(
    maxiter=80,
    maxfun=270,
    maxls=20,
    ftol=1e-10,
    gtol=1e-6,
    checkpoint_every=20,
)
MODEL_IDS = tuple(MODEL_IDS)
assert MODEL_IDS == ("C0", "C1", "C2-map", "C2-free", "C3")

FIXTURES: dict[str, dict[str, Any]] = {
    "N1": {
        "truth_kind": "same_internal_shape_spatially_separated",
        "same_shape": True,
        "truth_seed": 260101,
        "draw_seed": 260201,
        "exposure_seed": 260301,
        "exposure_sigma": 0.0,
        "exposure_label": "ones_sigma0",
        "init_seed": 260401,
    },
    "N2": {
        "truth_kind": "same_internal_shape_spatially_separated",
        "same_shape": True,
        "truth_seed": 260102,
        "draw_seed": 260202,
        "exposure_seed": 260302,
        "exposure_sigma": 0.4,
        "exposure_label": "lognormal_sigma0.4_no_dropout",
        "init_seed": 260402,
    },
    "P1": {
        "truth_kind": "different_internal_shapes",
        "same_shape": False,
        "truth_seed": 260103,
        "draw_seed": 260203,
        "exposure_seed": 260303,
        "exposure_sigma": 0.0,
        "exposure_label": "ones_sigma0",
        "init_seed": 260403,
    },
    "P2": {
        "truth_kind": "different_internal_shapes",
        "same_shape": False,
        "truth_seed": 260104,
        "draw_seed": 260204,
        "exposure_seed": 260304,
        "exposure_sigma": 0.4,
        "exposure_label": "lognormal_sigma0.4_no_dropout",
        "init_seed": 260404,
    },
}


class CalibrationError(RuntimeError):
    """Raised when a frozen prepare/evaluation contract cannot be verified."""


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
        raise CalibrationError("nonfinite value cannot enter JSON metadata")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CalibrationError("JSON root must be an object: %s" % path)
    return value


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _array_sha256(values: np.ndarray, dtype: str) -> str:
    canonical = np.asarray(values, dtype=dtype, order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _count_arrays_sha256(counts: np.ndarray, diag_counts: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(counts, dtype="<i8", order="C").tobytes(order="C"))
    digest.update(np.asarray(diag_counts, dtype="<i8", order="C").tobytes(order="C"))
    return digest.hexdigest()


def _bounded_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearman(left, right))
    if not math.isfinite(value):
        return value
    return float(np.clip(value, -1.0, 1.0))


def _fit_config_dict() -> dict[str, Any]:
    return FIT_CONFIG.as_dict()


def _build_config(run_dir: Path) -> dict[str, Any]:
    q_init = q_from_p(P_INIT)
    return {
        "schema": RUN_SCHEMA,
        "run_id": run_dir.name,
        "status_at_freeze": "frozen_before_prepare_no_optimizer",
        "scope": {
            "biological_samples": 1,
            "synthetic_fixtures": list(FIXTURES),
            "synthetic_fixtures_are_biological_replicates": False,
            "chromosomes": N_CHROMOSOMES,
            "tracks": N_TRACKS,
            "final_bin_bp": FINAL_BIN,
            "origin_bp": 0,
            "full_grid_loci": FULL_FINAL_LOCI,
            "physical_beads": 2 * FULL_FINAL_LOCI,
            "fit_count": len(FIXTURES) * len(MODEL_IDS),
        },
        "input_policy": {
            "header_source": "inputs/P9016.snpfree.pairs.gz",
            "allowed_information": [
                "verified chromosome names, lengths, and header order",
                "frozen 1 Mb group totals",
            ],
            "forbidden": [
                "phase-bearing columns",
                "reference 3DG",
                "production coordinates",
                "real contacts as observations",
            ],
        },
        "grid": {
            "full_header_grid": True,
            "origin_bp": 0,
            "bin_size_bp": FINAL_BIN,
            "loci": FULL_FINAL_LOCI,
            "tracks": N_TRACKS,
            "all_eligible_pairs_in_rate_and_count_grid": True,
        },
        "group_totals_1mb": dict(TOTALS_1MB),
        "count_generation": {
            "mode": "fixed_total_multinomial_integer",
            "kernel": "v1",
            "epsilon": EPSILON,
            "r0": "2*l0",
            "p_gen": P_GEN,
            "diagonal_nuisance": "independent multinomial over bins using exposure squared",
            "zero_nonzero_accounting": "record actual zeros/nonzeros; do not impose a real zero count",
            "real_zero_count_3009436_used": False,
        },
        "fixtures": _jsonable(FIXTURES),
        "exposure_fit_policy": {
            "C0": "sqrt(observed_endpoint_count+10)/mean, recomputed from synthetic observed endpoints",
            "C1": "sqrt(observed_endpoint_count+10)/mean, recomputed from synthetic observed endpoints",
            "C2-map": "sqrt(observed_endpoint_count+10)/mean, recomputed from synthetic observed endpoints",
            "C2-free": "sqrt(observed_endpoint_count+10)/mean, recomputed from synthetic observed endpoints",
            "C3": "ones, full-grid mean one",
            "generation_exposure_is_never_passed_to_fit": True,
        },
        "model_variants": {model_id: model_spec(model_id).as_dict() for model_id in MODEL_IDS},
        "initialization": {
            "generator": "generate_truth(template, start_seed, same_shape=False)",
            "truth_independent": True,
            "one_global_center_then_scale": True,
            "max_radius": MAX_RADIUS,
            "p_init": P_INIT,
            "q_init": q_init,
            "five_variants_share_one_physical_x0_per_fixture": True,
            "not_native_assignment_initialization": True,
        },
        "fit_budget": _fit_config_dict(),
        "execution": {
            "prepare_runs_optimizer": False,
            "planned_optimizer_calls": 0,
            "max_workers_future": 4,
            "blas_threads_future": 1,
            "omp_threads_future": 1,
            "checkpoint_policy": "every 20 accepted iterations in future worker",
        },
        "evaluation_policy": {
            "candidate_hash_before_truth": True,
            "fixed_all_variant_common_mask": "full 20-chromosome non-diagonal pair grid",
            "overall_swap": "one geometry-best candidate swap per fixture, never local chromosome repair",
            "shape_error": "normalize candidate and truth by pooled offdiag RMS, then per-copy RMS",
            "N_truth_matrix": "use common truth matrix mean for both copies to avoid floating tie artifacts",
            "N_only_diagnostic": "RMS((candidate_copy_A-candidate_copy_B)/pooled_candidate_RMS); never a positive score",
            "truth_distance_rho": "record only; never tune seeds on it",
            "no_performance_gate": True,
        },
        "prohibitions": [
            "do not start optimizer during prepare",
            "do not launch worker until source/interface freeze is released",
            "do not read real reference, phase, production coordinates, or real contacts",
            "do not regenerate an invalid fixture with replacement seeds",
            "do not run R1/R3",
        ],
    }


def _distance_vector(coordinates: np.ndarray, chromosome_slice: slice) -> tuple[np.ndarray, np.ndarray]:
    n = chromosome_slice.stop - chromosome_slice.start
    local_i, local_j = np.triu_indices(n, k=1)
    indices_i = chromosome_slice.start + local_i
    indices_j = chromosome_slice.start + local_j
    delta_a = coordinates[0, indices_i] - coordinates[0, indices_j]
    delta_b = coordinates[1, indices_i] - coordinates[1, indices_j]
    return (
        np.linalg.norm(delta_a, axis=1).astype(np.float64, copy=False),
        np.linalg.norm(delta_b, axis=1).astype(np.float64, copy=False),
    )


def _truth_geometry_gate(template: Any, coordinates: np.ndarray, fixture: Mapping[str, Any]) -> dict[str, Any]:
    per_chromosome = []
    all_a: list[np.ndarray] = []
    all_b: list[np.ndarray] = []
    for chromosome, name in enumerate(template.chromosome_names):
        chromosome_slice = template.chromosome_slice(chromosome)
        distance_a, distance_b = _distance_vector(coordinates, chromosome_slice)
        all_a.append(distance_a)
        all_b.append(distance_b)
        scale = max(
            float(np.sqrt(np.mean(distance_a * distance_a))),
            float(np.sqrt(np.mean(distance_b * distance_b))),
            np.finfo(np.float64).tiny,
        )
        difference = distance_a - distance_b
        normalized_rms = float(np.sqrt(np.mean(difference * difference)) / scale)
        rho = _bounded_spearman(distance_a, distance_b)
        center_separation = float(
            np.linalg.norm(coordinates[0, chromosome_slice.start] - coordinates[1, chromosome_slice.start])
        )
        per_chromosome.append({
            "chromosome": name,
            "n_offdiag_pairs": int(len(distance_a)),
            "max_abs_internal_distance_difference": float(np.max(np.abs(difference))),
            "normalized_rms_internal_distance_difference": normalized_rms,
            "truth_distance_rho": rho,
            "center_separation": center_separation,
        })

    pooled_a = np.concatenate(all_a)
    pooled_b = np.concatenate(all_b)
    pooled_scale = max(
        float(np.sqrt(np.mean(pooled_a * pooled_a))),
        float(np.sqrt(np.mean(pooled_b * pooled_b))),
        np.finfo(np.float64).tiny,
    )
    pooled_normalized_rms = float(np.sqrt(np.mean((pooled_a - pooled_b) ** 2)) / pooled_scale)
    if fixture["same_shape"]:
        passed = (
            all(np.allclose(
                row_a, row_b, rtol=1e-10, atol=0.0,
            ) for row_a, row_b in zip(all_a, all_b))
            and min(row["center_separation"] for row in per_chromosome) > 1e-6
        )
        gate_name = "N_same_shape_and_separated"
    else:
        passed = (
            any(not np.array_equal(row_a, row_b) for row_a, row_b in zip(all_a, all_b))
            and min(row["normalized_rms_internal_distance_difference"] for row in per_chromosome) > 1e-6
        )
        gate_name = "P_different_internal_shapes"
    return {
        "gate": gate_name,
        "passed": bool(passed),
        "invalid_fixture_retained_if_false": True,
        "rtol_for_N_distance_equality": 1e-10,
        "center_separation_threshold": 1e-6,
        "P_min_normalized_rms_threshold": 1e-6,
        "pooled_normalized_rms_internal_distance_difference": pooled_normalized_rms,
        "pooled_truth_distance_rho": _bounded_spearman(pooled_a, pooled_b),
        "per_chromosome": per_chromosome,
    }


def _normalize_initialization(coordinates: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(coordinates, dtype=np.float64)
    pre_center = values.mean(axis=(0, 1))
    centered = values - pre_center
    pre_scale_max_radius = float(np.linalg.norm(centered, axis=2).max())
    if not math.isfinite(pre_scale_max_radius) or pre_scale_max_radius <= 0.0:
        raise CalibrationError("independent initialization has no nonzero radius")
    scale = MAX_RADIUS / pre_scale_max_radius
    normalized = centered * scale
    assert_inside_unit_ball(normalized)
    return normalized, {
        "center": pre_center.tolist(),
        "scale": float(scale),
        "pre_scale_max_radius": pre_scale_max_radius,
        "post_scale_max_radius": float(np.linalg.norm(normalized, axis=2).max()),
        "target_max_radius": MAX_RADIUS,
    }


def _count_audit(template: Any, counts: np.ndarray, diag_counts: np.ndarray) -> dict[str, Any]:
    counts = np.asarray(counts)
    diag_counts = np.asarray(diag_counts)
    groups = {
        "diag": diag_counts,
        "cis_offdiag": counts[template.cis_pair],
        "inter": counts[~template.cis_pair],
    }
    by_group = {}
    for group, values in groups.items():
        values = np.asarray(values)
        by_group[group] = {
            "eligible": int(values.size),
            "zero": int(np.count_nonzero(values == 0)),
            "nonzero": int(np.count_nonzero(values > 0)),
            "total": int(values.sum()),
            "min": int(values.min()),
            "max": int(values.max()),
        }
    return {
        "full_pair_grid": True,
        "n_loci_for_diag": int(template.n_loci),
        "n_non_diagonal_pairs": int(template.n_pairs),
        "by_group": by_group,
        "actual_totals_match_frozen": all(
            by_group[key]["total"] == TOTALS_1MB[key] for key in TOTALS_1MB
        ),
        "real_zero_count_3009436_used": False,
    }


def _save_counts_snapshot(path: Path, template: Any, counts: np.ndarray, diag_counts: np.ndarray,
                          endpoints: np.ndarray, audit: Mapping[str, Any]) -> None:
    """Save one compact raw-count snapshot for audit; no exposure or truth is stored."""
    np.savez_compressed(
        path,
        chromosome_names=np.asarray(template.chromosome_names),
        chromosome_lengths=np.asarray(template.chromosome_lengths, dtype=np.int64),
        bin_size=np.asarray(template.bin_size, dtype=np.int64),
        counts=np.asarray(counts, dtype=np.int64),
        diag_counts=np.asarray(diag_counts, dtype=np.int64),
        endpoint_counts=np.asarray(endpoints, dtype=np.int64),
        group_diag=np.asarray(TOTALS_1MB["diag"], dtype=np.int64),
        group_cis_offdiag=np.asarray(TOTALS_1MB["cis_offdiag"], dtype=np.int64),
        group_inter=np.asarray(TOTALS_1MB["inter"], dtype=np.int64),
        count_arrays_sha256=np.asarray(_count_arrays_sha256(counts, diag_counts)),
    )


def _load_counts_snapshot(path: Path, template: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        names = tuple(str(x) for x in payload["chromosome_names"].tolist())
        lengths = tuple(int(x) for x in payload["chromosome_lengths"].tolist())
        bin_size = int(payload["bin_size"])
        counts = payload["counts"].copy()
        diag = payload["diag_counts"].copy()
        endpoints = payload["endpoint_counts"].copy()
    if names != tuple(template.chromosome_names) or lengths != tuple(int(x) for x in template.chromosome_lengths) or bin_size != FINAL_BIN:
        raise CalibrationError("count snapshot header does not match prepared template")
    if not np.issubdtype(counts.dtype, np.integer) or not np.issubdtype(diag.dtype, np.integer):
        raise CalibrationError("count snapshot is not integer")
    expected_endpoints = endpoint_counts_from_aggregates(template, counts, diag)
    if not np.array_equal(endpoints, expected_endpoints):
        raise CalibrationError("count snapshot endpoint audit failed")
    return counts, diag, endpoints


def _write_truth(path: Path, coordinates: np.ndarray, exposure: np.ndarray,
                 metadata: Mapping[str, Any]) -> dict[str, Any]:
    _save_truth(path, coordinates, metadata, exposure)
    return {
        "npz": _relative(path, path.parents[1]),
        "npz_sha256": sha256_file(path),
        "metadata_json": _relative(Path(str(path) + ".json"), path.parents[1]),
        "metadata_sha256": sha256_file(Path(str(path) + ".json")),
    }


def _worker_manifest_isolated(manifest: Mapping[str, Any]) -> None:
    serialized = json.dumps(_jsonable(manifest), sort_keys=True).lower()
    forbidden_tokens = ("eval_truth", "truth_path", "truth_file", "truth_npz")
    found = [token for token in forbidden_tokens if token in serialized]
    if found:
        raise CalibrationError("worker manifest exposes forbidden truth path token(s): %s" % found)


def _build_worker_manifest(run_dir: Path, config_sha256: str,
                           fixture_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    runs = []
    starts = {}
    for fixture_id in FIXTURES:
        record = fixture_records[fixture_id]
        start_path = Path(record["start_path"])
        starts[fixture_id] = {
            "start_id": fixture_id,
            "path": _relative(start_path, run_dir),
            "coordinate_sha256": record["start_coordinate_sha256"],
            "p_init": P_INIT,
            "q_init": q_from_p(P_INIT),
            "coordinate_shape": [2, FULL_FINAL_LOCI, 3],
            "max_radius": record["start_max_radius"],
        }
        for model_id in MODEL_IDS:
            layer_path = Path(record["layer_paths"][model_id])
            runs.append({
                "fixture_id": fixture_id,
                "model_id": model_id,
                "data_path": _relative(layer_path, run_dir),
                "data_sha256": record["layer_audits"][model_id]["sha256"],
                "start_path": _relative(start_path, run_dir),
                "candidate_path": str(Path("results") / "candidates" / (fixture_id + "_" + model_id.replace("-", "_") + ".npz")),
                "fit_config": _fit_config_dict(),
                "fit_status": "not_started",
                "initial_coordinate_sha256": record["start_coordinate_sha256"],
            })
    manifest = {
        "schema": PREPARED_SCHEMA,
        "run_id": run_dir.name,
        "status": "prepared_no_optimizer",
        "optimizer_started": False,
        "planned_fit_count": len(runs),
        "model_ids": list(MODEL_IDS),
        "fixture_ids": list(FIXTURES),
        "config_sha256": config_sha256,
        "start_files": starts,
        "runs": runs,
        "worker_contract": {
            "inputs": ["one unlabeled synthetic integer layer", "one shared physical start", "one model_id", "one explicit FitConfig"],
            "truth_access": "none",
            "optimizer_call_in_prepare": False,
            "all_five_variants_share_start_hash_within_fixture": True,
            "candidate_hash_required_before_evaluation": True,
        },
    }
    _worker_manifest_isolated(manifest)
    return manifest


def _validate_templates(templates: Mapping[int, Any]) -> Any:
    if FINAL_BIN not in templates:
        raise CalibrationError("1 Mb template missing")
    template = templates[FINAL_BIN]
    if len(template.chromosome_names) != N_CHROMOSOMES:
        raise CalibrationError("expected 20 chromosome headers")
    if template.n_loci != FULL_FINAL_LOCI:
        raise CalibrationError("expected 2645 full-grid loci")
    if template.bin_size != FINAL_BIN:
        raise CalibrationError("expected 1 Mb final template")
    if np.any(template.locus_bin < 0) or np.any(template.locus_bin >= template.n_bins[template.locus_chromosome]):
        raise CalibrationError("template local bins are outside their chromosome")
    if any(int(template.locus_bin[template.chromosome_slice(c).start]) != 0 for c in range(N_CHROMOSOMES)):
        raise CalibrationError("1 Mb grid does not start at origin zero per chromosome")
    return template


def _source_manifest(run_dir: Path, config_sha256: str, template: Any) -> dict[str, Any]:
    relatives = (
        "pr/allele_calibration.py",
        "pr/v1_calibration.py",
        "pr/contact_model.py",
        "pr/genome.py",
        "pr/paths.py",
        "pr/score.py",
        "pr/allele_models.py",
        "pr/paired_run.py",
    )
    code = {}
    for relative in relatives:
        path = ROOT / relative
        if not path.is_file():
            raise CalibrationError("missing source file: %s" % relative)
        code[relative] = sha256_file(path)
    return {
        "schema": "p9016-r2-allele-calibration-source-manifest-v1",
        "config_sha256": config_sha256,
        "header_source": "inputs/P9016.snpfree.pairs.gz",
        "header_source_sha256": sha256_file(ROOT / "inputs/P9016.snpfree.pairs.gz"),
        "allowed_header_summary": {
            "chromosome_names": list(template.chromosome_names),
            "chromosome_lengths": [int(x) for x in template.chromosome_lengths],
            "n_chromosomes": len(template.chromosome_names),
            "n_loci_1mb": template.n_loci,
        },
        "code_sha256": code,
        "source_read_policy": "header names/lengths only; no phase/reference/production-coordinate values",
    }


def _write_readme(run_dir: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# 026 R2 Synthetic Allele Calibration Prepare",
        "",
        "状态：`prepared_no_optimizer`。本目录只完成 N1/N2/P1/P2 的 synthetic data、独立 start、隔离 truth 和 4x5 run plan；没有调用 optimizer，没有 worker fit，没有 R1/R3。",
        "",
        "## 冻结设计",
        "",
        "- 1 Mb from-zero full header grid：20 chromosomes、40 tracks、2645 loci、5290 physical beads。所有 eligible non-diagonal pairs 都进入 rate/count grid；same-bin diag 是独立 multinomial nuisance。",
        "- 生成 kernel：V1，`epsilon=1e-6`，`r0=2*l0`，`p_gen=0.8`。每个 fixture 的 diag/cis-offdiag/inter 总数固定为 438774/696680/568434。",
        "- N1/N2 是 internal-shape 相同但 center 分离的 negative fixtures；P1/P2 是两个 internal shape 不同的 positive fixtures。N1/P1 用 ones exposure；N2/P2 用 sigma=.4、无 dropout、mean-one exposure。",
        "- C0/C1/C2-map/C2-free 的 fit exposure 都从 synthetic observed endpoints 重新计算为 `sqrt(endpoint+10)/mean`；C3 只用 ones。generation exposure 不传给 fit。",
        "- 每个 fixture 的五个 variant 共用同一个独立 physical x0；x0 来自独立 `generate_truth(..., same_shape=False)`，再做一次全局 center/scale 到 max radius .8，`p0=.75`。",
        "- 固定 future budget：`maxiter=80, maxfun=270, maxls=20, ftol=1e-10, gtol=1e-6`，checkpoint every 20；未来最多 4 workers，BLAS/OMP=1。",
        "",
        "## Prepare gates",
        "",
        "prepare 的 20 个 B paired plans 均为 `fit_not_run=true`；B 的 `data_for_model` 视图和每个 layer 的 counts/endpoint/exposure 均已做 roundtrip 校验。fixture gate 失败时保留原 seed/artifact，不替换 seed。",
        "",
        "| fixture | truth gate | count zeros/nonzeros (diag/cis/inter) | shared x0 max radius |",
        "|---|---|---|---:|",
    ]
    for fixture_id in FIXTURES:
        record = summary["fixtures"][fixture_id]
        audit = record["count_audit"]["by_group"]
        gate = record["geometry_gate"]
        zeros_nonzeros = "/".join(
            "%d/%d" % (audit[group]["zero"], audit[group]["nonzero"])
            for group in ("diag", "cis_offdiag", "inter")
        )
        lines.append("| %s | %s | %s | %.17g |" % (
            fixture_id, "PASS" if gate["passed"] else "INVALID_RETAINED", zeros_nonzeros, record["start_max_radius"],
        ))
    lines.extend([
        "",
        "当前结果：`planned_fit_count=20`，`actual_fit_count=0`，`optimizer_started=false`。N/P truth-distance rho 只记录，不用于选 seed 或模型。",
        "",
        "## Worker-facing artifact boundary",
        "",
        "- [`work/prepared_manifest.json`](work/prepared_manifest.json) 只列无标签 integer layer、共享 start、model id、candidate 输出位置和 frozen FitConfig；不包含 truth path。",
        "- [`results/prepare_summary.json`](results/prepare_summary.json) 记录 counts/grid/gates/plan 和 eval-only artifact 索引。",
        "- [`provenance/truth_manifest.json`](provenance/truth_manifest.json) 属于 evaluator side；candidate 文件 hash 必须全部通过后才能打开其中的 truth。",
        "- [`provenance/source_manifest.json`](provenance/source_manifest.json) 记录 header/code hash；没有读取真实 reference、phase、production coordinates 或真实 contacts 作为观测。",
        "",
        "## Future evaluation adapter",
        "",
        "`python -m pr.allele_calibration evaluate --run-dir <this-run> --candidate-manifest results/candidate_manifest.json` 会先检查 20 个 candidate 文件 hash、full-grid shape、finite 值和 model physical domain，再加载 truth。它按 fixture 做一次 overall candidate swap，不做 local chromosome repair；N 的 truth distance matrix 用两 copy 的 mean，N-only copy difference RMS 只作诊断。",
        "",
        "主要产物：`config.json`、`work/prepared_manifest.json`、4 个 counts snapshot、20 个 variant layers、4 个 shared starts、4 个隔离 truth、`results/prepare_summary.json`。",
        "",
    ])
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_run(run_dir: str | Path) -> dict[str, Any]:
    """Prepare all synthetic artifacts and run only non-optimizer validation gates."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("work", "eval_truth", "results", "logs", "provenance"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists():
        raise CalibrationError("prepare refuses an existing config in a non-fresh run: %s" % config_path)
    config = _build_config(run_dir)
    _write_json(config_path, config)
    config_sha256 = sha256_file(config_path)

    templates = header_templates()
    template = _validate_templates(templates)
    source_manifest = _source_manifest(run_dir, config_sha256, template)
    _write_json(run_dir / "provenance" / "source_manifest.json", source_manifest)

    fixture_records: dict[str, dict[str, Any]] = {}
    truth_records: dict[str, Any] = {}
    gate_records: dict[str, Any] = {}
    interface_records: dict[str, Any] = {}

    for fixture_id, fixture in FIXTURES.items():
        # Truth and initialization use separate frozen seed streams. The initialization is
        # generated independently and then globally centered/scaled; it never reads truth.
        truth, truth_generation_metadata = generate_truth(
            template, int(fixture["truth_seed"]), same_shape=bool(fixture["same_shape"])
        )
        assert_inside_unit_ball(truth)
        independent_start, normalization_metadata = _normalize_initialization(
            generate_truth(template, int(fixture["init_seed"]), same_shape=False)[0]
        )
        assert_inside_unit_ball(independent_start)
        start_metadata = {
            "fixture_id": fixture_id,
            "generator": "generate_truth(template, init_seed, same_shape=False)",
            "truth_independent": True,
            "init_seed": int(fixture["init_seed"]),
            "normalization": normalization_metadata,
            "shared_by_model_ids": list(MODEL_IDS),
        }
        start = PairedStart(
            start_id=fixture_id,
            coordinates=independent_start,
            p_init=P_INIT,
            metadata=start_metadata,
        )
        start_path = run_dir / "work" / (fixture_id + "_shared_start.npz")
        save_paired_start(start_path, start)
        loaded_start = load_paired_start(start_path)
        if loaded_start.coordinate_sha256 != start.coordinate_sha256 or not np.array_equal(loaded_start.coordinates, start.coordinates):
            raise CalibrationError("shared initialization roundtrip failed for %s" % fixture_id)

        generation_exposure, generation_exposure_metadata = synthetic_exposure(
            template.n_loci, float(fixture["exposure_sigma"]), int(fixture["exposure_seed"]), capture_drop=False
        )
        if fixture["exposure_sigma"] == 0.0 and not np.array_equal(generation_exposure, np.ones(template.n_loci)):
            raise CalibrationError("sigma=0 exposure is not exactly ones for %s" % fixture_id)
        rates = generation_rates(template, truth, generation_exposure, P_GEN, "v1")
        counts, diag_counts = sample_conditional_counts(
            template, rates, generation_exposure, int(fixture["draw_seed"]), TOTALS_1MB
        )
        if counts.dtype.kind not in "iu" or diag_counts.dtype.kind not in "iu":
            raise CalibrationError("synthetic counts are not integer for %s" % fixture_id)
        endpoints = endpoint_counts_from_aggregates(template, counts, diag_counts)
        count_audit = _count_audit(template, counts, diag_counts)
        if not count_audit["actual_totals_match_frozen"]:
            raise CalibrationError("frozen count totals failed for %s" % fixture_id)
        counts_path = run_dir / "work" / (fixture_id + "_counts_snapshot.npz")
        _save_counts_snapshot(counts_path, template, counts, diag_counts, endpoints, count_audit)
        roundtrip_counts, roundtrip_diag, roundtrip_endpoints = _load_counts_snapshot(counts_path, template)
        if not np.array_equal(roundtrip_counts, counts) or not np.array_equal(roundtrip_diag, diag_counts) or not np.array_equal(roundtrip_endpoints, endpoints):
            raise CalibrationError("count snapshot roundtrip failed for %s" % fixture_id)

        observed_exposure, observed_endpoints = observed_endpoint_exposure(template, counts, diag_counts)
        if not np.array_equal(observed_endpoints, endpoints):
            raise CalibrationError("observed endpoint exposure used a different endpoint audit for %s" % fixture_id)
        ones_exposure = np.ones(template.n_loci, dtype=np.float64)
        ones_exposure /= ones_exposure.mean()
        if not np.all(np.isfinite(observed_exposure)) or not np.all(observed_exposure > 0.0):
            raise CalibrationError("observed fit exposure is invalid for %s" % fixture_id)

        gate = _truth_geometry_gate(template, truth, fixture)
        gate_records[fixture_id] = gate
        truth_metadata = dict(truth_generation_metadata)
        truth_metadata.update({
            "schema": TRUTH_SCHEMA,
            "fixture_id": fixture_id,
            "truth_kind": fixture["truth_kind"],
            "truth_seed": int(fixture["truth_seed"]),
            "draw_seed": int(fixture["draw_seed"]),
            "exposure_seed": int(fixture["exposure_seed"]),
            "generation_kernel": "v1",
            "generation_epsilon": EPSILON,
            "generation_r0": "2*l0",
            "p_gen": P_GEN,
            "generation_exposure": generation_exposure_metadata,
            "grid": {"origin_bp": 0, "bin_size_bp": FINAL_BIN, "n_loci": template.n_loci},
            "geometry_gate": gate,
            "truth_distance_rho_recorded_not_tuned": True,
            "truth_distance_matrix_policy_for_N": "common truth matrix mean at evaluation",
        })
        truth_path = run_dir / "eval_truth" / (fixture_id + "_truth_1mb.npz")
        truth_record = _write_truth(truth_path, truth, generation_exposure, truth_metadata)
        truth_record.update({
            "fixture_id": fixture_id,
            "generation_exposure_sha256": _array_sha256(generation_exposure, "<f8"),
            "coordinate_array_sha256": _array_sha256(truth, "<f8"),
            "truth_seed": int(fixture["truth_seed"]),
        })
        truth_records[fixture_id] = truth_record

        layer_paths: dict[str, str] = {}
        layer_audits: dict[str, Any] = {}
        for model_id in MODEL_IDS:
            if model_id == "C3":
                fit_exposure = ones_exposure
                exposure_mode = "uniform_ones"
                fit_exposure_rule = "ones"
            else:
                fit_exposure = observed_exposure
                exposure_mode = "observed_endpoint_synthetic_recomputed"
                fit_exposure_rule = "sqrt(observed_endpoint_count+10)/mean"
            data = synthetic_integer_clone(
                template,
                counts,
                diag_counts,
                fit_exposure,
                TOTALS_1MB,
                exposure_mode=exposure_mode,
                endpoint_counts=endpoints,
            )
            layer_path = run_dir / "work" / (fixture_id + "_" + model_id.replace("-", "_") + "_1mb_layer.npz")
            save_layer(layer_path, data)
            loaded = load_layer(layer_path)
            if not np.array_equal(loaded.counts, counts) or not np.array_equal(loaded.diag_counts, diag_counts):
                raise CalibrationError("variant layer changed counts for %s/%s" % (fixture_id, model_id))
            if not np.array_equal(loaded.endpoint_counts, endpoints) or not np.array_equal(loaded.exposure, fit_exposure):
                raise CalibrationError("variant layer failed endpoint/exposure roundtrip for %s/%s" % (fixture_id, model_id))
            loaded.assert_consistent()
            layer_paths[model_id] = str(layer_path)
            layer_audits[model_id] = {
                "path": _relative(layer_path, run_dir),
                "sha256": sha256_file(layer_path),
                "count_mode": loaded.count_mode,
                "exposure_mode": loaded.exposure_mode,
                "fit_exposure_rule": fit_exposure_rule,
                "fit_exposure_sha256": _array_sha256(fit_exposure, "<f8"),
                "group_totals": {key: int(value) for key, value in TOTALS_1MB.items()},
            }

        # Exercise B's unified model/data contract without invoking run_one_fit.
        base_data = load_layer(layer_paths["C0"])
        starts_for_plan = [loaded_start]
        planned = plan_paired_runs(base_data, starts_for_plan, model_ids=MODEL_IDS)
        if len(planned) != len(MODEL_IDS) or any(not row["fit_not_run"] for row in planned):
            raise CalibrationError("paired plan unexpectedly evaluated a fit for %s" % fixture_id)
        view_checks = {}
        for model_id in MODEL_IDS:
            model_view = data_for_model(base_data, model_id)
            model_view.assert_consistent()
            layer = load_layer(layer_paths[model_id])
            if not np.array_equal(model_view.counts, layer.counts) or not np.array_equal(model_view.diag_counts, layer.diag_counts):
                raise CalibrationError("B data_for_model changed counts for %s/%s" % (fixture_id, model_id))
            if not np.array_equal(model_view.exposure, layer.exposure):
                raise CalibrationError("B data_for_model exposure differs from prepared layer for %s/%s" % (fixture_id, model_id))
            view_checks[model_id] = True
        interface_records[fixture_id] = {
            "plan_rows": planned,
            "planned_fit_count": len(planned),
            "data_for_model_views_passed": view_checks,
            "run_one_fit_called": False,
        }
        fixture_records[fixture_id] = {
            "fixture_id": fixture_id,
            "truth_kind": fixture["truth_kind"],
            "count_snapshot_path": str(counts_path),
            "count_snapshot_sha256": sha256_file(counts_path),
            "count_arrays_sha256": _count_arrays_sha256(counts, diag_counts),
            "count_audit": count_audit,
            "observed_endpoint_exposure_sha256": _array_sha256(observed_exposure, "<f8"),
            "ones_exposure_sha256": _array_sha256(ones_exposure, "<f8"),
            "start_path": str(start_path),
            "start_coordinate_sha256": start.coordinate_sha256,
            "start_max_radius": start.max_radius,
            "start_normalization": normalization_metadata,
            "layer_paths": layer_paths,
            "layer_audits": layer_audits,
            "geometry_gate": gate,
            "interface_plan": interface_records[fixture_id],
        }

    worker_manifest = _build_worker_manifest(run_dir, config_sha256, fixture_records)
    worker_manifest_path = run_dir / "work" / "prepared_manifest.json"
    _write_json(worker_manifest_path, worker_manifest)
    _worker_manifest_isolated(worker_manifest)

    truth_manifest = {
        "schema": "p9016-r2-allele-calibration-truth-manifest-v1",
        "run_id": run_dir.name,
        "status": "isolated_for_evaluation_only",
        "worker_manifest_path": _relative(worker_manifest_path, run_dir),
        "candidate_hash_gate_required_before_open": True,
        "fixtures": truth_records,
    }
    truth_manifest_path = run_dir / "provenance" / "truth_manifest.json"
    _write_json(truth_manifest_path, truth_manifest)

    fixture_gate_pass = {fixture_id: bool(record["passed"]) for fixture_id, record in gate_records.items()}
    summary = {
        "schema": "p9016-r2-allele-calibration-prepare-summary-v1",
        "run_id": run_dir.name,
        "status": "prepared_no_optimizer",
        "optimizer_started": False,
        "fit_started": False,
        "planned_fit_count": len(FIXTURES) * len(MODEL_IDS),
        "actual_fit_count": 0,
        "templates": {
            "chromosomes": list(template.chromosome_names),
            "chromosome_lengths": [int(x) for x in template.chromosome_lengths],
            "n_loci": template.n_loci,
            "n_tracks": len(template.track_specs),
            "n_pairs": template.n_pairs,
            "origin_bp": 0,
            "bin_size_bp": FINAL_BIN,
        },
        "config_sha256": config_sha256,
        "source_manifest_sha256": sha256_file(run_dir / "provenance" / "source_manifest.json"),
        "worker_manifest": _relative(worker_manifest_path, run_dir),
        "truth_manifest": _relative(truth_manifest_path, run_dir),
        "fixture_gate_pass": fixture_gate_pass,
        "all_fixture_gates_pass": all(fixture_gate_pass.values()),
        "invalid_fixture_policy": "retain and report; no replacement seeds",
        "fixtures": fixture_records,
        "truth_records": truth_records,
        "interface_records": interface_records,
        "provenance": {
            "header_only": True,
            "real_reference_read": False,
            "real_phase_read": False,
            "real_contacts_read_as_observations": False,
            "truth_coordinates_exposed_to_worker": False,
        },
    }
    _write_json(run_dir / "results" / "prepare_summary.json", summary)
    _write_readme(run_dir, summary)
    return summary


def load_prepared_data(run_dir: str | Path, fixture_id: str, model_id: str) -> tuple[Any, PairedStart, dict[str, Any]]:
    """Load one worker-facing layer/start pair and verify its prepared hashes."""
    run_dir = Path(run_dir).resolve()
    manifest = _read_json(run_dir / "work" / "prepared_manifest.json")
    if manifest.get("schema") != PREPARED_SCHEMA or manifest.get("optimizer_started"):
        raise CalibrationError("prepared manifest is not available for worker input")
    matching = [
        row for row in manifest.get("runs", [])
        if row.get("fixture_id") == str(fixture_id) and row.get("model_id") == str(model_id)
    ]
    if len(matching) != 1:
        raise CalibrationError("prepared worker run is not unique for %s/%s" % (fixture_id, model_id))
    row = matching[0]
    data_path = _resolve_inside(row["data_path"], run_dir)
    start_path = _resolve_inside(row["start_path"], run_dir)
    if sha256_file(data_path) != row.get("data_sha256"):
        raise CalibrationError("prepared data hash changed for %s/%s" % (fixture_id, model_id))
    start = load_paired_start(start_path)
    if start.coordinate_sha256 != row.get("initial_coordinate_sha256"):
        raise CalibrationError("prepared start hash changed for %s/%s" % (fixture_id, model_id))
    data = load_layer(data_path)
    data.assert_consistent()
    return data, start, dict(row)


def _resolve_inside(path: str | Path, run_dir: Path) -> Path:
    candidate = (run_dir / Path(path)).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise CalibrationError("artifact path escapes run directory: %s" % path) from exc
    return candidate


def _candidate_file_coordinates(path: Path, template: Any) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            key = "coordinates" if "coordinates" in payload else ("coords" if "coords" in payload else None)
            if key is None:
                raise CalibrationError("candidate NPZ lacks coordinates/coords: %s" % path)
            return np.asarray(payload[key], dtype=np.float64).copy()
    by_track = {spec.name: spec for spec in template.track_specs}
    coordinates = np.full((2, template.n_loci, 3), np.nan, dtype=np.float64)
    seen = np.zeros((2, template.n_loci), dtype=bool)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                raise CalibrationError("candidate coordinate row has fewer than 5 fields at %s:%d" % (path, line_number))
            name = fields[0]
            if name not in by_track:
                raise CalibrationError("unknown candidate track %r at %s:%d" % (name, path, line_number))
            spec = by_track[name]
            try:
                position = int(fields[1])
                xyz = np.asarray([float(fields[2]), float(fields[3]), float(fields[4])], dtype=np.float64)
            except ValueError as exc:
                raise CalibrationError("invalid candidate numeric row at %s:%d" % (path, line_number)) from exc
            if position % FINAL_BIN != 0:
                raise CalibrationError("candidate position is not a 1 Mb origin-aligned bin at %s:%d" % (path, line_number))
            local_bin = position // FINAL_BIN
            chromosome_slice = template.chromosome_slice(spec.chromosome_index)
            if local_bin < 0 or local_bin >= chromosome_slice.stop - chromosome_slice.start:
                raise CalibrationError("candidate bin outside header grid at %s:%d" % (path, line_number))
            index = chromosome_slice.start + local_bin
            if seen[spec.copy_index, index]:
                raise CalibrationError("duplicate candidate coordinate at %s:%d" % (path, line_number))
            coordinates[spec.copy_index, index] = xyz
            seen[spec.copy_index, index] = True
    if not np.all(seen):
        raise CalibrationError("candidate text does not cover the full 40-track grid: %s" % path)
    return coordinates


def _validate_candidate_manifest(run_dir: Path, manifest: Mapping[str, Any], template: Any) -> list[dict[str, Any]]:
    if manifest.get("schema") != CANDIDATE_SCHEMA:
        raise CalibrationError("candidate manifest schema mismatch")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(FIXTURES) * len(MODEL_IDS):
        raise CalibrationError("candidate manifest must contain exactly 20 candidates")
    seen = set()
    validated = []
    # Hash and basic candidate validation are deliberately completed before truth loading.
    for record in candidates:
        if not isinstance(record, Mapping):
            raise CalibrationError("candidate record is not an object")
        fixture_id = str(record.get("fixture_id", ""))
        model_id = str(record.get("model_id", ""))
        key = (fixture_id, model_id)
        if fixture_id not in FIXTURES or model_id not in MODEL_IDS or key in seen:
            raise CalibrationError("invalid or duplicate candidate key: %s" % (key,))
        seen.add(key)
        relative = record.get("coordinate_path")
        expected_hash = str(record.get("coordinate_sha256", ""))
        if not relative or len(expected_hash) != 64:
            raise CalibrationError("candidate path/hash missing for %s" % (key,))
        path = _resolve_inside(str(relative), run_dir)
        if not path.is_file():
            raise CalibrationError("candidate file missing: %s" % path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise CalibrationError("candidate hash mismatch before truth access: %s" % path)
        coordinates = _candidate_file_coordinates(path, template)
        if coordinates.shape != (2, template.n_loci, 3) or not np.all(np.isfinite(coordinates)):
            raise CalibrationError("candidate coordinates are not finite/full-grid: %s" % path)
        validate_physical_for_model(model_id, coordinates)
        p = float(record.get("p", P_INIT))
        if not math.isfinite(p):
            raise CalibrationError("candidate p is nonfinite: %s" % key)
        q_from_p(p)
        validated.append({
            "fixture_id": fixture_id,
            "model_id": model_id,
            "path": path,
            "file_sha256": actual_hash,
            "coordinates": coordinates,
            "p": p,
            "fit_metadata": dict(record),
        })
    if seen != {(fixture_id, model_id) for fixture_id in FIXTURES for model_id in MODEL_IDS}:
        raise CalibrationError("candidate manifest does not cover all 4x5 fixture/model pairs")
    return validated


def _assigned_distance_metrics(template: Any, candidate: np.ndarray, truth: np.ndarray,
                               same_shape: bool) -> dict[str, Any]:
    candidate_a: list[np.ndarray] = []
    candidate_b: list[np.ndarray] = []
    truth_a: list[np.ndarray] = []
    truth_b: list[np.ndarray] = []
    per_chromosome_raw = []
    for chromosome, name in enumerate(template.chromosome_names):
        chromosome_slice = template.chromosome_slice(chromosome)
        ca, cb = _distance_vector(candidate, chromosome_slice)
        ta, tb = _distance_vector(truth, chromosome_slice)
        candidate_a.append(ca)
        candidate_b.append(cb)
        truth_a.append(ta)
        truth_b.append(tb)
        if same_shape:
            tm = (ta + tb) / 2.0
            ta_eval, tb_eval = tm, tm
        else:
            ta_eval, tb_eval = ta, tb
        per_chromosome_raw.append({
            "chromosome": name,
            "rho_A_mat": _bounded_spearman(ca, ta_eval),
            "rho_A_pat": _bounded_spearman(ca, tb_eval),
            "rho_B_mat": _bounded_spearman(cb, ta_eval),
            "rho_B_pat": _bounded_spearman(cb, tb_eval),
        })

    pooled_ca = np.concatenate(candidate_a)
    pooled_cb = np.concatenate(candidate_b)
    pooled_ta = np.concatenate(truth_a)
    pooled_tb = np.concatenate(truth_b)
    if same_shape:
        pooled_tm = (pooled_ta + pooled_tb) / 2.0
        pooled_ta_eval, pooled_tb_eval = pooled_tm, pooled_tm
    else:
        pooled_ta_eval, pooled_tb_eval = pooled_ta, pooled_tb
    raw_pooled = {
        "rho_A_mat": _bounded_spearman(pooled_ca, pooled_ta_eval),
        "rho_A_pat": _bounded_spearman(pooled_ca, pooled_tb_eval),
        "rho_B_mat": _bounded_spearman(pooled_cb, pooled_ta_eval),
        "rho_B_pat": _bounded_spearman(pooled_cb, pooled_tb_eval),
    }
    direct = (raw_pooled["rho_A_mat"] + raw_pooled["rho_B_pat"]) / 2.0
    cross = (raw_pooled["rho_A_pat"] + raw_pooled["rho_B_mat"]) / 2.0
    if direct >= cross:
        orientation = "direct"
        assigned_a, assigned_b = pooled_ta_eval, pooled_tb_eval
        selected_direct = direct
        selected_cross = cross
    else:
        orientation = "swapped"
        assigned_a, assigned_b = pooled_tb_eval, pooled_ta_eval
        selected_direct = cross
        selected_cross = direct
    contrast = selected_direct - selected_cross
    if orientation == "direct":
        a, b, c, d = (raw_pooled["rho_A_mat"], raw_pooled["rho_A_pat"], raw_pooled["rho_B_mat"], raw_pooled["rho_B_pat"])
    else:
        a, b, c, d = (raw_pooled["rho_B_mat"], raw_pooled["rho_B_pat"], raw_pooled["rho_A_mat"], raw_pooled["rho_A_pat"])
    margin_ref1 = a - b
    margin_ref2 = d - c
    candidate_pooled = np.concatenate(candidate_a + candidate_b)
    candidate_scale = max(
        float(np.sqrt(np.mean(candidate_pooled ** 2))),
        np.finfo(np.float64).tiny,
    )
    truth_pooled = np.concatenate([assigned_a, assigned_b])
    truth_scale = max(
        float(np.sqrt(np.mean(truth_pooled ** 2))),
        np.finfo(np.float64).tiny,
    )
    if orientation == "direct":
        assigned_candidate_a, assigned_candidate_b = pooled_ca, pooled_cb
    else:
        assigned_candidate_a, assigned_candidate_b = pooled_cb, pooled_ca
    err_a = float(np.sqrt(np.mean((assigned_candidate_a / candidate_scale - assigned_a / truth_scale) ** 2)))
    err_b = float(np.sqrt(np.mean((assigned_candidate_b / candidate_scale - assigned_b / truth_scale) ** 2)))
    n_only_rms = None
    if same_shape:
        n_only_rms = float(np.sqrt(np.mean(((pooled_ca - pooled_cb) / candidate_scale) ** 2)))
    return {
        "overall_orientation": orientation,
        "overall_direct_score": direct,
        "overall_cross_score": cross,
        "overall_matched": selected_direct,
        "overall_cross": selected_cross,
        "overall_contrast": contrast,
        "overall_fixed_ref_abcd": [a, b, c, d],
        "overall_margin_ref1_mat": margin_ref1,
        "overall_margin_ref2_pat": margin_ref2,
        "raw_pooled_four_rho": raw_pooled,
        "per_chromosome_four_rho": per_chromosome_raw,
        "pooled_candidate_offdiag_rms": candidate_scale,
        "pooled_truth_offdiag_rms": truth_scale,
        "shape_error": {
            "copy_A_or_assigned_first": err_a,
            "copy_B_or_assigned_second": err_b,
            "both_copy_mean": (err_a + err_b) / 2.0,
            "both_copy_max": max(err_a, err_b),
        },
        "N_only_copy_difference_rms": n_only_rms,
        "N_only_copy_difference_is_not_a_positive_score": True,
        "truth_matrix_policy": "common mean for N, original mat/pat for P",
    }


def evaluate_run(run_dir: str | Path, candidate_manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Evaluate 20 candidates after hashes are verified, then load isolated truth."""
    run_dir = Path(run_dir).resolve()
    prepared_path = run_dir / "work" / "prepared_manifest.json"
    prepared = _read_json(prepared_path)
    if prepared.get("schema") != PREPARED_SCHEMA or prepared.get("optimizer_started"):
        raise CalibrationError("prepared manifest is not a clean no-optimizer manifest")
    candidate_path = Path(candidate_manifest_path) if candidate_manifest_path is not None else run_dir / "results" / "candidate_manifest.json"
    candidate_manifest = _read_json(candidate_path)
    templates = header_templates()
    template = _validate_templates(templates)
    validated_candidates = _validate_candidate_manifest(run_dir, candidate_manifest, template)
    # No truth path is opened above. Only after every candidate file hash and domain gate passes:
    truth_manifest = _read_json(run_dir / "provenance" / "truth_manifest.json")
    truth_records = truth_manifest.get("fixtures", {})
    results = []
    for candidate in validated_candidates:
        fixture_id = candidate["fixture_id"]
        truth_record = truth_records.get(fixture_id)
        if not isinstance(truth_record, Mapping):
            raise CalibrationError("truth manifest lacks fixture %s" % fixture_id)
        truth_path = _resolve_inside(str(truth_record["npz"]), run_dir)
        expected_truth_hash = str(truth_record["npz_sha256"])
        if sha256_file(truth_path) != expected_truth_hash:
            raise CalibrationError("isolated truth hash changed: %s" % truth_path)
        metadata_path = _resolve_inside(str(truth_record["metadata_json"]), run_dir)
        if sha256_file(metadata_path) != str(truth_record["metadata_sha256"]):
            raise CalibrationError("isolated truth metadata hash changed: %s" % metadata_path)
        truth, generation_exposure, truth_metadata = _load_truth(truth_path)
        if truth.shape != (2, template.n_loci, 3) or generation_exposure.shape != (template.n_loci,):
            raise CalibrationError("truth shape mismatch for %s" % fixture_id)
        metrics = _assigned_distance_metrics(
            template, candidate["coordinates"], truth, bool(FIXTURES[fixture_id]["same_shape"])
        )
        results.append({
            "fixture_id": fixture_id,
            "model_id": candidate["model_id"],
            "candidate_file_sha256": candidate["file_sha256"],
            "truth_file_sha256": expected_truth_hash,
            "truth_seed": truth_metadata.get("truth_seed"),
            "p": candidate["p"],
            "metrics": metrics,
        })
    output = {
        "schema": "p9016-r2-allele-calibration-evaluation-v1",
        "run_id": run_dir.name,
        "status": "complete",
        "candidate_hash_gate": "passed_before_truth_access",
        "candidate_count": len(results),
        "results": results,
        "interpretation": {
            "truth_is_evaluation_only": True,
            "N_only_copy_difference_not_a_positive_score": True,
            "truth_distance_rho_not_used_for_seed_or_model_selection": True,
            "no_biological_replicate_claim": True,
        },
    }
    _write_json(run_dir / "results" / "evaluation.json", output)
    return output


def prepare_and_write(run_dir: str | Path) -> dict[str, Any]:
    """CLI wrapper for the no-optimizer preparation stage."""
    return prepare_run(run_dir)


def _main() -> None:
    parser = argparse.ArgumentParser(description="R2 synthetic allele calibration prepare/evaluation adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--run-dir", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--run-dir", required=True)
    evaluate_parser.add_argument("--candidate-manifest", default=None)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_and_write(args.run_dir)
    else:
        result = evaluate_run(args.run_dir, args.candidate_manifest)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
