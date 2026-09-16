"""本轮 shared-capture formal run 的无 optimizer preparation。

只读取 SNP-free pairs、014 blind source 和纯 synthetic generator；不导入 evaluator，
不读取 phase/reference/evaluation output。所有 worker 输入与 eval_truth 分目录隔离。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from data_io import array_sha256, load_aggregate, save_aggregate, save_start, sha256_file, write_json  # noqa: E402
from pr import contact_model, reconstruction_init  # noqa: E402
from pr.v1_calibration import (  # noqa: E402
    EPSILON,
    FULL_FINAL_LOCI,
    N_CHROMOSOMES,
    N_TRACKS,
    generation_rates,
    generate_truth,
    header_templates,
    synthetic_exposure,
)

INPUT_PATH = (ROOT / "inputs/P9016.snpfree.pairs.gz").resolve()
KNOWN_MANIFEST = (ROOT / "docs/audits/next-step-r2-preparation-20260914T143656Z/synthetic_inputs/known_e_worker_input_manifest.json").resolve()
KNOWN_DIR = KNOWN_MANIFEST.parent
PREFLIGHT_ROOT = (ROOT / "test_res/039-20260914T164022Z-visibility-preflight").resolve()
APPROVED_014_ROOT = (ROOT / "test_res/014-20260912_153000-s0-genome-wide-fixed").resolve()

REAL_STAGES = (
    {"stage": "5Mb", "bin_size_bp": 5_000_000, "fg_cap": 612},
    {"stage": "2Mb", "bin_size_bp": 2_000_000, "fg_cap": 404},
    {"stage": "1Mb", "bin_size_bp": 1_000_000, "fg_cap": 486},
)
SYNTHETIC_STAGE = {"stage": "1Mb", "bin_size_bp": 1_000_000, "fg_cap": 486}
P2_N2 = {
    "P2": {
        "truth_seed": 260104, "blind_seed": 260404, "same_shape": False,
        "exposure_seed": 260304, "noise_seed": 450101,
        "truth_kind": "different_internal_shapes",
    },
    "N2": {
        "truth_seed": 260102, "blind_seed": 260402, "same_shape": True,
        "exposure_seed": 260302, "noise_seed": 450102,
        "truth_kind": "same_internal_shape_spatially_separated",
    },
}
P_GEN = 0.8
REAL_P_INIT = 0.75
P_INIT = 0.8
MAX_RADIUS = 0.8
NOISE_STD_FACTOR = 0.05
TOTAL_RECORDS = 1_703_888
TOTALS_REAL_1MB = {"diag": 438_774, "cis_offdiag": 696_680, "inter": 568_434}
N_OFF = TOTALS_REAL_1MB["cis_offdiag"] + TOTALS_REAL_1MB["inter"]
FG_TOTAL = 4 * sum(stage["fg_cap"] for stage in REAL_STAGES) + 10 * SYNTHETIC_STAGE["fg_cap"]
assert FG_TOTAL == 10_868


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
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite value cannot enter JSON")
        return float(value)
    return value


def _array_hash(values: np.ndarray) -> str:
    return array_sha256(np.asarray(values), "<f8")


def _configure_init() -> None:
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


def _write_event(event: dict[str, Any]) -> None:
    path = RUN / "logs" / "preparation.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(event), sort_keys=True, ensure_ascii=False) + "\n")


def _save_truth(path: Path, coordinates: np.ndarray, exposure: np.ndarray,
                metadata: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, coordinates=np.asarray(coordinates, dtype=np.float64),
                        generation_exposure=np.asarray(exposure, dtype=np.float64),
                        metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True, ensure_ascii=False)))
    metadata_path = Path(str(path) + ".json")
    write_json(metadata_path, metadata)
    return {
        "path": str(path.relative_to(RUN)), "sha256": sha256_file(path),
        "metadata_path": str(metadata_path.relative_to(RUN)),
        "metadata_sha256": sha256_file(metadata_path),
        "coordinate_sha256": _array_hash(coordinates),
        "exposure_sha256": _array_hash(exposure),
    }


def _save_expected(path: Path, counts: np.ndarray, diag: np.ndarray,
                   endpoints: np.ndarray, exposure: np.ndarray,
                   metadata: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, counts=np.asarray(counts, dtype=np.float64),
                        diag_counts=np.asarray(diag, dtype=np.float64),
                        endpoint_counts=np.asarray(endpoints, dtype=np.float64),
                        exposure=np.asarray(exposure, dtype=np.float64),
                        metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True, ensure_ascii=False)))
    return {
        "path": str(path.relative_to(RUN)), "sha256": sha256_file(path),
        "counts_sha256": _array_hash(counts), "diag_sha256": _array_hash(diag),
        "endpoint_sha256": _array_hash(endpoints), "exposure_sha256": _array_hash(exposure),
        "metadata": metadata,
    }


def _load_known_e(fixture: str) -> tuple[np.ndarray, dict[str, Any]]:
    with KNOWN_MANIFEST.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    spec = manifest["fixtures"][fixture]
    npy_record = spec["known_e_exposure_npy"]
    npy_path = (ROOT / npy_record["path"]).resolve()
    if sha256_file(npy_path) != npy_record["sha256"]:
        raise RuntimeError("known-e manifest hash mismatch: %s" % npy_path)
    values = np.asarray(np.load(npy_path, allow_pickle=False), dtype=np.float64)
    if values.shape != (FULL_FINAL_LOCI,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise RuntimeError("known-e vector has wrong shape/domain for %s" % fixture)
    if not np.isclose(float(values.mean()), 1.0, rtol=0.0, atol=1e-14):
        raise RuntimeError("known-e vector is not full-grid mean one for %s" % fixture)
    copied = RUN / "inputs" / (fixture + "_known_generating_e.npy")
    np.save(copied, values)
    copied_hash = sha256_file(copied)
    copied_values = np.asarray(np.load(copied, allow_pickle=False), dtype=np.float64)
    if not np.array_equal(copied_values, values):
        raise RuntimeError("copied known-e array changed for %s" % fixture)
    return values, {
        "source_manifest": str(KNOWN_MANIFEST.relative_to(ROOT)),
        "source_path": str(npy_path.relative_to(ROOT)),
        "source_sha256": npy_record["sha256"],
        "copied_path": str(copied.relative_to(RUN)),
        "copied_sha256": copied_hash,
        "array_sha256": _array_hash(values),
    }


def _array_bytes_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64, order="C").tobytes(order="C")).hexdigest()


def _make_blind_start(template: Any, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    blind, generation_metadata = generate_truth(template, int(seed), same_shape=False)
    center = blind.mean(axis=(0, 1))
    centered = blind - center
    pre_max = float(np.linalg.norm(centered, axis=2).max())
    if not math.isfinite(pre_max) or pre_max <= 0.0:
        raise RuntimeError("blind synthetic start has no positive radius")
    coordinates = centered * (MAX_RADIUS / pre_max)
    contact_model.assert_inside_unit_ball(coordinates)
    raw_y = contact_model.sphere_inverse(coordinates)
    return coordinates, raw_y, {
        "start_kind": "blind",
        "generator": "generate_truth(template, blind_seed, same_shape=False)",
        "blind_seed": int(seed), "truth_independent": True,
        "global_center": center.tolist(), "pre_scale_max_radius": pre_max,
        "post_scale_max_radius": float(np.linalg.norm(coordinates, axis=2).max()),
        "target_max_radius": MAX_RADIUS, "generator_metadata": generation_metadata,
    }


def _make_near_start(template: Any, truth: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    truth_y = contact_model.sphere_inverse(truth)
    rng = np.random.default_rng(int(seed))
    noise_std = NOISE_STD_FACTOR * float(template.l0)
    noise = rng.normal(0.0, noise_std, size=truth_y.shape)
    raw_y = truth_y + noise
    coordinates = contact_model.sphere_forward(raw_y)
    contact_model.assert_inside_unit_ball(coordinates)
    return coordinates, raw_y, {
        "start_kind": "near_oracle_informed_calibration",
        "generator": "sphere_forward(sphere_inverse(x_truth)+Normal(0,0.05*l0))",
        "noise_seed": int(seed), "noise_space": "raw-y",
        "noise_std": noise_std, "truth_not_a_worker_input": True,
        "initial_coordinate_rms_from_truth": float(np.sqrt(np.mean((coordinates - truth) ** 2))),
        "initial_raw_y_rms_noise": float(np.sqrt(np.mean(noise ** 2))),
    }


def _prepare_real_aggregates() -> tuple[dict[int, Any], dict[str, Any]]:
    result: dict[int, Any] = {}
    records: dict[str, Any] = {}
    expected = {
        5_000_000: {"raw_records": TOTAL_RECORDS, "raw_same_bin": 607_552,
                    "raw_cis_offdiag": 527_902, "raw_inter": 568_434},
        2_000_000: {"raw_records": TOTAL_RECORDS, "raw_same_bin": 516_046,
                    "raw_cis_offdiag": 619_408, "raw_inter": 568_434},
        1_000_000: {"raw_records": TOTAL_RECORDS, "raw_same_bin": 438_774,
                    "raw_cis_offdiag": 696_680, "raw_inter": 568_434},
    }
    for bin_size in (5_000_000, 2_000_000, 1_000_000):
        _write_event({"event": "real_aggregate_start", "bin_size_bp": bin_size})
        data = contact_model.load_frozen_p9016_aggregate(bin_size)
        budget = data.budget()
        exp = expected[bin_size]
        for key, value in exp.items():
            if budget[key] != value:
                raise RuntimeError("real %s budget mismatch: %s=%s expected %s" %
                                   (bin_size, key, budget[key], value))
        path = RUN / "inputs" / ("real_%d_aggregate.npz" % bin_size)
        record = save_aggregate(path, data)
        sidecar = path.with_suffix(".json")
        write_json(sidecar, record)
        record["sidecar"] = str(sidecar.relative_to(RUN))
        result[bin_size] = data
        records[str(bin_size)] = record
        _write_event({"event": "real_aggregate_end", "bin_size_bp": bin_size,
                      "budget": budget, "sha256": record["sha256"]})
    return result, records


def _prepare_real_initials(data_by_bin: dict[int, Any]) -> dict[str, Any]:
    _configure_init()
    result: dict[str, Any] = {}
    for candidate in ("consensus", "random"):
        first = data_by_bin[5_000_000]
        state = reconstruction_init.initialize_approved_candidate(
            candidate, tuple(first.chromosome_names), tuple(int(v) for v in first.chromosome_lengths), 5_000_000
        )
        current = {
            "coordinates": np.asarray(state["coords"], dtype=np.float64),
            "positions": np.asarray(state["positions"], dtype=np.int64),
            "chromosome_index": np.asarray(state["chromosome_index"], dtype=np.int32),
            "metadata": state["metadata"],
        }
        stages = {}
        for index, stage in enumerate(REAL_STAGES):
            if index > 0:
                data = data_by_bin[int(stage["bin_size_bp"])]
                current_state = reconstruction_init.warm_start_from_layer(
                    current["coordinates"], current["positions"], current["chromosome_index"],
                    tuple(data.chromosome_names), tuple(int(v) for v in data.chromosome_lengths),
                    int(stage["bin_size_bp"]), 1103 if candidate == "consensus" else 2207,
                )
                current = {
                    "coordinates": np.asarray(current_state["coords"], dtype=np.float64),
                    "positions": np.asarray(current_state["positions"], dtype=np.int64),
                    "chromosome_index": np.asarray(current_state["chromosome_index"], dtype=np.int32),
                    "metadata": current_state["metadata"],
                }
            coordinates = current["coordinates"]
            contact_model.assert_inside_unit_ball(coordinates)
            raw_y = contact_model.sphere_inverse(coordinates)
            path = RUN / "coords" / "initial" / ("real_%s_%s.npz" % (candidate, stage["stage"]))
            record = save_start(path, coordinates, raw_y, REAL_P_INIT, {
                "candidate": candidate, "stage": stage["stage"],
                "source": "014 approved blind root plus zero-optimization prolongation",
                "source_root": str(APPROVED_014_ROOT.relative_to(ROOT)),
                "no_optimization": True,
                "metadata": current["metadata"],
            })
            record["positions_sha256"] = array_sha256(
                np.broadcast_to(data_by_bin[int(stage["bin_size_bp"])].locus_bin * int(stage["bin_size_bp"]),
                                (2, data_by_bin[int(stage["bin_size_bp"])].n_loci)).copy(), "<i8"
            )
            record["chromosome_index_sha256"] = array_sha256(
                np.broadcast_to(data_by_bin[int(stage["bin_size_bp"])].locus_chromosome,
                                (2, data_by_bin[int(stage["bin_size_bp"])].n_loci)).copy(), "<i4"
            )
            stages[stage["stage"]] = record
        result[candidate] = {"stages": stages,
                            "source_sha256": state["metadata"]["source"]["source_sha256"]}
    if result["consensus"]["stages"]["5Mb"]["coordinate_sha256"] != result["consensus"]["stages"]["5Mb"]["coordinate_sha256"]:
        raise AssertionError("unreachable source-start hash assertion")
    return result


def _prepare_synthetic(template: Any, real_1mb: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    records: dict[str, Any] = {}
    truth_records: dict[str, Any] = {}
    for fixture, spec in P2_N2.items():
        _write_event({"event": "synthetic_start", "fixture": fixture})
        truth, truth_generation = generate_truth(template, int(spec["truth_seed"]), bool(spec["same_shape"]))
        contact_model.assert_inside_unit_ball(truth)
        known_e, known_meta = _load_known_e(fixture)
        generated_e, generated_e_meta = synthetic_exposure(template.n_loci, 0.4,
                                                            int(spec["exposure_seed"]), capture_drop=False)
        if not np.array_equal(known_e, generated_e):
            raise RuntimeError("known-e vector differs from pure generator for %s" % fixture)
        rates = generation_rates(template, truth, generated_e, P_GEN, "v1")
        counts = np.zeros(template.n_pairs, dtype=np.float64)
        off_mask = np.ones(template.n_pairs, dtype=bool)
        cis_mask = template.cis_pair
        denominator = float(rates[off_mask].sum())
        counts[:] = float(N_OFF) * rates / denominator
        # Counts use one global normalization only. Never force observed real cis/inter quotas.
        total_before_correction = float(counts.sum())
        if not np.isclose(total_before_correction, float(N_OFF), rtol=0.0, atol=1e-8):
            counts *= float(N_OFF) / total_before_correction
        positive_rates = rates > 0.0
        scale = counts[positive_rates] / rates[positive_rates]
        if float(np.ptp(scale)) > 1e-12 * max(1.0, float(np.max(np.abs(scale)))):
            raise RuntimeError("synthetic expected counts do not share one global rate scale")
        diag = np.asarray(real_1mb.diag_counts, dtype=np.float64).copy()
        endpoints = contact_model.endpoint_counts_from_aggregates(template, counts, diag)
        group_totals = {
            "diag": float(diag.sum()),
            "cis_offdiag": float(counts[cis_mask].sum()),
            "inter": float(counts[~cis_mask].sum()),
        }
        if not np.isclose(sum(group_totals.values()), float(TOTAL_RECORDS), rtol=0.0, atol=1e-8):
            raise RuntimeError("synthetic expected mass is not 1703888 for %s: %r" % (fixture, group_totals))
        expected_data = contact_model.synthetic_expected_clone(
            template, counts, diag, known_e, expected_group_totals=group_totals,
            exposure_mode="synthetic_known_generating_e", endpoint_counts=endpoints,
            rtol=1e-10, atol=1e-8,
        )
        data_path = RUN / "inputs" / (fixture + "_shared_expected_1Mb.npz")
        data_record = _save_expected(data_path, counts, diag, endpoints, known_e, {
            "fixture": fixture, "count_mode": "synthetic_expected",
            "generation_geometry": "pure_generate_truth_only",
            "generation_kernel": "v1", "epsilon": EPSILON, "r0": "2*l0",
            "p_gen": P_GEN, "group_totals": group_totals,
            "global_count_scale": float(N_OFF) / denominator,
            "group_scale_ratio_ptp": float(np.ptp(scale)),
            "group_quota_rescaling": False,
            "nraw_total": float(expected_data.raw_records),
            "diag_source": "real SNP-free 1Mb template diag vector; nuisance only",
            "no_integer_cast": True,
        })
        truth_path = RUN / "eval_truth" / (fixture + "_truth_1Mb.npz")
        truth_record = _save_truth(truth_path, truth, generated_e, {
            "schema": "p9016-shared-capture-truth-v1", "fixture": fixture,
            "truth_kind": spec["truth_kind"], "truth_seed": int(spec["truth_seed"]),
            "same_internal_shape": bool(spec["same_shape"]), "p_gen": P_GEN,
            "generation_kernel": "v1", "generation_epsilon": EPSILON, "generation_r0": "2*l0",
            "generation_exposure": generated_e_meta, "known_e_source": known_meta,
            "truth_generator_source": "pr/v1_calibration.py generate_truth; no reference",
            "truth_generator_source_sha256": sha256_file(ROOT / "pr/v1_calibration.py"),
            "prior_equilibrium_claim": False,
            "interpretation": "full-J drift diagnoses this fixture's prior bias only; not real biological prior error",
        })
        near, near_y, near_meta = _make_near_start(template, truth, int(spec["noise_seed"]))
        blind, blind_y, blind_meta = _make_blind_start(template, int(spec["blind_seed"]))
        start_records = {}
        for start_name, coordinates, raw_y, metadata in (("near", near, near_y, near_meta),
                                                          ("blind", blind, blind_y, blind_meta)):
            path = RUN / "inputs" / (fixture + "_" + start_name + "_start.npz")
            start_records[start_name] = save_start(path, coordinates, raw_y, P_INIT,
                                                    {"fixture": fixture, **metadata,
                                                     "p_init": P_INIT, "shared_by_models": ["S", "G"],
                                                     "raw_y_q_start_contract": True})
        records[fixture] = {
            "fixture": fixture, "data": data_record, "truth": truth_record,
            "known_e": known_meta, "starts": start_records,
            "group_totals": group_totals,
            "rates_sha256": _array_hash(rates),
            "truth_coordinate_sha256": _array_hash(truth),
            "generation_e_sha256": _array_hash(generated_e),
            "count_mode": expected_data.count_mode,
            "n_loci": expected_data.n_loci, "n_pairs": expected_data.n_pairs,
        }
        truth_records[fixture] = truth_record
        _write_event({"event": "synthetic_end", "fixture": fixture,
                      "group_totals": group_totals, "truth_sha256": truth_record["sha256"]})
    return records, truth_records


def _build_matrix(real_records: dict[str, Any], synthetic_records: dict[str, Any]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for model in ("S", "G"):
        for candidate in ("consensus", "random"):
            matrix.append({
                "fit_id": "real-%s-%s" % (model, candidate), "kind": "real", "model_id": model,
                "candidate": candidate, "data_mode": "raw_integer", "start_5Mb": str(
                    Path(real_records[candidate]["stages"]["5Mb"]["path"]).relative_to(RUN)),
                "stages": [dict(stage) for stage in REAL_STAGES],
                "weights": {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0},
                "formal_initialization": "same 014 5Mb root x0/raw-y/q; exact q carry after each accepted endpoint",
            })
    for fixture in ("P2", "N2"):
        for model in ("S", "G"):
            for start_name in ("near", "blind"):
                matrix.append({
                    "fit_id": "synthetic-%s-%s-%s-fullJ" % (fixture, model, start_name),
                    "kind": "synthetic", "fixture": fixture, "model_id": model,
                    "start_name": start_name, "objective_variant": "full-J",
                    "data_mode": "synthetic_expected", "stage": dict(SYNTHETIC_STAGE),
                    "data_path": synthetic_records[fixture]["data"]["path"],
                    "start_path": synthetic_records[fixture]["starts"][start_name]["path"],
                    "known_e_path": synthetic_records[fixture]["known_e"]["copied_path"],
                    "weights": {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0},
                })
        if fixture != "P2":
            continue
        for model in ("S", "G"):
            matrix.append({
                "fit_id": "synthetic-P2-%s-near-count-only" % model,
                "kind": "synthetic", "fixture": "P2", "model_id": model,
                "start_name": "near", "objective_variant": "count-only",
                "data_mode": "synthetic_expected", "stage": dict(SYNTHETIC_STAGE),
                "data_path": synthetic_records["P2"]["data"]["path"],
                "start_path": synthetic_records["P2"]["starts"]["near"]["path"],
                "known_e_path": synthetic_records["P2"]["known_e"]["copied_path"],
                "weights": {"count": 1.0, "bond": 0.0, "repulsion": 0.0, "bend": 0.0, "p_prior": 0.0},
            })
    if len(matrix) != 14:
        raise RuntimeError("formal fit matrix cardinality is not 14")
    stage_count = sum(len(row.get("stages", [row.get("stage")])) for row in matrix)
    if stage_count != 22:
        raise RuntimeError("formal stage matrix cardinality is not 22")
    return matrix


def prepare() -> dict[str, Any]:
    if not INPUT_PATH.is_file():
        raise RuntimeError("missing authorized SNP-free input: %s" % INPUT_PATH)
    RUN.mkdir(parents=True, exist_ok=True)
    for name in ("inputs", "coords", "plots", "logs", "checks", "eval_truth", "stages", "sidecars", "checkpoints"):
        (RUN / name).mkdir(parents=True, exist_ok=True)
    input_hash = sha256_file(INPUT_PATH)
    if input_hash != contact_model.FROZEN_P9016_SNPFREE_SHA256:
        raise RuntimeError("SNP-free SHA256 mismatch: %s" % input_hash)
    write_json(RUN / "inputs" / "source_input_manifest.json", {
        "path": str(INPUT_PATH.relative_to(ROOT)), "sha256": input_hash,
        "records": TOTAL_RECORDS, "columns": ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"],
        "phase_columns_forbidden": True,
    })
    real_data, real_records = _prepare_real_aggregates()
    template = header_templates()[1_000_000]
    if template.n_loci != FULL_FINAL_LOCI or len(template.chromosome_names) != N_CHROMOSOMES:
        raise RuntimeError("synthetic template does not match 20chr/2645-locus contract")
    initial_records = _prepare_real_initials(real_data)
    synthetic_records, truth_records = _prepare_synthetic(template, real_data[1_000_000])
    matrix = _build_matrix(initial_records, synthetic_records)
    source_files = {
        "pr/contact_model.py": sha256_file(ROOT / "pr/contact_model.py"),
        "pr/v1_calibration.py": sha256_file(ROOT / "pr/v1_calibration.py"),
        "pr/reconstruction_init.py": sha256_file(ROOT / "pr/reconstruction_init.py"),
        "pr/genome.py": sha256_file(ROOT / "pr/genome.py"),
        "pr/paths.py": sha256_file(ROOT / "pr/paths.py"),
        "test_res/039-20260914T164022Z-visibility-preflight/preflight.json": sha256_file(PREFLIGHT_ROOT / "preflight.json"),
        "test_res/039-20260914T164022Z-visibility-preflight/real_c0_parity_audit.json": sha256_file(PREFLIGHT_ROOT / "real_c0_parity_audit.json"),
        "known_e_worker_input_manifest.json": sha256_file(KNOWN_MANIFEST),
        "source/visibility_profile_base.py": sha256_file(SOURCE / "visibility_profile_base.py"),
        "source/gpu_variant_backend.py": sha256_file(SOURCE / "gpu_variant_backend.py"),
        "source/m1_preconditioner.py": sha256_file(SOURCE / "m1_preconditioner.py"),
        "source/shared_capture_objective.py": sha256_file(SOURCE / "shared_capture_objective.py"),
    }
    write_json(RUN / "inputs" / "source_hashes.json", source_files)
    stage_rows = []
    for fit in matrix:
        if fit["kind"] == "real":
            stage_rows.extend({"fit_id": fit["fit_id"], **stage} for stage in fit["stages"])
        else:
            stage_rows.append({"fit_id": fit["fit_id"], **fit["stage"]})
    formal_manifest = {
        "schema": "p9016-shared-capture-worker-manifest-v1",
        "run_id": RUN.name, "status": "prepared_no_optimizer", "optimizer_started": False,
        "truth_access": "controller receives no truth paths and never imports evaluator",
        "reference_access": False, "phase_access": False,
        "matrix": matrix, "stage_rows": stage_rows,
        "expected_fits": 14, "expected_stages": 22, "expected_outer_fg": FG_TOTAL,
        "optimizer": {
            "runner": "source/m1_preconditioner.py run_budgeted_lbfgs",
            "ftol": 0.0, "canonical_gtol": 1e-6, "maxls": 20,
            "maxiter_rule": "FGcap+1", "accepted_endpoint": "last accepted state",
            "no_early_accepted_iteration_cap": True,
        },
        "real_input_paths": {str(size): str(real_records[str(size)]["path"])
                             for size in (5_000_000, 2_000_000, 1_000_000)},
        "real_initial_paths": initial_records,
        "synthetic_inputs": synthetic_records,
        "source_hashes": source_files,
    }
    write_json(RUN / "inputs" / "formal_manifest.json", formal_manifest)
    evaluation_manifest = {
        "schema": "p9016-shared-capture-evaluation-manifest-v1",
        "run_id": RUN.name, "candidate_hash_gate_before_truth_or_reference": True,
        "truth_records": truth_records,
        "real_reference": {
            "path": "data/P9016.1m.3dg.gz", "role": "evaluation_only",
            "read_allowed_only_after_candidate_hashes": True,
        },
        "real_initial_controls": initial_records,
        "nulls": {
            "u_zero": "copyA=copyB candidate geometry",
            "random_u": {"draw_seeds": list(range(450500, 450516)),
                         "rule": "within-chromosome locus permutation of u=(XA-XB)/2; z fixed; one common global scale if needed"},
        },
        "bootstrap": {"seed": 450301, "draws": 10000, "unit": "paired chromosome", "biological_replicate_claim": False},
        "source_not_selection": True,
    }
    write_json(RUN / "inputs" / "evaluation_manifest.json", evaluation_manifest)
    config = {
        "schema": "p9016-shared-capture-round-v1",
        "run_id": RUN.name, "status": "prepared_no_optimizer",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "authorization": {"formal_started": False, "parent_release_required": True},
        "scientific_question": {
            "A": "shared capture intensity improves inter-space geometry and homolog structure recovery",
            "B": "known-truth near/blind gap reflects optimization or joint-target/regularization bias",
            "homolog_metric": "matched correlation and matched-cross; no center-distance surrogate",
        },
        "model": {
            "S": "separate cis/inter normalization with fixed production e",
            "G": "shared offdiag normalization with group-mass KL",
            "rate": "e_i e_j*[cis .5*(p*(KAA+KBB)+(1-p)*(KAB+KBA)); inter .25*(KAA+KAB+KBA+KBB)]",
            "kernel": "1e-6+(1-1e-6)*(1+d^2/r0^2)^-2",
            "r0": "2*(2*n_loci)^(-1/3)", "p": "bounded-q with p_prior internal 1e-4",
            "samebin": "saturated nuisance; excluded from shared structure normalization",
            "weights": {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0},
            "G_extra": "sum_g Ng*log[(Ng/Noff)/(Sg/(Scis+Sinter))]/Nraw",
            "fixed_production_e": True, "visibility_profile": False, "m1_precondition": False,
        },
        "data_boundary": {
            "snpfree": str(INPUT_PATH.relative_to(ROOT)), "snpfree_sha256": input_hash,
            "records": TOTAL_RECORDS, "chromosomes": N_CHROMOSOMES, "tracks": N_TRACKS,
            "real_layer_breakdown": {"5Mb": {"diag": 607552, "cis_offdiag": 527902, "inter": 568434},
                                     "2Mb": {"diag": 516046, "cis_offdiag": 619408, "inter": 568434},
                                     "1Mb": {"diag": 438774, "cis_offdiag": 696680, "inter": 568434}},
            "full_grid_zero_pairs_retained": True, "phase_opened": False, "reference_opened": False,
            "old_results_modified": False,
        },
        "matrix": {"fits": 14, "stages": 22, "real_fits": 4, "real_stages": 12,
                   "synthetic_fits": 10, "synthetic_stages": 10, "total_fg_cap": FG_TOTAL,
                   "formal_manifest": "inputs/formal_manifest.json"},
        "optimizer": {"ftol": 0.0, "canonical_gtol": 1e-6, "maxls": 20,
                      "maxiter_rule": "FGcap+1", "budget_logic": "hard accepted-FG cap",
                      "no_scientific_ftol_stop": True},
        "preflight": {"existing_039": "test_res/039-20260914T164022Z-visibility-preflight",
                       "parity_required": True, "G_KL_checks_required": True,
                       "reference_and_phase_not_opened": True},
        "hashes": {"input": input_hash, "source_manifest": source_files},
        "evaluation_manifest": "inputs/evaluation_manifest.json",
        "formal_release": "status changes to READY_FOR_FORMAL only after checks/preflight.py",
    }
    write_json(RUN / "config.json", config)
    write_json(RUN / "checks" / "prepare_summary.json", {
        "status": "prepared_no_optimizer", "optimizer_started": False,
        "real_records": real_records, "real_initials": initial_records,
        "synthetic": synthetic_records, "matrix": {"fits": 14, "stages": 22, "fg": FG_TOTAL},
        "truth_paths_isolated": True, "reference_opened": False, "phase_opened": False,
    })
    _write_event({"event": "prepare_complete", "fits": 14, "stages": 22,
                  "expected_fg": FG_TOTAL, "formal_started": False})
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare shared-capture run without optimizer")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    global RUN
    if args.run_dir is not None:
        RUN = Path(args.run_dir).resolve()
    result = prepare()
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
