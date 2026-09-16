"""Frozen-budget P3 synthetic calibration for Reconstruction V1.

The module has three deliberately separated stages:

* ``prepare`` reads only the verified SNP-free chromosome headers plus frozen
  count totals and writes unlabeled synthetic layer inputs and isolated truth;
* ``worker`` receives only one condition's unlabeled counts/exposure and an
  independent initialization, never a truth path;
* ``evaluate`` verifies candidate coordinate hashes before loading the isolated
  synthetic truth for R2/R3-style diagnostics.

It does not open P9016 phase columns, reference coordinates, oracle artifacts,
or production initialization coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from . import genome
from .contact_model import (
    EPSILON,
    FROZEN_P9016_CIS_RECORDS,
    FROZEN_P9016_INTER_RECORDS,
    FROZEN_P9016_RECORDS,
    JointObjective,
    aggregate_from_arrays,
    assert_inside_unit_ball,
    endpoint_counts_from_aggregates,
    q_from_p,
    sha256_file,
    sphere_forward,
    sphere_inverse,
    synthetic_expected_clone,
    synthetic_integer_clone,
    verify_frozen_snpfree,
    write_full_tracks,
)
from .joint_fit import JointCheckpoint, fit_joint
from .paths import SNPFREE
from .reconstruction_init import warm_start_from_layer
from .score import spearman


FINAL_BIN = 1_000_000
COARSE_BINS = (5_000_000, 2_000_000)
FULL_FINAL_LOCI = 2_645
N_CHROMOSOMES = 20
N_TRACKS = 40
TOTALS_1MB = {
    "diag": 438_774,
    "cis_offdiag": 696_680,
    "inter": 568_434,
}
assert sum(TOTALS_1MB.values()) == FROZEN_P9016_RECORDS
assert TOTALS_1MB["diag"] + TOTALS_1MB["cis_offdiag"] == FROZEN_P9016_CIS_RECORDS
assert TOTALS_1MB["inter"] == FROZEN_P9016_INTER_RECORDS

TRUTH_SEED = 4101
INIT_SEED = 5101
EXPOSURE_SEED = {"A": 4201, "C": 4203, "D": 4201}
DRAW_SEED = {"B": 6101, "C": 6103, "D": 6105}
P_GENERATING = 0.8
P_INITIAL = 0.75
TERRITORY_FACTOR = 0.65
CENTER_RADIUS = 0.55
CENTER_CANDIDATES = 4096
CHAIN_REJECTION_LIMIT = 10_000
NONADJACENT_MIN_FACTOR = 0.25
FRAGMENT_BINS = 20
TIE_TOL = 1e-12

LAYER_BUDGET = {
    5_000_000: 120,
    2_000_000: 80,
    1_000_000: 80,
}
FIT_FTOL = 1e-10
FIT_GTOL = 1e-6
FIT_MAXLS = 20

CONDITIONS = {
    "A": {
        "name": "noiseless_expected",
        "count_mode": "synthetic_expected",
        "truth_kind": "different_shape",
        "kernel": "v1",
        "p_gen": P_GENERATING,
        "exposure": "known_lognormal_sigma_0.4",
        "exposure_seed": 4201,
        "draw_seed": None,
        "fit_exposure": "known_synthetic",
    },
    "B": {
        "name": "known_sampling",
        "count_mode": "synthetic_integer",
        "truth_kind": "different_shape",
        "kernel": "v1",
        "p_gen": P_GENERATING,
        "exposure": "known_lognormal_sigma_0.4",
        "exposure_seed": 4201,
        "draw_seed": 6101,
        "fit_exposure": "known_synthetic",
    },
    "C": {
        "name": "capture_misspecified",
        "count_mode": "synthetic_integer",
        "truth_kind": "different_shape",
        "kernel": "sixth_power_capture_misspecified",
        "p_gen": P_GENERATING,
        "exposure": "lognormal_sigma_0.8_with_15pct_times_0.02",
        "exposure_seed": 4203,
        "draw_seed": 6103,
        "fit_exposure": "production_observed_endpoint",
    },
    "D": {
        "name": "same_shape_null",
        "count_mode": "synthetic_integer",
        "truth_kind": "same_internal_shape_translated",
        "kernel": "v1",
        "p_gen": P_GENERATING,
        "exposure": "known_lognormal_sigma_0.4",
        "exposure_seed": 4201,
        "draw_seed": 6105,
        "fit_exposure": "production_observed_endpoint",
    },
}


class CalibrationError(RuntimeError):
    """Raised when a preregistered calibration gate is violated."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    with open(path, "wt") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    with open(path, "at") as handle:
        handle.write(json.dumps(_jsonable(value), sort_keys=True) + "\n")


def _empty_header_template(names: tuple[str, ...], lengths: tuple[int, ...], bin_size: int):
    empty = np.empty(0, dtype=np.int64)
    return aggregate_from_arrays(names, lengths, empty, empty, empty, empty, bin_size)


def verify_frozen_run_manifest(run_dir: str | Path) -> dict:
    """Reject a formal stage when its registered source or protocol drifted."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    frozen_config = run_dir / "frozen_protocol" / "config.json"
    if not config_path.is_file() or not frozen_config.is_file():
        raise CalibrationError("formal config and frozen copy must exist")
    if sha256_file(config_path) != sha256_file(frozen_config):
        raise CalibrationError("formal config differs from its frozen copy")
    with open(config_path, "rt") as handle:
        config = json.load(handle)
    root = Path(__file__).resolve().parents[1]
    source_hashes = config.get("source_hashes", {})
    if not source_hashes:
        raise CalibrationError("formal config lacks a source manifest")
    for relative, expected in source_hashes.items():
        source_path = root / relative
        if not source_path.is_file() or sha256_file(source_path) != expected:
            raise CalibrationError("registered source hash mismatch: %s" % relative)
    for name, expected in config.get("protocols", {}).items():
        relative = expected["path"]
        source_path = root / relative
        frozen_path = run_dir / "frozen_protocol" / Path(relative).name
        if (not source_path.is_file() or not frozen_path.is_file()
                or sha256_file(source_path) != expected["sha256"]
                or sha256_file(frozen_path) != expected["sha256"]):
            raise CalibrationError("registered protocol hash mismatch: %s" % name)
    return {
        "config_sha256": sha256_file(config_path),
        "n_source_hashes": len(source_hashes),
        "n_protocol_hashes": len(config.get("protocols", {})),
    }


def header_templates() -> dict[int, Any]:
    """Read only the verified SNP-free header, never its body records."""
    verify_frozen_snpfree(SNPFREE)
    headers = genome.chrom_lengths(SNPFREE)
    names = tuple(name for name, _ in headers)
    lengths = tuple(int(length) for _, length in headers)
    if len(names) != N_CHROMOSOMES:
        raise CalibrationError("synthetic calibration requires all 20 SNP-free headers")
    templates = {size: _empty_header_template(names, lengths, size)
                 for size in (FINAL_BIN,) + COARSE_BINS}
    if templates[FINAL_BIN].n_loci != FULL_FINAL_LOCI:
        raise CalibrationError("final full-grid locus count does not match the frozen 2645")
    return templates


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        vector = rng.normal(size=3)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            return vector / norm
    raise CalibrationError("failed to draw a nonzero Gaussian direction")


def _uniform_volume_points(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    directions = np.empty((n, 3), dtype=np.float64)
    for index in range(n):
        directions[index] = _unit_vector(rng)
    radii = radius * rng.random(n) ** (1.0 / 3.0)
    return directions * radii[:, None]


def _farthest_point_centers(rng: np.random.Generator, n_tracks: int) -> np.ndarray:
    candidates = _uniform_volume_points(rng, CENTER_CANDIDATES, CENTER_RADIUS)
    chosen = np.empty(n_tracks, dtype=np.int64)
    chosen[0] = int(rng.integers(0, len(candidates)))
    nearest = np.sum((candidates - candidates[chosen[0]]) ** 2, axis=1)
    for index in range(1, n_tracks):
        chosen[index] = int(np.argmax(nearest))
        squared = np.sum((candidates - candidates[chosen[index]]) ** 2, axis=1)
        nearest = np.minimum(nearest, squared)
    return candidates[chosen[rng.permutation(n_tracks)]]


def _chain_offsets(rng: np.random.Generator, n_loci: int, l0: float,
                   territory_radius: float, centers: np.ndarray) -> tuple[np.ndarray, int]:
    """Generate one bounded chain relative to a territory center."""
    points = np.empty((n_loci, 3), dtype=np.float64)
    points[0] = 0.0
    previous_direction = _unit_vector(rng)
    rejected = 0
    for index in range(1, n_loci):
        accepted = False
        for _ in range(CHAIN_REJECTION_LIMIT):
            direction = 0.2 * previous_direction + 0.8 * _unit_vector(rng)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm == 0.0:
                rejected += 1
                continue
            direction /= direction_norm
            step = rng.uniform(0.85, 1.15) * l0
            proposal = points[index - 1] + step * direction
            if np.linalg.norm(proposal) > territory_radius:
                rejected += 1
                continue
            if np.any(np.linalg.norm(centers + proposal, axis=1) >= 1.0):
                rejected += 1
                continue
            if index > 1:
                old = points[:index - 1]
                if np.any(np.linalg.norm(old - proposal, axis=1) < NONADJACENT_MIN_FACTOR * l0):
                    rejected += 1
                    continue
            points[index] = proposal
            previous_direction = direction
            accepted = True
            break
        if not accepted:
            raise CalibrationError(
                "chain generation exhausted %d attempts at locus %d of %d"
                % (CHAIN_REJECTION_LIMIT, index, n_loci))
    return points, rejected


def generate_truth(template, seed: int, same_shape: bool) -> tuple[np.ndarray, dict]:
    """Generate a 40-track, header-complete synthetic polymer geometry."""
    rng = np.random.default_rng(seed)
    centers = _farthest_point_centers(rng, N_TRACKS)
    coordinates = np.empty((2, template.n_loci, 3), dtype=np.float64)
    rejected_total = 0
    per_track = {}
    for chromosome in range(len(template.chromosome_names)):
        slc = template.chromosome_slice(chromosome)
        n_loci = slc.stop - slc.start
        territory_radius = TERRITORY_FACTOR * (n_loci / template.n_loci) ** (1.0 / 3.0)
        if same_shape:
            offsets, rejected = _chain_offsets(
                rng, n_loci, template.l0, territory_radius,
                centers[2 * chromosome:2 * chromosome + 2],
            )
            rejected_total += rejected
            for copy in (0, 1):
                center = centers[2 * chromosome + copy]
                coordinates[copy, slc] = center + offsets
                per_track["c%02d%s" % (chromosome + 1, "ab"[copy])] = {
                    "center": center.tolist(),
                    "territory_radius": float(territory_radius),
                    "rejections": int(rejected),
                    "internal_shape_source": "shared_with_homolog",
                }
        else:
            for copy in (0, 1):
                center = centers[2 * chromosome + copy]
                offsets, rejected = _chain_offsets(
                    rng, n_loci, template.l0, territory_radius, center[None, :]
                )
                rejected_total += rejected
                coordinates[copy, slc] = center + offsets
                per_track["c%02d%s" % (chromosome + 1, "ab"[copy])] = {
                    "center": center.tolist(),
                    "territory_radius": float(territory_radius),
                    "rejections": int(rejected),
                    "internal_shape_source": "independent",
                }
    assert_inside_unit_ball(coordinates)
    return coordinates, {
        "seed": int(seed),
        "same_internal_shape": bool(same_shape),
        "center_candidates": CENTER_CANDIDATES,
        "center_radius": CENTER_RADIUS,
        "territory_factor": TERRITORY_FACTOR,
        "step_uniform_range_l0": [0.85, 1.15],
        "direction_rule": "normalize(0.2*previous + 0.8*standard_gaussian)",
        "nonadjacent_minimum_l0": NONADJACENT_MIN_FACTOR,
        "rejection_limit": CHAIN_REJECTION_LIMIT,
        "rejected_proposals": int(rejected_total),
        "max_radius": float(np.linalg.norm(coordinates, axis=2).max()),
        "per_track": per_track,
        "prior_equilibrium_claim": False,
    }


def synthetic_exposure(n_loci: int, sigma: float, seed: int,
                       capture_drop: bool = False) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    exposure = np.exp(rng.normal(0.0, sigma, n_loci))
    dropped = np.empty(0, dtype=np.int64)
    if capture_drop:
        n_drop = int(round(0.15 * n_loci))
        dropped = np.sort(rng.choice(n_loci, size=n_drop, replace=False))
        exposure[dropped] *= 0.02
    exposure /= exposure.mean()
    return exposure.astype(np.float64), {
        "seed": int(seed),
        "sigma": float(sigma),
        "capture_drop": bool(capture_drop),
        "n_capture_drop_bins": int(len(dropped)),
        "capture_drop_multiplier": 0.02 if capture_drop else None,
        "mean": float(exposure.mean()),
        "min": float(exposure.min()),
        "max": float(exposure.max()),
    }


def _kernel_values(delta: np.ndarray, r0: float, kernel: str) -> np.ndarray:
    ratio2 = np.sum(delta * delta, axis=-1) / (r0 * r0)
    if kernel == "v1":
        return EPSILON + (1.0 - EPSILON) * (1.0 + ratio2) ** -2
    if kernel == "sixth_power_capture_misspecified":
        return EPSILON + (1.0 - EPSILON) / (1.0 + ratio2 ** 3)
    raise ValueError("unknown synthetic generating kernel %s" % kernel)


def generation_rates(template, coordinates: np.ndarray, exposure: np.ndarray,
                     p: float, kernel: str, block_size: int = 65_536) -> np.ndarray:
    """Generate unlabeled full-E rates using a preregistered synthetic kernel."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    if coordinates.shape != (2, template.n_loci, 3):
        raise ValueError("synthetic coordinates must be (2, n_loci, 3)")
    if exposure.shape != (template.n_loci,):
        raise ValueError("synthetic exposure must be full-grid aligned")
    assert_inside_unit_ball(coordinates)
    rates = np.empty(template.n_pairs, dtype=np.float64)
    for start in range(0, template.n_pairs, block_size):
        stop = min(start + block_size, template.n_pairs)
        i = template.pair_i[start:stop]
        j = template.pair_j[start:stop]
        cis = template.cis_pair[start:stop]
        kaa = _kernel_values(coordinates[0, i] - coordinates[0, j], template.r0, kernel)
        kab = _kernel_values(coordinates[0, i] - coordinates[1, j], template.r0, kernel)
        kba = _kernel_values(coordinates[1, i] - coordinates[0, j], template.r0, kernel)
        kbb = _kernel_values(coordinates[1, i] - coordinates[1, j], template.r0, kernel)
        mixture = np.where(cis, 0.5 * (p * (kaa + kbb) + (1.0 - p) * (kab + kba)),
                           0.25 * (kaa + kab + kba + kbb))
        rates[start:stop] = exposure[i] * exposure[j] * mixture
    if not np.all(np.isfinite(rates)) or np.any(rates <= 0.0):
        raise CalibrationError("synthetic generation produced nonpositive or nonfinite rates")
    return rates


def expected_counts(template, rates: np.ndarray, exposure: np.ndarray,
                    totals: Mapping[str, int] = TOTALS_1MB) -> tuple[np.ndarray, np.ndarray]:
    rates = np.asarray(rates, dtype=np.float64)
    if rates.shape != (template.n_pairs,):
        raise ValueError("rates must cover full E")
    counts = np.zeros(template.n_pairs, dtype=np.float64)
    for mask, group in ((template.cis_pair, "cis_offdiag"), (~template.cis_pair, "inter")):
        denominator = float(rates[mask].sum())
        if denominator <= 0.0:
            raise CalibrationError("synthetic rate group has nonpositive sum")
        counts[mask] = float(totals[group]) * rates[mask] / denominator
    diag_weights = exposure * exposure
    diag = float(totals["diag"]) * diag_weights / float(diag_weights.sum())
    return counts, diag


def sample_conditional_counts(template, rates: np.ndarray, exposure: np.ndarray,
                              seed: int, totals: Mapping[str, int] = TOTALS_1MB) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = np.zeros(template.n_pairs, dtype=np.int64)
    for mask, group in ((template.cis_pair, "cis_offdiag"), (~template.cis_pair, "inter")):
        probabilities = rates[mask] / float(rates[mask].sum())
        counts[mask] = rng.multinomial(int(totals[group]), probabilities).astype(np.int64, copy=False)
    diag_probabilities = exposure * exposure
    diag_probabilities /= diag_probabilities.sum()
    diag = rng.multinomial(int(totals["diag"]), diag_probabilities).astype(np.int64, copy=False)
    return counts, diag


def observed_endpoint_exposure(template, counts: np.ndarray, diag_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    endpoint_counts = endpoint_counts_from_aggregates(template, counts, diag_counts)
    exposure = np.sqrt(endpoint_counts.astype(np.float64) + 10.0)
    exposure /= exposure.mean()
    return exposure, endpoint_counts


def _upper_pair_index(i: np.ndarray, j: np.ndarray, n_loci: int) -> np.ndarray:
    return i * (2 * n_loci - i - 1) // 2 + (j - i - 1)


def coarsen_from_final(fine, coarse_template, exposure_strategy: str):
    """Map 1 Mb synthetic aggregates to a coarser full header grid exactly."""
    if coarse_template.bin_size % fine.bin_size != 0:
        raise CalibrationError("coarse bin size must be an integer multiple of final bin size")
    mapped = coarse_template.global_locus(
        fine.locus_chromosome,
        (fine.locus_bin * fine.bin_size) // coarse_template.bin_size,
    ).astype(np.int64)
    integer_mode = fine.count_mode == "synthetic_integer"
    dtype = np.int64 if integer_mode else np.float64
    counts = np.zeros(coarse_template.n_pairs, dtype=dtype)
    diag = np.zeros(coarse_template.n_loci, dtype=dtype)
    np.add.at(diag, mapped, fine.diag_counts)
    for start in range(0, fine.n_pairs, 65_536):
        stop = min(start + 65_536, fine.n_pairs)
        source_count = fine.counts[start:stop]
        left = mapped[fine.pair_i[start:stop]]
        right = mapped[fine.pair_j[start:stop]]
        collapsed = left == right
        if collapsed.any():
            np.add.at(diag, left[collapsed], source_count[collapsed])
        keep = ~collapsed
        if keep.any():
            lo = np.minimum(left[keep], right[keep])
            hi = np.maximum(left[keep], right[keep])
            np.add.at(counts, _upper_pair_index(lo, hi, coarse_template.n_loci), source_count[keep])
    endpoints = endpoint_counts_from_aggregates(coarse_template, counts, diag)
    if exposure_strategy == "known_synthetic":
        exposure = np.zeros(coarse_template.n_loci, dtype=np.float64)
        np.add.at(exposure, mapped, fine.exposure)
        exposure /= exposure.mean()
        exposure_mode = "known_synthetic_coarsened_sum_approximate"
    elif exposure_strategy == "production_observed_endpoint":
        exposure = np.sqrt(endpoints.astype(np.float64) + 10.0)
        exposure /= exposure.mean()
        exposure_mode = "production_observed_endpoint_recomputed"
    else:
        raise ValueError("unknown coarse exposure strategy %s" % exposure_strategy)
    if integer_mode:
        return synthetic_integer_clone(
            coarse_template, counts, diag, exposure,
            exposure_mode=exposure_mode, endpoint_counts=endpoints,
        )
    return synthetic_expected_clone(
        coarse_template, counts, diag, exposure,
        exposure_mode=exposure_mode, endpoint_counts=endpoints,
    )


def _layer_group_totals(data) -> dict[str, float | int]:
    audit = data.budget()
    return {
        "diag": audit["aggregate_same_bin"],
        "cis_offdiag": audit["aggregate_cis_offdiag"],
        "inter": audit["aggregate_inter"],
    }


def save_layer(path: str | Path, data) -> None:
    totals = _layer_group_totals(data)
    np.savez_compressed(
        path,
        chromosome_names=np.asarray(data.chromosome_names),
        chromosome_lengths=np.asarray(data.chromosome_lengths, dtype=np.int64),
        bin_size=np.asarray(data.bin_size, dtype=np.int64),
        counts=data.counts,
        diag_counts=data.diag_counts,
        exposure=data.exposure,
        endpoint_counts=data.endpoint_counts,
        count_mode=np.asarray(data.count_mode),
        exposure_mode=np.asarray(data.exposure_mode),
        group_diag=np.asarray(totals["diag"]),
        group_cis_offdiag=np.asarray(totals["cis_offdiag"]),
        group_inter=np.asarray(totals["inter"]),
    )


def load_layer(path: str | Path):
    with np.load(path, allow_pickle=False) as payload:
        names = tuple(str(value) for value in payload["chromosome_names"].tolist())
        lengths = tuple(int(value) for value in payload["chromosome_lengths"].tolist())
        bin_size = int(payload["bin_size"])
        counts = payload["counts"].copy()
        diag = payload["diag_counts"].copy()
        exposure = payload["exposure"].copy()
        endpoints = payload["endpoint_counts"].copy()
        count_mode = str(payload["count_mode"].item())
        exposure_mode = str(payload["exposure_mode"].item())
        totals = {
            "diag": payload["group_diag"].item(),
            "cis_offdiag": payload["group_cis_offdiag"].item(),
            "inter": payload["group_inter"].item(),
        }
    template = _empty_header_template(names, lengths, bin_size)
    if count_mode == "synthetic_expected":
        return synthetic_expected_clone(
            template, counts, diag, exposure, totals,
            exposure_mode=exposure_mode, endpoint_counts=endpoints,
        )
    if count_mode == "synthetic_integer":
        totals = {key: int(value) for key, value in totals.items()}
        return synthetic_integer_clone(
            template, counts, diag, exposure, totals,
            exposure_mode=exposure_mode, endpoint_counts=endpoints,
        )
    raise CalibrationError("worker refuses unknown synthetic count mode %s" % count_mode)


def _save_initialization(path: str | Path, coords: np.ndarray, data, seed: int) -> None:
    assert_inside_unit_ball(coords)
    np.savez_compressed(
        path,
        coords=np.asarray(coords, dtype=np.float64),
        positions=(data.locus_bin * data.bin_size).astype(np.int64),
        chromosome_index=data.locus_chromosome.astype(np.int32),
        seed=np.asarray(seed, dtype=np.int64),
    )


def _load_initialization(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as payload:
        coords = payload["coords"].copy()
        positions = payload["positions"].copy()
        chromosome_index = payload["chromosome_index"].copy()
        seed = int(payload["seed"])
    assert_inside_unit_ball(coords)
    return coords, positions, chromosome_index, seed


def _truth_path(run_dir: Path, condition: str) -> Path:
    return run_dir / "eval_truth" / (condition + "_truth_1m.npz")


def _save_truth(path: Path, coordinates: np.ndarray, metadata: Mapping[str, Any],
                exposure: np.ndarray) -> None:
    assert_inside_unit_ball(coordinates)
    np.savez_compressed(path, coordinates=coordinates, exposure=exposure)
    _write_json(str(path) + ".json", dict(metadata))


def _load_truth(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as payload:
        coordinates = payload["coordinates"].copy()
        exposure = payload["exposure"].copy()
    with open(str(path) + ".json", "rt") as handle:
        metadata = json.load(handle)
    assert_inside_unit_ball(coordinates)
    return coordinates, exposure, metadata


def _condition_paths(run_dir: Path, condition: str, bin_size: int) -> Path:
    return run_dir / "work" / ("%s_%dmb_layer.npz" % (condition, bin_size // 1_000_000))


def prepare_run(run_dir: str | Path) -> dict:
    """Create all synthetic artifacts before any worker optimization starts."""
    run_dir = Path(run_dir)
    manifest = verify_frozen_run_manifest(run_dir)
    templates = header_templates()
    final_template = templates[FINAL_BIN]
    normal_truth, normal_meta = generate_truth(final_template, TRUTH_SEED, same_shape=False)
    null_truth, null_meta = generate_truth(final_template, TRUTH_SEED, same_shape=True)
    exposure_ab, exposure_ab_meta = synthetic_exposure(final_template.n_loci, 0.4, 4201)
    exposure_c, exposure_c_meta = synthetic_exposure(final_template.n_loci, 0.8, 4203, capture_drop=True)
    rates_normal = generation_rates(final_template, normal_truth, exposure_ab, P_GENERATING, "v1")
    rates_capture = generation_rates(final_template, normal_truth, exposure_c, P_GENERATING,
                                     "sixth_power_capture_misspecified")
    rates_null = generation_rates(final_template, null_truth, exposure_ab, P_GENERATING, "v1")

    prepared = {}
    for condition, spec in CONDITIONS.items():
        if condition in ("A", "B"):
            truth, truth_meta, exposure, rates = normal_truth, normal_meta, exposure_ab, rates_normal
            exposure_meta = exposure_ab_meta
        elif condition == "C":
            truth, truth_meta, exposure, rates = normal_truth, normal_meta, exposure_c, rates_capture
            exposure_meta = exposure_c_meta
        else:
            truth, truth_meta, exposure, rates = null_truth, null_meta, exposure_ab, rates_null
            exposure_meta = exposure_ab_meta

        if spec["count_mode"] == "synthetic_expected":
            counts, diag = expected_counts(final_template, rates, exposure)
            endpoints = endpoint_counts_from_aggregates(final_template, counts, diag)
            data_final = synthetic_expected_clone(
                final_template, counts, diag, exposure, TOTALS_1MB,
                exposure_mode="known_synthetic_well_specified", endpoint_counts=endpoints,
            )
        else:
            counts, diag = sample_conditional_counts(final_template, rates, exposure, spec["draw_seed"])
            endpoints = endpoint_counts_from_aggregates(final_template, counts, diag)
            if spec["fit_exposure"] == "known_synthetic":
                fit_exposure = exposure
                exposure_mode = "known_synthetic_well_specified"
            else:
                fit_exposure, endpoints = observed_endpoint_exposure(final_template, counts, diag)
                exposure_mode = "production_observed_endpoint"
            data_final = synthetic_integer_clone(
                final_template, counts, diag, fit_exposure, TOTALS_1MB,
                exposure_mode=exposure_mode, endpoint_counts=endpoints,
            )

        data_2m = coarsen_from_final(data_final, templates[2_000_000], spec["fit_exposure"])
        data_5m = coarsen_from_final(data_final, templates[5_000_000], spec["fit_exposure"])
        for data in (data_final, data_2m, data_5m):
            audit = data.budget()
            if not (audit["raw_conserved"] and audit["aggregate_conserved"]):
                raise CalibrationError("synthetic layer did not conserve its count budget")
        save_layer(_condition_paths(run_dir, condition, FINAL_BIN), data_final)
        save_layer(_condition_paths(run_dir, condition, 2_000_000), data_2m)
        save_layer(_condition_paths(run_dir, condition, 5_000_000), data_5m)

        truth_objective = JointObjective(data_final)
        truth_theta = truth_objective.pack(sphere_inverse(truth), p=P_GENERATING)
        _, _, truth_components = truth_objective.evaluate(truth_theta, need_gradient=False)
        truth_record = dict(truth_meta)
        truth_record.update({
            "condition": condition,
            "condition_name": spec["name"],
            "generating_kernel": spec["kernel"],
            "p_gen": P_GENERATING,
            "exposure_generation": exposure_meta,
            "truth_prior_components": {key: truth_components[key] for key in
                                       ("bond", "repulsion", "bend", "p_prior")},
            "truth_total_components": {key: truth_components[key] for key in
                                       ("count_nll_normalized", "conditional_nll_raw",
                                        "diag_profiled_nll_raw", "total")},
            "evaluation_only": True,
        })
        _save_truth(_truth_path(run_dir, condition), truth, truth_record, exposure)
        prepared[condition] = {
            "condition": spec,
            "final": _layer_group_totals(data_final),
            "2mb": _layer_group_totals(data_2m),
            "5mb": _layer_group_totals(data_5m),
            "data_sha256": {
                "1mb": sha256_file(_condition_paths(run_dir, condition, FINAL_BIN)),
                "2mb": sha256_file(_condition_paths(run_dir, condition, 2_000_000)),
                "5mb": sha256_file(_condition_paths(run_dir, condition, 5_000_000)),
            },
            "truth_path": str(_truth_path(run_dir, condition).relative_to(run_dir)),
            "truth_sha256": sha256_file(_truth_path(run_dir, condition)),
        }

    init_data = templates[5_000_000]
    init_coords, init_meta = generate_truth(init_data, INIT_SEED, same_shape=False)
    for condition in CONDITIONS:
        init_path = run_dir / "work" / (condition + "_independent_init_5mb.npz")
        _save_initialization(init_path, init_coords, init_data, INIT_SEED)
        prepared[condition]["initialization_sha256"] = sha256_file(init_path)
        prepared[condition]["initialization_seed"] = INIT_SEED
    result = {
        "frozen_manifest": manifest,
        "prepared_at_unix": time.time(),
        "header_only_input": verify_frozen_snpfree(SNPFREE),
        "full_grid_loci": final_template.n_loci,
        "initialization": init_meta,
        "conditions": prepared,
    }
    _write_json(run_dir / "work" / "preparation.json", result)
    _append_jsonl(run_dir / "logs" / "status.jsonl", {
        "event": "prepared_synthetic_inputs",
        "full_grid_loci": final_template.n_loci,
        "conditions": list(CONDITIONS),
    })
    return result


def _checkpoint_writer(run_dir: Path, condition: str, bin_size: int):
    prefix = run_dir / "checkpoints" / ("%s_%dmb" % (condition, bin_size // 1_000_000))

    def write(checkpoint: JointCheckpoint) -> None:
        npz_path = Path("%s_iter%04d.npz" % (prefix, checkpoint.iteration))
        json_path = Path("%s_iter%04d.json" % (prefix, checkpoint.iteration))
        np.savez_compressed(npz_path, theta=checkpoint.theta, y=checkpoint.y,
                            coordinates=checkpoint.coordinates)
        _write_json(json_path, {
            "iteration": checkpoint.iteration,
            "nfev": checkpoint.nfev,
            "elapsed_seconds": checkpoint.elapsed_seconds,
            "fun": checkpoint.fun,
            "p": checkpoint.p,
            "components": checkpoint.components,
        })

    return write


def _fit_one_layer(run_dir: Path, condition: str, data, initial_coords: np.ndarray,
                   p_init: float, q_init: float | None, candidate_seed: int) -> tuple[Any, dict]:
    bin_size = data.bin_size
    log_path = run_dir / "logs" / ("%s_%dmb.jsonl" % (condition, bin_size // 1_000_000))
    objective = JointObjective(data)
    y = sphere_inverse(initial_coords)
    initial_theta = objective.pack(y, p=p_init)
    if q_init is not None:
        initial_theta[-1] = q_init
    initial_value, _, initial_components = objective.evaluate(initial_theta, need_gradient=False)
    _append_jsonl(log_path, {
        "event": "layer_initial",
        "condition": condition,
        "bin_size": bin_size,
        "total": initial_value,
        "components": initial_components,
        "p_init": p_init,
        "q_init_carried": q_init,
    })

    def callback(entry: dict) -> None:
        if entry["iteration"] % 10 == 0:
            _append_jsonl(log_path, {
                "event": "iteration",
                "condition": condition,
                "bin_size": bin_size,
                **entry,
            })

    maxiter = LAYER_BUDGET[bin_size]
    fit = fit_joint(
        objective,
        y,
        p_init=p_init,
        q_init=q_init,
        maxiter=maxiter,
        maxfun=3 * maxiter + 30,
        maxls=FIT_MAXLS,
        ftol=FIT_FTOL,
        gtol=FIT_GTOL,
        callback=callback,
        checkpoint_every=10,
        checkpoint_hook=_checkpoint_writer(run_dir, condition, bin_size),
    )
    tolerance = 1e-10 * max(1.0, abs(float(initial_value)))
    nonincrease = bool(fit.fun <= float(initial_value) + tolerance)
    if fit.success:
        termination = "converged"
    elif fit.status == 1:
        termination = "budget_terminated"
    else:
        termination = "not_converged_other"
    record = {
        "condition": condition,
        "bin_size": bin_size,
        "initial_total": float(initial_value),
        "final_total": float(fit.fun),
        "nonincrease_tolerance": tolerance,
        "final_not_worse_than_initial": nonincrease,
        "optimizer_success": bool(fit.success),
        "optimizer_status": int(fit.status),
        "optimizer_message": fit.message,
        "termination": termination,
        "nit": fit.nit,
        "actual_nfev": fit.actual_nfev,
        "scipy_nfev": fit.scipy_nfev,
        "njev": fit.njev,
        "elapsed_seconds": fit.elapsed_seconds,
        "p": fit.p,
        "q": float(fit.theta[-1]),
        "components": fit.components,
        "history_path": "work/%s_%dmb_history.json" % (condition, bin_size // 1_000_000),
        "warm_start_candidate_base_seed": candidate_seed,
    }
    _write_json(run_dir / record["history_path"], {"history": fit.history})
    np.savez_compressed(run_dir / "coords" / ("%s_%dmb_theta.npz" %
                                                (condition, bin_size // 1_000_000)),
                        theta=fit.theta, coordinates=fit.coordinates)
    write_full_tracks(run_dir / "coords" / ("%s_%dmb.3dg" %
                                             (condition, bin_size // 1_000_000)), data,
                      fit.coordinates)
    _append_jsonl(log_path, {"event": "layer_terminal", **record})
    if not nonincrease:
        raise CalibrationError("final total exceeded initial total at %d Mb" % (bin_size // 1_000_000))
    return fit, record


def run_worker(run_dir: str | Path, condition: str) -> dict:
    """Fit one condition without loading any truth artifact."""
    if condition not in CONDITIONS:
        raise CalibrationError("unknown preregistered condition")
    run_dir = Path(run_dir)
    manifest = verify_frozen_run_manifest(run_dir)
    required = [
        _condition_paths(run_dir, condition, FINAL_BIN),
        _condition_paths(run_dir, condition, 2_000_000),
        _condition_paths(run_dir, condition, 5_000_000),
        run_dir / "work" / (condition + "_independent_init_5mb.npz"),
    ]
    if not all(path.is_file() for path in required):
        raise CalibrationError("synthetic worker inputs are absent; run prepare first")
    # The worker deliberately has no eval_truth path parameter or read operation.
    layers = {size: load_layer(_condition_paths(run_dir, condition, size))
              for size in (5_000_000, 2_000_000, 1_000_000)}
    coords, positions, chromosome_index, seed = _load_initialization(
        run_dir / "work" / (condition + "_independent_init_5mb.npz"))
    if coords.shape[1] != layers[5_000_000].n_loci:
        raise CalibrationError("independent initialization does not match 5 Mb layer")
    stage_records = []
    fit, record = _fit_one_layer(run_dir, condition, layers[5_000_000], coords,
                                 p_init=P_INITIAL, q_init=None, candidate_seed=seed)
    stage_records.append(record)
    previous_data = layers[5_000_000]
    previous_fit = fit
    for next_size in (2_000_000, 1_000_000):
        warm = warm_start_from_layer(
            previous_fit.coordinates,
            previous_data.locus_bin * previous_data.bin_size,
            previous_data.locus_chromosome,
            previous_data.chromosome_names,
            previous_data.chromosome_lengths,
            next_size,
            candidate_base_seed=seed,
        )
        fit, record = _fit_one_layer(
            run_dir,
            condition,
            layers[next_size],
            warm["coords"],
            p_init=P_INITIAL,
            q_init=float(previous_fit.theta[-1]),
            candidate_seed=seed,
        )
        record["warm_start"] = warm["metadata"]
        stage_records.append(record)
        previous_data = layers[next_size]
        previous_fit = fit
    final_npz = run_dir / "coords" / (condition + "_1mb_theta.npz")
    final_text = run_dir / "coords" / (condition + "_1mb.3dg")
    result = {
        "frozen_manifest": manifest,
        "condition": condition,
        "condition_name": CONDITIONS[condition]["name"],
        "status": "completed",
        "training_side_truth_access": False,
        "initialization_seed": seed,
        "layers": stage_records,
        "candidate_coordinate_npz": str(final_npz.relative_to(run_dir)),
        "candidate_coordinate_npz_sha256": sha256_file(final_npz),
        "candidate_coordinate_3dg": str(final_text.relative_to(run_dir)),
        "candidate_coordinate_3dg_sha256": sha256_file(final_text),
        "final_finite": bool(np.all(np.isfinite(previous_fit.coordinates))),
        "final_max_radius": float(np.linalg.norm(previous_fit.coordinates, axis=2).max()),
    }
    _write_json(run_dir / "work" / (condition + "_training.json"), result)
    _append_jsonl(run_dir / "logs" / "status.jsonl", {
        "event": "worker_completed",
        "condition": condition,
        "final_total": stage_records[-1]["final_total"],
        "termination": stage_records[-1]["termination"],
    })
    return result


def _pair_distances(coords: np.ndarray, slc: slice) -> np.ndarray:
    n = slc.stop - slc.start
    i, j = np.triu_indices(n, k=1)
    return np.linalg.norm(coords[slc.start + i] - coords[slc.start + j], axis=1)


def synthetic_r2(candidate: np.ndarray, truth: np.ndarray, data) -> dict:
    rows = []
    for chromosome, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(chromosome)
        n = slc.stop - slc.start
        candidate_distance = [_pair_distances(candidate[copy], slc) for copy in (0, 1)]
        truth_distance = [_pair_distances(truth[copy], slc) for copy in (0, 1)]
        rho = [[spearman(candidate_distance[left], truth_distance[right])
                for right in (0, 1)] for left in (0, 1)]
        direct = 0.5 * (rho[0][0] + rho[1][1])
        swapped = 0.5 * (rho[0][1] + rho[1][0])
        if not (math.isfinite(direct) and math.isfinite(swapped)):
            orientation = "unavailable"
            contrast = None
        elif abs(direct - swapped) <= TIE_TOL:
            orientation = "tie"
            contrast = 0.0
        elif direct > swapped:
            orientation = "direct"
            contrast = float(direct - swapped)
        else:
            orientation = "swapped"
            contrast = float(swapped - direct)
        rows.append({
            "chromosome_index": chromosome,
            "chromosome": name,
            "n_common_finite_pairs": int(n * (n - 1) // 2),
            "rho": rho,
            "direct_mean": float(direct),
            "swapped_mean": float(swapped),
            "orientation": orientation,
            "contrast": contrast,
        })
    contrast = [row["contrast"] for row in rows if row["contrast"] is not None]
    return {
        "per_chromosome": rows,
        "n_chromosomes": len(rows),
        "mean_contrast": float(np.mean(contrast)) if contrast else None,
        "n_tied_or_unavailable": int(sum(row["orientation"] in ("tie", "unavailable")
                                          for row in rows)),
    }


def _fragment_orientation(candidate: np.ndarray, truth: np.ndarray, start: int,
                          stop: int) -> tuple[int | None, float | None, float | None]:
    n = stop - start
    if n < 3:
        return None, None, None
    slc = slice(start, stop)
    candidate_distance = [_pair_distances(candidate[copy], slc) for copy in (0, 1)]
    truth_distance = [_pair_distances(truth[copy], slc) for copy in (0, 1)]
    direct = 0.5 * (spearman(candidate_distance[0], truth_distance[0])
                    + spearman(candidate_distance[1], truth_distance[1]))
    swapped = 0.5 * (spearman(candidate_distance[0], truth_distance[1])
                     + spearman(candidate_distance[1], truth_distance[0]))
    if not (math.isfinite(direct) and math.isfinite(swapped)) or abs(direct - swapped) <= TIE_TOL:
        return None, float(direct), float(swapped)
    return (0 if direct > swapped else 1), float(direct), float(swapped)


def synthetic_r3(candidate: np.ndarray, truth: np.ndarray, data) -> dict:
    per_chromosome = []
    for chromosome, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(chromosome)
        n = slc.stop - slc.start
        detail = []
        labels = []
        for local_start in range(0, n - FRAGMENT_BINS + 1, FRAGMENT_BINS):
            local_stop = local_start + FRAGMENT_BINS
            label, direct, swapped = _fragment_orientation(
                candidate, truth, slc.start + local_start, slc.start + local_stop)
            detail.append({
                "grid_start_bin": int(local_start),
                "grid_start_bp": int(local_start * data.bin_size),
                "label": label,
                "direct_mean": direct,
                "swapped_mean": swapped,
                "n_pairs": int(FRAGMENT_BINS * (FRAGMENT_BINS - 1) // 2),
            })
            labels.append(label)
        resolved = [label for label in labels if label is not None]
        counts = {label: resolved.count(label) for label in (0, 1)}
        if not resolved:
            result = {
                "applicable": False,
                "reason": "all_local_fragment_labels_tied_or_unavailable",
                "global_label": None,
                "frac_consistent": None,
                "longest_run": None,
                "n_walls": None,
            }
        elif counts[0] == counts[1]:
            result = {
                "applicable": False,
                "reason": "global_fragment_label_tie",
                "global_label": None,
                "frac_consistent": None,
                "longest_run": None,
                "n_walls": None,
            }
        else:
            global_label = 0 if counts[0] > counts[1] else 1
            walls = 0
            run = best = 0
            previous = None
            for label in labels:
                if label is None:
                    run = 0
                    previous = None
                    continue
                if previous is not None and label != previous:
                    walls += 1
                previous = label
                run = run + 1 if label == global_label else 0
                best = max(best, run)
            result = {
                "applicable": True,
                "reason": None,
                "global_label": global_label,
                "frac_consistent": counts[global_label] / len(resolved),
                "longest_run": int(best),
                "n_walls": int(walls),
            }
        per_chromosome.append({
            "chromosome_index": chromosome,
            "chromosome": name,
            "n_complete_20mb_fragments": len(detail),
            "n_resolved": len(resolved),
            "n_tied_or_unavailable": len(detail) - len(resolved),
            "n_terminal_bins_excluded": int(n % FRAGMENT_BINS),
            "detail": detail,
            **result,
        })
    applicable = [row for row in per_chromosome if row["applicable"]]
    return {
        "per_chromosome": per_chromosome,
        "n_chromosomes": len(per_chromosome),
        "n_applicable": len(applicable),
        "n_tied_or_unavailable": int(sum(not row["applicable"] for row in per_chromosome)),
        "mean_frac_consistent": (float(np.mean([row["frac_consistent"] for row in applicable]))
                                  if applicable else None),
        "mean_n_walls": (float(np.mean([row["n_walls"] for row in applicable]))
                         if applicable else None),
    }


def _collapse_coordinates(truth: np.ndarray) -> np.ndarray:
    midpoint = truth.mean(axis=0)
    return np.stack((midpoint, midpoint), axis=0)


def _random_field(data, seed: int = 7101) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return sphere_forward(rng.normal(0.0, 0.08, size=(2, data.n_loci, 3)))


def _fragment_swap_coordinates(truth: np.ndarray, data) -> np.ndarray:
    out = truth.copy()
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        n = slc.stop - slc.start
        for local_start in range(0, n, FRAGMENT_BINS):
            if (local_start // FRAGMENT_BINS) % 2:
                start = slc.start + local_start
                stop = min(slc.start + local_start + FRAGMENT_BINS, slc.stop)
                out[:, start:stop] = out[::-1, start:stop]
    return out


def _swap_one_chromosome(coords: np.ndarray, data, chromosome: int) -> np.ndarray:
    out = coords.copy()
    slc = data.chromosome_slice(chromosome)
    out[:, slc] = out[::-1, slc]
    return out


def _copy_shape_diagnostics(truth: np.ndarray, data) -> dict:
    rows = []
    for chromosome, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(chromosome)
        first = _pair_distances(truth[0], slc)
        second = _pair_distances(truth[1], slc)
        rows.append({
            "chromosome": name,
            "max_abs_internal_distance_difference": float(np.max(np.abs(first - second))),
            "internal_distance_spearman": spearman(first, second),
        })
    return {
        "per_chromosome": rows,
        "max_abs_internal_distance_difference": float(max(
            row["max_abs_internal_distance_difference"] for row in rows)),
    }


def _control_count_terms(data, truth: np.ndarray, p: float) -> dict:
    objective = JointObjective(data)
    controls = {
        "truth": truth,
        "collapse": _collapse_coordinates(truth),
        "random_field": _random_field(data),
        "fragment_swap": _fragment_swap_coordinates(truth, data),
    }
    terms = {name: objective.count_components_for_coordinates(coords, p)
             for name, coords in controls.items()}
    per_chromosome_swap_delta = []
    base = terms["truth"]["count_nll_normalized"]
    for chromosome in range(len(data.chromosome_names)):
        swapped = _swap_one_chromosome(truth, data, chromosome)
        term = objective.count_components_for_coordinates(swapped, p)["count_nll_normalized"]
        per_chromosome_swap_delta.append(float(term - base))
    return {
        "count_terms": {name: value["count_nll_normalized"] for name, value in terms.items()},
        "true_not_worse_than_fixed_controls": bool(all(
            terms["truth"]["count_nll_normalized"] <= terms[name]["count_nll_normalized"] + 1e-8
            for name in ("collapse", "random_field", "fragment_swap"))),
        "max_abs_per_chromosome_copy_swap_delta_per_record": float(
            max(abs(value) for value in per_chromosome_swap_delta)),
        "per_chromosome_copy_swap_delta_per_record": per_chromosome_swap_delta,
        "global_copy_swap_data_tolerance_per_record": 1e-8,
    }


def _load_candidate_after_hash(run_dir: Path, training: Mapping[str, Any]) -> np.ndarray:
    relative = training["candidate_coordinate_npz"]
    path = run_dir / relative
    actual = sha256_file(path)
    if actual != training["candidate_coordinate_npz_sha256"]:
        raise CalibrationError("candidate coordinate hash changed before evaluation")
    with np.load(path, allow_pickle=False) as payload:
        coordinates = payload["coordinates"].copy()
    assert_inside_unit_ball(coordinates)
    return coordinates


def evaluate_run(run_dir: str | Path) -> dict:
    """Perform isolated synthetic-truth evaluation only after coordinate hashes."""
    run_dir = Path(run_dir)
    manifest = verify_frozen_run_manifest(run_dir)
    results = {}
    for condition, spec in CONDITIONS.items():
        training_path = run_dir / "work" / (condition + "_training.json")
        if not training_path.is_file():
            raise CalibrationError("missing completed worker result for condition %s" % condition)
        with open(training_path, "rt") as handle:
            training = json.load(handle)
        if training.get("status") != "completed":
            raise CalibrationError("worker was not completed for condition %s" % condition)
        candidate = _load_candidate_after_hash(run_dir, training)
        data = load_layer(_condition_paths(run_dir, condition, FINAL_BIN))
        truth, _exposure, truth_metadata = _load_truth(_truth_path(run_dir, condition))
        objective = JointObjective(data)
        candidate_count = objective.count_components_for_coordinates(candidate, float(training["layers"][-1]["p"]))
        truth_count = objective.count_components_for_coordinates(truth, P_GENERATING)
        initial_path = run_dir / "work" / (condition + "_independent_init_5mb.npz")
        initial_5m, positions, chromosome_index, _seed = _load_initialization(initial_path)
        warm_initial = warm_start_from_layer(
            initial_5m, positions, chromosome_index,
            data.chromosome_names, data.chromosome_lengths, FINAL_BIN,
            candidate_base_seed=INIT_SEED,
        )["coords"]
        initial_count = objective.count_components_for_coordinates(warm_initial, P_INITIAL)
        controls = _control_count_terms(data, truth, P_GENERATING)
        r2 = synthetic_r2(candidate, truth, data)
        r3 = synthetic_r3(candidate, truth, data)
        initial_r2 = synthetic_r2(warm_initial, truth, data)
        initial_r3 = synthetic_r3(warm_initial, truth, data)
        result = {
            "condition": condition,
            "condition_name": spec["name"],
            "candidate_coordinate_sha256": training["candidate_coordinate_npz_sha256"],
            "synthetic_truth_sha256": sha256_file(_truth_path(run_dir, condition)),
            "numeric_gates": {
                "all_layers_nonincreasing": bool(all(
                    layer["final_not_worse_than_initial"] for layer in training["layers"])),
                "candidate_finite": bool(np.all(np.isfinite(candidate))),
                "candidate_strict_unit_ball": bool(np.linalg.norm(candidate, axis=2).max() < 1.0),
                "input_budget": data.budget(),
                "control_data_symmetry_pass": bool(
                    controls["max_abs_per_chromosome_copy_swap_delta_per_record"] <= 1e-8),
                "noiseless_truth_control_pass": (
                    controls["true_not_worse_than_fixed_controls"] if condition == "A" else None),
            },
            "count_and_prior": {
                "candidate_count_nll_normalized": candidate_count["count_nll_normalized"],
                "truth_count_nll_normalized_at_p_gen": truth_count["count_nll_normalized"],
                "initial_count_nll_normalized_at_p_init": initial_count["count_nll_normalized"],
                "candidate_minus_truth_count_nll": (
                    candidate_count["count_nll_normalized"] - truth_count["count_nll_normalized"]),
                "truth_minus_candidate_noise_gain_diagnostic": (
                    truth_count["count_nll_normalized"] - candidate_count["count_nll_normalized"]),
                "candidate_final_components": training["layers"][-1]["components"],
                "truth_metadata": truth_metadata,
            },
            "controls": controls,
            "r2": r2,
            "r3": r3,
            "initial_r2": initial_r2,
            "initial_r3": initial_r3,
            "same_shape_null_diagnostics": (
                _copy_shape_diagnostics(truth, data) if condition == "D" else None),
            "interpretation_boundary": (
                "synthetic calibration only; no R1 because synthetic record-copy labels were not generated; "
                "R2/R3 and count gain do not establish biological L2 recovery"
            ),
        }
        results[condition] = result
        _write_json(run_dir / "work" / (condition + "_evaluation.json"), result)
    summary = {
        "frozen_manifest": manifest,
        "conditions": results,
        "capture_misspecification_comparison": {
            "known_sampling_B_candidate_minus_truth": results["B"]["count_and_prior"]["candidate_minus_truth_count_nll"],
            "capture_misspecified_C_candidate_minus_truth": results["C"]["count_and_prior"]["candidate_minus_truth_count_nll"],
            "note": "difference is a joint kernel-plus-exposure misspecification sensitivity, not an attribution",
        },
        "biological_replicates": 1,
        "calibration_conditions_are_not_biological_replicates": True,
    }
    _write_json(run_dir / "metrics.json", summary)
    _append_jsonl(run_dir / "logs" / "status.jsonl", {
        "event": "synthetic_evaluation_completed",
        "conditions": list(CONDITIONS),
    })
    return summary


def write_summary_tsv(run_dir: str | Path) -> Path:
    """Write a compact non-graphical plot-table artifact for the formal run."""
    run_dir = Path(run_dir)
    with open(run_dir / "metrics.json", "rt") as handle:
        metrics = json.load(handle)
    path = run_dir / "plots" / "calibration_summary.tsv"
    with open(path, "wt") as handle:
        handle.write("condition\tcount_mode\tfinal_total\tcandidate_minus_truth_count_nll\t"
                     "r2_mean_contrast\tr3_n_applicable\tr3_mean_frac_consistent\n")
        for condition, result in metrics["conditions"].items():
            final_components = result["count_and_prior"]["candidate_final_components"]
            handle.write("%s\t%s\t%.17g\t%.17g\t%s\t%d\t%s\n" % (
                condition,
                CONDITIONS[condition]["count_mode"],
                final_components["total"],
                result["count_and_prior"]["candidate_minus_truth_count_nll"],
                ("" if result["r2"]["mean_contrast"] is None
                 else "%.17g" % result["r2"]["mean_contrast"]),
                result["r3"]["n_applicable"],
                ("" if result["r3"]["mean_frac_consistent"] is None
                 else "%.17g" % result["r3"]["mean_frac_consistent"]),
            ))
    return path


def _main() -> None:
    parser = argparse.ArgumentParser(description="P3 synthetic calibration runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "worker", "evaluate", "summary"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-dir", required=True)
        if command == "worker":
            child.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_run(args.run_dir)
    elif args.command == "worker":
        result = run_worker(args.run_dir, args.condition)
    elif args.command == "evaluate":
        result = evaluate_run(args.run_dir)
    else:
        result = {"summary_tsv": str(write_summary_tsv(args.run_dir))}
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
