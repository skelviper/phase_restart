#!/usr/bin/env python
"""036 C0 random 1 Mb checkpoint 的无重拟合后处理。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pr import contact_model

OUT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = ROOT / "test_res/036-20260914T064651Z-gpu-multires/checkpoints/C0/random_joint"
PUBLISHED_COORDS = ROOT / "test_res/036-20260914T064651Z-gpu-multires/coords/C0/random_joint/final-1m.3dg"
SNPFREE = ROOT / "inputs/P9016.snpfree.pairs.gz"
BIN_SIZE = 1_000_000
EXPECTED_ITERS = list(range(10, 241, 10))
EXPECTED_SHAPE = (2, 2645, 3)
EXPECTED_THETA = 6 * 2645 + 1
EPSILON = 1e-6
BLOCK_SIZE = 65_536
SERIALIZATION_TOL = 5e-15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint_iteration(path: Path) -> int:
    stem = path.stem
    return int(stem.rsplit("-", 1)[1])


def load_checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], int]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"coordinates", "theta", "y", "positions", "chromosome_index", "fullhistory_json"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{path} missing checkpoint fields: {sorted(missing)}")
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        theta = np.asarray(payload["theta"], dtype=np.float64).copy()
        raw_y = np.asarray(payload["y"], dtype=np.float64).copy()
        positions = np.asarray(payload["positions"], dtype=np.int64).copy()
        chromosome_index = np.asarray(payload["chromosome_index"], dtype=np.int64).copy()
        history = json.loads(str(payload["fullhistory_json"].item()))
    if coordinates.shape != EXPECTED_SHAPE or raw_y.shape != EXPECTED_SHAPE:
        raise ValueError(f"{path} coordinate/y shape mismatch: {coordinates.shape} / {raw_y.shape}")
    if theta.shape != (EXPECTED_THETA,):
        raise ValueError(f"{path} theta shape mismatch: {theta.shape}")
    if positions.shape != (2645,) or chromosome_index.shape != (2645,):
        raise ValueError(f"{path} grid metadata shape mismatch")
    if not all(np.all(np.isfinite(a)) for a in (coordinates, theta, raw_y)):
        raise ValueError(f"{path} contains non-finite coordinate state")
    iteration = checkpoint_iteration(path)
    entries = [entry for entry in history if int(entry.get("iteration", -1)) == iteration]
    if len(entries) != 1:
        raise ValueError(f"{path} history has {len(entries)} entries for iteration {iteration}")
    entry = entries[0]
    components = entry.get("components")
    if not isinstance(components, dict):
        raise ValueError(f"{path} history entry lacks components")
    return coordinates, theta, raw_y, positions, chromosome_index, entry, components, len(history)


def parse_published_coordinates(data: contact_model.AggregatedContacts) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.empty((2, data.n_loci, 3), dtype=np.float64)
    result.fill(np.nan)
    expected_rows = 0
    seen: set[tuple[str, int]] = set()
    track_to_copy_locus: dict[str, tuple[int, slice]] = {}
    for spec in data.track_specs:
        track_to_copy_locus[spec.name] = (spec.copy_index, data.chromosome_slice(spec.chromosome_index))
    with PUBLISHED_COORDS.open("rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"published coordinate row {line_no} has {len(fields)} fields")
            track, position_text = fields[0], fields[1]
            if track not in track_to_copy_locus:
                raise ValueError(f"published coordinate row {line_no} has unknown track {track}")
            try:
                position = int(position_text)
                xyz = np.asarray([float(v) for v in fields[2:]], dtype=np.float64)
            except ValueError as exc:
                raise ValueError(f"published coordinate row {line_no} is not numeric") from exc
            copy_index, slc = track_to_copy_locus[track]
            local_bin = position // data.bin_size
            if position != local_bin * data.bin_size or local_bin < 0 or local_bin >= slc.stop - slc.start:
                raise ValueError(f"published coordinate row {line_no} has invalid position {position}")
            key = (track, position)
            if key in seen:
                raise ValueError(f"duplicate published coordinate row {line_no}: {key}")
            seen.add(key)
            result[copy_index, slc.start + local_bin] = xyz
            expected_rows += 1
    if expected_rows != 5290 or len(seen) != 5290 or not np.all(np.isfinite(result)):
        raise ValueError("published coordinate file is not a complete finite 40-track grid")
    return result, {"rows": expected_rows, "tracks": len(track_to_copy_locus), "unique_rows": len(seen)}


def posterior_weights_for_block(x: np.ndarray, p: float, i: np.ndarray, j: np.ndarray,
                                cis: np.ndarray, r0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    daa = np.sqrt(np.sum((x[0, i] - x[0, j]) ** 2, axis=1))
    dab = np.sqrt(np.sum((x[0, i] - x[1, j]) ** 2, axis=1))
    dba = np.sqrt(np.sum((x[1, i] - x[0, j]) ** 2, axis=1))
    dbb = np.sqrt(np.sum((x[1, i] - x[1, j]) ** 2, axis=1))
    scale = r0 * r0
    kaa = EPSILON + (1.0 - EPSILON) * (1.0 + np.sum((x[0, i] - x[0, j]) ** 2, axis=1) / scale) ** -2
    kab = EPSILON + (1.0 - EPSILON) * (1.0 + np.sum((x[0, i] - x[1, j]) ** 2, axis=1) / scale) ** -2
    kba = EPSILON + (1.0 - EPSILON) * (1.0 + np.sum((x[1, i] - x[0, j]) ** 2, axis=1) / scale) ** -2
    kbb = EPSILON + (1.0 - EPSILON) * (1.0 + np.sum((x[1, i] - x[1, j]) ** 2, axis=1) / scale) ** -2
    waa = np.where(cis, 0.5 * p, 0.25)
    wab = np.where(cis, 0.5 * (1.0 - p), 0.25)
    wba = wab
    wbb = waa
    normalizer = waa * kaa + wab * kab + wba * kba + wbb * kbb
    if not np.all(np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
        raise FloatingPointError("non-positive or non-finite posterior normalizer")
    paa = waa * kaa / normalizer
    pab = wab * kab / normalizer
    pba = wba * kba / normalizer
    pbb = wbb * kbb / normalizer
    return daa, dab, dba, dbb, paa, pab, pba, pbb


def build_fixed_weights(data: contact_model.AggregatedContacts, x: np.ndarray, p: float) -> np.ndarray:
    weights = np.empty((4, data.n_pairs), dtype=np.float64)
    for start in range(0, data.n_pairs, BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, data.n_pairs)
        _, _, _, _, paa, pab, pba, pbb = posterior_weights_for_block(
            x, p, data.pair_i[start:stop], data.pair_j[start:stop],
            data.cis_pair[start:stop], data.r0)
        weights[:, start:stop] = (paa, pab, pba, pbb)
    return weights


def contact_distances(data: contact_model.AggregatedContacts, x: np.ndarray, p: float,
                      fixed_weights: np.ndarray | None) -> dict[str, float]:
    n_intra = 0
    n_inter = 0
    sums: dict[str, float] = {}
    for metric in ("posterior", "prior", "fixed"):
        sums[f"intra_{metric}"] = 0.0
        sums[f"inter_{metric}"] = 0.0
    for start in range(0, data.n_pairs, BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, data.n_pairs)
        i = data.pair_i[start:stop]
        j = data.pair_j[start:stop]
        cis = data.cis_pair[start:stop]
        count = data.counts[start:stop].astype(np.float64, copy=False)
        daa, dab, dba, dbb, paa, pab, pba, pbb = posterior_weights_for_block(x, p, i, j, cis, data.r0)
        prior_aa = np.where(cis, 0.5 * p, 0.25)
        prior_ab = np.where(cis, 0.5 * (1.0 - p), 0.25)
        post_distance = paa * daa + pab * dab + pba * dba + pbb * dbb
        prior_distance = prior_aa * daa + prior_ab * dab + prior_ab * dba + prior_aa * dbb
        if fixed_weights is None:
            fixed_distance = np.full_like(post_distance, np.nan)
        else:
            fixed_distance = (fixed_weights[0, start:stop] * daa
                              + fixed_weights[1, start:stop] * dab
                              + fixed_weights[2, start:stop] * dba
                              + fixed_weights[3, start:stop] * dbb)
        if not all(np.all(np.isfinite(v)) for v in (daa, dab, dba, dbb, post_distance, prior_distance, fixed_distance)):
            raise FloatingPointError("non-finite contact distance")
        cis_count = count[cis]
        inter_count = count[~cis]
        n_intra += int(np.sum(cis_count, dtype=np.int64))
        n_inter += int(np.sum(inter_count, dtype=np.int64))
        for metric, values in (("posterior", post_distance), ("prior", prior_distance), ("fixed", fixed_distance)):
            sums[f"intra_{metric}"] += float(np.dot(cis_count, values[cis]))
            sums[f"inter_{metric}"] += float(np.dot(inter_count, values[~cis]))
    if n_intra <= 0 or n_inter <= 0:
        raise ValueError("contact denominators are empty")
    result: dict[str, float] = {}
    for metric in ("posterior", "prior", "fixed"):
        result[f"distance_intra_{metric}"] = sums[f"intra_{metric}"] / n_intra
        result[f"distance_inter_{metric}"] = sums[f"inter_{metric}"] / n_inter
        result[f"distance_all_{metric}"] = (
            sums[f"intra_{metric}"] + sums[f"inter_{metric}"]) / (n_intra + n_inter)
    result["count_intra"] = float(n_intra)
    result["count_inter"] = float(n_inter)
    return result


def rg_and_center(x: np.ndarray) -> tuple[float, np.ndarray, float]:
    beads = x.reshape(-1, 3)
    center = beads.mean(axis=0)
    radius = np.sqrt(np.sum((beads - center) ** 2, axis=1))
    rg = float(np.sqrt(np.mean(radius ** 2)))
    return rg, center, float(np.max(radius))


def relative_percent(start: float, end: float) -> float:
    return float(100.0 * (end - start) / start) if start != 0.0 else float("nan")


def monotonic_flags(values: list[float], tol: float = 1e-12) -> dict[str, bool]:
    diffs = np.diff(np.asarray(values, dtype=np.float64))
    return {
        "nonincreasing": bool(np.all(diffs <= tol)),
        "nondecreasing": bool(np.all(diffs >= -tol)),
        "strictly_nonincreasing": bool(np.all(diffs < -tol)),
        "strictly_nondecreasing": bool(np.all(diffs > tol)),
    }


def make_plot(rows: list[dict[str, Any]]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "lines.linewidth": 1.0})
    iterations = [row["iteration"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=300)
    ax = axes[0]
    colors = {"intra": "#1f77b4", "inter": "#d62728"}
    for group in ("intra", "inter"):
        ax.plot(iterations, [row[f"distance_{group}_posterior"] for row in rows],
                color=colors[group], linestyle="-", label=f"{group} posterior")
        ax.plot(iterations, [row[f"distance_{group}_fixed"] for row in rows],
                color=colors[group], linestyle="--", label=f"{group} fixed iter10 posterior")
    ax.set_xlabel("Accepted iteration")
    ax.set_ylabel("Contact distance (model units)")
    ax.set_title("Contact distance")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, handlelength=2.2, loc="best")
    ax = axes[1]
    ax.plot(iterations, [row["Rg"] for row in rows], color="#2ca02c", linestyle="-", label="whole-cell Rg")
    ax.set_xlabel("Accepted iteration")
    ax.set_ylabel("Rg (model units)")
    ax.set_title("Whole-cell Rg")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, handlelength=2.2, loc="best")
    fig.tight_layout(pad=0.7, w_pad=1.2)
    path = OUT / "plots/contact_distance_rg.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> int:
    OUT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("plots").mkdir(parents=True, exist_ok=True)
    log_lines = []
    def log(message: str) -> None:
        log_lines.append(message)
        print(message, flush=True)

    checkpoint_paths = sorted(CHECKPOINT_ROOT.glob("1m-accepted-*.npz"), key=checkpoint_iteration)
    if [checkpoint_iteration(path) for path in checkpoint_paths] != EXPECTED_ITERS:
        raise ValueError("checkpoint set is not exactly accepted iterations 10..240")
    log(f"loading frozen SNP-free aggregation at {BIN_SIZE} bp")
    data = contact_model.load_frozen_p9016_aggregate(BIN_SIZE)
    audit = data.budget()
    expected_audit = {
        "raw_records": 1_703_888,
        "raw_same_bin": 438_774,
        "raw_cis_offdiag": 696_680,
        "raw_inter": 568_434,
        "aggregate_same_bin": 438_774,
        "aggregate_cis_offdiag": 696_680,
        "aggregate_inter": 568_434,
        "endpoint_total": 3_407_776,
        "n_loci": 2645,
        "n_chromosomes": 20,
    }
    for key, expected in expected_audit.items():
        if int(audit[key]) != expected:
            raise ValueError(f"budget mismatch for {key}: {audit[key]} != {expected}")
    log(f"raw budget verified: same-bin={audit['raw_same_bin']} intra-offdiag={audit['raw_cis_offdiag']} inter={audit['raw_inter']}")

    published, published_meta = parse_published_coordinates(data)
    rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    fixed_weights: np.ndarray | None = None
    fixed_weight_error = 0.0
    for index, checkpoint_path in enumerate(checkpoint_paths):
        iteration = checkpoint_iteration(checkpoint_path)
        x, theta, raw_y, positions, chromosome_index, history_entry, components, history_length = load_checkpoint(checkpoint_path)
        expected_positions = data.locus_bin.astype(np.int64) * BIN_SIZE
        if not np.array_equal(positions, expected_positions):
            raise ValueError(f"{checkpoint_path} positions are not the origin-0 1Mb grid")
        if not np.array_equal(chromosome_index, data.locus_chromosome.astype(np.int64)):
            raise ValueError(f"{checkpoint_path} chromosome_index does not match the full grid")
        transformed = contact_model.sphere_forward(raw_y)
        transform_error = float(np.max(np.abs(transformed - x)))
        if transform_error > SERIALIZATION_TOL:
            raise ValueError(f"{checkpoint_path} coordinate transform error {transform_error}")
        radius = np.linalg.norm(x, axis=-1)
        if not np.all(radius < 1.0) or not np.all(np.isfinite(radius)):
            raise ValueError(f"{checkpoint_path} physical coordinates leave strict unit ball")
        p, _ = contact_model.p_from_q(float(theta[-1]))
        if abs(float(history_entry["p"]) - p) > SERIALIZATION_TOL:
            raise ValueError(f"{checkpoint_path} p does not match theta")
        if index == 0:
            fixed_weights = build_fixed_weights(data, x, p)
            fixed_weight_error = float(np.max(np.abs(fixed_weights.sum(axis=0) - 1.0)))
            if fixed_weight_error > SERIALIZATION_TOL:
                raise ValueError(f"fixed iter10 posterior weights do not sum to one: {fixed_weight_error}")
        distances = contact_distances(data, x, p, fixed_weights)
        rg, center, max_bead_radius = rg_and_center(x)
        row: dict[str, Any] = {
            "iteration": iteration,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "p": float(p),
            "count_nll_normalized": float(components["count_nll_normalized"]),
            "count_nll_raw": float(components["count_nll_raw"]),
            "totalJ": float(components["total"]),
            "Rg": rg,
            "center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2]),
            "max_bead_radius": max_bead_radius,
            **distances,
        }
        for metric in ("posterior", "prior", "fixed"):
            for group in ("intra", "inter", "all"):
                row[f"distance_{group}_{metric}_over_Rg"] = row[f"distance_{group}_{metric}"] / rg
        if not all(math.isfinite(float(value)) for key, value in row.items() if key not in {"checkpoint_path", "checkpoint_sha256"}):
            raise ValueError(f"non-finite metric row at iteration {iteration}")
        rows.append(row)
        validation_rows.append({
            "iteration": iteration,
            "transform_max_abs": transform_error,
            "physical_max_radius": float(np.max(radius)),
            "theta_finite": bool(np.all(np.isfinite(theta))),
            "raw_y_finite": bool(np.all(np.isfinite(raw_y))),
            "coordinates_finite": bool(np.all(np.isfinite(x))),
            "history_length": history_length,
        })
        log(f"iteration {iteration}: posterior intra={row['distance_intra_posterior']:.8f} inter={row['distance_inter_posterior']:.8f} Rg={rg:.8f}")

    published_error = float(np.max(np.abs(x - published)))
    published_rms = float(np.sqrt(np.mean((x - published) ** 2)))
    published_match = bool(published_error <= SERIALIZATION_TOL)
    if not published_match:
        raise ValueError(f"last checkpoint differs from published coordinates by {published_error}")
    fieldnames = [
        "iteration", "checkpoint_path", "checkpoint_sha256", "p", "count_nll_normalized", "count_nll_raw", "totalJ",
        "distance_intra_posterior", "distance_inter_posterior", "distance_all_posterior",
        "distance_intra_prior", "distance_inter_prior", "distance_all_prior",
        "distance_intra_fixed", "distance_inter_fixed", "distance_all_fixed",
        "distance_intra_posterior_over_Rg", "distance_inter_posterior_over_Rg", "distance_all_posterior_over_Rg",
        "distance_intra_prior_over_Rg", "distance_inter_prior_over_Rg", "distance_all_prior_over_Rg",
        "distance_intra_fixed_over_Rg", "distance_inter_fixed_over_Rg", "distance_all_fixed_over_Rg",
        "Rg", "center_x", "center_y", "center_z", "max_bead_radius",
    ]
    with (OUT / "metrics.tsv").open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    plot_path = make_plot(rows)

    input_hashes = {
        "snpfree_pairs": {"path": str(SNPFREE), "sha256": sha256_file(SNPFREE)},
        "checkpoint_files": [{"path": str(path), "sha256": sha256_file(path)} for path in checkpoint_paths],
        "published_final_coordinates": {"path": str(PUBLISHED_COORDS), "sha256": sha256_file(PUBLISHED_COORDS)},
        "loader_sources": {
            "pr/contact_model.py": sha256_file(ROOT / "pr/contact_model.py"),
            "pr/genome.py": sha256_file(ROOT / "pr/genome.py"),
            "pr/pairs7.py": sha256_file(ROOT / "pr/pairs7.py"),
        },
    }
    primary_metrics = {}
    for metric in ("posterior", "prior", "fixed"):
        primary_metrics[metric] = {
            group: {
                "start": float(rows[0][f"distance_{group}_{metric}"]),
                "end": float(rows[-1][f"distance_{group}_{metric}"]),
                "change_percent": relative_percent(rows[0][f"distance_{group}_{metric}"], rows[-1][f"distance_{group}_{metric}"]),
                "monotonicity": monotonic_flags([row[f"distance_{group}_{metric}"] for row in rows]),
            }
            for group in ("intra", "inter", "all")
        }
    summary = {
        "run_id": OUT.name,
        "status": "completed_postprocessing",
        "fit_started": False,
        "model": "036 C0 random_joint 1Mb",
        "coordinate_units": "model dimensionless physical sphere coordinates",
        "source": {
            "checkpoint_root": str(CHECKPOINT_ROOT),
            "checkpoint_count": len(rows),
            "iterations": [int(row["iteration"]) for row in rows],
            "first_point_is": "accepted iteration 10; no fabricated iteration 0",
            "published_coordinate_manifest": str(PUBLISHED_COORDS),
        },
        "contact_definition": {
            "kernel": "K(d)=1e-6+(1-1e-6)*(1+d^2/r0^2)^-2",
            "r0": float(data.r0),
            "prior_weights": "intra AA/BB=p/2, AB/BA=(1-p)/2; inter all four=1/4",
            "posterior_weights": "w_ab*K(d_ab)/sum(w*K), inferred copy assignment",
            "primary": "counts-weighted posterior expected copy distance",
            "fixed_control": "iter10 posterior weights held fixed across all later geometry",
            "prior_control": "counts-weighted prior expected copy distance",
            "same_bin_excluded": True,
            "known_phase_oracle": False,
        },
        "raw_budget": {
            "all_records": 1_703_888,
            "same_bin_excluded": 438_774,
            "intra_offdiag": 696_680,
            "inter": 568_434,
            "distance_denominator": 1_265_114,
            "all_counts_weighted": True,
        },
        "Rg_definition": "sqrt(mean(||x_bead-mean(x_all_5290)||^2)); all 5290 physical beads equal weight; no per-step scaling",
        "primary_metrics": primary_metrics,
        "input_hashes": input_hashes,
        "artifacts": {
            "metrics_tsv": str(OUT / "metrics.tsv"),
            "plot_png": str(plot_path),
            "validation_json": str(OUT / "validation.json"),
        },
    }
    validation = {
        "status": "passed",
        "input_hashes": input_hashes,
        "checkpoint_schema": {
            "all_required_fields_present": True,
            "coordinates_shape": list(EXPECTED_SHAPE),
            "raw_y_shape": list(EXPECTED_SHAPE),
            "theta_length": EXPECTED_THETA,
            "stored_granularity": "accepted iterations every 10, 10..240",
        },
        "raw_budget": {
            "expected": expected_audit,
            "observed": {key: int(audit[key]) for key in expected_audit},
            "mask_and_rawcount_conserved": True,
            "distance_denominator": 1_265_114,
        },
        "finite_and_transform": {
            "all_rows_finite": True,
            "fixed_iter10_posterior_max_sum_error": fixed_weight_error,
            "serialization_tolerance": SERIALIZATION_TOL,
            "rows": validation_rows,
        },
        "last_checkpoint_vs_published": {
            "checkpoint": str(checkpoint_paths[-1]),
            "published": str(PUBLISHED_COORDS),
            "max_abs_difference": published_error,
            "rms_difference": published_rms,
            "published_rows": published_meta,
            "within_serialization_tolerance": published_match,
        },
        "no_forbidden_inputs_opened": ["reference", "phase", "evaluation outputs"],
    }
    json_dump(OUT / "summary.json", summary)
    json_dump(OUT / "validation.json", validation)
    config = {
        "run_id": OUT.name,
        "command": "conda run -n analysis python analyze_contact_rg.py",
        "purpose": "post-processing only; no refit and no objective reevaluation",
        "status": "completed_postprocessing",
        "data_boundary": {"training_input_only": True, "reference_opened": False, "phase_opened": False, "evaluation_opened": False},
        "parameters": {"bin_size_bp": BIN_SIZE, "block_size": BLOCK_SIZE, "epsilon": EPSILON, "r0": float(data.r0), "Rg_beads": 5290, "serialization_tolerance": SERIALIZATION_TOL},
        "input_hashes": input_hashes,
        "artifacts": {"metrics": "metrics.tsv", "summary": "summary.json", "validation": "validation.json", "plot": "plots/contact_distance_rg.png", "log": "logs/run.log", "script": "analyze_contact_rg.py"},
    }
    json_dump(OUT / "config.json", config)
    (OUT / "logs/run.log").write_text("\n".join(log_lines) + "\nSTATUS: completed_postprocessing\n", encoding="utf-8")
    log(f"published-coordinate max abs difference: {published_error:.3e}")
    log(f"wrote {OUT / 'metrics.tsv'}")
    log(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
