#!/usr/bin/env python3
"""Post-hash R2 and scale-free comparison for the 052 fixed Hickit arm."""
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CHROMS = ["chr%d" % value for value in range(1, 20)] + ["chrX"]
BIN = 1_000_000
OFF = 3_000_000
BOOTSTRAP_SEED = 37
BOOTSTRAP_DRAWS = 10_000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_coords(path: Path) -> dict[str, dict[int, np.ndarray]]:
    opener = gzip.open if path.suffix == ".gz" else open
    out: dict[str, dict[int, np.ndarray]] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 5:
                out.setdefault(fields[0], {})[int(fields[1])] = np.asarray(fields[2:5], dtype=float)
    return out


def raw_lengths(path: Path) -> dict[str, int]:
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#chromosome:"):
                _, name, length = line.split()
                out[name] = int(length)
            elif not line.startswith("#"):
                break
    if sorted(out) != sorted(CHROMS):
        raise RuntimeError("raw chromosome declaration does not match frozen 20 chromosomes")
    return out


def values(coords: dict[str, dict[int, np.ndarray]], name: str, positions: np.ndarray) -> np.ndarray:
    return np.asarray([coords.get(name, {}).get(int(pos), [np.nan, np.nan, np.nan]) for pos in positions], dtype=float)


def rho(left: np.ndarray, right: np.ndarray) -> float | None:
    value = spearmanr(left, right).statistic
    return None if not math.isfinite(value) else float(value)


def r2_from_support(a: np.ndarray, b: np.ndarray, mat: np.ndarray, pat: np.ndarray) -> dict:
    n = len(a)
    ii, jj = np.triu_indices(n, k=1)
    da = np.linalg.norm(a[ii] - a[jj], axis=1)
    db = np.linalg.norm(b[ii] - b[jj], axis=1)
    dm = np.linalg.norm(mat[ii] - mat[jj], axis=1)
    dp = np.linalg.norm(pat[ii] - pat[jj], axis=1)
    direct_parts = [rho(da, dm), rho(db, dp)]
    swapped_parts = [rho(da, dp), rho(db, dm)]
    if any(value is None for value in direct_parts + swapped_parts):
        return {"applicable": False, "n_common_beads": n, "n_common_distance_pairs": len(ii)}
    direct = float(np.mean(direct_parts))
    swapped = float(np.mean(swapped_parts))
    return {
        "applicable": True,
        "n_common_beads": n,
        "n_common_distance_pairs": int(len(ii)),
        "rho": {
            "copy_a_to_mat": direct_parts[0],
            "copy_b_to_pat": direct_parts[1],
            "copy_a_to_pat": swapped_parts[0],
            "copy_b_to_mat": swapped_parts[1]
        },
        "direct": direct,
        "swapped_assignment": swapped,
        "matched": max(direct, swapped),
        "swapped": min(direct, swapped),
        "contrast": abs(direct - swapped),
        "best_pairing": "direct" if direct > swapped else ("swapped" if swapped > direct else "tie")
    }


def rg(points: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1))))


def geometric_diagnostics(rows: list[dict], condition: str) -> dict:
    raw_rg = []
    homolog_ratio = []
    centroids = []
    for row in rows:
        arrays = row["arrays"]
        a, b = arrays[condition]
        rg_a, rg_b = rg(a), rg(b)
        mean_rg = (rg_a + rg_b) / 2.0
        raw_rg.extend([rg_a, rg_b])
        homolog_ratio.append(float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)) / mean_rg))
        centroids.append((a.mean(axis=0), b.mean(axis=0), mean_rg))
    cross = []
    for left in range(len(centroids)):
        for right in range(left + 1, len(centroids)):
            for c1 in centroids[left][:2]:
                for c2 in centroids[right][:2]:
                    cross.append(float(np.linalg.norm(c1 - c2) / ((centroids[left][2] + centroids[right][2]) / 2.0)))
    median_rg = float(np.median(raw_rg))
    return {
        "copy_rg": {"median": median_rg, "values": raw_rg, "normalized_by_genome_median": [value / median_rg for value in raw_rg]},
        "homolog_centroid_separation_over_mean_copy_rg": {"median": float(np.median(homolog_ratio)), "values": homolog_ratio},
        "interchromosome_copy_centroid_separation_over_mean_copy_rg": {"median": float(np.median(cross)), "n_copy_pairs": len(cross), "values": cross}
    }


def whole_genome_orthogonal_residual(rows: list[dict], condition: str) -> dict:
    candidate_points, reference_points = [], []
    pairings = {}
    for row in rows:
        arrays = row["arrays"]
        metrics = row["metrics"][condition]
        a, b = arrays[condition]
        mat, pat = arrays["reference"]
        pairing = metrics["best_pairing"]
        if pairing == "direct":
            candidate_points.extend((a, b))
            reference_points.extend((mat, pat))
        elif pairing == "swapped":
            candidate_points.extend((a, b))
            reference_points.extend((pat, mat))
        else:
            raise RuntimeError("whole-genome alignment cannot resolve copy tie for %s" % row["metrics"]["chromosome"])
        pairings[row["metrics"]["chromosome"]] = pairing
    x = np.vstack(candidate_points)
    y = np.vstack(reference_points)
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    x_rms = float(np.sqrt(np.mean(np.sum(x * x, axis=1))))
    y_rms = float(np.sqrt(np.mean(np.sum(y * y, axis=1))))
    if x_rms == 0.0 or y_rms == 0.0:
        raise RuntimeError("zero-RMS coordinates cannot be globally aligned")
    x /= x_rms
    y /= y_rms
    u, _singular, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    rotation_or_reflection = u @ vt
    residual = float(np.sqrt(np.mean(np.sum((x @ rotation_or_reflection - y) ** 2, axis=1))))
    return {"common_points": int(len(x)), "candidate_centered_rms_before_unit_normalization": x_rms, "reference_centered_rms_before_unit_normalization": y_rms, "unit_rms_optimal_orthogonal_residual": residual, "alignment": "one whole-genome orthogonal alignment after frozen per-chromosome copy matching; reflection allowed because distance constraints have no handedness", "orthogonal_determinant": float(np.linalg.det(rotation_or_reflection)), "per_chromosome_pairing": pairings, "interpretation": "auxiliary geometry diagnostic only; not used for fit, seed, threshold, or endpoint selection"}


def bootstrap(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = arr[rng.integers(0, len(arr), size=(BOOTSTRAP_DRAWS, len(arr)))].mean(axis=1)
    return {"n_chromosomes": int(len(arr)), "mean": float(arr.mean()), "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])], "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "interpretation": "technical/structural chromosome bootstrap within one cell; not biological replication or a p-value"}


def main(candidate_text: str, oracle_text: str, reference_text: str, raw_text: str, out_text: str) -> None:
    candidate_path, oracle_path = Path(candidate_text), Path(oracle_text)
    reference_path, raw_path, out_path = Path(reference_text), Path(raw_text), Path(out_text)
    candidate_sha_path = Path(str(candidate_path) + ".sha256")
    if not candidate_sha_path.exists():
        raise RuntimeError("candidate SHA256 record is required before reference access: %s" % candidate_sha_path)
    recorded_candidate_sha = candidate_sha_path.read_text(encoding="utf-8").split()[0]
    actual_candidate_sha = digest(candidate_path)
    if recorded_candidate_sha != actual_candidate_sha:
        raise RuntimeError("candidate SHA256 record does not match candidate coordinates")
    candidate = read_coords(candidate_path)
    oracle = read_coords(oracle_path)
    reference = read_coords(reference_path)
    lengths = raw_lengths(raw_path)
    rows = []
    per_chromosome = []
    for index, chrom in enumerate(CHROMS, start=1):
        positions = np.arange(OFF, OFF + ((lengths[chrom] - OFF) // BIN + 1) * BIN, BIN, dtype=int)
        imputed = (values(candidate, chrom + "a", positions), values(candidate, chrom + "b", positions))
        oracle_pair = (values(oracle, "c%02da" % index, positions), values(oracle, "c%02db" % index, positions))
        ref_pair = (values(reference, chrom + "(mat)", positions), values(reference, chrom + "(pat)", positions))
        support = np.ones(len(positions), dtype=bool)
        for array in imputed + oracle_pair + ref_pair:
            support &= np.isfinite(array).all(axis=1)
        row = {"chromosome": chrom, "n_grid_beads": int(len(positions)), "n_common_finite_beads": int(support.sum())}
        if support.sum() < 7:
            row.update({"status": "insufficient_common_finite_beads", "imputed": {"applicable": False}, "oracle014": {"applicable": False}})
            per_chromosome.append(row)
            continue
        arrays = {"imputed": (imputed[0][support], imputed[1][support]), "oracle014": (oracle_pair[0][support], oracle_pair[1][support]), "reference": (ref_pair[0][support], ref_pair[1][support])}
        row.update({"status": "applicable", "imputed": r2_from_support(*arrays["imputed"], *arrays["reference"]), "oracle014": r2_from_support(*arrays["oracle014"], *arrays["reference"])})
        if not (row["imputed"].get("applicable") and row["oracle014"].get("applicable")):
            row["status"] = "insufficient_or_nonfinite_distance_spearman"
            per_chromosome.append(row)
            continue
        rows.append({"arrays": arrays, "metrics": row})
        per_chromosome.append(row)
    if len(rows) != len(CHROMS):
        incomplete = {
            "status": "evaluation_failed_incomplete_common_support",
            "inputs": {"candidate": {"path": str(candidate_path), "sha256": actual_candidate_sha}, "candidate_sha_record": str(candidate_sha_path)},
            "policy": "all 20 frozen chromosomes must have applicable six-track common support; no silent denominator shrinkage",
            "per_chromosome": per_chromosome,
            "applicable_chromosomes": len(rows),
            "required_chromosomes": len(CHROMS)
        }
        out_path.write_text(json.dumps(incomplete, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError("incomplete common support: %d/%d chromosomes applicable" % (len(rows), len(CHROMS)))
    metrics = ["matched", "swapped", "contrast"]
    common_support_totals = {
        "sum_common_finite_beads": int(sum(row["metrics"]["imputed"]["n_common_beads"] for row in rows)),
        "sum_common_non_diagonal_distance_pairs": int(sum(row["metrics"]["imputed"]["n_common_distance_pairs"] for row in rows)),
        "counting_policy": "each chromosome position and each within-chromosome non-diagonal distance pair is counted once on the shared six-track support; copies are not separately duplicated"
    }
    paired = {}
    for metric in metrics:
        deltas = [row["metrics"]["imputed"][metric] - row["metrics"]["oracle014"][metric] for row in rows]
        entry = {"per_chromosome_delta_imputed_minus_014": deltas, "ties": int(sum(value == 0 for value in deltas)), "bootstrap": bootstrap(deltas)}
        if metric in {"matched", "contrast"}:
            entry.update({"wins_imputed": int(sum(value > 0 for value in deltas)), "losses_imputed": int(sum(value < 0 for value in deltas)), "interpretation": "positive delta is better agreement with reference"})
        else:
            entry.update({"higher_imputed": int(sum(value > 0 for value in deltas)), "lower_imputed": int(sum(value < 0 for value in deltas)), "interpretation": "cross/swap correlation is reported descriptively; higher is not a performance win"})
        paired[metric] = entry
    macro = {}
    for condition in ("imputed", "oracle014"):
        macro[condition] = {metric: float(np.mean([row["metrics"][condition][metric] for row in rows])) for metric in metrics}
    output = {
        "status": "evaluated_post_candidate_hash",
        "inputs": {"candidate": {"path": str(candidate_path), "sha256": digest(candidate_path)}, "oracle014": {"path": str(oracle_path), "sha256": digest(oracle_path)}, "reference": {"path": str(reference_path), "sha256": digest(reference_path)}},
        "policy": {"support": "per-chromosome six-track common finite 1Mb positions from 3Mb", "copy_matching": "whole-chromosome best of direct and swapped distance-Spearman", "metrics": "matched, swapped, and matched-minus-swapped contrast", "bootstrap": "paired imputed-minus-014 chromosome resampling", "014_note": "014 values are recomputed on this six-track common mask and can differ slightly from frozen 016 aggregate values"},
        "common_support_totals": common_support_totals,
        "per_chromosome": per_chromosome,
        "macro_mean": macro,
        "paired_imputed_minus_014": paired,
        "global_scale_rotation_invariant_diagnostics": {condition: geometric_diagnostics(rows, condition) for condition in ("reference", "oracle014", "imputed")},
        "whole_genome_unit_rms_orthogonal_residual": {condition: whole_genome_orthogonal_residual(rows, condition) for condition in ("oracle014", "imputed")},
        "limitations": ["one cell; bootstrap is not biological replication", "SNP-assisted imputation diagnostic, not blind recovery", "reference was not used for fit, seed, threshold, or endpoint selection", "fixed 1000-step budget does not establish native convergence"]
    }
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit("usage: evaluate_r2.py CANDIDATE ORACLE014 REFERENCE RAW OUT_JSON")
    main(*sys.argv[1:])
