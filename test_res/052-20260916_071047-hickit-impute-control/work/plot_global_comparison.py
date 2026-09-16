#!/usr/bin/env python3
"""Render the one approved global comparison figure for run 052."""
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHROMS = ["chr%d" % i for i in range(1, 20)] + ["chrX"]
BIN = 1_000_000
OFF = 3_000_000


def read_coords(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    out = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 5:
                out.setdefault(fields[0], {})[int(fields[1])] = np.asarray(fields[2:5], dtype=float)
    return out


def lengths(path: Path):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#chromosome:"):
                _, chrom, length = line.split()
                out[chrom] = int(length)
            elif not line.startswith("#"):
                return out
    return out


def grid_values(coords, track, positions):
    return np.asarray([coords.get(track, {}).get(int(pos), [np.nan, np.nan, np.nan]) for pos in positions], dtype=float)


def collect(candidate, oracle, reference, raw, results, condition):
    raw_lengths = lengths(raw)
    candidate_points, reference_points, color_ids = [], [], []
    pairings = results["whole_genome_unit_rms_orthogonal_residual"][condition]["per_chromosome_pairing"]
    for index, chrom in enumerate(CHROMS, start=1):
        positions = np.arange(OFF, OFF + ((raw_lengths[chrom] - OFF) // BIN + 1) * BIN, BIN, dtype=int)
        imputed = (grid_values(candidate, chrom + "a", positions), grid_values(candidate, chrom + "b", positions))
        oracle_pair = (grid_values(oracle, "c%02da" % index, positions), grid_values(oracle, "c%02db" % index, positions))
        reference_pair = (grid_values(reference, chrom + "(mat)", positions), grid_values(reference, chrom + "(pat)", positions))
        support = np.ones(len(positions), dtype=bool)
        for values in imputed + oracle_pair + reference_pair:
            support &= np.isfinite(values).all(axis=1)
        left, right = (imputed if condition == "imputed" else oracle_pair)
        ref_left, ref_right = reference_pair if pairings[chrom] == "direct" else reference_pair[::-1]
        candidate_points.extend((left[support], right[support]))
        reference_points.extend((ref_left[support], ref_right[support]))
        color_ids.extend((np.full(support.sum(), index - 1), np.full(support.sum(), index - 1)))
    return np.vstack(candidate_points), np.vstack(reference_points), np.concatenate(color_ids)


def normalized_alignment(candidate, reference):
    x = candidate - candidate.mean(axis=0)
    y = reference - reference.mean(axis=0)
    x /= np.sqrt(np.mean(np.sum(x * x, axis=1)))
    y /= np.sqrt(np.mean(np.sum(y * y, axis=1)))
    u, _s, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    return x @ (u @ vt), y


def scatter(ax, points, color_ids, title, colors):
    for index, color in enumerate(colors):
        mask = color_ids == index
        ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], s=2.0, color=color, alpha=0.72, linewidths=0)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-62)


def main(run_text: str):
    run = Path(run_text)
    result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    candidate = read_coords(run / "coords/hickit-impute-p075.3dg")
    oracle = read_coords(Path("test_res/014-20260912_153000-s0-genome-wide-fixed/coords/oracle.3dg"))
    reference = read_coords(Path("data/P9016.1m.3dg.gz"))
    raw = Path("data/P9016.pairs.gz")
    imputed, ref_i, color_ids = collect(candidate, oracle, reference, raw, result, "imputed")
    oracle_points, ref_o, _ = collect(candidate, oracle, reference, raw, result, "oracle014")
    imputed_aligned, ref_i = normalized_alignment(imputed, ref_i)
    oracle_aligned, ref_o = normalized_alignment(oracle_points, ref_o)
    if not np.allclose(ref_i, ref_o, rtol=0.0, atol=1e-12):
        raise RuntimeError("candidate comparisons did not retain the same six-track reference support")
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7})
    figure = plt.figure(figsize=(9, 3))
    axes = [figure.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    scatter(axes[0], ref_i, color_ids, "Reference", colors)
    scatter(axes[1], oracle_aligned, color_ids, "014 oracle (labeled subset)", colors)
    scatter(axes[2], imputed_aligned, color_ids, "Hickit imputed split", colors)
    limit = max(float(np.abs(points).max()) for points in (ref_i, oracle_aligned, imputed_aligned)) * 1.04
    for axis in axes:
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    figure.text(0.5, 0.02, "One global orthogonal alignment per candidate; reflection allowed; shared six-track support", ha="center", va="bottom", fontsize=7)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.10, wspace=0.01)
    output = run / "plots/whole_genome_reference_oracle_imputed.png"
    figure.savefig(output, dpi=300)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: plot_global_comparison.py RUN_DIR")
    main(sys.argv[1])
