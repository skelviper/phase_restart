#!/usr/bin/env python
"""仅供评估的 P9016 020 与 029 C0 报告。

本脚本不拟合、不依据 reference 进行选择，也不恢复 checkpoint。它先哈希所有 candidate coordinate endpoints，并在打开仅供评估的 reference 前检查已保存的 terminal provenance。
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 7,
    "pdf.fonttype": 42,
})


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
CONFIG_PATH = OUT / "config.json"
BIN = 1_000_000
OFFSET = 3_000_000
BOOTSTRAP_SEED = 9301
BOOTSTRAP_DRAWS = 10_000
TIE_TOL = 1e-12
CHROM_RE = re.compile(r"^c(\d\d)([ab])$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_grid() -> tuple[list[str], dict[str, int]]:
    source = ROOT / "test_res/020-20260913_071841-v1-p9016-joint/config.json"
    config = read_json(source)
    chroms = [item["name"] for item in config["coordinate_grid"]["chromosomes"]]
    lengths = {item["name"]: int(item["length_bp"])
               for item in config["coordinate_grid"]["chromosomes"]}
    if chroms != [*(f"chr{i}" for i in range(1, 20)), "chrX"]:
        raise RuntimeError(f"unexpected chromosome order: {chroms}")
    if len(chroms) != 20 or len(lengths) != 20:
        raise RuntimeError("expected exactly 20 chromosome headers")
    return chroms, lengths


def expected_candidate_positions(length_bp: int) -> set[int]:
    return set(range(0, int(length_bp), BIN))


def parse_candidate(path: Path, chroms: list[str], lengths: dict[str, int]) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    coords: dict[str, dict[int, np.ndarray]] = {}
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                raise RuntimeError(f"{path}: malformed coordinate row {line_no}")
            track = fields[0]
            match = CHROM_RE.match(track)
            if match is None:
                raise RuntimeError(f"{path}: invalid candidate track {track!r} at row {line_no}")
            chrom_index = int(match.group(1)) - 1
            copy_name = match.group(2)
            if chrom_index < 0 or chrom_index >= len(chroms):
                raise RuntimeError(f"{path}: chromosome index out of range at row {line_no}")
            chrom = chroms[chrom_index]
            position = int(fields[1])
            point = np.asarray([float(fields[2]), float(fields[3]), float(fields[4])], dtype=float)
            if position % BIN != 0 or position < 0 or position >= lengths[chrom]:
                raise RuntimeError(f"{path}: invalid 1Mb position {position} at row {line_no}")
            if not np.isfinite(point).all():
                raise RuntimeError(f"{path}: nonfinite endpoint at row {line_no}")
            target = coords.setdefault(track, {})
            if position in target:
                raise RuntimeError(f"{path}: duplicate endpoint {track}:{position}")
            target[position] = point
            row_count += 1
    expected_tracks = {f"c{i:02d}{copy}" for i in range(1, 21) for copy in "ab"}
    if set(coords) != expected_tracks:
        raise RuntimeError(f"{path}: expected 40 tracks, got {len(coords)}")
    endpoint_counts: dict[str, int] = {}
    missing_positions: dict[str, list[int]] = {}
    extra_positions: dict[str, list[int]] = {}
    for track, values in sorted(coords.items()):
        chrom = chroms[int(track[1:3]) - 1]
        expected = expected_candidate_positions(lengths[chrom])
        actual = set(values)
        endpoint_counts[track] = len(actual)
        missing_positions[track] = sorted(expected - actual)
        extra_positions[track] = sorted(actual - expected)
        if actual != expected:
            raise RuntimeError(f"{path}: {track} is not a complete full-grid endpoint set")
    all_points = np.concatenate([np.asarray(list(values.values())) for values in coords.values()])
    radii = np.linalg.norm(all_points, axis=1)
    audit = {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": row_count,
        "n_tracks": len(coords),
        "n_expected_tracks": 40,
        "endpoint_counts": endpoint_counts,
        "missing_positions": missing_positions,
        "extra_positions": extra_positions,
        "all_finite": bool(np.isfinite(all_points).all()),
        "max_radius": float(radii.max()),
        "unit_ball": bool(float(radii.max()) <= 1.0 + 1e-9),
        "full_grid": True,
    }
    if row_count != 5290 or not audit["all_finite"] or not audit["unit_ball"]:
        raise RuntimeError(f"{path}: endpoint audit failed: {audit}")
    return coords, audit


def parse_reference(path: Path) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    coords: dict[str, dict[int, np.ndarray]] = {}
    row_count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            track = fields[0]
            try:
                position = int(fields[1])
                point = np.asarray([float(fields[2]), float(fields[3]), float(fields[4])], dtype=float)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"reference malformed row {line_no}") from exc
            if not np.isfinite(point).all():
                raise RuntimeError(f"reference nonfinite endpoint at row {line_no}")
            target = coords.setdefault(track, {})
            if position in target:
                raise RuntimeError(f"reference duplicate endpoint {track}:{position}")
            target[position] = point
            row_count += 1
    expected = {f"chr{i}({copy})" for i in range(1, 20) for copy in ("mat", "pat")} | {"chrX(mat)", "chrX(pat)"}
    if not expected.issubset(coords):
        missing = sorted(expected - set(coords))
        raise RuntimeError(f"reference missing expected tracks: {missing}")
    return coords, {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": row_count,
        "n_tracks": len(coords),
        "expected_tracks_present": len(expected),
        "all_expected_tracks_present": True,
    }


def full_positions(length_bp: int) -> np.ndarray:
    return np.arange(OFFSET, int(length_bp), BIN, dtype=np.int64)


def candidate_prefix(chrom: str) -> str:
    index = 20 if chrom == "chrX" else int(chrom[3:])
    return f"c{index:02d}"


def finite_bins(track: dict[int, np.ndarray], positions: np.ndarray) -> list[int]:
    return [int(position) for position in positions
            if int(position) in track and np.isfinite(track[int(position)]).all()]


def distance_matrix(coords: dict[int, np.ndarray], positions: list[int]) -> np.ndarray:
    points = np.asarray([coords[position] for position in positions], dtype=float)
    delta = points[:, None, :] - points[None, :, :]
    distances = np.sqrt((delta * delta).sum(axis=2))
    np.fill_diagonal(distances, np.nan)
    return distances


def strict_upper(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    return matrix[np.triu_indices(n, k=1)]


def rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 20 or len(x) != len(y) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    value = spearmanr(x, y).statistic
    return float(value) if finite_number(value) else float("nan")


def score_method(method: str, track_a: str, track_b: str,
                 candidate: dict[str, dict[int, np.ndarray]],
                 ref_mat: dict[int, np.ndarray], ref_pat: dict[int, np.ndarray],
                 positions: list[int], full_n_bins: int,
                 main_common_n_pairs: int, supplemental: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    if not all(position in candidate.get(track_a, {}) and position in candidate.get(track_b, {})
               for position in positions):
        row = {
            "method": method, "n_bins": len(positions), "n_pairs": main_common_n_pairs,
            "n_common_pairs": main_common_n_pairs, "full_grid_bins": full_n_bins,
            "common_bins": len(positions), "supplemental": supplemental,
            "metric_status": "unavailable_missing_candidate_bins",
        }
        return row, {}
    d_a = distance_matrix(candidate[track_a], positions)
    d_b = distance_matrix(candidate[track_b], positions)
    d_mat = distance_matrix(ref_mat, positions)
    d_pat = distance_matrix(ref_pat, positions)
    a_mat = rho(strict_upper(d_a), strict_upper(d_mat))
    a_pat = rho(strict_upper(d_a), strict_upper(d_pat))
    b_mat = rho(strict_upper(d_b), strict_upper(d_mat))
    b_pat = rho(strict_upper(d_b), strict_upper(d_pat))
    raw = {"rho_a_mat": a_mat, "rho_a_pat": a_pat, "rho_b_mat": b_mat, "rho_b_pat": b_pat}
    row: dict[str, Any] = {
        "method": method,
        "track_a": track_a,
        "track_b": track_b,
        "n_bins": len(positions),
        "n_pairs": main_common_n_pairs,
        "n_common_pairs": main_common_n_pairs,
        "full_grid_bins": full_n_bins,
        "common_bins": len(positions),
        "supplemental": supplemental,
        **raw,
    }
    values = list(raw.values())
    if not all(finite_number(value) for value in values):
        row.update({"metric_status": "unavailable_nonfinite_rho", "direct_original": None,
                    "cross_original": None, "matched": None, "cross_matched": None,
                    "contrast": None, "margin_mat": None, "margin_pat": None,
                    "minmargin": None, "swap": None, "orientation": "unresolved"})
        return row, {"d_a": d_a, "d_b": d_b, "d_mat": d_mat, "d_pat": d_pat}
    direct = (a_mat + b_pat) / 2.0
    cross = (a_pat + b_mat) / 2.0
    row["direct_original"] = float(direct)
    row["cross_original"] = float(cross)
    if abs(direct - cross) <= TIE_TOL:
        row.update({
            "matched": float((direct + cross) / 2.0),
            "cross_matched": float((direct + cross) / 2.0),
            "contrast": 0.0,
            "margin_mat": None,
            "margin_pat": None,
            "minmargin": None,
            "swap": None,
            "orientation": "tie_average",
            "matched_mat_copy": None,
            "matched_pat_copy": None,
            "metric_status": "ok",
        })
    elif direct > cross:
        row.update({
            "matched": float(direct),
            "cross_matched": float(cross),
            "contrast": float(direct - cross),
            "margin_mat": float(a_mat - a_pat),
            "margin_pat": float(b_pat - b_mat),
            "minmargin": float(min(a_mat - a_pat, b_pat - b_mat)),
            "swap": False,
            "orientation": "direct",
            "matched_mat_copy": "a",
            "matched_pat_copy": "b",
            "metric_status": "ok",
        })
    else:
        # 只交换 candidate rows；reference mat/pat columns 保持固定。
        row.update({
            "matched": float(cross),
            "cross_matched": float(direct),
            "contrast": float(cross - direct),
            "margin_mat": float(b_mat - b_pat),
            "margin_pat": float(a_pat - a_mat),
            "minmargin": float(min(b_mat - b_pat, a_pat - a_mat)),
            "swap": True,
            "orientation": "swapped",
            "matched_mat_copy": "b",
            "matched_pat_copy": "a",
            "metric_status": "ok",
        })
    return row, {"d_a": d_a, "d_b": d_b, "d_mat": d_mat, "d_pat": d_pat}


def per_copy_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    if row.get("metric_status") != "ok":
        return []
    swap = row.get("swap") is True
    if swap:
        mappings = [("a", row["track_a"], "pat", row["rho_a_pat"], row["rho_a_mat"]),
                    ("b", row["track_b"], "mat", row["rho_b_mat"], row["rho_b_pat"])]
    else:
        mappings = [("a", row["track_a"], "mat", row["rho_a_mat"], row["rho_a_pat"]),
                    ("b", row["track_b"], "pat", row["rho_b_pat"], row["rho_b_mat"])]
    return [{
        "chromosome": row["chromosome"],
        "method": row["method"],
        "copy": copy,
        "candidate_track": track,
        "reference_copy": reference_copy,
        "spearman": float(matched),
        "spearman_other_reference": float(other),
        "n_bins": row["n_bins"],
        "n_pairs": row["n_pairs"],
        "swap": row["swap"],
        "orientation": row["orientation"],
    } for copy, track, reference_copy, matched, other in mappings]


def bootstrap(values: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    draws = values[indices].mean(axis=1)
    return {
        "n_chromosomes": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [float(item) for item in np.percentile(draws, [2.5, 97.5])],
        "seed": BOOTSTRAP_SEED,
        "n_boot": BOOTSTRAP_DRAWS,
        "unit": "chromosome",
        "interpretation": "technical/structural variation within one P9016 cell; not biological replication or a p-value",
    }


def write_tsv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t",
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: "" if row.get(column) is None else jsonable(row.get(column))
                             for column in columns})


def plot_two_boxes(path_stem: Path, values_by_method: list[tuple[str, np.ndarray]],
                   chroms: list[str], y_label: str, title: str) -> dict[str, Any]:
    colors = ["#2878b5", "#d77a1f"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    jitter = rng.uniform(-0.065, 0.065, size=(len(chroms), len(values_by_method)))
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    data = [np.asarray(values, dtype=float) for _, values in values_by_method]
    box = ax.boxplot(data, positions=np.arange(1, len(data) + 1), widths=0.46,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": "black", "linewidth": 0.9},
                     whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7},
                     boxprops={"linewidth": 0.7})
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(colors[index])
        patch.set_alpha(0.58)
    for chrom_index, chrom in enumerate(chroms):
        x_values = []
        y_values = []
        for method_index, (_, values) in enumerate(values_by_method):
            value = float(values[chrom_index])
            x = method_index + 1 + float(jitter[chrom_index, method_index])
            x_values.append(x)
            y_values.append(value)
            is_chr1 = chrom == "chr1"
            ax.scatter([x], [value], s=15 if is_chr1 else 10,
                       marker="D" if is_chr1 else "o",
                       facecolor=colors[method_index],
                       edgecolor="#b2182b" if is_chr1 else "white",
                       linewidth=0.8 if is_chr1 else 0.25, zorder=3)
        ax.plot(x_values, y_values, color="#999999", linewidth=0.55, alpha=0.65, zorder=1)
    all_values = np.concatenate(data)
    span = max(float(np.ptp(all_values)), 0.05)
    lower = float(all_values.min() - 0.10 * span)
    upper = float(all_values.max() + 0.14 * span)
    ax.set_ylim(lower, upper)
    ax.set_xticks([1, 2], [name for name, _ in values_by_method])
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y", color="#dddddd", linewidth=0.4, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.99, 0.02, "chr1 = red diamond", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#7f1d1d")
    fig.suptitle("P9016 | 20 chromosomes, one cell", fontsize=7, y=0.995)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.88)
    paths = {}
    for extension in ("png", "pdf"):
        target = path_stem.with_suffix("." + extension)
        fig.savefig(target, dpi=300)
        paths[extension] = str(target)
    plt.close(fig)
    return {"png": paths["png"], "pdf": paths["pdf"], "figsize_inches": [3.0, 3.0],
            "dpi": 300, "jitter_seed": BOOTSTRAP_SEED, "chr1_marker": "red diamond"}


def plot_per_copy(path_stem: Path, per_copy: list[dict[str, Any]], chroms: list[str]) -> dict[str, Any]:
    colors = ["#2878b5", "#d77a1f"]
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), squeeze=False)
    axes = axes[0]
    for panel_index, reference_copy in enumerate(("mat", "pat")):
        ax = axes[panel_index]
        grouped = []
        for method in ("020 multires", "029 C0 direct 1Mb"):
            lookup = {(row["chromosome"], row["method"]): row["spearman"]
                      for row in per_copy if row["reference_copy"] == reference_copy}
            grouped.append(np.asarray([lookup[(chrom, method)] for chrom in chroms], dtype=float))
        box = ax.boxplot(grouped, positions=[1, 2], widths=0.46, patch_artist=True,
                         showfliers=False, medianprops={"color": "black", "linewidth": 0.9},
                         whiskerprops={"linewidth": 0.7}, capprops={"linewidth": 0.7},
                         boxprops={"linewidth": 0.7})
        for index, patch in enumerate(box["boxes"]):
            patch.set_facecolor(colors[index])
            patch.set_alpha(0.58)
        jitter = rng.uniform(-0.065, 0.065, size=(len(chroms), 2))
        for chrom_index, chrom in enumerate(chroms):
            xs, ys = [], []
            for method_index, values in enumerate(grouped):
                value = float(values[chrom_index])
                x = method_index + 1 + float(jitter[chrom_index, method_index])
                xs.append(x); ys.append(value)
                is_chr1 = chrom == "chr1"
                ax.scatter([x], [value], s=15 if is_chr1 else 10,
                           marker="D" if is_chr1 else "o", facecolor=colors[method_index],
                           edgecolor="#b2182b" if is_chr1 else "white",
                           linewidth=0.8 if is_chr1 else 0.25, zorder=3)
            ax.plot(xs, ys, color="#999999", linewidth=0.55, alpha=0.65, zorder=1)
        all_values = np.concatenate(grouped)
        span = max(float(np.ptp(all_values)), 0.05)
        ax.set_ylim(float(all_values.min() - 0.10 * span), float(all_values.max() + 0.14 * span))
        ax.set_xticks([1, 2], ["020", "029 C0"])
        ax.set_title(f"reference {reference_copy}-matched copy", fontsize=7)
        ax.set_ylabel("Spearman rho" if panel_index == 0 else "")
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(axis="y", color="#dddddd", linewidth=0.4, alpha=0.7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("P9016 | per-reference-copy matched rho", fontsize=7, y=0.995)
    fig.text(0.5, 0.012, "Each point = chromosome; red diamond = chr1", ha="center", fontsize=7)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.84, wspace=0.28)
    paths = {}
    for extension in ("png", "pdf"):
        target = path_stem.with_suffix("." + extension)
        fig.savefig(target, dpi=300)
        paths[extension] = str(target)
    plt.close(fig)
    return {"png": paths["png"], "pdf": paths["pdf"], "figsize_inches": [6.0, 3.0],
            "dpi": 300, "jitter_seed": BOOTSTRAP_SEED + 1}


def render_heatmaps(path_stem: Path, maps: list[dict[str, Any]], full_positions: np.ndarray,
                    scale_info: dict[str, Any], row_labels: list[str], col_labels: list[str]) -> dict[str, Any]:
    matrices = [np.asarray(item["normalized"], dtype=float) for item in maps]
    finite_values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if not vmax > vmin:
        vmax = vmin + 1.0
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad("#d9d9d9")
    fig, axes = plt.subplots(3, 2, figsize=(6.0, 9.0), squeeze=False)
    extent = (float(full_positions[0] / 1e6 - 0.5),
              float(full_positions[-1] / 1e6 + 0.5),
              float(full_positions[0] / 1e6 - 0.5),
              float(full_positions[-1] / 1e6 + 0.5))
    ticks = [3, 50, 100, 150, 195]
    images = []
    for index, item in enumerate(maps):
        row = index // 2
        col = index % 2
        ax = axes[row][col]
        image = ax.imshow(np.ma.masked_invalid(item["normalized"]), origin="lower",
                          interpolation="none", aspect="equal", extent=extent,
                          cmap=cmap, vmin=vmin, vmax=vmax)
        images.append(image)
        ax.set_title(col_labels[col] if row == 0 else item["display_label"], fontsize=7)
        if col == 0:
            ax.set_ylabel(row_labels[row] + "\nposition (Mb)", fontsize=7)
        else:
            ax.set_ylabel("")
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.tick_params(axis="both", labelsize=7)
        if col == 1:
            ax.tick_params(axis="y", labelleft=False)
        if row == 2:
            ax.set_xlabel("position (Mb)", fontsize=7)
        else:
            ax.set_xlabel("")
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    # 预留专用 axes，使 colorbar 不会遮挡右侧图。
    fig.subplots_adjust(left=0.11, right=0.84, bottom=0.06, top=0.96, hspace=0.18, wspace=0.10)
    cax = fig.add_axes([0.87, 0.25, 0.025, 0.50])
    colorbar = fig.colorbar(sm, cax=cax)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label("distance / pooled two-copy chr1 median (dimensionless)", fontsize=7)
    fig.suptitle("P9016 chr1 distance maps | strict diagonal mask", fontsize=7, y=0.995)
    fig.text(0.5, 0.006, "Gray = missing common bin or diagonal; axes retain the full regular genomic span",
             ha="center", fontsize=7)
    paths = {}
    for extension in ("png", "pdf"):
        target = path_stem.with_suffix("." + extension)
        fig.savefig(target, dpi=300)
        paths[extension] = str(target)
    plt.close(fig)
    scale_info.update({"vmin": vmin, "vmax": vmax, "color_limits_clipping": "none"})
    return {"png": paths["png"], "pdf": paths["pdf"], "figsize_inches": [6.0, 9.0],
            "dpi": 300, "vmin": vmin, "vmax": vmax, "colormap": "coolwarm_r"}


def verify_terminal(selection020: dict[str, Any], termination020: dict[str, Any],
                    selection029: dict[str, Any], termination029: dict[str, Any],
                    endpoint_audits: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    if selection020.get("status") != "training_complete":
        raise RuntimeError("020 selection is not training_complete")
    if selection020["selection"]["selected_id"] != "random_joint":
        raise RuntimeError("020 selected id is not random_joint")
    attempts020 = selection020.get("attempts", [])
    if len(attempts020) != 6 or any(item.get("status") != "not_converged" for item in attempts020):
        raise RuntimeError("020 does not have six recorded not_converged stages")
    if termination020.get("job", {}).get("exit_code") != 0:
        raise RuntimeError("020 terminal audit job exit is not zero")
    if termination020.get("summary", {}).get("attempt_count") != 6:
        raise RuntimeError("020 terminal audit attempt count mismatch")
    if termination020.get("summary", {}).get("successful_scipy_attempts") != 0:
        raise RuntimeError("020 unexpectedly reports successful SciPy attempts")
    variant029 = next((item for item in selection029.get("variants", []) if item.get("model_id") == "C0"), None)
    if variant029 is None or variant029.get("selected_bundle_id") != "bundle2":
        raise RuntimeError("029 C0 selection is not bundle2")
    c0_candidates = {item["bundle_id"]: item for item in variant029.get("candidates", [])}
    if set(c0_candidates) != {"bundle1", "bundle2", "bundle3"}:
        raise RuntimeError("029 C0 selection does not preserve three starts")
    if any(not item.get("selection_eligible") or item.get("status") != "budget_not_converged"
           for item in c0_candidates.values()):
        raise RuntimeError("029 C0 selection has an ineligible or nonterminal start")
    if selection029.get("status") != "complete" or termination029.get("status") != "complete":
        raise RuntimeError("029 selection/termination is incomplete")
    if termination029.get("active_jobs") != [] or termination029.get("terminal_job_count") != 15:
        raise RuntimeError("029 does not report 15 terminal jobs and no active jobs")
    jobs = {item["job_id"]: item for item in termination029.get("jobs", [])}
    c0_terminal = {}
    for bundle in ("bundle1", "bundle2", "bundle3"):
        job_id = f"C0-{bundle}"
        item = jobs.get(job_id)
        if item is None or not item.get("fit_called") or not item.get("selection_eligible"):
            raise RuntimeError(f"missing 029 terminal audit for {job_id}")
        if item.get("status") != "budget_not_converged":
            raise RuntimeError(f"unexpected 029 terminal status for {job_id}")
        c0_terminal[bundle] = item
    return {
        "schema_version": "p9016-report-terminal-verification-v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "020": {
            "selection_status": selection020.get("status"),
            "selected_id": selection020["selection"]["selected_id"],
            "selected_coordinate_sha256_stored": selection020["selection"]["selected_coordinate"]["sha256"],
            "attempt_count": len(attempts020),
            "all_attempts_not_converged": True,
            "termination_audit_exit_code": termination020["job"]["exit_code"],
            "termination_summary": termination020["summary"],
            "initialization_source": "test_res/014-20260912_153000-s0-genome-wide-fixed/coords/random.3dg",
            "multiresolution": "5Mb->2Mb->1Mb",
        },
        "029_C0": {
            "selection_status": selection029.get("status"),
            "selected_bundle_id": variant029["selected_bundle_id"],
            "criterion": variant029["criterion"],
            "count_nll_by_bundle": {bundle: c0_candidates[bundle]["count_nll_normalized"]
                                    for bundle in sorted(c0_candidates)},
            "terminal_by_bundle": c0_terminal,
            "initialization_source": "test_res/025-20260913_135100-random-native-fullgrid-preflight/bundles/{bundle}/x0_normalized.npz",
            "direct_1mb_only": True,
            "maxiter_accepted": 480,
            "all_15_jobs_terminal": True,
        },
        "endpoint_audits": endpoint_audits,
        "selection_used_for_primary": {
            "020": "stored random_joint selection; no recomputation",
            "029_C0": "stored within-C0 bundle2 selection; no reference access",
        },
        "reference_read_order": "all candidate hashes and endpoint audits completed before reference text was opened",
        "phase_or_reference_used_for_selection": False,
        "fit_started_by_report": False,
        "report_exit_code": 0,
        "input_paths": {key: str(value) for key, value in paths.items()},
    }


def main() -> int:
    config = read_json(CONFIG_PATH)
    if config.get("status") != "frozen_before_computation":
        raise RuntimeError("config status is not frozen_before_computation")
    chroms, lengths = load_grid()
    source_paths = {
        "020_selection": ROOT / "test_res/020-20260913_071841-v1-p9016-joint/selection.json",
        "020_termination_audit": ROOT / "test_res/020-20260913_071841-v1-p9016-joint/termination_audit.json",
        "020_config": ROOT / "test_res/020-20260913_071841-v1-p9016-joint/config.json",
        "029_selection": ROOT / "test_res/029-20260913_161713-post020-allele-ablation-real/selection.json",
        "029_termination_audit": ROOT / "test_res/029-20260913_161713-post020-allele-ablation-real/termination_audit.json",
        "029_config": ROOT / "test_res/029-20260913_161713-post020-allele-ablation-real/config.json",
        "029_x0_manifest": ROOT / "test_res/029-20260913_161713-post020-allele-ablation-real/x0_manifest.json",
        "029_C0_bundle2_job": ROOT / "test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C0-bundle2/job.json",
        "029_C0_bundle2_final": ROOT / "test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C0-bundle2/attempts/attempt-001/final.json",
        "protocol_post020": ROOT / "docs/POST020_ALLELE_ABLATION_PROTOCOL.md",
        "protocol_post020_json": ROOT / "docs/POST020_ALLELE_ABLATION_PROTOCOL.json",
        "plan_genome_split": ROOT / "docs/PLAN-genome-allele-split.md",
        "r2_implementation": ROOT / "pr/r2comparison.py",
        "compatible_021_script": ROOT / "test_res/021-20260913-softall-v1-reference-spearman/compare.py",
        "snpfree_pairs": ROOT / "inputs/P9016.snpfree.pairs.gz",
    }
    for key, path in source_paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing source path {key}: {path}")

    input_specs = config["inputs"]
    coordinate_paths = {
        "020_multires_selected": Path(input_specs["primary_candidates"]["020_multires_selected"]["path"]),
        "029_C0_bundle2": Path(input_specs["primary_candidates"]["029_C0_bundle2"]["path"]),
        "029_C0_bundle1": Path(input_specs["supplemental_c0_starts"]["C0_bundle1"]["path"]),
        "029_C0_bundle3": Path(input_specs["supplemental_c0_starts"]["C0_bundle3"]["path"]),
    }
    coordinate_expected = {
        "020_multires_selected": input_specs["primary_candidates"]["020_multires_selected"]["sha256_expected"],
        "029_C0_bundle2": input_specs["primary_candidates"]["029_C0_bundle2"]["sha256_expected"],
        "029_C0_bundle1": input_specs["supplemental_c0_starts"]["C0_bundle1"]["sha256_expected"],
        "029_C0_bundle3": input_specs["supplemental_c0_starts"]["C0_bundle3"]["sha256_expected"],
    }
    for key, path in coordinate_paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing coordinate endpoint {key}: {path}")

    # candidate hashes 和 endpoint parsing 在触碰 reference path 之前完成。
    coordinate_hashes = {}
    candidate_maps = {}
    endpoint_audits = {}
    for key in ("020_multires_selected", "029_C0_bundle2", "029_C0_bundle1", "029_C0_bundle3"):
        actual = sha256_file(coordinate_paths[key])
        coordinate_hashes[key] = {"path": str(coordinate_paths[key]), "expected": coordinate_expected[key], "actual": actual,
                                  "matches_expected": actual == coordinate_expected[key]}
        if actual != coordinate_expected[key]:
            raise RuntimeError(f"coordinate hash mismatch for {key}: {actual} != {coordinate_expected[key]}")
        candidate_maps[key], endpoint_audits[key] = parse_candidate(coordinate_paths[key], chroms, lengths)
        endpoint_audits[key]["stored_final_json_coordinate_sha256"] = None
    if any(audit["rows"] != 5290 or audit["n_tracks"] != 40 for audit in endpoint_audits.values()):
        raise RuntimeError("candidate endpoint count/track audit failed")

    selection020 = read_json(source_paths["020_selection"])
    termination020 = read_json(source_paths["020_termination_audit"])
    selection029 = read_json(source_paths["029_selection"])
    termination029 = read_json(source_paths["029_termination_audit"])
    final_json_by_bundle = {}
    for bundle in ("bundle1", "bundle2", "bundle3"):
        final_path = ROOT / f"test_res/029-20260913_161713-post020-allele-ablation-real/jobs/C0-{bundle}/attempts/attempt-001/final.json"
        final_json_by_bundle[bundle] = read_json(final_path)
        endpoint_audits[f"029_C0_bundle{bundle[-1]}"]["stored_final_json_coordinate_sha256"] = final_json_by_bundle[bundle]["final"]["coordinates_file"]["sha256"]
        if final_json_by_bundle[bundle]["final"]["coordinates_file"]["sha256"] != coordinate_hashes[f"029_C0_bundle{bundle[-1]}"]["actual"]:
            raise RuntimeError(f"final.json coordinate hash mismatch for C0-{bundle}")
        if final_json_by_bundle[bundle]["final"]["coordinates_file"]["n_tracks"] != 40 or final_json_by_bundle[bundle]["final"]["coordinates_file"]["n_beads"] != 5290:
            raise RuntimeError(f"final.json endpoint dimensions mismatch for C0-{bundle}")
    terminal_verification = verify_terminal(selection020, termination020, selection029, termination029,
                                            endpoint_audits, coordinate_paths)

    # 现在才 hash 并打开仅供评估的 reference。
    reference_path = Path(input_specs["reference"]["path"])
    reference_expected = input_specs["reference"]["sha256_expected"]
    reference_actual = sha256_file(reference_path)
    if reference_actual != reference_expected:
        raise RuntimeError(f"reference hash mismatch: {reference_actual} != {reference_expected}")
    reference, reference_audit = parse_reference(reference_path)
    terminal_verification["reference"] = reference_audit
    terminal_verification["reference_loaded_after_all_candidate_hashes"] = True

    all_hashes = {}
    for key, path in source_paths.items():
        all_hashes[key] = {"path": str(path), "sha256": sha256_file(path)}
    for key, record in coordinate_hashes.items():
        all_hashes[key] = {"path": record["path"], "sha256": record["actual"], "expected": record["expected"]}
    all_hashes["reference"] = {"path": str(reference_path), "sha256": reference_actual, "expected": reference_expected}
    all_hashes["report_config"] = {"path": str(CONFIG_PATH), "sha256": sha256_file(CONFIG_PATH)}
    all_hashes["report_script"] = {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())}
    input_hash_manifest = {
        "schema_version": "p9016-report-input-hash-manifest-v1",
        "hashes": all_hashes,
        "candidate_hashes_completed_before_reference": True,
        "reference_opened_after_candidate_hashes": True,
        "phase_payload_opened": False,
        "fit_started": False,
    }
    write_json(OUT / "input_hash_manifest.json", input_hash_manifest)

    primary_specs = [
        ("020 multires", "020_multires_selected"),
        ("029 C0 direct 1Mb", "029_C0_bundle2"),
    ]
    supplemental_specs = [
        ("029 C0 bundle1", "029_C0_bundle1"),
        ("029 C0 bundle2", "029_C0_bundle2"),
        ("029 C0 bundle3", "029_C0_bundle3"),
    ]
    primary_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    per_copy: list[dict[str, Any]] = []
    supplemental_rows: list[dict[str, Any]] = []
    mask_records: dict[str, Any] = {}
    chr1_cache: dict[str, Any] = {}

    for chrom in chroms:
        full = full_positions(lengths[chrom])
        ref_mat_key, ref_pat_key = f"{chrom}(mat)", f"{chrom}(pat)"
        if ref_mat_key not in reference or ref_pat_key not in reference:
            raise RuntimeError(f"reference missing {chrom} mat/pat")
        track_sets = [
            reference[ref_mat_key], reference[ref_pat_key],
            candidate_maps["020_multires_selected"][candidate_prefix(chrom) + "a"],
            candidate_maps["020_multires_selected"][candidate_prefix(chrom) + "b"],
            candidate_maps["029_C0_bundle2"][candidate_prefix(chrom) + "a"],
            candidate_maps["029_C0_bundle2"][candidate_prefix(chrom) + "b"],
        ]
        finite_by_track = [set(finite_bins(track, full)) for track in track_sets]
        common = sorted(set.intersection(*finite_by_track))
        if len(common) < 4:
            raise RuntimeError(f"{chrom}: fewer than four main common bins")
        n_pairs = len(common) * (len(common) - 1) // 2
        record = {
            "chromosome": chrom,
            "header_length_bp": lengths[chrom],
            "full_grid_start_bp": OFFSET,
            "full_grid_end_last_start_bp": int(full[-1]) if len(full) else None,
            "full_grid_bins": int(len(full)),
            "full_grid_non_diagonal_pairs": int(len(full) * (len(full) - 1) // 2),
            "main_common_bins": int(len(common)),
            "main_common_non_diagonal_pairs": int(n_pairs),
            "common_positions_bp": common,
            "track_finite_bins": {
                ref_mat_key: len(finite_by_track[0]), ref_pat_key: len(finite_by_track[1]),
                f"020:{candidate_prefix(chrom)}a": len(finite_by_track[2]),
                f"020:{candidate_prefix(chrom)}b": len(finite_by_track[3]),
                f"029_C0:{candidate_prefix(chrom)}a": len(finite_by_track[4]),
                f"029_C0:{candidate_prefix(chrom)}b": len(finite_by_track[5]),
            },
        }
        for method, key in primary_specs:
            prefix = candidate_prefix(chrom)
            row, matrices = score_method(method, prefix + "a", prefix + "b", candidate_maps[key],
                                         reference[ref_mat_key], reference[ref_pat_key], common,
                                         len(full), n_pairs)
            row["chromosome"] = chrom
            row["mask_status"] = "ok" if len(common) == len(full) else "partial_reference_coordinate_coverage"
            row["common_mask_definition"] = "reference mat/pat + 020 + 029 C0 bundle2 finite endpoints"
            primary_rows.append(row)
            all_rows.append(dict(row))
            per_copy.extend(per_copy_rows(row))
            if chrom == "chr1":
                chr1_cache[method] = {"row": row, "matrices": matrices}
        # 所有 supplemental C0 starts 都在完全相同的 primary mask 上测试。
        for method, key in supplemental_specs:
            prefix = candidate_prefix(chrom)
            row, matrices = score_method(method, prefix + "a", prefix + "b", candidate_maps[key],
                                         reference[ref_mat_key], reference[ref_pat_key], common,
                                         len(full), n_pairs, supplemental=True)
            row["chromosome"] = chrom
            row["mask_status"] = "ok" if len(common) == len(full) else "partial_reference_coordinate_coverage"
            row["common_mask_definition"] = "primary 020 + 029 C0 bundle2 mask; supplemental candidate audited on same mask"
            supplemental_rows.append(row)
            all_rows.append(dict(row))
            if chrom == "chr1":
                chr1_cache[method] = {"row": row, "matrices": matrices}
        record["supplemental_common_mask_audit"] = {
            row["method"]: {
                "common_bins_tested": row["common_bins"],
                "n_common_pairs": row["n_common_pairs"],
                "metric_status": row["metric_status"],
                "finite_candidate_endpoints_on_main_mask": row["metric_status"] == "ok",
            } for row in supplemental_rows if row["chromosome"] == chrom
        }
        mask_records[chrom] = record

    main_columns = [
        "chromosome", "method", "track_a", "track_b", "full_grid_bins", "common_bins", "n_bins", "n_pairs", "n_common_pairs",
        "mask_status", "common_mask_definition", "metric_status", "rho_a_mat", "rho_a_pat", "rho_b_mat", "rho_b_pat",
        "direct_original", "cross_original", "matched", "cross_matched", "contrast", "margin_mat", "margin_pat", "minmargin",
        "swap", "orientation", "matched_mat_copy", "matched_pat_copy", "supplemental",
    ]
    write_tsv(OUT / "per_chromosome.tsv", primary_rows, main_columns)
    write_tsv(OUT / "per_chromosome_all_conditions.tsv", all_rows, main_columns)
    copy_columns = ["chromosome", "method", "copy", "candidate_track", "reference_copy", "spearman",
                    "spearman_other_reference", "n_bins", "n_pairs", "swap", "orientation"]
    write_tsv(OUT / "per_copy.tsv", per_copy, copy_columns)
    write_tsv(OUT / "sensitivity_c0_starts.tsv", supplemental_rows, main_columns)
    mask_tsv_rows = []
    for chrom in chroms:
        record = mask_records[chrom]
        prefix = candidate_prefix(chrom)
        condition_tracks = [
            ("reference_mat", reference[f"{chrom}(mat)"], None),
            ("reference_pat", reference[f"{chrom}(pat)"], None),
            ("020_multires", candidate_maps["020_multires_selected"][prefix + "a"],
             candidate_maps["020_multires_selected"][prefix + "b"]),
            ("029_C0_bundle2", candidate_maps["029_C0_bundle2"][prefix + "a"],
             candidate_maps["029_C0_bundle2"][prefix + "b"]),
            ("029_C0_bundle1", candidate_maps["029_C0_bundle1"][prefix + "a"],
             candidate_maps["029_C0_bundle1"][prefix + "b"]),
            ("029_C0_bundle3", candidate_maps["029_C0_bundle3"][prefix + "a"],
             candidate_maps["029_C0_bundle3"][prefix + "b"]),
        ]
        for condition, track_a, track_b in condition_tracks:
            finite_a = len(finite_bins(track_a, full_positions(lengths[chrom])))
            finite_b = finite_a if track_b is None else len(finite_bins(track_b, full_positions(lengths[chrom])))
            mask_tsv_rows.append({"chromosome": chrom, "condition": condition,
                                  "full_grid_bins": record["full_grid_bins"],
                                  "finite_bins_copy_a_or_track": finite_a,
                                  "finite_bins_copy_b": finite_b,
                                  "main_common_bins": record["main_common_bins"],
                                  "main_common_non_diagonal_pairs": record["main_common_non_diagonal_pairs"],
                                  "diagonal_excluded": True})
    write_tsv(OUT / "masks/mask_denominators.tsv", mask_tsv_rows,
              ["chromosome", "condition", "full_grid_bins", "finite_bins_copy_a_or_track", "finite_bins_copy_b",
               "main_common_bins", "main_common_non_diagonal_pairs", "diagonal_excluded"])
    write_json(OUT / "masks/common_positions.json", {
        "schema_version": "p9016-report-common-mask-v1",
        "grid": {"bin_size_bp": BIN, "offset_bp": OFFSET, "position_rule": ">=3Mb and <header length", "interpolation": False},
        "main_mask_conditions": ["reference mat", "reference pat", "020 multires c01a/c01b", "029 C0 bundle2 c01a/c01b"],
        "chromosomes": mask_records,
        "all_20_chromosomes_retained": len(mask_records) == 20,
    })

    # 所有 primary chromosome effects 使用固定的共享 bootstrap matrix。
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(chroms), size=(BOOTSTRAP_DRAWS, len(chroms)), dtype=np.int64)
    np.save(OUT / "bootstrap_index_matrix_seed9301.npy", indices)
    index_sha = hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()
    if not np.array_equal(np.load(OUT / "bootstrap_index_matrix_seed9301.npy", allow_pickle=False), indices):
        raise RuntimeError("bootstrap index matrix roundtrip failed")
    primary_lookup = {(row["chromosome"], row["method"]): row for row in primary_rows}
    matched_delta = np.asarray([primary_lookup[(chrom, "029 C0 direct 1Mb")]["matched"] -
                               primary_lookup[(chrom, "020 multires")]["matched"] for chrom in chroms], dtype=float)
    contrast_delta = np.asarray([primary_lookup[(chrom, "029 C0 direct 1Mb")]["contrast"] -
                                primary_lookup[(chrom, "020 multires")]["contrast"] for chrom in chroms], dtype=float)
    paired = {
        "matched_delta_029_minus_020": {
            **bootstrap(matched_delta, indices),
            "wins": int((matched_delta > TIE_TOL).sum()),
            "ties": int((np.abs(matched_delta) <= TIE_TOL).sum()),
            "losses": int((matched_delta < -TIE_TOL).sum()),
            "per_chromosome": {chrom: float(value) for chrom, value in zip(chroms, matched_delta)},
        },
        "contrast_delta_029_minus_020": {
            **bootstrap(contrast_delta, indices),
            "wins": int((contrast_delta > TIE_TOL).sum()),
            "ties": int((np.abs(contrast_delta) <= TIE_TOL).sum()),
            "losses": int((contrast_delta < -TIE_TOL).sum()),
            "per_chromosome": {chrom: float(value) for chrom, value in zip(chroms, contrast_delta)},
        },
        "direction": "029 minus 020",
        "bootstrap_index_matrix_sha256": index_sha,
        "bootstrap_index_matrix_path": str(OUT / "bootstrap_index_matrix_seed9301.npy"),
    }
    write_json(OUT / "bootstrap_summary.json", paired)

    condition_summary = {}
    for method, _key in primary_specs:
        rows = [primary_lookup[(chrom, method)] for chrom in chroms]
        matched_values = np.asarray([row["matched"] for row in rows], dtype=float)
        contrast_values = np.asarray([row["contrast"] for row in rows], dtype=float)
        condition_summary[method] = {
            "n_chromosomes": len(rows),
            "matched_mean": float(matched_values.mean()),
            "matched_median": float(np.median(matched_values)),
            "contrast_mean": float(contrast_values.mean()),
            "contrast_median": float(np.median(contrast_values)),
            "swap_counts": {
                "direct": int(sum(row["orientation"] == "direct" for row in rows)),
                "swapped": int(sum(row["orientation"] == "swapped" for row in rows)),
                "tie_average": int(sum(row["orientation"] == "tie_average" for row in rows)),
            },
            "n_finite_matched": int(np.isfinite(matched_values).sum()),
            "n_finite_contrast": int(np.isfinite(contrast_values).sum()),
        }
    sensitivity_summary = {}
    for method, key in supplemental_specs:
        rows = [row for row in supplemental_rows if row["method"] == method]
        values = np.asarray([row["matched"] for row in rows], dtype=float)
        contrasts = np.asarray([row["contrast"] for row in rows], dtype=float)
        sensitivity_summary[method] = {
            "n_chromosomes": len(rows),
            "matched_mean": float(values.mean()), "matched_median": float(np.median(values)),
            "contrast_mean": float(contrasts.mean()), "contrast_median": float(np.median(contrasts)),
            "all_on_primary_mask": all(row["metric_status"] == "ok" for row in rows),
        }

    # orientation 确定后保存 chr1 matrices。reference 和每个 method 的两个 copy 各有一个 pooled scalar；不独立缩放任何 copy。
    chr1_full = full_positions(lengths["chr1"])
    chr1_common = mask_records["chr1"]["common_positions_bp"]
    chr1_index = {int(position): index for index, position in enumerate(chr1_full)}
    chr1_matrix_items = []
    method_scale_info: dict[str, Any] = {}
    def add_matrix(method_label: str, display_label: str, track: dict[int, np.ndarray], method_key: str) -> None:
        raw_matrix = distance_matrix(track, chr1_common)
        method_scale_info.setdefault(method_key, {"raw_offdiag_values": []})["raw_offdiag_values"].append(strict_upper(raw_matrix))
        chr1_matrix_items.append({"method": method_key, "display_label": display_label, "raw_matrix": raw_matrix})
    add_matrix("Reference", "Reference mat", reference["chr1(mat)"], "Reference")
    add_matrix("Reference", "Reference pat", reference["chr1(pat)"], "Reference")
    for method_label, cache_key in (("020 multires", "020 multires"), ("029 C0 direct 1Mb", "029 C0 direct 1Mb")):
        row = chr1_cache[method_label]["row"]
        candidate = candidate_maps["020_multires_selected" if method_label == "020 multires" else "029_C0_bundle2"]
        prefix = "c01"
        if row["swap"] is True:
            mat_copy, pat_copy = "b", "a"
        else:
            mat_copy, pat_copy = "a", "b"
        add_matrix(method_label, f"{method_label}: c01{mat_copy} -> mat", candidate[f"{prefix}{mat_copy}"], method_label)
        add_matrix(method_label, f"{method_label}: c01{pat_copy} -> pat", candidate[f"{prefix}{pat_copy}"], method_label)
    for method_key, item in method_scale_info.items():
        pooled = np.concatenate(item["raw_offdiag_values"])
        scale = float(np.median(pooled))
        if not finite_number(scale) or scale <= 0:
            raise RuntimeError(f"invalid chr1 pooled scale for {method_key}: {scale}")
        item["pooled_offdiag_median"] = scale
        item["n_copy_matrices"] = len(item["raw_offdiag_values"])
        item["n_pooled_offdiag_values"] = int(len(pooled))
        item.pop("raw_offdiag_values")
    full_matrices: list[dict[str, Any]] = []
    npz_payload: dict[str, Any] = {
        "full_positions_bp": chr1_full,
        "common_positions_bp": np.asarray(chr1_common, dtype=np.int64),
    }
    for index, item in enumerate(chr1_matrix_items):
        scale = method_scale_info[item["method"]]["pooled_offdiag_median"]
        normalized = item["raw_matrix"] / scale
        embedded_raw = np.full((len(chr1_full), len(chr1_full)), np.nan, dtype=float)
        embedded_norm = np.full((len(chr1_full), len(chr1_full)), np.nan, dtype=float)
        idx = np.asarray([chr1_index[position] for position in chr1_common], dtype=int)
        embedded_raw[np.ix_(idx, idx)] = item["raw_matrix"]
        embedded_norm[np.ix_(idx, idx)] = normalized
        np.fill_diagonal(embedded_raw, np.nan); np.fill_diagonal(embedded_norm, np.nan)
        label_key = f"map_{index + 1}"
        npz_payload[label_key + "_raw"] = embedded_raw
        npz_payload[label_key + "_normalized"] = embedded_norm
        full_matrices.append({"method": item["method"], "display_label": item["display_label"],
                              "raw": embedded_raw, "normalized": embedded_norm})
    np.savez_compressed(OUT / "matrices/chr1_distance_matrices.npz", **npz_payload)
    map_metadata = {
        "schema_version": "p9016-report-chr1-distance-matrices-v1",
        "full_grid_positions_bp": chr1_full.tolist(),
        "full_grid_positions_mb": (chr1_full / 1e6).tolist(),
        "common_positions_bp": chr1_common,
        "n_full_grid_bins": len(chr1_full),
        "n_common_bins": len(chr1_common),
        "n_full_grid_non_diagonal_pairs": int(len(chr1_full) * (len(chr1_full) - 1) // 2),
        "n_common_non_diagonal_pairs": int(len(chr1_common) * (len(chr1_common) - 1) // 2),
        "diagonal_masked": True,
        "missing_bins_embedded_as_nan": True,
        "method_scales": method_scale_info,
        "map_labels": [item["display_label"] for item in full_matrices],
        "npz_path": str(OUT / "matrices/chr1_distance_matrices.npz"),
    }
    write_json(OUT / "matrices/chr1_scaling_factors.json", map_metadata)

    plot_metadata = {
        "primary_correlation": plot_two_boxes(OUT / "plots/correlation_boxplot",
                                               [("020 multires", np.asarray([primary_lookup[(chrom, "020 multires")]["matched"] for chrom in chroms])),
                                                ("029 C0 direct 1Mb", np.asarray([primary_lookup[(chrom, "029 C0 direct 1Mb")]["matched"] for chrom in chroms]))],
                                               chroms, "Matched Spearman rho", "Matched correlation"),
        "matched_minus_swapped_contrast": plot_two_boxes(OUT / "plots/contrast_boxplot",
                                                          [("020 multires", np.asarray([primary_lookup[(chrom, "020 multires")]["contrast"] for chrom in chroms])),
                                                           ("029 C0 direct 1Mb", np.asarray([primary_lookup[(chrom, "029 C0 direct 1Mb")]["contrast"] for chrom in chroms]))],
                                                          chroms, "matched - other Spearman rho", "Copy contrast"),
        "per_reference_copy": plot_per_copy(OUT / "plots/per_reference_copy_boxplot", per_copy, chroms),
        "chr1_distance_maps": render_heatmaps(OUT / "plots/chr1_distance_heatmaps", full_matrices, chr1_full,
                                               map_metadata, ["Reference", "020 multires", "029 C0 direct 1Mb"],
                                               ["mat", "pat"]),
    }
    write_json(OUT / "matrices/chr1_scaling_factors.json", map_metadata)

    chr1_rows = {row["method"]: row for row in primary_rows if row["chromosome"] == "chr1"}
    summary = {
        "schema_version": "p9016-020-vs-029-c0-reference-summary-v1",
        "status": "evaluation_complete",
        "run_id": config["run_id"],
        "scientific_unit": config["scientific_unit"],
        "primary_conditions": [name for name, _ in primary_specs],
        "condition_summary": condition_summary,
        "paired_effects": paired,
        "chr1": chr1_rows,
        "supplemental_c0_starts": sensitivity_summary,
        "mask_summary": {
            "n_chromosomes_expected": 20,
            "n_chromosomes_retained": len(mask_records),
            "all_same_main_mask_definition": True,
            "full_grid_bins_by_chromosome": {chrom: mask_records[chrom]["full_grid_bins"] for chrom in chroms},
            "main_common_bins_by_chromosome": {chrom: mask_records[chrom]["main_common_bins"] for chrom in chroms},
            "main_common_pairs_by_chromosome": {chrom: mask_records[chrom]["main_common_non_diagonal_pairs"] for chrom in chroms},
        },
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
                      "index_matrix_path": str(OUT / "bootstrap_index_matrix_seed9301.npy"),
                      "index_matrix_sha256": index_sha,
                      "unit": "chromosome", "interpretation": "technical/structural within one cell"},
        "heatmap": map_metadata,
        "plots": plot_metadata,
        "input_hash_manifest": str(OUT / "input_hash_manifest.json"),
        "terminal_verification": str(OUT / "terminal_verification.json"),
        "selection_recomputed": False,
        "reference_used_for_selection": False,
        "fit_started_by_report": False,
        "phase_payload_opened": False,
        "limitations": [
            "020 is a 5Mb->2Mb->1Mb multiresolution endpoint initialized from 014 random.3dg; 029 C0 is a direct 1Mb endpoint initialized from 025 native full-grid bundle starts.",
            "The conditions also differ in objective/context and finite 480-iteration budget; this is not an isolated resolution ablation.",
            "All endpoints are budget-limited not_converged records; numerical completion is not convergence.",
            "20 chromosomes are linked measurements from one cell; bootstrap intervals are not biological replicate uncertainty.",
            "A reference-relative correlation and positive contrast do not establish whole-chromosome allele recovery or a new L1/L2/L3 claim.",
        ],
    }
    write_json(OUT / "summary.json", summary)

    # 评估后重新 hash 所有 candidate/reference/input bytes，并断言未发生变更。
    post_hash_checks = {}
    for key, path in coordinate_paths.items():
        post_hash_checks[key] = sha256_file(path) == coordinate_hashes[key]["actual"]
    post_hash_checks["reference"] = sha256_file(reference_path) == reference_actual
    post_hash_checks["snpfree_pairs"] = sha256_file(source_paths["snpfree_pairs"]) == input_specs["snpfree_pairs"]["sha256_expected"]
    if not all(post_hash_checks.values()):
        raise RuntimeError(f"input bytes changed during report: {post_hash_checks}")
    terminal_verification["post_run_hash_unchanged"] = post_hash_checks
    write_json(OUT / "terminal_verification.json", terminal_verification)
    # 同时将运行后验证记录到 summary。
    summary["post_run_hash_unchanged"] = post_hash_checks
    write_json(OUT / "summary.json", summary)
    print(json.dumps({
        "status": "evaluation_complete",
        "output_dir": str(OUT),
        "primary_matched_mean": condition_summary["020 multires"]["matched_mean"],
        "c0_matched_mean": condition_summary["029 C0 direct 1Mb"]["matched_mean"],
        "paired_matched_delta": paired["matched_delta_029_minus_020"],
        "chr1": {key: {field: value for field, value in row.items()
                        if field in ("matched", "contrast", "orientation", "rho_a_mat", "rho_a_pat", "rho_b_mat", "rho_b_pat")}
                 for key, row in chr1_rows.items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise
