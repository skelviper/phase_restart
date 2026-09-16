#!/usr/bin/env python3
"""生成冻结的 P9016 seed124101 softall chr1 distance-map 补充结果。"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONFIG_DEFAULT = HERE / "config.json"
CHROMOSOMES = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
BIN = 1_000_000
OFFSET = 3_000_000
CHROMOSOME = "chr1"
CHROMOSOME_LENGTH = 195_471_971
POSITIONS_BP = np.arange(OFFSET, CHROMOSOME_LENGTH, BIN, dtype=np.int64)
EXPECTED_COLUMNS = ["chr", "start", "end", "bid", "copy", "diploid_bid", "x", "y", "z"]
EXPECTED_TRAIN_COLUMNS = "readID chr1 pos1 chr2 pos2 strand1 strand2"
COLORMAP = "coolwarm_r"
MISSING_COLOR = "#bdbdbd"
FIGURE_SIZE = (6.0, 6.0)
DPI = 300
FONT_SIZE = 7
V1_COLOR_VMAX_TEXT = "2.4009829719481615"


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def require_file(record: dict[str, Any], label: str) -> tuple[Path, str]:
    path = Path(record["path"]).resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} is unavailable: {path}")
    expected = str(record["sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError(f"{label} has an invalid frozen SHA256")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: {path}: {actual} != {expected}")
    return path, actual


def read_key_value_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != ["key", "value"]:
            raise RuntimeError(f"manifest is not a key/value TSV: {path}")
        for line_no, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise RuntimeError(f"manifest line {line_no} has {len(row)} columns: {path}")
            if row[0] in values:
                raise RuntimeError(f"manifest has duplicate key {row[0]}: {path}")
            values[row[0]] = row[1]
    return values


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"empty TSV header: {path}")
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({str(key): str(value) for key, value in row.items()})
    return rows


def validate_source(cfg: dict[str, Any]) -> dict[str, Any]:
    source = cfg["source"]
    coordinate_path, coordinate_sha = require_file(source["coordinate"], "softall coordinates")
    manifest_path, manifest_sha = require_file(source["manifest"], "softall manifest")
    train_log_path, train_log_sha = require_file(source["training_access_log"], "training access log")
    audit_log_path, audit_log_sha = require_file(source["audit_access_log"], "audit access log")
    eval_manifest_path, eval_manifest_sha = require_file(source["source_evaluation_manifest"], "source evaluation manifest")
    experiment_readme_path, experiment_readme_sha = require_file(source["source_experiment_readme"], "source experiment README")
    audit_readme_path, audit_readme_sha = require_file(source["source_audit_readme"], "source audit README")
    completion_path, completion_sha = require_file(source["source_completion"], "source completion")
    protocol_path, protocol_sha = require_file(source["source_protocol"], "source protocol")
    evaluator_path, evaluator_sha = require_file(source["source_evaluator"], "source evaluator")
    pairing_path, pairing_sha = require_file(
        {"path": cfg["pairing"]["source_table"], "sha256": cfg["pairing"]["source_table_sha256"]},
        "source geometry pairing table",
    )

    expected_coordinate = coordinate_path
    manifest = read_key_value_manifest(manifest_path)
    expected_manifest = {
        "sample": "P9016",
        "runner_family": "p9016_minimal",
        "config_name": "softall",
        "baseline": "softall",
        "mstep_graph_mode": "raw_expected_soft_all",
        "training_graph_mode": "raw_expected_soft_all",
        "train_pairs_columns": EXPECTED_TRAIN_COLUMNS,
        "train_pairs_column_count": "7",
        "n_raw": "1703888",
        "n_beads": "2633",
        "resolution": "1000000",
        "resolution_label": "1Mb",
        "n_iter": "30",
        "init_mode": "random_diploid",
        "init_seed": "124101",
        "force_mode": "physical",
        "uses_phase_labels": "0",
        "uses_charm_or_reference": "0",
        "uses_charm_for_training": "0",
        "reference_derived_positive_control": "0",
        "output_coords": str(expected_coordinate),
        "output_coords_sha256": source["coordinate"]["sha256"],
        "status": "OK",
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"manifest field {key}={manifest.get(key)!r}; expected {expected!r}")
    if any(token in manifest["train_pairs_columns"].lower() for token in ("phase", "snp", "reference")):
        raise RuntimeError("source training columns contain a forbidden evaluator field")

    train_log = read_json(train_log_path)
    audit_log = read_json(audit_log_path)
    if train_log.get("status") != "PASS" or train_log.get("returncode") != 0 or train_log.get("violations") != []:
        raise RuntimeError("training access log is not a clean PASS")
    raw_input = source["raw_training_input"]
    input_hashes = train_log.get("input_hashes", {})
    if input_hashes.get(raw_input["path"]) != raw_input["sha256"]:
        raise RuntimeError("training access log raw-input hash does not match frozen config")
    if train_log.get("observed_reads") != [raw_input["path"]]:
        raise RuntimeError("training access log observed reads are not raw-only")
    if train_log.get("allowed_reads") != [raw_input["path"]]:
        raise RuntimeError("training access log allowed reads are not raw-only")
    if audit_log.get("status") != "PASS" or audit_log.get("returncode") != 0 or audit_log.get("violations") != []:
        raise RuntimeError("source audit access log is not a clean PASS")

    evaluation_manifest = read_json(eval_manifest_path)
    summary = evaluation_manifest.get("summary", {})
    if not isinstance(summary, dict):
        raise RuntimeError("source evaluation manifest lacks summary")
    source_reference = str(summary.get("reference_3dg_path"))
    source_reference_sha = str(summary.get("reference_3dg_sha256"))
    if source_reference_sha != cfg["reference"]["sha256"]:
        raise RuntimeError("source evaluator reference SHA differs from current frozen reference")
    source_reference_path_matches_current = source_reference == cfg["reference"]["path"]
    if summary.get("reconstruction_path") != str(expected_coordinate):
        raise RuntimeError("source evaluator reconstruction path differs from selected coordinates")
    if summary.get("reconstruction_sha256") != source["coordinate"]["sha256"]:
        raise RuntimeError("source evaluator reconstruction SHA differs from selected coordinates")
    if summary.get("train_manifest_path") != str(manifest_path):
        raise RuntimeError("source evaluator manifest path differs from selected manifest")
    if summary.get("train_manifest_sha256") != source["manifest"]["sha256"]:
        raise RuntimeError("source evaluator manifest SHA differs from selected manifest")
    source_swaps = json.loads(str(summary["distance_per_chrom_copy_swaps_json"]))
    if int(source_swaps[CHROMOSOME]) != int(cfg["pairing"]["source_chr1_copy_swap"]):
        raise RuntimeError("source evaluation manifest chr1 geometry swap differs from frozen config")

    completion = read_json(completion_path)
    if completion.get("status") != "COMPLETE" or completion.get("mode") != "SEALED":
        raise RuntimeError("source completion is not sealed COMPLETE")
    if completion.get("scientific_success_claim") is not False:
        raise RuntimeError("source completion scientific-success flag changed")
    protocol = read_json(protocol_path)
    if protocol.get("selection") != "none" or protocol.get("seed_selection") != "none; retain and evaluate all six registered endpoints":
        raise RuntimeError("source protocol selection metadata changed")

    evaluator_text = evaluator_path.read_text(encoding="utf-8")
    if 'if value in {"mat", "copy0", "0"}' not in evaluator_text or 'if value in {"pat", "copy1", "1"}' not in evaluator_text:
        raise RuntimeError("source evaluator copy_to_int semantics are not the frozen mat/pat definition")
    if "def read_reference_3dg" not in evaluator_text or "def split_chrom_copy" not in evaluator_text:
        raise RuntimeError("source evaluator reference parser definition is unavailable")

    pairing_rows = read_tsv_rows(pairing_path)
    selected_rows = [row for row in pairing_rows if row.get("chrom") == CHROMOSOME and row.get("selected_for_eval") == "1"]
    if len(selected_rows) != 1:
        raise RuntimeError(f"expected one selected source geometry row for {CHROMOSOME}, got {len(selected_rows)}")
    selected_row = selected_rows[0]
    if int(selected_row["copy_swap"]) != int(cfg["pairing"]["source_chr1_copy_swap"]):
        raise RuntimeError("source geometry table swap differs from frozen config")

    record_hashes = {
        "coordinate": {"path": str(coordinate_path), "expected_sha256": source["coordinate"]["sha256"], "actual_sha256": coordinate_sha},
        "manifest": {"path": str(manifest_path), "expected_sha256": source["manifest"]["sha256"], "actual_sha256": manifest_sha},
        "training_access_log": {"path": str(train_log_path), "expected_sha256": source["training_access_log"]["sha256"], "actual_sha256": train_log_sha},
        "audit_access_log": {"path": str(audit_log_path), "expected_sha256": source["audit_access_log"]["sha256"], "actual_sha256": audit_log_sha},
        "source_evaluation_manifest": {"path": str(eval_manifest_path), "expected_sha256": source["source_evaluation_manifest"]["sha256"], "actual_sha256": eval_manifest_sha},
        "source_experiment_readme": {"path": str(experiment_readme_path), "expected_sha256": source["source_experiment_readme"]["sha256"], "actual_sha256": experiment_readme_sha},
        "source_audit_readme": {"path": str(audit_readme_path), "expected_sha256": source["source_audit_readme"]["sha256"], "actual_sha256": audit_readme_sha},
        "source_completion": {"path": str(completion_path), "expected_sha256": source["source_completion"]["sha256"], "actual_sha256": completion_sha},
        "source_protocol": {"path": str(protocol_path), "expected_sha256": source["source_protocol"]["sha256"], "actual_sha256": protocol_sha},
        "source_evaluator": {"path": str(evaluator_path), "expected_sha256": source["source_evaluator"]["sha256"], "actual_sha256": evaluator_sha},
        "source_geometry_pairing": {"path": str(pairing_path), "expected_sha256": cfg["pairing"]["source_table_sha256"], "actual_sha256": pairing_sha},
    }
    return {
        "record_hashes": record_hashes,
        "manifest_fields": {key: manifest[key] for key in ("sample", "runner_family", "config_name", "train_pairs_columns", "n_raw", "n_beads", "resolution", "n_iter", "init_seed", "status", "uses_phase_labels", "uses_charm_or_reference")},
        "training_access": {"status": train_log["status"], "returncode": train_log["returncode"], "observed_reads": train_log["observed_reads"], "violations": train_log["violations"]},
        "audit_access": {"status": audit_log["status"], "returncode": audit_log["returncode"], "violations": audit_log["violations"]},
        "source_reference": {"path": source_reference, "sha256": source_reference_sha, "sha256_matches_current_config": True, "path_matches_current_config": source_reference_path_matches_current},
        "source_geometry_row": selected_row,
        "source_evaluator_copy_semantics": "copy_to_int: mat/copy0/0 -> reference copy index 0; pat/copy1/1 -> reference copy index 1",
        "source_completion": {"status": completion["status"], "mode": completion["mode"], "scientific_success_claim": completion["scientific_success_claim"]},
    }


def load_softall_coordinates(path: Path, cfg: dict[str, Any]) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    coordinates: dict[tuple[str, int, int], np.ndarray] = {}
    locus_bids: dict[tuple[str, int], int] = {}
    bid_loci: dict[int, tuple[str, int]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise RuntimeError(f"unexpected softall coordinate header: {reader.fieldnames}")
        for line_no, row in enumerate(reader, start=2):
            try:
                chrom = str(row["chr"])
                start = int(row["start"])
                end = int(row["end"])
                bid = int(row["bid"])
                copy = int(row["copy"])
                diploid_bid = int(row["diploid_bid"])
                xyz = np.asarray([float(row[axis]) for axis in ("x", "y", "z")], dtype=float)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"non-numeric coordinate row {line_no}") from exc
            if chrom not in CHROMOSOMES or copy not in (0, 1):
                raise RuntimeError(f"invalid chromosome/copy at coordinate row {line_no}")
            if start < 0 or start % BIN != 0 or end <= start or bid < 0 or diploid_bid != 2 * bid + copy:
                raise RuntimeError(f"invalid interval/global-bin identity at coordinate row {line_no}")
            locus = (chrom, start)
            if locus in locus_bids and locus_bids[locus] != bid:
                raise RuntimeError(f"locus has inconsistent global bid at coordinate row {line_no}")
            if bid in bid_loci and bid_loci[bid] != locus:
                raise RuntimeError(f"global bid is assigned to multiple loci at coordinate row {line_no}")
            locus_bids[locus] = bid
            bid_loci[bid] = locus
            if not np.isfinite(xyz).all():
                raise RuntimeError(f"non-finite coordinate row {line_no}")
            key = (chrom, start, copy)
            if key in coordinates:
                raise RuntimeError(f"duplicate coordinate row for {key}")
            coordinates[key] = xyz

    expected_loci = int(cfg["source"]["coordinate"]["native_n_beads"])
    expected_particles = int(cfg["source"]["coordinate"]["expected_physical_particles"])
    expected_tracks = int(cfg["source"]["coordinate"]["expected_tracks"])
    loci = {(chrom, start) for chrom, start, _copy in coordinates}
    if len(loci) != expected_loci:
        raise RuntimeError(f"source coordinate genomic-locus count {len(loci)} != {expected_loci}")
    if set(bid_loci) != set(range(expected_loci)):
        raise RuntimeError("source global genomic-locus bids are not the contiguous native range")
    if len(coordinates) != expected_particles:
        raise RuntimeError(f"source coordinate physical-particle count {len(coordinates)} != {expected_particles}")
    if len(coordinates) != 2 * len(loci):
        raise RuntimeError("source coordinate rows are not exactly two copies per genomic locus")
    observed_tracks = {(chrom, copy) for chrom, _start, copy in coordinates}
    if len(observed_tracks) != expected_tracks or len(observed_tracks) != 2 * len(CHROMOSOMES):
        raise RuntimeError(f"source coordinate track count {len(observed_tracks)} != {expected_tracks}")
    for locus in loci:
        if {(locus[0], locus[1], copy) for copy in (0, 1)} - set(coordinates):
            raise RuntimeError(f"source locus does not have two copies: {locus}")

    by_track: dict[str, dict[int, np.ndarray]] = {}
    for chrom, start, copy in sorted(coordinates):
        chrom_index = CHROMOSOMES.index(chrom) + 1
        track_name = f"c{chrom_index:02d}{'a' if copy == 0 else 'b'}"
        by_track.setdefault(track_name, {})[start] = coordinates[(chrom, start, copy)]
    expected_track_names = {f"c{index:02d}{copy}" for index in range(1, len(CHROMOSOMES) + 1) for copy in ("a", "b")}
    if set(by_track) != expected_track_names:
        raise RuntimeError("source coordinate chromosome-copy track names are incomplete")
    chr1_counts = {track: len(by_track[track]) for track in ("c01a", "c01b")}
    summary = {
        "coordinate_rows": len(coordinates),
        "genomic_loci": len(loci),
        "physical_particles": len(coordinates),
        "chromosome_copy_tracks": len(by_track),
        "chromosomes": len(CHROMOSOMES),
        "native_n_beads_interpretation": "基因组 loci, not 物理粒子",
        "per_chr1_track_rows": chr1_counts,
        "per_track_rows": {track: len(by_track[track]) for track in sorted(by_track)},
    }
    return by_track, summary


def dense_track(by_track: dict[int, np.ndarray], positions: np.ndarray) -> np.ndarray:
    dense = np.full((len(positions), 3), np.nan, dtype=float)
    for index, position in enumerate(positions):
        value = by_track.get(int(position))
        if value is not None:
            dense[index] = value
    return dense


def dense_reference(reference: dict[str, dict[int, np.ndarray]], track_name: str, positions: np.ndarray) -> np.ndarray:
    if track_name not in reference:
        raise RuntimeError(f"reference track is unavailable: {track_name}")
    dense = np.full((len(positions), 3), np.nan, dtype=float)
    for index, position in enumerate(positions):
        value = reference[track_name].get(int(position))
        if value is not None:
            value = np.asarray(value, dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise RuntimeError(f"reference track has invalid coordinate: {track_name}:{position}")
            dense[index] = value
    return dense


def distance_matrix(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_bins = np.isfinite(points).all(axis=1)
    matrix = np.full((len(points), len(points)), np.nan, dtype=float)
    indices = np.flatnonzero(finite_bins)
    if len(indices):
        values = points[indices]
        delta = values[:, None, :] - values[None, :, :]
        matrix[np.ix_(indices, indices)] = np.sqrt(np.sum(delta * delta, axis=2))
    pair_mask = np.outer(finite_bins, finite_bins)
    np.fill_diagonal(pair_mask, False)
    return matrix, finite_bins, pair_mask


def common_pair_mask(bin_mask: np.ndarray) -> np.ndarray:
    pair_mask = np.outer(bin_mask, bin_mask)
    np.fill_diagonal(pair_mask, False)
    return pair_mask


def upper_values(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    upper = np.triu(mask, 1)
    return matrix[upper]


def rms_scale(matrices: np.ndarray, pair_mask: np.ndarray) -> float:
    values = np.concatenate([upper_values(matrix, pair_mask) for matrix in matrices])
    if len(values) == 0 or not np.isfinite(values).all():
        raise RuntimeError("RMS scale has no finite common non-diagonal values")
    scale = float(np.sqrt(np.mean(values * values)))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"invalid RMS scale: {scale}")
    return scale


def spearman_rho(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import rankdata

    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return float("nan")
    left_rank = rankdata(left[mask], method="average")
    right_rank = rankdata(right[mask], method="average")
    if np.ptp(left_rank) == 0 or np.ptp(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def fallback_pairing(raw_reference: np.ndarray, raw_copy_a: np.ndarray, raw_copy_b: np.ndarray, pair_mask: np.ndarray) -> dict[str, Any]:
    reference_mat, reference_pat = raw_reference
    candidates = {}
    for swap in (0, 1):
        candidate_mat = raw_copy_b if swap else raw_copy_a
        candidate_pat = raw_copy_a if swap else raw_copy_b
        rho_mat = spearman_rho(upper_values(reference_mat, pair_mask), upper_values(candidate_mat, pair_mask))
        rho_pat = spearman_rho(upper_values(reference_pat, pair_mask), upper_values(candidate_pat, pair_mask))
        candidates[str(swap)] = {"rho_mat": rho_mat, "rho_pat": rho_pat, "rho_sum": rho_mat + rho_pat}
    score0 = candidates["0"]["rho_sum"]
    score1 = candidates["1"]["rho_sum"]
    if math.isfinite(score1) and (not math.isfinite(score0) or score1 > score0 + 1e-12):
        selected = 1
    else:
        selected = 0
    return {"mode": "current_common_mask_rho_sum_fallback", "selected_swap": selected, "candidates": candidates, "tie_policy": "swap0: copy0/A -> maternal"}


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def pdf_page_size(path: Path) -> tuple[float, float]:
    data = path.read_bytes()
    matches = re.findall(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", data)
    if not matches:
        raise RuntimeError(f"could not parse PDF MediaBox: {path}")
    width, height = matches[0]
    return float(width), float(height)


def render_figure(normalized: np.ndarray, positions_bp: np.ndarray, vmax: float, labels: list[str], png_path: Path, pdf_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.linewidth": 0.6,
    })
    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad(MISSING_COLOR)
    if cmap.name != COLORMAP:
        raise RuntimeError(f"unexpected colormap: {cmap.name}")
    norm = Normalize(vmin=0.0, vmax=vmax, clip=False)
    positions_mb = positions_bp.astype(float) / 1_000_000.0
    half_bin_mb = BIN / 2_000_000.0
    extent = (
        float(positions_mb[0] - half_bin_mb),
        float(positions_mb[-1] + half_bin_mb),
        float(positions_mb[0] - half_bin_mb),
        float(positions_mb[-1] + half_bin_mb),
    )
    ticks = np.asarray([3, 50, 100, 150, 195], dtype=float)

    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    grid = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.0, 0.055],
        height_ratios=[1.0, 1.0],
        left=0.105,
        right=0.91,
        bottom=0.14,
        top=0.87,
        wspace=0.24,
        hspace=0.36,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    images = []
    for axis, matrix, title in zip(axes, normalized, labels):
        image = axis.imshow(matrix, origin="lower", interpolation="nearest", aspect="equal", extent=extent, cmap=cmap, norm=norm)
        images.append(image)
        axis.set_title(title, pad=4)
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.tick_params(width=0.6, length=2.5, pad=2)
        axis.set_xlabel("Genomic position (Mb)")
        axis.set_ylabel("Genomic position (Mb)")
    colorbar_axis = fig.add_subplot(grid[:, 2])
    colorbar = fig.colorbar(images[0], cax=colorbar_axis)
    colorbar.set_label("normalized 3D Euclidean distance", fontsize=FONT_SIZE, labelpad=4)
    colorbar.ax.tick_params(labelsize=FONT_SIZE, width=0.6, length=2.5)
    fig.suptitle("P9016 chr1 | softall seed124101", fontsize=FONT_SIZE, y=0.925)
    fig.text(0.105, 0.055, "Shared RMS scale by structure; red=near, blue=far; gray=missing", fontsize=FONT_SIZE, ha="left")
    fig.savefig(png_path, dpi=DPI)
    fig.savefig(pdf_path, dpi=DPI)
    plt.close(fig)


def symmetric_with_nan(matrix: np.ndarray) -> bool:
    if matrix.shape[0] != matrix.shape[1]:
        return False
    finite = np.isfinite(matrix) | np.isfinite(matrix.T)
    if not np.array_equal(np.isfinite(matrix), np.isfinite(matrix.T)):
        return False
    return bool(np.allclose(matrix[finite], matrix.T[finite], rtol=0, atol=1e-12))


def copy_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def validate_arrays(raw: np.ndarray, common_raw: np.ndarray, normalized: np.ndarray, bin_mask: np.ndarray, pair_mask: np.ndarray, positions: np.ndarray, scales: np.ndarray, vmax: float) -> dict[str, Any]:
    n_panels, n_bins, n_bins2 = raw.shape
    if (n_panels, n_bins, n_bins2) != (4, 193, 193):
        raise RuntimeError(f"unexpected raw matrix shape: {raw.shape}")
    if common_raw.shape != raw.shape or normalized.shape != raw.shape:
        raise RuntimeError("matrix shapes are inconsistent")
    if bin_mask.shape != (4, 193) or pair_mask.shape != (193, 193) or positions.shape != (193,):
        raise RuntimeError("mask or position shape is inconsistent")
    if not np.array_equal(pair_mask, np.outer(np.all(bin_mask, axis=0), np.all(bin_mask, axis=0)) & ~np.eye(193, dtype=bool)):
        raise RuntimeError("common pair mask is not derived from the common bin mask")
    if not all(symmetric_with_nan(matrix[panel]) for matrix in (raw, common_raw, normalized) for panel in range(4)):
        raise RuntimeError("distance matrices are not symmetric")
    if not np.array_equal(pair_mask, pair_mask.T) or np.any(np.diag(pair_mask)):
        raise RuntimeError("common pair mask is not symmetric non-diagonal")
    common_bins = np.all(bin_mask, axis=0)
    for panel in range(4):
        diagonal = np.diag(normalized[panel])
        if not np.allclose(diagonal[common_bins], 0.0, rtol=0, atol=1e-12):
            raise RuntimeError(f"normalized diagonal is not zero on common bins for panel {panel}")
        if np.any(np.isfinite(diagonal[~common_bins])):
            raise RuntimeError(f"normalized diagonal is finite outside common bins for panel {panel}")
        if np.any(np.isfinite(normalized[panel][~pair_mask & ~np.eye(193, dtype=bool)])):
            raise RuntimeError(f"normalized matrix has finite values outside common pairs for panel {panel}")
    finite_normalized = normalized[np.isfinite(normalized)]
    if len(finite_normalized) == 0 or np.min(finite_normalized) < -1e-12 or abs(float(np.max(finite_normalized)) - vmax) > 1e-12:
        raise RuntimeError("normalized values do not define the frozen common color range")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise RuntimeError("RMS scales are not finite and positive")
    if int(common_bins.sum()) <= 1 or int(pair_mask.sum() // 2) <= 0:
        raise RuntimeError("common mask has no usable non-diagonal pairs")
    return {
        "raw_matrices": list(raw.shape),
        "common_raw_matrices": list(common_raw.shape),
        "normalized_matrices": list(normalized.shape),
        "track_bin_finite_mask": list(bin_mask.shape),
        "common_bin_mask": list(common_bins.shape),
        "common_pair_mask": list(pair_mask.shape),
        "positions_bp": list(positions.shape),
        "raw_symmetric": True,
        "common_raw_symmetric": True,
        "normalized_symmetric": True,
        "common_pair_symmetric": True,
        "diagonal_zero_on_common_bins": True,
        "diagonal_excluded_from_pair_mask": True,
        "diagonal_excluded_from_rms": True,
        "n_common_bins": int(common_bins.sum()),
        "n_common_unordered_non_diagonal_pairs": int(pair_mask.sum() // 2),
        "n_all_unordered_non_diagonal_pairs": int(193 * 192 // 2),
        "normalized_min": float(np.min(finite_normalized)),
        "normalized_max": float(np.max(finite_normalized)),
        "color_norm_vmin": 0.0,
        "color_norm_vmax": float(vmax),
        "color_norm_clip": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    cfg = read_json(cfg_path)
    if cfg.get("status") != "config_frozen_before_compute":
        raise RuntimeError("config is not marked frozen before compute")
    if cfg.get("source", {}).get("selected_seed") != 124101 or cfg.get("source", {}).get("selected_endpoint") != "fixed_seed124101":
        raise RuntimeError("this script is frozen to source seed124101")
    if cfg.get("grid", {}).get("expected_n_bins") != 193 or cfg.get("grid", {}).get("bin_size_bp") != BIN or cfg.get("grid", {}).get("offset_bp") != OFFSET:
        raise RuntimeError("config grid differs from frozen chr1 grid")
    if list(POSITIONS_BP) != list(range(OFFSET, CHROMOSOME_LENGTH, BIN)):
        raise RuntimeError("internal chr1 positions differ from frozen grid")
    output_cfg = cfg["outputs"]
    npz_path = HERE / output_cfg["npz"]
    png_path = HERE / output_cfg["png"]
    pdf_path = HERE / output_cfg["pdf"]
    validation_path = HERE / output_cfg["validation"]
    provenance_path = HERE / output_cfg["provenance"]
    readme_path = HERE / output_cfg["readme"]
    gate_path = HERE / output_cfg["gate"]
    for path in (npz_path, png_path, pdf_path, validation_path, provenance_path, readme_path, gate_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite existing supplement artifact: {path}")

    source_evidence = validate_source(cfg)
    source_path = Path(cfg["source"]["coordinate"]["path"]).resolve()
    source_tracks, coordinate_summary = load_softall_coordinates(source_path, cfg)

    sys.path.insert(0, str(ROOT))
    from pr import ref3dg
    from pr.gate import EvalGate

    gate = EvalGate(str(gate_path))
    candidate_sha = gate.register("softall-chr1-distance-matrices", "candidate:softall-seed124101", str(source_path))
    if candidate_sha != cfg["source"]["coordinate"]["sha256"]:
        raise RuntimeError("candidate gate digest differs from frozen coordinate digest")
    gate.arm(ref3dg.STAGE)
    gate.require(ref3dg.STAGE)
    reference_path = Path(cfg["reference"]["path"]).resolve()
    reference_sha = sha256_file(reference_path)
    if reference_sha != cfg["reference"]["sha256"]:
        raise RuntimeError("current reference SHA differs from frozen config")
    reference_gate_sha = gate.register("softall-chr1-distance-matrices", "reference:P9016.1m.3dg.gz", str(reference_path))
    if reference_gate_sha != reference_sha:
        raise RuntimeError("reference gate digest differs from computed reference SHA")
    gate.require(ref3dg.STAGE)
    reference = ref3dg.load_reference(gate, path=str(reference_path))

    source_copy0 = dense_track(source_tracks["c01a"], POSITIONS_BP)
    source_copy1 = dense_track(source_tracks["c01b"], POSITIONS_BP)
    reference_mat = dense_reference(reference, "chr1(mat)", POSITIONS_BP)
    reference_pat = dense_reference(reference, "chr1(pat)", POSITIONS_BP)
    reference_matrices = np.stack([distance_matrix(reference_mat)[0], distance_matrix(reference_pat)[0]])
    source_copy_matrices = np.stack([distance_matrix(source_copy0)[0], distance_matrix(source_copy1)[0]])
    preliminary_matrices = np.concatenate([reference_matrices, source_copy_matrices])
    preliminary_bin_mask = np.stack([
        np.isfinite(reference_mat).all(axis=1),
        np.isfinite(reference_pat).all(axis=1),
        np.isfinite(source_copy0).all(axis=1),
        np.isfinite(source_copy1).all(axis=1),
    ])
    common_bins_preliminary = np.all(preliminary_bin_mask, axis=0)
    pair_mask = common_pair_mask(common_bins_preliminary)

    source_pairing = cfg["pairing"]
    source_swap = int(source_pairing["source_chr1_copy_swap"])
    if source_swap not in (0, 1):
        raise RuntimeError("invalid source chr1 geometry swap")
    if source_swap == 1:
        softall_mat, softall_pat = source_copy1, source_copy0
        softall_mat_matrix, softall_pat_matrix = source_copy_matrices[1], source_copy_matrices[0]
        softall_mat_name, softall_pat_name = "c01b", "c01a"
    else:
        softall_mat, softall_pat = source_copy0, source_copy1
        softall_mat_matrix, softall_pat_matrix = source_copy_matrices[0], source_copy_matrices[1]
        softall_mat_name, softall_pat_name = "c01a", "c01b"
    del softall_mat, softall_pat

    panels = np.stack([reference_matrices[0], reference_matrices[1], softall_mat_matrix, softall_pat_matrix])
    panel_bin_mask = np.stack([
        np.isfinite(reference_mat).all(axis=1),
        np.isfinite(reference_pat).all(axis=1),
        np.isfinite(source_copy1 if source_swap else source_copy0).all(axis=1),
        np.isfinite(source_copy0 if source_swap else source_copy1).all(axis=1),
    ])
    common_bins = np.all(panel_bin_mask, axis=0)
    common_pairs = common_pair_mask(common_bins)
    if not np.array_equal(common_pairs, pair_mask):
        raise RuntimeError("panel pairing changed the four-track common mask")

    common_raw = panels.copy()
    common_raw[:, ~common_pairs] = np.nan
    for panel_index in range(4):
        diagonal = np.diag_indices(len(POSITIONS_BP))
        common_raw[panel_index, diagonal[0], diagonal[1]] = np.where(common_bins, 0.0, np.nan)
    reference_scale = rms_scale(common_raw[:2], common_pairs)
    softall_scale = rms_scale(common_raw[2:], common_pairs)
    scales = np.asarray([reference_scale, softall_scale], dtype=float)
    panel_scales = np.asarray([reference_scale, reference_scale, softall_scale, softall_scale], dtype=float)
    normalized = panels / panel_scales[:, None, None]
    normalized[:, ~common_pairs] = np.nan
    diagonal = np.diag_indices(len(POSITIONS_BP))
    for panel_index in range(4):
        normalized[panel_index, diagonal[0], diagonal[1]] = np.where(common_bins, 0.0, np.nan)
    vmax = float(np.nanmax(normalized))
    if not math.isfinite(vmax) or vmax <= 0:
        raise RuntimeError(f"invalid common color maximum: {vmax}")

    labels = [
        "Reference + SNP | maternal",
        "Reference + SNP | paternal",
        "Softall | copy1 -> mat (geom)",
        "Softall | copy0 -> pat (geom)",
    ] if source_swap == 1 else [
        "Reference + SNP | maternal",
        "Reference + SNP | paternal",
        "Softall | copy0 -> mat (geom)",
        "Softall | copy1 -> pat (geom)",
    ]
    panel_track_names = ["chr1(mat)", "chr1(pat)", softall_mat_name, softall_pat_name]
    mask_counts = {
        "track_bin_finite": {label: int(mask.sum()) for label, mask in zip(labels, panel_bin_mask)},
        "track_bin_missing": {label: int((~mask).sum()) for label, mask in zip(labels, panel_bin_mask)},
        "common_bin_finite": int(common_bins.sum()),
        "common_bin_missing": int((~common_bins).sum()),
        "common_unordered_non_diagonal_pairs": int(common_pairs.sum() // 2),
        "all_unordered_non_diagonal_pairs": int(len(POSITIONS_BP) * (len(POSITIONS_BP) - 1) // 2),
        "track_individual_unordered_non_diagonal_pairs": {
            label: int(common_pair_mask(mask).sum() // 2) for label, mask in zip(labels, panel_bin_mask)
        },
    }

    render_figure(normalized, POSITIONS_BP, vmax, labels, png_path, pdf_path)
    np.savez_compressed(
        npz_path,
        raw_matrices=panels,
        common_raw_matrices=common_raw,
        normalized_matrices=normalized,
        positions_bp=POSITIONS_BP,
        track_bin_finite_mask=panel_bin_mask,
        track_pair_finite_mask=np.stack([common_pair_mask(mask) for mask in panel_bin_mask]),
        common_bin_mask=common_bins,
        common_pair_mask=common_pairs,
        scales=scales,
        panel_scales=panel_scales,
        color_norm=np.asarray([0.0, vmax], dtype=float),
        panel_labels=np.asarray(labels),
        panel_track_names=np.asarray(panel_track_names),
        pairing_json=np.asarray(json.dumps({"copy_swap": source_swap, "mapping": source_pairing["mapping"]}, sort_keys=True)),
        metadata_json=np.asarray(json.dumps({"schema": cfg["schema_version"], "sample": "P9016", "seed": 124101, "coordinate_units": "native uncalibrated", "colormap": COLORMAP}, sort_keys=True)),
    )

    # 只交付新命名的 softall 图；现有 V1 文件保持不变。
    delivery_dir = ROOT / "deliverables"
    delivery_dir.mkdir(parents=True, exist_ok=True)
    delivery_png = delivery_dir / "P9016-softall-seed124101-chr1-distance-matrices.png"
    delivery_pdf = delivery_dir / "P9016-softall-seed124101-chr1-distance-matrices.pdf"
    if delivery_png.exists() or delivery_pdf.exists():
        raise RuntimeError("refusing to overwrite existing softall delivery image")
    shutil.copy2(png_path, delivery_png)
    shutil.copy2(pdf_path, delivery_pdf)

    shape_validation = validate_arrays(panels, common_raw, normalized, panel_bin_mask, common_pairs, POSITIONS_BP, scales, vmax)
    gate_document = json.loads(gate_path.read_text(encoding="utf-8"))
    if not isinstance(gate_document, list):
        raise RuntimeError("fresh gate document must be a list of registration entries")
    png_size = png_dimensions(png_path)
    pdf_size = pdf_page_size(pdf_path)
    if png_size != (1800, 1800):
        raise RuntimeError(f"unexpected PNG dimensions: {png_size}")
    if not np.allclose(pdf_size, (432.0, 432.0), rtol=0, atol=1e-6):
        raise RuntimeError(f"unexpected PDF page size: {pdf_size}")

    config_sha = sha256_file(cfg_path)
    script_sha = sha256_file(Path(__file__).resolve())
    local_ref_loader = ROOT / "pr" / "ref3dg.py"
    local_ref_loader_sha = sha256_file(local_ref_loader)
    output_records = {
        "supplement_png": copy_record(png_path),
        "supplement_pdf": copy_record(pdf_path),
        "delivery_png": copy_record(delivery_png),
        "delivery_pdf": copy_record(delivery_pdf),
        "npz": copy_record(npz_path),
        "gate": copy_record(gate_path),
    }
    validation = {
        "schema": "supplemental_chr1_distance_matrices_softall.validation.v1",
        "status": "PASS",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "selected_seed": 124101,
            "coordinate": source_evidence["record_hashes"]["coordinate"],
            "genomic_loci": coordinate_summary["genomic_loci"],
            "physical_particles": coordinate_summary["physical_particles"],
            "chromosome_copy_tracks": coordinate_summary["chromosome_copy_tracks"],
            "native_n_beads_semantics": coordinate_summary["native_n_beads_interpretation"],
            "per_chr1_track_rows": coordinate_summary["per_chr1_track_rows"],
            "manifest_n_beads": int(cfg["source"]["coordinate"]["native_n_beads"]),
            "manifest_n_raw": int(cfg["source"]["source_training_summary"]["n_raw"]),
            "training_columns": cfg["source"]["coordinate"]["training_columns"],
            "uses_phase_labels": int(cfg["source"]["source_training_summary"]["uses_phase_labels"]),
            "uses_charm_or_reference": int(cfg["source"]["source_training_summary"]["uses_charm_or_reference"]),
        },
        "reference": {
            "path": str(reference_path),
            "expected_sha256": cfg["reference"]["sha256"],
            "actual_sha256": reference_sha,
            "matches_source_evaluation_manifest": source_evidence["source_reference"]["sha256_matches_current_config"],
            "loaded_via": "pr.ref3dg.load_reference(gate, path=...)",
        },
        "gate": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
            "document": gate_document,
            "source_registered_and_hashed_before_reference": True,
            "reference_loaded_after_eval_arm": True,
        },
        "grid": {
            "chromosome": CHROMOSOME,
            "bin_size_bp": BIN,
            "offset_bp": OFFSET,
            "chromosome_length_bp": CHROMOSOME_LENGTH,
            "n_bins": len(POSITIONS_BP),
            "first_position_bp": int(POSITIONS_BP[0]),
            "last_position_bp": int(POSITIONS_BP[-1]),
            "absolute_mb_axes": True,
        },
        "pairing": {
            "mode": "source_existing_geometry_match",
            "source_table": cfg["pairing"]["source_table"],
            "source_table_sha256": cfg["pairing"]["source_table_sha256"],
            "source_copy_swap": source_swap,
            "reference_copy0_is_maternal": True,
            "reference_copy1_is_paternal": True,
            "mapping": cfg["pairing"]["mapping"],
            "relation_to_source_evaluator": "same reference path/SHA and same copy_to_int semantics; source chr1 geometry row reused, not V1 pairing and not a new reference-based source selection",
            "fallback_rule_implemented_but_not_used": True,
        },
        "mask": mask_counts,
        "scales": {
            "reference_shared_rms": reference_scale,
            "softall_shared_rms": softall_scale,
            "panel_scales": panel_scales.tolist(),
            "definition": "one RMS per structure over both copies on the four-track common unordered non-diagonal pair set",
        },
        "color": {
            "colormap": COLORMAP,
            "missing_color": MISSING_COLOR,
            "vmin": 0.0,
            "vmax": vmax,
            "normalize_clip": False,
            "semantics": "red=near, blue=far",
        },
        "render": {
            "figure_size_inches": list(FIGURE_SIZE),
            "dpi": DPI,
            "font_size_pt": FONT_SIZE,
            "png_dimensions_px": list(png_size),
            "pdf_page_size_points": list(pdf_size),
            "panel_labels": labels,
            "panel_track_names": panel_track_names,
            "shared_colorbar": True,
        },
        "shape_validation": shape_validation,
        "coordinate_calibration": "none; native softall and reference units remain uncalibrated; no R=1 assumption",
        "artifacts": output_records,
        "code_and_config": {
            "config": {"path": str(cfg_path), "sha256": config_sha},
            "script": {"path": str(Path(__file__).resolve()), "sha256": script_sha},
            "local_guarded_reference_loader": {"path": str(local_ref_loader), "sha256": local_ref_loader_sha},
        },
        "scope_guard": {
            "no_training": True,
            "no_new_metrics_or_ci": True,
            "no_other_seed_rendered": True,
            "existing_v1_artifacts_modified": False,
        },
    }
    write_json(validation_path, validation)

    provenance = {
        "schema": "supplemental_chr1_distance_matrices_softall.provenance.v1",
        "status": "COMPLETE",
        "generated_utc": validation["generated_utc"],
        "supplement_root": str(HERE),
        "config": {"path": str(cfg_path), "sha256": config_sha},
        "source_identity": source_evidence,
        "selected_source": cfg["source"],
        "reference_identity": validation["reference"],
        "workflow_sequence": [
            "Validate frozen config and all source hashes without opening the reference payload.",
            "Hash and parse the selected seed124101 native softall coordinates; confirm 2633 基因组 loci, 5266 物理粒子, and 40 tracks.",
            "Register the selected coordinates in a fresh EvalGate and confirm the coordinate digest.",
            "Arm the eval stage; hash/register the current reference and confirm its SHA equals the source evaluator's frozen SHA.",
            "Load the reference only through pr.ref3dg.load_reference after the gate is armed.",
            "Use the source-version chr1 geometry swap=1 and construct the four-track common mask.",
            "Compute raw distances, shared per-structure RMS scales, common normalized color range, and the fixed 6x6 inch figure.",
        ],
        "reference_copy_semantics": {
            "source_evaluator": cfg["source"]["source_evaluator"],
            "definition": "source evaluator copy_to_int maps mat/copy0/0 to index 0 and pat/copy1/1 to index 1",
            "validated_against_source_code": True,
            "current_reference_sha_matches_source_eval_manifest": True,
        },
        "derived": {
            "coordinate_summary": coordinate_summary,
            "grid": validation["grid"],
            "pairing": validation["pairing"],
            "mask": mask_counts,
            "scales": validation["scales"],
            "color": validation["color"],
            "render": validation["render"],
            "coordinate_calibration": validation["coordinate_calibration"],
        },
        "validation": {"path": str(validation_path), "sha256": sha256_file(validation_path)},
        "artifacts": output_records,
        "scope_guard": validation["scope_guard"],
    }
    write_json(provenance_path, provenance)

    readme_lines = [
        "# P9016 chr1 softall 距离矩阵",
        "",
        "本目录是外部 phase3 正式 softall endpoint 的独立比较补充，不是 020 V1 训练候选，也未改动原 V1 图、ZIP 或配置。",
        "",
        "## 来源",
        f"- 样本/cohort：P9016，一个真实单细胞；variant `softall/raw_expected_soft_all`；seed `124101`；endpoint `fixed_seed124101`。",
        f"- 坐标：`{cfg['source']['coordinate']['path']}`。SHA256 `{cfg['source']['coordinate']['sha256']}`。",
        f"- native manifest：`{cfg['source']['manifest']['path']}`；`n_raw={cfg['source']['source_training_summary']['n_raw']}`，`n_beads={cfg['source']['coordinate']['native_n_beads']}`。这里 `n_beads` 是基因组 loci，不是物理粒子；坐标 TSV 实际解析为 `{coordinate_summary['genomic_loci']}` 个 loci、`{coordinate_summary['physical_particles']}` 个双 copy 物理粒子、`{coordinate_summary['chromosome_copy_tracks']}` 条 chromosome-copy 轨迹。",
        f"- 训练边界：7 列 `{EXPECTED_TRAIN_COLUMNS}`；`uses_phase_labels=0`，`uses_charm_or_reference=0`。",
        f"- 来源终态/审计：manifest `status=OK`；source completion `COMPLETE/SEALED`；训练与审计 access logs 均 `PASS` 且无 violations。",
        "",
        "## 参考结构与配对",
        f"- reference 仅用于评估：`{cfg['reference']['path']}`；SHA256 `{cfg['reference']['sha256']}`。来源评价器 `eval_manifest.json` 的历史挂载路径为 `{source_evidence['source_reference']['path']}`；路径不同但 SHA 相同，当前文件在坐标 gate arm 后通过 `pr.ref3dg.load_reference` 读取。",
        "- 源评价器定义 `copy0=mat`、`copy1=pat`；因此 source chr1 geometry `copy_swap=1` 映射为 reconstruction `copy1 -> maternal`、`copy0 -> paternal`。这不是 V1 的 `a=pat` 配对，也不是亲本身份判定。",
        f"- 复用源版本既有几何记录：`{cfg['pairing']['source_table']}`；selected chr1 `copy_swap=1`，source geometry Spearman `{cfg['pairing']['source_selected_geometry_spearman']}`。",
        "",
        "## 网格与计算",
        f"- chr1 1 Mb 绝对网格：`{OFFSET // BIN}..{int(POSITIONS_BP[-1] // BIN)} Mb`，共 `{len(POSITIONS_BP)}` bins；不压缩缺失轴。",
        f"- 观察单位：同一细胞的无序非对角基因组 bin 距离对；四轨共同有限 mask 为 `{mask_counts['common_bin_finite']}` 个 bins / `{mask_counts['common_unordered_non_diagonal_pairs']}` 个距离对。各轨 chr1 有限 bins：" + ", ".join(f"{label}={count}" for label, count in mask_counts["track_bin_finite"].items()) + ".",
        f"- reference 两 copy 共用 RMS scale `{reference_scale:.12g}`；softall 两 copy 共用 RMS scale `{softall_scale:.12g}`。坐标不做物理校准，不假定 `R=1`。",
        f"- 四个 panel 共用 Normalize `vmin=0, vmax={vmax:.12g}, clip=False`；colormap `{COLORMAP}`，红=近、蓝=远、缺失=`{MISSING_COLOR}`。本图 `vmax={vmax:.14f}`，而 V1 图为 `{V1_COLOR_VMAX_TEXT}`；跨图颜色浓淡不能直接解释为距离差。",
        "",
        "## 输出",
        f"- PNG：`{png_path}`（交付副本：`{delivery_png}`），尺寸 `{png_size[0]}x{png_size[1]} px`，300 dpi。",
        f"- PDF：`{pdf_path}`（交付副本：`{delivery_pdf}`），页面 `{pdf_size[0]:g}x{pdf_size[1]:g} pt`。",
        f"- NPZ：`{npz_path}`，含 raw/common-raw/normalized matrices、track/common masks、absolute positions、scales、panel labels 与 pairing metadata。",
        f"- 机器可读证据：`{validation_path}`、`{provenance_path}`、`{gate_path}`；冻结配置：`{cfg_path}`。",
        "",
        "图面板：上排 `Reference + SNP | maternal/paternal`；下排 `Softall | copy1 -> mat (geom)` 与 `copy0 -> pat (geom)`。本补充未训练、未新增 metrics/CI、未渲染另外两个 seed。",
    ]
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "PASS",
        "supplement": str(HERE),
        "source_coordinate_sha256": candidate_sha,
        "reference_sha256": reference_sha,
        "source_coordinate_rows": coordinate_summary["physical_particles"],
        "source_genomic_loci": coordinate_summary["genomic_loci"],
        "source_tracks": coordinate_summary["chromosome_copy_tracks"],
        "common_bins": mask_counts["common_bin_finite"],
        "common_pairs": mask_counts["common_unordered_non_diagonal_pairs"],
        "source_swap": source_swap,
        "reference_scale": reference_scale,
        "softall_scale": softall_scale,
        "color_vmax": vmax,
        "png": str(png_path),
        "pdf": str(pdf_path),
        "delivery_png": str(delivery_png),
        "delivery_pdf": str(delivery_pdf),
        "npz": str(npz_path),
        "validation": str(validation_path),
        "provenance": str(provenance_path),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
