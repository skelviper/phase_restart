#!/usr/bin/env python
"""审计此前完成的 native-FDG 调用和 full-grid linkage。

相对于实验本身，这是只读脚本：它解析已有 receipts、logs、bridge blobs、native outputs、坐标导出和 source snapshots。它绝不调用 native binary 或 optimizer。
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "test_res" / "030-20260914_c0-020-training-audit"
RUN025 = ROOT / "test_res" / "025-20260913_135100-random-native-fullgrid-preflight"
RUN014 = ROOT / "test_res" / "014-20260912_153000-s0-genome-wide-fixed"
BIN_SIZE = 1_000_000
BUNDLES = ("bundle1", "bundle2", "bundle3")
FORBIDDEN_COMPONENTS = {"evaluation-r2", "evaluation", "reference", "phase"}
FORBIDDEN_FILENAMES = {"P9016.1m.3dg.gz", "P9016.pairs.gz"}


def is_forbidden(path: Path) -> bool:
    return any(component in path.parts for component in FORBIDDEN_COMPONENTS) or path.name in FORBIDDEN_FILENAMES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def parse_3dg(path: Path, grid: dict[str, np.ndarray]) -> dict[str, Any]:
    lengths = np.asarray(grid["chromosome_lengths"], dtype=np.int64)
    n_bins = np.asarray(grid["n_bins"], dtype=np.int64)
    offsets = np.asarray(grid["offsets"], dtype=np.int64)
    track_names = [str(v) for v in grid["track_names"]]
    name_to_track = {name: index for index, name in enumerate(track_names)}
    coordinates = np.full((2, int(offsets[-1]), 3), np.nan, dtype=np.float64)
    seen = np.zeros((2, int(offsets[-1])), dtype=bool)
    order: list[str] = []
    rows = 0
    with path.open("rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_no}: expected five columns")
            name, position_text, *xyz_text = fields
            if name not in name_to_track:
                raise ValueError(f"{path}:{line_no}: unexpected track {name}")
            track = name_to_track[name]
            chromosome = track // 2
            copy = track % 2
            position = int(position_text)
            if position < 0 or position % BIN_SIZE != 0:
                raise ValueError(f"{path}:{line_no}: non-origin-0 position {position}")
            local = position // BIN_SIZE
            if local >= int(n_bins[chromosome]):
                raise ValueError(f"{path}:{line_no}: position outside track")
            index = int(offsets[chromosome] + local)
            if seen[copy, index]:
                raise ValueError(f"{path}:{line_no}: duplicate coordinate")
            xyz = np.asarray([float(value) for value in xyz_text], dtype=np.float64)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError(f"{path}:{line_no}: invalid coordinate")
            coordinates[copy, index] = xyz
            seen[copy, index] = True
            if not order or order[-1] != name:
                order.append(name)
            rows += 1
    expected_order = track_names
    missing_by_track: dict[str, list[int]] = {}
    for track, name in enumerate(track_names):
        chromosome = track // 2
        copy = track % 2
        slc = slice(int(offsets[chromosome]), int(offsets[chromosome + 1]))
        missing_local = np.flatnonzero(~seen[copy, slc])
        if len(missing_local):
            missing_by_track[name] = [int(value) * BIN_SIZE for value in missing_local]
    finite_coordinates = coordinates[np.isfinite(coordinates)].reshape(-1, 3) if np.isfinite(coordinates).any() else np.empty((0, 3), dtype=np.float64)
    return {
        "coordinates": coordinates,
        "rows": rows,
        "track_order": order,
        "expected_track_order": expected_order,
        "track_order_ok": order == expected_order,
        "complete": bool(seen.all()),
        "seen_count": int(seen.sum()),
        "expected_count": int(seen.size),
        "missing_by_track": missing_by_track,
        "file_sha256": sha256_file(path),
        "max_radius": float(np.linalg.norm(finite_coordinates, axis=1).max()) if len(finite_coordinates) else None,
    }


def bead_to_copyfirst(bead_coordinates: np.ndarray, grid: dict[str, np.ndarray]) -> np.ndarray:
    n_bins = np.asarray(grid["n_bins"], dtype=np.int64)
    offsets = np.asarray(grid["offsets"], dtype=np.int64)
    track_offsets = np.asarray(grid["track_offsets"], dtype=np.int64)
    result = np.empty((2, int(offsets[-1]), 3), dtype=np.float64)
    for chromosome in range(len(n_bins)):
        slc = slice(int(offsets[chromosome]), int(offsets[chromosome + 1]))
        for copy in (0, 1):
            track = 2 * chromosome + copy
            result[copy, slc] = bead_coordinates[int(track_offsets[track]):int(track_offsets[track + 1])]
    return result


def stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"count": int(len(values)), "finite": 0}
    return {
        "count": int(len(values)),
        "finite": int(len(finite)),
        "min": float(finite.min()),
        "p01": float(np.quantile(finite, 0.01)),
        "p10": float(np.quantile(finite, 0.10)),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p90": float(np.quantile(finite, 0.90)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(finite.max()),
        "frac_lt_0.005": float(np.mean(finite < 0.005)),
        "frac_lt_0.01": float(np.mean(finite < 0.01)),
        "frac_lt_0.02": float(np.mean(finite < 0.02)),
        "frac_lt_0.05": float(np.mean(finite < 0.05)),
        "frac_gt_0.2": float(np.mean(finite > 0.2)),
    }


def adjacent_summary(coordinates: np.ndarray, grid: dict[str, np.ndarray]) -> dict[str, Any]:
    n_bins = np.asarray(grid["n_bins"], dtype=np.int64)
    offsets = np.asarray(grid["offsets"], dtype=np.int64)
    by_chromosome: dict[str, Any] = {}
    pooled: list[np.ndarray] = []
    for chromosome, count in enumerate(n_bins):
        per_copy = []
        slc = slice(int(offsets[chromosome]), int(offsets[chromosome + 1]))
        for copy_name, copy in (("a", 0), ("b", 1)):
            distances = np.linalg.norm(np.diff(coordinates[copy, slc], axis=0), axis=1)
            per_copy.append({"copy": copy_name, **stats(distances)})
            pooled.append(distances)
        by_chromosome[f"chr{chromosome + 1 if chromosome < 19 else 'X'}"] = {
            "n_bins": int(count), "n_adjacent_edges_per_copy": int(max(int(count) - 1, 0)), "copies": per_copy,
            "pooled": stats(np.concatenate([np.asarray(item, dtype=np.float64) for item in [
                np.linalg.norm(np.diff(coordinates[0, slc], axis=0), axis=1),
                np.linalg.norm(np.diff(coordinates[1, slc], axis=0), axis=1),
            ]])) if int(count) > 1 else stats(np.asarray([], dtype=np.float64)),
        }
    pooled_values = np.concatenate(pooled) if pooled else np.asarray([], dtype=np.float64)
    return {"pooled": stats(pooled_values), "by_chromosome": by_chromosome}


def parse_native_log(path: Path) -> dict[str, Any]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mode_lines = [line for line in lines if "fdg_random_bridge: mode=" in line]
    max_nei_lines = [line for line in lines if "[M::hk_fdg]" in line and "max_nei:" in line]
    max_nei_match = re.search(r"max_nei:(-?\d+)", max_nei_lines[-1]) if max_nei_lines else None
    iter_pattern = re.compile(
        r"iter:(?P<iter>\d+) elapsed:(?P<elapsed>[0-9.]+)s rep_coef:(?P<rep>[0-9.]+) "
        r"RMS_force:(?P<rms>[0-9.]+) n_rep:(?P<nrep>[0-9.]+) "
        r"energy:(?P<ebb>[0-9.eE+-]+),(?P<econ>[0-9.eE+-]+),(?P<erep>[0-9.eE+-]+) "
        r"dist:(?P<dbb>[0-9.eE+-]+),(?P<dcon>[0-9.eE+-]+),(?P<drep>[0-9.eE+-]+)"
    )
    parsed = []
    for line in lines:
        match = iter_pattern.search(line)
        if match:
            row: dict[str, Any] = {"iteration": int(match.group("iter"))}
            for key in ("elapsed", "rep", "rms", "nrep", "ebb", "econ", "erep", "dbb", "dcon", "drep"):
                row[key] = float(match.group(key))
            parsed.append(row)
    return {
        "path": rel(path),
        "sha256": sha256_file(path),
        "line_count": len(lines),
        "mode_lines": mode_lines,
        "max_nei_lines": max_nei_lines,
        "max_nei": int(max_nei_match.group(1)) if max_nei_match else None,
        "iterations_logged": [int(row["iteration"]) for row in parsed],
        "first_iteration": parsed[0] if parsed else None,
        "last_iteration": parsed[-1] if parsed else None,
        "best_logged_iteration": int(min(parsed, key=lambda row: row["rms"])["iteration"]) if parsed else None,
        "best_logged_rms": float(min(row["rms"] for row in parsed)) if parsed else None,
        "last_iteration_is_1000": bool(parsed and parsed[-1]["iteration"] == 1000),
    }


def parse_hickit_log(path: Path) -> dict[str, Any]:
    parsed = parse_native_log(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    bead_lines = [line for line in lines if "hk_bmap_gen" in line]
    command = lines[0] if lines and lines[0].startswith("/") else None
    generated = re.search(r"generated (\d+) beads", "\n".join(bead_lines))
    merged = re.findall(r"(\d+) => (\d+)", "\n".join(bead_lines))
    return {
        "path": parsed["path"],
        "sha256": parsed["sha256"],
        "command": command,
        "max_nei": parsed["max_nei"],
        "max_nei_lines": parsed["max_nei_lines"],
        "generated_beads": int(generated.group(1)) if generated else None,
        "merge_steps": [[int(a), int(b)] for a, b in merged],
        "first_iteration": parsed["first_iteration"],
        "last_iteration": parsed["last_iteration"],
        "best_logged_iteration": parsed["best_logged_iteration"],
        "best_logged_rms": parsed["best_logged_rms"],
        "last_iteration_is_1000": parsed["last_iteration_is_1000"],
    }


def read_pair_header(path: Path) -> dict[str, Any]:
    headers: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            headers.append(line.rstrip("\n"))
    columns = next((line.split(":", 1)[1].strip() for line in headers if line.startswith("#columns:")), None)
    return {
        "path": rel(path),
        "sha256": sha256_file(path),
        "header_lines": headers,
        "columns": columns.split("\t") if columns else None,
        "has_n_nei_column": bool(columns and ("n_nei" in columns.split("\t") or "n_nei_corner" in columns.split("\t"))),
        "has_phase_columns": bool(columns and any("phase" in column for column in columns.split("\t"))),
    }

def parse_bridge(path: Path, grid: dict[str, np.ndarray], canonical: dict[str, np.ndarray]) -> dict[str, Any]:
    payload = path.read_bytes()
    header_struct = struct.Struct("<8s6I")
    if len(payload) < header_struct.size:
        raise ValueError(f"truncated bridge input: {path}")
    magic, version, bin_size, n_tracks, n_beads, n_raw, flags = header_struct.unpack_from(payload)
    offset = header_struct.size
    length_bytes = 4 * int(n_tracks)
    bead_bytes = 12 * int(n_beads)
    edge_bytes = 8 * int(n_raw)
    expected_size = header_struct.size + length_bytes + bead_bytes + edge_bytes
    if len(payload) != expected_size:
        raise ValueError(f"bridge byte size mismatch: {len(payload)} != {expected_size}")
    lengths = np.frombuffer(payload, dtype="<i4", count=int(n_tracks), offset=offset).copy()
    offset += length_bytes
    beads = np.frombuffer(payload, dtype="<i4", count=3 * int(n_beads), offset=offset).copy().reshape(int(n_beads), 3)
    offset += bead_bytes
    raw_edges = np.frombuffer(payload, dtype="<u4", count=2 * int(n_raw), offset=offset).copy().reshape(int(n_raw), 2)
    canonical_raw = np.asarray(canonical["raw_edges"], dtype="<u4")
    canonical_unique = np.asarray(canonical["unique_edges"], dtype="<u4")
    canonical_counts = np.asarray(canonical["unique_counts"], dtype="<i8")
    unique_edges, unique_counts = np.unique(raw_edges, axis=0, return_counts=True)
    track_lengths = np.asarray(grid["track_lengths"], dtype="<i4")
    grid_beads = np.asarray(grid["beads"], dtype="<i4")
    offsets = np.asarray(grid["track_offsets"], dtype=np.int64)
    track_sequence_ok = True
    expected_positions_ok = True
    for track, start in enumerate(offsets[:-1]):
        stop = int(offsets[track + 1])
        values = beads[int(start):stop]
        expected_track = np.full(len(values), track, dtype=np.int32)
        expected_start = np.arange(len(values), dtype=np.int64) * BIN_SIZE
        expected_end = np.minimum(expected_start + BIN_SIZE, int(track_lengths[track])).astype(np.int32)
        track_sequence_ok &= bool(np.array_equal(values[:, 0], expected_track))
        expected_positions_ok &= bool(np.array_equal(values[:, 1], expected_start.astype(np.int32)))
        expected_positions_ok &= bool(np.array_equal(values[:, 2], expected_end))
    return {
        "path": rel(path),
        "sha256": sha256_file(path),
        "byte_size": len(payload),
        "header": {
            "magic_ascii": magic.rstrip(b"\x00").decode("ascii"), "version": int(version),
            "bin_size_bp": int(bin_size), "n_tracks": int(n_tracks), "n_beads": int(n_beads),
            "n_raw_pairs": int(n_raw), "flags": int(flags),
        },
        "payload_checks": {
            "track_lengths_vs_grid_exact": bool(np.array_equal(lengths, track_lengths)),
            "beads_vs_grid_exact": bool(np.array_equal(beads, grid_beads)),
            "raw_edges_vs_canonical_exact": bool(np.array_equal(raw_edges, canonical_raw)),
            "unique_edges_vs_recomputed_exact": bool(np.array_equal(unique_edges, canonical_unique)),
            "unique_counts_vs_recomputed_exact": bool(np.array_equal(unique_counts.astype(np.int64), canonical_counts)),
            "canonical_raw_edge_count": int(len(canonical_raw)),
            "canonical_unique_edge_count": int(len(canonical_unique)),
            "canonical_integer_count_sum": int(canonical_counts.sum()),
            "all_edges_canonical_lo_lt_hi": bool(np.all(raw_edges[:, 0] < raw_edges[:, 1])),
            "track_sequence_ok": track_sequence_ok,
            "origin0_bp_positions_and_partial_ends_ok": expected_positions_ok,
            "no_trailing_bytes": True,
        },
    }


def output_header(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    header_struct = struct.Struct("<8s6I4f")
    if len(payload) < header_struct.size:
        raise ValueError(f"truncated output: {path}")
    magic, version, n_beads, n_binned, n_raw, n_iter, flags, native_unit, init_max, final_max, reserved = header_struct.unpack_from(payload)
    expected = header_struct.size + 2 * int(n_beads) * 3 * 4
    if len(payload) != expected:
        raise ValueError(f"output size mismatch: {path}")
    values = np.frombuffer(payload, dtype="<f4", count=2 * int(n_beads) * 3, offset=header_struct.size).copy()
    return {
        "path": rel(path), "sha256": sha256_file(path), "byte_size": len(payload),
        "magic_ascii": magic.rstrip(b"\x00").decode("ascii"), "version": int(version),
        "n_beads": int(n_beads), "n_binned_pairs": int(n_binned), "n_raw_pairs": int(n_raw),
        "n_iter": int(n_iter), "flags": int(flags), "native_unit": float(native_unit),
        "init_max_norm": float(init_max), "final_max_norm": float(final_max), "reserved": float(reserved),
        "init_bead_order": values[: int(n_beads) * 3].reshape(int(n_beads), 3).astype(np.float64),
        "final_bead_order": values[int(n_beads) * 3 :].reshape(int(n_beads), 3).astype(np.float64),
    }


def source_checks(source_paths: dict[str, Path]) -> dict[str, Any]:
    random_source = source_paths["random_native"].read_text(encoding="utf-8")
    bridge_source = source_paths["bridge"].read_text(encoding="utf-8")
    bin_source = source_paths["bin"].read_text(encoding="utf-8")
    fdg_source = source_paths["fdg"].read_text(encoding="utf-8")
    main_source = source_paths["main"].read_text(encoding="utf-8")
    count_source = source_paths["count"].read_text(encoding="utf-8")
    checks = {
        "python_grid_uses_origin0_bin_size": "np.arange(count, dtype=np.int64) * BIN_SIZE" in random_source and "beads_are_not_origin-0" not in random_source,
        "python_grid_writes_track_id_and_start_end": "beads[start:start + count, 0] = track" in random_source and "starts = np.arange(count, dtype=np.int64) * BIN_SIZE" in random_source and "ends = np.minimum(starts + BIN_SIZE, int(length))" in random_source,
        "python_mapping_track_2chr_plus_copy": "track = 2 * chromosome + copy" in random_source,
        "python_native_call_source_null_and_iterations": "hk_fdg(..., NULL, ..." in random_source and '"n_iter": 1000' in random_source,
        "bridge_validates_track_start_end": "bead inventory is not the complete origin-0 grid" in bridge_source and "validate_grid(lengths" in bridge_source,
        "bridge_builds_bmap_from_explicit_beads": "hk_bmap_gen_from_beads(dict, (int32_t)header.n_beads, beads" in bridge_source,
        "bridge_runs_from_scratch_source_null": "hk_fdg(&conf, target, NULL, &rng)" in bridge_source and "target_radius=%.9g iterations=%u seed=" in bridge_source,
        "bridge_zeroes_neighbor_support": "memset(p, 0, sizeof(*p))" in bridge_source and "p->_.phased_prob = 0.0f" in bridge_source,
        "014_main_computes_neighbors_before_bmap": "if (!(m->cols & 1<<6)) hk_pair_count_nei(m->n_pairs, m->pairs, radius, radius);" in main_source and "b = hk_bmap_gen" in main_source,
        "014_count_populates_neighbor_support": "pairs[st + a[i].i].n_nei = a[i].n;" in count_source,
        "fdg_uses_median_max_nei_for_contact_weight": "max_nei = ks_ksmall_int32_t(m->n_pairs, tmp, (int)(m->n_pairs * 0.5));" in fdg_source and "p->max_nei / max_nei" in fdg_source,
        "offcnt_groups_contiguous_beads_by_track": "m->beads[i].chr != m->beads[i-1].chr" in bin_source and "m->offcnt[m->beads[off].chr]" in bin_source,
        "fdg_backbone_uses_consecutive_bid_minus1_bid": "for (j = 1; j < cnt; ++j)" in fdg_source and "update_force(opt, x, bid - 1, bid, 1.0f, unit, d_opt, FORCE_BACKBONE" in fdg_source,
        "fdg_contacts_are_separate_loop": "for (i = 0; i < m->n_pairs; ++i) { // contact" in fdg_source,
        "fdg_saves_best_state_after_each_iteration": "if (s < best)" in fdg_source and "memcpy(best_x, m->x" in fdg_source and "memcpy(m->x, best_x" in fdg_source,
    }
    return {
        "checks": checks,
        "all_checks": all(checks.values()),
        "sha256": {key: sha256_file(path) for key, path in source_paths.items()},
        "line_evidence": {
            "random_native_grid_and_mapping": "random_native_init.py:228-297, 1169-1190",
            "random_native_call_and_normalization": "random_native_init.py:1193-1207, 1210-1247",
            "bridge_header_and_call": "fdg_random_bridge.c:162-190, 329-476",
            "offcnt_track_partition": "native/hickit/bin.c:38-50",
            "neighbor_count_and_fdg_weight": "native/hickit/main.c:240-246, native/hickit/count.c:132-158, native/hickit/fdg.c:622-630, 390-399",
            "backbone_force": "native/hickit/fdg.c:364-400",
            "best_state_return": "native/hickit/fdg.c:671-725",
        },
    }


def write_neighbor_tsv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    fields = ["state", "scope", "count", "finite", "min", "p10", "median", "mean", "p90", "p99", "max", "frac_lt_0.005", "frac_lt_0.01", "frac_lt_0.02", "frac_lt_0.05", "frac_gt_0.2"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for state, summary in summaries.items():
            for scope, values in (("all40tracks", summary["pooled"]), ("chr1", summary["by_chromosome"]["chr1"]["pooled"])):
                row = {"state": state, "scope": scope, **values}
                writer.writerow({key: repr(row.get(key, "")) if isinstance(row.get(key), float) else row.get(key, "") for key in fields})


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "random_native": RUN025 / "provenance" / "preflight_snapshot" / "random_native_init.py",
        "bridge": ROOT / "pr" / "native_bridge" / "fdg_random_bridge.c",
        "bin": ROOT / "native" / "hickit" / "bin.c",
        "fdg": ROOT / "native" / "hickit" / "fdg.c",
        "main": ROOT / "native" / "hickit" / "main.c",
        "count": ROOT / "native" / "hickit" / "count.c",
    }
    grid_path = RUN025 / "grid" / "fullgrid.npz"
    validation_path = RUN025 / "validation" / "POST020_ALLELE_ABLATION_NATIVE_VALIDATION.json"
    revision_path = RUN025 / "provenance" / "source_revision.json"
    preflight_config_path = RUN025 / "provenance" / "preflight_snapshot" / "config.json"
    frozen_config_path = RUN025 / "config.json"
    bridge_manifest_paths = {bundle: RUN025 / "bundles" / bundle / "graph" / "graph_manifest.json" for bundle in BUNDLES}
    canonical_paths = {bundle: RUN025 / "bundles" / bundle / "graph" / "canonical_edges.npz" for bundle in BUNDLES}
    bridge_paths = {bundle: RUN025 / "bundles" / bundle / "graph" / "bridge_input.rndbin" for bundle in BUNDLES}
    attempt_paths = {bundle: RUN025 / "bundles" / bundle / "native_attempt.json" for bundle in BUNDLES}
    output_paths = {bundle: RUN025 / "bundles" / bundle / "native_output.rndout" for bundle in BUNDLES}
    log_paths = {bundle: RUN025 / "logs" / f"{bundle}-native.stderr" for bundle in BUNDLES}
    raw_paths = {bundle: RUN025 / "bundles" / bundle / "raw_native.npz" for bundle in BUNDLES}
    x0_paths = {bundle: RUN025 / "bundles" / bundle / "x0_normalized.npz" for bundle in BUNDLES}
    x0_text_paths = {bundle: RUN025 / "bundles" / bundle / "x0_normalized.3dg" for bundle in BUNDLES}
    source014_path = RUN014 / "coords" / "random.3dg"
    source014_random_pairs_path = RUN014 / "work" / "random.pairs.gz"
    source014_native_log_paths = {name: RUN014 / "work" / f"{name}.hickit.log" for name in ("consensus", "random", "oracle")}
    extra_paths = {
        "grid": grid_path, "validation": validation_path, "revision": revision_path,
        "preflight_config": preflight_config_path, "frozen_config": frozen_config_path,
        "source014_random": source014_path,
        "source014_random_pairs": source014_random_pairs_path,
        **{f"source014_native_log_{name}": path for name, path in source014_native_log_paths.items()},
        **{f"graph_manifest_{bundle}": path for bundle, path in bridge_manifest_paths.items()},
        **{f"canonical_edges_{bundle}": path for bundle, path in canonical_paths.items()},
        **{f"bridge_input_{bundle}": path for bundle, path in bridge_paths.items()},
        **{f"native_attempt_{bundle}": path for bundle, path in attempt_paths.items()},
        **{f"native_output_{bundle}": path for bundle, path in output_paths.items()},
        **{f"native_stderr_{bundle}": path for bundle, path in log_paths.items()},
        **{f"raw_native_{bundle}": path for bundle, path in raw_paths.items()},
        **{f"x0_normalized_{bundle}": path for bundle, path in x0_paths.items()},
        **{f"x0_text_{bundle}": path for bundle, path in x0_text_paths.items()},
        **{f"source_{key}": path for key, path in source_paths.items()},
    }
    for key, path in extra_paths.items():
        if is_forbidden(path):
            raise AssertionError(f"forbidden path in native audit: {key} {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    freeze_files = [{"key": key, "path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for key, path in sorted(extra_paths.items())]
    freeze = {
        "schema": "reference-free-native-call-audit-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_file_count": len(freeze_files),
        "reference_opened": False, "phase_opened": False, "evaluation_outputs_opened": False,
        "native_rerun_called": False, "optimizer_called": False,
        "files": freeze_files,
    }
    freeze["freeze_sha256"] = hashlib.sha256(json.dumps(jsonable(freeze), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    freeze_path = AUDIT / "native_audit_frozen_inputs.json"
    write_json(freeze_path, freeze)

    grid = load_npz(grid_path)
    validation = read_json(validation_path)
    preflight_config = read_json(preflight_config_path)
    frozen_config = read_json(frozen_config_path)
    revision = read_json(revision_path)
    parsed014 = parse_3dg(source014_path, grid)
    source014_pair_header = read_pair_header(source014_random_pairs_path)
    source014_coordinates = parsed014["coordinates"]
    source014_valid = np.isfinite(source014_coordinates).all(axis=2)
    source014_center = source014_coordinates[source014_valid].mean(axis=0)
    source014_centered = source014_coordinates - source014_center
    source014_raw_centered_radius = float(np.linalg.norm(source014_centered[source014_valid], axis=1).max())
    source014_scale = 0.8 / source014_raw_centered_radius
    source014_centered_scaled = source014_centered * source014_scale
    source014_native = {name: parse_hickit_log(path) for name, path in source014_native_log_paths.items()}
    source_evidence = source_checks(source_paths)

    grid_checks = {
        "n_chromosomes": int(len(grid["chromosome_lengths"])) == 20,
        "n_loci": int(grid["n_bins"].sum()) == 2645,
        "n_beads": int(grid["beads"].shape[0]) == 5290,
        "n_tracks": int(len(grid["track_names"])) == 40,
        "origin0_positions": bool(
            np.array_equal(
                grid["positions"],
                np.concatenate(tuple(np.arange(int(count), dtype=np.int64) * BIN_SIZE for count in grid["n_bins"])),
            )
        ),
        "track_order": [str(value) for value in grid["track_names"]] == [f"c{chromosome:02d}{copy}" for chromosome in range(1, 21) for copy in ("a", "b")],
        "track_lengths_duplicate_per_copy": bool(np.array_equal(grid["track_lengths"], np.repeat(grid["chromosome_lengths"], 2))),
        "bead_inventory_shape": list(grid["beads"].shape) == [5290, 3],
    }
    grid_checks["all"] = all(grid_checks.values())

    native_calls: dict[str, Any] = {}
    bridge_audits: dict[str, Any] = {}
    neighbor_summaries: dict[str, dict[str, Any]] = {
        "014_random_source": adjacent_summary(source014_coordinates, grid),
        "014_random_source_centered_scaled": adjacent_summary(source014_centered_scaled, grid),
    }
    all_bundle_checks: dict[str, Any] = {}
    for bundle in BUNDLES:
        attempt = read_json(attempt_paths[bundle])
        manifest = read_json(bridge_manifest_paths[bundle])
        canonical = load_npz(canonical_paths[bundle])
        log = parse_native_log(log_paths[bundle])
        header = output_header(output_paths[bundle])
        raw = load_npz(raw_paths[bundle])
        x0 = load_npz(x0_paths[bundle])
        x0_text = parse_3dg(x0_text_paths[bundle], grid)
        bridge = parse_bridge(bridge_paths[bundle], grid, canonical)
        command = attempt["command"]
        output_hash_matches = attempt.get("output_sha256") == header["sha256"]
        raw_init_diff = float(np.max(np.abs(header["init_bead_order"] - raw["native_init_bead_order"])))
        raw_final_diff = float(np.max(np.abs(header["final_bead_order"] - raw["native_final_bead_order"])))
        native_init_cf = bead_to_copyfirst(header["init_bead_order"], grid)
        native_final_cf = bead_to_copyfirst(header["final_bead_order"], grid)
        x0_coords = np.asarray(x0["coordinates"], dtype=np.float64)
        x0_bead = np.asarray(x0["bead_order"], dtype=np.float64)
        x0_expected = native_final_cf - np.asarray(x0["center"], dtype=np.float64)
        x0_expected *= float(x0["scale"].ravel()[0])
        neighbor_summaries[f"025_x0_{bundle}"] = adjacent_summary(x0_coords, grid)
        centered_native_final = native_final_cf - np.asarray(x0["center"], dtype=np.float64)
        centered_native_final *= float(x0["scale"].ravel()[0])
        neighbor_summaries[f"025_native_final_centered_{bundle}"] = adjacent_summary(centered_native_final, grid)
        terminal = log["last_iteration"]
        call_checks = {
            "receipt_returncode_zero": attempt.get("returncode") == 0,
            "receipt_status_completed_zero": attempt.get("status") == "completed_zero",
            "exact_command_iterations_1000": "--iterations" in command and command[command.index("--iterations") + 1] == "1000",
            "exact_command_seed": "--seed" in command and command[command.index("--seed") + 1] == str(attempt["native_seed"]),
            "source_null_in_command_log": bool(log["mode_lines"] and "source=NULL" in log["mode_lines"][0]),
            "native_log_max_nei_zero": log["max_nei"] == 0,
            "backend_cpu_in_receipt_and_log": attempt.get("backend") == "CPU" and bool(log["mode_lines"] and "backend=CPU" in log["mode_lines"][0]),
            "threads_one_in_config": preflight_config["native"].get("threads") == 1,
            "receipt_iterations_1000": attempt.get("iterations") == 1000,
            "binary_output_hash_matches_receipt": output_hash_matches,
            "binary_header_magic_version_flags": header["magic_ascii"] == "RNDOUT1" and header["version"] == 1 and header["flags"] == 0,
            "binary_header_inventory": header["n_beads"] == 5290 and header["n_raw_pairs"] == 1265114 and header["n_iter"] == 1000,
            "stderr_reached_iter_1000": log["last_iteration_is_1000"],
            "native_arrays_match_binary": raw_init_diff == 0.0 and raw_final_diff == 0.0,
            "native_arrays_finite": bool(np.isfinite(header["init_bead_order"]).all() and np.isfinite(header["final_bead_order"]).all()),
            "native_to_copyfirst_finite": bool(np.isfinite(native_init_cf).all() and np.isfinite(native_final_cf).all()),
            "x0_matches_single_center_scale": float(np.max(np.abs(x0_coords - x0_expected))) == 0.0,
            "x0_text_matches_npz": float(np.max(np.abs(x0_text["coordinates"] - x0_coords))) == 0.0 and x0_text["track_order_ok"],
            "x0_strict_unit_ball": bool(np.all(np.linalg.norm(x0_coords, axis=2) < 1.0)),
            "x0_target_radius_0_8": abs(float(np.linalg.norm(x0_bead, axis=1).max()) - 0.8) <= 2e-6,
            "one_call_recorded": validation.get("native_calls_total") == 3 and attempt.get("retry_allowed") is False and attempt.get("retry", False) is False,
        }
        # 保存的 x0 bead-order 为 track-major；显式构造它以进行真正的相等性检查。
        expected_x0_bead = np.empty_like(x0_bead)
        for chromosome in range(20):
            slc = slice(int(grid["offsets"][chromosome]), int(grid["offsets"][chromosome + 1]))
            for copy in (0, 1):
                track = 2 * chromosome + copy
                expected_x0_bead[int(grid["track_offsets"][track]):int(grid["track_offsets"][track + 1])] = x0_coords[copy, slc]
        call_checks["x0_bead_order_matches_mapping"] = float(np.max(np.abs(x0_bead - expected_x0_bead))) == 0.0
        all_bundle_checks[bundle] = {**call_checks, "all": all(call_checks.values())}
        native_calls[bundle] = {
            "assignment_seed": attempt["assignment_seed"], "native_seed": attempt["native_seed"],
            "command": command, "receipt": {key: attempt.get(key) for key in ("returncode", "status", "iterations", "elapsed_seconds", "backend", "threads", "source", "input_sha256", "output_sha256")},
            "log": {key: log.get(key) for key in ("path", "sha256", "line_count", "max_nei", "max_nei_lines", "first_iteration", "last_iteration", "best_logged_iteration", "best_logged_rms", "last_iteration_is_1000")},
            "output_header": {key: value for key, value in header.items() if key not in ("init_bead_order", "final_bead_order")},
            "terminal_interpretation": "CPU hk_fdg completed exactly 1000 attempted FDG steps; output native_final is the best_x snapshot by returned RMS_force, while iter:1000 is the last attempted-step diagnostic; returncode=0 is budget completion, not convergence",
            "terminal_coordinates": {
                "raw_final_max_radius_from_header": header["final_max_norm"],
                "centered_raw_final_max_radius": float(np.linalg.norm(native_final_cf - x0["center"], axis=2).max()),
                "normalized_x0_max_radius": float(np.linalg.norm(x0_coords, axis=2).max()),
            },
            "last_attempt_log_metrics": terminal,
            "best_logged_iteration": log["best_logged_iteration"],
            "best_logged_rms": log["best_logged_rms"],
            "checks": call_checks,
        }
        bridge_audits[bundle] = {
            "manifest": {
                "force_graph_raw_records": manifest.get("force_graph_raw_records"),
                "force_graph_integer_count_sum": manifest.get("force_graph_integer_count_sum"),
                "force_graph_unique_edges": manifest.get("force_graph_unique_edges"),
                "same_bin_records_excluded": manifest.get("same_bin_records_excluded"),
                "zero_count_eligible_pairs_added": manifest.get("zero_count_eligible_pairs_added"),
                "native_not_called_prepare_metadata": manifest.get("native_not_called"),
            },
            "bridge": bridge,
            "checks": bridge["payload_checks"],
        }

    # 为 source 014 比较和三个 normalized starts 增加紧凑汇总。
    neighbor_rows = {}
    for state, summary in neighbor_summaries.items():
        neighbor_rows[state] = {
            "all40tracks": summary["pooled"],
            "chr1": summary["by_chromosome"]["chr1"]["pooled"],
        }
    write_neighbor_tsv(AUDIT / "native_neighbor_summary.tsv", neighbor_summaries)

    source014_neighbor_support = {
        "input_header": source014_pair_header,
        "logs": source014_native,
        "checks": {
            "all_three_logs_have_explicit_1000_step_commands": all(
                item["command"] is not None and "-n 1000" in item["command"] and "-b1m" in item["command"]
                for item in source014_native.values()
            ),
            "all_three_logs_have_nonzero_max_nei": all(item["max_nei"] is not None and item["max_nei"] > 0 for item in source014_native.values()),
            "014_random_input_has_no_neighbor_column": source014_pair_header["has_n_nei_column"] is False,
            "014_random_input_has_no_phase_columns": source014_pair_header["has_phase_columns"] is False,
            "all_three_logs_reach_iter_1000": all(item["last_iteration_is_1000"] for item in source014_native.values()),
        },
        "interpretation": "014 hickit logs report max_nei=19 for random, 69 for consensus, and 7 for oracle; main.c calls hk_pair_count_nei before hk_bmap_gen when -b is parsed, and count.c writes n_nei into each pair. In contrast, the 025 bridge memset-zeroes raw hk_pair records and its actual logs report max_nei=0, so fdg.c's median contact-weight scale makes every 025 contact k=1.",
    }
    source014_neighbor_support["checks"]["all"] = all(source014_neighbor_support["checks"].values())

    source_revision_summary = {
        "revision_id": revision.get("revision_id"),
        "status": revision.get("status"),
        "change_scope": revision.get("change_scope"),
        "forbidden_changes": revision.get("forbidden_changes"),
        "bundle1_native_source_sha256": revision.get("native_source_used_bundle1", {}).get("sha256"),
        "bundle2_bundle3_native_source_sha256": revision.get("new_source", {}).get("sha256"),
        "native_rerun_for_bundle1": revision.get("bug_fix_after_bundle1", {}).get("native_rerun_for_bundle1"),
        "interpretation": "bundle1 native output predates the Python postprocess fix; bundle2/3 use the atomic-receipt revision; revision metadata forbids graph, seed, C bridge, vendored-native, and reference/phase changes",
    }
    checks = {
        "grid_contract": grid_checks["all"],
        "014_source_grid_parse": parsed014["track_order_ok"] and parsed014["rows"] == parsed014["seen_count"] and parsed014["rows"] > 0,
        "source_linkage_checks": source_evidence["all_checks"],
        "014_neighbor_support_observed": source014_neighbor_support["checks"]["all"],
        "all_native_calls_and_terminal_logs": all(item["all"] for item in all_bundle_checks.values()),
        "all_bridge_payloads_match_canonical_and_grid": all(
            item["bridge"]["payload_checks"]["track_lengths_vs_grid_exact"]
            and item["bridge"]["payload_checks"]["beads_vs_grid_exact"]
            and item["bridge"]["payload_checks"]["raw_edges_vs_canonical_exact"]
            and item["bridge"]["payload_checks"]["unique_edges_vs_recomputed_exact"]
            and item["bridge"]["payload_checks"]["unique_counts_vs_recomputed_exact"]
            and item["bridge"]["payload_checks"]["all_edges_canonical_lo_lt_hi"]
            and item["bridge"]["payload_checks"]["track_sequence_ok"]
            and item["bridge"]["payload_checks"]["origin0_bp_positions_and_partial_ends_ok"]
            for item in bridge_audits.values()
        ),
        "all_neighbor_coordinates_finite": all(
            state.startswith("025_") and summary["pooled"].get("finite", 0) == summary["pooled"].get("count", -1)
            for state, summary in neighbor_summaries.items() if state.startswith("025_")
        ),
        "native_rerun_not_called_by_this_audit": True,
        "optimizer_not_called_by_this_audit": True,
        "no_forbidden_path": all(not is_forbidden(path) for path in extra_paths.values()),
    }
    result = {
        "schema": "reference-free-native-call-and-backbone-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": {
            "run": rel(RUN025), "native_calls_inspected": 3, "native_calls_rerun": 0,
            "reference_opened": False, "phase_opened": False, "evaluation_outputs_opened": False,
            "optimizer_called": False, "native_rerun_called": False,
        },
        "freeze": {"path": rel(freeze_path), "sha256": freeze["freeze_sha256"], "authorized_file_count": freeze["authorized_file_count"]},
        "checks": checks,
        "grid": {"summary": {"chromosomes": int(len(grid["chromosome_lengths"])), "loci": int(grid["n_bins"].sum()), "beads": int(grid["beads"].shape[0]), "tracks": int(len(grid["track_names"])), "bin_size_bp": BIN_SIZE, "origin_bp": 0, "track_order": [str(value) for value in grid["track_names"]]}, "checks": grid_checks, "file_sha256": sha256_file(grid_path)},
        "native_contract": {"preflight_config_native": preflight_config["native"], "frozen_config_native": frozen_config.get("native"), "validation_contract": validation["native_contract"]},
        "native_neighbor_support_comparison": source014_neighbor_support,
        "native_calls": native_calls,
        "bridge_and_backbone": {
            "per_bundle": bridge_audits,
            "backbone_source": source_evidence,
            "backbone_semantics": "native fdg.c iterates each dictionary track's contiguous offcnt range and applies FORCE_BACKBONE to consecutive bead ids bid-1,bid; contact pairs are processed in a separate loop",
            "contact_weight_semantics": "025 bridge raw hk_pair records are memset-zeroed, so max_nei=0 in all saved native logs and fdg.c's median max_nei scaling yields k=1 for every contact; standard 014 -b calls hk_pair_count_nei first and reports nonzero max_nei",
        },
        "fullgrid_bridge": {
            "header_contract": "RNDINP1, version 1, bin_size=1000000 bp, 40 tracks, 5290 beads, raw integer edges",
            "per_bundle": {bundle: bridge_audits[bundle]["bridge"] for bundle in BUNDLES},
        },
        "source_revision": source_revision_summary,
        "coordinate_parsing": {
            "014_random": {key: value for key, value in parsed014.items() if key != "coordinates"},
            "014_random_centered_scaled_to_x0_radius_0_8": {
                "center": source014_center.tolist(),
                "raw_centered_max_radius": source014_raw_centered_radius,
                "scale": float(source014_scale),
                "scaled_max_radius": float(np.linalg.norm(source014_centered_scaled[source014_valid], axis=1).max()),
                "missing_coordinates_remain_nan": bool(np.isnan(source014_centered_scaled).sum() == np.isnan(source014_coordinates).sum()),
            },
        },
        "neighbor_distance_summary": neighbor_rows,
        "neighbor_distance_definition": "Euclidean distance between coordinates at consecutive genomic bins within each ordered track; 025 x0 and centered/scaled 014 use radius 0.8, while raw 014 is retained to show the original scale; missing 014 bins yield nonfinite skipped edges",
        "limitations": [
            "The native output file is the best_x snapshot selected inside hk_fdg; the iter:1000 log is the last attempted-step diagnostic and need not describe the saved coordinate state exactly.",
            "The bridge/native path has no convergence test; 1000 attempted steps and returncode=0 therefore mean budget completion, not convergence.",
            "025 raw pair records have max_nei=0, unlike the standard 014 -b path; this is a confirmed native input/initialization pipeline difference, but this audit does not label it a protocol violation without the registered contract decision.",
            "Adjacent-distance summaries test ordering and local geometry only; they cannot establish biological correctness or L2 whole-chromosome recovery.",
            "014 random coordinates are a training-side source for scale/context, not a biological replicate or reference target.",
            "Bundle1 native output was not rerun after the postprocess mapping bug; its saved native bytes and corrected mapping are audited as-is.",
        ],
    }
    write_json(AUDIT / "native_audit.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "audit": rel(AUDIT)}, sort_keys=True))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
