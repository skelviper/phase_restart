"""Reconstruction V1 的冻结、无分相信息坐标初始化。

公开坐标表示采用 copy-first：``coords.shape ==
(2, N_loci, 3)``。轴 0 按任意 gauge 顺序 ``a, b``；locus 轴遵循传入的 SNP-free chromosome-header 顺序。本模块绝不打开带 phase 的 pairs 或 reference 3DG。只有 ``initialize_approved_candidate`` 执行文件 I/O，且只接受经过哈希核验的 014 blind consensus/random artifacts。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from copy import deepcopy

import numpy as np

from .paths import ROOT


DEFAULT_GATE_PATH = os.path.join(
    ROOT, "test_res", "014-20260912_153000-s0-genome-wide-fixed", "gate.json"
)
DEFAULT_COORD_DIR = os.path.join(
    ROOT, "test_res", "014-20260912_153000-s0-genome-wide-fixed", "coords"
)
APPROVED_SOURCES = {
    "consensus": {
        "candidate": "consensus",
        "tag": "consensus",
        "stage": "blind",
        "path": os.path.join(DEFAULT_COORD_DIR, "consensus.3dg"),
        "gate_path": DEFAULT_GATE_PATH,
        "sha256": "e76655732deb6b8386b1b77bc76ff45d7dba1384f6931337fee80d8f4aaa8e02",
        "base_seed": 1103,
    },
    "random": {
        "candidate": "random",
        "tag": "random",
        "stage": "blind",
        "path": os.path.join(DEFAULT_COORD_DIR, "random.3dg"),
        "gate_path": DEFAULT_GATE_PATH,
        "sha256": "9a48d73e1401e18349d11758e679da4c76da0904dbc467979079cb54bcd567d7",
        "base_seed": 2207,
    },
}

_CONTROL_SPACING_BP = 20_000_000
_TARGET_RADIUS = 0.8
_TRACK_RE = re.compile(r"^c([0-9]{2})([ab])$")


class InitializationError(RuntimeError):
    """当冻结 source 或坐标布局违反此契约时抛出。"""


def _sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_headers(names, header_lengths):
    if names is None or header_lengths is None:
        raise InitializationError("names and header_lengths must both be explicit")
    names = tuple(str(value) for value in names)
    lengths = tuple(int(value) for value in header_lengths)
    if not names or len(names) != len(lengths):
        raise InitializationError("names and header_lengths must have the same nonzero length")
    if len(set(names)) != len(names):
        raise InitializationError("chromosome names must be unique in header order")
    if any(length <= 0 for length in lengths):
        raise InitializationError("every chromosome header length must be positive")
    return names, lengths


def _as_bin_size(bin_size):
    try:
        value = int(bin_size)
    except (TypeError, ValueError) as exc:
        raise InitializationError("bin_size must be a positive integer number of base pairs") from exc
    if value <= 0:
        raise InitializationError("bin_size must be positive")
    return value


def _track_name(chromosome_index, copy_index):
    return "c%02d%s" % (chromosome_index + 1, "ab"[copy_index])


def full_grid_layout(names, header_lengths, bin_size):
    """构建不读取 contacts 或 models、从零开始的完整基因组网格。

    返回与全局 locus 轴对齐的 positions 和 chromosome indices。末端的部分 bin 始终保留。
    """
    names, lengths = _as_headers(names, header_lengths)
    bin_size = _as_bin_size(bin_size)
    positions, chromosome_index, offsets = [], [], [0]
    for chromosome, length in enumerate(lengths):
        count = (length + bin_size - 1) // bin_size
        positions.append(np.arange(count, dtype=np.int64) * bin_size)
        chromosome_index.append(np.full(count, chromosome, dtype=np.int32))
        offsets.append(offsets[-1] + count)
    return {
        "names": names,
        "header_lengths": lengths,
        "bin_size": bin_size,
        "positions": np.concatenate(positions),
        "chromosome_index": np.concatenate(chromosome_index),
        "locus_offsets": tuple(offsets),
        "n_loci": int(offsets[-1]),
        "track_order": tuple(
            _track_name(chromosome, copy)
            for chromosome in range(len(names))
            for copy in range(2)
        ),
    }


def swap_copy_first(coords):
    """返回 copy-first 坐标数组的全局 A/B gauge 互补。"""
    array = np.asarray(coords, dtype=float)
    if array.ndim != 3 or array.shape[0] != 2 or array.shape[2] != 3:
        raise InitializationError("copy-first coordinates must have shape (2, N_loci, 3)")
    return array[::-1].copy()


def _parse_track_name(track, n_chromosomes):
    match = _TRACK_RE.match(track)
    if match is None:
        raise InitializationError("unexpected coordinate track name: %s" % track)
    chromosome = int(match.group(1)) - 1
    copy = 0 if match.group(2) == "a" else 1
    if chromosome < 0 or chromosome >= n_chromosomes:
        raise InitializationError("track %s is outside the supplied header order" % track)
    return chromosome, copy


def _parse_3dg(path):
    headers = {}
    rows = {}
    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.startswith("#chromosome:"):
                fields = line.split()
                if len(fields) != 3:
                    raise InitializationError("malformed #chromosome header at line %d" % line_no)
                try:
                    length = int(fields[2])
                except ValueError as exc:
                    raise InitializationError("non-integer coordinate header length at line %d" % line_no) from exc
                if length <= 0 or not fields[1] or fields[1] in headers:
                    raise InitializationError("invalid or duplicate coordinate header at line %d" % line_no)
                headers[fields[1]] = length
                continue
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                raise InitializationError("malformed coordinate row at line %d" % line_no)
            try:
                position = int(fields[1])
                xyz = np.asarray([float(value) for value in fields[2:5]], dtype=float)
            except ValueError as exc:
                raise InitializationError("non-numeric coordinate row at line %d" % line_no) from exc
            if position < 0 or not np.isfinite(xyz).all():
                raise InitializationError("invalid coordinate row at line %d" % line_no)
            rows.setdefault(fields[0], []).append((position, xyz))
    if not headers or not rows:
        raise InitializationError("coordinate source must contain headers and coordinates")
    return headers, rows


def _normalise_track_rows(rows, expected_length, track):
    if not rows:
        raise InitializationError("track %s has no coordinates" % track)
    positions = np.asarray([row[0] for row in rows], dtype=np.int64)
    coords = np.asarray([row[1] for row in rows], dtype=float)
    if (positions >= expected_length).any() or not np.isfinite(coords).all():
        raise InitializationError("track %s has coordinates outside its header or nonfinite values" % track)
    order = np.argsort(positions, kind="stable")
    positions, coords = positions[order], coords[order]
    unique, starts, counts = np.unique(positions, return_index=True, return_counts=True)
    averaged = np.empty((len(unique), 3), dtype=float)
    for index, (start, count) in enumerate(zip(starts, counts)):
        averaged[index] = coords[start:start + count].mean(axis=0)
    return unique, averaged, counts


def _interpolate_track(source_positions, source_coords, source_counts, target_positions):
    """插值一条数值基因组轨迹，并记录每类填充。"""
    if len(source_positions) == 0:
        raise InitializationError("cannot interpolate an empty coordinate track")
    target_positions = np.asarray(target_positions, dtype=np.int64)
    exact = np.isin(target_positions, source_positions)
    outside = (target_positions < source_positions[0]) | (target_positions > source_positions[-1])
    endpoint = (~exact) & outside
    interpolated = (~exact) & (~outside)
    values = np.empty((len(target_positions), 3), dtype=float)
    for dimension in range(3):
        values[:, dimension] = np.interp(target_positions, source_positions, source_coords[:, dimension])
    repeated_positions = source_positions[source_counts > 1]
    repeated_target = np.isin(target_positions, repeated_positions)
    audit = {
        "source_rows": int(source_counts.sum()),
        "source_unique_positions": int(len(source_positions)),
        "source_duplicate_rows": int((source_counts - 1).sum()),
        "exact": int(exact.sum()),
        "interpolated": int(interpolated.sum()),
        "endpoint_filled": int(endpoint.sum()),
        "repeated_target_positions": int(repeated_target.sum()),
        "target_loci": int(len(target_positions)),
    }
    if audit["exact"] + audit["interpolated"] + audit["endpoint_filled"] != len(target_positions):
        raise InitializationError("interpolation categories do not cover the target grid")
    return values, audit, (~exact) | repeated_target


def _coerce_source_track(value, track):
    if isinstance(value, dict):
        positions = value.get("positions")
        coords = value.get("coords")
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        positions, coords = value
    else:
        raise InitializationError("source track %s must provide (positions, coords)" % track)
    positions = np.asarray(positions, dtype=np.int64)
    coords = np.asarray(coords, dtype=float)
    if positions.ndim != 1 or coords.shape != (len(positions), 3):
        raise InitializationError("source track %s positions/coords have incompatible shapes" % track)
    if not np.isfinite(coords).all() or (positions < 0).any():
        raise InitializationError("source track %s contains invalid values" % track)
    return positions, coords


def _expected_source_tracks(n_chromosomes, candidate):
    if candidate == "consensus":
        return tuple(_track_name(chromosome, 0) for chromosome in range(n_chromosomes))
    if candidate == "random":
        return tuple(
            _track_name(chromosome, copy)
            for chromosome in range(n_chromosomes)
            for copy in range(2)
        )
    raise InitializationError("candidate must be 'consensus' or 'random'")


def expand_tracks_to_full_grid(source_tracks, names, header_lengths, bin_size, candidate):
    """将内存中的 source 轨迹扩展为完整的 copy-first 网格。

    ``source_tracks`` 将 native/V1 track names 映射到 ``(positions, coords)``。positions 可以无序。返回坐标不会被居中、缩放、扰动或以其他方式优化。
    """
    layout = full_grid_layout(names, header_lengths, bin_size)
    names, lengths = layout["names"], layout["header_lengths"]
    expected = _expected_source_tracks(len(names), candidate)
    if set(source_tracks) != set(expected):
        raise InitializationError("source tracks do not match the approved %s inventory" % candidate)
    coords = np.empty((2, layout["n_loci"], 3), dtype=float)
    perturb_mask = np.zeros((2, layout["n_loci"]), dtype=bool)
    per_track = {}
    for chromosome, length in enumerate(lengths):
        locus_slice = slice(layout["locus_offsets"][chromosome], layout["locus_offsets"][chromosome + 1])
        target_positions = layout["positions"][locus_slice]
        source_by_copy = (0,) if candidate == "consensus" else (0, 1)
        expanded = {}
        for copy in source_by_copy:
            source_track = _track_name(chromosome, copy)
            positions, values = _coerce_source_track(source_tracks[source_track], source_track)
            rows = list(zip(positions.tolist(), values))
            unique, averaged, counts = _normalise_track_rows(rows, length, source_track)
            expanded[copy] = _interpolate_track(unique, averaged, counts, target_positions)
        for copy in (0, 1):
            source_copy = 0 if candidate == "consensus" else copy
            values, audit, new_or_repeated = expanded[source_copy]
            target_track = _track_name(chromosome, copy)
            coords[copy, locus_slice] = values
            # Consensus 只有一条 source trajectory，并复制到两个候选拷贝。
            perturb_mask[copy, locus_slice] = True if candidate == "consensus" else new_or_repeated
            audit = dict(audit)
            audit.update({
                "source_track": _track_name(chromosome, source_copy),
                "target_track": target_track,
                "source_copy_reused": bool(candidate == "consensus"),
                "perturb_loci": int(perturb_mask[copy, locus_slice].sum()),
            })
            per_track[target_track] = audit
    if not np.isfinite(coords).all():
        raise InitializationError("expanded coordinates must be finite")
    return {
        "coords": coords,
        "layout": layout,
        "per_track": per_track,
        "perturb_mask": perturb_mask,
    }


def _center_and_scale(coords, target_radius):
    flat = np.asarray(coords, dtype=float).reshape(-1, 3)
    center = flat.mean(axis=0)
    centered = np.asarray(coords, dtype=float) - center
    max_radius_before = float(np.linalg.norm(centered, axis=2).max())
    if not np.isfinite(max_radius_before) or max_radius_before <= 0:
        raise InitializationError("incoming coordinates have no nonzero finite global radius")
    scale = float(target_radius / max_radius_before)
    out = centered * scale
    return out, {
        "mode": "global_center_then_uniform_scale",
        "center": center.tolist(),
        "scale": scale,
        "max_radius_before": max_radius_before,
        "max_radius_after": float(np.linalg.norm(out, axis=2).max()),
        "target_radius": float(target_radius),
    }


def _control_half_difference(layout, rng):
    u = np.empty((layout["n_loci"], 3), dtype=float)
    per_chromosome = []
    target_rms = 0.06
    for chromosome, name in enumerate(layout["names"]):
        locus_slice = slice(layout["locus_offsets"][chromosome], layout["locus_offsets"][chromosome + 1])
        positions = layout["positions"][locus_slice]
        terminal = int(positions[-1])
        controls = np.arange(0, terminal + 1, _CONTROL_SPACING_BP, dtype=np.int64)
        if controls[-1] != terminal:
            controls = np.append(controls, terminal)
        control_values = rng.normal(size=(len(controls), 3))
        field = np.empty((len(positions), 3), dtype=float)
        for dimension in range(3):
            field[:, dimension] = np.interp(positions, controls, control_values[:, dimension])
        raw_rms = float(np.sqrt(np.mean(np.sum(field * field, axis=1))))
        if raw_rms <= 0 or not np.isfinite(raw_rms):
            raise InitializationError("random consensus half-difference unexpectedly vanished")
        field *= target_rms / raw_rms
        u[locus_slice] = field
        per_chromosome.append({
            "chromosome": name,
            "control_points": int(len(controls)),
            "raw_rms": raw_rms,
            "target_rms": target_rms,
            "actual_rms": float(np.sqrt(np.mean(np.sum(field * field, axis=1)))),
        })
    return u, per_chromosome


def _validate_output(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 3 or coords.shape[0] != 2 or coords.shape[2] != 3:
        raise InitializationError("output coordinates must have shape (2, N_loci, 3)")
    if not np.isfinite(coords).all():
        raise InitializationError("output coordinates contain nonfinite values")
    max_radius = float(np.linalg.norm(coords, axis=2).max())
    if not max_radius < 1.0:
        raise InitializationError("output coordinates must satisfy norm < 1")
    return max_radius


def _result(coords, layout, metadata):
    max_radius = _validate_output(coords)
    metadata = deepcopy(metadata)
    metadata.update({
        "coordinate_axis_order": "copy_first_a_b_locus_xyz",
        "coordinate_shape": [int(value) for value in coords.shape],
        "n_loci": int(layout["n_loci"]),
        "n_tracks": int(len(layout["track_order"])),
        "track_order": list(layout["track_order"]),
        "track_mapping": [
            {
                "track": _track_name(chromosome, copy),
                "chromosome_index": chromosome,
                "chromosome_name": layout["names"][chromosome],
                "copy_index": copy,
            }
            for chromosome in range(len(layout["names"]))
            for copy in range(2)
        ],
        "locus_offsets": list(layout["locus_offsets"]),
        "all_finite": True,
        "max_radius": max_radius,
    })
    return {
        "coords": np.asarray(coords, dtype=float),
        "positions": layout["positions"].copy(),
        "chromosome_index": layout["chromosome_index"].copy(),
        "names": tuple(layout["names"]),
        "header_lengths": tuple(layout["header_lengths"]),
        "bin_size": int(layout["bin_size"]),
        "locus_offsets": tuple(layout["locus_offsets"]),
        "metadata": metadata,
    }


def initialize_from_tracks(source_tracks, names, header_lengths, bin_size, candidate, seed,
                           target_radius=_TARGET_RADIUS):
    """从内存中的无 phase 轨迹创建归一化的 V1 初始化。

    这个通用函数从不读取路径，适用于 synthetic 测试和适配器。真实 P9016 启动必须调用 :func:`initialize_approved_candidate`，由它核验冻结 source。
    """
    if candidate not in ("consensus", "random"):
        raise InitializationError("candidate must be 'consensus' or 'random'")
    target_radius = float(target_radius)
    if not 0 < target_radius < 1:
        raise InitializationError("target_radius must be strictly between zero and one")
    started = time.monotonic()
    expanded = expand_tracks_to_full_grid(source_tracks, names, header_lengths, bin_size, candidate)
    layout = expanded["layout"]
    coords, first_normalization = _center_and_scale(expanded["coords"], target_radius)
    rng = np.random.default_rng(int(seed))
    u_metadata = None
    u = None
    if candidate == "consensus":
        u, per_chromosome = _control_half_difference(layout, rng)
        coords[0] += u
        coords[1] -= u
        u_metadata = {
            "seed": int(seed),
            "control_spacing_bp": _CONTROL_SPACING_BP,
            "radius_definition": "nuclear_ball_radius_1_before_final_global_rescale",
            "per_chromosome": per_chromosome,
            "nonzero": bool(np.linalg.norm(u) > 0),
        }
    l0 = float((2 * layout["n_loci"]) ** (-1.0 / 3.0))
    perturb_scale = 0.025 * l0
    perturb_mask = expanded["perturb_mask"]
    noise = rng.normal(scale=perturb_scale, size=coords.shape)
    coords[perturb_mask] += noise[perturb_mask]
    coords, final_normalization = _center_and_scale(coords, target_radius)
    if u_metadata is not None:
        for row in u_metadata["per_chromosome"]:
            row["post_final_field_rms"] = row["actual_rms"] * final_normalization["scale"]
    metadata = {
        "mode": "in_memory_shape_transform",
        "candidate": candidate,
        "seed": int(seed),
        "full_grid_from_zero": True,
        "l0": l0,
        "perturbation": {
            "scale": perturb_scale,
            "perturbed_coordinates": int(perturb_mask.sum()),
            "rule": "new_or_repeated_coordinates; consensus source is duplicated into both copies",
        },
        "initial_normalization": first_normalization,
        "final_normalization": final_normalization,
        "per_track": expanded["per_track"],
        "half_difference": u_metadata,
        "elapsed_sec": time.monotonic() - started,
        "replicate_interpretation": "optimization initialization only; not a biological replicate",
    }
    return _result(coords, layout, metadata)


def _resolve_gate_path(path):
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(ROOT, path))


def load_approved_source(candidate, names, header_lengths):
    """在 gate/hash 核验后加载唯一一个白名单中的 blind 014 source。"""
    if candidate not in APPROVED_SOURCES:
        raise InitializationError("only approved blind candidates consensus/random may be loaded")
    spec = APPROVED_SOURCES[candidate]
    names, lengths = _as_headers(names, header_lengths)
    source_path = os.path.abspath(spec["path"])
    gate_path = os.path.abspath(spec["gate_path"])
    if not os.path.isfile(source_path) or not os.path.isfile(gate_path):
        raise InitializationError("approved source or its gate file is unavailable")
    actual_digest = _sha256_file(source_path)
    if actual_digest != spec["sha256"]:
        raise InitializationError("approved source SHA256 mismatch for %s" % candidate)
    with open(gate_path) as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise InitializationError("approved source gate must contain a list")
    matches = [
        entry for entry in entries
        if entry.get("stage") == spec["stage"]
        and entry.get("tag") == spec["tag"]
        and entry.get("sha256") == spec["sha256"]
        and _resolve_gate_path(entry.get("path", "")) == source_path
    ]
    if len(matches) != 1:
        raise InitializationError("approved source gate entry is missing, ambiguous, or not blind")
    headers, rows = _parse_3dg(source_path)
    expected_tracks = _expected_source_tracks(len(names), candidate)
    if set(headers) != set(expected_tracks) or set(rows) != set(expected_tracks):
        raise InitializationError("approved source track inventory does not match candidate %s" % candidate)
    tracks = {}
    for track in expected_tracks:
        chromosome, _copy = _parse_track_name(track, len(names))
        if headers[track] != lengths[chromosome]:
            raise InitializationError("approved source header length mismatch for %s" % track)
        _normalise_track_rows(rows[track], lengths[chromosome], track)
        tracks[track] = (
            np.asarray([row[0] for row in rows[track]], dtype=np.int64),
            np.asarray([row[1] for row in rows[track]], dtype=float),
        )
    return {
        "tracks": tracks,
        "metadata": {
            "candidate": candidate,
            "source_path": source_path,
            "source_sha256": actual_digest,
            "gate_path": gate_path,
            "gate_entry": matches[0],
            "gate_stage": spec["stage"],
            "gate_tag": spec["tag"],
        },
    }


def initialize_approved_candidate(candidate, names, header_lengths, bin_size):
    """不做优化地初始化两个冻结的 P9016 blind 候选之一。

    consensus 候选使用 seed 1103 和平滑的非零 half-difference；random 使用 seed 2207，并保留其两条 native source 轨迹。两种选择都不代表独立拟合或生物学重复。
    """
    if candidate not in APPROVED_SOURCES:
        raise InitializationError("only consensus and random are approved real-data initializations")
    started = time.monotonic()
    source = load_approved_source(candidate, names, header_lengths)
    spec = APPROVED_SOURCES[candidate]
    result = initialize_from_tracks(
        source["tracks"], names, header_lengths, bin_size, candidate, seed=spec["base_seed"]
    )
    result["metadata"]["source"] = source["metadata"]
    result["metadata"]["mode"] = "approved_014_blind_initialization"
    result["metadata"]["transform_elapsed_sec"] = result["metadata"]["elapsed_sec"]
    result["metadata"]["elapsed_sec"] = time.monotonic() - started
    return result


def _coerce_layer_arrays(coords, positions, chromosome_index):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 3 or coords.shape[0] != 2 or coords.shape[2] != 3:
        raise InitializationError("warm-start coords must have shape (2, N, 3)")
    if not np.isfinite(coords).all():
        raise InitializationError("warm-start coords must be finite")
    n_loci = coords.shape[1]
    positions = np.asarray(positions, dtype=np.int64)
    chromosome_index = np.asarray(chromosome_index, dtype=np.int32)
    if positions.shape == (n_loci,):
        positions = np.broadcast_to(positions, (2, n_loci)).copy()
    if chromosome_index.shape == (n_loci,):
        chromosome_index = np.broadcast_to(chromosome_index, (2, n_loci)).copy()
    if positions.shape != (2, n_loci) or chromosome_index.shape != (2, n_loci):
        raise InitializationError("warm-start positions and chromosome_index must be (N,) or (2, N)")
    if (positions < 0).any() or (chromosome_index < 0).any():
        raise InitializationError("warm-start positions/chromosome_index must be nonnegative")
    return coords, positions, chromosome_index


def warm_start_from_layer(coords, positions, chromosome_index, names, header_lengths, bin_size,
                          candidate_base_seed):
    """将上一层的 40-track 层插值到更细的完整网格。

    保留传入的全细胞坐标系和尺度。只扰动新引入/重复的位点，然后将达到或超出单位球的坐标径向裁剪到 ``1 - 1e-6``。
    """
    started = time.monotonic()
    layout = full_grid_layout(names, header_lengths, bin_size)
    incoming, source_positions, source_chromosome = _coerce_layer_arrays(
        coords, positions, chromosome_index
    )
    if (source_chromosome >= len(layout["names"])).any():
        raise InitializationError("warm-start chromosome_index exceeds supplied header inventory")
    out = np.empty((2, layout["n_loci"], 3), dtype=float)
    perturb_mask = np.zeros((2, layout["n_loci"]), dtype=bool)
    per_track = {}
    for chromosome, length in enumerate(layout["header_lengths"]):
        target_slice = slice(layout["locus_offsets"][chromosome], layout["locus_offsets"][chromosome + 1])
        target_positions = layout["positions"][target_slice]
        for copy in (0, 1):
            keep = source_chromosome[copy] == chromosome
            if not keep.any():
                raise InitializationError("warm-start is missing track %s" % _track_name(chromosome, copy))
            raw_positions = source_positions[copy, keep]
            raw_coords = incoming[copy, keep]
            rows = list(zip(raw_positions.tolist(), raw_coords))
            unique, averaged, counts = _normalise_track_rows(
                rows, length, _track_name(chromosome, copy)
            )
            values, audit, new_or_repeated = _interpolate_track(
                unique, averaged, counts, target_positions
            )
            out[copy, target_slice] = values
            perturb_mask[copy, target_slice] = new_or_repeated
            audit.update({
                "source_track": _track_name(chromosome, copy),
                "target_track": _track_name(chromosome, copy),
                "source_copy_reused": False,
                "perturb_loci": int(new_or_repeated.sum()),
            })
            per_track[_track_name(chromosome, copy)] = audit
    l0 = float((2 * layout["n_loci"]) ** (-1.0 / 3.0))
    perturb_scale = 0.025 * l0
    seed = 3301 + layout["bin_size"] // 1_000_000 + int(candidate_base_seed)
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=perturb_scale, size=out.shape)
    out[perturb_mask] += noise[perturb_mask]
    limit = 1.0 - 1e-6
    radii = np.linalg.norm(out, axis=2)
    clipped = radii >= limit
    if clipped.any():
        out[clipped] *= (limit / radii[clipped])[:, None]
    metadata = {
        "mode": "multiresolution_warm_start",
        "candidate_base_seed": int(candidate_base_seed),
        "seed": int(seed),
        "full_grid_from_zero": True,
        "l0": l0,
        "perturbation": {
            "scale": perturb_scale,
            "perturbed_coordinates": int(perturb_mask.sum()),
            "rule": "new_or_repeated_coordinates_only",
        },
        "normalization": {
            "mode": "preserve_previous_all_cell_frame_and_scale",
            "incoming_max_radius": float(np.linalg.norm(incoming, axis=2).max()),
            "radial_clip_limit": limit,
            "clipped_coordinates": int(clipped.sum()),
        },
        "per_track": per_track,
        "elapsed_sec": time.monotonic() - started,
        "replicate_interpretation": "optimization initialization only; not a biological replicate",
    }
    return _result(out, layout, metadata)
