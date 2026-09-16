#!/usr/bin/env python3
"""Prepare and gate paired random native-FDG starts without phase/reference input.

Preparation uses the frozen SNP-free ledger and the exact ``genome.random_assignment``
rule.  It canonicalizes copy labels in integer graph space before writing an independent
RNDINP1 bridge blob.  The native bridge is invoked only with ``--run-released`` and a
configuration explicitly released by the parent protocol owner.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pr import genome  # noqa: E402
from pr.contact_model import q_from_p  # noqa: E402


BIN_SIZE = 1_000_000
N_CHROMOSOMES = 20
N_TRACKS = 40
NORMALIZED_MAX_RADIUS = 0.8
P_INIT = 0.75
Q_INIT = q_from_p(P_INIT)
INPUT_SHA256 = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"
RAW_RECORDS = 1_703_888
CIS_RECORDS = 1_135_454
SAME_BIN_RECORDS = 438_774
CIS_OFFDIAG_RECORDS = 696_680
INTER_RECORDS = 568_434
FORCE_RECORDS = CIS_OFFDIAG_RECORDS + INTER_RECORDS
EXPECTED_CHROMOSOMES = tuple([f"chr{i}" for i in range(1, 20)] + ["chrX"])
SEED_BUNDLES = (
    ("bundle1", 250101, 250201),
    ("bundle2", 250102, 250202),
    ("bundle3", 250103, 250203),
)

INPUT_MAGIC = b"RNDINP1\0"
OUTPUT_MAGIC = b"RNDOUT1\0"
INPUT_VERSION = 1
OUTPUT_VERSION = 1

NATIVE_OBJECTS = (
    "sdict.o", "io.o", "pair.o", "count.o", "phase.o", "bin.o", "fdg.o",
    "image.o", "view3d.o", "fdg_gpu_stub.o",
)
NATIVE_DEFAULTS = {
    "target_radius": 10.0,
    "n_iter": 1000,
    "step": 0.01,
    "coef_moment": 0.9,
    "max_f": 50.0,
    "contact_target": 0.0,
    "k_rel_rep": 0.05,
    "d_r": 2.0,
    "k_bend": 0.0,
    "k_confine": 0.0,
    "d_confine": 12.0,
    "d_b1": 0.1,
    "d_b2": 1.1,
    "d_c1": 0.5,
    "d_c2": 1.5,
    "d_c3": 2.0,
    "derived_c_c1": 1.5,
    "derived_c_c2": 0.125,
    "backend": "CPU",
    "threads": 1,
    "source": None,
    "initialization": "hk_fdg_init(rng, n_beads, target_radius), via hk_fdg(..., NULL, ...)",
}


class RandomNativeError(RuntimeError):
    """Raised when the random-native preflight contract is violated."""


class CanonicalizationError(RandomNativeError):
    """Raised for an unresolved copy-label signature tie."""

    def __init__(self, chromosome: int, reason: str):
        self.chromosome = int(chromosome)
        self.reason = str(reason)
        super().__init__(f"chr{chromosome + 1} canonicalization failed: {reason}")


@dataclass(frozen=True)
class FullGrid:
    lengths: np.ndarray
    n_bins: np.ndarray
    offsets: np.ndarray
    positions: np.ndarray
    chromosome_index: np.ndarray
    beads: np.ndarray
    track_names: tuple[str, ...]
    track_lengths: np.ndarray
    track_offsets: np.ndarray

    @property
    def n_chromosomes(self) -> int:
        return int(len(self.lengths))

    @property
    def n_loci(self) -> int:
        return int(len(self.positions))

    @property
    def n_beads(self) -> int:
        return int(len(self.beads))

    @property
    def n_tracks(self) -> int:
        return int(len(self.track_names))

    def chromosome_slice(self, chromosome: int) -> slice:
        start = int(self.offsets[chromosome])
        return slice(start, start + int(self.n_bins[chromosome]))

    def track_slice(self, track: int) -> slice:
        start = int(self.track_offsets[track])
        return slice(start, start + int(self.n_bins[track // 2]))


@dataclass(frozen=True)
class RawContacts:
    ci: np.ndarray
    p1: np.ndarray
    cj: np.ndarray
    p2: np.ndarray
    record_id: np.ndarray
    cis: np.ndarray
    same_bin: np.ndarray

    @property
    def n_records(self) -> int:
        return int(len(self.ci))


@dataclass(frozen=True)
class CanonicalGraph:
    assignment_k1: np.ndarray
    assignment_k2: np.ndarray
    canonical_k1: np.ndarray
    canonical_k2: np.ndarray
    swapped_by_chromosome: np.ndarray
    raw_edges: np.ndarray
    raw_record_ids: np.ndarray
    unique_edges: np.ndarray
    unique_counts: np.ndarray
    bridge_blob: bytes
    signature_hash: str
    signature_rows: tuple[dict[str, Any], ...]
    swap_reason: tuple[str, ...]

    @property
    def n_raw_edges(self) -> int:
        return int(len(self.raw_edges))

    @property
    def n_unique_edges(self) -> int:
        return int(len(self.unique_edges))

    @property
    def total_count(self) -> int:
        return int(self.unique_counts.sum(dtype=np.int64))


@dataclass(frozen=True)
class NativeOutput:
    n_beads: int
    n_binned_pairs: int
    n_raw_pairs: int
    n_iter: int
    native_unit: float
    init_max_norm: float
    final_max_norm: float
    native_init_bead_order: np.ndarray
    native_final_bead_order: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    return sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def relpath(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def make_full_grid(lengths: Sequence[int]) -> FullGrid:
    values = np.asarray(lengths, dtype=np.int64)
    if values.ndim != 1 or len(values) != N_CHROMOSOMES or np.any(values <= 0):
        raise RandomNativeError("P9016 grid requires 20 positive chromosome lengths")
    n_bins = ((values + BIN_SIZE - 1) // BIN_SIZE).astype(np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(n_bins)))
    positions = np.concatenate(tuple(
        np.arange(int(count), dtype=np.int64) * BIN_SIZE for count in n_bins
    ))
    chromosome_index = np.concatenate(tuple(
        np.full(int(count), chromosome, dtype=np.int32)
        for chromosome, count in enumerate(n_bins)
    ))
    track_names = tuple(
        f"c{chromosome + 1:02d}{copy_name}"
        for chromosome in range(len(values))
        for copy_name in ("a", "b")
    )
    track_n_bins = np.repeat(n_bins, 2)
    track_offsets = np.concatenate((np.asarray([0], dtype=np.int64),
                                    np.cumsum(track_n_bins)))
    track_lengths = np.repeat(values, 2)
    beads = np.empty((int(track_offsets[-1]), 3), dtype=np.int32)
    for track, length in enumerate(track_lengths):
        start = int(track_offsets[track])
        count = int(track_n_bins[track])
        starts = np.arange(count, dtype=np.int64) * BIN_SIZE
        ends = np.minimum(starts + BIN_SIZE, int(length))
        beads[start:start + count, 0] = track
        beads[start:start + count, 1] = starts.astype(np.int32)
        beads[start:start + count, 2] = ends.astype(np.int32)
    grid = FullGrid(
        lengths=values.copy(),
        n_bins=n_bins,
        offsets=offsets,
        positions=positions,
        chromosome_index=chromosome_index,
        beads=beads,
        track_names=track_names,
        track_lengths=track_lengths,
        track_offsets=track_offsets,
    )
    validate_full_grid(grid)
    return grid


def validate_full_grid(grid: FullGrid) -> None:
    if grid.n_chromosomes != N_CHROMOSOMES or grid.n_tracks != N_TRACKS:
        raise RandomNativeError("grid is not 20 chromosomes x 40 tracks")
    if grid.n_loci != 2645 or grid.n_beads != 5290:
        raise RandomNativeError(
            f"P9016 grid inventory is {grid.n_loci} loci/{grid.n_beads} beads, "
            "expected 2645/5290"
        )
    if not np.array_equal(grid.positions, np.concatenate(tuple(
            np.arange(int(count), dtype=np.int64) * BIN_SIZE for count in grid.n_bins))):
        raise RandomNativeError("grid positions are not origin-0 complete bins")
    for track, length in enumerate(grid.track_lengths):
        values = grid.beads[grid.track_slice(track)]
        count = (int(length) + BIN_SIZE - 1) // BIN_SIZE
        starts = np.arange(count, dtype=np.int64) * BIN_SIZE
        ends = np.minimum(starts + BIN_SIZE, int(length))
        if len(values) != count:
            raise RandomNativeError("track count is not ceil(length/bin_size)")
        if not np.array_equal(values[:, 0], np.full(count, track, dtype=np.int32)):
            raise RandomNativeError("bead track order changed")
        if not np.array_equal(values[:, 1], starts.astype(np.int32)):
            raise RandomNativeError("bead starts are not origin-0")
        if not np.array_equal(values[:, 2], ends.astype(np.int32)):
            raise RandomNativeError("terminal partial bin was not retained")


def grid_audit(grid: FullGrid, grid_path: Path) -> dict[str, Any]:
    np.savez_compressed(
        grid_path,
        chromosome_lengths=grid.lengths,
        n_bins=grid.n_bins,
        offsets=grid.offsets,
        positions=grid.positions,
        chromosome_index=grid.chromosome_index,
        beads=grid.beads,
        track_names=np.asarray(grid.track_names),
        track_lengths=grid.track_lengths,
        track_offsets=grid.track_offsets,
    )
    return {
        "bin_size_bp": BIN_SIZE,
        "origin_bp": 0,
        "bin_rule": "ceil(length_bp/bin_size_bp), start=bin*bin_size_bp, end=min((bin+1)*bin_size_bp,length_bp)",
        "n_chromosomes": grid.n_chromosomes,
        "n_loci": grid.n_loci,
        "n_physical_beads": grid.n_beads,
        "n_tracks": grid.n_tracks,
        "track_order": list(grid.track_names),
        "chromosome_lengths_bp": [int(x) for x in grid.lengths],
        "n_bins_per_chromosome": [int(x) for x in grid.n_bins],
        "locus_offsets": [int(x) for x in grid.offsets],
        "track_offsets": [int(x) for x in grid.track_offsets],
        "explicit_bead_inventory_sha256": sha256_array(grid.beads, "<i4"),
        "artifact": relpath(grid_path, ROOT),
        "artifact_sha256": sha256_file(grid_path),
        "observed_tracks_used_to_make_grid": False,
        "dummy_contacts_added": False,
    }


def raw_record_order_hash(records: RawContacts) -> str:
    digest = hashlib.sha256()
    for values, dtype in ((records.ci, "<i4"), (records.p1, "<i8"),
                          (records.cj, "<i4"), (records.p2, "<i8")):
        digest.update(np.asarray(values, dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


def load_frozen_contacts(input_path: Path, raw_path: Path) -> tuple[RawContacts, dict[str, Any]]:
    if sha256_file(input_path) != INPUT_SHA256:
        raise RandomNativeError("SNP-free input SHA differs from frozen P9016 digest")
    lengths = genome.chrom_lengths(str(input_path))
    if tuple(name for name, _ in lengths) != EXPECTED_CHROMOSOMES:
        raise RandomNativeError("chromosome headers are not the frozen numeric order")
    data = genome.load_all(str(input_path))
    n = len(data["ci"])
    if n != RAW_RECORDS:
        raise RandomNativeError(f"raw record count is {n}, expected {RAW_RECORDS}")
    cis = np.asarray(data["cis"], dtype=bool)
    same_bin = cis & ((data["p1"] // BIN_SIZE) == (data["p2"] // BIN_SIZE))
    if int(cis.sum()) != CIS_RECORDS:
        raise RandomNativeError("cis record count differs from frozen ledger")
    if int(same_bin.sum()) != SAME_BIN_RECORDS:
        raise RandomNativeError("same-bin ledger count differs from frozen ledger")
    if int((cis & ~same_bin).sum()) != CIS_OFFDIAG_RECORDS:
        raise RandomNativeError("cis off-diagonal ledger count differs from frozen ledger")
    if int((~cis).sum()) != INTER_RECORDS:
        raise RandomNativeError("inter ledger count differs from frozen ledger")
    if np.any(data["ci"] > data["cj"]):
        raise RandomNativeError("inter records are not in numeric chromosome order")
    if np.any(cis & (data["p1"] > data["p2"])):
        raise RandomNativeError("cis records are not in numeric coordinate order")
    records = RawContacts(
        ci=np.asarray(data["ci"], dtype=np.int32),
        p1=np.asarray(data["p1"], dtype=np.int64),
        cj=np.asarray(data["cj"], dtype=np.int32),
        p2=np.asarray(data["p2"], dtype=np.int64),
        record_id=np.arange(n, dtype=np.int64),
        cis=cis,
        same_bin=same_bin,
    )
    np.savez_compressed(
        raw_path,
        record_id=records.record_id,
        ci=records.ci,
        p1=records.p1,
        cj=records.cj,
        p2=records.p2,
        cis=records.cis,
        same_bin=records.same_bin,
    )
    audit = {
        "path": str(input_path.resolve()),
        "sha256": sha256_file(input_path),
        "raw_records": records.n_records,
        "cis_records": int(records.cis.sum()),
        "same_bin_records": int(records.same_bin.sum()),
        "cis_offdiag_records": int(np.sum(records.cis & ~records.same_bin)),
        "inter_records": int(np.sum(~records.cis)),
        "force_records": int(np.sum(~records.same_bin)),
        "record_order": "original SNP-free file row order; record_id=0..n-1",
        "record_order_numeric_coordinates_sha256": raw_record_order_hash(records),
        "numeric_coordinate_encoding": {
            "ci_cj": "signed int32 chromosome indices",
            "p1_p2": "signed int64 base-pair coordinates",
            "same_bin": "cis and floor_divide(numeric_position, 1000000) equality",
        },
        "phase_or_reference_read": False,
        "labels_provenance": "none; no phase-bearing payload opened",
        "artifact": relpath(raw_path, ROOT),
        "artifact_sha256": sha256_file(raw_path),
    }
    return records, audit


def _encode_rows(rows: np.ndarray) -> bytes:
    values = np.asarray(rows, dtype="<u4")
    if values.ndim != 2 or values.shape[1] == 0:
        raise RandomNativeError("signature rows must have a non-empty tuple width")
    if len(values) > 0xFFFFFFFF:
        raise RandomNativeError("signature row count exceeds uint32")
    return struct.pack("<I", len(values)) + np.ascontiguousarray(values).tobytes(order="C")


def _sparse_rows(columns: Sequence[np.ndarray], width: int) -> tuple[np.ndarray, bytes]:
    if not columns:
        rows = np.empty((0, width), dtype=np.uint32)
    else:
        rows_in = np.column_stack(tuple(np.asarray(c, dtype=np.int64) for c in columns))
        if rows_in.ndim != 2 or rows_in.shape[1] != width:
            raise RandomNativeError("signature tuple width mismatch")
        if len(rows_in):
            unique, counts = np.unique(rows_in, axis=0, return_counts=True)
            rows = np.column_stack((unique, counts)).astype(np.uint32, copy=False)
        else:
            rows = np.empty((0, width + 1), dtype=np.uint32)
    return rows, _encode_rows(rows)


def cis_signature(records: RawContacts, k1: np.ndarray, k2: np.ndarray,
                  chromosome: int, copy: int) -> tuple[np.ndarray, bytes]:
    mask = (records.cis & ~records.same_bin & (records.ci == chromosome) &
            (records.cj == chromosome) & (k1 == copy) & (k2 == copy))
    local0 = records.p1[mask] // BIN_SIZE
    local1 = records.p2[mask] // BIN_SIZE
    if np.any(local0 >= local1):
        raise RandomNativeError("cis off-diagonal signature contains non-increasing bins")
    return _sparse_rows((local0, local1), 2)


def incidence_signature(records: RawContacts, k1: np.ndarray, k2: np.ndarray,
                        chromosome: int, copy: int) -> tuple[np.ndarray, bytes]:
    inter = ~records.cis
    left = inter & (records.ci == chromosome) & (k1 == copy)
    right = inter & (records.cj == chromosome) & (k2 == copy)
    other_chr = np.concatenate((records.cj[left], records.ci[right]))
    self_local = np.concatenate((records.p1[left] // BIN_SIZE,
                                 records.p2[right] // BIN_SIZE))
    other_local = np.concatenate((records.p2[left] // BIN_SIZE,
                                  records.p1[right] // BIN_SIZE))
    # Partner copy is intentionally absent from this tiebreak tuple.
    return _sparse_rows((other_chr, self_local, other_local), 3)


def _signature_digest(cis0: bytes, cis1: bytes, inc0: bytes, inc1: bytes) -> str:
    return sha256_bytes(b"CIS0" + cis0 + b"CIS1" + cis1 +
                        b"INC0" + inc0 + b"INC1" + inc1)


def _validate_assignments(records: RawContacts, k1: np.ndarray, k2: np.ndarray) -> None:
    if k1.shape != (records.n_records,) or k2.shape != (records.n_records,):
        raise RandomNativeError("assignment arrays are not record-aligned")
    if np.any((k1 < 0) | (k1 > 1) | (k2 < 0) | (k2 > 1)):
        raise RandomNativeError("assignment contains a copy outside {0,1}")
    if np.any(records.cis & (k1 != k2)):
        raise RandomNativeError("cis assignment violates shared-random-copy rule")


def _apply_swaps(records: RawContacts, k1: np.ndarray, k2: np.ndarray,
                 swap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out1 = np.asarray(k1, dtype=np.int8).copy()
    out2 = np.asarray(k2, dtype=np.int8).copy()
    for chromosome, bit in enumerate(np.asarray(swap, dtype=bool)):
        if bit:
            left = records.ci == chromosome
            right = records.cj == chromosome
            out1[left] = 1 - out1[left]
            out2[right] = 1 - out2[right]
    return out1, out2


def _build_edges(records: RawContacts, grid: FullGrid, k1: np.ndarray,
                 k2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    included = ~records.same_bin
    ids = np.flatnonzero(included)
    if len(ids) == 0:
        raise RandomNativeError("force graph has no off-diagonal records")
    track0 = 2 * records.ci[ids].astype(np.int64) + k1[ids].astype(np.int64)
    track1 = 2 * records.cj[ids].astype(np.int64) + k2[ids].astype(np.int64)
    local0 = records.p1[ids] // BIN_SIZE
    local1 = records.p2[ids] // BIN_SIZE
    bid0 = grid.track_offsets[track0] + local0
    bid1 = grid.track_offsets[track1] + local1
    lo = np.minimum(bid0, bid1)
    hi = np.maximum(bid0, bid1)
    if np.any(lo == hi) or np.any(hi >= grid.n_beads):
        raise RandomNativeError("canonical assignment produced a self/out-of-grid edge")
    order = np.lexsort((records.record_id[ids], hi, lo))
    raw_edges = np.column_stack((lo[order], hi[order])).astype("<u4", copy=False)
    raw_ids = records.record_id[ids][order].astype("<i8", copy=False)
    unique_edges, unique_counts = np.unique(raw_edges, axis=0, return_counts=True)
    return (raw_edges, raw_ids, unique_edges.astype("<u4", copy=False),
            unique_counts.astype("<i8", copy=False))


def pack_bridge_input(grid: FullGrid, raw_edges: np.ndarray) -> bytes:
    edges = np.asarray(raw_edges, dtype="<u4")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
        raise RandomNativeError("bridge edge table must be non-empty (n,2)")
    if np.any(edges[:, 0] >= edges[:, 1]) or np.any(edges[:, 1] >= grid.n_beads):
        raise RandomNativeError("bridge edge table is not canonical")
    header = struct.pack("<8s6I", INPUT_MAGIC, INPUT_VERSION, BIN_SIZE,
                         grid.n_tracks, grid.n_beads, len(edges), 0)
    return b"".join((
        header,
        np.asarray(grid.track_lengths, dtype="<i4").tobytes(order="C"),
        np.asarray(grid.beads, dtype="<i4").tobytes(order="C"),
        np.ascontiguousarray(edges).tobytes(order="C"),
    ))


def canonicalize_assignment(records: RawContacts, grid: FullGrid,
                            k1: np.ndarray, k2: np.ndarray) -> CanonicalGraph:
    _validate_assignments(records, k1, k2)
    swaps = np.zeros(grid.n_chromosomes, dtype=bool)
    signature_rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    signature_parts: list[bytes] = []
    for chromosome in range(grid.n_chromosomes):
        cis_entries = []
        inc_entries = []
        for copy in (0, 1):
            cis_rows, cis_bytes = cis_signature(records, k1, k2, chromosome, copy)
            inc_rows, inc_bytes = incidence_signature(records, k1, k2, chromosome, copy)
            cis_entries.append((cis_rows, cis_bytes))
            inc_entries.append((inc_rows, inc_bytes))
        cis0, cis1 = cis_entries[0][1], cis_entries[1][1]
        inc0, inc1 = inc_entries[0][1], inc_entries[1][1]
        signature_hash = _signature_digest(cis0, cis1, inc0, inc1)
        signature_parts.extend((cis0, cis1, inc0, inc1))
        if cis0 < cis1:
            reason = "cis_signature_lower_copy0"
        elif cis1 < cis0:
            swaps[chromosome] = True
            reason = "cis_signature_lower_copy1"
        elif inc0 < inc1:
            reason = "incidence_tiebreak_lower_copy0_partner_copy_ignored"
        elif inc1 < inc0:
            swaps[chromosome] = True
            reason = "incidence_tiebreak_lower_copy1_partner_copy_ignored"
        else:
            raise CanonicalizationError(
                chromosome,
                "cis and partner-copy-ignored inter incidence signatures are identical; "
                "seed invalid/preflight failed",
            )
        reasons.append(reason)
        signature_rows.append({
            "chromosome_index": chromosome,
            "chromosome": EXPECTED_CHROMOSOMES[chromosome]
            if chromosome < len(EXPECTED_CHROMOSOMES) else f"chr{chromosome + 1}",
            "canonical_copy": "b" if swaps[chromosome] else "a",
            "swap": int(swaps[chromosome]),
            "reason": reason,
            "signature_sha256": signature_hash,
            "cis_signature_sha256": [sha256_bytes(cis0), sha256_bytes(cis1)],
            "incidence_signature_sha256": [sha256_bytes(inc0), sha256_bytes(inc1)],
            "cis_rows": [int(len(item[0])) for item in cis_entries],
            "incidence_rows": [int(len(item[0])) for item in inc_entries],
        })
    canonical_k1, canonical_k2 = _apply_swaps(records, k1, k2, swaps)
    raw_edges, raw_ids, unique_edges, unique_counts = _build_edges(
        records, grid, canonical_k1, canonical_k2)
    if int(len(raw_edges)) != int(np.sum(~records.same_bin)):
        raise RandomNativeError("canonical force graph dropped same-bin/off-diagonal ledger rows")
    if int(unique_counts.sum(dtype=np.int64)) != FORCE_RECORDS and records.n_records == RAW_RECORDS:
        raise RandomNativeError("real force graph total is not 1,265,114")
    blob = pack_bridge_input(grid, raw_edges)
    return CanonicalGraph(
        assignment_k1=np.asarray(k1, dtype=np.int8).copy(),
        assignment_k2=np.asarray(k2, dtype=np.int8).copy(),
        canonical_k1=canonical_k1,
        canonical_k2=canonical_k2,
        swapped_by_chromosome=swaps,
        raw_edges=raw_edges,
        raw_record_ids=raw_ids,
        unique_edges=unique_edges,
        unique_counts=unique_counts,
        bridge_blob=blob,
        signature_hash=sha256_bytes(b"".join(signature_parts)),
        signature_rows=tuple(signature_rows),
        swap_reason=tuple(reasons),
    )


def assignment_swap(records: RawContacts, k1: np.ndarray, k2: np.ndarray,
                    chromosomes: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    bits = np.zeros(N_CHROMOSOMES, dtype=bool)
    for chromosome in chromosomes:
        if chromosome < 0 or chromosome >= N_CHROMOSOMES:
            raise RandomNativeError("swap chromosome is outside P9016")
        bits[int(chromosome)] = True
    return _apply_swaps(records, k1, k2, bits)


def edge_table_hash(graph: CanonicalGraph) -> str:
    table = np.column_stack((graph.unique_edges.astype("<u8"),
                             graph.unique_counts.astype("<u8")))
    return sha256_array(table, "<u8")


def run_swap_checks(records: RawContacts, grid: FullGrid, graph: CanonicalGraph) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for label, selected in (
        ("single_chr", tuple((i,) for i in range(N_CHROMOSOMES))),
        ("all_chr", (tuple(range(N_CHROMOSOMES)),)),
    ):
        for selection in selected:
            swapped_k1, swapped_k2 = assignment_swap(
                records, graph.assignment_k1, graph.assignment_k2, selection,
            )
            swapped_graph = canonicalize_assignment(records, grid, swapped_k1, swapped_k2)
            graph_equal = (
                np.array_equal(graph.raw_edges, swapped_graph.raw_edges) and
                np.array_equal(graph.unique_edges, swapped_graph.unique_edges) and
                np.array_equal(graph.unique_counts, swapped_graph.unique_counts)
            )
            blob_equal = graph.bridge_blob == swapped_graph.bridge_blob
            if not graph_equal or not blob_equal:
                raise RandomNativeError(
                    f"copy-swap canonical graph/blob changed for {label} {selection}"
                )
            checks.append({
                "scope": label,
                "chromosomes": [int(i) for i in selection],
                "swap_vector": [int(i in selection) for i in range(N_CHROMOSOMES)],
                "canonical_graph_sha256": edge_table_hash(swapped_graph),
                "bridge_blob_sha256": sha256_bytes(swapped_graph.bridge_blob),
                "graph_byteidentity": bool(graph_equal),
                "blob_byteidentity": bool(blob_equal),
                "native_called": False,
            })
    if len(checks) != 21:
        raise RandomNativeError("expected 20 single-chromosome plus one all-chromosome checks")
    return {
        "single_chromosome_count": 20,
        "all_chromosome_count": 1,
        "checks": checks,
        "native_equivariance_asserted": False,
        "interpretation": "graph/blob copy-label invariance only; native seed is not an equivariance proof",
    }


def _fixture_records() -> tuple[RawContacts, FullGrid, np.ndarray, np.ndarray]:
    # Four chromosomes with a partial terminal bin and deliberately unused bins.
    grid = make_fixture_grid((2_500_000, 3_100_000, 1_500_000, 4_200_000))
    ci = np.asarray([
        0, 0, 0, 1, 1, 2, 2, 2, 3, 3,
    ], dtype=np.int32)
    p1 = np.asarray([
        0, 1_000_000, 2_100_000, 0, 1_000_000, 0, 0, 100, 0, 3_000_000,
    ], dtype=np.int64)
    cj = ci.copy()
    p2 = np.asarray([
        1_000_000, 2_000_000, 2_400_000, 1_000_000, 3_000_000, 1_000_000,
        1_000_000, 200, 1_000_000, 4_000_000,
    ], dtype=np.int64)
    cis = ci == cj
    same_bin = cis & ((p1 // BIN_SIZE) == (p2 // BIN_SIZE))
    records = RawContacts(ci, p1, cj, p2, np.arange(len(ci), dtype=np.int64), cis, same_bin)
    k1 = np.asarray([0, 1, 0, 0, 1, 0, 0, 1, 0, 1], dtype=np.int8)
    k2 = k1.copy()
    return records, grid, k1, k2


def make_fixture_grid(lengths: Sequence[int]) -> FullGrid:
    values = np.asarray(lengths, dtype=np.int64)
    n_bins = ((values + BIN_SIZE - 1) // BIN_SIZE).astype(np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(n_bins)))
    positions = np.concatenate(tuple(
        np.arange(int(count), dtype=np.int64) * BIN_SIZE for count in n_bins
    ))
    chromosome_index = np.concatenate(tuple(
        np.full(int(count), chromosome, dtype=np.int32)
        for chromosome, count in enumerate(n_bins)
    ))
    track_names = tuple(
        f"c{chromosome + 1:02d}{copy_name}"
        for chromosome in range(len(values)) for copy_name in ("a", "b")
    )
    track_n_bins = np.repeat(n_bins, 2)
    track_offsets = np.concatenate((np.asarray([0], dtype=np.int64),
                                    np.cumsum(track_n_bins)))
    track_lengths = np.repeat(values, 2)
    beads = np.empty((int(track_offsets[-1]), 3), dtype=np.int32)
    for track, length in enumerate(track_lengths):
        start = int(track_offsets[track])
        count = int(track_n_bins[track])
        starts = np.arange(count, dtype=np.int64) * BIN_SIZE
        beads[start:start + count, 0] = track
        beads[start:start + count, 1] = starts.astype(np.int32)
        beads[start:start + count, 2] = np.minimum(starts + BIN_SIZE, int(length)).astype(np.int32)
    return FullGrid(values, n_bins, offsets, positions, chromosome_index, beads,
                    track_names, track_lengths, track_offsets)


def run_fixture_tests() -> dict[str, Any]:
    records, grid, k1, k2 = _fixture_records()
    if int(grid.beads[2, 2]) != 2_500_000:
        raise RandomNativeError("fixture terminal partial bin check failed")
    observed_loci = np.unique(np.concatenate((
        grid.offsets[records.ci] + records.p1 // BIN_SIZE,
        grid.offsets[records.cj] + records.p2 // BIN_SIZE,
    )))
    if not bool(np.any(~np.isin(np.arange(grid.n_loci), observed_loci))):
        raise RandomNativeError("fixture did not retain all-zero bins")
    graph = canonicalize_assignment(records, grid, k1, k2)
    baseline_blob = graph.bridge_blob
    baseline_edges = graph.unique_edges.copy()
    baseline_counts = graph.unique_counts.copy()
    for mask in range(16):
        selected = tuple(chromosome for chromosome in range(4) if mask & (1 << chromosome))
        bits = np.zeros(len(grid.lengths), dtype=bool)
        for chromosome in selected:
            bits[chromosome] = True
        swapped_k1, swapped_k2 = _apply_swaps(records, k1, k2, bits)
        candidate = canonicalize_assignment(records, grid, swapped_k1, swapped_k2)
        if (candidate.bridge_blob != baseline_blob or
                not np.array_equal(candidate.unique_edges, baseline_edges) or
                not np.array_equal(candidate.unique_counts, baseline_counts)):
            raise RandomNativeError("fixture 2^4 copy-swap enumeration is not invariant")

    tie_records = RawContacts(
        ci=np.asarray([0, 1], dtype=np.int32),
        p1=np.asarray([100, 100], dtype=np.int64),
        cj=np.asarray([0, 1], dtype=np.int32),
        p2=np.asarray([200, 200], dtype=np.int64),
        record_id=np.asarray([0, 1], dtype=np.int64),
        cis=np.asarray([True, True]),
        same_bin=np.asarray([True, True]),
    )
    tie_failed = False
    tie_reason = ""
    try:
        canonicalize_assignment(tie_records, grid, np.asarray([0, 0], dtype=np.int8),
                                np.asarray([0, 0], dtype=np.int8))
    except CanonicalizationError as exc:
        tie_failed = True
        tie_reason = str(exc)
    if not tie_failed:
        raise RandomNativeError("fixture exact tie did not fail fast")
    return {
        "status": "passed",
        "enumerated_swap_vectors": 16,
        "partial_terminal_end_bp": int(grid.beads[2, 2]),
        "all_zero_bins_retained": True,
        "tie_failfast": True,
        "tie_reason": tie_reason,
        "native_called": False,
    }


def run_timed(command: Sequence[str], stdout_path: Path, stderr_path: Path,
              env: dict[str, str] | None = None) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    result = subprocess.run(tuple(command), check=False, capture_output=True, text=True,
                            env=env)
    elapsed = time.perf_counter() - clock
    ended = datetime.now(timezone.utc)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return {
        "command": [str(item) for item in command],
        "started_utc": started.isoformat().replace("+00:00", "Z"),
        "ended_utc": ended.isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": float(elapsed),
        "returncode": int(result.returncode),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def inspect_bridge(bridge: Path, input_path: Path, logs_dir: Path,
                   bundle_id: str) -> dict[str, Any]:
    stdout_path = logs_dir / f"{bundle_id}-inspect.stdout"
    stderr_path = logs_dir / f"{bundle_id}-inspect.stderr"
    receipt = run_timed((str(bridge), "--inspect", "--input", str(input_path)),
                        stdout_path, stderr_path)
    if receipt["returncode"] != 0:
        raise RandomNativeError(f"{bundle_id} native bmap inspect failed: {receipt['stderr']}")
    try:
        stats = json.loads(receipt["stdout"].strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RandomNativeError(f"{bundle_id} inspect did not return JSON") from exc
    if stats.get("status") != "inspected" or stats.get("n_raw_pairs") != FORCE_RECORDS:
        raise RandomNativeError(f"{bundle_id} inspect inventory mismatch: {stats}")
    if stats.get("native_count_sum") != FORCE_RECORDS:
        raise RandomNativeError(f"{bundle_id} native aggregation did not conserve count")
    receipt["stats"] = stats
    write_json(logs_dir / f"{bundle_id}-inspect.receipt.json", receipt)
    return receipt


def source_object_binary_provenance(bridge: Path) -> dict[str, Any]:
    source_path = ROOT / "pr" / "native_bridge" / "fdg_random_bridge.c"
    native = ROOT / "native" / "hickit"
    source_sha = {
        relpath(source_path, ROOT): sha256_file(source_path),
        relpath(native / "fdg.c", ROOT): sha256_file(native / "fdg.c"),
        relpath(native / "bin.c", ROOT): sha256_file(native / "bin.c"),
        relpath(native / "hickit.h", ROOT): sha256_file(native / "hickit.h"),
        relpath(native / "hkpriv.h", ROOT): sha256_file(native / "hkpriv.h"),
        relpath(native / "krng.h", ROOT): sha256_file(native / "krng.h"),
    }
    object_sha = {}
    for name in NATIVE_OBJECTS:
        path = native / name
        if not path.exists():
            raise RandomNativeError(f"vendored native object missing: {path}")
        object_sha[relpath(path, ROOT)] = sha256_file(path)
    if not bridge.exists():
        raise RandomNativeError(f"compiled random bridge missing: {bridge}")
    python_sha = {
        relpath(HERE, ROOT): sha256_file(HERE),
        relpath(ROOT / "pr" / "genome.py", ROOT): sha256_file(ROOT / "pr" / "genome.py"),
        relpath(ROOT / "pr" / "contact_model.py", ROOT): sha256_file(ROOT / "pr" / "contact_model.py"),
    }
    build_command = [
        "cc", "-std=c99", "-O2", "-g", "-Wall", "-Wextra", "-Wc++-compat",
        "-fno-fast-math", "-fno-unsafe-math-optimizations", "-I", "native/hickit",
        "pr/native_bridge/fdg_random_bridge.c",
        *[f"native/hickit/{name}" for name in NATIVE_OBJECTS],
        "-o", relpath(bridge, ROOT), "-lm", "-lz",
    ]
    return {
        "build_command": build_command,
        "build_working_directory": str(ROOT),
        "python_sha256": python_sha,
        "source_sha256": source_sha,
        "object_sha256": object_sha,
        "binary_sha256": {relpath(bridge, ROOT): sha256_file(bridge)},
        "hash_domains_are_separate": True,
    }


def prepare_root(root: Path) -> None:
    if root.exists():
        existing = [item for item in root.iterdir()
                    if item.name != "native" and item.name != ".DS_Store"]
        if existing:
            raise RandomNativeError(f"refusing to reuse non-empty experiment root: {root}")
    else:
        root.mkdir(parents=True)
    for relative in (
        "cohort", "grid", "bundles", "logs", "plots", "run_status", "provenance",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "native" / "build").mkdir(parents=True, exist_ok=True)


def write_readme(root: Path, bridge: Path) -> None:
    text = f"""# Random Native Full-Grid Preflight (025)\n\nThis fresh run prepares three paired random-assignment integer graphs for P9016 at a\ncomplete origin-0 1 Mb grid. It is phase-free and reference-free. The prepared status is\n`prepared_native_pending`; this turn did not call `hk_fdg` on real or synthetic data.\n\n## Commands\n\nBuild the independent bridge (vendored objects are linked, never edited):\n\n```bash\ncd {ROOT}\ncc -O2 -g -Wall -Wextra -Wc++-compat -fno-fast-math -fno-unsafe-math-optimizations \\\n  -I{ROOT / 'native/hickit'} {ROOT / 'pr/native_bridge/fdg_random_bridge.c'} \\\n  {" ".join(str(ROOT / 'native/hickit' / name) for name in NATIVE_OBJECTS)} \\\n  -o {bridge} -lm -lz\n```\n\nPrepare and inspect only:\n\n```bash\nsource /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh\nconda activate analysis\npython {ROOT / 'pr/random_native_init.py'} --prepare-only --run-root {root}\n```\n\nThe released command is intentionally gated twice: pass `--run-released`, and set\n`config.json.status` to `released_for_real_native` only after the parent has frozen the\nprotocol and total budget. It performs at most one native call per bundle and refuses any\nexisting attempt or output:\n\n```bash\nsource /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh\nconda activate analysis\npython {ROOT / 'pr/random_native_init.py'} --run-released --run-root {root}\n```\n\n## Fixed bundle contract\n\n- bundle1: assignment seed `250101`, native seed `250201`\n- bundle2: assignment seed `250102`, native seed `250202`\n- bundle3: assignment seed `250103`, native seed `250203`\n- exact `genome.random_assignment` rule: cis ends share one random copy; inter ends draw independently\n- `p_init=0.75`, `q_from_p(0.75)={Q_INIT:.17g}`; p is not used for assignment\n- native: `hk_fdg(&conf, target, NULL, &rng)`, CPU, one thread, `n_iter=1000`, default params recorded in `config.json`\n- normalization: one global center over all 5290 beads, then one uniform scale to max radius `0.8`\n- same normalized x0 bytes and hash must be written before any future C0/C1/C2-map/C2-free/C3 variant fit\n\nF15 canonicalization is integer-only and occurs before the edge blob: cis copy signatures first,\nthen partner-copy-ignored inter endpoint-incidence signatures; unresolved ties fail the seed.\nThe manifest records 20 single-chromosome plus one all-chromosome graph/blob swap check per bundle.\n`native_equivariance_asserted=false` is deliberate.\n"""
    (root / "README.md").write_text(text, encoding="utf-8")


def prepare(args: argparse.Namespace) -> int:
    root = Path(args.run_root).resolve()
    bridge = Path(args.bridge).resolve() if args.bridge else root / "native" / "build" / "fdg_random_bridge"
    prepare_root(root)
    if not bridge.exists() or not bridge.is_file():
        raise RandomNativeError(f"build the independent bridge before prepare-only: {bridge}")
    provenance = source_object_binary_provenance(bridge)
    input_path = ROOT / "inputs" / "P9016.snpfree.pairs.gz"
    grid = make_full_grid([length for _, length in genome.chrom_lengths(str(input_path))])
    grid_info = grid_audit(grid, root / "grid" / "fullgrid.npz")
    records, cohort_info = load_frozen_contacts(input_path, root / "cohort" / "raw_records.npz")
    fixture_info = run_fixture_tests()
    bundle_manifests = []
    for bundle_id, assignment_seed, native_seed in SEED_BUNDLES:
        k1, k2 = genome.random_assignment(
            {"ci": records.ci, "cj": records.cj, "cis": records.cis},
            seed=assignment_seed,
        )
        graph = canonicalize_assignment(records, grid, k1, k2)
        bundle_root = root / "bundles" / bundle_id
        bundle_root.mkdir(parents=True, exist_ok=False)
        graph_root = bundle_root / "graph"
        graph_root.mkdir(parents=True, exist_ok=False)
        ledger_path = graph_root / "assignment_ledger.npz"
        np.savez_compressed(
            ledger_path,
            record_id=records.record_id,
            cis=records.cis,
            same_bin=records.same_bin,
            assignment_k1=graph.assignment_k1,
            assignment_k2=graph.assignment_k2,
            canonical_k1=graph.canonical_k1,
            canonical_k2=graph.canonical_k2,
        )
        edge_path = graph_root / "canonical_edges.npz"
        np.savez_compressed(
            edge_path,
            raw_record_ids=graph.raw_record_ids,
            raw_edges=graph.raw_edges,
            unique_edges=graph.unique_edges,
            unique_counts=graph.unique_counts,
            swapped_by_chromosome=graph.swapped_by_chromosome.astype(np.int8),
        )
        blob_path = graph_root / "bridge_input.rndbin"
        blob_path.write_bytes(graph.bridge_blob)
        swap_info = run_swap_checks(records, grid, graph)
        inspect = inspect_bridge(bridge, blob_path, root / "logs", bundle_id)
        canonical_audit = {
            "schema": "random-native-canonical-audit-v1",
            "bundle_id": bundle_id,
            "assignment_seed": assignment_seed,
            "native_seed": native_seed,
            "rule": {
                "primary": "per chromosome cis offdiag sparse(local_bin_i,local_bin_j,count), numeric sorted and fixed little-endian encoding",
                "primary_choice": "smaller byte key is canonical copy A",
                "tiebreak": "per chromosome inter endpoint incidence(other_chr,self_local_bin,other_local_bin,count), partner copy ignored; numeric sorted and fixed little-endian encoding",
                "tie": "if both keys equal, seed invalid/preflight failed; no original-slot preservation and no retry",
                "endpoint_transform": "toggle every endpoint k1/k2 for the selected chromosome, then build one canonical edge blob",
            },
            "swap_vector": graph.swapped_by_chromosome.astype(int).tolist(),
            "signaturehash": graph.signature_hash,
            "signature_rows": list(graph.signature_rows),
            "swap_reason": list(graph.swap_reason),
            "swap_invariance": swap_info,
            "native_equivariance_asserted": False,
            "canonical_labels_are_final_x0_labels": True,
            "undo_to_input_slots": False,
        }
        canonical_path = bundle_root / "canonical_audit.json"
        write_json(canonical_path, canonical_audit)
        graph_manifest = {
            "schema": "random-native-graph-manifest-v1",
            "status": "prepared_native_pending",
            "bundle_id": bundle_id,
            "assignment_seed": assignment_seed,
            "native_seed": native_seed,
            "raw_records": records.n_records,
            "same_bin_records_excluded": int(records.same_bin.sum()),
            "cis_offdiag_records": int(np.sum(records.cis & ~records.same_bin)),
            "inter_records": int(np.sum(~records.cis)),
            "force_graph_raw_records": graph.n_raw_edges,
            "force_graph_unique_edges": graph.n_unique_edges,
            "force_graph_integer_count_sum": graph.total_count,
            "zero_count_eligible_pairs_added": False,
            "native_not_called": True,
            "graph_table_sha256": edge_table_hash(graph),
            "bridge_input_sha256": sha256_bytes(graph.bridge_blob),
            "artifacts": {
                "assignment_ledger": {"path": relpath(ledger_path, ROOT),
                                       "sha256": sha256_file(ledger_path)},
                "canonical_edges": {"path": relpath(edge_path, ROOT),
                                     "sha256": sha256_file(edge_path)},
                "bridge_input": {"path": relpath(blob_path, ROOT),
                                  "sha256": sha256_file(blob_path)},
                "canonical_audit": {"path": relpath(canonical_path, ROOT),
                                     "sha256": sha256_file(canonical_path)},
                "inspect_receipt": inspect,
            },
        }
        graph_manifest_path = graph_root / "graph_manifest.json"
        write_json(graph_manifest_path, graph_manifest)
        bundle_manifest = {
            "bundle_id": bundle_id,
            "assignment_seed": assignment_seed,
            "native_seed": native_seed,
            "status": "prepared_native_pending",
            "graph_manifest": relpath(graph_manifest_path, ROOT),
            "canonical_audit": relpath(canonical_path, ROOT),
            "future_native_attempt": {
                "status": "pending_parent_release",
                "attempt_path": relpath(bundle_root / "native_attempt.json", ROOT),
                "raw_native_npz": relpath(bundle_root / "raw_native.npz", ROOT),
                "raw_native_3dg": relpath(bundle_root / "raw_native.3dg", ROOT),
                "native_init_3dg": relpath(bundle_root / "native_init.3dg", ROOT),
                "x0_normalized_npz": relpath(bundle_root / "x0_normalized.npz", ROOT),
                "x0_normalized_3dg": relpath(bundle_root / "x0_normalized.3dg", ROOT),
            },
        }
        bundle_manifests.append(bundle_manifest)
        write_json(bundle_root / "bundle_manifest.json", bundle_manifest)
        write_json(root / "run_status" / f"{bundle_id}.json", {
            "status": "prepared_native_pending",
            "bundle_id": bundle_id,
            "assignment_seed": assignment_seed,
            "native_seed": native_seed,
            "native_attempt_started": False,
            "native_equivariance_asserted": False,
        })
    config = {
        "schema": "random-native-fullgrid-v1",
        "status": "prepared_native_pending",
        "declared_status": "prepared; no native optimizer called",
        "run_root": relpath(root, ROOT),
        "artifact_root": relpath(root, ROOT),
        "cohort": cohort_info,
        "grid": grid_info,
        "random_assignment": {
            "implementation": "pr.genome.random_assignment",
            "cis_rule": "one shared random copy per contact",
            "inter_rule": "independent random copy per endpoint",
            "p_init_participates": False,
            "phase_statistics_imported": False,
        },
        "p_init": {
            "p": P_INIT,
            "q": Q_INIT,
            "q_source": "pr.contact_model.q_from_p(0.75)",
            "future_joint_objective_p_init": P_INIT,
        },
        "bundles": [
            {
                "bundle_id": bundle_id,
                "assignment_seed": assignment_seed,
                "native_seed": native_seed,
                "interpretation": "optimization initialization repeat only; not data or biological replicate",
            }
            for bundle_id, assignment_seed, native_seed in SEED_BUNDLES
        ],
        "native": {
            **NATIVE_DEFAULTS,
            "bridge_mode": "independent from-scratch random native",
            "bridge_input_magic": "RNDINP1\\0",
            "bridge_output_magic": "RNDOUT1\\0",
            "input_has_coordinate_payload": False,
            "source_coordinates_from_014_020_022": False,
            "hk_fdg_copy_x_used": False,
            "native_init_capture": "exported hk_fdg_init called with seed, then RNG reset before hk_fdg; no source_jitter claimed",
            "n_iter_unique_once_per_bundle": True,
            "no_retry": True,
        },
        "normalization": {
            "mode": "one global center then one uniform scale across all 5290 beads",
            "target_max_radius": NORMALIZED_MAX_RADIUS,
            "per_chromosome_transform": False,
            "per_copy_transform": False,
            "mapping_applied_once_after_native": True,
            "raw_native_and_normalized_x0_both_written": True,
        },
        "force_graph": {
            "mode": "integer per-record random assignment then canonical graph aggregation",
            "same_bin_in_force_graph": False,
            "raw_structural_records": FORCE_RECORDS,
            "integer_count_sum_required": FORCE_RECORDS,
            "zero_count_eligible_pairs_added": False,
            "full_raw_ledger_retained": True,
            "likelihood_unit": "aggregated Cij; initialization only; no likelihood evaluated here",
        },
        "f15_canonicalization": {
            "implemented_before_native": True,
            "tie_policy": "seed invalid/preflight failed",
            "canonical_labels_final": True,
            "undo_required": False,
            "all_bundles_graph_only_checks": "20 single-chromosome swaps + 1 all-chromosome swap per bundle",
            "native_equivariance_asserted": False,
        },
        "future_variants": [
            {"id": "C0", "change": "original V1 sphere"},
            {"id": "C1", "change": "bend weight 0"},
            {"id": "C2-map", "change": "identity-kernel sphere reparameterization"},
            {"id": "C2-free", "change": "no hard sphere descriptive sensitivity"},
            {"id": "C3", "change": "uniform exposure"},
        ],
        "future_joint_start_gate": {
            "joint_objective_owner": "parent-side 8738... interface; this file does not modify it",
            "must_write_x0_to_disk_before_variant_fit": True,
            "same_physical_x0_across_variants": True,
            "x0_hash_source": "x0_normalized.npz file bytes per bundle",
            "reference_or_phase_selection": False,
        },
        "native_provenance": provenance,
        "artifacts": {
            "fullgrid": relpath(root / "grid" / "fullgrid.npz", ROOT),
            "raw_records": relpath(root / "cohort" / "raw_records.npz", ROOT),
            "bundles": bundle_manifests,
        },
        "preflight": {
            "fixture": fixture_info,
            "native_inspect_called": True,
            "native_fit_called": False,
            "real_or_synthetic_hk_fdg_called": False,
            "status": "prepared_native_pending",
        },
        "forbidden_inputs_not_read": [
            "reference/phase payloads",
            "data/P9016.1m.3dg.gz",
            "old 014/020/022 coordinate payloads",
        ],
    }
    config_path = root / "config.json"
    write_json(config_path, config)
    manifest = {
        "schema": "random-native-fullgrid-manifest-v1",
        "status": "prepared_native_pending",
        "declared_status": "prepared; await parent protocol/budget release",
        "prepared_utc": utc_now(),
        "config": {"path": relpath(config_path, ROOT), "sha256": sha256_file(config_path)},
        "bridge": provenance,
        "grid": grid_info,
        "cohort": cohort_info,
        "bundles": bundle_manifests,
        "native_equivariance_asserted": False,
        "native_fit_called": False,
        "all_future_joint_starts_hash_gate": True,
    }
    write_json(root / "manifest.json", manifest)
    write_readme(root, bridge)
    write_json(root / "run_status" / "run_status.json", {
        "status": "prepared_native_pending",
        "declared_status": "prepared; no native optimizer called",
        "updated_utc": utc_now(),
        "bundles": {item["bundle_id"]: "prepared_native_pending" for item in bundle_manifests},
        "native_fit_called": False,
        "native_equivariance_asserted": False,
    })
    print(json.dumps({
        "status": "prepared_native_pending",
        "run_root": str(root),
        "config": str(config_path),
        "manifest": str(root / "manifest.json"),
        "bridge": str(bridge),
        "grid": {"n_loci": grid.n_loci, "n_beads": grid.n_beads, "n_tracks": grid.n_tracks},
        "cohort": {"raw_records": records.n_records, "same_bin": int(records.same_bin.sum()),
                   "force_records": FORCE_RECORDS},
        "bundles": [item["bundle_id"] for item in bundle_manifests],
        "native_fit_called": False,
    }, sort_keys=True))
    return 0


def read_native_output(path: Path, grid: FullGrid) -> NativeOutput:
    data = path.read_bytes()
    header = struct.Struct("<8s6I4f")
    if len(data) < header.size:
        raise RandomNativeError("random native output is truncated")
    magic, version, n_beads, n_binned, n_raw, n_iter, flags, native_unit, init_max, final_max, _ = header.unpack_from(data)
    if magic != OUTPUT_MAGIC or version != OUTPUT_VERSION or flags != 0:
        raise RandomNativeError("random native output header is invalid")
    if n_beads != grid.n_beads or n_raw != FORCE_RECORDS or n_iter != int(NATIVE_DEFAULTS["n_iter"]):
        raise RandomNativeError("random native output inventory or budget differs from config")
    n_values = int(n_beads) * 3
    expected = header.size + 2 * n_values * 4
    if len(data) != expected:
        raise RandomNativeError("random native output has trailing/missing bytes")
    values = np.frombuffer(data, dtype="<f4", count=2 * n_values,
                           offset=header.size).copy()
    init = values[:n_values].reshape((n_beads, 3))
    final = values[n_values:].reshape((n_beads, 3))
    if not np.all(np.isfinite(init)) or not np.all(np.isfinite(final)):
        raise RandomNativeError("random native output coordinates are non-finite")
    if not np.isfinite(native_unit) or native_unit <= 0:
        raise RandomNativeError("random native unit is invalid")
    return NativeOutput(int(n_beads), int(n_binned), int(n_raw), int(n_iter),
                        float(native_unit), float(init_max), float(final_max), init, final)


def bead_to_copyfirst(bead_coordinates: np.ndarray, grid: FullGrid) -> np.ndarray:
    values = np.asarray(bead_coordinates, dtype=np.float64)
    if values.shape != (grid.n_beads, 3) or not np.all(np.isfinite(values)):
        raise RandomNativeError("native bead coordinates have invalid shape/finiteness")
    result = np.empty((2, grid.n_loci, 3), dtype=np.float64)
    for chromosome in range(grid.n_chromosomes):
        locus_slice = grid.chromosome_slice(chromosome)
        for copy in (0, 1):
            track = 2 * chromosome + copy
            result[copy, locus_slice] = values[grid.track_slice(track)]
    return result


def write_3dg(path: Path, bead_coordinates: np.ndarray, grid: FullGrid) -> None:
    values = np.asarray(bead_coordinates, dtype=np.float64)
    if values.shape != (grid.n_beads, 3) or not np.all(np.isfinite(values)):
        raise RandomNativeError("cannot write invalid 3dg coordinates")
    with path.open("w", encoding="utf-8") as handle:
        for track, name in enumerate(grid.track_names):
            bead_slice = grid.track_slice(track)
            for bead, position in zip(values[bead_slice], grid.beads[bead_slice, 1]):
                handle.write(f"{name}\t{int(position)}\t{bead[0]:.17g}\t{bead[1]:.17g}\t{bead[2]:.17g}\n")


def normalize_x0(native_final: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    values = np.asarray(native_final, dtype=np.float64)
    center = values.mean(axis=0)
    centered = values - center
    radius = np.linalg.norm(centered, axis=1)
    max_radius = float(radius.max())
    if not np.isfinite(max_radius) or max_radius <= 0:
        raise RandomNativeError("native final coordinates have no positive global radius")
    scale = float(NORMALIZED_MAX_RADIUS / max_radius)
    mapped = centered * scale
    mapped_max = float(np.linalg.norm(mapped, axis=1).max())
    if not np.all(np.isfinite(mapped)) or not np.isclose(mapped_max, NORMALIZED_MAX_RADIUS,
                                                          rtol=0.0, atol=2e-6):
        raise RandomNativeError("global max-radius .8 normalization failed")
    return mapped, center, scale, max_radius


def run_one_bundle(root: Path, bundle: dict[str, Any], grid: FullGrid,
                   bridge: Path, config_sha256: str) -> bool:
    bundle_root = root / "bundles" / bundle["bundle_id"]
    graph_root = bundle_root / "graph"
    input_path = graph_root / "bridge_input.rndbin"
    output_path = bundle_root / "native_output.rndout"
    attempt_path = bundle_root / "native_attempt.json"
    stdout_path = root / "logs" / f"{bundle['bundle_id']}-native.stdout"
    stderr_path = root / "logs" / f"{bundle['bundle_id']}-native.stderr"
    for path in (attempt_path, output_path, bundle_root / "raw_native.npz",
                 bundle_root / "raw_native.3dg", bundle_root / "native_init.3dg",
                 bundle_root / "x0_normalized.npz", bundle_root / "x0_normalized.3dg"):
        if path.exists():
            raise RandomNativeError(f"refusing duplicate native attempt/output: {path}")
    env = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    command = (
        str(bridge), "--input", str(input_path), "--output", str(output_path),
        "--iterations", str(int(NATIVE_DEFAULTS["n_iter"])), "--seed", str(bundle["native_seed"]),
    )
    receipt = run_timed(command, stdout_path, stderr_path, env=env)
    receipt.update({
        "schema": "random-native-attempt-v1",
        "recorded_utc": utc_now(),
        "status": "completed_zero" if receipt["returncode"] == 0 else "failed_nonzero",
        "bundle_id": bundle["bundle_id"],
        "assignment_seed": int(bundle["assignment_seed"]),
        "native_seed": int(bundle["native_seed"]),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "iterations": int(NATIVE_DEFAULTS["n_iter"]),
        "backend": "CPU",
        "threads": 1,
        "source": None,
        "config_sha256_at_attempt": config_sha256,
        "retry": False,
    })
    write_json(attempt_path, receipt)
    if receipt["returncode"] != 0:
        write_json(root / "run_status" / f"{bundle['bundle_id']}.json", {
            "status": "failed_native_nonzero",
            "bundle_id": bundle["bundle_id"],
            "native_attempt_started": True,
            "attempt_path": relpath(attempt_path, ROOT),
            "native_fit_called": True,
            "retry_allowed": False,
        })
        return False
    output = read_native_output(output_path, grid)
    raw_final = output.native_final_bead_order.astype(np.float64)
    raw_init = output.native_init_bead_order.astype(np.float64)
    x0, center, scale, max_radius = normalize_x0(raw_final)
    raw_native_npz = bundle_root / "raw_native.npz"
    np.savez_compressed(
        raw_native_npz,
        native_init_bead_order=raw_init,
        native_final_bead_order=raw_final,
        native_init_coordinates=bead_to_copyfirst(raw_init, grid),
        native_final_coordinates=bead_to_copyfirst(raw_final, grid),
        native_unit=np.asarray([output.native_unit], dtype=np.float64),
        init_max_norm=np.asarray([output.init_max_norm], dtype=np.float64),
        final_max_norm=np.asarray([output.final_max_norm], dtype=np.float64),
        n_binned_pairs=np.asarray([output.n_binned_pairs], dtype=np.int64),
        source=np.asarray(["NULL"], dtype="<U4"),
    )
    raw_native_3dg = bundle_root / "raw_native.3dg"
    native_init_3dg = bundle_root / "native_init.3dg"
    write_3dg(raw_native_3dg, raw_final, grid)
    write_3dg(native_init_3dg, raw_init, grid)
    x0_npz = bundle_root / "x0_normalized.npz"
    x0_beads = np.empty((grid.n_beads, 3), dtype=np.float64)
    for chromosome in range(grid.n_chromosomes):
        slc = grid.chromosome_slice(chromosome)
        for copy in (0, 1):
            x0_beads[grid.track_slice(2 * chromosome + copy)] = x0[copy, slc]
    np.savez_compressed(
        x0_npz,
        coordinates=x0,
        bead_order=x0_beads,
        center=center,
        scale=np.asarray([scale], dtype=np.float64),
        raw_max_radius=np.asarray([max_radius], dtype=np.float64),
        target_max_radius=np.asarray([NORMALIZED_MAX_RADIUS], dtype=np.float64),
        source_native_output_sha256=np.asarray([sha256_file(output_path)]),
    )
    x0_3dg = bundle_root / "x0_normalized.3dg"
    write_3dg(x0_3dg, x0_beads, grid)
    status = {
        "status": "completed_zero",
        "bundle_id": bundle["bundle_id"],
        "native_attempt_started": True,
        "native_fit_called": True,
        "native_output_sha256": sha256_file(output_path),
        "raw_native_npz": {"path": relpath(raw_native_npz, ROOT), "sha256": sha256_file(raw_native_npz)},
        "raw_native_3dg": {"path": relpath(raw_native_3dg, ROOT), "sha256": sha256_file(raw_native_3dg)},
        "native_init_3dg": {"path": relpath(native_init_3dg, ROOT), "sha256": sha256_file(native_init_3dg)},
        "x0_normalized_npz": {"path": relpath(x0_npz, ROOT), "sha256": sha256_file(x0_npz)},
        "x0_normalized_3dg": {"path": relpath(x0_3dg, ROOT), "sha256": sha256_file(x0_3dg)},
        "x0_hash_written_before_future_variants": True,
        "native_equivariance_asserted": False,
        "retry_allowed": False,
    }
    write_json(root / "run_status" / f"{bundle['bundle_id']}.json", status)
    write_json(bundle_root / "x0_manifest.json", {
        "schema": "random-native-x0-manifest-v1",
        "status": "completed_zero",
        "bundle_id": bundle["bundle_id"],
        "assignment_seed": bundle["assignment_seed"],
        "native_seed": bundle["native_seed"],
        "native_output": {"path": relpath(output_path, ROOT), "sha256": sha256_file(output_path)},
        "native_source": None,
        "native_init_capture": "hk_fdg_init exported symbol probe with RNG reset; no source_jitter",
        "normalization": {
            "center": center.tolist(),
            "scale": scale,
            "raw_max_radius": max_radius,
            "target_max_radius": NORMALIZED_MAX_RADIUS,
            "all_beads_single_transform": True,
        },
        "x0_normalized": {
            "npz_path": relpath(x0_npz, ROOT),
            "npz_sha256": sha256_file(x0_npz),
            "3dg_path": relpath(x0_3dg, ROOT),
            "3dg_sha256": sha256_file(x0_3dg),
            "coordinates_shape": list(x0.shape),
            "dtype": "float64",
            "text_precision": "%.17g",
        },
        "future_variant_gate": "write and verify these hashes before any C0/C1/C2-map/C2-free/C3 fit",
    })
    return True


def run_released(args: argparse.Namespace) -> int:
    root = Path(args.run_root).resolve()
    config_path = root / "config.json"
    if not config_path.exists():
        raise RandomNativeError(f"prepared config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "released_for_real_native":
        raise RandomNativeError(
            "native fit is gated: parent must set config.status=released_for_real_native "
            "after protocol/budget freeze"
        )
    if config.get("preflight", {}).get("real_or_synthetic_hk_fdg_called"):
        raise RandomNativeError("config claims a prior hk_fdg call; refusing duplicate release")
    grid_archive = np.load(root / "grid" / "fullgrid.npz", allow_pickle=False)
    grid = make_full_grid(grid_archive["chromosome_lengths"].tolist())
    bridge = Path(args.bridge).resolve() if args.bridge else root / "native" / "build" / "fdg_random_bridge"
    provenance = source_object_binary_provenance(bridge)
    if provenance["binary_sha256"].get(relpath(bridge, ROOT)) != config["native_provenance"]["binary_sha256"].get(relpath(bridge, ROOT)):
        raise RandomNativeError("released bridge binary SHA differs from prepared config")
    config_sha = sha256_file(config_path)
    outcomes = []
    for bundle in config["bundles"]:
        outcomes.append(run_one_bundle(root, bundle, grid, bridge, config_sha))
    all_ok = all(outcomes)
    status = "completed_native_attempts" if all_ok else "native_attempt_failed"
    config["preflight"] = dict(config.get("preflight", {}))
    config["preflight"].update({
        "status": status,
        "native_fit_called": True,
        "real_or_synthetic_hk_fdg_called": True,
        "native_attempt_count": len(outcomes),
        "native_attempts_unique_per_bundle": True,
        "config_sha256_at_attempt": config_sha,
    })
    # Keep the parent release status immutable while persisting completion evidence.
    write_json(config_path, config)
    write_json(root / "run_status" / "run_status.json", {
        "status": status,
        "updated_utc": utc_now(),
        "bundles": {bundle["bundle_id"]: ("completed_zero" if ok else "failed")
                     for bundle, ok in zip(config["bundles"], outcomes)},
        "native_fit_called": True,
        "native_attempt_count": len(outcomes),
        "retry_allowed": False,
        "config_sha256_at_attempt": config_sha,
    })
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": status,
        "completed_utc": utc_now(),
        "config_sha256_at_attempt": config_sha,
        "native_fit_called": True,
        "native_equivariance_asserted": False,
    })
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": status,
        "run_root": str(root),
        "bundles": [bundle["bundle_id"] for bundle in config["bundles"]],
        "native_attempt_count": len(outcomes),
        "retry_allowed": False,
    }, sort_keys=True))
    return 0 if all_ok else 1


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run-released", action="store_true")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--bridge", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.prepare_only:
            return prepare(args)
        return run_released(args)
    except (RandomNativeError, OSError, ValueError) as exc:
        print(f"random_native_init: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
