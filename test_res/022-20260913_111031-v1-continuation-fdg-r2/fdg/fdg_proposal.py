"""native FDG proposal 的确定性、无 phase preparation。

本模块仅负责 022 FDG proposal 边界。它接收已通过项目 SNP-free gate 加载的数组；绝不打开 reference 或带 phase 的 payload。Native execution 保留在 fdg_bridge 中。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import struct
import subprocess
import time
from typing import Iterable, Sequence

import numpy as np


BIN_SIZE = 1_000_000
N_COPIES = 2
N_STATES = 4
EPSILON = 1e-6
P_FLOOR = 1e-4
INTERIOR_TARGET = 1.0 - 1e-6
ALLOC_SEED = 220701
NATIVE_SEED = 220702
ALPHA_SEQUENCE = tuple(2.0 ** (-i) for i in range(11))
STATE_COPIES = np.asarray(((0, 0), (0, 1), (1, 0), (1, 1)), dtype=np.int8)
INPUT_MAGIC = b"FDGBIN1\0"
OUTPUT_MAGIC = b"FDGOUT1\0"
INPUT_VERSION = 1
OUTPUT_VERSION = 1


class FdgProtocolError(ValueError):
    """对象违反冻结 FDG proposal contract 时抛出。"""


@dataclass(frozen=True)
class FullGrid:
    """显式的 0-based full grid 与 native bead inventory。"""

    chromosome_lengths: np.ndarray
    bin_size: int
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
        return int(len(self.chromosome_lengths))

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
class RawRecords:
    """按数值 genome order 排列的原始、无标签 contact records。"""

    ci: np.ndarray
    p1: np.ndarray
    cj: np.ndarray
    p2: np.ndarray
    record_id: np.ndarray

    @property
    def n_records(self) -> int:
        return int(len(self.ci))

    def sorted_numeric(self) -> "RawRecords":
        order = np.lexsort((self.p2, self.p1, self.cj, self.ci))
        return RawRecords(
            ci=self.ci[order].copy(),
            p1=self.p1[order].copy(),
            cj=self.cj[order].copy(),
            p2=self.p2[order].copy(),
            record_id=self.record_id[order].copy(),
        )


@dataclass(frozen=True)
class CanonicalCoordinates:
    """按每条染色体的确定性 copy order 排列的坐标。"""

    coordinates: np.ndarray
    swapped_by_chromosome: np.ndarray
    keys: tuple[tuple[bytes, bytes], ...]


@dataclass(frozen=True)
class PosteriorResult:
    """四状态 posterior 以及 raw-to-full-grid endpoint mapping。"""

    g1: np.ndarray
    g2: np.ndarray
    same_bin: np.ndarray
    probabilities: np.ndarray
    p: float
    r0: float


@dataclass(frozen=True)
class AllocationResult:
    """每条非对角原始记录一个 integer state；对角记录为 -1。"""

    state: np.ndarray
    probabilities: np.ndarray
    seed: int

    @property
    def n_records(self) -> int:
        return int(len(self.state))


@dataclass(frozen=True)
class ForceGraph:
    """原始 assigned force records 以及确定性的聚合 edge table。"""

    raw_edges: np.ndarray
    raw_record_ids: np.ndarray
    included_mask: np.ndarray
    unique_edges: np.ndarray
    unique_counts: np.ndarray

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
class BridgeOutput:
    """fdg_bridge 返回的高精度二进制输出。"""

    n_beads: int
    n_binned_pairs: int
    n_raw_pairs: int
    n_iter: int
    source_avg_bb: float
    native_unit: float
    max_abs_jitter: float
    native_init_bead_order: np.ndarray
    native_final_bead_order: np.ndarray


@dataclass(frozen=True)
class BridgeRun:
    """可审计的一次性 bridge invocation；失败会返回，不重试。"""

    command: tuple[str, ...]
    started_utc: str
    ended_utc: str
    elapsed_seconds: float
    returncode: int
    input_sha256: str
    output_sha256: str | None
    stdout: str
    stderr: str
def _as_int_array(name: str, values: Iterable[int], dtype=np.int64) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise FdgProtocolError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.array_equal(array, array.astype(np.int64)):
            raise FdgProtocolError(f"{name} must contain exact integers")
    return array.astype(dtype, copy=True)


def make_full_grid(chromosome_lengths: Sequence[int], bin_size: int = BIN_SIZE) -> FullGrid:
    """构建完整的 0-based grid，包括末端 partial bins。"""
    lengths = _as_int_array("chromosome_lengths", chromosome_lengths, np.int64)
    if len(lengths) == 0 or np.any(lengths <= 0):
        raise FdgProtocolError("chromosome lengths must be positive")
    if bin_size <= 0:
        raise FdgProtocolError("bin_size must be positive")
    n_bins = ((lengths + int(bin_size) - 1) // int(bin_size)).astype(np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(n_bins)))
    positions = np.concatenate(tuple(
        np.arange(int(count), dtype=np.int64) * int(bin_size)
        for count in n_bins
    ))
    chromosome_index = np.concatenate(tuple(
        np.full(int(count), chromosome, dtype=np.int32)
        for chromosome, count in enumerate(n_bins)
    ))
    track_names = tuple(
        f"c{chromosome + 1:02d}{copy_name}"
        for chromosome in range(len(lengths))
        for copy_name in ("a", "b")
    )
    track_n_bins = np.repeat(n_bins, 2)
    track_offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(track_n_bins)))
    beads = np.empty((int(track_offsets[-1]), 3), dtype=np.int32)
    for track, length in enumerate(np.repeat(lengths, 2)):
        start = int(track_offsets[track])
        count = int(track_n_bins[track])
        starts = np.arange(count, dtype=np.int64) * int(bin_size)
        ends = np.minimum(starts + int(bin_size), int(length))
        beads[start:start + count, 0] = track
        beads[start:start + count, 1] = starts.astype(np.int32)
        beads[start:start + count, 2] = ends.astype(np.int32)
    grid = FullGrid(
        chromosome_lengths=lengths,
        bin_size=int(bin_size),
        n_bins=n_bins,
        offsets=offsets,
        positions=positions,
        chromosome_index=chromosome_index,
        beads=beads,
        track_names=track_names,
        track_lengths=np.repeat(lengths, 2),
        track_offsets=track_offsets,
    )
    validate_full_grid(grid)
    return grid


def validate_full_grid(grid: FullGrid) -> None:
    """检查 bridge 需要的精确显式 bead partition。"""
    if grid.bin_size != BIN_SIZE:
        raise FdgProtocolError("the FDG proposal is frozen at 1 Mb")
    if grid.n_tracks != 2 * grid.n_chromosomes:
        raise FdgProtocolError("track count is not two copies per chromosome")
    if grid.n_beads != 2 * grid.n_loci:
        raise FdgProtocolError("physical bead count is not two per locus")
    for track, length in enumerate(grid.track_lengths):
        bead_slice = grid.track_slice(track)
        beads = grid.beads[bead_slice]
        expected_count = (int(length) + grid.bin_size - 1) // grid.bin_size
        expected_starts = np.arange(expected_count, dtype=np.int64) * grid.bin_size
        expected_ends = np.minimum(expected_starts + grid.bin_size, int(length))
        if len(beads) != expected_count:
            raise FdgProtocolError("track bead count does not equal ceil(length/bin)")
        if not np.all(beads[:, 0] == track):
            raise FdgProtocolError("beads are not grouped in explicit track order")
        if not np.array_equal(beads[:, 1], expected_starts.astype(np.int32)):
            raise FdgProtocolError("track starts do not form a complete 0-based grid")
        if not np.array_equal(beads[:, 2], expected_ends.astype(np.int32)):
            raise FdgProtocolError("track ends do not retain terminal partial bin")
    if int(grid.n_loci) != 2645 and grid.n_chromosomes == 20:
        raise FdgProtocolError("P9016 full grid must contain 2645 loci")
    if int(grid.n_beads) != 5290 and grid.n_chromosomes == 20:
        raise FdgProtocolError("P9016 full grid must contain 5290 physical beads")


def make_raw_records(ci: Iterable[int], p1: Iterable[int], cj: Iterable[int],
                     p2: Iterable[int], n_chromosomes: int,
                     record_id: Iterable[int] | None = None) -> RawRecords:
    """归一化 endpoint，并在数值排序前分配稳定 id。"""
    ci_array = _as_int_array("ci", ci, np.int32)
    p1_array = _as_int_array("p1", p1, np.int64)
    cj_array = _as_int_array("cj", cj, np.int32)
    p2_array = _as_int_array("p2", p2, np.int64)
    n = len(ci_array)
    if any(len(array) != n for array in (p1_array, cj_array, p2_array)):
        raise FdgProtocolError("raw endpoint arrays have inconsistent lengths")
    if record_id is None:
        ids = np.arange(n, dtype=np.int64)
    else:
        ids = _as_int_array("record_id", record_id, np.int64)
        if len(ids) != n or len(np.unique(ids)) != n:
            raise FdgProtocolError("record_id must be unique and aligned")
    if np.any(ci_array < 0) or np.any(ci_array >= n_chromosomes):
        raise FdgProtocolError("left chromosome outside header order")
    if np.any(cj_array < 0) or np.any(cj_array >= n_chromosomes):
        raise FdgProtocolError("right chromosome outside header order")
    reverse_chrom = ci_array > cj_array
    if np.any(reverse_chrom):
        old_ci = ci_array.copy()
        old_p1 = p1_array.copy()
        ci_array[reverse_chrom] = cj_array[reverse_chrom]
        p1_array[reverse_chrom] = p2_array[reverse_chrom]
        cj_array[reverse_chrom] = old_ci[reverse_chrom]
        p2_array[reverse_chrom] = old_p1[reverse_chrom]
    reverse_cis = (ci_array == cj_array) & (p1_array > p2_array)
    if np.any(reverse_cis):
        old_p1 = p1_array.copy()
        p1_array[reverse_cis] = p2_array[reverse_cis]
        p2_array[reverse_cis] = old_p1[reverse_cis]
    return RawRecords(ci_array, p1_array, cj_array, p2_array, ids).sorted_numeric()


def validate_raw_records(records: RawRecords, grid: FullGrid) -> None:
    """验证所有 raw records，不过滤任何行。"""
    if records.n_records == 0:
        raise FdgProtocolError("raw record set must not be empty")
    if np.any(records.ci < 0) or np.any(records.ci >= grid.n_chromosomes):
        raise FdgProtocolError("raw left chromosome outside grid")
    if np.any(records.cj < 0) or np.any(records.cj >= grid.n_chromosomes):
        raise FdgProtocolError("raw right chromosome outside grid")
    for chrom, length in enumerate(grid.chromosome_lengths):
        left = records.ci == chrom
        right = records.cj == chrom
        if np.any(records.p1[left] < 0) or np.any(records.p1[left] >= int(length)):
            raise FdgProtocolError("left endpoint outside chromosome interval")
        if np.any(records.p2[right] < 0) or np.any(records.p2[right] >= int(length)):
            raise FdgProtocolError("right endpoint outside chromosome interval")
    if np.any((records.ci == records.cj) & (records.p1 > records.p2)):
        raise FdgProtocolError("cis records are not numerically canonical")
    if np.any(records.ci > records.cj):
        raise FdgProtocolError("inter records are not in numeric chromosome order")


def _little_endian_track_key(track: np.ndarray) -> bytes:
    values = np.asarray(track, dtype="<f8", order="C")
    if values.ndim != 2 or values.shape[1] != 3:
        raise FdgProtocolError("a track must have shape (n_loci_on_chr, 3)")
    if not np.all(np.isfinite(values)):
        raise FdgProtocolError("canonicalization requires finite coordinates")
    return values.tobytes(order="C")


def canonicalize_coordinates(coordinates: np.ndarray, grid: FullGrid) -> CanonicalCoordinates:
    """为每条染色体独立确定性地选择 copy order。"""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (2, grid.n_loci, 3) or not np.all(np.isfinite(values)):
        raise FdgProtocolError("coordinates must be finite with shape (2,n_loci,3)")
    canonical = values.copy()
    swapped = np.zeros(grid.n_chromosomes, dtype=bool)
    keys: list[tuple[bytes, bytes]] = []
    for chromosome in range(grid.n_chromosomes):
        slc = grid.chromosome_slice(chromosome)
        key0 = _little_endian_track_key(values[0, slc])
        key1 = _little_endian_track_key(values[1, slc])
        keys.append((key0, key1))
        if key1 < key0:
            canonical[:, slc] = values[::-1, slc]
            swapped[chromosome] = True
    return CanonicalCoordinates(canonical, swapped, tuple(keys))


def undo_canonicalization(canonical: np.ndarray, grid: FullGrid,
                          swapped_by_chromosome: Iterable[bool]) -> np.ndarray:
    """将规范坐标返回原始 V1 copy slots。"""
    values = np.asarray(canonical, dtype=np.float64)
    swapped = np.asarray(tuple(swapped_by_chromosome), dtype=bool)
    if values.shape != (2, grid.n_loci, 3) or len(swapped) != grid.n_chromosomes:
        raise FdgProtocolError("canonical inverse inputs have incompatible shapes")
    result = values.copy()
    for chromosome, is_swapped in enumerate(swapped):
        if is_swapped:
            slc = grid.chromosome_slice(chromosome)
            result[:, slc] = result[::-1, slc]
    return result


def swap_copy_subset(coordinates: np.ndarray, grid: FullGrid,
                     chromosomes: Sequence[int] | None = None) -> np.ndarray:
    """创建纯坐标的 copy-label swap fixture。"""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (2, grid.n_loci, 3):
        raise FdgProtocolError("coordinates have incompatible shape")
    selected = range(grid.n_chromosomes) if chromosomes is None else chromosomes
    result = values.copy()
    for chromosome in selected:
        if chromosome < 0 or chromosome >= grid.n_chromosomes:
            raise FdgProtocolError("swap chromosome outside grid")
        slc = grid.chromosome_slice(int(chromosome))
        result[:, slc] = result[::-1, slc]
    return result


def p_from_q(q: float) -> float:
    """应用精确的 bounded-logistic V1 parameterization。"""
    if not np.isfinite(q):
        raise FdgProtocolError("q must be finite")
    if q >= 0.0:
        z = float(np.exp(-q))
        logistic = 1.0 / (1.0 + z)
    else:
        z = float(np.exp(q))
        logistic = z / (1.0 + z)
    return float(P_FLOOR + (1.0 - 2.0 * P_FLOOR) * logistic)


def _bounded_kernel_squared(distance_squared: np.ndarray, r0: float) -> np.ndarray:
    base = 1.0 + distance_squared / (r0 * r0)
    return EPSILON + (1.0 - EPSILON) * base ** -2


def posterior_from_coordinates(coordinates: np.ndarray, p: float,
                               records: RawRecords, grid: FullGrid,
                               block_size: int = 65_536) -> PosteriorResult:
    """计算每条 raw row 的四状态 V1 finite-kernel posterior。"""
    validate_raw_records(records, grid)
    if not P_FLOOR <= float(p) <= 1.0 - P_FLOOR:
        raise FdgProtocolError("p lies outside the bounded V1 interval")
    if block_size <= 0:
        raise FdgProtocolError("block_size must be positive")
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (2, grid.n_loci, 3) or not np.all(np.isfinite(values)):
        raise FdgProtocolError("coordinates must be finite with shape (2,n_loci,3)")
    b1 = records.p1 // grid.bin_size
    b2 = records.p2 // grid.bin_size
    g1 = grid.offsets[records.ci] + b1
    g2 = grid.offsets[records.cj] + b2
    same_bin = g1 == g2
    if np.any(same_bin & (records.ci != records.cj)):
        raise FdgProtocolError("same-bin record cannot be inter-chromosomal")
    probabilities = np.zeros((records.n_records, N_STATES), dtype=np.float64)
    r0 = 2.0 * (2.0 * grid.n_loci) ** (-1.0 / 3.0)
    for start in range(0, records.n_records, block_size):
        stop = min(start + block_size, records.n_records)
        i = g1[start:stop].astype(np.int64, copy=False)
        j = g2[start:stop].astype(np.int64, copy=False)
        aa = values[0, i] - values[0, j]
        ab = values[0, i] - values[1, j]
        ba = values[1, i] - values[0, j]
        bb = values[1, i] - values[1, j]
        kaa = _bounded_kernel_squared(np.sum(aa * aa, axis=1), r0)
        kab = _bounded_kernel_squared(np.sum(ab * ab, axis=1), r0)
        kba = _bounded_kernel_squared(np.sum(ba * ba, axis=1), r0)
        kbb = _bounded_kernel_squared(np.sum(bb * bb, axis=1), r0)
        cis = records.ci[start:stop] == records.cj[start:stop]
        block = probabilities[start:stop]
        block[:, 0] = np.where(cis, 0.5 * p, 0.25) * kaa
        block[:, 1] = np.where(cis, 0.5 * (1.0 - p), 0.25) * kab
        block[:, 2] = np.where(cis, 0.5 * (1.0 - p), 0.25) * kba
        block[:, 3] = np.where(cis, 0.5 * p, 0.25) * kbb
    probabilities[same_bin] = 0.0
    non_diag = ~same_bin
    sums = probabilities[non_diag].sum(axis=1)
    if np.any(~np.isfinite(sums)) or np.any(sums <= 0.0):
        raise FdgProtocolError("posterior normalization is non-finite or zero")
    probabilities[non_diag] /= sums[:, None]
    if np.any(~np.isfinite(probabilities[non_diag])):
        raise FdgProtocolError("posterior contains non-finite values")
    return PosteriorResult(g1.astype(np.int64), g2.astype(np.int64), same_bin,
                           probabilities, float(p), float(r0))


def integer_allocate(posterior: PosteriorResult, seed: int = ALLOC_SEED) -> AllocationResult:
    """为每条非对角记录抽取一个 state，保留所有 raw rows。"""
    probabilities = np.asarray(posterior.probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != N_STATES:
        raise FdgProtocolError("posterior must have four state columns")
    state = np.full(len(probabilities), -1, dtype=np.int8)
    non_diag = ~np.asarray(posterior.same_bin, dtype=bool)
    if len(non_diag) != len(probabilities):
        raise FdgProtocolError("posterior diagonal mask is misaligned")
    rng = np.random.default_rng(int(seed))
    rows = probabilities[non_diag]
    if len(rows):
        sums = rows.sum(axis=1)
        if np.any(~np.isfinite(sums)) or np.any(sums <= 0.0):
            raise FdgProtocolError("integer allocation received invalid posterior sums")
        normalized = rows / sums[:, None]
        cumulative = np.cumsum(normalized, axis=1)
        draws = rng.random(len(rows))
        assigned = np.sum(draws[:, None] > cumulative, axis=1).astype(np.int8)
        assigned = np.minimum(assigned, N_STATES - 1).astype(np.int8)
        state[non_diag] = assigned
    return AllocationResult(state, probabilities.copy(), int(seed))


def state_count_table(posterior: PosteriorResult,
                      allocation: AllocationResult) -> np.ndarray:
    """返回 `(g1,g2,total,state00,state01,state10,state11)` audit rows。"""
    probabilities = np.asarray(allocation.probabilities, dtype=np.float64)
    if len(probabilities) != len(posterior.same_bin) or \
            len(allocation.state) != len(probabilities):
        raise FdgProtocolError("state audit arrays are not aligned")
    non_diag = ~np.asarray(posterior.same_bin, dtype=bool)
    states = np.asarray(allocation.state[non_diag], dtype=np.int8)
    if np.any(states < 0) or np.any(states >= N_STATES):
        raise FdgProtocolError("non-diagonal state audit contains an unassigned row")
    pairs = np.column_stack((posterior.g1[non_diag], posterior.g2[non_diag]))
    if len(pairs) == 0:
        return np.empty((0, 7), dtype=np.int64)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    counts = np.zeros((len(unique), 1 + N_STATES), dtype=np.int64)
    np.add.at(counts[:, 0], inverse, 1)
    for state in range(N_STATES):
        np.add.at(counts[:, 1 + state], inverse, states == state)
    if not np.array_equal(counts[:, 0], counts[:, 1:].sum(axis=1)):
        raise FdgProtocolError("state counts do not conserve each raw bin pair")
    return np.column_stack((unique.astype(np.int64), counts))


def force_graph_from_allocation(records: RawRecords, posterior: PosteriorResult,
                                allocation: AllocationResult, grid: FullGrid) -> ForceGraph:
    """将 assigned 的非对角记录映射为规范、无序的 bead edges。"""
    if records.n_records != len(allocation.state) or records.n_records != len(posterior.g1):
        raise FdgProtocolError("allocation and posterior are not aligned to raw records")
    included = ~posterior.same_bin
    if np.any(allocation.state[included] < 0) or np.any(allocation.state[~included] >= 0):
        raise FdgProtocolError("diag/non-diag allocation mask is inconsistent")
    edge_rows: list[tuple[int, int]] = []
    edge_ids: list[int] = []
    for row in np.flatnonzero(included):
        state = int(allocation.state[row])
        copy0, copy1 = (int(v) for v in STATE_COPIES[state])
        track0 = 2 * int(records.ci[row]) + copy0
        track1 = 2 * int(records.cj[row]) + copy1
        local0 = int(records.p1[row] // grid.bin_size)
        local1 = int(records.p2[row] // grid.bin_size)
        bid0 = int(grid.track_offsets[track0]) + local0
        bid1 = int(grid.track_offsets[track1]) + local1
        if bid0 == bid1:
            raise FdgProtocolError("non-diagonal allocation produced a self edge")
        if bid1 < bid0:
            bid0, bid1 = bid1, bid0
        edge_rows.append((bid0, bid1))
        edge_ids.append(int(records.record_id[row]))
    raw_edges = np.asarray(edge_rows, dtype=np.uint32).reshape((-1, 2))
    raw_record_ids = np.asarray(edge_ids, dtype=np.int64)
    if len(raw_edges):
        order = np.lexsort((raw_edges[:, 1], raw_edges[:, 0], raw_record_ids))
        raw_edges = raw_edges[order]
        raw_record_ids = raw_record_ids[order]
        unique_edges, unique_counts = np.unique(raw_edges, axis=0, return_counts=True)
        unique_counts = unique_counts.astype(np.int64, copy=False)
    else:
        unique_edges = np.empty((0, 2), dtype=np.uint32)
        unique_counts = np.empty(0, dtype=np.int64)
    graph = ForceGraph(raw_edges, raw_record_ids, included.copy(),
                       unique_edges.astype(np.uint32), unique_counts)
    if graph.total_count != int(included.sum()):
        raise FdgProtocolError("force graph integer count is not record-conserving")
    if np.any(graph.unique_edges[:, 0] >= graph.unique_edges[:, 1]):
        raise FdgProtocolError("force graph contains unordered or self edges")
    return graph


def coordinates_to_bead_order(coordinates: np.ndarray, grid: FullGrid) -> np.ndarray:
    """将 copy-first V1 坐标转换为 native track/bead order。"""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (2, grid.n_loci, 3) or not np.all(np.isfinite(values)):
        raise FdgProtocolError("coordinates must be finite with V1 shape")
    result = np.empty((grid.n_beads, 3), dtype=np.float32)
    for chromosome in range(grid.n_chromosomes):
        slc = grid.chromosome_slice(chromosome)
        for copy in range(2):
            track = 2 * chromosome + copy
            track_slc = grid.track_slice(track)
            result[track_slc] = values[copy, slc].astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise FdgProtocolError("float32 native source conversion is non-finite")
    return result


def bead_order_to_coordinates(bead_coordinates: np.ndarray, grid: FullGrid) -> np.ndarray:
    """将 native track/bead order 转回 V1 copy-first order。"""
    raw_values = np.asarray(bead_coordinates)
    if raw_values.dtype.kind != "f":
        raise FdgProtocolError("native coordinates must use a floating dtype")
    values = raw_values.astype(np.float64, copy=False)
    if values.shape != (grid.n_beads, 3) or not np.all(np.isfinite(values)):
        raise FdgProtocolError("native coordinates have incompatible shape or finiteness")
    result = np.empty((2, grid.n_loci, 3), dtype=np.float64)
    for chromosome in range(grid.n_chromosomes):
        slc = grid.chromosome_slice(chromosome)
        for copy in range(2):
            track = 2 * chromosome + copy
            track_slc = grid.track_slice(track)
            result[copy, slc] = values[track_slc].astype(np.float64)
    return result


def pack_bridge_input(grid: FullGrid, graph: ForceGraph,
                      canonical_coordinates: np.ndarray) -> bytes:
    """序列化显式 beads、原始 integer edges 和 float32 source。"""
    validate_full_grid(grid)
    if graph.n_raw_edges == 0:
        raise FdgProtocolError("native force graph must contain an off-diagonal edge")
    bead_source = coordinates_to_bead_order(canonical_coordinates, grid)
    edges = np.asarray(graph.raw_edges, dtype=np.uint32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise FdgProtocolError("force edges must have shape (n,2)")
    if np.any(edges[:, 0] >= edges[:, 1]) or np.any(edges[:, 1] >= grid.n_beads):
        raise FdgProtocolError("bridge input contains invalid canonical bead ids")
    header = struct.pack(
        "<8s6I", INPUT_MAGIC, INPUT_VERSION, grid.bin_size, grid.n_tracks,
        grid.n_beads, graph.n_raw_edges, 0,
    )
    lengths = np.asarray(grid.track_lengths, dtype="<i4").tobytes(order="C")
    beads = np.asarray(grid.beads, dtype="<i4").tobytes(order="C")
    edge_bytes = np.asarray(edges, dtype="<u4").tobytes(order="C")
    source_bytes = bead_source.astype("<f4", copy=False).tobytes(order="C")
    return b"".join((header, lengths, beads, edge_bytes, source_bytes))


def read_bridge_output(path: str | Path, grid: FullGrid) -> BridgeOutput:
    """读取 bridge 的固定 binary header 和两个 float32 arrays。"""
    data = Path(path).read_bytes()
    header_struct = struct.Struct("<8s6I4f")
    if len(data) < header_struct.size:
        raise FdgProtocolError("bridge output is truncated")
    unpacked = header_struct.unpack_from(data, 0)
    magic, version, n_beads, n_binned, n_raw, n_iter, flags = unpacked[:7]
    source_avg_bb, native_unit, max_abs_jitter, _ = unpacked[7:]
    if magic != OUTPUT_MAGIC or version != OUTPUT_VERSION or flags != 0:
        raise FdgProtocolError("bridge output header is invalid")
    if n_beads != grid.n_beads or n_raw == 0 or n_iter == 0:
        raise FdgProtocolError("bridge output inventory or budget is invalid")
    n_values = int(n_beads) * 3
    expected_size = header_struct.size + 2 * n_values * 4
    if len(data) != expected_size:
        raise FdgProtocolError("bridge output has trailing or missing bytes")
    values = np.frombuffer(data, dtype="<f4", count=2 * n_values,
                           offset=header_struct.size).copy()
    init = values[:n_values].reshape((n_beads, 3))
    final = values[n_values:].reshape((n_beads, 3))
    if not np.all(np.isfinite(init)) or not np.all(np.isfinite(final)):
        raise FdgProtocolError("bridge output coordinates are non-finite")
    if not np.isfinite(source_avg_bb) or source_avg_bb <= 0.0:
        raise FdgProtocolError("bridge source average backbone is invalid")
    if not np.isfinite(native_unit) or native_unit <= 0.0:
        raise FdgProtocolError("bridge native unit is invalid")
    return BridgeOutput(
        n_beads=int(n_beads), n_binned_pairs=int(n_binned), n_raw_pairs=int(n_raw),
        n_iter=int(n_iter), source_avg_bb=float(source_avg_bb),
        native_unit=float(native_unit), max_abs_jitter=float(max_abs_jitter),
        native_init_bead_order=init, native_final_bead_order=final,
    )


def map_native_to_unit_ball(native_coordinates: np.ndarray,
                             grid: FullGrid) -> tuple[np.ndarray, np.ndarray, float, float]:
    """将一个全局 V1 interior map 应用于 raw native coordinates。"""
    values = np.asarray(native_coordinates, dtype=np.float64)
    if values.shape != (2, grid.n_loci, 3) or not np.all(np.isfinite(values)):
        raise FdgProtocolError("native proposal has invalid V1 shape or finiteness")
    flat = values.reshape((-1, 3))
    center = flat.mean(axis=0)
    centered = values - center
    max_radius = float(np.linalg.norm(centered.reshape((-1, 3)), axis=1).max())
    if not np.isfinite(max_radius):
        raise FdgProtocolError("native proposal radius is non-finite")
    scale = max_radius / INTERIOR_TARGET if max_radius >= INTERIOR_TARGET else 1.0
    mapped = centered / scale
    if not np.all(np.isfinite(mapped)):
        raise FdgProtocolError("global V1 map is non-finite")
    norms = np.linalg.norm(mapped.reshape((-1, 3)), axis=1)
    if np.any(norms >= 1.0):
        raise FdgProtocolError("global V1 map did not produce strict interior coordinates")
    return mapped, center, float(scale), max_radius


def sphere_inverse(coordinates: np.ndarray) -> np.ndarray:
    """将合法的 V1 坐标转换为无约束 optimizer variables。"""
    values = np.asarray(coordinates, dtype=np.float64)
    norms2 = np.sum(values * values, axis=-1, keepdims=True)
    if values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise FdgProtocolError("sphere inverse requires finite (...,3) coordinates")
    if np.any(norms2 >= 1.0):
        raise FdgProtocolError("sphere inverse requires strict unit-ball interior")
    return values / np.sqrt(1.0 - norms2)


def backtracking_coordinates(anchor: np.ndarray, full_proposal: np.ndarray,
                              grid: FullGrid) -> tuple[tuple[float, np.ndarray], ...]:
    """返回冻结的 alpha-ordered physical trial coordinates。"""
    anchor_values = np.asarray(anchor, dtype=np.float64)
    proposal_values = np.asarray(full_proposal, dtype=np.float64)
    for name, values in (("anchor", anchor_values), ("full_proposal", proposal_values)):
        if values.shape != (2, grid.n_loci, 3) or not np.all(np.isfinite(values)):
            raise FdgProtocolError(f"{name} has invalid V1 coordinate shape")
    return tuple(
        (float(alpha), anchor_values + float(alpha) * (proposal_values - anchor_values))
        for alpha in ALPHA_SEQUENCE
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_native_bridge(bridge_path: str | Path, input_path: str | Path,
                      output_path: str | Path, input_blob: bytes,
                      iterations: int = 1000, seed: int = NATIVE_SEED) -> BridgeRun:
    """运行一次固定的 bridge call，并保留全部失败证据。"""
    bridge = Path(bridge_path)
    input_file = Path(input_path)
    output_file = Path(output_path)
    if iterations <= 0:
        raise FdgProtocolError("native iterations must be positive")
    if output_file.exists():
        raise FdgProtocolError(f"refusing to overwrite native output: {output_file}")
    input_file.write_bytes(input_blob)
    command = (
        str(bridge), "--input", str(input_file), "--output", str(output_file),
        "--iterations", str(int(iterations)), "--seed", str(int(seed)),
    )
    started = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = time.perf_counter() - started_clock
    ended = datetime.now(timezone.utc)
    output_sha = sha256_file(output_file) if output_file.exists() else None
    return BridgeRun(
        command=command,
        started_utc=started.isoformat().replace("+00:00", "Z"),
        ended_utc=ended.isoformat().replace("+00:00", "Z"),
        elapsed_seconds=float(elapsed),
        returncode=int(result.returncode),
        input_sha256=sha256_bytes(input_blob),
        output_sha256=output_sha,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def build_integer_proposal(coordinates: np.ndarray, q: float,
                           records: RawRecords, grid: FullGrid,
                           allocation_seed: int = ALLOC_SEED) -> tuple[
                               CanonicalCoordinates, PosteriorResult,
                               AllocationResult, ForceGraph, bytes]:
    """在内存中运行完整的确定性 preparation pipeline。"""
    validate_full_grid(grid)
    validate_raw_records(records, grid)
    canonical = canonicalize_coordinates(coordinates, grid)
    posterior = posterior_from_coordinates(canonical.coordinates, p_from_q(q), records, grid)
    allocation = integer_allocate(posterior, seed=allocation_seed)
    graph = force_graph_from_allocation(records, posterior, allocation, grid)
    blob = pack_bridge_input(grid, graph, canonical.coordinates)
    return canonical, posterior, allocation, graph, blob


def swap_invariance_hashes(coordinates: np.ndarray, q: float, records: RawRecords,
                           grid: FullGrid, chromosomes: Sequence[int] | None = None,
                           allocation_seed: int = ALLOC_SEED) -> dict[str, str]:
    """在纯 copy-label swap 后比较规范坐标/edges/blob。"""
    first = build_integer_proposal(coordinates, q, records, grid, allocation_seed)
    swapped_input = swap_copy_subset(coordinates, grid, chromosomes)
    second = build_integer_proposal(swapped_input, q, records, grid, allocation_seed)
    if not np.array_equal(first[0].coordinates, second[0].coordinates):
        raise FdgProtocolError("canonical coordinates changed under copy swap")
    if not np.array_equal(first[3].unique_edges, second[3].unique_edges) or \
            not np.array_equal(first[3].unique_counts, second[3].unique_counts):
        raise FdgProtocolError("edge-count table changed under copy swap")
    if first[1].probabilities.shape != second[1].probabilities.shape or \
            not np.array_equal(first[1].probabilities, second[1].probabilities):
        raise FdgProtocolError("posterior changed under copy swap")
    if first[2].state.shape != second[2].state.shape or \
            not np.array_equal(first[2].state, second[2].state):
        raise FdgProtocolError("integer allocation changed under copy swap")
    first_hash = sha256_bytes(first[4])
    second_hash = sha256_bytes(second[4])
    if first_hash != second_hash:
        raise FdgProtocolError("native source blob changed under copy swap")
    return {
        "canonical_coordinates_sha256": hashlib.sha256(
            np.asarray(first[0].coordinates, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
        "edge_table_sha256": hashlib.sha256(
            np.column_stack((first[3].unique_edges, first[3].unique_counts))
            .astype("<i8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "bridge_source_blob_sha256": first_hash,
    }


__all__ = [
    "ALPHA_SEQUENCE", "ALLOC_SEED", "BIN_SIZE", "CanonicalCoordinates",
    "ForceGraph", "FullGrid", "FdgProtocolError", "INTERIOR_TARGET",
    "NATIVE_SEED", "PosteriorResult", "RawRecords", "AllocationResult",
    "BridgeOutput", "BridgeRun", "backtracking_coordinates", "bead_order_to_coordinates",
    "build_integer_proposal", "canonicalize_coordinates", "coordinates_to_bead_order",
    "force_graph_from_allocation", "integer_allocate", "make_full_grid",
    "make_raw_records", "map_native_to_unit_ball", "pack_bridge_input", "p_from_q",
    "posterior_from_coordinates", "read_bridge_output", "sha256_bytes", "sha256_file",
    "sphere_inverse", "state_count_table", "swap_copy_subset", "swap_invariance_hashes",
    "undo_canonicalization", "validate_full_grid", "validate_raw_records",
]
