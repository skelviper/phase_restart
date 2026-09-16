"""Independent exact Torch CPU/CUDA backend for formal experiment 020.

The backend follows the archived 020 executed contact_model.py formulas while
using sparse observed counts and an implicit complete upper-triangle pair grid.
It never imports phase-bearing data or reference coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable

# These caps must be present before NumPy/SciPy/Torch initialize their BLAS pools.
for _thread_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_key] = "1"

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
import torch


FROZEN_SNPFREE_SHA256 = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"
FROZEN_RECORDS = 1_703_888
FROZEN_INTRA = 1_135_454
FROZEN_INTER = 568_434
FROZEN_CHROMOSOME_HEADER = (
    ("chr1", 195_471_971), ("chr2", 182_113_224), ("chr3", 160_039_680),
    ("chr4", 156_508_116), ("chr5", 151_834_684), ("chr6", 149_736_546),
    ("chr7", 145_441_459), ("chr8", 129_401_213), ("chr9", 124_595_110),
    ("chr10", 130_694_993), ("chr11", 122_082_543), ("chr12", 120_129_022),
    ("chr13", 120_421_639), ("chr14", 124_902_244), ("chr15", 104_043_685),
    ("chr16", 98_207_768), ("chr17", 94_987_271), ("chr18", 90_702_639),
    ("chr19", 61_431_566), ("chrX", 171_031_299),
)
FROZEN_BIN_SIZES = (5_000_000, 2_000_000, 1_000_000, 500_000,
                    200_000, 100_000, 50_000, 20_000)
EPSILON = 1e-6
P_FLOOR = 1e-4
P_PRIOR_STRENGTH = 1e-4


class BackendError(RuntimeError):
    """Raised for a frozen input, device, or numerical contract failure."""


@dataclass(frozen=True)
class SparseAggregate:
    """One full-grid layer with only nonzero off-diagonal counts stored."""

    chromosome_names: tuple[str, ...]
    chromosome_lengths: np.ndarray
    bin_size: int
    n_bins: np.ndarray
    offsets: np.ndarray
    locus_chromosome: np.ndarray
    locus_bin: np.ndarray
    endpoint_counts: np.ndarray
    exposure: np.ndarray
    observed_flat: np.ndarray
    observed_counts: np.ndarray
    diag_counts: np.ndarray
    raw_records: int
    raw_same_bin: int
    raw_cis_offdiag: int
    raw_inter: int
    conditional_factorial_constant: float
    diag_factorial_sum: float

    @property
    def n_loci(self) -> int:
        return int(len(self.locus_chromosome))

    @property
    def n_pairs(self) -> int:
        return self.n_loci * (self.n_loci - 1) // 2

    @property
    def l0(self) -> float:
        return float((2.0 * self.n_loci) ** (-1.0 / 3.0))

    @property
    def r0(self) -> float:
        return 2.0 * self.l0

    @property
    def repulsion_threshold(self) -> float:
        # This is the value executed by archived 020 contact_model.py.
        return 0.7 * self.l0

    def budget(self) -> dict[str, Any]:
        observed_total = int(self.observed_counts.sum())
        return {
            "n_loci": self.n_loci,
            "n_eligible_pairs": self.n_pairs,
            "n_observed_nonzero_pairs": int(len(self.observed_flat)),
            "n_zero_eligible_pairs": int(self.n_pairs - len(self.observed_flat)),
            "n_diag_bins": self.n_loci,
            "raw_records": int(self.raw_records),
            "raw_same_bin": int(self.raw_same_bin),
            "raw_cis_offdiag": int(self.raw_cis_offdiag),
            "raw_inter": int(self.raw_inter),
            "aggregate_same_bin": int(self.diag_counts.sum()),
            "aggregate_offdiag": observed_total,
            "endpoint_total": int(self.endpoint_counts.sum()),
            "conditional_factorial_constant_omitted": float(self.conditional_factorial_constant),
            "diag_factorial_sum_included": float(self.diag_factorial_sum),
            "factorial_constant_status": "integer_count_constants_recorded",
            "raw_conserved": bool(self.raw_records == self.raw_same_bin + self.raw_cis_offdiag + self.raw_inter),
            "aggregate_conserved": bool(int(self.diag_counts.sum()) == self.raw_same_bin
                                         and observed_total == self.raw_cis_offdiag + self.raw_inter),
            "endpoint_conserved": bool(int(self.endpoint_counts.sum()) == 2 * self.raw_records),
            "pair_storage": "sparse_nonzero_observed_plus_implicit_complete_upper_triangle",
        }

    def assert_consistent(self) -> None:
        if self.n_pairs != self.n_loci * (self.n_loci - 1) // 2:
            raise BackendError("eligible pair grid is not complete")
        if len(self.observed_flat) != len(self.observed_counts):
            raise BackendError("sparse count arrays have different lengths")
        if len(self.observed_flat) and np.any(np.diff(self.observed_flat) <= 0):
            raise BackendError("sparse pair indices are not strictly increasing")
        if len(self.observed_flat) and (self.observed_flat[0] < 0 or self.observed_flat[-1] >= self.n_pairs):
            raise BackendError("sparse pair index outside complete upper triangle")
        if (not np.issubdtype(self.observed_counts.dtype, np.integer)
                or not np.issubdtype(self.diag_counts.dtype, np.integer)):
            raise BackendError("raw sparse and same-bin counts must remain integers")
        if np.any(self.observed_counts < 0) or np.any(self.diag_counts < 0):
            raise BackendError("raw counts must be nonnegative")
        if (self.raw_records < 0 or self.raw_same_bin < 0 or self.raw_cis_offdiag < 0
                or self.raw_inter < 0):
            raise BackendError("raw contact totals must be nonnegative")
        if self.diag_factorial_sum < 0.0 or not math.isfinite(self.diag_factorial_sum):
            raise BackendError("same-bin factorial constant is invalid")
        if not math.isfinite(self.conditional_factorial_constant):
            raise BackendError("conditional factorial constant is invalid")
        if not np.all(np.isfinite(self.exposure)) or np.any(self.exposure <= 0.0):
            raise BackendError("exposure is not finite positive")
        if not np.isclose(float(self.exposure.mean()), 1.0, rtol=0.0, atol=1e-14):
            raise BackendError("exposure mean is not one")
        audit = self.budget()
        if not (audit["raw_conserved"] and audit["aggregate_conserved"] and audit["endpoint_conserved"]):
            raise BackendError("contact budget is not conserved")


def sha256_file(path: str | Path, block_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _upper_pair_index(i: np.ndarray, j: np.ndarray, n_loci: int) -> np.ndarray:
    return i * (2 * n_loci - i - 1) // 2 + (j - i - 1)


def _header_and_records(path: str | Path):
    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    names: list[str] = []
    lengths: list[int] = []
    columns: list[str] | None = None
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.startswith("#chromosome:"):
                fields = line.rstrip("\n").split()
                if len(fields) != 3:
                    raise BackendError(f"malformed chromosome header at line {line_no}")
                names.append(fields[1])
                lengths.append(int(fields[2]))
                continue
            if line.startswith("#columns:"):
                columns = line.rstrip("\n").split(":", 1)[1].strip().split("\t")
                continue
            if line.startswith("#"):
                continue
            break
    expected = ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"]
    if columns != expected or any("phase" in c.lower() or "prob" in c.lower() for c in (columns or [])):
        raise BackendError(f"training schema is not the exact seven-column SNP-free schema: {columns}")
    if not names or len(set(names)) != len(names):
        raise BackendError("chromosome header inventory is empty or duplicated")
    header = tuple(zip(names, (int(value) for value in lengths)))
    if header != FROZEN_CHROMOSOME_HEADER:
        raise BackendError(f"frozen chromosome header mismatch: {header}")
    return tuple(names), np.asarray(lengths, dtype=np.int64)


def load_sparse_aggregate(path: str | Path, bin_size: int = 1_000_000,
                          verify_hash: bool = True) -> SparseAggregate:
    """Load all raw records and aggregate without materializing pair-grid arrays."""
    path = str(path)
    if not isinstance(bin_size, (int, np.integer)) or int(bin_size) not in FROZEN_BIN_SIZES:
        raise BackendError(f"bin_size must be one of frozen resolutions {FROZEN_BIN_SIZES}: {bin_size!r}")
    bin_size = int(bin_size)
    if verify_hash:
        actual = sha256_file(path)
        if actual != FROZEN_SNPFREE_SHA256:
            raise BackendError(f"SNP-free SHA256 mismatch: {actual} != {FROZEN_SNPFREE_SHA256}")
    names, lengths = _header_and_records(path)
    n_bins = ((lengths + int(bin_size) - 1) // int(bin_size)).astype(np.int64)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(n_bins[:-1])))
    n_loci = int(n_bins.sum())
    chrom_index = {name: i for i, name in enumerate(names)}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        n_records = sum(1 for line in handle if not line.startswith("#"))
    g1 = np.empty(n_records, dtype=np.int64)
    g2 = np.empty(n_records, dtype=np.int64)
    cursor = 0
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 7:
                raise BackendError(f"row {line_no} has {len(fields)} fields, expected 7")
            try:
                c1, p1, c2, p2 = fields[1], int(fields[2]), fields[3], int(fields[4])
            except (IndexError, ValueError) as exc:
                raise BackendError(f"invalid record at line {line_no}") from exc
            if c1 not in chrom_index or c2 not in chrom_index or p1 < 0 or p2 < 0:
                raise BackendError(f"invalid chromosome/position at line {line_no}")
            i1, i2 = chrom_index[c1], chrom_index[c2]
            if p1 >= lengths[i1] or p2 >= lengths[i2]:
                raise BackendError(f"position outside chromosome header at line {line_no}")
            first = int(offsets[i1] + p1 // int(bin_size))
            second = int(offsets[i2] + p2 // int(bin_size))
            if first > second and i1 == i2:
                first, second = second, first
            g1[cursor] = first
            g2[cursor] = second
            cursor += 1
    if cursor != n_records:
        raise BackendError(f"record count changed between input passes: {cursor} != {n_records}")
    if len(g1) != FROZEN_RECORDS:
        raise BackendError(f"record count mismatch: {len(g1)} != {FROZEN_RECORDS}")
    locus_chromosome = np.repeat(np.arange(len(names), dtype=np.int32), n_bins)
    locus_bin = np.concatenate([np.arange(int(n), dtype=np.int64) for n in n_bins])
    cis_raw = locus_chromosome[g1] == locus_chromosome[g2]
    same_bin = g1 == g2
    if np.any(same_bin & ~cis_raw):
        raise BackendError("inter-chromosome record landed in same global bin")
    raw_same_bin = int(same_bin.sum())
    raw_cis_offdiag = int((cis_raw & ~same_bin).sum())
    raw_inter = int((~cis_raw).sum())
    if raw_inter != FROZEN_INTER or raw_same_bin + raw_cis_offdiag != FROZEN_INTRA:
        raise BackendError(
            "frozen raw budget mismatch at every resolution: "
            f"same_bin={raw_same_bin}, cis_offdiag={raw_cis_offdiag}, inter={raw_inter}")
    eligible = ~(same_bin)
    flat = _upper_pair_index(np.minimum(g1[eligible], g2[eligible]),
                             np.maximum(g1[eligible], g2[eligible]), n_loci)
    observed_flat, observed_counts = np.unique(flat, return_counts=True)
    observed_flat = observed_flat.astype(np.int64, copy=False)
    observed_counts = observed_counts.astype(np.int64, copy=False)
    diag_counts = np.bincount(g1[same_bin], minlength=n_loci).astype(np.int64, copy=False)
    endpoint_counts = np.bincount(np.concatenate((g1, g2)), minlength=n_loci).astype(np.int64, copy=False)
    exposure = np.sqrt(endpoint_counts.astype(np.float64) + 10.0)
    exposure /= exposure.mean()
    conditional_factorial_constant = 0.0
    # Partition observed sparse counts by layer without building an N^2 mask.
    eligible_cis = cis_raw[eligible]
    cis_counts = np.unique(flat[eligible_cis], return_counts=True)[1]
    inter_counts = np.unique(flat[~eligible_cis], return_counts=True)[1]
    positive_cis = cis_counts[cis_counts > 1]
    positive_inter = inter_counts[inter_counts > 1]
    conditional_factorial_constant += -float(gammaln(float(raw_cis_offdiag) + 1.0))
    conditional_factorial_constant += float(gammaln(positive_cis.astype(np.float64) + 1.0).sum())
    conditional_factorial_constant += -float(gammaln(float(raw_inter) + 1.0))
    conditional_factorial_constant += float(gammaln(positive_inter.astype(np.float64) + 1.0).sum())
    positive_diag = diag_counts[diag_counts > 1]
    diag_factorial_sum = float(gammaln(positive_diag.astype(np.float64) + 1.0).sum())
    result = SparseAggregate(
        chromosome_names=names,
        chromosome_lengths=lengths,
        bin_size=int(bin_size),
        n_bins=n_bins,
        offsets=offsets,
        locus_chromosome=locus_chromosome,
        locus_bin=locus_bin,
        endpoint_counts=endpoint_counts,
        exposure=exposure,
        observed_flat=observed_flat,
        observed_counts=observed_counts,
        diag_counts=diag_counts,
        raw_records=int(len(g1)),
        raw_same_bin=raw_same_bin,
        raw_cis_offdiag=raw_cis_offdiag,
        raw_inter=raw_inter,
        conditional_factorial_constant=conditional_factorial_constant,
        diag_factorial_sum=diag_factorial_sum,
    )
    result.assert_consistent()
    return result


def sphere_forward(y: torch.Tensor) -> torch.Tensor:
    radius = torch.sqrt(1.0 + torch.sum(y * y, dim=-1, keepdim=True))
    return y / radius


def sphere_pullback(y: torch.Tensor, gradient_x: torch.Tensor) -> torch.Tensor:
    radius = torch.sqrt(1.0 + torch.sum(y * y, dim=-1, keepdim=True))
    x = y / radius
    return (gradient_x - x * torch.sum(x * gradient_x, dim=-1, keepdim=True)) / radius


def p_from_q(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sigmoid = torch.sigmoid(q)
    span = 1.0 - 2.0 * P_FLOOR
    p = P_FLOOR + span * sigmoid
    return p, span * sigmoid * (1.0 - sigmoid)


class TorchObjective:
    """Analytic exact objective evaluated on CPU or CUDA in bounded pair tiles."""

    def __init__(self, data: SparseAggregate, device: str = "cpu", tile_rows: int = 32,
                 dtype: torch.dtype = torch.float64):
        data.assert_consistent()
        if tile_rows <= 0:
            raise ValueError("tile_rows must be positive")
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise BackendError("CUDA requested but torch.cuda.is_available() is false")
        self.data = data
        self.device = requested
        self.dtype = dtype
        self.tile_rows = int(tile_rows)
        self.n_parameters = 6 * data.n_loci + 1
        self._locus_chromosome = torch.as_tensor(data.locus_chromosome, dtype=torch.int64, device=requested)
        self._exposure = torch.as_tensor(data.exposure, dtype=dtype, device=requested)
        self._observed_flat = torch.as_tensor(data.observed_flat, dtype=torch.int64, device=requested)
        self._observed_counts = torch.as_tensor(data.observed_counts, dtype=dtype, device=requested)
        self._diag_counts = torch.as_tensor(data.diag_counts, dtype=dtype, device=requested)
        self._cache_theta: np.ndarray | None = None
        self._cache_value: float | None = None
        self._cache_gradient: np.ndarray | None = None
        self._cache_components: dict[str, Any] | None = None
        # Host-to-device transfers may be queued; upload timing callers rely on
        # this barrier to include completion of every constructor transfer.
        self._sync()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def synchronize(self) -> None:
        """Synchronize all queued work for explicit scoped timing boundaries."""
        self._sync()

    def _pair_rows(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.arange(start, stop, dtype=np.int64)
        i = np.repeat(rows, self.data.n_loci - rows - 1)
        j = np.concatenate([np.arange(row + 1, self.data.n_loci, dtype=np.int64) for row in rows])
        flat = _upper_pair_index(i, j, self.data.n_loci)
        return i, j, flat

    def _pair_block_tensors(self, start: int, stop: int):
        i, j, flat = self._pair_rows(start, stop)
        i_t = torch.as_tensor(i, dtype=torch.int64, device=self.device)
        j_t = torch.as_tensor(j, dtype=torch.int64, device=self.device)
        flat_t = torch.as_tensor(flat, dtype=torch.int64, device=self.device)
        counts = torch.zeros(flat_t.shape, dtype=self.dtype, device=self.device)
        if not len(self.data.observed_flat):
            valid = torch.zeros(flat_t.shape, dtype=torch.bool, device=self.device)
        else:
            positions = torch.searchsorted(self._observed_flat, flat_t)
            valid = positions < len(self.data.observed_flat)
            safe_positions = torch.clamp(positions, max=len(self.data.observed_flat) - 1)
            valid = valid & (self._observed_flat[safe_positions] == flat_t)
            counts[valid] = self._observed_counts[safe_positions[valid]]
        cis = self._locus_chromosome[i_t] == self._locus_chromosome[j_t]
        exposure = self._exposure[i_t] * self._exposure[j_t]
        return i_t, j_t, cis, counts, exposure

    def _rate_block(self, x: torch.Tensor, i: torch.Tensor, j: torch.Tensor,
                    cis: torch.Tensor, p: torch.Tensor, with_kernels: bool = False):
        aa = x[0, i] - x[0, j]
        ab = x[0, i] - x[1, j]
        ba = x[1, i] - x[0, j]
        bb = x[1, i] - x[1, j]
        r0sq = self.data.r0 * self.data.r0
        def kernel(delta: torch.Tensor):
            base = 1.0 + torch.sum(delta * delta, dim=-1) / r0sq
            value = EPSILON + (1.0 - EPSILON) * base.pow(-2)
            gradient = (-4.0 * (1.0 - EPSILON) / r0sq) * delta * base.pow(-3).unsqueeze(-1)
            return value, gradient
        kaa, gaa = kernel(aa)
        kab, gab = kernel(ab)
        kba, gba = kernel(ba)
        kbb, gbb = kernel(bb)
        same = kaa + kbb
        cross = kab + kba
        mixture = torch.where(cis, 0.5 * (p * same + (1.0 - p) * cross), 0.25 * (same + cross))
        rate = self._exposure[i] * self._exposure[j] * mixture
        if not with_kernels:
            return rate
        return rate, (kaa, kab, kba, kbb), (gaa, gab, gba, gbb), self._exposure[i] * self._exposure[j]

    def _count_layer(self, x: torch.Tensor, p: torch.Tensor, need_gradient: bool):
        sum_cis = torch.zeros((), dtype=self.dtype, device=self.device)
        sum_inter = torch.zeros((), dtype=self.dtype, device=self.device)
        log_cis = torch.zeros((), dtype=self.dtype, device=self.device)
        log_inter = torch.zeros((), dtype=self.dtype, device=self.device)
        for start in range(0, max(0, self.data.n_loci - 1), self.tile_rows):
            stop = min(start + self.tile_rows, self.data.n_loci - 1)
            i, j, cis, counts, exposure = self._pair_block_tensors(start, stop)
            rate = self._rate_block(x, i, j, cis, p)
            sum_cis = sum_cis + rate[cis].sum()
            sum_inter = sum_inter + rate[~cis].sum()
            observed = counts * torch.log(rate)
            log_cis = log_cis + observed[cis].sum()
            log_inter = log_inter + observed[~cis].sum()
        group_cis = torch.as_tensor(float(self.data.raw_cis_offdiag), dtype=self.dtype, device=self.device)
        group_inter = torch.as_tensor(float(self.data.raw_inter), dtype=self.dtype, device=self.device)
        conditional = group_cis * torch.log(sum_cis) - log_cis + group_inter * torch.log(sum_inter) - log_inter
        positive = self._diag_counts[self._diag_counts > 0]
        diag = (positive - positive * torch.log(positive) + torch.lgamma(positive + 1.0)).sum()
        count_raw = conditional + diag
        components: dict[str, Any] = {
            "sum_rate_cis_offdiag": float(sum_cis.detach().cpu().item()),
            "sum_rate_inter": float(sum_inter.detach().cpu().item()),
            "observed_log_rate_cis_offdiag": float(log_cis.detach().cpu().item()),
            "observed_log_rate_inter": float(log_inter.detach().cpu().item()),
            "conditional_nll_raw": float(conditional.detach().cpu().item()),
            "conditional_factorial_constant_omitted": float(self.data.conditional_factorial_constant),
            "diag_factorial_sum_included": float(self.data.diag_factorial_sum),
            "count_factorial_status": "integer_count_constants_recorded",
            "diag_profiled_nll_raw": float(diag.detach().cpu().item()),
            "count_nll_raw": float(count_raw.detach().cpu().item()),
            "count_nll_normalized": float((count_raw / float(self.data.raw_records)).detach().cpu().item()),
        }
        if not need_gradient:
            return components, None, None
        gradient_x = torch.zeros_like(x)
        gradient_p = torch.zeros((), dtype=self.dtype, device=self.device)
        for start in range(0, max(0, self.data.n_loci - 1), self.tile_rows):
            stop = min(start + self.tile_rows, self.data.n_loci - 1)
            i, j, cis, counts, exposure = self._pair_block_tensors(start, stop)
            rate, kernels, kernel_gradients, exposure = self._rate_block(x, i, j, cis, p, with_kernels=True)
            kaa, kab, kba, kbb = kernels
            gaa, gab, gba, gbb = kernel_gradients
            weights = -counts / rate
            weights = weights + torch.where(
                cis,
                group_cis / sum_cis,
                group_inter / sum_inter,
            )
            same_coefficient = exposure * torch.where(cis, 0.5 * p, torch.as_tensor(0.25, dtype=self.dtype, device=self.device)) * weights
            cross_coefficient = exposure * torch.where(cis, 0.5 * (1.0 - p), torch.as_tensor(0.25, dtype=self.dtype, device=self.device)) * weights
            for copy_i, copy_j, gradient_kernel, coefficient in (
                (0, 0, gaa, same_coefficient),
                (0, 1, gab, cross_coefficient),
                (1, 0, gba, cross_coefficient),
                (1, 1, gbb, same_coefficient),
            ):
                term = coefficient.unsqueeze(-1) * gradient_kernel
                gradient_x[copy_i].index_add_(0, i, term)
                gradient_x[copy_j].index_add_(0, j, -term)
            derivative_p = 0.5 * exposure * (kaa + kbb - kab - kba)
            gradient_p = gradient_p + (weights[cis] * derivative_p[cis]).sum()
        normalizer = float(self.data.raw_records)
        return components, gradient_x / normalizer, gradient_p / normalizer

    def _bond_layer(self, x: torch.Tensor):
        gradient = torch.zeros_like(x)
        value = torch.zeros((), dtype=self.dtype, device=self.device)
        n_term = 0
        for chromosome in range(len(self.data.chromosome_names)):
            start = int(self.data.offsets[chromosome])
            stop = start + int(self.data.n_bins[chromosome])
            for copy in (0, 1):
                delta = x[copy, start + 1:stop] - x[copy, start:stop - 1]
                distance = torch.sqrt(torch.sum(delta * delta, dim=1))
                scaled = distance / self.data.l0
                low = torch.relu(0.75 - scaled)
                high = torch.relu(scaled - 1.25)
                value = value + (low * low + high * high).sum()
                derivative_distance = (-2.0 * low + 2.0 * high) / self.data.l0
                coefficient = torch.where(distance > 0.0, derivative_distance / distance, torch.zeros_like(distance))
                term = coefficient.unsqueeze(-1) * delta
                gradient[copy, start + 1:stop].index_add_(0, torch.arange(stop - start - 1, device=self.device), term)
                gradient[copy, start:stop - 1].index_add_(0, torch.arange(stop - start - 1, device=self.device), -term)
                n_term += stop - start - 1
        if n_term == 0:
            return torch.zeros((), dtype=self.dtype, device=self.device), gradient
        return value / n_term, gradient / n_term

    def _repulsion_piece(self, delta: torch.Tensor):
        distance = torch.sqrt(torch.sum(delta * delta, dim=-1))
        hinge = torch.relu(1.0 - distance / self.data.repulsion_threshold)
        derivative_distance = -2.0 * hinge / self.data.repulsion_threshold
        coefficient = torch.where(distance > 0.0, derivative_distance / distance, torch.zeros_like(distance))
        return (hinge * hinge).sum(), coefficient.unsqueeze(-1) * delta

    def _repulsion_layer(self, x: torch.Tensor):
        gradient = torch.zeros_like(x)
        value = torch.zeros((), dtype=self.dtype, device=self.device)
        for start in range(0, max(0, self.data.n_loci - 1), self.tile_rows):
            stop = min(start + self.tile_rows, self.data.n_loci - 1)
            i, j, _cis, _counts, _exposure = self._pair_block_tensors(start, stop)
            for copy_i, copy_j in ((0, 0), (0, 1), (1, 0), (1, 1)):
                piece, term = self._repulsion_piece(x[copy_i, i] - x[copy_j, j])
                value = value + piece
                gradient[copy_i].index_add_(0, i, term)
                gradient[copy_j].index_add_(0, j, -term)
        piece, term = self._repulsion_piece(x[0] - x[1])
        value = value + piece
        gradient[0] = gradient[0] + term
        gradient[1] = gradient[1] - term
        normalizer = float(2 * self.data.n_loci)
        return value / normalizer, gradient / normalizer

    def _bend_layer(self, x: torch.Tensor):
        gradient = torch.zeros_like(x)
        value = torch.zeros((), dtype=self.dtype, device=self.device)
        n_term = 0
        l0sq = self.data.l0 * self.data.l0
        for chromosome in range(len(self.data.chromosome_names)):
            start = int(self.data.offsets[chromosome])
            stop = start + int(self.data.n_bins[chromosome])
            for copy in (0, 1):
                second = x[copy, start + 2:stop] - 2.0 * x[copy, start + 1:stop - 1] + x[copy, start:stop - 2]
                value = value + (second * second).sum() / l0sq
                term = 2.0 * second / l0sq
                gradient[copy, start:stop - 2] = gradient[copy, start:stop - 2] + term
                gradient[copy, start + 1:stop - 1] = gradient[copy, start + 1:stop - 1] - 2.0 * term
                gradient[copy, start + 2:stop] = gradient[copy, start + 2:stop] + term
                n_term += stop - start - 2
        if n_term == 0:
            return torch.zeros((), dtype=self.dtype, device=self.device), gradient
        return value / n_term, gradient / n_term

    def evaluate(self, theta: np.ndarray, need_gradient: bool = True):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,) or not np.all(np.isfinite(theta)):
            raise BackendError("theta has wrong shape or nonfinite values")
        theta_t = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        y = theta_t[:-1].reshape(2, self.data.n_loci, 3)
        q = theta_t[-1]
        x = sphere_forward(y)
        p, dpdq = p_from_q(q)
        count_components, count_gradient_x, count_gradient_p = self._count_layer(x, p, need_gradient)
        bond, bond_gradient = self._bond_layer(x)
        repulsion, repulsion_gradient = self._repulsion_layer(x)
        bend, bend_gradient = self._bend_layer(x)
        p_prior = -P_PRIOR_STRENGTH * torch.log(p * (1.0 - p))
        p_prior_derivative = P_PRIOR_STRENGTH * (1.0 / (1.0 - p) - 1.0 / p)
        total = (torch.as_tensor(count_components["count_nll_normalized"], dtype=self.dtype, device=self.device)
                 + p_prior + bond + repulsion + 0.01 * bend)
        components = dict(count_components)
        components.update({
            "p": float(p.detach().cpu().item()),
            "p_prior": float(p_prior.detach().cpu().item()),
            "bond": float(bond.detach().cpu().item()),
            "repulsion": float(repulsion.detach().cpu().item()),
            "bend": float(bend.detach().cpu().item()),
            "total": float(total.detach().cpu().item()),
            "repulsion_threshold_l0": 0.7,
        })
        if not need_gradient:
            self._sync()
            return components["total"], None, components
        combined_x = count_gradient_x + bond_gradient + repulsion_gradient + 0.01 * bend_gradient
        gradient_y = sphere_pullback(y, combined_x)
        gradient_q = dpdq * (count_gradient_p + p_prior_derivative)
        gradient = torch.cat((gradient_y.reshape(-1), gradient_q.reshape(1)))
        value = components["total"]
        gradient_np = gradient.detach().cpu().numpy().astype(np.float64, copy=False)
        self._sync()
        if not np.all(np.isfinite(gradient_np)) or not math.isfinite(value):
            raise FloatingPointError("nonfinite objective or gradient")
        return value, gradient_np.copy(), components

    def value_and_grad(self, theta: np.ndarray):
        value, gradient, components = self.evaluate(theta, need_gradient=True)
        self._cache_theta = np.asarray(theta, dtype=np.float64).copy()
        self._cache_value = float(value)
        self._cache_gradient = gradient.copy()
        self._cache_components = dict(components)
        return value, gradient

    def components(self, theta: np.ndarray) -> dict[str, Any]:
        theta = np.asarray(theta, dtype=np.float64)
        if self._cache_theta is not None and np.array_equal(theta, self._cache_theta):
            return dict(self._cache_components or {})
        return self.evaluate(theta, need_gradient=False)[2]

    def coordinates_and_p(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        theta_t = torch.as_tensor(np.asarray(theta, dtype=np.float64), dtype=self.dtype, device=self.device)
        y = theta_t[:-1].reshape(2, self.data.n_loci, 3)
        x = sphere_forward(y)
        p, _ = p_from_q(theta_t[-1])
        self._sync()
        coordinates = x.detach().cpu().numpy().astype(np.float64, copy=False)
        return coordinates.copy(), float(p.detach().cpu().item())


def fit_l_bfgs(objective: TorchObjective, theta0: np.ndarray, maxiter: int = 50,
               maxfun: int | None = None, maxls: int = 20, ftol: float = 1e-10,
               gtol: float = 1e-6, checkpoint_every: int | None = None,
               checkpoint_hook: Any | None = None) -> dict[str, Any]:
    """Optional host SciPy optimizer; CUDA synchronization is explicit in callbacks."""
    if checkpoint_hook is not None and (checkpoint_every is None or checkpoint_every <= 0):
        checkpoint_every = 10
    started = time.perf_counter()
    calls = 0
    history: list[dict[str, Any]] = []
    theta0 = np.asarray(theta0, dtype=np.float64).copy()
    initial_value, initial_gradient, initial_components = objective.evaluate(theta0, need_gradient=True)
    last_gradient = np.asarray(initial_gradient, dtype=np.float64).copy()
    last_theta = theta0.copy()
    history.append({
        "iteration": 0, "nfev": 0, "elapsed_seconds": 0.0,
        "fun": float(initial_value), "gradient_l2": float(np.linalg.norm(last_gradient)),
        "gradient_max_abs": float(np.max(np.abs(last_gradient))),
        "components": initial_components,
    })

    def loss_gradient(theta):
        nonlocal calls, last_gradient, last_theta
        calls += 1
        value, gradient = objective.value_and_grad(theta)
        last_gradient = np.asarray(gradient, dtype=np.float64).copy()
        last_theta = np.asarray(theta, dtype=np.float64).copy()
        return value, gradient

    def callback(theta):
        nonlocal last_gradient, last_theta
        theta = np.asarray(theta, dtype=np.float64)
        if not np.array_equal(theta, last_theta):
            _, refreshed_gradient = objective.value_and_grad(theta)
            last_gradient = np.asarray(refreshed_gradient, dtype=np.float64).copy()
            last_theta = theta.copy()
        components = objective.components(theta)
        iteration = len(history)
        history.append({
            "iteration": iteration,
            "nfev": calls,
            "elapsed_seconds": time.perf_counter() - started,
            "fun": float(components["total"]),
            "gradient_l2": float(np.linalg.norm(last_gradient)),
            "gradient_max_abs": float(np.max(np.abs(last_gradient))),
            "components": components,
        })
        if checkpoint_hook is not None and checkpoint_every is not None and iteration % checkpoint_every == 0:
            coordinates, p = objective.coordinates_and_p(theta)
            checkpoint_hook({
                "iteration": iteration,
                "nfev": calls,
                "elapsed_seconds": time.perf_counter() - started,
                "fun": float(components["total"]),
                "p": p,
                "theta": theta.copy(),
                "coordinates": coordinates,
                "components": dict(components),
                "history": list(history),
            })

    options = {"maxiter": int(maxiter), "maxls": int(maxls), "ftol": float(ftol), "gtol": float(gtol)}
    if maxfun is not None:
        options["maxfun"] = int(maxfun)
    result = minimize(loss_gradient, theta0, method="L-BFGS-B", jac=True, callback=callback, options=options)
    final_value, final_gradient, final_components = objective.evaluate(result.x, need_gradient=True)
    message = str(result.message)
    if result.success:
        termination_class = "converged"
    elif int(result.status) == 1 or int(result.nfev) >= int(maxfun or 2**63 - 1) or int(result.nit) >= int(maxiter):
        termination_class = "budget_not_converged"
    elif int(result.status) == 2 or "ABNORMAL" in message.upper() or "LINE SEARCH" in message.upper():
        termination_class = "line_search_abort"
    else:
        termination_class = "optimizer_failed"
    return {
        "theta": np.asarray(result.x, dtype=np.float64),
        "coordinates": objective.coordinates_and_p(result.x)[0],
        "p": objective.coordinates_and_p(result.x)[1],
        "initial_total": float(initial_value),
        "final_total": float(final_value),
        "initial_components": initial_components,
        "final_components": final_components,
        "final_gradient_l2": float(np.linalg.norm(final_gradient)),
        "final_gradient_max_abs": float(np.max(np.abs(final_gradient))),
        "success": bool(result.success),
        "status": int(result.status),
        "termination_class": termination_class,
        "message": message,
        "nit": int(result.nit),
        "nfev": int(calls),
        "scipy_nfev": int(getattr(result, "nfev", calls)),
        "njev": int(getattr(result, "njev", 0)),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }


def threadpool_audit() -> dict[str, Any]:
    """Report effective native thread pools after the pre-import caps."""
    result: dict[str, Any] = {
        key: os.environ.get(key) for key in
        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    }
    try:
        from threadpoolctl import threadpool_info
        result["libraries"] = [
            {key: info.get(key) for key in ("user_api", "internal_api", "prefix", "version", "num_threads")}
            for info in threadpool_info()
        ]
    except Exception as exc:  # pragma: no cover - diagnostic fallback only
        result["threadpoolctl_error"] = f"{type(exc).__name__}: {exc}"
    return result


def require_cuda() -> dict[str, Any]:
    """Return a machine-readable probe without changing global CUDA state."""
    result: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if not result["cuda_available"]:
        result["status"] = "unavailable"
        result["error"] = "No CUDA GPUs are available"
        return result
    result["status"] = "available"
    result["devices"] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        result["devices"].append({"index": index, "name": props.name,
                                  "total_memory": int(props.total_memory)})
    return result


def initial_theta_from_coordinates(data: SparseAggregate, coordinates: np.ndarray, p: float = 0.75) -> np.ndarray:
    """Pack physical coordinates into the exact unconstrained optimizer variables."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (2, data.n_loci, 3):
        raise BackendError("coordinates have wrong copy-first shape")
    norm2 = np.sum(coordinates * coordinates, axis=-1, keepdims=True)
    if np.any(norm2 >= 1.0) or not np.all(np.isfinite(norm2)):
        raise BackendError("coordinates must be strictly inside unit ball")
    y = coordinates / np.sqrt(1.0 - norm2)
    z = (float(p) - P_FLOOR) / (1.0 - 2.0 * P_FLOOR)
    if not 0.0 < z < 1.0:
        raise BackendError("p must be strictly inside bounded interval")
    q = math.log(z / (1.0 - z))
    return np.concatenate((y.reshape(-1), np.asarray([q], dtype=np.float64)))
