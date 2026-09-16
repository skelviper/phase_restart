"""Phase-free full-grid contact aggregation and continuous V1 objective.

This module is deliberately independent of ``run.py`` and native FDG.  Its
training-side loaders accept only the seven-column SNP-free pairs schema.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import gammaln

from . import genome
from .paths import SNPFREE


FROZEN_P9016_SNPFREE_SHA256 = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"
FROZEN_P9016_RAW_SHA256 = "071a6cc76bfad543ea1ace6ee1ce3022b30ac1b1e50a9f0c3a3a1967b9649505"
FROZEN_P9016_RECORDS = 1_703_888
FROZEN_P9016_CIS_RECORDS = 1_135_454
FROZEN_P9016_INTER_RECORDS = 568_434

EPSILON = 1e-6
P_FLOOR = 1e-4
P_PRIOR_STRENGTH = 1e-4
DEFAULT_BLOCK_SIZE = 65_536


@dataclass(frozen=True)
class TrackSpec:
    """One output track, tied explicitly to SNP-free header order."""

    name: str
    chromosome_index: int
    chromosome_name: str
    copy_index: int


@dataclass
class AggregatedContacts:
    """Full-grid aggregate counts at one fixed genomic resolution.

    ``pair_i``/``pair_j`` enumerate every structural eligible pair, not only
    pairs observed in the input.  ``diag_counts`` is a separate same-genomic-bin
    count layer, indexed over the same full locus grid.
    """

    chromosome_names: tuple[str, ...]
    chromosome_lengths: np.ndarray
    bin_size: int
    n_bins: np.ndarray
    offsets: np.ndarray
    locus_chromosome: np.ndarray
    locus_bin: np.ndarray
    endpoint_counts: np.ndarray
    exposure: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    cis_pair: np.ndarray
    counts: np.ndarray
    diag_counts: np.ndarray
    raw_records: float
    raw_same_bin: float
    raw_cis_offdiag: float
    raw_inter: float
    conditional_factorial_constant: float
    diag_factorial_sum: float
    count_mode: str = "raw_integer"
    exposure_mode: str = "observed_endpoint"
    expected_rtol: float = 1e-10
    expected_atol: float = 1e-10

    @property
    def n_loci(self) -> int:
        return int(len(self.locus_chromosome))

    @property
    def n_pairs(self) -> int:
        return int(len(self.pair_i))

    @property
    def l0(self) -> float:
        return float((2.0 * self.n_loci) ** (-1.0 / 3.0))

    @property
    def r0(self) -> float:
        return 2.0 * self.l0

    @property
    def track_specs(self) -> tuple[TrackSpec, ...]:
        specs = []
        for ci, name in enumerate(self.chromosome_names):
            for copy_index in (0, 1):
                specs.append(TrackSpec(
                    name=genome.track(ci, copy_index),
                    chromosome_index=ci,
                    chromosome_name=name,
                    copy_index=copy_index,
                ))
        return tuple(specs)

    def global_locus(self, chromosome_index: np.ndarray | int,
                     local_bin: np.ndarray | int) -> np.ndarray:
        """Map header-order chromosome/bin coordinates to the full global grid."""
        ci = np.asarray(chromosome_index, dtype=np.int64)
        b = np.asarray(local_bin, dtype=np.int64)
        if np.any(ci < 0) or np.any(ci >= len(self.chromosome_names)):
            raise ValueError("chromosome index outside header-order grid")
        if np.any(b < 0) or np.any(b >= self.n_bins[ci]):
            raise ValueError("local bin outside full header grid")
        return self.offsets[ci] + b

    def chromosome_slice(self, chromosome_index: int) -> slice:
        if chromosome_index < 0 or chromosome_index >= len(self.chromosome_names):
            raise ValueError("chromosome index outside header-order grid")
        start = int(self.offsets[chromosome_index])
        return slice(start, start + int(self.n_bins[chromosome_index]))

    def budget(self) -> dict:
        """Return exact raw or tight synthetic-expected count accounting."""
        raw_mode = self.count_mode == "raw_integer"
        integer_mode = self.count_mode in ("raw_integer", "synthetic_integer")
        number = int if integer_mode else float
        aggregate_cis = number(self.counts[self.cis_pair].sum())
        aggregate_inter = number(self.counts[~self.cis_pair].sum())
        result = {
            "count_mode": self.count_mode,
            "exposure_mode": self.exposure_mode,
            "budget_unit": ("raw_records" if raw_mode else
                            ("synthetic_integer_records" if integer_mode
                             else "synthetic_expected_count_mass")),
            "raw_records": number(self.raw_records),
            "raw_same_bin": number(self.raw_same_bin),
            "raw_cis_offdiag": number(self.raw_cis_offdiag),
            "raw_inter": number(self.raw_inter),
            "aggregate_same_bin": number(self.diag_counts.sum()),
            "aggregate_cis_offdiag": aggregate_cis,
            "aggregate_inter": aggregate_inter,
            "endpoint_total": number(self.endpoint_counts.sum()),
            "n_chromosomes": int(len(self.chromosome_names)),
            "n_loci": int(self.n_loci),
            "n_eligible_pairs": int(self.n_pairs),
            "n_eligible_cis_offdiag_pairs": int(self.cis_pair.sum()),
            "n_eligible_inter_pairs": int((~self.cis_pair).sum()),
            "n_zero_eligible_pairs": int((self.counts == 0).sum()),
            "n_diag_bins": int(len(self.diag_counts)),
            "diag_nuisance_parameters": int(len(self.diag_counts)),
            "diag_profile": "per_bin_saturated_poisson",
            "conditional_count_factorial_constant_omitted": (
                float(self.conditional_factorial_constant) if integer_mode else None),
            "diag_count_factorial_sum_included": (
                float(self.diag_factorial_sum) if integer_mode else None),
            "factorial_constant_status": (
                "integer_count_constants_recorded" if integer_mode
                else "not_defined_for_fractional_expected_cross_entropy"),
        }
        if integer_mode:
            result["raw_conserved"] = bool(
                result["raw_records"] == result["raw_same_bin"]
                + result["raw_cis_offdiag"] + result["raw_inter"])
            result["aggregate_conserved"] = bool(
                result["raw_records"] == result["aggregate_same_bin"]
                + result["aggregate_cis_offdiag"] + result["aggregate_inter"])
            result["endpoint_conserved"] = bool(
                result["endpoint_total"] == 2 * result["raw_records"])
        else:
            result["raw_conserved"] = bool(np.isclose(
                result["raw_records"], result["raw_same_bin"]
                + result["raw_cis_offdiag"] + result["raw_inter"],
                rtol=self.expected_rtol, atol=self.expected_atol))
            result["aggregate_conserved"] = bool(np.isclose(
                result["raw_records"], result["aggregate_same_bin"]
                + result["aggregate_cis_offdiag"] + result["aggregate_inter"],
                rtol=self.expected_rtol, atol=self.expected_atol))
            result["endpoint_conserved"] = None
        return result

    def assert_consistent(self) -> None:
        if self.count_mode not in ("raw_integer", "synthetic_integer", "synthetic_expected"):
            raise AssertionError("unknown count mode")
        if self.expected_rtol < 0.0 or self.expected_atol < 0.0:
            raise AssertionError("expected-count tolerances must be nonnegative")
        if self.n_pairs != self.n_loci * (self.n_loci - 1) // 2:
            raise AssertionError("eligible pair grid is not complete")
        if self.counts.shape != (self.n_pairs,) or self.diag_counts.shape != (self.n_loci,):
            raise AssertionError("count arrays do not match the full grid")
        if (not np.all(np.isfinite(self.counts)) or not np.all(np.isfinite(self.diag_counts))
                or np.any(self.counts < 0.0) or np.any(self.diag_counts < 0.0)):
            raise AssertionError("counts must be finite and nonnegative")
        if not np.isclose(float(self.exposure.mean()), 1.0, rtol=0.0, atol=1e-14):
            raise AssertionError("full-grid exposure mean is not one")
        if not np.all(np.isfinite(self.exposure)) or not np.all(self.exposure > 0.0):
            raise AssertionError("fixed exposures must be finite and positive")
        audit = self.budget()
        if self.count_mode in ("raw_integer", "synthetic_integer"):
            if (not np.issubdtype(self.counts.dtype, np.integer)
                    or not np.issubdtype(self.diag_counts.dtype, np.integer)
                    or not np.issubdtype(self.endpoint_counts.dtype, np.integer)):
                raise AssertionError("integer count modes must retain exact integer counts")
            if not (audit["raw_conserved"] and audit["aggregate_conserved"]
                    and audit["endpoint_conserved"]):
                raise AssertionError("integer aggregate contact budget is not conserved")
        elif not (audit["raw_conserved"] and audit["aggregate_conserved"]):
            raise AssertionError("synthetic expected count mass is not conserved")


@dataclass(frozen=True)
class ObjectiveWeights:
    """Fixed V1 weights; tests can disable a component without changing formulas."""

    count: float = 1.0
    p_prior: float = 1.0
    bond: float = 1.0
    repulsion: float = 1.0
    bend: float = 0.01


def sha256_file(path: str | Path, block_bytes: int = 1 << 20) -> str:
    """Return the byte-level SHA256 without interpreting pairs content."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_snpfree(path: str | Path = SNPFREE) -> dict:
    """Verify the authorized SNP-free payload without opening raw phase data."""
    actual = sha256_file(path)
    if actual != FROZEN_P9016_SNPFREE_SHA256:
        raise RuntimeError(
            "SNP-free SHA256 mismatch: got %s, expected %s" %
            (actual, FROZEN_P9016_SNPFREE_SHA256))
    return {
        "snpfree_path": str(path),
        "snpfree_sha256": actual,
        "source_raw_sha256_frozen": FROZEN_P9016_RAW_SHA256,
    }


def _integer_vector(name: str, values: Iterable[int], length: int | None = None) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("%s must be one-dimensional" % name)
    if length is not None and len(raw) != length:
        raise ValueError("%s has length %d, expected %d" % (name, len(raw), length))
    out = raw.astype(np.int64, copy=False)
    if not np.array_equal(raw, out):
        raise ValueError("%s must contain exact integers" % name)
    return out


def _positive_log_factorial_sum(counts: np.ndarray) -> float:
    positive = np.asarray(counts, dtype=np.int64)
    positive = positive[positive > 1]
    return float(gammaln(positive.astype(np.float64) + 1.0).sum()) if len(positive) else 0.0


def _upper_pair_index(i: np.ndarray, j: np.ndarray, n_loci: int) -> np.ndarray:
    """Flat row-major index for a complete upper triangle with ``i < j``."""
    return i * (2 * n_loci - i - 1) // 2 + (j - i - 1)


def aggregate_from_arrays(chromosome_names: Sequence[str], chromosome_lengths: Sequence[int],
                          ci: Iterable[int], p1: Iterable[int], cj: Iterable[int],
                          p2: Iterable[int], bin_size: int) -> AggregatedContacts:
    """Aggregate all raw records into the complete full-header bin-pair grid.

    This accepts unlabeled numeric arrays so small synthetic fixtures can use the
    same construction as the production loader. It neither creates folds nor
    accepts label columns.
    """
    if int(bin_size) != bin_size or bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")
    bin_size = int(bin_size)
    names = tuple(str(name) for name in chromosome_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("chromosome names must be nonempty and unique")
    lengths = _integer_vector("chromosome_lengths", chromosome_lengths, len(names))
    if np.any(lengths <= 0):
        raise ValueError("chromosome lengths must be positive")
    n_bins = ((lengths + bin_size - 1) // bin_size).astype(np.int64)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(n_bins[:-1])))
    n_loci = int(n_bins.sum())

    ci = _integer_vector("ci", ci)
    p1 = _integer_vector("p1", p1, len(ci))
    cj = _integer_vector("cj", cj, len(ci))
    p2 = _integer_vector("p2", p2, len(ci))
    n_records = int(len(ci))
    for side_name, chrom, pos in (("first", ci, p1), ("second", cj, p2)):
        invalid_chromosome = (chrom < 0) | (chrom >= len(names))
        if invalid_chromosome.any():
            first = int(np.flatnonzero(invalid_chromosome)[0])
            raise ValueError("%s endpoint record %d has invalid chromosome" % (side_name, first))
        invalid_position = (pos < 0) | (pos >= lengths[chrom])
        if invalid_position.any():
            first = int(np.flatnonzero(invalid_position)[0])
            raise ValueError(
                "%s endpoint record %d falls outside [0, chromosome_length)" %
                (side_name, first))

    b1 = p1 // bin_size
    b2 = p2 // bin_size
    if np.any(b1 >= n_bins[ci]) or np.any(b2 >= n_bins[cj]):
        raise AssertionError("valid endpoints mapped outside terminal full-grid bins")
    g1 = offsets[ci] + b1
    g2 = offsets[cj] + b2
    endpoint_counts = np.bincount(
        np.concatenate((g1, g2)), minlength=n_loci).astype(np.int64, copy=False)

    cis_raw = ci == cj
    same_bin = g1 == g2
    if np.any(same_bin & ~cis_raw):
        raise AssertionError("global same-bin record cannot be inter-chromosomal")
    non_diag = ~same_bin
    raw_same_bin = int(same_bin.sum())
    raw_cis_offdiag = int((cis_raw & non_diag).sum())
    raw_inter = int((~cis_raw).sum())

    pair_i64, pair_j64 = np.triu_indices(n_loci, k=1)
    pair_i = pair_i64.astype(np.int32, copy=False)
    pair_j = pair_j64.astype(np.int32, copy=False)
    n_pairs = int(len(pair_i))
    if non_diag.any():
        lo = np.minimum(g1[non_diag], g2[non_diag])
        hi = np.maximum(g1[non_diag], g2[non_diag])
        flat = _upper_pair_index(lo, hi, n_loci)
        counts = np.bincount(flat, minlength=n_pairs).astype(np.int64, copy=False)
    else:
        counts = np.zeros(n_pairs, dtype=np.int64)
    diag_counts = np.bincount(g1[same_bin], minlength=n_loci).astype(np.int64, copy=False)
    locus_chromosome = np.repeat(np.arange(len(names), dtype=np.int32), n_bins)
    locus_bin = np.concatenate([np.arange(n, dtype=np.int64) for n in n_bins])
    cis_pair = (locus_chromosome[pair_i] == locus_chromosome[pair_j])
    exposure = np.sqrt(endpoint_counts.astype(np.float64) + 10.0)
    exposure /= exposure.mean()

    conditional_constant = 0.0
    for group_mask, group_total in ((cis_pair, raw_cis_offdiag),
                                    (~cis_pair, raw_inter)):
        if group_total:
            conditional_constant += -float(gammaln(float(group_total) + 1.0))
            conditional_constant += _positive_log_factorial_sum(counts[group_mask])
    diag_factorial_sum = _positive_log_factorial_sum(diag_counts)

    result = AggregatedContacts(
        chromosome_names=names,
        chromosome_lengths=lengths,
        bin_size=bin_size,
        n_bins=n_bins,
        offsets=offsets,
        locus_chromosome=locus_chromosome,
        locus_bin=locus_bin,
        endpoint_counts=endpoint_counts,
        exposure=exposure,
        pair_i=pair_i,
        pair_j=pair_j,
        cis_pair=cis_pair,
        counts=counts,
        diag_counts=diag_counts,
        raw_records=n_records,
        raw_same_bin=raw_same_bin,
        raw_cis_offdiag=raw_cis_offdiag,
        raw_inter=raw_inter,
        conditional_factorial_constant=conditional_constant,
        diag_factorial_sum=diag_factorial_sum,
    )
    result.assert_consistent()
    return result


def synthetic_expected_clone(template: AggregatedContacts, counts: np.ndarray,
                             diag_counts: np.ndarray, exposure: np.ndarray,
                             expected_group_totals: Mapping[str, float] | None = None,
                             exposure_mode: str = "synthetic_known",
                             endpoint_counts: np.ndarray | None = None,
                             rtol: float = 1e-10, atol: float = 1e-10) -> AggregatedContacts:
    """Build an explicit fractional expected-count calibration dataset.

    This is only for synthetic noiseless cross-entropy checks. It retains the
    template's complete header grid and track mapping, but does not pretend that
    fractional expected masses are Poisson observations. Factorial constants are
    consequently marked unavailable and never used for candidate selection.
    Production SNP-free aggregation remains ``raw_integer`` and exact.
    """
    template.assert_consistent()
    if not exposure_mode:
        raise ValueError("synthetic expected exposure_mode must be explicit")
    if not (math.isfinite(rtol) and math.isfinite(atol) and rtol >= 0.0 and atol >= 0.0):
        raise ValueError("synthetic expected tolerances must be finite and nonnegative")
    counts = np.asarray(counts, dtype=np.float64)
    diag_counts = np.asarray(diag_counts, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    if counts.shape != (template.n_pairs,) or diag_counts.shape != (template.n_loci,):
        raise ValueError("synthetic count arrays must match the template full grid")
    if exposure.shape != (template.n_loci,):
        raise ValueError("synthetic exposure must match the template full grid")
    if (not np.all(np.isfinite(counts)) or not np.all(np.isfinite(diag_counts))
            or np.any(counts < 0.0) or np.any(diag_counts < 0.0)):
        raise ValueError("synthetic expected counts must be finite and nonnegative")
    if not np.all(np.isfinite(exposure)) or not np.all(exposure > 0.0):
        raise ValueError("synthetic exposure must be finite and positive")
    if not np.isclose(float(exposure.mean()), 1.0, rtol=0.0, atol=1e-14):
        raise ValueError("synthetic exposure must already have full-grid mean one")

    actual = {
        "diag": float(diag_counts.sum()),
        "cis_offdiag": float(counts[template.cis_pair].sum()),
        "inter": float(counts[~template.cis_pair].sum()),
    }
    if expected_group_totals is None:
        expected = actual
    else:
        expected = {str(key): float(value) for key, value in expected_group_totals.items()}
        required = set(actual)
        if set(expected) != required:
            raise ValueError("expected_group_totals must contain exactly diag, cis_offdiag, inter")
        for group, observed in actual.items():
            if not math.isfinite(expected[group]) or expected[group] < 0.0:
                raise ValueError("expected group totals must be finite and nonnegative")
            if not np.isclose(observed, expected[group], rtol=rtol, atol=atol):
                raise ValueError(
                    "synthetic expected %s mass %.17g does not match supplied %.17g"
                    % (group, observed, expected[group]))
    if endpoint_counts is None:
        endpoint_counts = template.endpoint_counts.copy()
    else:
        endpoint_counts = np.asarray(endpoint_counts, dtype=np.float64)
        if endpoint_counts.shape != (template.n_loci,):
            raise ValueError("synthetic endpoint counts must match the template full grid")
        if not np.all(np.isfinite(endpoint_counts)) or np.any(endpoint_counts < 0.0):
            raise ValueError("synthetic endpoint counts must be finite and nonnegative")

    result = AggregatedContacts(
        chromosome_names=template.chromosome_names,
        chromosome_lengths=template.chromosome_lengths.copy(),
        bin_size=template.bin_size,
        n_bins=template.n_bins.copy(),
        offsets=template.offsets.copy(),
        locus_chromosome=template.locus_chromosome.copy(),
        locus_bin=template.locus_bin.copy(),
        endpoint_counts=np.asarray(endpoint_counts).copy(),
        exposure=exposure.copy(),
        pair_i=template.pair_i.copy(),
        pair_j=template.pair_j.copy(),
        cis_pair=template.cis_pair.copy(),
        counts=counts.copy(),
        diag_counts=diag_counts.copy(),
        raw_records=expected["diag"] + expected["cis_offdiag"] + expected["inter"],
        raw_same_bin=expected["diag"],
        raw_cis_offdiag=expected["cis_offdiag"],
        raw_inter=expected["inter"],
        conditional_factorial_constant=float("nan"),
        diag_factorial_sum=float("nan"),
        count_mode="synthetic_expected",
        exposure_mode=str(exposure_mode),
        expected_rtol=float(rtol),
        expected_atol=float(atol),
    )
    result.assert_consistent()
    return result


def endpoint_counts_from_aggregates(template: AggregatedContacts, counts: np.ndarray,
                                    diag_counts: np.ndarray) -> np.ndarray:
    """Derive two-endpoint mass per full-grid locus from aggregate counts."""
    counts = np.asarray(counts)
    diag_counts = np.asarray(diag_counts)
    if counts.shape != (template.n_pairs,) or diag_counts.shape != (template.n_loci,):
        raise ValueError("aggregate arrays must match the template full grid")
    dtype = np.result_type(counts.dtype, diag_counts.dtype, np.int64)
    endpoint_counts = (2 * diag_counts).astype(dtype, copy=True)
    np.add.at(endpoint_counts, template.pair_i, counts)
    np.add.at(endpoint_counts, template.pair_j, counts)
    return endpoint_counts


def synthetic_integer_clone(template: AggregatedContacts, counts: np.ndarray,
                            diag_counts: np.ndarray, exposure: np.ndarray,
                            group_totals: Mapping[str, int] | None = None,
                            exposure_mode: str = "synthetic_known",
                            endpoint_counts: np.ndarray | None = None) -> AggregatedContacts:
    """Build an explicitly integer synthetic dataset on existing full-grid metadata.

    Unlike :func:`synthetic_expected_clone`, this represents conditional
    multinomial integer draws. Its count-factorial constants are mathematically
    defined and retained for audit, while the optimizer still omits the
    candidate-independent conditional constants.
    """
    template.assert_consistent()
    if not exposure_mode:
        raise ValueError("synthetic integer exposure_mode must be explicit")
    raw_counts = np.asarray(counts)
    raw_diag = np.asarray(diag_counts)
    if raw_counts.shape != (template.n_pairs,) or raw_diag.shape != (template.n_loci,):
        raise ValueError("synthetic integer arrays must match the template full grid")
    if (not np.issubdtype(raw_counts.dtype, np.integer)
            or not np.issubdtype(raw_diag.dtype, np.integer)):
        raise ValueError("synthetic integer counts must use an integer dtype")
    counts = raw_counts.astype(np.int64, copy=False)
    diag_counts = raw_diag.astype(np.int64, copy=False)
    if np.any(counts < 0) or np.any(diag_counts < 0):
        raise ValueError("synthetic integer counts must be nonnegative")
    exposure = np.asarray(exposure, dtype=np.float64)
    if exposure.shape != (template.n_loci,):
        raise ValueError("synthetic exposure must match the template full grid")
    if not np.all(np.isfinite(exposure)) or not np.all(exposure > 0.0):
        raise ValueError("synthetic exposure must be finite and positive")
    if not np.isclose(float(exposure.mean()), 1.0, rtol=0.0, atol=1e-14):
        raise ValueError("synthetic exposure must already have full-grid mean one")

    actual = {
        "diag": int(diag_counts.sum()),
        "cis_offdiag": int(counts[template.cis_pair].sum()),
        "inter": int(counts[~template.cis_pair].sum()),
    }
    if group_totals is None:
        totals = actual
    else:
        totals = {str(key): value for key, value in group_totals.items()}
        if set(totals) != set(actual):
            raise ValueError("group_totals must contain exactly diag, cis_offdiag, inter")
        for group, observed in actual.items():
            value = totals[group]
            if not isinstance(value, (int, np.integer)) or int(value) < 0:
                raise ValueError("synthetic integer group totals must be nonnegative integers")
            if observed != int(value):
                raise ValueError(
                    "synthetic integer %s total %d does not match supplied %d"
                    % (group, observed, int(value)))
        totals = {group: int(value) for group, value in totals.items()}

    derived_endpoints = endpoint_counts_from_aggregates(template, counts, diag_counts)
    if endpoint_counts is None:
        endpoint_counts = derived_endpoints
    else:
        endpoint_counts = np.asarray(endpoint_counts)
        if endpoint_counts.shape != (template.n_loci,):
            raise ValueError("synthetic endpoint counts must match the template full grid")
        if not np.issubdtype(endpoint_counts.dtype, np.integer):
            raise ValueError("synthetic integer endpoint counts must use an integer dtype")
        endpoint_counts = endpoint_counts.astype(np.int64, copy=False)
        if not np.array_equal(endpoint_counts, derived_endpoints):
            raise ValueError("synthetic endpoint counts must equal aggregate-derived endpoints")

    conditional_constant = 0.0
    for mask, total in ((template.cis_pair, totals["cis_offdiag"]),
                        (~template.cis_pair, totals["inter"])):
        if total:
            conditional_constant += -float(gammaln(float(total) + 1.0))
            conditional_constant += _positive_log_factorial_sum(counts[mask])
    result = AggregatedContacts(
        chromosome_names=template.chromosome_names,
        chromosome_lengths=template.chromosome_lengths.copy(),
        bin_size=template.bin_size,
        n_bins=template.n_bins.copy(),
        offsets=template.offsets.copy(),
        locus_chromosome=template.locus_chromosome.copy(),
        locus_bin=template.locus_bin.copy(),
        endpoint_counts=np.asarray(endpoint_counts, dtype=np.int64).copy(),
        exposure=exposure.copy(),
        pair_i=template.pair_i.copy(),
        pair_j=template.pair_j.copy(),
        cis_pair=template.cis_pair.copy(),
        counts=counts.copy(),
        diag_counts=diag_counts.copy(),
        raw_records=totals["diag"] + totals["cis_offdiag"] + totals["inter"],
        raw_same_bin=totals["diag"],
        raw_cis_offdiag=totals["cis_offdiag"],
        raw_inter=totals["inter"],
        conditional_factorial_constant=conditional_constant,
        diag_factorial_sum=_positive_log_factorial_sum(diag_counts),
        count_mode="synthetic_integer",
        exposure_mode=str(exposure_mode),
        expected_rtol=0.0,
        expected_atol=0.0,
    )
    result.assert_consistent()
    return result


def load_aggregate(path: str | Path = SNPFREE, bin_size: int = 1_000_000,
                   verify_frozen_hash: bool = False) -> AggregatedContacts:
    """Load a SNP-free pairs file and aggregate every record without a fold split."""
    path = str(path)
    if verify_frozen_hash:
        verify_frozen_snpfree(path)
    lengths = genome.chrom_lengths(path)
    contacts = genome.load_all(path)
    names = tuple(name for name, _ in lengths)
    if tuple(contacts["names"]) != names:
        raise RuntimeError("header order changed while loading SNP-free contacts")
    result = aggregate_from_arrays(
        names,
        [length for _, length in lengths],
        contacts["ci"], contacts["p1"], contacts["cj"], contacts["p2"],
        bin_size,
    )
    return result


def load_frozen_p9016_aggregate(bin_size: int) -> AggregatedContacts:
    """Load the one authorized real input and assert its frozen raw budget."""
    result = load_aggregate(SNPFREE, bin_size=bin_size, verify_frozen_hash=True)
    audit = result.budget()
    expected = {
        "raw_records": FROZEN_P9016_RECORDS,
        "raw_cis_offdiag": FROZEN_P9016_CIS_RECORDS - audit["raw_same_bin"],
        "raw_inter": FROZEN_P9016_INTER_RECORDS,
    }
    if audit["raw_records"] != expected["raw_records"]:
        raise RuntimeError("frozen P9016 record count mismatch")
    if audit["raw_inter"] != expected["raw_inter"]:
        raise RuntimeError("frozen P9016 inter count mismatch")
    if audit["raw_same_bin"] + audit["raw_cis_offdiag"] != FROZEN_P9016_CIS_RECORDS:
        raise RuntimeError("frozen P9016 cis count mismatch")
    if audit["n_chromosomes"] != 20:
        raise RuntimeError("frozen P9016 must retain all 20 chromosome headers")
    return result


def sphere_forward(y: np.ndarray, return_inverse_radius: bool = False):
    """Map unconstrained vectors strictly inside the unit ball."""
    y = np.asarray(y, dtype=np.float64)
    if y.shape[-1] != 3:
        raise ValueError("sphere coordinates must end in three dimensions")
    if not np.all(np.isfinite(y)):
        raise ValueError("unconstrained coordinates must be finite")
    radius = np.sqrt(1.0 + np.sum(y * y, axis=-1, keepdims=True))
    x = y / radius
    if return_inverse_radius:
        return x, 1.0 / radius
    return x


def sphere_pullback(y: np.ndarray, gradient_x: np.ndarray) -> np.ndarray:
    """Analytic Jacobian-vector product for :func:`sphere_forward`."""
    y = np.asarray(y, dtype=np.float64)
    gradient_x = np.asarray(gradient_x, dtype=np.float64)
    if y.shape != gradient_x.shape or y.shape[-1] != 3:
        raise ValueError("sphere pullback requires matching (..., 3) arrays")
    x, inverse_radius = sphere_forward(y, return_inverse_radius=True)
    return inverse_radius * (gradient_x - x * np.sum(x * gradient_x, axis=-1, keepdims=True))


def sphere_inverse(x: np.ndarray) -> np.ndarray:
    """Map strictly interior coordinates back to unconstrained variables."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != 3:
        raise ValueError("sphere coordinates must end in three dimensions")
    norm2 = np.sum(x * x, axis=-1, keepdims=True)
    if np.any(norm2 >= 1.0) or not np.all(np.isfinite(norm2)):
        raise ValueError("sphere inverse requires coordinates strictly inside the unit ball")
    return x / np.sqrt(1.0 - norm2)


def assert_inside_unit_ball(x: np.ndarray) -> None:
    norms = np.sqrt(np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=-1))
    if not np.all(np.isfinite(norms)) or np.any(norms >= 1.0):
        raise AssertionError("all physical beads must lie strictly inside the unit ball")


def bounded_kernel(delta: np.ndarray, r0: float, epsilon: float = EPSILON,
                   with_gradient: bool = False):
    """Finite kernel and, optionally, derivative with respect to ``delta``."""
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape[-1] != 3 or r0 <= 0.0:
        raise ValueError("kernel requires (..., 3) deltas and positive r0")
    scaled = np.sum(delta * delta, axis=-1) / (r0 * r0)
    base = 1.0 + scaled
    kernel = epsilon + (1.0 - epsilon) * base ** -2
    if not with_gradient:
        return kernel
    gradient = (-4.0 * (1.0 - epsilon) / (r0 * r0)
                * delta * base[..., None] ** -3)
    return kernel, gradient


def _stable_logistic(q: float) -> float:
    if q >= 0.0:
        z = math.exp(-q)
        return 1.0 / (1.0 + z)
    z = math.exp(q)
    return z / (1.0 + z)


def p_from_q(q: float) -> tuple[float, float]:
    """Bounded cis mixture value and analytic derivative with respect to q."""
    s = _stable_logistic(float(q))
    span = 1.0 - 2.0 * P_FLOOR
    return P_FLOOR + span * s, span * s * (1.0 - s)


def q_from_p(p: float) -> float:
    """Inverse bounded-logistic parameterization for a valid initial p."""
    z = (float(p) - P_FLOOR) / (1.0 - 2.0 * P_FLOOR)
    if not 0.0 < z < 1.0:
        raise ValueError("initial p must lie strictly within the bounded interval")
    return float(math.log(z / (1.0 - z)))


class JointObjective:
    """Analytic V1 full-grid observation objective and polymer prior."""

    def __init__(self, data: AggregatedContacts, weights: ObjectiveWeights | None = None,
                 block_size: int = DEFAULT_BLOCK_SIZE,
                 repulsion_block_size: int | None = None):
        data.assert_consistent()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if repulsion_block_size is None:
            repulsion_block_size = block_size
        if repulsion_block_size <= 0:
            raise ValueError("repulsion_block_size must be positive")
        self.data = data
        self.weights = weights if weights is not None else ObjectiveWeights()
        self.block_size = int(block_size)
        self.repulsion_block_size = int(repulsion_block_size)
        # Keep the exact optimizer theta object; callbacks can reuse this result
        # only after an exact elementwise comparison, never an allclose match.
        self._last_theta = None
        self._last_value = None
        self._last_components = None

    @property
    def n_parameters(self) -> int:
        return 6 * self.data.n_loci + 1

    def pack(self, y: np.ndarray, p: float = 0.75) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (2, self.data.n_loci, 3):
            raise ValueError("y must have shape (2, n_loci, 3)")
        if not np.all(np.isfinite(y)):
            raise ValueError("initial y must be finite")
        return np.concatenate((y.ravel(), np.array([q_from_p(p)], dtype=np.float64)))

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 1 or len(theta) != self.n_parameters:
            raise ValueError("theta has %d parameters, expected %d" %
                             (theta.size, self.n_parameters))
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must be finite")
        y = theta[:-1].reshape(2, self.data.n_loci, 3)
        return y, float(theta[-1])

    def coordinates_and_p(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        y, q = self.unpack(theta)
        x = sphere_forward(y)
        assert_inside_unit_ball(x)
        p, _ = p_from_q(q)
        return x, p

    def _rate_block(self, x: np.ndarray, i: np.ndarray, j: np.ndarray,
                    cis: np.ndarray, p: float, with_gradient: bool = False):
        exposure = self.data.exposure[i] * self.data.exposure[j]
        aa = x[0, i] - x[0, j]
        ab = x[0, i] - x[1, j]
        ba = x[1, i] - x[0, j]
        bb = x[1, i] - x[1, j]
        if with_gradient:
            kaa, gaa = bounded_kernel(aa, self.data.r0, with_gradient=True)
            kab, gab = bounded_kernel(ab, self.data.r0, with_gradient=True)
            kba, gba = bounded_kernel(ba, self.data.r0, with_gradient=True)
            kbb, gbb = bounded_kernel(bb, self.data.r0, with_gradient=True)
        else:
            kaa = bounded_kernel(aa, self.data.r0)
            kab = bounded_kernel(ab, self.data.r0)
            kba = bounded_kernel(ba, self.data.r0)
            kbb = bounded_kernel(bb, self.data.r0)
        same = kaa + kbb
        cross = kab + kba
        mixture = np.where(cis, 0.5 * (p * same + (1.0 - p) * cross),
                           0.25 * (same + cross))
        rate = exposure * mixture
        if not with_gradient:
            return rate
        return rate, (kaa, kab, kba, kbb), (gaa, gab, gba, gbb), exposure

    def rates_for_pair_indices(self, x: np.ndarray, p: float,
                               indices: np.ndarray | None = None) -> np.ndarray:
        """Return rates for selected full-grid pair rows; intended for audit/tests."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (2, self.data.n_loci, 3):
            raise ValueError("x must have shape (2, n_loci, 3)")
        if indices is None:
            indices = np.arange(self.data.n_pairs, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.data.n_pairs):
            raise ValueError("pair index outside full eligible grid")
        return self._rate_block(x, self.data.pair_i[indices], self.data.pair_j[indices],
                                self.data.cis_pair[indices], float(p), with_gradient=False)

    def _diag_nll(self) -> float:
        """Candidate-independent same-bin profile; fractional mode is cross-entropy."""
        positive = self.data.diag_counts[self.data.diag_counts > 0].astype(np.float64)
        if len(positive) == 0:
            return 0.0
        value = positive - positive * np.log(positive)
        if self.data.count_mode in ("raw_integer", "synthetic_integer"):
            value = value + gammaln(positive + 1.0)
        return float(np.sum(value))

    def _count_nll_and_gradient(self, x: np.ndarray, p: float,
                                need_gradient: bool):
        data = self.data
        sums = np.zeros(2, dtype=np.float64)
        observed_log = np.zeros(2, dtype=np.float64)
        group_total = np.array([data.raw_cis_offdiag, data.raw_inter], dtype=np.float64)

        # First pass: full eligible-grid normalization and observed log-rate term.
        for start in range(0, data.n_pairs, self.block_size):
            stop = min(start + self.block_size, data.n_pairs)
            cis = data.cis_pair[start:stop]
            rate = self._rate_block(x, data.pair_i[start:stop], data.pair_j[start:stop],
                                    cis, p, with_gradient=False)
            count = data.counts[start:stop]
            for group, mask in ((0, cis), (1, ~cis)):
                if mask.any():
                    sums[group] += float(rate[mask].sum())
                    observed_log[group] += float(np.dot(
                        count[mask].astype(np.float64), np.log(rate[mask])))
        if np.any((group_total > 0.0) & (sums <= 0.0)):
            raise FloatingPointError("positive contact group has nonpositive full-grid rate sum")
        conditional = 0.0
        for group in (0, 1):
            if group_total[group] > 0.0:
                conditional += group_total[group] * math.log(sums[group]) - observed_log[group]
        diag = self._diag_nll()
        components = {
            "count_mode": data.count_mode,
            "count_factorial_status": (
                "integer_count_constants_recorded"
                if data.count_mode in ("raw_integer", "synthetic_integer")
                else "not_defined_for_fractional_expected_cross_entropy"),
            "sum_rate_cis_offdiag": float(sums[0]),
            "sum_rate_inter": float(sums[1]),
            "observed_log_rate_cis_offdiag": float(observed_log[0]),
            "observed_log_rate_inter": float(observed_log[1]),
            "conditional_nll_raw": float(conditional),
            "diag_profiled_nll_raw": float(diag),
            "count_nll_raw": float(conditional + diag),
            "count_nll_normalized": float((conditional + diag) / data.raw_records),
        }
        if not need_gradient:
            return components, np.zeros_like(x), 0.0

        gradient_x = np.zeros_like(x)
        gradient_p = 0.0
        # Second pass: exact derivative of the same complete full-grid sums.
        for start in range(0, data.n_pairs, self.block_size):
            stop = min(start + self.block_size, data.n_pairs)
            i = data.pair_i[start:stop]
            j = data.pair_j[start:stop]
            cis = data.cis_pair[start:stop]
            count = data.counts[start:stop].astype(np.float64)
            rate, kernels, kernel_gradients, exposure = self._rate_block(
                x, i, j, cis, p, with_gradient=True)
            weights = -count / rate
            if cis.any() and group_total[0] > 0.0:
                weights[cis] += group_total[0] / sums[0]
            inter = ~cis
            if inter.any() and group_total[1] > 0.0:
                weights[inter] += group_total[1] / sums[1]

            kaa, kab, kba, kbb = kernels
            gaa, gab, gba, gbb = kernel_gradients
            same_coefficient = exposure * np.where(cis, 0.5 * p, 0.25) * weights
            cross_coefficient = exposure * np.where(cis, 0.5 * (1.0 - p), 0.25) * weights
            for copy_i, copy_j, gradient_kernel, coefficient in (
                    (0, 0, gaa, same_coefficient),
                    (0, 1, gab, cross_coefficient),
                    (1, 0, gba, cross_coefficient),
                    (1, 1, gbb, same_coefficient)):
                term = coefficient[:, None] * gradient_kernel
                np.add.at(gradient_x[copy_i], i, term)
                np.add.at(gradient_x[copy_j], j, -term)
            if cis.any():
                derivative_p = 0.5 * exposure * (kaa + kbb - kab - kba)
                gradient_p += float(np.dot(weights[cis], derivative_p[cis]))

        normalizer = float(data.raw_records)
        return components, gradient_x / normalizer, gradient_p / normalizer

    def _bond_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        gradient = np.zeros_like(x)
        value = 0.0
        n_term = 0
        l0 = self.data.l0
        for chromosome_index in range(len(self.data.chromosome_names)):
            slc = self.data.chromosome_slice(chromosome_index)
            n = slc.stop - slc.start
            if n < 2:
                continue
            for copy_index in (0, 1):
                left = x[copy_index, slc.start:slc.stop - 1]
                right = x[copy_index, slc.start + 1:slc.stop]
                delta = right - left
                distance = np.sqrt(np.sum(delta * delta, axis=1))
                scaled = distance / l0
                low = np.maximum(0.0, 0.75 - scaled)
                high = np.maximum(0.0, scaled - 1.25)
                value += float(np.sum(low * low + high * high))
                derivative_distance = (-2.0 * low + 2.0 * high) / l0
                coefficient = np.divide(
                    derivative_distance, distance, out=np.zeros_like(distance), where=distance > 0.0)
                term = coefficient[:, None] * delta
                gradient[copy_index, slc.start + 1:slc.stop] += term
                gradient[copy_index, slc.start:slc.stop - 1] -= term
                n_term += len(distance)
        if n_term == 0:
            return 0.0, gradient
        return value / n_term, gradient / n_term

    @staticmethod
    def _repulsion_piece(delta: np.ndarray, threshold: float):
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        hinge = np.maximum(0.0, 1.0 - distance / threshold)
        derivative_distance = -2.0 * hinge / threshold
        coefficient = np.divide(
            derivative_distance, distance, out=np.zeros_like(distance), where=distance > 0.0)
        return float(np.sum(hinge * hinge)), coefficient[..., None] * delta

    def _repulsion_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        data = self.data
        gradient = np.zeros_like(x)
        value = 0.0
        threshold = 0.7 * data.l0
        for start in range(0, data.n_pairs, self.repulsion_block_size):
            stop = min(start + self.repulsion_block_size, data.n_pairs)
            i = data.pair_i[start:stop]
            j = data.pair_j[start:stop]
            for copy_i, copy_j in ((0, 0), (0, 1), (1, 0), (1, 1)):
                delta = x[copy_i, i] - x[copy_j, j]
                piece_value, term = self._repulsion_piece(delta, threshold)
                value += piece_value
                np.add.at(gradient[copy_i], i, term)
                np.add.at(gradient[copy_j], j, -term)
        homolog_value, homolog_term = self._repulsion_piece(x[0] - x[1], threshold)
        value += homolog_value
        gradient[0] += homolog_term
        gradient[1] -= homolog_term
        normalizer = float(2 * data.n_loci)
        return value / normalizer, gradient / normalizer

    def _bend_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        gradient = np.zeros_like(x)
        value = 0.0
        n_term = 0
        l0sq = self.data.l0 ** 2
        for chromosome_index in range(len(self.data.chromosome_names)):
            slc = self.data.chromosome_slice(chromosome_index)
            n = slc.stop - slc.start
            if n < 3:
                continue
            for copy_index in (0, 1):
                second = (x[copy_index, slc.start + 2:slc.stop]
                          - 2.0 * x[copy_index, slc.start + 1:slc.stop - 1]
                          + x[copy_index, slc.start:slc.stop - 2])
                value += float(np.sum(second * second) / l0sq)
                term = 2.0 * second / l0sq
                gradient[copy_index, slc.start:slc.stop - 2] += term
                gradient[copy_index, slc.start + 1:slc.stop - 1] -= 2.0 * term
                gradient[copy_index, slc.start + 2:slc.stop] += term
                n_term += len(second)
        if n_term == 0:
            return 0.0, gradient
        return value / n_term, gradient / n_term

    def evaluate(self, theta: np.ndarray, need_gradient: bool = True):
        """Return total, analytic gradient, and separately auditable components."""
        y, q = self.unpack(theta)
        x = sphere_forward(y)
        assert_inside_unit_ball(x)
        p, derivative_q = p_from_q(q)
        count_components, gradient_x, gradient_p = self._count_nll_and_gradient(
            x, p, need_gradient)
        bond, bond_gradient = self._bond_and_gradient(x)
        repulsion, repulsion_gradient = self._repulsion_and_gradient(x)
        bend, bend_gradient = self._bend_and_gradient(x)
        p_prior = -P_PRIOR_STRENGTH * math.log(p * (1.0 - p))
        p_prior_derivative_p = P_PRIOR_STRENGTH * (1.0 / (1.0 - p) - 1.0 / p)
        components = dict(count_components)
        components.update({
            "p": float(p),
            "p_prior": float(p_prior),
            "bond": float(bond),
            "repulsion": float(repulsion),
            "bend": float(bend),
        })
        total = (self.weights.count * components["count_nll_normalized"]
                 + self.weights.p_prior * p_prior
                 + self.weights.bond * bond
                 + self.weights.repulsion * repulsion
                 + self.weights.bend * bend)
        components["total"] = float(total)
        if not need_gradient:
            return float(total), None, components
        combined_x = (self.weights.count * gradient_x
                      + self.weights.bond * bond_gradient
                      + self.weights.repulsion * repulsion_gradient
                      + self.weights.bend * bend_gradient)
        gradient_y = sphere_pullback(y, combined_x)
        gradient_q = derivative_q * (
            self.weights.count * gradient_p
            + self.weights.p_prior * p_prior_derivative_p)
        gradient = np.concatenate((gradient_y.ravel(), np.array([gradient_q], dtype=np.float64)))
        if not np.all(np.isfinite(gradient)) or not math.isfinite(total):
            raise FloatingPointError("nonfinite V1 objective or analytic gradient")
        return float(total), gradient, components

    def count_components_for_coordinates(self, x: np.ndarray, p: float) -> dict:
        """Evaluate only the label-free count layer for fixed physical coordinates."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (2, self.data.n_loci, 3):
            raise ValueError("x must have shape (2, n_loci, 3)")
        assert_inside_unit_ball(x)
        if not math.isfinite(float(p)) or not P_FLOOR <= float(p) <= 1.0 - P_FLOOR:
            raise ValueError("fixed p must lie in the bounded V1 interval")
        components, _, _ = self._count_nll_and_gradient(x, float(p), need_gradient=False)
        return components

    def cached_value_and_components(self, theta: np.ndarray):
        """Return the exact latest value/components when ``theta`` is unchanged.

        The cache intentionally retains the original float64 theta object instead
        of a rounded log representation. A caller only receives it after exact
        elementwise equality, so an updated iterate cannot be mistaken for the
        one that supplied the analytic gradient.
        """
        if self._last_theta is None:
            return None
        candidate = np.asarray(theta, dtype=np.float64)
        if candidate.shape != self._last_theta.shape:
            return None
        if not np.array_equal(candidate, self._last_theta):
            return None
        return self._last_value, self._last_components

    def value_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, components = self.evaluate(theta, need_gradient=True)
        # Do not copy theta: SciPy's callback returns the exact current iterate,
        # and matching is subsequently exact rather than tolerance-based.
        self._last_theta = np.asarray(theta, dtype=np.float64)
        self._last_value = value
        self._last_components = components
        return value, gradient

    def components(self, theta: np.ndarray) -> dict:
        cached = self.cached_value_and_components(theta)
        if cached is not None:
            return cached[1]
        _, _, components = self.evaluate(theta, need_gradient=False)
        return components


def tracks_from_coordinates(data: AggregatedContacts, x: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
    """Return all 40/header-order tracks with every full-grid bin represented."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (2, data.n_loci, 3):
        raise ValueError("x must have shape (2, n_loci, 3)")
    assert_inside_unit_ball(x)
    result: dict[str, dict[int, np.ndarray]] = {}
    for spec in data.track_specs:
        slc = data.chromosome_slice(spec.chromosome_index)
        result[spec.name] = {
            int(local_bin * data.bin_size): x[spec.copy_index, global_index].copy()
            for global_index, local_bin in zip(range(slc.start, slc.stop),
                                                data.locus_bin[slc], strict=True)
        }
    return result


def write_full_tracks(path: str | Path, data: AggregatedContacts, x: np.ndarray) -> None:
    """Write every copy and full-grid bin in explicit SNP-free-header track order."""
    tracks = tracks_from_coordinates(data, x)
    with open(path, "wt") as handle:
        for spec in data.track_specs:
            for position, xyz in tracks[spec.name].items():
                handle.write("%s\t%d\t%.17g\t%.17g\t%.17g\n" %
                             (spec.name, position, xyz[0], xyz[1], xyz[2]))
