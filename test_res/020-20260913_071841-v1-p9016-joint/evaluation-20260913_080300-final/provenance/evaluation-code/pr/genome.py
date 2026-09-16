"""Genome-wide data layer: all 20 chromosomes, cis and inter contacts, 40 tracks.

Track naming: `c{1-based chromosome index:02d}{a|b}`, e.g. `c01a`, `c20b`.
The reference names the same objects `chr1(mat)` / `chr1(pat)`; the correspondence is a
gauge and is resolved by geometry, never assumed.
"""
import gzip
import os

import numpy as np

from .paths import BIN, OFF, SNPFREE
from . import pairs7

N_CHR = 20


def chrom_lengths(pairs_path):
    """[(name, length)] in file order, from the #chromosome header lines."""
    pairs7.assert_snpfree(pairs_path)
    out = []
    with gzip.open(pairs_path, "rt") as f:
        for line in f:
            if line.startswith("#chromosome:"):
                p = line.split()
                if len(p) != 3:
                    raise pairs7.PhaseLeakError(
                        "malformed #chromosome header in %s: %r" % (pairs_path, line.rstrip()))
                try:
                    length = int(p[2])
                except ValueError as exc:
                    raise pairs7.PhaseLeakError(
                        "non-integer chromosome length in %s: %r" % (pairs_path, line.rstrip())) from exc
                if not p[1] or length <= 0:
                    raise pairs7.PhaseLeakError(
                        "invalid chromosome header in %s: %r" % (pairs_path, line.rstrip()))
                out.append((p[1], length))
            elif not line.startswith("#"):
                break
    if not out:
        raise pairs7.PhaseLeakError("no #chromosome headers in %s" % pairs_path)
    if len({c for c, _ in out}) != len(out):
        raise pairs7.PhaseLeakError("duplicate chromosome headers in %s" % pairs_path)
    return out


def track(ci, k):
    """Track name for chromosome index `ci` (0-based) and copy `k` in {0,1}."""
    return "c%02d%s" % (ci + 1, "ab"[k])


def load_all(path=SNPFREE):
    """All cis and inter contacts as index arrays.

    Returns dict with
        ci, p1, cj, p2 : int32 arrays (chromosome index and position of each end)
        cis            : bool, both ends on the same chromosome
    Positions are put in file order (no reordering across chromosomes: the pairs file is
    sorted chr1-chr2-pos1-pos2, so a trans record is stored with chr1 <= chr2 already).
    """
    # `chrom_lengths` calls assert_snpfree, and iter_records below validates every
    # body row.  Keep both checks here: a phase-bearing header must fail before any
    # training-side record is interpreted.
    pairs7.assert_snpfree(path)
    names = [c for c, _ in chrom_lengths(path)]
    idx = {c: i for i, c in enumerate(names)}
    ci, p1, cj, p2 = [], [], [], []
    for row_no, c in enumerate(pairs7.iter_records(path), start=1):
        try:
            a, b = idx[c[1]], idx[c[3]]
        except KeyError as exc:
            raise pairs7.PhaseLeakError(
                "row %d references chromosome absent from #chromosome headers: %s"
                % (row_no, exc.args[0])) from exc
        x, y = int(c[2]), int(c[4])
        if a == b:
            x, y, _ = pairs7.canonical_cis(x, y)
        ci.append(a); p1.append(x); cj.append(b); p2.append(y)
    d = {"ci": np.asarray(ci, np.int32), "p1": np.asarray(p1, np.int64),
         "cj": np.asarray(cj, np.int32), "p2": np.asarray(p2, np.int64),
         "names": names}
    d["cis"] = d["ci"] == d["cj"]
    return d


def n_bins_per_chrom(lengths):
    return [int((L - OFF) // BIN) + 1 for _, L in lengths]


def bin_of(pos):
    return (np.asarray(pos) - OFF) // BIN


def global_bin(ci, pos, counts):
    """Global bin index across the concatenated chromosome bin axis."""
    off = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return (off[np.asarray(ci)] + bin_of(pos)).astype(np.int64)


def random_assignment(d, seed=0):
    """Per-contact copy index (cis: one copy for the molecule; inter: one per end)."""
    rng = np.random.default_rng(seed)
    n = len(d["ci"])
    k1 = rng.integers(0, 2, n)
    k2 = np.where(d["cis"], k1, rng.integers(0, 2, n))
    return k1.astype(np.int8), k2.astype(np.int8)
