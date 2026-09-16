"""SNP-free pairs I/O: exactly seven columns, hard rejection of phase fields.

`run.py prepare` streams `data/P9016.pairs.gz` and writes the seven-column file
used by every training-side step. The loader refuses any file whose `#columns`
line mentions a phase field, so a phase-bearing file cannot be read by accident.
"""
import gzip
import os

from .paths import COL7, PAIRS, SNPFREE

FORBIDDEN = ("phase", "prob")


class PhaseLeakError(RuntimeError):
    pass


def _open(path, mode="rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def write_snpfree(src=PAIRS, dst=SNPFREE):
    """Strip phase0/phase1 (and any phase_prob* column) from the pairs file."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    header = []
    body_written = 0
    with _open(src) as fi, gzip.open(dst, "wt") as fo:
        keep = None
        for line in fi:
            if line.startswith("#"):
                if line.startswith("#columns:"):
                    names = line.rstrip("\n").split(":", 1)[1].strip().split("\t")
                    keep = [k for k, n in enumerate(names)
                            if not any(t in n.lower() for t in FORBIDDEN)]
                    kept = [names[k] for k in keep]
                    assert kept == COL7, (kept, COL7)
                    fo.write("#columns:" + "\t".join(kept) + "\n")
                else:
                    header.append(line)
                    fo.write(line)
                continue
            if keep is None:
                raise PhaseLeakError("no #columns line before the first record in %s" % src)
            c = line.rstrip("\n").split("\t")
            fo.write("\t".join(c[k] for k in keep) + "\n")
            body_written += 1
    return {"records": body_written, "columns": COL7}


def read_columns_line(path):
    with _open(path) as f:
        for line in f:
            if line.startswith("#columns:"):
                return line.rstrip("\n").split(":", 1)[1].strip().split("\t")
            if not line.startswith("#"):
                break
    raise PhaseLeakError("no #columns line in %s" % path)


def assert_snpfree(path):
    """Hard rejection: a phase-bearing file may never be opened here."""
    cols = read_columns_line(path)
    bad = [c for c in cols if any(t in c.lower() for t in FORBIDDEN)]
    if bad:
        raise PhaseLeakError(
            "refusing to load %s: #columns mentions %s" % (path, bad))
    if cols != COL7:
        raise PhaseLeakError("refusing to load %s: columns are %s, expected %s"
                             % (path, cols, COL7))
    return True


def canonical_cis(pos1, pos2):
    """Return numeric cis positions in ascending order and whether ends swapped.

    Callers that carry per-end metadata must swap that metadata when the returned
    flag is true.  Keeping this rule in one place prevents positional
    canonicalisation from silently detaching an end label from its coordinate.
    """
    a, b = int(pos1), int(pos2)
    return (b, a, True) if a > b else (a, b, False)


def iter_records(path):
    """Yield validated seven-column data rows from an SNP-free pairs file.

    Header validation alone is insufficient: an appended eighth column or a
    truncated data row would otherwise enter training through positional array
    indexing.  This is deliberately a strict reader used by both single- and
    genome-wide loaders.
    """
    assert_snpfree(path)
    with _open(path) as f:
        for line_no, line in enumerate(f, start=1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(COL7):
                raise PhaseLeakError(
                    "refusing to load %s: row %d has %d columns, expected %d"
                    % (path, line_no, len(fields), len(COL7)))
            if not fields[1] or not fields[3]:
                raise PhaseLeakError(
                    "refusing to load %s: row %d has an empty chromosome field"
                    % (path, line_no))
            try:
                p1, p2 = int(fields[2]), int(fields[4])
            except ValueError as exc:
                raise PhaseLeakError(
                    "refusing to load %s: row %d has non-integer positions"
                    % (path, line_no)) from exc
            if p1 < 0 or p2 < 0:
                raise PhaseLeakError(
                    "refusing to load %s: row %d has a negative position"
                    % (path, line_no))
            yield fields


def load(path=SNPFREE, chrom=None):
    """Load the SNP-free table. Returns (pos1, pos2) int64 arrays, pos1 <= pos2.

    There is no label column and none can be smuggled in: `assert_snpfree` runs
    first and the file has exactly seven columns.
    """
    p1, p2 = [], []
    for c in iter_records(path):
        if c[1] != c[3]:
            continue                           # cis only (F1: 98.2% of intra)
        if chrom is not None and c[1] != chrom:
            continue
        a, b, _ = canonical_cis(c[2], c[4])   # numeric, never string order
        p1.append(a)
        p2.append(b)
    import numpy as np
    return np.asarray(p1, dtype=np.int64), np.asarray(p2, dtype=np.int64)
