"""Reference 3DG reader -- evaluator side only, gate-guarded.

`data/P9016.1m.3dg.gz` is the target, never an input to any fit. The gate must
be armed (i.e. some stage's coordinates have been written and hashed) before
this module will open it.
"""
import gzip

import numpy as np

from .paths import BIN, OFF, REF3DG

STAGE = "eval"
# F3 / corrected header block: phase0 -> chrN(pat), phase1 -> chrN(mat).
LABEL_TRACK = {0: "pat", 1: "mat"}


def load_reference(gate, path=REF3DG):
    gate.require(STAGE)
    coords = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            a = line.split()
            if len(a) < 5:
                continue
            coords.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
    return coords


def track(coords, chrom, suffix):
    return coords["%s(%s)" % (chrom, suffix)]


def distance(coords, key, b1, b2):
    """Distance for arrays of bin indices; NaN when a bead is missing."""
    m = coords.get(key, {})
    out = np.full(len(b1), np.nan)
    for i, (x, y) in enumerate(zip(b1, b2)):
        va = m.get(int(OFF + x * BIN))
        vb = m.get(int(OFF + y * BIN))
        if va is None or vb is None:
            continue
        out[i] = float(np.linalg.norm(va - vb))
    return out


def reference_accuracy(coords, chrom, bins1, bins2, labels):
    """Fraction of cis contacts (same-bin excluded) whose true copy is nearer.

    The phase convention is frozen: phase0 -> pat and phase1 -> mat.  The
    reverse direction remains a named legacy diagnostic only; it may not choose
    the reported ceiling.  Ties count as 1/2.
    """
    keep = (bins1 != bins2) & (labels >= 0)
    b1, b2, lb = bins1[keep], bins2[keep], labels[keep]
    d_pat = distance(coords, "%s(pat)" % chrom, b1, b2)
    d_mat = distance(coords, "%s(mat)" % chrom, b1, b2)
    ok = np.isfinite(d_pat) & np.isfinite(d_mat)
    d_pat, d_mat, lb = d_pat[ok], d_mat[ok], lb[ok]
    n = len(lb)
    tie = d_pat == d_mat
    res = {"n": int(n), "n_samebin_excluded": int((bins1 == bins2).sum()),
           "tie_rate": float(tie.mean()) if n else float("nan")}
    own = np.where(lb == 0, d_pat, d_mat)
    other = np.where(lb == 0, d_mat, d_pat)
    fixed = float(((own < other).astype(float) + 0.5 * tie).mean()) if n else float("nan")
    inverse = float(((other < own).astype(float) + 0.5 * tie).mean()) if n else float("nan")
    res.update({"acc": fixed, "orientation": 0,
                "acc_fixed_phase0_pat_phase1_mat": fixed,
                "acc_legacy_truth_reversed": inverse,
                "acc_legacy_truth_max": max(fixed, inverse) if n else float("nan")})
    return res
