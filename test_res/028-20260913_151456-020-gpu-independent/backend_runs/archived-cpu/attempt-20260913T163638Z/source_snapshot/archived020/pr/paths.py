"""Project constants. Nothing here reads data."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAIRS = os.path.join(ROOT, "data", "P9016.pairs.gz")
REF3DG = os.path.join(ROOT, "data", "P9016.1m.3dg.gz")
SNPFREE = os.path.join(ROOT, "inputs", "P9016.snpfree.pairs.gz")
HICKIT = os.path.join(ROOT, "native", "hickit", "hickit")

BIN = 1_000_000
OFF = 3_000_000            # first non-empty bin on every chromosome
FRAG_BINS = 20             # Stage 1 fragment size: 20 bins = 20 Mb

CHRS = ["chr%d" % k for k in range(1, 20)] + ["chrX"]

# Seven columns: the full pairs record with phase0/phase1 removed.
COL7 = ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"]

# Fold ids from the mixed hash: train = 0..5, val = 6..7, test = 8..9.
N_FOLD = 10
TRAIN_FOLDS = (0, 1, 2, 3, 4, 5)
VAL_FOLDS = (6, 7)
TEST_FOLDS = (8, 9)

# Distance exponents screened on the VALIDATION fold only (never on test).
ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def bin_of(pos):
    """Genomic position -> bin index. Requires pos >= OFF."""
    return (pos - OFF) // BIN


def pos_of(b):
    """Bin index -> the bin's start position."""
    return OFF + b * BIN


def safe_name(chrom, idx):
    """FDG track name for copy `idx` of `chrom`.

    hickit cannot have a chromosome named both 'chr1' (haploid consensus) and a
    paired copy in the same file, so the two-copy runs use dedicated names.
    """
    return "cc%02d%s" % (CHRS.index(chrom), "ab"[idx])
