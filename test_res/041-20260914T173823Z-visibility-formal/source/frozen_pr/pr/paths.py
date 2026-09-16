"""项目常量；本模块不会读取数据。"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAIRS = os.path.join(ROOT, "data", "P9016.pairs.gz")
REF3DG = os.path.join(ROOT, "data", "P9016.1m.3dg.gz")
SNPFREE = os.path.join(ROOT, "inputs", "P9016.snpfree.pairs.gz")
HICKIT = os.path.join(ROOT, "native", "hickit", "hickit")

BIN = 1_000_000
OFF = 3_000_000            # 每条染色体的第一个非空 bin
FRAG_BINS = 20             # Stage 1 片段大小：20 个 bin，即 20 Mb

CHRS = ["chr%d" % k for k in range(1, 20)] + ["chrX"]

# 七列：去除 phase0/phase1 后的完整 pairs 记录。
COL7 = ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"]

# mixed hash 产生的 fold id：train = 0..5，val = 6..7，test = 8..9。
N_FOLD = 10
TRAIN_FOLDS = (0, 1, 2, 3, 4, 5)
VAL_FOLDS = (6, 7)
TEST_FOLDS = (8, 9)

# 只在 VALIDATION fold 上筛选 distance exponent（绝不使用 test）。
ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


def bin_of(pos):
    """将基因组位置转为 bin 索引；要求 pos >= OFF。"""
    return (pos - OFF) // BIN


def pos_of(b):
    """将 bin 索引转为该 bin 的起始位置。"""
    return OFF + b * BIN


def safe_name(chrom, idx):
    """返回 `chrom` 的第 `idx` 个 copy 所用的 FDG track name。

    hickit 不能在同一文件中同时使用 chromosome 名 `chr1`（haploid consensus）和 paired copy，因此 two-copy runs 使用专用名称。
    """
    return "cc%02d%s" % (CHRS.index(chrom), "ab"[idx])
