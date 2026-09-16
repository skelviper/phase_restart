"""052 轮共享路径与常量（只读复用冻结实现，不改旧目录任何字节）。"""
from __future__ import annotations

from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round"
SOURCE_045_SRC = S045 / "source"
RUN_014 = ROOT / "test_res/014-20260912_153000-s0-genome-wide-fixed"

AGGREGATE_1MB = S045 / "inputs/real_1000000_aggregate.npz"
AGGREGATE_1MB_SHA256 = "80984d804f8ae0552f6bab34a77a03e778073ac87f3b3ec6f04f96de74137420"
AGGREGATE_051 = {  # 051 已冻结的粗层 aggregate（20/10/5/2Mb），只读
    b: ROOT / ("test_res/051-20260916T064500-g-random-baseline-ext/inputs/real_%d_aggregate.npz" % b)
    for b in (20_000_000, 10_000_000, 5_000_000, 2_000_000)
}
SNPFREE = ROOT / "inputs/P9016.snpfree.pairs.gz"
SNPFREE_SHA256 = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"

# 冻结对照
BASELINE_NPZ = ROOT / "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.npz"
BASELINE_NPZ_SHA256 = "6bb93bf570cf688fd8b02dccf14ca0867c1f831d5e123036ced626856f8d8752"
BASELINE_3DG = ROOT / "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.3dg"
BASELINE_3DG_SHA256 = "ee5eb1545db9bfeeabcc24e704707f5f61a5793e6f245091347373442dc0032b"
EXTRA_NPZ = ROOT / "test_res/051-20260916T064500-g-random-baseline-ext/coords/A-extra-levels/1Mb.npz"
EXTRA_NPZ_SHA256 = "60711facfa41e60b7d5283e18f5aa221e227561d007c4a82729c06a36c02757a"
EXTRA_3DG = ROOT / "test_res/051-20260916T064500-g-random-baseline-ext/coords/A-extra-levels/1Mb.3dg"
EXTRA_3DG_SHA256 = "d8f779e7c85271fca2df89512645b2ddcd9d79134839e6b62e656cc1b74aa29f"

MASK_SNAPSHOT = ROOT / "test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/frozen_legacy_mask_snapshot.npz"
MASK_SNAPSHOT_SHA256 = "9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9"
REFERENCE_PATH = ROOT / "data/P9016.1m.3dg.gz"
REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"

# 链定义：stage 名字 -> 整数 bp（绝不从名字推断 bin size）
STAGES = ("40Mb", "10Mb", "5Mb", "2Mb", "1Mb", "500kb", "200kb")
STAGE_BIN = {"40Mb": 40_000_000, "10Mb": 10_000_000, "5Mb": 5_000_000, "2Mb": 2_000_000,
             "1Mb": 1_000_000, "500kb": 500_000, "200kb": 200_000}
STAGE_FG_CAP = {"40Mb": 200, "10Mb": 200, "5Mb": 612, "2Mb": 404, "1Mb": 486,
                "500kb": 200, "200kb": 100}
TOTAL_FG_CAP = 2202
NEWLY_AGGREGATED_BINS = (500_000, 200_000)
REUSED_051_BINS = (20_000_000, 10_000_000, 5_000_000, 2_000_000)

FIT_ID = "new-chain"
WEIGHTS = {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}
P_INIT = 0.75
BASE_SEED = 2207

# 评价冻结常量
EVAL_OFFSET_BP = 3_000_000
EVAL_BIN_BP = 1_000_000
EVAL_CHROMOSOMES = 20
EVAL_VALID_LOCI = 2447
EVAL_INTER_PAIRS = 157_529
MIN_COMMON_PAIRS = 3
GEOMETRY_TIE_TOL = 1e-12

VERSIONS = ("Baseline", "Extra levels", "200 kb -> 1 Mb")


def stage_bin(stage: str) -> int:
    if stage not in STAGE_BIN:
        raise KeyError("unknown stage %r" % stage)
    return int(STAGE_BIN[stage])


def bin_stage_name(bin_size: int) -> str:
    for stage, value in STAGE_BIN.items():
        if int(value) == int(bin_size):
            return stage
    raise KeyError("no stage name for bin size %r" % bin_size)
