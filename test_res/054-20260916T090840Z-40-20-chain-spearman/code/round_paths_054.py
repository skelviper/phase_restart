"""054 轮共享路径与冻结常量。

本轮只做一件事：跑唯一一条**正确**的共享多分辨率链

    40Mb(复用 052 已冻结端点) -> 20Mb -> 10Mb -> 5Mb -> 2Mb -> 1Mb  (->
    500kb -> 200kb -> 逐 bin 算术均值粗化回 1Mb)

旧 052 链缺 20Mb 层（40 -> 10），其 1Mb/200kb 端点不能回答"逐步增加分辨率"的问题；
053 已把它们当正确结果显示过一次。本轮不修改 046/051/052/053 任何字节：052 只提供
代码实现、聚合输入与 40Mb 前缀端点；046/051 只提供 1Mb 对照端点与冻结评价快照。

冻结的科学设置见 run 目录 config.json；本文件只放路径、常量和哈希。
"""
from __future__ import annotations

from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]

# 冻结实现来源（只读导入，不改字节）
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round"
SOURCE_045_SRC = S045 / "source"
RUN_014 = ROOT / "test_res/014-20260912_153000-s0-genome-wide-fixed"

RUN046 = ROOT / "test_res/046-UTC-real-cell-shared-capture"
RUN051 = ROOT / "test_res/051-20260916T064500-g-random-baseline-ext"
RUN052 = ROOT / "test_res/052-20260916T080116Z-200kb-coarsened-1mb-chain"

FIT_ID = "new-chain"
CHAIN_TAG = "40-20-chain"

# ------------------------------------------------------------------ 复用输入
# 全部为已冻结 aggregate，只读复用；本 run 不重建、不跑 preflight/smoke。
REUSED_AGGREGATES: dict[int, tuple[Path, str]] = {
    40_000_000: (RUN052 / "inputs/real_40000000_aggregate.npz",
                 "ff3631f3e3beeb1ee5a8c8278cc44d3ecbceb2f50c09f8ceab1f8599d6fd7955"),
    20_000_000: (RUN051 / "inputs/real_20000000_aggregate.npz",
                 "d689f445aa8853811b121979abf7f0c0460b1807510a538a71e8c708bd2bb384"),
    10_000_000: (RUN051 / "inputs/real_10000000_aggregate.npz",
                 "452073b03ccf4513db268d937968f90c46b07079f45ca1f1d6e5b61c69b33f6c"),
    5_000_000: (RUN051 / "inputs/real_5000000_aggregate.npz",
                "f73a37452da1935d2d90131aac301f48fec87c82481d230afea0bbbda1b7a777"),
    2_000_000: (RUN051 / "inputs/real_2000000_aggregate.npz",
                "08eb35ba0b07bfa652ae46162053cc575f16d48e9c603e60216a29157e95eca9"),
    1_000_000: (S045 / "inputs/real_1000000_aggregate.npz",
                "80984d804f8ae0552f6bab34a77a03e778073ac87f3b3ec6f04f96de74137420"),
    500_000: (RUN052 / "inputs/real_500000_aggregate.npz",
              "658ca20d6ccb0eedd691960c57305f403c183e89ecf649d50fbac14aa60e6016"),
    200_000: (RUN052 / "inputs/real_200000_aggregate.npz",
              "119631597e6931c3fa734975b1a160cb9db9407606f9047d515b190af5df86d2"),
}
EXPECTED_RAW_RECORDS = 1_703_888
EXPECTED_N_LOCI = {40_000_000: 78, 20_000_000: 144, 10_000_000: 276, 5_000_000: 538,
                   2_000_000: 1329, 1_000_000: 2645, 500_000: 5278, 200_000: 13181}

# ------------------------------------------------------------------ 复用 40Mb 前缀
# 052 的 40Mb 端点：同一次 014 无标签 random 盲源（seed 2207, p=0.75）在真实
# 40,000,000 bp 网格上的展开，同目标/同数据/同权重，200 FG 后 last accepted；
# terminal 明确 reference_opened=false、phase_opened=false。完全符合本轮前缀条件，
# 因此作为 reference-free 已冻结前缀登记复用，**不重跑、不伪造新的 40Mb 终态**。
REUSED_PREFIX_STAGE = "40Mb"
REUSED_PREFIX_BIN = 40_000_000
REUSED_PREFIX_NPZ = RUN052 / "coords/new-chain/40Mb.npz"
REUSED_PREFIX_NPZ_SHA256 = "905575fed78d381f3effc751bb87b7ac68c9423a9b994cebd6bd9b8bbe9ad152"
REUSED_PREFIX_3DG = RUN052 / "coords/new-chain/40Mb.3dg"
REUSED_PREFIX_3DG_SHA256 = "f81557e00724da5041ded899c828794ffa6a7538fcdd02839ccb0b88b683bcdd"
REUSED_PREFIX_TERMINAL = RUN052 / "logs/new-chain-40Mb.terminal.json"
REUSED_PREFIX_TERMINAL_SHA256 = "5b2220802ed5409671c64f14d4703d06087e706a62b597cdffd1fb82cb1d9292"
REUSED_PREFIX_FG = 200
REUSED_PREFIX_STATUS = "budget_not_converged"
REUSED_PREFIX_REASON = "fg_budget_exhausted"

# ------------------------------------------------------------------ 链定义
# stage 名 -> 整数 bp，绝不从名字推断 bin size。40Mb 只在端点查找中出现（复用，不运行）。
STAGE_BIN = {"40Mb": 40_000_000, "20Mb": 20_000_000, "10Mb": 10_000_000, "5Mb": 5_000_000,
             "2Mb": 2_000_000, "1Mb": 1_000_000, "500kb": 500_000, "200kb": 200_000}
RUN_STAGES = ("20Mb", "10Mb", "5Mb", "2Mb", "1Mb", "500kb", "200kb")
ENDPOINT_STAGES = ("40Mb",) + RUN_STAGES
STAGE_FG_CAP = {"20Mb": 200, "10Mb": 200, "5Mb": 612, "2Mb": 404, "1Mb": 486,
                "500kb": 200, "200kb": 100}
NEW_FG_TOTAL = 2202                     # 本轮新计算：20Mb..200kb
GROUP3_FG = REUSED_PREFIX_FG + 200 + 200 + 612 + 404 + 486   # 2102：第三组到 1Mb
GROUP4_FG = GROUP3_FG + 200 + 100                            # 2402：第四组走完 200kb
BASELINE_FG = 1502
EXTRA_FG = 1902

WEIGHTS = {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}
P_INIT = 0.75
BASE_SEED = 2207

# ------------------------------------------------------------------ 冻结对照（只读）
BASELINE_NPZ = RUN046 / "base_remaining/coords/real-G-random/1Mb.npz"
BASELINE_NPZ_SHA256 = "6bb93bf570cf688fd8b02dccf14ca0867c1f831d5e123036ced626856f8d8752"
EXTRA_NPZ = RUN051 / "coords/A-extra-levels/1Mb.npz"
EXTRA_NPZ_SHA256 = "60711facfa41e60b7d5283e18f5aa221e227561d007c4a82729c06a36c02757a"

# 本轮新候选（由本 run 写盘；SHA 在评价前现算并落盘）
CHAIN_1MB_NPZ = RUN / "coords" / FIT_ID / "1Mb.npz"
CHAIN_200KB_NPZ = RUN / "coords" / FIT_ID / "200kb.npz"
CHAIN_COARSE_1MB_NPZ = RUN / "coords/200kb-to-1Mb/coarsened1Mb.npz"

MASK_SNAPSHOT = RUN046 / "evaluation_final/results/frozen_legacy_mask_snapshot.npz"
MASK_SNAPSHOT_SHA256 = "9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9"
REFERENCE_PATH = ROOT / "data/P9016.1m.3dg.gz"
REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"

# ------------------------------------------------------------------ 冻结评价常量
EVAL_OFFSET_BP = 3_000_000
EVAL_BIN_BP = 1_000_000
EVAL_CHROMOSOMES = 20
EVAL_VALID_LOCI = 2447
EVAL_INTER_PAIRS = 157_529
MIN_COMMON_PAIRS = 3
GEOMETRY_TIE_TOL = 1e-12
# 既有两组必须逐位复现的 20-chr 均值（046 / 051 冻结值）
FROZEN_EXPECTED_MEANS = {
    "baseline": {"same": 0.6048764784390889, "cross": 0.3776082995623037},
    "extra_levels": {"same": 0.6266194401328646, "cross": 0.3847369040948677},
}

CHR1_INDEX = 0
COARSE_BP_FOR_MATRIX = {40_000_000: 78, 20_000_000: 144, 10_000_000: 276, 5_000_000: 538,
                        2_000_000: 1329, 1_000_000: 2645, 500_000: 5278, 200_000: 13181}

# 交付与展示规范
PANEL_IN = 3.0
DPI = 300
FONT_PT = 7
COLORS = {"same": "#2b6cb0", "cross": "#dd6b20"}

# 四个箱线图组的冻结顺序与标签（图内英文）
BOX_GROUPS = (
    {"key": "baseline", "tick": "Baseline\n5\u21922\u21921 Mb\n1502 FG",
     "lineage": "014 random -> 5Mb 612 -> 2Mb 404 -> 1Mb 486"},
    {"key": "extra_levels", "tick": "Extra levels\n20\u219210\u21925\u21922\u21921 Mb\n1902 FG",
     "lineage": "014 random -> 20Mb 200 -> 10Mb 200 -> 5Mb 612 -> 2Mb 404 -> 1Mb 486"},
    {"key": "chain_1Mb", "tick": "Shared chain (40 Mb start)\n40\u219220\u219210\u21925\u21922\u21921 Mb\n2102 FG",
     "lineage": "014 random -> 40Mb 200 (reused frozen prefix) -> 20Mb 200 -> 10Mb 200 -> "
                "5Mb 612 -> 2Mb 404 -> 1Mb 486"},
    {"key": "chain_coarse_1Mb", "tick": "Full chain 200 kb\u21921 Mb\n40/20/10/5/2/1 Mb+500/200 kb\n2402 FG",
     "lineage": "same shared chain, then 500kb 200 -> 200kb 100, arithmetic-mean coarsening to 1Mb"},
)

# chr1 距离矩阵列序（主侧已明确，不得改序）
MATRIX_COLUMNS = ("Reference", "extra_levels", "chain_1Mb", "chain_coarse_1Mb")


def stage_bin(stage: str) -> int:
    if stage not in STAGE_BIN:
        raise KeyError("unknown stage %r" % stage)
    return int(STAGE_BIN[stage])


def stage_of_endpoint(path) -> str:
    stem = str(path).rsplit("/", 1)[-1]
    if stem.endswith(".npz"):
        stem = stem[:-4]
    if stem not in STAGE_BIN:
        raise RuntimeError("endpoint %s does not name a chain stage" % path)
    return stem
