"""049 轮共享路径与常量。

只读复用 045 训练侧实现；不改旧目录任何字节。
"""
from __future__ import annotations

from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
SOURCE_045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round"
SOURCE_045_SRC = SOURCE_045 / "source"
RUN_046 = ROOT / "test_res/046-UTC-real-cell-shared-capture"

AGGREGATE_1MB = SOURCE_045 / "inputs/real_1000000_aggregate.npz"
AGGREGATE_1MB_SHA256 = "80984d804f8ae0552f6bab34a77a03e778073ac87f3b3ec6f04f96de74137420"
INITIAL_CONSENSUS_1MB = SOURCE_045 / "coords/initial/real_consensus_1Mb.npz"
INITIAL_RANDOM_1MB = SOURCE_045 / "coords/initial/real_random_1Mb.npz"
INITIAL_2MB = {"consensus": SOURCE_045 / "coords/initial/real_consensus_2Mb.npz",
               "random": SOURCE_045 / "coords/initial/real_random_2Mb.npz"}
INITIAL_5MB = {"consensus": SOURCE_045 / "coords/initial/real_consensus_5Mb.npz",
               "random": SOURCE_045 / "coords/initial/real_random_5Mb.npz"}
PROLONGATION_SEED = {"consensus": 1103, "random": 2207}

BASELINE_3DG = RUN_046 / "coords/real-extension-G-full-J/1Mb.3dg"
BASELINE_NPZ = RUN_046 / "coords/real-extension-G-full-J/1Mb.npz"
BASELINE_3DG_SHA256 = "4301d4df6e89c1417690599d6687a9e83b18fa37596de5d6c11a911d867d69ea"
BASELINE_NPZ_SHA256 = "116e906e790493afa956445c527b38da11b2fdcaed482fd225f578f188df699d"
MASK_SNAPSHOT = RUN_046 / "evaluation_final/results/frozen_legacy_mask_snapshot.npz"
MASK_SNAPSHOT_SHA256 = "9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9"
REFERENCE_PATH = ROOT / "data/P9016.1m.3dg.gz"
REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"

FIT_FG_CAP = 1502
FTOL = 0.0
CANONICAL_GTOL = 1e-6
MAXLS = 20
PAIR_BLOCK = 262_144
LOSSES = ("A", "B", "C")
SOLVERS = ("raw", "ms")
SOURCES = ("consensus", "random")
FULL_WEIGHTS = {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}
P_INIT = 0.75
RANDOM_U_SEEDS = tuple(range(450500, 450516))
BOOTSTRAP_SEED = 450301
BOOTSTRAP_DRAWS = 10000
PROBE_SEED = 461001
PERMUTATION_SEED = 461100
PERMUTATION_DRAWS = 9999


def fit_id(loss: str, solver: str, source: str) -> str:
    if loss not in LOSSES or solver not in SOLVERS or source not in SOURCES:
        raise ValueError("unknown fit factor")
    return "%s-%s-%s" % (loss, solver, source)


def all_fit_ids() -> list[str]:
    return [fit_id(loss, solver, source) for loss in LOSSES for solver in SOLVERS for source in SOURCES]
