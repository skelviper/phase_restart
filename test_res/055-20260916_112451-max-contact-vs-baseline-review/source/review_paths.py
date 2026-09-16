"""055 轮路径与常量（只读引用旧运行，不改任何旧字节）。"""
from __future__ import annotations

from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
CONFIG = RUN / "config.json"
EVAL = RUN / "eval"
PLOTS = RUN / "plots"
LOGS = RUN / "logs"

MASK = ROOT / ("test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/"
               "frozen_legacy_mask_snapshot.npz")
MASK_SHA256 = "9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9"

REFERENCE = ROOT / "data/P9016.1m.3dg.gz"
REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"

# 回归核对用的旧冻结表（只读）
R2_049 = ROOT / ("test_res/049-20260915T162917Z-max-contact-unified-multiscale/"
                 "evaluation/results/r2_per_chromosome.tsv")
PEARSON_051 = ROOT / ("test_res/051-20260916T064500-g-random-baseline-ext/"
                      "evaluation/per_chromosome.tsv")

CHROMOSOMES = ["chr%d" % index for index in range(1, 20)] + ["chrX"]
GROUP_ORDER = ["Reference", "Baseline-046-G-random", "B-hard-observed", "C-max-rate"]
CANDIDATE_ORDER = GROUP_ORDER[1:]
TIE_TOLERANCE = 1e-12
JITTER_SEED = 5501

__all__ = ["RUN", "ROOT", "CONFIG", "EVAL", "PLOTS", "LOGS", "MASK", "MASK_SHA256",
           "REFERENCE", "REFERENCE_SHA256", "R2_049", "PEARSON_051", "CHROMOSOMES",
           "GROUP_ORDER", "CANDIDATE_ORDER", "TIE_TOLERANCE", "JITTER_SEED"]
