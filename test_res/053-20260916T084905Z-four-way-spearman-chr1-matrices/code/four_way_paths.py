"""053 轮共享路径、冻结常量与候选哈希核对。

本目录只做绘图追加：读取既有结果（046 / 051 / 052）与冻结评价快照，不重新拟合、
不追加优化、不新增实验。四个候选的 npz 在打开参考 3DG 之前先写/核对 SHA256。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
RUN046 = ROOT / "test_res/046-UTC-real-cell-shared-capture"
RUN051 = ROOT / "test_res/051-20260916T064500-g-random-baseline-ext"
RUN052 = ROOT / "test_res/052-20260916T080116Z-200kb-coarsened-1mb-chain"

# ---------- 四个候选（三个既有端点 + 新增的 40->10->5->2->1Mb 直接 1Mb 端点） ----------
CANDIDATES: dict[str, dict] = {
    "Baseline": {
        "key": "baseline",
        "npz": RUN046 / "base_remaining/coords/real-G-random/1Mb.npz",
        "sha256": "6bb93bf570cf688fd8b02dccf14ca0867c1f831d5e123036ced626856f8d8752",
        "own_budget_fg": 1502,
        "lineage": "014 random -> 5Mb 612 -> 2Mb 404 -> 1Mb 486",
        "tick": "Baseline\n5 \u2192 2 \u2192 1 Mb\n1502 FG",
    },
    "Extra levels": {
        "key": "extra_levels",
        "npz": RUN051 / "coords/A-extra-levels/1Mb.npz",
        "sha256": "60711facfa41e60b7d5283e18f5aa221e227561d007c4a82729c06a36c02757a",
        "own_budget_fg": 1902,
        "lineage": "014 random -> 20Mb 200 -> 10Mb 200 -> 5Mb 612 -> 2Mb 404 -> 1Mb 486",
        "tick": "Extra levels\n20 \u2192 10 \u2192 5 \u2192 2 \u2192 1 Mb\n1902 FG",
    },
    "New chain 1 Mb": {
        "key": "new_chain_1Mb",
        "npz": RUN052 / "coords/new-chain/1Mb.npz",
        "sha256": None,  # 新读入的既有端点：写盘后核对本次哈希
        "own_budget_fg": 1902,
        "lineage": "014 random -> 40Mb 200 -> 10Mb 200 -> 5Mb 612 -> 2Mb 404 -> 1Mb 486",
        "tick": "New chain 1 Mb\n40 \u2192 10 \u2192 5 \u2192 2 \u2192 1 Mb\n1902 FG",
    },
    "200 kb -> 1 Mb": {
        "key": "new_200kb_to_1Mb",
        "npz": RUN052 / "coords/200kb-to-1Mb/coarsened1Mb.npz",
        "sha256": None,
        "own_budget_fg": 2202,
        "lineage": "014 random -> 40Mb 200 -> 10Mb 200 -> 5Mb 612 -> 2Mb 404 -> 1Mb 486 "
                   "-> 500kb 200 -> 200kb 100, then arithmetic-mean coarsening to 1Mb",
        "tick": "200 kb \u2192 1 Mb\nfull chain coarsened\n2202 FG",
    },
}
VERSION_ORDER = ("Baseline", "Extra levels", "New chain 1 Mb", "200 kb -> 1 Mb")

# 200kb 细层端点（只用于 chr1 矩阵展示，不进 Spearman 分母）
NEW_CHAIN_200KB_NPZ = RUN052 / "coords/new-chain/200kb.npz"
NEW_CHAIN_200KB_SHA256 = "c24ba8eabeac722f09ef5c6a4b6137a62984265e31523493fbb6d0c2954ad1b6"
AGGREGATE_200KB = RUN052 / "inputs/real_200000_aggregate.npz"

# 冻结支持与参考
MASK_SNAPSHOT = RUN046 / "evaluation_final/results/frozen_legacy_mask_snapshot.npz"
MASK_SNAPSHOT_SHA256 = "9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9"
REFERENCE_PATH = ROOT / "data/P9016.1m.3dg.gz"
REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"

# 与 052 完全一致的既有结果（只读一致性核对）
RUN052_SUMMARY = RUN052 / "eval/summary.json"
RUN052_TSV = RUN052 / "eval/per_chromosome_spearman.tsv"

# 评价冻结常量（逐字沿用 052/code/round_paths_052.py）
EVAL_OFFSET_BP = 3_000_000
EVAL_BIN_BP = 1_000_000
EVAL_CHROMOSOMES = 20
EVAL_VALID_LOCI = 2447
EVAL_INTER_PAIRS = 157_529
MIN_COMMON_PAIRS = 3
GEOMETRY_TIE_TOL = 1e-12

CHR1_INDEX = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_candidate_hashes(audit_path: Path) -> dict:
    """在打开参考之前写盘：四个候选的实际 SHA256 与冻结期望逐项核对。"""
    audit: dict[str, dict] = {}
    for name in VERSION_ORDER:
        spec = CANDIDATES[name]
        path = Path(spec["npz"])
        if not path.is_file():
            raise RuntimeError("missing candidate for %s: %s" % (name, path))
        actual = sha256_file(path)
        expected = spec["sha256"]
        if expected is not None and actual != expected:
            raise RuntimeError("frozen candidate SHA mismatch for %s: %s" % (name, actual))
        audit[name] = {
            "npz": str(path.relative_to(ROOT)),
            "npz_sha256": actual,
            "frozen_expected_sha256": expected,
            "frozen_match": True if expected is None else bool(actual == expected),
            "own_budget_fg": spec["own_budget_fg"],
            "lineage": spec["lineage"],
        }
    fine = Path(NEW_CHAIN_200KB_NPZ)
    actual_fine = sha256_file(fine)
    if actual_fine != NEW_CHAIN_200KB_SHA256:
        raise RuntimeError("200kb endpoint SHA mismatch: %s" % actual_fine)
    audit["__chr1_matrix_only__/New chain 200 kb"] = {
        "npz": str(fine.relative_to(ROOT)),
        "npz_sha256": actual_fine,
        "frozen_expected_sha256": NEW_CHAIN_200KB_SHA256,
        "frozen_match": True,
        "own_budget_fg": 2202,
        "lineage": "new chain continuation to 500kb 200 -> 200kb 100 (native 200kb grid)",
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    return audit
