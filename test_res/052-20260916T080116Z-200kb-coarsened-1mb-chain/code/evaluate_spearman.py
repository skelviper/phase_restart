"""052 评价：三版本（Baseline / Extra levels / 200kb->1Mb）在冻结 old21 支持上的 signed Spearman。

隔离规则（config.json 冻结）：

* 本脚本在三个版本坐标与哈希全部落盘后才运行；参考 3DG 只在这里被打开一次；
* 参考坐标不参与任何初始化 / 训练 / 停止 / 选择；
* positions / pair_i / pair_j / common 逐字取自 046 evaluation_final 的 frozen_legacy_mask_snapshot.npz，
  三版本共用同一支持（2447 个有效 loci、157,529 个 intra pair、20 条染色体），任何缩减都直接报错；
* index = offset(chromosome) + bp // 1_000_000，offset = cumsum(ceil(chromosome_length/1e6))；
* 距离是候选（1Mb）坐标的欧氏距离；指标是有符号 Spearman rho（平均秩，scipy.stats.spearmanr 等价）；
* 每 chr 用 4 个 rho 的最大配对和选择一次 whole-chr A/B swap（本次按 Spearman 选）；
  same = 匹配两 copy 的 rho 平均，cross = 互换两 copy 的 rho 平均；
* 每版本 20 个 same / 20 个 cross；NA 明确保留，不 nanmean 后宣称 n=20。

只描述单细胞：无 bootstrap、无显著性、无新 null、无 R1/R3。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from round_paths_052 import (BASELINE_NPZ, BASELINE_NPZ_SHA256, EVAL_BIN_BP, EVAL_CHROMOSOMES,  # noqa: E402
                             EVAL_INTER_PAIRS, EVAL_VALID_LOCI, EXTRA_NPZ, EXTRA_NPZ_SHA256,
                             GEOMETRY_TIE_TOL, MASK_SNAPSHOT, MASK_SNAPSHOT_SHA256, MIN_COMMON_PAIRS,
                             REFERENCE_PATH, REFERENCE_SHA256, VERSIONS)

EVAL_DIR = RUN / "eval"
COARSE_NPZ = RUN / "coords" / "200kb-to-1Mb" / "coarsened1Mb.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz_coordinates(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
    if coordinates.shape != expected_shape:
        raise RuntimeError("candidate %s shape %s != %s" % (path, coordinates.shape, expected_shape))
    if not np.all(np.isfinite(coordinates)):
        raise RuntimeError("candidate %s has nonfinite coordinates" % path)
    return coordinates


def load_reference_tracks(path: Path) -> dict[str, dict[int, np.ndarray]]:
    """解析冻结参考 3DG（.gz），与 046 evaluator 的 3dg 分支同规则；非有限行保留为 NaN。"""
    tracks: dict[str, dict[int, np.ndarray]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                track = str(fields[0])
                position = int(fields[1])
                point = np.asarray([float(item) for item in fields[2:5]], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("invalid reference row at %s:%d" % (path, line_number)) from exc
            if not np.isfinite(point).all():
                point = np.full(3, np.nan, dtype=np.float64)
            target = tracks.setdefault(track, {})
            if position in target:
                raise RuntimeError("duplicate reference coordinate %s:%d" % (track, position))
            target[position] = point
    return tracks


def dense_points(tracks, track: str, positions: np.ndarray) -> np.ndarray:
    rows = tracks.get(track, {})
    points = np.full((len(positions), 3), np.nan, dtype=np.float64)
    for index, position in enumerate(positions):
        value = rows.get(int(position))
        if value is not None and np.isfinite(value).all():
            points[index] = np.asarray(value, dtype=np.float64)
    return points


def distance(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    delta = np.asarray(points[pair_i] - points[pair_j], dtype=np.float64)
    return np.sqrt(np.sum(delta * delta, axis=1))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """有符号 Spearman rho；平均秩，等价 scipy.stats.spearmanr 的统计量。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < MIN_COMMON_PAIRS or len(x) != len(y):
        return float("nan")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else float("nan")


def derive_rho(rho: dict[str, float]) -> dict:
    """按 Spearman 的最大配对和选择一次 whole-chr A/B swap。"""
    required = ("A_mat", "A_pat", "B_mat", "B_pat")
    if not all(math.isfinite(float(rho[key])) for key in required):
        return {"rho": {k: (float(rho[k]) if math.isfinite(float(rho[k])) else None) for k in required},
                "direct": None, "swapped": None, "matched": None, "cross": None, "contrast": None,
                "orientation": "undefined", "defined": False}
    direct = (float(rho["A_mat"]) + float(rho["B_pat"])) / 2.0
    swapped = (float(rho["A_pat"]) + float(rho["B_mat"])) / 2.0
    base = {"rho": {k: float(rho[k]) for k in required}, "direct": direct, "swapped": swapped}
    if abs(direct - swapped) <= GEOMETRY_TIE_TOL:
        return {**base, "matched": (direct + swapped) / 2.0, "cross": (direct + swapped) / 2.0,
                "contrast": 0.0, "orientation": "tie", "defined": True}
    if direct > swapped:
        return {**base, "matched": direct, "cross": swapped, "contrast": direct - swapped,
                "orientation": "direct", "defined": True}
    return {**base, "matched": swapped, "cross": direct, "contrast": swapped - direct,
            "orientation": "swapped", "defined": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-hashes", action="store_true")
    args = parser.parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 候选与哈希（必须在打开参考之前） ----------
    candidates = {
        "Baseline": {"npz": BASELINE_NPZ, "npz_sha256": BASELINE_NPZ_SHA256, "fg": 1502,
                     "lineage": "014 random -> 5Mb 612 -> 2Mb 404 -> 1Mb 486"},
        "Extra levels": {"npz": EXTRA_NPZ, "npz_sha256": EXTRA_NPZ_SHA256, "fg": 1902,
                         "lineage": "014 random -> 20Mb 200 -> 10Mb 200 -> 5Mb 612 -> 2Mb 404 -> 1Mb 486"},
        VERSIONS[2]: {"npz": COARSE_NPZ, "npz_sha256": None, "fg": 2202,
                      "lineage": "014 random -> 40Mb 200 -> 10Mb 200 -> 5Mb 612 -> 2Mb 404 -> 1Mb 486 "
                                 "-> 500kb 200 -> 200kb 100, then arithmetic-mean coarsening to 1Mb"},
    }
    candidate_audit = {}
    for name, spec in candidates.items():
        path = Path(spec["npz"])
        if not path.is_file():
            raise RuntimeError("missing candidate for %s: %s" % (name, path))
        actual = sha256_file(path)
        if spec["npz_sha256"] is not None and actual != spec["npz_sha256"]:
            raise RuntimeError("frozen candidate SHA mismatch for %s: %s" % (name, actual))
        candidate_audit[name] = {"npz": str(path.relative_to(ROOT)), "npz_sha256": actual,
                                 "frozen_expected_sha256": spec["npz_sha256"],
                                 "own_budget_fg": spec["fg"], "lineage": spec["lineage"]}
    (EVAL_DIR / "candidate_hashes.json").write_text(
        json.dumps(candidate_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---------- 2. 参考（此时才第一次打开） ----------
    reference_sha = sha256_file(REFERENCE_PATH)
    if reference_sha != REFERENCE_SHA256:
        raise RuntimeError("reference 3DG SHA mismatch")
    tracks = load_reference_tracks(REFERENCE_PATH)

    # ---------- 3. 冻结支持 ----------
    mask_sha = sha256_file(MASK_SNAPSHOT)
    if mask_sha != MASK_SNAPSHOT_SHA256:
        raise RuntimeError("frozen mask snapshot SHA mismatch")
    with np.load(MASK_SNAPSHOT, allow_pickle=False) as archive:
        masks = {key: np.asarray(archive[key]) for key in archive.files}

    # ---------- 4. 坐标（1Mb 完整 grid） ----------
    fine = np.load(RUN / "inputs" / "real_200000_aggregate.npz", allow_pickle=False)
    chromosome_names = tuple(str(x) for x in fine["chromosome_names"])
    chromosome_lengths = [int(x) for x in fine["chromosome_lengths"]]
    if len(chromosome_names) != EVAL_CHROMOSOMES:
        raise RuntimeError("chromosome inventory changed")
    n_bins = [(L + EVAL_BIN_BP - 1) // EVAL_BIN_BP for L in chromosome_lengths]
    offsets = np.concatenate(([0], np.cumsum(n_bins[:-1])))
    n_loci_1mb = int(sum(n_bins))
    n_loci_200kb = int(fine["locus_bin"].shape[0])
    fine_shape = (2, n_loci_200kb, 3)
    coords_by_version = {
        "Baseline": load_npz_coordinates(Path(candidates["Baseline"]["npz"]), (2, n_loci_1mb, 3)),
        "Extra levels": load_npz_coordinates(Path(candidates["Extra levels"]["npz"]), (2, n_loci_1mb, 3)),
        VERSIONS[2]: load_npz_coordinates(COARSE_NPZ, (2, n_loci_1mb, 3)),
    }
    if fine_shape[1] != 13181:
        raise RuntimeError("200kb grid changed")

    # ---------- 5. 逐染色体 Spearman ----------
    per_chromosome_rows = []
    per_version = {name: {"matched": [], "cross": [], "contrast": []} for name in VERSIONS}
    defined_counts = {name: {"same": 0, "cross": 0, "contrast": 0} for name in VERSIONS}
    na_by_version = {name: [] for name in VERSIONS}
    total_pairs = 0
    valid_loci_total = 0
    for ci, name in enumerate(chromosome_names):
        positions = np.asarray(masks["chr%d_positions" % ci], dtype=np.int64)
        pair_i = np.asarray(masks["chr%d_pair_i" % ci], dtype=np.int64)
        pair_j = np.asarray(masks["chr%d_pair_j" % ci], dtype=np.int64)
        common = np.asarray(masks["chr%d_common" % ci], dtype=bool)
        expected_positions = np.arange(3_000_000, chromosome_lengths[ci], EVAL_BIN_BP, dtype=np.int64)
        if not np.array_equal(positions, expected_positions):
            raise RuntimeError("frozen positions do not match the documented rule for %s" % name)
        expected_pairs = np.triu_indices(len(positions), k=1)
        if not (np.array_equal(pair_i, expected_pairs[0]) and np.array_equal(pair_j, expected_pairs[1])):
            raise RuntimeError("frozen pair grid changed for %s" % name)
        global_indices = int(offsets[ci]) + positions // EVAL_BIN_BP
        if global_indices.max() >= n_loci_1mb or global_indices.min() < 0:
            raise RuntimeError("global index mapping escaped the 1Mb grid for %s" % name)
        total_pairs += int(common.sum())
        valid_loci_total += int(np.unique(np.concatenate((pair_i[common], pair_j[common]))).size)

        ref_mat = distance(dense_points(tracks, "%s(mat)" % name, positions), pair_i[common], pair_j[common])
        ref_pat = distance(dense_points(tracks, "%s(pat)" % name, positions), pair_i[common], pair_j[common])
        row = {"chromosome": name, "chromosome_index": ci, "n_pairs": int(common.sum()),
               "n_positions": int(len(positions)),
               "n_valid_loci_with_common_pairs": int(
                   np.unique(np.concatenate((pair_i[common], pair_j[common]))).size)}
        for version in VERSIONS:
            coords = coords_by_version[version]
            candidate_a = distance(coords[0, global_indices], pair_i[common], pair_j[common])
            candidate_b = distance(coords[1, global_indices], pair_i[common], pair_j[common])
            rho = {"A_mat": spearman(candidate_a, ref_mat), "A_pat": spearman(candidate_a, ref_pat),
                   "B_mat": spearman(candidate_b, ref_mat), "B_pat": spearman(candidate_b, ref_pat)}
            derived = derive_rho(rho)
            key = {"Baseline": "baseline", "Extra levels": "extra_levels",
                   VERSIONS[2]: "new_200kb_to_1Mb"}[version]
            row["%s_A_mat" % key] = derived["rho"]["A_mat"]
            row["%s_A_pat" % key] = derived["rho"]["A_pat"]
            row["%s_B_mat" % key] = derived["rho"]["B_mat"]
            row["%s_B_pat" % key] = derived["rho"]["B_pat"]
            row["%s_same" % key] = derived["matched"]
            row["%s_cross" % key] = derived["cross"]
            row["%s_contrast" % key] = derived["contrast"]
            row["%s_orientation" % key] = derived["orientation"]
            if derived["defined"]:
                per_version[version]["matched"].append(derived["matched"])
                per_version[version]["cross"].append(derived["cross"])
                per_version[version]["contrast"].append(derived["contrast"])
                defined_counts[version]["same"] += 1
                defined_counts[version]["cross"] += 1
                defined_counts[version]["contrast"] += 1
            else:
                na_by_version[version].append(name)
        per_chromosome_rows.append(row)

    if total_pairs != EVAL_INTER_PAIRS or valid_loci_total != EVAL_VALID_LOCI:
        raise RuntimeError("frozen support changed: pairs=%d loci=%d" % (total_pairs, valid_loci_total))

    # ---------- 6. 写出表与 summary ----------
    columns = ["chromosome", "chromosome_index", "n_pairs", "n_positions", "n_valid_loci_with_common_pairs"]
    for version in ("baseline", "extra_levels", "new_200kb_to_1Mb"):
        columns += ["%s_%s" % (version, f) for f in
                    ("A_mat", "A_pat", "B_mat", "B_pat", "same", "cross", "contrast", "orientation")]
    tsv_path = EVAL_DIR / "per_chromosome_spearman.tsv"
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in per_chromosome_rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, float):
                    values.append("NA" if not math.isfinite(value) else "%.6f" % value)
                else:
                    values.append(str(value))
            handle.write("\t".join(values) + "\n")

    def stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "median": None, "n_defined": 0, "n_na": 20}
        array = np.asarray(values, dtype=np.float64)
        return {"mean": float(array.mean()), "median": float(np.median(array)),
                "min": float(array.min()), "max": float(array.max()),
                "n_defined": int(len(array)), "n_na": int(20 - len(array)),
                "values_in_chromosome_order": [float(v) for v in array]}

    summary = {
        "schema": "p9016-round052-spearman-summary-v1",
        "metric": "signed Spearman rho (average-tie ranks) on euclidean distances of 1Mb coarse coordinates",
        "support": {
            "source": str(MASK_SNAPSHOT.relative_to(ROOT)), "sha256": mask_sha,
            "n_chromosomes": EVAL_CHROMOSOMES, "n_valid_loci": valid_loci_total,
            "n_intra_pairs": total_pairs,
            "index_rule": "global_index = offset(chromosome) + bp // 1000000, "
                          "offset = cumsum(ceil(chromosome_length/1000000))",
        },
        "reference": {"path": str(REFERENCE_PATH.relative_to(ROOT)), "sha256": reference_sha,
                      "use": "read once after all candidate hashes were written; never used for "
                             "initialization, training, stopping or selection"},
        "candidates": candidate_audit,
        "per_version": {},
        "budget_limitation": "the three versions do not share an equal FG budget "
                             "(1502 vs 1902 vs 2202); the comparison is not equal-cost",
        "scope": "one cell, 20 chromosomes as correlated measurements; the 40 copy tracks are not "
                 "independent replicates; no bootstrap, no significance test, no new null, no R1/R3",
    }
    for version in VERSIONS:
        key = {"Baseline": "baseline", "Extra levels": "extra_levels",
               VERSIONS[2]: "new_200kb_to_1Mb"}[version]
        same = stats(per_version[version]["matched"])
        cross = stats(per_version[version]["cross"])
        contrast = stats(per_version[version]["contrast"])
        summary["per_version"][key] = {
            "label": version, "own_budget_fg": candidates[version]["fg"],
            "same": same, "cross": cross, "difference_same_minus_cross": contrast,
            "n_undefined_chromosomes": na_by_version[version],
        }
    (EVAL_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print(json.dumps({key: {metric: value["mean"] for metric, value in row.items()
                            if isinstance(value, dict) and "mean" in value}
                      for key, row in summary["per_version"].items()}, indent=2))
    print("support pairs", total_pairs, "loci", valid_loci_total)
    print("tsv", tsv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
