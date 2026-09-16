"""053 评价：四版本（Baseline / Extra levels / New chain 1 Mb / 200kb->1Mb 粗化）在冻结支持上的 signed Spearman。

追加请求只加一个候选端点（40->10->5->2->1Mb 直接 1Mb），其余逻辑逐字沿用
052/code/evaluate_spearman.py：

* 四个候选 npz 先写盘/核对 SHA256，之后才打开参考 3DG；参考只用于评价与展示，
  不参与任何初始化 / 训练 / 停止 / 选择；
* positions / pair_i / pair_j / common 逐字取自 046 evaluation_final 的
  frozen_legacy_mask_snapshot.npz，四个版本共用同一支持（2447 个有效 loci、
  157,529 个 intra pair、20 条染色体），任何缩减都直接报错；
* index = offset(chromosome) + bp // 1_000_000，offset = cumsum(ceil(chromosome_length/1e6))；
* 距离是候选 1Mb 坐标的欧氏距离；指标是有符号 Spearman rho（平均秩）；
* 每 chr 用 4 个 rho 的最大配对和选择一次 whole-chr A/B swap；
  same = 匹配两 copy 的 rho 平均，cross = 互换两 copy 的 rho 平均；
* NA 明确保留，不 nanmean 后宣称 n=20；
* 三个既有版本的逐 chr 数值与 052/eval 逐字核对（只读一致性检查，不重拟合）。

只描述单细胞：观察单位是染色体，不是生物学重复；无 bootstrap、无显著性、无新 null。
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
EVAL_DIR = RUN / "eval"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from four_way_paths import (AGGREGATE_200KB, CANDIDATES, EVAL_BIN_BP, EVAL_CHROMOSOMES,  # noqa: E402
                            EVAL_INTER_PAIRS, EVAL_OFFSET_BP, EVAL_VALID_LOCI, GEOMETRY_TIE_TOL,
                            MASK_SNAPSHOT, MASK_SNAPSHOT_SHA256, MIN_COMMON_PAIRS, REFERENCE_PATH,
                            REFERENCE_SHA256, ROOT, RUN052_SUMMARY, RUN052_TSV, VERSION_ORDER,
                            sha256_file, verify_candidate_hashes)

CONSISTENCY_TOL = 1e-12


def load_npz_coordinates(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
    if coordinates.shape != expected_shape:
        raise RuntimeError("candidate %s shape %s != %s" % (path, coordinates.shape, expected_shape))
    if not np.all(np.isfinite(coordinates)):
        raise RuntimeError("candidate %s has nonfinite coordinates" % path)
    return coordinates


def load_reference_tracks(path: Path) -> dict[str, dict[int, np.ndarray]]:
    """解析冻结参考 3DG（.gz）；非有限行保留为 NaN（本文件无 NaN 行）。"""
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


def check_legacy_consistency(rows: list[dict]) -> dict:
    """三个既有版本与 052/eval 的逐 chr 数值核对（只读；不重拟合任何候选）。"""
    report: dict[str, dict] = {"max_abs_diff": {}, "compared_values": 0}
    summary_052 = json.loads(Path(RUN052_SUMMARY).read_text(encoding="utf-8"))
    for version in ("Baseline", "Extra levels", "200 kb -> 1 Mb"):
        key = CANDIDATES[version]["key"]
        ref = summary_052["per_version"][key]
        mine = per_version_stats(rows, key)
        worst = 0.0
        for metric in ("same", "cross", "difference_same_minus_cross"):
            for field in ("mean", "median"):
                if ref[metric][field] is None or mine[metric][field] is None:
                    if ref[metric][field] != mine[metric][field]:
                        raise RuntimeError("legacy summary NA mismatch %s %s %s" % (version, metric, field))
                    continue
                worst = max(worst, abs(float(ref[metric][field]) - float(mine[metric][field])))
            if int(ref[metric]["n_defined"]) != int(mine[metric]["n_defined"]):
                raise RuntimeError("legacy summary n_defined mismatch %s %s" % (version, metric))
        if worst > CONSISTENCY_TOL:
            raise RuntimeError("legacy summary mismatch for %s: %.3e" % (version, worst))
        report["max_abs_diff"]["%s/summary_mean_median" % key] = worst

    # 逐 chr 数值：与 052 的 TSV 逐项比较
    header, ref_rows = None, []
    with Path(RUN052_TSV).open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if header is None:
                header = fields
                continue
            ref_rows.append(dict(zip(header, fields)))
    if len(ref_rows) != len(rows):
        raise RuntimeError("052 tsv row count changed")
    compared = 0
    for mine, ref in zip(rows, ref_rows):
        if mine["chromosome"] != ref["chromosome"]:
            raise RuntimeError("052 tsv chromosome order changed")
        for version in ("Baseline", "Extra levels", "200 kb -> 1 Mb"):
            key = CANDIDATES[version]["key"]
            for field in ("A_mat", "A_pat", "B_mat", "B_pat", "same", "cross", "contrast"):
                raw = ref["%s_%s" % (key, field)]
                expected = float("nan") if raw == "NA" else float(raw)
                actual = float(mine["%s_%s" % (key, field)])
                if math.isnan(expected) and math.isnan(actual):
                    continue
                if math.isnan(expected) != math.isnan(actual):
                    raise RuntimeError("052 tsv NA mismatch %s %s %s" % (mine["chromosome"], key, field))
                # 052 的 TSV 只写到 6 位小数，比较前把本次数值按同一格式取整
                worst = abs(expected - float("%.6f" % actual))
                if worst > CONSISTENCY_TOL:
                    raise RuntimeError("052 tsv mismatch %s %s %s: %.3e" % (mine["chromosome"], key, field, worst))
                report["max_abs_diff"]["tsv:%s/%s" % (key, field)] = max(
                    report["max_abs_diff"].get("tsv:%s/%s" % (key, field), 0.0), worst)
                compared += 1
    report["compared_values"] = compared
    report["tolerance"] = CONSISTENCY_TOL
    report["status"] = "three legacy versions reproduce 052/eval exactly"
    return report


def per_version_stats(rows: list[dict], key: str) -> dict[str, dict]:
    out = {}
    for metric, column in (("same", "same"), ("cross", "cross"),
                           ("difference_same_minus_cross", "contrast")):
        values = [float(row["%s_%s" % (key, column)]) for row in rows]
        values = [v for v in values if math.isfinite(v)]
        if values:
            array = np.asarray(values, dtype=np.float64)
            out[metric] = {"mean": float(array.mean()), "median": float(np.median(array)),
                           "min": float(array.min()), "max": float(array.max()),
                           "n_defined": int(array.size), "n_na": int(len(rows) - array.size)}
        else:
            out[metric] = {"mean": None, "median": None, "min": None, "max": None,
                           "n_defined": 0, "n_na": len(rows)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-hashes", action="store_true")
    args = parser.parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 候选与哈希（必须在打开参考之前） ----------
    audit = verify_candidate_hashes(EVAL_DIR / "candidate_hashes.json")

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

    # ---------- 4. 坐标网格（只读 200kb aggregate 的小型 grid 字段，不碰 8686 万 pair 数组） ----------
    with np.load(AGGREGATE_200KB, allow_pickle=False) as fine:
        chromosome_names = tuple(str(x) for x in fine["chromosome_names"])
        chromosome_lengths = [int(x) for x in fine["chromosome_lengths"]]
        fine_n_bins = [int(x) for x in fine["n_bins"]]
        fine_offsets = [int(x) for x in fine["offsets"]]
        fine_bin_bp = int(fine["bin_size"])
        n_loci_200kb = int(fine["locus_bin"].shape[0])
    if len(chromosome_names) != EVAL_CHROMOSOMES:
        raise RuntimeError("chromosome inventory changed")
    if fine_bin_bp != 200_000 or fine_n_bins != [(L + 199_999) // 200_000 for L in chromosome_lengths]:
        raise RuntimeError("200kb grid changed")
    if fine_offsets != list(np.concatenate(([0], np.cumsum(fine_n_bins[:-1])))):
        raise RuntimeError("200kb offsets changed")
    if n_loci_200kb != 13181 or int(np.sum(fine_n_bins)) != n_loci_200kb:
        raise RuntimeError("200kb locus count changed")

    n_bins_1mb = [(L + EVAL_BIN_BP - 1) // EVAL_BIN_BP for L in chromosome_lengths]
    offsets = np.concatenate(([0], np.cumsum(n_bins_1mb[:-1])))
    n_loci_1mb = int(sum(n_bins_1mb))
    coords_by_version = {
        name: load_npz_coordinates(Path(CANDIDATES[name]["npz"]), (2, n_loci_1mb, 3))
        for name in VERSION_ORDER
    }

    # ---------- 5. 逐染色体 Spearman ----------
    per_chromosome_rows = []
    per_version = {name: {"matched": [], "cross": [], "contrast": []} for name in VERSION_ORDER}
    defined_counts = {name: {"same": 0, "cross": 0, "contrast": 0} for name in VERSION_ORDER}
    na_by_version = {name: [] for name in VERSION_ORDER}
    total_pairs = 0
    valid_loci_total = 0
    for ci, name in enumerate(chromosome_names):
        positions = np.asarray(masks["chr%d_positions" % ci], dtype=np.int64)
        pair_i = np.asarray(masks["chr%d_pair_i" % ci], dtype=np.int64)
        pair_j = np.asarray(masks["chr%d_pair_j" % ci], dtype=np.int64)
        common = np.asarray(masks["chr%d_common" % ci], dtype=bool)
        expected_positions = np.arange(EVAL_OFFSET_BP, chromosome_lengths[ci], EVAL_BIN_BP, dtype=np.int64)
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
        for version in VERSION_ORDER:
            coords = coords_by_version[version]
            candidate_a = distance(coords[0, global_indices], pair_i[common], pair_j[common])
            candidate_b = distance(coords[1, global_indices], pair_i[common], pair_j[common])
            rho = {"A_mat": spearman(candidate_a, ref_mat), "A_pat": spearman(candidate_a, ref_pat),
                   "B_mat": spearman(candidate_b, ref_mat), "B_pat": spearman(candidate_b, ref_pat)}
            derived = derive_rho(rho)
            key = CANDIDATES[version]["key"]
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

    # ---------- 6. 既有三版本一致性核对（只读 052 结果） ----------
    consistency = check_legacy_consistency(per_chromosome_rows)

    # ---------- 7. 写出表与 summary ----------
    columns = ["chromosome", "chromosome_index", "n_pairs", "n_positions", "n_valid_loci_with_common_pairs"]
    for version in VERSION_ORDER:
        key = CANDIDATES[version]["key"]
        columns += ["%s_%s" % (key, f) for f in
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

    summary = {
        "schema": "p9016-round053-four-way-spearman-summary-v1",
        "metric": "signed Spearman rho (average-tie ranks) on euclidean distances of 1Mb coarse coordinates",
        "appended_candidate": {
            "label": "New chain 1 Mb",
            "npz": audit["New chain 1 Mb"]["npz"], "npz_sha256": audit["New chain 1 Mb"]["npz_sha256"],
            "note": "existing endpoint of the 40 -> 10 -> 5 -> 2 -> 1 Mb chain; not the 20Mb extra chain "
                    "and not the coarsened 1Mb",
        },
        "support": {
            "source": str(MASK_SNAPSHOT.relative_to(ROOT)), "sha256": mask_sha,
            "n_chromosomes": EVAL_CHROMOSOMES, "n_valid_loci": valid_loci_total,
            "n_intra_pairs": total_pairs,
            "index_rule": "global_index = offset(chromosome) + bp // 1000000, "
                          "offset = cumsum(ceil(chromosome_length/1000000))",
            "common_denominator": "all four versions use the identical frozen positions / pair_i / pair_j / "
                                  "common mask; no per-version subsetting",
        },
        "reference": {"path": str(REFERENCE_PATH.relative_to(ROOT)), "sha256": reference_sha,
                      "use": "read once after all candidate hashes were written; never used for "
                             "initialization, training, stopping or selection"},
        "candidates": audit,
        "legacy_consistency_check": consistency,
        "per_version": {},
        "budget_limitation": "the four versions do not share an equal FG budget "
                             "(1502 / 1902 / 1902 / 2202); the comparison is not equal-cost and all "
                             "endpoints are budget_not_converged",
        "scope": "one cell, 20 chromosomes as correlated measurements (observation unit = chromosome, not a "
                 "biological replicate); no new fitting, no re-optimization, no bootstrap, no significance "
                 "test, no new null, no R1/R3",
    }
    for version in VERSION_ORDER:
        key = CANDIDATES[version]["key"]
        stats = per_version_stats(per_chromosome_rows, key)
        summary["per_version"][key] = {
            "label": version, "tick_label": CANDIDATES[version]["tick"],
            "own_budget_fg": CANDIDATES[version]["own_budget_fg"],
            "same": stats["same"], "cross": stats["cross"],
            "difference_same_minus_cross": stats["difference_same_minus_cross"],
            "n_undefined_chromosomes": na_by_version[version],
        }
    (EVAL_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8")

    print(json.dumps({key: {metric: (None if value["mean"] is None else round(value["mean"], 6))
                            for metric, value in row.items()
                            if isinstance(value, dict) and "mean" in value}
                      for key, row in summary["per_version"].items()}, indent=2))
    print("support pairs", total_pairs, "loci", valid_loci_total)
    print("legacy consistency:", json.dumps(consistency["max_abs_diff"], sort_keys=True))
    print("tsv", tsv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
