"""055 轮评价：历史 max-contact 端点（049 B/C）与当前 baseline（046 G-random base）及 reference 的
same/cross/contrast 描述性对照。

流程（只读旧产物，不改旧字节）：
1. 先对四个 3DG（Reference / Baseline / B / C）与 mask 计算 SHA256 并写
   eval/pre_reference_hash_gate.json；全部匹配冻结记录后才解析 reference。
2. 按冻结 legacy mask 的公共支持（每 chr 的 positions / pair_i / pair_j / common）取四组共享交集，
   每 chr 计算 A_mat / A_pat / B_mat / B_pat 的有符号 Pearson（次口径 Spearman），
   same = max(direct, swapped)、cross = min、contrast = same - cross，整条染色体一次 A/B 定向。
3. reference self control：same 恒为 1（自相关退化），cross 为 mat/pat 距离向量相关。
4. 回归核对：与 049 冻结 R2（B/C）和 051 冻结 Pearson（baseline）逐 chr 比较；交换不变性、热图对称/零对角/缺失保持、
   支持集计数与 mask 冻结计数核对。
5. 输出 per_chromosome.tsv / summary.json / validation.json / boxplot_data.json / chr1_panels.npz。

本脚本不做拟合、不做 bootstrap、不做 R1/R3、不声明 L2。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from review_paths import (CHROMOSOMES, CANDIDATE_ORDER, CONFIG, EVAL, GROUP_ORDER, LOGS,
                          MASK, MASK_SHA256, PEARSON_051, PLOTS, R2_049, REFERENCE,
                          REFERENCE_SHA256, ROOT, RUN, TIE_TOLERANCE)  # noqa: E402

from scipy.stats import rankdata  # noqa: E402


# --------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_3dg(path: Path) -> dict[str, dict[int, np.ndarray]]:
    tracks: dict[str, dict[int, np.ndarray]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 5:
                continue
            tracks.setdefault(fields[0], {})[int(fields[1])] = np.asarray(
                [float(value) for value in fields[2:5]], dtype=np.float64)
    return tracks


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return float("nan")
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float((left * left).sum()) * float((right * right).sum()))
    if denominator <= 0.0:
        return float("nan")
    return float((left * right).sum() / denominator)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return float("nan")
    return pearson(rankdata(left), rankdata(right))


def distances(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    delta = points[pair_i] - points[pair_j]
    return np.sqrt(np.sum(delta * delta, axis=1))


def whole_cell_rg(tracks: dict[str, dict[int, np.ndarray]]) -> float:
    beads = []
    for table in tracks.values():
        for value in table.values():
            beads.append(value)
    if not beads:
        return float("nan")
    array = np.asarray(beads, dtype=np.float64)
    center = array.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((array - center) ** 2, axis=1))))


def reference_copy_names(tracks: dict[str, dict[int, np.ndarray]], chromosome: str) -> tuple[str, str]:
    mat = pat = None
    for track in tracks:
        match = re.match(r"^%s\((mat|pat)\)$" % re.escape(chromosome), track)
        if not match:
            continue
        if match.group(1) == "mat":
            mat = track
        else:
            pat = track
    if mat is None or pat is None:
        raise RuntimeError("reference chromosome %s is missing a mat/pat copy" % chromosome)
    return mat, pat


def orient(direct: float, swapped: float) -> dict[str, object]:
    if not (math.isfinite(direct) and math.isfinite(swapped)):
        return {"same": float("nan"), "cross": float("nan"), "contrast": float("nan"),
                "orientation": "undefined"}
    if abs(direct - swapped) <= TIE_TOLERANCE:
        mean = (direct + swapped) / 2.0
        return {"same": mean, "cross": mean, "contrast": 0.0, "orientation": "unresolved_tie"}
    if direct > swapped:
        return {"same": direct, "cross": swapped, "contrast": direct - swapped,
                "orientation": "direct"}
    return {"same": swapped, "cross": direct, "contrast": swapped - direct,
            "orientation": "swapped"}


def stats(values: list[float]) -> dict[str, object]:
    finite = [value for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"n_defined": 0, "n_total": len(values), "mean": None, "median": None,
                "min": None, "max": None}
    array = np.asarray(finite, dtype=np.float64)
    return {"n_defined": int(array.size), "n_total": len(values),
            "mean": float(array.mean()), "median": float(np.median(array)),
            "min": float(array.min()), "max": float(array.max())}


def read_frozen_regression() -> tuple[dict, dict]:
    frozen_049: dict[tuple[str, str], dict[str, float]] = {}
    with open(R2_049, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            row = dict(zip(header, fields))
            if row.get("metric") != "pearson":
                continue
            frozen_049[(row["candidate_id"], row["chromosome"])] = {
                "rho_A_mat": float(row["rho_A_mat"]), "rho_A_pat": float(row["rho_A_pat"]),
                "rho_B_mat": float(row["rho_B_mat"]), "rho_B_pat": float(row["rho_B_pat"]),
                "matched": float(row["matched"]), "cross": float(row["cross"]),
                "n_pairs": int(row["n_pairs"]),
            }
    frozen_051: dict[str, dict[str, float]] = {}
    with open(PEARSON_051, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            row = dict(zip(header, fields))
            if row.get("condition") != "baseline":
                continue
            frozen_051[row["chr"]] = {
                "same": float(row["same"]), "cross": float(row["cross"]),
                "contrast": float(row["contrast"]), "n_pairs": int(row["n_pairs"]),
                "A_mat": float(row["A_mat"]), "A_pat": float(row["A_pat"]),
                "B_mat": float(row["B_mat"]), "B_pat": float(row["B_pat"]),
            }
    return frozen_049, frozen_051


# --------------------------------------------------------------------------- main
def main() -> int:
    started = time.time()
    config = json.loads(CONFIG.read_text())
    EVAL.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    dataset_paths = {name: ROOT / config["datasets"][name]["path"] for name in GROUP_ORDER}
    mask_sha = sha256_file(MASK)
    candidate_hashes = {name: sha256_file(path) for name, path in dataset_paths.items()}
    reference_sha = candidate_hashes["Reference"]

    deviations: list[str] = []
    if mask_sha != MASK_SHA256:
        deviations.append("mask_sha256_mismatch")
    if reference_sha != REFERENCE_SHA256:
        deviations.append("reference_sha256_mismatch")
    for name in CANDIDATE_ORDER:
        if candidate_hashes[name] != config["datasets"][name]["sha256"]:
            deviations.append("candidate_sha256_mismatch:%s" % name)
    if deviations:
        raise RuntimeError("frozen input hash mismatch: %s" % deviations)

    gate = {
        "schema": "p9016-round055-pre-reference-hash-gate-v1",
        "status": "PASS",
        "reference_opened": False,
        "phase_opened": False,
        "write_order": "this file is written before data/P9016.1m.3dg.gz is parsed",
        "mask": {"path": str(MASK.relative_to(ROOT)), "sha256": mask_sha},
        "datasets": {name: {"path": str(path.relative_to(ROOT)), "sha256": candidate_hashes[name],
                            "role": config["datasets"][name]["role"]}
                     for name, path in dataset_paths.items()},
        "reference_expected_sha256": config["datasets"]["Reference"]["sha256"],
        "deviations": [],
    }
    (EVAL / "pre_reference_hash_gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    # ---- reference is opened only past the hash gate
    reference = parse_3dg(REFERENCE)
    tracks = {"Reference": reference}
    for name in CANDIDATE_ORDER:
        tracks[name] = parse_3dg(dataset_paths[name])
    rg_values = {name: whole_cell_rg(table) for name, table in tracks.items()}
    bead_counts = {name: sum(len(table) for table in tracks[name].values())
                   for name in GROUP_ORDER}

    with np.load(MASK, allow_pickle=False) as payload:
        legacy = {key: payload[key] for key in payload.keys()}

    frozen_049, frozen_051 = read_frozen_regression()

    rows: list[dict[str, object]] = []
    support_audit: list[dict[str, object]] = []
    boxplot: dict[str, dict[str, list[float]]] = {
        name: {"same": [], "cross": [], "contrast": [], "same_spearman": [], "cross_spearman": [],
               "contrast_spearman": []} for name in GROUP_ORDER}
    chr1_panels: dict[str, object] = {}
    chr1_panel_scale: dict[str, float] = {}
    position_coverage: dict[str, list[int]] = {}
    position_total = 0

    for chromosome_index, chromosome in enumerate(CHROMOSOMES):
        key = "chr%d" % chromosome_index
        positions = np.asarray(legacy[key + "_positions"], dtype=np.int64)
        pair_i = np.asarray(legacy[key + "_pair_i"], dtype=np.int64)
        pair_j = np.asarray(legacy[key + "_pair_j"], dtype=np.int64)
        common = np.asarray(legacy[key + "_common"], dtype=bool)
        position_total += int(positions.size)

        ref_mat_name, ref_pat_name = reference_copy_names(reference, chromosome)
        track_names: dict[str, tuple[str, str]] = {"Reference": (ref_mat_name, ref_pat_name)}
        for name in CANDIDATE_ORDER:
            track_names[name] = ("c%02da" % (chromosome_index + 1),
                                 "c%02db" % (chromosome_index + 1))

        points: dict[str, list[np.ndarray]] = {}
        coverage: dict[str, list[int]] = {}
        for name in GROUP_ORDER:
            arrays = []
            present_per_copy = []
            for copy_index, track in enumerate(track_names[name]):
                table = tracks[name].get(track)
                if table is None:
                    raise RuntimeError("missing track %s in %s" % (track, name))
                values = np.full((positions.size, 3), np.nan, dtype=np.float64)
                present = 0
                for row_index, position in enumerate(positions.tolist()):
                    value = table.get(int(position))
                    if value is not None:
                        values[row_index] = value
                        present += 1
                arrays.append(values)
                present_per_copy.append(present)
            coverage[name] = present_per_copy
            points[name] = arrays
        for name in GROUP_ORDER:
            position_coverage.setdefault(name, [0, 0])
            position_coverage[name] = [position_coverage[name][copy_index] + coverage[name][copy_index]
                                       for copy_index in (0, 1)]

        shared = common.copy()
        dropped: dict[str, int] = {}
        for name in GROUP_ORDER:
            for copy_index in (0, 1):
                finite = (np.isfinite(points[name][copy_index][pair_i]).all(axis=1)
                          & np.isfinite(points[name][copy_index][pair_j]).all(axis=1))
                dropped[name] = dropped.get(name, 0) + int((~finite & shared).sum())
                shared &= finite
        used_i, used_j = pair_i[shared], pair_j[shared]
        n_shared = int(shared.sum())

        reference_distance = [distances(points["Reference"][copy_index], used_i, used_j)
                              for copy_index in (0, 1)]
        ref_mat_distance, ref_pat_distance = reference_distance
        ref_cross = pearson(ref_mat_distance, ref_pat_distance)
        ref_cross_spearman = spearman(ref_mat_distance, ref_pat_distance)

        if bool(np.array_equal(used_i, pair_i[common]) and np.array_equal(used_j, pair_j[common])):
            pass

        for name in GROUP_ORDER:
            if name == "Reference":
                rho = {"A_mat": 1.0, "A_pat": ref_cross, "B_mat": ref_cross, "B_pat": 1.0}
                rho_spearman = {"A_mat": 1.0, "A_pat": ref_cross_spearman,
                                "B_mat": ref_cross_spearman, "B_pat": 1.0}
                copy_distance = [ref_mat_distance, ref_pat_distance]
            else:
                distance_a = distances(points[name][0], used_i, used_j)
                distance_b = distances(points[name][1], used_i, used_j)
                rho = {"A_mat": pearson(distance_a, ref_mat_distance),
                       "A_pat": pearson(distance_a, ref_pat_distance),
                       "B_mat": pearson(distance_b, ref_mat_distance),
                       "B_pat": pearson(distance_b, ref_pat_distance)}
                rho_spearman = {"A_mat": spearman(distance_a, ref_mat_distance),
                                "A_pat": spearman(distance_a, ref_pat_distance),
                                "B_mat": spearman(distance_b, ref_mat_distance),
                                "B_pat": spearman(distance_b, ref_pat_distance)}
                copy_distance = [distance_a, distance_b]
            if chromosome_index == 0:
                # 显示尺度（复刻 scripts/plot_3dg_comparison.py 的历史热图口径）：
                # 该数据集 chr1 两个拷贝在共享 common pairs 上的距离合并取一个 raw median。
                chr1_panel_scale[name] = float(np.nanmedian(np.concatenate(copy_distance)))
            direct = (rho["A_mat"] + rho["B_pat"]) / 2.0
            swapped = (rho["A_pat"] + rho["B_mat"]) / 2.0
            derived = orient(direct, swapped)
            direct_s = (rho_spearman["A_mat"] + rho_spearman["B_pat"]) / 2.0
            swapped_s = (rho_spearman["A_pat"] + rho_spearman["B_mat"]) / 2.0
            derived_s = orient(direct_s, swapped_s)

            # 交换不变性：把候选两条 copy 整体互换后 same/cross 必须逐位不变
            if name != "Reference":
                swapped_rho = {"A_mat": rho["B_mat"], "A_pat": rho["B_pat"],
                               "B_mat": rho["A_mat"], "B_pat": rho["A_pat"]}
                swap_derived = orient((swapped_rho["A_mat"] + swapped_rho["B_pat"]) / 2.0,
                                      (swapped_rho["A_pat"] + swapped_rho["B_mat"]) / 2.0)
                swap_invariance_error = max(abs(swap_derived["same"] - derived["same"]),
                                            abs(swap_derived["cross"] - derived["cross"]),
                                            abs(swap_derived["contrast"] - derived["contrast"]))
            else:
                swap_invariance_error = 0.0

            rows.append({
                "chr": chromosome, "dataset": name, "n_pairs": n_shared,
                "n_pairs_legacy_common": int(common.sum()), "n_positions": int(positions.size),
                "A_mat": rho["A_mat"], "A_pat": rho["A_pat"], "B_mat": rho["B_mat"],
                "B_pat": rho["B_pat"], "direct": direct, "swapped": swapped,
                "same": derived["same"], "cross": derived["cross"],
                "contrast": derived["contrast"], "orientation": derived["orientation"],
                "same_spearman": derived_s["same"], "cross_spearman": derived_s["cross"],
                "contrast_spearman": derived_s["contrast"],
                "orientation_spearman": derived_s["orientation"],
                "swap_invariance_error": swap_invariance_error,
            })
            boxplot[name]["same"].append(float(derived["same"]))
            boxplot[name]["cross"].append(float(derived["cross"]))
            boxplot[name]["contrast"].append(float(derived["contrast"]))
            boxplot[name]["same_spearman"].append(float(derived_s["same"]))
            boxplot[name]["cross_spearman"].append(float(derived_s["cross"]))
            boxplot[name]["contrast_spearman"].append(float(derived_s["contrast"]))

            if chromosome_index == 0:
                chr1_panels.setdefault("panels", {})
                if name == "Reference":
                    panel_copies = (0, 1)  # mat, pat
                elif derived["orientation"] == "swapped":
                    panel_copies = (1, 0)  # copy B -> mat, copy A -> pat
                else:
                    panel_copies = (0, 1)
                chr1_panels["panels"][name] = {
                    "copies": list(panel_copies),
                    "orientation": derived["orientation"],
                    "coordinates": [points[name][copy_index].copy() for copy_index in panel_copies],
                }

        support_audit.append({
            "chr": chromosome,
            "n_pairs_legacy_common": int(common.sum()),
            "n_pairs_shared": n_shared,
            "n_pairs_dropped": int(common.sum()) - n_shared,
            "dropped_by_dataset": {name: int(count) for name, count in dropped.items()},
            "track_names": {name: list(track_names[name]) for name in GROUP_ORDER},
            "position_coverage": {name: [int(value) for value in coverage[name]]
                                  for name in GROUP_ORDER},
            "n_positions": int(positions.size),
        })

    # ---- chr1 heatmap matrices (raw distance / per-dataset chr1 common-pair median scale)
    chr1_bins = np.asarray(legacy["chr0_positions"], dtype=np.int64)
    matrices: dict[str, object] = {}
    finite_max = 0.0
    nan_counts: dict[str, list[int]] = {}
    for name in GROUP_ORDER:
        panels = []
        nan_count = []
        scale = chr1_panel_scale[name]
        for copy_index in (0, 1):
            coords = chr1_panels["panels"][name]["coordinates"][copy_index]
            delta = coords[:, None, :] - coords[None, :, :]
            with np.errstate(invalid="ignore"):
                matrix = np.sqrt(np.sum(delta * delta, axis=2)) / scale
            missing = ~np.isfinite(coords).all(axis=1)
            matrix[missing, :] = np.nan
            matrix[:, missing] = np.nan
            finite = np.isfinite(matrix)
            if finite.any():
                finite_max = max(finite_max, float(matrix[finite].max()))
            nan_count.append(int((~finite).sum()))
            panels.append(matrix)
        matrices[name] = np.stack(panels, axis=0)
        nan_counts[name] = nan_count

    np.savez_compressed(
        EVAL / "chr1_panels.npz",
        bins=chr1_bins,
        reference=matrices["Reference"],
        baseline=matrices["Baseline-046-G-random"],
        b_hard_observed=matrices["B-hard-observed"],
        c_max_rate=matrices["C-max-rate"],
        group_order=np.asarray(GROUP_ORDER, dtype=object),
        panel_scale_raw_median=np.asarray([chr1_panel_scale[name] for name in GROUP_ORDER],
                                          dtype=np.float64),
        whole_cell_rg=np.asarray([rg_values[name] for name in GROUP_ORDER], dtype=np.float64),
        unified_vmax=np.asarray([finite_max], dtype=np.float64),
    )

    # ---- regression checks against frozen records
    checks: list[dict[str, object]] = []

    def add_check(name: str, ok: bool, evidence: dict[str, object]) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "evidence": evidence})

    rows_by = {(row["dataset"], row["chr"]): row for row in rows}

    mapping_049 = {"B-hard-observed": "B-raw-consensus", "C-max-rate": "C-ms-random"}
    for dataset_name, frozen_name in mapping_049.items():
        worst_value, worst_key = 0.0, None
        pair_mismatch = 0
        for chromosome in CHROMOSOMES:
            frozen = frozen_049.get((frozen_name, chromosome))
            if frozen is None:
                pair_mismatch += 1
                continue
            mine = rows_by[(dataset_name, chromosome)]
            if int(frozen["n_pairs"]) != int(mine["n_pairs"]):
                pair_mismatch += 1
            for metric_name, mine_key in (("rho_A_mat", "A_mat"), ("rho_A_pat", "A_pat"),
                                          ("rho_B_mat", "B_mat"), ("rho_B_pat", "B_pat"),
                                          ("matched", "same"), ("cross", "cross")):
                error = abs(float(frozen[metric_name]) - float(mine[mine_key]))
                if error > worst_value:
                    worst_value, worst_key = error, "%s:%s" % (chromosome, metric_name)
        add_check("regression_049_r2_pearson_%s" % dataset_name,
                  worst_value < 1e-9 and pair_mismatch == 0,
                  {"frozen_table": str(R2_049.relative_to(ROOT)),
                   "max_abs_diff": worst_value, "worst_key": worst_key,
                   "n_pairs_or_row_mismatch": pair_mismatch, "tolerance": 1e-9})

    worst_value, worst_key, pair_mismatch = 0.0, None, 0
    for chromosome in CHROMOSOMES:
        frozen = frozen_051.get(chromosome)
        if frozen is None:
            pair_mismatch += 1
            continue
        mine = rows_by[("Baseline-046-G-random", chromosome)]
        if int(frozen["n_pairs"]) != int(mine["n_pairs"]):
            pair_mismatch += 1
        for metric_name, mine_key in (("same", "same"), ("cross", "cross"),
                                      ("contrast", "contrast"), ("A_mat", "A_mat"),
                                      ("A_pat", "A_pat"), ("B_mat", "B_mat"), ("B_pat", "B_pat")):
            error = abs(float(frozen[metric_name]) - float(mine[mine_key]))
            if error > worst_value:
                worst_value, worst_key = error, "%s:%s" % (chromosome, metric_name)
    add_check("regression_051_baseline_pearson", worst_value < 1e-9 and pair_mismatch == 0,
              {"frozen_table": str(PEARSON_051.relative_to(ROOT)), "max_abs_diff": worst_value,
               "worst_key": worst_key, "n_pairs_or_row_mismatch": pair_mismatch,
               "tolerance": 1e-9})

    baseline_same_mean = stats(boxplot["Baseline-046-G-random"]["same"])["mean"]
    add_check("regression_051_baseline_same_mean", abs(baseline_same_mean - 0.6023605799695414) < 1e-12,
              {"observed": baseline_same_mean, "published": 0.6023605799695414, "tolerance": 1e-12})

    summary_053 = ROOT / ("test_res/053-20260916T084905Z-four-way-spearman-chr1-matrices/"
                          "eval/summary.json")
    if summary_053.exists():
        frozen_053 = json.loads(summary_053.read_text())
        entry = frozen_053["per_version"]["baseline"]
        mine_spearman = {"same": stats(boxplot["Baseline-046-G-random"]["same_spearman"])["mean"],
                         "cross": stats(boxplot["Baseline-046-G-random"]["cross_spearman"])["mean"],
                         "contrast": stats(boxplot["Baseline-046-G-random"]["contrast_spearman"])["mean"]}
        errors = {"same": abs(entry["same"]["mean"] - mine_spearman["same"]),
                  "cross": abs(entry["cross"]["mean"] - mine_spearman["cross"]),
                  "contrast": abs(entry["difference_same_minus_cross"]["mean"] - mine_spearman["contrast"])}
        add_check("regression_053_baseline_spearman",
                  max(errors.values()) < 1e-12,
                  {"frozen_table": str(summary_053.relative_to(ROOT)),
                   "frozen_candidate_npz_sha256": frozen_053["candidates"]["Baseline"]["npz_sha256"],
                   "abs_diff": errors, "observed": mine_spearman,
                   "tolerance": 1e-12})
    else:
        add_check("regression_053_baseline_spearman", False,
                  {"error": "frozen 053 summary.json is missing"})

    swap_error = max(float(row["swap_invariance_error"]) for row in rows)
    add_check("copy_swap_invariance", swap_error == 0.0,
              {"max_abs_diff": swap_error, "rule": "swapping candidate copy labels leaves "
                                                   "same/cross/contrast unchanged"})

    add_check("support_matches_frozen_mask",
              sum(int(entry["n_pairs_shared"]) for entry in support_audit) == 157529,
              {"shared_total": sum(int(entry["n_pairs_shared"]) for entry in support_audit),
               "mask_common_total": 157529,
               "dropped_total": sum(int(entry["n_pairs_dropped"]) for entry in support_audit)})

    dropped_total = sum(int(entry["n_pairs_dropped"]) for entry in support_audit)
    add_check("all_20_chromosomes_defined",
              len({row["chr"] for row in rows}) == 20
              and all(math.isfinite(float(row["same"])) and math.isfinite(float(row["cross"]))
                      for row in rows),
              {"n_chromosomes": len({row["chr"] for row in rows}), "n_rows": len(rows)})

    add_check("reference_self_same_is_one",
              all(float(rows_by[("Reference", chromosome)]["same"]) == 1.0
                  for chromosome in CHROMOSOMES),
              {"note": "reference same is degenerate by construction; cross is the mat/pat "
                       "distance correlation on the same support",
               "cross_mean": stats(boxplot["Reference"]["cross"])["mean"]})

    symmetry_error = 0.0
    diagonal_error = 0.0
    zero_diagonal_missing = 0
    for name in GROUP_ORDER:
        for panel in matrices[name]:
            finite = np.isfinite(panel)
            diff = np.abs(panel[finite] - panel.T[finite])
            if diff.size:
                symmetry_error = max(symmetry_error, float(diff.max()))
            for index in range(panel.shape[0]):
                if finite[index, index]:
                    diagonal_error = max(diagonal_error, abs(float(panel[index, index])))
                else:
                    zero_diagonal_missing += 1
    add_check("chr1_matrix_symmetry_and_zero_diagonal",
              symmetry_error < 1e-12 and diagonal_error == 0.0,
              {"max_symmetry_error": symmetry_error, "max_abs_diagonal": diagonal_error,
               "missing_diagonal_entries": zero_diagonal_missing,
               "nan_counts": nan_counts, "unified_vmax": finite_max})

    candidate_position_coverage = {name: position_coverage[name] for name in CANDIDATE_ORDER}
    add_check("candidate_position_full_coverage",
              all(value == position_total
                  for values in candidate_position_coverage.values() for value in values),
              {"positions_total": position_total, "coverage_per_copy": candidate_position_coverage,
               "reference_coverage_per_copy": position_coverage["Reference"],
               "note": "mask positions map numerically onto 1Mb locus bp keys; no string "
                       "comparison and no compressed index are used"})

    add_check("chr1_panel_scale_positive",
              all(math.isfinite(value) and value > 0.0 for value in chr1_panel_scale.values()),
              {"panel_scale_raw_median": chr1_panel_scale,
               "historical_convention": "scripts/plot_3dg_comparison.py build_matrix lines 268-275 "
                                        "(combined median of the two copies' common-pair distances)",
               "note": "display normalization only; it does not enter same/cross/contrast"})

    validation = {
        "schema": "p9016-round055-validation-v1",
        "status": "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL",
        "checks": checks,
        "deviations": [],
        "reference_first_opened_utc": None,
        "notes": [
            "reference was parsed only after eval/pre_reference_hash_gate.json was written",
            "epoch time of this run is recorded in logs/terminal.json",
            "heatmap scale corrected before any figure existed: per-dataset chr1 two-copy combined "
            "raw median replaces the whole-cell Rg wording recorded earlier in config.json "
            "(see config.json heatmap_scale_correction); same/cross/contrast are unaffected",
        ],
    }
    validation["reference_first_opened_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))

    # ---- summary
    summary = {
        "schema": "p9016-round055-summary-v1",
        "run_id": config["run_id"],
        "metric": "signed Pearson of pairwise Euclidean distances; same = max(direct, swapped), "
                  "cross = min(direct, swapped), contrast = same - cross",
        "group_order": GROUP_ORDER,
        "support": {
            "mask": str(MASK.relative_to(ROOT)),
            "mask_sha256": mask_sha,
            "denominator": "per-chromosome legacy mask common pairs intersected with finite beads "
                           "in all four datasets and both copies",
            "pairs_total": sum(int(entry["n_pairs_shared"]) for entry in support_audit),
            "mask_common_total": 157529,
            "pairs_dropped": dropped_total,
            "chromosomes": 20,
        },
        "whole_cell_rg": rg_values,
        "bead_counts": bead_counts,
        "per_group_pearson": {
            name: {"same": stats(boxplot[name]["same"]), "cross": stats(boxplot[name]["cross"]),
                   "contrast": stats(boxplot[name]["contrast"])} for name in GROUP_ORDER},
        "per_group_spearman": {
            name: {"same": stats(boxplot[name]["same_spearman"]),
                   "cross": stats(boxplot[name]["cross_spearman"]),
                   "contrast": stats(boxplot[name]["contrast_spearman"])} for name in GROUP_ORDER},
        "chr1_readout": {name: {"same": rows_by[(name, "chr1")]["same"],
                                "cross": rows_by[(name, "chr1")]["cross"],
                                "contrast": rows_by[(name, "chr1")]["contrast"],
                                "orientation": rows_by[(name, "chr1")]["orientation"],
                                "n_pairs": rows_by[(name, "chr1")]["n_pairs"]}
                         for name in GROUP_ORDER},
        "chr1_panels": {
            "bins": int(chr1_bins.size),
            "unified_vmax": finite_max,
            "nan_counts": nan_counts,
            "panel_scale_raw_median": chr1_panel_scale,
            "whole_cell_rg_secondary": rg_values,
            "normalization": "each dataset's chr1 two copies combined, one raw median over their "
                             "shared common-pair distances (historical build_matrix convention in "
                             "scripts/plot_3dg_comparison.py); whole-cell Rg is kept only as a "
                             "secondary scatter readout and is not the heatmap scale",
            "columns": "column 0 = copy mapped to reference mat, column 1 = copy mapped to "
                       "reference pat (whole-chromosome orientation for candidates)",
        },
        "support_audit": support_audit,
        "scientific_boundary": {
            "refit": False, "bootstrap": False, "R1_R3": False,
            "terminal_states": {name: config["datasets"][name].get("terminal") for name in GROUP_ORDER},
            "n_biological_replicates": 1,
            "l2_proven": False,
            "note": "descriptive endpoint comparison on one cell; 20 chromosomes are correlated "
                    "measurements, not biological replicates; all endpoints are budget_not_converged",
        },
    }

    columns = ["chr", "dataset", "n_pairs", "n_pairs_legacy_common", "n_positions",
               "A_mat", "A_pat", "B_mat", "B_pat", "direct", "swapped", "same", "cross",
               "contrast", "orientation", "same_spearman", "cross_spearman",
               "contrast_spearman", "orientation_spearman", "swap_invariance_error"]
    tsv_lines = ["\t".join(columns)]
    for row in rows:
        tsv_lines.append("\t".join(
            ("%.12g" % row[column]) if isinstance(row[column], float) else str(row[column])
            for column in columns))
    (EVAL / "per_chromosome.tsv").write_text("\n".join(tsv_lines) + "\n")

    (EVAL / "boxplot_data.json").write_text(json.dumps({
        "schema": "p9016-round055-boxplot-data-v1",
        "group_order": GROUP_ORDER,
        "chromosomes": CHROMOSOMES,
        "pearson": {name: {key: boxplot[name][key] for key in
                           ("same", "cross", "contrast")} for name in GROUP_ORDER},
        "spearman": {name: {key: boxplot[name][key] for key in
                            ("same_spearman", "cross_spearman", "contrast_spearman")}
                     for name in GROUP_ORDER},
        "reference_control_note": "Reference same is 1 by construction (self-correlation); "
                                  "Reference cross is the mat/pat distance correlation per chromosome",
    }, indent=2) + "\n")
    (EVAL / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (EVAL / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    terminal = {
        "schema": "p9016-round055-terminal-v1",
        "run_id": config["run_id"],
        "status": "complete" if validation["status"] == "PASS" else "complete_with_failed_checks",
        "validation_status": validation["status"],
        "checks_total": len(checks),
        "checks_failed": [check["check"] for check in checks if check["status"] != "PASS"],
        "steps": {"refit": 0, "bootstrap": 0, "reference_parsed": True, "phase_parsed": False},
        "wall_seconds": time.time() - started,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (LOGS / "terminal.json").write_text(json.dumps(terminal, indent=2) + "\n")

    print("validation:", validation["status"])
    for check in checks:
        print("  %-46s %s" % (check["check"], check["status"]))
    for name in GROUP_ORDER:
        same = stats(boxplot[name]["same"])
        cross = stats(boxplot[name]["cross"])
        contrast = stats(boxplot[name]["contrast"])
        print("%-24s same %.6f | cross %.6f | contrast %.6f | n %d" % (
            name, same["mean"], cross["mean"], contrast["mean"], same["n_defined"]))
    print("chr1 same:", {name: round(float(rows_by[(name, "chr1")]["same"]), 6) for name in GROUP_ORDER})
    print("shared pairs:", summary["support"]["pairs_total"], "dropped:", dropped_total)
    print("rg:", rg_values)
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
