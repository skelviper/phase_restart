"""051 统一评价：per-chromosome signed Pearson（same/cross, best swap）与绘图输入。

评价进程（唯一读 reference 的进程）在这里做：

1. 读 4 组 1Mb 坐标（baseline / A / B / C）与 reference 3DG。
2. 用 046 冻结 legacy 21-mask 的 public support：每 chr 的 positions / pair_i / pair_j /
   common，与 4 组坐标全部一致（C 的 1Mb 支持已确认覆盖该 support）。
3. 每 chr 计算 4 个 signed Pearson（A_mat/A_pat/B_mat/B_pat）；
   same = (A_best + B_other)/2、cross = (A_other + B_best)/2，best 由
   (A_mat+B_pat) vs (A_pat+B_mat) 的 whole-chr 比较决定（不是逐 bin 局部调标签）。
4. 输出 per_chromosome.tsv（含 NA）与 fig2 数据；坐标缺失/无定义一律 NA，不填 0。

索引映射：global = offset + positions // 1Mb，positions 来自 mask 的真实数值 bp，
不使用压缩下标。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
for _path in (str(S049),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from frozen_imports import data_io  # noqa: E402
from round_runner import sha256_file, write_json  # noqa: E402

REFERENCE = ROOT / "data/P9016.1m.3dg.gz"
REFERENCE_SHA = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
LEGACY_MASK = ROOT / ("test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/"
                      "frozen_legacy_mask_snapshot.npz")
AGGREGATE_1MB = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/inputs/real_1000000_aggregate.npz"
CONDITIONS = {
    "baseline": ROOT / "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.3dg",
    "extra-levels": RUN / "coords/A-extra-levels/1Mb.3dg",
    "no-bend": RUN / "coords/B-no-bend/1Mb.3dg",
    "reference-beads": RUN / "coords/C-reference-beads/1Mb.3dg",
}


def parse_3dg(path: Path) -> dict[str, dict[int, np.ndarray]]:
    tracks: dict[str, dict[int, np.ndarray]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 5:
                continue
            track = fields[0]
            position = int(fields[1])
            tracks.setdefault(track, {})[position] = np.asarray(
                [float(value) for value in fields[2:5]], dtype=np.float64)
    return tracks


def reference_copy_order(tracks: dict[str, dict[int, np.ndarray]], chromosome: str) -> tuple[str, str]:
    """返回 (copy0, copy1) track 名；copy0 表示文件中先出现的 mat/pat 归属。"""
    first_seen = []
    for track in tracks:
        match = re.match(r"^%s\((mat|pat)\)$" % re.escape(chromosome), track)
        if match and track not in first_seen:
            first_seen.append(track)
    if len(first_seen) != 2:
        raise RuntimeError("reference chromosome %s does not have exactly two copies" % chromosome)
    return first_seen[0], first_seen[1]


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


def distances(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    delta = points[pair_i] - points[pair_j]
    return np.sqrt(np.sum(delta * delta, axis=1))


def evaluate(condition_paths: dict[str, Path]) -> dict[str, object]:
    if sha256_file(REFERENCE) != REFERENCE_SHA:
        raise RuntimeError("reference 3DG SHA256 mismatch")
    data = data_io.load_aggregate(AGGREGATE_1MB)
    names = [str(name) for name in data.chromosome_names]
    offsets = np.asarray(data.offsets, dtype=np.int64)
    with np.load(LEGACY_MASK, allow_pickle=False) as payload:
        legacy = {key: payload[key] for key in payload.keys()}
    candidate_hashes = {name: sha256_file(path) for name, path in condition_paths.items()}
    write_json(RUN / "evaluation" / "candidate_hash_gate.json",
               {"schema": "p9016-round051-candidate-hash-gate-v1",
                "hashed_before_reference_read": True,
                "candidates": {name: {"path": str(path.relative_to(ROOT)), "sha256": digest}
                               for (name, path), digest in zip(condition_paths.items(),
                                                               candidate_hashes.values())},
                "reference_path": str(REFERENCE.relative_to(ROOT)),
                "reference_sha256_expected": REFERENCE_SHA})
    reference = parse_3dg(REFERENCE)
    candidate_tracks = {name: parse_3dg(path) for name, path in condition_paths.items()}
    rows: list[dict[str, object]] = []
    figure: dict[str, dict[str, list[float]]] = {name: {"same": [], "cross": []}
                                                 for name in condition_paths}
    figure_legacy: dict[str, dict[str, list[float]]] = {name: {"same": [], "cross": []}
                                                        for name in condition_paths}
    matrix_input: dict[str, object] = {}
    support_audit: list[dict[str, object]] = []
    for chromosome_index, chromosome in enumerate(names):
        key = "chr%d" % chromosome_index
        positions = np.asarray(legacy[key + "_positions"], dtype=np.int64)
        pair_i = np.asarray(legacy[key + "_pair_i"], dtype=np.int64)
        pair_j = np.asarray(legacy[key + "_pair_j"], dtype=np.int64)
        common = np.asarray(legacy[key + "_common"], dtype=bool)
        ref_tracks = reference_copy_order(reference, chromosome)
        grid_index = positions // 1_000_000 + int(offsets[chromosome_index])
        used_i, used_j = pair_i[common], pair_j[common]
        legacy_common_pairs = int(common.sum())

        def points_of(tracks, copy, track_name):
            table = tracks.get(track_name)
            if table is None:
                raise RuntimeError("missing track %s" % track_name)
            values = np.full((len(positions), 3), np.nan, dtype=np.float64)
            for index, global_index in enumerate(grid_index.tolist()):
                position = int(data.locus_bin[global_index]) * 1_000_000
                if position in table:
                    values[index] = table[position]
            return values

        ref_points = [points_of(reference, copy, ref_tracks[copy]) for copy in (0, 1)]
        candidate_points = {}
        for name, tracks in candidate_tracks.items():
            candidate_points[name] = [
                points_of(tracks, copy, "c%02d%s" % (chromosome_index + 1, "ab"[copy]))
                for copy in (0, 1)]

        # 统一公共支持：同一 chr 上所有条件与 reference 共用同一 pair 集合。任何条件的
        # 珠子在某个 pair 端点缺失时，该 pair 从所有条件的同一分母中剔除并单独计数。
        union_ok = np.ones(used_i.shape[0], dtype=bool)
        loss_rows = {}
        for name, points in candidate_points.items():
            for copy in (0, 1):
                finite = (np.isfinite(points[copy][used_i]).all(axis=1)
                          & np.isfinite(points[copy][used_j]).all(axis=1))
                loss_rows[name] = loss_rows.get(name, 0) + int((~finite & union_ok).sum())
                union_ok &= finite
        for copy in (0, 1):
            finite = (np.isfinite(ref_points[copy][used_i]).all(axis=1)
                      & np.isfinite(ref_points[copy][used_j]).all(axis=1))
            union_ok &= finite
        n_used = int(union_ok.sum())
        support_audit.append({
            "chr": chromosome,
            "n_pairs_legacy_common": legacy_common_pairs,
            "n_pairs_shared_support": n_used,
            "n_pairs_dropped": legacy_common_pairs - n_used,
            "dropped_by_condition_first_seen": {name: int(count) for name, count in loss_rows.items()},
        })
        legacy_finite = np.ones(used_i.shape[0], dtype=bool)
        for copy in (0, 1):
            legacy_finite &= (np.isfinite(ref_points[copy][used_i]).all(axis=1)
                              & np.isfinite(ref_points[copy][used_j]).all(axis=1))
        if n_used == 0:
            for name in candidate_points:
                rows.append({"chr": chromosome, "condition": name, "same": float("nan"),
                             "cross": float("nan"), "contrast": float("nan"), "n_pairs": 0,
                             "n_pairs_legacy": legacy_common_pairs,
                             "n_pairs_lost_missing_bead": legacy_common_pairs,
                             "mapping": "undefined", "n_positions": int(len(positions)),
                             "A_mat": float("nan"), "A_pat": float("nan"),
                             "B_mat": float("nan"), "B_pat": float("nan")})
            continue
        si, sj = used_i[union_ok], used_j[union_ok]
        ref_mat = distances(ref_points[0], si, sj)
        ref_pat = distances(ref_points[1], si, sj)
        legacy_stats = {}
        if bool(legacy_finite.all()):
            li, lj = used_i, used_j
            l_ref_mat = distances(ref_points[0], li, lj)
            l_ref_pat = distances(ref_points[1], li, lj)
            for name, points in candidate_points.items():
                l_a = distances(points[0], li, lj)
                l_b = distances(points[1], li, lj)
                l_rho = {"A_mat": pearson(l_a, l_ref_mat), "A_pat": pearson(l_a, l_ref_pat),
                         "B_mat": pearson(l_b, l_ref_mat), "B_pat": pearson(l_b, l_ref_pat)}
                l_direct = l_rho["A_mat"] + l_rho["B_pat"]
                l_swapped = l_rho["A_pat"] + l_rho["B_mat"]
                if l_direct >= l_swapped:
                    l_same = l_direct / 2.0
                    l_cross = l_swapped / 2.0
                else:
                    l_same = l_swapped / 2.0
                    l_cross = l_direct / 2.0
                legacy_stats[name] = {
                    "same": l_same, "cross": l_cross, "contrast": l_same - l_cross,
                    "mapping": ("unresolved_tie" if abs(l_direct - l_swapped) <= 1e-12
                                else ("direct" if l_direct > l_swapped else "swapped")),
                    "rho": l_rho, "n_pairs": int(legacy_finite.sum()),
                }
                if not all(math.isfinite(value) for value in l_rho.values()):
                    legacy_stats[name]["same"] = float("nan")
                    legacy_stats[name]["cross"] = float("nan")
                    legacy_stats[name]["contrast"] = float("nan")
                    legacy_stats[name]["mapping"] = "undefined"
        for name, points in candidate_points.items():
            cand_a = distances(points[0], si, sj)
            cand_b = distances(points[1], si, sj)
            rho = {"A_mat": pearson(cand_a, ref_mat), "A_pat": pearson(cand_a, ref_pat),
                   "B_mat": pearson(cand_b, ref_mat), "B_pat": pearson(cand_b, ref_pat)}
            direct = rho["A_mat"] + rho["B_pat"]
            swapped = rho["A_pat"] + rho["B_mat"]
            if not all(math.isfinite(value) for value in rho.values()):
                mapping = "undefined"
                same = cross = contrast = float("nan")
            elif abs(direct - swapped) <= 1e-12:
                mapping = "unresolved_tie"
                same = (rho["A_mat"] + rho["B_pat"]) / 2.0
                cross = (rho["A_pat"] + rho["B_mat"]) / 2.0
                contrast = same - cross
            elif direct > swapped:
                mapping = "direct"
                same = (rho["A_mat"] + rho["B_pat"]) / 2.0
                cross = (rho["A_pat"] + rho["B_mat"]) / 2.0
                contrast = same - cross
            else:
                mapping = "swapped"
                same = (rho["A_pat"] + rho["B_mat"]) / 2.0
                cross = (rho["A_mat"] + rho["B_pat"]) / 2.0
                contrast = same - cross
            legacy_denominator = legacy_stats.get(name, {})
            legacy_same = legacy_denominator.get("same", float("nan"))
            if math.isfinite(legacy_same):
                figure_legacy[name]["same"].append(legacy_same)
                figure_legacy[name]["cross"].append(legacy_denominator["cross"])
            rows.append({
                "chr": chromosome, "condition": name,
                "same": same, "cross": cross, "contrast": contrast,
                "n_pairs": n_used, "n_pairs_legacy": legacy_common_pairs,
                "n_pairs_lost_missing_bead": legacy_common_pairs - n_used,
                "mapping": mapping, "n_positions": int(len(positions)),
                "A_mat": rho["A_mat"], "A_pat": rho["A_pat"],
                "B_mat": rho["B_mat"], "B_pat": rho["B_pat"],
                "same_legacy": legacy_same,
                "cross_legacy": legacy_denominator.get("cross", float("nan")),
                "contrast_legacy": legacy_denominator.get("contrast", float("nan")),
                "mapping_legacy": legacy_denominator.get("mapping", "undefined"),
            })
            if math.isfinite(same) and math.isfinite(cross):
                figure[name]["same"].append(same)
                figure[name]["cross"].append(cross)
            if chromosome_index == 0:
                matrix_input.setdefault(name, []).append({
                    "candidate_copyA": cand_a, "candidate_copyB": cand_b,
                    "reference_mat": ref_mat, "reference_pat": ref_pat,
                    "pair_i": si, "pair_j": sj, "positions": positions,
                    "mapping": mapping, "n_pairs": n_used,
                })
                matrix_input.setdefault("chr1_mapping", {})[name] = mapping
                matrix_input.setdefault("chr1_rho", {})[name] = {
                    key: float(value) for key, value in rho.items()}
        if chromosome_index == 0:
            matrix_input["reference_copy_order"] = list(ref_tracks)
    write_json(RUN / "evaluation" / "shared_support_audit.json",
               {"schema": "p9016-round051-shared-support-v1",
                "rule": "per-chromosome shared support = frozen legacy 21-mask common pairs "
                        "intersected with the beads all four conditions and the reference keep; "
                        "dropped pairs are reported, never silently removed",
                "chromosomes": support_audit,
                "total_pairs_legacy_common": int(sum(row["n_pairs_legacy_common"]
                                                     for row in support_audit)),
                "total_pairs_shared_support": int(sum(row["n_pairs_shared_support"]
                                                      for row in support_audit))})
    return {"rows": rows, "figure": figure, "figure_legacy": figure_legacy, "chr1": matrix_input,
            "reference_sha256": REFERENCE_SHA,
            "support_audit": support_audit,
            "legacy_mask_path": str(LEGACY_MASK.relative_to(ROOT))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(RUN / "evaluation"))
    args = parser.parse_args()
    missing = {name: str(path.relative_to(ROOT)) for name, path in CONDITIONS.items() if not path.is_file()}
    if missing:
        raise SystemExit("missing candidate coordinates: %s" % json.dumps(missing))
    result = evaluate(CONDITIONS)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    header = ["chr", "condition", "same", "cross", "contrast", "n_pairs", "mapping",
              "n_positions", "A_mat", "A_pat", "B_mat", "B_pat",
              "same_legacy", "cross_legacy", "contrast_legacy", "mapping_legacy",
              "n_pairs_legacy", "n_pairs_lost_missing_bead"]
    lines = ["\t".join(header)]
    for row in result["rows"]:
        values = []
        for column in header:
            value = row[column]
            if isinstance(value, float):
                values.append("NA" if not math.isfinite(value) else "%.12g" % value)
            else:
                values.append(str(value))
        lines.append("\t".join(values))
    (outdir / "per_chromosome.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {}
    for name, values in result["figure"].items():
        for key in ("same", "cross"):
            array = np.asarray(values[key], dtype=np.float64)
            finite = np.isfinite(array)
            # 每组保留全部 20 项；若任一 chr 为 NA，主 mean/median 记 None，
            # defined-only 统计另列，绝不丢行后静默 nanmean。
            defined_only = array[finite]
            summary.setdefault(name, {})[key] = {
                "n_chromosomes": int(array.size),
                "n_defined_chr": int(finite.sum()),
                "n_undefined_chr": int((~finite).sum()),
                "mean": float(array.mean()) if bool(finite.all()) else None,
                "median": float(np.median(array)) if bool(finite.all()) else None,
                "mean_defined_only": float(defined_only.mean()) if defined_only.size else None,
                "median_defined_only": float(np.median(defined_only)) if defined_only.size else None,
                "values": [None if not math.isfinite(value) else float(value) for value in array],
            }
    summary_legacy = {}
    for name, values in result["figure_legacy"].items():
        for key in ("same", "cross"):
            array = np.asarray(values[key], dtype=np.float64)
            finite = np.isfinite(array)
            summary_legacy.setdefault(name, {})[key] = {
                "n_chromosomes": int(array.size), "n_defined_chr": int(finite.sum()),
                "n_undefined_chr": int((~finite).sum()),
                "mean": float(array.mean()) if bool(finite.all()) else None,
                "mean_defined_only": float(array[finite].mean()) if finite.any() else None,
                "values": [None if not math.isfinite(value) else float(value) for value in array],
            }
    write_json(outdir / "pearson_summary.json", {
        "schema": "p9016-round051-pearson-summary-v2",
        "metric": "per-chromosome signed Pearson of intra-chromosomal distance vectors "
                  "on the frozen legacy 21-mask public support",
        "unit_of_observation": "chromosome (20 correlated measurements of one cell, not biological replicates)",
        "conditions": list(CONDITIONS),
        "summary": summary,
        "summary_legacy_denominator": summary_legacy,
        "primary_denominator": "shared support: frozen legacy 21-mask common pairs intersected "
                               "with the beads all four conditions and the reference keep "
                               "(149094 of 157529 pairs); see shared_support_audit.json",
        "secondary_denominator": "legacy: frozen 21-mask common pairs where the reference has "
                                 "finite coordinates (reproduces the published baseline "
                                 "matched Pearson 0.6023605799695414 on the chromosomes where it "
                                 "is defined)",
        "reference_sha256": result["reference_sha256"],
        "legacy_mask_path": result["legacy_mask_path"],
    })
    arrays = {}
    for name in CONDITIONS:
        entries = result["chr1"].get(name)
        if not entries:
            continue
        for key in ("candidate_copyA", "candidate_copyB", "reference_mat", "reference_pat",
                    "pair_i", "pair_j", "positions"):
            arrays["%s_%s" % (name, key)] = np.asarray(entries[0][key])

    np.savez_compressed(outdir / "chr1_matrix_input.npz", **arrays)
    write_json(outdir / "chr1_matrix_meta.json", {
        "reference_copy_order": result["chr1"]["reference_copy_order"],
        "conditions": list(CONDITIONS),
        "chr1_mapping": result["chr1"]["chr1_mapping"],
        "chr1_rho": result["chr1"]["chr1_rho"],
        "panels": "2 rows (matched-to-mat / matched-to-pat) x 5 columns "
                  "(Reference / baseline / extra-levels / no-bend / reference-beads)",
        "scale_rule": "one optimal positive scale per condition on chr1, fitted jointly over both "
                      "copies by least squares to the reference distance vectors, then multiplied "
                      "(not divided); reference scale fixed at 1; uniform scaling does not change "
                      "Pearson.",
    })
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
