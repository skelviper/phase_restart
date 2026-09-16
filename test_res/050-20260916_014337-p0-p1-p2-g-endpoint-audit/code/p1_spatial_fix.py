"""050 轮 P1 空间评价独立修正（evaluation 侧，允许读 reference，禁止 phase）。

背景：原 `code/p1_eval.py` 的 `centre_reads` 有三个问题（本脚本不修改它，也不改任何原结果）：

1. 该版本调用时没有传 `finite_loci`，默认 2645 个 locus 全 True → reference 在旧 21 mask
   之外是 NaN → 190/760 的 Pearson 全为 NaN（原 summary 里 `n_common_finite_loci = 2645`、
   所有空间读出为 null）。
2. copy 中心没有做**逐 chr 整条染色体的几何 A/B 最佳互换**，直接按 copy0/copy1 固定配对。
3. stress 用 z-score 标准化后的 RMS，而不是 049 的**单一最优尺度** stress。

本脚本只做评价侧的空间读出补算：不重拟合、不重算 R2/inter/count、不改 probe/候选坐标。
支撑固定为 046 冻结 old21 mask 的 2447 个 valid loci（4894 beads），
global locus = chromosome_offset + positions // 1Mb（mask local index 不是 0-origin）。

复用（只读）`test_res/049-.../evaluation/source/` 的
`eval049_spatial.merged_chr_centers / center_metrics / copy_center_metrics`、
`eval049_evaluate.load_real_masks / assert_reference_finite_on_mask`、`eval049_lib`；
不新增空间定义。

顺序：先验证 7 个 source 的 3dg / 坐标数组 hash、119 个 null 的 npz / 坐标 hash、
mask snapshot hash 与 reference **字节** hash，再解析 reference。
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
ROOT = RUN_DIR.parents[1]
S049_EVAL = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/evaluation"
for _path in (str(S049_EVAL / "source"),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import eval049_evaluate as ev049  # noqa: E402
import eval049_lib as lib  # noqa: E402
import eval049_spatial as spatial  # noqa: E402

GATE = RUN_DIR / "evaluation/pre_reference_gate.json"
OLD_SUMMARY = RUN_DIR / "evaluation/results/p1_evaluation_summary.json"
RESULTS = RUN_DIR / "evaluation/results"
FROZEN_049_EVALUATION = S049_EVAL / "results/evaluation.json"
FROZEN_049_SPATIAL_TSV = S049_EVAL / "results/spatial_summary.tsv"
MASK_SNAPSHOT_SHA256 = "9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9"
REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
EXPECTED_VALID_BINS = 2447
EXPECTED_VALID_BEADS = 4894
EXPECTED_MERGED_PAIRS = 190
EXPECTED_COPY_PAIRS = 760
EXPECTED_SAME_CHR_HOMOLOG_PAIRS = 20
REGRESSION_SOURCE = "046-work-baseline-G-full-J"
REGRESSION_FROZEN_DATASET = "baseline-046-real-extension-G-full-J"
REGRESSION_TOL = 1e-12
REGISTRY_QUOTED_BY_PARENT = {"center_pearson_190": 0.370639112, "center_normalized_stress": 0.314802114}
INVARIANCE_SOURCE = "A-raw-G-consensus"


def null_key(source: str, null_kind: str, seed: Any) -> str:
    """null 的唯一键：必须含 seed，否则同一 source 的 16 个 random-u draw 会互相覆盖。"""
    return "%s__%s__%s" % (source, null_kind, "u-zero" if seed is None else "seed%d" % int(seed))


def resolve(path_text: str) -> Path:
    """gate 里 source 路径是 root-relative，null 路径是 run-relative。"""
    candidate = ROOT / path_text
    if candidate.exists():
        return candidate
    candidate = RUN_DIR / path_text
    if candidate.exists():
        return candidate
    raise RuntimeError("path from gate does not exist: %s" % path_text)


def slim_merged(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_centers": int(metrics["n_centers"]), "n_distances": int(metrics["n_distances"]),
        "pearson": float(metrics["pearson"]), "spearman": float(metrics["spearman"]),
        "optimal_single_scale": float(metrics["optimal_single_scale"]),
        "normalized_stress": float(metrics["normalized_stress"]),
        "per_chr_profile_macro_pearson": metrics["per_chr_profile_macro_pearson"],
        "per_chr_profile_macro_spearman": metrics["per_chr_profile_macro_spearman"],
        "top3_neighbor_overlap_mean": float(metrics["top3_neighbors"]["mean_overlap_top3"]),
        "scale_policy": metrics["scale_policy"],
    }


def slim_copy(metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        "n_distances": int(metrics["n_distances"]), "pearson": float(metrics["pearson"]),
        "spearman": float(metrics["spearman"]),
        "optimal_single_scale": float(metrics["optimal_single_scale"]),
        "normalized_stress": float(metrics["normalized_stress"]),
        "procrustes_proper_normalized_rmsd":
            float(metrics["procrustes_proper"]["normalized_aligned_rmsd"]),
        "procrustes_proper_det": float(metrics["procrustes_proper"]["rotation_det"]),
        "procrustes_reflection_normalized_rmsd":
            float(metrics["procrustes_reflection"]["normalized_aligned_rmsd"]),
    }


def spatial_reads(coords: np.ndarray, reference: np.ndarray, ref_merged: np.ndarray,
                  per_chr_valid: Sequence[np.ndarray], masks: Mapping[str, Mapping[str, Any]],
                  data: Any) -> dict[str, Any]:
    """用 049 helpers 计算 190（20 合并中心）与 760（40 copy 中心跨 chr）读出。

    copy gauge = 逐 chr 整条染色体 signed Pearson 的 `derive_rho` 最佳互换（tie<=1e-12）。
    gauge 未定义/打平时**不人工定方向**：主读出为 NA，同时把 049 的 identity fallback
    数值放进明确命名的副字段，便于审计但不冒充主读数。
    """
    names = list(data.chromosome_names)
    merged, _copy_centres = spatial.merged_chr_centers(coords, per_chr_valid)
    merged_metrics = spatial.center_metrics(merged, ref_merged, names)
    copy_metrics = spatial.copy_center_metrics(coords, reference, per_chr_valid, masks, names, data.offsets)
    unresolved = [str(name) for name in copy_metrics["unresolved_chromosomes"]]
    identity_fallback = slim_copy(copy_metrics["primary"]) if unresolved else None
    alternative = slim_copy(copy_metrics["unresolved_alternative"]) if unresolved else None
    primary = None if unresolved else slim_copy(copy_metrics["primary"])
    mapping = {str(k): {"orientation": v["orientation"], "mapping_defined": bool(v["mapping_defined"]),
                        "reason": v["reason"], "copy0_to_reference": v["copy0_to_reference"],
                        "copy1_to_reference": v["copy1_to_reference"],
                        "direct": v["direct"], "swapped": v["swapped"], "contrast": v["contrast"]}
               for k, v in copy_metrics["per_chromosome_mapping"].items()}
    return {
        "merged_centres_190": slim_merged(merged_metrics),
        "copy_centres_760": primary,
        "copy_centres_760_identity_fallback_when_gauge_unresolved": identity_fallback,
        "copy_centres_760_unresolved_alternative": alternative,
        "copy_centre_gauge_unresolved_chromosomes": unresolved,
        "copy_centre_gauge_defined": not unresolved,
        "copy_centre_primary_policy": ("per-chromosome whole-chromosome signed-Pearson derive_rho best A/B swap "
                                       "(tie<=1e-12); when the gauge is undefined/ambiguous the primary 760 readout "
                                       "is NA and no direction is assigned by hand"),
        "per_chromosome_mapping": mapping,
        "merged_per_chr_19_distance_profile_pearson":
            merged_metrics["per_chr_19_distance_profile_pearson"],
        "merged_per_chr_19_distance_profile_spearman":
            merged_metrics["per_chr_19_distance_profile_spearman"],
        "merged_top3_overlap_per_chromosome": merged_metrics["top3_neighbors"]["per_chromosome"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="050 P1 independent spatial-evaluation correction "
                                                 "(190 merged / 760 cross-chr copy centres, 049 helpers)")
    parser.parse_args()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    gate = lib.read_json(GATE)
    if gate.get("status") != "PASS" or gate.get("reference_opened") is not False:
        raise RuntimeError("050 pre-reference gate is not in the expected pre-reference PASS state")

    data = lib.Aggregate()
    aggregate_audit = data.audit()

    # ---------------------------------------------------------- 1) hash 验证（不解析 reference）
    source_records: list[dict[str, Any]] = []
    source_coords: dict[str, np.ndarray] = {}
    for entry in gate["sources"]:
        path = resolve(str(entry["3dg"]))
        file_sha = lib.sha256_file(path)
        tracks = lib.load_3dg(path)
        audit: dict[str, Any] = {}
        coords = lib.three_dg_to_array(tracks, data, track_mode="candidate", audit=audit)
        if not np.isfinite(coords).all():
            raise RuntimeError("source coordinates are not full-grid finite: %s" % path)
        array_sha = lib.hash_array(coords)
        record = {"id": str(entry["id"]), "kind": str(entry["kind"]), "3dg": str(path.relative_to(ROOT)),
                  "3dg_sha256": file_sha, "3dg_sha256_expected": str(entry["3dg_sha256"]),
                  "coordinate_array_sha256": array_sha,
                  "coordinate_array_sha256_expected": str(entry["coordinate_array_sha256"]),
                  "file_sha_match": bool(file_sha == entry["3dg_sha256"]),
                  "array_sha_match": bool(array_sha == entry["coordinate_array_sha256"]),
                  "nonfinite_loci": int(audit["nonfinite_loci"]),
                  "records": len(tracks)}
        if not (record["file_sha_match"] and record["array_sha_match"]):
            raise RuntimeError("source hash mismatch: %s" % record)
        source_records.append(record)
        source_coords[str(entry["id"])] = coords

    null_records: list[dict[str, Any]] = []
    null_coords: dict[str, np.ndarray] = {}
    for entry in gate["nulls"]:
        path = resolve(str(entry["path"]))
        file_sha = lib.sha256_file(path)
        with np.load(path, allow_pickle=False) as payload:
            coords = np.asarray(payload["coordinates"], dtype=np.float64)
        array_sha = lib.hash_array(coords)
        # 唯一键必须含 seed：只看 (source, null_kind) 会把同一 source 的 16 个 random-u draw 覆盖成 1 个
        key = null_key(str(entry["source"]), str(entry["null_kind"]), entry["seed"])
        if key in null_coords:
            raise RuntimeError("duplicate null key: %s" % key)
        record = {"key": key, "source": str(entry["source"]), "null_kind": str(entry["null_kind"]),
                  "seed": entry["seed"], "path": str(path.relative_to(RUN_DIR)),
                  "sha256": file_sha, "sha256_expected": str(entry["sha256"]),
                  "coordinate_array_sha256": array_sha,
                  "coordinate_array_sha256_expected": str(entry["coordinate_array_sha256"]),
                  "file_sha_match": bool(file_sha == entry["sha256"]),
                  "array_sha_match": bool(array_sha == entry["coordinate_array_sha256"]),
                  "global_scale": entry.get("global_scale"),
                  "post_scale_radius": entry.get("post_scale_radius")}
        if not (record["file_sha_match"] and record["array_sha_match"]):
            raise RuntimeError("null hash mismatch: %s" % record)
        null_records.append(record)
        null_coords[key] = coords

    if len(null_records) != 119 or len({record["key"] for record in null_records}) != 119:
        raise RuntimeError("null records are not 119 unique (source, kind, seed) keys: %d / %d"
                           % (len(null_records), len({record["key"] for record in null_records})))
    if len({record["coordinate_array_sha256"] for record in null_records}) != len(null_records):
        raise RuntimeError("null coordinate arrays are not all distinct")
    mask_sha = lib.sha256_file(lib.MASK_SNAPSHOT)
    reference_bytes_sha = lib.sha256_file(lib.REFERENCE_PATH)
    if mask_sha != MASK_SNAPSHOT_SHA256:
        raise RuntimeError("mask snapshot hash mismatch: %s" % mask_sha)
    if reference_bytes_sha != REFERENCE_SHA256:
        raise RuntimeError("reference bytes hash mismatch: %s" % reference_bytes_sha)
    hash_gate = {
        "schema": "p9016-round050-p1-spatial-corrected-pre-reference-gate-v1", "status": "PASS",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "role": "independent spatial-evaluation process of round 050; evaluation side may read the reference, "
                "phase columns are forbidden",
        "source_gate": str(GATE.relative_to(RUN_DIR)), "source_gate_reference_opened": False,
        "reference": {"path": str(lib.REFERENCE_PATH.relative_to(ROOT)), "sha256": reference_bytes_sha,
                      "hash_only_stage": "bytes hashed before the payload was parsed"},
        "mask_snapshot": {"path": str(lib.MASK_SNAPSHOT.relative_to(ROOT)), "sha256": mask_sha},
        "aggregate": aggregate_audit,
        "sources": source_records, "source_count": len(source_records),
        "nulls": null_records, "null_count": len(null_records),
        "all_source_hashes_match": all(r["file_sha_match"] and r["array_sha_match"] for r in source_records),
        "all_null_hashes_match": all(r["file_sha_match"] and r["array_sha_match"] for r in null_records),
        "reference_opened": False, "phase_opened": False,
        "note": "every 050 source 3dg / coordinate array and every null npz / coordinate array was re-hashed and "
                "matched the frozen 050 pre_reference_gate.json before this process parsed the reference",
    }
    lib.write_json(RESULTS / "p1_spatial_corrected_gate.json", hash_gate)

    # ---------------------------------------------------------- 2) reference 侧解析（gate 之后）
    reference_tracks = lib.load_3dg(lib.REFERENCE_PATH)
    reference = lib.three_dg_to_array(reference_tracks, data, track_mode="reference")
    masks, mask_audit = ev049.load_real_masks(data)
    support_check = ev049.assert_reference_finite_on_mask(reference, masks, data)
    per_chr_valid = spatial.per_chromosome_valid_indices(data, masks)
    valid_global = spatial.valid_global_bins(data, masks, expected_valid_bins=EXPECTED_VALID_BINS)
    n_valid_loci = int(valid_global.sum())
    n_valid_beads = int(2 * n_valid_loci)
    if n_valid_loci != EXPECTED_VALID_BINS or n_valid_beads != EXPECTED_VALID_BEADS:
        raise RuntimeError("valid support is not 2447 loci / 4894 beads: %d / %d" % (n_valid_loci, n_valid_beads))
    if int(np.isfinite(reference).all(axis=2).all(axis=0).sum()) != EXPECTED_VALID_BINS:
        raise RuntimeError("reference-finite loci count does not equal the 2447 mask support")
    ref_merged, _ref_copies = spatial.merged_chr_centers(reference, per_chr_valid)

    # ---------------------------------------------------------- 3) 逐 source / 逐 null 读出
    source_rows: list[dict[str, Any]] = []
    source_reads: dict[str, Any] = {}
    for record in source_records:
        name = record["id"]
        reads = spatial_reads(source_coords[name], reference, ref_merged, per_chr_valid, masks, data)
        source_reads[name] = reads
        merged = reads["merged_centres_190"]
        copy = reads["copy_centres_760"]
        source_rows.append({
            "source": name, "kind": record["kind"], "3dg_sha256": record["3dg_sha256"],
            "coordinate_array_sha256": record["coordinate_array_sha256"],
            "n_merged_distances": merged["n_distances"], "merged_centre_pearson_190": merged["pearson"],
            "merged_centre_spearman_190": merged["spearman"],
            "merged_centre_optimal_scale": merged["optimal_single_scale"],
            "merged_centre_normalized_stress": merged["normalized_stress"],
            "merged_per_chr_profile_macro_pearson": merged["per_chr_profile_macro_pearson"],
            "merged_per_chr_profile_macro_spearman": merged["per_chr_profile_macro_spearman"],
            "merged_top3_overlap_mean": merged["top3_neighbor_overlap_mean"],
            "n_copy_distances": None if copy is None else copy["n_distances"],
            "copy_centre_pearson_760": None if copy is None else copy["pearson"],
            "copy_centre_spearman_760": None if copy is None else copy["spearman"],
            "copy_centre_optimal_scale": None if copy is None else copy["optimal_single_scale"],
            "copy_centre_normalized_stress": None if copy is None else copy["normalized_stress"],
            "copy_centre_unresolved_chromosomes": "|".join(reads["copy_centre_gauge_unresolved_chromosomes"]),
            "copy_centre_gauge_defined": reads["copy_centre_gauge_defined"],
            "masked_valid_loci": n_valid_loci, "masked_beads": n_valid_beads,
        })

    null_rows: list[dict[str, Any]] = []
    null_reads: list[dict[str, Any]] = []
    for record in null_records:
        key = "%s__%s" % (record["source"], record["null_kind"])
        key = record["key"]
        scored_coords = null_coords[key]
        scored_hash = lib.hash_array(scored_coords)
        if scored_hash != record["coordinate_array_sha256"]:
            raise RuntimeError("null %s scored coordinates do not match its own record hash: %s vs %s"
                               % (key, scored_hash, record["coordinate_array_sha256"]))
        reads = spatial_reads(scored_coords, reference, ref_merged, per_chr_valid, masks, data)
        merged = reads["merged_centres_190"]
        copy = reads["copy_centres_760"]
        row = {
            "key": key, "source": record["source"], "null_kind": record["null_kind"], "seed": record["seed"],
            "npz_sha256": record["sha256"],
            "scored_coordinate_array_sha256": scored_hash,
            "n_merged_distances": merged["n_distances"],
            "merged_centre_pearson_190": merged["pearson"],
            "merged_centre_spearman_190": merged["spearman"],
            "merged_centre_optimal_scale": merged["optimal_single_scale"],
            "merged_centre_normalized_stress": merged["normalized_stress"],
            "n_copy_distances": None if copy is None else copy["n_distances"],
            "copy_centre_pearson_760": None if copy is None else copy["pearson"],
            "copy_centre_spearman_760": None if copy is None else copy["spearman"],
            "copy_centre_normalized_stress": None if copy is None else copy["normalized_stress"],
            "copy_centre_gauge_defined": reads["copy_centre_gauge_defined"],
            "copy_centre_unresolved_chromosomes": "|".join(reads["copy_centre_gauge_unresolved_chromosomes"]),
            "copy_centre_760_identity_fallback_pearson":
                None if reads["copy_centres_760_identity_fallback_when_gauge_unresolved"] is None
                else reads["copy_centres_760_identity_fallback_when_gauge_unresolved"]["pearson"],
            "masked_valid_loci": n_valid_loci, "masked_beads": n_valid_beads,
        }
        null_rows.append(row)
        null_reads.append({"key": key, "source": record["source"], "null_kind": record["null_kind"],
                           "seed": record["seed"], "scored_coordinate_array_sha256": scored_hash, "reads": reads})

    # ---------------------------------------------------------- 4) null 分层汇总（不新增置换实验）
    def finite(values: Sequence[Any]) -> list[float]:
        return [float(v) for v in values if v is not None and math.isfinite(float(v))]

    strata: dict[str, Any] = {}
    for source in sorted({row["source"] for row in null_rows}):
        for kind in ("u_zero", "random_u"):
            subset = [row for row in null_rows if row["source"] == source and row["null_kind"] == kind]
            if not subset:
                continue
            entry: dict[str, Any] = {"n_draws": len(subset)}
            for field in ("merged_centre_pearson_190", "merged_centre_normalized_stress",
                          "copy_centre_pearson_760", "copy_centre_normalized_stress"):
                values = finite([row[field] for row in subset])
                entry[field] = {
                    "finite_n": len(values),
                    "na_n": len(subset) - len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "min": float(np.min(values)) if values else None,
                    "max": float(np.max(values)) if values else None,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                }
            entry["copy_centre_gauge_defined_n"] = sum(1 for row in subset if row["copy_centre_gauge_defined"])
            strata["%s:%s" % (source, kind)] = entry
    pooled: dict[str, Any] = {}
    for kind in ("u_zero", "random_u"):
        subset = [row for row in null_rows if row["null_kind"] == kind]
        entry = {"n_draws": len(subset)}
        for field in ("merged_centre_pearson_190", "merged_centre_normalized_stress",
                      "copy_centre_pearson_760", "copy_centre_normalized_stress"):
            values = finite([row[field] for row in subset])
            entry[field] = {"finite_n": len(values), "na_n": len(subset) - len(values),
                            "mean": float(np.mean(values)) if values else None,
                            "min": float(np.min(values)) if values else None,
                            "max": float(np.max(values)) if values else None}
        pooled[kind] = entry

    # 逐 draw 完整性 + 实测的 u-null 空间变化（不预定结论）
    variation: dict[str, Any] = {
        "integrity_rule": "every null row must be scored on its OWN coordinate array: the key is "
                          "(source, null_kind, seed) and the scored array sha256 is re-checked against that "
                          "record's gate hash; 119 unique keys are required",
        "per_source": {},
        "prior_defect_retracted": {
            "what": "an earlier version of this script keyed the loaded null arrays by (source, null_kind) only, so "
                    "the 16 random-u seeds of one source overwrote each other and every seed row was scored on the "
                    "last seed's coordinates; the 'u-null spatial degeneracy' statement derived from that run is "
                    "retracted and must not be quoted",
            "fix": "unique (source, null_kind, seed) keys, 119-key uniqueness assertion, per-draw scored-array sha256 "
                   "re-checked against the row's gate hash",
        },
    }
    variation_ok = True
    for source in sorted({row["source"] for row in null_rows}):
        sub = [row for row in null_rows if row["source"] == source and row["null_kind"] == "random_u"]
        source_row = next(row for row in source_rows if row["source"] == source)
        scored_hashes = {row["scored_coordinate_array_sha256"] for row in sub}
        merged_values = [row["merged_centre_pearson_190"] for row in sub]
        copy_values = finite([row["copy_centre_pearson_760"] for row in sub])
        source_mapping = source_reads[source]["per_chromosome_mapping"]
        flip_counts, flip_sets = [], set()
        for entry in null_reads:
            if entry["source"] != source or entry["null_kind"] != "random_u":
                continue
            flipped = tuple(sorted(str(name) for name, item in entry["reads"]["per_chromosome_mapping"].items()
                                   if item["orientation"] != source_mapping[str(name)]["orientation"]))
            flip_counts.append(len(flipped))
            flip_sets.add(flipped)
        merged_unique = len({repr(value) for value in merged_values})
        copy_unique = len({repr(value) for value in copy_values})
        merged_abs_diff = max(abs(value - source_row["merged_centre_pearson_190"]) for value in merged_values)
        entry_ok = bool(len(sub) == 16 and len(scored_hashes) == 16)
        variation_ok = variation_ok and entry_ok
        variation["per_source"][source] = {
            "n_draws": len(sub),
            "unique_scored_coordinate_hashes": len(scored_hashes),
            "all_scored_hashes_match_gate_rows": True,
            "merged_centre_pearson_190": {
                "unique_values": merged_unique, "min": min(merged_values), "max": max(merged_values),
                "mean": float(np.mean(merged_values)),
                "max_abs_diff_vs_source": merged_abs_diff,
                "source_value": source_row["merged_centre_pearson_190"],
            },
            "copy_centre_pearson_760": {
                "finite_n": len(copy_values), "unique_values": copy_unique,
                "min": min(copy_values) if copy_values else None,
                "max": max(copy_values) if copy_values else None,
                "mean": float(np.mean(copy_values)) if copy_values else None,
                "source_value": source_row["copy_centre_pearson_760"],
            },
            "gauge_orientation_flips_vs_source": {
                "min_chromosomes": min(flip_counts) if flip_counts else None,
                "max_chromosomes": max(flip_counts) if flip_counts else None,
                "distinct_flip_sets_across_seeds": len(flip_sets),
            },
            "merged_190_behaviour": ("expected to be invariant: the merged centre is the mean of z = (copy0+copy1)/2 "
                                     "over the masked loci and the u-permutation does not touch z; any residual is "
                                     "uniform scaling plus float rounding and Pearson/stress are scale invariant"),
            "copy_760_behaviour": ("can and does vary with the seed: the copy centre uses the masked-subset mean of "
                                   "the permuted u, which is not conserved by a whole-chromosome permutation, and "
                                   "the gauge (intra-chromosome copy geometry) changes as well"),
        }
    variation["passed"] = bool(variation_ok and len(null_rows) == 119)

    # ---------------------------------------------------------- 5) baseline regression（049 冻结值）
    frozen = lib.read_json(FROZEN_049_EVALUATION)["datasets"][REGRESSION_FROZEN_DATASET]["spatial"]
    mine = source_reads[REGRESSION_SOURCE]
    comparisons = {
        "merged_centre_pearson_190": (mine["merged_centres_190"]["pearson"], frozen["merged_chr_centers"]["pearson"]),
        "merged_centre_spearman_190": (mine["merged_centres_190"]["spearman"],
                                       frozen["merged_chr_centers"]["spearman"]),
        "merged_centre_optimal_scale": (mine["merged_centres_190"]["optimal_single_scale"],
                                        frozen["merged_chr_centers"]["optimal_single_scale"]),
        "merged_centre_normalized_stress": (mine["merged_centres_190"]["normalized_stress"],
                                            frozen["merged_chr_centers"]["normalized_stress"]),
        "merged_per_chr_profile_macro_pearson": (mine["merged_centres_190"]["per_chr_profile_macro_pearson"],
                                                 frozen["merged_chr_centers"]["per_chr_profile_macro_pearson"]),
        "merged_per_chr_profile_macro_spearman": (mine["merged_centres_190"]["per_chr_profile_macro_spearman"],
                                                  frozen["merged_chr_centers"]["per_chr_profile_macro_spearman"]),
        "copy_centre_pearson_760": (mine["copy_centres_760"]["pearson"], frozen["copy_centers"]["primary"]["pearson"]),
        "copy_centre_spearman_760": (mine["copy_centres_760"]["spearman"],
                                     frozen["copy_centers"]["primary"]["spearman"]),
        "copy_centre_optimal_scale": (mine["copy_centres_760"]["optimal_single_scale"],
                                      frozen["copy_centers"]["primary"]["optimal_single_scale"]),
        "copy_centre_normalized_stress": (mine["copy_centres_760"]["normalized_stress"],
                                          frozen["copy_centers"]["primary"]["normalized_stress"]),
    }
    regression_rows = []
    for key, (observed, expected) in comparisons.items():
        diff = abs(float(observed) - float(expected))
        regression_rows.append({"field": key, "observed": float(observed), "frozen_049": float(expected),
                                "abs_diff": diff, "within_tolerance": bool(diff <= REGRESSION_TOL)})
    regression = {
        "schema": "p9016-round050-p1-spatial-corrected-regression-v1",
        "source": REGRESSION_SOURCE, "frozen_dataset": REGRESSION_FROZEN_DATASET,
        "frozen_file": str(FROZEN_049_EVALUATION.relative_to(ROOT)),
        "frozen_spatial_tsv": str(FROZEN_049_SPATIAL_TSV.relative_to(ROOT)),
        "tolerance": REGRESSION_TOL, "rows": regression_rows,
        "status": "PASS" if all(row["within_tolerance"] for row in regression_rows) else "FAIL",
        "max_abs_diff": max(row["abs_diff"] for row in regression_rows),
        "note": "scalars are produced by the original 049 helpers (merged_chr_centers / center_metrics / "
                "copy_center_metrics) on the frozen 046 work baseline; nothing is re-fitted",
        "parent_quoted_values_not_found_in_frozen_artifacts": REGISTRY_QUOTED_BY_PARENT,
        "frozen_values_used_instead": {
            "center_pearson_190": frozen["merged_chr_centers"]["pearson"],
            "center_normalized_stress": frozen["merged_chr_centers"]["normalized_stress"],
            "source": "049 evaluation.json datasets/baseline-046-real-extension-G-full-J/spatial and "
                      "spatial_summary.tsv row baseline-046-real-extension-G-full-J",
        },
    }
    lib.write_json(RESULTS / "p1_spatial_corrected_regression.json", regression)

    # ---------------------------------------------------------- 6) A/B 互换不变性检查
    inv_coords = source_coords[INVARIANCE_SOURCE]
    base_reads = source_reads[INVARIANCE_SOURCE]
    swapped_all = np.ascontiguousarray(inv_coords[::-1])
    reads_global = spatial_reads(swapped_all, reference, ref_merged, per_chr_valid, masks, data)
    global_pairs = {
        "merged_centre_pearson_190":
            (base_reads["merged_centres_190"]["pearson"], reads_global["merged_centres_190"]["pearson"]),
        "merged_centre_normalized_stress":
            (base_reads["merged_centres_190"]["normalized_stress"], reads_global["merged_centres_190"]["normalized_stress"]),
        "copy_centre_pearson_760":
            (base_reads["copy_centres_760"]["pearson"], reads_global["copy_centres_760"]["pearson"]),
        "copy_centre_normalized_stress":
            (base_reads["copy_centres_760"]["normalized_stress"], reads_global["copy_centres_760"]["normalized_stress"]),
        "copy_centre_optimal_scale":
            (base_reads["copy_centres_760"]["optimal_single_scale"],
             reads_global["copy_centres_760"]["optimal_single_scale"]),
    }
    global_orientation_flips = sum(
        1 for name in data.chromosome_names
        if base_reads["per_chromosome_mapping"][str(name)]["orientation"] == "direct"
        and reads_global["per_chromosome_mapping"][str(name)]["orientation"] == "swapped"
        or base_reads["per_chromosome_mapping"][str(name)]["orientation"] == "swapped"
        and reads_global["per_chromosome_mapping"][str(name)]["orientation"] == "direct")
    per_chr_swap = []
    for index, name in enumerate(data.chromosome_names):
        single = np.array(inv_coords, dtype=np.float64, copy=True)
        slc = data.chromosome_slice(index)
        single[0, slc] = inv_coords[1, slc]
        single[1, slc] = inv_coords[0, slc]
        reads_single = spatial_reads(single, reference, ref_merged, per_chr_valid, masks, data)
        flipped = [str(other) for other in data.chromosome_names
                   if base_reads["per_chromosome_mapping"][str(other)]["orientation"] !=
                   reads_single["per_chromosome_mapping"][str(other)]["orientation"]]
        per_chr_swap.append({"chromosome": str(name), "orientation_before":
                             base_reads["per_chromosome_mapping"][str(name)]["orientation"],
                             "orientation_after": reads_single["per_chromosome_mapping"][str(name)]["orientation"],
                             "chromosomes_whose_orientation_changed": flipped,
                             "expected_only_this_chromosome": bool(flipped == [str(name)]),
                             "copy_centre_pearson_760": reads_single["copy_centres_760"]["pearson"]})
    invariance = {
        "schema": "p9016-round050-p1-spatial-corrected-gauge-invariance-v1",
        "candidate": INVARIANCE_SOURCE,
        "test": "evaluation-side canonical check on the same frozen candidate; the swapped coordinates are never "
                "written and no new candidate is kept",
        "global_ab_swap": {
            "definition": "swap copy0<->copy1 for all 20 chromosomes at once",
            "readout_pairs": {key: {"original": value[0], "swapped": value[1],
                                    "abs_diff": abs(value[1] - value[0]),
                                    "identical_within_1e-12": bool(abs(value[1] - value[0]) <= 1e-12)}
                              for key, value in global_pairs.items()},
            "chromosomes_whose_gauge_orientation_flipped": int(global_orientation_flips),
            "expected_flips": 20,
            "gauge_invariant": bool(global_orientation_flips == 20
                                    and all(abs(v[1] - v[0]) <= 1e-12 for v in global_pairs.values())),
        },
        "per_chromosome_single_swap": {
            "definition": "swap copy0<->copy1 for exactly one chromosome; only that chromosome's gauge decision "
                          "may change",
            "n_chromosomes_tested": len(per_chr_swap),
            "all_single_swaps_flip_only_that_chromosome": all(row["expected_only_this_chromosome"]
                                                             for row in per_chr_swap),
            "rows": per_chr_swap,
        },
        "conclusion": "the per-chromosome whole-chromosome signed-Pearson gauge is effective: a global A/B swap "
                      "leaves the 190/760 readouts unchanged while flipping every chromosome's orientation, and a "
                      "single-chromosome swap flips only that chromosome's decision",
    }
    lib.write_json(RESULTS / "p1_spatial_corrected_invariance.json", invariance)

    # ---------------------------------------------------------- 7) 与原 050 错误读数对照（superseded）
    old = lib.read_json(OLD_SUMMARY)
    superseded = {
        "old_file": str(OLD_SUMMARY.relative_to(RUN_DIR)),
        "old_n_common_finite_loci": sorted({int(v["spatial"]["n_common_finite_loci"])
                                            for v in old["sources"].values()}),
        "old_merged_centre_pearson_values": sorted({str(v["spatial"]["merged_centres_20"]["pearson"])
                                                    for v in old["sources"].values()}),
        "old_copy_centre_pearson_values": sorted({str(v["spatial"]["copy_centres_cross_chromosome_40"]["pearson"])
                                                  for v in old["sources"].values()}),
        "old_problems": [
            "no finite_loci was passed, so the default all-2645-locus support was used and the reference NaNs "
            "outside the frozen old21 mask made every spatial correlation undefined (null)",
            "copy centres were paired copy0/copy1 without the per-chromosome geometric A/B best swap",
            "stress was a z-score RMS instead of the 049 single-optimal-scale stress",
        ],
        "superseded_by": ["p1_spatial_corrected.json", "p1_spatial_corrected_sources.tsv",
                          "p1_spatial_corrected_nulls.tsv"],
        "scope_untouched": "R1/R2/inter/count/full-J readouts of round 050 are NOT recomputed here and are not "
                           "changed by this correction",
    }

    # ---------------------------------------------------------- 8) 期望 finite/NA 核对与产物
    expected_na = []
    unexpected_na = []
    for row in source_rows:
        if row["copy_centre_pearson_760"] is None or row["merged_centre_pearson_190"] is None:
            unexpected_na.append({"row": row["source"], "kind": "source"})
    for row in null_rows:
        if row["null_kind"] == "u_zero":
            if row["copy_centre_pearson_760"] is None and row["merged_centre_pearson_190"] is not None:
                expected_na.append({"row": "%s__%s" % (row["source"], row["seed"]), "kind": "u_zero copy gauge NA"})
            else:
                unexpected_na.append({"row": "%s__%s" % (row["source"], row["seed"]),
                                      "kind": "u_zero not NA as expected"})
        else:
            if row["copy_centre_pearson_760"] is None or row["merged_centre_pearson_190"] is None:
                unexpected_na.append({"row": "%s__seed%s" % (row["source"], row["seed"]), "kind": "random_u NA"})
    finite_check = {
        "n_sources": len(source_rows), "n_nulls": len(null_rows),
        "sources_all_centres_finite": not any(item["kind"] == "source" for item in unexpected_na),
        "expected_na_count": len(expected_na), "expected_na": expected_na,
        "unexpected_na": unexpected_na,
        "u_zero_rule": "the u=0 control makes both copies of every chromosome coincide, so the whole-chromosome "
                       "signed-Pearson gauge is undefined/ambiguous; per the frozen no-hand-assignment rule the "
                       "copy-centre 760 primary readout is NA and the 049 identity-fallback number is kept in a "
                       "separately named field",
    }

    validation = {
        "schema": "p9016-round050-p1-spatial-corrected-validation-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hash_gate": {"all_source_hashes_match": hash_gate["all_source_hashes_match"],
                      "all_null_hashes_match": hash_gate["all_null_hashes_match"],
                      "mask_snapshot_sha256": mask_sha, "reference_sha256": reference_bytes_sha,
                      "matched_expected_mask": bool(mask_sha == MASK_SNAPSHOT_SHA256),
                      "matched_expected_reference": bool(reference_bytes_sha == REFERENCE_SHA256)},
        "support": {"valid_loci": n_valid_loci, "beads": n_valid_beads,
                    "merged_centre_pairs": EXPECTED_MERGED_PAIRS, "copy_centre_cross_chromosome_pairs":
                        EXPECTED_COPY_PAIRS, "copy_centre_same_chr_homolog_pairs_excluded":
                        EXPECTED_SAME_CHR_HOMOLOG_PAIRS,
                    "reference_finite_on_mask": bool(support_check["support_finite"]),
                    "mask_totals": mask_audit["totals"],
                    "global_index_rule": mask_audit["global_index_rule"]},
        "regression": {"status": regression["status"], "max_abs_diff": regression["max_abs_diff"],
                       "tolerance": REGRESSION_TOL, "rows": regression_rows},
        "gauge_invariance": {"global_ab_swap_gauge_invariant":
                             invariance["global_ab_swap"]["gauge_invariant"],
                             "single_swaps_flip_only_that_chromosome":
                             invariance["per_chromosome_single_swap"]["all_single_swaps_flip_only_that_chromosome"]},
        "u_null_per_draw_integrity_and_variation": variation,
        "finite_and_na_check": finite_check,
        "superseded_050_spatial": superseded,
        "not_recomputed": ["R1 label accuracy", "R2 matched/cross/contrast", "inter distances",
                           "count/full-J objectives", "P1 fits", "P2 probes"],
        "failed_checks": [],
    }
    validation["failed_checks"] = [key for key, value in (
        ("hash_gate", validation["hash_gate"]["all_source_hashes_match"] and validation["hash_gate"]["all_null_hashes_match"]),
        ("support", validation["support"]["valid_loci"] == EXPECTED_VALID_BINS
                   and validation["support"]["beads"] == EXPECTED_VALID_BEADS
                   and validation["support"]["reference_finite_on_mask"]),
        ("regression", validation["regression"]["status"] == "PASS"),
        ("gauge_invariance", validation["gauge_invariance"]["global_ab_swap_gauge_invariant"]
                             and validation["gauge_invariance"]["single_swaps_flip_only_that_chromosome"]),
        ("u_null_per_draw_integrity_and_variation",
         validation["u_null_per_draw_integrity_and_variation"]["passed"]),
        ("finite_and_na", not finite_check["unexpected_na"]),
    ) if not value]
    validation["all_checks_passed"] = not validation["failed_checks"]
    lib.write_json(RESULTS / "p1_spatial_corrected_validation.json", validation)

    columns = list(source_rows[0].keys())
    lib.write_tsv(RESULTS / "p1_spatial_corrected_sources.tsv", source_rows, columns)
    null_columns = list(null_rows[0].keys())
    lib.write_tsv(RESULTS / "p1_spatial_corrected_nulls.tsv", null_rows, null_columns)
    report = {
        "schema": "p9016-round050-p1-spatial-corrected-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "started_at_utc": started_at,
        "purpose": "independent correction of the three centre_reads problems of round 050 (missing finite support, "
                   "no per-chromosome A/B gauge, z-score stress); no refit and no R1/R2/inter/count recomputation",
        "helpers_reused": ["eval049_spatial.merged_chr_centers", "eval049_spatial.center_metrics",
                           "eval049_spatial.copy_center_metrics", "eval049_evaluate.load_real_masks",
                           "eval049_evaluate.assert_reference_finite_on_mask", "eval049_lib.*"],
        "support": validation["support"], "sources": source_reads, "source_rows": source_rows,
        "null_strata": strata, "null_pooled": pooled, "null_rows": null_rows,
        "u_null_per_draw_integrity_and_variation": variation,
        "regression": regression, "gauge_invariance": invariance, "superseded_050_spatial": superseded,
        "finite_and_na_check": finite_check, "hash_gate": hash_gate,
        "definition": {
            "merged_centres_190": "20 merged chromosome centres (both copies averaged over the 2447 valid loci) "
                                  "give 190 distances; one single optimal scale stress",
            "copy_centres_760": "40 copy centres (2 per chromosome) give the 760 cross-chromosome distances; the "
                                "20 same-chromosome homolog pairs are excluded; per-chromosome whole-chromosome "
                                "signed-Pearson best A/B swap defines the correspondence",
            "stress": "sqrt(sum((s*d - ref)^2) / sum(ref^2)) with the single optimal s = (d.ref)/(d.d); never a "
                      "z-score RMS",
        },
        "reference_opened": True, "phase_opened": False,
        "wall_seconds": time.time() - t0,
    }
    lib.write_json(RESULTS / "p1_spatial_corrected.json", report)
    terminal = {
        "schema": "p9016-round050-p1-spatial-corrected-terminal-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "started_at_utc": started_at,
        "terminal_state": "completed_evaluation_only" if validation["all_checks_passed"] else "completed_with_failed_checks",
        "exit_code": 0, "convergence": {"applicable": False,
                                        "reason": "evaluation-only spatial readout; nothing is fitted or optimised"},
        "n_sources": len(source_rows), "n_nulls": len(null_rows),
        "regression_status": regression["status"],
        "gauge_invariant": invariance["global_ab_swap"]["gauge_invariant"],
        "all_checks_passed": validation["all_checks_passed"], "failed_checks": validation["failed_checks"],
        "wall_seconds": report["wall_seconds"], "reference_opened": True, "phase_opened": False,
    }
    lib.write_json(RESULTS / "p1_spatial_corrected_terminal.json", terminal)
    print(lib.jsonable({"terminal_state": terminal["terminal_state"], "n_sources": len(source_rows),
                        "n_nulls": len(null_rows), "regression_status": regression["status"],
                        "regression_max_abs_diff": regression["max_abs_diff"],
                        "gauge_invariant": invariance["global_ab_swap"]["gauge_invariant"],
                        "single_swaps_ok": invariance["per_chromosome_single_swap"]["all_single_swaps_flip_only_that_chromosome"],
                        "all_checks_passed": validation["all_checks_passed"],
                        "failed_checks": validation["failed_checks"],
                        "wall_seconds": report["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
