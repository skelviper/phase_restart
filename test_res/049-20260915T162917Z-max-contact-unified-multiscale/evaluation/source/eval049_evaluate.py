#!/usr/bin/env python
"""049 正式评价：R2 / inter / 空间 / bootstrap / 标签置换 null / null draw。

**reference gate**：只有 (1) ``gates/pre_reference_gate.json`` 存在且 PASS、
(2) 父侧通过 ``--authorized-by-parent`` 明确授权时，本脚本才打开
``data/P9016.1m.3dg.gz``，并在打开前记录 UTC 时刻。mask snapshot 同样只在 gate 之后加载。
本脚本不读 phase 列、不做任何拟合、不选端点（端点/显示选择已由训练侧冻结）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval049_inputs as inputs  # noqa: E402
import eval049_lib as lib  # noqa: E402
import eval049_spatial as spatial  # noqa: E402

GATE_PATH = lib.EVAL / "gates/pre_reference_gate.json"
REFERENCE_OPEN_PATH = lib.EVAL / "gates/reference_open.json"
RESULTS = lib.EVAL / "results"
NULL_DIR = lib.EVAL / "nulls"
LOG_PREFIX = "eval049"


# ------------------------------------------------------------------ gate


def load_gate(authorized_by_parent: str) -> dict[str, Any]:
    if not GATE_PATH.is_file():
        raise RuntimeError("pre-reference gate missing: %s (run eval049_prep.py first)" % GATE_PATH)
    gate = lib.read_json(GATE_PATH)
    if gate.get("status") != "PASS":
        raise RuntimeError("pre-reference gate status is %r" % gate.get("status"))
    if gate.get("reference_opened") is not False:
        raise RuntimeError("gate reports the reference was already opened")
    if not authorized_by_parent.strip():
        raise RuntimeError("reference open requires explicit --authorized-by-parent text")
    gate["authorized_by_parent"] = authorized_by_parent.strip()
    return gate


def open_reference(gate: Mapping[str, Any], *, reference_path: Path | None = None,
                   expected_sha256: str | None = None, record_path: Path | None = None,
                   gate_path: Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """先落盘首次打开证据，再读取 reference；读取失败也保留 failed 状态。

    路径/期望 hash 可被代码自检覆盖；正式调用一律使用默认（真实 reference 与真实 gate 路径）。
    """
    reference_path = Path(reference_path) if reference_path is not None else lib.REFERENCE_PATH
    expected_sha256 = str(expected_sha256) if expected_sha256 is not None else lib.REFERENCE_SHA256
    record_path = Path(record_path) if record_path is not None else REFERENCE_OPEN_PATH
    gate_path = Path(gate_path) if gate_path is not None else GATE_PATH
    previous = lib.read_json(record_path) if Path(record_path).is_file() else {}
    opened_utc = str(previous.get("reference_first_opened_utc") or lib.utc_now())
    attempt = (int(previous.get("open_attempt_count") or 1) + 1) if previous else 1
    record: dict[str, Any] = {
        "schema": "p9016-049-reference-open-record-v1",
        "status": "opening",
        "reference_first_opened_utc": opened_utc,
        "open_attempt_count": attempt,
        "this_attempt_started_utc": lib.utc_now(),
        "previous_attempts": previous.get("attempt_history", []) + ([{
            "status": previous.get("status"), "attempt": previous.get("open_attempt_count"),
            "error": previous.get("error"), "started_utc": previous.get("this_attempt_started_utc"),
        }] if previous else []),
        "reference_path": str(reference_path),
        "expected_sha256": expected_sha256,
        "phase_columns_read": False,
        "gate_path": str(gate_path),
        "authorization": gate.get("authorized_by_parent"),
        "track_mode": "reference",
    }
    lib.write_json(record_path, record)
    gate_updated = dict(gate)
    gate_updated["reference_first_opened_utc"] = opened_utc
    gate_updated["reference_open_status"] = "opening"
    lib.write_json(gate_path, gate_updated)
    try:
        if not reference_path.is_file():
            raise RuntimeError("reference file missing: %s" % reference_path)
        actual = lib.sha256_file(reference_path)
        if actual != expected_sha256:
            raise RuntimeError("reference SHA mismatch: %s" % actual)
        tracks = lib.load_3dg(reference_path)
        data = lib.Aggregate()
        audit: dict[str, Any] = {}
        reference = lib.three_dg_to_array(tracks, data, track_mode="reference", audit=audit)
        finite = np.isfinite(reference).all(axis=2)
        if audit["missing_track_count"] > 0:
            raise RuntimeError("reference 3DG is missing %d expected tracks, e.g. %s"
                               % (audit["missing_track_count"], audit["missing_tracks"][:5]))
        if int((~finite).sum()) == int(finite.size):
            raise RuntimeError("reference parse produced an all-NaN array; track naming/mapping is wrong")
        record.update({
            "status": "opened", "reference_sha256": actual, "tracks": len(tracks),
            "rows": int(reference.shape[0] * reference.shape[1]),
            "nonfinite_loci": int((~finite).sum()),
            "finite_loci": int(finite.sum()),
            "track_audit": {key: value for key, value in audit.items() if key != "per_chromosome_finite"},
            "per_chromosome_finite": audit["per_chromosome_finite"],
            "gate_sha256_at_open": lib.sha256_file(gate_path),
        })
        lib.write_json(record_path, record)
        gate_updated["reference_open_status"] = "opened"
        gate_updated["reference_open_record"] = str(record_path)
        lib.write_json(gate_path, gate_updated)
        return reference, record
    except Exception as exc:
        record.update({"status": "failed", "error": repr(exc)})
        lib.write_json(record_path, record)
        gate_updated["reference_open_status"] = "failed"
        gate_updated["reference_open_error"] = repr(exc)
        lib.write_json(gate_path, gate_updated)
        raise


def assert_reference_finite_on_mask(reference: np.ndarray, masks: Mapping[str, Mapping[str, Any]],
                                    data: lib.Aggregate) -> dict[str, Any]:
    """真实缺失允许，但冻结 mask 的 common 支持必须全部 finite。"""
    rows = []
    total_missing_support = 0
    for ci, name in enumerate(data.chromosome_names):
        mask = masks[str(name)]
        global_indices = lib.mask_global_indices(data.offsets, mask)
        points = reference[:, global_indices]
        finite = np.isfinite(points).all(axis=2)
        valid_local = np.asarray(mask["valid_local_bins"], dtype=np.int64)
        missing_valid = int((~finite[:, valid_local]).sum())
        total_missing_support += missing_valid
        rows.append({"chromosome": str(name), "mask_positions": int(len(global_indices)),
                     "valid_bins": int(len(valid_local)),
                     "nonfinite_positions_total": int((~finite).sum()), "nonfinite_in_support": missing_valid})
    if total_missing_support:
        raise RuntimeError("reference has %d nonfinite values inside the frozen mask support" % total_missing_support)
    return {"support_finite": True, "nonfinite_in_support": 0, "per_chromosome": rows,
            "policy": "real reference gaps allowed elsewhere; the frozen common support must stay finite"}


# --------------------------------------------------------------- mask/inter


def load_real_masks(data: lib.Aggregate) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """加载 046 冻结 old21 mask snapshot（仅 evaluation 侧）。"""
    actual = lib.sha256_file(lib.MASK_SNAPSHOT)
    validation_046 = lib.read_json(lib.MASK_VALIDATION_046) if lib.MASK_VALIDATION_046.is_file() else {}
    recorded_046 = str((validation_046.get("frozen_legacy_mask_worker") or {}).get("output_sha256") or "")
    if actual != lib.MASK_SNAPSHOT_SHA256:
        raise RuntimeError("frozen mask snapshot hash mismatch: %s" % actual)
    if recorded_046 and recorded_046 != actual:
        raise RuntimeError("046 creation-time mask hash disagrees with the file on disk: %s vs %s" % (recorded_046, actual))
    config_typo = lib.MASK_SNAPSHOT_SHA256_IN_049_CONFIG != actual
    with np.load(lib.MASK_SNAPSHOT, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    masks: dict[str, dict[str, Any]] = {}
    rows = []
    for ci, name in enumerate(data.chromosome_names):
        positions = np.asarray(arrays["chr%d_positions" % ci], dtype=np.int64)
        pair_i = np.asarray(arrays["chr%d_pair_i" % ci], dtype=np.int64)
        pair_j = np.asarray(arrays["chr%d_pair_j" % ci], dtype=np.int64)
        common = np.asarray(arrays["chr%d_common" % ci], dtype=bool)
        if not np.array_equal(positions, np.arange(lib.MASK_OFFSET_BP, int(data.chromosome_lengths[ci]), lib.BIN_SIZE_BP)):
            raise RuntimeError("mask positions for %s do not follow the old21 numeric 3Mb-offset rule" % name)
        if len(positions) != int(data.n_bins[ci]) - lib.MASK_OFFSET_BP // lib.BIN_SIZE_BP:
            raise RuntimeError("mask positions for %s do not cover bins 3..n-1 (%d vs %d)"
                               % (name, len(positions), int(data.n_bins[ci]) - 3))
        if int(positions[-1]) // lib.BIN_SIZE_BP != int(data.n_bins[ci]) - 1:
            raise RuntimeError("mask positions for %s do not end on the last chromosome bin" % name)
        expected_pairs = np.triu_indices(len(positions), k=1)
        if not (np.array_equal(pair_i, expected_pairs[0]) and np.array_equal(pair_j, expected_pairs[1])):
            raise RuntimeError("mask pair indices for %s changed" % name)
        valid = np.zeros(len(positions), dtype=bool)
        valid[pair_i[common]] = True
        valid[pair_j[common]] = True
        if not np.array_equal(common, valid[pair_i] & valid[pair_j]):
            raise RuntimeError("mask common flag does not factorize for %s" % name)
        masks[name] = {"chromosome": name, "chromosome_index": ci, "positions": positions,
                       "pair_i": pair_i, "pair_j": pair_j, "common": common,
                       "valid_local_bins": np.flatnonzero(valid),
                       "n_bins": int(len(positions)), "n_total_non_diagonal_pairs": int(len(pair_i)),
                       "n_common_pairs": int(common.sum())}
        rows.append({"chromosome": name, "n_bins": int(len(positions)),
                     "n_total_non_diagonal_pairs": int(len(pair_i)), "n_common_pairs": int(common.sum()),
                     "n_valid_bins": int(valid.sum()),
                     "local_bin_offsets": [int(positions[0]) // lib.BIN_SIZE_BP,
                                           int(positions[-1]) // lib.BIN_SIZE_BP]})
    totals = {"n_total_non_diagonal_pairs": sum(row["n_total_non_diagonal_pairs"] for row in rows),
              "n_common_pairs": sum(row["n_common_pairs"] for row in rows),
              "valid_bins": sum(row["n_valid_bins"] for row in rows)}
    for key, expected in (("n_total_non_diagonal_pairs", lib.EXPECTED_MASK["n_total_non_diagonal_pairs"]),
                          ("n_common_pairs", lib.EXPECTED_MASK["n_common_pairs"]),
                          ("valid_bins", lib.EXPECTED_MASK["valid_bins"])):
        if totals[key] != expected:
            raise RuntimeError("frozen mask total %s=%d expected=%d" % (key, totals[key], expected))
    valid_global = spatial.valid_global_bins(data, masks)
    audit = {"schema": "p9016-049-mask-audit-v1", "status": "PASS",
             "snapshot_path": lib.rel(lib.MASK_SNAPSHOT), "snapshot_sha256": actual,
             "snapshot_sha256_expected_in_049_config": lib.MASK_SNAPSHOT_SHA256_IN_049_CONFIG,
             "config_sha256_differs_by_transcription_typo": bool(config_typo),
             "snapshot_sha256_recorded_by_046_at_creation": recorded_046,
             "snapshot_sha256_046_record_path": lib.rel(lib.MASK_VALIDATION_046) if lib.MASK_VALIDATION_046.is_file() else None,
             "sha_note": ("049/config.json 记录的 mask sha256 与文件实际哈希相差 1 个字符（...9221541f55f... vs ...9221547f55f...）；"
                          "文件本身与 046 创建时记录一致，未修改。config 不在评价侧写范围内，故只记录不改。"),
             "per_chromosome": rows, "totals": totals,
             "global_index_rule": "global = chromosome_offset + positions // 1Mb (mask local index is never used as 0-origin)",
             "valid_bins": int(valid_global.sum()),
             "support_statement": "common 2447 valid loci = 4894 beads; no new finite-intersection denominator shrink, missing recorded as NA"}
    return masks, audit


def build_inter_cache(data: lib.Aggregate, masks: Mapping[str, Mapping[str, Any]], reference: np.ndarray,
                      *, frozen_expectations: bool = True) -> dict[str, Any]:
    valid = spatial.valid_global_bins(data, masks,
                                      expected_valid_bins=(lib.EXPECTED_MASK["valid_bins"] if frozen_expectations else None))
    within_valid = int(sum(math.comb(len(np.asarray(masks[str(name)]["valid_local_bins"])), 2)
                           for name in data.chromosome_names))
    if frozen_expectations and within_valid != lib.EXPECTED_MASK["n_common_pairs"]:
        raise RuntimeError("valid-bin within-chromosome pair count changed: %d" % within_valid)
    expected_inter = int(math.comb(int(valid.sum()), 2) - within_valid)
    all_inter = ~data.cis_pair
    pair_i = np.asarray(data.pair_i[all_inter], dtype=np.int64)
    pair_j = np.asarray(data.pair_j[all_inter], dtype=np.int64)
    keep = valid[pair_i] & valid[pair_j]
    pair_i, pair_j = pair_i[keep], pair_j[keep]
    if len(pair_i) != expected_inter:
        raise RuntimeError("inter valid-bin pair count mismatch: %d != %d" % (len(pair_i), expected_inter))
    if frozen_expectations and len(pair_i) != lib.EXPECTED_MASK["inter_locus_pairs"]:
        raise RuntimeError("inter valid-bin pair count %d != frozen %d" % (len(pair_i), lib.EXPECTED_MASK["inter_locus_pairs"]))
    ref_dist = lib.four_distances(reference, pair_i, pair_j)
    if not np.isfinite(ref_dist).all():
        raise RuntimeError("reference nonfinite on the frozen inter valid-bin set")
    ref_sorted = np.sort(ref_dist, axis=0)
    reference_flat = ref_sorted.ravel()
    return {"valid_global_bins": valid, "pair_i": pair_i, "pair_j": pair_j,
            "reference_sorted": ref_sorted, "reference_flat": reference_flat,
            "reference_rank": rankdata(reference_flat),
            "valid_bin_count": int(valid.sum()), "within_valid_pair_count": within_valid,
            "eligible_locus_pairs": int(len(pair_i)), "denominator": int(4 * len(pair_i)),
            "sort_rule": "four-copy distances sorted per locus pair then concatenated in state order (matches 046 frozen definition)"}


def inter_metrics(coords: np.ndarray, cache: Mapping[str, Any], *, order_statistics: bool) -> dict[str, Any]:
    pair_i = np.asarray(cache["pair_i"], dtype=np.int64)
    pair_j = np.asarray(cache["pair_j"], dtype=np.int64)
    dist = lib.four_distances(coords, pair_i, pair_j)
    nonfinite = int((~np.isfinite(dist)).sum())
    result: dict[str, Any] = {
        "eligible_locus_pairs": int(len(pair_i)), "denominator": int(4 * len(pair_i)),
        "valid_bin_count": int(cache["valid_bin_count"]), "nonfinite_value_count": nonfinite,
        "valid_bin_policy": "old21 common-mask participating bins only; no reference-finite network expansion",
        "order_statistic_vector": "sorted four-copy distances per inter-chromosome locus pair",
    }
    if nonfinite:
        result.update({"pearson": None, "spearman": None, "status": "NA_nonfinite_candidate",
                       "na_policy": "missing recorded as NA; denominator not shrunk"})
        return result
    candidate_sorted = np.sort(dist, axis=0)
    candidate_flat = candidate_sorted.ravel()
    result["pearson"] = lib.metric("pearson", candidate_flat, np.asarray(cache["reference_flat"], dtype=np.float64))
    result["spearman"] = lib.spearman_ranked(candidate_flat, np.asarray(cache["reference_rank"], dtype=np.float64))
    result["status"] = "ok"
    if order_statistics:
        reference_sorted = np.asarray(cache["reference_sorted"], dtype=np.float64)
        result["per_order_statistic"] = {
            "order_%d" % k: {"pearson": lib.metric("pearson", candidate_sorted[k], reference_sorted[k]),
                             "spearman": lib.metric("spearman", candidate_sorted[k], reference_sorted[k])}
            for k in range(4)}
    return result


# ------------------------------------------------------------- 元数据抽取

METADATA_FIELDS = ("wall_s", "canonical_grad", "p", "q", "q_source", "outer_fg", "full_j",
                   "count_nll_by_loss", "full_j_components", "terminal", "outer_fg_actual", "fg_cap")


def metadata_of(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """把 manifest 已保留的元数据传给 dataset（不用 None 覆盖已知值）。"""
    row = row or {}
    out: dict[str, Any] = {}
    for field in METADATA_FIELDS:
        value = row.get(field)
        if value is None:
            raw = row.get("raw_record") or {}
            for alias in inputs.ALIASES.get(field, ()):
                if raw.get(alias) is not None:
                    value = raw[alias]
                    break
        out[field] = value
    raw = row.get("raw_record") or {}
    out["own_count_nll"] = own_count_nll(row)
    out["own_full_j"] = out.get("full_j")
    out["common_G_count"] = common_g_count(row)
    out["common_G_fullJ"] = common_g_full_j(row)
    out["common_G_fields_present"] = bool(out["common_G_count"] is not None or out["common_G_fullJ"] is not None)
    out["raw_full_j_keys"] = {key: raw[key] for key in ("fullJ_A", "fullJ_B", "fullJ_C", "fullJ_by_loss",
                                                       "full_j_by_loss", "fullJ", "fullJ_value")
                              if key in raw}
    return out


def own_count_nll(row: Mapping[str, Any] | None) -> float | None:
    row = row or {}
    loss = str(row.get("loss") or "")
    block = row.get("count_nll_by_loss")
    if not loss or not isinstance(block, Mapping):
        return None
    entry = block.get(loss)
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, Mapping):
        for key in ("own_count", "count_nll", "value", "nll"):
            if isinstance(entry.get(key), (int, float)):
                return float(entry[key])
    return None


def common_g_count(row: Mapping[str, Any] | None) -> float | None:
    """共同原 G count：count_nll_by_loss['A']（预 ref 封存字段，不重算）。"""
    row = row or {}
    block = row.get("count_nll_by_loss")
    if not isinstance(block, Mapping):
        return None
    entry = block.get("A")
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, Mapping):
        for key in ("own_count", "count_nll", "value", "nll"):
            if isinstance(entry.get(key), (int, float)):
                return float(entry[key])
    return None


def common_g_full_j(row: Mapping[str, Any] | None) -> float | None:
    """共同原 G fullJ：raw_record['fullJ_A'] 或 fullJ_by_loss['A']。"""
    raw = (row or {}).get("raw_record") or {}
    for key in ("fullJ_A", "full_j_A"):
        if isinstance(raw.get(key), (int, float)):
            return float(raw[key])
    for key in ("fullJ_by_loss", "full_j_by_loss", "fullJ_components_by_loss"):
        block = raw.get(key)
        if isinstance(block, Mapping) and isinstance(block.get("A"), (int, float)):
            return float(block["A"])
    return None


# ------------------------------------------------------------- 数据集指标


def radius_of_gyration(coords: np.ndarray) -> dict[str, Any]:
    beads = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    center = beads.mean(axis=0)
    radius = np.sqrt(np.sum((beads - center) ** 2, axis=1))
    return {"whole_cell_Rg": float(np.sqrt(np.mean(radius ** 2))), "beads": int(len(beads)),
            "max_radius": float(np.linalg.norm(beads, axis=1).max())}


def dataset_metrics(coords: np.ndarray, reference: np.ndarray, data: lib.Aggregate,
                    masks: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Any],
                    per_chr_valid: Sequence[np.ndarray], names: Sequence[str],
                    permutation_draws: int = lib.PERMUTATION_DRAWS) -> dict[str, Any]:
    r2 = lib.r2_result(coords, reference, names, data.offsets, masks)
    inter = inter_metrics(coords, cache, order_statistics=True)
    cand_centers, _ = spatial.merged_chr_centers(coords, per_chr_valid)
    ref_centers, _ = spatial.merged_chr_centers(reference, per_chr_valid)
    centers = spatial.center_metrics(cand_centers, ref_centers, names)
    permutation = spatial.label_permutation_null(cand_centers, ref_centers, centers["spearman"],
                                                 draws=permutation_draws)
    copy_centers = spatial.copy_center_metrics(coords, reference, per_chr_valid, masks, names, data.offsets)
    return {"r2": r2, "inter": inter,
            "spatial": {"merged_chr_centers": centers, "label_permutation_null": permutation,
                        "copy_centers": copy_centers},
            "geometry": radius_of_gyration(coords)}


def null_metrics(coords: np.ndarray, reference: np.ndarray, data: lib.Aggregate,
                 masks: Mapping[str, Mapping[str, Any]], cache: Mapping[str, Any],
                 per_chr_valid: Sequence[np.ndarray], names: Sequence[str]) -> dict[str, Any]:
    r2 = lib.r2_result(coords, reference, names, data.offsets, masks)
    inter = inter_metrics(coords, cache, order_statistics=False)
    cand_centers, _ = spatial.merged_chr_centers(coords, per_chr_valid)
    ref_centers, _ = spatial.merged_chr_centers(reference, per_chr_valid)
    return {"r2": r2, "inter": inter,
            "spatial": {"merged_chr_centers": spatial.center_metrics(cand_centers, ref_centers, names)},
            "geometry": radius_of_gyration(coords)}


# ---------------------------------------------------------------- bootstrap


def bootstrap_indices() -> tuple[np.ndarray, dict[str, Any]]:
    indices = np.random.default_rng(lib.BOOTSTRAP_SEED).integers(0, 20, size=(lib.BOOTSTRAP_DRAWS, 20), dtype=np.int64)
    evidence: dict[str, Any] = {"seed": lib.BOOTSTRAP_SEED, "draws": lib.BOOTSTRAP_DRAWS, "shape": list(indices.shape)}
    if lib.BOOTSTRAP_MATRIX_046.is_file():
        frozen = np.load(lib.BOOTSTRAP_MATRIX_046)
        evidence["046_matrix_path"] = lib.rel(lib.BOOTSTRAP_MATRIX_046)
        evidence["046_matrix_sha256"] = lib.sha256_file(lib.BOOTSTRAP_MATRIX_046)
        evidence["same_index_matrix_as_046"] = bool(frozen.shape == indices.shape and np.array_equal(frozen, indices))
        if not evidence["same_index_matrix_as_046"]:
            raise RuntimeError("bootstrap index matrix does not reproduce 046's frozen matrix")
    else:
        evidence["046_matrix_path"] = None
        evidence["same_index_matrix_as_046"] = None
    return indices, evidence


def bootstrap_fixed(left: Sequence[Any], right: Sequence[Any], names: Sequence[str],
                    indices: np.ndarray) -> dict[str, Any]:
    a = np.asarray([float(v) if v is not None and math.isfinite(float(v)) else np.nan for v in left], dtype=np.float64)
    b = np.asarray([float(v) if v is not None and math.isfinite(float(v)) else np.nan for v in right], dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    diff = a - b
    output: dict[str, Any] = {
        "denominator_chromosomes": 20, "defined_chromosomes": int(np.count_nonzero(finite)),
        "defined_chromosome_names": [str(n) for n, keep in zip(names, finite) if keep],
        "undefined_chromosomes": [str(n) for n, keep in zip(names, finite) if not keep],
        "seed": lib.BOOTSTRAP_SEED, "draws": lib.BOOTSTRAP_DRAWS,
        "unit": "paired chromosome technical/structural bootstrap within one cell; not biological replicates",
        "fixed_index_matrix_shape": list(indices.shape),
        "full_20_estimate_defined": bool(np.all(finite)),
    }
    if np.all(finite):
        draws = diff[indices].mean(axis=1)
        output.update({"mean": float(diff.mean()),
                       "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                       "draw_summary": {"mean": float(draws.mean()), "std": float(draws.std(ddof=1)),
                                        "min": float(draws.min()), "max": float(draws.max())}})
    else:
        output.update({"mean": None, "ci95": [None, None], "draw_summary": None,
                       "note": "full-20 estimate undefined; all-20 macro reported as NA, no silent chromosome dropping"})
    defined_diff = diff[finite]
    output["defined_only_descriptive"] = {
        "label": "defined-only descriptive; not the fixed-20 estimate",
        "n": int(len(defined_diff)),
        "mean": float(defined_diff.mean()) if len(defined_diff) else None,
        "std": float(defined_diff.std(ddof=1)) if len(defined_diff) > 1 else None,
    }
    output["left_wins"] = int(np.count_nonzero(finite & (a > b + lib.GEOMETRY_TIE_TOL)))
    output["right_wins"] = int(np.count_nonzero(finite & (b > a + lib.GEOMETRY_TIE_TOL)))
    output["ties"] = int(np.count_nonzero(finite & (np.abs(a - b) <= lib.GEOMETRY_TIE_TOL)))
    return output


def paired_comparison(rows: Mapping[str, Mapping[str, Any]], left_id: str, right_id: str, label: str,
                      indices: np.ndarray) -> dict[str, Any]:
    left, right = rows[left_id], rows[right_id]
    names = [str(row["chromosome"]) for row in left["r2"]["per_chromosome"]]
    left_rows = {row["chromosome"]: row for row in left["r2"]["per_chromosome"]}
    right_rows = {row["chromosome"]: row for row in right["r2"]["per_chromosome"]}
    output: dict[str, Any] = {"label": label, "left": left_id, "right": right_id,
                             "denominator_chromosomes": 20, "metrics": {}}
    for metric_name in ("pearson", "spearman"):
        for field in lib.R2_FIELDS:
            left_values = [left_rows[name]["metrics"][metric_name].get(field) for name in names]
            right_values = [right_rows[name]["metrics"][metric_name].get(field) for name in names]
            output["metrics"]["%s:%s" % (metric_name, field)] = bootstrap_fixed(left_values, right_values, names, indices)
    return output


def comparison_plan(fit_ids: Sequence[str]) -> list[tuple[str, str, str]]:
    plan: list[tuple[str, str, str]] = []
    for fit_id in fit_ids:
        plan.append((fit_id, "baseline-046-real-extension-G-full-J", "endpoint-minus-baseline"))
    for source in lib.SOURCES:
        for solver in lib.SOLVERS:
            plan.append(("%s-%s-%s" % ("B", solver, source), "%s-%s-%s" % ("A", solver, source), "same-source B-minus-A"))
            plan.append(("%s-%s-%s" % ("C", solver, source), "%s-%s-%s" % ("A", solver, source), "same-source C-minus-A"))
    for loss in lib.LOSES:
        for source in lib.SOURCES:
            plan.append(("%s-ms-%s" % (loss, source), "%s-raw-%s" % (loss, source), "same-source ms-minus-raw"))
    for source in lib.SOURCES:
        plan.append(("initial-%s" % source, "baseline-046-real-extension-G-full-J", "initial-minus-baseline"))
    return plan


# ------------------------------------------------------------------- 输出


def r2_rows(dataset_id: str, kind: str, terminal: str, r2: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_chr: list[dict[str, Any]] = []
    for row in r2["per_chromosome"]:
        for metric_name in ("pearson", "spearman"):
            entry = row["metrics"][metric_name]
            per_chr.append({
                "candidate_id": dataset_id, "kind": kind, "terminal": terminal,
                "chromosome": row["chromosome"], "metric": metric_name, "n_pairs": row["n_pairs"],
                "direct": entry["direct"], "swapped": entry["swapped"], "matched": entry["matched"],
                "cross": entry["cross"], "contrast": entry["contrast"], "orientation": entry["orientation"],
                "geometry_tie": entry["geometry_tie"], "copy_A_margin": entry["copy_A_margin"],
                "copy_B_margin": entry["copy_B_margin"], "matched_mat_margin": entry["matched_mat_margin"],
                "matched_pat_margin": entry["matched_pat_margin"], "min_margin": entry["min_margin"],
                "derived_defined": entry["derived_defined"],
                "rho": json.dumps(entry["rho"], sort_keys=True) if entry["rho"] else "NA",
                "rho_A_mat": (entry["rho"] or {}).get("A_mat"), "rho_A_pat": (entry["rho"] or {}).get("A_pat"),
                "rho_B_mat": (entry["rho"] or {}).get("B_mat"), "rho_B_pat": (entry["rho"] or {}).get("B_pat"),
            })
    summary: list[dict[str, Any]] = []
    for metric_name in ("pearson", "spearman"):
        row: dict[str, Any] = {"candidate_id": dataset_id, "kind": kind, "terminal": terminal, "metric": metric_name,
                               "defined_chromosomes": r2["defined_chromosome_counts"][metric_name]["matched"],
                               "denominator_chromosomes": 20,
                               "full_20_estimate_defined": r2["macro_full_20_defined"][metric_name]["matched"]}
        for field in lib.R2_FIELDS + ("direct", "swapped"):
            # 主列 = fixed-20（任一 chr 缺失即 NA）；defined-only 另列且明确命名
            row[field] = r2["macro_full_20_all_chromosomes_required"][metric_name][field]
            row["defined_only_%s" % field] = r2["macro_equal_chromosome_weight_defined_only"][metric_name][field]
            row["%s_defined_n" % field] = r2["defined_chromosome_counts"][metric_name][field]
            row["%s_full_20_defined" % field] = r2["macro_full_20_defined"][metric_name][field]
        summary.append(row)
    return per_chr, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--authorized-by-parent", required=True)
    parser.add_argument("--smoke", action="store_true", help="limited development run; marked smoke=true, never formal")
    parser.add_argument("--limit-datasets", type=int, default=None)
    parser.add_argument("--limit-nulls", type=int, default=None)
    parser.add_argument("--permutation-draws", type=int, default=lib.PERMUTATION_DRAWS)
    args = parser.parse_args()

    gate = load_gate(args.authorized_by_parent)
    data = lib.Aggregate()
    aggregate_audit = data.audit()
    if lib.sha256_file(lib.AGGREGATE_1MB) != lib.AGGREGATE_1MB_SHA256:
        raise RuntimeError("045 aggregate hash changed after the gate")
    if lib.sha256_file(lib.resolve_path(gate["manifest"]["path"])) != gate["manifest"]["sha256"]:
        raise RuntimeError("endpoint manifest changed after the gate")
    if lib.sha256_file(lib.resolve_path(gate["selection"]["path"])) != gate["selection"]["sha256"]:
        raise RuntimeError("selection file changed after the gate")

    manifest = inputs.load_manifest(Path(args.manifest))
    selection = inputs.load_selection(Path(args.selection))
    reference, reference_record = open_reference(gate)
    masks, mask_audit = load_real_masks(data)
    mask_audit["reference_support_finite"] = assert_reference_finite_on_mask(reference, masks, data)
    cache = build_inter_cache(data, masks, reference)
    per_chr_valid = spatial.per_chromosome_valid_indices(data, masks)
    names = list(data.chromosome_names)

    datasets: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in manifest["fits"]:
        fit_id = str(row["fit_id"])
        payload = lib.load_coords_npz(lib.resolve_path(row["npz_path"]))
        if payload["coordinate_array_sha256"] != gate_sha(gate, fit_id):
            raise RuntimeError("endpoint coordinates changed after the gate: %s" % fit_id)
        metadata = metadata_of(row)
        # manifest 缺失 p/q 时回退到 NPZ 内的 p/q（只补齐，不用 None 覆盖已知值）
        for field in ("p", "q"):
            if metadata.get(field) is None and field in payload:
                metadata[field] = float(np.asarray(payload[field]).item())
                metadata["%s_from_npz_fallback" % field] = True
        if metadata.get("p") is not None:
            metadata["p_from_q_bit_exact"] = bool(
                abs(float(contact_q_from_p(float(metadata["p"]))) - float(metadata["q"])) < 1e-12
                if metadata.get("q") is not None else False)
        datasets[fit_id] = {"candidate_id": fit_id, "kind": "formal_fit", "loss": row["loss"], "solver": row["solver"],
                            "source": row["source"], "terminal": str(row["terminal"]), "coords": payload["coordinates"],
                            "coordinate_array_sha256": payload["coordinate_array_sha256"],
                            "npz_path": lib.rel(lib.resolve_path(row["npz_path"])),
                            **metadata}
        order.append(fit_id)
    initial_records: dict[str, Mapping[str, Any]] = {}
    for row in manifest["initials"]:
        initial_records[inputs_initial_name(row)] = row
    for name in lib.SOURCES:
        path = lib.INITIAL_NPZ[name]
        payload = lib.load_coords_npz(path)
        dataset_id = "initial-%s" % name
        record = initial_records.get(name)
        datasets[dataset_id] = {"candidate_id": dataset_id, "kind": "initial_control",
                                "loss": (record or {}).get("loss"), "solver": (record or {}).get("solver"),
                                "source": name, "terminal": str((record or {}).get("terminal")
                                                                or "initial_zero_optimization"),
                                "coords": payload["coordinates"],
                                "coordinate_array_sha256": payload["coordinate_array_sha256"],
                                "npz_path": lib.rel(path),
                                **metadata_of(record)}
        order.append(dataset_id)
    baseline_payload = lib.load_coords_npz(lib.BASELINE_NPZ)
    if baseline_payload["coordinate_array_sha256"] != lib.BASELINE_COORD_ARRAY_SHA256:
        raise RuntimeError("046 baseline coordinates changed")
    baseline_metadata = metadata_of(manifest["baseline"])
    if baseline_metadata.get("outer_fg") is None:
        baseline_metadata["outer_fg"] = 1988
    datasets["baseline-046-real-extension-G-full-J"] = {
        "candidate_id": "baseline-046-real-extension-G-full-J", "kind": "baseline", "loss": "A", "solver": "raw",
        "source": "random", "terminal": str((manifest["baseline"] or {}).get("terminal") or "budget_not_converged"),
        "coords": baseline_payload["coordinates"],
        "coordinate_array_sha256": baseline_payload["coordinate_array_sha256"], "npz_path": lib.rel(lib.BASELINE_NPZ),
        **baseline_metadata}
    order.append("baseline-046-real-extension-G-full-J")
    if args.limit_datasets:
        order = order[:args.limit_datasets]

    for dataset_id in order:
        entry = datasets[dataset_id]
        metrics = dataset_metrics(entry["coords"], reference, data, masks, cache, per_chr_valid, names,
                                  permutation_draws=args.permutation_draws)
        entry.update(metrics)
        print("  evaluated %s" % dataset_id, flush=True)

    indices, bootstrap_evidence = bootstrap_indices()
    rows = {dataset_id: datasets[dataset_id] for dataset_id in order}
    comparisons: list[dict[str, Any]] = []
    for left, right, label in comparison_plan([d for d in order if d in lib.EXPECTED_FIT_IDS]):
        if left not in rows or right not in rows:
            continue
        comparisons.append(paired_comparison(rows, left, right, label, indices))

    null_evaluations: list[dict[str, Any]] = []
    null_sources = gate["nulls_new"] + gate["nulls_reused_baseline"]
    if args.limit_nulls:
        null_sources = null_sources[:args.limit_nulls]
    for record in null_sources:
        path = lib.resolve_path(record["path"])
        actual = lib.sha256_file(path)
        if actual != str(record["sha256"]):
            raise RuntimeError("null file changed after the gate: %s" % path)
        with np.load(path, allow_pickle=False) as payload:
            coords = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        if lib.hash_array(coords) != str(record["coordinate_array_sha256"]):
            raise RuntimeError("null coordinate array changed: %s" % path)
        metrics = null_metrics(coords, reference, data, masks, cache, per_chr_valid, names)
        null_evaluations.append({"source_candidate_id": record["source_candidate_id"], "null_kind": record["null_kind"],
                                 "seed": record.get("seed"), "global_scale": record.get("global_scale"),
                                 "reused_from": record.get("reused_from"), "path": lib.rel(path),
                                 **metrics})
        done = len(null_evaluations)
        if done % 10 == 0 or done == len(null_sources):
            print("  null progress %d/%d" % (done, len(null_sources)), flush=True)
    expected_null_total = len(gate["nulls_new"]) + len(gate["nulls_reused_baseline"])
    if expected_null_total != 153 or len(gate["nulls_new"]) != 136 or len(gate["nulls_reused_baseline"]) != 17:
        raise RuntimeError("gate null accounting broken: new=%d reused=%d total=%d (expected 136+17=153)"
                           % (len(gate["nulls_new"]), len(gate["nulls_reused_baseline"]), expected_null_total))
    if not args.limit_nulls and len(null_evaluations) != expected_null_total:
        raise RuntimeError("evaluated %d null draws, expected %d" % (len(null_evaluations), expected_null_total))
    print("  evaluated %d null draws (new %d + reused baseline %d)"
          % (len(null_evaluations), len(gate["nulls_new"]), len(gate["nulls_reused_baseline"])), flush=True)

    write_results(data, aggregate_audit, gate, reference_record, mask_audit, cache, datasets, order,
                  comparisons, indices, bootstrap_evidence, null_evaluations, args, selection)
    validation_and_terminal(gate, reference_record, mask_audit, datasets, order, comparisons,
                            bootstrap_evidence, null_evaluations, args)
    print("EVALUATION %s" % ("SMOKE COMPLETE (not formal)" if args.smoke else "COMPLETE"), flush=True)
    return 0


def baseline_regression(datasets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """与 046 既有 baseline 指标表做 1e-12 parity（expected 在 pre-reference 阶段从既有 TSV 提取）。"""
    path = RESULTS / "baseline_regression_expected.json"
    expected = lib.read_json(path)
    baseline = datasets["baseline-046-real-extension-G-full-J"]
    atol = 1e-12
    mismatches: list[dict[str, Any]] = []
    checked = 0
    by_chr = {row["chromosome"]: row for row in baseline["r2"]["per_chromosome"]}
    for name, per_metric in expected["per_chromosome_four_rho"].items():
        if name not in by_chr:
            mismatches.append({"scope": "per_chromosome", "chromosome": name, "reason": "missing"})
            continue
        for metric_name, rho in per_metric.items():
            mine = by_chr[name]["metrics"][metric_name]["rho"]
            for key in ("A_mat", "A_pat", "B_mat", "B_pat"):
                checked += 1
                if mine.get(key) is None or abs(float(mine[key]) - float(rho[key])) > atol:
                    mismatches.append({"scope": "per_chromosome", "chromosome": name, "metric": metric_name,
                                       "key": key, "mine": mine.get(key), "expected": rho[key]})
    macro = baseline["r2"]["macro_equal_chromosome_weight_defined_only"]
    for key, expected_value in (("pearson:matched", expected["pearson_matched"]),
                                ("pearson:cross", expected["pearson_cross"]),
                                ("spearman:matched", expected["spearman_matched"]),
                                ("spearman:cross", expected["spearman_cross"])):
        metric_name, field = key.split(":")
        mine = macro[metric_name][field]
        checked += 1
        if mine is None or abs(float(mine) - float(expected_value)) > atol:
            mismatches.append({"scope": "macro", "field": key, "mine": mine, "expected": expected_value})
    inter = baseline["inter"]
    for key, expected_value in (("pearson", expected["inter_pearson"]), ("spearman", expected["inter_spearman"])):
        checked += 1
        if inter.get(key) is None or abs(float(inter[key]) - float(expected_value)) > atol:
            mismatches.append({"scope": "inter", "field": key, "mine": inter.get(key), "expected": expected_value})
    return {"status": "PASS" if not mismatches else "FAIL", "atol": atol, "checked_values": checked,
            "mismatch_count": len(mismatches), "mismatches": mismatches[:50],
            "expected_source": expected.get("source"), "expected_path": lib.rel(path),
            "definition": "046 frozen baseline (real-extension-G-full-J) re-measured under the 049 evaluation code"}


def validation_and_terminal(gate: Mapping[str, Any], reference_record: Mapping[str, Any],
                            mask_audit: Mapping[str, Any], datasets: Mapping[str, Mapping[str, Any]],
                            order: Sequence[str], comparisons: Sequence[Mapping[str, Any]],
                            bootstrap_evidence: Mapping[str, Any], null_evaluations: Sequence[Mapping[str, Any]],
                            args: argparse.Namespace) -> None:
    regression = baseline_regression(datasets)
    baseline_entry = datasets.get("baseline-046-real-extension-G-full-J")
    checks = {
        "reference_sha256_verified": reference_record["reference_sha256"] == lib.REFERENCE_SHA256,
        "reference_opened_after_gate": bool(gate.get("reference_first_opened_utc")),
        "phase_columns_read": False,
        "mask_totals_frozen": mask_audit["totals"],
        "mask_snapshot_sha256": mask_audit["snapshot_sha256"],
        "mask_config_sha256_typo_recorded": mask_audit.get("config_sha256_differs_by_transcription_typo"),
        "mask_sha256_matches_046_creation_record": bool(
            mask_audit.get("snapshot_sha256_recorded_by_046_at_creation") == mask_audit["snapshot_sha256"]),
        "mask_audit_status": mask_audit["status"],
        "global_index_rule": mask_audit["global_index_rule"],
        "bootstrap_index_matrix_matches_046": bootstrap_evidence.get("same_index_matrix_as_046"),
        "baseline_regression": regression["status"],
        "baseline_regression_checked_values": regression["checked_values"],
        "dataset_count": len(order),
        "comparison_count": len(comparisons),
        "null_draws_evaluated": len(null_evaluations),
        "smoke": bool(args.smoke),
    }
    status = "PASS"
    if baseline_entry is None or regression["status"] != "PASS" or mask_audit["status"] != "PASS":
        status = "FAIL"
    if args.smoke:
        status = "SMOKE_ONLY_NOT_FORMAL"
    deviations = list(gate.get("deviations", [])) + [{
        "item": "049/config.json mask sha256 transcription typo",
        "detail": ("config.json 记为 %s，文件实际为 %s；文件哈希与 046 evaluation_final/results/mask_validation.json 中创建时记录"
                   "（frozen_legacy_mask_worker.output_sha256）一致，文件未被修改。评价侧只记录、不改 config（不在写范围）。"
                   % (mask_audit.get("snapshot_sha256_expected_in_049_config"), mask_audit["snapshot_sha256"])),
        "impact": "mask 内容与冻结计数（176201/157529/2447/2835152）一致，评价数值不受影响",
    }]
    validation = {"schema": "p9016-049-evaluation-validation-v1", "status": status, "created_utc": lib.utc_now(),
                  "checks": checks, "baseline_regression": regression, "deviations": deviations,
                  "smoke_note": ("limited development run; numbers are not the formal evaluation" if args.smoke else None)}
    lib.write_json(RESULTS / "validation.json", validation)
    terminal = {
        "schema": "p9016-049-evaluation-terminal-v1", "created_utc": lib.utc_now(),
        "evaluation_terminal": ("smoke_complete" if args.smoke else "complete"),
        "exit_code_zero_is_not_scientific_success": True,
        "reference_first_opened_utc": reference_record["reference_first_opened_utc"],
        "gate": {"path": lib.rel(GATE_PATH), "authorized_by_parent": gate.get("authorized_by_parent"),
                 "terminal_counts_at_gate": gate.get("terminal_counts")},
        "datasets": [{"candidate_id": dataset_id, "kind": datasets[dataset_id]["kind"],
                      "source_terminal": datasets[dataset_id]["terminal"],
                      "outer_fg": datasets[dataset_id].get("outer_fg"),
                      "npz_path": datasets[dataset_id].get("npz_path")} for dataset_id in order],
        "nulls": {"new_files": len(gate["nulls_new"]), "reused_baseline_files": len(gate["nulls_reused_baseline"]),
                  "expected_total": 153, "accounting_ok": len(null_evaluations) == 153 and not args.smoke,
                  "draws_evaluated": len(null_evaluations),
                  "draw_policy": "u0 + 16 within-chromosome u permutations; technical draws within one cell, not biological replicates"},
        "bootstrap": dict(bootstrap_evidence),
        "comparisons": len(comparisons),
        "cost_note": "046 baseline cumulative 1988 FG include 5Mb/2Mb stages; this round uses 1502 all-1Mb full-grid FG per fit; wall costs are not equal",
        "not_reported_this_round": ["R1", "R3", "L2 claims", "phase columns"],
        "validation_status": status,
        "deviations": deviations if "deviations" in dir() else gate.get("deviations", []),
        "reference_open_attempts": reference_record.get("open_attempt_count"),
        "reference_first_opened_utc": reference_record.get("reference_first_opened_utc"),
    }
    lib.write_json(RESULTS / "terminal.json", terminal)


def contact_q_from_p(p: float) -> float:
    """与 pr.contact_model.q_from_p 同一约定（P_FLOOR=1e-6），仅用于一致性标注，不参与任何指标。"""
    floor = 1e-6
    z = (float(p) - floor) / (1.0 - 2.0 * floor)
    return float(math.log(z / (1.0 - z))) if 0.0 < z < 1.0 else float("nan")


def inputs_initial_name(row: Mapping[str, Any]) -> str:
    source = row.get("source")
    if source in lib.SOURCES:
        return str(source)
    fit_id = str(row.get("fit_id") or "")
    for name in lib.SOURCES:
        if fit_id.endswith(name):
            return name
    raise RuntimeError("cannot determine initial control name from manifest entry: %s" % sorted(row.keys()))


def dataset_common_g_row(dataset_id: str, entry: Mapping[str, Any], order: Sequence[str]) -> dict[str, Any]:
    return {
        "candidate_id": dataset_id, "kind": entry["kind"], "terminal": entry.get("terminal"),
        "loss": entry.get("loss"), "solver": entry.get("solver"), "source": entry.get("source"),
        "own_count_nll": entry.get("own_count_nll"), "own_full_j": entry.get("own_full_j"),
        "common_G_count": entry.get("common_G_count"), "common_G_fullJ": entry.get("common_G_fullJ"),
        "common_G_fields_present": entry.get("common_G_fields_present"),
        "outer_fg": entry.get("outer_fg"), "wall_s": entry.get("wall_s"),
        "canonical_grad": entry.get("canonical_grad"), "p": entry.get("p"), "q": entry.get("q"),
        "q_source": entry.get("q_source"), "npz_path": entry.get("npz_path"),
        "count_nll_by_loss": json.dumps(entry.get("count_nll_by_loss"), sort_keys=True)
        if entry.get("count_nll_by_loss") is not None else None,
    }


def gate_sha(gate: Mapping[str, Any], fit_id: str) -> str:
    for row in gate["fits"]:
        if str(row["fit_id"]) == fit_id:
            return str(row["coordinate_array_sha256"])
    raise RuntimeError("fit %s missing from the gate record" % fit_id)


def write_results(data: lib.Aggregate, aggregate_audit: Mapping[str, Any], gate: Mapping[str, Any],
                  reference_record: Mapping[str, Any], mask_audit: Mapping[str, Any], cache: Mapping[str, Any],
                  datasets: Mapping[str, Mapping[str, Any]], order: Sequence[str],
                  comparisons: Sequence[Mapping[str, Any]], indices: np.ndarray,
                  bootstrap_evidence: Mapping[str, Any], null_evaluations: Sequence[Mapping[str, Any]],
                  args: argparse.Namespace, selection: Mapping[str, Any]) -> None:
    per_chr_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    inter_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    for dataset_id in order:
        entry = datasets[dataset_id]
        rows_chr, rows_summary = r2_rows(dataset_id, entry["kind"], entry["terminal"], entry["r2"])
        per_chr_rows.extend(rows_chr)
        summary_rows.extend(rows_summary)
        inter = entry["inter"]
        inter_row: dict[str, Any] = {"candidate_id": dataset_id, "kind": entry["kind"], "terminal": entry["terminal"],
                                     "pearson": inter.get("pearson"), "spearman": inter.get("spearman"),
                                     "eligible_locus_pairs": inter["eligible_locus_pairs"],
                                     "denominator": inter["denominator"], "valid_bin_count": inter["valid_bin_count"],
                                     "nonfinite_value_count": inter["nonfinite_value_count"], "status": inter["status"]}
        for k in range(4):
            block = (inter.get("per_order_statistic") or {}).get("order_%d" % k, {})
            inter_row["pearson_order_%d" % k] = block.get("pearson")
            inter_row["spearman_order_%d" % k] = block.get("spearman")
        inter_rows.append(inter_row)
        centers = entry["spatial"]["merged_chr_centers"]
        copy_centers = entry["spatial"]["copy_centers"]
        permutation = entry["spatial"]["label_permutation_null"]
        primary = copy_centers["primary"]
        alternative = copy_centers["unresolved_alternative"]
        spatial_rows.append({
            "candidate_id": dataset_id, "kind": entry["kind"], "terminal": entry["terminal"],
            "center_pearson_190": centers["pearson"], "center_spearman_190": centers["spearman"],
            "center_optimal_scale": centers["optimal_single_scale"], "center_normalized_stress": centers["normalized_stress"],
            "center_per_chr_profile_macro_pearson": centers["per_chr_profile_macro_pearson"],
            "center_per_chr_profile_macro_spearman": centers["per_chr_profile_macro_spearman"],
            "center_top3_overlap_mean": centers["top3_neighbors"]["mean_overlap_top3"],
            "label_permutation_p_ge_observed": permutation["one_sided_p_ge_observed"],
            "label_permutation_null_mean_spearman": permutation["spearman_null"]["mean"],
            "copy_center_n_distances_760": primary["n_distances"],
            "copy_center_pearson_760": primary["pearson"], "copy_center_spearman_760": primary["spearman"],
            "copy_center_normalized_stress": primary["normalized_stress"],
            "copy_center_unresolved_chromosomes": ";".join(copy_centers["unresolved_chromosomes"]) or "none",
            "copy_center_alternative_pearson_760": (alternative or {}).get("pearson"),
            "copy_center_alternative_spearman_760": (alternative or {}).get("spearman"),
            "copy_center_alternative_normalized_stress": (alternative or {}).get("normalized_stress"),
            "procrustes_proper_normalized_rmsd": primary["procrustes_proper"]["normalized_aligned_rmsd"],
            "procrustes_proper_det": primary["procrustes_proper"]["rotation_det"],
            "procrustes_reflection_normalized_rmsd": primary["procrustes_reflection"]["normalized_aligned_rmsd"],
            "whole_cell_Rg": entry["geometry"]["whole_cell_Rg"],
        })

    null_rows: list[dict[str, Any]] = []
    for entry in null_evaluations:
        # 主列 = fixed-20 口径（任一 chr 未定义即 NA，例如 u0 的 margin）；defined-only 另列
        pearson = entry["r2"]["macro_full_20_all_chromosomes_required"]["pearson"]
        spearman = entry["r2"]["macro_full_20_all_chromosomes_required"]["spearman"]
        pearson_only = entry["r2"]["macro_equal_chromosome_weight_defined_only"]["pearson"]
        spearman_only = entry["r2"]["macro_equal_chromosome_weight_defined_only"]["spearman"]
        row: dict[str, Any] = {"source_candidate_id": entry["source_candidate_id"], "null_kind": entry["null_kind"],
                               "seed": entry["seed"], "global_scale": entry["global_scale"],
                               "reused_from": entry.get("reused_from"), "path": entry["path"],
                               "inter_pearson": entry["inter"].get("pearson"), "inter_spearman": entry["inter"].get("spearman"),
                               "center_spearman_190": entry["spatial"]["merged_chr_centers"]["spearman"],
                               "whole_cell_Rg": entry["geometry"]["whole_cell_Rg"],
                               "r2_defined_chromosomes": entry["r2"]["defined_chromosome_counts"]["pearson"]["matched"]}
        for metric_name, block, block_only in (("pearson", pearson, pearson_only), ("spearman", spearman, spearman_only)):
            for field in lib.R2_FIELDS:
                row["%s_%s" % (metric_name, field)] = block.get(field)
                row["%s_%s_defined_only" % (metric_name, field)] = block_only.get(field)
        null_rows.append(row)

    null_summary: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in null_rows:
        by_source.setdefault(str(row["source_candidate_id"]), []).append(row)
    metrics_of_interest = ["pearson_matched", "pearson_cross", "pearson_contrast", "pearson_min_margin",
                          "spearman_matched", "spearman_cross", "spearman_contrast", "spearman_min_margin",
                          "inter_pearson", "inter_spearman", "center_spearman_190", "whole_cell_Rg", "global_scale"]
    for source_id, rows in sorted(by_source.items()):
        for kind in ("u_zero", "random_u"):
            subset = [row for row in rows if row["null_kind"] == kind]
            for metric_name in metrics_of_interest:
                values = [float(row[metric_name]) for row in subset
                          if row.get(metric_name) is not None and math.isfinite(float(row[metric_name]))]
                null_summary.append({
                    "source_candidate_id": source_id, "null_kind": kind, "draw_count": len(subset),
                    "metric": metric_name, "defined_draws": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "min": float(np.min(values)) if values else None,
                    "max": float(np.max(values)) if values else None,
                    "range": float(np.max(values) - np.min(values)) if values else None,
                })

    common_rows = [dataset_common_g_row(dataset_id, datasets[dataset_id], order) for dataset_id in order]
    lib.write_tsv(RESULTS / "endpoint_common_g.tsv", common_rows)
    evaluation = {
        "schema": "p9016-049-evaluation-v1",
        "smoke": bool(args.smoke),
        "smoke_note": ("SMOKE/DEVELOPMENT RUN — limited datasets/nulls/draws; not the formal evaluation"
                       if args.smoke else None),
        "run_id": lib.RUN.name,
        "created_utc": lib.utc_now(),
        "reference": dict(reference_record),
        "aggregate": dict(aggregate_audit),
        "mask": dict(mask_audit),
        "gate": {"path": lib.rel(GATE_PATH), "sha256": lib.sha256_file(GATE_PATH),
                 "authorized_by_parent": gate.get("authorized_by_parent")},
        "bootstrap_indices": dict(bootstrap_evidence),
        "selection": {"path": lib.rel(Path(str(selection["path"]))), "sha256": selection["sha256"],
                      "source_selection": {k: v["selected_source"] for k, v in selection["source_selection"].items()},
                      "display_endpoints": {loss: selection["display"][loss]["fit_id"] for loss in lib.LOSES},
                      "consumes": "endpoint own count and terminal only; loss values are not cross-ranked"},
        "comparisons": list(comparisons),
        "endpoint_common_g": common_rows,
        "common_g_policy": ("common_G_count = manifest count_nll_by_loss['A']; common_G_fullJ = raw_record fullJ_A / "
                            "fullJ_by_loss['A']; own_count_nll / own_full_j are the fit's own loss values; "
                            "all consumed from pre-reference sealed fields, never recomputed"),
        "null_summary": null_summary,
        "null_draw_count": len(null_rows),
        "datasets": {dataset_id: {key: value for key, value in datasets[dataset_id].items() if key != "coords"}
                     for dataset_id in order},
        "null_draws": [{key: value for key, value in entry.items() if key != "coords"} for entry in null_evaluations],
        "support_policy": {
            "r2_and_inter": mask_audit["support_statement"],
            "spatial": mask_audit["support_statement"],
            "null_spatial": "null draws use the same helper and the same common support",
            "missing": "NA recorded; denominators are never shrunk and no chromosome is dropped",
        },
    }
    lib.write_json(RESULTS / "evaluation.json", evaluation)
    lib.write_tsv(RESULTS / "r2_per_chromosome.tsv", per_chr_rows)
    lib.write_tsv(RESULTS / "r2_summary.tsv", summary_rows)
    lib.write_tsv(RESULTS / "inter_summary.tsv", inter_rows)
    lib.write_tsv(RESULTS / "spatial_summary.tsv", spatial_rows)
    lib.write_tsv(RESULTS / "null_per_draw.tsv", null_rows)
    lib.write_tsv(RESULTS / "null_summary.tsv", null_summary)

    bootstrap_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        for metric_name, entry in comparison["metrics"].items():
            row = {"label": comparison["label"], "left": comparison["left"], "right": comparison["right"],
                   "metric_field": metric_name, "defined_chromosomes": entry["defined_chromosomes"],
                   "mean": entry["mean"], "ci95_low": entry["ci95"][0], "ci95_high": entry["ci95"][1],
                   "left_wins": entry["left_wins"], "right_wins": entry["right_wins"], "ties": entry["ties"]}
            bootstrap_rows.append(row)
    lib.write_tsv(RESULTS / "paired_bootstrap.tsv", bootstrap_rows)
    lib.write_json(RESULTS / "paired_bootstrap.json", {"schema": "p9016-049-paired-bootstrap-v1",
                                                       "indices": bootstrap_evidence, "comparisons": list(comparisons)})
    lib.write_json(RESULTS / "null_evaluations.json",
                   {"schema": "p9016-049-null-evaluations-v1", "draws": len(null_evaluations),
                    "summary": null_summary,
                    "note": "16 random-u draws are technical permutations within one cell, NOT biological replicates"})


if __name__ == "__main__":
    sys.exit(main())
