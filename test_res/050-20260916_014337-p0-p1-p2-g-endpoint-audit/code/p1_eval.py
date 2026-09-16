"""P1 评价入口（050 轮）：4 支 fork + 两个 base + 046 工作 baseline + u0/16 random-u 对照。

边界与顺序：

1. 先把全部候选 / base / baseline / null 坐标写出并哈希（`pre_reference_gate.json`），
   之后才打开 reference；
2. R2 用冻结 old21 mask：逐 chr 整条染色体只允许一次 A/B 交换，同一几何匹配同时用于
   Pearson 与 Spearman；matched / cross / contrast / min_margin 全部报出；
3. 每个 null 用**自己的可解析支持**，不与主候选共同支持取交集；u0 的 R3 tie / 方向未定义
   是预期 null，不进入主比较掩码；
4. paired bootstrap 复用 046 冻结的 seed 450301 / 10000×20 索引矩阵，固定 20 chr 分母，
   未定义 chr 保留在声明里；
5. 空间读出用 20 个合并中心（190 距）与 40 个 copy 中心跨 chr（760 距），
   不用 pooled inter 代替摆位。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
ROOT = RUN_DIR.parents[1]
S046_EVAL = ROOT / "test_res/046-UTC-real-cell-shared-capture/evaluation_final"
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
for _path in (str(S046_EVAL), str(S049), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import evaluator as ev  # noqa: E402  046 冻结评价器（只调用其只读函数）
import analysis_core as ac  # noqa: E402
from pr import contact_model  # noqa: E402

MASK_SNAPSHOT = S046_EVAL / "results/frozen_legacy_mask_snapshot.npz"
MASK_SNAPSHOT_SHA256 = "9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9"
BOOTSTRAP_INDICES = S046_EVAL / "bootstrap_indices_seed450301_10000x20.npy"
MASK_LOCK = ROOT / "docs/audits/multires-r2-preparation-20260914_041826/mask_lock.json"
BIN_SIZE_BP = 1_000_000
MASK_OFFSET_BP = 3_000_000
RANDOM_U_SEEDS = tuple(range(450500, 450516))
GEOMETRY_TIE_TOL = 1e-12

ENDPOINTS = {
    "A-raw-G-consensus": {"base": "G-consensus", "solver": "raw"},
    "A-ms-G-consensus": {"base": "G-consensus", "solver": "ms"},
    "A-raw-G-random": {"base": "G-random", "solver": "raw"},
    "A-ms-G-random": {"base": "G-random", "solver": "ms"},
}
EXTRA_SOURCES = {
    "046-base-G-consensus": "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-consensus/1Mb.3dg",
    "046-base-G-random": "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.3dg",
    "046-work-baseline-G-full-J": "test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.3dg",
}
PAIRS = [
    ("A-raw-G-consensus", "A-ms-G-consensus", "solver: raw-minus-ms at the G-consensus base"),
    ("A-raw-G-random", "A-ms-G-random", "solver: raw-minus-ms at the G-random base"),
    ("A-raw-G-consensus", "046-base-G-consensus", "fork: raw-minus-base at G-consensus"),
    ("A-ms-G-consensus", "046-base-G-consensus", "fork: ms-minus-base at G-consensus"),
    ("A-raw-G-random", "046-base-G-random", "fork: raw-minus-base at G-random"),
    ("A-ms-G-random", "046-base-G-random", "fork: ms-minus-base at G-random"),
    ("A-raw-G-consensus", "046-work-baseline-G-full-J", "new-minus-046-work-baseline"),
    ("A-ms-G-consensus", "046-work-baseline-G-full-J", "new-minus-046-work-baseline"),
    ("A-raw-G-random", "046-work-baseline-G-full-J", "new-minus-046-work-baseline"),
    ("A-ms-G-random", "046-work-baseline-G-full-J", "new-minus-046-work-baseline"),
]
R2_FIELDS = ("matched", "cross", "contrast", "copy_A_margin", "copy_B_margin",
             "matched_mat_margin", "matched_pat_margin", "min_margin")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes(order="C")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ev._jsonable(value), indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join("" if row.get(key) is None else str(row.get(key)) for key in columns) + "\n")


def candidate_npz_path(endpoint: str) -> Path:
    solver = ENDPOINTS[endpoint]["solver"]
    base = ENDPOINTS[endpoint]["base"]
    return RUN_DIR / "p1" / "coords" / ("A-%s-%s" % (solver, base)) / "1Mb.npz"


def candidate_3dg_path(endpoint: str) -> Path:
    solver = ENDPOINTS[endpoint]["solver"]
    base = ENDPOINTS[endpoint]["base"]
    return RUN_DIR / "p1" / "coords" / ("A-%s-%s" % (solver, base)) / "1Mb.3dg"


def load_full_grid_3dg(path: Path, data: Any) -> np.ndarray:
    tracks = ev._load_tracks(path, "3dg", tuple(data.chromosome_names))
    coords = np.full((2, int(data.n_loci), 3), np.nan, dtype=np.float64)
    for ci, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(ci)
        positions = np.asarray(data.locus_bin[slc], dtype=np.int64) * BIN_SIZE_BP
        for copy, suffix in enumerate(("a", "b")):
            rows = tracks.get("c%02d%s" % (ci + 1, suffix), {})
            for local, position in enumerate(positions):
                point = rows.get(int(position))
                if point is not None:
                    coords[copy, slc.start + local] = point
    if not np.isfinite(coords).all():
        raise RuntimeError("coordinate payload is not full-grid finite: %s" % path)
    contact_model.assert_inside_unit_ball(coords)
    return coords


def _mask_global_indices(data: Any, chromosome_index: int, positions: np.ndarray) -> np.ndarray:
    slc = data.chromosome_slice(chromosome_index)
    return slc.start + np.asarray(positions, dtype=np.int64) // int(data.bin_size)


def build_masks(data: Any, reference: np.ndarray) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """从冻结 old21 快照重建 mask，并对 positions / triu 索引做 bit parity。"""
    if sha256_file(MASK_SNAPSHOT) != MASK_SNAPSHOT_SHA256:
        raise RuntimeError("frozen legacy mask snapshot SHA256 mismatch")
    with np.load(MASK_SNAPSHOT, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    expected_counts = {}
    if MASK_LOCK.exists():
        lock = json.loads(MASK_LOCK.read_text(encoding="utf-8"))
        expected_counts = {str(row["chromosome"]): row for row in lock.get("expected_by_chromosome", [])}
    masks: dict[str, dict[str, Any]] = {}
    parity_rows = []
    for ci, name in enumerate(data.chromosome_names):
        length = int(data.chromosome_lengths[ci])
        positions = np.arange(MASK_OFFSET_BP, length, BIN_SIZE_BP, dtype=np.int64)
        pair_i, pair_j = np.triu_indices(len(positions), k=1)
        frozen_positions = arrays["chr%d_positions" % ci]
        frozen_pair_i = arrays["chr%d_pair_i" % ci]
        frozen_pair_j = arrays["chr%d_pair_j" % ci]
        common = arrays["chr%d_common" % ci].astype(bool)
        checks = {
            "positions": bool(np.array_equal(positions, frozen_positions)),
            "triu_pair_i": bool(np.array_equal(pair_i, frozen_pair_i)),
            "triu_pair_j": bool(np.array_equal(pair_j, frozen_pair_j)),
        }
        if not all(checks.values()):
            raise RuntimeError("frozen old21 mask parity failure for %s: %s" % (name, checks))
        # ref_mat/ref_pat 必须先把 mask 的 positions 映射到该染色体的全局 locus，再算距离。
        # 早先版本直接用整条染色体切片，索引与 positions 不对应，是一处**误导缓存**（已修正）。
        # 046 `_r2_result` 自己用 global_indices 重算，从不读这两个字段，因此主 R2 数值不受影响。
        global_indices = _mask_global_indices(data, ci, positions)
        ref_mat = ev._distance(reference[0, global_indices], pair_i, pair_j)
        ref_pat = ev._distance(reference[1, global_indices], pair_i, pair_j)
        valid = np.zeros(len(positions), dtype=bool)
        valid[pair_i[common]] = True
        valid[pair_j[common]] = True
        if not np.array_equal(common, valid[pair_i] & valid[pair_j]):
            raise RuntimeError("old21 common mask does not factorize for %s" % name)
        counts = {"n_bins": int(len(positions)), "n_total_non_diagonal_pairs": int(len(pair_i)),
                  "n_common_pairs": int(common.sum())}
        row = expected_counts.get(name)
        if row is not None:
            for key in counts:
                if int(row[key]) != counts[key]:
                    raise RuntimeError("old21 count mismatch %s.%s: %s vs %s" % (name, key, counts[key], row[key]))
        masks[name] = {"chromosome": name, "chromosome_index": ci, "positions": positions,
                       "pair_i": pair_i, "pair_j": pair_j, "common": common,
                       "valid_local_bins": np.flatnonzero(valid), "ref_mat": ref_mat, "ref_pat": ref_pat,
                       **counts}
        parity_rows.append({"chromosome": name, **counts, **checks})
    totals = {"n_total_non_diagonal_pairs": sum(row["n_total_non_diagonal_pairs"] for row in masks.values()),
              "n_common_pairs": sum(row["n_common_pairs"] for row in masks.values())}
    if totals != {"n_total_non_diagonal_pairs": 176201, "n_common_pairs": 157529}:
        raise RuntimeError("frozen old21 totals changed: %s" % totals)
    validation = {"schema": "p9016-round050-p1-old21-mask-validation-v1", "status": "PASS",
                  "snapshot": str(MASK_SNAPSHOT.relative_to(ROOT)), "snapshot_sha256": MASK_SNAPSHOT_SHA256,
                  "per_chromosome": parity_rows, "totals": totals,
                  "exact_fields": ["positions", "triu_pair_i", "triu_pair_j", "common"],
                  "note": "rebuilt from the frozen snapshot without re-running the frozen legacy worker; "
                          "positions and triangle indices are re-derived and bit-compared"}
    return masks, validation


def centre_reads(coords: np.ndarray, reference: np.ndarray, data: Any,
                 finite_loci: np.ndarray | None = None) -> dict[str, Any]:
    """20 个合并中心（190 距）与 40 个 copy 中心跨 chr（760 距）的空间读出。

    两个结构使用同一组参考有限 locus，避免候选侧多算 bead 造成的不对称分母。
    """
    if finite_loci is None:
        finite_loci = np.ones(int(data.n_loci), dtype=bool)
    merged, copy_centres, copy_chrom = [], [], []
    ref_merged, ref_copies = [], []
    for ci, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(ci)
        keep = np.flatnonzero(finite_loci[slc])
        if len(keep) == 0:
            raise RuntimeError("no common finite locus on %s" % name)
        global_keep = slc.start + keep
        block = coords[:, global_keep].reshape(-1, 3)
        merged.append(block.mean(axis=0))
        for copy in (0, 1):
            copy_centres.append(coords[copy, global_keep].mean(axis=0))
            copy_chrom.append(ci)
        ref_merged.append(reference[:, global_keep].reshape(-1, 3).mean(axis=0))
        for copy in (0, 1):
            ref_copies.append(reference[copy, global_keep].mean(axis=0))
    merged = np.asarray(merged)
    copy_centres = np.asarray(copy_centres)
    copy_chrom = np.asarray(copy_chrom)
    ref_merged = np.asarray(ref_merged)
    ref_copies = np.asarray(ref_copies)

    i, j = np.triu_indices(20, k=1)
    merged_d = np.linalg.norm(merged[i] - merged[j], axis=1)
    ref_merged_d = np.linalg.norm(ref_merged[i] - ref_merged[j], axis=1)
    cross = np.array([[a, b] for a in range(40) for b in range(a + 1, 40)
                      if copy_chrom[a] != copy_chrom[b]], dtype=np.int64)
    copy_d = np.linalg.norm(copy_centres[cross[:, 0]] - copy_centres[cross[:, 1]], axis=1)
    ref_copy_d = np.linalg.norm(ref_copies[cross[:, 0]] - ref_copies[cross[:, 1]], axis=1)

    def correlate(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
        pearson = ev._metric("pearson", left, right)
        spearman = ev._metric("spearman", left, right)
        standardised = (left - left.mean()) / (left.std() or 1.0)
        ref_standardised = (right - right.mean()) / (right.std() or 1.0)
        stress = float(np.sqrt(np.mean((standardised - ref_standardised) ** 2)))
        return {"pearson": pearson, "spearman": spearman, "normalised_stress": stress,
                "n": int(len(left))}
    return {
        "n_common_finite_loci": int(finite_loci.sum()),
        "merged_centres_20": {"n_distances": int(len(merged_d)), **correlate(merged_d, ref_merged_d)},
        "copy_centres_cross_chromosome_40": {"n_distances": int(len(copy_d)),
                                             **correlate(copy_d, ref_copy_d)},
        "definition": "20 merged chromosome centres (both copies averaged) give 190 distances; the 40 copy "
                      "centres give 760 distances between centres of different chromosomes. Both use the same "
                      "reference-finite locus set. Pooled inter distances are never substituted for these "
                      "placement readouts.",
    }


def main() -> int:
    argparse.ArgumentParser().parse_args()
    out = RUN_DIR / "evaluation"
    results = out / "results"
    results.mkdir(parents=True, exist_ok=True)
    data = ev._load_data()

    # ---------------- coordinates, hashes and nulls before the reference ----------
    sources: dict[str, dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        npz_path = candidate_npz_path(endpoint)
        dg_path = candidate_3dg_path(endpoint)
        if not npz_path.is_file() or not dg_path.is_file():
            raise RuntimeError("P1 endpoint is missing: %s" % npz_path)
        payload = ev._load_candidate_npz(npz_path, (2, int(data.n_loci), 3))
        sources[endpoint] = {"kind": "fit", "npz": npz_path, "npz_sha256": sha256_file(npz_path),
                             "3dg": dg_path, "3dg_sha256": sha256_file(dg_path),
                             "coordinates": payload["coordinates"], "p": payload["p"], "q": payload["q"],
                             "raw_y": payload["raw_y"]}
    for name, relative in EXTRA_SOURCES.items():
        path = ROOT / relative
        npz = path.with_suffix(".npz")
        p_value = None
        if npz.is_file():
            with np.load(npz, allow_pickle=False) as payload:
                p_value = float(np.asarray(payload["p"]).item())
        sources[name] = {"kind": "reference_source", "3dg": path, "3dg_sha256": sha256_file(path),
                         "npz": npz if npz.is_file() else None,
                         "npz_sha256": sha256_file(npz) if npz.is_file() else None,
                         "coordinates": load_full_grid_3dg(path, data), "p": p_value, "q": None}

    null_dir = out / "nulls"
    null_dir.mkdir(parents=True, exist_ok=True)
    null_records = []
    for name, source in sources.items():
        variants = [("u_zero", None)] + [("random_u", seed) for seed in RANDOM_U_SEEDS]
        for kind, seed in variants:
            coords, audit = (ev._make_u_zero(source["coordinates"]) if seed is None
                             else ev._make_random_u(source["coordinates"], data, seed))
            suffix = "u-zero" if seed is None else "random-u-%d" % seed
            path = null_dir / ("%s__%s.npz" % (name, suffix))
            np.savez_compressed(path, coordinates=coords)
            with np.load(path, allow_pickle=False) as payload:
                readback = np.asarray(payload["coordinates"], dtype=np.float64)
            if not np.array_equal(readback, coords):
                raise RuntimeError("null readback failure: %s" % path)
            null_records.append({"source": name, "null_kind": kind, "seed": seed,
                                 "path": str(path.relative_to(RUN_DIR)), "sha256": sha256_file(path),
                                 "coordinate_array_sha256": array_sha256(coords),
                                 "coordinates": coords, **audit})

    reference_sha = sha256_file(ev.REFERENCE_PATH)
    if reference_sha != ev.REFERENCE_SHA256:
        raise RuntimeError("reference 3DG SHA256 mismatch")
    gate = {
        "schema": "p9016-round050-p1-pre-reference-hash-gate-v1", "status": "PASS",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference_opened": False, "phase_opened": False,
        "sources": [{"id": name, "kind": source["kind"],
                     "npz": (str(source["npz"].relative_to(ROOT)) if source.get("npz") else None),
                     "npz_sha256": source.get("npz_sha256"),
                     "3dg": str(source["3dg"].relative_to(ROOT)), "3dg_sha256": source["3dg_sha256"],
                     "coordinate_array_sha256": array_sha256(source["coordinates"])}
                    for name, source in sources.items()],
        "nulls": [{key: value for key, value in row.items() if key != "coordinates"} for row in null_records],
        "null_count": len(null_records),
        "reference": {"path": str(ev.REFERENCE_PATH.relative_to(ROOT)), "sha256": reference_sha,
                      "hash_only_stage": "bytes hashed before any payload parse"},
        "snpfree": {"path": str(ev.RAW_PAIRS_PATH.relative_to(ROOT)),
                    "sha256": sha256_file(ev.RAW_PAIRS_PATH)},
        "bootstrap_indices": {"path": str(BOOTSTRAP_INDICES.relative_to(ROOT)),
                              "sha256": sha256_file(BOOTSTRAP_INDICES)},
        "mask_snapshot": {"path": str(MASK_SNAPSHOT.relative_to(ROOT)), "sha256": MASK_SNAPSHOT_SHA256},
        "note": "every P1 endpoint, base, work baseline and null coordinate payload was written and hashed "
                "before the reference was parsed; the reference row records bytes-only hashing.",
    }
    write_json(out / "pre_reference_gate.json", gate)
    write_json(out / "post_reference_gate.json", {
        "schema": "p9016-round050-p1-post-reference-gate-v1", "reference_opened": True,
        "opened_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})

    # ---------------- reference-side reads ---------------------------------------
    _tracks, reference, _actual = ev._load_reference(data)
    masks, mask_validation = build_masks(data, reference)
    write_json(results / "mask_validation.json", mask_validation)
    inter_cache = ev._build_inter_cache(data, masks, reference)
    finite_loci = np.isfinite(reference).all(axis=2).all(axis=0)
    if not finite_loci.any():
        raise RuntimeError("reference has no finite locus")
    indices = np.load(BOOTSTRAP_INDICES)
    if indices.shape != (10_000, 20):
        raise RuntimeError("frozen bootstrap index matrix shape changed: %s" % (indices.shape,))

    def r2_for(coords: np.ndarray) -> dict[str, Any]:
        return ev._r2_result(coords, reference, data, masks)

    def full_j_for(coords: np.ndarray, p: float) -> dict[str, Any]:
        audit = ac.three_loss_components(data, coords, p)
        return {"fullJ_A": audit["fullJ_A"], "fullJ_B": audit["fullJ_B"], "fullJ_C": audit["fullJ_C"],
                "count_A": audit["count_A"], "count_B": audit["count_B"], "count_C": audit["count_C"],
                "Zsum": audit["Zsum"], "Zmax": audit["Zmax"],
                "regularization_weighted": audit["regularization_total"],
                "regularizers_weighted": audit["regularizers_weighted"],
                "Nraw": audit["Nraw"], "n_off": audit["n_off"]}

    rows: list[dict[str, Any]] = []
    per_source: dict[str, Any] = {}
    for name, source in sources.items():
        coords = source["coordinates"]
        p_value = source["p"]
        if p_value is None:
            p_value = 0.75
        r2 = r2_for(coords)
        rg, centre, peak = ev._rg(coords)
        inter = ev._inter_result(coords, inter_cache)
        spatial = centre_reads(coords, reference, data, finite_loci)
        full_j = full_j_for(coords, p_value) if name in ENDPOINTS else None
        per_source[name] = {"r2": r2, "rg": rg, "peak_radius": peak, "inter": inter,
                            "spatial": spatial, "full_j": full_j}
        for metric in ("pearson", "spearman"):
            macro = r2["macro_equal_chromosome_weight_defined_only"][metric]
            undefined = r2["undefined_chromosomes_by_field"][metric]["matched"]
            rows.append({
                "source": name, "kind": source["kind"], "metric": metric,
                "matched_macro": macro["matched"], "cross_macro": macro["cross"],
                "contrast_macro": macro["contrast"], "min_margin_macro": macro["min_margin"],
                "copy_A_margin_macro": macro["copy_A_margin"], "copy_B_margin_macro": macro["copy_B_margin"],
                "defined_chromosomes": r2["defined_chromosome_counts"][metric]["matched"],
                "undefined_chromosomes": "|".join(undefined),
                "matched_macro_all20": (macro["matched"] if not undefined else None),
                "inter_pearson": inter["pearson"], "inter_spearman": inter["spearman"],
                "whole_cell_rg": rg,
                "merged_centre_pearson": spatial["merged_centres_20"]["pearson"],
                "merged_centre_spearman": spatial["merged_centres_20"]["spearman"],
                "copy_centre_cross_pearson": spatial["copy_centres_cross_chromosome_40"]["pearson"],
                "copy_centre_cross_spearman": spatial["copy_centres_cross_chromosome_40"]["spearman"],
                "fullJ_A": (full_j or {}).get("fullJ_A"), "count_A": (full_j or {}).get("count_A"),
            })
    write_tsv(results / "p1_source_summary.tsv", rows)

    # nulls report their own support
    null_rows = []
    for record in null_records:
        r2 = r2_for(record["coordinates"])
        inter = ev._inter_result(record["coordinates"], inter_cache)
        spatial = centre_reads(record["coordinates"], reference, data, finite_loci)
        rg, _centre, _peak = ev._rg(record["coordinates"])
        for metric in ("pearson", "spearman"):
            macro = r2["macro_equal_chromosome_weight_defined_only"][metric]
            undefined = r2["undefined_chromosomes_by_field"][metric]["matched"]
            null_rows.append({
                "source": record["source"], "null_kind": record["null_kind"], "seed": record["seed"],
                "metric": metric, "matched_macro_defined_only": macro["matched"],
                "contrast_macro_defined_only": macro["contrast"],
                "matched_macro_all20": (macro["matched"] if not undefined else None),
                "defined_chromosomes": r2["defined_chromosome_counts"][metric]["matched"],
                "undefined_chromosomes": "|".join(undefined),
                "inter_pearson": inter["pearson"], "whole_cell_rg": rg,
                "copy_centre_cross_pearson": spatial["copy_centres_cross_chromosome_40"]["pearson"],
                "global_scale": record.get("global_scale"),
            })
    write_tsv(results / "p1_null_summary.tsv", null_rows)
    null_aggregate: dict[str, Any] = {}
    for metric in ("pearson", "spearman"):
        for key in ("matched_macro_defined_only", "contrast_macro_defined_only"):
            for kind in ("u_zero", "random_u"):
                values = [row[key] for row in null_rows
                          if row["metric"] == metric and row["null_kind"] == kind and row[key] is not None]
                null_aggregate["%s:%s:%s" % (metric, key, kind)] = {
                    "n": len(values), "mean": float(np.mean(values)) if values else None,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    write_json(results / "p1_null_aggregate.json", {
        "schema": "p9016-round050-p1-null-aggregate-v1", "aggregate": null_aggregate,
        "support_rule": "each null is scored on its own resolvable support and is never intersected into the "
                        "main candidate common support; u0 R3 tie/undefined direction is an expected null "
                        "property, not a mask restriction"})

    # paired bootstrap on the fixed 20-chromosome denominator
    paired = []
    for left_id, right_id, label in PAIRS:
        if left_id not in per_source or right_id not in per_source:
            continue
        entry = ev._paired([{"candidate_id": name, "r2": value["r2"]} for name, value in per_source.items()],
                           left_id, right_id, label, indices)
        entry["direction"] = "delta = left minus right"
        paired.append(entry)
    write_json(results / "p1_paired_bootstrap.json", {
        "schema": "p9016-round050-p1-paired-bootstrap-v1", "seed": 450301, "draws": 10_000,
        "unit": "paired chromosome resampling within one cell; technical/structural variation, not biological "
                "replication", "rows": paired})

    paired_rows = []
    for entry in paired:
        for key, value in entry["metrics"].items():
            paired_rows.append({"comparison": entry["label"], "left": entry["left"], "right": entry["right"],
                                "metric": key, "mean": value["mean"], "ci95_low": value["ci95"][0],
                                "ci95_high": value["ci95"][1], "left_wins": value["left_wins"],
                                "right_wins": value["right_wins"], "ties": value["ties"],
                                "defined_chromosomes": value["defined_chromosomes"],
                                "full_20_estimate_defined": value["full_20_estimate_defined"],
                                "defined_only_mean": value["defined_only_descriptive"]["mean"]})
    write_tsv(results / "p1_paired_bootstrap.tsv", paired_rows)

    summary = {
        "schema": "p9016-round050-p1-evaluation-summary-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": {name: {"kind": source["kind"], "3dg": str(source["3dg"].relative_to(ROOT)),
                           "3dg_sha256": source["3dg_sha256"], "npz_sha256": source.get("npz_sha256"),
                           "rg": per_source[name]["rg"], "peak_radius": per_source[name]["peak_radius"],
                           "inter_pearson": per_source[name]["inter"]["pearson"],
                           "spatial": per_source[name]["spatial"],
                           "full_j": per_source[name]["full_j"]}
                   for name, source in sources.items()},
        "mask_totals": mask_validation["totals"],
        "inter_cache": {key: value for key, value in inter_cache.items()
                        if key in ("valid_bin_count", "within_valid_pair_count", "eligible_locus_pairs",
                                   "denominator", "global_locus_index_assertion")},
        "null_count": len(null_records),
        "null_aggregate": null_aggregate,
        "comparison_pairs": [list(pair) for pair in PAIRS],
        "accuracy_rule": "R2 macro values use equal chromosome weight; a per-metric macro is None when any of "
                         "the 20 chromosomes is undefined, and the defined-only value plus the explicit "
                         "undefined list is always reported next to it",
        "scope": "local optimization diagnostic; the four P1 endpoints are forked from frozen 046 bases and are "
                 "not new independent blind reconstructions",
        "reference_opened": True, "phase_opened": False,
    }
    write_json(results / "p1_evaluation_summary.json", summary)
    print(json.dumps({"sources": len(sources), "nulls": len(null_records), "pairs": len(paired_rows)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
