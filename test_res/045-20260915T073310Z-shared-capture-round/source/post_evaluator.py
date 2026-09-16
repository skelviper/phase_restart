"""Post-fit evaluator with hash-gated frozen old21 R2 masks."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent
for path in (ROOT, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_io import load_aggregate, sha256_file, write_json  # noqa: E402
from pr import contact_model  # noqa: E402
from pr.score import spearman  # noqa: E402

REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
BIN_OFFSET_BP = 3_000_000
BOOTSTRAP_SEED = 450301
BOOTSTRAP_DRAWS = 10_000
RANDOM_U_SEEDS = tuple(range(450500, 450516))
GEOMETRY_TIE_TOL = 1e-12
MIN_COMMON_PAIRS = 20
MASK_LOCK_PATH = ROOT / "docs/audits/multires-r2-preparation-20260914_041826/mask_lock.json"
MASK_MANIFEST_PATH = ROOT / "test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/evaluation_manifest.json"
MASK_LOCK_SHA256 = "d0c325dea1289374152a327557605f8bbd52267b025cd7b57689be3e3666c49e"
MASK_MANIFEST_SHA256 = "fa1b26c834c173604c21f954d494cece8e053dd970e7cb6111f561b349574acb"
OLD036_ANCHOR_PATH = ROOT / "test_res/036-20260914T064651Z-gpu-multires/coords/C0/random_joint/final-1m.3dg"
OLD036_ANCHOR_SHA256 = "18c06860f78525e55045e911db05981dd431a86c3bf9ee3ebb5046845b0ccc55"
OLD038_ANCHOR_TSV = ROOT / "test_res/038-20260914T143812Z-gpu-m1-formal/evaluation-r2-20260914T153134Z/r2_real_historical_036_C0_anchor_x20chr.tsv"
OLD038_ANCHOR_TSV_SHA256 = "d44bfe73155edea0f6bb107ed0e41488428a03b230648e381c80066ed4651e21"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value


def _write(path: Path, value: Any) -> None:
    write_json(path, _jsonable(value))


def _resolve(path_text: str | Path) -> Path:
    path = (RUN / Path(path_text)).resolve()
    try:
        path.relative_to(RUN.resolve())
    except ValueError as exc:
        raise RuntimeError("artifact escapes run directory: %s" % path_text) from exc
    return path


def _hash_array(values: np.ndarray, dtype: str = "<f8") -> str:
    array = np.asarray(values, dtype=dtype, order="C")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    return float(np.dot(xc, yc) / denom) if denom else float("nan")


def _metric(metric: str, x: np.ndarray, y: np.ndarray) -> float:
    value = _pearson(x, y) if metric == "pearson" else float(spearman(x, y))
    return value if math.isfinite(value) else float("nan")


def _distance(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    delta = points[pair_i] - points[pair_j]
    return np.sqrt(np.sum(delta * delta, axis=1))


def _four_correlations(candidate: np.ndarray, reference: np.ndarray,
                       pair_i: np.ndarray, pair_j: np.ndarray,
                       override: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    ca, cb = _distance(candidate[0], pair_i, pair_j), _distance(candidate[1], pair_i, pair_j)
    if override is None:
        rm, rp = _distance(reference[0], pair_i, pair_j), _distance(reference[1], pair_i, pair_j)
    else:
        rm, rp = np.asarray(override[0]), np.asarray(override[1])
    output: dict[str, Any] = {"n_pairs": int(len(pair_i)), "metrics": {}}
    for metric in ("pearson", "spearman"):
        rho = {"A_mat": _metric(metric, ca, rm), "A_pat": _metric(metric, ca, rp),
               "B_mat": _metric(metric, cb, rm), "B_pat": _metric(metric, cb, rp)}
        rho_finite = all(math.isfinite(float(value)) for value in rho.values())
        enough_pairs = len(pair_i) >= MIN_COMMON_PAIRS
        undefined_reason = None
        if not enough_pairs:
            undefined_reason = "common_pairs_below_MIN_COMMON_PAIRS"
        elif not rho_finite:
            undefined_reason = "one_or_more_of_four_rho_nonfinite"
        direct = float(np.mean([rho["A_mat"], rho["B_pat"]])) if rho_finite else float("nan")
        swapped = float(np.mean([rho["A_pat"], rho["B_mat"]])) if rho_finite else float("nan")
        if undefined_reason is not None:
            orientation = "undefined"
            matched = cross = contrast = None
            margin_a = margin_b = margin_mat = margin_pat = min_margin = None
        elif abs(direct - swapped) <= GEOMETRY_TIE_TOL:
            orientation = "unresolved_tie"
            matched = cross = float((direct + swapped) / 2.0)
            contrast = 0.0
            margin_a = margin_b = margin_mat = margin_pat = min_margin = None
        elif direct > swapped:
            orientation = "direct"
            matched, cross, contrast = direct, swapped, direct - swapped
            margin_a = rho["A_mat"] - rho["A_pat"]
            margin_b = rho["B_pat"] - rho["B_mat"]
            margin_mat, margin_pat = margin_a, margin_b
            min_margin = min(margin_a, margin_b)
        else:
            orientation = "swapped"
            matched, cross, contrast = swapped, direct, swapped - direct
            margin_a = rho["A_pat"] - rho["A_mat"]
            margin_b = rho["B_mat"] - rho["B_pat"]
            margin_mat, margin_pat = margin_b, margin_a
            min_margin = min(margin_a, margin_b)
        output["metrics"][metric] = {
            "rho": rho, "direct": direct if math.isfinite(direct) else None,
            "swapped": swapped if math.isfinite(swapped) else None,
            "matched": matched, "cross": cross, "contrast": contrast,
            "orientation": orientation, "geometry_tie": orientation == "unresolved_tie",
            "margin_copy_A": margin_a, "margin_copy_B": margin_b,
            "margin_mat": margin_mat, "margin_pat": margin_pat, "min_margin": min_margin,
            "n_pairs": int(len(pair_i)),
            "derived_defined": undefined_reason is None,
            "undefined_reason": undefined_reason,
        }
    output["scale_rms_candidate"] = float(np.sqrt(np.mean(np.concatenate([ca, cb]) ** 2)))
    output["scale_rms_reference"] = float(np.sqrt(np.mean(np.concatenate([rm, rp]) ** 2)))
    return output


def _chromosome_metrics(candidate: np.ndarray, reference: np.ndarray, data: Any, *,
                        position_offset_bp: int = BIN_OFFSET_BP,
                        same_shape_truth_mean: bool = False,
                        frozen_masks: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    rows = []
    for chromosome, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(chromosome)
        frozen = frozen_masks.get(str(name)) if frozen_masks is not None else None
        if frozen is not None:
            local = np.asarray(frozen["local_bins"], dtype=np.int64)
            pair_i = np.asarray(frozen["pair_i"], dtype=np.int64)
            pair_j = np.asarray(frozen["pair_j"], dtype=np.int64)
            keep = np.asarray(frozen["common"], dtype=bool)
        else:
            start = int(position_offset_bp // data.bin_size)
            if start >= slc.stop - slc.start - 1:
                rows.append({"chromosome": str(name), "status": "insufficient_bins"})
                continue
            local = np.arange(start, slc.stop - slc.start, dtype=np.int64)
            i, j = np.triu_indices(len(local), k=1)
            pair_i, pair_j = local[i], local[j]
            keep = None
        cand = candidate[:, slc.start:slc.stop][:, local]
        ref = reference[:, slc.start:slc.stop][:, local]
        if keep is not None:
            pair_i, pair_j = pair_i[keep], pair_j[keep]
        else:
            finite = np.isfinite(cand).all(axis=2).all(axis=0) & np.isfinite(ref).all(axis=2).all(axis=0)
            finite = finite[pair_i] & finite[pair_j]
            pair_i, pair_j = pair_i[finite], pair_j[finite]
        override = None
        if same_shape_truth_mean:
            common = (_distance(ref[0], pair_i, pair_j) + _distance(ref[1], pair_i, pair_j)) / 2.0
            override = (common, common)
        row = _four_correlations(cand, ref, pair_i, pair_j, override)
        row.update({"chromosome": str(name), "chromosome_index": chromosome,
                    "position_offset_bp": int(position_offset_bp), "status": "ok"})
        rows.append(row)
    metric_keys = ("matched", "cross", "contrast", "min_margin", "margin_copy_A", "margin_copy_B", "margin_mat", "margin_pat")
    macro, defined = {}, {}
    for metric in ("pearson", "spearman"):
        macro[metric], defined[metric] = {}, {}
        for key in metric_keys:
            values = [row["metrics"][metric][key] for row in rows if row.get("status") == "ok"
                      and row["metrics"][metric][key] is not None
                      and math.isfinite(float(row["metrics"][metric][key]))]
            macro[metric][key] = float(np.mean(values)) if values else None
            defined[metric][key] = len(values)
    return {"per_chromosome": rows, "macro_equal_chromosome_weight": macro,
            "defined_chromosome_counts": defined,
            "metric_definition": "numeric strict upper triangle of frozen common finite non-diagonal pairs",
            "homolog_primary": "matched and matched-cross with both copy margins; no center distance"}


def _inter_metrics(candidate: np.ndarray, reference: np.ndarray, data: Any) -> dict[str, Any]:
    inter = ~np.asarray(data.cis_pair, dtype=bool)
    pair_i, pair_j = np.asarray(data.pair_i[inter]), np.asarray(data.pair_j[inter])
    finite = np.isfinite(candidate).all(axis=2).all(axis=0) & np.isfinite(reference).all(axis=2).all(axis=0)
    keep = finite[pair_i] & finite[pair_j]
    pair_i, pair_j = pair_i[keep], pair_j[keep]
    candidate_distances = np.stack((
        _distance(candidate[0], pair_i, pair_j),
        np.sqrt(np.sum((candidate[0, pair_i] - candidate[1, pair_j]) ** 2, axis=1)),
        np.sqrt(np.sum((candidate[1, pair_i] - candidate[0, pair_j]) ** 2, axis=1)),
        _distance(candidate[1], pair_i, pair_j)), axis=1)
    reference_distances = np.stack((
        _distance(reference[0], pair_i, pair_j),
        np.sqrt(np.sum((reference[0, pair_i] - reference[1, pair_j]) ** 2, axis=1)),
        np.sqrt(np.sum((reference[1, pair_i] - reference[0, pair_j]) ** 2, axis=1)),
        _distance(reference[1], pair_i, pair_j)), axis=1)
    cf = np.sort(candidate_distances, axis=1)
    rf = np.sort(reference_distances, axis=1)
    x, y = cf.ravel(), rf.ravel()
    return {"pearson": _pearson(x, y), "spearman": float(spearman(x, y)),
            "eligible_locus_pairs": int(len(pair_i)), "denominator": int(4 * len(pair_i)),
            "order_statistic_vector": "sorted_four_copy_distances_per_inter_locus_pair",
            "high_absolute_r_not_packing_proof": True}


def _distance_matrix_errors(candidate: np.ndarray, truth: np.ndarray, data: Any,
                            same_shape_truth_mean: bool) -> dict[str, Any]:
    copy_rms, truth_rms, raw_copy, norm_copy = [[], []], [], [], []
    common_err, own_err = [[], []], [[], []]
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        i, j = np.triu_indices(slc.stop - slc.start, k=1)
        ca, cb = _distance(candidate[0, slc], i, j), _distance(candidate[1, slc], i, j)
        ta, tb = _distance(truth[0, slc], i, j), _distance(truth[1, slc], i, j)
        tc = (ta + tb) / 2.0 if same_shape_truth_mean else None
        copy_rms[0].append(float(np.sqrt(np.mean(ca * ca))))
        copy_rms[1].append(float(np.sqrt(np.mean(cb * cb))))
        ca_s = max(float(np.sqrt(np.mean(ca * ca))), np.finfo(float).tiny)
        cb_s = max(float(np.sqrt(np.mean(cb * cb))), np.finfo(float).tiny)
        raw_copy.append(float(np.sqrt(np.mean((ca - cb) ** 2))))
        norm_copy.append(float(np.sqrt(np.mean((ca / ca_s - cb / cb_s) ** 2))))
        if same_shape_truth_mean:
            tc_s = max(float(np.sqrt(np.mean(tc * tc))), np.finfo(float).tiny)
            truth_rms.append(float(tc_s))
            common_err[0].append(float(np.sqrt(np.mean((ca - tc) ** 2)) / tc_s))
            common_err[1].append(float(np.sqrt(np.mean((cb - tc) ** 2)) / tc_s))
            own_err[0].append(float(np.sqrt(np.mean((ca / ca_s - tc / tc_s) ** 2))))
            own_err[1].append(float(np.sqrt(np.mean((cb / cb_s - tc / tc_s) ** 2))))
        else:
            ta_s = max(float(np.sqrt(np.mean(ta * ta))), np.finfo(float).tiny)
            tb_s = max(float(np.sqrt(np.mean(tb * tb))), np.finfo(float).tiny)
            truth_rms.extend([ta_s, tb_s])
            common_err[0].append(float(np.sqrt(np.mean((ca - ta) ** 2)) / ta_s))
            common_err[1].append(float(np.sqrt(np.mean((cb - tb) ** 2)) / tb_s))
            own_err[0].append(float(np.sqrt(np.mean((ca / ca_s - ta / ta_s) ** 2))))
            own_err[1].append(float(np.sqrt(np.mean((cb / cb_s - tb / tb_s) ** 2))))
    return {"truth_matrix_policy": "mean A/B distance matrix for N2" if same_shape_truth_mean else "copy-specific truth matrices",
            "candidate_copy_distance_rms_mean": [float(np.mean(x)) for x in copy_rms],
            "truth_distance_rms_mean": float(np.mean(truth_rms)),
            "candidate_copy_common_scale_rms_difference_mean": float(np.mean(raw_copy)),
            "candidate_copy_own_rms_normalized_matrix_difference_mean": float(np.mean(norm_copy)),
            "copy_A_common_scale_error_to_truth_mean": float(np.mean(common_err[0])),
            "copy_B_common_scale_error_to_truth_mean": float(np.mean(common_err[1])),
            "copy_A_own_rms_normalized_shape_error_to_truth_mean": float(np.mean(own_err[0])),
            "copy_B_own_rms_normalized_shape_error_to_truth_mean": float(np.mean(own_err[1])),
            "common_scale_difference_is_not_shape_only": True, "uses_no_one_minus_rho_recovery_score": True}


def _load_simple_tracks(path: Path) -> dict[str, dict[int, np.ndarray]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    tracks: dict[str, dict[int, np.ndarray]] = {}
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 5:
                tracks.setdefault(fields[0], {})[int(fields[1])] = np.asarray(fields[2:5], dtype=np.float64)
    return tracks


def _load_frozen_mask_inputs_after_candidate_gate() -> tuple[dict[str, Any], dict[str, dict[str, dict[int, np.ndarray]]], list[dict[str, Any]]]:
    if sha256_file(MASK_LOCK_PATH) != MASK_LOCK_SHA256 or sha256_file(MASK_MANIFEST_PATH) != MASK_MANIFEST_SHA256:
        raise RuntimeError("old21 mask lock/manifest hash mismatch")
    lock = json.loads(MASK_LOCK_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MASK_MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = [dict(row) for row in manifest.get("coordinates", []) if row.get("mask_included") is True]
    if lock.get("condition_count") != 21 or len(rows) != 21 or lock.get("new_endpoint_inclusion") is not False:
        raise RuntimeError("old21 mask condition contract changed")
    structures, hashes = {}, []
    for row in rows:
        path = Path(str(row["path"])).resolve()
        actual = sha256_file(path)
        if actual != str(row["sha256"]):
            raise RuntimeError("old21 mask coordinate hash mismatch: %s" % path)
        structures[str(row["condition_id"])] = _load_simple_tracks(path)
        hashes.append({"condition_id": str(row["condition_id"]), "path": str(path), "sha256": actual})
    return lock, structures, hashes


def _build_frozen_masks(data: Any, lock: Mapping[str, Any], historical: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
                        reference: Mapping[str, Mapping[int, np.ndarray]]) -> dict[str, dict[str, Any]]:
    expected = {str(row["chromosome"]): row for row in lock["expected_by_chromosome"]}
    masks = {}
    for chromosome, name in enumerate(data.chromosome_names):
        length = int(data.chromosome_lengths[chromosome])
        positions = np.arange(BIN_OFFSET_BP, length, int(data.bin_size), dtype=np.int64)
        local = positions // int(data.bin_size)
        pair_i, pair_j = np.triu_indices(len(positions), k=1)
        common = np.ones(len(pair_i), dtype=bool)
        for structures in historical.values():
            for track in ("c%02da" % (chromosome + 1), "c%02db" % (chromosome + 1)):
                rows = structures.get(track)
                if rows is None:
                    continue
                points = np.asarray([rows.get(int(position), [np.nan, np.nan, np.nan]) for position in positions], dtype=np.float64)
                common &= np.isfinite(_distance(points, pair_i, pair_j))
        for suffix in ("mat", "pat"):
            rows = reference.get("%s(%s)" % (name, suffix), {})
            points = np.asarray([rows.get(int(position), [np.nan, np.nan, np.nan]) for position in positions], dtype=np.float64)
            common &= np.isfinite(_distance(points, pair_i, pair_j))
        counts = {"n_bins": int(len(positions)), "n_total_non_diagonal_pairs": int(len(pair_i)), "n_common_pairs": int(common.sum())}
        expected_row = expected.get(str(name))
        if expected_row is None or any(counts[key] != int(expected_row[key]) for key in counts):
            raise RuntimeError("frozen old21 mask mismatch %s: %r vs %r" % (name, counts, expected_row))
        masks[str(name)] = {"positions": positions, "local_bins": local, "pair_i": pair_i, "pair_j": pair_j, "common": common, **counts}
    if sum(row["n_common_pairs"] for row in masks.values()) != 157529 or sum(row["n_total_non_diagonal_pairs"] for row in masks.values()) != 176201:
        raise RuntimeError("frozen old21 mask totals changed")
    return masks


def _load_reference(path: Path) -> dict[str, dict[int, np.ndarray]]:
    if sha256_file(path) != REFERENCE_SHA256:
        raise RuntimeError("reference SHA256 mismatch")
    return _load_simple_tracks(path)


def _reference_arrays(reference: Mapping[str, Mapping[int, np.ndarray]], data: Any) -> np.ndarray:
    result = np.full((2, data.n_loci, 3), np.nan, dtype=np.float64)
    for chromosome, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(chromosome)
        for copy, suffix in enumerate(("mat", "pat")):
            rows = reference.get("%s(%s)" % (name, suffix), {})
            for local, position in enumerate(data.locus_bin[slc] * int(data.bin_size)):
                if int(position) in rows:
                    result[copy, slc.start + local] = rows[int(position)]
    return result


def _validate_candidates(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema") != "p9016-shared-capture-candidate-manifest-v1" or len(manifest.get("candidates", [])) != 14:
        raise RuntimeError("candidate manifest must contain exactly 14 endpoints")
    result, seen = [], set()
    for row in manifest["candidates"]:
        ident = str(row["candidate_id"])
        if ident in seen:
            raise RuntimeError("duplicate candidate %s" % ident)
        seen.add(ident)
        path = _resolve(row["coordinate_path"])
        if sha256_file(path) != row["coordinate_sha256"]:
            raise RuntimeError("candidate file hash mismatch %s" % path)
        with np.load(path, allow_pickle=False) as payload:
            coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        if coordinates.shape != (2, 2645, 3) or not np.all(np.isfinite(coordinates)):
            raise RuntimeError("candidate shape/domain failure %s" % path)
        contact_model.assert_inside_unit_ball(coordinates)
        if _hash_array(coordinates) != row["coordinate_array_sha256"]:
            raise RuntimeError("candidate array hash mismatch %s" % path)
        result.append({**dict(row), "coordinates": coordinates, "path": path, "file_sha256": row["coordinate_sha256"]})
    return result


def _make_u_zero(coordinates: np.ndarray) -> np.ndarray:
    z = (coordinates[0] + coordinates[1]) / 2.0
    result = np.stack((z, z), axis=0)
    contact_model.assert_inside_unit_ball(result)
    return result


def _make_random_u(coordinates: np.ndarray, data: Any, seed: int) -> np.ndarray:
    z, u = (coordinates[0] + coordinates[1]) / 2.0, (coordinates[0] - coordinates[1]) / 2.0
    permuted = np.zeros_like(u)
    rng = np.random.default_rng(seed)
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        permuted[slc] = u[slc][rng.permutation(slc.stop - slc.start)]
    result = np.stack((z + permuted, z - permuted), axis=0)
    radius = float(np.linalg.norm(result, axis=2).max())
    if radius >= 1.0:
        result *= (1.0 - 1e-6) / radius
    contact_model.assert_inside_unit_ball(result)
    return result


def _write_nulls(rows: list[dict[str, Any]], data: Any) -> list[dict[str, Any]]:
    directory = RUN / "coords" / "nulls"
    directory.mkdir(parents=True, exist_ok=True)
    output = []
    for row in rows:
        if row.get("kind") not in ("real", "real_initial"):
            continue
        for kind, seed in [("u_zero", None)] + [("random_u", seed) for seed in RANDOM_U_SEEDS]:
            coordinates = _make_u_zero(row["coordinates"]) if seed is None else _make_random_u(row["coordinates"], data, seed)
            path = directory / (str(row["candidate_id"]) + "-" + kind + ("" if seed is None else "-%d" % seed) + ".npz")
            np.savez_compressed(path, coordinates=coordinates)
            output.append({"source_candidate_id": row["candidate_id"], "null_kind": kind, "seed": seed,
                           "path": str(path.relative_to(RUN)), "sha256": sha256_file(path),
                           "coordinate_array_sha256": _hash_array(coordinates)})
    return output


def _tracks_to_array(tracks: Mapping[str, Mapping[int, np.ndarray]], data: Any) -> np.ndarray:
    result = np.full((2, data.n_loci, 3), np.nan, dtype=np.float64)
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        for copy, suffix in enumerate(("a", "b")):
            rows = tracks.get("c%02d%s" % (chromosome + 1, suffix), {})
            for local, position in enumerate(data.locus_bin[slc] * int(data.bin_size)):
                if int(position) in rows:
                    result[copy, slc.start + local] = rows[int(position)]
    if not np.isfinite(result).all():
        raise RuntimeError("old036 anchor lacks complete 1Mb copy tracks")
    return result


def _historical_spearman_regression(data: Any, reference: np.ndarray, frozen_masks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if sha256_file(OLD036_ANCHOR_PATH) != OLD036_ANCHOR_SHA256 or sha256_file(OLD038_ANCHOR_TSV) != OLD038_ANCHOR_TSV_SHA256:
        raise RuntimeError("old036/038 regression input hash mismatch")
    anchor = _tracks_to_array(_load_simple_tracks(OLD036_ANCHOR_PATH), data)
    new = _chromosome_metrics(anchor, reference, data, frozen_masks=frozen_masks)
    new_by_chr = {row["chromosome"]: row for row in new["per_chromosome"]}
    with OLD038_ANCHOR_TSV.open("r", encoding="utf-8", newline="") as handle:
        old_rows = list(csv.DictReader(handle))
    fields = ("n_bins", "n_total_non_diagonal_pairs", "n_common_pairs_frozen", "rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat", "direct", "swapped", "matched", "cross", "contrast", "margin_mat", "margin_pat", "minmargin")
    mismatches, per_chr = [], []
    for old in old_rows:
        name, row = str(old["chromosome"]), new_by_chr.get(str(old["chromosome"]))
        if row is None:
            mismatches.append({"chromosome": name, "field": "row"})
            continue
        metric = row["metrics"]["spearman"]
        mask = frozen_masks[name]
        values = {"n_bins": len(mask["positions"]), "n_total_non_diagonal_pairs": mask["n_total_non_diagonal_pairs"], "n_common_pairs_frozen": mask["n_common_pairs"],
                  "rho_A_mat": metric["rho"]["A_mat"], "rho_A_pat": metric["rho"]["A_pat"], "rho_B_mat": metric["rho"]["B_mat"], "rho_B_pat": metric["rho"]["B_pat"],
                  "direct": metric["direct"], "swapped": metric["swapped"], "matched": metric["matched"], "cross": metric["cross"], "contrast": metric["contrast"], "margin_mat": metric["margin_mat"], "margin_pat": metric["margin_pat"], "minmargin": metric["min_margin"]}
        max_diff = 0.0
        for field in fields:
            old_value, new_value = float(old[field]), values[field]
            if new_value is None or abs(float(new_value) - old_value) > 1e-12:
                mismatches.append({"chromosome": name, "field": field, "old": old_value, "new": new_value})
            elif math.isfinite(float(new_value)):
                max_diff = max(max_diff, abs(float(new_value) - old_value))
        if old.get("orientation") != metric["orientation"] or (old.get("geometry_tie") == "True") != bool(metric["geometry_tie"]):
            mismatches.append({"chromosome": name, "field": "orientation_or_geometry_tie"})
        per_chr.append({"chromosome": name, "numeric_max_abs_diff": max_diff,
                        "same": not any(item.get("chromosome") == name for item in mismatches)})
    output = {"schema": "p9016-shared-capture-old036-spearman-regression-v1", "status": "PASS" if not mismatches else "FAIL",
              "metric": "Spearman only; old protocol CI not applied", "numeric_atol": 1e-12, "numeric_rtol": 0.0,
              "old_anchor_path": str(OLD036_ANCHOR_PATH.relative_to(ROOT)), "old_anchor_sha256": OLD036_ANCHOR_SHA256,
              "old038_tsv_path": str(OLD038_ANCHOR_TSV.relative_to(ROOT)), "old038_tsv_sha256": OLD038_ANCHOR_TSV_SHA256,
              "same_frozen_positions_pair_i_pair_j_common": True, "mask_counts": {"total": 176201, "common": 157529, "chromosomes": 20},
              "numeric_fields": list(fields), "mismatch_count": len(mismatches), "mismatches": mismatches[:100], "per_chromosome": per_chr,
              "not_used_for_selection": True}
    _write(RUN / "checks" / "old036_spearman_regression.json", output)
    if mismatches:
        raise RuntimeError("old036 Spearman regression failed: %d mismatches" % len(mismatches))
    return output


def _bootstrap(left: list[float], right: list[float]) -> dict[str, Any]:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if not len(a):
        return {"n_chromosomes": 0, "mean": None, "ci95": [None, None], "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray([np.mean((a - b)[rng.integers(0, len(a), len(a))]) for _ in range(BOOTSTRAP_DRAWS)])
    return {"n_chromosomes": len(a), "mean": float(np.mean(a - b)), "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "unit": "paired chromosome technical/structural bootstrap; not biological replicates"}


def _trajectory_rows() -> list[dict[str, Any]]:
    output = []
    for path in sorted((RUN / "stages").glob("*/[125]Mb.accepted_history.json")):
        for entry in json.loads(path.read_text(encoding="utf-8")):
            c = entry.get("components", {})
            output.append({"fit_id": path.parent.name, "stage": path.name.split(".")[0], "iteration": entry.get("iteration"), "nfev": entry.get("nfev"), "fun": entry.get("fun"), "p": entry.get("p"),
                           "group_mass_kl_normalized": c.get("group_mass_kl_normalized"), "sum_rate_cis_offdiag": c.get("sum_rate_cis_offdiag"), "sum_rate_inter": c.get("sum_rate_inter"), "sum_rate_offdiag": c.get("sum_rate_offdiag"),
                           "profiled_intensity_cis": c.get("profiled_intensity_cis"), "profiled_intensity_inter": c.get("profiled_intensity_inter"), "shared_profiled_intensity": c.get("shared_profiled_intensity"), "count_nll_normalized": c.get("count_nll_normalized"),
                           "weighted_count": c.get("weighted_count"), "weighted_bond": c.get("weighted_bond"), "weighted_repulsion": c.get("weighted_repulsion"), "weighted_bend": c.get("weighted_bend"), "weighted_p_prior": c.get("weighted_p_prior"), "canonical_gradient_max_abs": entry.get("canonical_gradient_max_abs")})
    return output


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate() -> dict[str, Any]:
    candidate_manifest = json.loads((RUN / "results/candidate_manifest.json").read_text(encoding="utf-8"))
    candidates = _validate_candidates(candidate_manifest)
    data = load_aggregate(RUN / "inputs/real_1000000_aggregate.npz")
    formal = json.loads((RUN / "inputs/formal_manifest.json").read_text(encoding="utf-8"))
    initial_controls = []
    for source in ("consensus", "random"):
        record = formal["real_initial_paths"][source]["stages"]["1Mb"]
        path = _resolve(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError("initial-control hash mismatch")
        with np.load(path, allow_pickle=False) as payload:
            coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        contact_model.assert_inside_unit_ball(coordinates)
        initial_controls.append({"candidate": source, "path": str(path.relative_to(RUN)), "sha256": sha256_file(path), "coordinate_array_sha256": _hash_array(coordinates)})
        candidates.append({"candidate_id": "initial-%s" % source, "kind": "real_initial", "candidate": source, "model_id": "initial", "coordinates": coordinates, "file_sha256": sha256_file(path)})
    null_records = _write_nulls(candidates, data)
    for record in null_records:
        path = _resolve(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError("null hash changed before evaluation")
        with np.load(path, allow_pickle=False) as payload:
            contact_model.assert_inside_unit_ball(np.asarray(payload["coordinates"], dtype=np.float64))
    mask_lock, historical, mask_hashes = _load_frozen_mask_inputs_after_candidate_gate()
    reference_path = ROOT / "data/P9016.1m.3dg.gz"
    reference_structures = _load_reference(reference_path)
    reference = _reference_arrays(reference_structures, data)
    if not np.isfinite(reference).all():
        raise RuntimeError("reference does not cover common 1Mb grid")
    frozen_masks = _build_frozen_masks(data, mask_lock, historical, reference_structures)
    regression = _historical_spearman_regression(data, reference, frozen_masks)
    real_results = []
    for row in candidates:
        if row.get("kind") in ("real", "real_initial"):
            metrics = _chromosome_metrics(row["coordinates"], reference, data, frozen_masks=frozen_masks)
            metrics["inter"] = _inter_metrics(row["coordinates"], reference, data)
            metrics.update({"candidate_id": row["candidate_id"], "model_id": row.get("model_id"), "candidate": row.get("candidate"), "candidate_file_sha256": row["file_sha256"]})
            real_results.append(metrics)
    eval_manifest = json.loads((RUN / "inputs/evaluation_manifest.json").read_text(encoding="utf-8"))
    synthetic_results = []
    for row in candidates:
        if row.get("kind") != "synthetic":
            continue
        record = eval_manifest["truth_records"][str(row["fixture"])]
        if sha256_file(_resolve(record["path"])) != record["sha256"] or sha256_file(_resolve(record["metadata_path"])) != record["metadata_sha256"]:
            raise RuntimeError("synthetic truth hash mismatch")
        with np.load(_resolve(record["path"]), allow_pickle=False) as payload:
            truth = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        same_shape = str(row["fixture"]) == "N2"
        metrics = _chromosome_metrics(row["coordinates"], truth, data, position_offset_bp=0, same_shape_truth_mean=same_shape)
        metrics["inter"] = _inter_metrics(row["coordinates"], truth, data)
        metrics["distance_matrix_errors"] = _distance_matrix_errors(row["coordinates"], truth, data, same_shape)
        metrics.update({"candidate_id": row["candidate_id"], "fixture": row["fixture"], "model_id": row["model_id"], "start_name": row.get("start_name"), "objective_variant": row.get("objective_variant"), "candidate_file_sha256": row["file_sha256"], "truth_file_sha256": record["sha256"], "prior_equilibrium_claim": False})
        synthetic_results.append(metrics)
    paired = {}
    by_id = {row["candidate_id"]: row for row in real_results}
    for source in ("consensus", "random"):
        g, s = by_id.get("real-G-%s" % source), by_id.get("real-S-%s" % source)
        if g is None or s is None:
            continue
        gr = {x["chromosome"]: x for x in g["per_chromosome"] if x.get("status") == "ok"}
        sr = {x["chromosome"]: x for x in s["per_chromosome"] if x.get("status") == "ok"}
        common = sorted(set(gr) & set(sr))
        for metric in ("pearson", "spearman"):
            for key in ("matched", "contrast", "min_margin"):
                ga = np.asarray([gr[name]["metrics"][metric][key] for name in common], dtype=float)
                sa = np.asarray([sr[name]["metrics"][metric][key] for name in common], dtype=float)
                finite = np.isfinite(ga) & np.isfinite(sa)
                paired["G_minus_S-%s-%s-%s" % (source, metric, key)] = {"G_wins": int(np.sum(ga[finite] > sa[finite] + GEOMETRY_TIE_TOL)), "S_wins": int(np.sum(sa[finite] > ga[finite] + GEOMETRY_TIE_TOL)), "ties": int(np.sum(np.abs(ga[finite] - sa[finite]) <= GEOMETRY_TIE_TOL)), "bootstrap": _bootstrap(ga[finite].tolist(), sa[finite].tolist()), "chromosomes_defined": [name for name, keep in zip(common, finite) if keep], "chromosomes_total": common}
    trajectory = _trajectory_rows()
    output = {"schema": "p9016-shared-capture-evaluation-v1", "run_id": RUN.name, "status": "complete", "candidate_hash_gate": "passed before old21 mask/reference/truth payload access",
              "reference": {"path": str(reference_path.relative_to(ROOT)), "sha256": REFERENCE_SHA256, "read_stage": "after_candidate_initial_null_mask_hashes"},
              "real_results": real_results, "synthetic_results": synthetic_results, "paired_G_minus_S": paired, "null_artifacts": null_records, "initial_controls": initial_controls,
              "old036_spearman_regression": regression,
              "frozen_old21_mask": {"lock_path": str(MASK_LOCK_PATH.relative_to(ROOT)), "lock_sha256": MASK_LOCK_SHA256, "manifest_path": str(MASK_MANIFEST_PATH.relative_to(ROOT)), "manifest_sha256": MASK_MANIFEST_SHA256, "condition_count": 21, "total_non_diagonal_pairs": 176201, "common_pairs": 157529, "position_rule": "range(3000000, chromosome_length_bp, 1000000)", "payload_hashes": mask_hashes, "per_chromosome": {name: {key: row[key] for key in ("n_bins", "n_total_non_diagonal_pairs", "n_common_pairs")} for name, row in frozen_masks.items()}},
              "trajectory_rows": len(trajectory),
              "interpretation": {"primary_homolog": "Pearson matched/cross/contrast/margins; Spearman retained separately", "inter_metric": "sorted four-copy distances, denominator=4*eligible common inter pairs", "inter_order_statistic_caveat": "absolute inter r is not packing proof", "N2_common_truth": "mean A/B distance matrices; report common-scale and own-RMS-normalized errors", "chromosome_bootstrap": "technical/structural within one cell, not biological replicates", "no_one_minus_rho_recovery": True}}
    _write(RUN / "results/evaluation.json", output)
    _write_tsv(RUN / "results/trajectory_summary.tsv", trajectory)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate shared-capture endpoints after hash gate")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("command", choices=("evaluate",))
    args = parser.parse_args()
    global RUN
    if args.run_dir is not None:
        RUN = Path(args.run_dir).resolve()
    result = evaluate()
    print(json.dumps(_jsonable({"status": result["status"], "real_results": len(result["real_results"]), "synthetic_results": len(result["synthetic_results"])}), indent=2))


if __name__ == "__main__":
    main()
