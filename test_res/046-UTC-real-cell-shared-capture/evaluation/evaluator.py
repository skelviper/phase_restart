"""Final real-only post-fit evaluator.

This file is intentionally outside the pinned training source tree. It evaluates
only the four P9016 real-cell endpoints after their candidate, initial-control,
and derived-null byte hashes are sealed. Synthetic paths are rejected.
"""
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

RUN = Path(__file__).resolve().parent.parent
SOURCE_RUN = RUN.parent / "045-20260915T073310Z-shared-capture-round"
ROOT = RUN.parents[1]
SOURCE = SOURCE_RUN / "source"
for path in (ROOT, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_io import load_aggregate  # noqa: E402
from formal_controller import _objective as _frozen_objective  # noqa: E402
from pr import contact_model  # noqa: E402
from pr.score import spearman  # noqa: E402

REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
MASK_LOCK_PATH = ROOT / "docs/audits/multires-r2-preparation-20260914_041826/mask_lock.json"
MASK_MANIFEST_PATH = ROOT / "test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/evaluation_manifest.json"
MASK_LOCK_SHA256 = "d0c325dea1289374152a327557605f8bbd52267b025cd7b57689be3e3666c49e"
MASK_MANIFEST_SHA256 = "fa1b26c834c173604c21f954d494cece8e053dd970e7cb6111f561b349574acb"
OLD036_ANCHOR_PATH = ROOT / "test_res/036-20260914T064651Z-gpu-multires/coords/C0/random_joint/final-1m.3dg"
OLD036_ANCHOR_SHA256 = "18c06860f78525e55045e911db05981dd431a86c3bf9ee3ebb5046845b0ccc55"
OLD038_ANCHOR_TSV = ROOT / "test_res/038-20260914T143812Z-gpu-m1-formal/evaluation-r2-20260914T153134Z/r2_real_historical_036_C0_anchor_x20chr.tsv"
OLD038_ANCHOR_TSV_SHA256 = "d44bfe73155edea0f6bb107ed0e41488428a03b230648e381c80066ed4651e21"
BIN_OFFSET_BP = 3_000_000
GEOMETRY_TIE_TOL = 1e-12
MIN_COMMON_PAIRS = 20
RANDOM_U_SEEDS = tuple(range(450500, 450516))
BOOTSTRAP_SEED = 450301
BOOTSTRAP_DRAWS = 10_000
FULL_J_WEIGHTS = {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8", order="C").tobytes(order="C")).hexdigest()


def _source_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (SOURCE_RUN / path).resolve()


def _metric(metric: str, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    if metric == "pearson":
        xc, yc = x - x.mean(), y - y.mean()
        denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
        return float(np.dot(xc, yc) / denom) if denom else float("nan")
    value = float(spearman(x, y))
    return value if math.isfinite(value) else float("nan")


def _distance(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    delta = points[pair_i] - points[pair_j]
    return np.sqrt(np.sum(delta * delta, axis=1))


def _four_correlations(candidate: np.ndarray, reference: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> dict[str, Any]:
    ca, cb = _distance(candidate[0], pair_i, pair_j), _distance(candidate[1], pair_i, pair_j)
    rm, rp = _distance(reference[0], pair_i, pair_j), _distance(reference[1], pair_i, pair_j)
    output = {"n_pairs": int(len(pair_i)), "metrics": {}}
    for metric in ("pearson", "spearman"):
        rho = {"A_mat": _metric(metric, ca, rm), "A_pat": _metric(metric, ca, rp),
               "B_mat": _metric(metric, cb, rm), "B_pat": _metric(metric, cb, rp)}
        finite = all(math.isfinite(float(value)) for value in rho.values())
        reason = None if len(pair_i) >= MIN_COMMON_PAIRS and finite else (
            "common_pairs_below_MIN_COMMON_PAIRS" if len(pair_i) < MIN_COMMON_PAIRS else "one_or_more_of_four_rho_nonfinite")
        direct = float(np.mean([rho["A_mat"], rho["B_pat"]])) if finite else float("nan")
        swapped = float(np.mean([rho["A_pat"], rho["B_mat"]])) if finite else float("nan")
        if reason is not None:
            orientation = "undefined"
            matched = cross = contrast = None
            margins = (None, None, None, None, None)
        elif abs(direct - swapped) <= GEOMETRY_TIE_TOL:
            orientation = "unresolved_tie"
            matched = cross = float((direct + swapped) / 2.0)
            contrast = 0.0
            margins = (None, None, None, None, None)
        elif direct > swapped:
            orientation = "direct"
            matched, cross, contrast = direct, swapped, direct - swapped
            ma, mb = rho["A_mat"] - rho["A_pat"], rho["B_pat"] - rho["B_mat"]
            margins = (ma, mb, ma, mb, min(ma, mb))
        else:
            orientation = "swapped"
            matched, cross, contrast = swapped, direct, swapped - direct
            ma, mb = rho["A_pat"] - rho["A_mat"], rho["B_mat"] - rho["B_pat"]
            margins = (ma, mb, mb, ma, min(ma, mb))
        output["metrics"][metric] = {
            "rho": rho, "direct": direct if math.isfinite(direct) else None,
            "swapped": swapped if math.isfinite(swapped) else None, "matched": matched, "cross": cross,
            "contrast": contrast, "orientation": orientation, "geometry_tie": orientation == "unresolved_tie",
            "margin_copy_A": margins[0], "margin_copy_B": margins[1], "margin_mat": margins[2],
            "margin_pat": margins[3], "min_margin": margins[4], "n_pairs": int(len(pair_i)),
            "derived_defined": reason is None, "undefined_reason": reason,
        }
    return output


def _load_tracks(path: Path, format_name: str, chromosome_names: tuple[str, ...]) -> dict[str, dict[int, np.ndarray]]:
    if format_name in ("native_tsv", "coords_tsv"):
        tracks: dict[str, dict[int, np.ndarray]] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"chr", "copy", "start", "x", "y", "z"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise RuntimeError("native_tsv missing required fields: %s" % sorted(required))
            index = {name: i for i, name in enumerate(chromosome_names)}
            for line_no, row in enumerate(reader, start=2):
                try:
                    chromosome = str(row["chr"])
                    copy = int(row["copy"])
                    position = int(row["start"])
                    point = np.asarray([float(row[axis]) for axis in ("x", "y", "z")], dtype=np.float64)
                    track = "c%02d%s" % (index[chromosome] + 1, "ab"[copy])
                except (KeyError, ValueError, TypeError) as exc:
                    raise RuntimeError("invalid native_tsv row %d in %s" % (line_no, path)) from exc
                if copy not in (0, 1) or position < 0 or position % 1_000_000 != 0:
                    raise RuntimeError("invalid native_tsv coordinate row %d" % line_no)
                if not np.isfinite(point).all():
                    point[:] = np.nan
                target = tracks.setdefault(track, {})
                if position in target:
                    raise RuntimeError("duplicate track coordinate %s:%d" % (track, position))
                target[position] = point
        return tracks
    if format_name not in ("3dg", "3dg_text"):
        raise RuntimeError("unsupported coordinate format %s" % format_name)
    tracks = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                track, position = str(fields[0]), int(fields[1])
                point = np.asarray([float(item) for item in fields[2:5]], dtype=np.float64)
            except (ValueError, TypeError) as exc:
                raise RuntimeError("invalid 3dg row %d in %s" % (line_no, path)) from exc
            if not np.isfinite(point).all():
                point[:] = np.nan
            target = tracks.setdefault(track, {})
            if position in target:
                raise RuntimeError("duplicate track coordinate %s:%d" % (track, position))
            target[position] = point
    return tracks


def _load_candidates() -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    combined = json.loads((RUN / "combined_base_manifest.json").read_text(encoding="utf-8"))
    extension_manifest = json.loads((RUN / "results/candidate_manifest.json").read_text(encoding="utf-8"))
    if combined.get("schema") != "p9016-real-combined-base-manifest-v1" or combined.get("endpoint_count") != 4:
        raise RuntimeError("combined base manifest must contain four endpoints")
    if extension_manifest.get("schema") != "p9016-real-extension-candidate-manifest-v1" or len(extension_manifest.get("candidates", [])) != 4:
        raise RuntimeError("extension candidate manifest must contain four endpoints")
    data = load_aggregate(SOURCE_RUN / "inputs/real_1000000_aggregate.npz")
    rows = []
    for fit_id, item in combined["endpoints"].items():
        rows.append({"candidate_id": fit_id, "fit_id": fit_id, "kind": "real_base", "model_id": item["model_id"],
                     "candidate": item["candidate"], "coordinate_path": item["endpoint_npz"],
                     "coordinate_sha256": item["endpoint_npz_sha256"]})
    rows.extend(extension_manifest["candidates"])
    if len(rows) != 8:
        raise RuntimeError("real combined candidate count must equal eight")
    candidates = []
    for row in rows:
        if row.get("kind") not in ("real_base", "real_extension") or row.get("fixture") is not None:
            raise RuntimeError("non-real candidate in combined manifest")
        path = (RUN / row["coordinate_path"]).resolve()
        if _sha(path) != row["coordinate_sha256"]:
            raise RuntimeError("candidate file hash mismatch %s" % path)
        with np.load(path, allow_pickle=False) as payload:
            coords = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        if coords.shape != (2, data.n_loci, 3) or not np.isfinite(coords).all():
            raise RuntimeError("candidate coordinate shape/domain failure")
        contact_model.assert_inside_unit_ball(coords)
        actual_array_sha256 = _hash_array(coords)
        if row.get("coordinate_array_sha256") is not None and actual_array_sha256 != row["coordinate_array_sha256"]:
            raise RuntimeError("candidate array hash mismatch")
        row["coordinate_array_sha256"] = actual_array_sha256
        row["file_sha256"] = row["coordinate_sha256"]
        candidates.append({**row, "coordinates": coords, "path": path, "candidate_id": str(row["candidate_id"])})
    return candidates, data, {"combined": combined, "extension": extension_manifest}


def _load_initial_controls(data: Any) -> list[dict[str, Any]]:
    formal = json.loads((SOURCE_RUN / "inputs/formal_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for candidate in ("consensus", "random"):
        record = formal["real_initial_paths"][candidate]["stages"]["1Mb"]
        path = _source_path(record["path"])
        if _sha(path) != record["sha256"]:
            raise RuntimeError("initial control hash mismatch")
        with np.load(path, allow_pickle=False) as payload:
            coords = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        if coords.shape != (2, data.n_loci, 3) or not np.isfinite(coords).all():
            raise RuntimeError("initial control shape/domain failure")
        contact_model.assert_inside_unit_ball(coords)
        rows.append({"candidate_id": "initial-%s" % candidate, "kind": "real_initial", "candidate": candidate,
                     "model_id": "initial", "coordinates": coords, "path": path, "file_sha256": _sha(path),
                     "coordinate_array_sha256": _hash_array(coords)})
    return rows


def _full_j_diagnostics(candidates: list[dict[str, Any]], data: Any) -> list[dict[str, Any]]:
    """Score every fixed endpoint under the same original full-J objective."""
    output = []
    for row in candidates:
        objective = _frozen_objective(data, str(row["model_id"]), FULL_J_WEIGHTS, None)
        raw_y = objective.raw_from_physical(row["coordinates"])
        with np.load(row["path"], allow_pickle=False) as payload:
            q = float(np.asarray(payload["q"]).item())
            p = float(np.asarray(payload["p"]).item())
        theta = objective.pack(raw_y, p=p)
        theta[-1] = q
        value, _, components = objective.evaluate(theta, need_gradient=False)
        output.append({"candidate_id": row["candidate_id"], "kind": row.get("kind"), "model_id": row["model_id"],
                       "candidate": row.get("candidate"), "variant": row.get("variant", "base-full-J"),
                       "counterfactual_full_J": row.get("variant") == "count-only", "p": p, "q": q,
                       "full_J_total": float(value), "full_J_weights": FULL_J_WEIGHTS, "components": components,
                       "count_nll_normalized": components.get("count_nll_normalized"),
                       "diagnostic_only": True, "used_for_stop_or_selection": False})
    return output


def _make_u_zero(coords: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    z = (coords[0] + coords[1]) / 2.0
    result = np.stack((z, z), axis=0)
    contact_model.assert_inside_unit_ball(result)
    radius = float(np.linalg.norm(result, axis=2).max())
    return result, {"global_scale": 1.0, "pre_scale_radius": radius, "post_scale_radius": radius}


def _make_random_u(coords: np.ndarray, data: Any, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    z = (coords[0] + coords[1]) / 2.0
    u = (coords[0] - coords[1]) / 2.0
    permuted = np.zeros_like(u)
    rng = np.random.default_rng(seed)
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        permuted[slc] = u[slc][rng.permutation(slc.stop - slc.start)]
    raw = np.stack((z + permuted, z - permuted), axis=0)
    pre = float(np.linalg.norm(raw, axis=2).max())
    scale = 1.0 if pre < 1.0 else (1.0 - 1e-6) / pre
    result = raw * scale
    contact_model.assert_inside_unit_ball(result)
    return result, {"global_scale": scale, "pre_scale_radius": pre, "post_scale_radius": float(np.linalg.norm(result, axis=2).max())}


def _write_nulls(sources: list[dict[str, Any]], data: Any) -> list[dict[str, Any]]:
    out = []
    directory = RUN / "coords/nulls"
    directory.mkdir(parents=True, exist_ok=True)
    for source in sources:
        for kind, seed in [("u_zero", None)] + [("random_u", seed) for seed in RANDOM_U_SEEDS]:
            coords, audit = _make_u_zero(source["coordinates"],) if seed is None else _make_random_u(source["coordinates"], data, seed)
            name = "%s-%s%s.npz" % (source["candidate_id"], kind, "" if seed is None else "-%d" % seed)
            path = directory / name
            np.savez_compressed(path, coordinates=coords)
            out.append({"source_candidate_id": source["candidate_id"], "null_kind": kind, "seed": seed,
                        "path": str(path.relative_to(RUN)), "sha256": _sha(path), "coordinate_array_sha256": _hash_array(coords),
                        **audit})
    return out


def _build_masks(data: Any, lock: Mapping[str, Any], historical: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
                 reference_tracks: Mapping[str, Mapping[int, np.ndarray]]) -> dict[str, dict[str, Any]]:
    expected = {str(row["chromosome"]): row for row in lock["expected_by_chromosome"]}
    masks = {}
    for chromosome, name in enumerate(data.chromosome_names):
        length = int(data.chromosome_lengths[chromosome])
        positions = np.arange(BIN_OFFSET_BP, length, int(data.bin_size), dtype=np.int64)
        local = positions // int(data.bin_size)
        pair_i, pair_j = np.triu_indices(len(positions), k=1)
        common = np.ones(len(pair_i), dtype=bool)
        for condition_id, tracks in historical.items():
            suffixes = ("a",) if condition_id == "consensus014" else ("a", "b")
            for suffix in suffixes:
                track = "c%02d%s" % (chromosome + 1, suffix)
                if track not in tracks:
                    raise RuntimeError("missing required track %s in frozen condition %s" % (track, condition_id))
                points = np.asarray([tracks[track].get(int(position), [np.nan] * 3) for position in positions], dtype=np.float64)
                common &= np.isfinite(_distance(points, pair_i, pair_j))
        for suffix in ("mat", "pat"):
            track = reference_tracks.get("%s(%s)" % (name, suffix))
            if track is None:
                raise RuntimeError("reference lacks required track %s(%s)" % (name, suffix))
            points = np.asarray([track.get(int(position), [np.nan] * 3) for position in positions], dtype=np.float64)
            common &= np.isfinite(_distance(points, pair_i, pair_j))
        valid_bins = np.zeros(len(positions), dtype=bool)
        valid_bins[pair_i[common]] = True
        valid_bins[pair_j[common]] = True
        counts = {"n_bins": int(len(positions)), "n_total_non_diagonal_pairs": int(len(pair_i)), "n_common_pairs": int(common.sum())}
        row = expected.get(str(name))
        if row is None or any(counts[key] != int(row[key]) for key in counts):
            raise RuntimeError("frozen old21 mask mismatch %s: %r vs %r" % (name, counts, row))
        masks[str(name)] = {"positions": positions, "local_bins": local, "pair_i": pair_i, "pair_j": pair_j,
                            "common": common, "valid_local_bins": np.flatnonzero(valid_bins), **counts}
    if sum(item["n_common_pairs"] for item in masks.values()) != 157529 or sum(item["n_total_non_diagonal_pairs"] for item in masks.values()) != 176201:
        raise RuntimeError("frozen old21 mask totals changed")
    return masks


def _chromosome_metrics(candidate: np.ndarray, reference: np.ndarray, data: Any, masks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for chromosome, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(chromosome)
        mask = masks[str(name)]
        local = np.asarray(mask["local_bins"], dtype=np.int64)
        pair_i = np.asarray(mask["pair_i"], dtype=np.int64)
        pair_j = np.asarray(mask["pair_j"], dtype=np.int64)
        common = np.asarray(mask["common"], dtype=bool)
        pair_i, pair_j = pair_i[common], pair_j[common]
        cand = candidate[:, slc.start:slc.stop][:, local]
        ref = reference[:, slc.start:slc.stop][:, local]
        row = _four_correlations(cand, ref, pair_i, pair_j)
        row.update({"chromosome": str(name), "chromosome_index": chromosome, "status": "ok"})
        rows.append(row)
    keys = ("matched", "cross", "contrast", "min_margin", "margin_copy_A", "margin_copy_B", "margin_mat", "margin_pat")
    macro, defined = {}, {}
    for metric in ("pearson", "spearman"):
        macro[metric], defined[metric] = {}, {}
        for key in keys:
            values = [row["metrics"][metric][key] for row in rows if row["metrics"][metric][key] is not None and math.isfinite(float(row["metrics"][metric][key]))]
            macro[metric][key] = float(np.mean(values)) if values else None
            defined[metric][key] = len(values)
    return {"per_chromosome": rows, "macro_equal_chromosome_weight": macro, "defined_chromosome_counts": defined,
            "denominator_chromosomes": len(data.chromosome_names), "metric_definition": "frozen old21 common finite pairs",
            "homolog_primary": "matched/cross/contrast plus copy-A/copy-B margins; no center-distance surrogate"}


def _valid_loci(data: Any, masks: Mapping[str, Mapping[str, Any]]) -> np.ndarray:
    valid = np.zeros(data.n_loci, dtype=bool)
    for chromosome, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(chromosome)
        mask = masks[str(name)]
        valid[slc.start + np.asarray(mask["valid_local_bins"], dtype=np.int64)] = True
    return valid


def _inter_metrics(candidate: np.ndarray, reference: np.ndarray, data: Any, masks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    valid_loci = _valid_loci(data, masks)
    inter = ~np.asarray(data.cis_pair, dtype=bool)
    pair_i = np.asarray(data.pair_i[inter], dtype=np.int64)
    pair_j = np.asarray(data.pair_j[inter], dtype=np.int64)
    keep = valid_loci[pair_i] & valid_loci[pair_j]
    pair_i, pair_j = pair_i[keep], pair_j[keep]
    finite = np.isfinite(candidate[:, pair_i]).all(axis=(0, 2)) & np.isfinite(reference[:, pair_i]).all(axis=(0, 2)) & np.isfinite(candidate[:, pair_j]).all(axis=(0, 2)) & np.isfinite(reference[:, pair_j]).all(axis=(0, 2))
    if not finite.all():
        raise RuntimeError("reference/candidate nonfinite on frozen inter valid-bin set")
    candidate_distances = np.stack((_distance(candidate[0], pair_i, pair_j),
                                    np.sqrt(np.sum((candidate[0, pair_i] - candidate[1, pair_j]) ** 2, axis=1)),
                                    np.sqrt(np.sum((candidate[1, pair_i] - candidate[0, pair_j]) ** 2, axis=1)),
                                    _distance(candidate[1], pair_i, pair_j)), axis=1)
    reference_distances = np.stack((_distance(reference[0], pair_i, pair_j),
                                    np.sqrt(np.sum((reference[0, pair_i] - reference[1, pair_j]) ** 2, axis=1)),
                                    np.sqrt(np.sum((reference[1, pair_i] - reference[0, pair_j]) ** 2, axis=1)),
                                    _distance(reference[1], pair_i, pair_j)), axis=1)
    x, y = np.sort(candidate_distances, axis=1).ravel(), np.sort(reference_distances, axis=1).ravel()
    return {"pearson": _metric("pearson", x, y), "spearman": _metric("spearman", x, y),
            "eligible_locus_pairs": int(len(pair_i)), "denominator": int(4 * len(pair_i)),
            "valid_locus_count": int(valid_loci.sum()), "valid_bin_policy": "old21 common-mask participating bins only",
            "order_statistic_vector": "sorted_four_copy_distances_per_inter_locus_pair", "high_absolute_r_not_packing_proof": True}


def _bootstrap(left: list[float], right: list[float], names: list[str]) -> dict[str, Any]:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    defined_names = [name for name, keep in zip(names, finite) if keep]
    undefined_names = [name for name, keep in zip(names, finite) if not keep]
    a, b = a[finite], b[finite]
    output = {"denominator_chromosomes": len(names), "defined_chromosomes": len(a), "undefined_chromosomes": undefined_names,
              "defined_chromosome_names": defined_names, "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
              "unit": "paired chromosome technical/structural bootstrap; not biological replicates"}
    if not len(a):
        output.update({"mean": None, "ci95": [None, None]})
        return output
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.asarray([np.mean((a - b)[rng.integers(0, len(a), len(a))]) for _ in range(BOOTSTRAP_DRAWS)])
    output.update({"mean": float(np.mean(a - b)), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]})
    return output


def _paired(real_results: list[dict[str, Any]], left_id: str, right_id: str, label: str, data: Any) -> dict[str, Any]:
    by_id = {row["candidate_id"]: row for row in real_results}
    left, right = by_id.get(left_id), by_id.get(right_id)
    if left is None or right is None:
        return {"status": "undefined_missing_candidate", "left": left_id, "right": right_id, "denominator_chromosomes": len(data.chromosome_names)}
    left_rows = {row["chromosome"]: row for row in left["r2"]["per_chromosome"]}
    right_rows = {row["chromosome"]: row for row in right["r2"]["per_chromosome"]}
    names = [str(name) for name in data.chromosome_names]
    output = {"left": left_id, "right": right_id, "denominator_chromosomes": len(names), "metrics": {}}
    keys = ("matched", "cross", "contrast", "min_margin", "margin_copy_A", "margin_copy_B", "margin_mat", "margin_pat")
    for metric in ("pearson", "spearman"):
        for key in keys:
            a = [left_rows.get(name, {}).get("metrics", {}).get(metric, {}).get(key, np.nan) for name in names]
            b = [right_rows.get(name, {}).get("metrics", {}).get(metric, {}).get(key, np.nan) for name in names]
            finite = np.isfinite(a) & np.isfinite(b)
            output["metrics"]["%s:%s" % (metric, key)] = {
                "left_wins": int(np.sum(np.asarray(a)[finite] > np.asarray(b)[finite] + GEOMETRY_TIE_TOL)),
                "right_wins": int(np.sum(np.asarray(b)[finite] > np.asarray(a)[finite] + GEOMETRY_TIE_TOL)),
                "ties": int(np.sum(np.abs(np.asarray(a)[finite] - np.asarray(b)[finite]) <= GEOMETRY_TIE_TOL)),
                "bootstrap": _bootstrap(a, b, names), "denominator_chromosomes": len(names)}
    output["label"] = label
    return output


def _load_reference_after_gate(data: Any) -> tuple[dict[str, dict[int, np.ndarray]], str]:
    reference_path = ROOT / "data/P9016.1m.3dg.gz"
    if _sha(reference_path) != REFERENCE_SHA256:
        raise RuntimeError("reference SHA mismatch")
    return _load_tracks(reference_path, "3dg", tuple(data.chromosome_names)), _sha(reference_path)


def _trajectory_rows() -> list[dict[str, Any]]:
    formal = json.loads((SOURCE_RUN / "inputs/formal_manifest.json").read_text(encoding="utf-8"))
    data_cache = {int(size): load_aggregate(Path(path)) for size, path in formal["real_input_paths"].items()}
    rows = []
    for fit in formal["matrix"]:
        if fit.get("kind") != "real":
            continue
        fit_id = str(fit["fit_id"])
        for stage in ("5Mb", "2Mb", "1Mb"):
            stage_dir = RUN / "stages" / fit_id
            record_path = stage_dir / (stage + ".json")
            if not record_path.is_file():
                continue
            bin_size = int(next(item["bin_size_bp"] for item in fit["stages"] if item["stage"] == stage))
            data = data_cache[bin_size]
            for checkpoint in sorted((RUN / "checkpoints" / fit_id / stage).glob("accepted-*.npz")):
                with np.load(checkpoint, allow_pickle=False) as payload:
                    coords = np.asarray(payload["coordinates"], dtype=np.float64)
                    theta = np.asarray(payload["theta"], dtype=np.float64)
                    iteration = int(np.asarray(payload["iteration"]).item())
                p = float(contact_model.p_from_q(float(theta[-1]))[0])
                pair_i, pair_j = data.pair_i.astype(np.int64), data.pair_j.astype(np.int64)
                d2 = [np.sum((coords[copy_i, pair_i] - coords[copy_j, pair_j]) ** 2, axis=1) for copy_i, copy_j in ((0, 0), (0, 1), (1, 0), (1, 1))]
                r0 = 2.0 * (2.0 * data.n_loci) ** (-1.0 / 3.0)
                kernels = [1e-6 + (1.0 - 1e-6) * (1.0 + values / (r0 * r0)) ** -2 for values in d2]
                cis = np.asarray(data.cis_pair, dtype=bool)
                wk = np.where(cis, 0.5 * (p * (kernels[0] + kernels[3]) + (1.0 - p) * (kernels[1] + kernels[2])),
                              0.25 * (kernels[0] + kernels[1] + kernels[2] + kernels[3]))
                counts = np.asarray(data.counts, dtype=np.float64)
                def weighted(mask: np.ndarray, values: np.ndarray) -> float | None:
                    total = float(counts[mask].sum())
                    return float(np.sum(counts[mask] * values[mask]) / total) if total > 0 else None
                radius = [float(np.sqrt(np.mean(np.sum((coords[copy] - coords[copy].mean(axis=0)) ** 2, axis=1)))) for copy in (0, 1)]
                argmax = int(np.argmax(wk))
                rows.append({"fit_id": fit_id, "model_id": fit["model_id"], "candidate": fit["candidate"], "stage": stage,
                             "iteration": iteration, "nfev": int(np.asarray(payload["nfev"]).item()) if "nfev" in payload else None,
                             "argmax_wK": float(wk[argmax]), "argmax_wK_pair_i": int(pair_i[argmax]), "argmax_wK_pair_j": int(pair_j[argmax]),
                             "weighted_intra_wK": weighted(cis, wk), "weighted_inter_wK": weighted(~cis, wk),
                             "weighted_intra_distance": weighted(cis, np.mean(np.sqrt(np.stack(d2, axis=0)), axis=0)),
                             "weighted_inter_distance": weighted(~cis, np.mean(np.sqrt(np.stack(d2, axis=0)), axis=0)),
                             "Rg_copy_A": radius[0], "Rg_copy_B": radius[1], "checkpoint": str(checkpoint.relative_to(RUN))})
    return rows


def _trajectory_rows_combined() -> list[dict[str, Any]]:
    formal = json.loads((SOURCE_RUN / "inputs/formal_manifest.json").read_text(encoding="utf-8"))
    data_cache = {int(size): load_aggregate(Path(path)) for size, path in formal["real_input_paths"].items()}
    combined = json.loads((RUN / "combined_base_manifest.json").read_text(encoding="utf-8"))
    stage_items = list(combined["stages"])
    extension_manifest_path = RUN / "extension_manifest.json"
    if extension_manifest_path.is_file():
        extension_manifest = json.loads(extension_manifest_path.read_text(encoding="utf-8"))
        stage_items.extend({"fit_id": arm["fit_id"], "stage": "1Mb", "lineage": "extension"} for arm in extension_manifest.get("arms", []))
    rows = []
    size_by_stage = {"5Mb": 5_000_000, "2Mb": 2_000_000, "1Mb": 1_000_000}
    for item in stage_items:
        fit_id, stage = item["fit_id"], item["stage"]
        if item.get("lineage") == "extension":
            checkpoint_dir = RUN / "checkpoints" / fit_id / stage
        else:
            lineage = Path(item["record_path"]).parts[0]
            checkpoint_root = "trajectory_inputs/reused_base" if lineage == "reused_base" else lineage
            checkpoint_dir = RUN / checkpoint_root / "checkpoints" / fit_id / stage
        data = data_cache[size_by_stage[stage]]
        if not checkpoint_dir.is_dir():
            continue
        parts = fit_id.split("-")
        if fit_id.startswith("real-extension-"):
            model_id, candidate = parts[2], "-".join(parts[3:])
        else:
            model_id = parts[1] if len(parts) > 1 else "unknown"
            candidate = parts[2] if len(parts) > 2 else "unknown"
        for checkpoint in sorted(checkpoint_dir.glob("accepted-*.npz")):
            with np.load(checkpoint, allow_pickle=False) as payload:
                coords = np.asarray(payload["coordinates"], dtype=np.float64)
                theta = np.asarray(payload["theta"], dtype=np.float64)
                iteration = int(np.asarray(payload["iteration"]).item())
                nfev = int(np.asarray(payload["nfev"]).item()) if "nfev" in payload else None
            p = float(contact_model.p_from_q(float(theta[-1]))[0])
            pair_i, pair_j = data.pair_i.astype(np.int64), data.pair_j.astype(np.int64)
            d2 = [np.sum((coords[copy_i, pair_i] - coords[copy_j, pair_j]) ** 2, axis=1) for copy_i, copy_j in ((0, 0), (0, 1), (1, 0), (1, 1))]
            r0 = 2.0 * (2.0 * data.n_loci) ** (-1.0 / 3.0)
            kernels = [1e-6 + (1.0 - 1e-6) * (1.0 + values / (r0 * r0)) ** -2 for values in d2]
            cis = np.asarray(data.cis_pair, dtype=bool)
            wk = np.where(cis, 0.5 * (p * (kernels[0] + kernels[3]) + (1.0 - p) * (kernels[1] + kernels[2])),
                          0.25 * (kernels[0] + kernels[1] + kernels[2] + kernels[3]))
            counts = np.asarray(data.counts, dtype=np.float64)
            def weighted(mask: np.ndarray, values: np.ndarray) -> float | None:
                total = float(counts[mask].sum())
                return float(np.sum(counts[mask] * values[mask]) / total) if total > 0 else None
            # Primary Rg is whole-cell, unweighted, centered by the mean of all physical beads.
            rg = [float(np.sqrt(np.mean(np.sum((coords[copy] - coords[copy].mean(axis=0)) ** 2, axis=1)))) for copy in (0, 1)]
            argmax = int(np.argmax(wk))
            rows.append({"fit_id": fit_id, "model_id": model_id, "candidate": candidate, "stage": stage,
                         "iteration": iteration, "nfev": nfev, "whole_cell_beads": int(data.n_loci),
                         "argmax_wK": float(wk[argmax]), "argmax_wK_pair_i": int(pair_i[argmax]), "argmax_wK_pair_j": int(pair_j[argmax]),
                         "weighted_intra_wK": weighted(cis, wk), "weighted_inter_wK": weighted(~cis, wk),
                         "weighted_intra_distance": weighted(cis, np.mean(np.sqrt(np.stack(d2, axis=0)), axis=0)),
                         "weighted_inter_distance": weighted(~cis, np.mean(np.sqrt(np.stack(d2, axis=0)), axis=0)),
                         "Rg_whole_cell_copy_A": rg[0], "Rg_whole_cell_copy_B": rg[1],
                         "Rg_definition": "all physical beads, whole-cell mean center, no count weights",
                         "checkpoint": str(checkpoint.relative_to(RUN))})
    return rows


    if _sha(OLD036_ANCHOR_PATH) != OLD036_ANCHOR_SHA256 or _sha(OLD038_ANCHOR_TSV) != OLD038_ANCHOR_TSV_SHA256:
        raise RuntimeError("old036/038 regression hash mismatch")
    tracks = _load_tracks(OLD036_ANCHOR_PATH, "3dg", tuple(data.chromosome_names))
    anchor = np.full((2, data.n_loci, 3), np.nan, dtype=np.float64)
    for chromosome, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(chromosome)
        for copy, suffix in enumerate(("a", "b")):
            rows = tracks.get("c%02d%s" % (chromosome + 1, suffix), {})
            for local, position in enumerate(data.locus_bin[slc] * int(data.bin_size)):
                if int(position) in rows:
                    anchor[copy, slc.start + local] = rows[int(position)]
    if not np.isfinite(anchor).all():
        raise RuntimeError("old036 anchor incomplete")
    new = _chromosome_metrics(anchor, reference, data, masks)
    new_by = {row["chromosome"]: row for row in new["per_chromosome"]}
    with OLD038_ANCHOR_TSV.open("r", encoding="utf-8", newline="") as handle:
        old_rows = list(csv.DictReader(handle))
    fields = ("n_bins", "n_total_non_diagonal_pairs", "n_common_pairs_frozen", "rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat", "direct", "swapped", "matched", "cross", "contrast", "margin_mat", "margin_pat", "minmargin")
    mismatches = []
    for old in old_rows:
        name = str(old["chromosome"])
        row = new_by.get(name)
        if row is None:
            mismatches.append({"chromosome": name, "field": "row"})
            continue
        metric = row["metrics"]["spearman"]
        mask = masks[name]
        values = {"n_bins": len(mask["positions"]), "n_total_non_diagonal_pairs": mask["n_total_non_diagonal_pairs"], "n_common_pairs_frozen": mask["n_common_pairs"],
                  "rho_A_mat": metric["rho"]["A_mat"], "rho_A_pat": metric["rho"]["A_pat"], "rho_B_mat": metric["rho"]["B_mat"], "rho_B_pat": metric["rho"]["B_pat"],
                  "direct": metric["direct"], "swapped": metric["swapped"], "matched": metric["matched"], "cross": metric["cross"], "contrast": metric["contrast"], "margin_mat": metric["margin_mat"], "margin_pat": metric["margin_pat"], "minmargin": metric["min_margin"]}
        for field in fields:
            if values[field] is None or abs(float(values[field]) - float(old[field])) > 1e-12:
                mismatches.append({"chromosome": name, "field": field, "old": old[field], "new": values[field]})
    output = {"schema": "p9016-real-only-old036-spearman-regression-v1", "status": "PASS" if not mismatches else "FAIL",
              "numeric_atol": 1e-12, "numeric_rtol": 0.0, "mismatch_count": len(mismatches), "mismatches": mismatches[:100],
              "old_anchor_sha256": OLD036_ANCHOR_SHA256, "old038_tsv_sha256": OLD038_ANCHOR_TSV_SHA256, "not_used_for_selection": True}
    _write(RUN / "results/old036_spearman_regression.json", output)
    if mismatches:
        raise RuntimeError("old036 Spearman regression failed")
    return output


def _plot(results: list[dict[str, Any]], trajectory: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        _write(RUN / "results/plot_status.json", {"status": "unavailable", "error": str(exc)})
        return []
    paths = []
    fig, ax = plt.subplots(figsize=(3, 2.2), dpi=300)
    labels, values = [], []
    for row in results:
        labels.append(row["candidate_id"])
        values.append(row["r2"]["macro_equal_chromosome_weight"]["pearson"]["matched"] or np.nan)
    ax.bar(np.arange(len(labels)), values, color="#365f8d")
    ax.set_ylabel("Pearson matched")
    ax.set_xlabel("Real endpoint")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=5)
    fig.tight_layout()
    path = RUN / "results/r2_matched_summary.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(str(path.relative_to(RUN)))
    fig, ax = plt.subplots(figsize=(3, 2.2), dpi=300)
    for fit_id in sorted({row["fit_id"] for row in trajectory}):
        subset = [row for row in trajectory if row["fit_id"] == fit_id]
        ax.plot([row["iteration"] for row in subset], [row["argmax_wK"] for row in subset], label=fit_id, linewidth=0.7)
    ax.set_xlabel("Accepted iteration")
    ax.set_ylabel("argmax(wK)")
    if trajectory:
        ax.legend(fontsize=4, ncol=2)
    fig.tight_layout()
    path = RUN / "results/trajectory_wK.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths.append(str(path.relative_to(RUN)))
    return paths


def evaluate() -> dict[str, Any]:
    candidates, data, candidate_manifest = _load_candidates()
    initial = _load_initial_controls(data)
    sources = candidates + initial
    null_records = _write_nulls(sources, data)
    gate_records = []
    for row in sources:
        gate_records.append({"candidate_id": row["candidate_id"], "path": str(row["path"]), "sha256": row["file_sha256"]})
    gate_records.extend({key: record[key] for key in ("source_candidate_id", "null_kind", "seed", "path", "sha256")} for record in null_records)
    for record in gate_records:
        path = (RUN / record["path"]).resolve() if not Path(record["path"]).is_absolute() else Path(record["path"])
        if _sha(path) != record["sha256"]:
            raise RuntimeError("pre-reference hash gate changed: %s" % path)
    _write(RUN / "results/pre_reference_hash_gate.json", {"status": "PASS", "reference_opened": False, "phase_opened": False,
                                                            "candidate_manifest_sha256": _sha(RUN / "results/candidate_manifest.json"),
                                                             "combined_base_manifest_sha256": _sha(RUN / "combined_base_manifest.json"),
                                                            "records": gate_records, "candidate_count": 8, "initial_count": 2,
                                                            "null_count": len(null_records), "synthetic_count": 0})
    full_j_diagnostics = _full_j_diagnostics(candidates, data)
    _write(RUN / "results/full_j_endpoint_diagnostics.json", {"schema": "p9016-real-endpoint-full-j-diagnostics-v1",
                                                                  "diagnostic_only": True, "used_for_stop_or_selection": False,
                                                                  "weights": FULL_J_WEIGHTS, "rows": full_j_diagnostics,
                                                                  "reference_opened": False, "synthetic_count": 0})
    # Only after every candidate/initial/null hash has passed may masks and reference be parsed.
    mask_lock = json.loads(MASK_LOCK_PATH.read_text(encoding="utf-8"))
    mask_manifest = json.loads(MASK_MANIFEST_PATH.read_text(encoding="utf-8"))
    historical = {}
    mask_hashes = []
    for row in mask_manifest.get("coordinates", []):
        if row.get("mask_included") is not True:
            continue
        path = Path(str(row["path"])).resolve()
        if _sha(path) != str(row["sha256"]):
            raise RuntimeError("historical mask coordinate hash mismatch")
        historical[str(row["condition_id"])] = _load_tracks(path, str(row.get("format", "3dg")), tuple(data.chromosome_names))
        mask_hashes.append({"condition_id": str(row["condition_id"]), "path": str(path), "sha256": _sha(path), "format": row.get("format")})
    reference_tracks, reference_sha = _load_reference_after_gate(data)
    reference = np.full((2, data.n_loci, 3), np.nan, dtype=np.float64)
    for chromosome, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(chromosome)
        for copy, suffix in enumerate(("mat", "pat")):
            track = reference_tracks.get("%s(%s)" % (name, suffix), {})
            for local, position in enumerate(data.locus_bin[slc] * int(data.bin_size)):
                if int(position) in track:
                    reference[copy, slc.start + local] = track[int(position)]
    masks = _build_masks(data, mask_lock, historical, reference_tracks)
    real_results = []
    for row in candidates + initial:
        real_results.append({"candidate_id": row["candidate_id"], "kind": row["kind"], "candidate": row.get("candidate"),
                             "variant": row.get("variant", "base-full-J"), "model_id": row.get("model_id"), "coordinates_sha256": row["coordinate_array_sha256"],
                             "r2": _chromosome_metrics(row["coordinates"], reference, data, masks),
                             "inter": _inter_metrics(row["coordinates"], reference, data, masks)})
    null_results = []
    source_map = {row["candidate_id"]: row for row in sources}
    for record in null_records:
        with np.load(RUN / record["path"], allow_pickle=False) as payload:
            coords = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        null_results.append({**record, "r2": _chromosome_metrics(coords, reference, data, masks),
                             "inter": _inter_metrics(coords, reference, data, masks)})
    selection_path = RUN / "combined_base_manifest.json"
    selection = candidate_manifest["combined"].get("selected_sources", {})
    paired = {}
    for candidate in ("consensus", "random"):
        paired["G_minus_S-%s" % candidate] = _paired(real_results, "real-G-%s" % candidate, "real-S-%s" % candidate, candidate, data)
    selected_ids = {}
    for model in ("S", "G"):
        selected = selection.get(model, {}).get("selected_by_own_count_nll")
        selected_ids[model] = "real-%s-%s" % (model, selected) if selected in ("consensus", "random") else None
    if selected_ids["S"] and selected_ids["G"]:
        paired["G_selected_minus_S_selected"] = _paired(real_results, selected_ids["G"], selected_ids["S"], "selected_by_own_count_nll", data)
    extension_diagnostics = {}
    for model in ("S", "G"):
        base_id = selected_ids.get(model)
        if base_id:
            for variant in ("full-J", "count-only"):
                extension_id = "real-extension-%s-%s" % (model, variant)
                extension_diagnostics["%s_vs_selected_base" % extension_id] = _paired(real_results, extension_id, base_id,
                                                                                         "extension_vs_selected_base_%s" % variant, data)
    regression = _old036_regression(data, reference, masks)
    output = {"schema": "p9016-real-only-evaluation-v3", "status": "complete", "scope": "P9016 real cell only",
              "synthetic_evaluation": "cancelled_by_latest_user_scope", "candidate_hash_gate": "passed before mask/reference access",
              "reference": {"path": str((ROOT / "data/P9016.1m.3dg.gz").relative_to(ROOT)), "sha256": reference_sha,
                            "read_stage": "after all eight endpoints, two initial controls, and 170 null hashes"},
              "real_results": real_results, "full_J_endpoint_diagnostics": full_j_diagnostics,
              "extension_diagnostics": extension_diagnostics, "null_results": null_results, "null_summary": {"sources": len(sources), "draws_per_source": 17},
              "initial_controls": [{"candidate_id": row["candidate_id"], "sha256": row["file_sha256"]} for row in initial],
              "paired_G_minus_S": paired, "selected_sources": selected_ids, "source_selection_file": str(selection_path.relative_to(RUN)),
              "old036_spearman_regression": regression,
              "frozen_old21_mask": {"lock_sha256": MASK_LOCK_SHA256, "manifest_sha256": MASK_MANIFEST_SHA256, "condition_count": 21,
                                    "total_non_diagonal_pairs": 176201, "common_pairs": 157529, "position_rule": "range(3000000, chromosome_length_bp, 1000000)",
                                    "native_tsv_and_single_copy_parser": True, "per_chromosome": {name: {key: row[key] for key in ("n_bins", "n_total_non_diagonal_pairs", "n_common_pairs")} for name, row in masks.items()},
                                    "payload_hashes": mask_hashes},
              "trajectory_rows": trajectory, "trajectory_row_count": len(trajectory),
              "trajectory_inputs_manifest": {"path": "trajectory_inputs/manifest.json", "sha256": _sha(RUN / "trajectory_inputs/manifest.json"), "reused_missing_points_are_NA": True},
              "plot_paths": _plot(real_results, trajectory),
              "metric_rules": {"MIN_COMMON_PAIRS": MIN_COMMON_PAIRS, "tie_atol": GEOMETRY_TIE_TOL,
                               "inter": "old21 common-mask participating bins, sorted four-copy distances, full finite check; no reference finite expansion",
                               "nulls": "same endpoint and frozen mask for u0 plus 16 random-u draws; global rescale/radius recorded",
                               "bootstrap_denominator": "all 20 chromosomes; undefined chromosomes retained in denominator"}}
    _write(RUN / "results/evaluation.json", output)
    rows = []
    for result in real_results:
        for row in result["r2"]["per_chromosome"]:
            for metric in ("pearson", "spearman"):
                item = {"candidate_id": result["candidate_id"], "kind": result["kind"], "chromosome": row["chromosome"], "metric": metric}
                item.update(row["metrics"][metric])
                rows.append(item)
    if rows:
        with (RUN / "results/r2_per_chromosome.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(rows)
    if full_j_diagnostics:
        diagnostic_rows = []
        for item in full_j_diagnostics:
            row = {"candidate_id": item["candidate_id"], "kind": item["kind"], "model_id": item["model_id"],
                   "candidate": item.get("candidate"), "variant": item.get("variant"), "counterfactual_full_J": item["counterfactual_full_J"],
                   "full_J_total": item["full_J_total"], "count_nll_normalized": item.get("count_nll_normalized")}
            for key, value in item["components"].items():
                row["component:" + str(key)] = value
            diagnostic_rows.append(row)
        with (RUN / "results/full_j_endpoint_diagnostics.tsv").open("w", encoding="utf-8", newline="") as handle:
            fields = list(diagnostic_rows[0])
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader(); writer.writerows(diagnostic_rows)
    null_rows = []
    for result in null_results:
        macro = result["r2"]["macro_equal_chromosome_weight"]
        null_rows.append({"source_candidate_id": result["source_candidate_id"], "null_kind": result["null_kind"], "seed": result["seed"],
                          "global_scale": result.get("global_scale"), "pre_scale_radius": result.get("pre_scale_radius"),
                          "post_scale_radius": result.get("post_scale_radius"), "pearson_matched": macro["pearson"]["matched"],
                          "spearman_matched": macro["spearman"]["matched"], "inter_pearson": result["inter"]["pearson"], "inter_spearman": result["inter"]["spearman"]})
    if null_rows:
        with (RUN / "results/null_evaluations.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(null_rows[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(null_rows)
    if trajectory:
        with (RUN / "results/trajectory_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trajectory[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(trajectory)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("evaluate",))
    args = parser.parse_args()
    result = evaluate()
    print(json.dumps({"status": result["status"], "real_results": len(result["real_results"]), "null_results": len(result["null_results"]),
                      "reference_sha256": result["reference"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
