"""所有056状态哈希完成后的统一reference评价；训练侧不导入本模块。"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
for path in (ROOT, S045):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import data_io  # noqa: E402
from pr.solver_state import load_solver_state, sha256_file  # noqa: E402

MASK = ROOT / "test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/frozen_legacy_mask_snapshot.npz"
MASK_SHA = "9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9"
REFERENCE = ROOT / "data/P9016.1m.3dg.gz"
REFERENCE_SHA = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
BASELINE = ROOT / "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.npz"
BASELINE_SHA = "6bb93bf570cf688fd8b02dccf14ca0867c1f831d5e123036ced626856f8d8752"
TSV055 = ROOT / "test_res/055-20260916_112451-max-contact-vs-baseline-review/eval/per_chromosome.tsv"
CHROMOSOMES = ["chr%d" % index for index in range(1, 20)] + ["chrX"]
TIE_TOL = 1e-12


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def parse_3dg(path):
    tracks = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) == 5:
                tracks.setdefault(fields[0], {})[int(fields[1])] = np.asarray(fields[2:5], dtype=np.float64)
    return tracks


def reference_names(reference, chromosome):
    found = {}
    for name in reference:
        match = re.match(r"^%s\((mat|pat)\)$" % re.escape(chromosome), name)
        if match:
            found[match.group(1)] = name
    if set(found) != {"mat", "pat"}:
        raise RuntimeError("reference copies missing for %s" % chromosome)
    return found["mat"], found["pat"]


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64); right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(); right = right - right.mean()
    denominator = math.sqrt(float(np.dot(left, left)) * float(np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")


def distance(points, i, j):
    delta = points[i] - points[j]
    return np.sqrt(np.sum(delta * delta, axis=1))


def rho_for(coords, data, chromosome_index, positions, pair_i, pair_j, common, reference, chromosome):
    slc = data.chromosome_slice(chromosome_index)
    candidate_index = slc.start + positions // int(data.bin_size)
    candidate = np.asarray(coords)[:, candidate_index]
    ref_names = reference_names(reference, chromosome)
    ref = []
    for name in ref_names:
        values = np.full((len(positions), 3), np.nan, dtype=np.float64)
        table = reference[name]
        for index, position in enumerate(positions.tolist()):
            if int(position) in table:
                values[index] = table[int(position)]
        ref.append(values)
    finite = common.copy()
    for values in (candidate[0], candidate[1], ref[0], ref[1]):
        finite &= np.isfinite(values[pair_i]).all(axis=1) & np.isfinite(values[pair_j]).all(axis=1)
    if not np.array_equal(finite, common):
        raise RuntimeError("shared 055 support dropped for %s" % chromosome)
    i, j = pair_i[common], pair_j[common]
    da, db = distance(candidate[0], i, j), distance(candidate[1], i, j)
    dm, dp = distance(ref[0], i, j), distance(ref[1], i, j)
    return {"A_mat": pearson(da, dm), "A_pat": pearson(da, dp),
            "B_mat": pearson(db, dm), "B_pat": pearson(db, dp)}, int(common.sum())


def metrics(rho, fixed_orientation=None):
    direct = (rho["A_mat"] + rho["B_pat"]) / 2.0
    swapped = (rho["A_pat"] + rho["B_mat"]) / 2.0
    best = "direct" if direct > swapped else "swapped"
    if abs(direct - swapped) <= TIE_TOL:
        best = "unresolved_tie"
    orientation = best if fixed_orientation is None else fixed_orientation
    if orientation == "direct":
        margin_a = rho["A_mat"] - rho["A_pat"]
        margin_b = rho["B_pat"] - rho["B_mat"]
        same, cross = direct, swapped
    elif orientation == "swapped":
        margin_a = rho["A_pat"] - rho["A_mat"]
        margin_b = rho["B_mat"] - rho["B_pat"]
        same, cross = swapped, direct
    elif orientation == "unresolved_tie":
        # Tie means the two assignment means are equal, not that each copy has zero preference.
        # Retain the direct-convention signed per-copy margins instead of erasing cancellation.
        margin_a = rho["A_mat"] - rho["A_pat"]
        margin_b = rho["B_pat"] - rho["B_mat"]
        same = cross = (direct + swapped) / 2.0
    else:
        raise RuntimeError("unsupported orientation: %s" % orientation)
    return {**rho, "direct": direct, "swapped": swapped, "same": same, "cross": cross,
            "contrast": same - cross, "orientation": orientation,
            "best_orientation": best, "margin_A": margin_a, "margin_B": margin_b,
            "min_margin": min(margin_a, margin_b)}


def macro(rows):
    fields = ("same", "cross", "contrast", "margin_A", "margin_B", "min_margin")
    return {field: float(np.mean([row[field] for row in rows])) for field in fields}


def baseline_orientations():
    result = {}
    frozen_rows = {}
    with TSV055.open("rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            if row["dataset"] == "Baseline-046-G-random":
                result[row["chr"]] = row["orientation"]
                frozen_rows[row["chr"]] = row
    if set(result) != set(CHROMOSOMES):
        raise RuntimeError("055 baseline orientation table incomplete")
    return result, frozen_rows


def candidate_inventory():
    candidates = []
    exp1 = json.loads((RUN / "states/experiment1/manifest.json").read_text(encoding="utf-8"))
    for row in exp1["states"]:
        candidates.append({"experiment": 1, "candidate_id": "exp1/" + row["state_id"], **row})
    nonref = json.loads((RUN / "results/experiment3_nonreference.json").read_text(encoding="utf-8"))
    for row in nonref["rows"]:
        fit_id = row["fit_id"]
        path = RUN / "states/experiment3" / fit_id / "1Mb/solver_state.npz"
        candidates.append({"experiment": 3, "kind": "endpoint", "candidate_id": "exp3/" + fit_id,
                           "fit_id": fit_id, "path": str(path.relative_to(ROOT)),
                           "file_sha256": row["solver_state_sha256"]})
    splices = json.loads((RUN / "states/experiment3_splices/manifest.json").read_text(encoding="utf-8"))
    for row in splices["states"]:
        candidates.append({"experiment": 3, "kind": "splice",
                           "candidate_id": "exp3/%s/%s" % (row["fit_id"], row["state_id"]), **row})
    return candidates


def load_coords(candidate):
    path = ROOT / candidate["path"]
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["coordinates"], dtype=np.float64).copy()


def main():
    pre = json.loads((RUN / "results/pre_reference_hash_gate.json").read_text(encoding="utf-8"))
    if (not pre.get("all_pending_states_hashed") or pre.get("reference_opened")
            or not pre.get("all_actual_artifact_hashes_matched") or pre.get("artifact_count") != 90):
        raise RuntimeError("nonreference hash gate missing")
    gate_artifact_deviations = []
    for item in pre["artifacts"]:
        actual = sha256_file(ROOT / item["path"])
        if actual != item["sha256"]:
            gate_artifact_deviations.append(item["path"])
    if gate_artifact_deviations:
        raise RuntimeError("candidate artifact changed after nonreference gate: %r" % gate_artifact_deviations)
    candidates = candidate_inventory()
    deviations = []
    for candidate in candidates:
        actual = sha256_file(ROOT / candidate["path"])
        if actual != candidate["file_sha256"]:
            deviations.append("candidate_hash:%s" % candidate["candidate_id"])
    fixed_hashes = {"mask": sha256_file(MASK), "reference": sha256_file(REFERENCE),
                    "baseline_npz": sha256_file(BASELINE)}
    if fixed_hashes != {"mask": MASK_SHA, "reference": REFERENCE_SHA, "baseline_npz": BASELINE_SHA}:
        deviations.append("frozen_input_hash_mismatch")
    gate = {"schema": "p9016-056-reference-hash-gate-v1", "reference_opened": False,
            "candidate_count": len(candidates), "candidate_hashes_checked": True,
            "all_90_candidate_state_mask_export_artifacts_rechecked": True,
            "nonreference_gate_sha256": sha256_file(RUN / "results/pre_reference_hash_gate.json"),
            "fixed_hashes": fixed_hashes, "deviations": deviations, "status": "PASS" if not deviations else "FAIL"}
    write_json(RUN / "reference_eval/pre_reference_hash_gate.json", gate)
    if deviations:
        raise RuntimeError("pre-reference hash gate failed")
    # Reference and mask are parsed only after the persisted gate above exists.
    reference = parse_3dg(REFERENCE)
    with np.load(MASK, allow_pickle=False) as payload:
        legacy = {key: np.asarray(payload[key]) for key in payload.files}
    data = data_io.load_aggregate(RUN / "inputs/G-original_1000000_aggregate.npz")
    baseline_map, frozen_rows = baseline_orientations()
    all_rows = []
    summaries = []
    endpoint_maps = {}
    # Endpoints before splices guarantees that each splice can reuse its endpoint mapping.
    candidates.sort(key=lambda row: (row["experiment"], 1 if row.get("kind") == "splice" else 0,
                                     row["candidate_id"]))
    for candidate in candidates:
        coords = load_coords(candidate)
        chromosome_rows = []
        if candidate["experiment"] == 1:
            fixed_map = dict(baseline_map)
            if candidate.get("kind") == "whole":
                chromosome = candidate["chromosome"]
                fixed_map[chromosome] = "swapped" if fixed_map[chromosome] == "direct" else "direct"
        elif candidate.get("kind") == "endpoint":
            fixed_map = None
        else:
            fixed_map = endpoint_maps[candidate["fit_id"]]
        for ci, chromosome in enumerate(CHROMOSOMES):
            key = "chr%d" % ci
            positions = np.asarray(legacy[key + "_positions"], dtype=np.int64)
            pair_i = np.asarray(legacy[key + "_pair_i"], dtype=np.int64)
            pair_j = np.asarray(legacy[key + "_pair_j"], dtype=np.int64)
            common = np.asarray(legacy[key + "_common"], dtype=bool)
            rho, n_pairs = rho_for(coords, data, ci, positions, pair_i, pair_j, common,
                                   reference, chromosome)
            row = metrics(rho, None if fixed_map is None else fixed_map[chromosome])
            row.update({"candidate_id": candidate["candidate_id"], "experiment": candidate["experiment"],
                        "kind": candidate.get("kind"), "fit_id": candidate.get("fit_id"),
                        "state_id": candidate.get("state_id"), "chromosome": chromosome,
                        "n_pairs": n_pairs})
            chromosome_rows.append(row); all_rows.append(row)
        if sum(row["n_pairs"] for row in chromosome_rows) != 157529:
            raise RuntimeError("055 common support is not 157529")
        if candidate["experiment"] == 3 and candidate.get("kind") == "endpoint":
            endpoint_maps[candidate["fit_id"]] = {row["chromosome"]: row["orientation"]
                                                    for row in chromosome_rows}
        summaries.append({"candidate_id": candidate["candidate_id"], "experiment": candidate["experiment"],
                          "kind": candidate.get("kind"), "fit_id": candidate.get("fit_id"),
                          "state_id": candidate.get("state_id"), "macro": macro(chromosome_rows),
                          "pairs_total": 157529,
                          "orientation_source": ("055_baseline_fixed_or_transported" if candidate["experiment"] == 1
                                                 else "endpoint_best_geometry" if candidate.get("kind") == "endpoint"
                                                 else "endpoint_fixed_reused")})
    # Frozen 055 baseline is reproduced before any interpretation.
    original = [row for row in all_rows if row["candidate_id"] == "exp1/original"]
    max_regression_error = 0.0
    for row in original:
        frozen = frozen_rows[row["chromosome"]]
        for key in ("same", "cross", "contrast"):
            max_regression_error = max(max_regression_error, abs(row[key] - float(frozen[key])))
    if max_regression_error >= 1e-10:
        raise RuntimeError("055 baseline reference metrics not reproduced")
    summary_by = {row["candidate_id"]: row for row in summaries}
    comparisons = []
    for seed in (560101, 560102):
        original_id = "exp3/G-original-seed%d" % seed
        offdiag_id = "exp3/G-offdiag-e-seed%d" % seed
        a, b = summary_by[original_id], summary_by[offdiag_id]
        comparisons.append({"seed": seed,
                            "matched_delta_offdiag_minus_original": b["macro"]["same"] - a["macro"]["same"],
                            "min_margin_delta_offdiag_minus_original": b["macro"]["min_margin"] - a["macro"]["min_margin"]})
    result = {"schema": "p9016-056-unified-reference-evaluation-v1", "status": "complete",
              "support": {"mask_sha256": MASK_SHA, "pairs_total": 157529,
                          "chromosomes": 20, "shared_across_all_candidates": True},
              "candidate_count": len(candidates), "summaries": summaries,
              "experiment3_comparisons": comparisons,
              "baseline_055_max_abs_regression_error": max_regression_error,
              "reference_opened_after_hash_gate": True, "phase_opened": False}
    write_json(RUN / "reference_eval/metrics.json", result)
    fields = ["candidate_id", "experiment", "kind", "fit_id", "state_id", "chromosome", "n_pairs",
              "A_mat", "A_pat", "B_mat", "B_pat", "direct", "swapped", "same", "cross", "contrast",
              "orientation", "best_orientation", "margin_A", "margin_B", "min_margin"]
    path = RUN / "reference_eval/per_chromosome.tsv"
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\t".join(fields) + "\n")
        for row in all_rows:
            handle.write("\t".join("" if row.get(field) is None else str(row.get(field)) for field in fields) + "\n")
    print(json.dumps({"status": "complete", "candidates": len(candidates),
                      "pairs_total_each": 157529, "baseline_regression_error": max_regression_error,
                      "experiment3_comparisons": comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
