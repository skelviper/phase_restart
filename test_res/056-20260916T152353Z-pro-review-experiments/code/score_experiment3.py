"""056实验3无reference terminal/test/splice单forward评分与辅助sweep账本。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
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
from nonref_core import fixed_state_fullgrid, jsonable  # noqa: E402
from pr import contact_model  # noqa: E402
from pr.solver_state import load_solver_state, sha256_file  # noqa: E402

CONDITIONS = ("G-original", "G-offdiag-e")
SEEDS = (560101, 560102)
BIN_SIZES = (5_000_000, 2_000_000, 1_000_000)
SPLICES = {"chr1": (65, 130), "chr8": (43, 86), "chr19": (20, 41), "chrX": (57, 114)}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def array_hash(array):
    return hashlib.sha256(np.asarray(array, order="C").tobytes(order="C")).hexdigest()


def test_aggregates():
    path = RUN / "inputs/test_contacts.npz"
    with np.load(path, allow_pickle=False) as payload:
        ci = np.asarray(payload["ci"], dtype=np.int64)
        p1 = np.asarray(payload["p1"], dtype=np.int64)
        cj = np.asarray(payload["cj"], dtype=np.int64)
        p2 = np.asarray(payload["p2"], dtype=np.int64)
    train_template = data_io.load_aggregate(RUN / "inputs/G-original_1000000_aggregate.npz")
    outputs = {}
    budgets = []
    for bin_size in BIN_SIZES:
        data = contact_model.aggregate_from_arrays(
            tuple(train_template.chromosome_names), tuple(int(x) for x in train_template.chromosome_lengths),
            ci, p1, cj, p2, bin_size)
        outputs[bin_size] = data
        budgets.append({"bin_size": bin_size, **data.budget()})
    write_json(RUN / "results/split_resolution_budgets.json", {
        "train": [{"condition": "G-original", "bin_size": bin_size,
                   **data_io.load_aggregate(RUN / "inputs" / ("G-original_%d_aggregate.npz" % bin_size)).budget()}
                  for bin_size in BIN_SIZES],
        "test": budgets,
        "same_record_fold_reaggregated_each_resolution": True,
    })
    return outputs


def swap_suffix(coords, data, chromosome, start):
    result = np.asarray(coords, dtype=np.float64).copy()
    slc = data.chromosome_slice(chromosome)
    suffix = slice(slc.start + int(start), slc.stop)
    result[:, suffix] = result[::-1, suffix]
    return result


def save_splice(fit_id, state_id, coords, metadata):
    path = RUN / "states/experiment3_splices" / fit_id / (state_id + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError("refusing to overwrite splice state")
    with path.open("xb") as handle:
        np.savez_compressed(handle, coordinates=np.asarray(coords, dtype=np.float64))
    return {"fit_id": fit_id, "state_id": state_id, "path": str(path.relative_to(ROOT)),
            "file_sha256": sha256_file(path), "coordinate_sha256": array_hash(coords), **metadata}


def main():
    terminal = json.loads((RUN / "logs/training_terminal.json").read_text(encoding="utf-8"))
    if terminal.get("status") != "complete":
        raise RuntimeError("training is not complete")
    test_by_bin = test_aggregates()
    test_base = test_by_bin[1_000_000]
    rows = []
    splice_manifest = []
    ledger = []
    name_to_index = None
    for seed in SEEDS:
        for condition in CONDITIONS:
            fit_id = "%s-seed%d" % (condition, seed)
            train = data_io.load_aggregate(RUN / "inputs" / ("%s_1000000_aggregate.npz" % condition))
            if name_to_index is None:
                name_to_index = {str(name): index for index, name in enumerate(train.chromosome_names)}
            state_path = RUN / "states/experiment3" / fit_id / "1Mb/solver_state.npz"
            state = load_solver_state(state_path)
            coords, p = state["coordinates"], state["p"]
            train_record, _ = fixed_state_fullgrid(
                train, coords, p, keep_inter_arrays=False, include_penalties=False)
            ledger.append({"category": "terminal_train_readback", "fit_id": fit_id,
                           "eligible_contact_pair_forward_sweeps": 1,
                           "regularizer_full_pair_sweeps": 0})
            stage = json.loads((RUN / "stages" / fit_id / "1Mb.json").read_text(encoding="utf-8"))
            expected_count = float(stage["endpoint"]["components"]["count_nll_normalized"])
            count_error = abs(float(train_record["count_nll_per_raw_record"]) - expected_count)
            if count_error >= 1e-10:
                raise RuntimeError("terminal train count readback mismatch: %s %.3e" % (fit_id, count_error))
            test_data = replace(test_base, exposure=np.asarray(train.exposure, dtype=np.float64).copy(),
                                exposure_mode="train_fixed_%s" % condition)
            test_data.assert_consistent()
            test_record, _ = fixed_state_fullgrid(
                test_data, coords, p, keep_inter_arrays=False, include_penalties=False)
            ledger.append({"category": "heldout_test_score", "fit_id": fit_id,
                           "eligible_contact_pair_forward_sweeps": 1,
                           "regularizer_full_pair_sweeps": 0})
            splice_rows = []
            for chromosome, starts in SPLICES.items():
                for start in starts:
                    state_id = "splice-%s-%03d" % (chromosome, start)
                    changed = swap_suffix(coords, train, name_to_index[chromosome], start)
                    pointwise = bool(np.all(
                        (np.all(changed[0] == coords[0], axis=1) & np.all(changed[1] == coords[1], axis=1))
                        | (np.all(changed[0] == coords[1], axis=1) & np.all(changed[1] == coords[0], axis=1))))
                    if not pointwise:
                        raise RuntimeError("splice did not conserve unordered point sets")
                    manifest_row = save_splice(fit_id, state_id, changed, {
                        "chromosome": chromosome, "splice_index": start,
                        "splice_bp": int(start * train.bin_size), "pointset_conserved": True})
                    splice_manifest.append(manifest_row)
                    score, _ = fixed_state_fullgrid(
                        train, changed, p, keep_inter_arrays=False, include_penalties=False)
                    ledger.append({"category": "endpoint_splice_score", "fit_id": fit_id,
                                   "state_id": state_id,
                                   "eligible_contact_pair_forward_sweeps": 1,
                                   "regularizer_full_pair_sweeps": 0})
                    delta = float(score["offdiag_data_nat_per_contact"]
                                  - train_record["offdiag_data_nat_per_contact"])
                    splice_rows.append({"state_id": state_id, "chromosome": chromosome,
                                        "splice_index": start, "delta_nat_per_offdiag": delta,
                                        "score": score})
            deltas = np.asarray([row["delta_nat_per_offdiag"] for row in splice_rows])
            rows.append({
                "fit_id": fit_id, "condition": condition, "seed": seed,
                "solver_state_sha256": state["sha256"],
                "terminal_count_readback_abs_error": count_error,
                "train": train_record,
                "test_primary_nll_nat_per_offdiag": float(test_record["offdiag_data_no_K0_nat_per_contact"]),
                "test_with_K0_nll_nat_per_offdiag": float(test_record["offdiag_data_nat_per_contact"]),
                "test_Noff": int(test_data.raw_cis_offdiag + test_data.raw_inter),
                "test": test_record,
                "splices": splice_rows,
                "splice_positive_count": int(np.count_nonzero(deltas > 0.0)),
                "splice_median_delta_nat_per_offdiag": float(np.median(deltas)),
            })
            print(json.dumps({"event": "056_aux_endpoint_complete", "fit_id": fit_id,
                              "splice_positive_count": int(np.count_nonzero(deltas > 0.0)),
                              "splice_median": float(np.median(deltas))}, sort_keys=True), flush=True)
    by = {(row["seed"], row["condition"]): row for row in rows}
    gains = []
    for seed in SEEDS:
        original = by[(seed, "G-original")]
        offdiag = by[(seed, "G-offdiag-e")]
        gains.append({"seed": seed,
                      "heldout_gain_original_minus_offdiag_nat_per_contact":
                          float(original["test_primary_nll_nat_per_offdiag"]
                                - offdiag["test_primary_nll_nat_per_offdiag"])})
    total_contact_sweeps = int(sum(item["eligible_contact_pair_forward_sweeps"] for item in ledger))
    total_regularizer_sweeps = int(sum(item["regularizer_full_pair_sweeps"] for item in ledger))
    if total_contact_sweeps != 40 or total_regularizer_sweeps != 0:
        raise RuntimeError("normal-path auxiliary sweep ledger is not 4+4+32=40")
    if total_contact_sweeps + total_regularizer_sweeps > 64:
        raise RuntimeError("auxiliary full-pair sweep cap exceeded")
    result = {
        "schema": "p9016-056-experiment3-nonreference-score-v1",
        "status": "complete", "rows": rows, "heldout_gains": gains,
        "auxiliary_ledger": ledger,
        "auxiliary_eligible_contact_pair_forward_sweeps": total_contact_sweeps,
        "auxiliary_regularizer_full_pair_sweeps": total_regularizer_sweeps,
        "auxiliary_total_full_pair_sweeps": total_contact_sweeps + total_regularizer_sweeps,
        "auxiliary_cap": 64, "training_fg_excluded_from_auxiliary": True,
        "stage_cpu_initial_checks": 12,
        "stage_cpu_initial_check_full_pair_sweeps": 0,
        "normal_path_decomposition": "4 terminal train readbacks + 4 held-out tests + 32 endpoint splices",
        "reference_opened": False, "phase_opened": False,
    }
    write_json(RUN / "results/experiment3_nonreference.json", result)
    write_json(RUN / "states/experiment3_splices/manifest.json", {"states": splice_manifest})
    artifacts = []
    exp1_manifest_path = RUN / "states/experiment1/manifest.json"
    exp1_manifest = json.loads(exp1_manifest_path.read_text(encoding="utf-8"))
    for item in exp1_manifest["states"]:
        path = ROOT / item["path"]
        actual = sha256_file(path)
        if actual != item["file_sha256"]:
            raise RuntimeError("experiment1 state hash changed: %s" % item["state_id"])
        artifacts.append({"role": "experiment1_state", "state_id": item["state_id"],
                          "path": item["path"], "sha256": actual})
    for row in rows:
        fit_id = row["fit_id"]
        artifact_path = RUN / "states/experiment3" / fit_id / "1Mb/artifacts.json"
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        for role, key in (("experiment3_solver_state", "solver_state"),
                          ("experiment3_presence_mask", "presence_mask"),
                          ("experiment3_export", "export")):
            expected = payload[key]["sha256"]
            path = Path(payload[key]["path"])
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError("experiment3 endpoint artifact hash changed: %s/%s" % (fit_id, key))
            artifacts.append({"role": role, "fit_id": fit_id,
                              "path": str(path.relative_to(ROOT)), "sha256": actual})
    for item in splice_manifest:
        path = ROOT / item["path"]
        actual = sha256_file(path)
        if actual != item["file_sha256"]:
            raise RuntimeError("experiment3 splice state hash changed: %s/%s" %
                               (item["fit_id"], item["state_id"]))
        artifacts.append({"role": "experiment3_splice_state", "fit_id": item["fit_id"],
                          "state_id": item["state_id"], "path": item["path"], "sha256": actual})
    pre_reference = {
        "schema": "p9016-056-all-candidate-artifact-hash-gate-v2",
        "experiment1_manifest": {"path": str(exp1_manifest_path.relative_to(ROOT)),
                                 "sha256": sha256_file(exp1_manifest_path)},
        "experiment3_splice_manifest": {
            "path": str((RUN / "states/experiment3_splices/manifest.json").relative_to(ROOT)),
            "sha256": sha256_file(RUN / "states/experiment3_splices/manifest.json")},
        "artifacts": artifacts, "artifact_count": len(artifacts),
        "expected_decomposition": {"experiment1_states": 46, "experiment3_solver_states": 4,
                                   "experiment3_presence_masks": 4, "experiment3_exports": 4,
                                   "experiment3_splice_states": 32},
        "all_actual_artifact_hashes_matched": True,
        "all_pending_states_hashed": True, "reference_opened": False,
    }
    if len(artifacts) != 90:
        raise RuntimeError("pre-reference artifact inventory is not 46+12+32=90")
    write_json(RUN / "results/pre_reference_hash_gate.json", pre_reference)
    print(json.dumps({"status": "complete", "auxiliary_sweeps": total_contact_sweeps,
                      "gains": gains}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
