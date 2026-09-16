"""运行056实验1/2的无reference正式诊断与硬门。"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nonref_core import (array_sha256, data_io, diag_variant_data,
                         fixed_state_fullgrid, jsonable, physical_count_cell,
                         strict_pointset_equal)
from pr import contact_model
from pr.solver_state import sha256_file

BASELINE = ROOT / "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.npz"
AGGREGATE = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/inputs/real_1000000_aggregate.npz"
BASELINE_SHA = "6bb93bf570cf688fd8b02dccf14ca0867c1f831d5e123036ced626856f8d8752"
AGGREGATE_SHA = "80984d804f8ae0552f6bab34a77a03e778073ac87f3b3ec6f04f96de74137420"
ATOL = 1e-10
SPLICES = {"chr1": (65, 130), "chr8": (43, 86), "chr19": (20, 41), "chrX": (57, 114)}
RANDOM_U_SEEDS = tuple(range(450500, 450516))
DIAG_SHIFTS = (65, 61, 53, 52, 50, 50, 48, 43, 41, 43, 41, 40, 40, 41, 35, 33, 31, 30, 20, 57)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_coordinates(state_id: str, coordinates: np.ndarray, audit: dict) -> dict:
    path = RUN / "states" / "experiment1" / (state_id + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError("refusing to overwrite state %s" % path)
    with path.open("xb") as handle:
        np.savez_compressed(handle, coordinates=np.asarray(coordinates, dtype=np.float64))
    return {"state_id": state_id, "path": str(path.relative_to(ROOT)),
            "file_sha256": sha256_file(path),
            "coordinate_sha256": array_sha256(coordinates), **audit}


def swap_whole(coords, data, chromosome):
    result = np.asarray(coords, dtype=np.float64).copy()
    slc = data.chromosome_slice(chromosome)
    result[:, slc] = result[::-1, slc]
    return result


def swap_suffix(coords, data, chromosome, start):
    result = np.asarray(coords, dtype=np.float64).copy()
    slc = data.chromosome_slice(chromosome)
    suffix = slice(slc.start + int(start), slc.stop)
    result[:, suffix] = result[::-1, suffix]
    return result


def make_u0(coords):
    z = (coords[0] + coords[1]) / 2.0
    raw = np.stack((z, z), axis=0)
    pre = float(np.linalg.norm(raw, axis=2).max())
    scale = 1.0 if pre < 1.0 else (1.0 - 1e-6) / pre
    return raw * scale, {"kind": "u0", "global_scale": scale, "pre_scale_radius": pre}


def make_random_u(coords, data, seed):
    z = (coords[0] + coords[1]) / 2.0
    u = (coords[0] - coords[1]) / 2.0
    permuted = np.zeros_like(u)
    rng = np.random.default_rng(int(seed))
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        permuted[slc] = u[slc][rng.permutation(slc.stop - slc.start)]
    raw = np.stack((z + permuted, z - permuted), axis=0)
    pre = float(np.linalg.norm(raw, axis=2).max())
    scale = 1.0 if pre < 1.0 else (1.0 - 1e-6) / pre
    return raw * scale, {"kind": "random_u", "seed": int(seed),
                         "global_scale": scale, "pre_scale_radius": pre}


def differences(record, baseline, arrays, baseline_arrays):
    fields = ("total_normalized", "count_nll_per_raw_record",
              "offdiag_data_nat_per_contact", "Zinter")
    out = {field + "_abs": abs(float(record[field]) - float(baseline[field])) for field in fields}
    for group in ("penalties_raw", "penalties_weighted"):
        for key in baseline[group]:
            out[group + ":" + key + "_abs"] = abs(float(record[group][key]) - float(baseline[group][key]))
    for key in baseline_arrays:
        out[key + "_max_abs"] = float(np.max(np.abs(arrays[key] - baseline_arrays[key])))
    out["Zinter_relative"] = out["Zinter_abs"] / abs(float(baseline["Zinter"]))
    return out


def experiment1(data, coords, p):
    states = [("original", coords, {"kind": "original", "strict": True})]
    for ci, name in enumerate(data.chromosome_names):
        states.append(("whole-%s" % name, swap_whole(coords, data, ci),
                       {"kind": "whole", "chromosome": str(name), "strict": True}))
    name_to_index = {str(name): index for index, name in enumerate(data.chromosome_names)}
    for name, starts in SPLICES.items():
        for start in starts:
            states.append(("splice-%s-%03d" % (name, start),
                           swap_suffix(coords, data, name_to_index[name], start),
                           {"kind": "splice", "chromosome": name, "splice_index": start,
                            "splice_bp": int(start * data.bin_size), "strict": True}))
    u0, u0_meta = make_u0(coords)
    states.append(("u0", u0, {**u0_meta, "strict": False}))
    for seed in RANDOM_U_SEEDS:
        random_u, metadata = make_random_u(coords, data, seed)
        states.append(("random-u-%d" % seed, random_u, {**metadata, "strict": False}))
    if len(states) != 46:
        raise RuntimeError("experiment1 state count changed")
    rows = []
    state_manifest = []
    baseline_record = None
    baseline_arrays = None
    hard_failures = []
    for index, (state_id, state_coords, metadata) in enumerate(states):
        pointset = strict_pointset_equal(coords, state_coords) if metadata["strict"] else None
        if metadata["strict"] and not pointset:
            hard_failures.append({"state_id": state_id, "field": "pointset", "value": False})
        state_manifest.append(save_coordinates(state_id, state_coords, {**metadata,
                                           "pointset_conserved": pointset}))
        record, arrays = fixed_state_fullgrid(data, state_coords, p, keep_inter_arrays=True)
        row = {"state_index": index, "state_id": state_id, **metadata,
               "pointset_conserved": pointset, **record}
        if baseline_record is None:
            baseline_record, baseline_arrays = record, arrays
            row["differences_from_original"] = {"all_zero_by_definition": True}
        else:
            diff = differences(record, baseline_record, arrays, baseline_arrays)
            row["differences_from_original"] = diff
            if metadata["strict"]:
                for field in ("inter_mixture_rate_max_abs", "inter_normalized_rate_max_abs",
                              "inter_sorted_four_distances_max_abs", "Zinter_abs"):
                    if not float(diff[field]) < ATOL:
                        hard_failures.append({"state_id": state_id, "field": field,
                                              "value": diff[field], "threshold": ATOL})
            if metadata["kind"] == "whole":
                full_fields = ["total_normalized_abs", "count_nll_per_raw_record_abs",
                               "offdiag_data_nat_per_contact_abs"]
                full_fields += [key for key in diff if key.startswith("penalties_")]
                for field in full_fields:
                    if not float(diff[field]) < ATOL:
                        hard_failures.append({"state_id": state_id, "field": field,
                                              "value": diff[field], "threshold": ATOL})
        rows.append(row)
        print(json.dumps({"event": "experiment1_state", "index": index + 1,
                          "total": len(states), "state_id": state_id}, sort_keys=True), flush=True)
    splice_delta = [float(row["offdiag_data_nat_per_contact"] - baseline_record["offdiag_data_nat_per_contact"])
                    for row in rows if row["kind"] == "splice"]
    science = {"positive_count": int(np.count_nonzero(np.asarray(splice_delta) > 0.0)),
               "median_delta_nat_per_offdiag": float(np.median(splice_delta)),
               "deltas": splice_delta}
    science["pass"] = science["positive_count"] >= 7 and science["median_delta_nat_per_offdiag"] >= 0.001
    result = {
        "schema": "p9016-056-experiment1-v1", "status": "PASS" if not hard_failures else "FAIL",
        "state_count": len(rows), "full_grid_evaluations": len(rows), "cap": 48,
        "hard_invariance_atol": ATOL, "hard_failures": hard_failures,
        "scientific_gate": science, "rows": rows,
        "reference_opened": False, "phase_opened": False,
    }
    write_json(RUN / "results/experiment1.json", result)
    write_json(RUN / "states/experiment1/manifest.json", {"states": state_manifest})
    return result


def shifted_diag(data):
    result = np.asarray(data.diag_counts, dtype=np.int64).copy()
    for ci, shift in enumerate(DIAG_SHIFTS):
        slc = data.chromosome_slice(ci)
        result[slc] = np.roll(result[slc], int(shift))
    return result


def exposure_from(diag, offdiag_endpoints):
    values = np.sqrt(np.asarray(offdiag_endpoints, dtype=np.float64) + 2.0 * np.asarray(diag) + 10.0)
    return values / values.mean()


def experiment2(data, coords, p):
    original_diag = np.asarray(data.diag_counts, dtype=np.int64)
    variants = {"original": original_diag.copy(), "shift": shifted_diag(data),
                "doubled": 2 * original_diag}
    offdiag_endpoints = np.asarray(data.endpoint_counts, dtype=np.int64) - 2 * original_diag
    original_exposure = np.asarray(data.exposure, dtype=np.float64)
    rows = []
    gradients = {}
    for diag_name, diag in variants.items():
        exposure_modes = {
            "locked_original": original_exposure,
            "production_recomputed": exposure_from(diag, offdiag_endpoints),
        }
        for exposure_name, exposure in exposure_modes.items():
            variant_data = diag_variant_data(data, diag, exposure)
            for denominator_name, denominator in (("Nraw", float(variant_data.raw_records)),
                                                   ("Noff", float(data.raw_cis_offdiag + data.raw_inter))):
                key = "%s__%s__%s" % (diag_name, exposure_name, denominator_name)
                record, gradient = physical_count_cell(variant_data, coords, p, denominator)
                gradients[key] = gradient
                rows.append({"cell_id": key, "diag": diag_name, "exposure": exposure_name,
                             "normalizer": denominator_name, "diag_total": int(diag.sum()),
                             "exposure_sha256": array_sha256(exposure), **record})
                print(json.dumps({"event": "experiment2_cell", "cell_id": key,
                                  "index": len(rows), "total": 12}, sort_keys=True), flush=True)
    if len(rows) != 12:
        raise RuntimeError("experiment2 cell count changed")
    np.savez_compressed(RUN / "results/experiment2_gradients.npz", **gradients)
    locked_error = float(np.max(np.abs(
        gradients["shift__locked_original__Nraw"] - gradients["original__locked_original__Nraw"])))
    scaling = []
    by_id = {row["cell_id"]: row for row in rows}
    for diag_name in variants:
        for exposure_name in ("locked_original", "production_recomputed"):
            raw = by_id["%s__%s__Nraw" % (diag_name, exposure_name)]
            off = by_id["%s__%s__Noff" % (diag_name, exposure_name)]
            ratio = raw["native_Nraw"] / off["denominator"]
            gradient_error = float(np.max(np.abs(
                gradients["%s__%s__Noff" % (diag_name, exposure_name)]
                - gradients["%s__%s__Nraw" % (diag_name, exposure_name)] * ratio)))
            value_error = abs(float(off["count_value"]) - float(raw["count_value"]) * ratio)
            conditional_error = abs(float(off["conditional_value"]) - float(raw["conditional_value"]) * ratio)
            scaling.append({"diag": diag_name, "exposure": exposure_name, "ratio": ratio,
                            "gradient_max_abs_error": gradient_error,
                            "value_abs_error": value_error,
                            "conditional_abs_error": conditional_error})
    g0 = gradients["original__production_recomputed__Nraw"]
    gs = gradients["shift__production_recomputed__Nraw"]
    gd = gradients["doubled__production_recomputed__Noff"]
    gd0 = gradients["original__production_recomputed__Noff"]
    sensitivity = {
        "primary_shift_production_Nraw_L2_relative": float(np.linalg.norm(gs - g0) / max(np.linalg.norm(g0), 1e-300)),
        "primary_shift_production_Nraw_Linf_relative": float(np.max(np.abs(gs - g0)) / max(np.max(np.abs(g0)), 1e-300)),
        "secondary_doubled_production_Noff_L2_relative": float(np.linalg.norm(gd - gd0) / max(np.linalg.norm(gd0), 1e-300)),
    }
    primary = sensitivity["primary_shift_production_Nraw_L2_relative"]
    sensitivity["interpretation"] = "high_priority" if primary > 0.10 else ("low_priority" if primary < 0.01 else "uncertain")
    hard_failures = []
    if not locked_error < ATOL:
        hard_failures.append({"field": "locked_shift_gradient_max_abs", "value": locked_error, "threshold": ATOL})
    for row in scaling:
        for field in ("gradient_max_abs_error", "value_abs_error", "conditional_abs_error"):
            if not float(row[field]) < ATOL:
                hard_failures.append({"cell": {"diag": row["diag"], "exposure": row["exposure"]},
                                      "field": field, "value": row[field], "threshold": ATOL})
    result = {
        "schema": "p9016-056-experiment2-v1", "status": "PASS" if not hard_failures else "FAIL",
        "cells": len(rows), "value_gradient_evaluations": len(rows),
        "objective_fg_calls": int(sum(row["objective_fg_calls"] for row in rows)),
        "pair_kernel_passes": int(sum(row["pair_kernel_passes"] for row in rows)),
        "locked_shift_gradient_max_abs": locked_error, "normalizer_scaling": scaling,
        "sensitivity": sensitivity, "hard_invariance_atol": ATOL,
        "hard_failures": hard_failures, "rows": rows,
        "reference_opened": False, "phase_opened": False,
    }
    write_json(RUN / "results/experiment2.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("1", "2", "both"), default="both")
    args = parser.parse_args()
    if sha256_file(BASELINE) != BASELINE_SHA or sha256_file(AGGREGATE) != AGGREGATE_SHA:
        raise RuntimeError("frozen baseline/aggregate SHA mismatch")
    data = data_io.load_aggregate(AGGREGATE)
    with np.load(BASELINE, allow_pickle=False) as payload:
        coords = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        p = float(np.asarray(payload["p"]).item())
    terminal = {"started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "reference_opened": False, "phase_opened": False, "status": "running"}
    write_json(RUN / "logs/nonref_terminal.json", terminal)
    try:
        if args.experiment in ("1", "both"):
            e1 = experiment1(data, coords, p)
            if e1["status"] != "PASS":
                raise RuntimeError("experiment1 hard invariance gate failed")
        if args.experiment in ("2", "both"):
            e2 = experiment2(data, coords, p)
            if e2["status"] != "PASS":
                raise RuntimeError("experiment2 hard invariance gate failed")
        terminal.update({"status": "complete", "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        write_json(RUN / "logs/nonref_terminal.json", terminal)
        return 0
    except Exception as error:
        terminal.update({"status": "failure", "error_type": type(error).__name__, "error": str(error),
                         "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        write_json(RUN / "logs/nonref_terminal.json", terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
