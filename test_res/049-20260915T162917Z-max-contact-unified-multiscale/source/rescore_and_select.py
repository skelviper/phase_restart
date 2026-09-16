"""12 端点 + 2 initial + baseline 的共同 rescore 与预注册选择。

在打开 reference 之前写出（schema 与评价侧 `evaluation/results/REQUIRED_INPUTS.md` 一致）：
  results/endpoint_manifest_pre_reference.json   （顶层 fits / initials / baseline）
  results/selection_pre_reference.json           （顶层 per_loss.{A,B,C}）
  coords/initial-consensus/1Mb.{3dg,npz 引用}    （统一 15 dataset 回读）
  coords/initial-random/1Mb.3dg
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_core import load_layer, three_loss_components, whole_cell_rg
from frozen_imports import contact_model
from round_paths import (BASELINE_3DG, BASELINE_NPZ, INITIAL_CONSENSUS_1MB, INITIAL_RANDOM_1MB,
                         LOSSES, RUN, SOURCES, all_fit_ids)
from round_runner import array_sha256, jsonable, sha256_file

RESULTS = RUN / "results"
COORDS = RUN / "coords"
TIE_TOL = 1e-9
COUNT_KEY = {"A": "count_A", "B": "count_B", "C": "count_C"}
FULLJ_KEY = {"A": "fullJ_A", "B": "fullJ_B", "C": "fullJ_C"}
REL = RUN.parents[1]  # 仓库根


def read_endpoint(path: Path) -> dict:
    """读取端点坐标；缺 q 时由 p 派生并显式标注，绝不修改源文件。"""
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        raw_y = np.asarray(payload["raw_y"], dtype=np.float64).copy() if "raw_y" in keys else None
        p = float(np.asarray(payload["p"] if "p" in keys else payload["p_init"]).item())
        if "q" in keys:
            q = float(np.asarray(payload["q"]).item())
            q_source = "stored"
        else:
            q = float(contact_model.q_from_p(p))
            q_source = "derived_q_from_p"
    contact_model.assert_inside_unit_ball(coordinates)
    return {"coordinates": coordinates, "raw_y": raw_y, "p": p, "q": q, "q_source": q_source,
            "keys": sorted(keys)}


def rescore(data, entry: dict) -> dict:
    audit = three_loss_components(data, entry["coordinates"], entry["p"])
    return {"count_A": audit["count_A"], "count_B": audit["count_B"], "count_C": audit["count_C"],
            "count_nll_by_loss": audit["count_nll_by_loss"],
            "fullJ_A": audit["fullJ_A"], "fullJ_B": audit["fullJ_B"], "fullJ_C": audit["fullJ_C"],
            "fullJ_by_loss": audit["fullJ_by_loss"],
            "Zsum": audit["Zsum"], "Zmax": audit["Zmax"],
            "sum_C_log_gamma_max": audit["sum_C_log_gamma_max"],
            "B_minus_A_formula": audit["B_minus_A_formula"], "B_minus_A_actual": audit["B_minus_A_actual"],
            "C_minus_B_formula": audit["C_minus_B_formula"], "C_minus_B_actual": audit["C_minus_B_actual"],
            "regularization_total": audit["regularization_total"],
            "regularizers_raw": audit["regularizers_raw"],
            "regularizers_weighted": audit["regularizers_weighted"],
            "whole_cell_rg": whole_cell_rg(entry["coordinates"]),
            "p": audit["p"]}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REL))
    except ValueError:
        return str(path)


def main() -> int:
    data = load_layer(1_000_000)
    fits: list[dict] = []
    by_id: dict[str, dict] = {}

    for fit in all_fit_ids():
        loss, solver, source = fit.split("-")
        npz = COORDS / fit / "1Mb.npz"
        three_dg = COORDS / fit / "1Mb.3dg"
        canonical = COORDS / fit / "1Mb.canonical.npz"
        if not npz.exists() or not three_dg.exists():
            raise RuntimeError("missing endpoint artifact for %s" % fit)
        endpoint = read_endpoint(npz)
        record = json.loads((RUN / "stages" / fit / "1Mb.json").read_text(encoding="utf-8"))
        row = {
            "fit_id": fit, "kind": "fit", "loss": loss, "solver": solver, "source": source,
            "coords_npz_path": rel(npz), "coords_npz_sha256": sha256_file(npz),
            "three_dg_path": rel(three_dg), "three_dg_sha256": sha256_file(three_dg),
            "canonical_npz_path": rel(canonical) if canonical.exists() else None,
            "canonical_npz_sha256": sha256_file(canonical) if canonical.exists() else None,
            "raw_y_array_sha256": array_sha256(endpoint["raw_y"]) if endpoint["raw_y"] is not None else None,
            "coordinate_array_sha256": array_sha256(endpoint["coordinates"]),
            "terminal": record.get("status"), "outerFG": record.get("outer_fg_actual"),
            "terminal_reason": record.get("terminal_reason"), "fg_cap": record.get("fg_cap"),
            "fit_wall_seconds": record.get("fit_wall_seconds"),
            "canonical_gradient_max_abs": (record.get("endpoint") or {}).get("canonical_gradient_max_abs"),
            "p": endpoint["p"], "q": endpoint["q"], "q_source": endpoint["q_source"],
            "npz_keys": endpoint["keys"],
        }
        row.update(rescore(data, endpoint))
        row["fullJ"] = row[FULLJ_KEY[loss]]
        fits.append(row)
        by_id[fit] = row

    initials: list[dict] = []
    for source, npz_source in (("consensus", INITIAL_CONSENSUS_1MB), ("random", INITIAL_RANDOM_1MB)):
        label = "initial-%s" % source
        endpoint = read_endpoint(npz_source)
        target_dir = COORDS / label
        target_dir.mkdir(parents=True, exist_ok=True)
        three_dg = target_dir / "1Mb.3dg"
        contact_model.write_full_tracks(three_dg, data, endpoint["coordinates"])
        readback = _read_tracks(three_dg, data)
        if not np.array_equal(readback, endpoint["coordinates"]):
            raise RuntimeError("initial 3DG readback mismatch for %s" % label)
        row = {
            "fit_id": label, "endpoint_id": label, "kind": "initial_control", "source": source,
            "coords_npz_path": rel(npz_source), "coords_npz_sha256": sha256_file(npz_source),
            "three_dg_path": rel(three_dg), "three_dg_sha256": sha256_file(three_dg),
            "three_dg_readback_equal": True,
            "coordinate_array_sha256": array_sha256(endpoint["coordinates"]),
            "raw_y_array_sha256": array_sha256(endpoint["raw_y"]) if endpoint["raw_y"] is not None else None,
            "terminal": "zero_optimization_initial_control", "outerFG": 0,
            "p": endpoint["p"], "q": endpoint["q"], "q_source": endpoint["q_source"],
            "npz_keys": endpoint["keys"],
            "lineage": "014 approved blind root plus zero-optimization prolongation",
        }
        row.update(rescore(data, endpoint))
        row["fullJ"] = row["fullJ_A"]
        initials.append(row)

    baseline_entry = read_endpoint(BASELINE_NPZ)
    baseline = {
        "fit_id": "baseline-046-G-full-J", "endpoint_id": "baseline-046-G-full-J",
        "kind": "frozen_historical_baseline",
        "loss": "A", "solver": "raw", "source": "random",
        "coords_npz_path": rel(BASELINE_NPZ), "coords_npz_sha256": sha256_file(BASELINE_NPZ),
        "three_dg_path": rel(BASELINE_3DG), "three_dg_sha256": sha256_file(BASELINE_3DG),
        "coordinate_array_sha256": array_sha256(baseline_entry["coordinates"]),
        "raw_y_array_sha256": array_sha256(baseline_entry["raw_y"]) if baseline_entry["raw_y"] is not None else None,
        "terminal": "budget_not_converged", "outerFG": 1988, "terminal_reason": "fg_budget_exhausted",
        "p": baseline_entry["p"], "q": baseline_entry["q"], "q_source": baseline_entry["q_source"],
        "npz_keys": baseline_entry["keys"],
        "cost_note": "1988 FG include 5Mb/2Mb stages; this round uses 1502 all-1Mb FG per fit",
    }
    baseline.update(rescore(data, baseline_entry))
    baseline["fullJ"] = baseline["fullJ_A"]

    manifest = {"schema": "p9016-max-contact-endpoint-manifest-pre-reference-v2",
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "run_id": RUN.name, "fit_count": len(fits),
                "fits": fits, "initials": initials, "baseline": baseline,
                "rescore_definition": {
                    "count_nll_by_loss": "all three count objectives at the same endpoint coordinates and p",
                    "count_A": "original marginal/G count objective (common rescore)",
                    "fullJ": "own-loss count + the shared weighted regularizers",
                    "identity_B_minus_A": "count_B - count_A = sum_C C*(-log gamma_max)/Nraw >= 0",
                    "identity_C_minus_B": "count_C - count_B = (Noff/Nraw)*log(Zmax/Zsum) <= 0",
                    "regularization": "weighted: 1*bond + 1*repulsion + 0.01*bend + 1*p_prior",
                    "fg_accounting": "18,024 FG is the formal 12-fit matrix upper bound / actual (12 x 1502) and "
                                     "excludes engineering gates, the discarded launch-1 attempt and diagnostic rescores",
                },
                "reference_opened": False, "phase_opened": False}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "endpoint_manifest_pre_reference.json").write_text(
        json.dumps(jsonable(manifest), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")

    # ---- selection（contract: per_loss.{A,B,C}.source_selection.{raw,ms}.selected_source）
    per_loss: dict[str, dict] = {}
    for loss in LOSSES:
        key = COUNT_KEY[loss]
        source_selection: dict[str, dict] = {}
        for solver in ("raw", "ms"):
            candidates = [by_id["%s-%s-%s" % (loss, solver, source)] for source in SOURCES]
            rows = [{"source": row["source"], "fit_id": row["fit_id"],
                     "own_count": row[key], "own_fullJ": row[FULLJ_KEY[loss]],
                     "count_nll_by_loss": row["count_nll_by_loss"],
                     "coords_npz_path": row["coords_npz_path"],
                     "coords_npz_sha256": row["coords_npz_sha256"]} for row in candidates]
            consensus = next(row for row in rows if row["source"] == "consensus")
            random_row = next(row for row in rows if row["source"] == "random")
            margin = float(consensus["own_count"]) - float(random_row["own_count"])
            if abs(margin) <= TIE_TOL:
                selected, rule = "consensus", "own final count objective; |delta| <= 1e-9 -> consensus"
            else:
                selected = "consensus" if margin < 0 else "random"
                rule = "minimum own final count objective"
            source_selection[solver] = {"selected_source": selected, "rule": rule, "tie_tolerance": TIE_TOL,
                                        "margin_consensus_minus_random": margin,
                                        "selected_fit_id": "%s-%s-%s" % (loss, solver, selected),
                                        "candidates": rows}
        four = [row for solver in ("raw", "ms") for row in source_selection[solver]["candidates"]]
        best = min(four, key=lambda row: float(row["own_count"]))
        per_loss[loss] = {
            "loss_name": {"A": "marginal_G", "B": "hard_observed", "C": "max_rate"}[loss],
            "source_selection": source_selection,
            "display_endpoint": {
                "solver": best["fit_id"].split("-")[1], "source": best["source"], "fit_id": best["fit_id"],
                "count": best["own_count"], "coordsSHA": best["coords_npz_sha256"],
                "coords_npz_path": best["coords_npz_path"],
                "selection": "minimum own count objective over the four endpoints of this loss, frozen before the "
                             "reference is opened"},
            "cross_loss_warning": "A/B/C are different objectives and must not be ranked against each other; use the "
                                  "common count_A / fullJ_A rescore for interpretation",
        }

    selection = {"schema": "p9016-max-contact-selection-pre-reference-v2",
                 "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "manifest": "endpoint_manifest_pre_reference.json",
                 "per_loss": per_loss,
                 "baseline": {"fit_id": "baseline-046-G-full-J", "endpoint_id": "baseline-046-G-full-J",
                              "count_A": baseline["count_A"], "count_B": baseline["count_B"],
                              "count_C": baseline["count_C"], "fullJ": baseline["fullJ"],
                              "coords_npz_path": baseline["coords_npz_path"],
                              "coords_npz_sha256": baseline["coords_npz_sha256"]},
                 "initials": [{"fit_id": row["fit_id"], "endpoint_id": row["endpoint_id"], "count_A": row["count_A"],
                               "count_B": row["count_B"], "count_C": row["count_C"], "fullJ": row["fullJ"],
                               "coords_npz_path": row["coords_npz_path"],
                               "coords_npz_sha256": row["coords_npz_sha256"],
                               "three_dg_path": row["three_dg_path"],
                               "three_dg_sha256": row["three_dg_sha256"]} for row in initials],
                 "reference_opened": False, "phase_opened": False}
    (RESULTS / "selection_pre_reference.json").write_text(
        json.dumps(jsonable(selection), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"status": "ok", "fits": len(fits), "initials": len(initials),
                      "display": {loss: per_loss[loss]["display_endpoint"]["fit_id"] for loss in LOSSES},
                      "selected": {loss: {solver: per_loss[loss]["source_selection"][solver]["selected_source"]
                                          for solver in ("raw", "ms")} for loss in LOSSES}}))
    return 0


def _read_tracks(path: Path, data) -> np.ndarray:
    from frozen_imports import formal_controller
    return formal_controller._read_full_tracks(path, data)


if __name__ == "__main__":
    raise SystemExit(main())
