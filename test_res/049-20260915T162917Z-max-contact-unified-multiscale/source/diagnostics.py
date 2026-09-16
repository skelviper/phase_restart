"""049 PLAN §7 三项无 reference 机制诊断（分析层修正版）。

用法：
    python source/diagnostics.py posterior     # 需要 12 fit 完成（含真正 final endpoint）
    python source/diagnostics.py cross         # 只用 046 baseline
    python source/diagnostics.py probe         # 只用 046 baseline
    python source/diagnostics.py all
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_core import (coarse_pair_index, fine_normalized_rates, load_layer,
                           map_switch_fraction, observed_pair_tables, pair_linear_index,
                           posterior_summary, three_loss_components, whole_cell_rg)
from frozen_imports import contact_model
from round_paths import BASELINE_NPZ, PROBE_SEED, RUN, all_fit_ids
from round_runner import sha256_file

OUT = RUN / "diagnostics"
DATA = None
CORRECTIONS = [
    "probe data_count_nll_delta previously omitted the normalizer; it is now the full normalized count delta and is "
    "asserted against count_A_delta",
    "regularization_total previously summed the raw bend value instead of the weighted 0.01*bend",
    "posterior now includes the true final endpoint instead of the last 10-accepted checkpoint",
    "the unauthorized posterior neighbourhood pseudo-structure Rg was removed; Rg uses the real 5290 coordinates only",
    "cross-resolution chr-pair table now splits 20 cis + 190 inter rows and reports per-row KL contributions that sum "
    "to the total KL; relative mass differences are no longer labelled as KL",
    "probe directions are labelled as smooth common-mode versus local independent-copy, so the contrast is not "
    "attributed to frequency alone",
]


def get_data():
    global DATA
    if DATA is None:
        DATA = load_layer(1_000_000)
    return DATA


def write_json(name: str, value) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False,
                               default=str) + "\n", encoding="utf-8")
    return path


def write_tsv(name: str, rows: list[dict], columns: list[str]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join("" if row.get(key) is None else str(row.get(key)) for key in columns) + "\n")
    return path


def baseline_coordinates() -> tuple[np.ndarray, float]:
    with np.load(BASELINE_NPZ, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        p = float(np.asarray(payload["p"]).item())
    contact_model.assert_inside_unit_ball(coordinates)
    return coordinates, p


# ------------------------------------------------------------------ posterior
POSTERIOR_COLUMNS = ["fit_id", "state", "iteration", "nfev", "fun", "p", "counts_all", "counts_intra",
                     "counts_inter", "entropy_all", "entropy_intra", "entropy_inter",
                     "max_posterior_all", "max_posterior_intra", "max_posterior_inter",
                     "frac_gamma_max_ge_0.9_all", "frac_gamma_max_ge_0.9_intra", "frac_gamma_max_ge_0.9_inter",
                     "map_switch_all_vs_iter0", "map_switch_intra_vs_iter0", "map_switch_inter_vs_iter0",
                     "expected_distance_all", "expected_distance_intra", "expected_distance_inter",
                     "expected_distance_all_frozen_iter0", "expected_distance_intra_frozen_iter0",
                     "expected_distance_inter_frozen_iter0",
                     "distance_over_rg_all", "distance_over_rg_intra", "distance_over_rg_inter",
                     "distance_over_rg_all_frozen_iter0", "distance_over_rg_intra_frozen_iter0",
                     "distance_over_rg_inter_frozen_iter0",
                     "rg_whole_cell_all_5290_beads", "state_source", "optimizer_nit", "coordinates_path"]


def _posterior_row(fit: str, state: str, iteration: int, *, coordinates: np.ndarray, p_value: float,
                   tables: dict, reference: dict | None, nfev, fun, coordinates_path: str) -> dict:
    stats = posterior_summary(tables)
    row = {"fit_id": fit, "state": state, "iteration": iteration, "nfev": nfev, "fun": fun, "p": p_value,
           "counts_all": stats["all"]["counts"], "counts_intra": stats["intra"]["counts"],
           "counts_inter": stats["inter"]["counts"],
           "entropy_all": stats["all"]["mean_entropy"], "entropy_intra": stats["intra"]["mean_entropy"],
           "entropy_inter": stats["inter"]["mean_entropy"],
           "max_posterior_all": stats["all"]["mean_max_posterior"],
           "max_posterior_intra": stats["intra"]["mean_max_posterior"],
           "max_posterior_inter": stats["inter"]["mean_max_posterior"],
           "frac_gamma_max_ge_0.9_all": stats["all"]["fraction_gamma_max_ge_0.9"],
           "frac_gamma_max_ge_0.9_intra": stats["intra"]["fraction_gamma_max_ge_0.9"],
           "frac_gamma_max_ge_0.9_inter": stats["inter"]["fraction_gamma_max_ge_0.9"],
           "expected_distance_all": stats["all"]["mean_expected_distance"],
           "expected_distance_intra": stats["intra"]["mean_expected_distance"],
           "expected_distance_inter": stats["inter"]["mean_expected_distance"],
           "rg_whole_cell_all_5290_beads": whole_cell_rg(coordinates),
           "coordinates_path": coordinates_path}
    rg = row["rg_whole_cell_all_5290_beads"]
    for label in ("all", "intra", "inter"):
        row["distance_over_rg_%s" % label] = (row["expected_distance_%s" % label] / rg) if rg else None
    if reference is None:
        row.update({"map_switch_all_vs_iter0": 0.0, "map_switch_intra_vs_iter0": 0.0,
                    "map_switch_inter_vs_iter0": 0.0,
                    "expected_distance_all_frozen_iter0": row["expected_distance_all"],
                    "expected_distance_intra_frozen_iter0": row["expected_distance_intra"],
                    "expected_distance_inter_frozen_iter0": row["expected_distance_inter"]})
        for label in ("all", "intra", "inter"):
            row["distance_over_rg_%s_frozen_iter0" % label] = (
                row["expected_distance_%s_frozen_iter0" % label] / rg) if rg else None
        return row
    switch = map_switch_fraction(stats["argmax"], reference["argmax"], tables["counts"], tables["cis"])
    row.update({"map_switch_all_vs_iter0": switch["all"]["map_switch_fraction"],
                "map_switch_intra_vs_iter0": switch["intra"]["map_switch_fraction"],
                "map_switch_inter_vs_iter0": switch["inter"]["map_switch_fraction"]})
    frozen_distance = np.sum(reference["gamma"] * tables["distances"], axis=0)
    for label, mask in (("all", np.ones_like(tables["cis"])), ("intra", tables["cis"]), ("inter", ~tables["cis"])):
        weight = tables["counts"][mask]
        total = float(weight.sum())
        value = float(np.sum(weight * frozen_distance[mask]) / total) if total else None
        row["expected_distance_%s_frozen_iter0" % label] = value
        row["distance_over_rg_%s_frozen_iter0" % label] = (value / rg) if (value is not None and rg) else None
    return row


def run_posterior() -> dict:
    data = get_data()
    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for fit in all_fit_ids():
        stage_record = json.loads((RUN / "stages" / fit / "1Mb.json").read_text(encoding="utf-8"))
        history = json.loads((RUN / "stages" / fit / "1Mb.accepted_history.json").read_text(encoding="utf-8"))
        by_iteration = {int(row["iteration"]): row for row in history}
        fit_rows: list[dict] = []
        reference: dict | None = None
        states: list[dict] = []
        for checkpoint in sorted((RUN / "checkpoints" / fit / "1Mb").glob("accepted-*.npz")):
            with np.load(checkpoint, allow_pickle=False) as payload:
                theta = np.asarray(payload["theta"], dtype=np.float64)
                states.append({"state": "checkpoint", "iteration": int(np.asarray(payload["iteration"]).item()),
                               "coordinates": np.asarray(payload["coordinates"], dtype=np.float64).copy(),
                               "p_from_theta": float(contact_model.p_from_q(float(theta[-1]))[0]),
                               "path": str(checkpoint.relative_to(RUN))})
        if not states or states[0]["iteration"] != 0:
            raise RuntimeError("first checkpoint for %s is not iteration 0" % fit)
        final_npz = RUN / "coords" / fit / "1Mb.npz"
        with np.load(final_npz, allow_pickle=False) as payload:
            final_coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
            final_p = float(np.asarray(payload["p"]).item())
        endpoint_record = stage_record.get("endpoint") or {}
        states.append({"state": "final_endpoint", "iteration": -1, "coordinates": final_coordinates,
                       "path": str(final_npz.relative_to(RUN)), "p_override": final_p,
                       "nfev_override": int(stage_record.get("outer_fg_actual") or 0),
                       "fun_override": float(endpoint_record.get("total")) if endpoint_record.get("total") is not None else None,
                       "nit_override": int((stage_record.get("optimizer") or {}).get("nit", -1))})
        for entry in states:
            iteration = int(entry["iteration"])
            history_row = by_iteration.get(iteration)
            if iteration == 0:
                p_value = float(stage_record["initial_p"])
                state_source = "stage_initial_p"
            elif "p_override" in entry:
                p_value = float(entry["p_override"])
                state_source = "endpoint_npz_p"
            else:
                derived = float(entry["p_from_theta"])
                if history_row is None:
                    p_value = derived
                    state_source = "checkpoint_theta_p_from_q_history_missing"
                else:
                    p_value = float(history_row["p"])
                    if abs(p_value - derived) > 1e-12:
                        raise RuntimeError("checkpoint p disagrees with history for %s iteration %d: %.17g vs %.17g"
                                           % (fit, iteration, p_value, derived))
                    state_source = "history_p_cross_checked_with_theta"
            tables = observed_pair_tables(data, entry["coordinates"], p_value)
            if history_row is not None:
                nfev_value, fun_value = int(history_row["nfev"]), float(history_row["fun"])
            elif "nfev_override" in entry:
                nfev_value, fun_value = int(entry["nfev_override"]), entry["fun_override"]
            else:
                nfev_value, fun_value = 0, None
            row = _posterior_row(fit, entry["state"], iteration, coordinates=entry["coordinates"], p_value=p_value,
                                 tables=tables, reference=reference, nfev=nfev_value, fun=fun_value,
                                 coordinates_path=entry["path"])
            row["state_source"] = state_source
            row["optimizer_nit"] = int(entry.get("nit_override", -1))
            if reference is None:
                stats = posterior_summary(tables)
                reference = {"argmax": stats["argmax"].copy(), "gamma": tables["gamma"].copy(),
                             "counts": tables["counts"].copy(), "cis": tables["cis"].copy()}
                row["state"] = "iter0"
            fit_rows.append(row)
            rows.append(row)
        write_tsv("posterior_%s.tsv" % fit, fit_rows, POSTERIOR_COLUMNS)
        checkpoints_only = [row for row in fit_rows if row["state"] == "checkpoint"]
        summary[fit] = {"state_count": len(fit_rows), "checkpoint_count": len(checkpoints_only),
                        "iter0": fit_rows[0] if fit_rows else {},
                        "final": fit_rows[-1] if fit_rows else {},
                        "last_finite_checkpoint": checkpoints_only[-1] if checkpoints_only else {}}
    write_tsv("posterior_all_states.tsv", rows, POSTERIOR_COLUMNS)
    report = {"schema": "p9016-max-contact-posterior-diagnostic-v2",
              "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "definition": {
                  "posterior": "gamma_s = t_s / sum_s t_s using the original marginal (sum) mixture; the max branch is never used to fake determinism",
                  "weighting": "every statistic is count-weighted over the 1,265,114 off-diagonal counts and split into all / intra / inter with their own counts",
                  "expected_distance_dynamic": "count-weighted mean of sum_s gamma_s(now,ij) * d_s(ij) at the current coordinates",
                  "expected_distance_frozen_iter0": "count-weighted mean of sum_s gamma_s(iter0,ij) * d_s(ij) using the SAME current coordinates",
                  "rg_whole_cell": "sqrt(mean over all 5290 real physical beads of squared distance to the bead centroid); no pseudo-structure and no count weight",
                  "final_endpoint": "the true final endpoint from coords/{fit}/1Mb.npz, not the last multiple-of-10 checkpoint",
                  "scope": "model-inference diagnostic only, NOT real allele accuracy"},
              "corrections_applied": CORRECTIONS,
              "fit_count": len(summary), "state_count": len(rows), "summary": summary,
              "reference_opened": False, "phase_opened": False}
    write_json("posterior_diagnostic.json", report)
    return report


# ------------------------------------------------------------ cross resolution
def _layer_mapping(data_fine, data_coarse) -> np.ndarray:
    chromosome = np.asarray(data_fine.locus_chromosome, dtype=np.int64)
    start_bp = np.asarray(data_fine.locus_bin, dtype=np.int64) * int(data_fine.bin_size)
    return coarse_pair_index(data_coarse, chromosome, start_bp)


def run_cross_resolution() -> dict:
    data1 = load_layer(1_000_000)
    coordinates, p_value = baseline_coordinates()
    rates = fine_normalized_rates(data1, coordinates, p_value)
    counts = np.asarray(data1.counts, dtype=np.float64)
    pair_i = np.asarray(data1.pair_i, dtype=np.int64)
    pair_j = np.asarray(data1.pair_j, dtype=np.int64)
    report = {"schema": "p9016-max-contact-cross-resolution-closure-v2",
              "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "source": {"baseline_npz": str(BASELINE_NPZ.relative_to(RUN.parents[1])),
                         "baseline_npz_sha256": sha256_file(BASELINE_NPZ), "p": p_value},
              "corrections_applied": CORRECTIONS,
              "levels": {}, "reference_opened": False, "phase_opened": False}
    chr_pair_rows: list[dict] = []
    for bin_size in (2_000_000, 5_000_000):
        dataC = load_layer(bin_size)
        mapped = _layer_mapping(data1, dataC)
        coarse_i = mapped[pair_i]
        coarse_j = mapped[pair_j]
        on_diag = coarse_i == coarse_j
        n_coarse = int(dataC.n_loci)
        coarse_rates = np.zeros(int(dataC.n_pairs), dtype=np.float64)
        coarse_counts = np.zeros(int(dataC.n_pairs), dtype=np.float64)
        keep = ~on_diag
        linear = pair_linear_index(dataC, coarse_i[keep], coarse_j[keep])
        np.add.at(coarse_rates, linear, rates[keep])
        np.add.at(coarse_counts, linear, counts[keep])
        diag_counts_lost = float(counts[on_diag].sum())
        diag_rate_mass_lost = float(rates[on_diag].sum())
        weight = np.zeros(n_coarse, dtype=np.float64)
        accumulated = np.zeros((2, n_coarse, 3), dtype=np.float64)
        for copy in (0, 1):
            np.add.at(accumulated[copy], mapped, coordinates[copy])
        np.add.at(weight, mapped, 1.0)
        defined = weight > 0.0
        if not bool(defined.all()):
            raise RuntimeError("some coarse bins have no fine representative point")
        coarse_coords = np.zeros((2, n_coarse, 3), dtype=np.float64)
        for copy in (0, 1):
            coarse_coords[copy][defined] = accumulated[copy][defined] / weight[defined][:, None]
        point_rates = fine_normalized_rates(dataC, coarse_coords, p_value)
        prob_fine = coarse_rates / coarse_rates.sum()
        prob_point = point_rates / point_rates.sum()
        both = (prob_fine > 0.0) & (prob_point > 0.0)
        kl_terms = np.zeros_like(prob_fine)
        kl_terms[both] = prob_fine[both] * np.log(prob_fine[both] / prob_point[both])
        kl = float(kl_terms.sum())
        union = (prob_fine > 0.0) | (prob_point > 0.0)
        pearson = float(np.corrcoef(prob_fine[union], prob_point[union])[0, 1])
        log_pearson = float(np.corrcoef(np.log(prob_fine[union]), np.log(prob_point[union]))[0, 1])
        coarse_cis = np.asarray(dataC.cis_pair, dtype=bool)
        group = {
            "fine_sum_cis_mass": float(prob_fine[coarse_cis].sum()),
            "fine_sum_inter_mass": float(prob_fine[~coarse_cis].sum()),
            "point_cis_mass": float(prob_point[coarse_cis].sum()),
            "point_inter_mass": float(prob_point[~coarse_cis].sum()),
            "aggregate_cis_mass": float(np.asarray(dataC.counts)[coarse_cis].sum() / np.asarray(dataC.counts).sum()),
            "aggregate_inter_mass": float(np.asarray(dataC.counts)[~coarse_cis].sum() / np.asarray(dataC.counts).sum()),
        }
        locus_chromosome = np.asarray(dataC.locus_chromosome, dtype=np.int64)
        chr_i = locus_chromosome[np.asarray(dataC.pair_i, dtype=np.int64)]
        chr_j = locus_chromosome[np.asarray(dataC.pair_j, dtype=np.int64)]
        rows_this: list[dict] = []
        names = list(dataC.chromosome_names)

        def make_row(group_name: str, a: int, b: int, mask: np.ndarray) -> dict:
            mass_fine = float(prob_fine[mask].sum())
            mass_point = float(prob_point[mask].sum())
            return {"group": group_name, "chr_a": a, "chr_b": b, "chr_a_name": names[a], "chr_b_name": names[b],
                    "pair_count": int(mask.sum()), "mass_fine_sum": mass_fine, "mass_coarse_point": mass_point,
                    "kl_contribution": float(kl_terms[mask].sum()),
                    "relative_mass_difference": ((mass_fine - mass_point) / mass_point) if mass_point > 0 else None}

        for chromosome in range(len(names)):
            mask = (chr_i == chromosome) & (chr_j == chromosome)
            rows_this.append(make_row("cis", chromosome, chromosome, mask))
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                mask = (chr_i == a) & (chr_j == b)
                rows_this.append(make_row("inter", a, b, mask))
        cis_rows = [row for row in rows_this if row["group"] == "cis"]
        inter_rows = [row for row in rows_this if row["group"] == "inter"]
        if len(cis_rows) != 20 or len(inter_rows) != 190:
            raise RuntimeError("chr-pair split must be 20 cis + 190 inter, got %d + %d"
                               % (len(cis_rows), len(inter_rows)))
        kl_sum = float(sum(row["kl_contribution"] for row in rows_this))
        if abs(kl_sum - kl) > 1e-12 * max(1.0, abs(kl)):
            raise RuntimeError("chr-pair KL contributions do not sum to the total KL: %.17g vs %.17g" % (kl_sum, kl))
        for row in rows_this:
            chr_pair_rows.append({"bin_size_bp": int(bin_size), **row})
        count_match = float(np.max(np.abs(coarse_counts - np.asarray(dataC.counts, dtype=np.float64))))
        report["levels"][str(bin_size)] = {
            "bin_size_bp": int(bin_size), "coarse_n_loci": n_coarse, "coarse_n_pairs": int(dataC.n_pairs),
            "fine_pairs_mapped_to_coarse_diag": int(on_diag.sum()),
            "counts_mapped_to_coarse_diag": diag_counts_lost,
            "rate_mass_mapped_to_coarse_diag": diag_rate_mass_lost,
            "max_abs_count_mismatch_vs_coarse_aggregate": count_match,
            "counts_conserved": bool(count_match == 0.0),
            "kl_fine_sum_vs_coarse_point": kl,
            "kl_sum_over_chr_pair_groups": kl_sum,
            "chr_pair_group_count": {"cis": len(cis_rows), "inter": len(inter_rows)},
            "pearson_probability": pearson, "pearson_log_probability": log_pearson,
            "support_union_pairs": int(union.sum()),
            "group_mass": group,
            "notes": {"not_nested": "2Mb and 5Mb grids are not nested with each other",
                      "absolute_nll_not_compared": True,
                      "relative_mass_difference_is_not_kl": True},
        }
    write_json("cross_resolution_closure.json", report)
    write_tsv("cross_resolution_chr_pairs.tsv", chr_pair_rows,
              ["bin_size_bp", "group", "chr_a", "chr_a_name", "chr_b", "chr_b_name", "pair_count",
               "mass_fine_sum", "mass_coarse_point", "kl_contribution", "relative_mass_difference"])
    return report


# ---------------------------------------------------------------------- probe
def _direction_smooth_common_mode(n_loci: int, data, rng: np.random.Generator) -> np.ndarray:
    """平滑方向：两 copy 共用同一位移（共模）。"""
    direction = np.zeros((1, n_loci, 3), dtype=np.float64)
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        n = int(slc.stop - slc.start)
        for mode in (1, 2, 3):
            amplitude = rng.normal(size=3)
            phase = np.cos(np.pi * mode * (np.arange(n) + 0.5) / n)
            direction[0, slc, :] += phase[:, None] * amplitude[None, :]
    return np.broadcast_to(direction, (2, n_loci, 3)).copy()


def _direction_local_independent_copies(n_loci: int, rng: np.random.Generator) -> np.ndarray:
    """局部方向：两 copy 独立位移（含差模）。"""
    return rng.normal(size=(2, n_loci, 3))


def _prepare_direction(raw: np.ndarray) -> np.ndarray:
    direction = np.asarray(raw, dtype=np.float64)
    direction = direction - direction.reshape(-1, 3).mean(axis=0)[None, None, :]
    rms = float(np.sqrt(np.mean(np.sum(direction.reshape(-1, 3) ** 2, axis=1))))
    return direction / rms


def _feasible_amplitude(coordinates: np.ndarray, direction: np.ndarray, target: float):
    scale = float(target)
    for _ in range(80):
        candidate = coordinates + scale * direction
        peak = float(np.linalg.norm(candidate.reshape(-1, 3), axis=1).max())
        if peak < 1.0:
            return candidate, scale, peak
        scale *= 0.5
    raise RuntimeError("probe amplitude could not be made feasible")


def _kabsch_rms(reference: np.ndarray, moving: np.ndarray) -> dict:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    mov = np.asarray(moving, dtype=np.float64).reshape(-1, 3)
    ref_c = ref - ref.mean(axis=0)
    mov_c = mov - mov.mean(axis=0)
    covariance = mov_c.T @ ref_c
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    residual = mov_c @ rotation.T - ref_c
    return {"rms_deformation": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
            "rotation_det": float(np.linalg.det(rotation)), "rotation": rotation.tolist()}


DIRECTION_SEMANTICS = {
    "smooth_common_mode": {
        "id": "smooth_common_mode",
        "label": "smooth common-mode direction (both copies receive the SAME displacement)",
        "construction": "per-chromosome DCT modes 1-3 with Gaussian amplitudes, shared by both copies, "
                        "global translation removed, whole-cell displacement RMS normalized",
    },
    "local_independent_copies": {
        "id": "local_independent_copies",
        "label": "local direction with independent per-copy displacements (contains a difference mode)",
        "construction": "iid Gaussian per bead and per copy, global translation removed, "
                        "whole-cell displacement RMS normalized",
    },
}


def run_probe() -> dict:
    data = get_data()
    n_loci = int(data.n_loci)
    coordinates, p_value = baseline_coordinates()
    rng = np.random.default_rng(PROBE_SEED)
    directions = {
        "smooth_common_mode": _prepare_direction(_direction_smooth_common_mode(n_loci, data, rng)),
        "local_independent_copies": _prepare_direction(_direction_local_independent_copies(n_loci, rng)),
    }
    baseline_rg = whole_cell_rg(coordinates)
    baseline_rates = fine_normalized_rates(data, coordinates, p_value)
    baseline_audit = three_loss_components(data, coordinates, p_value)
    counts = np.asarray(data.counts, dtype=np.float64)
    observed = counts > 0.0
    rows = []
    probe_dir = OUT / "probe_coords"
    probe_dir.mkdir(parents=True, exist_ok=True)
    for name, direction in directions.items():
        for fraction in (0.01, 0.05):
            for sign in (1.0, -1.0):
                target = sign * fraction * baseline_rg
                candidate, actual, peak = _feasible_amplitude(coordinates, direction, target)
                contact_model.assert_inside_unit_ball(candidate)
                audit = three_loss_components(data, candidate, p_value)
                probe_rates = fine_normalized_rates(data, candidate, p_value)
                q_base = baseline_rates / baseline_rates.sum()
                q_probe = probe_rates / probe_rates.sum()
                kl = float(np.sum(q_base * np.log(q_base / q_probe)))
                observed_log_delta = float(np.sum(counts[observed]
                                                  * np.log(baseline_rates[observed] / probe_rates[observed])))
                normalizer_delta_raw = baseline_audit["n_off"] * math.log(audit["Zsum"] / baseline_audit["Zsum"])
                count_delta_raw = observed_log_delta + normalizer_delta_raw
                normalized_delta = count_delta_raw / baseline_audit["Nraw"]
                count_a_delta = audit["count_A"] - baseline_audit["count_A"]
                if abs(normalized_delta - count_a_delta) > 1e-9:
                    raise AssertionError("probe count delta does not match count_A delta: %.3e vs %.3e"
                                         % (normalized_delta, count_a_delta))
                alignment = _kabsch_rms(coordinates, candidate)
                path = probe_dir / ("%s_%s%.2f.npz" % (name, "plus" if sign > 0 else "minus", fraction))
                np.savez_compressed(path, coordinates=candidate, baseline_coordinates=coordinates,
                                    direction=direction, actual_amplitude=np.asarray(actual),
                                    fraction=np.asarray(fraction), sign=np.asarray(sign), p=np.asarray(p_value))
                rows.append({
                    "probe": name, "direction_label": DIRECTION_SEMANTICS[name]["label"],
                    "fraction_of_baseline_rg": fraction, "sign": sign,
                    "target_amplitude": target, "actual_amplitude": actual,
                    "amplitude_shrunk": bool(abs(actual - target) > 0.0),
                    "peak_radius": peak, "baseline_whole_cell_rg": baseline_rg,
                    "kl_q_base_vs_q_probe": kl,
                    "Noff_over_Nraw_times_kl": (baseline_audit["n_off"] / baseline_audit["Nraw"]) * kl,
                    "observed_log_rate_delta_raw": observed_log_delta,
                    "normalizer_delta_raw": normalizer_delta_raw,
                    "count_delta_raw": count_delta_raw,
                    "data_count_nll_delta": normalized_delta,
                    "count_A_delta": count_a_delta,
                    "count_delta_identity_abs_error": abs(normalized_delta - count_a_delta),
                    "fullJ_delta_A": audit["fullJ_A"] - baseline_audit["fullJ_A"],
                    "fullJ_delta_B": audit["fullJ_B"] - baseline_audit["fullJ_B"],
                    "fullJ_delta_C": audit["fullJ_C"] - baseline_audit["fullJ_C"],
                    "regularization_delta_weighted": audit["regularization_total"] - baseline_audit["regularization_total"],
                    "bond_delta": audit["regularizers_weighted"]["bond"] - baseline_audit["regularizers_weighted"]["bond"],
                    "repulsion_delta": audit["regularizers_weighted"]["repulsion"] - baseline_audit["regularizers_weighted"]["repulsion"],
                    "bend_delta_weighted": audit["regularizers_weighted"]["bend"] - baseline_audit["regularizers_weighted"]["bend"],
                    "bend_delta_raw": audit["regularizers_raw"]["bend"] - baseline_audit["regularizers_raw"]["bend"],
                    "p_prior_delta": audit["regularizers_weighted"]["p_prior"] - baseline_audit["regularizers_weighted"]["p_prior"],
                    "rms_deformation_after_proper_rigid": alignment["rms_deformation"],
                    "rotation_det": alignment["rotation_det"],
                    "probe_npz": str(path.relative_to(RUN)),
                    "probe_npz_sha256": sha256_file(path),
                })
    report = {"schema": "p9016-max-contact-local-observability-probe-v2",
              "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "seed": PROBE_SEED, "baseline_npz": str(BASELINE_NPZ.relative_to(RUN.parents[1])),
              "baseline_whole_cell_rg": baseline_rg, "baseline_p": p_value,
              "direction_definition": DIRECTION_SEMANTICS,
              "label_caveat": "the two directions differ in BOTH smoothness and copy symmetry; the contrast must not "
                              "be attributed to frequency alone",
              "centre_caveat": "smooth_common_mode uses per-chromosome DCT modes 1-3 with zero per-chromosome mean, so "
                               "the 20 chromosome centres are unchanged and it mainly probes internal smooth deformation; "
                               "local_independent_copies is dominated by per-bead perturbations with only a small random "
                               "centre drift. Therefore KL>0 on these 8 probes does NOT establish that chromosome centre "
                               "placement is identifiable or well constrained; no weak-direction search over centre "
                               "placement was performed here. Whether real placement improves is answered by the unified "
                               "190-centre / 760-copy-centre readings.",
              "amplitude_rule": "all beads move along one straight line; if any bead would leave the strict ball the "
                                "whole amplitude is halved uniformly (no per-bead clipping)",
              "e_and_p": "exposure and p fixed at the baseline values; nothing is fitted",
              "count_delta_definition": "data_count_nll_delta = [sum_C C*log(r_base/r_probe) + Noff*log(Z_probe/Z_base)] / Nraw, "
                                        "asserted equal to count_A(probe) - count_A(base) (both normalized by Nraw)",
              "scope": "local directional observability only, NOT a global identifiability proof",
              "corrections_applied": CORRECTIONS,
              "probes": rows, "reference_opened": False, "phase_opened": False}
    write_json("probe_diagnostic.json", report)
    write_tsv("probe_diagnostic.tsv", rows, list(rows[0].keys()))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=("posterior", "cross", "probe", "all"))
    args = parser.parse_args()
    if args.part in ("cross", "all"):
        run_cross_resolution()
    if args.part in ("probe", "all"):
        run_probe()
    if args.part in ("posterior", "all"):
        run_posterior()
    print(json.dumps({"status": "ok", "part": args.part}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
