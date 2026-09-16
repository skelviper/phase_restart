#!/usr/bin/env python3
"""使用冻结的 V1 无标签规则完成一次 native FDG 输出。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE.parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fdg_proposal as fp  # noqa: E402
from pr import genome  # noqa: E402
from pr.contact_model import JointObjective, aggregate_from_arrays, verify_frozen_snpfree  # noqa: E402


EXPECTED_COORD_SHA = "49501f5b38d699fb2c9c8849616edd02b70fccd5ae39e1efe6036e17f5f620e7"
EXPECTED_THETA_SHA = "588bafce1d7652fa8cf17cb060d23807dd6098c3e072fd9fc52784a145860a57"
EXPECTED_INPUT_SHA = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"


def fail(message: str) -> None:
    raise RuntimeError(message)


def write_3dg(path: Path, coordinates: np.ndarray, grid: fp.FullGrid) -> None:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (2, grid.n_loci, 3) or not np.all(np.isfinite(values)):
        fail("cannot write a non-finite or malformed 3dg coordinate array")
    with path.open("wt", encoding="utf-8", newline="\n") as handle:
        for chromosome in range(grid.n_chromosomes):
            locus_slice = grid.chromosome_slice(chromosome)
            for copy in range(2):
                track = 2 * chromosome + copy
                for local, bead in enumerate(grid.beads[grid.track_slice(track)]):
                    point = values[copy, locus_slice.start + local]
                    handle.write(
                        f"{grid.track_names[track]}\t{int(bead[1])}\t"
                        f"{point[0]:.17g}\t{point[1]:.17g}\t{point[2]:.17g}\n"
                    )


def json_components(components: dict) -> dict:
    result = {}
    for key, value in components.items():
        if isinstance(value, (np.integer, int)):
            result[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def load_full_data(input_path: Path, lengths: list[tuple[str, int]]):
    input_audit = verify_frozen_snpfree(input_path)
    if input_audit["snpfree_sha256"] != EXPECTED_INPUT_SHA:
        fail("SNP-free input SHA differs from frozen config")
    contacts = genome.load_all(str(input_path))
    data = aggregate_from_arrays(
        [name for name, _ in lengths], [length for _, length in lengths],
        contacts["ci"], contacts["p1"], contacts["cj"], contacts["p2"], fp.BIN_SIZE,
    )
    if (data.raw_records, data.raw_same_bin, data.raw_cis_offdiag, data.raw_inter) != \
            (1_703_888, 438_774, 696_680, 568_434):
        fail("full original aggregate budget differs from frozen audit")
    records = fp.make_raw_records(
        contacts["ci"], contacts["p1"], contacts["cj"], contacts["p2"],
        n_chromosomes=len(lengths),
    )
    if records.n_records != data.raw_records:
        fail("full raw records changed during finalization")
    return data, records, input_audit


def main() -> None:
    config_path = HERE / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "released_for_real_fdg":
        fail("config is not in the released pre-finalization state")
    prepared_path = HERE / "build" / "prepared_state.npz"
    output_path = HERE / "build" / "bridge_output.bin"
    receipt_path = HERE / "native_attempt.json"
    manifest_path = HERE / "proposal_manifest.json"
    mapping_audit_path = HERE / "canonical_swap_mapping_audit.json"
    if not prepared_path.exists() or not output_path.exists() or not receipt_path.exists() or \
            not mapping_audit_path.exists():
        fail("prepared state, native output, receipt, or mapping audit is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("returncode") != 0 or receipt.get("status") != "completed_zero":
        fail("native attempt did not complete with exit code zero")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared_native_pending":
        fail("proposal manifest is not the single prepared attempt")
    mapping_audit = json.loads(mapping_audit_path.read_text(encoding="utf-8"))
    if mapping_audit.get("schema") != "fdg-canonical-swap-mapping-audit-v1" or \
            len(mapping_audit.get("swap_cases", {})) != 21 or \
            mapping_audit.get("native_input_mapping", {}).get("explicit_bead_inventory_byte_equal") is not True or \
            mapping_audit.get("native_output_mapping", {}).get("native_bead_to_v1_to_bead_byte_equal") is not True:
        fail("canonical swap/mapping audit is incomplete")
    input_blob_path = HERE / "build" / "bridge_input.bin"
    if fp.sha256_file(input_blob_path) != manifest.get("artifacts", {}).get("input_blob_sha256"):
        fail("prepared input blob changed before finalization")
    if receipt.get("input_sha256") != manifest.get("artifacts", {}).get("input_blob_sha256"):
        fail("native receipt input SHA differs from prepared proposal")
    if receipt.get("output_sha256") != fp.sha256_file(output_path):
        fail("native receipt output SHA differs from actual output")

    anchor = config["anchor"]
    theta_path = Path(anchor["theta_path"])
    coordinates_path = Path(anchor["coordinates_path"])
    if fp.sha256_file(theta_path) != EXPECTED_THETA_SHA or \
            fp.sha256_file(coordinates_path) != EXPECTED_COORD_SHA:
        fail("released anchor changed before finalization")
    with np.load(prepared_path, allow_pickle=False) as prepared:
        anchor_coordinates = np.asarray(prepared["anchor_coordinates"], dtype=np.float64).copy()
        canonical_anchor = np.asarray(prepared["canonical_coordinates"], dtype=np.float64).copy()
        swapped_bits = np.asarray(prepared["swapped_by_chromosome"], dtype=bool).copy()
        anchor_theta = np.asarray(prepared["anchor_theta"], dtype=np.float64).copy()
        grid_lengths = np.asarray(prepared["grid_lengths"], dtype=np.int64).copy()
    grid = fp.make_full_grid([int(value) for value in grid_lengths])
    if anchor_coordinates.shape != (2, grid.n_loci, 3) or anchor_theta.shape != (6 * grid.n_loci + 1,):
        fail("prepared anchor shape is invalid")
    if not np.all(np.isfinite(anchor_coordinates)) or not np.all(np.isfinite(anchor_theta)):
        fail("prepared anchor contains non-finite values")
    if not np.array_equal(fp.canonicalize_coordinates(anchor_coordinates, grid).coordinates,
                          canonical_anchor):
        fail("prepared canonical anchor is not reproducible")
    if not np.isclose(anchor_theta[-1], anchor["q_anchor"], rtol=0.0, atol=1e-15):
        fail("prepared anchor q differs from released q")

    lengths = genome.chrom_lengths(str(PROJECT_ROOT / config["input"]["path"]))
    if len(lengths) != grid.n_chromosomes or not np.array_equal(
            grid.chromosome_lengths, np.asarray([length for _, length in lengths], dtype=np.int64)):
        fail("input/grid chromosome lengths changed")
    data, records, input_audit = load_full_data(PROJECT_ROOT / config["input"]["path"], lengths)
    if data.n_loci != grid.n_loci:
        fail("full aggregate and explicit native grid have different loci")

    bridge_output = fp.read_bridge_output(output_path, grid)
    if bridge_output.n_raw_pairs != 1_265_114:
        fail("native output raw edge count differs from the frozen force graph")
    raw_native_canonical = fp.bead_order_to_coordinates(
        bridge_output.native_final_bead_order, grid,
    )
    raw_path = RUN_ROOT / config["outputs"]["raw_coordinates"]
    mapped_npz_path = RUN_ROOT / config["outputs"]["mapped_coordinates"]
    mapped_3dg_path = RUN_ROOT / config["outputs"]["mapped_3dg"]
    accepted_npz_path = RUN_ROOT / config["outputs"]["accepted_coordinates"]
    accepted_3dg_path = RUN_ROOT / config["outputs"]["accepted_3dg"]
    accepted_theta_path = RUN_ROOT / config["outputs"]["accepted_theta"]
    decision_path = RUN_ROOT / config["outputs"]["decision_log"]
    for path in (raw_path, mapped_npz_path, mapped_3dg_path, accepted_npz_path,
                 accepted_3dg_path, accepted_theta_path, decision_path):
        if path.exists():
            fail(f"refusing to overwrite final artifact: {path}")
    write_3dg(raw_path, raw_native_canonical, grid)
    mapped_canonical, center, scale, native_max_radius = fp.map_native_to_unit_ball(
        raw_native_canonical, grid,
    )
    mapped_coordinates = fp.undo_canonicalization(mapped_canonical, grid, swapped_bits)
    anchor_canonical_check = fp.canonicalize_coordinates(anchor_coordinates, grid)
    if not np.array_equal(anchor_canonical_check.coordinates, canonical_anchor):
        fail("anchor canonicalization changed at finalization")
    if not np.all(np.linalg.norm(mapped_coordinates.reshape((-1, 3)), axis=1) < 1.0):
        fail("mapped proposal is not strict interior")
    np.savez_compressed(
        mapped_npz_path,
        coordinates=mapped_coordinates,
        canonical_coordinates=mapped_canonical,
        center=center,
        scale=np.asarray([scale]),
        native_max_radius=np.asarray([native_max_radius]),
        native_unit=np.asarray([bridge_output.native_unit]),
        source_avg_bb=np.asarray([bridge_output.source_avg_bb]),
        q=np.asarray([anchor_theta[-1]]),
        p=np.asarray([fp.p_from_q(float(anchor_theta[-1]))]),
        swapped_by_chromosome=swapped_bits,
    )
    write_3dg(mapped_3dg_path, mapped_coordinates, grid)

    objective = JointObjective(data)
    anchor_total, _, anchor_components = objective.evaluate(anchor_theta, need_gradient=False)
    anchor_x, anchor_p = objective.coordinates_and_p(anchor_theta)
    anchor_coordinate_delta = float(np.max(np.abs(anchor_x - anchor_coordinates)))
    if anchor_coordinate_delta > 1e-12:
        fail("theta-derived anchor coordinates differ materially from released coordinates")
    if not np.isclose(anchor_total, config["anchor"]["anchor_objective_total"], rtol=0.0, atol=1e-10):
        fail("V1 anchor objective differs from parent continuation audit")
    if not np.isclose(anchor_components["count_nll_normalized"],
                       config["anchor"]["anchor_count_nll_normalized"], rtol=0.0, atol=1e-10):
        fail("V1 anchor count NLL differs from parent continuation audit")
    q = float(anchor_theta[-1])
    trial_records = []
    selected = None
    for alpha in fp.ALPHA_SEQUENCE:
        trial_x = anchor_x + float(alpha) * (mapped_coordinates - anchor_x)
        trial = {
            "alpha": float(alpha),
            "finite": bool(np.all(np.isfinite(trial_x))),
            "full_grid_shape": list(trial_x.shape) == [2, grid.n_loci, 3],
            "strict_interior": False,
        }
        if trial["finite"] and trial["full_grid_shape"]:
            norms = np.linalg.norm(trial_x.reshape((-1, 3)), axis=1)
            trial["strict_interior"] = bool(np.all(norms < 1.0))
        if trial["finite"] and trial["full_grid_shape"] and trial["strict_interior"]:
            trial_y = fp.sphere_inverse(trial_x)
            trial_theta = anchor_theta.copy()
            trial_theta[:-1] = trial_y.ravel()
            trial_theta[-1] = q
            total, _, components = objective.evaluate(trial_theta, need_gradient=False)
            trial["theta_q"] = float(trial_theta[-1])
            trial["total"] = float(total)
            trial["count_nll_normalized"] = float(components["count_nll_normalized"])
            trial["components"] = json_components(components)
            trial["objective_improved"] = bool(total < anchor_total - 1e-10)
            trial["count_within_tolerance"] = bool(
                components["count_nll_normalized"] <=
                anchor_components["count_nll_normalized"] + 1e-9
            )
            trial["passes_acceptance"] = bool(
                trial["objective_improved"] and trial["count_within_tolerance"]
            )
            if trial["passes_acceptance"] and selected is None:
                selected = (float(alpha), trial_x.copy(), trial_theta.copy())
                trial["selection"] = "accepted_first"
            elif trial["passes_acceptance"]:
                trial["selection"] = "not_selected_after_first_accept"
            else:
                trial["selection"] = "rejected"
        else:
            trial.update({
                "total": None,
                "count_nll_normalized": None,
                "components": None,
                "objective_improved": False,
                "count_within_tolerance": False,
                "passes_acceptance": False,
                "selection": "rejected_invalid_trial",
            })
        trial_records.append(trial)

    if selected is None:
        acceptance_status = "rejected_no_label_free_improvement"
        accepted_kind = "anchor_pointer"
        accepted_coordinates = anchor_coordinates
        accepted_theta = anchor_theta
        shutil.copyfile(theta_path, accepted_npz_path)
        shutil.copyfile(theta_path, accepted_theta_path)
        shutil.copyfile(coordinates_path, accepted_3dg_path)
        selected_alpha = None
    else:
        acceptance_status = "accepted_first_finite_full_grid_objective_and_count"
        accepted_kind = "trial_alpha"
        selected_alpha, accepted_coordinates, accepted_theta = selected
        np.savez_compressed(
            accepted_npz_path, theta=accepted_theta, coordinates=accepted_coordinates,
            q=np.asarray([q]), p=np.asarray([anchor_p]), alpha=np.asarray([selected_alpha]),
        )
        shutil.copyfile(accepted_npz_path, accepted_theta_path)
        write_3dg(accepted_3dg_path, accepted_coordinates, grid)

    selected_record = next(
        (record for record in trial_records if record.get("selection") == "accepted_first"), None,
    )
    decision = {
        "schema": "fdg-acceptance-v1",
        "status": acceptance_status,
        "recorded_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "accepted_kind": accepted_kind,
        "selected_alpha": selected_alpha,
        "q_fixed": q,
        "p_fixed": anchor_p,
        "anchor": {
            "theta_path": str(theta_path),
            "theta_sha256": fp.sha256_file(theta_path),
            "coordinates_path": str(coordinates_path),
            "coordinates_sha256": fp.sha256_file(coordinates_path),
            "theta_derived_coordinate_max_abs_delta": anchor_coordinate_delta,
            "objective_total": float(anchor_total),
            "count_nll_normalized": float(anchor_components["count_nll_normalized"]),
            "components": json_components(anchor_components),
        },
        "native": {
            "attempt_receipt": str(receipt_path),
            "attempt_receipt_sha256": fp.sha256_file(receipt_path),
            "output_path": str(output_path),
            "output_sha256": fp.sha256_file(output_path),
            "n_beads": bridge_output.n_beads,
            "n_binned_pairs": bridge_output.n_binned_pairs,
            "n_raw_pairs": bridge_output.n_raw_pairs,
            "n_iter": bridge_output.n_iter,
            "source_avg_bb": bridge_output.source_avg_bb,
            "native_unit": bridge_output.native_unit,
            "max_abs_jitter": bridge_output.max_abs_jitter,
        },
        "mapping": {
            "raw_native_max_radius": native_max_radius,
            "global_center": center.tolist(),
            "global_scale": scale,
            "target_interior_radius": fp.INTERIOR_TARGET,
            "mapped_max_radius": float(np.linalg.norm(
                mapped_coordinates.reshape((-1, 3)), axis=1).max(),
            ),
            "canonical_swap_bits": swapped_bits.astype(int).tolist(),
        },
        "alpha_trials": trial_records,
        "selected_record": selected_record,
        "mapping_audit": {
            "path": str(mapping_audit_path),
            "sha256": fp.sha256_file(mapping_audit_path),
            "schema": mapping_audit["schema"],
            "swap_cases": len(mapping_audit["swap_cases"]),
            "native_input_output_bead_mapping": mapping_audit["native_output_mapping"]["native_bead_to_v1_to_bead_byte_equal"],
        },
        "artifacts": {
            "raw_coordinates": str(raw_path),
            "raw_coordinates_sha256": fp.sha256_file(raw_path),
            "mapped_coordinates": str(mapped_npz_path),
            "mapped_coordinates_sha256": fp.sha256_file(mapped_npz_path),
            "mapped_3dg": str(mapped_3dg_path),
            "mapped_3dg_sha256": fp.sha256_file(mapped_3dg_path),
            "accepted_coordinates": str(accepted_npz_path),
            "accepted_coordinates_sha256": fp.sha256_file(accepted_npz_path),
            "accepted_3dg": str(accepted_3dg_path),
            "accepted_3dg_sha256": fp.sha256_file(accepted_3dg_path),
            "accepted_theta": str(accepted_theta_path),
            "accepted_theta_sha256": fp.sha256_file(accepted_theta_path),
            "canonical_mapping_audit": str(mapping_audit_path),
            "canonical_mapping_audit_sha256": fp.sha256_file(mapping_audit_path),
            "decision_log": str(decision_path),
        },
        "selection_constraints": {
            "objective": "V1 JointObjective total on original full AggregatedContacts",
            "strict_objective_margin": 1e-10,
            "count_nll_tolerance": 1e-9,
            "reference_phase_r2_used": False,
        },
        "input_ledger": {
            "path": str(PROJECT_ROOT / config["input"]["path"]),
            "sha256": input_audit["snpfree_sha256"],
            "raw_records": data.raw_records,
            "same_bin": data.raw_same_bin,
            "cis_offdiag": data.raw_cis_offdiag,
            "inter": data.raw_inter,
        },
    }
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    decision["artifacts"]["decision_log_sha256"] = fp.sha256_file(decision_path)
    manifest.update({
        "status": acceptance_status,
        "finalized_utc": decision["recorded_utc"],
        "native": decision["native"],
        "mapping": decision["mapping"],
        "mapping_audit": decision["mapping_audit"],
        "acceptance": {
            "status": acceptance_status,
            "selected_alpha": selected_alpha,
            "trials": trial_records,
            "decision_log": str(decision_path),
            "decision_log_sha256": decision["artifacts"]["decision_log_sha256"],
        },
        "artifacts": {
            **manifest.get("artifacts", {}),
            **decision["artifacts"],
        },
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    config["status"] = "completed_accepted" if selected is not None else "completed_rejected_to_anchor"
    config["outputs"]["anchor_fill_required"] = False
    config["outputs"]["decision_log_sha256"] = decision["artifacts"]["decision_log_sha256"]
    config["outputs"]["canonical_mapping_audit_sha256"] = decision["artifacts"]["canonical_mapping_audit_sha256"]
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": config["status"],
        "acceptance_status": acceptance_status,
        "selected_alpha": selected_alpha,
        "anchor_total": anchor_total,
        "anchor_count_nll_normalized": anchor_components["count_nll_normalized"],
        "raw_native_sha256": decision["artifacts"]["raw_coordinates_sha256"],
        "mapped_3dg_sha256": decision["artifacts"]["mapped_3dg_sha256"],
        "accepted_3dg_sha256": decision["artifacts"]["accepted_3dg_sha256"],
        "accepted_theta_sha256": decision["artifacts"]["accepted_theta_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
