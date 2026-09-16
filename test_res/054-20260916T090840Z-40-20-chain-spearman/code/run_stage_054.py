"""054 正确共享链运行器：40Mb(复用) -> 20Mb -> 10Mb -> 5Mb -> 2Mb -> 1Mb -> 500kb -> 200kb。

每个进程只跑一层：读本链上一层端点 npz，按冻结 warm_start_from_layer（真实 bp；
只扰动新增/重复 loci；径向 clip）prolongation 到本层，写出
coords/new-chain/<stage>.{npz,3dg}、stages/new-chain/<stage>.json 与
logs/new-chain-<stage>.terminal.json。

与 052 的差别只有两点：链定义里补回 20Mb 层；40Mb 用 052 已冻结端点（登记为复用前缀）。
objective / 优化器 / 停止判据 / 权重 / e 全部复用 045/049 的冻结实现（普通 raw L-BFGS，
非 multiscale，固定 production e，bend=0.01，其它权重 1，pair_block=262144，tile_rows=32）。

本进程只读无标签 contacts；不打开 phase 列，也不打开 reference 3DG。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bootstrap_054 import (SharedCaptureObjective, contact_model, data_io, formal_controller,  # noqa: E402
                           reconstruction_init, round_runner)
from round_paths_054 import (EXPECTED_N_LOCI, EXPECTED_RAW_RECORDS, FIT_ID, GROUP3_FG, GROUP4_FG,  # noqa: E402
                             NEW_FG_TOTAL, P_INIT, REUSED_AGGREGATES, REUSED_PREFIX_BIN,
                             REUSED_PREFIX_FG, REUSED_PREFIX_NPZ, REUSED_PREFIX_NPZ_SHA256,
                             RUN_STAGES, STAGE_BIN, STAGE_FG_CAP, WEIGHTS, stage_bin,
                             stage_of_endpoint, ROOT)


def load_layer(stage: str):
    """本层 aggregate：全部为已冻结复用输入，只核对，不重建。"""
    bin_size = stage_bin(stage)
    if bin_size not in REUSED_AGGREGATES:
        raise RuntimeError("stage %s has no frozen aggregate input" % stage)
    path, expected_sha = REUSED_AGGREGATES[bin_size]
    if not path.is_file():
        raise RuntimeError("missing layer aggregate for stage %s: %s" % (stage, path))
    actual_sha = round_runner.sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError("stage %s aggregate SHA mismatch: %s" % (stage, actual_sha))
    data = data_io.load_aggregate(path)
    if int(data.bin_size) != int(bin_size):
        raise RuntimeError("stage %s layer bin_size mismatch: %d != %d" % (stage, int(data.bin_size), bin_size))
    expected_bins = np.array([(int(L) + bin_size - 1) // bin_size for L in data.chromosome_lengths],
                             dtype=np.int64)
    if not np.array_equal(np.asarray(data.n_bins, dtype=np.int64), expected_bins):
        raise RuntimeError("stage %s does not sit on a real %d bp grid" % (stage, bin_size))
    if int(data.n_loci) != int(expected_bins.sum()) != EXPECTED_N_LOCI[int(bin_size)]:
        raise RuntimeError("stage %s grid size mismatch" % stage)
    if int(data.budget()["raw_records"]) != EXPECTED_RAW_RECORDS:
        raise RuntimeError("stage %s lost records" % stage)
    return data


def previous_role(previous: Path) -> str:
    return "reused_frozen_prefix" if previous.resolve() == REUSED_PREFIX_NPZ.resolve() \
        else "new_chain_endpoint"


def load_start(stage: str, previous: Path | None, data):
    """返回 (coordinates, raw_y, p, q)：只从本链上一层端点 prolongation。"""
    bin_size = stage_bin(stage)
    if previous is None:
        raise RuntimeError("stage %s has no previous endpoint; the 40Mb root is reused, not run" % stage)
    if previous_role(previous) == "reused_frozen_prefix":
        actual = round_runner.sha256_file(previous)
        if actual != REUSED_PREFIX_NPZ_SHA256:
            raise RuntimeError("reused 40Mb prefix SHA mismatch: %s" % actual)
    with np.load(previous, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        p_init = float(np.asarray(payload["p"]).item())
        q_init = float(np.asarray(payload["q"]).item())
    if coordinates.shape[1] == int(data.n_loci):
        raise RuntimeError("previous endpoint already has the target grid; chain prolongation expected")
    previous_stage = stage_of_endpoint(previous)
    previous_data = load_layer(previous_stage)
    if int(previous_data.n_loci) != int(coordinates.shape[1]):
        raise RuntimeError("previous endpoint shape does not match its own layer grid")
    positions = np.broadcast_to(
        (np.asarray(previous_data.locus_bin, dtype=np.int64) * int(previous_data.bin_size)),
        (2, coordinates.shape[1])).copy()
    chromosomes = np.broadcast_to(np.asarray(previous_data.locus_chromosome, dtype=np.int32),
                                  (2, coordinates.shape[1])).copy()
    warm = reconstruction_init.warm_start_from_layer(
        coordinates, positions, chromosomes, tuple(data.chromosome_names),
        tuple(int(v) for v in data.chromosome_lengths), bin_size, 2207)
    coordinates = np.asarray(warm["coords"], dtype=np.float64)
    contact_model.assert_inside_unit_ball(coordinates)
    raw_y = contact_model.sphere_inverse(coordinates)
    return coordinates, raw_y, float(p_init), float(q_init), warm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=RUN_STAGES)
    parser.add_argument("--previous", required=True, help="本链上一层端点 npz（20Mb 为复用的 40Mb 前缀）")
    args = parser.parse_args()
    stage = str(args.stage)
    bin_size = stage_bin(stage)
    fg_cap = int(STAGE_FG_CAP[stage])
    previous = Path(args.previous)
    if stage == "20Mb" and previous_role(previous) != "reused_frozen_prefix":
        raise RuntimeError("the 20Mb layer must prolongate the reused 40Mb prefix")

    round_runner.install(RUN)
    formal_controller.RUN = RUN
    data = load_layer(stage)
    coordinates, raw_y, p_init, q_init, warm = load_start(stage, previous, data)
    if coordinates.shape != (2, int(data.n_loci), 3):
        raise RuntimeError("stage %s start shape mismatch" % stage)
    sphere = contact_model.sphere_forward(raw_y)
    sphere_error = float(np.max(np.abs(sphere - coordinates)))
    if sphere_error > 1e-10:
        raise RuntimeError("start raw_y does not map back to coordinates")

    mode = "V0-fixed-production-e" if data.count_mode == "raw_integer" else "V0-known-generating-e"
    state: dict[str, object] = {"objective": None}

    def factory(d, model_id, w, known_e):
        if state["objective"] is None:
            state["objective"] = SharedCaptureObjective(
                d, model_id, weights=w, mode=mode, device="cuda", pair_block=262_144,
                inner_cap=80, cg_cap=80, profile_warm_start=True, known_e=None)
        return state["objective"]

    formal_controller._objective = factory

    row = {
        "fit_id": FIT_ID, "model_id": "G", "kind": "real", "fixture": None, "candidate": "random",
        "start_name": "054-40-20-chain",
        "objective_variant": "G/full-J/raw/fixed-production-e",
        "weights": dict(WEIGHTS),
        "stages": [{"stage": stage, "bin_size_bp": bin_size, "fg_cap": fg_cap}],
    }
    role = previous_role(previous)
    formal_controller._event({"event": "stage_start", "fit_id": FIT_ID, "stage": stage,
                              "fg_cap": fg_cap, "bend": WEIGHTS["bend"],
                              "previous": str(previous), "previous_role": role,
                              "reference_opened": False, "phase_opened": False})
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.perf_counter()
    record = formal_controller.stage_fit(row, row["stages"][0], data, coordinates, p_init, q_init, None,
                                        initial_raw_y=raw_y)
    wall = time.perf_counter() - t0
    stage_path = RUN / "stages" / FIT_ID / ("%s.json" % stage)
    lineage_before = REUSED_PREFIX_FG + int(np.sum([STAGE_FG_CAP[s] for s in RUN_STAGES[:RUN_STAGES.index(stage)]]))
    terminal = {
        "schema": "p9016-round054-stage-terminal-v1",
        "fit_id": FIT_ID, "stage": stage, "bin_size_bp": bin_size, "fg_cap": fg_cap,
        "weights": dict(WEIGHTS), "status": record.get("status"),
        "terminal_reason": record.get("terminal_reason"),
        "outer_fg_actual": record.get("outer_fg_actual"),
        "last_accepted_endpoint": record.get("last_accepted_endpoint"),
        "canonical_gradient_max_abs": (record.get("optimizer") or {}).get("canonical_gradient_max_abs"),
        "wall_seconds": float(wall), "started_at_utc": started,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_loci": int(data.n_loci), "n_pairs": int(data.n_pairs),
        "raw_records": float(data.budget()["raw_records"]),
        "layer_aggregate": {
            "path": str(REUSED_AGGREGATES[bin_size][0].relative_to(ROOT)),
            "sha256": REUSED_AGGREGATES[bin_size][1], "reused_not_rebuilt": True,
        },
        "previous_endpoint": str(previous),
        "previous_endpoint_sha256": round_runner.sha256_file(previous),
        "previous_endpoint_role": role,
        "prefix_fg_reused": REUSED_PREFIX_FG if role == "reused_frozen_prefix" else 0,
        "new_fg_this_stage": fg_cap,
        "lineage_fg_total_through_this_stage": lineage_before + fg_cap,
        "start_coordinates_sha256": round_runner.array_sha256(coordinates),
        "start_raw_y_sha256": round_runner.array_sha256(raw_y),
        "start_sphere_mapping_max_abs_error": sphere_error,
        "start_p_init": float(p_init), "start_q_init": float(q_init),
        "warm_start": {"mode": warm["metadata"].get("mode"),
                       "candidate_base_seed": warm["metadata"].get("candidate_base_seed"),
                       "seed": warm["metadata"].get("seed"),
                       "perturbation": warm["metadata"].get("perturbation"),
                       "clipped_coordinates": warm["metadata"]["normalization"]["clipped_coordinates"],
                       "preserved_frame_and_scale": True},
        "endpoint": record.get("endpoint") or {},
        "artifact_hashes": record.get("artifact_hashes"),
        "stage_record_path": str(stage_path.relative_to(ROOT)),
        "stage_record_sha256": round_runner.sha256_file(stage_path) if stage_path.is_file() else None,
        "reference_opened": False, "phase_opened": False,
        "budget_honesty": "each layer has a deliberately small FG cap; exit code 0 is not convergence",
        "termination_semantics": "status/terminal_reason plus the canonical raw-y/q gradient is the "
                                 "scientific status; exit code only says the runner finished",
    }
    if record.get("status") == "failure":
        terminal["error"] = record.get("error")
        terminal["error_type"] = record.get("error_type")
    round_runner.write_json(RUN / "logs" / ("%s-%s.terminal.json" % (FIT_ID, stage)), terminal)
    formal_controller._event({"event": "stage_end", "fit_id": FIT_ID, "stage": stage,
                              "status": terminal["status"], "outer_fg_actual": terminal["outer_fg_actual"]})
    print(json.dumps({key: terminal.get(key) for key in (
        "stage", "status", "terminal_reason", "outer_fg_actual", "last_accepted_endpoint",
        "canonical_gradient_max_abs", "wall_seconds", "lineage_fg_total_through_this_stage")}, indent=2))
    return 0 if record.get("status") != "failure" else 2


if __name__ == "__main__":
    raise SystemExit(main())
