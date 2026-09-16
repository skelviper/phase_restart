"""052 新链运行器：40Mb -> 10Mb -> 5Mb -> 2Mb -> 1Mb -> 500kb -> 200kb。

每个进程只跑一层：读本链上一层端点 npz，按冻结 prohibition 规则 prolongation 到本层，
写出 coords/new-chain/<stage>.{npz,3dg}、stages/new-chain/<stage>.json 与
logs/new-chain-<stage>.terminal.json。

与既有 baseline / extra-levels 的差别只有两点：层结构和 FG 预算由 config.json 冻结；
objective / 优化器 / 停止判据 / 权重 / e 全部复用 045/049 的冻结实现（普通 raw L-BFGS，
非 multiscale，固定 production e，bend=0.01，其它权重 1）。

起点规则：40Mb 用 014 无标签 random 盲源（seed 2207）在真实 40,000,000 bp 网格上的展开；
其后每层只从本链上一层端点 prolongation，绝不在 5Mb 重置 baseline 起点。

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

from bootstrap_052 import (SharedCaptureObjective, contact_model, data_io, formal_controller,  # noqa: E402
                           reconstruction_init, round_runner)
from round_paths_052 import (FIT_ID, P_INIT, STAGES, STAGE_FG_CAP, WEIGHTS, stage_bin)  # noqa: E402

EXPECTED_1MB_SHA = "80984d804f8ae0552f6bab34a77a03e778073ac87f3b3ec6f04f96de74137420"


def layer_path(bin_size: int) -> Path:
    """本层 aggregate 的位置：1Mb / 20-2Mb 复用冻结文件，其余层是本 run 自己新建的。"""
    from round_paths_052 import AGGREGATE_051, AGGREGATE_1MB
    if int(bin_size) == 1_000_000:
        return Path(AGGREGATE_1MB)
    if int(bin_size) in AGGREGATE_051:
        return Path(AGGREGATE_051[int(bin_size)])
    return RUN / "inputs" / ("real_%d_aggregate.npz" % int(bin_size))


def load_layer(stage: str):
    bin_size = stage_bin(stage)
    path = layer_path(bin_size)
    if not path.is_file():
        raise RuntimeError("missing layer aggregate for stage %s: %s" % (stage, path))
    if bin_size == 1_000_000 and round_runner.sha256_file(path) != EXPECTED_1MB_SHA:
        raise RuntimeError("frozen 1Mb aggregate SHA mismatch")
    data = data_io.load_aggregate(path)
    if int(data.bin_size) != int(bin_size):
        raise RuntimeError("stage %s layer bin_size mismatch: %d != %d" % (stage, int(data.bin_size), bin_size))
    expected_bins = np.array([(int(L) + bin_size - 1) // bin_size for L in data.chromosome_lengths],
                             dtype=np.int64)
    if not np.array_equal(np.asarray(data.n_bins, dtype=np.int64), expected_bins):
        raise RuntimeError("stage %s does not sit on a real %d bp grid" % (stage, bin_size))
    if int(data.n_loci) != int(expected_bins.sum()):
        raise RuntimeError("stage %s grid size mismatch" % stage)
    if int(data.budget()["raw_records"]) != 1_703_888:
        raise RuntimeError("stage %s lost records" % stage)
    return data


def load_start(stage: str, previous: Path | None, data):
    """返回 (coordinates, raw_y, p, q)；只从本链上一层端点 prolongation，或 40Mb 零优化起点。"""
    bin_size = stage_bin(stage)
    if previous is None:
        if stage != "40Mb":
            raise RuntimeError("only the 40Mb layer may start from the zero-optimization root")
        path = RUN / "coords" / "initial" / "random_40Mb.npz"
        coordinates, raw_y, p_init, meta = data_io.load_start(path)
        if int(coordinates.shape[1]) != int(data.n_loci):
            raise RuntimeError("40Mb root grid mismatch")
        if not bool(meta.get("no_optimization")):
            raise RuntimeError("40Mb root is not a zero-optimization initialization")
        return np.asarray(coordinates, dtype=np.float64), np.asarray(raw_y, dtype=np.float64), \
            float(p_init), float(contact_model.q_from_p(float(p_init)))

    with np.load(previous, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        p_init = float(np.asarray(payload["p"]).item())
        q_init = float(np.asarray(payload["q"]).item())
    if coordinates.shape[1] == int(data.n_loci):
        raise RuntimeError("previous endpoint already has the target grid; chain prolongation expected")
    # 上一层的真实 bp 坐标（不是从 stage 名字推断）：locus_bin * previous bin size
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
    return coordinates, raw_y, float(p_init), float(q_init)


def stage_of_endpoint(path: Path) -> str:
    name = Path(path).stem
    for stage in STAGES:
        if stage == name:
            return stage
    raise RuntimeError("endpoint %s does not name a chain stage" % path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--previous", default=None, help="本链上一层端点 npz；40Mb 层省略")
    parser.add_argument("--preflight", action="store_true",
                        help="只构建本层 objective 并测显存/一次 evaluate，不做拟合")
    args = parser.parse_args()
    stage = str(args.stage)
    bin_size = stage_bin(stage)
    fg_cap = int(STAGE_FG_CAP[stage])
    previous = Path(args.previous) if args.previous else None
    if stage == "40Mb" and previous is not None:
        raise RuntimeError("the 40Mb layer is the chain root")

    round_runner.install(RUN)
    formal_controller.RUN = RUN
    data = load_layer(stage)
    if args.preflight:
        # 预检只关心本层 objective 的显存/单次 FG 成本，起点用一个零 raw 点，不读任何端点。
        raw_y = np.zeros((2, int(data.n_loci), 3), dtype=np.float64)
        coordinates = contact_model.sphere_forward(raw_y)
        p_init, q_init = float(P_INIT), float(contact_model.q_from_p(float(P_INIT)))
        sphere_error = 0.0
    else:
        coordinates, raw_y, p_init, q_init = load_start(stage, previous, data)
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

    if args.preflight:
        import torch
        objective = factory(data, "G", formal_controller._weights({"weights": WEIGHTS}), None)
        theta = objective.pack(raw_y, p=p_init)
        theta[-1] = q_init
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        value, gradient, components = objective.evaluate(theta, need_gradient=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        report = {
            "stage": stage, "bin_size_bp": bin_size, "n_loci": int(data.n_loci),
            "n_pairs": int(data.n_pairs), "raw_records": int(data.budget()["raw_records"]),
            "value": float(value), "gradient_finite": bool(np.all(np.isfinite(gradient))),
            "seconds_per_fg": float(elapsed),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "cuda_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "components_keys": sorted(components.keys()),
        }
        (RUN / "logs").mkdir(parents=True, exist_ok=True)
        (RUN / "logs" / ("preflight-%s.json" % stage)).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    row = {
        "fit_id": FIT_ID, "model_id": "G", "kind": "real", "fixture": None, "candidate": "random",
        "start_name": "052-new-chain",
        "objective_variant": "G/full-J/raw/fixed-production-e",
        "weights": dict(WEIGHTS),
        "stages": [{"stage": stage, "bin_size_bp": bin_size, "fg_cap": fg_cap}],
    }
    formal_controller._event({"event": "stage_start", "fit_id": FIT_ID, "stage": stage,
                              "fg_cap": fg_cap, "bend": WEIGHTS["bend"],
                              "previous": str(previous) if previous else None,
                              "reference_opened": False, "phase_opened": False})
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.perf_counter()
    record = formal_controller.stage_fit(row, row["stages"][0], data, coordinates, p_init, q_init, None,
                                        initial_raw_y=raw_y)
    wall = time.perf_counter() - t0
    stage_path = RUN / "stages" / FIT_ID / ("%s.json" % stage)
    terminal = {
        "schema": "p9016-round052-stage-terminal-v1",
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
        "previous_endpoint": str(previous) if previous else None,
        "previous_endpoint_sha256": (round_runner.sha256_file(previous) if previous else None),
        "start_coordinates_sha256": round_runner.array_sha256(coordinates),
        "start_raw_y_sha256": round_runner.array_sha256(raw_y),
        "start_sphere_mapping_max_abs_error": sphere_error,
        "start_p_init": float(p_init), "start_q_init": float(q_init),
        "endpoint": record.get("endpoint") or {},
        "artifact_hashes": record.get("artifact_hashes"),
        "stage_record_path": str(stage_path.relative_to(RUN.parents[1])),
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
        "canonical_gradient_max_abs", "wall_seconds")}, indent=2))
    return 0 if record.get("status") != "failure" else 2


if __name__ == "__main__":
    raise SystemExit(main())
