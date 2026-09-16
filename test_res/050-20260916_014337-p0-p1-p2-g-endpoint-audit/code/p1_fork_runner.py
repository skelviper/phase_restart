"""P1 fork runner（050 轮）：046 两个 G 基础 1Mb 端点 × {raw, ms}，每支 486 FG。

边界：

* 只读导入 045/049 的冻结训练侧实现（`formal_controller.stage_fit`、
  `m1_preconditioner.run_budgeted_lbfgs`、049 的 `MaxContactObjective` /
  `MultiscaleObjective` / `ChainPreconditioner`），不改旧目录任何字节。
* 起点来自 046 冻结 NPZ：`coordinates` / `raw_y` / `p` / `q` 逐元素精确携带；
  同一 base 的 raw 与 ms 两支共享同一份起点状态。
* 目标固定为 loss A（= 045 原 G，full-J），权重 count1/bond1/repulsion1/bend.01/p_prior1，
  production e、sphere、原 p/q 参数化、ftol=0、canonical raw_y/q gtol 1e-6、maxls=20、
  精确 FG cap、最后 accepted endpoint 约定，全部沿用原实现。
* 本进程不 import 任何 label / reference 模块，也不打开 phase 列或 3DG reference。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
ROOT = RUN_DIR.parents[1]
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
for _path in (str(S049),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import round_runner  # noqa: E402
from frozen_imports import contact_model, formal_controller  # noqa: E402
from max_contact_objective import MaxContactObjective  # noqa: E402
from multiscale_objective import MultiscaleObjective, check_roundtrip, make_preconditioner  # noqa: E402
from round_paths import AGGREGATE_1MB, AGGREGATE_1MB_SHA256  # noqa: E402
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective  # noqa: E402

FULL_WEIGHTS = {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0}
FG_CAP = 486
PAIR_BLOCK = 262_144
BASES = {
    "G-consensus": {
        "npz": ROOT / "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-consensus/1Mb.npz",
        "npz_sha256": "1a98d00842bc848433849251f334422dad78bd5be33e76b9b406ed4ece9f5c38",
    },
    "G-random": {
        "npz": ROOT / "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.npz",
        "npz_sha256": "6bb93bf570cf688fd8b02dccf14ca0867c1f831d5e123036ced626856f8d8752",
    },
}
FORBIDDEN_MODULES = ("pr.refeval", "pr.ref3dg", "pr.labels", "pr.reconstruction_report",
                     "pr.reconstruction_evaluate", "allele_r2", "allele_experiment")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray, dtype: str = "<f8") -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype=dtype)).tobytes(order="C")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(round_runner.jsonable(value), indent=2, sort_keys=True,
                               ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_base(base_id: str) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any]]:
    spec = BASES[base_id]
    path = Path(spec["npz"])
    actual = sha256_file(path)
    if actual != spec["npz_sha256"]:
        raise RuntimeError("base NPZ SHA256 mismatch for %s" % path)
    with np.load(path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        raw_y = np.asarray(payload["raw_y"], dtype=np.float64).copy()
        p = float(np.asarray(payload["p"]).item())
        q = float(np.asarray(payload["q"]).item())
        theta = np.asarray(payload["theta"], dtype=np.float64).copy()
    metadata = {
        "npz": str(path.relative_to(ROOT)), "npz_sha256": actual,
        "coordinates_shape": list(coordinates.shape),
        "coordinates_sha256": array_sha256(coordinates),
        "raw_y_sha256": array_sha256(raw_y),
        "q_float64_sha256": array_sha256(np.asarray([q], dtype=np.float64)),
        "p": p, "q": q,
        "theta_len": int(theta.shape[0]),
        "theta_tail_is_q": bool(theta.shape[0] == raw_y.size + 1 and theta[-1] == q),
        "theta_head_is_raw_y_flat": bool(theta.shape[0] == raw_y.size + 1
                                         and np.array_equal(theta[:-1], raw_y.reshape(-1))),
    }
    return coordinates, raw_y, p, q, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True, choices=("raw", "ms"))
    parser.add_argument("--base", required=True, choices=tuple(BASES))
    parser.add_argument("--fg-cap", type=int, default=FG_CAP)
    args = parser.parse_args()

    fit_id = "A-%s-%s" % (args.solver, args.base)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    p1_root = RUN_DIR / "p1"
    p1_root.mkdir(parents=True, exist_ok=True)
    gates_dir = p1_root / "gates"
    gate_path = gates_dir / ("%s.preflight.json" % fit_id)
    terminal_path = p1_root / "results" / ("%s.terminal.json" % fit_id)

    round_runner.install(p1_root)
    round_runner.set_active("A", args.solver)
    aggregate_sha = sha256_file(AGGREGATE_1MB)
    if aggregate_sha != AGGREGATE_1MB_SHA256:
        raise RuntimeError("frozen 1Mb aggregate SHA256 mismatch: %s" % aggregate_sha)
    data = round_runner.load_data()
    weights = PenaltyWeights(**FULL_WEIGHTS)
    mode = "V0-fixed-production-e" if data.count_mode == "raw_integer" else "V0-known-generating-e"
    coordinates, raw_y, p, q, base_meta = load_base(args.base)

    # ---- preflight gates -------------------------------------------------
    contact_model.assert_inside_unit_ball(coordinates)
    sphere = contact_model.sphere_forward(raw_y)
    sphere_error = float(np.max(np.abs(sphere - coordinates)))
    p_from_q, _ = contact_model.p_from_q(q)
    base_objective = MaxContactObjective(
        data, "A", weights=weights, mode=mode, device="cuda", pair_block=PAIR_BLOCK,
        inner_cap=80, cg_cap=80, profile_warm_start=True, known_e=None)
    g_objective = SharedCaptureObjective(
        data, model_id="G", weights=weights, mode=mode, device="cuda", pair_block=PAIR_BLOCK,
        inner_cap=80, cg_cap=80, profile_warm_start=True, known_e=None)
    theta = np.concatenate((raw_y.reshape(-1), np.asarray([q], dtype=np.float64)))
    value_a, grad_a, comp_a = base_objective.evaluate(theta, need_gradient=True)
    value_g, grad_g, comp_g = g_objective.evaluate(theta, need_gradient=True)
    parity = {
        "value_lossA": float(value_a), "value_045_G": float(value_g),
        "value_abs_diff": abs(float(value_a) - float(value_g)),
        "gradient_max_abs_diff": float(np.max(np.abs(np.asarray(grad_a) - np.asarray(grad_g)))),
        "count_nll_abs_diff": abs(float(comp_a["count_nll_normalized"]) - float(comp_g["count_nll_normalized"])),
        "count_nll_normalized": float(comp_a["count_nll_normalized"]),
        "total": float(comp_a["total"]),
        "note": "measured at THIS base start; the 049 gate1 value 9.542018998907452 belongs to the "
                "046 real-extension-G-full-J point and is not an expectation for this start",
    }
    preconditioner = make_preconditioner(data)
    roundtrip = float(check_roundtrip(data))
    y_probe = np.asarray(raw_y, dtype=np.float64)
    pre_roundtrip = float(np.max(np.abs(preconditioner.Pinv_apply(preconditioner.P_apply(y_probe)) - y_probe)))
    ms_objective = MultiscaleObjective(base_objective, preconditioner)
    theta_ms = ms_objective.pack(raw_y, p=p)
    theta_ms[-1] = q
    value_ms = float(ms_objective.evaluate(theta_ms, need_gradient=False)[0])
    _v, grad_ms, _c = ms_objective.evaluate(theta_ms, need_gradient=True)
    canonical_from_ms = ms_objective.canonical_raw_gradient(theta_ms, grad_ms)
    canonical_diff = float(np.max(np.abs(canonical_from_ms - np.asarray(grad_a))))

    gates = {
        "schema": "p9016-round050-p1-preflight-v1",
        "fit_id": fit_id, "solver": args.solver, "base": args.base, "fg_cap": int(args.fg_cap),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base": base_meta,
        "start_state": {
            "coordinates_sha256": array_sha256(coordinates),
            "raw_y_sha256": array_sha256(raw_y),
            "q_float64_sha256": array_sha256(np.asarray([q], dtype=np.float64)),
            "p": p, "q": q,
            "sphere_mapping_max_abs_error": sphere_error,
            "sphere_mapping_tolerance": 1e-10,
            "p_from_q_bit_exact": bool(p_from_q == p),
            "carried_exactly_from_frozen_npz": True,
            "note": "raw and ms arms are launched from this same file, so their start coordinates / "
                    "raw_y / q are bit-identical; the equality is re-asserted in the pair audit",
        },
        "loss_a_vs_045_g_parity_at_this_start": parity,
        "parity_status": "PASS" if (parity["value_abs_diff"] <= 1e-9
                                    and parity["gradient_max_abs_diff"] <= 1e-12
                                    and parity["count_nll_abs_diff"] <= 1e-9) else "FAIL",
        "multiscale": {
            "roundtrip_random_error": roundtrip,
            "roundtrip_at_start_max_abs_error": pre_roundtrip,
            "symbol_min": preconditioner.min_symbol, "symbol_max": preconditioner.max_symbol,
            "scales": list(preconditioner.scales),
            "value_ms_minus_value_a": value_ms - float(value_a),
            "canonical_gradient_vs_base_gradient_max_abs_diff": canonical_diff,
            "canonical_gradient_space": "raw_y/q",
            "status": "PASS" if (roundtrip <= 1e-12 and pre_roundtrip <= 1e-12
                                 and abs(value_ms - float(value_a)) <= 1e-9
                                 and canonical_diff <= 1e-9) else "FAIL",
        },
        "data": {"n_loci": int(data.n_loci), "n_pairs": int(data.n_pairs),
                 "count_mode": str(data.count_mode), "exposure_mode": str(data.exposure_mode),
                 "raw_records": float(data.raw_records), "weighted_mode": mode,
                 "aggregate_path": str(AGGREGATE_1MB.relative_to(ROOT)),
                 "aggregate_sha256_measured": aggregate_sha,
                 "aggregate_sha256_expected": AGGREGATE_1MB_SHA256,
                 "aggregate_sha256_match": bool(aggregate_sha == AGGREGATE_1MB_SHA256),
                 "n_pairs_expected": 3496690},
        "reference_opened": False, "phase_opened": False,
    }
    if aggregate_sha != AGGREGATE_1MB_SHA256:
        raise RuntimeError("frozen 1Mb aggregate SHA256 mismatch before fit")
    write_json(gate_path, gates)
    allowed_solvers = ("raw", "ms")
    if args.solver in allowed_solvers and gates["parity_status"] != "PASS":
        raise RuntimeError("loss A vs 045 G parity failed at %s" % args.base)
    if args.solver == "ms" and gates["multiscale"]["status"] != "PASS":
        raise RuntimeError("multiscale preflight failed at %s" % args.base)

    # ---- formal fit ------------------------------------------------------
    row = {
        "fit_id": fit_id, "model_id": "A", "kind": "real", "fixture": None,
        "candidate": args.base, "start_name": "046-frozen-base-%s" % args.base,
        "objective_variant": "loss-A-marginal_G/%s/full-J" % args.solver,
        "weights": dict(FULL_WEIGHTS),
        "stages": [{"stage": "1Mb", "bin_size_bp": 1_000_000, "fg_cap": int(args.fg_cap)}],
    }
    formal_controller._event({"event": "fit_start", "fit_id": fit_id, "loss": "A", "solver": args.solver,
                              "base": args.base, "fg_cap": int(args.fg_cap), "at_utc": started,
                              "reference_opened": False, "phase_opened": False})
    fit_started = time.perf_counter()
    record = formal_controller.stage_fit(row, row["stages"][0], data, coordinates, p, q, None,
                                         initial_raw_y=raw_y)
    wall = time.perf_counter() - fit_started
    repair = None
    canonical = None
    stage_record_path = p1_root / "stages" / fit_id / "1Mb.json"
    if record.get("status") != "failure":
        repair = round_runner.repair_stage_artifacts(fit_id, "A", args.solver, data, run_dir=p1_root)
        canonical = round_runner.canonicalize_endpoint(fit_id, args.solver, run_dir=p1_root)
        # repair 只改盘上的 stage JSON；内存 record 仍是修正前版本，必须重读后再写 terminal，
        # 否则 ms 支会把 z 梯度当成 raw_y_q_gradient_inf，且 optimizer_gradient_inf 为空。
        record = json.loads(stage_record_path.read_text(encoding="utf-8"))

    # ---- isolation audit (training process must never touch phase/reference) ----
    loaded_forbidden = sorted(name for name in sys.modules if name in FORBIDDEN_MODULES)
    isolation = {
        "forbidden_modules_loaded": loaded_forbidden,
        "training_process_clean": not loaded_forbidden,
        "reference_opened": False, "phase_opened": False,
        "note": "training-side process; labels/reference payloads are never opened here",
    }
    endpoint = dict(record.get("endpoint") or {})
    terminal = {
        "schema": "p9016-round050-p1-terminal-v1",
        "fit_id": fit_id, "solver": args.solver, "base": args.base,
        "loss_id": "A", "loss_name": "marginal_G", "target": "G/full-J",
        "weights": dict(FULL_WEIGHTS), "fg_cap": int(args.fg_cap),
        "status": record.get("status"), "terminal_reason": record.get("terminal_reason"),
        "outer_fg_actual": record.get("outer_fg_actual"),
        "last_accepted_endpoint": record.get("last_accepted_endpoint"),
        "fit_wall_seconds": float(wall),
        "canonical_gradient_max_abs": endpoint.get("canonical_gradient_max_abs"),
        "canonical_gradient_norm": endpoint.get("canonical_gradient_norm"),
        "raw_y_q_gradient_inf": endpoint.get("raw_y_q_gradient_inf"),
        "optimizer_gradient_inf": endpoint.get("optimizer_gradient_inf"),
        "count_nll_normalized": (endpoint.get("components") or {}).get("count_nll_normalized"),
        "total": endpoint.get("total"),
        "p": endpoint.get("p"), "q": endpoint.get("q"),
        "started_at_utc": started,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_frozen": {"ftol": 0.0, "canonical_gtol": 1e-6, "maxls": 20,
                          "fg_cap": int(args.fg_cap),
                          "sphere": "frozen contact_model unit ball",
                          "exposure": "fixed production e",
                          "preconditioner": ("P = I + (I+25L)^-1 + (I+400L)^-1, DCT-II ortho, z = P^-1 raw_y"
                                             if args.solver == "ms" else None)},
        "preflight_gate": str(gate_path.relative_to(RUN_DIR)),
        "gradient_label_correction": repair,
        "canonical_artifact": canonical,
        "terminal_source": {
            "stage_record_path": str(stage_record_path.relative_to(RUN_DIR)),
            "stage_record_sha256": sha256_file(stage_record_path) if stage_record_path.is_file() else None,
            "re_read_after_repair": bool(repair is not None),
            "rule": "endpoint fields are taken from the repaired on-disk stage JSON, so raw_y_q_gradient_inf is "
                    "the canonical raw_y/q gradient for both solvers and optimizer_gradient_inf keeps the "
                    "optimizer-space value; canonical_gradient_max_abs/norm are the original stage fields",
        },
        "aggregate_sha256_measured": aggregate_sha,
        "aggregate_sha256_expected": AGGREGATE_1MB_SHA256,
        "record_path": str(stage_record_path.relative_to(RUN_DIR)),
        "termination_semantics": "exit code 0 only means the runner terminated; the scientific status is "
                                 "status/terminal_reason and the canonical raw_y/q gradient",
        "isolation": isolation,
        "scope_statement": "local optimization diagnostic from a frozen 046 endpoint; not a new independent "
                           "blind reconstruction and not a biological replicate",
        "reference_opened": False, "phase_opened": False,
    }
    write_json(terminal_path, terminal)
    write_json(gates_dir / ("%s.isolation.json" % fit_id), isolation)
    formal_controller._event({"event": "fit_end", "fit_id": fit_id, "status": terminal["status"],
                              "terminal_reason": terminal["terminal_reason"],
                              "outer_fg_actual": terminal["outer_fg_actual"],
                              "at_utc": terminal["completed_at_utc"]})
    print(json.dumps({k: terminal[k] for k in (
        "fit_id", "status", "terminal_reason", "outer_fg_actual", "last_accepted_endpoint",
        "fit_wall_seconds", "canonical_gradient_max_abs", "count_nll_normalized", "total", "p", "q")},
        indent=2))
    return 0 if record.get("status") != "failure" else 2


if __name__ == "__main__":
    raise SystemExit(main())
