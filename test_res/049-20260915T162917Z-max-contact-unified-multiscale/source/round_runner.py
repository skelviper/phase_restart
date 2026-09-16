"""049 正式运行适配层。

只在新目录内做局部 monkey-patch，把 045 `formal_controller.stage_fit` 的 objective
工厂换成 A/B/C × raw/ms，并让旧 runner 的 canonical 梯度判定认识 multiscale wrapper。
旧 045/046 文件不改一字节；`formal_controller.RUN` 在本进程内被指向 049 运行目录。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from frozen_imports import data_io, formal_controller, m1_preconditioner
from max_contact_objective import LOSS_SPEC, MaxContactObjective
from multiscale_objective import MultiscaleObjective, make_preconditioner
from round_paths import (AGGREGATE_1MB, FIT_FG_CAP, FULL_WEIGHTS, INITIAL_CONSENSUS_1MB,
                         INITIAL_RANDOM_1MB, P_INIT, PAIR_BLOCK, RUN, SOURCES)
from shared_capture_objective import PenaltyWeights

INITIAL_PATHS = {"consensus": INITIAL_CONSENSUS_1MB, "random": INITIAL_RANDOM_1MB}
_ACTIVE: dict[str, Any] = {"loss": None, "solver": None, "preconditioner": None, "reference_opened": False}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray, dtype: str = "<f8") -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_data():
    data = data_io.load_aggregate(AGGREGATE_1MB)
    data.assert_consistent()
    if int(data.n_loci) != 2645 or int(data.n_pairs) != 3496690:
        raise RuntimeError("1Mb aggregate grid mismatch")
    if float(data.raw_records) != 1703888.0 or float(data.raw_cis_offdiag) != 696680.0 or float(data.raw_inter) != 568434.0:
        raise RuntimeError("1Mb aggregate denominator mismatch")
    return data


def load_initial(source: str) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    if source not in SOURCES:
        raise ValueError("unknown blind source")
    coordinates, raw_y, p_init, metadata = data_io.load_start(INITIAL_PATHS[source])
    if not bool(metadata.get("no_optimization")):
        raise RuntimeError("initial control is not a zero-optimization prolongation")
    if "014 approved blind root" not in str(metadata.get("source", "")):
        raise RuntimeError("initial control lineage is not the 014 approved blind root")
    return coordinates, raw_y, float(p_init), metadata


# --------------------------------------------------------------------- patch
def _patched_objective(data: Any, model_id: str, weights: PenaltyWeights, known_e: np.ndarray | None):
    loss = str(model_id).upper()
    solver = str(_ACTIVE.get("solver") or "")
    if loss not in LOSS_SPEC:
        raise RuntimeError("patched objective received an unknown loss id: %s" % model_id)
    if solver not in ("raw", "ms"):
        raise RuntimeError("active solver is not set")
    if known_e is not None:
        raise RuntimeError("real round must not receive known-e")
    mode = "V0-fixed-production-e" if data.count_mode == "raw_integer" else "V0-known-generating-e"
    base = MaxContactObjective(
        data, loss, weights=weights, mode=mode, device="cuda", pair_block=PAIR_BLOCK,
        inner_cap=80, cg_cap=80, profile_warm_start=True, known_e=known_e,
    )
    if solver == "ms":
        preconditioner = _ACTIVE.get("preconditioner")
        if preconditioner is None:
            preconditioner = make_preconditioner(data)
            _ACTIVE["preconditioner"] = preconditioner
        return MultiscaleObjective(base, preconditioner)
    return base


def _patched_canonical_gradient(objective: Any, theta: np.ndarray, optimizer_gradient: np.ndarray):
    hook = getattr(objective, "canonical_raw_gradient", None)
    if callable(hook):
        return np.asarray(hook(theta, optimizer_gradient), dtype=np.float64)
    return m1_preconditioner.__dict__["_ORIGINAL_CANONICAL_GRADIENT"](objective, theta, optimizer_gradient)


def install(run_dir: Path = RUN) -> None:
    """把本进程的 stage_fit 输出与 objective 工厂指向 049。"""
    if not hasattr(m1_preconditioner, "_ORIGINAL_CANONICAL_GRADIENT"):
        m1_preconditioner._ORIGINAL_CANONICAL_GRADIENT = m1_preconditioner._canonical_gradient
    m1_preconditioner._canonical_gradient = _patched_canonical_gradient
    formal_controller._objective = _patched_objective
    formal_controller.RUN = Path(run_dir)


def set_active(loss: str, solver: str) -> None:
    if loss not in LOSS_SPEC:
        raise ValueError("unknown loss")
    if solver not in ("raw", "ms"):
        raise ValueError("unknown solver")
    _ACTIVE["loss"] = loss
    _ACTIVE["solver"] = solver
    _ACTIVE["preconditioner"] = None


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def canonicalize_endpoint(fit_id_value: str, solver: str, run_dir: Path | None = None) -> dict[str, Any]:
    """在 stage 产物之外单独写 canonical npz（optimizer_theta 与 raw_y/q 分开标注）。"""
    base_dir = Path(run_dir) if run_dir is not None else RUN
    directory = base_dir / "coords" / fit_id_value
    stage_npz = directory / "1Mb.npz"
    payload = _read_npz(stage_npz)
    coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
    raw_y = np.asarray(payload["raw_y"], dtype=np.float64)
    theta = np.asarray(payload["theta"], dtype=np.float64)
    p = float(np.asarray(payload["p"]).item())
    q = float(np.asarray(payload["q"]).item())
    from frozen_imports import contact_model
    if solver == "ms":
        optimizer_theta = theta
        canonical_raw_theta = np.concatenate((raw_y.reshape(-1), np.asarray([q], dtype=np.float64)))
    else:
        optimizer_theta = theta
        canonical_raw_theta = theta
    sphere = contact_model.sphere_forward(raw_y)
    mapping_error = float(np.max(np.abs(sphere - coordinates)))
    target = directory / "1Mb.canonical.npz"
    np.savez_compressed(
        target,
        coordinates=coordinates,
        raw_y=raw_y,
        optimizer_theta=optimizer_theta,
        canonical_raw_theta=canonical_raw_theta,
        p=np.asarray(p, dtype=np.float64),
        q=np.asarray(q, dtype=np.float64),
        solver=np.asarray(solver),
    )
    readback = _read_npz(target)
    if not np.array_equal(readback["raw_y"], raw_y) or not np.array_equal(readback["optimizer_theta"], optimizer_theta):
        raise RuntimeError("canonical npz readback mismatch")
    return {
        "path": str(target.relative_to(base_dir)),
        "sha256": sha256_file(target),
        "raw_y_array_sha256": array_sha256(raw_y),
        "optimizer_theta_array_sha256": array_sha256(optimizer_theta),
        "coordinates_array_sha256": array_sha256(coordinates),
        "sphere_mapping_max_abs_error": mapping_error,
        "q": q,
        "p": p,
        "p_from_q_bit_exact": bool(abs(contact_model.p_from_q(q)[0] - p) == 0.0),
    }


def fit_row(loss: str, solver: str, source: str) -> dict[str, Any]:
    return {
        "fit_id": "%s-%s-%s" % (loss, solver, source),
        "model_id": loss,
        "loss_id": loss,
        "loss_name": LOSS_SPEC[loss]["name"],
        "solver_id": solver,
        "kind": "real",
        "candidate": source,
        "start_name": "real_%s_1Mb_zero_optimization_prolongation" % source,
        "objective_variant": "max-contact-%s/%s/full-J" % (LOSS_SPEC[loss]["name"], solver),
        "fixture": None,
        "weights": dict(FULL_WEIGHTS),
        "stages": [{"stage": "1Mb", "bin_size_bp": 1_000_000, "fg_cap": FIT_FG_CAP}],
    }


def repair_stage_artifacts(fit_id_value: str, loss: str, solver: str, data: Any,
                           run_dir: Path | None = None) -> dict[str, Any]:
    """修正旧 stage_fit 在 ms 分支下两处梯度标注（不改 045 文件，只改本目录产物）。

    (1) iter0 NPZ 把 initial optimizer 梯度同时写进 optimizer_gradient/canonical_gradient；
        ms 时 canonical 必须是 P^-1 gz。
    (2) record.endpoint.raw_y_q_gradient_inf 取的是 readback optimizer 梯度（ms 即 z 梯度），
        必须以真正的 canonical raw_y/q 值报告，另存 optimizer_gradient_inf。
    """
    base_dir = Path(run_dir) if run_dir is not None else RUN
    record_path = base_dir / "stages" / fit_id_value / "1Mb.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    weights = PenaltyWeights(**{key: float(value) for key, value in record["weights"].items()})
    probe = _patched_objective(data, loss, weights, None)

    iter0_path = base_dir / "checkpoints" / fit_id_value / "1Mb" / "accepted-00000.npz"
    payload = _read_npz(iter0_path)
    optimizer_theta = np.asarray(payload["theta"], dtype=np.float64)
    optimizer_gradient = np.asarray(payload["optimizer_gradient"], dtype=np.float64)
    canonical = (probe.canonical_raw_gradient(optimizer_theta, optimizer_gradient)
                 if solver == "ms" else optimizer_gradient.copy())
    np.savez_compressed(
        iter0_path, iteration=np.asarray(payload["iteration"], dtype=np.int64),
        nfev=np.asarray(payload["nfev"], dtype=np.int64), theta=optimizer_theta,
        raw_y=np.asarray(payload["raw_y"], dtype=np.float64), coordinates=np.asarray(payload["coordinates"], dtype=np.float64),
        optimizer_gradient=optimizer_gradient, canonical_gradient=canonical,
        solver=np.asarray(solver), loss=np.asarray(loss),
        gradient_space_optimizer=np.asarray("optimizer_variables"),
        gradient_space_canonical=np.asarray("raw_y_q"),
    )
    readback = _read_npz(iter0_path)
    if not np.array_equal(readback["canonical_gradient"], canonical):
        raise RuntimeError("iter0 canonical gradient readback mismatch")
    if not np.array_equal(readback["optimizer_gradient"], optimizer_gradient):
        raise RuntimeError("iter0 optimizer gradient readback mismatch")

    endpoint = dict(record.get("endpoint") or {})
    reported_optimizer_inf = float(endpoint.get("raw_y_q_gradient_inf", float("nan")))
    canonical_inf = float(endpoint.get("canonical_gradient_max_abs", float("nan")))
    if solver == "ms":
        endpoint["optimizer_gradient_inf"] = reported_optimizer_inf
        endpoint["raw_y_q_gradient_inf"] = canonical_inf
        endpoint["optimizer_gradient_inf_original_field_value"] = reported_optimizer_inf
    else:
        endpoint["optimizer_gradient_inf"] = canonical_inf
        endpoint["raw_y_q_gradient_inf"] = canonical_inf
    endpoint["gradient_reporting_repair"] = {
        "applied": True, "solver": solver,
        "method": "canonical raw_y/q gradient = P^-1 gz for the multiscale solver; record fields relabelled",
        "iter0_npz_canonical_gradient_source": "P^-1 applied to the stored optimizer gradient",
        "endpoint_field_rule": "raw_y_q_gradient_inf = result.canonical_gradient_max_abs; optimizer_gradient_inf = old readback value",
    }
    record["endpoint"] = endpoint
    record["artifact_repair"] = {
        "component": "049 round adapter (045 formal_controller.py unchanged)",
        "iter0_npz": str(iter0_path.relative_to(base_dir)),
        "iter0_npz_sha256": sha256_file(iter0_path),
        "canonical_gradient_inf": canonical_inf,
        "optimizer_gradient_inf": reported_optimizer_inf,
    }
    hashes = dict(record.get("artifact_hashes") or {})
    hashes["iter0_npz_sha256"] = sha256_file(iter0_path)
    record["artifact_hashes"] = hashes
    if isinstance(record.get("iter0_artifact"), dict):
        record["iter0_artifact"]["sha256"] = hashes["iter0_npz_sha256"]
        record["iter0_artifact"]["gradient_label_repair"] = True
    record_path.write_text(json.dumps(jsonable(record), sort_keys=True, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")
    return {"iter0_npz_sha256": hashes["iter0_npz_sha256"], "canonical_gradient_inf": canonical_inf,
            "optimizer_gradient_inf": reported_optimizer_inf,
            "record_sha256": sha256_file(record_path),
            "canonical_matches_pinv": bool(solver != "ms" or float(np.max(np.abs(
                canonical - probe.canonical_raw_gradient(optimizer_theta, optimizer_gradient)))) == 0.0)}


def run_one_fit(loss: str, solver: str, source: str) -> dict[str, Any]:
    """正式单 fit 入口：1Mb、1502 FG、全程 full grid。"""
    install()
    set_active(loss, solver)
    data = load_data()
    coordinates, raw_y, p_init, metadata = load_initial(source)
    q_init = float(__import__("frozen_imports").contact_model.q_from_p(p_init))
    row = fit_row(loss, solver, source)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    formal_controller._event({"event": "fit_start", "fit_id": row["fit_id"], "loss": loss, "solver": solver,
                              "source": source, "fg_cap": FIT_FG_CAP, "at_utc": started,
                              "reference_opened": False, "phase_opened": False})
    record = formal_controller.stage_fit(row, row["stages"][0], data, coordinates, p_init, q_init, None,
                                         initial_raw_y=raw_y)
    canonical = None
    repair = None
    if record.get("status") != "failure":
        repair = repair_stage_artifacts(row["fit_id"], loss, solver, data)
        canonical = canonicalize_endpoint(row["fit_id"], solver)
    summary = {
        "schema": "p9016-max-contact-fit-terminal-v1",
        "fit_id": row["fit_id"], "loss_id": loss, "loss_name": LOSS_SPEC[loss]["name"], "solver_id": solver,
        "source": source, "fg_cap": FIT_FG_CAP, "status": record.get("status"),
        "terminal_reason": record.get("terminal_reason"),
        "outer_fg_actual": record.get("outer_fg_actual"),
        "last_accepted_endpoint": record.get("last_accepted_endpoint"),
        "fit_wall_seconds": record.get("fit_wall_seconds"),
        "canonical_gradient_max_abs": (record.get("endpoint") or {}).get("canonical_gradient_max_abs"),
        "canonical_gradient_norm": (record.get("endpoint") or {}).get("canonical_gradient_norm"),
        "count_nll_normalized": ((record.get("endpoint") or {}).get("components") or {}).get("count_nll_normalized"),
        "total": (record.get("endpoint") or {}).get("total"),
        "p": (record.get("endpoint") or {}).get("p"),
        "q": (record.get("endpoint") or {}).get("q"),
        "started_at_utc": started,
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "initial_metadata": metadata,
        "initial_raw_y_source_sha256": array_sha256(raw_y),
        "initial_p": p_init,
        "initial_q": q_init,
        "initial_state_hashes": record.get("initial_state_hashes"),
        "record_path": str((RUN / "stages" / row["fit_id"] / "1Mb.json").relative_to(RUN)),
        "record_sha256": sha256_file(RUN / "stages" / row["fit_id"] / "1Mb.json") if repair else None,
        "canonical_artifact": canonical,
        "gradient_label_repair": repair,
        "reference_opened": False,
        "phase_opened": False,
        "exit_code_semantics": "0 means the runner terminated; scientific status is the terminal_reason",
    }
    write_json(RUN / "results" / "fits" / (row["fit_id"] + ".json"), summary)
    formal_controller._event({"event": "fit_end", "fit_id": row["fit_id"], "status": record.get("status"),
                              "terminal_reason": record.get("terminal_reason"),
                              "outer_fg_actual": record.get("outer_fg_actual"),
                              "at_utc": summary["completed_at_utc"]})
    return summary


__all__ = ["install", "set_active", "load_data", "load_initial", "run_one_fit", "fit_row",
           "canonicalize_endpoint", "sha256_file", "array_sha256", "write_json", "jsonable"]
