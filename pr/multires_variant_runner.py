"""Post-020 allele 变体的独立多分辨率 controller。

C0 复现仍由 ``pr.c0_controlled_runner`` 负责。本模块只进行准备，并在显式 C0 gate 通过后运行 C1/C2-map/C2-free/C3。每个变体/来源 pair 拥有完整的 5 Mb -> 2 Mb -> 1 Mb chain。这里不导入 evaluator、带 phase 的输入、reference coordinate 或 native FDG path。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from . import allele_models, contact_model, joint_fit, paired_run, reconstruct, reconstruction_init, reconstruction_report
from .gate import sha256_file

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve()
OLD_RUN = ROOT / "test_res" / "020-20260913_071841-v1-p9016-joint"
C0_RUN = ROOT / "test_res" / "033-20260914_044246-c0-controlled-reproduction"
C0_GATE = C0_RUN / "C0_gate.json"
C0_PREFLIGHT = C0_RUN / "preflight.json"
C0_PROTOCOL = C0_RUN / "controlled_protocol.json"

VARIANTS = ("C1", "C2-map", "C2-free", "C3")
ALL_VARIANTS = ("C0",) + VARIANTS
CANDIDATES = tuple(reconstruct.DEFAULT_CANDIDATES)
STAGES = tuple(reconstruct.DEFAULT_STAGES)
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
P_INIT = 0.75
CHECKPOINT_EVERY = 10
TIE_TOLERANCE = 1e-9
STRICT_ATOL = 1e-12
STRICT_RTOL = 1e-10
FD_ATOL = 1e-5
FD_RTOL = 5e-5

EXPECTED_LAYER_BUDGETS = {
    "5m": {
        "bin_size_bp": 5_000_000,
        "n_loci": 538,
        "n_eligible_pairs": 144_453,
        "n_zero_eligible_pairs": 22_184,
        "raw_same_bin": 607_552,
        "raw_cis_offdiag": 527_902,
        "raw_inter": 568_434,
    },
    "2m": {
        "bin_size_bp": 2_000_000,
        "n_loci": 1_329,
        "n_eligible_pairs": 882_456,
        "n_zero_eligible_pairs": 534_496,
        "raw_same_bin": 516_046,
        "raw_cis_offdiag": 619_408,
        "raw_inter": 568_434,
    },
    "1m": {
        "bin_size_bp": 1_000_000,
        "n_loci": 2_645,
        "n_eligible_pairs": 3_496_690,
        "n_zero_eligible_pairs": 3_009_436,
        "raw_same_bin": 438_774,
        "raw_cis_offdiag": 696_680,
        "raw_inter": 568_434,
    },
}

SOURCE_PATHS = (
    "pr/multires_variant_runner.py",
    "pr/allele_models.py",
    "pr/paired_run.py",
    "pr/contact_model.py",
    "pr/joint_fit.py",
    "pr/reconstruction_init.py",
    "pr/reconstruct.py",
    "pr/reconstruction_report.py",
    "pr/genome.py",
    "pr/pairs7.py",
    "pr/paths.py",
    "pr/gate.py",
)
COMPARATOR_PATHS = (
    OLD_RUN / "config.json",
    OLD_RUN / "selection.json",
    OLD_RUN / "termination_audit.json",
    *(OLD_RUN / "stages" / candidate.candidate_id / (stage.label + ".json")
      for candidate in CANDIDATES for stage in STAGES),
)

class VariantRunnerError(RuntimeError):
    """冻结的变体或 release 契约违反时抛出。"""

@dataclass(frozen=True)
class VariantPlan:
    variant: str
    candidate_id: str
    stage: reconstruct.StageSpec

    @property
    def attempt_id(self) -> str:
        return "%s-%s-%s" % (self.variant, self.candidate_id, self.stage.label)

def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VariantRunnerError("nonfinite value cannot enter formal metadata")
        return value
    return value

def _json_bytes(value: Any) -> bytes:
    return (json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

def _write_exclusive(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()

def _write_bytes_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()

def _write_atomic_json(path: Path, value: Any) -> str:
    """替换可变的 status metadata，但不削弱 exclusive artifacts。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()

def _sha256_array(value: np.ndarray, dtype: str | None = None) -> str:
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(np.dtype(dtype), copy=False)
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()

def _max_abs(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b), initial=0.0))

def _within(left: float, right: float, atol: float = STRICT_ATOL,
            rtol: float = STRICT_RTOL) -> bool:
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=rtol))

def _set_threads(threads: int) -> None:
    if int(threads) != 1:
        raise VariantRunnerError("variant controller is frozen to one BLAS/OpenMP thread")
    for name in THREAD_ENV:
        os.environ[name] = "1"

def _fresh_root(path: str | os.PathLike[str]) -> Path:
    requested = Path(path)
    root = requested if requested.is_absolute() else ROOT / requested
    root = root.resolve()
    if os.path.lexists(root):
        raise FileExistsError("refusing to reuse existing variant output: %s" % root)
    root.mkdir(parents=True)
    return root

def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))

def _source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise VariantRunnerError("missing source required by variant controller: %s" % path)
        result[relative] = sha256_file(path)
    return result

def _comparator_hashes() -> list[dict[str, Any]]:
    rows = []
    for path in COMPARATOR_PATHS:
        if not path.is_file():
            raise VariantRunnerError("missing 020 comparator file: %s" % path)
        rows.append({"path": _relative(ROOT, path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return rows

def _validate_context(context: reconstruct.RunContext) -> None:
    reconstruct._validate_context(context)
    if tuple(stage.label for stage in context.stages) != ("5m", "2m", "1m"):
        raise VariantRunnerError("variant schedule must be 5m, 2m, then 1m")
    if tuple(stage.maxiter for stage in context.stages) != (300, 200, 240):
        raise VariantRunnerError("variant maxiter must be 300/200/240")
    if tuple(candidate.candidate_id for candidate in context.candidates) != ("consensus_joint", "random_joint"):
        raise VariantRunnerError("variant candidate order changed")

def _expected_positions(data: contact_model.AggregatedContacts) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(data.locus_bin, dtype=np.int64) * int(data.bin_size),
            np.asarray(data.locus_chromosome, dtype=np.int64))

def _validate_layer(data: contact_model.AggregatedContacts,
                    context: reconstruct.RunContext,
                    stage: reconstruct.StageSpec) -> dict[str, Any]:
    data.assert_consistent()
    if int(data.bin_size) != int(stage.bin_size):
        raise VariantRunnerError("%s aggregate bin size changed" % stage.label)
    if tuple(data.chromosome_names) != tuple(name for name, _ in context.headers):
        raise VariantRunnerError("%s chromosome order changed" % stage.label)
    if tuple(int(v) for v in data.chromosome_lengths) != tuple(length for _, length in context.headers):
        raise VariantRunnerError("%s chromosome lengths changed" % stage.label)
    audit = data.budget()
    if context.strict_p9016:
        expected = EXPECTED_LAYER_BUDGETS[stage.label]
        for key, value in expected.items():
            actual = int(data.bin_size) if key == "bin_size_bp" else int(audit[key])
            if actual != int(value):
                raise VariantRunnerError("%s budget %s=%d expected %d" %
                                         (stage.label, key, actual, int(value)))
        required = {
            "raw_records": 1_703_888,
            "raw_same_bin": expected["raw_same_bin"],
            "raw_cis_offdiag": expected["raw_cis_offdiag"],
            "raw_inter": 568_434,
            "aggregate_same_bin": expected["raw_same_bin"],
            "aggregate_cis_offdiag": expected["raw_cis_offdiag"],
            "aggregate_inter": 568_434,
            "endpoint_total": 3_407_776,
            "n_diag_bins": expected["n_loci"],
        }
        for key, value in required.items():
            if int(audit[key]) != int(value):
                raise VariantRunnerError("%s required budget %s changed" % (stage.label, key))
    if not (audit["raw_conserved"] and audit["aggregate_conserved"] and audit["endpoint_conserved"]):
        raise VariantRunnerError("%s aggregate budget is not conserved" % stage.label)
    expected_exposure = np.sqrt(np.asarray(data.endpoint_counts, dtype=np.float64) + 10.0)
    expected_exposure /= expected_exposure.mean()
    exposure_error = _max_abs(data.exposure, expected_exposure)
    if exposure_error != 0.0:
        raise VariantRunnerError("%s exposure formula changed" % stage.label)
    return {"audit": _jsonable(audit), "exposure_formula_max_abs": exposure_error,
            "array_sha256": {
                "pair_i": _sha256_array(data.pair_i, "<i4"),
                "pair_j": _sha256_array(data.pair_j, "<i4"),
                "cis_pair": _sha256_array(data.cis_pair, "|b1"),
                "counts": _sha256_array(data.counts, "<i8"),
                "diag_counts": _sha256_array(data.diag_counts, "<i8"),
                "endpoint_counts": _sha256_array(data.endpoint_counts, "<i8"),
                "exposure": _sha256_array(data.exposure, "<f8"),
                "locus_chromosome": _sha256_array(data.locus_chromosome, "<i4"),
                "locus_bin": _sha256_array(data.locus_bin, "<i8"),
                "n_bins": _sha256_array(data.n_bins, "<i8"),
                "offsets": _sha256_array(data.offsets, "<i8"),
            }}

def _validate_model_coordinates(model_id: str, coordinates: np.ndarray,
                                 n_loci: int) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (2, n_loci, 3) or not np.all(np.isfinite(values)):
        raise VariantRunnerError("%s coordinates have invalid shape or nonfinite values" % model_id)
    allele_models.validate_physical_for_model(model_id, values)
    return values

def _track_name(chromosome_index: int, copy_index: int) -> str:
    return "c%02d%s" % (chromosome_index + 1, "ab"[copy_index])

def transfer_layer(coords: np.ndarray, positions: np.ndarray,
                   chromosome_index: np.ndarray, names: Sequence[str],
                   lengths: Sequence[int], bin_size: int, candidate_base_seed: int,
                   model_id: str) -> dict[str, Any]:
    """复制冻结的 warm-start 插值，并显式处理 free 域。

    插值、重复项处理、噪声分配和 seed 与 ``reconstruction_init.warm_start_from_layer`` 的操作相同。只有最终径向操作不同：有界模型使用注册的 1-1e-6 clip；C2-free 有意不执行 sphere inverse、clip 或 rescale。
    """
    layout = reconstruction_init.full_grid_layout(names, lengths, int(bin_size))
    incoming, source_positions, source_chromosome = reconstruction_init._coerce_layer_arrays(
        coords, positions, chromosome_index)
    if (source_chromosome >= len(layout["names"])).any():
        raise VariantRunnerError("warm-start chromosome index exceeds header inventory")
    out = np.empty((2, layout["n_loci"], 3), dtype=np.float64)
    perturb_mask = np.zeros((2, layout["n_loci"]), dtype=bool)
    per_track = {}
    for chromosome, length in enumerate(layout["header_lengths"]):
        target_slice = slice(layout["locus_offsets"][chromosome], layout["locus_offsets"][chromosome + 1])
        target_positions = layout["positions"][target_slice]
        for copy in (0, 1):
            keep = source_chromosome[copy] == chromosome
            if not keep.any():
                raise VariantRunnerError("warm-start is missing track %s" % _track_name(chromosome, copy))
            raw_positions = source_positions[copy, keep]
            raw_coords = incoming[copy, keep]
            rows = list(zip(raw_positions.tolist(), raw_coords))
            unique, averaged, counts = reconstruction_init._normalise_track_rows(
                rows, length, _track_name(chromosome, copy))
            values, audit, new_or_repeated = reconstruction_init._interpolate_track(
                unique, averaged, counts, target_positions)
            out[copy, target_slice] = values
            perturb_mask[copy, target_slice] = new_or_repeated
            audit = dict(audit)
            audit.update({"source_track": _track_name(chromosome, copy),
                          "target_track": _track_name(chromosome, copy),
                          "source_copy_reused": False,
                          "perturb_loci": int(new_or_repeated.sum())})
            per_track[_track_name(chromosome, copy)] = audit
    l0 = float((2 * layout["n_loci"]) ** (-1.0 / 3.0))
    perturb_scale = 0.025 * l0
    seed = 3301 + int(layout["bin_size"]) // 1_000_000 + int(candidate_base_seed)
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=perturb_scale, size=out.shape)
    out[perturb_mask] += noise[perturb_mask]
    clip_applied = False
    clip_limit = None
    if model_id != "C2-free":
        clip_limit = 1.0 - 1e-6
        radii = np.linalg.norm(out, axis=2)
        clipped = radii >= clip_limit
        if clipped.any():
            out[clipped] *= (clip_limit / radii[clipped])[:, None]
            clip_applied = True
        clipped_count = int(clipped.sum())
    else:
        clipped_count = 0
    metadata = {
        "mode": "multiresolution_warm_start",
        "model_id": str(model_id),
        "candidate_base_seed": int(candidate_base_seed),
        "seed": int(seed),
        "full_grid_from_zero": True,
        "l0": l0,
        "perturbation": {"scale": perturb_scale,
                          "perturbed_coordinates": int(perturb_mask.sum()),
                          "rule": "new_or_repeated_coordinates_only"},
        "normalization": {"mode": "preserve_previous_all_cell_frame_and_scale",
                           "incoming_max_radius": float(np.linalg.norm(incoming, axis=2).max()),
                           "radial_clip_limit": clip_limit,
                           "clipped_coordinates": clipped_count,
                           "clip_applied": clip_applied,
                           "free_domain_no_clip": model_id == "C2-free"},
        "per_track": per_track,
        "elapsed_sec": 0.0,
        "replicate_interpretation": "optimization initialization only; not a biological replicate",
    }
    return {"coords": out, "positions": layout["positions"].copy(),
            "chromosome_index": layout["chromosome_index"].copy(),
            "names": tuple(layout["names"]), "header_lengths": tuple(layout["header_lengths"]),
            "bin_size": int(layout["bin_size"]), "locus_offsets": tuple(layout["locus_offsets"]),
            "metadata": metadata}

def _map_diagnostics(objective: contact_model.JointObjective) -> dict[str, Any]:
    method = getattr(objective, "map_diagnostics", None)
    if method is None:
        result = {
            "map": "sphere_forward_unit_ball",
            "physical_domain": "strict_unit_ball",
            "objective_eval_count": None,
            "nonidentity_map_eval_count": None,
            "nonidentity_bead_eval_count": None,
            "nonidentity_map_eval_fraction": None,
            "max_raw_radius_seen": None,
            "max_physical_radius_seen": None,
            "line_search_probes_included": "shared_objective_not_instrumented",
        }
    else:
        result = dict(method())
    maximum = result.get("max_physical_radius_seen")
    result["ever_physical_radius_gt1"] = bool(maximum is not None and float(maximum) > 1.0)
    result["used_for_selection_or_budget"] = False
    return _jsonable(result)

def _checkpoint_write(path: Path, checkpoint: Any, history: Sequence[Mapping[str, Any]],
                      positions: np.ndarray, chromosome_index: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(list(history)), sort_keys=True, allow_nan=False)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            coordinates=np.asarray(checkpoint.coordinates, dtype=np.float64),
            theta=np.asarray(checkpoint.theta, dtype=np.float64),
            y=np.asarray(checkpoint.y, dtype=np.float64),
            positions=np.asarray(positions, dtype=np.int64),
            chromosome_index=np.asarray(chromosome_index, dtype=np.int64),
            fullhistory_json=np.asarray(payload),
        )
        handle.flush()
        os.fsync(handle.fileno())

def _solver_status(result: Any, stage: reconstruct.StageSpec) -> tuple[str, bool, str]:
    nit = int(getattr(result, "nit", 0) or 0)
    actual_nfev = int(getattr(result, "actual_nfev", getattr(result, "nfev", 0)) or 0)
    message = str(getattr(result, "message", ""))
    upper = message.upper()
    iteration_limit = nit >= stage.maxiter or "TOTAL NO. OF ITERATIONS REACHED LIMIT" in upper
    function_limit = actual_nfev >= stage.maxfun or ("TOTAL NO. OF F" in upper and "EVALUATIONS EXCEEDS LIMIT" in upper)
    budget = iteration_limit or function_limit
    if bool(getattr(result, "success", False)):
        return "converged", False, "solver_reported_success"
    if budget:
        return "not_converged", True, "budget_exhausted"
    return "not_converged", False, "solver_reported_nonconvergence"

def _fit_free_domain_aware(objective: contact_model.JointObjective, initial_y: np.ndarray,
                           p_init: float = P_INIT, q_init: float | None = None,
                           maxiter: int = 50, maxfun: int | None = None,
                           maxls: int = 20, ftol: float = 1e-12, gtol: float = 1e-7,
                           callback: Callable[[dict], None] | None = None,
                           checkpoint_every: int | None = None,
                           checkpoint_hook: Callable[[joint_fit.JointCheckpoint], None] | None = None) -> joint_fit.JointFitResult:
    """冻结的 ``joint_fit`` 循环，但去除仅有界模式的最终断言。"""
    if maxiter <= 0 or (maxfun is not None and maxfun <= 0):
        raise ValueError("maxiter/maxfun must be positive")
    if checkpoint_hook is not None and checkpoint_every is None:
        checkpoint_every = 10
    started = time.perf_counter()
    theta0 = objective.pack(initial_y, p=p_init)
    if q_init is not None:
        theta0[-1] = float(q_init)
    initial_value, _, initial_components = objective.evaluate(theta0, need_gradient=False)
    history = [{"iteration": 0, "nfev": 0, "elapsed_seconds": time.perf_counter() - started,
                "fun": float(initial_value), "p": float(initial_components["p"]),
                "components": initial_components}]
    actual_nfev = 0
    last_gradient = None
    last_theta = None

    def loss_and_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal actual_nfev, last_gradient, last_theta
        actual_nfev += 1
        value, gradient = objective.value_and_grad(theta)
        last_theta = np.asarray(theta, dtype=np.float64).copy()
        last_gradient = np.asarray(gradient, dtype=np.float64).copy()
        return float(value), np.asarray(gradient, dtype=np.float64)

    def recorder(theta: np.ndarray) -> None:
        cached = objective.cached_value_and_components(theta)
        if cached is None:
            value, _, components = objective.evaluate(theta, need_gradient=False)
        else:
            value, components = cached
        if last_theta is not None and last_gradient is not None and np.array_equal(theta, last_theta):
            gradient_norm = float(np.linalg.norm(last_gradient))
        else:
            _, gradient, _ = objective.evaluate(theta, need_gradient=True)
            gradient_norm = float(np.linalg.norm(gradient))
        iteration = len(history)
        elapsed = time.perf_counter() - started
        entry = {"iteration": iteration, "nfev": actual_nfev, "elapsed_seconds": elapsed,
                 "fun": float(value), "p": float(components["p"]),
                 "gradient_norm": gradient_norm, "components": components}
        history.append(entry)
        if checkpoint_hook is not None and checkpoint_every is not None and iteration % checkpoint_every == 0:
            raw, _ = objective.unpack(theta)
            coordinates, p = objective.coordinates_and_p(theta)
            checkpoint_hook(joint_fit.JointCheckpoint(
                iteration=iteration, nfev=actual_nfev, elapsed_seconds=elapsed,
                fun=float(value), p=float(p), theta=np.asarray(theta, dtype=np.float64).copy(),
                y=raw.copy(), coordinates=coordinates.copy(), components=dict(components),
                gradient_norm=gradient_norm,
            ))
        if callback is not None:
            callback(entry)

    options = {"maxiter": int(maxiter), "maxls": int(maxls), "ftol": float(ftol), "gtol": float(gtol)}
    if maxfun is not None:
        options["maxfun"] = int(maxfun)
    raw_result: OptimizeResult = minimize(loss_and_gradient, theta0, method="L-BFGS-B", jac=True,
                                           callback=recorder, options=options)
    raw_final, _ = objective.unpack(raw_result.x)
    coordinates, p = objective.coordinates_and_p(raw_result.x)
    objective.validate_physical_coordinates(coordinates)  # type: ignore[attr-defined]
    cached = objective.cached_value_and_components(raw_result.x)
    if cached is None:
        final_value, _, components = objective.evaluate(raw_result.x, need_gradient=False)
    else:
        final_value, components = cached
    elapsed = time.perf_counter() - started
    if not history or history[-1]["fun"] != float(final_value):
        history.append({"iteration": len(history), "nfev": actual_nfev,
                        "elapsed_seconds": elapsed, "fun": float(final_value),
                        "p": float(p), "components": components})
    scipy_nfev = int(getattr(raw_result, "nfev", actual_nfev))
    return joint_fit.JointFitResult(
        objective=objective, theta=np.asarray(raw_result.x, dtype=np.float64).copy(),
        y=raw_final.copy(), coordinates=np.asarray(coordinates, dtype=np.float64).copy(),
        p=float(p), fun=float(final_value), components=components,
        success=bool(raw_result.success), status=int(raw_result.status), message=str(raw_result.message),
        nit=int(raw_result.nit), nfev=int(actual_nfev), actual_nfev=int(actual_nfev),
        scipy_nfev=scipy_nfev, njev=int(getattr(raw_result, "njev", 0) or 0),
        elapsed_seconds=elapsed, history=history,
    )

def _fit_layer(root: Path, context: reconstruct.RunContext, model_id: str,
               candidate: reconstruct.CandidateSpec, stage: reconstruct.StageSpec,
               data: contact_model.AggregatedContacts, initial_coordinates: np.ndarray,
               positions: np.ndarray, chromosome_index: np.ndarray, q_init: float | None,
               fit_callable: Callable[..., Any] | None = None) -> tuple[Any, dict[str, Any], np.ndarray]:
    model_data = allele_models.data_for_model(data, model_id)
    objective = allele_models.objective_for_model(model_data, model_id)
    initial_coordinates = _validate_model_coordinates(model_id, initial_coordinates, data.n_loci)
    raw_initial = allele_models.raw_coordinates_from_physical(objective, initial_coordinates)
    theta0 = objective.pack(raw_initial, p=P_INIT)
    if q_init is not None:
        theta0[-1] = float(q_init)
    initial_total, _, initial_components = objective.evaluate(theta0, need_gradient=False)
    live_history: list[dict[str, Any]] = [{
        "iteration": 0, "nfev": 0, "elapsed_seconds": 0.0, "fun": float(initial_total),
        "p": float(initial_components["p"]), "components": _jsonable(dict(initial_components)),
    }]
    accepted_activity: list[dict[str, Any]] = []
    checkpoint_paths: list[str] = []

    def callback(entry: Mapping[str, Any]) -> None:
        item = _jsonable(dict(entry))
        item["map_diagnostics"] = _map_diagnostics(objective)
        live_history.append(item)
        accepted_activity.append({"iteration": int(item["iteration"]),
                                  "nfev": int(item["nfev"]),
                                  "map_diagnostics": item["map_diagnostics"]})

    def checkpoint_hook(checkpoint: joint_fit.JointCheckpoint) -> None:
        entry = {"iteration": int(checkpoint.iteration), "nfev": int(checkpoint.nfev),
                 "elapsed_seconds": float(checkpoint.elapsed_seconds), "fun": float(checkpoint.fun),
                 "p": float(checkpoint.p), "components": _jsonable(dict(checkpoint.components))}
        history = list(live_history)
        if not history or int(history[-1]["iteration"]) != entry["iteration"]:
            history.append(entry)
        path = root / "checkpoints" / model_id / candidate.candidate_id / (
            "%s-accepted-%04d.npz" % (stage.label, checkpoint.iteration))
        _checkpoint_write(path, checkpoint, history, positions, chromosome_index)
        checkpoint_paths.append(_relative(root, path))

    started = time.perf_counter()
    if fit_callable is not None:
        fit = fit_callable
    elif model_id == "C2-free":
        fit = _fit_free_domain_aware
    else:
        # C0/C1/C2-map/C3 保留完全相同的原始 fit_joint entry point。
        fit = joint_fit.fit_joint
    result = fit(
        objective, raw_initial, p_init=P_INIT, q_init=q_init,
        maxiter=stage.maxiter, maxfun=stage.maxfun, maxls=20,
        ftol=1e-10, gtol=1e-6, callback=callback,
        checkpoint_every=CHECKPOINT_EVERY, checkpoint_hook=checkpoint_hook,
    )
    elapsed = time.perf_counter() - started
    final_coordinates = _validate_model_coordinates(
        model_id, np.asarray(getattr(result, "coordinates"), dtype=np.float64), data.n_loci)
    result_theta = np.asarray(getattr(result, "theta"), dtype=np.float64)
    if result_theta.shape != (objective.n_parameters,) or not np.all(np.isfinite(result_theta)):
        raise VariantRunnerError("%s/%s returned invalid theta" % (model_id, stage.label))
    final_components = getattr(result, "components", None)
    if not isinstance(final_components, Mapping) or "total" not in final_components:
        raise VariantRunnerError("%s/%s result lacks objective components" % (model_id, stage.label))
    final_total = float(getattr(result, "fun"))
    if final_total > float(initial_total) + 1e-10 * max(1.0, abs(float(initial_total))):
        raise VariantRunnerError("%s/%s final objective increased" % (model_id, stage.label))
    status, budget_exhausted, reason = _solver_status(result, stage)
    history = _jsonable(getattr(result, "history", live_history))
    final_diag = _map_diagnostics(objective)
    fit_payload = {
        "status": status,
        "termination_reason": reason,
        "budget_exhausted": budget_exhausted,
        "initial_total": float(initial_total),
        "final_total": final_total,
        "initial_components": _jsonable(dict(initial_components)),
        "final_components": _jsonable(dict(final_components)),
        "p": float(getattr(result, "p")),
        "q_in": None if q_init is None else float(q_init),
        "q_out": float(result_theta[-1]),
        "solver": {
            "success": bool(getattr(result, "success")), "status": int(getattr(result, "status", 0)),
            "message": str(getattr(result, "message", "")), "nit": int(getattr(result, "nit", 0)),
            "nfev": int(getattr(result, "nfev", 0)),
            "actual_nfev": int(getattr(result, "actual_nfev", getattr(result, "nfev", 0))),
            "scipy_nfev": int(getattr(result, "scipy_nfev", getattr(result, "nfev", 0))),
            "njev": int(getattr(result, "njev", 0)), "elapsed_seconds": float(getattr(result, "elapsed_seconds", elapsed)),
            "maxiter": stage.maxiter, "maxfun": stage.maxfun, "maxls": 20,
            "ftol": 1e-10, "gtol": 1e-6,
        },
        "checkpoint_paths": checkpoint_paths,
        "history": history,
        "mapping_diagnostics": final_diag,
        "accepted_mapping_diagnostics": accepted_activity,
        "model_id": model_id,
        "physical_domain": allele_models.model_spec(model_id).physical_domain,
        "coordinate_parameterization": allele_models.model_spec(model_id).coordinate_parameterization,
    }
    return result, fit_payload, final_coordinates

def _write_stage(root: Path, model_id: str, candidate_id: str, stage: str,
                 payload: Mapping[str, Any]) -> str:
    path = root / "stages" / model_id / candidate_id / (stage + ".json")
    _write_exclusive(path, payload)
    return _relative(root, path)

def _write_candidate_status(root: Path, model_id: str, candidate_id: str,
                            payload: Mapping[str, Any]) -> None:
    _write_atomic_json(root / "run_status" / "candidates" / (model_id + "__" + candidate_id + ".json"), payload)

def _stage_log(root: Path, model_id: str, candidate_id: str, stage: str, message: str) -> None:
    path = root / "logs" / (model_id + "-" + candidate_id + "-" + stage + ".log")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("at", encoding="utf-8") as handle:
        handle.write("[%s] %s\n" % (dt.datetime.now().strftime("%H:%M:%S"), message))

def _validate_initialization(state: Mapping[str, Any], data: contact_model.AggregatedContacts,
                             model_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    coordinates = _validate_model_coordinates(model_id, np.asarray(state["coords"], dtype=np.float64), data.n_loci)
    positions, chromosome_index = _expected_positions(data)
    if not np.array_equal(np.asarray(state.get("positions"), dtype=np.int64), positions):
        raise VariantRunnerError("%s initial positions are not the complete origin-0 grid" % model_id)
    if not np.array_equal(np.asarray(state.get("chromosome_index"), dtype=np.int64), chromosome_index):
        raise VariantRunnerError("%s initial chromosome grid changed" % model_id)
    if tuple(state.get("names")) != tuple(data.chromosome_names):
        raise VariantRunnerError("%s initial chromosome names changed" % model_id)
    if tuple(int(v) for v in state.get("header_lengths")) != tuple(int(v) for v in data.chromosome_lengths):
        raise VariantRunnerError("%s initial chromosome lengths changed" % model_id)
    if int(state.get("bin_size")) != int(data.bin_size):
        raise VariantRunnerError("%s initial bin size changed" % model_id)
    return coordinates, positions, chromosome_index, _jsonable(dict(state.get("metadata", {})))

def _run_chain(root: Path, context: reconstruct.RunContext, config: Mapping[str, Any],
               model_id: str, candidate: reconstruct.CandidateSpec,
               load_data: Callable[[int], contact_model.AggregatedContacts] | None = None,
               initialize: Callable[..., Mapping[str, Any]] | None = None,
               warm_start: Callable[..., Mapping[str, Any]] | None = None,
               fit_callable: Callable[..., Any] | None = None) -> dict[str, Any]:
    load_data = contact_model.load_frozen_p9016_aggregate if load_data is None else load_data
    initialize = reconstruction_init.initialize_approved_candidate if initialize is None else initialize
    names = tuple(name for name, _ in context.headers)
    lengths = tuple(int(length) for _, length in context.headers)
    attempts: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    previous_coordinates = None
    previous_positions = None
    previous_chromosome = None
    carried_q = None
    final_path = None
    final_sha = None
    final_q = None
    _write_candidate_status(root, model_id, candidate.candidate_id,
                            {"status": "running", "started_at_utc": _utc_now(), "current_stage": None})
    failed = False
    for index, stage in enumerate(context.stages):
        attempt_id = "%s-%s-%s" % (model_id, candidate.candidate_id, stage.label)
        stage_payload: dict[str, Any] = {
            "model_id": model_id, "candidate_id": candidate.candidate_id,
            "attempt_id": attempt_id, "stage": stage.label, "bin_size_bp": stage.bin_size,
            "started_at_utc": _utc_now(),
        }
        if failed:
            stage_payload.update({"status": "not_run_after_prior_failure",
                                  "failure_reason": "prior stage failed", "completed_at_utc": _utc_now()})
            record_path = _write_stage(root, model_id, candidate.candidate_id, stage.label, stage_payload)
            attempts.append({"attempt_id": attempt_id, "model_id": model_id,
                             "candidate_id": candidate.candidate_id, "stage": stage.label,
                             "status": "not_run_after_prior_failure", "is_terminal": False,
                             "stage_record": record_path, "failure_reason": "prior stage failed"})
            layers.append({"stage": stage.label, "path": record_path, "status": "not_run_after_prior_failure"})
            continue
        try:
            data = load_data(stage.bin_size)
            budget = _validate_layer(data, context, stage)
            if previous_coordinates is None:
                initialized = initialize(candidate.initialization_candidate, names, lengths, stage.bin_size)
            else:
                if warm_start is None:
                    initialized = transfer_layer(previous_coordinates, previous_positions, previous_chromosome,
                                                 names, lengths, stage.bin_size, candidate.base_seed, model_id)
                else:
                    initialized = warm_start(previous_coordinates, previous_positions, previous_chromosome,
                                             names, lengths, stage.bin_size, candidate.base_seed, model_id)
            initial_coordinates, positions, chromosome_index, init_meta = _validate_initialization(
                initialized, data, model_id)
            initial_path = root / "coords" / model_id / candidate.candidate_id / ("initial-%s.3dg" % stage.label)
            initial_path.parent.mkdir(parents=True, exist_ok=True)
            initial_info = paired_run.write_coordinates(initial_path, data, model_id, initial_coordinates)
            result, fit_payload, final_coordinates = _fit_layer(
                root, context, model_id, candidate, stage, data, initial_coordinates,
                positions, chromosome_index, carried_q, fit_callable=fit_callable)
            final_path_obj = root / "coords" / model_id / candidate.candidate_id / ("final-%s.3dg" % stage.label)
            final_path_obj.parent.mkdir(parents=True, exist_ok=True)
            final_info = paired_run.write_coordinates(final_path_obj, data, model_id, final_coordinates)
            stage_payload.update({
                "status": fit_payload["status"], "completed_at_utc": _utc_now(),
                "data_budget": budget, "initialization": init_meta,
                "initial_coordinates": {**initial_info, "path": _relative(root, initial_path)},
                "final_coordinates": {**final_info, "path": _relative(root, final_path_obj)},
                "fit": fit_payload,
            })
            record_path = _write_stage(root, model_id, candidate.candidate_id, stage.label, stage_payload)
            attempts.append({"attempt_id": attempt_id, "model_id": model_id,
                             "candidate_id": candidate.candidate_id, "stage": stage.label,
                             "status": fit_payload["status"], "is_terminal": False,
                             "stage_record": record_path, "budget_exhausted": fit_payload["budget_exhausted"]})
            layers.append({"stage": stage.label, "path": record_path,
                           "status": fit_payload["status"], "data_budget": budget})
            previous_coordinates = final_coordinates
            previous_positions = positions
            previous_chromosome = chromosome_index
            carried_q = float(np.asarray(result.theta, dtype=np.float64)[-1])
            if index == len(context.stages) - 1:
                final_path, final_sha, final_q = _relative(root, final_path_obj), final_info["sha256"], carried_q
        except Exception as exc:
            reason = "%s: %s" % (type(exc).__name__, exc)
            stage_payload.update({"status": "failed", "failed_at_utc": _utc_now(),
                                  "failure_reason": reason, "traceback": traceback.format_exc(limit=12)})
            record_path = _write_stage(root, model_id, candidate.candidate_id, stage.label, stage_payload)
            attempts.append({"attempt_id": attempt_id, "model_id": model_id,
                             "candidate_id": candidate.candidate_id, "stage": stage.label,
                             "status": "failed", "is_terminal": True,
                             "stage_record": record_path, "failure_reason": reason})
            layers.append({"stage": stage.label, "path": record_path, "status": "failed"})
            _write_candidate_status(root, model_id, candidate.candidate_id,
                                    {"status": "failed", "updated_at_utc": _utc_now(),
                                     "current_stage": stage.label, "failure_reason": reason})
            failed = True
    if failed or final_path is None or final_sha is None or final_q is None:
        reason = "chain failed before final 1 Mb output"
        prior = next((row.get("failure_reason") for row in attempts if row["status"] == "failed"), reason)
        return {"id": candidate.candidate_id, "model_id": model_id,
                "initialization": candidate.initialization_candidate, "base_seed": candidate.base_seed,
                "model_signature": config["models"][model_id]["model_signature"],
                "prior_config_sha256": config["models"][model_id]["prior_config_sha256"],
                "parameter_dimension": config["models"][model_id]["parameter_dimension"],
                "coordinates": None, "count_nll_per_record": None,
                "failure_reason": prior or reason, "attempts": attempts, "layers": layers,
                "optimization": {"terminal_status": "failed",
                                  "terminal_attempt_id": next(row["attempt_id"] for row in attempts if row["is_terminal"])}}
    attempts[-1]["is_terminal"] = True
    _write_candidate_status(root, model_id, candidate.candidate_id,
                            {"status": "awaiting_final_count_rescore", "updated_at_utc": _utc_now(),
                             "current_stage": "1m"})
    return {"id": candidate.candidate_id, "model_id": model_id,
            "initialization": candidate.initialization_candidate, "base_seed": candidate.base_seed,
            "model_signature": config["models"][model_id]["model_signature"],
            "prior_config_sha256": config["models"][model_id]["prior_config_sha256"],
            "parameter_dimension": config["models"][model_id]["parameter_dimension"],
            "final_coordinates_path": final_path, "final_coordinates_sha256": final_sha,
            "final_q": final_q, "attempts": attempts, "layers": layers,
            "optimization": {"terminal_status": attempts[-1]["status"],
                              "terminal_attempt_id": attempts[-1]["attempt_id"]}}

def _read_model_coordinates(path: Path, data: contact_model.AggregatedContacts,
                            model_id: str, expected_sha: str | None = None) -> np.ndarray:
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise VariantRunnerError("coordinate hash mismatch: %s" % path)
    rows: dict[str, dict[int, np.ndarray]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 5:
                raise VariantRunnerError("coordinate line %d has %d fields" % (line_no, len(fields)))
            try:
                track, position = fields[0], int(fields[1])
                xyz = np.asarray([float(v) for v in fields[2:5]], dtype=np.float64)
            except ValueError as exc:
                raise VariantRunnerError("coordinate line %d is not numeric" % line_no) from exc
            if position < 0 or not np.all(np.isfinite(xyz)):
                raise VariantRunnerError("coordinate line %d is invalid" % line_no)
            by_position = rows.setdefault(track, {})
            if position in by_position:
                raise VariantRunnerError("coordinate duplicate %s:%d" % (track, position))
            by_position[position] = xyz
    expected_tracks = {spec.name for spec in data.track_specs}
    if set(rows) != expected_tracks:
        raise VariantRunnerError("coordinate track inventory changed")
    result = np.empty((2, data.n_loci, 3), dtype=np.float64)
    for spec in data.track_specs:
        slc = data.chromosome_slice(spec.chromosome_index)
        expected_positions = [int(v * data.bin_size) for v in data.locus_bin[slc]]
        if set(rows[spec.name]) != set(expected_positions):
            raise VariantRunnerError("coordinate full grid changed for %s" % spec.name)
        for global_index, position in zip(range(slc.start, slc.stop), expected_positions, strict=True):
            result[spec.copy_index, global_index] = rows[spec.name][position]
    return _validate_model_coordinates(model_id, result, data.n_loci)

def _rescore_candidate(root: Path, context: reconstruct.RunContext,
                       config: Mapping[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    stage = context.stages[-1]
    data = contact_model.load_frozen_p9016_aggregate(stage.bin_size)
    budget = _validate_layer(data, context, stage)
    path = root / candidate["final_coordinates_path"]
    coordinates = _read_model_coordinates(path, data, candidate["model_id"], candidate["final_coordinates_sha256"])
    objective = allele_models.objective_for_model(data, candidate["model_id"])
    raw = allele_models.raw_coordinates_from_physical(objective, coordinates)
    theta = objective.pack(raw, p=P_INIT)
    theta[-1] = float(candidate["final_q"])
    total, _, components = objective.evaluate(theta, need_gradient=False)
    count_model = {field: float(components[field]) for field in reconstruction_report.COUNT_MODEL_FIELDS}
    if not math.isfinite(count_model["count_nll_normalized"]):
        raise VariantRunnerError("final count score is nonfinite")
    candidate.update({
        "coordinates": {"path": candidate["final_coordinates_path"],
                        "sha256": candidate["final_coordinates_sha256"],
                        "model_id": candidate["model_id"],
                        "physical_domain": allele_models.model_spec(candidate["model_id"]).physical_domain},
        "count_nll_per_record": count_model["count_nll_normalized"],
        "count_model": count_model,
        "selection_rescore": {
            "coordinate_readback": True, "q_fixed_from_final_fit": float(candidate["final_q"]),
            "total": float(total), "data_budget": budget,
            "diagnostics": {key: _jsonable(value) for key, value in components.items()
                            if key not in reconstruction_report.COUNT_MODEL_FIELDS},
        },
    })
    candidate["optimization"] = {"terminal_status": candidate["attempts"][-1]["status"],
                                  "terminal_attempt_id": candidate["attempts"][-1]["attempt_id"]}
    return candidate

def select_variant_candidate(candidates: Sequence[Mapping[str, Any]],
                              candidate_order: Sequence[str] = ("consensus_joint", "random_joint")) -> str:
    """只按 count NLL 在一个变体内选择，并使用冻结的平局顺序。"""
    usable = []
    by_id = {str(item["id"]): item for item in candidates}
    for candidate_id in candidate_order:
        item = by_id.get(str(candidate_id))
        if item is None:
            continue
        if item.get("optimization", {}).get("terminal_status") == "failed":
            continue
        value = item.get("count_nll_per_record")
        if value is None or not math.isfinite(float(value)):
            continue
        usable.append((str(candidate_id), float(value)))
    if not usable:
        raise VariantRunnerError("all candidates failed before variant selection")
    minimum = min(value for _, value in usable)
    for candidate_id, value in usable:
        if value <= minimum + TIE_TOLERANCE:
            return candidate_id
    raise AssertionError("unreachable selection branch")

def _model_contract(model_id: str, final_loci: int) -> dict[str, Any]:
    spec = allele_models.model_spec(model_id)
    definition = {
        "model_id": model_id,
        "model": spec.as_dict(),
        "source_hashes": {key: sha256_file(ROOT / key) for key in
                          ("pr/allele_models.py", "pr/contact_model.py", "pr/joint_fit.py",
                           "pr/reconstruction_init.py", "pr/paired_run.py")},
        "parameter_dimension": 6 * int(final_loci) + 1,
        "selection_metric": "count_nll_per_record",
        "selection_uses_priors": False,
        "selection_includes_diagonal_profile": True,
    }
    signature = hashlib.sha256(_json_bytes(definition)).hexdigest()
    prior_payload = {"weights": spec.as_dict()["objective_weights"],
                     "p_floor": contact_model.P_FLOOR,
                     "p_prior_strength": contact_model.P_PRIOR_STRENGTH,
                     "repulsion": "0.7*l0", "epsilon": contact_model.EPSILON}
    return {"model_signature": signature,
            "prior_config_sha256": hashlib.sha256(_json_bytes(prior_payload)).hexdigest(),
            "parameter_dimension": int(definition["parameter_dimension"]),
            "definition": definition}

def _planned_rows(variants: Sequence[str] = VARIANTS) -> list[dict[str, Any]]:
    return [{"attempt_id": VariantPlan(variant, candidate.candidate_id, stage).attempt_id,
             "variant": variant, "candidate_id": candidate.candidate_id, "stage": stage.label,
             "bin_size_bp": stage.bin_size, "maxiter": stage.maxiter, "maxfun": stage.maxfun}
            for variant in variants for candidate in CANDIDATES for stage in STAGES]

def _build_protocol(context: reconstruct.RunContext, preflight_sha256: str,
                    artifact_root: Path | None = None) -> dict[str, Any]:
    final_loci = sum((int(length) + 1_000_000 - 1) // 1_000_000 for _, length in context.headers)
    planned = _planned_rows()
    source_assets = []
    for candidate in CANDIDATES:
        source = context.source_assets[candidate.initialization_candidate]
        source_assets.append({"candidate_id": candidate.candidate_id,
                              "initialization": candidate.initialization_candidate,
                              "base_seed": candidate.base_seed, "stage": "blind",
                              "path": str(source["path"]), "sha256": str(source["sha256"]),
                              "gate_path": str(source["gate_path"]), "gate_sha256": str(source["gate_sha256"])})
    protocol = {
        "schema": "multires-variant-protocol-v1",
        "status": "frozen",
        "created_at_utc": _utc_now(),
        "scope": "phase-free/reference-free post-020 variant training; C0 held by 033",
        "cohort": dict(context.cohort),
        "input": {"path": str(Path(context.data_path).resolve()), "sha256": context.data_sha256,
                  "training_columns": ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"],
                  "uses_all_1703888_records": True, "raw_source_sha256": contact_model.FROZEN_P9016_RAW_SHA256},
        "grid": {"origin_bp": 0, "full_grid": True, "final_bin_size_bp": 1_000_000,
                 "n_loci": final_loci, "n_physical_beads": 2 * final_loci, "n_tracks": 40},
        "sources": source_assets,
        "source_code_sha256": _source_hashes(),
        "020_comparator": {"run": _relative(ROOT, OLD_RUN), "files": _comparator_hashes()},
        "033_c0_inputs": {"protocol_path": _relative(ROOT, C0_PROTOCOL),
                           "protocol_sha256": sha256_file(C0_PROTOCOL),
                           "preflight_path": _relative(ROOT, C0_PREFLIGHT),
                           "preflight_sha256": sha256_file(C0_PREFLIGHT),
                           "gate_path": _relative(ROOT, C0_GATE),
                           "gate_required_before_launch": True},
        "variants": [allele_models.model_spec(model).as_dict() for model in ALL_VARIANTS],
        "remaining_variants": list(VARIANTS),
        "layer_budgets": EXPECTED_LAYER_BUDGETS,
        "objective_contract": {
            "epsilon": contact_model.EPSILON, "r0": "2*l0", "l0": "(2*N_loci)^(-1/3)",
            "repulsion_threshold": "0.7*l0", "weights": {"count": 1.0, "p_prior": 1.0,
            "bond": 1.0, "repulsion": 1.0, "bend": 0.01},
            "p_floor": contact_model.P_FLOOR, "p_prior_strength": contact_model.P_PRIOR_STRENGTH,
            "exposure": "sqrt(endpoint_count + 10) / fullgrid mean; C3 all ones",
            "same_bin": "per-layer per-bin saturated Poisson nuisance",
        },
        "initialization_contract": {
            "source_sha256": {"consensus": source_assets[0]["sha256"], "random": source_assets[1]["sha256"]},
            "target_radius": 0.8, "full_grid_origin_bp": 0,
            "interpolation": "numeric genomic sorting; bracket linear interpolation; nearest endpoint fill",
            "consensus": "cXXa shared Z plus seed-1103 smooth half-difference",
            "random": "retain both source copies with base seed 2207",
            "first_layer_common_center_uniform_scale_maxR": 0.8,
            "jitter": "0.025*l0 on new/interpolated/endpoint-filled/repeated loci",
            "warm_seed": "3301 + bin_size//1e6 + candidate_base_seed",
            "between_layer_frame": "preserve previous physical frame; no center/rescale",
            "bounded_warm_clip": "radial clip to 1-1e-6",
            "C2_free_warm_clip": "forbidden; direct physical coordinates retained",
        },
        "optimization": {
            "schedule": [{"label": stage.label, "bin_size_bp": stage.bin_size,
                           "maxiter": stage.maxiter, "maxfun": stage.maxfun, "maxls": 20,
                           "ftol": 1e-10, "gtol": 1e-6,
                           "checkpoint_every_accepted": CHECKPOINT_EVERY} for stage in STAGES],
            "p_init_first_layer": P_INIT, "carry_raw_q_between_layers": True,
            "independent_chain_per_variant_source": True, "budget_not_convergence": True,
        },
        "domain_contract": {
            "C1": "sphere_forward_unit_ball, bend weight 0",
            "C2-map": "identity-core radius <=0.9 with smooth outer ball, physical radius <1",
            "C2-free": "direct Cartesian physical coordinates; no sphere inverse/clip/rescale",
            "C3": "sphere_forward_unit_ball, exposure ones",
            "initial_common_center_scale": "shared initialization only, not a free optimization constraint",
        },
        "mapping_diagnostics": {
            "record_all_objective_calls": True, "record_trial_and_accepted_when_available": True,
            "fields": ["objective_eval_count", "nonidentity_map_eval_count",
                       "nonidentity_bead_eval_count", "max_raw_radius_seen",
                       "max_physical_radius_seen", "ever_physical_radius_gt1"],
            "selection_or_budget_use": False,
        },
        "selection": {
            "criterion": "count_nll_per_record", "direction": "minimize",
            "tie_tolerance_per_record": TIE_TOLERANCE, "tie_break": "preregistered_order",
            "order": [candidate.candidate_id for candidate in CANDIDATES],
            "within_each_variant_only": True, "includes_candidate_invariant_diagonal_layer": True,
            "includes_priors": False, "reference_used": False, "phase_used": False,
        },
        "attempt_plan": {
            "planned_variant_count": len(VARIANTS), "planned_candidate_count": len(CANDIDATES),
            "planned_stage_count": len(STAGES), "planned_attempt_count": len(planned),
            "planned_attempts": planned, "executed_at_prepare": 0,
            "blocked_until_c0_gate": len(planned), "combined_with_033_c0_attempts": 30,
        },
        "preflight": {"path": "preflight.json", "sha256": preflight_sha256,
                      "artifact_root": None if artifact_root is None else str(artifact_root)},
        "runtime": {"max_candidate_workers": 4, "recommended_candidate_workers": 4,
                     "threads": 1, "thread_env": {name: "1" for name in THREAD_ENV},
                     "native_fdg_started": False},
        "boundary": {"phase_used": False, "reference_used": False,
                      "oracle_coordinates_opened": False, "evaluation_outputs_opened": False},
        "release": {"status": "blocked_pending_c0_gate", "required_gate": _relative(ROOT, C0_GATE),
                     "release_action": "parent sends explicit release after C0_gate.json status=passed"},
    }
    return protocol

def _build_config(context: reconstruct.RunContext, protocol: Mapping[str, Any],
                  protocol_sha256: str, preflight_sha256: str,
                  formal_out: Path, workers: int) -> dict[str, Any]:
    final_loci = int(protocol["grid"]["n_loci"])
    return {
        "schema": "multires-variant-config-v1",
        "purpose": "formal_phase_free_variant_training_only",
        "protocol_sha256": protocol_sha256,
        "preflight_sha256": preflight_sha256,
        "cohort": dict(context.cohort),
        "coordinate_grid": dict(protocol["grid"]),
        "input": dict(protocol["input"]),
        "variants": list(VARIANTS),
        "candidate_order": [candidate.candidate_id for candidate in CANDIDATES],
        "models": {model: _model_contract(model, final_loci) for model in VARIANTS},
        "optimization": dict(protocol["optimization"]),
        "selection": dict(protocol["selection"]),
        "runtime": {"candidate_workers": int(workers), "thread_env": {name: "1" for name in THREAD_ENV}},
        "formal_output_path": str(formal_out),
        "training_boundary": {"phase_used": False, "reference_used": False,
                              "oracle_coordinates_opened": False, "evaluation_outputs_opened": False,
                              "native_fdg_started": False},
        "prepared_not_launched": True,
    }

def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise VariantRunnerError("cannot read JSON %s: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise VariantRunnerError("JSON object required: %s" % path)
    return value

def _verify_c0_release(path: str | os.PathLike[str]) -> dict[str, Any]:
    gate_path = Path(path)
    if not gate_path.is_absolute():
        gate_path = ROOT / gate_path
    gate_path = gate_path.resolve()
    if gate_path != C0_GATE.resolve():
        raise VariantRunnerError("variant release must use the fixed 033 C0_gate.json")
    gate = _read_json(gate_path)
    if gate.get("schema") != "c0-reference-free-numeric-gate-v1" or gate.get("status") != "passed":
        raise VariantRunnerError("C0 gate is not passed; variant stages remain blocked")
    if gate.get("reference_free") is not True or gate.get("phase_free") is not True:
        raise VariantRunnerError("C0 gate boundary is not reference/phase free")
    differences = gate.get("differences", [])
    if differences:
        raise VariantRunnerError("C0 gate contains numerical differences")
    checks = gate.get("checks", {})
    required = ("all_initial_physical_states_byte_equal", "all_final_physical_states_byte_equal",
                "all_checkpoint_states_byte_equal", "all_solver_raw_fields_match",
                "selection_same_and_rule_frozen")
    if any(checks.get(key) is not True for key in required):
        raise VariantRunnerError("C0 gate lacks required parity checks")
    return {"path": _relative(ROOT, gate_path), "sha256": sha256_file(gate_path),
            "schema": gate["schema"], "status": gate["status"]}

def _verify_prepared_docs(protocol_path: Path | None, preflight_path: Path | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    protocol = None
    preflight = None
    if protocol_path is not None:
        protocol = _read_json(protocol_path)
        if protocol.get("schema") != "multires-variant-protocol-v1" or protocol.get("status") != "frozen":
            raise VariantRunnerError("prepared protocol is not frozen")
    if preflight_path is not None:
        preflight = _read_json(preflight_path)
        if preflight.get("schema") != "multires-variant-preflight-v1" or preflight.get("status") != "passed":
            raise VariantRunnerError("prepared preflight is not passed")
    return protocol, preflight

def _copy_exclusive(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as left, target.open("xb") as right:
        shutil.copyfileobj(left, right)
        right.flush()
        os.fsync(right.fileno())
    return sha256_file(target)

def _candidate_failure_after_rescore(candidate: dict[str, Any], reason: str) -> dict[str, Any]:
    candidate["coordinates"] = None
    candidate["count_nll_per_record"] = None
    candidate["failure_reason"] = reason
    candidate["attempts"][-1]["is_terminal"] = False
    candidate["attempts"].append({"attempt_id": "%s-final_count_rescore" % candidate["id"],
                                  "model_id": candidate["model_id"], "candidate_id": candidate["id"],
                                  "stage": "final_count_rescore", "status": "failed", "is_terminal": True,
                                  "failure_reason": reason})
    candidate["optimization"] = {"terminal_status": "failed",
                                  "terminal_attempt_id": candidate["attempts"][-1]["attempt_id"]}
    return candidate

def _worker_entry(root_text: str, context: reconstruct.RunContext,
                  config: Mapping[str, Any], model_id: str,
                  candidate: reconstruct.CandidateSpec) -> dict[str, Any]:
    _set_threads(1)
    return _run_chain(Path(root_text), context, config, model_id, candidate)

def _run_chains(root: Path, context: reconstruct.RunContext, config: Mapping[str, Any],
                workers: int) -> list[dict[str, Any]]:
    jobs = [(model, candidate) for model in VARIANTS for candidate in CANDIDATES]
    if int(workers) == 1:
        return [_run_chain(root, context, config, model, candidate) for model, candidate in jobs]
    process_context = multiprocessing.get_context("spawn")
    results: dict[tuple[str, str], dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=int(workers), mp_context=process_context) as pool:
        futures = {pool.submit(_worker_entry, str(root), context, config, model, candidate): (model, candidate)
                   for model, candidate in jobs}
        for future in as_completed(futures):
            model, candidate = futures[future]
            try:
                results[(model, candidate.candidate_id)] = future.result()
            except Exception as exc:
                reason = "worker_failure: %s: %s" % (type(exc).__name__, exc)
                attempts = []
                for stage in STAGES:
                    path = _write_stage(root, model, candidate.candidate_id, stage.label, {
                        "model_id": model, "candidate_id": candidate.candidate_id,
                        "attempt_id": "%s-%s-%s" % (model, candidate.candidate_id, stage.label),
                        "stage": stage.label, "status": "failed", "failure_reason": reason,
                    })
                    attempts.append({"attempt_id": "%s-%s-%s" % (model, candidate.candidate_id, stage.label),
                                     "model_id": model, "candidate_id": candidate.candidate_id,
                                     "stage": stage.label, "status": "failed" if stage == STAGES[0] else "not_run_after_prior_failure",
                                     "is_terminal": stage == STAGES[0], "stage_record": path,
                                     "failure_reason": reason})
                results[(model, candidate.candidate_id)] = {
                    "id": candidate.candidate_id, "model_id": model,
                    "initialization": candidate.initialization_candidate, "base_seed": candidate.base_seed,
                    "model_signature": config["models"][model]["model_signature"],
                    "prior_config_sha256": config["models"][model]["prior_config_sha256"],
                    "parameter_dimension": config["models"][model]["parameter_dimension"],
                    "coordinates": None, "count_nll_per_record": None,
                    "failure_reason": reason, "attempts": attempts, "layers": [],
                    "optimization": {"terminal_status": "failed", "terminal_attempt_id": attempts[0]["attempt_id"]},
                }
    return [results[(model, candidate.candidate_id)] for model, candidate in jobs]

def _write_selection(root: Path, context: reconstruct.RunContext, config: Mapping[str, Any],
                     candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selections = {}
    selected_files = {}
    for model in VARIANTS:
        subset = [item for item in candidates if item["model_id"] == model]
        selected_id = select_variant_candidate(subset)
        selected = next(item for item in subset if item["id"] == selected_id)
        source = root / selected["coordinates"]["path"]
        target = root / "selected" / (model + ".3dg")
        digest = _copy_exclusive(source, target)
        selected_files[model] = {"path": _relative(root, target), "sha256": digest,
                                 "source_candidate_id": selected_id,
                                 "source_coordinate_path": selected["coordinates"]["path"],
                                 "source_coordinate_sha256": selected["coordinates"]["sha256"],
                                 "physical_domain": allele_models.model_spec(model).physical_domain}
        selections[model] = {"selected_id": selected_id,
                             "candidate_scores": {item["id"]: item.get("count_nll_per_record") for item in subset},
                             "tie_tolerance_per_record": TIE_TOLERANCE,
                             "criterion": "count_nll_per_record", "reference_used": False, "phase_used": False}
    attempts = [attempt for candidate in candidates for attempt in candidate.get("attempts", [])]
    return {
        "schema": "multires-variant-selection-v1", "status": "training_complete",
        "created_at_utc": _utc_now(), "cohort": dict(context.cohort),
        "coordinate_grid": dict(config["coordinate_grid"]), "objective_contract": dict(config["selection"]),
        "models": {model: config["models"][model] for model in VARIANTS},
        "candidates": list(_jsonable(candidates)), "attempts": attempts,
        "selection": {"per_variant": selections, "selected_files": selected_files,
                       "all_attempts_accounted_for": len(attempts) == 24,
                       "reference_used": False, "phase_used": False},
        "training_boundary": dict(config["training_boundary"]),
    }

def run_variants(outdir: str | os.PathLike[str], *, c0_gate: str | os.PathLike[str],
                 protocol_path: str | os.PathLike[str] | None = None,
                 preflight_path: str | os.PathLike[str] | None = None,
                 workers: int = 4, threads: int = 1) -> dict[str, Any]:
    """仅在固定的 033 C0 gate 通过后运行其余 24 个阶段。"""
    if int(workers) < 1 or int(workers) > 4:
        raise VariantRunnerError("variant controller permits one to four candidate workers")
    _set_threads(threads)
    release = _verify_c0_release(c0_gate)
    prepared_protocol, prepared_preflight = _verify_prepared_docs(
        None if protocol_path is None else Path(protocol_path),
        None if preflight_path is None else Path(preflight_path),
    )
    context = reconstruct.production_context()
    _validate_context(context)
    root = _fresh_root(outdir)
    started = _utc_now()
    try:
        if prepared_protocol is None:
            preflight_stub = {"schema": "multires-variant-preflight-v1", "status": "passed"}
            preflight_sha = hashlib.sha256(_json_bytes(preflight_stub)).hexdigest()
            protocol = _build_protocol(context, preflight_sha)
        else:
            protocol = prepared_protocol
            preflight_sha = sha256_file(Path(preflight_path)) if preflight_path is not None else str(protocol["preflight"]["sha256"])
        protocol_sha = hashlib.sha256(_json_bytes(protocol)).hexdigest()
        config = _build_config(context, protocol, protocol_sha, preflight_sha, root, workers)
        _write_exclusive(root / "protocol.json", protocol)
        _write_exclusive(root / "config.json", config)
        _write_exclusive(root / "provenance" / "source_hashes.json", {
            "schema": "multires-source-hashes-v1", "source_code_sha256": _source_hashes(),
            "c0_gate": release, "protocol_sha256": protocol_sha, "preflight_sha256": preflight_sha,
        })
        for relative in ("coords", "checkpoints", "logs", "stages", "selected", "run_status/candidates"):
            (root / relative).mkdir(parents=True, exist_ok=True)
        _write_exclusive(root / "run_status" / "run_status.json", {
            "status": "training", "started_at_utc": started,
            "candidates": {model + "__" + candidate.candidate_id: "pending"
                           for model in VARIANTS for candidate in CANDIDATES},
        })
        candidates = _run_chains(root, context, config, workers)
        finalized = []
        for candidate in candidates:
            if candidate.get("optimization", {}).get("terminal_status") == "failed":
                finalized.append(candidate)
                continue
            try:
                finalized.append(_rescore_candidate(root, context, config, candidate))
            except Exception as exc:
                finalized.append(_candidate_failure_after_rescore(
                    candidate, "final_count_rescore: %s: %s" % (type(exc).__name__, exc)))
        selection = _write_selection(root, context, config, finalized)
        selection_sha = _write_exclusive(root / "selection.json", selection)
        attempts = [attempt for candidate in finalized for attempt in candidate.get("attempts", [])]
        terminal = {
            "schema": "multires-variant-terminal-evidence-v1", "status": "training_complete",
            "runner_exit_code": 0, "started_at_utc": started, "ended_at_utc": _utc_now(),
            "planned_stage_attempt_count": 24, "executed_stage_attempt_count": sum(
                1 for attempt in attempts if attempt.get("status") not in ("not_run_after_prior_failure",)),
            "attempts": attempts, "selection_sha256": selection_sha,
            "release_gate": release, "training_boundary": config["training_boundary"],
        }
        terminal_sha = _write_exclusive(root / "terminal_evidence.json", terminal)
        receipt = {"schema": "multires-variant-run-receipt-v1", "status": "training_complete",
                   "runner_exit_code": 0, "outdir": str(root), "planned_stage_attempt_count": 24,
                   "executed_stage_attempt_count": terminal["executed_stage_attempt_count"],
                   "selected_by_variant": {model: selection["selection"]["per_variant"][model]["selected_id"]
                                           for model in VARIANTS},
                   "selection": {"path": "selection.json", "sha256": selection_sha},
                   "terminal": {"path": "terminal_evidence.json", "sha256": terminal_sha},
                   "c0_gate": release, "training_boundary": config["training_boundary"]}
        receipt_sha = _write_exclusive(root / "run_receipt.json", receipt)
        _write_atomic_json(root / "run_status" / "run_status.json", {
            "status": "training_complete", "ended_at_utc": _utc_now(),
            "receipt": "run_receipt.json", "receipt_sha256": receipt_sha,
        })
        return {"status": "training_complete", "outdir": str(root),
                "selection_path": str(root / "selection.json"),
                "terminal_path": str(root / "terminal_evidence.json"),
                "receipt_path": str(root / "run_receipt.json")}
    except Exception as exc:
        failure = {"schema": "multires-variant-run-failure-v1", "status": "failed",
                   "created_at_utc": _utc_now(), "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc(limit=16),
                   "remaining_blocked_stage_attempt_count": 24,
                   "training_boundary": {"phase_used": False, "reference_used": False,
                                         "evaluation_outputs_opened": False, "native_fdg_started": False}}
        if not (root / "run_failure.json").exists():
            _write_exclusive(root / "run_failure.json", failure)
        raise

def _synthetic_data(bin_size: int = 10) -> contact_model.AggregatedContacts:
    records = [
        (0, 0, 0, 10), (0, 10, 0, 20), (0, 20, 0, 10),
        (1, 0, 1, 10), (0, 0, 1, 0), (0, 10, 1, 10),
        (0, 20, 1, 20), (0, 0, 0, 0),
    ]
    arrays = tuple(np.asarray(part, dtype=np.int64) for part in zip(*records))
    return contact_model.aggregate_from_arrays(("chrA", "chrB"), (30, 30), *arrays, bin_size)

def _parity_fake_fit(records: list[dict[str, Any]]) -> Callable[..., Any]:
    def fake(objective: contact_model.JointObjective, initial_y: np.ndarray, **kwargs: Any) -> Any:
        records.append({"initial_y": np.asarray(initial_y, dtype=np.float64).copy(),
                        "kwargs": {key: value for key, value in kwargs.items()
                                   if key not in ("callback", "checkpoint_hook")}})
        theta = objective.pack(initial_y, p=kwargs["p_init"])
        if kwargs.get("q_init") is not None:
            theta[-1] = float(kwargs["q_init"])
        value, _, components = objective.evaluate(theta, need_gradient=False)
        coordinates, p = objective.coordinates_and_p(theta)
        entry = {"iteration": 10, "nfev": 10, "elapsed_seconds": 0.01,
                 "fun": float(value), "p": float(p), "components": dict(components)}
        checkpoint = joint_fit.JointCheckpoint(
            iteration=10, nfev=10, elapsed_seconds=0.01, fun=float(value), p=float(p),
            theta=theta.copy(), y=np.asarray(initial_y).copy(), coordinates=coordinates.copy(),
            components=dict(components), gradient_norm=0.0)
        kwargs["checkpoint_hook"](checkpoint)
        kwargs["callback"](entry)
        return joint_fit.JointFitResult(
            objective=objective, theta=theta.copy(), y=np.asarray(initial_y).copy(),
            coordinates=coordinates.copy(), p=float(p), fun=float(value), components=dict(components),
            success=True, status=0, message="synthetic parity fit", nit=10, nfev=10,
            actual_nfev=10, scipy_nfev=10, njev=10, elapsed_seconds=0.01, history=[entry])
    return fake

def _parity_probe() -> dict[str, Any]:
    data = _synthetic_data(10)
    headers = (("chrA", 30), ("chrB", 30))
    context = reconstruct.RunContext(
        headers=headers, data_path="synthetic", data_sha256="0" * 64,
        cohort={"sample_id": "synthetic", "biological_samples": 1,
                "raw_contacts": 8, "intra_contacts": 6, "inter_contacts": 2,
                "snpfree_sha256": "0" * 64}, source_assets={}, baseline_assets=(),
        stages=(reconstruct.StageSpec("fixture", 10, 2),), strict_p9016=False)
    candidate = CANDIDATES[0]
    state = {
        "coords": np.asarray([[[0.02, 0.03, 0.01]] * data.n_loci,
                              [[-0.02, 0.02, 0.03]] * data.n_loci], dtype=np.float64),
    }
    coordinates = state["coords"]
    positions, chromosome = _expected_positions(data)
    old_records: list[dict[str, Any]] = []
    new_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="multires-c0-parity-") as directory:
        old_root = Path(directory) / "old"
        new_root = Path(directory) / "new"
        for root in (old_root, new_root):
            (root / "checkpoints").mkdir(parents=True)
            (root / "logs").mkdir(parents=True)
        old_fit = _parity_fake_fit(old_records)
        new_fit = _parity_fake_fit(new_records)
        old_hooks = reconstruct.RunnerHooks(lambda _bin: data, lambda *_args: state,
                                             lambda *_args: state, old_fit)
        old_result, old_payload, old_coords = reconstruct._fit_layer(
            old_root, context, candidate, context.stages[0], data, coordinates,
            positions, chromosome, -0.31, old_hooks)
        new_result, new_payload, new_coords = _fit_layer(
            new_root, context, "C0", candidate, context.stages[0], data, coordinates,
            positions, chromosome, -0.31, fit_callable=new_fit)
        kwargs_equal = old_records[0]["kwargs"] == new_records[0]["kwargs"]
        initial_y_diff = _max_abs(old_records[0]["initial_y"], new_records[0]["initial_y"])
        state_diff = max(_max_abs(old_result.theta, new_result.theta),
                         _max_abs(old_result.coordinates, new_result.coordinates),
                         abs(float(old_result.fun) - float(new_result.fun)))
        component_diff = max(abs(float(old_result.components[key]) - float(new_result.components[key]))
                             for key in old_result.components
                             if isinstance(old_result.components[key], (int, float)))
        old_cp = np.load(old_root / "checkpoints" / "consensus_joint" / "fixture-accepted-0010.npz", allow_pickle=False)
        new_cp = np.load(new_root / "checkpoints" / "C0" / "consensus_joint" / "fixture-accepted-0010.npz", allow_pickle=False)
        checkpoint_diff = max(_max_abs(old_cp[key], new_cp[key]) for key in ("coordinates", "theta", "y"))
        old_cp.close()
        new_cp.close()
        writer_old = Path(directory) / "old.3dg"
        writer_new = Path(directory) / "new.3dg"
        reconstruct._write_coordinates_exclusive(writer_old, data, coordinates)
        paired_run.write_coordinates(writer_new, data, "C0", coordinates)
        writer_equal = writer_old.read_bytes() == writer_new.read_bytes()
    selection_rows = [
        {"id": "consensus_joint", "count_nll_per_record": 1.0,
         "optimization": {"terminal_status": "not_converged"}},
        {"id": "random_joint", "count_nll_per_record": 1.0 + 0.5e-9,
         "optimization": {"terminal_status": "not_converged"}},
    ]
    selection_equal = select_variant_candidate(selection_rows) == reconstruct.select_candidate(
        selection_rows, ("consensus_joint", "random_joint"))
    max_difference = max(initial_y_diff, state_diff, component_diff, checkpoint_diff)
    passed = bool(kwargs_equal and writer_equal and selection_equal and max_difference <= STRICT_ATOL)
    if not passed:
        raise VariantRunnerError("C0 path parity probe failed")
    return {"status": "passed", "max_abs_difference": max_difference,
            "initial_y_max_abs": initial_y_diff, "solver_state_max_abs": state_diff,
            "component_max_abs": component_diff, "checkpoint_state_max_abs": checkpoint_diff,
            "solver_kwargs_equal": kwargs_equal, "callback_and_checkpoint_hooks_exercised": True,
            "writer_bytes_equal": writer_equal, "selection_rule_equal": selection_equal,
            "tested": ["initial_sphere_inverse", "q_init", "solver_kwargs", "callback",
                        "checkpoint", "model_writer", "count_nll_tie_selection"]}

def _domain_probe(root: Path) -> dict[str, Any]:
    data = _synthetic_data(10)
    raw = np.zeros((2, data.n_loci, 3), dtype=np.float64)
    raw[0, 0] = [1.25, 0.0, 0.0]
    mapped = allele_models.objective_for_model(data, "C2-map", block_size=3, repulsion_block_size=3)
    theta = mapped.pack(raw, p=0.75)
    value, gradient, _ = mapped.evaluate(theta, need_gradient=True)
    direction = np.zeros_like(theta)
    direction[0] = 1.0
    step = 1e-6
    plus = mapped.evaluate(theta + step * direction, need_gradient=False)[0]
    minus = mapped.evaluate(theta - step * direction, need_gradient=False)[0]
    numeric = float((plus - minus) / (2.0 * step))
    analytic = float(gradient[0])
    if not np.isclose(analytic, numeric, atol=FD_ATOL, rtol=FD_RTOL):
        raise VariantRunnerError("C2-map gradient probe failed")
    free = allele_models.objective_for_model(data, "C2-free", block_size=3, repulsion_block_size=3)
    outside = np.zeros((2, data.n_loci, 3), dtype=np.float64)
    outside[0, 0] = [1.2, 0.0, 0.0]
    free_raw = allele_models.raw_coordinates_from_physical(free, outside)
    roundtrip = allele_models.physical_coordinates_from_raw(free, free_raw)
    free_theta = free.pack(free_raw, p=0.75)
    free_coordinates, _ = free.coordinates_and_p(free_theta)
    free.evaluate(free_theta, need_gradient=False)
    path = root / "work" / "c2-free-outside1.3dg"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = paired_run.write_coordinates(path, data, "C2-free", outside)
    with path.open(encoding="utf-8") as handle:
        line = next((line.rstrip("\n") for line in handle if line.startswith("c01a\t0\t")), None)
    if line is None or "1.2" not in line or writer["serialization_clip"]["applied"] is not False:
        raise VariantRunnerError("C2-free writer did not preserve radius 1.2")
    c1 = allele_models.objective_for_model(data, "C1")
    c0 = allele_models.objective_for_model(data, "C0")
    c1_theta = c1.pack(allele_models.raw_coordinates_from_physical(c1, np.zeros_like(outside)), p=0.75)
    c0_total, _, c0_components = c0.evaluate(c1_theta, need_gradient=False)
    c1_total, _, c1_components = c1.evaluate(c1_theta, need_gradient=False)
    bend_delta = float(c0_total - c1_total)
    expected_bend_delta = float(c0.weights.bend * c0_components["bend"])
    c3_data = allele_models.data_for_model(data, "C3")
    arrays_equal = all(np.array_equal(getattr(data, key), getattr(c3_data, key))
                       for key in ("pair_i", "pair_j", "cis_pair", "counts", "diag_counts",
                                   "endpoint_counts", "locus_chromosome", "locus_bin", "n_bins", "offsets"))
    if not arrays_equal or not np.array_equal(c3_data.exposure, np.ones(data.n_loci)):
        raise VariantRunnerError("C3 data view changed non-exposure fields")
    if abs(bend_delta - expected_bend_delta) > STRICT_ATOL + STRICT_RTOL:
        raise VariantRunnerError("C1 bend probe failed")
    map_diag = _map_diagnostics(mapped)
    free_diag = _map_diagnostics(free)
    return {
        "status": "passed", "c2_map_gradient": {"analytic": analytic, "central_difference": numeric,
                                                     "abs_difference": abs(analytic - numeric),
                                                     "atol": FD_ATOL, "rtol": FD_RTOL},
        "c2_map_activity": map_diag,
        "c2_free": {"roundtrip_max_abs": _max_abs(roundtrip, outside),
                     "evaluate_radius": float(np.linalg.norm(free_coordinates[0, 0])),
                     "ever_physical_radius_gt1": free_diag["ever_physical_radius_gt1"],
                     "writer_sha256": writer["sha256"], "writer_domain": writer["physical_domain"],
                     "serialization_clip": writer["serialization_clip"], "preserved_line": line},
        "c1": {"observed_bend_delta": bend_delta, "expected_bend_delta": expected_bend_delta,
                "abs_difference": abs(bend_delta - expected_bend_delta)},
        "c3": {"non_exposure_arrays_equal": arrays_equal,
                "uniform_exposure_max_abs_from_one": float(np.max(np.abs(c3_data.exposure - 1.0)))},
        "objective_value_finite": bool(math.isfinite(value)),
    }

def _transfer_probe() -> dict[str, Any]:
    names = ("chrA", "chrB")
    lengths = (30, 30)
    source_layout = reconstruction_init.full_grid_layout(names, lengths, 10)
    coords = np.zeros((2, source_layout["n_loci"], 3), dtype=np.float64)
    for copy in (0, 1):
        for index in range(source_layout["n_loci"]):
            coords[copy, index] = [0.01 * (index + 1), 0.02 * (copy + 1), 0.003 * index]
    original = reconstruction_init.warm_start_from_layer(
        coords, source_layout["positions"], source_layout["chromosome_index"],
        names, lengths, 5, 1103)
    ours = transfer_layer(coords, source_layout["positions"], source_layout["chromosome_index"],
                          names, lengths, 5, 1103, "C1")
    bounded_diff = _max_abs(original["coords"], ours["coords"])
    bounded_positions_equal = np.array_equal(original["positions"], ours["positions"]) and np.array_equal(
        original["chromosome_index"], ours["chromosome_index"])
    outside = coords.copy()
    outside[0, 0] = [1.2, 0.0, 0.0]
    free = transfer_layer(outside, source_layout["positions"], source_layout["chromosome_index"],
                          names, lengths, 5, 1103, "C2-free")
    free_radius = float(np.linalg.norm(free["coords"][0, 0]))
    if bounded_diff != 0.0 or not bounded_positions_equal or free_radius != 1.2:
        raise VariantRunnerError("layer transfer probe failed")
    return {"status": "passed", "bounded_max_abs_difference_vs_original": bounded_diff,
            "bounded_positions_equal": bounded_positions_equal,
            "free_outside_radius_preserved": free_radius,
            "free_clip_applied": free["metadata"]["normalization"]["clip_applied"],
            "seed_formula": "3301 + bin_size//1e6 + candidate_base_seed",
            "jitter_scale_formula": "0.025*l0", "additional_center_rescale": False}

def _real_layer_probe(context: reconstruct.RunContext) -> dict[str, Any]:
    """核验每个 real aggregate layer，不构造 objective 拟合。"""
    rows = {}
    started = time.perf_counter()
    for stage in STAGES:
        data = contact_model.load_frozen_p9016_aggregate(stage.bin_size)
        checked = _validate_layer(data, context, stage)
        rows[stage.label] = {"audit": checked["audit"],
                             "array_sha256": checked["array_sha256"],
                             "exposure_formula_max_abs": checked["exposure_formula_max_abs"]}
    return {"status": "passed", "rows": rows,
            "elapsed_seconds": time.perf_counter() - started,
            "selection_or_hyperparameter_use": False}

def _archived_transfer_probe(context: reconstruct.RunContext) -> dict[str, Any]:
    rows = []
    max_difference = 0.0
    q_max_difference = 0.0
    for candidate in CANDIDATES:
        for left_stage, right_stage in zip(STAGES[:-1], STAGES[1:], strict=True):
            left_doc = _read_json(OLD_RUN / "stages" / candidate.candidate_id / (left_stage.label + ".json"))
            right_doc = _read_json(OLD_RUN / "stages" / candidate.candidate_id / (right_stage.label + ".json"))
            checkpoint_text = left_doc["fit"]["checkpoint_paths"][-1]
            checkpoint_path = OLD_RUN / checkpoint_text
            with np.load(checkpoint_path, allow_pickle=False) as payload:
                coords = payload["coordinates"].copy()
                positions = payload["positions"].copy()
                chromosome = payload["chromosome_index"].copy()
            expected = reconstruction_init.warm_start_from_layer(
                coords, positions, chromosome,
                tuple(name for name, _ in context.headers), tuple(int(length) for _, length in context.headers),
                right_stage.bin_size, candidate.base_seed)
            ours = transfer_layer(
                coords, positions, chromosome,
                tuple(name for name, _ in context.headers), tuple(int(length) for _, length in context.headers),
                right_stage.bin_size, candidate.base_seed, "C1")
            difference = _max_abs(expected["coords"], ours["coords"])
            position_equal = np.array_equal(expected["positions"], ours["positions"]) and np.array_equal(
                expected["chromosome_index"], ours["chromosome_index"])
            q_difference = abs(float(right_doc["fit"]["q_in"]) - float(left_doc["fit"]["q_out"]))
            max_difference = max(max_difference, difference)
            q_max_difference = max(q_max_difference, q_difference)
            rows.append({"candidate_id": candidate.candidate_id, "from": left_stage.label,
                         "to": right_stage.label, "checkpoint": _relative(ROOT, checkpoint_path),
                         "physical_transfer_max_abs": difference, "positions_equal": position_equal,
                         "q_in_minus_previous_q_out_abs": q_difference})
    if max_difference != 0.0 or q_max_difference > STRICT_ATOL + STRICT_RTOL:
        raise VariantRunnerError("archived 020 transfer/q parity failed")
    return {"status": "passed", "rows": rows, "max_abs_difference": max_difference,
            "q_max_abs_difference": q_max_difference, "selection_or_hyperparameter_use": False}

def _inherited_layer_probe() -> dict[str, Any]:
    preflight = _read_json(C0_PREFLIGHT)
    if preflight.get("schema") != "c0-controlled-preflight-v1" or preflight.get("status") != "passed":
        raise VariantRunnerError("033 inherited preflight is not passed")
    rows = {}
    max_difference = 0.0
    for label, expected in EXPECTED_LAYER_BUDGETS.items():
        audit = preflight["layers"][label]["audit"]
        differences = {}
        for key, value in expected.items():
            actual_value = preflight["layers"][label]["denominator_contract"][key] if key == "bin_size_bp" else audit[key]
            differences[key] = abs(int(actual_value) - int(value))
            max_difference = max(max_difference, float(differences[key]))
        rows[label] = {"budget_differences": differences,
                       "raw_conserved": bool(audit["raw_conserved"]),
                       "aggregate_conserved": bool(audit["aggregate_conserved"]),
                       "endpoint_conserved": bool(audit["endpoint_conserved"])}
    if max_difference != 0.0:
        raise VariantRunnerError("033 inherited layer contract changed")
    return {"status": "passed", "source": _relative(ROOT, C0_PREFLIGHT),
            "source_sha256": sha256_file(C0_PREFLIGHT), "rows": rows,
            "max_abs_difference": max_difference}

def run_preflight(context: reconstruct.RunContext, root: Path) -> dict[str, Any]:
    """只运行 synthetic/reference-free controller 检查；绝不优化。"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "work").mkdir(parents=True, exist_ok=True)
    if context.strict_p9016:
        inherited = _inherited_layer_probe()
        archived = _archived_transfer_probe(context)
        real_layers = _real_layer_probe(context)
    else:
        inherited = {"status": "not_run_for_synthetic_context"}
        archived = {"status": "not_run_for_synthetic_context"}
        real_layers = {"status": "not_run_for_synthetic_context"}
    parity = _parity_probe()
    transfer = _transfer_probe()
    domain = _domain_probe(root)
    planned = _planned_rows()
    if len(planned) != 24 or len({row["attempt_id"] for row in planned}) != 24:
        raise VariantRunnerError("variant plan does not contain exactly 24 unique stages")
    result = {
        "schema": "multires-variant-preflight-v1", "status": "passed", "created_at_utc": _utc_now(),
        "checks": {
            "exact_24_stage_plan": True, "c0_path_equivalence": parity["status"] == "passed",
            "archived_020_transfer_and_q": archived["status"] in ("passed", "not_run_for_synthetic_context"),
            "inherited_033_full_grid_budget": inherited["status"] in ("passed", "not_run_for_synthetic_context"),
            "real_full_grid_layer_validator": real_layers["status"] in ("passed", "not_run_for_synthetic_context"),
            "variant_layer_transfer": transfer["status"] == "passed",
            "domain_specific_map_free_writer": domain["status"] == "passed",
            "no_optimization_called": True, "no_native_fdg_called": True,
            "phase_or_reference_opened": False, "r2_or_evaluation_opened": False,
        },
        "tolerances": {"state_atol": STRICT_ATOL, "state_rtol": STRICT_RTOL,
                       "finite_difference_atol": FD_ATOL, "finite_difference_rtol": FD_RTOL,
                       "state_gate": "byte_equal_preferred; numeric parity fails above atol+rtol"},
        "plan": {"variants": list(VARIANTS), "candidate_order": [candidate.candidate_id for candidate in CANDIDATES],
                 "stages": _planned_rows()},
        "inherited_033_layers": inherited, "archived_020_transfer": archived,
        "real_full_grid_layers": real_layers,
        "c0_path_parity": parity, "variant_layer_transfer": transfer,
        "domain_and_objective": domain,
        "selection_contract": {"criterion": "count_nll_per_record", "tie_tolerance_per_record": TIE_TOLERANCE,
                               "tie_order": [candidate.candidate_id for candidate in CANDIDATES],
                               "includes_diagonal": True, "includes_priors": False,
                               "reference_used": False, "phase_used": False},
    }
    return result

def prepare(outdir: str | os.PathLike[str], formal_out: str | os.PathLike[str],
            *, workers: int = 4, threads: int = 1) -> dict[str, Any]:
    """创建不可变的 preparation 证据，不启动任何变体阶段。"""
    if int(workers) < 1 or int(workers) > 4:
        raise VariantRunnerError("prepared worker count must be one to four")
    _set_threads(threads)
    context = reconstruct.production_context()
    _validate_context(context)
    root = _fresh_root(outdir)
    formal_path = Path(formal_out)
    if not formal_path.is_absolute():
        formal_path = (ROOT / formal_path).resolve()
    if os.path.lexists(formal_path):
        raise FileExistsError("planned formal output already exists: %s" % formal_path)
    started = _utc_now()
    try:
        preflight = run_preflight(context, root)
        preflight_sha = _write_exclusive(root / "preflight.json", preflight)
        protocol = _build_protocol(context, preflight_sha, root)
        protocol_sha = _write_exclusive(root / "variant_protocol.json", protocol)
        config = _build_config(context, protocol, protocol_sha, preflight_sha, formal_path, workers)
        config_sha = _write_exclusive(root / "variant_config.json", config)
        source_hashes = _source_hashes()
        source_manifest_sha = _write_exclusive(root / "source_hashes.json", {
            "schema": "multires-source-hashes-v1", "source_code_sha256": source_hashes,
            "033_c0_protocol_sha256": sha256_file(C0_PROTOCOL),
            "033_c0_preflight_sha256": sha256_file(C0_PREFLIGHT),
            "020_comparator": _comparator_hashes(),
        })
        launch_command = (
            "source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh && conda activate analysis && "
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
            "python -m pr.multires_variant_runner run-variants "
            "--out %s --c0-gate %s --protocol %s --preflight %s --workers %d --threads 1"
            % (formal_path, C0_GATE, root / "variant_protocol.json", root / "preflight.json", workers)
        )
        _write_bytes_exclusive(root / "launch_command.txt", (launch_command + "\n").encode("utf-8"))
        terminal = {
            "schema": "multires-variant-terminal-evidence-v1", "status": "prepared_not_launched",
            "runner_exit_code": 0, "started_at_utc": started, "ended_at_utc": _utc_now(),
            "planned_stage_attempt_count": 24, "executed_stage_attempt_count": 0,
            "remaining_blocked_stage_attempt_count": 24, "preflight_tests_executed": True,
            "preflight_sha256": preflight_sha, "protocol_sha256": protocol_sha,
            "training_boundary": {"phase_used": False, "reference_used": False,
                                  "oracle_coordinates_opened": False, "evaluation_outputs_opened": False,
                                  "native_fdg_started": False},
            "release": {"status": "blocked_pending_c0_gate", "gate": _relative(ROOT, C0_GATE)},
        }
        terminal_sha = _write_exclusive(root / "terminal_evidence.json", terminal)
        receipt = {
            "schema": "multires-variant-prepared-receipt-v1", "status": "prepared_not_launched",
            "runner_exit_code": 0, "started_at_utc": started, "ended_at_utc": _utc_now(),
            "artifact_root": str(root), "formal_output_path": str(formal_path),
            "planned_stage_attempt_count": 24, "executed_stage_attempt_count": 0,
            "remaining_blocked_stage_attempt_count": 24,
            "preflight": {"path": "preflight.json", "sha256": preflight_sha},
            "protocol": {"path": "variant_protocol.json", "sha256": protocol_sha},
            "config": {"path": "variant_config.json", "sha256": config_sha},
            "source_hashes": {"path": "source_hashes.json", "sha256": source_manifest_sha},
            "terminal": {"path": "terminal_evidence.json", "sha256": terminal_sha},
            "launch_command": "launch_command.txt", "c0_gate_required": _relative(ROOT, C0_GATE),
            "selection": {"criterion": "count_nll_per_record", "tie_tolerance_per_record": TIE_TOLERANCE,
                          "order": [candidate.candidate_id for candidate in CANDIDATES],
                          "reference_used": False, "phase_used": False},
            "training_boundary": terminal["training_boundary"],
        }
        receipt_sha = _write_exclusive(root / "run_receipt.json", receipt)
        readme = """# 多分辨率变体准备\n\n状态：`prepared_not_launched`。本目录只包含 C1/C2-map/C2-free/C3 的 24 个阶段计划与无 reference 预检；没有启动真实变体拟合，也没有读取 phase、reference 或 R2/evaluation 文件。\n\n每个变体/来源 独立执行 `5m 300/930 -> 2m 200/630 -> 1m 240/750`，首层从同一盲来源物理起点开始，后续只传递该变体自己的物理状态与 raw `q`。C2-free 的跨层传递与写出不调用 sphere inverse、clip、center/rescale。\n\n真实启动必须使用 `launch_command.txt`，并在固定 033 `C0_gate.json` 为 passed 后由父侧显式放行。\n"""
        _write_bytes_exclusive(root / "README.zh-CN.md", readme.encode("utf-8"))
        return {"status": "prepared_not_launched", "artifact_root": str(root),
                "preflight_path": str(root / "preflight.json"),
                "protocol_path": str(root / "variant_protocol.json"),
                "config_path": str(root / "variant_config.json"),
                "terminal_path": str(root / "terminal_evidence.json"),
                "receipt_path": str(root / "run_receipt.json"),
                "launch_command_path": str(root / "launch_command.txt"),
                "receipt_sha256": receipt_sha}
    except Exception as exc:
        failure = {"schema": "multires-variant-preparation-failure-v1", "status": "failed",
                   "created_at_utc": _utc_now(), "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc(limit=16),
                   "prepared_not_launched": True}
        if not (root / "preparation_failure.json").exists():
            _write_exclusive(root / "preparation_failure.json", failure)
        raise

def _main(argv: Sequence[str] | None = None) -> int:
    parser = __import__("argparse").ArgumentParser(description="Prepare/run post-020 multiresolution variants")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--out", required=True)
    prepare_parser.add_argument("--formal-out", required=True)
    prepare_parser.add_argument("--workers", type=int, default=4)
    prepare_parser.add_argument("--threads", type=int, default=1)
    run_parser = sub.add_parser("run-variants")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--c0-gate", required=True)
    run_parser.add_argument("--protocol")
    run_parser.add_argument("--preflight")
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.out, args.formal_out, workers=args.workers, threads=args.threads)
        else:
            result = run_variants(args.out, c0_gate=args.c0_gate, protocol_path=args.protocol,
                                  preflight_path=args.preflight, workers=args.workers, threads=args.threads)
        print(json.dumps(_jsonable(result), sort_keys=True))
        return 0
    except (VariantRunnerError, FileExistsError, reconstruct.ReconstructionError,
            reconstruction_init.InitializationError, ValueError, OSError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(_main())
