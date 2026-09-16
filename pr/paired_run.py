"""Post-020 allele sensitivity variants 的单层 paired runner。

本 runner 有意不拥有 native initialization。调用者提供三个已 materialize、成对的 physical starts（例如来自独立、显式的 full-grid native bridge）。全部五个变体使用一个 1 Mb objective layer 和调用者提供的 L-BFGS-B budget；这里不选择 budget。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from . import contact_model
from .allele_models import (
    MODEL_IDS,
    P_INIT,
    model_spec,
    objective_for_model,
    physical_coordinates_from_raw,
    raw_coordinates_from_physical,
    validate_physical_for_model,
)
from .joint_fit import JointCheckpoint


SELECTION_TIE_TOL = 1e-12


class PairedRunError(RuntimeError):
    """paired-run contract 在拟合前或拟合中无效时抛出。"""


@dataclass(frozen=True)
class FitConfig:
    """显式的单层 solver 设置；maxiter 和 maxfun 没有默认值。"""

    maxiter: int
    maxfun: int
    maxls: int = 20
    ftol: float = 1e-10
    gtol: float = 1e-6
    checkpoint_every: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.maxiter, (int, np.integer)) or int(self.maxiter) <= 0:
            raise ValueError("maxiter must be a positive integer supplied by the frozen config")
        if not isinstance(self.maxfun, (int, np.integer)) or int(self.maxfun) <= 0:
            raise ValueError("maxfun must be a positive integer supplied by the frozen config")
        if not isinstance(self.maxls, (int, np.integer)) or int(self.maxls) <= 0:
            raise ValueError("maxls must be a positive integer")
        for name, value in (("ftol", self.ftol), ("gtol", self.gtol)):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("%s must be finite and nonnegative" % name)
        if self.checkpoint_every is not None and int(self.checkpoint_every) <= 0:
            raise ValueError("checkpoint_every must be positive when supplied")

    def scipy_options(self) -> dict[str, int | float]:
        return {
            "maxiter": int(self.maxiter),
            "maxfun": int(self.maxfun),
            "maxls": int(self.maxls),
            "ftol": float(self.ftol),
            "gtol": float(self.gtol),
        }

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "maxiter": int(self.maxiter),
            "maxfun": int(self.maxfun),
            "maxls": int(self.maxls),
            "ftol": float(self.ftol),
            "gtol": float(self.gtol),
            "checkpoint_every": (
                None if self.checkpoint_every is None else int(self.checkpoint_every)
            ),
        }


@dataclass(frozen=True)
class PairedStart:
    """所有 model 变体逐字节共享的一份 physical start。"""

    start_id: str
    coordinates: np.ndarray
    p_init: float = P_INIT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.start_id):
            raise ValueError("start_id must be nonempty")
        values = np.asarray(self.coordinates, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] != 2 or values.shape[2] != 3:
            raise ValueError("coordinates must have shape (2, n_loci, 3)")
        if not np.all(np.isfinite(values)):
            raise ValueError("paired start coordinates must be finite")
        object.__setattr__(self, "coordinates", values.copy())
        if not math.isfinite(float(self.p_init)):
            raise ValueError("p_init must be finite")
        # q_from_p 提供与 V1 pack() 完全相同的严格 bounded interval。
        contact_model.q_from_p(float(self.p_init))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

    @property
    def coordinate_sha256(self) -> str:
        values = np.asarray(self.coordinates, dtype="<f8", order="C")
        return hashlib.sha256(values.tobytes(order="C")).hexdigest()

    @property
    def max_radius(self) -> float:
        return float(np.linalg.norm(np.asarray(self.coordinates, dtype=np.float64), axis=2).max())

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_id": str(self.start_id),
            "p_init": float(self.p_init),
            "coordinate_sha256": self.coordinate_sha256,
            "coordinate_shape": list(np.asarray(self.coordinates).shape),
            "max_radius": self.max_radius,
            "metadata": _jsonable(dict(self.metadata)),
        }


@dataclass
class PairedFitResult:
    """带显式 domain 和 map audit 字段的单层数值结果。"""

    model_id: str
    start_id: str
    objective: contact_model.JointObjective
    theta: np.ndarray
    raw_coordinates: np.ndarray
    coordinates: np.ndarray
    p: float
    fun: float
    components: dict[str, Any]
    success: bool
    status: int
    message: str
    nit: int
    nfev: int
    scipy_nfev: int
    njev: int
    elapsed_seconds: float
    history: list[dict[str, Any]]
    initial_total: float
    initial_roundtrip_max_abs: float
    initial_coordinate_sha256: str
    map_diagnostics: dict[str, Any]

    @property
    def count_nll_normalized(self) -> float:
        return float(self.components["count_nll_normalized"])

    def as_dict(self) -> dict[str, Any]:
        spec = model_spec(self.model_id)
        return {
            "model": spec.as_dict(),
            "model_id": self.model_id,
            "start_id": self.start_id,
            "initial_coordinate_sha256": self.initial_coordinate_sha256,
            "initial_total": float(self.initial_total),
            "initial_roundtrip_max_abs": float(self.initial_roundtrip_max_abs),
            "final_total": float(self.fun),
            "final_count_nll_normalized": self.count_nll_normalized,
            "p": float(self.p),
            "q": float(self.theta[-1]),
            "success": bool(self.success),
            "status": int(self.status),
            "message": self.message,
            "nit": int(self.nit),
            "nfev": int(self.nfev),
            "scipy_nfev": int(self.scipy_nfev),
            "njev": int(self.njev),
            "elapsed_seconds": float(self.elapsed_seconds),
            "components": _jsonable(dict(self.components)),
            "map_diagnostics": _jsonable(dict(self.map_diagnostics)),
            "history": _jsonable(self.history),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite value cannot enter paired-run metadata")
    return value


def _coordinates_hash(coordinates: np.ndarray) -> str:
    values = np.asarray(coordinates, dtype="<f8", order="C")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def save_paired_start(path: str | Path, start: PairedStart) -> Path:
    """写出一份 physical start，不改变它，也不加入隐藏 rescale。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(_jsonable(dict(start.metadata)), sort_keys=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            coordinates=np.asarray(start.coordinates, dtype=np.float64),
            start_id=np.asarray(str(start.start_id)),
            p_init=np.asarray(float(start.p_init), dtype=np.float64),
            metadata_json=np.asarray(metadata),
        )
    return path


def load_paired_start(path: str | Path, start_id: str | None = None) -> PairedStart:
    """加载已 materialize 的 physical start，不访问 native/reference。"""
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        if "coordinates" not in payload:
            raise PairedRunError("paired start lacks coordinates: %s" % path)
        coordinates = payload["coordinates"].copy()
        stored_id = str(payload["start_id"].item()) if "start_id" in payload else path.stem
        p_init = float(payload["p_init"].item()) if "p_init" in payload else P_INIT
        metadata = {}
        if "metadata_json" in payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
    return PairedStart(start_id=stored_id if start_id is None else str(start_id),
                       coordinates=coordinates, p_init=p_init, metadata=metadata)


def _validate_start_for_design(data: contact_model.AggregatedContacts,
                               start: PairedStart) -> None:
    values = np.asarray(start.coordinates, dtype=np.float64)
    if values.shape != (2, data.n_loci, 3):
        raise PairedRunError(
            "%s coordinates have shape %s, expected (2,%d,3)" %
            (start.start_id, values.shape, data.n_loci))
    # C0 和 C2-map 都存在，因此每个共享 paired start 即使 C2-free 本身能接受更大范围，
    # 也必须对 strict unit-ball variants 有效。
    validate_physical_for_model("C0", values)
    contact_model.q_from_p(float(start.p_init))


def plan_paired_runs(data: contact_model.AggregatedContacts,
                     starts: Sequence[PairedStart],
                     model_ids: Sequence[str] = MODEL_IDS) -> list[dict[str, Any]]:
    """创建可审计的 5-by-N plan，不评估也不优化任何内容。"""
    data.assert_consistent()
    if int(data.bin_size) != 1_000_000:
        raise PairedRunError("post-020 paired runner is frozen to a 1 Mb layer")
    starts = tuple(starts)
    if not starts:
        raise PairedRunError("at least one paired start is required")
    if len({str(start.start_id) for start in starts}) != len(starts):
        raise PairedRunError("paired start ids must be unique")
    model_ids = tuple(str(model_id) for model_id in model_ids)
    if not model_ids:
        raise PairedRunError("at least one model variant is required")
    if len(set(model_ids)) != len(model_ids):
        raise PairedRunError("model ids must be unique")
    for model_id in model_ids:
        model_spec(model_id)
    for start in starts:
        _validate_start_for_design(data, start)
    rows = []
    for model_id in model_ids:
        spec = model_spec(model_id)
        for start in starts:
            rows.append({
                "model_id": model_id,
                "start_id": str(start.start_id),
                "initial_coordinate_sha256": start.coordinate_sha256,
                "initial_max_radius": start.max_radius,
                "p_init": float(start.p_init),
                "coordinate_parameterization": spec.coordinate_parameterization,
                "physical_domain": spec.physical_domain,
                "exposure_change": spec.exposure_change,
                "fit_not_run": True,
            })
    return rows


def _default_map_diagnostics(objective: contact_model.JointObjective) -> dict[str, Any]:
    return {
        "map": "sphere_forward_unit_ball",
        "physical_domain": "strict_unit_ball",
        "objective_eval_count": None,
        "nonidentity_map_eval_count": None,
        "nonidentity_map_eval_fraction": None,
        "counts_only_evaluate_calls": False,
        "line_search_probes_included": "shared_objective_not_instrumented",
    }


def _map_diagnostics(objective: contact_model.JointObjective) -> dict[str, Any]:
    method = getattr(objective, "map_diagnostics", None)
    if method is None:
        return _default_map_diagnostics(objective)
    return dict(method())


def run_one_fit(data: contact_model.AggregatedContacts, start: PairedStart,
                model_id: str, fit_config: FitConfig,
                block_size: int = contact_model.DEFAULT_BLOCK_SIZE,
                repulsion_block_size: int | None = None,
                callback: Callable[[dict[str, Any]], None] | None = None,
                checkpoint_hook: Callable[[JointCheckpoint], None] | None = None) -> PairedFitResult:
    """运行一个显式 budget 的 1 Mb 拟合；没有默认值，也不裁剪坐标。"""
    _check_model_in_start(model_id)
    data.assert_consistent()
    if int(data.bin_size) != 1_000_000:
        raise PairedRunError("run_one_fit requires a 1 Mb AggregatedContacts layer")
    _validate_start_for_design(data, start)
    objective = objective_for_model(
        data, model_id, block_size=block_size,
        repulsion_block_size=repulsion_block_size,
    )
    initial_coordinates = np.asarray(start.coordinates, dtype=np.float64).copy()
    raw_initial = raw_coordinates_from_physical(objective, initial_coordinates)
    theta0 = objective.pack(raw_initial, p=float(start.p_init))
    roundtrip = physical_coordinates_from_raw(objective, raw_initial)
    roundtrip_error = float(np.max(np.abs(roundtrip - initial_coordinates)))
    if not math.isfinite(roundtrip_error) or roundtrip_error > 5e-12:
        raise PairedRunError(
            "%s initial physical round-trip error %.6g exceeds 5e-12" %
            (model_id, roundtrip_error))

    started = time.perf_counter()
    initial_value, _, initial_components = objective.evaluate(theta0, need_gradient=False)
    history: list[dict[str, Any]] = [{
        "iteration": 0,
        "nfev": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "fun": float(initial_value),
        "p": float(initial_components["p"]),
        "components": dict(initial_components),
    }]
    actual_nfev = 0
    last_theta: np.ndarray | None = None
    last_gradient: np.ndarray | None = None

    def loss_and_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal actual_nfev, last_theta, last_gradient
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
        if (last_theta is not None and last_gradient is not None
                and np.array_equal(np.asarray(theta, dtype=np.float64), last_theta)):
            gradient_norm = float(np.linalg.norm(last_gradient))
        else:
            _, gradient, _ = objective.evaluate(theta, need_gradient=True)
            gradient_norm = float(np.linalg.norm(gradient))
        entry = {
            "iteration": len(history),
            "nfev": int(actual_nfev),
            "elapsed_seconds": time.perf_counter() - started,
            "fun": float(value),
            "p": float(components["p"]),
            "gradient_norm": gradient_norm,
            "components": dict(components),
        }
        history.append(entry)
        if (checkpoint_hook is not None and fit_config.checkpoint_every is not None
                and entry["iteration"] % int(fit_config.checkpoint_every) == 0):
            raw, _ = objective.unpack(theta)
            coordinates, p = objective.coordinates_and_p(theta)
            checkpoint_hook(JointCheckpoint(
                iteration=int(entry["iteration"]),
                nfev=int(actual_nfev),
                elapsed_seconds=float(entry["elapsed_seconds"]),
                fun=float(value),
                p=float(p),
                theta=np.asarray(theta, dtype=np.float64).copy(),
                y=raw.copy(),
                coordinates=coordinates.copy(),
                components=dict(components),
                gradient_norm=gradient_norm,
            ))
        if callback is not None:
            callback(dict(entry))

    raw_result: OptimizeResult = minimize(
        loss_and_gradient,
        theta0,
        method="L-BFGS-B",
        jac=True,
        callback=recorder,
        options=fit_config.scipy_options(),
    )
    raw_final, _ = objective.unpack(raw_result.x)
    final_coordinates, final_p = objective.coordinates_and_p(raw_result.x)
    _validate_objective_coordinates(objective, final_coordinates)
    cached = objective.cached_value_and_components(raw_result.x)
    if cached is None:
        final_value, _, final_components = objective.evaluate(raw_result.x, need_gradient=False)
    else:
        final_value, final_components = cached
    elapsed = time.perf_counter() - started
    if not history or history[-1]["fun"] != float(final_value):
        history.append({
            "iteration": len(history),
            "nfev": int(actual_nfev),
            "elapsed_seconds": elapsed,
            "fun": float(final_value),
            "p": float(final_p),
            "components": dict(final_components),
        })
    return PairedFitResult(
        model_id=str(model_id),
        start_id=str(start.start_id),
        objective=objective,
        theta=np.asarray(raw_result.x, dtype=np.float64).copy(),
        raw_coordinates=raw_final.copy(),
        coordinates=np.asarray(final_coordinates, dtype=np.float64).copy(),
        p=float(final_p),
        fun=float(final_value),
        components=dict(final_components),
        success=bool(raw_result.success),
        status=int(raw_result.status),
        message=str(raw_result.message),
        nit=int(raw_result.nit),
        nfev=int(actual_nfev),
        scipy_nfev=int(getattr(raw_result, "nfev", actual_nfev)),
        njev=int(getattr(raw_result, "njev", 0) or 0),
        elapsed_seconds=float(elapsed),
        history=history,
        initial_total=float(initial_value),
        initial_roundtrip_max_abs=roundtrip_error,
        initial_coordinate_sha256=start.coordinate_sha256,
        map_diagnostics=_map_diagnostics(objective),
    )


def _check_model_in_start(model_id: str) -> None:
    try:
        model_spec(model_id)
    except ValueError as exc:
        raise PairedRunError(str(exc)) from exc


def _validate_objective_coordinates(objective: contact_model.JointObjective,
                                     coordinates: np.ndarray) -> None:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (2, objective.data.n_loci, 3):
        raise PairedRunError("objective returned coordinates with an invalid shape")
    validator = getattr(objective, "validate_physical_coordinates", None)
    if validator is not None:
        validator(values)
    else:
        contact_model.assert_inside_unit_ball(values)


def run_paired(data: contact_model.AggregatedContacts,
               starts: Sequence[PairedStart], fit_config: FitConfig,
               model_ids: Sequence[str] = MODEL_IDS,
               block_size: int = contact_model.DEFAULT_BLOCK_SIZE,
               repulsion_block_size: int | None = None) -> list[PairedFitResult]:
    """使用一份共享的显式 solver config 运行请求的 paired variants。"""
    plan_paired_runs(data, starts, model_ids=model_ids)
    results: list[PairedFitResult] = []
    for model_id in tuple(model_ids):
        for start in tuple(starts):
            results.append(run_one_fit(
                data, start, model_id, fit_config,
                block_size=block_size,
                repulsion_block_size=repulsion_block_size,
            ))
    return results


def select_best_start(results: Sequence[PairedFitResult],
                      tie_tol: float = SELECTION_TIE_TOL) -> PairedFitResult:
    """在一个 model 内使用有限 count window 和稳定 start id 选择最优项。"""
    results = tuple(results)
    if not results:
        raise PairedRunError("cannot select from an empty result set")
    if not math.isfinite(float(tie_tol)) or float(tie_tol) < 0.0:
        raise ValueError("tie_tol must be finite and nonnegative")
    model_set = {result.model_id for result in results}
    if len(model_set) != 1:
        raise PairedRunError("selection cannot compare different prior/exposure variants")
    values: list[tuple[PairedFitResult, float]] = []
    start_ids: set[str] = set()
    for result in results:
        start_id = str(result.start_id)
        if start_id in start_ids:
            raise PairedRunError("selection requires unique start ids")
        start_ids.add(start_id)
        value = float(result.count_nll_normalized)
        if not math.isfinite(value):
            raise PairedRunError(
                "%s/%s has nonfinite count_nll_normalized and is not selectable" %
                (result.model_id, start_id))
        values.append((result, value))
    minimum = min(value for _result, value in values)
    eligible = [item for item in values if item[1] <= minimum + float(tie_tol)]
    return min(eligible, key=lambda item: str(item[0].start_id))[0]


def write_coordinates(path: str | Path, data: contact_model.AggregatedContacts,
                      model_id: str, coordinates: np.ndarray) -> dict[str, Any]:
    """写出带 model-specific validation 且不裁剪的 full-grid 坐标文件。"""
    path = Path(path)
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (2, data.n_loci, 3):
        raise PairedRunError("coordinates must have shape (2, n_loci, 3)")
    validate_physical_for_model(model_id, values)
    with path.open("x", encoding="utf-8") as handle:
        for spec in data.track_specs:
            chromosome_slice = data.chromosome_slice(spec.chromosome_index)
            for global_index in range(chromosome_slice.start, chromosome_slice.stop):
                position = int(data.locus_bin[global_index] * data.bin_size)
                xyz = values[spec.copy_index, global_index]
                handle.write("%s\t%d\t%.17g\t%.17g\t%.17g\n" %
                             (spec.name, position, xyz[0], xyz[1], xyz[2]))
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "model_id": str(model_id),
        "physical_domain": model_spec(model_id).physical_domain,
        "full_grid": True,
        "n_tracks": len(data.track_specs),
        "n_beads": int(2 * data.n_loci),
        "max_radius": float(np.linalg.norm(values, axis=2).max()),
        "serialization_clip": {"clipped_coordinates": 0, "applied": False},
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "FitConfig",
    "PairedFitResult",
    "PairedRunError",
    "PairedStart",
    "SELECTION_TIE_TOL",
    "load_paired_start",
    "plan_paired_runs",
    "run_one_fit",
    "run_paired",
    "save_paired_start",
    "select_best_start",
    "write_coordinates",
]
