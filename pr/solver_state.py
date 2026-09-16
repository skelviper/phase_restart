"""不可变、全有限 solver state 与独立 presence-aware 3DG 导出。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from . import contact_model


class SolverStateError(RuntimeError):
    """solver state 或导出契约不满足。"""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_state(*, coordinates: np.ndarray, raw_y: np.ndarray, theta: np.ndarray,
                   p: float, q: float, atol: float = 1e-12) -> dict[str, Any]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    raw_y = np.asarray(raw_y, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    p = float(p)
    q = float(q)
    if coordinates.ndim != 3 or coordinates.shape[0] != 2 or coordinates.shape[2] != 3:
        raise SolverStateError("coordinates must have shape (2,n_loci,3)")
    if raw_y.shape != coordinates.shape:
        raise SolverStateError("raw_y shape differs from coordinates")
    if theta.shape != (coordinates.size + 1,):
        raise SolverStateError("theta shape differs from flattened raw_y plus q")
    arrays = (coordinates, raw_y, theta, np.asarray([p, q], dtype=np.float64))
    if not all(np.isfinite(values).all() for values in arrays):
        raise SolverStateError("solver state must be entirely finite")
    theta_raw_error = float(np.max(np.abs(theta[:-1] - raw_y.reshape(-1))))
    theta_q_error = float(abs(theta[-1] - q))
    mapped = contact_model.sphere_forward(raw_y)
    coordinate_error = float(np.max(np.abs(mapped - coordinates)))
    mapped_p = float(contact_model.p_from_q(q)[0])
    p_error = float(abs(mapped_p - p))
    errors = {
        "theta_raw_y_max_abs": theta_raw_error,
        "theta_q_abs": theta_q_error,
        "sphere_coordinate_max_abs": coordinate_error,
        "q_p_abs": p_error,
    }
    if max(errors.values()) > float(atol):
        raise SolverStateError("inconsistent solver state: %r" % errors)
    contact_model.assert_inside_unit_ball(coordinates)
    return {
        "shape": list(coordinates.shape),
        "finite": True,
        "consistency_atol": float(atol),
        **errors,
    }


def write_solver_state(path: str | Path, *, coordinates: np.ndarray, raw_y: np.ndarray,
                       theta: np.ndarray, p: float, q: float,
                       atol: float = 1e-12) -> dict[str, Any]:
    path = Path(path)
    audit = validate_state(coordinates=coordinates, raw_y=raw_y, theta=theta,
                           p=p, q=q, atol=atol)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("xb")
    except FileExistsError as exc:
        raise SolverStateError("refusing to overwrite solver state: %s" % path) from exc
    with handle:
        np.savez_compressed(
            handle,
            coordinates=np.asarray(coordinates, dtype=np.float64),
            raw_y=np.asarray(raw_y, dtype=np.float64),
            theta=np.asarray(theta, dtype=np.float64),
            p=np.asarray(float(p), dtype=np.float64),
            q=np.asarray(float(q), dtype=np.float64),
        )
    loaded = load_solver_state(path, atol=atol)
    return {"path": str(path), "sha256": sha256_file(path), **audit,
            "readback": loaded["audit"]}


def load_solver_state(path: str | Path, *, atol: float = 1e-12) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {"coordinates", "raw_y", "theta", "p", "q"}
        if set(payload.files) != required:
            raise SolverStateError("solver state keys differ: %r" % sorted(payload.files))
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        raw_y = np.asarray(payload["raw_y"], dtype=np.float64).copy()
        theta = np.asarray(payload["theta"], dtype=np.float64).copy()
        p = float(np.asarray(payload["p"]).item())
        q = float(np.asarray(payload["q"]).item())
    audit = validate_state(coordinates=coordinates, raw_y=raw_y, theta=theta,
                           p=p, q=q, atol=atol)
    return {"coordinates": coordinates, "raw_y": raw_y, "theta": theta,
            "p": p, "q": q, "audit": audit, "sha256": sha256_file(path)}


def write_presence_mask(path: str | Path, mask: np.ndarray, n_loci: int) -> dict[str, Any]:
    path = Path(path)
    mask = np.asarray(mask)
    if mask.shape != (2, int(n_loci)) or mask.dtype != np.bool_:
        raise SolverStateError("presence mask must be bool shape (2,n_loci)")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("xb")
    except FileExistsError as exc:
        raise SolverStateError("refusing to overwrite presence mask: %s" % path) from exc
    with handle:
        np.savez_compressed(handle, presence=mask)
    with np.load(path, allow_pickle=False) as payload:
        readback = np.asarray(payload["presence"])
    if readback.dtype != np.bool_ or not np.array_equal(readback, mask):
        raise SolverStateError("presence mask readback mismatch")
    return {"path": str(path), "sha256": sha256_file(path),
            "shape": list(mask.shape), "present": int(mask.sum()),
            "absent": int(mask.size - mask.sum())}


def export_present_3dg(path: str | Path, data: Any, coordinates: np.ndarray,
                       presence: np.ndarray) -> dict[str, Any]:
    path = Path(path)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    presence = np.asarray(presence)
    if coordinates.shape != (2, int(data.n_loci), 3) or not np.isfinite(coordinates).all():
        raise SolverStateError("export coordinates must be finite full-grid coordinates")
    if presence.shape != (2, int(data.n_loci)) or presence.dtype != np.bool_:
        raise SolverStateError("export presence mask has wrong shape or dtype")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise SolverStateError("refusing to overwrite 3DG export: %s" % path) from exc
    rows = 0
    by_track: dict[str, int] = {}
    with handle:
        for spec in data.track_specs:
            slc = data.chromosome_slice(spec.chromosome_index)
            kept = 0
            for global_index in range(slc.start, slc.stop):
                if not presence[spec.copy_index, global_index]:
                    continue
                position = int(data.locus_bin[global_index]) * int(data.bin_size)
                xyz = coordinates[spec.copy_index, global_index]
                handle.write("%s\t%d\t%.17g\t%.17g\t%.17g\n" %
                             (spec.name, position, xyz[0], xyz[1], xyz[2]))
                rows += 1
                kept += 1
            by_track[spec.name] = kept
    if rows != int(presence.sum()):
        raise SolverStateError("export row count differs from presence mask")
    return {"path": str(path), "sha256": sha256_file(path), "rows": rows,
            "by_track": by_track, "bin_size": int(data.bin_size),
            "track_count": len(data.track_specs)}


__all__ = [
    "SolverStateError", "export_present_3dg", "load_solver_state", "sha256_file",
    "validate_state", "write_presence_mask", "write_solver_state",
]
