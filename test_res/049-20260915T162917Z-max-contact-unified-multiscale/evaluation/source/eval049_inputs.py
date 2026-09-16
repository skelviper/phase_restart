#!/usr/bin/env python
"""049 输入适配层：对接训练侧冻结 schema（不猜 NPZ keys，只读已声明字段）。

冻结 schema（049 run 2026-09-15 期间由父侧给定）：
- 端点坐标：``049/coords/{fit_id}/1Mb.npz``（keys: coordinates/raw_y/theta/p/q）+ ``1Mb.3dg``
- 逐 fit 终态：``049/stages/{fit_id}/1Mb.json`` 与 ``049/results/fits/{fit_id}.json``
- 汇总 manifest：``049/results/endpoint_manifest_pre_reference.json``
- 选择：``049/results/selection_pre_reference.json``

字段名允许少量别名（见别名表），缺失时抛出带可用 key 列表的显式错误，不静默取默认值。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import eval049_lib as lib

ALIASES: dict[str, tuple[str, ...]] = {
    "fit_id": ("fit_id", "id", "name"),
    "loss": ("loss", "loss_id"),
    "solver": ("solver", "solver_id"),
    "source": ("source", "blind_source", "source_id"),
    "npz_path": ("coords_npz_path", "npz_path", "coordinates_npz_path", "npz", "coords_path"),
    "npz_sha256": ("coords_npz_sha256", "npz_sha256", "coordinate_npz_sha256", "coords_sha256"),
    "tdg_path": ("three_dg_path", "tdg_path", "3dg_path", "coords_3dg_path", "three_dg"),
    "tdg_sha256": ("three_dg_sha256", "tdg_sha256", "3dg_sha256", "coordinate_3dg_sha256"),
    "terminal": ("terminal", "status", "terminal_state"),
    "outer_fg": ("outerFG", "outer_fg", "outer_fg_actual", "accepted_fg", "fg_accepted", "fg"),
    "full_j": ("fullJ", "fullJ_value", "full_j", "full_j_final", "full_j_value"),
    "count_nll_by_loss": ("count_nll_by_loss", "count_nll", "count_objectives"),
    "coordinate_array_sha256": ("coordinate_array_sha256", "coords_array_sha256", "coordinates_array_sha256"),
    "kind": ("kind", "role"),
    "p": ("p", "p_final"),
    "q": ("q", "q_final"),
    "q_source": ("q_source", "q_provenance"),
    "wall_s": ("fit_wall_seconds", "wall_s", "wall_seconds", "wall", "wall_time_seconds"),
    "canonical_grad": ("canonical_gradient_max_abs", "canonical_grad_inf", "canonical_gradient_inf_norm",
                       "canonical_gtol_observed", "canonical_gradient_max"),
    "full_j_components": ("fullJ_components", "full_j_components", "penalty_components"),
}

# 12 fits 的冻结 schema 要求这些字段必须存在（不允许静默变成 None）
FIT_REQUIRED_FIELDS = ("terminal", "outer_fg", "full_j", "count_nll_by_loss", "p", "q")
INITIAL_BASELINE_REQUIRED_FIELDS = ("terminal",)


def pick(record: Mapping[str, Any], field: str, *, required: bool = True) -> Any:
    for alias in ALIASES[field]:
        if alias in record and record[alias] is not None:
            return record[alias]
    if required:
        raise RuntimeError("manifest record missing field '%s' (aliases %s); available keys: %s"
                           % (field, list(ALIASES[field]), sorted(record.keys())))
    return None


# 全部可识别字段：无论 required_fields 如何，都要逐字段 pick 并保留（控制记录也保留可用元数据）
RECOGNIZED_FIELDS = ("outer_fg", "full_j", "count_nll_by_loss", "p", "q", "q_source", "wall_s",
                     "canonical_grad", "full_j_components", "coordinate_array_sha256", "kind")


def normalize_record(record: Mapping[str, Any], *, require_paths: bool = True, require_tdg: bool = True,
                     required_fields: Sequence[str] = FIT_REQUIRED_FIELDS) -> dict[str, Any]:
    """逐字段 pick：required 只决定“缺失是否报错”，不决定“是否存在输出里”。"""
    out: dict[str, Any] = {"fit_id": pick(record, "fit_id", required=True)}
    for field in ("loss", "solver", "source", "terminal") + RECOGNIZED_FIELDS:
        out[field] = pick(record, field, required=(field in required_fields))
    for field in ("npz_path", "npz_sha256"):
        out[field] = pick(record, field, required=require_paths)
    for field in ("tdg_path", "tdg_sha256"):
        out[field] = pick(record, field, required=bool(require_paths and require_tdg))
    out["raw_record"] = dict(record)
    return out


def load_manifest(path: Path) -> dict[str, Any]:
    """把 endpoint_manifest_pre_reference.json 归一化为 fits/initials/baseline 三段。"""
    path = Path(path)
    payload = lib.read_json(path)
    fits_raw: Sequence[Mapping[str, Any]] = []
    initials_raw: Sequence[Mapping[str, Any]] = []
    baseline_raw: Mapping[str, Any] | None = None
    for key in ("fits", "endpoints", "fit_endpoints"):
        if isinstance(payload.get(key), list):
            fits_raw = payload[key]
            break
    for key in ("initials", "initial_controls", "initial"):
        if isinstance(payload.get(key), list):
            initials_raw = payload[key]
            break
    for key in ("baseline", "frozen_baseline"):
        if isinstance(payload.get(key), Mapping):
            baseline_raw = payload[key]
            break
    if not fits_raw:
        raise RuntimeError("endpoint manifest has no fits/endpoints list; top-level keys: %s" % sorted(payload.keys()))
    fits = [normalize_record(row, required_fields=FIT_REQUIRED_FIELDS) for row in fits_raw]
    initials = [normalize_record(row, require_paths=True, require_tdg=False,
                                 required_fields=INITIAL_BASELINE_REQUIRED_FIELDS)
                for row in initials_raw] if initials_raw else []
    baseline = (normalize_record(baseline_raw, require_paths=True, require_tdg=False,
                                 required_fields=INITIAL_BASELINE_REQUIRED_FIELDS) if baseline_raw else None)
    fit_ids = [str(row["fit_id"]) for row in fits]
    if len(fit_ids) != 12 or len(set(fit_ids)) != 12:
        raise RuntimeError("endpoint manifest must contain exactly 12 unique fits, got %d/%d"
                           % (len(fit_ids), len(set(fit_ids))))
    missing = [fid for fid in lib.EXPECTED_FIT_IDS if fid not in set(fit_ids)]
    if missing:
        raise RuntimeError("endpoint manifest missing fit ids: %s" % missing)
    for row in fits:
        fid = str(row["fit_id"])
        loss, solver, source = fid.split("-")
        for field, expected in (("loss", loss), ("solver", solver), ("source", source)):
            if row.get(field) is not None and str(row[field]) != expected:
                raise RuntimeError("fit %s field %s=%r disagrees with fit_id" % (fid, field, row[field]))
            row[field] = expected
    return {"path": path, "sha256": lib.sha256_file(path), "fits": fits, "initials": initials,
            "baseline": baseline, "raw": payload}


def load_selection(path: Path) -> dict[str, Any]:
    """归一化 selection_pre_reference.json：source 选择 + per-loss display endpoint。"""
    path = Path(path)
    payload = lib.read_json(path)
    if payload.get("reference_opened") not in (False, None):
        raise RuntimeError("selection file claims reference was opened: %s" % path)
    per_loss = payload.get("per_loss")
    if not isinstance(per_loss, Mapping) or set(per_loss) < set(lib.LOSES):
        raise RuntimeError("selection file missing per_loss{A,B,C}; keys: %s" % sorted(payload.keys()))
    source_selection: dict[str, Any] = {}
    display: dict[str, Any] = {}
    for loss in lib.LOSES:
        block = per_loss[loss]
        ss = block.get("source_selection")
        if not isinstance(ss, Mapping):
            raise RuntimeError("per_loss.%s missing source_selection" % loss)
        for solver in lib.SOLVERS:
            entry = ss.get(solver)
            if not isinstance(entry, Mapping):
                raise RuntimeError("per_loss.%s.source_selection missing %s" % (loss, solver))
            selected = None
            for key in ("selected_source", "selected", "winner", "source"):
                if entry.get(key) in lib.SOURCES:
                    selected = str(entry[key])
                    break
            if selected is None:
                raise RuntimeError("per_loss.%s.source_selection.%s has no selected_source in %s; keys: %s"
                                   % (loss, solver, list(lib.SOURCES), sorted(entry.keys())))
            source_selection["%s-%s" % (loss, solver)] = {
                "selected_source": selected, "rule": entry.get("rule"),
                "candidates": entry.get("candidates"), "margin": entry.get("margin"),
                "raw_entry": dict(entry),
            }
        disp = block.get("display_endpoint")
        if not isinstance(disp, Mapping):
            raise RuntimeError("per_loss.%s missing display_endpoint" % loss)
        solver, source = disp.get("solver"), disp.get("source")
        if solver not in lib.SOLVERS or source not in lib.SOURCES:
            raise RuntimeError("per_loss.%s.display_endpoint invalid solver/source: %s" % (loss, sorted(disp.keys())))
        display[loss] = {"fit_id": "%s-%s-%s" % (loss, solver, source), "solver": solver, "source": source,
                         "count": disp.get("count"), "coords_sha256": disp.get("coordsSHA", disp.get("coords_sha256")),
                         "raw_entry": dict(disp)}
    return {"path": path, "sha256": lib.sha256_file(path), "source_selection": source_selection,
            "display": display, "raw": payload}
