"""Reconstruction V1 的受守门最终评估与交付编排。

本模块是完成训练运行后使用的评估侧入口。它有意明确无 phase training 与评估输入的边界：

* 在调用任何评估器加载器 前，先检查 selection/provenance/candidate 坐标；
* 先将坐标路径注册到新的 :class:`EvalGate`，再 arm gate 以访问 phase/reference；
* contacts、labels 和 reference 通过注入回调加载，使完整顺序可用 synthetic fixture 测试；
* 绝不根据 reference metrics 重新计算 selection。``selected_id`` 是 ``selection.json`` 携带的训练时选择。

默认回调是生产 P9016 回调。测试和未来的正式 wrapper 可以注入与 ``reconstruction_report.guarded_final_evaluation_inputs`` 相同签名的回调。
"""
from __future__ import annotations

import argparse
import datetime as dt
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import genome, ref3dg, refeval, reconstruction_report as report
from .gate import EvalGate, sha256_file
from .paths import BIN, OFF, ROOT


EVALUATION_SCHEMA_VERSION = "reconstruction-evaluation-v1"
EVALUATION_CODE_SNAPSHOT_SCHEMA_VERSION = "reconstruction-evaluation-code-snapshot-v1"
P9016_REFERENCE_3DG_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
EVALUATION_CODE_SOURCES = (
    "pr/reconstruction_evaluate.py",
    "pr/reconstruction_report.py",
    "pr/refeval.py",
    "pr/ref3dg.py",
    "pr/genome.py",
    "pr/pairs7.py",
    "pr/gate.py",
)


class EvaluationError(RuntimeError):
    """最终评估编排或交付无效时抛出。"""


def _safe(value: Any) -> Any:
    """将 NumPy/path 值转换为严格兼容 JSON 的基础类型。"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _write_json(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_cell(value: Any) -> str:
    return json.dumps(_safe(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _resolve_asset_path(value: Any, base: Path) -> Path:
    """解析正式运行保存的路径，但不打开其 payload。"""
    if not isinstance(value, str) or not value:
        raise EvaluationError("asset path must be a non-empty string")
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = (base / path, Path(ROOT) / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    # 为后续守门保留确定性的路径，以便报告为 unavailable。
    return (base / path).resolve()


def baseline_specs_from_selection(
    selection_path: str | os.PathLike[str],
    document: Mapping[str, Any] | None = None,
) -> dict[str, report.BaselineSpec]:
    """从 selection 元数据构建显式的 baseline source 规格。

    ``pr.reconstruct`` 将 014 source gate 和坐标引用保存在 ``source_gate``/``source_coordinate`` 下。该辅助函数只读取 selection 元数据；source 字节由 ``verify_baseline_source`` 在稍后打开。
    """
    selection_file = Path(selection_path).resolve()
    if document is None:
        try:
            document = json.loads(selection_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError("cannot read selection metadata for baseline specs") from exc
    assets = document.get("baseline_assets") if isinstance(document, Mapping) else None
    if not isinstance(assets, list):
        raise EvaluationError("selection baseline_assets must be a list")
    result: dict[str, report.BaselineSpec] = {}
    for index, raw in enumerate(assets):
        if not isinstance(raw, Mapping):
            raise EvaluationError("baseline_assets[%d] must be an object" % index)
        tag = raw.get("tag")
        role = raw.get("role")
        gate = raw.get("source_gate") or raw.get("source_gate_path")
        coordinate = raw.get("source_coordinate") or raw.get("coordinates")
        if isinstance(gate, Mapping):
            gate_path = gate.get("path")
            gate_sha = gate.get("sha256")
        else:
            gate_path = gate
            gate_sha = raw.get("source_gate_sha256")
        if isinstance(coordinate, Mapping):
            coordinate_path = coordinate.get("path")
            coordinate_sha = coordinate.get("sha256")
        else:
            coordinate_path = coordinate
            coordinate_sha = raw.get("coordinates_sha256")
        if not all(isinstance(value, str) and value for value in
                   (tag, role, gate_path, gate_sha, coordinate_path, coordinate_sha)):
            raise EvaluationError(
                "baseline_assets[%d] lacks explicit tag/role/source gate/coordinate metadata" % index
            )
        if tag in result:
            raise EvaluationError("selection baseline_assets contains duplicate tag %s" % tag)
        result[str(tag)] = report.BaselineSpec(
            tag=str(tag),
            role=str(role),
            source_gate_path=str(_resolve_asset_path(gate_path, selection_file.parent)),
            source_gate_sha256=str(gate_sha),
            coordinates_path=str(_resolve_asset_path(coordinate_path, selection_file.parent)),
            coordinates_sha256=str(coordinate_sha),
        )
    return result


def _normalise_baseline_specs(
    selection_path: Path,
    value: Mapping[str, report.BaselineSpec | Mapping[str, Any]] | None,
    document: Mapping[str, Any],
) -> dict[str, report.BaselineSpec]:
    if value is None:
        return baseline_specs_from_selection(selection_path, document)
    result: dict[str, report.BaselineSpec] = {}
    for tag, raw in value.items():
        if isinstance(raw, report.BaselineSpec):
            spec = raw
        elif isinstance(raw, Mapping):
            fields = dict(raw)
            fields.setdefault("tag", tag)
            try:
                spec = report.BaselineSpec(**fields)
            except TypeError as exc:
                raise EvaluationError("malformed baseline spec for %s" % tag) from exc
        else:
            raise EvaluationError("baseline spec for %s must be BaselineSpec or object" % tag)
        result[str(tag)] = spec
    return result


def _normalise_input_spec(
    value: report.EvaluatorInputAuditSpec | Mapping[str, Any] | None,
) -> report.EvaluatorInputAuditSpec:
    if isinstance(value, report.EvaluatorInputAuditSpec):
        return value
    if not isinstance(value, Mapping):
        raise EvaluationError("evaluator_input_spec is required")
    try:
        return report.EvaluatorInputAuditSpec(**dict(value))
    except TypeError as exc:
        raise EvaluationError("malformed evaluator_input_spec") from exc


def _default_candidate_loader(grid: report.FullGrid) -> Callable[[str], Any]:
    def load(path: str) -> Any:
        return report.read_full_grid_coordinates(path, grid)
    return load


def _default_baseline_loader(_grid: report.FullGrid) -> Callable[[str, str], Any]:
    def load(_tag: str, path: str) -> Any:
        # source gate 已核验其摘要和轨迹清单。
        # 严格坐标解析器防止格式错误的 source 行进入指标函数。
        # source 行进入指标函数。
        return report._read_coordinate_rows(Path(path).resolve())
    return load


def _default_contacts_loader(snpfree_path: str) -> Callable[[], Any]:
    def load() -> Any:
        return genome.load_all(snpfree_path)
    return load


def alignment_verifier(contacts: Any, raw_path: str) -> dict[str, Any]:
    """按数值并按顺序将每条 raw 记录与 SNP-free contacts 比较。

    这里只检查染色体和坐标列。phase 列保持不变，直到 gate arm 后调用 ``refeval.load_labels_two``。对于 cis 记录，端点顺序按数值规范化，并保留 swap 计数作为审计证据。
    """
    if not isinstance(contacts, Mapping):
        raise report.SelectionValidationError("contacts must be a mapping for record alignment")
    required = ("ci", "p1", "cj", "p2", "names")
    if any(key not in contacts for key in required):
        raise report.SelectionValidationError("contacts lacks fields required for raw record alignment")
    ci = np.asarray(contacts["ci"], dtype=np.int64)
    p1 = np.asarray(contacts["p1"], dtype=np.int64)
    cj = np.asarray(contacts["cj"], dtype=np.int64)
    p2 = np.asarray(contacts["p2"], dtype=np.int64)
    names = tuple(str(value) for value in contacts["names"])
    n = len(ci)
    if any(array.ndim != 1 or len(array) != n for array in (p1, cj, p2)):
        raise report.SelectionValidationError("contacts coordinate arrays are not aligned")
    index = {name: i for i, name in enumerate(names)}
    if len(index) != len(names):
        raise report.SelectionValidationError("contacts chromosome names are not unique")
    path = Path(raw_path).resolve()
    opener = gzip.open if path.suffix == ".gz" else open
    seen = 0
    cis_swaps = 0
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    raise report.SelectionValidationError(
                        "raw alignment row %d has fewer than five coordinate fields" % line_no
                    )
                try:
                    left_name, right_name = fields[1], fields[3]
                    left = index[left_name]
                    right = index[right_name]
                    left_pos, right_pos = int(fields[2]), int(fields[4])
                except (KeyError, ValueError) as exc:
                    raise report.SelectionValidationError(
                        "raw alignment row %d has invalid chromosome or coordinate" % line_no
                    ) from exc
                if left == right and left_pos > right_pos:
                    left_pos, right_pos = right_pos, left_pos
                    left, right = right, left
                    cis_swaps += 1
                if seen >= n:
                    raise report.SelectionValidationError(
                        "raw pairs contain more records than SNP-free contacts at row %d" % line_no
                    )
                observed = (int(ci[seen]), int(p1[seen]), int(cj[seen]), int(p2[seen]))
                expected = (left, left_pos, right, right_pos)
                if observed != expected:
                    raise report.SelectionValidationError(
                        "raw/SNP-free alignment mismatch at record %d (raw line %d): expected %s observed %s"
                        % (seen, line_no, expected, observed)
                    )
                seen += 1
    except OSError as exc:
        raise report.SelectionValidationError("cannot read raw pairs for alignment: %s" % exc) from exc
    if seen != n:
        raise report.SelectionValidationError(
            "raw pairs contain %d records, expected full SNP-free M=%d" % (seen, n)
        )
    return {
        "aligned": True,
        "raw_records": int(seen),
        "contacts_records": int(n),
        "canonicalized_cis_endpoint_swaps": int(cis_swaps),
        "comparison": {
            "record_order": "raw file order equals SNP-free loader order",
            "chromosome_and_coordinates": "numeric comparison",
            "cis_endpoint_policy": "ascending numeric pos1<=pos2 with endpoint metadata swapped later by labels loader",
            "phase_fields_inspected": False,
        },
    }


_DEFAULT_ALIGNMENT_VERIFIER = alignment_verifier


def _default_labels_loader(gate: EvalGate) -> Callable[[Any, str], Any]:
    def load(contacts: Any, raw_path: str) -> Any:
        return refeval.load_labels_two(gate, contacts, pairs_path=raw_path)
    return load


def _default_reference_loader(gate: EvalGate) -> Callable[[str], Any]:
    def load(path: str) -> Any:
        return ref3dg.load_reference(gate, path=path)
    return load


def _labels_pair(labels: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(labels, Mapping):
        a1, a2 = labels.get("a1"), labels.get("a2")
    elif isinstance(labels, (tuple, list)) and len(labels) == 2:
        a1, a2 = labels
    else:
        raise EvaluationError("labels loader must return aligned (a1, a2) arrays")
    a1 = np.asarray(a1, dtype=np.int8)
    a2 = np.asarray(a2, dtype=np.int8)
    if a1.ndim != 1 or a2.ndim != 1 or len(a1) != len(a2):
        raise EvaluationError("labels loader returned non-aligned arrays")
    return a1, a2


def _metric_bins(length_bp: int) -> int:
    # full training grid 从零开始。OFF=3 Mb 是独立的 metric grid，
    # 只包含严格小于 chromosome header length 的 starts。
    return len(range(OFF, int(length_bp), BIN))


def _require_contacts(contacts: Any, grid: report.FullGrid) -> dict[str, Any]:
    if not isinstance(contacts, Mapping):
        raise EvaluationError("contacts loader must return a mapping")
    required = ("ci", "p1", "cj", "p2", "cis", "names")
    if any(key not in contacts for key in required):
        raise EvaluationError("contacts loader result lacks required fields: %s" % (required,))
    names = tuple(str(value) for value in contacts["names"])
    expected_names = tuple(chromosome.name for chromosome in grid.chromosomes)
    if names != expected_names:
        raise EvaluationError("contacts chromosome order disagrees with frozen coordinate grid")
    arrays = {key: np.asarray(contacts[key]) for key in ("ci", "p1", "cj", "p2", "cis")}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1 or next(iter(lengths), 0) == 0:
        raise EvaluationError("contacts arrays must be non-empty and aligned")
    if not np.array_equal(arrays["cis"].astype(bool), arrays["ci"] == arrays["cj"]):
        raise EvaluationError("contacts.cis disagrees with chromosome indices")
    result = dict(contacts)
    result.update(arrays)
    result["names"] = names
    return result


def _termination_flags(fit: Mapping[str, Any], audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """规范化 iteration/function 限制证据，不重写旧字段。"""
    audit = audit if isinstance(audit, Mapping) else {}
    solver = fit.get("solver", {}) if isinstance(fit.get("solver", {}), Mapping) else {}
    status = fit.get("status")
    iteration = audit.get("iteration_limit_reached")
    function = audit.get("function_limit_reached")
    if not isinstance(iteration, bool):
        nit, maxiter = solver.get("nit"), solver.get("maxiter")
        iteration = bool(status == "not_converged" and isinstance(nit, int)
                         and isinstance(maxiter, int) and nit >= maxiter)
    if not isinstance(function, bool):
        actual_nfev = solver.get("actual_nfev", solver.get("nfev"))
        maxfun = solver.get("maxfun")
        message = str(solver.get("message", ""))
        function = bool(
            (isinstance(actual_nfev, int) and isinstance(maxfun, int) and actual_nfev >= maxfun)
            or ("TOTAL NO. OF F" in message.upper() and "EVALUATIONS EXCEEDS LIMIT" in message.upper())
        )
    legacy = fit.get("budget_exhausted") if isinstance(fit.get("budget_exhausted"), bool) else None
    reached = bool(iteration or function or legacy)
    return {
        "iteration_limit_reached": bool(iteration),
        "function_limit_reached": bool(function),
        "budget_reached": reached,
        "legacy_budget_exhausted": legacy,
        "evidence_source": "termination_audit" if audit else "stage_fit_solver_fallback",
    }


def _stage_termination_audit(audit: Any, stage: Any) -> Mapping[str, Any] | None:
    """从可能的 audit JSON 结构中选择某个阶段的记录。"""
    if not isinstance(audit, Mapping):
        return None
    stages = audit.get("stages")
    if isinstance(stages, Mapping):
        value = stages.get(stage)
        return value if isinstance(value, Mapping) else None
    if isinstance(stages, list):
        for value in stages:
            if isinstance(value, Mapping) and value.get("stage") == stage:
                return value
    candidates = audit.get("candidates")
    if isinstance(candidates, Mapping):
        value = candidates.get(stage)
        if isinstance(value, Mapping):
            return value
    return audit


def _candidate_termination_audit(
    verified: report.VerifiedSelection, candidate_id: str, raw: Mapping[str, Any]
) -> dict[str, Any] | None:
    """在训练验证后加载可选的独立 termination audit。"""
    declared = raw.get("termination_audit")
    if isinstance(declared, Mapping) and not declared.get("path"):
        payload: Any = declared
        source_path = None
    else:
        path_value = raw.get("termination_audit_path")
        declared_sha = None
        if isinstance(declared, Mapping):
            path_value = declared.get("path", path_value)
            declared_sha = declared.get("sha256")
        if path_value is None:
            top = verified.document.get("termination_audit")
            if isinstance(top, Mapping):
                path_value = top.get("path")
                declared_sha = top.get("sha256")
            elif isinstance(top, str):
                path_value = top
        if path_value is None:
            fallback = Path(verified.selection_path).parent / "termination_audit.json"
            path_value = str(fallback) if fallback.is_file() else None
        if path_value is None:
            return None
        source_path = _resolve_asset_path(path_value, Path(verified.selection_path).parent)
        if not source_path.is_file():
            raise EvaluationError("declared termination audit is unavailable: %s" % source_path)
        if declared_sha is not None and sha256_file(source_path) != str(declared_sha):
            raise EvaluationError("termination audit SHA256 mismatch for %s" % source_path)
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError("cannot read termination audit: %s" % source_path) from exc
    if isinstance(payload, Mapping):
        candidates = payload.get("candidates")
        if isinstance(candidates, Mapping) and candidate_id in candidates:
            payload = candidates[candidate_id]
        elif isinstance(candidates, list):
            payload = next((item for item in candidates
                            if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id), payload)
    return {"source_path": None if source_path is None else str(source_path),
            "sha256": None if source_path is None else sha256_file(source_path),
            "payload": payload}


def _solver_layer_metadata(
    verified: report.VerifiedSelection, raw: Mapping[str, Any],
    termination_audit: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """读取冻结的阶段汇总，不携带庞大的优化器历史记录。"""
    result = []
    for layer in raw.get("layers", []):
        if not isinstance(layer, Mapping) or not layer.get("path"):
            continue
        stage_path = _resolve_asset_path(layer["path"], Path(verified.selection_path).parent)
        if not stage_path.is_file():
            continue
        try:
            payload = json.loads(stage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fit = payload.get("fit", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(fit, Mapping):
            continue
        solver = fit.get("solver", {})
        stage_name = layer.get("stage", payload.get("stage"))
        stage_audit = fit.get("termination_audit")
        if not isinstance(stage_audit, Mapping):
            stage_audit = payload.get("termination_audit") if isinstance(payload, Mapping) else None
        if not isinstance(stage_audit, Mapping):
            stage_audit = _stage_termination_audit(
                termination_audit.get("payload") if isinstance(termination_audit, Mapping) else None,
                stage_name,
            )
        flags = _termination_flags(fit, stage_audit)
        result.append({
            "stage": stage_name,
            "status": fit.get("status", layer.get("status")),
            "budget_exhausted": fit.get("budget_exhausted"),
            "iteration_limit_reached": flags["iteration_limit_reached"],
            "function_limit_reached": flags["function_limit_reached"],
            "budget_reached": flags["budget_reached"],
            "termination_evidence_source": flags["evidence_source"],
            "initial_total": fit.get("initial_total"),
            "final_total": fit.get("final_total"),
            "p": fit.get("p"),
            "solver": dict(solver) if isinstance(solver, Mapping) else solver,
            "final_components": fit.get("final_components"),
            "stage_record": str(stage_path),
        })
    return result


def _candidate_metadata(verified: report.VerifiedSelection) -> dict[str, dict[str, Any]]:
    document_candidates = {
        str(item.get("id")): item for item in verified.document.get("candidates", [])
        if isinstance(item, Mapping)
    }
    attempts = verified.document.get("attempts", [])
    metadata: dict[str, dict[str, Any]] = {}
    for candidate_id in verified.document.get("preregistered_candidate_ids", []):
        candidate = verified.candidates[candidate_id]
        raw = document_candidates.get(candidate_id, {})
        candidate_attempts = [dict(item) for item in attempts
                              if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id]
        budget_values = [item.get("budget_exhausted") for item in candidate_attempts
                         if isinstance(item.get("budget_exhausted"), bool)]
        solver_audit = _candidate_termination_audit(verified, candidate_id, raw)
        solver_layers = _solver_layer_metadata(verified, raw, solver_audit)
        iteration_flags = [layer.get("iteration_limit_reached") for layer in solver_layers]
        function_flags = [layer.get("function_limit_reached") for layer in solver_layers]
        reached_flags = [layer.get("budget_reached") for layer in solver_layers]
        legacy_budget = bool(any(budget_values)) if budget_values else None
        item: dict[str, Any] = {
            "candidate_id": candidate_id,
            "terminal_status": candidate.terminal_status,
            "coordinates": candidate.inventory.as_dict() if candidate.inventory is not None else None,
            "count_nll_per_record": candidate.count_nll_per_record,
            "failure_reason": candidate.failure_reason,
            "optimization": raw.get("optimization", {}),
            "attempts": candidate_attempts,
            # 原样保留 runner 的 legacy field；下面使用显式派生标志解释面向人的 budget。
            # 下面的 flags 用于面向人的 budget 解释。
            "budget_exhausted": legacy_budget,
            "iteration_limit_reached": bool(any(iteration_flags)) if iteration_flags else None,
            "function_limit_reached": bool(any(function_flags)) if function_flags else None,
            "budget_reached": bool(any(reached_flags)) if reached_flags else legacy_budget,
            "count_model": raw.get("count_model"),
            "selection_rescore": raw.get("selection_rescore"),
            "termination_audit": solver_audit,
            "solver_layers": solver_layers,
        }
        if isinstance(raw.get("count_model"), Mapping):
            item["p_is_nuisance_coefficient"] = raw["count_model"].get("p")
            item["prior_components"] = {
                key: raw["count_model"].get(key) for key in report.PRIOR_REPORT_FIELDS
            }
        metadata[candidate_id] = item
    return metadata


def _finite_values(values: Sequence[Any]) -> list[float]:
    result = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            result.append(numeric)
    return result


def _mean(values: Sequence[Any]) -> float | None:
    finite = _finite_values(values)
    return float(np.mean(finite)) if finite else None


def _comparison(
    chromosome_names: Sequence[str],
    rows: Mapping[str, Mapping[str, Any]],
    metric: str,
    value_path: tuple[str, ...],
    denominator_path: tuple[str, ...],
    candidate_id: str,
    control_id: str = "random",
) -> dict[str, Any]:
    deltas: dict[str, float | None] = {}
    denominators: dict[str, Any] = {}
    for chromosome in chromosome_names:
        row = rows.get(chromosome)
        value: Any = None
        denominator: Any = None
        if row is not None:
            current: Any = row
            for key in value_path:
                current = current.get(key) if isinstance(current, Mapping) else None
            value = current
            current = row
            for key in denominator_path:
                current = current.get(key) if isinstance(current, Mapping) else None
            denominator = current
        deltas[chromosome] = None if value is None else float(value)
        denominators[chromosome] = denominator
    # 使用稳定的、按 metric 区分的 seeds，使报告跨进程可复现。
    stable_seed = {"R1": 101, "R2": 102, "R3": 103}.get(metric, 109)
    bootstrap = report.paired_chromosome_bootstrap(deltas, seed=stable_seed)
    per_chromosome = [
        {"chromosome": chromosome, "delta_candidate_minus_random": value,
         "paired_denominator": denominators[chromosome]}
        for chromosome, value in deltas.items()
    ]
    return {
        "candidate_id": candidate_id,
        "control_id": control_id,
        "metric": metric,
        "direction": "delta_candidate_minus_random",
        "per_chromosome": per_chromosome,
        "wins": bootstrap["wins"],
        "ties": bootstrap["ties"],
        "losses": bootstrap["losses"],
        "mean_delta": bootstrap["mean"],
        "ci95": bootstrap["ci95"],
        "paired_chromosome_bootstrap": bootstrap,
        "note": "20 chromosomes are linked measurements in one cell; bootstrap is technical/structural, not biological replication",
    }


def _candidate_summary(
    candidate_id: str,
    metadata: Mapping[str, Any],
    chromosome_names: Sequence[str],
    per_chromosome: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = {
        chromosome: per_chromosome[chromosome]["candidates"].get(candidate_id)
        for chromosome in chromosome_names
        if candidate_id in per_chromosome[chromosome].get("candidates", {})
    }
    r1 = [row["R1"].get("selected", {}).get("accuracy")
          if isinstance(row["R1"].get("selected"), Mapping) else None
          for row in rows.values()]
    r1_ref = [row["R1"].get("reference_ceiling", {}).get("accuracy")
              if isinstance(row["R1"].get("reference_ceiling"), Mapping) else None
              for row in rows.values()]
    r1_oracle = [row["R1"].get("oracle_fit_ceiling") for row in rows.values()]
    r2 = [row["R2"].get("by_candidate", {}).get("selected", {}).get("contrast")
          for row in rows.values()]
    r3 = [row["R3"].get("by_candidate", {}).get("selected", {}).get("frac_consistent")
          for row in rows.values()]
    walls = [row["R3"].get("by_candidate", {}).get("selected", {}).get("n_walls")
             for row in rows.values()]
    r3_status_counts: Counter[str] = Counter()
    r3_fragment_totals = Counter()
    for row in rows.values():
        selected_r3 = row["R3"].get("by_candidate", {}).get("selected", {})
        r3_status_counts.update(_r3_status_counts(selected_r3))
        for key in ("n_fragments_total", "n_fragments_applicable", "n_fragments_insufficient", "n_fragments_tied"):
            r3_fragment_totals[key] += int(selected_r3.get(key) or 0)
    summary = {
        "candidate_id": candidate_id,
        "terminal_status": metadata["terminal_status"],
        "count_nll_per_record": metadata["count_nll_per_record"],
        "failure_reason": metadata["failure_reason"],
        "selected_by_training_count_nll": bool(candidate_id == metadata.get("selected_id_frozen")),
        "R1_macro_mean": _mean(r1),
        "R1_reference_ceiling_macro_mean": _mean(r1_ref),
        "R1_oracle_fit_ceiling_macro_mean": _mean(r1_oracle),
        "R1_chromosomes": len(_finite_values(r1)),
        "R1_denominator_total": int(sum(
            int(row["R1"].get("paired_denominator", {}).get("n_common", 0))
            for row in rows.values()
        )),
        "R2_contrast_macro_mean": _mean(r2),
        "R2_chromosomes": len(_finite_values(r2)),
        "R3_frac_consistent_macro_mean": _mean(r3),
        "R3_n_walls_macro_mean": _mean(walls),
        "R3_chromosomes": len(_finite_values(r3)),
        "R3_status_counts": dict(sorted(r3_status_counts.items())),
        "R3_fragments_total": r3_fragment_totals["n_fragments_total"],
        "R3_fragments_applicable": r3_fragment_totals["n_fragments_applicable"],
        "R3_fragments_insufficient": r3_fragment_totals["n_fragments_insufficient"],
        "R3_fragments_tied": r3_fragment_totals["n_fragments_tied"],
    }
    if metadata["terminal_status"] != "failed":
        summary["comparison_vs_random"] = {
            "R1": _comparison(
                chromosome_names, rows, "R1",
                ("R1", "delta_candidate_minus_random"),
                ("R1", "paired_denominator", "n_common"), candidate_id,
            ),
            "R2": _comparison(
                chromosome_names, rows, "R2",
                ("R2", "selected_vs_random", "delta_candidate_minus_random"),
                ("R2", "selected_vs_random", "n_common_pairs"), candidate_id,
            ),
            "R3": _comparison(
                chromosome_names, rows, "R3",
                ("R3", "selected_vs_random", "delta_candidate_minus_random"),
                ("R3", "selected_vs_random", "n_common_fragments"), candidate_id,
            ),
        }
    else:
        summary["comparison_vs_random"] = None
    return summary


def _fixed_baseline_comparison(
    chromosome_names: Sequence[str], per_chromosome: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """明确保留既有的 oracle-vs-random 上限比较。"""
    rows: dict[str, Mapping[str, Any]] = {}
    for chromosome in chromosome_names:
        candidates = per_chromosome[chromosome].get("candidates", {})
        if not candidates:
            continue
        first = next(iter(candidates.values()))
        rows[chromosome] = {
            "R1": {"delta_candidate_minus_random": first["R1"].get("delta_oracle_minus_random"),
                   "paired_denominator": first["R1"].get("paired_denominator", {})},
            "R2": {"selected_vs_random": {
                "delta_candidate_minus_random": first["R2"].get("oracle_vs_random", {}).get("delta_left_minus_right"),
                "n_common_pairs": first["R2"].get("oracle_vs_random", {}).get("n_common_pairs", 0),
            }},
            "R3": {"selected_vs_random": {
                "delta_candidate_minus_random": first["R3"].get("oracle_vs_random", {}).get("delta_left_minus_right"),
                "n_common_fragments": first["R3"].get("oracle_vs_random", {}).get("n_common_fragments", 0),
            }},
        }
    return {
        "candidate_id": "oracle",
        "control_id": "random",
        "direction": "delta_candidate_minus_random",
        "R1": _comparison(chromosome_names, rows, "R1",
                          ("R1", "delta_candidate_minus_random"),
                          ("R1", "paired_denominator", "n_common"), "oracle"),
        "R2": _comparison(chromosome_names, rows, "R2",
                          ("R2", "selected_vs_random", "delta_candidate_minus_random"),
                          ("R2", "selected_vs_random", "n_common_pairs"), "oracle"),
        "R3": _comparison(chromosome_names, rows, "R3",
                          ("R3", "selected_vs_random", "delta_candidate_minus_random"),
                          ("R3", "selected_vs_random", "n_common_fragments"), "oracle"),
    }


def _augment_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    """在每个读出中暴露一个稳定的 candidate-minus-random 符号。"""
    r1 = row["R1"]
    if "delta_selected_minus_random" in r1:
        r1["delta_candidate_minus_random"] = r1.get("delta_selected_minus_random")
    for key in ("R2", "R3"):
        paired = row[key].get("selected_vs_random")
        if isinstance(paired, Mapping):
            paired["delta_candidate_minus_random"] = paired.get("delta_selected_minus_random")
    return row


def _r3_status_counts(r3: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in r3.get("detail", []):
        status = str(item.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _write_per_chromosome_tsv(
    path: Path,
    verified: report.VerifiedSelection,
    per_chromosome: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    columns = (
        "chromosome", "candidate_id", "terminal_status", "failure_reason",
        "selected_by_training_count_nll", "count_nll_per_record",
        "R1_accuracy", "R1_reference_ceiling", "R1_oracle_fit_ceiling",
        "R1_delta_candidate_minus_random", "R1_n_denominator",
        "R1_denominator_json", "R2_contrast", "R2_delta_candidate_minus_random",
        "R2_n_common_pairs", "R3_frac_consistent", "R3_longest_run", "R3_n_walls",
        "R3_delta_candidate_minus_random", "R3_n_common_fragments", "R3_status_counts_json",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_candidates = {
        str(item.get("id")): item for item in verified.document.get("candidates", [])
        if isinstance(item, Mapping)
    }
    for candidate_id in verified.document.get("preregistered_candidate_ids", []):
        candidate = verified.candidates[candidate_id]
        raw = raw_candidates.get(candidate_id, {})
        if candidate.terminal_status == "failed":
            rows.append({
                "chromosome": "ALL", "candidate_id": candidate_id,
                "terminal_status": "failed", "failure_reason": candidate.failure_reason,
                "selected_by_training_count_nll": candidate_id == verified.selected_id,
                "count_nll_per_record": None,
                **{column: None for column in columns[6:]},
            })
            continue
        for chromosome in per_chromosome:
            row = per_chromosome[chromosome]["candidates"][candidate_id]
            reference_ceiling = row["R1"].get("reference_ceiling")
            paired_r1 = row["R1"].get("paired_denominator", {})
            paired_r2 = row["R2"].get("selected_vs_random", {})
            paired_r3 = row["R3"].get("selected_vs_random", {})
            rows.append({
                "chromosome": chromosome,
                "candidate_id": candidate_id,
                "terminal_status": candidate.terminal_status,
                "failure_reason": None,
                "selected_by_training_count_nll": candidate_id == verified.selected_id,
                "count_nll_per_record": candidate.count_nll_per_record,
                "R1_accuracy": row["R1"].get("selected", {}).get("accuracy")
                if isinstance(row["R1"].get("selected"), Mapping) else None,
                "R1_reference_ceiling": reference_ceiling.get("accuracy")
                if isinstance(reference_ceiling, Mapping) else None,
                "R1_oracle_fit_ceiling": row["R1"].get("oracle_fit_ceiling"),
                "R1_delta_candidate_minus_random": row["R1"].get("delta_candidate_minus_random"),
                "R1_n_denominator": paired_r1.get("n_common"),
                "R1_denominator_json": _json_cell(paired_r1),
                "R2_contrast": row["R2"].get("by_candidate", {}).get("selected", {}).get("contrast"),
                "R2_delta_candidate_minus_random": paired_r2.get("delta_candidate_minus_random"),
                "R2_n_common_pairs": paired_r2.get("n_common_pairs"),
                "R3_frac_consistent": row["R3"].get("by_candidate", {}).get("selected", {}).get("frac_consistent"),
                "R3_longest_run": row["R3"].get("by_candidate", {}).get("selected", {}).get("longest_run"),
                "R3_n_walls": row["R3"].get("by_candidate", {}).get("selected", {}).get("n_walls"),
                "R3_delta_candidate_minus_random": paired_r3.get("delta_candidate_minus_random"),
                "R3_n_common_fragments": paired_r3.get("n_common_fragments"),
                "R3_status_counts_json": _json_cell(_r3_status_counts(
                    row["R3"].get("by_candidate", {}).get("selected", {})
                )),
            })
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join("" if row.get(column) is None else str(row.get(column))
                                   for column in columns) + "\n")
    return {"path": str(path), "rows": len(rows), "columns": list(columns)}


def _copy_coordinate_payload(root: Path, source: Path, expected_sha: str, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    actual = sha256_file(target)
    if actual != expected_sha:
        raise EvaluationError("coordinate delivery copy changed digest for %s" % source)
    return {"path": str(target), "sha256": actual, "source_path": str(source)}


def _evaluation_code_snapshot(root: Path) -> dict[str, Any]:
    rows = []
    for source_text in EVALUATION_CODE_SOURCES:
        source = Path(ROOT) / source_text
        if not source.is_file():
            raise EvaluationError("evaluation code source is unavailable: %s" % source)
        destination = root / "provenance" / "evaluation-code" / source_text
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        rows.append({
            "source_path": source_text,
            "snapshot_path": str(destination.relative_to(root)),
            "sha256": sha256_file(destination),
        })
    return {"schema_version": EVALUATION_CODE_SNAPSHOT_SCHEMA_VERSION, "files": rows}


def _trace_from_selection(verified: report.VerifiedSelection, candidate_id: str) -> list[dict[str, Any]]:
    raw = next((item for item in verified.document.get("candidates", [])
                if isinstance(item, Mapping) and item.get("id") == candidate_id), None)
    if not isinstance(raw, Mapping):
        return []
    trace: list[dict[str, Any]] = []
    for stage_index, layer in enumerate(raw.get("layers", [])):
        if not isinstance(layer, Mapping) or not layer.get("path"):
            continue
        path = Path(verified.selection_path).parent / str(layer["path"])
        if not path.is_file():
            continue
        try:
            stage = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fit = stage.get("fit", {}) if isinstance(stage, Mapping) else {}
        for item in fit.get("history", []) if isinstance(fit, Mapping) else []:
            if not isinstance(item, Mapping):
                continue
            entry = {"iteration": float(stage_index * 1_000_000 + float(item.get("iteration", 0)))}
            if "fun" in item:
                entry["total"] = item["fun"]
            if "p" in item:
                entry["p"] = item["p"]
            components = item.get("components")
            if isinstance(components, Mapping):
                for key in ("count_nll_normalized", "bond", "repulsion", "bend", "p_prior"):
                    if key in components:
                        entry[key] = components[key]
            trace.append(entry)
    return trace


def _gallery_tracks(grid: report.FullGrid, kind: str) -> list[dict[str, Any]]:
    if kind == "reference":
        return [
            {"track": "%s(%s)" % (chromosome.name, suffix),
             "chromosome": chromosome.name, "copy": suffix}
            for chromosome in grid.chromosomes for suffix in ("mat", "pat")
        ]
    if kind == "consensus":
        return [
            {"track": report.track_map_for_grid(grid)[2 * index]["track"],
             "chromosome": chromosome.name, "copy": "single"}
            for index, chromosome in enumerate(grid.chromosomes)
        ]
    return [
        {"track": entry["track"], "chromosome": entry["chromosome"], "copy": entry["copy"]}
        for entry in report.track_map_for_grid(grid)
    ]


def _write_distance_map_preview(
    path: Path,
    selected: Mapping[str, Mapping[int, np.ndarray]],
    reference: Mapping[str, Mapping[int, np.ndarray]],
    grid: report.FullGrid,
    chromosome_index: int,
    n_bins: int,
) -> dict[str, Any]:
    if n_bins < 2:
        raise EvaluationError("cannot render a distance map without at least two metric bins")
    preview_bins = min(int(n_bins), 256)
    positions = [OFF + index * BIN for index in range(preview_bins)]
    chromosome = grid.chromosomes[chromosome_index].name
    keys = (
        ("selected copy a", report.track_map_for_grid(grid)[2 * chromosome_index]["track"], selected),
        ("selected copy b", report.track_map_for_grid(grid)[2 * chromosome_index + 1]["track"], selected),
        ("reference mat", "%s(mat)" % chromosome, reference),
        ("reference pat", "%s(pat)" % chromosome, reference),
    )
    from . import figs

    matrices = []
    for label, track, structures in keys:
        matrix = np.full((preview_bins, preview_bins), np.nan, dtype=float)
        points = [structures.get(track, {}).get(position) for position in positions]
        for i, left in enumerate(points):
            if left is None:
                continue
            left = np.asarray(left, dtype=float)
            for j, right in enumerate(points):
                if right is None:
                    continue
                right = np.asarray(right, dtype=float)
                if np.isfinite(left).all() and np.isfinite(right).all():
                    matrix[i, j] = float(np.linalg.norm(left - right))
        matrices.append((label, matrix))
    finite_parts = [matrix[np.isfinite(matrix)] for _label, matrix in matrices if np.isfinite(matrix).any()]
    finite = np.concatenate(finite_parts) if finite_parts else np.empty(0, dtype=float)
    vmax = float(np.max(finite)) if len(finite) else 1.0
    if vmax <= 0 or not math.isfinite(vmax):
        vmax = 1.0
    panel = [(label, matrix, 0.0, vmax) for label, matrix in matrices]
    path.parent.mkdir(parents=True, exist_ok=True)
    figs.distance_maps(path, panel, ncols=2,
                       title="raw distance maps: %s; coolwarm_r" % chromosome)
    return {
        "figure": str(path), "chromosome": chromosome,
        "chromosome_index": chromosome_index, "preview_bins": preview_bins,
        "bin_size_bp": BIN, "offset_bp": OFF, "colormap": "coolwarm_r",
        "coordinate_transform": "none",
    }


def _render_readme(
    verified: report.VerifiedSelection,
    metrics: Mapping[str, Any],
    delivery: Mapping[str, Any],
    *,
    l2_conclusion: str,
) -> str:
    metadata = metrics["candidate_metadata"]
    selected_id = verified.selected_id
    lines = [
        "# P9016 Single-Cell Final Evaluation and Delivery",
        "",
        "**Status:** `evaluation_complete`. This directory was generated as a new evaluation subdirectory after training; it does not overwrite formal training files.",
        "",
        "## Training Selection and Frozen Boundary",
        "",
        "- There is 1 biological sample; the frozen scope comprises all 1,703,888 contacts = 1,135,454 intrachromosomal contacts + 568,434 interchromosomal contacts.",
        "- SNP-free training input SHA256: `%s`; the training grid is the complete 1 Mb grid starting at 0, with %d loci / %d physical beads; all 40 tracks lie within the same dimensionless R=1 nuclear sphere. R=1 has no micrometer-scale calibration." % (
            verified.document["cohort"]["snpfree_sha256"], verified.grid.n_loci, verified.grid.n_physical_beads),
        "- Candidate selection strictly follows training-time `%s` / `%s` and the preregistered order; this evaluation did not reselect by reference. Frozen selected_id: `%s`." % (
            verified.document["objective_contract"]["criterion"],
            verified.document["objective_contract"]["direction"], selected_id),
        "- Frozen protocol/config/code/data hashes, every member of the training-code snapshot and protocol manifest, complete attempt records, and every candidate-coordinate SHA were revalidated. The code snapshot does not depend on the current workspace source.",
        "",
        "### Candidate Terminal States",
        "",
        "- `budget_exhausted` is a legacy runner field; the final budget interpretation uses the independent `termination_audit` when available, otherwise it is derived from the solver's `nit/maxiter` and `nfev/maxfun`. A `not_converged` result that reaches the iteration limit must be stated explicitly as “iteration budget reached, not converged” and must not be interpreted as budget not reached.",
    ]
    for candidate_id in verified.document.get("preregistered_candidate_ids", []):
        item = metadata[candidate_id]
        if item["terminal_status"] == "failed":
            lines.append("- `%s`: terminal=`failed`; explicit failure reason: %s; no coordinates and no selectable NLL." %
                         (candidate_id, item["failure_reason"]))
            continue
        model = item.get("count_model") or {}
        prior = item.get("prior_components") or {}
        attempts = item.get("attempts", [])
        budget = item.get("budget_reached")
        lines.append(
            "- `%s`: terminal=`%s`; count_nll_per_record=`%s`; p=`%s` (nuisance coefficient, not the same-copy percentage); prior(bond,repulsion,bend,p_prior)=(%s,%s,%s,%s); budget_reached=`%s` (iteration_limit_reached=`%s`, function_limit_reached=`%s`, legacy budget_exhausted=`%s`); training_selected=`%s`." % (
                candidate_id, item["terminal_status"], item["count_nll_per_record"], model.get("p"),
                prior.get("bond"), prior.get("repulsion"), prior.get("bend"), prior.get("p_prior"),
                budget, item.get("iteration_limit_reached"), item.get("function_limit_reached"),
                item.get("budget_exhausted"), candidate_id == selected_id),
        )
        if attempts:
            lines.append("  Attempt records:" + "; ".join(
                "%s=%s%s" % (attempt.get("stage"), attempt.get("status"),
                              ",budget" if attempt.get("budget_exhausted") else "")
                for attempt in attempts
            ))
        if item.get("solver_layers"):
            lines.append("  Solver-layer terminal states:" + "; ".join(
                "%s=%s,nit=%s,nfev=%s,total=%s->%s,budget_reached=%s(iteration=%s,function=%s,legacy=%s)" % (
                    layer.get("stage"), layer.get("status"),
                    (layer.get("solver") or {}).get("nit") if isinstance(layer.get("solver"), Mapping) else None,
                    (layer.get("solver") or {}).get("nfev") if isinstance(layer.get("solver"), Mapping) else None,
                    layer.get("initial_total"), layer.get("final_total"), layer.get("budget_reached"),
                    layer.get("iteration_limit_reached"), layer.get("function_limit_reached"),
                    layer.get("budget_exhausted"))
                for layer in item["solver_layers"]
            ))
    lines.extend([
        "",
        "## Input Audit and Gating",
        "",
        "- Before reading `phase`/`reference`, verify the raw `pairs` SHA, the `reference` SHA (the published 017 digest `%s` required for strict P9016), and the completed-hash 017 provenance declaration; SNP-free contacts are loaded from the frozen data snapshot." % P9016_REFERENCE_3DG_SHA256,
        "- Alignment compares chromosome and coordinate values record by record; cis endpoints are normalized numerically, and endpoint swaps are included in the alignment evidence; the `labels` loader returns full-length M for both a1/a2 and swaps the `phase` endpoints in sync; only then is `reference` read.",
        "- This directory stores the full `record_alignment` summary, input versions, gating entries, and evaluator code snapshot; none of these are written back to the training freeze list.",
        "",
        "## Metric Definitions",
        "",
        "- R1 evaluates only records that are on the same chromosome, phased to the same copy, non-diagonal, and on the OFF=3 Mb metric grid. For each candidate, fixed random baseline, fully phased S0 oracle-fit, and reference ceiling, metrics are computed on the common record mask where candidate/random/oracle/reference are all finite; ties, missing values, out-of-grid records, and denominators are listed separately. This is neither held-out generalization nor a mathematical absolute upper bound. Interchromosomal contacts are not used for haplotype accuracy.",
        "- R2 uses the six-track, jointly finite, non-diagonal paired mask for `candidate`/`random`/`reference`, with copy pairing determined by each chromosome's geometric gauge freedom.",
        "- R3 uses jointly parseable 20 Mb fragments; `labelled`, `local_geometry_tie`, `insufficient_common_finite_pairs`, missing states, and broken segments are all retained. Majority is a directional baseline; a lower wall count is not named L2 by itself.",
        "- Every preregistered candidate that did not fail is compared with the fixed random baseline, with direction standardized as `delta_candidate_minus_random`. Win counts and paired-chromosome bootstrap CI describe technical/structural variation within one cell, not biological-replicate uncertainty; the old macro-average subtraction with different denominators is not used.",
        "- The consensus single-track R1/R2/R3 contrast is n/a, but its single-structure baseline is retained. All negative/unavailable results are preserved as observed; candidate structures are not omitted because L2 has not been established.",
        "",
        "## L2 Boundary",
        "",
        "- L2 conclusion: %s" % l2_conclusion,
        "- R3 is the primary L2 readout; a lower wall count or a local majority alone is not evidence that copy identity has been recovered along the whole chromosome.",
        "",
        "## Delivery Files",
        "",
        "- Download the raw R1 unit-ball coordinates for `selected`: `%s`; track map: `%s`." %
        (delivery["selected_coordinates"]["path"], delivery["track_map"]["path"]),
        "- Static 3D for `selected`: `%s`; self-contained canvas HTML: `%s`. The HTML rotation/zoom/chr/copy toggles, hover behavior, and raw export only change the display projection; they do not change raw coordinates." %
        (delivery["selected_figure"]["figure"], delivery["selected_html"]["html"]),
        "- Copies of all candidate coordinates, `metrics.json`, the per-chromosome TSV, summaries, R3 stripes, track coverage, candidate gallery, and coolwarm_r distance maps are in this evaluation subdirectory. Each gallery panel is independently centered and scaled to unit RMS for structural diagnostics only; it does not claim a shared coordinate system or an absolute micrometer scale.",
        "- This directory does not package the real `phase`-bearing pairs or reference contents; it stores only their version hashes, alignment evidence, and the evaluator code snapshot.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(_safe({
            "selected_id_frozen": metrics["selection"]["selected_id_frozen"],
            "candidate_summary": metrics["candidate_summary"],
            "candidate_vs_random": metrics["candidate_vs_random"],
            "fixed_oracle_vs_random": metrics["fixed_baseline_comparison"],
        }), indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
    ])
    return "\n".join(lines)


def _create_output_dir(selection_path: Path, output_dir: str | os.PathLike[str] | None) -> Path:
    if output_dir is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        root = selection_path.parent / ("evaluation-%s" % stamp)
        suffix = 0
        while os.path.lexists(root):
            suffix += 1
            root = selection_path.parent / ("evaluation-%s-%02d" % (stamp, suffix))
    else:
        root = Path(output_dir).resolve()
    if root == selection_path.parent:
        raise EvaluationError("evaluation output must be a new child directory, not the training run root")
    if os.path.lexists(root):
        raise FileExistsError("refusing to reuse evaluation output directory: %s" % root)
    root.mkdir(parents=True, exist_ok=False)
    for relative in ("coordinates/candidates", "metrics", "plots", "provenance"):
        (root / relative).mkdir(parents=True, exist_ok=False)
    return root


def _relative(root: Path, path: Path | str) -> str:
    return str(Path(path).resolve().relative_to(root.resolve()))


def run_final_evaluation(
    selection_path: str | os.PathLike[str],
    *,
    evaluator_input_spec: report.EvaluatorInputAuditSpec | Mapping[str, Any] | None = None,
    baseline_specs: Mapping[str, report.BaselineSpec | Mapping[str, Any]] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    candidate_loader: Callable[[str], Any] | None = None,
    baseline_loader: Callable[[str, str], Any] | None = None,
    contacts_loader: Callable[[], Any] | None = None,
    alignment_verifier: Callable[[Any, str], Mapping[str, Any]] | None = None,
    alignment_verifier_callback: Callable[[Any, str], Mapping[str, Any]] | None = None,
    labels_loader: Callable[[Any, str], Any] | None = None,
    reference_loader: Callable[[str], Any] | None = None,
    strict_p9016: bool = True,
    l2_conclusion: str = "L2 has not been established: the final conclusion must integrate whole-chromosome R3 consistency, the common denominator, and paired CI; this entry point does not automatically promote a low wall count or local majority to L2.",
    synthetic_fixture: bool = False,
) -> dict[str, Any]:
    """运行完整的受守门评估器，并写出新的交付目录。

    注入回调时所需的签名为：

    ``candidate_loader(path)`` -> 坐标 mapping；
    ``baseline_loader(tag, path)`` -> 坐标 mapping；
    ``contacts_loader()`` -> 与 ``genome.load_all`` 兼容的 mapping；
    ``alignment_verifier(contacts, raw_path)`` -> ``{"aligned": True, ...}``；
    ``labels_loader(contacts, raw_path)`` -> ``(a1, a2)`` arrays；
    ``reference_loader(path)`` -> reference 坐标 mapping。

    不注入回调时，函数使用生产 SNP-free、``refeval.load_labels_two`` 和 ``ref3dg.load_reference`` 加载器。它不会启动训练，也不会根据 reference metrics 选择候选。
    """
    selection_file = Path(selection_path).resolve()
    # 纯 preflight 为构建默认闭包 和 baseline specs 提供已核验的网格/数据快照。
    # 它不能读取 labels/reference。
    preverified = report.verify_training_complete(selection_file, strict_p9016=strict_p9016)
    specs = _normalise_baseline_specs(selection_file, baseline_specs, preverified.document)
    input_spec = _normalise_input_spec(evaluator_input_spec)
    root = _create_output_dir(selection_file, output_dir)
    try:
        # 在任何评估器加载器 前审计已发布的 P9016 reference 摘要。
        # 这里只哈希字节，不解析 phase/reference payload。
        preflight_input_audit = report.verify_evaluator_input_versions(
            input_spec, strict_p9016=strict_p9016
        )
        if (strict_p9016 and
                preflight_input_audit["reference_3dg"]["actual_sha256"] != P9016_REFERENCE_3DG_SHA256):
            raise report.SelectionValidationError(
                "P9016 reference 3DG SHA256 differs from the frozen 017 digest"
            )
    except Exception:
        _write_json(root / "failure.json", {
            "status": "evaluation_input_audit_failed",
            "selection_path": str(selection_file),
            "registered_coordinate_paths": [],
        })
        raise
    gate = EvalGate(str(root / "gate.json"))
    registered_paths: set[str] = set()
    alignment_evidence: list[Mapping[str, Any]] = []
    candidate_load = candidate_loader or _default_candidate_loader(preverified.grid)
    baseline_load = baseline_loader or _default_baseline_loader(preverified.grid)
    snpfree_path = preverified.frozen_provenance["data"]["path"]
    contacts_load = contacts_loader or _default_contacts_loader(snpfree_path)
    if alignment_verifier is not None and alignment_verifier_callback is not None:
        raise EvaluationError("provide only one of alignment_verifier and alignment_verifier_callback")
    alignment_load = alignment_verifier_callback or alignment_verifier or _DEFAULT_ALIGNMENT_VERIFIER
    labels_load = labels_loader or _default_labels_loader(gate)
    reference_load = reference_loader or _default_reference_loader(gate)

    def register_coordinate(stage_tag: str, path_text: str) -> None:
        path = Path(path_text).resolve()
        key = str(path)
        if key in registered_paths:
            raise EvaluationError("duplicate coordinate registration: %s" % path)
        digest = gate.register("final-evaluation", stage_tag, str(path))
        registered_paths.add(key)
        if not digest:
            raise EvaluationError("coordinate gate did not return a digest for %s" % path)

    def guarded_candidate_load(path_text: str) -> Any:
        value = candidate_load(path_text)
        register_coordinate("candidate:%s" % Path(path_text).name, path_text)
        return value

    def guarded_baseline_load(tag: str, path_text: str) -> Any:
        value = baseline_load(tag, path_text)
        register_coordinate("baseline:%s" % tag, path_text)
        return value

    def guarded_alignment(contacts: Any, raw_path: str) -> Mapping[str, Any]:
        value = alignment_load(contacts, raw_path)
        if not isinstance(value, Mapping):
            raise EvaluationError("alignment verifier must return an object")
        alignment_evidence.append(dict(value))
        return value

    def guarded_labels(contacts: Any, raw_path: str) -> Any:
        # baseline paths 在 specs 中显式给出；在此之前没有打开 baseline payload，
        # 但每个 source file 都已由紧接其 loader 调用前的守门完成哈希核验。
        # （守门就在其 loader 调用前执行）。
        expected = {
            str(Path(candidate.coordinates_path).resolve())
            for candidate in preverified.candidates.values() if candidate.coordinates_path is not None
        }
        expected.update(str(Path(spec.coordinates_path).resolve()) for spec in specs.values())
        if registered_paths != expected:
            raise report.SelectionValidationError(
                "coordinate gate cannot arm: registered=%s expected=%s" %
                (sorted(registered_paths), sorted(expected))
            )
        gate.arm(refeval.STAGE)
        gate.require(refeval.STAGE)
        return labels_load(contacts, raw_path)

    def guarded_reference(path_text: str) -> Any:
        gate.require(refeval.STAGE)
        return reference_load(path_text)

    try:
        loaded = report.guarded_final_evaluation_inputs(
            selection_file,
            baseline_specs=specs,
            evaluator_input_spec=input_spec,
            candidate_loader=guarded_candidate_load,
            baseline_loader=guarded_baseline_load,
            contacts_loader=contacts_load,
            alignment_verifier=guarded_alignment,
            labels_loader=guarded_labels,
            reference_loader=guarded_reference,
            strict_p9016=strict_p9016,
        )
    except Exception:
        # 即使守门中止，输出路径仍是新建且可检查的 failure directory；但不能在此时声称 evaluator artifact 完成。
        # failure directory；但不能在此时声称 evaluator artifact 完成。
        _write_json(root / "failure.json", {
            "status": "evaluation_guard_failed",
            "selection_path": str(selection_file),
            "registered_coordinate_paths": sorted(registered_paths),
        })
        raise

    verified = loaded["verified_selection"]
    contacts = _require_contacts(loaded["contacts"], verified.grid)
    a1, a2 = _labels_pair(loaded["labels"])
    if len(a1) != len(contacts["ci"]):
        raise EvaluationError("labels and contacts do not have the same M")
    lab = refeval.single_label(a1, a2)
    reference = loaded["reference"]
    candidate_coordinates = {
        str(candidate_id): structs for candidate_id, structs in loaded["candidate_coordinates"].items()
    }
    baseline_coordinates = loaded["baseline_coordinates"]
    if set(candidate_coordinates) != {
        candidate_id for candidate_id, candidate in verified.candidates.items()
        if candidate.coordinates_path is not None
    }:
        raise EvaluationError("guard returned an incomplete nonfailed candidate mapping")
    if set(baseline_coordinates) != {"consensus", "random", "oracle"}:
        raise EvaluationError("guard returned an incomplete baseline mapping")

    chromosome_names = [chromosome.name for chromosome in verified.grid.chromosomes]
    n_bins_by_chromosome = {
        chromosome.name: _metric_bins(chromosome.length_bp)
        for chromosome in verified.grid.chromosomes
    }
    refeval.clear_cache()
    per_chromosome: dict[str, dict[str, Any]] = {}
    for chromosome_index, chromosome in enumerate(verified.grid.chromosomes):
        name = chromosome.name
        indices = np.where(contacts["cis"] & (contacts["ci"] == chromosome_index)
                           & (contacts["cj"] == chromosome_index))[0]
        b1 = ((contacts["p1"][indices] - OFF) // BIN).astype(np.int64)
        b2 = ((contacts["p2"][indices] - OFF) // BIN).astype(np.int64)
        n_metric_bins = n_bins_by_chromosome[name]
        grid_mask = ((b1 >= 0) & (b2 >= 0) & (b1 < n_metric_bins) & (b2 < n_metric_bins))
        evaluated = report.evaluate_preregistered_candidates(
            candidates=candidate_coordinates,
            selected_id=verified.selected_id,
            random=baseline_coordinates["random"],
            oracle=baseline_coordinates["oracle"],
            consensus=baseline_coordinates["consensus"],
            reference=reference,
            chromosome_index=chromosome_index,
            chromosome_name=name,
            labels=lab[indices],
            b1=b1,
            b2=b2,
            n_metric_bins=n_metric_bins,
        )
        candidate_rows = {
            candidate_id: _augment_metric_row(dict(row))
            for candidate_id, row in evaluated["candidates"].items()
        }
        first_candidate = next(iter(candidate_rows.values())) if candidate_rows else None
        fixed_baselines = {
            "consensus": first_candidate["consensus"] if first_candidate is not None else None,
            "random": {
                "R1": first_candidate["R1"].get("random") if first_candidate is not None else None,
                "R2": first_candidate["R2"].get("by_candidate", {}).get("random") if first_candidate is not None else None,
                "R3": first_candidate["R3"].get("by_candidate", {}).get("random") if first_candidate is not None else None,
            },
            "oracle": {
                "R1": first_candidate["R1"].get("oracle") if first_candidate is not None else None,
                "R2": first_candidate["R2"].get("by_candidate", {}).get("oracle") if first_candidate is not None else None,
                "R3": first_candidate["R3"].get("by_candidate", {}).get("oracle") if first_candidate is not None else None,
            },
        }
        per_chromosome[name] = {
            "chromosome": name,
            "chromosome_index": chromosome_index,
            "n_cis_records": int(len(indices)),
            "n_metric_grid_records": int(grid_mask.sum()),
            "n_out_of_grid_records": int((~grid_mask).sum()),
            "n_bins_metric_grid": int(n_metric_bins),
            "candidates": candidate_rows,
            "fixed_baselines": fixed_baselines,
            "candidate_order": list(candidate_rows),
        }

    metadata = _candidate_metadata(verified)
    for item in metadata.values():
        item["selected_id_frozen"] = verified.selected_id
    candidate_summary = {
        candidate_id: _candidate_summary(candidate_id, metadata[candidate_id],
                                         chromosome_names, per_chromosome)
        for candidate_id in verified.document.get("preregistered_candidate_ids", [])
    }
    candidate_vs_random = {
        candidate_id: candidate_summary[candidate_id]["comparison_vs_random"]
        for candidate_id in candidate_summary
        if candidate_summary[candidate_id]["terminal_status"] != "failed"
    }
    fixed_comparison = _fixed_baseline_comparison(chromosome_names, per_chromosome)

    # 将每个未失败 candidate 的 payload 复制到新的 evaluation directory；
    # failed candidate 有意不获得 coordinate path。
    delivered_candidates: dict[str, Any] = {}
    for candidate_id, candidate in verified.candidates.items():
        if candidate.coordinates_path is None:
            delivered_candidates[candidate_id] = {
                "candidate_id": candidate_id, "terminal_status": candidate.terminal_status,
                "failure_reason": candidate.failure_reason, "coordinates": None,
            }
            continue
        target = root / "coordinates" / "candidates" / (candidate_id + ".3dg")
        delivered_candidates[candidate_id] = {
            "candidate_id": candidate_id, "terminal_status": candidate.terminal_status,
            "coordinates": _copy_coordinate_payload(
                root, Path(candidate.coordinates_path), candidate.inventory.sha256, target
            ),
        }
    selected_candidate = verified.selected
    if selected_candidate.coordinates_path is None or selected_candidate.inventory is None:
        raise EvaluationError("selected candidate has no verified coordinate payload")
    selected_coordinate = _copy_coordinate_payload(
        root, Path(selected_candidate.coordinates_path), selected_candidate.inventory.sha256,
        root / "coordinates" / "selected.3dg",
    )
    track_map = report.write_track_map(root / "track_map.json", verified.grid)

    selected_structs = candidate_coordinates[verified.selected_id]
    selected_r3 = {
        name: per_chromosome[name]["candidates"][verified.selected_id]["R3"]["by_candidate"]["selected"]
        for name in chromosome_names
    }
    selected_r3["__chromosome_lengths__"] = {
        chromosome.name: chromosome.length_bp for chromosome in verified.grid.chromosomes
    }
    plot_dir = root / "plots"
    selected_figure = report.write_selected_structure_figure(
        plot_dir / "selected_structure_3d.png", selected_structs, verified.grid,
        title="selected raw R=1 unit-ball structure",
    )
    selected_html_path = plot_dir / "selected_structure_3d.html"
    selected_html_path.write_text(
        report._canvas_html(selected_structs, verified.grid, "selected raw R=1 unit-ball structure"),
        encoding="utf-8",
    )
    coverage = report.write_track_coverage_table(plot_dir / "track_coverage.tsv", selected_structs, verified.grid)
    stripes = report.write_fragment_stripes(
        plot_dir / "fragment_consistency_stripes.png",
        plot_dir / "fragment_consistency_stripes.tsv",
        selected_r3, chromosome_names,
    )
    trace = _trace_from_selection(verified, verified.selected_id)
    trace_delivery = report.write_optimizer_trace(plot_dir / "optimizer_trace.png", trace)

    gallery_panels = []
    for candidate_id, structs in candidate_coordinates.items():
        gallery_panels.append({
            "label": "candidate:%s" % candidate_id,
            "structures": structs,
            "tracks": _gallery_tracks(verified.grid, "candidate"),
        })
    for tag in ("random", "oracle", "consensus"):
        gallery_panels.append({
            "label": "baseline:%s" % tag,
            "structures": baseline_coordinates[tag],
            "tracks": _gallery_tracks(verified.grid, tag),
        })
    gallery_panels.append({
        "label": "reference:evaluation-only",
        "structures": reference,
        "tracks": _gallery_tracks(verified.grid, "reference"),
    })
    gallery = report.write_independent_comparison_figure(plot_dir / "candidate_gallery.png", gallery_panels)
    distance_chromosome_index = next(
        (index for index, chromosome in enumerate(verified.grid.chromosomes)
         if n_bins_by_chromosome[chromosome.name] >= 2), None
    )
    distance_maps = None
    if distance_chromosome_index is not None:
        distance_maps = _write_distance_map_preview(
            plot_dir / "selected_distance_maps.png", selected_structs, reference,
            verified.grid, distance_chromosome_index,
            n_bins_by_chromosome[verified.grid.chromosomes[distance_chromosome_index].name],
        )

    code_snapshot = _evaluation_code_snapshot(root)
    alignment = dict(loaded["record_alignment"])
    if alignment_evidence:
        alignment["evidence"] = _safe(alignment_evidence[-1])
    a1_known = a1 >= 0
    a2_known = a2 >= 0
    labelled = {
        "n_records": int(len(a1)),
        "n_both_ends_labelled": int((a1_known & a2_known).sum()),
        "n_labelled_cis": int((a1_known & a2_known & contacts["cis"]).sum()),
        "n_labelled_inter": int((a1_known & a2_known & ~contacts["cis"]).sum()),
        "n_labelled_inter_crosscopy": int((a1_known & a2_known & ~contacts["cis"] & (a1 != a2)).sum()),
    }
    delivery = {
        "path": str(root),
        "selected_coordinates": {**selected_coordinate, "path": _relative(root, selected_coordinate["path"])},
        "track_map": {**track_map, "path": _relative(root, track_map["path"])},
        "selected_figure": {**selected_figure, "figure": _relative(root, selected_figure["figure"])},
        "selected_html": {
            "html": _relative(root, selected_html_path), "mode": "vanilla_canvas_embedded",
            "external_cdn": False, "coordinate_transform": "display_projection_only_raw_export_unchanged",
        },
        "track_coverage": {**coverage, "table": _relative(root, coverage["table"])},
        "fragment_stripes": {
            **stripes, "figure": _relative(root, stripes["figure"]), "table": _relative(root, stripes["table"])
        },
        "gallery": {**gallery, "figure": _relative(root, gallery["figure"])},
        "renderer_provenance": report.renderer_provenance(),
    }
    if trace_delivery is not None:
        delivery["optimizer_trace"] = {**trace_delivery, "figure": _relative(root, trace_delivery["figure"])}
    if distance_maps is not None:
        delivery["distance_maps"] = {**distance_maps, "figure": _relative(root, distance_maps["figure"])}

    metrics = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "evaluation_complete",
        "synthetic_fixture": bool(synthetic_fixture),
        "selection": {
            "selection_path": str(selection_file),
            "selected_id_frozen": verified.selected_id,
            "reference_reselection_performed": False,
            "rule": verified.document["selection"]["rule"],
            "criterion": verified.document["objective_contract"]["criterion"],
            "direction": verified.document["objective_contract"]["direction"],
        },
        "cohort": verified.document["cohort"],
        "coordinate_grid": {
            "bin_size_bp": verified.grid.bin_size_bp,
            "origin_bp": verified.grid.origin_bp,
            "n_loci": verified.grid.n_loci,
            "n_physical_beads": verified.grid.n_physical_beads,
            "nuclear_radius": report.NUCLEAR_RADIUS,
            "coordinate_units": "dimensionless_R1",
            "metric_grid_offset_bp": OFF,
            "metric_bins_by_chromosome": n_bins_by_chromosome,
        },
        "evaluator_input_audit": loaded["evaluator_input_audit"],
        "evaluator_input_audit_preflight": preflight_input_audit,
        "record_alignment": alignment,
        "labelled_evaluator_records": labelled,
        "evaluation_gate": {
            "armed_stage": refeval.STAGE,
            "entries": gate.entries,
            "all_coordinate_paths_registered_before_arm": True,
        },
        "candidate_metadata": metadata,
        "candidate_delivery": delivered_candidates,
        "baseline_metadata": {
            tag: {
                "tag": item.tag, "role": item.role,
                "source_gate_path": item.source_gate_path,
                "coordinates_path": item.coordinates_path,
                "inventory": item.inventory.as_dict(),
            }
            for tag, item in loaded["verified_baselines"].items()
        },
        "training_attempts": verified.document.get("attempts", []),
        "metric_policy": {
            "R1": "same-chromosome same-copy labelled non-diagonal OFF=3Mb records; candidate/random/oracle/reference common finite mask; reference and fully phased S0 oracle-fit ceilings",
            "R2": "candidate/random/reference six-track common finite non-diagonal pair mask with per-chromosome geometry gauge",
            "R3": "common resolvable 20Mb fragments; tie/missing/insufficient states explicit and break runs",
            "inter_contacts": "used by training but never used for haplotype accuracy",
            "consensus": "single-track structural baseline; two-copy contrast n/a",
            "bootstrap": "paired chromosome resampling within one cell; technical/structural, not biological replicate",
            "direction": "delta_candidate_minus_random",
            "p": "nuisance kernel-mixture coefficient, not same-copy percentage",
        },
        "per_chromosome": per_chromosome,
        "candidate_summary": candidate_summary,
        "candidate_vs_random": candidate_vs_random,
        "fixed_baseline_comparison": fixed_comparison,
        "delivery": delivery,
        "evaluation_code_snapshot": code_snapshot,
        "l2_conclusion": l2_conclusion,
    }
    perchr_path = root / "metrics" / "per_chromosome.tsv"
    metrics["per_chromosome_table"] = _write_per_chromosome_tsv(perchr_path, verified, per_chromosome)
    summary = {
        "status": metrics["status"], "selected_id_frozen": verified.selected_id,
        "reference_reselection_performed": False,
        "candidate_summary": candidate_summary,
        "candidate_vs_random": candidate_vs_random,
        "fixed_baseline_comparison": fixed_comparison,
        "delivery": delivery,
    }
    metrics_path = root / "metrics" / "metrics.json"
    summary_path = root / "metrics" / "summary.json"
    _write_json(metrics_path, metrics)
    _write_json(summary_path, summary)
    metrics["summary_path"] = _relative(root, summary_path)
    metrics["metrics_path"] = _relative(root, metrics_path)
    _write_json(metrics_path, metrics)
    _write_json(root / "config.json", {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "selection_path": str(selection_file),
        "output_dir": str(root),
        "strict_p9016": strict_p9016,
        "synthetic_fixture": synthetic_fixture,
        "evaluator_input_spec": _safe(input_spec.__dict__),
        "baseline_specs": {tag: _safe(spec.__dict__) for tag, spec in specs.items()},
        "code_snapshot": code_snapshot,
        "loader_contract": {
            "candidate": "candidate_loader(path)",
            "baseline": "baseline_loader(tag,path)",
            "contacts": "contacts_loader()",
            "alignment": "alignment_verifier(contacts,raw_path)",
            "labels": "labels_loader(contacts,raw_path)",
            "reference": "reference_loader(path)",
        },
    })
    readme_path = root / "README.md"
    readme_path.write_text(_render_readme(verified, metrics, delivery, l2_conclusion=l2_conclusion), encoding="utf-8")
    return {
        "outdir": str(root),
        "evaluation_dir": str(root),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "readme_path": str(readme_path),
        "metrics": metrics,
        "verified_selection": verified,
    }


# 供需要命令式 API 的调用者使用的简短 alias。
run = run_final_evaluation


def _cli_baseline_specs(path: str | None) -> dict[str, report.BaselineSpec] | None:
    if path is None:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError("cannot read --baseline-specs-json") from exc
    if not isinstance(value, Mapping):
        raise EvaluationError("--baseline-specs-json must contain an object keyed by tag")
    result = {}
    for tag, raw in value.items():
        if not isinstance(raw, Mapping):
            raise EvaluationError("baseline spec %s must be an object" % tag)
        fields = dict(raw)
        fields.setdefault("tag", tag)
        result[str(tag)] = report.BaselineSpec(**fields)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="guarded final reconstruction evaluation")
    parser.add_argument("--selection", required=True, help="training selection.json")
    parser.add_argument("--raw-pairs", required=True, help="phase-bearing raw pairs used only by evaluator audit/labels")
    parser.add_argument("--reference-3dg", required=True, help="evaluator-only reference 3DG")
    parser.add_argument("--reference-017-provenance", required=True, help="hashed 017 provenance JSON")
    parser.add_argument("--reference-017-provenance-sha256", required=True)
    parser.add_argument("--raw-sha256", default=report.RAW_P9016_PAIRS_SHA256)
    parser.add_argument("--record-count", type=int, default=report.RAW_P9016_RECORDS)
    parser.add_argument("--baseline-specs-json", help="optional explicit BaselineSpec mapping; defaults to selection metadata")
    parser.add_argument("--out", help="new evaluation child directory; default evaluation-UTC under training run")
    parser.add_argument("--non-strict-p9016", action="store_true", help="only for synthetic/non-P9016 fixtures")
    parser.add_argument("--l2-conclusion", default="L2 未证：最终结论必须结合全 chromosome R3 一致性、共同分母和 paired CI；本入口不把低 wall 或局部 majority 自动升级为 L2。")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    spec = report.EvaluatorInputAuditSpec(
        raw_pairs_path=args.raw_pairs,
        reference_3dg_path=args.reference_3dg,
        reference_017_provenance_path=args.reference_017_provenance,
        reference_017_provenance_sha256=args.reference_017_provenance_sha256,
        expected_raw_pairs_sha256=args.raw_sha256,
        expected_record_count=args.record_count,
    )
    result = run_final_evaluation(
        args.selection,
        evaluator_input_spec=spec,
        baseline_specs=_cli_baseline_specs(args.baseline_specs_json),
        output_dir=args.out,
        strict_p9016=not args.non_strict_p9016,
        l2_conclusion=args.l2_conclusion,
    )
    print(json.dumps({
        "evaluation_dir": result["evaluation_dir"],
        "metrics_path": result["metrics_path"],
        "summary_path": result["summary_path"],
        "readme_path": result["readme_path"],
        "selected_id": result["metrics"]["selection"]["selected_id_frozen"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
