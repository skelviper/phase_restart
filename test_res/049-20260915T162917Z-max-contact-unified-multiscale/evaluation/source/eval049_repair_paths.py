#!/usr/bin/env python
"""049 派生输出路径修复（**不重算任何科学指标**）。

背景：`evaluate` 早期版本对 manifest 里已经是仓库相对形式（``test_res/049-.../coords/...``）的
`npz_path` 直接调用 `lib.rel()`，得到 CWD 相对的 ``evaluation/test_res/...`` 错误路径。本脚本：

1. 读取**已保存**的 `results/evaluation.json` 与 `results/terminal.json`；
2. 只用冻结 manifest 把 12 个 fit 的 `npz_path` 换成 `resolve_path` 之后的 run 相对路径；
3. 逐字段深比较修复前后的 JSON（排除被修字段），断言**数值与所有其它字段完全不变**；
4. 重写这两个文件，并把 old/new sha256、被修字段数写入 `results/validation.json` 的 `path_repair`；
5. 对所有记录路径做存在性检查，写 `results/path_check.json`。

不读 reference、不重算 R2/inter/空间/bootstrap/null。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval049_inputs as inputs  # noqa: E402
import eval049_lib as lib  # noqa: E402

RESULTS = lib.EVAL / "results"
PATH_FIELDS = ("npz_path", "three_dg_path", "path")


def strip_paths(value: Any) -> Any:
    """复制 JSON 树并去掉所有路径字段，用于数值/其它字段的严格比较。"""
    if isinstance(value, dict):
        return {k: strip_paths(v) for k, v in value.items() if k not in PATH_FIELDS and not k.endswith("_path")}
    if isinstance(value, list):
        return [strip_paths(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    evaluation_path = RESULTS / "evaluation.json"
    terminal_path = RESULTS / "terminal.json"
    validation_path = RESULTS / "validation.json"
    before_sha = {"evaluation.json": lib.sha256_file(evaluation_path), "terminal.json": lib.sha256_file(terminal_path)}
    evaluation = lib.read_json(evaluation_path)
    terminal = lib.read_json(terminal_path)
    validation = lib.read_json(validation_path)
    manifest = inputs.load_manifest(Path(args.manifest))

    corrected: dict[str, str] = {}
    for row in manifest["fits"]:
        fit_id = str(row["fit_id"])
        corrected[fit_id] = lib.rel(lib.resolve_path(row["npz_path"]))
    for fit_id, path in corrected.items():
        if not (lib.RUN / path).is_file():
            raise RuntimeError("corrected path does not exist: %s -> %s" % (fit_id, path))

    eval_before, term_before = copy.deepcopy(evaluation), copy.deepcopy(terminal)
    fixed_eval, fixed_term = 0, 0
    for fit_id, path in corrected.items():
        entry = evaluation["datasets"].get(fit_id)
        if entry is not None:
            if entry.get("npz_path") != path:
                entry["npz_path"] = path
                fixed_eval += 1
        for row in evaluation.get("endpoint_common_g", []):
            if row.get("candidate_id") == fit_id and row.get("npz_path") != path:
                row["npz_path"] = path
        for row in terminal.get("datasets", []):
            if row.get("candidate_id") == fit_id and row.get("npz_path") != path:
                row["npz_path"] = path
                fixed_term += 1

    if strip_paths(evaluation) != strip_paths(eval_before):
        raise RuntimeError("evaluation.json changed outside path fields; refusing to write")
    if strip_paths(terminal) != strip_paths(term_before):
        raise RuntimeError("terminal.json changed outside path fields; refusing to write")

    lib.write_json(evaluation_path, evaluation)
    lib.write_json(terminal_path, terminal)

    # 记录路径存在性检查
    checks: list[dict[str, Any]] = []
    def record(scope: str, identifier: str, value: Any) -> None:
        if value is None:
            return
        path = lib.resolve_path(value)
        checks.append({"scope": scope, "id": identifier, "recorded": str(value), "resolved": str(path),
                       "exists": bool(path.is_file())})

    for dataset_id, entry in evaluation["datasets"].items():
        record("evaluation.datasets", dataset_id, entry.get("npz_path"))
    for row in evaluation.get("null_draws", []):
        record("evaluation.null_draws", "%s/%s/%s" % (row["source_candidate_id"], row["null_kind"], row["seed"]),
               row.get("path"))
    gate = lib.read_json(lib.EVAL / "gates/pre_reference_gate.json")
    for row in gate.get("fits", []):
        record("gate.fits", str(row["fit_id"]), row.get("npz_path"))
        record("gate.fits", str(row["fit_id"]) + ":3dg", row.get("three_dg_path"))
    for row in gate.get("initials", []):
        record("gate.initials", str(row["initial"]), row.get("path"))
    record("gate.baseline", "baseline", (gate.get("baseline") or {}).get("npz_path"))
    for row in gate.get("nulls_new", []) + gate.get("nulls_reused_baseline", []):
        record("gate.nulls", "%s/%s/%s" % (row["source_candidate_id"], row["null_kind"], row["seed"]), row.get("path"))
    missing = [row for row in checks if not row["exists"]]
    path_check = {"schema": "p9016-049-path-check-v1", "created_utc": lib.utc_now(),
                  "checked": len(checks), "missing": len(missing), "status": "PASS" if not missing else "FAIL",
                  "failures": missing, "records": checks}
    lib.write_json(RESULTS / "path_check.json", path_check)
    if missing:
        raise RuntimeError("recorded paths missing on disk: %s" % missing[:5])

    validation["path_repair"] = {
        "reason": ("早期版本对已经是仓库相对形式的 manifest npz_path 直接 lib.rel()，产生 "
                   "evaluation/test_res/049-.../coords/... 错误路径；现改为 resolve_path 后再 rel"),
        "numeric_or_other_fields_changed": 0,
        "verified_by": "strip_paths 深比较（排除 _path 字段）在写入前断言完全相等",
        "fixed_fields": {"evaluation.datasets[*].npz_path": fixed_eval,
                         "terminal.datasets[*].npz_path": fixed_term,
                         "endpoint_common_g[*].npz_path": fixed_eval},
        "sha256_before": before_sha,
        "sha256_after": {"evaluation.json": lib.sha256_file(evaluation_path),
                         "terminal.json": lib.sha256_file(terminal_path)},
        "path_check": {"checked": len(checks), "missing": len(missing), "status": path_check["status"],
                       "report": lib.rel(RESULTS / "path_check.json")},
        "recomputed_metrics": False,
    }
    lib.write_json(validation_path, validation)

    print("PATH REPAIR PASS")
    print("  fixed evaluation.datasets npz_path: %d ; terminal.datasets npz_path: %d" % (fixed_eval, fixed_term))
    print("  numeric/other fields changed: 0 (deep-compared before write)")
    print("  path existence: %d checked, %d missing" % (len(checks), len(missing)))
    print("  example: A-raw-consensus -> %s" % corrected["A-raw-consensus"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
