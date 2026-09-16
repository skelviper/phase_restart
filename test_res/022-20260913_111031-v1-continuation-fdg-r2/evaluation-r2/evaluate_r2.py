"""在 parent paths 填充后运行冻结的仅 R2 评估。

脚本先哈希每个 candidate/control 坐标，将其逐一注册到新的 EvalGate，然后才打开 gate 并读取仅供评估的 reference。由于 R2 不需要 labels，它不会打开带 phase 的 pairs。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pr import r2comparison, ref3dg  # noqa: E402
from pr.gate import EvalGate, sha256_file  # noqa: E402
from pr.refeval import STAGE as EVAL_STAGE  # noqa: E402
from pr.reconstruction_report import _read_coordinate_rows  # noqa: E402


CONFIG_SCHEMA = "p9016-r2-evaluation-config-v1"
GATE_STAGE = EVAL_STAGE
EXPECTED_REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot read JSON config %s: %s" % (path, exc)) from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("config must contain a JSON object")
    return dict(value)


def _resolve(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise RuntimeError("%s must be a lowercase SHA256 digest" % label)
    return value


def _grid_from_config(config: Mapping[str, Any], config_path: Path) -> list[dict[str, Any]]:
    raw_grid = config.get("grid")
    if isinstance(raw_grid, Mapping) and isinstance(raw_grid.get("chromosomes"), list):
        chromosomes = raw_grid["chromosomes"]
        if chromosomes:
            return [{"name": str(item["name"]), "length_bp": int(item["length_bp"])} for item in chromosomes]
    selection_path = config.get("selection_path")
    if selection_path:
        selection = _read_json(_resolve(str(selection_path), config_path.parent))
        coordinate_grid = selection.get("coordinate_grid")
        if isinstance(coordinate_grid, Mapping) and isinstance(coordinate_grid.get("chromosomes"), list):
            return [{"name": str(item["name"]), "length_bp": int(item["length_bp"])}
                    for item in coordinate_grid["chromosomes"]]
    raise RuntimeError("config must provide grid.chromosomes or selection_path metadata")


def _parse_native_tsv(path: Path, chromosome_names: list[str]) -> dict[str, dict[int, np.ndarray]]:
    """将无 phase 的 softall interval TSV 解析为规范的 cNN[a/b] 轨迹。"""
    result: dict[str, dict[int, np.ndarray]] = {}
    index = {name: index for index, name in enumerate(chromosome_names)}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chr", "copy", "start", "x", "y", "z"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RuntimeError("native TSV %s lacks required fields %s" % (path, sorted(required)))
        for line_no, row in enumerate(reader, start=2):
            try:
                chromosome = str(row["chr"])
                chromosome_index = index[chromosome]
                copy = int(row["copy"])
                position = int(row["start"])
                point = np.asarray([float(row[axis]) for axis in ("x", "y", "z")], dtype=float)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("invalid native TSV row %d in %s" % (line_no, path)) from exc
            if copy not in (0, 1) or position < 0 or position % r2comparison.BIN_SIZE_BP != 0:
                raise RuntimeError("invalid native TSV chromosome/copy/start at row %d in %s" % (line_no, path))
            if not np.isfinite(point).all():
                raise RuntimeError("nonfinite native TSV coordinate at row %d in %s" % (line_no, path))
            track = "c%02d%s" % (chromosome_index + 1, "ab"[copy])
            target = result.setdefault(track, {})
            if position in target:
                raise RuntimeError("duplicate native TSV bead %s:%d" % (track, position))
            target[position] = point
    return result


def _load_coordinates(path: Path, format_name: str, chromosome_names: list[str]) -> dict[str, dict[int, np.ndarray]]:
    if format_name in ("3dg", "3dg_text"):
        return _read_coordinate_rows(path)
    if format_name in ("native_tsv", "coords_tsv"):
        return _parse_native_tsv(path, chromosome_names)
    raise RuntimeError("unsupported coordinate format %s" % format_name)


def _verify_and_load_candidates(
    config: Mapping[str, Any], config_path: Path, output_dir: Path,
    chromosome_names: list[str],
) -> tuple[dict[str, r2comparison.R2Condition], dict[str, Any], EvalGate]:
    raw_conditions = config.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise RuntimeError("config.conditions must be a non-empty list")
    gate = EvalGate(str(output_dir / "gate.json"))
    loaded: dict[str, r2comparison.R2Condition] = {}
    provenance = []
    skipped = []
    seen_ids = set()
    for item in raw_conditions:
        if not isinstance(item, Mapping):
            raise RuntimeError("each condition must be an object")
        condition_id = str(item.get("id", ""))
        if not condition_id or condition_id in seen_ids:
            raise RuntimeError("condition ids must be unique and non-empty")
        seen_ids.add(condition_id)
        if item.get("enabled", True) is False:
            skipped_item = {
                "condition_id": condition_id,
                "role": str(item.get("role", "main")),
                "status": str(item.get("endpoint_status", "disabled")),
                "reason": "disabled_optional_condition_not_pre-registered_for_this_run",
                "registered_in_gate": False,
            }
            path_text = item.get("path")
            expected_sha_text = item.get("sha256")
            if path_text and expected_sha_text:
                expected_sha = _require_sha(expected_sha_text, "%s.sha256" % condition_id)
                path = _resolve(str(path_text), config_path.parent)
                if not path.is_file():
                    raise RuntimeError("missing disabled coordinate file for %s: %s" % (condition_id, path))
                actual_sha = sha256_file(path)
                if actual_sha != expected_sha:
                    raise RuntimeError("disabled coordinate SHA256 mismatch for %s: %s != %s" %
                                       (condition_id, actual_sha, expected_sha))
                skipped_item.update({"path": str(path), "sha256": actual_sha, "hash_verified": True})
            else:
                skipped_item["hash_verified"] = False
            skipped.append(skipped_item)
            continue
        path_text = item.get("path")
        expected_sha = _require_sha(item.get("sha256"), "%s.sha256" % condition_id)
        if not isinstance(path_text, str) or not path_text:
            raise RuntimeError("%s.path is required before real evaluation" % condition_id)
        path = _resolve(path_text, config_path.parent)
        if not path.is_file():
            raise RuntimeError("missing coordinate file for %s: %s" % (condition_id, path))
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError("coordinate SHA256 mismatch for %s: %s != %s" %
                               (condition_id, actual_sha, expected_sha))
        gate.register(GATE_STAGE, "r2:%s" % condition_id, str(path))
        coordinates = _load_coordinates(path, str(item.get("format", "3dg")), chromosome_names)
        n_copies = int(item.get("n_copies", 2))
        endpoint_status = str(item.get("endpoint_status", "accepted"))
        accepted_as = item.get("accepted_as")
        display_name = str(item.get("display_name", condition_id))
        if endpoint_status.lower() == "rejected":
            accepted_text = accepted_as or item.get("anchor_condition_id") or "unknown"
            if "rejected" not in display_name.lower():
                display_name += "\\n(rejected; accepted=%s)" % str(accepted_text)
        loaded[condition_id] = r2comparison.R2Condition(
            condition_id=condition_id,
            display_name=display_name,
            structures=coordinates,
            n_copies=n_copies,
            role=str(item.get("role", "main")),
            endpoint_status=endpoint_status,
            accepted_as=accepted_as,
        )
        provenance.append({
            "condition_id": condition_id,
            "path": str(path),
            "sha256": actual_sha,
            "format": str(item.get("format", "3dg")),
            "role": str(item.get("role", "main")),
            "endpoint_status": endpoint_status,
            "accepted_as": accepted_as,
            "anchor_condition_id": item.get("anchor_condition_id"),
        })
    return loaded, {"coordinates": provenance, "skipped": skipped}, gate


def _reject_duplicate_accepted_trials(config: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    """阻止将已接受的 anchor duplicate 呈现为 improvement。"""
    by_id = {item["condition_id"]: item for item in provenance.get("coordinates", [])}
    raw_conditions = config.get("conditions", [])
    for item in raw_conditions:
        if not isinstance(item, Mapping) or item.get("enabled", True) is False:
            continue
        if str(item.get("role", "")) != "accepted_trial":
            continue
        condition_id = str(item.get("id"))
        anchor_id = item.get("anchor_condition_id")
        if str(item.get("accepted_as", "")).lower() == "anchor":
            raise RuntimeError("accepted_trial %s is marked accepted=anchor; disable it instead of reporting an improvement" % condition_id)
        if anchor_id in by_id and condition_id in by_id and by_id[anchor_id]["sha256"] == by_id[condition_id]["sha256"]:
            raise RuntimeError("accepted_trial %s has the same coordinate SHA256 as anchor %s; do not report a duplicate as FDG improvement" % (condition_id, anchor_id))


def _load_reference_after_gate(config: Mapping[str, Any], config_path: Path, gate: EvalGate) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = config.get("reference")
    if not isinstance(raw, Mapping):
        raise RuntimeError("config.reference is required")
    path_text = raw.get("path")
    if not isinstance(path_text, str) or not path_text:
        raise RuntimeError("reference.path is required")
    expected_sha = _require_sha(raw.get("sha256"), "reference.sha256")
    if expected_sha != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("reference.sha256 must match the frozen P9016 reference digest")
    path = _resolve(path_text, config_path.parent)
    if not path.is_file():
        raise RuntimeError("reference file is missing: %s" % path)
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError("reference SHA256 mismatch: %s != %s" % (actual_sha, expected_sha))
    gate.arm(GATE_STAGE)
    reference = ref3dg.load_reference(gate, path=str(path))
    return reference, {"path": str(path), "sha256": actual_sha, "loaded_after_gate_arm": True}


def _readme(config: Mapping[str, Any], result: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    summary = result["summary"]["condition_summary"]
    fdg_config = next((item for item in config.get("conditions", [])
                       if isinstance(item, Mapping) and item.get("id") == "fdg_proposal"), {})
    anchor_condition_id = str(fdg_config.get("anchor_condition_id", "anchor"))
    lines = [
        "# P9016 V1 延续 / FDG：仅 R2 评价",
        "",
        "**状态：** `evaluation_complete`。本目录只评价 R2，不运行或汇报 R1/R3。",
        "",
        "## 冻结方法",
        "",
        "- 单个 P9016 单细胞，20 条染色体；1 Mb 完整网格，起点 OFF=3 Mb，终点严格小于染色体 header length。",
        "- 观测单位是无序、非对角的基因组 bin 对。每条染色体的共同有限 mask 同时包含所有主条件、纳入比较的对照和 reference 两条轨迹。",
        "- 双轨条件在同一 mask 上计算 `rho_A_mat`、`rho_A_pat`、`rho_B_mat`、`rho_B_pat`；`direct=(A_mat+B_pat)/2`，`cross=(A_pat+B_mat)/2`。几何选择较大者为 `matched`、较小者为 `other`；两者在 1e-12 内相等时取平均且 `contrast=0`。",
        "- `similarity=matched`，`contrast=matched-other`。这是 best-geometry gauge 下的非负量；正值不等于优于 random 或已完成 L2 恢复。配对 delta 保留负值。",
        "- 单轨 consensus 只提供单轨相似度对照：`similarity=(rho_A_mat+rho_A_pat)/2`，要求两个参考 rho 都有限，并把 pairing/geometry_status 写为 `reference_mean`；其双轨 contrast 固定为 n/a，不复制轨迹制造零值。",
        "- constant、nonfinite 或共同 pair 数不足 20 时写入 n/a 与原因；染色体和条件仍保留在表格中。",
        "- bootstrap 固定 seed=9301、10000 次 chromosome resampling；这是同一细胞内的技术/结构 CI，不是生物学重复或 p-value。",
        "",
        "## Gate 与输入",
        "",
        "- 所有 candidate/control 坐标先逐个计算 SHA256 并登记到全新的 `EvalGate`，之后才 arm gate 并读取仅供评价的 reference。R2 不需要带 phase 的 pairs，因此本目录不打开 phase payload。",
        "- `endpoint_status`、`accepted_as` 只记录训练侧无标签接受语义。FDG full proposal 总是保留；若被训练规则拒绝，proposal 仍照实出现在图和表中并标记 `rejected; accepted=anchor (%s)`。accepted trial 只有在非平凡 alpha<1 且不同于 anchor 时才启用；R2 不改变接受决定。" % anchor_condition_id,
        "",
        "## R2 摘要",
        "",
        "| 条件 | 相似度均值 | 相似度中位数 | contrast 均值 | contrast 中位数 | 有效染色体数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition_id, item in summary.items():
        def fmt(value):
            return "n/a" if value is None else "%.6f" % float(value)
        lines.append("| `%s` | %s | %s | %s | %s | %d/%d |" % (
            condition_id, fmt(item["similarity_macro_mean"]), fmt(item["similarity_median"]),
            fmt(item["contrast_macro_mean"]), fmt(item["contrast_median"]),
            item["similarity_n_finite"], item["n_chromosomes_expected"]
        ))
    lines.extend([
        "",
        "配对比较详见 `r2_summary.json`；每个 comparison 同时保留每染色体 delta、wins/ties/losses、缺失状态和 bootstrap CI。",
        "",
        "## 交付",
        "",
        "- `r2_main.png` / `r2_main.pdf`：四个主条件的双面板图；若配置 accepted trial 为 supplement，另有 `r2_supplement.png` / `.pdf`。",
        "- `r2_controls.png` / `r2_controls.pdf`：random014 与 consensus014 的配套对照图。",
        "- `r2_per_chromosome.tsv` / `.csv`：所有条件的 20 条染色体记录、四 rho、matched/other/contrast、mask/coverage/status。",
        "- `r2_results.json`、`r2_summary.json`、`evaluation_manifest.json`：完整 R2 结果与输入/gate 溯源信息。",
        "",
        "输入摘要：",
        "```json",
        json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False, sort_keys=True),
        "```",
        "",
    ])
    return "\n".join(lines)


def run(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = _read_json(config_file)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise RuntimeError("unsupported R2 config schema")
    if config.get("status") not in ("ready", "run"):
        raise RuntimeError("config.status must be ready before real evaluation; current=%r" % config.get("status"))
    cohort = config.get("cohort", {})
    if cohort.get("sample_id") not in (None, "P9016") or cohort.get("n_chromosomes") not in (None, 20):
        raise RuntimeError("R2 config cohort must be P9016 with 20 chromosomes")
    grid_meta = config.get("grid", {})
    if isinstance(grid_meta, Mapping):
        if grid_meta.get("bin_size_bp", r2comparison.BIN_SIZE_BP) != r2comparison.BIN_SIZE_BP:
            raise RuntimeError("R2 grid bin_size_bp must be 1000000")
        if grid_meta.get("offset_bp", r2comparison.GRID_OFFSET_BP) != r2comparison.GRID_OFFSET_BP:
            raise RuntimeError("R2 grid offset_bp must be 3000000")
    output_value = config.get("output_dir", "results")
    output_dir = _resolve(str(output_value), config_file.parent)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to reuse non-empty R2 output directory: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    chromosomes = _grid_from_config(config, config_file)
    if len(chromosomes) != 20:
        raise RuntimeError("P9016 R2 evaluation requires exactly 20 chromosomes")
    chromosome_names = [item["name"] for item in chromosomes]
    expected_names = ["chr%d" % index for index in range(1, 20)] + ["chrX"]
    if chromosome_names != expected_names:
        raise RuntimeError("P9016 R2 evaluation requires chr1..chr19,chrX in frozen order")
    conditions, coordinate_manifest, gate = _verify_and_load_candidates(
        config, config_file, output_dir, chromosome_names,
    )
    _reject_duplicate_accepted_trials(config, coordinate_manifest)
    # 这是第一次加载 reference，并且发生在所有坐标哈希完成之后。
    reference, reference_manifest = _load_reference_after_gate(config, config_file, gate)
    evaluated = r2comparison.evaluate_genome(chromosomes, conditions, reference)
    condition_ids = list(conditions)
    main_ids = [str(item) for item in config.get("main_condition_ids", [])]
    control_ids = [str(item) for item in config.get("control_condition_ids", [])]
    supplement_ids = [str(item) for item in config.get("supplement_condition_ids", [])]
    declared_ids = main_ids + control_ids + supplement_ids
    if len(set(declared_ids)) != len(declared_ids) or set(declared_ids) != set(condition_ids):
        raise RuntimeError("main/control/supplement condition lists must partition enabled config.conditions")
    comparisons = config.get("comparisons", [])
    summary_outputs = r2comparison.write_outputs(
        output_dir, evaluated, condition_ids,
        main_condition_ids=main_ids,
        control_condition_ids=control_ids,
        supplement_condition_ids=supplement_ids,
        comparisons=comparisons,
        seed=int(config.get("bootstrap", {}).get("seed", r2comparison.BOOTSTRAP_SEED)),
        n_boot=int(config.get("bootstrap", {}).get("n_boot", r2comparison.BOOTSTRAP_DRAWS)),
        title="P9016 | V1 continuation / FDG | R2 only",
    )
    manifest = {
        "schema_version": CONFIG_SCHEMA,
        "status": "evaluation_complete",
        "config_path": str(config_file),
        "reference": reference_manifest,
        "coordinates": coordinate_manifest["coordinates"],
        "skipped_conditions": coordinate_manifest["skipped"],
        "gate": {"stage": GATE_STAGE, "entries": gate.entries, "armed": sorted(gate.armed)},
        "reference_loaded_after_all_coordinate_hashes": True,
        "phase_payload_opened": False,
        "selection_changed": False,
        "r2_only": True,
        "code": {
            "r2_module": str(ROOT / "pr" / "r2comparison.py"),
            "r2_module_sha256": sha256_file(ROOT / "pr" / "r2comparison.py"),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "outputs": summary_outputs["paths"],
    }
    r2comparison.write_json(output_dir / "evaluation_manifest.json", manifest)
    (output_dir / "config.json").write_text(
        json.dumps(_jsonable(config), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_readme(config, summary_outputs, manifest), encoding="utf-8")
    return {"output_dir": str(output_dir), "manifest": manifest, "summary": summary_outputs["summary"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen P9016 R2-only evaluation")
    parser.add_argument("--config", required=True, help="ready R2 config JSON")
    args = parser.parse_args(argv)
    result = run(args.config)
    print(json.dumps(_jsonable({"output_dir": result["output_dir"], "summary": result["summary"]}),
                     indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
