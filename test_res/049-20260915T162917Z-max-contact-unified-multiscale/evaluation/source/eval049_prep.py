#!/usr/bin/env python
"""049 评价侧 pre-reference 阶段：输入封存核对 + null 生成 + gate JSON。

本脚本**不读** ``data/P9016.1m.3dg.gz``、**不读** phase 列、**不读** 046 mask snapshot。
只在全部候选/initial/null 写出并 hash 之后，由父侧授权才运行 evaluate 阶段。

用法：
    python eval049_prep.py --manifest <.../results/endpoint_manifest_pre_reference.json> \
        --selection <.../results/selection_pre_reference.json>
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval049_inputs as inputs  # noqa: E402
import eval049_lib as lib  # noqa: E402

GATE_PATH = lib.EVAL / "gates/pre_reference_gate.json"
FORMAL_MANIFEST_045 = lib.SOURCE_RUN_045 / "inputs/formal_manifest.json"
_AGGREGATE_CACHE: dict[str, Any] = {}
INPUT_INDEX_PATH = lib.EVAL / "results/input_index_pre_reference.json"
NULL_DIR = lib.EVAL / "nulls"
REG_EXPECTED_PATH = lib.EVAL / "results/baseline_regression_expected.json"
NULL_VARIANTS = [("u_zero", None)] + [("random_u", seed) for seed in lib.RANDOM_U_SEEDS]


def baseline_null_id(seed: int | None) -> str:
    return "baseline-046-real-extension-G-full-J__" + ("u-zero" if seed is None else "random-u-%d" % seed)


def extract_own_count(record: Mapping[str, Any], loss: str) -> float | None:
    block = record.get("count_nll_by_loss")
    if not isinstance(block, Mapping):
        return None
    entry = block.get(loss)
    if entry is None:
        return None
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, Mapping):
        for key in ("own_count", "count_nll", "value", "nll", "own"):
            if isinstance(entry.get(key), (int, float)):
                return float(entry[key])
    return None


def extract_full_j(record: Mapping[str, Any]) -> float | None:
    value = record.get("full_j")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        for key in ("total", "full_j", "value"):
            if isinstance(value.get(key), (int, float)):
                return float(value[key])
    return None


def verify_endpoint(record: Mapping[str, Any], fit_index: int | None = None) -> dict[str, Any]:
    npz_path = lib.resolve_path(record["npz_path"])
    tdg_path = lib.resolve_path(record["tdg_path"])
    if not npz_path.is_file():
        raise RuntimeError("endpoint NPZ missing: %s" % npz_path)
    actual_npz_sha = lib.sha256_file(npz_path)
    if actual_npz_sha != str(record["npz_sha256"]):
        raise RuntimeError("endpoint NPZ hash mismatch %s: %s != %s" % (npz_path, actual_npz_sha, record["npz_sha256"]))
    if not tdg_path.is_file():
        raise RuntimeError("endpoint 3DG missing: %s" % tdg_path)
    actual_tdg_sha = lib.sha256_file(tdg_path)
    if actual_tdg_sha != str(record["tdg_sha256"]):
        raise RuntimeError("endpoint 3DG hash mismatch %s: %s != %s" % (tdg_path, actual_tdg_sha, record["tdg_sha256"]))
    payload = lib.load_coords_npz(npz_path)
    declared_array = record.get("coordinate_array_sha256")
    if declared_array is not None and str(declared_array) != payload["coordinate_array_sha256"]:
        raise RuntimeError("endpoint coordinate array hash mismatch %s" % npz_path)
    readback = lib.readback_parity(payload["coordinates"], lib.load_3dg(tdg_path), _AGGREGATE_CACHE["data"])
    if not readback["within_tolerance"]:
        raise RuntimeError("3DG export readback mismatch for %s: exact=%s max abs diff %r > atol %r"
                           % (record.get("fit_id"), readback["exact_array_equal"],
                              readback["max_abs_difference"], readback["atol"]))
    terminal = str(record.get("terminal"))
    if terminal not in ("converged", "budget_not_converged", "not_converged", "failure"):
        raise RuntimeError("endpoint %s has unknown terminal state %r" % (record.get("fit_id"), terminal))
    return {"npz_path": npz_path, "npz_sha256": actual_npz_sha, "tdg_path": tdg_path, "tdg_sha256": actual_tdg_sha,
            "coordinates": payload["coordinates"], "coordinate_array_sha256": payload["coordinate_array_sha256"],
            "npz_keys": payload["keys"], "terminal": terminal, "readback": readback}


def initial_name_of(row: Mapping[str, Any]) -> str:
    source = row.get("source")
    if source in lib.SOURCES:
        return str(source)
    fit_id = str(row.get("fit_id") or "")
    for name in lib.SOURCES:
        if fit_id.endswith(name):
            return name
    raise RuntimeError("cannot determine initial control name from manifest entry: %s" % sorted(row.keys()))


def verify_initial(name: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """用 045 冻结 formal_manifest（metadata-only）核对两个 blind initial 的血统与字节。"""
    manifest = lib.read_json(FORMAL_MANIFEST_045)
    if manifest.get("phase_access") is not False or manifest.get("reference_access") is not False:
        raise RuntimeError("045 formal manifest does not assert phase/reference isolation")
    entry = manifest["real_initial_paths"][name]
    stage = entry["stages"]["1Mb"]
    path = lib.INITIAL_NPZ[name]
    actual = lib.sha256_file(path)
    if actual != lib.INITIAL_NPZ_SHA256[name]:
        raise RuntimeError("initial NPZ hash mismatch for %s: %s" % (name, actual))
    if str(stage["sha256"]) != actual or Path(str(stage["path"])).resolve() != path.resolve():
        raise RuntimeError("initial %s disagrees with the 045 formal manifest stage record" % name)
    expected_source = {
        "consensus": "e76655732deb6b8386b1b77bc76ff45d7dba1384f6931337fee80d8f4aaa8e02",
        "random": "9a48d73e1401e18349d11758e679da4c76da0904dbc467979079cb54bcd567d7",
    }[name]
    if str(entry["source_sha256"]) != expected_source:
        raise RuntimeError("initial %s 014 blind source_sha256 mismatch: %s" % (name, entry["source_sha256"]))
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        coordinates = np.asarray(archive["coordinates"], dtype=np.float64).copy()
        raw_y = np.asarray(archive["raw_y"], dtype=np.float64).copy()
        p_init = float(np.asarray(archive["p_init"]).item())
    if metadata.get("no_optimization") is not True:
        raise RuntimeError("initial %s is not a zero-optimization blind control" % name)
    if "014" not in str(metadata.get("source_root")):
        raise RuntimeError("initial %s source_root lost the 014 blind root" % name)
    if str(metadata.get("candidate")) != name or str(metadata.get("stage")) != "1Mb":
        raise RuntimeError("initial %s metadata candidate/stage mismatch" % name)
    if lib.hash_array(coordinates) != str(stage["coordinate_sha256"]):
        raise RuntimeError("initial %s coordinate array hash disagrees with 045 manifest" % name)
    if lib.hash_array(raw_y) != str(stage["raw_y_sha256"]):
        raise RuntimeError("initial %s raw_y array hash disagrees with 045 manifest" % name)
    configured = {"consensus": "a3f11b6830ae8c147e5d23d01b42616235be44b52ae05f4c58d450fd63468e3c",
                  "random": "d9a9b480d5a34ccc786ef538051931c8c0d0524292681f7fdc428de1f8942c08"}[name]
    if str(stage["raw_y_sha256"]) != configured:
        raise RuntimeError("initial %s raw_y hash disagrees with the 049 config record" % name)
    if abs(p_init - 0.75) > 1e-15 or abs(float(stage["p_init"]) - 0.75) > 1e-15:
        raise RuntimeError("initial %s p_init is not the frozen 0.75" % name)
    payload = lib.load_coords_npz(path)
    if record is not None and record.get("npz_sha256") is not None and str(record["npz_sha256"]) != actual:
        raise RuntimeError("manifest initial %s hash disagrees with 045 artifact" % name)
    # 049 若为 initial 导出配套 3DG，则一并核对（字段已在契约中，不新增别名）
    tdg_evidence = None
    tdg_path = (record or {}).get("tdg_path")
    if tdg_path:
        candidate = lib.resolve_path(tdg_path)
        if not candidate.is_file():
            raise RuntimeError("manifest initial %s three_dg_path missing on disk: %s" % (name, candidate))
        tdg_actual = lib.sha256_file(candidate)
        declared = (record or {}).get("tdg_sha256")
        if declared is not None and str(declared) != tdg_actual:
            raise RuntimeError("initial %s 3DG hash mismatch: %s != %s" % (name, tdg_actual, declared))
        readback = lib.readback_parity(coordinates, lib.load_3dg(candidate), _AGGREGATE_CACHE["data"])
        if not readback["within_tolerance"]:
            raise RuntimeError("initial %s 3DG readback mismatch (exact=%s, max abs diff %r > atol %r)"
                               % (name, readback["exact_array_equal"], readback["max_abs_difference"], readback["atol"]))
        tdg_evidence = {"three_dg_path": lib.rel(candidate), "three_dg_sha256": tdg_actual, **readback}
    return {
        "initial": name, "path": path, "sha256": actual,
        "coordinate_array_sha256": payload["coordinate_array_sha256"],
        "raw_y_sha256": lib.hash_array(raw_y), "p_init": p_init, "coordinates": coordinates,
        "lineage": {
            "no_optimization": metadata.get("no_optimization"),
            "source": metadata.get("source"),
            "source_root": metadata.get("source_root"),
            "stage": metadata.get("stage"),
            "blind_source_sha256": str(entry["source_sha256"]),
            "source_manifest": lib.rel(FORMAL_MANIFEST_045),
            "source_manifest_sha256": lib.sha256_file(FORMAL_MANIFEST_045),
            "phase_access": manifest.get("phase_access"), "reference_access": manifest.get("reference_access"),
        },
        "three_dg": tdg_evidence,
        "manifest_metadata": {
            "p": (record or {}).get("p"), "q": (record or {}).get("q"), "q_source": (record or {}).get("q_source"),
            "full_j": (record or {}).get("full_j"), "outer_fg": (record or {}).get("outer_fg"),
            "count_nll_by_loss": (record or {}).get("count_nll_by_loss"),
            "wall_s": (record or {}).get("wall_s"), "canonical_grad": (record or {}).get("canonical_grad"),
            "terminal": (record or {}).get("terminal"),
        },
    }


def verify_baseline() -> dict[str, Any]:
    actual = lib.sha256_file(lib.BASELINE_NPZ)
    if actual != lib.BASELINE_NPZ_SHA256:
        raise RuntimeError("046 baseline NPZ hash mismatch: %s" % actual)
    tdg_sha = lib.sha256_file(lib.BASELINE_3DG)
    if tdg_sha != lib.BASELINE_3DG_SHA256:
        raise RuntimeError("046 baseline 3DG hash mismatch: %s" % tdg_sha)
    payload = lib.load_coords_npz(lib.BASELINE_NPZ)
    if payload["coordinate_array_sha256"] != lib.BASELINE_COORD_ARRAY_SHA256:
        raise RuntimeError("046 baseline coordinate array hash mismatch: %s" % payload["coordinate_array_sha256"])
    return {"npz_path": lib.BASELINE_NPZ, "npz_sha256": actual, "tdg_path": lib.BASELINE_3DG, "tdg_sha256": tdg_sha,
            "coordinate_array_sha256": payload["coordinate_array_sha256"], "coordinates": payload["coordinates"],
            "terminal": "budget_not_converged", "npz_keys": payload["keys"]}


def verify_fit_json(fit_id: str, verified: Mapping[str, Any], fits_dir: Path, stages_dir: Path,
                    manifest_record: Mapping[str, Any]) -> dict[str, Any]:
    """核对 results/fits/{fit_id}.json 与 stages/{fit_id}/1Mb.json，并抽取 wall / canonical 梯度等字段。"""
    evidence: dict[str, Any] = {"checked": [], "missing": [], "mismatches": [], "extracted": {}}
    extract_fields = ("terminal", "outer_fg", "full_j", "wall_s", "canonical_grad", "count_nll_by_loss", "p", "q")
    for path in (fits_dir / ("%s.json" % fit_id), stages_dir / fit_id / "1Mb.json"):
        if not path.is_file():
            evidence["missing"].append(str(path))
            continue
        record = lib.read_json(path)
        flat = json.dumps(record, sort_keys=True)
        entry = {"path": str(path), "sha256": lib.sha256_file(path), "present_fields": {}}
        for field in extract_fields:
            value = inputs.pick(record, field, required=False)
            entry["present_fields"][field] = value is not None
            if value is not None:
                evidence["extracted"].setdefault(field, []).append({"path": str(path), "value": value})
        evidence["checked"].append(entry)
        for alias in inputs.ALIASES["terminal"]:
            if isinstance(record.get(alias), str) and record[alias] != verified["terminal"]:
                evidence["mismatches"].append({"path": str(path), "field": alias, "value": record[alias],
                                               "expected": verified["terminal"]})
        for alias in inputs.ALIASES["npz_sha256"]:
            if isinstance(record.get(alias), str) and record[alias] != verified["npz_sha256"]:
                evidence["mismatches"].append({"path": str(path), "field": alias, "value": record[alias],
                                               "expected": verified["npz_sha256"]})
        for alias in inputs.ALIASES["tdg_sha256"]:
            if isinstance(record.get(alias), str) and record[alias] != verified["tdg_sha256"]:
                evidence["mismatches"].append({"path": str(path), "field": alias, "value": record[alias],
                                               "expected": verified["tdg_sha256"]})
    if evidence["mismatches"]:
        raise RuntimeError("per-fit JSON disagrees with verified endpoint: %s" % evidence["mismatches"])
    for field in ("outer_fg", "full_j", "p", "q"):
        manifest_value = manifest_record.get(field)
        for item in evidence["extracted"].get(field, []):
            value = item["value"]
            same = (abs(float(value) - float(manifest_value)) <= 1e-9 * max(1.0, abs(float(manifest_value)))
                    if isinstance(value, (int, float)) and isinstance(manifest_value, (int, float)) else value == manifest_value)
            if not same:
                raise RuntimeError("per-fit JSON %s field %s=%r disagrees with manifest %r"
                                   % (item["path"], field, value, manifest_value))
    kinds: dict[str, list[str]] = {}
    for record in evidence["checked"]:
        kinds.setdefault(str(record["path"]), [])
    evidence["scalar_fields"] = {
        field: [item["value"] for item in evidence["extracted"].get(field, [])] for field in extract_fields
    }
    if not evidence["checked"]:
        evidence["note"] = "per-fit JSON not found; manifest fields used as-is"
    return evidence


def build_nulls(sources: Sequence[Mapping[str, Any]], data: lib.Aggregate, null_dir: Path) -> list[dict[str, Any]]:
    null_dir.mkdir(parents=True, exist_ok=True)
    slices = data.chromosome_slices()
    records: list[dict[str, Any]] = []
    for source in sources:
        coords = np.asarray(source["coordinates"], dtype=np.float64)
        for kind, seed in NULL_VARIANTS:
            if seed is None:
                variant, audit = lib.make_u_zero(coords)
                suffix = "u-zero"
            else:
                variant, audit = lib.make_random_u(coords, slices, seed)
                suffix = "random-u-%d" % seed
            path = null_dir / ("%s__%s.npz" % (source["candidate_id"], suffix))
            np.savez_compressed(path, coordinates=variant)
            with np.load(path, allow_pickle=False) as payload:
                readback = np.asarray(payload["coordinates"], dtype=np.float64).copy()
            if readback.shape != variant.shape or not np.array_equal(readback, variant):
                raise RuntimeError("null readback failure: %s" % path)
            records.append({
                "source_candidate_id": str(source["candidate_id"]), "null_kind": kind, "seed": seed,
                "path": lib.rel(path), "sha256": lib.sha256_file(path),
                "coordinate_array_sha256": lib.hash_array(variant), **audit,
            })
    return records


def verify_baseline_null_reuse(data: lib.Aggregate) -> dict[str, Any]:
    """复用 046 同 candidate SHA 的既有 baseline null；并本地重算 u_zero/2 个 seed 做 parity。"""
    gate = lib.read_json(lib.BASELINE_GATE_046)
    records = [row for row in gate["records"] if row.get("source_candidate_id") == "real-extension-G-full-J"]
    if len(records) != 17:
        raise RuntimeError("046 baseline null record count %d != 17" % len(records))
    out: list[dict[str, Any]] = []
    by_suffix: dict[str, Mapping[str, Any]] = {}
    for row in records:
        path = (lib.RUN_046 / str(row["path"])).resolve()
        if not path.is_file():
            raise RuntimeError("046 baseline null missing: %s" % path)
        actual = lib.sha256_file(path)
        if actual != str(row["sha256"]):
            raise RuntimeError("046 baseline null hash changed: %s" % path)
        suffix = "u-zero" if row["null_kind"] == "u_zero" else "random-u-%d" % row["seed"]
        by_suffix[suffix] = row
        with np.load(path, allow_pickle=False) as payload:
            array = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        if lib.hash_array(array) != str(row["coordinate_array_sha256"]):
            raise RuntimeError("046 baseline null coordinate array hash changed: %s" % path)
        out.append({"source_candidate_id": "baseline-046-real-extension-G-full-J", "reused_from": "046",
                    "null_kind": row["null_kind"], "seed": row["seed"], "path": lib.rel(path),
                    "sha256": actual, "coordinate_array_sha256": row["coordinate_array_sha256"],
                    "global_scale": row.get("global_scale"), "reuse_verified": True})
    baseline_coords = lib.load_coords_npz(lib.BASELINE_NPZ)["coordinates"]
    parity: dict[str, Any] = {}
    slices = data.chromosome_slices()
    checks = [("u-zero", None)] + [("random-u-%d" % seed, seed) for seed in lib.RANDOM_U_SEEDS[:2]]
    for suffix, seed in checks:
        if seed is None:
            variant, audit = lib.make_u_zero(baseline_coords)
        else:
            variant, audit = lib.make_random_u(baseline_coords, slices, seed)
        recalculated = lib.hash_array(variant)
        expected = str(by_suffix[suffix]["coordinate_array_sha256"])
        parity[suffix] = {"recalculated_coordinate_array_sha256": recalculated, "046_record": expected,
                          "match": bool(recalculated == expected),
                          "global_scale_mine": audit["global_scale"],
                          "global_scale_046": by_suffix[suffix].get("global_scale")}
    if not all(item["match"] for item in parity.values()):
        raise RuntimeError("baseline null reuse parity failed: %s" % parity)
    return {"records": out, "parity_spot_checks": parity,
            "policy": "046 same-candidate-SHA null files reused in place (read-only); full 17-file hash re-verified"}


def candidate_records(raw_candidates: Any, loss: str, solver: str) -> dict[str, dict[str, Any]]:
    """selection 里的 candidates 可能是 mapping(source->value/record) 或 list[record]，统一归一化。"""
    items: list[dict[str, Any]] = []
    if isinstance(raw_candidates, Mapping):
        for key, value in raw_candidates.items():
            if isinstance(value, Mapping):
                items.append({**dict(value), "source": value.get("source", key)})
            else:
                items.append({"source": key, "own_count": value})
    elif isinstance(raw_candidates, (list, tuple)):
        items = [dict(item) for item in raw_candidates if isinstance(item, Mapping)]
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        source = item.get("source")
        if source not in lib.SOURCES:
            fit_id = str(item.get("fit_id") or "")
            tail = fit_id.split("-")[-1] if fit_id else ""
            source = tail if tail in lib.SOURCES else None
        if source not in lib.SOURCES:
            continue
        declared = None
        for key in ("own_count", "count_nll", "count", "value"):
            if isinstance(item.get(key), (int, float)):
                declared = float(item[key])
                break
        out[str(source)] = {
            "declared_count": declared,
            "declared_full_j": (item.get("own_fullJ") if isinstance(item.get("own_fullJ"), (int, float))
                                else item.get("fullJ") if isinstance(item.get("fullJ"), (int, float)) else None),
            "coords_npz_sha256": item.get("coords_npz_sha256"),
            "fit_id": item.get("fit_id"),
        }
    return out


def coords_sha_verdict(declared_sha: Any, npz_sha256: Any, array_sha256: Any) -> dict[str, Any]:
    """selection 的 coordsSHA 指 npz 文件 SHA256；同时接受 coordinate array hash 并记录匹配类型。"""
    if declared_sha is None:
        return {"status": "absent", "matched": None}
    declared = str(declared_sha)
    if npz_sha256 is not None and declared == str(npz_sha256):
        return {"status": "match_npz_file_sha256", "matched": "npz_file_sha256"}
    if array_sha256 is not None and declared == str(array_sha256):
        return {"status": "match_coordinate_array_sha256", "matched": "coordinate_array_sha256"}
    return {"status": "mismatch", "matched": None, "declared": declared,
            "npz_sha256": npz_sha256, "coordinate_array_sha256": array_sha256}


def selection_checks(manifest: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    """核对 selection 只消费 endpoint count 与 terminal，且规则与 PLAN §8 一致。"""
    fits = {str(row["fit_id"]): row for row in manifest["fits"]}
    count_by_fit_loss: dict[str, dict[str, Any]] = {}
    for fit_id, row in fits.items():
        count_by_fit_loss[fit_id] = {
            "own_count": extract_own_count(row, str(row["loss"])),
            "counts_all_losses": {loss: extract_own_count(row, loss) for loss in lib.LOSES},
            "terminal": row.get("terminal"),
            "full_j": extract_full_j(row),
            "coordinate_array_sha256": row.get("coordinate_array_sha256"),
            "npz_sha256": row.get("npz_sha256"),
        }
    problems: list[str] = []
    source_checks: dict[str, Any] = {}
    for loss in lib.LOSES:
        for solver in lib.SOLVERS:
            key = "%s-%s" % (loss, solver)
            entry = selection["source_selection"][key]
            declared_candidates = candidate_records(entry.get("candidates"), loss, solver)
            candidates: dict[str, Any] = {}
            for source in lib.SOURCES:
                fit_id = "%s-%s-%s" % (loss, solver, source)
                candidates[source] = {"fit_id": fit_id, "own_count": count_by_fit_loss[fit_id]["own_count"],
                                      "terminal": count_by_fit_loss[fit_id]["terminal"],
                                      "full_j": count_by_fit_loss[fit_id]["full_j"],
                                      "declared": declared_candidates.get(source)}
            usable = [candidates[source]["own_count"] for source in lib.SOURCES]
            decision = None
            if all(value is not None for value in usable):
                order_by_count = sorted(lib.SOURCES, key=lambda src: candidates[src]["own_count"])
                best = candidates[order_by_count[0]]["own_count"]
                tied = [src for src in lib.SOURCES if abs(candidates[src]["own_count"] - best) <= lib.SELECTION_TIE_TOL]
                decision = "consensus" if "consensus" in tied else order_by_count[0]
            entry_out = {"selected_source": entry["selected_source"], "rule": entry.get("rule"),
                         "candidates": candidates, "recomputed_decision": decision,
                         "tie_tolerance": lib.SELECTION_TIE_TOL,
                         "margin": (abs(usable[0] - usable[1]) if all(v is not None for v in usable) else None)}
            if decision is None:
                problems.append("%s: could not recompute source selection (missing count_nll_by_loss)" % key)
            elif decision != entry["selected_source"]:
                problems.append("%s: selection selected %s but recomputed %s" % (key, entry["selected_source"], decision))
            for source in lib.SOURCES:
                declared = candidates[source]["declared"]
                if declared is None:
                    continue
                actual = candidates[source]["own_count"]
                if declared.get("declared_count") is not None and actual is not None and \
                        abs(declared["declared_count"] - actual) > 1e-12:
                    problems.append("%s/%s: selection candidate count %r != manifest count %r"
                                    % (key, source, declared["declared_count"], actual))
                if declared.get("declared_full_j") is not None and candidates[source]["full_j"] is not None and \
                        abs(declared["declared_full_j"] - candidates[source]["full_j"]) > 1e-12:
                    problems.append("%s/%s: selection candidate own_fullJ %r != manifest fullJ %r"
                                    % (key, source, declared["declared_full_j"], candidates[source]["full_j"]))
                npz_sha = candidates[source].get("fit_id") and count_by_fit_loss[candidates[source]["fit_id"]]["npz_sha256"]
                verdict = coords_sha_verdict(declared.get("coords_npz_sha256"), npz_sha,
                                            count_by_fit_loss[candidates[source]["fit_id"]]["coordinate_array_sha256"])
                if verdict["status"] == "mismatch":
                    problems.append("%s/%s: selection coords_npz_sha256 mismatch: %s" % (key, source, verdict))
            entry_out["terminal_states"] = {src: candidates[src]["terminal"] for src in lib.SOURCES}
            source_checks[key] = entry_out
    display_checks: dict[str, Any] = {}
    for loss in lib.LOSES:
        entries = {}
        for solver in lib.SOLVERS:
            for source in lib.SOURCES:
                fit_id = "%s-%s-%s" % (loss, solver, source)
                entries[fit_id] = {"own_count": count_by_fit_loss[fit_id]["own_count"],
                                   "terminal": count_by_fit_loss[fit_id]["terminal"],
                                   "npz_sha256": count_by_fit_loss[fit_id]["npz_sha256"],
                                   "coordinate_array_sha256": count_by_fit_loss[fit_id]["coordinate_array_sha256"]}
        declared = selection["display"][loss]["fit_id"]
        values = {fid: item["own_count"] for fid, item in entries.items()}
        recomputed = None
        tied: list[str] = []
        if all(value is not None for value in values.values()):
            best = min(values.values())
            tied = sorted(fid for fid, value in values.items() if abs(value - best) <= lib.SELECTION_TIE_TOL)
            recomputed = tied[0]
        verdict = coords_sha_verdict(selection["display"][loss].get("coords_sha256"), entries[declared]["npz_sha256"],
                                     entries[declared]["coordinate_array_sha256"])
        display_checks[loss] = {"declared_fit_id": declared, "recomputed_argmin": recomputed, "tied_fit_ids": tied,
                                "coords_sha_verdict": verdict, "candidates": entries}
        if recomputed is None:
            problems.append("loss %s: could not recompute display endpoint" % loss)
        elif declared not in tied:
            problems.append("loss %s: display endpoint %s not in argmin tie set %s" % (loss, declared, tied))
        if verdict["status"] == "mismatch":
            problems.append("loss %s: display coordsSHA mismatch: %s" % (loss, verdict))
    if problems:
        raise RuntimeError("selection consistency failure: %s" % problems)
    return {"source_selection": source_checks, "display": display_checks,
            "coords_sha_convention": ("selection coordsSHA is compared against the endpoint npz file SHA256 first and the "
                                      "coordinate array SHA256 second; the matched kind is recorded per entry"),
            "note": "selection consumes endpoint own count and terminal only; loss values are not cross-ranked"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=False, default=None)
    parser.add_argument("--selection", required=False, default=None)
    parser.add_argument("--fits-dir", default=None)
    parser.add_argument("--stages-dir", default=None)
    parser.add_argument("--nulls-dir", default=None)
    parser.add_argument("--gate-path", default=None)
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--reg-expected-path", default=None)
    parser.add_argument("--code-fixture", action="store_true",
                        help="isolated code-path check with synthetic endpoints; never a formal gate")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    if args.selfcheck:
        return selfcheck()

    if not args.manifest or not args.selection:
        print("prep requires --manifest and --selection (or --selfcheck)", file=sys.stderr)
        return 2
    fits_dir = Path(args.fits_dir) if args.fits_dir else lib.RUN / "results/fits"
    stages_dir = Path(args.stages_dir) if args.stages_dir else lib.RUN / "stages"
    null_dir = Path(args.nulls_dir) if args.nulls_dir else NULL_DIR
    gate_path = Path(args.gate_path) if args.gate_path else GATE_PATH
    index_path = Path(args.index_path) if args.index_path else INPUT_INDEX_PATH
    reg_expected_path = Path(args.reg_expected_path) if args.reg_expected_path else REG_EXPECTED_PATH

    data = lib.Aggregate()
    _AGGREGATE_CACHE["data"] = data
    audit = data.audit()
    aggregate_sha = lib.sha256_file(lib.AGGREGATE_1MB)
    if aggregate_sha != lib.AGGREGATE_1MB_SHA256:
        raise RuntimeError("045 aggregate hash mismatch: %s" % aggregate_sha)

    manifest = inputs.load_manifest(Path(args.manifest))
    selection = inputs.load_selection(Path(args.selection))
    for row in manifest["fits"]:
        row.update(verify_endpoint(row))
        evidence = verify_fit_json(str(row["fit_id"]), row, Path(fits_dir), Path(stages_dir), row)
        row["fit_json_evidence"] = evidence
        for field, source in (("wall_s", "wall_s"), ("canonical_grad", "canonical_grad")):
            values = evidence["scalar_fields"].get(source, [])
            if values and row.get(field) is None:
                row[field] = values[0]
        if not evidence["checked"]:
            row.setdefault("terminal_reporting_note", "no per-fit JSON found under results/fits or stages")
    initials = ([verify_initial(initial_name_of(row), row) for row in manifest["initials"]]
                if manifest["initials"] else
                [verify_initial(name, {"npz_sha256": lib.INITIAL_NPZ_SHA256[name]}) for name in lib.SOURCES])
    if sorted(row["initial"] for row in initials) != sorted(lib.SOURCES):
        raise RuntimeError("initial controls must be exactly consensus and random, got %s"
                           % [row["initial"] for row in initials])
    baseline = verify_baseline()
    baseline["manifest_metadata"] = {key: manifest["baseline"].get(key) for key in
                                     ("p", "q", "q_source", "full_j", "outer_fg", "count_nll_by_loss",
                                      "wall_s", "canonical_grad", "terminal")} if manifest["baseline"] else None
    checks = selection_checks(manifest, selection)

    selected_sources: list[dict[str, Any]] = []
    by_id = {str(row["fit_id"]): row for row in manifest["fits"]}
    for loss in lib.LOSES:
        for solver in lib.SOLVERS:
            key = "%s-%s" % (loss, solver)
            chosen = selection["source_selection"][key]["selected_source"]
            selected_sources.append(by_id["%s-%s-%s" % (loss, solver, chosen)])
    null_sources = [{"candidate_id": str(row["fit_id"]), "coordinates": row["coordinates"]} for row in selected_sources]
    null_sources += [{"candidate_id": "initial-%s" % row["initial"], "coordinates": row["coordinates"]} for row in initials]
    null_records = build_nulls(null_sources, data, null_dir)
    baseline_nulls = verify_baseline_null_reuse(data)
    expected_new = len(null_sources) * len(NULL_VARIANTS)
    if len(null_sources) != 8 or len(NULL_VARIANTS) != 17:
        raise RuntimeError("null design changed: %d sources x %d variants" % (len(null_sources), len(NULL_VARIANTS)))
    if len(null_records) != 136 or len(null_records) != expected_new:
        raise RuntimeError("expected exactly 136 new null files, got %d (sources x variants = %d)"
                           % (len(null_records), expected_new))
    if len(baseline_nulls["records"]) != 17:
        raise RuntimeError("expected exactly 17 reused baseline nulls, got %d" % len(baseline_nulls["records"]))
    total_null_draws = len(null_records) + len(baseline_nulls["records"])
    if total_null_draws != 153:
        raise RuntimeError("null draw total must be 136 + 17 = 153, got %d" % total_null_draws)
    null_files_on_disk = sorted(Path(null_dir).glob("*.npz"))
    if len(null_files_on_disk) != 136:
        raise RuntimeError("nulls directory holds %d npz files, expected 136" % len(null_files_on_disk))

    expected = lib.load_baseline_expected_from_046(data.chromosome_names)
    lib.write_json(reg_expected_path, expected)

    terminal_counts: dict[str, int] = {}
    for row in manifest["fits"]:
        terminal_counts[str(row["terminal"])] = terminal_counts.get(str(row["terminal"]), 0) + 1

    gate = {
        "schema": "p9016-049-evaluation-pre-reference-gate-v1",
        "code_fixture": bool(args.code_fixture),
        "status": "PASS",
        "created_utc": lib.utc_now(),
        "reference_first_opened_utc": None,
        "reference_opened": False,
        "phase_opened": False,
        "synthetic_formal_runs": 0,
        "run_id": lib.RUN.name,
        "aggregate": {"path": lib.rel(lib.AGGREGATE_1MB), "sha256": aggregate_sha, **audit},
        "manifest": {"path": lib.rel(manifest["path"]), "sha256": manifest["sha256"]},
        "selection": {"path": lib.rel(selection["path"]), "sha256": selection["sha256"]},
        "fits": [{"fit_id": str(row["fit_id"]), "loss": row["loss"], "solver": row["solver"], "source": row["source"],
                  "terminal": str(row["terminal"]), "npz_path": lib.rel(lib.resolve_path(row["npz_path"])),
                  "npz_sha256": row["npz_sha256"], "coordinate_array_sha256": row["coordinate_array_sha256"],
                  "three_dg_path": lib.rel(lib.resolve_path(row["tdg_path"])), "three_dg_sha256": row["tdg_sha256"],
                  "outer_fg": row.get("outer_fg"), "count_nll_by_loss": row.get("count_nll_by_loss"),
                  "full_j": row.get("full_j"), "p": row.get("p"), "q": row.get("q"),
                  "npz_keys": row["npz_keys"], "fit_json_evidence": row["fit_json_evidence"],
                  "readback": row.get("readback")} for row in manifest["fits"]],
        "initials": [{"initial": row["initial"], "path": lib.rel(row["path"]), "sha256": row["sha256"],
                      "coordinate_array_sha256": row["coordinate_array_sha256"], "raw_y_sha256": row["raw_y_sha256"],
                      "p_init": row["p_init"], "lineage": row["lineage"], "three_dg": row.get("three_dg"),
                     "manifest_metadata": row.get("manifest_metadata")}
                     for row in initials],
        "baseline": {"candidate_id": "baseline-046-real-extension-G-full-J",
                     "npz_path": lib.rel(baseline["npz_path"]), "npz_sha256": baseline["npz_sha256"],
                     "coordinate_array_sha256": baseline["coordinate_array_sha256"],
                     "three_dg_path": lib.rel(baseline["tdg_path"]), "three_dg_sha256": baseline["tdg_sha256"],
                     "terminal": baseline["terminal"], "manifest_metadata": baseline.get("manifest_metadata"),
                     "role": "post-hoc adopted working comparison, 1988 FG including 5Mb/2Mb stages; not cost-equal"},
        "nulls_new": null_records,
        "nulls_reused_baseline": baseline_nulls["records"],
        "null_policy": {"new_null_files": len(null_records), "reused_baseline_files": len(baseline_nulls["records"]),
                        "total_null_draws": total_null_draws, "expected_total": 153,
                        "nulls_directory_file_count": len(null_files_on_disk),
                        "kinds": ["u_zero", "random_u"], "random_u_seeds": list(lib.RANDOM_U_SEEDS),
                        "z_policy": "z=(A+B)/2 fixed per chromosome; u=(A-B)/2 permuted within the same chromosome",
                        "rescale_policy": "single whole-cell rescale only when the rebuilt cell exits the unit sphere",
                        "baseline_null_reuse_parity": baseline_nulls["parity_spot_checks"]},
        "selection_checks": checks,
        "terminal_counts": terminal_counts,
        "deviations": [{
            "item": "pre-gate read of frozen mask snapshot metadata",
            "requirement": "PLAN/task 要求 mask snapshot 只在评估侧（gate 之后）加载",
            "what_was_read_before_gate": ("046/evaluation_final/results/frozen_legacy_mask_snapshot.npz 的 keys、各数组 dtype/shape"
                                          "（chr0..chr4 的 positions/pair_i/pair_j/common 仅 shape），以及 chr{c}_positions 的"
                                          "数值（c=0,1,2：数组长度与前两个/末一个元素），用于核对 old21 数值规则的 bin 数"),
            "what_was_not_read_before_gate": ("chr*_common（参考有限性衍生的布尔位）、pair_i/pair_j 的数值、任何参考坐标或 phase 列；"
                                              "mask 本身只在 evaluate 阶段（gate 之后）由 load_real_masks 加载"),
            "impact": ("positions 由 range(3_000_000, chr_len, 1_000_000) 与染色体长度唯一确定，未参与训练、候选选择或任何指标计算；"
                       "该提前读取不改变任何评价数值，但确实偏离了“整个 mask gate 后 load”的操作要求"),
            "disclosed_by": "评价执行方主动记录，未补造额外审计",
        }],
        "budget": {
            "frozen_outer_fg_per_fit": 1502,
            "observed_outer_fg": {str(row["fit_id"]): row.get("outer_fg") for row in manifest["fits"]},
            "deviations": [str(row["fit_id"]) for row in manifest["fits"] if row.get("outer_fg") != 1502],
            "wall_seconds": {str(row["fit_id"]): row.get("wall_s") for row in manifest["fits"]},
            "canonical_gradient_max_abs": {str(row["fit_id"]): row.get("canonical_grad") for row in manifest["fits"]},
            "cost_note": "046 baseline cumulative 1988 FG include 5Mb/2Mb stages; this round is 1502 all-1Mb full-grid FG per fit",
        },
    }
    lib.write_json(gate_path, gate)

    index = {
        "schema": "p9016-049-evaluation-input-index-v1", "created_utc": gate["created_utc"],
        "reference_opened": False,
        "fits": [{"fit_id": str(row["fit_id"]), "loss": row["loss"], "solver": row["solver"], "source": row["source"],
                  "terminal": str(row["terminal"]), "npz_path": lib.rel(lib.resolve_path(row["npz_path"])),
                  "npz_sha256": row["npz_sha256"], "coordinate_array_sha256": row["coordinate_array_sha256"],
                  "three_dg_path": lib.rel(lib.resolve_path(row["tdg_path"])), "three_dg_sha256": row["tdg_sha256"],
                  "outer_fg": row.get("outer_fg"), "count_nll_by_loss": row.get("count_nll_by_loss"),
                  "full_j": row.get("full_j"), "p": row.get("p"), "q": row.get("q")} for row in manifest["fits"]],
        "initials": [{"candidate_id": "initial-%s" % row["initial"], "path": lib.rel(row["path"]),
                      "sha256": row["sha256"], "coordinate_array_sha256": row["coordinate_array_sha256"]}
                     for row in initials],
        "baseline": {"candidate_id": "baseline-046-real-extension-G-full-J",
                     "npz_path": lib.rel(baseline["npz_path"]), "npz_sha256": baseline["npz_sha256"],
                     "coordinate_array_sha256": baseline["coordinate_array_sha256"],
                     "three_dg_path": lib.rel(baseline["tdg_path"]), "three_dg_sha256": baseline["tdg_sha256"]},
        "null_sources": {"selected_endpoints": [str(row["fit_id"]) for row in selected_sources],
                         "initials": ["initial-consensus", "initial-random"],
                         "selection_rules": {key: selection["source_selection"][key]["selected_source"]
                                             for key in sorted(selection["source_selection"])},
                         "display_endpoints": {loss: selection["display"][loss]["fit_id"] for loss in lib.LOSES}},
        "gate_path": lib.rel(gate_path),
    }
    lib.write_json(index_path, index)

    print("PREP PASS")
    print("  fits verified: %d  terminals: %s" % (len(manifest["fits"]), terminal_counts))
    print("  new null files: %d  reused baseline nulls: %d" % (len(null_records), len(baseline_nulls["records"])))
    print("  selected endpoints: %s" % [str(row["fit_id"]) for row in selected_sources])
    print("  display endpoints: %s" % {loss: selection["display"][loss]["fit_id"] for loss in lib.LOSES})
    print("  gate: %s" % gate_path)
    return 0


def selfcheck() -> int:
    """纯 helper 自检：合成小数据，不读 reference/mask/正式候选，不构成科学运行。"""
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(2, lib.N_LOCI, 3)) * 0.05
    lib.assert_inside_unit_ball(coords)
    zero, audit_zero = lib.make_u_zero(coords)
    assert np.allclose(zero[0], zero[1])
    assert lib.hash_array(zero) == lib.hash_array(lib.make_u_zero(coords)[0])
    assert audit_zero["global_scale"] == 1.0 or audit_zero["global_scale"] > 0
    slices = [slice(0, 5), slice(5, 12)]
    perm, audit_perm = lib.make_random_u(coords, slices, 450500)
    z = (coords[0] + coords[1]) / 2.0
    u = (coords[0] - coords[1]) / 2.0
    rebuilt_u = ((perm[0] - perm[1]) / 2.0) / audit_perm["global_scale"]
    assert np.allclose((perm[0] + perm[1]) / 2.0 / audit_perm["global_scale"], z)
    for slc in slices:
        assert np.allclose(np.sort(rebuilt_u[slc].ravel()), np.sort(u[slc].ravel()))
    rho_ok = lib.derive_rho({"A_mat": 0.5, "A_pat": 0.2, "B_mat": 0.2, "B_pat": 0.5})
    assert rho_ok["orientation"] == "direct" and abs(rho_ok["contrast"] - 0.3) < 1e-12
    rho_tie = lib.derive_rho({"A_mat": 0.4, "A_pat": 0.3, "B_mat": 0.4, "B_pat": 0.3})
    assert rho_tie["geometry_tie"] and rho_tie["orientation"] == "unresolved_tie"
    rho_na = lib.derive_rho({"A_mat": 0.4, "A_pat": float("nan"), "B_mat": 0.3, "B_pat": 0.4})
    assert not rho_na["derived_defined"]
    import eval049_spatial as spatial
    centers = rng.normal(size=(20, 3))
    m = spatial.center_metrics(centers, centers, ["chr%d" % (i + 1) for i in range(20)])
    assert abs(m["pearson"] - 1.0) < 1e-12 and m["top3_neighbors"]["mean_overlap_top3"] == 3.0
    null = spatial.label_permutation_null(centers, centers, m["spearman"], draws=50, seed=461100)
    assert null["draws"] == 50 and 0.0 < null["one_sided_p_ge_observed"] <= 1.0
    proc = spatial.procrustes_global(centers * 2.0 + 1.0, centers, allow_reflection=False)
    assert abs(proc["rotation_det"] - 1.0) < 1e-9 and proc["normalized_aligned_rmsd"] < 1e-9
    data = lib.Aggregate()
    data.audit()
    print("SELFCHECK PASS (synthetic helpers only; not a formal run, no reference opened)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
