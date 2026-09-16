#!/usr/bin/env python3
"""只导出下一步 R2 端点所需的 synthetic 生成 exposure。

本准备辅助程序刻意只从每个 026 truth 归档读取 ``exposure`` 成员。它不会加载 ``coordinates`` 成员，也不会调用 optimizer 或 native engine。面向 worker 的 manifest 不包含 truth 路径；evaluator 溯源另行写出。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from zipfile import ZipFile

import numpy as np


PREP_DIR = Path(__file__).resolve().parent
ROOT = PREP_DIR.parents[2]
SOURCE_DIR = ROOT / "test_res" / "026-20260913_221709-allele-calibration-prepare"
TRUTH_MANIFEST_PATH = SOURCE_DIR / "provenance" / "truth_manifest.json"
N_LOCI = 2645
FIXTURES = ("P2", "N2")


class ExposurePreparationError(RuntimeError):
    """generation exposure 违反冻结输入契约时抛出。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExposurePreparationError("JSON root is not an object: %s" % path)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_exposure_only(path: Path) -> tuple[np.ndarray, list[str]]:
    """返回 exposure 和 archive member 名称，不读取 coordinates.npy。"""
    with ZipFile(path) as archive:
        members = [info.filename for info in archive.infolist()]
    expected_members = {"coordinates.npy", "exposure.npy"}
    if set(members) != expected_members:
        raise ExposurePreparationError(
            "unexpected truth archive members for %s: %s" % (path, members)
        )
    # 选择该 key 是 preparation 中唯一的 NPZ payload 访问。
    with np.load(path, allow_pickle=False) as payload:
        if "exposure" not in payload.files:
            raise ExposurePreparationError("truth archive lacks exposure key: %s" % path)
        exposure = np.asarray(payload["exposure"], dtype="<f8").copy()
    return exposure, members


def _validate_vector(
    fixture_id: str,
    exposure: np.ndarray,
    metadata: Mapping[str, Any],
    expected_exposure_sha: str | None,
) -> dict[str, Any]:
    if exposure.shape != (N_LOCI,):
        raise ExposurePreparationError("%s exposure shape is %s, expected (%d,)" % (fixture_id, exposure.shape, N_LOCI))
    if exposure.dtype != np.dtype("<f8"):
        raise ExposurePreparationError("%s exposure dtype is not little-endian float64" % fixture_id)
    if not np.all(np.isfinite(exposure)) or not np.all(exposure > 0.0):
        raise ExposurePreparationError("%s exposure is not finite and strictly positive" % fixture_id)
    mean = float(exposure.mean())
    minimum = float(exposure.min())
    maximum = float(exposure.max())
    if not np.isclose(mean, 1.0, rtol=0.0, atol=1e-12):
        raise ExposurePreparationError("%s exposure mean is not one: %.17g" % (fixture_id, mean))
    generation = metadata.get("generation_exposure")
    if not isinstance(generation, Mapping):
        raise ExposurePreparationError("%s metadata lacks generation_exposure" % fixture_id)
    checks = {
        "mean": (mean, float(generation["mean"])),
        "min": (minimum, float(generation["min"])),
        "max": (maximum, float(generation["max"])),
    }
    for label, (actual, expected) in checks.items():
        if not np.isclose(actual, expected, rtol=0.0, atol=2e-15):
            raise ExposurePreparationError(
                "%s exposure %s differs from generation metadata: %.17g != %.17g" %
                (fixture_id, label, actual, expected)
            )
    array_sha = _array_sha256(exposure)
    if expected_exposure_sha is not None and array_sha != expected_exposure_sha:
        raise ExposurePreparationError(
            "%s exposure SHA differs from the frozen truth manifest: %s != %s" %
            (fixture_id, array_sha, expected_exposure_sha)
        )
    return {
        "shape": list(exposure.shape),
        "dtype": "<f8",
        "length": int(exposure.size),
        "finite": True,
        "strictly_positive": True,
        "mean": mean,
        "min": minimum,
        "max": maximum,
        "mean_one_atol": 1e-12,
        "array_sha256": array_sha,
        "matches_generation_metadata": True,
        "matches_frozen_truth_manifest_exposure_sha256": expected_exposure_sha is None or array_sha == expected_exposure_sha,
    }


def _verify_roundtrip(path: Path, expected: np.ndarray, key: str) -> None:
    with np.load(path, allow_pickle=False) as payload:
        if payload.files != [key]:
            raise ExposurePreparationError("%s has unexpected keys: %s" % (path, payload.files))
        actual = np.asarray(payload[key], dtype="<f8")
    if not np.array_equal(actual, expected):
        raise ExposurePreparationError("roundtrip changed exposure: %s" % path)


def prepare_inputs(output_dir: str | Path = PREP_DIR / "synthetic_inputs") -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_manifest = _read_json(TRUTH_MANIFEST_PATH)
    truth_records = truth_manifest.get("fixtures", {})
    if not isinstance(truth_records, Mapping):
        raise ExposurePreparationError("truth manifest lacks fixture records")

    records: dict[str, Any] = {}
    worker_arms: dict[str, Any] = {}
    evaluator_provenance: dict[str, Any] = {}
    for fixture_id in FIXTURES:
        truth_npz = SOURCE_DIR / "eval_truth" / (fixture_id + "_truth_1mb.npz")
        metadata_path = Path(str(truth_npz) + ".json")
        counts_path = SOURCE_DIR / "work" / (fixture_id + "_counts_snapshot.npz")
        start_path = SOURCE_DIR / "work" / (fixture_id + "_shared_start.npz")
        production_layer = SOURCE_DIR / "work" / (fixture_id + "_C0_1mb_layer.npz")
        for required in (truth_npz, metadata_path, counts_path, start_path, production_layer):
            if not required.is_file():
                raise ExposurePreparationError("missing 026 source artifact: %s" % required)
        truth_record = truth_records.get(fixture_id)
        if not isinstance(truth_record, Mapping):
            raise ExposurePreparationError("truth manifest lacks fixture: %s" % fixture_id)
        metadata = _read_json(metadata_path)
        exposure, archive_members = _load_exposure_only(truth_npz)
        audit = _validate_vector(
            fixture_id,
            exposure,
            metadata,
            str(truth_record.get("generation_exposure_sha256")) if truth_record.get("generation_exposure_sha256") else None,
        )

        npy_path = output_dir / (fixture_id + "_generation_exposure.npy")
        npz_path = output_dir / (fixture_id + "_generation_exposure.npz")
        json_path = output_dir / (fixture_id + "_generation_exposure.json")
        np.save(npy_path, exposure)
        np.savez_compressed(npz_path, exposure=exposure)
        json_input = {
            "schema_version": "p9016-next-step-synthetic-exposure-input-v1",
            "fixture_id": fixture_id,
            "input_role": "generation_exposure_for_known_e_only",
            "contains_coordinates": False,
            "contains_truth_coordinates": False,
            "key": "exposure",
            "shape": list(exposure.shape),
            "dtype": "<f8",
            "exposure": exposure.tolist(),
            "audit": audit,
        }
        _write_json(json_path, json_input)
        npy_roundtrip = np.asarray(np.load(npy_path, allow_pickle=False), dtype="<f8")
        if not np.array_equal(npy_roundtrip, exposure):
            raise ExposurePreparationError("npy roundtrip changed exposure: %s" % npy_path)
        _verify_roundtrip(npz_path, exposure, "exposure")
        json_roundtrip = np.asarray(_read_json(json_path)["exposure"], dtype="<f8")
        if not np.array_equal(json_roundtrip, exposure):
            raise ExposurePreparationError("JSON roundtrip changed exposure: %s" % json_path)

        file_records = {
            "npy": {"path": _relative(npy_path), "sha256": _sha256_file(npy_path)},
            "npz": {"path": _relative(npz_path), "sha256": _sha256_file(npz_path)},
            "json": {"path": _relative(json_path), "sha256": _sha256_file(json_path)},
        }
        records[fixture_id] = {
            "fixture_id": fixture_id,
            "exposure_label": str(metadata.get("generation_exposure", {}).get("sigma")),
            "source": {
                "truth_npz_path": _relative(truth_npz),
                "truth_npz_sha256_declared": truth_record.get("npz_sha256"),
                "metadata_path": _relative(metadata_path),
                "metadata_sha256": _sha256_file(metadata_path),
                "archive_members_listed_without_payload_read": archive_members,
                "read_npz_key": "exposure",
                "coordinates_payload_read": False,
            },
            "audit": audit,
            "outputs": file_records,
            "output_array_sha256": audit["array_sha256"],
        }
        # 这是提供给 training child 的唯一 manifest。它不含
        # eval_truth path 或 coordinate/truth payload。
        worker_arms[fixture_id] = {
            "fixture_id": fixture_id,
            "known_e_exposure": file_records["npz"],
            "known_e_exposure_npy": file_records["npy"],
            "known_e_exposure_json": file_records["json"],
            "count_snapshot": {"path": _relative(counts_path), "sha256": _sha256_file(counts_path)},
            "shared_start": {"path": _relative(start_path), "sha256": _sha256_file(start_path)},
            "production_e_layer": {"path": _relative(production_layer), "sha256": _sha256_file(production_layer)},
            "cohort": {
                "sample_id": "synthetic_%s" % fixture_id,
                "n_loci_1mb": N_LOCI,
                "tracks": 40,
                "physical_beads": 5290,
                "bin_size_bp": 1000000,
                "count_snapshot_is_unlabeled": True,
            },
            "coordinate_or_truth_payload_in_manifest": False,
        }
        evaluator_provenance[fixture_id] = {
            "fixture_id": fixture_id,
            "truth_npz_path": _relative(truth_npz),
            "truth_npz_sha256_declared": truth_record.get("npz_sha256"),
            "truth_metadata_path": _relative(metadata_path),
            "truth_metadata_sha256": _sha256_file(metadata_path),
            "truth_coordinate_payload_opened": False,
            "generation_exposure_key_read": "exposure",
            "generation_exposure_array_sha256": audit["array_sha256"],
        }

    worker_manifest = {
        "schema_version": "p9016-next-step-known-e-worker-input-v1",
        "status": "ready_for_parent_training_release",
        "scope": "P2/N2 only; M0/known-e arms; no training executed here",
        "optimizer_started": False,
        "native_called": False,
        "truth_coordinates_exposed": False,
        "fixtures": worker_arms,
        "arm_plan": [
            {"fixture_id": fixture, "arm_id": fixture + "-M0-known-e", "model_id": "M0", "exposure_policy": "known-e"}
            for fixture in FIXTURES
        ],
        "production_reuse_plan": [
            {"fixture_id": fixture, "arm_ids": [fixture + "-M0-production-e", fixture + "-M1-production-e"],
             "layer_reuse": "production_e_layer", "exposure_policy": "production-e"}
            for fixture in FIXTURES
        ],
    }
    worker_manifest_path = output_dir / "known_e_worker_input_manifest.json"
    _write_json(worker_manifest_path, worker_manifest)

    provenance = {
        "schema_version": "p9016-next-step-exposure-provenance-v1",
        "status": "prepared_only",
        "source_run": _relative(SOURCE_DIR),
        "truth_manifest_path": _relative(TRUTH_MANIFEST_PATH),
        "truth_manifest_sha256": _sha256_file(TRUTH_MANIFEST_PATH),
        "fixtures": evaluator_provenance,
        "coordinates_payload_read": False,
        "reference_payload_read": False,
        "real_candidate_payload_read": False,
        "optimizer_started": False,
        "native_called": False,
    }
    provenance_path = output_dir / "exposure_provenance.json"
    _write_json(provenance_path, provenance)

    manifest = {
        "schema_version": "p9016-next-step-exposure-manifest-v1",
        "status": "prepared_only",
        "source_scope": "026 P2/N2 generation exposure only",
        "n_loci": N_LOCI,
        "fixtures": records,
        "worker_manifest": _relative(worker_manifest_path),
        "worker_manifest_sha256": _sha256_file(worker_manifest_path),
        "evaluator_provenance": _relative(provenance_path),
        "evaluator_provenance_sha256": _sha256_file(provenance_path),
        "coordinates_payload_read": False,
        "reference_payload_read": False,
        "optimizer_started": False,
        "native_called": False,
    }
    manifest_path = output_dir / "exposure_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(PREP_DIR / "synthetic_inputs"))
    args = parser.parse_args()
    try:
        result = prepare_inputs(args.output_dir)
    except (ExposurePreparationError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "status": result["status"],
        "manifest": str(Path(args.output_dir).resolve() / "exposure_manifest.json"),
        "fixtures": list(result["fixtures"]),
        "coordinates_payload_read": result["coordinates_payload_read"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
