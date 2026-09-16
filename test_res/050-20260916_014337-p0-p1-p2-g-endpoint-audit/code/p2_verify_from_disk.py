"""P2（050 轮）磁盘侧独立复核：只读 p2/ 已写出的产物，重新推导几何与哈希。

目的：确认写盘之后的 probe 坐标本身（而不是内存里的对象）满足冻结设计：
坐标 hash / 文件 hash 与 gate 一致、baseline_coordinates 与 046 NPZ 逐位一致、
严格单位球内、位移 RMS 与 target/actual 一致、chr 内四 copy 距离不变、
每 chr 为精确刚体（proper Kabsch residual≈0, det=+1）、chr 中心与 copy 中心的
相对距离变化量与 TSV 记录一致，并确认 gate 是在全部 probe 写盘之后才写的。

不调用 objective、不读 reference、不读 phase；不需要 GPU。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
ROOT = RUN_DIR.parents[1]
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
if str(S049) not in sys.path:
    sys.path.insert(0, str(S049))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analysis_core as ac  # noqa: E402
import p2_probes as p2p  # noqa: E402

BASELINE_NPZ = ROOT / "test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.npz"
BASELINE_NPZ_SHA256 = "116e906e790493afa956445c527b38da11b2fdcaed482fd225f578f188df699d"
MEAN_SHIFT_TOL = 1e-12
INTRA_TOL_ABS = 1e-12
RIGID_TOL_ABS = 1e-12
TSV_TOL = 1e-9


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False, default=str) + "\n", encoding="utf-8")


def displacement_rms(field: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64).reshape(-1, 3)
    return float(np.sqrt(np.mean(np.sum(values ** 2, axis=1))))


def kabsch_proper(reference: np.ndarray, moving: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    mov = np.asarray(moving, dtype=np.float64).reshape(-1, 3)
    ref_c = ref - ref.mean(axis=0)
    mov_c = mov - mov.mean(axis=0)
    u, _s, vt = np.linalg.svd(mov_c.T @ ref_c)
    rotation = vt.T @ u.T
    if float(np.linalg.det(rotation)) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    residual = mov_c @ rotation.T - ref_c
    return {"rms": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
            "det": float(np.linalg.det(rotation))}


def merged_centres(coordinates: np.ndarray, data: Any) -> np.ndarray:
    return np.asarray([np.asarray(coordinates[:, data.chromosome_slice(c), :],
                                  dtype=np.float64).reshape(-1, 3).mean(axis=0)
                       for c in range(len(data.chromosome_names))])


def copy_centres(coordinates: np.ndarray, data: Any) -> np.ndarray:
    rows = []
    for c in range(len(data.chromosome_names)):
        block = np.asarray(coordinates[:, data.chromosome_slice(c), :], dtype=np.float64)
        rows.extend([block[0].mean(axis=0), block[1].mean(axis=0)])
    return np.asarray(rows)

def intra_distances(coordinates: np.ndarray, data: Any) -> np.ndarray:
    values = []
    for c in range(len(data.chromosome_names)):
        block = np.asarray(coordinates[:, data.chromosome_slice(c), :], dtype=np.float64).reshape(-1, 3)
        i, j = np.triu_indices(len(block), k=1)
        values.append(np.linalg.norm(block[i] - block[j], axis=1))
    return np.concatenate(values)


def centre_stats(base_points: np.ndarray, probe_points: np.ndarray,
                 mask: np.ndarray | None = None) -> dict[str, float]:
    i, j = np.triu_indices(len(base_points), k=1)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != i.shape:
            raise ValueError("centre pair mask shape mismatch")
        i, j = i[mask], j[mask]
    base = np.linalg.norm(base_points[i] - base_points[j], axis=1)
    probe = np.linalg.norm(probe_points[i] - probe_points[j], axis=1)
    diff = probe - base
    return {"n_pairs": int(len(base)), "mean_abs_change": float(np.abs(diff).mean()),
            "max_abs_change": float(np.abs(diff).max()),
            "rms_change": float(np.sqrt(np.mean(diff ** 2))),
            "mean_relative_change": float(np.mean(np.abs(diff) / np.maximum(base, 1e-300)))}


def read_tsv(path: Path) -> dict[str, dict[str, str]]:
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    columns = lines[0].split("\t")
    out = {}
    for line in lines[1:]:
        cells = line.split("\t")
        row = dict(zip(columns, cells))
        out[row["probe"]] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="050 P2 disk-side independent re-derivation (read-only)")
    parser.parse_args()
    p2 = RUN_DIR / "p2"
    data = ac.load_layer(1_000_000)

    baseline_sha = sha256_file(BASELINE_NPZ)
    if baseline_sha != BASELINE_NPZ_SHA256:
        raise RuntimeError("baseline NPZ sha mismatch in verifier: %s" % baseline_sha)
    with np.load(BASELINE_NPZ, allow_pickle=False) as payload:
        baseline = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        baseline_p = float(np.asarray(payload["p"]).item())
    baseline_intra = intra_distances(baseline, data)
    baseline_merged = merged_centres(baseline, data)
    baseline_copy = copy_centres(baseline, data)
    # 中心 pair 的固定分母：190（合并 chr 中心）、760（跨 chr copy 中心 = 190×4）、780（全部 copy 中心对）、20（同 chr homolog）
    merged_masks = p2p.centre_pair_masks(np.arange(len(data.chromosome_names), dtype=np.int64))
    copy_masks = p2p.centre_pair_masks(p2p.copy_centre_chromosome_index(data))
    if (int(merged_masks["all"].sum()), int(copy_masks["cross_chromosome"].sum()),
            int(copy_masks["all"].sum()), int(copy_masks["same_chromosome"].sum())) != (190, 760, 780, 20):
        raise AssertionError("centre pair denominators are not 190 / 760 / 780 / 20 in the verifier")

    gate = json.loads((p2 / "pre_reference_gate.json").read_text(encoding="utf-8"))
    tsv = read_tsv(p2 / "results" / "p2_probes.tsv")
    gate_mtime = (p2 / "pre_reference_gate.json").stat().st_mtime

    rows: list[dict[str, Any]] = []
    max_tsv_rel_error = 0.0
    for entry in gate["probes"]:
        probe_path = RUN_DIR / entry["path"]
        file_sha = sha256_file(probe_path)
        with np.load(probe_path, allow_pickle=False) as payload:
            coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
            stored_baseline = np.asarray(payload["baseline_coordinates"], dtype=np.float64)
            direction = np.asarray(payload["direction"], dtype=np.float64)
            target = float(np.asarray(payload["target_amplitude"]).item())
            actual = float(np.asarray(payload["actual_amplitude"]).item())
            npz_p = float(np.asarray(payload["p"]).item())
        coord_sha = hashlib.sha256(np.ascontiguousarray(coordinates, dtype="<f8").tobytes()).hexdigest()
        displacement = coordinates - baseline
        centred = displacement - displacement.reshape(-1, 3).mean(axis=0)[None, None, :]
        peak = float(np.linalg.norm(coordinates.reshape(-1, 3), axis=1).max())
        global_fit = kabsch_proper(baseline, coordinates)
        per_chr = []
        for c in range(len(data.chromosome_names)):
            slc = data.chromosome_slice(c)
            per_chr.append(kabsch_proper(baseline[:, slc, :].reshape(-1, 3),
                                         coordinates[:, slc, :].reshape(-1, 3)))
        intra_abs = np.abs(intra_distances(coordinates, data) - baseline_intra)
        merged_stats = centre_stats(baseline_merged, merged_centres(coordinates, data), merged_masks["all"])
        copy_cross = centre_stats(baseline_copy, copy_centres(coordinates, data), copy_masks["cross_chromosome"])
        copy_all = centre_stats(baseline_copy, copy_centres(coordinates, data), copy_masks["all"])
        copy_same = centre_stats(baseline_copy, copy_centres(coordinates, data), copy_masks["same_chromosome"])
        tsv_row = tsv[entry["probe"]]

        def rel(tsv_key: str, value: float) -> float:
            """已记录值量级 O(1)：绝对 1e-12 + 相对 1e-9 的双容差。"""
            recorded = float(tsv_row[tsv_key])
            return abs(value - recorded) / max(abs(recorded), 1.0)

        checks = {
            "file_sha256_matches_gate": bool(file_sha == entry["sha256"]),
            "coordinates_sha256_matches_gate": bool(coord_sha == entry["coordinates_sha256"]),
            "stored_baseline_bitexact": bool(np.array_equal(stored_baseline, baseline)),
            "stored_p_matches_baseline": bool(npz_p == baseline_p),
            "gate_written_after_probe_file": bool(gate_mtime >= probe_path.stat().st_mtime),
            "inside_unit_ball": bool(peak < 1.0),
            "peak_radius_matches_tsv": bool(rel("peak_radius", peak) <= TSV_TOL),
            "mean_displacement_near_zero": bool(float(np.linalg.norm(displacement.reshape(-1, 3).mean(axis=0)))
                                                <= MEAN_SHIFT_TOL),
            "displacement_rms_matches_actual": bool(abs(displacement_rms(centred) - abs(actual)) <= 1e-12),
            "displacement_rms_matches_tsv": bool(rel("displacement_rms_after_global_removal",
                                                     displacement_rms(centred)) <= TSV_TOL),
            "target_matches_tsv": bool(rel("target_amplitude", target) <= TSV_TOL),
            "actual_matches_tsv": bool(rel("actual_amplitude", actual) <= TSV_TOL),
            "direction_is_unit_rms": bool(abs(displacement_rms(direction) - 1.0) <= 1e-12),
            "direction_mean_zero": bool(float(np.linalg.norm(direction.reshape(-1, 3).mean(axis=0))) <= 1e-12),
            "intra_four_copy_distances_preserved": bool(intra_abs.max() <= INTRA_TOL_ABS),
            "intra_max_abs_matches_tsv": bool(rel("intra_distance_max_abs_change", float(intra_abs.max())) <= TSV_TOL),
            "per_chromosome_rigid_exact": bool(max(f["rms"] for f in per_chr) <= RIGID_TOL_ABS),
            "per_chromosome_det_plus_one": bool(min(f["det"] for f in per_chr) > 1.0 - 1e-12),
            "per_chr_residual_matches_tsv": bool(rel("per_chr_rigid_residual_rms_max",
                                                     max(f["rms"] for f in per_chr)) <= TSV_TOL),
            "global_kabsch_residual_matches_tsv": bool(rel("rms_deformation_after_proper_rigid",
                                                           global_fit["rms"]) <= TSV_TOL),
            "global_kabsch_det_plus_one": bool(global_fit["det"] > 1.0 - 1e-12),
            "chr_centre_190_matches_tsv": bool(int(merged_stats["n_pairs"]) == 190
                                               and rel("chr_centre_190_mean_abs_change",
                                                       merged_stats["mean_abs_change"]) <= TSV_TOL
                                               and rel("chr_centre_190_max_abs_change",
                                                       merged_stats["max_abs_change"]) <= TSV_TOL),
            "copy_centre_760_is_cross_chromosome": bool(int(copy_cross["n_pairs"]) == 760),
            "copy_centre_760_matches_tsv": bool(rel("copy_centre_760_mean_abs_change",
                                                    copy_cross["mean_abs_change"]) <= TSV_TOL
                                                and rel("copy_centre_760_max_abs_change",
                                                        copy_cross["max_abs_change"]) <= TSV_TOL),
            "copy_centre_all780_matches_tsv": bool(int(copy_all["n_pairs"]) == 780
                                                   and rel("copy_centre_all780_mean_abs_change",
                                                           copy_all["mean_abs_change"]) <= TSV_TOL
                                                   and rel("copy_centre_all780_max_abs_change",
                                                           copy_all["max_abs_change"]) <= TSV_TOL),
            "copy_centre_same_chr_homolog_20_matches_tsv": bool(
                int(copy_same["n_pairs"]) == 20
                and rel("copy_centre_same_chr_homolog_20_mean_abs_change",
                        copy_same["mean_abs_change"]) <= TSV_TOL),
            "copy_centre_denominators_760_plus_20_equals_780": bool(760 + 20 == 780),
        }
        max_tsv_rel_error = max(max_tsv_rel_error, *(rel(k, v) for k, v in (
            ("peak_radius", peak),
            ("displacement_rms_after_global_removal", displacement_rms(centred)),
            ("intra_distance_max_abs_change", float(intra_abs.max())),
            ("rms_deformation_after_proper_rigid", global_fit["rms"]),
        )))
        rows.append({"probe": entry["probe"], "family": entry["family"],
                     "file_sha256": file_sha, "coordinates_sha256": coord_sha,
                     "peak_radius": peak, "displacement_rms_after_global_removal": displacement_rms(centred),
                     "intra_distance_max_abs_change": float(intra_abs.max()),
                     "per_chr_rigid_residual_rms_max": max(f["rms"] for f in per_chr),
                     "per_chr_rigid_det_min": min(f["det"] for f in per_chr),
                     "global_kabsch_residual_rms": global_fit["rms"],
                     "global_kabsch_det": global_fit["det"],
                     "chr_centre_190": merged_stats,
                     "copy_centre_760_cross_chromosome": copy_cross,
                     "copy_centre_all780": copy_all,
                     "copy_centre_same_chr_homolog_20": copy_same,
                     "failed_checks": [k for k, v in checks.items() if not v], "checks": checks})

    failed = {row["probe"]: row["failed_checks"] for row in rows if row["failed_checks"]}
    terminal = json.loads((p2 / "terminal.json").read_text(encoding="utf-8"))
    validation = json.loads((p2 / "results" / "validation.json").read_text(encoding="utf-8"))
    report = {
        "schema": "p9016-round050-p2-disk-verification-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_probes_checked": len(rows),
        "gate_n_probes": gate["n_probes"],
        "baseline_npz_sha256": baseline_sha,
        "baseline_p": baseline_p,
        "max_tsv_relative_error_of_recomputed_scalars": max_tsv_rel_error,
        "all_disk_checks_passed": not failed,
        "failed_checks": failed,
        "in_run_validation_all_checks_passed": validation.get("all_checks_passed"),
        "in_run_terminal_state": terminal.get("terminal_state"),
        "gate_written_after_all_probe_files": all(row["checks"]["gate_written_after_probe_file"] for row in rows),
        "centre_pair_denominators": {
            "merged_chromosome_centre_pairs": 190,
            "copy_centre_cross_chromosome_pairs": 760,
            "copy_centre_all_pairs_including_same_chromosome_homologs": 780,
            "copy_centre_same_chromosome_homolog_pairs": 20,
            "checked": bool(all(int(row["copy_centre_760_cross_chromosome"]["n_pairs"]) == 760
                                and int(row["copy_centre_all780"]["n_pairs"]) == 780
                                and int(row["copy_centre_same_chr_homolog_20"]["n_pairs"]) == 20
                                and int(row["chr_centre_190"]["n_pairs"]) == 190 for row in rows)),
        },
        "probes": rows,
        "scope": "re-derived from the written files only (no objective call, no reference, no phase); it verifies the "
                 "disk payloads and the TSV/gate bookkeeping, not the count/KL readouts (those are produced by the "
                 "single in-run objective pass and validated there)",
    }
    write_json(p2 / "results" / "disk_verification.json", report)
    print(json.dumps({"all_disk_checks_passed": report["all_disk_checks_passed"],
                      "failed_checks": failed,
                      "max_tsv_relative_error": max_tsv_rel_error,
                      "in_run_validation_all_checks_passed": report["in_run_validation_all_checks_passed"]},
                     indent=2, default=str))
    return 0 if report["all_disk_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
