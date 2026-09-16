"""052 粗化：200kb 物理坐标 -> 原始 1Mb bin 的算术均值。

规则（config.json 冻结）：

* 每 chr、每 copy、按 key = floor(original_bp / 1_000_000) 把 200kb 的物理 xyz 直接取算术均值；
* 每个完整 1Mb bin 恰好 5 颗 200kb 珠；染色体末端部分 bin 用实际 1..5 颗；
* 不取每第 5 颗、不平均距离矩阵、不重优化、不重新居中或缩放；
* 保留真实 1Mb bp 坐标与缺口（部分 bin 只平均真实存在的珠子，不做外推）。

输出 coords/200kb-to-1Mb/coarsened1Mb.{npz,3dg} 与 200kb 端点副本 200kb.{npz,3dg}，
以及一个只记录每 bin contributing beads 数与总数守恒的小 json。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bootstrap_052 import contact_model, data_io, round_runner  # noqa: E402
from round_paths_052 import FIT_ID, ROOT, stage_bin  # noqa: E402

COARSE_BP = 1_000_000
OUT_DIR = RUN / "coords" / "200kb-to-1Mb"
MANIFEST = OUT_DIR / "coarsening_manifest.json"


def _read_full_tracks(path: Path, data) -> np.ndarray:
    """按冻结 045 的同一规则把 .3dg 读回 (2, n_loci, 3)，用于写读一致性断言。"""
    coords = np.full((2, data.n_loci, 3), np.nan, dtype=np.float64)
    by_name = {spec.name: spec for spec in data.track_specs}
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5 or fields[0] not in by_name:
                raise RuntimeError("invalid full-track row: %r" % line)
            spec = by_name[fields[0]]
            local_bin = int(fields[1]) // int(data.bin_size)
            slc = data.chromosome_slice(spec.chromosome_index)
            matches = np.flatnonzero(data.locus_bin[slc] == local_bin)
            if len(matches) != 1:
                raise RuntimeError("track row does not map to one grid locus")
            index = int(slc.start + matches[0])
            seen.add((spec.copy_index, index))
            coords[spec.copy_index, index] = np.asarray(fields[2:5], dtype=np.float64)
    if len(seen) != 2 * int(data.n_loci):
        raise RuntimeError("coarsened 3DG is incomplete")
    return coords


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="200kb 端点 npz，缺省 coords/new-chain/200kb.npz")
    args = parser.parse_args()
    source_path = Path(args.source) if args.source else (RUN / "coords" / FIT_ID / "200kb.npz")
    if not source_path.is_file():
        raise RuntimeError("missing 200kb endpoint: %s" % source_path)
    with np.load(source_path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        raw_y = np.asarray(payload["raw_y"], dtype=np.float64).copy()
        p_value = float(np.asarray(payload["p"]).item())
        q_value = float(np.asarray(payload["q"]).item())

    fine = data_io.load_aggregate(RUN / "inputs" / "real_200000_aggregate.npz")
    if int(fine.bin_size) != stage_bin("200kb"):
        raise RuntimeError("200kb layer bin_size mismatch")
    if coordinates.shape != (2, int(fine.n_loci), 3):
        raise RuntimeError("200kb endpoint shape %s does not match the 200kb grid %d"
                           % (coordinates.shape, int(fine.n_loci)))
    contact_model.assert_inside_unit_ball(coordinates)

    # 每个 coarse bin 的期望 200kb 珠数：只由各 chr 真实长度与完整 fine 网格决定。
    # 末端剩余 bp > 800kb 的 chr，其末端 1Mb bin 仍然有 5 颗；因此 partial bin 数不是 20，也不能硬编码。
    expected_beads = []
    for length in fine.chromosome_lengths:
        n_fine = (int(length) + int(fine.bin_size) - 1) // int(fine.bin_size)
        sub = np.arange(n_fine, dtype=np.int64) * int(fine.bin_size)
        expected_beads.append(np.bincount(sub // COARSE_BP,
                                          minlength=-(-int(length) // COARSE_BP)))
    n_loci_1mb = int(sum(len(x) for x in expected_beads))
    expected_total = int(sum(int(x.sum()) for x in expected_beads))
    expected_partial_bins = int(sum(int(np.count_nonzero((x > 0) & (x < 5))) for x in expected_beads))
    expected_five_bins = int(sum(int(np.count_nonzero(x == 5)) for x in expected_beads))
    if n_loci_1mb != int(sum((int(L) + COARSE_BP - 1) // COARSE_BP
                             for L in fine.chromosome_lengths)):
        raise RuntimeError("expected coarse grid size is inconsistent")
    if expected_total != int(fine.n_loci):
        raise RuntimeError("expected coarse bead total does not match the 200kb grid")
    if expected_partial_bins + expected_five_bins != n_loci_1mb:
        raise RuntimeError("expected bead-count categories do not cover the coarse grid")

    out = np.full((2, n_loci_1mb, 3), np.nan, dtype=np.float64)
    contributing = np.zeros((2, n_loci_1mb), dtype=np.int64)
    per_chromosome = []
    for ci, name in enumerate(fine.chromosome_names):
        slc = fine.chromosome_slice(ci)
        local_bin = np.asarray(fine.locus_bin[slc], dtype=np.int64)
        bp = local_bin * int(fine.bin_size)
        coarse_key = bp // COARSE_BP                      # 原基因组 1Mb 坐标的 bin key
        n_coarse = int(coarse_key.max()) + 1
        # 本染色体在 1Mb 完整 grid 里的起点（不是靠 chr%d 假设）
        offset = int(np.sum([(int(L) + COARSE_BP - 1) // COARSE_BP
                             for L in fine.chromosome_lengths[:ci]]))
        summed = np.zeros((2, n_coarse, 3), dtype=np.float64)
        counts = np.zeros((2, n_coarse), dtype=np.int64)
        for copy in (0, 1):
            np.add.at(summed[copy], coarse_key, coordinates[copy, slc])
            np.add.at(counts[copy], coarse_key, 1)
        for copy in (0, 1):
            present = counts[copy] > 0
            out[copy, offset:offset + n_coarse][present] = (
                summed[copy][present] / counts[copy][present][:, None])
            contributing[copy, offset:offset + n_coarse] = counts[copy]
            # 逐位核对本 chr 的期望珠数（只有末端允许 1..4 颗）
            if not np.array_equal(counts[copy], expected_beads[ci]):
                raise RuntimeError("coarse bead count mismatch for %s copy %d" % (name, copy))
        per_chromosome.append({
            "chromosome": str(name), "chromosome_index": ci, "global_offset": offset,
            "n_fine_bins": int(len(local_bin)), "n_coarse_bins": n_coarse,
            "expected_beads_per_coarse_bin": [int(v) for v in expected_beads[ci]],
            "bins_with_5_beads": int(np.count_nonzero(counts == 5)),
            "bins_with_partial_1_to_4": int(np.count_nonzero((counts > 0) & (counts < 5))),
            "bins_missing": int(np.count_nonzero(counts == 0)),
            "bead_count_histogram": {str(k): int(v) for k, v in
                                     zip(*np.unique(counts, return_counts=True))},
        })

    if not np.all(np.isfinite(out)):
        raise RuntimeError("coarsened 1Mb candidate has nonfinite bins")
    radii = np.linalg.norm(out, axis=2)
    if float(radii.max()) > 1.0 + 1e-12:
        raise RuntimeError("coarsened bin mean left the unit ball by %.3g" % (float(radii.max()) - 1.0))
    total_beads = int(contributing.sum())
    if total_beads != 2 * expected_total or total_beads != 2 * int(fine.n_loci):
        raise RuntimeError("coarsening lost 200kb beads: %d != %d" % (total_beads, 2 * expected_total))
    if contributing.min() < 1:
        raise RuntimeError("coarsening left an empty 1Mb bin")
    # 每颗 coarse bin 的珠数必须落在 1..5，且 partial 数与"由真实长度推得的期望值"一致
    complete = contributing == 5
    partial = (contributing > 0) & (contributing < 5)
    if int(np.count_nonzero(contributing > 5)) != 0:
        raise RuntimeError("a 1Mb bin received more than 5 sub-bins")
    if int(complete.sum()) != 2 * expected_five_bins:
        raise RuntimeError("5-bead bin count %d != expected %d" % (int(complete.sum()),
                                                                  2 * expected_five_bins))
    if int(partial.sum()) != 2 * expected_partial_bins:
        raise RuntimeError("partial bin count %d != expected %d (per chr/copy from real length)"
                           % (int(partial.sum()), 2 * expected_partial_bins))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 200kb 端点副本（新链最终细层端点）落在 coords/200kb-to-1Mb/
    fine_3dg_src = RUN / "coords" / FIT_ID / "200kb.3dg"
    fine_npz_src = RUN / "coords" / FIT_ID / "200kb.npz"
    fine_3dg = OUT_DIR / "200kb.3dg"
    fine_npz = OUT_DIR / "200kb.npz"
    for src, dst in ((fine_3dg_src, fine_3dg), (fine_npz_src, fine_npz)):
        if not src.is_file():
            raise RuntimeError("missing chain endpoint copy source: %s" % src)
        dst.write_bytes(src.read_bytes())

    # 用完整 1Mb grid 写出 .npz / .3dg（write_full_tracks 需要 header 层）
    ref_layer = data_io.load_aggregate(ROOT / "test_res/045-20260915T073310Z-shared-capture-round"
                                              "/inputs/real_1000000_aggregate.npz")
    if int(ref_layer.n_loci) != n_loci_1mb:
        raise RuntimeError("frozen 1Mb layer grid changed")
    npz_path = OUT_DIR / "coarsened1Mb.npz"
    # raw_y 必须与粗化后的 coordinates 自洽：由 sphere_inverse 得到并做往返断言（不写零占位）
    coarse_raw_y = contact_model.sphere_inverse(out)
    roundtrip_error = float(np.max(np.abs(contact_model.sphere_forward(coarse_raw_y) - out)))
    if not math.isfinite(roundtrip_error) or roundtrip_error > 1e-15:
        raise RuntimeError("coarsened raw_y roundtrip error %.3g exceeds 1e-15" % roundtrip_error)
    if not np.all(np.isfinite(coarse_raw_y)):
        raise RuntimeError("coarsened raw_y is nonfinite")
    np.savez_compressed(npz_path, coordinates=out, raw_y=coarse_raw_y, p=np.asarray(p_value),
                        q=np.asarray(q_value), coarsening="arithmetic_mean_of_200kb_sub_bins",
                        contributing_beads=contributing)
    d3g_path = OUT_DIR / "coarsened1Mb.3dg"
    contact_model.write_full_tracks(d3g_path, ref_layer, out)
    readback = _read_full_tracks(d3g_path, ref_layer)
    if not np.array_equal(readback, out):
        raise RuntimeError("coarsened 3DG write/readback changed coordinates")
    manifest = {
        "schema": "p9016-round052-coarsening-v1",
        "source_200kb_endpoint": str(source_path.relative_to(ROOT)),
        "source_200kb_endpoint_sha256": round_runner.sha256_file(source_path),
        "source_200kb_coords_sha256": round_runner.array_sha256(coordinates),
        "source_200kb_n_loci": int(fine.n_loci),
        "rule": "per chromosome, per copy, arithmetic mean of the 200kb physical xyz with key "
                "floor(original_bp/1000000); no every-5th picking, no distance averaging, "
                "no re-optimization, no re-centering, no rescaling",
        "target_grid": {"bin_size_bp": COARSE_BP, "n_loci": n_loci_1mb,
                        "grid": "frozen 1Mb header grid (full 20 chromosome grids kept)"},
        "contributing_beads_total": total_beads,
        "expected_total": 2 * expected_total,
        "expected_total_from_fine_grid": 2 * expected_total,
        "bins_with_5_beads": int(complete.sum()),
        "expected_bins_with_5_beads": 2 * expected_five_bins,
        "bins_with_partial_1_to_4": int(partial.sum()),
        "expected_bins_with_partial_1_to_4": 2 * expected_partial_bins,
        "bins_empty": int(np.count_nonzero(contributing == 0)),
        "per_chromosome_expected_beads": [int(x.sum()) for x in expected_beads],
        "raw_y_roundtrip_max_abs_error": roundtrip_error,
        "per_chromosome": per_chromosome,
        "output_npz": str(npz_path.relative_to(ROOT)),
        "output_npz_sha256": round_runner.sha256_file(npz_path),
        "output_3dg": str(d3g_path.relative_to(ROOT)),
        "output_3dg_sha256": round_runner.sha256_file(d3g_path),
        "output_coords_sha256": round_runner.array_sha256(out),
        "reference_opened": False, "phase_opened": False,
        "note": "this manifest is written before any reference 3DG read",
    }
    round_runner.write_json(MANIFEST, manifest)
    print(json.dumps({k: manifest[k] for k in (
        "contributing_beads_total", "expected_total", "bins_with_5_beads",
        "bins_with_partial_1_to_4", "bins_empty", "output_3dg_sha256",
        "output_npz_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
