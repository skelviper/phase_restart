"""C 组导出裁剪：3DG 只写真实存在的珠子（保留原 bp，不压缩间隔），并同步 stage/terminal。

045 的 3DG track 名是 ``c01a``/``c01b``（不是 chr1(mat)）。本脚本对 C 的三层 endpoint
做一次性重写：mask 为 False 的珠子行删除，NPZ 里对应坐标置为 NaN（保持 shape 与祖先
线索），stage/terminal 的 hash 与导出计数更新。A/B 不动。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
for _path in (str(HERE),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

STAGE_BIN = {"20Mb": 20_000_000, "10Mb": 10_000_000, "5Mb": 5_000_000,
             "2Mb": 2_000_000, "1Mb": 1_000_000}
FIT = "C-reference-beads"
TRACK = re.compile(r"^c(\d{2})([ab])$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    sys.path.insert(0, str(RUN.parent / "049-20260915T162917Z-max-contact-unified-multiscale/source"))
    from frozen_imports import data_io
    from round_runner import sha256_file

    summary = {}
    for stage in ("5Mb", "2Mb", "1Mb"):
        bin_size = STAGE_BIN[stage]
        support_path = RUN / "inputs" / ("reference_support_%s.npz" % stage)
        with np.load(support_path, allow_pickle=False) as payload:
            mask = np.asarray(payload["mask"], dtype=bool)
            stored_bin = int(np.asarray(payload["bin_size"]).item())
        if stored_bin != bin_size:
            raise RuntimeError("support bin_size mismatch at %s" % stage)
        data = data_io.load_aggregate(RUN / "inputs" / ("real_%d_aggregate.npz" % bin_size)) \
            if bin_size != 1_000_000 else data_io.load_aggregate(
                ROOT / "test_res/045-20260915T073310Z-shared-capture-round/inputs/"
                       "real_1000000_aggregate.npz")
        names = [str(name) for name in data.chromosome_names]
        offsets = np.asarray(data.offsets, dtype=np.int64)
        npz_path = RUN / "coords" / FIT / ("%s.npz" % stage)
        with np.load(npz_path, allow_pickle=False) as payload:
            coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
        n_loci = int(data.n_loci)
        if coordinates.shape[1] != n_loci:
            raise RuntimeError("C endpoint NPZ does not match the full grid at %s" % stage)
        d3g_path = RUN / "coords" / FIT / ("%s.3dg" % stage)
        lines = []
        kept = 0
        dropped = 0
        for chromosome_index, chromosome in enumerate(names):
            slc = data.chromosome_slice(chromosome_index)
            positions = (np.asarray(data.locus_bin[slc]) * bin_size).astype(np.int64)
            for copy in (0, 1):
                track = "c%02d%s" % (chromosome_index + 1, "ab"[copy])
                values_index = int(offsets[chromosome_index]) + positions // bin_size
                keep = mask[copy, slc] & np.isfinite(coordinates[copy, slc]).all(axis=1)
                for position, index, is_kept in zip(positions.tolist(), values_index.tolist(),
                                                    keep.tolist()):
                    if not is_kept:
                        dropped += 1
                        continue
                    xyz = coordinates[copy, index]
                    lines.append("%s\t%d\t%.17g\t%.17g\t%.17g" % (
                        track, position, xyz[0], xyz[1], xyz[2]))
                    kept += 1
        d3g_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # NPZ 保留 full grid，缺失珠子坐标写 NaN
        with np.load(npz_path, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.keys()}
        nan_coordinates = coordinates.copy()
        nan_raw_y = np.asarray(arrays["raw_y"], dtype=np.float64).copy()
        for chromosome_index in range(len(names)):
            slc = data.chromosome_slice(chromosome_index)
            for copy in (0, 1):
                missing = ~mask[copy, slc]
                nan_coordinates[copy, slc][missing] = np.nan
                nan_raw_y[copy, slc][missing] = np.nan
        arrays["coordinates"] = nan_coordinates
        arrays["raw_y"] = nan_raw_y
        np.savez_compressed(npz_path, **arrays)
        record = {
            "stage": stage, "bin_size": bin_size, "n_loci": n_loci,
            "kept_rows": kept, "dropped_rows": dropped,
            "expected_masked_beads": int(mask.sum()),
            "d3g_sha256": sha256_file(d3g_path), "npz_sha256": sha256_file(npz_path),
            "rule": "3DG exports only real beads at their original bp; NPZ keeps the full grid "
                    "with NaN at missing beads",
        }
        summary[stage] = record
        for target in (RUN / "stages" / FIT / ("%s.json" % stage),
                       RUN / "logs" / ("%s-%s.terminal.json" % (FIT, stage))):
            if not target.is_file():
                continue
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload["export_filter"] = record
            hashes = dict(payload.get("artifact_hashes") or {})
            hashes["coordinate_3dg_sha256"] = record["d3g_sha256"]
            hashes["coordinate_npz_sha256"] = record["npz_sha256"]
            hashes["coordinate_3dg_sha256_exported"] = record["d3g_sha256"]
            payload["artifact_hashes"] = hashes
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    (RUN / "logs" / "C_export_filter.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
