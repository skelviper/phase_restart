"""051 轮输入准备：粗层 aggregate + A/B 盲起点轨迹（只读复用 045 冻结实现）。

输出（全部落在本 run 的 inputs/ 与 coords/initial/）：

* ``inputs/real_<bin>_aggregate.npz``：由冻结 SNP-free 输入聚合的 full grid（20/10/5/2Mb；
  1Mb 直接复用 045 冻结文件并核对 SHA）。
* ``coords/initial/random_<stage>.npz``：A 组 20->10->5->2->1Mb 零优化 prolongation 轨迹。
* ``coords/initial/base_start_5Mb.npz``：B/C 组共用的 5Mb 起点，断言与 A 组 5Mb 逐元素一致
  且与 046 基线使用的 045 ``real_random_5Mb`` 逐元素一致。

本进程只读无标签 contacts；不打开 phase 列，也不打开 reference 3DG。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
for _path in (str(S049),):
    if _path not in sys.path:
        sys.path.insert(0, str(_path))

from frozen_imports import contact_model, data_io, reconstruction_init  # noqa: E402
from round_paths import AGGREGATE_1MB, AGGREGATE_1MB_SHA256  # noqa: E402
from round_runner import sha256_file, write_json  # noqa: E402

BIN_SIZES = (20_000_000, 10_000_000, 5_000_000, 2_000_000)
STAGES = ("20Mb", "10Mb", "5Mb", "2Mb", "1Mb")
BIN_OF_STAGE = {"20Mb": 20_000_000, "10Mb": 10_000_000, "5Mb": 5_000_000,
                "2Mb": 2_000_000, "1Mb": 1_000_000}
P_INIT = 0.75
BASE_SEED = 2207
ROOT_014 = ROOT / "test_res/014-20260912_153000-s0-genome-wide-fixed"
BASE_045_5MB = S045.parent / "coords/initial/real_random_5Mb.npz"


def configure_init() -> None:
    reconstruction_init.ROOT = str(ROOT)
    reconstruction_init.DEFAULT_GATE_PATH = str(ROOT_014 / "gate.json")
    reconstruction_init.DEFAULT_COORD_DIR = str(ROOT_014 / "coords")
    for candidate, spec in list(reconstruction_init.APPROVED_SOURCES.items()):
        patched = dict(spec)
        patched["path"] = str(ROOT_014 / "coords" / (candidate + ".3dg"))
        patched["gate_path"] = str(ROOT_014 / "gate.json")
        reconstruction_init.APPROVED_SOURCES[candidate] = patched


def array_sha(values: np.ndarray, dtype: str = "<f8") -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype=dtype)).tobytes(order="C")).hexdigest()


def build_aggregates() -> tuple[dict[int, object], dict[str, object]]:
    data_by_bin = {}
    records = {}
    for bin_size in BIN_SIZES:
        data = contact_model.load_frozen_p9016_aggregate(bin_size)
        record = data_io.save_aggregate(RUN / "inputs" / ("real_%d_aggregate.npz" % bin_size), data)
        record["budget"] = data.budget()
        write_json(RUN / "inputs" / ("real_%d_aggregate.json" % bin_size), record)
        data_by_bin[bin_size] = data
        records[str(bin_size)] = record
    frozen = data_io.load_aggregate(AGGREGATE_1MB)
    actual = sha256_file(AGGREGATE_1MB)
    if actual != AGGREGATE_1MB_SHA256:
        raise RuntimeError("045 frozen 1Mb aggregate SHA mismatch")
    data_by_bin[1_000_000] = frozen
    records["1000000"] = {"path": str(AGGREGATE_1MB.relative_to(ROOT)), "sha256": actual,
                          "reused_from": "045", "budget": frozen.budget()}
    return data_by_bin, records


def main() -> int:
    configure_init()
    data_by_bin = {}
    records = {}
    for bin_size in BIN_SIZES:
        path = RUN / "inputs" / ("real_%d_aggregate.npz" % bin_size)
        data = data_io.load_aggregate(path)
        data_by_bin[bin_size] = data
        records[str(bin_size)] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                                  "budget": data.budget(), "reused_from_disk": True}
    frozen = data_io.load_aggregate(AGGREGATE_1MB)
    actual = sha256_file(AGGREGATE_1MB)
    if actual != AGGREGATE_1MB_SHA256:
        raise RuntimeError("045 frozen 1Mb aggregate SHA mismatch")
    data_by_bin[1_000_000] = frozen
    records["1000000"] = {"path": str(AGGREGATE_1MB.relative_to(ROOT)), "sha256": actual,
                          "reused_from": "045", "budget": frozen.budget()}

    # 19-26 层：20Mb 与 10Mb 由 014 random 轨迹直接 expand 到 full grid；5Mb 起点与
    # 046 baseline 使用的 045 real_random_5Mb 逐元素相同，再零优化 prolongation 到 2/1Mb。
    def build_at(bin_size: int, target_bin: int, seed_direct: bool):
        data = data_by_bin[target_bin]
        state = reconstruction_init.initialize_approved_candidate(
            "random", tuple(data.chromosome_names), tuple(int(v) for v in data.chromosome_lengths), target_bin)
        return {"coordinates": np.asarray(state["coords"], dtype=np.float64),
                "positions": np.asarray(state["positions"], dtype=np.int64),
                "chromosome_index": np.asarray(state["chromosome_index"], dtype=np.int32),
                "metadata": state["metadata"]}

    starts = {}
    out = {}
    for stage in STAGES:
        bin_size = BIN_OF_STAGE[stage]
        data = data_by_bin[bin_size]
        if stage in ("20Mb", "10Mb", "5Mb"):
            current = build_at(bin_size, bin_size, True)
        else:
            warm = reconstruction_init.warm_start_from_layer(
                current["coordinates"], current["positions"], current["chromosome_index"],
                tuple(data.chromosome_names), tuple(int(v) for v in data.chromosome_lengths), bin_size, BASE_SEED)
            current = {"coordinates": np.asarray(warm["coords"], dtype=np.float64),
                       "positions": np.asarray(warm["positions"], dtype=np.int64),
                       "chromosome_index": np.asarray(warm["chromosome_index"], dtype=np.int32),
                       "metadata": warm["metadata"]}
        coordinates = current["coordinates"]
        contact_model.assert_inside_unit_ball(coordinates)
        raw_y = contact_model.sphere_inverse(coordinates)
        record = data_io.save_start(RUN / "coords" / "initial" / ("random_%s.npz" % stage),
                                    coordinates, raw_y, P_INIT,
                                    {"candidate": "random", "stage": stage,
                                     "prolongation_seed": BASE_SEED,
                                     "source": "014 approved blind root plus zero-optimization prolongation",
                                     "no_optimization": True,
                                     "n_loci": int(data.n_loci),
                                     "metadata": current["metadata"]})
        out[stage] = record
        starts[stage] = (coordinates, raw_y)

    # B/C 共用 5Mb 起点：与 A 组 5Mb 逐元素一致
    coords5 = starts["5Mb"][0]
    raw5 = starts["5Mb"][1]
    c045, r045, p045, meta045 = data_io.load_start(BASE_045_5MB)
    legacy_match = bool(np.array_equal(coords5, c045) and np.array_equal(raw5, r045) and float(p045) == P_INIT)
    write_json(RUN / "coords" / "initial" / "base_start_5Mb.json",
               data_io.save_start(RUN / "coords" / "initial" / "base_start_5Mb.npz", coords5, raw5, P_INIT,
                                  {"candidate": "random", "stage": "5Mb",
                                   "shared_by": ["B-no-bend", "C-reference-beads"],
                                   "matches_045_real_random_5Mb": legacy_match,
                                   "lineage": "014 approved blind root plus zero-optimization prolongation",
                                   "no_optimization": True}))
    summary = {"schema": "p9016-round051-prep-v1",
               "bin_sizes": [int(b) for b in data_by_bin],
               "aggregates": records,
               "starts": out,
               "base_start_5Mb_matches_045": legacy_match,
               "reference_opened": False, "phase_opened": False}
    if not legacy_match:
        raise RuntimeError("A-group 5Mb start does not reproduce the 045 random 5Mb start")
    write_json(RUN / "inputs" / "prep_manifest.json", summary)
    for stage in STAGES:
        d = data_by_bin[BIN_OF_STAGE[stage]]
        print(stage, "n_loci", int(d.n_loci), "n_pairs", int(d.n_pairs), "budget", d.budget(),
              "coords_sha", out[stage]["coordinate_sha256"][:16])
    print("base_start_5Mb_matches_045", legacy_match)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
