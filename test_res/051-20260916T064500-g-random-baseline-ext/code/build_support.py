"""C 组 reference-beads support 构建（唯一允许接触 reference 的进程）。

只输出布尔 mask 与位置/计数审计：每层每 copy 的有效 locus 集合。reference 的 xyz
数值、phase 列都不写出、不进入训练进程。support 只作珠子 mask 使用，不参与初始化、
不参与目标、不参与正则。

映射规则（用户冻结）：

* 1Mb 层：保留 reference 中该 copy 存在且 xyz 有限的 locus。
* 更粗层：1Mb 有效 locus 按数值 bin 归并（数值 position -> 基因组 bin，不用压缩
  下标），粗 bin 中至少含一个有效 1Mb locus 即视为存在。先用完整数据 coarsen 再按
  该层 presence 筛 contacts。
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
for _path in (str(S049),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from frozen_imports import data_io  # noqa: E402
from round_runner import sha256_file, write_json  # noqa: E402

REFERENCE = ROOT / "data/P9016.1m.3dg.gz"
REFERENCE_SHA = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
STAGE_BIN = {"20Mb": 20_000_000, "10Mb": 10_000_000, "5Mb": 5_000_000,
             "2Mb": 2_000_000, "1Mb": 1_000_000}


def load_reference_beads() -> tuple[list[str], dict[tuple[str, int], list[int]]]:
    actual = sha256_file(REFERENCE)
    if actual != REFERENCE_SHA:
        raise RuntimeError("reference 3DG SHA256 mismatch")
    beads: dict[tuple[str, int], list[int]] = {}
    names: list[str] = []
    with gzip.open(REFERENCE, "rt") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 5:
                continue
            match = re.match(r"^(.+)\((mat|pat)\)$", fields[0])
            if match is None:
                raise RuntimeError("unexpected reference track name %r" % fields[0])
            chromosome = match.group(1)
            copy = 0 if match.group(2) == "mat" else 1
            if chromosome not in names:
                names.append(chromosome)
            position = int(fields[1])
            xyz = np.asarray([float(value) for value in fields[2:5]], dtype=np.float64)
            if np.all(np.isfinite(xyz)):
                beads.setdefault((chromosome, copy), []).append(position)
    return names, beads


def main() -> int:
    names, beads = load_reference_beads()
    layers = {}
    audit: dict[str, object] = {"schema": "p9016-round051-reference-support-v1",
                                "reference_sha256": REFERENCE_SHA,
                                "reference_path": str(REFERENCE.relative_to(ROOT)),
                                "stage_bin_sizes": {k: int(v) for k, v in STAGE_BIN.items()},
                                "layers": {}}
    previous = None
    for stage, bin_size in STAGE_BIN.items():
        data = data_io.load_aggregate(RUN / "inputs" / ("real_%d_aggregate.npz" % bin_size)) \
            if bin_size != 1_000_000 else data_io.load_aggregate(
                ROOT / "test_res/045-20260915T073310Z-shared-capture-round/inputs/"
                       "real_1000000_aggregate.npz")
        n_loci = int(data.n_loci)
        mask = np.zeros((2, n_loci), dtype=bool)
        for chromosome_index, chromosome in enumerate(data.chromosome_names):
            slc = data.chromosome_slice(chromosome_index)
            positions = (np.asarray(data.locus_bin[slc]) * bin_size).astype(np.int64)
            for copy in (0, 1):
                source = beads.get((str(chromosome), copy))
                if not source:
                    continue
                source_bins = {(int(position) // bin_size) for position in source}
                mask[copy, slc] = np.asarray([int(index) in source_bins
                                              for index in data.locus_bin[slc]], dtype=bool)
        layer = {"bin_size": int(bin_size), "n_loci": n_loci, "mask": mask}
        stage_record = {
            "bin_size": int(bin_size), "n_loci": n_loci,
            "n_valid_loci_copyA": int(mask[0].sum()), "n_valid_loci_copyB": int(mask[1].sum()),
            "n_shared_loci": int(np.count_nonzero(mask[0] & mask[1])),
            "n_missing_loci_copyA": int(n_loci - mask[0].sum()),
            "n_missing_loci_copyB": int(n_loci - mask[1].sum()),
            "coordinates_sha256_of_mask": data_io.array_sha256(mask.astype(np.int8)),
        }
        audit["layers"][stage] = stage_record
        layers[stage] = layer
        np.savez_compressed(RUN / "inputs" / ("reference_support_%s.npz" % stage),
                            mask=mask.astype(np.int8), bin_size=np.asarray(bin_size, dtype=np.int64),
                            n_loci=np.asarray(n_loci, dtype=np.int64))
        previous = layer

    # 1Mb 层与老 21-mask 的关系（只读对照，不改变分母）
    legacy_path = ROOT / ("test_res/046-UTC-real-cell-shared-capture/evaluation_final/results/"
                          "frozen_legacy_mask_snapshot.npz")
    if legacy_path.is_file():
        with np.load(legacy_path, allow_pickle=False) as payload:
            audit["legacy_mask_keys"] = list(payload.keys())
            audit["legacy_mask_path"] = str(legacy_path.relative_to(ROOT))
    write_json(RUN / "inputs" / "reference_support_manifest.json", audit)
    print(json.dumps(audit["layers"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
