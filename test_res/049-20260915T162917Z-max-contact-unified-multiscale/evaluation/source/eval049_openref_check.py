#!/usr/bin/env python
"""049 reference 打开路径的隔离代码自检（不读真实 reference、不动真实 gate）。

覆盖父侧指出的两个风险：
1. 3DG 命名：reference 用 ``chr1(mat)/chr1(pat)``，candidate 用 ``c01a/c01b``；
   用同一条人造坐标分别写两种文本验证解析一致，并验证用错模式会硬失败而不是静默全 NaN。
2. 首次打开证据：``open_reference`` 必须在真正读取前把带 UTC 的记录落盘；
   读取失败时记录与 gate 都要留下 ``failed`` 状态。
所有路径都通过参数覆盖到 ``evaluation/codecheck/openref/``，真实 gate 与 reference 不被触碰。
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval049_evaluate as ev  # noqa: E402
import eval049_lib as lib  # noqa: E402

OUT = lib.EVAL / "codecheck/openref"


def write_3dg(path: Path, coords: np.ndarray, data: lib.Aggregate, mode: str, *, gz: bool = False) -> None:
    lines = []
    for ci, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(ci)
        positions = np.asarray(data.locus_bin[slc], dtype=np.int64) * int(data.bin_size)
        for copy in (0, 1):
            track = lib.track_name(ci, str(name), copy, mode)
            for local, position in enumerate(positions):
                point = coords[copy, slc.start + local]
                lines.append("%s\t%d\t%.17g\t%.17g\t%.17g" % (track, int(position), point[0], point[1], point[2]))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    if gz:
        with gzip.open(path, "wb") as handle:
            handle.write(payload)
    else:
        path.write_bytes(payload)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    data = lib.Aggregate()
    rng = np.random.default_rng(17)
    coords = rng.normal(size=(2, data.n_loci, 3)) * 0.04
    lib.assert_inside_unit_ball(coords)

    candidate_tdg = OUT / "candidate_naming.3dg.gz"
    reference_tdg = OUT / "reference_naming.3dg.gz"
    write_3dg(candidate_tdg, coords, data, "candidate", gz=True)
    write_3dg(reference_tdg, coords, data, "reference", gz=True)

    cand_array = lib.three_dg_to_array(lib.load_3dg(candidate_tdg), data, track_mode="candidate")
    ref_array = lib.three_dg_to_array(lib.load_3dg(reference_tdg), data, track_mode="reference")
    assert np.array_equal(cand_array, coords) and np.array_equal(ref_array, coords)
    assert np.array_equal(cand_array, ref_array)
    wrong = lib.three_dg_to_array(lib.load_3dg(candidate_tdg), data, track_mode="reference")
    assert int(np.isfinite(wrong).sum()) == 0, "用错命名模式必须全 NaN（正是被 gate 拦住的错误）"

    gate = {"status": "PASS", "reference_opened": False, "authorized_by_parent": "code selfcheck",
            "fits": [], "manifest": {}, "selection": {}, "nulls_new": [], "nulls_reused_baseline": []}
    record_path = OUT / "reference_open.json"
    gate_path = OUT / "gate.json"
    lib.write_json(gate_path, gate)

    # (1) 缺失文件：必须先留下 opening/failed 记录
    missing = OUT / "does_not_exist.3dg.gz"
    try:
        ev.open_reference(gate, reference_path=missing, expected_sha256="0" * 64,
                          record_path=record_path, gate_path=gate_path)
        raise AssertionError("missing reference must raise")
    except RuntimeError as exc:
        assert "missing" in str(exc), exc
    failed_record = json.loads(record_path.read_text(encoding="utf-8"))
    failed_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert failed_record["status"] == "failed" and failed_record["error"]
    assert failed_record["reference_first_opened_utc"], "首次打开 UTC 必须在读取前落盘"
    assert failed_gate["reference_first_opened_utc"] == failed_record["reference_first_opened_utc"]
    assert failed_gate["reference_open_status"] == "failed"

    # (2) 命名错误（candidate 命名的文件按 reference 模式读）：必须硬失败而不是静默 NaN
    try:
        ev.open_reference(gate, reference_path=candidate_tdg, expected_sha256=lib.sha256_file(candidate_tdg),
                          record_path=record_path, gate_path=gate_path)
        raise AssertionError("candidate-named 3DG must not pass the reference parser")
    except RuntimeError as exc:
        assert "missing" in str(exc)
    assert json.loads(record_path.read_text(encoding="utf-8"))["status"] == "failed"

    # (3) 正确命名：成功打开，记录 finite 计数
    reference, record = ev.open_reference(gate, reference_path=reference_tdg,
                                          expected_sha256=lib.sha256_file(reference_tdg),
                                          record_path=record_path, gate_path=gate_path)
    assert record["status"] == "opened" and record["finite_loci"] == int(reference.shape[0] * reference.shape[1])
    assert np.array_equal(reference, coords)
    assert json.loads(gate_path.read_text(encoding="utf-8"))["reference_open_status"] == "opened"
    assert not (lib.EVAL / "gates/reference_open.json").exists(), "真实 gate 目录不得被自检写入"

    print("OPENREF CHECK PASS: naming-mode parser + first-open evidence ordering validated "
          "(synthetic 3DG only; real reference and real gate untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
