"""054 前缀登记：核对并登记复用的 40Mb 端点与各层 aggregate 输入（reference-free）。

在拟合开始前运行一次。这里只做核对与登记，不重跑 40Mb、不重建 aggregate：

* 40Mb 前缀：SHA256 与 052 terminal 记录的 artifact 哈希逐项一致；bin=40,000,000、
  78 loci、1,703,888 raw records；status=budget_not_converged / fg_budget_exhausted /
  last_accepted_endpoint=true / reference_opened=false / phase_opened=false；
* 各层输入 aggregate：SHA256 与冻结值一致，bin_size 与 sum(ceil(header/bin)) 一致，
  raw_records 守恒 1,703,888。

输出 coords/reused/40Mb-prefix-registration.json 与 inputs/reused_aggregates.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bootstrap_054 import data_io, round_runner  # noqa: E402
from round_paths_054 import (EXPECTED_N_LOCI, EXPECTED_RAW_RECORDS, REUSED_AGGREGATES,  # noqa: E402
                             REUSED_PREFIX_3DG, REUSED_PREFIX_3DG_SHA256, REUSED_PREFIX_BIN,
                             REUSED_PREFIX_FG, REUSED_PREFIX_NPZ, REUSED_PREFIX_NPZ_SHA256,
                             REUSED_PREFIX_REASON, REUSED_PREFIX_STATUS, REUSED_PREFIX_TERMINAL,
                             REUSED_PREFIX_TERMINAL_SHA256, ROOT, STAGE_BIN, stage_of_endpoint)


def main() -> int:
    # ---------- 1. 40Mb 前缀 ----------
    for path, expected, label in (
            (REUSED_PREFIX_NPZ, REUSED_PREFIX_NPZ_SHA256, "40Mb endpoint npz"),
            (REUSED_PREFIX_3DG, REUSED_PREFIX_3DG_SHA256, "40Mb endpoint 3dg"),
            (REUSED_PREFIX_TERMINAL, REUSED_PREFIX_TERMINAL_SHA256, "40Mb terminal json")):
        actual = round_runner.sha256_file(path)
        if actual != expected:
            raise RuntimeError("%s SHA mismatch: %s != %s" % (label, actual, expected))
    terminal = json.loads(Path(REUSED_PREFIX_TERMINAL).read_text(encoding="utf-8"))
    if int(terminal["bin_size_bp"]) != REUSED_PREFIX_BIN or int(terminal["n_loci"]) != EXPECTED_N_LOCI[40_000_000]:
        raise RuntimeError("reused 40Mb prefix is not on the real 40,000,000 bp / 78 loci grid")
    if float(terminal["raw_records"]) != float(EXPECTED_RAW_RECORDS):
        raise RuntimeError("reused 40Mb prefix lost raw records")
    if terminal["status"] != REUSED_PREFIX_STATUS or terminal["terminal_reason"] != REUSED_PREFIX_REASON:
        raise RuntimeError("reused 40Mb prefix terminal status changed: %s / %s"
                           % (terminal["status"], terminal["terminal_reason"]))
    if not bool(terminal["last_accepted_endpoint"]) or int(terminal["outer_fg_actual"]) != REUSED_PREFIX_FG:
        raise RuntimeError("reused 40Mb prefix is not the 200-FG last accepted endpoint")
    if bool(terminal["reference_opened"]) or bool(terminal["phase_opened"]):
        raise RuntimeError("reused 40Mb prefix was not produced reference-free")
    if stage_of_endpoint(REUSED_PREFIX_NPZ) != "40Mb":
        raise RuntimeError("reused prefix path does not name the 40Mb stage")
    with np.load(REUSED_PREFIX_NPZ, allow_pickle=False) as payload:
        coords = np.asarray(payload["coordinates"], dtype=np.float64)
        p_value = float(np.asarray(payload["p"]).item())
        q_value = float(np.asarray(payload["q"]).item())
    if coords.shape != (2, EXPECTED_N_LOCI[40_000_000], 3):
        raise RuntimeError("reused 40Mb prefix coordinate shape %s" % (coords.shape,))
    registration = {
        "schema": "p9016-round054-reused-40mb-prefix-v1",
        "role": "reference-free frozen prefix of the shared 40 -> 20 -> 10 -> 5 -> 2 -> 1 Mb chain",
        "reused_not_rerun": True,
        "no_new_40mb_terminal_fabricated": True,
        "source_run": str(REUSED_PREFIX_NPZ.parents[2].relative_to(ROOT)),
        "source_endpoint_npz": str(REUSED_PREFIX_NPZ.relative_to(ROOT)),
        "source_endpoint_npz_sha256": REUSED_PREFIX_NPZ_SHA256,
        "source_endpoint_3dg": str(REUSED_PREFIX_3DG.relative_to(ROOT)),
        "source_endpoint_3dg_sha256": REUSED_PREFIX_3DG_SHA256,
        "source_terminal_json": str(REUSED_PREFIX_TERMINAL.relative_to(ROOT)),
        "source_terminal_json_sha256": REUSED_PREFIX_TERMINAL_SHA256,
        "terminal_status": terminal["status"],
        "terminal_reason": terminal["terminal_reason"],
        "outer_fg_actual": int(terminal["outer_fg_actual"]),
        "last_accepted_endpoint": bool(terminal["last_accepted_endpoint"]),
        "bin_size_bp": REUSED_PREFIX_BIN,
        "n_loci": int(coords.shape[1]),
        "raw_records": float(terminal["raw_records"]),
        "start_p_init": terminal["start_p_init"],
        "endpoint_p": p_value, "endpoint_q": q_value,
        "initialization": {
            "source": "test_res/052-.../coords/initial/random_40Mb.npz",
            "mode": "approved_014_blind_initialization",
            "seed": 2207, "p_init": 0.75,
            "no_optimization_root": True,
            "same_objective_data_weights_as_this_run": True,
        },
        "reference_opened_by_source": bool(terminal["reference_opened"]),
        "phase_opened_by_source": bool(terminal["phase_opened"]),
        "reused_prefix_fg": REUSED_PREFIX_FG,
        "note": "052 ran 40 -> 10 and skipped 20 Mb; only its 40 Mb endpoint is reused here. "
                "The 20 Mb layer of this run is a prolongation of this endpoint, never 051's "
                "already-optimized 20 Mb.",
    }
    out = RUN / "coords/reused/40Mb-prefix-registration.json"
    round_runner.write_json(out, registration)

    # ---------- 2. 各层 aggregate 输入 ----------
    layers = {}
    for bin_size, (path, expected_sha) in sorted(REUSED_AGGREGATES.items()):
        actual = round_runner.sha256_file(path)
        if actual != expected_sha:
            raise RuntimeError("aggregate SHA mismatch for %d bp: %s" % (bin_size, actual))
        data = data_io.load_aggregate(path)
        if int(data.bin_size) != int(bin_size):
            raise RuntimeError("aggregate bin_size mismatch for %d bp" % bin_size)
        expected_bins = np.asarray([(int(L) + int(bin_size) - 1) // int(bin_size)
                                    for L in data.chromosome_lengths], dtype=np.int64)
        if not np.array_equal(np.asarray(data.n_bins, dtype=np.int64), expected_bins):
            raise RuntimeError("aggregate %d bp is not on the real bp grid" % bin_size)
        if int(data.n_loci) != int(expected_bins.sum()) != EXPECTED_N_LOCI[int(bin_size)]:
            raise RuntimeError("aggregate %d bp loci mismatch" % bin_size)
        if int(data.budget()["raw_records"]) != EXPECTED_RAW_RECORDS:
            raise RuntimeError("aggregate %d bp lost records" % bin_size)
        layers[str(int(bin_size))] = {
            "stage": [k for k, v in STAGE_BIN.items() if int(v) == int(bin_size)][0],
            "path": str(path.relative_to(ROOT)), "sha256": actual,
            "bin_size_bp": int(bin_size), "n_loci": int(data.n_loci),
            "sum_ceil_header_over_bin": int(expected_bins.sum()),
            "n_pairs": int(data.n_pairs), "raw_records": float(data.budget()["raw_records"]),
            "reused_not_rebuilt": True,
        }
    manifest = {
        "schema": "p9016-round054-reused-aggregates-v1",
        "role": "frozen per-layer aggregate inputs; reused read-only, never rebuilt in this run",
        "raw_records_conserved": EXPECTED_RAW_RECORDS,
        "layers": layers,
        "note": "052's 500kb/200kb aggregates were already generated from the raw 7-column real-bp "
                "records and verified to fold back to 1 Mb; no preflight/smoke is rerun here.",
    }
    round_runner.write_json(RUN / "inputs/reused_aggregates.json", manifest)
    print(json.dumps({"prefix": str(out.relative_to(ROOT)),
                      "layers": {k: v["n_loci"] for k, v in layers.items()},
                      "all_checks": "passed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
