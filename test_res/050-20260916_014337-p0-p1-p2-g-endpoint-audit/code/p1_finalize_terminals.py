"""P1 terminal 派生字段修正（050 轮）。

背景：`p1_fork_runner.py` 在 `round_runner.repair_stage_artifacts` 写回 stage JSON 之后，
早期支的 terminal 仍从**修正前**的内存 record 取 endpoint 派生字段，因此对 ms 支会把
optimizer 空间 z 梯度写进 `raw_y_q_gradient_inf`，并让 `optimizer_gradient_inf` 为空。

本脚本只重写派生元数据：scientific 终态字段（status / terminal_reason / outer_fg_actual /
last_accepted_endpoint / canonical_gradient_max_abs / canonical_gradient_norm）必须与盘上
stage JSON 完全一致，否则报错退出；不重跑任何拟合，不改任何坐标。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
ROOT = RUN_DIR.parents[1]
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
import sys  # noqa: E402

if str(S049) not in sys.path:
    sys.path.insert(0, str(S049))
from round_paths import AGGREGATE_1MB, AGGREGATE_1MB_SHA256  # noqa: E402

FIT_IDS = ("A-raw-G-consensus", "A-ms-G-consensus", "A-raw-G-random", "A-ms-G-random")
FROZEN_TERMINAL_KEYS = ("status", "terminal_reason", "outer_fg_actual", "last_accepted_endpoint")
ENDPOINT_KEYS = ("p", "q", "canonical_gradient_max_abs", "canonical_gradient_norm",
                 "raw_y_q_gradient_inf", "optimizer_gradient_inf")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    aggregate_sha = sha256_file(AGGREGATE_1MB)
    if aggregate_sha != AGGREGATE_1MB_SHA256:
        raise RuntimeError("frozen 1Mb aggregate SHA256 mismatch: %s" % aggregate_sha)
    patched = []
    for fit_id in FIT_IDS:
        stage_path = RUN_DIR / "p1" / "stages" / fit_id / "1Mb.json"
        terminal_path = RUN_DIR / "p1" / "results" / ("%s.terminal.json" % fit_id)
        if not stage_path.is_file() or not terminal_path.is_file():
            print("skip %s (stage or terminal missing)" % fit_id)
            continue
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        endpoint = dict(stage.get("endpoint") or {})
        for key in FROZEN_TERMINAL_KEYS:
            if terminal.get(key) != stage.get(key):
                raise RuntimeError("%s: scientific terminal field %s disagrees with the repaired stage JSON "
                                   "(%r vs %r)" % (fit_id, key, terminal.get(key), stage.get(key)))
        before = {key: terminal.get(key) for key in ENDPOINT_KEYS}
        for key in ENDPOINT_KEYS:
            if key in endpoint:
                terminal[key] = endpoint[key]
            elif key in ("raw_y_q_gradient_inf", "optimizer_gradient_inf"):
                terminal[key] = None
        terminal["count_nll_normalized"] = (endpoint.get("components") or {}).get("count_nll_normalized")
        terminal["total"] = endpoint.get("total")
        terminal["derived_field_correction"] = {
            "applied": True,
            "reason": "the terminal was originally built from the pre-repair in-memory record; endpoint derived "
                      "fields are now re-read from the repaired on-disk stage JSON",
            "stage_record_path": str(stage_path.relative_to(RUN_DIR)),
            "stage_record_sha256": sha256_file(stage_path),
            "before": before,
            "after": {key: terminal.get(key) for key in ENDPOINT_KEYS},
            "scientific_fields_unchanged": list(FROZEN_TERMINAL_KEYS),
            "fit_rerun": False,
            "coordinates_changed": False,
        }
        terminal["aggregate_sha256_measured"] = aggregate_sha
        terminal["aggregate_sha256_expected"] = AGGREGATE_1MB_SHA256
        terminal_path.write_text(json.dumps(terminal, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        patched.append({"fit_id": fit_id, "before": before,
                        "after": {key: terminal.get(key) for key in ENDPOINT_KEYS}})
    print(json.dumps({"patched": patched, "aggregate_sha256": aggregate_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
