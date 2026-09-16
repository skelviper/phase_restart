#!/usr/bin/env python3
"""恰好启动一次已发布的 native FDG proposal，并保留其 receipt。"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fdg_proposal as fp  # noqa: E402


def main() -> int:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    if config["status"] != "released_for_real_fdg":
        raise RuntimeError("config is not released_for_real_fdg")
    build_dir = HERE / "build"
    bridge = HERE / "build" / "fdg_bridge"
    input_path = build_dir / "bridge_input.bin"
    output_path = build_dir / "bridge_output.bin"
    if not bridge.exists() or not input_path.exists():
        raise RuntimeError("prepared bridge binary/input is missing")
    if fp.sha256_file(bridge) != config["native"]["bridge_binary_sha256"]:
        raise RuntimeError("bridge binary SHA differs from released config")
    expected_input_sha = json.loads(
        (HERE / "proposal_manifest.json").read_text(encoding="utf-8")
    )["artifacts"]["input_blob_sha256"]
    if fp.sha256_file(input_path) != expected_input_sha:
        raise RuntimeError("bridge input SHA differs from prepared proposal manifest")
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite existing native output: {output_path}")
    receipt_path = HERE / "native_attempt.json"
    if receipt_path.exists():
        raise RuntimeError(f"refusing to overwrite existing native receipt: {receipt_path}")
    attempt = fp.run_native_bridge(
        bridge, input_path, output_path, input_path.read_bytes(),
        iterations=int(config["native"]["n_iter"]),
        seed=int(config["native"]["native_seed"]),
    )
    receipt = asdict(attempt)
    receipt.update({
        "schema": "fdg-native-attempt-v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "completed_zero" if attempt.returncode == 0 else "failed_nonzero",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "iterations": int(config["native"]["n_iter"]),
        "seed": int(config["native"]["native_seed"]),
        "backend": config["native"]["backend"],
        "threads": int(config["native"]["threads"]),
    })
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "returncode": attempt.returncode,
        "input_sha256": attempt.input_sha256,
        "output_sha256": attempt.output_sha256,
        "elapsed_seconds": attempt.elapsed_seconds,
        "receipt": str(receipt_path),
    }, sort_keys=True))
    return int(attempt.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
