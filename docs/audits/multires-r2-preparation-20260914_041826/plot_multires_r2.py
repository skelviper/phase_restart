"""已完成多分辨率 R2 结果的仅绘图入口。

本 wrapper 只读取 evaluation_results.json 和绘图数组。它不会打开 candidate coordinates、reference、pairs/rawphase 或 native engine。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multires_r2 import PreparationError, render_result_plots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render prepared P9016 multiresolution R2 result arrays")
    parser.add_argument("--results", required=True, help="completed evaluation_results.json")
    parser.add_argument("--output-dir", required=True, help="fresh plot output directory")
    args = parser.parse_args(argv)
    try:
        result = render_result_plots(args.results, args.output_dir)
    except (PreparationError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output_dir": result["output_dir"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
