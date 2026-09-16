"""单 fit CLI：python source/run_fit.py <loss> <solver> <source>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from round_runner import run_one_fit  # noqa: E402


def main() -> int:
    if len(sys.argv) != 4:
        print(json.dumps({"error": "usage: run_fit.py <loss> <solver> <source>"}))
        return 64
    loss, solver, source = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        summary = run_one_fit(loss, solver, source)
    except Exception as error:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(json.dumps({"fit_id": "%s-%s-%s" % (loss, solver, source), "exit": "exception",
                          "error_type": type(error).__name__, "error": str(error)}))
        return 2
    print(json.dumps({"fit_id": summary["fit_id"], "status": summary["status"],
                      "terminal_reason": summary["terminal_reason"],
                      "outer_fg_actual": summary["outer_fg_actual"],
                      "fit_wall_seconds": summary["fit_wall_seconds"],
                      "count_nll_normalized": summary["count_nll_normalized"],
                      "total": summary["total"],
                      "canonical_gradient_max_abs": summary["canonical_gradient_max_abs"]}))
    if summary["status"] == "failure":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
