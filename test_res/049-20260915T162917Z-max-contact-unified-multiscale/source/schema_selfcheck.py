"""写出 manifest/selection 后，用评价侧只读适配层做 schema 自检。

只 import `evaluation/source/eval049_inputs.py` 并调用 load_manifest / load_selection；
不运行 prep、不打开 reference、不加载 mask、不写 evaluation/ 下任何文件。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
EVAL_SOURCE = RUN / "evaluation" / "source"
sys.path.insert(0, str(EVAL_SOURCE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval049_inputs as ev  # noqa: E402
from round_paths import all_fit_ids  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest_path = RUN / "results" / "endpoint_manifest_pre_reference.json"
    selection_path = RUN / "results" / "selection_pre_reference.json"
    manifest = ev.load_manifest(manifest_path)
    selection = ev.load_selection(selection_path)
    records = []
    problems = []
    for row in manifest["fits"]:
        records.append({"role": "fit", **row})
    for row in manifest["initials"]:
        records.append({"role": "initial", **row})
    if manifest["baseline"] is not None:
        records.append({"role": "baseline", **manifest["baseline"]})
    for row in records:
        fit_id = row.get("fit_id")
        if not fit_id:
            problems.append({"role": row["role"], "problem": "missing fit_id"})
            continue
        npz_value, npz_digest = row.get("npz_path"), row.get("npz_sha256")
        if not npz_value or not npz_digest:
            problems.append({"fit_id": fit_id, "problem": "missing coords path/sha256"})
        else:
            path = Path(npz_value)
            if not path.is_absolute():
                path = RUN.parents[1] / npz_value
            if not path.exists():
                problems.append({"fit_id": fit_id, "problem": "path does not exist: %s" % npz_value})
            elif sha256(path) != npz_digest:
                problems.append({"fit_id": fit_id, "problem": "coords sha256 mismatch for %s" % npz_value})
        if row["role"] == "fit" and row.get("tdg_path"):
            path = Path(row["tdg_path"])
            if not path.is_absolute():
                path = RUN.parents[1] / row["tdg_path"]
            if not path.exists():
                problems.append({"fit_id": fit_id, "problem": "3dg missing"})
            elif sha256(path) != row["tdg_sha256"]:
                problems.append({"fit_id": fit_id, "problem": "3dg sha256 mismatch"})
    fit_ids = [row["fit_id"] for row in manifest["fits"]]
    missing = [fid for fid in all_fit_ids() if fid not in fit_ids]
    if missing:
        problems.append({"problem": "missing fit ids", "missing": missing})
    report = {
        "schema": "p9016-max-contact-schema-selfcheck-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.relative_to(RUN.parents[1])),
        "manifest_sha256": manifest["sha256"],
        "selection_path": str(selection_path.relative_to(RUN.parents[1])),
        "selection_sha256": selection["sha256"],
        "fit_count": len(manifest["fits"]),
        "initial_count": len(manifest["initials"]),
        "baseline_present": manifest["baseline"] is not None,
        "all_fit_ids_present": not missing,
        "selection_source": {key: value["selected_source"] for key, value in selection["source_selection"].items()},
        "selection_display": {loss: block["fit_id"] for loss, block in selection["display"].items()},
        "problems": problems,
        "reference_opened": False,
        "phase_opened": False,
        "note": "read-only schema check through the evaluation adapter; prep/reference/mask are NOT touched",
    }
    out = RUN / "results" / "schema_selfcheck.json"
    out.write_text(json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"status": "PASS" if not problems else "FAIL", "fits": len(manifest["fits"]),
                      "initials": len(manifest["initials"]), "problems": problems[:5],
                      "display": report["selection_display"]}))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
