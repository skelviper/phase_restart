#!/usr/bin/env python3
"""Phase-A diagnostic artifacts 的运行后小型完整性验证器。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
required = (
    "README.md", "config.json", "results/provenance.json", "results/data_budget.json",
    "results/model_semantics_audit.json", "results/source_hash_comparison.json", "results/history.tsv",
    "results/history_summary.json", "results/endpoint_gradients.json",
    "results/radius_sphere.json", "results/radius_sphere.tsv", "results/gradient_summary.tsv",
    "results/geometry_summary.tsv",
    "results/geometry_details.json", "results/recommendations.json",
    "results/plot_manifest.json", "results/manifest.json", "logs/analysis_runtime.json",
    "logs/terminal_exit.json",
)
missing = [name for name in required if not (ROOT / name).is_file()]
if missing:
    raise SystemExit("missing artifacts: " + ", ".join(missing))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = json.loads((RESULTS / "manifest.json").read_text())
assert manifest["file_count"] == len(manifest["files"])
for entry in manifest["files"]:
    path = ROOT.parent.parent / entry["path"]
    assert path.is_file()
    assert path.stat().st_size == entry["size_bytes"]
    assert sha256(path) == entry["sha256"]

budget = json.loads((RESULTS / "data_budget.json").read_text())
assert budget["raw_records"] == 1703888
assert budget["raw_cis_offdiag"] == 696680
assert budget["raw_inter"] == 568434
assert budget["raw_same_bin"] == 438774
assert budget["n_eligible_pairs"] == 3496690
source_compare = json.loads((RESULTS / "source_hash_comparison.json").read_text())
assert source_compare["files"]["joint_fit.py"]["current_equals_020"] is False
assert source_compare["files"]["contact_model.py"]["current_equals_020"] is True

with (RESULTS / "history.tsv").open() as handle:
    history = list(csv.DictReader(handle, delimiter="\t"))
assert len(history) == 1727
assert sum(row["source"] == "020" for row in history) == 1486
assert sum(row["source"] == "022" for row in history) == 241

endpoints = json.loads((RESULTS / "endpoint_gradients.json").read_text())
for payload in endpoints.values():
    check = payload["joint_objective_gradient_recomposition"]
    assert check["passed_abs_1e-10"]
    assert check["max_abs_difference"] == 0.0

radii = json.loads((RESULTS / "radius_sphere.json").read_text())
for payload in radii.values():
    assert len(payload["per_track"]) == 40
    assert payload["global"]["boundary_counts"]["radius_ge_1.00"] == 0
with (RESULTS / "radius_sphere.tsv").open() as handle:
    radius_rows = list(csv.DictReader(handle, delimiter="\t"))
assert len(radius_rows) == 80
with (RESULTS / "gradient_summary.tsv").open() as handle:
    gradient_rows = list(csv.DictReader(handle, delimiter="\t"))
assert len(gradient_rows) == 10

geometry = json.loads((RESULTS / "geometry_details.json").read_text())
for payload in geometry.values():
    rep = payload["repulsion"]
    assert rep["model_distance_term_denominator"] == 13989405
    assert rep["near_total"] >= 0
    assert abs(rep["near_fraction_total"] - rep["near_total"] / 13989405) < 1e-15

plots = json.loads((RESULTS / "plot_manifest.json").read_text())
assert plots["dpi"] == 300 and plots["text_size_pt"] == 7
for name in plots["plots"]:
    path = ROOT.parent.parent / name
    with Image.open(path) as image:
        assert image.size[0] >= 500 and image.size[1] >= 500
        assert image.info.get("dpi", (0, 0))[0] > 299
        assert image.getextrema()[:3] != ((0, 0), (0, 0), (0, 0))

print(json.dumps({"status": "passed", "history_rows": len(history), "endpoint_count": len(endpoints), "geometry_endpoint_count": len(geometry), "plot_count": len(plots["plots"])}, sort_keys=True))
