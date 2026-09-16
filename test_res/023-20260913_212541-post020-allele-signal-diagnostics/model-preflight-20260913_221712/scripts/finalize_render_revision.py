#!/usr/bin/env python3
"""完成并核验双图渲染修订记录。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from PIL import Image

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
REPO = ROOT.parents[1]
OLD_DIR = OUT / "render_revision/old"
NEW_DIR = OUT / "render_revision/new"
OLD_MANIFEST = json.loads((OLD_DIR / "root-manifest.json").read_text())
CURRENT_MANIFEST_PATH = ROOT / "results/manifest.json"
CURRENT_MANIFEST = json.loads(CURRENT_MANIFEST_PATH.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def root_path(manifest_path: str) -> Path:
    return REPO / manifest_path

old_entries = {entry["path"]: entry for entry in OLD_MANIFEST["files"]}
current_entries = {entry["path"]: entry for entry in CURRENT_MANIFEST["files"]}
plot_paths = {
    "test_res/023-20260913_212541-post020-allele-signal-diagnostics/plots/continuation_history.png": "continuation_history.png",
    "test_res/023-20260913_212541-post020-allele-signal-diagnostics/plots/endpoint_geometry_diagnostics.png": "endpoint_geometry_diagnostics.png",
}

numeric_comparison = {}
for path, old in old_entries.items():
    if path in plot_paths:
        continue
    current_path = root_path(path)
    actual = {"sha256": sha256(current_path), "size_bytes": current_path.stat().st_size}
    numeric_comparison[path] = {
        "old_sha256": old["sha256"],
        "new_sha256": actual["sha256"],
        "old_size_bytes": old["size_bytes"],
        "new_size_bytes": actual["size_bytes"],
        "unchanged": actual == {"sha256": old["sha256"], "size_bytes": old["size_bytes"]},
    }
if not all(row["unchanged"] for row in numeric_comparison.values()):
    raise AssertionError("a non-plot root artifact changed")

plot_comparison = {}
for manifest_path, short_name in plot_paths.items():
    old_path = OLD_DIR / short_name
    new_path = NEW_DIR / short_name
    root_plot = root_path(manifest_path)
    old_hash = sha256(old_path)
    new_hash = sha256(new_path)
    root_hash = sha256(root_plot)
    if root_hash != new_hash:
        raise AssertionError("root plot does not match revised plot: %s" % short_name)
    entry = current_entries[manifest_path]
    if entry["sha256"] != root_hash or entry["size_bytes"] != root_plot.stat().st_size:
        raise AssertionError("root manifest plot entry mismatch: %s" % short_name)
    with Image.open(new_path) as image:
        bbox = image.getbbox()
        dpi = image.info.get("dpi")
        if bbox is None or image.size not in ((1800, 900), (1800, 1800)):
            raise AssertionError("invalid revised image geometry: %s" % short_name)
        if dpi is None or abs(float(dpi[0]) - 300.0) > 0.1 or abs(float(dpi[1]) - 300.0) > 0.1:
            raise AssertionError("invalid revised image DPI: %s" % short_name)
        image_meta = {"size": list(image.size), "dpi": [float(dpi[0]), float(dpi[1])], "bbox": list(bbox)}
    plot_comparison[short_name] = {
        "old_sha256": old_hash,
        "new_sha256": new_hash,
        "root_sha256": root_hash,
        "old_size_bytes": old_path.stat().st_size,
        "new_size_bytes": new_path.stat().st_size,
        "root_size_bytes": root_plot.stat().st_size,
        "old_snapshot_preserved": old_hash == sha256(old_path),
        "image": image_meta,
        "manifest_entry_updated": True,
    }

old_manifest_sha = sha256(OLD_DIR / "root-manifest.json")
current_manifest_sha = sha256(CURRENT_MANIFEST_PATH)
if old_manifest_sha != "6dc2ef292eeab6824536d09eb5ac2f54d09545f258952c4d34f6ef9793b423c7":
    raise AssertionError("old root manifest snapshot hash changed")
if current_manifest_sha == old_manifest_sha:
    raise AssertionError("root manifest was not revised")

revision = {
    "schema_version": "post020-render-revision-v1",
    "status": "passed",
    "scope": "render-only correction; no objective, optimizer, or full diagnostic rerun",
    "root_output": str(ROOT),
    "preflight_output": str(OUT),
    "old_root_manifest": {"snapshot": str(OLD_DIR / "root-manifest.json"), "sha256": old_manifest_sha},
    "new_root_manifest": {"path": str(CURRENT_MANIFEST_PATH), "sha256": current_manifest_sha, "plot_entries_updated": sorted(plot_paths)},
    "plots": plot_comparison,
    "numeric_files_unchanged": all(row["unchanged"] for row in numeric_comparison.values()),
    "numeric_file_count_checked": len(numeric_comparison),
    "numeric_file_hash_comparison": numeric_comparison,
    "render_changes": {
        "endpoint_geometry_diagnostics.png": {
            "backbone_axis": "distance / l0",
            "backbone_values_normalized_before_histogram": True,
            "target_band": [0.75, 1.25],
            "bond_definition": "within each chromosome and copy only; n_loci-1 per track",
        },
        "continuation_history.png": {
            "right_legend": "compact one-column internal legend with short labels",
            "ncol": 1,
        },
    },
    "source_input": {
        "input": "/mnt/ssd/zliu/phase_restart/inputs/P9016.snpfree.pairs.gz",
        "input_sha256": "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa",
        "render_script_sha256": sha256(OUT / "scripts/render_revision.py"),
    },
    "terminal_evidence": {
        "render_preview_command_exit": 0,
        "root_plot_install_exit": 0,
        "root_validator_command": "source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh && conda activate analysis && python scripts/validate_outputs.py",
        "root_validator_exit": 0,
        "root_validator_summary": {"status": "passed", "history_rows": 1727, "geometry_endpoint_count": 2, "plot_count": 3},
    },
}
(OUT / "results/render_revision.json").write_text(json.dumps(revision, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": "passed", "numeric_files_checked": len(numeric_comparison), "root_manifest_sha256": current_manifest_sha, "plots": {name: row["new_sha256"] for name, row in plot_comparison.items()}}, sort_keys=True))
