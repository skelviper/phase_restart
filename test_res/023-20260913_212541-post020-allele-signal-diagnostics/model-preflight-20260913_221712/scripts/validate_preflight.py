#!/usr/bin/env python3
"""独立 model-preflight 输出的完整性检查。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = json.loads((ROOT / "results/model_preflight.json").read_text())
assert RESULT["status"] == "passed"
assert RESULT["optimizer_called"] is False
assert RESULT["forbidden_modules_imported"] == []
assert RESULT["all_real_endpoints_r_lt_0.90"] is True
assert RESULT["sealed_023_manifest"]["unchanged"] is True
assert RESULT["C3_data_contract"]["only_exposure_field_changed"] is True
assert all(RESULT["C3_data_contract"]["budget_fields_preserved_except_exposure_mode"].values())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

for entry in RESULT["source_hashes"].values():
    path = Path(entry["path"])
    assert path.is_file()
    assert sha256(path) == entry["sha256"]
assert RESULT["input"]["sha256"] == sha256(Path(RESULT["input"]["path"]))
manifest = json.loads((ROOT / "results/manifest.json").read_text())
assert manifest["file_count"] == len(manifest["files"])
for entry in manifest["files"]:
    path = ROOT / entry["path"]
    assert path.is_file()
    assert path.stat().st_size == entry["size_bytes"]
    assert sha256(path) == entry["sha256"]
for endpoint in ("020_random_joint_1m", "022_random_joint_continuation_480"):
    assert RESULT["endpoints"][endpoint]["C0_same_p_x_as_frozen_v1"]["passed"]
    c2 = RESULT["endpoints"][endpoint]["C2_map_vs_free_and_physical_recomposition"]
    assert c2["passed"]
    assert c2["map_vs_free_total_abs_difference"] == 0.0
    assert c2["map_vs_free_gradient_max_abs_difference"] == 0.0
    assert c2["map_physical_x_gradient_max_abs_difference"] == 0.0
    assert c2["free_physical_x_gradient_max_abs_difference"] == 0.0
    assert RESULT["endpoints"][endpoint]["C1_bend_removed_only"]["passed"]
    assert RESULT["endpoints"][endpoint]["C3_finite"]["passed"]

fixture = RESULT["free_writer_fixture"]
fixture_path = ROOT / "results/free-writer-fixture.3dg"
assert fixture["implicit_clip_detected"] is False
assert fixture["writer_metadata"]["serialization_clip"]["clipped_coordinates"] == 0
assert fixture_path.is_file()
assert "1.2" in fixture_path.read_text()
assert not (ROOT / "results/bounded-writer-rejection.3dg").exists()
revision = json.loads((ROOT / "results/render_revision.json").read_text())
assert revision["status"] == "passed"
assert revision["numeric_files_unchanged"] is True
assert revision["numeric_file_count_checked"] == 25
assert revision["terminal_evidence"]["root_validator_exit"] == 0
assert revision["new_root_manifest"]["plot_entries_updated"] == sorted(revision["new_root_manifest"]["plot_entries_updated"])
for name, plot in revision["plots"].items():
    assert plot["old_snapshot_preserved"] is True
    assert plot["manifest_entry_updated"] is True
    assert plot["new_sha256"] == plot["root_sha256"]
    assert plot["new_size_bytes"] == plot["root_size_bytes"]
    assert plot["image"]["size"] in ([1800, 900], [1800, 1800])
    assert abs(plot["image"]["dpi"][0] - 300.0) < 0.1
    assert abs(plot["image"]["dpi"][1] - 300.0) < 0.1

print(json.dumps({
    "status": "passed",
    "endpoints": 2,
    "c0_c2_c1_c3_contracts": "passed",
    "c3_pairs": RESULT["C3_data_contract"]["n_pairs"],
    "free_writer_max_radius": fixture["writer_metadata"]["max_radius"],
    "rss_peak_gib": RESULT["rss_peak"]["ru_maxrss_gib"],
}, sort_keys=True))
