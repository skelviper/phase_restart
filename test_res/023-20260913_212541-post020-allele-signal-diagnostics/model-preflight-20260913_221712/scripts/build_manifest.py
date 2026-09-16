#!/usr/bin/env python3
"""在不触碰已封存 023 的前提下，为独立 model-preflight artifacts 哈希。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/manifest.json"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

files = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path == MANIFEST:
        continue
    if "__pycache__" in path.parts:
        continue
    files.append({"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
MANIFEST.write_text(json.dumps({
    "schema_version": "post020-model-preflight-manifest-v1",
    "root": str(ROOT),
    "manifest_excludes": ["results/manifest.json", "*/__pycache__"],
    "file_count": len(files),
    "files": files,
}, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": "completed", "file_count": len(files), "manifest": str(MANIFEST)}, sort_keys=True))
