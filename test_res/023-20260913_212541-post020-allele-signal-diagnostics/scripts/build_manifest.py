#!/usr/bin/env python3
"""为已完成的 Stage-A artifacts 目录写出哈希 manifest。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
MANIFEST = ROOT / "results/manifest.json"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

files = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path == MANIFEST:
        continue
    if ".mplconfig" in path.parts or "__pycache__" in path.parts:
        continue
    files.append({
        "path": str(path.relative_to(REPO)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    })

payload = {
    "schema_version": "post020-artifact-manifest-v1",
    "root": str(ROOT.relative_to(REPO)),
    "manifest_excludes": ["results/manifest.json", ".mplconfig/", "*/__pycache__/"],
    "file_count": len(files),
    "files": files,
}
MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": "completed", "manifest": str(MANIFEST), "file_count": len(files)}, sort_keys=True))
