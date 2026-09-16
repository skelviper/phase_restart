"""Compile the workspace-local fused CUDA extension without launching a kernel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

RUN = Path(__file__).resolve().parents[2]
SOURCE = RUN / "source"
LOG = RUN / "logs" / "fused-build.json"
sys.path.insert(0, str(SOURCE))

from fused.fused_objective import _BUILD, _EXTENSION_NAME, _SOURCE, load_fused_extension  # noqa: E402
import torch  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    started = time.perf_counter()
    load_fused_extension(verbose=True)
    payload = {
        "status": "compiled",
        "purpose": "workspace_local_fused_cuda_extension_build_only",
        "extension_name": _EXTENSION_NAME,
        "source": str(_SOURCE),
        "source_sha256": sha256(_SOURCE),
        "build_directory": str(_BUILD),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_kernel_launched": False,
        "elapsed_seconds": time.perf_counter() - started,
        "max_jobs": 2,
        "torch_extensions_dir": str(_BUILD),
    }
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    LOG.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
