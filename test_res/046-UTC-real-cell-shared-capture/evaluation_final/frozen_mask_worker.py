#!/usr/bin/env python3
"""Run the byte-frozen old21 mask builder under its CPython 3.13 pyc runtime.

The main evaluator remains in the analysis environment (Python 3.12).  This
small process is used only because the published pyc has CPython 3.13 magic;
it never fits, selects, or reads anything except the post-gate old21/reference
coordinates needed for legacy mask parity.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import marshal
from typing import Any

import numpy as np
try:
    import scipy
    SCIPY_VERSION = scipy.__version__
except Exception:
    SCIPY_VERSION = None

ROOT = Path(__file__).resolve().parents[3]
FROZEN_DIR = ROOT / "docs/audits/visibility-r2-preparation-20260914T164603Z/frozen_pr"
PACKAGE = "evaluation_frozen_worker_pr"
R2_PYC = FROZEN_DIR / "r2comparison.pyc"


class _SourceLoader(importlib.abc.Loader):
    def __init__(self, path: Path, name: str) -> None:
        self.path, self.name = path, name

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: Any) -> None:
        module.__file__ = str(self.path)
        module.__package__ = PACKAGE
        module.__loader__ = self
        exec(compile(self.path.read_bytes(), str(self.path), "exec"), module.__dict__)


class _BytecodeLoader(importlib.abc.Loader):
    def __init__(self, path: Path, name: str) -> None:
        self.path, self.name = path, name

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: Any) -> None:
        module.__file__ = str(self.path)
        module.__package__ = PACKAGE
        module.__loader__ = self
        payload = self.path.read_bytes()
        exec(marshal.loads(payload[16:]), module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        prefix = PACKAGE + "."
        if not fullname.startswith(prefix):
            return None
        name = fullname[len(prefix):]
        if not name or "." in name:
            return None
        source = FROZEN_DIR / (name + ".py")
        if source.is_file():
            return importlib.machinery.ModuleSpec(fullname, _SourceLoader(source, name), origin=str(source))
        bytecode = FROZEN_DIR / (name + ".pyc")
        if bytecode.is_file():
            return importlib.machinery.ModuleSpec(fullname, _BytecodeLoader(bytecode, name), origin=str(bytecode))
        return None


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_frozen() -> tuple[Any, Any]:
    package_spec = importlib.machinery.ModuleSpec(PACKAGE, loader=None, is_package=True)
    package = importlib.util.module_from_spec(package_spec)
    package.__path__ = [str(FROZEN_DIR)]
    package.__package__ = PACKAGE
    sys.modules[PACKAGE] = package
    sys.meta_path.insert(0, _Finder())
    return importlib.import_module(PACKAGE + ".allele_r2"), importlib.import_module(PACKAGE + ".r2comparison")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    names = tuple(str(x) for x in config["chromosome_names"])
    lengths = tuple(int(x) for x in config["chromosome_lengths"])
    allele_r2, r2 = _load_frozen()
    structures = {}
    conditions = {}
    for item in config["conditions"]:
        cid = str(item["condition_id"])
        structures[cid] = allele_r2._load_coordinates(Path(item["path"]), str(item["format"]), names)
        conditions[cid] = r2.R2Condition(
            condition_id=cid, display_name=cid, structures=structures[cid],
            n_copies=int(item["n_copies"]), role="frozen_mask_input",
            endpoint_status="historical_locked", accepted_as=cid,
        )
    reference = allele_r2._load_coordinates(Path(config["reference_path"]), "3dg", names)
    arrays: dict[str, np.ndarray] = {}
    rows = []
    for ci, (name, length) in enumerate(zip(names, lengths)):
        mask = r2.build_common_mask(name, length, conditions, reference, chromosome_index=ci)
        arrays[f"chr{ci}_positions"] = np.asarray(mask.positions, dtype=np.int64)
        arrays[f"chr{ci}_pair_i"] = np.asarray(mask.pair_i, dtype=np.int64)
        arrays[f"chr{ci}_pair_j"] = np.asarray(mask.pair_j, dtype=np.int64)
        arrays[f"chr{ci}_common"] = np.asarray(mask.common, dtype=np.bool_)
        rows.append({"chromosome": name, "n_bins": mask.n_bins, "n_total_non_diagonal_pairs": mask.n_total_pairs, "n_common_pairs": mask.n_common_pairs})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    meta = {
        "schema": "p9016-frozen-legacy-mask-worker-v1", "runtime": sys.version,
        "numpy_version": np.__version__, "scipy_version": SCIPY_VERSION,
        "frozen_r2comparison_pyc": str(R2_PYC), "frozen_r2comparison_sha256": _sha(R2_PYC),
        "frozen_coordinate_parser": str(FROZEN_DIR / "allele_r2.py"), "frozen_coordinate_parser_sha256": _sha(FROZEN_DIR / "allele_r2.py"),
        "rows": rows, "output": str(args.output), "output_sha256": _sha(args.output),
    }
    args.meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
