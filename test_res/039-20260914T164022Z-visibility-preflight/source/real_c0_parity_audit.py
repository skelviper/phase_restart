"""真实5/2/1 Mb同一 theta 的新V0与旧C0 Torch点态parity audit。

只读取SNP-free counts和014 blind consensus/warm-start；不打开phase、reference、
evaluation或038。结果独立于039主预检保存。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
for path in (SOURCE, SOURCE / "frozen_035", SOURCE / "frozen_pr", SOURCE / "frozen_037"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gpu_variant_backend import GPUVariantObjective  # noqa: E402
from pr import contact_model, reconstruction_init  # noqa: E402
from visibility_preflight import configure_014_paths  # noqa: E402
from visibility_profile import VisibilityGPUObjective  # noqa: E402

INPUT_PATH = (ROOT / "inputs" / "P9016.snpfree.pairs.gz").resolve()
EXPECTED = {
    5_000_000: {"n_loci": 538, "cis_offdiag": 527902, "same_bin_diag": 607552},
    2_000_000: {"n_loci": 1329, "cis_offdiag": 619408, "same_bin_diag": 516046},
    1_000_000: {"n_loci": 2645, "cis_offdiag": 696680, "same_bin_diag": 438774},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()


def main() -> int:
    configure_014_paths()
    rows = []
    previous = None
    names_lengths = None
    started = time.perf_counter()
    for bin_size in (5_000_000, 2_000_000, 1_000_000):
        data = contact_model.load_aggregate(INPUT_PATH, bin_size=bin_size, verify_frozen_hash=True)
        expected = EXPECTED[bin_size]
        budget = data.budget()
        if int(data.n_loci) != expected["n_loci"]:
            raise RuntimeError("unexpected n_loci at %d" % bin_size)
        if int(budget["aggregate_cis_offdiag"]) != expected["cis_offdiag"]:
            raise RuntimeError("unexpected cis offdiag at %d" % bin_size)
        if int(budget["aggregate_same_bin"]) != expected["same_bin_diag"]:
            raise RuntimeError("unexpected same-bin total at %d" % bin_size)
        if names_lengths is None:
            names_lengths = tuple(zip(data.chromosome_names, data.chromosome_lengths.tolist()))
        names = tuple(name for name, _ in names_lengths)
        lengths = tuple(int(length) for _, length in names_lengths)
        if previous is None:
            state = reconstruction_init.initialize_approved_candidate("consensus", names, lengths, bin_size)
        else:
            state = reconstruction_init.warm_start_from_layer(
                previous["coords"], previous["positions"], previous["chromosome_index"],
                names, lengths, bin_size, 1103)
        previous = state
        old = GPUVariantObjective(data, "C0", device="cuda", tile_rows=32,
                                  dtype=torch.float64, use_fused=False, diagnostics=False)
        new = VisibilityGPUObjective(data, "V0-fixed-production-e", device="cuda",
                                     pair_block=262_144)
        theta = new.pack(new.raw_from_physical(np.asarray(state["coords"], dtype=np.float64)), p=0.75)
        old_started = time.perf_counter()
        old_value, old_gradient, old_components = old.evaluate(theta, need_gradient=True)
        old.synchronize()
        old_wall = time.perf_counter() - old_started
        new_started = time.perf_counter()
        new_value, new_gradient, new_components = new.evaluate(theta, need_gradient=True)
        new.synchronize()
        new_wall = time.perf_counter() - new_started
        component_names = ("count_nll_normalized", "bond", "repulsion", "bend", "total")
        component_abs = {key: abs(float(old_components[key]) - float(new_components[key]))
                         for key in component_names}
        rows.append({
            "bin_size_bp": int(bin_size),
            "n_loci": int(data.n_loci),
            "same_theta_p": 0.75,
            "old_c0_total": float(old_value),
            "new_v0_total": float(new_value),
            "value_abs": abs(float(old_value) - float(new_value)),
            "component_abs": component_abs,
            "raw_y_q_gradient_max_abs": float(np.max(np.abs(old_gradient - new_gradient))),
            "old_wall_seconds": float(old_wall),
            "new_wall_seconds": float(new_wall),
            "old_backend": "035 GPUVariantObjective Torch float64 use_fused=False",
            "new_backend": "039 VisibilityGPUObjective V0-fixed-production-e float64",
            "pass": bool(
                abs(float(old_value) - float(new_value)) <= 2e-12
                and max(component_abs.values()) <= 2e-12
                and float(np.max(np.abs(old_gradient - new_gradient))) <= 5e-11
            ),
        })
    result = {
        "schema": "real-c0-v0-parity-audit-v1",
        "status": "passed" if all(row["pass"] for row in rows) else "failed",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "input": {
            "path": str(INPUT_PATH),
            "sha256": sha256_file(INPUT_PATH),
            "phase_opened": False,
            "reference_opened": False,
            "evaluation_opened": False,
            "038_opened": False,
        },
        "layers": rows,
    }
    write_json(Path(__file__).resolve().parent.parent / "real_c0_parity_audit.json", result)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
