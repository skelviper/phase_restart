"""Compare independent and archived objectives at one real 020 1 Mb checkpoint."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

RUN = Path(__file__).resolve().parents[1]
WORKSPACE = RUN.parents[1]
ARCHIVED_ROOT = RUN / "source" / "archived020"
sys.path.insert(0, str(RUN / "source"))
sys.path.insert(0, str(ARCHIVED_ROOT))

from gpu_backend import (TorchObjective, load_sparse_aggregate, require_cuda,
                         threadpool_audit)  # noqa: E402
from pr.contact_model import JointObjective as ArchivedObjective  # noqa: E402
from pr.contact_model import load_aggregate  # noqa: E402
import pr.joint_fit  # noqa: E402,F401
import pr.reconstruction_init  # noqa: E402,F401

for _module_name in ("pr.contact_model", "pr.joint_fit", "pr.reconstruction_init"):
    _module = sys.modules.get(_module_name)
    if _module is None or not Path(_module.__file__).resolve().is_relative_to(ARCHIVED_ROOT):
        raise RuntimeError(f"archived source isolation failed for {_module_name}: {_module}")


CHECKPOINT = WORKSPACE / "test_res" / "020-20260913_071841-v1-p9016-joint" / "checkpoints" / "random_joint" / "1m-accepted-0240.npz"
STAGE_JSON = WORKSPACE / "test_res" / "020-20260913_071841-v1-p9016-joint" / "stages" / "random_joint" / "1m.json"
INPUT = WORKSPACE / "inputs" / "P9016.snpfree.pairs.gz"
OUT = RUN / "logs" / "real_1m_checkpoint_validation.json"


def finite_components(components):
    return {str(k): float(v) for k, v in components.items() if isinstance(v, (float, int))}


def main() -> None:
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
        limiter.__enter__()
    except Exception:
        limiter = None
    payload = {
        "timing_scope": "audit_wall_time_including_input_loader_constructor_and_objective_gradient",
        "timing_interpretation": "single-call numerical audit only; not a CPU/GPU speed comparison",
        "thread_pools_before": threadpool_audit(),
        "archived_production_block_size": 65_536,
        "independent_audit_tile_rows": 32,
        "checkpoint": str(CHECKPOINT),
    }
    checkpoint = np.load(CHECKPOINT, allow_pickle=False)
    theta = np.asarray(checkpoint["theta"], dtype=np.float64)
    checkpoint_shape = tuple(int(v) for v in checkpoint["coordinates"].shape)
    checkpoint_history = json.loads(str(checkpoint["fullhistory_json"].item()))
    checkpoint_last = checkpoint_history[-1]
    with STAGE_JSON.open(encoding="utf-8") as handle:
        stage = json.load(handle)
    expected = stage["fit"]["final_components"]
    payload.update({
        "checkpoint_theta_shape": list(theta.shape),
        "checkpoint_coordinate_shape": list(checkpoint_shape),
        "checkpoint_iteration": int(checkpoint_last["iteration"]),
        "checkpoint_nfev": int(checkpoint_last["nfev"]),
        "checkpoint_history_last": checkpoint_last,
        "stored_020_final_components": expected,
        "phase_or_reference_opened": False,
    })

    started = time.perf_counter()
    sparse = load_sparse_aggregate(INPUT, 1_000_000, verify_hash=True)
    independent = TorchObjective(sparse, device="cpu", tile_rows=32)
    ind_value, ind_gradient, ind_components = independent.evaluate(theta, need_gradient=True)
    payload["independent_torch"] = {
        "audit_wall_seconds_including_loader": time.perf_counter() - started,
        "objective_backend": "independent Torch CPU",
        "total": float(ind_value),
        "gradient_l2": float(np.linalg.norm(ind_gradient)),
        "gradient_max_abs": float(np.max(np.abs(ind_gradient))),
        "components": finite_components(ind_components),
    }
    del independent, sparse
    gc.collect()

    started = time.perf_counter()
    archived_data = load_aggregate(INPUT, bin_size=1_000_000, verify_frozen_hash=True)
    archived = ArchivedObjective(archived_data, block_size=65_536, repulsion_block_size=65_536)
    old_value, old_gradient, old_components = archived.evaluate(theta, need_gradient=True)
    payload["archived_numpy"] = {
        "audit_wall_seconds_including_loader": time.perf_counter() - started,
        "objective_backend": "archived NumPy CPU",
        "block_size": 65_536,
        "total": float(old_value),
        "gradient_l2": float(np.linalg.norm(old_gradient)),
        "gradient_max_abs": float(np.max(np.abs(old_gradient))),
        "components": finite_components(old_components),
    }
    component_keys = ("conditional_nll_raw", "diag_profiled_nll_raw", "count_nll_raw",
                      "count_nll_normalized", "p", "p_prior", "bond", "repulsion", "bend", "total")
    payload["independent_minus_archived"] = {
        key: float(ind_components[key] - old_components[key]) for key in component_keys
    }
    payload["max_abs_component_difference"] = max(abs(v) for v in payload["independent_minus_archived"].values())
    payload["max_abs_gradient_difference"] = float(np.max(np.abs(ind_gradient - old_gradient)))
    payload["independent_minus_stored_020"] = {
        key: float(ind_components[key] - expected[key]) for key in component_keys if key in expected
    }
    payload["max_abs_difference_vs_stored_020"] = max(abs(v) for v in payload["independent_minus_stored_020"].values())
    payload["validation_tolerance"] = {
        "raw_component_abs_max": 1e-6,
        "gradient_abs_max": 1e-8,
        "reason": "independent bounded tile order differs from archived NumPy summation order; normalized criterion and gradients are compared tightly",
    }
    payload["thread_pools_after"] = threadpool_audit()
    payload["status"] = "passed" if payload["max_abs_component_difference"] < 1e-6 and payload["max_abs_gradient_difference"] < 1e-8 else "failed"
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if limiter is not None:
        limiter.__exit__(None, None, None)
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
