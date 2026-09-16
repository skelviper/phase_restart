"""Fresh-process formal bootstrap probe for the shared-capture preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

# This import is intentionally first: formal_controller inserts frozen_pr/frozen_037
# before it imports pr, matching an independent formal_controller invocation.
import formal_controller as controller  # noqa: E402
from data_io import sha256_file, write_json  # noqa: E402
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective  # noqa: E402


def _sha(path: str | Path) -> str:
    return sha256_file(Path(path))


def _module_record(module: object) -> dict[str, str]:
    path = Path(str(getattr(module, "__file__", ""))).resolve()
    return {"file": str(path), "sha256": _sha(path)}


def run(run_dir: Path) -> dict:
    controller.RUN = run_dir.resolve()
    manifest = controller._load_manifest()
    data_1mb = controller._load_real_data(manifest, 1_000_000)
    import pr.contact_model as formal_contact_model
    import pr.reconstruction_init as formal_reconstruction_init
    import visibility_profile_base as formal_visibility_backend
    import gpu_variant_backend as formal_gpu_backend
    import m1_preconditioner as formal_fg_runner

    modules = {
        "formal_controller": _module_record(controller),
        "pr.contact_model": _module_record(formal_contact_model),
        "pr.reconstruction_init": _module_record(formal_reconstruction_init),
        "visibility_profile_base": _module_record(formal_visibility_backend),
        "gpu_variant_backend": _module_record(formal_gpu_backend),
        "m1_preconditioner": _module_record(formal_fg_runner),
    }
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    results = {}
    for fixture in ("P2", "N2"):
        fit = next(row for row in manifest["matrix"] if row.get("fixture") == fixture and row.get("objective_variant") == "full-J")
        data = controller._load_synthetic_data(fit)
        known_e = controller._load_known_e(fit)
        expected_path = controller._resolve(fit["data_path"])
        known_e_path = controller._resolve(fit["known_e_path"])
        truth_path = run_dir / "eval_truth" / (fixture + "_truth_1Mb.npz")
        with np.load(truth_path, allow_pickle=False) as payload:
            truth = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        records = {"fixture": fixture, "formal_loader_used": True,
                   "counts_sha256": _sha(expected_path), "known_e_sha256": _sha(known_e_path),
                   "counts_shape": list(data.counts.shape), "counts_dtype": str(data.counts.dtype),
                   "known_e_shape": list(known_e.shape), "known_e_dtype": str(known_e.dtype),
                   "natural_group_totals": {"diag": float(data.diag_counts.sum()),
                                            "cis_offdiag": float(data.counts[data.cis_pair].sum()),
                                            "inter": float(data.counts[~data.cis_pair].sum())},
                   "device": device, "models": {}}
        weights = PenaltyWeights(count=1.0, bond=0.0, repulsion=0.0, bend=0.0, p_prior=0.0)
        for model in ("S", "G"):
            objective = SharedCaptureObjective(data, model_id=model, weights=weights,
                                               mode="V0-known-generating-e", known_e=known_e,
                                               device=device, pair_block=262144, inner_cap=16, cg_cap=16,
                                               profile_warm_start=False)
            raw = objective.raw_from_physical(truth)
            theta = objective.pack(raw, p=0.8)
            theta[-1] = float(formal_contact_model.q_from_p(0.8))
            value, gradient, components = objective.evaluate(theta, need_gradient=True)
            records["models"][model] = {
                "value": float(value), "gradient_inf": float(np.max(np.abs(gradient))),
                "group_mass_kl_normalized": float(components["group_mass_kl_normalized"]),
                "observed_cis_mass": float(components["observed_cis_mass"]),
                "observed_inter_mass": float(components["observed_inter_mass"]),
                "count_nll_normalized": float(components["count_nll_normalized"]),
                "conditional_nll_raw": float(components["conditional_nll_raw"]),
                "diag_profiled_nll_raw": float(components["diag_profiled_nll_raw"]),
            }
        results[fixture] = records
    output = {"schema": "p9016-formal-loader-probe-v1", "run_id": run_dir.name,
              "bootstrap": "fresh process formal_controller first; frozen pr before objective imports",
              "modules": modules, "fixtures": results, "optimizer_started": False,
              "reference_opened": False, "phase_opened": False}
    write_json(run_dir / "checks/formal_loader_probe.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = run(Path(args.run_dir))
    print(json.dumps({"status": "PASS", "device": result["fixtures"]["P2"]["device"], "fixtures": list(result["fixtures"])}))


if __name__ == "__main__":
    main()
