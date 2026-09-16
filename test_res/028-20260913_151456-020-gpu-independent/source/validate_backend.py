"""Small exactness and numerical validation for the independent backend."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "test_res" / "028-20260913_151456-020-gpu-independent"
sys.path.insert(0, str(RUN / "source"))
sys.path.insert(0, str(RUN / "source" / "archived020"))

from gpu_backend import SparseAggregate, TorchObjective, _upper_pair_index, require_cuda  # noqa: E402
from pr.contact_model import JointObjective as ArchivedObjective  # noqa: E402
from pr.contact_model import aggregate_from_arrays  # noqa: E402
import pr.joint_fit  # noqa: E402,F401
import pr.reconstruction_init  # noqa: E402,F401

for _module_name in ("pr.contact_model", "pr.joint_fit", "pr.reconstruction_init"):
    _module = sys.modules.get(_module_name)
    if _module is None or not Path(_module.__file__).resolve().is_relative_to(RUN / "source" / "archived020"):
        raise RuntimeError(f"archived source isolation failed for {_module_name}: {_module}")


def fixture_sparse() -> tuple[SparseAggregate, dict]:
    names = ("chr1", "chr2")
    lengths = np.asarray([7, 5], dtype=np.int64)
    bin_size = 2
    n_bins = ((lengths + bin_size - 1) // bin_size).astype(np.int64)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(n_bins[:-1])))
    ci = np.asarray([0, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    p1 = np.asarray([0, 2, 4, 0, 1, 1, 3, 6, 4], dtype=np.int64)
    cj = np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 1], dtype=np.int64)
    p2 = np.asarray([0, 2, 6, 0, 1, 3, 3, 0, 4], dtype=np.int64)
    g1 = offsets[ci] + p1 // bin_size
    g2 = offsets[cj] + p2 // bin_size
    same = g1 == g2
    cis = ci == cj
    eligible = ~same
    n_loci = int(n_bins.sum())
    flat = _upper_pair_index(np.minimum(g1[eligible], g2[eligible]),
                             np.maximum(g1[eligible], g2[eligible]), n_loci)
    observed_flat, observed_counts = np.unique(flat, return_counts=True)
    locus_chromosome = np.repeat(np.arange(len(names), dtype=np.int32), n_bins)
    locus_bin = np.concatenate([np.arange(int(n), dtype=np.int64) for n in n_bins])
    diag_counts = np.bincount(g1[same], minlength=n_loci).astype(np.int64)
    endpoint_counts = np.bincount(np.concatenate((g1, g2)), minlength=n_loci).astype(np.int64)
    exposure = np.sqrt(endpoint_counts.astype(np.float64) + 10.0)
    exposure /= exposure.mean()
    cis_counts = np.unique(flat[cis[eligible]], return_counts=True)[1]
    inter_counts = np.unique(flat[~cis[eligible]], return_counts=True)[1]
    from scipy.special import gammaln
    conditional_constant = (
        -float(gammaln(int((cis & ~same).sum()) + 1.0))
        + float(gammaln(cis_counts[cis_counts > 1].astype(np.float64) + 1.0).sum())
        - float(gammaln(int((~cis).sum()) + 1.0))
        + float(gammaln(inter_counts[inter_counts > 1].astype(np.float64) + 1.0).sum())
    )
    diag_factorial_sum = float(gammaln(diag_counts[diag_counts > 1].astype(np.float64) + 1.0).sum())
    data = SparseAggregate(
        chromosome_names=names,
        chromosome_lengths=lengths,
        bin_size=bin_size,
        n_bins=n_bins,
        offsets=offsets,
        locus_chromosome=locus_chromosome,
        locus_bin=locus_bin,
        endpoint_counts=endpoint_counts,
        exposure=exposure,
        observed_flat=observed_flat.astype(np.int64),
        observed_counts=observed_counts.astype(np.int64),
        diag_counts=diag_counts,
        raw_records=len(g1),
        raw_same_bin=int(same.sum()),
        raw_cis_offdiag=int((cis & ~same).sum()),
        raw_inter=int((~cis).sum()),
        conditional_factorial_constant=conditional_constant,
        diag_factorial_sum=diag_factorial_sum,
    )
    data.assert_consistent()
    archived = aggregate_from_arrays(names, lengths, ci, p1, cj, p2, bin_size)
    return data, {"archived": archived, "ci": ci, "p1": p1, "cj": cj, "p2": p2}


def main() -> None:
    data, extra = fixture_sparse()
    archived_data = extra["archived"]
    rng = np.random.default_rng(9417)
    y = rng.normal(0.0, 0.13, size=(2, data.n_loci, 3)).astype(np.float64)
    archived_obj = ArchivedObjective(archived_data, block_size=3, repulsion_block_size=3)
    theta = archived_obj.pack(y, p=0.67)
    archived_total, archived_grad, archived_components = archived_obj.evaluate(theta, need_gradient=True)
    independent_obj = TorchObjective(data, device="cpu", tile_rows=2)
    independent_total, independent_grad, independent_components = independent_obj.evaluate(theta, need_gradient=True)
    component_errors = {}
    for key in ("conditional_nll_raw", "diag_profiled_nll_raw", "count_nll_raw", "count_nll_normalized",
                "p", "p_prior", "bond", "repulsion", "bend", "total"):
        component_errors[key] = abs(float(independent_components[key]) - float(archived_components[key]))
    gradient_error = float(np.max(np.abs(independent_grad - archived_grad)))
    value_error = abs(float(independent_total) - float(archived_total))

    finite_difference_errors = []
    for index in np.linspace(0, len(theta) - 1, num=min(12, len(theta)), dtype=int):
        step = 1e-6
        plus = theta.copy(); plus[index] += step
        minus = theta.copy(); minus[index] -= step
        f_plus = independent_obj.evaluate(plus, need_gradient=False)[0]
        f_minus = independent_obj.evaluate(minus, need_gradient=False)[0]
        finite_difference_errors.append(abs((f_plus - f_minus) / (2.0 * step) - independent_grad[index]))

    swapped = theta.copy()
    swapped[:-1] = y[::-1].reshape(-1)
    swapped_total = independent_obj.evaluate(swapped, need_gradient=False)[0]
    coordinates, _ = independent_obj.coordinates_and_p(theta)
    max_radius = float(np.linalg.norm(coordinates, axis=-1).max())
    result = {
        "status": "passed" if value_error < 1e-11 and gradient_error < 1e-9 and max(finite_difference_errors) < 2e-6 and abs(swapped_total - independent_total) < 1e-11 else "failed",
        "fixture_budget": data.budget(),
        "archived_budget": archived_data.budget(),
        "value_abs_error": value_error,
        "max_component_abs_error": max(component_errors.values()),
        "component_abs_errors": component_errors,
        "max_gradient_abs_error": gradient_error,
        "max_finite_difference_abs_error": max(finite_difference_errors),
        "finite_difference_abs_errors": finite_difference_errors,
        "copy_swap_objective_abs_error": abs(swapped_total - independent_total),
        "max_radius": max_radius,
        "finite_ball": bool(max_radius < 1.0),
        "cuda_probe": require_cuda(),
        "model_repulsion_threshold": "0.7*l0",
        "reference_or_phase_opened": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
