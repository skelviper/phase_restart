"""Independent five-variant Torch/fused-CUDA adapter for multiresolution fits.

This module is intentionally outside the frozen CPU runner.  It reuses the
read-only 028 TorchObjective and fused ordered-row CUDA extension, then applies
the registered C0-C3 coordinate/exposure contracts around that backend.  No
phase-bearing data, reference coordinates, or evaluation modules are imported.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import sys
from typing import Any

# Set caps before importing NumPy, SciPy, or Torch.
for _thread_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_key] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
RUN_028 = ROOT / "test_res" / "028-20260913_151456-020-gpu-independent"
SOURCE_028 = RUN_028 / "source"
if str(SOURCE_028) not in sys.path:
    sys.path.insert(0, str(SOURCE_028))

# These imports are the immutable, already-validated 028 training-side backend.
from gpu_backend import (  # type: ignore  # noqa: E402
    BackendError,
    EPSILON,
    P_FLOOR,
    P_PRIOR_STRENGTH,
    TorchObjective as _TorchObjective,
    p_from_q as _torch_p_from_q,
    sphere_forward,
    sphere_pullback,
)
from fused.fused_objective import FusedObjective as _FusedObjective  # type: ignore  # noqa: E402

from pr import allele_models, contact_model  # noqa: E402


VARIANTS = ("C0", "C1", "C2-map", "C2-free", "C3")
SPHERE_VARIANTS = frozenset(("C0", "C1", "C3"))


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SparseView:
    """The 028 sparse/implicit-grid data contract backed by current aggregates."""

    chromosome_names: tuple[str, ...]
    chromosome_lengths: np.ndarray
    bin_size: int
    n_bins: np.ndarray
    offsets: np.ndarray
    locus_chromosome: np.ndarray
    locus_bin: np.ndarray
    endpoint_counts: np.ndarray
    exposure: np.ndarray
    observed_flat: np.ndarray
    observed_counts: np.ndarray
    diag_counts: np.ndarray
    raw_records: int
    raw_same_bin: int
    raw_cis_offdiag: int
    raw_inter: int
    conditional_factorial_constant: float
    diag_factorial_sum: float
    exposure_mode: str

    @property
    def n_loci(self) -> int:
        return int(len(self.locus_chromosome))

    @property
    def n_pairs(self) -> int:
        return self.n_loci * (self.n_loci - 1) // 2

    @property
    def l0(self) -> float:
        return float((2.0 * self.n_loci) ** (-1.0 / 3.0))

    @property
    def r0(self) -> float:
        return 2.0 * self.l0

    @property
    def repulsion_threshold(self) -> float:
        return 0.7 * self.l0

    @classmethod
    def from_aggregate(cls, data: contact_model.AggregatedContacts,
                       model_id: str) -> "SparseView":
        data.assert_consistent()
        counts = np.asarray(data.counts, dtype=np.int64)
        observed_flat = np.flatnonzero(counts > 0).astype(np.int64, copy=False)
        observed_counts = counts[observed_flat].astype(np.int64, copy=False)
        exposure_mode = "uniform" if model_id == "C3" else str(data.exposure_mode)
        exposure = (np.ones(data.n_loci, dtype=np.float64)
                    if model_id == "C3"
                    else np.asarray(data.exposure, dtype=np.float64).copy())
        result = cls(
            chromosome_names=tuple(data.chromosome_names),
            chromosome_lengths=np.asarray(data.chromosome_lengths, dtype=np.int64).copy(),
            bin_size=int(data.bin_size),
            n_bins=np.asarray(data.n_bins, dtype=np.int64).copy(),
            offsets=np.asarray(data.offsets, dtype=np.int64).copy(),
            locus_chromosome=np.asarray(data.locus_chromosome, dtype=np.int32).copy(),
            locus_bin=np.asarray(data.locus_bin, dtype=np.int64).copy(),
            endpoint_counts=np.asarray(data.endpoint_counts, dtype=np.int64).copy(),
            exposure=exposure,
            observed_flat=observed_flat,
            observed_counts=observed_counts,
            diag_counts=np.asarray(data.diag_counts, dtype=np.int64).copy(),
            raw_records=int(data.raw_records),
            raw_same_bin=int(data.raw_same_bin),
            raw_cis_offdiag=int(data.raw_cis_offdiag),
            raw_inter=int(data.raw_inter),
            conditional_factorial_constant=float(data.conditional_factorial_constant),
            diag_factorial_sum=float(data.diag_factorial_sum),
            exposure_mode=exposure_mode,
        )
        result.assert_consistent()
        return result

    def budget(self) -> dict[str, Any]:
        observed_total = int(self.observed_counts.sum())
        return {
            "n_loci": self.n_loci,
            "n_eligible_pairs": self.n_pairs,
            "n_observed_nonzero_pairs": int(len(self.observed_flat)),
            "n_zero_eligible_pairs": int(self.n_pairs - len(self.observed_flat)),
            "n_diag_bins": self.n_loci,
            "raw_records": self.raw_records,
            "raw_same_bin": self.raw_same_bin,
            "raw_cis_offdiag": self.raw_cis_offdiag,
            "raw_inter": self.raw_inter,
            "aggregate_same_bin": int(self.diag_counts.sum()),
            "aggregate_offdiag": observed_total,
            "endpoint_total": int(self.endpoint_counts.sum()),
            "raw_conserved": self.raw_records == self.raw_same_bin + self.raw_cis_offdiag + self.raw_inter,
            "aggregate_conserved": (int(self.diag_counts.sum()) == self.raw_same_bin
                                    and observed_total == self.raw_cis_offdiag + self.raw_inter),
            "endpoint_conserved": int(self.endpoint_counts.sum()) == 2 * self.raw_records,
            "exposure_mode": self.exposure_mode,
        }

    def assert_consistent(self) -> None:
        if self.n_pairs != self.n_loci * (self.n_loci - 1) // 2:
            raise BackendError("sparse view does not represent the complete upper triangle")
        if self.observed_flat.shape != self.observed_counts.shape:
            raise BackendError("sparse observed arrays have different shapes")
        if len(self.observed_flat) and np.any(np.diff(self.observed_flat) <= 0):
            raise BackendError("sparse observed indices are not strictly increasing")
        if len(self.observed_flat) and (self.observed_flat[0] < 0 or self.observed_flat[-1] >= self.n_pairs):
            raise BackendError("sparse observed index is outside the complete pair grid")
        if np.any(self.observed_counts <= 0) or np.any(self.diag_counts < 0):
            raise BackendError("sparse/diagonal counts must be nonnegative, with positive sparse counts")
        if not np.all(np.isfinite(self.exposure)) or np.any(self.exposure <= 0.0):
            raise BackendError("exposure must be finite and positive")
        if not np.isclose(float(self.exposure.mean()), 1.0, rtol=0.0, atol=1e-14):
            raise BackendError("exposure mean must remain one")
        audit = self.budget()
        if not (audit["raw_conserved"] and audit["aggregate_conserved"] and audit["endpoint_conserved"]):
            raise BackendError("contact budget is not conserved")


class GPUVariantObjective:
    """Five-contract objective using the 028 CPU/Torch or fused CUDA backend."""

    def __init__(self, data: contact_model.AggregatedContacts, model_id: str,
                 device: str = "cuda", tile_rows: int = 32,
                 dtype: torch.dtype = torch.float64,
                 use_fused: bool = True, diagnostics: bool = True):
        model_id = str(model_id)
        if model_id not in VARIANTS:
            raise ValueError("unknown variant %r; expected %s" % (model_id, VARIANTS))
        if dtype is not torch.float64:
            raise BackendError("GPU variant backend is frozen to float64")
        self.model_id = model_id
        self.spec = allele_models.model_spec(model_id)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise BackendError("CUDA requested but torch.cuda.is_available() is false")
        if self.device.type not in ("cpu", "cuda"):
            raise BackendError("device must be cpu or cuda")
        self.dtype = dtype
        self.tile_rows = int(tile_rows)
        if self.tile_rows <= 0:
            raise ValueError("tile_rows must be positive")
        self.diagnostics_enabled = bool(diagnostics)
        self.data = SparseView.from_aggregate(data, model_id)
        if self.device.type == "cuda" and use_fused:
            self._backend = _FusedObjective(self.data, device=str(self.device),
                                            tile_rows=self.tile_rows, dtype=dtype)
            self.backend = "028_fused_cuda"
        else:
            self._backend = _TorchObjective(self.data, device=str(self.device),
                                            tile_rows=self.tile_rows, dtype=dtype)
            self.backend = "028_torch_cpu" if self.device.type == "cpu" else "028_torch_cuda"
        self.n_parameters = 6 * self.data.n_loci + 1
        self._last_theta: np.ndarray | None = None
        self._last_value: float | None = None
        self._last_gradient: np.ndarray | None = None
        self._last_components: dict[str, Any] | None = None
        self._objective_eval_count = 0
        self._nonidentity_map_eval_count = 0
        self._nonidentity_map_bead_count = 0
        self._max_raw_radius = 0.0
        self._max_physical_radius = 0.0
        self._sync()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def synchronize(self) -> None:
        self._sync()

    def _map_raw(self, raw: torch.Tensor) -> torch.Tensor:
        if self.model_id in SPHERE_VARIANTS:
            return sphere_forward(raw)
        if self.model_id == "C2-free":
            return raw
        radius = torch.sqrt(torch.sum(raw * raw, dim=-1, keepdim=True))
        outer = radius > float(allele_models.IDENTITY_RADIUS)
        safe_radius = torch.clamp(radius, min=torch.finfo(raw.dtype).tiny)
        t = (radius - float(allele_models.IDENTITY_RADIUS)) / float(allele_models.IDENTITY_WIDTH)
        hyp = torch.sqrt(1.0 + t * t)
        mapped_radius = float(allele_models.IDENTITY_RADIUS) + float(allele_models.IDENTITY_WIDTH) * t / hyp
        mapped_radius = torch.clamp(mapped_radius, max=float(np.nextafter(1.0, 0.0)))
        mapped = raw * (mapped_radius / safe_radius)
        return torch.where(outer, mapped, raw)

    def _pullback(self, raw: torch.Tensor, gradient_x: torch.Tensor) -> torch.Tensor:
        if self.model_id in SPHERE_VARIANTS:
            return sphere_pullback(raw, gradient_x)
        if self.model_id == "C2-free":
            return gradient_x
        radius = torch.sqrt(torch.sum(raw * raw, dim=-1, keepdim=True))
        outer = radius > float(allele_models.IDENTITY_RADIUS)
        safe_radius = torch.clamp(radius, min=torch.finfo(raw.dtype).tiny)
        t = (radius - float(allele_models.IDENTITY_RADIUS)) / float(allele_models.IDENTITY_WIDTH)
        hyp = torch.sqrt(1.0 + t * t)
        mapped_radius = float(allele_models.IDENTITY_RADIUS) + float(allele_models.IDENTITY_WIDTH) * t / hyp
        scale = mapped_radius / safe_radius
        radial_derivative = 1.0 / (hyp * hyp * hyp)
        unit = raw / safe_radius
        radial_component = torch.sum(unit * gradient_x, dim=-1, keepdim=True)
        mapped_gradient = (scale * gradient_x
                           + (radial_derivative - scale) * unit * radial_component)
        return torch.where(outer, mapped_gradient, gradient_x)

    def _record_map(self, raw: torch.Tensor, physical: torch.Tensor) -> None:
        if not self.diagnostics_enabled:
            return
        raw_radius = torch.linalg.vector_norm(raw, dim=-1)
        physical_radius = torch.linalg.vector_norm(physical, dim=-1)
        self._objective_eval_count += 1
        self._max_raw_radius = max(self._max_raw_radius, float(raw_radius.detach().max().cpu().item()))
        self._max_physical_radius = max(self._max_physical_radius,
                                        float(physical_radius.detach().max().cpu().item()))
        if self.model_id == "C2-map":
            nonidentity = raw_radius > float(allele_models.IDENTITY_RADIUS)
            self._nonidentity_map_eval_count += int(bool(nonidentity.any().detach().cpu().item()))
            self._nonidentity_map_bead_count += int(nonidentity.sum().detach().cpu().item())

    def _count_fused(self, x: torch.Tensor, p: float,
                     need_gradient: bool):
        full, sparse = self._backend._pair_terms(x, p)
        full_summary, grad_r_cis, grad_r_inter, grad_rep = full
        sparse_summary, grad_observed = sparse
        sum_cis = 0.5 * full_summary[0]
        sum_inter = 0.5 * full_summary[1]
        d_r_cis_dp = 0.5 * full_summary[2]
        repulsion = (0.5 * full_summary[3] + full_summary[4]) / float(2 * self.data.n_loci)
        zero = torch.zeros((), dtype=self.dtype, device=self.device)
        group_cis = float(self.data.raw_cis_offdiag)
        group_inter = float(self.data.raw_inter)
        if group_cis > 0.0:
            conditional_cis = group_cis * torch.log(sum_cis) - sparse_summary[0]
            alpha_cis = group_cis / sum_cis
        else:
            conditional_cis = zero
            alpha_cis = zero
        if group_inter > 0.0:
            conditional_inter = group_inter * torch.log(sum_inter) - sparse_summary[1]
            alpha_inter = group_inter / sum_inter
        else:
            conditional_inter = zero
            alpha_inter = zero
        conditional = conditional_cis + conditional_inter
        positive = self._backend._diag_counts[self._backend._diag_counts > 0]
        diag = (positive - positive * torch.log(positive) + torch.lgamma(positive + 1.0)).sum()
        count_raw = conditional + diag
        normalized = count_raw / float(self.data.raw_records)
        packed = torch.stack((sum_cis, sum_inter, sparse_summary[0], sparse_summary[1],
                              conditional, diag, count_raw, normalized))
        packed_np = packed.detach().cpu().numpy()
        components = {
            "count_mode": "raw_integer",
            "count_factorial_status": "integer_count_constants_recorded",
            "sum_rate_cis_offdiag": float(packed_np[0]),
            "sum_rate_inter": float(packed_np[1]),
            "observed_log_rate_cis_offdiag": float(packed_np[2]),
            "observed_log_rate_inter": float(packed_np[3]),
            "conditional_nll_raw": float(packed_np[4]),
            "conditional_factorial_constant_omitted": float(self.data.conditional_factorial_constant),
            "diag_factorial_sum_included": float(self.data.diag_factorial_sum),
            "diag_profiled_nll_raw": float(packed_np[5]),
            "count_nll_raw": float(packed_np[6]),
            "count_nll_normalized": float(packed_np[7]),
        }
        if not need_gradient:
            return components, None, None, repulsion, None
        count_gradient_x = (alpha_cis * grad_r_cis + alpha_inter * grad_r_inter - grad_observed)
        count_gradient_x = count_gradient_x / float(self.data.raw_records)
        count_gradient_p = (alpha_cis * d_r_cis_dp - sparse_summary[2]) / float(self.data.raw_records)
        return components, count_gradient_x, count_gradient_p, repulsion, grad_rep / float(2 * self.data.n_loci)

    def _count_terms(self, x: torch.Tensor, p_tensor: torch.Tensor,
                     p_scalar: float, need_gradient: bool):
        if self.backend == "028_fused_cuda":
            return self._count_fused(x, p_scalar, need_gradient)
        components, gradient_x, gradient_p = self._backend._count_layer(
            x, p_tensor, need_gradient)
        components = dict(components)
        components.setdefault("count_mode", "raw_integer")
        components.setdefault("count_factorial_status", "integer_count_constants_recorded")
        return components, gradient_x, gradient_p, None, None

    def evaluate(self, theta: np.ndarray, need_gradient: bool = True):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,) or not np.all(np.isfinite(theta)):
            raise BackendError("theta has wrong shape or nonfinite values")
        theta_t = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        raw = theta_t[:-1].reshape(2, self.data.n_loci, 3)
        physical = self._map_raw(raw)
        self._record_map(raw, physical)
        self.validate_physical_coordinates_tensor(physical)
        p_scalar, dpdq_scalar = contact_model.p_from_q(float(theta[-1]))
        p_tensor = torch.as_tensor(p_scalar, dtype=self.dtype, device=self.device)
        count_components, count_gradient_x, count_gradient_p, fused_repulsion, fused_repulsion_gradient = self._count_terms(
            physical, p_tensor, p_scalar, need_gradient)
        bond, bond_gradient = self._backend._bond_layer(physical)
        bend, bend_gradient = self._backend._bend_layer(physical)
        if fused_repulsion is None:
            repulsion, repulsion_gradient = self._backend._repulsion_layer(physical)
        else:
            repulsion, repulsion_gradient = fused_repulsion, fused_repulsion_gradient
        p_prior = -P_PRIOR_STRENGTH * torch.log(p_tensor * (1.0 - p_tensor))
        p_prior_derivative = P_PRIOR_STRENGTH * (1.0 / (1.0 - p_tensor) - 1.0 / p_tensor)
        weights = self.spec.weights
        components = dict(count_components)
        scalar_items = {
            "p": p_tensor,
            "p_prior": p_prior,
            "bond": bond,
            "repulsion": repulsion,
            "bend": bend,
        }
        for key, value in scalar_items.items():
            components[key] = float(value.detach().cpu().item())
        components["repulsion_threshold_l0"] = 0.7
        total = (float(weights.count) * torch.as_tensor(components["count_nll_normalized"], dtype=self.dtype, device=self.device)
                 + float(weights.p_prior) * p_prior
                 + float(weights.bond) * bond
                 + float(weights.repulsion) * repulsion
                 + float(weights.bend) * bend)
        components["total"] = float(total.detach().cpu().item())
        if not need_gradient:
            self._sync()
            return components["total"], None, components
        combined_x = (float(weights.count) * count_gradient_x
                      + float(weights.bond) * bond_gradient
                      + float(weights.repulsion) * repulsion_gradient
                      + float(weights.bend) * bend_gradient)
        gradient_raw = self._pullback(raw, combined_x)
        gradient_q = torch.as_tensor(dpdq_scalar, dtype=self.dtype, device=self.device) * (
            float(weights.count) * count_gradient_p
            + float(weights.p_prior) * p_prior_derivative)
        gradient = torch.cat((gradient_raw.reshape(-1), gradient_q.reshape(1)))
        gradient_np = gradient.detach().cpu().numpy().astype(np.float64, copy=False)
        self._sync()
        if not np.all(np.isfinite(gradient_np)) or not math.isfinite(components["total"]):
            raise FloatingPointError("nonfinite variant objective or gradient")
        return components["total"], gradient_np.copy(), components

    def value_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, components = self.evaluate(theta, need_gradient=True)
        self._last_theta = np.asarray(theta, dtype=np.float64).copy()
        self._last_value = float(value)
        self._last_gradient = np.asarray(gradient, dtype=np.float64).copy()
        self._last_components = dict(components)
        return float(value), self._last_gradient.copy()

    def cached_value_and_components(self, theta: np.ndarray):
        if self._last_theta is None:
            return None
        candidate = np.asarray(theta, dtype=np.float64)
        if candidate.shape != self._last_theta.shape or not np.array_equal(candidate, self._last_theta):
            return None
        return self._last_value, dict(self._last_components or {})

    def components(self, theta: np.ndarray) -> dict[str, Any]:
        cached = self.cached_value_and_components(theta)
        if cached is not None:
            return cached[1]
        return self.evaluate(theta, need_gradient=False)[2]

    def validate_physical_coordinates_tensor(self, coordinates: torch.Tensor) -> None:
        if not torch.all(torch.isfinite(coordinates)):
            raise BackendError("physical coordinates are nonfinite")
        if self.model_id != "C2-free":
            radius = torch.linalg.vector_norm(coordinates, dim=-1)
            if bool(torch.any(radius >= 1.0).detach().cpu().item()):
                raise BackendError("bounded variant left the strict unit ball")

    def validate_physical_coordinates(self, coordinates: np.ndarray) -> None:
        values = np.asarray(coordinates, dtype=np.float64)
        if values.shape != (2, self.data.n_loci, 3) or not np.all(np.isfinite(values)):
            raise BackendError("physical coordinates have invalid shape or nonfinite values")
        allele_models.validate_physical_for_model(self.model_id, values)

    def raw_from_physical(self, coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinates, dtype=np.float64)
        self.validate_physical_coordinates(values)
        if self.model_id in SPHERE_VARIANTS:
            norm2 = np.sum(values * values, axis=-1, keepdims=True)
            return values / np.sqrt(1.0 - norm2)
        if self.model_id == "C2-map":
            return allele_models.identity_interior_inverse(values)
        return values.copy()

    def physical_coordinates_from_raw(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (2, self.data.n_loci, 3) or not np.all(np.isfinite(raw)):
            raise BackendError("raw coordinates have invalid shape or nonfinite values")
        tensor = torch.as_tensor(raw, dtype=self.dtype, device=self.device)
        result = self._map_raw(tensor).detach().cpu().numpy().astype(np.float64, copy=False)
        self.validate_physical_coordinates(result)
        self._sync()
        return result.copy()

    def pack(self, raw: np.ndarray, p: float = 0.75) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (2, self.data.n_loci, 3) or not np.all(np.isfinite(raw)):
            raise BackendError("raw coordinates have invalid shape or nonfinite values")
        return np.concatenate((raw.reshape(-1), np.asarray([contact_model.q_from_p(float(p))], dtype=np.float64)))

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,) or not np.all(np.isfinite(theta)):
            raise BackendError("theta has wrong shape or nonfinite values")
        return theta[:-1].reshape(2, self.data.n_loci, 3), float(theta[-1])

    def coordinates_and_p(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        raw, q = self.unpack(theta)
        coordinates = self.physical_coordinates_from_raw(raw)
        p, _ = contact_model.p_from_q(q)
        return coordinates, float(p)

    def map_diagnostics(self) -> dict[str, Any]:
        evaluations = self._objective_eval_count
        return {
            "map": self.spec.coordinate_parameterization,
            "physical_domain": self.spec.physical_domain,
            "objective_eval_count": int(evaluations),
            "nonidentity_map_eval_count": int(self._nonidentity_map_eval_count),
            "nonidentity_map_eval_fraction": (float(self._nonidentity_map_eval_count / evaluations)
                                                if evaluations else 0.0),
            "nonidentity_bead_eval_count": int(self._nonidentity_map_bead_count),
            "max_raw_radius_seen": float(self._max_raw_radius),
            "max_physical_radius_seen": float(self._max_physical_radius),
            "identity_radius": (float(allele_models.IDENTITY_RADIUS)
                                 if self.model_id == "C2-map" else None),
            "counts_only_evaluate_calls": True,
            "line_search_probes_included": True,
            "used_for_selection_or_budget": False,
        }

    def backend_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "implementation": self.backend,
            "variant": self.model_id,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "tile_rows": self.tile_rows,
            "pair_storage": "implicit_complete_upper_triangle_plus_sparse_observed_csr",
            "n_loci": self.data.n_loci,
            "observed_nonzero_pairs": int(len(self.data.observed_flat)),
            "full_grid_pairs": self.data.n_pairs,
            "data_resident_on_device": self.device.type == "cuda",
            "precision": "float64_no_mixed_precision_no_sampling",
            "reduction": ("028_cuda_ordered_row_warp_double_reduction"
                          if self.backend == "028_fused_cuda" else "Torch tiled double reduction"),
            "source_028_gpu_backend_sha256": sha256_file(SOURCE_028 / "gpu_backend.py"),
            "source_028_fused_objective_sha256": sha256_file(SOURCE_028 / "fused" / "fused_objective.py"),
            "source_028_cuda_kernel_sha256": sha256_file(SOURCE_028 / "fused" / "cuda_pair_objective.cu"),
        }
        if self.backend == "028_fused_cuda":
            result.update(self._backend.backend_metadata())
        return result

    @staticmethod
    def source_manifest() -> dict[str, Any]:
        paths = {
            "028_gpu_backend": SOURCE_028 / "gpu_backend.py",
            "028_fused_objective": SOURCE_028 / "fused" / "fused_objective.py",
            "028_cuda_pair_kernel": SOURCE_028 / "fused" / "cuda_pair_objective.cu",
        }
        return {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()}


def make_objective(data: contact_model.AggregatedContacts, model_id: str,
                   device: str = "cuda", tile_rows: int = 32,
                   use_fused: bool = True, diagnostics: bool = True) -> GPUVariantObjective:
    """Construct one of the five frozen contracts without mutating input data."""
    return GPUVariantObjective(data, model_id=model_id, device=device, tile_rows=tile_rows,
                               use_fused=use_fused, diagnostics=diagnostics)


def cuda_probe() -> dict[str, Any]:
    result: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if not result["cuda_available"]:
        result.update({"status": "unavailable", "error": "No CUDA GPUs are available"})
        return result
    result["status"] = "available"
    result["devices"] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        result["devices"].append({"index": index, "name": props.name,
                                   "total_memory": int(props.total_memory)})
    return result


__all__ = ["GPUVariantObjective", "SparseView", "VARIANTS", "cuda_probe", "make_objective"]
