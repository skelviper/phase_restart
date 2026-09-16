"""Exact fused CUDA objective for the independent 020 backend.

The public class deliberately mirrors ``gpu_backend.TorchObjective`` while
moving the complete eligible-pair work into a workspace-local CUDA extension.
The extension never receives phase-bearing data or reference coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.cpp_extension import load

from gpu_backend import (  # type: ignore
    BackendError,
    EPSILON,
    P_FLOOR,
    P_PRIOR_STRENGTH,
    TorchObjective as _TorchObjective,
    p_from_q,
    sphere_forward,
    sphere_pullback,
)


_SOURCE = Path(__file__).resolve().parent / "cuda_pair_objective.cu"
_RUN = Path(__file__).resolve().parents[2]
_BUILD = _RUN / "work" / "fused_build"
_EXTENSION_NAME = "dsh_fused_pair_objective"
_EXTENSION: Any | None = None


@dataclass(frozen=True)
class SparseCSR:
    """Upper-triangle observed edges in outgoing and incoming row order."""

    out_ptr: np.ndarray
    out_j: np.ndarray
    out_count: np.ndarray
    in_ptr: np.ndarray
    in_i: np.ndarray
    in_count: np.ndarray


def _source_sha256() -> str:
    digest = hashlib.sha256()
    with _SOURCE.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_sparse_csr(data) -> SparseCSR:
    """Convert sorted sparse triangular indices without constructing the grid."""
    n = int(data.n_loci)
    flat = np.asarray(data.observed_flat, dtype=np.int64)
    count = np.asarray(data.observed_counts, dtype=np.int64)
    if flat.ndim != 1 or count.shape != flat.shape:
        raise BackendError("sparse observed arrays have incompatible shapes")
    if len(flat) and (np.any(np.diff(flat) <= 0) or np.any(count <= 0)):
        raise BackendError("sparse observed edges must be sorted and positive")
    out_i = np.empty(len(flat), dtype=np.int64)
    out_j = np.empty(len(flat), dtype=np.int64)
    if len(flat):
        if n < 2:
            raise BackendError("observed edge exists on a one-locus grid")
        rows = np.arange(n - 1, dtype=np.int64)
        row_start = rows * (2 * np.int64(n) - rows - 1) // 2
        out_i = np.searchsorted(row_start, flat, side="right").astype(np.int64) - 1
        if np.any(out_i < 0) or np.any(out_i >= n - 1):
            raise BackendError("observed triangular index maps outside an upper row")
        out_j = out_i + 1 + flat - row_start[out_i]
        if np.any(out_j <= out_i) or np.any(out_j >= n):
            raise BackendError("observed triangular index maps outside an edge")
        reconstructed = out_i * (2 * np.int64(n) - out_i - 1) // 2 + (out_j - out_i - 1)
        if not np.array_equal(reconstructed, flat):
            raise BackendError("observed triangular inversion failed")
    out_i32 = out_i.astype(np.int32, copy=False)
    out_j32 = out_j.astype(np.int32, copy=False)
    out_count64 = count.astype(np.int64, copy=False)
    out_ptr = np.zeros(n + 1, dtype=np.int64)
    if len(out_i):
        out_ptr[1:] = np.cumsum(np.bincount(out_i, minlength=n), dtype=np.int64)
    incoming_order = np.argsort(out_j, kind="stable") if len(out_j) else np.empty(0, dtype=np.int64)
    in_i32 = out_i32[incoming_order]
    in_count64 = out_count64[incoming_order]
    in_ptr = np.zeros(n + 1, dtype=np.int64)
    if len(in_i32):
        in_j = out_j32[incoming_order].astype(np.int64, copy=False)
        in_ptr[1:] = np.cumsum(np.bincount(in_j, minlength=n), dtype=np.int64)
    return SparseCSR(
        out_ptr=np.ascontiguousarray(out_ptr),
        out_j=np.ascontiguousarray(out_j32),
        out_count=np.ascontiguousarray(out_count64),
        in_ptr=np.ascontiguousarray(in_ptr),
        in_i=np.ascontiguousarray(in_i32),
        in_count=np.ascontiguousarray(in_count64),
    )


def load_fused_extension(*, verbose: bool = False):
    """Build/load the CUDA extension only in the 028 workspace."""
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not _SOURCE.is_file():
        raise BackendError(f"fused CUDA source is missing: {_SOURCE}")
    _BUILD.mkdir(parents=True, exist_ok=True)
    # These are process-local build settings; no package or shared environment is changed.
    os.environ["TORCH_EXTENSIONS_DIR"] = str(_BUILD)
    os.environ["MAX_JOBS"] = "2"
    _EXTENSION = load(
        name=_EXTENSION_NAME,
        sources=[str(_SOURCE)],
        build_directory=str(_BUILD),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _EXTENSION


def _p_from_q_host(q: float) -> tuple[float, float]:
    """Stable scalar equivalent of archived020.p_from_q."""
    if q >= 0.0:
        z = math.exp(-q)
        sigmoid = 1.0 / (1.0 + z)
    else:
        z = math.exp(q)
        sigmoid = z / (1.0 + z)
    span = 1.0 - 2.0 * P_FLOOR
    return P_FLOOR + span * sigmoid, span * sigmoid * (1.0 - sigmoid)


class FusedObjective(_TorchObjective):
    """Drop-in float64 objective using implicit ordered CUDA rows and sparse CSR."""

    def __init__(self, data, device: str = "cuda", tile_rows: int = 32,
                 dtype: torch.dtype = torch.float64):
        if dtype != torch.float64:
            raise BackendError("fused objective is frozen to float64")
        requested = torch.device(device)
        if requested.type != "cuda":
            raise BackendError("fused objective requires a CUDA device")
        if not torch.cuda.is_available():
            raise BackendError("CUDA requested but torch.cuda.is_available() is false")
        data.assert_consistent()
        self._extension = load_fused_extension()
        super().__init__(data, device=device, tile_rows=tile_rows, dtype=dtype)
        self._chromosome_i32 = torch.as_tensor(
            np.asarray(data.locus_chromosome, dtype=np.int32),
            dtype=torch.int32, device=self.device)
        csr = _build_sparse_csr(data)
        self._out_ptr = torch.as_tensor(csr.out_ptr, dtype=torch.int64, device=self.device)
        self._out_j = torch.as_tensor(csr.out_j, dtype=torch.int32, device=self.device)
        self._out_count = torch.as_tensor(csr.out_count, dtype=torch.int64, device=self.device)
        self._in_ptr = torch.as_tensor(csr.in_ptr, dtype=torch.int64, device=self.device)
        self._in_i = torch.as_tensor(csr.in_i, dtype=torch.int32, device=self.device)
        self._in_count = torch.as_tensor(csr.in_count, dtype=torch.int64, device=self.device)
        positive_diag = np.asarray(data.diag_counts[data.diag_counts > 0], dtype=np.float64)
        self._diag_nll = float(
            np.sum(positive_diag - positive_diag * np.log(positive_diag))
            + data.diag_factorial_sum)
        self._group_cis = float(data.raw_cis_offdiag)
        self._group_inter = float(data.raw_inter)
        self._normalizer = float(data.raw_records)
        self._csr = csr
        self._source_hash = _source_sha256()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def synchronize(self) -> None:
        self._sync()

    def _pair_terms(self, x: torch.Tensor, p: float):
        full = self._extension.full_terms(
            x, self._exposure, self._chromosome_i32, float(p),
            float(self.data.r0), float(self.data.repulsion_threshold))
        sparse = self._extension.sparse_terms(
            x, self._exposure, self._chromosome_i32,
            self._out_ptr, self._out_j, self._out_count,
            self._in_ptr, self._in_i, self._in_count,
            float(p), float(self.data.r0))
        return full, sparse

    def evaluate(self, theta: np.ndarray, need_gradient: bool = True):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,) or not np.all(np.isfinite(theta)):
            raise BackendError("theta has wrong shape or nonfinite values")
        theta_t = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        y = theta_t[:-1].reshape(2, self.data.n_loci, 3)
        x = sphere_forward(y)
        p_value, dpdq_value = _p_from_q_host(float(theta[-1]))
        full, sparse = self._pair_terms(x, p_value)
        full_summary, grad_r_cis, grad_r_inter, grad_rep = full
        sparse_summary, grad_observed = sparse

        # Ordered full rows visit each unordered edge twice. Coordinate gradients
        # are central-endpoint rows and therefore are not divided by two.
        sum_r_cis = 0.5 * full_summary[0]
        sum_r_inter = 0.5 * full_summary[1]
        d_r_cis_dp = 0.5 * full_summary[2]
        repulsion = (0.5 * full_summary[3] + full_summary[4]) / float(2 * self.data.n_loci)
        positive_zero = torch.zeros((), dtype=self.dtype, device=self.device)
        if self._group_cis > 0.0:
            conditional_cis = self._group_cis * torch.log(sum_r_cis) - sparse_summary[0]
            alpha_cis = self._group_cis / sum_r_cis
        else:
            conditional_cis = positive_zero
            alpha_cis = positive_zero
        if self._group_inter > 0.0:
            conditional_inter = self._group_inter * torch.log(sum_r_inter) - sparse_summary[1]
            alpha_inter = self._group_inter / sum_r_inter
        else:
            conditional_inter = positive_zero
            alpha_inter = positive_zero
        conditional = conditional_cis + conditional_inter
        diag = torch.as_tensor(self._diag_nll, dtype=self.dtype, device=self.device)
        count_raw = conditional + diag
        count_normalized = count_raw / self._normalizer
        p = torch.as_tensor(p_value, dtype=self.dtype, device=self.device)
        dpdq = torch.as_tensor(dpdq_value, dtype=self.dtype, device=self.device)
        p_prior = -P_PRIOR_STRENGTH * torch.log(p * (1.0 - p))
        p_prior_derivative = P_PRIOR_STRENGTH * (1.0 / (1.0 - p) - 1.0 / p)
        bond, bond_gradient = self._bond_layer(x)
        bend, bend_gradient = self._bend_layer(x)
        total = count_normalized + p_prior + bond + repulsion + 0.01 * bend

        # All components cross the device boundary as one packed transfer.
        packed = torch.stack((
            sum_r_cis, sum_r_inter,
            sparse_summary[0], sparse_summary[1],
            conditional, diag, count_raw, count_normalized,
            p, p_prior, bond, repulsion, bend, total,
        )).detach().cpu().numpy()
        self._sync()
        if not np.all(np.isfinite(packed)):
            raise FloatingPointError("nonfinite fused objective components")
        components: dict[str, Any] = {
            "sum_rate_cis_offdiag": float(packed[0]),
            "sum_rate_inter": float(packed[1]),
            "observed_log_rate_cis_offdiag": float(packed[2]),
            "observed_log_rate_inter": float(packed[3]),
            "conditional_nll_raw": float(packed[4]),
            "conditional_factorial_constant_omitted": float(self.data.conditional_factorial_constant),
            "diag_factorial_sum_included": float(self.data.diag_factorial_sum),
            "count_factorial_status": "integer_count_constants_recorded",
            "diag_profiled_nll_raw": float(packed[5]),
            "count_nll_raw": float(packed[6]),
            "count_nll_normalized": float(packed[7]),
            "p": float(packed[8]),
            "p_prior": float(packed[9]),
            "bond": float(packed[10]),
            "repulsion": float(packed[11]),
            "bend": float(packed[12]),
            "total": float(packed[13]),
            "repulsion_threshold_l0": 0.7,
        }
        if not need_gradient:
            return components["total"], None, components

        count_gradient_x = (alpha_cis * grad_r_cis + alpha_inter * grad_r_inter - grad_observed)
        count_gradient_x = count_gradient_x / self._normalizer
        count_gradient_p = (alpha_cis * d_r_cis_dp - sparse_summary[2]) / self._normalizer
        combined_x = count_gradient_x + bond_gradient + grad_rep / float(2 * self.data.n_loci) + 0.01 * bend_gradient
        gradient_y = sphere_pullback(y, combined_x)
        gradient_q = dpdq * (count_gradient_p + p_prior_derivative)
        gradient = torch.cat((gradient_y.reshape(-1), gradient_q.reshape(1)))
        gradient_np = gradient.detach().cpu().numpy().astype(np.float64, copy=False)
        self._sync()
        if not np.all(np.isfinite(gradient_np)):
            raise FloatingPointError("nonfinite fused analytic gradient")
        return components["total"], gradient_np.copy(), components

    def value_and_grad(self, theta: np.ndarray):
        value, gradient, components = self.evaluate(theta, need_gradient=True)
        self._cache_theta = np.asarray(theta, dtype=np.float64).copy()
        self._cache_value = float(value)
        self._cache_gradient = gradient.copy()
        self._cache_components = dict(components)
        return value, gradient

    def components(self, theta: np.ndarray) -> dict[str, Any]:
        theta = np.asarray(theta, dtype=np.float64)
        if self._cache_theta is not None and np.array_equal(theta, self._cache_theta):
            return dict(self._cache_components or {})
        return self.evaluate(theta, need_gradient=False)[2]

    def coordinates_and_p(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_parameters,) or not np.all(np.isfinite(theta)):
            raise BackendError("theta has wrong shape or nonfinite values")
        theta_t = torch.as_tensor(theta, dtype=self.dtype, device=self.device)
        y = theta_t[:-1].reshape(2, self.data.n_loci, 3)
        x = sphere_forward(y)
        coordinates = x.detach().cpu().numpy().astype(np.float64, copy=False)
        self._sync()
        p_value, _ = _p_from_q_host(float(theta[-1]))
        return coordinates.copy(), p_value

    def backend_metadata(self) -> dict[str, Any]:
        """Return auditable implementation and current device metadata."""
        props = torch.cuda.get_device_properties(self.device)
        device_index = torch.cuda.current_device() if self.device.index is None else self.device.index
        return {
            "implementation": "workspace_local_cuda_fused_ordered_rows_csr",
            "extension_name": _EXTENSION_NAME,
            "source": str(_SOURCE),
            "source_sha256": self._source_hash,
            "build_directory": str(_BUILD),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device": {"index": device_index, "name": props.name,
                       "total_memory": int(props.total_memory)},
            "dtype": str(self.dtype),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "observed_nonzero_pairs": int(len(self._csr.out_j)),
            "pair_index_storage": "implicit_ordered_rows_plus_sparse_two_direction_csr",
            "global_coordinate_atomics": False,
            "fast_math": False,
        }


__all__ = ["FusedObjective", "SparseCSR", "load_fused_extension"]
