"""链内多尺度线性预条件 P = I + (I+5^2 L)^-1 + (I+20^2 L)^-1。

L 是每条染色体 1Mb 链的 path-graph Neumann 离散 Laplacian，长度为该染色体全部 bins。
用 scipy.fft 的 DCT-II(norm="ortho") 对角化：lambda_k = 4*sin^2(pi*k/(2n))。

本模块不做任何优化、不冻结内部形状、不按 intra/inter 拆分。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.fft import dct, idct

SCALES = (5.0, 20.0)


def neumann_laplacian(n: int) -> np.ndarray:
    """显式 path-graph Neumann Laplacian，只用于小 fixture  Parity 检查。"""
    if n <= 0:
        raise ValueError("n must be positive")
    matrix = np.zeros((n, n), dtype=np.float64)
    for index in range(n - 1):
        matrix[index, index] += 1.0
        matrix[index + 1, index + 1] += 1.0
        matrix[index, index + 1] -= 1.0
        matrix[index + 1, index] -= 1.0
    return matrix


def neumann_eigenvalues(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return 4.0 * np.sin(np.pi * k / (2.0 * n)) ** 2


def multiscale_symbol(n: int, scales: Sequence[float] = SCALES) -> np.ndarray:
    """p_k = 1 + sum_s 1/(1 + s^2 * lambda_k)，全部严格为正。"""
    lam = neumann_eigenvalues(n)
    symbol = np.ones(n, dtype=np.float64)
    for scale in scales:
        symbol = symbol + 1.0 / (1.0 + float(scale) ** 2 * lam)
    if not np.all(symbol > 0.0):
        raise AssertionError("multiscale symbol must be strictly positive")
    return symbol


class ChainPreconditioner:
    """按染色体分块作用在 (2, n_loci, 3) raw-y 数组上的 P 与 P^-1。"""

    def __init__(self, data: Any, scales: Sequence[float] = SCALES, identity: bool = False):
        self.data = data
        self.scales = tuple(float(value) for value in scales)
        self.identity = bool(identity)
        self.slices = [data.chromosome_slice(index) for index in range(len(data.chromosome_names))]
        self.lengths = [int(slc.stop - slc.start) for slc in self.slices]
        self.symbols = [np.ones(length) if self.identity else multiscale_symbol(length, self.scales)
                        for length in self.lengths]
        if sum(self.lengths) != int(data.n_loci):
            raise AssertionError("chromosome slices do not tile the locus grid")
        self.min_symbol = float(min(float(value.min()) for value in self.symbols))
        self.max_symbol = float(max(float(value.max()) for value in self.symbols))

    def _apply_symbol(self, values: np.ndarray, divide: bool) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (2, int(self.data.n_loci), 3):
            raise ValueError("preconditioner input must have shape (2, n_loci, 3)")
        result = np.array(array, dtype=np.float64, copy=True, order="C")
        for slc, symbol in zip(self.slices, self.symbols):
            block = result[:, slc, :]
            spectra = dct(block, type=2, axis=1, norm="ortho")
            factor = (1.0 / symbol) if divide else symbol
            spectra = spectra * factor[None, :, None]
            result[:, slc, :] = idct(spectra, type=2, axis=1, norm="ortho")
        return result

    def P_apply(self, y: np.ndarray) -> np.ndarray:
        return self._apply_symbol(y, divide=False)

    def Pinv_apply(self, z: np.ndarray) -> np.ndarray:
        return self._apply_symbol(z, divide=True)

    def dense_matrix(self, chromosome_index: int) -> np.ndarray:
        """小 fixture 用：显式构造 P = I + Σ_s (I + s^2 L)^-1。"""
        n = self.lengths[chromosome_index]
        laplacian = neumann_laplacian(n)
        eye = np.eye(n)
        matrix = np.eye(n)
        for scale in self.scales:
            matrix = matrix + np.linalg.inv(eye + float(scale) ** 2 * laplacian)
        return matrix

    def diagnostics(self) -> dict[str, Any]:
        return {
            "preconditioner": "chain_multiscale_neumann_path_graph",
            "scales": list(self.scales),
            "identity_mode": bool(self.identity),
            "chromosome_lengths": list(self.lengths),
            "symbol_min": self.min_symbol,
            "symbol_max": self.max_symbol,
            "symbol_minus_one_max": self.max_symbol - 1.0,
            "definition": "P = I + (I + 25L)^-1 + (I + 400L)^-1 with DCT-II ortho eigenbasis",
            "all_frequencies_present": True,
            "internal_shape_frozen": False,
            "intra_inter_split": False,
        }


__all__ = ["ChainPreconditioner", "neumann_laplacian", "neumann_eigenvalues", "multiscale_symbol", "SCALES"]
