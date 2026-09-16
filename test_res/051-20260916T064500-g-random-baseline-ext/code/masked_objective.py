"""C 组（reference-beads support ablation）的逐 copy 珠子支持适配器（v2，重写版）。

授权范围：训练进程只接受 ``support.npz`` 的布尔 mask（每 copy 每层珠子是否存在），
不接触 reference xyz、不接触 phase。

科学要求（主侧 2026-09-16 明确）：

* 缺失珠子必须**真正退出**目标，不能用“坐标置零 + 只换分母”冒充：
  - 逐 copy 的四项 kernel 及其梯度按存在 mask 乘 0/1（置零坐标下 K(0)=1，不是 0，
    所以 kernel 必须显式 mask，`_kernel_block` 是唯一入口）；same/cross 由 masked
    四项重算；
  - bond 只保留“真实存在且原 bp 相邻”的 edge；bend 只保留三个真实存在且相邻的
    triple；repulsion 只保留两端在该 copy-pair family 都存在的 term（含两条对角项）；
  - 每个归一化分母按真实有效计数重算，不是把父类 full-grid 值乘比例。
* G count 只在**筛好的数据**上运行原 G 的 `_count_and_gradient`：locus union 保留、
  pair 在两端两 copy 都存在时保留、diag 用真实 `diag_counts` 按同一 pair 集合过滤、
  Nraw/Noff 用 filtered 值、e 用 production 公式在有效端点上重算并在 union lanes 上
  归一到均值 1，再在完整 grid 上补零（缺失位置 e=0，恰好不参与）。
* 缺失珠子的 coordinate 是惰性的：移到任何位置都不改变目标与 active 梯度。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from shared_capture_objective import EPSILON, SharedCaptureObjective  # noqa: E402


class PerCopySupportObjective(SharedCaptureObjective):
    """G 目标 + 逐 copy 珠子支持；缺失珠子真正退出 count/bond/bend/repulsion。"""

    def __init__(self, data: Any, model_id: str, *, mask: np.ndarray,
                 filtered: Any, exposure_valid: np.ndarray, **kwargs: Any):
        super().__init__(filtered, model_id, **kwargs)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (2, int(data.n_loci)):
            raise ValueError("per-copy support mask must have shape (2, n_loci)")
        if not mask.any():
            raise ValueError("per-copy support mask is empty")
        if filtered.n_loci != data.n_loci:
            raise ValueError("filtered data must keep the full-grid locus layout")
        self.full_data = data
        self.data = filtered
        self.support_mask = mask.copy()
        self._mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        self._n_valid = (int(mask[0].sum()), int(mask[1].sum()))
        self._n_shared = int(np.count_nonzero(mask[0] & mask[1]))
        self._tile_rows = int(self._physics._backend.tile_rows)
        self._lane_np = mask.any(axis=0)
        self._lane = torch.as_tensor(self._lane_np, dtype=torch.bool, device=self.device)
        self._lane_mask = self._lane_np
        self._n_union = int(np.count_nonzero(self._lane_mask))
        self._epsilon = float(EPSILON)
        self._r0sq = float(data.r0 * data.r0)
        self._repulsion_threshold = 0.7 * float(data.l0)
        self._l0 = float(data.l0)

        pair_i = np.asarray(data.pair_i, dtype=np.int64)
        pair_j = np.asarray(data.pair_j, dtype=np.int64)
        cis = np.asarray(data.cis_pair, dtype=bool)
        counts_full = np.asarray(data.counts, dtype=np.float64)
        counts = np.asarray(filtered.counts, dtype=np.float64)
        lane_bool = mask.any(axis=0)
        eligible = lane_bool[pair_i] & lane_bool[pair_j]
        if not np.array_equal(counts > 0.0, (counts_full > 0.0) & eligible):
            raise RuntimeError("filtered counts are inconsistent with the lane-union support")
        self._eligible = eligible
        self._cis_np = cis
        self._cis = torch.as_tensor(cis, dtype=torch.bool, device=self.device)
        self._diag_valid_np = np.asarray(filtered.diag_counts, dtype=np.float64)
        self._diag_valid = torch.as_tensor(self._diag_valid_np, dtype=self.dtype, device=self.device)
        self._group_totals = (float(filtered.raw_cis_offdiag), float(filtered.raw_inter))
        self._removed_counts = (float(data.raw_cis_offdiag) - self._group_totals[0],
                                float(data.raw_inter) - self._group_totals[1])
        self._removed_pairs = (int(np.count_nonzero(cis & ~eligible)),
                               int(np.count_nonzero(~cis & ~eligible)))
        self._observed_removed = int(np.count_nonzero((counts_full > 0.0) & ~eligible))

        bond_terms = 0
        bend_terms = 0
        bond_full = 0
        bend_full = 0
        for chrom in range(len(data.chromosome_names)):
            slc = data.chromosome_slice(chrom)
            n_bins = int(data.n_bins[chrom])
            bond_full += 2 * max(0, n_bins - 1)
            bend_full += 2 * max(0, n_bins - 2)
            for copy in (0, 1):
                m = mask[copy, slc]
                bond_terms += int(np.count_nonzero(m[:-1] & m[1:]))
                if n_bins >= 3:
                    bend_terms += int(np.count_nonzero(m[:-2] & m[1:-1] & m[2:]))
        self._bond_terms, self._bend_terms = int(bond_terms), int(bend_terms)
        self._bond_terms_full, self._bend_terms_full = int(bond_full), int(bend_full)
        if self._bond_terms <= 0 or self._bend_terms <= 0:
            raise ValueError("support mask leaves no bond/bend terms")

        n_loci = int(data.n_loci)
        # repulsion 归一化沿用原公式的“每个 physical bead 平均”：分母 = 有效珠子数 Nactive
        # （all-ones 时 Nactive = 2*n_loci，与冻结 backend 逐位一致）。审计里另给真实
        # active term 数 = 同 copy 无序 pair + 跨 copy 无序 pair（含同 locus 的 A/B 对角项）。
        self._n_active_beads = int(self._n_valid[0] + self._n_valid[1])
        within = self._n_valid[0] * (self._n_valid[0] - 1) // 2 + self._n_valid[1] * (
            self._n_valid[1] - 1) // 2
        cross_pairs = self._n_valid[0] * self._n_valid[1]
        self._repulsion_active_terms = float(within + cross_pairs)
        self._repulsion_terms = float(self._n_active_beads) if self._n_active_beads > 0 else 1.0
        self._repulsion_terms_full = float(2 * n_loci)
        self._repulsion_valid_aa = None
        self._repulsion_valid_bb = None

        exposure_valid = np.asarray(exposure_valid, dtype=np.float64).copy()
        if exposure_valid.shape != (n_loci,) or not np.all(np.isfinite(exposure_valid)):
            raise ValueError("exposure_valid must be a finite length-n_loci vector")
        if np.any(exposure_valid[~self._lane_mask] != 0.0):
            raise ValueError("exposure_valid must be zero on missing lanes")
        if not np.all(exposure_valid[self._lane_mask] > 0.0):
            raise ValueError("exposure_valid must be positive on kept lanes")
        exposure_valid[self._lane_mask] /= float(exposure_valid[self._lane_mask].mean())
        self._e_valid = exposure_valid
        self._e_valid_t = torch.as_tensor(exposure_valid, dtype=self.dtype, device=self.device)
        # 父类要求 fixed e 全正且 full-grid 均值 1。这里只放形状占位；真正 count 由本类
        # 把 _e_valid_t 交给父类 _count_and_gradient（唯一消费 e 的入口）。
        placeholder = exposure_valid.copy()
        placeholder[~self._lane_mask] = float(exposure_valid[self._lane_mask].mean())
        placeholder = placeholder / placeholder.mean()
        self._e_placeholder = torch.as_tensor(placeholder, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ audit
    def support_audit(self) -> dict[str, Any]:
        return {
            "n_loci": int(self.full_data.n_loci),
            "n_valid_loci_copyA": self._n_valid[0],
            "n_valid_loci_copyB": self._n_valid[1],
            "n_union_loci": self._n_union,
            "n_shared_loci": self._n_shared,
            "n_missing_loci_copyA": int(self.full_data.n_loci - self._n_valid[0]),
            "n_missing_loci_copyB": int(self.full_data.n_loci - self._n_valid[1]),
            "bond_terms_valid": self._bond_terms, "bond_terms_full": self._bond_terms_full,
            "bend_terms_valid": self._bend_terms, "bend_terms_full": self._bend_terms_full,
            "repulsion_active_terms": self._repulsion_active_terms,
            "repulsion_normalizer_nactive": self._repulsion_terms,
            "repulsion_normalizer_full": self._repulsion_terms_full,
            "n_active_beads": self._n_active_beads,
            "repulsion_terms_full": self._repulsion_terms_full,
            "group_total_valid_cis": self._group_totals[0],
            "group_total_valid_inter": self._group_totals[1],
            "count_denominator_valid_offdiag": float(sum(self._group_totals)),
            "filtered_raw_records": float(self.data.raw_records),
            "filtered_diag_bins_positive": int(np.count_nonzero(self._diag_valid_np > 0.0)),
            "original_raw_records": float(self.full_data.raw_records),
            "original_cis_offdiag": float(self.full_data.raw_cis_offdiag),
            "original_inter": float(self.full_data.raw_inter),
            "removed_counts_cis": self._removed_counts[0],
            "removed_counts_inter": self._removed_counts[1],
            "removed_pairs_cis": self._removed_pairs[0],
            "removed_pairs_inter": self._removed_pairs[1],
            "observed_positive_pairs_removed": self._observed_removed,
            "exposure_mean_valid": 1.0,
            "exposure_min_valid": float(self._e_valid[self._lane_mask].min()),
            "exposure_max_valid": float(self._e_valid[self._lane_mask].max()),
        }

    # ------------------------------------------------------------------ kernel
    def _kernel_block(self, x: torch.Tensor, i: torch.Tensor, j: torch.Tensor,
                      p: torch.Tensor, with_gradient: bool):
        """按逐 copy 存在 mask 计算四项 kernel 及其梯度；same/cross 由 masked 值重算。"""
        del p
        mask_i = self._mask[:, i]
        mask_j = self._mask[:, j]
        r0sq = self._r0sq
        eps = self._epsilon
        one_minus = 1.0 - eps
        kernels = []
        gradients = []
        for copy_i, copy_j in ((0, 0), (0, 1), (1, 0), (1, 1)):
            delta = x[copy_i, i] - x[copy_j, j]
            base = 1.0 + torch.sum(delta * delta, dim=-1) / r0sq
            active = (mask_i[copy_i] & mask_j[copy_j]).to(x.dtype)
            kernels.append((eps + one_minus * base.pow(-2)) * active)
            if with_gradient:
                gradient = (-4.0 * one_minus / r0sq) * delta * base.pow(-3).unsqueeze(-1)
                gradients.append(gradient * active.unsqueeze(-1))
        kaa, kab, kba, kbb = kernels
        if not with_gradient:
            return (kaa, kab, kba, kbb, kaa + kbb, kab + kba), (None, None, None, None)
        gaa, gab, gba, gbb = gradients
        return (kaa, kab, kba, kbb, kaa + kbb, kab + kba), (gaa, gab, gba, gbb)

    # ------------------------------------------------------------------ physics
    def _bond_layer(self, x: torch.Tensor):
        gradient = torch.zeros_like(x)
        value = torch.zeros((), dtype=self.dtype, device=self.device)
        for chromosome in range(len(self.data.chromosome_names)):
            start = int(self.data.offsets[chromosome])
            stop = start + int(self.data.n_bins[chromosome])
            if stop - start < 2:
                continue
            for copy in (0, 1):
                delta = x[copy, start + 1:stop] - x[copy, start:stop - 1]
                distance = torch.sqrt(torch.sum(delta * delta, dim=1))
                scaled = distance / self._l0
                low = torch.relu(0.75 - scaled)
                high = torch.relu(scaled - 1.25)
                active_t = (self._mask[copy, start + 1:stop]
                            & self._mask[copy, start:stop - 1]).to(self.dtype)
                value = value + ((low * low + high * high) * active_t).sum()
                derivative = (-2.0 * low + 2.0 * high) / self._l0
                coefficient = torch.where(distance > 0.0, derivative / distance,
                                          torch.zeros_like(distance)) * active_t
                term = coefficient.unsqueeze(-1) * delta
                index = torch.arange(stop - start - 1, device=self.device)
                gradient[copy, start + 1:stop].index_add_(0, index, term)
                gradient[copy, start:stop - 1].index_add_(0, index, -term)
        return value / float(self._bond_terms), gradient / float(self._bond_terms)

    def _bend_layer(self, x: torch.Tensor):
        gradient = torch.zeros_like(x)
        value = torch.zeros((), dtype=self.dtype, device=self.device)
        l0sq = self._l0 * self._l0
        for chromosome in range(len(self.data.chromosome_names)):
            start = int(self.data.offsets[chromosome])
            stop = start + int(self.data.n_bins[chromosome])
            if stop - start < 3:
                continue
            for copy in (0, 1):
                second = (x[copy, start + 2:stop] - 2.0 * x[copy, start + 1:stop - 1]
                          + x[copy, start:stop - 2])
                active_t = (self._mask[copy, start + 2:stop]
                            & self._mask[copy, start + 1:stop - 1]
                            & self._mask[copy, start:stop - 2]).to(self.dtype)
                value = value + ((second * second).sum(dim=1) / l0sq * active_t).sum()
                term = (2.0 * second / l0sq) * active_t.unsqueeze(-1)
                gradient[copy, start:stop - 2] = gradient[copy, start:stop - 2] + term
                gradient[copy, start + 1:stop - 1] = gradient[copy, start + 1:stop - 1] - 2.0 * term
                gradient[copy, start + 2:stop] = gradient[copy, start + 2:stop] + term
        return value / float(self._bend_terms), gradient / float(self._bend_terms)

    def _repulsion_piece(self, delta: torch.Tensor):
        distance = torch.sqrt(torch.sum(delta * delta, dim=-1))
        hinge = torch.relu(1.0 - distance / self._repulsion_threshold)
        derivative_distance = -2.0 * hinge / self._repulsion_threshold
        coefficient = torch.where(distance > 0.0, derivative_distance / distance,
                                  torch.zeros_like(distance))
        return hinge * hinge, coefficient.unsqueeze(-1) * delta

    def _repulsion_layer(self, x: torch.Tensor):
        """原公式的四 family (i<j)（照冻结 backend 的 tile_rows 枚举）+ 同 locus A/B 对角项；
        每个 term 按其 copy 的珠子存在性保留，分母为有效珠子数 Nactive。"""
        gradient = torch.zeros_like(x)
        value = torch.zeros((), dtype=self.dtype, device=self.device)
        n_loci = int(self.data.n_loci)
        for start in range(0, max(0, n_loci - 1), self._tile_rows):
            stop = min(start + self._tile_rows, n_loci - 1)
            i, j, _cis, _counts, _exposure = self._physics._backend._pair_block_tensors(start, stop)
            mask_0 = self._mask[0].index_select(0, i)
            mask_0j = self._mask[0].index_select(0, j)
            mask_1 = self._mask[1].index_select(0, i)
            mask_1j = self._mask[1].index_select(0, j)
            lane_i = self._lane.index_select(0, i)
            lane_j = self._lane.index_select(0, j)
            for copy_i, copy_j in ((0, 0), (0, 1), (1, 0), (1, 1)):
                if copy_i == 0 and copy_j == 0:
                    active = mask_0 & mask_0j
                elif copy_i == 1 and copy_j == 1:
                    active = mask_1 & mask_1j
                elif copy_i == 0:
                    active = mask_0 & mask_1j
                else:
                    active = mask_1 & mask_0j
                piece, term = self._repulsion_piece(x[copy_i, i] - x[copy_j, j])
                active_f = active.to(self.dtype)
                value = value + (piece * active_f).sum()
                term = term * active_f.unsqueeze(-1)
                gradient[copy_i].index_add_(0, i, term)
                gradient[copy_j].index_add_(0, j, -term)
        piece, term = self._repulsion_piece(x[0] - x[1])
        active_t = self._mask[0].to(self.dtype) * self._mask[1].to(self.dtype)
        value = value + (piece * active_t).sum()
        term = term * active_t.unsqueeze(-1)
        gradient[0] = gradient[0] + term
        gradient[1] = gradient[1] - term
        return value / self._repulsion_terms, gradient / self._repulsion_terms

    def _physics_layers(self, x: torch.Tensor):
        bond, bond_gradient = self._bond_layer(x)
        repulsion, repulsion_gradient = self._repulsion_layer(x)
        bend, bend_gradient = self._bend_layer(x)
        return bond, bond_gradient, repulsion, repulsion_gradient, bend, bend_gradient

    # ------------------------------------------------------------------ evaluate
    def _count_and_gradient(self, x: torch.Tensor, p: torch.Tensor, k: torch.Tensor,
                            e, need_gradient: bool):
        """父类传入的 e 只是形状占位；这里换成真实有效支持 e（缺失 lane 为 0）。"""
        return super()._count_and_gradient(x, p, k, self._e_valid_t, need_gradient)

    def evaluate(self, theta: np.ndarray, need_gradient: bool = True):
        backup = self._fixed_e
        self._fixed_e = self._e_placeholder
        try:
            return super().evaluate(theta, need_gradient=need_gradient)
        finally:
            self._fixed_e = backup
