#!/usr/bin/env python
"""049 评价代码自检：用**合成小对象**跑通 evaluation 数据通路，不读 reference、不读 mask snapshot。

这不是科学运行：不产生任何 049 指标结论，输出仅用于证明代码路径（R2/inter/空间/bootstrap/null）
在真实 gate 之前已通过恒等式与守恒检查。合成对象规模为 20 chr / 228 loci（每 chr common pairs ≥ MIN_COMMON_PAIRS=20，190 中心距离与 760 copy 距离全通路）。
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval049_evaluate as ev  # noqa: E402
import eval049_lib as lib  # noqa: E402
import eval049_spatial as spatial  # noqa: E402


class TinyAggregate:
    """只提供 evaluation 侧用到的字段，避免触碰任何正式输入。"""

    def __init__(self) -> None:
        bins = [12, 13, 11, 10, 12, 11, 13, 12, 11, 10, 12, 13, 11, 12, 10, 11, 12, 13, 10, 11]
        self.chromosome_names = tuple("chr%d" % (i + 1) for i in range(len(bins)))
        self.n_bins = np.asarray(bins, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(bins)[:-1]]).astype(np.int64)
        self.chromosome_lengths = np.asarray([(n + 2) * 1_000_000 for n in bins], dtype=np.int64)
        self.bin_size = 1_000_000
        self.locus_bin = np.concatenate([np.arange(n, dtype=np.int64) for n in self.n_bins])
        self.n_loci = int(self.locus_bin.size)
        i, j = np.triu_indices(self.n_loci, k=1)
        self.pair_i, self.pair_j = i.astype(np.int64), j.astype(np.int64)
        chrom = np.concatenate([np.full(n, c, dtype=np.int64) for c, n in enumerate(self.n_bins)])
        self.cis_pair = chrom[self.pair_i] == chrom[self.pair_j]
        self.counts = np.ones(len(self.pair_i), dtype=np.int64)
        self.diag_counts = np.zeros(self.n_loci, dtype=np.int64)
        self.raw_records = 100

    def chromosome_slice(self, index: int) -> slice:
        return slice(int(self.offsets[index]), int(self.offsets[index] + self.n_bins[index]))

    def chromosome_slices(self) -> list[slice]:
        return [self.chromosome_slice(i) for i in range(len(self.chromosome_names))]


def tiny_masks(data: TinyAggregate) -> dict[str, dict[str, object]]:
    masks = {}
    for ci, name in enumerate(data.chromosome_names):
        n = int(data.n_bins[ci])
        positions = (np.arange(3, n, dtype=np.int64) * lib.BIN_SIZE_BP)
        n_positions = len(positions)
        pair_i, pair_j = np.triu_indices(n_positions, k=1)
        common = np.ones(len(pair_i), dtype=bool)
        keep = np.zeros(n_positions, dtype=bool)
        keep[pair_i[common]] = True
        keep[pair_j[common]] = True
        masks[str(name)] = {"chromosome": str(name), "chromosome_index": ci, "positions": positions,
                            "pair_i": pair_i.astype(np.int64), "pair_j": pair_j.astype(np.int64),
                            "common": common, "valid_local_bins": np.flatnonzero(keep),
                            "n_bins": n_positions, "n_total_non_diagonal_pairs": int(len(pair_i)),
                            "n_common_pairs": int(common.sum())}
    return masks


def main() -> int:
    data = TinyAggregate()
    masks = tiny_masks(data)
    names = list(data.chromosome_names)
    rng = np.random.default_rng(11)
    reference = rng.normal(size=(2, data.n_loci, 3)) * 0.05
    lib.assert_inside_unit_ball(reference)
    cache = ev.build_inter_cache(data, masks, reference, frozen_expectations=False)
    per_chr_valid = spatial.per_chromosome_valid_indices(data, masks)
    valid = spatial.valid_global_bins(data, masks)
    expected_inter = int(np.count_nonzero((~data.cis_pair) & valid[data.pair_i] & valid[data.pair_j]))
    if cache["eligible_locus_pairs"] != expected_inter:
        raise AssertionError("inter pair denominator mismatch: %d != %d"
                             % (cache["eligible_locus_pairs"], expected_inter))

    identical = ev.dataset_metrics(reference, reference, data, masks, cache, per_chr_valid, names,
                                   permutation_draws=64)
    for metric_name in ("pearson", "spearman"):
        block = identical["r2"]["macro_equal_chromosome_weight_defined_only"][metric_name]
        assert abs(block["matched"] - 1.0) < 1e-12, (metric_name, block)
        assert block["cross"] < 1.0 and block["contrast"] > 0.0, (metric_name, block)
        assert abs(block["contrast"] - (block["matched"] - block["cross"])) < 1e-12
        row = identical["r2"]["per_chromosome"][0]["metrics"][metric_name]
        assert row["orientation"] == "direct" and not row["geometry_tie"]
        assert abs(row["rho"]["A_mat"] - 1.0) < 1e-12 and abs(row["rho"]["B_pat"] - 1.0) < 1e-12
    assert abs(identical["inter"]["pearson"] - 1.0) < 1e-12
    assert abs(identical["inter"]["spearman"] - 1.0) < 1e-12
    centers = identical["spatial"]["merged_chr_centers"]
    assert centers["n_distances"] == 190, centers["n_distances"]
    assert abs(centers["pearson"] - 1.0) < 1e-12 and abs(centers["normalized_stress"]) < 1e-12
    assert centers["top3_neighbors"]["mean_overlap_top3"] == float(min(3, len(names) - 1))
    # 每 chr 19 距离 profile：19/19 全定义且 rho=1（不得因 MIN_COMMON_PAIRS=20 变成 NA）
    for profile_key in ("per_chr_19_distance_profile_pearson", "per_chr_19_distance_profile_spearman"):
        profile = centers[profile_key]
        assert len(profile) == 20, len(profile)
        assert all(value is not None and math.isfinite(float(value)) and abs(float(value) - 1.0) < 1e-12
                   for value in profile.values()), (profile_key, profile)
    assert 0.0 < identical["spatial"]["label_permutation_null"]["one_sided_p_ge_observed"] <= 1.0
    copy_centers = identical["spatial"]["copy_centers"]
    primary = copy_centers["primary"]
    assert primary["n_distances"] == 760, primary["n_distances"]
    assert abs(primary["pearson"] - 1.0) < 1e-12
    assert abs(primary["procrustes_proper"]["rotation_det"] - 1.0) < 1e-9
    assert primary["procrustes_proper"]["normalized_aligned_rmsd"] < 1e-12
    assert copy_centers["unresolved_chromosomes"] == []

    # 交换全部 chr 的 A/B 拷贝：matched/cross 必须互换，contrast 符号翻转
    swapped = np.stack((reference[1], reference[0]), axis=0)
    swapped_metrics = ev.dataset_metrics(swapped, reference, data, masks, cache, per_chr_valid, names,
                                         permutation_draws=16)
    base_row = identical["r2"]["per_chromosome"][0]["metrics"]["pearson"]
    swap_row = swapped_metrics["r2"]["per_chromosome"][0]["metrics"]["pearson"]
    assert abs(swap_row["direct"] - base_row["swapped"]) < 1e-12
    assert abs(swap_row["swapped"] - base_row["direct"]) < 1e-12
    assert abs(swapped_metrics["inter"]["pearson"] - identical["inter"]["pearson"]) < 1e-12, "inter is A/B swap invariant"
    assert abs(swapped_metrics["spatial"]["merged_chr_centers"]["pearson"] - 1.0) < 1e-12, "merged centers are swap invariant"

    # 3DG 命名解析：同一组坐标分别写成 candidate(cNNx) 与 reference(chrN(mat/pat)) 两种文本，必须得到同一数组
    cand_tracks, ref_tracks = {}, {}
    for ci, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(ci)
        positions = np.asarray(data.locus_bin[slc], dtype=np.int64) * int(data.bin_size)
        for copy in (0, 1):
            cand_key = lib.track_name(ci, str(name), copy, "candidate")
            ref_key = lib.track_name(ci, str(name), copy, "reference")
            assert cand_key == "c%02d%s" % (ci + 1, "ab"[copy]), cand_key
            assert ref_key == "%s(%s)" % (name, ("mat", "pat")[copy]), ref_key
            cand_tracks[cand_key] = {int(p): reference[copy, slc.start + i] for i, p in enumerate(positions)}
            ref_tracks[ref_key] = {int(p): reference[copy, slc.start + i] for i, p in enumerate(positions)}
    cand_audit, ref_audit = {}, {}
    array_candidate = lib.three_dg_to_array(cand_tracks, data, track_mode="candidate", audit=cand_audit)
    array_reference = lib.three_dg_to_array(ref_tracks, data, track_mode="reference", audit=ref_audit)
    assert np.array_equal(array_candidate, array_reference), "candidate/reference naming must map identically"
    assert np.array_equal(array_candidate, reference)
    assert cand_audit["missing_track_count"] == 0 and ref_audit["missing_track_count"] == 0
    assert ref_audit["track_mode"] == "reference" and ref_audit["found_track_count"] == 2 * len(names)
    # 真实缺失：丢掉一个 reference track 后只应影响该 copy，不整体变 NaN
    partial_tracks = {k: v for k, v in ref_tracks.items() if k != lib.track_name(0, str(names[0]), 1, "reference")}
    partial = lib.three_dg_to_array(partial_tracks, data, track_mode="reference")
    assert np.isfinite(partial[0]).all() and not np.isfinite(partial[1, data.chromosome_slice(0)]).all()
    assert np.isfinite(partial[1, data.chromosome_slice(1)]).all()
    try:
        lib.three_dg_to_array(ref_tracks, data, track_mode="bogus")
        raise AssertionError("unknown track_mode must raise")
    except ValueError:
        pass

    # full-20 macro 规则：只让 1 个 chr 的 rho 非有限 → defined=19、fixed-20=NULL、defined-only 有值
    degenerate = reference.copy()
    slc0 = data.chromosome_slice(0)
    degenerate[:, slc0] = degenerate[:, slc0][:, :1]  # chr1 两条 copy 各自退化为单点 → 该 chr 距离恒定
    degenerate_metrics = ev.dataset_metrics(degenerate, reference, data, masks, cache, per_chr_valid, names,
                                            permutation_draws=8)
    deg_r2 = degenerate_metrics["r2"]
    for metric_name in ("pearson", "spearman"):
        assert deg_r2["defined_chromosome_counts"][metric_name]["matched"] == 19, \
            deg_r2["defined_chromosome_counts"][metric_name]
        assert deg_r2["macro_full_20_all_chromosomes_required"][metric_name]["matched"] is None
        assert deg_r2["macro_full_20_defined"][metric_name]["matched"] is False
        assert deg_r2["macro_equal_chromosome_weight_defined_only"][metric_name]["matched"] is not None
    deg_rows = ev.r2_rows("degenerate", "selfcheck", "n/a", deg_r2)[1]
    for row in deg_rows:
        assert row["matched"] is None and row["defined_only_matched"] is not None
        assert row["full_20_estimate_defined"] is False
        assert row["defined_chromosomes"] == 19
    # u0 的 margin 必须是 NA（tie 规则使 per-chr margin 未定义）
    zero_na, _ = lib.make_u_zero(reference)
    zero_na_metrics = ev.null_metrics(zero_na, reference, data, masks, cache, per_chr_valid, names)
    zero_macro = zero_na_metrics["r2"]["macro_full_20_all_chromosomes_required"]["pearson"]
    assert zero_macro["matched"] is not None and zero_macro["min_margin"] is None, zero_macro
    assert zero_na_metrics["r2"]["macro_equal_chromosome_weight_defined_only"]["pearson"]["min_margin"] is None

    # Procrustes 正确性：已知 proper rotation + translation + uniform scale 必须 RMSD≈0
    rng_rot = np.random.default_rng(4242)
    q, _ = np.linalg.qr(rng_rot.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1.0
    assert abs(np.linalg.det(q) - 1.0) < 1e-12
    base_points = rng_rot.normal(size=(25, 3))
    moved = 2.5 * (base_points @ q) + np.asarray([3.0, -1.5, 0.75])
    proc = spatial.procrustes_global(moved, base_points, allow_reflection=False)
    assert abs(proc["rotation_det"] - 1.0) < 1e-9, proc["rotation_det"]
    # cand=moved, ref=base ⇒ 最优 scale 是 1/2.5，旋转是 q 的转置（row-vector 约定）
    assert abs(proc["optimal_uniform_scale"] - 1.0 / 2.5) < 1e-9, proc["optimal_uniform_scale"]
    assert proc["aligned_rmsd"] < 1e-9, proc["aligned_rmsd"]
    assert np.allclose(np.asarray(proc["rotation_matrix"]), q.T, atol=1e-9)
    # 反射敏感性：镜像数据下 proper 拟合有残差，允许反射后残差≈0 且 det≈-1
    mirror = base_points.copy()
    mirror[:, 0] *= -1.0
    proc_proper = spatial.procrustes_global(mirror, base_points, allow_reflection=False)
    proc_reflect = spatial.procrustes_global(mirror, base_points, allow_reflection=True)
    assert abs(proc_proper["rotation_det"] - 1.0) < 1e-9 and proc_proper["aligned_rmsd"] > 1e-6
    assert proc_reflect["rotation_det"] < 0.0 and proc_reflect["aligned_rmsd"] < 1e-9

    # whole-chr 对应规则：copy0 局部偏好（A_mat>A_pat）与两 copy 联合规则冲突时，必须服从 derive_rho
    conflict = {"A_mat": 0.60, "A_pat": 0.50, "B_mat": 0.95, "B_pat": 0.10}
    decision = spatial.mapping_from_rho(conflict)
    assert decision["defined"] and decision["orientation"] == "swapped", decision
    assert decision["copy0_to_reference"] == 1 and decision["copy1_to_reference"] == 0, decision
    tie_decision = spatial.mapping_from_rho({"A_mat": 0.4, "A_pat": 0.3, "B_mat": 0.4, "B_pat": 0.3})
    assert not tie_decision["defined"] and tie_decision["orientation"] == "unresolved_tie", tie_decision
    un_decision = spatial.mapping_from_rho({"A_mat": float("nan"), "A_pat": 0.3, "B_mat": 0.4, "B_pat": 0.3})
    assert not un_decision["defined"] and un_decision["orientation"] == "undefined", un_decision

    # null 通路：u_zero 与 random-u
    zero, audit = lib.make_u_zero(reference)
    zero_metrics = ev.null_metrics(zero, reference, data, masks, cache, per_chr_valid, names)
    assert audit["global_scale"] > 0 and zero_metrics["inter"]["status"] == "ok"
    perm, _ = lib.make_random_u(reference, data.chromosome_slices(), 450500)
    perm_metrics = ev.null_metrics(perm, reference, data, masks, cache, per_chr_valid, names)
    assert perm_metrics["r2"]["macro_equal_chromosome_weight_defined_only"]["pearson"]["contrast"] is not None
    assert perm_metrics["inter"]["pearson"] is not None

    # bootstrap 与比较通路
    indices = np.random.default_rng(lib.BOOTSTRAP_SEED).integers(0, 20, size=(200, 20), dtype=np.int64)
    boot_names = ["chr%d" % (i + 1) for i in range(20)]
    same = ev.bootstrap_fixed([0.5] * 20, [0.5] * 20, boot_names, indices)
    assert same["mean"] == 0.0 and same["ties"] == 20 and same["left_wins"] == 0
    assert same["denominator_chromosomes"] == 20 and same["defined_chromosomes"] == 20
    left = {"r2": identical["r2"]}
    right = {"r2": swapped_metrics["r2"]}
    comparison = ev.paired_comparison({"l": left, "r": right}, "l", "r", "selfcheck", indices)
    assert "pearson:matched" in comparison["metrics"]
    plan = ev.comparison_plan(list(lib.EXPECTED_FIT_IDS))
    assert len(plan) == 12 + 2 + 8 + 6, len(plan)  # 12 fit-baseline, 2 initial-baseline, 8 loss contrasts, 6 solver contrasts

    print("SYNTH CHECK PASS: evaluation data path validated on a 20-chr synthetic object "
          "(not a formal run; reference and mask snapshot untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
