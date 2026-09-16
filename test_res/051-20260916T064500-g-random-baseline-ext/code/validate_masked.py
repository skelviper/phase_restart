"""C 组 masked 目标的必要正确性检查（v3 union 规则）。

1. all-ones mask 与原 G 的 value + gradient 逐元素 parity（0 差）。
2. 缺失 lane（两 copy 都不存在）的坐标惰性：value 与 active 自由度梯度完全不变，
   缺失自由度梯度为 0。
3. 单边存在 locus 的 contacts 必须保留：只删 copy0 珠子时 removed pair = 0，
   observed 项不变。
4. 缺口例子：两 copy 都缺的珠子，其相邻 edge/triple 真的从 bond/bend 消失。
5. 方向中心有限差分（active 自由度）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
for _path in (str(HERE), str(S049)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import round_runner  # noqa: E402
from frozen_imports import contact_model, data_io  # noqa: E402
from masked_objective import PerCopySupportObjective  # noqa: E402
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective  # noqa: E402
from support_data import build_filtered  # noqa: E402

WEIGHTS = PenaltyWeights(1.0, 1.0, 1.0, 0.01, 1.0)
MODE = "V0-fixed-production-e"


def make_objective(data):
    return SharedCaptureObjective(data, "G", weights=WEIGHTS, mode=MODE, device="cuda",
                                  pair_block=262_144, inner_cap=80, cg_cap=80,
                                  profile_warm_start=True, known_e=None)


def make_masked(data, mask):
    filtered, exposure, audit = build_filtered(data, mask)
    objective = PerCopySupportObjective(data, "G", weights=WEIGHTS, mask=mask, filtered=filtered,
                                        exposure_valid=exposure, mode=MODE, device="cuda",
                                        pair_block=262_144, inner_cap=80, cg_cap=80,
                                        profile_warm_start=True, known_e=None)
    return objective, audit


def random_theta(data, seed=7):
    rng = np.random.default_rng(seed)
    raw = rng.normal(scale=0.35, size=(2, int(data.n_loci), 3))
    return np.concatenate((raw.reshape(-1), np.asarray([0.4])))


def lane_active(n_loci, lane):
    lane_xyz = np.broadcast_to(np.asarray(lane)[:, None], (int(n_loci), 3))
    return np.concatenate([np.broadcast_to(lane_xyz, (2, int(n_loci), 3)).reshape(-1),
                           np.asarray([True])])


def check_stage(stage: str, results: list[tuple[str, bool, str]]) -> None:
    bin_size = {"5Mb": 5_000_000, "2Mb": 2_000_000, "1Mb": 1_000_000}[stage]
    path = (ROOT / "test_res/045-20260915T073310Z-shared-capture-round/inputs/"
                   "real_1000000_aggregate.npz") if bin_size == 1_000_000 \
        else RUN / "inputs" / ("real_%d_aggregate.npz" % bin_size)
    data = data_io.load_aggregate(path)
    with np.load(RUN / "inputs" / ("reference_support_%s.npz" % stage)) as payload:
        mask = np.asarray(payload["mask"], dtype=bool)
    theta = random_theta(data)

    ones = np.ones_like(mask)
    masked, _audit = make_masked(data, ones)
    parent = make_objective(data)
    _vp, grad_p, comp_p = parent.evaluate(theta, need_gradient=True)
    _vm, grad_m, comp_m = masked.evaluate(theta, need_gradient=True)
    for key in ("count_nll_normalized", "bond", "bend", "repulsion", "total"):
        results.append(("all-ones %s parity" % key,
                        bool(np.isclose(comp_p[key], comp_m[key], rtol=0.0, atol=1e-9)),
                        "parent=%.12g masked=%.12g" % (comp_p[key], comp_m[key])))
    grad_diff = float(np.max(np.abs(grad_p - grad_m)))
    results.append(("all-ones gradient parity", bool(grad_diff <= 1e-9),
                    "max|dp| = %.3e" % grad_diff))

    masked, _audit = make_masked(data, mask)
    value_a, grad_a, _comp = masked.evaluate(theta, need_gradient=True)
    lane = np.asarray(mask).any(axis=0)
    active = lane_active(data.n_loci, lane)
    moved = theta.copy()
    missing_xyz = np.broadcast_to(~lane[:, None], (int(data.n_loci), 3))
    moved[:-1].reshape(2, int(data.n_loci), 3)[:, missing_xyz] = 0.9
    value_b, grad_b, _comp_b = masked.evaluate(moved, need_gradient=True)
    results.append(("missing-lane laziness (value)",
                    bool(np.isclose(value_a, value_b, rtol=0.0, atol=1e-12)),
                    "delta=%.3e" % abs(value_a - value_b)))
    active_diff = float(np.max(np.abs((grad_a - grad_b)[active])))
    results.append(("missing-lane laziness (active gradient)", bool(active_diff <= 1e-12),
                    "max|d| = %.3e" % active_diff))
    missing_slice = grad_a[~active]
    missing_max = float(np.max(np.abs(missing_slice))) if missing_slice.size else 0.0
    results.append(("missing-lane gradient is zero", bool(missing_max <= 1e-12),
                    "max|g_missing| = %.3e" % missing_max))
    probe = theta.copy()
    probe[int(np.flatnonzero(active[:-1])[0])] += 0.05
    value_c = masked.evaluate(probe, need_gradient=False)[0]
    results.append(("active coordinate changes the objective",
                    bool(abs(value_c - value_a) > 1e-9), "delta=%.3e" % abs(value_c - value_a)))

    small = make_small_template()
    check_single_sided_locus(small, results)
    check_gap_and_finite_difference(small, results)


def make_small_template():
    lengths = (6_000_000, 6_000_000, 6_000_000)
    records = []
    rng = np.random.default_rng(11)
    for chrom in range(3):
        for bin_index in range(6):
            for other in range(bin_index + 1, 6):
                for _ in range(int(rng.integers(1, 4))):
                    records.append((chrom, bin_index * 1_000_000 + 10, chrom,
                                    other * 1_000_000 + 10))
        for other_chrom in range(chrom + 1, 3):
            for _ in range(3):
                records.append((chrom, bin_index * 1_000_000 + 10, other_chrom,
                                bin_index * 1_000_000 + 10))
    ci = np.asarray([row[0] for row in records], dtype=np.int64)
    p1 = np.asarray([row[1] for row in records], dtype=np.int64)
    cj = np.asarray([row[2] for row in records], dtype=np.int64)
    p2 = np.asarray([row[3] for row in records], dtype=np.int64)
    return contact_model.aggregate_from_arrays(("c0", "c1", "c2"), lengths, ci, p1, cj, p2, 1_000_000)


def check_single_sided_locus(data, results: list[tuple[str, bool, str]]) -> None:
    mask = np.ones((2, int(data.n_loci)), dtype=bool)
    mask[1, 3] = False
    masked, audit = make_masked(data, mask)
    results.append(("single-sided locus keeps count pairs",
                    audit["removed_pairs_cis"] == 0 and audit["removed_pairs_inter"] == 0,
                    "removed cis=%d inter=%d" % (audit["removed_pairs_cis"],
                                                 audit["removed_pairs_inter"])))
    lane = mask.any(axis=0)
    results.append(("single-sided locus stays in lane",
                    bool(lane[3]) and int(lane.sum()) == int(data.n_loci),
                    "lane[bin3]=%s union=%d" % (bool(lane[3]), int(lane.sum()))))
    theta = random_theta(data, seed=23)
    parent = make_objective(data)
    _v, _g, comp_p = parent.evaluate(theta, need_gradient=True)
    _v2, _g2, comp_m = masked.evaluate(theta, need_gradient=True)
    # 单边 locus 的 contacts 必须仍然进入 observed 项：正 rate 的 observed pair 数不变。
    positive_parent = int(comp_p["n_observed_pairs_with_positive_rate"]
                          if "n_observed_pairs_with_positive_rate" in comp_p else -1)
    import torch
    counts = np.asarray(data.counts, dtype=np.float64)
    observed = counts > 0.0
    theta_t = torch.as_tensor(np.asarray(theta, dtype=np.float64), dtype=torch.float64,
                              device="cuda")
    raw = theta_t[:-1].reshape(2, int(data.n_loci), 3)
    x = masked._physics._map_raw(raw)
    p_t = torch.as_tensor(float(comp_m["p"]), dtype=torch.float64, device="cuda")
    k_masked = masked._build_k(x, p_t).detach().cpu().numpy()
    kept = int(np.count_nonzero(observed & (k_masked > 0.0)))
    results.append(("single-sided locus observed pairs retained",
                    kept == int(np.count_nonzero(observed)),
                    "observed pairs kept=%d of %d (positive_parent=%d)" % (
                        kept, int(np.count_nonzero(observed)), positive_parent)))


def check_gap_and_finite_difference(data, results: list[tuple[str, bool, str]]) -> None:
    both_missing = np.ones((2, int(data.n_loci)), dtype=bool)
    both_missing[:, 3] = False
    theta = random_theta(data, seed=13)
    masked, audit = make_masked(data, both_missing)
    # 模板 3 chr x 6 bin：每 copy 15 edge / 12 triple，两 copy 共 30 / 24；
    # 两 copy 都删中间 bin -> 少 2 edge / 少 3 triple（每 copy）。
    results.append(("gap removes bond edges", masked._bond_terms == 2 * (15 - 2),
                    "bond_terms_valid=%d expected=%d" % (masked._bond_terms, 2 * (15 - 2))))
    results.append(("gap removes bend triples", masked._bend_terms == 2 * (12 - 3),
                    "bend_terms_valid=%d expected=%d" % (masked._bend_terms, 2 * (12 - 3))))
    results.append(("gap removes count pairs",
                    audit["removed_pairs_cis"] + audit["removed_pairs_inter"] > 0,
                    "removed cis=%d inter=%d" % (audit["removed_pairs_cis"],
                                                 audit["removed_pairs_inter"])))
    lane = np.asarray(both_missing).any(axis=0)
    active = lane_active(data.n_loci, lane)
    _value, gradient, _comp = masked.evaluate(theta, need_gradient=True)
    index = int(np.flatnonzero(active[:-1])[0])
    step = 1e-5
    plus = theta.copy()
    minus = theta.copy()
    plus[index] += step
    minus[index] -= step
    numeric = (masked.evaluate(plus, need_gradient=False)[0]
               - masked.evaluate(minus, need_gradient=False)[0]) / (2.0 * step)
    analytic = float(gradient[index])
    relative = abs(numeric - analytic) / max(1e-12, abs(numeric))
    results.append(("finite difference (active dof)", bool(relative <= 1e-5),
                    "numeric=%.9g analytic=%.9g rel=%.3e" % (numeric, analytic, relative)))
    missing_slice = gradient[~active]
    missing_max = float(np.max(np.abs(missing_slice))) if missing_slice.size else 0.0
    results.append(("gap example missing-lane gradient zero", bool(missing_max <= 1e-12),
                    "max|g_missing| = %.3e" % missing_max))


def check_prolongation_parity(results: list[tuple[str, bool, str]]) -> None:
    """all-ones 时 C 的跨层 prolongation 必须与原 warm_start_from_layer 逐位一致。"""
    sys.path.insert(0, str(ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"))
    from frozen_imports import reconstruction_init
    small_names = ("c0", "c1", "c2")
    lengths = (6_000_000, 6_000_000, 6_000_000)
    rng = np.random.default_rng(5)
    coordinates = rng.normal(scale=0.3, size=(2, 18, 3))
    positions = np.broadcast_to(
        np.concatenate([np.arange(6) * 1_000_000 for _ in range(3)]), (2, 18)).copy()
    chromosomes = np.broadcast_to(np.repeat(np.arange(3), 6), (2, 18)).copy()
    ones = np.ones((2, 18), dtype=bool)
    warm = reconstruction_init.warm_start_from_layer(
        coordinates, positions, chromosomes, small_names, lengths, 1_000_000, 2207)
    reference = np.asarray(warm["coords"], dtype=np.float64)
    # C 路径：只用有效点构造 source track，再 expand + 同 seed/scale/clip 扰动
    source_tracks = {}
    for chromosome_index in range(3):
        slc = slice(chromosome_index * 6, chromosome_index * 6 + 6)
        for copy in (0, 1):
            keep = ones[copy, slc]
            source_tracks["c%02d%s" % (chromosome_index + 1, "ab"[copy])] = (
                positions[copy, slc][keep].copy(), coordinates[copy, slc][keep].copy())
    expanded = reconstruction_init.expand_tracks_to_full_grid(
        source_tracks, small_names, lengths, 1_000_000, "random")
    candidate = np.asarray(expanded["coords"], dtype=np.float64)
    perturb_mask = np.asarray(expanded["perturb_mask"], dtype=bool)
    l0 = float((2 * 18) ** (-1.0 / 3.0))
    rng2 = np.random.default_rng(3301 + 1 + 2207)
    noise = rng2.normal(scale=0.025 * l0, size=candidate.shape)
    candidate[perturb_mask] += noise[perturb_mask]
    limit = 1.0 - 1e-6
    radii = np.linalg.norm(candidate, axis=2)
    clipped = radii >= limit
    if clipped.any():
        candidate[clipped] *= (limit / radii[clipped])[:, None]
    diff = float(np.max(np.abs(candidate - reference)))
    results.append(("C prolongation all-ones parity with warm_start",
                    bool(diff == 0.0), "max|d| = %.3e" % diff))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="5Mb", choices=("20Mb", "10Mb", "5Mb", "2Mb", "1Mb"))
    args = parser.parse_args()
    round_runner.install(RUN)
    round_runner.set_active("A", "raw")
    results: list[tuple[str, bool, str]] = []
    check_stage(args.stage, results)
    check_prolongation_parity(results)
    for name, passed, detail in results:
        print("%-48s %s  %s" % (name, "PASS" if passed else "FAIL", detail))
    failed = [name for name, passed, _ in results if not passed]
    print("SUMMARY", "PASS" if not failed else "FAIL", "stage=%s" % args.stage, failed)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
