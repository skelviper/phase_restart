"""049 必要工程门。全部通过后才启动正式 12 fit。

运行：python source/gates.py
输出：gates/gates_report.json 以及各门的单独 JSON。
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from frozen_imports import contact_model, data_io, formal_controller, m1_preconditioner, reconstruction_init
from fixture import seed_theta, small_fixture
from max_contact_objective import MaxContactObjective
from multiscale_objective import MultiscaleObjective, make_preconditioner
from round_paths import (AGGREGATE_1MB, BASELINE_NPZ, INITIAL_2MB, INITIAL_5MB, LOSSES,
                         PAIR_BLOCK, PROLONGATION_SEED, ROOT, RUN, SOURCE_045, SOURCE_045_SRC)
from round_runner import array_sha256, install, load_data, load_initial, set_active, sha256_file
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective

GATES = RUN / "gates"
SCRATCH = ROOT / "scratch/049-gate-integration"
MODES = "V0-fixed-production-e"
WEIGHTS = PenaltyWeights()
FROZEN_TRACKED = [
    SOURCE_045_SRC / "shared_capture_objective.py",
    SOURCE_045_SRC / "formal_controller.py",
    SOURCE_045_SRC / "m1_preconditioner.py",
    SOURCE_045_SRC / "visibility_profile_base.py",
    SOURCE_045_SRC / "data_io.py",
    SOURCE_045_SRC / "frozen_pr/pr/contact_model.py",
    ROOT / "test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.3dg",
    ROOT / "test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.npz",
    ROOT / "native/hickit/PROVENANCE.md",
]


def record(name: str, status: str, **evidence: Any) -> dict[str, Any]:
    row = {"gate": name, "status": status, "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "evidence": evidence}
    GATES.mkdir(parents=True, exist_ok=True)
    (GATES / (name + ".json")).write_text(
        json.dumps(row, sort_keys=True, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    print("[%s] %s %s" % (status, name, json.dumps(evidence, default=str)[:400]), flush=True)
    return row


def objective(data: Any, loss: str, pair_block: int = PAIR_BLOCK) -> MaxContactObjective:
    return MaxContactObjective(data, loss, weights=WEIGHTS, mode=MODES, device="cuda",
                               pair_block=pair_block, inner_cap=80, cg_cap=80, profile_warm_start=True, known_e=None)


def g_wrapper(data: Any, pair_block: int = PAIR_BLOCK) -> SharedCaptureObjective:
    return SharedCaptureObjective(data, model_id="G", weights=WEIGHTS, mode=MODES, device="cuda",
                                  pair_block=pair_block, inner_cap=80, cg_cap=80, profile_warm_start=True,
                                  known_e=None)


def tracked_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in FROZEN_TRACKED if path.exists()}


# --------------------------------------------------------------------- gates
def gate_loss_a_parity() -> dict[str, Any]:
    data = load_data()
    with np.load(BASELINE_NPZ, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        p = float(np.asarray(payload["p"]).item())
        q = float(np.asarray(payload["q"]).item())
    contact_model.assert_inside_unit_ball(coordinates)
    raw_y = contact_model.sphere_inverse(coordinates)
    theta = np.concatenate((raw_y.reshape(-1), np.asarray([q], dtype=np.float64)))
    loss_a = objective(data, "A")
    value_a, gradient_a, components_a = loss_a.evaluate(theta, True)
    wrapper = g_wrapper(data)
    value_g, gradient_g, components_g = wrapper.evaluate(theta, True)
    value_diff = abs(float(value_a) - float(value_g))
    gradient_diff = float(np.max(np.abs(np.asarray(gradient_a) - np.asarray(gradient_g))))
    count_diff = abs(float(components_a["count_nll_normalized"]) - float(components_g["count_nll_normalized"]))
    status = "PASS" if (value_diff <= 1e-9 and gradient_diff <= 1e-12 and count_diff <= 1e-9) else "FAIL"
    return record("gate1_lossA_vs_045_G_parity", status, baseline_npz=str(BASELINE_NPZ.relative_to(ROOT)),
                  baseline_sha256=sha256_file(BASELINE_NPZ), n_loci=int(data.n_loci), n_pairs=int(data.n_pairs),
                  value_lossA=float(value_a), value_045_G=float(value_g), value_abs_diff=value_diff,
                  count_nll_abs_diff=count_diff, gradient_max_abs_diff=gradient_diff,
                  value_tolerance=1e-9, gradient_tolerance=1e-12,
                  note="gradient residual is GPU atomic index_add ordering noise, not an algebraic difference")


def _fd_directional(obj: Any, theta: np.ndarray, direction: np.ndarray, step: float) -> float:
    plus = np.asarray(theta, dtype=np.float64) + float(step) * direction
    minus = np.asarray(theta, dtype=np.float64) - float(step) * direction
    value_plus, _, _ = obj.evaluate(plus, False)
    value_minus, _, _ = obj.evaluate(minus, False)
    return (float(value_plus) - float(value_minus)) / (2.0 * float(step))


def _max_state_table(obj: Any, theta: np.ndarray) -> dict[str, Any]:
    """每个 pair 的四状态 argmax 与 top1-top2 间隔，用于判断 FD 是否跨分支。"""
    import torch as _torch
    values = _torch.as_tensor(np.asarray(theta, dtype=np.float64), dtype=_torch.float64, device="cuda")
    x = obj._physics._map_raw(values[:-1].reshape(2, int(obj.data.n_loci), 3))
    p, _ = contact_model.p_from_q(float(theta[-1]))
    p_t = _torch.as_tensor(p, dtype=_torch.float64, device="cuda")
    kernels, _ = obj._kernel_block(x, obj._pair_i, obj._pair_j, p_t, with_gradient=False)
    stacked, _ = obj._block_mixture(kernels, p_t, obj._cis)
    top2 = stacked.topk(2, dim=0).values
    gap = top2[0] - top2[1]
    argmax = stacked.max(dim=0).indices.detach().cpu().numpy().astype(np.int64)
    return {"argmax": argmax, "gap_min": float(gap.min()), "gap_max": float(gap.max()),
            "tied_pairs": int((gap == 0).sum())}


def _q_only_fd(obj: Any, theta: np.ndarray, step: float, count_only: bool) -> tuple[float, float]:
    plus = np.array(theta, dtype=np.float64, copy=True)
    minus = np.array(theta, dtype=np.float64, copy=True)
    plus[-1] += float(step)
    minus[-1] -= float(step)
    if count_only:
        value_plus = float(obj.count_value_and_grad(plus)[0])
        value_minus = float(obj.count_value_and_grad(minus)[0])
    else:
        value_plus = float(obj.evaluate(plus, False)[0])
        value_minus = float(obj.evaluate(minus, False)[0])
    return (value_plus - value_minus) / (2.0 * float(step))


def gate_finite_difference() -> dict[str, Any]:
    data = small_fixture()
    evidence: dict[str, Any] = {"fixture_n_loci": int(data.n_loci), "fixture_n_pairs": int(data.n_pairs),
                                "checks": [], "q_only_checks": [], "tie_gap": [],
                                "worst_relative_error": 0.0, "worst_q_only_relative_error": 0.0,
                                "branch_crossing_directions": 0, "all_max_unique": True}
    worst = 0.0
    for p_value in (0.2, 0.5, 0.8):
        theta = seed_theta(data, p=p_value, seed=4910 + int(p_value * 10))
        probe = objective(data, "B", pair_block=64)
        table = _max_state_table(probe, theta)
        evidence["tie_gap"].append({"p": p_value, "gap_min": table["gap_min"], "gap_max": table["gap_max"],
                                    "tied_pairs": table["tied_pairs"]})
        if table["tied_pairs"] > 0:
            evidence["all_max_unique"] = False
        rng = np.random.default_rng(4921)
        direction = np.zeros_like(theta)
        direction[:-1] = rng.normal(size=theta[:-1].shape)
        direction[-1] = 0.37
        direction /= np.linalg.norm(direction)
        for loss in LOSSES:
            obj = objective(data, loss, pair_block=64)
            value, gradient, _components = obj.evaluate(theta, True)
            analytic = float(np.dot(np.asarray(gradient, dtype=np.float64), direction))
            step = 1e-8
            stable = True
            if loss in ("B", "C"):
                plus_table = _max_state_table(obj, theta + step * direction)
                minus_table = _max_state_table(obj, theta - step * direction)
                stable = (np.array_equal(table["argmax"], plus_table["argmax"])
                          and np.array_equal(table["argmax"], minus_table["argmax"]))
            numeric = _fd_directional(obj, theta, direction, step)
            scale = max(abs(analytic), abs(numeric), 1e-8)
            relative = abs(analytic - numeric) / scale
            if stable:
                worst = max(worst, relative)
            else:
                evidence["branch_crossing_directions"] += 1
            evidence["checks"].append({"loss": loss, "p": p_value, "kind": "directional", "step": step,
                                       "branch_stable": bool(stable), "analytic": analytic, "numeric": numeric,
                                       "relative_error": relative})
    worst_count_only = 0.0
    q_branch_crossings = 0
    for p_value in (0.2, 0.5, 0.8):
        theta = seed_theta(data, p=p_value, seed=6100 + int(p_value * 10))
        probe = objective(data, "B", pair_block=64)
        table = _max_state_table(probe, theta)
        step_q = 1e-6
        plus_q = np.array(theta, dtype=np.float64, copy=True)
        minus_q = np.array(theta, dtype=np.float64, copy=True)
        plus_q[-1] += step_q
        minus_q[-1] -= step_q
        plus_table = _max_state_table(probe, plus_q)
        minus_table = _max_state_table(probe, minus_q)
        plus_ok = int(table["tied_pairs"]) == int(plus_table["tied_pairs"])
        minus_ok = int(table["tied_pairs"]) == int(minus_table["tied_pairs"])
        for loss in LOSSES:
            obj = objective(data, loss, pair_block=64)
            _value, gradient, _components = obj.evaluate(theta, True)
            _count_value, count_gradient = obj.count_value_and_grad(theta)
            for count_only, label in ((False, "full_J"), (True, "count")):
                analytic = float(gradient[-1]) if not count_only else float(count_gradient[-1])
                numeric = _q_only_fd(obj, theta, step_q, count_only)
                scale = max(abs(analytic), abs(numeric), 1e-8)
                relative = abs(analytic - numeric) / scale
                stable = bool(plus_ok and minus_ok)
                if not stable:
                    q_branch_crossings += 1
                if not count_only:
                    worst = max(worst, relative)
                else:
                    worst_count_only = max(worst_count_only, relative)
                evidence["q_only_checks"].append({"loss": loss, "p": p_value, "observable": label,
                                                  "analytic_dF_dq": analytic, "numeric_dF_dq": numeric,
                                                  "relative_error": relative, "branch_stable": stable,
                                                  "tie_gap_min": table["gap_min"],
                                                  "tie_gap_min_plus": plus_table["gap_min"],
                                                  "tie_gap_min_minus": minus_table["gap_min"]})
    evidence["worst_q_only_relative_error"] = worst
    evidence["worst_q_only_count_only_relative_error"] = worst_count_only
    evidence["q_only_branch_crossings"] = q_branch_crossings
    # sum 分支 p 梯度必须等于 045 原 G：同一坐标下 loss A 与 G wrapper 的 dF/dq 逐位对比
    theta = seed_theta(data, p=0.8, seed=4941)
    loss_a = objective(data, "A", pair_block=64)
    wrapper = g_wrapper(data, pair_block=64)
    _va, gradient_a, _ca = loss_a.evaluate(theta, True)
    _vg, gradient_g, _cg = wrapper.evaluate(theta, True)
    evidence["sum_branch_p_gradient_vs_045_G"] = {
        "loss_A_dF_dq": float(gradient_a[-1]), "G_wrapper_dF_dq": float(gradient_g[-1]),
        "abs_diff": abs(float(gradient_a[-1]) - float(gradient_g[-1]))}
    evidence["worst_relative_error"] = worst
    status = "PASS" if (worst <= 1e-5 and worst_count_only <= 1e-5 and q_branch_crossings == 0
                        and evidence["all_max_unique"]
                        and evidence["branch_crossing_directions"] == 0
                        and abs(evidence["sum_branch_p_gradient_vs_045_G"]["abs_diff"]) <= 1e-12) else "FAIL"
    return record("gate2_finite_difference_all_losses", status, **evidence)


def gate_ties_and_gauge() -> dict[str, Any]:
    data = small_fixture()
    theta_base = seed_theta(data, p=0.8, seed=4941)
    n_loci = int(data.n_loci)
    import torch as _torch
    evidence: dict[str, Any] = {}

    # (a) 并列 subgradient 权重规则
    tie_probe = objective(data, "B", pair_block=64)
    probe_matrix = np.array([[2.0, 0.4, 1.0, 5.0],
                             [0.5, 0.4, 0.2, 5.0],
                             [0.5, 0.4, 0.3, 5.0],
                             [2.0, 0.4, 0.4, 1.0]])
    tie_weights = tie_probe._tie_weights(_torch.as_tensor(probe_matrix, dtype=_torch.float64, device="cuda"))
    expected = np.array([[0.5, 0.25, 1.0, 1.0 / 3.0],
                         [0.0, 0.25, 0.0, 1.0 / 3.0],
                         [0.0, 0.25, 0.0, 1.0 / 3.0],
                         [0.5, 0.25, 0.0, 0.0]])
    evidence["tie_weight_matrix"] = tie_weights.detach().cpu().numpy().tolist()
    evidence["tie_weight_expected"] = expected.tolist()
    tie_ok = bool(np.allclose(tie_weights.detach().cpu().numpy(), expected, atol=1e-15))

    # (b) p=0.5 且两 copy 完全相同 => 每个 pair 四状态 t 精确并列 => tie 权重全为 0.25
    theta_tie = seed_theta(data, p=0.5, seed=6201)
    flat = theta_tie[:-1].reshape(2, n_loci, 3)
    flat[1] = flat[0]
    theta = np.concatenate((flat.reshape(-1), theta_tie[-1:]))
    flat2 = theta_base[:-1].reshape(2, n_loci, 3)
    flat2[1] = flat2[0]
    theta_two_way = np.concatenate((flat2.reshape(-1), theta_base[-1:]))

    def tie_weight_columns(theta_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obj = objective(data, "B", pair_block=64)
        vals = _torch.as_tensor(theta_value, dtype=_torch.float64, device="cuda")
        xs = obj._physics._map_raw(vals[:-1].reshape(2, n_loci, 3))
        pv, _ = contact_model.p_from_q(float(theta_value[-1]))
        pv_t = _torch.as_tensor(pv, dtype=_torch.float64, device="cuda")
        kernels, _ = obj._kernel_block(xs, obj._pair_i, obj._pair_j, pv_t, with_gradient=False)
        stacked, _ = obj._block_mixture(kernels, pv_t, obj._cis)
        weights = obj._tie_weights(stacked).detach().cpu().numpy()
        cis_mask = np.asarray(obj._cis.detach().cpu().numpy(), dtype=bool)
        return weights[:, cis_mask], weights[:, ~cis_mask]

    cis_w, inter_w = tie_weight_columns(theta)
    evidence["four_way_tie_p_0.5"] = {"cis_weights_unique": np.unique(np.round(cis_w, 12)).tolist(),
                                      "inter_weights_unique": np.unique(np.round(inter_w, 12)).tolist(),
                                      "cis_all_quarter": bool(np.allclose(cis_w, 0.25)),
                                      "inter_all_quarter": bool(np.allclose(inter_w, 0.25))}
    four_way_ok = bool(np.allclose(cis_w, 0.25) and np.allclose(inter_w, 0.25))
    cis_w2, inter_w2 = tie_weight_columns(theta_two_way)
    evidence["two_way_tie_p_0.8"] = {"p": float(flat2.dtype.type(0) + contact_model.p_from_q(float(theta_two_way[-1]))[0]),
                                     "cis_weights_unique": np.unique(np.round(cis_w2, 12)).tolist(),
                                     "inter_weights_unique": np.unique(np.round(inter_w2, 12)).tolist(),
                                     "cis_is_AA_BB_half": bool(np.allclose(cis_w2, np.array([[0.5], [0.0], [0.0], [0.5]]))),
                                     "inter_all_quarter": bool(np.allclose(inter_w2, 0.25))}
    two_way_ok = bool(np.allclose(cis_w2, np.array([[0.5], [0.0], [0.0], [0.5]]))
                      and np.allclose(inter_w2, 0.25))

    # (c) p=0.5 且 A==B 的四态全 tie：独立解析断言
    #     r_max = r_sum/4，Z_max = Z_sum/4  =>  C_count == A_count，
    #     B_count == A_count + (Noff/Nraw)*log 4；三者的坐标与 q 梯度在此特例必须一致。
    count_rows = []
    for loss in LOSSES:
        obj = objective(data, loss, pair_block=64)
        value, gradient = obj.count_value_and_grad(theta)
        count_rows.append({"loss": loss, "count_value": float(value),
                           "gradient": np.asarray(gradient, dtype=np.float64).copy()})
    by_loss = {row["loss"]: row for row in count_rows}
    noff = float(data.raw_cis_offdiag) + float(data.raw_inter)
    nraw = float(data.raw_records)
    predicted_shift = noff * math.log(4.0) / nraw
    value_c_minus_a = abs(by_loss["C"]["count_value"] - by_loss["A"]["count_value"])
    value_b_shift_error = abs((by_loss["B"]["count_value"] - by_loss["A"]["count_value"]) - predicted_shift)
    gradient_ab = float(np.max(np.abs(by_loss["A"]["gradient"] - by_loss["B"]["gradient"])))
    gradient_ac = float(np.max(np.abs(by_loss["A"]["gradient"] - by_loss["C"]["gradient"])))
    four_way_analytic_ok = bool(value_c_minus_a <= 1e-9 and value_b_shift_error <= 1e-9
                                and gradient_ab <= 1e-10 and gradient_ac <= 1e-10)
    evidence["four_way_analytic_assertion"] = {
        "Noff": noff, "Nraw": nraw, "predicted_B_minus_A": predicted_shift,
        "count_values": {key: by_loss[key]["count_value"] for key in LOSSES},
        "observed_B_minus_A": by_loss["B"]["count_value"] - by_loss["A"]["count_value"],
        "observed_C_minus_A": by_loss["C"]["count_value"] - by_loss["A"]["count_value"],
        "value_C_minus_A_abs": value_c_minus_a, "value_B_shift_error": value_b_shift_error,
        "gradient_A_vs_B_max_abs": gradient_ab, "gradient_A_vs_C_max_abs": gradient_ac,
        "pass": four_way_analytic_ok,
        "derivation": "A==B copies and p=0.5 tie all four states: r_max=r_sum/4 and Z_max=Z_sum/4",
    }

    # (c2) count 层单侧方向导数必须夹住解析次梯度（纯 raw_y 方向，q 不扰动）
    rng = np.random.default_rng(4951)
    bracket_rows = []
    bracket_ok = True
    for tie_label, theta_value in (("four_way_p_0.5", theta), ("two_way_p_0.8", theta_two_way)):
        direction = np.zeros_like(theta_value)
        direction[:-1] = rng.normal(size=theta_value[:-1].shape)
        direction[-1] = 0.0
        direction /= np.linalg.norm(direction)
        for loss in LOSSES:
            obj = objective(data, loss, pair_block=64)
            base_count = float(obj.count_value_and_grad(theta_value)[0])
            _cv, count_gradient = obj.count_value_and_grad(theta_value)
            analytic = float(np.dot(np.asarray(count_gradient, dtype=np.float64), direction))
            step = 1e-8
            plus = theta_value + step * direction
            minus = theta_value - step * direction
            forward = (float(obj.count_value_and_grad(plus)[0]) - base_count) / step
            backward = (base_count - float(obj.count_value_and_grad(minus)[0])) / step
            low, high = min(forward, backward), max(forward, backward)
            margin = min(analytic - low, high - analytic)
            ok = margin >= -1e-6 * max(1.0, abs(analytic))
            bracket_ok = bracket_ok and ok
            bracket_rows.append({"tie": tie_label, "loss": loss, "analytic": analytic,
                                 "one_sided_forward": forward, "one_sided_backward": backward,
                                 "margin": margin, "inside_bracket": ok})
        # 同口径纯 q 方向（count 层）也做单侧夹逼
        q_direction = np.zeros_like(theta_value)
        q_direction[-1] = 1.0 / math.sqrt(float(data.n_loci) * 6.0 + 1.0)
        for loss in LOSSES:
            obj = objective(data, loss, pair_block=64)
            base_count = float(obj.count_value_and_grad(theta_value)[0])
            _cv, count_gradient = obj.count_value_and_grad(theta_value)
            analytic = float(np.asarray(count_gradient, dtype=np.float64)[-1] * q_direction[-1])
            step = 1e-8
            forward = (float(obj.count_value_and_grad(theta_value + step * q_direction)[0]) - base_count) / step
            backward = (base_count - float(obj.count_value_and_grad(theta_value - step * q_direction)[0])) / step
            low, high = min(forward, backward), max(forward, backward)
            margin = min(analytic - low, high - analytic)
            ok = margin >= -1e-6 * max(1.0, abs(analytic))
            bracket_ok = bracket_ok and ok
            bracket_rows.append({"tie": tie_label, "loss": loss, "direction": "q_only", "analytic": analytic,
                                 "one_sided_forward": forward, "one_sided_backward": backward,
                                 "margin": margin, "inside_bracket": ok})
    evidence["count_layer_one_sided_bracket"] = bracket_rows

    # (d) copy swap 下 value 不变、gradient 协变
    swap_rows = []
    swap_ok = True
    for loss in LOSSES:
        obj = objective(data, loss, pair_block=64)
        value, gradient, _components = obj.evaluate(theta, True)
        swapped = theta.copy()
        view = swapped[:-1].reshape(2, n_loci, 3)
        view[:] = view[::-1].copy()
        value_swapped, gradient_swapped, _c = obj.evaluate(swapped, True)
        value_diff = abs(float(value) - float(value_swapped))
        mapped = np.asarray(gradient_swapped, dtype=np.float64)
        mapped[:-1] = mapped[:-1].reshape(2, n_loci, 3)[::-1].reshape(-1)
        gradient_diff = float(np.max(np.abs(np.asarray(gradient, dtype=np.float64) - mapped)))
        ok = value_diff <= 1e-10 and gradient_diff <= 1e-8
        swap_ok = swap_ok and ok
        swap_rows.append({"loss": loss, "value_abs_diff": value_diff,
                          "gradient_covariance_max_abs_diff": gradient_diff, "pass": ok})
    evidence["copy_swap"] = swap_rows
    status = "PASS" if (tie_ok and four_way_ok and four_way_analytic_ok and two_way_ok
                        and bracket_ok and swap_ok) else "FAIL"
    return record("gate3_exact_tie_subgradient_and_gauge", status, **evidence)


def gate_symmetry_invariance() -> dict[str, Any]:
    data = small_fixture()
    theta = seed_theta(data, p=0.8, seed=4961)
    n_loci = int(data.n_loci)
    rows = []
    swap_ok = True
    rotate_ok = True
    rng = np.random.default_rng(4971)
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] = -matrix[:, 0]
    for loss in LOSSES:
        obj = objective(data, loss, pair_block=64)
        value, _gradient, _components = obj.evaluate(theta, True)
        # 单条染色体整体 A/B 交换
        swapped = theta.copy()
        flat = swapped[:-1].reshape(2, n_loci, 3)
        slc = data.chromosome_slice(0)
        flat[:, slc, :] = flat[::-1, slc, :].copy()
        value_swapped = float(obj.evaluate(swapped, False)[0])
        # 全局 proper rotation
        coordinates = contact_model.sphere_forward(theta[:-1].reshape(2, n_loci, 3))
        rotated = coordinates @ matrix.T
        rotated_theta = np.concatenate((contact_model.sphere_inverse(rotated).reshape(-1), theta[-1:]))
        value_rotated = float(obj.evaluate(rotated_theta, False)[0])
        swap_diff = abs(value - value_swapped)
        rotate_diff = abs(value - value_rotated)
        swap_ok = swap_ok and swap_diff <= 1e-10
        rotate_ok = rotate_ok and rotate_diff <= 1e-10
        rows.append({"loss": loss, "value": value, "value_chr1_AB_swapped": value_swapped,
                     "value_globally_rotated": value_rotated, "swap_abs_diff": swap_diff,
                     "rotation_abs_diff": rotate_diff})
    status = "PASS" if (swap_ok and rotate_ok) else "FAIL"
    return record("gate4_ab_swap_and_rotation_invariance", status, rows=rows,
                  global_rotation_matrix=matrix.tolist(), rotation_det=float(np.linalg.det(matrix)))


def gate_preconditioner() -> dict[str, Any]:
    data = small_fixture(bins=(4, 3, 5, 2, 6))
    pre = make_preconditioner(data)
    rng = np.random.default_rng(4981)
    y = rng.normal(size=(2, int(data.n_loci), 3)) * 0.4
    other = rng.normal(size=(2, int(data.n_loci), 3)) * 0.4
    roundtrip = float(np.max(np.abs(pre.Pinv_apply(pre.P_apply(y)) - y)))
    inverse_roundtrip = float(np.max(np.abs(pre.P_apply(pre.Pinv_apply(y)) - y)))
    symmetry = abs(float(np.sum(pre.P_apply(y) * other) - np.sum(y * pre.P_apply(other))))
    positivity = bool(pre.min_symbol > 0.0)
    dense_rows = []
    dense_ok = True
    for index, slc in enumerate(pre.slices):
        dense = pre.dense_matrix(index)
        for component in range(3):
            block = np.zeros((2, int(data.n_loci), 3))
            block[:, slc, component] = y[:, slc, component]
            applied = pre.P_apply(block)[0, slc, component]
            reference = dense @ y[0, slc, component]
            error = float(np.max(np.abs(applied - reference)))
            dense_ok = dense_ok and error <= 1e-12
            dense_rows.append({"chromosome_index": index, "n_bins": int(dense.shape[0]), "component": component,
                               "max_abs_error": error})
    # 链内独立性：只在 chr0 上有值，P 后其余染色体必须保持 0
    isolated = np.zeros((2, int(data.n_loci), 3))
    slc0 = pre.slices[0]
    isolated[:, slc0, :] = y[:, slc0, :]
    applied = pre.P_apply(isolated)
    mask = np.ones(int(data.n_loci), dtype=bool)
    mask[slc0] = False
    outside = applied[0][mask]
    chain_error = float(np.max(np.abs(outside))) if outside.size else 0.0
    # identity fixture parity
    identity = make_preconditioner(data, identity=True)
    identity_theta = seed_theta(data, p=0.7, seed=4991)
    base = objective(data, "B", pair_block=64)
    wrapped_identity = MultiscaleObjective(base, identity)
    wrapped_real = MultiscaleObjective(base, pre)
    value_base, gradient_base, _c = base.evaluate(identity_theta, True)
    theta_identity = identity_theta.copy()
    theta_identity[:-1] = identity.Pinv_apply(identity_theta[:-1].reshape(2, data.n_loci, 3)).reshape(-1)
    value_identity, gradient_identity, _c2 = wrapped_identity.evaluate(theta_identity, True)
    identity_error = float(np.max(np.abs(np.asarray(gradient_identity) - np.asarray(gradient_base))))
    theta_real = identity_theta.copy()
    theta_real[:-1] = pre.Pinv_apply(identity_theta[:-1].reshape(2, data.n_loci, 3)).reshape(-1)
    value_real, gradient_real, _c3 = wrapped_real.evaluate(theta_real, True)
    canonical = wrapped_real.canonical_raw_gradient(theta_real, gradient_real)
    canonical_error = float(np.max(np.abs(canonical - np.asarray(gradient_base))))
    optimizer_vs_canonical = float(np.max(np.abs(np.asarray(gradient_real) - canonical)))
    status = "PASS" if (roundtrip <= 1e-12 and inverse_roundtrip <= 1e-12 and symmetry <= 1e-10 and positivity
                        and dense_ok and chain_error == 0.0 and abs(value_identity - value_base) <= 1e-12
                        and identity_error <= 1e-12 and canonical_error <= 1e-10) else "FAIL"
    return record("gate5_preconditioner", status, roundtrip_error=roundtrip, inverse_roundtrip_error=inverse_roundtrip,
                  symmetry_residual=symmetry, symbol_positive=positivity, symbol_min=pre.min_symbol,
                  symbol_max=pre.max_symbol, dense_rows=dense_rows[:6], dense_all_pass=dense_ok,
                  chromosome_isolation_max_abs=chain_error,
                  identity_wrapper_value_diff=abs(float(value_identity) - float(value_base)),
                  identity_wrapper_gradient_diff=identity_error,
                  real_P_value_diff=abs(float(value_real) - float(value_base)),
                  canonical_vs_true_raw_gradient_diff=canonical_error,
                  optimizer_gradient_vs_canonical_diff=optimizer_vs_canonical,
                  note="canonical diff must be large enough to show P is not identity")


def gate_runner_canonical() -> dict[str, Any]:
    install()
    set_active("B", "ms")
    data = small_fixture()
    theta = seed_theta(data, p=0.7, seed=5001)
    n_loci = int(data.n_loci)
    pre = make_preconditioner(data)
    base = objective(data, "B", pair_block=64)
    ms = MultiscaleObjective(base, pre)
    checkpoints: list[dict[str, Any]] = []
    result = m1_preconditioner.run_budgeted_lbfgs(
        ms, theta[:-1].reshape(2, n_loci, 3), p_init=0.7, q_init=float(theta[-1]),
        maxfun=6, maxiter=7, maxls=20, ftol=0.0, canonical_gtol=0.0, checkpoint_every=1,
        checkpoint_hook=lambda entry: checkpoints.append({
            "theta": np.asarray(entry["theta"]).copy(), "raw_y": np.asarray(entry["raw_y"]).copy(),
            "coordinates": np.asarray(entry["coordinates"]).copy(),
            "canonical_gradient": np.asarray(entry["canonical_gradient"]).copy()}))
    _value, optimizer_gradient, _components = ms.evaluate(result.theta, True)
    canonical = ms.canonical_raw_gradient(result.theta, optimizer_gradient)
    # 独立比对：同一物理点的 base（raw_y 参数化）value_and_grad，绝不重复 canonical 公式
    raw_theta_independent = ms.canonical_raw_theta(result.theta)
    base_value, base_gradient_independent = base.value_and_grad(raw_theta_independent)
    reported = float(np.max(np.abs(canonical)))
    raw_y, q = ms.unpack(result.theta)
    sphere = contact_model.sphere_forward(raw_y)
    coordinate_error = float(np.max(np.abs(sphere - result.coordinates)))
    y_error = float(np.max(np.abs(raw_y - result.y)))
    # 独立 FD：真 raw_y 方向导数 vs canonical 梯度
    rng = np.random.default_rng(5011)
    direction = rng.normal(size=(2, n_loci, 3))
    direction /= np.linalg.norm(direction)
    step = 1e-6
    plus = np.concatenate(((raw_y + step * direction).reshape(-1), [q]))
    minus = np.concatenate(((raw_y - step * direction).reshape(-1), [q]))
    numeric = (float(base.evaluate(plus, False)[0]) - float(base.evaluate(minus, False)[0])) / (2.0 * step)
    analytic = float(np.sum(canonical[:-1].reshape(2, n_loci, 3) * direction))
    relative = abs(analytic - numeric) / max(abs(numeric), 1e-8)
    optimizer_norm = float(np.max(np.abs(optimizer_gradient)))
    history_reported = float(result.history[-1].get("canonical_gradient_max_abs", float("nan")))
    checkpoint = checkpoints[-1]
    checkpoint_sphere = contact_model.sphere_forward(checkpoint["raw_y"])
    checkpoint_coordinate_error = float(np.max(np.abs(checkpoint_sphere - checkpoint["coordinates"])))
    checkpoint_theta_roundtrip = float(np.max(np.abs(ms._optimizer_theta(ms._raw_theta(checkpoint["theta"])) - checkpoint["theta"])))
    distinction = float(np.max(np.abs(np.asarray(optimizer_gradient) - canonical)) / max(reported, 1e-12))
    independent_error = float(np.max(np.abs(np.asarray(base_gradient_independent) - canonical)))
    independent_value_error = abs(float(base_value) - float(result.fun))
    status = "PASS" if (coordinate_error <= 1e-12 and y_error == 0.0 and relative <= 1e-5
                        and checkpoint_coordinate_error <= 1e-12
                        and abs(reported - history_reported) <= 1e-9
                        and independent_error <= 1e-12 and independent_value_error <= 1e-12
                        and distinction > 0.01) else "FAIL"
    return record("gate6_runner_canonical_gradient_nonidentity_P", status,
                  maxfun=6, terminal_reason=result.terminal_reason, nfev=int(result.nfev),
                  reported_canonical_gradient_max_abs=reported, history_reported_canonical_max_abs=history_reported,
                  optimizer_gradient_max_abs=optimizer_norm,
                  optimizer_vs_canonical_relative=float(np.max(np.abs(np.asarray(optimizer_gradient) - canonical))
                                                        / max(reported, 1e-12)),
                  canonical_fd_directional_analytic=analytic, canonical_fd_directional_numeric=numeric,
                  canonical_fd_relative_error=relative,
                  result_y_vs_raw_unpack_max_abs=y_error, result_y_sphere_vs_coordinates_max_abs=coordinate_error,
                  checkpoint_raw_y_sphere_vs_coordinates_max_abs=checkpoint_coordinate_error,
                  checkpoint_optimizer_theta_roundtrip_max_abs=checkpoint_theta_roundtrip,
                  checkpoint_count=len(checkpoints),
                  independent_base_gradient_max_abs_diff=independent_error,
                  independent_base_value_abs_diff=independent_value_error,
                  independent_check="base.value_and_grad(canonical_raw_theta) at the same physical point")


def gate_full_integration() -> dict[str, Any]:
    """12 cell × 2 FG，走同一正式 stage_fit 入口，写 scratch fixture。"""
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    install(SCRATCH)
    data = load_data()
    rows = []
    start_hashes: dict[str, set] = {"consensus": set(), "random": set()}
    stage_z = {loss: {} for loss in LOSSES}
    worst_stage = 0.0
    for loss in LOSSES:
        for solver in ("raw", "ms"):
            for source in ("consensus", "random"):
                set_active(loss, solver)
                coordinates, raw_y, p_init, metadata = load_initial(source)
                q_init = float(contact_model.q_from_p(p_init))
                row = {"fit_id": "%s-%s-%s" % (loss, solver, source), "model_id": loss, "kind": "real",
                       "candidate": source, "start_name": "gate", "objective_variant": "gate", "fixture": None,
                       "weights": {"count": 1.0, "bond": 1.0, "repulsion": 1.0, "bend": 0.01, "p_prior": 1.0},
                       "stages": [{"stage": "1Mb", "bin_size_bp": 1_000_000, "fg_cap": 2}]}
                started = time.perf_counter()
                rec = formal_controller.stage_fit(row, row["stages"][0], data, coordinates, p_init, q_init, None,
                                                  initial_raw_y=raw_y)
                wall = time.perf_counter() - started
                repair = None
                if rec.get("status") != "failure":
                    from round_runner import canonicalize_endpoint, repair_stage_artifacts
                    repair = repair_stage_artifacts(row["fit_id"], loss, solver, data, run_dir=SCRATCH)
                    rec = json.loads((SCRATCH / "stages" / row["fit_id"] / "1Mb.json").read_text(encoding="utf-8"))
                    canonicalize_endpoint(row["fit_id"], solver, run_dir=SCRATCH)
                    with np.load(SCRATCH / "checkpoints" / row["fit_id"] / "1Mb" / "accepted-00000.npz",
                                 allow_pickle=False) as payload:
                        iter0_canonical = np.asarray(payload["canonical_gradient"], dtype=np.float64)
                        iter0_optimizer = np.asarray(payload["optimizer_gradient"], dtype=np.float64)
                    expected = (make_preconditioner(data).Pinv_apply(iter0_optimizer[:-1].reshape(2, int(data.n_loci), 3))
                                if solver == "ms" else iter0_optimizer[:-1].reshape(2, int(data.n_loci), 3))
                    iter0_canonical_error = float(np.max(np.abs(iter0_canonical[:-1].reshape(2, int(data.n_loci), 3) - expected)))
                    q_unchanged = bool(iter0_canonical[-1] == iter0_optimizer[-1])
                else:
                    iter0_canonical_error = float("nan")
                    q_unchanged = False
                stage_z[loss][row["fit_id"]] = wall
                endpoint = SCRATCH / "coords" / row["fit_id"] / "1Mb.npz"
                endpoint3dg = SCRATCH / "coords" / row["fit_id"] / "1Mb.3dg"
                readback_ok = False
                endpoint_record_keys = {}
                if endpoint.exists() and endpoint3dg.exists():
                    with np.load(endpoint, allow_pickle=False) as payload:
                        endpoint_coords = np.asarray(payload["coordinates"], dtype=np.float64).copy()
                        endpoint_raw = np.asarray(payload["raw_y"], dtype=np.float64).copy()
                    tracks = formal_controller._read_full_tracks(endpoint3dg, data)
                    readback_ok = bool(np.array_equal(tracks, endpoint_coords))
                    sphere_ok = float(np.max(np.abs(contact_model.sphere_forward(endpoint_raw) - endpoint_coords)))
                    endpoint_point = rec.get("endpoint") or {}
                    endpoint_record_keys = {
                        "endpoint_p_present": "p" in endpoint_point and math.isfinite(float(endpoint_point["p"])),
                        "endpoint_q_present": "q" in endpoint_point and math.isfinite(float(endpoint_point["q"])),
                        "endpoint_total_present": "total" in endpoint_point and math.isfinite(float(endpoint_point["total"])),
                        "canonical_gradient_max_abs_present": "canonical_gradient_max_abs" in endpoint_point,
                        "history_present": bool(rec.get("history")),
                        "status_present": bool(rec.get("status")),
                    }
                else:
                    sphere_ok = float("nan")
                iter0 = SCRATCH / "checkpoints" / row["fit_id"] / "1Mb" / "accepted-00000.npz"
                start_hashes[source].add(rec.get("initial_state_hashes", {}).get("raw_y_sha256"))
                rows.append({"fit_id": row["fit_id"], "status": rec.get("status"),
                             "terminal_reason": rec.get("terminal_reason"),
                             "outer_fg_actual": rec.get("outer_fg_actual"), "fg_cap": rec.get("fg_cap"),
                             "last_accepted_endpoint": rec.get("last_accepted_endpoint"),
                             "wall_seconds": wall, "iter0_exists": iter0.exists(),
                             "endpoint_npz_exists": endpoint.exists(), "endpoint_3dg_exists": endpoint3dg.exists(),
                             "three_dg_readback_equal": readback_ok,
                             "endpoint_raw_y_sphere_max_abs": sphere_ok,
                             "initial_raw_y_sha256": rec.get("initial_state_hashes", {}).get("raw_y_sha256"),
                             "endpoint_record_keys": endpoint_record_keys,
                             "iter0_canonical_vs_pinv_optimizer_max_abs": iter0_canonical_error,
                             "iter0_q_unchanged": q_unchanged,
                             "endpoint_raw_y_q_gradient_inf": (rec.get("endpoint") or {}).get("raw_y_q_gradient_inf"),
                             "endpoint_optimizer_gradient_inf": (rec.get("endpoint") or {}).get("optimizer_gradient_inf"),
                             "hard_error": bool(rec.get("hard_error", False))})
                worst_stage = max(worst_stage, wall)
    install(RUN)
    statuses_ok = all(row["status"] in ("converged", "not_converged", "budget_not_converged") for row in rows)
    fg_ok = all(int(row["outer_fg_actual"]) <= 2 for row in rows)
    readback_ok = all(row["three_dg_readback_equal"] for row in rows)
    sphere_ok = all(math.isfinite(row["endpoint_raw_y_sphere_max_abs"]) and row["endpoint_raw_y_sphere_max_abs"] <= 1e-12
                    for row in rows)
    keys_ok = all(all(bool(value) for value in row["endpoint_record_keys"].values()) for row in rows)
    iter0_ok = all(math.isfinite(row["iter0_canonical_vs_pinv_optimizer_max_abs"])
                   and row["iter0_canonical_vs_pinv_optimizer_max_abs"] <= 1e-10 and row["iter0_q_unchanged"]
                   for row in rows)
    gradient_labels_ok = all(row["endpoint_raw_y_q_gradient_inf"] is not None
                             and row["endpoint_optimizer_gradient_inf"] is not None for row in rows)
    isolation_ok = all(len(values) == 1 for values in start_hashes.values()) and len(
        set.union(*start_hashes.values())) == 2
    status = "PASS" if (len(rows) == 12 and statuses_ok and fg_ok and readback_ok and sphere_ok and isolation_ok
                        and keys_ok and iter0_ok and gradient_labels_ok) else "FAIL"
    return record("gate7_12cell_2fg_full_integration", status, rows=rows,
                  shared_start_isolation=isolation_ok, endpoint_record_keys_ok=keys_ok,
                  iter0_canonical_repair_ok=iter0_ok, gradient_labels_ok=gradient_labels_ok,
                  start_hashes={key: sorted(str(v) for v in values) for key, values in start_hashes.items()},
                  worst_cell_wall_seconds=worst_stage, scratch=str(SCRATCH.relative_to(ROOT)))


def gate_lineage_and_denominators() -> dict[str, Any]:
    data = load_data()
    budget = data.budget()
    denominators = {
        "raw_records": float(data.raw_records), "raw_same_bin": float(data.raw_same_bin),
        "raw_cis_offdiag": float(data.raw_cis_offdiag), "raw_inter": float(data.raw_inter),
        "Noff": float(data.raw_cis_offdiag) + float(data.raw_inter),
        "n_pairs_full_grid": int(data.n_pairs), "n_loci": int(data.n_loci),
        "counts_cis": float(np.asarray(data.counts)[np.asarray(data.cis_pair)].sum()),
        "counts_inter": float(np.asarray(data.counts)[~np.asarray(data.cis_pair)].sum()),
        "diag_counts_total": float(np.asarray(data.diag_counts).sum()),
    }
    denominators_ok = (denominators["raw_records"] == 1703888.0 and denominators["raw_same_bin"] == 438774.0
                       and denominators["raw_cis_offdiag"] == 696680.0 and denominators["raw_inter"] == 568434.0
                       and denominators["Noff"] == 1265114.0 and denominators["n_pairs_full_grid"] == 3496690
                       and denominators["counts_cis"] == 696680.0 and denominators["counts_inter"] == 568434.0
                       and denominators["diag_counts_total"] == 438774.0)
    # zero-optimization prolongation 可复现性：只用 045 冻结文件
    lineage_rows = []
    lineage_ok = True
    data_by_bin = {2_000_000: data_io.load_aggregate(SOURCE_045 / "inputs/real_2000000_aggregate.npz"),
                   1_000_000: data}
    for source in ("consensus", "random"):
        with np.load(INITIAL_5MB[source], allow_pickle=False) as payload:
            coords5 = np.asarray(payload["coordinates"], dtype=np.float64).copy()
            meta5 = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        with np.load(INITIAL_2MB[source], allow_pickle=False) as payload:
            coords2_frozen = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        with np.load(INITIAL_5MB[source], allow_pickle=False) as payload:
            raw5 = np.asarray(payload["raw_y"], dtype=np.float64).copy()
        seed = PROLONGATION_SEED[source]
        data2 = data_by_bin[2_000_000]
        # 用 5Mb 层自身的网格元数据构造 positions/chromosome_index
        data5 = data_io.load_aggregate(SOURCE_045 / "inputs/real_5000000_aggregate.npz")
        positions5 = data5.locus_bin * int(data5.bin_size)
        chrom5 = data5.locus_chromosome
        warm2 = reconstruction_init.warm_start_from_layer(
            coords5, positions5, chrom5, tuple(data2.chromosome_names),
            tuple(int(v) for v in data2.chromosome_lengths), 2_000_000, seed)
        warm1 = reconstruction_init.warm_start_from_layer(
            np.asarray(warm2["coords"], dtype=np.float64), np.asarray(warm2["positions"], dtype=np.int64),
            np.asarray(warm2["chromosome_index"], dtype=np.int32), tuple(data.chromosome_names),
            tuple(int(v) for v in data.chromosome_lengths), 1_000_000, seed)
        coordinates, raw_y, p_init, metadata = load_initial(source)
        error2 = float(np.max(np.abs(np.asarray(warm2["coords"], dtype=np.float64) - coords2_frozen)))
        error1 = float(np.max(np.abs(np.asarray(warm1["coords"], dtype=np.float64) - coordinates)))
        raw_error = float(np.max(np.abs(contact_model.sphere_inverse(coordinates) - raw_y)))
        lineage_ok = lineage_ok and error2 <= 1e-12 and error1 <= 1e-12 and raw_error <= 1e-12
        lineage_rows.append({"source": source, "seed": seed, "prolongation_5to2_max_abs_error": error2,
                             "prolongation_2to1_max_abs_error": error1,
                             "raw_y_vs_sphere_inverse_max_abs_error": raw_error,
                             "five_mb_gate_stage": meta5.get("metadata", {}).get("source", {}).get("gate_stage"),
                             "five_mb_source_sha256": meta5.get("metadata", {}).get("source", {}).get("source_sha256"),
                             "five_mb_no_optimization": meta5.get("no_optimization"),
                             "one_mb_raw_y_sha256": array_sha256(raw_y),
                             "reference_opened": False})
    status = "PASS" if (denominators_ok and lineage_ok) else "FAIL"
    return record("gate8_lineage_and_denominators", status, denominators=denominators,
                  denominators_ok=denominators_ok, lineage=lineage_rows, budget=budget)


def gate_frozen_bytes(pre: dict[str, str]) -> dict[str, Any]:
    post = tracked_hashes()
    changed = sorted(key for key in set(pre) | set(post) if pre.get(key) != post.get(key))
    status = "PASS" if not changed else "FAIL"
    return record("gate9_frozen_bytes_unchanged", status, tracked_files=sorted(pre), changed=changed,
                  hashes=post)


def main() -> int:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    pre = tracked_hashes()
    results = [gate_loss_a_parity(), gate_finite_difference(), gate_ties_and_gauge(),
               gate_symmetry_invariance(), gate_preconditioner(), gate_runner_canonical(),
               gate_full_integration(), gate_lineage_and_denominators()]
    results.append(gate_frozen_bytes(pre))
    report = {"schema": "p9016-max-contact-gates-v1", "started_at_utc": started,
              "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "gate_count": len(results),
              "passed": sum(1 for row in results if row["status"] == "PASS"),
              "failed": [row["gate"] for row in results if row["status"] != "PASS"],
              "gates": results, "reference_opened": False, "phase_opened": False, "synthetic_formal_fits": 0}
    GATES.mkdir(parents=True, exist_ok=True)
    (GATES / "gates_report.json").write_text(json.dumps(report, sort_keys=True, indent=2, default=str,
                                                        allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if not report["failed"] else "FAIL", "passed": report["passed"],
                      "failed": report["failed"]}))
    return 0 if not report["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
