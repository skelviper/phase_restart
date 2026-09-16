"""P2（050 轮）：046 工作 baseline 上的无参考、无标签中心 / 刚体块可辨识性 probe。

设计（root config.json 的 p2 段已冻结，本脚本不因参考或评价结果改动）：

* family 1 `centre_translation`：每条染色体两个 copy 一起平移同一个逐 chr 向量，
  保持该 chr 内部几何与同 chr 两 copy 的相对几何。
* family 2 `rigid_block_rotation`：每条染色体两个 copy 作为整体绕该 chr 的共同中心
  做真正 SO(3) 旋转（Rodrigues，det=+1），同样保持上述几何。实现要点：位移场必须是
  `(R - I)(x - c)`；**不能**用 `R(x - c)`（那是已旋转坐标，把它按幅度线性混合得到的是
  “缩放+旋转”的非刚性映射）。又因为把固定旋转位移场乘以 s≠1 会破坏正交性，本脚本按冻结
  幅度定义对每条 probe 解一个标量 α，使**精确旋转**位移场的全细胞 RMS 等于 target
  （“按 RMS 标定角度”），因此 probe 坐标仍然是逐 chr 的严格刚体变换。
* 幅度：±0.01 / ±0.05 × baseline whole-cell Rg；两种 family 共用同一条幅度定义：
  **去掉全局均值位移之后的全细胞位移 RMS**。
* 方向由 seed 461001 预先固定，只看 baseline 坐标，不看 reference / phase。
* 全局平移消去：位移场减去全局均值位移（等价于整细胞平移，不破坏任何块的刚性，
  也不改变任何 chr 间相对几何），随后整体归一化到 RMS=1。
  **probe 坐标没有去掉全局转动**：本脚本只消全局平移分量。整体转动分量用 proper
  Kabsch 对齐后的 residual RMS 独立量化（field `rms_deformation_after_proper_rigid`），
  该读数不改变 probe 坐标与幅度定义，check 也拆成 `translation_removed` 与
  `rotation_accounted_in_aligned_readout` 两项，绝不声称原坐标已去全局 rotation。
* 球域：若严格单位球会被违反，整条 probe 用一个统一因子折半缩小并如实报告实际 RMS；
  不做逐 bead 投影 / 裁剪，也不对已旋转坐标做线性混合。
* e 与 p 固定在 baseline 值（p 直接取文件精确值），不拟合、不扫幅、不选点。
* 本进程不打开 reference 3DG / phase 列，不 import pr.refeval / pr.ref3dg / pr.labels；
  允许读的旧代码只有 049 训练侧 analysis_core（three_loss_components /
  fine_normalized_rates / whole_cell_rg / load_layer），049 旧目录只读。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
ROOT = RUN_DIR.parents[1]
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
if str(S049) not in sys.path:
    sys.path.insert(0, str(S049))

import analysis_core as ac  # noqa: E402
from frozen_imports import contact_model  # noqa: E402

BASELINE_NPZ = ROOT / "test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.npz"
BASELINE_3DG = ROOT / "test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.3dg"
BASELINE_NPZ_SHA256 = "116e906e790493afa956445c527b38da11b2fdcaed482fd225f578f188df699d"
BASELINE_3DG_SHA256 = "4301d4df6e89c1417690599d6687a9e83b18fa37596de5d6c11a911d867d69ea"
BASELINE_NPZ_RAW_Y_SHA256 = None  # 不设：raw_y 不是本轮的冻结锚
# 046 evaluation_final / 049 pre_reference_gate 共同 rescore 的冻结值（count 9.529134634154618 =
# 9.5291346341546177 的 float64 表示；fullJ 在 046 是 ...452，在 049 gate 是 ...453，相差 1 ulp）
FROZEN_COUNT_A = 9.5291346341546177
FROZEN_FULLJ_A = 9.542018998907453
FROZEN_PARITY_TOL = 1e-8

NRAW = 1703888
N_OFF = 1265114
N_CIS_RAW = 696680
N_INTER_RAW = 568434
N_DIAG = 438774
N_LOCI = 2645
N_PAIRS = 3496690
# 3,496,690 是完整 off-diagonal pair 网格（零计数 pair 全部保留，不裁掉）；
# 其中 counts>0 的 pair 只有 487,254 个（1,265,114 条 offdiag 记录分布其上）。
# 依据：046 evaluation_final/results/trajectory_summary.tsv 的 696680/568434/1265114/83947/403307/487254 列。
N_OBSERVED_PAIRS = 487254
N_ZERO_COUNT_PAIRS = N_PAIRS - N_OBSERVED_PAIRS

SEED = 461001
FAMILIES = ("centre_translation", "rigid_block_rotation")
FRACTIONS = (0.01, 0.05)
SIGNS = (1.0, -1.0)
BASE_ROTATION_ANGLE_RAD = 1.0e-3
FORBIDDEN_MODULES = ("pr.refeval", "pr.ref3dg", "pr.labels")
INTRA_TOL_ABS = 1e-12
RATE_TOL_REL = 1e-12
MEAN_SHIFT_TOL = 1e-12
RIGID_TOL_ABS = 1e-12
COUNT_IDENTITY_TOL = 1e-9


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False, default=str) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns and not key.startswith("_"):
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join("" if row.get(key) is None else str(row.get(key))
                                   for key in columns) + "\n")


# ------------------------------------------------------------------ geometry
def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise RuntimeError("direction vector must be nonzero")
    return vector / norm


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues 公式，det=+1 的真 SO(3) 旋转。"""
    axis = unit(axis)
    x, y, z = axis
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ], dtype=np.float64)


def remove_global_translation(field: np.ndarray) -> tuple[np.ndarray, float]:
    """减去全局均值位移（整细胞平移）；返回处理后的场与被移除均值位移的范数。"""
    values = np.asarray(field, dtype=np.float64)
    mean = values.reshape(-1, 3).mean(axis=0)
    return values - mean[None, None, :], float(np.linalg.norm(mean))


def displacement_rms(field: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64).reshape(-1, 3)
    return float(np.sqrt(np.mean(np.sum(values ** 2, axis=1))))


def normalise_rms(field: np.ndarray) -> tuple[np.ndarray, float]:
    rms = displacement_rms(field)
    if rms <= 0.0:
        raise RuntimeError("displacement field has zero RMS")
    return np.asarray(field, dtype=np.float64) / rms, rms


def feasible_amplitude(coordinates: np.ndarray, direction: np.ndarray, target: float):
    """严格单位球内可行幅度：不合法时整条 probe 统一折半，绝不逐 bead 裁剪。"""
    scale = float(target)
    halvings = 0
    for _ in range(80):
        candidate = coordinates + scale * direction
        peak = float(np.linalg.norm(candidate.reshape(-1, 3), axis=1).max())
        if peak < 1.0:
            return candidate, scale, peak, halvings
        scale *= 0.5
        halvings += 1
    raise RuntimeError("probe amplitude could not be made feasible")


def rotation_displacement(block: np.ndarray, centre: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """真 SO(3) 旋转的位移场 (R(angle, axis) - I)(x - centre)，即刚体位移，不是已旋转坐标。"""
    offset = np.asarray(block, dtype=np.float64) - np.asarray(centre, dtype=np.float64)[None, None, :]
    matrix = rotation_matrix(axis, angle)
    return np.einsum("ij,bkj->bki", matrix, offset) - offset


def rigid_rotation_field(coordinates: np.ndarray, data: Any, centres: list[np.ndarray],
                         axes: list[np.ndarray], signs: list[float], angle_sign: float,
                         target_abs: float, max_iter: int = 40, tol: float = 1e-14):
    """按 RMS 标定角度：解标量 alpha 使精确刚体旋转位移场的全细胞（去全局均值后）RMS = target_abs。

    返回 (field, alpha, history)。field 是精确刚体位移场；整体缩放由 alpha 承担，绝不对固定
    位移场做线性缩放（那会破坏正交性）。
    """
    field = np.zeros_like(np.asarray(coordinates, dtype=np.float64))
    alpha = 1.0
    history: list[dict[str, float]] = []
    for _ in range(int(max_iter)):
        for chromosome in range(len(data.chromosome_names)):
            slc = data.chromosome_slice(chromosome)
            field[:, slc, :] = rotation_displacement(
                coordinates[:, slc, :], centres[chromosome], axes[chromosome],
                float(angle_sign) * alpha * BASE_ROTATION_ANGLE_RAD * signs[chromosome])
        centred, _removed = remove_global_translation(field)
        rms = displacement_rms(centred)
        history.append({"alpha": float(alpha), "displacement_rms": float(rms)})
        if rms <= 0.0:
            raise RuntimeError("degenerate rotation displacement field")
        if abs(rms - float(target_abs)) <= float(tol) * float(target_abs):
            break
        alpha *= float(target_abs) / rms
    return field, float(alpha), history


def kabsch_proper(reference: np.ndarray, moving: np.ndarray) -> dict[str, Any]:
    """proper Kabsch（强制 det=+1）：把 moving 刚体对齐到 reference，返回 residual RMS 与旋转量。"""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    mov = np.asarray(moving, dtype=np.float64).reshape(-1, 3)
    ref_c = ref - ref.mean(axis=0)
    mov_c = mov - mov.mean(axis=0)
    u, _s, vt = np.linalg.svd(mov_c.T @ ref_c)
    rotation = vt.T @ u.T
    if float(np.linalg.det(rotation)) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    residual = mov_c @ rotation.T - ref_c
    trace = float(np.trace(rotation))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))
    return {"rms": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
            "det": float(np.linalg.det(rotation)), "angle_deg": angle,
            "rotation": rotation.tolist()}


def intra_distances(coordinates: np.ndarray, data: Any) -> np.ndarray:
    """每条染色体内部全部 bead 对的欧氏距离（含同 chr 两 copy 的四个 copy 组合与同 locus 跨 copy 对）。"""
    values = []
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        block = np.asarray(coordinates[:, slc, :], dtype=np.float64).reshape(-1, 3)
        i, j = np.triu_indices(len(block), k=1)
        values.append(np.linalg.norm(block[i] - block[j], axis=1))
    return np.concatenate(values)


def merged_chromosome_centres(coordinates: np.ndarray, data: Any) -> np.ndarray:
    rows = []
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        rows.append(np.asarray(coordinates[:, slc, :], dtype=np.float64).reshape(-1, 3).mean(axis=0))
    return np.asarray(rows)


def copy_centres(coordinates: np.ndarray, data: Any) -> np.ndarray:
    rows = []
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        block = np.asarray(coordinates[:, slc, :], dtype=np.float64)
        rows.append(block[0].mean(axis=0))
        rows.append(block[1].mean(axis=0))
    return np.asarray(rows)


def point_distance_stats(base_points: np.ndarray, probe_points: np.ndarray,
                         mask: np.ndarray | None = None) -> dict[str, Any]:
    """点上三角 pair 距离变化统计；mask 用于把分母固定成明确的集合（例如跨 chr 的 760 个 copy 中心对）。"""
    i, j = np.triu_indices(len(base_points), k=1)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != i.shape:
            raise ValueError("centre pair mask shape mismatch")
        i, j = i[mask], j[mask]
    base = np.linalg.norm(base_points[i] - base_points[j], axis=1)
    probe = np.linalg.norm(probe_points[i] - probe_points[j], axis=1)
    diff = probe - base
    return {"n_pairs": int(len(base)),
            "mean_baseline_distance": float(base.mean()),
            "signed_mean_change": float(diff.mean()),
            "mean_abs_change": float(np.abs(diff).mean()),
            "max_abs_change": float(np.abs(diff).max()),
            "rms_change": float(np.sqrt(np.mean(diff ** 2))),
            "mean_relative_change": float(np.mean(np.abs(diff) / np.maximum(base, 1e-300)))}


def copy_centre_chromosome_index(data: Any) -> np.ndarray:
    """40 个 copy 中心点对应的染色体下标（每条 chr 两个：copy0、copy1）。"""
    return np.repeat(np.arange(len(data.chromosome_names), dtype=np.int64), 2)


def centre_pair_masks(chromosome_index: np.ndarray) -> dict[str, np.ndarray]:
    """中心点对（上三角 i<j）的固定分母掩码。

    40 个 copy 中心全对 = C(40,2) = 780，其中同 chr 的两个 homolog 中心 20 对不是跨 chr 摆位信息；
    跨 chr 集合 = 190 个 chr 组合 × 4 个 copy 组合 = 760，是主读出字段 `copy_centre_760_*` 的分母。
    """
    n = len(chromosome_index)
    i, j = np.triu_indices(n, k=1)
    same = chromosome_index[i] == chromosome_index[j]
    return {"all": np.ones(len(i), dtype=bool), "same_chromosome": same, "cross_chromosome": ~same,
            "pair_i": i, "pair_j": j}


def per_chromosome_rigid(coordinates: np.ndarray, candidate: np.ndarray, data: Any) -> dict[str, Any]:
    residuals, dets, angles = [], [], []
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        ref = np.asarray(coordinates[:, slc, :], dtype=np.float64).reshape(-1, 3)
        mov = np.asarray(candidate[:, slc, :], dtype=np.float64).reshape(-1, 3)
        fit = kabsch_proper(ref, mov)
        residuals.append(fit["rms"])
        dets.append(fit["det"])
        angles.append(fit["angle_deg"])
    return {"residual_rms_max": float(np.max(residuals)),
            "residual_rms_median": float(np.median(residuals)),
            "rotation_det_min": float(np.min(dets)), "rotation_det_max": float(np.max(dets)),
            "rotation_angle_deg_max": float(np.max(angles)),
            "rotation_angle_deg_median": float(np.median(angles)),
            "per_chromosome_angle_deg": [float(a) for a in angles],
            "per_chromosome_residual_rms": [float(r) for r in residuals]}


def baseline_gradient_consistency(data: Any, coordinates: np.ndarray, p_value: float,
                                  seed: int = SEED + 1, n_random: int = 2,
                                  eps: float = 1e-5) -> dict[str, Any]:
    """baseline 点的 value/gradient 自洽：loss A 的解析方向导数 vs 中心差分（不拟合任何东西）。

    config.p2.checks 里 “baseline value/gradient consistency” 的落地：值的一致性由 frozen
    count/fullJ parity 给出，这里补上梯度与同一点值的自洽性（同一 objective、同一 theta 空间）。
    方向取 2 个固定随机方向 + 1 个梯度对齐方向（后者 |解析导数| = |grad|，信噪比最高）。
    步长 1e-5 是实测选择：1e-3 时截断误差主导（rel~1e-3），1e-5 时数值底噪约 1e-10。
    """
    objective = ac.build_objective(data, "A")
    raw_y = np.asarray(contact_model.sphere_inverse(coordinates), dtype=np.float64)
    theta = np.concatenate((raw_y.reshape(-1),
                            np.asarray([float(contact_model.q_from_p(float(p_value)))], dtype=np.float64)))
    value, gradient, _components = objective.evaluate(theta, True)
    if hasattr(gradient, "detach"):
        gradient = gradient.detach().cpu().numpy()
    gradient = np.asarray(gradient, dtype=np.float64)
    gradient_norm = float(np.linalg.norm(gradient))
    rng = np.random.default_rng(int(seed))
    vectors = []
    for _ in range(int(n_random)):
        vector = rng.normal(size=theta.size)
        vectors.append(("random_%d" % len(vectors), vector / float(np.linalg.norm(vector))))
    vectors.append(("gradient_aligned", gradient / gradient_norm))
    directions = []
    for label, vector in vectors:
        plus, _g1, _c1 = objective.evaluate(theta + eps * vector, False)
        minus, _g2, _c2 = objective.evaluate(theta - eps * vector, False)
        finite_difference = (float(plus) - float(minus)) / (2.0 * float(eps))
        analytic = float(np.dot(gradient, vector))
        directions.append({"direction": label, "analytic_directional_derivative": analytic,
                           "finite_difference": finite_difference,
                           "abs_error": abs(finite_difference - analytic),
                           "rel_error": abs(finite_difference - analytic) / max(abs(analytic), 1e-12)})
    max_abs = max(row["abs_error"] for row in directions)
    aligned_rel = [row["rel_error"] for row in directions if row["direction"] == "gradient_aligned"][0]
    return {"objective": "loss A full-J (count NLL + weighted regularisation) at the baseline point; "
                         "nothing is fitted and no reference is used",
            "value": float(value), "gradient_inf_norm": float(np.max(np.abs(gradient))),
            "gradient_norm": gradient_norm, "n_directions": len(directions),
            "central_difference_step": float(eps), "seed": int(seed), "directions": directions,
            "max_abs_error": float(max_abs), "gradient_aligned_rel_error": float(aligned_rel),
            "abs_tolerance": 1e-8, "gradient_aligned_rel_tolerance": 1e-4,
            "passed": bool(max_abs <= 1e-8 and aligned_rel <= 1e-4)}


# ------------------------------------------------------------------ directions
def build_directions(data: Any, coordinates: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """按固定 seed 生成平移方向场（只消全局平移，RMS=1）与旋转 family 的固定轴/符号/中心。"""
    n_loci = int(data.n_loci)
    directions: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {}
    scales: dict[str, float] = {}

    translation = np.zeros((2, n_loci, 3), dtype=np.float64)
    vectors = []
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        vector = unit(rng.normal(size=3))
        vectors.append(vector)
        translation[0, slc, :] = vector
        translation[1, slc, :] = vector
    field, removed = remove_global_translation(translation)
    directions["centre_translation"], scales["centre_translation"] = normalise_rms(field)
    audit["centre_translation"] = {
        "label": "both copies of one chromosome translated together by a common per-chromosome vector",
        "construction": "per-chromosome rng.normal(size=3) normalised to unit length, shared by both copies; "
                        "global mean displacement removed; whole-cell displacement RMS normalised to 1",
        "n_chromosome_vectors": len(vectors),
        "removed_global_mean_displacement_norm_before_normalisation": removed,
        "rms_before_normalisation": scales["centre_translation"],
        "per_chromosome_unit_vectors": [list(map(float, v)) for v in vectors],
    }

    axes, sign_list, centres = [], [], []
    for chromosome in range(len(data.chromosome_names)):
        slc = data.chromosome_slice(chromosome)
        axes.append(unit(rng.normal(size=3)))
        sign_list.append(1.0 if float(rng.random()) < 0.5 else -1.0)
        centres.append(np.asarray(coordinates[:, slc, :], dtype=np.float64).reshape(-1, 3).mean(axis=0))
    det_list = [float(np.linalg.det(rotation_matrix(axes[c], BASE_ROTATION_ANGLE_RAD * sign_list[c])))
                for c in range(len(axes))]
    audit["rigid_block_rotation"] = {
        "label": "both copies of one chromosome rotated together as a rigid block about their common centre",
        "construction": "per-chromosome rng.normal(size=3) axis normalised to unit length and a per-chromosome "
                        "random sign, both fixed by the seed; the probe displacement field is the EXACT SO(3) "
                        "rotation field (R - I)(x - c) about the merged chromosome centre, with a single scalar "
                        "angle-scale alpha solved so that the whole-cell displacement RMS (after removal of the "
                        "single global mean displacement) equals the frozen target amplitude; a fixed rotation "
                        "displacement field is never linearly scaled because that would break orthogonality",
        "base_angle_rad_used_as_relative_weighting": BASE_ROTATION_ANGLE_RAD,
        "construction_rotation_det_min": float(np.min(det_list)),
        "construction_rotation_det_max": float(np.max(det_list)),
        "per_chromosome_axis": [list(map(float, a)) for a in axes],
        "per_chromosome_angle_sign": [float(s) for s in sign_list],
    }
    return {"directions": directions, "audit": audit, "rms_before_normalisation": scales,
            "rotation_axes": axes, "rotation_signs": sign_list, "rotation_centres": centres}


# ------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description="050 P2 label-free centre / rigid-block probe (no fitting, no reference)")
    parser.parse_args()
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.time()
    p2 = RUN_DIR / "p2"
    (p2 / "probes").mkdir(parents=True, exist_ok=True)
    (p2 / "results").mkdir(parents=True, exist_ok=True)

    data = ac.load_layer(1_000_000)
    denominators = {
        "n_loci": int(data.n_loci), "n_pairs": int(data.n_pairs),
        "Nraw": int(data.raw_records), "cis_offdiag_raw": int(data.raw_cis_offdiag),
        "inter_raw": int(data.raw_inter), "diag_saturated_raw": int(np.asarray(data.diag_counts).sum()),
        "Noff": int(np.asarray(data.counts).sum()),
        "observed_pairs": int((np.asarray(data.counts) > 0).sum()),
        "zero_count_pairs_retained": int((np.asarray(data.counts) == 0).sum()),
        "cis_locus_pairs": int(np.asarray(data.cis_pair).sum()),
    }
    expected_denominators = {"n_loci": N_LOCI, "n_pairs": N_PAIRS, "Nraw": NRAW, "cis_offdiag_raw": N_CIS_RAW,
                             "inter_raw": N_INTER_RAW, "diag_saturated_raw": N_DIAG, "Noff": N_OFF,
                             "observed_pairs": N_OBSERVED_PAIRS,
                             "zero_count_pairs_retained": N_ZERO_COUNT_PAIRS}
    denominator_mismatch = {key: (denominators[key], value) for key, value in expected_denominators.items()
                            if denominators[key] != value}
    if denominator_mismatch:
        raise RuntimeError("P2 denominator mismatch vs frozen config: %s" % denominator_mismatch)

    baseline_npz_sha = sha256_file(BASELINE_NPZ)
    if baseline_npz_sha != BASELINE_NPZ_SHA256:
        raise RuntimeError("046 work baseline NPZ SHA256 mismatch: %s" % baseline_npz_sha)
    baseline_3dg_sha = sha256_file(BASELINE_3DG)
    if baseline_3dg_sha != BASELINE_3DG_SHA256:
        raise RuntimeError("046 work baseline 3DG SHA256 mismatch: %s" % baseline_3dg_sha)
    with np.load(BASELINE_NPZ, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        p_value = float(np.asarray(payload["p"]).item())
        q_value = float(np.asarray(payload["q"]).item())
        npz_keys = list(payload.files)
    contact_model.assert_inside_unit_ball(coordinates)
    q_from_p = float(contact_model.q_from_p(p_value))
    p_from_q, _dpdq = contact_model.p_from_q(q_value)

    baseline_rg = ac.whole_cell_rg(coordinates)
    baseline_rates = ac.fine_normalized_rates(data, coordinates, p_value)
    baseline_audit = ac.three_loss_components(data, coordinates, p_value)
    baseline_intra = intra_distances(coordinates, data)
    baseline_merged = merged_chromosome_centres(coordinates, data)
    baseline_copy = copy_centres(coordinates, data)
    merged_index = np.arange(len(data.chromosome_names), dtype=np.int64)
    copy_index = copy_centre_chromosome_index(data)
    merged_masks = centre_pair_masks(merged_index)
    copy_masks = centre_pair_masks(copy_index)
    if not (merged_masks["all"].sum() == 190 and copy_masks["cross_chromosome"].sum() == 760
            and copy_masks["all"].sum() == 780 and copy_masks["same_chromosome"].sum() == 20):
        raise AssertionError("centre pair denominators are not 190 / 760 / 780 / 20")
    counts = np.asarray(data.counts, dtype=np.float64)
    observed = counts > 0.0
    cis_mask = np.asarray(data.cis_pair, dtype=bool)
    inter_mask = ~cis_mask

    parity = {
        "count_A_recomputed": float(baseline_audit["count_A"]),
        "count_A_frozen": FROZEN_COUNT_A,
        "count_A_abs_diff": abs(float(baseline_audit["count_A"]) - FROZEN_COUNT_A),
        "fullJ_A_recomputed": float(baseline_audit["fullJ_A"]),
        "fullJ_A_frozen": FROZEN_FULLJ_A,
        "fullJ_A_abs_diff": abs(float(baseline_audit["fullJ_A"]) - FROZEN_FULLJ_A),
        "tolerance": FROZEN_PARITY_TOL,
        "source": "046 evaluation_final full_j_endpoint_diagnostics.json (9.529134634154618 / 9.542018998907452) "
                  "and 049 pre_reference_gate.json (9.542018998907453, 1 ulp apart)",
    }
    parity["passed"] = bool(parity["count_A_abs_diff"] <= FROZEN_PARITY_TOL
                            and parity["fullJ_A_abs_diff"] <= FROZEN_PARITY_TOL)
    if not parity["passed"]:
        raise RuntimeError("baseline value parity failed: %s" % parity)
    rates_sum_rel_error = abs(float(baseline_rates.sum()) - float(baseline_audit["Zsum"])) / float(baseline_audit["Zsum"])
    if not bool(np.all(np.isfinite(baseline_rates)) and np.all(baseline_rates > 0.0)):
        raise RuntimeError("baseline rates contain non-positive or non-finite values")
    gradient_check = baseline_gradient_consistency(data, coordinates, p_value)

    rng = np.random.default_rng(SEED)
    built = build_directions(data, coordinates, rng)
    directions = built["directions"]
    rms_before_normalisation = built["rms_before_normalisation"]

    rows: list[dict[str, Any]] = []
    probe_index: list[dict[str, Any]] = []
    for family in FAMILIES:
        for fraction in FRACTIONS:
            for sign in SIGNS:
                target = float(sign) * float(fraction) * baseline_rg
                rotation_meta: dict[str, Any] | None = None
                if family == "centre_translation":
                    direction = directions["centre_translation"]
                    candidate, actual, peak, halvings = feasible_amplitude(coordinates, direction, target)
                else:
                    # 旋转 family：对每条 probe 解一个标量 alpha，使精确刚体旋转位移场的 RMS = target
                    rms_target = abs(target)
                    halvings = 0
                    while True:
                        field, alpha, history = rigid_rotation_field(
                            coordinates, data, built["rotation_centres"], built["rotation_axes"],
                            built["rotation_signs"], float(sign), rms_target)
                        centred_field, removed_mean = remove_global_translation(field)
                        rms_measured = displacement_rms(centred_field)
                        candidate = coordinates + centred_field
                        peak = float(np.linalg.norm(candidate.reshape(-1, 3), axis=1).max())
                        if peak < 1.0:
                            break
                        rms_target *= 0.5
                        halvings += 1
                        if halvings > 40:
                            raise RuntimeError("rotation probe amplitude could not be made feasible")
                    actual = float(sign) * rms_measured
                    direction = centred_field / actual
                    rotation_meta = {
                        "rotation_angle_scale_alpha": float(alpha),
                        "rotation_alpha_iterations": int(len(history)),
                        "rotation_rms_residual_rel": float(abs(rms_measured - rms_target) / rms_target),
                        "rotation_field_global_mean_removed_norm": float(removed_mean),
                        "rotation_angle_deg_per_chromosome": math.degrees(abs(alpha) * BASE_ROTATION_ANGLE_RAD),
                        "rotation_alpha_history": history,
                    }
                contact_model.assert_inside_unit_ball(candidate)
                displacement = candidate - coordinates
                raw_rms = displacement_rms(displacement)
                mean_displacement = displacement.reshape(-1, 3).mean(axis=0)
                centred, removed_mean_norm = remove_global_translation(displacement)
                post_rms = displacement_rms(centred)

                audit = ac.three_loss_components(data, candidate, p_value)
                probe_rates = ac.fine_normalized_rates(data, candidate, p_value)
                if not bool(np.all(np.isfinite(probe_rates)) and np.all(probe_rates > 0.0)):
                    raise RuntimeError("probe rates contain non-positive or non-finite values")
                q_base = baseline_rates / baseline_rates.sum()
                q_probe = probe_rates / probe_rates.sum()
                kl = float(np.sum(q_base * np.log(q_base / q_probe)))
                observed_log_delta = float(np.sum(counts[observed]
                                                  * np.log(baseline_rates[observed] / probe_rates[observed])))
                normalizer_delta = float(baseline_audit["n_off"]
                                         * math.log(float(audit["Zsum"]) / float(baseline_audit["Zsum"])))
                count_delta_raw = observed_log_delta + normalizer_delta
                normalized_delta = count_delta_raw / float(baseline_audit["Nraw"])
                count_a_delta = float(audit["count_A"] - baseline_audit["count_A"])
                identity_error = abs(normalized_delta - count_a_delta)
                if identity_error > COUNT_IDENTITY_TOL:
                    raise AssertionError("probe count delta does not match count_A delta: %.3e" % identity_error)

                candidate_intra = intra_distances(candidate, data)
                intra_abs = np.abs(candidate_intra - baseline_intra)
                cis_rel = np.abs(probe_rates[cis_mask] / baseline_rates[cis_mask] - 1.0)
                inter_rel = np.abs(probe_rates[inter_mask] / baseline_rates[inter_mask] - 1.0)
                rigid = per_chromosome_rigid(coordinates, candidate, data)
                global_fit = kabsch_proper(coordinates, candidate)
                merged_probe = merged_chromosome_centres(candidate, data)
                copy_probe = copy_centres(candidate, data)
                merged_stats = point_distance_stats(baseline_merged, merged_probe, merged_masks["all"])
                copy_cross = point_distance_stats(baseline_copy, copy_probe, copy_masks["cross_chromosome"])
                copy_all = point_distance_stats(baseline_copy, copy_probe, copy_masks["all"])
                copy_same = point_distance_stats(baseline_copy, copy_probe, copy_masks["same_chromosome"])
                if (merged_stats["n_pairs"], copy_cross["n_pairs"], copy_all["n_pairs"], copy_same["n_pairs"]) \
                        != (190, 760, 780, 20):
                    raise AssertionError("centre pair denominators are not 190 / 760 / 780 / 20")
                centre_disp = np.linalg.norm(merged_probe - baseline_merged, axis=1)
                expected_angle = float(rotation_meta["rotation_angle_deg_per_chromosome"]) if rotation_meta else 0.0

                name = "%s_%s_%s" % (family, "plus" if sign > 0 else "minus", ("%.2f" % fraction).replace(".", "p"))
                path = p2 / "probes" / ("%s.npz" % name)
                np.savez_compressed(
                    path, coordinates=candidate, baseline_coordinates=coordinates, direction=direction,
                    family=np.asarray(family), fraction=np.asarray(fraction), sign=np.asarray(sign),
                    target_amplitude=np.asarray(target), actual_amplitude=np.asarray(actual),
                    p=np.asarray(p_value), q=np.asarray(q_value),
                    baseline_npz=np.asarray(str(BASELINE_NPZ)), baseline_npz_sha256=np.asarray(BASELINE_NPZ_SHA256),
                    baseline_3dg_sha256=np.asarray(BASELINE_3DG_SHA256), seed=np.asarray(SEED),
                )
                probe_sha = sha256_file(path)
                coordinates_sha = hashlib.sha256(np.ascontiguousarray(candidate, dtype="<f8").tobytes()).hexdigest()
                probe_index.append({"probe": name, "path": str(path.relative_to(RUN_DIR)),
                                    "sha256": probe_sha, "coordinates_sha256": coordinates_sha,
                                    "family": family, "fraction_of_baseline_rg": float(fraction),
                                    "sign": float(sign), "bytes": int(path.stat().st_size)})

                rows.append({
                    "probe": name, "family": family,
                    "direction_construction": built["audit"][family]["construction"],
                    "fraction_of_baseline_rg": float(fraction), "sign": float(sign),
                    "target_amplitude": float(target), "actual_amplitude": float(actual),
                    "actual_displacement_rms_abs": abs(float(actual)),
                    "amplitude_shrink_factor": float(actual) / float(target),
                    "amplitude_shrunk": bool(halvings > 0),
                    "amplitude_halvings": int(halvings),
                    "peak_radius": float(peak), "baseline_whole_cell_rg": float(baseline_rg),
                    "displacement_rms_raw": raw_rms,
                    "displacement_rms_after_global_removal": post_rms,
                    "amplitude_rms_consistency_abs_error": abs(post_rms - abs(float(actual))),
                    "global_mean_displacement_norm": float(np.linalg.norm(mean_displacement)),
                    "global_mean_displacement_removed_norm": removed_mean_norm,
                    "kl_q_base_vs_q_probe": kl,
                    "Noff_over_Nraw_times_kl": (float(baseline_audit["n_off"]) / float(baseline_audit["Nraw"])) * kl,
                    "observed_log_rate_delta_raw": observed_log_delta,
                    "normalizer_delta_raw": normalizer_delta,
                    "count_delta_raw": count_delta_raw,
                    "data_count_nll_delta": normalized_delta,
                    "count_A_delta": count_a_delta,
                    "count_delta_identity_abs_error": identity_error,
                    "fullJ_delta_A": float(audit["fullJ_A"] - baseline_audit["fullJ_A"]),
                    "fullJ_delta_B": float(audit["fullJ_B"] - baseline_audit["fullJ_B"]),
                    "fullJ_delta_C": float(audit["fullJ_C"] - baseline_audit["fullJ_C"]),
                    "count_delta_B": float(audit["count_B"] - baseline_audit["count_B"]),
                    "count_delta_C": float(audit["count_C"] - baseline_audit["count_C"]),
                    "regularization_delta_weighted": float(audit["regularization_total"]
                                                           - baseline_audit["regularization_total"]),
                    "bond_delta": float(audit["regularizers_weighted"]["bond"]
                                        - baseline_audit["regularizers_weighted"]["bond"]),
                    "repulsion_delta": float(audit["regularizers_weighted"]["repulsion"]
                                             - baseline_audit["regularizers_weighted"]["repulsion"]),
                    "bend_delta_weighted": float(audit["regularizers_weighted"]["bend"]
                                                 - baseline_audit["regularizers_weighted"]["bend"]),
                    "bend_delta_raw": float(audit["regularizers_raw"]["bend"]
                                            - baseline_audit["regularizers_raw"]["bend"]),
                    "p_prior_delta": float(audit["regularizers_weighted"]["p_prior"]
                                           - baseline_audit["regularizers_weighted"]["p_prior"]),
                    "intra_distance_max_abs_change": float(intra_abs.max()),
                    "intra_distance_max_rel_change": float((intra_abs / np.maximum(baseline_intra, 1e-300)).max()),
                    "intra_distance_n_pairs": int(baseline_intra.size),
                    "cis_rate_max_rel_change": float(cis_rel.max()),
                    "cis_rate_mean_rel_change": float(cis_rel.mean()),
                    "cis_rate_n_pairs": int(cis_mask.sum()),
                    "inter_rate_max_rel_change": float(inter_rel.max()),
                    "inter_rate_mean_rel_change": float(inter_rel.mean()),
                    "chr_centre_190_n_pairs": merged_stats["n_pairs"],
                    "chr_centre_190_mean_abs_change": merged_stats["mean_abs_change"],
                    "chr_centre_190_max_abs_change": merged_stats["max_abs_change"],
                    "chr_centre_190_rms_change": merged_stats["rms_change"],
                    "chr_centre_190_mean_relative_change": merged_stats["mean_relative_change"],
                    "chr_centre_190_signed_mean_change": merged_stats["signed_mean_change"],
                    "chr_centre_displacement_norm_max": float(centre_disp.max()),
                    "chr_centre_displacement_norm_mean": float(centre_disp.mean()),
                    "copy_centre_760_n_pairs": copy_cross["n_pairs"],
                    "copy_centre_760_mean_abs_change": copy_cross["mean_abs_change"],
                    "copy_centre_760_max_abs_change": copy_cross["max_abs_change"],
                    "copy_centre_760_rms_change": copy_cross["rms_change"],
                    "copy_centre_760_mean_relative_change": copy_cross["mean_relative_change"],
                    "copy_centre_all780_n_pairs": copy_all["n_pairs"],
                    "copy_centre_all780_mean_abs_change": copy_all["mean_abs_change"],
                    "copy_centre_all780_max_abs_change": copy_all["max_abs_change"],
                    "copy_centre_all780_rms_change": copy_all["rms_change"],
                    "copy_centre_all780_mean_relative_change": copy_all["mean_relative_change"],
                    "copy_centre_same_chr_homolog_20_n_pairs": copy_same["n_pairs"],
                    "copy_centre_same_chr_homolog_20_mean_abs_change": copy_same["mean_abs_change"],
                    "copy_centre_same_chr_homolog_20_max_abs_change": copy_same["max_abs_change"],
                    "copy_centre_same_chr_homolog_20_rms_change": copy_same["rms_change"],
                    "copy_centre_same_chr_homolog_20_mean_relative_change": copy_same["mean_relative_change"],
                    "per_chr_rigid_residual_rms_max": rigid["residual_rms_max"],
                    "per_chr_rigid_residual_rms_median": rigid["residual_rms_median"],
                    "per_chr_rigid_rotation_det_min": rigid["rotation_det_min"],
                    "per_chr_rigid_rotation_det_max": rigid["rotation_det_max"],
                    "per_chr_rotation_angle_deg_max": rigid["rotation_angle_deg_max"],
                    "per_chr_rotation_angle_deg_median": rigid["rotation_angle_deg_median"],
                    "realised_rotation_angle_deg_expected": expected_angle,
                    "realised_rotation_angle_deg_abs_error_max":
                        float(np.max(np.abs(np.asarray(rigid["per_chromosome_angle_deg"]) - expected_angle))),
                    "rotation_angle_scale_alpha": None if rotation_meta is None
                    else rotation_meta["rotation_angle_scale_alpha"],
                    "rotation_alpha_iterations": None if rotation_meta is None
                    else rotation_meta["rotation_alpha_iterations"],
                    "rotation_rms_residual_rel": None if rotation_meta is None
                    else rotation_meta["rotation_rms_residual_rel"],
                    "rms_deformation_after_proper_rigid": global_fit["rms"],
                    "global_kabsch_rotation_det": global_fit["det"],
                    "global_kabsch_rotation_angle_deg": global_fit["angle_deg"],
                    "probe_npz": str(path.relative_to(RUN_DIR)), "probe_npz_sha256": probe_sha,
                    "coordinates_sha256": coordinates_sha,
                    # 下划线开头的键只进 JSON（保留 alpha 迭代细节），不写进 TSV
                    "_rotation_alpha_history": None if rotation_meta is None
                    else rotation_meta["rotation_alpha_history"],
                })

    def col(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    rotation_rows = [row for row in rows if row["family"] == "rigid_block_rotation"]

    geometry = {
        "translation_removed": {
            "definition": "the applied displacement field of every probe has zero global mean displacement "
                          "(the single global mean displacement of the construction field was removed before the "
                          "RMS normalisation); the probe coordinates were NOT rotated back",
            "mean_displacement_max_norm": max(col("global_mean_displacement_norm")),
            "mean_displacement_removed_max_norm": max(col("global_mean_displacement_removed_norm")),
            "tolerance": MEAN_SHIFT_TOL,
            "passed": bool(max(col("global_mean_displacement_norm")) <= MEAN_SHIFT_TOL),
            "scope_note": "only the global TRANSLATION component is removed from the probe coordinates; no global "
                          "rotation component is removed from them",
        },
        "rotation_accounted_in_aligned_readout": {
            "definition": "the global rotation component is only quantified, not removed from the probe "
                          "coordinates, by a proper Kabsch alignment of the probe onto the baseline "
                          "(field rms_deformation_after_proper_rigid)",
            "rms_deformation_after_proper_rigid_min": min(col("rms_deformation_after_proper_rigid")),
            "rms_deformation_after_proper_rigid_max": max(col("rms_deformation_after_proper_rigid")),
            "global_kabsch_rotation_det_min": min(col("global_kabsch_rotation_det")),
            "global_kabsch_rotation_det_max": max(col("global_kabsch_rotation_det")),
            "passed_det_plus_one": bool(min(col("global_kabsch_rotation_det")) > 1.0 - 1e-12
                                        and max(col("global_kabsch_rotation_det")) < 1.0 + 1e-12),
            "is_pass_fail_check": False,
            "note": "reported readout only; it never redefines the frozen probe coordinates or amplitudes",
        },
        "intra_chromosome_four_copy_distances_preserved": {
            "definition": "all bead-pair distances inside each chromosome of the baseline (which contains, for every "
                          "intra-chromosomal locus pair, all four copy combinations) are unchanged",
            "n_pairs": int(rows[0]["intra_distance_n_pairs"]),
            "max_abs_change": max(col("intra_distance_max_abs_change")),
            "max_rel_change": max(col("intra_distance_max_rel_change")),
            "tolerance_abs": INTRA_TOL_ABS,
            "passed": bool(max(col("intra_distance_max_abs_change")) <= INTRA_TOL_ABS),
        },
        "cis_raw_rates_preserved": {
            "definition": "with e and p fixed, the raw marginal-G rate of every same-chromosome pair is unchanged",
            "n_pairs": int(rows[0]["cis_rate_n_pairs"]),
            "max_rel_change": max(col("cis_rate_max_rel_change")),
            "tolerance_rel": RATE_TOL_REL,
            "passed": bool(max(col("cis_rate_max_rel_change")) <= RATE_TOL_REL),
            "inter_rates_may_change": {"max_rel_change": max(col("inter_rate_max_rel_change")),
                                       "mean_rel_change": max(col("inter_rate_mean_rel_change"))},
        },
        "per_chromosome_rigid_motion_exact": {
            "definition": "each chromosome (both copies together) is moved by an exact rigid motion: proper Kabsch "
                          "of the probe block onto the baseline block leaves no residual and det=+1",
            "residual_rms_max": max(col("per_chr_rigid_residual_rms_max")),
            "rotation_det_min": min(col("per_chr_rigid_rotation_det_min")),
            "rotation_det_max": max(col("per_chr_rigid_rotation_det_max")),
            "tolerance_abs": RIGID_TOL_ABS,
            "passed": bool(max(col("per_chr_rigid_residual_rms_max")) <= RIGID_TOL_ABS
                           and min(col("per_chr_rigid_rotation_det_min")) > 1.0 - 1e-12),
        },
        "rotation_family_angle_calibration": {
            "definition": "for the rotation family the realised per-chromosome SO(3) angle (proper Kabsch fit) "
                          "equals the RMS-calibrated angle |alpha|*base_angle, and the calibrated field RMS hits the "
                          "target amplitude",
            "max_abs_angle_error_deg": max(float(row["realised_rotation_angle_deg_abs_error_max"])
                                          for row in rotation_rows),
            "max_angle_deg": max(float(row["realised_rotation_angle_deg_expected"]) for row in rotation_rows),
            "max_rms_residual_rel": max(float(row["rotation_rms_residual_rel"]) for row in rotation_rows),
            "max_alpha_iterations": int(max(int(row["rotation_alpha_iterations"]) for row in rotation_rows)),
            "tolerance_deg": 1e-6,
            "passed": bool(max(float(row["realised_rotation_angle_deg_abs_error_max"]) for row in rotation_rows) <= 1e-6
                           and max(float(row["rotation_rms_residual_rel"]) for row in rotation_rows) <= 1e-12),
            "forbidden_constructions_absent": {
                "rotated_coordinates_R_times_offset_used_as_direction": False,
                "linear_scaling_of_a_fixed_rotation_displacement_field": False,
                "note": "the exact SO(3) field (R-I)(x-c) is rebuilt at the calibrated angle for every probe",
            },
        },
        "inside_unit_ball": {
            "peak_radius_max": max(col("peak_radius")),
            "amplitude_shrunk_probes": [row["probe"] for row in rows if row["amplitude_shrunk"]],
            "passed": bool(max(col("peak_radius")) < 1.0),
        },
        "count_delta_identity": {
            "definition": "data_count_nll_delta = [sum_C C*log(r_base/r_probe) + Noff*log(Z_probe/Z_base)] / Nraw "
                          "must equal count_A(probe) - count_A(base), both normalised by Nraw",
            "max_abs_error": max(col("count_delta_identity_abs_error")),
            "tolerance": COUNT_IDENTITY_TOL,
            "passed": bool(max(col("count_delta_identity_abs_error")) <= COUNT_IDENTITY_TOL),
        },
        "amplitude_definition_realised": {
            "definition": "the realised whole-cell displacement RMS after removal of the single global mean "
                          "displacement equals |actual_amplitude|",
            "max_abs_error": max(col("amplitude_rms_consistency_abs_error")),
            "passed": bool(max(col("amplitude_rms_consistency_abs_error")) <= 1e-12),
        },
        "baseline_parity_with_frozen_rescore": {**parity,
                                               "denominators": denominators,
                                               "baseline_rates_sum_vs_Zsum_rel_error": rates_sum_rel_error,
                                               "baseline_rates_all_positive": True},
        "baseline_value_gradient_consistency": gradient_check,
        "p_and_q": {"p_from_npz": p_value, "q_from_npz": q_value, "q_from_p_npz": q_from_p,
                    "abs_diff_q": abs(q_value - q_from_p), "p_from_q_npz": float(p_from_q),
                    "abs_diff_p": abs(p_value - float(p_from_q)), "npz_keys": npz_keys,
                    "passed": bool(abs(q_value - q_from_p) <= 1e-12 and abs(p_value - float(p_from_q)) <= 1e-15)},
        "centre_pair_denominators": {
            "definition": "fixed denominators for the centre readouts: 190 merged chromosome-centre pairs; the "
                          "primary copy-centre readout is the CROSS-CHROMOSOME set 190 x 4 = 760 copy-centre pairs; "
                          "the 20 same-chromosome homolog copy-centre pairs are reported separately and never enter "
                          "a 760 field",
            "merged_chromosome_centre_pairs": 190,
            "copy_centre_cross_chromosome_pairs": 760,
            "copy_centre_all_pairs_including_same_chromosome_homologs": 780,
            "copy_centre_same_chromosome_homolog_pairs": 20,
            "max_copy_centre_cross_chromosome_mean_abs_change": max(col("copy_centre_760_mean_abs_change")),
            "max_copy_centre_all780_mean_abs_change": max(col("copy_centre_all780_mean_abs_change")),
            "max_copy_centre_same_chr_homolog_mean_abs_change": max(
                col("copy_centre_same_chr_homolog_20_mean_abs_change")),
            "passed": bool(all(int(row["chr_centre_190_n_pairs"]) == 190
                               and int(row["copy_centre_760_n_pairs"]) == 760
                               and int(row["copy_centre_all780_n_pairs"]) == 780
                               and int(row["copy_centre_same_chr_homolog_20_n_pairs"]) == 20 for row in rows)),
        },
        "violates_rigidity_forbidden_patterns": {
            "per_bead_clipping_used": False, "linear_blend_of_rotated_coordinates_used": False,
            "amplitude_scaling": "a single scalar times one fixed unit direction field per probe",
        },
    }
    failed = [key for key, value in geometry.items()
              if isinstance(value, dict) and "passed" in value and value["passed"] is False]
    descent = [{"probe": row["probe"], "fullJ_delta_A": row["fullJ_delta_A"], "count_A_delta": row["count_A_delta"],
                "kl_q_base_vs_q_probe": row["kl_q_base_vs_q_probe"]} for row in rows if row["fullJ_delta_A"] < 0.0]
    summary = {
        "n_probes": len(rows),
        "baseline_rg": baseline_rg,
        "baseline_p": p_value,
        "fullJ_delta_A_range": [min(col("fullJ_delta_A")), max(col("fullJ_delta_A"))],
        "count_A_delta_range": [min(col("count_A_delta")), max(col("count_A_delta"))],
        "kl_range": [min(col("kl_q_base_vs_q_probe")), max(col("kl_q_base_vs_q_probe"))],
        "probes_with_fullJ_delta_A_negative": descent,
        "descent_direction_found": bool(len(descent) > 0),
        "sign_symmetry_check": "each family/amplitude has a + and a - direction; a symmetric pair means the "
                               "baseline is near a local stationary point along that direction",
        "next_round_minimal_suggestion": (
            "no automatic sweep and no fit were appended in this round. All eight probes raise full-J A, so this "
            "evidence alone justifies no line-search start; the probe set covers only 8 fixed directions at two "
            "amplitudes and can neither prove nor disprove that the chromosome-centre placement is identifiable or "
            "optimal. If a future round wants the cheapest next test on the centre block, the minimal one is a "
            "single 1-D line search along the best-scoring centre_translation direction with the same fixed e/p, "
            "reported with the same count/KL/full-J readouts and no reference."),
    }

    report = {
        "schema": "p9016-round050-p2-probe-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "started_at_utc": started_at,
        "seed": SEED,
        "baseline": {"npz": str(BASELINE_NPZ.relative_to(ROOT)), "npz_sha256": baseline_npz_sha,
                     "3dg": str(BASELINE_3DG.relative_to(ROOT)), "3dg_sha256": baseline_3dg_sha,
                     "whole_cell_rg": baseline_rg, "p": p_value, "q": q_value,
                     "label": "046 work baseline real-extension-G-full-J",
                     "frozen_rescore": {"count_A": FROZEN_COUNT_A, "fullJ_A": FROZEN_FULLJ_A}},
        "families": list(FAMILIES), "amplitudes_of_baseline_rg": list(FRACTIONS), "signs": list(SIGNS),
        "amplitude_definition": "whole-cell displacement RMS of the probe displacement field after the single "
                                "global mean displacement has been removed; the same rule defines the translation "
                                "and rotation amplitude, so the two families are directly comparable. The rotation "
                                "family is a real SO(3) transform: for each probe a scalar angle-scale alpha is "
                                "solved so that the exact rigid displacement field (R-I)(x-c) has the target RMS; a "
                                "fixed rotation field is never linearly scaled (that would break orthogonality) and "
                                "no per-bead clipping is used.",
        "direction_rule": "fixed in advance from seed 461001; the reference is never consulted",
        "direction_audit": built["audit"],
        "e_and_p": "exposure e and p are fixed at the baseline values; nothing is fitted and no sweep is appended",
        "count_delta_definition": "data_count_nll_delta = [sum_C C*log(r_base/r_probe) + Noff*log(Z_probe/Z_base)] "
                                  "/ Nraw, asserted equal to count_A(probe)-count_A(base)",
        "reference_free_readouts": "KL(q_base||q_probe) over the FULL off-diagonal grid (all 3,496,690 pairs incl. "
                                   "zero-count pairs), Noff/Nraw*KL, observed-log and normalizer terms separately, "
                                   "count_A delta, full-J A/B/C deltas, weighted bond/repulsion/bend/p_prior deltas",
        "geometry": geometry, "geometry_failed_checks": failed, "summary": summary,
        "probes": rows,
        "scope": "8 probes are the frozen base design. A few directions can neither prove nor disprove global "
                 "identifiability; KL>0 is not presupposed, no magnitude is required to match any other probe, and "
                 "no reference-based evaluation of these probes exists in this round.",
        "isolation": {
            "process_scope": "this P2 process opened only inputs/P9016.snpfree.pairs.gz aggregate, the 046 baseline "
                             "NPZ and the 046 baseline 3DG (for its frozen hash), plus the read-only 049 training-side "
                             "analysis_core code; it never opened data/P9016.1m.3dg.gz, phase columns or any "
                             "reference-derived structure or mask",
            "session_scope_caveat": "process-level isolation only: other processes of round 050 (P0/P1 evaluation) "
                                    "do read the reference in separate processes; this is not a claim that the whole "
                                    "session never read the reference",
            "forbidden_modules_loaded": sorted(name for name in sys.modules if name in FORBIDDEN_MODULES),
        },
        "wall_seconds": time.time() - t0,
    }
    report["isolation"]["clean"] = not report["isolation"]["forbidden_modules_loaded"]
    write_tsv(p2 / "results" / "p2_probes.tsv", rows)
    write_json(p2 / "results" / "p2_probes.json", report)
    write_json(p2 / "results" / "validation.json", {
        "schema": "p9016-round050-p2-validation-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_parity_with_frozen_rescore": geometry["baseline_parity_with_frozen_rescore"],
        "baseline_value_gradient_consistency": geometry["baseline_value_gradient_consistency"],
        "p_and_q": geometry["p_and_q"],
        "translation_removed": geometry["translation_removed"],
        "rotation_accounted_in_aligned_readout": geometry["rotation_accounted_in_aligned_readout"],
        "intra_chromosome_four_copy_distances_preserved": geometry["intra_chromosome_four_copy_distances_preserved"],
        "cis_raw_rates_preserved": geometry["cis_raw_rates_preserved"],
        "per_chromosome_rigid_motion_exact": geometry["per_chromosome_rigid_motion_exact"],
        "inside_unit_ball": geometry["inside_unit_ball"],
        "count_delta_identity": geometry["count_delta_identity"],
        "amplitude_definition_realised": geometry["amplitude_definition_realised"],
        "full_grid_denominators": geometry["baseline_parity_with_frozen_rescore"]["denominators"],
        "failed_checks": failed,
        "all_checks_passed": not failed,
        "reference_opened": False, "phase_opened": False,
    })
    gate = {
        "schema": "p9016-round050-p2-probe-pre-reference-gate-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_3dg": {"path": str(BASELINE_3DG.relative_to(ROOT)), "sha256": baseline_3dg_sha},
        "baseline_npz": {"path": str(BASELINE_NPZ.relative_to(ROOT)), "sha256": baseline_npz_sha},
        "probes": probe_index, "n_probes": len(probe_index),
        "reference_opened": False, "phase_opened": False,
        "note": "all eight probe coordinate payloads were written and hashed (file sha256 and coordinate-byte "
                "sha256) before this gate file was written; every P2 quantity reported in p2/results uses no "
                "reference at all. These 8 probes carry no reference-based evaluation and none is claimed.",
    }
    write_json(p2 / "pre_reference_gate.json", gate)
    terminal = {
        "schema": "p9016-round050-p2-terminal-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "started_at_utc": started_at,
        "terminal_state": "completed_no_fit_diagnostic",
        "exit_code": 0,
        "convergence": {"applicable": False,
                        "reason": "P2 fits nothing: e and p are fixed at the baseline values and the eight frozen "
                                  "probes are evaluated directly, so converged/not_converged semantics do not apply"},
        "n_probes": len(rows), "all_checks_passed": not failed, "failed_checks": failed,
        "baseline_parity_passed": parity["passed"],
        "gate_file": "p2/pre_reference_gate.json",
        "results": ["p2/results/p2_probes.tsv", "p2/results/p2_probes.json", "p2/results/validation.json"],
        "wall_seconds": time.time() - t0,
        "reference_opened": False, "phase_opened": False,
    }
    write_json(p2 / "terminal.json", terminal)
    print(json.dumps({"terminal_state": terminal["terminal_state"], "n_probes": len(rows),
                      "all_checks_passed": not failed, "failed_checks": failed,
                      "baseline_rg": baseline_rg, "baseline_p": p_value,
                      "fullJ_delta_A_range": summary["fullJ_delta_A_range"],
                      "wall_seconds": report["wall_seconds"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:  # 失败也要留真实终态证据
        p2_dir = Path(__file__).resolve().parent.parent / "p2"
        write_json(p2_dir / "terminal.json", {
            "schema": "p9016-round050-p2-terminal-v1",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "terminal_state": "failed_exception", "exit_code": 1,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "convergence": {"applicable": False, "reason": "run aborted before completion"},
            "reference_opened": False, "phase_opened": False})
        raise
