#!/usr/bin/env python
"""049 评价侧公共库：冻结常量、冻结纯 helper 局部副本、输入适配。

本模块不读 reference、不读 phase、不读 046 的 mask snapshot（那些只在
eval049_evaluate.py 通过 gate 之后才加载）。所有 helper 是从
046/evaluation_final/evaluator.py 复制的小型纯函数，保持数值口径一致；
不 import 046 的 evaluator，也不调用其旧入口或旧 writer。
"""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

RUN = Path(__file__).resolve().parents[2]
EVAL = RUN / "evaluation"
ROOT = RUN.parents[1]
SOURCE_RUN_045 = RUN.parent / "045-20260915T073310Z-shared-capture-round"
RUN_046 = RUN.parent / "046-UTC-real-cell-shared-capture"

AGGREGATE_1MB = SOURCE_RUN_045 / "inputs/real_1000000_aggregate.npz"
AGGREGATE_1MB_SHA256 = "80984d804f8ae0552f6bab34a77a03e778073ac87f3b3ec6f04f96de74137420"
MASK_SNAPSHOT = RUN_046 / "evaluation_final/results/frozen_legacy_mask_snapshot.npz"
# 实际文件哈希；与 046 创建时记录（mask_validation.json frozen_legacy_mask_worker.output_sha256）一致
MASK_SNAPSHOT_SHA256 = "9c551c6a4586a9221547f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9"
# 049/config.json 中记录的同一文件哈希（相差 1 个字符，属配置转录笔误，文件本身未变）
MASK_SNAPSHOT_SHA256_IN_049_CONFIG = "9c551c6a4586a9221541f55f7a47211fa6a57a77e1a3b5771667cac271ef28d9"
MASK_VALIDATION_046 = RUN_046 / "evaluation_final/results/mask_validation.json"
BASELINE_NPZ = RUN_046 / "coords/real-extension-G-full-J/1Mb.npz"
BASELINE_NPZ_SHA256 = "116e906e790493afa956445c527b38da11b2fdcaed482fd225f578f188df699d"
BASELINE_3DG = RUN_046 / "coords/real-extension-G-full-J/1Mb.3dg"
BASELINE_3DG_SHA256 = "4301d4df6e89c1417690599d6687a9e83b18fa37596de5d6c11a911d867d69ea"
BASELINE_COORD_ARRAY_SHA256 = "855e3f8ab4a08c31e5f13fe852fc232b450792381ef2665fcd0e7f1e9887a6dc"
BASELINE_NULL_DIR_046 = RUN_046 / "evaluation_final/nulls"
BASELINE_NULL_PREFIX_046 = "real-extension-G-full-J__"
BASELINE_GATE_046 = RUN_046 / "evaluation_final/results/pre_reference_hash_gate.json"
BASELINE_R2_PER_CHR_046 = RUN_046 / "evaluation_final/results/r2_per_chromosome.tsv"
BASELINE_R2_SUMMARY_046 = RUN_046 / "evaluation_final/results/r2_summary.tsv"
BOOTSTRAP_MATRIX_046 = RUN_046 / "evaluation_final/bootstrap_indices_seed450301_10000x20.npy"
INITIAL_NPZ = {
    "consensus": SOURCE_RUN_045 / "coords/initial/real_consensus_1Mb.npz",
    "random": SOURCE_RUN_045 / "coords/initial/real_random_1Mb.npz",
}
INITIAL_NPZ_SHA256 = {
    "consensus": "3b3e30925f33459804d95b3ba3636b24f85978eb62ba3e8143a6c763ec4b2c81",
    "random": "524e6df323899a0549e9e07a6d4a202c09c2d2778a6a4d283dcca11a2c38cc7d",
}
REFERENCE_PATH = ROOT / "data/P9016.1m.3dg.gz"
REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"

LOSES = ("A", "B", "C")
SOLVERS = ("raw", "ms")
SOURCES = ("consensus", "random")
EXPECTED_FIT_IDS = tuple("%s-%s-%s" % (l, s, v) for l in LOSES for s in SOLVERS for v in SOURCES)

BIN_SIZE_BP = 1_000_000
MASK_OFFSET_BP = 3_000_000
N_LOCI = 2645
N_CHROMOSOMES = 20
MIN_COMMON_PAIRS = 20
GEOMETRY_TIE_TOL = 1e-12
SELECTION_TIE_TOL = 1e-9
BOOTSTRAP_SEED = 450301
BOOTSTRAP_DRAWS = 10_000
RANDOM_U_SEEDS = tuple(range(450500, 450516))
PERMUTATION_SEED = 461100
PERMUTATION_DRAWS = 9999
EXPECTED_MASK = {
    "n_total_non_diagonal_pairs": 176_201,
    "n_common_pairs": 157_529,
    "valid_bins": 2447,
    "inter_locus_pairs": 2_835_152,
    "inter_denominator": 11_340_608,
}
EXPECTED_AGGREGATE = {
    "n_loci": 2645,
    "n_pairs": 3_496_690,
    "cis_pairs": 184_016,
    "counts_cis": 696_680,
    "counts_inter": 568_434,
    "counts_offdiag": 1_265_114,
    "diag_counts": 438_774,
    "raw_records": 1_703_888,
}
R2_FIELDS = ("matched", "cross", "contrast", "copy_A_margin", "copy_B_margin",
             "matched_mat_margin", "matched_pat_margin", "min_margin")


# ---------------------------------------------------------------- 基础工具


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_array(value: np.ndarray) -> str:
    array = np.asarray(value, dtype="<f8", order="C")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), sort_keys=True, indent=2,
                               ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("JSON object required: %s" % path)
    return value


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


RUN_RELATIVE_PREFIXES = ("coords/", "results/", "stages/", "logs/", "gates/", "diagnostics/",
                         "source/", "plots/", "nulls/", "evaluation/")
ROOT_RELATIVE_PREFIXES = ("test_res/", "inputs/", "data/", "docs/", "pr/", "scripts/", "native/")


def resolve_path(value: Any, *, run: Path | None = None, root: Path | None = None) -> Path:
    """统一路径解析（manifest/selection/endpoint/null 共用，避免各处随意拼接）。

    - 绝对路径：原样返回；
    - 以 run 目录名（如 ``049-.../coords/...``）或 ``test_res/...`` 开头：相对仓库根 ``lib.ROOT``；
    - 以 ``coords/``、``results/``、``stages/`` 等 run 内目录开头：相对 run 目录 ``lib.RUN``；
    - 其它相对路径：默认相对 run 目录。
    """
    run = Path(run) if run is not None else RUN
    root = Path(root) if root is not None else ROOT
    text = str(value).strip()
    if not text:
        raise RuntimeError("empty path value")
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    if text.startswith("./"):
        text = text[2:]
    if text.startswith(ROOT_RELATIVE_PREFIXES):
        return (root / text).resolve()
    first = text.split("/", 1)[0]
    if first == run.name:
        # 形如 049-xxx/coords/...：相对 run 的父目录（即 test_res）
        return (run.parent / text).resolve()
    if first.startswith("test_res"):
        return (root / text).resolve()
    if text.startswith(RUN_RELATIVE_PREFIXES):
        return (run / text).resolve()
    return (run / text).resolve()


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(RUN.resolve()))
    except ValueError:
        return str(Path(path).resolve())


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            cells = []
            for column in columns:
                value = row.get(column)
                if isinstance(value, float):
                    cells.append("%.17g" % value if math.isfinite(value) else "NA")
                elif value is None:
                    cells.append("NA")
                else:
                    cells.append(str(value))
            handle.write("\t".join(cells) + "\n")


# ------------------------------------------------- 从 046 复制的纯 helper 副本


def metric_min(metric_name: str, x: np.ndarray, y: np.ndarray, min_common_pairs: int) -> float:
    """带可配置最小样本数的 Pearson/Spearman；样本不足或任一常量则返回 NaN。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < int(min_common_pairs) or len(x) != len(y) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    if metric_name == "pearson":
        xc, yc = x - x.mean(), y - y.mean()
        denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
        return float(np.dot(xc, yc) / denom) if denom else float("nan")
    rx, ry = rankdata(x), rankdata(y)
    xc, yc = rx - rx.mean(), ry - ry.mean()
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    return float(np.dot(xc, yc) / denom) if denom else float("nan")


def metric(metric_name: str, x: np.ndarray, y: np.ndarray) -> float:
    """R2/inter 口径：沿用 046 的 MIN_COMMON_PAIRS=20 门槛。"""
    return metric_min(metric_name, x, y, MIN_COMMON_PAIRS)


def spearman_ranked(x: np.ndarray, reference_rank: np.ndarray) -> float:
    rx = rankdata(x)
    if np.ptp(rx) == 0.0 or np.ptp(reference_rank) == 0.0:
        return float("nan")
    xc, yc = rx - rx.mean(), reference_rank - reference_rank.mean()
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    return float(np.dot(xc, yc) / denom) if denom else float("nan")


def distance(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    delta = np.asarray(points[pair_i] - points[pair_j], dtype=np.float64)
    return np.sqrt(np.sum(delta * delta, axis=1))


def four_distances(coords: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    return np.stack((
        distance(coords[0], pair_i, pair_j),
        np.sqrt(np.sum((coords[0, pair_i] - coords[1, pair_j]) ** 2, axis=1)),
        np.sqrt(np.sum((coords[1, pair_i] - coords[0, pair_j]) ** 2, axis=1)),
        distance(coords[1], pair_i, pair_j),
    ), axis=0)


def derive_rho(rho: Mapping[str, float]) -> dict[str, Any]:
    required = ("A_mat", "A_pat", "B_mat", "B_pat")
    if not all(math.isfinite(float(rho[key])) for key in required):
        return {
            "rho": {key: (float(rho[key]) if math.isfinite(float(rho[key])) else None) for key in required},
            "direct": None, "swapped": None, "matched": None, "cross": None, "contrast": None,
            "orientation": "undefined", "geometry_tie": False,
            "copy_A_margin": None, "copy_B_margin": None, "matched_mat_margin": None,
            "matched_pat_margin": None, "min_margin": None,
            "derived_defined": False, "undefined_reason": "one_or_more_of_four_rho_nonfinite",
        }
    direct = (rho["A_mat"] + rho["B_pat"]) / 2.0
    swapped = (rho["A_pat"] + rho["B_mat"]) / 2.0
    base = {"rho": {key: float(rho[key]) for key in required}, "direct": float(direct), "swapped": float(swapped)}
    if abs(direct - swapped) <= GEOMETRY_TIE_TOL:
        return {**base, "matched": float((direct + swapped) / 2.0), "cross": float((direct + swapped) / 2.0),
                "contrast": 0.0, "orientation": "unresolved_tie", "geometry_tie": True,
                "copy_A_margin": None, "copy_B_margin": None, "matched_mat_margin": None,
                "matched_pat_margin": None, "min_margin": None, "derived_defined": True, "undefined_reason": None}
    if direct > swapped:
        copy_a = rho["A_mat"] - rho["A_pat"]
        copy_b = rho["B_pat"] - rho["B_mat"]
        return {**base, "matched": float(direct), "cross": float(swapped), "contrast": float(direct - swapped),
                "orientation": "direct", "geometry_tie": False,
                "copy_A_margin": float(copy_a), "copy_B_margin": float(copy_b),
                "matched_mat_margin": float(copy_a), "matched_pat_margin": float(copy_b),
                "min_margin": float(min(copy_a, copy_b)), "derived_defined": True, "undefined_reason": None}
    copy_a = rho["A_pat"] - rho["A_mat"]
    copy_b = rho["B_mat"] - rho["B_pat"]
    return {**base, "matched": float(swapped), "cross": float(direct), "contrast": float(swapped - direct),
            "orientation": "swapped", "geometry_tie": False,
            "copy_A_margin": float(copy_a), "copy_B_margin": float(copy_b),
            "matched_mat_margin": float(copy_b), "matched_pat_margin": float(copy_a),
            "min_margin": float(min(copy_a, copy_b)), "derived_defined": True, "undefined_reason": None}


def mask_global_indices(offsets: np.ndarray, mask: Mapping[str, Any]) -> np.ndarray:
    """chr_offset + positions // 1Mb：mask local index 不是 0-origin。"""
    chromosome_index = int(mask["chromosome_index"])
    positions = np.asarray(mask["positions"], dtype=np.int64)
    return int(offsets[chromosome_index]) + positions // BIN_SIZE_BP


def r2_result(coords: np.ndarray, reference: np.ndarray, chromosome_names: Sequence[str],
              offsets: np.ndarray, masks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    per_chromosome: list[dict[str, Any]] = []
    for ci, name in enumerate(chromosome_names):
        mask = masks[str(name)]
        global_indices = mask_global_indices(offsets, mask)
        candidate = coords[:, global_indices]
        ref = reference[:, global_indices]
        pair_i = np.asarray(mask["pair_i"], dtype=np.int64)
        pair_j = np.asarray(mask["pair_j"], dtype=np.int64)
        common = np.asarray(mask["common"], dtype=bool)
        candidate_a = distance(candidate[0], pair_i[common], pair_j[common])
        candidate_b = distance(candidate[1], pair_i[common], pair_j[common])
        reference_mat = distance(ref[0], pair_i[common], pair_j[common])
        reference_pat = distance(ref[1], pair_i[common], pair_j[common])
        rho_p = {"A_mat": metric("pearson", candidate_a, reference_mat),
                 "A_pat": metric("pearson", candidate_a, reference_pat),
                 "B_mat": metric("pearson", candidate_b, reference_mat),
                 "B_pat": metric("pearson", candidate_b, reference_pat)}
        rho_s = {"A_mat": metric("spearman", candidate_a, reference_mat),
                 "A_pat": metric("spearman", candidate_a, reference_pat),
                 "B_mat": metric("spearman", candidate_b, reference_mat),
                 "B_pat": metric("spearman", candidate_b, reference_pat)}
        pearson_row = derive_rho(rho_p)
        spearman_row = derive_rho(rho_s)
        per_chromosome.append({
            "chromosome": str(name), "chromosome_index": ci, "n_pairs": int(mask["n_common_pairs"]),
            "n_total_non_diagonal_pairs": int(mask["n_total_non_diagonal_pairs"]),
            "metrics": {"pearson": pearson_row, "spearman": spearman_row},
            "status": "ok" if pearson_row["derived_defined"] and spearman_row["derived_defined"] else "undefined",
        })
    macro: dict[str, dict[str, float | None]] = {}
    defined: dict[str, dict[str, int]] = {}
    undefined: dict[str, dict[str, list[str]]] = {}
    full20: dict[str, dict[str, float | None]] = {}
    full20_defined: dict[str, dict[str, bool]] = {}
    for metric_name in ("pearson", "spearman"):
        macro[metric_name], defined[metric_name], undefined[metric_name] = {}, {}, {}
        full20[metric_name], full20_defined[metric_name] = {}, {}
        for field in R2_FIELDS + ("direct", "swapped"):
            values, missing = [], []
            for row in per_chromosome:
                value = row["metrics"][metric_name].get(field)
                if value is not None and math.isfinite(float(value)):
                    values.append(float(value))
                else:
                    missing.append(str(row["chromosome"]))
            # 主表/paired CI 口径：20 chr 全部定义才给值，任一缺失即 NULL
            complete = len(values) == len(chromosome_names) and not missing
            full20[metric_name][field] = float(np.mean(values)) if complete else None
            full20_defined[metric_name][field] = bool(complete)
            macro[metric_name][field] = float(np.mean(values)) if values else None
            defined[metric_name][field] = len(values)
            undefined[metric_name][field] = missing
    return {
        "per_chromosome": per_chromosome,
        "macro_full_20_all_chromosomes_required": full20,
        "macro_full_20_defined": full20_defined,
        "macro_equal_chromosome_weight_defined_only": macro,
        "macro_policy": ("main table and paired CI use macro_full_20_all_chromosomes_required (NULL if any chromosome "
                         "is undefined); macro_equal_chromosome_weight_defined_only is descriptive only and must not "
                         "replace the fixed-20 estimate"),
        "defined_chromosome_counts": defined,
        "undefined_chromosomes_by_field": undefined,
        "denominator_chromosomes": len(chromosome_names),
        "metric_definition": "frozen 046 old21 common finite pairs; mask local index never used as 0-origin",
        "primary_margins": "matched_mat_margin/matched_pat_margin/min_margin; candidate A/B are not fixed parental labels",
    }


def make_u_zero(coords: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    z = (coords[0] + coords[1]) / 2.0
    raw = np.stack((z, z), axis=0)
    pre = float(np.linalg.norm(raw.reshape(-1, 3), axis=1).max())
    scale = 1.0 if pre < 1.0 else (1.0 - 1e-6) / pre
    result = raw * scale
    assert_inside_unit_ball(result)
    return result, {"global_scale": scale, "pre_scale_radius": pre,
                    "post_scale_radius": float(np.linalg.norm(result.reshape(-1, 3), axis=1).max()),
                    "permutation": "u_zero", "z_fixed": True, "rescale_scope": "whole_cell_only_if_outside_sphere"}


def make_random_u(coords: np.ndarray, chromosome_slices: Sequence[slice], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    z = (coords[0] + coords[1]) / 2.0
    u = (coords[0] - coords[1]) / 2.0
    permuted = np.zeros_like(u)
    rng = np.random.default_rng(seed)
    for slc in chromosome_slices:
        permuted[slc] = u[slc][rng.permutation(slc.stop - slc.start)]
    raw = np.stack((z + permuted, z - permuted), axis=0)
    pre = float(np.linalg.norm(raw.reshape(-1, 3), axis=1).max())
    scale = 1.0 if pre < 1.0 else (1.0 - 1e-6) / pre
    result = raw * scale
    assert_inside_unit_ball(result)
    return result, {"global_scale": scale, "pre_scale_radius": pre,
                    "post_scale_radius": float(np.linalg.norm(result.reshape(-1, 3), axis=1).max()),
                    "permutation": "within_chromosome_u", "z_fixed": True,
                    "rescale_scope": "whole_cell_only_if_outside_sphere"}


def assert_inside_unit_ball(x: np.ndarray) -> None:
    norms = np.sqrt(np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=-1))
    if not np.all(np.isfinite(norms)) or np.any(norms >= 1.0):
        raise AssertionError("all physical beads must lie strictly inside the unit ball")


# ------------------------------------------------------------------ 输入加载


class Aggregate:
    """045 1Mb aggregate 的只读最小视图（无 phase、无 reference）。"""

    def __init__(self, path: Path = AGGREGATE_1MB) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as payload:
            self.chromosome_names = tuple(str(x) for x in payload["chromosome_names"].tolist())
            self.chromosome_lengths = np.asarray(payload["chromosome_lengths"], dtype=np.int64)
            self.bin_size = int(np.asarray(payload["bin_size"]).item())
            self.n_bins = np.asarray(payload["n_bins"], dtype=np.int64)
            self.offsets = np.asarray(payload["offsets"], dtype=np.int64)
            self.locus_bin = np.asarray(payload["locus_bin"], dtype=np.int64)
            self.locus_chromosome = np.asarray(payload["locus_chromosome"], dtype=np.int32)
            self.pair_i = np.asarray(payload["pair_i"], dtype=np.int64)
            self.pair_j = np.asarray(payload["pair_j"], dtype=np.int64)
            self.cis_pair = np.asarray(payload["cis_pair"], dtype=bool)
            self.counts = np.asarray(payload["counts"], dtype=np.int64)
            self.diag_counts = np.asarray(payload["diag_counts"], dtype=np.int64)
            self.raw_records = int(np.asarray(payload["raw_records"]).item())
            self.exposure = np.asarray(payload["exposure"], dtype=np.float64)

    @property
    def n_loci(self) -> int:
        return int(self.locus_bin.size)

    @property
    def n_pairs(self) -> int:
        return int(self.pair_i.size)

    def chromosome_slice(self, index: int) -> slice:
        return slice(int(self.offsets[index]), int(self.offsets[index] + self.n_bins[index]))

    def chromosome_slices(self) -> list[slice]:
        return [self.chromosome_slice(i) for i in range(len(self.chromosome_names))]

    def audit(self) -> dict[str, Any]:
        result = {
            "n_loci": self.n_loci,
            "n_pairs": self.n_pairs,
            "cis_pairs": int(self.cis_pair.sum()),
            "counts_cis": int(self.counts[self.cis_pair].sum()),
            "counts_inter": int(self.counts[~self.cis_pair].sum()),
            "counts_offdiag": int(self.counts.sum()),
            "diag_counts": int(self.diag_counts.sum()),
            "raw_records": self.raw_records,
            "bin_size": self.bin_size,
            "n_chromosomes": len(self.chromosome_names),
        }
        for key, expected in EXPECTED_AGGREGATE.items():
            if int(result[key]) != int(expected):
                raise RuntimeError("frozen 1Mb aggregate mismatch %s=%s expected=%s" % (key, result[key], expected))
        return result


def load_coords_npz(path: Path, *, allow_missing_extra: bool = True) -> dict[str, Any]:
    """读候选/初始 npz：coordinates 必须存在且形状/有限性/球域合法。"""
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        keys = sorted(payload.files)
        if "coordinates" not in keys:
            raise RuntimeError("coordinate NPZ missing 'coordinates': %s (present keys %s)" % (path, keys))
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        extras = {key: np.asarray(payload[key]).copy() for key in keys if key != "coordinates"}
    expected_shape = (2, N_LOCI, 3)
    if coordinates.shape != expected_shape:
        raise RuntimeError("coordinates shape %s != %s: %s" % (coordinates.shape, expected_shape, path))
    if not np.isfinite(coordinates).all():
        raise RuntimeError("coordinates nonfinite: %s" % path)
    assert_inside_unit_ball(coordinates)
    record = {"path": path, "keys": keys, "coordinates": coordinates,
              "coordinate_array_sha256": hash_array(coordinates), "extra_keys": sorted(extras)}
    for key in ("raw_y", "theta", "p", "q", "p_init"):
        if key in extras:
            record[key] = extras[key]
    return record


def load_3dg(path: Path) -> dict[str, dict[int, np.ndarray]]:
    tracks: dict[str, dict[int, np.ndarray]] = {}
    opener = __import__("gzip").open if Path(path).suffix == ".gz" else open
    with opener(Path(path), "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                track = str(fields[0])
                position = int(fields[1])
                point = np.asarray([float(item) for item in fields[2:5]], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("invalid 3DG row %d in %s" % (line_no, path)) from exc
            if not np.isfinite(point).all():
                point = np.full(3, np.nan, dtype=np.float64)
            target = tracks.setdefault(track, {})
            if position in target:
                raise RuntimeError("duplicate coordinate %s:%d in %s" % (track, position, path))
            target[position] = point
    return tracks


CANDIDATE_TRACK_SUFFIXES = ("a", "b")
REFERENCE_TRACK_SUFFIXES = ("mat", "pat")


def track_name(chromosome_index: int, chromosome_name: str, copy_index: int, track_mode: str) -> str:
    if track_mode == "candidate":
        return "c%02d%s" % (chromosome_index + 1, CANDIDATE_TRACK_SUFFIXES[copy_index])
    if track_mode == "reference":
        return "%s(%s)" % (chromosome_name, REFERENCE_TRACK_SUFFIXES[copy_index])
    raise ValueError("unknown track_mode %r" % track_mode)


def three_dg_to_array(tracks: Mapping[str, Mapping[int, np.ndarray]], data: "Aggregate", *,
                      track_mode: str = "candidate", audit: dict[str, Any] | None = None) -> np.ndarray:
    """把 3DG 文本还原成 (2, n_loci, 3)（缺失为 NaN）。

    track_mode="candidate"：``c01a``/``c01b``；
    track_mode="reference"：``chr1(mat)``/``chr1(pat)``（真实 reference 的命名）。
    """
    if track_mode not in ("candidate", "reference"):
        raise ValueError("unknown track_mode %r" % track_mode)
    coords = np.full((2, int(data.n_loci), 3), np.nan, dtype=np.float64)
    found_tracks: list[str] = []
    missing_tracks: list[str] = []
    per_chromosome: list[dict[str, Any]] = []
    for ci, name in enumerate(data.chromosome_names):
        slc = data.chromosome_slice(ci)
        positions = np.asarray(data.locus_bin[slc], dtype=np.int64) * int(data.bin_size)
        finite_here = []
        for copy in (0, 1):
            track = track_name(ci, str(name), copy, track_mode)
            rows = tracks.get(track, {})
            if track in tracks:
                found_tracks.append(track)
            else:
                missing_tracks.append(track)
            for local, position in enumerate(positions):
                point = rows.get(int(position))
                if point is not None:
                    coords[copy, slc.start + local] = point
            finite_here.append(int(np.count_nonzero(np.isfinite(coords[copy, slc.start:slc.stop]).all(axis=1))))
        per_chromosome.append({"chromosome": str(name), "positions": int(len(positions)),
                               "finite_copy_a_or_mat": finite_here[0], "finite_copy_b_or_pat": finite_here[1]})
    if audit is not None:
        audit.update({"track_mode": track_mode, "found_tracks": found_tracks, "missing_tracks": missing_tracks,
                      "found_track_count": len(found_tracks), "missing_track_count": len(missing_tracks),
                      "per_chromosome_finite": per_chromosome,
                      "nonfinite_loci": int((~np.isfinite(coords).all(axis=2)).sum()),
                      "total_loci": int(coords.shape[0] * coords.shape[1])})
    return coords


READBACK_ATOL = 1e-12


def readback_parity(npz_coords: np.ndarray, tracks: Mapping[str, Mapping[int, np.ndarray]],
                    data: Aggregate, *, atol: float = READBACK_ATOL) -> dict[str, Any]:
    """3DG 回读必须与 npz 物理坐标逐元素相当：exact 相等或 <=1e-12 容差，否则调用方必须报错。"""
    array = three_dg_to_array(tracks, data)
    finite = np.isfinite(array).all(axis=2)
    if not finite.all():
        raise RuntimeError("3DG readback has %d nonfinite loci" % int((~finite).sum()))
    diff = float(np.abs(array - npz_coords).max())
    exact = bool(np.array_equal(array, npz_coords))
    return {"max_abs_difference": diff, "exact_array_equal": exact, "atol": float(atol),
            "within_tolerance": bool(exact or diff <= float(atol)),
            "rows": int(array.shape[0] * array.shape[1])}


def parse_rho_cell(text: str) -> dict[str, float]:
    value = ast.literal_eval(text)
    if not isinstance(value, dict):
        raise RuntimeError("rho cell is not a dict: %r" % text)
    return {str(k): float(v) for k, v in value.items()}


def load_baseline_expected_from_046(chromosome_names: Sequence[str]) -> dict[str, Any]:
    """只读 046 既有 TSV 元数据，构造回归 expected（不解析 reference）。"""
    import csv
    per_chromosome: dict[str, dict[str, dict[str, float]]] = {str(name): {} for name in chromosome_names}
    with BASELINE_R2_PER_CHR_046.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["candidate_id"] != "real-extension-G-full-J":
                continue
            per_chromosome[str(row["chromosome"])][str(row["metric"])] = parse_rho_cell(row["rho"])
    with BASELINE_R2_SUMMARY_046.open("r", encoding="utf-8", newline="") as handle:
        summary = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    base = summary["real-extension-G-full-J"]
    return {
        "source": {"per_chromosome_tsv": str(BASELINE_R2_PER_CHR_046), "summary_tsv": str(BASELINE_R2_SUMMARY_046)},
        "per_chromosome_four_rho": per_chromosome,
        "inter_pearson": float(base["inter_pearson"]),
        "inter_spearman": float(base["inter_spearman"]),
        "pearson_matched": float(base["pearson_matched"]),
        "pearson_cross": float(base["pearson_cross"]),
        "spearman_matched": float(base["spearman_matched"]),
        "spearman_cross": float(base["spearman_cross"]),
    }
