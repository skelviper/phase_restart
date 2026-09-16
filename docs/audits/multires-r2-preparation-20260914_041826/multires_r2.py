"""用于已准备多分辨率 R2 release 的受 gate 控制的评价器与绘图数据构建器。

本模块有意独立于历史 020/029 runner，不包含拟合或 native 引擎调用。Preparation validation 只读取 JSON protocol/config/mask metadata。真实评价由显式 release gate 控制，只有所有 candidate 和 frozen-mask 坐标哈希通过后才读取 reference。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PREP_DIR = Path(__file__).resolve().parent
ROOT = PREP_DIR.parents[2]
CONFIG_PATH = PREP_DIR / "config.json"
PROTOCOL_PATH = PREP_DIR / "r2_protocol.json"
MASK_LOCK_PATH = PREP_DIR / "mask_lock.json"
MASK_MANIFEST_DEFAULT = ROOT / "test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/evaluation_manifest.json"

VARIANT_IDS = ("C0", "C1", "C2-map", "C2-free", "C3")
SOURCE_IDS = ("consensus_joint_base1103", "random_joint_base2207")
SOURCE_DISPLAY = {
    "consensus_joint_base1103": "consensus-derived initialization (014 base1103; 40-track two-copy)",
    "random_joint_base2207": "random-derived initialization (014 base2207; 40-track two-copy)",
}
ENDPOINT_IDS = tuple("%s-%s" % (variant, source) for variant in VARIANT_IDS for source in SOURCE_IDS)
METRICS = ("matched", "cross", "contrast", "minmargin")
RAW_RHOS = ("rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat")
FULL_COLUMNS = (
    "endpoint_id", "variant_id", "source_id", "source_display", "chromosome",
    "chromosome_index", "endpoint_status", "coordinate_status", "n_copies",
    "n_bins", "n_total_non_diagonal_pairs", "n_common_pairs_frozen",
    "metric_status", "reason", "rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat",
    "direct", "swapped", "matched", "cross", "contrast", "margin_mat",
    "margin_pat", "minmargin", "orientation", "geometry_tie", "both_positive",
    "margin_class", "rho_reasons_json",
)
DELTA_COLUMNS = (
    "source_id", "source_display", "variant_id", "endpoint_id", "chromosome",
    "c0_endpoint_id", "metric", "variant_minus_c0", "status", "reason",
)
ORIENTATION_TIE_TOL = 1e-12
SELECTION_TIE_TOL_PER_RECORD = 1e-9
MIN_COMMON_PAIRS = 20
REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
MASK_MANIFEST_SHA256 = "fa1b26c834c173604c21f954d494cece8e053dd970e7cb6111f561b349574acb"
LEGACY_ALLELE_R2_PATH = ROOT / "pr/allele_r2.py"
LEGACY_ALLELE_R2_SHA256 = "56dacfe9a3edc13db4804fa7d5ce7c40b9401a1d1dd62ddbaa66d0d6ad5b1672"
LEGACY_R2COMPARISON_PATH = ROOT / "pr/r2comparison.py"
LEGACY_R2COMPARISON_SHA256 = "2979118aed25ac571d8f476eec4f6ec7687649f5a3e0fc43618194a293b4ed6e"
PUBLISHED_R2_PATH = ROOT / "test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_per_chromosome.tsv"
PUBLISHED_R2_SHA256 = "5e578b5ef8876f57866f9dfa325a8738adab70ebc4913498c570cc2b07883ed1"
HISTORICAL_020_ID = "v1_original_random_joint"
HISTORICAL_CPU_ANCHOR_ID = "historical_cpu_anchor_020_c0"
GPU_SELECTION_SCHEMA = "gpu-multires-selection-v1"
GPU_TERMINAL_SCHEMA = "gpu-multires-terminal-evidence-v1"
POST_ACCEPTANCE_NUMERIC_FIELDS = (
    ("rho_A_mat", "rho_A_mat"), ("rho_A_pat", "rho_A_pat"),
    ("rho_B_mat", "rho_B_mat"), ("rho_B_pat", "rho_B_pat"),
    ("direct", "direct_original"), ("swapped", "cross_original"),
    ("matched", "matched"), ("cross", "cross"), ("contrast", "contrast"),
    ("margin_mat", "margin_mat"), ("margin_pat", "margin_pat"), ("minmargin", "minmargin"),
)

# expected mask counts 也复制到 mask_lock.json，使未来 evaluator 在已发布的 21-condition mask 变化时中止。
EXPECTED_MASK_COUNTS = (
    ("chr1", 193, 18528, 17578),
    ("chr2", 180, 16110, 14878),
    ("chr3", 158, 12403, 12090),
    ("chr4", 154, 11781, 10731),
    ("chr5", 149, 11026, 10296),
    ("chr6", 147, 10731, 10585),
    ("chr7", 143, 10153, 8128),
    ("chr8", 127, 8001, 7503),
    ("chr9", 122, 7381, 7021),
    ("chr10", 128, 8128, 7875),
    ("chr11", 120, 7140, 7021),
    ("chr12", 118, 6903, 5995),
    ("chr13", 118, 6903, 6328),
    ("chr14", 122, 7381, 5778),
    ("chr15", 102, 5151, 4753),
    ("chr16", 96, 4560, 4278),
    ("chr17", 92, 4186, 3916),
    ("chr18", 88, 3828, 3741),
    ("chr19", 59, 1711, 1653),
    ("chrX", 169, 14196, 7381),
)
CHROMOSOMES = (
    ("chr1", 195471971), ("chr2", 182113224), ("chr3", 160039680),
    ("chr4", 156508116), ("chr5", 151834684), ("chr6", 149736546),
    ("chr7", 145441459), ("chr8", 129401213), ("chr9", 124595110),
    ("chr10", 130694993), ("chr11", 122082543), ("chr12", 120129022),
    ("chr13", 120421639), ("chr14", 124902244), ("chr15", 104043685),
    ("chr16", 98207768), ("chr17", 94987271), ("chr18", 90702639),
    ("chr19", 61431566), ("chrX", 171031299),
)
EXPECTED_MASK_IDS = (
    "C0-bundle1", "C0-bundle2", "C0-bundle3", "C1-bundle1", "C1-bundle2", "C1-bundle3",
    "C2-map-bundle1", "C2-map-bundle2", "C2-map-bundle3", "C2-free-bundle1", "C2-free-bundle2", "C2-free-bundle3",
    "C3-bundle1", "C3-bundle2", "C3-bundle3", "softall_seed124101", "v1_original_random_joint",
    "v1_continuation", "fdg_proposal", "random014", "consensus014",
)


class PreparationError(RuntimeError):
    """Preparation 或 release 契约不允许评价时抛出。"""


def _reject_json_constant(token: str) -> Any:
    raise ValueError("non-finite JSON constant is forbidden: %s" % token)


def read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except FileNotFoundError as exc:
        raise PreparationError("missing JSON artifact: %s" % target) from exc
    except json.JSONDecodeError as exc:
        raise PreparationError("invalid JSON artifact: %s" % target) from exc
    if not isinstance(value, Mapping):
        raise PreparationError("JSON artifact must be an object: %s" % target)
    return dict(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True,
                                 ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PreparationError("%s must be a lowercase SHA256" % label)
    return value


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def resolve_path(value: str | Path, base: Path = PREP_DIR) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] in {"data", "test_res", "docs", "pr", "inputs"}:
        return (ROOT / path).resolve()
    return (base / path).resolve()


def chromosome_grid(config: Mapping[str, Any]) -> list[tuple[str, int]]:
    rows = config.get("grid", {}).get("chromosomes")
    if not isinstance(rows, list) or len(rows) != 20:
        raise PreparationError("config grid must contain 20 chromosomes")
    result = [(str(row["name"]), int(row["length_bp"])) for row in rows]
    if tuple(result) != CHROMOSOMES:
        raise PreparationError("config chromosome grid differs from frozen P9016 grid")
    return result


def grid_positions(length_bp: int, *, offset_bp: int = 3_000_000, bin_size_bp: int = 1_000_000) -> np.ndarray:
    if int(length_bp) <= int(offset_bp):
        return np.empty(0, dtype=np.int64)
    return np.arange(int(offset_bp), int(length_bp), int(bin_size_bp), dtype=np.int64)


def default_tracks(chromosome_index: int, n_copies: int = 2) -> tuple[str, ...]:
    if n_copies not in (1, 2):
        raise PreparationError("n_copies must be one or two")
    prefix = "c%02d" % (int(chromosome_index) + 1)
    return tuple(prefix + suffix for suffix in ("a", "b")[:n_copies])


def _mask_n_copies(condition_id: str) -> int:
    return 1 if condition_id == "consensus014" else 2


def _load_3dg(path: Path) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {}
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
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
                raise PreparationError("invalid 3dg row %d in %s" % (line_no, path)) from exc
            if position in result.setdefault(track, {}):
                raise PreparationError("duplicate coordinate %s:%d" % (track, position))
            result[track][position] = point if np.isfinite(point).all() else np.full(3, np.nan, dtype=np.float64)
    return result


def _load_native_tsv(path: Path) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chr", "copy", "start", "x", "y", "z"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise PreparationError("native TSV lacks required fields: %s" % sorted(required))
        chromosome_index = {name: index for index, (name, _length) in enumerate(CHROMOSOMES)}
        for line_no, row in enumerate(reader, start=2):
            try:
                chromosome = str(row["chr"])
                copy_index = int(row["copy"])
                position = int(row["start"])
                point = np.asarray([float(row[axis]) for axis in ("x", "y", "z")], dtype=np.float64)
                index = chromosome_index[chromosome]
            except (KeyError, TypeError, ValueError) as exc:
                raise PreparationError("invalid native TSV row %d in %s" % (line_no, path)) from exc
            if copy_index not in (0, 1) or position < 0 or position % 1_000_000 != 0:
                raise PreparationError("invalid native TSV chromosome/copy/start at row %d" % line_no)
            track = "c%02d%s" % (index + 1, "ab"[copy_index])
            if position in result.setdefault(track, {}):
                raise PreparationError("duplicate coordinate %s:%d" % (track, position))
            result[track][position] = point if np.isfinite(point).all() else np.full(3, np.nan, dtype=np.float64)
    return result


def load_coordinates(path: Path, format_name: str) -> dict[str, dict[int, np.ndarray]]:
    if format_name in ("3dg", "3dg_text"):
        return _load_3dg(path)
    if format_name in ("native_tsv", "coords_tsv"):
        return _load_native_tsv(path)
    raise PreparationError("unsupported coordinate format: %s" % format_name)


def dense_points(structures: Mapping[str, Mapping[int, np.ndarray]], track: str,
                 positions: np.ndarray) -> np.ndarray:
    points = np.full((len(positions), 3), np.nan, dtype=np.float64)
    rows = structures.get(track)
    if not isinstance(rows, Mapping):
        return points
    for index, position in enumerate(positions):
        value = rows.get(int(position))
        if value is None:
            value = rows.get(str(int(position)))
        if value is None:
            continue
        try:
            point = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if point.shape == (3,) and np.isfinite(point).all():
            points[index] = point
    return points


def distance_vector(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    values = np.full(len(pair_i), np.nan, dtype=np.float64)
    if not len(pair_i):
        return values
    finite = np.isfinite(points[pair_i]).all(axis=1) & np.isfinite(points[pair_j]).all(axis=1)
    if finite.any():
        delta = points[pair_i[finite]] - points[pair_j[finite]]
        values[finite] = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float64), dtype=np.float64)
    return values


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata_average(np.asarray(x, dtype=np.float64))
    ry = _rankdata_average(np.asarray(y, dtype=np.float64))
    sx = float(rx.std())
    sy = float(ry.std())
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def rho_with_reason(candidate: np.ndarray, reference: np.ndarray) -> tuple[float | None, str | None]:
    if len(candidate) < MIN_COMMON_PAIRS:
        return None, "insufficient_common_pairs"
    if not (np.isfinite(candidate).all() and np.isfinite(reference).all()):
        return None, "nonfinite_distance_vector"
    if np.ptp(candidate) == 0:
        return None, "constant_candidate_distance"
    if np.ptp(reference) == 0:
        return None, "constant_reference_distance"
    value = spearman_rho(candidate, reference)
    if not np.isfinite(value):
        return None, "nonfinite_spearman"
    return float(value), None


def derive_four_rho(rhos: Mapping[str, Any]) -> dict[str, Any]:
    """根据单个 candidate chromosome 推导冻结的最佳交换 metrics。"""
    result: dict[str, Any] = {
        "direct": None, "swapped": None, "matched": None, "cross": None,
        "contrast": None, "margin_mat": None, "margin_pat": None, "minmargin": None,
        "orientation": "unresolved_missing", "geometry_tie": False,
        "both_positive": False, "margin_class": "not_applicable", "metric_status": "unavailable",
        "reason": None,
    }
    values = []
    missing_reasons = []
    for key in RAW_RHOS:
        value = rhos.get(key)
        if _finite(value):
            values.append(float(value))
        else:
            missing_reasons.append("%s=%s" % (key, rhos.get("%s_reason" % key, "missing")))
    if missing_reasons:
        result["reason"] = "; ".join(missing_reasons)
        return result
    a_mat, a_pat, b_mat, b_pat = (float(rhos[key]) for key in RAW_RHOS)
    direct = (a_mat + b_pat) / 2.0
    cross = (a_pat + b_mat) / 2.0
    result["direct"] = float(direct)
    result["swapped"] = float(cross)
    if abs(direct - cross) <= ORIENTATION_TIE_TOL:
        matched = float((direct + cross) / 2.0)
        result.update({
            "matched": matched, "cross": matched, "contrast": 0.0,
            "orientation": "unresolved_tie", "geometry_tie": True,
            "metric_status": "ok", "reason": "direct_cross_tie_within_1e-12",
        })
        return result
    if direct > cross:
        a, b, c, d = a_mat, a_pat, b_mat, b_pat
        result["orientation"] = "direct"
    else:
        a, b, c, d = b_mat, b_pat, a_mat, a_pat
        result["orientation"] = "swapped"
    matched = (a + d) / 2.0
    other = (b + c) / 2.0
    margin_mat = a - b
    margin_pat = d - c
    result.update({
        "matched": float(matched), "cross": float(other), "contrast": float(matched - other),
        "margin_mat": float(margin_mat), "margin_pat": float(margin_pat),
        "minmargin": float(min(margin_mat, margin_pat)), "metric_status": "ok",
        "margin_class": "both_positive" if margin_mat > ORIENTATION_TIE_TOL and margin_pat > ORIENTATION_TIE_TOL
        else ("both_negative" if margin_mat < -ORIENTATION_TIE_TOL and margin_pat < -ORIENTATION_TIE_TOL else "one_negative"),
    })
    result["both_positive"] = result["margin_class"] == "both_positive"
    return result


def _load_mask_lock(config: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = resolve_path(config["mask_lock"]["expected_pair_counts_path"], PREP_DIR)
    lock = read_json(lock_path)
    if lock.get("schema_version") != "p9016-multires-r2-mask-lock-v1":
        raise PreparationError("mask lock schema mismatch")
    if lock.get("manifest_sha256") != MASK_MANIFEST_SHA256:
        raise PreparationError("mask lock published manifest SHA mismatch")
    expected = [(str(item["chromosome"]), int(item["n_bins"]), int(item["n_total_non_diagonal_pairs"]), int(item["n_common_pairs"]))
               for item in lock.get("expected_by_chromosome", [])]
    if tuple(expected) != EXPECTED_MASK_COUNTS:
        raise PreparationError("mask lock expected counts changed")
    if lock.get("condition_count") != 21 or lock.get("new_endpoint_inclusion") is not False:
        raise PreparationError("mask lock must be the fixed 21-condition mask")
    return lock


def validate_preparation(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """验证 preparation contracts，不打开 candidate 或 reference 字节。"""
    config_file = Path(config_path).resolve()
    config = read_json(config_file)
    if config.get("schema_version") != "p9016-multires-r2-config-v1":
        raise PreparationError("unexpected preparation config schema")
    if config.get("status") != "prepared_only_pending_parent_release":
        raise PreparationError("preparation config is not pending release")
    protocol_file = resolve_path(str(config["protocol_path"]), config_file.parent)
    protocol = read_json(protocol_file)
    protocol_sha = sha256_file(protocol_file)
    if config.get("protocol_sha256") != protocol_sha:
        raise PreparationError("config protocol SHA does not match protocol file")
    if protocol.get("schema_version") != "p9016-multires-r2-protocol-v1":
        raise PreparationError("unexpected protocol schema")
    if protocol.get("training_definition", {}).get("main_training_backend") != "GPU":
        raise PreparationError("main training backend must be GPU")
    if protocol.get("training_definition", {}).get("main_endpoint_backend") != "GPU":
        raise PreparationError("main endpoint backend must be GPU")
    if protocol.get("training_definition", {}).get("gpu_training_plan") != "all 5 variants x 2 sources x 3 layers = 30 stages, each restarted from the original 014 source":
        raise PreparationError("GPU 30-stage training plan is not frozen")
    backend_policy = protocol.get("training_definition", {}).get("backend_policy", {})
    if (backend_policy.get("gpu_selection_schema") != GPU_SELECTION_SCHEMA
            or backend_policy.get("gpu_terminal_schema") != GPU_TERMINAL_SCHEMA
            or backend_policy.get("gpu_formal_output_path") != "test_res/036-20260914T064651Z-gpu-multires"
            or backend_policy.get("gpu_preflight_required_checks") !=
            ["cuda_target_visible", "float64_value_gradient_parity", "protocol_consistency", "gpu_c0_training_complete"]
            or backend_policy.get("cpu_byte_identity_required") is not False
            or backend_policy.get("cpu_r2_match_required") is not False):
        raise PreparationError("GPU success payload/preflight boundary is not frozen")
    if protocol.get("evaluation_definition", {}).get("orientation", {}).get("tie_tolerance") != ORIENTATION_TIE_TOL:
        raise PreparationError("orientation tie tolerance is not 1e-12")
    selection = protocol.get("training_definition", {}).get("endpoint_selection", {})
    if selection.get("tie_tolerance_per_record") != SELECTION_TIE_TOL_PER_RECORD:
        raise PreparationError("selection tie tolerance is not 1e-9")
    if selection.get("tie_order") != list(SOURCE_IDS):
        raise PreparationError("selection tie order is not the frozen source order")
    if selection.get("includes_diag_constants") is not True or selection.get("includes_priors") is not False:
        raise PreparationError("selection objective flags are not frozen")
    chromosome_grid(config)
    training = config.get("training", {})
    if (training.get("main_backend") != "GPU" or training.get("main_endpoint_backend") != "GPU"
            or training.get("gpu_stage_count") != 30
            or training.get("gpu_controller_schema") != "pending_child_terminal_release"
            or training.get("gpu_selection_schema") != GPU_SELECTION_SCHEMA
            or training.get("gpu_terminal_schema") != GPU_TERMINAL_SCHEMA
            or training.get("gpu_formal_output_path") != "test_res/036-20260914T064651Z-gpu-multires"
            or training.get("gpu_preflight_required_checks") !=
            ["cuda_target_visible", "float64_value_gradient_parity", "protocol_consistency", "gpu_c0_training_complete"]
            or training.get("cpu_byte_identity_required") is not False
            or training.get("cpu_r2_match_required") is not False):
        raise PreparationError("GPU main training contract is not frozen")
    if training.get("cpu_033_and_cpu_020_role") != "independent historical anchors; not GPU C0 and not GPU selection inputs":
        raise PreparationError("CPU historical anchor role changed")
    if config.get("cohort", {}).get("raw_record_count") != 1703888:
        raise PreparationError("raw record count changed")
    if config.get("cohort", {}).get("track_count") != 40 or config.get("cohort", {}).get("final_physical_beads") != 5290:
        raise PreparationError("cohort geometry count changed")
    if config.get("mask_lock", {}).get("manifest_sha256") != MASK_MANIFEST_SHA256:
        raise PreparationError("config mask manifest SHA mismatch")
    mask_lock = _load_mask_lock(config)
    manifest_path = resolve_path(config["mask_lock"]["manifest_path"], config_file.parent)
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != MASK_MANIFEST_SHA256:
        raise PreparationError("published mask manifest SHA mismatch")
    manifest = read_json(manifest_path)
    metadata = manifest.get("mask_metadata")
    if not isinstance(metadata, list) or len(metadata) != 20:
        raise PreparationError("published mask manifest lacks 20 mask records")
    observed = tuple((str(item["chromosome"]), int(item["n_bins"]), int(item["n_total_non_diagonal_pairs"]), int(item["n_common_pairs"]))
                    for item in metadata)
    if observed != EXPECTED_MASK_COUNTS:
        raise PreparationError("published mask metadata pair counts differ from frozen mask lock")
    endpoints = config.get("endpoints")
    if [item.get("id") for item in endpoints] != list(ENDPOINT_IDS):
        raise PreparationError("config endpoint order/count differs from 10-endpoint release")
    if any(item.get("n_copies") != 2 for item in endpoints):
        raise PreparationError("all 10 new endpoints must be two-copy models")
    plot_policy = config.get("plot_policy", {})
    if (plot_policy.get("main_conditions") != ["C0", "C1", "C2-map", "C2-free", "C3", "020 / C0 CPU"]
            or plot_policy.get("main_gpu_condition_count") != 5
            or plot_policy.get("historical_cpu_anchor_label") != "020 / C0 CPU"
            or plot_policy.get("historical_cpu_anchor_style") != "gray"
            or plot_policy.get("historical_cpu_anchor_is_gpu_endpoint") is not False
            or plot_policy.get("historical_cpu_anchor_enters_gpu_selection") is not False
            or plot_policy.get("source_strata_conditions") != list(VARIANT_IDS)
            or plot_policy.get("source_strata_backend") != "GPU"):
        raise PreparationError("GPU plot/CPU historical-anchor contract changed")
    pending_endpoints = sum(item.get("path") is None and item.get("sha256") is None for item in endpoints)
    if pending_endpoints != 10:
        raise PreparationError("preparation config must keep all 10 endpoint paths/digests pending")
    if config.get("reference", {}).get("sha256") != REFERENCE_SHA256:
        raise PreparationError("reference SHA differs from frozen P9016 reference")
    post = config.get("post_release_acceptance", {})
    if (post.get("status") != "future_after_hash_gate"
            or post.get("legacy_mask_builder_path") != "pr/allele_r2.py"
            or post.get("legacy_mask_builder_sha256") != LEGACY_ALLELE_R2_SHA256
            or post.get("legacy_common_mask_builder_path") != "pr/r2comparison.py"
            or post.get("legacy_common_mask_builder_sha256") != LEGACY_R2COMPARISON_SHA256
            or post.get("published_r2_table_path") != "test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_per_chromosome.tsv"
            or post.get("published_condition_id") != HISTORICAL_020_ID
            or post.get("published_comparison_atol") != 1e-12
            or post.get("mask_compare_fields") != ["positions", "pair_i", "pair_j", "common"]
            or post.get("published_table_only") is not True
            or post.get("use_020_raw_evaluation") is not False
            or post.get("c0_selected_cpu_gpu_diagnostic") is not True
             or post.get("c0_cpu_gpu_diagnostic_nonblocking") is not True
             or post.get("c0_cpu_gpu_diagnostic_reject_on_difference") is not False
             or post.get("historical_cpu_anchor_label") != "020 / C0 CPU"
             or post.get("historical_cpu_anchor_is_not_gpu_endpoint") is not True
             or post.get("gpu_main_endpoints_same_backend") is not True):
        raise PreparationError("post-release acceptance contract changed")
    return {
        "schema_version": "p9016-multires-r2-preparation-validation-v1",
        "status": "PASS",
        "prepared_only": True,
        "evaluation_not_run": True,
        "candidate_coordinates_opened": 0,
        "reference_opened": False,
        "phase_payload_opened": False,
        "fit_called": False,
        "native_called": False,
        "protocol_sha256": protocol_sha,
        "published_mask_manifest_sha256": manifest_sha,
        "mask_counts_match": True,
        "mask_condition_count": 21,
        "endpoint_count": 10,
        "main_backend": "GPU",
        "gpu_stage_count": 30,
        "historical_cpu_anchor": "020 / C0 CPU",
        "historical_cpu_anchor_is_gpu_endpoint": False,
        "source_strata_backend": "GPU",
        "pending_endpoint_paths": pending_endpoints,
        "selection_tie_tolerance_per_record": SELECTION_TIE_TOL_PER_RECORD,
        "orientation_tie_tolerance": ORIENTATION_TIE_TOL,
        "post_release_acceptance": "future_after_hash_gate",
        "post_release_mask_compare_fields": ["positions", "pair_i", "pair_j", "common"],
        "post_release_published_condition_id": HISTORICAL_020_ID,
        "post_release_numeric_atol": 1e-12,
        "source_order": list(SOURCE_IDS),
        "note": "No candidate coordinate, reference 3DG, pairs/rawphase, fit, or native input was opened.",
    }


def _validate_hash_file(path: Path, expected: Any, label: str) -> str:
    expected_sha = _require_sha(expected, label)
    if not path.is_file():
        raise PreparationError("missing locked file for %s: %s" % (label, path))
    actual = sha256_file(path)
    if actual != expected_sha:
        raise PreparationError("SHA mismatch for %s: %s != %s" % (label, actual, expected_sha))
    return actual


def _selection_value(item: Mapping[str, Any]) -> float | None:
    value = item.get("count_nll_per_record")
    return float(value) if _finite(value) else None


def _validate_selection(release: Mapping[str, Any], endpoints: Mapping[str, Mapping[str, Any]]) -> None:
    proof = release.get("train_only_selection_proof")
    if not isinstance(proof, Mapping) or proof.get("status") != "complete":
        raise PreparationError("train-only selection proof is not complete")
    if proof.get("criterion") != "count_nll_per_record" or proof.get("reported_field_alias") != "final count_nll_normalized":
        raise PreparationError("selection proof criterion/alias mismatch")
    if proof.get("tie_tolerance_per_record") != SELECTION_TIE_TOL_PER_RECORD:
        raise PreparationError("selection proof uses the wrong 1e-9 selection tie tolerance")
    if proof.get("tie_order") != list(SOURCE_IDS):
        raise PreparationError("selection proof source order mismatch")
    if proof.get("includes_diag_constants") is not True or proof.get("includes_priors") is not False:
        raise PreparationError("selection proof objective flags mismatch")
    if proof.get("scope") != "within_variant_between_two_sources_only":
        raise PreparationError("selection proof scope mismatch")
    if proof.get("reference_read") is not False or proof.get("r2_read") is not False or proof.get("phase_payload_read") is not False:
        raise PreparationError("selection proof is not train-only")
    selection_by_variant = release.get("selection_by_variant")
    if not isinstance(selection_by_variant, Mapping) or set(selection_by_variant) != set(VARIANT_IDS):
        raise PreparationError("selection proof must retain all five variants")
    for variant in VARIANT_IDS:
        record = selection_by_variant[variant]
        if not isinstance(record, Mapping):
            raise PreparationError("selection record is not an object: %s" % variant)
        for key, value in (("criterion", "count_nll_per_record"), ("reported_field_alias", "final count_nll_normalized"),
                           ("tie_tolerance_per_record", SELECTION_TIE_TOL_PER_RECORD), ("tie_order", list(SOURCE_IDS)),
                           ("includes_diag_constants", True), ("includes_priors", False)):
            if record.get(key) != value:
                raise PreparationError("selection record %s.%s mismatch" % (variant, key))
        candidate_rows = record.get("candidates")
        expected_ids = ["%s-%s" % (variant, source) for source in SOURCE_IDS]
        if (not isinstance(candidate_rows, list) or not all(isinstance(row, Mapping) for row in candidate_rows)
                or [row.get("endpoint_id") for row in candidate_rows] != expected_ids):
            raise PreparationError("selection record %s must list both sources in order" % variant)
        eligible: dict[str, float] = {}
        for row in candidate_rows:
            endpoint_id = str(row["endpoint_id"])
            if row.get("selection_eligible"):
                value = _selection_value(row)
                endpoint_value = _selection_value(endpoints[endpoint_id])
                if (value is None or endpoint_value is None
                        or not _finite(row.get("count_nll_normalized"))
                        or not _finite(endpoints[endpoint_id].get("count_nll_normalized"))
                        or abs(value - endpoint_value) > 1e-12):
                    raise PreparationError("selection value mismatch for %s" % endpoint_id)
                eligible[endpoint_id] = value
        if not eligible:
            expected_selected = None
            expected_status = "no_eligible_endpoint"
        else:
            minimum = min(eligible.values())
            expected_selected = next(endpoint_id for endpoint_id in expected_ids
                                     if endpoint_id in eligible and eligible[endpoint_id] <= minimum + SELECTION_TIE_TOL_PER_RECORD)
            expected_status = "tie_resolved_pre_registered_order" if len(eligible) == 2 and all(
                abs(value - minimum) <= SELECTION_TIE_TOL_PER_RECORD for value in eligible.values()
            ) else "unique_minimum"
        if record.get("selected_endpoint_id") != expected_selected:
            raise PreparationError("selection winner mismatch for %s" % variant)
        expected_source = endpoints[expected_selected]["source_id"] if expected_selected else None
        if record.get("selected_source_id") != expected_source:
            raise PreparationError("selection source mismatch for %s" % variant)
        if record.get("tie_status") != expected_status:
            raise PreparationError("selection tie status mismatch for %s" % variant)


def validate_release(config_path: str | Path, release_path: str | Path) -> dict[str, Any]:
    """验证 parent release，不打开坐标或 reference payload。"""
    config_file = Path(config_path).resolve()
    release_file = Path(release_path).resolve()
    config = read_json(config_file)
    release = read_json(release_file)
    if release.get("schema_version") != "p9016-multires-r2-release-contract-v1":
        raise PreparationError("release schema mismatch")
    if release.get("status") not in ("locked_for_evaluation", "released"):
        raise PreparationError("release is still pending or not locked")
    if release.get("prepared_config", {}).get("sha256") != sha256_file(config_file):
        raise PreparationError("release prepared config SHA mismatch")
    protocol_file = resolve_path(release.get("protocol", {}).get("path", ""), release_file.parent)
    mask_lock_file = resolve_path(release.get("mask_lock", {}).get("path", ""), release_file.parent)
    if _validate_hash_file(protocol_file, release.get("protocol", {}).get("sha256"), "release.protocol") != sha256_file(protocol_file):
        raise PreparationError("protocol hash check failed")
    if _validate_hash_file(mask_lock_file, release.get("mask_lock", {}).get("sha256"), "release.mask_lock") != sha256_file(mask_lock_file):
        raise PreparationError("mask lock hash check failed")
    if release.get("mask_lock", {}).get("published_manifest_sha256") != MASK_MANIFEST_SHA256:
        raise PreparationError("release does not lock the published 029 mask manifest")
    reference = release.get("reference", {})
    if reference.get("path") != config.get("reference", {}).get("path") or reference.get("sha256") != REFERENCE_SHA256:
        raise PreparationError("release reference lock mismatch")
    source_locks = release.get("source_locks")
    if not isinstance(source_locks, list) or [item.get("source_id") for item in source_locks] != list(SOURCE_IDS):
        raise PreparationError("release source lock order mismatch")
    for item in source_locks:
        if item.get("status") != "locked":
            raise PreparationError("source lock is not complete: %s" % item.get("source_id"))
        for key in ("raw_source_sha256", "snapshot_sha256"):
            _require_sha(item.get(key), "source.%s.%s" % (item.get("source_id"), key))
    controller = release.get("training_controller", {})
    if (controller.get("status") != "complete" or controller.get("expected_endpoint_count") != 10
            or controller.get("main_backend") != "GPU"
            or controller.get("main_endpoint_backend") != "GPU"
            or controller.get("expected_stage_count") != 30
            or not isinstance(controller.get("gpu_controller_schema"), str)
            or controller.get("gpu_controller_schema") in ("", "pending_child_terminal_release")
            or controller.get("gpu_selection_schema") != GPU_SELECTION_SCHEMA
            or controller.get("gpu_terminal_schema") != GPU_TERMINAL_SCHEMA):
        raise PreparationError("training controller is not a complete unified-GPU 30-stage release")
    if controller.get("terminal_endpoint_count") != 10 or controller.get("active_jobs") != [] or controller.get("all_endpoint_terminal") is not True:
        raise PreparationError("training controller terminal evidence is incomplete")
    for key in ("manifest_sha256", "selection_sha256", "termination_sha256"):
        _require_sha(controller.get(key), "training_controller.%s" % key)
    for key in ("manifest_path", "selection_path", "termination_path"):
        if not isinstance(controller.get(key), str) or not controller.get(key):
            raise PreparationError("training controller evidence path missing: %s" % key)
    preflight = release.get("gpu_preflight_gate", {})
    required_preflight = ["cuda_target_visible", "float64_value_gradient_parity", "protocol_consistency", "gpu_c0_training_complete"]
    if (preflight.get("status") != "PASS"
            or preflight.get("required_checks") != required_preflight
            or preflight.get("cpu_byte_identity_required") is not False
            or preflight.get("cpu_r2_match_required") is not False):
        raise PreparationError("GPU CUDA/gradient/protocol/C0 preflight gate is not PASS")
    _require_sha(preflight.get("evidence_sha256"), "gpu_preflight_gate.evidence_sha256")
    if not isinstance(preflight.get("evidence_path"), str) or not preflight.get("evidence_path"):
        raise PreparationError("GPU preflight evidence path missing")
    gate = release.get("c0_numeric_gate", {})
    if (gate.get("status") != "PASS" or gate.get("backend") != "GPU"
            or gate.get("reference_read") is not False or gate.get("phase_read") is not False
            or gate.get("historical_cpu_033_gate_is_not_gpu_c0") is not True):
        raise PreparationError("GPU C0 numeric no-reference gate is not PASS")
    if gate.get("record_count") != 1703888 or gate.get("track_count") != 40:
        raise PreparationError("C0 numeric gate cohort count mismatch")
    _require_sha(gate.get("evidence_sha256"), "c0_numeric_gate.evidence_sha256")
    if not isinstance(gate.get("evidence_path"), str) or not gate.get("evidence_path"):
        raise PreparationError("C0 numeric gate evidence path missing")
    endpoint_rows = release.get("endpoints")
    if not isinstance(endpoint_rows, list) or [item.get("id") for item in endpoint_rows] != list(ENDPOINT_IDS):
        raise PreparationError("release must preserve all 10 endpoint rows in frozen order")
    endpoints = {str(item["id"]): item for item in endpoint_rows}
    for endpoint_id, item in endpoints.items():
        variant = next((candidate for candidate in VARIANT_IDS if endpoint_id.startswith(candidate + "-")), None)
        if variant is None:
            raise PreparationError("endpoint ID has unknown variant: %s" % endpoint_id)
        source = endpoint_id[len(variant) + 1:]
        if item.get("variant_id") != variant or item.get("source_id") != source:
            raise PreparationError("endpoint variant/source mismatch: %s" % endpoint_id)
        if item.get("n_copies") != 2 or item.get("backend") != "GPU":
            raise PreparationError("all new endpoints must be two-copy GPU models: %s" % endpoint_id)
        if not isinstance(item.get("endpoint_status"), str) or item["endpoint_status"] in ("", "pending_parent_release"):
            raise PreparationError("endpoint status is missing: %s" % endpoint_id)
        for key in ("terminal_record_sha256",):
            _require_sha(item.get(key), "%s.%s" % (endpoint_id, key))
        coordinate_status = item.get("coordinate_status")
        if coordinate_status == "available":
            if not isinstance(item.get("coordinate_path"), str) or not item.get("coordinate_path"):
                raise PreparationError("available endpoint lacks coordinate path: %s" % endpoint_id)
            _require_sha(item.get("coordinate_sha256"), "%s.coordinate_sha256" % endpoint_id)
        elif coordinate_status == "failed":
            if not isinstance(item.get("failure_reason"), str) or not item.get("failure_reason"):
                raise PreparationError("failed endpoint lacks failure reason: %s" % endpoint_id)
            if item.get("coordinate_path") is not None or item.get("coordinate_sha256") is not None:
                raise PreparationError("failed endpoint must not supply coordinate: %s" % endpoint_id)
        else:
            raise PreparationError("endpoint coordinate status is not locked: %s" % endpoint_id)
        if item.get("source_hashes_complete") is not True or not isinstance(item.get("source_hashes"), Mapping):
            raise PreparationError("endpoint source hashes incomplete: %s" % endpoint_id)
        for key, value in item["source_hashes"].items():
            _require_sha(value, "%s.source_hashes.%s" % (endpoint_id, key))
        if not isinstance(item.get("fit_called"), bool):
            raise PreparationError("endpoint fit_called flag missing: %s" % endpoint_id)
        if item.get("coordinate_status") == "available" and _selection_value(item) is None:
            raise PreparationError("available endpoint lacks count selection value: %s" % endpoint_id)
    _validate_selection(release, endpoints)
    proof = release["train_only_selection_proof"]
    _require_sha(proof.get("evidence_sha256"), "train_only_selection_proof.evidence_sha256")
    if not isinstance(proof.get("evidence_path"), str) or not proof.get("evidence_path"):
        raise PreparationError("selection proof evidence path missing")
    code_hashes = release.get("code_hashes", {})
    for path_key, sha_key in (("evaluator_path", "evaluator_sha256"), ("plotter_path", "plotter_sha256"),
                               ("protocol_path", "protocol_sha256"), ("config_path", "config_sha256"),
                               ("mask_lock_path", "mask_lock_sha256")):
        code_path = resolve_path(code_hashes.get(path_key, ""), release_file.parent)
        _validate_hash_file(code_path, code_hashes.get(sha_key), "code.%s" % path_key)
    post = release.get("post_release_acceptance", {})
    if (post.get("status") != "required_after_hash_gate"
            or post.get("legacy_mask_builder_path") != "pr/allele_r2.py"
            or post.get("legacy_mask_builder_sha256") != LEGACY_ALLELE_R2_SHA256
            or post.get("legacy_common_mask_builder_path") != "pr/r2comparison.py"
            or post.get("legacy_common_mask_builder_sha256") != LEGACY_R2COMPARISON_SHA256
            or post.get("published_r2_table_path") != "test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/r2_per_chromosome.tsv"
            or post.get("published_r2_table_sha256") != PUBLISHED_R2_SHA256
            or post.get("published_condition_id") != HISTORICAL_020_ID
            or post.get("published_numeric_atol") != 1e-12
            or post.get("mask_compare_fields") != ["positions", "pair_i", "pair_j", "common"]
            or post.get("mask_compare_mode") != "exact_array_equality_per_chromosome"
            or post.get("use_020_raw_evaluation") is not False
            or post.get("c0_selected_cpu_gpu_diagnostic") is not True
             or post.get("c0_cpu_gpu_diagnostic_nonblocking") is not True
             or post.get("c0_cpu_gpu_diagnostic_reject_on_difference") is not False
             or post.get("historical_cpu_anchor_label") != "020 / C0 CPU"
             or post.get("historical_cpu_anchor_is_not_gpu_endpoint") is not True
             or post.get("gpu_main_endpoints_same_backend") is not True):
        raise PreparationError("release post-acceptance contract mismatch")
    return {"config": config, "release": release, "endpoints": endpoints,
            "protocol_path": protocol_file, "mask_lock_path": mask_lock_file,
            "release_path": release_file}


def _mask_manifest_records(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    manifest_path = resolve_path(config["mask_lock"]["manifest_path"], PREP_DIR)
    manifest_sha = _validate_hash_file(manifest_path, MASK_MANIFEST_SHA256, "published_mask_manifest")
    manifest = read_json(manifest_path)
    rows = manifest.get("coordinates")
    if not isinstance(rows, list) or [item.get("condition_id") for item in rows] != list(EXPECTED_MASK_IDS):
        raise PreparationError("published mask manifest condition order changed")
    return manifest_path, manifest, [dict(item) for item in rows]


def _hash_release_evidence(release_info: Mapping[str, Any]) -> dict[str, Any]:
    """在任何 reference access 之前 hash terminal/selection/gate evidence。"""
    release = release_info["release"]
    release_file = Path(release_info["release_path"])
    controller = release["training_controller"]
    controller_hashes = {}
    for path_key, sha_key in (("manifest_path", "manifest_sha256"),
                              ("selection_path", "selection_sha256"),
                              ("termination_path", "termination_sha256")):
        path = resolve_path(controller[path_key], release_file.parent)
        controller_hashes[path_key] = {"path": str(path), "sha256": _validate_hash_file(path, controller[sha_key], "training_controller.%s" % path_key)}
    gate = release["c0_numeric_gate"]
    gate_path = resolve_path(gate["evidence_path"], release_file.parent)
    gate_hash = {"path": str(gate_path), "sha256": _validate_hash_file(gate_path, gate["evidence_sha256"], "c0_numeric_gate.evidence")}
    proof = release["train_only_selection_proof"]
    proof_path = resolve_path(proof["evidence_path"], release_file.parent)
    proof_hash = {"path": str(proof_path), "sha256": _validate_hash_file(proof_path, proof["evidence_sha256"], "train_only_selection_proof.evidence")}
    return {"training_controller": controller_hashes, "c0_numeric_gate": gate_hash, "train_only_selection_proof": proof_hash}


def hash_locked_inputs(config: Mapping[str, Any], release_info: Mapping[str, Any]) -> dict[str, Any]:
    """在 reference access 之前 hash 所有 candidate、evidence 和 frozen-mask 文件。"""
    endpoints = release_info["endpoints"]
    release_file = Path(release_info["release_path"])
    candidate_hashes = []
    for endpoint_id in ENDPOINT_IDS:
        item = endpoints[endpoint_id]
        terminal_path = resolve_path(item["terminal_record_path"], release_file.parent)
        terminal_sha = _validate_hash_file(terminal_path, item["terminal_record_sha256"], "%s.terminal_record" % endpoint_id)
        record = {"endpoint_id": endpoint_id, "terminal_record_path": str(terminal_path), "terminal_record_sha256": terminal_sha}
        if item.get("coordinate_status") == "available":
            coordinate_path = resolve_path(item["coordinate_path"], release_file.parent)
            coordinate_sha = _validate_hash_file(coordinate_path, item["coordinate_sha256"], "%s.coordinate" % endpoint_id)
            record.update({"coordinate_path": str(coordinate_path), "coordinate_sha256": coordinate_sha})
        candidate_hashes.append(record)
    mask_manifest_path, manifest, mask_rows = _mask_manifest_records(config)
    mask_hashes = []
    for item in mask_rows:
        path = resolve_path(item["path"], mask_manifest_path.parent)
        actual = _validate_hash_file(path, item.get("sha256"), "mask.%s" % item["condition_id"])
        mask_hashes.append({"condition_id": item["condition_id"], "path": str(path), "sha256": actual})
    evidence_hashes = _hash_release_evidence(release_info)
    return {"candidate_hashes": candidate_hashes, "evidence_hashes": evidence_hashes,
            "mask_manifest_path": str(mask_manifest_path),
            "mask_manifest_sha256": sha256_file(mask_manifest_path), "mask_hashes": mask_hashes,
            "mask_rows": mask_rows, "candidate_hashes_complete": True,
            "evidence_hashes_complete": True, "mask_hashes_complete": True}


def load_mask_structures(mask_rows: Sequence[Mapping[str, Any]], lock_info: Mapping[str, Any]) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    by_id: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    path_by_id = {item["condition_id"]: Path(item["path"]) for item in lock_info["mask_hashes"]}
    for item in mask_rows:
        condition_id = str(item["condition_id"])
        by_id[condition_id] = load_coordinates(path_by_id[condition_id], str(item.get("format", "3dg")))
    return by_id


def load_reference_after_lock(config: Mapping[str, Any], release_file: Path) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    spec = config["reference"]
    path = resolve_path(spec["path"], release_file.parent)
    actual = _validate_hash_file(path, spec["sha256"], "reference")
    return load_coordinates(path, "3dg"), {"path": str(path), "sha256": actual, "loaded_after_all_candidate_and_mask_hashes": True}


def _build_mask_for_chromosome(chromosome: str, chromosome_index: int, length_bp: int,
                               mask_structures: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
                               reference: Mapping[str, Mapping[int, np.ndarray]]) -> dict[str, Any]:
    positions = grid_positions(length_bp)
    pair_i, pair_j = np.triu_indices(len(positions), k=1)
    common = np.ones(len(pair_i), dtype=bool)
    for condition_id in EXPECTED_MASK_IDS:
        structures = mask_structures[condition_id]
        for track in default_tracks(chromosome_index, _mask_n_copies(condition_id)):
            common &= np.isfinite(distance_vector(dense_points(structures, track, positions), pair_i, pair_j))
    ref_mat = distance_vector(dense_points(reference, "%s(mat)" % chromosome, positions), pair_i, pair_j)
    ref_pat = distance_vector(dense_points(reference, "%s(pat)" % chromosome, positions), pair_i, pair_j)
    common &= np.isfinite(ref_mat) & np.isfinite(ref_pat)
    return {"chromosome": chromosome, "chromosome_index": chromosome_index, "positions": positions,
            "pair_i": pair_i, "pair_j": pair_j, "common": common, "ref_mat": ref_mat,
            "ref_pat": ref_pat, "n_bins": int(len(positions)), "n_total_non_diagonal_pairs": int(len(pair_i)),
            "n_common_pairs": int(common.sum()), "status": "ok" if int(common.sum()) >= MIN_COMMON_PAIRS else "insufficient_common_pairs"}


def build_frozen_masks(config: Mapping[str, Any], mask_structures: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
                       reference: Mapping[str, Mapping[int, np.ndarray]], mask_lock: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    expected = {item["chromosome"]: item for item in mask_lock["expected_by_chromosome"]}
    masks = {}
    for index, (chromosome, length_bp) in enumerate(chromosome_grid(config)):
        mask = _build_mask_for_chromosome(chromosome, index, length_bp, mask_structures, reference)
        expected_row = expected[chromosome]
        for key in ("n_bins", "n_total_non_diagonal_pairs", "n_common_pairs"):
            if mask[key] != int(expected_row[key]):
                raise PreparationError("frozen mask count mismatch for %s: %s=%s expected=%s" %
                                       (chromosome, key, mask[key], expected_row[key]))
        masks[chromosome] = mask
    return masks


def compare_mask_exact(new_masks: Mapping[str, Mapping[str, Any]],
                       legacy_masks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """比较冻结 mask 的 positions 和 bits，而非只比较聚合 pair counts。"""
    by_chromosome = []
    fields = ("positions", "pair_i", "pair_j", "common")
    for chromosome, _length in CHROMOSOMES:
        left = new_masks.get(chromosome)
        right = legacy_masks.get(chromosome)
        chromosome_checks = []
        for field in fields:
            if left is None or right is None:
                same = False
                mismatch_count = None
            else:
                left_array = np.asarray(left[field])
                right_array = np.asarray(right[field])
                same = bool(np.array_equal(left_array, right_array))
                if left_array.shape != right_array.shape:
                    mismatch_count = None
                else:
                    mismatch_count = int(np.count_nonzero(left_array != right_array))
            chromosome_checks.append({"field": field, "same": same, "mismatch_count": mismatch_count})
        by_chromosome.append({"chromosome": chromosome, "checks": chromosome_checks,
                              "same": all(item["same"] for item in chromosome_checks)})
    passed = all(item["same"] for item in by_chromosome)
    return {"schema_version": "p9016-multires-r2-mask-exact-comparison-v1",
            "status": "PASS" if passed else "FAIL", "atol": 0.0,
            "fields": list(fields), "by_chromosome": by_chromosome,
            "same_positions_and_pair_bits": passed}


def _legacy_rebuild_masks_after_gate(config: Mapping[str, Any], lock_info: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[int, np.ndarray]]], dict[str, Any]]:
    """通过冻结的 legacy parser/common-mask path 重建 mask。

    该函数仅在 hash_locked_inputs 和 reference hashing 之后调用。在此处导入并解析 legacy path，可使 parity check 独立于本模块的 coordinate parser 和 mask construction。
    """
    legacy_sha = _validate_hash_file(LEGACY_ALLELE_R2_PATH, LEGACY_ALLELE_R2_SHA256, "legacy_coordinate_parser")
    legacy_common_sha = _validate_hash_file(LEGACY_R2COMPARISON_PATH, LEGACY_R2COMPARISON_SHA256, "legacy_common_mask_builder")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from pr import allele_r2 as legacy_allele_r2
    from pr import r2comparison as legacy_r2comparison

    chromosome_names = [name for name, _length in chromosome_grid(config)]
    path_by_id = {item["condition_id"]: Path(item["path"]) for item in lock_info["mask_hashes"]}
    legacy_structures = {}
    for item in lock_info["mask_rows"]:
        condition_id = str(item["condition_id"])
        legacy_structures[condition_id] = legacy_allele_r2._load_coordinates(
            path_by_id[condition_id], str(item.get("format", "3dg")), chromosome_names)
    reference_path = resolve_path(config["reference"]["path"], PREP_DIR)
    legacy_reference = legacy_allele_r2._load_coordinates(reference_path, "3dg", chromosome_names)
    legacy_masks = {}
    for chromosome_index, (chromosome, length_bp) in enumerate(chromosome_grid(config)):
        conditions = {
            condition_id: legacy_r2comparison.R2Condition(
                condition_id=condition_id, display_name=condition_id,
                structures=legacy_structures[condition_id], n_copies=_mask_n_copies(condition_id),
                role="frozen_mask_input", endpoint_status="historical_locked", accepted_as=condition_id)
            for condition_id in EXPECTED_MASK_IDS
        }
        old_mask = legacy_r2comparison.build_common_mask(
            chromosome, length_bp, conditions, legacy_reference, chromosome_index=chromosome_index)
        legacy_masks[chromosome] = {
            "chromosome": chromosome, "chromosome_index": chromosome_index,
            "positions": np.asarray(old_mask.positions, dtype=np.int64),
            "pair_i": np.asarray(old_mask.pair_i, dtype=np.int64),
            "pair_j": np.asarray(old_mask.pair_j, dtype=np.int64),
            "common": np.asarray(old_mask.common, dtype=bool),
            "n_bins": old_mask.n_bins, "n_total_non_diagonal_pairs": old_mask.n_total_pairs,
            "n_common_pairs": old_mask.n_common_pairs,
        }
    return legacy_masks, legacy_structures, {
        "legacy_mask_builder_path": str(LEGACY_ALLELE_R2_PATH),
        "legacy_mask_builder_sha256": legacy_sha,
        "legacy_common_mask_path": str(LEGACY_R2COMPARISON_PATH),
        "legacy_common_mask_sha256": legacy_common_sha,
        "legacy_coordinate_parser": "pr.allele_r2._load_coordinates",
        "legacy_common_mask_builder": "pr.r2comparison.build_common_mask",
        "reference_parser": "pr.allele_r2._load_coordinates",
    }


def _published_number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PreparationError("published R2 value is not numeric: %s" % value) from exc
    return parsed if math.isfinite(parsed) else None


def compare_published_r2_rows(new_rows: Sequence[Mapping[str, Any]], published_rows: Sequence[Mapping[str, Any]],
                              *, endpoint_label: str) -> dict[str, Any]:
    """将新 evaluator 行与一个已发布的 029 condition 比较。"""
    expected_chromosomes = [name for name, _length in CHROMOSOMES]
    published_by_chr = {str(row.get("chromosome")): row for row in published_rows}
    new_by_chr = {str(row.get("chromosome")): row for row in new_rows}
    mismatches = []
    compared_values = 0
    for chromosome in expected_chromosomes:
        old = published_by_chr.get(chromosome)
        new = new_by_chr.get(chromosome)
        if old is None or new is None:
            mismatches.append({"chromosome": chromosome, "field": "row", "reason": "missing_row"})
            continue
        if int(new.get("n_common_pairs_frozen", -1)) != int(old.get("n_common_pairs", -2)):
            mismatches.append({"chromosome": chromosome, "field": "n_common_pairs", "new": new.get("n_common_pairs_frozen"), "published": old.get("n_common_pairs")})
        for new_field, old_field in POST_ACCEPTANCE_NUMERIC_FIELDS:
            actual = _published_number(new.get(new_field))
            expected = _published_number(old.get(old_field))
            if actual is None and expected is None:
                continue
            compared_values += 1
            if actual is None or expected is None or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                mismatches.append({"chromosome": chromosome, "field": new_field, "published_field": old_field,
                                   "new": actual, "published": expected, "abs_diff": None if actual is None or expected is None else abs(actual - expected)})
        old_orientation = str(old.get("orientation") or "")
        if old_orientation and str(new.get("orientation") or "") != old_orientation:
            mismatches.append({"chromosome": chromosome, "field": "orientation", "new": new.get("orientation"), "published": old_orientation})
    return {"schema_version": "p9016-multires-r2-published-parity-v1", "status": "PASS" if not mismatches else "FAIL",
            "endpoint_label": endpoint_label, "published_condition_id": HISTORICAL_020_ID,
            "published_table_path": str(PUBLISHED_R2_PATH), "published_table_sha256": PUBLISHED_R2_SHA256,
            "atol": 1e-12, "numeric_fields": [{"new": left, "published": right} for left, right in POST_ACCEPTANCE_NUMERIC_FIELDS],
            "chromosome_count_expected": 20, "chromosome_count_published": len(published_by_chr),
            "compared_numeric_values": compared_values, "mismatches": mismatches}


def _read_published_r2_after_gate() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_sha = _validate_hash_file(PUBLISHED_R2_PATH, PUBLISHED_R2_SHA256, "published_029_r2_per_chromosome")
    with PUBLISHED_R2_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [row for row in rows if row.get("condition_id") == HISTORICAL_020_ID]
    if len(selected) != len(CHROMOSOMES):
        raise PreparationError("published 029 table has %d %s rows, expected 20" % (len(selected), HISTORICAL_020_ID))
    return selected, {"path": str(PUBLISHED_R2_PATH), "sha256": actual_sha,
                      "condition_id": HISTORICAL_020_ID, "row_count": len(selected),
                      "source": "029 published r2_per_chromosome.tsv only; no 020 raw evaluation table"}


def post_release_acceptance(config: Mapping[str, Any], release_info: Mapping[str, Any],
                            lock_info: Mapping[str, Any], masks: Mapping[str, Mapping[str, Any]],
                            evaluated_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """执行目标 post-release mask 与 historical-020 parity acceptance。"""
    legacy_masks, legacy_structures, legacy_provenance = _legacy_rebuild_masks_after_gate(config, lock_info)
    mask_comparison = compare_mask_exact(masks, legacy_masks)
    published_rows, published_provenance = _read_published_r2_after_gate()
    historical_endpoint = {
        "id": HISTORICAL_020_ID, "variant_id": "historical_020", "source_id": "random_joint_base2207",
        "n_copies": 2, "endpoint_status": "historical_locked", "coordinate_status": "available",
    }
    historical_rows = []
    for chromosome, _length in CHROMOSOMES:
        historical_rows.append(evaluate_endpoint_chromosome(
            historical_endpoint, legacy_structures[HISTORICAL_020_ID], masks[chromosome]))
    historical_parity = compare_published_r2_rows(historical_rows, published_rows, endpoint_label=HISTORICAL_020_ID)
    historical_anchor_rows = []
    for row in historical_rows:
        anchor = dict(row)
        anchor.update({
            "endpoint_id": HISTORICAL_CPU_ANCHOR_ID,
            "variant_id": "historical_cpu_anchor",
            "source_id": "historical_cpu_anchor",
            "source_display": "020 / C0 CPU",
            "endpoint_status": "historical_locked",
            "coordinate_status": "historical_anchor",
        })
        historical_anchor_rows.append(anchor)
    selected_c0 = release_info["release"]["selection_by_variant"]["C0"].get("selected_endpoint_id")
    if selected_c0 == "C0-random_joint_base2207":
        c0_rows = [row for row in evaluated_rows if row.get("endpoint_id") == selected_c0]
        c0_diagnostic = compare_published_r2_rows(c0_rows, published_rows, endpoint_label=selected_c0)
        c0_diagnostic["diagnostic_status"] = "PASS" if c0_diagnostic["status"] == "PASS" else "DIFFERENCE_OBSERVED"
    else:
        c0_diagnostic = {
            "status": "NOT_APPLICABLE",
            "diagnostic_status": "NOT_APPLICABLE",
            "selected_c0_endpoint_id": selected_c0,
            "rule": "compare GPU C0 to historical CPU 020 only when the GPU-selected C0 source is random",
        }
    c0_diagnostic["acceptance_effect"] = "non_blocking_diagnostic_only"
    c0_diagnostic["reject_on_difference"] = False
    passed = mask_comparison["status"] == "PASS" and historical_parity["status"] == "PASS"
    return {
        "schema_version": "p9016-multires-r2-post-release-acceptance-v2", "status": "PASS" if passed else "FAIL",
        "acceptance_phase": "after_gpu_candidate_mask_terminal_selection_source_code_protocol_hash_gate",
        "mask_exact_comparison": mask_comparison, "historical_020_random_parity": historical_parity,
        "gpu_cpu_c0_diagnostic": c0_diagnostic, "historical_cpu_anchor_rows": historical_anchor_rows,
        "legacy_provenance": legacy_provenance,
        "published_r2_provenance": published_provenance,
        "published_table_only": True, "used_020_raw_evaluation_values": False,
        "reference_was_read_before_hook": True, "candidate_rows_not_recomputed": True,
        "orientation_atol": ORIENTATION_TIE_TOL, "numeric_atol": 1e-12,
        "gpu_main_endpoints_same_backend": True,
        "historical_cpu_anchor_label": "020 / C0 CPU",
    }


def _na_row(endpoint: Mapping[str, Any], chromosome: Mapping[str, Any], reason: str) -> dict[str, Any]:
    row = {column: None for column in FULL_COLUMNS}
    row.update({
        "endpoint_id": endpoint["id"], "variant_id": endpoint["variant_id"], "source_id": endpoint["source_id"],
        "source_display": SOURCE_DISPLAY[endpoint["source_id"]], "chromosome": chromosome["chromosome"],
        "chromosome_index": chromosome["chromosome_index"], "endpoint_status": endpoint.get("endpoint_status"),
        "coordinate_status": endpoint.get("coordinate_status"), "n_copies": 2,
        "n_bins": chromosome["n_bins"], "n_total_non_diagonal_pairs": chromosome["n_total_non_diagonal_pairs"],
        "n_common_pairs_frozen": chromosome["n_common_pairs"], "metric_status": "unavailable",
        "reason": reason, "orientation": "failed" if endpoint.get("coordinate_status") == "failed" else "unresolved_missing",
        "geometry_tie": False, "both_positive": False, "margin_class": "not_applicable",
        "rho_reasons_json": "{}",
    })
    return row


def evaluate_endpoint_chromosome(endpoint: Mapping[str, Any], structures: Mapping[str, Mapping[int, np.ndarray]],
                                  mask: Mapping[str, Any]) -> dict[str, Any]:
    row = _na_row(endpoint, mask, "")
    row["reason"] = None
    pair_i = mask["pair_i"]
    pair_j = mask["pair_j"]
    common = mask["common"]
    if int(common.sum()) < MIN_COMMON_PAIRS:
        row["reason"] = "insufficient_common_pairs"
        return row
    chromosome_index = int(mask["chromosome_index"])
    rhos: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    for key, track, ref_key in (("rho_A_mat", "c%02da" % (chromosome_index + 1), "ref_mat"),
                                 ("rho_A_pat", "c%02da" % (chromosome_index + 1), "ref_pat"),
                                 ("rho_B_mat", "c%02db" % (chromosome_index + 1), "ref_mat"),
                                 ("rho_B_pat", "c%02db" % (chromosome_index + 1), "ref_pat")):
        candidate_dist = distance_vector(dense_points(structures, track, mask["positions"]), pair_i, pair_j)[common]
        reference_dist = np.asarray(mask[ref_key], dtype=np.float64)[common]
        value, reason = rho_with_reason(candidate_dist, reference_dist)
        rhos[key] = value
        if reason is not None:
            reasons[key] = reason
            rhos["%s_reason" % key] = reason
    derived = derive_four_rho(rhos)
    row.update({key: rhos.get(key) for key in RAW_RHOS})
    row.update(derived)
    row["rho_reasons_json"] = json.dumps(reasons, sort_keys=True, separators=(",", ":"))
    return row


def _rows_for_endpoint(endpoint: Mapping[str, Any], structures: Mapping[str, Mapping[int, np.ndarray]] | None,
                       masks: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for chromosome_index, (chromosome, _length) in enumerate(chromosome_grid(config)):
        mask = masks[chromosome]
        if endpoint.get("coordinate_status") != "available":
            row = _na_row(endpoint, mask, str(endpoint.get("failure_reason", "planned_endpoint_failed_before_r2")))
        elif structures is None:
            row = _na_row(endpoint, mask, "coordinate_not_loaded")
        else:
            row = evaluate_endpoint_chromosome(endpoint, structures, mask)
        rows.append(row)
    return rows


def _write_tsv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for raw in rows:
            writer.writerow({column: "" if raw.get(column) is None else _jsonable(raw.get(column)) for column in columns})
            count += 1
    return count


def representative_rows(release: Mapping[str, Any], endpoints: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for variant in VARIANT_IDS:
        selection = release["selection_by_variant"][variant]
        endpoint_id = selection.get("selected_endpoint_id")
        endpoint = endpoints.get(endpoint_id) if endpoint_id else None
        result.append({
            "variant_id": variant, "representative_endpoint_id": endpoint_id,
            "representative_source_id": endpoint.get("source_id") if endpoint else None,
            "representative_source_display": SOURCE_DISPLAY.get(endpoint.get("source_id")) if endpoint else None,
            "selection_status": selection.get("tie_status"), "criterion": selection.get("criterion"),
            "reported_field_alias": selection.get("reported_field_alias"),
            "selection_tie_tolerance_per_record": selection.get("tie_tolerance_per_record"),
            "selection_candidates_json": json.dumps(_jsonable(selection.get("candidates", [])), sort_keys=True, separators=(",", ":")),
        })
    return result


def source_delta_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_endpoint_chr = {(row["endpoint_id"], row["chromosome"]): row for row in rows}
    result = []
    for source in SOURCE_IDS:
        c0_id = "C0-%s" % source
        for variant in VARIANT_IDS:
            endpoint_id = "%s-%s" % (variant, source)
            for chromosome, _length in CHROMOSOMES:
                left = by_endpoint_chr[(endpoint_id, chromosome)]
                right = by_endpoint_chr[(c0_id, chromosome)]
                for metric in METRICS:
                    left_value = left.get(metric)
                    right_value = right.get(metric)
                    if _finite(left_value) and _finite(right_value):
                        delta = float(left_value) - float(right_value)
                        status, reason = "ok", None
                    else:
                        delta = None
                        status = "n/a"
                        reason = "variant_or_c0_metric_unavailable"
                    result.append({
                        "source_id": source, "source_display": SOURCE_DISPLAY[source], "variant_id": variant,
                        "endpoint_id": endpoint_id, "chromosome": chromosome, "c0_endpoint_id": c0_id,
                        "metric": metric, "variant_minus_c0": delta, "status": status, "reason": reason,
                    })
    return result


def _rows_by_endpoint_chr(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(row["endpoint_id"]), str(row["chromosome"])): row for row in rows}


def build_plot_arrays(rows: Sequence[Mapping[str, Any]], representatives: Sequence[Mapping[str, Any]],
                      historical_anchor_rows: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    by_key = _rows_by_endpoint_chr(rows)
    anchor_rows = list(historical_anchor_rows or [])
    if anchor_rows:
        by_key.update(_rows_by_endpoint_chr(anchor_rows))
    main_endpoint_ids = {item["variant_id"]: item.get("representative_endpoint_id") for item in representatives}
    main_conditions = list(VARIANT_IDS)
    if anchor_rows:
        main_endpoint_ids["020 / C0 CPU"] = HISTORICAL_CPU_ANCHOR_ID
        main_conditions.append("020 / C0 CPU")
    groups = [("main", main_conditions, main_endpoint_ids)]
    groups.extend(
        ("source_%s" % source, list(VARIANT_IDS), {variant: "%s-%s" % (variant, source) for variant in VARIANT_IDS})
        for source in SOURCE_IDS
    )
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "condition_order": main_conditions,
        "main_condition_order": main_conditions,
        "source_condition_order": list(VARIANT_IDS),
        "historical_cpu_anchor": {
            "id": HISTORICAL_CPU_ANCHOR_ID,
            "label": "020 / C0 CPU",
            "included": bool(anchor_rows),
            "style": "gray",
            "non_biological_replicate": True,
            "excluded_from_gpu_selection": True,
        },
        "metrics": list(METRICS), "n_chromosomes": 20, "arrays": {},
    }
    for label, condition_ids, endpoint_for_condition in groups:
        for metric in METRICS:
            array = np.full((len(condition_ids), len(CHROMOSOMES)), np.nan, dtype=np.float64)
            for condition_index, condition_id in enumerate(condition_ids):
                endpoint_id = endpoint_for_condition.get(condition_id)
                if endpoint_id is None:
                    continue
                for chromosome_index, (chromosome, _length) in enumerate(CHROMOSOMES):
                    row = by_key.get((endpoint_id, chromosome))
                    value = row.get(metric) if row is not None else None
                    if _finite(value):
                        array[condition_index, chromosome_index] = float(value)
            key = "%s_%s" % (label, metric)
            arrays[key] = array
            metadata["arrays"][key] = {
                "shape": list(array.shape), "n_finite": int(np.isfinite(array).sum()),
                "n_a": int(np.isnan(array).sum()), "condition_ids": condition_ids,
                "endpoint_ids": endpoint_for_condition,
            }
    return arrays, metadata


def save_plot_arrays(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()})
    write_json(path.with_name("plot_arrays_metadata.json"), metadata)


def _plot_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return -0.1, 0.1
    span = max(float(np.ptp(finite)), 0.05)
    lower = min(0.0, float(finite.min()) - 0.12 * span)
    upper = max(0.05, float(finite.max()) + 0.18 * span)
    if upper <= lower:
        upper = lower + 0.1
    return lower, upper


def render_four_panel(path_stem: Path, arrays: Mapping[str, np.ndarray], *, label: str,
                      condition_labels: Sequence[str] | None = None) -> dict[str, Any]:
    """渲染带显式 n/a counts 的 2x2 四指标 boxplot。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#7F7F7F")
    condition_labels = tuple(condition_labels or VARIANT_IDS)
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 6.0), squeeze=False)
    positions = np.arange(1, len(condition_labels) + 1, dtype=float)
    for axis, metric in zip(axes.flat, METRICS):
        values = np.asarray(arrays[metric], dtype=np.float64)
        finite_values = [value[np.isfinite(value)] for value in values]
        present = [(position, value) for position, value in zip(positions, finite_values) if len(value)]
        if present:
            box = axis.boxplot([item[1] for item in present], positions=[item[0] for item in present],
                               widths=0.48, patch_artist=True, showfliers=False,
                               medianprops={"color": "black", "linewidth": 0.8},
                               whiskerprops={"linewidth": 0.65}, capprops={"linewidth": 0.65},
                               boxprops={"linewidth": 0.65})
            present_index = 0
            for condition_index, value in enumerate(finite_values):
                if not len(value):
                    continue
                box["boxes"][present_index].set_facecolor(colors[condition_index])
                box["boxes"][present_index].set_alpha(0.58)
                present_index += 1
        for chromosome_index in range(values.shape[1]):
            finite_mask = np.isfinite(values[:, chromosome_index])
            if finite_mask.sum() >= 2:
                axis.plot(positions[finite_mask], values[finite_mask, chromosome_index], color="#8c8c8c",
                          linewidth=0.45, alpha=0.58, zorder=1)
            for condition_index in range(values.shape[0]):
                if np.isfinite(values[condition_index, chromosome_index]):
                    axis.scatter([positions[condition_index]], [values[condition_index, chromosome_index]],
                                 s=9, color=colors[condition_index], edgecolors="white", linewidths=0.22, zorder=3)
        n_a = [int(np.isnan(value).sum()) for value in values]
        tick_labels = []
        for condition, missing in zip(condition_labels, n_a):
            display = "020 /\nC0 CPU" if condition == "020 / C0 CPU" else str(condition)
            tick_labels.append("%s\nn/a\n%d/20" % (display, missing))
        axis.set_xticks(positions, tick_labels)
        axis.set_title(metric, pad=4)
        axis.set_ylabel("Spearman-derived rho")
        axis.set_ylim(*_plot_limits(values))
        axis.axhline(0.0, color="#555555", linewidth=0.45, linestyle="--", zorder=0)
        axis.grid(axis="y", color="#dddddd", linewidth=0.35, alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", labelsize=7, pad=2)
    fig.suptitle("P9016 R2 | %s" % label, fontsize=7, y=0.985)
    fig.text(0.5, 0.012, "Dots = chromosome; gray lines link the same chromosome; one cell, not biological replicates",
             ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.935, wspace=0.30, hspace=0.48)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for extension in ("png", "pdf"):
        target = path_stem.with_suffix("." + extension)
        fig.savefig(target, dpi=300, bbox_inches=None, pad_inches=0.04)
        paths[extension] = str(target)
    plt.close(fig)
    return {"png": paths["png"], "pdf": paths["pdf"], "metrics": list(METRICS),
            "condition_ids": list(condition_labels), "condition_labels": list(condition_labels), "n_a_by_metric": {metric: int(np.isnan(arrays[metric]).sum()) for metric in METRICS},
            "dpi": 300, "base_panel_inches": 3.0, "font_size_pt": 7, "linked_measurements": 20}


def render_result_plots(results_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """从已完成的 result JSON 渲染图，不打开坐标/reference。"""
    result = read_json(results_path)
    if result.get("status") != "evaluation_complete":
        raise PreparationError("plotter requires evaluation_complete results")
    rows = result.get("rows")
    reps = result.get("representatives")
    post_acceptance = result.get("post_release_acceptance", {})
    historical_anchor_rows = post_acceptance.get("historical_cpu_anchor_rows", [])
    if not isinstance(rows, list) or not isinstance(reps, list) or not isinstance(historical_anchor_rows, list):
        raise PreparationError("result JSON lacks rows/representatives/historical CPU anchor")
    arrays, metadata = build_plot_arrays(rows, reps, historical_anchor_rows)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise PreparationError("plot output must be fresh and empty: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    save_plot_arrays(output / "plot_arrays.npz", arrays, metadata)
    plot_metadata = {}
    for label, prefix, key, condition_labels in (
            ("main count-selected GPU representatives + historical CPU anchor", "main_r2_four_panel", "main", tuple(metadata["main_condition_order"])),
            ("consensus-derived initialization (base1103), GPU conditions", "source_consensus_joint_base1103_r2_four_panel", "source_consensus_joint_base1103", tuple(metadata["source_condition_order"])),
            ("random-derived initialization (base2207), GPU conditions", "source_random_joint_base2207_r2_four_panel", "source_random_joint_base2207", tuple(metadata["source_condition_order"]))):
        metric_arrays = {metric: arrays["%s_%s" % (key, metric)] for metric in METRICS}
        plot_metadata[key] = render_four_panel(output / prefix, metric_arrays, label=label,
                                               condition_labels=condition_labels)
    write_json(output / "plot_metadata.json", plot_metadata)
    return {"status": "plots_complete", "output_dir": str(output), "plot_metadata": plot_metadata}


def _write_mask_provenance(path: Path, lock_info: Mapping[str, Any], masks: Mapping[str, Mapping[str, Any]]) -> None:
    rows = []
    for chromosome, mask in masks.items():
        rows.append({"chromosome": chromosome, "n_bins": mask["n_bins"],
                     "n_total_non_diagonal_pairs": mask["n_total_non_diagonal_pairs"],
                     "n_common_pairs": mask["n_common_pairs"], "status": mask["status"],
                     "mask_source_condition_count": 21, "new_endpoint_included": False})
    write_json(path, {"schema_version": "p9016-multires-r2-mask-provenance-v1", "status": "frozen_mask_reused",
                      "published_manifest_path": lock_info["mask_manifest_path"],
                      "published_manifest_sha256": lock_info["mask_manifest_sha256"],
                      "historical_input_hashes": lock_info["mask_hashes"], "by_chromosome": rows,
                      "new_endpoints_entered_mask": False, "mask_scope_reselected": False})


def evaluate_released(config_path: str | Path, release_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """仅对完整 parent release 运行真实 R2 评估。"""
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise PreparationError("evaluation output must be fresh and empty: %s" % output)
    config = read_json(Path(config_path).resolve())
    release_info = validate_release(config_path, release_path)
    output.mkdir(parents=True, exist_ok=False)
    lock_info = hash_locked_inputs(config, release_info)
    # 这是 evaluator 可以 hash/open reference 的第一个位置。
    reference, reference_provenance = load_reference_after_lock(config, Path(release_info["release_path"]))
    mask_lock = _load_mask_lock(config)
    mask_structures = load_mask_structures(lock_info["mask_rows"], lock_info)
    masks = build_frozen_masks(config, mask_structures, reference, mask_lock)
    loaded_candidates: dict[str, dict[str, dict[int, np.ndarray]] | None] = {}
    release_file = Path(release_info["release_path"])
    for endpoint_id in ENDPOINT_IDS:
        endpoint = release_info["endpoints"][endpoint_id]
        if endpoint.get("coordinate_status") == "available":
            loaded_candidates[endpoint_id] = load_coordinates(resolve_path(endpoint["coordinate_path"], release_file.parent), "3dg")
        else:
            loaded_candidates[endpoint_id] = None
    rows: list[dict[str, Any]] = []
    for endpoint_id in ENDPOINT_IDS:
        endpoint = release_info["endpoints"][endpoint_id]
        rows.extend(_rows_for_endpoint(endpoint, loaded_candidates[endpoint_id], masks, config))
    post_acceptance = post_release_acceptance(config, release_info, lock_info, masks, rows)
    write_json(output / "post_release_acceptance.json", post_acceptance)
    if post_acceptance["status"] != "PASS":
        raise PreparationError("post-release acceptance failed; see %s" % (output / "post_release_acceptance.json"))
    representative = representative_rows(release_info["release"], release_info["endpoints"])
    deltas = source_delta_rows(rows)
    _write_tsv(output / "r2_full_10endpoint_x20chr.tsv", FULL_COLUMNS, rows)
    _write_tsv(output / "source_stratified_variant_minus_c0.tsv", DELTA_COLUMNS, deltas)
    _write_tsv(output / "representatives_by_variant.tsv", tuple(representative[0].keys()), representative)
    arrays, array_metadata = build_plot_arrays(rows, representative, post_acceptance["historical_cpu_anchor_rows"])
    save_plot_arrays(output / "plot_arrays.npz", arrays, array_metadata)
    _write_mask_provenance(output / "mask_provenance.json", lock_info, masks)
    input_provenance = {
        "schema_version": "p9016-multires-r2-input-provenance-v1", "status": "evaluation_complete",
        "release_path": str(release_file), "release_sha256": sha256_file(release_file),
        "candidate_and_terminal_hashes": lock_info["candidate_hashes"],
        "release_evidence_hashes": lock_info["evidence_hashes"],
        "mask_manifest_path": lock_info["mask_manifest_path"], "mask_manifest_sha256": lock_info["mask_manifest_sha256"],
        "mask_input_hashes": lock_info["mask_hashes"], "reference": reference_provenance,
        "source_locks_declared": release_info["release"]["source_locks"],
        "selection_proof_declared": release_info["release"]["train_only_selection_proof"],
        "post_release_acceptance": post_acceptance,
        "reference_read_after_all_candidate_and_mask_hashes": True,
    }
    write_json(output / "input_provenance.json", input_provenance)
    plot_dir = output / "plots"
    plot_metadata = {}
    for label, key, stem, condition_labels in (
            ("main count-selected GPU representatives + historical CPU anchor", "main", "main_r2_four_panel", tuple(array_metadata["main_condition_order"])),
            ("consensus-derived initialization (base1103), GPU conditions", "source_consensus_joint_base1103", "source_consensus_joint_base1103_r2_four_panel", tuple(array_metadata["source_condition_order"])),
            ("random-derived initialization (base2207), GPU conditions", "source_random_joint_base2207", "source_random_joint_base2207_r2_four_panel", tuple(array_metadata["source_condition_order"]))):
        metric_arrays = {metric: arrays["%s_%s" % (key, metric)] for metric in METRICS}
        plot_metadata[key] = render_four_panel(plot_dir / stem, metric_arrays, label=label,
                                               condition_labels=condition_labels)
    validation = validate_result_rows(rows, masks, representative)
    validation["status"] = "PASS"
    validation["reference_read_after_all_hashes"] = True
    validation["evaluation_not_run_before_release"] = False
    write_json(output / "validation.json", validation)
    terminal_evidence = {
        "schema_version": "p9016-multires-r2-terminal-evidence-v1", "status": "evaluation_complete",
        "evaluator_fit_called": False, "native_called": False, "phase_payload_opened": False,
        "candidate_count": 10, "row_count": len(rows), "reference_loaded_after_all_hashes": True,
        "release_terminal_controller": release_info["release"]["training_controller"],
        "release_evidence_hashes": lock_info["evidence_hashes"],
        "post_release_acceptance": {"path": str(output / "post_release_acceptance.json"), "status": post_acceptance["status"]},
        "candidate_and_mask_hashes_complete": True, "outputs_written": True,
    }
    write_json(output / "terminal_evidence.json", terminal_evidence)
    result_payload = {
        "schema_version": "p9016-multires-r2-results-v1", "status": "evaluation_complete",
        "metric_scope": "R2 geometry only; Spearman-derived readout, not statistical R-squared",
        "prepared_only": False, "evaluation_not_run": False,
        "protocol": {"path": str(resolve_path(config["protocol_path"], Path(config_path).parent)),
                     "sha256": sha256_file(resolve_path(config["protocol_path"], Path(config_path).parent))},
        "release_path": str(release_file), "release_sha256": sha256_file(release_file),
        "reference": reference_provenance, "mask": {"condition_count": 21, "new_endpoint_included": False},
        "backend_identity": {
            "main_endpoint_backend": "GPU", "main_endpoint_count": 10, "gpu_stage_count": 30,
            "historical_cpu_anchor": "020 / C0 CPU", "historical_cpu_anchor_is_gpu_endpoint": False,
            "gpu_cpu_c0_difference_role": "non_blocking_diagnostic_only",
        },
        "representatives": representative, "rows": rows, "source_delta_row_count": len(deltas),
        "post_release_acceptance": post_acceptance,
        "plot_metadata": plot_metadata, "validation": validation,
        "selection_recomputed": False, "real_fit_started_by_evaluator": False, "native_called": False,
        "phase_payload_opened": False,
    }
    write_json(output / "evaluation_results.json", result_payload)
    readme = """# P9016 多分辨率 R2 评价

状态：`evaluation_complete`。这是经过 release gate 的 R2 几何评价；没有拟合、没有 native 调用，没有 R1/R3，也没有新的 p-value/CI。

- 10 个 GPU endpoint（5 个变体 × 2 个原始盲来源）各保留 20 条染色体记录，失败 arm 仍为 n/a；CPU 020/033 只作为独立历史锚点，不是 GPU endpoint。
- 共同 mask 复用发布的 029 21-condition mask，新 endpoint 不进入 mask；mask 溯源和 20 条距离对数在 `mask_provenance.json`。
- 主 2×2 图为五个 GPU 按 count-selected 选择的代表条件加灰色 `020 / C0 CPU` 历史锚点；按来源区分的箱线图只展示五个 GPU 条件，不把 CPU 锚点当作来源或 biorep。
- 20 个点是同一 P9016 细胞的关联测量，不是生物学重复。
- 训练选择仍是 release 中锁定的 `count_nll_per_record` / `final count_nll_normalized`，选择平局容差=1e-9 且按预注册来源顺序；R2 方向平局独立为 1e-12。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return {"status": "evaluation_complete", "output_dir": str(output), "row_count": len(rows),
            "endpoint_count": 10, "reference": reference_provenance, "plot_metadata": plot_metadata}


def validate_result_rows(rows: Sequence[Mapping[str, Any]], masks: Mapping[str, Mapping[str, Any]],
                        representatives: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_rows = len(ENDPOINT_IDS) * len(CHROMOSOMES)
    if len(rows) != expected_rows:
        raise PreparationError("full table row count mismatch: %d != %d" % (len(rows), expected_rows))
    for row in rows:
        if row.get("chromosome") not in masks:
            raise PreparationError("row has unknown chromosome")
        if row.get("n_common_pairs_frozen") != masks[row["chromosome"]]["n_common_pairs"]:
            raise PreparationError("row mask count differs from frozen mask")
    return {"schema_version": "p9016-multires-r2-validation-v1", "row_count": len(rows),
            "expected_row_count": expected_rows, "endpoint_count": 10, "chromosome_count": 20,
            "mask_counts_fixed": True, "n_a_by_metric": {metric: int(sum(not _finite(row.get(metric)) for row in rows)) for metric in METRICS},
            "representative_count": len(representatives), "no_p_values_or_ci": True}


def synthetic_check() -> dict[str, Any]:
    direct = derive_four_rho({"rho_A_mat": 0.8, "rho_A_pat": 0.2, "rho_B_mat": 0.1, "rho_B_pat": 0.7})
    swapped = derive_four_rho({"rho_A_mat": 0.1, "rho_A_pat": 0.7, "rho_B_mat": 0.8, "rho_B_pat": 0.2})
    tie = derive_four_rho({"rho_A_mat": 0.5, "rho_A_pat": 0.5, "rho_B_mat": 0.5, "rho_B_pat": 0.5})
    missing = derive_four_rho({"rho_A_mat": 0.5, "rho_A_pat": None, "rho_B_mat": 0.4, "rho_B_pat": 0.6,
                               "rho_A_pat_reason": "nonfinite_distance_vector"})
    checks = {
        "direct_best_swap_metrics": direct["orientation"] == "direct" and np.isclose(direct["matched"], 0.75) and np.isclose(direct["contrast"], 0.6),
        "candidate_label_exchange_invariant": all(np.isclose(direct[key], swapped[key]) for key in ("matched", "cross", "contrast", "minmargin")),
        "orientation_tie_is_unresolved": tie["orientation"] == "unresolved_tie" and tie["contrast"] == 0.0 and tie["margin_mat"] is None and not tie["both_positive"],
        "missing_copy_is_na": missing["metric_status"] == "unavailable" and missing["matched"] is None,
        "selection_tolerance_distinct_from_orientation": SELECTION_TIE_TOL_PER_RECORD > ORIENTATION_TIE_TOL,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise AssertionError("synthetic checks failed: %s" % ", ".join(failed))
    return {"schema_version": "p9016-multires-r2-synthetic-check-v1", "status": "PASS", "checks": checks,
            "orientation_tie_tolerance": ORIENTATION_TIE_TOL, "selection_tie_tolerance_per_record": SELECTION_TIE_TOL_PER_RECORD}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepared multiresolution P9016 R2 evaluator")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-preparation", action="store_true")
    modes.add_argument("--synthetic-check", action="store_true")
    modes.add_argument("--evaluate-released", action="store_true")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--release-manifest")
    parser.add_argument("--output-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate_preparation:
            print(json.dumps(_jsonable(validate_preparation(args.config)), indent=2, sort_keys=True))
            return 0
        if args.synthetic_check:
            print(json.dumps(_jsonable(synthetic_check()), indent=2, sort_keys=True))
            return 0
        if not args.release_manifest or not args.output_dir:
            raise PreparationError("--release-manifest and --output-dir are required for --evaluate-released")
        result = evaluate_released(args.config, args.release_manifest, args.output_dir)
        print(json.dumps(_jsonable({"status": result["status"], "output_dir": result["output_dir"], "row_count": result["row_count"]}), sort_keys=True))
        return 0
    except (PreparationError, AssertionError, ValueError, OSError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
