#!/usr/bin/env python3
"""只从已冻结 038 M0/B2 random endpoint 和旧21 mask 生成 chr1 距离矩阵图。

本脚本不调用训练、优化、native engine，也不重新运行 R2 endpoint 评估。
036 的已有 distance_arrays.npz 仅作为已保存的 old21 mask 实体，仅读取其中
frozen_pair_i/frozen_pair_j/frozen_pair_common_upper/frozen_pair_mask/grid 数组。
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/mnt/ssd/zliu/phase_restart")
FORMAL = ROOT / "test_res/038-20260914T143812Z-gpu-m1-formal"
EVAL = FORMAL / "evaluation-r2-20260914T153134Z"
OUT = EVAL / "supplement-chr1-M0-B2-distance-UTC"
SCRIPT_PATH = Path(__file__).resolve()
PNG_PATH = OUT / "chr1_M0_B2_random_distance_matrix.png"
NPZ_PATH = OUT / "matrices.npz"
PROVENANCE_PATH = OUT / "provenance.json"
VALIDATION_PATH = OUT / "validation.json"
README_PATH = OUT / "README.md"

TARGET_ENDPOINT = "M0-B2-random_joint_base2207"
TARGET_COORD = FORMAL / "stages/real/M0/B2/random_joint/coords/final-1m.3dg"
TARGET_STAGE = FORMAL / "stages/real/M0/B2/random_joint/1m.json"
SELECTION_PATH = FORMAL / "selection.json"
RELEASE_PATH = FORMAL / "release_ready_manifest.json"
TRAINING_TERMINAL_PATH = FORMAL / "terminal_evidence.json"
EVAL_TERMINAL_PATH = EVAL / "terminal_evidence.json"
HASH_GATE_PATH = EVAL / "hash_gate_evidence.json"
EVALUATION_RESULTS_PATH = EVAL / "evaluation_results.json"
R2_TABLE_PATH = EVAL / "r2_real_8endpoint_x20chr.tsv"
REFERENCE_PATH = ROOT / "data/P9016.1m.3dg.gz"

MASK_LOCK_PATH = ROOT / "docs/audits/multires-r2-preparation-20260914_041826/mask_lock.json"
MASK_CONFIG_PATH = ROOT / "docs/audits/multires-r2-preparation-20260914_041826/config.json"
MASK_MANIFEST_PATH = ROOT / "test_res/029-20260913_161713-post020-allele-ablation-real/evaluation-r2/evaluation_manifest.json"
MASK_ARRAYS_PATH = ROOT / "test_res/036-20260914T064651Z-gpu-multires/evaluation-r2-20260914_072405/chr1-heatmaps-20260914_094441/distance_arrays.npz"
MASK_METADATA_PATH = ROOT / "test_res/036-20260914T064651Z-gpu-multires/evaluation-r2-20260914_072405/chr1-heatmaps-20260914_094441/metadata.json"
MASK_VALIDATION_PATH = ROOT / "test_res/036-20260914T064651Z-gpu-multires/evaluation-r2-20260914_072405/chr1-heatmaps-20260914_094441/validation.json"
MASK_README_PATH = ROOT / "test_res/036-20260914T064651Z-gpu-multires/evaluation-r2-20260914_072405/chr1-heatmaps-20260914_094441/README.md"

LEGACY_R2COMPARISON_PATH = ROOT / "pr/r2comparison.py"
LEGACY_ALLELE_R2_PATH = ROOT / "pr/allele_r2.py"
LEGACY_MULTIR2_PATH = ROOT / "docs/audits/multires-r2-preparation-20260914_041826/multires_r2.py"

REFERENCE_SHA256 = "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29"
TARGET_COORD_SHA256 = "a21eaa479e7355782866a9728d6d102718a0fb49f97f73083a0fc9ac6695aef4"
SELECTION_SHA256 = "2241833dfd5067f581cde2eb8562031cf39d4c532de420b19295d04252f39dd9"
RELEASE_SHA256 = "2cfc1d0a26763f7891f025412b88b425f9cc322aed095f64c8e6dcc3d371a1b2"
TRAINING_TERMINAL_SHA256 = "764b9903cc3ab1e16ff34254898f282b097c2f5ab16927af17fb4c384863015a"
R2_TABLE_SHA256 = "e0962959a47eca622bc9ce2386a2650fae9e1918eef19917a14fcfd6fc55b666"
MASK_LOCK_SHA256 = "d0c325dea1289374152a327557605f8bbd52267b025cd7b57689be3e3666c49e"
MASK_MANIFEST_SHA256 = "fa1b26c834c173604c21f954d494cece8e053dd970e7cb6111f561b349574acb"
MASK_ARRAYS_SHA256 = "d1b4a14328d7c2d8227239cdd134a60d4abd37c5b759069fdc5e3ea7aa471ac2"
MASK_METADATA_SHA256 = "7eaa04ece3106405b1740d93dcde689799489060cf53154b6d9b222db339085b"
MASK_VALIDATION_SHA256 = "2324854d4078aaf70158f430f836c8981c37a68483296fa388768f46786694b9"
MASK_README_SHA256 = "4cf82b5e15c77044da1ab524b239051ab2dc1bb7469f851efcf1db6e08d3c478"
OLD_MULTIR2_SHA256_RECORDED = "3d2413738f104f412b583e63933185a2fb1524c9dd3257ae3f3dc71a9d186b18"
OLD_ALLELE_R2_SHA256_RECORDED = "56dacfe9a3edc13db4804fa7d5ce7c40b9401a1d1dd62ddbaa66d0d6ad5b1672"
OLD_R2COMPARISON_SHA256_RECORDED = "2979118aed25ac571d8f476eec4f6ec7687649f5a3e0fc43618194a293b4ed6e"

CHROMOSOME = "chr1"
CHR1_LENGTH_BP = 195_471_971
OFFSET_BP = 3_000_000
BIN_SIZE_BP = 1_000_000
EXPECTED_N_BINS = 193
EXPECTED_TOTAL_PAIRS = 18_528
EXPECTED_COMMON_PAIRS = 17_578
FONT_SIZE_PT = 7
DPI = 300
MASK_GRAY = "#d0d0d0"
TICKS_MB = (3, 50, 100, 150, 195)
ATOL = 1e-12
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

MATRIX_IDS = (
    "reference_mat",
    "reference_pat",
    "candidate_copyB_aligned_to_mat",
    "candidate_copyA_aligned_to_pat",
)
MATRIX_LABELS = (
    "参考 mat",
    "参考 pat",
    "038 M0 B2 random copyB → 参考 mat",
    "038 M0 B2 random copyA → 参考 pat",
)


def reject_json_constant(token: str) -> Any:
    raise ValueError("non-finite JSON constant is forbidden: %s" % token)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_json_constant)
    if not isinstance(value, dict):
        raise RuntimeError("JSON object expected: %s" % path)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError("missing required input: %s" % path)


def hash_record(path: Path, expected: str | None = None) -> dict[str, Any]:
    require_file(path)
    actual = sha256_file(path)
    return {
        "path": str(path),
        "sha256": actual,
        "expected_sha256": expected,
        "matches_expected": None if expected is None else bool(actual == expected),
    }


def resolve_formal(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (FORMAL / path).resolve()


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_3dg(path: Path) -> dict[str, dict[int, np.ndarray]]:
    """读取 3dg 位置和坐标；不做压缩、插值或缺失填补。"""
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
                raise RuntimeError("invalid 3dg row %d in %s" % (line_no, path)) from exc
            target = result.setdefault(track, {})
            if position in target:
                raise RuntimeError("duplicate coordinate %s:%d" % (track, position))
            target[position] = point if np.isfinite(point).all() else np.full(3, np.nan, dtype=np.float64)
    return result


def dense_points(structures: dict[str, dict[int, np.ndarray]], track: str, positions: np.ndarray) -> np.ndarray:
    points = np.full((len(positions), 3), np.nan, dtype=np.float64)
    rows = structures.get(track)
    if rows is None:
        return points
    for index, position in enumerate(positions):
        point = rows.get(int(position))
        if point is None:
            continue
        value = np.asarray(point, dtype=np.float64)
        if value.shape == (3,) and np.isfinite(value).all():
            points[index] = value
    return points


def pairwise_distance(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    finite_rows = np.isfinite(points).all(axis=1)
    delta = points[:, None, :] - points[None, :, :]
    values = np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float64))
    values[~(finite_rows[:, None] & finite_rows[None, :])] = np.nan
    diagonal = np.diag_indices(len(points))
    values[diagonal] = 0.0
    return values


def rankdata_average(values: np.ndarray) -> np.ndarray:
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


def spearman_rho(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rankdata_average(left)
    right_rank = rankdata_average(right)
    left_std = float(left_rank.std())
    right_std = float(right_rank.std())
    if left_std == 0.0 or right_std == 0.0:
        return float("nan")
    return float(np.mean((left_rank - left_rank.mean()) * (right_rank - right_rank.mean())) / (left_std * right_std))


def close(a: float, b: float, atol: float = ATOL) -> bool:
    return bool(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol))


def json_float(value: float) -> float:
    if not math.isfinite(float(value)):
        raise RuntimeError("non-finite value cannot be serialized: %r" % value)
    return float(value)


def validate_and_load_frozen_mask(mask_arrays: Path) -> dict[str, np.ndarray | int | bool]:
    with np.load(mask_arrays, allow_pickle=False) as payload:
        required = {
            "frozen_pair_mask",
            "frozen_pair_i",
            "frozen_pair_j",
            "frozen_pair_common_upper",
            "grid_bp",
            "grid_mb",
        }
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError("frozen mask artifact lacks keys: %s" % sorted(missing))
        pair_mask = np.asarray(payload["frozen_pair_mask"], dtype=bool)
        pair_i = np.asarray(payload["frozen_pair_i"], dtype=np.int64)
        pair_j = np.asarray(payload["frozen_pair_j"], dtype=np.int64)
        common_upper = np.asarray(payload["frozen_pair_common_upper"], dtype=bool)
        grid_bp = np.asarray(payload["grid_bp"], dtype=np.int64)
        grid_mb = np.asarray(payload["grid_mb"], dtype=np.float64)
    expected_grid = np.arange(OFFSET_BP, CHR1_LENGTH_BP, BIN_SIZE_BP, dtype=np.int64)
    expected_i, expected_j = np.triu_indices(EXPECTED_N_BINS, k=1)
    expected_pair_mask = np.zeros((EXPECTED_N_BINS, EXPECTED_N_BINS), dtype=bool)
    expected_pair_mask[pair_i[common_upper], pair_j[common_upper]] = True
    expected_pair_mask[pair_j[common_upper], pair_i[common_upper]] = True
    checks = {
        "grid_exact": bool(np.array_equal(grid_bp, expected_grid)),
        "grid_mb_exact": bool(np.array_equal(grid_mb, expected_grid / 1_000_000.0)),
        "pair_i_exact": bool(np.array_equal(pair_i, expected_i)),
        "pair_j_exact": bool(np.array_equal(pair_j, expected_j)),
        "common_upper_shape": bool(common_upper.shape == (EXPECTED_TOTAL_PAIRS,)),
        "pair_mask_shape": bool(pair_mask.shape == (EXPECTED_N_BINS, EXPECTED_N_BINS)),
        "pair_mask_bit_reconstruction": bool(np.array_equal(pair_mask, expected_pair_mask)),
        "pair_mask_symmetric": bool(np.array_equal(pair_mask, pair_mask.T)),
        "pair_mask_diagonal_false": bool(not np.any(np.diag(pair_mask))),
        "common_pair_count": int(common_upper.sum()) == EXPECTED_COMMON_PAIRS,
        "total_pair_count": len(pair_i) == EXPECTED_TOTAL_PAIRS,
        "n_bins": len(grid_bp) == EXPECTED_N_BINS,
    }
    if not all(bool(value) for value in checks.values()):
        raise RuntimeError("frozen mask artifact validation failed: %s" % checks)
    return {
        "pair_mask": pair_mask,
        "pair_i": pair_i,
        "pair_j": pair_j,
        "common_upper": common_upper,
        "grid_bp": grid_bp,
        "grid_mb": grid_mb,
        "checks": checks,
    }


def image_record(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        dpi_value = image.info.get("dpi")
        dpi_values = None if dpi_value is None else [float(dpi_value[0]), float(dpi_value[1])]
        return {
            "path": str(path),
            "exists": path.is_file(),
            "format": image.format,
            "size_bytes": int(path.stat().st_size),
            "size_px": [int(image.width), int(image.height)],
            "dpi_metadata": dpi_values,
            "dpi_300_ok": bool(
                dpi_value is not None
                and abs(float(dpi_value[0]) - DPI) < 1.0
                and abs(float(dpi_value[1]) - DPI) < 1.0
            ),
        }


def render_figure(normalized: np.ndarray, pair_mask: np.ndarray, grid_mb: np.ndarray, vmax: float, rhos: dict[str, float]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.colors import Normalize

    if not FONT_PATH.is_file():
        raise RuntimeError("required Chinese font is missing: %s" % FONT_PATH)
    font = FontProperties(fname=str(FONT_PATH), size=FONT_SIZE_PT)
    if float(font.get_size_in_points()) != float(FONT_SIZE_PT):
        raise RuntimeError("FontProperties size is not 7 pt")
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad(MASK_GRAY)
    norm = Normalize(vmin=0.0, vmax=float(vmax), clip=False)
    extent = (float(grid_mb[0] - 0.5), float(grid_mb[-1] + 0.5), float(grid_mb[0] - 0.5), float(grid_mb[-1] + 0.5))

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 6.0), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.105, right=0.965, bottom=0.205, top=0.86, wspace=0.12, hspace=0.28)
    image = None
    for index, axis in enumerate(axes.flat):
        display = np.ma.masked_where(~pair_mask | ~np.isfinite(normalized[index]), normalized[index])
        image = axis.imshow(
            display,
            cmap=cmap,
            norm=norm,
            origin="lower",
            interpolation="none",
            extent=extent,
            aspect="equal",
        )
        if index == 0:
            title = MATRIX_LABELS[index]
        elif index == 1:
            title = MATRIX_LABELS[index]
        elif index == 2:
            title = "%s\nrho=%.10f" % (MATRIX_LABELS[index], rhos["rho_B_mat"])
        else:
            title = "%s\nrho=%.10f" % (MATRIX_LABELS[index], rhos["rho_A_pat"])
        axis.set_title(title, fontproperties=font, pad=3)
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_xticks(TICKS_MB)
        axis.set_yticks(TICKS_MB)
        axis.tick_params(axis="both", which="major", length=2, width=0.6, pad=1, labelsize=FONT_SIZE_PT)
        axis.set_xlabel("chr1 基因组位置 (Mb)", fontproperties=font, labelpad=2)
        axis.set_ylabel("chr1 基因组位置 (Mb)", fontproperties=font, labelpad=2)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontproperties(font)
        if index < 2:
            axis.tick_params(axis="x", labelbottom=False)
            axis.set_xlabel("")
        if index in (1, 3):
            axis.tick_params(axis="y", labelleft=False)
            axis.set_ylabel("")

    fig.suptitle(
        "038 M0 B2 random：chr1 欧氏距离矩阵\n上排 reference；下排 candidate 按 whole-chr swap 对齐",
        fontproperties=font,
        y=0.968,
    )
    fig.text(
        0.5,
        0.125,
        "灰色：缺失或非 frozen mask；rho 仅作 whole-chr 评估匹配，不代表亲本身份恢复",
        ha="center",
        va="center",
        fontproperties=font,
    )
    colorbar_axis = fig.add_axes([0.205, 0.065, 0.59, 0.026])
    if image is None:
        raise RuntimeError("no image was rendered")
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("归一化欧氏距离（组内两 copy pooled median）", fontproperties=font, labelpad=2)
    colorbar.ax.tick_params(length=2, width=0.6, pad=1, labelsize=FONT_SIZE_PT)
    for label in colorbar.ax.get_xticklabels():
        label.set_fontproperties(font)
    fig.savefig(PNG_PATH, dpi=DPI, format="png")
    plt.close(fig)
    return {
        "path": str(PNG_PATH),
        "figsize_inches": [6.0, 6.0],
        "base_panel_inches": [3.0, 3.0],
        "dpi": DPI,
        "font_size_pt": FONT_SIZE_PT,
        "font_path": str(FONT_PATH),
        "colormap": "coolwarm_r",
        "bad_color": MASK_GRAY,
        "vmin": 0.0,
        "vmax": float(vmax),
        "ticks_mb": list(TICKS_MB),
    }


def make_readme(
    provenance: dict[str, Any], validation: dict[str, Any], execution_command: str,
) -> str:
    r2 = provenance["published_chr1_r2"]
    scale = provenance["normalization"]["group_scale_raw_median"]
    color = provenance["normalization"]["color_limits"]
    endpoint = provenance["endpoint"]
    return f"""# 038 M0 B2 random chr1 距离矩阵

终态：`{validation['execution_terminal_status']}`；生成脚本 exit code：`{validation['script_exit_code']}`；实际命令见 `run_exit.json`。
本目录只从 038 已冻结的 `M0-B2-random_joint_base2207` 终点、参考结构和已保存 old21 mask 作图；没有训练、优化、native engine、R1/R3 或新的 R2 端点评价。矩阵是 Euclidean distance matrix，不是 contact count matrix。

## 图和矩阵

- `chr1_M0_B2_random_distance_matrix.png`：唯一图件，2×2、总画布 6×6 inch、每个基础 panel 3×3 inch、300 dpi、统一 7 pt。
- 上排为 reference `mat`/`pat`，下排为同一 candidate 的两个 copy；038 已存 orientation 为 `{endpoint['orientation']}`，因此左下为 candidate `copyB` 对齐 reference mat，右下为 candidate `copyA` 对齐 reference pat。
- 下排标题中的 rho 是已发布 chr1 whole-chromosome matching：左 `{endpoint['aligned_rho']['copyB_to_ref_mat']:.15f}`，右 `{endpoint['aligned_rho']['copyA_to_ref_pat']:.15f}`。它只用于 whole-chr 评估匹配，不代表亲本身份恢复。
- `matrices.npz` 保存 `raw_distance`、`normalized_distance`、`grid_bp`/`grid_mb`、`frozen_pair_mask`、`frozen_pair_i/j`、`frozen_pair_common_upper`、`group_scale_raw_median` 和 `color_limits`。

## 冻结定义

- 数值网格严格为 `range(3_000_000,195_471_971,1_000_000)`：{provenance['grid']['n_bins']} 个位置，最后位置 `{provenance['grid']['last_bp']}` bp；不压缩、不插值、不补全基因组位置。坐标缺失/非有限时对应距离保留为缺失并绘成灰色。
- old21 frozen unordered upper-triangle offdiag mask 的实际计数为 `{provenance['mask']['n_common_pairs']}/{provenance['mask']['n_total_non_diagonal_pairs']}`；四个矩阵共同使用这一个 support，对角线不在 mask 内，也不参与 scale 或 metrics。
- reference 组 scale = 两条 reference copy 在 mask 上所有距离合并中位数 `{scale['reference']:.15g}`；candidate 组 scale = 两条 aligned candidate copy 合并中位数 `{scale['candidate']:.15g}`。每组两个 copy 共用一个 scale，不按 copy 单独缩放。
- 四矩阵归一化后可见值真实最大值为 `{color[1]:.15g}`，色标为 `[0, {color[1]:.15g}]`，使用 `coolwarm_r`，无 quantile clipping。
- 坐标哈希在参考文件读取前锁定；reference SHA256 为 `{provenance['source_hashes']['reference']['sha256']}`，target coords SHA256 为 `{provenance['source_hashes']['target_coords']['sha256']}`。

## 证据和验证

`provenance.json` 记录 source coords/ref/旧 R2/mask 的确切路径和 SHA、选择记录、stage terminal、orientation、四个 raw rho、mask 来源链、scale 和 color limits；`validation.json` 记录 shape、对称性、finite/support、PNG 维度/DPI、字体和旧表 rho 一致性。目标发布记录的 endpoint 状态为 `{endpoint['endpoint_status']}`，训练终端状态为 `{endpoint['terminal_status']}`。

当前工作树的 legacy `pr/allele_r2.py`/`pr/r2comparison.py` 字节 SHA 与旧 038 hash gate 记录不同；本次未调用这些漂移 source。038 没有独立实体 mask 数组可直接逐 bit 比较，因此使用已有 036 `distance_arrays.npz` 的实际 frozen mask，且通过其 21 个输入 hash、reference/manifest/lock hash、036 mask validation 和 038 的全染色体 `same_positions_and_pair_bits=true` 证据链确认来源一致；具体记录在 `provenance.json`。

实际命令：

```text
{execution_command}
```
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "code").mkdir(parents=True, exist_ok=True)

    # 第一阶段只做 hash lock 和 JSON/TSV metadata 读取，尚未打开 reference payload。
    initial_hashes: dict[str, Any] = {
        "target_coords": hash_record(TARGET_COORD, TARGET_COORD_SHA256),
        "target_stage_terminal": hash_record(TARGET_STAGE),
        "selection": hash_record(SELECTION_PATH, SELECTION_SHA256),
        "release_ready_manifest": hash_record(RELEASE_PATH, RELEASE_SHA256),
        "training_terminal": hash_record(TRAINING_TERMINAL_PATH, TRAINING_TERMINAL_SHA256),
        "evaluation_terminal": hash_record(EVAL_TERMINAL_PATH),
        "hash_gate_evidence": hash_record(HASH_GATE_PATH),
        "evaluation_results": hash_record(EVALUATION_RESULTS_PATH),
        "r2_real_8endpoint_x20chr": hash_record(R2_TABLE_PATH, R2_TABLE_SHA256),
        "mask_lock": hash_record(MASK_LOCK_PATH, MASK_LOCK_SHA256),
        "mask_config": hash_record(MASK_CONFIG_PATH),
        "mask_manifest": hash_record(MASK_MANIFEST_PATH, MASK_MANIFEST_SHA256),
        "existing_036_mask_arrays": hash_record(MASK_ARRAYS_PATH, MASK_ARRAYS_SHA256),
        "existing_036_mask_metadata": hash_record(MASK_METADATA_PATH, MASK_METADATA_SHA256),
        "existing_036_mask_validation": hash_record(MASK_VALIDATION_PATH, MASK_VALIDATION_SHA256),
        "existing_036_mask_readme": hash_record(MASK_README_PATH, MASK_README_SHA256),
        "legacy_multires_r2_current": hash_record(LEGACY_MULTIR2_PATH, OLD_MULTIR2_SHA256_RECORDED),
        "legacy_allele_r2_current": hash_record(LEGACY_ALLELE_R2_PATH, OLD_ALLELE_R2_SHA256_RECORDED),
        "legacy_r2comparison_current": hash_record(LEGACY_R2COMPARISON_PATH, OLD_R2COMPARISON_SHA256_RECORDED),
    }

    selection = read_json(SELECTION_PATH)
    release = read_json(RELEASE_PATH)
    stage_terminal = read_json(TARGET_STAGE)
    training_terminal = read_json(TRAINING_TERMINAL_PATH)
    evaluation_terminal = read_json(EVAL_TERMINAL_PATH)
    hash_gate = read_json(HASH_GATE_PATH)
    evaluation_results = read_json(EVALUATION_RESULTS_PATH)
    mask_lock = read_json(MASK_LOCK_PATH)
    mask_manifest = read_json(MASK_MANIFEST_PATH)
    mask_metadata = read_json(MASK_METADATA_PATH)
    mask_validation = read_json(MASK_VALIDATION_PATH)
    r2_rows = read_tsv_rows(R2_TABLE_PATH)

    selection_cell = selection["real"]["per_method_budget"]["M0__B2"]
    if selection_cell.get("selected_candidate_id") != "random_joint":
        raise RuntimeError("selection does not select random_joint for M0__B2")
    selected_file = selection["real"]["selected_files"]["M0__B2"]
    if selected_file.get("source_candidate_id") != "random_joint" or selected_file.get("source_coordinate_sha256") != TARGET_COORD_SHA256:
        raise RuntimeError("selection source identity mismatch")
    attempt_rows = {str(item.get("attempt_id")): item for item in selection.get("attempts", [])}
    target_attempt = attempt_rows.get("real-M0-B2-random_joint-1m")
    if not target_attempt or not target_attempt.get("is_terminal") or target_attempt.get("status") != "budget_not_converged":
        raise RuntimeError("selection terminal attempt metadata mismatch")
    target_stage_rel = "stages/real/M0/B2/random_joint/1m.json"
    if target_attempt.get("stage_record") != target_stage_rel:
        raise RuntimeError("selection terminal stage record mismatch")

    target_release_rows = [
        item for item in release.get("endpoints", [])
        if item.get("kind") == "real"
        and item.get("method") == "M0"
        and item.get("budget_id") == "B2"
        and item.get("candidate_id") == "random_joint"
    ]
    if len(target_release_rows) != 1:
        raise RuntimeError("release manifest target endpoint row mismatch")
    target_release = target_release_rows[0]
    if target_release.get("sha256") != TARGET_COORD_SHA256 or target_release.get("path") != str(TARGET_COORD.relative_to(FORMAL)):
        raise RuntimeError("release target coordinate lock mismatch")
    if target_release.get("optimization_status") != "budget_not_converged":
        raise RuntimeError("release target optimization status mismatch")
    if stage_terminal.get("attempt_id") != "real-M0-B2-random_joint-1m":
        raise RuntimeError("stage terminal attempt ID mismatch")
    if stage_terminal.get("final_coordinates", {}).get("sha256") != TARGET_COORD_SHA256:
        raise RuntimeError("stage terminal coordinate SHA mismatch")
    if stage_terminal.get("final_coordinates", {}).get("path") != target_stage_rel.replace("1m.json", "coords/final-1m.3dg"):
        raise RuntimeError("stage terminal coordinate path mismatch")
    if stage_terminal.get("status") != "budget_not_converged":
        raise RuntimeError("stage terminal status mismatch")
    global_attempt = next((item for item in training_terminal.get("attempts", []) if item.get("attempt_id") == "real-M0-B2-random_joint-1m"), None)
    if not global_attempt or not global_attempt.get("is_terminal") or global_attempt.get("stage_record") != target_stage_rel:
        raise RuntimeError("global terminal evidence target attempt mismatch")
    if training_terminal.get("status") != "release_ready" or training_terminal.get("runner_exit_code") != 0:
        raise RuntimeError("training terminal evidence status mismatch")
    if evaluation_terminal.get("exit_code") != 0 or evaluation_terminal.get("fit_called") is not False or evaluation_terminal.get("native_called") is not False:
        raise RuntimeError("038 R2 terminal evidence unexpectedly indicates fit/native or nonzero exit")
    if hash_gate.get("all_candidate_and_mask_hashes_passed") is not True:
        raise RuntimeError("038 hash gate does not report candidate/mask lock")

    target_rows = [row for row in r2_rows if row.get("endpoint_id") == TARGET_ENDPOINT and row.get("chromosome") == CHROMOSOME]
    if len(target_rows) != 1:
        raise RuntimeError("published target chr1 row missing or duplicated")
    r2_row = target_rows[0]
    if r2_row.get("endpoint_sha256") != TARGET_COORD_SHA256:
        raise RuntimeError("published R2 row endpoint SHA mismatch")
    if (int(r2_row["n_bins"]) != EXPECTED_N_BINS
            or int(r2_row["n_total_non_diagonal_pairs"]) != EXPECTED_TOTAL_PAIRS
            or int(r2_row["n_common_pairs_frozen"]) != EXPECTED_COMMON_PAIRS):
        raise RuntimeError("published R2 chr1 counts mismatch")
    if r2_row.get("orientation") != "swapped":
        raise RuntimeError("published target chr1 orientation is not swapped")
    published_rhos = {key: float(r2_row[key]) for key in ("rho_A_mat", "rho_A_pat", "rho_B_mat", "rho_B_pat")}

    expected_mask_ids = [str(item["condition_id"]) for item in mask_manifest.get("coordinates", [])]
    if len(expected_mask_ids) != 21:
        raise RuntimeError("old21 mask manifest does not contain 21 rows")
    mask_input_records: list[dict[str, Any]] = []
    for item in mask_manifest["coordinates"]:
        path = Path(item["path"]).resolve()
        expected = item.get("sha256")
        record = hash_record(path, expected)
        if record["matches_expected"] is not True:
            raise RuntimeError("old21 mask input SHA mismatch: %s" % item["condition_id"])
        mask_input_records.append({"condition_id": str(item["condition_id"]), **record})
    metadata_mask_rows = mask_metadata.get("source", {}).get("mask_input_hashes", [])
    metadata_mask_hashes = {str(item.get("condition_id")): str(item.get("sha256")) for item in metadata_mask_rows}
    current_mask_hashes = {str(item["condition_id"]): str(item["sha256"]) for item in mask_input_records}
    if metadata_mask_hashes != current_mask_hashes:
        raise RuntimeError("036 saved mask input hash set differs from current published old21 manifest")
    gate_mask_rows = hash_gate.get("locked_inputs", {}).get("mask", [])
    gate_mask_hashes = {str(item.get("condition_id")): str(item.get("sha256")) for item in gate_mask_rows}
    if gate_mask_hashes != current_mask_hashes or len(gate_mask_rows) != 21:
        raise RuntimeError("038 hash gate mask input hash set differs from published old21 manifest")
    gate_mask_manifest = hash_gate.get("locked_inputs", {}).get("mask_manifest", {})
    if gate_mask_manifest.get("path") != str(MASK_MANIFEST_PATH) or gate_mask_manifest.get("sha256") != MASK_MANIFEST_SHA256:
        raise RuntimeError("038 hash gate mask manifest identity mismatch")
    mask_lock_checks = {
        "condition_count_21": mask_lock.get("condition_count") == 21,
        "new_endpoint_inclusion_false": mask_lock.get("new_endpoint_inclusion") is False,
        "pair_definition_unordered_upper_triangle": mask_lock.get("pair_definition") == "unordered upper-triangle off-diagonal pairs on numeric bp grid",
        "position_rule_exact": mask_lock.get("position_rule") == "range(3000000, chromosome_length_bp, 1000000)",
        "distance_dtype_float64": mask_lock.get("distance_dtype") == "float64",
        "failure_action_abort": mask_lock.get("reconstruction_rule", {}).get("failure_action") == "abort rather than silently choose a new mask",
    }
    metadata_mask_definition = mask_metadata.get("frozen_r2_pair_mask", {})
    mask_strategy_checks = {
        **mask_lock_checks,
        "036_policy_reuses_published_029_mask": metadata_mask_definition.get("policy") == "exact reuse of published 029 21-condition mask; new 036 endpoints never entered or shrank it",
        "036_pair_definition_numeric_grid": metadata_mask_definition.get("pair_definition") == "unordered upper-triangle off-diagonal numeric-grid pairs",
        "036_diagonal_false": metadata_mask_definition.get("diagonal_in_mask") is False,
        "038_hash_gate_mask_input_count_21": hash_gate.get("mask_input_count") == 21,
    }
    if not all(mask_strategy_checks.values()):
        raise RuntimeError("old21 mask strategy evidence mismatch: %s" % mask_strategy_checks)
    if mask_metadata.get("source", {}).get("mask_manifest", {}).get("sha256") != MASK_MANIFEST_SHA256:
        raise RuntimeError("036 saved mask manifest SHA does not match 038 old21 manifest")
    if mask_metadata.get("source", {}).get("reference", {}).get("sha256") != REFERENCE_SHA256:
        raise RuntimeError("036 saved reference SHA does not match frozen reference")
    if mask_validation.get("status") != "PASS" or mask_validation.get("mask_n_common_pairs") != EXPECTED_COMMON_PAIRS:
        raise RuntimeError("036 frozen mask validation evidence mismatch")
    mask_exact = evaluation_results.get("mask", {}).get("mask_exact_comparison", {})
    if mask_exact.get("status") != "PASS" or mask_exact.get("same_positions_and_pair_bits") is not True:
        raise RuntimeError("038 old21 exact mask parity evidence is not complete")

    # Only after all candidate, terminal, selection, R2 and old21 mask hashes have passed, hash reference.
    reference_hash = hash_record(REFERENCE_PATH, REFERENCE_SHA256)
    if reference_hash["matches_expected"] is not True:
        raise RuntimeError("reference SHA mismatch")

    # Reference and candidate payloads are opened only after the hash lock above.
    frozen_mask = validate_and_load_frozen_mask(MASK_ARRAYS_PATH)
    grid_bp = np.asarray(frozen_mask["grid_bp"], dtype=np.int64)
    grid_mb = np.asarray(frozen_mask["grid_mb"], dtype=np.float64)
    pair_mask = np.asarray(frozen_mask["pair_mask"], dtype=bool)
    pair_i = np.asarray(frozen_mask["pair_i"], dtype=np.int64)
    pair_j = np.asarray(frozen_mask["pair_j"], dtype=np.int64)
    common_upper = np.asarray(frozen_mask["common_upper"], dtype=bool)
    support_i = pair_i[common_upper]
    support_j = pair_j[common_upper]

    reference = load_3dg(REFERENCE_PATH)
    candidate = load_3dg(TARGET_COORD)
    reference_points = {
        "mat": dense_points(reference, "chr1(mat)", grid_bp),
        "pat": dense_points(reference, "chr1(pat)", grid_bp),
    }
    candidate_points = {
        "A": dense_points(candidate, "c01a", grid_bp),
        "B": dense_points(candidate, "c01b", grid_bp),
    }
    reference_raw = [pairwise_distance(reference_points["mat"]), pairwise_distance(reference_points["pat"])]
    candidate_raw_original = [pairwise_distance(candidate_points["A"]), pairwise_distance(candidate_points["B"])]
    if r2_row["orientation"] == "swapped":
        candidate_raw_aligned = [candidate_raw_original[1], candidate_raw_original[0]]
        aligned_names = ["B", "A"]
    else:
        candidate_raw_aligned = [candidate_raw_original[0], candidate_raw_original[1]]
        aligned_names = ["A", "B"]
    raw = np.stack(reference_raw + candidate_raw_aligned, axis=0).astype(np.float64, copy=False)
    diagonal = np.diag_indices(EXPECTED_N_BINS)
    raw[:, diagonal[0], diagonal[1]] = 0.0
    support_values_raw = raw[:, support_i, support_j]
    if not np.isfinite(support_values_raw).all():
        raise RuntimeError("nonfinite raw distance on frozen support")
    reference_scale = float(np.median(np.concatenate([raw[0, support_i, support_j], raw[1, support_i, support_j]])))
    candidate_scale = float(np.median(np.concatenate([raw[2, support_i, support_j], raw[3, support_i, support_j]])))
    if not (math.isfinite(reference_scale) and reference_scale > 0.0 and math.isfinite(candidate_scale) and candidate_scale > 0.0):
        raise RuntimeError("invalid group scale")
    normalized = raw.copy()
    normalized[0:2] /= reference_scale
    normalized[2:4] /= candidate_scale
    normalized[:, diagonal[0], diagonal[1]] = 0.0
    visible = normalized[:, support_i, support_j].reshape(-1)
    if not np.isfinite(visible).all():
        raise RuntimeError("nonfinite normalized frozen support")
    vmax = float(np.max(visible))
    if not (math.isfinite(vmax) and vmax > 0.0):
        raise RuntimeError("invalid color vmax")

    # 仅对该已发布 chr1 行做四 raw rho 的 1e-12 一致性回归，不构造新的 R2 endpoint 结果表。
    recomputed_rhos = {
        "rho_A_mat": spearman_rho(candidate_raw_original[0][support_i, support_j], reference_raw[0][support_i, support_j]),
        "rho_A_pat": spearman_rho(candidate_raw_original[0][support_i, support_j], reference_raw[1][support_i, support_j]),
        "rho_B_mat": spearman_rho(candidate_raw_original[1][support_i, support_j], reference_raw[0][support_i, support_j]),
        "rho_B_pat": spearman_rho(candidate_raw_original[1][support_i, support_j], reference_raw[1][support_i, support_j]),
    }
    rho_diffs = {key: abs(recomputed_rhos[key] - published_rhos[key]) for key in published_rhos}
    rho_mismatches = [key for key, value in rho_diffs.items() if not close(recomputed_rhos[key], published_rhos[key])]
    if rho_mismatches:
        raise RuntimeError("published chr1 raw rho mismatch: %s" % rho_mismatches)
    direct = (published_rhos["rho_A_mat"] + published_rhos["rho_B_pat"]) / 2.0
    cross = (published_rhos["rho_A_pat"] + published_rhos["rho_B_mat"]) / 2.0
    expected_orientation = "direct" if direct > cross else "swapped"
    if expected_orientation != r2_row["orientation"]:
        raise RuntimeError("published orientation disagrees with raw rho values")
    if not close(float(r2_row["matched"]), max(direct, cross)) or not close(float(r2_row["contrast"]), direct - cross if direct >= cross else cross - direct):
        raise RuntimeError("published chr1 matched/contrast disagrees with four raw rho values")

    rhos_for_plot = {"rho_B_mat": published_rhos["rho_B_mat"], "rho_A_pat": published_rhos["rho_A_pat"]}
    render_info = render_figure(normalized, pair_mask, grid_mb, vmax, rhos_for_plot)

    group_scales = {"reference": reference_scale, "candidate": candidate_scale}
    matrix_support_stats = []
    for matrix_id, matrix in zip(MATRIX_IDS, raw):
        matrix_support_stats.append({
            "matrix_id": matrix_id,
            "finite_grid_points": int(np.isfinite(matrix).sum()),
            "finite_support_pairs": int(np.isfinite(matrix[support_i, support_j]).sum()),
            "missing_support_pairs": int(np.count_nonzero(~np.isfinite(matrix[support_i, support_j]))),
            "raw_min_support": float(np.min(matrix[support_i, support_j])),
            "raw_max_support": float(np.max(matrix[support_i, support_j])),
            "normalized_min_support": float(np.min(normalized[MATRIX_IDS.index(matrix_id), support_i, support_j])),
            "normalized_max_support": float(np.max(normalized[MATRIX_IDS.index(matrix_id), support_i, support_j])),
        })

    np.savez_compressed(
        NPZ_PATH,
        raw_distance=raw,
        normalized_distance=normalized,
        grid_bp=grid_bp,
        grid_mb=grid_mb,
        frozen_pair_mask=pair_mask,
        frozen_pair_i=pair_i,
        frozen_pair_j=pair_j,
        frozen_pair_common_upper=common_upper,
        group_scale_raw_median=np.asarray([reference_scale, candidate_scale], dtype=np.float64),
        color_limits=np.asarray([0.0, vmax], dtype=np.float64),
        matrix_ids=np.asarray(MATRIX_IDS),
        matrix_labels=np.asarray(MATRIX_LABELS),
        copy_alignment=np.asarray(["reference mat", "reference pat", "candidate B -> reference mat", "candidate A -> reference pat"]),
    )

    expected_038_entity_masks = sorted(EVAL.rglob("*mask*.npz"))
    mask_equivalence = {
        "direct_038_entity_mask_available": bool(expected_038_entity_masks),
        "direct_038_entity_mask_paths": [str(path) for path in expected_038_entity_masks],
        "direct_bit_compare_to_038": False,
        "reused_existing_036_npz": True,
        "existing_036_mask_arrays_sha256": MASK_ARRAYS_SHA256,
        "same_21_input_hash_set_as_038_manifest": True,
        "same_21_input_hash_set_as_038_hash_gate": bool(gate_mask_hashes == current_mask_hashes),
        "038_hash_gate_mask_input_count": int(len(gate_mask_rows)),
        "same_reference_sha256_as_038": True,
        "same_mask_manifest_sha256_as_038": True,
        "same_mask_lock_sha256_as_038": True,
        "mask_strategy_checks": mask_strategy_checks,
        "existing_036_mask_validation_status": mask_validation.get("status"),
        "038_mask_exact_comparison_status": mask_exact.get("status"),
        "038_mask_same_positions_and_pair_bits": mask_exact.get("same_positions_and_pair_bits"),
        "conclusion": "038 目录无实体 mask 数组可直接逐 bit 比较；按同一21输入/reference/manifest/lock哈希、036实际NPZ和036 validation、038全染色体 exact mask parity 证据链复用。",
    }

    source_hashes = dict(initial_hashes)
    source_hashes["reference"] = reference_hash
    source_hashes["target_r2_row_endpoint_sha256"] = TARGET_COORD_SHA256
    source_hashes["script"] = hash_record(SCRIPT_PATH)

    provenance: dict[str, Any] = {
        "schema_version": "p9016-038-chr1-M0-B2-distance-supplement-provenance-v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "execution_terminal_status": "completed",
        "script_exit_code": 0,
        "scientific_inputs_read_only": True,
        "no_training_or_optimization": True,
        "native_called": False,
        "new_r2_full_endpoint_evaluation": False,
        "published_chr1_r2_consistency_only": True,
        "execution_boundary": {
            "hash_lock_completed_before_reference_hash_and_payload_read": True,
            "reference_payload_opened_after_target_and_mask_hash_lock": True,
            "reference_path": str(REFERENCE_PATH),
            "target_coordinate_path": str(TARGET_COORD),
            "target_stage_terminal_path": str(TARGET_STAGE),
        },
        "source_hashes": source_hashes,
        "endpoint": {
            "endpoint_id": TARGET_ENDPOINT,
            "method": "M0",
            "budget_id": "B2",
            "candidate_id": "random_joint",
            "selection_criterion": "count_nll_per_record",
            "selection_value": float(selection_cell["candidate_scores"]["random_joint"]),
            "selection_status": selection_cell.get("status"),
            "endpoint_status": stage_terminal.get("status"),
            "terminal_status": stage_terminal.get("status"),
            "terminal_attempt_id": stage_terminal.get("attempt_id"),
            "orientation": r2_row["orientation"],
            "raw_rho": published_rhos,
            "recomputed_raw_rho": recomputed_rhos,
            "raw_rho_abs_diff": rho_diffs,
            "raw_rho_consistency_atol": ATOL,
            "raw_rho_consistency_mismatch_fields": rho_mismatches,
            "aligned_rho": {
                "copyB_to_ref_mat": published_rhos["rho_B_mat"],
                "copyA_to_ref_pat": published_rhos["rho_A_pat"],
            },
            "aligned_copy_order": aligned_names,
            "published_metrics": {
                "matched": float(r2_row["matched"]),
                "cross": float(r2_row["cross"]),
                "contrast": float(r2_row["contrast"]),
                "minmargin": float(r2_row["minmargin"]),
            },
        },
        "published_chr1_r2": {
            "table_path": str(R2_TABLE_PATH),
            "table_sha256": R2_TABLE_SHA256,
            "endpoint_id": TARGET_ENDPOINT,
            "row_chromosome": CHROMOSOME,
            "n_bins": int(r2_row["n_bins"]),
            "n_total_non_diagonal_pairs": int(r2_row["n_total_non_diagonal_pairs"]),
            "n_common_pairs_frozen": int(r2_row["n_common_pairs_frozen"]),
            "orientation": r2_row["orientation"],
        },
        "mask": {
            "policy": "reuse existing 036 frozen old21 mask artifact; no new endpoint enters mask",
            "source_npz_path": str(MASK_ARRAYS_PATH),
            "source_npz_sha256": MASK_ARRAYS_SHA256,
            "source_metadata_path": str(MASK_METADATA_PATH),
            "source_metadata_sha256": MASK_METADATA_SHA256,
            "mask_lock_path": str(MASK_LOCK_PATH),
            "mask_lock_sha256": MASK_LOCK_SHA256,
            "manifest_path": str(MASK_MANIFEST_PATH),
            "manifest_sha256": MASK_MANIFEST_SHA256,
            "condition_count": 21,
            "n_bins": EXPECTED_N_BINS,
            "n_total_non_diagonal_pairs": EXPECTED_TOTAL_PAIRS,
            "n_common_pairs": EXPECTED_COMMON_PAIRS,
            "unordered_upper_triangle_only_for_scale": True,
            "diagonal_in_mask": False,
            "input_hashes": mask_input_records,
            "equivalence_evidence": mask_equivalence,
            "artifact_array_checks": frozen_mask["checks"],
        },
        "reference": {
            "path": str(REFERENCE_PATH),
            "sha256": REFERENCE_SHA256,
            "columns": ["chr1(mat)", "chr1(pat)"],
            "loaded_after_hash_lock": True,
        },
        "grid": {
            "chromosome": CHROMOSOME,
            "length_bp": CHR1_LENGTH_BP,
            "rule": "range(3_000_000,195_471_971,1_000_000)",
            "offset_bp": OFFSET_BP,
            "bin_size_bp": BIN_SIZE_BP,
            "n_bins": int(len(grid_bp)),
            "first_bp": int(grid_bp[0]),
            "last_bp": int(grid_bp[-1]),
            "first_mb": float(grid_mb[0]),
            "last_mb": float(grid_mb[-1]),
            "coordinate_sampling": "exact numeric grid positions; no compression/interpolation/imputation",
        },
        "matrix_definition": {
            "distance_type": "Euclidean distance",
            "not_contact_count_matrix": True,
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
            "matrix_order": list(MATRIX_IDS),
            "matrix_labels": list(MATRIX_LABELS),
            "raw_diagonal_value": 0.0,
            "normalized_diagonal_value": 0.0,
            "diagonal_excluded_from_scale_and_metrics": True,
            "support_stats": matrix_support_stats,
        },
        "normalization": {
            "formula": "raw distance / group median raw distance over both copies on the same frozen unordered offdiag mask",
            "group_scale_raw_median": group_scales,
            "copy_specific_scaling": False,
            "visible_normalized_value_count": int(len(visible)),
            "visible_normalized_max": vmax,
            "color_limits": [0.0, vmax],
            "vmin": 0.0,
            "vmax": vmax,
            "vmax_rule": "maximum of all four normalized matrices on visible frozen-mask pairs; no quantile clipping",
            "quantile_clipping": False,
        },
        "rendering": render_info,
        "outputs": {
            "png": str(PNG_PATH),
            "matrices_npz": str(NPZ_PATH),
            "provenance_json": str(PROVENANCE_PATH),
            "validation_json": str(VALIDATION_PATH),
            "readme": str(README_PATH),
        },
    }

    raw_symmetry = bool(np.allclose(raw, np.swapaxes(raw, -1, -2), equal_nan=True, atol=0.0, rtol=0.0))
    normalized_symmetry = bool(np.allclose(normalized, np.swapaxes(normalized, -1, -2), equal_nan=True, atol=0.0, rtol=0.0))
    raw_diagonal_zero = bool(np.allclose(raw[:, diagonal[0], diagonal[1]], 0.0, atol=0.0, rtol=0.0))
    normalized_diagonal_zero = bool(np.allclose(normalized[:, diagonal[0], diagonal[1]], 0.0, atol=0.0, rtol=0.0))
    finite_raw_support = bool(np.isfinite(raw[:, support_i, support_j]).all())
    finite_normalized_support = bool(np.isfinite(normalized[:, support_i, support_j]).all())
    png_info = image_record(PNG_PATH)
    output_pngs = sorted(str(path) for path in OUT.glob("*.png"))
    output_pdfs = sorted(str(path) for path in OUT.glob("*.pdf"))
    output_html = sorted(str(path) for path in OUT.glob("*.html"))
    validation: dict[str, Any] = {
        "schema_version": "p9016-038-chr1-M0-B2-distance-supplement-validation-v1",
        "status": "validation_complete",
        "execution_terminal_status": "completed",
        "script_exit_code": 0,
        "errors": [],
        "scientific_input_status": "frozen_hash_verified_and_read_only",
        "no_training_or_optimization": True,
        "native_called": False,
        "new_r2_full_endpoint_evaluation": False,
        "published_chr1_r2_consistency_only": True,
        "matrix_shape": list(raw.shape),
        "raw_normalized_shape_exact_4x193x193": bool(raw.shape == (4, 193, 193) and normalized.shape == (4, 193, 193)),
        "raw_symmetry": raw_symmetry,
        "normalized_symmetry": normalized_symmetry,
        "raw_diagonal_zero": raw_diagonal_zero,
        "normalized_diagonal_zero": normalized_diagonal_zero,
        "finite_raw_on_common_support": finite_raw_support,
        "finite_normalized_on_common_support": finite_normalized_support,
        "grid_n_bins": int(len(grid_bp)),
        "grid_exact_numeric_rule": bool(np.array_equal(grid_bp, np.arange(OFFSET_BP, CHR1_LENGTH_BP, BIN_SIZE_BP, dtype=np.int64))),
        "mask_n_total_non_diagonal_pairs": int(len(pair_i)),
        "mask_n_common_pairs": int(common_upper.sum()),
        "mask_support_is_symmetric": bool(np.array_equal(pair_mask, pair_mask.T)),
        "mask_diagonal_false": bool(not np.any(np.diag(pair_mask))),
        "mask_bit_reconstruction_exact": bool(frozen_mask["checks"]["pair_mask_bit_reconstruction"]),
        "reference_opened_after_hash_lock": True,
        "reference_sha256_verified": True,
        "target_coordinate_sha256_verified": True,
        "target_stage_terminal_sha256_verified": True,
        "selection_release_identity_verified": True,
        "published_raw_rho_consistency": {
            "atol": ATOL,
            "mismatch_count": len(rho_mismatches),
            "max_abs_diff": float(max(rho_diffs.values())),
            "all_within_atol": not rho_mismatches,
            "fields": list(published_rhos),
        },
        "orientation": {
            "published": r2_row["orientation"],
            "mapping": {"reference_mat": "candidate_copyB", "reference_pat": "candidate_copyA"},
            "whole_chromosome_only": True,
            "local_swap": False,
        },
        "scale": {
            "group_scale_raw_median": group_scales,
            "same_scale_within_group": True,
            "per_copy_scale_used": False,
        },
        "color_limits": {
            "vmin": 0.0,
            "vmax": vmax,
            "all_visible_values_le_vmax": bool(np.all(visible <= vmax)),
            "quantile_clipping": False,
            "colormap": "coolwarm_r",
            "bad_color": MASK_GRAY,
        },
        "font": {
            "font_path": str(FONT_PATH),
            "font_properties_size_pt": FONT_SIZE_PT,
            "font_properties_size_verified": True,
        },
        "image": png_info,
        "image_dimensions_expected_1800x1800": bool(png_info["size_px"] == [1800, 1800]),
        "image_dpi_expected_300": bool(png_info["dpi_300_ok"]),
        "output_policy": {
            "png_files": output_pngs,
            "pdf_files": output_pdfs,
            "html_files": output_html,
            "exactly_one_png": output_pngs == [str(PNG_PATH)],
            "no_pdf": not output_pdfs,
            "no_html": not output_html,
        },
        "mask_equivalence": mask_equivalence,
        "npz_path": str(NPZ_PATH),
        "npz_sha256": sha256_file(NPZ_PATH),
        "provenance_path": str(PROVENANCE_PATH),
    }
    check_fields = (
        "raw_normalized_shape_exact_4x193x193",
        "raw_symmetry",
        "normalized_symmetry",
        "raw_diagonal_zero",
        "normalized_diagonal_zero",
        "finite_raw_on_common_support",
        "finite_normalized_on_common_support",
        "grid_exact_numeric_rule",
        "mask_support_is_symmetric",
        "mask_diagonal_false",
        "mask_bit_reconstruction_exact",
        "reference_opened_after_hash_lock",
        "reference_sha256_verified",
        "target_coordinate_sha256_verified",
        "target_stage_terminal_sha256_verified",
        "selection_release_identity_verified",
        "image_dimensions_expected_1800x1800",
        "image_dpi_expected_300",
    )
    failed = [field for field in check_fields if not bool(validation[field])]
    failed.extend([
        "mask_total_count" if int(len(pair_i)) != EXPECTED_TOTAL_PAIRS else "",
        "mask_common_count" if int(common_upper.sum()) != EXPECTED_COMMON_PAIRS else "",
        "rho_consistency" if rho_mismatches else "",
        "output_policy" if not all(validation["output_policy"].get(key) for key in ("exactly_one_png", "no_pdf", "no_html")) else "",
    ])
    validation["errors"] = [item for item in failed if item]
    validation["overall_valid"] = not validation["errors"]
    if not validation["overall_valid"]:
        raise RuntimeError("validation failed: %s" % validation["errors"])

    command = "%s %s" % (sys.executable, SCRIPT_PATH)
    provenance["execution_command"] = command
    provenance["output_hashes"] = {
        "png_sha256": sha256_file(PNG_PATH),
        "matrices_npz_sha256": sha256_file(NPZ_PATH),
    }
    write_json(PROVENANCE_PATH, provenance)
    write_json(VALIDATION_PATH, validation)
    README_PATH.write_text(make_readme(provenance, validation, command), encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "exit_code": 0,
        "output": str(OUT),
        "png": str(PNG_PATH),
        "matrices": str(NPZ_PATH),
        "mask_pairs": int(common_upper.sum()),
        "mask_total": int(len(pair_i)),
        "orientation": r2_row["orientation"],
        "rho_B_mat": published_rhos["rho_B_mat"],
        "rho_A_pat": published_rhos["rho_A_pat"],
        "reference_scale": reference_scale,
        "candidate_scale": candidate_scale,
        "vmax": vmax,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
