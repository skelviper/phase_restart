#!/usr/bin/env python3
"""双 3DG 比较制图入口（精简实现）。

输入一个 candidate 3DG 和一个 reference 3DG（可 gzip），一条命令生成两张 PNG：
1. `plots/{chrom}_{N}Mb_distance_matrix.png`：2x2 距离矩阵（reference mat/pat + candidate 两条拷贝）；
2. `plots/whole_genome_1Mb_3d_scatter.png`：全 genome 两 panel 3D scatter。

固定规则：
- 默认矩阵 grid 覆盖两文件该 chromosome 的完整 numeric range，缺 bin 保持 NaN（图中灰色）；
  给出 `--mask` 时改用 mask 的 positions 与其 common bits。矩阵数值与 `--no-align` 无关。
- 两组数据各自全 finite beads 共用一个 center 和 whole-cell RMS；不做 per-copy scaling、不做视角优化。
- 方向默认 `auto`：按 direct/swapped 的 Pearson 均值择大，差值 <= 1e-12 记为 `unresolved_tie` 并按 direct 画图。
- 散点默认对齐（`--no-align` 可关）：每条 chromosome 用共同有限 position 的双拷贝内部距离四个 Pearson
  定 copy 对应（相关无法定义时记为 `unresolved` 并稳定用 direct，不丢 chromosome/点），再用全部对应点的
  display 坐标拟合**一次**全局 Kabsch 刚体旋转，只把 R 应用于 candidate display 坐标；
  det(R) = +1、不镜像、不重新缩放、不逐 chromosome/copy 旋转，whole-cell 原点不动（不加平移）。

默认输出目录是 `<repo>/scratch/plot_3dg_comparison`，一次调用只写两张 PNG 和 `metrics.json`。

科学隔离：candidate 的 SHA256 只算一次，在打开 reference 之前算出并记进 `metrics.json`；本脚本不加载训练数据、
不重跑评价、不改冻结模块，只读两份 3DG。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import rankdata

# 复用项目已有的 tab20 chromosome color 定义；该模块不读取 reference。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pr.viz3d import _chrom_colors  # noqa: E402

CHROMOSOMES = tuple([f"chr{i}" for i in range(1, 20)] + ["chrX"])
REFERENCE_COPIES = ("mat", "pat")
CANDIDATE_COPIES = ("copyA", "copyB")
TIE_TOLERANCE = 1e-12


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    """把 numpy 标量/数组转成可写 JSON 的朴素类型。"""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_3dg(path: Path) -> dict[str, dict[int, np.ndarray]]:
    """按 numeric position 读取 3DG，非有限坐标转成 NaN。"""
    tracks: dict[str, dict[int, np.ndarray]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                track = str(fields[0])
                position = int(fields[1])
                xyz = np.asarray([float(x) for x in fields[2:5]], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid 3DG row {line_no}: {path}") from exc
            if not np.isfinite(xyz).all():
                xyz = np.full(3, np.nan, dtype=np.float64)
            rows = tracks.setdefault(track, {})
            if position in rows:
                raise ValueError(f"duplicate {track}:{position} in {path}")
            rows[position] = xyz
    if not tracks:
        raise ValueError(f"no 3DG rows in {path}")
    return tracks


def candidate_track(chrom_index: int, copy_index: int) -> str:
    return f"c{chrom_index + 1:02d}{'ab'[copy_index]}"


def reference_track(chromosome: str, copy_index: int) -> str:
    return f"{chromosome}({REFERENCE_COPIES[copy_index]})"


def chromosome_positions(tracks: dict[str, dict[int, np.ndarray]], chromosome: str) -> np.ndarray:
    """该 chromosome 两条 track 出现过的全部 position，升序。"""
    ci = CHROMOSOMES.index(chromosome)
    positions: set[int] = set()
    for copy_index in range(2):
        positions.update(tracks.get(candidate_track(ci, copy_index), {}))
        positions.update(tracks.get(reference_track(chromosome, copy_index), {}))
    if not positions:
        raise ValueError(f"no 3DG positions for chromosome {chromosome}")
    return np.asarray(sorted(positions), dtype=np.int64)


def finite_positions(rows: dict[int, np.ndarray]) -> np.ndarray:
    """该 track 全部 finite 坐标的 position，升序。"""
    return np.asarray(sorted(int(p) for p, value in rows.items() if np.isfinite(value).all()), dtype=np.int64)


def points_on_grid(tracks: dict[str, dict[int, np.ndarray]], track: str, positions: np.ndarray) -> np.ndarray:
    output = np.full((len(positions), 3), np.nan, dtype=np.float64)
    rows = tracks.get(track, {})
    for index, position in enumerate(positions):
        point = rows.get(int(position))
        if point is not None and np.isfinite(point).all():
            output[index] = point
    return output


def distance_vector(points: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    delta = points[pair_i] - points[pair_j]
    return np.sqrt(np.sum(delta * delta, axis=1))


def corr(left: np.ndarray, right: np.ndarray) -> float:
    keep = np.isfinite(left) & np.isfinite(right)
    x = np.asarray(left[keep], dtype=np.float64)
    y = np.asarray(right[keep], dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    keep = np.isfinite(left) & np.isfinite(right)
    if int(keep.sum()) < 2:
        return float("nan")
    return corr(rankdata(np.asarray(left[keep]), method="average"), rankdata(np.asarray(right[keep]), method="average"))


def numeric_grid(candidate: dict[str, dict[int, np.ndarray]], reference: dict[str, dict[int, np.ndarray]], chromosome: str, resolution: int) -> np.ndarray:
    """默认 numeric full grid：覆盖两文件该 chromosome 的完整 range，保留缺 bin。"""
    positions = chromosome_positions({**candidate, **reference}, chromosome)
    start = (int(positions[0]) // resolution) * resolution
    stop = ((int(positions[-1]) + resolution - 1) // resolution) * resolution
    grid = np.arange(start, stop + resolution, resolution, dtype=np.int64)
    if len(grid) < 2:
        raise ValueError(f"numeric grid has fewer than two positions for {chromosome}")
    return grid


def load_mask(path: Path, chromosome: str, resolution: int) -> dict[str, Any]:
    """读取 frozen snapshot 的一条 chromosome mask，不调用 worker。"""
    ci = CHROMOSOMES.index(chromosome)
    prefix = f"chr{ci}_"
    with np.load(path, allow_pickle=False) as archive:
        required = {prefix + key for key in ("positions", "pair_i", "pair_j", "common")}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"mask missing fields: {sorted(missing)}")
        positions = np.asarray(archive[prefix + "positions"], dtype=np.int64).copy()
        pair_i = np.asarray(archive[prefix + "pair_i"], dtype=np.int64).copy()
        pair_j = np.asarray(archive[prefix + "pair_j"], dtype=np.int64).copy()
        common = np.asarray(archive[prefix + "common"], dtype=bool).copy()
    if len(positions) < 2 or np.any(np.diff(positions) <= 0):
        raise ValueError("mask positions must be strictly increasing")
    if np.any(positions % resolution != 0):
        raise ValueError("mask positions are not aligned to resolution")
    expected_i, expected_j = np.triu_indices(len(positions), 1)
    if not np.array_equal(pair_i, expected_i) or not np.array_equal(pair_j, expected_j) or common.shape != pair_i.shape:
        raise ValueError("mask pair arrays are not strict upper triangle")
    return {"positions": positions, "pair_i": pair_i, "pair_j": pair_j, "common": common, "source": str(path.resolve())}


def orientation_choice(raw_rhos: dict[str, float], requested: str) -> tuple[str, int, int]:
    """返回 (orientation label, reference copy index for panel 3, 4)。"""
    direct = (raw_rhos["A_mat"] + raw_rhos["B_pat"]) / 2.0
    swapped = (raw_rhos["A_pat"] + raw_rhos["B_mat"]) / 2.0
    if requested == "direct":
        return "direct", 0, 1
    if requested == "swapped":
        return "swapped", 1, 0
    if abs(direct - swapped) <= TIE_TOLERANCE:
        return "unresolved_tie", 0, 1
    return ("direct", 0, 1) if direct > swapped else ("swapped", 1, 0)


def distance_on_grid(tracks: dict[str, dict[int, np.ndarray]], track: str, positions: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    return distance_vector(points_on_grid(tracks, track, positions), pair_i, pair_j)


def build_matrix(candidate: dict[str, dict[int, np.ndarray]], reference: dict[str, dict[int, np.ndarray]], chromosome: str, resolution: int, mask: dict[str, Any] | None, requested_orientation: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ci = CHROMOSOMES.index(chromosome)
    candidate_tracks = (candidate_track(ci, 0), candidate_track(ci, 1))
    reference_tracks = (reference_track(chromosome, 0), reference_track(chromosome, 1))
    if mask is not None:
        positions = mask["positions"]
        pair_i, pair_j, common = mask["pair_i"], mask["pair_j"], mask["common"]
    else:
        positions = numeric_grid(candidate, reference, chromosome, resolution)
        pair_i, pair_j = np.triu_indices(len(positions), 1)
        finite_endpoints = np.ones(len(positions), dtype=bool)
        for tracks, track_names in ((candidate, candidate_tracks), (reference, reference_tracks)):
            for track in track_names:
                finite_endpoints &= np.isfinite(points_on_grid(tracks, track, positions)).all(axis=1)
        common = finite_endpoints[pair_i] & finite_endpoints[pair_j]
    if not common.any():
        raise ValueError(f"no valid common pair for {chromosome}; nothing to compare")
    used_i, used_j = pair_i[common], pair_j[common]
    cand_a = distance_on_grid(candidate, candidate_tracks[0], positions, used_i, used_j)
    cand_b = distance_on_grid(candidate, candidate_tracks[1], positions, used_i, used_j)
    ref_mat = distance_on_grid(reference, reference_tracks[0], positions, used_i, used_j)
    ref_pat = distance_on_grid(reference, reference_tracks[1], positions, used_i, used_j)
    raw_rhos = {
        "A_mat": corr(cand_a, ref_mat),
        "A_pat": corr(cand_a, ref_pat),
        "B_mat": corr(cand_b, ref_mat),
        "B_pat": corr(cand_b, ref_pat),
    }
    spearman_rhos = {
        "A_mat": spearman(cand_a, ref_mat),
        "A_pat": spearman(cand_a, ref_pat),
        "B_mat": spearman(cand_b, ref_mat),
        "B_pat": spearman(cand_b, ref_pat),
    }
    if not all(np.isfinite(value) for value in raw_rhos.values()):
        raise ValueError("Pearson comparison is not finite for this grid/mask")
    orientation, cand_first, cand_second = orientation_choice(raw_rhos, requested_orientation)
    direct_score = (raw_rhos["A_mat"] + raw_rhos["B_pat"]) / 2.0
    swapped_score = (raw_rhos["A_pat"] + raw_rhos["B_mat"]) / 2.0

    n = len(positions)
    raw = np.full((4, n * n), np.nan, dtype=np.float64)
    vectors = (ref_mat, ref_pat, cand_b if cand_first else cand_a, cand_b if cand_second else cand_a)
    flat_ij = used_i * n + used_j
    for panel_index, values in enumerate(vectors):
        raw[panel_index, flat_ij] = values
        raw[panel_index, used_j * n + used_i] = values
    raw = raw.reshape(4, n, n)

    # 共同尺度：reference mat+pat 与 candidate copyA+copyB 各用一个 raw median。
    scales = np.asarray([
        np.nanmedian(np.concatenate([ref_mat, ref_pat])),
        np.nanmedian(np.concatenate([cand_a, cand_b])),
    ], dtype=np.float64)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("non-positive matrix normalization scale")
    normalized = raw / scales[np.asarray([0, 0, 1, 1])][:, None, None]
    visible = normalized[np.isfinite(normalized)]
    color_limits = np.asarray([0.0, float(np.max(visible))], dtype=np.float64)
    if not np.isfinite(color_limits[1]) or color_limits[1] <= 0:
        raise ValueError("invalid matrix color limit")

    matrix = {
        "normalized_distance": normalized,
        "grid_bp": positions,
        "grid_mb": positions.astype(np.float64) / 1e6,
        "group_scale_raw_median": scales,
        "color_limits": color_limits,
        "panel_labels": (
            "Reference mat", "Reference pat",
            f"Candidate {CANDIDATE_COPIES[1] if cand_first else CANDIDATE_COPIES[0]} -> mat",
            f"Candidate {CANDIDATE_COPIES[1] if cand_second else CANDIDATE_COPIES[0]} -> pat",
        ),
        "panel_rho_key": ("B_mat" if cand_first else "A_mat", "B_pat" if cand_second else "A_pat"),
    }
    checks = {
        "computed_pearson_rho": raw_rhos,
        "computed_spearman_rho": spearman_rhos,
        "orientation": orientation,
        "direct_score": float(direct_score),
        "swapped_score": float(swapped_score),
        "n_positions": int(n),
        "n_total_pairs": int(len(pair_i)),
        "n_common_pairs": int(common.sum()),
    }
    return matrix, checks


def plot_matrix(matrix: dict[str, Any], checks: dict[str, Any], chromosome: str, out_path: Path) -> None:
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad("#bdbdbd")
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 6.0), dpi=300, squeeze=False)
    grid_mb = matrix["grid_mb"]
    bin_bp = int(np.median(np.diff(matrix["grid_bp"]))) if len(grid_mb) > 1 else 1
    bin_mb = bin_bp / 1e6
    extent = [float(grid_mb[0] - bin_mb / 2), float(grid_mb[-1] + bin_mb / 2), float(grid_mb[0] - bin_mb / 2), float(grid_mb[-1] + bin_mb / 2)]
    tick_values = np.unique(np.asarray([grid_mb[0], grid_mb[len(grid_mb) // 3], grid_mb[2 * len(grid_mb) // 3], grid_mb[-1]])) if len(grid_mb) > 5 else grid_mb
    titles = list(matrix["panel_labels"][:2])
    for key, label in zip(matrix["panel_rho_key"], matrix["panel_labels"][2:]):
        titles.append(f"{label}\nComputed {key}={checks['computed_pearson_rho'][key]:.6f}")
    image = None
    for panel_index, ax in enumerate(axes.flat):
        image = ax.imshow(np.ma.masked_invalid(matrix["normalized_distance"][panel_index]), origin="lower", extent=extent,
                          interpolation="none", aspect="equal", cmap=cmap, vmin=float(matrix["color_limits"][0]), vmax=float(matrix["color_limits"][1]))
        ax.set_title(titles[panel_index], fontsize=7, pad=4)
        ax.set_xlabel("Genomic position (Mb)", fontsize=7)
        ax.set_ylabel("Genomic position (Mb)", fontsize=7)
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.tick_params(labelsize=7, length=2, pad=1)
    cax = fig.add_axes([0.90, 0.18, 0.025, 0.66])
    cbar = fig.colorbar(image, cax=cax, aspect=28)
    cbar.set_label("Normalized Euclidean distance", fontsize=7)
    cbar.ax.tick_params(labelsize=7, length=2, pad=1)
    fig.suptitle(f"{chromosome} {bin_mb:g}-Mb distance matrices", fontsize=7, y=0.985)
    matched = max(float(checks["direct_score"]), float(checks["swapped_score"]))
    cross = min(float(checks["direct_score"]), float(checks["swapped_score"]))
    fig.text(0.5, 0.018,
             f"Matched {matched:.6f} | Cross {cross:.6f} | Contrast {matched - cross:.6f}\n"
             f"Common pairs: {checks['n_common_pairs']:,} / {checks['n_total_pairs']:,}",
             ha="center", va="bottom", fontsize=7, linespacing=1.15)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.15, top=0.90, wspace=0.25, hspace=0.34)
    fig.savefig(out_path, dpi=300, format="png")
    plt.close(fig)


def build_scatter(candidate: dict[str, dict[int, np.ndarray]], reference: dict[str, dict[int, np.ndarray]], resolution: int) -> dict[str, Any]:
    """union grid 上的两 dataset 坐标；各自中心化并除以 whole-cell RMS。

    列空间按 chromosome/copy 分块，块宽取两 dataset 的较大者；块内空位用 (chrom, copy) = -1 标记，
    对应坐标保持 NaN，既不参与中心/RMS，也不进入散点。
    """
    datasets = (reference, candidate)
    per_copy = [np.zeros((len(CHROMOSOMES), 2), dtype=np.int64) for _ in datasets]
    tracks_by_copy: list[list[list[dict[int, np.ndarray]]]] = [[[] for _ in range(2)] for _ in datasets]
    for di, tracks in enumerate(datasets):
        for ci, chromosome in enumerate(CHROMOSOMES):
            for k in range(2):
                track = candidate_track(ci, k) if di else reference_track(chromosome, k)
                rows = tracks.get(track, {})
                positions = sorted(rows)
                if positions and any(int(position) % resolution != 0 for position in positions):
                    raise ValueError(f"unaligned position in {chromosome}")
                per_copy[di][ci, k] = len(positions)
                tracks_by_copy[di][k].append(rows)
    capacities = np.maximum(per_copy[0], per_copy[1])
    slot_first = np.concatenate([[0], np.cumsum(capacities.ravel())[:-1]]).reshape(capacities.shape)
    n_grid = int(capacities.sum())
    if n_grid == 0:
        raise ValueError("candidate/reference have no grid points")

    raw = np.full((2, 2, n_grid, 3), np.nan, dtype=np.float64)
    chrom_index = np.full(n_grid, -1, dtype=np.int64)
    copy_index = np.full(n_grid, -1, dtype=np.int64)
    slot_position = np.full(n_grid, -1, dtype=np.int64)
    for di in range(2):
        for ci in range(len(CHROMOSOMES)):
            for k in range(2):
                rows = tracks_by_copy[di][k][ci]
                first = int(slot_first[ci, k])
                for step, position in enumerate(sorted(rows)):
                    slot_position[first + step] = int(position)
                    point = rows.get(int(position))
                    if point is not None and np.isfinite(point).all():
                        raw[di, k, first + step] = point
                span = int(per_copy[di][ci, k])
                chrom_index[first:first + span] = ci
                copy_index[first:first + span] = k
    finite = np.isfinite(raw).all(axis=3)
    finite &= (chrom_index[None, None, :] >= 0) & (copy_index[None, None, :] >= 0)
    if not finite.any():
        raise ValueError("scatter has no finite coordinates")
    display = np.full_like(raw, np.nan)
    rms = np.full(2, np.nan, dtype=np.float64)
    for di in range(2):
        values = raw[di][finite[di]]
        center = np.mean(values, axis=0)
        scale = float(np.sqrt(np.mean(np.sum((values - center) ** 2, axis=1))))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scatter RMS scale is invalid")
        display[di][finite[di]] = (values - center) / scale
        rms[di] = scale
    plotted = display[np.isfinite(display).all(axis=3)]
    return {
        "display_xyz": display,
        "finite": finite,
        "chrom_index": chrom_index,
        "copy_index": copy_index,
        "slot_position": slot_position,
        "total_grid_per_dataset": n_grid,
        "finite_count": finite.sum(axis=(1, 2)).astype(np.int64),
        "scatter_rms": rms,
        "axis_limits": scatter_axis_limits(plotted),
    }


def scatter_axis_limits(plotted: np.ndarray) -> np.ndarray:
    """由当前 display 坐标（旋转前后都调用）给出共用轴范围。"""
    low, high = float(np.min(plotted)), float(np.max(plotted))
    margin = max((high - low) * 0.05, 0.02)
    return np.asarray([low - margin, high + margin], dtype=np.float64)


def alignment_orientation(raw_rhos: dict[str, float]) -> tuple[str, tuple[int, int]]:
    """逐 chromosome 对齐用 copy 对应，返回 (label, 每个 candidate copy 对应的 reference copy index)。

    相关无法定义时记为 unresolved 并稳定用 direct，不丢 chromosome/点。
    """
    if not all(np.isfinite(value) for value in raw_rhos.values()):
        return "unresolved", (0, 1)
    direct = (raw_rhos["A_mat"] + raw_rhos["B_pat"]) / 2.0
    swapped = (raw_rhos["A_pat"] + raw_rhos["B_mat"]) / 2.0
    if abs(direct - swapped) <= TIE_TOLERANCE:
        return "unresolved_tie", (0, 1)
    return ("direct", (0, 1)) if direct > swapped else ("swapped", (1, 0))


def chromosome_alignment_choice(candidate: dict[str, dict[int, np.ndarray]], reference: dict[str, dict[int, np.ndarray]], chromosome: str) -> dict[str, Any]:
    """一条 chromosome 的 copy 对应：共同有限 position 上的双拷贝内部距离四个 Pearson。"""
    ci = CHROMOSOMES.index(chromosome)
    candidate_tracks = (candidate_track(ci, 0), candidate_track(ci, 1))
    reference_tracks = (reference_track(chromosome, 0), reference_track(chromosome, 1))
    owners = ((candidate, candidate_tracks[0]), (candidate, candidate_tracks[1]), (reference, reference_tracks[0]), (reference, reference_tracks[1]))
    positions: np.ndarray | None = None
    for tracks, track in owners:
        found = finite_positions(tracks.get(track, {}))
        positions = found if positions is None else np.intersect1d(positions, found)
    info: dict[str, Any] = {
        "chromosome": chromosome,
        "n_common_positions": int(len(positions)) if positions is not None else 0,
        "orientation": "unresolved",
        "copy_pairs": [[CANDIDATE_COPIES[0], REFERENCE_COPIES[0]], [CANDIDATE_COPIES[1], REFERENCE_COPIES[1]]],
        "computed_pearson_rho": None,
        "reference_copy_for_candidate_copy": [0, 1],
    }
    if positions is not None and len(positions) >= 2:
        pair_i, pair_j = np.triu_indices(len(positions), 1)
        distances = [distance_vector(points_on_grid(tracks, track, positions), pair_i, pair_j) for tracks, track in owners]
        raw_rhos = {
            "A_mat": corr(distances[0], distances[2]),
            "A_pat": corr(distances[0], distances[3]),
            "B_mat": corr(distances[1], distances[2]),
            "B_pat": corr(distances[1], distances[3]),
        }
        orientation, pair = alignment_orientation(raw_rhos)
        info.update(orientation=orientation, computed_pearson_rho=raw_rhos, reference_copy_for_candidate_copy=list(pair),
                    copy_pairs=[[CANDIDATE_COPIES[k], REFERENCE_COPIES[j]] for k, j in enumerate(pair)])
    return info


def kabsch_rotation(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """拟合把 source 映射到 target 的无缩放刚体旋转 R（det = +1，不镜像）。

    允许对拟合子集临时去质心；返回的 R 只做旋转，调用方不得再加平移。
    """
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    sign = float(np.sign(np.linalg.det(vt.T @ u.T)))
    if sign == 0.0:
        sign = 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, float(np.linalg.det(rotation))


def fit_global_rotation(candidate: dict[str, dict[int, np.ndarray]], reference: dict[str, dict[int, np.ndarray]], points: dict[str, Any]) -> dict[str, Any]:
    """20 条 chromosome 各定 copy 对应，再用全部对应点的 display 坐标拟合一次全局 Kabsch 旋转。

    每条 chromosome 的对应关系同时用两条 leg（copyA 与其 reference copy、copyB 与其 reference copy）；
    只取 display 坐标都 finite 的配对点，不做逐 chromosome/copy 旋转。
    """
    chromosomes: list[dict[str, Any]] = []
    source_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    display = points["display_xyz"]
    for ci, chromosome in enumerate(CHROMOSOMES):
        info = chromosome_alignment_choice(candidate, reference, chromosome)
        fit_points = 0
        for candidate_index, reference_index in enumerate(info.pop("reference_copy_for_candidate_copy")):
            candidate_mask = points["finite"][1, candidate_index] & (points["chrom_index"] == ci) & (points["copy_index"] == candidate_index)
            reference_mask = points["finite"][0, reference_index] & (points["chrom_index"] == ci) & (points["copy_index"] == reference_index)
            candidate_positions = points["slot_position"][candidate_mask]
            reference_positions = points["slot_position"][reference_mask]
            shared = np.intersect1d(candidate_positions, reference_positions)
            # build_scatter 在每个 (chromosome, copy) block 内按 position 升序放点，因此 searchsorted 可直接回查 slot。
            source_chunks.append(display[1, candidate_index][candidate_mask][np.searchsorted(candidate_positions, shared)])
            target_chunks.append(display[0, reference_index][reference_mask][np.searchsorted(reference_positions, shared)])
            fit_points += int(len(shared))
        info["fit_points"] = fit_points
        chromosomes.append(info)
    source = np.vstack(source_chunks)
    target = np.vstack(target_chunks)
    if len(source) < 3:
        raise ValueError(f"global alignment fit needs at least 3 common points, found {len(source)}")
    rotation, det_rotation = kabsch_rotation(source, target)
    return {
        "aligned": True,
        "method": ("per-chromosome copy correspondence by Pearson of intra-copy distances (direct/swapped mean, tie/unresolved -> direct), "
                   "then one global Kabsch rotation fitted on all corresponding display coordinates; det=+1, no mirror, no rescale, "
                   "no per-chromosome or per-copy rotation, no translation (whole-cell origin fixed)"),
        "det_rotation": det_rotation,
        "rotation_matrix": rotation,
        "total_fit_points": int(len(source)),
        "chromosomes": chromosomes,
    }


def disabled_alignment() -> dict[str, Any]:
    """--no-align：不旋转，沿用原 display 坐标（历史复现）。"""
    return {
        "aligned": False,
        "method": "disabled_by_--no-align",
        "det_rotation": 1.0,
        "rotation_matrix": np.eye(3, dtype=np.float64),
        "total_fit_points": 0,
        "chromosomes": [],
    }


def apply_global_rotation(points: dict[str, Any], rotation: np.ndarray) -> None:
    """把 R 应用于全部 candidate display 坐标；reference 不动，不加平移。"""
    candidate_display = points["display_xyz"][1]
    mask = points["finite"][1]
    candidate_display[mask] = candidate_display[mask] @ rotation.T
    display = points["display_xyz"]
    points["axis_limits"] = scatter_axis_limits(display[np.isfinite(display).all(axis=3)])


def plot_scatter(points: dict[str, Any], out_path: Path, candidate_id: str | None, resolution: int, aligned: bool) -> None:
    colors = _chrom_colors(CHROMOSOMES)
    markers = ("o", "^")
    fig = plt.figure(figsize=(8.4, 4.8), dpi=300)
    axes = (
        fig.add_axes([0.055, 0.30, 0.36, 0.625], projection="3d"),
        fig.add_axes([0.46, 0.30, 0.36, 0.625], projection="3d"),
    )
    for di, ax in enumerate(axes):
        for ci in range(len(CHROMOSOMES)):
            for k in range(2):
                # display_xyz 的第一轴是 dataset；先取该 dataset 再按 chromosome/copy/slot 布尔筛选。
                selected = points["finite"][di, k] & (points["chrom_index"] == ci) & (points["copy_index"] == k)
                if selected.any():
                    xyz = points["display_xyz"][di, k][selected]
                    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=4.0, marker=markers[k], color=colors[CHROMOSOMES[ci]],
                               alpha=0.72, linewidths=0.0, depthshade=False)
        ax.set_xlim(*points["axis_limits"])
        ax.set_ylim(*points["axis_limits"])
        ax.set_zlim(*points["axis_limits"])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-55)
        ax.set_xlabel("X", fontsize=7, labelpad=1)
        ax.set_ylabel("Y", fontsize=7, labelpad=1)
        ax.set_zlabel("Z", fontsize=7, labelpad=1)
        ax.tick_params(labelsize=7, pad=0, length=2)
    ref_n, cand_n = int(points["finite_count"][0]), int(points["finite_count"][1])
    bin_label = f"{resolution / 1e6:g}-Mb"
    axes[0].set_title(f"Reference {bin_label} coordinates\nmat/pat; finite {ref_n:,} / union grid {points['total_grid_per_dataset']:,}", fontsize=7, pad=1)
    axes[1].set_title(f"{candidate_id or 'Candidate'} {bin_label} coordinates\ncopyA/copyB; finite {cand_n:,} / union grid {points['total_grid_per_dataset']:,}", fontsize=7, pad=1)
    chr_handles = [Line2D([0], [0], marker="o", linestyle="None", markersize=3.5, color=colors[chrom], label=chrom) for chrom in CHROMOSOMES]
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=4.0, color="black", label="mat / copyA"),
        Line2D([0], [0], marker="^", linestyle="None", markersize=4.0, color="black", label="pat / copyB"),
    ]
    fig.legend(handles=chr_handles, title="Chromosome color", title_fontsize=7, fontsize=7, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, 0.13), frameon=False, handletextpad=0.3, columnspacing=0.7, labelspacing=0.2)
    fig.legend(handles=marker_handles, title="Copy marker", title_fontsize=7, fontsize=7, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.035), frameon=False, handletextpad=0.3, columnspacing=0.9, labelspacing=0.2)
    fig.suptitle(f"Whole-genome {bin_label} 3D scatter (centered / unit-RMS{'; Global rigid alignment' if aligned else ''})", fontsize=7, y=0.985)
    fig.savefig(out_path, dpi=300, format="png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare candidate/reference 3DG: chr distance matrix + whole-genome 3D scatter")
    parser.add_argument("candidate_pos", nargs="?", type=Path, help="candidate 3DG path")
    parser.add_argument("reference_pos", nargs="?", type=Path, help="reference 3DG or .gz path")
    parser.add_argument("--candidate", type=Path, default=None, help="candidate 3DG path (flag form)")
    parser.add_argument("--reference", type=Path, default=None, help="reference 3DG or .gz path (flag form)")
    parser.add_argument("--chrom", default="chr1", choices=CHROMOSOMES, help="chromosome for the 2x2 matrix")
    parser.add_argument("--resolution", type=int, default=1_000_000, help="numeric bin resolution in bp")
    parser.add_argument("--mask", type=Path, default=None, help="optional frozen mask NPZ with chr<N>_positions/pair_i/pair_j/common")
    parser.add_argument("--orientation", choices=("auto", "direct", "swapped"), default="auto", help="candidate copy-to-reference orientation for the chr matrix panels")
    parser.add_argument("--no-align", dest="align", action="store_false", help="disable the default whole-genome global rigid alignment (historical reproduction)")
    parser.set_defaults(align=True)
    parser.add_argument("--candidate-id", default=None, help="label for the candidate scatter title (default: file stem)")
    parser.add_argument("--outdir", type=Path, default=ROOT / "scratch" / "plot_3dg_comparison", help="output directory; relative paths resolve against the repo root (default: <repo>/scratch/plot_3dg_comparison)")
    args = parser.parse_args()
    args.candidate = args.candidate or args.candidate_pos
    args.reference = args.reference or args.reference_pos
    if args.candidate is None or args.reference is None:
        parser.error("candidate and reference 3DG paths are required")
    if args.resolution <= 0:
        parser.error("resolution must be positive")
    return args


def main() -> int:
    start = time.perf_counter()
    args = parse_args()
    # 默认和相对 outdir 一律锚定项目 ROOT，避免从别处调用时在根目录堆产物。
    outdir: Path = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    plots = outdir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    candidate_path, reference_path = args.candidate.resolve(), args.reference.resolve()
    for path in (candidate_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"not a regular file: {path}")
    for path in ([args.mask.resolve()] if args.mask else []):
        if not path.is_file():
            raise FileNotFoundError(f"not a regular file: {path}")

    # 科学隔离：candidate 只算一次 SHA256，且在打开 reference 之前算出。
    candidate_sha = sha256_file(candidate_path)
    candidate_tracks = parse_3dg(candidate_path)
    reference_tracks = parse_3dg(reference_path)

    mask = load_mask(args.mask.resolve(), args.chrom, args.resolution) if args.mask else None
    matrix, checks = build_matrix(candidate_tracks, reference_tracks, args.chrom, args.resolution, mask, args.orientation)
    matrix_png = plots / f"{args.chrom}_{args.resolution // 1_000_000}Mb_distance_matrix.png"
    plot_matrix(matrix, checks, args.chrom, matrix_png)
    points = build_scatter(candidate_tracks, reference_tracks, args.resolution)
    alignment = fit_global_rotation(candidate_tracks, reference_tracks, points) if args.align else disabled_alignment()
    if alignment["aligned"]:
        apply_global_rotation(points, np.asarray(alignment["rotation_matrix"], dtype=np.float64))
    scatter_png = plots / "whole_genome_1Mb_3d_scatter.png"
    plot_scatter(points, scatter_png, args.candidate_id or candidate_path.stem, args.resolution, bool(alignment["aligned"]))

    matched = max(checks["direct_score"], checks["swapped_score"])
    cross = min(checks["direct_score"], checks["swapped_score"])
    metrics_path = outdir / "metrics.json"
    metrics = {
        "schema": "p9016-3dg-comparison-metrics-v1",
        "generated_utc": now_utc(),
        "wall_time_seconds": round(float(time.perf_counter() - start), 3),
        "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
        "candidate": {"path": str(candidate_path), "sha256": candidate_sha},
        "reference": {"path": str(reference_path)},
        "chromosome": args.chrom,
        "resolution": args.resolution,
        "mask": {"path": mask["source"], "n_positions": int(len(mask["positions"])), "n_common_pairs": checks["n_common_pairs"], "n_total_pairs": checks["n_total_pairs"]} if mask else None,
        "orientation": {"requested": args.orientation, "used": checks["orientation"], "direct_score": checks["direct_score"], "swapped_score": checks["swapped_score"]},
        "alignment": alignment,
        "paired_pearson": {"matched": matched, "cross": cross, "contrast": matched - cross, **checks["computed_pearson_rho"]},
        "paired_spearman": checks["computed_spearman_rho"],
        "matrix": {"n_positions": checks["n_positions"], "n_common_pairs": checks["n_common_pairs"], "n_total_pairs": checks["n_total_pairs"],
                   "group_scale_raw_median": matrix["group_scale_raw_median"], "color_limits": matrix["color_limits"],
                   "panel_labels": list(matrix["panel_labels"])},
        "scatter": {"union_grid_per_dataset": int(points["total_grid_per_dataset"]), "finite_reference": int(points["finite_count"][0]),
                    "finite_candidate": int(points["finite_count"][1]), "whole_cell_rms": points["scatter_rms"],
                    "axis_limits": points["axis_limits"]},
        "outputs": {"matrix_png": str(matrix_png), "scatter_png": str(scatter_png), "metrics_json": str(metrics_path)},
    }
    write_json(metrics_path, metrics)
    print(json.dumps({"status": "completed", "wall_time_seconds": metrics["wall_time_seconds"], "matrix_png": str(matrix_png),
                      "scatter_png": str(scatter_png), "orientation": checks["orientation"], "matched": matched, "cross": cross,
                      "aligned": bool(alignment["aligned"]), "det_rotation": alignment["det_rotation"],
                      "total_fit_points": alignment["total_fit_points"],
                      "n_common_pairs": checks["n_common_pairs"], "n_total_pairs": checks["n_total_pairs"],
                      "finite_reference": int(points["finite_count"][0]), "finite_candidate": int(points["finite_count"][1])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
