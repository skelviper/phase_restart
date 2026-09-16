#!/usr/bin/env python3
"""两个 diploid 3DG 并排比较制图入口（精简实现）。

`plot_3dg_comparison.py` 把第二个文件固定当作 reference（`chrN(mat)` / `chrN(pat)`），
因此它无法比较两个候选结构（例如 oracle 与工作 baseline），两者都叫 `c<NNa/b>`。
本脚本只做这一件事：把 A、B 两个双拷贝结构的同一条 chromosome 画成 2x2 距离矩阵。

固定规则：
- A、B 各自两条 copy track 的名字由 `--a-copy-prefix` / `--b-copy-prefix` 决定（默认 `c{ci:02d}` + `a`/`b`）。
- grid 是 A、B 四条 track 在该 chromosome 上出现过的 position 并集；共同对 = 四条 track 都 finite 的非对角 pair。
- 方向默认 `auto`：按 `direct`/`swapped` 的 Pearson 均值择大，差 <= 1e-12 记 `unresolved_tie` 并按 direct 画图。
  四 panel 统一为 A copyA / A copyB / B 对应 copy / B 对应 copy，即同一列是同一"侧"，便于直接看差异。
- 共同尺度：四条距离向量合起来一个 median，四个 panel 共用同一 vmin/vmax 与同一 colorbar，
  因此 panel 之间的深浅可直接比；灰色是缺 bin 或非同源 pair。
- 只写一张 PNG 和一份 metrics.json；不做刚体对齐、不重跑评价、不改冻结模块。

科学隔离：A 的 SHA256 在打开 B 之前算出并写进 metrics.json；本脚本只读两份 3DG。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHROMOSOMES = tuple([f"chr{i}" for i in range(1, 20)] + ["chrX"])
TIE_TOLERANCE = 1e-12


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_3dg(path: Path) -> dict[str, dict[int, np.ndarray]]:
    """读 3DG（可 gzip）；key 形如 `c01a` 或 `chr1(mat)`。"""
    opener = gzip.open if path.suffix == ".gz" else open
    tracks: dict[str, dict[int, np.ndarray]] = {}
    with opener(path, "rt") as handle:  # type: ignore[operator]
        for line in handle:
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                position = int(fields[1])
                point = np.asarray([float(x) for x in fields[2:5]], dtype=np.float64)
            except ValueError:
                continue
            tracks.setdefault(fields[0], {})[position] = point
    if not tracks:
        raise ValueError(f"no usable 3DG rows in {path}")
    return tracks


def copy_tracks(tracks: dict[str, dict[int, np.ndarray]], prefix: str, path: Path) -> tuple[str, str]:
    names = (f"{prefix}a", f"{prefix}b")
    missing = [name for name in names if name not in tracks]
    if missing:
        raise ValueError(f"{path} has no track(s) {missing}; available: {sorted(tracks)[:6]}")
    return names


def points_on_grid(tracks: dict[str, dict[int, np.ndarray]], track: str, positions: np.ndarray) -> np.ndarray:
    output = np.full((len(positions), 3), np.nan, dtype=np.float64)
    rows = tracks.get(track, {})
    for index, position in enumerate(positions):
        point = rows.get(int(position))
        if point is not None and np.isfinite(point).all():
            output[index] = point
    return output


def corr(left: np.ndarray, right: np.ndarray) -> float:
    keep = np.isfinite(left) & np.isfinite(right)
    x = np.asarray(left[keep], dtype=np.float64)
    y = np.asarray(right[keep], dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_mask(path: Path, chromosome: str, resolution: int) -> dict[str, Any]:
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
    if not np.array_equal(pair_i, expected_i) or not np.array_equal(pair_j, expected_j):
        raise ValueError("mask pair arrays are not strict upper triangle")
    return {"positions": positions, "pair_i": pair_i, "pair_j": pair_j, "common": common, "source": str(path.resolve())}


def numeric_grid(a_tracks: dict[str, dict[int, np.ndarray]], b_tracks: dict[str, dict[int, np.ndarray]],
                 a_names: tuple[str, str], b_names: tuple[str, str], resolution: int) -> np.ndarray:
    """A、B 四条 track 出现过的 position 并集，且对齐到 resolution；缺 bin 由 NaN 覆盖。"""
    seen: set[int] = set()
    for tracks, names in ((a_tracks, a_names), (b_tracks, b_names)):
        for name in names:
            seen.update(int(position) for position in tracks.get(name, {}))
    positions = np.asarray(sorted(seen), dtype=np.int64)
    positions = positions[positions % resolution == 0]
    if len(positions) < 2:
        raise ValueError("fewer than two usable grid positions")
    return positions


def orientation_choice(raw_rhos: dict[str, float], requested: str) -> tuple[str, int, int]:
    direct = raw_rhos["A0_B0"] + raw_rhos["A1_B1"]
    swapped = raw_rhos["A0_B1"] + raw_rhos["A1_B0"]
    if requested == "direct":
        return "direct", 0, 1
    if requested == "swapped":
        return "swapped", 1, 0
    if not (np.isfinite(direct) and np.isfinite(swapped)):
        return "unresolved", 0, 1
    if np.isclose(direct, swapped, rtol=0.0, atol=TIE_TOLERANCE):
        return "unresolved_tie", 0, 1
    return ("direct", 0, 1) if direct > swapped else ("swapped", 1, 0)


def plot_matrix(grid_mb: np.ndarray, grid_bp: np.ndarray, normalized: np.ndarray, titles: list[str],
                color_limits: np.ndarray, footer: str, chromosome: str, bin_mb: float, out_path: Path) -> None:
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad("#bdbdbd")
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 6.0), dpi=300, squeeze=False)
    extent = [float(grid_mb[0] - bin_mb / 2), float(grid_mb[-1] + bin_mb / 2),
              float(grid_mb[0] - bin_mb / 2), float(grid_mb[-1] + bin_mb / 2)]
    tick_values = np.unique(np.asarray([grid_mb[0], grid_mb[len(grid_mb) // 3],
                                        grid_mb[2 * len(grid_mb) // 3], grid_mb[-1]])) if len(grid_mb) > 5 else grid_mb
    image = None
    for panel_index, ax in enumerate(axes.flat):
        image = ax.imshow(np.ma.masked_invalid(normalized[panel_index]), origin="lower", extent=extent,
                          interpolation="none", aspect="equal", cmap=cmap,
                          vmin=float(color_limits[0]), vmax=float(color_limits[1]))
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
    fig.text(0.5, 0.018, footer, ha="center", va="bottom", fontsize=7, linespacing=1.15)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.15, top=0.90, wspace=0.25, hspace=0.34)
    fig.savefig(out_path, dpi=300, format="png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot two diploid 3DG structures side by side (chr distance matrix)")
    parser.add_argument("a_pos", nargs="?", type=Path, help="structure A 3DG path")
    parser.add_argument("b_pos", nargs="?", type=Path, help="structure B 3DG path")
    parser.add_argument("--a", type=Path, default=None, help="structure A 3DG path (flag form)")
    parser.add_argument("--b", type=Path, default=None, help="structure B 3DG path (flag form)")
    parser.add_argument("--a-label", default=None, help="panel label for A (default: file stem)")
    parser.add_argument("--b-label", default=None, help="panel label for B (default: file stem)")
    parser.add_argument("--chrom", default="chr1", choices=CHROMOSOMES)
    parser.add_argument("--resolution", type=int, default=1_000_000)
    parser.add_argument("--a-copy-prefix", default=None, help="track prefix for A copies (default: c<NN> from --chrom)")
    parser.add_argument("--b-copy-prefix", default=None, help="track prefix for B copies (default: c<NN> from --chrom)")
    parser.add_argument("--mask", type=Path, default=None, help="optional frozen mask NPZ (chr<N>_positions/pair_i/pair_j/common)")
    parser.add_argument("--orientation", choices=("auto", "direct", "swapped"), default="auto")
    parser.add_argument("--outdir", type=Path, default=ROOT / "scratch" / "plot_3dg_pair")
    parser.add_argument("--png-name", default=None, help="output PNG file name (default: <chrom>_<N>Mb_pair_distance_matrix.png)")
    args = parser.parse_args()
    args.a = args.a or args.a_pos
    args.b = args.b or args.b_pos
    if args.a is None or args.b is None:
        parser.error("two 3DG paths are required")
    if args.resolution <= 0:
        parser.error("resolution must be positive")

    outdir: Path = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    plots = outdir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    a_path, b_path = args.a.resolve(), args.b.resolve()
    for path in (a_path, b_path):
        if not path.is_file():
            raise FileNotFoundError(f"not a regular file: {path}")
    if args.mask is not None and not args.mask.resolve().is_file():
        raise FileNotFoundError(f"not a regular file: {args.mask.resolve()}")

    # 科学隔离：A 的 SHA256 只算一次，且在打开 B 之前算出。
    a_sha = sha256_file(a_path)
    a_tracks = parse_3dg(a_path)
    b_tracks = parse_3dg(b_path)

    ci = CHROMOSOMES.index(args.chrom)
    a_prefix = args.a_copy_prefix or f"c{ci + 1:02d}"
    b_prefix = args.b_copy_prefix or f"c{ci + 1:02d}"
    a_names = copy_tracks(a_tracks, a_prefix, a_path)
    b_names = copy_tracks(b_tracks, b_prefix, b_path)

    if args.mask is not None:
        mask = load_mask(args.mask.resolve(), args.chrom, args.resolution)
        positions, pair_i, pair_j = mask["positions"], mask["pair_i"], mask["pair_j"]
        common = mask["common"].copy()
    else:
        mask = None
        positions = numeric_grid(a_tracks, b_tracks, a_names, b_names, args.resolution)
        pair_i, pair_j = np.triu_indices(len(positions), 1)
        common = np.ones(len(pair_i), dtype=bool)
    finite_beads = []
    for tracks, names in ((a_tracks, a_names), (b_tracks, b_names)):
        for name in names:
            finite_beads.append(np.isfinite(points_on_grid(tracks, name, positions)).all(axis=1))
    for beads in finite_beads:
        common &= beads[pair_i] & beads[pair_j]
    if not common.any():
        raise ValueError(f"no valid common pair for {args.chrom}")
    used_i, used_j = pair_i[common], pair_j[common]

    vectors = {}
    for tag, tracks, names in (("A0", a_tracks, a_names[0]), ("A1", a_tracks, a_names[1]),
                               ("B0", b_tracks, b_names[0]), ("B1", b_tracks, b_names[1])):
        points = points_on_grid(tracks, names, positions)
        delta = points[used_i] - points[used_j]
        vectors[tag] = np.sqrt(np.sum(delta * delta, axis=1))
    raw_rhos = {
        "A0_B0": corr(vectors["A0"], vectors["B0"]),
        "A0_B1": corr(vectors["A0"], vectors["B1"]),
        "A1_B0": corr(vectors["A1"], vectors["B0"]),
        "A1_B1": corr(vectors["A1"], vectors["B1"]),
    }
    if not all(np.isfinite(value) for value in raw_rhos.values()):
        raise ValueError("Pearson comparison is not finite for this grid/mask")
    orientation, b_first, b_second = orientation_choice(raw_rhos, args.orientation)
    direct_score = (raw_rhos["A0_B0"] + raw_rhos["A1_B1"]) / 2.0
    swapped_score = (raw_rhos["A0_B1"] + raw_rhos["A1_B0"]) / 2.0

    n = len(positions)
    panel_vectors = (vectors["A0"], vectors["A1"],
                     vectors["B1"] if b_first else vectors["B0"],
                     vectors["B0"] if b_second else vectors["B1"])
    scale = float(np.nanmedian(np.concatenate(panel_vectors)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("non-positive matrix normalization scale")
    raw = np.full((4, n, n), np.nan, dtype=np.float64)
    for panel_index, values in enumerate(panel_vectors):
        raw[panel_index, used_i, used_j] = values
        raw[panel_index, used_j, used_i] = values
    normalized = raw / scale
    visible = normalized[np.isfinite(normalized)]
    color_limits = np.asarray([0.0, float(np.max(visible))], dtype=np.float64)

    a_label = args.a_label or a_path.stem
    b_label = args.b_label or b_path.stem
    titles = [f"{a_label} copyA", f"{a_label} copyB",
              f"{b_label} {'copyB' if b_first else 'copyA'} (rho={raw_rhos['A0_B1' if b_first else 'A0_B0']:.4f})",
              f"{b_label} {'copyA' if b_second else 'copyB'} (rho={raw_rhos['A1_B0' if b_second else 'A1_B1']:.4f})"]
    matched = max(direct_score, swapped_score)
    cross = min(direct_score, swapped_score)
    footer = (f"Matched {matched:.6f} | Cross {cross:.6f} | Contrast {matched - cross:.6f} | orientation {orientation}\n"
              f"Common pairs: {int(common.sum()):,} / {int(len(pair_i)):,} | "
              f"grid positions: {n} | shared scale (raw median) {scale:.4f}")
    bin_bp = int(np.median(np.diff(positions))) if n > 1 else args.resolution
    png = plots / (args.png_name or f"{args.chrom}_{args.resolution // 1_000_000}Mb_pair_distance_matrix.png")
    plot_matrix(positions.astype(np.float64) / 1e6, positions, normalized, titles, color_limits, footer,
                args.chrom, bin_bp / 1e6, png)

    metrics = {
        "schema": "p9016-3dg-pair-metrics-v1",
        "generated_utc": now_utc(),
        "a": {"path": str(a_path), "sha256": a_sha, "label": a_label, "copies": list(a_names), "copy_prefix": a_prefix},
        "b": {"path": str(b_path), "label": b_label, "copies": list(b_names), "copy_prefix": b_prefix},
        "chromosome": args.chrom,
        "resolution": args.resolution,
        "mask": {"path": mask["source"], "n_common_pairs": int(common.sum()), "n_total_pairs": int(len(pair_i))} if mask else None,
        "orientation": {"requested": args.orientation, "used": orientation,
                        "direct_score": float(direct_score), "swapped_score": float(swapped_score),
                        "b_first": b_first, "b_second": b_second},
        "computed_pearson_rho": raw_rhos,
        "matched": float(matched),
        "cross": float(cross),
        "contrast": float(matched - cross),
        "grid_positions": int(n),
        "shared_scale_raw_median": scale,
        "color_limits": [float(color_limits[0]), float(color_limits[1])],
        "outputs": {"matrix_png": str(png)},
    }
    metrics_path = outdir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "matrix_png": str(png), "metrics_json": str(metrics_path),
                      "orientation": orientation, "matched": matched, "cross": cross,
                      "n_common_pairs": int(common.sum()), "n_total_pairs": int(len(pair_i))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
