#!/usr/bin/env python3
"""根据冻结的统一矩阵 NPZ 仅重新渲染 footer。

本脚本有意不打开坐标、reference 文件或 R2 记录。它只加载已有的归一化矩阵和冻结的全局 color norm，重写 PNG/PDF 图，并验证可见文本布局。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
NPZ_PATH = HERE / "unified_chr1_distance_matrices.npz"
V1 = "v1_original_random_joint"
V2 = "v1_continuation"
C0S = ("C0_bundle1", "C0_bundle2", "C0_bundle3")
ALL_MAIN = (V1, V2, *C0S)
X0 = "x0_bundle2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"invalid PNG: {path}")
    return tuple(int(value) for value in np.frombuffer(header[16:24], dtype=">u4"))


def visible_text_layout(fig: Any, label: str, Text: Any) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    visible: list[tuple[str, Any]] = []
    outside: list[dict[str, Any]] = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not str(text.get_text()).strip():
            continue
        bbox = text.get_window_extent(renderer=renderer)
        text_value = str(text.get_text()).replace("\n", " / ")
        visible.append((text_value, bbox))
        if bbox.x0 < 0.0 or bbox.y0 < 0.0 or bbox.x1 > float(width) or bbox.y1 > float(height):
            outside.append({"text": text_value, "bbox": [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)]})
    overlaps: list[dict[str, Any]] = []
    for index, (left_text, left) in enumerate(visible):
        for right_text, right in visible[index + 1 :]:
            intersection_width = min(left.x1, right.x1) - max(left.x0, right.x0)
            intersection_height = min(left.y1, right.y1) - max(left.y0, right.y0)
            if intersection_width > 0.25 and intersection_height > 0.25:
                overlaps.append({"left": left_text, "right": right_text, "intersection_px2": float(intersection_width * intersection_height)})
    result = {
        "label": label,
        "canvas_px": [int(width), int(height)],
        "visible_text_count": len(visible),
        "outside_canvas": outside,
        "text_overlaps": overlaps,
        "all_visible_text_inside": not outside,
        "visible_text_nonoverlap": not overlaps,
    }
    if outside or overlaps:
        raise RuntimeError(f"layout failure for {label}: {json.dumps(result, ensure_ascii=False)}")
    return result


def render_one(npz: Any, condition_id: str, title: str, footer: str, output_stem: str) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.text import Text

    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.6,
    })
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad("#bdbdbd")
    positions_mb = np.asarray(npz["positions_bp"], dtype=np.float64) / 1e6
    half_bin_mb = 0.5
    extent = (float(positions_mb[0] - half_bin_mb), float(positions_mb[-1] + half_bin_mb), float(positions_mb[0] - half_bin_mb), float(positions_mb[-1] + half_bin_mb))
    ticks = np.asarray([3, 50, 100, 150, 195], dtype=float)
    matrix = np.asarray(npz[f"{condition_id}_normalized_matrices"], dtype=np.float64)
    labels = [str(value) for value in np.asarray(npz[f"{condition_id}_panel_labels"])]
    vmax = float(np.asarray(npz["global_color_norm"], dtype=np.float64)[1])
    fig = plt.figure(figsize=(6.0, 6.0), dpi=300)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.055], height_ratios=[1.0, 1.0], left=0.105, right=0.91, bottom=0.14, top=0.87, wspace=0.24, hspace=0.36)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    images = []
    norm = Normalize(vmin=0.0, vmax=vmax, clip=False)
    for axis, panel, panel_title in zip(axes, matrix, labels):
        image = axis.imshow(panel, origin="lower", interpolation="nearest", aspect="equal", extent=extent, cmap=cmap, norm=norm)
        images.append(image)
        axis.set_title(panel_title, pad=4)
        axis.set_xlim(extent[0], extent[1]); axis.set_ylim(extent[2], extent[3])
        axis.set_xticks(ticks); axis.set_yticks(ticks); axis.tick_params(width=0.6, length=2.5, pad=2)
        axis.set_xlabel("Genomic position (Mb)"); axis.set_ylabel("Genomic position (Mb)")
    colorbar_axis = fig.add_subplot(grid[:, 2])
    colorbar = fig.colorbar(images[0], cax=colorbar_axis)
    colorbar.set_label("normalized 3D Euclidean distance", fontsize=7, labelpad=4)
    colorbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    fig.suptitle(title, fontsize=7, y=0.925)
    fig.text(0.5, 0.045, footer, ha="center", va="center", fontsize=7, linespacing=1.15)
    layout = visible_text_layout(fig, condition_id, Text)
    png_path = HERE / f"{output_stem}.png"
    pdf_path = HERE / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, format="png")
    fig.savefig(pdf_path, dpi=300, format="pdf")
    plt.close(fig)
    return {"condition_id": condition_id, "png": str(png_path), "pdf": str(pdf_path), "png_sha256": sha256(png_path), "pdf_sha256": sha256(pdf_path), "png_dimensions": list(png_dimensions(png_path)), "figure_inches": [6.0, 6.0], "footer": footer, "layout": layout, "vmin": 0.0, "vmax": vmax, "colormap": "coolwarm_r"}


def render_index(npz: Any) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.text import Text

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 0.6})
    cmap = plt.get_cmap("coolwarm_r").copy(); cmap.set_bad("#bdbdbd")
    positions_mb = np.asarray(npz["positions_bp"], dtype=np.float64) / 1e6
    extent = (float(positions_mb[0] - 0.5), float(positions_mb[-1] + 0.5), float(positions_mb[0] - 0.5), float(positions_mb[-1] + 0.5))
    ticks = np.asarray([3, 50, 100, 150, 195], dtype=float)
    vmax = float(np.asarray(npz["global_color_norm"], dtype=np.float64)[1])
    order = (*ALL_MAIN, X0)
    fig = plt.figure(figsize=(6.0, 18.0), dpi=300)
    grid = fig.add_gridspec(6, 3, width_ratios=[1.0, 1.0, 0.055], left=0.11, right=0.91, bottom=0.055, top=0.965, wspace=0.25, hspace=0.5)
    norm = Normalize(vmin=0.0, vmax=vmax, clip=False)
    images = []
    for row_index, condition_id in enumerate(order):
        matrix = np.asarray(npz[f"{condition_id}_normalized_matrices"], dtype=np.float64)[2:]
        tracks = [str(value) for value in np.asarray(npz[f"{condition_id}_panel_tracks"])[2:]]
        for col_index, (panel, track) in enumerate(zip(matrix, tracks)):
            axis = fig.add_subplot(grid[row_index, col_index])
            image = axis.imshow(panel, origin="lower", interpolation="nearest", aspect="equal", extent=extent, cmap=cmap, norm=norm)
            images.append(image)
            axis.set_title(f"{condition_id} | {track}", pad=3)
            axis.set_xlim(extent[0], extent[1]); axis.set_ylim(extent[2], extent[3])
            axis.set_xticks(ticks); axis.set_yticks(ticks); axis.tick_params(width=0.6, length=2.5, pad=2)
            axis.set_xlabel("Genomic position (Mb)"); axis.set_ylabel("Genomic position (Mb)")
    colorbar_axis = fig.add_subplot(grid[:, 2])
    colorbar = fig.colorbar(images[0], cax=colorbar_axis)
    colorbar.set_label("normalized 3D Euclidean distance", fontsize=7, labelpad=4); colorbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    fig.suptitle("P9016 chr1 | unified candidate-panel index", fontsize=7, y=0.985)
    fig.text(0.5, 0.018, "Pooled two-copy RMS; common mask; near=red, far=blue, missing=gray\nShared color scale: 0-2.4182; whole-chromosome R2 alignment", ha="center", va="center", fontsize=7, linespacing=1.15)
    layout = visible_text_layout(fig, "unified_index", Text)
    png_path = HERE / "unified_chr1_distance_matrices.png"; pdf_path = HERE / "unified_chr1_distance_matrices.pdf"
    fig.savefig(png_path, dpi=300, format="png"); fig.savefig(pdf_path, dpi=300, format="pdf"); plt.close(fig)
    return {"condition_id": "unified_index", "png": str(png_path), "pdf": str(pdf_path), "png_sha256": sha256(png_path), "pdf_sha256": sha256(pdf_path), "png_dimensions": list(png_dimensions(png_path)), "figure_inches": [6.0, 18.0], "footer": "Pooled two-copy RMS; common mask; near=red, far=blue, missing=gray\nShared color scale: 0-2.4182; whole-chromosome R2 alignment", "layout": layout, "vmin": 0.0, "vmax": vmax, "colormap": "coolwarm_r"}


def main() -> None:
    npz_before = sha256(NPZ_PATH)
    with np.load(NPZ_PATH, allow_pickle=False) as npz:
        vmax = float(np.asarray(npz["global_color_norm"], dtype=np.float64)[1])
        if not math.isclose(vmax, 2.418206822915286, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"frozen global vmax changed: {vmax}")
        records = []
        main_footer = "Pooled two-copy RMS; common mask; near=red, far=blue, missing=gray\nShared color scale: 0-2.4182; whole-chromosome R2 alignment"
        labels = {
            V1: "020 historical selected",
            V2: "022 historical continuation",
            "C0_bundle1": "029 C0 bundle1 seed",
            "C0_bundle2": "029 C0 bundle2 seed",
            "C0_bundle3": "029 C0 bundle3 seed",
        }
        stems = {
            V1: "020_historical_selected_chr1_distance_matrices",
            V2: "022_historical_continuation_chr1_distance_matrices",
            "C0_bundle1": "C0_bundle1_seed_chr1_distance_matrices",
            "C0_bundle2": "C0_bundle2_seed_chr1_distance_matrices",
            "C0_bundle3": "C0_bundle3_seed_chr1_distance_matrices",
        }
        for condition_id in ALL_MAIN:
            records.append(render_one(npz, condition_id, f"P9016 chr1 | {labels[condition_id]} (1 Mb)", main_footer, stems[condition_id]))
        x0_footer = "Pooled two-copy RMS; common mask; near=red, far=blue, missing=gray\nShared color scale: 0-2.4182; x0 best-swap (chr1 agrees with endpoint)"
        records.append(render_one(npz, X0, "P9016 chr1 | 025 x0 bundle2 posthoc (1 Mb)", x0_footer, "x0_bundle2_chr1_distance_matrices"))
        index_record = render_index(npz)
    npz_after = sha256(NPZ_PATH)
    if npz_after != npz_before:
        raise RuntimeError("frozen unified NPZ changed during render-only revision")
    output = {
        "schema": "p9016-render-revision-short-footer-v1",
        "status": "PASS",
        "mode": "render_only_from_frozen_unified_npz",
        "input_npz": str(NPZ_PATH),
        "input_npz_sha256_before": npz_before,
        "input_npz_sha256_after": npz_after,
        "coordinates_reopened": False,
        "reference_reopened": False,
        "rho_recomputed": False,
        "r2_recomputed": False,
        "global_vmax": vmax,
        "figures": records + [index_record],
        "all_visible_text_inside": all(record["layout"]["all_visible_text_inside"] for record in records + [index_record]),
        "all_visible_text_nonoverlap": all(record["layout"]["visible_text_nonoverlap"] for record in records + [index_record]),
    }
    (HERE / "render_revision_short_footer.json").write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
