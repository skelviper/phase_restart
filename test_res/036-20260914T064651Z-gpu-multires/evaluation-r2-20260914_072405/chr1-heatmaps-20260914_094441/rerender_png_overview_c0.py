#!/usr/bin/env python3
"""为锁定的 chr1 heatmaps 进行仅 PNG 的布局和逐拷贝 rho 重渲染。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/ssd/zliu/phase_restart")
OUT_DIR = ROOT / "test_res/036-20260914T064651Z-gpu-multires/evaluation-r2-20260914_072405/chr1-heatmaps-20260914_094441"
ARRAYS_PATH = OUT_DIR / "distance_arrays.npz"
METADATA_PATH = OUT_DIR / "metadata.json"
FONT_SIZE_PT = 7
DPI = 300
V_MIN = 0.0
CONDITION_LABELS = (
    "Reference",
    "020 / C0 CPU",
    "C0 GPU",
    "C1 GPU",
    "C2-map GPU",
    "C2-free GPU",
    "C3 GPU",
)
COPY_LABELS = (
    ("reference maternal", "reference paternal"),
    ("mat-matched", "pat-matched"),
    ("mat-matched", "pat-matched"),
    ("mat-matched", "pat-matched"),
    ("mat-matched", "pat-matched"),
    ("mat-matched", "pat-matched"),
    ("mat-matched", "pat-matched"),
)
# 已有 R2 原始四 rho 值，固定整条染色体交换映射。
PANEL_RHOS = (
    (None, None),
    (0.5601095185310723, 0.7797499716726471),
    (0.5715367833009062, 0.7913232211609275),
    (0.48442414540361334, 0.6762453108999376),
    (0.4022258022132569, 0.6860351091686574),
    (0.4022258022132569, 0.6860351091686574),
    (0.5657909010737924, 0.7936893429768374),
)


def masked_display(matrix: np.ndarray, pair_mask: np.ndarray) -> np.ma.MaskedArray:
    values = np.asarray(matrix, dtype=np.float64).copy()
    values[~pair_mask] = np.nan
    diagonal = np.diag_indices(values.shape[0])
    values[diagonal] = 0.0
    return np.ma.masked_invalid(values)


def set_ticks(axis: object, positions_mb: np.ndarray) -> None:
    ticks = np.asarray([3, 50, 100, 150, 195], dtype=np.float64)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xlim(float(positions_mb[0] - 0.5), float(positions_mb[-1] + 0.5))
    axis.set_ylim(float(positions_mb[0] - 0.5), float(positions_mb[-1] + 0.5))
    axis.set_xlabel("chr1 position (Mb)", fontsize=FONT_SIZE_PT)
    axis.set_ylabel("chr1 position (Mb)", fontsize=FONT_SIZE_PT)
    axis.tick_params(axis="both", labelsize=FONT_SIZE_PT, length=2, pad=1)


def rho_text(value: float | None, prefix: str = "rho=") -> str:
    return "" if value is None else f" | {prefix}{value:.4f}"


def main() -> int:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    arrays = np.load(ARRAYS_PATH)
    normalized = np.asarray(arrays["normalized_distance"], dtype=np.float64)
    pair_mask = np.asarray(arrays["frozen_pair_mask"], dtype=bool)
    positions_mb = np.asarray(arrays["grid_mb"], dtype=np.float64)
    vmax = float(metadata["normalization"]["vmax"])
    if normalized.shape != (7, 2, 193, 193):
        raise RuntimeError(f"unexpected normalized shape: {normalized.shape}")
    if int(pair_mask.sum() // 2) != 17578:
        raise RuntimeError("frozen pair mask changed")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "figure.titlesize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "font.family": "DejaVu Sans",
        }
    )
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad("#d0d0d0")
    norm = matplotlib.colors.Normalize(vmin=V_MIN, vmax=vmax, clip=False)
    extent = (
        float(positions_mb[0] - 0.5),
        float(positions_mb[-1] + 0.5),
        float(positions_mb[0] - 0.5),
        float(positions_mb[-1] + 0.5),
    )

    # 仅针对 overview：在两行 suptitle 与 Reference row 之间留出明确空间，
    # 并将共享 colorbar label 保持在 canvas 内。
    fig, axes = plt.subplots(7, 2, figsize=(6.8, 21.0), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.12, right=0.89, top=0.945, bottom=0.075, wspace=0.18, hspace=0.62)
    image = None
    for row_index, condition_label in enumerate(CONDITION_LABELS):
        for copy_index, copy_label in enumerate(COPY_LABELS[row_index]):
            axis = axes[row_index, copy_index]
            image = axis.imshow(
                masked_display(normalized[row_index, copy_index], pair_mask),
                cmap=cmap,
                norm=norm,
                origin="lower",
                interpolation="none",
                extent=extent,
                aspect="equal",
            )
            axis.set_title(
                f"{condition_label}\n{copy_label}{rho_text(PANEL_RHOS[row_index][copy_index], 'chr1 rho=')}",
                fontsize=FONT_SIZE_PT,
                pad=2,
            )
            set_ticks(axis, positions_mb)
            if copy_index == 1:
                axis.set_ylabel("")
                axis.tick_params(axis="y", labelleft=False)
        axes[row_index, 0].text(
            -0.36,
            0.5,
            condition_label,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=FONT_SIZE_PT,
        )
    fig.suptitle(
        "P9016 chr1 distance matrices | one cell / 1 Mb / whole-chr best-swap\n"
        "mat-matched / pat-matched = evaluation alignment, not training parent labels",
        fontsize=FONT_SIZE_PT,
        y=0.989,
    )
    cbar_axis = fig.add_axes([0.13, 0.032, 0.73, 0.010])
    colorbar = fig.colorbar(image, cax=cbar_axis, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=FONT_SIZE_PT, length=2, pad=1)
    colorbar.set_label("distance / median distance on frozen R2 pairs", fontsize=FONT_SIZE_PT, labelpad=3)
    fig.savefig(OUT_DIR / "overview.png", dpi=DPI, format="png")
    fig.savefig(OUT_DIR / "overview.pdf", dpi=DPI, format="pdf")
    plt.close(fig)

    # 七张 detail figures 均使用已接受的 C0 layout、逐 copy 的原始 rho labels、
    # 相同的 axes/ticks 以及同一冻结 normalization/color scale。
    detail_stems = (
        "reference",
        "020_C0_CPU",
        "C0",
        "C1",
        "C2-map",
        "C2-free",
        "C3",
    )
    for row_index, (condition_label, stem) in enumerate(zip(CONDITION_LABELS, detail_stems)):
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(left=0.09, right=0.91, top=0.86, bottom=0.24, wspace=0.14)
        image = None
        for copy_index, copy_label in enumerate(COPY_LABELS[row_index]):
            axis = axes[0, copy_index]
            image = axis.imshow(
                masked_display(normalized[row_index, copy_index], pair_mask),
                cmap=cmap,
                norm=norm,
                origin="lower",
                interpolation="none",
                extent=extent,
                aspect="equal",
            )
            axis.set_title(
                f"{copy_label}{rho_text(PANEL_RHOS[row_index][copy_index])}",
                fontsize=FONT_SIZE_PT,
                pad=2,
            )
            set_ticks(axis, positions_mb)
            if copy_index == 1:
                axis.set_ylabel("")
                axis.tick_params(axis="y", labelleft=False)
        fig.suptitle(f"{condition_label} | chr1 whole-chromosome distance", fontsize=FONT_SIZE_PT, y=0.965)
        cbar_axis = fig.add_axes([0.12, 0.105, 0.76, 0.035])
        colorbar = fig.colorbar(image, cax=cbar_axis, orientation="horizontal")
        colorbar.ax.tick_params(labelsize=FONT_SIZE_PT, length=2, pad=1)
        colorbar.set_label("distance / median distance on frozen R2 pairs", fontsize=FONT_SIZE_PT, labelpad=1)
        fig.savefig(OUT_DIR / f"detail_{stem}.png", dpi=DPI, format="png")
        fig.savefig(OUT_DIR / f"detail_{stem}.pdf", dpi=DPI, format="pdf")
        plt.close(fig)
    print(json.dumps({"status": "PASS", "overview": str(OUT_DIR / "overview.png"), "detail_C0": str(OUT_DIR / "detail_C0.png"), "detail_count": 7, "vmax": vmax, "mask_pairs": int(pair_mask.sum() // 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
