#!/usr/bin/env python3
"""仅渲染修正两张 Stage-A 图。

脚本读取已保存的 history TSV 和 endpoint NPZ 坐标。它只使用冻结的 aggregate 恢复最近距离 panel，绝不评估 objective 或调用 optimizer。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
REPO = OUT.parents[2]
INPUT = REPO / "inputs/P9016.snpfree.pairs.gz"
RUN020 = REPO / "test_res/020-20260913_071841-v1-p9016-joint"
RUN022 = REPO / "test_res/022-20260913_111031-v1-continuation-fdg-r2"
ROOT_HISTORY = ROOT / "results/history.tsv"
ROOT_PLOTS = ROOT / "plots"
NEW = OUT / "render_revision/new"
CURRENT_PR = REPO / "pr"

sys.path.insert(0, str(REPO))
from pr import contact_model  # noqa: E402

contact_model.SNPFREE = str(INPUT)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_history() -> list[dict[str, str]]:
    with ROOT_HISTORY.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_endpoint(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["y"], dtype=np.float64), np.asarray(payload["coordinates"], dtype=np.float64)


def nearest_repulsion_distances(data: Any, x: np.ndarray) -> np.ndarray:
    n = data.n_loci
    nearest = np.full((2, n), np.inf, dtype=np.float64)
    block_size = 65536
    for start in range(0, data.n_pairs, block_size):
        stop = min(start + block_size, data.n_pairs)
        i = data.pair_i[start:stop]
        j = data.pair_j[start:stop]
        for copy_i, copy_j in ((0, 0), (0, 1), (1, 0), (1, 1)):
            distance = np.linalg.norm(x[copy_i, i] - x[copy_j, j], axis=1)
            np.minimum.at(nearest[copy_i], i, distance)
            np.minimum.at(nearest[copy_j], j, distance)
    homolog = np.linalg.norm(x[0] - x[1], axis=1)
    nearest = np.minimum(nearest, homolog[None, :])
    if not np.all(np.isfinite(nearest)):
        raise AssertionError("nearest repulsion inventory incomplete")
    return nearest.ravel()


def render_continuation(history: list[dict[str, str]]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6})
    rows = [row for row in history if row["source"] == "022"]
    x = np.asarray([int(row["iteration"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    axes[0].plot(x, [float(row["total"]) for row in rows], color="#222222", lw=0.8, label="total")
    axes[0].plot(x, [float(row["count_nll_normalized"]) for row in rows], color="#3366aa", lw=0.8, ls="--", label="count NLL")
    axes[0].set_title("022 continuation (local 0-240)")
    axes[0].set_xlabel("local accepted iteration")
    axes[0].set_ylabel("objective / count NLL")
    axes[0].legend(frameon=False, loc="best")
    axes[0].grid(alpha=0.2)
    for key, color, label in (("bend", "#cc5533", "bend x 0.01"),
                              ("bond", "#3366aa", "bond"),
                              ("repulsion", "#228855", "repulsion"),
                              ("p_prior", "#8855aa", "p-prior")):
        scale = 0.01 if key == "bend" else 1.0
        axes[1].plot(x, [scale * float(row[key]) for row in rows], lw=0.8, label=label, color=color)
    axes[1].set_title("weighted priors")
    axes[1].set_xlabel("local accepted iteration")
    axes[1].set_ylabel("contribution")
    # 在该 axes 内放置紧凑的单列 legend；ncol=2 时，handles/text 可能触及相邻的 y tick labels。
    axes[1].legend(frameon=False, loc="upper right", ncol=1, fontsize=6,
                   borderaxespad=0.25, handlelength=1.4, labelspacing=0.25)
    axes[1].grid(alpha=0.2)
    path = NEW / "continuation_history.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def render_geometry(data: Any, endpoint_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    colors = {"020": "#3366aa", "022": "#cc5533"}
    labels = {"020": "020", "022": "022"}
    l0 = float(data.l0)
    threshold = 0.7 * l0
    fig, axes = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)
    for short, (y, x) in endpoint_data.items():
        radius = np.linalg.norm(x, axis=2).ravel()
        order = np.argsort(radius)
        axes[0, 0].plot(radius[order], np.arange(1, len(radius) + 1) / len(radius), lw=0.8,
                         color=colors[short], label=labels[short])
        inverse = 1.0 / np.sqrt(1.0 + np.sum(y * y, axis=2)).ravel()
        radial = inverse ** 3
        tangential = inverse
        order = np.argsort(radius)
        axes[0, 1].plot(radius[order], radial[order], lw=0.5, color=colors[short], label=labels[short] + " radial")
        axes[0, 1].plot(radius[order], tangential[order], lw=0.5, ls="--", color=colors[short], label=labels[short] + " tangential")
        bond_relative = np.concatenate([
            np.linalg.norm(np.diff(x[:, data.chromosome_slice(ci)], axis=1), axis=2).ravel()
            for ci, _name in enumerate(data.chromosome_names)
        ]) / l0
        axes[1, 0].hist(bond_relative, bins=50, density=True, histtype="step", lw=0.8,
                         color=colors[short], label=labels[short])
        nearest = nearest_repulsion_distances(data, x)
        axes[1, 1].hist(nearest, bins=50, density=True, histtype="step", lw=0.8,
                         color=colors[short], label=labels[short])
    axes[0, 0].axvline(0.9, color="#666666", lw=0.5, ls=":")
    axes[0, 0].axvline(0.99, color="#666666", lw=0.5, ls="--")
    axes[0, 0].set_title("radius CDF")
    axes[0, 0].set_xlabel("radius")
    axes[0, 0].set_ylabel("fraction beads")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set_title("sphere Jacobian attenuation")
    axes[0, 1].set_xlabel("radius")
    axes[0, 1].set_ylabel("singular attenuation")
    axes[0, 1].legend(frameon=False, ncol=2)
    # Bond values 在绘图前显式归一化，与轴标签和 0.75--1.25 target band 一致。
    axes[1, 0].axvspan(0.75, 1.25, color="#dddddd", alpha=0.4)
    axes[1, 0].set_title("backbone bond")
    axes[1, 0].set_xlabel("distance / l0")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].axvline(threshold, color="#222222", lw=0.7, ls="--", label="0.7 l0")
    axes[1, 1].set_title("nearest repulsion distance")
    axes[1, 1].set_xlabel("distance")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].legend(frameon=False)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    path = NEW / "endpoint_geometry_diagnostics.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def main() -> None:
    NEW.mkdir(parents=True, exist_ok=True)
    data = contact_model.load_frozen_p9016_aggregate(1_000_000)
    history = load_history()
    endpoint_specs = {
        "020": RUN020 / "checkpoints/random_joint/1m-accepted-0240.npz",
        "022": RUN022 / "theta/final-theta.npz",
    }
    endpoint_data = {short: load_endpoint(path) for short, path in endpoint_specs.items()}
    paths = [render_continuation(history), render_geometry(data, endpoint_data)]
    numeric_files = [path for path in ROOT.rglob("*") if path.is_file()
                     and not path.is_relative_to(OUT)
                     and "plots" not in path.parts
                     and path.name != "manifest.json"]
    preview = {
        "schema_version": "post020-render-revision-preview-v1",
        "status": "rendered_to_preflight_only",
        "old_root_plot_hashes": {
            "continuation_history.png": sha256(ROOT_PLOTS / "continuation_history.png"),
            "endpoint_geometry_diagnostics.png": sha256(ROOT_PLOTS / "endpoint_geometry_diagnostics.png"),
        },
        "new_preflight_plot_hashes": {path.name: sha256(path) for path in paths},
        "root_manifest_before_sha256": sha256(ROOT / "results/manifest.json"),
        "root_numeric_file_hashes_before": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(numeric_files)},
        "numeric_unchanged_at_preview": True,
        "render_changes": {
            "endpoint_geometry_diagnostics": "backbone bond histogram uses distance/l0 before plotting; target band remains 0.75--1.25",
            "continuation_history": "right-panel legend is compact one-column with short labels and remains inside right axes",
        },
        "objective_or_optimizer_called": False,
        "source_input": {"input": str(INPUT), "input_sha256": sha256(INPUT), "contact_model_sha256": sha256(CURRENT_PR / "contact_model.py")},
    }
    write_json = lambda path, value: path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    write_json(OUT / "results/render_revision_preview.json", preview)
    print(json.dumps({"status": "rendered", "new_plots": [str(path) for path in paths], "numeric_unchanged": True}, sort_keys=True))


if __name__ == "__main__":
    main()
