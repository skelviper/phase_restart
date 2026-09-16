"""图形辅助函数：3 英寸基础面板、300 DPI、统一 7 pt 文字。"""
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", ".mplcache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _save(fig, path):
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def dose_response(path, series, null_band=None, gauge=None):
    """series：{label: (x_mb, delta, lo, hi, color)}。"""
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    if null_band is not None:
        ax.axhspan(-null_band, null_band, color="0.85", zorder=0,
                   label="null spread (random splits)")
    if gauge is not None:
        ax.axhline(gauge, color="0.4", ls=":", zorder=1, label="engine gauge asymmetry")
    for label, (x, d, lo, hi, col) in series.items():
        x, d = np.asarray(x), np.asarray(d)
        o = np.argsort(x)
        ax.errorbar(x[o], d[o], yerr=[d[o] - np.asarray(lo)[o], np.asarray(hi)[o] - d[o]],
                    marker="o", ms=2.5, capsize=1.5, color=col, label=label)
    ax.set_xlabel("flipped region (Mb)")
    ax.set_ylabel("held-out rho drop vs oracle")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    return _save(fig, path)


def strata_bars(path, ids, strata, stratum_names, title=""):
    """strata：{cid: {stratum: (rho, n)}}。"""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    n_s = len(stratum_names)
    w = 0.8 / n_s
    xs = np.arange(len(ids))
    for k, s in enumerate(stratum_names):
        v = [strata[c].get(s, {}).get("rho", np.nan) for c in ids]
        ax.bar(xs + (k - (n_s - 1) / 2) * w, v, w, label=s)
    ax.set_xticks(xs)
    ax.set_xticklabels(ids, rotation=60, ha="right")
    ax.set_ylabel("held-out Spearman rho")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    return _save(fig, path)


def masked_effect(path, ids, delta, lo, hi, null_delta):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    xs = np.arange(len(ids))
    ax.axhline(0, color="0.3", lw=0.6)
    ax.bar(xs, null_delta, 0.55, color="0.8", label="null (two half-data oracles)")
    ax.errorbar(xs, delta, yerr=[np.asarray(delta) - np.asarray(lo),
                                 np.asarray(hi) - np.asarray(delta)],
                fmt="o", ms=3, capsize=1.5, color="C3", label="oracle - candidate")
    ax.set_xticks(xs)
    ax.set_xticklabels(ids, rotation=60, ha="right")
    ax.set_ylabel("rho drop on the damaged bin pairs")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    return _save(fig, path)


def distance_maps(path, panel, ncols=2, title=""):
    """panel：由 (key, matrix, vmin, vmax) 组成的列表。"""
    n = len(panel)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows),
                             squeeze=False)
    for ax, (key, M, vmin, vmax) in zip(axes.ravel(), panel):
        im = ax.imshow(M, cmap="coolwarm_r", vmin=vmin, vmax=vmax, origin="lower",
                       interpolation="nearest")
        ax.set_title(key)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=7)
    return _save(fig, path)


def dist_matrix(structs, track, bins, pos_of):
    """返回指定 track 在 `bins` 上的 dense distance matrix；缺失位置为 NaN。"""
    n = len(bins)
    M = np.full((n, n), np.nan)
    m = structs.get(track, {})
    for i in range(n):
        for j in range(i, n):
            a, b = pos_of(bins[i]), pos_of(bins[j])
            if a in m and b in m:
                M[i, j] = M[j, i] = float(np.linalg.norm(m[a] - m[b]))
    return M
