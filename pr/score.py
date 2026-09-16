"""留出评分：汇总 Spearman、分层读出和配对 bootstrap 置信区间。

候选划分的预测量是 `d_A^-a + d_B^-a`（两套结构），或 `d_Z^-a`（一套结构），与留出数据折中的 bin 对计数比较。`a` 只能在 VALIDATION 数据折上选择。

同区间（b == b'）区间对不含结构信息：从所有结构指标中排除，并单独报告（AGENTS.md corrected protocol 4）。
"""
import numpy as np
from scipy.stats import rankdata

from .paths import ALPHAS, BIN


def spearman(x, y):
    """通过平均秩计算 Spearman rho；任一侧为常量时返回 NaN。"""
    rx = rankdata(x)
    ry = rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def predictor(d, alpha, floor=1e-3):
    return np.maximum(d, floor) ** (-alpha)


class TestSet:
    """返回用于报告的留出 bin pairs、计数和分层。"""

    def __init__(self, bins1, bins2, counts, frag1, frag2):
        self.b1 = bins1
        self.b2 = bins2
        self.C = counts.astype(float)
        self.f1 = frag1
        self.f2 = frag2
        self.sep = np.abs(bins1 - bins2)
        self.n = len(bins1)
        self.masks = {
            "all": np.ones(self.n, dtype=bool),
            "intra_frag": frag1 == frag2,
            "xshort": (frag1 != frag2) & (self.sep <= 40),
            "xlong": (frag1 != frag2) & (self.sep > 40),
        }

    def add_mechanistic_masks(self, t):
        """受影响：恰有一个端点位于翻转的片段中。"""
        a, b = t[self.f1], t[self.f2]
        self.masks["unaffected"] = (a == b)
        self.masks["affected"] = (a != b)

    def report(self, pred, masks=None):
        out = {}
        for name, m in (masks or self.masks).items():
            idx = np.where(m & np.isfinite(pred))[0]
            if len(idx) < 20 or np.std(pred[idx]) == 0:
                out[name] = (float("nan"), int(len(idx)))
                continue
            out[name] = (spearman(pred[idx], self.C[idx]), int(len(idx)))
        return out


def select_alpha(val_pred_parts, val_counts, alphas=ALPHAS):
    """只在 VALIDATION fold 上选择指数（绝不使用 test）。"""
    best, best_a = -np.inf, alphas[0]
    for a in alphas:
        pred = predictor(val_pred_parts[0], a)
        if len(val_pred_parts) > 1:
            pred = pred + predictor(val_pred_parts[1], a)
        r = spearman(pred, val_counts)
        if np.isfinite(r) and r > best:
            best, best_a = r, a
    return best_a, best


def paired_bootstrap(C, pred_ref, pred_alt, n_boot=2000, seed=0, alpha=95):
    """在成对重采样的 bin pairs 上，计算 rho(ref) - rho(alt) 的置信区间。"""
    n = len(C)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    k = 0
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        r1 = spearman(pred_ref[b], C[b])
        r2 = spearman(pred_alt[b], C[b])
        if np.isfinite(r1) and np.isfinite(r2):
            d[k] = r1 - r2
            k += 1
    if k < 100:
        return float("nan"), float("nan"), np.array([])
    d = d[:k]
    lo, hi = np.percentile(d, [(100 - alpha) / 2, 100 - (100 - alpha) / 2])
    return float(lo), float(hi), d


def paired_bootstrap_gaugeavg(C, ref_a, ref_b, alt_a, alt_b, n_boot=2000, seed=0, alpha=95):
    """计算 mean(rho(ref_a), rho(ref_b)) - mean(rho(alt_a), rho(alt_b)) 的置信区间。

    `copyA`/`copyB` 是 gauge labels。候选及其 label-complement 属于同一假设，因此 gauge-invariant 读出取两者 rho 的均值；比较均值还会抵消引擎在两条 tracks 之间的一阶不对称。
    """
    n = len(C)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    k = 0
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        ra = (spearman(ref_a[b], C[b]) + spearman(ref_b[b], C[b])) / 2.0
        rb = (spearman(alt_a[b], C[b]) + spearman(alt_b[b], C[b])) / 2.0
        if np.isfinite(ra) and np.isfinite(rb):
            d[k] = ra - rb
            k += 1
    if k < 100:
        return float("nan"), float("nan"), np.array([])
    d = d[:k]
    lo, hi = np.percentile(d, [(100 - alpha) / 2, 100 - (100 - alpha) / 2])
    return float(lo), float(hi), d


def bin_pair_counts(bins1, bins2, keep):
    """将接触记录聚合为带计数的唯一 bin pairs（保留同区间对，以便单独处理）。"""
    s = {}
    for a, b in zip(bins1[keep], bins2[keep]):
        k = (int(a), int(b))
        s[k] = s.get(k, 0) + 1
    K = np.array(sorted(s), dtype=np.int64)
    C = np.array([s[tuple(k)] for k in K], dtype=float)
    return K[:, 0], K[:, 1], C


def separation_profile(bins1, bins2):
    return np.abs(bins1 - bins2) * BIN
