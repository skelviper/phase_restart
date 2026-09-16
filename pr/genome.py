"""全基因组数据层：20 条染色体、cis 与 inter contacts、40 条轨迹。

轨迹命名为 `c{1-based chromosome index:02d}{a|b}`，例如 `c01a`、`c20b`。
reference 对同一对象使用 `chr1(mat)` / `chr1(pat)`；对应关系是 gauge，由几何解析，不能预先假定。
"""
import gzip
import os

import numpy as np

from .paths import BIN, OFF, SNPFREE
from . import pairs7

N_CHR = 20


def chrom_lengths(pairs_path):
    """返回文件顺序中的 [(name, length)]，来源是 `#chromosome` header 行。"""
    pairs7.assert_snpfree(pairs_path)
    out = []
    with gzip.open(pairs_path, "rt") as f:
        for line in f:
            if line.startswith("#chromosome:"):
                p = line.split()
                if len(p) != 3:
                    raise pairs7.PhaseLeakError(
                        "malformed #chromosome header in %s: %r" % (pairs_path, line.rstrip()))
                try:
                    length = int(p[2])
                except ValueError as exc:
                    raise pairs7.PhaseLeakError(
                        "non-integer chromosome length in %s: %r" % (pairs_path, line.rstrip())) from exc
                if not p[1] or length <= 0:
                    raise pairs7.PhaseLeakError(
                        "invalid chromosome header in %s: %r" % (pairs_path, line.rstrip()))
                out.append((p[1], length))
            elif not line.startswith("#"):
                break
    if not out:
        raise pairs7.PhaseLeakError("no #chromosome headers in %s" % pairs_path)
    if len({c for c, _ in out}) != len(out):
        raise pairs7.PhaseLeakError("duplicate chromosome headers in %s" % pairs_path)
    return out


def track(ci, k):
    """返回 chromosome index `ci`（从 0 开始）和 copy `k`（取 0 或 1）的轨迹名。"""
    return "c%02d%s" % (ci + 1, "ab"[k])


def load_all(path=SNPFREE):
    """以索引数组返回全部 cis 与 inter contacts。

    返回的 dict 包含：
        ci, p1, cj, p2：int32 数组，分别为每个端点的 chromosome index 和 position
        cis           ：bool，表示两个端点是否在同一条染色体上
    position 保持文件顺序（不跨 chromosome 重排：pairs 文件按 chr1-chr2-pos1-pos2 排序，因此 trans record 已满足 chr1 <= chr2）。
    """
    # `chrom_lengths` 会调用 assert_snpfree，下面的 iter_records 会验证每个 body row。
    # 两项检查都保留：含 phase 的 header 必须在解析 training-side record 之前失败。
    pairs7.assert_snpfree(path)
    names = [c for c, _ in chrom_lengths(path)]
    idx = {c: i for i, c in enumerate(names)}
    ci, p1, cj, p2 = [], [], [], []
    for row_no, c in enumerate(pairs7.iter_records(path), start=1):
        try:
            a, b = idx[c[1]], idx[c[3]]
        except KeyError as exc:
            raise pairs7.PhaseLeakError(
                "row %d references chromosome absent from #chromosome headers: %s"
                % (row_no, exc.args[0])) from exc
        x, y = int(c[2]), int(c[4])
        if a == b:
            x, y, _ = pairs7.canonical_cis(x, y)
        ci.append(a); p1.append(x); cj.append(b); p2.append(y)
    d = {"ci": np.asarray(ci, np.int32), "p1": np.asarray(p1, np.int64),
         "cj": np.asarray(cj, np.int32), "p2": np.asarray(p2, np.int64),
         "names": names}
    d["cis"] = d["ci"] == d["cj"]
    return d


def n_bins_per_chrom(lengths):
    return [int((L - OFF) // BIN) + 1 for _, L in lengths]


def bin_of(pos):
    return (np.asarray(pos) - OFF) // BIN


def global_bin(ci, pos, counts):
    """返回拼接染色体 bin 轴上的全局 bin index。"""
    off = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return (off[np.asarray(ci)] + bin_of(pos)).astype(np.int64)


def random_assignment(d, seed=0):
    """为每条 contact 返回 copy index（cis：分子对应一个 copy；inter：每个端点各取一个）。"""
    rng = np.random.default_rng(seed)
    n = len(d["ci"])
    k1 = rng.integers(0, 2, n)
    k2 = np.where(d["cis"], k1, rng.integers(0, 2, n))
    return k1.astype(np.int8), k2.astype(np.int8)
