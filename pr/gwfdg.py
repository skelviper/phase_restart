"""全基因组 FDG 封装：一个体积内包含 40 条轨迹（20 条染色体 × 2 个 copy）。

本模块编码了两个均已测量的引擎事实：

1. `hickit -b` 在**解析时**执行拟合，因此 `-n` 必须位于 `-b` **之前**
   （`-n 3000 -b1m`），否则迭代次数会静默保持为 1000。
2. `update_force` 中的 repulsion 项不检查 chromosome，因此全部 40 条轨迹共享一个
   排除体积；backbone force 保持在单条染色体内。这不是显式的有限
   nuclear-ball 约束，也不是径向 confinement potential。

测得成本（全部 1,703,888 条 contacts、40 条轨迹）：1 Mb 每 iteration 0.375 s（1000 次为 375 s），
2 Mb 每 iteration 0.118 s，5 Mb 每 iteration 0.022 s。
"""
import gzip
import os
import subprocess
import time

import numpy as np

from .genome import track
from .paths import HICKIT


def write_pairs(path, contacts, k1, k2, lengths, keep=None, tracks=None):
    """写出 N-track pairs 文件。

    contacts：来自 pr.genome.load_all 的 dict
    k1, k2：每条 contact 的 copy index（提供 `tracks` 时忽略）
    tracks：可选的 (track1, track2) 名称数组 pair，single-structure case 使用
    keep：对 contacts 的可选 boolean mask
    """
    ci, p1, cj, p2 = contacts["ci"], contacts["p1"], contacts["cj"], contacts["p2"]
    if keep is not None:
        ci, p1, cj, p2 = ci[keep], p1[keep], cj[keep], p2[keep]
    if tracks is None:
        if keep is not None:
            k1, k2 = k1[keep], k2[keep]
        need = sorted(set(ci.tolist()) | set(cj.tolist()))
        names = [track(c, k) for c in need for k in (0, 1)]
        t1 = np.array([track(c, k) for c, k in zip(ci, k1)], dtype=object)
        t2 = np.array([track(c, k) for c, k in zip(cj, k2)], dtype=object)
    else:
        t1, t2 = tracks
        if keep is not None:                      # 行必须与 positions 保持对齐
            t1, t2 = t1[keep], t2[keep]
        names = sorted(set(t1.tolist()) | set(t2.tolist()))
    with gzip.open(path, "wt") as f:
        f.write("## pairs format v1.0\n#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
        for nm in names:
            ci_of = int(nm[1:3]) - 1
            f.write("#chromosome: %s %d\n" % (nm, lengths[ci_of]))
        f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
        out = []
        for i in range(len(ci)):
            out.append(".\t%s\t%d\t%s\t%d\t+\t+\n" % (t1[i], p1[i], t2[i], p2[i]))
        f.write("".join(out))
    return path


def run(path, out_path, n_iter=1000, bin_size="1m", init=None, seed=1, log=None):
    """使用明确的 initialization seed 进行拟合；返回 (out_path, wall_seconds)。"""
    cmd = [HICKIT, "-P1", "-s", str(seed), "-i", path, "-n", str(n_iter)]
    if init:
        cmd += ["-I", init]
    cmd += ["-b" + bin_size, "-O", out_path]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t
    if r.returncode != 0:
        raise RuntimeError("hickit failed (%d): %s" % (r.returncode, r.stderr[-600:]))
    if log:
        with open(log, "w") as f:
            f.write(" ".join(cmd) + "\n" + r.stderr)
    return out_path, dt


def read_3dg(path):
    """返回 {track: {bin_start_pos: np.array([x,y,z])}}。"""
    d = {}
    with open(path) as f:
        for line in f:
            if line[0] == "#":
                continue
            a = line.split()
            if len(a) < 5:
                continue
            d.setdefault(a[0], {})[int(a[1])] = np.array([float(x) for x in a[2:5]])
    return d


def beads(structs, trk, positions):
    """在 `trk` 轨迹上堆叠 `positions` 的坐标；缺失行填 NaN。"""
    m = structs.get(trk, {})
    out = np.full((len(positions), 3), np.nan)
    for i, p in enumerate(positions):
        v = m.get(int(p))
        if v is not None:
            out[i] = v
    return out
