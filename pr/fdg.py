"""随仓库提供的 native hickit FDG 引擎封装。

native core 与 /work/phase3/hickit 字节一致，绝不编辑；这里的内容全部是进程衔接。

用法模式（兼容 benchmark_v1）：
    write_pairs(path, chrom, pos1, pos2, assign)   # assign None -> 单轨
    run_fdg(path, out3dg, seed=1)                  # hickit -P1 -s SEED -i .. -b1m -O ..
    read_3dg(out3dg)                               # {track: {pos: xyz}}
"""
import gzip
import os
import subprocess

import numpy as np

from .paths import HICKIT, safe_name


def write_pairs(path, chrom, p1, p2, assign=None, span=300_000_000):
    """写出上三角的单轨或双轨 pairs 文件。

    assign=None  -> 名为 `chrom` 的单轨
    assign=k     -> 双轨 pair 的第 k 个 copy（k 属于 {0,1}）
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tracks = [chrom] if assign is None else [safe_name(chrom, 0), safe_name(chrom, 1)]
    with gzip.open(path, "wt") as f:
        f.write("## pairs format v1.0\n")
        f.write("#sorted: chr1-chr2-pos1-pos2\n#shape: upper triangle\n")
        for t in tracks:
            f.write("#chromosome: %s %d\n" % (t, span))
        f.write("#columns:readID\tchr1\tpos1\tchr2\tpos2\tstrand1\tstrand2\n")
        if assign is None:
            for a, b in zip(p1, p2):
                f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (chrom, a, chrom, b))
        else:
            for a, b, k in zip(p1, p2, assign):
                nm = tracks[int(k)]
                f.write(".\t%s\t%d\t%s\t%d\t+\t+\n" % (nm, a, nm, b))
    return path


def run_fdg(pairs_path, out_path, seed=1, max_iter=None):
    # hickit 在解析 -b 时执行 FDG，因此所有 fit 选项都必须置于其前。
    cmd = [HICKIT, "-P1", "-s", str(seed), "-i", pairs_path]
    if max_iter is not None:
        cmd += ["-n", str(max_iter)]
    cmd += ["-b1m", "-O", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("hickit failed (%d): %s" % (r.returncode, r.stderr[-500:]))
    return out_path


def read_3dg(path):
    """返回 {track_name: {bin_start_pos: np.array([x,y,z])}}。"""
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


def fit(path, chrom, p1, p2, assign=None, seed=1, out_dir=None, tag="fit"):
    """一次调用完成写出、拟合和解析；返回 (structures 字典, out_path)。"""
    out_dir = out_dir or os.path.dirname(os.path.abspath(path))
    pp = os.path.join(out_dir, tag + ".pairs.gz")
    o3 = os.path.join(out_dir, tag + ".3dg")
    write_pairs(pp, chrom, p1, p2, assign)
    run_fdg(pp, o3, seed=seed)
    return read_3dg(o3), o3


def pair_distances(structs, track, p1, p2, snap=True):
    """返回 `track` 中每个 (p1, p2) pair 的距离；bin 缺少 bead 时为 NaN。

    native 引擎在每个 1 Mb bin 起点写出一个 bead，因此查找前会将原始 contact positions 向下 snap 到 bin 网格。OFF 和 BIN 都是 BIN 的倍数，所以该转换是精确的。
    """
    from .paths import BIN
    m = structs.get(track, {})
    out = np.full(len(p1), np.nan)
    cache = {}
    for idx, (a, b) in enumerate(zip(p1, p2)):
        ka = int(a) // BIN * BIN if snap else int(a)
        kb = int(b) // BIN * BIN if snap else int(b)
        if ka not in cache:
            cache[ka] = m.get(ka)
        if kb not in cache:
            cache[kb] = m.get(kb)
        va, vb = cache[ka], cache[kb]
        if va is None or vb is None:
            continue
        out[idx] = float(np.linalg.norm(va - vb))
    return out
