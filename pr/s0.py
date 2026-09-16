"""Stage S0：全基因组等位拆分的三个参考点。

    consensus  每条染色体一条轨迹（20 条轨迹），全部 1.7M 条接触
    random     40 条轨迹，每条接触的 copy 随机抽取
    oracle     40 条轨迹，copy 取自分相标签（仅完整标注的记录）

每个候选都是在共享排除体积中独立进行的全基因组拟合。独立拟合不共享 reference 坐标系；比较使用评估器的几何规范。
`oracle` 是完整标签子集上的有标签拟合对照；它刻画这一固定输入与单次 FDG 配置的条件性拟合表现，不保证构成真实结构或接触信息的恢复上限。
"""
import os
import time

import numpy as np

from . import gwfdg
from .genome import track


def candidate_tracks(contacts, k1, k2):
    t1 = np.array([track(c, k) for c, k in zip(contacts["ci"], k1)], dtype=object)
    t2 = np.array([track(c, k) for c, k in zip(contacts["cj"], k2)], dtype=object)
    return t1, t2


def fit(name, contacts, lengths, k1, k2, workdir, coordsdir, n_iter=1000,
        bin_size="1m", keep=None, single=False, init=None, seed=1):
    """写出并拟合一个候选；返回 (structs, out_path, seconds, n_records)。"""
    pp = os.path.join(workdir, "%s.pairs.gz" % name)
    o3 = os.path.join(coordsdir, "%s.3dg" % name)
    if lengths and isinstance(lengths[0], (tuple, list)):
        lengths = [int(L) for _n, L in lengths]
    if single:
        t1 = np.array([track(c, 0) for c in contacts["ci"]], dtype=object)
        t2 = np.array([track(c, 0) for c in contacts["cj"]], dtype=object)
        gwfdg.write_pairs(pp, contacts, None, None, lengths, keep=keep, tracks=(t1, t2))
        n = len(contacts["ci"]) if keep is None else int(keep.sum())
    else:
        gwfdg.write_pairs(pp, contacts, k1, k2, lengths, keep=keep)
        n = len(contacts["ci"]) if keep is None else int(keep.sum())
    _, dt = gwfdg.run(pp, o3, n_iter=n_iter, bin_size=bin_size, init=init, seed=seed,
                       log=os.path.join(workdir, "%s.hickit.log" % name))
    return gwfdg.read_3dg(o3), o3, dt, n


def oracle_keep(a1, a2):
    """oracle 仅可使用两端都已标注的记录（报告规则 2）。"""
    return (a1 >= 0) & (a2 >= 0)


def oracle_copies(a1, a2):
    """每个 END 一个 copy index。cis 记录的两端不总在同一 copy（F1：349,025 条中有 6,114 条），且 76,438 条 inter 记录的两端属于不同 copy；若保留每条记录一个 label，会丢掉全部这些记录并静默压低上限。"""
    return a1.astype(np.int8), a2.astype(np.int8)
