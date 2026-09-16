"""两套拟合结构的刚体对齐与片段拼接。

这是 fragment-swap 实验的无标签交叉检查：不对 contacts 重标（跨 wall 的 pair 会含混），而是直接拼接两条已拟合轨迹。copy X 在翻转 block 外保留自己的坐标，在 block 内接收 copy Y 的坐标；先做一次全局刚体叠合，再把 Y 的坐标放入 Y 自己的坐标系。这样得到的是真实 chimera：局部正确、全局错接，而且 contact level 完全没有歧义。

结构尺度未校准，因此这里的比较都基于 rank，或在上述 Procrustes 对齐后进行（AGENTS.md reporting rule 6）。
"""
import numpy as np


def _common(pos_a, pos_b):
    return sorted(set(pos_a) & set(pos_b))


def _matrix(m, poses):
    return np.array([m[p] for p in poses], dtype=float)


def procrustes(src, dst):
    """计算将 src 点映射到 dst、且不缩放的刚体变换 (R, t)。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(A)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, mu_d - R @ mu_s


def apply_rigid(m, R, t):
    return {p: R @ v + t for p, v in m.items()}


def align_to(structs, moving_key, fixed_key):
    """将 `moving_key` 全局叠合到 `fixed_key`，返回新的 track dict。"""
    fixes = sorted(structs[fixed_key])
    common = _common(structs[moving_key], structs[fixed_key])
    if len(common) < 3:
        return None, None
    src = _matrix(structs[moving_key], common)
    dst = _matrix(structs[fixed_key], common)
    R, t = procrustes(src, dst)
    out = apply_rigid(structs[moving_key], R, t)
    # 保留 moving track 的每个 bead，并完成对齐
    return out, float(np.sqrt((( (src @ R.T + t) - dst) ** 2).sum(1).mean()))


def splice(structs, key_a, key_b, block_bins):
    """构造 chimera：各处使用 key_a，仅在 `block_bins` 使用 key_b 的对齐坐标。"""
    aligned_b, rmsd = align_to(structs, key_b, key_a)
    if aligned_b is None:
        return None, None
    out = {p: v.copy() for p, v in structs[key_a].items()}
    for b in block_bins:
        if b in aligned_b:
            out[b] = aligned_b[b].copy()
    return out, rmsd


def complement_splice(structs, key_a, key_b, block_bins):
    """返回 chimera pair 的另一半。"""
    return splice(structs, key_b, key_a, block_bins)
