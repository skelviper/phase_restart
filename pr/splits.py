"""候选 contact 划分。

以下两类刻意分开：

BLIND（只能接触 SNP-free 表）
    consensus    一套结构，使用全部 contacts
    random       每条 contact 独立进行 Bernoulli(1/2) 划分

ORACLE INSTRUMENTS（由 phase labels 构建；从不是方法，而是上限或刻意损坏的上限）
    oracle        真实 phase label
    gauge_all     重标每个 bin；纯 gauge 变化，必须等于 oracle
    wall_m        翻转前 m 个完整 fragment（在 20*m Mb 处形成一个 domain wall）
    island_m      翻转连续的 m 个内部 fragment（形成两个 domain wall）
    alt           每隔一个 fragment 翻转（domain wall 数量最多）
    micro_k       翻转前 k 个 bin；粒度小于 fragment，用于寻找 detection limit

所有内容都表示为逐 bin 的布尔翻转 mask，因此 sub-fragment wall 不需要特殊分支。

domain wall 会让跨 wall 的 contacts 真正变得含混，因为 chimera 不是一对可实现的轨迹。因此实现并同时报告两条 contact-level 规则：

    relabel：仅当较小位置一端的 flip mask 被置位时翻转。
             这直接按“交换完整 fragment 的 A/B labels”解释：每个 copy 都会变成真实 A 与真实 B 的 chimera。在此规则下，候选及其 complement 是不同的 FDG 输入（该规则对 gauge 敏感）。
    rewire：两端的 mask 不一致时才翻转。每个 fragment 保留自己的正确内部几何，只打乱跨 wall 的连线。该规则对 mask 取 complement 完全不变，因此自身没有 gauge 歧义。
"""
import numpy as np

from .paths import FRAG_BINS


def n_fragments(n_bins):
    return (n_bins + FRAG_BINS - 1) // FRAG_BINS


def frag_of(bins, n_bins):
    f = np.asarray(bins, dtype=np.int64) // FRAG_BINS
    return np.clip(f, 0, n_fragments(n_bins) - 1)


def bin_sign(kind, n_bins):
    """返回逐 bin 翻转 mask（长度为 n_bins 的 bool 数组）。"""
    tb = np.zeros(n_bins, dtype=bool)
    nf = n_fragments(n_bins)
    if kind == "oracle":
        pass
    elif kind == "gauge_all":
        tb[:] = True
    elif kind.startswith("wall"):
        m = int(kind[4:])
        tb[:min(m * FRAG_BINS, n_bins)] = True
    elif kind.startswith("island"):
        m = int(kind[6:])
        c = (nf // 2) * FRAG_BINS
        tb[c:min(c + m * FRAG_BINS, n_bins)] = True
    elif kind == "alt":
        tb[:] = (np.arange(n_bins) // FRAG_BINS) % 2 == 1
    elif kind.startswith("micro"):
        k = int(kind[5:])
        tb[:min(k, n_bins)] = True
    else:
        raise ValueError("unknown flip pattern %r" % kind)
    return tb


def apply_flip(true_label, bins1, bins2, tb, rule="relabel"):
    """根据逐 bin 翻转 mask 返回一组候选 labels。"""
    a = tb[np.asarray(bins1, dtype=np.int64)]
    if rule == "relabel":
        flip = a
    elif rule == "rewire":
        flip = (a != tb[np.asarray(bins2, dtype=np.int64)])
    else:
        raise ValueError(rule)
    return true_label ^ flip.astype(true_label.dtype)


def random_split(n, seed):
    return np.random.default_rng(seed).integers(0, 2, n)


def n_walls(tb):
    """计算沿 bin 轴的符号变化次数。"""
    if tb.size < 2:
        return 0
    return int(np.sum(tb[1:] != tb[:-1]))


def flipped_mb(tb):
    return int(tb.sum())
