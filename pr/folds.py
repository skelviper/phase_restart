"""根据数值 bin 对的混合哈希划分训练/验证/测试数据折。

评审缺陷 2：`(i*7919 + j*104729) % 2` 会退化为 `(i+j) % 2`，也就是基因组间距的奇偶性，从而把所有同区间对都送入同一个数据折。下面的哈希同时对两个索引做乘法混合再雪崩混合，因此不由 i+j、|i-j| 或任一单独索引决定数据折编号。
"""
import numpy as np

from .paths import N_FOLD, TEST_FOLDS, TRAIN_FOLDS, VAL_FOLDS

_M1 = np.uint64(0x9E3779B1)
_M2 = np.uint64(0x85EBCA77)
_M3 = np.uint64(0x2545F491)
_C = np.uint64(0x165667B1)
_S15 = np.uint64(15)
_S13 = np.uint64(13)


def fold_of_scalar(i, j):
    """保留精确的标量参考实现，供向量化版本测试。

    uint64 运算会回绕；numpy 会对此发出警告，这是预期行为，且与 benchmark_v1.py 完全一致。
    """
    with np.errstate(over="ignore"):
        h = (np.uint64(i) * _M1) ^ (np.uint64(j) * _M2)
        h ^= np.uint64(i * j + 0x165667B1)
        h = np.uint64(h) ^ (np.uint64(h) >> _S15)
        h = np.uint64(h) * _M3
        h = np.uint64(h) ^ (np.uint64(h) >> _S13)
    return int(h % np.uint64(N_FOLD))


def fold_of_array(i, j):
    """从 bin 索引整数数组生成 [0, N_FOLD) 内的数据折编号。

    该函数与 `docs/exploration/benchmark_v1.py` 按字节一致，因此 Stage 1 数值可以直接和冻结的 Stage 0 基线比较。
    """
    i = np.asarray(i, dtype=np.uint64)
    j = np.asarray(j, dtype=np.uint64)
    h = i * _M1 ^ j * _M2
    h = h ^ (i * j + _C)
    h = h ^ (h >> _S15)
    h = h * _M3
    h = h ^ (h >> _S13)
    return (h % np.uint64(N_FOLD)).astype(np.int64)


def fold_of(i, j):
    return int(fold_of_array(np.array([i]), np.array([j]))[0])


def masks(folds):
    return (np.isin(folds, TRAIN_FOLDS),
            np.isin(folds, VAL_FOLDS),
            np.isin(folds, TEST_FOLDS))


def sanity_check(n=20000, seed=0):
    """断言该哈希不受基因组间距混淆。"""
    rng = np.random.default_rng(seed)
    # 每个 fold 都必须收到同区间对：直接枚举这些区间对
    i0 = np.arange(0, 200)
    f0 = fold_of_array(i0, i0)
    cnt0 = np.bincount(f0, minlength=N_FOLD)
    assert cnt0.min() > 0, cnt0
    # fold 分布不应能由距离的奇偶性预测
    i = rng.integers(0, 200, n)
    d = rng.integers(0, 200, n)
    j = i + d
    f = fold_of_array(i, j)
    for p in (0, 1):
        c = np.bincount(f[(d % 2) == p], minlength=N_FOLD) / max(1, (d % 2 == p).sum())
        assert abs(c.max() - c.min()) < 0.05, c
    # fold 不能只由 |i-j| 决定
    for dd in (1, 2, 5, 20, 50):
        fd = fold_of_array(i, i + dd)
        assert len(np.unique(fd)) >= 5, (dd, np.bincount(fd))
    return True


def regression_against_reference(n=3000, seed=1):
    """断言向量化哈希与 benchmark_v1 的标量哈希完全一致。"""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, 200, n)
    j = rng.integers(0, 200, n)
    v = fold_of_array(i, j)
    for k in range(n):
        assert int(v[k]) == fold_of_scalar(int(i[k]), int(j[k])), (int(i[k]), int(j[k]))
    return True
