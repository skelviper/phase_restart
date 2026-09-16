"""小 fixture 构造：走与生产完全相同的 aggregate_from_arrays 路径。

只用于工程门与有限差分，不进入正式 12 fit。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from frozen_imports import contact_model


def small_fixture(bins: tuple[int, ...] = (4, 3, 5), bin_size: int = 1_000, seed: int = 4901,
                  records: int = 900) -> Any:
    names = tuple("chr%d" % (index + 1) for index in range(len(bins)))
    lengths = [int(value) * int(bin_size) for value in bins]
    rng = np.random.default_rng(int(seed))
    ci = rng.integers(0, len(bins), size=int(records))
    cj = rng.integers(0, len(bins), size=int(records))
    p1 = np.array([int(rng.integers(0, lengths[c])) for c in ci], dtype=np.int64)
    p2 = np.array([int(rng.integers(0, lengths[c])) for c in cj], dtype=np.int64)
    data = contact_model.aggregate_from_arrays(names, lengths, ci, p1, cj, p2, int(bin_size))
    data.assert_consistent()
    return data


def seed_theta(data: Any, p: float = 0.8, scale: float = 0.35, seed: int = 4902) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    y = rng.normal(size=(2, int(data.n_loci), 3)) * float(scale)
    coordinates = contact_model.sphere_forward(y)
    contact_model.assert_inside_unit_ball(coordinates)
    theta = np.concatenate((y.reshape(-1), np.asarray([contact_model.q_from_p(float(p))])))
    return theta


__all__ = ["small_fixture", "seed_theta"]
