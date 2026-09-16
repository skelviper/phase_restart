"""冻结导入引导：只读复用 045/049/051 训练侧实现，不改旧文件字节。

只暴露本链需要的东西：contact_model / data_io / reconstruction_init / formal_controller /
shared_capture_objective / round_runner 的独立函数（不导入 051 的 masked_objective）。
"""
from __future__ import annotations

import sys

from round_paths_052 import ROOT, S045, S049, SOURCE_045_SRC

for _path in (S049, SOURCE_045_SRC, SOURCE_045_SRC / "frozen_035", SOURCE_045_SRC / "frozen_037",
              SOURCE_045_SRC / "frozen_pr", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import data_io  # noqa: E402,F401
import formal_controller  # noqa: E402,F401
import round_runner  # noqa: E402,F401
import shared_capture_objective  # noqa: E402,F401
from pr import contact_model, reconstruction_init  # noqa: E402,F401
from shared_capture_objective import PenaltyWeights, SharedCaptureObjective  # noqa: E402,F401

from round_paths_052 import RUN  # noqa: E402


def install() -> None:
    """把 045 formal_controller 的输出目录指向本 052 运行目录（只在本进程内）。"""
    formal_controller.RUN = RUN


__all__ = ["contact_model", "data_io", "reconstruction_init", "formal_controller",
           "shared_capture_objective", "round_runner", "PenaltyWeights",
           "SharedCaptureObjective", "install", "RUN", "S045", "S049", "SOURCE_045_SRC", "ROOT"]
