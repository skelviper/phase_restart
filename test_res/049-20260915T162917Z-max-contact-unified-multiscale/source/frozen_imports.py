"""sys.path 引导：只读导入 045 冻结实现（不改旧文件字节）。"""
from __future__ import annotations

import sys

from round_paths import ROOT, SOURCE_045_SRC

for _path in (SOURCE_045_SRC, SOURCE_045_SRC / "frozen_035", SOURCE_045_SRC / "frozen_037",
              SOURCE_045_SRC / "frozen_pr", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import data_io  # noqa: E402,F401
import formal_controller  # noqa: E402,F401
import m1_preconditioner  # noqa: E402,F401
import shared_capture_objective  # noqa: E402,F401
from pr import contact_model, reconstruction_init  # noqa: E402,F401

__all__ = ["data_io", "formal_controller", "m1_preconditioner", "shared_capture_objective",
           "contact_model", "reconstruction_init"]
