"""坐标哈希 gate。

AGENTS.md 的硬边界是：“只有坐标写出并完成哈希后，evaluator 才能读取 phase 列和 3DG。”

本 gate 将这条规则落实为可执行约束，而不是口头承诺。除非调用者提供的 gate 已由某个写出并哈希坐标的 stage 打开，`pr.labels` 和 `pr.ref3dg` 都拒绝打开文件。
"""
import hashlib
import json
import os


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class EvalGate:
    """记录已写出的坐标，并在之后打开 label access。"""

    def __init__(self, log_path=None):
        self.entries = []          # {"stage", "tag", "path", "sha256"}
        self.armed = set()         # 现在可以读取 label 的 stage 名称
        self.log_path = log_path
        if log_path and os.path.exists(log_path):
            with open(log_path) as f:
                self.entries = json.load(f)

    def register(self, stage, tag, path):
        """对已写出的坐标文件计算哈希，并返回 digest。"""
        digest = sha256_file(path)
        self.entries.append({"stage": stage, "tag": tag,
                             "path": os.path.relpath(path),
                             "sha256": digest})
        self._flush()
        return digest

    def arm(self, stage):
        """允许指定 stage 读取 label / reference。"""
        if not self.entries:
            raise RuntimeError(
                "refusing to arm %r: no coordinates have been written and hashed" % stage)
        self.armed.add(stage)

    def require(self, stage):
        if stage not in self.armed:
            raise RuntimeError(
                "gate is closed for stage %r: labels and the reference 3DG may only be "
                "read after that stage's coordinates are written and hashed" % stage)

    def digest_of(self, tag):
        for e in self.entries:
            if e["tag"] == tag:
                return e["sha256"]
        return None

    def _flush(self):
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.entries, f, indent=2)
