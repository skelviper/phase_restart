"""Coordinate-hash gate.

AGENTS.md hard boundary: "The evaluator may read phase columns and the 3DG only
after coordinates are written and hashed."

The gate makes that an executable rule rather than a promise. `pr.labels` and
`pr.ref3dg` refuse to open a file unless the caller presents a gate that has
already been armed by a stage which wrote and hashed its coordinates.
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
    """Records written coordinates and arms label access afterwards."""

    def __init__(self, log_path=None):
        self.entries = []          # {"stage", "tag", "path", "sha256"}
        self.armed = set()         # stage names that may now read labels
        self.log_path = log_path
        if log_path and os.path.exists(log_path):
            with open(log_path) as f:
                self.entries = json.load(f)

    def register(self, stage, tag, path):
        """Hash a written coordinate file. Returns the digest."""
        digest = sha256_file(path)
        self.entries.append({"stage": stage, "tag": tag,
                             "path": os.path.relpath(path),
                             "sha256": digest})
        self._flush()
        return digest

    def arm(self, stage):
        """Allow label / reference reads for the named stage."""
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
