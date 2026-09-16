"""SNP-free pairs I/O：严格七列，并硬拒绝 phase 字段。

`run.py prepare` 流式读取 `data/P9016.pairs.gz`，写出供所有训练侧步骤使用的七列文件。只要 `#columns` 行提到 phase 字段，加载器就会拒绝该文件，避免误读带 phase 的输入。
"""
import gzip
import os

from .paths import COL7, PAIRS, SNPFREE

FORBIDDEN = ("phase", "prob")


class PhaseLeakError(RuntimeError):
    pass


def _open(path, mode="rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def write_snpfree(src=PAIRS, dst=SNPFREE):
    """从 pairs 文件剥离 phase0/phase1 及所有 phase_prob* 列。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    header = []
    body_written = 0
    with _open(src) as fi, gzip.open(dst, "wt") as fo:
        keep = None
        for line in fi:
            if line.startswith("#"):
                if line.startswith("#columns:"):
                    names = line.rstrip("\n").split(":", 1)[1].strip().split("\t")
                    keep = [k for k, n in enumerate(names)
                            if not any(t in n.lower() for t in FORBIDDEN)]
                    kept = [names[k] for k in keep]
                    assert kept == COL7, (kept, COL7)
                    fo.write("#columns:" + "\t".join(kept) + "\n")
                else:
                    header.append(line)
                    fo.write(line)
                continue
            if keep is None:
                raise PhaseLeakError("no #columns line before the first record in %s" % src)
            c = line.rstrip("\n").split("\t")
            fo.write("\t".join(c[k] for k in keep) + "\n")
            body_written += 1
    return {"records": body_written, "columns": COL7}


def read_columns_line(path):
    with _open(path) as f:
        for line in f:
            if line.startswith("#columns:"):
                return line.rstrip("\n").split(":", 1)[1].strip().split("\t")
            if not line.startswith("#"):
                break
    raise PhaseLeakError("no #columns line in %s" % path)


def assert_snpfree(path):
    """硬拒绝：这里绝不能打开带 phase 的文件。"""
    cols = read_columns_line(path)
    bad = [c for c in cols if any(t in c.lower() for t in FORBIDDEN)]
    if bad:
        raise PhaseLeakError(
            "refusing to load %s: #columns mentions %s" % (path, bad))
    if cols != COL7:
        raise PhaseLeakError("refusing to load %s: columns are %s, expected %s"
                             % (path, cols, COL7))
    return True


def canonical_cis(pos1, pos2):
    """按数值升序返回 cis positions，并标记端点是否交换。

    携带逐端元数据的调用者必须在该标记为 true 时同步交换元数据。将规则集中在此处，可以防止位置规范化时把端点 label 与坐标悄悄拆开。
    """
    a, b = int(pos1), int(pos2)
    return (b, a, True) if a > b else (a, b, False)


def iter_records(path):
    """从 SNP-free pairs 文件逐条返回已验证的七列数据行。

    仅验证 header 不够：附加的第八列或被截断的数据行，否则可能通过位置数组索引进入训练。该读取器有意保持严格，供单染色体和全基因组加载器共同使用。
    """
    assert_snpfree(path)
    with _open(path) as f:
        for line_no, line in enumerate(f, start=1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(COL7):
                raise PhaseLeakError(
                    "refusing to load %s: row %d has %d columns, expected %d"
                    % (path, line_no, len(fields), len(COL7)))
            if not fields[1] or not fields[3]:
                raise PhaseLeakError(
                    "refusing to load %s: row %d has an empty chromosome field"
                    % (path, line_no))
            try:
                p1, p2 = int(fields[2]), int(fields[4])
            except ValueError as exc:
                raise PhaseLeakError(
                    "refusing to load %s: row %d has non-integer positions"
                    % (path, line_no)) from exc
            if p1 < 0 or p2 < 0:
                raise PhaseLeakError(
                    "refusing to load %s: row %d has a negative position"
                    % (path, line_no))
            yield fields


def load(path=SNPFREE, chrom=None):
    """读取 SNP-free 表，返回 (pos1, pos2) int64 数组，且 pos1 <= pos2。

    表中没有 label 列，也不能偷偷加入：先运行 `assert_snpfree`，并且文件必须严格包含七列。
    """
    p1, p2 = [], []
    for c in iter_records(path):
        if c[1] != c[3]:
            continue                           # 仅 cis（F1：intra 中的 98.2%）
        if chrom is not None and c[1] != chrom:
            continue
        a, b, _ = canonical_cis(c[2], c[4])   # 按数值排序，不按字符串顺序
        p1.append(a)
        p2.append(b)
    import numpy as np
    return np.asarray(p1, dtype=np.int64), np.asarray(p2, dtype=np.int64)
