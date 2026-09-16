"""052 输入准备：各层 aggregate + 链起点（40Mb）。

规则（config.json 冻结）：

* 40/10/5/2/1Mb 复用既有冻结 aggregate（1Mb 来自 045，20/10/5/2Mb 来自 051），逐个核 SHA；
* 500kb 与 200kb 只用原始 7 列无标签 pairs 的精确 bp 重新聚合，绝不把粗层 counts 拆成细 bin；
* 每层都必须满足 raw_records == 1,703,888，并且每层用整数算术折叠回 1Mb 后与冻结 1Mb aggregate
  逐数组完全一致（counts / diag_counts / endpoint_counts / group 汇总）；
* 链起点是 014 无标签 random 盲源（seed 2207）在 40Mb 完整网格上的展开（frozen 0.025*l0 扰动 + 半径 clip），
  p_init = 0.75，不做任何优化。

本进程只读无标签 contacts；不打开 phase 列，也不打开 reference 3DG。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bootstrap_052 import contact_model, data_io, reconstruction_init  # noqa: E402
from round_paths_052 import (AGGREGATE_051, AGGREGATE_1MB, AGGREGATE_1MB_SHA256, BASE_SEED,  # noqa: E402
                             EVAL_CHROMOSOMES, NEWLY_AGGREGATED_BINS, P_INIT, REUSED_051_BINS,
                             ROOT, RUN_014, SNPFREE, SNPFREE_SHA256, STAGES, stage_bin)

INPUTS = RUN / "inputs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    def jsonable(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {str(k): jsonable(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [jsonable(v) for v in item]
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def configure_init() -> None:
    reconstruction_init.ROOT = str(ROOT)
    reconstruction_init.DEFAULT_GATE_PATH = str(RUN_014 / "gate.json")
    reconstruction_init.DEFAULT_COORD_DIR = str(RUN_014 / "coords")
    for candidate, spec in list(reconstruction_init.APPROVED_SOURCES.items()):
        patched = dict(spec)
        patched["path"] = str(RUN_014 / "coords" / (candidate + ".3dg"))
        patched["gate_path"] = str(RUN_014 / "gate.json")
        reconstruction_init.APPROVED_SOURCES[candidate] = patched


# --------------------------------------------------------------------------------------
# 精确折叠：层 -> 更粗层（整数算术的 bin scatter-add，绝不是距离平均）
# --------------------------------------------------------------------------------------
def build_layer(names, lengths, bin_size: int, counts, diag_counts, endpoint_counts, template,
                raw_same_bin, raw_cis_offdiag, raw_inter, conditional_factorial_constant,
                diag_factorial_sum) -> object:
    """按 header 顺序构造一个完整网格的 AggregatedContacts 层（raw_integer 守恒）。

    所有 raw 预算与 factorial 常量都由调用方按本层分辨率显式给出：
    ``template`` 只提供 header 顺序（染色体名与长度），不再借用其它层的分辨率相关常量。
    """
    if len(names) != len(lengths):
        raise RuntimeError("layer header mismatch")
    if not np.array_equal(np.asarray(template.chromosome_lengths, dtype=np.int64),
                          np.asarray(lengths, dtype=np.int64)):
        raise RuntimeError("layer chromosome lengths do not match the header")
    lengths = [int(v) for v in lengths]
    n_bins = np.array([(L + int(bin_size) - 1) // int(bin_size) for L in lengths], dtype=np.int64)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(n_bins[:-1])))
    n_loci = int(n_bins.sum())
    locus_chromosome = np.repeat(np.arange(len(names), dtype=np.int32), n_bins)
    locus_bin = np.concatenate([np.arange(n, dtype=np.int64) for n in n_bins])
    pair_i64, pair_j64 = np.triu_indices(n_loci, k=1)
    pair_i = pair_i64.astype(np.int32, copy=False)
    pair_j = pair_j64.astype(np.int32, copy=False)
    cis_pair = (locus_chromosome[pair_i] == locus_chromosome[pair_j])
    exposure = np.sqrt(np.asarray(endpoint_counts, dtype=np.float64) + 10.0)
    exposure /= exposure.mean()
    counts = np.asarray(counts, dtype=np.int64)
    diag_counts = np.asarray(diag_counts, dtype=np.int64)
    # 未观测 pair 的 factorial 指纹逐分辨率重算（20 条染色体的 cis-offdiag / inter 分组不变）
    positive_counts = counts[counts > 1]
    constant = (-float(gammaln(float(raw_cis_offdiag) + 1.0))
                - float(gammaln(float(raw_inter) + 1.0))
                + float(gammaln(positive_counts.astype(np.float64) + 1.0).sum()))
    data = contact_model.AggregatedContacts(
        chromosome_names=tuple(str(n) for n in names),
        chromosome_lengths=np.asarray(lengths, dtype=np.int64),
        bin_size=int(bin_size), n_bins=n_bins, offsets=offsets,
        locus_chromosome=locus_chromosome, locus_bin=locus_bin,
        endpoint_counts=np.asarray(endpoint_counts, dtype=np.int64), exposure=exposure,
        pair_i=pair_i, pair_j=pair_j, cis_pair=cis_pair,
        counts=counts, diag_counts=diag_counts,
        raw_records=float(int(raw_same_bin) + int(raw_cis_offdiag) + int(raw_inter)),
        raw_same_bin=float(raw_same_bin), raw_cis_offdiag=float(raw_cis_offdiag),
        raw_inter=float(raw_inter),
        conditional_factorial_constant=float(conditional_factorial_constant),
        diag_factorial_sum=float(diag_factorial_sum))
    data.assert_consistent()
    budget = data.budget()
    total = (int(budget["aggregate_same_bin"]) + int(budget["aggregate_cis_offdiag"])
             + int(budget["aggregate_inter"]))
    if total != int(1_703_888) or int(budget["raw_records"]) != int(1_703_888):
        raise RuntimeError("layer %d does not conserve the 1703888 raw records" % bin_size)
    if int(budget["n_chromosomes"]) != 20:
        raise RuntimeError("layer %d lost chromosomes" % bin_size)
    if not np.isclose(constant, float(conditional_factorial_constant), rtol=0.0, atol=1e-6):
        raise RuntimeError("layer %d factorial constant is not reproducible from its own counts" % bin_size)
    return data


def fold_layer(source, bin_size: int) -> object:
    """把 source 层精确折叠到 bin_size 层（bin_size 必须是 source.bin_size 的整数倍）。"""
    ratio = int(bin_size) // int(source.bin_size)
    if ratio < 1 or ratio * int(source.bin_size) != int(bin_size):
        raise RuntimeError("fold requires a positive integer bin ratio")
    names = tuple(source.chromosome_names)
    lengths = [int(v) for v in source.chromosome_lengths]
    target_bins = [(L + int(bin_size) - 1) // int(bin_size) for L in lengths]
    source_bins = [int(v) for v in source.n_bins]
    for L, n_src, n_tgt in zip(lengths, source_bins, target_bins):
        if n_src != (L + int(source.bin_size) - 1) // int(source.bin_size):
            raise RuntimeError("source layer bin count mismatch")
        if n_tgt != -(-n_src // ratio):
            raise RuntimeError("target layer terminal bin count mismatch")
    target_n = np.array(target_bins, dtype=np.int64)
    target_offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(target_n[:-1])))
    n_loci = int(target_n.sum())
    # source locus -> target 全局 locus
    src_chrom = np.asarray(source.locus_chromosome, dtype=np.int64)
    src_local = np.asarray(source.locus_bin, dtype=np.int64)
    src_to_tgt = target_offsets[src_chrom] + (src_local // ratio)

    src_i = src_to_tgt[np.asarray(source.pair_i, dtype=np.int64)]
    src_j = src_to_tgt[np.asarray(source.pair_j, dtype=np.int64)]
    # 折叠到更粗层时，落在同一个粗 bin 内的细 pair 不再是一个 off-diagonal 记录；
    # 它的接触质量属于该粗 bin 的 within-bin（diag 层）预算，由折叠后的 diag_counts 承担。
    # 这里不把它塞进 target 的 off-diagonal，也不悄悄丢掉它的质量：单独记录被折叠的质量。
    src_counts = np.asarray(source.counts, dtype=np.int64)
    collapsed = src_i == src_j
    collapsed_mass = int(src_counts[collapsed].sum())
    collapsed_records = int(np.count_nonzero(collapsed))
    keep = ~collapsed
    flat = src_i[keep] * n_loci + src_j[keep]
    buffer = np.bincount(flat, weights=src_counts[keep].astype(np.float64),
                         minlength=n_loci * n_loci)

    locus_chromosome = np.repeat(np.arange(len(names), dtype=np.int32), target_n)
    locus_bin = np.concatenate([np.arange(n, dtype=np.int64) for n in target_n])
    pair_i64, pair_j64 = np.triu_indices(n_loci, k=1)
    pair_i = pair_i64.astype(np.int32, copy=False)
    pair_j = pair_j64.astype(np.int32, copy=False)
    raw_counts = buffer[pair_i.astype(np.int64) * n_loci + pair_j.astype(np.int64)]
    counts = np.rint(raw_counts).astype(np.int64)
    if not np.array_equal(raw_counts, counts.astype(np.float64)):
        raise RuntimeError("folded counts are not exact integers")
    if int(counts.sum()) + collapsed_mass != int(np.asarray(source.counts, dtype=np.int64).sum()):
        raise RuntimeError("fold lost off-diagonal contact mass")
    diag_counts = np.bincount(src_to_tgt, weights=np.asarray(source.diag_counts, dtype=np.int64),
                              minlength=n_loci).astype(np.int64)
    # 本层 diag 层 = 上一层 diag 层折叠后的质量 + 被折叠进同一粗 bin 的 cis 细 pair 质量。
    # 折叠不会把 cis 记录变成 inter，也不会把 inter 折叠进同 bin（不同染色体的 bin 不会相同）。
    collapsed_cis_mass = int(src_counts[collapsed & np.asarray(source.cis_pair, dtype=bool)].sum())
    if collapsed_cis_mass != collapsed_mass:
        raise RuntimeError("fold collapsed a non-cis pair into a same coarse bin")
    np.add.at(diag_counts, src_i[collapsed], src_counts[collapsed])
    folded_same_bin = int(diag_counts.sum())
    cis_pair = (locus_chromosome[pair_i] == locus_chromosome[pair_j])
    folded_cis_offdiag = int(counts[np.asarray(cis_pair, dtype=bool)].sum())
    folded_inter = int(counts[~np.asarray(cis_pair, dtype=bool)].sum())
    if folded_same_bin + folded_cis_offdiag + folded_inter != 1_703_888:
        raise RuntimeError("folded layer does not conserve the raw records")
    if folded_inter != int(source.raw_inter):
        raise RuntimeError("fold changed the inter-chromosomal budget")
    folded_meta = {"collapsed_source_pairs": collapsed_records, "collapsed_source_mass": collapsed_mass,
                   "folded_same_bin_budget": folded_same_bin,
                   "collapsed_cis_mass": collapsed_cis_mass}
    endpoint_counts = np.bincount(src_to_tgt, weights=np.asarray(source.endpoint_counts, dtype=np.int64),
                                  minlength=n_loci).astype(np.int64)
    positive = counts[counts > 1]
    constant = (-float(gammaln(float(folded_cis_offdiag) + 1.0))
                - float(gammaln(float(folded_inter) + 1.0))
                + float(gammaln(positive.astype(np.float64) + 1.0).sum()))
    diag_positive = diag_counts[diag_counts > 1]
    diag_factorial = float(gammaln(diag_positive.astype(np.float64) + 1.0).sum())
    folded_meta["conditional_factorial_constant"] = constant
    folded_meta["diag_factorial_sum"] = diag_factorial
    data = build_layer(names, lengths, bin_size, counts, diag_counts, endpoint_counts, source,
                       raw_same_bin=folded_same_bin, raw_cis_offdiag=folded_cis_offdiag,
                       raw_inter=folded_inter, conditional_factorial_constant=constant,
                       diag_factorial_sum=diag_factorial)
    if not np.array_equal(np.asarray(data.locus_chromosome, dtype=np.int32), locus_chromosome):
        raise RuntimeError("folded layer locus axis mismatch")
    if not np.array_equal(np.asarray(data.locus_bin, dtype=np.int64), locus_bin):
        raise RuntimeError("folded layer locus_bin mismatch")
    if not np.array_equal(np.asarray(data.cis_pair, dtype=bool), cis_pair):
        raise RuntimeError("folded layer cis_pair mismatch")
    return data


RAW_CACHE: dict = {}


def raw_contacts() -> dict:
    """原始 7 列无标签 pairs 的精确 bp（只读一次）。"""
    if "d" not in RAW_CACHE:
        from pr import genome
        contacts = genome.load_all(str(SNPFREE))
        RAW_CACHE["d"] = {k: np.asarray(contacts[k], dtype=np.int64)
                          for k in ("ci", "cj", "p1", "p2")}
    return RAW_CACHE["d"]


def raw_diagnostics(bin_size: int) -> dict:
    """直接从原始 7 列无标签 pairs 的精确 bp 计算本层 same-bin / cis-offdiag / inter 预算。"""
    d = raw_contacts()
    ci, cj, p1, p2 = d["ci"], d["cj"], d["p1"], d["p2"]
    cis = ci == cj
    same = cis & ((p1 // int(bin_size)) == (p2 // int(bin_size)))
    return {"raw_records": int(len(ci)), "raw_same_bin": int(same.sum()),
            "raw_cis_offdiag": int((cis & ~same).sum()), "raw_inter": int((~cis).sum())}


def fold_check(source, target) -> dict:
    """把两层放到同一个分辨率上逐数组比较（细层 -> 粗层折叠；粗层 -> 由 1Mb 折叠过来对照）。

    返回的 ``direction`` 说明这次比较是谁折叠到谁；两者在 counts / diag_counts /
    endpoint_counts 上必须完全一致（bp 映射正确性的主检查）。
    """
    if int(source.bin_size) < int(target.bin_size):
        folded, anchor = fold_layer(source, int(target.bin_size)), target
        direction = "source_folded_to_target"
    elif int(source.bin_size) > int(target.bin_size):
        folded, anchor = fold_layer(target, int(source.bin_size)), source
        direction = "target_folded_to_source"
    else:
        folded, anchor, direction = source, target, "same_resolution"
    checks = {}
    checks["direction"] = direction
    checks["source_bin_size_bp"] = int(source.bin_size)
    checks["target_bin_size_bp"] = int(target.bin_size)
    for field in ("counts", "diag_counts", "endpoint_counts"):
        left = np.asarray(getattr(folded, field), dtype=np.int64)
        right = np.asarray(getattr(anchor, field), dtype=np.int64)
        checks[field] = {"exact": bool(np.array_equal(left, right)),
                         "folded_sum": int(left.sum()), "anchor_sum": int(right.sum()),
                         "mismatch_count": int(np.count_nonzero(left != right))}
    checks["raw_budget"] = {
        "folded": [int(folded.raw_same_bin), int(folded.raw_cis_offdiag), int(folded.raw_inter)],
        "anchor": [int(anchor.raw_same_bin), int(anchor.raw_cis_offdiag), int(anchor.raw_inter)],
    }
    checks["n_loci"] = [int(folded.n_loci), int(anchor.n_loci)]
    checks["all_exact"] = bool(all(checks[f]["exact"] for f in ("counts", "diag_counts", "endpoint_counts")))
    if not checks["all_exact"]:
        raise RuntimeError("fold mismatch between bin %d and bin %d"
                           % (int(source.bin_size), int(target.bin_size)))
    if checks["n_loci"][0] != checks["n_loci"][1]:
        raise RuntimeError("fold grid size mismatch between bin %d and bin %d"
                           % (int(source.bin_size), int(target.bin_size)))
    if checks["raw_budget"]["folded"] != checks["raw_budget"]["anchor"]:
        raise RuntimeError("fold raw budget mismatch between bin %d and bin %d"
                           % (int(source.bin_size), int(target.bin_size)))
    return checks


def main() -> int:
    started = time.perf_counter()
    configure_init()
    actual_snpfree = sha256_file(SNPFREE)
    if actual_snpfree != SNPFREE_SHA256:
        raise RuntimeError("snpfree input SHA mismatch")
    frozen = data_io.load_aggregate(AGGREGATE_1MB)
    if sha256_file(AGGREGATE_1MB) != AGGREGATE_1MB_SHA256:
        raise RuntimeError("045 frozen 1Mb aggregate SHA mismatch")
    if int(frozen.n_loci) != 2645 or int(frozen.n_pairs) != 3_496_690:
        raise RuntimeError("frozen 1Mb grid is not the expected 2645 loci")

    layers = {1_000_000: frozen}
    records = {"1000000": {"path": str(AGGREGATE_1MB.relative_to(ROOT)), "sha256": AGGREGATE_1MB_SHA256,
                           "reused_from": "045", "n_loci": int(frozen.n_loci), "n_pairs": int(frozen.n_pairs)}}

    # 1) 复用 051 的粗层 aggregate（20/10/5/2Mb）
    for bin_size in REUSED_051_BINS:
        path = AGGREGATE_051[bin_size]
        data = data_io.load_aggregate(path)
        if int(data.bin_size) != int(bin_size):
            raise RuntimeError("051 aggregate bin_size mismatch for %d" % bin_size)
        layers[int(bin_size)] = data
        records[str(bin_size)] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                                  "reused_from": "051", "n_loci": int(data.n_loci),
                                  "n_pairs": int(data.n_pairs)}

    # 2) 40Mb 层：由冻结 1Mb aggregate 精确整数折叠
    layers[40_000_000] = fold_layer(frozen, 40_000_000)
    path_40 = INPUTS / "real_40000000_aggregate.npz"
    record_40 = data_io.save_aggregate(path_40, layers[40_000_000])
    record_40["budget"] = layers[40_000_000].budget()
    records["40000000"] = {**record_40, "path": str(path_40.relative_to(ROOT)),
                           "sha256": sha256_file(path_40),
                           "source": "exact integer fold of the 045 frozen 1Mb aggregate"}

    # 3) 500kb / 200kb 层：只用原始 7 列无标签 pairs 的精确 bp 重新聚合
    for bin_size in NEWLY_AGGREGATED_BINS:
        data = contact_model.load_frozen_p9016_aggregate(bin_size)
        path = INPUTS / ("real_%d_aggregate.npz" % bin_size)
        record = data_io.save_aggregate(path, data)
        record["budget"] = data.budget()
        layers[int(bin_size)] = data
        records[str(bin_size)] = {**record, "path": str(path.relative_to(ROOT)),
                                  "sha256": sha256_file(path),
                                  "source": "raw 7-column unlabeled bp re-aggregation"}

    # 4) 每层规模与守恒；raw 预算另行从原始 bp 独立重算（不在层之间互相比）
    raw_audit = {}
    conservation = {}
    for bin_size, data in sorted(layers.items()):
        raw = raw_diagnostics(bin_size)
        budget = data.budget()
        total = (int(budget["aggregate_same_bin"]) + int(budget["aggregate_cis_offdiag"])
                 + int(budget["aggregate_inter"]))
        conservation[str(bin_size)] = {
            "n_loci": int(data.n_loci), "n_pairs": int(data.n_pairs),
            "raw_records": int(budget["raw_records"]), "raw_same_bin": int(budget["raw_same_bin"]),
            "raw_cis_offdiag": int(budget["raw_cis_offdiag"]), "raw_inter": int(budget["raw_inter"]),
            "aggregate_same_bin": int(budget["aggregate_same_bin"]),
            "aggregate_cis_offdiag": int(budget["aggregate_cis_offdiag"]),
            "aggregate_inter": int(budget["aggregate_inter"]),
            "aggregate_total": total, "n_chromosomes": int(budget["n_chromosomes"]),
        }
        raw_audit[str(bin_size)] = {
            "from_raw_bp": raw, "layer": {k: int(budget[k]) for k in
                                          ("raw_records", "raw_same_bin", "raw_cis_offdiag", "raw_inter")},
            "matches": bool(raw["raw_records"] == int(budget["raw_records"])
                            and raw["raw_same_bin"] == int(budget["raw_same_bin"])
                            and raw["raw_cis_offdiag"] == int(budget["raw_cis_offdiag"])
                            and raw["raw_inter"] == int(budget["raw_inter"])),
        }
        if not raw_audit[str(bin_size)]["matches"]:
            raise RuntimeError("layer %d raw budget disagrees with the raw bp recomputation" % bin_size)
        if int(budget["raw_records"]) != 1_703_888 or total != 1_703_888:
            raise RuntimeError("layer %d lost records" % bin_size)
        if int(budget["n_chromosomes"]) != EVAL_CHROMOSOMES:
            raise RuntimeError("layer %d lost chromosomes" % bin_size)

    # 5) 与冻结 1Mb aggregate 的分辨率一致性（bp 映射正确性的主检查）
    #    * 500kb / 200kb 是本轮从原始 bp 新建的细层：折叠回 1Mb 必须逐数组一致；
    #    * 40Mb 是本轮由冻结 1Mb 折叠得到的粗层：从 1Mb 折叠过来必须逐数组一致；
    #    * 20/10/5/2Mb 是冻结文件整份复用（bincount 聚合，不做细化），只做 raw 预算核对。
    #    2Mb->1Mb 这类非整数比不能折叠，因此不设置跨 1Mb 的伪折叠检查。
    fold_checks = {}
    for bin_size in (500_000, 200_000):
        fold_checks[str(bin_size)] = fold_check(layers[bin_size], frozen)
    fold_checks["40000000"] = fold_check(frozen, layers[40_000_000])
    # 200kb 与 500kb 不是嵌套网格（500000/200000 = 2.5，无整数比），
    # 二者的共同参照只能是 1Mb：上面两条 1Mb 折叠检查已经覆盖。不做伪折叠。
    # 40Mb 层必须是真实的 40,000,000 bp 网格，而不是别的 stage 的标签
    grid_audit = {}
    for bin_size, data in sorted(layers.items()):
        expected_bins = np.array([(int(L) + int(bin_size) - 1) // int(bin_size)
                                  for L in data.chromosome_lengths], dtype=np.int64)
        expected_offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(expected_bins[:-1])))
        expected_locus_bin = np.concatenate([np.arange(n, dtype=np.int64) for n in expected_bins])
        grid_audit[str(bin_size)] = {
            "bin_size_bp": int(data.bin_size),
            "n_bins": [int(v) for v in data.n_bins],
            "expected_n_bins_ceil_length_over_bin": [int(v) for v in expected_bins],
            "offsets": [int(v) for v in data.offsets],
            "expected_offsets": [int(v) for v in expected_offsets],
            "n_loci": int(data.n_loci), "expected_n_loci": int(expected_bins.sum()),
            "bin_size_matches_layer_key": int(data.bin_size) == int(bin_size),
            "n_bins_match": bool(np.array_equal(np.asarray(data.n_bins, dtype=np.int64), expected_bins)),
            "offsets_match": bool(np.array_equal(np.asarray(data.offsets, dtype=np.int64), expected_offsets)),
            "locus_bin_match": bool(np.array_equal(np.asarray(data.locus_bin, dtype=np.int64), expected_locus_bin)),
            "locus_bin_bp_identity": bool(np.array_equal(
                np.asarray(data.locus_bin, dtype=np.int64) * int(data.bin_size),
                np.asarray(data.locus_bin, dtype=np.int64) * int(bin_size))),
        }
        row = grid_audit[str(bin_size)]
        if not (row["bin_size_matches_layer_key"] and row["n_bins_match"] and row["offsets_match"]
                and row["locus_bin_match"] and row["n_loci"] == row["expected_n_loci"]):
            raise RuntimeError("layer %d does not sit on its own real bp grid" % bin_size)
    if grid_audit["40000000"]["n_loci"] != 78:
        raise RuntimeError("40Mb layer is not the real 40,000,000 bp grid (expected 78 loci)")

    # 6) 链起点：40Mb 完整网格上的 014 random 盲源 + frozen 扰动（不优化）
    start_bin = stage_bin("40Mb")
    start_data = layers[start_bin]
    state = reconstruction_init.initialize_approved_candidate(
        "random", tuple(start_data.chromosome_names),
        tuple(int(v) for v in start_data.chromosome_lengths), start_bin)
    coordinates = np.asarray(state["coords"], dtype=np.float64)
    if coordinates.shape != (2, int(start_data.n_loci), 3):
        raise RuntimeError("40Mb start shape mismatch")
    contact_model.assert_inside_unit_ball(coordinates)
    raw_y = contact_model.sphere_inverse(coordinates)
    mapping_error = float(np.max(np.abs(contact_model.sphere_forward(raw_y) - coordinates)))
    if mapping_error > 1e-10:
        raise RuntimeError("40Mb start raw_y does not map back to coordinates")
    start_record = data_io.save_start(RUN / "coords" / "initial" / "random_40Mb.npz",
                                      coordinates, raw_y, P_INIT,
                                      {"candidate": "random", "stage": "40Mb",
                                       "source": "014 approved blind root (seed 2207) expanded to the 40Mb full grid",
                                       "no_optimization": True, "chain_root": True, "p_init": P_INIT,
                                       "n_loci": int(start_data.n_loci), "metadata": state["metadata"]})
    perturbed = int(np.asarray(state["metadata"]["perturbation"]["perturbed_coordinates"]))

    manifest = {
        "schema": "p9016-round052-prep-v1", "run_dir": str(RUN.relative_to(ROOT)),
        "snpfree_sha256": actual_snpfree, "frozen_1mb_aggregate_sha256": AGGREGATE_1MB_SHA256,
        "layers": {str(k): records[str(k)] for k in sorted(layers)},
        "conservation": conservation, "raw_budget_audit_from_pairs_file": raw_audit,
        "fold_to_1mb_checks": fold_checks,
        "grid_audit": grid_audit,
        "chain_start_40Mb": start_record, "chain_start_p_init": P_INIT, "chain_start_seed": BASE_SEED,
        "chain_start_perturbed_loci": perturbed,
        "chain_start_sphere_mapping_max_abs_error": mapping_error,
        "elapsed_sec": time.perf_counter() - started,
        "reference_opened": False, "phase_opened": False,
    }
    write_json(INPUTS / "prep_manifest.json", manifest)
    for stage in STAGES:
        data = layers[stage_bin(stage)]
        print("%-6s bin=%-9d n_loci=%-6d n_pairs=%-10d raw_records=%d" % (
            stage, stage_bin(stage), data.n_loci, data.n_pairs, int(data.budget()["raw_records"])))
    print("fold exact for every layer:", all(v["all_exact"] for v in fold_checks.values()))
    print("40Mb start coords sha256:", start_record["coordinate_sha256"])
    print("elapsed_sec", round(time.perf_counter() - started, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
