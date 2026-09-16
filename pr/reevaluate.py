"""只评估、不重新拟合的全基因组 S0 冻结坐标读取器。

本模块绝不调用 FDG wrapper。它先根据 source run 的 gate 核验源坐标哈希，再打开 evaluator 侧的 phase labels 或 reference 3DG，最后写入新的正式运行目录；该目录包含 metrics、figures 和指向不可变坐标文件的 provenance。
"""
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from . import genome, gwfdg, pairs7, refeval, viz3d
from .gate import EvalGate, sha256_file
from .paths import PAIRS, REF3DG, ROOT, SNPFREE

REQUIRED_CANDIDATES = ("consensus", "random", "oracle")
FROZEN_COUNTS = {"n_contacts": 1_703_888, "n_cis": 1_135_454, "n_inter": 568_434,
                 "n_chromosomes": 20}


def _safe(value):
    """将 numpy/非有限值转换为严格兼容 JSON 的基础类型。"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _write_json(path, value):
    with open(path, "w") as out:
        json.dump(_safe(value), out, indent=2, sort_keys=True)
        out.write("\n")


def _git_info():
    def call(*args):
        proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
        return proc.stdout.strip() if proc.returncode == 0 else "<unavailable: %s>" % proc.stderr.strip()
    tracked = ["run.py"] + sorted(
        os.path.relpath(path, ROOT) for path in Path(ROOT, "pr").glob("*.py"))
    return {"head": call("git", "rev-parse", "HEAD"),
            "status_short": call("git", "status", "--short").splitlines(),
            "code_sha256": {path: sha256_file(os.path.join(ROOT, path)) for path in tracked}}


def default_output_dir():
    root = Path(ROOT) / "test_res"
    used = []
    for item in root.iterdir():
        prefix = item.name.split("-", 1)[0]
        if item.is_dir() and prefix.isdigit():
            used.append(int(prefix))
    next_id = max(used, default=0) + 1
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(root / ("%03d-%s-s0-reevaluated" % (next_id, stamp)))


def _manifest_versions():
    manifest_path = os.path.join(ROOT, "inputs", "manifest.json")
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    declared_source = os.path.realpath(os.path.join(ROOT, manifest["source"]))
    declared_output = os.path.realpath(os.path.join(ROOT, manifest["output"]))
    if declared_source != os.path.realpath(PAIRS):
        raise RuntimeError("manifest source %s is not the PAIRS path actually used by this project (%s)"
                           % (declared_source, PAIRS))
    if declared_output != os.path.realpath(SNPFREE):
        raise RuntimeError("manifest output %s is not the SNPFREE path subsequently loaded (%s)"
                           % (declared_output, SNPFREE))
    checks = []
    for role, path_key, hash_key in (("phase-bearing source", "source", "source_sha256"),
                                     ("SNP-free training input", "output", "output_sha256")):
        path = os.path.join(ROOT, manifest[path_key])
        actual = sha256_file(path)
        expected = manifest[hash_key]
        checks.append({"role": role, "path": manifest[path_key], "expected_sha256": expected,
                       "actual_sha256": actual, "matches": actual == expected})
        if actual != expected:
            raise RuntimeError("input SHA256 mismatch for %s: %s != %s"
                               % (path, actual, expected))
    pairs7.assert_snpfree(SNPFREE)
    return {"manifest": manifest, "checks": checks}


def _source_path(entry, source_dir):
    path = Path(entry["path"])
    if path.is_absolute():
        return path
    root_relative = Path(ROOT) / path
    if root_relative.exists():
        return root_relative
    return source_dir / path


def _expected_tracks(tag, n_chromosomes):
    if tag == "consensus":
        return {genome.track(ci, 0) for ci in range(n_chromosomes)}
    return {genome.track(ci, copy) for ci in range(n_chromosomes) for copy in (0, 1)}


def _coordinate_inventory(path, expected_tracks):
    """只读取可信的坐标语法，并拒绝格式错误或非有限记录。"""
    tracks = {}
    seen = set()
    malformed = []
    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 5:
                malformed.append("line %d has %d fields" % (line_no, len(fields)))
                continue
            try:
                pos = int(fields[1])
                xyz = [float(item) for item in fields[2:5]]
            except ValueError:
                malformed.append("line %d is not numeric" % line_no)
                continue
            if pos < 0 or not all(math.isfinite(item) for item in xyz):
                malformed.append("line %d has invalid position or coordinate" % line_no)
                continue
            key = (fields[0], pos)
            if key in seen:
                malformed.append("line %d duplicates %s:%d" % (line_no, fields[0], pos))
                continue
            seen.add(key)
            tracks[fields[0]] = tracks.get(fields[0], 0) + 1
    if malformed:
        raise RuntimeError("invalid coordinate file %s: %s" % (path, "; ".join(malformed[:5])))
    found = set(tracks)
    if found != expected_tracks:
        raise RuntimeError("coordinate track mismatch for %s: missing=%s unexpected=%s"
                           % (path, sorted(expected_tracks - found), sorted(found - expected_tracks)))
    if any(count == 0 for count in tracks.values()):
        raise RuntimeError("coordinate file %s has an empty track" % path)
    return {"path": os.path.relpath(path, ROOT), "tracks": len(tracks), "beads": int(sum(tracks.values())),
            "beads_per_track": dict(sorted(tracks.items())), "nonfinite_values": 0,
            "malformed_records": 0}


def _verify_source_coordinates(source):
    source_dir = Path(source).resolve()
    source_gate_path = source_dir / "gate.json"
    if not source_gate_path.exists():
        raise RuntimeError("source run has no gate.json: %s" % source_gate_path)
    with open(source_gate_path) as handle:
        entries = json.load(handle)
    by_tag = {}
    for entry in entries:
        if entry.get("tag") in REQUIRED_CANDIDATES:
            if entry["tag"] in by_tag:
                raise RuntimeError("source gate has duplicate tag %s" % entry["tag"])
            by_tag[entry["tag"]] = entry
    missing = [tag for tag in REQUIRED_CANDIDATES if tag not in by_tag]
    if missing:
        raise RuntimeError("source gate lacks required coordinate tags: %s" % missing)

    inventory = {}
    verified_entries = []
    for tag in REQUIRED_CANDIDATES:
        entry = by_tag[tag]
        path = _source_path(entry, source_dir)
        if not path.exists():
            raise RuntimeError("source coordinate is missing: %s" % path)
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            raise RuntimeError("source coordinate SHA256 mismatch for %s: %s != %s"
                               % (path, actual, entry.get("sha256")))
        inv = _coordinate_inventory(path, _expected_tracks(tag, 20))
        inv.update({"tag": tag, "source_gate_sha256": entry["sha256"], "actual_sha256": actual})
        inventory[tag] = inv
        verified_entries.append({"stage": "source-coordinate-verified", "tag": tag,
                                 "path": os.path.relpath(path, ROOT), "sha256": actual,
                                 "source_run": os.path.relpath(source_dir, ROOT)})
    return source_dir, inventory, verified_entries


def _bootstrap_chromosomes(values, seed=0, n_boot=10_000):
    values = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not len(values):
        return {"n_chromosomes": 0, "mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"n_chromosomes": int(len(values)), "mean": float(values.mean()),
            "lo": float(lo), "hi": float(hi),
            "interpretation": "chromosome resampling within one cell; technical/structural variation, not biological replication"}


def _paired_r3(a, b):
    """只在两候选都能解析的 fragment 上比较现有 R3 labels。"""
    if not (a.get("applicable") and b.get("applicable")):
        return {"n_common_fragments": 0, "a": None, "b": None, "delta_b_minus_a": None}
    ad = {item["grid_start_bin"]: item for item in a["detail"] if item.get("label") is not None}
    bd = {item["grid_start_bin"]: item for item in b["detail"] if item.get("label") is not None}
    common = sorted(set(ad) & set(bd))
    if not common:
        return {"n_common_fragments": 0, "a": None, "b": None, "delta_b_minus_a": None}
    def consistency(items, result):
        # 全局多数 label 平局时有两个同样有效的方向。其 gauge-invariant 一致性
        # 是两个方向的均值（=0.5），不能与 null label 比较。
        if result.get("global_label") is None:
            return 0.5
        return float(np.mean([items[key]["label"] == result["global_label"] for key in common]))
    aa = consistency(ad, a)
    bb = consistency(bd, b)
    return {"n_common_fragments": int(len(common)), "a": aa, "b": bb,
            "delta_b_minus_a": float(bb - aa),
            "note": "paired subset preserves each candidate's own majority-label definition"}


def _paired_r2(struct_a, struct_b, chrom_index, chrom_name, ref, n_bins, gauge_a, gauge_b):
    """在一组六轨迹共同的有限非对角 mask 上比较 R2 contrast。"""
    if not (gauge_a and gauge_a.get("applicable") and gauge_b and gauge_b.get("applicable")):
        return {"n_common_pairs": 0, "a": None, "b": None, "delta_b_minus_a": None}
    i, j = np.triu_indices(n_bins, k=1)
    values = []
    for structs in (struct_a, struct_b):
        values.extend([refeval.ours_dists(structs, chrom_index, copy, i, j, n_bins)
                       for copy in (0, 1)])
    values.extend([refeval.ref_dists(ref, chrom_name, copy, i, j, n_bins) for copy in (0, 1)])
    common = np.ones(len(i), dtype=bool)
    for value in values:
        common &= np.isfinite(value)
    if common.sum() < refeval.MIN_RHO_PAIRS:
        return {"n_common_pairs": int(common.sum()), "a": None, "b": None,
                "delta_b_minus_a": None}

    ref_mat, ref_pat = values[4][common], values[5][common]

    def contrast(left, right, gauge):
        r0m = refeval._rho(left[common], ref_mat)
        r0p = refeval._rho(left[common], ref_pat)
        r1m = refeval._rho(right[common], ref_mat)
        r1p = refeval._rho(right[common], ref_pat)
        direct = (r0m + r1p) / 2.0
        swapped = (r0p + r1m) / 2.0
        if gauge.get("orientation") == "direct":
            return float(direct - swapped)
        if gauge.get("orientation") == "swapped":
            return float(swapped - direct)
        # 几何平局：对两个带符号方向取均值，结果严格为零。
        return 0.0

    ca = contrast(values[0], values[1], gauge_a)
    cb = contrast(values[2], values[3], gauge_b)
    return {"n_common_pairs": int(common.sum()), "a": ca, "b": cb,
            "delta_b_minus_a": None if ca is None or cb is None else float(cb - ca),
            "note": "each candidate retains its geometry-fixed gauge on the shared pair set"}


def _candidate_summary(per_chromosome, candidate):
    r1, ref_ceiling, oracle_ceiling, r2, r3, walls = [], [], [], [], [], []
    r1_denominators = []
    for row in per_chromosome.values():
        values = row[candidate]
        metric = values["R1"]
        if metric.get("accuracy") is not None:
            r1.append(metric["accuracy"])
            ref_ceiling.append(metric.get("reference_ceiling"))
            oracle_ceiling.append(metric.get("oracle_fit_ceiling"))
            r1_denominators.append(metric.get("n_denominator", 0))
        if values["R2"].get("contrast") is not None:
            r2.append(values["R2"]["contrast"])
        if values["R3"].get("frac_consistent") is not None:
            r3.append(values["R3"]["frac_consistent"])
        if values["R3"].get("n_walls") is not None:
            walls.append(values["R3"]["n_walls"])
    mean = lambda xs: float(np.mean(xs)) if xs else None
    return {"R1_macro_mean": mean(r1), "R1_reference_ceiling_macro_mean": mean(ref_ceiling),
            "R1_oracle_fit_ceiling_macro_mean": mean(oracle_ceiling),
            "R1_chromosomes": len(r1), "R1_denominator_total": int(sum(r1_denominators)),
            "R1_denominators_by_chromosome": r1_denominators,
            "R2_contrast_macro_mean": mean(r2), "R2_chromosomes": len(r2),
            "R3_frac_consistent_macro_mean": mean(r3), "R3_n_walls_macro_mean": mean(walls),
            "R3_chromosomes": len(r3)}


def _comparison(per_chromosome, structs, ref, names, nbins):
    """比较 oracle 与 random，并报告成对 universe 和按染色体 bootstrap。"""
    rows = []
    r1_delta, r2_delta, r3_delta = [], [], []
    for ci, name in enumerate(names):
        row = per_chromosome[name]
        random = row["random"]
        oracle = row["oracle"]
        # random.R1 保存 oracle-fit score，且严格使用 random/ref/oracle 的共同记录集。
        r1 = {"n_common_records": random["R1"].get("n_denominator", 0),
              "random": random["R1"].get("accuracy"),
              "oracle": random["R1"].get("oracle_fit_ceiling")}
        r1["delta_oracle_minus_random"] = (None if r1["random"] is None or r1["oracle"] is None
                                            else float(r1["oracle"] - r1["random"]))
        r2 = _paired_r2(structs["random"], structs["oracle"], ci, name, ref, nbins[ci],
                        random["R2"].get("gauge"), oracle["R2"].get("gauge"))
        r3 = _paired_r3(random["R3"], oracle["R3"])
        rows.append({"chromosome": name, "R1": r1, "R2": r2, "R3_frac_consistent": r3})
        r1_delta.append(r1["delta_oracle_minus_random"])
        r2_delta.append(r2["delta_b_minus_a"])
        r3_delta.append(r3["delta_b_minus_a"])

    def aggregate(values, key):
        usable = [value for value in values if value is not None and np.isfinite(value)]
        return {"wins_oracle": int(sum(value > 0 for value in usable)),
                "ties": int(sum(value == 0 for value in usable)),
                "losses_oracle": int(sum(value < 0 for value in usable)),
                "bootstrap": _bootstrap_chromosomes(usable, seed=37 + key),
                "per_chromosome_delta": usable}
    return {"candidate": "oracle", "control": "random", "per_chromosome": rows,
            "R1": aggregate(r1_delta, 1), "R2": aggregate(r2_delta, 2),
            "R3_frac_consistent": aggregate(r3_delta, 3),
            "bootstrap_interpretation": "20 linked chromosomes from one cell; not biological replicate uncertainty"}


def _readme(result):
    source = result["source_coordinates"]
    summaries = result["candidate_summary"]
    comparison = result["comparison_oracle_vs_random"]
    lines = [
        "# S0 现有坐标重新评估",
        "",
        "**状态：** `evaluated-existing-coordinates`。本次运行未调用 FDG，也未重新拟合任何结构。",
        "",
        "## 冻结信息与来源",
        "",
        "- 源坐标：`%s`" % source["source_run"],
        "- 输入接触：总计 %d = %d cis + %d inter，20 条染色体。" % (
            result["cohort"]["n_contacts"], result["cohort"]["n_cis"], result["cohort"]["n_inter"]),
        "- 在读取定相标签或参考结构之前，已根据源门控完成坐标 SHA256 核验。",
        "- 源轨迹/珠点：" + "; ".join(
            "%s %d/%d" % (tag, source["inventory"][tag]["tracks"], source["inventory"][tag]["beads"])
            for tag in REQUIRED_CANDIDATES),
        "",
        "## 修正后的指标规则",
        "",
        "- R1 使用每条染色体四条轨迹的几何规范，并在共同的有限非对角分箱对上计算。`phase0=pat`、`phase1=mat`；真实标签从不选择候选方向。几何平局取两个方向的均值；几何不足时为 `null`。",
        "- 候选 R1、参考结构上限和 oracle-fit 上限共用每行声明的记录分母。日志中的 `n_cis` 不是 R1 分母。",
        "- Consensus 是单轨结构基线：两拷贝 R1/R2/R3 为 `applicable=false` / `null`；它的两条单轨参考 rho 仍保留在 R2 中。",
        "- R3 保留 20 Mb 局部标签的多数定义。平局或不可用的片段从分母中排除，并打断 `longest_run` / `n_walls` 的相邻关系。",
        "- 指标使用预注册的 3 Mb 坐标网格；网格外记录单独计数。三维散点图保留完整坐标轨迹，并在其 TSV 溯源信息中说明这一差异。",
        "",
        "## 宏观汇总",
        "",
        "| 候选 | R1 宏观均值 | 参考结构上限 | oracle-fit 上限 | R1 分母 | R2 对比 | R3 一致比例 | R3 墙数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    def fmt(value, digits=4):
        return "n/a" if value is None else ("%%.%df" % digits) % value
    for tag in REQUIRED_CANDIDATES:
        summary = summaries[tag]
        lines.append("| `%s` | %s | %s | %s | %d 条记录，覆盖 %d 条染色体 | %s | %s | %s |" % (
            tag, fmt(summary["R1_macro_mean"]), fmt(summary["R1_reference_ceiling_macro_mean"]),
            fmt(summary["R1_oracle_fit_ceiling_macro_mean"]), summary["R1_denominator_total"],
            summary["R1_chromosomes"], fmt(summary["R2_contrast_macro_mean"]),
            fmt(summary["R3_frac_consistent_macro_mean"], 3),
            fmt(summary["R3_n_walls_macro_mean"], 2)))
    lines.extend([
        "",
        "## Oracle 与 random",
        "",
        "三项比较指标都会在 `metrics/metrics.json` 中逐染色体声明成对分母。bootstrap 在同一细胞内重采样染色体，因此只描述技术/结构变异。",
        "",
        "| 指标 | oracle 胜 / 平 / 负 | 平均差值 | 按染色体 bootstrap 的 95% CI |",
        "| --- | ---: | ---: | --- |",
    ])
    for key, label in (("R1", "R1"), ("R2", "R2 对比（`contrast`）"), ("R3_frac_consistent", "R3 一致比例（`frac_consistent`）")):
        metric = comparison[key]
        boot = metric["bootstrap"]
        lines.append("| %s | %d / %d / %d | %s | [%s, %s] |" % (
            label, metric["wins_oracle"], metric["ties"], metric["losses_oracle"],
            fmt(boot["mean"]), fmt(boot["lo"]), fmt(boot["hi"])))
    lines.extend([
        "",
        "## 图件与数据",
        "",
        "- `plots/fig5_territories_3d.png`：reference、oracle（真实标签的评估上限）、random 和单轨 consensus。每个面板独立居中并缩放到单位 RMS；它们不是全局对齐的坐标系。",
        "- `plots/fig5_territories_3d.tsv`：每个绘制坐标的归一化、哈希、命令和网格说明。",
        "- `plots/fig6_scale_free_diagnostics.png`：无尺度同源体/染色体领地读出，使用拷贝特异 Rg 和所有异染色体拷贝质心对。",
        "- `plots/fig6_scale_free_diagnostics.tsv`：逐染色体的透明输入和溯源信息。",
        "  完整的 014 图形生成脚本及溯源信息不可用，因此这些值不能与其历史 fig6 值直接比较。",
        "",
        "## 解释边界",
        "",
        "这是对冻结 S0 坐标的评估修正，不是盲等位拷贝拆分恢复，也不是 L2 证据。没有任何基于参考结构的指标用于选择拟合结果、超参数或候选。",
        "",
    ])
    return "\n".join(lines)


def run(args):
    started = time.time()
    outdir = Path(args.out or default_output_dir()).resolve()
    if outdir.exists():
        raise RuntimeError("refusing to reuse formal evaluation directory: %s" % outdir)
    for sub in ("logs", "plots", "metrics", "results"):
        (outdir / sub).mkdir(parents=True, exist_ok=True)
    regression_log = getattr(args, "regression_log", None)
    if regression_log:
        source_log = Path(regression_log).resolve()
        if not source_log.is_file():
            raise RuntimeError("regression test log does not exist: %s" % source_log)
        shutil.copyfile(source_log, outdir / "logs" / "regression-tests.log")
    events = []

    def log(message):
        line = "[%s] %s" % (dt.datetime.now().strftime("%H:%M:%S"), message)
        print(line, flush=True)
        events.append(line)

    log("evaluation-only start; FDG fitting is disabled by this command")
    versions = _manifest_versions()
    log("input manifest SHA256 checks passed; strict seven-column schema accepted")
    source_dir, inventory, gate_entries = _verify_source_coordinates(args.source)
    _write_json(outdir / "logs" / "input-coordinate-verification.json", {
        "input_versions": versions,
        "source_run": os.path.relpath(source_dir, ROOT),
        "source_coordinate_inventory": inventory,
        "source_gate_entries_reverified": gate_entries,
        "result": "all manifest and source-coordinate SHA256 checks passed before evaluator access",
    })
    log("source 014 coordinate hashes and track inventories passed")
    source_gate_out = outdir / "gate.json"
    _write_json(source_gate_out, gate_entries)
    gate = EvalGate(str(source_gate_out))
    gate.arm(refeval.STAGE)
    log("new evaluation gate records verified existing coordinates; evaluator access armed")

    lengths = genome.chrom_lengths(SNPFREE)
    contacts = genome.load_all(SNPFREE)
    n_cis = int(contacts["cis"].sum())
    cohort = {"n_contacts": int(len(contacts["ci"])), "n_cis": n_cis,
              "n_inter": int(len(contacts["ci"]) - n_cis), "n_chromosomes": len(lengths)}
    if cohort != FROZEN_COUNTS:
        raise RuntimeError("frozen cohort mismatch: expected %s, observed %s" % (FROZEN_COUNTS, cohort))
    nbins = genome.n_bins_per_chrom(lengths)
    log("frozen cohort verified: 1703888 = 1135454 cis + 568434 inter")

    # 只有 source 坐标通过其 gate 后，才能读取 phase labels 和 reference。
    a1, a2 = refeval.load_labels_two(gate, contacts)
    lab = refeval.single_label(a1, a2)
    ref = refeval.load_reference(gate)
    reference_sha = sha256_file(REF3DG)
    structs = {tag: gwfdg.read_3dg(os.path.join(ROOT, inventory[tag]["path"]))
               for tag in REQUIRED_CANDIDATES}
    keep = (a1 >= 0) & (a2 >= 0)
    labelled = {"n_both_ends_labelled": int(keep.sum()),
                "n_labelled_cis": int((keep & contacts["cis"]).sum()),
                "n_labelled_inter": int((keep & ~contacts["cis"]).sum()),
                "n_labelled_inter_crosscopy": int((keep & ~contacts["cis"] & (a1 != a2)).sum())}
    log("phase/reference read only after gate; %d fully labelled records available to evaluator"
        % labelled["n_both_ends_labelled"])

    result = {
        "status": "evaluated-existing-coordinates",
        "fdg_refit_started": False,
        "source_coordinates": {"source_run": os.path.relpath(source_dir, ROOT), "inventory": inventory},
        "input_versions": versions,
        "reference": {"path": os.path.relpath(REF3DG, ROOT), "sha256": reference_sha},
        "cohort": cohort,
        "labelled_evaluator_records": labelled,
        "metric_grid": {"bin_bp": 1_000_000, "offset_bp": 3_000_000,
                        "description": "R1/R2/R3 use bins [3 Mb, chromosome header length); full-coordinate figures retain all beads"},
        "evaluation_policy": {
            "R1": "common finite labeled non-diagonal records; geometric gauge; phase0=pat phase1=mat; geometry tie averages orientations",
            "R2": "four correlations on one common finite non-diagonal bin-pair set; single consensus is not a two-copy contrast",
            "R3": "20 Mb local labels; unavailable/tied fragments excluded and break adjacency",
            "selection": "no candidate, fit or hyperparameter was selected from reference metrics",
        },
        "per_chromosome": {},
    }

    refeval.clear_cache()
    for ci, (chrom, _length) in enumerate(lengths):
        selected = np.where(contacts["cis"] & (contacts["ci"] == ci))[0]
        b1 = genome.bin_of(contacts["p1"][selected])
        b2 = genome.bin_of(contacts["p2"][selected])
        grid = ((b1 >= 0) & (b2 >= 0) & (b1 < nbins[ci]) & (b2 < nbins[ci]))
        row = {"n_cis_records": int(len(selected)), "n_metric_grid_records": int(grid.sum()),
               "n_out_of_grid_records": int((~grid).sum()), "n_bins_metric_grid": int(nbins[ci])}
        r2_by_candidate = {}
        for tag in REQUIRED_CANDIDATES:
            r2_by_candidate[tag] = refeval.r2_table(structs[tag], ci, chrom, ref, nbins[ci])
        oracle_gauge = r2_by_candidate["oracle"].get("gauge")
        for tag in REQUIRED_CANDIDATES:
            r1 = refeval.r1_accuracy(structs[tag], ci, chrom, lab[selected], b1, b2, nbins[ci],
                                     ref=ref, gauge=r2_by_candidate[tag].get("gauge"),
                                     oracle_structs=structs["oracle"], oracle_gauge=oracle_gauge)
            r3 = refeval.r3_fragments(structs[tag], ci, chrom, ref, nbins[ci])
            row[tag] = {"R1": r1, "R2": r2_by_candidate[tag], "R3": r3}
        oracle_r1 = row["oracle"]["R1"]
        row["oracle_fit_ceiling"] = {"accuracy": oracle_r1.get("accuracy"),
                                      "reference_ceiling": oracle_r1.get("reference_ceiling"),
                                      "n_denominator": oracle_r1.get("n_denominator"),
                                      "gauge": oracle_r1.get("gauge")}
        result["per_chromosome"][chrom] = row
        log("%-5s cis=%6d grid=%6d out_grid=%4d | R1 denom random/oracle=%d/%d | R2 random/oracle=%s/%s | R3 walls random/oracle=%s/%s"
            % (chrom, row["n_cis_records"], row["n_metric_grid_records"], row["n_out_of_grid_records"],
               row["random"]["R1"].get("n_denominator", 0), row["oracle"]["R1"].get("n_denominator", 0),
               "n/a" if row["random"]["R2"].get("contrast") is None else "%.4f" % row["random"]["R2"]["contrast"],
               "n/a" if row["oracle"]["R2"].get("contrast") is None else "%.4f" % row["oracle"]["R2"]["contrast"],
               row["random"]["R3"].get("n_walls"), row["oracle"]["R3"].get("n_walls")))

    result["candidate_summary"] = {tag: _candidate_summary(result["per_chromosome"], tag)
                                   for tag in REQUIRED_CANDIDATES}
    result["comparison_oracle_vs_random"] = _comparison(result["per_chromosome"], structs, ref,
                                                          [name for name, _ in lengths], nbins)
    log("macro summaries and paired chromosome bootstrap comparisons completed")

    command = " ".join(sys.argv)
    provenance = {"command": command, "status": result["status"], "source_coordinates": result["source_coordinates"],
                  "reference_sha256": reference_sha, "metric_grid": result["metric_grid"],
                  "gauge": result["evaluation_policy"]["R1"]}
    panels = [("reference", ref, "reference"), ("oracle", structs["oracle"], "oracle"),
              ("random", structs["random"], "random"),
              ("consensus", structs["consensus"], "consensus")]
    plot_dir = outdir / "plots"
    result["fig5"] = viz3d.write_territory_figure(str(plot_dir / "fig5_territories_3d.png"),
                                                     str(plot_dir / "fig5_territories_3d.tsv"),
                                                     panels, [name for name, _ in lengths], provenance)
    result["fig6"] = viz3d.write_scale_free_diagnostics(str(plot_dir / "fig6_scale_free_diagnostics.png"),
                                                          str(plot_dir / "fig6_scale_free_diagnostics.tsv"),
                                                          panels, [name for name, _ in lengths], provenance)
    log("fig5/fig6 and transparent data tables written")

    result["runtime_sec"] = time.time() - started
    result["provenance"] = {"command": command, "git": _git_info(),
                             "source_gate_verified": True,
                             "reference_read_after_coordinate_gate": True}
    _write_json(outdir / "metrics" / "metrics.json", result)
    _write_json(outdir / "results" / "summary.json", {
        "status": result["status"], "fdg_refit_started": False, "candidate_summary": result["candidate_summary"],
        "comparison_oracle_vs_random": result["comparison_oracle_vs_random"],
        "source_coordinates": result["source_coordinates"], "runtime_sec": result["runtime_sec"]})
    _write_json(outdir / "results.json", result)
    _write_json(outdir / "config.json", {"cmd": "reevaluate", "source": args.source,
                                           "out": str(outdir), "command": command,
                                           "regression_log": regression_log,
                                           "fdg_refit_started": False})
    (outdir / "README.md").write_text(_readme(_safe(result)))
    log("STATUS: evaluated-existing-coordinates; no FDG refit was started")
    (outdir / "logs" / "run.log").write_text("\n".join(events) + "\n")
    return str(outdir), result
