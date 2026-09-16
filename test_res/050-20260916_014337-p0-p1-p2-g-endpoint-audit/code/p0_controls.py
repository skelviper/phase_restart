"""P0 对照与配对统计（050 轮）：fixed random / consensus / oracle 支持、u0 与 16 random-u 对照。

同一进程内：

1. 读取 5 个已冻结 P0 端点的全网格坐标，按 046 冻结规则生成 u0 与 16 个 random-u
   （seed 450500..450515，逐染色体内置换 u，保持 z），写成全网格 3DG 并哈希；
2. 把「5 端点 + 85 个 null」作为 candidate 交给 `endpoint_r1r3_eval.evaluate`
   （该入口会先把所有坐标哈希并写 pre_reference_gate，再 arm gate 读 phase/reference）；
3. 每个 null 用自己的可解析支持报 R1/R3/R2，不与主候选共同支持取交集；
4. 主比较的 paired bootstrap 复用 046 冻结的 seed 450301 / 10000×20 索引矩阵，
   固定 20 chr 分母，任一 chr undefined 时主值 None，同时给 defined-only 与逐 chr 胜出。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
ROOT = RUN_DIR.parents[1]
S046_EVAL = ROOT / "test_res/046-UTC-real-cell-shared-capture/evaluation_final"
S049 = ROOT / "test_res/049-20260915T162917Z-max-contact-unified-multiscale/source"
for _path in (str(HERE), str(S046_EVAL), str(S049)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import endpoint_r1r3_eval as ere  # noqa: E402
import evaluator as ev  # noqa: E402

ENDPOINTS = {
    "046-base-G-random": "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.3dg",
    "046-work-baseline-G-full-J": "test_res/046-UTC-real-cell-shared-capture/coords/real-extension-G-full-J/1Mb.3dg",
    "049-A-ms-random": "test_res/049-20260915T162917Z-max-contact-unified-multiscale/coords/A-ms-random/1Mb.3dg",
    "049-B-raw-consensus": "test_res/049-20260915T162917Z-max-contact-unified-multiscale/coords/B-raw-consensus/1Mb.3dg",
    "049-C-ms-random": "test_res/049-20260915T162917Z-max-contact-unified-multiscale/coords/C-ms-random/1Mb.3dg",
}
BASELINES = {
    "random": {"id": "fixed014-random", "role": "blind_baseline",
               "path": "test_res/014-20260912_153000-s0-genome-wide-fixed/coords/random.3dg",
               "sha256": "9a48d73e1401e18349d11758e679da4c76da0904dbc467979079cb54bcd567d7"},
    "consensus": {"id": "fixed014-consensus", "role": "blind_baseline",
                  "path": "test_res/014-20260912_153000-s0-genome-wide-fixed/coords/consensus.3dg",
                  "sha256": "e76655732deb6b8386b1b77bc76ff45d7dba1384f6931337fee80d8f4aaa8e02"},
    "oracle": {"id": "s0-oracle", "role": "evaluation_ceiling",
               "path": "test_res/014-20260912_153000-s0-genome-wide-fixed/coords/oracle.3dg",
               "sha256": "502cd64944dcbd278735b3c9df4d6df720907cedd6965c2f0cc728de6abbfc29"},
}
RANDOM_U_SEEDS = tuple(range(450500, 450516))
BOOTSTRAP_INDICES = S046_EVAL / "bootstrap_indices_seed450301_10000x20.npy"
GENOME_ORDER = ["chr%d" % k for k in range(1, 20)] + ["chrX"]
TIE_TOL = 1e-12

COMPARISONS = [
    ("046-base-G-random", "fixed_random", "046 base G-random minus fixed 014 random"),
    ("046-work-baseline-G-full-J", "fixed_random", "046 work baseline minus fixed 014 random"),
    ("049-A-ms-random", "fixed_random", "049 A-ms-random minus fixed 014 random"),
    ("049-B-raw-consensus", "fixed_random", "049 B-raw-consensus minus fixed 014 random"),
    ("049-C-ms-random", "fixed_random", "049 C-ms-random minus fixed 014 random"),
    ("049-A-ms-random", "046-work-baseline-G-full-J", "049 A-ms-random minus 046 work baseline"),
    ("049-B-raw-consensus", "046-work-baseline-G-full-J", "049 B-raw-consensus minus 046 work baseline"),
    ("049-C-ms-random", "046-work-baseline-G-full-J", "049 C-ms-random minus 046 work baseline"),
    ("046-base-G-random", "046-work-baseline-G-full-J", "046 base G-random minus 046 work baseline"),
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ev._jsonable(value), indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join("" if row.get(key) is None else str(row.get(key)) for key in columns) + "\n")


def load_full_grid(path: Path, lengths: Mapping[str, int]) -> np.ndarray:
    """读全网格 3DG 文本为 (2, n_loci, 3)，顺序按 GENOME_ORDER。"""
    tracks: dict[str, dict[int, np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 5:
                raise RuntimeError("bad 3DG row %d in %s" % (line_no, path))
            tracks.setdefault(fields[0], {})[int(fields[1])] = np.asarray(
                [float(x) for x in fields[2:5]], dtype=np.float64)
    total = sum(int(np.ceil(lengths[name] / 1_000_000)) for name in GENOME_ORDER)
    coords = np.full((2, total, 3), np.nan, dtype=np.float64)
    offset = 0
    for ci, name in enumerate(GENOME_ORDER):
        positions = list(range(0, lengths[name], 1_000_000))
        for copy, suffix in enumerate(("a", "b")):
            rows = tracks.get("c%02d%s" % (ci + 1, suffix), {})
            for local, position in enumerate(positions):
                point = rows.get(position)
                if point is not None:
                    coords[copy, offset + local] = point
        offset += len(positions)
    if not np.isfinite(coords).all():
        raise RuntimeError("coordinate payload is not full-grid finite: %s" % path)
    return coords


def write_full_grid(path: Path, coords: np.ndarray, lengths: Mapping[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        offset = 0
        for ci, name in enumerate(GENOME_ORDER):
            positions = list(range(0, lengths[name], 1_000_000))
            for copy, suffix in enumerate(("a", "b")):
                for local, position in enumerate(positions):
                    point = coords[copy, offset + local]
                    handle.write("c%02d%s\t%d\t%.17g\t%.17g\t%.17g\n"
                                 % (ci + 1, suffix, position, point[0], point[1], point[2]))
            offset += len(positions)


def chromosome_slices(lengths: Mapping[str, int]) -> list[slice]:
    slices, start = [], 0
    for name in GENOME_ORDER:
        count = len(range(0, lengths[name], 1_000_000))
        slices.append(slice(start, start + count))
        start += count
    return slices


def make_u_zero(coords: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    z = (coords[0] + coords[1]) / 2.0
    raw = np.stack((z, z), axis=0)
    pre = float(np.linalg.norm(raw.reshape(-1, 3), axis=1).max())
    scale = 1.0 if pre < 1.0 else (1.0 - 1e-6) / pre
    result = raw * scale
    return result, {"global_scale": scale, "pre_scale_radius": pre,
                    "post_scale_radius": float(np.linalg.norm(result.reshape(-1, 3), axis=1).max()),
                    "permutation": "u_zero", "z_fixed": True}


def make_random_u(coords: np.ndarray, slices: Sequence[slice], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    z = (coords[0] + coords[1]) / 2.0
    u = (coords[0] - coords[1]) / 2.0
    permuted = np.zeros_like(u)
    rng = np.random.default_rng(seed)
    for slc in slices:
        permuted[slc] = u[slc][rng.permutation(slc.stop - slc.start)]
    raw = np.stack((z + permuted, z - permuted), axis=0)
    pre = float(np.linalg.norm(raw.reshape(-1, 3), axis=1).max())
    scale = 1.0 if pre < 1.0 else (1.0 - 1e-6) / pre
    result = raw * scale
    return result, {"global_scale": scale, "pre_scale_radius": pre,
                    "post_scale_radius": float(np.linalg.norm(result.reshape(-1, 3), axis=1).max()),
                    "permutation": "within_chromosome_u", "z_fixed": True}


SOURCE_COLUMNS = {
    "fixed_random": {"R1": "r1_fixed_random",
                     "R2_contrast_spearman": "r2_fixed_random_contrast_spearman",
                     "R3_frac_consistent": "r3_fixed_random_frac_consistent",
                     "R3_n_walls": "r3_fixed_random_n_walls"},
    "oracle": {"R1": "r1_oracle_fit_ceiling", "R2_contrast_spearman": None,
               "R3_frac_consistent": None, "R3_n_walls": None},
    "reference_ceiling": {"R1": "r1_reference_ceiling", "R2_contrast_spearman": None,
                          "R3_frac_consistent": None, "R3_n_walls": None},
    "fixed_consensus": {"R1": None, "R2_contrast_spearman": "r2_consensus_contrast_spearman",
                        "R3_frac_consistent": "r3_consensus_frac_consistent",
                        "R3_n_walls": "r3_consensus_n_walls"},
}
CANDIDATE_COLUMNS = {"R1": "r1_accuracy", "R2_contrast_spearman": "r2_selected_contrast_spearman",
                     "R3_frac_consistent": "r3_frac_consistent", "R3_n_walls": "r3_n_walls"}


def series(rows: Sequence[Mapping[str, Any]], source: str, metric: str) -> list[Any]:
    """按固定 GENOME_ORDER 取一条 20 chr 序列。

    candidate id 用该候选自己的读出；`fixed_random` / `fixed_consensus` / `oracle` /
    `reference_ceiling` 从同一 `paired_r1` 结果的对应列取，因此与候选使用同一个共同记录支持。
    单轨 consensus 没有两 copy R1（R1 列为 None 是正确语义，不是缺失）。
    """
    lookup = {row["chromosome"]: row for row in rows if row["candidate_id"] == source}
    if lookup:
        column = CANDIDATE_COLUMNS[metric]
    else:
        column = SOURCE_COLUMNS[source][metric]
        if column is None:
            return [None] * len(GENOME_ORDER)
        lookup = {}
        for row in rows:
            lookup.setdefault(row["chromosome"], row)
    return [lookup.get(name, {}).get(column) for name in GENOME_ORDER]


def main() -> int:
    out = RUN_DIR / "p0" / "controls"
    out.mkdir(parents=True, exist_ok=True)
    data = ev._load_data()
    lengths = {str(name): int(length) for name, length in zip(data.chromosome_names, data.chromosome_lengths)}
    if list(lengths) != GENOME_ORDER:
        raise RuntimeError("chromosome order mismatch: %s" % list(lengths))
    if int(data.n_loci) != 2645:
        raise RuntimeError("n_loci mismatch: %d" % data.n_loci)
    slices = chromosome_slices(lengths)

    endpoint_coords = {}
    for name, relative in ENDPOINTS.items():
        endpoint_coords[name] = load_full_grid(ROOT / relative, lengths)

    null_dir = out / "nulls"
    null_rows = []
    for name, coords in endpoint_coords.items():
        variants = [("u_zero", None)] + [("random_u", seed) for seed in RANDOM_U_SEEDS]
        for kind, seed in variants:
            if seed is None:
                probe, audit = make_u_zero(coords)
                suffix = "u-zero"
            else:
                probe, audit = make_random_u(coords, slices, seed)
                suffix = "random-u-%d" % seed
            path = null_dir / ("%s__%s.3dg" % (name, suffix))
            write_full_grid(path, probe, lengths)
            null_rows.append({"source": name, "null_kind": kind, "seed": seed,
                              "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                              **audit})

    manifest = {
        "schema": ere.SCHEMA,
        "label": "p0-controls-u0-randomu",
        "raw_pairs_path": str(ev.REFERENCE_PATH.parent / "P9016.pairs.gz"),
        "raw_pairs_sha256": "071a6cc76bfad543ea1ace6ee1ce3022b30ac1b1e50a9f0c3a3a1967b9649505",
        "snpfree_path": str(ROOT / "inputs/P9016.snpfree.pairs.gz"),
        "snpfree_sha256": "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa",
        "reference_3dg_path": str(ev.REFERENCE_PATH),
        "reference_3dg_sha256": ev.REFERENCE_SHA256,
        "record_count": 1703888,
        "note": "u0 and 16 random-u controls for the five frozen P0 endpoints; the endpoints are repeated so "
                "every comparison uses the same code path and the same pre-reference hash gate.",
        "candidates": [
            {"id": name, "role": "frozen_endpoint", "path": str(ROOT / relative),
             "sha256": sha256_file(ROOT / relative)}
            for name, relative in ENDPOINTS.items()
        ] + [
            {"id": "%s__%s" % (row["source"], "u-zero" if row["seed"] is None else "random-u-%d" % row["seed"]),
             "role": "null_control", "path": str(ROOT / row["path"]), "sha256": row["sha256"]}
            for row in null_rows
        ],
        "baselines": {
            tag: {"id": spec["id"], "role": spec["role"], "path": str(ROOT / spec["path"]),
                  "sha256": spec["sha256"]}
            for tag, spec in BASELINES.items()
        },
    }
    write_json(out / "manifest_controls.json", manifest)
    write_json(out / "null_index.json", {"schema": "p9016-round050-p0-null-index-v1", "nulls": null_rows,
                                         "n_nulls": len(null_rows),
                                         "rule": "u0 and 16 within-chromosome random-u permutations per "
                                                 "endpoint, seeds 450500..450515, identical to the frozen 046 rule"})
    result = ere.evaluate(manifest, out, label=manifest["label"])
    summary, rows = result["summary"], result["rows"]

    # 主比较：每个端点的共同记录支持上同时给出 fixed random / oracle / reference 上限
    per_chrom = ere.__dict__.get("_last_per_chromosome")
    controls = {}
    for chromosome in GENOME_ORDER:
        pass
    write_json(out / "endpoint_support.json", {
        "schema": "p9016-round050-p0-endpoint-support-v1",
        "per_candidate": {cid: {"R1_denominator_per_chromosome": row["R1_denominator_per_chromosome"],
                                "R1_per_chromosome": row["R1_per_chromosome"],
                                "R1_macro_mean_all20": row["R1_macro_mean_all20"],
                                "R1_undefined_chromosomes": row["R1_undefined_chromosomes"],
                                "R2_contrast_defined_chromosomes": row["R2_contrast_defined_chromosomes"],
                                "R3_frac_consistent_defined_chromosomes": row["R3_frac_consistent_defined_chromosomes"]}
                            for cid, row in summary["candidates"].items()},
        "support_rule": "the R1 common support of each candidate is the joint support of candidate + fixed random "
                        "+ oracle + reference; it is reported per chromosome and never silently shrunk",
    })

    indices = np.load(BOOTSTRAP_INDICES)
    if indices.shape != (10_000, 20):
        raise RuntimeError("frozen bootstrap index matrix shape changed")

    fixed_random_r1 = {}
    oracle_r1 = {}
    fixed_random_r3 = {}
    reference_ceiling = {}
    for cid in ENDPOINTS:
        fixed_random_r1[cid] = series(summary, rows, cid, "r1_accuracy")
    bootstrap_rows = []
    comparisons = []
    for left, right, label in COMPARISONS:
        for metric, field in (("R1", "r1_accuracy"), ("R2_contrast_spearman", "r2_selected_contrast_spearman"),
                              ("R3_frac_consistent", "r3_frac_consistent"),
                              ("R3_n_walls", "r3_n_walls")):
            left_values = series(summary, rows, left, field)
            right_values = series(summary, rows, right, field)
            entry = ev._bootstrap_fixed(left_values, right_values, GENOME_ORDER, indices)
            comparisons.append({"label": label, "left": left, "right": right, "metric": metric,
                                "direction": "delta = left minus right", **entry})
            bootstrap_rows.append({
                "comparison": label, "left": left, "right": right, "metric": metric,
                "mean": entry["mean"], "ci95_low": entry["ci95"][0], "ci95_high": entry["ci95"][1],
                "left_wins": entry["left_wins"], "right_wins": entry["right_wins"], "ties": entry["ties"],
                "defined_chromosomes": entry["defined_chromosomes"],
                "full_20_estimate_defined": entry["full_20_estimate_defined"],
                "undefined_chromosomes": "|".join(entry["undefined_chromosomes"]),
                "defined_only_mean": entry["defined_only_descriptive"]["mean"],
                "defined_only_n": entry["defined_only_descriptive"]["n"],
            })
    write_tsv(out / "paired_bootstrap.tsv", bootstrap_rows)
    write_json(out / "paired_bootstrap.json", {
        "schema": "p9016-round050-p0-paired-bootstrap-v1", "seed": 450301, "draws": 10_000,
        "index_matrix": str(BOOTSTRAP_INDICES.relative_to(ROOT)), "index_matrix_sha256": sha256_file(BOOTSTRAP_INDICES),
        "unit": "paired chromosome resampling within one cell; technical/structural variation, not biological "
                "replication",
        "rule": "fixed 20-chromosome denominator; if any chromosome is undefined the primary mean/CI is null and "
                "the defined-only descriptive value and n are reported next to it",
        "rows": comparisons})

    # null 支持：u0 / random-u 各自的 R1 与 R3，不与主比较取交集
    null_support = []
    for row in null_rows:
        cid = "%s__%s" % (row["source"], "u-zero" if row["seed"] is None else "random-u-%d" % row["seed"])
        entry = summary["candidates"][cid]
        null_support.append({
            "source": row["source"], "null_kind": row["null_kind"], "seed": row["seed"],
            "R1_macro_mean_all20": entry["R1_macro_mean_all20"],
            "R1_macro_mean_defined_only": entry["R1_macro_mean_defined_only"],
            "R1_defined_chromosomes": entry["R1_defined_chromosomes"],
            "R1_undefined_chromosomes": "|".join(entry["R1_undefined_chromosomes"]),
            "R1_denominator_total_declared20": entry["R1_denominator_total_declared20"],
            "R1_pooled_over_defined_common_records": entry["R1_pooled_over_defined_common_records"],
            "R2_contrast_spearman_macro_mean_all20": entry["R2_contrast_spearman_macro_mean_all20"],
            "R3_frac_consistent_macro_mean_all20": entry["R3_frac_consistent_macro_mean_all20"],
            "R3_frac_consistent_defined_chromosomes": entry["R3_frac_consistent_defined_chromosomes"],
            "R3_fragments_applicable": entry["R3_fragments_applicable"],
            "R3_fragments_tied": entry["R3_fragments_tied"],
            "R3_status_counts": json.dumps(entry["R3_status_counts"], sort_keys=True),
            "post_scale_radius": row["post_scale_radius"], "global_scale": row["global_scale"],
        })
    write_tsv(out / "null_support.tsv", null_support)
    aggregate = {}
    for kind in ("u_zero", "random_u"):
        for field in ("R1_macro_mean_defined_only", "R2_contrast_spearman_macro_mean_all20",
                      "R3_frac_consistent_macro_mean_all20"):
            values = [row[field] for row in null_support
                      if row["null_kind"] == kind and row[field] is not None]
            aggregate["%s:%s" % (kind, field)] = {
                "n": len(values), "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    write_json(out / "null_aggregate.json", {
        "schema": "p9016-round050-p0-null-aggregate-v1", "aggregate": aggregate,
        "rule": "each null reports its own resolvable support; null resolvability is never intersected into the "
                "main candidate comparison mask, and a u0 R3 tie is an expected null property"})

    write_json(out / "run_terminal.json", {
        "schema": "p9016-round050-p0-controls-terminal-v1",
        "status": "terminal",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_candidates": len(manifest["candidates"]), "n_nulls": len(null_rows),
        "wall_seconds": summary["wall_seconds"],
        "reference_opened": True, "phase_opened": True,
        "scope": "pure evaluation; no fit and no coordinate modification of any non-null candidate",
    })
    print(json.dumps({"candidates": len(manifest["candidates"]), "nulls": len(null_rows),
                      "bootstrap_rows": len(bootstrap_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
