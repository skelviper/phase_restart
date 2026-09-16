"""P1 null 的 R1/R3 补充（050 轮）：4 支 fork 端点的 u0 与 16 random-u。

与 046 冻结规则一致（逐染色体内置换 u，保持 z；球外整体缩小），先写全网格 3DG 并哈希，
再由独立 R1/R3 入口评价；每个 null 用自身可解析支持上报，不与主比较取交集。
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
ROOT = RUN_DIR.parents[1]
for _path in (str(HERE),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import endpoint_r1r3_eval as ere  # noqa: E402
import p0_controls as pc  # noqa: E402
import p1_eval as pe  # noqa: E402

ARMS = {solver: ["A-%s-G-consensus" % solver, "A-%s-G-random" % solver] for solver in ("raw", "ms")}
BASELINES = {
    "random": {"id": "fixed014-random", "role": "blind_baseline",
               "path": str(ROOT / "test_res/014-20260912_153000-s0-genome-wide-fixed/coords/random.3dg"),
               "sha256": "9a48d73e1401e18349d11758e679da4c76da0904dbc467979079cb54bcd567d7"},
    "consensus": {"id": "fixed014-consensus", "role": "blind_baseline",
                  "path": str(ROOT / "test_res/014-20260912_153000-s0-genome-wide-fixed/coords/consensus.3dg"),
                  "sha256": "e76655732deb6b8386b1b77bc76ff45d7dba1384f6931337fee80d8f4aaa8e02"},
    "oracle": {"id": "s0-oracle", "role": "evaluation_ceiling",
               "path": str(ROOT / "test_res/014-20260912_153000-s0-genome-wide-fixed/coords/oracle.3dg"),
               "sha256": "502cd64944dcbd278735b3c9df4d6df720907cedd6965c2f0cc728de6abbfc29"},
}


def main() -> int:
    out = RUN_DIR / "evaluation" / "nulls_r1r3"
    out.mkdir(parents=True, exist_ok=True)
    data = pe.ev._load_data()
    lengths = {str(name): int(length)
               for name, length in zip(data.chromosome_names, data.chromosome_lengths)}
    slices = pc.chromosome_slices(lengths)
    arms = [arm for solver in ("raw", "ms") for arm in ARMS[solver]]
    coords = {arm: pe.load_full_grid_3dg(pe.candidate_3dg_path(arm), data) for arm in arms}

    null_dir = out / "nulls"
    null_index = []
    for arm, values in coords.items():
        variants = [("u_zero", None)] + [("random_u", seed) for seed in pc.RANDOM_U_SEEDS]
        for kind, seed in variants:
            if seed is None:
                probe, audit = pc.make_u_zero(values)
                suffix = "u-zero"
            else:
                probe, audit = pc.make_random_u(values, slices, seed)
                suffix = "random-u-%d" % seed
            path = null_dir / ("%s__%s.3dg" % (arm, suffix))
            pc.write_full_grid(path, probe, lengths)
            null_index.append({"source": arm, "null_kind": kind, "seed": seed,
                               "path": str(path.relative_to(ROOT)),
                               "id": "%s__%s" % (arm, suffix), **audit})

    manifest = {
        "schema": ere.SCHEMA, "label": "p1-nulls-r1r3",
        "raw_pairs_path": str(ROOT / "data/P9016.pairs.gz"),
        "raw_pairs_sha256": "071a6cc76bfad543ea1ace6ee1ce3022b30ac1b1e50a9f0c3a3a1967b9649505",
        "snpfree_path": str(ROOT / "inputs/P9016.snpfree.pairs.gz"),
        "snpfree_sha256": "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa",
        "reference_3dg_path": str(ROOT / "data/P9016.1m.3dg.gz"),
        "reference_3dg_sha256": "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29",
        "record_count": 1703888,
        "note": "P1 arms are repeated as candidates so that every comparison shares one code path and one "
                "pre-reference hash gate; all 85 null coordinate payloads are hashed before the reference opens.",
        "candidates": [
            {"id": arm, "role": "p1_fork_arm", "path": str(pe.candidate_3dg_path(arm)),
             "sha256": ere.sha256_file(pe.candidate_3dg_path(arm))} for arm in arms
        ] + [
            {"id": row["id"], "role": "null_control", "path": str(ROOT / row["path"]),
             "sha256": ere.sha256_file(ROOT / row["path"])} for row in null_index
        ],
        "baselines": BASELINES,
    }
    ere.write_json(out / "manifest_nulls_r1r3.json", manifest)
    ere.write_json(out / "null_index.json", {"schema": "p9016-round050-p1-null-index-v1",
                                             "nulls": null_index, "n_nulls": len(null_index)})
    result = ere.evaluate(manifest, out, label="p1-nulls-r1r3")
    summary, rows = result["summary"], result["rows"]

    support = []
    for row in null_index:
        entry = summary["candidates"][row["id"]]
        support.append({
            "source": row["source"], "null_kind": row["null_kind"], "seed": row["seed"],
            "R1_macro_mean_all20": entry["R1_macro_mean_all20"],
            "R1_macro_mean_defined_only": entry["R1_macro_mean_defined_only"],
            "R1_defined_chromosomes": entry["R1_defined_chromosomes"],
            "R1_undefined_chromosomes": "|".join(entry["R1_undefined_chromosomes"]),
            "R1_pooled_over_defined_common_records": entry["R1_pooled_over_defined_common_records"],
            "R1_denominator_total_declared20": entry["R1_denominator_total_declared20"],
            "R2_contrast_spearman_macro_mean_all20": entry["R2_contrast_spearman_macro_mean_all20"],
            "R3_frac_consistent_macro_mean_all20": entry["R3_frac_consistent_macro_mean_all20"],
            "R3_frac_consistent_defined_chromosomes": entry["R3_frac_consistent_defined_chromosomes"],
            "R3_fragments_applicable": entry["R3_fragments_applicable"],
            "R3_fragments_tied": entry["R3_fragments_tied"],
            "R3_status_counts": json.dumps(entry["R3_status_counts"], sort_keys=True),
            "post_scale_radius": row["post_scale_radius"], "global_scale": row["global_scale"],
        })
    ere.write_tsv(out / "null_r1r3_support.tsv", support)
    aggregate = {}
    for kind in ("u_zero", "random_u"):
        for field in ("R1_macro_mean_defined_only", "R3_frac_consistent_macro_mean_all20",
                      "R2_contrast_spearman_macro_mean_all20"):
            values = [row[field] for row in support if row["null_kind"] == kind and row[field] is not None]
            aggregate["%s:%s" % (kind, field)] = {
                "n": len(values), "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    ere.write_json(out / "null_r1r3_aggregate.json", {
        "schema": "p9016-round050-p1-null-r1r3-aggregate-v1", "aggregate": aggregate,
        "n_nulls": len(support),
        "rule": "each null uses its own resolvable support; a u0 R3 tie or undefined direction is an expected "
                "null property and is never intersected into the main candidate mask"})
    ere.write_json(out / "run_terminal.json", {
        "schema": "p9016-round050-p1-null-r1r3-terminal-v1", "status": "terminal",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_candidates": len(manifest["candidates"]), "n_nulls": len(null_index),
        "wall_seconds": summary["wall_seconds"], "reference_opened": True, "phase_opened": True})
    print(json.dumps({"n_nulls": len(support),
                      "aggregate": aggregate}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
