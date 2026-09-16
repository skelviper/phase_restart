"""014 单轨 consensus 基线的共享结构读出（050 轮）。

角色说明：014 `consensus` 每条染色体只有一条轨迹，因此

* R1（需要同 copy 两轨迹）**不可用**是正确语义，不是缺数；
* 双 copy 的 matched/cross/contrast **不可用**；
* 但 `refeval.r2_table` 仍给出 `single_track_baseline.a0_mat` / `a0_pat` 两条
  共享结构 rho 及其各自 n —— 这是“单轨共识 vs 参考两个 copy”的读出，**独立支持**、
  不得当作双拷贝恢复。

本脚本同时列出 020 冻结 `metrics.json` 中同一 014 consensus 的逐 chr 值作为交叉引用，
不重算 020，也不改动任何既有结果。
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
ROOT = RUN_DIR.parents[1]
for _path in (str(HERE),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import endpoint_r1r3_eval as ere  # noqa: E402

CANDIDATE = {
    "id": "046-base-G-random", "role": "carrier only (the single-track readout does not depend on the candidate)",
    "path": str(ROOT / "test_res/046-UTC-real-cell-shared-capture/base_remaining/coords/real-G-random/1Mb.3dg"),
    "sha256": "ee5eb1545db9bfeeabcc24e704707f5f61a5793e6f245091347373442dc0032b",
}
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
TWENTY_020 = ROOT / "test_res/020-20260913_071841-v1-p9016-joint/evaluation-20260913_080300-final/metrics/metrics.json"


def main() -> int:
    out = RUN_DIR / "evaluation" / "consensus_single_track"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": ere.SCHEMA, "label": "consensus-single-track-readout",
        "raw_pairs_path": str(ROOT / "data/P9016.pairs.gz"),
        "raw_pairs_sha256": "071a6cc76bfad543ea1ace6ee1ce3022b30ac1b1e50a9f0c3a3a1967b9649505",
        "snpfree_path": str(ROOT / "inputs/P9016.snpfree.pairs.gz"),
        "snpfree_sha256": "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa",
        "reference_3dg_path": str(ROOT / "data/P9016.1m.3dg.gz"),
        "reference_3dg_sha256": "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29",
        "record_count": 1703888,
        "note": "single carrier candidate; the 014 single-track consensus readout is a property of the consensus "
                "baseline and the reference only, so it is candidate-independent",
        "candidates": [CANDIDATE], "baselines": BASELINES,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = ere.evaluate(manifest, out, label="consensus-single-track-readout")
    per_chromosome = result["per_chromosome"]

    rows, macro_mat, macro_pat, ns = [], [], [], []
    for name, evaluated in per_chromosome.items():
        consensus = ((evaluated["candidates"][CANDIDATE["id"]]["R2"].get("by_candidate") or {})
                     .get("consensus") or {})
        baseline = consensus.get("single_track_baseline") or {}
        a0_mat = baseline.get("a0_mat") or {}
        a0_pat = baseline.get("a0_pat") or {}
        rows.append({"chromosome": name, "r2_applicable": consensus.get("applicable"),
                     "r2_reason": consensus.get("reason"),
                     "a0_mat_rho": a0_mat.get("rho"), "a0_mat_n": a0_mat.get("n"),
                     "a0_pat_rho": a0_pat.get("rho"), "a0_pat_n": a0_pat.get("n")})
        if a0_mat.get("rho") is not None:
            macro_mat.append(float(a0_mat["rho"]))
        if a0_pat.get("rho") is not None:
            macro_pat.append(float(a0_pat["rho"]))
        if a0_mat.get("n") is not None:
            ns.append(int(a0_mat["n"]))

    reference_020 = None
    if TWENTY_020.is_file():
        frozen = json.loads(TWENTY_020.read_text(encoding="utf-8"))
        per_chr = frozen.get("per_chromosome") or {}
        mat, pat = [], []
        for name, entry in per_chr.items():
            consensus = (((entry.get("candidates") or {}).get("consensus_joint") or {})
                         .get("R2", {}).get("by_candidate", {}).get("consensus") or {})
            baseline = consensus.get("single_track_baseline") or {}
            if (baseline.get("a0_mat") or {}).get("rho") is not None:
                mat.append(float(baseline["a0_mat"]["rho"]))
            if (baseline.get("a0_pat") or {}).get("rho") is not None:
                pat.append(float(baseline["a0_pat"]["rho"]))
        reference_020 = {
            "path": str(TWENTY_020.relative_to(ROOT)),
            "chromosomes": len(per_chr),
            "a0_mat_rho_macro": float(np.mean(mat)) if mat else None,
            "a0_pat_rho_macro": float(np.mean(pat)) if pat else None,
            "note": "020 frozen evaluation of the SAME 014 consensus baseline; quoted for cross-reference and not "
                    "recomputed here",
        }

    ere.write_tsv(out / "single_track_readout.tsv", rows)
    payload = {
        "schema": "p9016-round050-consensus-single-track-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline": {"tag": "consensus", "path": BASELINES["consensus"]["path"],
                     "sha256": BASELINES["consensus"]["sha256"],
                     "structure": "014 fixed single-trajectory consensus (one track per chromosome)"},
        "this_round": {
            "a0_mat_rho_macro_mean": float(np.mean(macro_mat)) if macro_mat else None,
            "a0_pat_rho_macro_mean": float(np.mean(macro_pat)) if macro_pat else None,
            "chromosomes_with_readout": len(macro_mat),
            "a0_mat_n_total": int(sum(ns)),
            "a0_mat_n_per_chromosome": {row["chromosome"]: row["a0_mat_n"] for row in rows},
        },
        "cross_reference_020": reference_020,
        "role_note": "the 014 consensus is a SINGLE-track structure. R1, R3 and the two-copy matched/cross/contrast "
                     "are correctly unavailable and are reported as NA, not as missing data. The a0_mat/a0_pat rho "
                     "values are shared-structure readouts against the two reference copies on their own support "
                     "(n differs per chromosome) and must not be read as two-copy recovery. The u0 control is a "
                     "different object: it is a two-copy structure collapsed onto z, used to show that the "
                     "copy-identity degree of freedom carries no resolvable signal.",
        "rows": rows,
    }
    ere.write_json(out / "single_track_readout.json", payload)
    print(json.dumps({"a0_mat_macro": payload["this_round"]["a0_mat_rho_macro_mean"],
                      "a0_pat_macro": payload["this_round"]["a0_pat_rho_macro_mean"],
                      "cross_ref_020": reference_020}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
