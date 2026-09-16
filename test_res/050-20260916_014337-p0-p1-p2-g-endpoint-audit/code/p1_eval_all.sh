#!/usr/bin/env bash
# P1 评价驱动：先生成 R1/R3 manifest（对 4 支新端点 + 014 冻结 baseline），
# 再跑独立 R1/R3 入口，最后跑 R2/空间/对照评价入口。
set -u
cd "$(dirname "$0")/.."
PY=/mnt/ssd/zliu/miniforge3/envs/analysis/bin/python
ROOT=/mnt/ssd/zliu/phase_restart
mkdir -p evaluation/logs

"$PY" - <<'PYEOF'
import hashlib, json, pathlib
ROOT = pathlib.Path("/mnt/ssd/zliu/phase_restart")
RUN = ROOT / "test_res/050-20260916_014337-p0-p1-p2-g-endpoint-audit"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
cands = []
for solver in ("raw", "ms"):
    for base in ("G-consensus", "G-random"):
        p = RUN / "p1/coords" / ("A-%s-%s" % (solver, base)) / "1Mb.3dg"
        if not p.is_file():
            raise SystemExit("missing %s" % p)
        cands.append({"id": "A-%s-%s" % (solver, base), "role": "p1 fork arm (local optimization diagnostic)",
                      "path": str(p), "sha256": sha(p)})
bases = {
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
manifest = {
    "schema": "p9016-endpoint-r1r3-evaluation-v1",
    "label": "p1-r1r3-four-arms",
    "raw_pairs_path": str(ROOT / "data/P9016.pairs.gz"),
    "raw_pairs_sha256": "071a6cc76bfad543ea1ace6ee1ce3022b30ac1b1e50a9f0c3a3a1967b9649505",
    "snpfree_path": str(ROOT / "inputs/P9016.snpfree.pairs.gz"),
    "snpfree_sha256": "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa",
    "reference_3dg_path": str(ROOT / "data/P9016.1m.3dg.gz"),
    "reference_3dg_sha256": "1ca82ef4785bc800d9b7ca5fadafa8de9ff028d5f5e0df41183ad087217cea29",
    "record_count": 1703888,
    "selection_note": "no selection happens here: all four P1 arms are reported, and the two 046 bases are the "
                      "comparison references, not candidates of this selection.",
    "candidates": cands,
    "baselines": bases,
}
target = RUN / "evaluation" / "manifest_r1r3_new4.json"
target.write_text(json.dumps(manifest, indent=2) + "\n")
print("manifest written with %d candidates" % len(cands))
PYEOF

code=$?
if [ "$code" -ne 0 ]; then echo "manifest build failed"; exit "$code"; fi

"$PY" code/endpoint_r1r3_eval.py --manifest evaluation/manifest_r1r3_new4.json \
    --out evaluation/r1r3_new4 --label p1-r1r3-four-arms \
    > evaluation/logs/r1r3_new4.log 2>&1
code_r1r3=$?
echo "r1r3 exit=${code_r1r3}"

"$PY" code/p1_eval.py > evaluation/logs/p1_eval.log 2>&1
code_eval=$?
echo "p1_eval exit=${code_eval}"

echo "P1_EVAL_DRIVER_EXIT=$(( code_r1r3 != 0 || code_eval != 0 ))"
exit $(( code_r1r3 != 0 || code_eval != 0 ))
