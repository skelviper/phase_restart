#!/usr/bin/env bash
# 用同一入口重算 P0 三个目录，追加逐片段 R3 表；旧 summary 先另存为 *.preregen.json 以便逐字段比对。
set -u
cd "$(dirname "$0")/.."
PY=/mnt/ssd/zliu/miniforge3/envs/analysis/bin/python

for spec in "regression020:p0/manifest_regression020.json:p0-regression020-v2" \
            "formal5:p0/manifest_formal5.json:p0-formal-5-endpoints" \
            "controls:p0/controls/manifest_controls.json:p0-controls-u0-randomu"; do
  IFS=: read -r name manifest label <<< "$spec"
  if [ ! -f "$manifest" ]; then echo "skip ${name}: no manifest"; continue; fi
  if [ "$name" = "controls" ] && [ ! -f p0/controls/results/r1_r3_summary.json ]; then
    echo "skip controls: first run has not finished"
    continue
  fi
  out="p0/${name}"
  if [ -f "$out/r1_r3_summary.json" ]; then
    cp "$out/r1_r3_summary.json" "$out/r1_r3_summary.preregen.json"
  fi
  echo "=== rerun ${name} -> ${out} $(date -u +%FT%TZ) ==="
  "$PY" code/endpoint_r1r3_eval.py --manifest "$manifest" --out "$out" --label "$label" \
      > "$out/../logs/${name}.regen.log" 2>&1
  echo "=== ${name} exit=$? $(date -u +%FT%TZ) ==="
done

"$PY" - <<'PYEOF'
import json, pathlib, math
RUN = pathlib.Path("/mnt/ssd/zliu/phase_restart/test_res/050-20260916_014337-p0-p1-p2-g-endpoint-audit")
report = {"schema": "p9016-round050-fragment-table-regeneration-v1",
          "purpose": "the compact per-fragment R3 table was added after the first P0 runs; the same entry point "
                     "was re-executed so the table comes from the same in-memory detail instead of a separate "
                     "recomputation",
          "directories": {}}
for name in ("regression020", "formal5", "controls"):
    d = RUN / "p0" / name
    old_path = d / "r1_r3_summary.preregen.json"
    new_path = d / "r1_r3_summary.json"
    frag = d / "r3_fragments.tsv"
    entry = {"has_preregen": old_path.is_file(), "has_fragment_table": frag.is_file(),
             "fragment_rows": (sum(1 for _ in frag.open()) - 1) if frag.is_file() else None}
    if old_path.is_file() and new_path.is_file():
        old = json.loads(old_path.read_text())
        new = json.loads(new_path.read_text())
        diffs = []
        for cid, row in old["candidates"].items():
            for key, value in row.items():
                if key.startswith("R3_") or key.startswith("R1_") or key.startswith("R2_"):
                    other = new["candidates"].get(cid, {}).get(key)
                    same = value == other or (isinstance(value, float) and isinstance(other, float)
                                              and math.isclose(value, other, rel_tol=0, abs_tol=0))
                    if not same:
                        diffs.append({"candidate": cid, "field": key, "before": value, "after": other})
        entry["scalar_field_differences"] = diffs
        entry["identical"] = not diffs
    report["directories"][name] = entry
(RUN / "p0" / "fragment_table_regeneration.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PYEOF
