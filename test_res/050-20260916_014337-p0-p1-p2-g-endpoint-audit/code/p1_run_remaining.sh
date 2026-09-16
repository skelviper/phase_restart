#!/usr/bin/env bash
# P1 剩余三支串行驱动（GPU 单卡，避免并发争用）。
set -u
cd "$(dirname "$0")/.."
PY=/mnt/ssd/zliu/miniforge3/envs/analysis/bin/python
LOGDIR=p1/logs
mkdir -p "$LOGDIR"
overall=0
for spec in "raw G-random" "ms G-consensus" "ms G-random"; do
  set -- $spec
  solver="$1"; base="$2"
  fit_id="A-${solver}-${base}"
  echo "=== ${fit_id} start $(date -u +%FT%TZ) ==="
  "$PY" code/p1_fork_runner.py --solver "$solver" --base "$base" \
      > "$LOGDIR/${fit_id}.stdout.log" 2> "$LOGDIR/${fit_id}.stderr.log"
  code=$?
  echo "=== ${fit_id} exit=${code} $(date -u +%FT%TZ) ==="
  if [ "$code" -ne 0 ]; then overall="$code"; fi
done
echo "P1_DRIVER_EXIT=${overall}"
exit "$overall"
