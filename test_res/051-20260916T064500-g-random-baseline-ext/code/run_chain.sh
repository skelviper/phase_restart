#!/usr/bin/env bash
# 051 条件链式执行：A = 20/10/5/2/1Mb，B = 5/2/1Mb（bend=0），C = 5/2/1Mb（reference beads）。
# 用法：run_chain.sh <A|B|C> [起始阶段]；起始阶段之前必须已有 endpoint npz。
set -euo pipefail
COND="${1:?condition A|B|C}"
START="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(dirname "$HERE")"
cd "$RUN"
source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh
conda activate analysis
case "$COND" in
  A) STAGES=(20Mb 10Mb 5Mb 2Mb 1Mb) ;;
  B|C) STAGES=(5Mb 2Mb 1Mb) ;;
  *) echo "unknown condition $COND" >&2; exit 2 ;;
esac
case "$COND" in
  A) FIT="A-extra-levels" ;;
  B) FIT="B-no-bend" ;;
  C) FIT="C-reference-beads" ;;
esac
PREV=""
if [[ -n "$START" ]]; then
  for STAGE in "${STAGES[@]}"; do
    if [[ "$STAGE" == "$START" ]]; then break; fi
    PREV="coords/${FIT}/${STAGE}.npz"
  done
  if [[ -z "$PREV" || ! -f "$PREV" ]]; then echo "missing start endpoint for $START" >&2; exit 3; fi
fi
LAUNCHED=0
for STAGE in "${STAGES[@]}"; do
  if [[ -n "$START" && $LAUNCHED -eq 0 ]]; then
    if [[ "$STAGE" != "$START" ]]; then continue; fi
    LAUNCHED=1
  fi
  ARGS=(--condition "$COND" --stage "$STAGE")
  if [[ -n "$PREV" ]]; then ARGS+=(--previous "$PREV"); fi
  echo "[chain $COND] stage $STAGE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set +e
  python code/run_stage.py "${ARGS[@]}" >> "logs/${FIT}-${STAGE}.out" 2>&1
  RC=$?
  set -e
  if [[ $RC -ne 0 ]]; then echo "[chain $COND] stage $STAGE exit $RC" >&2; exit $RC; fi
  PREV="coords/${FIT}/${STAGE}.npz"
  if [[ ! -f "$PREV" ]]; then echo "[chain $COND] missing $PREV" >&2; exit 3; fi
done
echo "[chain $COND] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
