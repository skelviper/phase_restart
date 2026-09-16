#!/usr/bin/env bash
# 054 正确共享链：20Mb -> 10Mb -> 5Mb -> 2Mb -> 1Mb -> 500kb -> 200kb
# 40Mb 不运行：20Mb 从 052 的已登记 reference-free 冻结端点 prolongation。
# 每层一个独立进程；任一层非零退出立即停止，不续跑、不换起点。
set -uo pipefail
cd "$(dirname "$0")/.."
RUN_DIR="$(pwd)"
PY=python
LOGS="$RUN_DIR/logs"
mkdir -p "$LOGS" "$RUN_DIR/coords/new-chain" "$RUN_DIR/stages/new-chain" "$RUN_DIR/checkpoints"

PREFIX_40MB="$RUN_DIR/../052-20260916T080116Z-200kb-coarsened-1mb-chain/coords/new-chain/40Mb.npz"
declare -A PREV=( ["20Mb"]="$PREFIX_40MB" ["10Mb"]="coords/new-chain/20Mb.npz" \
                  ["5Mb"]="coords/new-chain/10Mb.npz" ["2Mb"]="coords/new-chain/5Mb.npz" \
                  ["1Mb"]="coords/new-chain/2Mb.npz" ["500kb"]="coords/new-chain/1Mb.npz" \
                  ["200kb"]="coords/new-chain/500kb.npz" )
CHAIN=(20Mb 10Mb 5Mb 2Mb 1Mb 500kb 200kb)

overall=0
START_STAGE="${1:-20Mb}"
started=0
for stage in "${CHAIN[@]}"; do
  if [ "$started" -eq 0 ] && [ "$stage" != "$START_STAGE" ]; then
    echo "=== skip $stage (already finished before --start $START_STAGE)"
    continue
  fi
  started=1
  prev="${PREV[$stage]}"
  echo "=== stage $stage (prev='${prev}') start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  args=(code/run_stage_054.py --stage "$stage" --previous "$prev")
  "$PY" "${args[@]}" > "$LOGS/new-chain-$stage.stdout.log" 2> "$LOGS/new-chain-$stage.stderr.log"
  code=$?
  echo "=== stage $stage exit=$code $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ "$code" -ne 0 ]; then
    overall=$code
    echo "STOP: stage $stage failed with exit $code"
    break
  fi
done
echo "CHAIN_OVERALL_EXIT=$overall"
exit "$overall"
