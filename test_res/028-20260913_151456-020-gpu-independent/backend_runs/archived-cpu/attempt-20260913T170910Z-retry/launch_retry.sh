#!/usr/bin/env bash
set -u

attempt=/mnt/ssd/zliu/phase_restart/test_res/028-20260913_151456-020-gpu-independent/backend_runs/archived-cpu/attempt-20260913T170910Z-retry
driver=/mnt/ssd/zliu/phase_restart/test_res/028-20260913_151456-020-gpu-independent/source/run_pipeline.py
input_path=/mnt/ssd/zliu/phase_restart/inputs/P9016.snpfree.pairs.gz
log_path="$attempt/terminal.stdout.log"

# The benchmark stdout and stderr are one immutable machine-generated stream.
exec > "$log_path" 2>&1

source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh
conda activate analysis
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

command_text="python $driver --backend archived_cpu --device cpu --through 1m --run-dir $attempt"
launcher_pid=$$
runner_pid=""
runner_rc="not_started"
start_wall_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_wall_epoch_seconds=$(date +%s.%N)
start_monotonic_ns=$(python -c 'import time; print(time.monotonic_ns())')
printf '%s\n' "$launcher_pid" > "$attempt/launcher.pid"
printf '%s\n' "$start_wall_utc" > "$attempt/launcher.start.utc"

write_terminal_status() {
    local shell_rc=$1
    local end_wall_utc end_wall_epoch_seconds end_monotonic_ns
    end_wall_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    end_wall_epoch_seconds=$(date +%s.%N)
    end_monotonic_ns=$(python -c 'import time; print(time.monotonic_ns())')
    export STATUS_ATTEMPT="$attempt"
    export STATUS_COMMAND="$command_text"
    export STATUS_LAUNCHER_PID="$launcher_pid"
    export STATUS_RUNNER_PID="$runner_pid"
    export STATUS_RUNNER_RC="$runner_rc"
    export STATUS_SHELL_RC="$shell_rc"
    export STATUS_START_WALL_UTC="$start_wall_utc"
    export STATUS_END_WALL_UTC="$end_wall_utc"
    export STATUS_START_WALL_EPOCH="$start_wall_epoch_seconds"
    export STATUS_END_WALL_EPOCH="$end_wall_epoch_seconds"
    export STATUS_START_MONO_NS="$start_monotonic_ns"
    export STATUS_END_MONO_NS="$end_monotonic_ns"
    export STATUS_INPUT_SHA256=f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa
    export STATUS_CONFIG_SHA256=14deef2734fb0b13798d394d6bf0336c23c6e2a3958ec60f532c48f959e6ae7e
    export STATUS_DRIVER_SHA256=e0e5211048039f75ba29ea8598a26214cf971ede5af5df7c69ed64aff27ca012
    export STATUS_OBJECTIVE_SHA256=cf19833febc97e346bbaa24f3520c385fdcfb18180957a23b59838f7402e8ace
    export STATUS_SOURCE_MANIFEST="$attempt/provenance/source-manifest.json"
    export STATUS_LOG_PATH="$log_path"
    python - <<'PY'
import json
import os
from pathlib import Path

attempt = Path(os.environ['STATUS_ATTEMPT'])
start_mono = int(os.environ['STATUS_START_MONO_NS'])
end_mono = int(os.environ['STATUS_END_MONO_NS'])
runner_pid = os.environ['STATUS_RUNNER_PID']
try:
    runner_pid_json = int(runner_pid)
except ValueError:
    runner_pid_json = None
try:
    runner_rc_json = int(os.environ['STATUS_RUNNER_RC'])
except ValueError:
    runner_rc_json = None
status = {
    'schema_version': '028-archived-cpu-retry-terminal-status-v1',
    'status': 'terminal',
    'command': os.environ['STATUS_COMMAND'],
    'attempt_directory': str(attempt),
    'launcher_pid': int(os.environ['STATUS_LAUNCHER_PID']),
    'runner_pid': runner_pid_json,
    'runner_exit_code': runner_rc_json,
    'shell_exit_code': int(os.environ['STATUS_SHELL_RC']),
    'start_wall_utc': os.environ['STATUS_START_WALL_UTC'],
    'end_wall_utc': os.environ['STATUS_END_WALL_UTC'],
    'start_wall_epoch_seconds': float(os.environ['STATUS_START_WALL_EPOCH']),
    'end_wall_epoch_seconds': float(os.environ['STATUS_END_WALL_EPOCH']),
    'start_monotonic_ns': start_mono,
    'end_monotonic_ns': end_mono,
    'elapsed_monotonic_seconds': (end_mono - start_mono) / 1e9,
    'input_sha256': os.environ['STATUS_INPUT_SHA256'],
    'config_sha256': os.environ['STATUS_CONFIG_SHA256'],
    'driver_sha256': os.environ['STATUS_DRIVER_SHA256'],
    'objective_sha256': os.environ['STATUS_OBJECTIVE_SHA256'],
    'source_manifest': os.environ['STATUS_SOURCE_MANIFEST'],
    'terminal_log': os.environ['STATUS_LOG_PATH'],
    'phase_or_reference_opened': False,
    'thread_environment': {
        'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'candidate_workers': 1,
    },
}
(attempt / 'terminal.status.json').write_text(json.dumps(status, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(attempt / 'terminal.exitcode').write_text(str(int(os.environ['STATUS_SHELL_RC'])) + '\n', encoding='ascii')
(attempt / 'launcher.end.utc').write_text(os.environ['STATUS_END_WALL_UTC'] + '\n', encoding='ascii')
PY
}

on_exit() {
    local shell_rc=$?
    trap - EXIT
    if [ "$runner_rc" = "not_started" ]; then
        runner_rc=$shell_rc
    fi
    write_terminal_status "$shell_rc"
    exit "$shell_rc"
}
trap on_exit EXIT

printf '%s\n' "[$start_wall_utc] launcher_pid=$launcher_pid"
printf '%s\n' "[$start_wall_utc] command=$command_text"
printf '%s\n' "[$start_wall_utc] thread_caps OMP=1 OPENBLAS=1 MKL=1 NUMEXPR=1"
python "$driver" --backend archived_cpu --device cpu --through 1m --run-dir "$attempt" &
runner_pid=$!
printf '%s\n' "$runner_pid" > "$attempt/runner.pid"
printf '%s\n' "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] runner_pid=$runner_pid started"
wait "$runner_pid"
runner_rc=$?
printf '%s\n' "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] runner_exit_code=$runner_rc"
exit "$runner_rc"
