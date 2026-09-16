"""在控制器层修复后刷新 035 protocol/config/source manifest。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "test_res" / "035-20260914T060945Z-gpu-multires-preflight"
SOURCE = ARTIFACT / "source"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from pr import reconstruct  # noqa: E402
import gpu_multires_controller as controller  # noqa: E402


def json_bytes(value):
    return json.dumps(controller._jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_artifact(path: Path, value):
    payload = json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main():
    preflight_path = ARTIFACT / "preflight.json"
    preflight = controller.cpu_runner._read_json(preflight_path)
    preflight.pop("controller_schema_smoke", None)
    preflight.setdefault("evidence_sha256", {}).pop("controller_schema_test.json", None)
    preflight_sha = write_artifact(preflight_path, preflight)
    context = reconstruct.production_context()
    reconstruct._validate_context(context)
    protocol = controller._protocol(context, preflight_path.resolve(), ARTIFACT)
    protocol_path = ARTIFACT / "protocol.json"
    protocol_sha = write_artifact(protocol_path, protocol)
    old_config = controller.cpu_runner._read_json(ARTIFACT / "config.json")
    formal_output = Path(str(old_config["formal_output_path"])).resolve()
    config = controller._config(context, protocol, protocol_sha, formal_output)
    config_path = ARTIFACT / "config.json"
    config_sha = write_artifact(config_path, config)
    source_manifest = {
        "schema": "gpu-multires-source-hashes-v1",
        "source_code_sha256": controller._source_hashes(),
        "protocol_sha256": protocol_sha,
        "config_sha256": config_sha,
        "input_sha256": str(context.data_sha256),
        "approved_014_sources": {key: value["sha256"] for key, value in context.source_assets.items()},
        "c0_gate_sha256": controller.sha256_file(ROOT / "test_res/033-20260914_044246-c0-controlled-reproduction/C0_gate.json"),
    }
    source_path = ARTIFACT / "provenance" / "source_hashes.json"
    source_sha = write_artifact(source_path, source_manifest)
    launch_path = ARTIFACT / "launch_command.txt"
    launch = (
        "source /mnt/ssd/zliu/miniforge3/etc/profile.d/conda.sh\n"
        "conda activate analysis\n"
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n"
        f"python {str((SOURCE / 'gpu_multires_controller.py').resolve())} run "
        f"--artifact {str(ARTIFACT.resolve())} --output {str(formal_output)} --device cuda --workers 1\n"
    ).encode("ascii")
    launch_path.write_bytes(launch)
    launch_sha = hashlib.sha256(launch).hexdigest()
    terminal = {
        "schema": "gpu-multires-preflight-terminal-v1",
        "status": "preflight_passed_formal_pending_parent_acceptance",
        "formal_run_launched": False,
        "metadata": {
            "formal_output_suggestion": str(formal_output),
            "protocol": {"path": str(protocol_path), "sha256": protocol_sha},
            "config": {"path": str(config_path), "sha256": config_sha},
            "source_hashes": {"path": str(source_path), "sha256": source_sha},
            "launch_command": {"path": str(launch_path), "sha256": launch_sha},
            "planned_trajectory_count": len(controller.VARIANTS) * len(controller.CANDIDATES),
            "planned_stage_attempt_count": len(controller._planned_rows()),
            "protocol_sha256_recomputed": protocol_sha,
        },
        "planned_trajectory_count": len(controller.VARIANTS) * len(controller.CANDIDATES),
        "planned_stage_attempt_count": len(controller._planned_rows()),
        "preflight": {"path": str(preflight_path), "sha256": preflight_sha},
        "phase_used": False,
        "reference_used": False,
    }
    terminal_path = ARTIFACT / "terminal_prep.json"
    terminal_sha = write_artifact(terminal_path, terminal)
    print(json.dumps({
        "status": "refreshed",
        "preflight_sha256": preflight_sha,
        "protocol_sha256": protocol_sha,
        "config_sha256": config_sha,
        "source_hashes_sha256": source_sha,
        "launch_sha256": launch_sha,
        "terminal_prep_sha256": terminal_sha,
        "formal_output": str(formal_output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
