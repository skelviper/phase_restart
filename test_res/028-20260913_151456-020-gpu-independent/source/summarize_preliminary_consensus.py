"""Machine-generated preliminary paired timing for the completed consensus chain only."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import summarize_benchmark as benchmark

RUN = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
CANDIDATE = "consensus_joint"
INPUT_SHA256 = benchmark.FROZEN_INPUT_SHA256


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def validate_chain(attempt: Path, backend_label: str, expected_backend: str, config_hash: str) -> dict:
    root = benchmark.backend_root(attempt, backend_label)
    snapshot_summary = None
    snapshot_manifest = None
    snapshot_stage_root = None
    if backend_label == "fused-cuda":
        summary_path, snapshot_stage_root, snapshot_manifest = benchmark.authoritative_summary(
            attempt, root, backend_label
        )
        benchmark.validate_snapshot_entries(snapshot_manifest)
        snapshot_summary = load_json(summary_path)
        if snapshot_summary.get("status") != "training_complete" or snapshot_summary.get("through") != "1m":
            raise ValueError("GPU immutable through-1m summary is not terminal")
        if snapshot_summary.get("input_sha256") != INPUT_SHA256 or snapshot_summary.get("config_sha256") != config_hash:
            raise ValueError("GPU immutable through-1m summary provenance mismatch")
        stage_root = snapshot_stage_root
    else:
        stage_root = root
    rows = []
    source_paths = []
    for label, bin_size, maxiter, maxfun in benchmark.STAGES:
        path = stage_root / "stages" / CANDIDATE / f"{label}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing completed consensus stage: {path}")
        row = benchmark.validate_stage(
            path, CANDIDATE, label, bin_size, maxiter, maxfun, INPUT_SHA256, config_hash
        )
        if expected_backend not in str(row.get("backend")):
            raise ValueError(f"{path}: backend identity does not contain {expected_backend}")
        if row.get("n_tracks") != 40:
            raise ValueError(f"{path}: consensus stage does not export all 40 tracks")
        rows.append(row)
        source_paths.append({"stage": label, "path": str(path), "sha256": file_sha256(path)})
    if len(rows) != 3 or not all(row["budget_validated"] for row in rows):
        raise ValueError("consensus chain did not pass all three stage budget checks")
    return {
        "attempt": str(attempt),
        "backend_label": backend_label,
        "backend": expected_backend,
        "pipeline_summary": str(attempt / "provenance" / "through-1m" / "pipeline-summary.json") if snapshot_manifest else None,
        "immutable_snapshot_manifest": str(attempt / "provenance" / "through-1m-snapshot.json") if snapshot_manifest else None,
        "stage_files": source_paths,
        "stages": rows,
    }


def paired_timing(cpu: dict, gpu: dict) -> dict:
    cpu_rows = {row["stage"]: row for row in cpu["stages"]}
    gpu_rows = {row["stage"]: row for row in gpu["stages"]}
    rows = []
    for label, _, _, _ in benchmark.STAGES:
        c = cpu_rows[label]
        g = gpu_rows[label]
        rows.append({
            "candidate": CANDIDATE,
            "stage": label,
            "cpu_optimizer_phase_seconds": c["optimizer_only_seconds"],
            "gpu_optimizer_phase_seconds": g["optimizer_only_seconds"],
            "cpu_scoped_stage_seconds": c["scoped_end_to_end_seconds"],
            "gpu_scoped_stage_seconds": g["scoped_end_to_end_seconds"],
            "cpu_over_gpu_optimizer_phase_ratio": c["optimizer_only_seconds"] / g["optimizer_only_seconds"],
            "cpu_over_gpu_scoped_stage_ratio": c["scoped_end_to_end_seconds"] / g["scoped_end_to_end_seconds"],
            "cpu_nfev": c["nfev"],
            "gpu_nfev": g["nfev"],
            "cpu_nit": c["nit"],
            "gpu_nit": g["nit"],
            "cpu_termination_class": c["termination_class"],
            "gpu_termination_class": g["termination_class"],
        })
    cpu_optimizer = sum(row["cpu_optimizer_phase_seconds"] for row in rows)
    gpu_optimizer = sum(row["gpu_optimizer_phase_seconds"] for row in rows)
    cpu_scoped = sum(row["cpu_scoped_stage_seconds"] for row in rows)
    gpu_scoped = sum(row["gpu_scoped_stage_seconds"] for row in rows)
    return {
        "rows": rows,
        "consensus_chain_sum": {
            "cpu_optimizer_phase_seconds": cpu_optimizer,
            "gpu_optimizer_phase_seconds": gpu_optimizer,
            "cpu_scoped_stage_seconds": cpu_scoped,
            "gpu_scoped_stage_seconds": gpu_scoped,
            "cpu_over_gpu_optimizer_phase_ratio": cpu_optimizer / gpu_optimizer,
            "cpu_over_gpu_scoped_stage_ratio": cpu_scoped / gpu_scoped,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-attempt", type=Path, required=True)
    parser.add_argument("--gpu-attempt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RUN / "logs" / "preliminary_consensus_timing.json")
    args = parser.parse_args()
    cpu_attempt = resolve(args.cpu_attempt)
    gpu_attempt = resolve(args.gpu_attempt)
    cpu_config_hash = benchmark.expected_config_hash(cpu_attempt)
    gpu_config_hash = benchmark.expected_config_hash(gpu_attempt)
    if cpu_config_hash != gpu_config_hash:
        raise ValueError("CPU/GPU config hashes differ")
    cpu = validate_chain(cpu_attempt, "archived-cpu", "archived-cpu", cpu_config_hash)
    gpu = validate_chain(gpu_attempt, "fused-cuda", "fused-cuda", gpu_config_hash)
    postfit_path = RUN / "logs" / "fused-final-20k-repeatability.json"
    postfit = load_json(postfit_path)
    if postfit.get("status") != "passed" or not all(postfit.get("checks", {}).values()):
        raise ValueError("GPU 20 kb postfit repeatability log is not fully passed")
    build_path = RUN / "logs" / "fused-build.json"
    build = load_json(build_path)
    timing = paired_timing(cpu, gpu)
    random_root = benchmark.backend_root(cpu_attempt, "archived-cpu")
    random_stage_files = []
    for label, _, _, _ in benchmark.STAGES:
        path = random_root / "stages" / "random_joint" / f"{label}.json"
        if path.exists():
            random_stage_files.append({"stage": label, "path": str(path), "sha256": file_sha256(path)})
    output = args.out if args.out.is_absolute() else WORKSPACE / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "preliminary_consensus_chain_only",
        "artifact_integrity": "machine_generated_direct_json_dump",
        "candidate_scope": [CANDIDATE],
        "full_two_candidate_final_benchmark": "pending_random_joint_terminal_5m_2m_1m",
        "selection": "not_performed_in_preliminary_chain",
        "input_sha256": INPUT_SHA256,
        "config_sha256": cpu_config_hash,
        "phase_or_reference_opened": False,
        "cpu": cpu,
        "gpu": gpu,
        "timing": timing,
        "timing_definition": {
            "optimizer_phase": "stage fit timer including initial/final evaluations, accepted checkpoint I/O and history writes",
            "scoped_stage": "stage-scoped preprocessing through export/state save; excludes process imports and first compilation",
            "cuda_upload": "backend construction, CSR setup and extension-cache load; not pure PCIe transfer",
            "compile_seconds": build.get("elapsed_seconds"),
            "compile_log": str(build_path),
            "command_wall": "not inferred from timestamps; no cross-launch command-wall ratio reported",
        },
        "gpu_postfit_repeatability": {
            "path": str(postfit_path),
            "sha256": file_sha256(postfit_path),
            "status": postfit.get("status"),
            "checks_all_true": all(postfit.get("checks", {}).values()),
            "coordinates_array_sha256": postfit.get("coordinates", {}).get("saved_array_sha256"),
            "coordinates_file_sha256": postfit.get("coordinates", {}).get("saved_file_sha256")
            or postfit.get("saved_state", {}).get("coordinates_file_sha256"),
            "repeatability_mode": postfit.get("repeatability_mode"),
        },
        "random_joint_observed_but_excluded": {
            "terminal_through_1m": len(random_stage_files) == 3,
            "stage_files": random_stage_files,
            "used_in_timing": False,
            "used_in_selection": False,
        },
        "interpretation": "The paired ratios describe the completed consensus chain only and are not the final two-candidate formal speedup.",
        "no_l2_claim": True,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "output": str(output),
        "consensus_chain_sum": timing["consensus_chain_sum"],
        "random_joint_terminal_through_1m": payload["random_joint_observed_but_excluded"]["terminal_through_1m"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
