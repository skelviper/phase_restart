"""Summarize accepted fused-CUDA fine stages without running objective evaluations."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import summarize_benchmark as benchmark

RUN = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
FINE_STAGES = (
    ("500k", 500_000),
    ("200k", 200_000),
    ("100k", 100_000),
    ("50k", 50_000),
    ("20k", 20_000),
)
CANDIDATE = "random_joint"
DEFAULT_ATTEMPT = RUN / "backend_runs" / "fused-cuda" / "attempt-20260913T165036Z"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else WORKSPACE / path


def summarize_stage(path: Path, label: str, bin_size: int, input_hash: str, config_hash: str) -> dict:
    row = benchmark.validate_stage(path, CANDIDATE, label, bin_size, 240, 750, input_hash, config_hash)
    stage = load_json(path)
    budget = stage["data_budget"]
    fit = stage["fit"]
    if row["n_tracks"] != 40 or row["termination_class"] not in {"converged", "budget_not_converged"}:
        raise ValueError(f"{path}: fine stage is not an accepted 40-track terminal result")
    if row["termination_class"] == "budget_not_converged" and row["nit"] < 240 and row["nfev"] < 750:
        raise ValueError(f"{path}: budget_not_converged did not reach configured fine budget")
    if budget["n_loci"] != stage["resume_state"].get("n_loci"):
        raise ValueError(f"{path}: n_loci mismatch between budget and saved state")
    return {
        "stage": label,
        "bin_size_bp": bin_size,
        "n_loci": budget["n_loci"],
        "n_pairs": budget["n_eligible_pairs"],
        "n_observed_nonzero_pairs": budget["n_observed_nonzero_pairs"],
        "raw_records": budget["raw_records"],
        "optimizer_phase_seconds": row["optimizer_only_seconds"],
        "scoped_stage_seconds": row["scoped_end_to_end_seconds"],
        "backend_constructor_seconds": row["backend_constructor_seconds"],
        "upload_seconds": row["upload_seconds"],
        "warmup_seconds": row["warmup_seconds"],
        "preprocess_seconds": row["preprocess_seconds"],
        "export_seconds": row["export_seconds"],
        "nit": row["nit"],
        "nfev": row["nfev"],
        "scipy_nfev": row["scipy_nfev"],
        "accepted_iterations": row["accepted_iterations"],
        "peak_cuda_memory_bytes": row["peak_cuda_memory_bytes"],
        "status": row["status"],
        "termination_class": row["termination_class"],
        "initial_total": row["initial_total"],
        "final_total": row["final_total"],
        "delta_total_initial_minus_final": row["initial_total"] - row["final_total"],
        "initial_count_nll_normalized": row["initial_count_nll_normalized"],
        "final_count_nll_normalized": row["final_count_nll_normalized"],
        "coordinates_sha256": row["coordinates_sha256"],
        "positions_sha256": row["positions_sha256"],
        "chromosomes_sha256": row["chromosomes_sha256"],
        "stage_json_sha256": file_sha256(path),
        "stage_json": str(path),
        "budget_validated": row["budget_validated"],
        "full_grid": row["n_tracks"] == 40,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    columns = [
        "stage", "bin_size_bp", "n_loci", "n_pairs", "n_observed_nonzero_pairs", "raw_records",
        "optimizer_phase_seconds", "scoped_stage_seconds", "backend_constructor_seconds", "upload_seconds",
        "warmup_seconds", "preprocess_seconds", "export_seconds", "nit", "nfev", "scipy_nfev",
        "accepted_iterations", "peak_cuda_memory_bytes", "status", "termination_class",
        "initial_total", "final_total", "delta_total_initial_minus_final",
        "initial_count_nll_normalized", "final_count_nll_normalized", "coordinates_sha256",
        "positions_sha256", "chromosomes_sha256", "stage_json_sha256", "budget_validated", "full_grid",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_figure(rows: list[dict], path: Path) -> None:
    labels = [row["stage"] for row in rows]
    x = np.arange(len(labels))
    width = 0.36
    optimizer = [row["optimizer_phase_seconds"] for row in rows]
    scoped = [row["scoped_stage_seconds"] for row in rows]
    delta_j = [row["delta_total_initial_minus_final"] for row in rows]
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    figure, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), dpi=300)
    axes[0].bar(x - width / 2, optimizer, width, label="optimization phase", color="#4c78a8")
    axes[0].bar(x + width / 2, scoped, width, label="scoped stage", color="#f58518")
    axes[0].set_title("GPU fine-stage timing")
    axes[0].set_ylabel("seconds (log scale)")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].bar(x, delta_j, width * 1.6, color="#54a24b")
    axes[1].set_title("Within-stage J reduction")
    axes[1].set_ylabel("initial J - final J")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Random-joint selected fine continuation; J is compared only within each resolution", fontsize=7)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, default=DEFAULT_ATTEMPT)
    parser.add_argument("--csv", type=Path, default=RUN / "logs" / "gpu_fine_timing.csv")
    parser.add_argument("--json", type=Path, default=RUN / "logs" / "gpu_fine_timing.json")
    parser.add_argument("--plot", type=Path, default=RUN / "plots" / "gpu_fine_timing.png")
    args = parser.parse_args()
    attempt = resolve(args.attempt)
    root = benchmark.backend_root(attempt, "fused-cuda")
    config_hash = benchmark.expected_config_hash(attempt)
    input_hash = benchmark.file_sha256(benchmark.INPUT)
    if input_hash != benchmark.FROZEN_INPUT_SHA256:
        raise ValueError("input hash is not the frozen P9016 input")
    rows = []
    for label, bin_size in FINE_STAGES:
        path = root / "stages" / CANDIDATE / f"{label}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing GPU fine stage: {path}")
        rows.append(summarize_stage(path, label, bin_size, input_hash, config_hash))
    output_csv = resolve(args.csv)
    output_json = resolve(args.json)
    output_plot = resolve(args.plot)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed_gpu_fine_stage_summary",
        "artifact_integrity": "machine_generated_direct_json_dump",
        "candidate": CANDIDATE,
        "attempt": str(attempt),
        "input_sha256": input_hash,
        "config_sha256": config_hash,
        "fine_stage_order": [label for label, _ in FINE_STAGES],
        "stage_count": len(rows),
        "rows": rows,
        "quality_scope": "initial_to_final_J is reported within each resolution only; absolute NLL is not compared across resolutions",
        "no_new_objective_evaluation": True,
        "csv_path": str(output_csv),
        "plot_path": str(output_plot),
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(rows, output_csv)
    write_figure(rows, output_plot)
    print(json.dumps({"status": payload["status"], "json": str(output_json), "csv": str(output_csv), "plot": str(output_plot)}, indent=2))


if __name__ == "__main__":
    main()
