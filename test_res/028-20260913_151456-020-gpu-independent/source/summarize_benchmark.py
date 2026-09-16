"""Machine-generated CPU/GPU terminal comparison for the formal 1 Mb runs.

The script refuses to emit a final speedup until both backends contain both
candidates at 5m, 2m and 1m with matching frozen provenance and complete grids.
It reads stage JSON only; no timing values are transcribed into this module.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
INPUT = WORKSPACE / "inputs" / "P9016.snpfree.pairs.gz"
STAGES = (
    ("5m", 5_000_000, 300, 930),
    ("2m", 2_000_000, 200, 630),
    ("1m", 1_000_000, 240, 750),
)
CANDIDATES = ("consensus_joint", "random_joint")
FROZEN_INPUT_SHA256 = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"
FROZEN_COUNTS = {
    "raw_records": 1_703_888,
    "raw_intra": 1_135_454,
    "raw_inter": 568_434,
    "endpoint_total": 3_407_776,
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_workspace_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else WORKSPACE / path


def backend_root(attempt: Path, backend_label: str) -> Path:
    candidate = attempt / "backend_runs" / backend_label
    if candidate.is_dir():
        return candidate
    if (attempt / "stages").is_dir():
        return attempt
    raise FileNotFoundError(f"backend output directory not found: {candidate}")


def summary_path(root: Path) -> Path:
    path = root / "pipeline-summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing terminal pipeline summary: {path}")
    return path


def authoritative_summary(attempt: Path, root: Path, backend_label: str) -> tuple[Path, Path | None, dict | None]:
    """Use the immutable 1 Mb snapshot when fine continuation changed live files."""
    if backend_label != "fused-cuda":
        return summary_path(root), None, None
    manifest_path = attempt / "provenance" / "through-1m-snapshot.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing immutable fused 1m snapshot manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    entry = next((row for row in manifest.get("entries", []) if row.get("relative") == "pipeline-summary.json"), None)
    if entry is None:
        raise ValueError("fused 1m snapshot manifest has no pipeline-summary entry")
    source = Path(entry["source"])
    snapshot = Path(entry["snapshot"])
    if source != root / "pipeline-summary.json":
        raise ValueError("fused 1m snapshot source does not match live backend summary")
    if not source.exists() or not snapshot.exists():
        raise FileNotFoundError("fused 1m source or snapshot pipeline-summary is missing")
    if file_sha256(snapshot) != entry["sha256"]:
        raise ValueError("fused 1m pipeline-summary snapshot hash does not match immutable manifest")
    if manifest.get("pipeline_summary_sha256") != entry["sha256"]:
        raise ValueError("fused 1m manifest pipeline_summary_sha256 mismatch")
    return snapshot, snapshot.parent, manifest


def validate_snapshot_entries(manifest: dict | None) -> None:
    if manifest is None:
        return
    for entry in manifest.get("entries", []):
        relative = entry.get("relative", "")
        if not (relative.startswith("stages/") or relative == "selection-1m.json"):
            continue
        for key in ("source", "snapshot"):
            path = Path(entry[key])
            if not path.exists() or file_sha256(path) != entry.get("sha256"):
                raise ValueError(f"immutable fused 1m snapshot hash mismatch: {path}")

def expected_config_hash(attempt: Path) -> str:
    config = attempt / "config.json"
    if not config.exists():
        config = RUN / "config.json"
    return file_sha256(config)


def validate_budget(stage: dict, label: str) -> None:
    budget = stage.get("data_budget", {})
    if budget.get("raw_records") != FROZEN_COUNTS["raw_records"]:
        raise ValueError(f"{stage['candidate_id']}/{label}: raw record total mismatch")
    if budget.get("raw_same_bin", 0) + budget.get("raw_cis_offdiag", 0) != FROZEN_COUNTS["raw_intra"]:
        raise ValueError(f"{stage['candidate_id']}/{label}: raw intra total mismatch")
    if budget.get("raw_inter") != FROZEN_COUNTS["raw_inter"]:
        raise ValueError(f"{stage['candidate_id']}/{label}: raw inter total mismatch")
    if budget.get("endpoint_total") != FROZEN_COUNTS["endpoint_total"]:
        raise ValueError(f"{stage['candidate_id']}/{label}: endpoint total mismatch")
    if not all(budget.get(key) is True for key in ("raw_conserved", "aggregate_conserved", "endpoint_conserved")):
        raise ValueError(f"{stage['candidate_id']}/{label}: conservation flags are not all true")


def validate_stage(stage_path: Path, candidate: str, label: str, bin_size: int,
                   maxiter: int, maxfun: int, input_hash: str, config_hash: str) -> dict:
    stage = load_json(stage_path)
    if (stage.get("candidate_id"), stage.get("stage"), stage.get("bin_size_bp")) != (candidate, label, bin_size):
        raise ValueError(f"stage identity mismatch: {stage_path}")
    validate_budget(stage, label)
    final_coordinates = stage.get("final_coordinates", {})
    if final_coordinates.get("n_tracks") != 40 or not final_coordinates.get("full_grid"):
        raise ValueError(f"{candidate}/{label}: full 40-track grid export missing")
    coordinate_path = resolve_workspace_path(final_coordinates["path"])
    if not coordinate_path.exists() or file_sha256(coordinate_path) != final_coordinates.get("sha256"):
        raise ValueError(f"{candidate}/{label}: exported coordinate hash mismatch")
    state = stage.get("resume_state", {})
    if state.get("input_sha256") != input_hash or state.get("config_sha256") != config_hash:
        raise ValueError(f"{candidate}/{label}: resume provenance hash mismatch")
    shape = state.get("coordinates_shape")
    if shape is None or len(shape) != 3 or shape[0] != 2:
        raise ValueError(f"{candidate}/{label}: saved coordinate state is not copy-first")
    if not state.get("coordinates_sha256"):
        raise ValueError(f"{candidate}/{label}: resume coordinate hash is missing")
    if state.get("positions_sha256") != final_coordinates.get("positions_sha256"):
        raise ValueError(f"{candidate}/{label}: position grid hash is inconsistent")
    if state.get("chromosomes_sha256") != final_coordinates.get("chromosomes_sha256"):
        raise ValueError(f"{candidate}/{label}: chromosome grid hash is inconsistent")
    fit = stage.get("fit", {})
    history = stage.get("accepted_history", [])
    if not history or any("gradient_l2" not in row or "gradient_max_abs" not in row for row in history):
        raise ValueError(f"{candidate}/{label}: accepted history lacks gradient norms")
    if fit.get("nit", 0) > maxiter or fit.get("nfev", 0) > maxfun:
        raise ValueError(f"{candidate}/{label}: optimizer budget exceeded")
    termination_class = fit.get("termination_class")
    if termination_class not in {"converged", "budget_not_converged", "line_search_abort", "optimizer_failed"}:
        raise ValueError(f"{candidate}/{label}: unknown termination class")
    if termination_class == "budget_not_converged" and not (
        int(fit.get("nit", 0)) >= int(maxiter)
        or int(fit.get("nfev", 0)) >= int(maxfun)
        or int(fit.get("scipy_nfev", 0)) >= int(maxfun)
    ):
        raise ValueError(f"{candidate}/{label}: budget_not_converged without actual budget limit")
    timing = stage.get("timing", {})
    final_components = fit.get("final_components", {})
    return {
        "candidate": candidate,
        "stage": label,
        "bin_size_bp": bin_size,
        "expected_maxiter": maxiter,
        "expected_maxfun": maxfun,
        "backend": stage.get("backend"),
        "status": stage.get("status"),
        "termination_class": termination_class,
        "scipy_status": fit.get("status"),
        "scipy_message": fit.get("message"),
        "success": fit.get("success"),
        "nit": fit.get("nit"),
        "nfev": fit.get("nfev"),
        "scipy_nfev": fit.get("scipy_nfev"),
        "accepted_history_length": len(history),
        "checkpoint_count": len(stage.get("checkpoint_paths", [])),
        "checkpoint_io_inside_optimizer_timer": True,
        "optimizer_only_seconds": timing.get("optimizer_only_seconds"),
        "scoped_end_to_end_seconds": timing.get("scoped_end_to_end_seconds"),
        "preprocess_seconds": timing.get("preprocess_seconds"),
        "initialization_seconds": timing.get("initialization_seconds"),
        "backend_constructor_seconds": timing.get("backend_constructor_seconds"),
        "upload_seconds": timing.get("upload_seconds"),
        "warmup_seconds": timing.get("warmup_seconds"),
        "export_seconds": timing.get("export_seconds"),
        "objective_evaluations": timing.get("objective_evaluations"),
        "accepted_iterations": timing.get("accepted_iterations"),
        "peak_cuda_memory_bytes": timing.get("peak_cuda_memory_bytes"),
        "budget_validated": True,
        "coordinates_sha256": final_coordinates.get("sha256"),
        "positions_sha256": final_coordinates.get("positions_sha256"),
        "chromosomes_sha256": final_coordinates.get("chromosomes_sha256"),
        "n_tracks": final_coordinates.get("n_tracks"),
        "n_beads": final_coordinates.get("n_beads"),
        "initial_total": fit.get("initial_total"),
        "final_total": fit.get("final_total"),
        "initial_count_nll_normalized": fit.get("initial_components", {}).get("count_nll_normalized"),
        "final_count_nll_normalized": final_components.get("count_nll_normalized"),
        "final_gradient_l2": fit.get("final_gradient_l2"),
        "final_gradient_max_abs": fit.get("final_gradient_max_abs"),
        "final_components": final_components,
    }


def validate_backend(attempt: Path, backend_label: str, expected_backend: str,
                     input_hash: str, config_hash: str) -> dict:
    root = backend_root(attempt, backend_label)
    authoritative_path, snapshot_stage_root, snapshot_manifest = authoritative_summary(attempt, root, backend_label)
    validate_snapshot_entries(snapshot_manifest)
    summary = load_json(authoritative_path)
    if summary.get("status") != "training_complete" or summary.get("through") != "1m":
        raise ValueError(f"{backend_label}: authoritative terminal summary is not complete through 1m")
    if summary.get("input_sha256") != input_hash or summary.get("config_sha256") != config_hash:
        raise ValueError(f"{backend_label}: terminal summary provenance mismatch")
    rows = []
    for label, bin_size, maxiter, maxfun in STAGES:
        for candidate in CANDIDATES:
            stage_root = snapshot_stage_root if snapshot_stage_root is not None else root
            path = stage_root / "stages" / candidate / f"{label}.json"
            if not path.exists():
                raise FileNotFoundError(f"missing required stage: {path}")
            row = validate_stage(path, candidate, label, bin_size, maxiter, maxfun, input_hash, config_hash)
            if expected_backend not in str(row.get("backend")):
                raise ValueError(f"{path}: backend identity does not contain {expected_backend}")
            if label == "1m":
                if row.get("n_beads") != 5290 or row.get("n_tracks") != 40:
                    raise ValueError(f"{path}: 1 Mb physical grid is not 5290 beads/40 tracks")
                if load_json(path).get("resume_state", {}).get("coordinates_shape") != [2, 2645, 3]:
                    raise ValueError(f"{path}: 1 Mb coordinate state is not [2,2645,3]")
            rows.append(row)
    selection_path = (snapshot_stage_root / "selection-1m.json") if snapshot_stage_root is not None else root / "selection-1m.json"
    selection = load_json(selection_path)
    if selection.get("criterion") != "count_nll_normalized":
        raise ValueError(f"{backend_label}: selection criterion is not count_nll_normalized")
    if selection.get("candidate_ids") != list(CANDIDATES):
        raise ValueError(f"{backend_label}: selection candidate order is not frozen")
    final_values = {
        row["candidate"]: row["final_count_nll_normalized"]
        for row in rows if row["stage"] == "1m"
    }
    selection_values = selection.get("values", {})
    if set(selection_values) != set(CANDIDATES) or any(
        not math.isclose(float(selection_values[candidate]), float(final_values[candidate]), rel_tol=0.0, abs_tol=1e-12)
        for candidate in CANDIDATES
    ):
        raise ValueError(f"{backend_label}: selection values do not match final 1 Mb stage components")
    selected_expected = min(CANDIDATES, key=lambda candidate: (float(final_values[candidate]), CANDIDATES.index(candidate)))
    if selection.get("selected_candidate") != selected_expected:
        raise ValueError(f"{backend_label}: selected candidate is not the count_nll_normalized minimum")
    if not math.isclose(float(selection.get("selected_value")), float(final_values[selected_expected]), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{backend_label}: selected_value does not match selected 1 Mb final value")
    blocked_termination_rows = [
        row for row in rows if row["termination_class"] not in {"converged", "budget_not_converged"}
    ]
    if blocked_termination_rows:
        raise ValueError(f"{backend_label}: final comparison refuses nonterminal optimizer terminations: {blocked_termination_rows}")
    return {
        "attempt": str(attempt),
        "backend_label": backend_label,
        "backend": expected_backend,
        "pipeline_summary": str(authoritative_path),
        "immutable_1m_snapshot_manifest": str(attempt / "provenance" / "through-1m-snapshot.json") if snapshot_manifest else None,
        "selection": selection,
        "stages": rows,
        "termination_gate": "all stages converged or reached an actual configured iteration/function budget",
    }


def ratios(cpu: dict, gpu: dict) -> list[dict]:
    cpu_rows = {(row["candidate"], row["stage"]): row for row in cpu["stages"]}
    gpu_rows = {(row["candidate"], row["stage"]): row for row in gpu["stages"]}
    output = []
    for key in cpu_rows:
        c = cpu_rows[key]
        g = gpu_rows[key]
        output.append({
            "candidate": key[0],
            "stage": key[1],
            "bin_size_bp": c["bin_size_bp"],
            "cpu_status": c["status"],
            "gpu_status": g["status"],
            "cpu_initial_total": c["initial_total"],
            "gpu_initial_total": g["initial_total"],
            "cpu_final_total": c["final_total"],
            "gpu_final_total": g["final_total"],
            "cpu_initial_count_nll_normalized": c["initial_count_nll_normalized"],
            "gpu_initial_count_nll_normalized": g["initial_count_nll_normalized"],
            "cpu_final_count_nll_normalized": c["final_count_nll_normalized"],
            "gpu_final_count_nll_normalized": g["final_count_nll_normalized"],
            "optimizer_cpu_seconds": c["optimizer_only_seconds"],
            "optimizer_gpu_seconds": g["optimizer_only_seconds"],
            "optimizer_cpu_over_gpu_ratio": c["optimizer_only_seconds"] / g["optimizer_only_seconds"],
            "scoped_cpu_seconds": c["scoped_end_to_end_seconds"],
            "scoped_gpu_seconds": g["scoped_end_to_end_seconds"],
            "scoped_cpu_over_gpu_ratio": c["scoped_end_to_end_seconds"] / g["scoped_end_to_end_seconds"],
            "cpu_nfev": c["nfev"],
            "gpu_nfev": g["nfev"],
            "cpu_nit": c["nit"],
            "gpu_nit": g["nit"],
            "cpu_termination_class": c["termination_class"],
            "gpu_termination_class": g["termination_class"],
        })
    return output


def timing_totals(timing: list[dict]) -> dict:
    def aggregate(rows: list[dict]) -> dict:
        optimizer_cpu = sum(row["optimizer_cpu_seconds"] for row in rows)
        optimizer_gpu = sum(row["optimizer_gpu_seconds"] for row in rows)
        scoped_cpu = sum(row["scoped_cpu_seconds"] for row in rows)
        scoped_gpu = sum(row["scoped_gpu_seconds"] for row in rows)
        return {
            "stage_count": len(rows),
            "stages": [row["stage"] for row in rows],
            "optimizer_cpu_seconds": optimizer_cpu,
            "optimizer_gpu_seconds": optimizer_gpu,
            "optimizer_cpu_over_gpu_ratio_of_sums": optimizer_cpu / optimizer_gpu,
            "scoped_cpu_seconds": scoped_cpu,
            "scoped_gpu_seconds": scoped_gpu,
            "scoped_cpu_over_gpu_ratio_of_sums": scoped_cpu / scoped_gpu,
        }

    return {
        "by_candidate": {
            candidate: aggregate([row for row in timing if row["candidate"] == candidate])
            for candidate in CANDIDATES
        },
        "all_candidates": aggregate(timing),
        "ratio_definition": "sum(CPU stage seconds) / sum(GPU stage seconds); never a mean of per-stage ratios",
        "stage_sum_not_command_wall": True,
    }


def write_timing_csv(comparison: dict, path: Path) -> None:
    columns = [
        "candidate", "stage", "bin_size_bp",
        "optimizer_cpu_seconds", "optimizer_gpu_seconds", "optimizer_cpu_over_gpu_ratio",
        "scoped_cpu_seconds", "scoped_gpu_seconds", "scoped_cpu_over_gpu_ratio",
        "cpu_nit", "gpu_nit", "cpu_nfev", "gpu_nfev",
        "cpu_status", "gpu_status", "cpu_termination_class", "gpu_termination_class",
        "cpu_initial_total", "gpu_initial_total", "cpu_final_total", "gpu_final_total",
        "cpu_initial_count_nll_normalized", "gpu_initial_count_nll_normalized",
        "cpu_final_count_nll_normalized", "gpu_final_count_nll_normalized",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison["timing_comparison"])


def write_plot(comparison: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    stages = [row[0] for row in STAGES]
    stage_rows = {}
    for row in comparison["timing_comparison"]:
        stage_rows.setdefault(row["stage"], []).append(row)
    cpu_optimizer = [np.mean([row["optimizer_cpu_seconds"] for row in stage_rows[label]]) for label in stages]
    gpu_optimizer = [np.mean([row["optimizer_gpu_seconds"] for row in stage_rows[label]]) for label in stages]
    cpu_scoped = [np.mean([row["scoped_cpu_seconds"] for row in stage_rows[label]]) for label in stages]
    gpu_scoped = [np.mean([row["scoped_gpu_seconds"] for row in stage_rows[label]]) for label in stages]
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    figure, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), dpi=300)
    x = np.arange(len(stages))
    width = 0.36
    axes[0].bar(x - width / 2, cpu_optimizer, width, label="archived CPU", color="#4c78a8")
    axes[0].bar(x + width / 2, gpu_optimizer, width, label="fused CUDA", color="#f58518")
    axes[0].set_title("Optimization phase incl. checkpoints\n(mean across two initializations)")
    axes[1].bar(x - width / 2, cpu_scoped, width, label="archived CPU", color="#4c78a8")
    axes[1].bar(x + width / 2, gpu_scoped, width, label="fused CUDA", color="#f58518")
    axes[1].set_title("Scoped end-to-end\n(mean across two initializations)")
    for axis in axes:
        axis.set_xticks(x, stages)
        axis.set_ylabel("seconds (log scale)")
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-attempt", type=Path, required=True)
    parser.add_argument("--gpu-attempt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=RUN)
    args = parser.parse_args()
    input_hash = file_sha256(INPUT)
    if input_hash != FROZEN_INPUT_SHA256:
        raise ValueError("input SHA256 does not match frozen cohort")
    cpu_attempt = args.cpu_attempt if args.cpu_attempt.is_absolute() else WORKSPACE / args.cpu_attempt
    gpu_attempt = args.gpu_attempt if args.gpu_attempt.is_absolute() else WORKSPACE / args.gpu_attempt
    cpu_hash = expected_config_hash(cpu_attempt)
    gpu_hash = expected_config_hash(gpu_attempt)
    if cpu_hash != gpu_hash:
        raise ValueError(f"CPU/GPU config hashes differ: {cpu_hash} vs {gpu_hash}")
    cpu = validate_backend(cpu_attempt, "archived-cpu", "archived-cpu", input_hash, cpu_hash)
    gpu = validate_backend(gpu_attempt, "fused-cuda", "fused-cuda", input_hash, gpu_hash)
    timing = ratios(cpu, gpu)
    machine_log = RUN / "logs" / "fused-torch-baseline-1m-machine.json"
    build_log = RUN / "logs" / "fused-build.json"
    launch_manifest = cpu_attempt / "launch-manifest.json"
    launch_metadata = load_json(launch_manifest) if launch_manifest.exists() else {}
    terminal_status_path = cpu_attempt / "terminal.status.json"
    if not terminal_status_path.exists():
        raise FileNotFoundError(f"missing CPU terminal status: {terminal_status_path}")
    terminal_status = load_json(terminal_status_path)
    if terminal_status.get("status") != "terminal" or terminal_status.get("runner_exit_code") != 0 or terminal_status.get("shell_exit_code") != 0:
        raise ValueError(f"CPU terminal status is not clean: {terminal_status}")
    cpu_terminal_status = {
        "path": str(terminal_status_path),
        "sha256": file_sha256(terminal_status_path),
        "status": terminal_status.get("status"),
        "runner_exit_code": terminal_status.get("runner_exit_code"),
        "shell_exit_code": terminal_status.get("shell_exit_code"),
    }
    comparison = {
        "status": "completed_terminal_comparison",
        "artifact_integrity": "machine_generated_direct_json_dump",
        "input_sha256": input_hash,
        "config_sha256": cpu_hash,
        "cpu": cpu,
        "gpu": gpu,
        "timing_comparison": timing,
        "timing_totals": timing_totals(timing),
        "cpu_terminal_status": cpu_terminal_status,
        "timing_definition": {
            "optimizer_only": "optimization phase timer; initial/final evaluations plus accepted checkpoint/history I/O are inside this timer",
            "scoped_end_to_end": "stage timer from preprocess through export and state save",
            "warmup_upload_constructor": "reported separately and excluded from optimizer-only ratio; backend_constructor_seconds is canonical, while CUDA upload_seconds covers fused construction, CSR setup and extension-cache load rather than pure PCIe transfer",
            "gpu_compilation": "reported in fused backend metadata/microbenchmark; not folded into steady-state ratio",
            "stage_sums": "sum of the three 5m/2m/1m stage timers per candidate and across both candidates; not command wall-clock",
        },
        "gpu_compilation": {
            "source_log": str(build_log),
            "status": load_json(build_log).get("status") if build_log.exists() else "missing",
            "compile_seconds": load_json(build_log).get("elapsed_seconds") if build_log.exists() else None,
            "excluded_from_steady_state_pipeline_ratios": True,
        },
        "selection_match": cpu["selection"].get("selected_candidate") == gpu["selection"].get("selected_candidate"),
        "selection": {
            "cpu": cpu["selection"],
            "gpu": gpu["selection"],
        },
        "supplementary_history": {
            "original_020": "historical optimizer wall times are supplementary only and are not used for this speedup",
            "single_checkpoint_machine_audit": str(machine_log),
            "machine_audit_formal_pipeline_started": load_json(machine_log).get("formal_pipeline_started") if machine_log.exists() else None,
        },
        "limitations": {
            "cpu_gpu_overlap": launch_metadata.get("overlap_observation") or launch_metadata.get("resource_recordings", {}).get("observed_gpu_fine_work") or launch_metadata.get("gpu_overlap_note") or "not recorded",
            "hardware_context": "GPU stages ran in a GPU-visible execution context; archived CPU stage ran in this workspace context",
            "l2_claim": "none; timing/numerical pipeline comparison is not haplotype-recovery evidence",
        },
    }
    out_dir = args.out_dir if args.out_dir.is_absolute() else WORKSPACE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "logs" / "benchmark_comparison.json"
    csv_path = out_dir / "logs" / "benchmark_comparison.csv"
    plot_path = out_dir / "plots" / "benchmark_timing.png"
    audit_path = out_dir / "logs" / "terminal_benchmark_audit.json"
    comparison["csv_path"] = str(csv_path)
    comparison["plot_path"] = str(plot_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)
        handle.write("\n")
    audit = {
        "status": "passed",
        "artifact_integrity": "machine_generated_direct_json_dump",
        "checked": {
            "input_hash": input_hash,
            "config_hash_equal": cpu_hash == gpu_hash,
            "six_cpu_stage_files": len(cpu["stages"]) == 6,
            "six_gpu_stage_files": len(gpu["stages"]) == 6,
            "all_exports_have_40_tracks": all(row["n_tracks"] == 40 for row in cpu["stages"] + gpu["stages"]),
            "all_budgets_conserved": all(row["budget_validated"] for row in cpu["stages"] + gpu["stages"]),
            "selection_after_both": all(item["selection"].get("status") == "selected_after_both_candidates_1m" for item in (cpu, gpu)),
            "coordinate_hashes_checked": True,
            "termination_classes_checked": True,
            "cpu_terminal_status_clean": terminal_status.get("status") == "terminal" and terminal_status.get("runner_exit_code") == 0 and terminal_status.get("shell_exit_code") == 0,
        },
        "comparison_path": str(comparison_path),
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "cpu_terminal_status": cpu_terminal_status,
        "no_l2_claim": True,
    }
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_timing_csv(comparison, csv_path)
    write_plot(comparison, plot_path)
    print(json.dumps({"status": "passed", "comparison": str(comparison_path), "csv": str(csv_path), "audit": str(audit_path), "plot": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()
