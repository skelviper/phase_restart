"""核验并补做056 G-original端点的development-validation copy连接实验A。"""
from __future__ import annotations

import csv
from dataclasses import replace
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
SOURCE = ROOT / "test_res/056-20260916T152353Z-pro-review-experiments"
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
for path in (ROOT, SOURCE / "code", S045):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import data_io  # noqa: E402
from nonref_core import fixed_state_fullgrid  # noqa: E402
from pr import contact_model  # noqa: E402
from pr.solver_state import load_solver_state, sha256_file  # noqa: E402


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, order="C").tobytes(order="C")).hexdigest()


def swap_suffix(coords, data, chromosome: str, start: int) -> np.ndarray:
    index = {str(name): i for i, name in enumerate(data.chromosome_names)}[chromosome]
    slc = data.chromosome_slice(index)
    suffix = slice(slc.start + int(start), slc.stop)
    result = np.asarray(coords, dtype=np.float64).copy()
    result[:, suffix] = result[::-1, suffix]
    return result


def pointset_equal(original: np.ndarray, changed: np.ndarray) -> bool:
    direct = np.all(changed[0] == original[0], axis=1) & np.all(changed[1] == original[1], axis=1)
    swapped = np.all(changed[0] == original[1], axis=1) & np.all(changed[1] == original[0], axis=1)
    return bool(np.all(direct | swapped))


def load_dev_aggregate(train):
    with np.load(SOURCE / "inputs/test_contacts.npz", allow_pickle=False) as payload:
        ci = np.asarray(payload["ci"], dtype=np.int64)
        p1 = np.asarray(payload["p1"], dtype=np.int64)
        cj = np.asarray(payload["cj"], dtype=np.int64)
        p2 = np.asarray(payload["p2"], dtype=np.int64)
    observed = contact_model.aggregate_from_arrays(
        tuple(train.chromosome_names), tuple(int(x) for x in train.chromosome_lengths),
        ci, p1, cj, p2, 1_000_000)
    dev = replace(observed, exposure=np.asarray(train.exposure, dtype=np.float64).copy(),
                  exposure_mode="train_fixed_G-original")
    dev.assert_consistent()
    return dev


def main() -> int:
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    terminal_path = RUN / "terminal.json"
    terminal = {
        "schema": "p9016-copy-link-dev-validation-a-terminal-v1",
        "status": "running",
        "started_at_utc": utc_now(),
        "experiment_A": "running",
        "experiment_B": "not_started",
        "reference_opened": False,
        "phase_opened": False,
    }
    write_json(terminal_path, terminal)
    started = time.perf_counter()
    try:
        source_checks = []
        for relative, expected in config["source_hashes"].items():
            path = ROOT / relative
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"source SHA256 mismatch: {relative}: {actual} != {expected}")
            source_checks.append({"path": relative, "sha256": actual, "matched": True})

        train = data_io.load_aggregate(SOURCE / "inputs/G-original_1000000_aggregate.npz")
        train.assert_consistent()
        dev = load_dev_aggregate(train)
        expected_shape = (2, int(config["n_loci"]), 3)
        if int(dev.n_loci) != config["n_loci"] or int(dev.n_pairs) != config["eligible_pairs"]:
            raise RuntimeError("development grid identity mismatch")
        if int(dev.raw_cis_offdiag + dev.raw_inter) != config["development_validation"]["offdiag_records"]:
            raise RuntimeError("development Noff mismatch")
        if int(dev.raw_records) != config["development_validation"]["records"]:
            raise RuntimeError("development record count mismatch")
        if not np.array_equal(dev.exposure, train.exposure):
            raise RuntimeError("development exposure is not the frozen train exposure")

        previous = json.loads((SOURCE / "results/experiment3_nonreference.json").read_text(encoding="utf-8"))
        previous_rows = {int(row["seed"]): row for row in previous["rows"]
                         if row["condition"] == "G-original"}
        manifest = json.loads((SOURCE / "states/experiment3_splices/manifest.json").read_text(encoding="utf-8"))
        splice_items = {(int(item["fit_id"].rsplit("seed", 1)[1]), item["state_id"]): item
                        for item in manifest["states"] if item["fit_id"].startswith("G-original-seed")}
        expected_states = [(chrom, int(start), f"splice-{chrom}-{int(start):03d}")
                           for chrom, starts in config["splices"].items() for start in starts]
        if len(expected_states) != 8 or len(splice_items) != 16:
            raise RuntimeError("expected exactly 8 splices per seed and 16 reusable splice artifacts")

        prepared = {}
        artifact_checks = []
        for seed in config["seeds"]:
            fit_id = f"G-original-seed{seed}"
            state_path = SOURCE / "states/experiment3" / fit_id / "1Mb/solver_state.npz"
            state = load_solver_state(state_path)
            expected_solver = config["endpoint_solver_hashes"][str(seed)]
            if state["sha256"] != expected_solver or state["coordinates"].shape != expected_shape:
                raise RuntimeError(f"endpoint identity mismatch for seed {seed}")
            coords = state["coordinates"]
            seed_states = [("original", None, None, coords, state["sha256"], array_sha256(coords))]
            artifact_checks.append({"seed": seed, "state_id": "original", "path": str(state_path.relative_to(ROOT)),
                                    "file_sha256": state["sha256"], "coordinate_sha256": array_sha256(coords),
                                    "identity_verified": True})
            for chromosome, splice_index, state_id in expected_states:
                item = splice_items[(seed, state_id)]
                path = ROOT / item["path"]
                actual_file = sha256_file(path)
                with np.load(path, allow_pickle=False) as payload:
                    changed = np.asarray(payload["coordinates"], dtype=np.float64)
                actual_coord = array_sha256(changed)
                regenerated = swap_suffix(coords, train, chromosome, splice_index)
                if (actual_file != item["file_sha256"] or actual_coord != item["coordinate_sha256"]
                        or not np.array_equal(changed, regenerated) or not pointset_equal(coords, changed)):
                    raise RuntimeError(f"splice identity mismatch for {fit_id}/{state_id}")
                seed_states.append((state_id, chromosome, splice_index, changed, actual_file, actual_coord))
                artifact_checks.append({"seed": seed, "state_id": state_id, "path": item["path"],
                                        "file_sha256": actual_file, "coordinate_sha256": actual_coord,
                                        "identity_verified": True, "pointset_conserved": True})
            prepared[seed] = {"fit_id": fit_id, "p": float(state["p"]), "states": seed_states}

        preflight = {
            "status": "passed",
            "source_hashes": source_checks,
            "artifacts": artifact_checks,
            "artifact_count": len(artifact_checks),
            "development_budget": dev.budget(),
            "train_exposure_sha256": array_sha256(train.exposure),
            "development_exposure_sha256": array_sha256(dev.exposure),
            "formal_fullgrid_calls_before_preflight": 0,
            "reference_opened": False,
            "phase_opened": False,
        }
        write_json(RUN / "preflight.json", preflight)

        rows = []
        call_ledger = []
        total_host = 0.0
        total_gpu = 0.0
        for seed in config["seeds"]:
            entry = prepared[seed]
            original_nll = None
            for call_index, (state_id, chromosome, splice_index, coords, file_hash, coord_hash) in enumerate(entry["states"], 1):
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                host_start = time.perf_counter()
                event_start.record()
                score, _ = fixed_state_fullgrid(
                    dev, coords, entry["p"], keep_inter_arrays=False, include_penalties=False)
                event_end.record()
                event_end.synchronize()
                host_seconds = time.perf_counter() - host_start
                gpu_seconds = float(event_start.elapsed_time(event_end)) / 1000.0
                total_host += host_seconds
                total_gpu += gpu_seconds
                if score["objective_fg_calls"] != 1 or score["pair_kernel_forward_passes"] != 1:
                    raise RuntimeError("one logical state did not use exactly one contact forward")
                if score["regularizer_full_pair_passes"] != 0 or score["penalties_raw"] or score["penalties_weighted"]:
                    raise RuntimeError("regularizer was evaluated in experiment A")
                no_k0 = float(score["offdiag_data_no_K0_nat_per_contact"])
                direct = math.log(float(score["Zall"])) - float(score["observed_log_rate_offdiag"]) / float(score["Noff"])
                if abs(no_k0 - direct) >= 1e-12 or int(score["Noff"]) != config["development_validation"]["offdiag_records"]:
                    raise RuntimeError("primary metric formula or denominator mismatch")
                if state_id == "original":
                    original_nll = no_k0
                    previous_original = float(previous_rows[seed]["test_primary_nll_nat_per_offdiag"])
                    if abs(original_nll - previous_original) >= 1e-10:
                        raise RuntimeError(f"original development score did not reproduce 056 for seed {seed}")
                    delta = 0.0
                    train_delta = 0.0
                else:
                    delta = no_k0 - float(original_nll)
                    old_splice = next(row for row in previous_rows[seed]["splices"] if row["state_id"] == state_id)
                    train_delta = float(old_splice["delta_nat_per_offdiag"])
                row = {
                    "seed": seed, "state_id": state_id, "chromosome": chromosome,
                    "splice_index": splice_index, "development_nll_nat_per_offdiag": no_k0,
                    "development_delta_splice_minus_original": delta,
                    "train_delta_splice_minus_original": train_delta,
                    "Noff": int(score["Noff"]), "eligible_pairs": int(dev.n_pairs),
                    "Zall": float(score["Zall"]),
                    "observed_log_rate_offdiag": float(score["observed_log_rate_offdiag"]),
                    "p": float(entry["p"]), "coordinate_sha256": coord_hash,
                    "artifact_sha256": file_hash, "host_wall_seconds": host_seconds,
                    "gpu_event_seconds": gpu_seconds,
                }
                rows.append(row)
                call_ledger.append({"formal_call": len(call_ledger) + 1, "seed": seed,
                                    "state_id": state_id, "contact_fullgrid_forward": 1,
                                    "training_fg": 0, "regularizer_fullgrid": 0,
                                    "host_wall_seconds": host_seconds, "gpu_event_seconds": gpu_seconds})

        if len(call_ledger) != config["formal_fullgrid_calls"]:
            raise RuntimeError("formal full-grid call count is not exactly 18")
        gates = []
        for seed in config["seeds"]:
            deltas = np.asarray([row["development_delta_splice_minus_original"] for row in rows
                                 if row["seed"] == seed and row["state_id"] != "original"], dtype=np.float64)
            positive = int(np.count_nonzero(deltas > 0.0))
            median = float(np.median(deltas))
            passed = (positive >= config["gate"]["per_seed_min_positive_of_8"]
                      and median >= config["gate"]["per_seed_min_median_delta_nat_per_contact"])
            gates.append({"seed": seed, "positive_count": positive, "n_splices": 8,
                          "median_delta_nat_per_offdiag": median,
                          "positive_count_gate": positive >= config["gate"]["per_seed_min_positive_of_8"],
                          "median_gate": median >= config["gate"]["per_seed_min_median_delta_nat_per_contact"],
                          "passed": passed})
        overall_pass = all(item["passed"] for item in gates)

        results = {
            "schema": "p9016-copy-link-dev-validation-a-results-v1",
            "status": "complete",
            "experiment_A_gate": "PASS" if overall_pass else "FAIL",
            "classification": ("development-validation copy-link support meets frozen threshold"
                               if overall_pass else "support insufficient/uncertain under frozen threshold"),
            "rows": rows, "per_seed_gates": gates,
            "metric": config["primary_metric"],
            "development_validation_not_fresh_test": True,
            "record_level_not_molecule_isolation": True,
            "seeds_are_initialization_repeats_not_biological_replicates": True,
            "reference_opened": False, "phase_opened": False,
            "experiment_B": "not_started",
        }
        write_json(RUN / "results.json", results)
        with (RUN / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
            fields = ["seed", "state_id", "chromosome", "splice_index",
                      "development_nll_nat_per_offdiag", "development_delta_splice_minus_original",
                      "train_delta_splice_minus_original", "Noff", "eligible_pairs", "p",
                      "host_wall_seconds", "gpu_event_seconds", "coordinate_sha256", "artifact_sha256"]
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        ledger = {
            "schema": "p9016-copy-link-dev-validation-a-ledger-v1",
            "formal_contact_fullgrid_forwards": len(call_ledger),
            "expected_formal_contact_fullgrid_forwards": 18,
            "training_fg_calls": 0, "regularizer_fullgrid_sweeps": 0,
            "random_u_calls": 0, "cpu_preflight_fullgrid_calls": 0,
            "host_wall_seconds_sum": total_host,
            "gpu_event_seconds_sum": total_gpu,
            "timing_note": "host wall and CUDA event elapsed are separately measured; neither is inferred",
            "calls": call_ledger,
        }
        write_json(RUN / "ledger.json", ledger)
        readme_lines = [
            "# 057 copy 连接 development-validation 实验 A",
            "",
            "本 run 只补做实验 A；实验 B 未启动。未修改 056、046、冻结源码、协议、mask 或 checkpoint，也未调用 051 `fix_export`。",
            "",
            "## 固定口径",
            "",
            "- P9016 单细胞，20 条染色体、40 条 copy；两个 seed 仅是初始化重复，不是生物学重复。",
            "- 使用 056 固定 record fold。20% 数据此前已经看过，故称 development validation，不称全新 test；readID 全为 `.`，这是 record-level split，不保证 molecule isolation。",
            "- 固定 G-original endpoint 的 `x/p` 与 80% train exposure；只把 counts 换成 development-validation contacts，不重算 exposure。",
            "- 主指标为 `log(Z_full) - sum(Cdev * log(rate)) / 253067`；无 K0、无正则，`Z_full` 覆盖 3,496,690 个 eligible pairs（含零计数）。",
            "- 共 18 次且仅 18 次正式 contact full-grid forward：每 seed 1 个 original + 8 个固定 splice；训练 FG=0，random-u=0。",
            "",
            "## A 门结果",
            "",
        ]
        for gate in gates:
            readme_lines.append(f"- seed {gate['seed']}: `{gate['positive_count']}/8` 个 delta > 0，median=`{gate['median_delta_nat_per_offdiag']:.12g}` nat/offdiag contact，门={'PASS' if gate['passed'] else 'FAIL'}。")
        readme_lines += [
            "",
            f"**实验 A 总门：{'PASS' if overall_pass else 'FAIL'}。** " +
            ("该结果只支持这些固定 copy 连接破坏在 development validation 上被计数目标排斥；不证明 L2。" if overall_pass
             else "按预冻结规则归类为支持不足/不确定，不作 L3；本轮停止且不启动 B。"),
            "",
            "## 终态与后续",
            "",
            "- 科学终态记录在 `terminal.json`；exit code 0 不是判门依据。",
            "- 明细见 `summary.tsv`、`results.json`；18 次调用和实测 host/GPU 时间见 `ledger.json`；输入与状态核验见 `preflight.json`。",
            "- 实验 B 状态为 `not_started`。A 完成后暂停，等待父代理确认数学缓存方案。",
            "- 046 baseline 保持不变；L2 仍未证明，不作 L3。",
        ]
        (RUN / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
        terminal.update({
            "status": "complete",
            "completed_at_utc": utc_now(),
            "host_wall_seconds": time.perf_counter() - started,
            "experiment_A": "PASS" if overall_pass else "FAIL",
            "experiment_A_scientific_status": ("threshold_passed" if overall_pass else "support_insufficient_or_uncertain"),
            "formal_contact_fullgrid_forwards": len(call_ledger),
            "training_fg_calls": 0,
            "experiment_B": "not_started",
            "reference_opened": False,
            "phase_opened": False,
            "l2_supported": False,
            "l3_claimed": False,
        })
        write_json(terminal_path, terminal)
        print(json.dumps({"status": terminal["status"], "experiment_A": terminal["experiment_A"],
                          "formal_fullgrid_forwards": len(call_ledger), "gates": gates}, sort_keys=True))
        return 0
    except Exception as error:
        terminal.update({"status": "failure", "completed_at_utc": utc_now(),
                         "host_wall_seconds": time.perf_counter() - started,
                         "experiment_A": "not_completed", "experiment_B": "not_started",
                         "error_type": type(error).__name__, "error": str(error),
                         "reference_opened": False, "phase_opened": False})
        write_json(terminal_path, terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
