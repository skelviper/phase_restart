"""Bounded driver smoke: selection, hashed resume, export and collision refusal."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import numpy as np

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "source"))
import run_pipeline as driver  # noqa: E402
from run_pipeline import (  # noqa: E402
    BackendError,
    BackendAdapter,
    array_sha256,
    choose_selection,
    fit_l_bfgs,
    full_positions,
    load_state,
    save_state,
    write_coordinates_with_offsets,
)
from pr.contact_model import aggregate_from_arrays  # noqa: E402

class _FakeData:
    def __init__(self, bin_size):
        self.bin_size = bin_size
        self.chromosome_names = ("chr1", "chr2")
        self.chromosome_lengths = np.asarray([5, 4], dtype=np.int64)
        self.n_loci = len(full_positions(self.chromosome_lengths, bin_size)[0])

    def budget(self):
        return {
            "n_loci": self.n_loci,
            "n_eligible_pairs": self.n_loci * (self.n_loci - 1) // 2,
            "raw_records": 1,
            "raw_same_bin": 0,
            "raw_cis_offdiag": 1,
            "raw_inter": 0,
            "aggregate_same_bin": 0,
            "aggregate_offdiag": 1,
            "endpoint_total": 2,
            "raw_conserved": True,
            "aggregate_conserved": True,
            "endpoint_conserved": True,
        }


class _FakeObjective:
    def __init__(self, data):
        self.data = data

    def _components(self):
        return {"count_nll_normalized": 1.0, "total": 1.0}

    def evaluate(self, theta, need_gradient=False):
        gradient = np.zeros_like(theta) if need_gradient else None
        return 1.0, gradient, self._components()

    def value_and_grad(self, theta):
        return 1.0, np.zeros_like(theta)

    def components(self, theta):
        return self._components()

    def coordinates_and_p(self, theta):
        return np.zeros((2, self.data.n_loci, 3), dtype=np.float64), 0.75


class _FakeAdapter:
    kind = "archived_cpu"
    device = "cpu"
    label = "smoke"

    def load_data(self, input_path, bin_size):
        return _FakeData(bin_size)

    def objective(self, data):
        return _FakeObjective(data)

    def pack_theta(self, objective, data, coordinates, p):
        return np.zeros(6 * data.n_loci + 1, dtype=np.float64)

    def synchronize(self, objective):
        return None

    def setup_timing_name(self):
        return "archived_constructor_seconds"
OUT = RUN / "logs" / "driver_smoke.json"


def main() -> None:
    checks = {}
    names = ("chr1", "chr2")
    lengths = np.asarray([7, 5], dtype=np.int64)
    ci = np.asarray([0, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    p1 = np.asarray([0, 2, 4, 0, 1, 1, 3, 6, 4], dtype=np.int64)
    cj = np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 1], dtype=np.int64)
    p2 = np.asarray([0, 2, 6, 0, 1, 3, 3, 0, 4], dtype=np.int64)
    archived_data = aggregate_from_arrays(names, lengths, ci, p1, cj, p2, 2)
    archived_adapter = BackendAdapter("archived_cpu", "cpu", 3)
    archived_objective = archived_adapter.objective(archived_data)
    archived_theta = archived_objective.pack(np.zeros((2, archived_data.n_loci, 3), dtype=np.float64), p=0.67)
    archived_fit = fit_l_bfgs(archived_objective, archived_theta, maxiter=1, maxfun=4, checkpoint_every=None)
    checks["archived_cpu_adapter"] = bool(
        archived_adapter.label == "archived-cpu"
        and "count_nll_normalized" in archived_objective.components(archived_theta)
        and archived_fit["termination_class"] in {"converged", "budget_not_converged", "line_search_abort", "optimizer_failed"}
    )
    with tempfile.TemporaryDirectory(prefix="020-driver-smoke-") as temp:
        root = Path(temp)
        backend = root / "backend_runs" / "archived-cpu"
        lengths = np.asarray([5, 4], dtype=np.int64)
        positions, chromosomes, _ = full_positions(lengths, 2)
        coords = np.zeros((2, len(positions), 3), dtype=np.float64)
        coords[0, :, 0] = 0.1
        coords[1, :, 1] = -0.1
        theta = np.arange(6 * len(positions) + 1, dtype=np.float64)
        theta[-1] = 0.75
        data = SimpleNamespace(
            n_loci=len(positions), chromosome_names=("chr1", "chr2"),
            chromosome_lengths=lengths,
        )
        exported = write_coordinates_with_offsets(root / "coords" / "tiny.3dg", coords, positions, lengths, 2)
        checks["export_full_grid"] = exported["full_grid"] and exported["n_tracks"] == 4
        ref = save_state(
            backend, "random_joint", "1m", 2, data, coords, positions, chromosomes,
            theta, "config-hash", "input-hash",
        )
        loaded = load_state(backend, ref, "random_joint", "1m", 2, data, "config-hash", "input-hash")
        checks["resume_hash_roundtrip"] = bool(
            np.array_equal(loaded["coordinates"], coords)
            and np.array_equal(loaded["positions"], positions)
            and np.array_equal(loaded["chromosomes"], chromosomes)
            and np.array_equal(loaded["theta"], theta)
            and loaded["carried_q"] == theta[-1]
        )
        selection = choose_selection(
            [
                {"candidate_id": "consensus_joint", "final_count_nll_normalized": 3.0,
                 "state_ref": {"candidate_id": "consensus_joint"}},
                {"candidate_id": "random_joint", "final_count_nll_normalized": 2.0,
                 "state_ref": ref},
            ],
            "config-hash", "input-hash",
        )
        checks["selection_after_both"] = selection["selected_candidate"] == "random_joint"
        existing = backend / "stages" / "random_joint" / "1m.json"
        existing.parent.mkdir(parents=True)
        existing.write_text("existing\n", encoding="utf-8")
        try:
            from run_pipeline import assert_absent
            assert_absent(existing, "stage summary random_joint/1m")
        except BackendError:
            checks["collision_refusal"] = True
        else:
            checks["collision_refusal"] = False
        checks["state_hash_present"] = bool(ref["theta_sha256"] == array_sha256(theta))
        original_stages = driver.STAGES
        original_parse = driver.parse_3dg
        original_expand = driver.expand_tracks
        try:
            driver.STAGES = (("5m", 2, 1, 4), ("2m", 1, 1, 4))
            driver.parse_3dg = lambda path, header_lengths: {}
            driver.expand_tracks = lambda tracks, header_lengths, bin_size: (
                np.zeros((2, len(full_positions(header_lengths, bin_size)[0]), 3), dtype=np.float64),
                full_positions(header_lengths, bin_size)[0],
                full_positions(header_lengths, bin_size)[1],
            )
            smoke_backend = root / "backend_runs" / "smoke"
            result = driver.run_candidate(
                smoke_backend, root / "input.gz", "input-hash", "config-hash", _FakeAdapter(),
                "random_joint", "random", 2207, 0, 1,
            )
            stage_5m = driver.read_stage(smoke_backend, "random_joint", "5m")
            stage_2m = driver.read_stage(smoke_backend, "random_joint", "2m")
            checks["multistage_transition"] = (
                result["stages"][-1]["stage"] == "2m"
                and stage_2m["initialization"]["source_stage"] == "5m"
                and stage_2m["resume_state"]["bin_size_bp"] == 1
                and stage_2m["accepted_history"][0]["gradient_l2"] == 0.0
                and stage_5m["status"] in {"converged", "budget_not_converged", "line_search_abort", "optimizer_failed"}
            )
        finally:
            driver.STAGES = original_stages
            driver.parse_3dg = original_parse
            driver.expand_tracks = original_expand
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "phase_or_reference_opened": False,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
