#!/usr/bin/env python3
"""针对 022 FDG proposal 边界的微型、无 reference 检查。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fdg_proposal as fp  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    config = json.loads((HERE / "config.json").read_text())
    require(config["status"] in ("preflight_only", "released_for_real_fdg",
                                  "completed_accepted", "completed_rejected_to_anchor"),
            "config status is outside the preflight/released/completed states")
    require(config["anchor"]["use_020_as_anchor"] is False, "020 cannot be an anchor")

    lengths = [2_300_000, 1_700_000]
    grid = fp.make_full_grid(lengths)
    require(grid.n_loci == 5 and grid.n_beads == 10 and grid.n_tracks == 4,
            "tiny explicit grid inventory mismatch")
    require(np.all(grid.beads[:, 1] % grid.bin_size == 0), "bead starts are not 0-based")
    require(int(grid.beads[2, 2]) == 2_300_000, "terminal partial bin was not retained")

    coords = np.asarray([
        [[-0.30, 0.00, 0.00], [-0.20, 0.02, 0.01], [-0.10, 0.00, 0.02],
         [0.10, 0.01, 0.00], [0.20, 0.00, 0.02]],
        [[-0.29, 0.04, 0.01], [-0.18, 0.06, 0.00], [-0.08, 0.04, 0.03],
         [0.11, 0.05, 0.01], [0.22, 0.04, 0.03]],
    ], dtype=np.float64)
    records = fp.make_raw_records(
        ci=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        p1=[100, 100, 1_100_000, 100, 100, 100, 1_100_000, 2_100_000, 100, 100],
        cj=[0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        p2=[200, 1_100_000, 2_100_000, 200, 1_100_000, 100, 1_100_000, 100, 1_100_000, 1_600_000],
        n_chromosomes=2,
    )
    fp.validate_raw_records(records, grid)
    canonical, posterior, allocation, graph, blob = fp.build_integer_proposal(
        coords, q=0.3, records=records, grid=grid,
    )
    require(records.n_records == 10, "tiny raw record count mismatch")
    require(int(posterior.same_bin.sum()) == 2, "tiny diagonal count mismatch")
    require(graph.n_raw_edges == 8 and graph.total_count == 8,
            "diag rows leaked or off-diagonal rows were lost")
    require(np.all(graph.unique_edges[:, 0] < graph.unique_edges[:, 1]),
            "force graph contains a self/reversed edge")
    require(int(np.sum(allocation.state >= 0)) == 8 and int(np.sum(allocation.state < 0)) == 2,
            "integer allocation did not preserve the diag mask")
    state_table = fp.state_count_table(posterior, allocation)
    require(int(state_table[:, 2].sum()) == 8 and
            np.array_equal(state_table[:, 2], state_table[:, 3:].sum(axis=1)),
            "state counts did not conserve each non-diagonal bin pair")
    require(fp.sha256_bytes(blob) == fp.sha256_bytes(
        fp.pack_bridge_input(grid, graph, canonical.coordinates)),
            "bridge blob is not deterministic")

    # 精确的 tie 逐字节相同，因此不能改变结果。
    tie_coords = coords.copy()
    tie_coords[:, grid.chromosome_slice(0)] = tie_coords[0, grid.chromosome_slice(0)]
    tie_canonical = fp.canonicalize_coordinates(tie_coords, grid)
    tie_swapped = fp.canonicalize_coordinates(
        fp.swap_copy_subset(tie_coords, grid, chromosomes=[0]), grid,
    )
    require(not bool(tie_canonical.swapped_by_chromosome[0]), "exact tie was not explicit")
    require(np.array_equal(tie_canonical.coordinates, tie_swapped.coordinates),
            "exact tie changed after a copy swap")

    invariance = {"all": fp.swap_invariance_hashes(coords, 0.3, records, grid)}
    for chromosome in range(grid.n_chromosomes):
        invariance[f"chr{chromosome + 1}"] = fp.swap_invariance_hashes(
            coords, 0.3, records, grid, chromosomes=[chromosome],
        )
    require(len(set(item["bridge_source_blob_sha256"] for item in invariance.values())) == 1,
            "copy-swap source blobs are not identical")

    bridge = HERE / "build" / "fdg_bridge"
    require(bridge.exists(), f"bridge binary missing: {bridge}")
    with tempfile.TemporaryDirectory(prefix="fdg-preflight-") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "input.bin"
        output_paths = (temporary_path / "output-1.bin", temporary_path / "output-2.bin")
        outputs = []
        attempts = []
        for output_path in output_paths:
            attempt = fp.run_native_bridge(
                bridge, input_path, output_path, blob, iterations=2,
                seed=fp.NATIVE_SEED,
            )
            require(attempt.returncode == 0,
                    f"native bridge failed ({attempt.returncode}): {attempt.stderr}")
            attempts.append(attempt)
            outputs.append(fp.read_bridge_output(output_path, grid))
        require(all(attempt.input_sha256 == attempts[0].input_sha256 for attempt in attempts),
                "bridge run records disagree on input hash")
        require(output_paths[0].read_bytes() == output_paths[1].read_bytes(),
                "same-seed native bridge output is not byte-identical")
        output = outputs[0]
        source_beads = fp.coordinates_to_bead_order(canonical.coordinates, grid)
        init_delta = np.max(np.abs(output.native_init_bead_order.astype(np.float64)
                                   - source_beads.astype(np.float64)))
        require(output.n_beads == grid.n_beads and output.n_raw_pairs == graph.n_raw_edges,
                "native bridge changed bead/raw-edge inventory")
        require(output.n_binned_pairs <= graph.n_raw_edges and output.n_iter == 2,
                "native bridge output budget or aggregation is invalid")
        require(output.source_avg_bb > 0.0 and output.native_unit > 0.0,
                "native source scale is invalid")
        require(output.max_abs_jitter > 0.0 and np.isclose(output.max_abs_jitter, init_delta),
                "native source jitter was not captured exactly")
        raw_final = fp.bead_order_to_coordinates(output.native_final_bead_order, grid)
        mapped, center, scale, max_radius = fp.map_native_to_unit_ball(raw_final, grid)
        require(np.all(np.linalg.norm(mapped.reshape((-1, 3)), axis=1) < 1.0),
                "global V1 map is not strict interior")
        inverse = fp.sphere_inverse(mapped)
        roundtrip = inverse / np.sqrt(1.0 + np.sum(inverse * inverse, axis=-1, keepdims=True))
        require(np.allclose(roundtrip, mapped, rtol=0.0, atol=2e-15),
                "sphere inverse roundtrip failed")
        trials = fp.backtracking_coordinates(mapped, mapped, grid)
        require(tuple(alpha for alpha, _ in trials) == fp.ALPHA_SEQUENCE,
                "alpha sequence is not frozen")

    print(json.dumps({
        "status": "passed",
        "grid": {"n_loci": grid.n_loci, "n_beads": grid.n_beads,
                 "n_tracks": grid.n_tracks},
        "records": {"all": records.n_records, "same_bin": int(posterior.same_bin.sum()),
                    "force_raw": graph.n_raw_edges, "force_unique": graph.n_unique_edges},
        "p": posterior.p,
        "r0": posterior.r0,
        "canonical_swap_bits": canonical.swapped_by_chromosome.astype(int).tolist(),
        "swap_invariance": invariance,
        "bridge_source_blob_sha256": hashlib.sha256(blob).hexdigest(),
        "bridge_checked": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
