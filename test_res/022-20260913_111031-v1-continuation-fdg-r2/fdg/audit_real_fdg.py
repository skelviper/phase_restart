#!/usr/bin/env python3
"""审计已发布的 full-grid copy-swap 不变性和原生 bead 映射。"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fdg_proposal as fp  # noqa: E402


def fail(message: str) -> None:
    raise RuntimeError(message)


def hash_array(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def main() -> None:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    if config["status"] != "released_for_real_fdg":
        fail("canonical audit must run before final status transition")
    build_dir = HERE / "build"
    input_path = build_dir / "bridge_input.bin"
    output_path = build_dir / "bridge_output.bin"
    prepared_path = build_dir / "prepared_state.npz"
    ledger_path = build_dir / "allocation_ledger.npz"
    if not all(path.exists() for path in (input_path, output_path, prepared_path, ledger_path)):
        fail("prepared/input/output/ledger artifact is missing")
    with np.load(prepared_path, allow_pickle=False) as prepared:
        anchor = np.asarray(prepared["anchor_coordinates"], dtype=np.float64).copy()
        canonical_anchor = np.asarray(prepared["canonical_coordinates"], dtype=np.float64).copy()
        grid_lengths = np.asarray(prepared["grid_lengths"], dtype=np.int64).copy()
    grid = fp.make_full_grid([int(value) for value in grid_lengths])
    with np.load(ledger_path, allow_pickle=False) as ledger:
        records = fp.RawRecords(
            ci=np.asarray(ledger["ci"], dtype=np.int32).copy(),
            p1=np.asarray(ledger["p1"], dtype=np.int64).copy(),
            cj=np.asarray(ledger["cj"], dtype=np.int32).copy(),
            p2=np.asarray(ledger["p2"], dtype=np.int64).copy(),
            record_id=np.asarray(ledger["record_id"], dtype=np.int64).copy(),
        )
        g1 = np.asarray(ledger["g1"], dtype=np.int64).copy()
        g2 = np.asarray(ledger["g2"], dtype=np.int64).copy()
        same_bin = np.asarray(ledger["same_bin"], dtype=bool).copy()
        state = np.asarray(ledger["state"], dtype=np.int8).copy()
    posterior = fp.PosteriorResult(
        g1=g1, g2=g2, same_bin=same_bin,
        probabilities=np.zeros((records.n_records, 4), dtype=np.float64),
        p=0.0, r0=0.0,
    )
    allocation = fp.AllocationResult(
        state=state, probabilities=posterior.probabilities, seed=fp.ALLOC_SEED,
    )
    graph = fp.force_graph_from_allocation(records, posterior, allocation, grid)
    if graph.n_raw_edges != 1_265_114 or graph.total_count != 1_265_114:
        fail("ledger reconstruction did not conserve all non-diagonal records")
    blob = input_path.read_bytes()
    if fp.sha256_bytes(blob) != json.loads(
            (HERE / "proposal_manifest.json").read_text(encoding="utf-8")
    )["artifacts"]["input_blob_sha256"]:
        fail("prepared source blob SHA changed")
    header_struct = struct.Struct("<8s6I")
    if len(blob) < header_struct.size:
        fail("bridge input is truncated")
    magic, version, bin_size, n_tracks, n_beads, n_raw, flags = header_struct.unpack_from(blob)
    if (magic, version, bin_size, n_tracks, n_beads, n_raw, flags) != (
            fp.INPUT_MAGIC, 1, grid.bin_size, grid.n_tracks, grid.n_beads,
            graph.n_raw_edges, 0):
        fail("bridge input header does not match prepared grid/graph")
    bead_start = header_struct.size + 4 * grid.n_tracks
    bead_end = bead_start + 12 * grid.n_beads
    edge_start = bead_end
    edge_end = edge_start + 8 * graph.n_raw_edges
    source_start = edge_end
    expected_beads = np.frombuffer(blob, dtype="<i4", count=grid.n_beads * 3,
                                   offset=bead_start).reshape((grid.n_beads, 3))
    expected_edges = np.frombuffer(blob, dtype="<u4", count=graph.n_raw_edges * 2,
                                   offset=edge_start).reshape((graph.n_raw_edges, 2))
    expected_source = np.frombuffer(blob, dtype="<f4", count=grid.n_beads * 3,
                                    offset=source_start).reshape((grid.n_beads, 3))
    if not np.array_equal(expected_beads, grid.beads):
        fail("native input bead inventory is not one-to-one with explicit grid")
    if not np.array_equal(expected_edges, graph.raw_edges):
        fail("native input raw edge order differs from ledger reconstruction")
    expected_source_from_v1 = fp.coordinates_to_bead_order(canonical_anchor, grid)
    if not np.array_equal(expected_source, expected_source_from_v1):
        fail("native input source is not one-to-one with canonical V1 coordinates")
    output = fp.read_bridge_output(output_path, grid)
    final_v1 = fp.bead_order_to_coordinates(output.native_final_bead_order, grid)
    output_roundtrip = fp.coordinates_to_bead_order(final_v1, grid)
    if not np.array_equal(output_roundtrip, output.native_final_bead_order):
        fail("native output bead/V1 mapping is not one-to-one")
    if not np.all(np.isfinite(final_v1)):
        fail("native output contains non-finite mapped bead coordinates")

    base_blob_sha = fp.sha256_bytes(blob)
    base_canonical_sha = hash_array(canonical_anchor, "<f8")
    base_edge_sha = hash_array(
        np.column_stack((graph.unique_edges, graph.unique_counts)), "<i8",
    )
    swap_cases: list[tuple[str, list[int] | None]] = [("all", None)]
    swap_cases.extend((f"chr{chromosome + 1:02d}", [chromosome])
                       for chromosome in range(grid.n_chromosomes))
    swap_results = {}
    for label, chromosomes in swap_cases:
        swapped_input = fp.swap_copy_subset(anchor, grid, chromosomes)
        canonical = fp.canonicalize_coordinates(swapped_input, grid)
        canonical_sha = hash_array(canonical.coordinates, "<f8")
        if not np.array_equal(canonical.coordinates, canonical_anchor):
            fail(f"canonical coordinate bytes changed for swap case {label}")
        swap_blob = fp.pack_bridge_input(grid, graph, canonical.coordinates)
        swap_blob_sha = fp.sha256_bytes(swap_blob)
        if swap_blob_sha != base_blob_sha:
            fail(f"bridge source blob changed for swap case {label}")
        swap_results[label] = {
            "input_coordinates_sha256": hash_array(swapped_input, "<f8"),
            "canonical_coordinates_sha256": canonical_sha,
            "canonical_byte_equal": True,
            "edge_table_sha256": base_edge_sha,
            "edge_table_byte_equal": True,
            "bridge_source_blob_sha256": swap_blob_sha,
            "bridge_source_blob_byte_equal": True,
            "swap_bits_on_swapped_input": canonical.swapped_by_chromosome.astype(int).tolist(),
        }

    audit_path = HERE / "canonical_swap_mapping_audit.json"
    if audit_path.exists():
        fail(f"refusing to overwrite audit artifact: {audit_path}")
    audit = {
        "schema": "fdg-canonical-swap-mapping-audit-v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "reader_precision": "float64_preserving_for_anchor_and_v1; float32_only_at_native_bridge_boundary",
        "canonical_base_coordinates_sha256": base_canonical_sha,
        "edge_table_sha256": base_edge_sha,
        "bridge_input_sha256": base_blob_sha,
        "grid": {"n_loci": grid.n_loci, "n_beads": grid.n_beads,
                 "n_tracks": grid.n_tracks, "n_chromosomes": grid.n_chromosomes},
        "ledger": {"raw_records": records.n_records, "same_bin": int(same_bin.sum()),
                   "force_records": graph.n_raw_edges, "unique_edges": graph.n_unique_edges},
        "native_input_mapping": {
            "explicit_bead_inventory_byte_equal": True,
            "raw_edge_inventory_byte_equal": True,
            "canonical_source_float32_byte_equal": True,
            "source_start_offset": source_start,
        },
        "native_output_mapping": {
            "output_sha256": fp.sha256_file(output_path),
            "n_beads": output.n_beads,
            "n_raw_pairs": output.n_raw_pairs,
            "native_bead_to_v1_to_bead_byte_equal": True,
            "finite": True,
        },
        "swap_cases": swap_results,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "audit": str(audit_path),
        "all_and_single_chromosome_cases": len(swap_cases),
        "canonical_coordinates_sha256": base_canonical_sha,
        "edge_table_sha256": base_edge_sha,
        "bridge_input_sha256": base_blob_sha,
        "native_output_sha256": audit["native_output_mapping"]["output_sha256"],
        "native_input_output_bead_mapping": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
