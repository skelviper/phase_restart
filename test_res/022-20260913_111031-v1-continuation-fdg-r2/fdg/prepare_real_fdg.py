#!/usr/bin/env python3
"""为唯一一次 native FDG 调用准备已发布的 022 anchor。"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fdg_proposal as fp  # noqa: E402
from pr import genome  # noqa: E402
from pr.contact_model import aggregate_from_arrays, verify_frozen_snpfree  # noqa: E402


EXPECTED_COORD_SHA = "49501f5b38d699fb2c9c8849616edd02b70fccd5ae39e1efe6036e17f5f620e7"
EXPECTED_THETA_SHA = "588bafce1d7652fa8cf17cb060d23807dd6098c3e072fd9fc52784a145860a57"
EXPECTED_INPUT_SHA = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_anchor_3dg(path: Path, grid: fp.FullGrid) -> np.ndarray:
    rows = []
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 5:
                fail(f"anchor 3dg line {line_number} does not have five fields")
            name, position, x, y, z = fields
            rows.append((name, int(position), float(x), float(y), float(z)))
    if len(rows) != grid.n_beads:
        fail(f"anchor 3dg has {len(rows)} rows, expected {grid.n_beads}")
    coordinates_bead = np.empty((grid.n_beads, 3), dtype=np.float64)
    for bead, (name, position, x, y, z) in enumerate(rows):
        expected_name = grid.track_names[next(
            track for track in range(grid.n_tracks)
            if grid.track_slice(track).start <= bead < grid.track_slice(track).stop
        )]
        expected_position = int(grid.beads[bead, 1])
        if name != expected_name or position != expected_position:
            fail(f"anchor 3dg row {bead} changes explicit track/order")
        coordinates_bead[bead] = (x, y, z)
    return fp.bead_order_to_coordinates(coordinates_bead, grid)


def main() -> None:
    config_path = HERE / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "released_for_real_fdg":
        fail("config is not released for the one real FDG proposal")
    anchor = config["anchor"]
    theta_path = Path(anchor["theta_path"])
    coordinates_path = Path(anchor["coordinates_path"])
    if fp.sha256_file(theta_path) != EXPECTED_THETA_SHA or \
            fp.sha256_file(coordinates_path) != EXPECTED_COORD_SHA:
        fail("released anchor SHA does not match parent handoff")
    with np.load(theta_path, allow_pickle=False) as archive:
        theta = np.asarray(archive["theta"], dtype=np.float64).copy()
        coordinates = np.asarray(archive["coordinates"], dtype=np.float64).copy()
    if theta.shape != tuple(anchor["theta_shape"]):
        fail("released theta shape does not match config")
    if coordinates.shape != tuple(anchor["coordinates_shape"]):
        fail("released coordinates shape does not match config")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(coordinates)):
        fail("released anchor contains non-finite values")
    if not np.isclose(theta[-1], anchor["q_anchor"], rtol=0.0, atol=1e-15):
        fail("theta[-1] differs from released q")
    if not np.isclose(fp.p_from_q(float(theta[-1])), anchor["p_anchor"], rtol=0.0, atol=1e-15):
        fail("released p does not match bounded q map")

    input_path = PROJECT_ROOT / config["input"]["path"]
    lengths = genome.chrom_lengths(str(input_path))
    grid = fp.make_full_grid([length for _, length in lengths])
    expected_names = tuple(
        [f"chr{chromosome + 1}" for chromosome in range(19)] + ["chrX"]
    )
    if tuple(name for name, _ in lengths) != expected_names:
        fail("input chromosome names are not the frozen numeric order")
    anchor_3dg_coordinates = read_anchor_3dg(coordinates_path, grid)
    coordinate_file_max_abs_delta = float(np.max(np.abs(anchor_3dg_coordinates - coordinates)))
    if not np.isfinite(coordinate_file_max_abs_delta) or coordinate_file_max_abs_delta > 1e-6:
        fail("released coordinate 3dg differs materially from NPZ coordinates")

    input_path = PROJECT_ROOT / config["input"]["path"]
    input_audit = verify_frozen_snpfree(input_path)
    if input_audit["snpfree_sha256"] != EXPECTED_INPUT_SHA:
        fail("SNP-free input SHA differs from frozen config")
    contacts = genome.load_all(str(input_path))
    data = aggregate_from_arrays(
        [name for name, _ in lengths], [length for _, length in lengths],
        contacts["ci"], contacts["p1"], contacts["cj"], contacts["p2"], fp.BIN_SIZE,
    )
    if data.raw_records != 1_703_888 or data.raw_same_bin != 438_774 or \
            data.raw_cis_offdiag != 696_680 or data.raw_inter != 568_434:
        fail("full original aggregate budget differs from frozen audit")
    records = fp.make_raw_records(
        contacts["ci"], contacts["p1"], contacts["cj"], contacts["p2"],
        n_chromosomes=grid.n_chromosomes,
    )
    if records.n_records != data.raw_records:
        fail("raw record loader changed the full ledger cardinality")
    canonical, posterior, allocation, graph, blob = fp.build_integer_proposal(
        coordinates, float(theta[-1]), records, grid,
    )
    if canonical.coordinates.shape != (2, grid.n_loci, 3):
        fail("canonical anchor shape changed")
    if int(posterior.same_bin.sum()) != data.raw_same_bin:
        fail("posterior diagonal ledger count differs from aggregate")
    if graph.n_raw_edges != 1_265_114 or graph.total_count != 1_265_114:
        fail("force graph does not contain every non-diagonal raw record")
    diag_counts = np.bincount(
        posterior.g1[posterior.same_bin], minlength=grid.n_loci,
    ).astype(np.int64, copy=False)
    if not np.array_equal(diag_counts, data.diag_counts):
        fail("diag ledger is not identical to original aggregate")
    non_diag = ~posterior.same_bin
    lo = np.minimum(posterior.g1[non_diag], posterior.g2[non_diag])
    hi = np.maximum(posterior.g1[non_diag], posterior.g2[non_diag])
    flat = lo * (2 * grid.n_loci - lo - 1) // 2 + (hi - lo - 1)
    full_counts = np.zeros(data.n_pairs, dtype=np.int64)
    np.add.at(full_counts, flat, 1)
    if not np.array_equal(full_counts, data.counts):
        fail("non-diagonal raw ledger is not identical to full aggregate counts")
    state_table = fp.state_count_table(posterior, allocation)
    if int(state_table[:, 2].sum()) != 1_265_114:
        fail("integer state table does not conserve every non-diagonal record")

    build_dir = HERE / "build"
    input_blob_path = build_dir / "bridge_input.bin"
    output_path = build_dir / "bridge_output.bin"
    if output_path.exists():
        fail(f"refusing to overwrite existing native output: {output_path}")
    input_blob_path.write_bytes(blob)
    ledger_path = build_dir / "allocation_ledger.npz"
    np.savez_compressed(
        ledger_path,
        record_id=records.record_id,
        ci=records.ci,
        p1=records.p1,
        cj=records.cj,
        p2=records.p2,
        g1=posterior.g1,
        g2=posterior.g2,
        same_bin=posterior.same_bin,
        state=allocation.state,
    )
    prepared_path = build_dir / "prepared_state.npz"
    np.savez_compressed(
        prepared_path,
        anchor_coordinates=coordinates,
        canonical_coordinates=canonical.coordinates,
        swapped_by_chromosome=canonical.swapped_by_chromosome,
        anchor_theta=theta,
        p=np.asarray([posterior.p]),
        r0=np.asarray([posterior.r0]),
        grid_lengths=grid.chromosome_lengths,
    )
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest_path = HERE / "proposal_manifest.json"
    manifest = {
        "schema": "fdg-proposal-manifest-v1",
        "status": "prepared_native_pending",
        "prepared_utc": now,
        "anchor": {
            "theta_path": str(theta_path),
            "theta_sha256": fp.sha256_file(theta_path),
            "coordinates_path": str(coordinates_path),
            "coordinates_sha256": fp.sha256_file(coordinates_path),
            "q": float(theta[-1]),
            "p": float(posterior.p),
            "coordinates_shape": list(coordinates.shape),
            "theta_shape": list(theta.shape),
            "coordinate_3dg_max_abs_delta_from_npz": coordinate_file_max_abs_delta,
            "coordinate_input_source": "theta_npz.coordinates_float64",
            "canonical_swap_bits": canonical.swapped_by_chromosome.astype(int).tolist(),
        },
        "input": {
            "path": str(input_path),
            "sha256": input_audit["snpfree_sha256"],
            "raw_records": records.n_records,
            "same_bin": int(posterior.same_bin.sum()),
            "cis_offdiag": int(np.sum((records.ci == records.cj) & non_diag)),
            "inter": int(np.sum(records.ci != records.cj)),
        },
        "grid": {
            "bin_size": grid.bin_size,
            "n_chromosomes": grid.n_chromosomes,
            "n_loci": grid.n_loci,
            "n_beads": grid.n_beads,
            "n_tracks": grid.n_tracks,
            "track_order": list(grid.track_names),
        },
        "posterior": {
            "epsilon": fp.EPSILON,
            "p_floor": fp.P_FLOOR,
            "r0": posterior.r0,
            "allocation_seed": allocation.seed,
            "state_order": ["copy0-copy0", "copy0-copy1", "copy1-copy0", "copy1-copy1"],
        },
        "force_graph": {
            "raw_records": graph.n_raw_edges,
            "unique_edges": graph.n_unique_edges,
            "count_sum": graph.total_count,
            "same_bin_included": False,
            "zero_count_eligible_pairs_added": False,
            "state_table_rows": int(len(state_table)),
        },
        "artifacts": {
            "input_blob": str(input_blob_path),
            "input_blob_sha256": fp.sha256_file(input_blob_path),
            "allocation_ledger": str(ledger_path),
            "allocation_ledger_sha256": fp.sha256_file(ledger_path),
            "prepared_state": str(prepared_path),
            "prepared_state_sha256": fp.sha256_file(prepared_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "raw_records": records.n_records,
        "same_bin": int(posterior.same_bin.sum()),
        "force_records": graph.n_raw_edges,
        "n_loci": grid.n_loci,
        "n_beads": grid.n_beads,
        "n_tracks": grid.n_tracks,
        "input_blob_sha256": manifest["artifacts"]["input_blob_sha256"],
        "allocation_ledger_sha256": manifest["artifacts"]["allocation_ledger_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
