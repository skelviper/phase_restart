"""准备056实验3固定record split、train-only aggregates与contact-independent起点。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
S045 = ROOT / "test_res/045-20260915T073310Z-shared-capture-round/source"
for path in (ROOT, S045):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import data_io  # noqa: E402
from pr import contact_model, v1_calibration  # noqa: E402
from pr.solver_state import sha256_file, write_presence_mask, write_solver_state  # noqa: E402

PAIRS = ROOT / "inputs/P9016.snpfree.pairs.gz"
PAIRS_SHA = "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa"
DOMAIN = b"P9016-contact-fold-v1\0"
SEED = 560301
THRESHOLD = 14757395258967642112
EXPECTED_EXAMPLE_HEX = "50393031362d636f6e746163742d666f6c642d7631000000000000088cad0000000000000000007b01001300000000000001c802"
EXPECTED_EXAMPLE_HASH = 10704282925414366136
BIN_SIZES = (5_000_000, 2_000_000, 1_000_000)
INIT_SEEDS = (560101, 560102)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def strand_code(value: str) -> int:
    if value == "+":
        return 1
    if value == "-":
        return 2
    encoded = value.encode("ascii")
    if len(encoded) != 1 or encoded[0] > 127:
        raise ValueError("strand must be +, -, or one ASCII byte")
    return 128 + encoded[0]


def endpoint_bytes(chromosome_index: int, position: int, strand: str) -> bytes:
    return struct.pack(">HQB", int(chromosome_index), int(position), strand_code(strand))


def canonical_record_bytes(a: tuple[int, int, str], b: tuple[int, int, str]) -> bytes:
    left = endpoint_bytes(*a)
    right = endpoint_bytes(*b)
    if right < left:
        left, right = right, left
    return DOMAIN + struct.pack(">Q", SEED) + left + right


def fold_hash(a: tuple[int, int, str], b: tuple[int, int, str]) -> int:
    digest = hashlib.blake2b(canonical_record_bytes(a, b), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def encoding_selfcheck() -> dict:
    a = (0, 123, "+")
    b = (19, 456, "-")
    payload = canonical_record_bytes(a, b)
    direct = fold_hash(a, b)
    reverse = fold_hash(b, a)
    checks = {
        "domain_hex": DOMAIN.hex(),
        "domain_ends_single_nul": DOMAIN.endswith(b"\0") and not DOMAIN.endswith(b"\\0"),
        "example_hex": payload.hex(),
        "expected_example_hex": EXPECTED_EXAMPLE_HEX,
        "example_hash_uint64": direct,
        "expected_example_hash_uint64": EXPECTED_EXAMPLE_HASH,
        "reverse_hash_uint64": reverse,
        "reverse_same": direct == reverse,
        "threshold_literal": THRESHOLD,
        "exact_four_fifths_floor": (4 * (1 << 64)) // 5,
        "threshold_difference": THRESHOLD - (4 * (1 << 64)) // 5,
    }
    if not (checks["domain_ends_single_nul"] and checks["example_hex"] == EXPECTED_EXAMPLE_HEX
            and direct == EXPECTED_EXAMPLE_HASH and direct == reverse
            and checks["threshold_difference"] == 820):
        raise AssertionError("fold encoding selfcheck failed: %r" % checks)
    return checks


def load_and_split():
    if sha256_file(PAIRS) != PAIRS_SHA:
        raise RuntimeError("pairs7 SHA mismatch")
    names = []
    lengths = []
    columns = None
    records = []
    with gzip.open(PAIRS, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#chromosome:"):
                fields = line.split()
                names.append(fields[1]); lengths.append(int(fields[2]))
            elif line.startswith("#columns:"):
                columns = line.rstrip("\n").split(":", 1)[1].split("\t")
            elif line.startswith("#"):
                continue
            else:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 7 or fields[0] != ".":
                    raise RuntimeError("non-seven-column or non-dot readID record")
                records.append(fields)
    if columns != ["readID", "chr1", "pos1", "chr2", "pos2", "strand1", "strand2"]:
        raise RuntimeError("unexpected seven-column schema")
    if len(names) != 20 or len(records) != 1_703_888:
        raise RuntimeError("cohort size changed")
    by_name = {name: index for index, name in enumerate(names)}
    n = len(records)
    ci = np.empty(n, dtype=np.int16); cj = np.empty(n, dtype=np.int16)
    p1 = np.empty(n, dtype=np.int64); p2 = np.empty(n, dtype=np.int64)
    s1 = np.empty(n, dtype=np.uint8); s2 = np.empty(n, dtype=np.uint8)
    hashes = np.empty(n, dtype=np.uint64)
    for index, fields in enumerate(records):
        ci[index] = by_name[fields[1]]; p1[index] = int(fields[2])
        cj[index] = by_name[fields[3]]; p2[index] = int(fields[4])
        s1[index] = strand_code(fields[5]); s2[index] = strand_code(fields[6])
        hashes[index] = fold_hash((int(ci[index]), int(p1[index]), fields[5]),
                                  (int(cj[index]), int(p2[index]), fields[6]))
    train = hashes < np.uint64(THRESHOLD)
    return tuple(names), np.asarray(lengths, dtype=np.int64), {
        "ci": ci, "p1": p1, "cj": cj, "p2": p2, "strand1": s1, "strand2": s2,
        "fold_hash": hashes, "train": train,
    }


def subset_budget(arrays, mask):
    ci, cj = arrays["ci"][mask], arrays["cj"][mask]
    p1, p2 = arrays["p1"][mask], arrays["p2"][mask]
    cis = ci == cj
    return {"records": int(mask.sum()), "cis": int(cis.sum()), "inter": int((~cis).sum()),
            "same_raw_position": int(np.count_nonzero(cis & (p1 == p2)))}


@dataclass(frozen=True)
class GridOnly:
    chromosome_names: tuple[str, ...]
    chromosome_lengths: np.ndarray
    n_bins: np.ndarray
    offsets: np.ndarray
    n_loci: int
    l0: float

    def chromosome_slice(self, chromosome_index):
        start = int(self.offsets[chromosome_index])
        return slice(start, start + int(self.n_bins[chromosome_index]))


def grid_only(data):
    return GridOnly(tuple(data.chromosome_names), np.asarray(data.chromosome_lengths).copy(),
                    np.asarray(data.n_bins).copy(), np.asarray(data.offsets).copy(),
                    int(data.n_loci), float(data.l0))


def make_initials(template):
    grid = grid_only(template)
    output = []
    for seed in INIT_SEEDS:
        coordinates, metadata = v1_calibration.generate_truth(grid, seed=seed, same_shape=False)
        # Dependency checks use independently constructed grid-only views; counts never enter adapter.
        zero_grid = grid_only(replace(template, counts=np.zeros_like(template.counts),
                                      diag_counts=np.zeros_like(template.diag_counts),
                                      endpoint_counts=np.zeros_like(template.endpoint_counts)))
        permutation = np.arange(len(template.counts))[::-1]
        perm_grid = grid_only(replace(template, counts=np.asarray(template.counts)[permutation].copy()))
        zero_coordinates, _ = v1_calibration.generate_truth(zero_grid, seed=seed, same_shape=False)
        perm_coordinates, _ = v1_calibration.generate_truth(perm_grid, seed=seed, same_shape=False)
        if not (np.array_equal(coordinates, zero_coordinates)
                and np.array_equal(coordinates, perm_coordinates)):
            raise AssertionError("polymer initialization depends on contact arrays")
        raw_y = contact_model.sphere_inverse(coordinates)
        p = 0.75
        q = float(contact_model.q_from_p(p))
        theta = np.concatenate((raw_y.reshape(-1), np.asarray([q], dtype=np.float64)))
        directory = RUN / "states/experiment3/initial" / ("seed-%d" % seed) / "5Mb"
        state = write_solver_state(directory / "solver_state.npz", coordinates=coordinates,
                                   raw_y=raw_y, theta=theta, p=p, q=q)
        presence = np.ones((2, template.n_loci), dtype=bool)
        mask = write_presence_mask(directory / "presence_mask.npz", presence, template.n_loci)
        output.append({"seed": seed, "state": state, "presence": mask,
                       "coordinates_sha256": hashlib.sha256(coordinates.tobytes(order="C")).hexdigest(),
                       "dependency_zero_counts_bitwise": True,
                       "dependency_permuted_counts_bitwise": True,
                       "metadata": metadata})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.parse_args()
    selfcheck = encoding_selfcheck()
    names, lengths, arrays = load_and_split()
    train = arrays["train"]
    test = ~train
    unique_hash, multiplicities = np.unique(arrays["fold_hash"], return_counts=True)
    duplicate_groups = int(np.count_nonzero(multiplicities > 1))
    # 8-byte collisions and exact duplicates are both absent in this frozen input.
    if duplicate_groups != 0:
        raise RuntimeError("canonical/hash groups are no longer singleton")
    for label, mask in (("train", train), ("test", test)):
        path = RUN / "inputs" / (label + "_contacts.npz")
        if path.exists():
            raise RuntimeError("refusing to overwrite split contacts")
        with path.open("xb") as handle:
            np.savez_compressed(handle, **{key: np.asarray(value)[mask] for key, value in arrays.items()
                                          if key not in ("train",)})
    aggregate_records = []
    condition_data = {}
    for bin_size in BIN_SIZES:
        base = contact_model.aggregate_from_arrays(
            names, lengths, arrays["ci"][train], arrays["p1"][train],
            arrays["cj"][train], arrays["p2"][train], bin_size)
        offdiag_endpoints = np.asarray(base.endpoint_counts, dtype=np.int64) - 2 * np.asarray(base.diag_counts, dtype=np.int64)
        exposure_off = np.sqrt(offdiag_endpoints.astype(np.float64) + 10.0)
        exposure_off /= exposure_off.mean()
        off = replace(base, exposure=exposure_off, exposure_mode="train_offdiag_endpoint")
        off.assert_consistent()
        base = replace(base, exposure_mode="train_all_endpoint")
        base.assert_consistent()
        for condition, data in (("G-original", base), ("G-offdiag-e", off)):
            path = RUN / "inputs" / ("%s_%d_aggregate.npz" % (condition, bin_size))
            record = data_io.save_aggregate(path, data)
            aggregate_records.append({"condition": condition, "bin_size": bin_size, **record})
            condition_data[(condition, bin_size)] = data
        if not (np.array_equal(base.counts, off.counts)
                and np.array_equal(base.diag_counts, off.diag_counts)
                and float(base.raw_records) == float(off.raw_records)):
            raise AssertionError("conditions differ beyond exposure")
    initials = make_initials(condition_data[("G-original", 5_000_000)])
    manifest = {
        "schema": "p9016-056-experiment3-preparation-v1",
        "status": "prepared",
        "fold": {"algorithm": "BLAKE2b-64", "seed": SEED,
                 "threshold_literal": str(THRESHOLD), "selfcheck": selfcheck,
                 "train_fraction": float(train.mean()), "duplicate_groups": duplicate_groups,
                 "all_groups_singleton": True, "unknown_molecule_isolation": False,
                 "group_overlap": 0},
        "budget": {"all": subset_budget(arrays, np.ones_like(train, dtype=bool)),
                   "train": subset_budget(arrays, train), "test": subset_budget(arrays, test)},
        "split_files": {label: {"path": str((RUN / "inputs" / (label + "_contacts.npz")).relative_to(ROOT)),
                                  "sha256": sha256_file(RUN / "inputs" / (label + "_contacts.npz"))}
                        for label in ("train", "test")},
        "aggregates": aggregate_records,
        "initials": initials,
        "reference_opened": False, "phase_opened": False,
    }
    write_json(RUN / "inputs/experiment3_manifest.json", manifest)
    print(json.dumps({"status": "prepared", "train": int(train.sum()), "test": int(test.sum()),
                      "train_fraction": float(train.mean()), "duplicate_groups": duplicate_groups}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
