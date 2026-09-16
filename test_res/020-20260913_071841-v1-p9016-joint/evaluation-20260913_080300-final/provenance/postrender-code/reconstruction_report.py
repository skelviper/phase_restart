"""Final reconstruction report guards, pure metrics, and local renderers.

Nothing in this module opens P9016 phase fields, the reference 3DG, or an oracle
coordinate by default.  Evaluation-side inputs are supplied through explicit
loaders only after :func:`guarded_evaluation_inputs` revalidates a completed
training selection, every usable candidate coordinate file, and every baseline
source gate.

The selection document deliberately separates immutable training provenance from
mutable status.  ``frozen_provenance`` points to hashed protocol/config/code/data
snapshots.  ``status`` and optimization notes remain outside those snapshots.
"""
from __future__ import annotations

import gzip
import html
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .gate import sha256_file


SELECTION_SCHEMA_VERSION = "reconstruction-selection-v1"
TRACK_MAP_SCHEMA_VERSION = "reconstruction-track-map-v1"
TRAINING_CODE_MANIFEST_SCHEMA_VERSION = "reconstruction-training-code-snapshot-v1"
PROTOCOL_MANIFEST_SCHEMA_VERSION = "reconstruction-protocol-snapshot-v1"
FROZEN_PROTOCOL_SHA256 = "cf2ae36fdd4dec6e05d32001a5ef498add7bb62e5908a4373b2316fc380ed8e5"
FINAL_BIN_BP = 1_000_000
FULL_GRID_ORIGIN_BP = 0
METRIC_GRID_OFFSET_BP = 3_000_000
FRAGMENT_BP = 20_000_000
NUCLEAR_RADIUS = 1.0
NUCLEAR_RADIUS_TOL = 1e-6
RAW_P9016_PAIRS_SHA256 = "071a6cc76bfad543ea1ace6ee1ce3022b30ac1b1e50a9f0c3a3a1967b9649505"
RAW_P9016_RECORDS = 1_703_888
FROZEN_P9016 = {
    "sample_id": "P9016",
    "biological_samples": 1,
    "raw_contacts": 1_703_888,
    "intra_contacts": 1_135_454,
    "inter_contacts": 568_434,
    "snpfree_sha256": "f37ed9cc022a7b37653dddb3e3302be7406204d3848971a333a902afb9a3c9aa",
    "n_chromosomes": 20,
    "full_grid_loci": 2_645,
    "full_grid_physical_beads": 5_290,
}
TERMINAL_STATUSES = frozenset(("converged", "not_converged", "failed"))
COUNT_MODEL_FIELDS = (
    "count_nll_normalized", "conditional_nll_raw", "diag_profiled_nll_raw", "count_nll_raw",
    "p", "bond", "repulsion", "bend", "p_prior", "total",
)
PRIOR_REPORT_FIELDS = ("bond", "repulsion", "bend", "p_prior")
CANVAS_CHROMOSOME_COLORS = (
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
    "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
    "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
)
MUTABLE_PROVENANCE_KEYS = frozenset(("status", "run_status", "notes", "selected_id",
                                     "optimizer_status", "optimization"))


class SelectionValidationError(RuntimeError):
    """Raised before evaluator-side label or reference access is allowed."""


@dataclass(frozen=True)
class Chromosome:
    """One chromosome in the SNP-free header order."""

    name: str
    length_bp: int


@dataclass(frozen=True)
class FullGrid:
    """The final 1 Mb coordinate grid, beginning at genomic position zero."""

    chromosomes: tuple[Chromosome, ...]
    bin_size_bp: int = FINAL_BIN_BP
    origin_bp: int = FULL_GRID_ORIGIN_BP

    @property
    def n_loci(self) -> int:
        return sum(len(self.positions(chromosome)) for chromosome in self.chromosomes)

    @property
    def n_physical_beads(self) -> int:
        return 2 * self.n_loci

    @property
    def n_tracks(self) -> int:
        return 2 * len(self.chromosomes)

    def positions(self, chromosome: Chromosome) -> tuple[int, ...]:
        return tuple(range(self.origin_bp, chromosome.length_bp, self.bin_size_bp))

    def expected_positions_by_track(self) -> dict[str, tuple[int, ...]]:
        return {
            entry["track"]: self.positions(self.chromosomes[entry["chromosome_index"]])
            for entry in track_map_for_grid(self)
        }


@dataclass(frozen=True)
class CoordinateInventory:
    """Finite-coordinate inventory produced without loading labels or references."""

    path: str
    sha256: str
    n_tracks: int
    n_beads: int
    beads_per_track: dict[str, int]
    full_grid_complete: bool
    max_radius: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "n_tracks": self.n_tracks,
            "n_beads": self.n_beads,
            "beads_per_track": dict(self.beads_per_track),
            "full_grid_complete": self.full_grid_complete,
            "max_radius": self.max_radius,
        }


@dataclass(frozen=True)
class VerifiedCandidate:
    """A candidate whose terminal state has been checked against selection.json."""

    candidate_id: str
    terminal_status: str
    coordinates_path: str | None
    inventory: CoordinateInventory | None
    count_nll_per_record: float | None
    failure_reason: str | None


@dataclass(frozen=True)
class VerifiedSelection:
    """Validated training selection; it does not carry labels or reference coordinates."""

    selection_path: str
    document: dict[str, Any]
    grid: FullGrid
    track_map_path: str
    candidates: dict[str, VerifiedCandidate]
    selected_id: str
    frozen_provenance: dict[str, dict[str, Any]]

    @property
    def selected(self) -> VerifiedCandidate:
        return self.candidates[self.selected_id]


@dataclass(frozen=True)
class BaselineSpec:
    """Explicit source-gate contract for a baseline loaded after training completion."""

    tag: str
    role: str
    source_gate_path: str
    source_gate_sha256: str
    coordinates_path: str
    coordinates_sha256: str


@dataclass(frozen=True)
class VerifiedBaseline:
    """A baseline coordinate asset verified against its immutable source gate."""

    tag: str
    role: str
    source_gate_path: str
    coordinates_path: str
    inventory: CoordinateInventory


@dataclass(frozen=True)
class EvaluatorInputAuditSpec:
    """Immutable evaluator-input version contract used only after training guards pass."""

    raw_pairs_path: str
    reference_3dg_path: str
    reference_017_provenance_path: str
    reference_017_provenance_sha256: str
    expected_raw_pairs_sha256: str = RAW_P9016_PAIRS_SHA256
    expected_record_count: int = RAW_P9016_RECORDS


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionValidationError("%s must be an object" % label)
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SelectionValidationError("%s must be a list" % label)
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SelectionValidationError("%s must be a non-empty string" % label)
    return value


def _require_sha256(value: Any, label: str) -> str:
    value = _require_text(value, label)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise SelectionValidationError("%s must be a lowercase SHA256 digest" % label)
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionValidationError("%s must be a finite number" % label)
    value = float(value)
    if not math.isfinite(value):
        raise SelectionValidationError("%s must be finite" % label)
    return value


def _resolve_existing(path_text: Any, base: Path, label: str) -> Path:
    path_text = _require_text(path_text, label)
    path = Path(path_text)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise SelectionValidationError("%s does not exist or is not a regular file: %s" % (label, path))
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionValidationError("cannot read %s: %s" % (label, exc)) from exc
    return dict(_require_mapping(value, label))


def _relative_or_absolute(path: Path) -> str:
    return str(path)


def _track_name(chromosome_index: int, copy_index: int) -> str:
    return "c%02d%s" % (chromosome_index + 1, "ab"[copy_index])


def track_map_for_grid(grid: FullGrid) -> list[dict[str, Any]]:
    """Return native c01a/c01b... tracks in raw SNP-free header order."""
    rows = []
    for chromosome_index, chromosome in enumerate(grid.chromosomes):
        for copy_index, copy_name in enumerate(("a", "b")):
            rows.append({
                "track": _track_name(chromosome_index, copy_index),
                "chromosome_index": chromosome_index,
                "chromosome": chromosome.name,
                "copy": copy_name,
            })
    return rows


def write_track_map(path: str | os.PathLike[str], grid: FullGrid) -> dict[str, Any]:
    """Write the immutable native-track mapping required by a formal run."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TRACK_MAP_SCHEMA_VERSION,
        "coordinate_grid": {
            "bin_size_bp": grid.bin_size_bp,
            "origin_bp": grid.origin_bp,
            "n_loci": grid.n_loci,
            "n_physical_beads": grid.n_physical_beads,
            "nuclear_radius": 1.0,
            "coordinate_units": "dimensionless_R1",
        },
        "tracks": track_map_for_grid(grid),
    }
    with target.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"path": str(target), "sha256": sha256_file(target), "n_tracks": grid.n_tracks}


def _grid_from_document(value: Any, strict_p9016: bool) -> FullGrid:
    document = _require_mapping(value, "coordinate_grid")
    if document.get("bin_size_bp") != FINAL_BIN_BP:
        raise SelectionValidationError("coordinate_grid.bin_size_bp must be 1000000")
    if document.get("origin_bp") != FULL_GRID_ORIGIN_BP:
        raise SelectionValidationError("coordinate_grid.origin_bp must be 0 for the full training grid")
    if document.get("full_grid") is not True:
        raise SelectionValidationError("coordinate_grid.full_grid must be true")
    if document.get("nuclear_radius") != 1.0:
        raise SelectionValidationError("coordinate_grid.nuclear_radius must be the dimensionless value 1.0")
    if document.get("coordinate_units") != "dimensionless_R1":
        raise SelectionValidationError("coordinate_grid.coordinate_units must be dimensionless_R1")
    rows = _require_list(document.get("chromosomes"), "coordinate_grid.chromosomes")
    chromosomes = []
    seen = set()
    for index, row in enumerate(rows):
        row = _require_mapping(row, "coordinate_grid.chromosomes[%d]" % index)
        name = _require_text(row.get("name"), "coordinate_grid.chromosomes[%d].name" % index)
        length = row.get("length_bp")
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise SelectionValidationError("coordinate_grid.chromosomes[%d].length_bp must be positive" % index)
        if name in seen:
            raise SelectionValidationError("coordinate_grid has duplicate chromosome %s" % name)
        seen.add(name)
        chromosomes.append(Chromosome(name, length))
    grid = FullGrid(tuple(chromosomes), FINAL_BIN_BP, FULL_GRID_ORIGIN_BP)
    if document.get("expected_loci") != grid.n_loci:
        raise SelectionValidationError("coordinate_grid.expected_loci does not match header-derived full grid")
    if document.get("expected_physical_beads") != grid.n_physical_beads:
        raise SelectionValidationError(
            "coordinate_grid.expected_physical_beads does not match two-copy full grid")
    if strict_p9016:
        if len(chromosomes) != FROZEN_P9016["n_chromosomes"]:
            raise SelectionValidationError("P9016 selection must retain all 20 header chromosomes")
        if grid.n_loci != FROZEN_P9016["full_grid_loci"]:
            raise SelectionValidationError("P9016 full grid must contain 2645 loci, not %d" % grid.n_loci)
        if grid.n_physical_beads != FROZEN_P9016["full_grid_physical_beads"]:
            raise SelectionValidationError("P9016 full grid must contain 5290 physical beads")
    return grid


def _verify_track_map(reference: Any, base: Path, grid: FullGrid) -> str:
    reference = _require_mapping(reference, "track_map")
    path = _resolve_existing(reference.get("path"), base, "track_map.path")
    expected_sha = _require_sha256(reference.get("sha256"), "track_map.sha256")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise SelectionValidationError("track_map SHA256 mismatch for %s" % path)
    payload = _read_json(path, "track_map")
    if payload.get("schema_version") != TRACK_MAP_SCHEMA_VERSION:
        raise SelectionValidationError("track_map has an unsupported schema version")
    if payload.get("tracks") != track_map_for_grid(grid):
        raise SelectionValidationError("track_map does not match the SNP-free header-order native tracks")
    coordinate_grid = _require_mapping(payload.get("coordinate_grid"), "track_map.coordinate_grid")
    if (coordinate_grid.get("bin_size_bp") != grid.bin_size_bp
            or coordinate_grid.get("origin_bp") != grid.origin_bp
            or coordinate_grid.get("n_loci") != grid.n_loci
            or coordinate_grid.get("n_physical_beads") != grid.n_physical_beads
            or coordinate_grid.get("nuclear_radius") != 1.0
            or coordinate_grid.get("coordinate_units") != "dimensionless_R1"):
        raise SelectionValidationError("track_map grid metadata disagrees with selection.json")
    return _relative_or_absolute(path)


def _manifest_relative_path(value: Any, label: str) -> str:
    text = _require_text(value, label)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise SelectionValidationError("%s must be run-relative: %s" % (label, text))
    return text


def _verify_snapshot_members(path: Path, base: Path, schema_version: str,
                             label: str) -> dict[str, Any]:
    manifest = _read_json(path, label)
    if manifest.get("schema_version") != schema_version:
        raise SelectionValidationError("%s has an unsupported schema version" % label)
    rows = _require_list(manifest.get("files"), "%s.files" % label)
    if not rows:
        raise SelectionValidationError("%s.files may not be empty" % label)
    seen_source = set()
    seen_snapshot = set()
    members = []
    for index, value in enumerate(rows):
        row = _require_mapping(value, "%s.files[%d]" % (label, index))
        source_path = _manifest_relative_path(
            row.get("source_path"), "%s.files[%d].source_path" % (label, index))
        snapshot_path_text = _manifest_relative_path(
            row.get("snapshot_path"), "%s.files[%d].snapshot_path" % (label, index))
        expected_sha = _require_sha256(
            row.get("sha256"), "%s.files[%d].sha256" % (label, index))
        if source_path in seen_source or snapshot_path_text in seen_snapshot:
            raise SelectionValidationError("%s contains duplicate source or snapshot paths" % label)
        snapshot_path = _resolve_existing(snapshot_path_text, base,
                                          "%s.files[%d].snapshot_path" % (label, index))
        if sha256_file(snapshot_path) != expected_sha:
            raise SelectionValidationError("%s member SHA256 mismatch for %s" % (label, snapshot_path))
        seen_source.add(source_path)
        seen_snapshot.add(snapshot_path_text)
        members.append({"source_path": source_path, "snapshot_path": snapshot_path_text,
                        "sha256": expected_sha})
    return {"member_count": len(members), "members": members}


def _verify_training_code_manifest(path: Path, base: Path) -> dict[str, Any]:
    """Validate every immutable training-code snapshot named by a code manifest."""
    return _verify_snapshot_members(path, base, TRAINING_CODE_MANIFEST_SCHEMA_VERSION,
                                    "training code snapshot manifest")


def _verify_protocol_manifest(path: Path, base: Path) -> dict[str, Any]:
    """Validate protocol snapshot members without consulting workspace sources."""
    result = _verify_snapshot_members(path, base, PROTOCOL_MANIFEST_SCHEMA_VERSION,
                                      "protocol snapshot manifest")
    protocol_members = [row for row in result["members"]
                        if row["source_path"] == "docs/RECONSTRUCTION_V1_PROTOCOL.md"]
    if len(protocol_members) != 1 or protocol_members[0]["sha256"] != FROZEN_PROTOCOL_SHA256:
        raise SelectionValidationError("protocol snapshot does not contain the frozen protocol bytes")
    return result


def _verify_frozen_provenance(value: Any, base: Path, strict_p9016: bool,
                               cohort: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    document = _require_mapping(value, "frozen_provenance")
    mutable = sorted(MUTABLE_PROVENANCE_KEYS & set(document))
    if mutable:
        raise SelectionValidationError(
            "frozen_provenance contains mutable run fields: %s" % ", ".join(mutable))
    verified: dict[str, dict[str, Any]] = {}
    for role in ("protocol", "config", "code", "data"):
        entry = _require_mapping(document.get(role), "frozen_provenance.%s" % role)
        path = _resolve_existing(entry.get("path"), base, "frozen_provenance.%s.path" % role)
        expected_sha = _require_sha256(entry.get("sha256"), "frozen_provenance.%s.sha256" % role)
        hash_source = _require_text(entry.get("hash_source"), "frozen_provenance.%s.hash_source" % role)
        if role in ("protocol", "config", "data") and hash_source != "file_bytes":
            raise SelectionValidationError(
                "frozen_provenance.%s.hash_source must be file_bytes" % role)
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise SelectionValidationError("frozen %s SHA256 mismatch for %s" % (role, path))
        result: dict[str, Any] = {"path": _relative_or_absolute(path), "sha256": actual_sha,
                                  "hash_source": hash_source}
        if role == "code":
            if hash_source == "snapshot_manifest" or path.name == "training-code-manifest.json":
                result.update(_verify_training_code_manifest(path, base))
            elif hash_source != "file_bytes":
                raise SelectionValidationError(
                    "frozen_provenance.code.hash_source must be snapshot_manifest or file_bytes")
        elif role == "protocol" and (
                entry.get("manifest_schema") == PROTOCOL_MANIFEST_SCHEMA_VERSION
                or path.name == "protocol-manifest.json"):
            if entry.get("manifest_schema") not in (None, PROTOCOL_MANIFEST_SCHEMA_VERSION):
                raise SelectionValidationError("frozen protocol manifest schema is unsupported")
            result.update(_verify_protocol_manifest(path, base))
        verified[role] = result
    if strict_p9016 and verified["data"]["sha256"] != cohort["snpfree_sha256"]:
        raise SelectionValidationError("frozen data digest must be the declared P9016 SNP-free input digest")
    return verified


def _validate_cohort(value: Any, strict_p9016: bool) -> Mapping[str, Any]:
    cohort = _require_mapping(value, "cohort")
    required = ("sample_id", "biological_samples", "raw_contacts", "intra_contacts",
                "inter_contacts", "snpfree_sha256")
    for key in required:
        if key not in cohort:
            raise SelectionValidationError("cohort.%s is required" % key)
    for key in ("biological_samples", "raw_contacts", "intra_contacts", "inter_contacts"):
        if isinstance(cohort[key], bool) or not isinstance(cohort[key], int):
            raise SelectionValidationError("cohort.%s must be an integer" % key)
    if cohort["biological_samples"] != 1:
        raise SelectionValidationError("cohort.biological_samples must be 1 for this single-cell report")
    if cohort["raw_contacts"] != cohort["intra_contacts"] + cohort["inter_contacts"]:
        raise SelectionValidationError("cohort raw/intra/inter contact counts are inconsistent")
    _require_sha256(cohort["snpfree_sha256"], "cohort.snpfree_sha256")
    if strict_p9016:
        for key, expected in FROZEN_P9016.items():
            if key in ("n_chromosomes", "full_grid_loci", "full_grid_physical_beads"):
                continue
            if cohort.get(key) != expected:
                raise SelectionValidationError("P9016 cohort.%s must equal %r" % (key, expected))
    return cohort


def _validate_contract(value: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    contract = _require_mapping(value, "objective_contract")
    required = {
        "criterion": "count_nll_per_record",
        "direction": "minimize",
        "conditional_count_objective": True,
        "includes_candidate_invariant_diagonal_layer": True,
        "includes_priors": False,
        "uses_all_contacts": True,
        "selection_is_label_free": True,
        "reference_used_for_selection": False,
        "phase_used_for_selection": False,
        "bin_size_bp": FINAL_BIN_BP,
        "tie_break": "preregistered_order",
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise SelectionValidationError("objective_contract.%s must equal %r" % (key, expected))
    if _finite_number(contract.get("tie_tolerance_per_record"),
                      "objective_contract.tie_tolerance_per_record") != 1e-9:
        raise SelectionValidationError("objective_contract.tie_tolerance_per_record must be 1e-9")
    count_model_fields = _require_list(contract.get("count_model_fields"),
                                       "objective_contract.count_model_fields")
    if count_model_fields != list(COUNT_MODEL_FIELDS):
        raise SelectionValidationError("objective_contract.count_model_fields must match the frozen count model")
    prior_names = _require_list(contract.get("prior_component_names"),
                                "objective_contract.prior_component_names")
    if prior_names != list(PRIOR_REPORT_FIELDS):
        raise SelectionValidationError("objective_contract.prior_component_names must match frozen prior diagnostics")
    model = _require_mapping(contract.get("model_contract"),
                             "objective_contract.model_contract")
    # The model signature is immutable and lets every candidate prove that its
    # dimensionality and prior configuration were identical before selection.
    _require_sha256(model.get("model_signature"), "objective_contract.model_contract.model_signature")
    _require_sha256(model.get("prior_config_sha256"),
                    "objective_contract.model_contract.prior_config_sha256")
    dimension = model.get("parameter_dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise SelectionValidationError("objective_contract.model_contract.parameter_dimension must be positive")
    if model.get("bin_size_bp") != FINAL_BIN_BP:
        raise SelectionValidationError("objective_contract.model_contract.bin_size_bp must be 1000000")
    if model.get("prior_component_names") != prior_names:
        raise SelectionValidationError("model_contract prior components must match objective_contract")
    return contract, model


def _read_coordinate_rows(path: Path) -> dict[str, dict[int, np.ndarray]]:
    opener = gzip.open if path.suffix == ".gz" else open
    coordinates: dict[str, dict[int, np.ndarray]] = {}
    try:
        with opener(path, "rt") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise SelectionValidationError("coordinate %s line %d has %d fields, expected 5"
                                                   % (path, line_no, len(fields)))
                track = fields[0]
                try:
                    position = int(fields[1])
                    xyz = np.asarray([float(value) for value in fields[2:5]], dtype=float)
                except ValueError as exc:
                    raise SelectionValidationError("coordinate %s line %d is not numeric" % (path, line_no)) from exc
                if position < 0 or not np.isfinite(xyz).all():
                    raise SelectionValidationError("coordinate %s line %d is non-finite or negative" % (path, line_no))
                by_position = coordinates.setdefault(track, {})
                if position in by_position:
                    raise SelectionValidationError("coordinate %s duplicates %s:%d" % (path, track, position))
                by_position[position] = xyz
    except OSError as exc:
        raise SelectionValidationError("cannot read coordinate %s: %s" % (path, exc)) from exc
    if not coordinates:
        raise SelectionValidationError("coordinate file is empty: %s" % path)
    return coordinates


def _max_coordinate_radius(coordinates: Mapping[str, Mapping[int, np.ndarray]]) -> float:
    radii = [float(np.linalg.norm(np.asarray(xyz, dtype=float)))
             for rows in coordinates.values() for xyz in rows.values()]
    return max(radii, default=0.0)


def read_full_grid_coordinates(path: str | os.PathLike[str], grid: FullGrid,
                                expected_sha256: str | None = None) -> dict[str, dict[int, np.ndarray]]:
    """Read a coordinate file only after enforcing finite 40-track full-grid syntax."""
    target = Path(path).resolve()
    if not target.is_file():
        raise SelectionValidationError("coordinate file is unavailable: %s" % target)
    if expected_sha256 is not None and sha256_file(target) != _require_sha256(expected_sha256, "coordinates.sha256"):
        raise SelectionValidationError("coordinate SHA256 mismatch for %s" % target)
    coordinates = _read_coordinate_rows(target)
    expected = grid.expected_positions_by_track()
    found_tracks = set(coordinates)
    if found_tracks != set(expected):
        raise SelectionValidationError("coordinate track mismatch: missing=%s unexpected=%s" % (
            sorted(set(expected) - found_tracks), sorted(found_tracks - set(expected))))
    for track, positions in expected.items():
        observed = set(coordinates[track])
        if observed != set(positions):
            raise SelectionValidationError("coordinate full-grid mismatch for %s: missing=%d unexpected=%d" % (
                track, len(set(positions) - observed), len(observed - set(positions))))
    max_radius = _max_coordinate_radius(coordinates)
    if max_radius > NUCLEAR_RADIUS + NUCLEAR_RADIUS_TOL:
        raise SelectionValidationError("coordinate violates the raw dimensionless R=1 nuclear ball: max radius %.9g" % max_radius)
    return coordinates


def verify_full_grid_structures(structs: Mapping[str, Mapping[int, np.ndarray]], grid: FullGrid) -> dict[str, Any]:
    """Check in-memory selected coordinates before rendering a full 40-track deliverable."""
    expected = grid.expected_positions_by_track()
    if set(structs) != set(expected):
        raise SelectionValidationError("in-memory structure track mismatch: missing=%s unexpected=%s" % (
            sorted(set(expected) - set(structs)), sorted(set(structs) - set(expected))))
    max_radius = 0.0
    counts = {}
    for track, positions in expected.items():
        rows = structs[track]
        if set(rows) != set(positions):
            raise SelectionValidationError("in-memory full-grid mismatch for %s" % track)
        for position, xyz in rows.items():
            point = np.asarray(xyz, dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise SelectionValidationError("in-memory coordinate is non-finite or malformed at %s:%d" %
                                               (track, position))
            max_radius = max(max_radius, float(np.linalg.norm(point)))
        counts[track] = len(rows)
    if max_radius > NUCLEAR_RADIUS + NUCLEAR_RADIUS_TOL:
        raise SelectionValidationError("in-memory coordinates violate the raw dimensionless R=1 nuclear ball")
    return {"n_tracks": len(counts), "n_beads": int(sum(counts.values())), "max_radius": max_radius,
            "beads_per_track": counts}


def coordinate_inventory(path: str | os.PathLike[str], grid: FullGrid,
                         expected_sha256: str | None = None) -> CoordinateInventory:
    """Validate a final candidate coordinate file against the entire 0-based grid."""
    target = Path(path).resolve()
    coordinates = read_full_grid_coordinates(target, grid, expected_sha256)
    counts = {track: len(rows) for track, rows in sorted(coordinates.items())}
    return CoordinateInventory(
        path=_relative_or_absolute(target),
        sha256=sha256_file(target),
        n_tracks=len(counts),
        n_beads=sum(counts.values()),
        beads_per_track=counts,
        full_grid_complete=True,
        max_radius=_max_coordinate_radius(coordinates),
    )


def _validate_component_map(value: Any, expected_names: Sequence[str], label: str) -> dict[str, float]:
    values = _require_mapping(value, label)
    if set(values) != set(expected_names):
        raise SelectionValidationError("%s must contain exactly %s" % (label, list(expected_names)))
    return {name: _finite_number(values[name], "%s.%s" % (label, name)) for name in expected_names}


def _validate_candidates(document: Mapping[str, Any], base: Path, grid: FullGrid,
                         contract: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, VerifiedCandidate]:
    preregistered = _require_list(document.get("preregistered_candidate_ids"),
                                  "preregistered_candidate_ids")
    if not preregistered or any(not isinstance(item, str) or not item for item in preregistered):
        raise SelectionValidationError("preregistered_candidate_ids must be non-empty strings")
    if len(set(preregistered)) != len(preregistered):
        raise SelectionValidationError("preregistered_candidate_ids contains duplicates")
    candidates = _require_list(document.get("candidates"), "candidates")
    ids = []
    candidate_by_id = {}
    for index, value in enumerate(candidates):
        candidate = _require_mapping(value, "candidates[%d]" % index)
        candidate_id = _require_text(candidate.get("id"), "candidates[%d].id" % index)
        if candidate_id in candidate_by_id:
            raise SelectionValidationError("candidates contains duplicate id %s" % candidate_id)
        ids.append(candidate_id)
        candidate_by_id[candidate_id] = candidate
    if ids != preregistered:
        raise SelectionValidationError("candidates must list every pre-registered candidate once and in frozen order")

    attempts = _require_list(document.get("attempts"), "attempts")
    attempts_by_candidate: dict[str, list[Mapping[str, Any]]] = {candidate_id: [] for candidate_id in preregistered}
    attempt_ids = set()
    for index, value in enumerate(attempts):
        attempt = _require_mapping(value, "attempts[%d]" % index)
        attempt_id = _require_text(attempt.get("attempt_id"), "attempts[%d].attempt_id" % index)
        candidate_id = _require_text(attempt.get("candidate_id"), "attempts[%d].candidate_id" % index)
        _require_text(attempt.get("stage"), "attempts[%d].stage" % index)
        _require_text(attempt.get("status"), "attempts[%d].status" % index)
        if attempt_id in attempt_ids:
            raise SelectionValidationError("attempts contains duplicate attempt_id %s" % attempt_id)
        if candidate_id not in attempts_by_candidate:
            raise SelectionValidationError("attempt %s names an unregistered candidate" % attempt_id)
        if not isinstance(attempt.get("is_terminal"), bool):
            raise SelectionValidationError("attempts[%d].is_terminal must be boolean" % index)
        attempt_ids.add(attempt_id)
        attempts_by_candidate[candidate_id].append(attempt)

    prior_names = list(contract["prior_component_names"])
    verified: dict[str, VerifiedCandidate] = {}
    for candidate_id in preregistered:
        candidate = candidate_by_id[candidate_id]
        optimization = _require_mapping(candidate.get("optimization"), "candidate %s optimization" % candidate_id)
        terminal_status = _require_text(optimization.get("terminal_status"),
                                        "candidate %s optimization.terminal_status" % candidate_id)
        if terminal_status not in TERMINAL_STATUSES:
            raise SelectionValidationError("candidate %s has invalid terminal status %s" % (candidate_id, terminal_status))
        terminal_attempt_id = _require_text(optimization.get("terminal_attempt_id"),
                                            "candidate %s optimization.terminal_attempt_id" % candidate_id)
        declared_attempt_ids = _require_list(candidate.get("attempt_ids"), "candidate %s attempt_ids" % candidate_id)
        observed_attempt_ids = [attempt["attempt_id"] for attempt in attempts_by_candidate[candidate_id]]
        if declared_attempt_ids != observed_attempt_ids:
            raise SelectionValidationError("candidate %s attempt_ids omit or reorder journal entries" % candidate_id)
        terminal_attempts = [attempt for attempt in attempts_by_candidate[candidate_id]
                             if attempt["is_terminal"]]
        if len(terminal_attempts) != 1 or terminal_attempts[0]["attempt_id"] != terminal_attempt_id:
            raise SelectionValidationError("candidate %s must have exactly one terminal attempt" % candidate_id)
        if terminal_attempts[0]["status"] != terminal_status:
            raise SelectionValidationError("candidate %s terminal attempt status disagrees with candidate status" % candidate_id)
        if candidate.get("model_signature") != model["model_signature"]:
            raise SelectionValidationError("candidate %s does not use the frozen model signature" % candidate_id)
        if candidate.get("parameter_dimension") != model["parameter_dimension"]:
            raise SelectionValidationError("candidate %s does not use the frozen parameter dimension" % candidate_id)
        if candidate.get("prior_config_sha256") != model["prior_config_sha256"]:
            raise SelectionValidationError("candidate %s does not use the frozen prior configuration" % candidate_id)

        if terminal_status == "failed":
            if candidate.get("coordinates") is not None:
                raise SelectionValidationError("failed candidate %s must declare coordinates as null" % candidate_id)
            reason = _require_text(candidate.get("failure_reason"), "candidate %s failure_reason" % candidate_id)
            if candidate.get("count_nll_per_record") is not None:
                raise SelectionValidationError("failed candidate %s must not present a selectable count objective" % candidate_id)
            verified[candidate_id] = VerifiedCandidate(candidate_id, terminal_status, None, None, None, reason)
            continue

        coordinates_ref = _require_mapping(candidate.get("coordinates"), "candidate %s coordinates" % candidate_id)
        path = _resolve_existing(coordinates_ref.get("path"), base, "candidate %s coordinates.path" % candidate_id)
        expected_sha = _require_sha256(coordinates_ref.get("sha256"),
                                       "candidate %s coordinates.sha256" % candidate_id)
        inventory = coordinate_inventory(path, grid, expected_sha)
        score = _finite_number(candidate.get("count_nll_per_record"),
                               "candidate %s count_nll_per_record" % candidate_id)
        count_model = _validate_component_map(candidate.get("count_model"), COUNT_MODEL_FIELDS,
                                              "candidate %s count_model" % candidate_id)
        if not math.isclose(score, count_model["count_nll_normalized"], rel_tol=0.0, abs_tol=1e-15):
            raise SelectionValidationError(
                "candidate %s count_nll_per_record must equal count_model.count_nll_normalized, never total" % candidate_id)
        if not 0.0 <= count_model["p"] <= 1.0:
            raise SelectionValidationError("candidate %s count_model.p must be a nuisance mixture coefficient in [0, 1]" % candidate_id)
        if "combined_objective" in candidate or "objective_total" in candidate:
            raise SelectionValidationError(
                "candidate %s may not combine physical/numerical priors into count_nll_per_record" % candidate_id)
        verified[candidate_id] = VerifiedCandidate(candidate_id, terminal_status,
                                                    _relative_or_absolute(path), inventory, score, None)
    return verified


def _validate_selection_choice(document: Mapping[str, Any], candidates: Mapping[str, VerifiedCandidate],
                               contract: Mapping[str, Any]) -> str:
    selection = _require_mapping(document.get("selection"), "selection")
    if selection.get("rule") != "minimize_count_nll_per_record":
        raise SelectionValidationError("selection.rule must minimize count_nll_per_record")
    for key in ("reference_used", "phase_used", "all_attempts_accounted_for"):
        expected = False if key != "all_attempts_accounted_for" else True
        if selection.get(key) is not expected:
            raise SelectionValidationError("selection.%s must equal %r" % (key, expected))
    if selection.get("tie_break") != contract["tie_break"]:
        raise SelectionValidationError("selection.tie_break must use pre-registered order")
    selected_id = _require_text(selection.get("selected_id"), "selection.selected_id")
    if selected_id not in candidates:
        raise SelectionValidationError("selection.selected_id is not a registered candidate")
    if candidates[selected_id].terminal_status == "failed":
        raise SelectionValidationError("selection.selected_id may not be a failed candidate")
    usable = [candidate for candidate in candidates.values()
              if candidate.terminal_status != "failed" and candidate.count_nll_per_record is not None]
    if not usable:
        raise SelectionValidationError("no finite non-failed candidate is available for selection")
    best = min(candidate.count_nll_per_record for candidate in usable)
    tolerance = float(contract["tie_tolerance_per_record"])
    expected_id = next(candidate.candidate_id for candidate in usable
                       if candidate.count_nll_per_record <= best + tolerance)
    if selected_id != expected_id:
        raise SelectionValidationError(
            "selection.selected_id violates count_nll_per_record minimization and frozen tie order")
    return selected_id


def _validate_baseline_assets(value: Any, candidate_ids: Sequence[str], strict_p9016: bool) -> None:
    assets = _require_list(value, "baseline_assets")
    tags = []
    for index, item in enumerate(assets):
        item = _require_mapping(item, "baseline_assets[%d]" % index)
        asset_id = _require_text(item.get("id"), "baseline_assets[%d].id" % index)
        tag = _require_text(item.get("tag"), "baseline_assets[%d].tag" % index)
        role = _require_text(item.get("role"), "baseline_assets[%d].role" % index)
        if asset_id in candidate_ids:
            raise SelectionValidationError("baseline asset %s must not appear in candidate inventory" % asset_id)
        if tag == "oracle" and role != "evaluation_ceiling":
            raise SelectionValidationError("oracle baseline may only be an evaluation ceiling")
        if tag in ("consensus", "random") and role != "blind_baseline":
            raise SelectionValidationError("%s baseline must be marked blind_baseline" % tag)
        tags.append(tag)
    if len(set(tags)) != len(tags):
        raise SelectionValidationError("baseline_assets contains duplicate tags")
    if strict_p9016 and set(tags) != {"consensus", "random", "oracle"}:
        raise SelectionValidationError("P9016 report requires consensus/random/oracle baseline assets")


def verify_training_complete(selection_path: str | os.PathLike[str], *, strict_p9016: bool = True) -> VerifiedSelection:
    """Revalidate every pre-registered candidate before evaluator-side access.

    This function reads only the selection document, frozen local provenance,
    track map, and coordinate files.  It never opens phase-bearing pairs, a 3DG
    reference, or any baseline coordinate asset.
    """
    selection_file = Path(selection_path).resolve()
    if not selection_file.is_file():
        raise SelectionValidationError("selection.json is unavailable: %s" % selection_file)
    document = _read_json(selection_file, "selection.json")
    if document.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise SelectionValidationError("selection.json has an unsupported schema version")
    if document.get("status") != "training_complete":
        raise SelectionValidationError("selection.status must be training_complete before evaluation")
    base = selection_file.parent
    cohort = _validate_cohort(document.get("cohort"), strict_p9016)
    grid = _grid_from_document(document.get("coordinate_grid"), strict_p9016)
    frozen_provenance = _verify_frozen_provenance(document.get("frozen_provenance"), base,
                                                   strict_p9016, cohort)
    track_map_path = _verify_track_map(document.get("track_map"), base, grid)
    contract, model = _validate_contract(document.get("objective_contract"))
    candidates = _validate_candidates(document, base, grid, contract, model)
    _validate_baseline_assets(document.get("baseline_assets"), list(candidates), strict_p9016)
    selected_id = _validate_selection_choice(document, candidates, contract)
    return VerifiedSelection(_relative_or_absolute(selection_file), document, grid, track_map_path,
                             candidates, selected_id, frozen_provenance)


def _expected_baseline_tracks(grid: FullGrid, tag: str) -> set[str]:
    if tag == "consensus":
        return {_track_name(index, 0) for index in range(len(grid.chromosomes))}
    if tag in ("random", "oracle"):
        return set(grid.expected_positions_by_track())
    raise SelectionValidationError("unsupported baseline tag %s" % tag)


def _inventory_baseline_coordinates(path: Path, expected_sha: str, expected_tracks: set[str]) -> CoordinateInventory:
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise SelectionValidationError("baseline coordinate SHA256 mismatch for %s" % path)
    coordinates = _read_coordinate_rows(path)
    found_tracks = set(coordinates)
    if found_tracks != expected_tracks:
        raise SelectionValidationError("baseline track mismatch: missing=%s unexpected=%s" % (
            sorted(expected_tracks - found_tracks), sorted(found_tracks - expected_tracks)))
    counts = {track: len(rows) for track, rows in sorted(coordinates.items())}
    return CoordinateInventory(_relative_or_absolute(path), actual_sha, len(counts), sum(counts.values()),
                               counts, False, _max_coordinate_radius(coordinates))


def verify_baseline_source(spec: BaselineSpec, grid: FullGrid) -> VerifiedBaseline:
    """Verify a baseline's source gate and finite coordinates without reading labels/ref."""
    if spec.tag not in ("consensus", "random", "oracle"):
        raise SelectionValidationError("unsupported baseline tag %s" % spec.tag)
    expected_role = "evaluation_ceiling" if spec.tag == "oracle" else "blind_baseline"
    if spec.role != expected_role:
        raise SelectionValidationError("baseline %s must have role %s" % (spec.tag, expected_role))
    source_gate = Path(spec.source_gate_path).resolve()
    coordinates = Path(spec.coordinates_path).resolve()
    if not source_gate.is_file() or not coordinates.is_file():
        raise SelectionValidationError("baseline source gate or coordinate file is unavailable")
    expected_gate_sha = _require_sha256(spec.source_gate_sha256, "baseline.source_gate_sha256")
    if sha256_file(source_gate) != expected_gate_sha:
        raise SelectionValidationError("baseline source gate SHA256 mismatch for %s" % source_gate)
    entries = _read_json_list(source_gate, "baseline source gate")
    matching = [entry for entry in entries if entry.get("tag") == spec.tag]
    if len(matching) != 1:
        raise SelectionValidationError("baseline source gate must contain exactly one %s entry" % spec.tag)
    expected_coordinate_sha = _require_sha256(spec.coordinates_sha256, "baseline.coordinates_sha256")
    if matching[0].get("sha256") != expected_coordinate_sha:
        raise SelectionValidationError("baseline source gate digest disagrees with explicit coordinate digest")
    inventory = _inventory_baseline_coordinates(coordinates, expected_coordinate_sha,
                                                _expected_baseline_tracks(grid, spec.tag))
    return VerifiedBaseline(spec.tag, spec.role, _relative_or_absolute(source_gate),
                            _relative_or_absolute(coordinates), inventory)


def _read_json_list(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionValidationError("cannot read %s: %s" % (label, exc)) from exc
    values = _require_list(value, label)
    return [dict(_require_mapping(item, "%s entry" % label)) for item in values]


def _resolve_direct_file(path_text: Any, label: str) -> Path:
    path = Path(_require_text(path_text, label)).resolve()
    if not path.is_file():
        raise SelectionValidationError("%s does not exist or is not a regular file: %s" % (label, path))
    return path


def _read_json_value(path: Path, label: str) -> Any:
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionValidationError("cannot read %s: %s" % (label, exc)) from exc


def _json_contains_sha256(value: Any, digest: str) -> bool:
    if isinstance(value, str):
        return value == digest
    if isinstance(value, Mapping):
        return any(_json_contains_sha256(item, digest) for item in value.values())
    if isinstance(value, list):
        return any(_json_contains_sha256(item, digest) for item in value)
    return False


def verify_evaluator_input_versions(spec: EvaluatorInputAuditSpec, *, strict_p9016: bool = True) -> dict[str, Any]:
    """Audit raw-pairs/reference versions before either file is parsed as evaluation input."""
    expected_raw_sha = _require_sha256(spec.expected_raw_pairs_sha256, "expected_raw_pairs_sha256")
    if isinstance(spec.expected_record_count, bool) or not isinstance(spec.expected_record_count, int):
        raise SelectionValidationError("expected_record_count must be an integer")
    if strict_p9016:
        if expected_raw_sha != RAW_P9016_PAIRS_SHA256:
            raise SelectionValidationError("P9016 evaluator must use the frozen raw pairs SHA256")
        if spec.expected_record_count != RAW_P9016_RECORDS:
            raise SelectionValidationError("P9016 evaluator must retain all 1703888 raw records")
    raw_pairs = _resolve_direct_file(spec.raw_pairs_path, "evaluator raw_pairs_path")
    reference = _resolve_direct_file(spec.reference_3dg_path, "evaluator reference_3dg_path")
    provenance = _resolve_direct_file(spec.reference_017_provenance_path,
                                      "evaluator reference_017_provenance_path")
    actual_raw_sha = sha256_file(raw_pairs)
    if actual_raw_sha != expected_raw_sha:
        raise SelectionValidationError("evaluator raw pairs SHA256 mismatch for %s" % raw_pairs)
    expected_provenance_sha = _require_sha256(spec.reference_017_provenance_sha256,
                                              "reference_017_provenance_sha256")
    actual_provenance_sha = sha256_file(provenance)
    if actual_provenance_sha != expected_provenance_sha:
        raise SelectionValidationError("017 reference provenance SHA256 mismatch for %s" % provenance)
    actual_reference_sha = sha256_file(reference)
    declared = _read_json_value(provenance, "017 reference provenance")
    if not _json_contains_sha256(declared, actual_reference_sha):
        raise SelectionValidationError(
            "actual reference 3DG SHA256 is absent from the verified 017 provenance declaration")
    return {
        "raw_pairs": {"path": str(raw_pairs), "expected_sha256": expected_raw_sha,
                      "actual_sha256": actual_raw_sha, "expected_records": spec.expected_record_count},
        "reference_3dg": {"path": str(reference), "actual_sha256": actual_reference_sha,
                            "declared_in_017_provenance": True},
        "reference_017_provenance": {"path": str(provenance), "expected_sha256": expected_provenance_sha,
                                       "actual_sha256": actual_provenance_sha},
    }


def _contact_record_count(contacts: Any) -> int:
    contacts = _require_mapping(contacts, "contacts_loader result")
    if "ci" not in contacts:
        raise SelectionValidationError("contacts_loader result must contain ci for record alignment")
    try:
        return len(contacts["ci"])
    except TypeError as exc:
        raise SelectionValidationError("contacts_loader result ci must be a record array") from exc


def _validate_record_alignment(value: Any, expected_records: int, contacts_records: int) -> dict[str, Any]:
    alignment = _require_mapping(value, "alignment_verifier result")
    if alignment.get("aligned") is not True:
        raise SelectionValidationError("alignment_verifier must assert aligned=true before phase parsing")
    for key in ("raw_records", "contacts_records"):
        count = alignment.get(key)
        if isinstance(count, bool) or not isinstance(count, int):
            raise SelectionValidationError("alignment_verifier result.%s must be an integer" % key)
        if count != expected_records:
            raise SelectionValidationError("alignment_verifier result.%s does not retain full M" % key)
    if alignment["contacts_records"] != contacts_records:
        raise SelectionValidationError("alignment_verifier contacts_records disagrees with contacts_loader")
    return {"aligned": True, "raw_records": alignment["raw_records"],
            "contacts_records": alignment["contacts_records"]}


def _validate_label_record_count(labels: Any, expected_records: int) -> int:
    if isinstance(labels, Mapping):
        arrays = [labels.get("a1"), labels.get("a2")]
    elif isinstance(labels, (tuple, list)) and len(labels) == 2:
        arrays = list(labels)
    else:
        raise SelectionValidationError("labels_loader must return an aligned (a1, a2) pair or mapping")
    try:
        lengths = [len(array) for array in arrays]
    except TypeError as exc:
        raise SelectionValidationError("labels_loader returned non-array phase labels") from exc
    if lengths != [expected_records, expected_records]:
        raise SelectionValidationError("labels_loader phase arrays do not retain full M")
    return expected_records


def guarded_final_evaluation_inputs(
        selection_path: str | os.PathLike[str], *, baseline_specs: Mapping[str, BaselineSpec],
        evaluator_input_spec: EvaluatorInputAuditSpec, candidate_loader: Callable[[str], Any],
        baseline_loader: Callable[[str, str], Any], contacts_loader: Callable[[], Any],
        alignment_verifier: Callable[[Any, str], Mapping[str, Any]],
        labels_loader: Callable[[Any, str], Any], reference_loader: Callable[[str], Any],
        strict_p9016: bool = True) -> dict[str, Any]:
    """Production evaluator entrance with version audit and full-M record alignment.

    No raw phase field or reference coordinate parser is invoked until immutable
    training selection, all usable candidate files, baseline source gates, and
    raw/reference version evidence have passed.
    """
    verified = verify_training_complete(selection_path, strict_p9016=strict_p9016)
    expected_tags = {"consensus", "random", "oracle"}
    if set(baseline_specs) != expected_tags:
        raise SelectionValidationError("baseline_specs must explicitly provide consensus, random, and oracle")
    baselines = {tag: verify_baseline_source(baseline_specs[tag], verified.grid)
                 for tag in ("consensus", "random", "oracle")}
    input_audit = verify_evaluator_input_versions(evaluator_input_spec, strict_p9016=strict_p9016)
    candidate_coordinates = {
        candidate_id: candidate_loader(candidate.coordinates_path)
        for candidate_id, candidate in verified.candidates.items()
        if candidate.coordinates_path is not None
    }
    baseline_coordinates = {tag: baseline_loader(tag, item.coordinates_path)
                            for tag, item in baselines.items()}
    contacts = contacts_loader()
    contacts_records = _contact_record_count(contacts)
    expected_records = input_audit["raw_pairs"]["expected_records"]
    alignment = _validate_record_alignment(
        alignment_verifier(contacts, input_audit["raw_pairs"]["path"]), expected_records, contacts_records)
    labels = labels_loader(contacts, input_audit["raw_pairs"]["path"])
    alignment["label_records"] = _validate_label_record_count(labels, expected_records)
    reference = reference_loader(input_audit["reference_3dg"]["path"])
    return {
        "verified_selection": verified,
        "verified_baselines": baselines,
        "evaluator_input_audit": input_audit,
        "record_alignment": alignment,
        "candidate_coordinates": candidate_coordinates,
        "selected_coordinates": candidate_coordinates[verified.selected_id],
        "baseline_coordinates": baseline_coordinates,
        "contacts": contacts,
        "labels": labels,
        "reference": reference,
    }


def guarded_evaluation_inputs(
        selection_path: str | os.PathLike[str], *, baseline_specs: Mapping[str, BaselineSpec],
        candidate_loader: Callable[[str], Any], baseline_loader: Callable[[str, str], Any],
        labels_loader: Callable[[], Any], reference_loader: Callable[[], Any],
        strict_p9016: bool = True) -> dict[str, Any]:
    """Open injected test/compatibility inputs after coordinate and baseline guards.

    Production P9016 evaluation must use :func:`guarded_final_evaluation_inputs`
    so raw/reference version audit and full-M record alignment cannot be skipped.
    """
    if strict_p9016:
        raise SelectionValidationError(
            "strict P9016 evaluation must use guarded_final_evaluation_inputs with evaluator input audit")
    verified = verify_training_complete(selection_path, strict_p9016=strict_p9016)
    expected_tags = {"consensus", "random", "oracle"}
    if set(baseline_specs) != expected_tags:
        raise SelectionValidationError("baseline_specs must explicitly provide consensus, random, and oracle")
    baselines = {tag: verify_baseline_source(baseline_specs[tag], verified.grid)
                 for tag in ("consensus", "random", "oracle")}
    selected = verified.selected
    if selected.coordinates_path is None:
        raise SelectionValidationError("selected candidate has no verified coordinates")
    # All guards above intentionally complete before any evaluator-only loader runs.
    candidate_coordinates = {
        candidate_id: candidate_loader(candidate.coordinates_path)
        for candidate_id, candidate in verified.candidates.items()
        if candidate.coordinates_path is not None
    }
    baseline_coordinates = {tag: baseline_loader(tag, item.coordinates_path)
                            for tag, item in baselines.items()}
    labels = labels_loader()
    reference = reference_loader()
    return {
        "verified_selection": verified,
        "verified_baselines": baselines,
        "candidate_coordinates": candidate_coordinates,
        "selected_coordinates": candidate_coordinates[verified.selected_id],
        "baseline_coordinates": baseline_coordinates,
        "labels": labels,
        "reference": reference,
    }


# ---------------------------------------------------------------------------
# Pure evaluation helpers.  These functions receive labels/reference explicitly.
# ---------------------------------------------------------------------------
def _refeval():
    # Importing this module has no file-read side effects.  Keeping it local makes
    # the boundary clear: only injected arrays/dictionaries reach these helpers.
    from . import refeval
    return refeval


def _validate_metric_vectors(labels: Any, b1: Any, b2: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    b1 = np.asarray(b1, dtype=np.int64)
    b2 = np.asarray(b2, dtype=np.int64)
    if labels.ndim != 1 or b1.ndim != 1 or b2.ndim != 1 or not (len(labels) == len(b1) == len(b2)):
        raise ValueError("labels, b1, and b2 must be aligned one-dimensional arrays")
    return labels, b1, b2


def _primary_on_common(refeval: Any, d0: np.ndarray, d1: np.ndarray, labels: np.ndarray,
                       gauge: Mapping[str, Any]) -> dict[str, Any]:
    primary = refeval._primary_accuracy(d0, d1, labels, gauge)
    return {
        "accuracy": primary["accuracy"],
        "accuracy_policy": primary["policy"],
        "orientation_direct": primary["orientation_direct"],
        "orientation_swapped": primary["orientation_swapped"],
        "candidate_ties": int(primary["ties"].sum()),
        "candidate_tie_rate": float(primary["ties"].mean()) if len(labels) else None,
        "gauge": dict(gauge),
    }


def paired_r1(selected: Mapping[str, Mapping[int, np.ndarray]],
              random: Mapping[str, Mapping[int, np.ndarray]],
              oracle: Mapping[str, Mapping[int, np.ndarray]],
              reference: Mapping[str, Mapping[int, np.ndarray]], *, chromosome_index: int,
              chromosome_name: str, labels: Any, b1: Any, b2: Any, n_bins: int) -> dict[str, Any]:
    """R1 on the one selected/random/oracle/reference common record denominator.

    The caller must pass intra-chromosomal records for one chromosome only.  A
    label below zero represents cross-copy/unknown phase and is excluded; inter
    contacts must never be supplied to this haplotype readout.
    """
    refeval = _refeval()
    labels, b1, b2 = _validate_metric_vectors(labels, b1, b2)
    members = {"selected": selected, "random": random, "oracle": oracle}
    eligible = (labels >= 0) & (b1 != b2)
    in_grid = ((b1 >= 0) & (b2 >= 0) & (b1 < n_bins) & (b2 < n_bins))
    eligible &= in_grid
    vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    finite: dict[str, np.ndarray] = {}
    gauges = {}
    for tag, structs in members.items():
        d0 = refeval.ours_dists(structs, chromosome_index, 0, b1, b2, n_bins)
        d1 = refeval.ours_dists(structs, chromosome_index, 1, b1, b2, n_bins)
        vectors[tag] = (d0, d1)
        finite[tag] = np.isfinite(d0) & np.isfinite(d1)
        gauges[tag] = refeval.geometry_gauge(structs, chromosome_index, chromosome_name,
                                             reference, n_bins)
    ref_mat = refeval.ref_dists(reference, chromosome_name, 0, b1, b2, n_bins)
    ref_pat = refeval.ref_dists(reference, chromosome_name, 1, b1, b2, n_bins)
    finite["reference"] = np.isfinite(ref_mat) & np.isfinite(ref_pat)

    common = eligible.copy()
    missing = {}
    for tag in ("selected", "random", "oracle", "reference"):
        missing[tag] = int((common & ~finite[tag]).sum())
        common &= finite[tag]
    denominator = {
        "n_records": int(len(labels)),
        "n_label_or_crosscopy_excluded": int((labels < 0).sum()),
        "n_samebin_excluded": int(((labels >= 0) & (b1 == b2)).sum()),
        "n_out_of_grid_records": int(((b1 < 0) | (b2 < 0) | (b1 >= n_bins) | (b2 >= n_bins)).sum()),
        "n_out_of_grid_excluded": int(((labels >= 0) & (b1 != b2) & ~in_grid).sum()),
        "n_eligible_before_finite": int(eligible.sum()),
        "n_selected_missing_excluded": missing["selected"],
        "n_random_missing_excluded": missing["random"],
        "n_oracle_missing_excluded": missing["oracle"],
        "n_reference_missing_excluded": missing["reference"],
        "n_common": int(common.sum()),
        "n_denominator": int(common.sum()),
    }
    result: dict[str, Any] = {"applicable": False, "paired_denominator": denominator,
                              "selected": None, "random": None, "oracle": None,
                              "reference_ceiling": None, "oracle_fit_ceiling": None,
                              "delta_selected_minus_random": None,
                              "delta_oracle_minus_random": None}
    if not common.any():
        result["reason"] = "no_common_finite_labeled_non_diagonal_records"
        return result
    by_member = {}
    for tag in ("selected", "random", "oracle"):
        d0, d1 = vectors[tag]
        by_member[tag] = _primary_on_common(refeval, d0[common], d1[common], labels[common], gauges[tag])
        by_member[tag]["n_denominator"] = int(common.sum())
    reference_accuracy, reference_ties = refeval._reference_accuracy(ref_mat[common], ref_pat[common], labels[common])
    selected_accuracy = by_member["selected"]["accuracy"]
    random_accuracy = by_member["random"]["accuracy"]
    oracle_accuracy = by_member["oracle"]["accuracy"]
    result.update({
        "selected": by_member["selected"],
        "random": by_member["random"],
        "oracle": by_member["oracle"],
        "reference_ceiling": {
            "accuracy": reference_accuracy,
            "policy": "fixed_phase0_pat_phase1_mat",
            "ties": int(reference_ties.sum()),
            "tie_rate": float(reference_ties.mean()),
            "n_denominator": int(common.sum()),
        },
        "oracle_fit_ceiling": oracle_accuracy,
        "delta_selected_minus_random": (None if selected_accuracy is None or random_accuracy is None
                                          else float(selected_accuracy - random_accuracy)),
        "delta_oracle_minus_random": (None if oracle_accuracy is None or random_accuracy is None
                                        else float(oracle_accuracy - random_accuracy)),
        "applicable": selected_accuracy is not None and random_accuracy is not None and oracle_accuracy is not None,
    })
    if not result["applicable"]:
        result["reason"] = "insufficient_geometry_for_one_or_more_gauges"
    return result


def _shared_r2_contrast(refeval: Any, d0: np.ndarray, d1: np.ndarray, ref_mat: np.ndarray,
                        ref_pat: np.ndarray, gauge: Mapping[str, Any]) -> float | None:
    r0m = refeval._rho(d0, ref_mat)
    r0p = refeval._rho(d0, ref_pat)
    r1m = refeval._rho(d1, ref_mat)
    r1p = refeval._rho(d1, ref_pat)
    if not all(np.isfinite(value) for value in (r0m, r0p, r1m, r1p)):
        return None
    direct = (r0m + r1p) / 2.0
    swapped = (r0p + r1m) / 2.0
    if gauge.get("orientation") == "direct":
        return float(direct - swapped)
    if gauge.get("orientation") == "swapped":
        return float(swapped - direct)
    if gauge.get("applicable"):
        return 0.0
    return None


def paired_r2(left: Mapping[str, Mapping[int, np.ndarray]],
              right: Mapping[str, Mapping[int, np.ndarray]],
              reference: Mapping[str, Mapping[int, np.ndarray]], *, chromosome_index: int,
              chromosome_name: str, n_bins: int, left_name: str = "selected",
              right_name: str = "random") -> dict[str, Any]:
    """Compare two two-copy R2 contrasts using the required six-track common mask."""
    refeval = _refeval()
    left_gauge = refeval.geometry_gauge(left, chromosome_index, chromosome_name, reference, n_bins)
    right_gauge = refeval.geometry_gauge(right, chromosome_index, chromosome_name, reference, n_bins)
    i, j = np.triu_indices(n_bins, k=1)
    left_values = [refeval.ours_dists(left, chromosome_index, copy, i, j, n_bins) for copy in (0, 1)]
    right_values = [refeval.ours_dists(right, chromosome_index, copy, i, j, n_bins) for copy in (0, 1)]
    ref_values = [refeval.ref_dists(reference, chromosome_name, copy, i, j, n_bins) for copy in (0, 1)]
    all_values = left_values + right_values + ref_values
    common = np.ones(len(i), dtype=bool)
    for value in all_values:
        common &= np.isfinite(value)
    out = {
        "applicable": False,
        "left_name": left_name,
        "right_name": right_name,
        "mask_policy": "six-track-common-finite-non-diagonal-distance-pairs",
        "n_total_non_diagonal_pairs": int(len(i)),
        "n_common_pairs": int(common.sum()),
        "left_gauge": left_gauge,
        "right_gauge": right_gauge,
        "left": None,
        "right": None,
        "delta_right_minus_left": None,
        "delta_left_minus_right": None,
        "delta_selected_minus_random": None,
    }
    if common.sum() < refeval.MIN_RHO_PAIRS:
        out["reason"] = "insufficient_six_track_common_pairs"
        return out
    left_contrast = _shared_r2_contrast(refeval, left_values[0][common], left_values[1][common],
                                        ref_values[0][common], ref_values[1][common], left_gauge)
    right_contrast = _shared_r2_contrast(refeval, right_values[0][common], right_values[1][common],
                                         ref_values[0][common], ref_values[1][common], right_gauge)
    out.update({
        "left": left_contrast,
        "right": right_contrast,
        "delta_right_minus_left": (None if left_contrast is None or right_contrast is None
                                     else float(right_contrast - left_contrast)),
        "delta_left_minus_right": (None if left_contrast is None or right_contrast is None
                                     else float(left_contrast - right_contrast)),
        "delta_selected_minus_random": (None if left_name != "selected" or right_name != "random"
                                          or left_contrast is None or right_contrast is None
                                          else float(left_contrast - right_contrast)),
        "applicable": left_contrast is not None and right_contrast is not None,
    })
    if not out["applicable"]:
        out["reason"] = "insufficient_geometry_for_one_or_more_gauges"
    return out


def paired_r3(left: Mapping[str, Any], right: Mapping[str, Any], *, left_name: str = "selected",
              right_name: str = "random") -> dict[str, Any]:
    """Compare R3 consistency only on fragment labels resolved for both candidates."""
    if not (left.get("applicable") and right.get("applicable")):
        return {"applicable": False, "n_common_fragments": 0, "left": None, "right": None,
                "delta_right_minus_left": None, "delta_left_minus_right": None,
                "delta_selected_minus_random": None, "reason": "one_or_both_R3_unavailable"}
    left_rows = {row["grid_start_bin"]: row for row in left.get("detail", []) if row.get("label") is not None}
    right_rows = {row["grid_start_bin"]: row for row in right.get("detail", []) if row.get("label") is not None}
    common = sorted(set(left_rows) & set(right_rows))
    if not common:
        return {"applicable": False, "n_common_fragments": 0, "left": None, "right": None,
                "delta_right_minus_left": None, "delta_left_minus_right": None,
                "delta_selected_minus_random": None, "reason": "no_common_resolved_fragments"}

    def consistency(rows: Mapping[int, Mapping[str, Any]], metric: Mapping[str, Any]) -> float:
        global_label = metric.get("global_label")
        if global_label is None:
            return 0.5
        return float(np.mean([rows[key]["label"] == global_label for key in common]))

    left_value = consistency(left_rows, left)
    right_value = consistency(right_rows, right)
    return {
        "applicable": True,
        "left_name": left_name,
        "right_name": right_name,
        "n_common_fragments": int(len(common)),
        "left": left_value,
        "right": right_value,
        "delta_right_minus_left": float(right_value - left_value),
        "delta_left_minus_right": float(left_value - right_value),
        "delta_selected_minus_random": (float(left_value - right_value)
                                          if left_name == "selected" and right_name == "random" else None),
        "note": "Each candidate retains its own global-majority definition; tied majorities score 0.5.",
    }


def evaluate_chromosome(*, selected: Mapping[str, Mapping[int, np.ndarray]],
                        random: Mapping[str, Mapping[int, np.ndarray]],
                        oracle: Mapping[str, Mapping[int, np.ndarray]],
                        consensus: Mapping[str, Mapping[int, np.ndarray]],
                        reference: Mapping[str, Mapping[int, np.ndarray]], chromosome_index: int,
                        chromosome_name: str, labels: Any, b1: Any, b2: Any,
                        n_metric_bins: int) -> dict[str, Any]:
    """Evaluate one chromosome from explicitly injected labels/reference/coordinates.

    ``n_metric_bins`` is the existing OFF=3 Mb metric grid.  Full 0-based
    training-grid coverage belongs in provenance/coverage output and never leaks
    0--2 Mb records into R1/R2/R3.
    """
    refeval = _refeval()
    labels, b1, b2 = _validate_metric_vectors(labels, b1, b2)
    r1 = paired_r1(selected, random, oracle, reference, chromosome_index=chromosome_index,
                   chromosome_name=chromosome_name, labels=labels, b1=b1, b2=b2,
                   n_bins=n_metric_bins)
    r2_by_candidate = {
        "selected": refeval.r2_table(selected, chromosome_index, chromosome_name, reference, n_metric_bins),
        "random": refeval.r2_table(random, chromosome_index, chromosome_name, reference, n_metric_bins),
        "oracle": refeval.r2_table(oracle, chromosome_index, chromosome_name, reference, n_metric_bins),
        "consensus": refeval.r2_table(consensus, chromosome_index, chromosome_name, reference, n_metric_bins),
    }
    r3_by_candidate = {
        "selected": refeval.r3_fragments(selected, chromosome_index, chromosome_name, reference, n_metric_bins),
        "random": refeval.r3_fragments(random, chromosome_index, chromosome_name, reference, n_metric_bins),
        "oracle": refeval.r3_fragments(oracle, chromosome_index, chromosome_name, reference, n_metric_bins),
        "consensus": refeval.r3_fragments(consensus, chromosome_index, chromosome_name, reference, n_metric_bins),
    }
    consensus_r1 = {
        "applicable": False,
        "accuracy": None,
        "reference_ceiling": None,
        "oracle_fit_ceiling": None,
        "reason": "single_trajectory_has_no_two-copy_R1",
    }
    return {
        "chromosome": chromosome_name,
        "metric_grid": {
            "bin_size_bp": FINAL_BIN_BP,
            "offset_bp": METRIC_GRID_OFFSET_BP,
            "n_bins": int(n_metric_bins),
            "out_of_grid_records": int(((b1 < 0) | (b2 < 0) | (b1 >= n_metric_bins) | (b2 >= n_metric_bins)).sum()),
        },
        "R1": r1,
        "R2": {
            "by_candidate": r2_by_candidate,
            "selected_vs_random": paired_r2(selected, random, reference,
                                               chromosome_index=chromosome_index,
                                               chromosome_name=chromosome_name,
                                               n_bins=n_metric_bins,
                                               left_name="selected", right_name="random"),
            "oracle_vs_random": paired_r2(oracle, random, reference,
                                             chromosome_index=chromosome_index,
                                             chromosome_name=chromosome_name,
                                             n_bins=n_metric_bins,
                                             left_name="oracle", right_name="random"),
        },
        "R3": {
            "by_candidate": r3_by_candidate,
            "selected_vs_random": paired_r3(r3_by_candidate["selected"], r3_by_candidate["random"],
                                               left_name="selected", right_name="random"),
            "oracle_vs_random": paired_r3(r3_by_candidate["oracle"], r3_by_candidate["random"],
                                             left_name="oracle", right_name="random"),
        },
        "consensus": {"R1": consensus_r1, "R2": r2_by_candidate["consensus"],
                        "R3": r3_by_candidate["consensus"]},
    }


def evaluate_preregistered_candidates(*, candidates: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
                                      selected_id: str,
                                      random: Mapping[str, Mapping[int, np.ndarray]],
                                      oracle: Mapping[str, Mapping[int, np.ndarray]],
                                      consensus: Mapping[str, Mapping[int, np.ndarray]],
                                      reference: Mapping[str, Mapping[int, np.ndarray]],
                                      chromosome_index: int, chromosome_name: str, labels: Any,
                                      b1: Any, b2: Any, n_metric_bins: int) -> dict[str, Any]:
    """Evaluate every non-failed pre-registered initialization without reselection.

    ``selected_id`` is the immutable training-time count-NLL choice.  It is
    retained as provenance only; no reference-derived R1/R2/R3 result can change
    it.  Every candidate receives its own candidate/random/oracle/reference
    common R1 record denominator and paired R2/R3 control readouts.
    """
    if selected_id not in candidates:
        raise ValueError("selected_id must be present in non-failed candidate coordinates")
    if not candidates:
        raise ValueError("at least one non-failed pre-registered candidate is required")
    per_candidate = {}
    for candidate_id, candidate_structs in candidates.items():
        row = evaluate_chromosome(selected=candidate_structs, random=random, oracle=oracle,
                                  consensus=consensus, reference=reference,
                                  chromosome_index=chromosome_index,
                                  chromosome_name=chromosome_name, labels=labels, b1=b1, b2=b2,
                                  n_metric_bins=n_metric_bins)
        row["candidate_id"] = candidate_id
        row["selected_by_training_count_nll"] = candidate_id == selected_id
        per_candidate[candidate_id] = row
    return {
        "selected_id_frozen": selected_id,
        "reference_reselection_performed": False,
        "candidate_order": list(candidates),
        "candidates": per_candidate,
        "note": "All non-failed pre-registered candidates are evaluated; reference readouts never alter selected_id.",
    }


def paired_chromosome_bootstrap(deltas_by_chromosome: Mapping[str, float | None], *, seed: int = 37,
                                 n_boot: int = 10_000) -> dict[str, Any]:
    """Summarize linked chromosome deltas without calling them biological replicates."""
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    usable = [(chromosome, float(value)) for chromosome, value in deltas_by_chromosome.items()
              if value is not None and math.isfinite(float(value))]
    values = np.asarray([value for _chromosome, value in usable], dtype=float)
    per_chromosome = [{"chromosome": chromosome, "delta": value} for chromosome, value in usable]
    result = {
        "per_chromosome": per_chromosome,
        "n_chromosomes": int(len(values)),
        "wins": int((values > 0).sum()),
        "ties": int((values == 0).sum()),
        "losses": int((values < 0).sum()),
        "mean": None,
        "ci95": None,
        "interpretation": "chromosome resampling within one cell; technical/structural variation, not biological replicate uncertainty",
    }
    if not len(values):
        return result
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    result["mean"] = float(values.mean())
    result["ci95"] = [float(value) for value in np.percentile(draws, [2.5, 97.5])]
    return result


# ---------------------------------------------------------------------------
# Delivery renderers.  They consume explicitly supplied coordinates only.
# ---------------------------------------------------------------------------
def _finite_track_points(structs: Mapping[str, Mapping[int, np.ndarray]], track: str) -> tuple[np.ndarray, np.ndarray]:
    rows = structs.get(track, {})
    positions = []
    points = []
    for position, xyz in sorted(rows.items()):
        point = np.asarray(xyz, dtype=float)
        if np.isfinite(point).all():
            positions.append(int(position))
            points.append(point)
    if not points:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=float)
    return np.asarray(positions, dtype=np.int64), np.vstack(points)


def _chrom_colors(n_chromosomes: int) -> list[Any]:
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    return [cmap(index % 20) for index in range(n_chromosomes)]


def write_selected_structure_figure(path: str | os.PathLike[str],
                                    structs: Mapping[str, Mapping[int, np.ndarray]], grid: FullGrid,
                                    *, title: str = "selected 40-track structure") -> dict[str, Any]:
    """Render raw selected coordinates with a dimensionless R=1 nuclear boundary.

    No centering, rotation, Procrustes fit, RMS scaling, or other coordinate
    transformation is applied to the selected main panel.
    """
    verify_full_grid_structures(structs, grid)
    from . import figs
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    del figs
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    colors = _chrom_colors(len(grid.chromosomes))
    fig = plt.figure(figsize=(5.3, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    largest = 1.0
    for row in track_map_for_grid(grid):
        positions, points = _finite_track_points(structs, row["track"])
        if not len(points):
            continue
        largest = max(largest, float(np.abs(points).max()))
        linestyle = "-" if row["copy"] == "a" else "--"
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=colors[row["chromosome_index"]],
                linestyle=linestyle, linewidth=0.55, alpha=0.82)
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 15)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=2, cstride=2, color="0.55", linewidth=0.25, alpha=0.35)
    bound = max(1.05, largest * 1.05)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_zlim(-bound, bound)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x", labelpad=-6)
    ax.set_ylabel("y", labelpad=-6)
    ax.set_zlabel("z", labelpad=-6)
    ax.tick_params(labelsize=7, pad=-2)
    ax.view_init(elev=20, azim=-55)
    ax.set_title(title)
    chromosome_handles = [Line2D([0], [0], color=colors[index], linewidth=1.4, label=chromosome.name)
                          for index, chromosome in enumerate(grid.chromosomes)]
    copy_handles = [Line2D([0], [0], color="0.2", linestyle="-", linewidth=1.0, label="copy a"),
                    Line2D([0], [0], color="0.2", linestyle="--", linewidth=1.0, label="copy b")]
    legend = fig.legend(handles=chromosome_handles, ncol=5, title="chromosome", loc="lower center",
                        bbox_to_anchor=(0.5, 0.01), frameon=False, columnspacing=0.6,
                        handletextpad=0.25)
    fig.add_artist(legend)
    fig.legend(handles=copy_handles, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2,
               frameon=False, handletextpad=0.4)
    fig.text(0.5, 0.99, "raw R=1 unit-ball coordinates; nuclear radius is dimensionless, not physically calibrated",
             ha="center", va="top", fontsize=7)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.21, top=0.87)
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return {"figure": str(target), "coordinate_transform": "none", "nuclear_radius": 1.0,
            "nuclear_radius_units": "dimensionless"}


def write_independent_comparison_figure(path: str | os.PathLike[str],
                                        panels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Render optional comparison panels after independent center/unit-RMS scaling.

    Each panel supplies ``label``, ``structures``, and explicit ``tracks`` rows
    with ``track``, ``chromosome``, and ``copy`` keys.  This is diagnostic-only:
    it never asserts a shared reference coordinate frame.
    """
    if not panels:
        raise ValueError("comparison panels may not be empty")
    from . import figs
    import matplotlib.pyplot as plt

    del figs
    prepared = []
    chromosome_names = []
    for panel_index, panel in enumerate(panels):
        panel = _require_mapping(panel, "comparison panel %d" % panel_index)
        label = _require_text(panel.get("label"), "comparison panel label")
        structs = _require_mapping(panel.get("structures"), "comparison panel structures")
        tracks = _require_list(panel.get("tracks"), "comparison panel tracks")
        raw_rows = []
        for track_entry in tracks:
            track_entry = _require_mapping(track_entry, "comparison track")
            track = _require_text(track_entry.get("track"), "comparison track.track")
            chromosome = _require_text(track_entry.get("chromosome"), "comparison track.chromosome")
            copy = _require_text(track_entry.get("copy"), "comparison track.copy")
            positions, points = _finite_track_points(structs, track)
            if len(points):
                raw_rows.append((track, chromosome, copy, positions, points))
                chromosome_names.append(chromosome)
        if not raw_rows:
            raise ValueError("comparison panel %s has no finite coordinates" % label)
        all_points = np.vstack([points for _track, _chromosome, _copy, _positions, points in raw_rows])
        center = all_points.mean(axis=0)
        rms = float(np.sqrt(((all_points - center) ** 2).sum(axis=1).mean()))
        if not math.isfinite(rms) or rms <= 0:
            raise ValueError("comparison panel %s has non-positive RMS radius" % label)
        prepared.append((label, [(track, chromosome, copy, positions, (points - center) / rms)
                                 for track, chromosome, copy, positions, points in raw_rows],
                         {"center": center.tolist(), "rms_radius": rms, "n_beads": int(len(all_points))}))
    unique_chromosomes = list(dict.fromkeys(chromosome_names))
    colors = {name: color for name, color in zip(unique_chromosomes, _chrom_colors(len(unique_chromosomes)))}
    bound = max(1.0, max(float(np.abs(points).max()) for _label, rows, _meta in prepared
                         for _track, _chromosome, _copy, _positions, points in rows) * 1.05)
    ncols = min(3, len(prepared))
    nrows = int(math.ceil(len(prepared) / ncols))
    fig = plt.figure(figsize=(3.0 * ncols, 3.0 * nrows))
    metadata = {}
    for index, (label, rows, meta) in enumerate(prepared, start=1):
        metadata[label] = meta
        ax = fig.add_subplot(nrows, ncols, index, projection="3d")
        for _track, chromosome, copy, _positions, points in rows:
            style = "-" if copy in ("a", "mat", "single") else "--"
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=colors[chromosome],
                    linestyle=style, linewidth=0.55, alpha=0.8)
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_zlim(-bound, bound)
        ax.set_box_aspect((1, 1, 1))
        ax.tick_params(labelsize=7, pad=-2)
        ax.set_title(label)
    fig.text(0.5, 0.99, "diagnostic only: each panel independently centered and unit-RMS scaled; no shared reference frame",
             ha="center", va="top", fontsize=7)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.90, wspace=0.04, hspace=0.10)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return {"figure": str(target), "normalization": metadata,
            "coordinate_transform": "independent_center_unit_RMS_diagnostic_only"}


def _plotly_html(structs: Mapping[str, Mapping[int, np.ndarray]], grid: FullGrid, title: str) -> str | None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    colors = _chrom_colors(len(grid.chromosomes))
    figure = go.Figure()
    for row in track_map_for_grid(grid):
        positions, points = _finite_track_points(structs, row["track"])
        if not len(points):
            continue
        customdata = np.column_stack((
            np.repeat(row["chromosome"], len(points)),
            np.repeat(row["copy"], len(points)),
            ((positions - grid.origin_bp) // grid.bin_size_bp).astype(str),
            positions.astype(str),
        ))
        rgba = colors[row["chromosome_index"]]
        color = "rgb(%d,%d,%d)" % tuple(int(255 * value) for value in rgba[:3])
        figure.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="lines+markers",
            name="%s %s" % (row["chromosome"], row["copy"]),
            line={"color": color, "width": 3 if row["copy"] == "a" else 2, "dash": "solid" if row["copy"] == "a" else "dash"},
            marker={"color": color, "size": 2.0 if row["copy"] == "a" else 2.6, "symbol": "circle" if row["copy"] == "a" else "diamond"},
            customdata=customdata,
            hovertemplate="chromosome=%{customdata[0]}<br>copy=%{customdata[1]}<br>bin=%{customdata[2]}<br>position_bp=%{customdata[3]}<br>x=%{x:.4g}<br>y=%{y:.4g}<br>z=%{z:.4g}<extra></extra>",
        ))
    bound = 1.05
    all_points = [points for row in track_map_for_grid(grid)
                  for _positions, points in [_finite_track_points(structs, row["track"])] if len(points)]
    if all_points:
        bound = max(bound, float(np.abs(np.vstack(all_points)).max()) * 1.05)
    figure.update_layout(
        title=title + " (raw coordinates; R=1 is dimensionless)",
        scene={
            "xaxis": {"range": [-bound, bound], "title": "x"},
            "yaxis": {"range": [-bound, bound], "title": "y"},
            "zaxis": {"range": [-bound, bound], "title": "z"},
            "aspectmode": "cube",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 35},
    )
    return figure.to_html(full_html=True, include_plotlyjs=True, config={"responsive": True})


def _canvas_payload(structs: Mapping[str, Mapping[int, np.ndarray]], grid: FullGrid) -> dict[str, Any]:
    """Build raw-coordinate data for the dependency-free interactive viewer."""
    inventory = verify_full_grid_structures(structs, grid)
    tracks = []
    for entry in track_map_for_grid(grid):
        points = []
        for position, xyz in sorted(structs[entry["track"]].items()):
            point = np.asarray(xyz, dtype=float)
            points.append([int(position), float(point[0]), float(point[1]), float(point[2])])
        tracks.append({
            "id": entry["track"],
            "chromosome": entry["chromosome"],
            "copy": entry["copy"],
            "color": CANVAS_CHROMOSOME_COLORS[entry["chromosome_index"] % len(CANVAS_CHROMOSOME_COLORS)],
            "points": points,
        })
    return {
        "schema_version": "reconstruction-canvas-viewer-v1",
        "coordinate_space": "raw_dimensionless_R1",
        "bin_size_bp": grid.bin_size_bp,
        "origin_bp": grid.origin_bp,
        "nuclear_radius": NUCLEAR_RADIUS,
        "track_count": inventory["n_tracks"],
        "point_count": inventory["n_beads"],
        "max_radius": inventory["max_radius"],
        "tracks": tracks,
    }


def _canvas_html(structs: Mapping[str, Mapping[int, np.ndarray]], grid: FullGrid, title: str) -> str:
    """Return a self-contained canvas viewer whose projection never mutates raw coordinates."""
    payload = _canvas_payload(structs, grid)
    data_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    copy_controls = "".join(
        '<label class="toggle"><input type="checkbox" data-copy="%s" checked><span>copy %s</span></label>' %
        (html.escape(copy_name), html.escape(copy_name.upper())) for copy_name in ("a", "b"))
    chromosome_controls = "".join(
        '<label class="toggle"><input type="checkbox" data-chromosome="%s" checked><i style="background:%s"></i><span>%s</span></label>' %
        (html.escape(chromosome.name), CANVAS_CHROMOSOME_COLORS[index % len(CANVAS_CHROMOSOME_COLORS)],
         html.escape(chromosome.name))
        for index, chromosome in enumerate(grid.chromosomes))
    escaped_title = html.escape(title)
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>" + escaped_title + "</title><style>"
        "html,body{height:100%;margin:0;background:#f7f8f8;color:#152127;font:13px/1.35 Arial,sans-serif;}"
        ".viewer{min-height:100%;display:grid;grid-template-columns:minmax(0,1fr) 188px;gap:0;}"
        ".stage{position:relative;min-height:520px;background:#ffffff;border-right:1px solid #cbd3d5;}"
        "canvas{display:block;width:100%;height:100%;min-height:520px;touch-action:none;cursor:grab;}"
        "canvas:active{cursor:grabbing;}.title{position:absolute;left:14px;top:12px;font-weight:600;pointer-events:none;}"
        ".radius{position:absolute;right:14px;top:12px;color:#526166;pointer-events:none;}"
        ".hover{position:absolute;max-width:220px;padding:5px 7px;border:1px solid #9aa8ac;background:#ffffff;box-shadow:0 1px 4px #0002;pointer-events:none;font-size:12px;}"
        ".controls{padding:14px 12px;overflow:auto;background:#f7f8f8;}.controls h2{font-size:12px;margin:0 0 7px;font-weight:600;}"
        ".group{margin:0 0 16px;display:grid;gap:5px;}.toggle{display:flex;align-items:center;gap:6px;white-space:nowrap;}"
        ".toggle input{margin:0;}.toggle i{width:10px;height:10px;display:inline-block;border:1px solid #69777b;border-radius:1px;}"
        ".export{width:100%;margin:2px 0 16px;padding:5px 7px;border:1px solid #819095;border-radius:2px;background:#ffffff;color:#152127;font:inherit;cursor:pointer;text-align:left;}"
        ".export:hover{background:#eaf0f0;}"
        "@media(max-width:720px){.viewer{grid-template-columns:1fr;}.stage{border-right:0;border-bottom:1px solid #cbd3d5;min-height:460px;}canvas{min-height:460px;}.controls{display:grid;grid-template-columns:1fr 1fr;gap:14px;}.group{margin:0;}}"
        "</style></head><body><main class=\"viewer\"><section class=\"stage\"><canvas id=\"reconstruction-canvas\" aria-label=\""
        + escaped_title + "\"></canvas><div class=\"title\">" + escaped_title + "</div><div class=\"radius\">R=1</div><div id=\"hover\" class=\"hover\" hidden></div></section>"
        "<aside class=\"controls\"><section><h2>Copies</h2><div class=\"group\">" + copy_controls + "</div></section>"
        "<button id=\"export-raw\" class=\"export\" type=\"button\">Export raw coordinates</button>"
        "<section><h2>Chromosomes</h2><div class=\"group\">" + chromosome_controls + "</div></section></aside></main><script>"
        "\"use strict\";\nconst STRUCTURE_DATA=" + data_json + ";\n"
        "(function(){\n"
        "const Canvas3DMath=Object.freeze({\n"
        " rotatePoint:function(point,yaw,pitch){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);const x=cy*point[0]+sy*point[2],z=-sy*point[0]+cy*point[2];return [x,cp*point[1]-sp*z,sp*point[1]+cp*z];},\n"
        " projectPoint:function(point,width,height,zoom){const perspective=4/(4+point[2]);const scale=Math.min(width,height)*0.42*zoom*perspective;return {x:width/2+point[0]*scale,y:height/2-point[1]*scale,z:point[2]};}\n"
        "});\n"
        "const rawJSON=JSON.stringify(STRUCTURE_DATA);\n"
        "if(typeof globalThis!==\"undefined\"){globalThis.__RECONSTRUCTION_CANVAS_DATA=STRUCTURE_DATA;globalThis.__RECONSTRUCTION_CANVAS_MATH=Canvas3DMath;globalThis.ReconstructionCanvasViewer=Object.freeze({rawCoordinateJSON:function(){return rawJSON;},math:Canvas3DMath});}\n"
        "if(typeof document===\"undefined\")return;\n"
        "const canvas=document.getElementById(\"reconstruction-canvas\"),ctx=canvas.getContext(\"2d\"),hover=document.getElementById(\"hover\");\n"
        "const state={yaw:-0.72,pitch:0.34,zoom:1,copies:{a:true,b:true},chromosomes:{}};STRUCTURE_DATA.tracks.forEach(function(track){state.chromosomes[track.chromosome]=true;});\n"
        "const view={width:1,height:1,dpr:1};let projectedPoints=[],drag=null;\n"
        "function visible(track){return state.copies[track.copy]&&state.chromosomes[track.chromosome];}\n"
        "function project(raw){return Canvas3DMath.projectPoint(Canvas3DMath.rotatePoint(raw,state.yaw,state.pitch),view.width,view.height,state.zoom);}\n"
        "function curve(points,color,dash,alpha){ctx.save();ctx.strokeStyle=color;ctx.globalAlpha=alpha;ctx.lineWidth=1;ctx.setLineDash(dash);ctx.beginPath();points.forEach(function(raw,index){const p=project(raw);if(index===0)ctx.moveTo(p.x,p.y);else ctx.lineTo(p.x,p.y);});ctx.stroke();ctx.restore();}\n"
        "function sphere(){ctx.save();ctx.strokeStyle=\"#60747a\";ctx.globalAlpha=0.34;ctx.lineWidth=0.7;ctx.setLineDash([]);for(let lat=-60;lat<=60;lat+=30){const r=Math.cos(lat*Math.PI/180),y=Math.sin(lat*Math.PI/180),ring=[];for(let step=0;step<=48;step++){const t=step*Math.PI*2/48;ring.push([r*Math.cos(t),y,r*Math.sin(t)]);}curve(ring,\"#60747a\",[],0.34);}for(let lon=0;lon<180;lon+=30){const a=lon*Math.PI/180,ring=[];for(let step=0;step<=48;step++){const t=-Math.PI/2+step*Math.PI/48;ring.push([Math.cos(t)*Math.cos(a),Math.sin(t),Math.cos(t)*Math.sin(a)]);}curve(ring,\"#60747a\",[],0.34);}ctx.restore();}\n"
        "function render(){ctx.setTransform(view.dpr,0,0,view.dpr,0,0);ctx.clearRect(0,0,view.width,view.height);sphere();projectedPoints=[];STRUCTURE_DATA.tracks.forEach(function(track){if(!visible(track))return;const raw=track.points.map(function(point){return [point[1],point[2],point[3]];});curve(raw,track.color,track.copy===\"b\"?[5,4]:[],0.80);track.points.forEach(function(point){const display=project([point[1],point[2],point[3]]);projectedPoints.push({x:display.x,y:display.y,z:display.z,track:track,raw:point});});});projectedPoints.sort(function(left,right){return right.z-left.z;});projectedPoints.forEach(function(item){ctx.save();ctx.globalAlpha=0.90;ctx.fillStyle=item.track.color;ctx.beginPath();ctx.arc(item.x,item.y,item.track.copy===\"a\"?1.65:1.25,0,Math.PI*2);ctx.fill();ctx.restore();});}\n"
        "function resize(){const box=canvas.getBoundingClientRect();view.width=Math.max(1,box.width);view.height=Math.max(1,box.height);view.dpr=Math.max(1,window.devicePixelRatio||1);canvas.width=Math.round(view.width*view.dpr);canvas.height=Math.round(view.height*view.dpr);render();}\n"
        "function hideHover(){hover.hidden=true;}\n"
        "function pick(event){const box=canvas.getBoundingClientRect(),x=event.clientX-box.left,y=event.clientY-box.top;let best=null,bestDistance=100;projectedPoints.forEach(function(item){const dx=item.x-x,dy=item.y-y,distance=dx*dx+dy*dy;if(distance<bestDistance){best=item;bestDistance=distance;}});if(!best){hideHover();return;}const bin=(best.raw[0]-STRUCTURE_DATA.origin_bp)/STRUCTURE_DATA.bin_size_bp;hover.textContent=best.track.chromosome+\" | copy \"+best.track.copy.toUpperCase()+\" | bin \"+bin+\" | start \"+best.raw[0].toLocaleString();hover.style.left=Math.min(view.width-224,Math.max(8,x+12))+\"px\";hover.style.top=Math.min(view.height-34,Math.max(8,y+12))+\"px\";hover.hidden=false;}\n"
        "canvas.addEventListener(\"pointerdown\",function(event){drag={x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);hideHover();});\n"
        "canvas.addEventListener(\"pointermove\",function(event){if(drag){state.yaw+=(event.clientX-drag.x)*0.01;state.pitch=Math.max(-1.45,Math.min(1.45,state.pitch+(event.clientY-drag.y)*0.01));drag={x:event.clientX,y:event.clientY};render();}else pick(event);});\n"
        "canvas.addEventListener(\"pointerup\",function(event){drag=null;canvas.releasePointerCapture(event.pointerId);});canvas.addEventListener(\"pointercancel\",function(){drag=null;});canvas.addEventListener(\"pointerleave\",function(){if(!drag)hideHover();});\n"
        "canvas.addEventListener(\"wheel\",function(event){event.preventDefault();state.zoom=Math.max(0.35,Math.min(3,state.zoom*Math.exp(-event.deltaY*0.001)));render();},{passive:false});\n"
        "document.querySelectorAll(\"input[data-copy]\").forEach(function(input){input.addEventListener(\"change\",function(){state.copies[input.dataset.copy]=input.checked;hideHover();render();});});\n"
        "document.querySelectorAll(\"input[data-chromosome]\").forEach(function(input){input.addEventListener(\"change\",function(){state.chromosomes[input.dataset.chromosome]=input.checked;hideHover();render();});});\n"
        "document.getElementById(\"export-raw\").addEventListener(\"click\",function(){const blob=new Blob([rawJSON],{type:\"application/json\"}),url=URL.createObjectURL(blob),link=document.createElement(\"a\");link.href=url;link.download=\"selected_raw_coordinates.json\";document.body.appendChild(link);link.click();link.remove();URL.revokeObjectURL(url);});\n"
        "window.addEventListener(\"resize\",resize);resize();\n"
        "})();\n</script></body></html>"
    )


def write_offline_structure_html(path: str | os.PathLike[str],
                                structs: Mapping[str, Mapping[int, np.ndarray]], grid: FullGrid,
                                *, title: str, fallback_png: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Write self-contained Plotly HTML when available, else a canvas 3D viewer."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    verify_full_grid_structures(structs, grid)
    contents = _plotly_html(structs, grid, title)
    if contents is not None:
        if "cdn.plot.ly" in contents.lower():
            raise RuntimeError("offline Plotly renderer unexpectedly requested a CDN")
        target.write_text(contents)
        return {"html": str(target), "mode": "plotly_embedded", "external_cdn": False}
    # The formal static PNG is produced separately.  This HTML remains interactive
    # even when Plotly is absent, so it must not degrade to an image wrapper.
    del fallback_png
    canvas = _canvas_html(structs, grid, title)
    if "cdn." in canvas.lower() or "http://" in canvas.lower() or "https://" in canvas.lower():
        raise RuntimeError("canvas viewer unexpectedly contains an external resource")
    target.write_text(canvas)
    payload = _canvas_payload(structs, grid)
    return {"html": str(target), "mode": "vanilla_canvas_embedded", "external_cdn": False,
            "n_tracks": payload["track_count"], "n_points": payload["point_count"],
            "coordinate_transform": "display_projection_only_raw_export_unchanged"}


def write_track_coverage_table(path: str | os.PathLike[str],
                               structs: Mapping[str, Mapping[int, np.ndarray]], grid: FullGrid) -> dict[str, Any]:
    """Write raw full-training-grid track counts and coverage without metric truncation."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    expected = grid.expected_positions_by_track()
    for entry in track_map_for_grid(grid):
        positions, points = _finite_track_points(structs, entry["track"])
        n_expected = len(expected[entry["track"]])
        rows.append({**entry, "n_expected_full_grid": n_expected, "n_finite_coordinates": int(len(points)),
                     "coverage": float(len(points) / n_expected) if n_expected else None})
    columns = ["track", "chromosome_index", "chromosome", "copy", "n_expected_full_grid",
               "n_finite_coordinates", "coverage"]
    with target.open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[column]) for column in columns) + "\n")
    return {"table": str(target), "n_tracks": len(rows), "n_expected_full_grid": grid.n_physical_beads,
            "n_finite_coordinates": int(sum(row["n_finite_coordinates"] for row in rows))}


def write_fragment_stripes(path: str | os.PathLike[str], table_path: str | os.PathLike[str],
                           r3_by_chromosome: Mapping[str, Mapping[str, Any]],
                           chromosome_names: Sequence[str]) -> dict[str, Any]:
    """Render R3 labels, separating chromosome-end padding from invalid fragments."""
    from . import figs
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    del figs
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table_target = Path(table_path)
    table_target.parent.mkdir(parents=True, exist_ok=True)
    details = [list(r3_by_chromosome.get(name, {}).get("detail", [])) for name in chromosome_names]
    n_columns = max((len(items) for items in details), default=1)
    values = np.full((len(chromosome_names), n_columns), np.nan)
    padding = np.zeros((len(chromosome_names), n_columns), dtype=bool)
    lengths = r3_by_chromosome.get("__chromosome_lengths__", {})
    if not isinstance(lengths, Mapping):
        lengths = {}
    rows = []
    for row_index, (chromosome, items) in enumerate(zip(chromosome_names, details)):
        chromosome_length = lengths.get(chromosome)
        for column in range(n_columns):
            item = items[column] if column < len(items) else None
            start_bp = (item or {}).get("frag_start_bp")
            if start_bp is None:
                start_bp = METRIC_GRID_OFFSET_BP + column * FRAGMENT_BP
            is_padding = chromosome_length is not None and int(start_bp) >= int(chromosome_length)
            padding[row_index, column] = is_padding
            if item is not None:
                if item.get("status") == "labelled" and not is_padding:
                    values[row_index, column] = float(item["label"])
                rows.append({"chromosome": chromosome, "fragment_index": column,
                             "grid_start_bin": item.get("grid_start_bin"),
                             "frag_start_bp": item.get("frag_start_bp"),
                             "status": item.get("status"), "label": item.get("label"),
                             "n_common_pairs": item.get("n")})
    cmap = plt.get_cmap("coolwarm_r").copy()
    cmap.set_bad("0.72")
    fig, ax = plt.subplots(figsize=(5.8, max(3.0, 0.18 * len(chromosome_names) + 0.9)))
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    # Keep valid-but-unscored fragments grey, while cells outside the chromosome are white.
    white = np.ma.masked_where(~padding, np.ones_like(values))
    ax.imshow(white, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(["white"]), vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(chromosome_names)))
    ax.set_yticklabels(chromosome_names)
    ax.set_xticks(np.arange(n_columns))
    ax.set_xticklabels([str(index * 20) for index in range(n_columns)])
    ax.set_xlabel("Offset from 3 Mb evaluation origin (Mb)")
    ax.set_ylabel("chromosome")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, ticks=(0, 1))
    colorbar.ax.set_yticklabels(["local a=mat", "local a=pat"])
    fig.text(0.5, 0.005,
             "white = beyond chromosome end (padding); grey = valid fragment missing, insufficient, or geometry-tied; not scored",
             ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(bottom=0.20, left=0.12, right=0.92, top=0.98)
    fig.savefig(target, dpi=300)
    plt.close(fig)
    columns = ["chromosome", "fragment_index", "grid_start_bin", "frag_start_bp", "status", "label",
               "n_common_pairs"]
    with table_target.open("w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[column]) for column in columns) + "\n")
    return {"figure": str(target), "table": str(table_target), "n_chromosomes": len(chromosome_names),
            "n_fragments_max": n_columns, "missing_and_tied_color": "grey", "padding_color": "white",
            "padding_policy": "frag_start_bp >= chromosome.length_bp",
            "x_axis": "Offset from 3 Mb evaluation origin (Mb)"}


def write_optimizer_trace(path: str | os.PathLike[str], trace: Sequence[Mapping[str, Any]] | None) -> dict[str, Any] | None:
    """Plot supplied likelihood/optimizer traces; no log format is assumed or parsed."""
    if not trace:
        return None
    from . import figs
    import matplotlib.pyplot as plt

    del figs
    rows = [dict(_require_mapping(row, "optimizer trace row")) for row in trace]
    if any("iteration" not in row for row in rows):
        raise ValueError("optimizer trace rows require iteration")
    iterations = np.asarray([_finite_number(row["iteration"], "optimizer trace iteration") for row in rows])
    keys = sorted({key for row in rows for key, value in row.items()
                   if key != "iteration" and isinstance(value, (int, float)) and not isinstance(value, bool)})
    if not keys:
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for key in keys:
        values = np.asarray([float(row[key]) if key in row and math.isfinite(float(row[key])) else np.nan
                             for row in rows])
        ax.plot(iterations, values, label=key)
    ax.set_xlabel("iteration")
    ax.set_ylabel("reported value")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False)
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return {"figure": str(target), "series": keys}


def renderer_provenance() -> dict[str, str]:
    """Record evaluator-renderer code separately from immutable training code."""
    path = Path(__file__).resolve()
    return {"role": "evaluation_renderer", "path": str(path), "sha256": sha256_file(path),
            "hash_source": "file_bytes"}


def write_selected_structure_delivery(output_dir: str | os.PathLike[str],
                                      structs: Mapping[str, Mapping[int, np.ndarray]], grid: FullGrid,
                                      *, r3_by_chromosome: Mapping[str, Mapping[str, Any]] | None = None,
                                      optimizer_trace: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Write selected-only delivery assets without loading any comparison coordinate."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    png = write_selected_structure_figure(root / "selected_structure_3d.png", structs, grid)
    offline = write_offline_structure_html(root / "selected_structure_3d.html", structs, grid,
                                           title="selected 40-track structure")
    coverage = write_track_coverage_table(root / "track_coverage.tsv", structs, grid)
    result = {"selected_3d": png, "interactive_3d": offline, "track_coverage": coverage,
              "renderer_provenance": renderer_provenance(),
              "full_training_grid": {"origin_bp": grid.origin_bp, "bin_size_bp": grid.bin_size_bp,
                                      "loci": grid.n_loci, "physical_beads": grid.n_physical_beads},
              "metric_grid": {"offset_bp": METRIC_GRID_OFFSET_BP, "bin_size_bp": FINAL_BIN_BP,
                              "note": "R1/R2/R3 use OFF=3 Mb; full training coordinates remain unmodified."}}
    if r3_by_chromosome is not None:
        result["fragment_stripes"] = write_fragment_stripes(
            root / "fragment_consistency_stripes.png", root / "fragment_consistency_stripes.tsv",
            r3_by_chromosome, [chromosome.name for chromosome in grid.chromosomes])
    trace_result = write_optimizer_trace(root / "optimizer_trace.png", optimizer_trace)
    if trace_result is not None:
        result["optimizer_trace"] = trace_result
    return result


def render_final_readme(verified: VerifiedSelection, report: Mapping[str, Any]) -> str:
    """Render the Chinese final README from already-computed metrics/provenance.

    The caller supplies conclusions explicitly; this helper never infers L2 from
    low wall counts or a fragment majority alone.
    """
    if report.get("synthetic_fixture"):
        raise ValueError("synthetic fixture output must not be rendered as a P9016 final README")
    selection = verified.document
    contract = selection["objective_contract"]
    candidates = []
    for candidate_id in selection["preregistered_candidate_ids"]:
        candidate = verified.candidates[candidate_id]
        score = "n/a" if candidate.count_nll_per_record is None else "%.12g" % candidate.count_nll_per_record
        candidates.append("- `%s`: terminal=%s, count_nll_per_record=%s%s" % (
            candidate_id, candidate.terminal_status, score,
            "; reason=" + candidate.failure_reason if candidate.failure_reason else ""))
    l2 = _require_text(report.get("l2_conclusion"), "report.l2_conclusion")
    delivery = _require_mapping(report.get("delivery"), "report.delivery")
    metric_summary = _require_mapping(report.get("metric_summary"), "report.metric_summary")
    lines = [
        "# P9016 单细胞 20 染色体/40 轨迹重构",
        "",
        "## 训练与选择",
        "",
        "- 生物学样本数为 1；训练使用全部 1,703,888 条原始接触（1,135,454 intra + 568,434 inter）。",
        "- SNP-free 输入 SHA256：`%s`。最终坐标为 1 Mb、从 0 开始的完整训练网格：%d loci / %d physical beads，所有轨迹置于同一无量纲 R=1 核球坐标系。R=1 没有物理长度标定。" % (
            selection["cohort"]["snpfree_sha256"], verified.grid.n_loci, verified.grid.n_physical_beads),
        "- 候选选择只按 `%s`，方向 `%s`；它是完整 conditional count objective（含候选不变的 diagonal 层），不含 prior，也不读取 phase/reference。每 record tie 容差为 `%.1e`，按预注册顺序打平。" % (
            contract["criterion"], contract["direction"], contract["tie_tolerance_per_record"]),
        "- 所有候选使用同一 1 Mb 模型、prior 配置和参数维数；prior 各 component 仅作物理/数值诊断报告，不伪装为 count likelihood。",
        "- metrics 会完整列出每个 nonfailed 预注册初始化变体的 R1/R2/R3；`selected_id` 始终是训练时冻结的 count-NLL 选择，reference 读数不会重选候选。",
        "- `p` 是 cis 四项 kernel 混合的 nuisance 系数，不等于原始 contact 的真实 same-copy 比例，也不是 phasing accuracy；经过几何积分/曝光后的 record fraction 同样不能由 `p` 解读。",
        "- 选择的候选：`%s`。" % verified.selected_id,
        *candidates,
        "",
        "## 评估口径",
        "",
        "- R1 只使用 intra、same-copy 已定相、非同 bin 的记录。selected、random、oracle-fit 和 reference ceiling 均在每条染色体同一个 `paired_denominator` 上计算；报告 reference 与 oracle-fit 两个 ceiling、missing 与 tie 计数。inter 不计入 haplotype 准确率。",
        "- 实际 evaluator 在解析 phase/reference 前复核 raw pairs 固定 SHA、reference 3DG 实际 SHA 与 017 provenance 声明，并逐 record 对齐 raw、SNP-free contacts 和 labels 的完整 M；SNP-free 训练输入 hash 不替代这项审计。",
        "- R2 的单候选表使用四轨共同 finite 非对角 pair；paired selected/random 使用六轨共同 mask。R3 保留 20 Mb 多数标签，分别报告 total/applicable/tied/insufficient，并只在共同 fragment 上比较。",
        "- 20 条染色体是同一细胞中的链接结构测量。per-chromosome wins 和 bootstrap CI 仅描述细胞内技术/结构变异，不是生物学重复。single consensus 不是 two-copy contrast，R1/R2/R3 标为 n/a，同时保留其单轨结构相关基线。",
        "- full training grid 与 OFF=3 Mb metrics grid 分开报告；0--2 Mb 坐标保留在结构交付中，但绝不混入 R1/R2/R3，out-grid 分母单列。",
        "",
        "## 结论与交付",
        "",
        "- L2 结论：%s" % l2,
        "- 低 wall 数或 majority > 0.5 本身不构成 L2 恢复证据；若 L2 未恢复，仍完整交付所选 40 轨迹候选结构。",
        "- 精确坐标、track map 和图件路径：`%s`。" % _require_text(delivery.get("path"), "report.delivery.path"),
        "- 选择与评估的冻结 protocol/config/code/data 哈希、training-code manifest 的逐成员哈希、所有 attempts 和共同分母明细见本 formal run 的 `selection.json` 与 metrics 文件；evaluation renderer code 另列，不混入训练 code snapshot。",
        "",
        "## 指标摘要",
        "",
        "```json",
        json.dumps(metric_summary, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_final_readme(path: str | os.PathLike[str], verified: VerifiedSelection,
                       report: Mapping[str, Any]) -> str:
    """Write the final Chinese README only from a verified, non-synthetic report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_final_readme(verified, report))
    return str(target)
