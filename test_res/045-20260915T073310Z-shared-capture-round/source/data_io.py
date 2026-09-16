"""Run-local serialization helpers for full-grid AggregatedContacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pr.contact_model import AggregatedContacts


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype, order="C") if dtype else np.asarray(values, order="C")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _scalar(payload: Any, key: str, cast: Any = float) -> Any:
    value = payload[key]
    if np.asarray(value).ndim == 0:
        return cast(np.asarray(value).item())
    return cast(value)


def save_aggregate(path: str | Path, data: AggregatedContacts) -> dict[str, Any]:
    """写出 full grid，不丢失零 pair 或 group metadata。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        chromosome_names=np.asarray(data.chromosome_names),
        chromosome_lengths=np.asarray(data.chromosome_lengths, dtype=np.int64),
        bin_size=np.asarray(data.bin_size, dtype=np.int64),
        n_bins=np.asarray(data.n_bins, dtype=np.int64),
        offsets=np.asarray(data.offsets, dtype=np.int64),
        locus_chromosome=np.asarray(data.locus_chromosome, dtype=np.int32),
        locus_bin=np.asarray(data.locus_bin, dtype=np.int64),
        endpoint_counts=np.asarray(data.endpoint_counts),
        exposure=np.asarray(data.exposure, dtype=np.float64),
        pair_i=np.asarray(data.pair_i, dtype=np.int32),
        pair_j=np.asarray(data.pair_j, dtype=np.int32),
        cis_pair=np.asarray(data.cis_pair, dtype=np.bool_),
        counts=np.asarray(data.counts),
        diag_counts=np.asarray(data.diag_counts),
        raw_records=np.asarray(data.raw_records, dtype=np.float64),
        raw_same_bin=np.asarray(data.raw_same_bin, dtype=np.float64),
        raw_cis_offdiag=np.asarray(data.raw_cis_offdiag, dtype=np.float64),
        raw_inter=np.asarray(data.raw_inter, dtype=np.float64),
        conditional_factorial_constant=np.asarray(data.conditional_factorial_constant, dtype=np.float64),
        diag_factorial_sum=np.asarray(data.diag_factorial_sum, dtype=np.float64),
        count_mode=np.asarray(data.count_mode),
        exposure_mode=np.asarray(data.exposure_mode),
        expected_rtol=np.asarray(data.expected_rtol, dtype=np.float64),
        expected_atol=np.asarray(data.expected_atol, dtype=np.float64),
    )
    return {"path": str(path), "sha256": sha256_file(path), "bytes": int(path.stat().st_size),
            "n_loci": int(data.n_loci), "n_pairs": int(data.n_pairs), "budget": data.budget()}


def load_aggregate(path: str | Path) -> AggregatedContacts:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        names = tuple(str(value) for value in payload["chromosome_names"].tolist())
        data = AggregatedContacts(
            chromosome_names=names,
            chromosome_lengths=np.asarray(payload["chromosome_lengths"], dtype=np.int64).copy(),
            bin_size=int(np.asarray(payload["bin_size"]).item()),
            n_bins=np.asarray(payload["n_bins"], dtype=np.int64).copy(),
            offsets=np.asarray(payload["offsets"], dtype=np.int64).copy(),
            locus_chromosome=np.asarray(payload["locus_chromosome"], dtype=np.int32).copy(),
            locus_bin=np.asarray(payload["locus_bin"], dtype=np.int64).copy(),
            endpoint_counts=np.asarray(payload["endpoint_counts"]).copy(),
            exposure=np.asarray(payload["exposure"], dtype=np.float64).copy(),
            pair_i=np.asarray(payload["pair_i"], dtype=np.int32).copy(),
            pair_j=np.asarray(payload["pair_j"], dtype=np.int32).copy(),
            cis_pair=np.asarray(payload["cis_pair"], dtype=np.bool_).copy(),
            counts=np.asarray(payload["counts"]).copy(),
            diag_counts=np.asarray(payload["diag_counts"]).copy(),
            raw_records=float(np.asarray(payload["raw_records"]).item()),
            raw_same_bin=float(np.asarray(payload["raw_same_bin"]).item()),
            raw_cis_offdiag=float(np.asarray(payload["raw_cis_offdiag"]).item()),
            raw_inter=float(np.asarray(payload["raw_inter"]).item()),
            conditional_factorial_constant=float(np.asarray(payload["conditional_factorial_constant"]).item()),
            diag_factorial_sum=float(np.asarray(payload["diag_factorial_sum"]).item()),
            count_mode=str(np.asarray(payload["count_mode"]).item()),
            exposure_mode=str(np.asarray(payload["exposure_mode"]).item()),
            expected_rtol=float(np.asarray(payload["expected_rtol"]).item()),
            expected_atol=float(np.asarray(payload["expected_atol"]).item()),
        )
    data.assert_consistent()
    return data


def save_start(path: str | Path, coordinates: np.ndarray, raw_y: np.ndarray,
               p_init: float, metadata: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    raw_y = np.asarray(raw_y, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[0] != 2 or coordinates.shape[2] != 3:
        raise ValueError("coordinates must have shape (2,n,3)")
    if raw_y.shape != coordinates.shape:
        raise ValueError("raw_y shape mismatch")
    np.savez_compressed(path, coordinates=coordinates, raw_y=raw_y,
                        p_init=np.asarray(float(p_init), dtype=np.float64),
                        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, ensure_ascii=False)))
    return {"path": str(path), "sha256": sha256_file(path),
            "coordinate_sha256": array_sha256(coordinates, "<f8"),
            "raw_y_sha256": array_sha256(raw_y, "<f8"),
            "coordinate_shape": list(coordinates.shape), "p_init": float(p_init),
            "metadata": metadata}


def load_start(path: str | Path) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
        raw_y = np.asarray(payload["raw_y"], dtype=np.float64).copy()
        p_init = float(np.asarray(payload["p_init"]).item())
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    return coordinates, raw_y, p_init, metadata
