"""Post-020 allele model variants with isolated coordinate parameterizations.

The frozen V1 objective remains the default implementation.  This module adds
only the five preregistered variant descriptions, the uniform-exposure view used
by C3, and thin objective subclasses for the two C2 coordinate domains.

No native FDG, phase-bearing input, or reference artifact is imported here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from .contact_model import (
    AggregatedContacts,
    JointObjective,
    ObjectiveWeights,
    P_FLOOR,
    P_PRIOR_STRENGTH,
    assert_inside_unit_ball,
    p_from_q,
    sphere_forward,
    sphere_inverse,
)


MODEL_IDS = ("C0", "C1", "C2-map", "C2-free", "C3")
P_INIT = 0.75
IDENTITY_RADIUS = 0.90
IDENTITY_WIDTH = 0.10


@dataclass(frozen=True)
class AlleleModelSpec:
    """One preregistered model variant and its single intended change."""

    model_id: str
    description: str
    coordinate_parameterization: str
    physical_domain: str
    exposure_change: str
    weights: ObjectiveWeights

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "description": self.description,
            "coordinate_parameterization": self.coordinate_parameterization,
            "physical_domain": self.physical_domain,
            "exposure_change": self.exposure_change,
            "objective_weights": {
                "count": float(self.weights.count),
                "p_prior": float(self.weights.p_prior),
                "bond": float(self.weights.bond),
                "repulsion": float(self.weights.repulsion),
                "bend": float(self.weights.bend),
            },
            "p_init": P_INIT,
            "p_floor": P_FLOOR,
            "p_prior_strength": P_PRIOR_STRENGTH,
            "finite_kernel": {
                "epsilon": 1e-6,
                "radius": "2*l0",
            },
            "repulsion_threshold": "0.7*l0",
        }


def _check_model_id(model_id: str) -> str:
    model_id = str(model_id)
    if model_id not in MODEL_IDS:
        raise ValueError("unknown allele model %r; expected one of %s" %
                         (model_id, ", ".join(MODEL_IDS)))
    return model_id


def model_spec(model_id: str) -> AlleleModelSpec:
    """Return a fresh immutable description of one registered variant."""
    model_id = _check_model_id(model_id)
    base = ObjectiveWeights()
    if model_id == "C0":
        return AlleleModelSpec(
            model_id="C0",
            description="V1 baseline on the paired 1 Mb start",
            coordinate_parameterization="sphere_forward_unit_ball",
            physical_domain="strict_unit_ball",
            exposure_change="none_preserve_input",
            weights=base,
        )
    if model_id == "C1":
        return AlleleModelSpec(
            model_id="C1",
            description="V1 objective with bend weight set to zero",
            coordinate_parameterization="sphere_forward_unit_ball",
            physical_domain="strict_unit_ball",
            exposure_change="none_preserve_input",
            weights=replace(base, bend=0.0),
        )
    if model_id == "C2-map":
        return AlleleModelSpec(
            model_id="C2-map",
            description="Smooth identity-interior unit-ball parameterization",
            coordinate_parameterization="identity_interior_smooth_ball",
            physical_domain="strict_unit_ball",
            exposure_change="none_preserve_input",
            weights=base,
        )
    if model_id == "C2-free":
        return AlleleModelSpec(
            model_id="C2-free",
            description="Direct physical coordinates without a hard ball",
            coordinate_parameterization="direct_physical_unbounded",
            physical_domain="finite_unbounded",
            exposure_change="none_preserve_input",
            weights=base,
        )
    return AlleleModelSpec(
        model_id="C3",
        description="V1 objective with uniform full-grid exposure",
        coordinate_parameterization="sphere_forward_unit_ball",
        physical_domain="strict_unit_ball",
        exposure_change="uniform_full_grid",
        weights=base,
    )


def data_for_model(data: AggregatedContacts, model_id: str) -> AggregatedContacts:
    """Return the data view for a variant without changing counts or eligibility.

    C3 replaces only the exposure vector and its explicit mode label.  The
    endpoint audit, complete eligible pair grid, diagonal nuisance counts, and
    all conservation fields are retained byte-for-byte where applicable.
    """
    spec = model_spec(model_id)
    data.assert_consistent()
    if spec.exposure_change != "uniform_full_grid":
        return data
    exposure = np.ones(data.n_loci, dtype=np.float64)
    result = replace(data, exposure=exposure, exposure_mode="uniform")
    result.assert_consistent()
    if not np.array_equal(result.endpoint_counts, data.endpoint_counts):
        raise AssertionError("C3 exposure view changed endpoint audit counts")
    if not np.array_equal(result.pair_i, data.pair_i) or not np.array_equal(result.pair_j, data.pair_j):
        raise AssertionError("C3 exposure view changed eligible pair indexing")
    if not np.array_equal(result.cis_pair, data.cis_pair):
        raise AssertionError("C3 exposure view changed cis/inter pair classes")
    if not np.array_equal(result.counts, data.counts) or not np.array_equal(result.diag_counts, data.diag_counts):
        raise AssertionError("C3 exposure view changed count or diagonal layers")
    return result


def _as_xyz(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 3:
        raise ValueError("%s must have a final dimension of three" % name)
    if not np.all(np.isfinite(array)):
        raise ValueError("%s must be finite" % name)
    return array


def identity_interior_forward(y: np.ndarray) -> np.ndarray:
    """Map R^3 to the open unit ball, identity through radius 0.90.

    The outer radial map is C2 at 0.90 and approaches, but never intentionally
    crosses, radius one for finite ordinary floating-point inputs.
    """
    values = _as_xyz(y, "y")
    radius = np.linalg.norm(values, axis=-1)
    result = values.copy()
    outer = radius > IDENTITY_RADIUS
    if np.any(outer):
        t = (radius[outer] - IDENTITY_RADIUS) / IDENTITY_WIDTH
        hyp = np.hypot(1.0, t)
        s = IDENTITY_RADIUS + IDENTITY_WIDTH * (t / hyp)
        s = np.minimum(s, np.nextafter(1.0, 0.0))
        result[outer] = values[outer] * (s / radius[outer])[:, None]
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("identity-interior map produced nonfinite coordinates")
    return result


def identity_interior_inverse(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`identity_interior_forward` on the open unit ball."""
    values = _as_xyz(x, "x")
    radius = np.linalg.norm(values, axis=-1)
    if np.any(radius >= 1.0):
        raise ValueError("identity-interior inverse requires strict unit-ball coordinates")
    result = values.copy()
    outer = radius > IDENTITY_RADIUS
    if np.any(outer):
        u = (radius[outer] - IDENTITY_RADIUS) / IDENTITY_WIDTH
        denominator = np.sqrt((1.0 - u) * (1.0 + u))
        if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
            raise ValueError("identity-interior inverse is numerically saturated")
        t = u / denominator
        raw_radius = IDENTITY_RADIUS + IDENTITY_WIDTH * t
        result[outer] = values[outer] * (raw_radius / radius[outer])[:, None]
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("identity-interior inverse produced nonfinite coordinates")
    return result


def identity_interior_pullback(y: np.ndarray, gradient_x: np.ndarray) -> np.ndarray:
    """Return the transpose Jacobian product for the identity-interior map."""
    values = _as_xyz(y, "y")
    gradient = _as_xyz(gradient_x, "gradient_x")
    if values.shape != gradient.shape:
        raise ValueError("y and gradient_x must have matching shapes")
    radius = np.linalg.norm(values, axis=-1)
    result = gradient.copy()
    outer = radius > IDENTITY_RADIUS
    if np.any(outer):
        t = (radius[outer] - IDENTITY_RADIUS) / IDENTITY_WIDTH
        hyp = np.hypot(1.0, t)
        scale = (IDENTITY_RADIUS + IDENTITY_WIDTH * t / hyp) / radius[outer]
        radial_derivative = 1.0 / (hyp * hyp * hyp)
        unit = values[outer] / radius[outer, None]
        radial_component = np.sum(unit * gradient[outer], axis=1)
        result[outer] = (
            scale[:, None] * gradient[outer]
            + (radial_derivative - scale)[:, None] * unit * radial_component[:, None]
        )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("identity-interior pullback produced nonfinite gradient")
    return result


def validate_strict_unit_ball(coordinates: np.ndarray) -> None:
    """Validate the bounded C2-map physical domain."""
    assert_inside_unit_ball(_as_xyz(coordinates, "coordinates"))


def validate_finite_coordinates(coordinates: np.ndarray) -> None:
    """Validate the direct-x physical domain without imposing a radius."""
    _as_xyz(coordinates, "coordinates")


class _MappedJointObjective(JointObjective):
    """Small coordinate-map adapter around the unchanged V1 component code."""

    map_name = "abstract"
    physical_domain = "abstract"

    def __init__(self, data: AggregatedContacts, weights: ObjectiveWeights | None = None,
                 block_size: int = 65_536, repulsion_block_size: int | None = None):
        super().__init__(data, weights=weights, block_size=block_size,
                         repulsion_block_size=repulsion_block_size)
        self._objective_eval_count = 0
        self._nonidentity_map_eval_count = 0
        self._nonidentity_bead_eval_count = 0
        self._max_raw_radius = 0.0
        self._max_physical_radius = 0.0

    def _forward(self, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _inverse(self, coordinates: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _pullback(self, raw: np.ndarray, gradient_x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _validate(self, coordinates: np.ndarray) -> None:
        raise NotImplementedError

    def _map_for_objective(self, raw: np.ndarray) -> np.ndarray:
        raw = _as_xyz(raw, "raw coordinates")
        raw_radius = np.linalg.norm(raw, axis=-1)
        self._objective_eval_count += 1
        self._max_raw_radius = max(self._max_raw_radius, float(raw_radius.max(initial=0.0)))
        if self.map_name == "identity_interior_smooth_ball":
            nonidentity = raw_radius > IDENTITY_RADIUS
            self._nonidentity_map_eval_count += int(np.any(nonidentity))
            self._nonidentity_bead_eval_count += int(nonidentity.sum())
        coordinates = self._forward(raw)
        physical_radius = np.linalg.norm(coordinates, axis=-1)
        self._max_physical_radius = max(
            self._max_physical_radius, float(physical_radius.max(initial=0.0)))
        return coordinates

    def _map_without_count(self, raw: np.ndarray) -> np.ndarray:
        return self._forward(_as_xyz(raw, "raw coordinates"))

    def raw_from_physical(self, coordinates: np.ndarray) -> np.ndarray:
        coordinates = _as_xyz(coordinates, "physical coordinates")
        self._validate(coordinates)
        return self._inverse(coordinates)

    def validate_physical_coordinates(self, coordinates: np.ndarray) -> None:
        self._validate(coordinates)

    def coordinates_and_p(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        raw, q = self.unpack(theta)
        coordinates = self._map_without_count(raw)
        self._validate(coordinates)
        p, _ = p_from_q(q)
        return coordinates, p

    def count_components_for_coordinates(self, x: np.ndarray, p: float) -> dict:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (2, self.data.n_loci, 3):
            raise ValueError("x must have shape (2, n_loci, 3)")
        self._validate(x)
        if not math.isfinite(float(p)) or not P_FLOOR <= float(p) <= 1.0 - P_FLOOR:
            raise ValueError("fixed p must lie in the bounded V1 interval")
        components, _, _ = self._count_nll_and_gradient(x, float(p), need_gradient=False)
        return components

    def evaluate(self, theta: np.ndarray, need_gradient: bool = True):
        """Evaluate unchanged V1 terms after an explicit physical map."""
        raw, q = self.unpack(theta)
        coordinates = self._map_for_objective(raw)
        self._validate(coordinates)
        p, derivative_q = p_from_q(q)
        count_components, gradient_x, gradient_p = self._count_nll_and_gradient(
            coordinates, p, need_gradient)
        bond, bond_gradient = self._bond_and_gradient(coordinates)
        repulsion, repulsion_gradient = self._repulsion_and_gradient(coordinates)
        bend, bend_gradient = self._bend_and_gradient(coordinates)
        p_prior = -P_PRIOR_STRENGTH * math.log(p * (1.0 - p))
        p_prior_derivative_p = P_PRIOR_STRENGTH * (1.0 / (1.0 - p) - 1.0 / p)
        components = dict(count_components)
        components.update({
            "p": float(p),
            "p_prior": float(p_prior),
            "bond": float(bond),
            "repulsion": float(repulsion),
            "bend": float(bend),
        })
        total = (
            self.weights.count * components["count_nll_normalized"]
            + self.weights.p_prior * p_prior
            + self.weights.bond * bond
            + self.weights.repulsion * repulsion
            + self.weights.bend * bend
        )
        components["total"] = float(total)
        if not need_gradient:
            return float(total), None, components
        combined_x = (
            self.weights.count * gradient_x
            + self.weights.bond * bond_gradient
            + self.weights.repulsion * repulsion_gradient
            + self.weights.bend * bend_gradient
        )
        gradient_raw = self._pullback(raw, combined_x)
        gradient_q = derivative_q * (
            self.weights.count * gradient_p
            + self.weights.p_prior * p_prior_derivative_p
        )
        gradient = np.concatenate((gradient_raw.ravel(), np.array([gradient_q], dtype=np.float64)))
        if not np.all(np.isfinite(gradient)) or not math.isfinite(total):
            raise FloatingPointError("nonfinite mapped objective or analytic gradient")
        return float(total), gradient, components

    def map_diagnostics(self) -> dict[str, Any]:
        evaluations = self._objective_eval_count
        return {
            "map": self.map_name,
            "physical_domain": self.physical_domain,
            "objective_eval_count": int(evaluations),
            "nonidentity_map_eval_count": int(self._nonidentity_map_eval_count),
            "nonidentity_map_eval_fraction": (
                float(self._nonidentity_map_eval_count / evaluations)
                if evaluations else 0.0
            ),
            "nonidentity_bead_eval_count": int(self._nonidentity_bead_eval_count),
            "max_raw_radius_seen": float(self._max_raw_radius),
            "max_physical_radius_seen": float(self._max_physical_radius),
            "identity_radius": IDENTITY_RADIUS if self.map_name == "identity_interior_smooth_ball" else None,
            "counts_only_evaluate_calls": True,
            "line_search_probes_included": True,
        }


class IdentityInteriorJointObjective(_MappedJointObjective):
    """C2-map objective: same V1 terms with the smooth identity-interior map."""

    map_name = "identity_interior_smooth_ball"
    physical_domain = "strict_unit_ball"

    def _forward(self, raw: np.ndarray) -> np.ndarray:
        return identity_interior_forward(raw)

    def _inverse(self, coordinates: np.ndarray) -> np.ndarray:
        return identity_interior_inverse(coordinates)

    def _pullback(self, raw: np.ndarray, gradient_x: np.ndarray) -> np.ndarray:
        return identity_interior_pullback(raw, gradient_x)

    def _validate(self, coordinates: np.ndarray) -> None:
        validate_strict_unit_ball(coordinates)


class DirectPhysicalJointObjective(_MappedJointObjective):
    """C2-free objective: raw optimizer variables are finite physical x."""

    map_name = "direct_physical_unbounded"
    physical_domain = "finite_unbounded"

    def _forward(self, raw: np.ndarray) -> np.ndarray:
        return _as_xyz(raw, "physical coordinates").copy()

    def _inverse(self, coordinates: np.ndarray) -> np.ndarray:
        return _as_xyz(coordinates, "physical coordinates").copy()

    def _pullback(self, raw: np.ndarray, gradient_x: np.ndarray) -> np.ndarray:
        raw = _as_xyz(raw, "raw coordinates")
        gradient_x = _as_xyz(gradient_x, "gradient_x")
        if raw.shape != gradient_x.shape:
            raise ValueError("raw and gradient_x must have matching shapes")
        return gradient_x.copy()

    def _validate(self, coordinates: np.ndarray) -> None:
        validate_finite_coordinates(coordinates)


def objective_for_model(data: AggregatedContacts, model_id: str,
                        block_size: int = 65_536,
                        repulsion_block_size: int | None = None) -> JointObjective:
    """Construct one variant while leaving the frozen default class untouched."""
    spec = model_spec(model_id)
    model_data = data_for_model(data, model_id)
    if spec.coordinate_parameterization == "identity_interior_smooth_ball":
        return IdentityInteriorJointObjective(
            model_data, weights=spec.weights, block_size=block_size,
            repulsion_block_size=repulsion_block_size,
        )
    if spec.coordinate_parameterization == "direct_physical_unbounded":
        return DirectPhysicalJointObjective(
            model_data, weights=spec.weights, block_size=block_size,
            repulsion_block_size=repulsion_block_size,
        )
    return JointObjective(
        model_data, weights=spec.weights, block_size=block_size,
        repulsion_block_size=repulsion_block_size,
    )


def raw_coordinates_from_physical(objective: JointObjective,
                                   coordinates: np.ndarray) -> np.ndarray:
    """Convert a paired physical start into the objective's raw variables."""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (2, objective.data.n_loci, 3):
        raise ValueError("coordinates must have shape (2, n_loci, 3)")
    if hasattr(objective, "raw_from_physical"):
        return objective.raw_from_physical(values)  # type: ignore[attr-defined]
    assert_inside_unit_ball(values)
    return sphere_inverse(values)


def physical_coordinates_from_raw(objective: JointObjective,
                                   raw: np.ndarray) -> np.ndarray:
    """Convert one objective's raw variables to physical coordinates."""
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (2, objective.data.n_loci, 3):
        raise ValueError("raw coordinates must have shape (2, n_loci, 3)")
    if hasattr(objective, "_map_without_count"):
        coordinates = objective._map_without_count(values)  # type: ignore[attr-defined]
        objective.validate_physical_coordinates(coordinates)  # type: ignore[attr-defined]
        return coordinates
    coordinates = sphere_forward(values)
    assert_inside_unit_ball(coordinates)
    return coordinates


def validate_physical_for_model(model_id: str, coordinates: np.ndarray) -> None:
    """Apply the exact physical-domain validator for a registered variant."""
    values = np.asarray(coordinates, dtype=np.float64)
    spec = model_spec(model_id)
    if spec.physical_domain == "strict_unit_ball":
        assert_inside_unit_ball(values)
    else:
        validate_finite_coordinates(values)


__all__ = [
    "AlleleModelSpec",
    "DirectPhysicalJointObjective",
    "IDENTITY_RADIUS",
    "IDENTITY_WIDTH",
    "IdentityInteriorJointObjective",
    "MODEL_IDS",
    "P_INIT",
    "data_for_model",
    "identity_interior_forward",
    "identity_interior_inverse",
    "identity_interior_pullback",
    "model_spec",
    "objective_for_model",
    "physical_coordinates_from_raw",
    "raw_coordinates_from_physical",
    "validate_finite_coordinates",
    "validate_physical_for_model",
    "validate_strict_unit_ball",
]
