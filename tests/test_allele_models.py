import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pr.allele_models import (
    IDENTITY_RADIUS,
    MODEL_IDS,
    data_for_model,
    identity_interior_forward,
    identity_interior_inverse,
    identity_interior_pullback,
    model_spec,
    objective_for_model,
    raw_coordinates_from_physical,
)
from pr.contact_model import (
    JointObjective,
    ObjectiveWeights,
    aggregate_from_arrays,
    assert_inside_unit_ball,
    sphere_forward,
)


TEST_SEED = 20260913


def _data(bin_size=10):
    records = [
        (0, 0, 0, 10),       # cis 非对角
        (0, 10, 0, 20),      # cis 非对角
        (0, 20, 0, 10),      # 重复的 aggregate pair
        (1, 0, 1, 10),       # cis 非对角
        (0, 0, 1, 0),        # inter
        (0, 10, 1, 10),      # inter
        (0, 20, 1, 20),      # inter
        (0, 0, 0, 0),        # 对角 nuisance
    ]
    lengths = (30, 30)
    ci, p1, cj, p2 = (np.asarray(part, dtype=np.int64) for part in zip(*records))
    return aggregate_from_arrays(("chrA", "chrB"), lengths, ci, p1, cj, p2, bin_size)


def _one_mb_data():
    records = [
        (0, 0, 0, 1_000_000),
        (0, 1_000_000, 0, 2_000_000),
        (1, 0, 1, 1_000_000),
        (0, 0, 1, 0),
        (0, 1_000_000, 1, 1_000_000),
        (0, 2_000_000, 1, 2_000_000),
        (0, 0, 0, 0),
    ]
    ci, p1, cj, p2 = (np.asarray(part, dtype=np.int64) for part in zip(*records))
    return aggregate_from_arrays(
        ("chrA", "chrB"), (3_000_000, 3_000_000),
        ci, p1, cj, p2, 1_000_000,
    )


class MapperTests(unittest.TestCase):
    def test_inverse_forward_covers_interior_join_and_far_outer_radius(self):
        directions = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 3.0],
            [-2.0, 1.0, 1.0],
            [1.0, -1.0, 2.0],
        ], dtype=np.float64)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = np.asarray([0.0, 0.4, IDENTITY_RADIUS, 0.900001, 1.5, 3.0])
        raw = np.zeros((len(radii), 3), dtype=np.float64)
        raw[1:] = directions[:len(radii) - 1] * radii[1:, None]
        physical = identity_interior_forward(raw)
        self.assertTrue(np.all(np.linalg.norm(physical, axis=1) < 1.0))
        np.testing.assert_allclose(identity_interior_inverse(physical), raw,
                                   rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(physical[2], raw[2], rtol=0.0, atol=0.0)
        near_boundary = identity_interior_inverse(
            np.asarray([[0.999, 0.0, 0.0]], dtype=np.float64)
        )
        self.assertTrue(np.all(np.isfinite(near_boundary)))
        np.testing.assert_allclose(identity_interior_forward(near_boundary),
                                   [[0.999, 0.0, 0.0]], rtol=2e-12, atol=2e-12)

    def test_pullback_matches_directional_finite_difference_across_join(self):
        gradient = np.asarray([0.7, -1.1, 0.35], dtype=np.float64)
        direction = np.asarray([0.4, -0.6, 0.7], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        step = 1e-7
        for radius in (0.3, IDENTITY_RADIUS, 0.900001, 1.25, 3.0):
            y = np.asarray([radius, 0.0, 0.0], dtype=np.float64)
            analytic = float(np.sum(identity_interior_pullback(y, gradient) * direction))
            plus = identity_interior_forward(y + step * direction)
            minus = identity_interior_forward(y - step * direction)
            numeric = float(np.sum((plus - minus) * gradient) / (2.0 * step))
            self.assertAlmostEqual(analytic, numeric, delta=3e-6,
                                   msg="radius=%s" % radius)

    def test_map_and_free_objectives_have_correct_directional_gradients_without_fit(self):
        data = _data()
        rng = np.random.default_rng(TEST_SEED)
        raw = rng.normal(0.0, 0.18, size=(2, data.n_loci, 3))
        raw[0, 0] = [1.05, 0.0, 0.0]
        direction = rng.normal(size=raw.size + 1)
        direction /= np.linalg.norm(direction)
        for model_id in ("C2-map", "C2-free"):
            with self.subTest(model=model_id):
                objective = objective_for_model(data, model_id, block_size=3,
                                                repulsion_block_size=3)
                theta = objective.pack(raw, p=0.71)
                value, gradient, _ = objective.evaluate(theta)
                self.assertTrue(math.isfinite(value))
                self.assertTrue(np.all(np.isfinite(gradient)))
                step = 1e-6
                plus = theta + step * direction
                minus = theta - step * direction
                value_plus = objective.evaluate(plus, need_gradient=False)[0]
                value_minus = objective.evaluate(minus, need_gradient=False)[0]
                numeric = (value_plus - value_minus) / (2.0 * step)
                self.assertAlmostEqual(float(np.dot(gradient, direction)), numeric,
                                       delta=2e-4)


class VariantTests(unittest.TestCase):
    def test_registered_variants_have_exact_single_changes(self):
        self.assertEqual(MODEL_IDS, ("C0", "C1", "C2-map", "C2-free", "C3"))
        self.assertAlmostEqual(model_spec("C0").weights.bend, 0.01)
        self.assertAlmostEqual(model_spec("C1").weights.bend, 0.0)
        self.assertEqual(model_spec("C2-map").physical_domain, "strict_unit_ball")
        self.assertEqual(model_spec("C2-free").physical_domain, "finite_unbounded")
        self.assertEqual(model_spec("C3").exposure_change, "uniform_full_grid")
        self.assertEqual(model_spec("C0").as_dict()["repulsion_threshold"], "0.7*l0")

    def test_c0_factory_is_numerically_identical_to_original_objective(self):
        data = _data()
        rng = np.random.default_rng(TEST_SEED + 1)
        raw = rng.normal(0.0, 0.12, size=(2, data.n_loci, 3))
        weights = ObjectiveWeights()
        original = JointObjective(data, weights=weights, block_size=3,
                                  repulsion_block_size=3)
        candidate = objective_for_model(data, "C0", block_size=3,
                                       repulsion_block_size=3)
        theta = original.pack(raw, p=0.73)
        first = original.evaluate(theta)
        second = candidate.evaluate(theta)
        self.assertIs(type(candidate), JointObjective)
        self.assertAlmostEqual(first[0], second[0], places=13)
        np.testing.assert_allclose(first[1], second[1], rtol=1e-12, atol=1e-13)
        for key in first[2]:
            self.assertAlmostEqual(first[2][key], second[2][key], places=13,
                                   msg=key)

    def test_c3_changes_only_exposure_and_keeps_full_grid_and_diagonal_layer(self):
        data = _data()
        c3 = data_for_model(data, "C3")
        np.testing.assert_allclose(c3.exposure, np.ones(data.n_loci), rtol=0.0, atol=0.0)
        self.assertEqual(c3.exposure_mode, "uniform")
        self.assertEqual(c3.n_pairs, data.n_pairs)
        np.testing.assert_array_equal(c3.pair_i, data.pair_i)
        np.testing.assert_array_equal(c3.pair_j, data.pair_j)
        np.testing.assert_array_equal(c3.cis_pair, data.cis_pair)
        np.testing.assert_array_equal(c3.counts, data.counts)
        np.testing.assert_array_equal(c3.diag_counts, data.diag_counts)
        np.testing.assert_array_equal(c3.endpoint_counts, data.endpoint_counts)
        self.assertEqual(c3.budget()["n_zero_eligible_pairs"], data.budget()["n_zero_eligible_pairs"])
        self.assertEqual(c3.budget()["diag_nuisance_parameters"], data.budget()["diag_nuisance_parameters"])
        c3.assert_consistent()

    def test_c2_map_counts_all_evaluate_calls_and_free_accepts_outside_ball(self):
        data = _data()
        raw = np.zeros((2, data.n_loci, 3), dtype=np.float64)
        raw[0, 0] = [1.1, 0.0, 0.0]
        mapped = objective_for_model(data, "C2-map", block_size=3,
                                     repulsion_block_size=3)
        theta = mapped.pack(raw, p=0.75)
        mapped.evaluate(theta, need_gradient=False)
        mapped.evaluate(theta, need_gradient=True)
        diagnostics = mapped.map_diagnostics()
        self.assertEqual(diagnostics["objective_eval_count"], 2)
        self.assertEqual(diagnostics["nonidentity_map_eval_count"], 2)
        self.assertEqual(diagnostics["nonidentity_bead_eval_count"], 2)
        mapped.coordinates_and_p(theta)
        self.assertEqual(mapped.map_diagnostics()["objective_eval_count"], 2)

        outside = raw.copy()
        outside[0, 0] = [1.2, 0.0, 0.0]
        free = objective_for_model(data, "C2-free", block_size=3,
                                   repulsion_block_size=3)
        free_theta = free.pack(outside, p=0.75)
        coordinates, _ = free.coordinates_and_p(free_theta)
        np.testing.assert_array_equal(coordinates, outside)
        free.evaluate(free_theta, need_gradient=False)
        with self.assertRaises(AssertionError):
            raw_coordinates_from_physical(objective_for_model(data, "C0"), outside)


if __name__ == "__main__":
    unittest.main()
