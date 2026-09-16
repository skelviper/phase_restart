import math
import tempfile
import unittest

import numpy as np

from pr.contact_model import (JointObjective, ObjectiveWeights, aggregate_from_arrays,
                              assert_inside_unit_ball, bounded_kernel, p_from_q,
                              sphere_forward, sphere_pullback, synthetic_expected_clone,
                              tracks_from_coordinates, write_full_tracks)
from pr.joint_fit import fit_joint


BIN = 10


def _fixture_data():
    """两个三 bin 染色体，包含 diagonal、cis、inter 和 zero E pairs。"""
    records = [
        (0, 1, 0, 2),    # chrA bin 0 对角线
        (0, 3, 0, 4),    # chrA bin 0 对角线
        (0, 1, 0, 12),   # chrA 0-1
        (0, 12, 0, 21),  # chrA 1-2
        (0, 21, 0, 12),  # chrA 2-1，相同 aggregate pair
        (1, 1, 1, 12),   # chrB 0-1
        (1, 2, 1, 3),    # chrB bin 0 对角线
        (0, 3, 1, 2),    # inter 0-0
        (0, 12, 1, 12),  # inter 1-1
        (0, 21, 1, 21),  # inter 2-2
    ]
    ci, p1, cj, p2 = (np.array(part, dtype=np.int64) for part in zip(*records))
    return aggregate_from_arrays(("chrA", "chrB"), (30, 30), ci, p1, cj, p2, BIN)


def _fixture_y(data):
    rng = np.random.default_rng(20260913)
    return rng.normal(0.0, 0.22, size=(2, data.n_loci, 3)).astype(np.float64)


def _finite_difference(objective, theta, index, step=1e-6):
    plus = theta.copy()
    minus = theta.copy()
    plus[index] += step
    minus[index] -= step
    vp, _, _ = objective.evaluate(plus, need_gradient=False)
    vm, _, _ = objective.evaluate(minus, need_gradient=False)
    return (vp - vm) / (2.0 * step)


class AggregationTests(unittest.TestCase):
    def test_boundary_bins_full_grid_and_explicit_header_track_mapping(self):
        data = aggregate_from_arrays(
            ("chrA", "chrB"), (20, 21),
            np.array([0, 0, 1]), np.array([0, 19, 20]),
            np.array([0, 1, 1]), np.array([19, 20, 0]), BIN,
        )
        self.assertEqual(data.n_bins.tolist(), [2, 3])
        self.assertEqual(data.n_loci, 5)
        self.assertEqual(data.n_pairs, 10)
        self.assertEqual(int(data.global_locus(1, 2)), 4)
        self.assertEqual([spec.name for spec in data.track_specs],
                         ["c01a", "c01b", "c02a", "c02b"])
        self.assertEqual([spec.chromosome_name for spec in data.track_specs],
                         ["chrA", "chrA", "chrB", "chrB"])
        x = sphere_forward(np.full((2, data.n_loci, 3), 0.05))
        tracks = tracks_from_coordinates(data, x)
        self.assertEqual(list(tracks["c01a"]), [0, 10])
        self.assertEqual(list(tracks["c02b"]), [0, 10, 20])
        with tempfile.TemporaryDirectory() as directory:
            output = directory + "/full.3dg"
            write_full_tracks(output, data, x)
            with open(output, "rt") as handle:
                lines = handle.readlines()
            self.assertEqual(len(lines), 2 * data.n_loci)
            self.assertTrue(lines[0].startswith("c01a\t0\t"))
            self.assertTrue(lines[-1].startswith("c02b\t20\t"))
        with self.assertRaisesRegex(ValueError, "outside"):
            aggregate_from_arrays(
                ("chrA",), (20,), np.array([0]), np.array([20]),
                np.array([0]), np.array([0]), BIN,
            )

    def test_budget_exposure_and_same_bin_layer_are_conserved_and_finite(self):
        data = _fixture_data()
        audit = data.budget()
        self.assertTrue(audit["raw_conserved"])
        self.assertTrue(audit["aggregate_conserved"])
        self.assertTrue(audit["endpoint_conserved"])
        self.assertEqual(audit["raw_records"], 10)
        self.assertEqual(audit["raw_same_bin"], 3)
        self.assertEqual(audit["raw_cis_offdiag"], 4)
        self.assertEqual(audit["raw_inter"], 3)
        self.assertAlmostEqual(float(data.exposure.mean()), 1.0, places=14)
        objective = JointObjective(data, block_size=2, repulsion_block_size=2)
        theta = objective.pack(_fixture_y(data), p=0.73)
        first = objective.components(theta)
        moved = theta.copy()
        moved[:-1] *= 0.3
        second = objective.components(moved)
        self.assertTrue(math.isfinite(first["diag_profiled_nll_raw"]))
        positive_diag = data.diag_counts[data.diag_counts > 0].astype(np.float64)
        expected_diag = float(np.sum(positive_diag - positive_diag * np.log(positive_diag)
                                     + np.vectorize(math.lgamma)(positive_diag + 1.0)))
        self.assertAlmostEqual(first["diag_profiled_nll_raw"], expected_diag, places=14)
        self.assertAlmostEqual(first["diag_profiled_nll_raw"],
                               second["diag_profiled_nll_raw"], places=14)
        self.assertGreater(first["diag_profiled_nll_raw"], 0.0)

    def test_zero_count_pairs_enter_complete_normalization(self):
        data = aggregate_from_arrays(
            ("chrA",), (30,), np.array([0]), np.array([1]),
            np.array([0]), np.array([12]), BIN,
        )
        self.assertEqual(data.n_pairs, 3)
        self.assertEqual(int((data.counts == 0).sum()), 2)
        objective = JointObjective(
            data, weights=ObjectiveWeights(p_prior=0.0, bond=0.0, repulsion=0.0, bend=0.0),
            block_size=1,
        )
        theta = objective.pack(np.array([
            [[-0.12, 0.01, 0.02], [0.08, 0.11, -0.01], [0.21, -0.07, 0.02]],
            [[-0.08, -0.04, 0.01], [0.10, 0.06, 0.03], [0.18, 0.05, -0.06]],
        ]), p=0.7)
        x, p = objective.coordinates_and_p(theta)
        rates = objective.rates_for_pair_indices(x, p)
        observed = data.counts > 0
        components = objective.components(theta)
        self.assertGreater(float(rates.sum()), float(rates[observed].sum()))
        expected = math.log(float(rates.sum())) - math.log(float(rates[observed][0]))
        self.assertAlmostEqual(components["conditional_nll_raw"], expected, places=12)
        self.assertGreater(components["conditional_nll_raw"], 0.0)

    def test_synthetic_expected_clone_preserves_fractional_group_mass(self):
        template = _fixture_data()
        counts = np.zeros(template.n_pairs, dtype=np.float64)
        cis_weight = np.arange(1, int(template.cis_pair.sum()) + 1, dtype=np.float64)
        inter_weight = np.arange(1, int((~template.cis_pair).sum()) + 1, dtype=np.float64)
        counts[template.cis_pair] = 3.25 * cis_weight / cis_weight.sum()
        counts[~template.cis_pair] = 2.75 * inter_weight / inter_weight.sum()
        diag = np.array([0.10, 0.25, 0.15, 0.20, 0.05, 0.35], dtype=np.float64)
        exposure = np.linspace(0.8, 1.3, template.n_loci)
        exposure /= exposure.mean()
        totals = {"diag": 1.10, "cis_offdiag": 3.25, "inter": 2.75}
        synthetic = synthetic_expected_clone(
            template, counts, diag, exposure, totals,
            exposure_mode="synthetic_known_well_specified",
        )
        audit = synthetic.budget()
        self.assertEqual(audit["count_mode"], "synthetic_expected")
        self.assertEqual(audit["budget_unit"], "synthetic_expected_count_mass")
        self.assertAlmostEqual(audit["aggregate_same_bin"], 1.10, places=12)
        self.assertAlmostEqual(audit["aggregate_cis_offdiag"], 3.25, places=12)
        self.assertAlmostEqual(audit["aggregate_inter"], 2.75, places=12)
        self.assertIsNone(audit["conditional_count_factorial_constant_omitted"])
        self.assertIsNone(audit["endpoint_conserved"])
        objective = JointObjective(synthetic, block_size=2, repulsion_block_size=2)
        _, _, components = objective.evaluate(objective.pack(_fixture_y(synthetic), p=0.7))
        diag_positive = diag[diag > 0.0]
        expected_diag = float(np.sum(diag_positive - diag_positive * np.log(diag_positive)))
        self.assertAlmostEqual(components["diag_profiled_nll_raw"], expected_diag, places=12)
        self.assertEqual(components["count_factorial_status"],
                         "not_defined_for_fractional_expected_cross_entropy")
        with self.assertRaisesRegex(ValueError, "does not match"):
            synthetic_expected_clone(
                template, counts, diag, exposure,
                {"diag": 1.10, "cis_offdiag": 3.24, "inter": 2.75},
                exposure_mode="synthetic_known_well_specified",
            )


class ObjectiveMathTests(unittest.TestCase):
    def test_dense_and_block_scans_are_equivalent(self):
        data = _fixture_data()
        theta = JointObjective(data).pack(_fixture_y(data), p=0.71)
        dense = JointObjective(data, block_size=data.n_pairs + 10,
                               repulsion_block_size=data.n_pairs + 10)
        blocked = JointObjective(data, block_size=2, repulsion_block_size=3)
        v_dense, g_dense, c_dense = dense.evaluate(theta)
        v_block, g_block, c_block = blocked.evaluate(theta)
        self.assertAlmostEqual(v_dense, v_block, places=12)
        np.testing.assert_allclose(g_dense, g_block, rtol=1e-11, atol=1e-12)
        for key in ("conditional_nll_raw", "sum_rate_cis_offdiag", "sum_rate_inter",
                    "bond", "repulsion", "bend"):
            self.assertAlmostEqual(c_dense[key], c_block[key], places=12)

    def test_four_coordinate_components_match_finite_difference(self):
        data = _fixture_data()
        y = _fixture_y(data)
        selections = {
            "count": ObjectiveWeights(count=1.0, p_prior=0.0, bond=0.0, repulsion=0.0, bend=0.0),
            "bond": ObjectiveWeights(count=0.0, p_prior=0.0, bond=1.0, repulsion=0.0, bend=0.0),
            "repulsion": ObjectiveWeights(count=0.0, p_prior=0.0, bond=0.0, repulsion=1.0, bend=0.0),
            "bend": ObjectiveWeights(count=0.0, p_prior=0.0, bond=0.0, repulsion=0.0, bend=1.0),
        }
        for name, weights in selections.items():
            with self.subTest(component=name):
                objective = JointObjective(data, weights=weights, block_size=2,
                                           repulsion_block_size=2)
                theta = objective.pack(y, p=0.71)
                _, gradient, _ = objective.evaluate(theta)
                for index in (0, 7, 19):
                    numeric = _finite_difference(objective, theta, index)
                    self.assertAlmostEqual(gradient[index], numeric, delta=3e-5)

    def test_p_gradient_matches_finite_difference(self):
        data = _fixture_data()
        objective = JointObjective(
            data,
            weights=ObjectiveWeights(count=1.0, p_prior=1.0, bond=0.0, repulsion=0.0, bend=0.0),
            block_size=2,
        )
        theta = objective.pack(_fixture_y(data), p=0.71)
        _, gradient, _ = objective.evaluate(theta)
        numeric = _finite_difference(objective, theta, -1)
        self.assertAlmostEqual(gradient[-1], numeric, delta=3e-6)

    def test_sphere_transform_chain_rule_and_strict_ball(self):
        rng = np.random.default_rng(9)
        y = rng.normal(0.0, 0.4, size=(5, 3))
        gradient_x = rng.normal(0.0, 1.0, size=(5, 3))
        analytic = sphere_pullback(y, gradient_x)
        x = sphere_forward(y)
        assert_inside_unit_ball(x)
        for row, col in ((0, 0), (2, 1), (4, 2)):
            step = 1e-6
            plus = y.copy()
            minus = y.copy()
            plus[row, col] += step
            minus[row, col] -= step
            numeric = ((sphere_forward(plus) * gradient_x).sum()
                       - (sphere_forward(minus) * gradient_x).sum()) / (2.0 * step)
            self.assertAlmostEqual(analytic[row, col], numeric, delta=2e-7)

    def test_whole_chromosome_copy_swap_is_an_exact_gauge(self):
        data = _fixture_data()
        objective = JointObjective(data, block_size=2, repulsion_block_size=2)
        y = _fixture_y(data)
        theta = objective.pack(y, p=0.69)
        swapped_y = y.copy()
        slc = data.chromosome_slice(0)
        swapped_y[:, slc] = swapped_y[::-1, slc]
        swapped_theta = objective.pack(swapped_y, p=0.69)
        first, _, first_components = objective.evaluate(theta)
        second, _, second_components = objective.evaluate(swapped_theta)
        self.assertAlmostEqual(first, second, places=12)
        self.assertAlmostEqual(first_components["conditional_nll_raw"],
                               second_components["conditional_nll_raw"], places=12)

    def test_p_half_local_swap_preserves_data_but_can_change_chain_terms(self):
        data = _fixture_data()
        objective = JointObjective(data, block_size=2, repulsion_block_size=2)
        y = _fixture_y(data)
        theta = objective.pack(y, p=0.5)
        local = int(data.global_locus(0, 1))
        swapped_y = y.copy()
        swapped_y[:, local] = swapped_y[::-1, local]
        swapped_theta = objective.pack(swapped_y, p=0.5)
        first = objective.components(theta)
        second = objective.components(swapped_theta)
        self.assertAlmostEqual(first["conditional_nll_raw"], second["conditional_nll_raw"],
                               places=12)
        self.assertAlmostEqual(first["count_nll_normalized"], second["count_nll_normalized"],
                               places=12)
        chain_first = first["bond"] + objective.weights.bend * first["bend"]
        chain_second = second["bond"] + objective.weights.bend * second["bend"]
        self.assertGreater(abs(chain_first - chain_second), 1e-8)
        self.assertAlmostEqual(p_from_q(0.0)[0], 0.5, places=15)

    def test_collapsed_u_is_finite_without_residual_inversion(self):
        data = _fixture_data()
        y = _fixture_y(data)
        y[1] = y[0]
        objective = JointObjective(data, block_size=2, repulsion_block_size=2)
        value, gradient, components = objective.evaluate(objective.pack(y, p=0.75))
        self.assertTrue(math.isfinite(value))
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertTrue(math.isfinite(components["conditional_nll_raw"]))

    def test_inter_rate_uses_all_four_copy_combinations(self):
        data = _fixture_data()
        objective = JointObjective(data)
        pair_index = int(np.flatnonzero(~data.cis_pair)[0])
        i = int(data.pair_i[pair_index])
        j = int(data.pair_j[pair_index])
        x = np.zeros((2, data.n_loci, 3), dtype=np.float64)
        x[0, i] = [0.0, 0.0, 0.0]
        x[1, i] = [0.10, 0.0, 0.0]
        x[0, j] = [0.0, 0.20, 0.0]
        x[1, j] = [0.20, 0.30, 0.0]
        assert_inside_unit_ball(x)
        observed = objective.rates_for_pair_indices(x, 0.83, np.array([pair_index]))[0]
        manual = data.exposure[i] * data.exposure[j] / 4.0 * sum((
            bounded_kernel(x[0, i] - x[0, j], data.r0),
            bounded_kernel(x[0, i] - x[1, j], data.r0),
            bounded_kernel(x[1, i] - x[0, j], data.r0),
            bounded_kernel(x[1, i] - x[1, j], data.r0),
        ))
        self.assertAlmostEqual(observed, manual, places=14)

    def test_exact_value_gradient_cache_and_checkpoint_api(self):
        data = _fixture_data()
        objective = JointObjective(data, block_size=2, repulsion_block_size=2)
        theta = objective.pack(_fixture_y(data), p=0.72)
        value, _ = objective.value_and_grad(theta)
        cached = objective.cached_value_and_components(theta.copy())
        self.assertIsNotNone(cached)
        self.assertAlmostEqual(cached[0], value, places=14)
        self.assertIs(objective._last_theta, theta)
        changed = theta.copy()
        changed[0] = np.nextafter(changed[0], np.inf)
        self.assertIsNone(objective.cached_value_and_components(changed))

        checkpoints = []
        result = fit_joint(objective, _fixture_y(data), p_init=0.72, maxiter=4,
                           checkpoint_every=1, checkpoint_hook=checkpoints.append)
        self.assertEqual(result.nfev, result.actual_nfev)
        self.assertGreaterEqual(result.scipy_nfev, 1)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)
        self.assertTrue(all("nfev" in entry and "elapsed_seconds" in entry
                            for entry in result.history))
        self.assertGreaterEqual(len(checkpoints), 1)
        self.assertFalse(np.shares_memory(checkpoints[0].theta, result.theta))
        assert_inside_unit_ball(checkpoints[0].coordinates)

    def test_short_synthetic_optimization_decreases_and_stays_in_ball(self):
        data = _fixture_data()
        objective = JointObjective(data, block_size=2, repulsion_block_size=2)
        initial_y = _fixture_y(data)
        start, _, _ = objective.evaluate(objective.pack(initial_y, p=0.75), need_gradient=False)
        result = fit_joint(objective, initial_y, p_init=0.75, maxiter=25)
        self.assertLessEqual(result.fun, start + 1e-10)
        assert_inside_unit_ball(result.coordinates)
        self.assertGreaterEqual(len(result.history), 1)


if __name__ == "__main__":
    unittest.main()
