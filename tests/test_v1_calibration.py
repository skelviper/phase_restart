import tempfile
import unittest

import numpy as np

from pr.contact_model import (JointObjective, aggregate_from_arrays, endpoint_counts_from_aggregates,
                              sphere_inverse, synthetic_expected_clone, synthetic_integer_clone)
from pr.joint_fit import fit_joint
from pr.v1_calibration import (coarsen_from_final, expected_counts, generate_truth,
                               generation_rates, load_layer, observed_endpoint_exposure,
                               sample_conditional_counts, save_layer, synthetic_exposure,
                               synthetic_r2, synthetic_r3)


def _template(bin_size=1, lengths=(24, 24)):
    empty = np.empty(0, dtype=np.int64)
    return aggregate_from_arrays(("chrA", "chrB"), lengths, empty, empty, empty, empty,
                                 bin_size)


def _integer_data(template):
    counts = np.zeros(template.n_pairs, dtype=np.int64)
    cis = np.flatnonzero(template.cis_pair)
    inter = np.flatnonzero(~template.cis_pair)
    counts[cis[:4]] = [2, 3, 1, 4]
    counts[inter[:4]] = [2, 1, 3, 2]
    diag = np.arange(1, template.n_loci + 1, dtype=np.int64)
    endpoints = endpoint_counts_from_aggregates(template, counts, diag)
    exposure = np.sqrt(endpoints + 10.0)
    exposure /= exposure.mean()
    return synthetic_integer_clone(template, counts, diag, exposure,
                                   exposure_mode="test_integer", endpoint_counts=endpoints)


class SyntheticCloneTests(unittest.TestCase):
    def test_integer_clone_exact_budget_and_endpoint_derivation(self):
        template = _template()
        data = _integer_data(template)
        audit = data.budget()
        self.assertEqual(audit["count_mode"], "synthetic_integer")
        self.assertEqual(audit["budget_unit"], "synthetic_integer_records")
        self.assertTrue(audit["raw_conserved"])
        self.assertTrue(audit["aggregate_conserved"])
        self.assertTrue(audit["endpoint_conserved"])
        self.assertTrue(np.issubdtype(data.counts.dtype, np.integer))
        np.testing.assert_array_equal(
            data.endpoint_counts,
            endpoint_counts_from_aggregates(template, data.counts, data.diag_counts),
        )

    def test_expected_and_integer_coarsening_preserve_total_mass(self):
        fine_template = _template(bin_size=1, lengths=(8, 8))
        integer = _integer_data(fine_template)
        coarse_template = _template(bin_size=2, lengths=(8, 8))
        coarse_integer = coarsen_from_final(integer, coarse_template, "production_observed_endpoint")
        self.assertEqual(coarse_integer.count_mode, "synthetic_integer")
        self.assertEqual(coarse_integer.raw_records, integer.raw_records)
        self.assertTrue(coarse_integer.budget()["endpoint_conserved"])
        self.assertLessEqual(coarse_integer.raw_cis_offdiag, integer.raw_cis_offdiag)

        counts = integer.counts.astype(float) + 0.25
        diag = integer.diag_counts.astype(float) + 0.125
        exposure = np.linspace(0.7, 1.3, fine_template.n_loci)
        exposure /= exposure.mean()
        expected = synthetic_expected_clone(
            fine_template, counts, diag, exposure,
            exposure_mode="test_expected",
        )
        coarse_expected = coarsen_from_final(expected, coarse_template, "known_synthetic")
        self.assertEqual(coarse_expected.count_mode, "synthetic_expected")
        self.assertAlmostEqual(coarse_expected.raw_records, expected.raw_records, places=10)
        self.assertIsNone(coarse_expected.budget()["endpoint_conserved"])
        self.assertTrue(np.issubdtype(coarse_expected.counts.dtype, np.floating))

    def test_layer_round_trip_preserves_count_mode_and_totals(self):
        data = _integer_data(_template(bin_size=1, lengths=(8, 8)))
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/layer.npz"
            save_layer(path, data)
            loaded = load_layer(path)
        self.assertEqual(loaded.count_mode, "synthetic_integer")
        self.assertEqual(loaded.budget()["raw_records"], data.budget()["raw_records"])
        np.testing.assert_array_equal(loaded.counts, data.counts)


class GeometryAndSamplingTests(unittest.TestCase):
    def test_same_shape_truth_is_bounded_and_r2_r3_tie_aware(self):
        template = _template(bin_size=1, lengths=(20, 20))
        truth, metadata = generate_truth(template, seed=4101, same_shape=True)
        self.assertTrue(metadata["same_internal_shape"])
        self.assertLess(float(np.linalg.norm(truth, axis=2).max()), 1.0)
        r2 = synthetic_r2(truth, truth, template)
        self.assertTrue(all(row["orientation"] == "tie" for row in r2["per_chromosome"]))
        r3 = synthetic_r3(truth, truth, template)
        self.assertTrue(all(not row["applicable"] for row in r3["per_chromosome"]))
        self.assertTrue(all(row["reason"] == "all_local_fragment_labels_tied_or_unavailable"
                            for row in r3["per_chromosome"]))

    def test_conditional_sampling_preserves_requested_group_totals(self):
        template = _template(bin_size=1, lengths=(8, 8))
        truth, _ = generate_truth(template, seed=4101, same_shape=False)
        exposure, _ = synthetic_exposure(template.n_loci, sigma=0.4, seed=4201)
        rates = generation_rates(template, truth, exposure, p=0.8, kernel="v1")
        totals = {"diag": 11, "cis_offdiag": 13, "inter": 17}
        expected_pair, expected_diag = expected_counts(template, rates, exposure, totals)
        self.assertAlmostEqual(float(expected_pair[template.cis_pair].sum()), 13.0, places=12)
        self.assertAlmostEqual(float(expected_pair[~template.cis_pair].sum()), 17.0, places=12)
        self.assertAlmostEqual(float(expected_diag.sum()), 11.0, places=12)
        counts, diag = sample_conditional_counts(template, rates, exposure, seed=6101, totals=totals)
        self.assertEqual(int(counts[template.cis_pair].sum()), 13)
        self.assertEqual(int(counts[~template.cis_pair].sum()), 17)
        self.assertEqual(int(diag.sum()), 11)
        observed, endpoints = observed_endpoint_exposure(template, counts, diag)
        self.assertAlmostEqual(float(observed.mean()), 1.0, places=14)
        self.assertEqual(int(endpoints.sum()), 2 * sum(totals.values()))

    def test_q_carry_and_budget_options_bypass_saturated_p_inverse(self):
        template = _template(bin_size=1, lengths=(4, 4))
        data = _integer_data(template)
        coords, _ = generate_truth(template, seed=5101, same_shape=False)
        objective = JointObjective(data, block_size=8, repulsion_block_size=8)
        result = fit_joint(
            objective,
            sphere_inverse(coords),
            p_init=0.75,
            q_init=1000.0,
            maxiter=1,
            maxfun=33,
            maxls=20,
        )
        self.assertTrue(np.isfinite(result.fun))
        self.assertGreaterEqual(result.actual_nfev, 1)
        self.assertTrue(np.isfinite(result.theta[-1]))


if __name__ == "__main__":
    unittest.main()
