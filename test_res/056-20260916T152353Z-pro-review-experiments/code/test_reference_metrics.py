import unittest

from evaluate_reference import metrics


class ReferenceMetricTests(unittest.TestCase):
    def test_unresolved_geometry_tie_retains_signed_copy_margins(self):
        result = metrics({"A_mat": 0.8, "A_pat": 0.2, "B_mat": 0.8, "B_pat": 0.2})
        self.assertEqual(result["orientation"], "unresolved_tie")
        self.assertAlmostEqual(result["direct"], 0.5)
        self.assertAlmostEqual(result["swapped"], 0.5)
        self.assertAlmostEqual(result["margin_A"], 0.6)
        self.assertAlmostEqual(result["margin_B"], -0.6)
        self.assertAlmostEqual(result["min_margin"], -0.6)


if __name__ == "__main__":
    unittest.main()
