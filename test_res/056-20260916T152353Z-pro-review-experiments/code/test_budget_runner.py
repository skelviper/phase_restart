import unittest

import numpy as np

from budget_runner import run_budgeted_lbfgs_no_extra


class Data:
    n_loci = 1


class Quadratic:
    def __init__(self):
        self.data = Data()
        self.calls = 0
        self.cache = None

    def pack(self, y, p=0.75):
        return np.concatenate((np.asarray(y).reshape(-1), np.asarray([p])))

    def unpack(self, theta):
        return np.asarray(theta[:-1]).reshape(2, 1, 3), float(theta[-1])

    def coordinates_and_p(self, theta):
        return self.unpack(theta)

    def value_and_grad(self, theta):
        self.calls += 1
        theta = np.asarray(theta)
        value = float(np.dot(theta, theta) / 2.0)
        gradient = theta.copy()
        self.cache = (theta.copy(), value, {"p": float(theta[-1]), "total": value})
        return value, gradient

    def cached_value_and_components(self, theta):
        if self.cache is None or not np.array_equal(np.asarray(theta), self.cache[0]):
            return None
        return self.cache[1], dict(self.cache[2])


class BudgetRunnerTests(unittest.TestCase):
    def test_no_extra_objective_calls_and_finite_accepted_endpoint(self):
        objective = Quadratic()
        y = np.full((2, 1, 3), 0.2)
        result = run_budgeted_lbfgs_no_extra(
            objective, y, p_init=0.75, q_init=0.75, maxfun=5, maxiter=6,
            maxls=20, ftol=0.0, canonical_gtol=0.0)
        self.assertEqual(objective.calls, result.nfev)
        self.assertLessEqual(result.nfev, 5)
        self.assertEqual(result.validation_calls, 0)
        self.assertTrue(np.isfinite(result.theta).all())
        self.assertEqual(result.history[0]["nfev"], 1)
        self.assertEqual(result.history[0]["state_status"], "accepted_initial_budgeted")


if __name__ == "__main__":
    unittest.main()
