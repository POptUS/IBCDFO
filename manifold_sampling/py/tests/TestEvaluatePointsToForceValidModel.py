"""
Check that evaluate_points_to_force_valid_model never requests a point that
violates the bound constraints (i.e., that no clipping/projection is needed
downstream in call_user_scripts).
"""

import sys
import unittest
from unittest import mock

import ibcdfo.manifold_sampling  # noqa: F401  (ensures submodule is imported)
import numpy as np

# The package attribute of the same name is the function, so go through sys.modules
evalmod = sys.modules["ibcdfo.manifold_sampling.evaluate_points_to_force_valid_model"]


class TestEvaluatePointsToForceValidModel(unittest.TestCase):
    def _run(self, xk, L, U, delta):
        n = len(xk)
        p = n
        nf_max = 50
        fq_pars = {"Par": [np.sqrt(n), max(10, np.sqrt(n)), 0.001, 0.001, False], "npmax": 2 * n + 1}
        tol = {"hfun_test_mode": 0}

        X = np.zeros((nf_max, n))
        X[0] = xk
        F = np.zeros((nf_max, p))
        F[0] = xk
        h = np.zeros((nf_max, 1))
        Hash = {}
        Mdir = np.eye(n)  # no model points yet: all n directions are needed

        requested = []

        def spy(nf, X, F, h, Hash, Ffun, hfun, x_in, tol, L, U, allow_recalc=0):
            requested.append(np.array(x_in, dtype=float))
            nf += 1
            X[nf] = x_in
            F[nf] = x_in
            return nf, X, F, h, Hash, None

        with mock.patch.object(evalmod, "call_user_scripts", side_effect=spy):
            evalmod.evaluate_points_to_force_valid_model(n, 0, 0, delta, X, F, h, Mdir, 0, None, None, Hash, fq_pars, tol, nf_max, L, U)
        return requested

    def _assert_feasible(self, requested, L, U):
        self.assertGreater(len(requested), 0)
        for x in requested:
            self.assertTrue(np.all(x >= L - 1.0e-15), f"{x} violates lower bound {L}")
            self.assertTrue(np.all(x <= U + 1.0e-15), f"{x} violates upper bound {U}")

    def test_incumbent_at_lower_bound(self):
        # Positive step is feasible, negative is not
        L = np.zeros(3)
        U = np.ones(3)
        self._assert_feasible(self._run(L.copy(), L, U, 0.1), L, U)

    def test_incumbent_at_upper_bound(self):
        # Negative step is feasible, positive is not: a "always +" rule would fail here
        L = np.zeros(3)
        U = np.ones(3)
        req = self._run(U.copy(), L, U, 0.1)
        self._assert_feasible(req, L, U)
        for x in req:
            self.assertTrue(np.any(x < U))

    def test_incumbent_at_mixed_corner(self):
        L = -np.ones(3)
        U = np.ones(3)
        xk = np.array([-1.0, 1.0, 0.0])
        self._assert_feasible(self._run(xk, L, U, 0.25), L, U)

    def test_narrow_box_forces_coordinate_fallback(self):
        # Box narrower than delta in one coordinate
        L = np.array([0.0, 0.0, 0.0])
        U = np.array([1.0, 0.01, 1.0])
        xk = np.array([1.0, 0.0, 0.5])
        self._assert_feasible(self._run(xk, L, U, 0.5), L, U)

    def test_infinite_bounds(self):
        L = -np.inf * np.ones(3)
        U = np.inf * np.ones(3)
        self._assert_feasible(self._run(np.zeros(3), L, U, 0.1), L, U)


if __name__ == "__main__":
    unittest.main()
