"""
End-to-end tests of goombah() using minimize_quadratics_lhs as the TRSP
solver, ported from test_goombah_converge.m, test_goombah_failures.m, and
test_goombah_budget_sweep.m.
"""

import functools
import unittest

import numpy as np
from ibcdfo.goombah import goombah, minimize_quadratics_lhs
from ibcdfo.manifold_sampling import h_pw_maximum_squared
from ibcdfo.manifold_sampling.tests.general_smooth_h_funs import h_leastsquares

# BenDFO
from calfun import calfun
from dfoxs import dfoxs


def _bendfo_problem(row):
    # row is a 1-indexed row of dfo.dat, matching the MATLAB tests.
    DFO = np.loadtxt("dfo.dat")
    nprob, n, m, factor_power = DFO[row - 1, :]
    nprob, n, m = int(nprob), int(n), int(m)
    x0 = dfoxs(n, nprob, 10**factor_power)

    def Ffun(y):
        out = calfun(y, m, nprob, "smooth", np.nan, num_outs=2)[1]
        assert len(out) == m
        return np.squeeze(out)

    return Ffun, x0, n, m


def _lhs_trsp(seed):
    return functools.partial(minimize_quadratics_lhs, rng=np.random.default_rng(seed))


class TestGoombahLhsConverge(unittest.TestCase):
    def test_converges_to_known_minimum(self):
        # Row 1 of the More and Wild SIOPT benchmark is linear in x, so the
        # component models are exact, and h(F(x)) = sum(F(x)**2) has the
        # known minimum m - n = 36.
        Ffun, x0, n, m = _bendfo_problem(1)
        L, U = -np.inf * np.ones(n), np.inf * np.ones(n)
        nf_max = 400
        known_min = m - n

        X, F, h, xkin, flag = goombah(h_leastsquares, Ffun, nf_max, x0, L, U, _lhs_trsp(1), subprob_switch="linprog")

        self.assertLess(X.shape[0], nf_max, "goombah used its entire evaluation budget")
        self.assertLessEqual(abs(np.min(h) - known_min), 1e-6, "goombah did not converge to the known minimum")
        self.assertIn(flag, (0, -6), "flag should be 0 or -6 when the budget is not exhausted")

    def test_converges_to_known_minimum_rosenbrock(self):
        # Row 7 of the More and Wild SIOPT benchmark is the 2D Rosenbrock
        # function, whose unique minimum is 0 at x* = (1, 1), where both
        # residuals are exactly 0. This holds for both the sum-of-squares and
        # pointwise-maximum-of-squares h, unlike row 1's known minimum above.
        Ffun, x0, n, m = _bendfo_problem(7)
        L, U = -np.inf * np.ones(n), np.inf * np.ones(n)
        nf_max = 200
        known_min = 0.0

        for h_type, hfun in {"sum_squared": h_leastsquares, "pw_maximum_squared": h_pw_maximum_squared}.items():
            with self.subTest(h_type=h_type):
                X, F, h, xkin, flag = goombah(hfun, Ffun, nf_max, x0, L, U, _lhs_trsp(1), subprob_switch="linprog")

                self.assertLess(X.shape[0], nf_max, "goombah used its entire evaluation budget")
                self.assertLessEqual(abs(np.min(h) - known_min), 1e-6, "goombah did not converge to the known minimum")
                self.assertIn(flag, (0, -6), "flag should be 0 or -6 when the budget is not exhausted")


class TestGoombahLhsBudgetSweep(unittest.TestCase):
    def test_every_budget_in_a_small_range(self):
        # Runs goombah, with the LHS TRSP solver, on a tiny problem for every
        # evaluation budget nf_max, in steps of one, from its smallest
        # allowed value up through n + MAX_EXTRA, checking basic sanity
        # properties of the returned values at each budget.
        Ffun, x0, n, m = _bendfo_problem(7)
        L, U = -5000 * np.ones(n), 5000 * np.ones(n)
        MAX_EXTRA = 20

        for nf_max in range(n + 1, n + MAX_EXTRA + 1):
            with self.subTest(nf_max=nf_max):
                X, F, h, xkin, flag = goombah(h_pw_maximum_squared, Ffun, nf_max, x0, L, U, _lhs_trsp(1), subprob_switch="linprog")

                nf = X.shape[0]
                self.assertLessEqual(nf, nf_max, "Exceeded the evaluation budget")
                self.assertEqual(F.shape[0], nf)
                self.assertEqual(h.shape[0], nf)
                for j in range(nf):
                    expected_h, _, _ = h_pw_maximum_squared(F[j])
                    self.assertLessEqual(abs(h[j] - expected_h), 1e-12 * max(1, abs(h[j])), f"h is not hfun(F) at evaluation {j}")
                self.assertLessEqual(np.min(h), h[0], "Ended worse than the starting point")


class TestGoombahLhsFailures(unittest.TestCase):
    def setUp(self):
        self.n = 3
        self.x0 = np.array([10.0, 20.0, 30.0])
        self.Low = -np.inf * np.ones(self.n)
        self.Upp = np.inf * np.ones(self.n)
        self.hfun = h_pw_maximum_squared
        self.Ffun = lambda x: x
        self.trsp = _lhs_trsp(1)

    def test_failing_Ffun(self):
        # Passing a non-callable in place of Ffun should produce the
        # model-building/Ffun-failure flag.
        X, F, h, xkin, flag = goombah(self.hfun, self.x0, 100, self.x0, self.Low, self.Upp, self.trsp)
        self.assertEqual(flag, -1, "flag should be -1 when Ffun fails")

    def test_small_budget(self):
        # Only enough evaluation budget for the initial coordinate steps and
        # one more.
        nf_max_small = self.n + 2
        X, F, h, xkin, flag = goombah(self.hfun, self.Ffun, nf_max_small, self.x0, self.Low, self.Upp, self.trsp)
        self.assertLessEqual(X.shape[0], nf_max_small, "goombah grew the size of X")

    def test_fixed_coordinate(self):
        # Low == Upp in one coordinate is rejected by checkinputss (the same
        # check used by pounders and manifold_sampling_primal, and now also
        # by goombah.m). MATLAB's checkinputss reports this by returning
        # flag = -1; Python's raises ValueError instead, its established
        # exception-based idiom for the same check.
        Low_fixed, Upp_fixed = self.Low.copy(), self.Upp.copy()
        Low_fixed[0] = self.x0[0]
        Upp_fixed[0] = self.x0[0]

        with self.assertRaises(ValueError):
            goombah(self.hfun, self.Ffun, 100, self.x0, Low_fixed, Upp_fixed, self.trsp)

    def test_budget_exhausted(self):
        X, F, h, xkin, flag = goombah(self.hfun, self.Ffun, self.n + 5, self.x0, self.Low, self.Upp, self.trsp)
        self.assertEqual(flag, 1, "flag should be 1 when the budget is exhausted")
        self.assertEqual(X.shape[0], self.n + 5)
        self.assertEqual(F.shape[0], self.n + 5)
        self.assertEqual(h.shape[0], self.n + 5)


if __name__ == "__main__":
    unittest.main()
