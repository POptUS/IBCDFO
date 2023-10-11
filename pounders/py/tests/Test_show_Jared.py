"""
Unit test of compute function
"""

import unittest

import ibcdfo.pounders as pdrs
import numpy as np


class TestPoundersSimple(unittest.TestCase):
    def test_failing_objective(self):
        def failing_objective(x):
            fvec = x

            if np.random.uniform() < 0.1:
                fvec[0] = np.nan

            return fvec

        n = 3
        X0 = np.array([10, 20, 30])
        L = -np.inf * np.ones(n)
        U = np.inf * np.ones(n)

        np.random.seed(1)

        Opts = {"spsolver": 1}
        [X, F, flag, xk_best] = pdrs.pounders(failing_objective, X0, 3, 100, 1e-13, 0.1, 3, L, U, Options=Opts)
        self.assertEqual(flag, -3, "No NaN was encountered in this test, but should have been.")

    def test_pounders_maximizing_sum_squares(self):
        combinemodels = pdrs.neg_leastsquares
        hfun = lambda F: -1.0 * np.sum(F**2)
        func = lambda x: x
        n = 16
        X0 = 0.4 * np.ones(n)
        m = n
        L = 0.1 * np.ones(n)
        U = np.ones(n)

        Opts = {"spsolver": 1, "hfun": hfun, "combinemodels": combinemodels}
        [X, F, flag, xkin] = pdrs.pounders(func, X0, n, 100, 1e-13, 0.1, m, L, U, Options=Opts)

        self.assertTrue(np.linalg.norm(X[xkin] - U) <= 1e-8, "The optimum should be the upper bounds.")
