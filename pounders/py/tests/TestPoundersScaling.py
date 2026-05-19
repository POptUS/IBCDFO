"""
Unit test of simple functionality of pounders routine.
"""

import unittest

import ibcdfo
import numpy as np

import time


class TestPounders(unittest.TestCase):
    def test_basic_pounders_usage(self):

        # n [int] Dimension (number of continuous variables)
        n = 2
        X_0 = np.ones(n)

        # nf_max [int] Maximum number of function evaluations (>n+1) (100)
        nf_max = 60
        # g_tol [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
        g_tol = 10**-13
        # delta [dbl] Positive trust region radius (.1)
        delta = 0.1
        # Low [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
        Low = np.zeros((1, n))
        # Upp [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
        Upp = np.ones((1, n))

        # Loop over different numbers of residuals m
        for m in [10, 100, 1000, 10000, 100000]:
            with self.subTest(m=m):

                # Build a vecFun that outputs an m-dimensional residual for n-dimensional x
                # Here we use a linear model: F(x) = A x + b, with fixed random A, b per m
                rng = np.random.RandomState(123)  # deterministic for reproducibility
                A = rng.randn(m, n)
                b = rng.randn(m)

                def vecFun(x):
                    """
                    Input:
                        x is a numpy array (column / row vector) of length n
                    Output:
                        A @ x + b as a length-m 1D vector
                    """
                    x = np.atleast_1d(x).ravel()
                    return A.dot(x) + b

                Ffun = vecFun

                start_time = time.perf_counter()
                [X, F, hF, flag, xk_in_out] = ibcdfo.run_pounders(Ffun, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Model={"np_max": int(0.5 * (n + 1) * (n + 2))})
                elapsed = time.perf_counter() - start_time

                print(f"m = {m:5d}: runtime = {elapsed:.6f} seconds, flag = {flag}")
                print("\n")
                # Basic sanity check: we at least get an array of the right shape back
                # self.assertEqual(F.shape[0], m)
