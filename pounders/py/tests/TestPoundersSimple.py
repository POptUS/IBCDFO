"""
Unit test of compute function
"""

import unittest

import ibcdfo.pounders as pdrs
import numpy as np


class TestPounders(unittest.TestCase):
    def test_phi2eval(self):
        D = np.eye(3)
        T = np.zeros((3, 6))
        T[0, 0] = 0.5
        T[1, 3] = 0.5
        T[2, 5] = 0.5

        P = pdrs.phi2eval(D)
        self.assertTrue(np.all(P == T), "Test failed")

    def test_failing_objective(self):
        def failing_objective(x):
            fvec = x

            if np.random.uniform() < 0.1:
                fvec[0] = np.nan

            return fvec

        spsolver = 1
        nfmax = 1000
        g_tol = 1e-13
        n = 3
        m = 3

        X0 = np.array([10, 20, 30])
        L = -np.inf * np.ones(n)
        U = np.inf * np.ones(n)
        delta = 0.1
        printf = 0

        np.random.seed(1)

        Opts = {"spsolver": spsolver, "printf": printf}
        [X, F, flag, xk_best] = pdrs.pounders(failing_objective, X0, n, nfmax, g_tol, delta, m, L, U, Options=Opts)
        self.assertEqual(flag, -3, "No NaN was encountered in this test, but should have been.")

        F0 = np.array([1.0, 2.0])
        Prior = {"X_init": X0, "F_init": F0, "nfs": 2, "xk_init": 0}
        Opts = {"spsolver": spsolver, "printf": printf}

        [X, F, flag, xk_best] = pdrs.pounders(failing_objective, X0, n, nfmax, g_tol, delta, m, L, U, Prior=Prior, Options=Opts)
        self.assertEqual(flag, -1, "We are testing proper failure of pounders")

    def test_basic_pounders_usage(self):
        def vecFun(x):
            """
            Input:
                x is a numpy array (column / row vector)
            Output:
                x + x^2 as a row vector
            """
            if np.shape(x)[0] > 1:
                x = np.reshape(x, (1, max(np.shape(x))))
            return x + (x**2)

        # Sample calling syntax for pounders
        Ffun = vecFun
        # n [int] Dimension (number of continuous variables)
        n = 2
        # X0 [dbl] [min(fstart,1)-by-n] Set of initial points  (zeros(1,n))
        X0 = np.zeros((10, 2))
        X0[0, :] = 0.5 * np.ones((1, 2))
        # nfmax [int] Maximum number of function evaluations (>n+1) (100)
        nfmax = 60
        # g_tol [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
        g_tol = 10**-13
        # delta [dbl] Positive trust region radius (.1)
        delta = 0.1
        # nfs [int] Number of function values (at X0) known in advance (0)
        nfs = 10
        # m [int] number of residuals
        m = 2
        # F0 [dbl] [fstart-by-1] Set of known function values  ([])
        F0 = np.zeros((10, 2))
        # xind [int] Index of point in X0 at which to start from (1)
        xind = 0
        # Low [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
        Low = np.zeros((1, n))
        # Upp [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
        Upp = np.ones((1, n))

        np.random.seed(1)
        F0[0, :] = Ffun(X0[0, :])
        for i in range(1, 10):
            X0[i, :] = X0[0, :] + 0.2 * np.random.rand(1, 2) - 0.1
            F0[i, :] = Ffun(X0[i, :])

        Prior = {"X_init": X0, "F_init": F0, "nfs": nfs, "xk_init": xind}
        [X, F, flag, xkin] = pdrs.pounders(Ffun, X0[xind], n, nfmax, g_tol, delta, m, Low, Upp, Model={"npmax": int(0.5 * (n + 1) * (n + 2))}, Prior=Prior)

    def test_pounders_one_output(self):
        combinemodels = pdrs.identity_combine

        # Sample calling syntax for pounders
        Ffun = lambda x: np.sum(x)
        n = 16

        X0 = np.ones(n)
        nfmax = 200
        g_tol = 10**-13
        delta = 0.1
        nfs = 1
        m = 1
        F0 = Ffun(X0)
        xind = 0
        Low = -0.1 * np.arange(n)
        Upp = np.inf * np.ones(n)

        hfun = lambda F: F
        Opts = {"spsolver": 1, "hfun": hfun, "combinemodels": combinemodels}
        Prior = {"X_init": X0, "F_init": F0, "nfs": nfs, "xk_init": xind}
        [X, F, flag, xkin] = pdrs.pounders(Ffun, X0, n, nfmax, g_tol, delta, m, Low, Upp, Options=Opts, Model={}, Prior=Prior)

        self.assertTrue(np.linalg.norm(X[xkin] - Low) <= 1e-8, "The optimum should be the lower bounds.")

    def test_pounders_maximizing_sum_squares(self):
        combinemodels = pdrs.neg_leastsquares

        # Sample calling syntax for pounders
        Ffun = lambda x: x
        n = 16

        X0 = 0.4 * np.ones((n, 1))  # Test giving of column vector
        nfmax = 200
        g_tol = 10**-13
        delta = 0.1
        m = n
        Low = 0.1 * np.ones(n)
        Upp = np.ones(n)

        hfun = lambda F: -1.0 * np.sum(F**2)

        Opts = {"spsolver": 1, "hfun": hfun, "combinemodels": combinemodels, "printf": 2}
        [X, F, flag, xkin] = pdrs.pounders(Ffun, X0, n, nfmax, g_tol, delta, m, Low, Upp, Options=Opts, Model={})

        self.assertTrue(np.linalg.norm(X[xkin] - Upp) <= 1e-8, "The optimum should be the upper bounds.")
