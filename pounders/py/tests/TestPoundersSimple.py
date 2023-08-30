"""
Unit test of compute function
"""

import unittest
from pathlib import Path

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
        assert np.all(P == T), "Test failed"

    def test_failing_objective(self):
        def failing_objective(x):
            fvec = x

            if np.random.uniform() < 0.1:
                fvec[0] = np.nan

            return fvec

        spsolver = 1
        nfmax = 1000
        gtol = 1e-13
        n = 3
        m = 3

        X0 = np.array([10, 20, 30])
        npmax = 2 * n + 1
        L = -np.inf * np.ones(n)
        U = np.inf * np.ones(n)
        nfs = 0
        F0 = []
        xkin = 0
        delta = 0.1
        printf = 0

        np.random.seed(1)

        [X, F, flag, xk_best] = pdrs.pounders(failing_objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver)
        assert flag == -3, "No NaN was encountered in this test, but should have been."

        F0 = np.array([1.0, 2.0])
        nfs = 2
        [X, F, flag, xk_best] = pdrs.pounders(failing_objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver)
        assert flag == -1, "We are testing proper failure of pounders"

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
        # func is a function imported from calFun.py as calFun
        func = vecFun
        # n [int] Dimension (number of continuous variables)
        n = 2
        # X0 [dbl] [min(fstart,1)-by-n] Set of initial points  (zeros(1,n))
        X0 = np.zeros((10, 2))
        X0[0, :] = 0.5 * np.ones((1, 2))
        npmax = int(0.5 * (n + 1) * (n + 2))
        # nfmax [int] Maximum number of function evaluations (>n+1) (100)
        nfmax = 60
        # gtol [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
        gtol = 10**-13
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
        # printf [log] 1 Indicates you want output to screen (1)
        printf = True
        spsolver = 2

        np.random.seed(1)
        F0[0, :] = func(X0[0, :])
        for i in range(1, 10):
            X0[i, :] = X0[0, :] + 0.2 * np.random.rand(1, 2) - 0.1
            F0[i, :] = func(X0[i, :])

        [X, F, flag, xkin] = pdrs.pounders(func, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf, spsolver)

    def test_pounders_one_output(self):
        combinemodels = pdrs.identity_combine

        # Sample calling syntax for pounders
        func = lambda x: np.sum(x)
        n = 16

        X0 = np.ones(n)
        # npmax [int] Maximum number of interpolation points (>n+1) (2*n+1)
        npmax = 2 * n + 1
        # nfmax   [int] Maximum number of function evaluations (>n+1) (100)
        nfmax = 200
        # gtol [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
        gtol = 10**-13
        # delta [dbl] Positive trust region radius (.1)
        delta = 0.1
        # nfs  [int] Number of function values (at X0) known in advance (0)
        nfs = 1
        # m [int] number of residuals
        m = 1
        # F0 [dbl] [fstart-by-1] Set of known function values  ([])
        F0 = func(X0)
        # xind [int] Index of point in X0 at which to start from (0)
        xind = 0
        # Low [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
        Low = -0.1 * np.arange(n)
        # Upp [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
        Upp = np.inf * np.ones(n)
        # printf = True to indicate you want output to screen
        printf = False
        # Choose your solver:
        spsolver = 1

        hfun = lambda F: F

        [X, F, flag, xkin] = pdrs.pounders(func, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf, spsolver, hfun, combinemodels)

        print("DEBUG32:", X, X[xkin], np.linalg.norm(X[xkin] - Low), flush=True)
        assert np.all(X[xkin] == Low), "The optimum should be the lower bounds."
